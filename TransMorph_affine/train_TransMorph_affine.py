"""Training of TransMorph-Affine on the OrCaScore NIfTI dataset (moving CTAI ->
fixed CTI).

- Training: AdamW, MSE between the warped CTAI and the CTI at full
  resolution, synthetic misalignment with affine_aug on the moving image.
- Validation: real pairs (no augmentation), measured without the padding; the
  best validation SSIM defines best.pth.tar.
"""

# isort: skip_file
# (the repository root must be on sys.path before importing orcascore)
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from orcascore.affine_model import (  # noqa: E402
    affine_config_for_checkpoint,
    build_affine,
    default_affine_config,
    predict_affine,
)
from orcascore.dataset import (  # noqa: E402
    PairedCTDataset,
    padding_to_dict,
    read_split_ids,
)
from orcascore.io_utils import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_DIR,
    append_history,
    load_checkpoint,
    now,
    prepare_experiment,
    resolve_checkpoint,
    save_checkpoint,
    save_experiment_info,
    setup_device,
    start_logging,
)
from orcascore.metrics import validation_metrics  # noqa: E402
from orcascore.plots import (  # noqa: E402
    difference_image,
    read_spacing_zyx,
    save_comparison_figure,
)

# affine_aug ranges
ANGLE_RANGE_DEGREES = 10
TRANSLATION_RANGE = 0.1
SCALE_RANGE = 0.1


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Train TransMorph-Affine")
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="dataset folder (dataset.csv, config.json)",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="experiment name (folder inside --output_dir)",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="where experiments are saved",
    )
    parser.add_argument(
        "--train_split",
        default=os.path.join(DEFAULT_SPLIT_DIR, "train.csv"),
        help="CSV with the training case_ids (column ID)",
    )
    parser.add_argument(
        "--val_split",
        default=os.path.join(DEFAULT_SPLIT_DIR, "test.csv"),
        help="CSV with the validation case_ids (column ID)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=500, help="total number of epochs"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="keep a copy of the checkpoint every N epochs "
        "(last/best are always kept)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        default=None,
        help="resume from a checkpoint of this experiment (default: last); "
        "'last', 'best', 'epc_<n>' or a path to a .pth.tar file",
    )
    parser.add_argument(
        "--no_flip",
        action="store_true",
        help="disable the random flip augmentation",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="optional seed for torch/numpy"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="device"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index (cuda:N)"
    )
    return parser.parse_args()


def rotation_matrix(axis: int, angle: float) -> torch.Tensor:
    """4x4 homogeneous rotation matrix around one axis of the normalized grid.

    Args:
        axis: 0 = x, 1 = y, 2 = z.
        angle: rotation angle in radians.

    Returns:
        Rotation matrix of shape (4, 4).
    """
    cos, sin = math.cos(angle), math.sin(angle)
    matrix = torch.eye(4)
    first_axis, second_axis = [index for index in range(3) if index != axis]
    matrix[first_axis, first_axis] = cos
    matrix[second_axis, second_axis] = cos
    # same signs as the authors' affine_aug: -sin above the diagonal for x
    # and z, below for y
    if axis == 1:
        matrix[first_axis, second_axis] = sin
        matrix[second_axis, first_axis] = -sin
    else:
        matrix[first_axis, second_axis] = -sin
        matrix[second_axis, first_axis] = sin
    return matrix


def random_affine_matrix() -> torch.Tensor:
    """Random 4x4 affine matrix used by affine_aug.

    Composition R_x @ R_y @ R_z @ (scale + translation), with values drawn
    uniformly per axis:
        - rotation: ±ANGLE_RANGE_DEGREES;
        - scale: 1 ± SCALE_RANGE;
        - translation: ±TRANSLATION_RANGE (normalized coordinates, where the
          volume spans [-1, 1]).

    Values come from the `random` module in the order of the authors'
    affine_aug (angles, scales, translations).
    """
    angles = [
        random.uniform(-ANGLE_RANGE_DEGREES, ANGLE_RANGE_DEGREES)
        * math.pi
        / 180
        for _ in range(3)
    ]
    scales = [random.uniform(-SCALE_RANGE, SCALE_RANGE) for _ in range(3)]
    translations = [
        random.uniform(-TRANSLATION_RANGE, TRANSLATION_RANGE) for _ in range(3)
    ]
    scale_translation = torch.eye(4)
    for axis in range(3):
        scale_translation[axis, axis] = 1.0 + scales[axis]
        scale_translation[axis, 3] = translations[axis]
    matrix = rotation_matrix(axis=0, angle=angles[0]) @ rotation_matrix(
        axis=1, angle=angles[1]
    )
    matrix = matrix @ rotation_matrix(axis=2, angle=angles[2])
    return matrix @ scale_translation


def affine_aug(images: torch.Tensor, seed: int) -> torch.Tensor:
    """Applies a random affine (random_affine_matrix) to each sample of the
    batch, with bilinear interpolation and border padding.

    Args:
        images: (B, C, D, H, W) tensor.
        seed: seed of the `random` module; the training loop passes the
            iteration index, as in the authors' code.

    Returns:
        (B, C, D, H, W) tensor with the augmented images.
    """
    with torch.no_grad():
        random.seed(seed)
        theta = torch.stack(
            [random_affine_matrix() for _ in range(images.shape[0])]
        )
        theta = theta[:, :3, :].to(device=images.device, dtype=images.dtype)
        grid = F.affine_grid(theta, list(images.shape), align_corners=False)
        return F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


def train_one_epoch(
    model, transform, loader, optimizer, criterion, device, epoch: int
) -> float:
    """One training epoch; returns the mean loss."""
    model.train()
    losses = []
    for index, batch in enumerate(loader, start=1):
        moving = batch["x"].to(device)
        fixed = batch["y"].to(device)
        moving_augmented = affine_aug(moving, seed=index)
        output = predict_affine(
            model, transform, moving=moving_augmented, fixed=fixed
        )
        loss = criterion(output["warped"], fixed)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(
            f"Epoch {epoch} iter {index}/{len(loader)} loss {loss.item():.6f}"
        )
    return float(np.mean(losses))


def validate(
    model, transform, loader, device, figure_path: str = None
) -> dict:
    """Validation on the real pairs: SSIM per slice and MSE without padding,
    before and after the affine.

    Args:
        figure_path: if given, saves a figure of the first pair.

    Returns:
        dict with the mean "val_ssim_pre", "val_ssim", "val_mse_pre" and
        "val_mse" over the validation pairs.
    """
    model.eval()
    pre, post = [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            moving = batch["x"].to(device)
            fixed = batch["y"].to(device)
            padding = padding_to_dict(batch["padding"][0])
            output = predict_affine(
                model, transform, moving=moving, fixed=fixed
            )
            pre.append(validation_metrics(fixed, moving, padding=padding))
            post.append(
                validation_metrics(fixed, output["warped"], padding=padding)
            )
            if figure_path and index == 0:
                save_comparison_figure(
                    path=figure_path,
                    images={
                        "ACTC (moving)": moving,
                        "TC-ECC (fixed)": fixed,
                        "ACTC affine": output["warped"],
                        "|TC-ECC - ACTC|": difference_image(fixed, moving),
                        "|TC-ECC - ACTC affine|": difference_image(
                            fixed, output["warped"]
                        ),
                    },
                    spacing_zyx=read_spacing_zyx(
                        os.path.join(batch["pair_dir"][0], "cti.nii.gz")
                    ),
                    title=f"{batch['pair_id'][0]} - "
                    f"{os.path.basename(figure_path)}",
                )
    return {
        "val_ssim_pre": float(np.mean([m["ssim"] for m in pre])),
        "val_ssim": float(np.mean([m["ssim"] for m in post])),
        "val_mse_pre": float(np.mean([m["mse"] for m in pre])),
        "val_mse": float(np.mean([m["mse"] for m in post])),
    }


def main() -> None:
    """Trains the affine network with periodic validation and checkpoints."""
    args = parse_args()

    # create experiment folders and log file
    paths = prepare_experiment(output_dir=args.output_dir, name=args.name)
    log_path = start_logging(log_dir=paths["logs"], prefix="train")
    print(f"[{now()}] log: {log_path}")

    device = setup_device(device=args.device, gpu=args.gpu)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # data reading and loading
    train_ids = read_split_ids(args.train_split)
    val_ids = read_split_ids(args.val_split)
    train_set = PairedCTDataset(
        args.dataset_dir, case_ids=train_ids, random_flip=not args.no_flip
    )
    val_set = PairedCTDataset(args.dataset_dir, case_ids=val_ids)
    print(
        f"Pairs: {len(train_set)} train ({len(train_ids)} patients), "
        f"{len(val_set)} validation ({len(val_ids)} patients)"
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model
    config = default_affine_config()
    model, transform = build_affine(config=config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # resume from checkpoint if requested
    start_epoch, best_ssim = 0, -np.inf
    if args.resume:
        checkpoint_path = resolve_checkpoint(
            experiment_dir=paths["experiment"], checkpoint=args.resume
        )
        state = load_checkpoint(path=checkpoint_path, device=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        best_ssim = state.get("best_ssim", -np.inf)
        print(
            f"Resumed from {checkpoint_path} "
            f"(epoch {start_epoch}, best SSIM {best_ssim:.4f})"
        )

    # experiment info file (a new one on every resume)
    info_name = (
        f"experiment_resume_epoch_{start_epoch}.json"
        if args.resume
        else "experiment.json"
    )
    save_experiment_info(
        paths["experiment"],
        file_name=info_name,
        args=args,
        model_config=config,
        dataset_dir=args.dataset_dir,
        extra={"train_pairs": len(train_set), "val_pairs": len(val_set)},
    )

    history_path = os.path.join(paths["experiment"], "history.csv")
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time = time.time()
        print(f"\n[{now()}] ==== Epoch {epoch}/{args.num_epochs} ====")
        train_loss = train_one_epoch(
            model,
            transform,
            train_loader,
            optimizer,
            criterion,
            device=device,
            epoch=epoch,
        )

        # figure on the first epoch, every save_every epochs and on the last
        save_figure = (
            epoch == 1
            or epoch == args.num_epochs
            or (args.save_every > 0 and epoch % args.save_every == 0)
        )
        figure_path = None
        if save_figure:
            figure_path = os.path.join(
                paths["figures"], f"epoch_{epoch:04d}.png"
            )
        val = validate(
            model,
            transform,
            val_loader,
            device=device,
            figure_path=figure_path,
        )
        is_best = val["val_ssim"] > best_ssim
        best_ssim = max(best_ssim, val["val_ssim"])
        best_flag = " | best" if is_best else ""
        print(
            f"train loss {train_loss:.6f} "
            f"| val SSIM {val['val_ssim_pre']:.4f} -> {val['val_ssim']:.4f} "
            f"| val MSE {val['val_mse_pre']:.6f} -> {val['val_mse']:.6f}"
            f"{best_flag}"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_ssim": best_ssim,
            "val": val,
            "config": affine_config_for_checkpoint(config),
            "args": vars(args),
        }
        save_checkpoint(
            state,
            checkpoint_dir=paths["checkpoints"],
            epoch=epoch,
            save_every=args.save_every,
            is_best=is_best,
        )
        append_history(
            history_path,
            row={
                "epoch": epoch,
                "train_loss": train_loss,
                **val,
                "is_best": is_best,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - start_time, 1),
            },
        )
    print(f"[{now()}] Finished. Best validation SSIM: {best_ssim:.4f}")


if __name__ == "__main__":
    main()
