"""Inference of a trained TransMorph-Affine on the OrCaScore NIfTI dataset.

For every pair it saves, at full size (same grid and header as the pair's
cti.nii.gz):
    - ctai_afim.nii.gz: CTAI after the affine (bilinear);
    - arteria_afim.nii.gz: artery after the affine (nearest), when the pair
      has an artery;
    - transformacao.json: matrix (mat), inverse (inv_mat) and raw parameters.

And one metrics.csv with every metric before/after, with and without padding.
"""

# isort: skip_file
# (the repository root must be on sys.path before importing orcascore)
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from orcascore.affine_model import (  # noqa: E402
    apply_affine,
    load_affine,
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
    now,
    resolve_checkpoint,
    save_experiment_info,
    save_like_reference,
    setup_device,
    start_logging,
)
from orcascore.metrics import (  # noqa: E402
    print_summary,
    registration_row,
    to_numpy,
)


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference of TransMorph-Affine"
    )
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
        "--checkpoint",
        default="best",
        help="best, last, epc_<n> or a file path",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["val"],
        choices=["val", "train"],
        help="which splits to infer, val and/or train (default: val)",
    )
    parser.add_argument(
        "--train_split",
        default=os.path.join(DEFAULT_SPLIT_DIR, "train.csv"),
        help="CSV with the training case_ids (column ID)",
    )
    parser.add_argument(
        "--val_split",
        default=os.path.join(DEFAULT_SPLIT_DIR, "test.csv"),
        help="CSV with the validation/test case_ids (column ID)",
    )
    parser.add_argument(
        "--no_lpips", action="store_true", help="skip LPIPS (faster)"
    )
    parser.add_argument(
        "--no_save_volumes",
        action="store_true",
        help="only compute the metrics",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="device"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index (cuda:N)"
    )
    return parser.parse_args()


def save_pair_outputs(
    pair_out_dir: str,
    reference_path: str,
    warped: torch.Tensor,
    artery_warped,
    output: dict,
) -> None:
    """Writes the registered volumes (full size, header of the CTI) and the
    transform of one pair.

    Args:
        pair_out_dir: output folder of the pair.
        reference_path: the pair's cti.nii.gz (geometry of the outputs).
        warped: CTAI after the affine.
        artery_warped: artery after the affine, or None.
        output: result of predict_affine (mat, inv_mat, params).
    """
    os.makedirs(pair_out_dir, exist_ok=True)
    save_like_reference(
        to_numpy(warped),
        reference_path=reference_path,
        out_path=os.path.join(pair_out_dir, "ctai_afim.nii.gz"),
    )
    if artery_warped is not None:
        save_like_reference(
            to_numpy(artery_warped).astype(np.uint8),
            reference_path=reference_path,
            out_path=os.path.join(pair_out_dir, "arteria_afim.nii.gz"),
        )
    transform_info = {
        "convention": "normalized coordinates, align_corners=False, "
        "vector order (x, y, z)",
        "mat": output["mat"][0].cpu().tolist(),
        "inv_mat": output["inv_mat"][0].cpu().tolist(),
        "params": {
            key: value[0].cpu().tolist()
            for key, value in output["params"].items()
        },
    }
    with open(
        os.path.join(pair_out_dir, "transformacao.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(transform_info, file, indent=2)


def infer_set(
    model, transform, loader, set_name: str, out_dir: str, device, args
) -> list:
    """Infers every pair of one split; returns the metric rows."""
    rows = []
    lpips_device = None if args.no_lpips else str(device)
    with torch.no_grad():
        for batch in loader:
            pair_id = batch["pair_id"][0]
            print(f"[{now()}] {set_name} {pair_id}")
            moving = batch["x"].to(device)
            fixed = batch["y"].to(device)
            padding = padding_to_dict(batch["padding"][0])
            output = predict_affine(
                model, transform, moving=moving, fixed=fixed
            )

            # artery follows the moving image (CTAI)
            # calcification belongs to the fixed CTI
            artery_stages, artery_warped = None, None

            if bool(batch["has_artery"][0]):
                artery = batch["artery"].to(device)
                artery_warped = apply_affine(
                    transform, artery, output["mat"], mode="nearest"
                )
                artery_stages = {"pre": artery, "affine": artery_warped}
            calcification = (
                batch["calcification"]
                if bool(batch["has_calcification"][0])
                else None
            )

            row = registration_row(
                pair_id=pair_id,
                fixed=fixed,
                stages={"pre": moving, "affine": output["warped"]},
                padding=padding,
                calcification=calcification,
                artery_stages=artery_stages,
                lpips_device=lpips_device,
            )
            rows.append(
                {"set": set_name, "case_id": batch["case_id"][0], **row}
            )

            if not args.no_save_volumes:
                save_pair_outputs(
                    pair_out_dir=os.path.join(out_dir, set_name, pair_id),
                    reference_path=os.path.join(
                        batch["pair_dir"][0], "cti.nii.gz"
                    ),
                    warped=output["warped"],
                    artery_warped=artery_warped,
                    output=output,
                )
    return rows


def main() -> None:
    """Runs the inference on the chosen splits and writes volumes and
    metrics."""
    args = parse_args()

    # read the experiment folder and checkpoint
    experiment_dir = os.path.join(args.output_dir, args.name)
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"Experiment not found: {experiment_dir}")
    checkpoint_path = resolve_checkpoint(
        experiment_dir=experiment_dir, checkpoint=args.checkpoint
    )
    checkpoint_name = os.path.basename(checkpoint_path).replace(".pth.tar", "")

    # create the output folder and start logging
    out_dir = os.path.join(experiment_dir, f"infer_{checkpoint_name}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = start_logging(log_dir=out_dir, prefix="infer")
    print(f"[{now()}] log: {log_path}")
    device = setup_device(device=args.device, gpu=args.gpu)

    model, transform, config = load_affine(
        checkpoint_path=checkpoint_path, device=device
    )
    print(f"Checkpoint: {checkpoint_path}")
    save_experiment_info(
        out_dir,
        file_name="infer.json",
        args=args,
        model_config=config,
        dataset_dir=args.dataset_dir,
        extra={"checkpoint": checkpoint_path},
    )

    splits = {"val": args.val_split, "train": args.train_split}
    rows = []
    for set_name in args.sets:
        # read the pairs of this split (with labels, for the overlap metric)
        dataset = PairedCTDataset(
            args.dataset_dir,
            case_ids=read_split_ids(splits[set_name]),
            read_labels=True,
        )
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
        )

        set_rows = infer_set(
            model,
            transform,
            loader,
            set_name=set_name,
            out_dir=out_dir,
            device=device,
            args=args,
        )
        print(f"\n==== {set_name}: mean +/- std ({len(set_rows)} pairs) ====")
        print_summary(pd.DataFrame(set_rows))
        rows.extend(set_rows)

    # save the metrics of all splits in one CSV, one line per pair
    metrics_path = os.path.join(out_dir, "metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"[{now()}] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
