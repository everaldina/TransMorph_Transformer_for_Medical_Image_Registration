"""Registration metrics for a pair of images (fixed, moving), with or without
padding.

- SSIM and LPIPS: mean over axial slices (LPIPS is a 2D network, and a 3D
  SSIM would mix neighbours in z, where the spacing is several times larger
  than in XY);
- MSE and PSNR (and MAE, RMSE): over the whole volume;
- everything with data_range = 1 and without renormalization (images are
  already in [0, 1]);
- "padded": whole volume;
- "cropped": padding removed (the real extent of the pair, including the
  region outside the FOV, as in the alignment evaluation).
"""

import numpy as np
import pandas as pd
import torch
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from orcascore.dataset import AXES

LPIPS_BATCH = 8
_lpips_models = {}


def crop_padding(volume: np.ndarray, padding: dict) -> np.ndarray:
    """Removes the padding of the last three axes (z, y, x).

    Args:
        volume: array (..., z, y, x); leading axes (batch, channel) are kept.
        padding: {"z": [before, after], "y": [...], "x": [...]}, as in
            metadados.json.
    """
    slices = []
    for axis_index, axis in enumerate(AXES):
        # size of this spatial axis, counted from the end of the shape
        size = volume.shape[-3 + axis_index]
        before, after = padding[axis]

        # size - after (and not -after), so it also works when after == 0
        slices.append(slice(before, size - after))
    return volume[(..., *slices)]


def to_numpy(volume) -> np.ndarray:
    """Converts a tensor or array (1, 1, z, y, x) or (z, y, x) to a float32
    numpy array (z, y, x)."""
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().float().cpu().numpy()
    return np.asarray(volume, dtype=np.float32).reshape(volume.shape[-3:])


def get_lpips_model(device: str):
    """LPIPS (AlexNet) model, created once per device."""
    # imported here: lpips is only needed at inference and loads the
    # torchvision weights on first use
    import lpips

    if device not in _lpips_models:
        _lpips_models[device] = (
            lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        )
    return _lpips_models[device]


def lpips_per_slice(
    fixed: np.ndarray, moving: np.ndarray, device: str
) -> float:
    """Mean LPIPS over axial slices.

    Each slice goes from [0, 1] to the [-1, 1] range expected by LPIPS and is
    repeated in 3 channels; there is no per-slice renormalization.
    """
    model = get_lpips_model(device=device)
    values = []
    with torch.no_grad():
        for start in range(0, fixed.shape[0], LPIPS_BATCH):
            fixed_batch = torch.from_numpy(
                fixed[start:(start + LPIPS_BATCH)]
            ).to(device)
            moving_batch = torch.from_numpy(
                moving[start:(start + LPIPS_BATCH)]
            ).to(device)
            fixed_batch = (fixed_batch[:, None].repeat(1, 3, 1, 1) - 0.5) * 2
            moving_batch = (moving_batch[:, None].repeat(1, 3, 1, 1) - 0.5) * 2
            values.extend(model(fixed_batch, moving_batch).flatten().tolist())
    return float(np.mean(values))


def ssim_per_slice(fixed: np.ndarray, moving: np.ndarray) -> float:
    """Mean SSIM over axial slices, data_range = 1."""
    values = [
        structural_similarity(fixed_slice, moving_slice, data_range=1.0)
        for fixed_slice, moving_slice in zip(fixed, moving)
    ]
    return float(np.mean(values))


def image_metrics(
    fixed: np.ndarray, moving: np.ndarray, lpips_device: str = None
) -> dict:
    """SSIM and LPIPS per slice; MSE, PSNR, MAE and RMSE over the volume.

    Args:
        fixed, moving: (z, y, x) arrays in [0, 1], same shape.
        lpips_device: device for LPIPS; None skips LPIPS (value NaN).
    """
    mse = float(mean_squared_error(fixed, moving))
    return {
        "ssim": ssim_per_slice(fixed=fixed, moving=moving),
        "lpips": (
            lpips_per_slice(fixed, moving, device=lpips_device)
            if lpips_device
            else np.nan
        ),
        "mse": mse,
        "psnr": float(peak_signal_noise_ratio(fixed, moving, data_range=1.0)),
        "mae": float(np.mean(np.abs(fixed - moving))),
        "rmse": float(np.sqrt(mse)),
    }


def validation_metrics(fixed, moving, padding: dict) -> dict:
    """Metrics used every epoch: SSIM per slice and MSE, without the padding.

    Returns:
        {"ssim": ..., "mse": ...}
    """
    fixed_cropped = crop_padding(to_numpy(fixed), padding=padding)
    moving_cropped = crop_padding(to_numpy(moving), padding=padding)
    return {
        "ssim": ssim_per_slice(fixed=fixed_cropped, moving=moving_cropped),
        "mse": float(mean_squared_error(fixed_cropped, moving_cropped)),
    }


def calcification_overlap(
    calcification: np.ndarray, artery: np.ndarray
) -> dict:
    """Share of the calcification voxels that fall inside the artery.

    Returns:
        dict with:
            - "pp_calc_in_artery": % of calcification voxels inside the
              artery (NaN when one of the labels is empty);
            - "calc_in_artery_voxels": number of those voxels (NaN likewise);
            - "artery_voxels": number of artery voxels.
    """
    calcification = calcification > 0.5
    artery = artery > 0.5
    calcification_voxels = int(calcification.sum())
    artery_voxels = int(artery.sum())
    if calcification_voxels == 0 or artery_voxels == 0:
        return {
            "pp_calc_in_artery": np.nan,
            "calc_in_artery_voxels": np.nan,
            "artery_voxels": artery_voxels,
        }
    inside = int((calcification & artery).sum())
    return {
        "pp_calc_in_artery": 100 * inside / calcification_voxels,
        "calc_in_artery_voxels": inside,
        "artery_voxels": artery_voxels,
    }


def registration_row(
    pair_id: str,
    fixed,
    stages: dict,
    padding: dict,
    calcification=None,
    artery_stages: dict = None,
    lpips_device: str = None,
) -> dict:
    """One metrics row per pair: every stage x {padded, cropped}, plus the
    calcification overlap.

    Args:
        pair_id: identifier of the pair.
        fixed: CTI (fixed image).
        stages: moving image at each stage, e.g. {"pre": before
            registration, "affine": after the affine, ...}.
        padding: padding of the pair, as in metadados.json.
        calcification: calcification label (CTI grid, not warped); None
            skips the overlap.
        artery_stages: artery label at each stage (same keys as stages);
            None skips the overlap.
        lpips_device: device for LPIPS; None skips LPIPS.

    Returns:
        dict with the columns "<metric>_<stage>_<padded|cropped>",
        "pp_calc_in_artery_<stage>" and "artery_voxels_<stage>".
    """
    row = {"pair_id": pair_id}
    fixed = to_numpy(fixed)
    fixed_cropped = crop_padding(fixed, padding=padding)
    for stage, moving in stages.items():
        moving = to_numpy(moving)
        regions = {
            "padded": (fixed, moving),
            "cropped": (fixed_cropped, crop_padding(moving, padding=padding)),
        }
        for region, (fixed_region, moving_region) in regions.items():
            metrics = image_metrics(
                fixed=fixed_region,
                moving=moving_region,
                lpips_device=lpips_device,
            )
            for name, value in metrics.items():
                row[f"{name}_{stage}_{region}"] = value

    # overlap is the same with or without padding (labels are zero in the
    # padding)
    if calcification is not None and artery_stages is not None:
        calcification = to_numpy(calcification)
        for stage, artery in artery_stages.items():
            overlap = calcification_overlap(
                calcification=calcification, artery=to_numpy(artery)
            )
            row[f"pp_calc_in_artery_{stage}"] = overlap["pp_calc_in_artery"]
            row[f"artery_voxels_{stage}"] = overlap["artery_voxels"]
    return row


def print_summary(table: pd.DataFrame) -> None:
    """Prints mean +/- std of every numeric column of a metrics table."""
    numeric = table.select_dtypes(include="number")
    for column in numeric.columns:
        values = numeric[column].dropna()
        if len(values) == 0:
            continue
        print(
            f"  {column:<32} {values.mean():.4f} +/- {values.std(ddof=0):.4f} "
            f"(n={len(values)})"
        )
