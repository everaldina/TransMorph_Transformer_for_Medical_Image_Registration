"""Validation figures (axial and coronal middle slices)."""

# isort: skip_file
# (the Agg backend must be selected before pyplot is imported)
import matplotlib
matplotlib.use("Agg")  # no display on the training server

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402

from orcascore.metrics import to_numpy  # noqa: E402


def read_spacing_zyx(image_path: str) -> tuple:
    """Spacing of a NIfTI file in (z, y, x) order, read from the header."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    reader.ReadImageInformation()
    spacing_x, spacing_y, spacing_z = reader.GetSpacing()
    return spacing_z, spacing_y, spacing_x


def save_comparison_figure(
    path: str, images: dict, spacing_zyx: tuple, title: str
) -> None:
    """Saves the axial (middle z) and coronal (middle y) slices of each image
    side by side.

    Args:
        path: output PNG file.
        images: {"column title": volume (z, y, x) or tensor (1, 1, z, y, x)}.
        spacing_zyx: voxel spacing, used to draw the coronal slice with the
            right proportion.
        title: title of the figure.
    """
    volumes = {name: to_numpy(volume) for name, volume in images.items()}
    figure, axes = plt.subplots(
        2, len(volumes), figsize=(4 * len(volumes), 6), squeeze=False
    )
    coronal_aspect = spacing_zyx[0] / spacing_zyx[2]
    for column, (name, volume) in enumerate(volumes.items()):
        axes[0, column].imshow(
            volume[volume.shape[0] // 2], cmap="gray", vmin=0, vmax=1
        )
        axes[0, column].set_title(name)
        axes[1, column].imshow(
            volume[:, volume.shape[1] // 2, :],
            cmap="gray",
            vmin=0,
            vmax=1,
            aspect=coronal_aspect,
        )
        for row in range(2):
            axes[row, column].axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def difference_image(fixed, moving) -> np.ndarray:
    """|fixed - moving|, to show the residual misalignment."""
    return np.abs(to_numpy(fixed) - to_numpy(moving))
