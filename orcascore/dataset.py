"""Reading of the paired CTI x CTAI dataset (NIfTI) produced by
Alinhamento_Axial_LPIPS.
"""

import json
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

AXES = ("z", "y", "x")


def read_volume(path: str) -> np.ndarray:
    """Reads a NIfTI volume as an array in (z, y, x) order."""
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def read_split_ids(csv_path: str) -> list:
    """Reads the patient ids (case_id) of one split.

    Args:
        csv_path: CSV with one id per line, in the column "ID" (or the
            first column).

    Returns:
        list of case_id (str).
    """
    table = pd.read_csv(csv_path)
    column = "ID" if "ID" in table.columns else table.columns[0]
    return [str(case_id).strip() for case_id in table[column]]


def read_dataset_config(dataset_dir: str) -> dict:
    """Reads the dataset config.json (how the dataset was generated)."""
    with open(
        os.path.join(dataset_dir, "config.json"), encoding="utf-8"
    ) as file:
        return json.load(file)


def padding_to_dict(padding: torch.Tensor) -> dict:
    """Converts the padding of one sample to the dict format of
    metadados.json.

    Args:
        padding: (3, 2) array, one row [before, after] per axis:
            [[z_before, z_after],
             [y_before, y_after],
             [x_before, x_after]]

    Returns:
        {"z": [z_before, z_after], "y": [y_before, y_after],
         "x": [x_before, x_after]}
    """
    padding = np.asarray(padding).reshape(3, 2)
    return {
        axis: [int(padding[i, 0]), int(padding[i, 1])]
        for i, axis in enumerate(AXES)
    }


class PairedCTDataset(Dataset):
    """Aligned CTI x CTAI pairs of a list of patients (case_id).

    Rules:
        - the split is done by case_id, never by pair_id, so in the "mix"
          dataset the original and the cytran pairs of the same patient
          always fall on the same set;
        - volumes are used as stored: already in [0, 1] with a fixed window
          and already padded (no per-volume normalization, no extra padding);
        - a missing label (tem_arteria / tem_calcificacao False) returns
          zeros, together with the flag, so "no label" is distinguishable
          from "empty label".

    Each item is a dict with "x" (CTAI, moving), "y" (CTI, fixed), both
    (1, z, y, x) float32, "pair_id", "case_id", "pair_dir" and "padding"
    ((3, 2), see padding_to_dict). With read_labels it also has "artery",
    "calcification", "has_artery" and "has_calcification".
    """

    def __init__(
        self,
        dataset_dir: str,
        case_ids: list,
        read_labels: bool = False,
        random_flip: bool = False,
    ) -> None:
        """
        Args:
            dataset_dir: dataset folder (with dataset.csv and config.json).
            case_ids: patients of this split.
            read_labels: also read arteria.nii.gz and calcificacao.nii.gz.
            random_flip: flip every axis with probability 0.5; images,
                labels and padding are flipped together.
        """
        table = pd.read_csv(os.path.join(dataset_dir, "dataset.csv"))
        table["case_id"] = table["case_id"].astype(str)
        missing = sorted(set(case_ids) - set(table["case_id"]))
        if missing:
            print(
                "[PairedCTDataset] case_ids not found in dataset.csv: "
                f"{missing}"
            )
        self.table = table[table["case_id"].isin(case_ids)].reset_index(
            drop=True
        )
        if len(self.table) == 0:
            raise ValueError(
                f"No pair of {dataset_dir} matches the given case_ids."
            )
        self.dataset_dir = dataset_dir
        self.read_labels = read_labels
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.table)

    def pair_dir(self, index: int) -> str:
        """Folder of one pair."""
        return os.path.join(self.dataset_dir, self.table.loc[index, "pasta"])

    def _read_label(
        self, pair_dir: str, file_name: str, exists: bool, shape: tuple
    ) -> np.ndarray:
        """Reads one label as a float32 (z, y, x) array, or zeros when the
        pair has no such label.

        Args:
            pair_dir: folder of the pair.
            file_name: name of the label file.
            exists: whether the label exists (tem_arteria /
                tem_calcificacao); False returns zeros.
            shape: shape of the zero array for a missing label.
        """
        if not exists:
            return np.zeros(shape, dtype=np.float32)
        return read_volume(os.path.join(pair_dir, file_name)).astype(
            np.float32
        )

    def __getitem__(self, index: int) -> dict:
        row = self.table.loc[index]
        pair_dir = self.pair_dir(index)

        # images as [channel, z, y, x], x = moving, y = fixed
        x = read_volume(os.path.join(pair_dir, "ctai.nii.gz")).astype(
            np.float32
        )
        y = read_volume(os.path.join(pair_dir, "cti.nii.gz")).astype(
            np.float32
        )
        volumes = {"x": x, "y": y}

        if self.read_labels:
            volumes["artery"] = self._read_label(
                pair_dir=pair_dir,
                file_name="arteria.nii.gz",
                exists=bool(row["tem_arteria"]),
                shape=x.shape,
            )
            volumes["calcification"] = self._read_label(
                pair_dir=pair_dir,
                file_name="calcificacao.nii.gz",
                exists=bool(row["tem_calcificacao"]),
                shape=x.shape,
            )

        padding = np.array(
            [
                [row[f"padding_{axis}_antes"], row[f"padding_{axis}_depois"]]
                for axis in AXES
            ],
            dtype=np.int64,
        )

        # same random flip for every volume;
        # the padding of a flipped axis swaps sides
        if self.random_flip:
            for axis in range(3):
                if np.random.rand() < 0.5:
                    volumes = {
                        key: np.flip(volume, axis=axis)
                        for key, volume in volumes.items()
                    }
                    padding[axis] = padding[axis][::-1]

        sample = {
            key: torch.from_numpy(np.ascontiguousarray(volume)[None])
            for key, volume in volumes.items()
        }
        sample["pair_id"] = str(row["pair_id"])
        sample["case_id"] = str(row["case_id"])
        sample["pair_dir"] = pair_dir
        sample["padding"] = torch.from_numpy(padding)
        if self.read_labels:
            sample["has_artery"] = bool(row["tem_arteria"])
            sample["has_calcification"] = bool(row["tem_calcificacao"])
        return sample
