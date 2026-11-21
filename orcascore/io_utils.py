"""Experiment bookkeeping: devices, logging, checkpoints, experiment info,
history and NIfTI output.
"""

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Literal

import numpy as np
import SimpleITK as sitk
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPLIT_DIR = os.path.join(REPO_ROOT, "splits", "orcascore")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "experiments")


class Logger:
    """Writes everything printed both to the terminal and to a log file."""

    def __init__(self, log_path: str) -> None:
        self.terminal = sys.stdout
        self.log = open(log_path, "a", buffering=1, encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def now() -> str:
    """Current date and time as text (YYYY-MM-DD HH:MM:SS)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def setup_device(
    device: Literal["cuda", "cpu"] = "cuda", gpu: int = 0
) -> torch.device:
    """Selects the device and prints the available GPUs.

    With Docker --gpus device=N, the chosen GPU is cuda:0 inside the
    container.

    Args:
        device: "cuda" or "cpu".
        gpu: GPU index used when device is "cuda".
    """
    if device == "cpu":
        print("Using CPU")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; use --device cpu for a CPU run."
        )
    for index in range(torch.cuda.device_count()):
        print(f"  GPU #{index}: {torch.cuda.get_device_name(index)}")
    torch.cuda.set_device(gpu)
    print(f"Using GPU #{gpu}: {torch.cuda.get_device_name(gpu)}")
    return torch.device(f"cuda:{gpu}")


def prepare_experiment(output_dir: str, name: str) -> dict:
    """Creates the experiment folders and returns their paths.

    Layout: <output_dir>/<name>/{checkpoints, figures, logs}.

    Returns:
        dict with the keys "experiment", "checkpoints", "figures" and
        "logs", each with the corresponding path.
    """
    experiment_dir = os.path.join(output_dir, name)
    paths = {
        "experiment": experiment_dir,
        "checkpoints": os.path.join(experiment_dir, "checkpoints"),
        "figures": os.path.join(experiment_dir, "figures"),
        "logs": os.path.join(experiment_dir, "logs"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def start_logging(log_dir: str, prefix: str) -> str:
    """Redirects stdout/stderr to the terminal plus a timestamped log file.

    Returns:
        path of the log file (<log_dir>/<prefix>_<YYYYMMDD_HHMMSS>.log).
    """
    log_path = os.path.join(
        log_dir, f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    sys.stdout = Logger(log_path=log_path)
    sys.stderr = sys.stdout
    return log_path


def config_to_dict(config) -> dict:
    """Converts an ml_collections ConfigDict (or dict) to a plain
    JSON-friendly dict (tuples become lists)."""
    raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in raw.items()
    }


def git_info() -> dict:
    """Commit of the repository and whether there were uncommitted changes.

    Returns:
        {"commit": ..., "uncommitted_changes": ...}; both None when git is
        not available.
    """
    git = ["git", "-c", "safe.directory=*", "-C", REPO_ROOT]
    try:
        commit = subprocess.run(
            git + ["rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            git + ["status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}
    return {"commit": commit, "uncommitted_changes": bool(status)}


def save_experiment_info(
    experiment_dir: str,
    file_name: str,
    args,
    model_config,
    dataset_dir: str,
    extra: dict = None,
) -> str:
    """Saves everything needed to know what was run: command, arguments,
    network config, git commit, environment and the dataset config.json.

    An existing file is never overwritten: a timestamp is added to the name.

    Args:
        experiment_dir: folder where the JSON is written.
        file_name: e.g. "experiment.json", "experiment_resume_epoch_40.json",
            "infer.json".
        args: argparse namespace.
        model_config: network ConfigDict (or dict).
        dataset_dir: dataset folder (its config.json is copied).
        extra: any additional information (e.g. affine checkpoint used).

    Returns:
        path of the written file.
    """
    dataset_config_path = os.path.join(dataset_dir, "config.json")
    dataset_config = None
    if os.path.exists(dataset_config_path):
        with open(dataset_config_path, encoding="utf-8") as file:
            dataset_config = json.load(file)
    info = {
        "date": now(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "model_config": config_to_dict(model_config),
        "git": git_info(),  # commit of the code that was run
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
            "host": platform.node(),
        },
        "dataset_dir": dataset_dir,
        "dataset_config": dataset_config,
        "extra": extra or {},
    }
    path = os.path.join(experiment_dir, file_name)
    if os.path.exists(path):
        stem, extension = os.path.splitext(path)
        path = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2, ensure_ascii=False, default=str)
    return path


def append_history(history_path: str, row: dict) -> None:
    """Appends one epoch to history.csv (writes the header on the first
    row)."""
    write_header = not os.path.exists(history_path)
    with open(history_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    epoch: int,
    save_every: int,
    is_best: bool,
) -> None:
    """Saves the checkpoints of one epoch.

    Policy:
        - last.pth.tar every epoch (to resume);
        - epc_<epoch>.pth.tar every `save_every` epochs (never when
          save_every <= 0);
        - best.pth.tar whenever the validation improves.

    Args:
        state: dict saved with torch.save.
        checkpoint_dir: folder of the checkpoints.
        epoch: number of completed epochs (1-based).
        save_every: period of the epc_<epoch> copies.
        is_best: whether this epoch has the best validation so far.
    """
    last_path = os.path.join(checkpoint_dir, "last.pth.tar")
    # write to a temporary file first, so an interruption never leaves a
    # broken last.pth.tar
    temporary_path = f"{last_path}.tmp"
    torch.save(state, temporary_path)
    os.replace(temporary_path, last_path)
    if save_every > 0 and epoch % save_every == 0:
        shutil.copyfile(
            last_path, os.path.join(checkpoint_dir, f"epc_{epoch}.pth.tar")
        )
    if is_best:
        shutil.copyfile(
            last_path, os.path.join(checkpoint_dir, "best.pth.tar")
        )


def resolve_checkpoint(experiment_dir: str, checkpoint: str) -> str:
    """Path of a checkpoint of the experiment.

    Args:
        experiment_dir: folder of the experiment.
        checkpoint: "best", "last", "epc_<n>" or a file path.
    """
    if os.path.isfile(checkpoint):
        return checkpoint
    name = (
        checkpoint
        if checkpoint.endswith(".pth.tar")
        else f"{checkpoint}.pth.tar"
    )
    path = os.path.join(experiment_dir, "checkpoints", name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(path: str, device) -> dict:
    """Loads a checkpoint as a dict with the weights under "model_state".

    A file that holds only a state_dict, or the weights under "state_dict"
    (format of the authors' scripts), is wrapped in that same format.
    """
    state = torch.load(path, map_location=device)
    if "model_state" not in state:
        state = {"model_state": state.get("state_dict", state)}
    return state


def save_like_reference(
    array: np.ndarray, reference_path: str, out_path: str
) -> None:
    """Saves a (z, y, x) array as NIfTI with the geometry (spacing, origin,
    direction) of a reference image.

    Args:
        array: volume to save; must have the size of the reference.
        reference_path: the pair's cti.nii.gz, so the output overlays the
            dataset files.
        out_path: output .nii.gz file.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(reference_path)
    reader.ReadImageInformation()
    image = sitk.GetImageFromArray(array)
    if tuple(image.GetSize()) != tuple(reader.GetSize()):
        raise ValueError(
            f"Size {image.GetSize()} differs from the reference "
            f"{reader.GetSize()}."
        )
    image.SetSpacing(reader.GetSpacing())
    image.SetOrigin(reader.GetOrigin())
    image.SetDirection(reader.GetDirection())
    sitk.WriteImage(image, out_path, useCompression=True)
