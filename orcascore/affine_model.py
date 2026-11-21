"""Builds, loads and applies the TransMorph-Affine network from any folder.

TransMorph_affine/models and TransMorph/models are both packages named
"models", so the affine package is imported here under another name. Both the
affine scripts and the deformable scripts (which chain affine -> deformable)
use this module.
"""

import importlib
import importlib.util
import os
import sys

import ml_collections
import torch

from orcascore.io_utils import REPO_ROOT, config_to_dict, load_checkpoint

AFFINE_MODELS_DIR = os.path.join(REPO_ROOT, "TransMorph_affine", "models")
PACKAGE_NAME = "transmorph_affine_models"


def import_affine_module():
    """Imports TransMorph_affine/models/TransMorph_affine.py as
    transmorph_affine_models.TransMorph_affine.
    """
    module_name = f"{PACKAGE_NAME}.TransMorph_affine"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        os.path.join(AFFINE_MODELS_DIR, "__init__.py"),
        submodule_search_locations=[AFFINE_MODELS_DIR],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)
    return importlib.import_module(module_name)


def default_affine_config() -> ml_collections.ConfigDict:
    """Affine config of the OrCaScore dataset, defined in
    get_3DTransMorph_Affine_config (TransMorph_affine/models)."""
    module = import_affine_module()
    return module.configs.get_3DTransMorph_Affine_config()


def build_affine(config: ml_collections.ConfigDict, device) -> tuple:
    """Creates the affine network and the transform that applies its output.

    Returns:
        (model, transform): TransMorphAffine and AffineTransform, on `device`.
    """
    module = import_affine_module()
    model = module.TransMorphAffine(config).to(device)
    transform = module.AffineTransform().to(device)
    return model, transform


def load_affine(checkpoint_path: str, device) -> tuple:
    """Loads a trained affine network in eval mode.

    The config saved in the checkpoint is used when present, so the network
    always matches the one that was trained.

    Returns:
        (model, transform, config).
    """
    state = load_checkpoint(path=checkpoint_path, device=device)
    if "config" in state:
        config = ml_collections.ConfigDict(state["config"])
    else:
        config = default_affine_config()
    model, transform = build_affine(config=config, device=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, transform, config


def predict_affine(
    model, transform, moving: torch.Tensor, fixed: torch.Tensor
) -> dict:
    """Runs the affine network and applies the transform to `moving` at full
    resolution.

    Args:
        model: TransMorphAffine network.
        transform: AffineTransform object.
        moving: (B, 1, D, H, W) tensor, CTAI.
        fixed: (B, 1, D, H, W) tensor, CTI.

    Returns:
        dict with:
            - "warped": moving after the affine (bilinear);
            - "mat": (B, 3, 4) matrix that maps the CTI grid to the CTAI
              (normalized coordinates, align_corners=False, x/y/z order);
            - "inv_mat": (B, 3, 4) inverse of "mat";
            - "params": dict with "rotation", "scale" and "translation"
              (B, 3) and "shear" (B, 6).
    """
    rotation, scale, translation, shear = model((moving, fixed))
    warped, mat, inv_mat = transform(
        moving, rotation, scale, translation, shear
    )
    return {
        "warped": warped,
        "mat": mat,
        "inv_mat": inv_mat,
        "params": {
            "rotation": rotation,
            "scale": scale,
            "translation": translation,
            "shear": shear,
        },
    }


def apply_affine(
    transform, volume: torch.Tensor, mat: torch.Tensor, mode: str
) -> torch.Tensor:
    """Applies a saved affine matrix to a volume.

    Args:
        transform: AffineTransform object.
        volume: (B, C, D, H, W) tensor in the CTAI space.
        mat: (B, 3, 4) matrix ("mat" of predict_affine).
        mode: "bilinear" for images, "nearest" for labels.
    """
    return transform.apply_affine(volume, mat, mode)


def affine_config_for_checkpoint(config: ml_collections.ConfigDict) -> dict:
    """Config stored inside the checkpoints, so the network can be rebuilt."""
    return config_to_dict(config)
