"""Shared utilities: seeding, logging, config loading, image IO, checkpoints.

Heavy dependencies (``torch``) are imported lazily inside the functions that need
them so this module can be imported in a CPU-only / torch-less environment.
"""

from __future__ import annotations

import logging
import os
import copy
import random
from typing import Any, Dict, Optional

import numpy as np


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 1337) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = "sinhala_ocr", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that prints to stdout (UTF-8 safe)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def configure_stdout_utf8() -> None:
    """Best-effort: force UTF-8 stdout so Sinhala prints on Windows consoles."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into a copy of base (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a dict.

    If the file defines ``_inherits: other.yaml``, that file is loaded first and
    merged with the current file (child keys override).
    """
    import yaml

    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    inherit = cfg.pop("_inherits", None)
    if inherit:
        base_path = inherit if os.path.isabs(inherit) else os.path.join(os.path.dirname(path), inherit)
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[list]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` style CLI overrides (best-effort type casting)."""
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _cast(value)
    return cfg


def _cast(value: str) -> Any:
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


# --------------------------------------------------------------------------
# Image IO
# --------------------------------------------------------------------------
def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Read an image as a NumPy array (RGB or grayscale)."""
    import cv2
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(path: str, image: np.ndarray) -> None:
    """Write a NumPy image array to disk (expects RGB for 3-channel)."""
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


# --------------------------------------------------------------------------
# Device + checkpoints (torch)
# --------------------------------------------------------------------------
def get_device(preference: str = "auto"):
    """Resolve a torch device from a preference string (auto|cuda|cpu)."""
    import torch
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




CheckpointMode = str  # "auto" | "baseline" | "finetuned"


def _recognition_checkpoint_paths(cfg: Dict[str, Any], repo_root: Optional[os.PathLike] = None) -> tuple[str, str]:
    paths = cfg.get("paths") or {}
    models_dir = paths.get("models_dir", "models")
    root = os.fspath(repo_root) if repo_root is not None else os.getcwd()
    finetune_name = paths.get("finetune_best") or "crnn_finetuned.pth"
    baseline_name = paths.get("baseline_best") or "crnn_best.pth"
    finetune_path = os.path.join(root, models_dir, os.path.basename(finetune_name))
    baseline_path = os.path.join(root, models_dir, os.path.basename(baseline_name))
    return baseline_path, finetune_path


def resolve_recognition_checkpoint(
    cfg: Dict[str, Any],
    repo_root: Optional[os.PathLike] = None,
    *,
    mode: CheckpointMode = "auto",
    compare_to_poem_gt: bool = False,
    use_poem_finetune: bool = False,
) -> str:
    """Pick CRNN weights for inference.

    * **baseline** — always ``crnn_best.pth`` (general documents).
    * **finetuned** — ``crnn_finetuned.pth`` when present, else baseline.
    * **auto** (default) — finetuned only for poem evaluation
      (``compare_to_poem_gt`` or ``use_poem_finetune``); otherwise baseline.
    """
    mode = (mode or "auto").lower()
    if mode not in {"auto", "baseline", "finetuned"}:
        raise ValueError(f"Invalid checkpoint mode: {mode!r}")

    baseline_path, finetune_path = _recognition_checkpoint_paths(cfg, repo_root)

    if mode == "baseline":
        return baseline_path
    if mode == "finetuned":
        return finetune_path if os.path.isfile(finetune_path) else baseline_path

    use_finetune = bool(compare_to_poem_gt or use_poem_finetune)
    if use_finetune and os.path.isfile(finetune_path):
        return finetune_path
    return baseline_path

def save_checkpoint(path: str, model, optimizer=None, epoch: int = 0,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """Save a model (and optional optimizer) checkpoint."""
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None, map_location: str = "cpu"):
    """Load a checkpoint into ``model`` (and optionally ``optimizer``)."""
    import torch
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
