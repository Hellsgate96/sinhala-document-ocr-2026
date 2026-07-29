"""Recognition inference preprocessing — match training normalization in dataset.py.

Training expects grayscale line images: dark text on a light background, resized
to a fixed height, pixel values scaled to [0, 1] then normalized to roughly [-1, 1].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.dataset import resize_keep_height

_DEFAULT_PAD = 255

ArrayLike = Union[np.ndarray, Image.Image]


def _to_gray_uint8(image: ArrayLike) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
        return arr
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _pad_gray_to_min_height(gray: np.ndarray, min_h: int, pad_value: int = _DEFAULT_PAD) -> np.ndarray:
    h, w = gray.shape[:2]
    if h >= min_h:
        return gray
    top = (min_h - h) // 2
    bottom = min_h - h - top
    return cv2.copyMakeBorder(gray, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=pad_value)


def should_invert_polarity(gray: np.ndarray, threshold: float = 128.0) -> bool:
    """True when the background is darker than text (light-on-dark)."""
    return float(gray.mean()) < threshold


def prepare_line_for_recognition(
    image: ArrayLike,
    height: int = 48,
    max_width: int = 512,
    mode: str = "auto",
    auto_invert: bool = True,
    denoise: bool = False,
    min_model_width: int = 0,
    pad_to_height: bool = True,
    pad_value: int = _DEFAULT_PAD,
) -> Image.Image:
    """Prepare a line crop for CRNN input (grayscale, dark-on-light, resized).

    Parameters
    ----------
    mode:
        ``"auto"`` — invert when mean intensity < 128 if ``auto_invert`` is set.
        ``"none"`` — no polarity change.
        ``"invert"`` — always invert.
    """
    gray = _to_gray_uint8(image)

    if denoise:
        gray = cv2.medianBlur(gray, 3)

    if mode == "invert":
        gray = cv2.bitwise_not(gray)
    elif mode == "auto" and auto_invert and should_invert_polarity(gray):
        gray = cv2.bitwise_not(gray)

    if pad_to_height and gray.shape[0] < height:
        gray = _pad_gray_to_min_height(gray, height, pad_value=pad_value)

    pil = Image.fromarray(gray, mode="L")
    return resize_keep_height(
        pil, height, max_width, min_width=min_model_width, pad_value=pad_value,
    )


def line_image_to_tensor(
    img: Image.Image,
    height: int,
    max_width: int,
    channels: int = 1,
) -> torch.Tensor:
    """Convert a prepared PIL line image to a (1, C, H, W) tensor (same as dataset)."""
    img = resize_keep_height(img.convert("L" if channels == 1 else "RGB"), height, max_width)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if channels == 1:
        arr = arr[None, :, :]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0)


def prepare_line_tensor(
    image: ArrayLike,
    height: int = 48,
    max_width: int = 512,
    channels: int = 1,
    mode: str = "auto",
    auto_invert: bool = True,
    denoise: bool = False,
    min_model_width: int = 0,
    pad_to_height: bool = True,
) -> torch.Tensor:
    """Full path: numpy/PIL line crop -> model input tensor."""
    prepared = prepare_line_for_recognition(
        image,
        height=height,
        max_width=max_width,
        mode=mode,
        auto_invert=auto_invert,
        denoise=denoise,
        min_model_width=min_model_width,
        pad_to_height=pad_to_height,
    )
    return line_image_to_tensor(prepared, height, max_width, channels)


# Test-time augmentation. Every variant keeps the crop's aspect ratio, so all of
# them resize to the same width and therefore the same CTC sequence length - that
# is what makes their log-probabilities averageable in predict_tensor.
TTA_VARIANTS = ("none", "sharpen", "contrast")


def apply_tta_variant(gray: np.ndarray, variant: str) -> np.ndarray:
    """Photometric-only variant of a grayscale crop (shape is never changed)."""
    if variant == "none":
        return gray
    if variant == "sharpen":
        # Unsharp mask. On a 16 px line upscaled to 48 px the interpolation has
        # smeared the pre-base vowel signs into the consonant; this pulls the
        # stroke edges back apart.
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
        return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    if variant == "contrast":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    raise ValueError(f"unknown TTA variant: {variant!r}")


def prepare_line_tensor_variants(
    image: ArrayLike,
    variants=TTA_VARIANTS,
    **kwargs: Any,
) -> torch.Tensor:
    """Stack the TTA variants of one crop into a single (N, C, H, W) batch."""
    gray = _to_gray_uint8(image)
    tensors = [
        prepare_line_tensor(apply_tta_variant(gray, v), **kwargs) for v in variants
    ]
    return torch.cat(tensors, dim=0)


def prepared_line_for_display(
    image: ArrayLike,
    height: int = 48,
    max_width: int = 512,
    mode: str = "auto",
    auto_invert: bool = True,
    denoise: bool = False,
    min_model_width: int = 0,
    pad_to_height: bool = True,
) -> np.ndarray:
    """Return uint8 grayscale array after preparation (for debugging plots)."""
    pil = prepare_line_for_recognition(
        image, height=height, max_width=max_width,
        mode=mode, auto_invert=auto_invert, denoise=denoise,
        min_model_width=min_model_width, pad_to_height=pad_to_height,
    )
    return np.asarray(pil, dtype=np.uint8)


def inference_options_from_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge ``configs/default.yaml`` inference section with image geometry."""
    cfg = cfg or {}
    inf = dict(cfg.get("inference") or {})
    img = cfg.get("image") or {}
    return {
        "height": int(inf.get("image_height", img.get("height", 48))),
        "max_width": int(img.get("max_width", 512)),
        "channels": int(img.get("channels", 1)),
        "auto_invert": bool(inf.get("auto_invert", True)),
        "use_grayscale": bool(inf.get("use_grayscale", True)),
        "binarize": bool(inf.get("binarize", False)),
        "denoise": bool(inf.get("denoise", False)),
        "min_model_width": int(inf.get("min_model_width", 0)),
        "pad_to_height": bool(inf.get("pad_to_height", True)),
        "mode": "auto" if inf.get("auto_invert", True) else "none",
        "decode": str(inf.get("decode", "greedy")).lower(),
        "beam_width": int(inf.get("beam_width", 10)),
        # Character-LM shallow fusion (used only by decode: beam_lm).
        "lm_weight": float(inf.get("lm_weight", 0.0)),
        "insertion_bonus": float(inf.get("insertion_bonus", 0.0)),
        "beam_top_k": int(inf.get("beam_top_k", 8)),
        "lm_order": int(inf.get("lm_order", 6)),
        "tta": bool(inf.get("tta", False)),
        "tta_variants": tuple(inf.get("tta_variants") or TTA_VARIANTS),
    }


DECODE_KEYS = (
    "beam_width", "lm_weight", "insertion_bonus", "beam_top_k", "lm_order",
    "tta", "tta_variants",
)


def decode_kwargs_from_options(opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the decoder knobs to splat into ``predict_*`` calls."""
    opts = opts or {}
    kwargs: Dict[str, Any] = {"decode_mode": str(opts.get("decode", "greedy")).lower()}
    defaults = inference_options_from_config(None)
    for key in DECODE_KEYS:
        kwargs[key] = opts.get(key, defaults[key])
    return kwargs
