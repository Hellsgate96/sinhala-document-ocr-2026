"""Run inference with a trained CRNN: greedy CTC decoding on image(s).

Usage:
    python -m src.recognition.predict --checkpoint models/crnn_best.pth \
        --charset models/charset.json --image path/to/line.png
    python -m src.recognition.predict --checkpoint models/crnn_best.pth \
        --charset models/charset.json --folder data/synthetic_sample/images
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.charset import Charset
from src.recognition.decode import decode_log_probs
from src.recognition.inference import (inference_options_from_config, prepare_line_tensor,
                                       prepare_line_tensor_variants)
from src.recognition.model import build_crnn
from src.utils.common import (configure_stdout_utf8, get_device, load_checkpoint,
                              load_config)

ROOT = Path(__file__).resolve().parents[2]

LM_DECODE_MODES = {"beam_lm", "lm_beam", "lm"}


def resolve_decoder_lm(decode_mode: str, lm: Any = None, lm_order: int = 6):
    """Return the character LM for LM-fused decoding (memoised), else ``None``."""
    if lm is not None:
        return lm
    if str(decode_mode).lower() not in LM_DECODE_MODES:
        return None
    from src.postprocess.char_lm import build_char_lm

    return build_char_lm(str(ROOT), order=lm_order)



def digit_ratio(text: str) -> float:
    """Fraction of characters that are ASCII digits (0-9)."""
    if not text:
        return 0.0
    digits = sum(1 for ch in text if ch.isdigit())
    return digits / max(1, len(text))


def looks_like_garbage_prediction(text: str, digit_threshold: float = 0.4) -> bool:
    """Heuristic: mostly digits often means a failed recognition on prose lines."""
    stripped = text.strip()
    if not stripped:
        return False
    return digit_ratio(stripped) >= digit_threshold


def format_prediction_with_warning(text: str, ground_truth: str | None = None) -> str:
    """Return text, optionally prefixed with a low-confidence warning."""
    if ground_truth is not None:
        return text
    if looks_like_garbage_prediction(text):
        return f"[warn: low confidence / likely wrong] {text}"
    return text

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def image_to_tensor(
    path: str,
    height: int,
    max_width: int,
    channels: int,
    auto_invert: bool = True,
    denoise: bool = False,
    min_model_width: int = 0,
    pad_to_height: bool = True,
    tta: bool = False,
    tta_variants=None,
) -> torch.Tensor:
    """Load a line image, apply inference preprocessing, return (N, C, H, W)."""
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    image = bgr if bgr is not None else Image.open(path)
    build = prepare_line_tensor_variants if tta else prepare_line_tensor
    return build(
        image,
        **({"variants": tta_variants} if tta and tta_variants else {}),
        height=height,
        max_width=max_width,
        channels=channels,
        auto_invert=auto_invert,
        denoise=denoise,
        min_model_width=min_model_width,
        pad_to_height=pad_to_height,
    )


@torch.no_grad()
def predict_tensor(
    model,
    charset: Charset,
    tensor: torch.Tensor,
    device,
    decode_mode: str = "greedy",
    beam_width: int = 10,
    lm: Any = None,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
    beam_top_k: int = 8,
    lm_order: int = 6,
) -> str:
    tensor = tensor.to(device)
    log_probs = model(tensor)  # (T, B, C)
    if log_probs.shape[1] > 1:
        # A TTA batch: average the per-variant distributions in probability space
        # (a mean of log-probs would be a geometric mean, which lets any single
        # variant that is confidently wrong veto the others).
        log_probs = log_probs.exp().mean(dim=1, keepdim=True).clamp_min(1e-12).log()
    decode_mode = str(decode_mode).lower()
    if decode_mode == "greedy":
        indices = log_probs.argmax(2).squeeze(1)
        return charset.ctc_greedy_decode(indices.tolist())
    return decode_log_probs(
        log_probs[:, 0, :].cpu(),
        charset,
        mode=decode_mode,
        beam_width=beam_width,
        lm=resolve_decoder_lm(decode_mode, lm, lm_order),
        lm_weight=lm_weight,
        insertion_bonus=insertion_bonus,
        top_k=beam_top_k,
    )


@torch.no_grad()
def predict_image(
    model,
    charset: Charset,
    path: str,
    height: int,
    max_width: int,
    channels: int,
    device,
    auto_invert: bool = True,
    denoise: bool = False,
    min_model_width: int = 0,
    pad_to_height: bool = True,
    warn_garbage: bool = True,
    decode_mode: str = "greedy",
    beam_width: int = 10,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
    beam_top_k: int = 8,
    lm_order: int = 6,
    tta: bool = False,
    tta_variants=None,
) -> str:
    """Predict the transcript of one line image file."""
    tensor = image_to_tensor(
        path, height, max_width, channels,
        auto_invert=auto_invert, denoise=denoise,
        min_model_width=min_model_width, pad_to_height=pad_to_height,
        tta=tta, tta_variants=tta_variants,
    )
    text = predict_tensor(
        model, charset, tensor, device,
        decode_mode=decode_mode, beam_width=beam_width,
        lm_weight=lm_weight, insertion_bonus=insertion_bonus,
        beam_top_k=beam_top_k, lm_order=lm_order,
    )
    if warn_garbage:
        return format_prediction_with_warning(text)
    return text


@torch.no_grad()
def predict_line_array(
    model,
    charset: Charset,
    image: Union[np.ndarray, Image.Image],
    height: int,
    max_width: int,
    channels: int,
    device,
    auto_invert: bool = True,
    denoise: bool = False,
    min_model_width: int = 0,
    pad_to_height: bool = True,
    warn_garbage: bool = True,
    ground_truth: Optional[str] = None,
    decode_mode: str = "greedy",
    beam_width: int = 10,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
    beam_top_k: int = 8,
    lm_order: int = 6,
    tta: bool = False,
    tta_variants=None,
) -> str:
    """Predict from an in-memory line crop (BGR, grayscale, or PIL)."""
    build = prepare_line_tensor_variants if tta else prepare_line_tensor
    tensor = build(
        image,
        **({"variants": tta_variants} if tta and tta_variants else {}),
        height=height,
        max_width=max_width,
        channels=channels,
        auto_invert=auto_invert,
        denoise=denoise,
        min_model_width=min_model_width,
        pad_to_height=pad_to_height,
    )
    text = predict_tensor(
        model, charset, tensor, device,
        decode_mode=decode_mode, beam_width=beam_width,
        lm_weight=lm_weight, insertion_bonus=insertion_bonus,
        beam_top_k=beam_top_k, lm_order=lm_order,
    )
    if warn_garbage:
        return format_prediction_with_warning(text, ground_truth=ground_truth)
    return text


def predict_folder(
    model,
    charset,
    folder,
    height,
    max_width,
    channels,
    device,
    auto_invert: bool = True,
    denoise: bool = False,
) -> List[Tuple[str, str]]:
    results = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if os.path.splitext(path)[1].lower() in IMG_EXTS:
            results.append(
                (
                    path,
                    predict_image(
                        model, charset, path, height, max_width, channels, device,
                        auto_invert=auto_invert, denoise=denoise,
                    ),
                )
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="CRNN inference (CTC greedy decode).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--charset", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", help="path to a single line image")
    parser.add_argument("--folder", help="path to a folder of line images")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    configure_stdout_utf8()
    if not args.image and not args.folder:
        parser.error("provide --image or --folder")

    cfg = load_config(args.config)
    inf_opts = inference_options_from_config(cfg)
    device = get_device(args.device)
    charset = Charset.load(args.charset)
    model = build_crnn(charset.num_classes, cfg.get("model"),
                       in_channels=inf_opts["channels"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    h, mw, ch = inf_opts["height"], inf_opts["max_width"], inf_opts["channels"]
    auto_inv = inf_opts["auto_invert"]
    denoise = inf_opts["denoise"]
    min_w = inf_opts.get("min_model_width", 0)
    pad_h = inf_opts.get("pad_to_height", True)
    decode_mode = str(inf_opts.get("decode", "greedy")).lower()
    beam_width = int(inf_opts.get("beam_width", 10))

    if args.image:
        text = predict_image(
            model, charset, args.image, h, mw, ch, device,
            auto_invert=auto_inv, denoise=denoise,
            min_model_width=min_w, pad_to_height=pad_h,
            decode_mode=decode_mode, beam_width=beam_width,
        )
        print(f"{args.image}\t{text}")
    if args.folder:
        for path, text in predict_folder(
            model, charset, args.folder, h, mw, ch, device,
            auto_invert=auto_inv, denoise=denoise,
            min_model_width=min_w, pad_to_height=pad_h,
        ):
            print(f"{path}\t{text}")


if __name__ == "__main__":
    main()
