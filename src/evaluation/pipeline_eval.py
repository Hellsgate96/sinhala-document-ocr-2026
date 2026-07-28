"""Shared end-to-end (detect + recognize) pipeline evaluation helpers.

Used by ``scripts/eval_real_images.py`` (single image, optional GT) and the
batch/realistic evaluation scripts (``scripts/run_realistic_eval.py``,
``scripts/build_adversarial_pages.py``) so there is exactly one code path
between "detect lines on a page" and "corpus CER/WER for that page" - the
same path a real user's upload goes through.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from src.detection.text_detection import (
    binarize_for_detection,
    build_detector,
    crop_lines,
    estimate_page_skew,
    rotate_page,
)
from src.evaluation.metrics import corpus_cer, corpus_wer, cer as cer_fn
from src.recognition.predict import format_prediction_with_warning, predict_line_array


def run_pipeline_on_gray(
    model,
    charset,
    gray: np.ndarray,
    detector,
    inf_opts: Dict[str, Any],
    det_cfg: Dict[str, Any],
    device,
) -> Dict[str, Any]:
    """Detect lines on a grayscale page and recognize each: returns boxes, crops, texts.

    When ``detection.deskew`` is enabled (default) the page skew is estimated
    from the ink mask and the page is rotated upright before detection -
    slightly rotated photos otherwise merge adjacent lines in the projection
    profile. Returned boxes/crops (and the ``gray`` key) refer to the
    deskewed image; ``deskew_angle`` reports the applied correction.
    """
    angle = 0.0
    if bool(det_cfg.get("deskew", True)):
        mask = detector.ink_mask(gray) if hasattr(detector, "ink_mask") else binarize_for_detection(gray)
        angle = estimate_page_skew(mask, max_angle=float(det_cfg.get("deskew_max_angle", 5.0)))
        if abs(angle) >= float(det_cfg.get("deskew_min_angle", 0.3)):
            gray = rotate_page(gray, angle)
        else:
            angle = 0.0
    boxes = detector.detect(gray)
    crops = crop_lines(
        gray, boxes,
        padding_x=int(det_cfg.get("crop_padding_x", 10)),
        padding_y=int(det_cfg.get("crop_padding_y", 5)),
        min_crop_height=int(det_cfg.get("min_crop_height", 14)),
    )
    texts: List[str] = []
    display_texts: List[str] = []
    for crop in crops:
        text = predict_line_array(
            model, charset, crop,
            height=inf_opts["height"], max_width=inf_opts["max_width"], channels=inf_opts["channels"],
            device=device, auto_invert=inf_opts["auto_invert"], denoise=inf_opts["denoise"],
            min_model_width=inf_opts.get("min_model_width", 0), pad_to_height=inf_opts.get("pad_to_height", True),
            decode_mode=inf_opts.get("decode", "greedy"),
            warn_garbage=False,  # raw text for scoring - the CLI warning prefix must never pollute CER/WER
        )
        texts.append(text)
        display_texts.append(format_prediction_with_warning(text))
    return {
        "boxes": boxes,
        "crops": crops,
        "texts": texts,
        "display_texts": display_texts,
        "gray": gray,
        "deskew_angle": angle,
    }


def run_pipeline_on_image_path(
    model,
    charset,
    image_path: str,
    detector,
    inf_opts: Dict[str, Any],
    det_cfg: Dict[str, Any],
    device,
) -> Dict[str, Any]:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    result = run_pipeline_on_gray(model, charset, gray, detector, inf_opts, det_cfg, device)
    # Keep the debug overlay aligned with the (possibly deskewed) boxes/crops.
    angle = result.get("deskew_angle", 0.0)
    result["bgr"] = rotate_page(bgr, angle) if angle else bgr
    result["image_path"] = image_path
    return result


def score_against_gt(gt_lines: Sequence[str], pred_lines: Sequence[str]) -> Dict[str, Any]:
    """Order-aligned corpus CER/WER + per-line CER, matching min(len(gt), len(pred))
    lines (detection under/over-segmentation is NOT hidden - it is reflected in
    ``num_gt`` vs ``num_pred`` and in the lines that don't get aligned/scored)."""
    n = min(len(gt_lines), len(pred_lines))
    aligned_gt = list(gt_lines[:n])
    aligned_pred = list(pred_lines[:n])
    per_line = [
        {"line": i + 1, "ref": aligned_gt[i], "hyp": aligned_pred[i], "cer": cer_fn(aligned_gt[i], aligned_pred[i])}
        for i in range(n)
    ]
    return {
        "num_gt": len(gt_lines),
        "num_pred": len(pred_lines),
        "num_aligned": n,
        "corpus_cer": corpus_cer(aligned_gt, aligned_pred) if n else 1.0,
        "corpus_wer": corpus_wer(aligned_gt, aligned_pred) if n else 1.0,
        "per_line": per_line,
    }


def save_debug(debug_dir: str, bgr: np.ndarray, boxes, crops, texts, extra: Optional[Dict] = None) -> None:
    import json

    os.makedirs(debug_dir, exist_ok=True)
    canvas = bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(debug_dir, "boxes.jpg"), canvas)
    for i, crop in enumerate(crops, start=1):
        cv2.imwrite(os.path.join(debug_dir, f"line_{i:02d}.png"), crop)
    payload = {"lines": [{"line": i + 1, "text": t} for i, t in enumerate(texts)]}
    if extra:
        payload.update(extra)
    with open(os.path.join(debug_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
