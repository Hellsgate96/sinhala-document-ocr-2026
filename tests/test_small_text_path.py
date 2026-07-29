"""Regression tests for the small/low-resolution text path (jul28 round).

Two things broke on ~15-20 px lines and are pinned here:

1. ``inference.pad_to_height`` used to white-pad a short crop up to the model
   height, so an 18 px line kept its glyphs at 18 px inside a 48 px input while
   training had always *resized* to 48. Flipping it to false upscales instead.
2. ``scripts.generate_hard_lines.degrade_low_res`` now leaves half of its
   output at the crushed height so ``OCRLineDataset`` performs exactly the same
   upscale that inference does, and ``--tiny-ratio`` can force it on every
   style to build a dedicated small-text set.
"""
from __future__ import annotations

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from PIL import Image

from scripts.generate_hard_lines import degrade_low_res
from src.recognition.inference import (
    decode_kwargs_from_options,
    inference_options_from_config,
    prepared_line_for_display,
)
from src.utils.common import load_config


def _tiny_line(height=18, width=200):
    """A short crop: white background with one dark horizontal stroke."""
    gray = np.full((height, width), 240, dtype=np.uint8)
    gray[height // 3: height // 2, 10:width - 10] = 30
    return gray


def test_pad_to_height_false_upscales_instead_of_padding():
    gray = _tiny_line(height=18)
    padded = prepared_line_for_display(gray, height=48, max_width=512, pad_to_height=True)
    upscaled = prepared_line_for_display(gray, height=48, max_width=512, pad_to_height=False)

    assert padded.shape[0] == upscaled.shape[0] == 48
    # Padding keeps the ink confined to the middle 18/48 of the input; the
    # upscale spreads the same stroke over ~2.7x more rows.
    ink_rows_padded = int((padded.min(axis=1) < 128).sum())
    ink_rows_upscaled = int((upscaled.min(axis=1) < 128).sum())
    assert ink_rows_upscaled > ink_rows_padded * 1.5
    # It is also much wider, because aspect ratio is preserved from 18 px, not 48.
    assert upscaled.shape[1] > padded.shape[1]


def test_local_config_keeps_the_upscaling_path():
    cfg = load_config(os.path.join(ROOT, "configs", "local.yaml"))
    opts = inference_options_from_config(cfg)
    assert opts["pad_to_height"] is False


def test_decode_kwargs_round_trip():
    cfg = load_config(os.path.join(ROOT, "configs", "local.yaml"))
    kwargs = decode_kwargs_from_options(inference_options_from_config(cfg))
    assert kwargs["decode_mode"] in {"greedy", "beam", "beam_lm"}
    for key in ("beam_width", "lm_weight", "insertion_bonus", "beam_top_k", "lm_order"):
        assert key in kwargs
    # Must be splattable into predict_* without unexpected keywords.
    from src.recognition.predict import predict_image  # noqa: F401
    import inspect

    params = inspect.signature(predict_image).parameters
    assert set(kwargs).issubset(params)


def test_degrade_low_res_preserves_aspect_ratio_and_crushes_height():
    img = Image.new("RGB", (600, 90), (255, 255, 255))
    rng = random.Random(0)
    small = 0
    for _ in range(40):
        out = degrade_low_res(img, rng, min_h=11, max_h=26)
        w, h = out.size
        if h <= 26:
            small += 1
            assert 11 <= h <= 26
            # Aspect ratio held, so resize_keep_height yields the same width.
            assert abs(w / h - 600 / 90) < 0.35
        else:
            assert (w, h) == (600, 90)
    assert 5 < small < 35, "expected roughly half the samples to stay crushed"


def test_degrade_low_res_is_a_noop_when_already_small():
    img = Image.new("RGB", (60, 9), (255, 255, 255))
    out = degrade_low_res(img, random.Random(1), min_h=11, max_h=26)
    assert out.size == (60, 9)
