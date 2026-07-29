"""Tests for the optional test-time augmentation path.

TTA is measured but *off* by default (it wins on the real photographs and loses
on the synthetic pages - see RESULTS.md). These tests pin the two properties the
averaging depends on, so it stays usable and cannot silently break the default
single-variant path.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest
import torch

from src.recognition.inference import (
    TTA_VARIANTS,
    apply_tta_variant,
    inference_options_from_config,
    prepare_line_tensor,
    prepare_line_tensor_variants,
)
from src.utils.common import load_config


def _tiny_line(height=18, width=200):
    gray = np.full((height, width), 240, dtype=np.uint8)
    gray[height // 3: height // 2, 10:width - 10] = 30
    return gray


def test_variants_share_one_shape_so_log_probs_can_be_averaged():
    """The whole scheme relies on every variant giving the same CTC length."""
    gray = _tiny_line()
    single = prepare_line_tensor(gray, height=48, max_width=512, pad_to_height=False)
    batch = prepare_line_tensor_variants(gray, height=48, max_width=512, pad_to_height=False)

    assert batch.shape[0] == len(TTA_VARIANTS)
    assert batch.shape[1:] == single.shape[1:]
    # "none" must be bit-identical to the non-TTA path, so enabling TTA only ever
    # adds hypotheses rather than changing the baseline one.
    assert torch.allclose(batch[0:1], single)


def test_variants_actually_differ():
    gray = _tiny_line()
    batch = prepare_line_tensor_variants(gray, height=48, max_width=512, pad_to_height=False)
    for i in range(1, batch.shape[0]):
        assert not torch.allclose(batch[i: i + 1], batch[0:1])


def test_apply_variant_never_changes_shape_or_dtype():
    gray = _tiny_line(height=23, width=157)
    for variant in TTA_VARIANTS:
        out = apply_tta_variant(gray, variant)
        assert out.shape == gray.shape
        assert out.dtype == np.uint8


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        apply_tta_variant(_tiny_line(), "rotate")


def test_tta_is_off_in_the_delivered_config():
    cfg = load_config(os.path.join(ROOT, "configs", "local.yaml"))
    assert inference_options_from_config(cfg)["tta"] is False
