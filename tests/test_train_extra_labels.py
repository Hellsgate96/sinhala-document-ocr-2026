"""Regression test for the extra-labels base-directory inference bug fixed in
this overhaul: extra label files NOT under ``paths.real_dir`` (e.g. a
detector-in-the-loop page-crop supplement in a different directory) must
resolve their relative image paths against their OWN directory, not silently
against ``real_dir`` (which broke training with a FileNotFoundError)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.recognition.train import _infer_extra_base_dir


def test_labels_under_real_dir_use_real_dir():
    real_dir = os.path.join("data", "real")
    extra_path = os.path.join("data", "real", "labels", "poem_kanyawee.txt")
    assert _infer_extra_base_dir(extra_path, real_dir) == os.path.abspath(real_dir)


def test_labels_outside_real_dir_use_their_own_directory():
    real_dir = os.path.join("data", "real")
    extra_path = os.path.join("data", "synthetic_pages", "train_labels.txt")
    expected = os.path.abspath(os.path.join("data", "synthetic_pages"))
    assert _infer_extra_base_dir(extra_path, real_dir) == expected
