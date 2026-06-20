"""Unit tests for evaluation metrics (CER / WER / field accuracy)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.metrics import cer, corpus_cer, corpus_wer, field_accuracy, wer


def test_cer_perfect():
    assert cer("ලංකාව", "ලංකාව") == 0.0


def test_cer_one_substitution():
    # 5 reference chars, 1 wrong -> CER = 0.2
    assert abs(cer("hello", "hallo") - 0.2) < 1e-9


def test_wer_basic():
    # 3 reference words, 1 wrong -> WER = 1/3
    assert abs(wer("the quick fox", "the quick dog") - (1 / 3)) < 1e-9


def test_empty_reference():
    assert cer("", "") == 0.0
    assert cer("", "x") == 1.0


def test_corpus_aggregation():
    refs = ["abc", "de"]
    hyps = ["abd", "de"]          # 1 edit over 5 chars
    assert abs(corpus_cer(refs, hyps) - (1 / 5)) < 1e-9
    assert corpus_wer(refs, hyps) >= 0.0


def test_field_accuracy():
    gt = {"name": "ගම", "amount": "100"}
    pred = {"name": "ගම", "amount": "180"}
    assert abs(field_accuracy(pred, gt) - 0.5) < 1e-9


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_metrics: ALL PASSED")