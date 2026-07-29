"""Unit tests for training-log parser / curve helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.train_curves import (
    append_history_epoch,
    load_eval_summary,
    load_train_history,
    parse_train_log,
    resolve_train_history,
)


SAMPLE_LOG = """\
00:00:01 | INFO    | train | starting
00:01:00 | INFO    | train | epoch 001 | train_loss = 0.0917 | lr = 8.00e-05
00:01:30 | INFO    | train | epoch 001 | val CER = 0.0369 | val WER = 0.0973
00:02:00 | INFO    | train | epoch 002 | train_loss = 0.0662 | lr = 8.00e-05
00:02:30 | INFO    | train | epoch 002 | val CER = 0.0358 | val WER = 0.0967
00:03:00 | INFO    | train | epoch 003 | train_loss = 0.0552 | lr = 4.00e-05
00:03:30 | INFO    | train | epoch 003 | val CER = 0.0348 | val WER = 0.0912
00:03:31 | INFO    | train | training complete. best val CER = 0.0348
"""


def test_parse_train_log_utf8(tmp_path: Path):
    log = tmp_path / "train_sample.log"
    log.write_text(SAMPLE_LOG, encoding="utf-8")
    hist = parse_train_log(log)
    assert hist.epochs == [1, 2, 3]
    assert hist.train_loss[0] == 0.0917
    assert hist.val_cer[-1] == 0.0348
    assert hist.val_wer[0] == 0.0973
    assert hist.lr[-1] == 4.00e-05
    assert hist.best_val_cer == 0.0348


def test_parse_train_log_utf16(tmp_path: Path):
    log = tmp_path / "train_utf16.log"
    log.write_bytes(SAMPLE_LOG.encode("utf-16"))
    hist = parse_train_log(log)
    assert len(hist) == 3
    assert hist.best_val_cer == 0.0348


def test_load_and_append_history(tmp_path: Path):
    path = tmp_path / "train_history.json"
    append_history_epoch(path, epoch=1, train_loss=0.5, val_cer=0.1, val_wer=0.2, lr=1e-4)
    append_history_epoch(path, epoch=2, train_loss=0.4, val_cer=0.09, val_wer=0.18, lr=1e-4)
    # overwrite epoch 2
    append_history_epoch(path, epoch=2, train_loss=0.39, val_cer=0.088, val_wer=0.17, lr=5e-5)
    hist = load_train_history(path)
    assert hist.epochs == [1, 2]
    assert hist.train_loss[1] == 0.39
    assert hist.best_val_cer == 0.088


def test_resolve_bundled_history():
    hist = resolve_train_history(ROOT, models_dir=Path(ROOT) / "models_missing_xyz")
    assert len(hist) >= 1
    assert hist.best_val_cer is not None


def test_eval_summary_bundled():
    summary = load_eval_summary(ROOT)
    assert "sets" in summary
    assert any(s.get("held_out") for s in summary["sets"])
    assert any(s["id"] == "print_photos" for s in summary["sets"])


def test_history_list_of_records(tmp_path: Path):
    path = tmp_path / "h.json"
    path.write_text(
        json.dumps(
            {
                "history": [
                    {"epoch": 1, "train_loss": 1.0, "val_cer": 0.5, "val_wer": 0.6, "lr": 1e-3},
                ]
            }
        ),
        encoding="utf-8",
    )
    hist = load_train_history(path)
    assert hist.epochs == [1]
    assert hist.val_cer[0] == 0.5


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        test_parse_train_log_utf8(base)
        test_parse_train_log_utf16(base)
        test_load_and_append_history(base)
        test_history_list_of_records(base)
    test_resolve_bundled_history()
    test_eval_summary_bundled()
    print("test_train_curves: ALL PASSED")
