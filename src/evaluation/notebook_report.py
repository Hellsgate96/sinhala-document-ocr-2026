# -*- coding: utf-8 -*-
"""Notebook-friendly end-of-run evaluation report.

Used by ``notebooks/local_pipeline.ipynb`` and ``notebooks/colab_pipeline.ipynb``
so both print the same CER/WER / accuracy summary after inference.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.evaluation.metrics import (
    accuracy_pct,
    cer,
    corpus_cer,
    corpus_wer,
    wer,
)


def load_gt_lines(path: Union[str, Path]) -> List[str]:
    path = Path(path)
    if not path.is_file():
        return []
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# Back-compat alias used by notebooks.
_load_gt_lines = load_gt_lines


def resolve_page_gt(
    image_path: Optional[Union[str, Path]],
    repo_root: Optional[Union[str, Path]] = None,
    gt_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Find a sidecar ``.gt.txt`` or an explicit GT file for the test image."""
    if gt_path:
        p = Path(gt_path)
        return p if p.is_file() else None
    if not image_path:
        return None
    img = Path(image_path)
    sidecar = img.with_suffix(img.suffix + ".gt.txt")
    if not sidecar.is_file():
        sidecar = img.with_name(img.stem + ".gt.txt")
    if sidecar.is_file():
        return sidecar
    if repo_root:
        root = Path(repo_root)
        # Common held-out locations.
        for cand in [
            root / "data" / "eval_real" / "print_photos" / f"{img.stem}.gt.txt",
            root / "data" / "uploads" / f"{img.stem}.gt.txt",
        ]:
            if cand.is_file():
                return cand
    return None


def format_run_report(
    *,
    pred_lines: Sequence[str],
    gt_lines: Optional[Sequence[str]] = None,
    checkpoint_name: str = "crnn_best.pth",
    decode_mode: str = "beam_lm",
    image_height: int = 48,
    elapsed_s: Optional[float] = None,
    image_path: Optional[str] = None,
    post_correct: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a plain-text evaluation block for the end of a notebook run."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("EVALUATION METRICS")
    lines.append("=" * 72)
    lines.append(f"checkpoint     : {checkpoint_name}")
    lines.append(f"decode         : {decode_mode}")
    lines.append(f"post_correct   : {post_correct}")
    lines.append(f"image_height   : {image_height}")
    if image_path:
        lines.append(f"image          : {image_path}")
    lines.append(f"lines detected : {len(pred_lines)}")
    if elapsed_s is not None:
        lines.append(f"elapsed        : {elapsed_s:.2f}s")
    if extra:
        for k, v in extra.items():
            lines.append(f"{k:15}: {v}")

    lines.append("-" * 72)
    lines.append("Full transcription")
    lines.append("-" * 72)
    for i, t in enumerate(pred_lines, 1):
        lines.append(f"{i:02d}: {t}")

    if gt_lines:
        n = min(len(gt_lines), len(pred_lines))
        refs = list(gt_lines[:n])
        hyps = list(pred_lines[:n])
        c_cer = corpus_cer(refs, hyps)
        c_wer = corpus_wer(refs, hyps)
        lines.append("-" * 72)
        lines.append(f"Ground truth   : yes ({len(gt_lines)} GT lines, scoring {n} aligned)")
        if len(gt_lines) != len(pred_lines):
            lines.append(
                f"NOTE: GT lines ({len(gt_lines)}) != detected ({len(pred_lines)}); "
                "detection under/over-segmentation affects the score."
            )
        lines.append(
            f"Character Error Rate (CER) : {c_cer:.4f}  →  Character Accuracy: {accuracy_pct(c_cer):.2f}%"
        )
        lines.append(
            f"Word Error Rate (WER)      : {c_wer:.4f}  →  Word Accuracy: {accuracy_pct(c_wer):.2f}%"
        )
        lines.append("  (Character Accuracy = (1 − CER) × 100%; Word Accuracy = (1 − WER) × 100%)")
        lines.append("-" * 72)
        lines.append("Per-line CER / WER / accuracy")
        for i, (r, h) in enumerate(zip(refs, hyps), 1):
            lc, lw = cer(r, h), wer(r, h)
            lines.append(
                f"{i:02d}  CER={lc:.3f} ({accuracy_pct(lc):.1f}%)  "
                f"WER={lw:.3f} ({accuracy_pct(lw):.1f}%)"
            )
            if r != h:
                lines.append(f"    GT: {r}")
                lines.append(f"    PR: {h}")
    else:
        lines.append("-" * 72)
        lines.append("Ground truth   : not available")
        lines.append(
            "No CER/WER computed. Provide a sidecar `<image>.gt.txt` "
            "(one GT line per detected line)."
        )
        # Optional crude confidence proxy: fraction of lines that look non-empty.
        nonempty = sum(1 for t in pred_lines if t.strip())
        lines.append(f"non-empty lines: {nonempty}/{len(pred_lines)}")

    lines.append("=" * 72)
    return "\n".join(lines)


def print_run_report(**kwargs) -> str:
    report = format_run_report(**kwargs)
    print(report)
    return report


def timed_call(fn, *args, **kwargs):
    """Return ``(result, elapsed_seconds)`` for a callable."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
