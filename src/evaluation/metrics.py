"""Evaluation metrics for OCR: CER, WER, field-level accuracy and CPU timing.

Edit distance uses the ``editdistance`` package when available and transparently
falls back to a pure-Python implementation, so the core metrics run with no extra
dependencies (and without a GPU). Model evaluation utilities import torch lazily.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------
# Edit distance (with graceful fallback)
# --------------------------------------------------------------------------
def _pure_levenshtein(a: Sequence, b: Sequence) -> int:
    """Classic O(len(a) * len(b)) Levenshtein distance."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance using ``editdistance`` if installed, else pure Python."""
    try:
        import editdistance
        return int(editdistance.eval(a, b))
    except ImportError:
        return _pure_levenshtein(list(a), list(b))


# --------------------------------------------------------------------------
# CER / WER
# --------------------------------------------------------------------------
def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit_distance(chars) / len(reference chars)."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return edit_distance(list(reference), list(hypothesis)) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = edit_distance(words) / number of reference words."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def corpus_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Aggregate CER over a corpus (total edits / total reference characters)."""
    total_edits = 0
    total_chars = 0
    for ref, hyp in zip(references, hypotheses):
        total_edits += edit_distance(list(ref), list(hyp))
        total_chars += max(1, len(ref))
    return total_edits / total_chars if total_chars else 0.0


def corpus_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Aggregate WER over a corpus (total word edits / total reference words)."""
    total_edits = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = ref.split(), hyp.split()
        total_edits += edit_distance(r, h)
        total_words += max(1, len(r))
    return total_edits / total_words if total_words else 0.0


# --------------------------------------------------------------------------
# Field-level accuracy (forms / invoices / ID fields)
# --------------------------------------------------------------------------
def field_accuracy(predicted: Dict[str, str], ground_truth: Dict[str, str],
                   normalize: Optional[Callable[[str], str]] = None) -> float:
    """Fraction of fields whose predicted value exactly matches ground truth."""
    if not ground_truth:
        return 0.0
    norm = normalize or (lambda s: s.strip())
    correct = sum(
        1 for k, v in ground_truth.items()
        if norm(predicted.get(k, "")) == norm(v)
    )
    return correct / len(ground_truth)


# --------------------------------------------------------------------------
# Model evaluation (torch) + CPU timing
# --------------------------------------------------------------------------
def evaluate_model(model, dataloader, charset, device="cpu",
                   measure_cpu_time: bool = True) -> Dict:
    """Evaluate a CRNN over a DataLoader, returning a CER/WER + timing report."""
    import torch

    model.eval()
    model.to(device)
    references: List[str] = []
    hypotheses: List[str] = []
    per_sample: List[Dict] = []
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, _targets, _tlens, _widths, texts in dataloader:
            images = images.to(device)
            start = time.perf_counter()
            log_probs = model(images)                 # (T, B, C)
            total_time += time.perf_counter() - start
            total_images += images.size(0)

            preds = log_probs.argmax(2).permute(1, 0)  # (B, T)
            for i, text in enumerate(texts):
                hyp = charset.ctc_greedy_decode(preds[i].tolist())
                references.append(text)
                hypotheses.append(hyp)
                per_sample.append({"ref": text, "hyp": hyp,
                                   "cer": cer(text, hyp), "wer": wer(text, hyp)})

    report = {
        "num_samples": len(references),
        "cer": corpus_cer(references, hypotheses),
        "wer": corpus_wer(references, hypotheses),
        "per_sample": per_sample,
    }
    if measure_cpu_time and total_images:
        report["avg_inference_ms"] = 1000.0 * total_time / total_images
    return report


def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate a CRNN checkpoint (CER/WER).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--charset", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    from src.utils.common import configure_stdout_utf8, load_config
    from src.charset import Charset
    from src.data.dataset import build_dataloader
    from src.recognition.model import build_crnn
    from src.utils.common import load_checkpoint

    configure_stdout_utf8()
    args = _build_argparser().parse_args()
    cfg = load_config(args.config)
    charset = Charset.load(args.charset)
    model = build_crnn(charset.num_classes, cfg.get("model"),
                       in_channels=cfg["image"]["channels"])
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    loader = build_dataloader(
        args.labels, charset, batch_size=cfg["train"]["batch_size"],
        height=cfg["image"]["height"], max_width=cfg["image"]["max_width"],
        channels=cfg["image"]["channels"], shuffle=False,
    )
    report = evaluate_model(model, loader, charset, device=args.device)
    print(f"samples={report['num_samples']}  CER={report['cer']:.4f}  "
          f"WER={report['wer']:.4f}  avg_inference_ms={report.get('avg_inference_ms', 0):.2f}")