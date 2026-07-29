# -*- coding: utf-8 -*-
"""Report which "holdout" label files actually leaked into training.

Label sets accumulate over months and a later round can quietly fold a holdout
back into the training mix - which is exactly what happened to
``user_batch1_holdout.txt`` here. This script is the evidence behind the
``status`` column printed by ``scripts/run_eval_suite.py``: it compares the
transcripts of each evaluation set against every transcript the trainer sees.

Usage:
    python scripts/check_holdout_leakage.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import read_labels

# Everything merged in via --extra-labels in the documented training commands,
# plus the un-augmented sources those augmentations were derived from.
TRAINING_LABEL_FILES = (
    "data/real/labels/poem_kanyawee.txt",
    "data/real/labels/poem_kanyawee_aug.txt",
    "data/real/labels/user_batch1.txt",
    "data/real/labels/user_batch1_aug.txt",
    "data/real/labels/web_batch1.txt",
    "data/real/labels/web_batch1_aug.txt",
    "data/real/labels/web_batch1_acts.txt",
    "data/real/labels/web_batch1_acts_aug.txt",
)

EVAL_LABEL_FILES = (
    "data/real/labels/user_batch1_holdout.txt",
    "data/real/labels/web_batch1_holdout.txt",
    "data/real/labels/poem_kanyawee.txt",
)


def _texts(rel: str) -> List[Tuple[str, str]]:
    path = ROOT / rel
    if not path.is_file():
        return []
    return read_labels(str(path))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="append", default=None,
                   help="extra evaluation label file(s) to check")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    train_texts = set()
    train_images = set()
    for rel in TRAINING_LABEL_FILES:
        for image, text in _texts(rel):
            train_texts.add(text.strip())
            train_images.add(os.path.basename(image))
    print(f"training transcripts: {len(train_texts)} distinct "
          f"from {len(TRAINING_LABEL_FILES)} label file(s)\n")

    print(f"{'evaluation set':<32} {'n':>4} {'text seen':>10} {'image seen':>11}  verdict")
    for rel in list(EVAL_LABEL_FILES) + list(args.eval or []):
        rows = _texts(rel)
        if not rows:
            continue
        text_hits = sum(1 for _i, t in rows if t.strip() in train_texts)
        image_hits = sum(1 for i, _t in rows if os.path.basename(i) in train_images)
        if text_hits == len(rows):
            verdict = "IN TRAIN - not a holdout"
        elif text_hits:
            verdict = "PARTIAL leak"
        else:
            verdict = "clean holdout"
        print(f"{os.path.basename(rel):<32} {len(rows):>4} {text_hits:>10} {image_hits:>11}  {verdict}")

    print("\nFully held out (no transcript ever shown to the trainer):")
    print("  data/eval_real/print_photos/*.jpg   real photographed pages")
    print("  data/eval_pages/*.png               synthetic pages, separate generation run")
    print("  data/eval_real/adversarial/*.png    hand-built acceptance pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
