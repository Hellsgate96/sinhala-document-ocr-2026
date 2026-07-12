"""CLI: generate the detector-in-the-loop training supplement (v3 domain-gap fix).

Renders full synthetic pages (paragraph / bordered card / poem / mixed
Sinhala-English / letterhead), applies page-level phone-capture augmentation,
runs the SAME line detector used at inference time, and saves the detector's
actual output crops paired with their transcript - see
``src/data/page_synth.generate_detector_in_the_loop``.

Usage:
    python scripts/generate_pages.py --config configs/local.yaml --num-pages 3000

Output (``paths.synthetic_pages_dir``, default ``data/synthetic_pages``):
    images/page_XXXXXX_lNN.png, train_labels.txt, val_labels.txt

Feed the result into training with:
    python -m src.recognition.train --config configs/local.yaml \
        --extra-labels data/synthetic_pages/train_labels.txt
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.page_synth import LAYOUTS, generate_detector_in_the_loop
from src.data.synthetic_generator import load_corpus
from src.utils.common import configure_stdout_utf8, get_logger, load_config


def main():
    parser = argparse.ArgumentParser(description="Generate detector-in-the-loop training crops.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--num-pages", type=int, default=3000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("generate_pages")
    cfg = load_config(args.config)
    syn = cfg["synthetic"]

    out_dir = args.out or cfg["paths"].get("synthetic_pages_dir", "data/synthetic_pages")
    corpus = load_corpus(cfg["paths"].get("corpus"), warn=logger.warning)
    logger.info(f"loaded {len(corpus)} corpus lines")

    stats = generate_detector_in_the_loop(
        out_dir=out_dir,
        num_pages=args.num_pages,
        font_paths=syn["fonts"],
        font_sizes=syn["font_sizes"],
        corpus=corpus,
        detection_cfg=cfg.get("detection", {}),
        crop_padding_x=int(cfg.get("detection", {}).get("crop_padding_x", 10)),
        crop_padding_y=int(cfg.get("detection", {}).get("crop_padding_y", 5)),
        min_crop_height=int(cfg.get("detection", {}).get("min_crop_height", 14)),
        layouts=LAYOUTS,
        seed=args.seed,
        logger=logger,
        progress=not args.no_progress,
    )
    logger.info(f"done: {stats['num_crops']} crops from {stats['num_pages']} pages")
    logger.info(f"per-layout detector exact-match rate: {stats['match_rate']}")


if __name__ == "__main__":
    main()
