"""CLI: build a realistic, non-leaking held-out evaluation set of full
synthetic PAGES (v3 domain-gap fix, item 8 in the remediation plan).

Uses a disjoint RNG seed stream from training/detector-in-the-loop generation
and (when available) prefers font faces less represented during training, so
this is a genuine held-out check rather than re-scoring training data. Saves
full page images + line-ordered ground-truth transcripts (NOT pre-cropped -
the detector runs at evaluation time via run_realistic_eval.py, exactly like
a real upload).

Usage:
    python scripts/build_eval_pages.py --config configs/local.yaml --num-pages 10
    python scripts/run_realistic_eval.py --images-dir data/eval_pages \
        --checkpoint models/crnn_best.pth
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.page_synth import LAYOUTS, generate_eval_pages
from src.data.synthetic_generator import discover_fonts, load_corpus
from src.utils.common import configure_stdout_utf8, get_logger, load_config


def main():
    parser = argparse.ArgumentParser(description="Build realistic held-out eval pages.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--num-pages", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=987654321, help="disjoint from training seed")
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("build_eval_pages")
    cfg = load_config(args.config)
    syn = cfg["synthetic"]
    out_dir = args.out or cfg["paths"].get("eval_pages_dir", "data/eval_pages")

    corpus = load_corpus(cfg["paths"].get("corpus"), warn=logger.warning)
    # Prefer a font subset that skews away from the primary training default
    # (Nirmala UI) when alternatives are on disk, for a slightly more
    # out-of-distribution rendering style.
    candidates = [
        "fonts/NotoSerifSinhala-Regular.ttf",
        "fonts/AbhayaLibre-Regular.ttf",
        "fonts/Yaldevi-Variable.ttf",
        "fonts/NotoSansSinhala-Regular.ttf",
    ]
    fonts = discover_fonts(candidates, warn=logger.warning)
    if not fonts:
        fonts = syn["fonts"]
        logger.warning("no held-out font files found; falling back to training font list")

    paths = generate_eval_pages(
        out_dir=out_dir,
        num_pages=args.num_pages,
        font_paths=fonts,
        font_sizes=syn["font_sizes"],
        corpus=corpus,
        layouts=LAYOUTS,
        seed=args.seed,
        logger=logger,
    )
    logger.info(f"wrote {len(paths)} eval pages to {out_dir}")


if __name__ == "__main__":
    main()
