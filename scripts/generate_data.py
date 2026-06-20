"""CLI wrapper: generate synthetic Sinhala line data from the config.

Example:
    python scripts/generate_data.py --config configs/default.yaml --num 2000
    python scripts/generate_data.py --num 20 --out data/synthetic_sample
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the project root importable when run as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.synthetic_generator import generate, load_word_list  # noqa: E402
from src.utils.common import configure_stdout_utf8, get_logger, load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Sinhala OCR data.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num", type=int, default=None, help="number of samples")
    parser.add_argument("--out", default=None, help="output directory override")
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("generate_data")
    cfg = load_config(args.config)
    syn = cfg["synthetic"]

    out_dir = args.out or cfg["paths"]["synthetic_dir"]
    num = args.num or syn["num_samples"]
    words = load_word_list(cfg["paths"]["word_list"])

    counts = generate(
        out_dir=out_dir,
        num_samples=num,
        font_paths=syn["fonts"],
        font_sizes=syn["font_sizes"],
        words=words,
        min_words=syn["min_words"],
        max_words=syn["max_words"],
        augment=syn["augment"],
        split=syn["split"],
        seed=cfg["project"]["seed"],
        logger=logger,
    )
    logger.info(f"counts = {counts}")


if __name__ == "__main__":
    main()