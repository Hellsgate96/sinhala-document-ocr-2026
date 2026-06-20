"""CLI wrapper: generate synthetic Sinhala line data from the config.

Two paths are supported:

* Smoke test (default small run), e.g.::

      python scripts/generate_data.py --num 60 --out data/synthetic_sample

* Large training corpus (uses ``synthetic.large`` settings from the config)::

      python scripts/generate_data.py --large
      python scripts/generate_data.py --large --num 12000

The generator pulls vocabulary from ``paths.word_list`` and (when present) the
additional form-field vocabulary at ``paths.form_vocab``.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the project root importable when run as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.synthetic_generator import generate, load_word_lists  # noqa: E402
from src.utils.common import configure_stdout_utf8, get_logger, load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Sinhala OCR data.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num", type=int, default=None, help="number of samples")
    parser.add_argument("--out", default=None, help="output directory override")
    parser.add_argument("--seed", type=int, default=None, help="random seed override")
    parser.add_argument("--large", action="store_true",
                        help="use the large-corpus settings from synthetic.large")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm bar")
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("generate_data")
    cfg = load_config(args.config)
    syn = cfg["synthetic"]
    large_cfg = syn.get("large", {}) or {}

    # Resolve output directory + sample count (CLI > large/default config).
    if args.large:
        default_out = large_cfg.get("output_dir", cfg["paths"]["synthetic_dir"])
        default_num = large_cfg.get("num_samples", 10000)
    else:
        default_out = cfg["paths"]["synthetic_dir"]
        default_num = syn["num_samples"]
    out_dir = args.out or default_out
    num = args.num or default_num
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]

    # Vocabulary: main word list + optional form-field vocabulary.
    word_sources = [cfg["paths"]["word_list"]]
    form_vocab = cfg["paths"].get("form_vocab")
    if form_vocab:
        word_sources.append(form_vocab)
    words = load_word_lists(word_sources, warn=logger.warning)
    logger.info(f"loaded {len(words)} vocab entries from {word_sources}")

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
        seed=seed,
        logger=logger,
        numeric_ratio=float(syn.get("numeric_ratio", 0.12)),
        mixed_ratio=float(syn.get("mixed_ratio", 0.10)),
        progress=not args.no_progress,
    )
    logger.info(f"counts = {counts}")


if __name__ == "__main__":
    main()