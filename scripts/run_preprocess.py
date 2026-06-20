"""CLI wrapper: preprocess a folder of document images.

Example:
    python scripts/run_preprocess.py --input data/raw --output data/preprocessed
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocessing.preprocess import process_folder  # noqa: E402
from src.utils.common import configure_stdout_utf8  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Preprocess a folder of documents.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    configure_stdout_utf8()
    process_folder(args.input, args.output)


if __name__ == "__main__":
    main()