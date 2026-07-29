# -*- coding: utf-8 -*-
"""Turn a ``scripts/run_eval_suite.py`` JSON report into an error analysis:
every mis-read line (worst first) plus a ranked table of character-level
confusions across the whole set.

The confusion table is what drives each training round - e.g. the jul28 run
showed 9x ``ේ -> ී`` together with 7x spurious ``ෙ`` insertions, which
identified the pre-base kombuva being attached to the wrong consonant on
low-resolution lines rather than a generic "small text is blurry" problem.

Usage:
    python scripts/run_eval_suite.py --checkpoint models/crnn_best.pth \
        --out data/debug/suite.json
    python scripts/report_errors.py data/debug/suite.json --set print_photos
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Tuple


def _rows(res) -> List[Tuple[str, str, str, float]]:
    out: List[Tuple[str, str, str, float]] = []
    if res["kind"] == "page":
        for page in res["pages"]:
            for line in page["per_line"]:
                out.append((f"{page['image']}#{line['line']:02d}", line["ref"], line["hyp"], line["cer"]))
    else:
        for line in res["lines"]:
            out.append((line["image"], line["ref"], line["hyp"], -1.0))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("report")
    p.add_argument("--set", action="append", dest="sets", default=None,
                   help="restrict to named set(s) from the report")
    p.add_argument("--top", type=int, default=40, help="how many confusions to list")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    print(f"checkpoint={report.get('checkpoint')} decode={report.get('decode')} "
          f"pad_to_height={report.get('pad_to_height')}")
    confusions: Counter = Counter()
    for name, res in report["sets"].items():
        if args.sets and name not in args.sets:
            continue
        print("=" * 72)
        print(f"{name}  n={res['n']}  CER={res['corpus_cer']}")
        for tag, ref, hyp, line_cer in sorted(_rows(res), key=lambda r: -r[3]):
            if ref == hyp:
                continue
            shown = "" if line_cer < 0 else f"  cer={line_cer:.3f}"
            print(f"-- {tag}{shown}")
            print(f"   GT : {ref}")
            print(f"   PR : {hyp}")
            for op, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp).get_opcodes():
                if op != "equal":
                    confusions[(op, ref[i1:i2], hyp[j1:j2])] += 1

    print("=" * 72)
    print(f"top {args.top} character confusions (GT -> prediction):")
    for (op, a, b), n in confusions.most_common(args.top):
        print(f"  {n:>3}  {op:<9} {a!r} -> {b!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
