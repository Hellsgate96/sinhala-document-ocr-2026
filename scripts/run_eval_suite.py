# -*- coding: utf-8 -*-
"""Run the whole honest evaluation suite for one checkpoint and print a table.

Loads the model once and scores every set we track between training rounds:

* page sets (end-to-end detect + recognize, order-aligned corpus CER)
    - ``data/eval_pages``            synthetic realistic pages
    - ``data/eval_real/adversarial`` adversarial synthetic pages
    - ``data/eval_real/print_photos`` real photos, never trained on
* line-crop label sets (recognition only)
    - held-out user/web batches, in-train poem lines

Usage:
    python scripts/run_eval_suite.py --checkpoint models/crnn_best.pth \
        --name before --out data/debug/suite_before.json

    # override a single inference option to A/B it without editing configs:
    python scripts/run_eval_suite.py --checkpoint models/crnn_best.pth \
        --pad-to-height true --name before_padded
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.charset import Charset
from src.data.dataset import read_labels
from src.detection.text_detection import build_detector
from src.evaluation.metrics import corpus_cer, edit_distance
from src.evaluation.pipeline_eval import run_pipeline_on_image_path, score_against_gt
from src.recognition.inference import decode_kwargs_from_options, inference_options_from_config
from src.recognition.model import build_crnn
from src.recognition.predict import predict_image
from src.utils.common import configure_stdout_utf8, get_device, load_checkpoint, load_config

# name -> directory of ``*.png``/``*.jpg`` + matching ``*.gt.txt``
PAGE_SETS = {
    "eval_pages": "data/eval_pages",
    "adversarial": "data/eval_real/adversarial",
    "print_photos": "data/eval_real/print_photos",
}

# name -> (labels file, base dir for the relative image paths)
LABEL_SETS = {
    "user_batch1_holdout": ("data/real/labels/user_batch1_holdout.txt", "data/real"),
    "web_batch1_holdout": ("data/real/labels/web_batch1_holdout.txt", "data/real"),
    "poem_in_train": ("data/real/labels/poem_kanyawee.txt", "data/real"),
}

# How much each number is worth as evidence of generalisation. Verified with
# scripts/check_holdout_leakage.py - "user_batch1_holdout" keeps its historical
# name but all 41 of its transcripts were folded into training in a later round,
# so it is reported as in-train and must not be quoted as a holdout result.
SET_STATUS = {
    "print_photos": "HELD OUT (real photos, never trained on)",
    "eval_pages": "held out (synthetic pages, separate generation run)",
    "adversarial": "held out (synthetic, hand-built acceptance pages)",
    "web_batch1_holdout": "partly in train (6 of 14 transcripts seen)",
    "user_batch1_holdout": "IN TRAIN (all 41 transcripts seen) - reference only",
    "poem_in_train": "IN TRAIN - reference only",
}


def _read_gt(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _score_page_set(model, charset, inf_opts, det_cfg, detector, device, directory: Path) -> Dict[str, Any]:
    paths = sorted(
        p for p in glob.glob(str(directory / "*.png")) + glob.glob(str(directory / "*.jpg"))
    )
    edits = 0
    chars = 0
    pages = []
    for img_path in paths:
        gt_path = Path(os.path.splitext(img_path)[0] + ".gt.txt")
        if not gt_path.is_file():
            continue
        result = run_pipeline_on_image_path(
            model, charset, img_path, detector, inf_opts, det_cfg, device,
        )
        scored = score_against_gt(_read_gt(gt_path), result["texts"])
        ref = "".join(l["ref"] for l in scored["per_line"])
        hyp = "".join(l["hyp"] for l in scored["per_line"])
        edits += edit_distance(list(ref), list(hyp))
        chars += len(ref)
        pages.append({
            "image": os.path.basename(img_path),
            "num_gt": scored["num_gt"],
            "num_pred": scored["num_pred"],
            "corpus_cer": scored["corpus_cer"],
            "corpus_wer": scored["corpus_wer"],
            "per_line": scored["per_line"],
        })
    return {
        "kind": "page",
        "n": len(pages),
        "corpus_cer": (edits / chars) if chars else None,
        "pages": pages,
    }


def _score_label_set(model, charset, inf_opts, device, labels: Path, base: Path) -> Dict[str, Any]:
    rows = read_labels(str(labels))
    preds, gts, worst = [], [], []
    for rel, gt in rows:
        path = Path(rel)
        if not path.is_file():
            path = base / rel
        if not path.is_file():
            continue
        pred = predict_image(
            model, charset, str(path),
            inf_opts["height"], inf_opts["max_width"], inf_opts["channels"], device,
            auto_invert=inf_opts["auto_invert"], denoise=inf_opts["denoise"],
            min_model_width=inf_opts.get("min_model_width", 0),
            pad_to_height=inf_opts.get("pad_to_height", True),
            warn_garbage=False,
            **decode_kwargs_from_options(inf_opts),
        )
        preds.append(pred)
        gts.append(gt)
        worst.append({"image": rel, "ref": gt, "hyp": pred})
    return {
        "kind": "lines",
        "n": len(gts),
        "corpus_cer": corpus_cer(gts, preds) if gts else None,
        "lines": worst,
    }


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/crnn_best.pth")
    p.add_argument("--charset", default="models/charset.json")
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--name", default="")
    p.add_argument("--out", default="")
    p.add_argument("--pad-to-height", default=None, help="override inference.pad_to_height")
    p.add_argument("--decode", default=None, help="override inference.decode (greedy|beam|beam_lm)")
    p.add_argument("--lm-weight", type=float, default=None)
    p.add_argument("--insertion-bonus", type=float, default=None)
    p.add_argument("--beam-width", type=int, default=None)
    p.add_argument("--only", action="append", default=None, help="restrict to named set(s)")
    p.add_argument("--det", action="append", default=None,
                   help="override a detection config key, e.g. --det crop_padding_y=3")
    args = p.parse_args()

    configure_stdout_utf8()
    cfg = load_config(str(ROOT / args.config))
    device = get_device(cfg["train"].get("device", "auto"))
    inf_opts = inference_options_from_config(cfg)
    if args.pad_to_height is not None:
        inf_opts["pad_to_height"] = _as_bool(args.pad_to_height)
    if args.decode is not None:
        inf_opts["decode"] = args.decode
    if args.lm_weight is not None:
        inf_opts["lm_weight"] = args.lm_weight
    if args.insertion_bonus is not None:
        inf_opts["insertion_bonus"] = args.insertion_bonus
    if args.beam_width is not None:
        inf_opts["beam_width"] = args.beam_width

    charset = Charset.load(str(ROOT / args.charset))
    model = build_crnn(charset.num_classes, cfg.get("model"), in_channels=cfg["image"]["channels"]).to(device)
    ckpt = args.checkpoint if Path(args.checkpoint).is_absolute() else str(ROOT / args.checkpoint)
    load_checkpoint(ckpt, model, map_location=str(device))
    model.eval()

    det_cfg = dict(cfg.get("detection", {}))
    for override in args.det or []:
        key, _, value = override.partition("=")
        try:
            det_cfg[key] = int(value)
        except ValueError:
            det_cfg[key] = _as_bool(value) if value.lower() in {"true", "false"} else value
    detector = build_detector(det_cfg)

    wanted = set(args.only) if args.only else None
    report: Dict[str, Any] = {
        "checkpoint": os.path.basename(ckpt),
        "name": args.name,
        "pad_to_height": inf_opts["pad_to_height"],
        "decode": inf_opts.get("decode"),
        "lm_weight": inf_opts.get("lm_weight"),
        "insertion_bonus": inf_opts.get("insertion_bonus"),
        "beam_width": inf_opts.get("beam_width"),
        "sets": {},
    }
    for name, rel_dir in PAGE_SETS.items():
        if wanted and name not in wanted:
            continue
        directory = ROOT / rel_dir
        if not directory.is_dir():
            continue
        report["sets"][name] = _score_page_set(
            model, charset, inf_opts, det_cfg, detector, device, directory,
        )
    for name, (rel_labels, rel_base) in LABEL_SETS.items():
        if wanted and name not in wanted:
            continue
        labels = ROOT / rel_labels
        if not labels.is_file():
            continue
        report["sets"][name] = _score_label_set(
            model, charset, inf_opts, device, labels, ROOT / rel_base,
        )

    print(f"\n=== {args.name or os.path.basename(ckpt)} "
          f"(pad_to_height={inf_opts['pad_to_height']}, decode={inf_opts.get('decode')}, "
          f"lm_weight={inf_opts.get('lm_weight')}, bonus={inf_opts.get('insertion_bonus')}) ===")
    print(f"{'set':<22} {'n':>4}  {'CER':>8}  status")
    for name, res in report["sets"].items():
        cer_val = res["corpus_cer"]
        res["status"] = SET_STATUS.get(name, "")
        shown = f"{cer_val:>8.4f}" if cer_val is not None else f"{'n/a':>8}"
        print(f"{name:<22} {res['n']:>4}  {shown}  {res['status']}")
    # Per-page detail for the real photos: that is where regressions hide.
    photos = report["sets"].get("print_photos")
    if photos:
        for page in photos["pages"]:
            print(f"  {page['image']:<26} gt={page['num_gt']:>3} det={page['num_pred']:>3} "
                  f"cer={page['corpus_cer']:.4f}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
