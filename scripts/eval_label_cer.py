# -*- coding: utf-8 -*-
"""Evaluate a checkpoint on one or more label files (line-crop CER)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.charset import Charset
from src.data.dataset import read_labels
from src.evaluation.metrics import cer, corpus_cer
from src.recognition.inference import inference_options_from_config
from src.recognition.model import build_crnn
from src.recognition.predict import predict_image
from src.utils.common import get_device, load_checkpoint, load_config


def eval_one(model, charset, opts, device, labels_path: Path, base_dir: Path) -> dict:
    rows = read_labels(str(labels_path))
    preds, gts = [], []
    for rel, gt in rows:
        path = Path(rel)
        if not path.is_file():
            path = base_dir / rel
        if not path.is_file():
            path = ROOT / "data" / "real" / rel
        if not path.is_file():
            continue
        pred = predict_image(
            model,
            charset,
            str(path),
            opts["height"],
            opts["max_width"],
            opts["channels"],
            device,
            auto_invert=opts["auto_invert"],
            denoise=opts["denoise"],
            min_model_width=opts.get("min_model_width", 0),
            warn_garbage=False,
        )
        preds.append(pred)
        gts.append(gt)
    return {
        "labels": str(labels_path),
        "n": len(gts),
        "corpus_cer": corpus_cer(gts, preds) if gts else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--labels", action="append", required=True)
    p.add_argument("--base-dir", default="data/real")
    p.add_argument("--out", default="")
    p.add_argument("--name", default="")
    args = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(str(ROOT / args.config))
    device = get_device("auto")
    opts = inference_options_from_config(cfg)
    charset = Charset.load(str(ROOT / "models" / "charset.json"))
    model = build_crnn(charset.num_classes, cfg.get("model"), in_channels=cfg["image"]["channels"]).to(device)
    load_checkpoint(args.checkpoint if Path(args.checkpoint).is_absolute() else str(ROOT / args.checkpoint), model, map_location=str(device))
    model.eval()
    base = Path(args.base_dir)
    if not base.is_absolute():
        base = ROOT / base

    results = {"checkpoint": args.checkpoint, "name": args.name, "sets": []}
    for lab in args.labels:
        lp = Path(lab)
        if not lp.is_absolute():
            lp = ROOT / lp
        r = eval_one(model, charset, opts, device, lp, base)
        results["sets"].append(r)
        print(f"{args.name or Path(args.checkpoint).name} | {lp.name} | n={r['n']} | CER={r['corpus_cer']:.4f}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
