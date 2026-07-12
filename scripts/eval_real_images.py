"""Evaluate the trained CRNN on real document images: detect lines with the
configured detector (default ProjectionLineDetector) and recognize each line
with a given checkpoint. Optionally computes CER against a ground-truth
labels file (path<TAB>transcript, matched by order).

Usage:
    python scripts/eval_real_images.py --image data/uploads/test2.png \
        --checkpoint models/crnn_best.pth --gt data/real/labels/poem_kanyawee.txt \
        --debug-dir data/debug/v2_poem

    python scripts/eval_real_images.py --image data/uploads/real_capture_20260706T054800Z.jpg \
        --checkpoint models/crnn_best.pth --debug-dir data/debug/v2_card

See also ``scripts/run_realistic_eval.py`` for batch/corpus-level evaluation
over a whole directory of (image, ground-truth) pages, and
``scripts/build_adversarial_pages.py`` for the hand-built acceptance-test
pages. Both share the detect+recognize path in ``src/evaluation/pipeline_eval.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.charset import Charset
from src.detection.text_detection import build_detector
from src.evaluation.pipeline_eval import run_pipeline_on_image_path, save_debug, score_against_gt
from src.recognition.model import build_crnn
from src.recognition.inference import inference_options_from_config
from src.utils.common import configure_stdout_utf8, get_device, get_logger, load_checkpoint, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default="models/crnn_best.pth")
    p.add_argument("--charset", default="models/charset.json")
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--gt", default=None, help="optional path<TAB>transcript ground-truth file")
    p.add_argument("--debug-dir", default=None)
    args = p.parse_args()

    configure_stdout_utf8()
    logger = get_logger("eval_real")
    cfg = load_config(args.config)
    device = get_device(cfg["train"].get("device", "auto"))

    charset = Charset.load(args.charset)
    model = build_crnn(charset.num_classes, cfg.get("model"), in_channels=cfg["image"]["channels"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    inf_opts = inference_options_from_config(cfg)
    det_cfg = dict(cfg.get("detection", {}))
    detector = build_detector(det_cfg)

    image_path = args.image if os.path.isabs(args.image) else os.path.join(ROOT, args.image)
    result = run_pipeline_on_image_path(model, charset, image_path, detector, inf_opts, det_cfg, device)
    boxes, texts = result["boxes"], result["texts"]
    display_texts = result.get("display_texts", texts)
    logger.info("detected %d line box(es) with method=%s", len(boxes), det_cfg.get("method", "projection"))
    for i, text in enumerate(display_texts, start=1):
        print(f"Line {i:02d}: {text}")

    payload = {
        "source_image": image_path,
        "checkpoint": os.path.basename(args.checkpoint),
        "detection_method": det_cfg.get("method", "projection"),
        "num_lines": len(texts),
        "lines": [{"line": i, "text": t} for i, t in enumerate(texts, start=1)],
    }

    if args.gt:
        gt_path = args.gt if os.path.isabs(args.gt) else os.path.join(ROOT, args.gt)
        gt_rows = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    gt_rows.append("\t".join(parts[1:]))
        scored = score_against_gt(gt_rows, texts)
        for row in scored["per_line"]:
            print(f"  GT {row['line']:02d}: {row['ref']}")
            print(f"  CER {row['line']:02d}: {row['cer']:.4f}")
        print(f"Aligned lines: {scored['num_aligned']} (gt={scored['num_gt']}, detected={scored['num_pred']})")
        print(f"Overall CER (order-aligned): {scored['corpus_cer']:.4f}")
        payload["gt_aligned_cer"] = scored["corpus_cer"]
        payload["gt_num_lines"] = scored["num_gt"]
        payload["per_line_cer"] = scored["per_line"]

    if args.debug_dir:
        debug_dir = args.debug_dir if os.path.isabs(args.debug_dir) else os.path.join(ROOT, args.debug_dir)
        save_debug(debug_dir, result["bgr"], boxes, result["crops"], texts, extra=payload)
        logger.info("Saved debug outputs to %s", debug_dir)


if __name__ == "__main__":
    main()
