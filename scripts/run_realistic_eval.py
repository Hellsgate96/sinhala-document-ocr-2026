"""CLI: batch/corpus-level realistic evaluation over a directory of full page
images with ground-truth transcripts (``page_XXX.png`` + ``page_XXX.gt.txt``,
one line per row, top-to-bottom) - e.g. the output of
``scripts/build_eval_pages.py`` or ``scripts/build_adversarial_pages.py``.

Reports, per page and aggregated over the whole set: detected line count vs
ground-truth line count, order-aligned corpus CER/WER. This is the "did the
domain-gap fix actually work" signal (line-crop CER on synthetic val data is
NOT sufficient - see README).

Usage:
    python scripts/run_realistic_eval.py --images-dir data/eval_pages \
        --checkpoint models/crnn_best.pth --out data/debug/eval_pages_report.json

    # before/after comparison:
    python scripts/run_realistic_eval.py --images-dir data/eval_real/adversarial \
        --checkpoint models/crnn_best_pre_domaingap.pth --out data/debug/adv_before.json
    python scripts/run_realistic_eval.py --images-dir data/eval_real/adversarial \
        --checkpoint models/crnn_best.pth --out data/debug/adv_after.json
"""
from __future__ import annotations

import argparse
import glob
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


def _read_gt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--checkpoint", default="models/crnn_best.pth")
    p.add_argument("--charset", default="models/charset.json")
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--out", default=None, help="optional path to write a JSON report")
    p.add_argument("--debug-dir", default=None, help="optional dir to dump box overlays/crops per page")
    args = p.parse_args()

    configure_stdout_utf8()
    logger = get_logger("run_realistic_eval")
    cfg = load_config(args.config)
    device = get_device(cfg["train"].get("device", "auto"))

    charset = Charset.load(args.charset)
    model = build_crnn(charset.num_classes, cfg.get("model"), in_channels=cfg["image"]["channels"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    inf_opts = inference_options_from_config(cfg)
    det_cfg = dict(cfg.get("detection", {}))
    detector = build_detector(det_cfg)

    images_dir = args.images_dir if os.path.isabs(args.images_dir) else os.path.join(ROOT, args.images_dir)
    image_paths = sorted(
        p for p in glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
        if not p.endswith(".gt.txt")
    )

    pages = []
    total_edits = 0
    total_chars = 0
    total_gt_lines = 0
    total_pred_lines = 0
    for img_path in image_paths:
        gt_path = os.path.splitext(img_path)[0] + ".gt.txt"
        result = run_pipeline_on_image_path(model, charset, img_path, detector, inf_opts, det_cfg, device)
        texts = result["texts"]
        display_texts = result.get("display_texts", texts)
        entry = {"image": os.path.basename(img_path), "num_detected": len(texts), "lines": display_texts}
        if os.path.isfile(gt_path):
            gt_lines = _read_gt(gt_path)
            scored = score_against_gt(gt_lines, texts)
            entry.update(scored)
            from src.evaluation.metrics import edit_distance

            ref_chars = "".join(scored["per_line"][i]["ref"] for i in range(scored["num_aligned"]))
            hyp_chars = "".join(scored["per_line"][i]["hyp"] for i in range(scored["num_aligned"]))
            total_edits += edit_distance(list(ref_chars), list(hyp_chars))
            total_chars += len(ref_chars)
            total_gt_lines += scored["num_gt"]
            total_pred_lines += scored["num_pred"]
            print(f"{os.path.basename(img_path)}: gt_lines={scored['num_gt']} detected={scored['num_pred']} "
                 f"corpus_cer={scored['corpus_cer']:.4f} corpus_wer={scored['corpus_wer']:.4f}")
        else:
            print(f"{os.path.basename(img_path)}: detected={len(texts)} lines (no GT file)")
        pages.append(entry)
        if args.debug_dir:
            page_debug = os.path.join(args.debug_dir, os.path.splitext(os.path.basename(img_path))[0])
            save_debug(page_debug, result["bgr"], result["boxes"], result["crops"], texts)

    report = {"images_dir": images_dir, "checkpoint": os.path.basename(args.checkpoint), "pages": pages}
    if total_chars:
        overall_cer = total_edits / total_chars
        report["overall_corpus_cer"] = overall_cer
        report["total_gt_lines"] = total_gt_lines
        report["total_detected_lines"] = total_pred_lines
        print(f"\n=== OVERALL corpus CER = {overall_cer:.4f} over {len(pages)} page(s), "
             f"{total_gt_lines} gt lines / {total_pred_lines} detected lines ===")

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        logger.info(f"wrote report to {out_path}")


if __name__ == "__main__":
    main()
