"""Download Sinhala OCR Acts pages (CC-BY-4.0) and extract labeled line crops.

Dataset: avishadilhara/sinhala-ocr-lk-acts-1010 (Hugging Face, CC-BY-4.0).
Page-level GT is split on newlines; crops are kept only when the projection
detector line count matches the GT line count (detector-in-the-loop, no guesswork).

Outputs:
  data/real/pages/web_batch1/acts_*.png   (optional page copies)
  data/real/images/acts_*.png             (line crops)
  data/real/labels/web_batch1_acts.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.detection.text_detection import build_detector, crop_lines  # noqa: E402
from src.utils.common import configure_stdout_utf8, get_logger, load_config  # noqa: E402


def _split_gt(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").replace("\r\n", "\n").split("\n")]
    return [ln for ln in lines if ln]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HF Sinhala Acts OCR pages + line crops.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    parser.add_argument("--max-pages", type=int, default=250)
    parser.add_argument("--save-pages", action="store_true", help="Also save full page PNGs")
    parser.add_argument("--out-labels", default="data/real/labels/web_batch1_acts.txt")
    parser.add_argument("--out-images", default="data/real/images")
    parser.add_argument("--pages-dir", default="data/real/pages/web_batch1")
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("download_hf_acts")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets huggingface_hub")
        return 1

    cfg = load_config(os.path.join(ROOT, args.config))
    detector = build_detector(cfg.get("detection", {}))

    out_images = args.out_images if os.path.isabs(args.out_images) else os.path.join(ROOT, args.out_images)
    pages_dir = args.pages_dir if os.path.isabs(args.pages_dir) else os.path.join(ROOT, args.pages_dir)
    out_labels = args.out_labels if os.path.isabs(args.out_labels) else os.path.join(ROOT, args.out_labels)
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_labels), exist_ok=True)

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]
    # Streaming avoids downloading the full ~GB archive before first page.
    logger.info("streaming avishadilhara/sinhala-ocr-lk-acts-1010 ...")
    ds = load_dataset("avishadilhara/sinhala-ocr-lk-acts-1010", streaming=True)

    labels: List[Tuple[str, str]] = []
    kept_pages = 0
    skipped = 0
    page_idx = 0
    scanned = 0
    max_scan = max(args.max_pages * 40, 500)  # detector match rate can be low

    for split in splits:
        if split not in ds:
            logger.warning(f"split {split} missing; available={list(ds.keys())}")
            continue
        for row in ds[split]:
            if kept_pages >= args.max_pages or scanned >= max_scan:
                break
            scanned += 1
            page_idx += 1
            gt_lines = _split_gt(row.get("text") or "")
            if len(gt_lines) < 2:
                skipped += 1
                continue
            pil = row["image"]
            if not isinstance(pil, Image.Image):
                pil = Image.fromarray(np.array(pil))
            rgb = np.array(pil.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            boxes = detector.detect(bgr)
            if len(boxes) != len(gt_lines):
                skipped += 1
                continue
            det_cfg = cfg.get("detection", {}) or {}
            crops = crop_lines(
                bgr,
                boxes,
                padding_x=int(det_cfg.get("crop_padding_x", 10)),
                padding_y=int(det_cfg.get("crop_padding_y", 5)),
                min_crop_height=int(det_cfg.get("min_crop_height", 14)),
            )
            if len(crops) != len(gt_lines):
                skipped += 1
                continue

            stem = f"acts_{split}_{page_idx:04d}"
            if args.save_pages:
                page_path = os.path.join(pages_dir, f"{stem}.png")
                Image.fromarray(rgb).save(page_path)

            for li, (crop, gt) in enumerate(zip(crops, gt_lines), start=1):
                name = f"{stem}_line_{li:03d}.png"
                abs_path = os.path.join(out_images, name)
                if crop.ndim == 2:
                    Image.fromarray(crop).save(abs_path)
                else:
                    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(abs_path)
                labels.append((f"images/{name}", gt))
            kept_pages += 1
            if kept_pages % 10 == 0:
                logger.info(
                    f"kept_pages={kept_pages} lines={len(labels)} skipped={skipped} scanned={scanned}"
                )

        if kept_pages >= args.max_pages or scanned >= max_scan:
            break

    with open(out_labels, "w", encoding="utf-8") as f:
        for rel, gt in labels:
            f.write(f"{rel}\t{gt}\n")
    logger.info(
        f"done: pages_kept={kept_pages} lines={len(labels)} skipped={skipped} -> {out_labels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
