"""Prepare line crops and labels for the Kanyawee poem (real fine-tuning set)."""

from __future__ import annotations

import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.detection.text_detection import OpenCVLineDetector, crop_lines
from src.utils.common import configure_stdout_utf8, get_logger, load_config

GROUND_TRUTH = [
    "Kanyawee - කන්\u200dයාවී",
    "මිනිසා මරණ තුනක් ඇති මිනිසා බලා සිටී.",
    "නිරුවත් දෑසින් බලන්න කන්\u200dයාවී",
    "ලිහිල් සළුව අනතුරේ වැටෙද්දී",
    "පයෝධර තුඩු ඉකිබිඳිද්දී සංත්\u200dරාසයෙන්",
    "අසංවාදී සුසුම් වේගේ.. රිද්\u200dමයෙන් වයන්න වීණා කන්\u200dයාවී",
    "නියඟලා මල් පාට දේදුණු දෙබෑ කර එන ගිරා කොවුවන්",
    "පියුම් කැකුළක් තුඩින් පාරා බලෙන් පුබුදන හංසයා",
    "බලන්න නිරුවත් දෑසින් ඔකඳ වී රිද්\u200dමයෙන්",
    "රසා තලය කලඑළි ගන්වා වයන්න වීණා කන්\u200dයා.වී",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop poem lines and write real labels.")
    parser.add_argument(
        "--image",
        default="data/uploads/test2.png",
        help="Full-page poem image (photo or scan)",
    )
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument(
        "--out-labels",
        default="data/real/labels/poem_kanyawee.txt",
        help="Output tab-separated labels file",
    )
    parser.add_argument(
        "--out-images",
        default="data/real/images",
        help="Directory for poem_line_*.png crops",
    )
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("prepare_poem")

    image_path = args.image if os.path.isabs(args.image) else os.path.join(ROOT, args.image)
    if not os.path.isfile(image_path):
        logger.error("Image not found: %s", image_path)
        return 1

    cfg = load_config(os.path.join(ROOT, args.config))
    det_cfg = cfg.get("detection", {})

    page_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if page_bgr is None:
        logger.error("Could not read image: %s", image_path)
        return 1
    page_gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)

    detector = OpenCVLineDetector(
        dilate_kernel=tuple(det_cfg.get("dilate_kernel", [25, 3])),
        min_line_height=int(det_cfg.get("min_line_height", 8)),
        min_line_width=int(det_cfg.get("min_line_width", 20)),
    )
    boxes = detector.detect(page_gray)
    if not boxes:
        h, w = page_gray.shape[:2]
        boxes = [(0, 0, w, h)]
        logger.warning("No lines detected; using full page as one crop.")

    line_crops = crop_lines(
        page_gray,
        boxes,
        padding_x=int(det_cfg.get("crop_padding_x", 10)),
        padding_y=int(det_cfg.get("crop_padding_y", 5)),
        min_crop_height=int(det_cfg.get("min_crop_height", 14)),
    )

    n_gt = len(GROUND_TRUTH)
    n_det = len(line_crops)
    if n_det != n_gt:
        logger.warning(
            "Detected %d lines but ground truth has %d; aligning to min count.",
            n_det,
            n_gt,
        )
    n = min(n_det, n_gt)
    line_crops = line_crops[:n]
    transcripts = GROUND_TRUTH[:n]

    out_images = args.out_images if os.path.isabs(args.out_images) else os.path.join(ROOT, args.out_images)
    os.makedirs(out_images, exist_ok=True)
    out_labels = args.out_labels if os.path.isabs(args.out_labels) else os.path.join(ROOT, args.out_labels)
    os.makedirs(os.path.dirname(out_labels), exist_ok=True)

    rel_prefix = "images"
    rows: list[str] = []
    for i, (crop, text) in enumerate(zip(line_crops, transcripts), start=1):
        name = f"poem_line_{i:03d}.png"
        out_path = os.path.join(out_images, name)
        cv2.imwrite(out_path, crop)
        rows.append(f"{rel_prefix}/{name}\t{text}")
        logger.info("Wrote %s", out_path)

    with open(out_labels, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    logger.info("Wrote %d labels to %s", len(rows), out_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
