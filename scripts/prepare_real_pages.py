"""Crop user_batch pages using GT JSON modes and write train/holdout labels.

GT JSON keys (per page stem):
  mode: detect | equal_bands | manual_y
  polarity: std | inv  (for equal_bands)
  n, min_contrast, bands, skip_crop_indices, holdout, lines
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.detection.text_detection import (  # noqa: E402
    binarize_for_detection,
    build_detector,
    crop_lines,
    suppress_border_structures,
)
from src.utils.common import configure_stdout_utf8, get_logger, load_config  # noqa: E402

BBox = Tuple[int, int, int, int]


def _read_page_bgr(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr
    try:
        rgb = np.array(Image.open(path).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except OSError:
        return None


def _equal_band_boxes(gray: np.ndarray, n: int, polarity: str, min_contrast: int) -> List[BBox]:
    src = gray if polarity == "std" else 255 - gray
    binary = suppress_border_structures(binarize_for_detection(src, min_contrast=min_contrast))
    h, w = binary.shape
    profile = (binary > 0).sum(axis=1).astype(np.float32)
    ys = np.flatnonzero(profile > max(1, w * 0.002))
    if ys.size == 0:
        return [(0, 0, w, h)]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    step = (y1 - y0) / max(1, n)
    boxes: List[BBox] = []
    for i in range(n):
        a = int(round(y0 + i * step))
        b = int(round(y0 + (i + 1) * step))
        strip = binary[a:b]
        cols = (strip > 0).sum(axis=0)
        xs = np.flatnonzero(cols)
        if xs.size:
            x0, x1 = max(0, int(xs[0]) - 8), min(w, int(xs[-1]) + 8)
        else:
            x0, x1 = 0, w
        boxes.append((x0, a, x1 - x0, max(1, b - a)))
    return boxes


def _manual_y_boxes(gray: np.ndarray, bands: List[Dict[str, int]]) -> List[BBox]:
    h, w = gray.shape[:2]
    boxes: List[BBox] = []
    for band in bands:
        y0, y1 = int(band["y0"]), int(band["y1"])
        x0 = int(band.get("x0", 0))
        x1 = int(band.get("x1", w))
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, x0), min(w, x1)
        boxes.append((x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop real pages using GT JSON.")
    parser.add_argument("--pages-dir", default="data/real/pages/user_batch1")
    parser.add_argument("--gt-json", default="data/real/labels/user_batch1_gt.json")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--out-images", default="data/real/images")
    parser.add_argument("--out-labels", default="data/real/labels/user_batch1.txt")
    parser.add_argument("--out-holdout", default="data/real/labels/user_batch1_holdout.txt")
    parser.add_argument("--inventory", default="")
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("prepare_real_pages")

    pages_dir = args.pages_dir if os.path.isabs(args.pages_dir) else os.path.join(ROOT, args.pages_dir)
    gt_path = args.gt_json if os.path.isabs(args.gt_json) else os.path.join(ROOT, args.gt_json)
    out_images = args.out_images if os.path.isabs(args.out_images) else os.path.join(ROOT, args.out_images)
    out_labels = args.out_labels if os.path.isabs(args.out_labels) else os.path.join(ROOT, args.out_labels)
    out_holdout = args.out_holdout if os.path.isabs(args.out_holdout) else os.path.join(ROOT, args.out_holdout)
    inventory_path = args.inventory or os.path.join(pages_dir, "inventory.json")
    if not os.path.isabs(inventory_path):
        inventory_path = os.path.join(ROOT, inventory_path)

    with open(gt_path, encoding="utf-8") as f:
        gt_all: Dict[str, Any] = json.load(f)

    cfg = load_config(os.path.join(ROOT, args.config))
    det_cfg = cfg.get("detection", {})
    detector = build_detector(det_cfg)

    page_files = sorted(
        f for f in os.listdir(pages_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(os.path.dirname(out_labels), exist_ok=True)

    # Remove prior user crops
    for name in os.listdir(out_images):
        if name.startswith("user_p") and name.endswith(".png"):
            os.remove(os.path.join(out_images, name))

    inventory: Dict[str, Any] = {"pages": [], "meta": gt_all.get("_meta", {})}
    train_rows: List[str] = []
    hold_rows: List[str] = []
    page_index = 0

    for fname in page_files:
        stem = os.path.splitext(fname)[0]
        spec = gt_all.get(stem)
        if not isinstance(spec, dict) or "lines" not in spec:
            logger.warning("No GT for %s — skipping", fname)
            continue
        page_index += 1
        prefix = f"user_p{page_index:02d}"
        path = os.path.join(pages_dir, fname)
        page_bgr = _read_page_bgr(path)
        if page_bgr is None:
            logger.error("Could not read %s", path)
            return 1
        page_gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)

        mode = str(spec.get("mode", "detect"))
        lines_gt: List[str] = list(spec["lines"])
        holdout = bool(spec.get("holdout", False))

        if mode == "equal_bands":
            boxes = _equal_band_boxes(
                page_gray,
                n=int(spec.get("n", len(lines_gt))),
                polarity=str(spec.get("polarity", "std")),
                min_contrast=int(spec.get("min_contrast", 15)),
            )
        elif mode == "manual_y":
            boxes = _manual_y_boxes(page_gray, list(spec.get("bands", [])))
        else:
            boxes = detector.detect(page_gray)
            if not boxes:
                h, w = page_gray.shape[:2]
                boxes = [(0, 0, w, h)]

        crops = crop_lines(
            page_gray,
            boxes,
            padding_x=int(det_cfg.get("crop_padding_x", 10)),
            padding_y=int(det_cfg.get("crop_padding_y", 5)),
            min_crop_height=int(det_cfg.get("min_crop_height", 14)),
        )

        skip = set(int(i) for i in spec.get("skip_crop_indices", []))
        kept: List[np.ndarray] = []
        for i, crop in enumerate(crops):
            if i in skip:
                continue
            kept.append(crop)

        n = min(len(kept), len(lines_gt))
        if len(kept) != len(lines_gt):
            logger.warning(
                "%s: crops=%d gt=%d (after skip) — using first %d",
                fname,
                len(kept),
                len(lines_gt),
                n,
            )
        kept = kept[:n]
        transcripts = lines_gt[:n]

        page_entry: Dict[str, Any] = {
            "file": fname,
            "stem": stem,
            "prefix": prefix,
            "mode": mode,
            "holdout": holdout,
            "detected_lines": len(crops),
            "labeled_lines": n,
            "crops": [],
        }

        for i, (crop, text) in enumerate(zip(kept, transcripts), start=1):
            name = f"{prefix}_line_{i:03d}.png"
            cv2.imwrite(os.path.join(out_images, name), crop)
            rel = f"images/{name}"
            row = f"{rel}\t{text}"
            if holdout:
                hold_rows.append(row)
            else:
                train_rows.append(row)
            page_entry["crops"].append({"file": name, "transcript": text})
            logger.info("%s %s", name, text[:40])

        inventory["pages"].append(page_entry)

    with open(out_labels, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(train_rows) + ("\n" if train_rows else ""))
    with open(out_holdout, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(hold_rows) + ("\n" if hold_rows else ""))
    with open(inventory_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)

    # Markdown inventory for the user
    md_path = os.path.join(pages_dir, "INVENTORY.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# user_batch1 page inventory\n\n")
        f.write("| Page | Prefix | Lines | Holdout | Mode |\n|---|---|---:|---|---|\n")
        for p in inventory["pages"]:
            f.write(
                f"| {p['file']} | {p['prefix']} | {p['labeled_lines']} | "
                f"{'yes' if p['holdout'] else 'no'} | {p['mode']} |\n"
            )
        f.write(f"\nTrain lines: **{len(train_rows)}**  \n")
        f.write(f"Holdout lines: **{len(hold_rows)}**\n")

    logger.info("Train labels: %d -> %s", len(train_rows), out_labels)
    logger.info("Holdout labels: %d -> %s", len(hold_rows), out_holdout)
    logger.info("Inventory -> %s", inventory_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
