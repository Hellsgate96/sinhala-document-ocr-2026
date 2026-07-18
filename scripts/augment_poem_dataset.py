"""Create heavily augmented copies of real poem line crops for mixed training.

Writes:
  data/real/images_aug/poem_aug_XXX_YYY.png
  data/real/labels/poem_kanyawee_aug.txt

The augmented label file is meant to be passed to training via ``--extra-labels``
alongside synthetic + page-synth data so the *general* ``crnn_best.pth`` learns
real print style without a separate overfitted finetune checkpoint.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.dataset import read_labels
from src.data.synthetic_generator import apply_augmentations
from src.utils.common import configure_stdout_utf8, get_logger

AUGMENT = {
    "rotation": True,
    "rotation_max_deg": 4.0,
    "perspective": True,
    "blur": True,
    "gaussian_noise": True,
    "brightness_contrast": True,
    "jpeg_compression": True,
    "shadow": True,
    "defocus_blur": True,
    "paper_texture": True,
    "moire": True,
    "edge_artifact": True,
    "multi_jpeg": True,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment poem line crops for mixed training.")
    parser.add_argument(
        "--labels",
        default="data/real/labels/poem_kanyawee.txt",
        help="Source path<TAB>transcript labels",
    )
    parser.add_argument("--real-dir", default="data/real")
    parser.add_argument(
        "--out-labels",
        default="data/real/labels/poem_kanyawee_aug.txt",
    )
    parser.add_argument("--out-images", default="data/real/images_aug")
    parser.add_argument("--copies", type=int, default=80, help="Augmented copies per source line")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--include-original",
        action="store_true",
        default=True,
        help="Also list the original crops in the output labels file",
    )
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("augment_poem")
    rng = random.Random(args.seed)

    labels_path = args.labels if os.path.isabs(args.labels) else os.path.join(ROOT, args.labels)
    real_dir = args.real_dir if os.path.isabs(args.real_dir) else os.path.join(ROOT, args.real_dir)
    out_images = args.out_images if os.path.isabs(args.out_images) else os.path.join(ROOT, args.out_images)
    out_labels = args.out_labels if os.path.isabs(args.out_labels) else os.path.join(ROOT, args.out_labels)

    rows = read_labels(labels_path)
    if not rows:
        logger.error("No labels in %s", labels_path)
        return 1

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(os.path.dirname(out_labels), exist_ok=True)

    out_rows: list[str] = []
    if args.include_original:
        for rel, text in rows:
            out_rows.append(f"{rel}\t{text}")

    for i, (rel, text) in enumerate(rows, start=1):
        src = rel if os.path.isabs(rel) else os.path.join(real_dir, rel)
        if not os.path.isfile(src):
            logger.error("Missing image: %s", src)
            return 1
        # Augmentors expect RGB-like arrays; convert back to L for CRNN training.
        base = Image.open(src).convert("RGB")
        bg = (255, 255, 255)
        for j in range(1, args.copies + 1):
            aug = apply_augmentations(base.copy(), AUGMENT, bg, rng).convert("L")
            name = f"poem_aug_{i:03d}_{j:03d}.png"
            aug_path = os.path.join(out_images, name)
            aug.save(aug_path)
            # Paths under real_dir: images_aug/ is a sibling of images/
            out_rows.append(f"images_aug/{name}\t{text}")
        logger.info("Line %02d: wrote %d augmented copies", i, args.copies)

    with open(out_labels, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out_rows) + "\n")
    logger.info("Wrote %d label rows to %s", len(out_rows), out_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
