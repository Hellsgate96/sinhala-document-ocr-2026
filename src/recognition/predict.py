"""Run inference with a trained CRNN: greedy CTC decoding on image(s).

Usage:
    python -m src.recognition.predict --checkpoint models/crnn_best.pth \
        --charset models/charset.json --image path/to/line.png
    python -m src.recognition.predict --checkpoint models/crnn_best.pth \
        --charset models/charset.json --folder data/synthetic_sample/images
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from src.charset import Charset
from src.data.dataset import resize_keep_height
from src.recognition.model import build_crnn
from src.utils.common import (configure_stdout_utf8, get_device, load_checkpoint,
                              load_config)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def image_to_tensor(path: str, height: int, max_width: int, channels: int) -> torch.Tensor:
    """Load and normalize a single image to a (1, C, H, W) tensor."""
    mode = "L" if channels == 1 else "RGB"
    img = Image.open(path).convert(mode)
    img = resize_keep_height(img, height, max_width)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if channels == 1:
        arr = arr[None, :, :]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0)


@torch.no_grad()
def predict_image(model, charset: Charset, path: str, height: int,
                  max_width: int, channels: int, device) -> str:
    """Predict the transcript of one line image."""
    tensor = image_to_tensor(path, height, max_width, channels).to(device)
    log_probs = model(tensor)                  # (T, 1, C)
    indices = log_probs.argmax(2).squeeze(1)   # (T,)
    return charset.ctc_greedy_decode(indices.tolist())


def predict_folder(model, charset, folder, height, max_width, channels, device) -> List[Tuple[str, str]]:
    results = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if os.path.splitext(path)[1].lower() in IMG_EXTS:
            results.append((path, predict_image(model, charset, path, height,
                                                max_width, channels, device)))
    return results


def main():
    parser = argparse.ArgumentParser(description="CRNN inference (CTC greedy decode).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--charset", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", help="path to a single line image")
    parser.add_argument("--folder", help="path to a folder of line images")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    configure_stdout_utf8()
    if not args.image and not args.folder:
        parser.error("provide --image or --folder")

    cfg = load_config(args.config)
    device = get_device(args.device)
    charset = Charset.load(args.charset)
    model = build_crnn(charset.num_classes, cfg.get("model"),
                       in_channels=cfg["image"]["channels"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    h, mw, ch = cfg["image"]["height"], cfg["image"]["max_width"], cfg["image"]["channels"]
    if args.image:
        text = predict_image(model, charset, args.image, h, mw, ch, device)
        print(f"{args.image}\t{text}")
    if args.folder:
        for path, text in predict_folder(model, charset, args.folder, h, mw, ch, device):
            print(f"{path}\t{text}")


if __name__ == "__main__":
    main()