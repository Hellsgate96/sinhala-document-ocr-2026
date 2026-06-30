"""Quick local checks for prepare_line_for_recognition (CPU, no training)."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.common import configure_stdout_utf8, get_device, load_checkpoint, load_config
from src.charset import Charset
from src.recognition.inference import prepare_line_for_recognition, should_invert_polarity
from src.recognition.model import build_crnn
from src.recognition.predict import predict_image


def test_dark_bg_inverts() -> None:
    img = np.full((40, 200, 3), 60, dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert should_invert_polarity(gray)
    prep = np.asarray(prepare_line_for_recognition(gray, auto_invert=True))
    assert prep.mean() > 128


def test_synthetic_predict() -> None:
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    ckpt = os.path.join(ROOT, cfg["paths"]["models_dir"], "crnn_best.pth")
    charset_path = os.path.join(ROOT, cfg["paths"]["charset_path"])
    labels_path = os.path.join(ROOT, "data", "synthetic_sample", "test_labels.txt")
    if not (os.path.isfile(ckpt) and os.path.isfile(charset_path) and os.path.isfile(labels_path)):
        print("[skip] missing checkpoint or synthetic sample")
        return
    first_line = open(labels_path, encoding="utf-8").readline().strip()
    rel, gt = first_line.split("\t", 1)
    img_path = os.path.join(ROOT, "data", "synthetic_sample", rel.replace("/", os.sep))
    device = get_device("cpu")
    charset = Charset.load(charset_path)
    model = build_crnn(charset.num_classes, cfg.get("model"), in_channels=cfg["image"]["channels"])
    load_checkpoint(ckpt, model, map_location="cpu")
    model.eval()
    h, mw, ch = cfg["image"]["height"], cfg["image"]["max_width"], cfg["image"]["channels"]
    pred = predict_image(model, charset, img_path, h, mw, ch, device)
    print("GT:", gt)
    print("Pred:", pred)
    if not pred.strip():
        print("[warn] empty prediction (local checkpoint may be undertrained)")




def test_resize_height_48_debug_image() -> None:
    """Resize at config height should yield readable grayscale (save debug PNG)."""
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    h = int(cfg.get("inference", {}).get("image_height", cfg["image"]["height"]))
    img = np.full((80, 400), 245, dtype=np.uint8)
    cv2.putText(img, "Sinhala OCR", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    prep = np.asarray(
        prepare_line_for_recognition(
            img,
            height=h,
            auto_invert=False,
            min_model_width=int(cfg.get("inference", {}).get("min_model_width", 0)),
            pad_to_height=bool(cfg.get("inference", {}).get("pad_to_height", True)),
        )
    )
    assert prep.shape[0] == h
    assert prep.shape[0] == h and prep.shape[1] >= 32
    out_dir = os.path.join(ROOT, "data", "debug")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"prep_h{h}.png")
    cv2.imwrite(out_path, prep)
    print(f"[debug] wrote {out_path} shape={prep.shape}")


def main() -> None:
    configure_stdout_utf8()
    test_dark_bg_inverts()
    print("invert test OK")
    test_resize_height_48_debug_image()
    print("resize test OK")
    test_synthetic_predict()
    print("done")


if __name__ == "__main__":
    main()

