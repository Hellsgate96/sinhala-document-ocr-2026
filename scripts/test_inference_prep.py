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


def main() -> None:
    configure_stdout_utf8()
    test_dark_bg_inverts()
    print("invert test OK")
    test_synthetic_predict()
    print("done")


if __name__ == "__main__":
    main()
