"""Train the CRNN Sinhala line recognizer with CTC loss.

Loads ``configs/default.yaml``, builds the charset (saved alongside checkpoints),
trains on the synthetic train split, validates each epoch by Character Error Rate,
and saves the best checkpoint to ``models/``.

Example:
    python -m src.recognition.train --config configs/default.yaml \
        train.epochs=5 train.batch_size=32
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.charset import Charset
from src.data.dataset import build_dataloader
from src.evaluation.metrics import evaluate_model
from src.recognition.model import build_crnn
from src.utils.common import (apply_overrides, configure_stdout_utf8, get_device,
                              get_logger, load_config, save_checkpoint, set_seed)


def build_optimizer(model, cfg):
    name = cfg.get("optimizer", "adam").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, logger):
    model.train()
    running = 0.0
    for images, targets, target_lengths, _widths, _texts in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        log_probs = model(images)                       # (T, B, C)
        t, b, _ = log_probs.size()
        input_lengths = torch.full((b,), t, dtype=torch.long, device=device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths.to(device))
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser(description="Train the CRNN recognizer.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*", help="a.b.c=value overrides")
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("train")
    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg["project"]["seed"])

    device = get_device(cfg["train"].get("device", "auto"))
    logger.info(f"device = {device}")

    # Charset: reuse an existing one if present, else build + save the default.
    charset_path = cfg["paths"]["charset_path"]
    if os.path.isfile(charset_path):
        charset = Charset.load(charset_path)
    else:
        charset = Charset.build_default()
        charset.save(charset_path)
    logger.info(f"charset classes (incl. blank) = {charset.num_classes}")

    syn = cfg["paths"]["synthetic_dir"]
    train_loader = build_dataloader(
        os.path.join(syn, "train_labels.txt"), charset,
        batch_size=cfg["train"]["batch_size"], height=cfg["image"]["height"],
        max_width=cfg["image"]["max_width"], channels=cfg["image"]["channels"],
        shuffle=True, num_workers=cfg["train"]["num_workers"],
    )
    val_loader = build_dataloader(
        os.path.join(syn, "val_labels.txt"), charset,
        batch_size=cfg["train"]["batch_size"], height=cfg["image"]["height"],
        max_width=cfg["image"]["max_width"], channels=cfg["image"]["channels"],
        shuffle=False, num_workers=cfg["train"]["num_workers"],
    )

    model = build_crnn(charset.num_classes, cfg.get("model"),
                       in_channels=cfg["image"]["channels"]).to(device)
    criterion = nn.CTCLoss(blank=Charset.BLANK_INDEX, zero_infinity=True)
    optimizer = build_optimizer(model, cfg["train"])

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    best_cer = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                               cfg["train"].get("grad_clip", 0), logger)
        logger.info(f"epoch {epoch:03d} | train_loss = {loss:.4f}")

        if epoch % cfg["train"].get("val_every", 1) == 0:
            report = evaluate_model(model, val_loader, charset, device=device,
                                    measure_cpu_time=False)
            logger.info(f"epoch {epoch:03d} | val CER = {report['cer']:.4f} "
                        f"| val WER = {report['wer']:.4f}")
            save_checkpoint(os.path.join(cfg["paths"]["models_dir"], "crnn_last.pth"),
                            model, optimizer, epoch, extra={"cer": report["cer"]})
            if report["cer"] < best_cer:
                best_cer = report["cer"]
                save_checkpoint(os.path.join(cfg["paths"]["models_dir"], "crnn_best.pth"),
                                model, optimizer, epoch, extra={"cer": best_cer})
                logger.info(f"  -> new best CER {best_cer:.4f} (saved crnn_best.pth)")

    logger.info(f"training complete. best val CER = {best_cer:.4f}")


if __name__ == "__main__":
    main()