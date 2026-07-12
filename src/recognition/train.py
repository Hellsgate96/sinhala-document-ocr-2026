from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.charset import Charset
from src.data.dataset import build_dataloader
from src.evaluation.metrics import evaluate_model
from src.recognition.model import build_crnn
from src.utils.common import (
    apply_overrides,
    configure_stdout_utf8,
    get_device,
    get_logger,
    load_config,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def build_optimizer(model, cfg):
    name = cfg.get("optimizer", "adam").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg, total_epochs: int):
    """LR scheduler from ``train.lr_scheduler``: plateau (default) | cosine | none.

    * ``plateau`` - ReduceLROnPlateau on val CER (factor 0.5, patience 3).
    * ``cosine`` - CosineAnnealingLR over the full run.
    Returns (scheduler, kind) where kind is "plateau", "cosine" or None.
    """
    kind = str(cfg.get("lr_scheduler", "plateau")).lower()
    if kind in ("none", "off", ""):
        return None, None
    if kind == "cosine":
        min_lr = float(cfg.get("min_lr", 1e-5))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs), eta_min=min_lr)
        return sched, "cosine"
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(cfg.get("lr_factor", 0.5)),
        patience=int(cfg.get("lr_patience", 3)),
        min_lr=float(cfg.get("min_lr", 1e-6)))
    return sched, "plateau"


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, logger):
    model.train()
    running = 0.0
    for images, targets, target_lengths, _widths, _texts in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        log_probs = model(images)
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


def _infer_extra_base_dir(extra_path: str, real_dir: str) -> str:
    """Base dir for an extra label file's relative image paths.

    Files living under ``real_dir`` use ``real_dir`` itself as the base (the
    poem fine-tune convention: ``real_dir/labels/*.txt`` references images under
    the sibling ``real_dir/images/``). Anything else (e.g. a generator's own
    output dir, such as ``data/synthetic_pages``) uses its own directory,
    matching the primary synthetic dataset convention.
    """
    abs_extra = os.path.abspath(extra_path)
    abs_real = os.path.abspath(real_dir)
    if abs_extra == abs_real or abs_extra.startswith(abs_real + os.sep):
        return abs_real
    return os.path.dirname(abs_extra)


def _resolve_extra_label_paths(cfg: Dict[str, Any], cli_paths: Optional[List[str]]) -> List[str]:
    paths: List[str] = []
    if cli_paths:
        paths.extend(cli_paths)
    extra = cfg.get("paths", {}).get("extra_train_labels")
    if extra:
        if isinstance(extra, str):
            paths.append(extra)
        else:
            paths.extend(list(extra))
    seen = set()
    out: List[str] = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="Train the CRNN recognizer.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint to load before training")
    parser.add_argument(
        "--finetune-real-only",
        action="store_true",
        help="Train only on real/poem labels (see paths.poem_labels or --extra-labels)",
    )
    parser.add_argument(
        "--extra-labels",
        action="append",
        default=None,
        help="Additional label files merged into the training set (path<TAB>transcript)",
    )
    parser.add_argument("overrides", nargs="*", help="a.b.c=value overrides")
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("train")
    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg["project"]["seed"])

    device = get_device(cfg["train"].get("device", "auto"))
    logger.info(f"device = {device}")

    charset_path = cfg["paths"]["charset_path"]
    if os.path.isfile(charset_path):
        charset = Charset.load(charset_path)
    else:
        charset = Charset.build_default()
        charset.save(charset_path)
    logger.info(f"charset classes (incl. blank) = {charset.num_classes}")

    syn = cfg["paths"]["synthetic_dir"]
    train_labels = os.path.join(syn, "train_labels.txt")
    extra_paths = _resolve_extra_label_paths(cfg, args.extra_labels)
    real_dir = cfg["paths"].get("real_dir", "data/real")
    # Base dir per extra-labels file: labels living under `real_dir` (the poem
    # fine-tune convention - images/ is a SIBLING of labels/, not label file's own
    # dir) resolve there; any other extra-labels file (e.g. the v3 detector-in-the-
    # loop page supplement) resolves relative to its OWN directory, matching the
    # primary synthetic dataset convention.
    extra_base_dirs = [_infer_extra_base_dir(p, real_dir) for p in extra_paths] if extra_paths else None

    train_cfg = cfg["train"]
    extra_repeat = int(train_cfg.get("extra_label_repeat", 1))
    syn_max = train_cfg.get("synthetic_train_max")
    syn_max = int(syn_max) if syn_max not in (None, "", 0) else None
    sample_seed = int(cfg["project"].get("seed", 1337))

    finetune_real_only = bool(train_cfg.get("finetune_real_only", False)) or args.finetune_real_only
    poem_labels = cfg.get("paths", {}).get("poem_labels")
    real_train_labels = poem_labels or (extra_paths[0] if extra_paths else None)

    dl_common = dict(
        charset=charset,
        batch_size=train_cfg["batch_size"],
        height=cfg["image"]["height"],
        max_width=cfg["image"]["max_width"],
        channels=cfg["image"]["channels"],
        num_workers=train_cfg["num_workers"],
    )

    if finetune_real_only:
        if not real_train_labels or not os.path.isfile(real_train_labels):
            raise FileNotFoundError(
                "finetune_real_only requires paths.poem_labels or --extra-labels pointing to a labels file"
            )
        logger.info(f"finetune_real_only: training on {real_train_labels} (repeat={extra_repeat})")
        train_loader = build_dataloader(
            real_train_labels,
            base_dir=real_dir,
            shuffle=True,
            primary_label_repeat=extra_repeat,
            **dl_common,
        )
        val_loader = build_dataloader(
            real_train_labels,
            base_dir=real_dir,
            shuffle=False,
            extra_label_repeat=1,
            **dl_common,
        )
        total_epochs_override = train_cfg.get("finetune_real_only_epochs")
        if total_epochs_override:
            cfg["train"] = dict(cfg["train"])
            cfg["train"]["epochs"] = int(total_epochs_override)
    else:
        train_loader = build_dataloader(
            train_labels,
            shuffle=True,
            extra_label_paths=extra_paths or None,
            extra_base_dirs=extra_base_dirs,
            extra_label_repeat=extra_repeat,
            max_primary_samples=syn_max,
            primary_sample_seed=sample_seed,
            **dl_common,
        )
        if extra_paths:
            logger.info(
                f"merged {len(extra_paths)} extra label file(s) "
                f"(repeat={extra_repeat}, synthetic_cap={syn_max})"
            )
        val_loader = build_dataloader(
            os.path.join(syn, "val_labels.txt"),
            shuffle=False,
            **dl_common,
        )

    model = build_crnn(charset.num_classes, cfg.get("model"), in_channels=cfg["image"]["channels"]).to(device)
    criterion = nn.CTCLoss(blank=Charset.BLANK_INDEX, zero_infinity=True)
    optimizer = build_optimizer(model, cfg["train"])
    scheduler, scheduler_kind = build_scheduler(
        optimizer, cfg["train"], int(cfg["train"]["epochs"]))
    if scheduler_kind:
        logger.info(f"lr scheduler = {scheduler_kind}")

    resume_path = args.resume or cfg.get("paths", {}).get("resume_checkpoint")
    start_epoch = 0
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer=None, map_location=str(device))
        start_epoch = int(ckpt.get("epoch", 0))
        logger.info(f"resumed weights from {resume_path} (checkpoint epoch {start_epoch})")

    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    best_name = cfg.get("paths", {}).get("finetune_best") or "crnn_best.pth"
    last_name = cfg.get("paths", {}).get("finetune_last") or "crnn_last.pth"
    best_path = os.path.join(models_dir, os.path.basename(best_name))
    last_path = os.path.join(models_dir, os.path.basename(last_name))

    best_cer = float("inf")
    total_epochs = cfg["train"]["epochs"]
    patience = int(cfg["train"].get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    for epoch in range(1, total_epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg["train"].get("grad_clip", 0),
            logger,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"epoch {epoch:03d} | train_loss = {loss:.4f} | lr = {current_lr:.2e}")

        if epoch % cfg["train"].get("val_every", 1) == 0:
            report = evaluate_model(model, val_loader, charset, device=device, measure_cpu_time=False)
            logger.info(
                f"epoch {epoch:03d} | val CER = {report['cer']:.4f} | val WER = {report['wer']:.4f}"
            )
            save_checkpoint(last_path, model, optimizer, epoch, extra={"cer": report["cer"]})
            if report["cer"] < best_cer:
                best_cer = report["cer"]
                epochs_without_improvement = 0
                save_checkpoint(best_path, model, optimizer, epoch, extra={"cer": best_cer})
                logger.info(f"  -> new best CER {best_cer:.4f} (saved {os.path.basename(best_path)})")
            else:
                epochs_without_improvement += 1
                if patience and epochs_without_improvement >= patience:
                    logger.info(
                        f"early stopping: no val CER improvement in {patience} validations "
                        f"(best {best_cer:.4f})"
                    )
                    break
            if scheduler_kind == "plateau":
                scheduler.step(report["cer"])
        if scheduler_kind == "cosine":
            scheduler.step()

    logger.info(f"training complete. best val CER = {best_cer:.4f}")


if __name__ == "__main__":
    main()
