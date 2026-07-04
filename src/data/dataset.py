from __future__ import annotations

import os
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.charset import Charset


def read_labels(labels_path: str) -> List[Tuple[str, str]]:
    """Parse a ``path<TAB>transcript`` labels file into a list of pairs."""
    rows: List[Tuple[str, str]] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rows.append((parts[0], "\t".join(parts[1:])))
    return rows


def resize_keep_height(
    img: Image.Image,
    height: int,
    max_width: int,
    min_width: int = 0,
    pad_value: int = 255,
) -> Image.Image:
    """Resize to fixed height (aspect ratio), clamp width, optional horizontal pad."""
    w, h = img.size
    if h < 1:
        h = 1
    new_w = max(1, int(round(w * height / h)))
    new_w = min(new_w, max_width)
    resample = getattr(Image, "LANCZOS", Image.BILINEAR)
    out = img.resize((new_w, height), resample)
    if min_width and new_w < min_width:
        canvas = Image.new(img.mode, (min_width, height), pad_value)
        canvas.paste(out, (0, 0))
        out = canvas
    return out


def _merge_label_records(
    labels_path: str,
    extra_label_paths: Optional[List[str]] = None,
    extra_base_dirs: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Build (base_dir, rel_path, text) rows from one or more label files."""
    rows: List[Tuple[str, str, str]] = []
    primary_base = os.path.dirname(os.path.abspath(labels_path))
    for rel, text in read_labels(labels_path):
        rows.append((primary_base, rel, text))
    extras = extra_label_paths or []
    bases = extra_base_dirs or []
    for i, extra_path in enumerate(extras):
        base = bases[i] if i < len(bases) else os.path.dirname(os.path.abspath(extra_path))
        base = os.path.abspath(base)
        for rel, text in read_labels(extra_path):
            rows.append((base, rel, text))
    return rows


class OCRLineDataset(Dataset):
    """Line-level OCR dataset yielding (image_tensor, target, target_length, text)."""

    def __init__(self,
                 labels_path: str,
                 charset: Charset,
                 base_dir: Optional[str] = None,
                 height: int = 32,
                 max_width: int = 512,
                 channels: int = 1,
                 transform: Optional[Callable] = None,
                 extra_label_paths: Optional[List[str]] = None,
                 extra_base_dirs: Optional[List[str]] = None):
        if extra_label_paths:
            merged = _merge_label_records(labels_path, extra_label_paths, extra_base_dirs)
            self.records = [(rel, text) for _base, rel, text in merged]
            self._image_bases = [base for base, _rel, _text in merged]
        else:
            self.records = read_labels(labels_path)
            default_base = base_dir or os.path.dirname(os.path.abspath(labels_path))
            self._image_bases = [default_base] * len(self.records)
        self.charset = charset
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(labels_path))
        self.height = height
        self.max_width = max_width
        self.channels = channels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel_path: str, base_dir: Optional[str] = None) -> Image.Image:
        root = base_dir or self.base_dir
        path = rel_path if os.path.isabs(rel_path) else os.path.join(root, rel_path)
        mode = "L" if self.channels == 1 else "RGB"
        return Image.open(path).convert(mode)

    def __getitem__(self, index: int):
        rel_path, text = self.records[index]
        img = self._load_image(rel_path, self._image_bases[index])
        img = resize_keep_height(img, self.height, self.max_width)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.channels == 1:
            arr = arr[None, :, :]
        else:
            arr = np.transpose(arr, (2, 0, 1))
        arr = (arr - 0.5) / 0.5
        image = torch.from_numpy(arr)

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(self.charset.encode(text), dtype=torch.long)
        return image, target, len(target), text


def _ctc_collate_batch(batch, pad_value: float = 1.0):
    """Right-pad images to the batch max width (top-level for Windows multiprocessing)."""
    images, targets, target_lengths, texts = zip(*batch)
    c = images[0].shape[0]
    h = images[0].shape[1]
    widths = [im.shape[2] for im in images]
    max_w = max(widths)

    padded = torch.full((len(images), c, h, max_w), pad_value, dtype=torch.float32)
    for i, im in enumerate(images):
        padded[i, :, :, : im.shape[2]] = im

    flat_targets = torch.cat([t for t in targets]) if targets else torch.tensor([], dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    widths = torch.tensor(widths, dtype=torch.long)
    return padded, flat_targets, target_lengths, widths, list(texts)


def ctc_collate(pad_value: float = 1.0) -> Callable:
    """Build a collate_fn that right-pads images to the batch max width."""
    return partial(_ctc_collate_batch, pad_value=pad_value)


def build_dataloader(labels_path: str,
                     charset: Charset,
                     batch_size: int = 64,
                     height: int = 32,
                     max_width: int = 512,
                     channels: int = 1,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     base_dir: Optional[str] = None,
                     pad_value: float = 1.0,
                     extra_label_paths: Optional[List[str]] = None,
                     extra_base_dirs: Optional[List[str]] = None) -> DataLoader:
    """Convenience builder returning a ready DataLoader for a labels file."""
    dataset = OCRLineDataset(
        labels_path=labels_path, charset=charset, base_dir=base_dir,
        height=height, max_width=max_width, channels=channels,
        extra_label_paths=extra_label_paths,
        extra_base_dirs=extra_base_dirs,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=ctc_collate(pad_value=pad_value), drop_last=False,
    )
