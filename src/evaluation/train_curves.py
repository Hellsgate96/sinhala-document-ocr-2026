# -*- coding: utf-8 -*-
"""Parse training logs / history JSON and plot loss / CER / WER curves.

Used by ``notebooks/local_pipeline.ipynb`` and ``notebooks/colab_pipeline.ipynb``
so examiners see training dynamics without re-running training.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

# epoch 012 | train_loss = 0.0241 | lr = 4.00e-05
_RE_TRAIN = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s*\|\s*train_loss\s*=\s*(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(?:\s*\|\s*lr\s*=\s*(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?",
    re.IGNORECASE,
)
# epoch 012 | val CER = 0.0348 | val WER = 0.0912
_RE_VAL = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s*\|\s*val\s+CER\s*=\s*(?P<cer>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(?:\s*\|\s*val\s+WER\s*=\s*(?P<wer>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?",
    re.IGNORECASE,
)

# Preferred log names for the delivered checkpoint (RESULTS.md).
PREFERRED_LOG_NAMES: Sequence[str] = (
    "train_jul28.log",
    "train_jul29.log",
    "train_jul27.log",
    "train_v2.log",
    "train_web_batch.log",
    "train_user_batch1.log",
    "train_poem_mix.log",
)


@dataclass
class TrainHistory:
    """Per-epoch training metrics."""

    epochs: List[int] = field(default_factory=list)
    train_loss: List[Optional[float]] = field(default_factory=list)
    val_cer: List[Optional[float]] = field(default_factory=list)
    val_wer: List[Optional[float]] = field(default_factory=list)
    lr: List[Optional[float]] = field(default_factory=list)
    source: str = ""
    best_val_cer: Optional[float] = None

    def __len__(self) -> int:
        return len(self.epochs)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source: str = "") -> "TrainHistory":
        epochs = [int(e) for e in data.get("epochs", [])]
        n = len(epochs)

        def _pad(key: str) -> List[Optional[float]]:
            raw = list(data.get(key, []) or [])
            out: List[Optional[float]] = []
            for i in range(n):
                if i >= len(raw) or raw[i] is None:
                    out.append(None)
                else:
                    out.append(float(raw[i]))
            return out

        # Also accept list-of-records form: [{"epoch":1,"train_loss":..., ...}, ...]
        if not epochs and isinstance(data.get("history"), list):
            rows = data["history"]
            epochs = [int(r["epoch"]) for r in rows]
            hist = cls(
                epochs=epochs,
                train_loss=[_opt_float(r.get("train_loss")) for r in rows],
                val_cer=[_opt_float(r.get("val_cer")) for r in rows],
                val_wer=[_opt_float(r.get("val_wer")) for r in rows],
                lr=[_opt_float(r.get("lr")) for r in rows],
                source=source or str(data.get("source", "")),
                best_val_cer=_opt_float(data.get("best_val_cer")),
            )
        else:
            hist = cls(
                epochs=epochs,
                train_loss=_pad("train_loss"),
                val_cer=_pad("val_cer"),
                val_wer=_pad("val_wer"),
                lr=_pad("lr"),
                source=source or str(data.get("source", "")),
                best_val_cer=_opt_float(data.get("best_val_cer")),
            )
        if hist.best_val_cer is None:
            vals = [v for v in hist.val_cer if v is not None]
            hist.best_val_cer = min(vals) if vals else None
        return hist


def _opt_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_log_text(path: Path) -> str:
    """Read a train log, tolerating UTF-16 (common on Windows PowerShell redirects)."""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    # UTF-16 LE without BOM: null bytes on odd indexes in the first chunk.
    sample = raw[:200]
    if sample and sample[1:2] == b"\x00" and b"\x00" in sample[1::2]:
        try:
            return raw.decode("utf-16-le")
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_train_log(path: Union[str, Path]) -> TrainHistory:
    """Parse ``epoch N | train_loss = ...`` / ``val CER = ...`` lines from a log."""
    path = Path(path)
    text = _read_log_text(path)
    by_epoch: Dict[int, Dict[str, Optional[float]]] = {}

    for line in text.splitlines():
        m = _RE_TRAIN.search(line)
        if m:
            ep = int(m.group("epoch"))
            row = by_epoch.setdefault(ep, {})
            row["train_loss"] = float(m.group("loss"))
            if m.group("lr") is not None:
                row["lr"] = float(m.group("lr"))
            continue
        m = _RE_VAL.search(line)
        if m:
            ep = int(m.group("epoch"))
            row = by_epoch.setdefault(ep, {})
            row["val_cer"] = float(m.group("cer"))
            if m.group("wer") is not None:
                row["val_wer"] = float(m.group("wer"))

    epochs = sorted(by_epoch)
    hist = TrainHistory(
        epochs=epochs,
        train_loss=[by_epoch[e].get("train_loss") for e in epochs],
        val_cer=[by_epoch[e].get("val_cer") for e in epochs],
        val_wer=[by_epoch[e].get("val_wer") for e in epochs],
        lr=[by_epoch[e].get("lr") for e in epochs],
        source=str(path),
    )
    vals = [v for v in hist.val_cer if v is not None]
    hist.best_val_cer = min(vals) if vals else None
    return hist


def load_train_history(path: Union[str, Path]) -> TrainHistory:
    """Load history from JSON (array-of-columns or list-of-records)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"history": data}
    return TrainHistory.from_dict(data, source=str(path))


def save_train_history(hist: TrainHistory, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hist.to_dict(), indent=2), encoding="utf-8")
    return path


def find_train_log(
    models_dir: Union[str, Path],
    *,
    preferred: Sequence[str] = PREFERRED_LOG_NAMES,
) -> Optional[Path]:
    """Pick the best available train log under ``models_dir``.

    Preference order: preferred filenames that exist and parse, then newest
    ``train*.log`` / ``*.log`` with at least one epoch row.
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return None

    def _usable(p: Path) -> bool:
        if not p.is_file():
            return False
        try:
            return len(parse_train_log(p)) > 0
        except OSError:
            return False

    for name in preferred:
        cand = models_dir / name
        if _usable(cand):
            return cand

    candidates: List[Path] = []
    for pattern in ("train*.log", "*.log"):
        candidates.extend(models_dir.glob(pattern))
    # Newest first.
    candidates = sorted({c.resolve() for c in candidates}, key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        if _usable(cand):
            return cand
    return None


def resolve_train_history(
    repo_root: Union[str, Path],
    *,
    models_dir: Optional[Union[str, Path]] = None,
    log_path: Optional[Union[str, Path]] = None,
    history_path: Optional[Union[str, Path]] = None,
) -> TrainHistory:
    """Resolve history for notebooks: explicit path → preferred log → JSON → bundled.

    Prefers ``train_jul28.log`` (delivered checkpoint in RESULTS.md) over a
    newer incomplete ``train_history.json`` so demos match the shipped model.
    """
    root = Path(repo_root)
    models = Path(models_dir) if models_dir else root / "models"
    bundled = root / "data" / "metrics" / "train_history_jul28.json"

    if history_path:
        return load_train_history(history_path)
    if log_path:
        return parse_train_log(log_path)

    log = find_train_log(models)
    if log is not None:
        return parse_train_log(log)

    live_hist = models / "train_history.json"
    if live_hist.is_file():
        return load_train_history(live_hist)

    if bundled.is_file():
        return load_train_history(bundled)

    raise FileNotFoundError(
        f"No train history found under {models} or {bundled}. "
        "Place a train_*.log in models/ or data/metrics/train_history_jul28.json."
    )


def load_eval_summary(
    repo_root: Union[str, Path],
    *,
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load held-out / in-train CER summary for bar charts."""
    root = Path(repo_root)
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.extend(
        [
            root / "models" / "eval_summary.json",
            root / "data" / "metrics" / "eval_summary.json",
        ]
    )
    for cand in candidates:
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError("eval_summary.json not found under models/ or data/metrics/")


def plot_train_curves(
    hist: TrainHistory,
    *,
    title: Optional[str] = None,
    show_wer: bool = True,
    show_lr: bool = True,
    figsize: tuple = (10, 8),
    show: bool = False,
):
    """Plot train loss, val CER/(WER), and optional LR vs epoch.

    Returns a matplotlib Figure (caller can ``plt.show()`` or display in Jupyter).
    """
    import matplotlib.pyplot as plt

    n_rows = 2 + (1 if show_lr and any(v is not None for v in hist.lr) else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]
    epochs = hist.epochs
    src_name = Path(hist.source).name if hist.source else "train history"
    fig.suptitle(title or f"Training curves — {src_name}", fontsize=13, fontweight="bold")

    ax = axes[0]
    xs, ys = _xy(epochs, hist.train_loss)
    if xs:
        ax.plot(xs, ys, "o-", color="#1f77b4", linewidth=2, markersize=5, label="train loss")
    ax.set_ylabel("Train loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    xs, ys = _xy(epochs, hist.val_cer)
    if xs:
        ax.plot(xs, ys, "s-", color="#d62728", linewidth=2, markersize=5, label="val CER")
        if hist.best_val_cer is not None:
            ax.axhline(
                hist.best_val_cer,
                color="#d62728",
                linestyle="--",
                alpha=0.5,
                label=f"best CER = {hist.best_val_cer:.4f}",
            )
    if show_wer:
        xs, ys = _xy(epochs, hist.val_wer)
        if xs:
            ax.plot(xs, ys, "^-", color="#ff7f0e", linewidth=1.5, markersize=4, label="val WER")
    ax.set_ylabel("Validation error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    if n_rows >= 3:
        ax = axes[2]
        xs, ys = _xy(epochs, hist.lr)
        if xs:
            ax.plot(xs, ys, "D-", color="#2ca02c", linewidth=2, markersize=4, label="learning rate")
        ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_eval_cer_bars(
    summary: Mapping[str, Any],
    *,
    title: str = "Held-out vs in-train CER / Character Accuracy",
    figsize: tuple = (9, 5.0),
    show: bool = False,
    show_accuracy: bool = True,
):
    """Bar chart of end-to-end CER from ``eval_summary.json`` / RESULTS.md numbers.

    When ``show_accuracy`` is True, each bar is also labelled with
    Character Accuracy = (1 − CER) × 100%.
    """
    import matplotlib.pyplot as plt

    sets = list(summary.get("sets", []))
    if not sets:
        raise ValueError("eval summary has no 'sets' entries")

    labels = [s["label"] for s in sets]
    cers = [float(s["cer"]) for s in sets]
    held = [bool(s.get("held_out", False)) for s in sets]
    colors = ["#2ca02c" if h else "#7f7f7f" for h in held]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(labels)), cers, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Corpus CER (lower is better)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, cers):
        acc = max(0.0, 1.0 - val) * 100.0
        label = f"{val:.4f}"
        if show_accuracy:
            label = f"{val:.4f}\n({acc:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    # Legend
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#2ca02c", edgecolor="black", label="Held out"),
            Patch(facecolor="#7f7f7f", edgecolor="black", label="In training (reference)"),
        ],
        loc="upper right",
    )
    note = summary.get("note") or summary.get("checkpoint")
    if note:
        ax.text(0.01, -0.32, str(note), transform=ax.transAxes, fontsize=8, color="#444444")
    if show_accuracy:
        ax.text(
            0.01,
            -0.40,
            "Bar labels: CER (Character Accuracy % = (1 − CER) × 100)",
            transform=ax.transAxes,
            fontsize=8,
            color="#444444",
        )
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def format_eval_summary_table(summary: Mapping[str, Any]) -> str:
    """Plain-text table with CER, Character Accuracy %, optional WER."""
    rows = []
    header = (
        f"{'set':<28} {'CER':>8} {'CharAcc%':>10} {'WER':>8} {'WordAcc%':>10} {'held_out':>8}"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for s in summary.get("sets", []):
        cer_v = float(s["cer"])
        acc = max(0.0, 1.0 - cer_v) * 100.0
        wer_v = s.get("wer")
        if wer_v is None:
            wer_s, wacc_s = "—", "—"
        else:
            wer_f = float(wer_v)
            wer_s = f"{wer_f:.4f}"
            wacc_s = f"{max(0.0, 1.0 - wer_f) * 100.0:.2f}"
        rows.append(
            f"{s['label']:<28} {cer_v:8.4f} {acc:10.2f} {wer_s:>8} {wacc_s:>10} "
            f"{str(bool(s.get('held_out', False))):>8}"
        )
    return "\n".join(rows)


def summary_table_rows(hist: TrainHistory) -> List[Dict[str, Any]]:
    """Compact per-epoch rows for notebook display."""
    rows = []
    for i, ep in enumerate(hist.epochs):
        rows.append(
            {
                "epoch": ep,
                "train_loss": hist.train_loss[i],
                "val_cer": hist.val_cer[i],
                "val_wer": hist.val_wer[i],
                "lr": hist.lr[i],
            }
        )
    return rows


def _xy(epochs: Sequence[int], values: Sequence[Optional[float]]):
    xs, ys = [], []
    for e, v in zip(epochs, values):
        if v is not None:
            xs.append(e)
            ys.append(v)
    return xs, ys


def append_history_epoch(
    path: Union[str, Path],
    *,
    epoch: int,
    train_loss: float,
    val_cer: Optional[float] = None,
    val_wer: Optional[float] = None,
    lr: Optional[float] = None,
) -> TrainHistory:
    """Append one epoch to ``train_history.json`` (used by ``train.py``)."""
    path = Path(path)
    if path.is_file():
        hist = load_train_history(path)
    else:
        hist = TrainHistory(source=str(path))

    # Replace existing epoch row if re-run, else append.
    if epoch in hist.epochs:
        i = hist.epochs.index(epoch)
        hist.train_loss[i] = train_loss
        hist.val_cer[i] = val_cer
        hist.val_wer[i] = val_wer
        hist.lr[i] = lr
    else:
        hist.epochs.append(epoch)
        hist.train_loss.append(train_loss)
        hist.val_cer.append(val_cer)
        hist.val_wer.append(val_wer)
        hist.lr.append(lr)

    vals = [v for v in hist.val_cer if v is not None]
    hist.best_val_cer = min(vals) if vals else None
    hist.source = str(path)
    save_train_history(hist, path)
    return hist
