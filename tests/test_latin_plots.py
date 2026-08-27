# -*- coding: utf-8 -*-
"""Reproduce the Colab tofu failure: Noto Sans Sinhala as matplotlib default.

Noto has digits but not Latin letters, so bar numbers like ``0.0088 (99.1%)``
stay visible while titles/axes/legends become empty boxes and
``Glyph missing from font(s) Noto Sans Sinhala`` fires on IPython redraw.

Run:  python tests/test_latin_plots.py
"""

from __future__ import annotations

import os
import sys
import warnings
from io import BytesIO
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")

from src.evaluation.train_curves import (  # noqa: E402
    TrainHistory,
    plot_eval_cer_bars,
    plot_train_curves,
)
from src.utils.display import (  # noqa: E402
    apply_latin_font,
    find_sinhala_font,
    iter_text_artists,
)


def _force_noto_as_default() -> str | None:
    """Match Colab after ``apt-get install fonts-noto-core`` + old setup()."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    # Prefer the Latin-poor Noto face (Nirmala UI on Windows has Latin glyphs).
    path = Path(ROOT) / "fonts" / "NotoSansSinhala-Regular.ttf"
    if not path.is_file():
        found = find_sinhala_font()
        path = found if found is not None else None
    if path is None or not Path(path).is_file():
        return None
    path = Path(path)
    fm.fontManager.addfont(str(path))
    name = fm.FontProperties(fname=str(path)).get_name()
    plt.rcParams["font.family"] = name
    plt.rcParams["font.sans-serif"] = [name]
    return name


def _require_noto_default() -> str:
    name = _force_noto_as_default()
    if name and "sinhala" in name.lower():
        return name
    msg = "fonts/NotoSansSinhala-Regular.ttf not found (Colab installs fonts-noto-core)"
    try:
        import pytest
    except ImportError:
        pytest = None
    if pytest is not None:
        pytest.skip(msg)
    print("SKIP:", msg)
    return ""


def _sample_hist() -> TrainHistory:
    return TrainHistory(
        epochs=[1, 2, 3],
        train_loss=[0.09, 0.06, 0.024],
        val_cer=[0.037, 0.035, 0.0348],
        val_wer=[0.097, 0.094, 0.091],
        lr=[8e-5, 8e-5, 4e-5],
        source="train_jul28.log",
        best_val_cer=0.0348,
    )


def _sample_summary() -> dict:
    return {
        "checkpoint": "models/crnn_best.pth",
        "note": "Green bars; §3g adds safe word-level rules. Formula: (1 − CER) × 100.",
        "sets": [
            {"label": "Real photos (held out)", "cer": 0.0325, "held_out": True},
            {"label": "Synthetic pages (held out)", "cer": 0.0088, "held_out": True},
        ],
    }


def _print_figure(fig) -> None:
    """Same path IPython uses: ``fig.canvas.print_figure``."""
    buf = BytesIO()
    fig.canvas.print_figure(buf, format="png")


def _artist_font_names(fig) -> list[str]:
    names = []
    for t in iter_text_artists(fig):
        fp = t.get_fontproperties()
        bit = " ".join(
            str(x)
            for x in (fp.get_name(), fp.get_file(), fp.get_family())
            if x
        )
        names.append(bit)
    return names


def _assert_no_sinhala(fig, label: str) -> None:
    bad = [n for n in _artist_font_names(fig) if "sinhala" in n.lower()]
    assert not bad, f"{label}: text artists still using Sinhala font: {bad[:8]}"


def _glyph_warnings(caught) -> list[str]:
    out = []
    for item in caught:
        msg = str(item.message)
        if "Glyph" in msg or "glyph" in msg.lower():
            out.append(msg)
    return out


def test_eval_and_train_plots_survive_noto_default():
    name = _require_noto_default()
    if not name:
        return

    hist = _sample_hist()
    summary = _sample_summary()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig_curves = plot_train_curves(hist, show=False)
        fig_bars = plot_eval_cer_bars(summary, show=False)
        apply_latin_font(fig_curves)
        apply_latin_font(fig_bars)
        _print_figure(fig_curves)
        _print_figure(fig_bars)
        glyph = _glyph_warnings(caught)

    _assert_no_sinhala(fig_curves, "plot_train_curves")
    _assert_no_sinhala(fig_bars, "plot_eval_cer_bars")
    assert not glyph, "Glyph missing warnings after Latin bake:\n" + "\n".join(glyph[:12])

    # Captions / titles must still be real strings (not empty) and baked to a file.
    bar_title = fig_bars.axes[0].title.get_text()
    assert "Held-out" in bar_title or "CER" in bar_title
    title_file = fig_bars.axes[0].title.get_fontproperties().get_file() or ""
    assert title_file, "title FontProperties must pin fname= (family-only fails on Colab redraw)"
    assert "sinhala" not in title_file.lower()


def test_previous_family_only_bake_is_insufficient():
    """Document why f6fde6d still showed tofu: family= without fname + incomplete walk."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    name = _require_noto_default()
    if not name:
        return

    fig, ax = plt.subplots()
    ax.set_title("Held-out CER / Character Accuracy")
    ax.set_ylabel("Corpus CER (lower is better)")
    ax.text(0.5, 0.5, "Green §3g formula")
    # f6fde6d-style: family name only, title only.
    ax.title.set_fontproperties(fm.FontProperties(family="DejaVu Sans"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _print_figure(fig)
        glyph = _glyph_warnings(caught)
    plt.close(fig)
    assert glyph, "expected Glyph warnings when only the title gets family= DejaVu"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    test_previous_family_only_bake_is_insufficient()
    test_eval_and_train_plots_survive_noto_default()
    print("test_latin_plots: ALL PASSED")
