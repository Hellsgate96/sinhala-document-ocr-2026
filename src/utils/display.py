"""Jupyter / matplotlib helpers for Sinhala Unicode in notebooks.

Sinhala predictions are shown via UTF-8 print / HTML (``display_sinhala_table``),
not via matplotlib. Metric plots use English labels and must keep a Latin-capable
font (see ``use_latin_plots``) — Noto Sans Sinhala lacks Latin glyphs and causes
tofu boxes / ``Glyph missing`` warnings if set as the global matplotlib family.
"""

from __future__ import annotations

import contextlib
import glob
import html
import sys
import warnings
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from src.utils.common import configure_stdout_utf8

_registered_font_path: Optional[str] = None
_registered_font_name: Optional[str] = None

# Latin-capable stack for English metric plots (titles, axes, legends).
# Noto Sans Sinhala must never appear here — it has digits but not Latin letters.
_LATIN_SANS_SERIF = (
    "DejaVu Sans",
    "Liberation Sans",
    "Bitstream Vera Sans",
    "Arial",
    "Helvetica",
    "sans-serif",
)

_latin_font_path: Optional[str] = None
_latin_font_name: Optional[str] = None


def configure_display_utf8() -> None:
    """Enable UTF-8 on stdout/stderr (wraps ``configure_stdout_utf8``)."""
    configure_stdout_utf8()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _static_font_candidates() -> list[Path]:
    root = _project_root()
    candidates: list[Path] = []
    if sys.platform == "win32":
        candidates.extend(
            [
                Path(r"C:/Windows/Fonts/Nirmala.ttc"),
                Path(r"C:/Windows/Fonts/NirmalaB.ttf"),
                Path(r"C:/Windows/Fonts/iskpota.ttf"),
            ]
        )
    candidates.append(root / "fonts" / "NotoSansSinhala-Regular.ttf")
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/noto/NotoSansSinhala-Regular.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansSinhala-Regular.ttf"),
        ]
    )
    return candidates


def _glob_sinhala_fonts() -> list[Path]:
    patterns = (
        "/usr/share/fonts/**/*Sinhala*.ttf",
        "/usr/share/fonts/**/*Sinhala*.otf",
        str(_project_root() / "fonts" / "*Sinhala*.ttf"),
    )
    found: list[Path] = []
    for pattern in patterns:
        for hit in glob.glob(pattern, recursive=True):
            p = Path(hit)
            if p.is_file():
                found.append(p)
    return found


def find_sinhala_font(font_path: Optional[str] = None) -> Optional[Path]:
    """Return the first usable Sinhala-capable font file."""
    if font_path:
        explicit = Path(font_path)
        if explicit.is_file():
            return explicit
    for candidate in _static_font_candidates():
        if candidate.is_file():
            return candidate
    globbed = _glob_sinhala_fonts()
    return globbed[0] if globbed else None


def sinhala_font_css_family(font_path: Optional[str] = None) -> str:
    """CSS ``font-family`` stack for IPython HTML output."""
    path = find_sinhala_font(font_path)
    if path is not None:
        try:
            from matplotlib import font_manager

            prop = font_manager.FontProperties(fname=str(path))
            name = prop.get_name()
            return f"'{name}', 'Nirmala UI', 'Noto Sans Sinhala', sans-serif"
        except Exception:
            pass
    return "'Nirmala UI', 'Noto Sans Sinhala', sans-serif"


def setup_matplotlib_sinhala(
    font_path: Optional[str] = None,
    *,
    set_as_default: bool = False,
) -> Optional[str]:
    """Register a Sinhala font for HTML tables; never set it as matplotlib default.

    Noto Sans Sinhala lacks Latin glyphs (digits render, letters become tofu).
    English metric plots must keep DejaVu/Liberation. ``set_as_default`` is
    accepted but ignored.
    """
    global _registered_font_path, _registered_font_name

    from matplotlib import font_manager

    path = find_sinhala_font(font_path)
    if path is None:
        warnings.warn(
            "No Sinhala-capable font found for matplotlib. "
            "Sinhala plot labels may show missing glyphs (tofu). "
            "Predictions via print/HTML are unaffected. "
            "On Windows use Nirmala UI; on Linux install fonts-noto-core or "
            "run scripts/download_fonts.ps1 to fetch Noto Sans Sinhala into fonts/.",
            UserWarning,
            stacklevel=2,
        )
        return None

    font_manager.fontManager.addfont(str(path))
    prop = font_manager.FontProperties(fname=str(path))
    name = prop.get_name()

    _registered_font_path = str(path)
    _registered_font_name = name

    if set_as_default:
        warnings.warn(
            "setup_matplotlib_sinhala(set_as_default=True) is ignored: "
            "Noto Sans Sinhala lacks Latin glyphs and would tofu English plots. "
            "Pass FontProperties(fname=...) only for rare Sinhala matplotlib text.",
            UserWarning,
            stacklevel=2,
        )

    # Always restore a Latin default so addfont(Noto) cannot win later draws.
    ensure_latin_plot_fonts()
    return _registered_font_path


def latin_font_path() -> Optional[str]:
    """Absolute path to DejaVu Sans (or Liberation Sans) for ``FontProperties(fname=)``."""
    global _latin_font_path, _latin_font_name
    if _latin_font_path:
        return _latin_font_path

    import matplotlib
    from matplotlib import font_manager as fm

    candidates = [
        Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path(r"C:/Windows/Fonts/arial.ttf"),
        Path(r"C:/Windows/Fonts/calibri.ttf"),
    ]
    path: Optional[Path] = next((p for p in candidates if p.is_file()), None)
    if path is None:
        try:
            found = fm.findfont(
                fm.FontProperties(family="DejaVu Sans"),
                fallback_to_default=True,
            )
            if found and "sinhala" not in found.lower():
                path = Path(found)
        except Exception:
            path = None
    if path is None or not path.is_file():
        return None

    _latin_font_path = str(path)
    try:
        fm.fontManager.addfont(_latin_font_path)
    except (OSError, ValueError, RuntimeError):
        pass
    try:
        _latin_font_name = fm.FontProperties(fname=_latin_font_path).get_name()
    except Exception:
        _latin_font_name = "DejaVu Sans"
    return _latin_font_path


def latin_fontproperties():
    """``FontProperties`` pinned to a Latin TTF via ``fname`` (not family name).

    ``FontProperties(family='DejaVu Sans')`` leaves ``get_file()`` empty, so a
    later IPython ``print_figure`` can still resolve glyphs through a Noto
    Sans Sinhala rcParam. Pinning ``fname`` survives that redraw.
    """
    from matplotlib import font_manager as fm

    path = latin_font_path()
    if path:
        return fm.FontProperties(fname=path)
    return fm.FontProperties(family="DejaVu Sans")


def _latin_rc() -> dict[str, Any]:
    name = _latin_font_name or "DejaVu Sans"
    sans = [name, *[s for s in _LATIN_SANS_SERIF if s != name]]
    sans = [s for s in sans if "sinhala" not in str(s).lower()]
    return {
        "font.family": name,
        "font.sans-serif": sans,
        "axes.unicode_minus": False,
    }


def ensure_latin_plot_fonts() -> None:
    """Set matplotlib rcParams to DejaVu/Liberation and register the TTF.

    Call this after ``setup_matplotlib_sinhala`` and immediately before English
    metric plots. Do **not** use a generic ``sans-serif`` family: on Colab after
    ``fonts-noto-core``, fontconfig can still pick Noto Sans Sinhala.
    """
    import matplotlib.pyplot as plt

    latin_font_path()
    for key, value in _latin_rc().items():
        plt.rcParams[key] = value
    plt.rcParams["font.sans-serif"] = [
        f for f in plt.rcParams["font.sans-serif"] if "sinhala" not in str(f).lower()
    ]


def iter_text_artists(fig) -> Iterator[Any]:
    """Yield every matplotlib ``Text`` on ``fig`` (titles, ticks, legend, figtext)."""
    from matplotlib.text import Text

    seen: set[int] = set()
    stack: list[Any] = [fig]
    for ax in getattr(fig, "axes", []):
        stack.append(ax)
        for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None)):
            if axis is None:
                continue
            for meth in ("get_label", "get_offset_text", "get_ticklabels", "get_minorticklabels"):
                fn = getattr(axis, meth, None)
                if not callable(fn):
                    continue
                try:
                    got = fn()
                except Exception:
                    continue
                if isinstance(got, (list, tuple)):
                    stack.extend(got)
                else:
                    stack.append(got)
    while stack:
        artist = stack.pop()
        if artist is None:
            continue
        aid = id(artist)
        if aid in seen:
            continue
        seen.add(aid)
        if isinstance(artist, Text):
            yield artist
        getter = getattr(artist, "get_children", None)
        if callable(getter):
            try:
                stack.extend(getter() or [])
            except Exception:
                pass
        for extra_name in ("_suptitle", "legend_", "_legend", "texts"):
            extra = getattr(artist, extra_name, None)
            if extra is None:
                continue
            if extra_name == "texts":
                try:
                    stack.extend(list(extra))
                except TypeError:
                    pass
            else:
                stack.append(extra)


def apply_latin_font(fig) -> None:
    """Bake DejaVu/Liberation (by file path) onto every text artist.

    IPython's ``fig.canvas.print_figure`` redraws after the plot function
    returns, so family-name FontProperties are not enough — ``fname`` must be
    set on each Text. Tick labels recreated at draw time pick up ``rcParams``,
    so this also re-applies ``ensure_latin_plot_fonts``.
    """
    ensure_latin_plot_fonts()
    prop = latin_fontproperties()
    name = _latin_font_name or "DejaVu Sans"
    for t in iter_text_artists(fig):
        try:
            t.set_fontproperties(prop)
        except Exception:
            pass
    for ax in getattr(fig, "axes", []):
        try:
            ax.tick_params(axis="both", which="both", labelfontfamily=name)
        except (TypeError, ValueError, AttributeError):
            pass


def bake_latin_figure(fig) -> None:
    """Alias of ``apply_latin_font`` for notebook cells (stale-Drive tolerant)."""
    apply_latin_font(fig)


@contextlib.contextmanager
def use_latin_plots() -> Iterator[None]:
    """Force Latin fonts for English metric plotting (rc_context + persistent rc).

    ``rc_context`` covers the block. After exit we **re-apply** DejaVu as the
    session default so IPython ``print_figure`` cannot fall back to Noto.
    """
    import matplotlib.pyplot as plt

    ensure_latin_plot_fonts()
    with plt.rc_context(_latin_rc()):
        ensure_latin_plot_fonts()
        yield
    ensure_latin_plot_fonts()


def reload_plot_modules():
    """``importlib.reload`` display + train_curves (picks up a freshly copied Drive tree)."""
    import importlib

    import src.evaluation.train_curves as train_curves
    import src.utils.display as display

    display = importlib.reload(display)
    train_curves = importlib.reload(train_curves)
    display.ensure_latin_plot_fonts()
    return display, train_curves


def display_sinhala_table(
    rows: Sequence[Sequence[str]],
    headers: Optional[Sequence[str]] = None,
    font_path: Optional[str] = None,
) -> None:
    """Render a small HTML table with Sinhala-friendly font styling."""
    from IPython.display import HTML, display

    family = sinhala_font_css_family(font_path)
    parts = [
        f'<table style="font-family: {family}; border-collapse: collapse;">',
    ]
    if headers:
        parts.append("<thead><tr>")
        for h in headers:
            parts.append(
                f'<th style="border: 1px solid #ccc; padding: 4px 8px;">{html.escape(h)}</th>'
            )
        parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(
                f'<td style="border: 1px solid #ccc; padding: 4px 8px;">{html.escape(str(cell))}</td>'
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    display(HTML("".join(parts)))
