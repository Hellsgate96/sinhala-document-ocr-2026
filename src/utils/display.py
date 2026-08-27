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
from typing import Iterator, Optional, Sequence

from src.utils.common import configure_stdout_utf8

_registered_font_path: Optional[str] = None
_registered_font_name: Optional[str] = None

# Latin-capable stack for English metric plots (titles, axes, legends).
_LATIN_SANS_SERIF = (
    "DejaVu Sans",
    "Bitstream Vera Sans",
    "Arial",
    "Helvetica",
    "sans-serif",
)


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
    """Register a Sinhala font with matplotlib (for rare Sinhala plot labels).

    By default this does **not** change the global ``font.family`` — Sinhala OCR
    demos use print/HTML for predictions, and English metric plots need a
    Latin-capable face. Pass ``set_as_default=True`` only when deliberately
    drawing Sinhala text with matplotlib.
    """
    global _registered_font_path, _registered_font_name

    import matplotlib.pyplot as plt
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
        # Prefer Latin fallbacks first so English labels still render if this
        # font lacks Latin glyphs (Noto Sans Sinhala typically does).
        plt.rcParams["font.family"] = "sans-serif"
        sans = list(_LATIN_SANS_SERIF)
        if name not in sans:
            sans.append(name)
        plt.rcParams["font.sans-serif"] = sans

    return _registered_font_path


def ensure_latin_plot_fonts() -> None:
    """Set matplotlib rcParams to a Latin-capable sans-serif stack.

    Prefer this for English metric plots. Matplotlib often resolves glyph fonts
    lazily at draw time, so a temporary ``rc_context`` alone is not enough if
    the figure is shown after the context exits.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = list(_LATIN_SANS_SERIF)


def apply_latin_font(fig) -> None:
    """Bake DejaVu Sans (or first available Latin face) onto all text artists."""
    from matplotlib import font_manager

    prop = font_manager.FontProperties(family="DejaVu Sans")
    texts = []
    if getattr(fig, "_suptitle", None) is not None:
        texts.append(fig._suptitle)
    for ax in fig.axes:
        texts.extend(
            [
                ax.title,
                ax.xaxis.label,
                ax.yaxis.label,
                *ax.get_xticklabels(),
                *ax.get_yticklabels(),
                *ax.texts,
            ]
        )
        leg = ax.get_legend()
        if leg is not None:
            texts.extend(leg.get_texts())
            if leg.get_title() is not None:
                texts.append(leg.get_title())
    for t in texts:
        if t is not None:
            t.set_fontproperties(prop)


@contextlib.contextmanager
def use_latin_plots() -> Iterator[None]:
    """Force Latin fonts for the duration of English metric plotting.

    Sets rcParams for the block and does **not** restore a prior Sinhala face —
    English plots must keep a Latin-capable default for later Jupyter redraws.
    """
    ensure_latin_plot_fonts()
    yield


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
