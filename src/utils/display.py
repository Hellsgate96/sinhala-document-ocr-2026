"""Jupyter / matplotlib helpers for Sinhala Unicode in notebooks."""

from __future__ import annotations

import glob
import html
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.utils.common import configure_stdout_utf8

_registered_font_path: Optional[str] = None
_registered_font_name: Optional[str] = None


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


def setup_matplotlib_sinhala(font_path: Optional[str] = None) -> Optional[str]:
    """Register a Sinhala font with matplotlib and set it as the default family."""
    global _registered_font_path, _registered_font_name

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    path = find_sinhala_font(font_path)
    if path is None:
        warnings.warn(
            "No Sinhala-capable font found for matplotlib. "
            "Titles and labels may show missing glyphs (tofu). "
            "On Windows use Nirmala UI; on Linux install fonts-noto-core or "
            "run scripts/download_fonts.ps1 to fetch Noto Sans Sinhala into fonts/.",
            UserWarning,
            stacklevel=2,
        )
        return None

    font_manager.fontManager.addfont(str(path))
    prop = font_manager.FontProperties(fname=str(path))
    name = prop.get_name()
    plt.rcParams["font.family"] = name
    sans = [name]
    for fallback in ("DejaVu Sans", "sans-serif"):
        if fallback not in sans:
            sans.append(fallback)
    plt.rcParams["font.sans-serif"] = sans

    _registered_font_path = str(path)
    _registered_font_name = name
    return _registered_font_path


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
