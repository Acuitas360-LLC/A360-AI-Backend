"""
deck_creator_agent_2.py
=======================
Theme-aware, self-fitting executive deck generator for the commercial
analytics conversation stack.

Pipeline
--------
    messages  ->  parse_conversation   ->  blocks
              ->  profile_block        ->  deterministic data facts
              ->  generate_slide_content (LLM, verified against facts)
              ->  render_chart         ->  PNG at exact panel aspect
              ->  LayoutEngine         ->  slide
              ->  Presentation         ->  .pptx

Design principles carried over from the platform
------------------------------------------------
* Deterministic first. Python enforces correctness (fit, contrast, number
  formatting, superlative verification). The prompt is a nudge, never the
  sole enforcement mechanism.
* Theme is read by following the OOXML relationship chain
  slide -> slideLayout -> slideMaster -> theme. Never alphabetical theme1.xml.
* Nothing is written to a slide that has not been measured first.

Quick start
-----------
    from deck_creator_agent_2 import build_ppt
    build_ppt(messages)                      # drop-in replacement

    from deck_creator_agent_2 import DeckConfig, build_deck
    build_deck(messages, DeckConfig(
        template_path="aadibio_ppt.pptx",
        logo_path="aadi-logo.png",
        output_path="review.pptx",
    ))
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import textwrap
import unicodedata
import uuid
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

warnings.filterwarnings("ignore")

log = logging.getLogger("deck_creator")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[deck] %(levelname)s %(message)s"))
    log.addHandler(_h)
log.setLevel(os.environ.get("DECK_LOG_LEVEL", "INFO"))

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROMPT_DEBUG_REMAINING = 4

def _mojibake(symbol: str) -> str:
    return symbol.encode("utf-8").decode("latin-1")


def _mojibake_variants(symbol: str) -> set[str]:
    raw = symbol.encode("utf-8")
    variants = {raw.decode("latin-1")}
    try:
        variants.add(raw.decode("cp1252"))
    except UnicodeDecodeError:
        pass
    return variants


_MOJIBAKE_CHAR_REPLACEMENTS = {}
for _symbol in ("═", "─", "—", "–", "−", "…", "“", "”", "‘", "’", "→", "✔", "≤", "·"):
    for _bad in _mojibake_variants(_symbol):
        _MOJIBAKE_CHAR_REPLACEMENTS[_bad] = _symbol

_SUSPECT_MOJIBAKE_RE = re.compile(r"[ÂÃâ][^\s]{0,3}")


def _resolve_local_asset_path(path: str | None) -> str | None:
    candidate = str(path or "").strip()
    if not candidate:
        return None
    if os.path.isabs(candidate):
        return candidate if os.path.exists(candidate) else None

    search_roots = (
        MODULE_DIR,
        os.path.dirname(MODULE_DIR),
        os.path.dirname(os.path.dirname(MODULE_DIR)),
    )
    for root in search_roots:
        resolved = os.path.abspath(os.path.join(root, candidate))
        if os.path.exists(resolved):
            return resolved
    return None


def _repair_mojibake_text(text: str) -> str:
    if not text:
        return text
    repaired = text
    if _SUSPECT_MOJIBAKE_RE.search(repaired):
        for encoding in ("cp1252", "latin-1"):
            try:
                candidate = repaired.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if candidate and sum(candidate.count(ch) for ch in "ÂÃâ") < sum(
                    repaired.count(ch) for ch in "ÂÃâ"):
                repaired = candidate
                break
    for bad, good in _MOJIBAKE_CHAR_REPLACEMENTS.items():
        repaired = repaired.replace(bad, good)
    return repaired


def _strip_markdown_display(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    return text


def _normalise_display_text(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = _repair_mojibake_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = _strip_markdown_display(text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normalise_rich_text_fragment(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = _repair_mojibake_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    return text


def _debug_excerpt(value: Any, limit: int = 180) -> str:
    text = _normalise_display_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip(" ,;:.") + "…"


def _debug_list_preview(values: Sequence[Any], limit: int = 4) -> str:
    items = [_debug_excerpt(value, 40) for value in list(values or [])[:limit]]
    if len(values or []) > limit:
        items.append("…")
    return ", ".join(item for item in items if item)


def _env_nonempty(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _debug_prompt_payload_once(question: str, summary: str, facts: str,
                               columns: Sequence[str], has_chart: bool) -> None:
    global _PROMPT_DEBUG_REMAINING
    if _PROMPT_DEBUG_REMAINING <= 0:
        return
    _PROMPT_DEBUG_REMAINING -= 1
    dump = "\n".join([
        "[PPT PROMPT DEBUG] BEGIN",
        "<question>",
        _normalise_display_text(question),
        "</question>",
        "<summary>",
        _normalise_display_text(summary),
        "</summary>",
        "<facts>",
        _normalise_display_text(facts),
        "</facts>",
        "<sql_columns>",
        json.dumps([_normalise_display_text(col) for col in list(columns or [])],
                   ensure_ascii=False, indent=2),
        "</sql_columns>",
        f"<has_chart>{str(bool(has_chart)).lower()}</has_chart>",
        "[PPT PROMPT DEBUG] END",
    ])
    for line in dump.splitlines():
        log.info("%s", line)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL DEPENDENCIES  (import failures must never break the module)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pandas as pd
except Exception:                                            # pragma: no cover
    pd = None

try:
    import numpy as np
except Exception:                                            # pragma: no cover
    np = None

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Inches, Pt

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:                                            # pragma: no cover
    pass


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DeckHooks:
    """
    Helpers injected by the host app so the deck reuses its logic instead of
    duplicating it. Every hook is optional and falls back to the built-in.

        summary_splitter(text)   -> (overview, takeaways)
        period_collapser(df)     -> (display_df, [(label, value), ...])
        viz_sanitizer(code)      -> code
        viz_scope_builder(df)    -> dict   (exec globals)
        viz_layout(fig, df, ...) -> fig    (for_slide=True passed when accepted)
        viz_label_thinner(fig, w_px, h_px) -> fig
    """

    summary_splitter: Any = None
    period_collapser: Any = None
    viz_sanitizer: Any = None
    viz_scope_builder: Any = None
    viz_layout: Any = None
    # Re-runs the line-label collision test against the real slide rectangle.
    # Left unset by a host app, it is resolved from viz_polish directly.
    viz_label_thinner: Any = None

    def describe(self) -> str:
        on = [n for n in ("summary_splitter", "period_collapser", "viz_sanitizer",
                          "viz_scope_builder", "viz_layout", "viz_label_thinner")
              if getattr(self, n)]
        return ", ".join(on) or "none"


def autodiscover_hooks() -> DeckHooks:
    """Pick up viz_polish_2 automatically when it is on the import path."""
    hooks = DeckHooks()
    for name in ("viz_polish_2", "viz_polish"):
        try:
            module = __import__(name)
        except Exception:
            continue
        hooks.viz_sanitizer = getattr(module, "sanitize_viz_code", None)
        hooks.viz_scope_builder = getattr(module, "build_exec_scope", None)
        hooks.viz_layout = getattr(module, "apply_classic_layout", None)
        hooks.viz_label_thinner = getattr(module, "thin_line_labels", None)
        log.info("deck hooks from %s: %s", name, hooks.describe())
        break
    return hooks


@dataclass
class DeckConfig:
    """Everything tunable in one place. No module-level globals."""

    # inputs
    template_path: str | None = "aadibio_ppt.pptx"
    logo_path: str | None = "aadibio_logo.png"
    output_path: str = "final_presentation.pptx"

    # canvas — None means "inherit from template, else 13.333 x 7.5"
    slide_width_in: float | None = None
    slide_height_in: float | None = None

    # deck furniture
    # One question -> exactly one slide. The cover / agenda / closing slides
    # are opt-in; nothing extra is emitted unless explicitly turned on.
    cover_slide: bool = False
    agenda_slide: bool = False
    closing_slide: bool = False
    deck_title: str = "Commercial Performance Review"
    deck_subtitle: str = "Demand, momentum and field execution"
    brand_name: str = ""
    confidential_note: str = "Confidential — for internal use only"
    default_footnote: str = ""

    # content
    max_bullets: int = 3
    max_kpis: int = 3
    min_bullets: int = 1
    min_kpis: int = 1
    show_eyebrow: bool = False
    show_footnote: bool = False        # data caveats stay off the slide
    show_basis: bool = False           # demand basis stays off the slide
    speaker_notes: bool = False        # notes pane ships empty
    findings_label: str = "Key Findings"
    takeaways_label: str = "Key Takeaways"
    max_table_rows: int = 8
    show_definitions: bool = True
    sentiment_colors: bool = True

    # LLM
    llm_provider: str = "auto"          # auto | anthropic | openai | none
    llm_model: str | None = None
    llm_temperature: float = 0.0
    llm_max_retries: int = 2
    verify_numbers: bool = True

    # chart export
    chart_scale: int = 2
    chart_max_px: int = 2400
    workdir: str = ".deck_build"
    # Charts are styled by viz_polish_2 so a slide and the screen apply the
    # same rules: pinned month ticks on date axes, the point-budget gate on
    # data labels, trace styling. The deck then fits the result to the panel
    # and re-applies its own percentage / whole-number axis formatting.
    use_viz_polish_layout: bool = True
    # True = viz_polish's own default: show data labels and let its
    # geometry-based collision test drop the ones that would overlap.
    # (uniformtext only hides bar and pie labels, never scatter, which is why
    # that collision pass exists at all.)
    chart_force_labels: bool = True

    # shared helpers injected by the host app (see DeckHooks)
    hooks: Any = field(default_factory=autodiscover_hooks)

    # layout
    dark_cover: bool = True
    slide_numbers: bool = True
    # Brand mark in the bottom-right corner of every slide. Uses logo_path.
    footer_logo: bool = True
    footer_logo_w: float = 1.15          # max width in reference inches
    footer_logo_h: float = 0.34          # max height in reference inches


# ══════════════════════════════════════════════════════════════════════════════
# COLOR SCIENCE
# ══════════════════════════════════════════════════════════════════════════════


def hex_to_rgb(value: str | None) -> RGBColor | None:
    if not value:
        return None
    h = str(value).strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) == 8:                      # AARRGGBB / RRGGBBAA — drop alpha
        h = h[:6] if h[6:].lower() in ("ff", "00") else h[2:]
    if len(h) != 6:
        return None
    try:
        return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return None


def rgb_to_hex(c: RGBColor) -> str:
    return f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}"


def _srgb_channel(v: int) -> float:
    s = v / 255.0
    return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4


def luminance(c: RGBColor) -> float:
    """WCAG relative luminance."""
    return (0.2126 * _srgb_channel(c[0])
            + 0.7152 * _srgb_channel(c[1])
            + 0.0722 * _srgb_channel(c[2]))


def contrast_ratio(a: RGBColor, b: RGBColor) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def lighten(c: RGBColor, t: float) -> RGBColor:
    t = _clamp(t, 0.0, 1.0)
    return RGBColor(*[int(round(v + (255 - v) * t)) for v in c])


def darken(c: RGBColor, t: float) -> RGBColor:
    t = _clamp(t, 0.0, 1.0)
    return RGBColor(*[int(round(v * (1 - t))) for v in c])


def mix(a: RGBColor, b: RGBColor, t: float = 0.5) -> RGBColor:
    t = _clamp(t, 0.0, 1.0)
    return RGBColor(*[int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3)])


def saturation(c: RGBColor) -> float:
    mx, mn = max(c) / 255.0, min(c) / 255.0
    if mx == 0:
        return 0.0
    return (mx - mn) / mx


def hue(c: RGBColor) -> float:
    r, g, b = [v / 255.0 for v in c]
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0:
        return 0.0
    if mx == r:
        h = ((g - b) / d) % 6
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h * 60.0


def ensure_contrast(fg: RGBColor, bg: RGBColor, min_ratio: float = 4.5,
                    max_steps: int = 28) -> RGBColor:
    """Nudge *fg* darker (on light bg) or lighter (on dark bg) until readable."""
    if contrast_ratio(fg, bg) >= min_ratio:
        return fg
    going_dark = luminance(bg) > 0.45
    out = fg
    for _ in range(max_steps):
        out = darken(out, 0.07) if going_dark else lighten(out, 0.07)
        if contrast_ratio(out, bg) >= min_ratio:
            return out
        if (going_dark and sum(out) == 0) or (not going_dark and sum(out) == 765):
            break
    return RGBColor(0x11, 0x11, 0x11) if going_dark else RGBColor(0xFF, 0xFF, 0xFF)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def color_distance(a: RGBColor, b: RGBColor) -> float:
    """Cheap perceptual-ish distance; good enough to dedupe a theme palette."""
    rm = (a[0] + b[0]) / 2
    dr, dg, db = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt((2 + rm / 256) * dr * dr + 4 * dg * dg
                     + (2 + (255 - rm) / 256) * db * db)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT METRICS  —  the reason everything fits
# ══════════════════════════════════════════════════════════════════════════════
#
# Adobe Helvetica AFM advance widths (units per 1000 em). Arial is metric
# compatible with Helvetica, and every other common UI font is expressed as a
# scale factor against it. This gives deterministic, dependency-free text
# measurement: no PIL, no font files, no environment drift.

_W_REG = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667,
    "'": 191, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, "0": 556, "1": 556, "2": 556, "3": 556, "4": 556,
    "5": 556, "6": 556, "7": 556, "8": 556, "9": 556, ":": 278, ";": 278,
    "<": 584, "=": 584, ">": 584, "?": 556, "@": 1015, "A": 667, "B": 667,
    "C": 722, "D": 722, "E": 667, "F": 611, "G": 778, "H": 722, "I": 278,
    "J": 500, "K": 667, "L": 556, "M": 833, "N": 722, "O": 778, "P": 667,
    "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 278, "\\": 278, "]": 278, "^": 469,
    "_": 556, "`": 333, "a": 556, "b": 556, "c": 500, "d": 556, "e": 556,
    "f": 278, "g": 556, "h": 556, "i": 222, "j": 222, "k": 500, "l": 222,
    "m": 833, "n": 556, "o": 556, "p": 556, "q": 556, "r": 333, "s": 500,
    "t": 278, "u": 556, "v": 500, "w": 722, "x": 500, "y": 500, "z": 500,
    "{": 334, "|": 260, "}": 334, "~": 584,
}

_W_BOLD = {
    " ": 278, "!": 333, '"': 474, "#": 556, "$": 556, "%": 889, "&": 722,
    "'": 238, "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333,
    ".": 278, "/": 278, "0": 556, "1": 556, "2": 556, "3": 556, "4": 556,
    "5": 556, "6": 556, "7": 556, "8": 556, "9": 556, ":": 333, ";": 333,
    "<": 584, "=": 584, ">": 584, "?": 611, "@": 975, "A": 722, "B": 722,
    "C": 722, "D": 722, "E": 667, "F": 611, "G": 778, "H": 722, "I": 278,
    "J": 556, "K": 722, "L": 611, "M": 833, "N": 722, "O": 778, "P": 667,
    "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 333, "\\": 278, "]": 333, "^": 584,
    "_": 556, "`": 333, "a": 556, "b": 611, "c": 556, "d": 611, "e": 556,
    "f": 333, "g": 611, "h": 611, "i": 278, "j": 278, "k": 556, "l": 278,
    "m": 889, "n": 611, "o": 611, "p": 611, "q": 611, "r": 389, "s": 556,
    "t": 333, "u": 611, "v": 556, "w": 778, "x": 556, "y": 556, "z": 500,
    "{": 389, "|": 280, "}": 389, "~": 584,
}

# Width relative to Helvetica/Arial at the same point size.
_FONT_FACTOR = {
    "arial": 1.00, "helvetica": 1.00, "liberation sans": 1.00,
    "arial narrow": 0.82, "arial black": 1.10,
    "calibri": 0.91, "calibri light": 0.91, "aptos": 0.92,
    "aptos display": 0.92, "aptos narrow": 0.86,
    "segoe ui": 0.96, "segoe ui light": 0.96, "tahoma": 1.00,
    "verdana": 1.09, "trebuchet ms": 0.96, "corbel": 0.92,
    "candara": 0.93, "franklin gothic book": 0.94, "gill sans mt": 0.90,
    "century gothic": 1.05, "lato": 0.95, "open sans": 0.97,
    "roboto": 0.96, "source sans pro": 0.93, "noto sans": 0.99,
    "times new roman": 0.89, "cambria": 0.94, "georgia": 1.02,
    "garamond": 0.83, "book antiqua": 0.97, "palatino linotype": 0.97,
    "bookman old style": 1.03, "century schoolbook": 1.01,
    "courier new": 1.08, "consolas": 0.92,
}

_DEFAULT_FACTOR = 1.00           # unknown font -> assume Arial width (safe/wide)
_WIDTH_SAFETY = 1.02             # headroom for renderer/metric differences


def font_factor(font_name: str | None) -> float:
    if not font_name:
        return _DEFAULT_FACTOR
    return _FONT_FACTOR.get(str(font_name).strip().lower(), _DEFAULT_FACTOR)


def text_width_in(text: str, size_pt: float, *, bold: bool = False,
                  font: str | None = None, tracking_pt: float = 0.0) -> float:
    """Advance width of *text* in inches at *size_pt*."""
    if not text:
        return 0.0
    table = _W_BOLD if bold else _W_REG
    fallback = 600 if bold else 556
    units = sum(table.get(ch, fallback) for ch in text)
    width_pt = units / 1000.0 * size_pt * font_factor(font)
    width_pt += tracking_pt * max(len(text) - 1, 0)
    return width_pt / 72.0 * _WIDTH_SAFETY


def wrap_lines(text: str, max_width_in: float, size_pt: float, *,
               bold: bool = False, font: str | None = None,
               tracking_pt: float = 0.0, max_lines: int | None = None,
               ellipsis: bool = True) -> list[str]:
    """Greedy word wrap using real advance widths. Breaks over-long words."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    if max_width_in <= 0:
        return [text]

    def w(s: str) -> float:
        return text_width_in(s, size_pt, bold=bold, font=font,
                             tracking_pt=tracking_pt)

    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if w(candidate) <= max_width_in or not current:
            if w(candidate) <= max_width_in:
                current = candidate
                continue
            # single word longer than the line — hard-break it
            chunk = ""
            for ch in word:
                if w(chunk + ch) <= max_width_in or not chunk:
                    chunk += ch
                else:
                    lines.append(chunk)
                    chunk = ch
            current = chunk
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    if max_lines is not None and len(lines) > max_lines:
        kept = lines[:max_lines]
        if ellipsis and kept:
            last = kept[-1]
            while last and w(last + "…") > max_width_in:
                last = last[:-1]
            kept[-1] = last.rstrip(" ,;:") + "…"
        lines = kept
    return lines


def line_height_in(size_pt: float, line_spacing: float = 1.16) -> float:
    return size_pt * line_spacing / 72.0


def fit_text(text: str, box_w_in: float, box_h_in: float, *,
             max_pt: float, min_pt: float, bold: bool = False,
             font: str | None = None, line_spacing: float = 1.16,
             tracking_pt: float = 0.0, step: float = 0.5,
             hard_max_lines: int | None = None) -> tuple[float, list[str]]:
    """
    Largest point size in [min_pt, max_pt] whose wrapped text fits the box.

    Returns (size_pt, lines). If even min_pt overflows, the text is truncated
    with an ellipsis so nothing ever spills past its container.
    """
    size = max_pt
    while size >= min_pt:
        lines = wrap_lines(text, box_w_in, size, bold=bold, font=font,
                           tracking_pt=tracking_pt)
        max_lines = int(box_h_in // line_height_in(size, line_spacing))
        if hard_max_lines is not None:
            max_lines = min(max_lines, hard_max_lines)
        # max_lines == 0 means even one line is taller than the box: keep shrinking.
        if lines and max_lines >= 1 and len(lines) <= max_lines:
            return size, lines
        size = round(size - step, 2)

    size = min_pt
    max_lines = max(int(box_h_in // line_height_in(size, line_spacing)), 1)
    if hard_max_lines is not None:
        max_lines = min(max_lines, hard_max_lines)
    lines = wrap_lines(text, box_w_in, size, bold=bold, font=font,
                       tracking_pt=tracking_pt, max_lines=max_lines)
    return size, lines


def text_block_height_in(lines: Sequence[str], size_pt: float,
                         line_spacing: float = 1.16) -> float:
    return len(lines) * line_height_in(size_pt, line_spacing)


# ══════════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════════

_DML = "http://schemas.openxmlformats.org/drawingml/2006/main"
_PML = "http://schemas.openxmlformats.org/presentationml/2006/main"
_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_ORel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def qn_a(tag: str) -> str:
    """DrawingML qualified tag name."""
    return f"{{{_DML}}}{tag}"

_SAFE_BODY_FONTS = {"arial", "calibri", "cambria", "times new roman",
                    "courier new", "bookman old style", "century schoolbook",
                    "segoe ui", "verdana", "tahoma"}


@dataclass
class Theme:
    """Semantic palette + typography. Dict-compatible with the v1 theme."""

    # surfaces
    bg: RGBColor = RGBColor(0xFF, 0xFF, 0xFF)
    bg_dark: RGBColor = RGBColor(0x0D, 0x1B, 0x3E)
    surface: RGBColor = RGBColor(0xF4, 0xF7, 0xFB)
    surface_alt: RGBColor = RGBColor(0xEC, 0xF1, 0xF8)
    hairline: RGBColor = RGBColor(0xD8, 0xE0, 0xEA)

    # ink
    ink: RGBColor = RGBColor(0x14, 0x1C, 0x2E)
    ink_soft: RGBColor = RGBColor(0x3D, 0x4A, 0x60)
    ink_muted: RGBColor = RGBColor(0x6B, 0x77, 0x8A)
    ink_inverse: RGBColor = RGBColor(0xFF, 0xFF, 0xFF)

    # brand
    primary: RGBColor = RGBColor(0x1B, 0x50, 0x8F)
    accents: list = field(default_factory=list)
    chart_colors: list = field(default_factory=list)
    positive: RGBColor = RGBColor(0x1E, 0x7F, 0x4E)
    negative: RGBColor = RGBColor(0xB3, 0x26, 0x1E)
    neutral: RGBColor = RGBColor(0x6B, 0x77, 0x8A)

    # type
    font_head: str = "Calibri"
    font_body: str = "Calibri"

    # provenance
    source: str = "default"
    slide_w_in: float | None = None
    slide_h_in: float | None = None

    # ── legacy dict interface ────────────────────────────────────────────
    _LEGACY = {
        "background": "bg", "title_text": "ink", "body_text": "ink_soft",
        "muted_text": "ink_muted", "panel_bg": "surface", "kpi_bg": "surface",
        "kpi_border": "hairline", "kpi_label_text": "primary",
        "subtitle_label": "primary", "bullet_dot": "primary",
        "divider": "primary", "insight_label": "primary",
        "insight_text": "ink_soft", "font_title": "font_head",
        "font_body": "font_body",
    }

    def accent(self, i: int) -> RGBColor:
        pool = self.accents or [self.primary]
        return pool[i % len(pool)]

    def chart_color(self, i: int) -> RGBColor:
        pool = self.chart_colors or self.accents or [self.primary]
        return pool[i % len(pool)]

    def __getitem__(self, key: str) -> Any:
        if key in self._LEGACY:
            return getattr(self, self._LEGACY[key])
        if key.startswith("accent") and key[6:].isdigit():
            return self.accent(int(key[6:]) - 1)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default

    def describe(self) -> str:
        return (f"source={self.source} primary={rgb_to_hex(self.primary)} "
                f"ink={rgb_to_hex(self.ink)} dark={rgb_to_hex(self.bg_dark)} "
                f"accents={[rgb_to_hex(c) for c in self.accents]} "
                f"fonts={self.font_head}/{self.font_body}")


# ── OOXML relationship helpers ───────────────────────────────────────────────

def _resolve_part(source_part: str, target: str) -> str:
    """Resolve a relationship target against the part that declares it."""
    if target.startswith("/"):
        return target.lstrip("/")
    base = os.path.dirname(source_part)
    return os.path.normpath(os.path.join(base, target)).replace("\\", "/")


def _rels_path(part: str) -> str:
    d, f = os.path.dirname(part), os.path.basename(part)
    return f"{d}/_rels/{f}.rels" if d else f"_rels/{f}.rels"


def _read_rels(zf: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """rId -> (resolved_target, type). External targets are skipped."""
    import xml.etree.ElementTree as ET

    path = _rels_path(part)
    if path not in zf.namelist():
        return {}
    out: dict[str, tuple[str, str]] = {}
    try:
        root = ET.fromstring(zf.read(path))
    except Exception:
        return {}
    for rel in root.findall(f"{{{_REL}}}Relationship"):
        if rel.get("TargetMode") == "External":
            continue
        rid, tgt, typ = rel.get("Id"), rel.get("Target", ""), rel.get("Type", "")
        if rid and tgt:
            out[rid] = (_resolve_part(part, tgt), typ)
    return out


def _rels_of_type(rels: dict[str, tuple[str, str]], suffix: str) -> list[str]:
    return [t for t, typ in rels.values() if typ.endswith(suffix)]


def _find_theme_part(zf: zipfile.ZipFile) -> tuple[str | None, str | None]:
    """
    Follow the real chain: slide1 -> slideLayout -> slideMaster -> theme.
    Falls back to presentation -> first slideMaster -> theme.
    Returns (theme_part, master_part).
    """
    import xml.etree.ElementTree as ET

    names = set(zf.namelist())
    master = None

    # Preferred: whatever theme slide 1 actually renders with.
    slides = sorted(
        (n for n in names
         if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
        key=lambda n: int(re.search(r"(\d+)", os.path.basename(n)).group(1)),
    )
    if slides:
        layouts = _rels_of_type(_read_rels(zf, slides[0]), "slideLayout")
        if layouts:
            masters = _rels_of_type(_read_rels(zf, layouts[0]), "slideMaster")
            if masters:
                master = masters[0]

    # Fallback: presentation.xml -> sldMasterIdLst order.
    if master is None and "ppt/presentation.xml" in names:
        rels = _read_rels(zf, "ppt/presentation.xml")
        try:
            root = ET.fromstring(zf.read("ppt/presentation.xml"))
            lst = root.find(f"{{{_PML}}}sldMasterIdLst")
            if lst is not None:
                for node in lst:
                    rid = node.get(f"{{{_ORel}}}id")
                    if rid and rid in rels:
                        master = rels[rid][0]
                        break
        except Exception:
            pass
        if master is None:
            m = _rels_of_type(rels, "slideMaster")
            master = m[0] if m else None

    if master is None:
        cand = sorted(n for n in names
                      if re.fullmatch(r"ppt/slideMasters/slideMaster\d+\.xml", n))
        master = cand[0] if cand else None

    theme = None
    if master:
        th = _rels_of_type(_read_rels(zf, master), "theme")
        theme = th[0] if th else None
    if theme is None:
        cand = sorted(n for n in names
                      if re.fullmatch(r"ppt/theme/theme\d+\.xml", n))
        theme = cand[0] if cand else None
    return theme, master


def _parse_clr_scheme(zf: zipfile.ZipFile, theme_part: str) -> dict[str, RGBColor]:
    import xml.etree.ElementTree as ET

    out: dict[str, RGBColor] = {}
    try:
        root = ET.fromstring(zf.read(theme_part))
    except Exception:
        return out
    scheme = root.find(f".//{{{_DML}}}clrScheme")
    if scheme is None:
        return out
    for child in scheme:
        slot = child.tag.split("}")[-1]
        srgb = child.find(f"{{{_DML}}}srgbClr")
        sysc = child.find(f"{{{_DML}}}sysClr")
        c = None
        if srgb is not None:
            c = hex_to_rgb(srgb.get("val", ""))
        elif sysc is not None:
            c = hex_to_rgb(sysc.get("lastClr", "")) or (
                RGBColor(0, 0, 0) if sysc.get("val") == "windowText"
                else RGBColor(0xFF, 0xFF, 0xFF))
        if c is not None:
            out[slot] = c
    return out


def _parse_font_scheme(zf: zipfile.ZipFile, theme_part: str) -> tuple[str | None, str | None]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(zf.read(theme_part))
    except Exception:
        return None, None
    fs = root.find(f".//{{{_DML}}}fontScheme")
    if fs is None:
        return None, None

    def _pick(kind: str) -> str | None:
        node = fs.find(f"{{{_DML}}}{kind}/{{{_DML}}}latin")
        if node is None:
            return None
        tf = (node.get("typeface") or "").strip()
        return tf if tf and not tf.startswith("+") else None

    return _pick("majorFont"), _pick("minorFont")


def _parse_clr_map(zf: zipfile.ZipFile, master_part: str | None) -> dict[str, str]:
    """<p:clrMap> tells us which scheme slot each semantic role points at."""
    import xml.etree.ElementTree as ET

    default = {"bg1": "lt1", "tx1": "dk1", "bg2": "lt2", "tx2": "dk2"}
    if not master_part:
        return default
    try:
        root = ET.fromstring(zf.read(master_part))
    except Exception:
        return default
    node = root.find(f"{{{_PML}}}clrMap")
    if node is None:
        return default
    return {k: v for k, v in node.attrib.items()} or default


def _dominant_explicit_font(zf: zipfile.ZipFile) -> str | None:
    """Most frequent literal <a:latin typeface> across master, layouts, slides."""
    import xml.etree.ElementTree as ET

    counts: dict[str, int] = {}
    names = zf.namelist()
    parts = ([n for n in names if "slideMasters/slideMaster" in n and n.endswith(".xml")]
             + sorted(n for n in names if "slideLayouts/slideLayout" in n
                      and n.endswith(".xml"))[:6]
             + sorted(n for n in names
                      if re.fullmatch(r"ppt/slides/slide\d+\.xml", n))[:12])
    for part in parts:
        try:
            root = ET.fromstring(zf.read(part))
        except Exception:
            continue
        for latin in root.iter(f"{{{_DML}}}latin"):
            tf = (latin.get("typeface") or "").strip()
            if tf and not tf.startswith("+"):
                counts[tf] = counts.get(tf, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _slide_size_in(zf: zipfile.ZipFile) -> tuple[float, float] | None:
    import xml.etree.ElementTree as ET

    if "ppt/presentation.xml" not in zf.namelist():
        return None
    try:
        root = ET.fromstring(zf.read("ppt/presentation.xml"))
        node = root.find(f"{{{_PML}}}sldSz")
        if node is None:
            return None
        return int(node.get("cx")) / 914400, int(node.get("cy")) / 914400
    except Exception:
        return None


def _build_palette(raw: dict[str, RGBColor], clr_map: dict[str, str]) -> dict:
    """Map raw scheme slots -> semantic roles, enforcing readability."""
    white = RGBColor(0xFF, 0xFF, 0xFF)

    def slot(role: str, fallback: str) -> RGBColor | None:
        return raw.get(clr_map.get(role, fallback))

    tx1 = slot("tx1", "dk1") or RGBColor(0x14, 0x1C, 0x2E)
    tx2 = slot("tx2", "dk2")

    # Some templates map tx1 to a light color (dark-background decks). We render
    # on white, so flip to the darkest available scheme color instead.
    if luminance(tx1) > 0.5:
        pool = [c for c in raw.values() if luminance(c) < 0.35]
        tx1 = min(pool, key=luminance) if pool else RGBColor(0x14, 0x1C, 0x2E)

    ink = ensure_contrast(tx1, white, 8.5)
    ink_soft = ensure_contrast(mix(ink, white, 0.18), white, 6.5)
    ink_muted = ensure_contrast(mix(ink, white, 0.42), white, 4.5)

    # Accents, in scheme order, deduped perceptually.
    ordered: list[RGBColor] = []
    for key in ("accent1", "accent2", "accent3", "accent4", "accent5", "accent6"):
        c = raw.get(key)
        if c is None:
            continue
        if all(color_distance(c, e) > 45 for e in ordered):
            ordered.append(c)
    if not ordered:
        ordered = [RGBColor(0x1B, 0x50, 0x8F), RGBColor(0x00, 0x93, 0xB2),
                   RGBColor(0x5D, 0x6B, 0x8A)]

    # Primary: first accent with enough presence on white; else darkest scheme color.
    primary = next((c for c in ordered if contrast_ratio(c, white) >= 2.6), None)
    if primary is None:
        primary = ensure_contrast(ordered[0], white, 3.0)

    # Chart series colors must be visible on white.
    chart_colors = [c if contrast_ratio(c, white) >= 2.2
                    else ensure_contrast(c, white, 2.6) for c in ordered]
    if tx2 is not None and luminance(tx2) < 0.6 and all(
            color_distance(tx2, c) > 45 for c in chart_colors):
        chart_colors.append(tx2)

    # Dark surface for cover / closing.
    dark_pool = [c for c in list(raw.values()) + [primary] if luminance(c) < 0.25]
    bg_dark = min(dark_pool, key=luminance) if dark_pool else darken(primary, 0.72)
    if luminance(bg_dark) > 0.16:
        bg_dark = darken(bg_dark, 0.45)

    # Semantic status colors, borrowed from the brand palette when a hue fits.
    def by_hue(lo: float, hi: float) -> RGBColor | None:
        cands = [c for c in ordered
                 if saturation(c) > 0.35 and lo <= hue(c) <= hi]
        return cands[0] if cands else None

    positive = by_hue(95, 165) or RGBColor(0x1E, 0x7F, 0x4E)
    negative = by_hue(340, 360) or by_hue(0, 22) or RGBColor(0xB3, 0x26, 0x1E)

    tint = primary if saturation(primary) > 0.12 else ink
    surface = mix(tint, white, 0.955)
    surface_alt = mix(tint, white, 0.90)
    hairline = mix(tint, white, 0.78)

    return dict(
        bg=white, bg_dark=bg_dark, surface=surface, surface_alt=surface_alt,
        hairline=hairline, ink=ink, ink_soft=ink_soft, ink_muted=ink_muted,
        ink_inverse=white, primary=primary, accents=ordered,
        chart_colors=chart_colors,
        positive=ensure_contrast(positive, white, 3.2),
        negative=ensure_contrast(negative, white, 3.2),
        neutral=ink_muted,
    )


def extract_theme(path: str | None) -> Theme:
    """
    Read brand colors, fonts and canvas size from a .pptx/.potx template by
    following the OOXML relationship chain. Always returns a usable Theme.
    """
    theme = Theme()
    resolved_path = _resolve_local_asset_path(path)
    if not resolved_path:
        if path:
            log.warning("template not found: %s — using default theme", path)
        return theme

    try:
        with zipfile.ZipFile(resolved_path, "r") as zf:
            theme_part, master_part = _find_theme_part(zf)
            if not theme_part:
                log.warning("no theme part in %s — using default theme", resolved_path)
                return theme

            raw = _parse_clr_scheme(zf, theme_part)
            clr_map = _parse_clr_map(zf, master_part)
            major, minor = _parse_font_scheme(zf, theme_part)
            dominant = _dominant_explicit_font(zf)
            size = _slide_size_in(zf)

        palette = _build_palette(raw, clr_map)
        head = major or dominant or Theme.font_head
        body = minor or dominant or head
        theme = Theme(
            **palette,
            font_head=head,
            font_body=body,
            source=f"{os.path.basename(resolved_path)}::{os.path.basename(theme_part)}",
            slide_w_in=size[0] if size else None,
            slide_h_in=size[1] if size else None,
        )
        log.info("theme %s", theme.describe())
    except Exception as exc:                                  # pragma: no cover
        log.warning("theme extraction failed for %s (%s) — using defaults",
                    resolved_path, exc)
    return theme


def extract_theme_from_pptx(path: str) -> Theme:
    """v1 compatibility alias."""
    return extract_theme(path)


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION PARSING
# ══════════════════════════════════════════════════════════════════════════════

_SECTION_ALIASES = {
    "overview": "overview",
    "summary": "overview",
    "result summary": "overview",
    "key findings": "findings",
    "findings": "findings",
    "key finding": "findings",
    "takeaways": "takeaways",
    "key takeaways": "takeaways",
    "so what": "takeaways",
    "implications": "implications",
    "opportunity": "implications",
    "opportunity / implication": "implications",
    "opportunities": "implications",
    "recommendations": "implications",
    "caveats": "caveats",
    "notes": "caveats",
    "data notes": "caveats",
    "relevant questions": "followups",
    "follow-up questions": "followups",
    "follow up questions": "followups",
    "suggested questions": "followups",
}

_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*|__)?\s*([A-Za-z][A-Za-z /\-]{2,40}?)\s*"
    r"(?:\*\*|__)?\s*:?\s*$"
)


def _msg_role(msg: Any) -> str:
    """human | ai | other — works with langchain objects or plain dicts."""
    if isinstance(msg, dict):
        r = str(msg.get("role") or msg.get("type") or "").lower()
        return {"user": "human", "human": "human",
                "assistant": "ai", "ai": "ai"}.get(r, "other")
    t = str(getattr(msg, "type", "") or "").lower()
    if t in ("human", "ai"):
        return t
    name = type(msg).__name__.lower()
    if "human" in name:
        return "human"
    if "ai" in name or "assistant" in name:
        return "ai"
    return "other"


def _msg_content(msg: Any) -> str:
    c = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(c, list):                      # multi-part content blocks
        c = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p)
                     for p in c)
    return str(c or "")


def _msg_kwargs(msg: Any) -> dict:
    if isinstance(msg, dict):
        return msg.get("additional_kwargs") or msg.get("metadata") or {}
    return getattr(msg, "additional_kwargs", None) or {}


def split_summary_sections(text: str) -> dict[str, str]:
    """
    Split a summarizer response into canonical sections. Unlabelled prose
    lands in 'overview'. Returns only non-empty sections.
    """
    if not text:
        return {}
    sections: dict[str, list[str]] = {"overview": []}
    current = "overview"
    for line in str(text).splitlines():
        stripped = line.strip()
        m = _HEADING_RE.match(stripped) if stripped else None
        if m:
            key = _SECTION_ALIASES.get(m.group(1).strip().lower())
            if key:
                current = key
                sections.setdefault(current, [])
                continue
        # inline "Heading: content"
        if ":" in stripped and len(stripped.split(":", 1)[0]) < 42:
            head, rest = stripped.split(":", 1)
            key = _SECTION_ALIASES.get(head.strip().lower().lstrip("#* "))
            if key:
                current = key
                sections.setdefault(current, [])
                if rest.strip():
                    sections[current].append(rest.strip())
                continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "".join(v).strip()}


def _split_sections(text: str, hooks: DeckHooks | None) -> dict[str, str]:
    """
    Prefer the host app's splitter so a slide and the screen never disagree
    about where the overview ends and the takeaways begin.
    """
    builtin = split_summary_sections(text)
    splitter = getattr(hooks, "summary_splitter", None) if hooks else None
    if not splitter:
        return builtin

    try:
        overview, takeaways = splitter(text)
    except Exception as exc:
        log.warning("injected summary_splitter failed (%s) — using built-in", exc)
        return builtin

    sections = {}
    if overview and str(overview).strip():
        sections["overview"] = str(overview).strip()
    if takeaways and str(takeaways).strip():
        sections["takeaways"] = str(takeaways).strip()
    if not sections:
        return builtin

    # The app's splitter answers one question: where do the takeaways start.
    # Any finer sections the built-in recognised (implications, caveats) are
    # still useful, so merge them in without overriding the app's answer.
    for key, value in builtin.items():
        sections.setdefault(key, value)
    return sections


def _as_bullet_list(text: str) -> list[str]:
    raw = str(text or "")
    if not raw.strip():
        return []

    raw = _repair_mojibake_text(raw)
    raw = unicodedata.normalize("NFKC", raw)
    raw = raw.replace("\u00a0", " ")
    raw = _strip_markdown_display(raw)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r" *\n *", "\n", raw).strip()

    candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(candidates) <= 1:
        inline = re.split(
            r"(?=(?:^|\s)(?:[\-\*\u2022\u25CF\u25AA]|\d+[\.\)]|[A-Za-z][\.\)])\s+)",
            raw,
        )
        inline = [part.strip() for part in inline if part and part.strip()]
        if len(inline) > 1:
            candidates = inline
        elif ";" in raw:
            semicolon_parts = [part.strip() for part in raw.split(";") if part.strip()]
            if len(semicolon_parts) > 1:
                candidates = semicolon_parts
        elif "•" in raw:
            dot_parts = [part.strip() for part in raw.split("•") if part.strip()]
            if len(dot_parts) > 1:
                candidates = dot_parts

    out: list[str] = []
    for line in candidates:
        s = re.sub(r"^(?:[\-\*\u2022\u25CF\u25AA]|\d+[\.\)]|[A-Za-z][\.\)])\s*", "", line).strip()
        if s:
            out.append(s)
    return out


def _source_takeaway_points(block: dict, cfg: DeckConfig) -> list[str]:
    """
    Preserve explicit takeaway bullets from the assistant summary when present.

    The old Streamlit UI renders the extracted takeaways block directly with
    markdown, so list structure from the summary survives intact. Reusing that
    structure here keeps deck takeaways aligned with what the user saw in the
    working Streamlit flow.
    """
    sections = block.get("sections") or {}
    for key in ("takeaways", "findings"):
        text = str(sections.get(key) or "").strip()
        if not text:
            continue

        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        parsed = _as_bullet_list(text)
        explicit_list = (
            len(raw_lines) > 1
            or any(_BULLET_RE.match(line) for line in raw_lines)
            or ("•" in text)
        )
        if not explicit_list and len(parsed) <= 1:
            continue

        points: list[str] = []
        seen: set[str] = set()
        for item in parsed:
            cleaned = _clean_str(item, 170)
            if not cleaned:
                continue
            sig = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
            if sig in seen:
                continue
            seen.add(sig)
            points.append(cleaned)
            if len(points) >= cfg.max_bullets:
                break
        if points:
            return points
    return []


def parse_conversation(messages: Iterable[Any],
                       hooks: DeckHooks | None = None) -> list[dict]:
    """
    Turn a message stream into analysis blocks.

    Each block: question, summary, sections, sql, data, viz_code, followups.
    Robust to missing pieces, repeated AI messages and non-langchain inputs.
    """
    blocks: list[dict] = []
    current: dict | None = None

    def _new(question: str) -> dict:
        return {"question": question.strip(), "summary": None, "sections": {},
                "sql": None, "data": None, "viz_code": None,
                "viz_figure": None, "followups": []}

    for msg in messages or []:
        role = _msg_role(msg)
        content = _msg_content(msg)
        kwargs = _msg_kwargs(msg)
        mtype = kwargs.get("type")

        if role == "human":
            if current:
                blocks.append(current)
            current = _new(content)
            continue

        if role != "ai" or current is None:
            continue

        if mtype == "sql_result":
            current["data"] = kwargs.get("data")
            sql_query = kwargs.get("sql_query")
            if sql_query and not current["sql"]:
                current["sql"] = str(sql_query).strip()
            result_summary = kwargs.get("result_summary")
            if result_summary and not current["summary"]:
                body = str(result_summary).strip()
                if body:
                    current["summary"] = body
                    current["sections"] = _split_sections(body, hooks)
            continue

        if mtype == "visualization":
            code = kwargs.get("code")
            if code and str(code).strip().upper() not in ("NO_VISUALIZATION", "NONE"):
                current["viz_code"] = code
            figure = kwargs.get("figure")
            if isinstance(figure, dict):
                current["viz_figure"] = figure
            continue

        if mtype in ("table", "dataframe") and kwargs.get("data"):
            current["data"] = kwargs["data"]
            continue

        if "SQL Query Executed:" in content:
            after = content.split("SQL Query Executed:", 1)[1]
            sql = after.split("Result Summary:", 1)[0].strip()
            current["sql"] = sql or current["sql"]

        body = content
        if "Result Summary:" in content:
            body = content.split("Result Summary:", 1)[1].strip()
        elif current["summary"] is not None or len(content.strip()) < 80:
            continue

        for marker in ("Relevant Questions:", "Follow-up Questions:",
                       "Suggested Questions:"):
            if marker in body:
                body, tail = body.split(marker, 1)
                current["followups"] = _as_bullet_list(tail)[:4]
                break

        body = body.strip()
        if body:
            current["summary"] = body
            current["sections"] = _split_sections(body, hooks)

    if current:
        blocks.append(current)

    blocks = [b for b in blocks if b["question"]]
    log.info("parsed %d analysis block(s)", len(blocks))
    for index, block in enumerate(blocks, 1):
        sql_payload = block.get("data") if isinstance(block.get("data"), dict) else {}
        data_columns = list(sql_payload.get("columns") or [])
        data_rows = list(sql_payload.get("data") or [])
        log.info(
            "[deck-debug] block[%d] question=%s summary_len=%d sql=%s rows=%d cols=%s viz_code=%s viz_figure=%s followups=%d",
            index,
            _debug_excerpt(block.get("question"), 100),
            len(str(block.get("summary") or "")),
            bool(block.get("sql")),
            len(data_rows),
            _debug_list_preview(data_columns, 6) or "(none)",
            bool(block.get("viz_code")),
            isinstance(block.get("viz_figure"), dict),
            len(block.get("followups") or []),
        )
        if block.get("summary"):
            log.info("[deck-debug] block[%d] summary=%s", index,
                     _debug_excerpt(block.get("summary"), 220))
    return blocks


def extract_export_context(messages: Iterable[Any],
                           hooks: DeckHooks | None = None) -> dict[str, Any] | None:
    """
    Latest parsed analysis block, expanded into the field-by-field export shape
    the host app expects for preview and deck generation.
    """
    message_list = list(messages or [])
    blocks = parse_conversation(message_list, hooks=hooks)
    if not blocks:
        return None

    block = blocks[-1]
    sql_payload = block.get("data") if isinstance(block.get("data"), dict) else {}
    columns = list(sql_payload.get("columns") or [])
    rows = list(sql_payload.get("data") or [])

    return {
        "question": block.get("question"),
        "summary": block.get("summary"),
        "sql": block.get("sql"),
        "sql_columns": columns,
        "sql_rows": rows,
        "row_count": len(rows),
        "viz_code": block.get("viz_code"),
        "viz_figure_payload": block.get("viz_figure"),
        "followups": list(block.get("followups") or []),
        "normalized_messages": message_list,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_ACRONYMS = {
    "SQL", "MG", "ML", "PAP", "COM", "HCP", "NPI", "ID", "YTD", "QTD", "MTD",
    "TRX", "NRX", "ASP", "WAC", "GTN", "COGS", "ROI", "KPI", "DOT", "IQVIA",
    "AVG", "PCT", "USD", "NDC", "MSA", "HHS", "IDN", "GPO", "SP", "SD",
}
_PERIOD_RE = re.compile(r"^(R|P|L|T)\d{1,3}(W|M|Q|D|Y)$", re.I)
_ACRONYM_DISPLAY = {"MG": "mg", "ML": "mL", "PCT": "%", "AVG": "Avg",
                    "PERCENT": "%", "SD": "SD", "SP": "SP"}

_DATEISH = re.compile(r"(DATE|WEEK|MONTH|PERIOD|QUARTER|YEAR|DAY)$", re.I)
_META_COL = re.compile(
    r"(BUSINESS_?DAYS?|_COMPLETE$|^IS_|_FLAG$|_ID$|^ID$|ROW_?NUM|_KEY$|"
    r"_RANK$|_INDEX$)", re.I)
_VIAL_RE = re.compile(r"(VIAL|UNIT|EACH)S?", re.I)
_MONEY_RE = re.compile(r"(REVENUE|SALES_?\$|DOLLAR|AMOUNT|GROSS|NET_?SALES|PRICE)", re.I)
_PCT_RE = re.compile(r"(PCT|PERCENT|SHARE|RATE|GROWTH|CHANGE)", re.I)
_DISPLAY_LABEL_RE = re.compile(
    r"(NAME|TITLE|LABEL|ACCOUNT|CUSTOMER|PARENT|HCP|PROVIDER|PHYSICIAN|"
    r"PRESCRIBER|REGION|TERRITORY|DISTRICT|STATE|SPECIALTY|SEGMENT|GROUP)",
    re.I,
)
_IDENTIFIER_LABEL_RE = re.compile(
    r"(^ID$|_ID$|(^|_)(KEY|CODE|NPI|NDC|ZIP|POSTAL|UUID|GUID)($|_))",
    re.I,
)


def humanize_column(name: str) -> str:
    """WEEK_END_DATE -> Week End Date; RELMORA_TOTAL_MG_R4W -> Relmora Total mg R4W."""
    parts = re.split(r"[_\s]+", str(name).strip())
    out = []
    for p in parts:
        if not p:
            continue
        up = p.upper()
        if up in _ACRONYMS:
            out.append(_ACRONYM_DISPLAY.get(up, up))
        elif _PERIOD_RE.match(up):
            out.append(up)
        else:
            out.append(p.capitalize() if p.isupper() else p[:1].upper() + p[1:])
    return " ".join(out)


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T].*)?$")


def coerce_date_columns(df):
    """
    Parse date-named string columns into real dates.

    This decides how a trend chart looks. Given ISO strings, plotly and
    viz_polish see a categorical axis with one category per row, so the tick
    pinning that produces week-ending labels never runs and the axis falls
    back to interpolated month ticks. Snowflake usually returns date objects,
    but a cast, a UNION or a TO_CHAR in the generated SQL turns them into
    text, and the chart quietly changes shape.
    """
    if df is None or pd is None:
        return df
    for col in df.columns:
        if not _DATEISH.search(str(col)):
            continue
        try:
            series = df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                continue
            # pandas 3 gives text columns a dedicated "str" dtype rather than
            # object, so an `== object` test silently skips every string date.
            if not (series.dtype == object
                    or pd.api.types.is_string_dtype(series)):
                continue
            sample = series.dropna().head(25)
            if sample.empty or not all(isinstance(v, str)
                                       and _ISO_DATE.match(v.strip())
                                       for v in sample):
                continue
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().sum() >= max(series.notna().sum() * 0.9, 1):
                df[col] = parsed
        except Exception:
            continue
    return df


def to_dataframe(sql_data: Any):
    """Accepts {'columns': [...], 'data': [...]}, a DataFrame, or records."""
    if pd is None or sql_data is None:
        return None
    try:
        if isinstance(sql_data, pd.DataFrame):
            return coerce_date_columns(sql_data.copy())
        if isinstance(sql_data, dict) and "data" in sql_data:
            rows, cols = sql_data.get("data"), sql_data.get("columns")
            if not rows:
                return None
            df = pd.DataFrame(rows)
            if cols:
                keep = [c for c in cols if c in df.columns]
                extra = [c for c in df.columns if c not in keep]
                df = df[keep + extra] if keep else df
            return coerce_date_columns(df)
        if isinstance(sql_data, list) and sql_data:
            return coerce_date_columns(pd.DataFrame(sql_data))
    except Exception as exc:
        log.warning("to_dataframe failed: %s", exc)
    return None


def _is_numeric(series) -> bool:
    if pd is None:
        return False
    try:
        return bool(pd.api.types.is_numeric_dtype(series)) and not _is_boolish(series)
    except Exception:
        return False


def _is_boolish(series) -> bool:
    try:
        vals = set(series.dropna().unique().tolist())
        return vals.issubset({0, 1, True, False}) and len(vals) <= 2
    except Exception:
        return False


def _is_dateish(name: str, series=None) -> bool:
    if _DATEISH.search(str(name)):
        return True
    if series is not None and pd is not None:
        try:
            return bool(pd.api.types.is_datetime64_any_dtype(series))
        except Exception:
            return False
    return False


def compact_number(x: float, decimals: int = 1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    sign = "-" if v < 0 else ""
    v = abs(v)
    for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= cut:
            q = v / cut
            txt = f"{q:.{decimals}f}".rstrip("0").rstrip(".")
            return f"{sign}{txt}{suffix}"
    if v == int(v):
        return f"{sign}{int(v):,}"
    return f"{sign}{v:,.{decimals}f}"


def format_cell(column: str, value: Any) -> str:
    """Platform rounding convention, applied at render time only."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    col = str(column).upper()

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:                                   # numpy / pandas scalar -> python
            value = value.item()
        except Exception:
            pass

    if hasattr(value, "strftime"):
        return value.strftime("%d %b %Y")
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        try:                                   # "2025-12-12" -> "12 Dec 2025"
            import datetime as _dt
            return _dt.date.fromisoformat(value.strip()).strftime("%d %b %Y")
        except ValueError:
            pass
    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (int, float)):
        if _PCT_RE.search(col) and abs(value) <= 1 and isinstance(value, float):
            return f"{value * 100:,.1f}%"
        if _VIAL_RE.search(col):
            return f"{value:,.1f}"
        if _MONEY_RE.search(col):
            return f"${value:,.0f}"
        if isinstance(value, float) and not float(value).is_integer():
            return f"{value:,.2f}"
        return f"{int(value):,}"

    s = _normalise_display_text(value)
    return s if s else "—"


def _is_identifier_label_column(name: str) -> bool:
    return bool(_IDENTIFIER_LABEL_RE.search(str(name or "").upper()))


def _is_display_label_column(name: str) -> bool:
    up = str(name or "").upper()
    return bool(_DISPLAY_LABEL_RE.search(up)) and not _is_identifier_label_column(up)


def _choose_label_column(df, constants: dict[str, Any]) -> str | None:
    cols = list(df.columns)
    text_cols = [c for c in cols if c not in constants and not _is_numeric(df[c])]
    preferred = next((c for c in text_cols if _is_display_label_column(c)), None)
    if preferred is not None:
        return preferred
    non_meta = next((c for c in text_cols
                     if not _META_COL.search(str(c).upper())
                     and not _is_identifier_label_column(c)
                     and not _is_dateish(c, df[c])), None)
    if non_meta is not None:
        return non_meta
    dateish = next((c for c in text_cols if _is_dateish(c, df[c])), None)
    if dateish is not None:
        return dateish
    fallback = next((c for c in text_cols if not _is_identifier_label_column(c)), None)
    if fallback is not None:
        return fallback
    return next((c for c in cols if c not in constants), cols[0] if cols else None)


def constant_columns(df) -> dict[str, Any]:
    """Columns with one distinct value across >1 row — collapse into chips."""
    if df is None or pd is None or len(df) < 1:
        return {}
    out = {}
    for col in df.columns:
        try:
            uniq = df[col].dropna().unique()
        except Exception:
            continue
        if len(uniq) == 1:
            out[col] = uniq[0]
    return out


_SCOPE_RE = re.compile(r"(PRODUCT|BRAND|GEOGRAPHY|REGION|CHANNEL|BASIS|"
                       r"WINDOW|SCOPE|SEGMENT)", re.I)


def _collapsed_columns(df, hooks: DeckHooks | None) -> list[str]:
    """Columns the app lifts out of its grid, so the deck's table matches."""
    collapser = getattr(hooks, "period_collapser", None) if hooks else None
    if not collapser or df is None:
        return []
    try:
        display_df, _ = collapser(df)
        if display_df is None:
            return []
        return [c for c in df.columns if c not in list(display_df.columns)]
    except Exception as exc:
        log.warning("injected period_collapser failed: %s", exc)
        return []


def context_chips(df, max_chips: int = 4,
                  hooks: DeckHooks | None = None) -> list[str]:
    """
    Period / scope constants become header chips instead of table columns.
    Start/end date pairs are folded into a single range chip.
    """
    # The app already decides which period columns become context chips;
    # mirror that decision exactly rather than re-deriving it.
    collapser = getattr(hooks, "period_collapser", None) if hooks else None
    if collapser is not None:
        try:
            _, items = collapser(df)
            if items:
                return [f"{label}: {value}" for label, value in items][:max_chips]
        except Exception as exc:
            log.warning("injected period_collapser failed: %s", exc)

    consts = constant_columns(df)
    used: set[str] = set()
    chips: list[str] = []

    for col in list(consts):
        m = re.match(r"^(.*)_START_DATE$", str(col), re.I)
        if not m:
            continue
        end = next((c for c in consts
                    if str(c).upper() == f"{m.group(1).upper()}_END_DATE"), None)
        if end is None:
            continue
        used.update({col, end})
        base = humanize_column(m.group(1)) or "Period"
        chips.append(f"{base}: {format_cell(col, consts[col])} – "
                     f"{format_cell(end, consts[end])}")

    for col, val in consts.items():
        if col in used:
            continue
        up = str(col).upper()
        if _META_COL.search(up):
            continue
        if _is_dateish(col) or _PERIOD_RE.search(up) or _SCOPE_RE.search(up):
            chips.append(f"{humanize_column(col)}: {format_cell(col, val)}")
    return chips[:max_chips]


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC DATA PROFILE  (ground truth for the LLM and for verification)
# ══════════════════════════════════════════════════════════════════════════════


def profile_dataframe(df, hooks: DeckHooks | None = None) -> dict:
    """Compute the facts a slide is allowed to assert."""
    empty = {"rows": 0, "columns": [], "shape": "empty", "metrics": {},
             "label_col": None, "constants": {}, "chips": []}
    if df is None or pd is None or len(df) == 0:
        return empty

    cols = list(df.columns)
    constants = constant_columns(df)
    numeric = [c for c in cols if _is_numeric(df[c]) and c not in constants]
    if not numeric:
        numeric = [c for c in cols if _is_numeric(df[c])]

    label_col = _choose_label_column(df, constants)

    def label_at(idx) -> str:
        try:
            return format_cell(label_col, df[label_col].iloc[idx])
        except Exception:
            return f"row {idx + 1}"

    metrics: dict[str, dict] = {}
    for col in numeric:
        s = df[col]
        try:
            clean = s.dropna()
            if clean.empty:
                continue
            imax = int(clean.astype(float).idxmax())
            imin = int(clean.astype(float).idxmin())
            pos_max = df.index.get_loc(imax)
            pos_min = df.index.get_loc(imin)
            first, last = float(clean.iloc[0]), float(clean.iloc[-1])
            delta = last - first
            pct = (delta / abs(first) * 100.0) if first else None
            metrics[col] = {
                "min": float(clean.min()), "max": float(clean.max()),
                "min_at": label_at(pos_min), "max_at": label_at(pos_max),
                "first": first, "last": last, "sum": float(clean.sum()),
                "mean": float(clean.mean()), "delta": delta,
                "pct_change": pct, "n": int(clean.shape[0]),
            }
        except Exception:
            continue

    if len(df) == 1:
        shape = "single_row"
    elif label_col is not None and _is_dateish(label_col, df[label_col]):
        shape = "timeseries"
    elif len(df) <= 12:
        shape = "categorical"
    else:
        shape = "wide"

    return {"rows": int(len(df)), "columns": cols, "shape": shape,
            "metrics": metrics, "label_col": label_col,
            "constants": {k: format_cell(k, v) for k, v in constants.items()},
            "chips": context_chips(df, hooks=hooks),
            "drop_cols": _collapsed_columns(df, hooks)}


def facts_block(df, profile: dict, max_metrics: int = 8) -> str:
    """Compact, unambiguous fact sheet handed to the model."""
    if not profile or profile["rows"] == 0:
        return "NO_TABULAR_RESULT"

    lines = [f"rows={profile['rows']}  shape={profile['shape']}",
             f"columns={', '.join(map(str, profile['columns']))}"]
    if profile["constants"]:
        lines.append("constants: " + "; ".join(
            f"{humanize_column(k)}={v}" for k, v in profile["constants"].items()))

    if profile["rows"] == 1 and df is not None:
        lines.append("single-row values:")
        row = df.iloc[0]
        for col in profile["columns"]:
            lines.append(f"  {humanize_column(col)} = {format_cell(col, row[col])}")
        return "\n".join(lines)

    lines.append("verified metric facts (label column = "
                 f"{humanize_column(profile['label_col'])}):")
    for col, m in list(profile["metrics"].items())[:max_metrics]:
        pct = f"{m['pct_change']:+.1f}%" if m["pct_change"] is not None else "n/a"
        lines.append(
            f"  {humanize_column(col)}: total={compact_number(m['sum'])} "
            f"avg={compact_number(m['mean'])} "
            f"max={compact_number(m['max'])} at {m['max_at']} · "
            f"min={compact_number(m['min'])} at {m['min_at']} · "
            f"first={compact_number(m['first'])} last={compact_number(m['last'])} "
            f"first→last={pct}"
        )
    return "\n".join(lines)


# ── numeric verification ─────────────────────────────────────────────────────

_NUM_TOKEN = re.compile(r"-?\$?\d[\d,]*\.?\d*\s*(?:%|K|M|B|T|bn|mn)?", re.I)


def _token_to_float(token: str) -> float | None:
    t = token.strip().replace(",", "").replace("$", "")
    mult = 1.0
    if t[-1:].upper() in ("K", "M", "B", "T"):
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[t[-1].upper()]
        t = t[:-1]
    elif t.lower().endswith(("bn", "mn")):
        mult = 1e9 if t.lower().endswith("bn") else 1e6
        t = t[:-2]
    t = t.rstrip("%")
    try:
        return float(t) * mult
    except ValueError:
        return None


def allowed_numbers(df, summary: str) -> set[float]:
    """Every number the slide may legitimately mention."""
    allowed: set[float] = set()

    def add(v: float | None):
        if v is None or not math.isfinite(v):
            return
        allowed.add(round(v, 4))
        allowed.add(round(v))
        for scale in (1e3, 1e6, 1e9):
            allowed.add(round(v / scale, 1))
            allowed.add(round(v / scale, 2))

    for token in _NUM_TOKEN.findall(summary or ""):
        add(_token_to_float(token))

    if df is not None and pd is not None:
        for col in df.columns:
            if not _is_numeric(df[col]):
                for v in df[col].dropna().astype(str).head(60):
                    for token in _NUM_TOKEN.findall(v):
                        add(_token_to_float(token))
                continue
            series = df[col].dropna().astype(float)
            for v in series.head(400):
                add(float(v))
            if not series.empty:
                add(float(series.sum()))
                add(float(series.mean()))
                first, last = float(series.iloc[0]), float(series.iloc[-1])
                add(last - first)
                if first:
                    add((last - first) / abs(first) * 100.0)
    return allowed


def unverified_numbers(text: str, allowed: set[float],
                       tol: float = 0.02) -> list[str]:
    """Numbers asserted in *text* that no source value supports."""
    bad = []
    for token in _NUM_TOKEN.findall(text or ""):
        v = _token_to_float(token)
        if v is None or abs(v) < 2:            # ignore counts like "3 regions"
            continue
        ok = any(abs(v - a) <= max(tol * max(abs(a), 1.0), 0.51) for a in allowed)
        if not ok:
            bad.append(token.strip())
    return bad


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a commercial analytics lead at a life-sciences company, writing ONE \
executive slide for a brand review. Your audience is a VP of Commercial who \
has ninety seconds and no patience for restated tables.

You return a single JSON object. Nothing else. No prose, no markdown, no code \
fences, no commentary before or after.

Three rules override everything else:

1. DATA FIDELITY IS ABSOLUTE. Every number you write must appear in <facts> or \
<summary>. Never compute, extrapolate, round differently, annualise, or infer \
a figure. If a number you want is not there, write the sentence without it.
2. SUPERLATIVES MUST BE VERIFIED. "Peak", "highest", "lowest", "strongest", \
"weakest", "best", "worst" are only allowed when <facts> explicitly names that \
max/min. Otherwise use neutral language.
3. STAY INSIDE THE DOMAIN. Classify the question first, then use only that \
domain's vocabulary.
"""

USER_PROMPT = """\
<question>
{question}
</question>

<summary>
{summary}
</summary>

<facts>
{facts}
</facts>

<sql_columns>
{columns}
</sql_columns>

<has_chart>{has_chart}</has_chart>

════════════════════════════════════════════════════════════════════
STEP 1 — CLASSIFY THE DOMAIN (silently). Pick exactly one.

  SALES_PERFORMANCE   sales, demand, volume, vials, mg, growth, new accounts,
                      breadth, depth, adoption, own-brand performance
  COMPETITOR_DYNAMICS market share, share shift, capture rate, competitive
                      position, gaining/losing share
  FIELD_EXECUTION     calls, touches, reach, frequency, coverage, call plan,
                      rep activity
  CROSS_DOMAIN        only when the question explicitly links two of the above
                      ("did more calls drive sales lift?")

VOCABULARY FIREWALL — MANDATORY, no exceptions:
  · FIELD_EXECUTION      -> never use sales / revenue / share words.
  · SALES_PERFORMANCE    -> never use call / reach / frequency words unless
                            those columns appear in <sql_columns>.
  · COMPETITOR_DYNAMICS  -> never headline own-brand absolute sales.
  · If <facts> contradicts the question's apparent domain, trust <facts> and
    say so in `footnote`.

STEP 2 — CLASSIFY THE PATTERN (silently): TREND, SHORT_TERM_MOMENTUM,
REGIONAL, DISTRIBUTION, ACCOUNT_HEALTH, NEW_BUSINESS, ADOPTION, TIER,
MARKET_SHARE, TOP_N, SINGLE_POINT. MARKET_SHARE is valid only with
COMPETITOR_DYNAMICS.

════════════════════════════════════════════════════════════════════
STEP 3 — WRITE THE FIELDS

`title`     The question, corrected into clean title case. ≤ 12 words.
            Preserve its meaning exactly. Do NOT turn it into a conclusion.

`headline`  The answer in one line. ≤ 14 words. This is the "so what".
            It MUST take a position: up, down, flat, concentrated, at risk.
            Weak: "Weekly sales were analysed over 13 weeks."
            Strong: "Demand held up week to week, but the last 4 weeks slowed."

`bullets`   The "What this means" section. MINIMUM 1, MAXIMUM 3.
            Drawn strictly from the Takeaways / Key Findings content of
            <summary>. One idea each, ≤ 20 words each, each carrying a
            number or a named entity. No preamble ("The data shows that…").

            EVERY POINT MUST TAKE A DIFFERENT ANGLE. No exceptions.
            Pick from distinct dimensions, never two from the same one:
              · magnitude   — how big the movement is
              · direction   — which way, versus what baseline
              · concentration — where it is coming from (region, account, tier)
              · driver      — what changed underneath it
              · caveat      — what would make this reading wrong
            Two points that restate the same number in different words count
            as one point. Write fewer points rather than repeating yourself.
            Order: biggest movement first, then context, then caveat.

`kpis`      The "Key Findings" section. MINIMUM 1, MAXIMUM 3, ordered by
            importance. Each is {{"label", "value", "definition"}}.
            Every finding MUST be distinct — no two may share a label, repeat
            the same value, or measure the same thing twice.
            · KPI 1 MUST be the national / total-level metric.
              Preferred form: "prior → current (growth%)" in one value string.
                "370,125 mg → 259,205 mg (-11%)"
              Fallbacks in order: current value alone · growth alone ·
              (only if no national metric exists) the largest sub-level metric.
            · KPI 2 and 3: the next most significant cuts — geography, tier,
              segment, or a supporting rate.
            · `label` ≤ 4 words. `value` ≤ 28 characters, formatted for reading
              (thousands separators, K/M suffixes, % signs).
            · `definition` ≤ 14 words, says what the metric measures. Never
              repeats the value.

`insight`   2–3 sentences. Structure: what moved → why it matters →
            what to do or watch next. Anchored to the Implications /
            Opportunity content of <summary>. No new numbers.

`footnote`  One line of data caveat, or "". MANDATORY when <facts> shows an
            incomplete period, a partial week, a business-day mismatch, or a
            single-day window. Name the affected period explicitly.

`basis`     One of: "Commercial only", "Commercial + PAP", "Not stated".
            MANDATORY. Choose "Not stated" unless <summary> or <sql_columns>
            makes the demand basis explicit. Never guess.

`chart_caption`  ≤ 12 words describing what the chart plots, or "" when
            <has_chart> is false.

════════════════════════════════════════════════════════════════════
OUTPUT — this exact shape, valid JSON, double quotes, no trailing commas:

{{
  "title": "Are We Seeing Strong Short-Term Sales Momentum?",
  "headline": "Momentum turned negative: daily demand fell 11% versus the prior four weeks",
  "bullets": [
    "Recent 4-week volume fell to 259,205 mg from 370,125 mg.",
    "Daily average declined 19,480 mg to 17,280 mg, an 11% drop.",
    "Both windows are complete, so the comparison is like for like."
  ],
  "kpis": [
    {{"label": "National R4W Demand", "value": "370,125 → 259,205 mg (-11%)",
      "definition": "Total national volume, recent four weeks versus the prior four."}},
    {{"label": "Daily Average", "value": "19,480 → 17,280 mg",
      "definition": "Volume per business day, normalising for calendar differences."}},
    {{"label": "Momentum Call", "value": "Negative",
      "definition": "Direction of business-day-adjusted demand versus the prior window."}}
  ],
  "insight": "The decline survives business-day normalisation, so this is pace, not calendar. That points at underlying demand rather than a shortened month. Confirm whether the softness is broad or concentrated in a few accounts before acting.",
  "footnote": "",
  "basis": "Not stated",
  "chart_caption": ""
}}

FINAL CHECK before you answer:
  ✔ Every number traces to <facts> or <summary>
  ✔ 1-3 takeaways, each a genuinely different angle
  ✔ 1-3 findings, none repeating another's label or value
  ✔ KPI 1 is the national metric
  ✔ No vocabulary from another domain
  ✔ `basis` is present
  ✔ Output is one JSON object and nothing else
"""

_REPAIR_PROMPT = """\
Your previous reply was not valid JSON, or broke the contract:

{problem}

Return the corrected JSON object only. Same schema. No commentary.
"""


# ── LLM transport ────────────────────────────────────────────────────────────

def _resolve_provider(cfg: DeckConfig) -> str:
    if cfg.llm_provider != "auto":
        return cfg.llm_provider
    if _env_nonempty("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _env_nonempty("OPENAI_API_KEY") or _env_nonempty("AZURE_OPENAI_API_KEY"):
        return "openai"
    return "none"


def _call_llm(system: str, messages: list[dict], cfg: DeckConfig) -> str | None:
    provider = _resolve_provider(cfg)
    log.info("[deck-debug] provider state provider=%s anthropic_key=%s anthropic_model=%s openai_key=%s openai_model=%s",
             provider,
             bool(_env_nonempty("ANTHROPIC_API_KEY")),
             _env_nonempty("ANTHROPIC_MODEL") or "(default)",
             bool(_env_nonempty("OPENAI_API_KEY") or _env_nonempty("AZURE_OPENAI_API_KEY")),
             _env_nonempty("OPENAI_MODEL") or "(default)")
    try:
        if provider == "anthropic":
            from anthropic import Anthropic

            model = cfg.llm_model or _env_nonempty("ANTHROPIC_MODEL") or "claude-sonnet-5"
            log.info("[deck-debug] llm request provider=%s model=%s messages=%d temp=%s",
                     provider, model, len(messages), cfg.llm_temperature)
            client = Anthropic(timeout=60.0, max_retries=1)
            kwargs = dict(model=model, max_tokens=1600, system=system, messages=messages)
            if cfg.llm_temperature != 0.0:
                kwargs["temperature"] = cfg.llm_temperature
            try:
                resp = client.messages.create(**kwargs)
            except Exception as exc:
                # Newer model generations (Sonnet 5 / Opus 4.7+) reject
                # `temperature` outright if present — retry once without it.
                if "temperature" in str(exc).lower() and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    resp = client.messages.create(**kwargs)
                else:
                    raise
            text = "".join(b.text for b in resp.content if b.type == "text")
            log.info("[deck-debug] llm response provider=%s model=%s chars=%d preview=%s",
                     provider, model, len(text), _debug_excerpt(text, 220))
            return text

        if provider == "openai":
            from openai import OpenAI

            model = cfg.llm_model or _env_nonempty("OPENAI_MODEL") or "gpt-4.1"
            log.info("[deck-debug] llm request provider=%s model=%s messages=%d temp=%s",
                     provider, model, len(messages), cfg.llm_temperature)
            resp = OpenAI().chat.completions.create(
                model=model, temperature=cfg.llm_temperature,
                messages=[{"role": "system", "content": system}] + messages,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            log.info("[deck-debug] llm response provider=%s model=%s chars=%d preview=%s",
                     provider, model, len(text or ""), _debug_excerpt(text, 220))
            return text
    except Exception as exc:
        model = (cfg.llm_model
                 or _env_nonempty("ANTHROPIC_MODEL")
                 or _env_nonempty("OPENAI_MODEL")
                 or "(default)")
        log.warning("LLM call failed (%s/%s): %s", provider, model, exc)
    return None


# ── JSON extraction ──────────────────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
    """Pull one JSON object out of a model reply, repairing common damage."""
    if not text:
        return None
    s = str(text).strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()

    start = s.find("{")
    if start < 0:
        return None
    depth, end, in_str, esc = 0, None, False, False
    for i, ch in enumerate(s[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    candidate = s[start:end] if end else s[start:]

    for attempt in (candidate,
                    re.sub(r",\s*([}\]])", r"\1", candidate),
                    re.sub(r",\s*([}\]])", r"\1",
                           candidate.replace("“", '"').replace("”", '"')
                           .replace("’", "'").replace("\n", " "))):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


# ── schema coercion ──────────────────────────────────────────────────────────

def _clean_str(v: Any, limit: int | None = None) -> str:
    s = _normalise_display_text(v).strip('"')
    if limit and len(s) > limit:
        s = s[:limit].rstrip(" ,;:.") + "…"
    return s


def coerce_content(raw: dict, block: dict, cfg: DeckConfig) -> dict:
    """Force any model output into the exact shape the layout engine expects."""
    raw = raw or {}
    question = block.get("question", "Analysis")

    bullets = raw.get("bullets") or []
    if isinstance(bullets, str):
        bullets = _as_bullet_list(bullets)
    bullets = [_clean_str(b, 170) for b in bullets if _clean_str(b)][:cfg.max_bullets]

    kpis: list[dict] = []
    for item in (raw.get("kpis") or [])[:cfg.max_kpis]:
        if isinstance(item, dict):
            label = _clean_str(item.get("label"), 34)
            value = _clean_str(item.get("value") or item.get("val"), 30)
            definition = _clean_str(
                item.get("definition") or item.get("defination") or "", 110)
        else:
            parts = str(item).split(":", 1)
            label = _clean_str(parts[0], 34)
            value = _clean_str(parts[1] if len(parts) > 1 else "", 30)
            definition = ""
        if label or value:
            kpis.append({"label": label or "Metric",
                         "value": value or "—",
                         "definition": definition})

    title = _clean_str(raw.get("title") or question, 110)
    if title and title[-1] not in "?.!":
        title = title.rstrip(".")

    return {
        "eyebrow": _clean_str(raw.get("eyebrow") or "ANALYSIS", 28).upper(),
        "title": title,
        "headline": _clean_str(raw.get("headline"), 130),
        "bullets": bullets,
        "kpis": kpis,
        "insight": _clean_str(raw.get("insight"), 520),
        "footnote": _clean_str(raw.get("footnote"), 190),
        "basis": _clean_str(raw.get("basis") or "Not stated", 40),
        "chart_caption": _clean_str(raw.get("chart_caption"), 90),
    }


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "from", "by", "is", "are", "was", "were", "be", "been", "this",
    "that", "these", "those", "it", "its", "as", "than", "then", "so", "which",
    "while", "also", "over", "per", "across", "versus", "vs", "into", "more",
    "most", "less", "least", "has", "have", "had", "not", "no",
}


def _signature(text: str) -> set[str]:
    """Significant tokens, used to judge whether two points say the same thing."""
    words = re.findall(r"[a-z0-9%.,]+", str(text or "").lower())
    return {w.strip(".,") for w in words
            if w not in _STOPWORDS and len(w.strip(".,")) > 2}


def _too_similar(a: str, b: str, threshold: float = 0.55) -> bool:
    sa, sb = _signature(a), _signature(b)
    if not sa or not sb:
        return False
    overlap = len(sa & sb) / min(len(sa), len(sb))
    return overlap >= threshold


def enforce_variety(items: Sequence[Any], text_of, *, min_items: int,
                    max_items: int, threshold: float = 0.55) -> list:
    """
    Keep between *min_items* and *max_items* entries, dropping any that
    restate an entry already kept. Order is preserved, so the strongest
    point stays first.
    """
    kept: list[Any] = []
    for item in items or []:
        text = str(text_of(item) or "").strip()
        if not text:
            continue
        if any(_too_similar(text, text_of(k)) for k in kept):
            continue
        kept.append(item)
        if len(kept) == max_items:
            break
    return kept


def _dedupe_kpis(kpis: Sequence[dict], max_items: int) -> list[dict]:
    """Unique by label AND by displayed value — two tiles never twin."""
    kept: list[dict] = []
    seen_labels: set[str] = set()
    seen_values: set[str] = set()
    for kpi in kpis or []:
        label = re.sub(r"[^a-z0-9]", "", str(kpi.get("label", "")).lower())
        value = re.sub(r"[^a-z0-9]", "", str(kpi.get("value", "")).lower())
        if not label and not value:
            continue
        if label in seen_labels or (value and value in seen_values):
            continue
        if any(_too_similar(kpi.get("label", ""), k.get("label", "")) for k in kept):
            continue
        seen_labels.add(label)
        if value:
            seen_values.add(value)
        kept.append(kpi)
        if len(kept) == max_items:
            break
    return kept


def _ensure_min_kpis(kpis: list[dict], df, profile: dict,
                     cfg: DeckConfig) -> list[dict]:
    """At least one finding whenever the query returned anything at all."""
    if kpis or df is None or profile.get("rows", 0) == 0:
        return kpis
    metrics = profile.get("metrics") or {}
    if metrics:
        col, m = next(iter(metrics.items()))
        return [{"label": tidy_label(humanize_column(col)),
                 "value": _kpi_number(col, m["sum"] if not is_rate_like(col)
                                      else m["mean"])[:30],
                 "definition": f"Across {m['n']} rows in the result."}]
    try:
        col = list(df.columns)[0]
        return [{"label": tidy_label(humanize_column(col)),
                 "value": format_cell(col, df[col].iloc[0])[:30],
                 "definition": ""}]
    except Exception:
        return kpis


def _ensure_min_takeaways(points: list[str], content: dict) -> list[str]:
    """A slide always says at least one thing."""
    if points:
        return points
    for candidate in (content.get("headline"), content.get("insight")):
        sentences = _sentences(candidate or "")
        if sentences:
            return [_condense(sentences[0])]
    return []


def enforce_content_rules(content: dict, df, profile: dict,
                          cfg: DeckConfig) -> dict:
    """
    Final deterministic gate on what reaches a slide:
      * Key Findings  — 1 to 3, unique by label and by value
      * What this means — 1 to 3, no two making the same point
    """
    kpis = _dedupe_kpis(content.get("kpis") or [], cfg.max_kpis)
    kpis = _ensure_min_kpis(kpis, df, profile, cfg)
    content["kpis"] = kpis[:cfg.max_kpis]

    points = enforce_variety(content.get("bullets") or [], lambda x: x,
                             min_items=cfg.min_bullets, max_items=cfg.max_bullets)
    content["bullets"] = _ensure_min_takeaways(points, content)
    return content


_SUPERLATIVE = re.compile(
    r"\b(peak|peaked|highest|lowest|strongest|weakest|best|worst|record|"
    r"all-time|maximum|minimum)\b", re.I)


def verify_content(content: dict, df, block: dict, profile: dict) -> list[str]:
    """Return a list of contract violations, and scrub what can be scrubbed."""
    problems: list[str] = []
    allowed = allowed_numbers(df, block.get("summary") or "")

    for field_name in ("headline", "insight", "footnote"):
        bad = unverified_numbers(content.get(field_name, ""), allowed)
        if bad:
            problems.append(f"{field_name} cites unsupported numbers: {bad}")

    for i, b in enumerate(content.get("bullets", [])):
        bad = unverified_numbers(b, allowed)
        if bad:
            problems.append(f"bullet {i + 1} cites unsupported numbers: {bad}")

    for i, k in enumerate(content.get("kpis", [])):
        bad = unverified_numbers(k.get("value", ""), allowed)
        if bad:
            problems.append(f"kpi {i + 1} value cites unsupported numbers: {bad}")

    # Superlatives are only allowed when the fact sheet actually names a max/min.
    has_extremes = bool(profile.get("metrics"))
    joined = " ".join([content.get("headline", ""), content.get("insight", "")]
                      + list(content.get("bullets", [])))
    if _SUPERLATIVE.search(joined) and not has_extremes:
        problems.append("uses a superlative but no min/max fact exists")

    n_points = len(content.get("bullets", []))
    if profile.get("rows") and not 1 <= n_points <= 3:
        problems.append(f"expected 1-3 takeaways, got {n_points}")
    n_kpis = len(content.get("kpis", []))
    if profile.get("rows") and not 1 <= n_kpis <= 3:
        problems.append(f"expected 1-3 findings, got {n_kpis}")

    return problems


# ── deterministic fallback (no LLM required) ─────────────────────────────────

_EYEBROW_RULES = [
    (r"call|reach|frequenc|coverage|rep visit|touch", "FIELD EXECUTION"),
    (r"share|competitor|versus|\bvs\.?\b|capture rate", "SHARE DYNAMICS"),
    (r"momentum|short[- ]term|recent (4|four)", "SHORT-TERM MOMENTUM"),
    (r"region|geograph|territor|state|district", "REGIONAL PERFORMANCE"),
    (r"trend|over time|evolv|past year|history", "DEMAND TREND"),
    (r"account|new business|adopt|breadth|depth", "ACCOUNT BASE"),
    (r"forecast|project|outlook", "OUTLOOK"),
    (r"revenue|sales|demand", "SALES PERFORMANCE"),
]


def _sentences(text: str) -> list[str]:
    text = _normalise_display_text(text)
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"])', text)
    return [p.strip() for p in parts if len(p.strip()) > 12]


def _eyebrow_for(question: str) -> str:
    q = (question or "").lower()
    for pattern, label in _EYEBROW_RULES:
        if re.search(pattern, q):
            return label
    return "ANALYSIS"


def _titlecase_question(q: str) -> str:
    q = _normalise_display_text(q or "Analysis")
    small = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or",
             "the", "to", "vs", "with", "is", "are", "we", "our"}
    words = q.split(" ")
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i and lw in small:
            out.append(lw)
        elif w.isupper() and len(w) <= 5:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    text = " ".join(out)
    return text[:1].upper() + text[1:]


def _incomplete_period_note(df, profile: dict) -> str:
    """Deterministically detect partial windows worth flagging."""
    if df is None or pd is None or profile.get("rows", 0) == 0:
        return ""
    notes = []
    for col in df.columns:
        up = str(col).upper()
        if re.search(r"IS_.*COMPLETE|_COMPLETE$", up):
            try:
                if int(df[col].iloc[-1]) == 0:
                    notes.append(f"{humanize_column(col)} indicates a partial window.")
            except Exception:
                pass
        if re.search(r"BUSINESS_DAY_COUNT|BUSINESS_DAYS", up) and _is_numeric(df[col]):
            try:
                series = df[col].dropna().astype(float)
                if len(series) > 2 and series.iloc[-1] < series.median() * 0.6:
                    label = profile.get("label_col")
                    when = (format_cell(label, df[label].iloc[-1])
                            if label is not None else "the final period")
                    notes.append(
                        f"Latest period ({when}) has only "
                        f"{int(series.iloc[-1])} business day(s) and is incomplete; "
                        "not comparable to full periods.")
            except Exception:
                pass
    return " ".join(dict.fromkeys(notes))[:190]


def _basis_from_text(text: str, columns: Sequence[str]) -> str:
    blob = (text or "").lower() + " " + " ".join(map(str, columns)).lower()
    has_pap = "pap" in blob or "patient assistance" in blob
    has_com = "commercial" in blob or "com_" in blob or blob.count("com ") > 0
    if has_pap and has_com:
        return "Commercial + PAP"
    if has_pap:
        return "Commercial + PAP"
    if has_com:
        return "Commercial only"
    return "Not stated"


_RATE_LIKE = re.compile(
    r"(AVG|AVERAGE|MEAN|DAILY|PER_|_PER|RATE|PCT|PERCENT|SHARE|GROWTH|"
    r"RATIO|INDEX|SCORE)", re.I)

_PERIOD_TAIL = re.compile(
    r"(\s+(R|P|L|T)\d{1,3}[WMQDY])+(\s+VS(\s+(R|P|L|T)\d{1,3}[WMQDY])+)?$", re.I)


def is_rate_like(col: str) -> bool:
    """Averages, rates and percentages must never be summed."""
    return bool(_RATE_LIKE.search(str(col).upper()))


def tidy_label(label: str, limit: int = 30) -> str:
    """
    Trim a metric label to something a KPI tile can show: drop trailing period
    tokens (the chips and value already carry the window) and cut on a word
    boundary rather than mid-word.
    """
    text = _normalise_display_text(label)

    def _drop_noun(value: str) -> str:
        value = re.sub(r"\s+(Assessment|Classification|Indicator|Metric)$", "",
                       value, flags=re.I).strip()
        # The value already carries the % sign; "Share % — 20%" reads badly.
        return re.sub(r"\s*%$", "", value).strip()

    # Shed information only as far as the width forces us to, most valuable
    # first: keep the window token ("R13W") unless dropping it is what makes
    # the label fit.
    without_noun = _drop_noun(text)
    without_period = _PERIOD_TAIL.sub("", text).strip()
    minimal = _drop_noun(without_period)
    for candidate in (text, without_noun, minimal, without_period):
        if len(candidate) <= limit:
            text = candidate
            break
    else:
        text = minimal
    # "Total R4W Total mg" -> "R4W Total mg"
    words, deduped = text.split(" "), []
    for word in words:
        if deduped and word.lower() == deduped[-1].lower():
            continue
        if (word.lower() in ("total", "average")
                and any(w.lower() == word.lower() for w in deduped)):
            continue
        deduped.append(word)
    if len(deduped) > 1 and deduped[0].lower() in ("total", "average") and \
            any(w.lower() == deduped[0].lower() for w in deduped[1:]):
        deduped = deduped[1:]
    text = " ".join(deduped)
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return (text[:cut] if cut > limit * 0.5 else text[:limit]).rstrip(" ,-") + "…"


def metric_score(col: str, series=None) -> int:
    """How much a column deserves to be on an executive slide."""
    up = str(col).upper()
    if _META_COL.search(up):
        return -10
    if series is not None and (_is_dateish(col, series) or _is_boolish(series)):
        return -10
    score = 0
    if re.search(r"(TOTAL|SALES|REVENUE|VOLUME|DEMAND|_MG\b|MG$|VIAL|UNIT|QTY)", up):
        score += 4
    if re.search(r"(GROWTH|PCT|PERCENT|SHARE|RATE|CHANGE)", up):
        score += 3
    if re.search(r"(AVG|AVERAGE|MEAN|DAILY)", up):
        score += 2
    if re.search(r"(ACCOUNT|CUSTOMER|HCP|CALL|REACH|FREQUENCY|COVERAGE)", up):
        score += 2
    if re.search(r"(ASSESSMENT|STATUS|CLASSIFICATION|DIRECTION|TIER)", up):
        score += 2
    if re.search(r"COUNT$", up):
        score -= 1
    return score


_PERIOD_SUFFIX = re.compile(r"^(?P<base>.+?)_(?P<kind>R|P)(?P<n>\d{1,3})(?P<u>[WMQDY])$",
                            re.I)


def _period_pairs(columns: Sequence[str]) -> list[tuple[str, str, str]]:
    """Find (prior, recent, base) column triples like X_P4W / X_R4W."""
    buckets: dict[tuple[str, str], dict[str, str]] = {}
    for col in columns:
        m = _PERIOD_SUFFIX.match(str(col))
        if not m:
            continue
        key = (m.group("base").upper(), f"{m.group('n')}{m.group('u')}".upper())
        buckets.setdefault(key, {})[m.group("kind").upper()] = col
    pairs = []
    for (base, window), got in buckets.items():
        if "R" in got and "P" in got:
            pairs.append((got["P"], got["R"], f"{humanize_column(base)} {window}"))
    return pairs


def _growth_for(base: str, columns: Sequence[str], row) -> str:
    """Reuse an explicit growth column when the query already computed one."""
    stem = re.sub(r"[^A-Z0-9]", "", base.upper())
    for col in columns:
        up = re.sub(r"[^A-Z0-9]", "", str(col).upper())
        if "GROWTH" in up and stem[:14] in up:
            try:
                return str(row[col]).strip()
            except Exception:
                return ""
    return ""


def _describe_column(col: str, series=None) -> str:
    """
    A short, honest note on what a value measures. Derived from the column
    itself — no filler when there is nothing specific to say.
    """
    up = str(col).upper()
    window = ""
    m = _PERIOD_SUFFIX.match(str(col))
    if m:
        unit = {"W": "week", "M": "month", "Q": "quarter",
                "D": "day", "Y": "year"}.get(m.group("u").upper(), "period")
        kind = "most recent" if m.group("kind").upper() == "R" else "prior"
        window = f" over the {kind} {m.group('n')} {unit}s"

    if re.search(r"(ASSESSMENT|CLASSIFICATION|DIRECTION|STATUS)", up):
        return f"Direction the query reports{window or ' for this window'}."
    if re.search(r"(GROWTH|CHANGE)", up):
        return f"Change versus the prior equivalent window{window}."
    if re.search(r"(PCT|PERCENT|SHARE|RATE)", up):
        return f"Share of the total{window}."
    if re.search(r"(AVG|AVERAGE|MEAN|DAILY)", up):
        return f"Average per business day{window}."
    if re.search(r"(TOTAL|VOLUME|SALES|REVENUE|DEMAND|_MG\b|MG$)", up):
        return f"Total volume{window or ' reported by the query'}."
    if re.search(r"(ACCOUNT|CUSTOMER|HCP)", up):
        return f"Count of accounts{window}."
    return f"Reported by the query{window}." if window else ""


def _kpi_number(col: str, value: float) -> str:
    """
    KPI values obey the same rules as the axes: whole numbers, and a %
    sign whenever the column is a percentage. Without this the chart says
    "34%" while the card beside it says "34.4", and a fraction column reads
    as "0.7" instead of "68%".
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if _PCT_HINT.search(str(col)):
        pct = v * 100.0 if abs(v) <= 1.5 else v
        return f"{pct:,.0f}%"
    # Match the axes: no "101.0" on a card beside a chart reading 100, 120.
    if abs(v) < 1000:
        return f"{v:,.0f}"
    return compact_number(v)


def _entity_value(entity: str, amount: str, limit: int = 30) -> str:
    """
    'Entity · amount', trimmed on the entity side only.

    Slicing the whole string would cut the number itself — "· 300" became
    "· 3" — quietly reporting a different figure to the reader.
    """
    amount = _normalise_display_text(amount)
    entity = _normalise_display_text(entity)
    room = limit - len(amount) - 3
    if room < 4:
        return amount[:limit]
    return f"{_shorten_label(entity, room)} · {amount}"


def _metric_unit_suffix(col: str, value: float) -> str:
    up = str(col or "").upper()
    if _VIAL_RE.search(up):
        return "vial" if abs(float(value)) == 1 else "vials"
    if re.search(r"(^|_)MG($|_)", up):
        return "mg"
    return ""


def _metric_descriptor(col: str) -> str:
    up = str(col or "").upper()
    if "DEMAND" in up:
        return "demand"
    if "SALES" in up:
        return "sales"
    if "REVENUE" in up:
        return "revenue"
    if _VIAL_RE.search(up) or re.search(r"(^|_)MG($|_)", up):
        return "volume"
    return humanize_column(col).lower()


def _entity_kpi_labels(label_col: str | None) -> tuple[str, str, str]:
    up = str(label_col or "").upper()
    if re.search(r"(ACCOUNT|CUSTOMER|PARENT)", up):
        return "Top Account", "Bottom Account", "account"
    if re.search(r"(HCP|PROVIDER|PHYSICIAN|PRESCRIBER)", up):
        return "Top HCP", "Bottom HCP", "HCP"
    if re.search(r"(REGION|TERRITORY|DISTRICT|STATE|GEOGRAPH)", up):
        return "Top Geography", "Bottom Geography", "geography"
    return "Top Group", "Bottom Group", "group"


def select_kpis(df, profile: dict, cfg: DeckConfig) -> list[dict]:
    """
    Deterministic KPI selection. Shape-aware, metadata-aware, and ordered
    national/total first — never business-day counts or completeness flags.
    """
    if df is None or pd is None or profile.get("rows", 0) == 0:
        return []

    cols = list(df.columns)
    shape = profile.get("shape")
    kpis: list[dict] = []

    # ── single row: prior -> current pairs, then top-scoring measures ──
    if shape == "single_row":
        row = df.iloc[0]
        used: set[str] = set()
        for prior, recent, base in sorted(
                _period_pairs(cols),
                key=lambda t: -metric_score(t[1], df[t[1]])):
            if len(kpis) >= cfg.max_kpis or metric_score(recent, df[recent]) <= 0:
                continue
            growth = _growth_for(base.split(" ")[0], cols, row)
            value = (f"{format_cell(prior, row[prior])} → "
                     f"{format_cell(recent, row[recent])}")
            if growth:
                value += f" ({growth})"
            kpis.append({"label": tidy_label(base), "value": value[:30],
                         "definition": "Recent window versus the prior "
                                       "equivalent window."})
            used.update({prior, recent})

        scored = sorted(((metric_score(c, df[c]), c) for c in cols if c not in used),
                        key=lambda t: -t[0])
        for score, col in scored:
            if len(kpis) >= cfg.max_kpis or score <= 0:
                break
            if any(k["label"].lower().startswith(humanize_column(col).lower()[:12])
                   for k in kpis):
                continue
            kpis.append({"label": tidy_label(humanize_column(col)),
                         "value": format_cell(col, row[col])[:30],
                         "definition": _describe_column(col, df[col])})
        return kpis[:cfg.max_kpis]

    # ── many rows: window totals, then the primary measure's shape ──
    metrics = profile.get("metrics") or {}

    # A constant numeric column that names a window total is the headline
    # number the query already computed — better than anything we can derive.
    row0 = df.iloc[0]
    for col in df.columns:
        if len(kpis) >= 1:
            break
        up = str(col).upper()
        if col not in (profile.get("constants") or {}):
            continue
        if not _is_numeric(df[col]) or _META_COL.search(up):
            continue
        if not re.search(r"(TOTAL|SUM|OVERALL|NATIONAL)", up):
            continue
        kpis.append({
            "label": tidy_label(humanize_column(col)),
            "value": format_cell(col, row0[col])[:30],
            "definition": "Window total reported by the query.",
        })

    ranked = sorted(((metric_score(c, df[c]), c) for c in metrics),
                    key=lambda t: (-t[0], is_rate_like(t[1])))
    candidates = [c for score, c in ranked if score > 0] or list(metrics)[:1]
    if not candidates:
        return kpis[:cfg.max_kpis]

    # The headline measure should be an absolute quantity, not a rate.
    primary = next((c for c in candidates if not is_rate_like(c)), candidates[0])
    m = metrics[primary]
    unit = _metric_unit_suffix(primary, m["max"])
    label = humanize_column(primary)
    period_word = "periods" if shape == "timeseries" else "groups"

    def _amount(v: float) -> str:
        text = _kpi_number(primary, v)
        return f"{text}{(' ' + unit) if unit and '%' not in text else ''}"[:30]

    # Skip a derived total that merely restates the window total above.
    already_total = any(
        abs((_token_to_float(k["value"]) or 0) - m["sum"]) <= 0.01 * abs(m["sum"] or 1)
        for k in kpis)

    if is_rate_like(primary):
        kpis.append({
            "label": tidy_label(f"Average {label}"),
            "value": _amount(m["mean"]),
            "definition": f"Mean across all {m['n']} {period_word} shown.",
        })
    elif not already_total:
        kpis.append({
            "label": tidy_label(f"Total {label}"),
            "value": _amount(m["sum"]),
            "definition": f"Sum across all {m['n']} {period_word} shown.",
        })

    if shape == "timeseries":
        kpis.append({
            "label": tidy_label(f"Average per {period_word[:-1].title()}"),
            "value": _amount(m["mean"]),
            "definition": f"Mean {label} per period over the window.",
        })
        kpis.append({
            "label": "Highest Period",
            "value": _entity_value(m["max_at"], _kpi_number(primary, m["max"])),
            "definition": f"Largest single-period {label} in the window.",
        })
    else:
        top_label, bottom_label, entity_noun = _entity_kpi_labels(profile.get("label_col"))
        metric_descriptor = _metric_descriptor(primary)
        kpis.append({
            "label": top_label,
            "value": _amount(m["max"]),
            "definition": f"{_normalise_display_text(m['max_at'])}, the highest-"
                          f"{metric_descriptor} {entity_noun} in the chart.",
        })
        kpis.append({
            "label": bottom_label,
            "value": _amount(m["min"]),
            "definition": f"{_normalise_display_text(m['min_at'])}, the lowest-"
                          f"{metric_descriptor} {entity_noun} in the chart.",
        })

    seen, unique = set(), []
    for kpi in kpis:
        key = kpi["label"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(kpi)
    return unique[:cfg.max_kpis]


_META_SENTENCE = re.compile(
    r"^(the analysis|this analysis|the query|this query|the result|the data|"
    r"it looks at|the report|we (?:looked|analysed|analyzed)|"
    r"the following|this (?:slide|view|chart))", re.I)

_FILLER = re.compile(
    r"^(what stands out is that|what is notable is that|notably,|overall,|"
    r"in summary,|importantly,|of note,|that said,|in addition,|"
    r"the result shows that|the result shows|this indicates that)\s*", re.I)


def _condense(sentence: str, limit: int = 175) -> str:
    """Trim editorial filler and, if still long, cut at a clause boundary."""
    text = _FILLER.sub("", _normalise_display_text(sentence))
    text = re.sub(r"^(the analysis answers a straightforward business question:"
                  r"|the analysis answers a business question:)\s*", "", text,
                  flags=re.I)
    text = text[:1].upper() + text[1:] if text else text
    if len(text) <= limit:
        return text
    for sep in (". ", "; ", ", while ", ", which ", ", and ", ", but ",
                ", including ", ", suggesting ", ", indicating "):
        idx = text.rfind(sep, 0, limit)
        if idx > limit * 0.34:
            return text[:idx].rstrip(" ,;") + "."
    cut = text.rfind(" ", 0, limit)
    return text[:cut if cut > 0 else limit].rstrip(" ,;") + "…"


def fallback_content(block: dict, df, profile: dict, cfg: DeckConfig) -> dict:
    """Slide content built entirely from the summary + verified facts."""
    sections = block.get("sections") or {}
    summary = block.get("summary") or ""
    question = block.get("question") or "Analysis"
    source_takeaways = _source_takeaway_points(block, cfg)

    headline_body = sections.get("overview") or summary
    body = (sections.get("takeaways") or sections.get("findings")
            or sections.get("overview") or summary)
    sents = (_sentences(headline_body)
             or _sentences(body)
             or _sentences(summary))

    def rank(sentence: str) -> tuple:
        meta = 1 if _META_SENTENCE.match(sentence) else 0
        has_num = 0 if re.search(r"\d", sentence) else 1
        return (meta, has_num, len(sentence))

    ordered = sorted(sents, key=rank)

    # Pick the headline first, then keep it out of the bullet pool so the
    # slide never says the same thing twice.
    headline_src = ordered[0] if ordered else question
    headline = (_condense(headline_src, 120) if ordered
                else _titlecase_question(question))

    def _key(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.lower())[:48]

    bullets, seen = [], {_key(headline)}
    if source_takeaways:
        for point in source_takeaways:
            key = _key(point)
            if point and key not in seen:
                bullets.append(point)
                seen.add(key)
            if len(bullets) == cfg.max_bullets:
                break
    else:
        for sentence in ordered:
            condensed = _condense(sentence)
            key = _key(condensed)
            if condensed and key not in seen:
                bullets.append(condensed)
                seen.add(key)
            if len(bullets) == cfg.max_bullets:
                break

    # Prefer implications for the insight; otherwise use whatever the bullets
    # did not already say.
    insight = sections.get("implications") or ""
    if not insight:
        pool = [x for x in _sentences(sections.get("overview") or summary)
                if _key(_condense(x)) not in seen]
        insight = " ".join(pool[-2:]) or headline
    insight = " ".join(_condense(x, 240) for x in _sentences(insight)) or insight
    insight = re.sub(r"\s+", " ", insight).strip()

    kpis = select_kpis(df, profile, cfg)

    log.info(
        "[deck-debug] fallback question=%s sentence_count=%d ordered_first=%s source_takeaways=%s",
        _debug_excerpt(question, 90),
        len(ordered),
        _debug_excerpt(ordered[0] if ordered else "", 180),
        _debug_list_preview(source_takeaways, 3) or "(none)",
    )
    log.info(
        "[deck-debug] fallback headline=%s bullets=%s kpis=%s",
        _debug_excerpt(headline, 180),
        _debug_list_preview(bullets, 3) or "(none)",
        ", ".join(
            f"{_debug_excerpt(item.get('label'), 24)}={_debug_excerpt(item.get('value'), 30)}"
            for item in kpis[:3]
        ) or "(none)",
    )

    return {
        "eyebrow": _eyebrow_for(question),
        "title": _titlecase_question(question),
        "headline": headline,
        "bullets": bullets,
        "kpis": kpis,
        "insight": insight[:520],
        "footnote": _incomplete_period_note(df, profile),
        "basis": _basis_from_text(summary, profile.get("columns") or []),
        "chart_caption": "",
    }


def generate_slide_content(block: dict, cfg: DeckConfig | None = None,
                           df=None, profile: dict | None = None) -> dict:
    """
    LLM content with deterministic guard rails.

    Any field the model cannot support with the data is replaced by the
    deterministic fallback rather than shipped unverified.
    """
    cfg = cfg or DeckConfig()
    if df is None:
        df = to_dataframe(block.get("data"))
    if profile is None:
        profile = profile_dataframe(df, getattr(cfg, "hooks", None))

    baseline = fallback_content(block, df, profile, cfg)
    facts = facts_block(df, profile)
    columns = list(profile.get("columns") or [])
    has_chart = bool(block.get("viz_code"))
    provider = _resolve_provider(cfg)

    log.info(
        "[deck-debug] content start question=%s provider=%s rows=%s shape=%s cols=%s has_chart=%s summary_len=%d",
        _debug_excerpt(block.get("question"), 100),
        provider,
        profile.get("rows"),
        profile.get("shape"),
        _debug_list_preview(columns, 6) or "(none)",
        has_chart,
        len(str(block.get("summary") or "")),
    )
    log.info(
        "[deck-debug] baseline headline=%s insight=%s bullets=%s",
        _debug_excerpt(baseline.get("headline"), 180),
        _debug_excerpt(baseline.get("insight"), 200),
        _debug_list_preview(baseline.get("bullets") or [], 3) or "(none)",
    )

    _debug_prompt_payload_once(
        question=block.get("question", ""),
        summary=(block.get("summary") or "(no summary provided)")[:6000],
        facts=facts,
        columns=columns,
        has_chart=has_chart,
    )

    if provider == "none":
        log.info("no LLM provider configured — using deterministic content")
        return enforce_content_rules(baseline, df, profile, cfg)

    user = USER_PROMPT.format(
        question=block.get("question", ""),
        summary=(block.get("summary") or "(no summary provided)")[:6000],
        facts=facts,
        columns=", ".join(map(str, columns)) or "(none)",
        has_chart=str(has_chart).lower(),
    )
    messages = [{"role": "user", "content": user}]

    content, problems = None, ["no response"]
    for attempt in range(cfg.llm_max_retries + 1):
        log.info("[deck-debug] llm attempt=%d/%d question=%s", attempt + 1,
                 cfg.llm_max_retries + 1, _debug_excerpt(block.get("question"), 80))
        reply = _call_llm(SYSTEM_PROMPT, messages, cfg)
        if reply is None:
            log.warning("[deck-debug] llm reply missing attempt=%d question=%s",
                        attempt + 1, _debug_excerpt(block.get("question"), 80))
            break
        raw = extract_json(reply)
        if raw is None:
            problems = ["reply was not parseable JSON"]
            log.warning("[deck-debug] llm reply parse failed attempt=%d preview=%s",
                        attempt + 1, _debug_excerpt(reply, 240))
        else:
            content = coerce_content(raw, block, cfg)
            log.info(
                "[deck-debug] llm content attempt=%d headline=%s bullets=%s kpis=%s insight=%s",
                attempt + 1,
                _debug_excerpt(content.get("headline"), 180),
                _debug_list_preview(content.get("bullets") or [], 3) or "(none)",
                ", ".join(
                    f"{_debug_excerpt(item.get('label'), 24)}={_debug_excerpt(item.get('value'), 30)}"
                    for item in (content.get("kpis") or [])[:3]
                ) or "(none)",
                _debug_excerpt(content.get("insight"), 200),
            )
            problems = verify_content(content, df, block, profile) if cfg.verify_numbers else []
            if not problems:
                log.info("content ok on attempt %d", attempt + 1)
                return _finalise_content(content, baseline, block, df, profile, cfg)
            log.warning("[deck-debug] verification failed attempt=%d problems=%s",
                        attempt + 1, problems)
        if attempt < cfg.llm_max_retries:
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user",
                 "content": _REPAIR_PROMPT.format(problem="\n".join(problems))},
            ]

    if content is None:
        log.warning("falling back to deterministic content: %s", problems)
        return enforce_content_rules(baseline, df, profile, cfg)

    # Partial rescue: keep what verified, replace what did not.
    log.warning("content had %d unresolved issue(s); repairing fields: %s",
                len(problems), problems)
    for problem in problems:
        if problem.startswith("bullet") or "3 bullets" in problem:
            content["bullets"] = baseline["bullets"]
        elif problem.startswith("kpi"):
            content["kpis"] = baseline["kpis"]
        elif problem.startswith("headline"):
            content["headline"] = baseline["headline"]
        elif problem.startswith("insight"):
            content["insight"] = baseline["insight"]
        elif problem.startswith("footnote"):
            content["footnote"] = baseline["footnote"]
        elif "superlative" in problem:
            content["headline"] = _SUPERLATIVE.sub("notable", content["headline"])
    return _finalise_content(content, baseline, block, df, profile, cfg)


def _finalise_content(content: dict, baseline: dict, block: dict, df, profile: dict,
                      cfg: DeckConfig | None = None) -> dict:
    """Deterministic last word: caveats and basis are never left to the model."""
    cfg = cfg or DeckConfig()
    source_takeaways = _source_takeaway_points(block or {}, cfg)
    forced_note = _incomplete_period_note(df, profile)
    if forced_note and forced_note.lower() not in (content.get("footnote") or "").lower():
        content["footnote"] = forced_note
    if not content.get("basis") or content["basis"] == "Not stated":
        content["basis"] = baseline["basis"]
    if not content.get("bullets"):
        content["bullets"] = baseline["bullets"]
    elif source_takeaways:
        content["bullets"] = source_takeaways
    if not content.get("headline"):
        content["headline"] = baseline["headline"]
    if not content.get("eyebrow"):
        content["eyebrow"] = baseline["eyebrow"]
    final_content = enforce_content_rules(content, df, profile, cfg)
    log.info(
        "[deck-debug] final content title=%s headline=%s bullets=%s kpis=%s basis=%s footnote=%s",
        _debug_excerpt(final_content.get("title"), 100),
        _debug_excerpt(final_content.get("headline"), 180),
        _debug_list_preview(final_content.get("bullets") or [], 3) or "(none)",
        ", ".join(
            f"{_debug_excerpt(item.get('label'), 24)}={_debug_excerpt(item.get('value'), 30)}"
            for item in (final_content.get("kpis") or [])[:3]
        ) or "(none)",
        _debug_excerpt(final_content.get("basis"), 50),
        _debug_excerpt(final_content.get("footnote"), 160),
    )
    return final_content


# ══════════════════════════════════════════════════════════════════════════════
# CHART RENDERING
# ══════════════════════════════════════════════════════════════════════════════

_SEMANTIC_KEEP = "keep"


def _is_semantic_color(hexish: Any) -> bool:
    """Red / green encode status — never repaint those."""
    c = hex_to_rgb(hexish) if isinstance(hexish, str) else None
    if c is None:
        return False
    h, s = hue(c), saturation(c)
    if s < 0.35:
        return False
    return (h <= 18 or h >= 342) or (90 <= h <= 165)


def _shorten_label(label: str, limit: int) -> str:
    text = _normalise_display_text(label)
    return text if len(text) <= limit else text[:limit - 1].rstrip(" ,-") + "…"


def _categories_of(fig) -> list:
    """Ordered x categories across traces, if this is a categorical chart."""
    seen, out = set(), []
    for trace in fig.data:
        raw = getattr(trace, "x", None)
        if raw is None:
            continue
        for value in list(raw):
            if isinstance(value, str) and value not in seen:
                seen.add(value)
                out.append(value)
    return out


def _fit_category_axis(fig, categories: list, px_w: int, px_h: int,
                       tick_pt: float) -> int:
    """
    Angle and truncate x tick labels so they cannot eat the plot.

    Rotated labels grow downward with the length of the longest one, so on a
    slide panel — which is wide and short — plotly's automargin will happily
    hand 60% of the height to the axis. The label length is therefore chosen
    from a pixel budget rather than left to the renderer, and automargin is
    turned off so the budget actually holds.

    Returns the bottom margin, in pixels, that the labels need.
    """
    if not categories:
        return int(46 * tick_pt / 10.0)

    glyph = tick_pt * 0.62                       # mean glyph width in px
    longest = max(len(str(c)) for c in categories)

    # Horizontal labels only stay horizontal if every one fits its own slot.
    slot_px = px_w / max(len(categories), 1)
    angled = longest * glyph > slot_px * 0.92

    if not angled:
        fig.update_xaxes(tickangle=0)
        return int(tick_pt * 2.4)

    angle = 35.0
    budget_px = px_h * 0.30                      # labels never exceed 30%
    limit = int(budget_px / (math.sin(math.radians(angle)) * glyph))
    limit = max(8, min(limit, 22))

    if longest > limit:
        display = [_shorten_label(c, limit) for c in categories]
        fig.update_xaxes(tickmode="array", tickvals=categories, ticktext=display)
        longest = max(len(d) for d in display)

    fig.update_xaxes(tickangle=-angle)
    return int(math.sin(math.radians(angle)) * longest * glyph) + int(tick_pt * 1.4)


def _is_number(value) -> bool:
    """
    numpy.float64 subclasses float but numpy.int64 does NOT subclass int, so
    an isinstance check silently skips every integer-valued axis. Test by
    conversion instead.
    """
    if isinstance(value, bool) or value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


_PCT_HINT = re.compile(r"(%|percent|pct|share|rate|proportion|growth)", re.I)


def _axis_percent_mode(fig, axis_key: str, values: list) -> str:
    """
    "" (not a percentage) | "fraction" (0-1) | "points" (0-100).

    Decided from the axis title and the value range together: 0.42 needs a
    x100 format, 42 only needs a % suffix, and mislabelling either one
    misstates the number by two orders of magnitude.
    """
    title = ""
    try:
        axis = fig.layout[axis_key]
        title = getattr(getattr(axis, "title", None), "text", "") or ""
    except Exception:
        pass
    if not _PCT_HINT.search(str(title)):
        return ""
    numeric = [float(v) for v in values if _is_number(v)]
    if not numeric:
        return ""
    return "fraction" if max(abs(v) for v in numeric) <= 1.5 else "points"


def _format_numeric_axes(fig) -> dict:
    """
    Whole numbers on every numeric axis, with a % sign where the axis is a
    percentage. Returns {axis_key: percent_mode} for the label formatter.
    """
    modes: dict[str, str] = {}
    for axis_key in [k for k in fig.layout if str(k).startswith(("xaxis", "yaxis"))]:
        letter = "x" if str(axis_key).startswith("x") else "y"
        suffix = str(axis_key)[5:]                    # "" , "2", "3" …
        values: list = []
        for trace in fig.data:
            bound = getattr(trace, f"{letter}axis", None) or letter
            if str(bound).lstrip(letter) != suffix:
                continue
            raw = getattr(trace, letter, None)
            if raw is not None:
                values.extend(list(raw))

        if not any(_is_number(v) for v in values):
            continue                                  # categorical or dates

        mode = _axis_percent_mode(fig, axis_key, values)
        modes[axis_key] = mode
        try:
            axis = fig.layout[axis_key]
            if mode == "fraction":
                axis.update(tickformat=".0%", ticksuffix="")
            elif mode == "points":
                axis.update(tickformat=",d", ticksuffix="%")
            else:
                axis.update(tickformat=",d", ticksuffix="")
        except Exception:
            continue
    return modes


def _add_bar_labels(fig, theme: Theme, tick_pt: float,
                    axis_modes: dict | None = None) -> bool:
    """Value labels above bars. Returns True if any were added."""
    added = False
    for trace in fig.data:
        if getattr(trace, "type", "") != "bar":
            continue
        try:
            # plotly stores x/y as numpy arrays, so `arr or []` raises rather
            # than falling back. Test for None explicitly.
            raw = getattr(trace, "y", None)
            values = [] if raw is None else list(raw)
            if not values or len(values) > 16:
                continue
            existing_text = getattr(trace, "text", None)
            has_text = existing_text is not None and len(list(existing_text)) > 0
            if getattr(trace, "texttemplate", None) or has_text:
                added = True
                continue
            mode = (axis_modes or {}).get("yaxis", "")
            if mode == "fraction":
                trace.texttemplate = "%{y:.0%}"
            elif mode == "points":
                trace.texttemplate = "%{y:,.0f}%"
            else:
                trace.texttemplate = "%{y:,.0f}"
            trace.textposition = "outside"
            trace.cliponaxis = False
            trace.textfont = dict(family=theme.font_body, size=tick_pt,
                                  color=rgb_to_hex(theme.ink_muted))
            added = True
        except Exception:
            continue
    return added


def _restyle_figure(fig, theme: Theme, *, base_pt: float = 12.0,
                    px_w: int = 900, px_h: int = 420) -> None:
    """Force the agent's figure onto the brand system. Deterministic."""
    try:
        import plotly.graph_objects as go
    except Exception:                                        # pragma: no cover
        return

    axis_font = dict(family=theme.font_body, size=base_pt,
                     color=rgb_to_hex(theme.ink_soft))
    tick_font = dict(family=theme.font_body, size=base_pt - 1,
                     color=rgb_to_hex(theme.ink_muted))
    grid = rgb_to_hex(mix(theme.hairline, RGBColor(0xFF, 0xFF, 0xFF), 0.35))

    fig.update_layout(
        title=None,                       # the slide owns the title
        template="plotly_white",
        font=dict(family=theme.font_body, size=base_pt,
                  color=rgb_to_hex(theme.ink_soft)),
        colorway=[rgb_to_hex(c) for c in (theme.chart_colors or [theme.primary])],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=base_pt - 1),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        showlegend=len(fig.data) > 1,
        bargap=0.28,
        uniformtext=dict(minsize=int(base_pt - 2), mode="hide"),
        width=None, height=None,          # our exporter owns the canvas
        autosize=True,
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor=grid, linewidth=1, ticks="outside", ticklen=4,
        tickcolor=grid, tickfont=tick_font, title_font=axis_font,
        # automargin stays ON so plotly reserves room for the *sideways*
        # overhang of the first rotated label, which otherwise gets sliced off
        # at the paper edge. Runaway growth is prevented by capping label
        # length in _fit_category_axis rather than by disabling automargin.
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=grid, gridwidth=1, zeroline=False,
        linecolor="rgba(0,0,0,0)", tickfont=tick_font, title_font=axis_font,
        automargin=True,
    )

    categories = _categories_of(fig)
    tick_pt = base_pt - 1

    # Tick labels already name the categories, so an axis title like
    # "Parent Name" only pushes the plot further up the panel.
    if categories:
        try:
            fig.update_xaxes(title_text="")
        except Exception:
            pass

    bottom_px = _fit_category_axis(fig, categories, px_w, px_h, tick_pt)
    axis_modes = _format_numeric_axes(fig)
    labelled = _add_bar_labels(fig, theme, tick_pt, axis_modes)

    # A rotated y title longer than the plot is taller than the plot; trim it.
    # Done per axis: update_yaxes() would copy one title onto every y axis and
    # silently destroy the secondary axis label on a dual-axis chart.
    plot_px = max(px_h - bottom_px - 34, 60)
    limit = max(10, int(plot_px / (tick_pt * 0.62)))
    for key in fig.layout:
        if not str(key).startswith("yaxis"):
            continue
        try:
            axis = fig.layout[key]
            title = getattr(getattr(axis, "title", None), "text", None)
            if title:
                axis.title.text = _shorten_label(title, limit)
        except Exception:
            continue

    # Floors only: automargin grows these when a label needs more room.
    fig.update_layout(margin=dict(
        l=int(base_pt * 4.6), r=int(base_pt * 2.2),
        t=int(base_pt * (3.0 if labelled else 1.8)),
        b=max(bottom_px, int(base_pt * 2.2)),
        pad=2,
    ))

    palette = [rgb_to_hex(c) for c in (theme.chart_colors or [theme.primary])]
    slot = 0
    for trace in fig.data:
        ttype = getattr(trace, "type", "")
        try:
            if ttype == "pie":
                trace.marker.colors = [palette[i % len(palette)]
                                       for i in range(len(trace.labels or []))]
                trace.textfont = dict(family=theme.font_body, size=base_pt - 1)
                continue

            current = None
            if ttype == "bar":
                current = getattr(getattr(trace, "marker", None), "color", None)
            elif ttype in ("scatter", "scattergl"):
                current = getattr(getattr(trace, "line", None), "color", None)

            if isinstance(current, str) and _is_semantic_color(current):
                continue                                   # status color: leave it

            color = palette[slot % len(palette)]
            slot += 1
            if ttype == "bar":
                trace.marker.color = color
                trace.marker.line = dict(width=0)
            elif ttype in ("scatter", "scattergl"):
                if getattr(trace, "line", None) is not None:
                    trace.line.color = color
                    if not trace.line.width:
                        trace.line.width = 2.6
                if getattr(trace, "marker", None) is not None:
                    trace.marker.color = color
        except Exception:
            continue

    # ISO date categories read badly on a slide; relabel them compactly.
    try:
        cats = next((list(t.x) for t in fig.data
                     if getattr(t, "x", None) is not None and len(t.x)), [])
        iso = [c for c in cats
               if isinstance(c, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", c)]
        if cats and len(iso) == len(cats):
            import datetime as _dt

            ticktext = [_dt.date.fromisoformat(c).strftime("%d %b") for c in cats]
            fig.update_xaxes(tickmode="array", tickvals=cats, ticktext=ticktext)
    except Exception:
        pass

    # Tick angle and truncation are handled by _fit_category_axis above.


# Palette assignments made by the polish layer are slot colours, not meaning.
# Recolouring them to the brand palette is safe; anything outside that set was
# an explicit choice by the visualization agent and is left alone.
_KNOWN_PALETTES = {
    "#1F4E79", "#E8853B", "#6A9FB5", "#8C8C8C", "#7E5A9B", "#4C8C5A",
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
}


def _apply_brand_colors(fig, theme: Theme, *, base_pt: float = 11.0) -> None:
    """
    Brand-only restyle. Leaves margins, height, legend placement, tick angles
    and data labels exactly as the polish layer set them — those decisions were
    made against the real plot geometry and must not be second-guessed here.
    """
    palette = [rgb_to_hex(c) for c in (theme.chart_colors or [theme.primary])]
    known = {c.upper() for c in _KNOWN_PALETTES}

    try:
        fig.update_layout(
            title=None,                       # the slide owns the title
            colorway=palette,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=theme.font_body,
                      color=rgb_to_hex(theme.ink_soft)),
        )
    except Exception:
        pass

    slot = 0
    for trace in fig.data:
        ttype = getattr(trace, "type", "")
        try:
            if ttype == "pie":
                colors = getattr(getattr(trace, "marker", None), "colors", None)
                if colors:
                    trace.marker.colors = [palette[i % len(palette)]
                                           for i in range(len(colors))]
                continue

            current = None
            if ttype == "bar":
                current = getattr(getattr(trace, "marker", None), "color", None)
            elif ttype in ("scatter", "scattergl"):
                current = (getattr(getattr(trace, "line", None), "color", None)
                           or getattr(getattr(trace, "marker", None), "color", None))

            is_slot_color = (isinstance(current, str)
                             and current.upper() in known)
            if (isinstance(current, str) and not is_slot_color
                    and _is_semantic_color(current)):
                slot += 1
                continue

            color = palette[slot % len(palette)]
            slot += 1
            if ttype == "bar":
                trace.marker.color = color
            elif ttype in ("scatter", "scattergl"):
                if getattr(trace, "line", None) is not None:
                    trace.line.color = color
                if getattr(trace, "marker", None) is not None:
                    trace.marker.color = color
        except Exception:
            continue


def _call_viz_layout(hook, fig, df, *, force_labels: bool = False):
    """Forward the slide-mode flags the hook understands, and only those."""
    import inspect

    kwargs = {}
    try:
        params = inspect.signature(hook).parameters
        if "for_slide" in params:
            kwargs["for_slide"] = True
        if "force_labels" in params:
            kwargs["force_labels"] = force_labels
    except (TypeError, ValueError):
        kwargs = {}
    return hook(fig, df, **kwargs)


def _resolve_label_thinner(hooks):
    """
    Find viz_polish's thin_line_labels, hook or not.

    A host app that builds DeckHooks by hand (the Streamlit frontend does)
    predates this hook and leaves it unset, so falling back to the module
    keeps the fix working without a frontend change.
    """
    fn = getattr(hooks, "viz_label_thinner", None)
    if callable(fn):
        return fn
    for name in ("viz_polish_2", "viz_polish"):
        try:
            fn = getattr(__import__(name), "thin_line_labels", None)
        except Exception:
            continue
        if callable(fn):
            return fn
    return None


def _label_text(value, mode: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if mode == "fraction":
        return f"{v * 100:,.0f}%"
    if mode == "points":
        return f"{v:,.0f}%"
    return f"{v:,.0f}"


def _retemplate_labels(fig, axis_modes: dict) -> None:
    """
    Re-format data labels viz_polish already decided to show.

    Only labels that exist are touched: viz_polish suppresses labels that
    would collide, and a blanket texttemplate would bring every one of them
    back. Where it selected points individually (texttemplate "%{text}") the
    pre-built text array is rewritten entry by entry, preserving the blanks.
    """
    for trace in fig.data:
        template = getattr(trace, "texttemplate", None)
        if not template:
            continue
        horizontal = str(getattr(trace, "orientation", "") or "") == "h"
        letter = "x" if horizontal else "y"
        bound = str(getattr(trace, f"{letter}axis", None) or letter)
        suffix = bound.lstrip(letter)
        mode = axis_modes.get(f"{letter}axis{suffix}", "")

        try:
            if "%{text}" in str(template):
                raw = getattr(trace, letter, None)
                texts = getattr(trace, "text", None)
                if raw is None or texts is None:
                    continue
                values, texts = list(raw), list(texts)
                trace.text = [
                    "" if (t is None or str(t).strip() == "")
                    else _label_text(values[i], mode)
                    for i, t in enumerate(texts) if i < len(values)
                ]
            elif mode == "fraction":
                trace.texttemplate = "%{" + letter + ":.0%}"
            elif mode == "points":
                trace.texttemplate = "%{" + letter + ":,.0f}%"
            else:
                trace.texttemplate = "%{" + letter + ":,.0f}"
        except Exception:
            continue


def _x_is_dateish(fig, min_points: int = 12) -> bool:
    """
    True when the x axis carries dates that plotly will auto-thin.

    Needs enough points that plotly actually reduces the tick count — a
    six-point date axis labels every point and those labels are long.
    """
    values: list = []
    for trace in fig.data:
        raw = getattr(trace, "x", None)
        if raw is not None:
            values.extend(list(raw))
    if len(values) < min_points:
        return False
    sample = values[: min(len(values), 40)]
    hits = 0
    for v in sample:
        if hasattr(v, "strftime"):
            hits += 1
        elif isinstance(v, str) and re.match(r"^\d{4}-\d{2}(-\d{2})?", v.strip()):
            hits += 1
    return hits >= len(sample) * 0.9


def _fit_polished_to_panel(fig, px_w: int, px_h: int) -> tuple[int, int]:
    """
    Re-fit a viz_polish figure to the slide panel.

    _set_geometry builds height as top + bottom + a fixed plot target, which
    assumes a taller canvas than a wide slide panel provides. Honouring that
    height letterboxes the card; clamping it while leaving the margins alone
    crushes the plot between them. So the margins are scaled down instead,
    keeping the plot area at a workable share of the panel, and the legend
    offset is recomputed against the new plot height.
    """
    try:
        margin = fig.layout.margin
        top = int(margin.t or 36)
        bottom = int(margin.b or 40)
        left = int(margin.l or 80)
        right = int(margin.r or 56)
    except Exception:
        top, bottom, left, right = 36, 40, 80, 56

    # 0.46 left a 333px panel with a 180px plot — a 26-point line with data
    # labels needs more than that before the labels start overprinting the
    # trace. The furniture (ticks, axis title, legend) gives up the difference.
    min_plot = px_h * 0.55
    if top + bottom > px_h - min_plot:
        room = max(px_h - min_plot, 40)
        shrink = room / float(top + bottom)
        top = max(int(top * shrink), 18)
        bottom = max(int(bottom * shrink), 26)

    left = min(left, int(px_w * 0.18))
    right = min(right, int(px_w * 0.10))

    # Tick angle and formatting are viz_polish's call and are left alone:
    # a pinned date axis already carries the week-ending labels it chose.

    fig.update_layout(width=px_w, height=px_h,
                      margin=dict(l=left, r=right, t=top, b=bottom, pad=4))

    try:
        if fig.layout.showlegend and fig.layout.legend.y is not None \
                and float(fig.layout.legend.y) < 0:
            plot_px = max(px_h - top - bottom, 80)
            fig.update_layout(legend=dict(y=-(bottom * 0.62) / plot_px,
                                          yanchor="top"))
    except Exception:
        pass

    return max(px_w - left - right, 60), max(px_h - top - bottom, 40)


def _fallback_figure(df, go):
    """
    Build a chart from the dataframe alone, with no generated code.

    The visualization agent's code is the one part of a slide that can fail
    at runtime, and when it does the slide silently drops to a table. The
    shape of the data is enough to pick a defensible chart deterministically:
    a date-like column plots as a line, anything else as bars, against the
    first one or two numeric columns.
    """
    if df is None or pd is None or len(df) == 0:
        return None
    try:
        numeric = [c for c in df.columns
                   if _is_numeric(df[c]) and not _META_COL.search(str(c))]
        if not numeric:
            return None

        dateish = [c for c in df.columns
                   if _DATEISH.search(str(c)) or
                   pd.api.types.is_datetime64_any_dtype(df[c])]
        axis = next((c for c in dateish if c not in numeric), None)
        if axis is None:
            axis = next((c for c in df.columns
                         if c not in numeric and df[c].nunique() > 1), None)
        if axis is None or df[axis].nunique() < 2:
            return None

        frame = df[[axis] + numeric[:2]].dropna(subset=[axis])
        if len(frame) == 0:
            return None
        if len(frame) > 60:
            frame = frame.tail(60)
        is_trend = axis in dateish

        fig = go.Figure()
        for col in numeric[:2]:
            if is_trend:
                fig.add_trace(go.Scatter(x=frame[axis], y=frame[col],
                                         mode="lines+markers",
                                         name=humanize_column(col)))
            else:
                fig.add_trace(go.Bar(x=frame[axis], y=frame[col],
                                     name=humanize_column(col)))
        fig.update_layout(xaxis_title=humanize_column(axis),
                          yaxis_title=humanize_column(numeric[0]))
        log.info("fallback chart built: %s vs %s",
                 axis, ", ".join(numeric[:2]))
        return fig
    except Exception as exc:
        log.warning("fallback chart failed: %s", exc)
        return None


def _prepare_chart_figure(block: dict, theme: Theme, target_w_in: float,
                          target_h_in: float,
                          cfg: DeckConfig) -> tuple[Any, int, int] | None:
    """
    Execute the visualization agent's code and return the exact polished
    Plotly figure that the PPT chart panel expects.
    """
    code = block.get("viz_code")
    df = to_dataframe(block.get("data"))
    if df is None or len(df) == 0:
        log.info("chart skipped: no dataframe")
        return None

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except Exception as exc:
        log.warning("plotly unavailable (%s) — chart skipped", exc)
        return None

    def _figure_from_payload(payload: Any):
        if not isinstance(payload, dict):
            return None
        try:
            return go.Figure(payload)
        except Exception as exc:
            log.warning("stored visualization figure unusable (%s)", exc)
            return None

    payload_figure = _figure_from_payload(block.get("viz_figure"))
    fig = None
    if not code:
        fig = payload_figure
        if fig is None:
            return None

    hooks = getattr(cfg, "hooks", None)

    if fig is None and getattr(hooks, "viz_sanitizer", None):
        try:
            code = hooks.viz_sanitizer(code)
        except Exception as exc:
            log.warning("viz_sanitizer failed (%s) — using raw code", exc)
    if fig is None and not str(code or "").strip():
        log.info("chart skipped: sanitizer returned nothing")
        fig = payload_figure

    ns: dict[str, Any] | None = None
    if fig is None and getattr(hooks, "viz_scope_builder", None):
        try:
            ns = dict(hooks.viz_scope_builder(df.copy()))
        except Exception as exc:
            log.warning("viz_scope_builder failed (%s) — using built-in scope",
                        exc)
            ns = None
    if fig is None and ns is None:
        ns = {"df": df.copy(), "pd": pd, "np": np, "go": go, "px": px,
              "make_subplots": make_subplots}

    if fig is None:
        ns.setdefault("__builtins__", __builtins__)
        try:
            exec(compile(str(code), "<viz_agent>", "exec"), ns, ns)
        except Exception as exc:
            log.warning("visualization code failed: %s\ncolumns=%s\ncode:\n%s",
                        exc, list(df.columns), str(code)[:600])
            fig = payload_figure
        else:
            fig = ns.get("fig")
            if fig is None:
                fig = next((v for v in ns.values()
                            if isinstance(v, go.Figure)), None)
            if fig is None:
                log.warning("visualization code produced no figure\ncode:\n%s",
                            str(code)[:600])
                fig = payload_figure

    if fig is None:
        fig = _fallback_figure(df, go)
    if fig is None:
        log.warning("chart skipped: no figure and no fallback for '%s'",
                    str(block.get("question", ""))[:80])
        return None

    px_w = int(max(320, min(cfg.chart_max_px, target_w_in * 96)))
    px_h = int(max(220, target_h_in * 96))

    polished = False
    if cfg.use_viz_polish_layout and getattr(hooks, "viz_layout", None):
        try:
            fig.update_layout(width=px_w, height=px_h)
            result = _call_viz_layout(hooks.viz_layout, fig, df,
                                      force_labels=cfg.chart_force_labels)
            if result is not None:
                fig = result
            polished = True
        except Exception as exc:
            log.warning("viz_layout failed (%s) — falling back to built-in restyle", exc)

    if polished:
        _apply_brand_colors(fig, theme)
        axis_modes = _format_numeric_axes(fig)
        plot_w, plot_h = _fit_polished_to_panel(fig, px_w, px_h)
        _fit_axis_titles(fig, px_w, px_h, plot_w, plot_h)
        thinner = _resolve_label_thinner(hooks)
        if thinner:
            try:
                thinner(fig, plot_w, plot_h)
            except Exception as exc:
                log.warning("label thinning failed (%s) — labels keep the nominal selection", exc)
        else:
            log.info("no thin_line_labels available; line labels were sized for a nominal plot, not this %dx%d panel", plot_w, plot_h)
        _retemplate_labels(fig, axis_modes)
    else:
        _restyle_figure(fig, theme, base_pt=10.0, px_w=px_w, px_h=px_h)

    fig = _json_safe_figure(fig)
    return fig, px_w, px_h


def _plotly_figure_json(fig: Any) -> dict[str, Any]:
    try:
        from plotly.utils import PlotlyJSONEncoder
        payload = json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        log.warning("chart figure JSON serialisation failed: %s", exc)
        return {}


def build_chart_render_spec(block: dict, theme: Theme, target_w_in: float,
                            target_h_in: float,
                            cfg: DeckConfig) -> dict[str, Any] | None:
    prepared = _prepare_chart_figure(block, theme, target_w_in, target_h_in, cfg)
    if prepared is None:
        return None
    fig, px_w, px_h = prepared
    figure = _plotly_figure_json(fig)
    if not figure:
        return None
    return {
        "figure": figure,
        "width": px_w,
        "height": px_h,
        "scale": max(cfg.chart_scale, 2),
    }


def render_chart(block: dict, theme: Theme, target_w_in: float,
                 target_h_in: float, cfg: DeckConfig) -> str | None:
    """
    Execute the visualization agent's code and export a PNG whose pixel aspect
    matches the panel exactly, so the picture is never stretched.
    """
    prepared = _prepare_chart_figure(block, theme, target_w_in, target_h_in, cfg)
    if prepared is None:
        return None
    fig, px_w, px_h = prepared
    os.makedirs(cfg.workdir, exist_ok=True)
    path_out = os.path.join(cfg.workdir, f"chart_{uuid.uuid4().hex[:10]}.png")
    try:
        fig.write_image(path_out, width=px_w, height=px_h,
                        scale=max(cfg.chart_scale, 2))
    except Exception as exc:
        log.warning("chart export failed at %dx%d (%s). Is kaleido==0.2.1 installed?", px_w, px_h, exc)
        return None
    if not os.path.exists(path_out):
        log.warning("chart export wrote no file to %s", path_out)
        return None
    return path_out
def _wrap_title(text: str, max_px: float, size_pt: float,
                max_lines: int = 3) -> tuple[str, int]:
    """
    Greedy word wrap to a pixel budget. Returns (text with <br>, n lines).

    A title too long for max_lines is cut on the last line with an ellipsis
    rather than allowed to run over — an overrunning last line is exactly the
    clipping this function exists to prevent.
    """
    words = str(text).split()
    if not words:
        return str(text), 1

    def fits(s: str) -> bool:
        return text_width_in(s, size_pt) * 96 <= max_px

    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if not current or fits(trial):
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        kept = lines[:max_lines]
        tail = " ".join(lines[max_lines - 1:])
        while tail and not fits(tail + "…"):
            tail = tail[:-1]
        kept[-1] = (tail + "…") if tail else kept[-1]
        lines = kept

    # A single word longer than the budget has no space to break on, so it
    # gets cut mid-word. Without this a column name like a run-on identifier
    # still runs off the canvas.
    for i, line in enumerate(lines):
        if fits(line):
            continue
        cut = line
        while cut and not fits(cut + "…"):
            cut = cut[:-1]
        lines[i] = (cut + "…") if cut else line
    return "<br>".join(lines), len(lines)


def _fit_axis_titles(fig, px_w: int, px_h: int,
                     plot_w: int, plot_h: int) -> None:
    """
    Stop long axis titles being clipped by the figure edge.

    A y-axis title is drawn rotated and centred on the PLOT, not the figure.
    When the plot is short — which it is on a slide panel, especially once
    rotated category labels have taken a third of the height — a title like
    "Commercial + PAP Demand Vials" is longer than the space above and below
    its own centre, so both ends run off the canvas and the reader sees
    "Commercial + PAP Demand V".

    Wrapping it onto a second line halves the length and costs one line of
    left margin, which is far cheaper than losing the units off the end.
    """
    try:
        margin = fig.layout.margin
        top = int(margin.t or 0)
        bottom = int(margin.b or 0)
        left = int(margin.l or 0)
    except Exception:
        return

    for name in ("yaxis", "yaxis2"):
        axis = getattr(fig.layout, name, None)
        if axis is None or not (axis.title and axis.title.text):
            continue
        size = float(getattr(axis.title.font, "size", None) or 12)
        # Rotated: the title runs along the figure's HEIGHT, centred on the
        # plot. Usable length is twice the smaller distance from that centre
        # to a figure edge.
        centre = top + plot_h / 2.0
        budget = 2 * min(centre, px_h - centre) - 8
        if text_width_in(axis.title.text, size) * 96 <= budget:
            continue
        wrapped, lines = _wrap_title(axis.title.text, max(budget, 60), size)
        axis.title.text = wrapped
        if lines > 1:
            # Each extra line of a rotated title eats horizontal room.
            grow = int((lines - 1) * size * 1.25 * 96 / 72)
            left = min(left + grow, int(px_w * 0.28))
            log.info("y-axis title wrapped to %d lines to fit %dpx", lines,
                     int(budget))

    for name in ("xaxis", "xaxis2"):
        axis = getattr(fig.layout, name, None)
        if axis is None or not (axis.title and axis.title.text):
            continue
        size = float(getattr(axis.title.font, "size", None) or 12)
        if text_width_in(axis.title.text, size) * 96 <= plot_w - 8:
            continue
        wrapped, lines = _wrap_title(axis.title.text, max(plot_w - 8, 60),
                                     size, max_lines=2)
        axis.title.text = wrapped
        if lines > 1:
            bottom = min(bottom + int(size * 1.25 * 96 / 72), int(px_h * 0.55))

    fig.update_layout(margin=dict(l=left, r=int(margin.r or 0),
                                  t=top, b=bottom,
                                  pad=getattr(margin, "pad", 8) or 8))


def _json_safe_figure(fig):
    """
    Make a figure survive PNG export, whatever ended up inside it.

    Plotly serialises through orjson when it is installed, and orjson refuses
    datetime SUBCLASSES — so a single pandas.Timestamp in tickvals or an x
    array aborts the export with "Type is not JSON serializable: Timestamp".
    The on-screen chart never takes that path, which is why a trend chart can
    look perfect in the app and be absent from the deck.

    The figure is only rebuilt when it actually fails to serialise, so a clean
    figure is returned untouched.
    """
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
    except Exception:
        return fig

    try:
        pio.to_json(fig)
        return fig
    except Exception as exc:
        log.info("figure not JSON-serialisable (%s) — scrubbing", exc)

    def scrub(value):
        if pd is not None:
            if value is getattr(pd, "NaT", None):
                return None
            if isinstance(value, pd.Timestamp):
                return value.to_pydatetime()
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(v) for v in value]
        if np is not None and isinstance(value, np.generic):
            return value.item()
        return value

    try:
        cleaned = go.Figure(scrub(fig.to_plotly_json()))
        pio.to_json(cleaned)
        return cleaned
    except Exception as exc:
        log.warning("figure scrub failed (%s) — exporting as-is", exc)
        return fig


def _image_size_in(path: str, fallback_ratio: float = 0.6) -> tuple[float, float]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        return w, h
    except Exception:
        return 1000, int(1000 * fallback_ratio)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_REF_W, _REF_H = 13.333, 7.5          # geometry is authored at this size


@dataclass
class Geometry:
    """All slide geometry, derived from the canvas so any size works."""
    w: float
    h: float

    def __post_init__(self) -> None:
        self.s = self.w / _REF_W                       # linear scale factor
        self.margin_x = 0.60 * self.s
        self.margin_t = 0.42 * self.s
        self.margin_b = 0.34 * self.s
        self.gap = 0.22 * self.s
        self.gap_lg = 0.28 * self.s
        self.pad = 0.20 * self.s
        self.radius = 0.055
        self.content_l = self.margin_x
        self.content_w = self.w - 2 * self.margin_x
        self.footer_h = 0.20 * self.s
        self.footer_y = self.h - self.margin_b - self.footer_h

    def pt(self, ref_pt: float) -> float:
        return max(round(ref_pt * self.s, 1), 5.5)


class Deck:
    """Builds the presentation. One instance per output file."""

    def __init__(self, cfg: DeckConfig, theme: Theme | None = None):
        self.cfg = cfg
        self.theme = theme or extract_theme(cfg.template_path)

        w = cfg.slide_width_in or self.theme.slide_w_in or _REF_W
        h = cfg.slide_height_in or self.theme.slide_h_in or _REF_H
        if abs(w / max(h, 0.01) - 16 / 9) > 0.35:          # odd canvas -> 16:9
            w, h = _REF_W, _REF_H

        self.prs = Presentation()
        self.prs.slide_width = Inches(w)
        self.prs.slide_height = Inches(h)
        self.g = Geometry(w, h)
        self._n = 0

    # ── primitives ───────────────────────────────────────────────────────

    def _blank(self):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._n += 1
        return slide

    def _fill_bg(self, slide, color: RGBColor) -> None:
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def rect(self, slide, l, t, w, h, fill: RGBColor | None,
             *, border: RGBColor | None = None, border_pt: float = 0.75,
             rounded: bool = True, radius: float | None = None):
        shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
        shape = slide.shapes.add_shape(shape_type, Inches(l), Inches(t),
                                       Inches(max(w, 0.01)), Inches(max(h, 0.01)))
        if rounded:
            try:
                shape.adjustments[0] = radius if radius is not None else self.g.radius
            except Exception:
                pass
        if fill is None:
            shape.fill.background()
        else:
            shape.fill.solid()
            shape.fill.fore_color.rgb = fill
        if border is None:
            shape.line.fill.background()
        else:
            shape.line.color.rgb = border
            shape.line.width = Pt(border_pt)
        try:
            shape.shadow.inherit = False
        except Exception:
            pass
        shape.text_frame.word_wrap = True
        return shape

    def dot(self, slide, cx, cy, d, color: RGBColor):
        if slide is None:
            return None
        shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2),
                                       Inches(cy - d / 2), Inches(d), Inches(d))
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()
        try:
            shape.shadow.inherit = False
        except Exception:
            pass
        return shape

    def chip(self, slide, text: str, l, t, h, color: RGBColor,
             bg: RGBColor, *, font=None, size=None, pad=None) -> float:
        """Pill-shaped label. Returns its width so callers can flow chips."""
        font = font or self.theme.font_body
        size = size or self.g.pt(8)
        pad = pad if pad is not None else 0.12 * self.g.s
        w = text_width_in(text, size, bold=True, font=font) + pad * 2
        if slide is None:
            return w
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l),
                                       Inches(t), Inches(w), Inches(h))
        shape.adjustments[0] = 0.5
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg
        shape.line.fill.background()
        try:
            shape.shadow.inherit = False
        except Exception:
            pass
        self.text(slide, text, l, t, w, h, size=size, bold=True, color=color,
                  font=font, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
                  fit=False)
        return w

    def text(self, slide, text: str, l, t, w, h, *,
             size: float, bold: bool = False, italic: bool = False,
             color: RGBColor | None = None, font: str | None = None,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             line_spacing: float = 1.16, tracking_pt: float = 0.0,
             fit: bool = True, min_size: float | None = None,
             max_lines: int | None = None, uppercase: bool = False):
        """
        Measured text. Pre-wrapped to explicit lines so nothing reflows into
        an overflow. Returns the height actually consumed, in inches.
        """
        color = color or self.theme.ink
        font = font or self.theme.font_body
        content = _normalise_display_text(text)
        if uppercase:
            content = content.upper()
        if not content:
            return 0.0

        if fit:
            size, lines = fit_text(
                content, w, h, max_pt=size,
                min_pt=min_size if min_size is not None else max(size * 0.62, 6.0),
                bold=bold, font=font, line_spacing=line_spacing,
                tracking_pt=tracking_pt, hard_max_lines=max_lines)
        else:
            lines = wrap_lines(content, w, size, bold=bold, font=font,
                               tracking_pt=tracking_pt, max_lines=max_lines)

        if slide is None:                       # measure-only pass
            return text_block_height_in(lines, size, line_spacing)

        box = slide.shapes.add_textbox(Inches(l), Inches(t),
                                       Inches(w), Inches(max(h, 0.05)))
        tf = box.text_frame
        tf.word_wrap = True
        try:
            tf.auto_size = MSO_AUTO_SIZE.NONE
        except Exception:
            pass
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor

        for i, line in enumerate(lines):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align
            try:
                para.line_spacing = line_spacing
                para.space_before = Pt(0)
                para.space_after = Pt(0)
            except Exception:
                pass
            run = para.add_run()
            run.text = line
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.italic = italic
            run.font.name = font
            run.font.color.rgb = color
            if tracking_pt:
                run._r.get_or_add_rPr().set("spc", str(int(tracking_pt * 100)))
        return text_block_height_in(lines, size, line_spacing)

    def measure(self, text_body: str, w: float, *, size: float,
                bold: bool = False, font: str | None = None,
                line_spacing: float = 1.16, tracking_pt: float = 0.0) -> float:
        """Height this text will occupy at *size* in a *w*-wide box."""
        lines = wrap_lines(text_body, w, size, bold=bold,
                           font=font or self.theme.font_body,
                           tracking_pt=tracking_pt)
        return text_block_height_in(lines, size, line_spacing)

    def text_runs(self, slide, segments, l, t, w, h, *, size: float,
                  font: str | None = None, line_spacing: float = 1.24,
                  align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
                  min_size: float | None = None, max_lines: int | None = None,
                  fit: bool = True) -> float:
        """
        One wrapped block whose parts carry different weight or colour.

        `segments` is [(text, {"bold":…, "italic":…, "color":…}), …]. The
        block is measured as a whole, then each wrapped line is split back
        into runs by character offset, so a bold label and a regular value
        share a line and wrap together.
        """
        font = font or self.theme.font_body
        cleaned_segments = []
        for text_part, style in segments:
            cleaned = _normalise_rich_text_fragment(text_part)
            if cleaned:
                cleaned_segments.append((cleaned, dict(style or {})))
        segments = cleaned_segments
        if not segments:
            return 0.0

        full = "".join(t for t, _ in segments)
        any_bold = any(st.get("bold") for _, st in segments)

        if fit:
            size, lines = fit_text(
                full, w, h, max_pt=size,
                min_pt=min_size if min_size is not None else max(size * 0.7, 6.5),
                bold=any_bold, font=font, line_spacing=line_spacing,
                hard_max_lines=max_lines)
        else:
            lines = wrap_lines(full, w, size, bold=any_bold, font=font,
                               max_lines=max_lines)
        if slide is None:
            return text_block_height_in(lines, size, line_spacing)

        box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w),
                                       Inches(max(h, 0.05)))
        tf = box.text_frame
        tf.word_wrap = True
        try:
            tf.auto_size = MSO_AUTO_SIZE.NONE
        except Exception:
            pass
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor

        # Walk the wrapped lines against the original string so we know which
        # segment each character came from. Collapsed whitespace means we
        # advance the cursor by matching characters, not by raw slicing.
        cursor = 0
        for idx, line in enumerate(lines):
            para = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            para.alignment = align
            try:
                para.line_spacing = line_spacing
                para.space_before = Pt(0)
                para.space_after = Pt(0)
            except Exception:
                pass

            pieces: list[tuple[str, dict]] = []
            for ch in line:
                while cursor < len(full) and full[cursor] != ch:
                    cursor += 1
                style = {}
                seen = 0
                for text_part, part_style in segments:
                    if seen + len(text_part) > cursor:
                        style = part_style
                        break
                    seen += len(text_part)
                if pieces and pieces[-1][1] is style:
                    pieces[-1] = (pieces[-1][0] + ch, style)
                else:
                    pieces.append((ch, style))
                cursor += 1

            for text_part, style in pieces:
                run = para.add_run()
                run.text = text_part
                run.font.size = Pt(size)
                run.font.bold = bool(style.get("bold"))
                run.font.italic = bool(style.get("italic"))
                run.font.name = font
                run.font.color.rgb = style.get("color") or self.theme.ink_soft
        return text_block_height_in(lines, size, line_spacing)

    def picture_fit(self, slide, path: str, l, t, w, h):
        """Insert preserving aspect ratio, centered in the box. Never stretches."""
        px_w, px_h = _image_size_in(path)
        ratio = px_h / max(px_w, 1)
        draw_w, draw_h = w, w * ratio
        if draw_h > h:
            draw_h, draw_w = h, h / max(ratio, 0.001)
        return slide.shapes.add_picture(
            path, Inches(l + (w - draw_w) / 2), Inches(t + (h - draw_h) / 2),
            Inches(draw_w), Inches(draw_h))

    def logo(self, slide, l, t, max_w, max_h, align: str = "left"):
        path = self.cfg.logo_path
        if not path or not os.path.exists(path):
            return None
        try:
            px_w, px_h = _image_size_in(path)
            ratio = px_h / max(px_w, 1)
            w, h = max_w, max_w * ratio
            if h > max_h:
                h, w = max_h, max_h / max(ratio, 0.001)
            # A tall logo does not fill max_w. Left-aligning it inside the box
            # then leaves a ragged gap between the mark and the slide edge,
            # so the corner placement asks for right alignment instead.
            x = l + (max_w - w) if align == "right" else l
            return slide.shapes.add_picture(path, Inches(x),
                                            Inches(t + (max_h - h) / 2),
                                            Inches(w), Inches(h))
        except Exception as exc:
            log.warning("logo insert failed: %s", exc)
            return None

    def footer_logo(self, slide) -> float:
        """
        Brand mark in the bottom-right corner. Returns the width it consumed,
        including the gap that follows it, so the slide number can step left
        of the mark rather than printing on top of it.
        """
        g, cfg = self.g, self.cfg
        if not cfg.footer_logo or not cfg.logo_path:
            return 0.0
        if not os.path.exists(cfg.logo_path):
            log.info("footer logo skipped: %s not found", cfg.logo_path)
            return 0.0

        max_w = cfg.footer_logo_w * g.s
        max_h = cfg.footer_logo_h * g.s
        # The mark is taller than the footer text band, so it is centred on
        # that band and allowed to overhang into the bottom margin.
        top = g.footer_y + (g.footer_h - max_h) / 2
        shape = self.logo(slide, g.content_l + g.content_w - max_w, top,
                          max_w, max_h, align="right")
        if shape is None:
            return 0.0
        try:
            used = shape.width / 914400.0          # EMU -> inches
        except Exception:
            used = max_w
        return used + 0.16 * g.s

    # ── preview manifest helpers ────────────────────────────────────────

    def _preview_box(self, l: float, t: float, w: float, h: float) -> dict[str, float]:
        return {
            "x": round(l, 4),
            "y": round(t, 4),
            "w": round(max(w, 0.0), 4),
            "h": round(max(h, 0.0), 4),
        }

    def _preview_align(self, align) -> str:
        if align == PP_ALIGN.CENTER:
            return "center"
        if align == PP_ALIGN.RIGHT:
            return "right"
        return "left"

    def _preview_text(self, text_body: str, l, t, w, h, *,
                      size: float, bold: bool = False, italic: bool = False,
                      color: RGBColor | None = None, font: str | None = None,
                      align=PP_ALIGN.LEFT, anchor: str = "top",
                      line_spacing: float = 1.16, tracking_pt: float = 0.0,
                      fit: bool = True, min_size: float | None = None,
                      max_lines: int | None = None, uppercase: bool = False) -> dict[str, Any] | None:
        color = color or self.theme.ink
        font = font or self.theme.font_body
        content = _normalise_display_text(text_body)
        if uppercase:
            content = content.upper()
        if not content:
            return None

        if fit:
            size, lines = fit_text(
                content, w, h, max_pt=size,
                min_pt=min_size if min_size is not None else max(size * 0.62, 6.0),
                bold=bold, font=font, line_spacing=line_spacing,
                tracking_pt=tracking_pt, hard_max_lines=max_lines)
        else:
            lines = wrap_lines(content, w, size, bold=bold, font=font,
                               tracking_pt=tracking_pt, max_lines=max_lines)

        return {
            "box": self._preview_box(l, t, w, h),
            "lines": lines,
            "fontSizePt": size,
            "lineSpacing": line_spacing,
            "lineHeightIn": round(line_height_in(size, line_spacing), 4),
            "fontFamily": font,
            "color": rgb_to_hex(color),
            "bold": bool(bold),
            "italic": bool(italic),
            "align": self._preview_align(align),
            "anchor": anchor,
            "trackingPt": tracking_pt,
            "uppercase": bool(uppercase),
        }

    def _preview_text_runs(self, segments, l, t, w, h, *, size: float,
                           font: str | None = None, line_spacing: float = 1.24,
                           align=PP_ALIGN.LEFT, anchor: str = "top",
                           min_size: float | None = None,
                           max_lines: int | None = None,
                           fit: bool = True) -> dict[str, Any] | None:
        font = font or self.theme.font_body
        cleaned_segments = []
        for text_part, style in segments or []:
            cleaned = _normalise_rich_text_fragment(text_part)
            if cleaned:
                cleaned_segments.append((cleaned, dict(style or {})))
        segments = cleaned_segments
        if not segments:
            return None

        full = "".join(text for text, _ in segments)
        any_bold = any(style.get("bold") for _, style in segments)
        if fit:
            size, lines = fit_text(
                full, w, h, max_pt=size,
                min_pt=min_size if min_size is not None else max(size * 0.7, 6.5),
                bold=any_bold, font=font, line_spacing=line_spacing,
                hard_max_lines=max_lines)
        else:
            lines = wrap_lines(full, w, size, bold=any_bold, font=font,
                               max_lines=max_lines)

        line_runs: list[list[dict[str, Any]]] = []
        cursor = 0
        for line in lines:
            pieces: list[dict[str, Any]] = []
            line_cursor = cursor
            for ch in line:
                while line_cursor < len(full) and full[line_cursor] != ch:
                    line_cursor += 1
                style = {}
                seen = 0
                for text_part, part_style in segments:
                    if seen + len(text_part) > line_cursor:
                        style = part_style
                        break
                    seen += len(text_part)
                color = rgb_to_hex(style.get("color") or self.theme.ink_soft)
                bold_flag = bool(style.get("bold"))
                italic_flag = bool(style.get("italic"))
                if (pieces and pieces[-1]["color"] == color
                        and pieces[-1]["bold"] == bold_flag
                        and pieces[-1]["italic"] == italic_flag):
                    pieces[-1]["text"] += ch
                else:
                    pieces.append({
                        "text": ch,
                        "color": color,
                        "bold": bold_flag,
                        "italic": italic_flag,
                    })
                line_cursor += 1
            cursor = line_cursor
            line_runs.append(pieces)

        return {
            "box": self._preview_box(l, t, w, h),
            "lineRuns": line_runs,
            "fontSizePt": size,
            "lineSpacing": line_spacing,
            "lineHeightIn": round(line_height_in(size, line_spacing), 4),
            "fontFamily": font,
            "align": self._preview_align(align),
            "anchor": anchor,
        }

    def _preview_theme(self) -> dict[str, Any]:
        th = self.theme
        return {
            "background": rgb_to_hex(th.bg),
            "backgroundDark": rgb_to_hex(th.bg_dark),
            "surface": rgb_to_hex(th.surface),
            "surfaceAlt": rgb_to_hex(th.surface_alt),
            "hairline": rgb_to_hex(th.hairline),
            "ink": rgb_to_hex(th.ink),
            "inkSoft": rgb_to_hex(th.ink_soft),
            "inkMuted": rgb_to_hex(th.ink_muted),
            "inkInverse": rgb_to_hex(th.ink_inverse),
            "primary": rgb_to_hex(th.primary),
            "accents": [rgb_to_hex(color) for color in th.accents],
            "chartColors": [rgb_to_hex(color) for color in th.chart_colors],
            "positive": rgb_to_hex(th.positive),
            "negative": rgb_to_hex(th.negative),
            "neutral": rgb_to_hex(th.neutral),
            "fontHead": th.font_head,
            "fontBody": th.font_body,
        }

    def _preview_card_shell(self, label: str, l, t, w, h) -> tuple[dict[str, Any], float, float, float, float]:
        g, th = self.g, self.theme
        pad = g.pad
        head_h = 0.20 * g.s
        heading = self._preview_text(
            label, l + pad, t + pad - 0.02 * g.s, w - 2 * pad, head_h,
            size=g.pt(9.5), bold=True,
            color=ensure_contrast(th.primary, th.surface, 4.5),
            font=th.font_body, tracking_pt=1.4, uppercase=True,
            max_lines=1, fit=False,
        )
        inner_t = t + pad + head_h + 0.12 * g.s
        panel = {
            "box": self._preview_box(l, t, w, h),
            "fill": rgb_to_hex(th.surface),
            "borderColor": rgb_to_hex(th.hairline),
            "borderWidthPt": 0.75,
            "radius": g.radius,
            "heading": heading,
        }
        return panel, l + pad, inner_t, w - 2 * pad, max(t + h - pad - inner_t, 0.2 * g.s)

    def _preview_bulleted_card(self, label: str, entries: list, l, t, w, h,
                               *, size_pt: float, min_pt: float) -> dict[str, Any] | None:
        g, th = self.g, self.theme
        if not entries:
            return None

        panel, inner_l, inner_t, inner_w, inner_h = self._preview_card_shell(label, l, t, w, h)
        dot_d = 0.075 * g.s
        text_l = inner_l + dot_d + 0.13 * g.s
        text_w = max(inner_w - dot_d - 0.13 * g.s, 0.5 * g.s)
        spacing = 1.24

        size = size_pt
        while True:
            note_pt = max(round(size * 0.78, 1), 6.5)
            layout = self._bullet_layout(entries, text_w, size, note_pt, spacing)
            if layout["total"] <= inner_h or size <= min_pt:
                break
            size = round(size - 0.5, 2)

        blocks, gaps = layout["blocks"], layout["gaps"]
        note_gap = layout["note_gap"]
        if layout["total"] > inner_h:
            room = max(inner_h - sum(main + note for main, note in blocks), 0.0)
            wanted = note_gap * sum(1 for _, note in blocks if note) + sum(gaps)
            shrink = (room / wanted) if wanted > 0 else 0.0
            shrink = max(min(shrink, 1.0), 0.0)
            gaps = [value * shrink for value in gaps]
            note_gap *= shrink

        panel["entries"] = []
        panel["innerBox"] = self._preview_box(inner_l, inner_t, inner_w, inner_h)
        y = inner_t
        note_pt = max(round(size * 0.78, 1), 6.5)
        for index, entry in enumerate(entries):
            main_h, note_h = blocks[index]
            dot_y = y + line_height_in(size, spacing) / 2 - dot_d / 2
            panel["entries"].append({
                "dotBox": self._preview_box(inner_l, dot_y, dot_d, dot_d),
                "dotColor": rgb_to_hex(ensure_contrast(th.accent(index), th.surface, 2.4)),
                "text": self._preview_text_runs(
                    entry["segments"], text_l, y, text_w, main_h + 0.05 * g.s,
                    size=size, line_spacing=spacing, min_size=min_pt, fit=False,
                ),
                "note": self._preview_text(
                    entry.get("note", ""), text_l, y + main_h + note_gap,
                    text_w, note_h + 0.04 * g.s, size=note_pt, italic=True,
                    color=th.ink_muted, font=th.font_body, line_spacing=1.14,
                    max_lines=2, fit=False,
                ) if note_h else None,
            })
            if note_h:
                y += note_gap
            y += main_h + note_h + (gaps[index] if index < len(gaps) else 0.0)
        return panel

    def _preview_findings_panel(self, kpis: list[dict], l, t, w, h) -> dict[str, Any] | None:
        entries = [self._finding_entry(kpi) for kpi in kpis or []]
        entries = [entry for entry in entries if entry]
        return self._preview_bulleted_card(
            self.cfg.findings_label, entries, l, t, w, h,
            size_pt=self.g.pt(11), min_pt=self.g.pt(8.5),
        )

    def _preview_takeaways_panel(self, points: list[str], l, t, w, h) -> dict[str, Any] | None:
        th = self.theme
        entries = [{
            "segments": [(str(point).strip(), {"color": th.ink_soft})],
            "note": "",
        } for point in points or [] if str(point).strip()]
        return self._preview_bulleted_card(
            self.cfg.takeaways_label, entries, l, t, w, h,
            size_pt=self.g.pt(11), min_pt=self.g.pt(8.5),
        )

    def _preview_header(self, model: dict) -> tuple[dict[str, Any], float]:
        g, t = self.g, self.theme
        y = g.margin_t
        header: dict[str, Any] = {"chips": []}

        if self.cfg.show_eyebrow and model.get("eyebrow"):
            eyebrow_h = 0.22 * g.s
            header["eyebrow"] = self._preview_text(
                model["eyebrow"], g.content_l, y, g.content_w, eyebrow_h,
                size=g.pt(9.5), bold=True,
                color=ensure_contrast(t.primary, t.bg, 4.5),
                font=t.font_body, tracking_pt=1.5, uppercase=True,
                max_lines=1, fit=False,
            )
            y += eyebrow_h + 0.09 * g.s

        title_h = 0.86 * g.s
        header["title"] = self._preview_text(
            model.get("title", ""), g.content_l, y, g.content_w, title_h,
            size=g.pt(22), bold=True, color=t.ink, font=t.font_head,
            line_spacing=1.08, min_size=g.pt(15), max_lines=2,
        )
        title_block = header.get("title")
        if title_block:
            y += text_block_height_in(title_block["lines"], title_block["fontSizePt"], 1.08)

        headline = model.get("headline", "")
        if headline:
            y += 0.10 * g.s
            headline_h = 0.52 * g.s
            header["headline"] = self._preview_text(
                headline, g.content_l, y, g.content_w, headline_h,
                size=g.pt(13), bold=True,
                color=ensure_contrast(t.primary, t.bg, 5.0),
                font=t.font_body, line_spacing=1.14,
                min_size=g.pt(10), max_lines=2,
            )
            headline_block = header.get("headline")
            if headline_block:
                y += text_block_height_in(headline_block["lines"], headline_block["fontSizePt"], 1.14)

        chips = model.get("chips") or []
        if chips:
            y += 0.12 * g.s
            cx, chip_h = g.content_l, 0.24 * g.s
            for chip_text in chips:
                chip_color = ensure_contrast(t.primary, t.surface_alt, 4.5)
                chip_w = self.chip(None, chip_text, cx, y, chip_h, chip_color,
                                   t.surface_alt, size=g.pt(8))
                header["chips"].append({
                    "box": self._preview_box(cx, y, chip_w, chip_h),
                    "fill": rgb_to_hex(t.surface_alt),
                    "radius": 0.5,
                    "text": self._preview_text(
                        chip_text, cx, y, chip_w, chip_h, size=g.pt(8), bold=True,
                        color=chip_color, font=t.font_body, align=PP_ALIGN.CENTER,
                        anchor="middle", fit=False,
                    ),
                })
                cx += chip_w + 0.10 * g.s
                if cx > g.content_l + g.content_w - 1.0:
                    break
            y += chip_h

        y += 0.20 * g.s
        header["bottomY"] = round(y, 4)
        return header, y

    def _preview_footer(self, model: dict, slide_number: int) -> dict[str, Any]:
        g, th = self.g, self.theme
        cfg = self.cfg
        left_bits = []
        if cfg.show_footnote and model.get("footnote"):
            left_bits.append(model["footnote"])
        if cfg.show_basis:
            basis = model.get("basis")
            if basis and basis != "Not stated":
                left_bits.append(f"Demand basis: {basis}")
            elif basis == "Not stated":
                left_bits.append("Demand basis not stated in source")
        if cfg.confidential_note:
            left_bits.append(cfg.confidential_note)
        left = "   ·   ".join(left_bits)

        footer: dict[str, Any] = {}
        logo_w = 0.0
        if cfg.footer_logo and cfg.logo_path and os.path.exists(cfg.logo_path):
            max_w = cfg.footer_logo_w * g.s
            max_h = cfg.footer_logo_h * g.s
            px_w, px_h = _image_size_in(cfg.logo_path)
            ratio = px_h / max(px_w, 1)
            draw_w, draw_h = max_w, max_w * ratio
            if draw_h > max_h:
                draw_h, draw_w = max_h, max_h / max(ratio, 0.001)
            x = g.content_l + g.content_w - draw_w
            y = g.footer_y + (g.footer_h - draw_h) / 2
            logo_w = draw_w + 0.16 * g.s
            footer["logo"] = {
                "box": self._preview_box(x, y, draw_w, draw_h),
                "asset": "aadibio_logo",
                "preserveAspectRatio": "xMaxYMid meet",
            }

        right_w = 0.5 * g.s if cfg.slide_numbers else 0.0
        if left:
            footer["text"] = self._preview_text(
                left, g.content_l, g.footer_y,
                g.content_w - logo_w - right_w - 0.12 * g.s, g.footer_h,
                size=g.pt(7.5), color=th.ink_muted, font=th.font_body,
                anchor="middle", max_lines=1,
            )
        if cfg.slide_numbers:
            footer["slideNumber"] = self._preview_text(
                str(slide_number),
                g.content_l + g.content_w - logo_w - right_w,
                g.footer_y, right_w, g.footer_h,
                size=g.pt(8), color=th.ink_muted, font=th.font_body,
                align=PP_ALIGN.RIGHT, anchor="middle", fit=False,
            )
        return footer

    def _preview_table_panel(self, df, profile: dict, l, t, w, h) -> dict[str, Any] | None:
        if df is None or pd is None or len(df) == 0:
            return None
        g, th = self.g, self.theme

        drop = set(profile.get("constants") or {})
        drop.update(profile.get("drop_cols") or [])
        cols = [c for c in df.columns if c not in drop] or list(df.columns)

        max_rows = min(self.cfg.max_table_rows, len(df))
        truncated = len(df) > max_rows
        body = df.tail(max_rows) if profile.get("shape") == "timeseries" else df.head(max_rows)

        head_pt, cell_pt = g.pt(8.5), g.pt(9)
        cell_pad = 0.09 * g.s

        def col_width(col) -> float:
            header = humanize_column(col)
            widest = text_width_in(header, head_pt, bold=True, font=th.font_body)
            for value in body[col].head(max_rows):
                widest = max(widest, text_width_in(format_cell(col, value), cell_pt,
                                                   font=th.font_body))
            return widest + cell_pad * 2 + 0.06 * g.s

        widths = {col: col_width(col) for col in cols}
        while len(cols) > 2 and sum(widths[col] for col in cols) > w:
            victim = min(cols[1:], key=lambda col: (0 if _is_numeric(df[col]) else 1,
                                                    -widths[col]))
            cols.remove(victim)

        total = sum(widths[col] for col in cols)
        if total <= 0:
            return None
        if total < w:
            surplus = w - total
            first_extra = min(surplus * 0.5, widths[cols[0]] * 0.6)
            widths[cols[0]] += first_extra
            share = (surplus - first_extra) / max(len(cols) - 1, 1)
            for col in cols[1:]:
                widths[col] += share
        else:
            scale = w / total
            widths = {col: widths[col] * scale for col in cols}

        note_h = 0.20 * g.s if truncated else 0.0
        usable_h = max(h - note_h, 0.6 * g.s)
        n_rows = len(body) + 1
        row_h = min(0.38 * g.s, max(0.24 * g.s, usable_h / max(n_rows, 1)))
        table_h = row_h * n_rows
        table_t = t + max((usable_h - table_h) / 2, 0)

        panel: dict[str, Any] = {
            "box": self._preview_box(l, t, w, h),
            "tableBox": self._preview_box(l, table_t, w, table_h),
            "headerFontSizePt": head_pt,
            "cellFontSizePt": cell_pt,
            "rowHeightIn": round(row_h, 4),
            "cellPaddingIn": round(cell_pad, 4),
            "headerFill": rgb_to_hex(th.surface_alt),
            "headerTextColor": rgb_to_hex(ensure_contrast(th.primary, th.surface_alt, 5.0)),
            "bodyTextColor": rgb_to_hex(th.ink_soft),
            "columns": [],
            "rows": [],
        }

        for col in cols:
            panel["columns"].append({
                "key": str(col),
                "label": humanize_column(col),
                "widthIn": round(widths[col], 4),
                "align": "right" if _is_numeric(df[col]) else "left",
            })

        for row_index in range(len(body)):
            row_fill = rgb_to_hex(th.bg if row_index % 2 == 0 else th.surface)
            row = []
            for col in cols:
                row.append({
                    "text": format_cell(col, body[col].iloc[row_index]),
                    "align": "right" if _is_numeric(df[col]) else "left",
                    "fill": row_fill,
                })
            panel["rows"].append(row)

        if truncated:
            shown = "most recent" if profile.get("shape") == "timeseries" else "first"
            panel["note"] = self._preview_text(
                f"Showing the {shown} {max_rows} of {len(df)} rows",
                l, table_t + table_h + 0.05 * g.s, w, note_h,
                size=g.pt(7.5), italic=True, color=th.ink_muted,
                font=th.font_body, max_lines=1,
            )
        return panel

    def _preview_compose_chart(self, model: dict, body_t: float, floor: float) -> dict[str, Any]:
        g, th = self.g, self.theme
        findings = model.get("kpis") or []
        takeaways = model.get("bullets") or []

        band_h = 0.0
        if takeaways:
            band_h = min(self.takeaways_height(takeaways, g.content_w), 2.05 * g.s)
        band_t = floor - band_h if band_h else floor
        body_b = band_t - (0.24 * g.s if band_h else 0.0)
        body_h = max(body_b - body_t, 1.2 * g.s)
        rail_w = g.content_w * 0.31 if findings else 0.0
        chart_w = g.content_w - rail_w - (g.gap if rail_w else 0.0)
        pad = 0.11 * g.s

        manifest: dict[str, Any] = {}
        manifest["chartPanel"] = {
            "box": self._preview_box(g.content_l, body_t, chart_w, body_h),
            "fill": rgb_to_hex(th.bg),
            "borderColor": rgb_to_hex(th.hairline),
            "borderWidthPt": 0.75,
            "radius": g.radius,
            "imageBox": self._preview_box(g.content_l + pad, body_t + pad,
                                          chart_w - 2 * pad, body_h - 2 * pad),
            "chartFit": "contain",
        }
        if findings:
            manifest["findingsPanel"] = self._preview_findings_panel(
                findings, g.content_l + chart_w + g.gap, body_t, rail_w, body_h
            )
        if takeaways and band_h > 0:
            manifest["takeawaysPanel"] = self._preview_takeaways_panel(
                takeaways, g.content_l, band_t, g.content_w, band_h
            )
        return manifest

    def _preview_compose_metrics(self, model: dict, body_t: float, floor: float) -> dict[str, Any]:
        g = self.g
        findings = model.get("kpis") or []
        takeaways = model.get("bullets") or []
        body_h = max(floor - body_t, 1.0 * g.s)

        target_w = min(g.content_w, 6.9 * g.s)
        x = g.content_l + (g.content_w - target_w) / 2
        findings_h = self.findings_height(findings, target_w) if findings else 0.0
        takeaways_h = self.takeaways_height(takeaways, target_w) if takeaways else 0.0
        gap = 0.22 * g.s if findings_h and takeaways_h else 0.0
        total_h = findings_h + gap + takeaways_h
        start_y = body_t + max((body_h - total_h) / 2, 0.0)

        manifest: dict[str, Any] = {}
        if findings_h:
            manifest["findingsPanel"] = self._preview_findings_panel(
                findings, x, start_y, target_w, findings_h
            )
        if takeaways_h:
            manifest["takeawaysPanel"] = self._preview_takeaways_panel(
                takeaways, x, start_y + findings_h + gap, target_w, takeaways_h
            )
        return manifest

    def _preview_compose_table(self, model: dict, body_t: float, floor: float) -> dict[str, Any]:
        g = self.g
        findings = model.get("kpis") or []
        takeaways = model.get("bullets") or []

        band_h = 0.0
        if takeaways:
            band_h = min(self.takeaways_height(takeaways, g.content_w), 2.05 * g.s)
        band_t = floor - band_h if band_h else floor
        body_b = band_t - (0.24 * g.s if band_h else 0.0)
        body_h = max(body_b - body_t, 1.35 * g.s)

        right_w = g.content_w * 0.31 if findings else 0.0
        left_w = g.content_w - right_w - (g.gap if right_w else 0.0)
        findings_h = min(self.findings_height(findings, right_w), body_h) if findings else 0.0
        table_h = max(body_h, findings_h) if findings else body_h

        manifest: dict[str, Any] = {
            "tablePanel": self._preview_table_panel(
                model.get("df"), model.get("profile") or {},
                g.content_l, body_t, left_w, table_h
            )
        }
        if findings:
            findings_t = body_t + max((table_h - findings_h) / 2, 0.0)
            manifest["findingsPanel"] = self._preview_findings_panel(
                findings, g.content_l + left_w + g.gap, findings_t, right_w, findings_h
            )
        if takeaways and band_h > 0:
            manifest["takeawaysPanel"] = self._preview_takeaways_panel(
                takeaways, g.content_l, band_t, g.content_w, band_h
            )
        return manifest

    def _preview_compose_narrative(self, model: dict, body_t: float, floor: float) -> dict[str, Any]:
        g = self.g
        takeaways = model.get("bullets") or []
        if not takeaways:
            return {}
        body_h = max(floor - body_t, 1.0 * g.s)
        target_w = min(g.content_w, 7.2 * g.s)
        card_h = min(self.takeaways_height(takeaways, target_w), body_h)
        x = g.content_l + (g.content_w - target_w) / 2
        y = body_t + max((body_h - card_h) / 2, 0.0)
        return {
            "takeawaysPanel": self._preview_takeaways_panel(
                takeaways, x, y, target_w, card_h
            )
        }

    def build_preview_manifest(self, model: dict, slide_number: int = 1) -> dict[str, Any]:
        g = self.g
        header, body_t = self._preview_header(model)
        floor = g.footer_y - 0.22 * g.s
        layout = model.get("layout", "narrative")

        manifest: dict[str, Any] = {
            "layout": layout,
            "slideNumber": slide_number,
            "canvas": {
                "width": round(g.w, 4),
                "height": round(g.h, 4),
                "aspectRatio": round(g.w / max(g.h, 0.001), 4),
            },
            "background": rgb_to_hex(self.theme.bg),
            "theme": self._preview_theme(),
            "header": header,
            "footer": self._preview_footer(model, slide_number),
        }

        if layout == "chart_split":
            manifest.update(self._preview_compose_chart(model, body_t, floor))
        elif layout == "metric_grid":
            manifest.update(self._preview_compose_metrics(model, body_t, floor))
        elif layout == "table_split":
            manifest.update(self._preview_compose_table(model, body_t, floor))
        else:
            manifest.update(self._preview_compose_narrative(model, body_t, floor))

        return manifest

    # ── composite components ─────────────────────────────────────────────

    def header(self, slide, model: dict) -> float:
        """Eyebrow + question title + answer headline. Returns bottom edge."""
        g, t = self.g, self.theme
        y = g.margin_t

        if self.cfg.show_eyebrow and model.get("eyebrow"):
            y += self.text(slide, model["eyebrow"], g.content_l, y,
                           g.content_w, 0.22 * g.s, size=g.pt(9.5), bold=True,
                           color=ensure_contrast(t.primary, t.bg, 4.5),
                           font=t.font_body, tracking_pt=1.5, uppercase=True,
                           max_lines=1, fit=False)
            y += 0.09 * g.s

        y += self.text(slide, model.get("title", ""), g.content_l, y,
                       g.content_w, 0.86 * g.s, size=g.pt(22), bold=True,
                       color=t.ink, font=t.font_head, line_spacing=1.08,
                       min_size=g.pt(15), max_lines=2)

        headline = model.get("headline", "")
        if headline:
            y += 0.10 * g.s
            y += self.text(slide, headline, g.content_l, y, g.content_w,
                           0.52 * g.s, size=g.pt(13), bold=True,
                           color=ensure_contrast(t.primary, t.bg, 5.0),
                           font=t.font_body, line_spacing=1.14,
                           min_size=g.pt(10), max_lines=2)

        chips = model.get("chips") or []
        if chips:
            y += 0.12 * g.s
            cx, chip_h = g.content_l, 0.24 * g.s
            for chip in chips:
                cw = self.chip(slide, chip, cx, y, chip_h,
                               ensure_contrast(t.primary, t.surface_alt, 4.5),
                               t.surface_alt, size=g.pt(8))
                cx += cw + 0.10 * g.s
                if cx > g.content_l + g.content_w - 1.0:
                    break
            y += chip_h
        return y + 0.20 * g.s

    def kpi_tile(self, slide, kpi: dict, l, t, w, h, index: int,
                 *, value_pt: float | None = None) -> None:
        g, th = self.g, self.theme
        accent = th.accent(index)
        self.rect(slide, l, t, w, h, th.surface, border=th.hairline,
                  border_pt=0.75)

        pad = g.pad
        inner_l, inner_w = l + pad, w - 2 * pad
        dot_d = 0.11 * g.s
        self.dot(slide, inner_l + dot_d / 2, t + pad + 0.055 * g.s, dot_d,
                 ensure_contrast(accent, th.surface, 2.2))

        label_l = inner_l + dot_d + 0.09 * g.s
        self.text(slide, kpi.get("label", ""), label_l, t + pad - 0.02 * g.s,
                  inner_w - dot_d - 0.09 * g.s, 0.20 * g.s,
                  size=g.pt(9), bold=True,
                  color=ensure_contrast(th.primary, th.surface, 5.0),
                  font=th.font_body, tracking_pt=0.6, uppercase=True,
                  max_lines=1, fit=False)

        label_h = 0.20 * g.s
        remaining = h - 2 * pad - label_h - 0.04 * g.s
        definition = kpi.get("definition") if self.cfg.show_definitions else ""

        def_h = 0.0
        if definition and remaining > 0.50 * g.s:
            def_lines = 2 if remaining > 0.95 * g.s else 1
            def_h = min(0.17 * g.s * def_lines + 0.04 * g.s, remaining * 0.42)

        value_t = t + pad + label_h + 0.04 * g.s
        value_h = max(remaining - def_h, 0.20 * g.s)
        value_color = self._value_color(kpi.get("value", ""))
        self.text(slide, kpi.get("value", "—"), inner_l, value_t, inner_w,
                  value_h, size=value_pt or g.pt(19), bold=True,
                  color=value_color, font=th.font_head, line_spacing=1.06,
                  min_size=g.pt(10.5), anchor=MSO_ANCHOR.MIDDLE, max_lines=2)

        if def_h:
            self.text(slide, definition, inner_l, value_t + value_h + 0.02 * g.s,
                      inner_w, def_h, size=g.pt(7.5), italic=True,
                      color=th.ink_muted, font=th.font_body, line_spacing=1.14,
                      min_size=g.pt(6.5), anchor=MSO_ANCHOR.TOP,
                      max_lines=2 if def_h > 0.24 * g.s else 1)

    _VERDICT_NEG = re.compile(r"^(negative|declining|down|at risk|below)$", re.I)
    _VERDICT_POS = re.compile(r"^(positive|improving|up|growing|above)$", re.I)

    def _value_color(self, value: str) -> RGBColor:
        """Colour only unambiguous verdicts and bare deltas — never long values."""
        if not self.cfg.sentiment_colors:
            return self.theme.ink
        text = _normalise_display_text(value)
        if self._VERDICT_NEG.match(text):
            return self.theme.negative
        if self._VERDICT_POS.match(text):
            return self.theme.positive
        if re.fullmatch(r"[-−]\d[\d,.]*\s*%?", text):
            return self.theme.negative
        if re.fullmatch(r"\+\d[\d,.]*\s*%?", text):
            return self.theme.positive
        return self.theme.ink

    def section_label(self, slide, text_body: str, l, t, w) -> float:
        """Standalone section heading. Kept for callers outside the cards."""
        g, th = self.g, self.theme
        h = 0.21 * g.s
        self.text(slide, text_body, l, t, w, h, size=g.pt(9.5), bold=True,
                  color=ensure_contrast(th.primary, th.bg, 4.5),
                  font=th.font_body, tracking_pt=1.4, uppercase=True,
                  max_lines=1, fit=False)
        return h + 0.10 * g.s

    # ── single-card sections ─────────────────────────────────────────────
    #
    # A section is ONE card. The heading sits inside it and the content is a
    # short bulleted list — not a row of separate tiles.

    def _finding_entry(self, kpi: dict) -> dict:
        """
        A finding reads as 'Label — value' with the value carrying the
        emphasis, and a small muted note underneath saying what the value
        measures.
        """
        th = self.theme
        label = str(kpi.get("label") or "").strip()
        value = str(kpi.get("value") or "").strip()
        definition = str(kpi.get("definition") or "").strip()

        segments = []
        if label:
            segments.append((label, {"bold": True, "color": th.ink}))
            if value:
                segments.append(("  —  ", {"color": th.ink_muted}))
        if value:
            segments.append((value, {"bold": True,
                                     "color": self._value_color(value)}))
        if not segments:
            return {}
        return {"segments": segments,
                "note": definition if self.cfg.show_definitions else ""}

    def _card(self, slide, label: str, l, t, w, h) -> tuple:
        """Draw the section card and its heading. Returns the content box."""
        g, th = self.g, self.theme
        pad = g.pad
        if slide is not None:
            self.rect(slide, l, t, w, h, th.surface,
                      border=th.hairline, border_pt=0.75)
        head_h = 0.20 * g.s
        if slide is not None:
            self.text(slide, label, l + pad, t + pad - 0.02 * g.s,
                      w - 2 * pad, head_h, size=g.pt(9.5), bold=True,
                      color=ensure_contrast(th.primary, th.surface, 4.5),
                      font=th.font_body, tracking_pt=1.4, uppercase=True,
                      max_lines=1, fit=False)
        inner_t = t + pad + head_h + 0.12 * g.s
        return (l + pad, inner_t, w - 2 * pad,
                max(t + h - pad - inner_t, 0.2 * g.s))

    def _bullet_layout(self, entries: list, text_w: float, size: float,
                       note_pt: float, spacing: float) -> dict:
        """
        Heights and gaps for one bulleted card, in inches.

        Two spacing decisions live here rather than as literals at the draw
        site, because _card_height has to predict exactly what _bulleted_card
        will draw or the panel is sized wrong:

        * a note sits a hair below its headline, not welded to it;
        * an entry that wraps gets more air after it than a one-liner. A
          fixed gap reads fine between short bullets and cramped between a
          two-line headline with a two-line note and whatever follows, which
          is what makes a long finding look like a wall of text.
        """
        g = self.g
        line_h = line_height_in(size, spacing)
        blocks = [self._entry_heights(e, text_w, size, note_pt, spacing)
                  for e in entries]

        note_gap = 0.07 * g.s
        gaps: list[float] = []
        for i in range(len(entries) - 1):
            # "Long" means the headline wrapped, or the note ran to a second
            # line — either way the block reads as a paragraph, not a line.
            long_here = (blocks[i][0] > line_h * 1.5
                         or blocks[i][1] > line_height_in(note_pt, 1.14) * 1.5)
            long_next = (blocks[i + 1][0] > line_h * 1.5
                         or blocks[i + 1][1] > line_height_in(note_pt, 1.14) * 1.5)
            gaps.append(0.15 * g.s * (1.5 if (long_here or long_next) else 1.0))

        total = (sum(m + n for m, n in blocks)
                 + note_gap * sum(1 for _, n in blocks if n)
                 + sum(gaps))
        return {"blocks": blocks, "gaps": gaps, "note_gap": note_gap,
                "total": total}

    def _bulleted_card(self, slide, label: str, entries: list, l, t, w, h,
                       *, size_pt: float, min_pt: float) -> float:
        """
        Shared renderer for both sections: one card, a heading, then a dot
        bullet per entry. Entries are lists of (text, style) segments.
        Returns the total height consumed.
        """
        g, th = self.g, self.theme
        if not entries:
            return 0.0

        inner_l, inner_t, inner_w, inner_h = self._card(slide, label, l, t, w, h)
        dot_d = 0.075 * g.s
        text_l = inner_l + dot_d + 0.13 * g.s
        text_w = max(inner_w - dot_d - 0.13 * g.s, 0.5 * g.s)
        spacing = 1.24

        # One size for every bullet in the card, chosen so the whole list fits.
        size = size_pt
        while True:
            note_pt = max(round(size * 0.78, 1), 6.5)
            layout = self._bullet_layout(entries, text_w, size, note_pt, spacing)
            if layout["total"] <= inner_h or size <= min_pt:
                break
            size = round(size - 0.5, 2)

        blocks, gaps = layout["blocks"], layout["gaps"]
        note_gap = layout["note_gap"]
        if layout["total"] > inner_h:
            # Out of room even at the floor size: give the spacing back in
            # proportion rather than dropping it, so the ordering of the
            # gaps still reflects which entries are long.
            room = max(inner_h - sum(m + n for m, n in blocks), 0.0)
            wanted = note_gap * sum(1 for _, n in blocks if n) + sum(gaps)
            shrink = (room / wanted) if wanted > 0 else 0.0
            shrink = max(min(shrink, 1.0), 0.0)
            gaps = [x * shrink for x in gaps]
            note_gap *= shrink

        y = inner_t
        for i, entry in enumerate(entries):
            main_h, note_h = blocks[i]
            self.dot(slide, inner_l + dot_d / 2,
                     y + line_height_in(size, spacing) / 2, dot_d,
                     ensure_contrast(th.accent(i), th.surface, 2.4))
            self.text_runs(slide, entry["segments"], text_l, y, text_w,
                           main_h + 0.05 * g.s, size=size,
                           line_spacing=spacing, min_size=min_pt, fit=False)
            if note_h:
                self.text(slide, entry["note"], text_l,
                          y + main_h + note_gap,
                          text_w, note_h + 0.04 * g.s, size=note_pt,
                          italic=True, color=th.ink_muted, font=th.font_body,
                          line_spacing=1.14, max_lines=2, fit=False)
                y += note_gap
            y += main_h + note_h + (gaps[i] if i < len(gaps) else 0.0)
        return y - t

    def _entry_heights(self, entry: dict, text_w: float, size: float,
                       note_pt: float, spacing: float) -> tuple:
        """(main height, note height) for one bullet, in inches."""
        main = self.text_runs(None, entry.get("segments") or [], 0, 0, text_w,
                              99, size=size, line_spacing=spacing, fit=False)
        note_text = str(entry.get("note") or "").strip()
        if not note_text:
            return main, 0.0
        lines = wrap_lines(note_text, text_w, note_pt, font=self.theme.font_body,
                           max_lines=2)
        return main, text_block_height_in(lines, note_pt, 1.14) + 0.03 * self.g.s

    def _card_height(self, entries: list, w: float, size_pt: float) -> float:
        """Natural height of a bulleted card, padding and heading included."""
        g = self.g
        if not entries:
            return 0.0
        pad = g.pad
        dot_d = 0.075 * g.s
        text_w = max(w - 2 * pad - dot_d - 0.13 * g.s, 0.5 * g.s)
        note_pt = max(round(size_pt * 0.78, 1), 6.5)
        # Same pass the renderer uses, so a card is never sized for tighter
        # spacing than it will actually be drawn with.
        body = self._bullet_layout(entries, text_w, size_pt, note_pt,
                                   1.24)["total"]
        return pad * 2 + 0.20 * g.s + 0.12 * g.s + body

    def findings_panel(self, slide, kpis: list[dict], l, t, w, h,
                       orientation: str = "auto") -> None:
        """'Key Findings' — a single card holding one to three bullets."""
        entries = [self._finding_entry(k) for k in kpis or []]
        entries = [e for e in entries if e]
        self._bulleted_card(slide, self.cfg.findings_label, entries, l, t, w, h,
                            size_pt=self.g.pt(11), min_pt=self.g.pt(8.5))

    def takeaways_panel(self, slide, points: list[str], l, t, w, h) -> None:
        """'What this means' — a single card holding one to three bullets."""
        th = self.theme
        entries = [{"segments": [(str(p).strip(), {"color": th.ink_soft})],
                    "note": ""}
                   for p in points or [] if str(p).strip()]
        self._bulleted_card(slide, self.cfg.takeaways_label, entries, l, t, w, h,
                            size_pt=self.g.pt(11), min_pt=self.g.pt(8.5))

    def takeaways_height(self, points: list[str], w: float) -> float:
        th = self.theme
        entries = [{"segments": [(str(p).strip(), {"color": th.ink_soft})],
                    "note": ""}
                   for p in points or [] if str(p).strip()]
        return self._card_height(entries, w, self.g.pt(11))

    def findings_height(self, kpis: list[dict], w: float) -> float:
        entries = [self._finding_entry(k) for k in kpis or []]
        return self._card_height([e for e in entries if e], w, self.g.pt(11))

    def kpi_rail(self, slide, kpis: list[dict], l, t, w, h) -> None:
        if not kpis:
            return
        gap = 0.16 * self.g.s
        n = len(kpis)
        tile_h = (h - gap * (n - 1)) / n
        for i, kpi in enumerate(kpis):
            self.kpi_tile(slide, kpi, l, t + i * (tile_h + gap), w, tile_h, i)

    def kpi_strip(self, slide, kpis: list[dict], l, t, w, h,
                  *, value_pt: float | None = None) -> None:
        if not kpis:
            return
        gap = 0.20 * self.g.s
        n = len(kpis)
        tile_w = (w - gap * (n - 1)) / n
        for i, kpi in enumerate(kpis):
            self.kpi_tile(slide, kpi, l + i * (tile_w + gap), t, tile_w, h, i,
                          value_pt=value_pt)

    def bullet_row(self, slide, bullets: list[str], l, t, w, h) -> None:
        """Three takeaways as columns, each led by a numbered chip."""
        if not bullets:
            return
        g, th = self.g, self.theme
        gap = 0.30 * g.s
        n = len(bullets)
        col_w = (w - gap * (n - 1)) / n
        num_d = 0.26 * g.s
        for i, bullet in enumerate(bullets):
            x = l + i * (col_w + gap)
            chip = ensure_contrast(th.accent(i), th.bg, 2.6)
            self.dot(slide, x + num_d / 2, t + num_d / 2, num_d, chip)
            self.text(slide, str(i + 1), x, t, num_d, num_d, size=g.pt(9),
                      bold=True,
                      color=(RGBColor(0xFF, 0xFF, 0xFF)
                             if luminance(chip) < 0.45 else th.ink),
                      font=th.font_body,
                      align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, fit=False)
            self.text(slide, bullet, x + num_d + 0.12 * g.s, t - 0.03 * g.s,
                      col_w - num_d - 0.12 * g.s, h, size=g.pt(10.5),
                      color=th.ink_soft, font=th.font_body, line_spacing=1.22,
                      min_size=g.pt(8), max_lines=5)

    def bullet_list_height(self, bullets: list[str], w: float,
                           size: float | None = None) -> float:
        """Natural stacked height, used to centre short lists in tall panels."""
        if not bullets:
            return 0.0
        g = self.g
        num_d = 0.28 * g.s
        text_w = w - num_d - 0.16 * g.s
        size = size or g.pt(12)
        heights = [max(self.measure(b, text_w, size=size, line_spacing=1.26),
                       num_d) for b in bullets]
        return sum(heights) + 0.22 * g.s * (len(bullets) - 1)

    def insight_height(self, text_body: str, w: float) -> float:
        if not text_body:
            return 0.0
        g = self.g
        return (self.measure(text_body, w - 2 * g.pad, size=g.pt(11),
                             line_spacing=1.24)
                + 2 * g.pad + 0.20 * g.s + 0.05 * g.s)

    def bullet_list(self, slide, bullets: list[str], l, t, w, h) -> None:
        """
        Stacked takeaways sized to their content, so short lists sit tight
        instead of drifting apart across the panel.
        """
        if not bullets:
            return
        g, th = self.g, self.theme
        num_d = 0.28 * g.s
        text_l = l + num_d + 0.16 * g.s
        text_w = w - num_d - 0.16 * g.s
        spacing = 1.26

        size = g.pt(12)
        min_size = g.pt(9)
        while size >= min_size:
            heights = [max(self.measure(b, text_w, size=size,
                                        line_spacing=spacing), num_d)
                       for b in bullets]
            gap = min(0.30 * g.s, max(0.16 * g.s,
                                      (h - sum(heights)) / max(len(bullets), 1)))
            if sum(heights) + gap * (len(bullets) - 1) <= h:
                break
            size = round(size - 0.5, 2)
        else:
            heights = [max(self.measure(b, text_w, size=size,
                                        line_spacing=spacing), num_d)
                       for b in bullets]
            gap = 0.14 * g.s

        y = t
        for i, bullet in enumerate(bullets):
            chip = ensure_contrast(th.accent(i), th.bg, 2.6)
            self.dot(slide, l + num_d / 2, y + num_d / 2, num_d, chip)
            self.text(slide, str(i + 1), l, y, num_d, num_d, size=g.pt(9.5),
                      bold=True,
                      color=(RGBColor(0xFF, 0xFF, 0xFF)
                             if luminance(chip) < 0.45 else th.ink),
                      font=th.font_body,
                      align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, fit=False)
            self.text(slide, bullet, text_l, y - 0.015 * g.s, text_w,
                      heights[i] + 0.06 * g.s, size=size, color=th.ink_soft,
                      font=th.font_body, line_spacing=spacing,
                      min_size=min_size, anchor=MSO_ANCHOR.TOP, fit=False)
            y += heights[i] + gap

    def insight_band(self, slide, text_body: str, l, t, w, h,
                     label: str = "What this means") -> None:
        if not text_body:
            return
        g, th = self.g, self.theme
        pad = g.pad
        label_h = 0.20 * g.s

        # Size the card to its content rather than leaving a hollow block.
        needed = self.measure(text_body, w - 2 * pad, size=g.pt(11),
                              line_spacing=1.24) + 2 * pad + label_h + 0.05 * g.s
        h = max(min(h, needed), min(h, 0.86 * g.s))

        self.rect(slide, l, t, w, h, th.surface_alt, border=None)
        self.text(slide, label, l + pad, t + pad - 0.02 * g.s, w - 2 * pad,
                  label_h, size=g.pt(8.5), bold=True,
                  color=ensure_contrast(th.primary, th.surface_alt, 5.0),
                  font=th.font_body, tracking_pt=1.2, uppercase=True,
                  max_lines=1, fit=False)
        self.text(slide, text_body, l + pad, t + pad + label_h + 0.05 * g.s,
                  w - 2 * pad, h - 2 * pad - label_h - 0.05 * g.s,
                  size=g.pt(11), color=th.ink_soft, font=th.font_body,
                  line_spacing=1.24, min_size=g.pt(8.5), max_lines=4)

    def footer(self, slide, model: dict) -> None:
        g, th = self.g, self.theme
        cfg = self.cfg
        left_bits = []
        if cfg.show_footnote and model.get("footnote"):
            left_bits.append(model["footnote"])
        if cfg.show_basis:
            basis = model.get("basis")
            if basis and basis != "Not stated":
                left_bits.append(f"Demand basis: {basis}")
            elif basis == "Not stated":
                left_bits.append("Demand basis not stated in source")
        if cfg.confidential_note:
            left_bits.append(cfg.confidential_note)
        left = "   ·   ".join(left_bits)

        logo_w = self.footer_logo(slide)

        right_w = 0.5 * g.s if cfg.slide_numbers else 0.0
        if left:
            self.text(slide, left, g.content_l, g.footer_y,
                      g.content_w - logo_w - right_w - 0.12 * g.s, g.footer_h,
                      size=g.pt(7.5), color=th.ink_muted, font=th.font_body,
                      anchor=MSO_ANCHOR.MIDDLE, max_lines=1)
        if cfg.slide_numbers:
            self.text(slide, str(self._n),
                      g.content_l + g.content_w - logo_w - right_w,
                      g.footer_y, right_w, g.footer_h, size=g.pt(8),
                      color=th.ink_muted, font=th.font_body,
                      align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, fit=False)

    def notes(self, slide, text_body: str) -> None:
        if not text_body:
            return
        try:
            slide.notes_slide.notes_text_frame.text = text_body
        except Exception:
            pass

    # ── data table ───────────────────────────────────────────────────────

    _PLAIN_TABLE_STYLE = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"

    def _plain_table(self, table) -> None:
        """Strip PowerPoint's default blue banded style."""
        try:
            table.first_row = False
            table.horz_banding = False
            table.vert_banding = False
            tbl_pr = table._tbl.find(qn_a("tblPr"))
            if tbl_pr is not None:
                for child in list(tbl_pr):
                    if child.tag == qn_a("tableStyleId"):
                        child.text = self._PLAIN_TABLE_STYLE
                        break
                else:
                    node = tbl_pr.makeelement(qn_a("tableStyleId"), {})
                    node.text = self._PLAIN_TABLE_STYLE
                    tbl_pr.append(node)
        except Exception:
            pass

    def _table_rows(self, df, profile: dict) -> tuple:
        """(row count including header, truncated?) for the rendered table."""
        if df is None or pd is None or len(df) == 0:
            return 0, False
        max_rows = min(self.cfg.max_table_rows, len(df))
        return len(df.tail(max_rows)) + 1, len(df) > max_rows

    def _table_height(self, df, profile: dict, h: float) -> float:
        """Height data_table will actually occupy inside a *h*-tall box."""
        g = self.g
        n_rows, truncated = self._table_rows(df, profile)
        if not n_rows:
            return 0.0
        note_h = 0.20 * g.s if truncated else 0.0
        usable = max(h - note_h, 0.6 * g.s)
        row_h = min(0.38 * g.s, max(0.24 * g.s, usable / max(n_rows, 1)))
        return row_h * n_rows + note_h

    def data_table(self, slide, df, profile: dict, l, t, w, h) -> None:
        """Compact, brand-styled table. Columns are chosen to fit, never squeezed."""
        if df is None or pd is None or len(df) == 0:
            return
        g, th = self.g, self.theme

        drop = set(profile.get("constants") or {})
        drop.update(profile.get("drop_cols") or [])
        cols = [c for c in df.columns if c not in drop] or list(df.columns)

        max_rows = min(self.cfg.max_table_rows, len(df))
        truncated = len(df) > max_rows
        body = df.tail(max_rows) if profile.get("shape") == "timeseries" else df.head(max_rows)

        head_pt, cell_pt = g.pt(8.5), g.pt(9)
        cell_pad = 0.09 * g.s

        def col_width(col) -> float:
            header = humanize_column(col)
            widest = text_width_in(header, head_pt, bold=True, font=th.font_body)
            for v in body[col].head(max_rows):
                widest = max(widest, text_width_in(format_cell(col, v), cell_pt,
                                                   font=th.font_body))
            return widest + cell_pad * 2 + 0.06 * g.s

        widths = {c: col_width(c) for c in cols}
        while len(cols) > 2 and sum(widths[c] for c in cols) > w:
            victim = min(cols[1:], key=lambda c: (0 if _is_numeric(df[c]) else 1,
                                                  -widths[c]))
            cols.remove(victim)
        total = sum(widths[c] for c in cols)
        if total <= 0:
            return
        if total < w:
            surplus = w - total
            first_extra = min(surplus * 0.5, widths[cols[0]] * 0.6)
            widths[cols[0]] += first_extra
            share = (surplus - first_extra) / max(len(cols) - 1, 1)
            for c in cols[1:]:
                widths[c] += share
        else:
            scale = w / total
            widths = {c: widths[c] * scale for c in cols}

        note_h = 0.20 * g.s if truncated else 0.0
        h = max(h - note_h, 0.6 * g.s)
        n_rows = len(body) + 1
        row_h = min(0.38 * g.s, max(0.24 * g.s, h / max(n_rows, 1)))
        table_h = row_h * n_rows

        frame = slide.shapes.add_table(n_rows, len(cols), Inches(l),
                                       Inches(t + max((h - table_h) / 2, 0)),
                                       Inches(w), Inches(table_h))
        table = frame.table
        self._plain_table(table)

        for i, col in enumerate(cols):
            table.columns[i].width = Emu(int(Inches(widths[col])))
        for r in range(n_rows):
            table.rows[r].height = Emu(int(Inches(row_h)))

        def style_cell(cell, text_value, *, bold, color, fill, align):
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
            cell.margin_left = cell.margin_right = Emu(int(Inches(cell_pad)))
            cell.margin_top = cell.margin_bottom = Emu(int(Inches(0.02 * g.s)))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = False
            para = tf.paragraphs[0]
            para.alignment = align
            run = para.add_run()
            run.text = text_value
            run.font.size = Pt(head_pt if bold else cell_pt)
            run.font.bold = bold
            run.font.name = th.font_body
            run.font.color.rgb = color

        for i, col in enumerate(cols):
            align = PP_ALIGN.RIGHT if _is_numeric(df[col]) else PP_ALIGN.LEFT
            style_cell(table.cell(0, i), humanize_column(col), bold=True,
                       color=ensure_contrast(th.primary, th.surface_alt, 5.0),
                       fill=th.surface_alt, align=align)

        for r in range(len(body)):
            tint = th.bg if r % 2 == 0 else th.surface
            for i, col in enumerate(cols):
                align = PP_ALIGN.RIGHT if _is_numeric(df[col]) else PP_ALIGN.LEFT
                style_cell(table.cell(r + 1, i),
                           format_cell(col, body[col].iloc[r]), bold=False,
                           color=th.ink_soft, fill=tint, align=align)

        if truncated:
            shown = "most recent" if profile.get("shape") == "timeseries" else "first"
            self.text(slide, f"Showing the {shown} {max_rows} of {len(df)} rows",
                      l, t + max((h - table_h) / 2, 0) + table_h + 0.05 * g.s,
                      w, note_h, size=g.pt(7.5), italic=True,
                      color=th.ink_muted, font=th.font_body, max_lines=1)

    # ── slide compositions ───────────────────────────────────────────────

    def add_analysis_slide(self, model: dict):
        """Route a slide model to the right layout. Never overflows."""
        g = self.g
        slide = self._blank()
        self._fill_bg(slide, self.theme.bg)

        body_t = self.header(slide, model)
        floor = g.footer_y - 0.22 * g.s
        layout = model.get("layout", "narrative")

        if layout == "chart_split":
            self._compose_chart(slide, model, body_t, floor)
        elif layout == "metric_grid":
            self._compose_metrics(slide, model, body_t, floor)
        elif layout == "table_split":
            self._compose_table(slide, model, body_t, floor)
        else:
            self._compose_narrative(slide, model, body_t, floor)

        self.footer(slide, model)
        self.notes(slide, model.get("notes", ""))
        return slide

    def _compose_chart(self, slide, model, body_t, floor) -> None:
        """Chart + 'Key Findings' rail, with 'What this means' underneath."""
        g, th = self.g, self.theme
        takeaways = model.get("bullets") or []
        kpis = model.get("kpis") or []

        band_h = (min(self.takeaways_height(takeaways, g.content_w),
                      2.05 * g.s) if takeaways else 0.0)
        band_t = floor - band_h
        body_b = (band_t - 0.24 * g.s) if takeaways else floor
        body_h = max(body_b - body_t, 1.2 * g.s)

        rail_w = g.content_w * 0.31 if kpis else 0.0
        chart_w = g.content_w - rail_w - (g.gap if kpis else 0.0)

        self.rect(slide, g.content_l, body_t, chart_w, body_h,
                  th.bg, border=th.hairline, border_pt=0.75)
        card_pad = 0.11 * g.s
        path = model.get("chart_path")
        if path and os.path.exists(path):
            self.picture_fit(slide, path, g.content_l + card_pad, body_t + card_pad,
                             chart_w - 2 * card_pad, body_h - 2 * card_pad)

        if kpis:
            # Card height tracks the chart card so the two panels line up.
            self.findings_panel(slide, kpis, g.content_l + chart_w + g.gap,
                                body_t, rail_w, body_h)
        if takeaways:
            self.takeaways_panel(slide, takeaways, g.content_l, band_t,
                                 g.content_w, band_h)

    def _compose_metrics(self, slide, model, body_t, floor) -> None:
        """'Key Findings' across the width, 'What this means' beneath it."""
        g = self.g
        kpis = model.get("kpis") or []
        takeaways = model.get("bullets") or []

        avail = max(floor - body_t, 1.0 * g.s)
        findings_h = (min(self.findings_height(kpis, g.content_w),
                          avail * 0.52) if kpis else 0.0)
        gap = g.gap_lg if (kpis and takeaways) else 0.0
        takeaways_h = (min(self.takeaways_height(takeaways, g.content_w),
                           avail - findings_h - gap) if takeaways else 0.0)

        natural = findings_h + gap + takeaways_h
        offset = max(min((avail - natural) / 2, avail * 0.34), 0)
        y = body_t + offset

        if kpis:
            self.findings_panel(slide, kpis, g.content_l, y, g.content_w,
                                findings_h)
            y += findings_h + gap
        if takeaways:
            self.takeaways_panel(slide, takeaways, g.content_l, y,
                                 g.content_w, takeaways_h)

    def _compose_table(self, slide, model, body_t, floor) -> None:
        """Table + 'Key Findings' rail, with 'What this means' underneath."""
        g = self.g
        kpis = model.get("kpis") or []
        takeaways = model.get("bullets") or []

        band_h = (min(self.takeaways_height(takeaways, g.content_w),
                      2.05 * g.s) if takeaways else 0.0)
        band_t = floor - band_h
        body_b = (band_t - 0.24 * g.s) if takeaways else floor
        body_h = max(body_b - body_t, 1.0 * g.s)

        rail_w = g.content_w * 0.31 if kpis else 0.0
        table_w = g.content_w - rail_w - (g.gap if kpis else 0.0)

        self.data_table(slide, model.get("df"), model.get("profile") or {},
                        g.content_l, body_t, table_w, body_h)
        if kpis:
            # Match the rendered table block rather than the panel: a short
            # table would otherwise sit beside a full-height card.
            table_h = min(self._table_height(model.get("df"),
                                             model.get("profile") or {}, body_h),
                          body_h)
            self.findings_panel(slide, kpis, g.content_l + table_w + g.gap,
                                body_t, rail_w, table_h)
        if takeaways:
            self.takeaways_panel(slide, takeaways, g.content_l, band_t,
                                 g.content_w, band_h)

    def _compose_narrative(self, slide, model, body_t, floor) -> None:
        """No data returned — only 'What this means' has anything to say."""
        g = self.g
        takeaways = model.get("bullets") or []
        body_h = max(floor - body_t, 1.0 * g.s)
        if not takeaways:
            return
        natural = self.takeaways_height(takeaways, g.content_w)
        offset = max(min((body_h - natural) / 2, body_h * 0.34), 0)
        self.takeaways_panel(slide, takeaways, g.content_l, body_t + offset,
                             g.content_w, body_h - offset)

    # ── deck furniture ───────────────────────────────────────────────────

    def add_cover(self, questions: Sequence[str] = ()) -> None:
        g, th, cfg = self.g, self.theme, self.cfg
        slide = self._blank()
        dark = cfg.dark_cover
        bg = th.bg_dark if dark else th.bg
        self._fill_bg(slide, bg)

        ink = th.ink_inverse if dark else th.ink
        soft = mix(ink, bg, 0.30)
        accent = ensure_contrast(th.primary, bg, 3.4)

        top = g.margin_t + 0.10 * g.s
        if cfg.logo_path and os.path.exists(cfg.logo_path):
            plate_w, plate_h = 1.55 * g.s, 0.62 * g.s
            if dark:
                self.rect(slide, g.content_l, top, plate_w, plate_h,
                          RGBColor(0xFF, 0xFF, 0xFF), border=None, radius=0.10)
            pad = 0.10 * g.s
            self.logo(slide, g.content_l + pad, top + pad,
                      plate_w - 2 * pad, plate_h - 2 * pad)

        block_t = g.h * 0.40
        y = block_t
        y += self.text(slide, cfg.brand_name or "Commercial Analytics",
                       g.content_l, y, g.content_w, 0.24 * g.s, size=g.pt(10),
                       bold=True, color=accent, font=th.font_body,
                       tracking_pt=2.0, uppercase=True, max_lines=1, fit=False)
        y += 0.14 * g.s
        y += self.text(slide, cfg.deck_title, g.content_l, y,
                       g.content_w * 0.86, 1.5 * g.s, size=g.pt(40), bold=True,
                       color=ink, font=th.font_head, line_spacing=1.06,
                       min_size=g.pt(24), max_lines=3)
        if cfg.deck_subtitle:
            y += 0.30 * g.s
            self.text(slide, cfg.deck_subtitle, g.content_l, y,
                      g.content_w * 0.70, 0.7 * g.s, size=g.pt(15),
                      color=soft, font=th.font_body, line_spacing=1.20,
                      min_size=g.pt(11), max_lines=2)

        meta = [date.today().strftime("%d %B %Y")]
        if questions:
            meta.append(f"{len(questions)} question{'s' if len(questions) != 1 else ''}")
        if cfg.confidential_note:
            meta.append(cfg.confidential_note)
        # No footer mark here: the cover already carries the logo top-left.
        self.text(slide, "   ·   ".join(meta), g.content_l, g.footer_y,
                  g.content_w, g.footer_h, size=g.pt(8.5),
                  color=mix(ink, bg, 0.45), font=th.font_body,
                  anchor=MSO_ANCHOR.MIDDLE, max_lines=1)

    def add_agenda(self, items: Sequence[str]) -> None:
        if not items:
            return
        g, th = self.g, self.theme
        slide = self._blank()
        self._fill_bg(slide, th.bg)

        y = g.margin_t
        y += self.text(slide, "IN THIS REVIEW", g.content_l, y, g.content_w,
                       0.22 * g.s, size=g.pt(9.5), bold=True,
                       color=ensure_contrast(th.primary, th.bg, 4.5),
                       font=th.font_body, tracking_pt=1.5, max_lines=1, fit=False)
        y += 0.09 * g.s
        y += self.text(slide, "Questions answered in this deck", g.content_l, y,
                       g.content_w, 0.62 * g.s, size=g.pt(22), bold=True,
                       color=th.ink, font=th.font_head, max_lines=1)
        y += 0.34 * g.s

        items = list(items)[:8]
        avail_h = g.footer_y - 0.20 * g.s - y
        two_col = len(items) > 4
        col_w = (g.content_w - g.gap_lg) / 2 if two_col else g.content_w
        per_col = math.ceil(len(items) / 2) if two_col else len(items)
        row_h = min(1.05 * g.s, max(0.62 * g.s, avail_h / max(per_col, 1)))
        y += max((avail_h - row_h * per_col) / 2, 0)
        num_d = 0.30 * g.s

        for i, item in enumerate(items):
            col, row = (i // per_col, i % per_col) if two_col else (0, i)
            x = g.content_l + col * (col_w + g.gap_lg)
            ry = y + row * row_h
            chip = ensure_contrast(th.accent(i), th.bg, 2.6)
            self.dot(slide, x + num_d / 2, ry + num_d / 2, num_d, chip)
            self.text(slide, f"{i + 1}", x, ry, num_d, num_d, size=g.pt(10),
                      bold=True,
                      color=(RGBColor(0xFF, 0xFF, 0xFF)
                             if luminance(chip) < 0.45 else th.ink),
                      font=th.font_body,
                      align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, fit=False)
            self.text(slide, _titlecase_question(item), x + num_d + 0.16 * g.s,
                      ry - 0.02 * g.s, col_w - num_d - 0.16 * g.s,
                      row_h - 0.10 * g.s, size=g.pt(13.5), color=th.ink_soft,
                      font=th.font_body, line_spacing=1.20, min_size=g.pt(10),
                      max_lines=2)
        self.footer(slide, {})

    def add_closing(self, headlines: Sequence[str]) -> None:
        g, th = self.g, self.theme
        slide = self._blank()
        dark = self.cfg.dark_cover
        bg = th.bg_dark if dark else th.surface
        self._fill_bg(slide, bg)
        ink = th.ink_inverse if dark else th.ink
        soft = mix(ink, bg, 0.28)
        accent = ensure_contrast(th.primary, bg, 3.4)

        y = g.margin_t + 0.06 * g.s
        y += self.text(slide, "WHERE TO FOCUS NEXT", g.content_l, y, g.content_w,
                       0.22 * g.s, size=g.pt(9.5), bold=True, color=accent,
                       font=th.font_body, tracking_pt=1.6, max_lines=1, fit=False)
        y += 0.10 * g.s
        y += self.text(slide, "The short version", g.content_l, y, g.content_w,
                       0.66 * g.s, size=g.pt(26), bold=True, color=ink,
                       font=th.font_head, max_lines=1)
        y += 0.36 * g.s

        items = [h for h in headlines if h][:5]
        if items:
            avail = g.footer_y - 0.24 * g.s - y
            row_h = min(0.86 * g.s, max(0.58 * g.s, avail / len(items)))
            y += max((avail - row_h * len(items)) / 2, 0)
            num_d = 0.28 * g.s
            for i, item in enumerate(items):
                ry = y + i * row_h
                chip = ensure_contrast(th.accent(i), bg, 2.8)
                self.dot(slide, g.content_l + num_d / 2, ry + num_d / 2, num_d,
                         chip)
                self.text(slide, str(i + 1), g.content_l, ry, num_d, num_d,
                          size=g.pt(9), bold=True,
                          color=(RGBColor(0xFF, 0xFF, 0xFF)
                                 if luminance(chip) < 0.45 else th.bg_dark),
                          font=th.font_body, align=PP_ALIGN.CENTER,
                          anchor=MSO_ANCHOR.MIDDLE, fit=False)
                self.text(slide, item, g.content_l + num_d + 0.16 * g.s,
                          ry - 0.02 * g.s, g.content_w - num_d - 0.16 * g.s,
                          row_h - 0.12 * g.s, size=g.pt(13), color=soft,
                          font=th.font_body, line_spacing=1.22,
                          min_size=g.pt(10), max_lines=2)

        logo_w = 0.0
        if self.cfg.footer_logo and self.cfg.logo_path \
                and os.path.exists(self.cfg.logo_path):
            cfg_logo_w = self.cfg.footer_logo_w
            if dark:
                # A dark-ink mark disappears on the dark closing slide, so it
                # sits on the same white plate the cover uses.
                plate_w = (cfg_logo_w + 0.16) * g.s
                plate_h = (self.cfg.footer_logo_h + 0.10) * g.s
                self.rect(slide,
                          g.content_l + g.content_w - plate_w,
                          g.footer_y + (g.footer_h - plate_h) / 2,
                          plate_w, plate_h, RGBColor(0xFF, 0xFF, 0xFF),
                          border=None, radius=0.08)
            logo_w = self.footer_logo(slide)

        self.text(slide, self.cfg.confidential_note or "", g.content_l,
                  g.footer_y, g.content_w - logo_w, g.footer_h, size=g.pt(8),
                  color=mix(ink, bg, 0.50), font=th.font_body,
                  anchor=MSO_ANCHOR.MIDDLE, max_lines=1)

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self.prs.save(path)
        log.info("saved %s (%d slides)", path, len(self.prs.slides._sldIdLst))
        return path

    def measure_header(self, model: dict) -> float:
        """Header height without drawing — lets the chart be sized correctly."""
        return self.header(None, model)

    def chart_panel(self, model: dict) -> tuple[float, float]:
        """Exact inner dimensions of the chart card for this model."""
        g = self.g
        header_bottom = self.measure_header(model)
        floor = g.footer_y - 0.22 * g.s
        row_h = (min(self.takeaways_height(model.get("bullets") or [],
                                           g.content_w), 2.05 * g.s)
                 if model.get("bullets") else 0.0)
        body_b = (floor - row_h - 0.24 * g.s) if row_h else floor
        body_h = max(body_b - header_bottom, 1.2 * g.s)
        rail_w = g.content_w * 0.31 if model.get("kpis") else 0.0
        chart_w = g.content_w - rail_w - (g.gap if rail_w else 0.0)
        pad = 0.11 * g.s
        return chart_w - 2 * pad, body_h - 2 * pad


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE MODEL
# ══════════════════════════════════════════════════════════════════════════════


def choose_layout(model: dict) -> str:
    if model.get("chart_path"):
        return "chart_split"
    profile = model.get("profile") or {}
    rows = profile.get("rows", 0)
    if rows == 1:
        return "metric_grid"
    if rows > 1:
        return "table_split"
    return "narrative"


def _speaker_notes(block: dict, content: dict, layout: str,
                   cfg: DeckConfig | None = None) -> str:
    """
    Notes are off by default: the deck goes to clients, and the pane used to
    carry the executed SQL, the demand basis and suggested follow-ups. Set
    DeckConfig.speaker_notes=True to restore that internal audit trail.
    """
    if cfg is not None and not cfg.speaker_notes:
        return ""
    parts = []
    if layout == "chart_split" and content.get("insight"):
        parts.append("WHAT THIS MEANS\n" + content["insight"])
    if layout in ("table_split",) and content.get("bullets"):
        parts.append("KEY POINTS\n"
                     + "\n".join(f"- {b}" for b in content["bullets"]))
    if content.get("basis"):
        parts.append(f"DEMAND BASIS: {content['basis']}")
    if block.get("followups"):
        parts.append("SUGGESTED FOLLOW-UPS\n"
                     + "\n".join(f"- {q}" for q in block["followups"]))
    if block.get("sql"):
        sql = block["sql"]
        parts.append("SQL EXECUTED\n" + (sql if len(sql) < 3500
                                         else sql[:3500] + "\n… truncated"))
    return "\n\n".join(parts)


def build_slide_model(
    block: dict,
    cfg: DeckConfig,
    deck: Deck,
    chart_path_override: str | None = None,
    chart_render_spec_mode: bool = False,
) -> dict:
    """Content + data + chart + layout decision for a single analysis block."""
    df = to_dataframe(block.get("data"))
    profile = profile_dataframe(df, getattr(cfg, "hooks", None))
    content = generate_slide_content(block, cfg, df, profile)

    model: dict[str, Any] = dict(content)
    model.update(question=block.get("question", ""), df=df, profile=profile,
                 chips=profile.get("chips") or [], chart_path=None)

    if chart_path_override:
        model["chart_path"] = chart_path_override
        log.info("[deck-debug] chart override applied before layout path=%s",
                 chart_path_override)
    elif block.get("viz_code") or block.get("viz_figure"):
        try:
            panel_w, panel_h = deck.chart_panel(model)
            if chart_render_spec_mode:
                spec = build_chart_render_spec(block, deck.theme, panel_w,
                                               panel_h, cfg)
                if spec:
                    model["chart_render_spec"] = spec
                    model["chart_expected"] = True
                    log.info(
                        "[deck-debug] chart render spec prepared width=%s height=%s scale=%s",
                        spec.get("width"),
                        spec.get("height"),
                        spec.get("scale"),
                    )
            else:
                model["chart_path"] = render_chart(block, deck.theme, panel_w,
                                                   panel_h, cfg)
        except Exception as exc:
            log.warning("chart pipeline failed: %s", exc)

    model["layout"] = "chart_split" if model.get("chart_expected") else choose_layout(model)
    model["notes"] = _speaker_notes(block, content, model["layout"], cfg)
    log.info(
        "[deck-debug] slide model question=%s layout=%s chart=%s title=%s headline=%s",
        _debug_excerpt(block.get("question"), 90),
        model.get("layout"),
        bool(model.get("chart_path")),
        _debug_excerpt(model.get("title"), 100),
        _debug_excerpt(model.get("headline"), 180),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════


def build_deck(messages: Iterable[Any], cfg: DeckConfig | None = None) -> str:
    """
    End-to-end: conversation -> .pptx. Returns the output path.

    Safe by construction: a failure on one block costs that slide, not the deck.
    """
    cfg = cfg or DeckConfig()
    log.info("[deck-debug] build_deck start output=%s template=%s logo=%s hooks=%s",
             cfg.output_path, cfg.template_path, cfg.logo_path,
             getattr(cfg.hooks, "describe", lambda: "none")())
    blocks = parse_conversation(messages, getattr(cfg, "hooks", None))
    if not blocks:
        raise ValueError("No question/answer blocks found in the messages.")

    deck = Deck(cfg)
    models: list[dict] = []
    for i, block in enumerate(blocks, 1):
        try:
            models.append(build_slide_model(block, cfg, deck))
        except Exception as exc:
            log.error("block %d failed (%s) — skipping", i, exc)

    if not models:
        raise RuntimeError("Every block failed to build.")

    if cfg.cover_slide:
        deck.add_cover([m["question"] for m in models])
    if cfg.agenda_slide and len(models) > 1:
        deck.add_agenda([m["question"] for m in models])

    for model in models:
        try:
            deck.add_analysis_slide(model)
        except Exception as exc:
            log.error("slide '%s' failed to render: %s",
                      model.get("title", "?"), exc)

    if cfg.closing_slide and len(models) > 1:
        deck.add_closing([m.get("headline") or m.get("title") for m in models])

    path = deck.save(cfg.output_path)
    _cleanup_charts(models, cfg)
    log.info("[deck-debug] build_deck done slides=%d output=%s", len(models), path)
    return path


def _cleanup_charts(models: Sequence[dict], cfg: DeckConfig) -> None:
    for model in models:
        path = model.get("chart_path")
        if path and os.path.exists(path) and os.path.dirname(path) == cfg.workdir:
            try:
                os.remove(path)
            except OSError:
                pass
    try:
        if os.path.isdir(cfg.workdir) and not os.listdir(cfg.workdir):
            os.rmdir(cfg.workdir)
    except OSError:
        pass


# ── host-app helpers ─────────────────────────────────────────────────────────

def slice_turns(messages: Sequence[Any], n_turns: int = 1) -> list[Any]:
    """
    The last *n_turns* complete question/answer turns.

    Slicing a fixed message count (messages[-4:]) breaks whenever a turn has
    no visualization message: the window starts mid-turn on an AIMessage and
    every block before the next question is silently dropped. Cutting on
    HumanMessage boundaries keeps turns whole.
    """
    if not messages:
        return []
    starts = [i for i, m in enumerate(messages) if _msg_role(m) == "human"]
    if not starts:
        return list(messages)
    return list(messages[starts[max(len(starts) - max(n_turns, 1), 0)]:])


def build_deck_bytes(messages: Iterable[Any],
                     cfg: DeckConfig | None = None) -> tuple[bytes, str]:
    """
    Build and return (bytes, filename) without depending on a writable working
    directory — the shape st.download_button needs.
    """
    cfg = cfg or DeckConfig()
    log.info("[deck-debug] build_deck_bytes start")
    path = build_deck(messages, cfg)
    with open(path, "rb") as handle:
        payload = handle.read()
    name = os.path.basename(path)
    try:
        os.remove(path)
    except OSError:
        pass
    log.info("[deck-debug] build_deck_bytes done bytes=%d name=%s", len(payload), name)
    return payload, name


def build_slide_data(messages: Iterable[Any],
                     chart_path_overrides: list[str | None] | None = None,
                     cancel_check: Any = None,
                     chart_render_spec_mode: bool = False) -> list[dict]:
    """
    Compatibility wrapper for the current Aadibio preview API.

    Returns the per-slide model objects the preview layer already knows how to
    serialize, plus a backend-authored preview manifest built from the same
    layout engine used for the real PPT.
    """
    cfg = DeckConfig(
        template_path=_resolve_local_asset_path("aadibio_ppt.pptx") or "aadibio_ppt.pptx",
        logo_path=_resolve_local_asset_path("aadibio_logo.png") or "aadibio_logo.png",
    )
    log.info("[deck-debug] build_slide_data config template=%s logo=%s hooks=%s",
             cfg.template_path, cfg.logo_path,
             getattr(cfg.hooks, "describe", lambda: "none")())
    blocks = parse_conversation(messages, getattr(cfg, "hooks", None))
    if not blocks:
        return []

    deck = Deck(cfg)
    slides: list[dict] = []
    for index, block in enumerate(blocks):
        if cancel_check:
            cancel_check()
        override = (
            chart_path_overrides[index]
            if isinstance(chart_path_overrides, list) and index < len(chart_path_overrides)
            else None
        )
        model = build_slide_model(
            block,
            cfg,
            deck,
            chart_path_override=override,
            chart_render_spec_mode=chart_render_spec_mode,
        )
        model["preview_manifest"] = deck.build_preview_manifest(
            model, slide_number=index + 1
        )
        slides.append(model)
    return slides


def finalize_slide_data(
    slides: Sequence[dict],
    chart_path_overrides: list[str | None] | None = None,
    cancel_check: Any = None,
) -> list[dict]:
    """
    Attach browser-rendered chart PNGs to already-generated slide models and
    rebuild layout/preview metadata without regenerating titles, KPIs, bullets,
    or LLM-authored text.
    """
    cfg = DeckConfig(
        template_path=_resolve_local_asset_path("aadibio_ppt.pptx") or "aadibio_ppt.pptx",
        logo_path=_resolve_local_asset_path("aadibio_logo.png") or "aadibio_logo.png",
    )
    deck = Deck(cfg)
    finalized: list[dict] = []
    for index, slide in enumerate(slides):
        if cancel_check:
            cancel_check()
        model = dict(slide)
        if isinstance(chart_path_overrides, list) and index < len(chart_path_overrides):
            override = chart_path_overrides[index]
            if override:
                model["chart_path"] = override
                log.info("[deck-debug] finalized slide[%d] with browser chart path=%s",
                         index + 1, override)
        if model.get("chart_path"):
            model.pop("chart_expected", None)
        model["layout"] = choose_layout(model)
        model["preview_manifest"] = deck.build_preview_manifest(
            model, slide_number=index + 1
        )
        finalized.append(model)
    return finalized


def build_ppt_from_slide_data(slides: Sequence[dict],
                              uploaded_pptx_path: str | None = "aadibio_ppt.pptx",
                              logo_path: str | None = "aadibio_logo.png",
                              output_path: str = "final_presentation.pptx",
                              **overrides) -> str:
    """Render a PPT from prebuilt slide models without regenerating content."""
    if not slides:
        raise ValueError("No slide data available.")

    resolved_template = _resolve_local_asset_path(uploaded_pptx_path) or uploaded_pptx_path
    resolved_logo = _resolve_local_asset_path(logo_path) or logo_path
    cfg = DeckConfig(template_path=resolved_template, logo_path=resolved_logo,
                     output_path=output_path)
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    deck = Deck(cfg)
    if cfg.cover_slide:
        deck.add_cover([model.get("question", "") for model in slides])
    if cfg.agenda_slide and len(slides) > 1:
        deck.add_agenda([model.get("question", "") for model in slides])

    for model in slides:
        try:
            deck.add_analysis_slide(model)
        except Exception as exc:
            log.error("slide '%s' failed to render: %s",
                      model.get("title", "?"), exc)

    if cfg.closing_slide and len(slides) > 1:
        deck.add_closing([model.get("headline") or model.get("title") for model in slides])

    path = deck.save(cfg.output_path)
    _cleanup_charts(slides, cfg)
    log.info("[deck-debug] build_ppt_from_slide_data done slides=%d output=%s",
             len(slides), path)
    return path


def deck_filename(prefix: str = "commercial_review") -> str:
    return (f"{prefix}_{date.today().strftime('%Y%m%d')}"
            f"_{uuid.uuid4().hex[:6]}.pptx")


# ── v1 compatibility ─────────────────────────────────────────────────────────

def build_ppt(messages: Iterable[Any],
              uploaded_pptx_path: str | None = "aadibio_ppt.pptx",
              logo_path: str | None = "aadibio_logo.png",
              output_path: str = "final_presentation.pptx",
              **overrides) -> str:
    """Drop-in replacement for the v1 entry point."""
    resolved_template = _resolve_local_asset_path(uploaded_pptx_path) or uploaded_pptx_path
    resolved_logo = _resolve_local_asset_path(logo_path) or logo_path
    cfg = DeckConfig(template_path=resolved_template, logo_path=resolved_logo,
                     output_path=output_path)
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    log.info("[deck-debug] build_ppt entry output=%s template=%s logo=%s overrides=%s",
             cfg.output_path, cfg.template_path, cfg.logo_path,
             ", ".join(sorted(overrides.keys())) or "(none)")
    return build_deck(messages, cfg)


def create_ppt(slide_data: dict, prs: Presentation | None = None,
               theme: Theme | dict | None = None,
               logo_path: str | None = None) -> Presentation:
    """
    v1-style single-slide builder. Kept so existing callers keep working;
    new code should use build_deck().
    """
    resolved_logo = _resolve_local_asset_path(logo_path) or logo_path
    cfg = DeckConfig(template_path=None, logo_path=resolved_logo,
                     cover_slide=False, agenda_slide=False, closing_slide=False)
    deck = getattr(prs, "_deck", None) if prs is not None else None
    if deck is None:
        deck = Deck(cfg, theme if isinstance(theme, Theme) else None)
        try:
            deck.prs._deck = deck
        except Exception:
            pass

    model = dict(slide_data)
    model.setdefault("eyebrow", _eyebrow_for(model.get("title", "")))
    model.setdefault("headline", "")
    model.setdefault("kpis", [])
    model.setdefault("bullets", [])
    df = to_dataframe(model.get("data"))
    model["df"] = df
    model["profile"] = profile_dataframe(df)
    model.setdefault("chips", model["profile"].get("chips") or [])
    model.setdefault("basis", "Not stated")
    model["layout"] = choose_layout(model)
    deck.add_analysis_slide(model)
    return deck.prs


__all__ = [
    "DeckConfig", "DeckHooks", "Theme", "Deck", "autodiscover_hooks",
    "build_deck", "build_deck_bytes", "build_slide_data", "finalize_slide_data",
    "build_ppt_from_slide_data",
    "build_ppt", "create_ppt",
    "slice_turns", "deck_filename",
    "parse_conversation", "extract_export_context", "extract_theme", "extract_theme_from_pptx",
    "generate_slide_content", "build_chart_render_spec", "render_chart", "to_dataframe",
    "profile_dataframe", "SYSTEM_PROMPT", "USER_PROMPT",
]
