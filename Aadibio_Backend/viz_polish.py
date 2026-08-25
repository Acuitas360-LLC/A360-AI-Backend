"""
viz_polish.py
--------------------------------------------------------------------------
Deterministic presentation layer for LLM-generated Plotly figures.

The visualization agent decides WHAT to plot (chart type, x, y, traces).
This module decides HOW it looks: spacing, fonts, ticks, labels, legend,
colours, height, margins.

Nothing here depends on the LLM getting styling right, so the output is
consistent across every question. Every step is individually guarded, so a
failure in one step never loses the chart.

Usage (Streamlit):
    from viz_polish import sanitize_viz_code, apply_classic_layout
    exec(sanitize_viz_code(code), scope)
    fig = apply_classic_layout(scope.get("fig"))

Usage (deck export):
    fig = apply_classic_layout(fig, for_slide=True)
"""

from __future__ import annotations

import math
import re

import pandas as pd

# --------------------------------------------------------------------------
# THEME
# --------------------------------------------------------------------------

# Deep blue primary + warm accent: the second trace (usually growth %) always
# reads as clearly different from the first without shouting.
CLASSIC_PALETTE = [
    "#1F4E79",  # deep blue      - primary metric
    "#E8853B",  # warm orange    - secondary metric / growth
    "#6A9FB5",  # muted steel
    "#8C8C8C",  # grey
    "#7E5A9B",  # plum
    "#4C8C5A",  # green
]

FONT_FAMILY = "Segoe UI, Inter, Helvetica Neue, Arial, sans-serif"
INK = "#1A1A1A"
MUTED_INK = "#5A5A5A"
GRID = "#E6E6E6"

# Above this many plotted points, per-point data labels stop helping and start
# colliding, so they are dropped and the numbers live in the hover instead.
# Only consulted when force_labels=False; the default path resolves label
# collisions geometrically instead of by point budget.
MAX_LABELLED_POINTS = 12

# Nominal plot-area size used to decide which line labels would overlap.
# The figure is rendered at container width (width=None), so the exact pixel
# count is not knowable here — these are the typical Streamlit dimensions and
# only need to be right to within ~20% for the collision test to hold up.
PLOT_WIDTH_PX = 900
PLOT_HEIGHT_PX = 360

# Above this many x categories, tick labels are angled instead of horizontal.
ANGLE_TICKS_ABOVE = 6

PCT_HINTS = (
    "%", "pct", "percent", "growth", "change", "share",
    "rate", "wow", "mom", "qoq", "yoy",
)


# --------------------------------------------------------------------------
# CODE SANITISER
# --------------------------------------------------------------------------

def sanitize_viz_code(code: str) -> str:
    """Strip anything the model may emit that breaks or fights exec()."""
    if not code:
        return ""

    code = code.strip()
    code = re.sub(r"^```[A-Za-z]*\s*", "", code)
    code = re.sub(r"```\s*$", "", code)

    drop_prefixes = ("fig.show", "st.", "import streamlit", "plt.show")
    kept = [ln for ln in code.splitlines()
            if not ln.strip().startswith(drop_prefixes)]

    return "\n".join(kept).strip()


# --------------------------------------------------------------------------
# TIME GRANULARITY
# --------------------------------------------------------------------------

# Coarsest to finest. A monthly result set still carries week_end_date because
# the SQL generator is required to emit period boundaries, so the presence of a
# column says nothing about the grain — only the distinct counts do.
TIME_COLUMNS = (
    "year",
    "fiscal_year",
    "quarter_year",
    "quarter",
    "quarter_start_date",
    "quarter_end_date",
    "month_year",
    "month",
    "month_start_date",
    "month_end_date",
    "period_start_date",
    "period_end_date",
    "week_end_date",
    "week_start_date",
    "week",
    "transaction_date",
    "call_date",
    "date",
)


def pick_time_column(df):
    """
    Return the column that actually carries the result's time grain.

    The rule is distinct counts, not column names. If month_year and
    week_end_date have the same number of distinct values, there is exactly
    one row per month and week_end_date is just a boundary artifact — the
    grain is monthly. If week_end_date has more, the grain is weekly.

    Matching is case-insensitive: Snowflake returns MONTH_YEAR, pandas
    pipelines often lowercase it, and both must resolve.
    """
    if df is None or getattr(df, "empty", True):
        return None

    actual = {str(c).strip().lower(): c for c in df.columns}
    present = [actual[name] for name in TIME_COLUMNS if name in actual]
    if not present:
        return None

    counts = {c: int(df[c].nunique(dropna=True)) for c in present}
    finest = max(counts.values())

    # A single period cannot be disambiguated — leave whatever was plotted.
    if finest < 2:
        return None

    for col in present:                     # coarsest first
        if counts[col] == finest:
            return col
    return present[-1]


def _match_column(df, values):
    """Find which df column a trace was plotted from, by value set."""
    if df is None or values is None:
        return None
    try:
        plotted = set(pd.Series(list(values)).dropna())
    except Exception:
        return None
    if not plotted:
        return None

    for col in df.columns:
        try:
            if plotted <= set(df[col].dropna()):
                return col
        except TypeError:
            continue
    return None


def _align_time_axis(fig, df=None):
    """
    Re-point the X axis at the column matching the result's grain.

    A monthly question whose SQL emits one row per month gets month_year on
    the axis instead of the week-ending date that happens to sit in the same
    row. Values are remapped through a 1:1 lookup, so sorting and filtering
    done by the generated code are preserved.
    """
    target = pick_time_column(df)
    if target is None:
        return

    for tr in fig.data:
        source = _match_column(df, getattr(tr, "x", None))
        if source is None or source == target:
            continue
        if str(source).strip().lower() not in TIME_COLUMNS:
            continue

        pairs = df[[source, target]].dropna().drop_duplicates()
        if len(pairs) != pairs[source].nunique():
            continue                        # not 1:1, remapping would lie

        lookup = dict(zip(pairs[source], pairs[target]))
        tr.x = [lookup.get(v, v) for v in tr.x]

        if getattr(tr, "hovertemplate", None):
            tr.hovertemplate = None         # rebuilt later against new values

    if fig.layout.xaxis.title and fig.layout.xaxis.title.text:
        fig.layout.xaxis.title.text = target


# --------------------------------------------------------------------------
# EXEC SCOPE
# --------------------------------------------------------------------------

def build_exec_scope(df=None, **extra):
    """
    Namespace for exec()-ing generated visualization code.

    Every library the agent might reach for is preloaded, so code that omits
    its import lines still runs. Import statements in the generated code are
    harmless — they simply rebind the same names.
    """
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scope = {
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "make_subplots": make_subplots,
        "plotly": __import__("plotly"),
    }
    if df is not None:
        scope["df"] = df.copy()
    scope.update(extra)
    return scope


# --------------------------------------------------------------------------
# SMALL HELPERS
# --------------------------------------------------------------------------

def _text(value) -> str:
    return "" if value is None else str(value)


def _looks_percentage(*labels) -> bool:
    blob = " ".join(_text(v) for v in labels).lower()
    return any(hint in blob for hint in PCT_HINTS)


def _numeric_values(seq):
    out = []
    if seq is None:
        return out
    for v in seq:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:  # skip NaN
            out.append(f)
    return out


def _trace_values(tr):
    """Numeric values a trace draws on its value axis."""
    if getattr(tr, "type", "") in ("pie", "funnel"):
        return _numeric_values(getattr(tr, "values", None))
    if getattr(tr, "orientation", None) == "h":
        return _numeric_values(getattr(tr, "x", None))
    return _numeric_values(getattr(tr, "y", None))


def _trace_categories(tr):
    """String category labels a trace draws on its category axis."""
    axis = "y" if getattr(tr, "orientation", None) == "h" else "x"
    if getattr(tr, "type", "") == "pie":
        raw = getattr(tr, "labels", None)
    else:
        raw = getattr(tr, axis, None)
    if raw is None:
        return []
    return [str(v) for v in raw if isinstance(v, str)]


def _point_count(tr) -> int:
    for attr in ("x", "y", "values", "labels"):
        seq = getattr(tr, attr, None)
        if seq is not None:
            try:
                return len(seq)
            except TypeError:
                continue
    return 0


def _value_format(values, is_pct: bool) -> str:
    """Tick/label format chosen from the actual magnitude of the data."""
    if is_pct:
        return ".1f"
    peak = max((abs(v) for v in values), default=0)
    if peak >= 10_000:
        return ".3~s"   # 2.9M, 3.33M, 850k
    if peak >= 100:
        return ",.0f"    # 1,240
    if peak >= 10:
        return ",.1f"
    return ",.2f"


def _date_format(fig) -> str:
    """Pick a tick format from the actual spacing of the time axis."""
    stamps = []
    for tr in fig.data:
        raw = getattr(tr, "x", None)
        if raw is None:
            continue
        try:
            parsed = pd.to_datetime(pd.Series(list(raw)), errors="coerce").dropna()
        except Exception:
            continue
        stamps.extend(parsed.tolist())

    if len(stamps) < 3:
        return "%d %b %Y"

    gap = pd.Series(sorted(set(stamps))).diff().dropna().median()
    days = getattr(gap, "days", 0)

    if days >= 26:          # monthly, quarterly, yearly
        return "%b %Y"
    if days >= 5:           # weekly
        return "%d %b %Y"
    return "%d %b"          # daily


def _tick_block_px(fig, scale: float) -> int:
    """
    Estimated vertical pixels consumed by the x tick labels.

    Rotated labels grow downward with the length of the longest label, so the
    legend offset and bottom margin must both be derived from this rather than
    from a fixed guess.
    """
    axis = fig.layout.xaxis

    categories = []
    for tr in fig.data:
        categories.extend(_trace_categories(tr))
    if not categories:
        return int(22 * scale)          # single row of dates or numbers

    labels = list(axis.ticktext) if axis.ticktext else categories
    longest = max((len(str(lbl)) for lbl in labels), default=0)

    try:
        angle = abs(float(axis.tickangle))
    except (TypeError, ValueError):
        angle = 0.0

    if angle < 1:
        return int(24 * scale)

    glyph = 6.4 * scale                 # ~11px font, average glyph width
    return int(math.sin(math.radians(angle)) * longest * glyph) + int(14 * scale)


MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _period_label(value) -> str:
    """2024-08 -> Aug 2024, 2024-Q3 -> Q3 2024. Anything else is returned as is."""
    text = str(value).strip()

    match = re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])", text)
    if match:
        year, month = match.groups()
        return f"{MONTH_NAMES[int(month) - 1]} {year}"

    match = re.fullmatch(r"(\d{4})[-\s]?Q([1-4])", text, re.I)
    if match:
        year, quarter = match.groups()
        return f"Q{quarter} {year}"

    return text


def _shorten(label: str, limit: int) -> str:
    label = label.strip()
    return label if len(label) <= limit else label[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------------------------------

def apply_classic_layout(
    fig,
    df=None,
    for_slide: bool = False,
    max_labelled_points: int = MAX_LABELLED_POINTS,
    strip_floating_annotations: bool = True,
    force_labels: bool = True,
):
    """
    Return `fig` restyled into a clean, classic, uncrowded chart.

    for_slide     : bumps every font size for PowerPoint export.
    df            : unused today, accepted so callers can pass context safely.
    force_labels  : show data labels regardless of point/series count, relying
                     on uniformtext(mode="hide") to drop any label that would
                     actually collide. Set False to restore the old
                     conservative point-budget gating.
    """
    if fig is None:
        return None

    scale = 1.25 if for_slide else 1.0
    steps = (
        (_base_layout, dict(scale=scale)),
        (_align_time_axis, dict(df=df)),
        (_flip_long_category_bars, {}),
        (_style_traces, dict(scale=scale, max_labelled_points=max_labelled_points,
                              force_labels=force_labels)),
        (_style_axes, dict(scale=scale)),
        (_style_legend, dict(scale=scale)),
        (_declutter, dict(strip_floating_annotations=strip_floating_annotations)),
        (_set_geometry, dict(for_slide=for_slide, scale=scale)),
    )

    for step, kwargs in steps:
        try:
            step(fig, **kwargs)
        except Exception as exc:  # never lose a chart over cosmetics
            print(f"[viz_polish] {step.__name__} skipped: {exc}")

    return fig


# --------------------------------------------------------------------------
# STEP 1 — BASE LAYOUT
# --------------------------------------------------------------------------

def _base_layout(fig, scale: float = 1.0):
    fig.update_layout(
        template="plotly_white",
        colorway=CLASSIC_PALETTE,
        font=dict(family=FONT_FAMILY, size=int(12 * scale), color=INK),
        paper_bgcolor="white",
        plot_bgcolor="white",
        # Plotly hides any data label that cannot fit — the single most
        # effective guard against label collisions.
        uniformtext=dict(minsize=int(9 * scale), mode="hide"),
        bargap=0.30,
        bargroupgap=0.12,
        hoverlabel=dict(
            namelength=-1,
            bgcolor="white",
            bordercolor=GRID,
            font=dict(family=FONT_FAMILY, size=int(12 * scale), color=INK),
        ),
        separators=".,",
        width=None,  # let the container decide, never a hard-coded width
    )

    # Title: left-aligned, wrapped, with breathing room above the plot.
    title_text = ""
    if fig.layout.title and fig.layout.title.text:
        title_text = re.sub(r"<br\s*/?>", " ", fig.layout.title.text).strip()

    if title_text:
        fig.update_layout(
            title=dict(
                text=_wrap(title_text, 68),
                x=0.0,
                xref="paper",
                xanchor="left",
                y=0.97,
                yanchor="top",
                font=dict(size=int(17 * scale), color=INK),
            )
        )


def _wrap(text: str, width: int) -> str:
    """Wrap a title onto at most two lines at a word boundary."""
    if len(text) <= width:
        return text
    cut = text.rfind(" ", 0, width)
    if cut == -1:
        cut = width
    return text[:cut] + "<br>" + text[cut:].strip()


# --------------------------------------------------------------------------
# STEP 1b — ORIENTATION
# --------------------------------------------------------------------------

def _is_horizontal(fig) -> bool:
    return any(getattr(tr, "orientation", None) == "h" for tr in fig.data)


def _flip_long_category_bars(fig, min_categories: int = 11, min_label_len: int = 13):
    """
    A single bar series with many long account names is unreadable vertically —
    angled ticks turn into a wall of text. Rotating to horizontal bars is the
    classic fix: labels sit flat and the ranking reads top to bottom.

    Deliberately conservative: single trace, plain vertical bars only.
    """
    if len(fig.data) != 1:
        return

    tr = fig.data[0]
    if getattr(tr, "type", "") != "bar" or getattr(tr, "orientation", None) == "h":
        return
    if getattr(fig.layout, "yaxis2", None) is not None and fig.layout.yaxis2.overlaying:
        return

    categories = _trace_categories(tr)
    if len(categories) < min_categories:
        return
    if max((len(c) for c in categories), default=0) < min_label_len:
        return

    tr.x, tr.y = list(tr.y), list(tr.x)
    tr.orientation = "h"
    tr.hovertemplate = None

    x_title = fig.layout.xaxis.title.text if fig.layout.xaxis.title else None
    y_title = fig.layout.yaxis.title.text if fig.layout.yaxis.title else None
    fig.layout.xaxis.title.text = y_title
    fig.layout.yaxis.title.text = x_title

    # Largest bar on top, regardless of the order the SQL returned.
    fig.layout.yaxis.categoryorder = "total ascending"


def _positional_values(tr):
    """
    Numeric y-values in original point order, NaN kept as a placeholder
    rather than dropped.

    Unlike _trace_values (used for axis/format decisions, where dropping
    gaps is fine), label thinning has to index into this list by position
    and line it up 1:1 with tr.x — silently dropping NaNs here would shift
    every label onto the wrong point.
    """
    raw = getattr(tr, "y", None)
    if raw is None:
        return []
    out = []
    for v in raw:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _select_label_indices(values, texts, sides, font_px: float,
                          plot_w_px: float, plot_h_px: float) -> set:
    """
    Keep every label that fits; drop only the ones that would actually
    overlap something already placed.

    A flat budget ("show 12") is the wrong instrument, because whether two
    labels collide is a question about pixels, not point count. Adjacent
    weeks that swing 65 -> 133 put their labels 60px apart vertically and
    never touch, while a flat stretch at ~100 collides at the same spacing.
    So each candidate label is turned into a bounding box in plot pixels
    and tested against the boxes already placed; anything clear gets shown.

    Priority order matters because the first labels placed win the space:
    first point, last point, peak and trough go down first (those are the
    numbers a reader looks for), then the rest left to right.
    """
    n = len(values)
    if n == 0:
        return set()

    finite = [v for v in values if v == v]
    if not finite:
        return set()

    span = (max(finite) - min(finite)) or 1.0
    y_floor = min(finite)

    # Points are evenly spaced along the category/date axis in every chart
    # this module produces, so index position maps linearly to pixels.
    step_px = plot_w_px / max(n - 1, 1)

    def box(i):
        text = texts[i]
        if not text:
            return None
        # 0.62em average glyph width is a good fit for the digits and
        # separators these labels contain, plus half an em of breathing
        # room so boxes that merely touch still count as clear.
        half_w = (len(text) * 0.62 * font_px + 0.5 * font_px) / 2.0
        half_h = 1.4 * font_px / 2.0
        cx = i * step_px
        # The box centre is offset one line clear of the marker, on whichever
        # side _label_positions put this label.
        lift = -1.1 * font_px if sides[i].startswith("bottom") else 1.1 * font_px
        cy = ((values[i] - y_floor) / span) * plot_h_px + lift
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def clear(candidate, placed):
        ax0, ay0, ax1, ay1 = candidate
        for bx0, by0, bx1, by1 in placed:
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                return False
        return True

    numeric = [(i, v) for i, v in enumerate(values) if v == v]
    priority = [0, n - 1,
                max(numeric, key=lambda p: p[1])[0],
                min(numeric, key=lambda p: p[1])[0]]

    order = []
    seen = set()
    for i in priority + list(range(n)):
        if i not in seen and 0 <= i < n:
            seen.add(i)
            order.append(i)

    keep, placed = set(), []
    for i in order:
        b = box(i)
        if b is None or not clear(b, placed):
            continue
        keep.add(i)
        placed.append(b)

    return keep


def _label_positions(values, base: str):
    """
    Put each label on the side of the point the line isn't running through.

    A label above a trough gets struck through by the two segments climbing
    away from it, which is what makes a dense line chart look shredded even
    after collisions are resolved. Flipping local minima below the marker
    (and local maxima above, when the trace's base side is below) puts the
    text in the open wedge instead.
    """
    n = len(values)
    flipped = "bottom center" if base == "top center" else "top center"

    out = []
    for i, v in enumerate(values):
        if v != v:
            out.append(base)
            continue
        prev = values[i - 1] if i > 0 else v
        nxt = values[i + 1] if i < n - 1 else v
        prev = prev if prev == prev else v
        nxt = nxt if nxt == nxt else v

        trough = v <= prev and v <= nxt
        peak = v >= prev and v >= nxt

        if base == "top center":
            out.append(flipped if (trough and not peak) else base)
        else:
            out.append(flipped if (peak and not trough) else base)
    return out


def _format_number(v: float, fmt: str, suffix: str) -> str:
    """
    Python-side twin of the small d3-format vocabulary _value_format hands
    to Plotly's texttemplate.

    Thinned line labels are built as literal per-point text (so individual
    points can be left blank), which means the value has to be formatted
    here in Python rather than by Plotly's own "%{y:fmt}" template engine.
    """
    if v != v:  # NaN
        return ""
    if fmt == ".1f":
        out = f"{v:.1f}"
    elif fmt == ",.1f":
        out = f"{v:,.1f}"
    elif fmt == ",.2f":
        out = f"{v:,.2f}"
    elif fmt == ".3~s":
        out = _si_format(v)
    else:  # ",.0f" and any unrecognised fallback
        out = f"{v:,.0f}"
    return out + suffix


def _si_format(v: float) -> str:
    """3-sig-fig SI-suffix format mirroring d3's '.3~s' (e.g. 2.9M, 850k)."""
    sign = "-" if v < 0 else ""
    v = abs(v)
    for threshold, letter in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if v >= threshold:
            return f"{sign}{v / threshold:.3g}{letter}"
    return f"{sign}{v:.3g}"


# --------------------------------------------------------------------------
# STEP 2 — TRACES
# --------------------------------------------------------------------------

def _style_traces(fig, scale: float = 1.0, max_labelled_points: int = MAX_LABELLED_POINTS,
                   force_labels: bool = True):
    traces = list(fig.data)
    busiest = max((_point_count(tr) for tr in traces), default=0)
    n_series = len(traces)

    # A bar + line combo is the one case that stays capped even under
    # force_labels: the growth line traverses the tops of the bars, so line
    # labels sit exactly where the line runs and get struck through. Bar
    # labels are unaffected and still show.
    kinds = [getattr(tr, "type", "") for tr in traces]
    combo = (n_series == 2
             and kinds.count("bar") == 1
             and any(k in ("scatter", "scattergl") for k in kinds))

    if force_labels:
        # uniformtext(mode="hide") in _base_layout already drops any label
        # that doesn't fit without colliding, so it's safe to always attempt
        # labels here rather than pre-emptively suppressing by point count.
        label_bars = True
        label_lines = not combo
    elif n_series <= 1:
        label_bars = label_lines = busiest <= max_labelled_points + 2
    elif combo:
        label_bars = label_lines = False
    elif n_series == 2:
        label_bars = label_lines = busiest <= max_labelled_points // 2
    else:
        label_bars = label_lines = False

    label_size = int(10 * scale)

    line_seen = 0

    for idx, tr in enumerate(traces):
        kind = getattr(tr, "type", "")
        colour = CLASSIC_PALETTE[idx % len(CLASSIC_PALETTE)]
        is_pct = _looks_percentage(getattr(tr, "name", ""), getattr(tr, "yaxis", ""))
        fmt = _value_format(_trace_values(tr), is_pct)
        suffix = "%" if is_pct else ""

        # Legend entries: short, readable, no snake_case.
        if getattr(tr, "name", None):
            tr.name = _shorten(_prettify(tr.name), 28)

        if kind == "bar":
            horizontal_bar = getattr(tr, "orientation", None) == "h"
            tr.marker.color = colour
            tr.marker.line = dict(width=0)
            # Horizontal bars get one label per row, which can never collide
            # with its neighbours, so the point budget does not apply.
            if label_bars or horizontal_bar:
                tr.texttemplate = ("%{x:" + fmt + "}" + suffix
                                   if horizontal_bar
                                   else "%{y:" + fmt + "}" + suffix)
                tr.textposition = "outside"
                tr.textfont = dict(size=label_size, color=MUTED_INK)
                tr.cliponaxis = False
            else:
                tr.text = None
                tr.texttemplate = None

        elif kind in ("scatter", "scattergl"):
            tr.line.color = colour
            tr.line.width = 2.6 * scale
            tr.line.simplify = False
            if getattr(tr, "mode", None) and "markers" in tr.mode:
                tr.marker.size = 7 * scale
                tr.marker.color = colour
            elif tr.mode in (None, "lines"):
                tr.mode = "lines+markers"
                tr.marker.size = 6 * scale
                tr.marker.color = colour
            if label_lines:
                # uniformtext(mode="hide") — the collision guard relied on
                # for bars above — only ever applies to bar/pie-style text.
                # Plotly never auto-hides overlapping scatter/line labels, so
                # a 52-point R52W series labels every point and the numbers
                # stack. Collisions are therefore resolved here, in pixels:
                # every label that fits is kept, and only genuine overlaps
                # are dropped to hover.
                positions = _positional_values(tr)
                candidates = [_format_number(v, fmt, suffix) for v in positions]
                # Alternate the base side so two lines never stack labels.
                base_side = "top center" if line_seen % 2 == 0 else "bottom center"
                sides = _label_positions(positions, base_side)
                keep = _select_label_indices(
                    positions, candidates, sides, float(label_size),
                    PLOT_WIDTH_PX * (1.15 if scale > 1 else 1.0),
                    PLOT_HEIGHT_PX,
                )
                tr.text = [t if i in keep else ""
                           for i, t in enumerate(candidates)]
                tr.texttemplate = "%{text}"
                tr.textposition = sides
                tr.textfont = dict(size=label_size, color=MUTED_INK)
                tr.cliponaxis = False
                if "text" not in (tr.mode or ""):
                    tr.mode = (tr.mode or "lines+markers") + "+text"
            else:
                tr.text = None
                tr.texttemplate = None
                if tr.mode and "+text" in tr.mode:
                    tr.mode = tr.mode.replace("+text", "")
            line_seen += 1

        elif kind == "pie":
            tr.textinfo = "label+percent"
            tr.texttemplate = "%{label}<br>%{percent:.1%}"
            tr.textposition = "outside"
            tr.textfont = dict(size=int(11 * scale), color=INK)
            tr.marker.colors = CLASSIC_PALETTE[: _point_count(tr)]
            tr.marker.line = dict(color="white", width=1.5)
            tr.sort = True
            tr.hole = 0.0
            tr.automargin = True

        # Hover: one clean line per point, no truncation. Full category name
        # appears here even when the tick label had to be shortened.
        if kind != "pie" and not getattr(tr, "hovertemplate", None):
            if getattr(tr, "orientation", None) == "h":
                tr.hovertemplate = "%{y}<br>%{x:" + fmt + "}" + suffix + "<extra></extra>"
            else:
                tr.hovertemplate = "%{x}<br>%{y:" + fmt + "}" + suffix + "<extra></extra>"


ACRONYMS = {
    "Wow": "WoW", "Mom": "MoM", "Qoq": "QoQ", "Yoy": "YoY", "Ytd": "YTD",
    "Pap": "PAP", "Com": "COM", "Hcp": "HCP", "Npi": "NPI", "Id": "ID",
    "Ids": "IDs", "Mtor": "mTOR", "Avg": "Avg", "Qty": "Qty",
}


def _prettify(name: str) -> str:
    """snake_case column names -> readable axis and legend labels."""
    name = str(name).strip()
    if "_" in name or name.islower():
        name = name.replace("_", " ").strip().title()
    name = re.sub(r"\bPct\b", "%", name)
    return " ".join(ACRONYMS.get(word, word) for word in name.split(" "))


# --------------------------------------------------------------------------
# STEP 3 — AXES
# --------------------------------------------------------------------------

def _style_axes(fig, scale: float = 1.0):
    horizontal = _is_horizontal(fig)

    categories = []
    for tr in fig.data:
        categories.extend(_trace_categories(tr))
    categories = list(dict.fromkeys(categories))

    if horizontal:
        # A rotated title squeezed against the left edge adds nothing when the
        # category names are already spelled out down the axis.
        if fig.layout.yaxis.title:
            fig.layout.yaxis.title.text = ""
        _category_axis(fig, "yaxis", categories, scale, angle=False)
        _value_axis(fig, "xaxis", scale)
    else:
        _category_axis(fig, "xaxis", categories, scale, angle=True)
        for axis_name in ("yaxis", "yaxis2"):
            _value_axis(fig, axis_name, scale)

    # A shared tooltip reads best on a time axis; elsewhere the nearest
    # point is what the user is pointing at.
    fig.update_layout(hovermode="closest" if categories else "x unified")


def _category_axis(fig, axis_name: str, categories, scale: float, angle: bool):
    axis = getattr(fig.layout, axis_name, None)
    if axis is None:
        return

    longest = max((len(c) for c in categories), default=0)

    axis.update(
        showgrid=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
        ticklen=5,
        tickfont=dict(size=int(11 * scale), color=MUTED_INK),
        title_standoff=18,
        automargin=True,
    )

    if categories:
        # The tick labels already name the categories, so an axis title like
        # "Parent Name" only adds a row of text between the ticks and the
        # legend. Date and numeric axes keep their titles.
        if axis.title:
            axis.title.text = ""

        display = [_period_label(c) for c in categories]
        if display != list(categories):
            # Underlying values stay as "2024-08" so chronological order is
            # preserved; only the tick text becomes "Aug 2024".
            axis.update(tickmode="array", tickvals=categories, ticktext=display)
            longest = max((len(d) for d in display), default=0)
        elif longest > 18:
            axis.update(tickmode="array",
                        tickvals=categories,
                        ticktext=[_shorten(c, 18) for c in categories])
        if angle and (len(categories) > ANGLE_TICKS_ABOVE or longest > 10):
            axis.tickangle = -35
    else:
        _pin_date_ticks(fig, axis)

    _title_font(axis, scale)


def _pin_date_ticks(fig, axis, max_ticks: int = 12):
    """
    Put ticks on the data points, not on positions Plotly interpolates.

    A monthly series drawn on a continuous date axis gets ticks wherever the
    algorithm likes — mid-April, mid-May — and once those are formatted as
    "%b %Y" the axis reads Apr 2026, Apr 2026, May 2026. Pinning ticks to the
    actual periods makes one tick per period, always.
    """
    stamps = []
    for tr in fig.data:
        raw = getattr(tr, "x", None)
        if raw is None:
            continue
        try:
            parsed = pd.to_datetime(pd.Series(list(raw)), errors="coerce").dropna()
        except Exception:
            continue
        stamps.extend(parsed.tolist())

    fmt = _date_format(fig)

    if not stamps:
        axis.update(nticks=9, tickformat=fmt)
        return

    points = sorted(set(stamps))
    if len(points) > max_ticks:
        # Too many to label individually — thin them evenly, keeping the ends.
        step = -(-len(points) // max_ticks)
        points = points[::step] + [points[-1]]
        points = sorted(set(points))

    labels = [p.strftime(_strftime(fmt)) for p in points]

    # If the chosen format collapses distinct periods onto the same label,
    # fall back to a finer one rather than repeating text.
    if len(set(labels)) != len(labels):
        labels = [p.strftime("%d %b %Y") for p in points]

    axis.update(tickmode="array", tickvals=points, ticktext=labels)


def _strftime(plotly_fmt: str) -> str:
    """Plotly's d3 time formats used here happen to be valid strftime."""
    return plotly_fmt


def _value_axis(fig, axis_name: str, scale: float):
    axis = getattr(fig.layout, axis_name, None)
    if axis is None:
        return

    secondary = axis_name.endswith("2")
    letter = "x" if axis_name.startswith("x") else "y"

    values, is_pct = [], False
    for tr in fig.data:
        anchor = getattr(tr, letter + "axis", letter) or letter
        if anchor != (letter + "2" if secondary else letter):
            continue
        values.extend(_trace_values(tr))
        is_pct = is_pct or _looks_percentage(getattr(tr, "name", ""))

    is_pct = is_pct or _looks_percentage(axis.title.text if axis.title else "")
    fmt = _value_format(values, is_pct)

    axis.update(
        showgrid=not secondary,   # one grid only — never two overlaid
        gridcolor=GRID,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=GRID,
        showline=False,
        tickfont=dict(size=int(11 * scale), color=MUTED_INK),
        tickformat=fmt,
        ticksuffix="%" if is_pct else "",
        title_standoff=20,
        automargin=True,
    )
    _title_font(axis, scale)


def _title_font(axis, scale: float):
    if axis.title and axis.title.text:
        axis.title.text = _prettify(axis.title.text)
        axis.title.font = dict(size=int(12 * scale), color=MUTED_INK)


# --------------------------------------------------------------------------
# STEP 4 — LEGEND
# --------------------------------------------------------------------------

def _style_legend(fig, scale: float = 1.0):
    named = [tr for tr in fig.data
             if getattr(tr, "name", None) and getattr(tr, "type", "") != "pie"]

    # A single series needs no legend — the axis title already says what it is.
    if len(fig.data) <= 1 or len(named) <= 1:
        fig.update_layout(showlegend=False)
        return

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,          # placeholder — _set_geometry recomputes this
                              # from the measured tick block
            xanchor="center",
            x=0.5,
            title_text="",
            font=dict(size=int(11 * scale), color=MUTED_INK),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )


# --------------------------------------------------------------------------
# STEP 5 — DECLUTTER
# --------------------------------------------------------------------------

def _declutter(fig, strip_floating_annotations: bool = True):
    if not strip_floating_annotations:
        return

    # Drop callouts pinned to data coordinates (the editorial "West is the
    # weakest region" style notes). Paper-anchored annotations — subplot
    # titles, axis notes — are kept.
    kept = []
    for ann in (fig.layout.annotations or []):
        xref = _text(getattr(ann, "xref", "paper"))
        yref = _text(getattr(ann, "yref", "paper"))
        if xref.startswith("paper") and yref.startswith("paper"):
            kept.append(ann)
    fig.layout.annotations = tuple(kept)

    # Drop full-width dashed reference lines, which stack into an unreadable
    # ladder on multi-metric charts.
    shapes = []
    for shp in (fig.layout.shapes or []):
        xref = _text(getattr(shp, "xref", ""))
        # add_hline / add_vline anchor to "paper" or "x domain" depending on
        # the Plotly version — both mean "spans the whole plot".
        spans_plot = (getattr(shp, "type", "") == "line"
                      and (xref.startswith("paper") or "domain" in xref))
        if not spans_plot:
            shapes.append(shp)
    fig.layout.shapes = tuple(shapes)


# --------------------------------------------------------------------------
# STEP 6 — GEOMETRY
# --------------------------------------------------------------------------

def _set_geometry(fig, for_slide: bool = False, scale: float = 1.0):
    categories = []
    for tr in fig.data:
        categories.extend(_trace_categories(tr))
    n_cat = len(set(categories))
    n_traces = len(fig.data)
    horizontal = _is_horizontal(fig)

    # ---- top: title block ----
    has_title = bool(fig.layout.title and fig.layout.title.text)
    two_line_title = has_title and "<br>" in _text(fig.layout.title.text)
    top = 84 if two_line_title else (64 if has_title else 36)

    # ---- bottom: tick block, then axis title, then legend ----
    legend_on = bool(fig.layout.showlegend)
    tick_px = 0 if horizontal else _tick_block_px(fig, scale)
    x_title = fig.layout.xaxis.title
    title_px = int(28 * scale) if (x_title and x_title.text) else 0
    legend_px = int(38 * scale) if legend_on else 0
    bottom = int(22 * scale) + tick_px + title_px + legend_px

    # ---- height ----
    # The plot area is a fixed target and the label stack is added on top of
    # it, so long account names make the figure taller rather than squashing
    # the bars into a strip.
    if horizontal:
        plot_target = max(220, min(640, 30 * n_cat + 40))
    elif n_traces >= 3 or n_cat > 10:
        plot_target = 380
    else:
        plot_target = 340

    height = min(780, top + bottom + plot_target)
    if for_slide:
        height = min(height, 540)

    fig.update_layout(height=height, margin=dict(l=80, r=56, t=top, b=bottom, pad=8))

    # ---- legend sits below the tick block, never on top of it ----
    # legend.y is a fraction of the PLOT area, not of the figure, so the
    # offset has to be converted from the pixels the ticks actually occupy.
    if legend_on:
        plot_px = max(120, height - top - bottom)
        offset = (tick_px + title_px + int(12 * scale)) / plot_px
        fig.update_layout(legend=dict(y=-offset, yanchor="top"))
