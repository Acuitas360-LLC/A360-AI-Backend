"""
chart_engine_10.py
================
Deterministic Plotly chart builder for the A360 assistant.

Why this module exists
----------------------
Previously the LLM emitted raw Plotly code and the frontend `exec`'d it. That
works for exactly one fixed chart. The moment the user is allowed to change the
chart type, the X axis, or the set of Y metrics, LLM-generated code breaks --
it was written for one specific column layout.

So the responsibility is split:

    LLM  ->  decides the BEST DEFAULT view (VIZ_CONFIG, a plain dict)
    this ->  renders ANY valid (chart_type, x, y[]) combination the user picks

Every layout rule that used to live in the prompt (no IDs on axes, no
HH:MM:SS timestamps, horizontal legend below the plot, data labels on every
point, top-10 capping, dual axis for percentages, ...) is enforced here in
Python, so it can no longer regress between model calls.

Public API
----------
    infer_viz_config(df)                    -> dict   (fallback defaults)
    parse_viz_config(code, df)              -> dict   (LLM defaults + validation)
    column_roles(df)                        -> dict
    selectable_x_columns(df)                -> list
    selectable_y_columns(df)                -> list
    eligible_chart_types(df, x_col, y_cols) -> list
    max_metrics_for(chart_type)             -> int
    build_chart(df, chart_type, x_col, y_cols, ...) -> plotly Figure
"""

from __future__ import annotations

import math
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ALL_CHART_TYPES = [
    "Bar",
    "Bar + Line",
    "Grouped Bar",
    "Stacked Bar",
    "100% Stacked Bar",
    "Line",
    "Area",
    "Scatter",
    "Pie",
    "Box",
    "Histogram",
    "Heatmap",
]

SINGLE_METRIC_CHARTS = {"Pie", "Box", "Histogram", "Heatmap"}
# Level on bars, rate/growth on a line against a second axis.
COMBO_CHARTS = {"Bar + Line"}
ADDITIVE_ONLY_CHARTS = {"Stacked Bar", "100% Stacked Bar", "Pie"}
NO_AGG_CHARTS = {"Box", "Histogram"}

# Columns that must never appear on an axis, label, legend, hover or title.
ID_EXACT = {"id", "parent_id", "child_id", "npi", "zip", "valid_order", "is_business_day"}
ID_SUFFIXES = ("_id",)

# Preferred human-readable replacement for an ID column.
ID_TO_NAME = {
    "parent_id": "parent_name",
    "child_id": "child_name",
    "campus_id": "campus_account_name",
    "campus_region_id": "campus_region",
    "campus_territory_id": "campus_territory",
}

PERCENT_TOKENS = ("pct", "percent", "_%", "share", "growth", "rate", "ratio", "_chg", "change")
AVERAGE_TOKENS = ("avg", "average", "mean", "per_day", "per_business_day")

# Words that mark a column as a MEASURE, i.e. something to plot on Y and never
# on X. Without this an integer measure on a short result set (4 regions, 12
# months) looks like a low-cardinality dimension and gets offered as an axis.
MEASURE_TOKENS = (
    "qty", "quantity", "units", "vial", "sales", "revenue", "price", "amount",
    "budget", "forecast", "target", "count", "total", "sum", "demand", "trx",
    "nrx", "volume", "dollars", "spend", "attainment", "value",
)

KNOWN_TIME_COLUMNS = {
    "transaction_date", "week_end_date", "shipment_date", "therapy_start_date",
    "date", "month_year", "quarter_year", "year",
}

# Colour-blind-safe qualitative palette.
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

MAX_PIE_SLICES = 6
DEFAULT_TOP_N = 10

# Long time series get unreadable fast: 52 weekly points means 52 rotated tick
# labels and 52 data labels per trace. Past these thresholds the axis collapses
# to one tick per month (or quarter/year) and data labels are thinned out.
DENSE_AXIS_THRESHOLD = 14      # points on a time axis before ticks are thinned
MAX_TIME_POINTS = 400          # hard cap on points kept for a time axis
MAX_DATA_LABELS = None         # None = label every point; set an int to thin
HARD_LABEL_LIMIT = 250         # safety net: never print more labels than this
MAX_AXIS_TICKS = 16            # most date ticks ever printed on one axis

# Data label sizing and spacing.
#
# Labels used to be shrunk to 8pt on a dense series so that every point could
# keep one. 8pt is not readable on screen, and the alternating above/below
# placement that bought the horizontal room made the eye bounce over and under
# the line to follow a trend. Both are reversed here: the font stays readable
# and a consistent band is used, with the number of PRINTED labels reduced
# geometrically instead so nothing collides.
LABEL_FONT_MAX = 12            # readable default
LABEL_FONT_MIN = 10            # never smaller than this, whatever the density
LABEL_CHAR_W = 0.62            # glyph width as a fraction of font size
LABEL_GAP_PX = 10              # clear space required between two labels
PLOT_WIDTH_PX = 900            # assumed plot width; Streamlit container less margins
Y_RANGE_HEADROOM = 0.12        # fraction of span added so top labels are not clipped
LABEL_COLOUR = "#333333"       # readable on white, whatever the app theme
TICK_SPACING_BUDGET = 14       # date ticks assumed to fit without overlapping
DATE_TICK_ANGLE = -45          # rotation for date ticks, which are long


# --------------------------------------------------------------------------
# Column classification
# --------------------------------------------------------------------------

def is_id_column(col: str) -> bool:
    lc = str(col).strip().lower()
    return lc in ID_EXACT or lc.endswith(ID_SUFFIXES)


def is_percent_column(col: str) -> bool:
    lc = str(col).strip().lower()
    return any(tok in lc for tok in PERCENT_TOKENS)


def is_average_column(col: str) -> bool:
    lc = str(col).strip().lower()
    return any(tok in lc for tok in AVERAGE_TOKENS)


def is_measure_column(col: str) -> bool:
    """True for columns that are quantities to be measured, never dimensions to
    measure them against. Used to keep measures out of the X-axis picker."""
    lc = str(col).strip().lower()
    if is_percent_column(lc) or is_average_column(lc):
        return True
    return any(tok in lc for tok in MEASURE_TOKENS)


GROWTH_INTENT_TOKENS = (
    "growth", "grow", "growing", "decline", "declining", "increase", "decrease",
    "change", "trend", "trending", "momentum", "yoy", "mom", "qoq", "wow",
    "year over year", "month over month", "quarter over quarter",
    "week over week", "vs last", "compared to last", "uplift", "drop",
)


def question_wants_growth(question: str) -> bool:
    """True when the user's question is about change over time rather than
    level, and therefore deserves the growth metric on the chart."""
    if not question:
        return False
    text = str(question).lower()
    return any(tok in text for tok in GROWTH_INTENT_TOKENS)


def growth_columns(df: pd.DataFrame) -> list:
    """Numeric columns that express a rate of change rather than a level."""
    if df is None or df.empty:
        return []
    return [c for c in selectable_y_columns(df) if is_percent_column(c)]


def is_additive_column(col: str) -> bool:
    """False for percentages, rates, ratios and averages -- these must not be
    summed, stacked or turned into pie slices."""
    return not (is_percent_column(col) or is_average_column(col))


def time_kind(df: pd.DataFrame, col: str):
    """Return 'date', 'month_label', 'quarter_label', 'year' or None."""
    if col is None or col not in df.columns:
        return None
    lc = str(col).strip().lower()
    if lc == "month_year":
        return "month_label"
    if lc == "quarter_year":
        return "quarter_label"
    if lc == "year":
        return "year"
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return "date"
    if lc.endswith("_date") or lc == "date" or lc in KNOWN_TIME_COLUMNS:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return "date"
        except Exception:
            return None
    return None


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Recover dtypes lost in transit.

    Snowflake results are serialised to JSON and rebuilt with
    `pd.DataFrame(data=..., columns=...)`, which frequently leaves numeric
    columns as object/str (Decimal -> str). Without this, a perfectly good
    measure would never appear in the metric picker.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    for col in out.columns:
        if is_id_column(col) or pd.api.types.is_numeric_dtype(out[col]):
            continue
        if str(col).strip().lower() in KNOWN_TIME_COLUMNS - {"year"}:
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        non_null = out[col].notna().sum()
        if non_null and converted.notna().sum() / non_null > 0.8:
            out[col] = converted
    return out


def column_roles(df: pd.DataFrame) -> dict:
    """Split the dataframe columns into the roles the UI cares about."""
    numeric, categorical, temporal, ids = [], [], [], []
    for col in df.columns:
        if is_id_column(col):
            ids.append(col)
            continue
        if time_kind(df, col):
            temporal.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "temporal": temporal,
        "ids": ids,
    }


def resolve_id_column(df: pd.DataFrame, col: str) -> str:
    """Swap an ID column for its human-readable twin when one exists."""
    if not is_id_column(col):
        return col
    twin = ID_TO_NAME.get(str(col).lower())
    if twin and twin in df.columns:
        return twin
    base = re.sub(r"_id$", "", str(col).lower())
    for candidate in (f"{base}_name", f"{base}", f"{base}_account_name"):
        if candidate in df.columns and candidate != col:
            return candidate
    return col


def pretty_label(col: str) -> str:
    """qty_sold_extended -> Qty Sold Extended;  net_sales -> Net Sales."""
    if col is None:
        return ""
    text = str(col).replace("_", " ").strip()
    fixes = {
        "pct": "%", "qty": "Qty", "yoy": "YoY", "mom": "MoM",
        "qoq": "QoQ", "wow": "WoW", "npi": "NPI", "hcp": "HCP",
        "mg": "mg", "id": "ID",
    }
    words = []
    for word in text.split():
        words.append(fixes.get(word.lower(), word.capitalize()))
    return " ".join(words)


# --------------------------------------------------------------------------
# Selectable option sets
# --------------------------------------------------------------------------

def selectable_x_columns(df: pd.DataFrame) -> list:
    """Anything that can sensibly sit on the X axis: time first, then
    categoricals, then low-cardinality numerics. Never an ID."""
    roles = column_roles(df)

    # A column holding one repeated value carries no shape, so plotting against
    # it collapses every row onto a single X position. SQL routinely stamps the
    # reporting window on every row (period_start_date, data_as_of), and those
    # sort ahead of the real time column -- which is how a 26-week trend ended
    # up drawn as one point. `selectable_y_columns` already drops constants;
    # the X axis needs the same guard.
    varies = (lambda c: df[c].nunique(dropna=True) > 1) if len(df) > 1 \
        else (lambda c: True)

    options = [c for c in roles["temporal"] if varies(c)]
    options += [c for c in roles["categorical"]
                if df[c].nunique(dropna=True) > 0 and varies(c)]

    # A numeric column only belongs on the X axis when it behaves like a
    # discrete dimension. "Few distinct values" alone is not enough: on a short
    # result set (4 regions, 12 months) EVERY integer measure has few distinct
    # values, which is how net_sales and budget ended up in the axis picker.
    #
    # A real grouping key repeats -- there are duplicate values to aggregate
    # over -- so require nunique < len(df) as well as a small cardinality, and
    # reject anything whose name marks it as a measure.
    n_rows = len(df)
    for col in roles["numeric"]:
        if is_measure_column(col):
            continue
        if not pd.api.types.is_integer_dtype(df[col]):
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_rows >= 3 and 2 <= n_unique <= 25 and n_unique < n_rows:
            options.append(col)

    if not options:                       # last resort: anything plottable
        options = [c for c in df.columns if not is_id_column(c)]
    seen, ordered = set(), []
    for c in options:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def selectable_y_columns(df: pd.DataFrame, exclude=None) -> list:
    """Numeric measures only. Drops IDs, all-null and constant columns, plus
    anything in `exclude` (normally the column already used as the X axis)."""
    exclude = set(exclude or [])
    roles = column_roles(df)
    out = []
    for col in roles["numeric"]:
        if col in exclude:
            continue
        series = df[col]
        if series.isna().all():
            continue
        if series.nunique(dropna=True) <= 1 and len(df) > 1:
            continue
        out.append(col)
    return out


def max_metrics_for(chart_type: str) -> int:
    return 1 if chart_type in SINGLE_METRIC_CHARTS else 6


def eligible_chart_types(df: pd.DataFrame, x_col: str, y_cols: list) -> list:
    """Only offer chart types that will render meaningfully for this
    (x, y[]) selection. Never let the UI build a nonsense chart."""
    if df is None or df.empty or not y_cols:
        return []

    y_cols = [c for c in y_cols if c in df.columns]
    if not y_cols:
        return []

    eligible = ["Bar"]
    n_rows = len(df)
    n_x = df[x_col].nunique(dropna=True) if x_col in df.columns else 0
    is_time = time_kind(df, x_col) is not None
    all_additive = all(is_additive_column(c) for c in y_cols)
    all_non_negative = all(pd.to_numeric(df[c], errors="coerce").fillna(0).ge(0).all() for c in y_cols)

    mixed_units = len({is_percent_column(c) for c in y_cols}) > 1
    if len(y_cols) >= 2 and mixed_units:
        # A level and a rate on one axis makes the rate invisible; the combo
        # puts bars on the left axis and the rate on a line against the right.
        eligible.append("Bar + Line")

    if len(y_cols) >= 2:
        eligible.append("Grouped Bar")
        if all_additive and all_non_negative:
            eligible += ["Stacked Bar", "100% Stacked Bar"]

    if is_time or n_x > 1:
        eligible.append("Line")
        if all_non_negative:
            eligible.append("Area")

    eligible.append("Scatter")

    if (not is_time) and 2 <= n_x <= MAX_PIE_SLICES and len(y_cols) >= 1 \
            and is_additive_column(y_cols[0]) and all_non_negative:
        eligible.append("Pie")

    if n_rows >= 20:
        eligible.append("Histogram")

    if x_col in df.columns and n_x >= 2 and n_rows <= 200_000:
        per_group = df.groupby(x_col, observed=True)[y_cols[0]].count()
        if (per_group >= 5).sum() >= 2:
            eligible.append("Box")

    if len(y_cols) >= 2 and 2 <= n_x <= 30:
        eligible.append("Heatmap")

    return [c for c in ALL_CHART_TYPES if c in eligible]


# --------------------------------------------------------------------------
# Default configuration
# --------------------------------------------------------------------------

def needs_metric_comparison(df: pd.DataFrame) -> bool:
    """True when the result is a single row carrying no dimension at all.

    "What was demand this period vs last?" returns exactly this shape:
    one row, two or more level measures, no category and no date. There is
    nothing to put on the X axis, so the measures themselves become it.
    """
    if df is None or len(df) != 1:
        return False
    roles = column_roles(df)
    if roles["categorical"] or roles["temporal"]:
        return False
    levels = [c for c in selectable_y_columns(df) if is_additive_column(c)]
    return len(levels) >= 2


def metric_comparison_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a one-row result into a plottable (metric, value) frame.

    Only level measures are carried across. Percentages are deliberately
    dropped: a growth rate of 11.9 next to a volume of 4,821 on one axis makes
    the rate invisible, and the summary above the chart already states it.
    """
    if not needs_metric_comparison(df):
        return df
    levels = [c for c in selectable_y_columns(df) if is_additive_column(c)]
    row = df.iloc[0]
    return pd.DataFrame({
        "metric": [pretty_label(c) for c in levels],
        "value": [pd.to_numeric(row[c], errors="coerce") for c in levels],
    })


def infer_viz_config(df: pd.DataFrame) -> dict:
    """Data-driven fallback used when the LLM gives us nothing usable."""
    if df is None or df.empty:
        return {"no_visualization": True, "reason": "Empty result set."}

    y_options = selectable_y_columns(df)
    x_options = selectable_x_columns(df)

    if not y_options or not x_options:
        return {
            "no_visualization": True,
            "reason": "No numeric measure and dimension pair available to plot.",
        }

    roles = column_roles(df)

    # Prefer a genuine dimension. A column that is also a valid metric is a
    # measure that merely looks discrete, and using it as the axis produces
    # charts like "Total Qty by Total Qty".
    dimensions = [c for c in x_options if c not in y_options]
    temporal = [c for c in roles["temporal"] if c in x_options]
    if temporal:
        x_col = temporal[0]
    elif dimensions:
        x_col = dimensions[0]
    else:
        x_col = x_options[0]

    y_options = [c for c in y_options if c != x_col]
    if not y_options:
        return {
            "no_visualization": True,
            "reason": "No measure available to plot against a dimension.",
        }

    base = [c for c in y_options if is_additive_column(c)]
    y_cols = [base[0]] if base else [y_options[0]]

    growth = [c for c in y_options if is_percent_column(c)]
    if growth and growth[0] not in y_cols:
        y_cols.append(growth[0])

    chart_type = "Line" if time_kind(df, x_col) else "Bar"
    eligible = eligible_chart_types(df, x_col, y_cols)
    if chart_type not in eligible:
        chart_type = eligible[0] if eligible else "Bar"

    return {
        "no_visualization": False,
        "reason": None,
        "title": f"{' and '.join(pretty_label(c) for c in y_cols)} by {pretty_label(x_col)}",
        "chart_type": chart_type,
        "x_column": x_col,
        "y_columns": y_cols,
        "top_n": DEFAULT_TOP_N,
        "labels": {},
    }


def parse_viz_config(code: str, df: pd.DataFrame, question: str = None) -> dict:
    """Extract VIZ_CONFIG from the model output and validate every field
    against the real dataframe. Anything invalid is silently repaired from
    `infer_viz_config`, so the UI can never be handed a broken config.

    `question` is the user's original question. When it asks about growth or
    change and the result set carries a growth column, that column is pulled
    onto the default chart even if the model left it out -- asking "how is X
    growing" and being shown only the level is a wrong answer, and this is too
    important to leave to the model getting it right every time.
    """
    fallback = infer_viz_config(df)

    if df is None or df.empty:
        return fallback

    text = (code or "").strip()
    if not text or "NO_VISUALIZATION" in text.upper().split("VIZ_CONFIG")[0]:
        if "VIZ_CONFIG" not in text:
            return {"no_visualization": True,
                    "reason": "The agent determined this result is not suitable for a chart."}

    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    cfg = None
    if "VIZ_CONFIG" in text:
        # A config is a plain dict literal, so parse it as data first. Only
        # fall back to exec when literal_eval cannot handle it -- executing
        # model output is a last resort, not the default path.
        import ast
        match = re.search(r"VIZ_CONFIG\s*=\s*(\{.*\})", text, re.DOTALL)
        if match:
            snippet = match.group(1)
            # The greedy match may swallow trailing prose; walk the closing
            # braces back until something parses.
            while snippet:
                try:
                    candidate = ast.literal_eval(snippet)
                    if isinstance(candidate, dict):
                        cfg = candidate
                    break
                except Exception:
                    cut = snippet.rfind("}")
                    if cut <= 0:
                        break
                    snippet = snippet[:cut]

        if cfg is None:
            scope = {"pd": pd, "df": df.copy(), "__builtins__": {}}
            try:
                exec(text, scope)                              # noqa: S102
                candidate = scope.get("VIZ_CONFIG")
                if isinstance(candidate, dict):
                    cfg = candidate
            except Exception:
                cfg = None

    if not isinstance(cfg, dict):
        return fallback

    if cfg.get("no_visualization"):
        return {"no_visualization": True,
                "reason": cfg.get("reason") or "Not suitable for a chart."}

    # ---- validate X -------------------------------------------------------
    x_col = cfg.get("x_column")
    if isinstance(x_col, str) and x_col in df.columns:
        x_col = resolve_id_column(df, x_col)
    if not isinstance(x_col, str) or x_col not in df.columns or is_id_column(x_col):
        x_col = fallback.get("x_column")

    # ---- validate Y -------------------------------------------------------
    y_cols = cfg.get("y_columns") or cfg.get("y_column") or []
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    valid_y = selectable_y_columns(df)
    y_cols = [c for c in y_cols if c in valid_y and c != x_col]
    if not y_cols:
        y_cols = [c for c in fallback.get("y_columns", valid_y[:1]) if c != x_col]
    if not y_cols:
        return fallback

    # ---- validate chart type ---------------------------------------------
    chart_type = cfg.get("chart_type")
    eligible = eligible_chart_types(df, x_col, y_cols)
    if chart_type not in eligible:
        chart_type = fallback.get("chart_type") if fallback.get("chart_type") in eligible \
            else (eligible[0] if eligible else "Bar")

    # ---- growth pairing safety net ---------------------------------------
    if question_wants_growth(question):
        growth = growth_columns(df)
        if growth and not any(is_percent_column(c) for c in y_cols):
            base = [c for c in y_cols if not is_percent_column(c)]
            if base:
                y_cols = [base[0], growth[0]]
                if chart_type in SINGLE_METRIC_CHARTS or chart_type == "Bar":
                    chart_type = "Bar + Line"
                eligible = eligible_chart_types(df, x_col, y_cols)
                if chart_type not in eligible:
                    chart_type = ("Bar + Line" if "Bar + Line" in eligible
                                  else (eligible[0] if eligible else "Bar"))

    if chart_type in SINGLE_METRIC_CHARTS:
        y_cols = y_cols[:1]

    labels = cfg.get("labels") or cfg.get("y_labels") or {}
    if not isinstance(labels, dict):
        labels = {}

    try:
        top_n = int(cfg.get("top_n", DEFAULT_TOP_N))
    except Exception:
        top_n = DEFAULT_TOP_N

    return {
        "no_visualization": False,
        "reason": None,
        "title": cfg.get("title") or fallback.get("title"),
        "chart_type": chart_type,
        "x_column": x_col,
        "y_columns": y_cols,
        "top_n": max(3, min(top_n, 50)),
        "labels": labels,
    }


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

def _agg_for(col: str) -> str:
    """How repeated X values are combined. This is decided automatically and is
    not user-configurable: absolute measures are summed, while percentages,
    rates and averages are averaged (summing them would be meaningless)."""
    if is_percent_column(col) or is_average_column(col):
        return "mean"
    return "sum"


def _format_time_axis(plot_df: pd.DataFrame, x_col: str, kind: str) -> pd.DataFrame:
    """Sort chronologically, then render as a clean label. Never HH:MM:SS."""
    if kind == "date":
        parsed = pd.to_datetime(plot_df[x_col], errors="coerce")
        plot_df = plot_df.assign(**{"__sort__": parsed}).sort_values("__sort__")
        parsed = plot_df["__sort__"]
        n_dates = parsed.dt.normalize().nunique()

        # Format by the granularity of the data, never by how long the series
        # runs. Coarsening the label coarsens the DATA -- points sharing a
        # label get grouped together, silently turning a weekly series into
        # monthly totals. Axis readability is handled by tick collapsing
        # instead, which leaves every point intact.
        month_starts = parsed.notna().any() and bool((parsed.dt.day == 1).all())
        fmt = "%b %Y" if month_starts else "%d %b %Y"
        formatted = parsed.dt.strftime(fmt)
        if formatted.nunique() < n_dates:          # collapsed distinct dates
            fmt = "%d %b %Y"
            formatted = parsed.dt.strftime(fmt)

        plot_df[x_col] = formatted
        plot_df = plot_df.drop(columns="__sort__")
    elif kind == "month_label":
        plot_df = plot_df.sort_values(x_col)
        converted = pd.to_datetime(plot_df[x_col], format="%Y-%m", errors="coerce")
        if converted.notna().mean() > 0.8:
            plot_df[x_col] = converted.dt.strftime("%b %Y")
    elif kind in ("quarter_label", "year"):
        plot_df = plot_df.sort_values(x_col)
        plot_df[x_col] = plot_df[x_col].astype(str)
    return plot_df


def prepare_plot_df(df, x_col, y_cols, top_n=DEFAULT_TOP_N, chart_type="Bar"):
    """Aggregate to one row per X value, cap categories, and order the axis."""
    plot_df = df.copy()

    x_col = resolve_id_column(plot_df, x_col)
    y_cols = [c for c in y_cols if c in plot_df.columns and c != x_col]
    if x_col not in plot_df.columns or not y_cols:
        return plot_df, x_col, y_cols

    for col in y_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.dropna(subset=[x_col], how="all")
    kind = time_kind(plot_df, x_col)

    if chart_type in NO_AGG_CHARTS:
        if kind:
            plot_df = _format_time_axis(plot_df, x_col, kind)
        else:
            plot_df[x_col] = plot_df[x_col].fillna("Unknown").astype(str)
        return plot_df, x_col, y_cols

    if kind:
        plot_df = _format_time_axis(plot_df, x_col, kind)
        order = list(dict.fromkeys(plot_df[x_col].tolist()))
    else:
        plot_df[x_col] = plot_df[x_col].fillna("Unknown").astype(str)
        order = None

    agg_map = {col: _agg_for(col) for col in y_cols}
    grouped = plot_df.groupby(x_col, as_index=False, observed=True).agg(agg_map)

    if order is not None:
        grouped[x_col] = pd.Categorical(grouped[x_col], categories=order, ordered=True)
        grouped = grouped.sort_values(x_col)
        grouped[x_col] = grouped[x_col].astype(str)
        if len(grouped) > MAX_TIME_POINTS:
            grouped = grouped.tail(MAX_TIME_POINTS)
    else:
        grouped = grouped.sort_values(y_cols[0], ascending=False)
        if top_n and len(grouped) > top_n:
            grouped = grouped.head(top_n)

    grouped = grouped.drop_duplicates(subset=[x_col])
    return grouped, x_col, y_cols


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

def _value_fmt(col: str) -> str:
    return "%{y:.1f}%" if is_percent_column(col) else "%{y:,.0f}"


def _fmt_value(value, as_percent: bool) -> str:
    """Render one data label. Returns "" for missing values so the point is
    plotted without a stray label."""
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{value:,.1f}%" if as_percent else f"{value:,.0f}"
    except (TypeError, ValueError):
        return ""


def _label_positions(n: int, max_labels=MAX_DATA_LABELS) -> list:
    """Which points keep a visible data label.

    Every point is labelled by default. Readability on dense series comes from
    smaller text and alternating label positions rather than from dropping
    values. `HARD_LABEL_LIMIT` is only a safety net so a 400-point daily series
    cannot render thousands of labels; set `max_labels` to force thinning.
    """
    limit = max_labels if max_labels is not None else HARD_LABEL_LIMIT
    if n <= limit:
        return [True] * n
    stride = -(-n // limit)                         # ceil division
    keep = [i % stride == 0 for i in range(n)]
    keep[0] = True
    keep[-1] = True
    return keep


def _label_font_size(n: int) -> int:
    """Data label size.

    Density is handled by printing fewer labels, not by shrinking them, so this
    never drops below `LABEL_FONT_MIN`. A label the user cannot read is worth
    the same as no label at all.
    """
    if n > 40:
        return LABEL_FONT_MIN
    if n > 20:
        return LABEL_FONT_MAX - 1
    return LABEL_FONT_MAX


def _turning_points(values):
    """Classify each point as a local peak, a local trough, or on a slope, and
    score how sharply it turns.

    Returns {index: ("peak"|"trough", prominence)}. Prominence is the smaller
    of the two step heights either side, so a lone spike scores high and a
    point in a gentle drift scores near zero.
    """
    nums = [None if v is None or pd.isna(v) else float(v) for v in values]
    n = len(nums)
    out = {}
    for i, v in enumerate(nums):
        if v is None:
            continue
        prev = next((nums[j] for j in range(i - 1, -1, -1) if nums[j] is not None), None)
        nxt = next((nums[j] for j in range(i + 1, n) if nums[j] is not None), None)
        if prev is None or nxt is None:
            continue
        if v >= prev and v >= nxt:
            out[i] = ("peak", min(v - prev, v - nxt))
        elif v <= prev and v <= nxt:
            out[i] = ("trough", min(prev - v, nxt - v))
    return out


def _line_bands(values, trace_index=0):
    """Put a label on the outside of the curve: above a peak, below a trough.

    A single fixed band drops labels straight onto the line wherever the series
    is climbing. Following the shape keeps the text clear of the stroke and
    makes a spike read as a spike.
    """
    turns = _turning_points(values)
    default = "top center" if trace_index % 2 == 0 else "bottom center"
    return [("bottom center" if turns.get(i, ("", 0))[0] == "trough"
             else "top center" if turns.get(i, ("", 0))[0] == "peak"
             else default)
            for i in range(len(values))]


def _label_importance(values) -> list:
    """The order in which points should be offered a label.

    Most recent first, then the start, then every turning point sharpest-first,
    then whatever is left to fill the gaps. A stride grid was labelling every
    Nth point regardless of shape, which is why major peaks went bare while
    unremarkable mid-slope points kept a label.
    """
    n = len(values)
    turns = _turning_points(values)
    order = []
    for i in (n - 1, 0):
        if 0 <= i < n and i not in order:
            order.append(i)
    for i in sorted((i for i, (_, p) in turns.items() if p > 0),
                    key=lambda i: -turns[i][1]):
        if i not in order:
            order.append(i)
    order += [i for i in range(n) if i not in order]
    return order


def _label_keep(texts, values, font_px, slots, bands=None,
                plot_width_px=PLOT_WIDTH_PX) -> list:
    """Which points keep a printed label.

    Points are offered a label in importance order and accepted only when the
    text box clears every label already placed ON THE SAME BAND. Measuring the
    real box width per label -- rather than applying one stride to the whole
    series -- means a narrow "89" and a wide "-12.3%" are spaced on their own
    terms, and a peak sitting directly above a trough is not a conflict at all
    because the two sit on opposite sides of the line.
    """
    n = len(texts)
    if n <= 1:
        return [True] * n

    step = plot_width_px / max(slots, 1)
    half = [len(t) * font_px * LABEL_CHAR_W / 2 for t in texts]
    bands = bands or ["single"] * n

    placed = []

    def clears(i):
        for j in placed:
            if bands[j] != bands[i]:
                continue
            if abs(i - j) * step < half[i] + half[j] + LABEL_GAP_PX:
                return False
        return True

    for i in _label_importance(values):
        if texts[i] and clears(i):
            placed.append(i)

    keep = [False] * n
    for i in placed:
        keep[i] = True
    return keep


def _pad_y_range(fig, series, secondary=None):
    """Add headroom so a label on the highest point is not clipped by the
    plot edge or pushed into the title."""
    values = pd.concat([pd.to_numeric(s, errors="coerce") for s in series]).dropna()
    if values.empty:
        return
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    pad = (span * Y_RANGE_HEADROOM) if span else (abs(hi) * Y_RANGE_HEADROOM or 1.0)
    lo = min(lo, 0.0) if lo >= 0 and lo - pad < 0 else lo - pad
    kwargs = {"range": [lo, hi + pad]}
    if secondary is None:
        fig.update_yaxes(**kwargs)
    else:
        fig.update_yaxes(secondary_y=secondary, **kwargs)


def _text_positions(n: int, trace_index: int, chart_type: str):
    """Where each data label sits.

    On dense line/scatter series the labels alternate above and below the line,
    which doubles the horizontal room available to each one. The alternation is
    offset per trace so labels from different series do not line up.
    """
    if chart_type in ("Stacked Bar", "100% Stacked Bar"):
        return "inside"
    if chart_type in ("Bar", "Grouped Bar"):
        return "outside"
    # One consistent band per series. Alternating point by point doubled the
    # horizontal room but made the labels unreadable as a sequence -- the eye
    # has to zig-zag across the line to follow the values. Labels are thinned
    # by `_label_keep` instead, so a single band has room.
    return "top center" if trace_index % 2 == 0 else "bottom center"


def _series_text(values, as_percent: bool, keep: list) -> list:
    return [_fmt_value(v, as_percent) if k else ""
            for v, k in zip(values, keep)]


def _time_axis_ticks(labels: list):
    """Thin a dense date axis by printing every Nth date.

    The dates themselves are kept -- a weekly series shows real week dates such
    as "05 Jul 2025", never a rolled-up "Jul 2025" month label. Only the number
    of printed ticks is reduced, by sampling at an even stride so consecutive
    ticks are always the same distance apart and cannot collide.
    """
    n = len(labels)
    if n == 0:
        return None
    if n <= DENSE_AXIS_THRESHOLD:
        return None

    stride = max(1, -(-n // min(MAX_AXIS_TICKS, TICK_SPACING_BUDGET)))
    idx = list(range(0, n, stride))

    # Always finish on the most recent point; if that would crowd the previous
    # tick, replace it rather than squeezing an extra label in.
    last = n - 1
    if idx[-1] != last:
        if last - idx[-1] >= stride:
            idx.append(last)
        else:
            idx[-1] = last

    vals = [labels[i] for i in idx]
    return vals, list(vals)


def _apply_time_ticks(fig, labels: list) -> bool:
    """Print a readable subset of the date ticks. Returns True when applied so
    the caller can skip its own rotation fallback."""
    ticks = _time_axis_ticks(labels)
    if not ticks or len(ticks[0]) < 2:
        return False
    fig.update_xaxes(tickmode="array", tickvals=ticks[0], ticktext=ticks[1],
                     tickangle=DATE_TICK_ANGLE)
    return True


def _label_for(col: str, labels: dict) -> str:
    if labels and col in labels:
        return labels[col]
    return pretty_label(col)


def _base_layout(fig, title, x_label, height, n_traces):
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.97, yanchor="top",
                   font=dict(size=17)),
        height=height,
        margin=dict(t=90, b=130, l=70, r=70, pad=8),
        hovermode="closest",
        hoverlabel=dict(namelength=-1),
        legend=dict(orientation="h", yanchor="top", y=-0.22,
                    xanchor="center", x=0.5),
        showlegend=n_traces > 1,
        colorway=PALETTE,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        uniformtext=dict(mode="show", minsize=LABEL_FONT_MIN),
    )
    fig.update_xaxes(title_text=x_label, title_standoff=20,
                     showgrid=False, automargin=True)
    fig.update_yaxes(title_standoff=20, showgrid=True,
                     gridcolor="rgba(0,0,0,0.07)", automargin=True)
    return fig


def _height_for(chart_type, n_traces, n_points):
    height = 520
    if n_traces >= 3:
        height = 650
    if chart_type in ("Box", "Heatmap"):
        height = max(height, 600)
    if n_points > 15:
        height = max(height, 580)
    # Labelling every point on a dense series needs vertical room for the
    # alternating above/below placement.
    if n_points > 30:
        height = max(height, 620 + 40 * (n_traces - 1))
    return height


# --------------------------------------------------------------------------
# Chart builders
# --------------------------------------------------------------------------

def build_chart(df, chart_type="Bar", x_col=None, y_cols=None,
                title=None, labels=None,
                top_n=DEFAULT_TOP_N, show_labels=True):
    """Build any supported chart for any valid (x, y[]) selection.

    Always returns a Figure -- never raises for a merely awkward selection.
    """
    labels = labels or {}
    y_cols = [c for c in (y_cols or []) if c in df.columns]

    if df is None or df.empty or not y_cols or not x_col or x_col not in df.columns:
        return _empty_figure("Select an X axis and at least one metric.")

    # A column cannot be both the axis and a measure -- the axis gets cast to
    # string for grouping, which would then break the aggregation.
    y_cols = [c for c in y_cols if c != x_col]
    if not y_cols:
        return _empty_figure("Choose at least one metric other than the X axis column.")

    if chart_type in SINGLE_METRIC_CHARTS:
        y_cols = y_cols[:1]

    x_is_time = time_kind(df, x_col) is not None

    plot_df, x_col, y_cols = prepare_plot_df(
        df, x_col, y_cols, top_n=top_n, chart_type=chart_type
    )
    if plot_df.empty or not y_cols:
        return _empty_figure("No plottable rows after aggregation.")

    x_label = _label_for(x_col, labels)
    if not title:
        metric_names = " and ".join(_label_for(c, labels) for c in y_cols)
        title = f"{metric_names} by {x_label}"

    builders = {
        "Pie": _build_pie,
        "Histogram": _build_histogram,
        "Box": _build_box,
        "Heatmap": _build_heatmap,
    }
    if chart_type in builders:
        return builders[chart_type](plot_df, x_col, y_cols, title, x_label,
                                    labels, show_labels, x_is_time)

    return _build_xy(plot_df, x_col, y_cols, chart_type, title, x_label,
                     labels, show_labels, x_is_time)


def _empty_figure(message: str):
    fig = go.Figure()
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=message, showarrow=False,
                          x=0.5, y=0.5, xref="paper", yref="paper",
                          font=dict(size=14, color="#888"))],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_xy(plot_df, x_col, y_cols, chart_type, title, x_label, labels,
              show_labels, x_is_time=False):
    """Bar / Grouped Bar / Stacked Bar / 100% Stacked Bar / Line / Area / Scatter."""
    percent_flags = {is_percent_column(c) for c in y_cols}
    dual_axis = len(percent_flags) > 1 and len(y_cols) > 1
    combo = chart_type in COMBO_CHARTS

    if chart_type == "100% Stacked Bar":
        plot_df = plot_df.copy()
        for col in y_cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0)
        totals = plot_df[y_cols].sum(axis=1).replace(0, pd.NA)
        for col in y_cols:
            plot_df[col] = (plot_df[col] / totals * 100).fillna(0)
        dual_axis = False

    fig = make_subplots(specs=[[{"secondary_y": True}]]) if dual_axis else go.Figure()

    # Thin the data labels on dense series so they stop colliding. Every point
    # is still plotted and still available on hover -- only the printed labels
    # are reduced.
    n_points = len(plot_df)
    label_size = _label_font_size(n_points)

    # Bars share the category width between series, so a grouped bar has one
    # label slot per series per category; a line has one per point.
    bar_series = sum(1 for c in y_cols
                     if not (combo and is_percent_column(c))) if chart_type != "Line" else 1
    bar_slots = n_points * max(bar_series, 1)

    # One palette entry per metric, fixed up front. Deriving the colour from
    # the loop index inside each branch used to hand the same colour to a line
    # and a bar whenever the % metric was ordered before the level metric.
    colour = {col: PALETTE[i % len(PALETTE)] for i, col in enumerate(y_cols)}

    for i, col in enumerate(y_cols):
        name = _label_for(col, labels)
        is_pct = is_percent_column(col) or chart_type == "100% Stacked Bar"
        value_fmt = "%{y:.1f}%" if is_pct else "%{y:,.0f}"

        common = dict(
            x=plot_df[x_col],
            y=plot_df[col],
            name=name,
            cliponaxis=False,
            hovertemplate=f"{x_label}=%{{x}}<br>{name}={value_fmt}<extra></extra>",
        )
        values = plot_df[col].tolist()
        is_bar_trace = (chart_type in ("Bar", "Grouped Bar", "Stacked Bar",
                                       "100% Stacked Bar")
                        or (combo and not is_pct))

        # Line-like traces follow the shape of the curve; bars and the combo
        # rate line sit on one fixed band.
        if is_bar_trace or (combo and is_pct):
            positions = _text_positions(n_points, i, chart_type)
            bands = None
        else:
            positions = _line_bands(values, i)
            bands = positions

        if show_labels:
            texts = [_fmt_value(v, is_pct) for v in values]
            slots = bar_slots if is_bar_trace else n_points
            keep = _label_keep(texts, values, label_size, slots, bands=bands)
            common["text"] = [t if k else "" for t, k in zip(texts, keep)]
            # Explicit colour: Plotly otherwise inherits a tint that renders
            # washed-out against a white plot, which is half of "hard to read".
            common["textfont"] = dict(size=label_size, color=LABEL_COLOUR)

        if combo:
            # Rates ride the secondary axis as a line; levels stay as bars.
            if is_pct:
                trace = go.Scatter(
                    mode="lines+markers+text" if show_labels else "lines+markers",
                    line=dict(width=3 if n_points <= 30 else 2, simplify=False,
                              color=colour[col]),
                    marker=dict(size=7 if n_points <= 30 else 4),
                    # Always above the line in combo mode: the bars fill the
                    # lower half of the plot, so a label below the rate line
                    # lands on top of them.
                    textposition="top center",
                    **common,
                )
            else:
                trace = go.Bar(
                    marker_color=colour[col],
                    textposition="outside",
                    **common,
                )
        elif chart_type in ("Bar", "Grouped Bar", "Stacked Bar", "100% Stacked Bar"):
            trace = go.Bar(
                marker_color=colour[col],
                textposition=positions,
                **common,
            )
        elif chart_type == "Scatter":
            trace = go.Scatter(
                mode="markers+text" if show_labels else "markers",
                marker=dict(size=6 if n_points > 30 else 10,
                            color=colour[col]),
                textposition=positions,
                **common,
            )
        else:  # Line / Area
            trace = go.Scatter(
                mode="lines+markers+text" if show_labels else "lines+markers",
                fill="tozeroy" if chart_type == "Area" and len(y_cols) == 1 else
                     ("tonexty" if chart_type == "Area" and i > 0 else None),
                line=dict(width=2 if n_points > 30 else 2.5, simplify=False,
                          color=colour[col]),
                marker=dict(size=4 if n_points > 30 else 7),
                textposition=positions,
                **common,
            )

        if dual_axis:
            fig.add_trace(trace, secondary_y=is_percent_column(col))
        else:
            fig.add_trace(trace)

    height = _height_for(chart_type, len(y_cols), len(plot_df))
    _base_layout(fig, title, x_label, height, len(y_cols))

    if chart_type in ("Stacked Bar", "100% Stacked Bar"):
        fig.update_layout(barmode="stack")
    elif chart_type in ("Bar", "Grouped Bar", "Bar + Line"):
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)

    fig.update_xaxes(type="category")

    if dual_axis:
        abs_cols = [c for c in y_cols if not is_percent_column(c)]
        pct_cols = [c for c in y_cols if is_percent_column(c)]
        # Tint each axis to match its series, so which axis reads which line is
        # obvious without cross-referencing the legend.
        left_colour = colour[abs_cols[0]]
        right_colour = colour[pct_cols[0]]
        fig.update_yaxes(title_text=" / ".join(_label_for(c, labels) for c in abs_cols),
                         title_standoff=20, secondary_y=False,
                         title_font=dict(color=left_colour),
                         tickfont=dict(color=left_colour))
        fig.update_yaxes(title_text=" / ".join(_label_for(c, labels) for c in pct_cols),
                         title_standoff=20, secondary_y=True,
                         showgrid=False, ticksuffix="%", zeroline=True,
                         zerolinecolor="rgba(0,0,0,0.15)",
                         title_font=dict(color=right_colour),
                         tickfont=dict(color=right_colour))
    else:
        y_title = " / ".join(_label_for(c, labels) for c in y_cols)
        fig.update_yaxes(title_text=y_title, title_standoff=20)
        if chart_type == "100% Stacked Bar" or all(is_percent_column(c) for c in y_cols):
            fig.update_yaxes(ticksuffix="%")

    # Give the top label room rather than letting it ride into the title.
    if show_labels:
        if dual_axis:
            abs_cols = [c for c in y_cols if not is_percent_column(c)]
            pct_cols = [c for c in y_cols if is_percent_column(c)]
            if abs_cols:
                _pad_y_range(fig, [plot_df[c] for c in abs_cols], secondary=False)
            if pct_cols:
                _pad_y_range(fig, [plot_df[c] for c in pct_cols], secondary=True)
        elif chart_type not in ("Stacked Bar", "100% Stacked Bar"):
            _pad_y_range(fig, [plot_df[c] for c in y_cols])

    collapsed = _apply_time_ticks(fig, plot_df[x_col].tolist()) if x_is_time else False
    if not collapsed and len(plot_df) > 8:
        fig.update_xaxes(tickangle=-35)

    return fig


def _build_pie(plot_df, x_col, y_cols, title, x_label, labels, show_labels,
               x_is_time=False):
    col = y_cols[0]
    name = _label_for(col, labels)
    data = plot_df[[x_col, col]].copy()
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    data = data[data[col] > 0].sort_values(col, ascending=False)
    if data.empty:
        return _empty_figure("No positive values available for a pie chart.")

    if len(data) > MAX_PIE_SLICES:
        head = data.head(MAX_PIE_SLICES - 1)
        other = pd.DataFrame({x_col: ["Other"], col: [data[col].iloc[MAX_PIE_SLICES - 1:].sum()]})
        data = pd.concat([head, other], ignore_index=True)

    fig = go.Figure(go.Pie(
        labels=data[x_col],
        values=data[col],
        textinfo="label+percent",
        textposition="auto",
        insidetextorientation="horizontal",
        marker=dict(colors=PALETTE[: len(data)]),
        hovertemplate=f"{x_label}=%{{label}}<br>{name}=%{{value:,.0f}}"
                      f"<br>Share=%{{percent}}<extra></extra>",
        sort=False,
    ))
    _base_layout(fig, title, "", 540, 1)
    fig.update_layout(showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _build_histogram(plot_df, x_col, y_cols, title, x_label, labels, show_labels,
                     x_is_time=False):
    col = y_cols[0]
    name = _label_for(col, labels)
    values = pd.to_numeric(plot_df[col], errors="coerce").dropna()

    fig = go.Figure(go.Histogram(
        x=values,
        nbinsx=min(30, max(8, int(len(values) ** 0.5))),
        marker_color=PALETTE[0],
        hovertemplate=f"{name}=%{{x}}<br>Count=%{{y}}<extra></extra>",
    ))
    _base_layout(fig, f"Distribution of {name}", name, 520, 1)
    fig.update_yaxes(title_text="Count", title_standoff=20)
    return fig


def _build_box(plot_df, x_col, y_cols, title, x_label, labels, show_labels,
               x_is_time=False):
    col = y_cols[0]
    name = _label_for(col, labels)
    data = plot_df[[x_col, col]].copy()
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[col])

    keep = (data.groupby(x_col, observed=True)[col].median()
                .sort_values(ascending=False).head(DEFAULT_TOP_N).index)
    data = data[data[x_col].isin(keep)]

    fig = go.Figure()
    for i, group in enumerate(keep):
        subset = data[data[x_col] == group]
        fig.add_trace(go.Box(
            y=subset[col],
            name=str(group),
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean=True,
            hovertemplate=f"{x_label}={group}<br>{name}=%{{y:,.0f}}<extra></extra>",
        ))
    _base_layout(fig, f"{name} distribution by {x_label}", x_label, 600, 1)
    fig.update_yaxes(title_text=name, title_standoff=20)
    fig.update_layout(showlegend=False)
    return fig


def _build_heatmap(plot_df, x_col, y_cols, title, x_label, labels, show_labels,
                   x_is_time=False):
    """Metrics x categories matrix, normalised per metric so different units
    stay comparable. Raw values remain visible in the cell labels."""
    matrix, raw, row_labels = [], [], []
    for col in y_cols:
        values = pd.to_numeric(plot_df[col], errors="coerce").fillna(0)
        span = values.max() - values.min()
        scaled = (values - values.min()) / span if span else values * 0
        matrix.append(scaled.tolist())
        raw.append(values.tolist())
        row_labels.append(_label_for(col, labels))

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=plot_df[x_col].astype(str),
        y=row_labels,
        customdata=raw,
        colorscale="Blues",
        showscale=False,
        texttemplate="%{customdata:,.0f}" if show_labels else None,
        textfont=dict(size=10),
        hovertemplate=f"{x_label}=%{{x}}<br>%{{y}}=%{{customdata:,.0f}}<extra></extra>",
    ))
    _base_layout(fig, title, x_label, 600, 1)
    fig.update_yaxes(title_text="Metric", title_standoff=20, showgrid=False)
    fig.update_layout(showlegend=False)
    collapsed = _apply_time_ticks(fig, plot_df[x_col].astype(str).tolist()) if x_is_time else False
    if not collapsed and len(plot_df) > 8:
        fig.update_xaxes(tickangle=-35)
    return fig


# --------------------------------------------------------------------------
# Slide export
# --------------------------------------------------------------------------

# Plotly font sizes are pixels on the canvas. A chart styled for a ~1000px
# browser view and then exported and shrunk into a slide panel ends up with
# 3-4pt text. For slides the sizes are therefore specified in POINTS and
# converted using the export DPI, so what lands on the slide is predictable
# regardless of the panel size.
SLIDE_EXPORT_DPI = 200
SLIDE_FONT_PT = {
    "title": 13.0,
    "axis_title": 9.5,
    "tick": 9.0,
    "legend": 9.0,
    "label": 8.5,
}
SLIDE_MAX_DATA_LABELS = 12      # printed labels per trace on a slide
SLIDE_MAX_CATEGORIES = 12       # bars/points before the axis is unreadable
SLIDE_MAX_TICK_CHARS = 18       # long category names are truncated on slides
SLIDE_TITLE_LINE_CHARS = 58     # title wraps at this width, max 2 lines
SLIDE_MAX_AXIS_TITLE = 40       # a y-axis title longer than this is trimmed


def _shorten(text, limit):
    """Trim a label to `limit` characters, keeping whole words where possible."""
    text = str(text)
    if len(text) <= limit:
        return text
    cut = text[:max(1, limit - 1)].rstrip()
    if " " in cut and len(cut) - cut.rfind(" ") < 6:
        cut = cut[:cut.rfind(" ")].rstrip()
    return cut + "\u2026"


def _wrap_title(text, width=SLIDE_TITLE_LINE_CHARS, max_lines=2):
    """Wrap a chart title onto at most `max_lines` lines, truncating beyond."""
    text = str(text).strip()
    if len(text) <= width:
        return text
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
    if len(lines) < max_lines:
        lines.append(current)
    else:
        lines[-1] = _shorten(lines[-1] + " " + current, width)
    # A single unbroken token never wraps, so enforce the width per line too.
    return "<br>".join(_shorten(line, width) for line in lines[:max_lines])


def _thin_trace_text(trace, max_labels):
    """Blank out some data labels on a trace that is too dense for a slide."""
    text = list(getattr(trace, "text", None) or [])
    if not text:
        return
    filled = [i for i, v in enumerate(text) if v not in (None, "")]
    if len(filled) <= max_labels:
        return
    stride = -(-len(filled) // max_labels)
    keep = set(filled[::stride]) | {filled[0], filled[-1]}
    trace.text = [v if i in keep else "" for i, v in enumerate(text)]


def _slide_tick_geometry(ticks, tick_px, plot_width_px):
    """Work out the shallowest tick rotation that still avoids collisions.

    Rotating 45 degrees is safe but wastes vertical space -- on a short, wide
    slide panel that is exactly what squeezes the plot flat. Parallel rotated
    labels clear each other once the perpendicular gap between their baselines
    exceeds the text height, which gives the minimum angle directly.

    Returns (angle, vertical_px_needed).
    """
    if not ticks:
        return 0, 0

    longest = max(len(str(t)) for t in ticks)
    label_px = longest * tick_px * 0.55
    slot = plot_width_px / max(1, len(ticks))

    if label_px <= slot * 0.92:
        return 0, tick_px * 1.6                       # horizontal labels fit

    need_sin = min(0.95, (tick_px * 1.25) / slot)
    angle = max(20.0, math.degrees(math.asin(need_sin)))
    angle = min(45.0, angle)
    return -angle, label_px * math.sin(math.radians(angle)) + tick_px


def _current_ticks(fig):
    """The tick labels an x axis will actually render."""
    axis = getattr(fig.layout, "xaxis", None)
    if axis is None:
        return []
    if axis.ticktext:
        return [str(t) for t in axis.ticktext]
    for trace in fig.data:
        xs = getattr(trace, "x", None)
        if xs is not None and len(xs):
            return list(dict.fromkeys(str(v) for v in xs))
    return []


def _truncate_category_ticks(fig):
    """Shorten long categorical tick labels on a slide-bound figure.

    Left alone when the axis already carries explicit ticks (a thinned date
    axis) or when every label is short enough to render as-is.
    """
    axis = getattr(fig.layout, "xaxis", None)
    if axis is None or axis.tickmode == "array":
        return

    values = []
    for trace in fig.data:
        xs = getattr(trace, "x", None)
        if xs is not None and len(xs):
            values = [str(v) for v in xs]
            break
    if not values or not any(len(v) > SLIDE_MAX_TICK_CHARS for v in values):
        return

    ordered = list(dict.fromkeys(values))
    fig.update_xaxes(
        tickmode="array",
        tickvals=ordered,
        ticktext=[_shorten(v, SLIDE_MAX_TICK_CHARS) for v in ordered],
    )


def style_for_slide(fig, width_in, height_in, dpi=SLIDE_EXPORT_DPI,
                    max_labels=SLIDE_MAX_DATA_LABELS):
    """Restyle an on-screen figure so it stays legible inside a slide panel.

    `width_in` / `height_in` are the dimensions of the picture box on the
    slide. Rendering at exactly that aspect ratio means the image can fill the
    box without being stretched, and sizing the fonts from the same numbers
    puts real 9-13pt text on the slide instead of 3-4pt.

    Mutates and returns `fig`.
    """
    if fig is None:
        return None

    def px(points):
        return max(6, int(round(points / 72.0 * dpi)))

    width = int(round(width_in * dpi))
    height = int(round(height_in * dpi))

    has_legend = bool(fig.layout.showlegend)

    # Long category names are the main hazard here. A 44-character account name
    # rotated 35 degrees is ~500px wide, and Plotly's automargin will happily
    # surrender a quarter of the canvas to it, collapsing the plot area. Trim
    # the tick text before that can happen.
    _truncate_category_ticks(fig)

    tick_px = px(SLIDE_FONT_PT["tick"])
    title_px = px(SLIDE_FONT_PT["title"])
    axis_title_px = px(SLIDE_FONT_PT["axis_title"])

    # Size the margins from what the labels actually need, rather than a fixed
    # fraction. A flat 30% bottom on a short, wide panel leaves the plot area
    # squeezed to a sliver.
    left = int(width * 0.09)
    right = int(width * 0.04)
    ticks = _current_ticks(fig)
    angle, tick_band = _slide_tick_geometry(ticks, tick_px, width - left - right)

    title_text = fig.layout.title.text if fig.layout.title else None
    wrapped_title = _wrap_title(title_text) if title_text else None
    title_lines = (wrapped_title.count("<br>") + 1) if wrapped_title else 0

    # Lay the top band out in pixels. `title.y` anchors the CAP LINE, but the
    # line box extends roughly a third of the font size above it, so anchoring
    # near 1.0 pushes the first line off the top of the canvas. Reserve a real
    # gap instead of trusting a fraction.
    title_gap = int(title_px * 0.9) if title_lines else 0
    title_block = int(title_lines * title_px * 1.35)
    legend_block = int(px(SLIDE_FONT_PT["legend"]) * 2.4) if has_legend else 0

    top = title_gap + title_block + legend_block + int(tick_px * 0.4)
    bottom = int(tick_band + axis_title_px * 2.2)

    # Never let the furniture take more than it has to; the plot needs room.
    top = min(top, int(height * 0.38))
    bottom = min(bottom, int(height * 0.40))
    title_y = 1.0 - (title_gap / float(height))

    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=tick_px),
        margin=dict(t=top, b=bottom, l=left, r=right, pad=px(2)),
        uniformtext=dict(mode="show", minsize=px(SLIDE_FONT_PT["label"]) - 4),
    )
    fig.update_xaxes(tickangle=angle)

    if wrapped_title:
        fig.update_layout(title=dict(text=wrapped_title,
                                     font=dict(size=title_px),
                                     x=0.0, xanchor="left", xref="paper",
                                     y=title_y, yanchor="top", yref="container"))
    if has_legend:
        fig.update_layout(legend=dict(font=dict(size=px(SLIDE_FONT_PT["legend"])),
                                      orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="left", x=0.0))

    fig.update_xaxes(title_font=dict(size=px(SLIDE_FONT_PT["axis_title"])),
                     tickfont=dict(size=px(SLIDE_FONT_PT["tick"])),
                     title_standoff=px(6))
    fig.update_yaxes(title_font=dict(size=px(SLIDE_FONT_PT["axis_title"])),
                     tickfont=dict(size=px(SLIDE_FONT_PT["tick"])),
                     title_standoff=px(6))

    # "A / B / C" stacked metric names overflow the axis; the legend already
    # names every series, so trim it rather than let it push the plot inward.
    for axis in ("yaxis", "yaxis2"):
        layout_axis = getattr(fig.layout, axis, None)
        if layout_axis is None or not getattr(layout_axis, "title", None):
            continue
        current = layout_axis.title.text
        if current and len(current) > SLIDE_MAX_AXIS_TITLE:
            layout_axis.title.text = _shorten(current, SLIDE_MAX_AXIS_TITLE)

    label_px = px(SLIDE_FONT_PT["label"])

    def _restyle(trace):
        try:
            trace.textfont = dict(size=label_px)
        except Exception:
            pass
        _thin_trace_text(trace, max_labels)
        # Markers and lines were sized for a 1000px canvas.
        try:
            if trace.marker is not None and getattr(trace.marker, "size", None):
                trace.marker.size = max(4, int(trace.marker.size * dpi / 100))
        except Exception:
            pass
        try:
            if getattr(trace, "line", None) is not None and trace.line.width:
                trace.line.width = max(1, trace.line.width * dpi / 100)
        except Exception:
            pass

    fig.for_each_trace(_restyle)
    return fig


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

FILTER_MAX_UNIQUE = 60      # a column with more distinct values isn't filterable
FILTER_MIN_UNIQUE = 2       # nothing to choose from below this


def filterable_columns(df: pd.DataFrame) -> list:
    """Describe the columns a user can filter on.

    Returns a list of descriptors, time columns first:

        {"column": "week_end_date", "label": "Week End Date",
         "kind": "date", "values": ["21 Jun 2025", ...]}
        {"column": "region", "label": "Region",
         "kind": "category", "values": ["Central", "East", ...]}

    `values` for a date column are the formatted labels in chronological order,
    so a range picker snaps to real weeks rather than arbitrary calendar days.
    Measures are deliberately excluded -- filtering a chart by its own metric
    is a good way to draw a misleading picture.
    """
    if df is None or df.empty:
        return []

    out = []
    roles = column_roles(df)

    for col in roles["temporal"]:
        series = df[col].dropna()
        if series.empty:
            continue
        kind = time_kind(df, col)
        if kind == "date":
            ordered = pd.to_datetime(series, errors="coerce").dropna().sort_values().unique()
            values = [pd.Timestamp(v).strftime("%d %b %Y") for v in ordered]
        else:
            values = [str(v) for v in sorted(series.astype(str).unique())]
        if len(values) >= FILTER_MIN_UNIQUE:
            out.append({"column": col, "label": pretty_label(col),
                        "kind": "date", "values": values})

    for col in roles["categorical"]:
        values = df[col].dropna().astype(str).unique().tolist()
        if FILTER_MIN_UNIQUE <= len(values) <= FILTER_MAX_UNIQUE:
            out.append({"column": col, "label": pretty_label(col),
                        "kind": "category", "values": sorted(values)})

    return out


def _to_timestamp(value):
    try:
        return pd.to_datetime(value, errors="coerce", format="mixed")
    except Exception:
        return pd.NaT


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply a filter spec to the dataframe.

    `filters` maps a column name to either

        {"kind": "category", "values": ["East", "West"]}
        {"kind": "date", "start": "21 Jun 2025", "end": "13 Dec 2025"}

    Unknown columns and malformed entries are ignored rather than raising, so a
    stale filter saved against an older result set can never break a chart.
    Everything is plain strings, which keeps the spec JSON-serialisable and
    therefore able to travel with the conversation into the deck builder.
    """
    if df is None or df.empty or not filters:
        return df

    out = df
    for col, spec in (filters or {}).items():
        if col not in out.columns or not isinstance(spec, dict):
            continue

        kind = spec.get("kind")

        if kind == "category":
            values = spec.get("values")
            if not values:
                continue
            wanted = {str(v) for v in values}
            out = out[out[col].astype(str).isin(wanted)]

        elif kind == "date":
            start, end = spec.get("start"), spec.get("end")
            if not start and not end:
                continue
            if time_kind(out, col) == "date":
                as_dt = pd.to_datetime(out[col], errors="coerce")
                lo, hi = _to_timestamp(start), _to_timestamp(end)
                if pd.notna(lo):
                    out = out[as_dt >= lo]
                    as_dt = as_dt.loc[out.index]
                if pd.notna(hi):
                    out = out[as_dt <= hi]
            else:
                # month_year / quarter_year / year are sortable as strings
                as_str = out[col].astype(str)
                if start:
                    out = out[as_str >= str(start)]
                    as_str = as_str.loc[out.index]
                if end:
                    out = out[as_str <= str(end)]

    return out


def describe_filters(filters: dict, df: pd.DataFrame = None) -> str:
    """One-line human summary of the active filters, for captions and titles."""
    if not filters:
        return ""
    parts = []
    for col, spec in filters.items():
        if not isinstance(spec, dict):
            continue
        label = pretty_label(col)
        if spec.get("kind") == "category" and spec.get("values"):
            values = list(spec["values"])
            shown = ", ".join(map(str, values[:3]))
            if len(values) > 3:
                shown += f" +{len(values) - 3} more"
            parts.append(f"{label}: {shown}")
        elif spec.get("kind") == "date" and (spec.get("start") or spec.get("end")):
            parts.append(f"{label}: {spec.get('start', '')} to {spec.get('end', '')}")
    return " | ".join(parts)


_VIZ_CONFIG_RE = re.compile(r"\bVIZ_CONFIG\s*=", re.I)


def is_viz_config_payload(code: str | None) -> bool:
    return bool(_VIZ_CONFIG_RE.search(str(code or "")))


def _coerce_y_columns(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _repair_filters(df: pd.DataFrame, filters: dict | None) -> tuple[dict, list]:
    specs = filterable_columns(df)
    if not filters:
        return {}, specs

    valid_by_column = {
        spec["column"]: spec for spec in specs if isinstance(spec, dict) and spec.get("column")
    }
    repaired: dict[str, dict] = {}

    for column, spec in (filters or {}).items():
        if column not in valid_by_column or not isinstance(spec, dict):
            continue
        kind = spec.get("kind")
        allowed = valid_by_column[column]
        values = allowed.get("values") or []

        if kind == "category":
            chosen = [str(v) for v in (spec.get("values") or []) if str(v) in values]
            if chosen and len(chosen) < len(values):
                repaired[column] = {"kind": "category", "values": chosen}
        elif kind == "date":
            start = str(spec.get("start") or "").strip()
            end = str(spec.get("end") or "").strip()
            if start and start not in values:
                start = ""
            if end and end not in values:
                end = ""
            if start or end:
                repaired[column] = {"kind": "date", "start": start or None, "end": end or None}

    return repaired, specs


def build_interactive_chart_payload(
    df: pd.DataFrame,
    viz_code: str | None,
    *,
    question: str | None = None,
    selection: dict | None = None,
) -> dict:
    selection = selection or {}
    if df is None or df.empty:
        return {
            "editable": False,
            "no_visualization": True,
            "no_rows_match": False,
            "reason": "Empty result set.",
            "figure": None,
            "chart_selection": {},
            "default_selection": {},
            "x_options": [],
            "y_options": [],
            "chart_options": [],
            "filter_specs": [],
        }

    base_df = normalize_frame(df.copy())
    metric_comparison = bool(selection.get("metric_comparison")) or needs_metric_comparison(base_df)
    working_df = metric_comparison_frame(base_df) if metric_comparison else base_df
    default_cfg = parse_viz_config(viz_code or "", working_df, question=question)

    if default_cfg.get("no_visualization"):
        return {
            "editable": is_viz_config_payload(viz_code),
            "no_visualization": True,
            "no_rows_match": False,
            "reason": default_cfg.get("reason") or "Not suitable for a chart.",
            "figure": None,
            "chart_selection": {},
            "default_selection": {},
            "x_options": [],
            "y_options": [],
            "chart_options": [],
            "filter_specs": [],
        }

    x_options = selectable_x_columns(working_df)
    if not x_options:
        return {
            "editable": False,
            "no_visualization": True,
            "no_rows_match": False,
            "reason": "No column in this result can be used as an X axis.",
            "figure": None,
            "chart_selection": {},
            "default_selection": {},
            "x_options": [],
            "y_options": [],
            "chart_options": [],
            "filter_specs": [],
        }

    preferred_x = selection.get("x_column")
    default_x = default_cfg.get("x_column")
    x_col = preferred_x if preferred_x in x_options else default_x if default_x in x_options else x_options[0]

    y_options = selectable_y_columns(working_df, exclude=[x_col])
    if not y_options:
        return {
            "editable": False,
            "no_visualization": True,
            "no_rows_match": False,
            "reason": "No numeric measure left to plot against this X axis.",
            "figure": None,
            "chart_selection": {},
            "default_selection": {},
            "x_options": x_options,
            "y_options": [],
            "chart_options": [],
            "filter_specs": [],
        }

    preferred_y = _coerce_y_columns(selection.get("y_columns"))
    default_y = _coerce_y_columns(default_cfg.get("y_columns"))
    y_cols = [col for col in preferred_y if col in y_options and col != x_col]
    if not y_cols:
        y_cols = [col for col in default_y if col in y_options and col != x_col] or y_options[:1]

    filters, filter_specs = _repair_filters(working_df, selection.get("filters"))
    plot_source = apply_filters(working_df, filters)
    chart_source = plot_source if not plot_source.empty else working_df

    chart_options = eligible_chart_types(chart_source, x_col, y_cols) or ["Bar"]
    preferred_chart = selection.get("chart_type")
    default_chart = default_cfg.get("chart_type")
    chart_type = (
        preferred_chart if preferred_chart in chart_options
        else default_chart if default_chart in chart_options
        else chart_options[0]
    )

    limit = max_metrics_for(chart_type)
    if len(y_cols) > limit:
        y_cols = y_cols[:limit]
        chart_options = eligible_chart_types(chart_source, x_col, y_cols) or [chart_type]
        if chart_type not in chart_options:
            chart_type = chart_options[0]

    default_selection = {
        "chart_type": default_cfg.get("chart_type"),
        "x_column": default_cfg.get("x_column"),
        "y_columns": list(default_cfg.get("y_columns") or []),
        "title": default_cfg.get("title"),
        "labels": default_cfg.get("labels") or {},
        "top_n": default_cfg.get("top_n", DEFAULT_TOP_N),
        "filters": {},
        "is_default_view": True,
        "metric_comparison": metric_comparison,
    }
    is_default_view = (
        chart_type == default_selection["chart_type"]
        and x_col == default_selection["x_column"]
        and set(y_cols) == set(default_selection["y_columns"])
        and not filters
    )
    selection_out = {
        "chart_type": chart_type,
        "x_column": x_col,
        "y_columns": list(y_cols),
        "title": default_selection["title"] if is_default_view else None,
        "labels": default_selection["labels"],
        "top_n": default_selection["top_n"],
        "filters": filters,
        "is_default_view": is_default_view,
        "metric_comparison": metric_comparison,
    }

    figure = None
    if not plot_source.empty:
        figure = build_chart(
            plot_source,
            chart_type=selection_out["chart_type"],
            x_col=selection_out["x_column"],
            y_cols=selection_out["y_columns"],
            title=selection_out["title"],
            labels=selection_out["labels"],
            top_n=selection_out["top_n"],
        )

    return {
        "editable": is_viz_config_payload(viz_code),
        "no_visualization": False,
        "no_rows_match": plot_source.empty,
        "reason": "No rows match the current filters." if plot_source.empty else None,
        "figure": figure,
        "chart_selection": selection_out,
        "default_selection": default_selection,
        "x_options": x_options,
        "y_options": y_options,
        "chart_options": chart_options,
        "filter_specs": filter_specs,
    }
