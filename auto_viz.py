#!/usr/bin/env python3
"""
auto_viz.py — Intelligent Automatic Visualization Tool

Analyzes an arbitrary tabular dataset and generates a curated set of
analyst-quality visualizations. Rather than charting every column, it
profiles each column's semantic type, scores candidate charts by
usefulness, and outputs only the most meaningful plots.

Architecture overview:
  DataLoader → profile_columns() → ChartCandidateGenerator →
  select_final_charts() → render_charts() → write_report()

Web-readiness: run_analysis() is a pure-Python function that accepts a
file path and output directory — straightforward to wrap in a Flask/
FastAPI endpoint for a future upload-and-visualize web interface.
"""

import argparse
import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; safe for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_TOTAL_CHARTS = 40          # Hard cap on output charts
MAX_BIVARIATE_PAIRS = 20       # Max numeric×numeric scatter candidates
MAX_SUBGROUP_CHARTS = 12       # Max box/violin plots total
MAX_CATEGORIES_BEFORE_TOPN = 20  # Cardinality above which we use Top-N bar
TOP_N_CATEGORIES = 15          # How many bars in a Top-N bar chart
MAX_PAIRPLOT_COLS = 6          # Pairplot only when ≤ this many numeric cols
HIGH_CARDINALITY_RATIO = 0.8   # n_unique/n_rows above this → identifier/text
MIN_ROWS_FOR_HEXBIN = 1_000    # Use hexbin instead of scatter above this
CORRELATION_THRESHOLD = 0.25   # Min |r| to warrant a highlighted scatter
MISSINGNESS_THRESHOLD = 0.01   # Min fraction missing to appear in miss chart
DPI = 120                      # Output image resolution

# Semantic type labels
NUMERIC_CONTINUOUS  = "numeric_continuous"
NUMERIC_DISCRETE    = "numeric_discrete"
BINARY              = "binary"
BOOLEAN             = "boolean"
ORDINAL_CATEGORICAL = "ordinal_categorical"
NOMINAL_CATEGORICAL = "nominal_categorical"
DATETIME            = "datetime"
IDENTIFIER          = "identifier"
TEXT_FREEFORM       = "text_freeform"
LIKELY_TARGET       = "likely_target"
PERCENTAGE_RATE     = "percentage_rate"
CURRENCY            = "currency"
GEOGRAPHIC          = "geographic"

SINGLE_COLOR = "#4C72B0"
PALETTE      = "Set2"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    """All metadata inferred about a single column."""
    name:             str
    dtype:            str
    semantic_type:    str
    n_unique:         int
    n_missing:        int
    missing_fraction: float
    cardinality_ratio: float        # n_unique / n_rows
    sample_values:    List[Any]
    is_numeric:       bool
    is_categorical:   bool
    is_datetime:      bool
    is_usable:        bool          # False → skip this column entirely
    skip_reason:      Optional[str] = None
    notes:            List[str]     = field(default_factory=list)


@dataclass
class ChartCandidate:
    """A single chart that *could* be generated."""
    chart_id:         str
    chart_type:       str
    columns:          List[str]
    title:            str
    rationale:        str
    usefulness_score: float              = 0.0
    is_selected:      bool               = False
    output_path:      Optional[str]      = None
    kwargs:           Dict[str, Any]     = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Name-pattern regexes for semantic inference
# ──────────────────────────────────────────────────────────────────────────────

_RE_TARGET     = re.compile(
    r"\b(target|label|outcome|churn|survived|survival|fraud|default|"
    r"response|output|result|prediction|converted|purchased|clicked|"
    r"churned|died|admission|readmit|event|status|y|class)\b", re.I)

_RE_IDENTIFIER = re.compile(
    r"(?:^|_|(?<=[a-z]))(id|uuid|guid|index|key|row_?num|record|serial|hash|token|"
    r"code|ref|number)$", re.I)

_RE_GEOGRAPHIC = re.compile(
    r"\b(lat|lon|latitude|longitude|state|country|region|city|zip|"
    r"postal|county|province|territory|geo|location|place|address)\b", re.I)

_RE_PERCENTAGE = re.compile(
    r"\b(rate|pct|percent|percentage|ratio|proportion|share|fraction)\b", re.I)

_RE_CURRENCY   = re.compile(
    r"\b(price|cost|revenue|salary|wage|income|spend|amount|fee|charge|"
    r"budget|payment|earnings|profit|loss|value|worth)\b", re.I)

_RE_ORDINAL    = re.compile(
    r"\b(rank|tier|level|grade|score|rating|priority|severity|stage|"
    r"quartile|quintile|decile|education|degree)\b", re.I)

_RE_DATETIME_NAME = re.compile(
    r"\b(date|time|year|month|day|hour|timestamp|created|updated|dt)\b", re.I)


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """Loads CSV, TSV, Excel, and Parquet files into a DataFrame."""

    SUPPORTED = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}

    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = path.suffix.lower()
        if ext not in DataLoader.SUPPORTED:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {DataLoader.SUPPORTED}"
            )
        logger.info(f"Loading {ext} file: {file_path}")
        try:
            if ext == ".tsv":
                df = pd.read_csv(file_path, sep="\t", low_memory=False)
            elif ext == ".csv":
                df = pd.read_csv(file_path, low_memory=False)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(file_path)
            else:  # .parquet
                df = pd.read_parquet(file_path)
        except UnicodeDecodeError:
            # Fallback for CSV/TSV with non-UTF-8 encoding
            logger.warning("UTF-8 parse failed; retrying with latin-1 encoding.")
            df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
        return df


# ──────────────────────────────────────────────────────────────────────────────
# Semantic type inference
# ──────────────────────────────────────────────────────────────────────────────

def infer_semantic_type(
    col: pd.Series, col_name: str, n_rows: int
) -> Tuple[str, List[str]]:
    """
    Infer the semantic role of a column beyond its raw pandas dtype.

    Returns (semantic_type_constant, list_of_explanatory_notes).
    Decision flow:
      1. Datetime  →  2. Boolean  →  3. Numeric subtypes
      →  4. Object/string subtypes  →  5. Fallback
    """
    notes: List[str] = []
    n_unique = col.nunique(dropna=True)
    cardinality_ratio = n_unique / max(n_rows, 1)

    # ── 1. Datetime ──────────────────────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(col):
        return DATETIME, ["Native datetime dtype"]

    if col.dtype == object and (_RE_DATETIME_NAME.search(col_name) or True):
        sample = col.dropna().head(50)
        if len(sample):
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    notes.append("Values parse as datetime")
                    return DATETIME, notes
            except Exception:
                pass

    # ── 2. Boolean ────────────────────────────────────────────────────────────
    if col.dtype == bool:
        return BOOLEAN, ["Native bool dtype"]

    bool_vals = {True, False, 0, 1, "0", "1", "true", "false", "True", "False",
                 "yes", "no", "YES", "NO", "Yes", "No", "Y", "N", "y", "n"}
    if n_unique <= 2 and set(col.dropna().unique()).issubset(bool_vals):
        notes.append("Boolean-like values detected")
        return BOOLEAN, notes

    # ── 3. Numeric ────────────────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(col):

        # Pure identifier (e.g. user_id, row_id, PassengerId)
        if cardinality_ratio > 0.95 and (
            _RE_IDENTIFIER.search(col_name)
            or col.is_monotonic_increasing
            or col.is_monotonic_decreasing
        ):
            notes.append("Near-unique numeric column with identifier-like pattern/order → identifier")
            return IDENTIFIER, notes

        if n_unique == 2:
            notes.append("Two unique numeric values → binary")
            return BINARY, notes

        col_clean = col.dropna()

        # Percentage / rate
        if _RE_PERCENTAGE.search(col_name):
            if col_clean.max() <= 1.01:
                notes.append("Name + 0–1 range → proportion/rate")
                return PERCENTAGE_RATE, notes
            if col_clean.max() <= 100.01 and col_clean.min() >= 0:
                notes.append("Name + 0–100 range → percentage")
                return PERCENTAGE_RATE, notes

        # Currency / monetary
        if _RE_CURRENCY.search(col_name):
            notes.append("Name suggests monetary value")
            return CURRENCY, notes

        # Discrete: integer dtype and few unique values
        if pd.api.types.is_integer_dtype(col) and n_unique <= 20:
            notes.append(f"Integer, {n_unique} unique values → discrete")
            return NUMERIC_DISCRETE, notes

        # Non-integer or large unique count → continuous
        notes.append(f"{n_unique} unique numeric values → continuous")
        return NUMERIC_CONTINUOUS, notes

    # ── 4. Object / string ────────────────────────────────────────────────────
    if col.dtype == object:

        avg_len = col.dropna().astype(str).str.len().mean()

        # Free-form text
        if avg_len > 60:
            notes.append(f"Avg string length {avg_len:.0f} chars → free-form text")
            return TEXT_FREEFORM, notes

        # Identifier by name + high cardinality
        if _RE_IDENTIFIER.search(col_name) and cardinality_ratio > 0.5:
            notes.append("Name + high cardinality → identifier")
            return IDENTIFIER, notes

        # UUID / hash pattern check
        sample_str = col.dropna().head(10).astype(str)
        uuid_like = sample_str.str.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", case=False
        ).mean()
        if uuid_like > 0.8:
            notes.append("Values match UUID pattern → identifier")
            return IDENTIFIER, notes

        if cardinality_ratio > HIGH_CARDINALITY_RATIO:
            notes.append(f"Cardinality ratio {cardinality_ratio:.1%} → identifier/text")
            return IDENTIFIER, notes

        # Geographic
        if _RE_GEOGRAPHIC.search(col_name):
            notes.append("Name suggests geographic role")
            return GEOGRAPHIC, notes

        if n_unique == 2:
            notes.append("Two unique string values → binary")
            return BINARY, notes

        if _RE_ORDINAL.search(col_name) and n_unique <= 20:
            notes.append("Name suggests ordinal categorical")
            return ORDINAL_CATEGORICAL, notes

        notes.append(f"{n_unique} unique string values → nominal categorical")
        return NOMINAL_CATEGORICAL, notes

    # ── 5. Fallback ───────────────────────────────────────────────────────────
    notes.append("Fallback assignment")
    return NUMERIC_CONTINUOUS, notes


# ──────────────────────────────────────────────────────────────────────────────
# Column profiler
# ──────────────────────────────────────────────────────────────────────────────

def profile_columns(df: pd.DataFrame) -> List[ColumnProfile]:
    """
    Build a ColumnProfile for every column in df.
    Determines semantic type, usability, and skip reasons.
    """
    n_rows = len(df)
    profiles: List[ColumnProfile] = []

    for col_name in df.columns:
        col = df[col_name]
        n_unique        = col.nunique(dropna=True)
        n_missing       = int(col.isnull().sum())
        missing_fraction = n_missing / max(n_rows, 1)
        cardinality_ratio = n_unique / max(n_rows, 1)
        sample_values   = [str(v) for v in col.dropna().head(5).tolist()]

        sem_type, notes = infer_semantic_type(col, col_name, n_rows)

        # Post-hoc: override to LIKELY_TARGET when name matches known patterns
        if _RE_TARGET.search(col_name) and sem_type in (
            BINARY, BOOLEAN, NOMINAL_CATEGORICAL, NUMERIC_DISCRETE,
            ORDINAL_CATEGORICAL
        ):
            notes.append("Name matches target/outcome patterns → likely_target")
            sem_type = LIKELY_TARGET

        # Decide usability
        is_usable  = True
        skip_reason: Optional[str] = None

        if sem_type in (IDENTIFIER, TEXT_FREEFORM):
            is_usable   = False
            skip_reason = f"Semantic type '{sem_type}' — not analytically useful for plotting"
        elif sem_type == GEOGRAPHIC:
            is_usable   = False
            skip_reason = "Geographic column — skipped (no map renderer in this tool)"
        elif n_unique < 2:
            is_usable   = False
            skip_reason = f"Only {n_unique} unique value(s) — no variation to visualize"
        elif missing_fraction > 0.95:
            is_usable   = False
            skip_reason = f"{missing_fraction:.0%} missing — too sparse"

        is_numeric     = pd.api.types.is_numeric_dtype(col) or sem_type in (
            NUMERIC_CONTINUOUS, NUMERIC_DISCRETE, BINARY, BOOLEAN,
            PERCENTAGE_RATE, CURRENCY, LIKELY_TARGET,
        )
        is_categorical = sem_type in (
            BINARY, BOOLEAN, NOMINAL_CATEGORICAL, ORDINAL_CATEGORICAL, LIKELY_TARGET
        )
        is_datetime    = sem_type == DATETIME

        profiles.append(ColumnProfile(
            name=col_name, dtype=str(col.dtype),
            semantic_type=sem_type,
            n_unique=n_unique, n_missing=n_missing,
            missing_fraction=missing_fraction,
            cardinality_ratio=cardinality_ratio,
            sample_values=sample_values,
            is_numeric=is_numeric, is_categorical=is_categorical,
            is_datetime=is_datetime, is_usable=is_usable,
            skip_reason=skip_reason, notes=notes,
        ))

    return profiles


# ──────────────────────────────────────────────────────────────────────────────
# Chart candidate generator
# ──────────────────────────────────────────────────────────────────────────────

class ChartCandidateGenerator:
    """
    Generates ChartCandidate objects for every chart that *could* be
    meaningful.  Nothing is rendered here — this is pure decision logic.

    Extend this class to add new chart types: add a _generate_X() method
    and call it from generate_all().
    """

    def __init__(self, df: pd.DataFrame, profiles: List[ColumnProfile]):
        self.df          = df
        self.profiles    = profiles
        self.profile_map = {p.name: p for p in profiles}
        self.n_rows      = len(df)
        self.candidates: List[ChartCandidate] = []
        self._counter    = 0

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:03d}"

    def _usable(self)      -> List[ColumnProfile]: return [p for p in self.profiles if p.is_usable]
    def _numeric(self)     -> List[ColumnProfile]: return [p for p in self._usable() if p.is_numeric]
    def _categorical(self) -> List[ColumnProfile]: return [p for p in self._usable() if p.is_categorical]
    def _datetime(self)    -> List[ColumnProfile]: return [p for p in self._usable() if p.is_datetime]

    def _groupby_candidates(self, exclude: Optional[List[str]] = None) -> List[ColumnProfile]:
        """
        Categorical columns that make good grouping variables:
        interpretable, low-to-medium cardinality, not identifiers.
        """
        excl = set(exclude or [])
        return [
            p for p in self._categorical()
            if p.name not in excl
            and 2 <= p.n_unique <= 12
        ]

    def _add(self, candidate: ChartCandidate):
        self.candidates.append(candidate)

    # ── Public ───────────────────────────────────────────────────────────────

    def generate_all(self) -> List[ChartCandidate]:
        """Run all generation methods and return the full candidate list."""
        self._gen_univariate_numeric()
        self._gen_univariate_categorical()
        self._gen_numeric_numeric()
        self._gen_numeric_categorical()
        self._gen_categorical_categorical()
        self._gen_datetime_numeric()
        self._gen_correlation_heatmap()
        self._gen_pairplot()
        self._gen_missingness()
        return self.candidates

    # ── Univariate ───────────────────────────────────────────────────────────

    def _gen_univariate_numeric(self):
        for p in self._numeric():
            self._add(ChartCandidate(
                chart_id   = self._id("uni_hist"),
                chart_type = "histogram_kde",
                columns    = [p.name],
                title      = f"Distribution of {p.name}",
                rationale  = (
                    f"'{p.name}' ({p.semantic_type}) — histogram + KDE reveals distribution "
                    f"shape, skewness, modality, and outliers."
                ),
                kwargs={"semantic_type": p.semantic_type},
            ))

    def _gen_univariate_categorical(self):
        for p in self._categorical():
            if p.n_unique > MAX_CATEGORIES_BEFORE_TOPN:
                ctype     = "topn_bar"
                title     = f"Top {TOP_N_CATEGORIES} values — {p.name}"
                rationale = (
                    f"'{p.name}' has {p.n_unique} categories — showing top {TOP_N_CATEGORIES} "
                    f"by frequency to avoid an unreadable chart."
                )
            else:
                ctype     = "count_bar"
                title     = f"Frequency of {p.name}"
                rationale = (
                    f"'{p.name}' is categorical with {p.n_unique} unique values. "
                    f"Bar chart shows class balance and frequency distribution."
                )
            self._add(ChartCandidate(
                chart_id=self._id("uni_cat"), chart_type=ctype,
                columns=[p.name], title=title, rationale=rationale,
            ))

    # ── Bivariate: Numeric × Numeric ─────────────────────────────────────────

    def _gen_numeric_numeric(self):
        num_cols = self._numeric()
        pairs_added = 0
        for i, p1 in enumerate(num_cols):
            for p2 in num_cols[i + 1:]:
                if pairs_added >= MAX_BIVARIATE_PAIRS:
                    return
                try:
                    sub = self.df[[p1.name, p2.name]].dropna()
                    if len(sub) < 10:
                        continue
                    corr = sub.corr().iloc[0, 1]
                    if pd.isna(corr):
                        continue
                except Exception:
                    continue

                ctype = "hexbin" if self.n_rows >= MIN_ROWS_FOR_HEXBIN else "scatter"
                self._add(ChartCandidate(
                    chart_id   = self._id("biv_nn"),
                    chart_type = ctype,
                    columns    = [p1.name, p2.name],
                    title      = f"{p1.name} vs {p2.name}",
                    rationale  = (
                        f"Pearson r = {corr:.2f} between '{p1.name}' and '{p2.name}'. "
                        + ("Strong linear relationship." if abs(corr) >= 0.5
                           else "Moderate relationship — scatter may show non-linear structure."
                           if abs(corr) >= CORRELATION_THRESHOLD
                           else "Weak linear correlation; plotted to surface non-linear patterns.")
                    ),
                    kwargs={"correlation": float(corr)},
                ))
                pairs_added += 1

    # ── Bivariate: Numeric × Categorical ─────────────────────────────────────

    def _gen_numeric_categorical(self):
        for cat in self._groupby_candidates():
            for num in self._numeric():
                if num.name == cat.name:
                    continue
                # Violin for small category counts, box otherwise
                if cat.n_unique <= 8:
                    ctype = "violin"
                    rationale = (
                        f"Violin plot shows full distribution of '{num.name}' across "
                        f"{cat.n_unique} groups of '{cat.name}'."
                    )
                else:
                    ctype = "box"
                    rationale = (
                        f"Box plot compares '{num.name}' across {cat.n_unique} "
                        f"groups of '{cat.name}' — median, IQR, outliers."
                    )
                self._add(ChartCandidate(
                    chart_id   = self._id("biv_nc"),
                    chart_type = ctype,
                    columns    = [num.name, cat.name],
                    title      = f"{num.name} by {cat.name}",
                    rationale  = rationale,
                ))

    # ── Bivariate: Categorical × Categorical ─────────────────────────────────

    def _gen_categorical_categorical(self):
        cat_cols = self._categorical()
        seen: set = set()
        for i, p1 in enumerate(cat_cols):
            for p2 in cat_cols[i + 1:]:
                if p1.n_unique > 10 or p2.n_unique > 10:
                    continue
                pair = tuple(sorted([p1.name, p2.name]))
                if pair in seen:
                    continue
                seen.add(pair)
                self._add(ChartCandidate(
                    chart_id   = self._id("biv_cc"),
                    chart_type = "contingency_heatmap",
                    columns    = [p1.name, p2.name],
                    title      = f"{p1.name} × {p2.name} (contingency)",
                    rationale  = (
                        f"Contingency heatmap for '{p1.name}' ({p1.n_unique} cats) × "
                        f"'{p2.name}' ({p2.n_unique} cats). Reveals co-occurrence patterns."
                    ),
                ))

    # ── Datetime × Numeric ───────────────────────────────────────────────────

    def _gen_datetime_numeric(self):
        for dt in self._datetime():
            for num in self._numeric():
                self._add(ChartCandidate(
                    chart_id   = self._id("ts"),
                    chart_type = "time_series",
                    columns    = [dt.name, num.name],
                    title      = f"{num.name} over time",
                    rationale  = (
                        f"'{dt.name}' is a datetime column. Line chart of '{num.name}' "
                        f"surfaces trends, seasonality, and temporal anomalies."
                    ),
                ))
                # Grouped time series — limit to 2 grouping vars per metric
                for cat in self._groupby_candidates(exclude=[num.name])[:2]:
                    self._add(ChartCandidate(
                        chart_id   = self._id("ts_grp"),
                        chart_type = "time_series_grouped",
                        columns    = [dt.name, num.name, cat.name],
                        title      = f"{num.name} over time by {cat.name}",
                        rationale  = (
                            f"Grouped time series shows how '{num.name}' trends differ "
                            f"across '{cat.name}' groups — useful for detecting divergence."
                        ),
                    ))

    # ── Multivariate ─────────────────────────────────────────────────────────

    def _gen_correlation_heatmap(self):
        num_cols = self._numeric()
        if len(num_cols) >= 3:
            self._add(ChartCandidate(
                chart_id   = self._id("corr"),
                chart_type = "correlation_heatmap",
                columns    = [p.name for p in num_cols],
                title      = "Correlation Heatmap — Numeric Variables",
                rationale  = (
                    f"With {len(num_cols)} numeric columns, a correlation heatmap gives "
                    f"an at-a-glance summary of all pairwise linear relationships."
                ),
            ))

    def _gen_pairplot(self):
        num_cols = self._numeric()
        if 3 <= len(num_cols) <= MAX_PAIRPLOT_COLS:
            groupby = self._groupby_candidates()
            hue = groupby[0].name if groupby else None
            self._add(ChartCandidate(
                chart_id   = self._id("pair"),
                chart_type = "pairplot",
                columns    = [p.name for p in num_cols],
                title      = "Pairplot — Numeric Variables",
                rationale  = (
                    f"Pairplot across {len(num_cols)} numeric columns shows every pairwise "
                    f"scatter + marginal distribution."
                    + (f" Colored by '{hue}'." if hue else "")
                ),
                kwargs={"hue": hue},
            ))

    def _gen_missingness(self):
        missing_cols = [p for p in self.profiles if p.missing_fraction >= MISSINGNESS_THRESHOLD]
        if len(missing_cols) >= 2:
            self._add(ChartCandidate(
                chart_id   = self._id("miss"),
                chart_type = "missingness_bar",
                columns    = [p.name for p in missing_cols],
                title      = "Missing Data by Column",
                rationale  = (
                    f"{len(missing_cols)} columns have ≥{MISSINGNESS_THRESHOLD:.0%} missing "
                    f"values. Missingness chart highlights data quality issues."
                ),
            ))


# ──────────────────────────────────────────────────────────────────────────────
# Chart scorer
# ──────────────────────────────────────────────────────────────────────────────

def score_chart_usefulness(
    candidate: ChartCandidate,
    profile_map: Dict[str, ColumnProfile],
) -> float:
    """
    Assign a usefulness score in [0, 10].
    Higher = more analytically valuable. Used to rank and cap output.

    Factors considered:
    - Correlation strength for scatter/hexbin
    - Presence of target/outcome column
    - Cardinality suitability for the chosen chart type
    - Category count for box/violin (clean comparisons score higher)
    - Inherently high-value chart types (heatmap, time series)
    - Missingness severity
    """
    score = 5.0
    ctype = candidate.chart_type
    cols  = candidate.columns

    # ── Chart-type baseline adjustments ──────────────────────────────────────
    if ctype == "correlation_heatmap":
        score = 8.5
    elif ctype == "pairplot":
        score = 7.5 + (0.5 if candidate.kwargs.get("hue") else 0)
    elif ctype in ("time_series", "time_series_grouped"):
        score = 8.0
    elif ctype == "missingness_bar":
        total_miss = sum(
            profile_map[c].missing_fraction for c in cols if c in profile_map
        )
        score = min(10.0, 6.5 + total_miss * 2)

    # ── Correlation bonus for scatter / hexbin ────────────────────────────────
    if ctype in ("scatter", "hexbin"):
        corr = abs(candidate.kwargs.get("correlation", 0.0))
        score += corr * 3.5   # max +3.5 for r=1

    # ── Target column bonus ───────────────────────────────────────────────────
    for col in cols:
        p = profile_map.get(col)
        if p and p.semantic_type == LIKELY_TARGET:
            score += 1.5
            break   # count once

    # ── Penalise box/violin when too many categories ──────────────────────────
    if ctype in ("box", "violin") and len(cols) >= 2:
        cat_p = profile_map.get(cols[1])
        if cat_p:
            if cat_p.n_unique > 12:
                score -= 2.5
            elif cat_p.n_unique == 2:
                score += 1.0   # binary comparison is clean

    # ── Penalise charts on low-variation columns ──────────────────────────────
    for col in cols:
        p = profile_map.get(col)
        if p and p.n_unique < 3:
            score -= 1.5

    return max(0.0, min(10.0, score))


# ──────────────────────────────────────────────────────────────────────────────
# Chart selector
# ──────────────────────────────────────────────────────────────────────────────

def select_final_charts(
    candidates: List[ChartCandidate],
    df: pd.DataFrame,
    profile_map: Dict[str, ColumnProfile],
    max_charts: int = MAX_TOTAL_CHARTS,
) -> List[ChartCandidate]:
    """
    Score, deduplicate, and pick the final chart set.

    Deduplication key: (frozenset(columns), chart_type).
    Type-level caps prevent any single category from flooding the output.
    """
    # Score every candidate
    for c in candidates:
        c.usefulness_score = score_chart_usefulness(c, profile_map)

    # Sort descending by score
    candidates.sort(key=lambda c: c.usefulness_score, reverse=True)

    final:       List[ChartCandidate] = []
    seen_keys:   set = set()
    n_subgroup   = 0       # box + violin count
    n_scatter    = 0       # scatter + hexbin count

    for c in candidates:
        col_key  = frozenset(c.columns)
        dedup_key = (col_key, c.chart_type)
        if dedup_key in seen_keys:
            continue

        # Per-type caps
        if c.chart_type in ("box", "violin"):
            if n_subgroup >= MAX_SUBGROUP_CHARTS:
                continue
            n_subgroup += 1

        if c.chart_type in ("scatter", "hexbin"):
            if n_scatter >= MAX_BIVARIATE_PAIRS:
                continue
            n_scatter += 1

        seen_keys.add(dedup_key)
        c.is_selected = True
        final.append(c)

        if len(final) >= max_charts:
            break

    logger.info(
        f"Selected {len(final)} charts from {len(candidates)} candidates "
        f"(cap={max_charts})."
    )
    return final


# ──────────────────────────────────────────────────────────────────────────────
# Chart rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

# Global matplotlib style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f7f7f7",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})


def _save(fig: plt.Figure, path: Path, title: str):
    """Apply a suptitle, tighten layout, save, and close."""
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")


# ── Individual chart renderers ────────────────────────────────────────────────

def _chart_histogram_kde(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    col  = c.columns[0]
    data = df[col].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data, kde=True, ax=ax, color=SINGLE_COLOR, alpha=0.7, bins="auto")
    ax.set_xlabel(col); ax.set_ylabel("Count")
    mean_, med_ = data.mean(), data.median()
    ax.axvline(mean_, color="red",   ls="--", lw=1.3, label=f"Mean: {mean_:.2f}")
    ax.axvline(med_,  color="green", ls="--", lw=1.3, label=f"Median: {med_:.2f}")
    ax.legend(fontsize=9)
    _save(fig, path, c.title)


def _chart_count_bar(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    col    = c.columns[0]
    counts = df[col].dropna().value_counts().sort_values(ascending=False)
    w      = max(6, min(16, len(counts) * 0.65 + 2))
    fig, ax = plt.subplots(figsize=(w, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                palette=PALETTE, order=counts.index.astype(str))
    ax.set_xlabel(col); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path, c.title)


def _chart_topn_bar(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    col    = c.columns[0]
    counts = df[col].dropna().value_counts().head(TOP_N_CATEGORIES)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                palette=PALETTE, order=counts.index.astype(str))
    ax.set_xlabel(col); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path, c.title)


def _chart_scatter(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    xc, yc   = c.columns[0], c.columns[1]
    plot_df  = df[[xc, yc]].dropna()
    if len(plot_df) > 5_000:
        plot_df = plot_df.sample(5_000, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_df[xc], plot_df[yc], alpha=0.4, s=16, color=SINGLE_COLOR)
    try:
        m, b, r, _p, _se = stats.linregress(plot_df[xc], plot_df[yc])
        xs = np.linspace(plot_df[xc].min(), plot_df[xc].max(), 200)
        ax.plot(xs, m * xs + b, color="red", lw=1.8, label=f"r = {r:.2f}")
        ax.legend(fontsize=9)
    except Exception:
        pass
    ax.set_xlabel(xc); ax.set_ylabel(yc)
    _save(fig, path, c.title)


def _chart_hexbin(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    xc, yc   = c.columns[0], c.columns[1]
    plot_df  = df[[xc, yc]].dropna()
    fig, ax  = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(plot_df[xc], plot_df[yc], gridsize=40, cmap="Blues", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")
    ax.set_xlabel(xc); ax.set_ylabel(yc)
    _save(fig, path, c.title)


def _chart_box(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    num_col, cat_col = c.columns[0], c.columns[1]
    plot_df = df[[num_col, cat_col]].dropna()
    order   = plot_df[cat_col].value_counts().index.tolist()
    w = max(7, len(order) * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(w, 5))
    sns.boxplot(data=plot_df, x=cat_col, y=num_col, order=order, ax=ax, palette=PALETTE)
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path, c.title)


def _chart_violin(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    num_col, cat_col = c.columns[0], c.columns[1]
    plot_df = df[[num_col, cat_col]].dropna()
    order   = plot_df[cat_col].value_counts().index.tolist()
    w = max(7, len(order) * 1.1 + 2)
    fig, ax = plt.subplots(figsize=(w, 5))
    sns.violinplot(data=plot_df, x=cat_col, y=num_col, order=order,
                   ax=ax, palette=PALETTE, inner="quartile")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path, c.title)


def _chart_contingency_heatmap(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    c1, c2 = c.columns[0], c.columns[1]
    ct = pd.crosstab(df[c1], df[c2], normalize="index")
    h  = max(4, ct.shape[0] * 0.65 + 2)
    w  = max(6, ct.shape[1] * 0.85 + 2)
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(ct, annot=True, fmt=".0%", cmap="Blues", ax=ax, linewidths=0.4)
    ax.set_xlabel(c2); ax.set_ylabel(c1)
    _save(fig, path, c.title)


def _chart_time_series(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    dt_col, num_col = c.columns[0], c.columns[1]
    plot_df = df[[dt_col, num_col]].copy()
    plot_df[dt_col] = pd.to_datetime(plot_df[dt_col], errors="coerce")
    plot_df = plot_df.dropna().sort_values(dt_col)
    # Resample to daily mean if very dense
    if len(plot_df) > 500:
        plot_df = (
            plot_df.set_index(dt_col)[num_col]
            .resample("D").mean()
            .reset_index()
            .rename(columns={"index": dt_col})
        )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df[dt_col], plot_df[num_col],
            color=SINGLE_COLOR, lw=1.5, alpha=0.8, label=num_col)
    window = max(3, min(14, len(plot_df) // 8))
    if len(plot_df) >= window:
        rolling = plot_df[num_col].rolling(window, min_periods=1).mean()
        ax.plot(plot_df[dt_col], rolling,
                color="red", lw=2.0, alpha=0.8, label=f"{window}-period rolling avg")
    ax.set_xlabel(dt_col); ax.set_ylabel(num_col)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    _save(fig, path, c.title)


def _chart_time_series_grouped(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    dt_col, num_col, cat_col = c.columns[0], c.columns[1], c.columns[2]
    plot_df = df[[dt_col, num_col, cat_col]].copy()
    plot_df[dt_col] = pd.to_datetime(plot_df[dt_col], errors="coerce")
    plot_df = plot_df.dropna().sort_values(dt_col)
    groups  = plot_df[cat_col].value_counts().head(6).index
    palette = sns.color_palette(PALETTE, len(groups))
    fig, ax = plt.subplots(figsize=(10, 5))
    for grp, color in zip(groups, palette):
        sub = (
            plot_df[plot_df[cat_col] == grp]
            .groupby(dt_col)[num_col].mean()
            .reset_index()
        )
        ax.plot(sub[dt_col], sub[num_col], label=str(grp), color=color, lw=1.5, alpha=0.85)
    ax.set_xlabel(dt_col); ax.set_ylabel(num_col)
    ax.legend(title=cat_col, fontsize=9)
    fig.autofmt_xdate()
    _save(fig, path, c.title)


def _chart_correlation_heatmap(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    cols    = c.columns
    corr_df = df[cols].select_dtypes(include=[np.number]).corr()
    n       = len(corr_df)
    h       = max(6, n * 0.65 + 2)
    w       = max(7, n * 0.75 + 2)
    fig, ax = plt.subplots(figsize=(w, h))
    # Show lower triangle only (avoid redundancy)
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(
        corr_df, mask=~mask & ~np.eye(n, dtype=bool),
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        ax=ax, linewidths=0.4, square=True, vmin=-1, vmax=1,
        annot_kws={"size": max(8, 12 - n // 3)},
    )
    _save(fig, path, c.title)


def _chart_pairplot(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    cols    = c.columns
    hue     = c.kwargs.get("hue")
    all_cols = list(cols) + ([hue] if hue else [])
    plot_df = df[all_cols].dropna()
    if len(plot_df) > 2_000:
        plot_df = plot_df.sample(2_000, random_state=42)
    g = sns.pairplot(
        plot_df, vars=list(cols), hue=hue, palette=PALETTE,
        plot_kws={"alpha": 0.45, "s": 14},
        diag_kind="kde" if hue else "hist",
    )
    g.fig.suptitle(c.title, y=1.02, fontsize=13, fontweight="bold")
    g.fig.tight_layout()
    g.fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(g.fig)
    logger.info(f"  Saved: {path.name}")


def _chart_missingness_bar(c: ChartCandidate, df: pd.DataFrame, pm: Dict, path: Path):
    miss = pd.Series(
        {col: pm[col].missing_fraction for col in c.columns if col in pm}
    ).sort_values(ascending=False)
    colors = [
        "#d73027" if v > 0.3 else "#fc8d59" if v > 0.1 else "#fee08b"
        for v in miss.values
    ]
    w   = max(6, len(miss) * 0.65 + 2)
    fig, ax = plt.subplots(figsize=(w, 5))
    ax.bar(miss.index.astype(str), miss.values * 100, color=colors)
    ax.set_ylabel("% Missing"); ax.set_xlabel("Column")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(10, color="orange", ls="--", lw=1.2, alpha=0.8, label="10%")
    ax.axhline(30, color="red",    ls="--", lw=1.2, alpha=0.8, label="30%")
    ax.legend(title="Thresholds", fontsize=9)
    _save(fig, path, c.title)


# ── Renderer dispatch table ───────────────────────────────────────────────────

_RENDERERS = {
    "histogram_kde":        _chart_histogram_kde,
    "count_bar":            _chart_count_bar,
    "topn_bar":             _chart_topn_bar,
    "scatter":              _chart_scatter,
    "hexbin":               _chart_hexbin,
    "box":                  _chart_box,
    "violin":               _chart_violin,
    "contingency_heatmap":  _chart_contingency_heatmap,
    "time_series":          _chart_time_series,
    "time_series_grouped":  _chart_time_series_grouped,
    "correlation_heatmap":  _chart_correlation_heatmap,
    "pairplot":             _chart_pairplot,
    "missingness_bar":      _chart_missingness_bar,
}


# ──────────────────────────────────────────────────────────────────────────────
# render_charts
# ──────────────────────────────────────────────────────────────────────────────

def render_charts(
    selected: List[ChartCandidate],
    df: pd.DataFrame,
    profile_map: Dict[str, ColumnProfile],
    output_dir: Path,
) -> List[str]:
    """
    Render every selected chart and save to output_dir.
    Returns a list of warning strings for any charts that failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    render_warnings: List[str] = []

    for chart in selected:
        renderer = _RENDERERS.get(chart.chart_type)
        if renderer is None:
            msg = f"No renderer for chart type '{chart.chart_type}'"
            logger.warning(msg)
            render_warnings.append(msg)
            continue
        try:
            out_path = output_dir / f"{chart.chart_id}.png"
            renderer(chart, df, profile_map, out_path)
            chart.output_path = str(out_path)
        except Exception as exc:
            msg = f"Render failed for '{chart.title}': {exc}"
            logger.warning(msg)
            render_warnings.append(msg)

    return render_warnings


# ──────────────────────────────────────────────────────────────────────────────
# Report writer
# ──────────────────────────────────────────────────────────────────────────────

def write_report(
    input_path:        str,
    df:                pd.DataFrame,
    profiles:          List[ColumnProfile],
    selected:          List[ChartCandidate],
    analysis_warnings: List[str],
    output_dir:        Path,
):
    """Write analysis_report.json and analysis_report.md to output_dir."""

    # ── JSON ─────────────────────────────────────────────────────────────────
    report_dict = {
        "dataset":                input_path,
        "n_rows":                 len(df),
        "n_cols":                 len(df.columns),
        "total_charts_generated": len(selected),
        "warnings":               analysis_warnings,
        "column_profiles": [
            {
                "name":             p.name,
                "dtype":            p.dtype,
                "semantic_type":    p.semantic_type,
                "n_unique":         p.n_unique,
                "missing_fraction": round(p.missing_fraction, 4),
                "is_usable":        p.is_usable,
                "skip_reason":      p.skip_reason,
                "notes":            p.notes,
            }
            for p in profiles
        ],
        "charts": [
            {
                "chart_id":        c.chart_id,
                "chart_type":      c.chart_type,
                "columns":         c.columns,
                "title":           c.title,
                "rationale":       c.rationale,
                "usefulness_score": round(c.usefulness_score, 2),
                "output_path":     c.output_path,
            }
            for c in selected
        ],
        "skipped_columns": [
            {"name": p.name, "reason": p.skip_reason}
            for p in profiles if not p.is_usable
        ],
    }

    json_path = output_dir / "analysis_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2)
    logger.info(f"JSON report: {json_path}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    lines = [
        "# Data Inspection Report\n",
        f"**Dataset:** `{input_path}`  ",
        f"**Shape:** {len(df):,} rows × {len(df.columns)} columns  ",
        f"**Charts generated:** {len(selected)}  \n",
    ]

    if analysis_warnings:
        lines += ["\n## ⚠ Warnings\n"] + [f"- {w}" for w in analysis_warnings]

    lines += [
        "\n## Column Profiles\n",
        "| Column | Semantic Type | Unique | Missing | Usable | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for p in profiles:
        usable_str = "Yes" if p.is_usable else f"No — {p.skip_reason or ''}"
        notes_str  = "; ".join(p.notes[:2])
        lines.append(
            f"| {p.name} | `{p.semantic_type}` | {p.n_unique} | "
            f"{p.missing_fraction:.1%} | {usable_str} | {notes_str} |"
        )

    lines += ["\n## Selected Charts (ranked by usefulness)\n"]
    for c in sorted(selected, key=lambda x: x.usefulness_score, reverse=True):
        fname = Path(c.output_path).name if c.output_path else "N/A"
        rel_path = Path(c.output_path).name if c.output_path else ""
        lines += [
            f"\n### {c.title}",
            f"- **Type:** `{c.chart_type}`",
            f"- **Columns:** {', '.join(f'`{col}`' for col in c.columns)}",
            f"- **Score:** {c.usefulness_score:.1f} / 10",
            f"- **Rationale:** {c.rationale}",
            "",
            f"![{c.title}]({rel_path})",
        ]

    skipped = [p for p in profiles if not p.is_usable]
    if skipped:
        lines += ["\n## Skipped Columns\n"] + [
            f"- **{p.name}** — {p.skip_reason}" for p in skipped
        ]

    md_path = output_dir / "analysis_report.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info(f"Markdown report: {md_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline  (web-friendly entry point)
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(
    input_path: str,
    output_dir: str,
    max_charts: int = MAX_TOTAL_CHARTS,
) -> dict:
    """
    Full auto-viz pipeline.

    Parameters
    ----------
    input_path : path to the input dataset (CSV / Excel / Parquet)
    output_dir : directory where charts and reports will be written
    max_charts : hard cap on number of output charts

    Returns
    -------
    dict  — the structured analysis report (same as analysis_report.json)

    Design note
    -----------
    This function is intentionally a plain Python callable with no CLI
    dependencies, making it straightforward to wrap in a Flask / FastAPI
    endpoint:

        @app.post("/analyze")
        def analyze(file: UploadFile):
            tmp = save_upload(file)
            result = run_analysis(tmp, "static/results/")
            return result
    """
    out_dir = Path(output_dir)

    # 1. Load
    df = DataLoader.load(input_path)

    # 2. Profile
    logger.info("Profiling columns …")
    profiles    = profile_columns(df)
    profile_map = {p.name: p for p in profiles}
    skipped = [p for p in profiles if not p.is_usable]
    logger.info(
        f"  Usable: {len(profiles) - len(skipped)}  |  "
        f"Skipped: {len(skipped)}"
    )
    for p in skipped:
        logger.info(f"    skip '{p.name}': {p.skip_reason}")

    # 3. Generate candidates
    logger.info("Generating chart candidates …")
    gen        = ChartCandidateGenerator(df, profiles)
    candidates = gen.generate_all()
    logger.info(f"  {len(candidates)} candidates generated.")

    # 4. Select
    logger.info("Scoring and selecting charts …")
    selected = select_final_charts(candidates, df, profile_map, max_charts)

    # 5. Render
    logger.info(f"Rendering {len(selected)} charts → {out_dir}/")
    render_warnings = render_charts(selected, df, profile_map, out_dir)

    # 6. Build analysis warnings
    analysis_warnings: List[str] = list(render_warnings)
    for p in profiles:
        if not p.is_usable:
            continue
        if p.missing_fraction > 0.3:
            analysis_warnings.append(
                f"'{p.name}' has {p.missing_fraction:.0%} missing values — "
                f"charts may not be representative."
            )
        if p.is_numeric and p.semantic_type == NUMERIC_CONTINUOUS:
            col_data = df[p.name].dropna()
            if len(col_data) > 10:
                skew = col_data.skew()
                if abs(skew) > 2:
                    analysis_warnings.append(
                        f"'{p.name}' is highly skewed (skewness={skew:.1f}). "
                        f"Consider a log transform before modelling."
                    )

    # 7. Write reports
    logger.info("Writing reports …")
    write_report(input_path, df, profiles, selected, analysis_warnings, out_dir)

    logger.info(
        f"\nDone! {len(selected)} charts saved to '{out_dir}/'.\n"
        f"Reports: analysis_report.json / analysis_report.md"
    )

    # Return structured result (useful when called programmatically)
    return {
        "n_rows":       len(df),
        "n_cols":       len(df.columns),
        "n_charts":     len(selected),
        "output_dir":   str(out_dir),
        "warnings":     analysis_warnings,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="auto_viz",
        description="Intelligent automatic EDA visualizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python auto_viz.py --input data.csv
  python auto_viz.py --input survey.xlsx --output_dir charts/ --max_charts 25
  python auto_viz.py --input transactions.parquet --output_dir results/
        """,
    )
    p.add_argument("--input",       required=True,
                   help="Path to the dataset (CSV, TSV, Excel, or Parquet)")
    p.add_argument("--output_dir",  default="auto_viz_output",
                   help="Output directory for charts and reports (default: auto_viz_output/)")
    p.add_argument("--max_charts",  type=int, default=MAX_TOTAL_CHARTS,
                   help=f"Maximum charts to produce (default: {MAX_TOTAL_CHARTS})")
    return p


def main():
    args = _build_parser().parse_args()
    run_analysis(args.input, args.output_dir, args.max_charts)


if __name__ == "__main__":
    main()
