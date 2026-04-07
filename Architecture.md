# auto_viz — Architecture & Design Reference

## Overview

`auto_viz.py` is a single-file, modular EDA pipeline. It ingests an arbitrary tabular dataset, reasons about every column's semantic role, generates a scored list of candidate charts, selects the most useful subset, renders them, and writes a structured report. The design goal is *analyst-quality curation*, not chart quantity.

---

## Pipeline Stages

```
DataLoader
    │  loads CSV / TSV / Excel / Parquet → pd.DataFrame
    ▼
profile_columns()
    │  per-column: dtype + name + cardinality + value patterns
    │  → List[ColumnProfile]  (semantic_type, is_usable, skip_reason, …)
    ▼
ChartCandidateGenerator.generate_all()
    │  heuristic rules → List[ChartCandidate]
    │  (no rendering — pure decision objects)
    ▼
select_final_charts()
    │  score each candidate, sort descending,
    │  deduplicate by (columns, chart_type), apply per-type caps
    │  → List[ChartCandidate]  (selected subset)
    ▼
render_charts()
    │  dispatch each ChartCandidate to its renderer fn
    │  → PNG files in output_dir/
    ▼
write_report()
    └  analysis_report.json + analysis_report.md
```

The public entry point is `run_analysis(input_path, output_dir, max_charts)` — a plain callable with no CLI dependencies. This makes it trivial to wrap in a web endpoint (see "Web Readiness" below).

---

## Semantic Type System

Beyond pandas dtypes, each column is assigned one of these semantic roles:

| Type | Description | Key detection signals |
|---|---|---|
| `numeric_continuous` | Float or high-unique-count numeric | Large unique count, non-integer |
| `numeric_discrete` | Integer with few unique values | Integer dtype, ≤ 20 unique values |
| `binary` | Two-valued column | `n_unique == 2` |
| `boolean` | Native bool or yes/no/true/false | dtype bool or matching value set |
| `ordinal_categorical` | Ordered categories | Name hints: rank, tier, level, grade |
| `nominal_categorical` | Unordered categories | String dtype, low/medium cardinality |
| `datetime` | Time/date values | Native datetime dtype or parseable values |
| `identifier` | ID / key column with no analytical value | Name pattern (id, uuid, key…) + high cardinality |
| `text_freeform` | Free text | Avg string length > 60 chars |
| `likely_target` | Outcome / label column | Name matches outcome vocabulary |
| `percentage_rate` | Proportions or percentages | Name + 0–1 or 0–100 range |
| `currency` | Monetary values | Name contains price, cost, salary, etc. |
| `geographic` | Location fields | Name contains lat, lon, city, country, etc. |

Detection uses four independent signals in order of priority:
1. **Native dtype** (datetime, bool)
2. **Column name patterns** (regex against known vocabulary)
3. **Cardinality ratio** (n_unique / n_rows)
4. **Value patterns** (range checks, UUID match, datetime parse attempt)

Columns typed as `identifier`, `text_freeform`, or `geographic` are flagged `is_usable=False` and skipped globally. Columns with < 2 unique values or > 95% missing are also skipped.

---

## Chart Selection Heuristics

### Univariate

| Column type | Chart generated |
|---|---|
| Any numeric | Histogram + KDE (always) |
| Categorical, ≤ 20 unique | Count bar chart |
| Categorical, > 20 unique | Top-N bar chart (top 15 by frequency) |

### Bivariate: Numeric × Numeric

- Compute Pearson r for every pair.
- Use **scatter** (< 1,000 rows) or **hexbin** (≥ 1,000 rows).
- Correlation threshold: |r| ≥ 0.25 scores higher; all pairs are still candidates but ranked lower.
- Cap: `MAX_BIVARIATE_PAIRS` (default 20) to prevent explosion.

### Bivariate: Numeric × Categorical

- For every "good groupby" column (categorical, 2–12 unique values, not an identifier):
  - ≤ 8 groups → **violin** (full distribution visible)
  - > 8 groups → **box** (cleaner at higher cardinality)
- Cap: `MAX_SUBGROUP_CHARTS` (default 12) across all box/violin charts.

### Bivariate: Categorical × Categorical

- Only when both columns have ≤ 10 unique values.
- Generates a **contingency heatmap** (row-normalized proportions).

### Datetime × Numeric

- **Time series line chart** for every (datetime, numeric) pair.
- **Grouped time series** conditioned on up to 2 "good groupby" columns.
- Rolling average window auto-calculated from series length.

### Multivariate

| Condition | Chart |
|---|---|
| ≥ 3 numeric columns | Correlation heatmap (lower triangle) |
| 3–6 numeric columns | Pairplot (sampled to 2,000 rows if large) |
| ≥ 2 columns with ≥ 1% missing | Missingness bar chart |

---

## Usefulness Scoring

`score_chart_usefulness()` assigns each candidate a score in [0, 10]:

| Signal | Effect |
|---|---|
| Baseline | +5.0 |
| `correlation_heatmap` | Fixed 8.5 |
| `time_series` / `time_series_grouped` | Fixed 8.0 |
| `pairplot` | 7.5 (8.0 with hue) |
| Scatter/hexbin: |r| per unit | +3.5 × |r| |
| Column typed as `likely_target` | +1.5 (once per chart) |
| Binary grouping variable | +1.0 (clean two-group comparison) |
| Categorical variable with > 12 groups in box/violin | −2.5 |
| Any column with < 3 unique values | −1.5 per column |
| Missingness chart: total missing across all columns | +2× total_fraction |

Candidates are sorted descending, then selected greedily up to `max_charts` with deduplication on `(frozenset(columns), chart_type)`.

---

## Redundancy Controls

- **Deduplication key**: `(frozenset of column names, chart_type)` — no two identical charts.
- **Per-type caps**: box/violin capped at 12; scatter/hexbin capped at 20.
- **Total cap**: `--max_charts` flag (default 40).
- **Score-based pruning**: low-scoring candidates drop out before the cap is reached.
- **Column skipping**: identifiers, freeform text, and geographic columns are excluded at profiling time.

---

## Output Artifacts

```
output_dir/
├── uni_hist_001.png            # distribution charts
├── biv_nn_002.png              # scatter / hexbin
├── biv_nc_003.png              # box / violin
├── corr_004.png                # correlation heatmap
├── pair_005.png                # pairplot
├── ts_006.png                  # time series
├── miss_007.png                # missingness bar
├── analysis_report.json        # machine-readable full report
└── analysis_report.md          # human-readable summary
```

`analysis_report.json` structure:
```json
{
  "dataset": "...",
  "n_rows": ..., "n_cols": ...,
  "total_charts_generated": ...,
  "warnings": [...],
  "column_profiles": [{ "name", "semantic_type", "n_unique", "missing_fraction", "is_usable", ... }],
  "charts":          [{ "chart_id", "chart_type", "columns", "title", "rationale", "usefulness_score", "output_path" }],
  "skipped_columns": [{ "name", "reason" }]
}
```

---

## Web Readiness

The `run_analysis()` function is deliberately isolated from CLI concerns. A minimal Flask endpoint would be:

```python
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile, os
from auto_viz import run_analysis

app = Flask(__name__)

@app.post("/analyze")
def analyze():
    f    = request.files["dataset"]
    tmp  = tempfile.mkdtemp()
    path = os.path.join(tmp, secure_filename(f.filename))
    f.save(path)
    result = run_analysis(path, os.path.join(tmp, "results"))
    return jsonify(result)
```

For a production web tool, additional work would include:
- Async job queue (Celery / RQ) for large datasets
- Serving chart images from object storage (S3 / GCS)
- Streaming progress via WebSockets or SSE
- Per-user output directories and cleanup

---

## Assumptions and Limitations

### Assumptions
- **Tabular data with a header row.** Multi-index or melted formats are not handled.
- **Reasonable column names.** Semantic inference relies heavily on name patterns; single-letter or UUID column names won't get type overrides.
- **Single-sheet Excel files.** Only the first sheet is read.
- **Numeric-ness is pandas-detectable.** Numeric columns stored as object dtype (e.g., "1,234.56") are not coerced — pre-clean if needed.

### Limitations
- **No map/geo rendering.** Geographic columns are detected but skipped.
- **No interactive charts.** All output is static PNG. Plotly integration would require a backend swap (the dispatch table `_RENDERERS` is the right place to swap it).
- **No ML-driven insight.** Feature importance, cluster labels, and anomaly detection are not included; the tool is purely descriptive/EDA.
- **Correlation is Pearson only.** Non-linear relationships (e.g., U-shaped) that have near-zero r will score lower than they deserve.
- **Time series resampling is daily.** Datasets with sub-daily or multi-year spans may benefit from a different resampling frequency.
- **Pairplot skipped above 6 numeric columns.** A 7×7 pairplot is illegible; consider selecting a subset manually for large datasets.
- **Missing data visualization is simple.** For complex missingness patterns (MCAR vs MAR), the `missingno` library would add value.

### Extension Points
| What to extend | Where |
|---|---|
| Add a new semantic type | `infer_semantic_type()` |
| Add a new chart type | `ChartCandidateGenerator` + `_RENDERERS` dict |
| Change scoring weights | `score_chart_usefulness()` |
| Change selection caps | module-level constants at the top of the file |
| Swap matplotlib → Plotly | Replace renderer functions, keep everything else |
