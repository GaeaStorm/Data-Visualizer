# auto_viz

Intelligent automatic EDA visualizer. Give it any tabular dataset; it behaves like a thoughtful analyst — inspecting column types, deciding which charts are actually meaningful, and outputting a curated set of plots with a written rationale for each one.

---

## What it does

1. **Loads** your dataset (CSV, TSV, Excel, or Parquet).
2. **Profiles every column** — infers semantic type (continuous, discrete, categorical, datetime, identifier, target/outcome, currency, percentage, etc.) using dtype, column names, cardinality, and value patterns.
3. **Generates candidate charts** using explicit heuristics:
   - Univariate: histograms + KDE for numerics; bar charts for categoricals.
   - Bivariate: scatter/hexbin for numeric pairs; box/violin for numeric × categorical; contingency heatmaps for categorical pairs.
   - Time series: line charts and grouped trends when a datetime column exists.
   - Multivariate: correlation heatmap and pairplot for numeric-heavy datasets.
   - Missingness: bar chart when ≥ 2 columns have notable missing data.
4. **Scores and selects** the most useful subset (default cap: 40 charts), avoiding redundancy and chart spam.
5. **Renders** all selected charts as PNG files.
6. **Writes** a report in both JSON and Markdown explaining every decision.

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl pyarrow
```

`openpyxl` is needed for Excel support; `pyarrow` for Parquet. Both are optional if you only use CSV.

---

## Usage

### Basic

```bash
python auto_viz.py --input data.csv
```

Outputs charts and reports to `auto_viz_output/`.

### Custom output directory

```bash
python auto_viz.py --input data.csv --output_dir results/
```

### Limit chart count

```bash
python auto_viz.py --input survey.xlsx --output_dir charts/ --max_charts 20
```

### Supported file types

| Extension | Notes |
|---|---|
| `.csv` | Default comma separator; latin-1 fallback if UTF-8 fails |
| `.tsv` | Tab-separated |
| `.xlsx` / `.xls` | First sheet only |
| `.parquet` | Requires `pyarrow` |

---

## Output

After running, the output directory contains:

```
output_dir/
├── uni_hist_001.png       # Histogram + KDE for a numeric column
├── uni_cat_002.png        # Bar chart for a categorical column
├── biv_nn_003.png         # Scatter or hexbin for two numerics
├── biv_nc_004.png         # Violin or box plot for numeric × categorical
├── biv_cc_005.png         # Contingency heatmap for two categoricals
├── corr_006.png           # Correlation heatmap (all numerics)
├── pair_007.png           # Pairplot
├── ts_008.png             # Time series line chart
├── miss_009.png           # Missing data bar chart
├── analysis_report.json   # Machine-readable full report
└── analysis_report.md     # Human-readable summary
```

`analysis_report.md` lists every chart with its rationale and usefulness score, every skipped column with the skip reason, and any data quality warnings (high skewness, high missingness, etc.).

---

## Examples

### Titanic dataset

```bash
python auto_viz.py --input titanic.csv --output_dir titanic_charts/
```

Expected output:
- Histograms for `Age`, `Fare`
- Bar charts for `Sex`, `Pclass`, `Embarked`
- Violin/box plots for `Age by Survived`, `Fare by Pclass`, etc.
- Correlation heatmap across numeric columns
- `PassengerId` skipped as identifier

### Sales data with dates

```bash
python auto_viz.py --input sales.csv --output_dir sales_viz/
```

Expected output:
- Line chart: revenue over time
- Grouped time series: revenue over time by region
- Box plots: revenue by product category

### Survey data (mostly categorical)

```bash
python auto_viz.py --input survey.xlsx --output_dir survey_viz/ --max_charts 25
```

Expected output:
- Bar charts for each question
- Contingency heatmaps for related question pairs

---

## Programmatic usage

`run_analysis()` returns a summary dict and writes the same artifacts — no CLI needed:

```python
from auto_viz import run_analysis

result = run_analysis(
    input_path="data.csv",
    output_dir="results/",
    max_charts=30,
)
print(f"Generated {result['n_charts']} charts.")
print("Warnings:", result["warnings"])
```

This makes it straightforward to call from a Jupyter notebook, a web backend, or a batch processing script.

---

## Configuration

Key constants at the top of `auto_viz.py` can be tuned without touching logic:

| Constant | Default | Effect |
|---|---|---|
| `MAX_TOTAL_CHARTS` | 40 | Hard cap on output charts |
| `MAX_BIVARIATE_PAIRS` | 20 | Cap on scatter/hexbin candidates |
| `MAX_SUBGROUP_CHARTS` | 12 | Cap on box + violin charts combined |
| `TOP_N_CATEGORIES` | 15 | Bars shown in a Top-N bar chart |
| `MAX_PAIRPLOT_COLS` | 6 | Pairplot generated only when ≤ this many numerics |
| `MIN_ROWS_FOR_HEXBIN` | 1000 | Switch scatter → hexbin above this row count |
| `CORRELATION_THRESHOLD` | 0.25 | Min |r| to give a scatter a score bonus |
| `MISSINGNESS_THRESHOLD` | 0.01 | Min fraction missing to include a column in the missingness chart |
| `DPI` | 120 | Output image resolution |

---

## Interpreting the report

Each entry in `analysis_report.md` looks like:

```
### Age by Survived
- Type: `violin`
- Columns: `Age`, `Survived`
- Score: 8.0 / 10
- Rationale: Violin plot shows full distribution of 'Age' across 2 groups
  of 'Survived'. 'Survived' matches target/outcome patterns.
- File: `biv_nc_004.png`
```

**Score** is 0–10, where:
- 8–10: high-value chart (time series, correlation heatmap, target-related)
- 5–7: useful chart (well-correlated scatter, clean subgroup comparison)
- < 5: lower-priority chart (weak correlation, many categories)

Charts with scores below the cutoff are excluded automatically.

---

## Limitations

- Geographic columns (lat/lon, country, city) are detected but not plotted — no map renderer is included.
- Non-linear relationships with low Pearson r may be underscored.
- Very wide datasets (> 50 columns) may generate many candidates; lower `--max_charts` if needed.
- Excel files with multiple sheets: only the first sheet is read.
- Free-text columns (long strings) are skipped entirely.

See `Architecture.md` for the full heuristic framework and extension guide.
