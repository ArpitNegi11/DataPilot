# DataPilot Architecture & System Overview

## 1. System Summary
DataPilot is a hybrid Python/Streamlit data analysis and recommendation engine designed to ingest, clean, validate, analyze, explore, and report on business tabular datasets (sales, orders, subscriptions, inventory, project tasks) in CSV and Excel formats.

The system is structured as a dual-interface application:
1. **Interactive Web GUI (`app_streamlit.py`)**: End-to-end interactive exploration, automated profiling, data cleaning, KPI inference, interactive Plotly visualizations, ML predictive insights, natural language querying via DuckDB, and exportable reports.
2. **Headless CLI Pipeline (`src/main.py`)**: Batch script executing loading, cleaning, validation, anomaly detection, static visualization, ML modeling, and Jinja2 HTML/PDF report rendering.

---

## 2. High-Level Architecture & Component Map

```
┌────────────────────────────────────────────────────────┐
│                   Input Datasets                       │
│      (.xlsx, .xls, .csv / sample_data / file upload)    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               I/O & Ingestion (io_utils.py)             │
│  - Smart title banner & metadata row detection         │
│  - Empty column/row stripping                          │
│  - Format auto-detection & encoding fallbacks          │
└──────────────────────────┬─────────────────────────────┘
                           │ Cleaned DataFrame
                           ▼
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│ Cleaning (cleaning.py)  │         │ Validation              │
│ - Duplicate removal     │         │ (validation.py)         │
│ - Missing imputation    │         │ - Schema limits         │
│ - Text standardization  │         │ - Min/Max/Regex/Unique  │
│ - Date formatting       │         └─────────────────────────┘
└───────────┬─────────────┘
            │
            ├────────────────────────────────────────┬───────────────────────────────────────┐
            ▼                                        ▼                                       ▼
┌────────────────────────┐              ┌────────────────────────┐              ┌────────────────────────┐
│ Anomalies              │              │ Analytics & Semantic   │              │ Predictive ML          │
│ (anomalies.py)         │              │ Inference              │              │ (predictive.py)        │
│ - Z-Score detection    │              │ (analytics.py)         │              │ - Linear Regression    │
│ - Numerical outlier    │              │ - Column role detector │              │ - KMeans Clustering    │
│   flagging             │              │   (rev/profit/date/...)│              │ - Matplotlib chart gen │
└───────────┬────────────┘              └───────────┬────────────┘              └───────────┬────────────┘
            │                                       │                                       │
            ├───────────────────────────────────────┼───────────────────────────────────────┘
            ▼                                       ▼
┌────────────────────────┐              ┌────────────────────────┐
│ Visualizations         │              │ NL Query Engine        │
│ (visualize.py)         │              │ (query_engine.py)      │
│ - Matplotlib static    │              │ - In-memory DuckDB     │
│   (Report generation)  │              │ - Deterministic SQL    │
│ - Plotly interactive   │              │   query templates      │
│   (Streamlit app)      │              │ - Zero hallucinations  │
└───────────┬────────────┘              └───────────┬────────────┘
            │                                       │
            └───────────────────┬───────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────┐
│                 Output & Reporting                     │
│  - Jinja2 HTML Executive Report (templates/report...)  │
│  - wkhtmltopdf automated PDF generation                │
│  - Streamlit Interactive Web Application UI            │
│  - Cleaned dataset download (.csv / .xlsx)             │
└────────────────────────────────────────────────────────┘
```

---

## 3. Module Inventory & Responsibilities

| Module | Primary Responsibility | Key Functions / Classes | Current Dependencies |
| :--- | :--- | :--- | :--- |
| `src/io_utils.py` | Smart ingestion and export of tabular files | `load_file()`, `detect_and_clean_header()`, `save_file()` | `pandas`, `openpyxl` |
| `src/cleaning.py` | Data normalization and cleaning pipeline | `clean_data()`, `load_data()` | `pandas`, `numpy` |
| `src/validation.py` | Rule-based schema checking | `validate_data()` | `pandas`, `re` |
| `src/anomalies.py` | Outlier detection across numeric fields | `detect_anomalies()` | `scipy.stats`, `numpy` |
| `src/analytics.py` | Business metadata and column role inference | `infer_business_columns()` | `pandas`, `numpy` |
| `src/predictive.py` | ML models (Regression & KMeans) | `run_predictive_models()`, `_fig_to_base64()` | `scikit-learn`, `matplotlib` |
| `src/query_engine.py` | Deterministic natural language SQL query engine | `DataPilotQueryEngine`, `ask()`, `execute_sql()` | `duckdb`, `pandas` |
| `src/reporting.py` | Jinja2 HTML report and wkhtmltopdf generation | `generate_report()` | `jinja2`, `pdfkit`, `wkhtmltopdf` |
| `src/visualize.py` | Static charts (matplotlib/seaborn) & interactive charts (plotly) | `generate_visuals()`, `generate_interactive_visuals()` | `plotly.express`, `seaborn`, `matplotlib` |
| `src/main.py` | Headless CLI batch execution script | `run_pipeline()`, `load_config()`, `main()` | `argparse`, `yaml`, `pandas` |
| `app_streamlit.py` | Web-based interactive dashboard application | Full GUI layout & orchestration | `streamlit`, Plotly, and internal `src` modules |

---

## 4. Data Flow

### 4.1. Streamlit Web Flow (`app_streamlit.py`)
1. **Source Selection**: User uploads `.csv`/`.xlsx` or selects one of the 8 sample datasets from `sample_data/`.
2. **Ingestion & Profiling**: `io_utils.load_file()` reads the file, optionally stripping title/banner rows and blank columns. Streamlit renders summary metrics (row count, column count, missing value count, duplicates).
3. **Cleaning Pipeline**: `cleaning.clean_data()` converts numeric string columns, drops duplicates, imputes missing values (median for numerics, "Unknown" for objects), applies text standardization (title case), and formats dates (`%Y-%m-%d`).
4. **Validation & Anomalies**: `validation.validate_data()` validates against rules; `anomalies.detect_anomalies()` computes Z-score outliers ($|Z| > 3.0$).
5. **Interactive Visualizations**: `visualize.generate_interactive_visuals()` produces Plotly figures:
   - Missing values percentage horizontal bar chart
   - Top products / items bar chart
   - Category / segment breakdown donut chart
   - Monthly trend line chart
   - Correlation heatmap
   - Numeric feature distribution with box marginal
6. **Predictive Modeling**: `predictive.run_predictive_models()` fits Linear Regression if target pairs match and runs KMeans clustering (user-selectable $k \in [2, 6]$).
7. **Deterministic NL Querying**: User types or clicks natural language questions. `DataPilotQueryEngine` registers the cleaned DataFrame in an in-memory DuckDB instance and runs SQL queries, returning exact aggregates and dataframes.
8. **Export**: Renders HTML executive summary using Jinja2 and converts to PDF via `wkhtmltopdf` if present.

### 4.2. CLI Flow (`src/main.py`)
1. Reads input file via CLI `--input`, config via `--config`, outputs to `--outdir`.
2. Executes clean $\rightarrow$ validate $\rightarrow$ anomalies $\rightarrow$ static visuals $\rightarrow$ predictive models $\rightarrow$ outputs `cleaned.xlsx`, `report.html`, and `report.pdf`.

---

## 5. Architectural Strengths & Existing Assets
1. **Zero Hallucination SQL Engine**: `query_engine.py` executes directly on in-memory DuckDB, computing real numbers rather than guessing.
2. **Smart Header Detection**: `io_utils.py` handles real-world spreadsheet anomalies like title banners and blank rows.
3. **Rich Visualizations**: Dual visualization system supports high-interactivity Plotly for UI and static matplotlib figures for offline reports.
4. **Pre-configured Offline PDF Tooling**: `wkhtmltopdf` binary is bundled and installed, enabling automated offline PDF generation out-of-the-box.
5. **Rich Sample Data**: 8 comprehensive real-world business datasets are available in `sample_data/` covering sales, inventory, subscriptions, and project tracking.

---

## 6. Known Architectural Gaps & Technical Debt
1. **Missing Test Suite**: Zero unit or integration tests exist in the codebase.
2. **CLI Incompatibility with CSV**: `src/main.py` uses `pd.read_excel(input_file)` instead of `io_utils.load_file(input_file)`, crashing when passed CSV files.
3. **Config Schema Mismatch**:
   - `config/config.yaml` uses `validation_rules:` while `main.py` checks `config.get("validation", [])`.
   - `config/config.yaml` specifies `dropna_threshold: 0.5` and `fillna: ...` whereas `cleaning.py` searches for `config.get("fill_missing", {}).get("threshold")`.
   - `app_streamlit.py` passes `config.get("cleaning", {})` (which is empty in the current config).
4. **Analytics Module Underdeveloped**: `src/analytics.py` only implements column role inference (`infer_business_columns`); it lacks standalone business KPI metrics, margin calculations, and cohort logic.
5. **Hardcoded Predictive Relationships**: `predictive.py` hardcodes exact column names ("Experience"/"Salary", "Quantity"/"Sales", "Units"/"Total") rather than using inferred column roles.
6. **Streamlit Deprecation Warnings**: Uses `use_container_width=True` on dataframes and buttons (deprecated in modern Streamlit versions).
