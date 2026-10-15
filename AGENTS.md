# DataPilot Developer & Agent Guidelines

## 1. Project Overview
DataPilot is an AI Data Analyst & Recommendation Engine combining a Streamlit web GUI and a CLI pipeline. It processes tabular business data (CSV and Excel) for cleaning, validation, anomaly detection, DuckDB deterministic queries, interactive visualizations, and PDF/HTML reporting.

---

## 2. Environment & Execution Conventions

- **Python Virtual Environment**:
  - Always use the dedicated virtual environment located at `myenv\Scripts\python.exe`.
  - When running terminal commands, invoke:
    ```powershell
    myenv\Scripts\python.exe <script.py>
    # or
    myenv\Scripts\streamlit.exe run app_streamlit.py
    ```
- **Operating System**: Windows (PowerShell shell environment).
- **wkhtmltopdf Path**:
  - Executable path on Windows: `C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe`.
  - Bundled installer located in `install_dependencies/wkhtmltox-0.12.6-1.msvc2015-win64.exe`.

---

## 3. Core Development Principles

1. **Do Not Rewrite Working Features**:
   - Inspect existing implementations before making changes.
   - Refactor or augment incrementally rather than replacing working code wholesale.
2. **Deterministic Computations (Zero Hallucination)**:
   - Analytical numbers, aggregates, and KPIs must come directly from DuckDB or Pandas calculations on the actual dataset.
3. **Smart Ingestion First**:
   - Always load files through `src/io_utils.py:load_file()` rather than `pd.read_excel` or `pd.read_csv` directly, ensuring header banner detection and formatting resilience.
4. **Phase Discipline**:
   - Implement one phase at a time as outlined in [docs/IMPLEMENTATION_PLAN.md](file:///d:/codes/DataPilot/docs/IMPLEMENTATION_PLAN.md).
   - Keep [docs/PROGRESS.md](file:///d:/codes/DataPilot/docs/PROGRESS.md) synchronized after each phase.
5. **Testing Verification**:
   - Verify changes against both the CLI pipeline and the Streamlit application using sample datasets in `sample_data/`.