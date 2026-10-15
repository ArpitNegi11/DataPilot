# DataPilot – AI Data Analyst & Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple)
![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory%20SQL-yellow)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent%20Orchestrator-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Milestone-80%2B%25%20Completed-success)

---

## 🚀 Overview (80%+ Milestone)

**DataPilot** is an enterprise-grade AI Data Analyst and Recommendation Engine combining an interactive **Streamlit web GUI**, an automated **CLI batch pipeline**, a **LangGraph multi-agent orchestrator**, and an **in-memory DuckDB analytical engine**.

At this **80%+ Milestone**, DataPilot moves beyond basic dashboarding into an intelligent, autonomous analytical partner. It connects structured tabular data (CSV/Excel) with unstructured corporate governance documents (PDF/DOCX/TXT), detects operational anomalies and loss-making SKUs, runs conversational natural-language queries without mathematical hallucination, and generates prioritized, evidence-backed business recommendations.

---

## 🎯 80%+ Milestone Deliverables & Architecture

| Phase / Component | Milestone Status | Core Capabilities |
|---|---|---|
| **Phase 0: Audit & Architecture** | ✅ Verified | Baseline capabilities verified across 8 sample datasets; clean architectural boundaries established in `docs/ARCHITECTURE.md`. |
| **Phase 1: Analytics Foundation** | ✅ Verified | Header banner auto-detection, resilient date cleaning, config schema checks, and deterministic KPI computations in `src/analytics.py`. |
| **Phase 2: Natural Language Analytics** | ✅ Verified | DuckDB SQL translation in `src/query_engine.py` for date filters, min/max, averages, rankings, dimensional breakdowns, and raw SQL pass-through. |
| **Phase 3: Business Document RAG** | ✅ Verified | `src/document_rag.py` ingests `.pdf` (`pypdf`), `.docx` (`python-docx`), `.txt`, and `.md` with semantic chunking and TF-IDF cosine similarity search. |
| **Phase 4: Hybrid Query Engine** | ✅ Verified | `src/hybrid_engine.py` classifies intent (`data_only`, `document_only`, `hybrid`, `conversational`) to synthesize quantitative DuckDB numbers with qualitative policies. |
| **Phase 5: LangGraph Multi-Agent Orchestration** | ✅ Verified | `src/orchestrator.py` implements a `StateGraph` workflow featuring intent routing, SQL execution, validation guardrails, and automated retry loops. |
| **Phase 6: Prescriptive Recommendations** | ✅ Verified | `src/recommendations.py` automatically flags loss-making SKUs, discount margin dilution, and customer revenue concentration (>35%). |
| **Phase 7: Full Application Integration** | ✅ Verified | Redesigned `app_streamlit.py` into a modern 5-tab interface; upgraded `templates/report_template.html` and `src/reporting.py` with KPI cards & recommendation boxes. |
| **Phase 8: Automated Regression Suite** | ✅ Verified | 26 automated unit, integration, and UI tests in `tests/` passing with a 100% success rate. |
| **Multi-Provider LLM Integration** | ✅ Verified | `src/llm_client.py` supports Google Gemini (`gemini-3.5-flash`), OpenAI (`gpt-4o-mini`), and Ollama (`llama3.2`) with candidate cascade and human-analyst persona. |

---

## ✨ Key System Features

### 1. 🤖 LangGraph Multi-Agent Orchestrator (`src/orchestrator.py`)
- Coordinates the query lifecycle across intent routing, SQL generation, execution, mathematical validation, and human-like synthesis.
- **Self-Correcting Execution**: Validates DuckDB results and automatically triggers retry loops if an anomaly or execution issue occurs.

### 2. 🛡️ Strict Zero-Hallucination Design
- All quantitative statistics, KPIs, and analytical numbers are computed deterministically by **DuckDB** or **Pandas** on the actual uploaded dataset.
- LLMs are used strictly for **natural language translation** and **conversational synthesis**, citing verified figures directly from the database.

### 3. 📄 Business Document RAG (`src/document_rag.py`)
- Ingests corporate policy guidelines, annual review memos, and pricing rules across PDF, DOCX, TXT, and Markdown formats.
- Retrieves relevant excerpts with exact source citations and relevance scores.

### 4. 🔀 Hybrid Quantitative & Qualitative Questions (`src/hybrid_engine.py`)
- Handles complex business questions requiring both numbers and context (e.g. *"What is our total revenue from high-discount orders, and does this violate our sales pricing policy?"*).

### 5. 🎯 Data-Grounded Recommendation Engine (`src/recommendations.py`)
- Automatically assesses:
  - **Loss-Making SKUs**: Identifies products generating negative profit despite sales volume.
  - **Revenue Concentration Risk**: Flags single categories or segments contributing >35% of total revenue.
  - **Discount Margin Erosion**: Detects aggressive discounting strategies eroding EBITDA.
  - Categorizes actions by priority (**High**, **Medium**, **Low**) with explicit data evidence.

### 6. 📊 5-Tab Streamlit Web Application (`app_streamlit.py`)
- **Tab 1: 📊 Analytics & Visuals**: Core KPI metric cards, data quality health score, and interactive Plotly charts.
- **Tab 2: 💬 AI Analyst (LangGraph)**: Multi-agent conversational interface with live DuckDB SQL inspection and verified citations.
- **Tab 3: 🎯 Recommendations**: Prioritized, actionable business strategies grounded in detected anomalies.
- **Tab 4: 📚 Document Knowledge Base**: Searchable corporate document repository with real-time passage retrieval.
- **Tab 5: 📄 Export & Reports**: Instant downloads for cleaned datasets (CSV), styled HTML executive reports, and PDF documents.

---

## 📂 Project Structure

```text
DataPilot/
├── app_streamlit.py              # Streamlit Web GUI Application (5-Tab Modern UI)
├── requirements.txt              # Core package dependencies
├── .env.example                  # Environment configuration template
├── .gitignore                    # Comprehensive Git exclusion rules
│
├── src/                          # Core application modules
│   ├── analytics.py              # Business analytics, KPI calculators & column role inference
│   ├── anomalies.py              # Outlier & anomaly detection (IQR)
│   ├── cleaning.py               # Automated cleaning, imputation & resilient date parsing
│   ├── document_rag.py           # Document parser (PDF/DOCX/TXT), chunking & semantic search
│   ├── hybrid_engine.py          # Unified hybrid routing & multi-source synthesis
│   ├── io_utils.py               # Smart tabular ingestion (CSV/Excel header auto-detection)
│   ├── llm_client.py             # Multi-provider LLM client (Gemini 3.5, OpenAI, Ollama)
│   ├── main.py                   # Terminal CLI pipeline entry point
│   ├── orchestrator.py           # LangGraph StateGraph multi-agent orchestrator
│   ├── predictive.py             # ML models (Linear Regression, KMeans clustering)
│   ├── query_engine.py           # DuckDB natural language query engine & SQL translator
│   ├── recommendations.py        # Prescriptive business recommendation engine
│   ├── reporting.py              # Executive HTML & PDF report generation (wkhtmltopdf)
│   ├── validation.py             # Schema & rule-based validation
│   └── visualize.py              # Plotly interactive dashboards & static figures
│
├── config/
│   └── config.yaml               # Pipeline configuration & schema validation rules
│
├── sample_data/                  # Built-in demonstration datasets
│   ├── Enterprise_SaaS_Revenue_2023_2024.csv    # 750 SaaS subscriber accounts & churn scores
│   ├── Global_Retail_Omnichannel_Sales.csv      # 1,000 retail transactions & profit margins
│   ├── Healthcare_Pharma_Supply_Chain.csv       # 650 pharmaceutical batches & defect audits
│   ├── DataPilot_Sales_Demo.csv                 # E-commerce transactional demo
│   └── sample.xlsx                              # Baseline workbook sample
│
├── sample_documents/             # Business strategy memos & governance policies for RAG
│   ├── FY2023_Corporate_Strategy_Review.txt
│   ├── Sales_Discount_and_Pricing_Policy.txt
│   ├── SaaS_Enterprise_Expansion_Memo_2024.txt
│   └── Pharma_Quality_and_Compliance_Guideline.txt
│
├── templates/
│   └── report_template.html      # Jinja2 executive HTML report template
│
├── tests/                        # Automated test suite (26 passing tests)
│   ├── test_analytics.py
│   ├── test_anomalies.py
│   ├── test_app_ui.py
│   ├── test_cleaning.py
│   ├── test_document_rag.py
│   ├── test_hybrid_engine.py
│   ├── test_io_utils.py
│   ├── test_llm_client.py
│   ├── test_orchestrator.py
│   ├── test_pipeline_cli.py
│   ├── test_query_engine.py
│   ├── test_recommendations.py
│   └── test_validation.py
│
└── install_dependencies/
    └── wkhtmltox-0.12.6-1.msvc2015-win64.exe  # Bundled Windows installer for wkhtmltopdf
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository & Setup Virtual Environment
```powershell
# Clone the repository
git clone https://github.com/your-username/DataPilot.git
cd DataPilot

# Create and activate virtual environment
python -m venv myenv
myenv\Scripts\activate
```

### 2️⃣ Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables (Optional for LLM Intelligence)
Copy the template file `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Add your preferred API key in `.env`:
```env
# Google Gemini API Key (https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_key_here

# OpenAI API Key (https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_key_here

# Local Ollama endpoint (default)
OLLAMA_BASE_URL=http://localhost:11434/v1
```
*Note: If no API key is provided, DataPilot runs automatically on its high-speed deterministic DuckDB engine with zero configuration required.*

### 4️⃣ Install wkhtmltopdf (For PDF Report Exports)
1. Run the bundled installer located in `install_dependencies/wkhtmltox-0.12.6-1.msvc2015-win64.exe`.
2. Install to the default directory (`C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe`).

---

## ▶️ How to Run

### 🌐 1. Interactive Streamlit Web GUI (Recommended)
```powershell
myenv\Scripts\streamlit.exe run app_streamlit.py
```
Open your browser at `http://localhost:8501`:
1. Select one of the built-in demo datasets (SaaS, Retail, or Pharma) or upload your own CSV/Excel file.
2. In the sidebar, view the auto-detected LLM mode (or switch to Gemini, OpenAI, Ollama, or Deterministic) and click **"🧪 Test LLM Connection"**.
3. Explore the 5 UI tabs for analytics, LangGraph querying, recommendations, document RAG, and report exports.

### 💻 2. Batch CLI Pipeline
Execute the complete end-to-end cleaning, validation, analytics, and reporting pipeline from the terminal:
```powershell
myenv\Scripts\python.exe src/main.py --input sample_data/Enterprise_SaaS_Revenue_2023_2024.csv --outdir outputs --config config/config.yaml
```

CLI Arguments:
- `--input` *(required)*: Path to input CSV or Excel workbook.
- `--outdir` *(optional)*: Output directory for cleaned CSVs, charts, and reports (defaults to `outputs`).
- `--config` *(optional)*: Path to YAML configuration (defaults to `config/config.yaml`).

---

## 🧪 Automated Testing

DataPilot includes an automated test suite verifying every component from data cleaning and RAG to LangGraph orchestration and Streamlit UI rendering:

```powershell
myenv\Scripts\python.exe -m pytest -v
```

**Status**: 26/26 tests passing (100% pass rate).

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
