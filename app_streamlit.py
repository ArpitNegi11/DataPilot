import streamlit as st
import os
import pandas as pd
import numpy as np
import tempfile
import io

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src import cleaning, validation, anomalies, visualize, reporting, io_utils, predictive, analytics, recommendations
from src.main import load_config, extract_validation_rules
from src.document_rag import BusinessDocumentRAG
from src.orchestrator import DataPilotOrchestrator
from src.llm_client import LLMClient

# -----------------------------------------------------------------------------
# 1. Page Configuration & Executive Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataPilot – AI Data Analyst & Recommendation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F1F5F9;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, #0d1527 0%, #080c16 60%, #04070d 100%);
    color: #F1F5F9;
}

/* Main Container Spacing */
.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1440px !important;
}

/* Sidebar Top Padding Shift Upwards */
section[data-testid="stSidebar"] {
    background: rgba(11, 17, 32, 0.96) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(24px) !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1.25rem !important;
    padding-left: 1.15rem !important;
    padding-right: 1.15rem !important;
}
[data-testid="stSidebarCollapseButton"] {
    margin-top: 0.2rem !important;
}

/* Header & Banner */
.main-title {
    font-size: 2.05rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(120deg, #FFFFFF 20%, #38BDF8 60%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.sub-title {
    font-size: 0.92rem;
    color: #94A3B8;
    margin: 6px 0 16px 0;
    line-height: 1.5;
}

/* Glass & Recommendation Cards */
.glass-card {
    background: rgba(17, 24, 39, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
    backdrop-filter: blur(16px);
}
.rec-box {
    background: rgba(30, 41, 59, 0.75);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    border-left: 4px solid #38bdf8;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 4px solid #38bdf8;
}
.rec-high { border-left-color: #ef4444 !important; background: rgba(239, 68, 68, 0.06); }
.rec-medium { border-left-color: #f59e0b !important; background: rgba(245, 158, 11, 0.06); }
.rec-low { border-left-color: #10b981 !important; background: rgba(16, 185, 129, 0.06); }
.badge { font-size: 0.75rem; font-weight: 700; padding: 2px 10px; border-radius: 12px; }

/* Tabs Styling */
div[data-testid="stTabs"] button[role="tab"] {
    font-size: 0.90rem;
    font-weight: 600;
    color: #94A3B8;
    padding: 9px 18px;
    border-radius: 8px;
    margin-right: 4px;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #38BDF8 !important;
    background: rgba(56, 189, 248, 0.12);
    border-bottom: 2px solid #38BDF8 !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
    color: #FFFFFF !important;
    font-weight: 600;
    border-radius: 9px;
    border: none;
    padding: 8px 18px;
    font-size: 0.88rem;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.25);
}

pre, code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.84rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# App Header
st.markdown(
    """
    <div style="margin-bottom:16px;">
        <div class="main-title">DataPilot – AI Data Analyst & Recommendation Engine</div>
        <div class="sub-title">Deterministic tabular analytics, LangGraph multi-agent querying, business document RAG & executive prescriptive recommendations.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = BusinessDocumentRAG()
    sample_doc_dir = "sample_documents"
    if os.path.exists(sample_doc_dir):
        st.session_state.rag_engine.add_directory(sample_doc_dir)

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:10px; margin-top:-0.25rem; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.08);">
        <span style="font-size:1.7rem;">⚡</span>
        <div>
            <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">DataPilot AI</div>
            <div style="font-size:0.72rem; color:#38BDF8; font-weight:600; letter-spacing:0.04em;">80%+ ENTERPRISE ENGINE</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### ⚙️ Settings & Pipeline")
enable_insights = st.sidebar.checkbox("Enable Predictive Insights & ML", value=True)
report_type = st.sidebar.radio("Executive Report Type", ["HTML", "PDF", "Both"], index=0)
num_clusters = st.sidebar.slider("Number of Clusters (KMeans)", min_value=2, max_value=6, value=3)

# LLM Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Natural Language Query Engine")

providers_list = ["Deterministic Engine (No API Key)", "Google Gemini", "OpenAI", "Ollama (Local)"]
default_idx = 0
if (os.getenv("GEMINI_API_KEY") or "").strip():
    default_idx = 1
elif (os.getenv("OPENAI_API_KEY") or "").strip():
    default_idx = 2

llm_provider = st.sidebar.selectbox(
    "Query Intelligence Mode",
    providers_list,
    index=default_idx,
    help="Deterministic uses pure DuckDB pattern matching. Gemini, OpenAI, or Ollama translate free-form natural language queries dynamically into DuckDB SQL."
)

llm_api_key = ""
llm_model = None
llm_base_url = None

if llm_provider == "Google Gemini":
    env_gemini = (os.getenv("GEMINI_API_KEY") or "").strip()
    if env_gemini:
        masked_suffix = env_gemini[-4:] if len(env_gemini) >= 4 else "****"
        st.sidebar.caption(f"🔒 API Key loaded from `.env` (`...{masked_suffix}`)")
        override_key = st.sidebar.text_input("Override Gemini API Key", type="password", value="", placeholder="Leave blank to use .env key")
        llm_api_key = override_key.strip() if override_key.strip() else env_gemini
    else:
        llm_api_key = st.sidebar.text_input("Gemini API Key", type="password", value="", placeholder="AIzaSy...")
    llm_model = st.sidebar.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"], index=0)

elif llm_provider == "OpenAI":
    env_openai = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_openai:
        masked_suffix = env_openai[-4:] if len(env_openai) >= 4 else "****"
        st.sidebar.caption(f"🔒 API Key loaded from `.env` (`...{masked_suffix}`)")
        override_key = st.sidebar.text_input("Override OpenAI API Key", type="password", value="", placeholder="Leave blank to use .env key")
        llm_api_key = override_key.strip() if override_key.strip() else env_openai
    else:
        llm_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value="", placeholder="sk-...")
    llm_model = st.sidebar.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)

elif llm_provider == "Ollama (Local)":
    llm_model = st.sidebar.text_input("Ollama Model", value="llama3.2")
    llm_base_url = st.sidebar.text_input("Ollama Base URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))

llm_client = LLMClient(
    provider=llm_provider,
    api_key=llm_api_key,
    model=llm_model,
    base_url=llm_base_url
)

if llm_client.is_configured():
    st.sidebar.success(f"🟢 LLM Active: {llm_provider}")
    if st.sidebar.button("🧪 Test LLM Connection", width="stretch"):
        with st.sidebar.status("Testing API connection..."):
            try:
                test_reply = llm_client.chat("Say 'Connection verified! DataPilot is ready to analyze your data.'")
                if test_reply:
                    st.sidebar.success(test_reply)
                else:
                    st.sidebar.error("❌ Failed to get response from provider.")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
else:
    st.sidebar.info("⚡ Fast Deterministic SQL Active")

# Data Source
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Tabular Dataset")
input_choice = st.sidebar.radio("Data Source", ["Use Sample Dataset", "Upload File (CSV / Excel)"], index=0)

auto_detect_headers = st.sidebar.checkbox(
    "Smart Header Auto-Detection",
    value=True,
    help="Automatically strips title banners, empty leading rows/columns, and detects true table headers."
)

uploaded_file = None
sample_file_path = None

if input_choice == "Upload File (CSV / Excel)":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
else:
    sample_files = [
        f for f in sorted(os.listdir("sample_data"))
        if f.endswith((".csv", ".xlsx", ".xls")) and not f.startswith("~$") and not f.startswith("create")
    ]
    if sample_files:
        chosen_sample = st.sidebar.selectbox("Choose Sample Dataset", sample_files, index=0)
        sample_file_path = os.path.join("sample_data", chosen_sample)
    else:
        st.sidebar.warning("No sample files found in sample_data/")

# Document Upload for RAG
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Business Documents (RAG)")
doc_files = st.sidebar.file_uploader(
    "Upload Strategy / Policy Documents",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)
if doc_files:
    for doc in doc_files:
        bytes_data = doc.read()
        buf = io.BytesIO(bytes_data)
        st.session_state.rag_engine.add_document(buf, doc.name)
    st.sidebar.success(f"Indexed {len(doc_files)} document(s)! Total chunks: {len(st.session_state.rag_engine.chunks)}")

# -----------------------------------------------------------------------------
# 4. Main Pipeline Execution
# -----------------------------------------------------------------------------
if uploaded_file or sample_file_path:
    tmp_dir = tempfile.mkdtemp()

    if uploaded_file:
        file_name = uploaded_file.name
        file_path = os.path.join(tmp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = io_utils.load_file(file_path, auto_clean_header=auto_detect_headers)
        st.sidebar.success(f"Ingested: **{file_name}** ({df.shape[0]:,} rows, {df.shape[1]} cols)")
    else:
        file_name = os.path.basename(sample_file_path)
        file_path = sample_file_path
        df = io_utils.load_file(file_path, auto_clean_header=auto_detect_headers)
        st.sidebar.success(f"Loaded: **{file_name}** ({df.shape[0]:,} rows, {df.shape[1]} cols)")

    # Load config
    config = {}
    if os.path.exists("config/config.yaml"):
        try:
            config = load_config("config/config.yaml") or {}
        except Exception:
            config = {}

    # --- 1. Clean Data ---
    df_clean = cleaning.clean_data(df, config)

    # --- 2. Validation & Anomalies ---
    val_rules = extract_validation_rules(config)
    issues = validation.validate_data(df_clean, val_rules)
    anomalies_found = anomalies.detect_anomalies(df_clean)

    # --- 3. Business Analytics & Roles ---
    roles = analytics.infer_business_columns(df_clean)
    kpis = analytics.compute_kpis(df_clean, roles)
    quality_score = analytics.compute_data_quality_score(df, df_clean, issues, anomalies_found)

    # --- 4. Visualizations ---
    tmp_fig_dir = os.path.join(tmp_dir, "figures")
    static_figs = visualize.generate_visuals(df_clean, tmp_fig_dir)
    interactive_figs = visualize.generate_interactive_visuals(df_clean)

    # --- 5. Predictive ML Insights ---
    insights = {}
    if enable_insights:
        insights = predictive.run_predictive_models(df_clean, num_clusters=num_clusters)
        if "clustering" in insights and isinstance(insights["clustering"], dict):
            insights["clustering"]["summary"] = f"Identified {num_clusters} business clusters in the dataset"

    # --- 6. Strategic Recommendations ---
    rec_engine = recommendations.RecommendationEngine(df_clean, roles)
    business_recs = rec_engine.generate_recommendations()

    # Save cleaned file
    base_name, _ = os.path.splitext(file_name)
    cleaned_filename = f"cleaned_{base_name}.csv"
    cleaned_path = os.path.join(tmp_dir, cleaned_filename)
    df_clean.to_csv(cleaned_path, index=False)

    # Summary for Report
    summary = {
        "Original Rows": df.shape[0],
        "Original Columns": df.shape[1],
        "Cleaned Rows": df_clean.shape[0],
        "Columns After Cleaning": df_clean.shape[1],
        "Validation Issues Found": sum(len(v) for v in issues.values()),
        "Anomalies Detected": len(anomalies_found),
        "Data Quality Score": f"{quality_score['score']}% ({quality_score['grade']})",
        "Output File": cleaned_filename,
    }
    if kpis.get("total_revenue") is not None:
        summary["Total Revenue"] = f"${kpis['total_revenue']:,.2f}"
    if kpis.get("total_profit") is not None:
        summary["Total Profit"] = f"${kpis['total_profit']:,.2f}"
    if kpis.get("profit_margin_pct") is not None:
        summary["Profit Margin"] = f"{kpis['profit_margin_pct']:.1f}%"
    if kpis.get("average_order_value") is not None:
        summary["Average Order Value"] = f"${kpis['average_order_value']:,.2f}"

    # Generate HTML report
    report_path = os.path.join(tmp_dir, "report.html")
    reporting.generate_report(
        validation=issues,
        anomalies=anomalies_found,
        figures=static_figs,
        summary=summary,
        insights=insights,
        output_path=report_path,
        recommendations=business_recs
    )

    # =========================================================================
    # 5. UI TABS (5 Official Tabs)
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analytics & Visuals",
        "💬 AI Analyst (LangGraph)",
        "🎯 Recommendations",
        "📚 Document Knowledge Base",
        "📄 Export & Reports"
    ])

    # ---------------- TAB 1: Analytics & Visuals ----------------
    with tab1:
        st.subheader("💡 Core Business KPIs")
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        if kpis.get("total_revenue") is not None:
            kpi_c1.metric("Total Revenue", f"${kpis['total_revenue']:,.2f}")
        else:
            kpi_c1.metric("Total Rows", f"{len(df_clean):,}")

        if kpis.get("total_profit") is not None:
            kpi_c2.metric("Total Profit", f"${kpis['total_profit']:,.2f}")
        else:
            kpi_c2.metric("Columns", len(df_clean.columns))

        if kpis.get("profit_margin_pct") is not None:
            kpi_c3.metric("Profit Margin", f"{kpis['profit_margin_pct']:.1f}%")
        elif kpis.get("unique_products") is not None:
            kpi_c3.metric("Unique Products", f"{kpis['unique_products']:,}")
        else:
            kpi_c3.metric("Duplicates Dropped", f"{df.duplicated().sum():,}")

        kpi_c4.metric("Data Quality Health", f"{quality_score['score']}%", quality_score["grade"])

        st.markdown("---")
        st.subheader("📈 Interactive Visual Analytics")
        if interactive_figs:
            tab_config = []
            if "top_products" in interactive_figs:
                tab_config.append(("Top Items", "top_products"))
            if "category_breakdown" in interactive_figs:
                tab_config.append(("Category", "category_breakdown"))
            if "monthly_trend" in interactive_figs:
                tab_config.append(("Trends", "monthly_trend"))
            if "correlation_matrix" in interactive_figs:
                tab_config.append(("Correlation", "correlation_matrix"))
            if "regression_model" in interactive_figs:
                tab_config.append(("Regression", "regression_model"))
            if "missing_values" in interactive_figs:
                tab_config.append(("Missing Values", "missing_values"))

            dist_keys = sorted(k for k in interactive_figs if k.startswith("distribution"))
            if dist_keys:
                tab_config.append(("Distributions", dist_keys))

            if tab_config:
                tab_labels = [t[0] for t in tab_config]
                chart_tabs = st.tabs(tab_labels)
                for ctab, (label, key_or_keys) in zip(chart_tabs, tab_config):
                    with ctab:
                        if isinstance(key_or_keys, list):
                            for dk in key_or_keys:
                                st.plotly_chart(interactive_figs[dk], width="stretch", key=f"tab1_dist_{dk}")
                        else:
                            st.plotly_chart(interactive_figs[key_or_keys], width="stretch", key=f"tab1_chart_{key_or_keys}")
        else:
            st.info("No numerical or categorical columns available to plot.")

        with st.expander("🔎 Raw & Cleaned Data Profiling", expanded=False):
            st.dataframe(df_clean.head(10), width="stretch")
            st.write(df_clean.describe(include="all").transpose())

    # ---------------- TAB 2: AI Analyst (LangGraph) ----------------
    with tab2:
        st.subheader("💬 AI Data Analyst (LangGraph Multi-Agent Engine)")
        st.caption("Ask questions about numbers, business strategy, or hybrid insights. Computed deterministically via DuckDB + Document RAG.")

        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            if st.button("Highest profit product?", width="stretch", key="btn_q_profit"):
                st.session_state.current_query = "Which product generated the highest profit?"
        with q_col2:
            if st.button("Total revenue?", width="stretch", key="btn_q_rev"):
                st.session_state.current_query = "What is the total revenue?"
        with q_col3:
            if st.button("Sales by category?", width="stretch", key="btn_q_cat"):
                st.session_state.current_query = "Sales by category"

        user_query = st.text_input(
            "Enter your business query:",
            value=st.session_state.current_query,
            placeholder="e.g. Total sales in 2023 or Which product had the highest profit?",
            key="user_query_input"
        )

        if not llm_client.is_configured():
            st.caption("ℹ️ *Running with Deterministic Engine. Select Google Gemini, OpenAI, or Ollama in the sidebar for unlimited open-ended queries.*")

        if user_query:
            orchestrator = DataPilotOrchestrator(
                df=df_clean,
                rag_engine=st.session_state.rag_engine,
                llm_client=llm_client
            )
            with st.spinner("Orchestrating agent workflow & validating execution..."):
                response = orchestrator.run(user_query, roles)

            st.markdown(response["answer"])

            if response.get("executed_sql"):
                with st.expander("Show Executed DuckDB SQL (Zero Hallucination)", expanded=False):
                    st.code(response["executed_sql"], language="sql")

            if not response["data"].empty:
                st.dataframe(response["data"], width="stretch")

            if response.get("citations"):
                with st.expander("View Document Source Citations", expanded=False):
                    for cit in response["citations"]:
                        st.markdown(f"**Source:** `{cit['source']}` (Relevance: {cit['score']:.0%})")
                        st.caption(f"> \"{cit['content']}\"")

    # ---------------- TAB 3: Recommendations ----------------
    with tab3:
        st.subheader("🎯 Prioritized Business Recommendations")
        st.caption("Grounded in quantitative thresholds and performance disparities identified in your dataset.")

        for rec in business_recs:
            p_class = rec["priority"].lower()
            st.markdown(f"""
            <div class="rec-box rec-{p_class}">
                <span class="badge" style="background:#0284c7; color:white;">{rec['priority']} Priority</span>
                <span style="color:#94a3b8; font-size:0.85rem; margin-left:8px;">{rec['category']}</span>
                <h4 style="margin:6px 0; color:#f8fafc;">{rec['title']}</h4>
                <p style="color:#cbd5e1; margin-bottom:6px;"><b>Evidence:</b> {rec['observation']}</p>
                <div style="background:rgba(56, 189, 248, 0.1); padding:8px; border-radius:4px; color:#e2e8f0; font-size:0.9rem;">
                    <b>Recommended Action:</b> {rec['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- TAB 4: Document Knowledge Base ----------------
    with tab4:
        st.subheader("📚 Business Document Knowledge Base")
        st.write(f"Currently indexed chunks: **{len(st.session_state.rag_engine.chunks)}**")

        if st.session_state.rag_engine.chunks:
            doc_sources = set(c.source for c in st.session_state.rag_engine.chunks)
            st.write("Indexed Documents:", ", ".join([f"`{s}`" for s in doc_sources]))

            with st.expander("Search Document Knowledge Base", expanded=True):
                doc_q = st.text_input("Search passages:", placeholder="e.g. strategy, margins, retention, compliance", key="doc_search_input")
                if doc_q:
                    retrieved = st.session_state.rag_engine.retrieve(doc_q, top_k=5)
                    st.markdown(st.session_state.rag_engine.format_citations(retrieved))
        else:
            st.info("No documents uploaded yet. Upload PDF, DOCX, or TXT business files in the sidebar to enable Document RAG and Hybrid Q&A.")

    # ---------------- TAB 5: Export & Reports ----------------
    with tab5:
        st.subheader("📄 Download Cleaned Data & Executive Reports")
        out_c1, out_c2, out_c3 = st.columns(3)

        with out_c1:
            with open(cleaned_path, "rb") as f:
                st.download_button("📥 Cleaned Dataset (CSV)", data=f, file_name=cleaned_filename, mime="text/csv", width="stretch", key="export_dl_csv")

        with out_c2:
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button("📥 Executive Report (HTML)", data=f, file_name="DataPilot_Executive_Report.html", mime="text/html", width="stretch", key="export_dl_html")

        with out_c3:
            pdf_path = report_path.replace(".html", ".pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Executive Report (PDF)", data=f, file_name="DataPilot_Executive_Report.pdf", mime="application/pdf", width="stretch", key="export_dl_pdf")
            else:
                st.caption("PDF export generated via wkhtmltopdf")

else:
    st.info("👈 Upload a CSV/Excel file or select a sample business dataset from the sidebar to start analysis.")
