import streamlit as st
import os
import pandas as pd
import numpy as np
import tempfile

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

st.set_page_config(page_title="DataPilot – AI Data Analyst", page_icon="✈️", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #94a3b8; margin-bottom: 1.5rem; }
    .rec-box { background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 12px; border-left: 5px solid #38bdf8; }
    .rec-high { border-left-color: #ef4444; }
    .rec-medium { border-left-color: #f59e0b; }
    .rec-low { border-left-color: #10b981; }
    .badge { font-size: 0.75rem; font-weight: bold; padding: 2px 8px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">DataPilot – AI Data Analyst & Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deterministic tabular analytics, LangGraph multi-agent querying, business document RAG & executive recommendations.</div>', unsafe_allow_html=True)

# ---------------- Session State Initialization ----------------
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = BusinessDocumentRAG()
    # Check if sample docs exist or create initial strategy note
    sample_doc_dir = "sample_documents"
    if os.path.exists(sample_doc_dir):
        st.session_state.rag_engine.add_directory(sample_doc_dir)

# ---------------- Sidebar Controls ----------------
st.sidebar.header("⚙️ Settings & Pipeline")

enable_insights = st.sidebar.checkbox("Enable Predictive Insights & ML", value=True)
report_type = st.sidebar.radio("Executive Report Type", ["HTML", "PDF", "Both"], index=0)
num_clusters = st.sidebar.slider("Number of Clusters (KMeans)", min_value=2, max_value=6, value=3)

# ---------------- LLM Configuration ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Natural Language Query Engine")

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
    llm_model = st.sidebar.selectbox("Gemini Model", ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-pro", "gemini-3.8-flash"], index=0)

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
st.sidebar.subheader("📁 Tabular Dataset")
input_choice = st.sidebar.radio("Data Source", ["Use Sample Dataset", "Upload File (CSV / Excel)"], index=0)

auto_detect_headers = st.sidebar.checkbox("Smart Header Auto-Detection", value=True, help="Automatically strips title banners, empty leading rows/columns, and detects true table headers.")

uploaded_file = None
sample_file_path = None

if input_choice == "Upload File (CSV / Excel)":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
else:
    sample_files = [f for f in os.listdir("sample_data") if f.endswith((".csv", ".xlsx", ".xls")) and not f.startswith("create")]
    if sample_files:
        chosen_sample = st.sidebar.selectbox("Choose Sample Dataset", sample_files, index=0)
        sample_file_path = os.path.join("sample_data", chosen_sample)
    else:
        st.sidebar.warning("No sample files found in sample_data/")

# Document Upload for RAG
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Business Documents (RAG)")
doc_files = st.sidebar.file_uploader("Upload Strategy/Policy Documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
if doc_files:
    for doc in doc_files:
        bytes_data = doc.read()
        import io
        buf = io.BytesIO(bytes_data)
        st.session_state.rag_engine.add_document(buf, doc.name)
    st.sidebar.success(f"Indexed {len(doc_files)} document(s)! Total chunks: {len(st.session_state.rag_engine.chunks)}")

# Main Pipeline Execution
if uploaded_file or sample_file_path:
    tmp_dir = tempfile.mkdtemp()

    # Ingest file
    if uploaded_file:
        file_name = uploaded_file.name
        file_path = os.path.join(tmp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = io_utils.load_file(file_path, auto_clean_header=auto_detect_headers)
        st.sidebar.success(f"Ingested: **{file_name}** ({df.shape[0]} rows, {df.shape[1]} cols)")
    else:
        file_name = os.path.basename(sample_file_path)
        file_path = sample_file_path
        df = io_utils.load_file(file_path, auto_clean_header=auto_detect_headers)
        st.sidebar.success(f"Loaded: **{file_name}** ({df.shape[0]} rows, {df.shape[1]} cols)")

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

    # ================= UI TABS =================
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
                                st.plotly_chart(interactive_figs[dk], width="stretch", key=f"chart_{dk}")
                        else:
                            st.plotly_chart(interactive_figs[key_or_keys], width="stretch", key=f"chart_{key_or_keys}")
        else:
            st.info("No numerical or categorical columns available to plot.")

        with st.expander("🔎 Raw & Cleaned Data Profiling", expanded=False):
            st.dataframe(df_clean.head(10), width="stretch")
            st.write(df_clean.describe(include="all").transpose())

    # ---------------- TAB 2: AI Analyst (LangGraph) ----------------
    with tab2:
        st.subheader("💬 AI Data Analyst (LangGraph Multi-Agent Engine)")
        st.caption("Ask questions about numbers, business strategy, or hybrid insights. Computed deterministically via DuckDB + Document RAG.")

        # Quick prompt buttons with state persistence
        if "current_query" not in st.session_state:
            st.session_state.current_query = ""

        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            if st.button("Highest profit product?", width="stretch"):
                st.session_state.current_query = "Which product generated the highest profit?"
        with q_col2:
            if st.button("Total revenue?", width="stretch"):
                st.session_state.current_query = "What is the total revenue?"
        with q_col3:
            if st.button("Sales by category?", width="stretch"):
                st.session_state.current_query = "Sales by category"

        user_query = st.text_input(
            "Enter your business query:",
            value=st.session_state.current_query,
            placeholder="e.g. Total sales in 2023 or Which product had the highest profit?"
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
                doc_q = st.text_input("Search passages:", placeholder="e.g. strategy, margins, retention, compliance")
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
                st.download_button("📥 Cleaned Dataset (CSV)", data=f, file_name=cleaned_filename, mime="text/csv", width="stretch")

        with out_c2:
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button("📥 Executive Report (HTML)", data=f, file_name="DataPilot_Executive_Report.html", mime="text/html", width="stretch")

        with out_c3:
            pdf_path = report_path.replace(".html", ".pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Executive Report (PDF)", data=f, file_name="DataPilot_Executive_Report.pdf", mime="application/pdf", width="stretch")
            else:
                st.caption("PDF export generated via wkhtmltopdf")

else:
    st.info("👈 Upload a CSV/Excel file or select a sample business dataset from the sidebar to start analysis.")
