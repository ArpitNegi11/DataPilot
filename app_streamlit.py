import streamlit as st
import os
import tempfile
import importlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans

from src import cleaning, validation, anomalies, visualize, reporting, io_utils, predictive, analytics
try:
    importlib.reload(analytics)
except Exception:
    pass

from src.main import load_config
from src.analytics import infer_business_columns, generate_business_insights
from src.query_engine import DataPilotQueryEngine

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Executive Design System (CSS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataPilot – AI Data Analyst & Recommendation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Base & Typography */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F1F5F9;
    letter-spacing: -0.01em;
}

/* App Background */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0d1527 0%, #080c16 60%, #04070d 100%);
    color: #F1F5F9;
}

/* Clean Block Margins & Sizing */
.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1440px !important;
}

/* Sidebar Styling & Shift Upwards */
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
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #F8FAFC;
    font-weight: 700;
}

/* Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.12) 50%, rgba(236, 72, 153, 0.08) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    padding: 24px 30px;
    margin-bottom: 22px;
    backdrop-filter: blur(16px);
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}
.hero-title {
    font-size: 2.0rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(120deg, #FFFFFF 20%, #38BDF8 60%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 0.92rem;
    color: #94A3B8;
    margin: 8px 0 0 0;
    max-width: 720px;
    line-height: 1.55;
}
.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-cyan {
    background: rgba(6, 182, 212, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(6, 182, 212, 0.35);
}
.badge-violet {
    background: rgba(139, 92, 246, 0.15);
    color: #A78BFA;
    border: 1px solid rgba(139, 92, 246, 0.35);
}
.badge-emerald {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

/* Glass Cards & Proper Sizing */
.glass-card {
    background: rgba(17, 24, 39, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 25px -8px rgba(0, 0, 0, 0.45);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(56, 189, 248, 0.32);
}

/* KPI Metric Cards Grid */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}
.kpi-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.92) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(14px);
    min-height: 110px;
    transition: all 0.25s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.38);
    box-shadow: 0 10px 25px -6px rgba(6, 182, 212, 0.2);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #06B6D4, #3B82F6, #8B5CF6);
}
.kpi-icon {
    font-size: 1.25rem;
    margin-bottom: 6px;
    display: inline-block;
}
.kpi-title {
    font-size: 0.74rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.52rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin-bottom: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-sub {
    font-size: 0.74rem;
    color: #64748B;
    font-weight: 500;
}

/* Executive Insights & Proper Indentation */
.insight-item {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 4px solid #06B6D4;
    border-radius: 0 10px 10px 0;
    padding: 12px 18px 12px 20px;
    margin-bottom: 11px;
    font-size: 0.90rem;
    line-height: 1.6;
    color: #E2E8F0;
}
.insight-risk {
    border-left-color: #F43F5E !important;
    background: rgba(244, 63, 94, 0.06);
}
.insight-rec {
    border-left-color: #10B981 !important;
    background: rgba(16, 185, 129, 0.06);
}

/* Section Header Sizing */
h2, h3, h4 {
    letter-spacing: -0.02em;
}
h3 {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.8rem !important;
}
h4 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.6rem !important;
}

/* Tabs Customization with proper sizing */
div[data-testid="stTabs"] {
    margin-top: 0.5rem;
}
div[data-testid="stTabs"] button[role="tab"] {
    font-size: 0.90rem;
    font-weight: 600;
    color: #94A3B8;
    padding: 9px 18px;
    border-radius: 8px;
    transition: all 0.2s ease;
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
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(14, 165, 233, 0.4);
}

/* Code & SQL drawer */
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.84rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Sidebar Navigation & Dataset Selector (Defaults to Live Data)
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:10px; margin-top:-0.25rem; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.08);">
        <span style="font-size:1.7rem;">⚡</span>
        <div>
            <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">DataPilot AI</div>
            <div style="font-size:0.72rem; color:#38BDF8; font-weight:600; letter-spacing:0.04em;">EXECUTIVE ANALYTICS</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### 📂 Data Ingestion")

SAMPLE_MAP = {
    "🌟 Flagship Sales Demo (400 records)": "sample_data/DataPilot_Sales_Demo.csv",
    "🛒 Supermarket Retail Orders (70 records)": "sample_data/Supermarket-Sales-Sample-Data.xlsx",
    "📦 Inventory Records (46 records)": "sample_data/Inventory-Records-Sample-Data.xlsx",
    "👥 Subscription Cohorts (3,069 records)": "sample_data/Subscription-Cohort-Analysis-Data.xlsx",
    "📋 Employee Payroll Sample (5 records)": "sample_data/sample.xlsx",
    "📤 Upload Custom File (CSV / Excel)": "UPLOAD_MODE",
}

selected_source = st.sidebar.selectbox(
    "Select Active Dataset",
    list(SAMPLE_MAP.keys()),
    index=0,  # Default: Flagship Sales Demo guarantees data presence
)

uploaded_file = None
active_file_path = None
is_sample = True

if selected_source == "📤 Upload Custom File (CSV / Excel)":
    is_sample = False
    uploaded_file = st.sidebar.file_uploader(
        "Upload Spreadsheet",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV and Excel files (.xlsx, .xls)."
    )
else:
    active_file_path = SAMPLE_MAP[selected_source]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Analysis Parameters")

enable_insights = st.sidebar.toggle("Predictive Modeling & ML", value=True)
num_clusters = st.sidebar.slider("KMeans Cluster Segments", min_value=2, max_value=6, value=3)
auto_detect_headers = st.sidebar.toggle(
    "Smart Header Auto-Detection",
    value=True,
    help="Automatically detects column headers and trims decorative banner rows.",
)
report_type = st.sidebar.radio("Report Export Format", ["HTML", "PDF", "Both"], index=0)


# -----------------------------------------------------------------------------
# 3. Data Ingestion & Processing Pipeline
# -----------------------------------------------------------------------------
df_raw = None
file_display_name = ""

if not is_sample and uploaded_file is not None:
    file_display_name = uploaded_file.name
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, file_display_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        df_raw = io_utils.load_file(file_path, auto_clean_header=auto_detect_headers)
    except Exception as e:
        st.error(f"Error loading file: {e}")
elif is_sample and active_file_path and os.path.exists(active_file_path):
    file_display_name = os.path.basename(active_file_path)
    df_raw = io_utils.load_file(active_file_path, auto_clean_header=auto_detect_headers)


# -----------------------------------------------------------------------------
# 4. Empty State Handler (Guarantees zero blank screen)
# -----------------------------------------------------------------------------
if df_raw is None:
    st.markdown(
        """
        <div class="hero-container">
            <div>
                <h1 class="hero-title">Welcome to DataPilot</h1>
                <p class="hero-subtitle">
                    Upload your raw business data or select one of the pre-configured datasets below to immediately explore automated cleaning, interactive Plotly dashboards, natural language queries, and predictive insights.
                </p>
                <div class="badge-row">
                    <span class="hero-badge badge-cyan">⚡ Zero Setup Required</span>
                    <span class="hero-badge badge-violet">🤖 DuckDB Deterministic Engine</span>
                    <span class="hero-badge badge-emerald">📊 Interactive Plotly Dashboards</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🚀 Quick-Start Business Datasets")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:1.4rem; margin-bottom:6px;">🌟</div>
                <h4 style="margin:0 0 4px 0; color:#F8FAFC;">Sales & Revenue Demo</h4>
                <p style="font-size:0.84rem; color:#94A3B8; margin-bottom:12px; line-height:1.5;">
                    400 transactions with Products, Regional Segments, Discounts, Revenue, and Profit Margins.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load Sales Demo Dataset", use_container_width=True, key="quick_load_sales"):
            active_file_path = "sample_data/DataPilot_Sales_Demo.csv"
            df_raw = io_utils.load_file(active_file_path, auto_clean_header=True)
            file_display_name = os.path.basename(active_file_path)
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:1.4rem; margin-bottom:6px;">🛒</div>
                <h4 style="margin:0 0 4px 0; color:#F8FAFC;">Supermarket Retail</h4>
                <p style="font-size:0.84rem; color:#94A3B8; margin-bottom:12px; line-height:1.5;">
                    Retail department orders with product categories, customer types, and shipping dates.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load Supermarket Dataset", use_container_width=True, key="quick_load_supermarket"):
            active_file_path = "sample_data/Supermarket-Sales-Sample-Data.xlsx"
            df_raw = io_utils.load_file(active_file_path, auto_clean_header=True)
            file_display_name = os.path.basename(active_file_path)
            st.rerun()

    with col3:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:1.4rem; margin-bottom:6px;">👥</div>
                <h4 style="margin:0 0 4px 0; color:#F8FAFC;">Subscription Cohorts</h4>
                <p style="font-size:0.84rem; color:#94A3B8; margin-bottom:12px; line-height:1.5;">
                    3,069 recurring subscriber profiles with monthly billing, intervals, and churn tracking.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load Subscription Dataset", use_container_width=True, key="quick_load_cohort"):
            active_file_path = "sample_data/Subscription-Cohort-Analysis-Data.xlsx"
            df_raw = io_utils.load_file(active_file_path, auto_clean_header=True)
            file_display_name = os.path.basename(active_file_path)
            st.rerun()

    st.stop()


# -----------------------------------------------------------------------------
# 5. Core Pipeline Execution: Clean, Validate, Detect Anomalies & Insights
# -----------------------------------------------------------------------------
tmp_dir = tempfile.mkdtemp()

config = {}
if os.path.exists("config/config.yaml"):
    try:
        config = load_config("config/config.yaml") or {}
    except Exception:
        config = {}

# Clean data
df_clean = cleaning.clean_data(df_raw, config)

# Validation & Anomalies
validation_rules = config.get("validation_rules", [])
issues = validation.validate_data(df_clean, validation_rules)
anomalies_found = anomalies.detect_anomalies(df_clean)

# Automated Business Insights
business_insights = generate_business_insights(df_clean)
inferred_roles = business_insights.get("roles", {})
kpis = business_insights.get("kpis", {})

# Visualizations
tmp_fig_dir = os.path.join(tmp_dir, "figures")
static_figs = visualize.generate_visuals(df_clean, tmp_fig_dir)
interactive_figs = visualize.generate_interactive_visuals(df_clean)

# Predictive ML Models
insights_models = {}
if enable_insights:
    try:
        insights_models = predictive.run_predictive_models(df_clean, num_clusters=num_clusters)
        if "clustering" in insights_models and isinstance(insights_models["clustering"], dict):
            insights_models["clustering"]["summary"] = f"Identified {num_clusters} business clusters in dataset"
    except Exception as e:
        print(f"[PREDICTIVE_ERROR] {e}")

# Save Cleaned CSV
base_name, _ = os.path.splitext(file_display_name)
cleaned_filename = f"cleaned_{base_name}.csv"
cleaned_path = os.path.join(tmp_dir, cleaned_filename)
df_clean.to_csv(cleaned_path, index=False)

# Executive Report Generation
report_summary = {
    "Original Rows": df_raw.shape[0],
    "Original Columns": df_raw.shape[1],
    "Cleaned Rows": df_clean.shape[0],
    "Columns After Cleaning": df_clean.shape[1],
    "Validation Issues Found": sum(len(v) for v in issues.values()),
    "Columns With Issues": len(issues),
    "Anomalies Detected": len(anomalies_found),
    "Output File": cleaned_filename,
}
report_path = os.path.join(tmp_dir, "report.html")
reporting.generate_report(
    issues, anomalies_found, static_figs, report_summary, insights_models, report_path
)


# -----------------------------------------------------------------------------
# 6. Hero Header & Quick Stats Bar (Proper Sizing & Padding)
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-container">
        <div>
            <h1 class="hero-title">DataPilot Executive Intelligence</h1>
            <p class="hero-subtitle">
                Automated data hygiene, deterministic KPI tracking, interactive Plotly visualization, and predictive machine learning.
            </p>
            <div class="badge-row">
                <span class="hero-badge badge-cyan">📂 {file_display_name}</span>
                <span class="hero-badge badge-emerald">⚡ {df_clean.shape[0]:,} Rows Active</span>
                <span class="hero-badge badge-violet">🛡️ Cleaned & Validated</span>
                <span class="hero-badge badge-amber">🤖 AI Engine Active</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compute High-Level Metrics
rev_val = kpis.get("total_revenue", None)
prof_val = kpis.get("total_profit", None)
margin_val = kpis.get("profit_margin", None)
total_anomalies_count = sum(len(v) for v in anomalies_found.values())
duplicate_count = df_raw.duplicated().sum()
missing_cells = df_raw.isnull().sum().sum()
health_score = max(0, min(100, int(100 - (missing_cells / (df_raw.size or 1) * 60) - (duplicate_count / (len(df_raw) or 1) * 30))))

# KPI Cards Bar
st.markdown(
    f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <span class="kpi-icon">💰</span>
            <div class="kpi-title">Total Volume / Revenue</div>
            <div class="kpi-value">{"${:,.2f}".format(rev_val) if rev_val is not None else f"{len(df_clean):,} Records"}</div>
            <div class="kpi-sub">{"Avg ${:,.2f} / tx".format(kpis.get('avg_revenue', 0)) if rev_val is not None else "Active transactions"}</div>
        </div>
        <div class="kpi-card">
            <span class="kpi-icon">📈</span>
            <div class="kpi-title">Net Profit / Margin</div>
            <div class="kpi-value" style="color: {'#34D399' if (margin_val or 0) >= 0 else '#F43F5E'};">
                {f"{margin_val:.1f}% Margin" if margin_val is not None else f"{df_clean.shape[1]} Columns"}
            </div>
            <div class="kpi-sub">{"${:,.2f} total profit".format(prof_val) if prof_val is not None else "Analyzed dimensions"}</div>
        </div>
        <div class="kpi-card">
            <span class="kpi-icon">🧹</span>
            <div class="kpi-title">Data Cleanliness</div>
            <div class="kpi-value">{df_clean.shape[0]:,}</div>
            <div class="kpi-sub">{duplicate_count} duplicates removed • {missing_cells} imputed</div>
        </div>
        <div class="kpi-card">
            <span class="kpi-icon">🛡️</span>
            <div class="kpi-title">Data Quality Health</div>
            <div class="kpi-value" style="color:#38BDF8;">{health_score}%</div>
            <div class="kpi-sub">{'A+ Enterprise Grade' if health_score >= 90 else 'Good Quality'}</div>
        </div>
        <div class="kpi-card">
            <span class="kpi-icon">⚠️</span>
            <div class="kpi-title">Anomalies Detected</div>
            <div class="kpi-value" style="color: {'#FBBF24' if total_anomalies_count > 0 else '#34D399'};">
                {total_anomalies_count}
            </div>
            <div class="kpi-sub">{len(anomalies_found)} numeric columns flagged</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 7. Main Executive Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_cleaning, tab_visuals, tab_predictive, tab_query, tab_reports = st.tabs([
    "📊 Executive Overview & KPIs",
    "🧹 Data Profiling & Health",
    "📈 Visual Analytics Studio",
    "🤖 Predictive AI & Insights",
    "💬 Ask DataPilot (NL Query)",
    "📑 Reports & Export",
])


# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW & KPIS
# =============================================================================
with tab_overview:
    st.markdown("### 📊 Executive Summary & Core Performance")
    
    # Dual Highlight Charts with Unique Keys
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        if "top_products" in interactive_figs:
            st.plotly_chart(interactive_figs["top_products"], use_container_width=True, key="overview_chart_top_products")
        elif "monthly_trend" in interactive_figs:
            st.plotly_chart(interactive_figs["monthly_trend"], use_container_width=True, key="overview_chart_monthly_trend_left")
        elif interactive_figs:
            st.plotly_chart(list(interactive_figs.values())[0], use_container_width=True, key="overview_chart_fallback_0")
        else:
            st.info("Chart preview not available for this column format.")

    with col_right:
        if "category_breakdown" in interactive_figs:
            st.plotly_chart(interactive_figs["category_breakdown"], use_container_width=True, key="overview_chart_category_breakdown")
        elif "correlation_matrix" in interactive_figs:
            st.plotly_chart(interactive_figs["correlation_matrix"], use_container_width=True, key="overview_chart_correlation_matrix")
        elif len(interactive_figs) > 1:
            st.plotly_chart(list(interactive_figs.values())[1], use_container_width=True, key="overview_chart_fallback_1")
        else:
            st.info("Category breakdown not available.")

    # Time series secondary chart
    if "monthly_trend" in interactive_figs and "top_products" in interactive_figs:
        st.markdown("---")
        st.plotly_chart(interactive_figs["monthly_trend"], use_container_width=True, key="overview_chart_monthly_trend_bottom")

    # Cleaned Records Preview
    st.markdown("---")
    st.markdown("#### 📋 Cleaned Dataset Quick Preview")
    st.dataframe(df_clean.head(10), use_container_width=True, height=280)


# =============================================================================
# TAB 2: DATA PROFILING & HEALTH
# =============================================================================
with tab_cleaning:
    st.markdown("### 🧹 Data Hygiene & Automated Cleaning Audit")
    
    col_prof1, col_prof2 = st.columns([1, 1])
    
    with col_prof1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin-top:0; color:#38BDF8;">Dataset Quality Scorecard</h4>
                <ul style="list-style-type:none; padding-left:4px; line-height:2.1; font-size:0.92rem; color:#CBD5E1;">
                    <li><strong>Total Input Records:</strong> {df_raw.shape[0]:,}</li>
                    <li><strong>Clean Records Retained:</strong> {df_clean.shape[0]:,} ({df_clean.shape[0]/(df_raw.shape[0] or 1)*100:.1f}%)</li>
                    <li><strong>Duplicates Removed:</strong> {duplicate_count:,}</li>
                    <li><strong>Missing Cells Imputed:</strong> {missing_cells:,}</li>
                    <li><strong>Validation Rule Violations:</strong> {sum(len(v) for v in issues.values())}</li>
                    <li><strong>Outliers Detected (Z > 3.0):</strong> {total_anomalies_count}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_prof2:
        if "missing_values" in interactive_figs:
            st.plotly_chart(interactive_figs["missing_values"], use_container_width=True, key="cleaning_chart_missing_values")
        else:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; padding:38px 24px;">
                    <span style="font-size:2.4rem;">🎉</span>
                    <h4 style="color:#34D399; margin:10px 0 6px 0;">Zero Missing Values</h4>
                    <p style="font-size:0.85rem; color:#94A3B8; margin:0;">All columns maintain complete data integrity across all rows.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Detailed Audit & Expanders
    st.markdown("#### 🔍 Schema & Statistical Distribution")
    with st.expander("Show Detailed Column Statistics (Describe)", expanded=False):
        st.dataframe(df_clean.describe(include="all").transpose(), use_container_width=True, height=320)

    if issues:
        with st.expander("⚠️ Validation Issues Detected", expanded=True):
            for col_name, col_issues in issues.items():
                st.markdown(f"**{col_name}**: " + ", ".join(col_issues))

    if anomalies_found:
        with st.expander("🚨 Anomaly Details by Column", expanded=False):
            for col_name, values in anomalies_found.items():
                st.markdown(
                    f"**{col_name}**: {len(values)} outlier(s) flagged — values: `{', '.join(str(v) for v in values[:10])}`"
                    + (" ..." if len(values) > 10 else "")
                )


# =============================================================================
# TAB 3: VISUAL ANALYTICS STUDIO
# =============================================================================
with tab_visuals:
    st.markdown("### 📈 Interactive Visual Analytics Studio")
    
    chart_view = st.radio(
        "Select Visual Dashboard Mode",
        ["Curated Business Dashboards", "Custom Chart Explorer (Ad-Hoc Builder)"],
        horizontal=True,
    )

    if chart_view == "Curated Business Dashboards":
        if interactive_figs:
            visual_tabs_list = []
            if "top_products" in interactive_figs:
                visual_tabs_list.append(("🏆 Top Items", "top_products"))
            if "category_breakdown" in interactive_figs:
                visual_tabs_list.append(("🍩 Category Share", "category_breakdown"))
            if "monthly_trend" in interactive_figs:
                visual_tabs_list.append(("📅 Time Series Trends", "monthly_trend"))
            if "correlation_matrix" in interactive_figs:
                visual_tabs_list.append(("🌡️ Correlation Matrix", "correlation_matrix"))
            
            # Distributions
            dist_keys = sorted(k for k in interactive_figs if k.startswith("distribution"))
            if dist_keys:
                visual_tabs_list.append(("📊 Distributions", dist_keys))

            if visual_tabs_list:
                v_tabs = st.tabs([t[0] for t in visual_tabs_list])
                for v_tab, (label, fig_key) in zip(v_tabs, visual_tabs_list):
                    with v_tab:
                        if isinstance(fig_key, list):
                            for dk in fig_key:
                                st.plotly_chart(
                                    interactive_figs[dk],
                                    use_container_width=True,
                                    key=f"studio_curated_dist_{dk}"
                                )
                        else:
                            st.plotly_chart(
                                interactive_figs[fig_key],
                                use_container_width=True,
                                key=f"studio_curated_{fig_key}"
                            )
        else:
            st.info("No numerical or categorical data available to generate charts.")

    else:
        # Custom Ad-Hoc Chart Builder with proper sizing
        st.markdown("#### 🎨 Custom 2D Chart Explorer")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        
        all_cols = list(df_clean.columns)
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()

        with c_col1:
            chart_type = st.selectbox("Chart Type", ["Scatter Plot", "Bar Chart", "Box Plot", "Histogram", "Line Trend"], key="adhoc_chart_type")
        with c_col2:
            x_col = st.selectbox("X-Axis", all_cols, index=0, key="adhoc_x_axis")
        with c_col3:
            y_default = num_cols[0] if num_cols else all_cols[min(1, len(all_cols)-1)]
            y_col = st.selectbox("Y-Axis", all_cols, index=all_cols.index(y_default) if y_default in all_cols else 0, key="adhoc_y_axis")
        with c_col4:
            color_opts = ["None"] + all_cols
            color_choice = st.selectbox("Color / Group By", color_opts, index=0, key="adhoc_color_by")

        color_arg = None if color_choice == "None" else color_choice

        try:
            if chart_type == "Scatter Plot":
                custom_fig = px.scatter(
                    df_clean, x=x_col, y=y_col, color=color_arg,
                    template="plotly_dark", opacity=0.8,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
            elif chart_type == "Bar Chart":
                custom_fig = px.bar(
                    df_clean, x=x_col, y=y_col, color=color_arg,
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
            elif chart_type == "Box Plot":
                custom_fig = px.box(
                    df_clean, x=x_col, y=y_col, color=color_arg,
                    template="plotly_dark"
                )
            elif chart_type == "Histogram":
                custom_fig = px.histogram(
                    df_clean, x=x_col, color=color_arg,
                    template="plotly_dark", marginal="box"
                )
            elif chart_type == "Line Trend":
                custom_fig = px.line(
                    df_clean.sort_values(x_col), x=x_col, y=y_col, color=color_arg,
                    template="plotly_dark", markers=True
                )
            
            custom_fig.update_layout(
                paper_bgcolor="rgba(15, 23, 42, 0.8)",
                plot_bgcolor="rgba(15, 23, 42, 0.8)",
                font=dict(family="'Plus Jakarta Sans', sans-serif"),
                margin=dict(l=45, r=25, t=45, b=45),
                height=420,
            )
            st.plotly_chart(custom_fig, use_container_width=True, key="studio_adhoc_custom_chart")
        except Exception as e:
            st.warning(f"Could not render custom chart with chosen parameters: {e}")


# =============================================================================
# TAB 4: PREDICTIVE AI & DATA-DRIVEN INSIGHTS (REVAMPED SECTION 4)
# =============================================================================
with tab_predictive:
    st.markdown("### 🤖 Predictive AI & Data-Driven Insights Engine")
    st.caption("Deterministic business findings, statistical linear regressions with 'What-If' forecasting, and multi-dimensional cluster segmentation.")

    # -------------------------------------------------------------------------
    # Part A: Grounded Executive Business Findings with Proper Indentation
    # -------------------------------------------------------------------------
    st.markdown("#### 💡 Executive Takeaways & Performance Drivers")
    
    ins_c1, ins_c2 = st.columns(2)
    
    with ins_c1:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="margin:0 0 12px 0; color:#38BDF8; font-size:1.02rem;">🏆 Revenue & Volume Drivers</h4>
            """,
            unsafe_allow_html=True,
        )
        if business_insights.get("drivers"):
            for d in business_insights["drivers"]:
                st.markdown(f'<div class="insight-item">{d}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight-item">Total volume analyzed: <strong>{len(df_clean):,} records</strong> across {df_clean.shape[1]} attributes.</div>', unsafe_allow_html=True)

        if business_insights.get("risks"):
            st.markdown('<h4 style="margin:16px 0 10px 0; color:#F43F5E; font-size:1.02rem;">⚠️ Risk & Margin Alerts</h4>', unsafe_allow_html=True)
            for r in business_insights["risks"]:
                st.markdown(f'<div class="insight-item insight-risk">{r}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ins_c2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="margin:0 0 12px 0; color:#10B981; font-size:1.02rem;">🎯 Strategic Action Recommendations</h4>
            """,
            unsafe_allow_html=True,
        )
        recs = business_insights.get("recommendations", [])
        for rec in recs:
            st.markdown(f'<div class="insight-item insight-rec">💡 {rec}</div>', unsafe_allow_html=True)

        # Dataset Health Insight
        st.markdown(
            f"""
            <div class="insight-item" style="border-left-color:#8B5CF6;">
                🛡️ <strong>Data Integrity Score: {health_score}/100</strong>.
                {f"Imputed {missing_cells} missing values and eliminated {duplicate_count} duplicate records." if (missing_cells or duplicate_count) else "All records met zero-loss schema integrity standards."}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Part B: Interactive Linear Regression Studio & What-If Simulator
    # -------------------------------------------------------------------------
    st.markdown("#### 🔮 Predictive Linear Regression & What-If Simulator")
    
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) >= 2:
        # Determine best default pair
        default_x, default_y = num_cols[0], num_cols[1]
        if "Experience" in df_clean.columns and "Salary" in df_clean.columns:
            default_x, default_y = "Experience", "Salary"
        elif "Quantity" in df_clean.columns and "Sales" in df_clean.columns:
            default_x, default_y = "Quantity", "Sales"
        elif "Sales" in df_clean.columns and "Profit" in df_clean.columns:
            default_x, default_y = "Sales", "Profit"
        elif "Units" in df_clean.columns and "Total" in df_clean.columns:
            default_x, default_y = "Units", "Total"
        else:
            try:
                corr_m = df_clean[num_cols].corr().abs()
                np.fill_diagonal(corr_m.values, 0)
                if not corr_m.empty and corr_m.max().max() > 0.05:
                    pair = corr_m.stack().idxmax()
                    default_x, default_y = pair[0], pair[1]
            except Exception:
                pass

        reg_ctrl_col1, reg_ctrl_col2 = st.columns(2)
        with reg_ctrl_col1:
            sel_x = st.selectbox("Predictor Feature (X)", num_cols, index=num_cols.index(default_x) if default_x in num_cols else 0, key="reg_sel_x")
        with reg_ctrl_col2:
            y_candidates = [c for c in num_cols if c != sel_x] or num_cols
            sel_y = st.selectbox("Target Outcome (Y)", y_candidates, index=y_candidates.index(default_y) if default_y in y_candidates else 0, key="reg_sel_y")

        # Compute regression
        valid_reg = df_clean[[sel_x, sel_y]].dropna()
        if len(valid_reg) >= 3 and valid_reg[sel_x].nunique() > 1:
            X_data = valid_reg[[sel_x]].values
            y_data = valid_reg[sel_y].values

            lr_model = LinearRegression()
            lr_model.fit(X_data, y_data)
            y_pred = lr_model.predict(X_data)

            coef_val = float(lr_model.coef_[0])
            intercept_val = float(lr_model.intercept_)
            r2_val = float(r2_score(y_data, y_pred))
            rmse_val = float(np.sqrt(mean_squared_error(y_data, y_pred)))

            # Regression Plotly Chart
            fig_reg_interactive = px.scatter(
                valid_reg,
                x=sel_x,
                y=sel_y,
                opacity=0.75,
                color_discrete_sequence=["#38BDF8"],
                hover_data=[sel_x, sel_y],
                template="plotly_dark",
            )
            fig_reg_interactive.update_traces(name="Observed Transactions", showlegend=True)

            sort_order = np.argsort(X_data.flatten())
            fig_reg_interactive.add_trace(
                go.Scatter(
                    x=X_data.flatten()[sort_order],
                    y=y_pred[sort_order],
                    mode="lines",
                    name=f"Fit Line: {sel_y} = {coef_val:.2f}*{sel_x} + {intercept_val:.2f}",
                    line=dict(color="#F43F5E", width=3.5),
                )
            )
            fig_reg_interactive.update_layout(
                title=f"Linear Regression: {sel_y} vs. {sel_x} (R² = {r2_val:.3f})",
                paper_bgcolor="rgba(15, 23, 42, 0.8)",
                plot_bgcolor="rgba(15, 23, 42, 0.8)",
                font=dict(family="'Plus Jakarta Sans', sans-serif"),
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
                height=420,
                margin=dict(l=45, r=25, t=45, b=45),
            )

            col_chart, col_sim = st.columns([1.3, 0.7])
            with col_chart:
                st.plotly_chart(fig_reg_interactive, use_container_width=True, key="predictive_tab_reg_chart")

            with col_sim:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h4 style="margin:0 0 10px 0; color:#38BDF8;">Model Fit Summary</h4>
                        <div style="font-size:0.84rem; line-height:1.75; color:#CBD5E1;">
                            <strong>Formula:</strong> <code>{sel_y} = {coef_val:.2f}*{sel_x} + {intercept_val:.2f}</code><br>
                            <strong>Fit Quality (R²):</strong> <span style="color:#34D399; font-weight:700;">{r2_val:.4f}</span><br>
                            <strong>Marginal Slope:</strong> <code>{coef_val:.4f}</code> per unit<br>
                            <strong>Baseline Intercept:</strong> <code>{intercept_val:.2f}</code><br>
                            <strong>Root Mean Sq Error:</strong> <code>{rmse_val:.2f}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # What-If Simulator with clean sizing
                st.markdown("##### 🎛️ Interactive 'What-If' Predictor")
                min_x = float(valid_reg[sel_x].min())
                max_x = float(valid_reg[sel_x].max())
                mean_x = float(valid_reg[sel_x].mean())

                sim_x_val = st.number_input(
                    f"Enter hypothetical {sel_x} value:",
                    value=float(round(mean_x, 2)),
                    step=float(round((max_x - min_x) / 20 or 1.0, 2)),
                    key="what_if_num_input",
                )
                pred_sim_y = float(lr_model.predict([[sim_x_val]])[0])

                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg, rgba(6,182,212,0.15) 0%, rgba(99,102,241,0.15) 100%);
                                border:1px solid rgba(56,189,248,0.4); border-radius:12px; padding:16px 14px; text-align:center;">
                        <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:700;">Predicted {sel_y}</div>
                        <div style="font-size:1.6rem; font-weight:800; color:#F8FAFC; margin:4px 0;">
                            {"${:,.2f}".format(pred_sim_y) if any(t in sel_y.lower() for t in ["sales","revenue","profit","cost","salary","total"]) else f"{pred_sim_y:,.2f}"}
                        </div>
                        <div style="font-size:0.74rem; color:#38BDF8;">Based on deterministic regression fit</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(f"Insufficient unique variance between {sel_x} and {sel_y} to compute regression.")
    else:
        st.info("At least 2 numeric columns are required to run predictive regression.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Part C: Multi-Dimensional K-Means Clustering Studio
    # -------------------------------------------------------------------------
    st.markdown("#### 🧩 Multi-Dimensional Segmentation (K-Means Clustering)")
    
    numeric_df = df_clean.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] >= 2 and len(numeric_df) >= 3:
        cluster_k = max(2, min(int(num_clusters), len(numeric_df)))
        
        # Pick 2 features for 2D visualization
        col_c_feat1, col_c_feat2 = st.columns(2)
        with col_c_feat1:
            clust_x = st.selectbox("Cluster Feature 1 (X-Axis)", numeric_df.columns, index=0, key="clust_feat_1")
        with col_c_feat2:
            clust_y = st.selectbox("Cluster Feature 2 (Y-Axis)", numeric_df.columns, index=min(1, len(numeric_df.columns)-1), key="clust_feat_2")

        try:
            km = KMeans(n_clusters=cluster_k, n_init="auto", random_state=42)
            cluster_labels = km.fit_predict(numeric_df[[clust_x, clust_y]])
            
            clust_plot_df = numeric_df.copy()
            clust_plot_df["Cluster"] = [f"Cluster {c}" for c in cluster_labels]

            fig_cluster = px.scatter(
                clust_plot_df,
                x=clust_x,
                y=clust_y,
                color="Cluster",
                template="plotly_dark",
                opacity=0.8,
                color_discrete_sequence=["#06B6D4", "#8B5CF6", "#10B981", "#F59E0B", "#F43F5E", "#EC4899"],
                hover_data=[clust_x, clust_y],
            )
            # Add Centroids
            centroids = km.cluster_centers_
            fig_cluster.add_trace(
                go.Scatter(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    mode="markers",
                    name="Cluster Centroids",
                    marker=dict(size=14, color="#FFFFFF", symbol="x", line=dict(width=2, color="#000")),
                )
            )
            fig_cluster.update_layout(
                title=f"K-Means Customer & Transaction Segmentation (k={cluster_k})",
                paper_bgcolor="rgba(15, 23, 42, 0.8)",
                plot_bgcolor="rgba(15, 23, 42, 0.8)",
                font=dict(family="'Plus Jakarta Sans', sans-serif"),
                height=420,
                margin=dict(l=45, r=25, t=45, b=45),
            )

            col_clust_plot, col_clust_stats = st.columns([1.3, 0.7])
            with col_clust_plot:
                st.plotly_chart(fig_cluster, use_container_width=True, key="predictive_tab_cluster_chart")

            with col_clust_stats:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h4 style="margin:0 0 10px 0; color:#38BDF8;">Cluster Breakdown</h4>
                    """,
                    unsafe_allow_html=True,
                )
                clust_summary = clust_plot_df.groupby("Cluster")[[clust_x, clust_y]].mean().round(2)
                clust_counts = clust_plot_df["Cluster"].value_counts()
                clust_summary["Records"] = clust_counts

                st.dataframe(clust_summary, use_container_width=True, height=220)
                st.caption(f"Segmented {len(clust_plot_df):,} records into {cluster_k} distinct behavioral clusters based on {clust_x} and {clust_y}.")
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Clustering execution encountered an issue: {e}")
    else:
        st.info("Insufficient numeric features to compute K-Means cluster segmentation.")


# =============================================================================
# TAB 5: ASK DATAPILOT (NATURAL LANGUAGE QUERY)
# =============================================================================
with tab_query:
    st.markdown("### 💬 Ask DataPilot – Conversational NL-to-SQL Analytics")
    st.caption("Ask natural language business questions about the active dataset. Computed deterministically via DuckDB with zero hallucinations.")

    st.markdown("#### ⚡ Quick Questions (One-Click Prompts)")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_q = ""
    with q_col1:
        if st.button("💎 Highest Profit Item?", use_container_width=True, key="btn_quick_profit"):
            quick_q = "Which product generated the highest profit?"
    with q_col2:
        if st.button("💰 Total Revenue & Records?", use_container_width=True, key="btn_quick_rev"):
            quick_q = "What is the total revenue?"
    with q_col3:
        if st.button("📊 Sales by Category?", use_container_width=True, key="btn_quick_cat"):
            quick_q = "Sales by category"
    with q_col4:
        if st.button("🌍 Regional Breakdown?", use_container_width=True, key="btn_quick_reg"):
            quick_q = "Sales by region"

    query_input = st.text_input(
        "Enter your question or analytical command:",
        value=quick_q,
        placeholder="e.g. Which product generated the highest profit? or Total sales by category",
        key="nl_query_box",
    )

    if query_input:
        engine = DataPilotQueryEngine(df_clean)
        result = engine.ask(query_input, inferred_roles)

        st.markdown(
            f"""
            <div class="glass-card" style="border-left:4px solid #10B981; background:rgba(16,185,129,0.08);">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#34D399; font-weight:700;">Query Result</div>
                <div style="font-size:1.15rem; font-weight:600; color:#F8FAFC; margin-top:4px;">{result['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🛠️ Executed DuckDB SQL Statement", expanded=False):
            st.code(result["executed_sql"], language="sql")

        if not result["data"].empty:
            st.markdown("#### 📊 Query Result Table")
            st.dataframe(result["data"], use_container_width=True, height=240)


# =============================================================================
# TAB 6: REPORTS & EXPORT
# =============================================================================
with tab_reports:
    st.markdown("### 📑 Executive Reports & Output Downloads")
    st.caption("Export cleaned datasets, interactive HTML dashboards, and executive PDF summaries.")

    rep_col1, rep_col2, rep_col3 = st.columns(3)

    with rep_col1:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;">
                <span style="font-size:2rem;">📥</span>
                <h4 style="margin:8px 0 4px 0; color:#F8FAFC;">Cleaned Dataset</h4>
                <p style="font-size:0.82rem; color:#94A3B8; margin-bottom:14px; line-height:1.5;">Sanitized CSV with imputed values & standardized formats.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if os.path.exists(cleaned_path):
            with open(cleaned_path, "rb") as f:
                st.download_button(
                    "Download Cleaned CSV",
                    data=f,
                    file_name=cleaned_filename,
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_btn_csv",
                )

    with rep_col2:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;">
                <span style="font-size:2rem;">🌐</span>
                <h4 style="margin:8px 0 4px 0; color:#F8FAFC;">HTML Executive Report</h4>
                <p style="font-size:0.82rem; color:#94A3B8; margin-bottom:14px; line-height:1.5;">Standalone, self-contained HTML report with embedded visuals.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button(
                    "Download HTML Report",
                    data=f,
                    file_name="DataPilot_Executive_Report.html",
                    mime="text/html",
                    use_container_width=True,
                    key="dl_btn_html",
                )

    with rep_col3:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;">
                <span style="font-size:2rem;">📄</span>
                <h4 style="margin:8px 0 4px 0; color:#F8FAFC;">PDF Executive Report</h4>
                <p style="font-size:0.82rem; color:#94A3B8; margin-bottom:14px; line-height:1.5;">Print-ready executive summary generated via wkhtmltopdf.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pdf_path = report_path.replace(".html", ".pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    data=f,
                    file_name="DataPilot_Executive_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_btn_pdf",
                )
        else:
            st.caption("PDF generation requires wkhtmltopdf installed at default system path.")

    # Live Preview of Generated HTML Report
    st.markdown("---")
    st.markdown("#### 👁️ Live HTML Report Preview")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as rf:
            html_content = rf.read()
            st.components.v1.html(html_content, height=650, scrolling=True)
