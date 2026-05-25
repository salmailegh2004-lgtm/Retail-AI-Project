import os
import shutil
import glob
import json
import html
from datetime import datetime
import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from agents.orchestrator_agent import OrchestratorAgent


UPLOAD_DIR = "uploaded_data"
OUTPUT_DIR = "outputs"

st.set_page_config(
    page_title="Retail AI Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #09090b;
    color: #e4e4e7;
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f0f12 !important;
    border-right: 1px solid #1f1f27;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif; }

/* Remove default padding */
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1280px; }

/* ── Nav pill tabs ── */
.nav-tabs {
    display: flex;
    gap: 6px;
    background: #111117;
    border: 1px solid #1f1f27;
    border-radius: 12px;
    padding: 5px;
    margin-bottom: 2rem;
    width: fit-content;
}
.nav-tab {
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    color: #71717a;
    border: none;
    background: transparent;
    transition: all .2s ease;
    letter-spacing: .3px;
    text-decoration: none;
    display: inline-block;
}
.nav-tab:hover { color: #e4e4e7; background: #1a1a22; }
.nav-tab.active { background: #2563eb; color: #fff; }

/* ── Page header ── */
.page-hero {
    margin-bottom: 2.5rem;
}
.page-hero h1 {
    font-size: 28px;
    font-weight: 700;
    color: #f4f4f5;
    margin: 0 0 6px;
    letter-spacing: -.5px;
}
.page-hero p {
    font-size: 14px;
    color: #71717a;
    margin: 0;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .5px;
    text-transform: uppercase;
    background: #172554;
    color: #93c5fd;
    border: 1px solid #1e3a8a;
    margin-bottom: 10px;
}

/* ── Section label ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #52525b;
    margin: 0 0 12px;
}
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #d4d4d8;
    margin: 0 0 16px;
    letter-spacing: -.3px;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #111117 !important;
    border: 1px solid #1f1f27 !important;
    border-radius: 14px !important;
    padding: 20px 22px !important;
    transition: border-color .2s;
}
[data-testid="stMetric"]:hover { border-color: #2563eb !important; }
[data-testid="stMetricLabel"] { color: #71717a !important; font-size: 12px !important; font-weight: 500 !important; letter-spacing: .4px; }
[data-testid="stMetricValue"] { color: #f4f4f5 !important; font-size: 26px !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Cards / panels ── */
.card {
    background: #111117;
    border: 1px solid #1f1f27;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.card-accent {
    background: linear-gradient(135deg, #0d1b3e 0%, #111117 60%);
    border: 1px solid #1e3a8a;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
}

/* ── Info / alert banners ── */
.info-banner {
    background: #0c1a3a;
    border: 1px solid #1e3a8a;
    border-left: 3px solid #2563eb;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #93c5fd;
    margin-bottom: 16px;
}
.success-banner {
    background: #052e16;
    border: 1px solid #166534;
    border-left: 3px solid #16a34a;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #86efac;
    margin-bottom: 16px;
}
.warn-banner {
    background: #1c1400;
    border: 1px solid #713f12;
    border-left: 3px solid #ca8a04;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #fde68a;
    margin-bottom: 16px;
}

/* ── Buttons ── */
.stButton > button {
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    height: 44px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: .2px;
    transition: all .2s !important;
}
.stButton > button:hover { background: #1d4ed8 !important; transform: translateY(-1px); box-shadow: 0 4px 14px rgba(37,99,235,.35) !important; }

.stDownloadButton > button {
    background: #111117 !important;
    color: #93c5fd !important;
    border: 1px solid #1e3a8a !important;
    border-radius: 10px !important;
    height: 44px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all .2s !important;
}
.stDownloadButton > button:hover { background: #0c1a3a !important; border-color: #2563eb !important; }

/* ── Expanders ── */
div[data-testid="stExpander"] {
    background: #111117 !important;
    border: 1px solid #1f1f27 !important;
    border-radius: 12px !important;
    overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #d4d4d8 !important;
    font-size: 13px !important;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1f1f27;
    margin: 24px 0;
}

/* ── Sidebar nav items ── */
.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    color: #71717a;
    cursor: pointer;
    margin-bottom: 2px;
    transition: all .15s;
    text-decoration: none;
}
.sidebar-nav-item:hover { background: #1a1a22; color: #e4e4e7; }
.sidebar-nav-item.active { background: #172554; color: #93c5fd; }
.sidebar-nav-icon { font-size: 15px; width: 20px; text-align: center; }

/* ── File upload area ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #2a2a35 !important;
    border-radius: 12px !important;
    background: #0c0c10 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
thead tr th { background: #1a1a22 !important; color: #71717a !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: .5px; text-transform: uppercase; }

/* ── Text area ── */
textarea {
    background: #0c0c10 !important;
    border: 1px solid #1f1f27 !important;
    border-radius: 10px !important;
    color: #d4d4d8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label { font-size: 13px !important; color: #a1a1aa !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #2563eb !important; }

/* ── JSON viewer ── */
[data-testid="stJson"] { background: #0c0c10 !important; border-radius: 10px !important; border: 1px solid #1f1f27 !important; }

/* ── Line chart ── */
[data-testid="stArrowVegaLiteChart"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def generate_pdf_report(text_path, pdf_path):
    with open(text_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )
    styles = getSampleStyleSheet()

    title_style  = ParagraphStyle("TitleStyle",  parent=styles["Title"],    fontSize=18, leading=22, spaceAfter=14)
    heading_style= ParagraphStyle("HeadingStyle",parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=8)
    body_style   = ParagraphStyle("BodyStyle",   parent=styles["BodyText"], fontSize=9,  leading=12, spaceAfter=4)

    story = [Paragraph("AI-Powered Retail Business Intelligence Report", title_style), Spacer(1, 10)]

    for line in report_text.split("\n"):
        line = html.escape(line)
        if line.strip() == "":
            story.append(Spacer(1, 6))
        elif line.startswith(("1.","2.","3.","4.","5.","6.")):
            story.append(Paragraph(line, heading_style))
        elif line.startswith("=") or line.startswith("-"*10):
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(line, body_style))

    doc.build(story)
    return pdf_path


# ─── Session state init ─────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "overview"
if "pipeline_output" not in st.session_state:
    st.session_state.pipeline_output = None
if "run_mode" not in st.session_state:
    st.session_state.run_mode = None


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 18px 14px 10px; margin-bottom: 8px;">
        <div style="font-size:17px; font-weight:700; color:#f4f4f5; letter-spacing:-.3px;">Retail AI</div>
        <div style="font-size:11px; color:#52525b; font-weight:500; letter-spacing:.5px; text-transform:uppercase; margin-top:2px;">Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Navigation (Overview, Run, Metrics, HITL, Report)
    pages = [
        ("overview",    "🏠", "Overview"),
        ("run",         "⚡", "Run Analysis"),
        ("metrics",     "📊", "Metrics & Results"),
        ("hitl",        "👤", "Human Validation"),
        ("report",      "📄", "Report & Export"),
    ]

    for page_id, icon, label in pages:
        is_active = st.session_state.page == page_id
        if st.sidebar.button(f"{icon}  {label}", key=f"nav_{page_id}",
                             use_container_width=True):
            st.session_state.page = page_id
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Analysis config
    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)

    analysis_choice = st.radio(
        "Analysis mode",
        ["Forecasting & Peak Detection", "Anomaly Detection"],
        label_visibility="collapsed"
    )
    st.session_state.analysis_mode = (
        "forecasting" if analysis_choice == "Forecasting & Peak Detection" else "anomaly"
    )

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    if st.button("⚡  Run Analysis", use_container_width=True):
        st.session_state.page = "run"
        st.session_state.trigger_run = True
        st.rerun()

    # Status indicator
    if st.session_state.pipeline_output:
        st.markdown("""
        <div class="success-banner" style="margin-top:16px; font-size:12px;">
            ✓ Pipeline completed successfully
        </div>
        """, unsafe_allow_html=True)


# ─── Page: Overview ─────────────────────────────────────────────────────────────
def page_overview():
    st.markdown("""
    <div class="page-hero">
        <div class="badge">Retail Intelligence</div>
        <h1>Welcome to Retail AI</h1>
        <p>Multi-agent system for demand forecasting, anomaly detection, and automated business reporting.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Agents", "4")
    c2.metric("Analysis Modes", "2")
    c3.metric("Output Formats", "3")
    c4.metric("Model Type", "LSTM")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
        steps = [
            ("01", "Upload Data",      "Upload one or more CSV files containing retail transaction data."),
            ("02", "Select Mode",      "Choose between Demand Forecasting or Anomaly Detection analysis."),
            ("03", "Run Inference",    "The orchestrator dispatches specialized AI agents to process your data."),
            ("04", "Review Results",   "Explore metrics, training history, and interactive visualizations."),
            ("05", "Export Report",    "Download the generated report in TXT, JSON, or PDF format."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="card" style="display:flex;gap:18px;align-items:flex-start;padding:16px 20px;margin-bottom:10px;">
                <div style="min-width:32px;height:32px;background:#172554;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#93c5fd;font-family:'DM Mono',monospace;">{num}</div>
                <div>
                    <div style="font-size:14px;font-weight:600;color:#f4f4f5;margin-bottom:3px;">{title}</div>
                    <div style="font-size:13px;color:#71717a;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-title">Analysis Modes</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-accent" style="margin-bottom:12px;">
            <div style="font-size:13px;font-weight:700;color:#93c5fd;margin-bottom:6px;">📈 Forecasting & Peak Detection</div>
            <div style="font-size:13px;color:#a1a1aa;line-height:1.6;">
                Uses an LSTM neural network to predict future demand, identify seasonal peaks, 
                and generate inventory recommendations.
            </div>
        </div>
        <div class="card" style="border-color:#2a1a3a;">
            <div style="font-size:13px;font-weight:700;color:#c084fc;margin-bottom:6px;">🔍 Anomaly Detection</div>
            <div style="font-size:13px;color:#a1a1aa;line-height:1.6;">
                Autoencoder-based anomaly detection flags unusual transactions, 
                classifies risk levels, and identifies anomaly types.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.pipeline_output:
            mode = st.session_state.pipeline_output.get("mode", "—")
            st.markdown(f"""
            <div class="success-banner">
                ✓ Last run: <strong>{mode}</strong> mode completed
            </div>
            """, unsafe_allow_html=True)


# ─── Page: Run Analysis ──────────────────────────────────────────────────────────
def page_run():
    st.markdown("""
    <div class="page-hero">
        <div class="badge">Execute</div>
        <h1>Run Analysis</h1>
        <p>Upload your CSV data and trigger the multi-agent inference pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    if not uploaded_files:
        st.markdown("""
        <div class="warn-banner">
            ⚠ No files uploaded. Please upload at least one CSV file from the sidebar.
        </div>
        """, unsafe_allow_html=True)
        return

    # Dataset preview
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    for file in uploaded_files:
        df = pd.read_csv(file)
        with st.expander(f"📂 {file.name}  ·  {df.shape[0]:,} rows × {df.shape[1]} columns", expanded=True):
            st.dataframe(df.head(8), use_container_width=True)
        file.seek(0)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    mode_label = "Forecasting & Peak Detection" if st.session_state.analysis_mode == "forecasting" else "Anomaly Detection"
    st.markdown(f"""
    <div class="card-accent">
        <div style="font-size:12px;font-weight:600;color:#52525b;letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px;">Ready to Execute</div>
        <div style="display:flex;gap:32px;">
            <div>
                <div style="font-size:11px;color:#52525b;font-weight:500;margin-bottom:3px;">MODE</div>
                <div style="font-size:15px;font-weight:700;color:#93c5fd;">{mode_label}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#52525b;font-weight:500;margin-bottom:3px;">FILES</div>
                <div style="font-size:15px;font-weight:700;color:#f4f4f5;">{len(uploaded_files)} CSV</div>
            </div>
            <div>
                <div style="font-size:11px;color:#52525b;font-weight:500;margin-bottom:3px;">ENGINE</div>
                <div style="font-size:15px;font-weight:700;color:#f4f4f5;">Saved LSTM Model</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    trigger = getattr(st.session_state, "trigger_run", False)

    if st.button("⚡  Launch Pipeline", use_container_width=False) or trigger:
        st.session_state.trigger_run = False

        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        with st.spinner("Running inference pipeline..."):
            orchestrator = OrchestratorAgent(
                input_path=UPLOAD_DIR,
                output_dir=OUTPUT_DIR,
                mode=st.session_state.analysis_mode
            )
            pipeline_output = orchestrator.run()

        if pipeline_output is None:
            st.markdown("""
            <div style="background:#1c0505;border:1px solid #7f1d1d;border-left:3px solid #dc2626;border-radius:8px;padding:12px 16px;font-size:13px;color:#fca5a5;">
                ✕ Pipeline failed. Check the historical file logs below for details.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.session_state.pipeline_output = pipeline_output
            st.session_state.run_mode = pipeline_output.get("mode")
            st.markdown("""
            <div class="success-banner">
                ✓ Pipeline completed successfully. Navigate to <strong>Metrics & Results</strong> to explore the output.
            </div>
            """, unsafe_allow_html=True)

    # Log history
    log_path = os.path.join(OUTPUT_DIR, "orchestrator_logs.json")
    if os.path.exists(log_path):
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        with st.expander("📄 View Orchestrator File Logs (History)", expanded=False):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    st.json(json.load(f))
            except Exception as e:
                st.error(f"Could not load log history: {e}")


# ─── Page: Metrics & Results ─────────────────────────────────────────────────────
def page_metrics():
    st.markdown("""
    <div class="page-hero">
        <div class="badge">Results</div>
        <h1>Metrics & Results</h1>
        <p>Key performance indicators and summary statistics from the inference run.</p>
    </div>
    """, unsafe_allow_html=True)

    pipeline_output = st.session_state.pipeline_output

    if not pipeline_output:
        st.markdown('<div class="info-banner">ℹ Run the analysis pipeline first to see results here.</div>', unsafe_allow_html=True)
        return

    mode = pipeline_output.get("mode")

    # ── Forecasting metrics ──
    if mode == "forecasting":
        summary  = pipeline_output.get("forecasting_summary", {})
        trend    = summary.get("trend_analysis", {})
        inventory= summary.get("inventory_requirements", {})

        st.markdown('<div class="section-label">Forecast Overview</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Trend",   trend.get("trend", "N/A"))
        c2.metric("Demand Change",     f"{trend.get('change_percentage', 0):.2f}%")
        c3.metric("Detected Peaks",    summary.get("number_of_detected_future_peaks", 0))

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Inventory Predictions</div>', unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        c4.metric("Total Predicted Demand",   f"{inventory.get('total_predicted_demand', 0):,.0f} units")
        c5.metric("Recommended Inventory",    f"{inventory.get('total_recommended_inventory', 0):,.0f} units")

        # ─────────────────────────────────────────────
        # SALES VISUALIZATION
        # ─────────────────────────────────────────────
        forecast_path = os.path.join(OUTPUT_DIR, "forecasting_results.csv")
        test_path = os.path.join(OUTPUT_DIR, "test_actual_vs_predicted.csv")

        # Future forecast sales
        if os.path.exists(forecast_path):
            forecast_df = pd.read_csv(forecast_path)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Future Sales Forecast</div>', unsafe_allow_html=True)

            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Average Predicted Sales", f"{forecast_df['predicted_sales'].mean():.2f}")
            fc2.metric("Maximum Predicted Sales", f"{forecast_df['predicted_sales'].max():.2f}")
            fc3.metric("Minimum Predicted Sales", f"{forecast_df['predicted_sales'].min():.2f}")

            st.line_chart(forecast_df.set_index("date")["predicted_sales"])

            with st.expander("View Forecast Data"):
                st.dataframe(forecast_df, use_container_width=True)

        # Actual vs Predicted sales
        if os.path.exists(test_path):
            compare_df = pd.read_csv(test_path)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Actual vs Predicted Sales Comparison</div>', unsafe_allow_html=True)

            real_sales = compare_df["actual_sales"].sum()
            predicted_sales = compare_df["predicted_sales"].sum()

            ac1, ac2 = st.columns(2)
            ac1.metric("Total Actual Sales", f"{real_sales:,.2f}")
            ac2.metric("Total Predicted Sales", f"{predicted_sales:,.2f}")

            chart_df = compare_df[["date", "actual_sales", "predicted_sales"]].copy()
            chart_df = chart_df.set_index("date")
            st.line_chart(chart_df)

            with st.expander("View Comparison Data"):
                st.dataframe(compare_df, use_container_width=True)

    # ── Anomaly metrics ──
    elif mode == "anomaly":
        summary = pipeline_output.get("anomaly_summary", {})
        rate    = summary.get("anomaly_rate", 0)

        st.markdown('<div class="section-label">Detection Summary</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Records Analyzed",    summary.get("total_records_analyzed", "N/A"))
        c2.metric("Anomalies Detected",  summary.get("total_anomalies_detected", "N/A"))
        c3.metric("Anomaly Rate",        f"{rate*100:.2f}%" if isinstance(rate,(int,float)) else "N/A")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
            st.json(summary.get("risk_distribution", {}))
        with col2:
            st.markdown('<div class="section-title">Anomaly Type Distribution</div>', unsafe_allow_html=True)
            st.json(summary.get("anomaly_type_distribution", {}))


# ─── Page: Report & Export ────────────────────────────────────────────────────────
def page_report():
    from agents.report_agent import ReportGenerationAgent

    st.markdown("""
    <div class="page-hero">
        <div class="badge">Export</div>
        <h1>Report & Export</h1>
        <p>Generate the final report from analysis results and human validation, then download it.</p>
    </div>
    """, unsafe_allow_html=True)

    pipeline_output      = st.session_state.pipeline_output
    human_validation     = pipeline_output.get("human_validation_output") if pipeline_output else None

    report_path = os.path.join(OUTPUT_DIR, "final_business_report.txt")
    json_path   = os.path.join(OUTPUT_DIR, "final_business_report.json")
    pdf_path    = os.path.join(OUTPUT_DIR, "final_business_report.pdf")

    if not pipeline_output:
        st.markdown('<div class="info-banner">ℹ Step 1 — Run the analysis pipeline first.</div>', unsafe_allow_html=True)
        return

    if not human_validation:
        st.markdown("""
        <div class="warn-banner">
            ⚑ Step 2 — Human validation not submitted yet.<br>
            Go to <strong>👤 Human Validation</strong>, review the recommendations, then come back here.
        </div>
        """, unsafe_allow_html=True)
        return

    summary = human_validation.get("validation_summary", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Recommendations", summary.get("total_recommendations", "—"))
    c2.metric("Approved",        summary.get("approved_recommendations", "—"))
    c3.metric("Rejected",        summary.get("rejected_recommendations", "—"))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.button("⚙  Generate Final Report", use_container_width=False):
        with st.spinner("Generating report…"):
            try:
                report_agent = ReportGenerationAgent(
                    mode=pipeline_output.get("mode"),
                    forecasting_summary=pipeline_output.get("forecasting_summary"),
                    anomaly_summary=pipeline_output.get("anomaly_summary"),
                    recommendation_output=pipeline_output.get("recommendation_output"),
                    human_validation_output=human_validation,
                    output_dir=OUTPUT_DIR,
                    ollama_model="llama3.2"
                )
                final_report = report_agent.run()
                st.session_state.pipeline_output["final_report"] = final_report
                st.markdown('<div class="success-banner">✓ Report generated successfully.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                return

    if not os.path.exists(report_path):
        st.markdown('<div class="info-banner">ℹ Click <strong>Generate Final Report</strong> above to produce the report.</div>', unsafe_allow_html=True)
        return

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    st.markdown('<div class="section-title">Report Preview</div>', unsafe_allow_html=True)
    st.text_area("", report_text, height=460, label_visibility="collapsed")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Download</div>', unsafe_allow_html=True)
    col1, col2, col3, _ = st.columns([1, 1, 1, 2])

    with col1:
        st.download_button(
            "⬇ TXT", data=report_text,
            file_name="final_business_report.txt", mime="text/plain",
            use_container_width=True
        )
    with col2:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_text = f.read()
            st.download_button(
                "⬇ JSON", data=json_text,
                file_name="final_business_report.json", mime="application/json",
                use_container_width=True
            )
    with col3:
        try:
            generate_pdf_report(report_path, pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇ PDF", data=f,
                    file_name="final_business_report.pdf", mime="application/pdf",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")


# ─── Page: Human-in-the-Loop Validation ─────────────────────────────────────────
def page_hitl():
    st.markdown("""
    <div class="page-hero">
        <div class="badge">Human Review</div>
        <h1>Human Validation</h1>
        <p>Review and approve AI-generated recommendations before finalizing the report.</p>
    </div>
    """, unsafe_allow_html=True)

    pipeline_output = st.session_state.pipeline_output

    if not pipeline_output:
        st.markdown('<div class="info-banner">ℹ Run the analysis pipeline first to generate recommendations.</div>', unsafe_allow_html=True)
        return

    recommendations = pipeline_output.get("recommendation_output", {}).get("recommendations", [])

    if not recommendations:
        st.markdown('<div class="info-banner">ℹ No recommendations found in the pipeline output.</div>', unsafe_allow_html=True)
        return

    if "hitl_decisions" not in st.session_state or len(st.session_state.hitl_decisions) != len(recommendations):
        st.session_state.hitl_decisions = ["Approved"] * len(recommendations)

    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)

    for i, rec in enumerate(recommendations):
        with st.expander(
            f"[{rec.get('priority','—')}]  {rec.get('category','—')} — {rec.get('recommendation','')[:70]}…",
            expanded=True
        ):
            st.write(rec.get("recommendation", "—"))

            decision = st.radio(
                "Decision",
                ["Approved", "Rejected"],
                index=0 if st.session_state.hitl_decisions[i] == "Approved" else 1,
                key=f"decision_{i}",
                horizontal=True,
            )
            st.session_state.hitl_decisions[i] = decision

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.button("✓  Submit Validation", use_container_width=False):
        n_approved = st.session_state.hitl_decisions.count("Approved")
        n_rejected = st.session_state.hitl_decisions.count("Rejected")

        validation_results = [
            {
                "category":        rec.get("category", "—"),
                "priority":        rec.get("priority", "—"),
                "recommendation":  rec.get("recommendation", "—"),
                "approval_status": st.session_state.hitl_decisions[i],
            }
            for i, rec in enumerate(recommendations)
        ]

        validation_output = {
            "validation_results": validation_results,
            "validation_summary": {
                "total_recommendations":    len(recommendations),
                "approved_recommendations": n_approved,
                "rejected_recommendations": n_rejected,
                "final_status": "Validated" if n_rejected == 0 else "Partially Validated",
            },
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "human_validation_report.json"), "w") as f:
            json.dump(validation_output, f, indent=4)

        st.session_state.pipeline_output["human_validation_output"] = validation_output

        st.markdown(f"""
        <div class="success-banner">
            ✓ Validation submitted — <strong>{n_approved}</strong> approved · <strong>{n_rejected}</strong> rejected.
        </div>
        """, unsafe_allow_html=True)


# ─── Router ──────────────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "overview":
    page_overview()
elif page == "run":
    page_run()
elif page == "metrics":
    page_metrics()
elif page == "hitl":
    page_hitl()
elif page == "report":
    page_report()
