"""
Streamlit Web Dashboard for SupportIQ.
Provides executive KPI metrics, interactive natural-language querying, and operational anomaly detection.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.analytics import analytics_engine
from app.anomaly_detector import anomaly_detector
from app.config import settings
from app.data_loader import data_loader
from app.llm_engine import get_llm_provider
from app.query_engine import query_engine
from app.utils import time_execution

# Page Configuration
st.set_page_config(
    page_title="SupportIQ — Customer Support Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E232F;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2D3748;
        text-align: center;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    .badge-critical { color: #FF4D4D; font-weight: bold; }
    .badge-high { color: #FFA500; font-weight: bold; }
    .badge-medium { color: #FFD700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("⚡ SupportIQ")
    st.caption("AI-Powered Support Intelligence System")
    st.divider()

    st.subheader("System Status")
    try:
        df = data_loader.get_data()
        st.success(f"✓ Dataset: Loaded ({len(df)} records)")
    except Exception as e:
        st.error(f"Dataset Error: {e}")

    llm_configured = bool(settings.GROQ_API_KEY) or settings.LLM_PROVIDER == "mock"
    if settings.GROQ_API_KEY:
        st.info(f"Model: {settings.GROQ_MODEL} (Groq)")
    else:
        st.warning("Provider: Mock/Local Mode (No GROQ_API_KEY set)")

    ref_dt = data_loader.get_reference_timestamp()
    st.caption(f"Reference Time: {ref_dt.strftime('%Y-%m-%d %H:%M')}")
    st.caption(f"Mode: {settings.REFERENCE_TIME_MODE}")

    st.divider()
    st.markdown("### Quick Links")
    st.markdown("- [Swagger API Docs](http://localhost:8000/docs)")
    st.markdown("- [ReDoc](http://localhost:8000/redoc)")


# Main Dashboard
st.title("📊 SupportIQ Analytics & AI Assistant")
st.write("Ask natural-language questions about support tickets or inspect real-time operational anomalies.")

# 1. Executive KPI Summary
kpis = analytics_engine.get_kpi_summary()
anomalies_res = anomaly_detector.detect_all()

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Total Tickets", kpis.total_tickets)
with c2:
    st.metric("Open Tickets", kpis.open_tickets)
with c3:
    st.metric("Resolved Tickets", kpis.resolved_tickets)
with c4:
    st.metric("Critical Tickets", kpis.critical_tickets)
with c5:
    st.metric("Avg Rating", f"{kpis.avg_customer_rating or 0:.2f} / 5.0")
with c6:
    st.metric("Total Anomalies", anomalies_res.total, delta=f"{anomalies_res.summary.by_severity.get('CRITICAL', 0)} Critical", delta_color="inverse")

st.divider()

# 2. Natural Language Query Interface
st.subheader("💬 Ask AI Support Assistant")
st.write("Type a natural language question or select one of the suggested sample questions below:")

sample_queries = [
    "How many tickets are currently open?",
    "Which agent has the lowest average customer rating?",
    "Which agent resolved the most tickets?",
    "What is the average customer rating for Technical tickets?",
    "Show all Critical tickets not resolved within 12 hours.",
    "What is the company's annual revenue?",
]

# Initialize session state for query text
if "query_input_text" not in st.session_state:
    st.session_state["query_input_text"] = sample_queries[0]

def select_sample_query(query_text: str):
    st.session_state["query_input_text"] = query_text
    st.session_state["trigger_ask"] = True

# Quick selection buttons
chip_cols = st.columns(3)
for i, sq in enumerate(sample_queries):
    col = chip_cols[i % 3]
    col.button(
        f"👉 {sq}",
        key=f"sq_btn_{i}",
        use_container_width=True,
        on_click=select_sample_query,
        args=(sq,),
    )

user_query = st.text_input(
    "Enter your question about tickets:",
    key="query_input_text",
    placeholder="e.g. How many open high priority tickets do we have?",
)

ask_btn = st.button("🚀 Ask AI", type="primary")
should_ask = ask_btn or st.session_state.pop("trigger_ask", False)

if should_ask:
    query_to_run = user_query.strip()
    if not query_to_run:
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Processing natural language query..."):
            try:
                with time_execution() as timer:
                    planner = get_llm_provider()
                    plan = planner.generate_plan(query_to_run)
                    raw_result, meta, answer = query_engine.execute_plan(plan)

                # Display Answer
                if not plan.is_supported:
                    st.warning(f"⚠️ {answer}")
                else:
                    st.success(f"**Answer:** {answer}")

                    # Render table if result is list or dict
                    if isinstance(raw_result, list) and raw_result:
                        st.dataframe(pd.DataFrame(raw_result), use_container_width=True)
                    elif isinstance(raw_result, dict) and raw_result:
                        if isinstance(list(raw_result.values())[0], dict):
                            st.dataframe(pd.DataFrame.from_dict(raw_result, orient="index"), use_container_width=True)
                        else:
                            st.dataframe(pd.DataFrame(list(raw_result.items()), columns=["Group", "Count"]), use_container_width=True)

                with st.expander(" Inspect Query Plan & Execution Metadata"):
                    st.json({
                        "question": query_to_run,
                        "query_plan": plan.model_dump(),
                        "execution_metadata": {
                            "execution_time_ms": timer["elapsed_ms"],
                            "rows_scanned": meta.get("rows_scanned"),
                            "matched_rows": meta.get("matched_rows"),
                            "operation": plan.operation.value,
                            "llm_provider": settings.LLM_PROVIDER,
                        }
                    })
            except Exception as e:
                st.error(f"LLM service unavailable or error executing query: {str(e)}")

st.divider()

# 3. Operational Anomalies Section
st.subheader("🚨 Operational Anomalies")
st.write("Identified via deterministic SLA rules, IQR statistical bounds, and Isolation Forest ML.")

# Filter controls
f1, f2, f3 = st.columns(3)
with f1:
    sev_choice = st.selectbox("Severity Filter", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
with f2:
    method_choice = st.selectbox("Detection Method Filter", ["ALL", "RULE_SLA_BREACH", "IQR_STATISTICAL", "ISOLATION_FOREST_ML"])
with f3:
    include_ml_toggle = st.checkbox("Include Isolation Forest ML Anomalies", value=True)

active_sev = None if sev_choice == "ALL" else sev_choice
active_method = None if method_choice == "ALL" else method_choice

anom_data = anomaly_detector.detect_all(
    include_ml=include_ml_toggle,
    severity_filter=active_sev,
    method_filter=active_method,
)

st.markdown(
    f"Found **{anom_data.total}** anomalies matching filters "
    f"(Critical: {anom_data.summary.by_severity.get('CRITICAL', 0)}, "
    f"High: {anom_data.summary.by_severity.get('HIGH', 0)}, "
    f"Medium: {anom_data.summary.by_severity.get('MEDIUM', 0)})"
)

if anom_data.anomalies:
    table_rows = []
    for a in anom_data.anomalies:
        table_rows.append({
            "Ticket ID": a.ticket_id,
            "Severity": a.severity.value,
            "Anomaly Type": a.anomaly_type,
            "Method": a.detection_method,
            "Reason": a.reason,
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
else:
    st.info("No anomalies found for the selected filters.")

st.divider()

# 4. Dataset Overview & Visualizations
st.subheader("📈 Dataset Overview & Distributions")
v1, v2 = st.columns(2)

with v1:
    st.markdown("##### Tickets by Priority")
    pri_df = pd.DataFrame(list(kpis.priorities.items()), columns=["Priority", "Tickets"]).set_index("Priority")
    st.bar_chart(pri_df)

with v2:
    st.markdown("##### Tickets by Category")
    cat_df = pd.DataFrame(list(kpis.categories.items()), columns=["Category", "Tickets"]).set_index("Category")
    st.bar_chart(cat_df)

st.markdown("##### Agent Performance Leaderboard")
agent_perf = analytics_engine.get_agent_performance()
st.dataframe(pd.DataFrame(agent_perf), use_container_width=True)
