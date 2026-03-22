import streamlit as st
import os
from datetime import datetime

st.set_page_config(
    page_title="Nifty Options Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Dashboard ──────────────────────────────────────────
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161922;
        border-right: 1px solid #2a2d3e;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #a0a0a0;
        font-size: 14px;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #22c55e !important;
    }

    /* Custom Containers */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        background: #1a1d2e;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #2a2d3e;
    }
    .status-item {
        margin-right: 20px;
        font-size: 13px;
        color: #888;
    }
    .status-value {
        color: #3b82f6;
        font-weight: 600;
        margin-left: 5px;
    }
    .vix-regime {
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Signal Bar */
    .signal-bar {
        background: #0d2818;
        border: 1px solid #1a5c36;
        border-radius: 8px;
        padding: 15px 25px;
        margin-bottom: 20px;
    }
    .signal-title {
        color: #22c55e;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .signal-details {
        display: flex;
        gap: 30px;
        font-size: 14px;
        color: #b0b0b0;
    }

    /* Dashboard Grid Sections */
    .card {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 15px;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #2a2d3e; border-radius: 5px; }

    /* Hidden elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Nifty Intel")
    st.markdown("**Professional Grade Trading System**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠  Dashboard",
            "📂  Data Explorer",
            "🎯  Signal Engine",
            "📊  P&L Simulator",
            "🤖  Model Builder",
            "📓  Trade Journal",
            "📚  Strategy Guide",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(f"v3.2.0 • {datetime.now().strftime('%d %b %Y %H:%M')}")

# ── Page Router ─────────────────────────────────────────────────────────────
if "🏠" in page:
    from pages import dashboard
    dashboard.render()
elif "📂" in page:
    from pages import data_explorer
    data_explorer.render()
elif "🎯" in page:
    from pages import signal_engine
    signal_engine.render()
elif "📊" in page:
    from pages import pnl_simulator
    pnl_simulator.render()
elif "🤖" in page:
    from pages import model_builder
    model_builder.render()
elif "📓" in page:
    from pages import trade_journal
    trade_journal.render()
elif "📚" in page:
    from pages import strategy_guide
    strategy_guide.render()
