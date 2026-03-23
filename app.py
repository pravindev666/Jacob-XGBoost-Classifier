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
    st.markdown('<div style="font-size:11px;color:#888;margin-top:-5px;margin-bottom:10px;">Your AI-powered guide to smarter Nifty options trading</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠  Dashboard",
            "🔭  Multi-Horizon",
            "📂  Data Explorer",
            "🎯  Signal Engine",
            "📊  P&L Simulator",
            "🤖  Model Builder",
            "📓  Trade Journal",
            "📚  Strategy Guide",
        ],
        label_visibility="collapsed",
    )

    # Kid-friendly page descriptions
    page_tips = {
        "🏠": "📍 Your home screen — see today's signal, charts, and live data at a glance.",
        "🔭": "🔍 7 AI models vote on market direction across different timeframes.",
        "📂": "📁 Explore your raw data files and check if everything is loaded correctly.",
        "🎯": "🎯 Enter your model's prediction and get the exact trade to execute.",
        "📊": "🎲 Simulate thousands of trading scenarios to understand your edge.",
        "🤖": "🧠 Train the AI model on your data — this is where the magic happens.",
        "📓": "📝 Log every trade you make and track your performance over time.",
        "📚": "📖 Learn how each strategy works with interactive payoff diagrams.",
    }
    for emoji, tip in page_tips.items():
        if emoji in page:
            st.markdown(f'<div style="font-size:11px;color:#888;margin-top:5px;padding:6px 8px;background:#1e2130;border-radius:6px;">{tip}</div>', unsafe_allow_html=True)
            break

    st.markdown("---")
    st.caption(f"v3.3.0 • {datetime.now().strftime('%d %b %Y %H:%M')}")

# ── Page Router with Error Boundaries ────────────────────────────────────────
def _safe_render(page_module, page_name):
    """Render a page with error boundary — a crash in one page won't kill the app."""
    try:
        page_module.render()
    except Exception as e:
        st.error(f"⚠️ **{page_name}** encountered an error:")
        st.code(str(e))
        st.info("💡 Try refreshing the page, or check if your data files are in the `data/` folder.")
        import traceback
        with st.expander("Full error details", expanded=False):
            st.code(traceback.format_exc())

if "🏠" in page:
    from pages import dashboard
    _safe_render(dashboard, "Dashboard")
elif "🔭" in page:
    from pages import multi_horizon
    _safe_render(multi_horizon, "Multi-Horizon")
elif "📂" in page:
    from pages import data_explorer
    _safe_render(data_explorer, "Data Explorer")
elif "🎯" in page:
    from pages import signal_engine
    _safe_render(signal_engine, "Signal Engine")
elif "📊" in page:
    from pages import pnl_simulator
    _safe_render(pnl_simulator, "P&L Simulator")
elif "🤖" in page:
    from pages import model_builder
    _safe_render(model_builder, "Model Builder")
elif "📓" in page:
    from pages import trade_journal
    _safe_render(trade_journal, "Trade Journal")
elif "📚" in page:
    from pages import strategy_guide
    _safe_render(strategy_guide, "Strategy Guide")

