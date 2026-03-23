import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime

DARK_BG = "#0e1117"
CARD_BG = "#1a1d2e"


def init_journal():
    if "trades" not in st.session_state:
        st.session_state.trades = pd.DataFrame([
            {"Date": "2024-12-05", "Strategy": "Bull Put Spread",
             "Entry": "Sell 21700PE / Buy 21500PE", "Lots": 1,
             "Confidence": 62, "VIX": 16.8, "DTE": 7,
             "Premium": 2500, "PnL": 2500, "Status": "Closed ✅",
             "Notes": "Nifty held support, expired worthless"},
            {"Date": "2024-12-12", "Strategy": "Buy ATM Call",
             "Entry": "22000CE @ ₹180", "Lots": 1,
             "Confidence": 74, "VIX": 14.2, "DTE": 5,
             "Premium": 2700, "PnL": 3600, "Status": "Closed ✅",
             "Notes": "Strong breakout, booked at +50%"},
            {"Date": "2024-12-19", "Strategy": "Bear Call Spread",
             "Entry": "Sell 22200CE / Buy 22400CE", "Lots": 1,
             "Confidence": 60, "VIX": 22.1, "DTE": 4,
             "Premium": 2200, "PnL": 2200, "Status": "Closed ✅",
             "Notes": "VIX>20 regime, credit spread worked"},
            {"Date": "2025-01-08", "Strategy": "Buy ATM Put",
             "Entry": "21800PE @ ₹160", "Lots": 1,
             "Confidence": 71, "VIX": 13.5, "DTE": 6,
             "Premium": -2400, "PnL": -1440, "Status": "Closed ❌",
             "Notes": "Stopped at -50% loss, Nifty bounced"},
            {"Date": "2025-01-15", "Strategy": "Bull Put Spread",
             "Entry": "Sell 21500PE / Buy 21300PE", "Lots": 2,
             "Confidence": 58, "VIX": 19.4, "DTE": 7,
             "Premium": 1800, "PnL": 1800, "Status": "Closed ✅",
             "Notes": "Held through expiry"},
            {"Date": "2025-01-24", "Strategy": "Buy ATM Call",
             "Entry": "22500CE @ ₹200", "Lots": 1,
             "Confidence": 77, "VIX": 12.8, "DTE": 4,
             "Premium": 3000, "PnL": -1500, "Status": "Closed ❌",
             "Notes": "Overconfident, market reversed on FII data"},
        ])


def render():
    init_journal()
    st.markdown("## 📓 Trade Journal")
    st.markdown("Track every trade, learn from patterns, improve your edge.")
    st.markdown(
        '<div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#b0b0b0;">'
        '💡 <b style="color:#e0e0e0;">What is a trade journal?</b> Think of this like a diary for your trading. '
        'Every time you make a trade, you write down why you did it and what happened. '
        'Good traders learn from both their wins and losses!'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Add Trade ─────────────────────────────────────────────────────────────
    with st.expander("➕ Log New Trade", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            t_date = st.date_input("Trade Date", date.today())
            t_strat = st.selectbox("Strategy", [
                "Bull Put Spread", "Bear Call Spread",
                "Buy ATM Call", "Buy ATM Put"
            ])
        with c2:
            t_conf = st.slider("Confidence (%)", 40, 90, 60)
            t_vix = st.slider("VIX at entry", 10.0, 40.0, 18.0, 0.5)
        with c3:
            t_lots = st.number_input("Lots", 1, 10, 1)
            t_dte = st.number_input("DTE at entry", 1, 30, 7)
            t_pnl = st.number_input("Final P&L (₹, negative = loss)", -20000, 20000, 0, 100)
        with c4:
            t_entry = st.text_input("Entry description", "Sell XXPE / Buy XXPE")
            t_status = st.selectbox("Status", ["Open 🔵", "Closed ✅", "Closed ❌"])
            t_notes = st.text_area("Notes", height=80)

        if st.button("💾 Save Trade", type="primary"):
            new_row = pd.DataFrame([{
                "Date": str(t_date), "Strategy": t_strat, "Entry": t_entry,
                "Lots": t_lots, "Confidence": t_conf, "VIX": t_vix, "DTE": t_dte,
                "Premium": abs(t_pnl), "PnL": t_pnl, "Status": t_status, "Notes": t_notes
            }])
            st.session_state.trades = pd.concat(
                [st.session_state.trades, new_row], ignore_index=True)
            st.success("Trade logged!")

    trades = st.session_state.trades.copy()
    trades["PnL"] = pd.to_numeric(trades["PnL"], errors="coerce")
    trades["Confidence"] = pd.to_numeric(trades["Confidence"], errors="coerce")
    trades["VIX"] = pd.to_numeric(trades["VIX"], errors="coerce")
    closed = trades[trades["Status"].str.contains("Closed")]

    # ── Summary Metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    total_pnl = closed["PnL"].sum()
    wins = closed[closed["PnL"] > 0]
    losses = closed[closed["PnL"] < 0]
    win_rate = len(wins) / len(closed) * 100 if len(closed) > 0 else 0
    avg_win = wins["PnL"].mean() if len(wins) > 0 else 0
    avg_loss = losses["PnL"].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins["PnL"].sum() / losses["PnL"].sum()) if losses["PnL"].sum() != 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total P&L", f"₹{total_pnl:,.0f}",
              "Profit" if total_pnl > 0 else "Loss")
    m2.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W / {len(losses)}L")
    m3.metric("Avg Win", f"₹{avg_win:,.0f}")
    m4.metric("Avg Loss", f"₹{avg_loss:,.0f}")
    m5.metric("Profit Factor", f"{profit_factor:.2f}")

    # ── Charts ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cumulative P&L")
        pnl_sorted = closed.sort_values("Date")
        cum_pnl = pnl_sorted["PnL"].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=list(range(1, len(cum_pnl) + 1)),
            y=cum_pnl.values,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(
                color=["#22c55e" if p > 0 else "#ef4444"
                       for p in pnl_sorted["PnL"].values],
                size=8
            ),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)"
        ))
        fig_cum.add_hline(y=0, line_color="#555", line_width=1)
        fig_cum.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font_color="#b0b0b0", height=280, showlegend=False,
            xaxis_title="Trade #", yaxis_title="Cumulative P&L (₹)",
            margin=dict(l=10, r=10, t=10, b=40)
        )
        fig_cum.update_xaxes(gridcolor="#1e2130")
        fig_cum.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig_cum, use_container_width=True)

    with col2:
        st.markdown("#### P&L by Strategy")
        strat_pnl = closed.groupby("Strategy")["PnL"].sum().reset_index()
        fig_strat = go.Figure(go.Bar(
            x=strat_pnl["Strategy"],
            y=strat_pnl["PnL"],
            marker_color=["#22c55e" if p > 0 else "#ef4444"
                          for p in strat_pnl["PnL"]],
            text=[f"₹{p:,.0f}" for p in strat_pnl["PnL"]],
            textposition="outside"
        ))
        fig_strat.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font_color="#b0b0b0", height=280, showlegend=False,
            yaxis_title="Total P&L (₹)",
            margin=dict(l=10, r=10, t=10, b=60)
        )
        fig_strat.update_xaxes(tickangle=20)
        fig_strat.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig_strat, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Confidence vs P&L")
        fig_scatter = go.Figure(go.Scatter(
            x=closed["Confidence"],
            y=closed["PnL"],
            mode="markers",
            marker=dict(
                color=["#22c55e" if p > 0 else "#ef4444"
                       for p in closed["PnL"]],
                size=12, opacity=0.8
            ),
            text=closed["Strategy"],
        ))
        fig_scatter.add_hline(y=0, line_color="#555")
        fig_scatter.add_vline(x=55, line_dash="dash", line_color="#888",
                              annotation_text="Min threshold")
        fig_scatter.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font_color="#b0b0b0", height=260, showlegend=False,
            xaxis_title="Confidence (%)", yaxis_title="P&L (₹)",
            margin=dict(l=10, r=10, t=10, b=40)
        )
        fig_scatter.update_xaxes(gridcolor="#1e2130")
        fig_scatter.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col4:
        st.markdown("#### VIX vs P&L")
        fig_vix = go.Figure(go.Scatter(
            x=closed["VIX"],
            y=closed["PnL"],
            mode="markers",
            marker=dict(
                color=["#22c55e" if p > 0 else "#ef4444"
                       for p in closed["PnL"]],
                size=12, opacity=0.8
            ),
            text=closed["Strategy"],
        ))
        fig_vix.add_hline(y=0, line_color="#555")
        fig_vix.add_vline(x=20, line_dash="dash", line_color="#f59e0b",
                          annotation_text="VIX 20 threshold")
        fig_vix.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font_color="#b0b0b0", height=260, showlegend=False,
            xaxis_title="India VIX", yaxis_title="P&L (₹)",
            margin=dict(l=10, r=10, t=10, b=40)
        )
        fig_vix.update_xaxes(gridcolor="#1e2130")
        fig_vix.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig_vix, use_container_width=True)

    # ── Full Trade Log ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Full Trade Log")

    filter_strat = st.multiselect(
        "Filter by strategy",
        options=trades["Strategy"].unique().tolist(),
        default=trades["Strategy"].unique().tolist()
    )
    filtered = trades[trades["Strategy"].isin(filter_strat)]

    def style_pnl(val):
        if isinstance(val, (int, float)):
            return f"color: {'#22c55e' if val > 0 else '#ef4444' if val < 0 else '#b0b0b0'}"
        return ""

    styled = filtered.style.map(style_pnl, subset=["PnL"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Monthly Review Template ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Monthly Review")
    st.markdown("Answer these each month to improve your edge:")
    review_qs = [
        "Did I follow all 10 golden rules?",
        "Were my predictions above 55% confidence accurate?",
        "Did I manage risk properly (max ₹2K per trade)?",
        "Did I use credit spreads when VIX > 20?",
        "What was my win rate this month?",
        "What can I improve next month?",
    ]

    if "monthly_review" not in st.session_state:
        st.session_state.monthly_review = {q: "" for q in review_qs}

    for q in review_qs:
        val = st.text_area(q, value=st.session_state.monthly_review[q], key=f"review_{q}", height=60)
        st.session_state.monthly_review[q] = val

    if st.button("💾 Save Review Notes"):
        st.success("Review notes saved for this session!")
