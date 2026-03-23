import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from utils.logic import get_signal, get_trade_setup

DARK_BG = "#0e1117"
CARD_BG = "#1a1d2e"


def render():
    st.markdown("## 🎯 Signal Engine")
    st.markdown("Enter your model's prediction and market conditions to get the optimal trade.")
    st.markdown(
        '<div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#b0b0b0;">'
        '💡 <b style="color:#e0e0e0;">What is the Signal Engine?</b> After the AI predicts UP or DOWN, '
        'we need to know exactly *how* to trade it. Depending on the AI\'s confidence and market fear (VIX), '
        'this page will tell you the exact buttons to press on your broker app.'
        '</div>',
        unsafe_allow_html=True
    )

    # Pull live values from data if available
    import utils.data_loader as data_loader
    nifty_df = data_loader.load_nifty_daily()
    vix_df = data_loader.load_vix_daily()
    nifty = int(nifty_df.iloc[-1]['Close']) if not nifty_df.empty else 22000
    vix_default = float(vix_df.iloc[-1]['VIX']) if not vix_df.empty else 18.0

    # Pre-fill from trained model if available
    default_conf = 65
    default_dir_idx = 0
    model = st.session_state.get("trained_model")
    if model is not None:
        scaler = st.session_state.get("scaler")
        feat_cols = st.session_state.get("feat_cols")
        featured_df = st.session_state.get("featured_df")
        if scaler and feat_cols and featured_df is not None and not featured_df.empty:
            try:
                latest = featured_df[feat_cols].iloc[[-1]].fillna(0)
                latest_s = pd.DataFrame(scaler.transform(latest), columns=feat_cols)
                prob = model.predict_proba(latest_s)[0][1]
                default_conf = int(abs(prob - 0.5) * 200)
                default_dir_idx = 0 if prob > 0.5 else 1
            except Exception:
                pass

    # ── Input Controls ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown("#### Model Output")
        confidence = st.slider("Model Confidence (%)", 40, 90, min(max(default_conf, 40), 90), 1,
                               help="Your ML model's directional confidence score")
        direction = st.selectbox("Predicted Direction", ["UP ↑", "DOWN ↓"], index=default_dir_idx)
        days_to_expiry = st.slider("Days to Expiry (DTE)", 1, 30, 7)

    with col2:
        st.markdown("#### Market Conditions")
        vix_input = st.slider("India VIX", 10.0, 40.0, vix_default, 0.5, key="se_vix")
        iv_rank = st.slider("IV Rank (%)", 0, 100, 45,
                            help="Current IV relative to past 52 weeks. >50 = sell, <50 = buy")
        market_trend = st.selectbox("Market Trend (50-day)", ["Uptrend", "Sideways", "Downtrend"])

    with col3:
        st.markdown("#### Position Sizing")
        capital = st.number_input("Available Capital (₹)", value=100000, step=5000)
        risk_pct = st.slider("Risk per Trade (%)", 1.0, 3.0, 2.0, 0.5)
        max_risk = capital * risk_pct / 100
        st.markdown(f"**Max Risk per Trade: ₹{max_risk:,.0f}**")
        lots = st.number_input("Lots", 1, 5, 1)

    st.markdown("---")

    # ── Signal Output ────────────────────────────────────────────────────────
    dir_clean = "UP" if "UP" in direction else "DOWN"
    signal = get_signal(confidence, vix_input, dir_clean)
    setup = get_trade_setup(signal, nifty, vix_input, lots, max_risk, days_to_expiry)

    sig_colors = {
        "BUY_CALL": ("#22c55e", "#0d2818", "#1a5c36"),
        "BUY_PUT":  ("#22c55e", "#0d2818", "#1a5c36"),
        "BULL_PUT_SPREAD": ("#f59e0b", "#1a1200", "#5c4200"),
        "BEAR_CALL_SPREAD": ("#f59e0b", "#1a1200", "#5c4200"),
        "NO_TRADE": ("#ef4444", "#1a0d0d", "#5c1a1a"),
    }
    text_color, bg_color, border_color = sig_colors.get(signal["action"], ("#888", CARD_BG, "#333"))

    st.markdown(
        f'<div style="background:{bg_color};border:2px solid {border_color};'
        f'border-radius:12px;padding:20px 24px;margin-bottom:24px;">'
        f'<div style="font-size:22px;font-weight:700;color:{text_color};margin-bottom:6px;">'
        f'{signal["label"]}</div>'
        f'<div style="font-size:14px;color:#b0b0b0;">{signal["reason"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if signal["action"] != "NO_TRADE":
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Trade Setup")
            for k, v in setup["details"].items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:8px 0;border-bottom:1px solid #1a1d2e;">'
                    f'<span style="color:#888;font-size:13px;">{k}</span>'
                    f'<span style="color:#e0e0e0;font-size:13px;font-weight:500;">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown("#### Scenario Analysis")
            df_scenarios = pd.DataFrame(setup["scenarios"])
            df_styled = df_scenarios.style.map(
                lambda v: "color: #22c55e" if isinstance(v, str) and "+" in v
                else ("color: #ef4444" if isinstance(v, str) and "-" in v else ""),
                subset=["P&L"]
            )
            st.dataframe(df_styled, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### Payoff Diagram")
            spots = setup["payoff_spots"]
            pnls = setup["payoff_pnls"]

            fig = go.Figure()
            pos_x = [s for s, p in zip(spots, pnls) if p >= 0]
            pos_y = [p for p in pnls if p >= 0]
            neg_x = [s for s, p in zip(spots, pnls) if p < 0]
            neg_y = [p for p in pnls if p < 0]

            if pos_x:
                fig.add_trace(go.Scatter(x=pos_x, y=pos_y, fill="tozeroy",
                                         fillcolor="rgba(34,197,94,0.15)",
                                         line=dict(color="#22c55e", width=2), name="Profit"))
            if neg_x:
                fig.add_trace(go.Scatter(x=neg_x, y=neg_y, fill="tozeroy",
                                         fillcolor="rgba(239,68,68,0.15)",
                                         line=dict(color="#ef4444", width=2), name="Loss"))
            fig.add_hline(y=0, line_color="#555", line_width=1)
            fig.add_vline(x=nifty, line_dash="dot", line_color="#888",
                          annotation_text="Current", annotation_position="top")

            fig.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#b0b0b0", height=320, showlegend=False,
                xaxis_title="Nifty at Expiry", yaxis_title="P&L (₹)",
                margin=dict(l=10, r=10, t=10, b=40)
            )
            fig.update_xaxes(gridcolor="#1e2130")
            fig.update_yaxes(gridcolor="#1e2130")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Risk/Reward Summary")
            rr1, rr2, rr3 = st.columns(3)
            rr1.metric("Max Profit", setup["max_profit"])
            rr2.metric("Max Loss", setup["max_loss"])
            rr3.metric("Breakeven", setup["breakeven"])

    # ── Strategy Decision Tree ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Strategy Selection Logic")

    col_tree, col_rules = st.columns([1, 1])

    with col_tree:
        tree_data = [
            ["Confidence > 70%", "VIX < 20", "→ Buy ATM CE/PE", "#22c55e"],
            ["Confidence 55–70%", "Any VIX", "→ Credit Spread", "#f59e0b"],
            ["Confidence 55–70%", "VIX > 20", "→ Credit Spread (preferred)", "#f59e0b"],
            ["Confidence < 55%", "Any VIX", "→ No Trade", "#ef4444"],
        ]
        for row in tree_data:
            active = (
                (row[0] == "Confidence > 70%" and confidence >= 70) or
                (row[0] == "Confidence 55–70%" and 55 <= confidence < 70) or
                (row[0] == "Confidence < 55%" and confidence < 55)
            )
            st.markdown(
                f'<div style="background:{"#1a2030" if active else CARD_BG};'
                f'border:{"2px" if active else "1px"} solid {row[3] if active else "#2a2d3e"};'
                f'border-radius:8px;padding:10px 14px;margin-bottom:8px;'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'<div><span style="color:{row[3]};font-size:12px;font-weight:600;">{row[0]}</span>'
                f'<span style="color:#666;font-size:12px;"> · {row[1]}</span></div>'
                f'<span style="color:#e0e0e0;font-size:13px;font-weight:500;">{row[2]}</span>'
                f'{"<span style=\"color:" + row[3] + ";font-size:11px;\">◀ ACTIVE</span>" if active else ""}'
                f'</div>',
                unsafe_allow_html=True
            )

    with col_rules:
        st.markdown("**VIX-Based Adjustments**")
        vix_rules = [
            ("VIX < 15", "Prefer buying options — cheap premiums", vix_input < 15, "#22c55e"),
            ("VIX 15–20", "Balanced mix — normal conditions", 15 <= vix_input < 20, "#3b82f6"),
            ("VIX 20–25", "Prefer credit spreads — expensive options", 20 <= vix_input < 25, "#f59e0b"),
            ("VIX > 25", "Only credit spreads — very expensive", vix_input >= 25, "#ef4444"),
        ]
        for label, desc, active, color in vix_rules:
            st.markdown(
                f'<div style="background:{"#1a2030" if active else CARD_BG};'
                f'border:{"2px" if active else "1px"} solid {color if active else "#2a2d3e"};'
                f'border-radius:8px;padding:9px 14px;margin-bottom:6px;">'
                f'<div style="color:{color};font-size:12px;font-weight:600;">{label}</div>'
                f'<div style="color:#b0b0b0;font-size:12px;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
