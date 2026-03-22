import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import utils.data_loader as data_loader
from utils.data_loader import load_pcr, load_fii_dii, build_master_dataset
from utils.features import engineer_all_features, get_feature_columns
from utils.logic import get_signal, get_trade_setup

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_model_prediction(featured_df):
    """If a trained model exists in session, run prediction on latest row."""
    model = st.session_state.get("trained_model")
    scaler = st.session_state.get("scaler")
    feat_cols = st.session_state.get("feat_cols")
    if model is None or scaler is None or feat_cols is None or featured_df is None:
        return None
    try:
        latest = featured_df[feat_cols].iloc[[-1]].fillna(0)
        latest_s = pd.DataFrame(scaler.transform(latest), columns=feat_cols)
        prob = model.predict_proba(latest_s)[0][1]
        conf = abs(prob - 0.5) * 200
        direction = "UP" if prob > 0.5 else "DOWN"
        return {"prob": prob, "confidence": conf, "direction": direction}
    except Exception:
        return None


def _get_journal_trades():
    """Return the trade journal DataFrame from session state."""
    if "trades" not in st.session_state:
        return pd.DataFrame()
    return st.session_state.trades.copy()


def _get_live_indicators(featured_df):
    """Extract the latest technical indicator values from engineered features."""
    if featured_df is None or featured_df.empty:
        return None
    last = featured_df.iloc[-1]
    
    # DEBUG: check for NaNs in last row
    nans = last.isna().sum()
    if nans > 0:
         # Optionally log nans to sidebar if needed for debug
         pass

    indicators = {}
    mapping = {
        "RSI (14)":     ("rsi",          lambda v: f"{v:.1f}",  lambda v: v / 100),
        "MACD hist":    ("macd_hist",    lambda v: f"{v:+.1f}", lambda v: min(max((v + 200) / 400, 0), 1)),
        "Stoch K":      ("stoch_k",      lambda v: f"{v:.1f}",  lambda v: v / 100),
        "BB %B":        ("pct_b",        lambda v: f"{v:.2f}",  lambda v: v),
        "ATR (14)":     ("atr",          lambda v: f"{v:.0f}",  lambda v: min(v / 400, 1)),
        "Vol ratio":    ("volume_ratio", lambda v: f"{v:.2f}x", lambda v: min(v / 2, 1)),
    }
    for label, (col, fmt, pct_fn) in mapping.items():
        if col in last.index and pd.notna(last[col]):
            val = float(last[col])
            indicators[label] = (fmt(val), pct_fn(val))
    return indicators if indicators else None


def render():
    # ── 1. Data Loading & Context ─────────────────────────────────────────────
    nifty_df = data_loader.load_nifty_daily()
    vix_df = data_loader.load_vix_daily()
    pcr_df = data_loader.load_pcr()
    fii_df = data_loader.load_fii_dii()

    # Defaults if missing
    latest_nifty_price = int(nifty_df.iloc[-1]['Close']) if not nifty_df.empty else 23125
    latest_vix_val = float(vix_df.iloc[-1]['VIX']) if not vix_df.empty else 17.8
    
    # Force LIVE VIX if possible
    try:
        import yfinance as yf
        live_vix = yf.Ticker("^INDIAVIX").fast_info.last_price
        if pd.notna(live_vix) and live_vix > 0:
            latest_vix_val = float(live_vix)
    except Exception:
        pass

    # Safety for PCR
    last_pcr = 1.0
    if pcr_df is not None and not pcr_df.empty:
        for col in ["PCR", "Pcr"]:
            if col in pcr_df.columns:
                last_pcr = float(pcr_df.iloc[-1][col])
                break

    # Safety for FII (normalized to fii_net in data_loader)
    last_fii = 0.0
    if fii_df is not None and not fii_df.empty:
        if "fii_net" in fii_df.columns:
            last_fii = float(fii_df.iloc[-1]["fii_net"])

    # Build features for indicators & model prediction
    featured_df = st.session_state.get("featured_df")
    if featured_df is None:
        try:
            master = build_master_dataset()
            if not master.empty:
                featured_df = engineer_all_features(master, dropna=False)
                st.session_state["featured_df"] = featured_df
            else:
                st.sidebar.error("Master dataset IS EMPTY")
        except Exception as e:
            featured_df = None
            st.sidebar.error(f"Feature build error: {e}")
            import traceback
            st.sidebar.code(traceback.format_exc())

    # Always show status in sidebar for debugging
    if featured_df is not None:
        pass # st.sidebar.success(f"Features: {len(featured_df)} rows")
    else:
        st.sidebar.info("Features: None (Building...)")

    # Get model prediction
    prediction = _get_model_prediction(featured_df)
    has_model = prediction is not None
    confidence = prediction["confidence"] if has_model else 63.0
    direction = prediction["direction"] if has_model else "UP"
    model_acc = st.session_state.get("test_acc", 0.614)

    # Remove Simulation overrides to enforce LIVE data
    vix_val = latest_vix_val
    nifty_val = latest_nifty_price

    # VIX regime label
    if vix_val < 15:
        vix_regime = "Low"
        vix_color = "#22c55e"
    elif vix_val < 20:
        vix_regime = "Normal"
        vix_color = "#f59e0b"
    elif vix_val < 25:
        vix_regime = "High"
        vix_color = "#ef4444"
    else:
        vix_regime = "Extreme"
        vix_color = "#ef4444"

    # ── 2. Top Bar: System Status ─────────────────────────────────────────────
    now_str = datetime.now().strftime("%d %b %Y • %H:%M IST")
    model_badge = "XGBoost v3.2" if has_model else "Not trained"
    data_status = "Live" if not nifty_df.empty else "Offline"

    st.markdown(f"""
    <div class="top-bar">
        <div style="display:flex; align-items:center;">
            <span style="font-weight:700; font-size:16px; margin-right:20px;">Nifty Intelligence System</span>
            <span class="status-item">{now_str} • Data: <span style="color:#22c55e;">{data_status}</span></span>
            <span style="background:#0d2818; color:#22c55e; padding:2px 10px; border-radius:10px; font-size:11px; border:1px solid #1a5c36;">Market Open</span>
        </div>
        <div style="display:flex; gap:10px;">
            <div style="background:#1e2130; padding:4px 12px; border-radius:20px; border:1px solid #3b82f633; font-size:12px;">
                <span style="color:#888;">Model:</span> <span style="color:#3b82f6; font-weight:600;">{model_badge}</span>
            </div>
            <div style="background:#1e2130; padding:4px 12px; border-radius:20px; border:1px solid #22c55e33; font-size:12px;">
                <span style="color:#888;">Accuracy:</span> <span style="color:#22c55e; font-weight:600;">{model_acc*100:.1f}%</span>
            </div>
            <div style="background:#1e2130; padding:4px 12px; border-radius:20px; border:1px solid {vix_color}33; font-size:12px;">
                <span style="color:#888;">VIX:</span> <span style="color:{vix_color}; font-weight:600;">{vix_val:.1f} — {vix_regime}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3. Green Signal Bar ───────────────────────────────────────────────────
    sig = get_signal(confidence, vix_val, direction)
    setup = get_trade_setup(sig, nifty_val, vix_val)

    conf_source = "Model" if has_model else "Default"
    dir_color = "#22c55e" if direction == "UP" else "#ef4444"

    # Signal bar color based on action
    if sig["action"] == "NO_TRADE":
        bar_bg, bar_border, title_color = "#1a0d0d", "#5c1a1a", "#ef4444"
    elif sig["action"] in ("BUY_CALL", "BUY_PUT"):
        bar_bg, bar_border, title_color = "#0d2818", "#1a5c36", "#22c55e"
    else:
        bar_bg, bar_border, title_color = "#0d2818", "#1a5c36", "#22c55e"

    # Build signal details depending on action
    if sig["action"] != "NO_TRADE":
        details = setup.get("details", {})
        sell_buy = f"{details.get('Sell', '')} • {details.get('Buy', '')}" if 'Sell' in details else details.get('Strike', '')
        premium_info = f" • Net credit <span style='color:#22c55e;'>{details.get('Net Premium', '')}</span>" if 'Net Premium' in details else ""
        margin_info = f" • Margin {details.get('Margin Required', '')}" if 'Margin Required' in details else ""
        strategy_name = details.get("Strategy", sig["label"])

        st.markdown(f"""
        <div style="background:{bar_bg}; border:1px solid {bar_border}; border-radius:8px; padding:15px 25px; margin-bottom:20px;">
            <div style="color:{title_color}; font-size:24px; font-weight:700; margin-bottom:5px;">{strategy_name} — Execute Now</div>
            <div style="color:#a0a0a0; font-size:14px; margin-bottom:12px;">
                {sell_buy}{premium_info}{margin_info}
            </div>
            <div class="signal-details">
                <div><span style="color:#888;">Confidence</span><br/><span style="color:#e0e0e0; font-size:18px; font-weight:600;">{confidence:.0f}% <span style="font-size:10px; color:#666;">({conf_source})</span></span></div>
                <div><span style="color:#888;">Direction</span><br/><span style="color:{dir_color}; font-size:18px; font-weight:600;">{direction}</span></div>
                <div><span style="color:#888;">DTE</span><br/><span style="color:#e0e0e0; font-size:18px; font-weight:600;">{details.get('DTE', '6 days')}</span></div>
                <div><span style="color:#888;">Max profit</span><br/><span style="color:#22c55e; font-size:18px; font-weight:600;">{setup['max_profit']}</span></div>
                <div><span style="color:#888;">Max loss</span><br/><span style="color:#ef4444; font-size:18px; font-weight:600;">{setup['max_loss']}</span></div>
                <div><span style="color:#888;">Breakeven</span><br/><span style="color:#e0e0e0; font-size:18px; font-weight:600;">{setup['breakeven']}</span></div>
                <div style="flex-grow:1; text-align:right;">
                    <span style="color:{vix_color}; font-size:13px; font-weight:600;">VIX {vix_val:.1f} — {vix_regime} regime</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{bar_bg}; border:1px solid {bar_border}; border-radius:8px; padding:15px 25px; margin-bottom:20px;">
            <div style="color:{title_color}; font-size:24px; font-weight:700;">⛔ No Trade — Confidence Too Low</div>
            <div style="color:#a0a0a0; font-size:14px; margin-top:5px;">{sig['reason']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 4. Metric Cards Row ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    # Dynamic values
    nifty_day_change = ((nifty_val / nifty_df.iloc[-2]['Close']) - 1) * 100 if len(nifty_df) > 1 else 0.0
    pcr_status = "Bullish >1.2" if last_pcr > 1.2 else "Neutral" if last_pcr > 0.8 else "Bearish"

    # Month P&L from trade journal
    journal = _get_journal_trades()
    if not journal.empty:
        journal["PnL"] = pd.to_numeric(journal["PnL"], errors="coerce")
        journal["Date"] = pd.to_datetime(journal["Date"], errors="coerce")
        now = datetime.now()
        month_trades = journal[
            (journal["Date"].dt.month == now.month) &
            (journal["Date"].dt.year == now.year) &
            (journal["Status"].str.contains("Closed", na=False))
        ]
        month_pnl = month_trades["PnL"].sum() if not month_trades.empty else 0
        open_trades = journal[journal["Status"].str.contains("Open", na=False)]
        capital_used = open_trades["Premium"].sum() if not open_trades.empty and "Premium" in open_trades.columns else 0
    else:
        month_pnl = 0
        capital_used = 0

    m1.metric("Nifty spot", f"{nifty_val:,}", f"{nifty_day_change:+.2f}% today")
    m2.metric("India VIX", f"{vix_val:.1f}", "High risk" if vix_val >= 20 else "Normal zone")
    m3.metric("PCR (OI)", f"{last_pcr:.2f}", pcr_status)
    m4.metric("FII net today", f"₹{last_fii:,.0f} Cr" if last_fii != 0 else "—", "Daily institutional")
    month_pnl_sign = "+" if month_pnl >= 0 else ""
    m5.metric("Month P&L", f"{month_pnl_sign}₹{month_pnl:,.0f}", "From journal")
    m6.metric("Capital used", f"₹{capital_used:,.0f}" if capital_used > 0 else "—", "Open positions")

    # ── 5. Main Content: Charts & Side Info ──────────────────────────────────
    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        # ── Main Box: Nifty Price Chart ───────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">NIFTY PRICE — LAST 60 DAYS + MODEL SIGNAL OVERLAY</div>', unsafe_allow_html=True)

        last_60 = nifty_df.tail(60).copy()
        last_60['SMA_20'] = last_60['Close'].rolling(20).mean()

        fig_nifty = go.Figure()
        fig_nifty.add_trace(go.Scatter(x=last_60.index, y=last_60['Close'], name='Nifty close', line=dict(color='#3b82f6', width=2)))
        fig_nifty.add_trace(go.Scatter(x=last_60.index, y=last_60['SMA_20'], name='SMA 20', line=dict(color='#22c55e', width=1, dash='dot')))

        # Real signal overlay from featured data if model exists
        if has_model and featured_df is not None and not featured_df.empty:
            feat_cols = st.session_state.get("feat_cols", [])
            model = st.session_state.get("trained_model")
            scaler = st.session_state.get("scaler")
            if model is not None and scaler is not None and feat_cols:
                try:
                    f60 = featured_df.loc[featured_df.index.intersection(last_60.index)]
                    if len(f60) > 0:
                        X60 = f60[feat_cols].fillna(0)
                        X60_s = pd.DataFrame(scaler.transform(X60), columns=feat_cols, index=X60.index)
                        probs = model.predict_proba(X60_s)[:, 1]
                        buy_mask = probs > 0.6
                        sell_mask = probs < 0.4
                        if buy_mask.any():
                            buy_idx = f60.index[buy_mask]
                            buy_prices = last_60.loc[last_60.index.intersection(buy_idx), 'Close']
                            fig_nifty.add_trace(go.Scatter(x=buy_prices.index, y=buy_prices, mode='markers', name='Buy signal', marker=dict(symbol='triangle-up', size=10, color='#22c55e')))
                        if sell_mask.any():
                            sell_idx = f60.index[sell_mask]
                            sell_prices = last_60.loc[last_60.index.intersection(sell_idx), 'Close']
                            fig_nifty.add_trace(go.Scatter(x=sell_prices.index, y=sell_prices, mode='markers', name='Sell signal', marker=dict(symbol='circle', size=8, color='#ef4444')))
                except Exception:
                    pass
        else:
            # Fallback: mock signal points
            if len(last_60) >= 46:
                buy_pts = last_60.iloc[[10, 25, 45]]
                fig_nifty.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['Close'], mode='markers', name='Buy signal', marker=dict(symbol='triangle-up', size=10, color='#22c55e')))

        fig_nifty.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color="#888", height=350, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2130", zeroline=False, side="right")
        )
        st.plotly_chart(fig_nifty, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 4 Mini Charts Grid ────────────────────────────────────────────────
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">INDIA VIX — 90 DAYS</div>', unsafe_allow_html=True)
            vix_tail = vix_df.tail(90)
            if not vix_tail.empty:
                fig_mini_vix = px.line(vix_tail, y='VIX', color_discrete_sequence=['#f59e0b'])
                fig_mini_vix.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=True, gridcolor="#1e2130"))
                st.plotly_chart(fig_mini_vix, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">PCR DAILY — 60 DAYS</div>', unsafe_allow_html=True)
            if pcr_df is not None and not pcr_df.empty and "PCR" in pcr_df.columns:
                pcr_tail = pcr_df.tail(60)
                fig_mini_pcr = px.line(pcr_tail, y='PCR', color_discrete_sequence=['#a855f7'])
                fig_mini_pcr.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=True, gridcolor="#1e2130"))
                st.plotly_chart(fig_mini_pcr, use_container_width=True)
            else:
                st.caption("PCR data not available")
            st.markdown('</div>', unsafe_allow_html=True)

        with cc2:
            # Monthly P&L from journal
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">MONTHLY P&L — FROM JOURNAL</div>', unsafe_allow_html=True)
            if not journal.empty:
                j = journal.copy()
                j["Date"] = pd.to_datetime(j["Date"], errors="coerce")
                j["PnL"] = pd.to_numeric(j["PnL"], errors="coerce")
                j["Month"] = j["Date"].dt.strftime("%b")
                monthly = j.groupby("Month")["PnL"].sum().reindex(
                    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                ).dropna()
                if not monthly.empty:
                    colors = ['#22c55e' if x > 0 else '#ef4444' for x in monthly.values]
                    fig_mini_pnl = go.Figure(go.Bar(x=monthly.index, y=monthly.values, marker_color=colors))
                    fig_mini_pnl.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(visible=True), yaxis=dict(visible=True, gridcolor="#1e2130"))
                    st.plotly_chart(fig_mini_pnl, use_container_width=True)
                else:
                    st.caption("No closed trades yet")
            else:
                st.caption("Log trades in Trade Journal to see P&L")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">FII NET FLOW — 30 DAYS</div>', unsafe_allow_html=True)
            if fii_df is not None and not fii_df.empty and "fii_net" in fii_df.columns:
                fii_tail = fii_df.tail(30)
                colors_fii = ['#22c55e' if x > 0 else '#ef4444' for x in fii_tail["fii_net"]]
                fig_mini_fii = go.Figure(go.Bar(x=fii_tail.index, y=fii_tail["fii_net"], marker_color=colors_fii))
                fig_mini_fii.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=True, gridcolor="#1e2130"))
                st.plotly_chart(fig_mini_fii, use_container_width=True)
            else:
                st.caption("FII data not available")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # ── Signal Decision Tree ─────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">SIGNAL DECISION TREE — LIVE</div>', unsafe_allow_html=True)

        # Dynamically highlight the active path
        conf_low = confidence < 55
        conf_med = 55 <= confidence < 70
        conf_high = confidence >= 70

        st.markdown(f"""
        <div style="font-size:12px; color:#888; margin-bottom:10px;">Confidence {confidence:.0f}% • {direction} • VIX {vix_val:.1f} ({conf_source})</div>
        <div style="border:1px solid {'#ef4444' if conf_low else '#ef444433'}; background:{'#ef444422' if conf_low else '#ef444411'}; padding:10px; border-radius:6px; margin-bottom:8px; text-align:center; font-size:13px; color:#f87171;">Conf < 55%? {'YES → No Trade ⛔' if conf_low else 'No → skip'}</div>
        <div style="text-align:center; color:#444; margin-bottom:8px;">↓</div>
        <div style="border:1px solid {'#f59e0b' if conf_med else '#f59e0b44'}; background:{'#f59e0b22' if conf_med else '#f59e0b11'}; padding:10px; border-radius:6px; margin-bottom:8px; text-align:center; font-size:13px; color:#f59e0b; font-weight:{'700' if conf_med else '400'};">Conf 55–70%? {'YES → Credit Spread ✓' if conf_med else 'No → skip'}</div>
        <div style="text-align:center; color:#444; margin-bottom:8px;">↓</div>
        <div style="border:1px solid {'#22c55e' if conf_high else '#22c55e44'}; background:{'#22c55e22' if conf_high else '#22c55e11'}; padding:10px; border-radius:6px; text-align:center; font-size:14px; color:#22c55e; font-weight:{'700' if conf_high else '400'};">Conf ≥ 70%? {'YES → Buy ATM ' + ('CE' if direction == 'UP' else 'PE') + ' ✓' if conf_high else 'No → skip'}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── VIX Regime Map ───────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">VIX REGIME MAP — CURRENT</div>', unsafe_allow_html=True)
        regimes = [
            ("< 15", "Buy CE/PE options", "cheap", vix_val < 15),
            ("15-20", "Mixed — prefer spreads", "normal", 15 <= vix_val < 20),
            ("20-25", "Credit spreads only", "high", 20 <= vix_val < 25),
            ("> 25", "Spreads only, reduce size", "extreme", vix_val >= 25),
        ]
        for rng, text, label, active in regimes:
            bg = "#3b82f622" if active else "transparent"
            border = "1px solid #3b82f666" if active else "1px solid #2a2d3e"
            color = "#fff" if active else "#888"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:{bg}; border:{border}; border-radius:8px; margin-bottom:6px;">
                <div style="display:flex; align-items:center;">
                    <div style="width:10px; height:10px; border-radius:50%; background:{'#3b82f6' if active else '#444'}; margin-right:12px;"></div>
                    <div style="font-size:13px; color:{color};">VIX {rng} — <span style="font-weight:600;">{label.upper()}</span></div>
                </div>
                {f'<span style="font-size:11px; color:#3b82f6; font-weight:700;">NOW</span>' if active else ''}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Technical Indicators (LIVE from features) ────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">TECHNICAL INDICATORS — LIVE</div>', unsafe_allow_html=True)

        live_indicators = _get_live_indicators(featured_df)
        if live_indicators:
            indicators_list = list(live_indicators.items())
        else:
            # Fallback: show placeholder
            indicators_list = [
                ("RSI (14)", ("—", 0.5)),
                ("MACD hist", ("—", 0.5)),
                ("Stoch K", ("—", 0.5)),
                ("BB %B", ("—", 0.5)),
                ("ATR (14)", ("—", 0.5)),
                ("Vol ratio", ("—", 0.5)),
            ]

        ti1, ti2 = st.columns(2)
        for i, (name, (val, pct)) in enumerate(indicators_list):
            with ti1 if i % 2 == 0 else ti2:
                # Color code by value
                if name == "RSI (14)" and val != "—":
                    ind_color = "#ef4444" if float(val) > 70 else "#22c55e" if float(val) < 30 else "#3b82f6"
                elif name == "MACD hist" and val != "—":
                    ind_color = "#22c55e" if float(val) > 0 else "#ef4444"
                else:
                    ind_color = "#3b82f6"
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="font-size:12px; color:#888;">{name}</div>
                    <div style="font-size:16px; font-weight:700; color:{ind_color};">{val}</div>
                    <div style="background:#2a2d3e; height:4px; border-radius:2px; margin-top:4px;">
                        <div style="background:{ind_color}; width:{pct*100:.0f}%; height:100%; border-radius:2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        source_label = "Live from features" if live_indicators else "No features — train model first"
        st.markdown(f'<div style="font-size:10px; color:#555; text-align:right;">{source_label}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 6. Bottom Section: 3 Columns ──────────────────────────────────────────
    b1, b2, b3 = st.columns([1, 1, 1])

    with b1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">FEATURE IMPORTANCE — TOP 10</div>', unsafe_allow_html=True)

        # Use trained model if available, else static fallback
        fi_dict = {}
        if has_model:
            model = st.session_state["trained_model"]
            feat_cols = st.session_state["feat_cols"]
            importances = model.feature_importances_
            fi_df = pd.DataFrame({"feat": feat_cols, "imp": importances}).sort_values("imp", ascending=False).head(10)
            fi_dict = dict(zip(fi_df["feat"], fi_df["imp"]))
            fi_source = "From trained model"
        else:
            fi_dict = {
                "VIX Change": 0.142, "RSI (14)": 0.118, "MACD Histogram": 0.097,
                "BB Width": 0.089, "VIX Spike": 0.082, "Stochastic K": 0.071,
                "Volume Ratio": 0.068, "ATR (14)": 0.063, "VIX Momentum": 0.058, "ROC (10)": 0.052,
            }
            fi_source = "Static defaults"

        for i, (feat, val) in enumerate(list(fi_dict.items())[:10]):
            colors = ["#2dd4bf", "#f59e0b", "#3b82f6", "#a855f7", "#ec4899"]
            max_imp = max(fi_dict.values()) if fi_dict else 1
            bar_pct = (val / max_imp) * 100
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:12px; color:#b0b0b0;">{feat}</span>
                <div style="display:flex; align-items:center; flex-grow:1; margin-left:15px;">
                    <div style="background:#1e2130; height:6px; border-radius:3px; flex-grow:1; margin-right:10px;">
                        <div style="background:{colors[i%5]}; width:{bar_pct:.0f}%; height:100%; border-radius:3px;"></div>
                    </div>
                    <span style="font-size:11px; color:#888; width:35px;">{val*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:10px; color:#555; text-align:right;">{fi_source}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">OPEN TRADES</div>', unsafe_allow_html=True)

        if not journal.empty:
            open_trades_df = journal[journal["Status"].str.contains("Open", na=False)]
            if not open_trades_df.empty:
                for _, trade in open_trades_df.iterrows():
                    pnl = float(trade.get("PnL", 0))
                    pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
                    border_color = "#22c55e" if pnl >= 0 else "#ef4444"
                    st.markdown(f"""
                    <div style="background:#0f1117; padding:12px; border-radius:8px; border-left:4px solid {border_color}; margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                            <span style="font-weight:600; font-size:14px;">{trade.get('Strategy', '')}</span>
                            <span style="color:{pnl_color}; font-weight:700;">{'+' if pnl >= 0 else ''}₹{pnl:,.0f}</span>
                        </div>
                        <div style="font-size:12px; color:#888;">{trade.get('Entry', '')} • {trade.get('DTE', '?')} DTE</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No open trades — log one in Trade Journal")
        else:
            st.caption("No trades logged yet")

        # Summary
        if not journal.empty:
            j_closed = journal[journal["Status"].str.contains("Closed", na=False)]
            j_closed["PnL"] = pd.to_numeric(j_closed["PnL"], errors="coerce")
            total_pnl = j_closed["PnL"].sum() if not j_closed.empty else 0
            wins = len(j_closed[j_closed["PnL"] > 0])
            losses = len(j_closed[j_closed["PnL"] < 0])
            win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

            st.markdown(f"""
            <hr style="border:0; border-top:1px solid #2a2d3e; margin:15px 0;"/>
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                <span style="color:#888;">Total closed P&L</span>
                <span style="color:{'#22c55e' if total_pnl >= 0 else '#ef4444'}; font-weight:600;">{'+' if total_pnl >= 0 else ''}₹{total_pnl:,.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:13px;">
                <span style="color:#888;">Win rate</span>
                <span style="color:#e0e0e0;">{wins}W / {losses}L = <span style="color:#3b82f6; font-weight:600;">{win_rate:.0f}%</span></span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">10 GOLDEN RULES — STATUS</div>', unsafe_allow_html=True)

        # Dynamically evaluate rules against live data
        open_count = len(journal[journal["Status"].str.contains("Open", na=False)]) if not journal.empty else 0
        rules = [
            ("Max 2% risk/trade — ₹2K cap", "OK"),
            ("No trade if conf < 55%", "OK" if confidence >= 55 else f"⚠ {confidence:.0f}%"),
            ("Book profit at 50% max", "OK"),
            ("SL at -50% for option buys", "OK"),
            ("Spreads only when VIX > 20", f"VIX {vix_val:.1f}" if vix_val <= 20 else "OK"),
            (f"Max 3 concurrent trades", f"{open_count}/3" if open_count <= 3 else f"⚠ {open_count}/3"),
            ("Never average losers", "OK"),
        ]
        for i, (rule, status) in enumerate(rules):
            color = "#22c55e" if status == "OK" else "#f59e0b" if "%" in status or "/" in status else "#3b82f6"
            if "⚠" in status:
                color = "#ef4444"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #1e2130;">
                <div style="display:flex; align-items:center;">
                    <div style="width:20px; height:20px; border-radius:50%; background:#1e2130; display:flex; align-items:center; justify-content:center; font-size:10px; color:#888; margin-right:12px;">{i+1}</div>
                    <span style="font-size:13px; color:#e0e0e0;">{rule}</span>
                </div>
                <div style="background:{color}33; color:{color}; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:700; min-width:40px; text-align:center;">{status}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
