"""
pages/data_explorer.py
-----------------------
Shows your real CSV data, data quality checks, and feature preview.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import (
    load_nifty_daily, load_vix_daily, load_pcr, load_fii_dii,
    load_bank_nifty, load_sp500, load_vix_term, get_data_status,
    build_master_dataset
)
from utils.features import engineer_all_features, get_feature_columns, get_feature_groups

DARK_BG = "#0e1117"
CARD_BG = "#1a1d2e"


def render():
    st.markdown("## 📂 Data Explorer")
    st.markdown("Inspect your real CSV datasets, check data quality, and preview engineered features.")

    # ── Data status ──────────────────────────────────────────────────────────
    st.markdown("### Dataset Status")
    status = get_data_status()
    cols = st.columns(5)
    for idx, (fname, info) in enumerate(status.items()):
        with cols[idx % 5]:
            icon = "✅" if info["present"] else "❌"
            color = "#22c55e" if info["present"] else "#ef4444"
            st.markdown(
                f'<div style="background:{CARD_BG};border:1px solid '
                f'{"#1a5c36" if info["present"] else "#5c1a1a"};'
                f'border-radius:8px;padding:10px 12px;margin-bottom:8px;">'
                f'<div style="font-size:16px;">{icon}</div>'
                f'<div style="font-size:12px;color:{color};font-weight:600;">'
                f'{info["label"].split("⭐")[0].strip()}</div>'
                f'<div style="font-size:11px;color:#666;">{info["size"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    missing = [f for f, info in status.items() if not info["present"]]
    if missing:
        st.info(f"📁 Place your CSV files in the **`data/`** folder inside your project. "
                f"Missing: {len(missing)} files. The app works with synthetic data in the meantime.")
    else:
        st.success("✅ All datasets found! Using your real data.")

    st.markdown("---")

    # ── Tabs for each dataset ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Nifty Daily", "🌡️ India VIX", "📊 PCR + FII",
        "🏦 Bank Nifty", "🌍 SP500", "🔬 Master Dataset"
    ])

    with tab1:
        st.markdown("### Nifty Daily OHLCV")
        nifty = load_nifty_daily()
        _show_dataset_overview(nifty, "Nifty Daily", price_col="Close")

        st.markdown("#### Price Chart")
        fig = go.Figure()
        sample = nifty.tail(500)
        fig.add_trace(go.Candlestick(
            x=sample.index,
            open=sample.get("Open", sample["Close"]),
            high=sample.get("High", sample["Close"]),
            low=sample.get("Low", sample["Close"]),
            close=sample["Close"],
            name="Nifty",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444"
        ))
        # 20-day SMA overlay
        sample_sma = sample["Close"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=sample.index, y=sample_sma,
            line=dict(color="#3b82f6", width=1.5),
            name="SMA 20"
        ))
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font_color="#b0b0b0", height=380,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=40)
        )
        fig.update_xaxes(gridcolor="#1e2130")
        fig.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig, use_container_width=True)

        # Return distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Daily Return Distribution")
            returns = nifty["Close"].pct_change().dropna() * 100
            fig_ret = go.Figure(go.Histogram(
                x=returns, nbinsx=100,
                marker_color="#3b82f6", opacity=0.8
            ))
            fig_ret.add_vline(x=0, line_color="#888")
            fig_ret.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#b0b0b0", height=240,
                xaxis_title="Daily Return (%)",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=40)
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        with col2:
            st.markdown("#### Key Stats")
            ret = nifty["Close"].pct_change().dropna()
            stats = {
                "Total days":   f"{len(nifty):,}",
                "Date range":   f"{nifty.index[0].date()} → {nifty.index[-1].date()}",
                "Start price":  f"₹{nifty['Close'].iloc[0]:,.0f}",
                "Latest price": f"₹{nifty['Close'].iloc[-1]:,.0f}",
                "Total return": f"{(nifty['Close'].iloc[-1]/nifty['Close'].iloc[0]-1)*100:.0f}%",
                "Ann. volatility": f"{ret.std()*np.sqrt(252)*100:.1f}%",
                "Best day":     f"+{ret.max()*100:.2f}%",
                "Worst day":    f"{ret.min()*100:.2f}%",
                "Up days":      f"{(ret > 0).mean()*100:.1f}% of days",
            }
            for k, v in stats.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 0;border-bottom:1px solid #1a1d2e;">'
                    f'<span style="color:#888;font-size:13px;">{k}</span>'
                    f'<span style="color:#e0e0e0;font-size:13px;font-weight:500;">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with tab2:
        st.markdown("### India VIX Daily")
        vix = load_vix_daily()
        _show_dataset_overview(vix, "India VIX", price_col="VIX")

        # VIX time series
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=vix.index, y=vix["VIX"],
            line=dict(color="#f59e0b", width=1.5),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
            name="VIX"
        ))
        for lvl, color, label in [(15, "#22c55e", "VIX 15 (buy zone)"),
                                   (20, "#f59e0b", "VIX 20 (spread zone)"),
                                   (25, "#ef4444", "VIX 25 (danger)")]:
            fig_vix.add_hline(y=lvl, line_dash="dash", line_color=color,
                               annotation_text=label, annotation_position="right")
        fig_vix.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font_color="#b0b0b0", height=320,
            yaxis_title="India VIX", showlegend=False,
            margin=dict(l=10, r=10, t=10, b=40)
        )
        fig_vix.update_xaxes(gridcolor="#1e2130")
        fig_vix.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig_vix, use_container_width=True)

        # VIX regime distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### VIX Regime Distribution")
            bins = [0, 15, 20, 25, 100]
            labels = ["<15 (Low)", "15–20 (Normal)", "20–25 (High)", ">25 (Extreme)"]
            colors = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"]
            counts = pd.cut(vix["VIX"], bins=bins, labels=labels).value_counts()
            fig_reg = go.Figure(go.Bar(
                x=labels, y=counts.values,
                marker_color=colors,
                text=[f"{v/counts.sum()*100:.0f}%" for v in counts.values],
                textposition="outside"
            ))
            fig_reg.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#b0b0b0", height=260,
                yaxis_title="Days", showlegend=False,
                margin=dict(l=10, r=10, t=10, b=40)
            )
            st.plotly_chart(fig_reg, use_container_width=True)

        with col2:
            st.markdown("#### VIX Stats")
            st.metric("Current VIX", f"{vix['VIX'].iloc[-1]:.1f}")
            st.metric("1Y Avg", f"{vix['VIX'].tail(252).mean():.1f}")
            st.metric("All-time High", f"{vix['VIX'].max():.1f}")
            st.metric("All-time Low", f"{vix['VIX'].min():.1f}")
            st.metric("% days VIX < 15", f"{(vix['VIX'] < 15).mean()*100:.0f}%")
            st.metric("% days VIX > 20", f"{(vix['VIX'] > 20).mean()*100:.0f}%")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Put-Call Ratio (PCR)")
            pcr = load_pcr()
            if pcr is not None:
                _show_dataset_overview(pcr, "PCR", price_col="PCR")
                fig_pcr = go.Figure(go.Scatter(
                    x=pcr.index, y=pcr["PCR"],
                    line=dict(color="#a855f7", width=1.5), name="PCR"
                ))
                fig_pcr.add_hline(y=1.0, line_dash="dash", line_color="#888",
                                   annotation_text="Neutral 1.0")
                fig_pcr.add_hline(y=1.2, line_dash="dot", line_color="#22c55e",
                                   annotation_text="Bullish >1.2")
                fig_pcr.add_hline(y=0.8, line_dash="dot", line_color="#ef4444",
                                   annotation_text="Bearish <0.8")
                fig_pcr.update_layout(
                    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                    font_color="#b0b0b0", height=280,
                    yaxis_title="PCR", showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=40)
                )
                st.plotly_chart(fig_pcr, use_container_width=True)
            else:
                st.info("pcr_daily.csv not in data/ folder yet — add it to enable PCR features.")

        with col2:
            st.markdown("### FII/DII Flow")
            fii = load_fii_dii()
            if fii is not None:
                st.dataframe(fii.tail(30), use_container_width=True)
                # Plot first numeric column
                num_cols = fii.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    fig_fii = go.Figure(go.Bar(
                        x=fii.tail(60).index,
                        y=fii.tail(60)[num_cols[0]],
                        marker_color=["#22c55e" if v > 0 else "#ef4444"
                                      for v in fii.tail(60)[num_cols[0]]]
                    ))
                    fig_fii.update_layout(
                        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                        font_color="#b0b0b0", height=260,
                        title=num_cols[0], showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=40)
                    )
                    st.plotly_chart(fig_fii, use_container_width=True)
            else:
                st.info("fii_dii_daily.csv not in data/ folder yet.")

    with tab4:
        st.markdown("### Bank Nifty Daily")
        bnf = load_bank_nifty()
        if bnf is not None:
            _show_dataset_overview(bnf, "Bank Nifty", price_col="Close")
        else:
            st.info("bank_nifty_daily.csv not in data/ folder yet.")

    with tab5:
        st.markdown("### S&P 500 (Global Context)")
        sp = load_sp500()
        if sp is not None:
            _show_dataset_overview(sp, "S&P 500", price_col="Close")
        else:
            st.info("sp500_daily.csv not in data/ folder yet.")

    with tab6:
        st.markdown("### Master Dataset — All Features Combined")
        st.markdown("This is what gets fed into the ML model after merging all your CSVs.")

        if st.button("🔧 Build Master Dataset + Engineer Features", type="primary"):
            with st.spinner("Loading CSVs, merging, engineering 55+ features..."):
                master = build_master_dataset()
                featured = engineer_all_features(master)

            st.success(f"✅ Master dataset: **{len(featured):,} rows × {len(featured.columns)} columns**")

            # Show feature counts
            groups = get_feature_groups()
            cols = st.columns(4)
            for idx, (grp, feats) in enumerate(groups.items()):
                with cols[idx % 4]:
                    present = sum(1 for f in feats if f in featured.columns)
                    total = len(feats)
                    color = "#22c55e" if present == total else "#f59e0b"
                    st.markdown(
                        f'<div style="background:{CARD_BG};border-radius:8px;'
                        f'padding:10px 14px;margin-bottom:8px;">'
                        f'<div style="color:{color};font-size:13px;font-weight:600;">'
                        f'{grp}</div>'
                        f'<div style="color:#b0b0b0;font-size:12px;">'
                        f'{present}/{total} features active</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("#### Sample Data (last 20 rows)")
            feat_cols = get_feature_columns(featured)
            display_cols = ["Close"] + feat_cols[:15]  # show first 15 features
            st.dataframe(featured[display_cols].tail(20).round(4),
                         use_container_width=True)

            # Target distribution
            target_counts = featured["target"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total samples", f"{len(featured):,}")
            c2.metric("Up days (target=1)", f"{target_counts.get(1,0):,} "
                      f"({target_counts.get(1,0)/len(featured)*100:.1f}%)")
            c3.metric("Down days (target=0)", f"{target_counts.get(0,0):,} "
                      f"({target_counts.get(0,0)/len(featured)*100:.1f}%)")

            st.session_state["featured_df"] = featured
            st.success("✅ Dataset saved to session. Go to **Model Builder** to train!")
        else:
            st.info("👆 Click the button to merge all your CSVs and engineer features.")


def _show_dataset_overview(df, name: str, price_col: str):
    if df is None:
        st.warning(f"{name}: CSV file not found in data/ folder.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns)}")
    c3.metric("Date range", f"{df.index[0].date() if len(df) > 0 else 'N/A'}")
    c4.metric("Latest", f"{df.index[-1].date() if len(df) > 0 else 'N/A'}")
    with st.expander("Show raw data (last 20 rows)"):
        st.dataframe(df.tail(20).round(4), use_container_width=True)
