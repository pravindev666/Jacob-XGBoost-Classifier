import streamlit as st
import plotly.graph_objects as go
import numpy as np

DARK_BG = "#0e1117"
CARD_BG = "#1a1d2e"


def payoff_bull_put(spot_range, short_strike, long_strike, net_premium, lots=1):
    mult = 100 * lots
    pnls = []
    for s in spot_range:
        if s >= short_strike:
            pnl = net_premium * mult
        elif s <= long_strike:
            pnl = (net_premium - (short_strike - long_strike)) * mult
        else:
            pnl = (net_premium - (short_strike - s)) * mult
        pnls.append(pnl)
    return pnls


def payoff_bear_call(spot_range, short_strike, long_strike, net_premium, lots=1):
    mult = 100 * lots
    pnls = []
    for s in spot_range:
        if s <= short_strike:
            pnl = net_premium * mult
        elif s >= long_strike:
            pnl = (net_premium - (long_strike - short_strike)) * mult
        else:
            pnl = (net_premium - (s - short_strike)) * mult
        pnls.append(pnl)
    return pnls


def payoff_long_call(spot_range, strike, premium, lots=1):
    mult = 100 * lots
    return [(max(0, s - strike) - premium) * mult for s in spot_range]


def payoff_long_put(spot_range, strike, premium, lots=1):
    mult = 100 * lots
    return [(max(0, strike - s) - premium) * mult for s in spot_range]


def payoff_chart(spots, pnls, title, nifty=22000):
    fig = go.Figure()
    pos_x = [s for s, p in zip(spots, pnls) if p >= 0]
    pos_y = [p for p in pnls if p >= 0]
    neg_x = [s for s, p in zip(spots, pnls) if p < 0]
    neg_y = [p for p in pnls if p < 0]
    if pos_x:
        fig.add_trace(go.Scatter(x=pos_x, y=pos_y, fill="tozeroy",
                                  fillcolor="rgba(34,197,94,0.15)",
                                  line=dict(color="#22c55e", width=2.5), name="Profit"))
    if neg_x:
        fig.add_trace(go.Scatter(x=neg_x, y=neg_y, fill="tozeroy",
                                  fillcolor="rgba(239,68,68,0.15)",
                                  line=dict(color="#ef4444", width=2.5), name="Loss"))
    fig.add_hline(y=0, line_color="#555", line_width=1)
    fig.add_vline(x=nifty, line_dash="dot", line_color="#888",
                  annotation_text="Current", annotation_position="top")
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font_color="#b0b0b0", height=240, showlegend=False,
        title=dict(text=title, font_size=13, font_color="#e0e0e0"),
        xaxis_title="Nifty at Expiry", yaxis_title="P&L (₹)",
        margin=dict(l=10, r=10, t=40, b=40)
    )
    fig.update_xaxes(gridcolor="#1e2130")
    fig.update_yaxes(gridcolor="#1e2130")
    return fig


def render():
    st.markdown("## 📚 Strategy Guide")
    st.markdown("Complete reference for all four strategies in your system.")
    st.markdown(
        '<div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#b0b0b0;">'
        '💡 <b style="color:#e0e0e0;">What is a strategy?</b> Think of strategies like tools in a toolbox. '
        'Sometimes you need a hammer (Buy Options), and sometimes you need a screwdriver (Credit Spreads). '
        'This page shows you how each tool works and exactly when to use it.'
        '</div>',
        unsafe_allow_html=True
    )

    nifty = st.session_state.get("global_nifty", 22000)
    spots = list(range(int(nifty * 0.88), int(nifty * 1.12), int(nifty * 0.002)))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🟢 Bull Put Spread",
        "🟠 Bear Call Spread",
        "🔵 Buy ATM Call",
        "🔵 Buy ATM Put",
        "⚖️ Golden Rules"
    ])

    # ── Bull Put Spread ───────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Bull Put Spread")
        st.markdown("""
**When to use:** Model predicts UP with 55–70% confidence. VIX any level (preferred >20).
You collect premium and profit if Nifty stays above the short strike.
        """)
        c1, c2, c3 = st.columns(3)
        short_put = c1.number_input("Short Put Strike", value=int(nifty * 0.986 // 100 * 100), step=100)
        long_put = c2.number_input("Long Put Strike", value=int(nifty * 0.977 // 100 * 100), step=100)
        net_prem_bps = c3.number_input("Net Premium Collected (₹/unit)", value=25, step=5)
        lots_bps = c3.number_input("Lots", 1, 5, 1, key="bps_lots")

        pnls_bps = payoff_bull_put(spots, short_put, long_put, net_prem_bps, lots_bps)
        st.plotly_chart(payoff_chart(spots, pnls_bps, "Bull Put Spread Payoff", nifty),
                        use_container_width=True)

        max_profit = net_prem_bps * 100 * lots_bps
        max_loss = ((short_put - long_put) - net_prem_bps) * 100 * lots_bps
        breakeven = short_put - net_prem_bps
        margin = (short_put - long_put) * 100 * lots_bps

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Max Profit", f"₹{max_profit:,}")
        mc2.metric("Max Loss", f"₹{max_loss:,}")
        mc3.metric("Breakeven", f"{breakeven:,}")
        mc4.metric("Approx Margin", f"₹{margin:,}")

        st.markdown("---")
        st.markdown("**Execution Steps (Zerodha):**")
        steps = [
            "Open Kite → F&O → NIFTY options chain",
            f"SELL 1 lot {short_put} PE (current expiry) → Note premium",
            f"BUY 1 lot {long_put} PE (same expiry) → Net credit = difference",
            "Book profit when P&L reaches 50% of max profit",
            "For credit spreads: hold to expiry if trade is going your way",
            "Max loss is defined — no stop loss needed if you sized correctly",
        ]
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}.** {step}")

    # ── Bear Call Spread ──────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Bear Call Spread")
        st.markdown("""
**When to use:** Model predicts DOWN with 55–70% confidence. Especially good when VIX > 20.
Sell a call, buy a higher call. Profit if Nifty stays below your short strike.
        """)
        c1, c2, c3 = st.columns(3)
        short_call = c1.number_input("Short Call Strike", value=int(nifty * 1.01 // 100 * 100), step=100)
        long_call = c2.number_input("Long Call Strike", value=int(nifty * 1.018 // 100 * 100), step=100)
        net_prem_bcs = c3.number_input("Net Premium Collected (₹/unit)", value=22, step=5)
        lots_bcs = c3.number_input("Lots", 1, 5, 1, key="bcs_lots")

        pnls_bcs = payoff_bear_call(spots, short_call, long_call, net_prem_bcs, lots_bcs)
        st.plotly_chart(payoff_chart(spots, pnls_bcs, "Bear Call Spread Payoff", nifty),
                        use_container_width=True)

        max_profit_bcs = net_prem_bcs * 100 * lots_bcs
        max_loss_bcs = ((long_call - short_call) - net_prem_bcs) * 100 * lots_bcs
        breakeven_bcs = short_call + net_prem_bcs

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Max Profit", f"₹{max_profit_bcs:,}")
        mc2.metric("Max Loss", f"₹{max_loss_bcs:,}")
        mc3.metric("Breakeven", f"{breakeven_bcs:,}")
        mc4.metric("Win Condition", f"Nifty < {short_call:,} at expiry")

    # ── Buy ATM Call ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Buy ATM Call (CE)")
        st.markdown("""
**When to use:** Model predicts UP with ≥70% confidence AND VIX < 20 (cheap premiums).
Risk is limited to premium paid. Profit potential is theoretically unlimited.
        """)
        c1, c2, c3 = st.columns(3)
        call_strike = c1.number_input("Strike (ATM)", value=int(nifty // 100 * 100), step=100)
        call_prem = c2.number_input("Premium paid (₹/unit)", value=180, step=10)
        call_qty = c3.number_input("Quantity (units)", 10, 75, 15)

        pnls_call = payoff_long_call(spots, call_strike, call_prem, 1)
        pnls_call_scaled = [p * call_qty / 100 for p in pnls_call]
        st.plotly_chart(payoff_chart(spots, pnls_call_scaled, "Long ATM Call Payoff", nifty),
                        use_container_width=True)

        max_loss_c = call_prem * call_qty
        breakeven_c = call_strike + call_prem
        target_exit = call_prem * 1.5

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Max Risk", f"₹{max_loss_c:,}")
        mc2.metric("Breakeven at Expiry", f"{breakeven_c:,}")
        mc3.metric("Book Profit When", f"Premium > ₹{target_exit:.0f}")
        mc4.metric("Stop Loss When", f"Premium < ₹{call_prem*0.5:.0f}")

        st.info(f"💡 VIX rule: Only buy options when VIX < 20. "
                f"Current VIX: {st.session_state.get('global_vix', 18):.1f} "
                f"{'✅ OK to buy' if st.session_state.get('global_vix', 18) < 20 else '⚠️ VIX too high — use spread instead'}")

    # ── Buy ATM Put ───────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Buy ATM Put (PE)")
        st.markdown("""
**When to use:** Model predicts DOWN with ≥70% confidence AND VIX < 20.
Same logic as call buying but for bearish scenarios.
        """)
        c1, c2, c3 = st.columns(3)
        put_strike = c1.number_input("Strike (ATM)", value=int(nifty // 100 * 100), step=100, key="ps")
        put_prem = c2.number_input("Premium paid (₹/unit)", value=175, step=10)
        put_qty = c3.number_input("Quantity (units)", 10, 75, 15, key="pq")

        pnls_put = payoff_long_put(spots, put_strike, put_prem, 1)
        pnls_put_scaled = [p * put_qty / 100 for p in pnls_put]
        st.plotly_chart(payoff_chart(spots, pnls_put_scaled, "Long ATM Put Payoff", nifty),
                        use_container_width=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Max Risk", f"₹{put_prem * put_qty:,}")
        mc2.metric("Breakeven at Expiry", f"{put_strike - put_prem:,}")
        mc3.metric("Book Profit When", f"Premium > ₹{put_prem * 1.5:.0f}")
        mc4.metric("Stop Loss When", f"Premium < ₹{put_prem * 0.5:.0f}")

    # ── Golden Rules ──────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### The 10 Golden Rules")
        st.markdown("These are **non-negotiable**. Breaking any rule is the #1 reason traders blow up accounts.")

        rules = [
            ("1", "Max 2% risk per trade", "#ef4444",
             "Never risk more than ₹2,000 on a single trade. This is your capital preservation rule. "
             "Even with a 5-trade losing streak, you lose only 10% of capital."),
            ("2", "No trade if confidence < 55%", "#f59e0b",
             "Your model needs an edge. Below 55% confidence, you're worse than a coin flip. "
             "The Renaissance principle: only trade when you have a genuine edge."),
            ("3", "Book profit at 50% target", "#22c55e",
             "For option buyers: sell when premium doubles. For credit spreads: close at 50% of max profit. "
             "A bird in hand is worth two in the bush in options trading."),
            ("4", "Cut loss at -50% for option buys", "#ef4444",
             "If you bought a CE/PE for ₹180, exit when it hits ₹90. Don't let losers run. "
             "Credit spreads: hold to expiry (risk is already defined)."),
            ("5", "Use credit spreads when VIX > 20", "#f59e0b",
             "High VIX = expensive premiums = sell, don't buy. Collect rich premium with defined risk."),
            ("6", "Buy options when VIX < 15", "#22c55e",
             "Low VIX = cheap premiums = buy. Options are underpriced relative to likely moves."),
            ("7", "Max 3 concurrent trades", "#3b82f6",
             "Concentration risk. With only 3 trades, you can monitor each carefully and react quickly."),
            ("8", "Roll positions if needed", "#a855f7",
             "If a credit spread is going against you with >7 DTE, roll to next expiry to collect more premium."),
            ("9", "Review monthly performance", "#14b8a6",
             "Track win rate, P&L, drawdown. If win rate drops below 50% for 2 months, pause and retrain model."),
            ("10", "Never average losers", "#ef4444",
             "Adding to a losing position is how accounts blow up. Cut and move on."),
        ]

        for num, title, color, explanation in rules:
            with st.expander(f"Rule {num}: {title}", expanded=False):
                st.markdown(
                    f'<div style="border-left:4px solid {color};padding:12px 20px;'
                    f'border-radius:4px;background:{CARD_BG};">'
                    f'<p style="color:#d0d0d0;margin:0;line-height:1.7;">{explanation}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("### The Renaissance Principle")
        st.markdown("""
> *"We're right 50.75% of the time, but we're 100% right 50.75% of the time. You can make billions that way."*
> — Robert Mercer, Co-CEO Renaissance Technologies

**You don't need 90% accuracy.** You need:
- A small consistent edge (55–60% accuracy)
- Proper position sizing (2% risk per trade)
- Many iterations (8 trades/month × 12 months = 96 trades/year)
- Strict rule-following (no emotional overrides)

**Edge × Position Sizing × Iterations = Profit**
        """)

        ev_data = []
        for wr in np.arange(0.50, 0.71, 0.02):
            ev = (wr * 2250 - (1 - wr) * 2000) * 8
            ev_data.append({"Win Rate": f"{wr*100:.0f}%", "Monthly EV": ev,
                            "Annual EV": ev * 12})

        import pandas as pd
        df_ev = pd.DataFrame(ev_data)
        fig_ev = go.Figure(go.Bar(
            x=df_ev["Win Rate"], y=df_ev["Monthly EV"],
            marker_color=["#22c55e" if e > 5000 else "#f59e0b" if e > 0 else "#ef4444"
                          for e in df_ev["Monthly EV"]],
            text=[f"₹{e:,.0f}" for e in df_ev["Monthly EV"]],
            textposition="outside"
        ))
        fig_ev.add_hline(y=5000, line_dash="dash", line_color="#888",
                         annotation_text="₹5K/month target")
        fig_ev.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font_color="#b0b0b0", height=280, showlegend=False,
            yaxis_title="Expected Monthly P&L (₹)",
            title=dict(text="Why 60% Win Rate Is Enough", font_size=13, font_color="#e0e0e0"),
            margin=dict(l=10, r=10, t=40, b=40)
        )
        fig_ev.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig_ev, use_container_width=True)
