import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

DARK_BG = "#0e1117"
CARD_BG = "#1a1d2e"


def run_simulation(capital, win_rate, months, trades_per_month,
                   avg_win, avg_loss, n_sims=5000):
    """Monte Carlo simulation over N months."""
    np.random.seed(42)
    all_paths = []
    for _ in range(n_sims):
        cap = capital
        path = [cap]
        for _ in range(months):
            wins = np.random.binomial(trades_per_month, win_rate)
            losses = trades_per_month - wins
            monthly_pnl = wins * avg_win - losses * avg_loss
            cap += monthly_pnl
            path.append(cap)
        all_paths.append(path)
    return np.array(all_paths)


def render():
    st.markdown("## 📊 P&L Simulator")
    st.markdown("Monte Carlo simulation of your trading system's performance over time.")

    # ── Parameters ────────────────────────────────────────────────────────────
    with st.expander("⚙️ Simulation Parameters", expanded=True):
        p1, p2, p3 = st.columns(3)
        with p1:
            capital = st.number_input("Starting Capital (₹)", 50000, 500000, 100000, 10000)
            win_rate = st.slider("Win Rate (%)", 50, 75, 60) / 100
            months = st.slider("Simulation Months", 6, 36, 12)
        with p2:
            trades_pm = st.slider("Trades per Month", 4, 16, 8)
            avg_win = st.number_input("Avg Win per Trade (₹)", 500, 10000, 2250, 250)
            avg_loss = st.number_input("Avg Loss per Trade (₹)", 500, 5000, 2000, 250)
        with p3:
            n_sims = st.select_slider("Simulations", [1000, 5000, 10000], 5000)
            st.markdown("**Monthly Edge**")
            ev = win_rate * avg_win - (1 - win_rate) * avg_loss
            ev_total = ev * trades_pm
            st.metric("Expected Value / Trade", f"₹{ev:,.0f}")
            st.metric("Expected Monthly P&L", f"₹{ev_total:,.0f}")

    if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running simulations..."):
            paths = run_simulation(capital, win_rate, months, trades_pm, avg_win, avg_loss, n_sims)

        final_capitals = paths[:, -1]
        monthly_pnls = np.diff(paths, axis=1).flatten()

        # ── Summary Metrics ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Results")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Median Final Capital", f"₹{np.median(final_capitals):,.0f}",
                  f"{(np.median(final_capitals)/capital - 1)*100:.1f}%")
        m2.metric("P(Profit)", f"{(final_capitals > capital).mean()*100:.1f}%")
        m3.metric("P(Hit ₹5K/mo)", f"{(monthly_pnls > 5000).mean()*100:.1f}%")
        m4.metric("5th Percentile", f"₹{np.percentile(final_capitals, 5):,.0f}")
        m5.metric("95th Percentile", f"₹{np.percentile(final_capitals, 95):,.0f}")

        col1, col2 = st.columns(2)

        # ── Path fan chart ────────────────────────────────────────────────────
        with col1:
            st.markdown("#### Capital Paths (Fan Chart)")
            fig = go.Figure()
            x_axis = list(range(months + 1))

            p5 = np.percentile(paths, 5, axis=0)
            p25 = np.percentile(paths, 25, axis=0)
            p50 = np.percentile(paths, 50, axis=0)
            p75 = np.percentile(paths, 75, axis=0)
            p95 = np.percentile(paths, 95, axis=0)

            fig.add_trace(go.Scatter(
                x=x_axis + x_axis[::-1],
                y=list(p5 / 1000) + list(p95[::-1] / 1000),
                fill="toself", fillcolor="rgba(59,130,246,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="5–95%"
            ))
            fig.add_trace(go.Scatter(
                x=x_axis + x_axis[::-1],
                y=list(p25 / 1000) + list(p75[::-1] / 1000),
                fill="toself", fillcolor="rgba(59,130,246,0.18)",
                line=dict(color="rgba(0,0,0,0)"), name="25–75%"
            ))
            fig.add_trace(go.Scatter(x=x_axis, y=p50 / 1000,
                                     line=dict(color="#3b82f6", width=2.5),
                                     name="Median"))
            fig.add_hline(y=capital / 1000, line_dash="dot", line_color="#555",
                          annotation_text="Start")

            fig.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#b0b0b0", height=340,
                xaxis_title="Month", yaxis_title="Capital (₹K)",
                legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
                margin=dict(l=10, r=10, t=10, b=40)
            )
            fig.update_xaxes(gridcolor="#1e2130")
            fig.update_yaxes(gridcolor="#1e2130")
            st.plotly_chart(fig, use_container_width=True)

        # ── Final capital distribution ─────────────────────────────────────────
        with col2:
            st.markdown("#### Final Capital Distribution")
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=final_capitals / 1000, nbinsx=60,
                marker_color="#3b82f6", opacity=0.75
            ))
            fig2.add_vline(x=capital / 1000, line_dash="dash", line_color="#888",
                           annotation_text="Start")
            fig2.add_vline(x=np.median(final_capitals) / 1000, line_dash="dash",
                           line_color="#22c55e", annotation_text="Median")
            fig2.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#b0b0b0", height=340, showlegend=False,
                xaxis_title="Final Capital (₹K)", yaxis_title="Frequency",
                margin=dict(l=10, r=10, t=10, b=40)
            )
            fig2.update_xaxes(gridcolor="#1e2130")
            fig2.update_yaxes(gridcolor="#1e2130")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Monthly P&L stats ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Monthly P&L Statistics")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Probability Table**")
            targets = [0, 2000, 3000, 4000, 5000, 7500, 10000]
            probs = [(monthly_pnls > t).mean() * 100 for t in targets]
            df_prob = pd.DataFrame({
                "Monthly Target": [f"₹{t:,}" for t in targets],
                "Probability": [f"{p:.1f}%" for p in probs],
                "Chance": probs
            })
            fig_prob = go.Figure(go.Bar(
                x=[f"₹{t/1000:.0f}K" for t in targets],
                y=probs,
                marker_color=["#22c55e" if p > 60 else "#f59e0b" if p > 40 else "#ef4444"
                              for p in probs],
                text=[f"{p:.0f}%" for p in probs],
                textposition="outside"
            ))
            fig_prob.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#b0b0b0", height=240, showlegend=False,
                xaxis_title="Target", yaxis_title="Probability (%)",
                margin=dict(l=10, r=10, t=10, b=40)
            )
            fig_prob.update_yaxes(gridcolor="#1e2130", range=[0, 110])
            st.plotly_chart(fig_prob, use_container_width=True)

        with col4:
            st.markdown("**Win Rate Sensitivity**")
            wr_range = np.arange(0.50, 0.71, 0.01)
            ev_range = [(w * avg_win - (1 - w) * avg_loss) * trades_pm for w in wr_range]
            annual_range = [(1 + e / capital) ** 12 - 1 for e in ev_range]

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=wr_range * 100, y=[a * 100 for a in annual_range],
                line=dict(color="#f59e0b", width=2.5),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"
            ))
            fig_sens.add_vline(x=win_rate * 100, line_dash="dash", line_color="#3b82f6",
                               annotation_text="Your rate")
            fig_sens.add_hline(y=60, line_dash="dot", line_color="#22c55e",
                               annotation_text="60% annual target")
            fig_sens.update_layout(
                paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#b0b0b0", height=240, showlegend=False,
                xaxis_title="Win Rate (%)", yaxis_title="Annual Return (%)",
                margin=dict(l=10, r=10, t=10, b=40)
            )
            fig_sens.update_xaxes(gridcolor="#1e2130")
            fig_sens.update_yaxes(gridcolor="#1e2130")
            st.plotly_chart(fig_sens, use_container_width=True)

        # ── Scenario comparison table ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Scenario Comparison")
        scenarios = []
        for name, wr in [("Conservative", 0.55), ("Realistic", 0.60), ("Optimistic", 0.65)]:
            paths_s = run_simulation(capital, wr, months, trades_pm, avg_win, avg_loss, 2000)
            fc = paths_s[:, -1]
            monthly = np.diff(paths_s, axis=1).flatten()
            scenarios.append({
                "Scenario": name,
                "Win Rate": f"{wr*100:.0f}%",
                "Median Capital": f"₹{np.median(fc):,.0f}",
                "P(Positive Month)": f"{(monthly > 0).mean()*100:.1f}%",
                "P(Hit ₹5K/mo)": f"{(monthly > 5000).mean()*100:.1f}%",
                "Annual Return": f"{(np.median(fc)/capital - 1) / months * 12 * 100:.0f}%",
                "Max Drawdown (median)": f"₹{np.percentile(fc - capital, 10):,.0f}",
            })
        st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

    else:
        st.info("👆 Set your parameters above and click **Run Monte Carlo Simulation** to see results.")

        # Show a quick preview
        st.markdown("---")
        st.markdown("#### Quick Preview — Expected Monthly P&L")
        wr_vals = [55, 58, 60, 62, 65]
        pm_vals = [8] * 5
        aw, al = 2250, 2000
        expected = [(w/100 * aw - (1-w/100) * al) * 8 for w in wr_vals]

        fig_q = go.Figure(go.Bar(
            x=[f"{w}% win" for w in wr_vals],
            y=expected,
            marker_color=["#22c55e" if e > 5000 else "#f59e0b" if e > 0 else "#ef4444"
                          for e in expected],
            text=[f"₹{e:,.0f}" for e in expected],
            textposition="outside"
        ))
        fig_q.add_hline(y=5000, line_dash="dash", line_color="#888",
                        annotation_text="₹5K target")
        fig_q.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font_color="#b0b0b0", height=300, showlegend=False,
            yaxis_title="Expected Monthly P&L (₹)",
            margin=dict(l=10, r=10, t=10, b=40)
        )
        fig_q.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig_q, use_container_width=True)
