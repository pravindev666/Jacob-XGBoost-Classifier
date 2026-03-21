import numpy as np


def get_signal(confidence: float, vix: float, direction: str) -> dict:
    """
    Core signal engine: given model confidence, VIX, and direction,
    return the optimal trade action with reason.
    """
    if confidence < 55:
        return {
            "action": "NO_TRADE",
            "label": "⛔ No Trade",
            "reason": f"Confidence {confidence:.0f}% is below 55% threshold. "
                      "Wait for a stronger model signal. Patience is edge.",
        }

    if confidence >= 70:
        if direction == "UP":
            buying_ok = vix < 20
            return {
                "action": "BUY_CALL",
                "label": "🟢 Buy ATM Call (CE)" + (" — Ideal (low VIX)" if vix < 15 else ""),
                "reason": (
                    f"High confidence ({confidence:.0f}%) + UP signal. "
                    + (f"VIX {vix:.1f} < 20 — premiums are cheap, good to buy options."
                       if buying_ok else
                       f"VIX {vix:.1f} > 20 — premiums are expensive, consider Bull Put Spread instead.")
                ),
            }
        else:
            return {
                "action": "BUY_PUT",
                "label": "🟢 Buy ATM Put (PE)" + (" — Ideal (low VIX)" if vix < 15 else ""),
                "reason": (
                    f"High confidence ({confidence:.0f}%) + DOWN signal. "
                    + (f"VIX {vix:.1f} < 20 — buy options." if vix < 20
                       else f"VIX {vix:.1f} > 20 — consider Bear Call Spread.")
                ),
            }

    # Medium confidence (55–70%)
    if direction == "UP":
        return {
            "action": "BULL_PUT_SPREAD",
            "label": "🟡 Bull Put Spread" + (" — VIX Regime Preferred" if vix > 20 else ""),
            "reason": (
                f"Medium confidence ({confidence:.0f}%) + UP signal → Credit spread is safer. "
                f"VIX {vix:.1f}: "
                + ("High VIX = rich premiums, ideal for selling." if vix > 20
                   else "Normal conditions. Collect premium with defined risk.")
            ),
        }
    else:
        return {
            "action": "BEAR_CALL_SPREAD",
            "label": "🟡 Bear Call Spread" + (" — VIX Regime Preferred" if vix > 20 else ""),
            "reason": (
                f"Medium confidence ({confidence:.0f}%) + DOWN signal → Credit spread is safer. "
                f"VIX {vix:.1f}: "
                + ("High VIX = expensive calls, ideal to sell." if vix > 20
                   else "Normal conditions. Collect premium with defined risk.")
            ),
        }


def get_trade_setup(signal: dict, nifty: float, vix: float,
                    lots: int = 1, max_risk: float = 2000, dte: int = 7) -> dict:
    """Return detailed trade setup parameters based on the signal."""
    action = signal["action"]
    mult = 100 * lots

    if action == "BULL_PUT_SPREAD":
        short_strike = int(nifty * 0.986 // 100 * 100)
        long_strike = int(nifty * 0.977 // 100 * 100)
        net_prem = 25
        max_profit = net_prem * mult
        max_loss = (short_strike - long_strike - net_prem) * mult
        breakeven = short_strike - net_prem
        margin = (short_strike - long_strike) * mult

        spots = list(range(int(nifty * 0.93), int(nifty * 1.05), int(nifty * 0.002)))
        pnls = []
        for s in spots:
            if s >= short_strike:
                pnl = net_prem * mult
            elif s <= long_strike:
                pnl = (net_prem - (short_strike - long_strike)) * mult
            else:
                pnl = (net_prem - (short_strike - s)) * mult
            pnls.append(pnl)

        return {
            "details": {
                "Strategy": "Bull Put Spread",
                "Sell": f"{short_strike} PE @ ~₹45",
                "Buy": f"{long_strike} PE @ ~₹20",
                "Net Premium": f"₹{net_prem * mult:,}",
                "Lots": lots,
                "Margin Required": f"~₹{margin:,}",
                "Max Profit": f"₹{max_profit:,}",
                "Max Loss": f"₹{max_loss:,}",
                "DTE": f"{dte} days",
                "Book Profit At": f"50% = ₹{max_profit//2:,}",
            },
            "scenarios": [
                {"Nifty at Expiry": f"{int(nifty*1.02):,}", "Move": "+2%",
                 "P&L": f"+₹{max_profit:,}", "Outcome": "Full profit ✅"},
                {"Nifty at Expiry": f"{short_strike:,}", "Move": "At short",
                 "P&L": f"+₹{max_profit:,}", "Outcome": "Full profit ✅"},
                {"Nifty at Expiry": f"{breakeven:,}", "Move": "At breakeven",
                 "P&L": "₹0", "Outcome": "Break even"},
                {"Nifty at Expiry": f"{int((short_strike+long_strike)/2):,}", "Move": "Middle",
                 "P&L": f"-₹{max_loss//2:,}", "Outcome": "Partial loss"},
                {"Nifty at Expiry": f"{long_strike:,}", "Move": "At long",
                 "P&L": f"-₹{max_loss:,}", "Outcome": "Max loss ❌"},
            ],
            "payoff_spots": spots,
            "payoff_pnls": pnls,
            "max_profit": f"₹{max_profit:,}",
            "max_loss": f"-₹{max_loss:,}",
            "breakeven": f"{breakeven:,}",
        }

    elif action == "BEAR_CALL_SPREAD":
        short_strike = int(nifty * 1.009 // 100 * 100)
        long_strike = int(nifty * 1.018 // 100 * 100)
        net_prem = 22
        max_profit = net_prem * mult
        max_loss = (long_strike - short_strike - net_prem) * mult
        breakeven = short_strike + net_prem

        spots = list(range(int(nifty * 0.96), int(nifty * 1.07), int(nifty * 0.002)))
        pnls = []
        for s in spots:
            if s <= short_strike:
                pnl = net_prem * mult
            elif s >= long_strike:
                pnl = (net_prem - (long_strike - short_strike)) * mult
            else:
                pnl = (net_prem - (s - short_strike)) * mult
            pnls.append(pnl)

        return {
            "details": {
                "Strategy": "Bear Call Spread",
                "Sell": f"{short_strike} CE @ ~₹42",
                "Buy": f"{long_strike} CE @ ~₹20",
                "Net Premium": f"₹{net_prem * mult:,}",
                "Lots": lots,
                "Margin Required": f"~₹{(long_strike-short_strike)*mult:,}",
                "Max Profit": f"₹{max_profit:,}",
                "Max Loss": f"₹{max_loss:,}",
                "DTE": f"{dte} days",
                "Book Profit At": f"50% = ₹{max_profit//2:,}",
            },
            "scenarios": [
                {"Nifty at Expiry": f"{int(nifty*0.98):,}", "Move": "-2%",
                 "P&L": f"+₹{max_profit:,}", "Outcome": "Full profit ✅"},
                {"Nifty at Expiry": f"{short_strike:,}", "Move": "At short",
                 "P&L": f"+₹{max_profit:,}", "Outcome": "Full profit ✅"},
                {"Nifty at Expiry": f"{breakeven:,}", "Move": "At breakeven",
                 "P&L": "₹0", "Outcome": "Break even"},
                {"Nifty at Expiry": f"{long_strike:,}", "Move": "At long",
                 "P&L": f"-₹{max_loss:,}", "Outcome": "Max loss ❌"},
            ],
            "payoff_spots": spots,
            "payoff_pnls": pnls,
            "max_profit": f"₹{max_profit:,}",
            "max_loss": f"-₹{max_loss:,}",
            "breakeven": f"{breakeven:,}",
        }

    elif action in ("BUY_CALL", "BUY_PUT"):
        strike = int(nifty // 100 * 100)
        premium = 180
        qty = 15
        max_loss_v = premium * qty
        target = premium * 1.5
        sl = premium * 0.5

        spots = list(range(int(nifty * 0.92), int(nifty * 1.08), int(nifty * 0.002)))
        if action == "BUY_CALL":
            pnls = [(max(0, s - strike) - premium) * qty for s in spots]
            breakeven_v = strike + premium
        else:
            pnls = [(max(0, strike - s) - premium) * qty for s in spots]
            breakeven_v = strike - premium

        return {
            "details": {
                "Strategy": "Buy ATM " + ("CE" if action == "BUY_CALL" else "PE"),
                "Strike": f"{strike} {'CE' if action == 'BUY_CALL' else 'PE'}",
                "Premium": f"₹{premium}/unit",
                "Quantity": f"{qty} units",
                "Max Risk": f"₹{max_loss_v:,}",
                "Book Profit When": f"Premium > ₹{target:.0f}",
                "Stop Loss When": f"Premium < ₹{sl:.0f}",
                "DTE": f"{dte} days",
                "Breakeven at Expiry": f"{breakeven_v:,}",
            },
            "scenarios": [
                {"Nifty at Expiry": f"{int(nifty*1.02):,}", "Move": "+2%",
                 "P&L": f"+₹{(max(0, int(nifty*1.02)-strike)-premium)*qty:,}",
                 "Outcome": "Profit ✅" if action == "BUY_CALL" else "Loss ❌"},
                {"Nifty at Expiry": f"{strike:,}", "Move": "ATM",
                 "P&L": f"-₹{premium*qty:,}", "Outcome": "Full loss ❌"},
                {"Nifty at Expiry": f"{int(nifty*0.98):,}", "Move": "-2%",
                 "P&L": f"-₹{premium*qty:,}",
                 "Outcome": "Loss ❌" if action == "BUY_CALL" else "Profit ✅"},
            ],
            "payoff_spots": spots,
            "payoff_pnls": pnls,
            "max_profit": "Unlimited" if action == "BUY_CALL" else f"₹{(strike - premium)*qty:,}",
            "max_loss": f"-₹{max_loss_v:,}",
            "breakeven": f"{breakeven_v:,}",
        }

    else:  # NO_TRADE
        return {
            "details": {"Status": "No trade recommended"},
            "scenarios": [],
            "payoff_spots": [],
            "payoff_pnls": [],
            "max_profit": "N/A",
            "max_loss": "N/A",
            "breakeven": "N/A",
        }


def simulate_monthly_pnl(n: int = 10000, seed: int = 42) -> np.ndarray:
    """Simulate N months of P&L using the strategy mix."""
    np.random.seed(seed)
    results = []
    for _ in range(n):
        # Credit spreads: 4 trades, 70% win rate
        cs_wins = np.random.binomial(4, 0.70)
        cs_pnl = cs_wins * 1500 - (4 - cs_wins) * 2000

        # Directional: 4 trades, 60% win rate
        dir_wins = np.random.binomial(4, 0.60)
        dir_pnl = dir_wins * 3000 - (4 - dir_wins) * 2000

        results.append(cs_pnl + dir_pnl)
    return np.array(results)


def compute_feature_importance() -> dict:
    """Return feature importance — from trained model if available, else static defaults."""
    try:
        import streamlit as st
        model = st.session_state.get("trained_model")
        feat_cols = st.session_state.get("feat_cols")
        if model is not None and feat_cols is not None:
            import pandas as pd
            importances = model.feature_importances_
            fi_df = pd.DataFrame({"feat": feat_cols, "imp": importances})
            fi_df = fi_df.sort_values("imp", ascending=False).head(10)
            return dict(zip(fi_df["feat"], fi_df["imp"]))
    except Exception:
        pass
    # Static fallback
    return {
        "VIX Change": 0.142,
        "RSI (14)": 0.118,
        "MACD Histogram": 0.097,
        "BB Width": 0.089,
        "VIX Spike": 0.082,
        "Stochastic K": 0.071,
        "Volume Ratio": 0.068,
        "ATR (14)": 0.063,
        "VIX Momentum": 0.058,
        "ROC (10)": 0.052,
    }
