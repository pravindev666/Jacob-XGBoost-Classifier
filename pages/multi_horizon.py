import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import build_master_dataset
from utils.features import engineer_all_features, get_feature_columns

DARK_BG = "#0e1117"
CARD_BG = "#1a1d2e"

HORIZONS = {
    "1D":  {"col": "target",     "days": 1,  "color": "#378ADD", "options": "Weekly CE/PE"},
    "3D":  {"col": "target_3d",  "days": 3,  "color": "#5b9bd5", "options": "Weekly spread 7DTE"},
    "5D":  {"col": "target_5d",  "days": 5,  "color": "#1D9E75", "options": "Weekly or monthly spread ⭐"},
    "7D":  {"col": "target_7d",  "days": 7,  "color": "#22c55e", "options": "Monthly spread 10–14DTE ⭐"},
    "14D": {"col": "target_14d", "days": 14, "color": "#f59e0b", "options": "Monthly spread 21DTE"},
    "21D": {"col": "target_21d", "days": 21, "color": "#ef8c34", "options": "Monthly expiry spread"},
    "30D": {"col": "target_30d", "days": 30, "color": "#ef4444", "options": "Monthly only — 1 trade/mo"},
}


def _get_conviction(probs: dict) -> dict:
    """Compute conviction score from all horizon probabilities."""
    up_votes   = sum(1 for p in probs.values() if p > 0.5)
    down_votes = sum(1 for p in probs.values() if p <= 0.5)
    avg_prob   = np.mean(list(probs.values()))
    direction  = "UP" if avg_prob > 0.5 else "DOWN"
    conviction = up_votes if direction == "UP" else down_votes
    confidence = abs(avg_prob - 0.5) * 200
    return {
        "direction": direction,
        "up_votes":  up_votes,
        "down_votes": down_votes,
        "conviction": conviction,
        "confidence": confidence,
        "avg_prob": avg_prob,
    }


def render():
    st.markdown("## 📊 Multi-Horizon Intelligence")
    st.markdown("7 XGBoost models predict Nifty direction from 1 day to 30 days ahead — simultaneously.")
    st.markdown(
        '<div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#b0b0b0;">'
        '💡 <b style="color:#e0e0e0;">How this works:</b> Instead of guessing just tomorrow, we ask 7 separate AI models: '
        '"Will Nifty go up in 1 day? 3 days? 5 days? 7 days? 14 days? 21 days? 30 days?" '
        'When most models agree → 🟢 strong signal. When they disagree → 🔴 stay out.'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Check if models trained ───────────────────────────────────────────────
    models = st.session_state.get("mh_models", {})
    accuracies = st.session_state.get("mh_accuracies", {})

    tab_train, tab_conviction, tab_accuracy = st.tabs([
        "🏋️ Train All Horizons",
        "🎯 Conviction Panel",
        "📊 Accuracy Comparison",
    ])

    # ── Tab 1: Train ──────────────────────────────────────────────────────────
    with tab_train:
        st.markdown("### Train 7 XGBoost Models (one per horizon)")
        st.markdown(
            '<div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:12px;color:#b0b0b0;">'
            '💡 <b style="color:#e0e0e0;">What is training?</b> The AI reads years of past Nifty data and learns patterns. '
            'Like studying for an exam — the more history it reads, the better it predicts the future. '
            'Click the button below to start training all 7 models.'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            train_end = st.selectbox("Train until year", ["2019", "2020", "2021"], index=1)
            val_end   = st.selectbox("Validate until year", ["2022", "2023", "2024"], index=2)
            horizons_to_train = st.multiselect(
                "Horizons to train",
                list(HORIZONS.keys()),
                default=list(HORIZONS.keys())
            )
        with col2:
            n_trees   = st.slider("n_estimators", 100, 800, 500, 50)
            max_depth = st.slider("max_depth", 2, 6, 3)
            lr        = st.select_slider("learning_rate", [0.01, 0.05, 0.1], 0.01)

        st.info(f"This trains **{len(horizons_to_train)} separate models** using the same 55+ features "
                f"but different target labels (1D target, 5D target, 7D target etc). "
                f"Takes ~{len(horizons_to_train) * 15} seconds.")

        if st.button("🚀 Train All Horizon Models", type="primary"):
            try:
                master   = build_master_dataset()
                featured = engineer_all_features(master, dropna=True)
            except Exception as e:
                st.error(f"Data error: {e}")
                return

            feat_cols = get_feature_columns(featured)
            # Remove all target columns from features
            feat_cols = [c for c in feat_cols if not c.startswith("target")]
            X = featured[feat_cols].fillna(0)

            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score
            import xgboost as xgb

            train_mask = featured.index.year <= int(train_end)
            val_mask   = (featured.index.year > int(train_end)) & (featured.index.year <= int(val_end))
            test_mask  = featured.index.year > int(val_end)

            X_train = X[train_mask]
            X_val   = X[val_mask]
            X_test  = X[test_mask]

            scaler = StandardScaler()
            X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
            X_val_s   = pd.DataFrame(scaler.transform(X_val),       columns=feat_cols, index=X_val.index)
            X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=feat_cols, index=X_test.index)

            trained_models = {}
            acc_results    = {}

            prog = st.progress(0, "Training models...")
            for i, hz in enumerate(horizons_to_train):
                target_col = HORIZONS[hz]["col"]
                if target_col not in featured.columns:
                    continue

                y_train = featured[target_col][train_mask]
                y_val   = featured[target_col][val_mask]
                y_test  = featured[target_col][test_mask]

                # Align — drop rows where target is NaN
                valid_train = y_train.dropna().index
                valid_val   = y_val.dropna().index
                valid_test  = y_test.dropna().index

                model = xgb.XGBClassifier(
                    n_estimators=n_trees, max_depth=max_depth,
                    learning_rate=lr, subsample=0.8,
                    colsample_bytree=0.8, eval_metric="logloss",
                    random_state=42, early_stopping_rounds=30, n_jobs=-1
                )
                model.fit(
                    X_train_s.loc[valid_train], y_train.loc[valid_train],
                    eval_set=[(X_val_s.loc[valid_val], y_val.loc[valid_val])],
                    verbose=False
                )

                test_acc = accuracy_score(y_test.loc[valid_test],
                                          model.predict(X_test_s.loc[valid_test]))
                trained_models[hz] = model
                acc_results[hz] = {
                    "test_acc": test_acc,
                    "train_acc": accuracy_score(y_train.loc[valid_train],
                                               model.predict(X_train_s.loc[valid_train])),
                    "color": HORIZONS[hz]["color"],
                }

                pct = int((i + 1) / len(horizons_to_train) * 100)
                prog.progress(pct, f"Trained {hz} → Test: {test_acc*100:.1f}%")

            st.session_state["mh_models"]     = trained_models
            st.session_state["mh_scaler"]     = scaler
            st.session_state["mh_feat_cols"]  = feat_cols
            st.session_state["mh_featured"]   = featured
            st.session_state["mh_accuracies"] = acc_results

            st.success(f"✅ Trained {len(trained_models)} models!")
            for hz, acc in acc_results.items():
                col_str = "✅" if acc["test_acc"] > 0.55 else "⚠️"
                st.write(f"{col_str} **{hz}**: Test {acc['test_acc']*100:.1f}% | Train {acc['train_acc']*100:.1f}%")
        else:
            st.code("""# Each horizon trains on same features, different target
for horizon, target_col in [("1D","target"), ("5D","target_5d"), ("7D","target_7d")...]:
    model = XGBClassifier(n_estimators=500, max_depth=3, lr=0.01)
    model.fit(X_train, y_train[target_col],
              eval_set=[(X_val, y_val[target_col])],
              early_stopping_rounds=30)
    models[horizon] = model

# Live prediction
for hz, model in models.items():
    prob = model.predict_proba(X_today)[0][1]
    confidence = abs(prob - 0.5) * 200
""", language="python")

    # ── Tab 2: Conviction Panel ───────────────────────────────────────────────
    with tab_conviction:
        st.markdown("### All-Horizon Conviction — Live Predictions")
        st.markdown(
            '<div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:12px;color:#b0b0b0;">'
            '💡 <b style="color:#e0e0e0;">What is conviction?</b> If 6 out of 7 models agree the market will go UP, '
            'that\'s 6/7 conviction = very strong signal. If only 4/7 agree, it\'s weak. '
            'Think of it like asking 7 weather apps — if most agree, you can trust it.'
            '</div>',
            unsafe_allow_html=True
        )

        if not models:
            st.warning("⚠️ Train the models first in the **Train All Horizons** tab.")
        else:
            featured_df = st.session_state.get("mh_featured")
            scaler      = st.session_state.get("mh_scaler")
            feat_cols   = st.session_state.get("mh_feat_cols")

            if featured_df is None or scaler is None:
                st.warning("No feature data found. Retrain models.")
            else:
                latest = featured_df[feat_cols].iloc[[-1]].fillna(0)
                latest_s = pd.DataFrame(scaler.transform(latest), columns=feat_cols)

                # Get probabilities from all models
                probs = {}
                for hz, model in models.items():
                    prob = model.predict_proba(latest_s)[0][1]
                    probs[hz] = prob

                conv = _get_conviction(probs)

                # Big conviction display
                conv_color = "#22c55e" if conv["direction"] == "UP" else "#ef4444"
                st.markdown(
                    f'<div style="background:{CARD_BG};border:2px solid {conv_color};'
                    f'border-radius:12px;padding:20px 24px;margin-bottom:20px;display:flex;'
                    f'align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">'
                    f'<div>'
                    f'<div style="font-size:28px;font-weight:700;color:{conv_color};">'
                    f'{"↑" if conv["direction"]=="UP" else "↓"} {conv["direction"]} — '
                    f'{conv["conviction"]}/7 horizons agree</div>'
                    f'<div style="color:#b0b0b0;font-size:14px;margin-top:4px;">'
                    f'Avg confidence: {conv["confidence"]:.0f}% · '
                    f'Up votes: {conv["up_votes"]} · Down votes: {conv["down_votes"]}</div>'
                    f'</div>'
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:48px;font-weight:700;color:{conv_color};">'
                    f'{conv["conviction"]}/7</div>'
                    f'<div style="color:#888;font-size:12px;">Horizon votes</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Per-horizon cards
                cols = st.columns(7)
                for i, (hz, cfg) in enumerate(HORIZONS.items()):
                    if hz not in probs:
                        continue
                    prob = probs[hz]
                    conf = abs(prob - 0.5) * 200
                    direction = "UP" if prob > 0.5 else "DOWN"
                    d_color = "#22c55e" if prob > 0.5 else "#ef4444"
                    bg = "#0d2818" if prob > 0.5 else "#1a0d0d"
                    with cols[i]:
                        st.markdown(
                            f'<div style="background:{bg};border:1px solid {cfg["color"]};'
                            f'border-radius:8px;padding:10px 8px;text-align:center;">'
                            f'<div style="font-size:13px;font-weight:600;color:{cfg["color"]};">{hz}</div>'
                            f'<div style="font-size:18px;font-weight:700;color:{d_color};margin:4px 0;">'
                            f'{"↑" if prob>0.5 else "↓"} {prob*100:.0f}%</div>'
                            f'<div style="font-size:10px;color:#888;">conf {conf:.0f}%</div>'
                            f'<div style="height:4px;background:#2a2d3e;border-radius:2px;margin-top:6px;overflow:hidden;">'
                            f'<div style="height:100%;width:{min(conf,100):.0f}%;background:{d_color};border-radius:2px;"></div>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # What this means for trading
                st.markdown("---")
                st.markdown("#### What This Means for Today's Trade")

                # Read live VIX/Nifty from data_loader instead of stale session keys
                import utils.data_loader as dl
                _nifty_df = dl.load_nifty_daily()
                _vix_df = dl.load_vix_daily()
                vix = float(_vix_df.iloc[-1]['VIX']) if _vix_df is not None and not _vix_df.empty else 18.0
                nifty = int(_nifty_df.iloc[-1]['Close']) if _nifty_df is not None and not _nifty_df.empty else 22000

                # Determine best actionable horizon
                actionable = {hz: p for hz, p in probs.items()
                              if abs(p - 0.5) * 200 >= 55}

                if not actionable:
                    st.markdown(
                        f'<div style="background:#1a0d0d;border:2px solid #5c1a1a;'
                        f'border-radius:10px;padding:16px 20px;">'
                        f'<div style="font-size:18px;font-weight:600;color:#ef4444;">⛔ No Trade — All Horizons Below 55% Confidence</div>'
                        f'<div style="color:#b0b0b0;margin-top:6px;font-size:13px;">'
                        f'Highest confidence today: {max(abs(p-0.5)*200 for p in probs.values()):.0f}%. '
                        f'Threshold is 55%. Wait for a stronger setup. Patience is edge.</div>'
                        f'<div style="color:#888;font-size:12px;margin-top:8px;font-style:italic;">'
                        f'💡 Think of it like weather: if 4 out of 7 forecasters say "sunny" and 3 say "rain" — you bring an umbrella. '
                        f'Same here: when models disagree, we sit out and wait for clarity.</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    best_hz = max(actionable, key=lambda h: abs(probs[h] - 0.5))
                    best_prob = probs[best_hz]
                    best_conf = abs(best_prob - 0.5) * 200
                    best_dir = "UP" if best_prob > 0.5 else "DOWN"

                    if vix > 20:
                        strategy = "Bull Put Spread" if best_dir == "UP" else "Bear Call Spread"
                        reason = f"VIX {vix:.1f} > 20 — Credit spread only (no option buying)"
                    elif best_conf >= 70:
                        strategy = "Buy ATM Call (CE)" if best_dir == "UP" else "Buy ATM Put (PE)"
                        reason = f"VIX {vix:.1f} < 20 and high confidence — buy options"
                    else:
                        strategy = "Bull Put Spread" if best_dir == "UP" else "Bear Call Spread"
                        reason = f"Medium confidence — credit spread safer"

                    s_color = "#22c55e" if best_dir == "UP" else "#f59e0b"
                    st.markdown(
                        f'<div style="background:#0d2818;border:2px solid #1a5c36;'
                        f'border-radius:10px;padding:16px 20px;">'
                        f'<div style="font-size:20px;font-weight:600;color:{s_color};">'
                        f'⚡ {strategy} — Best signal from {best_hz} horizon</div>'
                        f'<div style="color:#b0b0b0;margin-top:6px;font-size:13px;">'
                        f'Direction: {best_dir} · Probability: {best_prob*100:.1f}% · Confidence: {best_conf:.0f}% · {reason}</div>'
                        f'<div style="color:#888;font-size:12px;margin-top:4px;">'
                        f'Other actionable horizons: {", ".join(h for h in actionable if h != best_hz) or "none"}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ── Tab 3: Accuracy Comparison ────────────────────────────────────────────
    with tab_accuracy:
        st.markdown("### Accuracy by Horizon")
        st.markdown(
            '<div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:12px;color:#b0b0b0;">'
            '💡 <b style="color:#e0e0e0;">What does accuracy mean here?</b> If a model has 58% accuracy, it correctly predicted '
            'the market direction 58 out of 100 times on data it had never seen before. '
            'That\'s enough to make money because we manage risk on every trade.'
            '</div>',
            unsafe_allow_html=True
        )

        if not accuracies:
            st.warning("Train models first in the **Train All Horizons** tab.")
        else:
            hz_labels = list(accuracies.keys())
            test_accs  = [accuracies[h]["test_acc"] * 100 for h in hz_labels]
            train_accs = [accuracies[h]["train_acc"] * 100 for h in hz_labels]
            # Strip legacy "88" alpha channel from old cached session state colors
            colors     = [
                c[:7] if isinstance(c, str) and c.startswith("#") and len(c) >= 7 else c 
                for c in [accuracies[h]["color"] for h in hz_labels]
            ]

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Train", x=hz_labels, y=train_accs,
                                     marker_color=colors, opacity=0.5))
                fig.add_trace(go.Bar(name="Test (live proxy)", x=hz_labels, y=test_accs,
                                     marker_color=colors,
                                     text=[f"{a:.1f}%" for a in test_accs],
                                     textposition="outside"))
                fig.add_hline(y=55, line_dash="dash", line_color="#888",
                              annotation_text="55% min threshold")
                fig.update_layout(
                    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                    font_color="#b0b0b0", height=320, barmode="group",
                    yaxis_range=[45, max(train_accs) + 10],
                    yaxis_title="Accuracy (%)",
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=10, r=10, t=10, b=40)
                )
                fig.update_yaxes(gridcolor="#1e2130")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Expected monthly P&L by horizon**")
                trades_per_month = {"1D": 20, "3D": 7, "5D": 4, "7D": 4,
                                    "14D": 2, "21D": 1, "30D": 1}
                avg_win, avg_loss = 2250, 2000

                pnl_data = []
                for hz in hz_labels:
                    if hz not in accuracies:
                        continue
                    wr = accuracies[hz]["test_acc"]
                    trades = trades_per_month.get(hz, 2)
                    ev = (wr * avg_win - (1-wr) * avg_loss) * trades
                    pnl_data.append({"Horizon": hz, "Win Rate": f"{wr*100:.1f}%",
                                      "Trades/mo": trades, "Expected P&L": f"₹{ev:,.0f}"})

                df = pd.DataFrame(pnl_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.markdown(
                    '<div style="background:#0d2818;border-radius:8px;padding:12px 14px;margin-top:8px;">'
                    '<div style="font-size:13px;font-weight:500;color:#22c55e;">Best horizons for P&L</div>'
                    '<div style="font-size:12px;color:#b0b0b0;margin-top:4px;">5D and 7D give the best balance of accuracy improvement + enough trades per month. '
                    '21D and 30D have highest accuracy but only 1 trade/month limits total P&L.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
