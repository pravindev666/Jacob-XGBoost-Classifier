"""
pages/model_builder.py  (upgraded — uses real data, trains real XGBoost)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import build_master_dataset
from utils.features import engineer_all_features, get_feature_columns, get_feature_groups

DARK_BG = "#0e1117"
CARD_BG = "#1a1d2e"


def render():
    st.markdown("## 🤖 Model Builder")
    st.markdown("Train a **real XGBoost model** on your actual Nifty + VIX + PCR + FII data.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Feature Engineering",
        "🏋️ Train Model",
        "📊 Results & Importance",
        "✅ Validation"
    ])

    with tab1:
        st.markdown("### Feature Engineering Pipeline")
        groups = get_feature_groups()
        colors = {
            "Price Transforms (10)": "#3b82f6",
            "Moving Averages (9)": "#22c55e",
            "Momentum (13)": "#f59e0b",
            "Volatility (7)": "#ef4444",
            "Volume (5)": "#a855f7",
            "India VIX (14) ⭐": "#14b8a6",
            "PCR (4)": "#f97316",
            "FII/DII Flow (3)": "#ec4899",
            "Global Context (3)": "#6366f1",
            "Calendar (7)": "#84cc16",
            "Lag Features (7)": "#06b6d4",
        }
        col1, col2 = st.columns([1, 1])
        with col1:
            counts = {}
            for grp, feats in groups.items():
                num = int(grp.split("(")[1].split(")")[0]) if "(" in grp else len(feats)
                counts[grp] = num
            fig_donut = go.Figure(go.Pie(
                labels=list(counts.keys()), values=list(counts.values()),
                hole=0.6, marker_colors=[colors.get(k, "#888") for k in counts],
                textinfo="value", textfont_size=11,
            ))
            fig_donut.update_layout(
                paper_bgcolor=DARK_BG, font_color="#b0b0b0", height=320, showlegend=False,
                annotations=[dict(text="55+\nFeatures", x=0.5, y=0.5,
                                  font_size=15, font_color="#e0e0e0", showarrow=False)],
                margin=dict(l=0, r=0, t=10, b=10)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        with col2:
            selected = st.selectbox("Explore feature group", list(groups.keys()))
            color = colors.get(selected, "#888")
            for feat in groups[selected]:
                st.markdown(
                    f'<div style="background:{CARD_BG};border-left:3px solid {color};'
                    f'border-radius:4px;padding:5px 12px;margin-bottom:3px;'
                    f'font-size:12px;font-family:monospace;color:#e0e0e0;">{feat}</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        insights = [
            ("VIX (14 features)", "#14b8a6", "Your #1 predictor. VIX spikes precede bottoms. VIX < 15 = buy options. VIX > 20 = sell premium."),
            ("PCR (4 features)", "#f97316", "Contrarian signal. PCR > 1.2 = too many bears = bullish reversal likely."),
            ("FII/DII (3 features)", "#ec4899", "FII net buying for 5 days in a row = strong bull signal. Nifty follows institutional money."),
            ("SP500 (2 features)", "#6366f1", "Nifty follows US markets with a lag. Gap-down on US fall is a tradeable signal."),
            ("Bank Nifty (2 features)", "#22c55e", "Bank Nifty leads Nifty. BNF outperformance today = Nifty follows tomorrow."),
        ]
        for title, color, desc in insights:
            st.markdown(
                f'<div style="background:{CARD_BG};border-left:4px solid {color};'
                f'border-radius:6px;padding:10px 16px;margin-bottom:8px;">'
                f'<div style="color:{color};font-size:13px;font-weight:600;">{title}</div>'
                f'<div style="color:#b0b0b0;font-size:12px;margin-top:3px;">{desc}</div>'
                f'</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### Train Real XGBoost on Your Data")
        col1, col2 = st.columns(2)
        with col1:
            train_end = st.selectbox("Train until year", ["2020", "2021", "2022"], index=1)
            val_end   = st.selectbox("Validate until year", ["2022", "2023", "2024"], index=1)
        with col2:
            n_trees  = st.slider("n_estimators", 100, 1000, 300, 50)
            max_depth = st.slider("max_depth", 3, 10, 5)
            lr        = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2], 0.05)
            subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.1)

        if st.button("🚀 Load Data & Train XGBoost", type="primary"):
            prog = st.progress(0, "Loading datasets...")
            try:
                master   = build_master_dataset()
                featured = engineer_all_features(master)
                prog.progress(30, f"Loaded {len(featured):,} samples")
            except Exception as e:
                st.error(f"Error: {e}")
                return

            feat_cols = get_feature_columns(featured)
            X = featured[feat_cols].fillna(0)
            y = featured["target"]

            train_mask = featured.index.year <= int(train_end)
            val_mask   = (featured.index.year > int(train_end)) & (featured.index.year <= int(val_end))
            test_mask  = featured.index.year > int(val_end)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val,   y_val   = X[val_mask],   y[val_mask]
            X_test,  y_test  = X[test_mask],  y[test_mask]
            st.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
            prog.progress(40, "Scaling features...")

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
            X_val_s   = pd.DataFrame(scaler.transform(X_val),       columns=feat_cols, index=X_val.index)
            X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=feat_cols, index=X_test.index)
            prog.progress(55, "Training XGBoost...")

            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=n_trees, max_depth=max_depth, learning_rate=lr,
                    subsample=subsample, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=42, early_stopping_rounds=30, n_jobs=-1
                )
                model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            except ImportError:
                st.error("Run: pip install xgboost")
                return

            from sklearn.metrics import accuracy_score
            train_acc = accuracy_score(y_train, model.predict(X_train_s))
            val_acc   = accuracy_score(y_val,   model.predict(X_val_s))
            test_acc  = accuracy_score(y_test,  model.predict(X_test_s))
            prog.progress(100, "Done!")

            st.session_state.update({
                "trained_model": model, "scaler": scaler, "feat_cols": feat_cols,
                "X_test": X_test_s, "y_test": y_test,
                "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
                "featured_df": featured
            })

            c1, c2, c3 = st.columns(3)
            c1.metric("Train Accuracy", f"{train_acc*100:.1f}%")
            c2.metric("Val Accuracy",   f"{val_acc*100:.1f}%")
            c3.metric("Test Accuracy",  f"{test_acc*100:.1f}%",
                      "✅ Good" if test_acc > 0.55 else "⚠️ Below 55%")
            if train_acc - val_acc > 0.1:
                st.warning("Overfitting detected. Reduce max_depth.")
            elif val_acc >= 0.57:
                st.success("Solid model! Go to Results & Importance tab.")
        else:
            st.code("""master   = build_master_dataset()   # All your CSVs merged
featured = engineer_all_features(master)   # 55+ features
X_train  = features[year <= train_end]    # Time-series split
model    = XGBClassifier(n_estimators=300, max_depth=5)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
prob     = model.predict_proba(X_today)[0][1]
confidence = abs(prob - 0.5) * 200""", language="python")

    with tab3:
        st.markdown("### Model Results")
        if "trained_model" not in st.session_state:
            st.info("Train the model first.")
            return

        model     = st.session_state["trained_model"]
        feat_cols = st.session_state["feat_cols"]
        X_test    = st.session_state["X_test"]
        y_test    = st.session_state["y_test"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top 20 Feature Importance")
            importance = model.feature_importances_
            fi_df = pd.DataFrame({"Feature": feat_cols, "Importance": importance})
            fi_df = fi_df.sort_values("Importance", ascending=False).head(20)
            group_colors = {"vix": "#14b8a6", "pcr": "#f97316", "fii": "#ec4899",
                            "rsi": "#f59e0b", "macd": "#f59e0b", "stoch": "#f59e0b",
                            "bb": "#ef4444", "atr": "#ef4444", "vol": "#a855f7",
                            "sma": "#22c55e", "ema": "#22c55e", "sp500": "#6366f1",
                            "bnf": "#22c55e", "return": "#3b82f6", "obv": "#a855f7"}
            bar_colors = []
            for feat in fi_df["Feature"]:
                c = "#3b82f6"
                for key, col in group_colors.items():
                    if key in feat.lower(): c = col; break
                bar_colors.append(c)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                marker_color=bar_colors, text=[f"{v:.3f}" for v in fi_df["Importance"]],
                textposition="outside"
            ))
            fig_fi.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font_color="#b0b0b0", height=520, showlegend=False,
                yaxis=dict(autorange="reversed"), margin=dict(l=10, r=60, t=10, b=10))
            fig_fi.update_xaxes(gridcolor="#1e2130")
            st.plotly_chart(fig_fi, use_container_width=True)

        with col2:
            st.markdown("#### Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, model.predict(X_test))
            total = cm.sum()
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Pred: DOWN", "Pred: UP"], y=["Actual: DOWN", "Actual: UP"],
                colorscale=[[0, "#1a1d2e"], [1, "#3b82f6"]],
                text=[[f"{v}\n({v/total*100:.0f}%)" for v in row] for row in cm],
                texttemplate="%{text}", textfont_size=14,
            ))
            fig_cm.update_layout(paper_bgcolor=DARK_BG, font_color="#b0b0b0",
                height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

            acc = (cm[0,0]+cm[1,1])/total
            prec = cm[1,1]/(cm[0,1]+cm[1,1]+1e-9)
            rec  = cm[1,1]/(cm[1,0]+cm[1,1]+1e-9)
            f1   = 2*prec*rec/(prec+rec+1e-9)
            mc1, mc2 = st.columns(2)
            mc1.metric("Accuracy", f"{acc*100:.1f}%")
            mc2.metric("F1 Score", f"{f1:.3f}")

            st.markdown("---")
            st.markdown("#### Today's Model Signal")
            featured_df = st.session_state.get("featured_df")
            if featured_df is not None:
                latest = featured_df[feat_cols].iloc[[-1]].fillna(0)
                latest_s = pd.DataFrame(
                    st.session_state["scaler"].transform(latest), columns=feat_cols)
                prob = model.predict_proba(latest_s)[0][1]
                conf = abs(prob - 0.5) * 200
                direction = "UP ↑" if prob > 0.5 else "DOWN ↓"
                color = "#22c55e" if prob > 0.5 else "#ef4444"
                conf_color = "#22c55e" if conf >= 70 else "#f59e0b" if conf >= 55 else "#ef4444"
                action = ("Buy ATM CE" if prob > 0.5 and conf >= 70 else
                          "Bull Put Spread" if prob > 0.5 and conf >= 55 else
                          "Buy ATM PE" if prob <= 0.5 and conf >= 70 else
                          "Bear Call Spread" if prob <= 0.5 and conf >= 55 else "No Trade")
                st.markdown(
                    f'<div style="background:{CARD_BG};border:2px solid {color};'
                    f'border-radius:12px;padding:20px;">'
                    f'<div style="font-size:22px;font-weight:700;color:{color};">'
                    f'{direction} — {action}</div>'
                    f'<div style="color:#b0b0b0;margin-top:8px;">'
                    f'Probability: <span style="color:{color};">{prob*100:.1f}%</span> &nbsp;|&nbsp;'
                    f'Confidence: <span style="color:{conf_color};">{conf:.0f}%</span></div>'
                    f'</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown("### Anti-Leakage Checklist")
        checks = [
            ("✅", "TimeSeriesSplit only", True, "No random shuffle — past trains future only"),
            ("✅", "All lags use .shift(1) or more", True, "No same-day future data in features"),
            ("✅", "Target = next day (shift -1)", True, "Predict tomorrow, not today"),
            ("✅", "Scaler fit on train set only", True, "Prevents test data bleeding into scaling"),
            ("✅", "Early stopping on validation", True, "XGBoost stops before memorising"),
            ("⚠️", "VIX is same-day", False, "Acceptable for EOD trading — VIX available at 3:30 PM"),
            ("⚠️", "PCR may be same-day", False, "Use previous day PCR to be safe"),
            ("❌", "Never use Close.shift(-1) in features", False, "That's the future!"),
        ]
        for icon, title, good, explanation in checks:
            color = "#22c55e" if icon == "✅" else "#f59e0b" if icon == "⚠️" else "#ef4444"
            st.markdown(
                f'<div style="background:{"#0d2818" if good else CARD_BG};'
                f'border:1px solid {color};border-radius:8px;padding:10px 16px;margin-bottom:8px;">'
                f'<div style="display:flex;gap:10px;align-items:center;">'
                f'<span style="font-size:16px;">{icon}</span>'
                f'<span style="color:{color};font-weight:600;font-size:13px;">{title}</span></div>'
                f'<div style="color:#b0b0b0;font-size:12px;padding-left:28px;margin-top:4px;">{explanation}</div>'
                f'</div>', unsafe_allow_html=True)
