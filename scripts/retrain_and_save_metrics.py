"""
scripts/retrain_and_save_metrics.py
-------------------------------------
Retrains XGBoost on all 7 horizons using latest CSV data.
Saves accuracy metrics to data/model_metrics.json.
The Streamlit app reads this JSON to show current model accuracy
without needing to retrain inside the app every time.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading CSVs...")

def load_csv(fname, date_col="Date"):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col).sort_index()
    return df

nifty = load_csv("nifty_daily.csv")
vix   = load_csv("vix_daily.csv")
pcr   = load_csv("pcr_daily.csv")
fii   = load_csv("fii_dii_daily.csv")
bnf   = load_csv("bank_nifty_daily.csv")
sp500 = load_csv("sp500_daily.csv")
vterm = load_csv("vix_term_daily.csv")

print(f"Nifty rows: {len(nifty)}, VIX rows: {len(vix)}")

# ── Build master dataset ─────────────────────────────────────────────────────
master = nifty[["Open", "High", "Low", "Close"]].copy()
if "Volume" in nifty.columns:
    master["Volume"] = nifty["Volume"]

if not vix.empty:
    for col in ["Close", "Vix Close", "India Vix", "Vix", "VIX"]:
        if col in vix.columns:
            vix["VIX"] = vix[col]; break
    if "VIX" not in vix.columns and len(vix.columns) > 0:
        vix["VIX"] = vix.iloc[:, 0]
    if "VIX" in vix.columns:
        master = master.join(vix[["VIX"]], how="left")
        master["VIX"] = master["VIX"].ffill()

if not pcr.empty:
    for col in ["PCR", "Pcr", "Put Call Ratio", "Pcr Oi"]:
        if col in pcr.columns:
            pcr.rename(columns={col: "PCR"}, inplace=True); break
    if "PCR" in pcr.columns:
        master = master.join(pcr[["PCR"]], how="left")
        master["PCR"] = master["PCR"].ffill()

if not fii.empty:
    for col in ["fii_net", "FII_Net", "FII_Net_Buy", "fii_net_buy"]:
        if col in fii.columns:
            fii.rename(columns={col: "FII_Net"}, inplace=True); break
    if "FII_Net" in fii.columns:
        master = master.join(fii[["FII_Net"]], how="left")

if not bnf.empty and "Close" in bnf.columns:
    master = master.join(bnf["Close"].pct_change().rename("BankNifty_Return"), how="left")

if not sp500.empty and "Close" in sp500.columns:
    master = master.join(sp500["Close"].pct_change().rename("SP500_Return"), how="left")

print(f"Master dataset: {len(master)} rows × {len(master.columns)} columns")

# ── Feature engineering (inline — no streamlit dependency) ───────────────────
def make_features(df):
    f = df.copy()
    f["daily_return"] = f["Close"].pct_change()
    f["log_return"]   = np.log(f["Close"] / f["Close"].shift(1))
    for w in [5, 10, 20, 50, 200]:
        f[f"sma_{w}"] = f["Close"].rolling(w).mean()
    f["ema_12"] = f["Close"].ewm(span=12).mean()
    f["ema_26"] = f["Close"].ewm(span=26).mean()
    f["price_vs_sma20"] = (f["Close"] - f["sma_20"]) / f["sma_20"]
    f["sma5_vs_sma20"]  = (f["sma_5"] - f["sma_20"]) / f["sma_20"]
    delta = f["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"]       = 100 - 100 / (1 + gain / (loss + 1e-9))
    f["macd"]      = f["ema_12"] - f["ema_26"]
    f["macd_sig"]  = f["macd"].ewm(span=9).mean()
    f["macd_hist"] = f["macd"] - f["macd_sig"]
    for p in [5, 10, 20]:
        f[f"roc_{p}"] = f["Close"].pct_change(p) * 100
    low14  = f["Low"].rolling(14).min()
    high14 = f["High"].rolling(14).max()
    f["stoch_k"] = (f["Close"] - low14) / (high14 - low14 + 1e-9) * 100
    tr = pd.concat([f["High"]-f["Low"],
                    (f["High"]-f["Close"].shift(1)).abs(),
                    (f["Low"] -f["Close"].shift(1)).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean()
    std20 = f["Close"].rolling(20).std()
    f["bb_width"] = (4 * std20) / (f["sma_20"] + 1e-9)
    f["pct_b"]    = (f["Close"] - (f["sma_20"] - 2*std20)) / (4*std20 + 1e-9)
    f["hist_vol_20"] = f["log_return"].rolling(20).std() * np.sqrt(252)
    if "VIX" in f.columns:
        f["vix_change"]   = f["VIX"].diff()
        f["vix_sma_5"]    = f["VIX"].rolling(5).mean()
        f["vix_momentum"] = f["VIX"] - f["VIX"].shift(5)
        f["vix_spike"]    = (f["VIX"] > 20).astype(int)
        f["vix_low"]      = (f["VIX"] < 15).astype(int)
        f["vix_nifty_corr"] = f["VIX"].rolling(20).corr(f["daily_return"])
        f["vix_lag_1"] = f["VIX"].shift(1)
    if "SP500_Return" in f.columns:
        f["sp500_positive"] = (f["SP500_Return"] > 0).astype(int)
    if "BankNifty_Return" in f.columns:
        f["bnf_leading"] = f["BankNifty_Return"].shift(1)
    if "PCR" in f.columns:
        f["pcr_high"] = (f["PCR"] > 1.2).astype(int)
        f["pcr_low"]  = (f["PCR"] < 0.8).astype(int)
    if "FII_Net" in f.columns:
        f["fii_positive"] = (f["FII_Net"] > 0).astype(int)
        f["fii_sma_5"]    = f["FII_Net"].rolling(5).mean()
    f["day_of_week"]  = f.index.dayofweek
    f["month"]        = f.index.month
    f["quarter"]      = f.index.quarter
    for lag in [1, 2, 3, 5]:
        f[f"return_lag_{lag}"] = f["daily_return"].shift(lag)

    # Targets — use float to preserve NaN for last N rows
    for col, n in [("target",1),("target_3d",3),("target_5d",5),
                   ("target_7d",7),("target_14d",14),("target_21d",21),("target_30d",30)]:
        future = f["Close"].shift(-n)
        f[col] = np.where(future.isna(), np.nan, (future > f["Close"]).astype(float))

    TARGET_COLS = ["target","target_3d","target_5d","target_7d",
                   "target_14d","target_21d","target_30d"]
    return f.dropna(subset=TARGET_COLS)

print("Engineering features...")
featured = make_features(master)
print(f"Featured: {len(featured)} rows × {len(featured.columns)} cols")

EXCLUDE = {"Open","High","Low","Close","Volume","VIX","PCR","FII_Net",
           "SP500_Return","BankNifty_Return","VIX_Term_Spread"}
TARGETS = {c for c in featured.columns if c.startswith("target")}
feat_cols = [c for c in featured.columns if c not in EXCLUDE and c not in TARGETS]

# ── Train all 7 horizon models ───────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

HORIZONS = {
    "1D":  "target",     "3D":  "target_3d",  "5D":  "target_5d",
    "7D":  "target_7d",  "14D": "target_14d", "21D": "target_21d",
    "30D": "target_30d",
}

# Time-series split
TRAIN_END = 2021
VAL_END   = 2024

X = featured[feat_cols].fillna(0)
train_mask = featured.index.year <= TRAIN_END
val_mask   = (featured.index.year > TRAIN_END) & (featured.index.year <= VAL_END)
test_mask  = featured.index.year > VAL_END

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X[train_mask]),
                          columns=feat_cols, index=X[train_mask].index)
X_val_s   = pd.DataFrame(scaler.transform(X[val_mask]),
                          columns=feat_cols, index=X[val_mask].index)
X_test_s  = pd.DataFrame(scaler.transform(X[test_mask]),
                          columns=feat_cols, index=X[test_mask].index)

metrics = {
    "updated_at": datetime.utcnow().isoformat() + "Z",
    "train_end":  TRAIN_END,
    "val_end":    VAL_END,
    "total_rows": len(featured),
    "train_rows": int(train_mask.sum()),
    "val_rows":   int(val_mask.sum()),
    "test_rows":  int(test_mask.sum()),
    "horizons":   {},
}

for hz, target_col in HORIZONS.items():
    y_train = featured[target_col][train_mask].dropna()
    y_val   = featured[target_col][val_mask].dropna()
    y_test  = featured[target_col][test_mask].dropna()

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
        early_stopping_rounds=30, n_jobs=-1
    )
    model.fit(
        X_train_s.loc[y_train.index], y_train,
        eval_set=[(X_val_s.loc[y_val.index], y_val)],
        verbose=False
    )

    test_acc = accuracy_score(y_test, model.predict(X_test_s.loc[y_test.index]))
    val_acc  = accuracy_score(y_val,  model.predict(X_val_s.loc[y_val.index]))
    print(f"  {hz}: val={val_acc*100:.1f}%  test={test_acc*100:.1f}%")

    metrics["horizons"][hz] = {
        "target_col":  target_col,
        "val_accuracy":  round(float(val_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
    }

# Save metrics JSON (Streamlit can read this without retraining)
metrics_path = os.path.join(DATA_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print()
print(f"Metrics saved to data/model_metrics.json")
print("Done!")
