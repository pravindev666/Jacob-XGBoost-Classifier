"""
features.py
-----------
Engineers all 55+ predictive features from the master dataset.
Uses your real CSV data: Nifty OHLCV + VIX + PCR + FII + BankNifty + SP500.
"""

import numpy as np
import pandas as pd


def engineer_all_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Input:  master DataFrame with columns: Open, High, Low, Close, Volume,
            VIX (optional), PCR (optional), FII_Net (optional),
            BankNifty_Return (optional), SP500_Return (optional)
    Output: DataFrame with 55+ features + 'target' column (1 = next day UP)
    """
    f = df.copy()

    # ── 1. Price transforms ──────────────────────────────────────────────────
    f["daily_return"]  = f["Close"].pct_change()
    f["log_return"]    = np.log(f["Close"] / f["Close"].shift(1))
    f["hl_range"]      = f["High"] - f["Low"]
    f["oc_range"]      = f["Close"] - f["Open"]
    f["body_size"]     = (f["Close"] - f["Open"]).abs()
    f["upper_shadow"]  = f["High"] - f[["Open", "Close"]].max(axis=1)
    f["lower_shadow"]  = f[["Open", "Close"]].min(axis=1) - f["Low"]
    tr = pd.concat([
        f["High"] - f["Low"],
        (f["High"] - f["Close"].shift(1)).abs(),
        (f["Low"]  - f["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    f["true_range"]    = tr
    f["gap_up"]        = (f["Open"] > f["Close"].shift(1)).astype(int)
    f["gap_pct"]       = (f["Open"] - f["Close"].shift(1)) / f["Close"].shift(1)

    # ── 2. Moving averages ───────────────────────────────────────────────────
    for w in [5, 10, 20, 50, 200]:
        f[f"sma_{w}"] = f["Close"].rolling(w).mean()
    f["ema_12"] = f["Close"].ewm(span=12).mean()
    f["ema_26"] = f["Close"].ewm(span=26).mean()
    f["ema_9"]  = f["Close"].ewm(span=9).mean()

    # Price vs MA signals
    f["price_vs_sma20"]  = (f["Close"] - f["sma_20"]) / f["sma_20"]
    f["price_vs_sma50"]  = (f["Close"] - f["sma_50"]) / f["sma_50"]
    f["sma5_vs_sma20"]   = (f["sma_5"] - f["sma_20"]) / f["sma_20"]
    f["sma20_vs_sma50"]  = (f["sma_20"] - f["sma_50"]) / f["sma_50"]
    f["golden_cross"]    = ((f["sma_50"] > f["sma_200"]) &
                            (f["sma_50"].shift(1) <= f["sma_200"].shift(1))).astype(int)
    f["death_cross"]     = ((f["sma_50"] < f["sma_200"]) &
                            (f["sma_50"].shift(1) >= f["sma_200"].shift(1))).astype(int)

    # ── 3. Momentum indicators ───────────────────────────────────────────────
    delta = f["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    f["rsi_overbought"] = (f["rsi"] > 70).astype(int)
    f["rsi_oversold"]   = (f["rsi"] < 30).astype(int)

    f["macd"]        = f["ema_12"] - f["ema_26"]
    f["macd_signal"] = f["macd"].ewm(span=9).mean()
    f["macd_hist"]   = f["macd"] - f["macd_signal"]
    f["macd_cross_up"]   = ((f["macd"] > f["macd_signal"]) &
                             (f["macd"].shift(1) <= f["macd_signal"].shift(1))).astype(int)
    f["macd_cross_down"] = ((f["macd"] < f["macd_signal"]) &
                             (f["macd"].shift(1) >= f["macd_signal"].shift(1))).astype(int)

    for p in [5, 10, 20]:
        f[f"roc_{p}"]      = f["Close"].pct_change(p) * 100
        f[f"momentum_{p}"] = f["Close"] - f["Close"].shift(p)

    low_14  = f["Low"].rolling(14).min()
    high_14 = f["High"].rolling(14).max()
    f["stoch_k"] = (f["Close"] - low_14) / (high_14 - low_14 + 1e-9) * 100
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()
    f["stoch_cross"] = ((f["stoch_k"] > f["stoch_d"]) &
                        (f["stoch_k"].shift(1) <= f["stoch_d"].shift(1))).astype(int)

    f["williams_r"] = (high_14 - f["Close"]) / (high_14 - low_14 + 1e-9) * -100
    tp = (f["High"] + f["Low"] + f["Close"]) / 3
    sma_tp = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    f["cci"] = (tp - sma_tp) / (0.015 * md + 1e-9)

    # ── 4. Volatility ────────────────────────────────────────────────────────
    f["atr"]      = tr.rolling(14).mean()
    std_20        = f["Close"].rolling(20).std()
    f["bb_upper"] = f["sma_20"] + 2 * std_20
    f["bb_lower"] = f["sma_20"] - 2 * std_20
    f["bb_width"] = (f["bb_upper"] - f["bb_lower"]) / (f["sma_20"] + 1e-9)
    f["pct_b"]    = (f["Close"] - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"] + 1e-9)
    f["hist_vol_10"] = f["log_return"].rolling(10).std() * np.sqrt(252)
    f["hist_vol_20"] = f["log_return"].rolling(20).std() * np.sqrt(252)
    f["vol_ratio"]   = f["hist_vol_10"] / (f["hist_vol_20"] + 1e-9)  # vol acceleration

    # ── 5. Volume ────────────────────────────────────────────────────────────
    if "Volume" in f.columns:
        vol_sma = f["Volume"].rolling(20).mean()
        f["volume_ratio"] = f["Volume"] / (vol_sma + 1e-9)
        dir_ = np.sign(f["Close"].diff())
        f["obv"] = (dir_ * f["Volume"]).cumsum()
        f["obv_ma"] = f["obv"].rolling(20).mean()
        f["obv_trend"] = (f["obv"] > f["obv_ma"]).astype(int)
        hl = f["hl_range"] + 1e-9
        mfm = ((f["Close"] - f["Low"]) - (f["High"] - f["Close"])) / hl
        f["cmf"] = (mfm * f["Volume"]).rolling(20).sum() / (f["Volume"].rolling(20).sum() + 1e-9)
    else:
        f["volume_ratio"] = 1.0
        f["obv_trend"]    = 0
        f["cmf"]          = 0

    # ── 6. India VIX features (your secret weapon) ──────────────────────────
    if "VIX" in f.columns:
        f["vix_change"]     = f["VIX"].diff()
        f["vix_pct_change"] = f["VIX"].pct_change()
        f["vix_sma_5"]      = f["VIX"].rolling(5).mean()
        f["vix_sma_10"]     = f["VIX"].rolling(10).mean()
        f["vix_sma_20"]     = f["VIX"].rolling(20).mean()
        f["vix_momentum"]   = f["VIX"] - f["VIX"].shift(5)
        f["vix_spike"]      = (f["VIX"] > 20).astype(int)
        f["vix_extreme"]    = (f["VIX"] > 25).astype(int)
        f["vix_low"]        = (f["VIX"] < 15).astype(int)
        f["vix_trend"]      = (f["VIX"] > f["vix_sma_10"]).astype(int)
        f["vix_vs_sma20"]   = (f["VIX"] - f["vix_sma_20"]) / (f["vix_sma_20"] + 1e-9)
        f["vol_regime"]     = pd.cut(
            f["VIX"], bins=[0, 12, 15, 20, 25, 100],
            labels=[0, 1, 2, 3, 4]).astype(float)
        # Nifty-VIX correlation (rolling 20)
        f["vix_nifty_corr"]  = f["VIX"].rolling(20).corr(f["daily_return"])
        # VIX reversal signal: spike then drop = buy signal
        f["vix_reversal"]    = ((f["VIX"].shift(1) > 20) & (f["vix_change"] < -1)).astype(int)

    # ── 7. PCR features ──────────────────────────────────────────────────────
    if "PCR" in f.columns:
        f["pcr_sma_5"]    = f["PCR"].rolling(5).mean()
        f["pcr_high"]     = (f["PCR"] > 1.2).astype(int)   # Bullish contrarian
        f["pcr_low"]      = (f["PCR"] < 0.8).astype(int)   # Bearish contrarian
        f["pcr_momentum"] = f["PCR"] - f["PCR"].shift(5)

    # ── 8. FII/DII flow features ─────────────────────────────────────────────
    if "FII_Net" in f.columns:
        f["fii_positive"]   = (f["FII_Net"] > 0).astype(int)
        f["fii_sma_5"]      = f["FII_Net"].rolling(5).mean()
        f["fii_momentum"]   = f["FII_Net"] - f["FII_Net"].shift(5)

    # ── 9. Global market context ─────────────────────────────────────────────
    if "SP500_Return" in f.columns:
        f["sp500_positive"]   = (f["SP500_Return"] > 0).astype(int)
        f["sp500_ma_cross"]   = f["SP500_Return"].rolling(5).mean()

    if "BankNifty_Return" in f.columns:
        f["bnf_nifty_spread"] = f["BankNifty_Return"] - f["daily_return"]
        f["bnf_leading"]      = f["BankNifty_Return"].shift(1)  # BNF leads Nifty?

    # ── 10. VIX term structure ───────────────────────────────────────────────
    if "VIX_Term_Spread" in f.columns:
        f["contango"]     = (f["VIX_Term_Spread"] > 0).astype(int)
        f["term_sma_5"]   = f["VIX_Term_Spread"].rolling(5).mean()

    # ── 11. Calendar / time features ────────────────────────────────────────
    f["day_of_week"]    = f.index.dayofweek        # 0=Mon, 4=Fri
    f["month"]          = f.index.month
    f["is_monday"]      = (f.index.dayofweek == 0).astype(int)
    f["is_friday"]      = (f.index.dayofweek == 4).astype(int)
    f["is_month_end"]   = (f.index.is_month_end).astype(int)
    f["is_month_start"] = (f.index.is_month_start).astype(int)
    f["quarter"]        = f.index.quarter

    # ── 12. Lag features (past 1-5 days) ────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        f[f"return_lag_{lag}"] = f["daily_return"].shift(lag)
    for lag in [1, 2]:
        if "VIX" in f.columns:
            f[f"vix_lag_{lag}"] = f["VIX"].shift(lag)

    # ── Target: 1 if next day Close > today Close ────────────────────────────
    f["target"] = (f["Close"].shift(-1) > f["Close"]).astype(int)

    if dropna:
        return f.dropna()
    else:
        # In live mode, ffill to ensure we have values for the latest row
        return f.ffill()


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excluding target and price cols)."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "VIX",
               "PCR", "FII_Net", "SP500_Return", "BankNifty_Return",
               "VIX_Term_Spread", "target"}
    return [c for c in df.columns if c not in exclude]


def get_feature_groups() -> dict:
    """Feature group descriptions for the UI."""
    return {
        "Price Transforms (10)": [
            "daily_return", "log_return", "hl_range", "oc_range",
            "body_size", "upper_shadow", "lower_shadow", "true_range",
            "gap_up", "gap_pct"
        ],
        "Moving Averages (9)": [
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_9", "ema_12", "ema_26",
            "price_vs_sma20", "price_vs_sma50",
            "sma5_vs_sma20", "sma20_vs_sma50",
            "golden_cross", "death_cross"
        ],
        "Momentum (13)": [
            "rsi", "rsi_overbought", "rsi_oversold",
            "macd", "macd_signal", "macd_hist", "macd_cross_up", "macd_cross_down",
            "roc_5", "roc_10", "roc_20",
            "stoch_k", "stoch_d", "stoch_cross",
            "williams_r", "cci"
        ],
        "Volatility (7)": [
            "atr", "bb_upper", "bb_lower", "bb_width", "pct_b",
            "hist_vol_10", "hist_vol_20", "vol_ratio"
        ],
        "Volume (5)": [
            "volume_ratio", "obv_trend", "cmf"
        ],
        "India VIX (14) ⭐": [
            "vix_change", "vix_pct_change", "vix_sma_5", "vix_sma_10",
            "vix_momentum", "vix_spike", "vix_extreme", "vix_low",
            "vix_trend", "vix_vs_sma20", "vol_regime",
            "vix_nifty_corr", "vix_reversal"
        ],
        "PCR (4)": [
            "pcr_sma_5", "pcr_high", "pcr_low", "pcr_momentum"
        ],
        "FII/DII Flow (3)": [
            "fii_positive", "fii_sma_5", "fii_momentum"
        ],
        "Global Context (3)": [
            "sp500_positive", "sp500_ma_cross", "bnf_nifty_spread", "bnf_leading"
        ],
        "Calendar (7)": [
            "day_of_week", "month", "is_monday", "is_friday",
            "is_month_end", "is_month_start", "quarter"
        ],
        "Lag Features (7)": [
            "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
            "vix_lag_1", "vix_lag_2"
        ],
    }
