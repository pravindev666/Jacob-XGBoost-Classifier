import os
import pandas as pd
import numpy as np
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _read(filename: str, date_col: str = None) -> pd.DataFrame | None:
    """Read a CSV from the data/ folder, return None if missing."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # Auto-detect date column
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")
            df = df.set_index(date_col).sort_index()
        else:
            # Try common date column names
            for col in ["Date", "date", "Datetime", "datetime", "timestamp", "TIME"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], dayfirst=False, errors="coerce")
                    # Remove rows with NaT index to avoid Cartesian join blowups
                    df = df.dropna(subset=[col])
                    df = df.set_index(col).sort_index()
                    # Remove duplicate indices
                    df = df[~df.index.duplicated(keep='last')]
                    break
        return df
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_nifty_daily() -> pd.DataFrame:
    """Load daily Nifty OHLCV. Standardise column names."""
    df = _read("nifty_daily.csv")
    if df is None:
        return pd.DataFrame()
    # Normalise column names to Open/High/Low/Close/Volume
    df.columns = [c.strip().title() for c in df.columns]
    for alias, target in [("Adj Close", "Close"), ("Ltp", "Close"),
                           ("Vol", "Volume"), ("Turnover", "Volume")]:
        if alias in df.columns and target not in df.columns:
            df[target] = df[alias]
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=300)
def load_nifty_15m() -> pd.DataFrame:
    """Load 15-minute Nifty data."""
    df = _read("nifty_15m_2001_to_now.csv")
    if df is None:
        return None
    df.columns = [c.strip().title() for c in df.columns]
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=300)
def load_vix_daily() -> pd.DataFrame:
    """Load daily India VIX."""
    df = _read("vix_daily.csv")
    if df is None:
        return pd.DataFrame()
    df.columns = [c.strip().title() for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    # Standardise VIX close column
    for alias in ["Close", "Vix Close", "India Vix", "Vix"]:
        if alias in df.columns:
            df["VIX"] = df[alias]
            break
    if "VIX" not in df.columns:
        df["VIX"] = df.iloc[:, 0]
    return df[["VIX"]].dropna()


@st.cache_data(ttl=300)
def load_pcr() -> pd.DataFrame:
    """Load Put-Call Ratio."""
    df = _read("pcr_daily.csv")
    if df is None:
        return None
    df.columns = [c.strip().title() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    # Standardise
    for alias in ["Pcr", "Put Call Ratio", "Pcr Oi", "Total Pcr"]:
        if alias in df.columns:
            df["PCR"] = df[alias]
            break
    if "PCR" not in df.columns and len(df.columns) > 0:
        df["PCR"] = df.iloc[:, 0]
    return df[["PCR"]].dropna() if "PCR" in df.columns else None


@st.cache_data(ttl=300)
def load_fii_dii() -> pd.DataFrame:
    """Load FII/DII institutional flow data."""
    df = _read("fii_dii_daily.csv")
    if df is None:
        return None
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    # Normalize: ensure 'fii_net' column exists
    for alias in ["fii_net", "fii", "net_fii", "fii_net_buy"]:
        if alias in df.columns:
            df["fii_net"] = df[alias]
            break
    if "fii_net" not in df.columns and len(df.columns) > 0:
        df["fii_net"] = df.iloc[:, 0]
    return df.dropna()


@st.cache_data(ttl=300)
def load_vix_term() -> pd.DataFrame:
    """Load VIX term structure."""
    df = _read("vix_term_daily.csv")
    if df is None:
        return None
    df.columns = [c.strip().title() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna()


@st.cache_data(ttl=300)
def load_bank_nifty() -> pd.DataFrame:
    """Load Bank Nifty daily OHLCV."""
    df = _read("bank_nifty_daily.csv")
    if df is None:
        return None
    df.columns = [c.strip().title() for c in df.columns]
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=300)
def load_sp500() -> pd.DataFrame:
    """Load S&P 500 daily — used as global market context feature."""
    df = _read("sp500_daily.csv")
    if df is None:
        return None
    df.columns = [c.strip().title() for c in df.columns]
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=300)
def load_events() -> pd.DataFrame:
    """Load market events / expiry calendar."""
    df = _read("events.csv")
    if df is None:
        return None
    return df


@st.cache_data(ttl=300)
def build_master_dataset() -> pd.DataFrame:
    """
    Merge all daily datasets into one master DataFrame
    ready for feature engineering and ML training.
    """
    nifty = load_nifty_daily()
    vix = load_vix_daily()

    # Start with Nifty
    master = nifty[["Open", "High", "Low", "Close"]].copy()
    if "Volume" in nifty.columns:
        master["Volume"] = nifty["Volume"]

    # Merge VIX
    if vix is not None:
        master = master.join(vix[["VIX"]], how="left")
        master["VIX"] = master["VIX"].fillna(method="ffill")

    # Merge PCR
    pcr = load_pcr()
    if pcr is not None:
        master = master.join(pcr[["PCR"]], how="left")
        master["PCR"] = master["PCR"].fillna(method="ffill")

    # Merge FII/DII net flow (normalized to 'fii_net' by load_fii_dii)
    fii = load_fii_dii()
    if fii is not None and "fii_net" in fii.columns:
        master = master.join(fii[["fii_net"]].rename(columns={"fii_net": "FII_Net"}), how="left")

    # Merge Bank Nifty return as a feature
    bnf = load_bank_nifty()
    if bnf is not None:
        bnf_ret = bnf["Close"].pct_change().rename("BankNifty_Return")
        master = master.join(bnf_ret, how="left")

    # Merge SP500 return as a global feature
    sp = load_sp500()
    if sp is not None:
        sp_ret = sp["Close"].pct_change().rename("SP500_Return")
        master = master.join(sp_ret, how="left")

    # Merge VIX term spread
    vterm = load_vix_term()
    if vterm is not None and len(vterm.columns) >= 2:
        # Term spread = far - near
        cols = vterm.columns.tolist()
        vterm["VIX_Term_Spread"] = vterm[cols[-1]] - vterm[cols[0]]
        master = master.join(vterm[["VIX_Term_Spread"]], how="left")

    master = master.sort_index()
    return master


def get_data_status() -> dict:
    """Return which files are present and which are missing."""
    files = {
        "nifty_daily.csv": "Nifty Daily OHLCV ⭐ Core",
        "nifty_15m_2001_to_now.csv": "Nifty 15-min OHLCV",
        "vix_daily.csv": "India VIX Daily ⭐ Core",
        "INDIAVIX_15minute_2001_now.csv": "India VIX 15-min",
        "bank_nifty_daily.csv": "Bank Nifty Daily",
        "sp500_daily.csv": "S&P 500 Daily (global context)",
        "fii_dii_daily.csv": "FII/DII Institutional Flow",
        "events.csv": "Market Events / Calendar",
        "pcr_daily.csv": "Put-Call Ratio Daily",
        "vix_term_daily.csv": "VIX Term Structure",
    }
    status = {}
    for fname, label in files.items():
        path = os.path.join(DATA_DIR, fname)
        exists = os.path.exists(path)
        rows = "—"
        if exists:
            try:
                df_tmp = pd.read_csv(path, nrows=10) # Quick check
                # For actual row count, we'd need to read the whole file, but 
                # let's just use approximate or skip for speed.
                # Actually, let's just do it for small files.
                if os.path.getsize(path) < 1024*1024: # < 1MB
                    rows = f"{len(pd.read_csv(path))} rows"
                else:
                    rows = "Large file"
            except:
                rows = "Error"
        
        status[fname] = {
            "label": label,
            "present": exists,
            "size": f"{os.path.getsize(path)/1024:.0f} KB" if exists else "—",
            "rows": rows
        }
    return status
