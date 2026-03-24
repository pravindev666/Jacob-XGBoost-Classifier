"""
scripts/fetch_and_append.py
----------------------------
Fetches the latest market data for all 10 CSV files.
Only appends NEW rows — never duplicates, never overwrites history.

Data sources:
  - Nifty 50, Bank Nifty, Nifty 15min  → yfinance (^NSEI, ^NSEBANK)
  - India VIX daily + 15min             → yfinance (^INDIAVIX)
  - S&P 500                             → yfinance (^GSPC)
  - FII/DII                             → NSE India website (niftyindices.com)
  - PCR (Put-Call Ratio)                → NSE website
  - VIX Term Structure                  → NSE website
  - Events                              → NSE holiday calendar
"""

import os
import sys
import pytz
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
IST           = pytz.timezone("Asia/Kolkata")
TODAY_IST     = datetime.now(IST).date()
FORCE_REFRESH = os.environ.get("FORCE_FULL_REFRESH", "no").lower() == "yes"

# Fetch window: last 7 days (catches weekend gaps + any missed days)
FETCH_DAYS    = 7 if not FORCE_REFRESH else 365 * 5   # 5 years if force refresh
START_DATE    = TODAY_IST - timedelta(days=FETCH_DAYS)

print(f"=== Data Update: {TODAY_IST} IST ===")
print(f"Fetch window: {START_DATE} → {TODAY_IST}")
print(f"Force refresh: {FORCE_REFRESH}")
print()


# ── Helper functions ─────────────────────────────────────────────────────────

def load_existing(filename: str) -> pd.DataFrame:
    """Load existing CSV, return empty DataFrame if file doesn't exist."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  [NEW FILE] {filename} will be created")
        return pd.DataFrame()
    df = pd.read_csv(path)
    
    # Standardise OHLCV column casing to match yfinance output
    renames = {}
    for c in df.columns:
        if c.lower() in ["date", "open", "high", "low", "close", "volume"]:
            renames[c] = c.strip().title()
        elif c.lower() == "vix":
            renames[c] = "VIX"
    df.rename(columns=renames, inplace=True)
    return df


def get_last_date(df: pd.DataFrame, date_col: str = "Date") -> date | None:
    """Return the most recent date in the existing CSV."""
    if df.empty or date_col not in df.columns:
        return None
    try:
        return pd.to_datetime(df[date_col]).max().date()
    except Exception:
        return None


def append_new_rows(existing: pd.DataFrame, new_data: pd.DataFrame,
                    date_col: str = "Date", filename: str = "") -> pd.DataFrame:
    """
    Append only rows that are newer than the last date in existing data.
    Deduplicates on date_col to prevent any double-counting.
    """
    if new_data.empty:
        print(f"  [SKIP] {filename} — no new data fetched")
        return existing

    if existing.empty:
        result = new_data.copy()
        print(f"  [INIT] {filename} — {len(result)} rows written (first time)")
        return result

    # Standardise date columns to proper datetime.date objects to avoid pandas TypeError string comparisons
    if date_col in existing.columns:
        existing[date_col] = pd.to_datetime(existing[date_col]).dt.date
    if date_col in new_data.columns:
        new_data[date_col] = pd.to_datetime(new_data[date_col]).dt.date

    last_date = get_last_date(existing, date_col)
    if last_date is None:
        return pd.concat([existing, new_data], ignore_index=True)

    # Only keep rows strictly newer than the last existing date
    new_data[date_col] = pd.to_datetime(new_data[date_col]).dt.date
    mask = new_data[date_col] > last_date
    truly_new = new_data[mask].copy()

    if truly_new.empty:
        print(f"  [UP TO DATE] {filename} — already current (last: {last_date})")
        return existing

    result = pd.concat([existing, truly_new], ignore_index=True)
    # Final dedup just in case
    result = result.drop_duplicates(subset=[date_col], keep="last")
    result = result.sort_values(date_col).reset_index(drop=True)
    print(f"  [UPDATED] {filename} — +{len(truly_new)} rows (last was {last_date}, now {truly_new[date_col].max()})")
    return result


def save_csv(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV in data/ folder."""
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)


def fetch_yfinance(ticker: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from yfinance for given ticker and interval."""
    try:
        import yfinance as yf
        period = f"{FETCH_DAYS}d" if not FORCE_REFRESH else "max"
        df = yf.download(
            ticker,
            start=START_DATE if not FORCE_REFRESH else None,
            period=period if FORCE_REFRESH else None,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # Flatten MultiIndex columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
        return df
    except Exception as e:
        print(f"    yfinance error for {ticker} ({interval}): {e}")
        return pd.DataFrame()


# ── 1. Nifty 50 Daily ────────────────────────────────────────────────────────
print("1. Nifty 50 Daily (nifty_daily.csv)")
existing = load_existing("nifty_daily.csv")
raw = fetch_yfinance("^NSEI", "1d")
if not raw.empty:
    # Standardise column names
    date_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw["Date"] = pd.to_datetime(raw[date_col]).dt.date
    new_df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    new_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    new_df = new_df.dropna(subset=["Close"])
    result = append_new_rows(existing, new_df, "Date", "nifty_daily.csv")
    save_csv(result, "nifty_daily.csv")
print()


# ── 2. Nifty 50 15-Minute ────────────────────────────────────────────────────
print("2. Nifty 50 15-min (nifty_15m_2001_to_now.csv)")
existing = load_existing("nifty_15m_2001_to_now.csv")
# yfinance only gives 60 days for 15m — so we fetch 60 days max
raw_15m = fetch_yfinance("^NSEI", "15m")
if not raw_15m.empty:
    dt_col = "Datetime" if "Datetime" in raw_15m.columns else "Date"
    raw_15m["Date"] = pd.to_datetime(raw_15m[dt_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    new_df = raw_15m[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    new_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    new_df = new_df.dropna(subset=["Close"])
    result = append_new_rows(existing, new_df, "Date", "nifty_15m_2001_to_now.csv")
    save_csv(result, "nifty_15m_2001_to_now.csv")
print()


# ── 3. India VIX Daily ───────────────────────────────────────────────────────
print("3. India VIX Daily (vix_daily.csv)")
existing = load_existing("vix_daily.csv")
raw = fetch_yfinance("^INDIAVIX", "1d")
if not raw.empty:
    dt_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw["Date"] = pd.to_datetime(raw[dt_col]).dt.date
    # VIX close is the actual VIX value
    new_df = raw[["Date", "Open", "High", "Low", "Close"]].copy()
    new_df.columns = ["Date", "Open", "High", "Low", "Close"]
    new_df = new_df.dropna(subset=["Close"])
    result = append_new_rows(existing, new_df, "Date", "vix_daily.csv")
    save_csv(result, "vix_daily.csv")
print()


# ── 4. India VIX 15-Minute ───────────────────────────────────────────────────
print("4. India VIX 15-min (INDIAVIX_15minute_2001_now.csv)")
existing = load_existing("INDIAVIX_15minute_2001_now.csv")
raw_vix15 = fetch_yfinance("^INDIAVIX", "15m")
if not raw_vix15.empty:
    dt_col = "Datetime" if "Datetime" in raw_vix15.columns else "Date"
    raw_vix15["Date"] = pd.to_datetime(raw_vix15[dt_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    new_df = raw_vix15[["Date", "Open", "High", "Low", "Close"]].copy()
    new_df.columns = ["Date", "Open", "High", "Low", "Close"]
    new_df = new_df.dropna(subset=["Close"])
    result = append_new_rows(existing, new_df, "Date", "INDIAVIX_15minute_2001_now.csv")
    save_csv(result, "INDIAVIX_15minute_2001_now.csv")
print()


# ── 5. Bank Nifty Daily ──────────────────────────────────────────────────────
print("5. Bank Nifty Daily (bank_nifty_daily.csv)")
existing = load_existing("bank_nifty_daily.csv")
raw = fetch_yfinance("^NSEBANK", "1d")
if not raw.empty:
    dt_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw["Date"] = pd.to_datetime(raw[dt_col]).dt.date
    new_df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    new_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    new_df = new_df.dropna(subset=["Close"])
    result = append_new_rows(existing, new_df, "Date", "bank_nifty_daily.csv")
    save_csv(result, "bank_nifty_daily.csv")
print()


# ── 6. S&P 500 Daily ─────────────────────────────────────────────────────────
print("6. S&P 500 Daily (sp500_daily.csv)")
existing = load_existing("sp500_daily.csv")
raw = fetch_yfinance("^GSPC", "1d")
if not raw.empty:
    dt_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw["Date"] = pd.to_datetime(raw[dt_col]).dt.date
    new_df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    new_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    new_df = new_df.dropna(subset=["Close"])
    result = append_new_rows(existing, new_df, "Date", "sp500_daily.csv")
    save_csv(result, "sp500_daily.csv")
print()


# ── 7. FII/DII Daily ─────────────────────────────────────────────────────────
print("7. FII/DII Daily (fii_dii_daily.csv)")
existing = load_existing("fii_dii_daily.csv")

def fetch_fii_dii() -> pd.DataFrame:
    """Fetch FII/DII data from NSE India."""
    rows = []
    # NSE provides this in their API
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    session = requests.Session()
    # First hit the main page to get cookies
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
    except Exception:
        pass

    try:
        # Try NSE participant data endpoint
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        resp = session.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for item in data:
                try:
                    row = {
                        "Date": pd.to_datetime(item.get("date", "")).date(),
                        "FII_Net_Buy": float(str(item.get("fii_net_buy", 0)).replace(",", "") or 0),
                        "DII_Net_Buy": float(str(item.get("dii_net_buy", 0)).replace(",", "") or 0),
                    }
                    rows.append(row)
                except Exception:
                    continue
    except Exception as e:
        print(f"    NSE FII/DII API error: {e}")
        # Fallback: jugaad-data
        try:
            from jugaad_data.nse import NSELive
            nse = NSELive()
            fii_data = nse.get_fii_dii_data()
            if fii_data:
                for item in fii_data:
                    rows.append({
                        "Date": pd.to_datetime(item.get("date", "")).date(),
                        "FII_Net_Buy": float(str(item.get("fiiNetBuyValue", 0)).replace(",", "") or 0),
                        "DII_Net_Buy": float(str(item.get("diiNetBuyValue", 0)).replace(",", "") or 0),
                    })
        except Exception as e2:
            print(f"    jugaad-data fallback also failed: {e2}")

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Date"])
    return df

new_fii = fetch_fii_dii()
if not new_fii.empty:
    result = append_new_rows(existing, new_fii, "Date", "fii_dii_daily.csv")
    save_csv(result, "fii_dii_daily.csv")
else:
    print("  [SKIP] fii_dii_daily.csv — could not fetch (NSE API may be rate-limited)")
print()


# ── 8. PCR (Put-Call Ratio) Daily ────────────────────────────────────────────
print("8. PCR Daily (pcr_daily.csv)")
existing = load_existing("pcr_daily.csv")

def fetch_pcr() -> pd.DataFrame:
    """Fetch Put-Call Ratio from NSE India options data."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
    except Exception:
        pass

    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        resp = session.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            # Calculate PCR from OI data
            records = data.get("records", {}).get("data", [])
            total_put_oi  = sum(r.get("PE", {}).get("openInterest", 0) for r in records if r.get("PE"))
            total_call_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in records if r.get("CE"))
            if total_call_oi > 0:
                pcr = round(total_put_oi / total_call_oi, 4)
                today_date = datetime.now(IST).date()
                return pd.DataFrame([{"Date": today_date, "PCR": pcr,
                                      "Put_OI": total_put_oi, "Call_OI": total_call_oi}])
    except Exception as e:
        print(f"    NSE PCR API error: {e}")
    return pd.DataFrame()

new_pcr = fetch_pcr()
if not new_pcr.empty:
    result = append_new_rows(existing, new_pcr, "Date", "pcr_daily.csv")
    save_csv(result, "pcr_daily.csv")
else:
    print("  [SKIP] pcr_daily.csv — NSE API unavailable")
print()


# ── 9. VIX Term Structure Daily ──────────────────────────────────────────────
print("9. VIX Term Structure (vix_term_daily.csv)")
existing = load_existing("vix_term_daily.csv")

def fetch_vix_term() -> pd.DataFrame:
    """
    Approximate VIX term structure using near and far month VIX.
    Near month = current ^INDIAVIX
    Far month = approximated from next month VIX futures (if available)
    or estimated as VIX * 1.02 (2% contango assumption) when not available.
    """
    try:
        import yfinance as yf
        # Near month = current VIX
        near = yf.download("^INDIAVIX", period="5d", interval="1d",
                           progress=False, auto_adjust=True)
        if near.empty:
            return pd.DataFrame()

        near_close = float(near["Close"].iloc[-1])
        today_date = datetime.now(IST).date()

        # Try to get far month VIX proxy (VIX1D or use 30-day estimate)
        # For NSE, a reasonable proxy is VIX + 2-4% (typical contango)
        # Replace with actual futures data if you have a broker API
        far_close = near_close * 1.025  # 2.5% contango assumption

        return pd.DataFrame([{
            "Date": today_date,
            "VIX_Near": round(near_close, 2),
            "VIX_Far":  round(far_close, 2),
            "Term_Spread": round(far_close - near_close, 2),
        }])
    except Exception as e:
        print(f"    VIX term error: {e}")
        return pd.DataFrame()

new_vterm = fetch_vix_term()
if not new_vterm.empty:
    result = append_new_rows(existing, new_vterm, "Date", "vix_term_daily.csv")
    save_csv(result, "vix_term_daily.csv")
print()


# ── 10. Events / NSE Calendar ────────────────────────────────────────────────
print("10. Events Calendar (events.csv)")
existing = load_existing("events.csv")

def fetch_nse_holidays() -> pd.DataFrame:
    """Fetch NSE trading holidays and expiry dates for the year."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
    except Exception:
        pass

    rows = []
    try:
        url = "https://www.nseindia.com/api/holiday-master?type=trading"
        resp = session.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for segment, holidays in data.items():
                if isinstance(holidays, list):
                    for h in holidays:
                        try:
                            rows.append({
                                "Date": pd.to_datetime(h.get("tradingDate", "")).date(),
                                "Event": h.get("description", "Holiday"),
                                "Type": "Holiday",
                            })
                        except Exception:
                            continue
    except Exception as e:
        print(f"    NSE holiday API error: {e}")

    # Add monthly expiry dates (last Thursday of each month)
    # NSE Nifty monthly options expire on last Thursday
    year = TODAY_IST.year
    for month in range(1, 13):
        # Find last Thursday of the month
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        for day in range(last_day, last_day - 7, -1):
            d = date(year, month, day)
            if d.weekday() == 3:  # Thursday
                rows.append({
                    "Date": d,
                    "Event": f"Monthly Expiry {d.strftime('%b %Y')}",
                    "Type": "Monthly_Expiry",
                })
                break

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    return df

new_events = fetch_nse_holidays()
if not new_events.empty:
    # For events, we merge by date+type (not just date)
    if not existing.empty:
        combined = pd.concat([existing, new_events], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Type"], keep="last")
        combined = combined.sort_values("Date").reset_index(drop=True)
        save_csv(combined, "events.csv")
        added = len(combined) - len(existing)
        print(f"  [UPDATED] events.csv — +{max(0, added)} entries")
    else:
        save_csv(new_events, "events.csv")
        print(f"  [INIT] events.csv — {len(new_events)} entries written")
else:
    print("  [SKIP] events.csv — NSE API unavailable")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=== Update complete ===")
print(f"Run at: {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}")
