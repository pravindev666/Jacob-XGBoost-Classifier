"""
scripts/show_row_counts.py
--------------------------
Prints row counts and latest date for each CSV file.
Called by GitHub Actions to show what changed.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FILES = [
    ("nifty_daily.csv",                   "Date",  "Nifty 50 Daily"),
    ("nifty_15m_2001_to_now.csv",          "Date",  "Nifty 50 15-min"),
    ("vix_daily.csv",                      "Date",  "India VIX Daily"),
    ("INDIAVIX_15minute_2001_now.csv",     "Date",  "India VIX 15-min"),
    ("bank_nifty_daily.csv",               "Date",  "Bank Nifty Daily"),
    ("sp500_daily.csv",                    "Date",  "S&P 500 Daily"),
    ("fii_dii_daily.csv",                  "Date",  "FII/DII Daily"),
    ("pcr_daily.csv",                      "Date",  "PCR Daily"),
    ("vix_term_daily.csv",                 "Date",  "VIX Term Structure"),
    ("events.csv",                         "Date",  "Events Calendar"),
]

print(f"{'File':<42} {'Rows':>7}  {'Latest Date':<14}")
print("-" * 68)

for filename, date_col, label in FILES:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  {label:<40} {'MISSING':>7}  {'—':<14}")
        continue
    try:
        df = pd.read_csv(path)
        rows = len(df)
        if date_col in df.columns:
            latest = pd.to_datetime(df[date_col]).max().strftime("%Y-%m-%d")
        else:
            latest = "unknown"
        print(f"  {label:<40} {rows:>7,}  {latest:<14}")
    except Exception as e:
        print(f"  {label:<40} {'ERROR':>7}  {str(e)[:20]}")
