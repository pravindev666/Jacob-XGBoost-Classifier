import sys
import os
import pandas as pd
import traceback

sys.path.append(os.getcwd())

try:
    from utils.data_loader import build_master_dataset
    from utils.features import engineer_all_features
    from pages.dashboard import _get_live_indicators
    
    print("Building data...")
    master = build_master_dataset()
    print(f"Master rows: {len(master)}")
    
    featured = engineer_all_features(master, dropna=False)
    print(f"Featured rows: {len(featured)}")
    
    if len(featured) == 0:
        print("Featured is empty! Let's check engineer_all_features logic")
        
        # Manually run the engineering steps without dropna
        f = master.copy()
        print(f"Initial f rows: {len(f)}")
        
        import utils.features as feat
        f_all = feat.engineer_all_features.__code__ 
        
        # Actually let's just modify feats inplace locally or read from feat
        # I'll just look at what's in master that might fail
        pass
    else:
        print("\nExtracting indicators...")
        indicators = _get_live_indicators(featured)
        print("\nResult:")
        print(indicators)
        last = featured.iloc[-1]
        print(f"\nNaNs in last row: {last.isna().sum()}")
    print("\nTarget columns:")
    for col in ["rsi", "macd_hist", "stoch_k", "pct_b", "atr", "volume_ratio"]:
        if col in featured.columns:
            print(f"{col}: {last[col]} (isna: {pd.isna(last[col])})")
        else:
            print(f"{col}: MISSING")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
