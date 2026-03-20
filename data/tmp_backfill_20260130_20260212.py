#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
One-off backfill script:
- Date range: 2025-09-01 .. 2025-11-19 (trading-day END labels, ET)
- Fill:
  1) Technical indicators (SMA/EMA/RSI/MACD/ATR/BB/...)
  2) total_addresses / million_dollar_addresses / ratio  (CoinMetrics)
  3) sentiment_score -> filled with 0.0 for history (cannot be reconstructed)
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd

from coinbase_ohlcv.features import compute_features_inplace
from coinbase_ohlcv.santiment_millionaire import fetch_millionaire_metrics_daily
from coinbase_ohlcv.update import (
    _shift_ymd_et,
    _parse_ymd_et,
    _ensure_cols,
)

# -----------------------
# CONFIG
# -----------------------

DATA_DIR = Path("coinbase-ohlcv/data")  # 按你真实路径调整
FILES = {
    "BTC": DATA_DIR / "BTC_aggregated.csv",
    "ETH": DATA_DIR / "ETH_aggregated.csv",
    "XRP": DATA_DIR / "XRP_aggregated.csv",
}

START_ET = "2025-09-01"
END_ET   = "2025-11-19"
LAG_DAYS = 1

SANTIMENT_SLUG = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "XRP": "xrp",
}

ADDR_COLS = ["total_addresses", "million_dollar_addresses", "ratio"]

# -----------------------
# HELPERS
# -----------------------

def trading_days_between(start_et: str, end_et: str):
    out = []
    cur = start_et
    while cur <= end_et:
        out.append(cur)
        cur = _shift_ymd_et(cur, 1)
    return out


def backfill_address_metrics(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    df["Date"] = df["Date"].astype(str)
    df = _ensure_cols(df, ADDR_COLS)

    days = trading_days_between(START_ET, END_ET)

    # 只挑缺失的日期
    miss_days = df.loc[
        (df["Date"].isin(days)) & df[ADDR_COLS].isna().any(axis=1),
        "Date"
    ].tolist()

    if not miss_days:
        print(f"[{asset}] address metrics: nothing to backfill")
        return df

    slug = SANTIMENT_SLUG.get(asset)
    if not slug:
        print(f"[{asset}] no slug, skip")
        return df

    # trading-day END → UTC day (lagged)
    utc_days = []
    td_to_utc = {}
    for td in miss_days:
        d_et = _parse_ymd_et(td) - timedelta(days=LAG_DAYS)
        u = d_et.astimezone(timezone.utc).strftime("%Y-%m-%d")
        td_to_utc[td] = u
        utc_days.append(u)

    min_utc = min(utc_days)
    max_utc = max(utc_days)

    from_dt = f"{min_utc}T00:00:00Z"
    to_dt = (
        datetime.strptime(max_utc, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc) + timedelta(days=1)
    ).strftime("%Y-%m-%dT00:00:00Z")

    print(f"[{asset}] fetch CoinMetrics {from_dt} -> {to_dt}")

    s_df = fetch_millionaire_metrics_daily(
        slug=slug,
        from_dt=from_dt,
        to_dt=to_dt,
    )

    if s_df is None or len(s_df) == 0:
        print(f"[{asset}] CoinMetrics empty")
        return df

    s_df = s_df[["Date"] + ADDR_COLS].copy()
    s_df["Date"] = s_df["Date"].astype(str)
    s_map = s_df.set_index("Date")[ADDR_COLS].to_dict("index")

    for td, utc in td_to_utc.items():
        row = s_map.get(utc)
        if not row:
            continue
        mask = df["Date"] == td
        for c in ADDR_COLS:
            if mask.any() and pd.isna(df.loc[mask, c]).any():
                df.loc[mask, c] = row[c]

    return df


# -----------------------
# MAIN
# -----------------------

def main():
    for asset, path in FILES.items():
        print(f"\n===== {asset} =====")
        if not path.exists():
            print(f"File not found: {path}")
            continue

        df = pd.read_csv(path)
        df["Date"] = df["Date"].astype(str)

        # 1) sentiment_score：历史补 0.0
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0
        else:
            mask = (df["Date"] >= START_ET) & (df["Date"] <= END_ET)
            df.loc[mask, "sentiment_score"] = df.loc[mask, "sentiment_score"].fillna(0.0)

        # 2) address metrics
        df = backfill_address_metrics(df, asset)

        # 保存一次（features 会直接 inplace 改这个文件）
        df.to_csv(path, index=False)

        # 3) technical indicators
        print(f"[{asset}] computing technical indicators")
        compute_features_inplace(
            path,
            overwrite_last_n_days=200,  # 覆盖范围 > 2025-09-01
        )

        print(f"[{asset}] DONE")

    print("\nAll done.")


if __name__ == "__main__":
    main()
