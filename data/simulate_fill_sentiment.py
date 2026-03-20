#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate-fill sentiment_score zeros by sampling from the empirical distribution
of non-zero sentiment_score values (bootstrap).

What it does:
- For each input CSV:
  - Find rows where sentiment_score == 0 (optionally include NaN)
  - Replace those values with random draws from the non-zero sentiment_score pool
  - Add a flag column: sentiment_score_is_simulated (1 if replaced else 0)
  - Write out to a new file (or in-place if --inplace)

Why bootstrap:
- Keeps the observed distribution shape (skew, fat tails), no parametric assumption.

WARNING:
- This creates synthetic values. Keep the flag column, and avoid mixing with ground-truth.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def fill_one_file(
    path: Path,
    seed: int,
    include_nan: bool,
    date_col: str,
    start: str | None,
    end: str | None,
    inplace: bool,
    output_suffix: str,
    min_pool: int,
    pool_window_days: int | None,
):
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"{path}: missing date column '{date_col}'")

    if "sentiment_score" not in df.columns:
        raise ValueError(f"{path}: missing column 'sentiment_score'")

    # Normalize date as string 'YYYY-MM-DD' if possible
    # (doesn't strictly require datetime parsing; safe for your files)
    df[date_col] = df[date_col].astype(str)

    # Optional range mask
    range_mask = pd.Series(True, index=df.index)
    if start is not None:
        range_mask &= (df[date_col] >= start)
    if end is not None:
        range_mask &= (df[date_col] <= end)

    # Decide which rows are "to fill"
    s = df["sentiment_score"]
    to_fill = range_mask & (s == 0)
    if include_nan:
        to_fill |= (range_mask & s.isna())

    n_fill = int(to_fill.sum())
    if n_fill == 0:
        print(f"[{path.name}] nothing to fill (no zeros/NaNs under given conditions).")
        out_path = path if inplace else path.with_name(path.stem + output_suffix + path.suffix)
        if not inplace:
            df.to_csv(out_path, index=False)
        return

    # Build sampling pool:
    # default: all non-zero (and non-NaN) in the whole file
    pool_mask = (~df["sentiment_score"].isna()) & (df["sentiment_score"] != 0)

    # Optional: pool window around the target date range
    if pool_window_days is not None and (start is not None or end is not None):
        # parse to datetime for windowing
        dts = pd.to_datetime(df[date_col], errors="coerce")
        df["_dt"] = dts

        # pick anchor range
        dt_min = pd.to_datetime(start) if start is not None else dts.min()
        dt_max = pd.to_datetime(end) if end is not None else dts.max()

        if pd.isna(dt_min) or pd.isna(dt_max):
            # fallback: ignore windowing
            pass
        else:
            w_start = dt_min - pd.Timedelta(days=pool_window_days)
            w_end = dt_max + pd.Timedelta(days=pool_window_days)
            pool_mask &= (df["_dt"] >= w_start) & (df["_dt"] <= w_end)

    pool = df.loc[pool_mask, "sentiment_score"].astype(float).values
    if pool.size < min_pool:
        raise ValueError(
            f"[{path.name}] sampling pool too small: {pool.size} < {min_pool}. "
            f"Try lowering --min-pool or widening --pool-window-days or using a larger dataset."
        )

    rng = np.random.default_rng(seed)

    # Draw with replacement
    draws = rng.choice(pool, size=n_fill, replace=True)

    # Insert
    df["sentiment_score_is_simulated"] = 0
    df.loc[to_fill, "sentiment_score"] = draws
    df.loc[to_fill, "sentiment_score_is_simulated"] = 1

    # Clean temp col
    if "_dt" in df.columns:
        df.drop(columns=["_dt"], inplace=True)

    out_path = path if inplace else path.with_name(path.stem + output_suffix + path.suffix)
    df.to_csv(out_path, index=False)

    print(
        f"[{path.name}] filled {n_fill} rows. "
        f"pool={pool.size}, seed={seed}. output={out_path.name}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "files",
        nargs="*",
        help="CSV files to process (default: BTC/ETH/XRP in /mnt/data if present).",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--include-nan", action="store_true", help="Also fill NaN sentiment_score")
    ap.add_argument("--date-col", type=str, default="Date")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the original file")
    ap.add_argument("--output-suffix", type=str, default="_sent_sim", help="Suffix for output file")
    ap.add_argument("--min-pool", type=int, default=50, help="Minimum non-zero pool size required")
    ap.add_argument(
        "--pool-window-days",
        type=int,
        default=None,
        help="If set, sample only from non-zero values within +/- this many days around [start,end].",
    )
    args = ap.parse_args()

    # Default files if none passed
    if not args.files:
        defaults = ["/mnt/data/BTC_aggregated.csv", "/mnt/data/ETH_aggregated.csv", "/mnt/data/XRP_aggregated.csv"]
        args.files = [p for p in defaults if Path(p).exists()]
        if not args.files:
            raise SystemExit("No input files provided and no defaults found.")

    for f in args.files:
        fill_one_file(
            path=Path(f),
            seed=args.seed,
            include_nan=args.include_nan,
            date_col=args.date_col,
            start=args.start,
            end=args.end,
            inplace=args.inplace,
            output_suffix=args.output_suffix,
            min_pool=args.min_pool,
            pool_window_days=args.pool_window_days,
        )


if __name__ == "__main__":
    main()
