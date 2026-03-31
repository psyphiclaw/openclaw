#!/usr/bin/env python3
"""Normalize multimodal behavioral data.

Supports Z-score, Min-Max, baseline correction, and percentile rank transforms.

Usage:
    python normalize_data.py --method zscore -i data.csv -o normalized.csv
    python normalize_data.py --method minmax -i data.csv -o normalized.csv --columns col1 col2
    python normalize_data.py --method baseline --baseline-start 0 --baseline-end 5 -i data.csv -o corrected.csv --time-col Time
    python normalize_data.py --method percentile -i data.csv -o ranked.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def zscore_normalize(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Z-score standardization: (x - mean) / std."""
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            print(f"Warning: column '{col}' not found, skipping", file=sys.stderr)
            continue
        vals = result[col].astype(float)
        mean, std = vals.mean(), vals.std()
        if std == 0:
            print(f"Warning: column '{col}' has zero std, setting to 0", file=sys.stderr)
            result[col] = 0.0
        else:
            result[col] = (vals - mean) / std
    return result


def minmax_normalize(df: pd.DataFrame, columns: list[str] | None = None, feature_range: tuple[float, float] = (0, 1)) -> pd.DataFrame:
    """Min-Max normalization to [min, max] range."""
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        vals = result[col].astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            result[col] = feature_range[0]
        else:
            result[col] = (vals - vmin) / (vmax - vmin) * (feature_range[1] - feature_range[0]) + feature_range[0]
    return result


def baseline_correct(
    df: pd.DataFrame,
    time_col: str,
    baseline_start: float,
    baseline_end: float,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Baseline correction: subtract mean of baseline period from all values."""
    cols = columns or [c for c in df.select_dtypes(include="number").columns if c != time_col]
    result = df.copy()

    if time_col not in df.columns:
        print(f"Error: time column '{time_col}' not found", file=sys.stderr)
        return result

    mask = (df[time_col] >= baseline_start) & (df[time_col] <= baseline_end)
    baseline_df = df.loc[mask]
    if baseline_df.empty:
        print(f"Warning: no data in baseline window [{baseline_start}, {baseline_end}]", file=sys.stderr)
        return result

    for col in cols:
        if col not in result.columns:
            continue
        baseline_mean = baseline_df[col].astype(float).mean()
        result[col] = result[col].astype(float) - baseline_mean
        print(f"  {col}: baseline mean = {baseline_mean:.4f}")

    return result


def percentile_rank(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert to percentile ranks (0-100)."""
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        result[col] = result[col].rank(pct=True) * 100
    return result


METHODS = {
    "zscore": zscore_normalize,
    "minmax": minmax_normalize,
    "baseline": baseline_correct,
    "percentile": percentile_rank,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize multimodal behavioral data")
    parser.add_argument("--method", choices=METHODS.keys(), required=True, help="Normalization method")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file (default: stdout)")
    parser.add_argument("--columns", nargs="+", default=None, help="Specific columns to normalize")
    # Baseline-specific options
    parser.add_argument("--time-col", default="Time", help="Time column name for baseline correction")
    parser.add_argument("--baseline-start", type=float, default=0, help="Baseline start time (seconds)")
    parser.add_argument("--baseline-end", type=float, default=5, help="Baseline end time (seconds)")
    # Min-Max range
    parser.add_argument("--min", type=float, default=0, help="Min-Max lower bound")
    parser.add_argument("--max", type=float, default=1, help="Min-Max upper bound")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)
    print(f"Input: {len(df)} rows x {len(df.columns)} columns")

    if args.method == "baseline":
        result = baseline_correct(df, args.time_col, args.baseline_start, args.baseline_end, args.columns)
    elif args.method == "minmax":
        result = minmax_normalize(df, args.columns, feature_range=(args.min, args.max))
    else:
        result = METHODS[args.method](df, args.columns)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        suffix = args.output.suffix.lower()
        if suffix == ".h5" or suffix == ".hdf5":
            result.to_hdf(args.output, key="data", mode="w")
        elif suffix == ".parquet":
            result.to_parquet(args.output)
        else:
            result.to_csv(args.output, index=False)
        print(f"Output: {args.output} ({len(result)} rows)")
    else:
        print(result.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
