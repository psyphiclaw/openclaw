#!/usr/bin/env python3
"""Clean multimodal behavioral data.

Handles missing values, outlier detection, artifact detection, and quality scoring.

Usage:
    python clean_data.py --missing drop --outliers zscore -i data.csv -o cleaned.csv
    python clean_data.py --missing interpolate --outliers iqr -i data.csv -o cleaned.csv
    python clean_data.py --quality-only -i data.csv
    python clean_data.py --artifacts eeg -i eeg_data.csv -o cleaned.csv --time-col Time
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------
def handle_missing(
    df: pd.DataFrame,
    strategy: str = "drop",
    threshold: float = 0.5,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Handle missing values according to strategy.

    Args:
        df: Input DataFrame.
        strategy: 'drop' (drop rows), 'fill_mean', 'fill_median', 'interpolate',
                  'ffill' (forward fill), 'bfill' (backward fill),
                  'drop_cols' (drop columns exceeding threshold).
        threshold: Fraction threshold for drop_cols.
        columns: Specific columns to process.

    Returns:
        (cleaned_df, report_dict).
    """
    cols = columns or df.columns.tolist()
    report: dict[str, int] = {"missing_before": int(df[cols].isna().sum().sum())}
    result = df.copy()

    if strategy == "drop":
        result = result.dropna(subset=cols)
    elif strategy == "drop_cols":
        na_frac = result[cols].isna().mean()
        drop_cols = na_frac[na_frac > threshold].index.tolist()
        result = result.drop(columns=drop_cols)
        report["dropped_cols"] = len(drop_cols)
    elif strategy == "fill_mean":
        for c in cols:
            if c in result.columns:
                result[c] = result[c].fillna(result[c].mean())
    elif strategy == "fill_median":
        for c in cols:
            if c in result.columns:
                result[c] = result[c].fillna(result[c].median())
    elif strategy == "interpolate":
        for c in cols:
            if c in result.columns:
                result[c] = result[c].interpolate(method="linear", limit_direction="both")
    elif strategy == "ffill":
        result[cols] = result[cols].ffill().bfill()
    elif strategy == "bfill":
        result[cols] = result[cols].bfill().ffill()

    report["missing_after"] = int(result[cols].isna().sum().sum())
    report["rows_removed"] = len(df) - len(result)
    return result, report


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------
def detect_outliers(
    df: pd.DataFrame,
    method: str = "zscore",
    threshold: float = 3.0,
    action: str = "clip",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect and handle outliers.

    Args:
        method: 'zscore', 'iqr', or 'mad'.
        threshold: Detection threshold.
        action: 'clip', 'remove', or 'mark' (add _outlier flag column).
        columns: Columns to check.

    Returns:
        (cleaned_df, report).
    """
    cols = columns or [c for c in df.select_dtypes(include="number").columns if not c.endswith("_outlier")]
    report: dict[str, Any] = {"method": method, "threshold": threshold}
    result = df.copy()
    total_outliers = 0

    for col in cols:
        if col not in result.columns:
            continue
        vals = result[col].astype(float)

        if method == "zscore":
            mean, std = vals.mean(), vals.std()
            if std == 0:
                continue
            z = np.abs((vals - mean) / std)
            outlier_mask = z > threshold
        elif method == "iqr":
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (vals < q1 - threshold * iqr) | (vals > q3 + threshold * iqr)
        elif method == "mad" and HAS_SCIPY:
            med = vals.median()
            mad = scipy_stats.median_abs_deviation(vals, scale="normal")
            if mad == 0:
                continue
            outlier_mask = np.abs((vals - med) / mad) > threshold
        else:
            continue

        n_outliers = int(outlier_mask.sum())
        total_outliers += n_outliers
        report[f"{col}_outliers"] = n_outliers

        if action == "clip" and n_outliers > 0:
            lo, hi = vals[~outlier_mask].min(), vals[~outlier_mask].max()
            result[col] = vals.clip(lo, hi)
        elif action == "remove":
            result = result.loc[~outlier_mask]
        elif action == "mark":
            result[f"{col}_outlier"] = outlier_mask.astype(int)

    report["total_outliers"] = total_outliers
    report["action"] = action
    return result, report


# ---------------------------------------------------------------------------
# Artifact detection (modality-specific)
# ---------------------------------------------------------------------------
def detect_artifacts(
    df: pd.DataFrame,
    modality: str,
    time_col: str = "Time",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect signal artifacts based on modality.

    Args:
        df: Input DataFrame.
        modality: 'eeg', 'eye', 'physio', 'face'.
        time_col: Time column name.
        columns: Columns to check.

    Returns:
        (df with _artifact flag columns, report).
    """
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    result = df.copy()
    report: dict[str, Any] = {"modality": modality}

    if modality == "eeg":
        # Detect voltage jumps > 100 μV and flat signals
        for col in cols:
            if col not in result.columns:
                continue
            vals = result[col].astype(float)
            diff = vals.diff().abs()
            artifact = (diff > 100) | (vals.abs() > 500)
            # Also detect flat segments (std < 0.01 in rolling window)
            if len(vals) > 10:
                rolling_std = vals.rolling(window=10, center=True).std()
                artifact = artifact | (rolling_std < 0.01)
            result[f"{col}_artifact"] = artifact.astype(int)
            report[f"{col}_artifacts"] = int(artifact.sum())

    elif modality == "eye":
        # Detect sudden jumps and loss-of-track (0 or NaN values)
        for col in cols:
            if col not in result.columns:
                continue
            vals = result[col].astype(float)
            diff = vals.diff().abs()
            artifact = (diff > vals.std() * 5) | (vals == 0) | vals.isna()
            result[f"{col}_artifact"] = artifact.astype(int)
            report[f"{col}_artifacts"] = int(artifact.sum())

    elif modality == "physio":
        # Detect ECG R-peak clipping and EDA saturation
        for col in cols:
            if col not in result.columns:
                continue
            vals = result[col].astype(float)
            artifact = (vals == vals.max()) | (vals == vals.min())
            result[f"{col}_artifact"] = artifact.astype(int)
            report[f"{col}_artifacts"] = int(artifact.sum())

    elif modality == "face":
        # Detect confidence < threshold and NaN bursts
        for col in cols:
            if col not in result.columns:
                continue
            vals = result[col].astype(float)
            if "confidence" in col.lower():
                artifact = vals < 0.5
            else:
                artifact = vals.isna()
            result[f"{col}_artifact"] = artifact.astype(int)
            report[f"{col}_artifacts"] = int(artifact.sum())

    return result, report


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------
def quality_score(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """Compute per-column quality scores (0-100).

    Scores based on: completeness, outlier ratio, artifact ratio.
    """
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    scores: dict[str, float] = {}

    for col in cols:
        if col not in df.columns:
            scores[col] = 0.0
            continue

        vals = df[col].dropna()
        total = len(df)

        # Completeness (40 points)
        completeness = len(vals) / total * 40 if total > 0 else 0

        # Outlier ratio (30 points)
        if HAS_SCIPY and len(vals) > 10:
            z = np.abs(scipy_stats.zscore(vals))
            outlier_ratio = (z > 3).mean()
            outlier_score = max(0, (1 - outlier_ratio * 5)) * 30
        else:
            outlier_score = 15  # Neutral if can't compute

        # Variance score (30 points) — penalize near-zero variance
        if len(vals) > 1:
            cv = vals.std() / (abs(vals.mean()) + 1e-10)
            variance_score = min(30, max(0, 30 - abs(np.log10(cv + 1e-10)) * 5))
        else:
            variance_score = 0

        scores[col] = round(completeness + outlier_score + variance_score, 1)

    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean multimodal behavioral data")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file")
    parser.add_argument("--missing", choices=["drop", "drop_cols", "fill_mean", "fill_median", "interpolate", "ffill", "bfill", "none"], default="none", help="Missing value strategy")
    parser.add_argument("--outliers", choices=["zscore", "iqr", "mad", "none"], default="none", help="Outlier detection method")
    parser.add_argument("--outlier-action", choices=["clip", "remove", "mark"], default="clip", help="Outlier handling action")
    parser.add_argument("--outlier-threshold", type=float, default=3.0, help="Outlier threshold")
    parser.add_argument("--artifacts", choices=["eeg", "eye", "physio", "face", "none"], default="none", help="Artifact detection modality")
    parser.add_argument("--time-col", default="Time", help="Time column name")
    parser.add_argument("--quality-only", action="store_true", help="Only compute quality scores")
    parser.add_argument("--columns", nargs="+", default=None, help="Specific columns to process")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)
    print(f"Input: {len(df)} rows x {len(df.columns)} columns\n")

    if args.quality_only:
        scores = quality_score(df, args.columns)
        print("=" * 50)
        print("数据质量评分 (0-100)")
        print("=" * 50)
        for col, score in sorted(scores.items(), key=lambda x: x[1]):
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            status = "✅" if score >= 80 else ("⚠️" if score >= 50 else "❌")
            print(f"  {status} {col:30s} {score:5.1f} {bar}")
        print("=" * 50)
        avg = np.mean(list(scores.values()))
        print(f"  平均分: {avg:.1f}")
        return 0

    result = df
    all_reports: list[dict[str, Any]] = []

    # Missing values
    if args.missing != "none":
        result, rpt = handle_missing(result, args.missing, columns=args.columns)
        all_reports.append(rpt)
        print(f"Missing values ({args.missing}): removed {rpt['rows_removed']} rows, {rpt['missing_after']} missing remain")

    # Outliers
    if args.outliers != "none":
        result, rpt = detect_outliers(result, args.outliers, args.outlier_threshold, args.outlier_action, args.columns)
        all_reports.append(rpt)
        print(f"Outliers ({args.outliers}, {args.outlier_action}): {rpt['total_outliers']} detected")

    # Artifacts
    if args.artifacts != "none":
        result, rpt = detect_artifacts(result, args.artifacts, args.time_col, args.columns)
        all_reports.append(rpt)
        art_total = sum(v for k, v in rpt.items() if k.endswith("_artifacts"))
        print(f"Artifacts ({args.artifacts}): {art_total} detected")

    # Quality score of result
    print()
    scores = quality_score(result, args.columns)
    avg_score = np.mean(list(scores.values())) if scores else 0
    print(f"Output quality score: {avg_score:.1f}/100")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"\nOutput: {args.output} ({len(result)} rows)")
    else:
        print(result.to_csv(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
