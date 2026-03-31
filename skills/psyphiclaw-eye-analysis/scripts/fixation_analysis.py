#!/usr/bin/env python3
"""Fixation analysis for eye-tracking data.

Computes fixation duration statistics, counts, time-to-first-fixation,
and transition matrices from standardized eye-tracking DataFrames.

Usage:
    python fixation_analysis.py gaze.parquet --output stats.csv --plot output.png
    python fixation_analysis.py gaze.parquet --transitions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"


def compute_fixation_duration_stats(
    df: pd.DataFrame,
    duration_col: str = "fixation_duration",
) -> dict:
    """Compute descriptive statistics for fixation durations.

    Args:
        df: DataFrame with fixation data (event-level or sample-level).
        duration_col: Column containing fixation durations in ms.

    Returns:
        Dictionary with mean, median, std, min, max, quartiles, skewness, kurtosis.
    """
    if duration_col not in df.columns:
        raise ValueError(f"Column '{duration_col}' not found. Available: {list(df.columns)}")

    durations = pd.to_numeric(df[duration_col], errors="coerce").dropna()
    if len(durations) == 0:
        return {"error": "No valid fixation durations found"}

    d = durations.values
    result: dict = {
        "n_fixations": len(d),
        "mean_ms": float(np.mean(d)),
        "median_ms": float(np.median(d)),
        "std_ms": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
        "min_ms": float(np.min(d)),
        "max_ms": float(np.max(d)),
        "q25_ms": float(np.percentile(d, 25)),
        "q75_ms": float(np.percentile(d, 75)),
        "iqr_ms": float(np.percentile(d, 75) - np.percentile(d, 25)),
        "skewness": float(stats.skew(d)),
        "kurtosis": float(stats.kurtosis(d)),
        "total_fixation_time_ms": float(np.sum(d)),
    }

    return result


def extract_fixations_from_samples(
    df: pd.DataFrame,
    index_col: str = "fixation_index",
) -> pd.DataFrame:
    """Extract fixation-level data from sample-level DataFrame.

    Args:
        df: Sample-level DataFrame with fixation_index column.
        index_col: Column containing fixation IDs.

    Returns:
        Fixation-level DataFrame with duration, position, etc.
    """
    if index_col not in df.columns:
        raise ValueError(f"Column '{index_col}' not found in DataFrame")

    fixations: list[dict] = []
    grouped = df.groupby(index_col)

    for fix_id, group in grouped:
        if pd.isna(fix_id):
            continue

        fixation: dict = {"fixation_index": int(fix_id)}

        if "timestamp" in df.columns:
            ts = group["timestamp"].dropna()
            if len(ts) > 1:
                fixation["duration_ms"] = ts.iloc[-1] - ts.iloc[0]
                fixation["start_time"] = ts.iloc[0]
                fixation["end_time"] = ts.iloc[-1]
            elif len(ts) == 1:
                fixation["duration_ms"] = 0
                fixation["start_time"] = ts.iloc[0]
                fixation["end_time"] = ts.iloc[0]

        for coord in ("gaze_x", "gaze_y"):
            if coord in df.columns:
                vals = pd.to_numeric(group[coord], errors="coerce").dropna()
                if len(vals) > 0:
                    fixation[f"mean_{coord}"] = float(vals.mean())
                    fixation[f"std_{coord}"] = float(vals.std())

        if "pupil_avg" in df.columns:
            vals = pd.to_numeric(group["pupil_avg"], errors="coerce").dropna()
            if len(vals) > 0:
                fixation["mean_pupil"] = float(vals.mean())

        fixations.append(fixation)

    return pd.DataFrame(fixations)


def compute_time_to_first_fixation(
    fixations: pd.DataFrame,
    start_time: float = 0.0,
    start_col: str = "start_time",
) -> dict:
    """Compute time to first fixation from a reference start time.

    Args:
        fixations: Fixation-level DataFrame.
        start_time: Reference time (ms).
        start_col: Column with fixation start times.

    Returns:
        Dictionary with first_fixation_time_ms and rank of each fixation.
    """
    if start_col not in fixations.columns:
        return {"error": f"Column '{start_col}' not found"}

    valid = fixations[fixations[start_col] >= start_time].sort_values(start_col)
    if len(valid) == 0:
        return {"first_fixation_time_ms": None, "n_fixations": 0}

    first_t = valid.iloc[0][start_col]
    return {
        "first_fixation_time_ms": float(first_t - start_time),
        "first_fixation_index": int(valid.iloc[0]["fixation_index"]),
        "n_fixations": len(valid),
    }


def compute_transition_matrix(
    df: pd.DataFrame,
    aoi_col: str = "aoi_id",
    fixation_index_col: str = "fixation_index",
    n_aois: Optional[int] = None,
) -> tuple[np.ndarray, list]:
    """Compute transition matrix between AOIs based on fixation sequence.

    Args:
        df: Sample or fixation-level DataFrame.
        aoi_col: Column with AOI IDs.
        fixation_index_col: Column with fixation index (for sample-level data).
        n_aois: Number of AOIs. Auto-detected if None.

    Returns:
        Tuple of (transition_matrix, aoi_labels).
    """
    # If sample-level, get one AOI per fixation (mode)
    if fixation_index_col in df.columns and aoi_col in df.columns:
        fix_aoi = df.groupby(fixation_index_col)[aoi_col].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
        ).dropna()
    elif aoi_col in df.columns:
        fix_aoi = df[aoi_col].dropna()
    else:
        raise ValueError(f"Column '{aoi_col}' not found")

    if len(fix_aoi) < 2:
        return np.array([]), []

    labels = sorted(fix_aoi.unique())
    if n_aois is not None:
        labels = labels[:n_aois]

    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)

    matrix = np.zeros((n, n), dtype=int)
    seq = fix_aoi.values
    for i in range(1, len(seq)):
        src, dst = seq[i - 1], seq[i]
        if src in label_to_idx and dst in label_to_idx:
            matrix[label_to_idx[src], label_to_idx[dst]] += 1

    return matrix, labels


def plot_fixation_durations(
    durations: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """Plot fixation duration histogram and distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    ax.hist(durations, bins=50, color=COLOR_PRIMARY, alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(durations), color=COLOR_ACCENT, linestyle="--", label=f"Mean: {np.mean(durations):.1f} ms")
    ax.axvline(np.median(durations), color="#2ECC71", linestyle="--", label=f"Median: {np.median(durations):.1f} ms")
    ax.set_xlabel("Fixation Duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Fixation Duration Distribution")
    ax.legend(fontsize=8)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot(durations, patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(COLOR_PRIMARY)
    bp["boxes"][0].set_alpha(0.5)
    ax.set_ylabel("Fixation Duration (ms)")
    ax.set_title("Fixation Duration Box Plot")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Fixation duration plot saved to: {output_path}")
    plt.close(fig)


def plot_transition_matrix(
    matrix: np.ndarray,
    labels: list,
    output_path: Optional[str] = None,
) -> None:
    """Plot AOI transition matrix as heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    n = len(labels)

    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = matrix / row_sums

    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("To AOI")
    ax.set_ylabel("From AOI")
    ax.set_title("AOI Transition Matrix (row-normalized)")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                       fontsize=8, color="white" if norm_matrix[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Transition Probability")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Transition matrix plot saved to: {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixation analysis for eye-tracking data.")
    parser.add_argument("input", type=Path, help="Input gaze data (.parquet or .csv).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output stats CSV.")
    parser.add_argument("--plot", type=Path, default=None, help="Output plot path.")
    parser.add_argument("--transitions", action="store_true", help="Compute transition matrix.")
    parser.add_argument("--aoi-col", type=str, default="aoi_id", help="AOI column name.")
    parser.add_argument(
        "--transition-plot", type=Path, default=None, help="Transition matrix plot path."
    )
    args = parser.parse_args()

    # Read input
    suffix = args.input.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # Extract fixations if sample-level
    if "fixation_index" in df.columns and "fixation_duration" not in df.columns:
        fixations = extract_fixations_from_samples(df)
        print(f"Extracted {len(fixations)} fixations from sample data.")
    elif "fixation_duration" in df.columns:
        fixations = df
    else:
        print("Error: No fixation data found (need fixation_index or fixation_duration column).")
        sys.exit(1)

    # Duration stats
    duration_stats = compute_fixation_duration_stats(fixations)
    print("Fixation Duration Statistics:")
    for k, v in duration_stats.items():
        print(f"  {k}: {v}")

    # Time to first fixation
    ttf = compute_time_to_first_fixation(fixations)
    print(f"\nTime to First Fixation: {ttf.get('first_fixation_time_ms', 'N/A')} ms")

    # Transition matrix
    if args.transitions and args.aoi_col in df.columns:
        matrix, labels = compute_transition_matrix(df, aoi_col=args.aoi_col)
        print(f"\nTransition Matrix ({len(labels)} AOIs):")
        print(pd.DataFrame(matrix, index=labels, columns=labels))
        if args.transition_plot:
            plot_transition_matrix(matrix, labels, str(args.transition_plot))

    # Plot
    if args.plot and "fixation_duration" in fixations.columns:
        durations = pd.to_numeric(fixations["fixation_duration"], errors="coerce").dropna().values
        plot_fixation_durations(durations, str(args.plot))

    # Save
    if args.output:
        stats_df = pd.DataFrame([duration_stats, ttf])
        stats_df.to_csv(args.output, index=False)
        print(f"Stats saved to: {args.output}")


if __name__ == "__main__":
    main()
