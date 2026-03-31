#!/usr/bin/env python3
"""Pupil analysis: preprocessing, event-locked responses, and cross-modal correlation.

Covers blink removal, interpolation, smoothing, event-locked pupillometry,
and correlation with external events (stimulus onset, task events, etc.).

Usage:
    python pupil_analysis.py gaze.parquet --output results.csv --plot pupil.png
    python pupil_analysis.py gaze.parquet --events events.csv --pre-ms 500 --post-ms 3000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate, signal, stats


COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"


def preprocess_pupil(
    df: pd.DataFrame,
    pupil_col: str = "pupil_avg",
    blink_col: str = "event",
    blink_label: str = "Blink",
    max_interpolation_gap_ms: float = 150.0,
    smoothing_window_ms: float = 50.0,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Preprocess pupil diameter data: remove blinks, interpolate, smooth.

    Args:
        df: DataFrame with pupil data.
        pupil_col: Pupil diameter column.
        blink_col: Event column for blink detection.
        blink_label: Label identifying blink events.
        max_interpolation_gap_ms: Max gap to interpolate (ms).
        smoothing_window_ms: Savitzky-Golay window (ms).
        ts_col: Timestamp column.

    Returns:
        DataFrame with added 'pupil_cleaned' column.
    """
    result = df.copy()
    pupil = pd.to_numeric(result[pupil_col], errors="coerce")
    ts = pd.to_numeric(result[ts_col], errors="coerce")

    # Mark blinks
    if blink_col in result.columns:
        blink_mask = result[blink_col].astype(str).str.strip().str.lower() == blink_label.lower()
        pupil.loc[blink_mask] = np.nan

    # Remove extreme outliers (> 3 SD from median)
    median_val = pupil.median()
    std_val = pupil.std()
    if pd.notna(std_val) and std_val > 0:
        outlier_mask = (pupil - median_val).abs() > 3 * std_val
        pupil.loc[outlier_mask] = np.nan

    # Interpolate gaps
    if max_interpolation_gap_ms > 0 and len(ts) > 1:
        median_interval = float(np.median(np.diff(ts.dropna().values)))
        max_gap_samples = int(max_interpolation_gap_ms / median_interval) if median_interval > 0 else 5
        result["pupil_cleaned"] = pupil.interpolate(method="linear", limit=max_gap_samples, limit_area="inside")
    else:
        result["pupil_cleaned"] = pupil

    # Smooth
    if smoothing_window_ms > 0 and len(ts) > 1:
        median_interval = float(np.median(np.diff(ts.dropna().values)))
        window_size = int(smoothing_window_ms / median_interval) if median_interval > 0 else 5
        window_size = max(window_size, 5) | 1  # Ensure odd
        if window_size < len(result):
            cleaned = result["pupil_cleaned"].values.astype(float)
            valid = ~np.isnan(cleaned)
            if valid.sum() > window_size:
                smoothed = np.full_like(cleaned, np.nan)
                smoothed[valid] = signal.savgol_filter(
                    cleaned[valid], window_length=window_size, polyorder=2
                )
                result["pupil_cleaned"] = smoothed

    return result


def compute_event_locked_pupil(
    df: pd.DataFrame,
    events: list[dict] | pd.DataFrame,
    pupil_col: str = "pupil_cleaned",
    ts_col: str = "timestamp",
    pre_ms: float = 500.0,
    post_ms: float = 3000.0,
    ts_unit: str = "ms",
) -> pd.DataFrame:
    """Compute event-locked pupil responses (pupillometry).

    Extracts pupil data in a window around each event and aligns them.

    Args:
        df: Preprocessed DataFrame with pupil data.
        events: List of event dicts or DataFrame with 'time' column.
        pupil_col: Cleaned pupil column.
        ts_col: Timestamp column.
        pre_ms: Pre-event window (ms).
        post_ms: Post-event window (ms).
        ts_unit: Timestamp unit ('ms' or 's').

    Returns:
        DataFrame with aligned pupil responses (rows=time bins, cols=events).
    """
    if isinstance(events, list):
        event_times = [e.get("time", e.get("timestamp", e.get("start_time"))) for e in events]
    else:
        event_times = events["time"].values if "time" in events.columns else (
            events.get("timestamp", events.get("start_time")).values
            if hasattr(events, "get") else []
        )

    pupil = pd.to_numeric(df[pupil_col], errors="coerce")
    ts = pd.to_numeric(df[ts_col], errors="coerce")

    # Convert to ms if needed
    ts_ms = ts * 1000 if ts_unit == "s" else ts

    # Create aligned epochs
    epoch_times = np.arange(-pre_ms, post_ms + 1, step=pre_ms / 100)  # 100 bins pre-window
    epochs: dict[str, list] = {"time_ms": epoch_times.tolist()}

    for i, ev_time in enumerate(event_times):
        ev_time_ms = ev_time * 1000 if ts_unit == "s" else ev_time
        aligned = np.full_like(epoch_times, np.nan, dtype=float)

        for j, t in enumerate(epoch_times):
            target_time = ev_time_ms + t
            # Find nearest sample
            nearest_idx = np.argmin(np.abs(ts_ms.values - target_time))
            if len(pupil) > nearest_idx:
                aligned[j] = pupil.iloc[nearest_idx]

        epochs[f"event_{i}"] = aligned.tolist()

    return pd.DataFrame(epochs)


def compute_pupil_summary(
    df: pd.DataFrame,
    pupil_col: str = "pupil_cleaned",
) -> dict:
    """Compute summary statistics for pupil diameter.

    Args:
        df: DataFrame with pupil data.
        pupil_col: Pupil column.

    Returns:
        Dictionary with summary statistics.
    """
    vals = pd.to_numeric(df[pupil_col], errors="coerce").dropna()
    if len(vals) == 0:
        return {"error": "No valid pupil data"}

    v = vals.values
    result: dict = {
        "n_valid_samples": len(v),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0,
        "median": float(np.median(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "range": float(np.max(v) - np.min(v)),
    }

    # Blink fraction
    if "event" in df.columns:
        blink_count = (df["event"].astype(str).str.strip().str.lower() == "blink").sum()
        result["blink_fraction"] = float(blink_count / len(df))
        result["n_blinks"] = int(blink_count)

    return result


def compute_cross_modal_correlation(
    pupil_series: pd.Series,
    other_series: pd.Series,
    method: str = "pearson",
) -> dict:
    """Compute correlation between pupil and another modality.

    Args:
        pupil_series: Pupil diameter time series.
        other_series: Other modality time series.
        method: Correlation method ('pearson', 'spearman').

    Returns:
        Dictionary with correlation coefficient and p-value.
    """
    # Align and drop NaN pairs
    combined = pd.concat([pupil_series, other_series], axis=1).dropna()
    if len(combined) < 10:
        return {"error": "Insufficient paired data"}

    if method == "spearman":
        r, p = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    else:
        r, p = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])

    return {"correlation": float(r), "p_value": float(p), "n_pairs": len(combined)}


def plot_pupil_preprocessing(
    df: pd.DataFrame,
    raw_col: str = "pupil_avg",
    cleaned_col: str = "pupil_cleaned",
    output_path: Optional[str] = None,
) -> None:
    """Plot raw vs cleaned pupil signal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4))

    raw = pd.to_numeric(df[raw_col], errors="coerce")
    cleaned = pd.to_numeric(df[cleaned_col], errors="coerce")

    step = max(1, len(raw) // 3000)
    ax.plot(raw.index[::step], raw.values[::step], alpha=0.3, color="gray", label="Raw", linewidth=0.5)
    ax.plot(cleaned.index[::step], cleaned.values[::step], color=COLOR_PRIMARY, label="Cleaned", linewidth=1)

    # Mark blink periods
    if "event" in df.columns:
        blink_mask = df["event"].astype(str).str.strip().str.lower() == "blink"
        if blink_mask.any():
            blink_indices = df.index[blink_mask]
            ax.scatter(
                blink_indices[::step], raw.loc[blink_indices].values[::step],
                color=COLOR_ACCENT, s=1, alpha=0.3, label="Blinks",
            )

    ax.set_xlabel("Sample")
    ax.set_ylabel("Pupil Diameter")
    ax.set_title("Pupil Diameter: Raw vs Cleaned")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Pupil preprocessing plot saved to: {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pupil analysis.")
    parser.add_argument("input", type=Path, help="Input gaze data (.parquet or .csv).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output results CSV.")
    parser.add_argument("--plot", type=Path, default=None, help="Output plot path.")
    parser.add_argument("--pupil-col", type=str, default="pupil_avg", help="Pupil column.")
    parser.add_argument("--events", type=Path, default=None, help="Events CSV for event-locked analysis.")
    parser.add_argument("--pre-ms", type=float, default=500.0, help="Pre-event window (ms).")
    parser.add_argument("--post-ms", type=float, default=3000.0, help="Post-event window (ms).")
    parser.add_argument("--max-gap-ms", type=float, default=150.0, help="Max interpolation gap (ms).")
    parser.add_argument("--smooth-ms", type=float, default=50.0, help="Smoothing window (ms).")
    args = parser.parse_args()

    # Read
    suffix = args.input.suffix.lower()
    df = pd.read_parquet(args.input) if suffix == ".parquet" else pd.read_csv(args.input)

    # Preprocess
    df = preprocess_pupil(
        df, pupil_col=args.pupil_col,
        max_interpolation_gap_ms=args.max_gap_ms,
        smoothing_window_ms=args.smooth_ms,
    )

    # Summary
    summary = compute_pupil_summary(df, "pupil_cleaned")
    print("Pupil Diameter Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Event-locked analysis
    if args.events:
        events_df = pd.read_csv(args.events)
        epochs = compute_event_locked_pupil(
            df, events_df, pre_ms=args.pre_ms, post_ms=args.post_ms
        )
        mean_response = epochs.drop(columns=["time_ms"]).mean(axis=1)
        print(f"\nEvent-locked pupil response: {len(epochs)} time bins, {len(epochs.columns) - 1} events")
        print(f"  Peak dilation: {mean_response.max():.2f} at {epochs['time_ms'].iloc[np.argmax(mean_response.values)]:.0f} ms")

    # Plot
    if args.plot:
        plot_pupil_preprocessing(df, args.pupil_col, "pupil_cleaned", str(args.plot))

    # Save
    if args.output:
        results = pd.DataFrame([summary])
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
