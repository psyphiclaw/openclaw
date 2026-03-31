#!/usr/bin/env python3
"""Saccade detection and analysis from raw gaze data.

Detects saccades using velocity-threshold methods and computes
amplitude, peak velocity, duration, and direction metrics.

Usage:
    python saccade_detection.py gaze.parquet --threshold 100 --plot output.png
    python saccade_detection.py gaze.parquet --min-duration 20 --min-amplitude 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal


COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"


def compute_velocities(
    df: pd.DataFrame,
    x_col: str = "gaze_x",
    y_col: str = "gaze_y",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Compute angular velocity from gaze coordinates.

    Args:
        df: DataFrame with gaze data.
        x_col, y_col: Gaze coordinate columns.
        ts_col: Timestamp column.

    Returns:
        DataFrame with added velocity columns (deg/s).
    """
    result = df.copy()
    x = pd.to_numeric(result[x_col], errors="coerce")
    y = pd.to_numeric(result[y_col], errors="coerce")
    ts = pd.to_numeric(result[ts_col], errors="coerce")

    dx = x.diff()
    dy = y.diff()
    dt = ts.diff()

    # Assume pixel coordinates — convert to degrees using estimated screen params
    # Using approximate: 1 degree ≈ 35 pixels at typical viewing distance
    pixel_to_deg = 1.0 / 35.0

    dist_pixels = np.sqrt(dx ** 2 + dy ** 2)
    dist_deg = dist_pixels * pixel_to_deg

    # Convert time to seconds
    dt_s = dt / 1000.0 if ts.median() > 1000 else dt

    velocity = dist_deg / dt_s.replace(0, np.nan)
    result["velocity_deg_s"] = velocity
    result["dx"] = dx
    result["dy"] = dy

    return result


def detect_saccades_velocity(
    df: pd.DataFrame,
    velocity_col: str = "velocity_deg_s",
    threshold: float = 100.0,
    min_duration_ms: float = 20.0,
    min_amplitude_deg: float = 1.0,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Detect saccades using velocity threshold.

    Args:
        df: DataFrame with velocity data.
        velocity_col: Column with velocity values (deg/s).
        threshold: Velocity threshold for saccade onset (deg/s).
        min_duration_ms: Minimum saccade duration (ms).
        min_amplitude_deg: Minimum saccade amplitude (degrees).
        ts_col: Timestamp column.

    Returns:
        DataFrame with detected saccade events.
    """
    vel = pd.to_numeric(df[velocity_col], errors="coerce").fillna(0).values
    timestamps = pd.to_numeric(df[ts_col], errors="coerce").values
    dx = df["dx"].values if "dx" in df.columns else np.zeros(len(df))
    dy = df["dy"].values if "dy" in df.columns else np.zeros(len(df))

    # Smooth velocity
    kernel_size = 5
    if len(vel) > kernel_size:
        kernel = np.ones(kernel_size) / kernel_size
        vel_smooth = np.convolve(vel, kernel, mode="same")
    else:
        vel_smooth = vel

    # Threshold crossings
    is_saccade = vel_smooth > threshold

    # Find saccade episodes (contiguous True regions)
    saccades: list[dict] = []
    in_saccade = False
    onset_idx = 0

    for i in range(len(is_saccade)):
        if is_saccade[i] and not in_saccade:
            onset_idx = i
            in_saccade = True
        elif not is_saccade[i] and in_saccade:
            # End of saccade episode
            offset_idx = i - 1
            saccade_data = _build_saccade_dict(
                onset_idx, offset_idx, timestamps, dx, dy, vel_smooth
            )
            if saccade_data is not None:
                saccades.append(saccade_data)
            in_saccade = False

    # Handle saccade at end of data
    if in_saccade:
        saccade_data = _build_saccade_dict(
            onset_idx, len(is_saccade) - 1, timestamps, dx, dy, vel_smooth
        )
        if saccade_data is not None:
            saccades.append(saccade_data)

    if not saccades:
        return pd.DataFrame()

    saccade_df = pd.DataFrame(saccades)

    # Filter by duration and amplitude
    if min_duration_ms > 0 and "duration_ms" in saccade_df.columns:
        saccade_df = saccade_df[saccade_df["duration_ms"] >= min_duration_ms]
    if min_amplitude_deg > 0 and "amplitude_deg" in saccade_df.columns:
        saccade_df = saccade_df[saccade_df["amplitude_deg"] >= min_amplitude_deg]

    return saccade_df.reset_index(drop=True)


def _build_saccade_dict(
    onset: int,
    offset: int,
    timestamps: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    vel: np.ndarray,
) -> Optional[dict]:
    """Build a saccade event dictionary from onset/offset indices."""
    if offset <= onset:
        return None

    pixel_to_deg = 1.0 / 35.0
    ts_unit = "ms" if np.median(timestamps) > 1000 else "s"
    ts_scale = 1000.0 if ts_unit == "s" else 1.0

    duration = (timestamps[offset] - timestamps[onset]) * (ts_scale if ts_unit == "s" else 1)
    amp = np.sum(np.sqrt(dx[onset:offset + 1] ** 2 + dy[onset:offset + 1] ** 2)) * pixel_to_deg
    peak_vel = float(np.max(vel[onset:offset + 1]))
    total_dx = np.sum(dx[onset:offset + 1]) * pixel_to_deg
    total_dy = np.sum(dy[onset:offset + 1]) * pixel_to_deg
    direction = np.degrees(np.arctan2(total_dy, total_dx))

    return {
        "onset_time": float(timestamps[onset]),
        "offset_time": float(timestamps[offset]),
        "duration_ms": float(duration) if ts_unit == "ms" else float(duration * 1000),
        "amplitude_deg": float(amp),
        "peak_velocity_deg_s": float(peak_vel),
        "direction_deg": float(direction),
        "onset_index": onset,
        "offset_index": offset,
    }


def compute_saccade_summary(saccades: pd.DataFrame) -> dict:
    """Compute summary statistics for detected saccades.

    Args:
        saccades: DataFrame of detected saccade events.

    Returns:
        Dictionary with summary statistics.
    """
    if saccades.empty:
        return {"n_saccades": 0}

    summary: dict = {"n_saccades": len(saccades)}

    for col, key_prefix in [
        ("duration_ms", "duration"),
        ("amplitude_deg", "amplitude"),
        ("peak_velocity_deg_s", "peak_velocity"),
        ("direction_deg", "direction"),
    ]:
        if col in saccades.columns:
            vals = saccades[col].dropna()
            if len(vals) > 0:
                summary[f"{key_prefix}_mean"] = float(vals.mean())
                summary[f"{key_prefix}_std"] = float(vals.std()) if len(vals) > 1 else 0
                summary[f"{key_prefix}_median"] = float(vals.median())
                summary[f"{key_prefix}_min"] = float(vals.min())
                summary[f"{key_prefix}_max"] = float(vals.max())

    # Direction distribution
    if "direction_deg" in saccades.columns:
        dirs = saccades["direction_deg"].values
        summary["horizontal_ratio"] = float(
            np.sum(np.abs(dirs) < 45) / len(dirs)
            if len(dirs) > 0
            else 0
        )
        summary["vertical_ratio"] = float(
            np.sum((np.abs(dirs) > 45) & (np.abs(dirs) < 135)) / len(dirs)
            if len(dirs) > 0
            else 0
        )

    return summary


def plot_saccade_analysis(
    saccades: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """Plot saccade metrics: amplitude histogram + polar direction plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Amplitude histogram
    ax = axes[0]
    if "amplitude_deg" in saccades.columns:
        ax.hist(saccades["amplitude_deg"], bins=40, color=COLOR_PRIMARY, alpha=0.7, edgecolor="white")
        ax.set_xlabel("Amplitude (degrees)")
        ax.set_ylabel("Count")
        ax.set_title("Saccade Amplitude Distribution")
    else:
        ax.text(0.5, 0.5, "No amplitude data", ha="center", va="center")

    # Duration histogram
    ax = axes[1]
    if "duration_ms" in saccades.columns:
        ax.hist(saccades["duration_ms"], bins=40, color=COLOR_ACCENT, alpha=0.7, edgecolor="white")
        ax.set_xlabel("Duration (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Saccade Duration Distribution")
    else:
        ax.text(0.5, 0.5, "No duration data", ha="center", va="center")

    # Polar direction plot
    ax = axes[2]
    ax.remove()
    ax = fig.add_subplot(133, projection="polar")
    if "direction_deg" in saccades.columns:
        directions_rad = np.radians(saccades["direction_deg"].values)
        bins = np.linspace(0, 2 * np.pi, 37)
        counts, _ = np.histogram(directions_rad, bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        width = 2 * np.pi / len(counts)
        ax.bar(centers, counts, width=width, color=COLOR_PRIMARY, alpha=0.7, edgecolor="white")
        ax.set_title("Saccade Direction", pad=20)
    else:
        ax.text(0, 0, "No direction data", ha="center", va="center")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saccade analysis plot saved to: {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Saccade detection and analysis.")
    parser.add_argument("input", type=Path, help="Input gaze data (.parquet or .csv).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output saccades CSV.")
    parser.add_argument("--threshold", type=float, default=100.0, help="Velocity threshold (deg/s).")
    parser.add_argument("--min-duration", type=float, default=20.0, help="Min duration (ms).")
    parser.add_argument("--min-amplitude", type=float, default=1.0, help="Min amplitude (deg).")
    parser.add_argument("--plot", type=Path, default=None, help="Output plot path.")
    parser.add_argument("--x-col", type=str, default="gaze_x", help="Gaze X column.")
    parser.add_argument("--y-col", type=str, default="gaze_y", help="Gaze Y column.")
    args = parser.parse_args()

    # Read
    suffix = args.input.suffix.lower()
    df = pd.read_parquet(args.input) if suffix == ".parquet" else pd.read_csv(args.input)

    # Compute velocities
    df = compute_velocities(df, x_col=args.x_col, y_col=args.y_col)

    # Detect saccades
    saccades = detect_saccades_velocity(
        df, threshold=args.threshold,
        min_duration_ms=args.min_duration, min_amplitude_deg=args.min_amplitude,
    )

    if saccades.empty:
        print("No saccades detected with current parameters.")
        return

    # Summary
    summary = compute_saccade_summary(saccades)
    print("Saccade Detection Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # Plot
    if args.plot:
        plot_saccade_analysis(saccades, str(args.plot))

    # Save
    if args.output:
        saccades.to_csv(args.output, index=False)
        print(f"Saccade events saved to: {args.output}")


if __name__ == "__main__":
    main()
