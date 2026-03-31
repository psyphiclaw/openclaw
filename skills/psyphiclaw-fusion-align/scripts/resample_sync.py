#!/usr/bin/env python3
"""Resample aligned modalities to a unified sampling rate and export as HDF5.

Usage:
    python resample_sync.py \
        --inputs eeg.csv face.csv physio.csv \
        --target-freq 250 \
        --method linear \
        --output synced_session.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Reuse session_manager data structure
sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_manager import MultiModalSession  # noqa: E402

BLUE = "#4A90D9"
RED = "#E74C3C"


def load_csv_as_arrays(
    path: str,
    timestamp_col: str = "timestamp",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a CSV and return (timestamps_ms, values, column_names).

    Values exclude the timestamp column.  Missing values are forward-filled
    then linearly interpolated.
    """
    df = pd.read_csv(path)
    for col in ("timestamp", "Timestamp", "TIMESTAMP", "time_ms"):
        if col in df.columns:
            timestamp_col = col
            break

    ts = pd.to_numeric(df[timestamp_col], errors="coerce").values
    value_cols = [c for c in df.columns if c != timestamp_col]
    vals = df[value_cols].apply(pd.to_numeric, errors="coerce").values
    # Handle NaN: forward fill then interpolate
    vals_df = pd.DataFrame(vals)
    vals_df = vals_df.ffill().interpolate(axis=0, limit_direction="both").fillna(0)
    return ts, vals_df.values, value_cols


def resample_signal(
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    target_freq: float,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a signal to *target_freq* Hz.

    Parameters
    ----------
    timestamps_ms : 1-D array of timestamps in milliseconds.
    values : 2-D array of shape ``(n_time, n_features)``.
    target_freq : Target sampling rate in Hz.
    method : One of ``linear``, ``nearest``, ``ffill``.

    Returns
    -------
    new_times_ms, new_values
    """
    if len(timestamps_ms) < 2:
        return timestamps_ms, values

    duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    n_new = max(int(duration_s * target_freq), 1)
    new_times_ms = np.linspace(timestamps_ms[0], timestamps_ms[-1], n_new)

    if method == "ffill":
        # Index of the last sample at or before each new time
        indices = np.searchsorted(timestamps_ms, new_times_ms, side="right") - 1
        indices = np.clip(indices, 0, len(values) - 1)
        new_values = values[indices]
    elif method == "nearest":
        indices = np.searchsorted(timestamps_ms, new_times_ms, side="left")
        indices = np.clip(indices, 0, len(values) - 1)
        new_values = values[indices]
    else:  # linear (default)
        new_values = np.column_stack([
            np.interp(new_times_ms, timestamps_ms, values[:, i])
            for i in range(values.shape[1])
        ])

    return new_times_ms, new_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample aligned modalities to a unified sampling rate."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Aligned CSV files (one per modality).",
    )
    parser.add_argument(
        "--modality-names", nargs="+", default=None,
        help="Modality names (default: filename stems).",
    )
    parser.add_argument(
        "--target-freq", type=float, default=None,
        help="Target sampling rate in Hz (default: highest among inputs).",
    )
    parser.add_argument(
        "--method", choices=["linear", "nearest", "ffill"], default="linear",
        help="Interpolation method (default: linear).",
    )
    parser.add_argument(
        "--session-name", default="synced_session",
        help="Name for the output session.",
    )
    parser.add_argument(
        "--output", required=True, help="Output HDF5 path (.h5).",
    )
    parser.add_argument(
        "--alignment-params", default=None,
        help="Optional JSON with alignment offsets to embed.",
    )
    args = parser.parse_args()

    # Load all modalities
    modalities: list[dict[str, Any]] = []
    max_freq = 0.0
    for i, fpath in enumerate(args.inputs):
        ts, vals, cols = load_csv_as_arrays(fpath)
        name = args.modality_names[i] if args.modality_names else Path(fpath).stem
        # Estimate sampling rate
        dt = np.median(np.diff(ts)) / 1000.0  # seconds
        freq = 1.0 / dt if dt > 0 else 0.0
        max_freq = max(max_freq, freq)
        modalities.append({
            "name": name,
            "timestamps_ms": ts,
            "values": vals,
            "columns": cols,
            "estimated_freq": freq,
            "source": str(fpath),
        })
        print(f"  [{name}] {len(ts)} samples, est. {freq:.1f} Hz, {vals.shape[1]} features")

    target_freq = args.target_freq if args.target_freq else max_freq
    print(f"\n🎯 Target frequency: {target_freq:.1f} Hz (method: {args.method})")

    # Resample
    resampled: list[dict[str, Any]] = []
    for mod in modalities:
        new_ts, new_vals = resample_signal(
            mod["timestamps_ms"], mod["values"], target_freq, args.method
        )
        resampled.append({
            "name": mod["name"],
            "timestamps_ms": new_ts,
            "values": new_vals,
            "columns": mod["columns"],
            "sampling_rate": target_freq,
            "source": mod["source"],
        })
        print(f"  [{mod['name']}] → {len(new_ts)} samples @ {target_freq:.1f} Hz")

    # Build MultiModalSession
    session = MultiModalSession(name=args.session_name)
    session.metadata["target_freq"] = target_freq
    session.metadata["resample_method"] = args.method

    if args.alignment_params:
        with open(args.alignment_params) as f:
            session.metadata["alignment"] = json.load(f)

    for mod in resampled:
        session.add_modality(
            name=mod["name"],
            data=pd.DataFrame(mod["values"], columns=mod["columns"]),
            timestamps_ms=mod["timestamps_ms"],
            sampling_rate=mod["sampling_rate"],
            source=mod["source"],
        )

    session.metadata["aligned"] = True

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session.save(out_path)

    print(f"\n✅ Synchronized session saved to {out_path}")
    print(f"   Modalities: {list(session.modalities.keys())}")
    print(f"   Duration: {(resampled[0]['timestamps_ms'][-1] - resampled[0]['timestamps_ms'][0])/1000:.2f} s")
    print(f"   Samples per modality: {len(resampled[0]['timestamps_ms'])}")


if __name__ == "__main__":
    main()
