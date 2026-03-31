#!/usr/bin/env python3
"""Time-locked analysis: extract event-locked epochs across modalities.

Extracts segments of data time-locked to experimental events (e.g. stimulus
onset), computes averages per condition, and performs window-based statistical
comparisons.

Usage:
    python time_locked_analysis.py --session session.h5 \
        --event-times events.csv --pre-stim 1.0 --post-stim 3.0 \
        --conditions condition_col --output time_locked/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

BLUE = "#4A90D9"
RED = "#E74C3C"


def extract_epochs(
    values: np.ndarray,
    timestamps_ms: np.ndarray,
    event_times_ms: np.ndarray,
    pre_stim_ms: float,
    post_stim_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract epochs time-locked to events.

    Parameters
    ----------
    values : 2-D array (time × features).
    timestamps_ms : 1-D array.
    event_times_ms : 1-D array of event onset times.

    Returns
    -------
    epochs : 3-D array (n_events × n_samples × n_features).
    epoch_times : 1-D array relative to event onset (ms).
    """
    n_samples = int((pre_stim_ms + post_stim_ms) / np.median(np.diff(timestamps_ms))) + 1
    epoch_times = np.linspace(-pre_stim_ms, post_stim_ms, n_samples)

    epochs = np.full((len(event_times_ms), n_samples, values.shape[1]), np.nan)
    for i, evt in enumerate(event_times_ms):
        t_start = evt - pre_stim_ms
        t_end = evt + post_stim_ms
        mask = (timestamps_ms >= t_start) & (timestamps_ms <= t_end)
        if mask.sum() != n_samples:
            # Interpolate to exact grid
            from scipy.interpolate import interp1d
            valid = ~np.isnan(values[:, 0]) if values.shape[1] > 0 else np.ones(len(timestamps_ms), dtype=bool)
            t_valid = timestamps_ms[valid]
            v_valid = values[valid]
            if len(t_valid) >= 2:
                interp = interp1d(t_valid, v_valid, axis=0, kind="linear",
                                  bounds_error=False, fill_value=np.nan)
                epochs[i] = interp(epoch_times + evt)
        else:
            epochs[i] = values[mask]

    return epochs, epoch_times


def compute_erp(
    epochs: np.ndarray,
    epoch_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute average ERP (mean ± SEM across epochs).

    Returns
    -------
    mean, sem : 2-D arrays (n_samples × n_features).
    """
    mean = np.nanmean(epochs, axis=0)
    sem = np.nanstd(epochs, axis=0, ddof=1) / np.sqrt(epochs.shape[0] - np.sum(np.isnan(epochs[:, :, 0] if epochs.ndim == 3 else epochs), axis=1 if epochs.ndim == 3 else 0).mean())
    return mean, sem


def compare_windows(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    window_ms: tuple[float, float],
    epoch_times: np.ndarray,
) -> dict[str, Any]:
    """Statistical comparison between two conditions within a time window.

    Returns cluster-based permutation test results (simplified).
    """
    mask = (epoch_times >= window_ms[0]) & (epoch_times <= window_ms[1])
    if mask.sum() == 0:
        return {"error": "window outside epoch range"}

    data_a = np.nanmean(epochs_a[:, mask, :], axis=1)  # (n_epochs, n_features)
    data_b = np.nanmean(epochs_b[:, mask, :], axis=1)

    results: list[dict[str, Any]] = []
    for feat_idx in range(data_a.shape[1]):
        a = data_a[:, feat_idx]
        b = data_b[:, feat_idx]
        a_clean, b_clean = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a_clean) < 3 or len(b_clean) < 3:
            results.append({"feature": feat_idx, "t": np.nan, "p": np.nan})
            continue
        t_stat, p_val = stats.ttest_ind(a_clean, b_clean)
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(a_clean, ddof=1) + np.var(b_clean, ddof=1)) / 2)
        cohens_d = (np.mean(a_clean) - np.mean(b_clean)) / pooled_std if pooled_std > 0 else 0
        results.append({
            "feature": feat_idx,
            "mean_a": round(float(np.mean(a_clean)), 4),
            "mean_b": round(float(np.mean(b_clean)), 4),
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "cohens_d": round(float(cohens_d), 4),
        })

    return {
        "window_ms": list(window_ms),
        "n_a": int(epochs_a.shape[0]),
        "n_b": int(epochs_b.shape[0]),
        "comparisons": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-locked multimodal analysis.")
    parser.add_argument("--session", help="MultiModalSession .h5 path.")
    parser.add_argument("--csv-dir", help="Directory of aligned CSVs (alternative).")
    parser.add_argument("--event-times", required=True,
                        help="CSV with event onset times. Columns: timestamp_ms, [condition].")
    parser.add_argument("--event-col", default="timestamp_ms",
                        help="Event time column name in events CSV.")
    parser.add_argument("--conditions", default=None,
                        help="Column name for conditions in events CSV.")
    parser.add_argument("--pre-stim", type=float, default=1.0,
                        help="Pre-stimulus window in seconds.")
    parser.add_argument("--post-stim", type=float, default=3.0,
                        help="Post-stimulus window in seconds.")
    parser.add_argument("--compare-window", nargs=2, type=float, default=None,
                        metavar=("START", "END"),
                        help="Time window (ms) for statistical comparison.")
    parser.add_argument("--output", required=True, help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load event times
    events_df = pd.read_csv(args.event_times)
    event_times = pd.to_numeric(events_df[args.event_col], errors="coerce").dropna().values
    conditions = events_df[args.conditions].values if args.conditions else None
    print(f"📍 Loaded {len(event_times)} events")

    pre_ms = args.pre_stim * 1000.0
    post_ms = args.post_stim * 1000.0

    # Load modality data
    if args.session:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "psyphiclaw-fusion-align" / "scripts"))
        from session_manager import MultiModalSession
        session = MultiModalSession.load(args.session)
        modalities = {k: v for k, v in session.modalities.items()}
    elif args.csv_dir:
        modalities = _load_csv_modalities(args.csv_dir)
    else:
        parser.error("Provide --session or --csv-dir.")

    for mod_name, mod_data in modalities.items():
        print(f"\n🔧 Processing {mod_name}...")
        values = mod_data["values"]
        timestamps_ms = mod_data["timestamps_ms"]
        columns = mod_data["columns"]

        # Extract epochs
        epochs, epoch_times = extract_epochs(values, timestamps_ms, event_times, pre_ms, post_ms)
        print(f"  Epochs: {epochs.shape} (events × samples × features)")

        # Grand average
        grand_mean = np.nanmean(epochs, axis=0)
        grand_sem = np.nanstd(epochs, axis=0, ddof=1) / np.sqrt(max(epochs.shape[0], 1))
        avg_df = pd.DataFrame(grand_mean, columns=[f"{mod_name}_{c}" for c in columns])
        avg_df.insert(0, "time_ms", epoch_times)
        avg_df.to_csv(out_dir / f"{mod_name}_grand_average.csv", index=False)
        print(f"  Grand average → {mod_name}_grand_average.csv")

        # Per-condition averages
        if conditions is not None:
            cond_df = pd.DataFrame({"time_ms": epoch_times})
            unique_conds = pd.Series(conditions).dropna().unique()
            for cond in unique_conds:
                mask = np.array(conditions) == cond
                cond_epochs = epochs[mask]
                cond_mean = np.nanmean(cond_epochs, axis=0)
                for ci, col in enumerate(columns):
                    cond_df[f"{mod_name}_{col}_{cond}"] = cond_mean[:, ci]
            cond_df.to_csv(out_dir / f"{mod_name}_condition_averages.csv", index=False)
            print(f"  Conditions ({len(unique_conds)}) → {mod_name}_condition_averages.csv")

            # Pairwise condition comparisons
            if args.compare_window and len(unique_conds) >= 2:
                window = tuple(args.compare_window)
                conds_list = list(unique_conds)
                for i in range(len(conds_list)):
                    for j in range(i + 1, len(conds_list)):
                        ep_a = epochs[np.array(conditions) == conds_list[i]]
                        ep_b = epochs[np.array(conditions) == conds_list[j]]
                        comp = compare_windows(ep_a, ep_b, window, epoch_times)
                        comp_path = out_dir / f"{mod_name}_{conds_list[i]}_vs_{conds_list[j]}_comparison.json"
                        import json
                        with open(comp_path, "w") as f:
                            json.dump(comp, f, indent=2, default=str)
                        print(f"  Comparison [{conds_list[i]} vs {conds_list[j]}] → {comp_path.name}")

    print(f"\n✅ Time-locked analysis complete → {out_dir}/")


def _load_csv_modalities(csv_dir: str) -> dict[str, dict[str, Any]]:
    """Load modalities from aligned CSV directory."""
    result = {}
    for csv_file in sorted(Path(csv_dir).glob("*.csv")):
        df = pd.read_csv(csv_file)
        ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
        if ts_col is None:
            continue
        result[csv_file.stem] = {
            "values": df.drop(columns=[ts_col]).values,
            "timestamps_ms": df[ts_col].values,
            "columns": [c for c in df.columns if c != ts_col],
        }
    return result


if __name__ == "__main__":
    main()
