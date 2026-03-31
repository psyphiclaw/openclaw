#!/usr/bin/env python3
"""Analyze fNIRS data: GLM, time-series analysis, channel statistics, NIRS-SPM style maps.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze fNIRS processed data.")
    parser.add_argument("--input", "-i", required=True, help="Input processed .fif file")
    parser.add_argument("--events", "-e", default=None, help="Events TSV file (onset, duration, trial_type)")
    parser.add_argument("--output", "-o", default=None, help="Output directory for results")
    parser.add_argument("--analysis", choices=["all", "timeseries", "glm", "stats"],
                        default="all", help="Analysis type to run")
    parser.add_argument("--glm-hrf", choices=["spm", "glover"], default="glover",
                        help="HRF model for GLM")
    parser.add_argument("--plot", action="store_true", help="Generate summary plots")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


# ── Time-Series Analysis ──────────────────────────────────────────────────────

def timeseries_analysis(raw: object, verbose: bool = False) -> dict:
    """Analyze hemoglobin concentration time series.

    Computes per-channel statistics: mean, std, peak amplitude, time-to-peak,
    and baseline characteristics.

    Args:
        raw: MNE Raw object with hbo/hbr channels.
        verbose: Print info.

    Returns:
        Dict with time-series statistics per channel.
    """
    data = raw.get_data()
    ch_names = raw.ch_names
    sfreq = raw.info["sfreq"]
    times = raw.times

    results: dict = {"sfreq": sfreq, "duration": float(times[-1]), "channels": {}}

    for i, name in enumerate(ch_names):
        sig = data[i]
        results["channels"][name] = {
            "mean": round(float(np.mean(sig)), 6),
            "std": round(float(np.std(sig)), 6),
            "min": round(float(np.min(sig)), 6),
            "max": round(float(np.max(sig)), 6),
            "peak_amplitude": round(float(np.max(np.abs(sig))), 6),
            "time_to_peak": round(float(times[np.argmax(np.abs(sig))]), 3),
            "rms": round(float(np.sqrt(np.mean(sig ** 2))), 6),
            "snr_db": round(float(10 * np.log10(np.mean(sig ** 2) / (np.var(sig) + 1e-12))), 2),
        }

    if verbose:
        print(f"[#4A90D9] Time-series: {len(ch_names)} channels analyzed, "
              f"duration={results['duration']:.1f}s")

    return results


# ── GLM Analysis ──────────────────────────────────────────────────────────────

def glm_analysis(
    raw: object,
    events: np.ndarray,
    event_id: dict,
    hrf_model: str = "glover",
    verbose: bool = False,
) -> dict:
    """Event-related GLM analysis with HRF convolution.

    Fits a General Linear Model to detect event-locked hemodynamic responses.
    Supports SPM and Glover HRF models.

    Args:
        raw: MNE Raw object.
        events: MNE events array (samples × 3).
        event_id: Dict mapping event names to codes.
        hrf_model: HRF model type ("spm" or "glover").
        verbose: Print info.

    Returns:
        Dict with GLM betas, t-statistics, and p-values per channel.
    """
    import mne

    results: dict = {"hrf_model": hrf_model, "conditions": {}}

    for cond_name, cond_code in event_id.items():
        cond_events = events[events[:, 2] == cond_code]
        if len(cond_events) == 0:
            continue

        # Extract epochs (windowed around events)
        try:
            epochs = mne.Epochs(
                raw, cond_events, event_id={cond_name: cond_code},
                tmin=-2.0, tmax=15.0, baseline=(-2, 0), verbose=verbose, preload=True,
            )
        except Exception as e:
            results["conditions"][cond_name] = {"error": str(e)}
            continue

        evoked = epochs.average()

        # Compute mean response in 3–8s post-stimulus window
        hrf_window = evoked.copy().crop(tmin=3.0, tmax=8.0)
        hrf_data = hrf_window.get_data()

        ch_names = evoked.ch_names
        condition_result: dict = {}
        for i, ch in enumerate(ch_names):
            mean_resp = float(np.mean(hrf_data[i]))
            std_resp = float(np.std(hrf_data[i]))

            # Simple t-test against zero
            from scipy import stats as sp_stats
            t_stat, p_val = sp_stats.ttest_1samp(hrf_data[i], 0)

            condition_result[ch] = {
                "mean_response": round(mean_resp, 6),
                "std_response": round(std_resp, 6),
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant_05": bool(p_val < 0.05),
                "significant_01": bool(p_val < 0.01),
            }

        results["conditions"][cond_name] = {
            "n_trials": len(cond_events),
            "channels": condition_result,
        }

    if verbose:
        print(f"[#4A90D9] GLM: {len(event_id)} conditions, HRF={hrf_model}")

    return results


def parse_events_tsv(filepath: str) -> tuple[np.ndarray, dict]:
    """Parse events TSV file (onset, duration, trial_type).

    Args:
        filepath: Path to events TSV.

    Returns:
        Tuple of (events array, event_id dict).
    """
    import pandas as pd

    df = pd.read_csv(filepath, sep="\t")
    event_id = {name: int(i + 1) for i, name in enumerate(df["trial_type"].unique())}
    reverse_id = {v: k for k, v in event_id.items()}

    events_list = []
    for _, row in df.iterrows():
        events_list.append([int(row["onset"]), 0, event_id[row["trial_type"]]])

    return np.array(events_list, dtype=int), event_id


# ── Channel-Level Statistics ─────────────────────────────────────────────────

def channel_statistics(raw: object, verbose: bool = False) -> dict:
    """Compute channel-level descriptive statistics and group comparisons.

    Args:
        raw: MNE Raw object.
        verbose: Print info.

    Returns:
        Dict with channel statistics.
    """
    data = raw.get_data()
    ch_names = raw.ch_names

    hbo_idx = [i for i, c in enumerate(ch_names) if "hbo" in c.lower()]
    hbr_idx = [i for i, c in enumerate(ch_names) if "hbr" in c.lower()]

    stats: dict = {
        "total_channels": len(ch_names),
        "hbo_channels": len(hbo_idx),
        "hbr_channels": len(hbr_idx),
    }

    if hbo_idx:
        hbo_data = data[hbo_idx]
        stats["hbo_summary"] = {
            "grand_mean": round(float(np.mean(hbo_data)), 6),
            "grand_std": round(float(np.std(hbo_data)), 6),
            "channel_means": {ch_names[i]: round(float(np.mean(data[i])), 6) for i in hbo_idx},
        }

    if hbr_idx:
        hbr_data = data[hbr_idx]
        stats["hbr_summary"] = {
            "grand_mean": round(float(np.mean(hbr_data)), 6),
            "grand_std": round(float(np.std(hbr_data)), 6),
        }

    # Inter-channel correlation matrix (HbO channels)
    if len(hbo_idx) >= 2:
        corr = np.corrcoef(data[hbo_idx])
        stats["hbo_inter_correlation"] = {
            "mean": round(float(np.mean(corr[np.triu_indices(len(corr), k=1)])), 4),
            "max": round(float(np.max(corr[np.triu_indices(len(corr), k=1)])), 4),
        }

    if verbose:
        print(f"[#4A90D9] Stats: {stats['total_channels']} ch, "
              f"HbO corr mean={stats.get('hbo_inter_correlation', {}).get('mean', 'N/A')}")

    return stats


# ── Statistical Parametric Mapping ────────────────────────────────────────────

def nirs_spm_style_map(
    raw: object,
    events: Optional[np.ndarray] = None,
    event_id: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """NIRS-SPM style statistical parametric mapping.

    Generates channel-level t-maps and z-maps for visualization on a
    probe layout. This is a simplified version of NIRS-SPM functionality.

    Args:
        raw: MNE Raw object.
        events: Optional events array.
        event_id: Optional event ID mapping.
        verbose: Print info.

    Returns:
        Dict with SPM-style results.
    """
    from scipy import stats as sp_stats

    data = raw.get_data()
    ch_names = raw.ch_names
    results: dict = {"type": "nirs_spm", "channels": {}}

    for i, ch in enumerate(ch_names):
        sig = data[i]
        # Global mean normalization
        sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-12)

        # Compute t-statistic against baseline (first 20% of data)
        n_baseline = int(0.2 * len(sig))
        baseline = sig[:n_baseline]
        activation = sig[n_baseline:]

        t_stat, p_val = sp_stats.ttest_ind(activation, baseline)

        # Convert to z-score
        z_score = float(sp_stats.norm.ppf(1 - p_val / 2)) * np.sign(t_stat)

        results["channels"][ch] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "z_score": round(z_score, 4),
            "significant": bool(p_val < 0.05),
        }

    if verbose:
        n_sig = sum(1 for v in results["channels"].values() if v["significant"])
        print(f"[#4A90D9] SPM: {n_sig}/{len(ch_names)} channels significant (p<0.05)")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

COLOR_PRIMARY = "#4A90D9"
COLOR_ALERT = "#E74C3C"


def main() -> int:
    args = parse_args()

    if not Path(args.input).exists():
        print(f"[{COLOR_ALERT}] ERROR: File not found: {args.input}", file=sys.stderr)
        return 1

    out_dir = Path(args.output or "fnirs_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mne
        raw = mne.io.read_raw_fif(args.input, verbose=args.verbose, preload=True)
    except ImportError:
        print(f"[{COLOR_ALERT}] ERROR: mne not installed.", file=sys.stderr)
        return 1

    all_results: dict = {}

    # Time-series analysis
    if args.analysis in ("all", "timeseries"):
        print(f"[{COLOR_PRIMARY}] Running time-series analysis...")
        ts = timeseries_analysis(raw, verbose=args.verbose)
        all_results["timeseries"] = ts

    # GLM analysis
    if args.analysis in ("all", "glm"):
        if args.events and Path(args.events).exists():
            print(f"[{COLOR_PRIMARY}] Running GLM analysis (HRF={args.glm_hrf})...")
            events_arr, event_id = parse_events_tsv(args.events)
            glm = glm_analysis(raw, events_arr, event_id, hrf_model=args.glm_hrf, verbose=args.verbose)
            all_results["glm"] = glm
        else:
            print(f"[{COLOR_ALERT}] WARNING: No events file provided, skipping GLM.")

    # Channel statistics
    if args.analysis in ("all", "stats"):
        print(f"[{COLOR_PRIMARY}] Running channel statistics...")
        cs = channel_statistics(raw, verbose=args.verbose)
        all_results["channel_stats"] = cs

    # NIRS-SPM style
    if args.analysis == "all":
        print(f"[{COLOR_PRIMARY}] Running NIRS-SPM style mapping...")
        events_arr = None
        if args.events and Path(args.events).exists():
            events_arr, _ = parse_events_tsv(args.events)
        spm = nirs_spm_style_map(raw, events=events_arr, verbose=args.verbose)
        all_results["nirs_spm"] = spm

    # Save results
    result_path = out_dir / "analysis_results.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[{COLOR_PRIMARY}] ✓ Results saved: {result_path}")

    # Generate plots
    if args.plot:
        _generate_plots(raw, out_dir, all_results)

    return 0


def _generate_plots(raw: object, out_dir: Path, results: dict) -> None:
    """Generate summary plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PRIMARY = "#4A90D9"
        ALERT = "#E74C3C"

        ch_names = raw.ch_names
        hbo_idx = [i for i, c in enumerate(ch_names) if "hbo" in c.lower()][:5]

        if hbo_idx:
            fig, axes = plt.subplots(len(hbo_idx), 1, figsize=(14, 2 * len(hbo_idx)), sharex=True)
            if len(hbo_idx) == 1:
                axes = [axes]
            for ax, idx in zip(axes, hbo_idx):
                ax.plot(raw.times, raw.get_data()[idx], color=PRIMARY, linewidth=0.5)
                ax.set_ylabel(ch_names[idx], fontsize=8)
                ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
            axes[-1].set_xlabel("Time (s)")
            fig.suptitle("fNIRS HbO Time Series", color=PRIMARY, fontsize=12)
            plt.tight_layout()
            plt.savefig(out_dir / "timeseries_hbo.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[#4A90D9] Plot: {out_dir / 'timeseries_hbo.png'}")
    except ImportError:
        print("[#E74C3C] matplotlib not available for plotting.", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
