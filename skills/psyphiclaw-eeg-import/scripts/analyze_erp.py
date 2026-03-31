#!/usr/bin/env python3
"""ERP analysis: component extraction, time-window amplitude, topomaps, condition comparison."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger("psyphiclaw-eeg-erp")

PRIMARY = "#4A90D9"
SECONDARY = "#E74C3C"

# Standard ERP component time windows (ms)
ERP_COMPONENTS = {
    "N1": (80, 150),
    "P1": (80, 130),
    "N170": (150, 220),
    "P2": (150, 280),
    "N2": (200, 350),
    "P3": (300, 500),
    "LPC": (500, 800),
    "N400": (300, 500),
}


def _load_epochs(file_path: str) -> mne.Epochs:
    """Load epochs from .fif file or create from raw."""
    path = Path(file_path)
    if path.suffix.lower() == ".fif":
        return mne.read_epochs(str(path), verbose="INFO")
    # Try loading as raw first
    ext = path.suffix.lower()
    if ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(path), preload=True, verbose="INFO")
    elif ext in (".mff", ".raw"):
        raw = mne.io.read_raw_egi(str(path), preload=True, verbose="INFO")
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="INFO")
    else:
        raise ValueError(f"Cannot load: {path}")
    raw.filter(0.1, 40.0, verbose="INFO")
    events, event_id = mne.events_from_annotations(raw, verbose="INFO")
    return mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=1.0,
                      baseline=(-0.2, 0), preload=True, verbose="INFO")


def _find_eeg_channels(epochs: mne.Epochs) -> list[str]:
    """Get EEG channel names."""
    picks = mne.pick_types(epochs.info, eeg=True)
    return [epochs.ch_names[i] for i in picks]


def extract_components(epochs: mne.Epochs, components: Optional[dict] = None,
                       ch_names: Optional[list[str]] = None) -> dict:
    """Extract ERP component mean amplitudes within time windows."""
    comps = components or ERP_COMPONENTS
    times = epochs.times * 1000  # to ms

    if ch_names is None:
        eeg_picks = mne.pick_types(epochs.info, eeg=True)
        if len(eeg_picks) == 0:
            raise ValueError("No EEG channels found")
        # Use common midline electrodes
        default_chs = ["Pz", "Cz", "Fz", "Oz"]
        ch_names = [c for c in default_chs if c in epochs.ch_names]
        if not ch_names:
            ch_names = [epochs.ch_names[eeg_picks[0]]]

    results = {}
    for comp_name, (tmin_ms, tmax_ms) in comps.items():
        # Check if time window is within epoch range
        tmin_s = tmin_ms / 1000.0
        tmax_s = tmax_ms / 1000.0
        if tmin_s < epochs.tmin or tmax_s > epochs.tmax:
            continue

        mask = (times >= tmin_ms) & (times <= tmax_ms)
        results[comp_name] = {"time_window_ms": (tmin_ms, tmax_ms)}

        for ch in ch_names:
            pick_idx = epochs.ch_names.index(ch)
            evoked_data = epochs.get_data()[:, pick_idx, :]
            mean_amp = np.mean(evoked_data[:, mask], axis=1)
            results[comp_name][ch] = {
                "mean_amplitude_uv": float(np.mean(mean_amp) * 1e6),
                "std_uv": float(np.std(mean_amp) * 1e6),
                "n_epochs": len(mean_amp),
            }

    return results


def time_window_amplitude(epochs: mne.Epochs, tmin_ms: float, tmax_ms: float,
                          conditions: Optional[list[str]] = None) -> dict:
    """Compute mean amplitude in a custom time window per condition."""
    times = epochs.times * 1000
    mask = (times >= tmin_ms) & (times <= tmax_ms)
    if not np.any(mask):
        raise ValueError(f"Time window {tmin_ms}-{tmax_ms}ms outside epoch range")

    conditions = conditions or list(epochs.event_id.keys())
    results = {"time_window_ms": (tmin_ms, tmax_ms)}

    eeg_picks = mne.pick_types(epochs.info, eeg=True)
    if len(eeg_picks) == 0:
        raise ValueError("No EEG channels")

    for cond in conditions:
        if cond not in epochs.event_id:
            continue
        evoked = epochs[cond].average()
        data = evoked.get_data(eeg_picks)[:, mask] * 1e6  # to µV
        results[cond] = {
            "mean_uv": float(np.mean(data)),
            "std_uv": float(np.std(data)),
            "grand_mean_uv": float(np.mean(evoked.get_data(eeg_picks)[:, mask], axis=(0, 1))),
        }

    return results


def compute_topomap(epochs: mne.Epochs, tmin_ms: float, tmax_ms: float,
                    condition: Optional[str] = None) -> tuple[np.ndarray, list]:
    """Compute ERP topomap data for a time window."""
    if condition and condition in epochs.event_id:
        evoked = epochs[condition].average()
    else:
        evoked = epochs.average()

    times = evoked.times * 1000
    mask = (times >= tmin_ms) & (times <= tmax_ms)
    if not np.any(mask):
        raise ValueError("Time window outside epoch range")

    eeg_picks = mne.pick_types(evoked.info, eeg=True)
    data = evoked.get_data(eeg_picks)[:, mask]
    mean_data = np.mean(data, axis=1)

    ch_names = [evoked.ch_names[i] for i in eeg_picks]
    return mean_data, ch_names


def compare_conditions(epochs: mne.Epochs, condition_a: str, condition_b: str,
                       ch: str = "Pz", tmin_ms: float = 300, tmax_ms: float = 500) -> dict:
    """Compare ERP between two conditions at a specific channel and time window."""
    for cond in (condition_a, condition_b):
        if cond not in epochs.event_id:
            raise ValueError(f"Condition '{cond}' not found. Available: {list(epochs.event_id.keys())}")

    if ch not in epochs.ch_names:
        raise ValueError(f"Channel '{ch}' not found")

    times = epochs.times * 1000
    mask = (times >= tmin_ms) & (times <= tmax_ms)
    pick = epochs.ch_names.index(ch)

    amps_a = epochs[condition_a].get_data()[:, pick, mask].mean(axis=1) * 1e6
    amps_b = epochs[condition_b].get_data()[:, pick, mask].mean(axis=1) * 1e6

    # Simple paired t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(amps_a, amps_b)
    cohen_d = (np.mean(amps_a) - np.mean(amps_b)) / np.std(amps_a - amps_b) if np.std(amps_a - amps_b) > 0 else 0

    return {
        "channel": ch,
        "time_window_ms": (tmin_ms, tmax_ms),
        condition_a: {"mean_uv": float(np.mean(amps_a)), "std_uv": float(np.std(amps_a)), "n": len(amps_a)},
        condition_b: {"mean_uv": float(np.mean(amps_b)), "std_uv": float(np.std(amps_b)), "n": len(amps_b)},
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohen_d": float(cohen_d),
        "significant_05": p_value < 0.05,
    }


def format_results(results: dict) -> str:
    """Format ERP results as readable text."""
    lines = ["=" * 55, "🧠 ERP Analysis Results", "=" * 55]
    for comp, data in results.items():
        if isinstance(data, dict) and "time_window_ms" in data:
            tmin, tmax = data["time_window_ms"]
            lines.append(f"\n  {comp} ({tmin}-{tmax} ms):")
            for ch, vals in data.items():
                if ch == "time_window_ms":
                    continue
                if isinstance(vals, dict) and "mean_amplitude_uv" in vals:
                    lines.append(f"    {ch}: {vals['mean_amplitude_uv']:.2f} ± {vals['std_uv']:.2f} µV (n={vals['n_epochs']})")
                elif isinstance(vals, dict) and "mean_uv" in vals:
                    lines.append(f"    {ch}: {vals['mean_uv']:.2f} ± {vals['std_uv']:.2f} µV")
    lines.append("=" * 55)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ERP analysis")
    parser.add_argument("file_path", type=str, help="Path to .fif epochs or raw EEG file")
    parser.add_argument("--erp-components", action="store_true", help="Extract standard ERP components")
    parser.add_argument("--time-window", nargs=2, type=float, metavar=("TMIN_MS", "TMAX_MS"),
                        help="Custom time window analysis (ms)")
    parser.add_argument("--conditions", nargs="+", help="Condition names for analysis")
    parser.add_argument("--compare", nargs=2, metavar=("COND_A", "COND_B"), help="Compare two conditions")
    parser.add_argument("--channel", type=str, default="Pz", help="Channel for comparison")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    epochs = _load_epochs(args.file_path)

    all_results = {}

    if args.erp_components:
        eeg_chs = [c for c in epochs.ch_names if c in ("Pz", "Cz", "Fz", "Oz", "P3", "P4")]
        if not eeg_chs:
            picks = mne.pick_types(epochs.info, eeg=True)
            eeg_chs = [epochs.ch_names[i] for i in picks[:3]]
        comp_results = extract_components(epochs, ch_names=eeg_chs)
        all_results.update(comp_results)
        print(format_results(comp_results))

    if args.time_window:
        tw_results = time_window_amplitude(epochs, args.time_window[0], args.time_window[1], args.conditions)
        all_results["custom_window"] = tw_results
        print(format_results({"custom_window": tw_results}))

    if args.compare:
        tmin = args.time_window[0] if args.time_window else 300
        tmax = args.time_window[1] if args.time_window else 500
        cmp = compare_conditions(epochs, args.compare[0], args.compare[1], args.channel, tmin, tmax)
        all_results["comparison"] = cmp
        print(f"\n  Comparison ({args.compare[0]} vs {args.compare[1]}):")
        print(f"    Channel: {cmp['channel']}")
        print(f"    {args.compare[0]}: {cmp[args.compare[0]]['mean_uv']:.2f} ± {cmp[args.compare[0]]['std_uv']:.2f} µV")
        print(f"    {args.compare[1]}: {cmp[args.compare[1]]['mean_uv']:.2f} ± {cmp[args.compare[1]]['std_uv']:.2f} µV")
        print(f"    t={cmp['t_statistic']:.3f}, p={cmp['p_value']:.4f}, d={cmp['cohen_d']:.3f}")
        sig = "✅ Significant" if cmp["significant_05"] else "❌ Not significant"
        print(f"    {sig} (α=0.05)")

    if args.output:
        import json
        Path(args.output).write_text(json.dumps(all_results, indent=2, default=str))
        logger.info("Results saved to %s", args.output)

    print("\n✅ ERP analysis complete")


if __name__ == "__main__":
    main()
