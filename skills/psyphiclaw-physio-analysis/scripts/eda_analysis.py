#!/usr/bin/env python3
"""Electrodermal Activity (EDA) Analysis.

Decomposes EDA into tonic (SCL) and phasic (SCR) components,
detects SCR peaks, and supports event-locked SCR analysis.
Outputs JSON statistics and PNG/HTML charts.

Typical usage:
    python eda_analysis.py --input eda.csv --fs 100 --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

# Color scheme
PRIMARY = "#4A90D9"
SECONDARY = "#E74C3C"
ACCENT = "#27AE60"
BG_COLOR = "#FAFAFA"


# ---------------------------------------------------------------------------
# Signal decomposition
# ---------------------------------------------------------------------------

def decompose_eda(eda: np.ndarray, fs: float, lpf_cutoff: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Decompose EDA into tonic (SCL) and phasic (SCR) components.

    Parameters
    ----------
    eda : array-like
        Raw EDA signal in micro-Siemens (μS).
    fs : float
        Sampling rate in Hz.
    lpf_cutoff : float
        Low-pass cutoff for tonic extraction (Hz). Default 0.05 Hz.

    Returns
    -------
    tonic : np.ndarray
        Skin Conductance Level (SCL) — the slow-varying tonic component.
    phasic : np.ndarray
        Skin Conductance Response (SCR) — the fast phasic component.
    """
    eda = np.asarray(eda, dtype=float)

    # Tonic: low-pass Butterworth at 0.05 Hz (4th order)
    b_tonic, a_tonic = signal.butter(4, lpf_cutoff, btype="lowpass", fs=fs)
    tonic = signal.filtfilt(b_tonic, a_tonic, eda)

    # Phasic = raw − tonic, then high-pass to remove residual drift
    phasic = eda - tonic
    b_phasic, a_phasic = signal.butter(4, lpf_cutoff, btype="highpass", fs=fs)
    phasic = signal.filtfilt(b_phasic, a_phasic, eda)
    # Ensure non-negative (SCR should be positive)
    phasic = np.maximum(phasic, 0.0)

    return tonic, phasic


# ---------------------------------------------------------------------------
# SCR peak detection
# ---------------------------------------------------------------------------

@dataclass
class SCRPeak:
    """A detected SCR peak."""
    onset_idx: int
    peak_idx: int
    offset_idx: int
    onset_time_s: float
    peak_time_s: float
    offset_time_s: float
    amplitude_us: float  # μS
    rise_time_s: float
    recovery_time_s: float


def detect_scr_peaks(
    phasic: np.ndarray,
    fs: float,
    onset_thresh: float = 0.02,
    peak_thresh: float = 0.05,
    min_rise: float = 0.5,
    max_duration: float = 5.0,
) -> list[SCRPeak]:
    """Detect SCR peaks in the phasic component.

    Parameters
    ----------
    phasic : array-like
        Phasic SCR signal (non-negative, in μS).
    fs : float
        Sampling rate in Hz.
    onset_thresh : float
        Minimum onset amplitude (μS).
    peak_thresh : float
        Minimum peak amplitude (μS).
    min_rise : float
        Minimum rise time in seconds (s).
    max_duration : float
        Maximum event duration in seconds.

    Returns
    -------
    List of SCRPeak.
    """
    phasic = np.asarray(phasic, dtype=float)
    min_dist = int(min_rise * fs)
    max_dur_samples = int(max_duration * fs)

    # Find peaks above threshold
    peak_indices, _ = signal.find_peaks(phasic, height=peak_thresh, distance=min_dist)

    peaks: list[SCRPeak] = []
    for pi in peak_indices:
        amplitude = float(phasic[pi])
        # Find onset: first point before peak where signal drops below onset_thresh
        onset = pi
        for k in range(pi - 1, max(pi - max_dur_samples, -1), -1):
            if phasic[k] < onset_thresh:
                onset = k + 1
                break
        else:
            onset = max(0, pi - max_dur_samples)

        # Find offset: first point after peak where signal drops below onset_thresh
        offset = pi
        for k in range(pi + 1, min(pi + max_dur_samples, len(phasic))):
            if phasic[k] < onset_thresh:
                offset = k
                break
        else:
            offset = min(len(phasic) - 1, pi + max_dur_samples)

        peaks.append(SCRPeak(
            onset_idx=int(onset),
            peak_idx=int(pi),
            offset_idx=int(offset),
            onset_time_s=onset / fs,
            peak_time_s=pi / fs,
            offset_time_s=offset / fs,
            amplitude_us=amplitude,
            rise_time_s=(pi - onset) / fs,
            recovery_time_s=(offset - pi) / fs,
        ))

    return peaks


# ---------------------------------------------------------------------------
# Event-locked SCR analysis
# ---------------------------------------------------------------------------

def event_locked_scr(
    phasic: np.ndarray,
    fs: float,
    event_times: list[float],
    pre: float = 1.0,
    post: float = 4.0,
) -> dict[str, float]:
    """Compute event-locked SCR statistics.

    Parameters
    ----------
    phasic : np.ndarray
        Phasic SCR signal.
    fs : float
        Sampling rate in Hz.
    event_times : list[float]
        Event marker times in seconds.
    pre : float
        Pre-event window in seconds.
    post : float
        Post-event window in seconds.

    Returns
    -------
    dict with mean_peak, max_peak, mean_latency, n_epochs, mean_auc.
    """
    if not event_times:
        return {"n_epochs": 0, "mean_peak_us": 0.0, "max_peak_us": 0.0, "mean_latency_s": 0.0, "mean_auc_us_s": 0.0}

    epoch_peak_vals: list[float] = []
    epoch_latencies: list[float] = []
    epoch_aucs: list[float] = []

    for et in event_times:
        start = int((et - pre) * fs)
        end = int((et + post) * fs)
        if start < 0 or end >= len(phasic) or end <= start:
            continue
        epoch = phasic[start:end]
        # Find peak in post-event window
        post_start = int(pre * fs)
        if post_start >= len(epoch):
            continue
        post_epoch = epoch[post_start:]
        if len(post_epoch) == 0:
            continue
        pk_val = float(np.max(post_epoch))
        pk_idx = int(np.argmax(post_epoch))
        epoch_peak_vals.append(pk_val)
        epoch_latencies.append(pk_idx / fs)
        # AUC of post-epoch
        epoch_aucs.append(float(np.trapz(post_epoch, dx=1.0 / fs)))

    if not epoch_peak_vals:
        return {"n_epochs": 0, "mean_peak_us": 0.0, "max_peak_us": 0.0, "mean_latency_s": 0.0, "mean_auc_us_s": 0.0}

    return {
        "n_epochs": len(epoch_peak_vals),
        "mean_peak_us": float(np.mean(epoch_peak_vals)),
        "max_peak_us": float(np.max(epoch_peak_vals)),
        "mean_latency_s": float(np.mean(epoch_latencies)),
        "std_latency_s": float(np.std(epoch_latencies)),
        "mean_auc_us_s": float(np.mean(epoch_aucs)),
        "std_auc_us_s": float(np.std(epoch_aucs)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eda(
    eda: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    peaks: list[SCRPeak],
    fs: float,
    output_dir: Path,
    event_times: list[float] | None = None,
) -> None:
    """Generate PNG and HTML plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t = np.arange(len(eda)) / fs

    fig = plt.figure(figsize=(18, 12), facecolor=BG_COLOR)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # 1. Raw EDA with decomposition
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, eda, color="gray", linewidth=0.6, alpha=0.7, label="Raw EDA")
    ax1.plot(t, tonic, color=PRIMARY, linewidth=1.2, label="Tonic (SCL)")
    ax1.plot(t, phasic + np.mean(tonic), color=SECONDARY, linewidth=0.8, alpha=0.8, label="Phasic (SCR) offset")
    if event_times:
        for et in event_times:
            ax1.axvline(et, color=ACCENT, linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title("EDA Decomposition", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Conductance (μS)")
    ax1.legend(fontsize=9)

    # 2. Phasic SCR with peaks
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, phasic, color=PRIMARY, linewidth=0.8)
    for pk in peaks:
        ax2.plot(pk.peak_time_s, pk.amplitude_us, "v", color=SECONDARY, markersize=7)
    ax2.set_title(f"Phasic SCR ({len(peaks)} peaks detected)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("SCR Amplitude (μS)")

    # 3. Tonic SCL
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, tonic, color=PRIMARY, linewidth=1.2)
    ax3.fill_between(t, tonic, alpha=0.15, color=PRIMARY)
    ax3.set_title("Tonic Skin Conductance Level", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("SCL (μS)")

    # 4. Peak amplitude histogram
    ax4 = fig.add_subplot(gs[2, 0])
    if peaks:
        amplitudes = [p.amplitude_us for p in peaks]
        ax4.hist(amplitudes, bins=20, color=PRIMARY, edgecolor="white", alpha=0.85)
        ax4.axvline(np.mean(amplitudes), color=SECONDARY, linestyle="--", linewidth=1.2,
                    label=f"Mean={np.mean(amplitudes):.3f}")
        ax4.legend(fontsize=9)
    ax4.set_title("SCR Peak Amplitudes", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Amplitude (μS)")
    ax4.set_ylabel("Count")

    # 5. Inter-peak interval histogram
    ax5 = fig.add_subplot(gs[2, 1])
    if len(peaks) > 1:
        ipis = [peaks[i + 1].onset_time_s - peaks[i].onset_time_s for i in range(len(peaks) - 1)]
        ax5.hist(ipis, bins=20, color=SECONDARY, edgecolor="white", alpha=0.85)
    ax5.set_title("Inter-SCR Intervals", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Interval (s)")
    ax5.set_ylabel("Count")

    plt.savefig(output_dir / "eda_analysis.png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

    # HTML
    html = _build_eda_html(peaks, tonic, phasic, event_times)
    (output_dir / "eda_analysis.html").write_text(html, encoding="utf-8")


def _build_eda_html(peaks: list[SCRPeak], tonic: np.ndarray, phasic: np.ndarray,
                    event_times: list[float] | None) -> str:
    amp_vals = [p.amplitude_us for p in peaks] if peaks else [0]
    summary = {
        "Total SCR Peaks": str(len(peaks)),
        "Mean Amplitude (μS)": f"{np.mean(amp_vals):.4f}",
        "Max Amplitude (μS)": f"{np.max(amp_vals):.4f}",
        "Mean Tonic SCL (μS)": f"{np.mean(tonic):.4f}",
        "Mean Phasic SCR (μS)": f"{np.mean(phasic):.4f}",
    }
    if event_times:
        summary["Event Markers"] = str(len(event_times))

    rows = "\n".join(f'<tr><td style="padding:4px 12px;color:#555">{k}</td><td style="padding:4px 12px;font-weight:600">{v}</td></tr>'
                     for k, v in summary.items())

    peak_rows = ""
    for i, p in enumerate(peaks[:50]):  # limit display
        peak_rows += f'<tr><td style="padding:2px 8px">{i+1}</td><td>{p.onset_time_s:.2f}</td><td>{p.peak_time_s:.2f}</td><td>{p.amplitude_us:.4f}</td><td>{p.rise_time_s:.2f}</td></tr>'

    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EDA Analysis Report</title>
<style>body{font-family:-apple-system,Segoe UI,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;background:#FAFAFA}
h1{color:#4A90D9;border-bottom:2px solid #E74C3C;padding-bottom:8px}
table{background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1);margin-bottom:16px}
tr:nth-child(even){background:#f4f7fb} th{background:#4A90D9;color:#fff;padding:6px 12px;text-align:left}</style></head><body>
<h1>Electrodermal Activity Analysis</h1>
<h3 style="color:#4A90D9">Summary</h3>
<table style="border-collapse:collapse;width:100%">{rows}</table>
<h3 style="color:#4A90D9">SCR Peaks (first 50)</h3>
<table><tr><th>#</th><th>Onset (s)</th><th>Peak (s)</th><th>Amplitude (μS)</th><th>Rise Time (s)</th></tr>
{peak_rows}</table>
<h3 style="color:#4A90D9">Charts</h3>
<p><img src="eda_analysis.png" style="max-width:100%;border-radius:8px"></p>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Electrodermal Activity (EDA) Analysis")
    p.add_argument("--input", "-i", required=True, type=Path, help="EDA signal file (CSV or whitespace-separated)")
    p.add_argument("--column", "-c", default=0, type=int, help="Column index (default: 0)")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz (recommended >= 100)")
    p.add_argument("--output-dir", "-o", type=Path, default=Path("eda_results"), help="Output directory")
    p.add_argument("--events", type=Path, default=None, help="File with event times (one per line, in seconds)")
    p.add_argument("--onset-thresh", type=float, default=0.02, help="SCR onset threshold μS (default: 0.02)")
    p.add_argument("--peak-thresh", type=float, default=0.05, help="SCR peak threshold μS (default: 0.05)")
    p.add_argument("--lpf-cutoff", type=float, default=0.05, help="Tonic low-pass cutoff Hz (default: 0.05)")
    p.add_argument("--json-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    if args.input.suffix == ".csv":
        df = pd.read_csv(args.input, header=None)
        eda = df.iloc[:, args.column].values
    else:
        eda = np.loadtxt(args.input)

    # Decompose
    tonic, phasic = decompose_eda(eda, args.fs, lpf_cutoff=args.lpf_cutoff)

    # Detect peaks
    peaks = detect_scr_peaks(phasic, args.fs, onset_thresh=args.onset_thresh, peak_thresh=args.peak_thresh)

    # Event-locked analysis
    event_times: list[float] = []
    event_stats: dict = {}
    if args.events and args.events.exists():
        event_times = np.loadtxt(args.events).tolist()
        if isinstance(event_times, float):
            event_times = [event_times]
        event_stats = event_locked_scr(phasic, args.fs, event_times)

    # Compute summary statistics
    amp_vals = [p.amplitude_us for p in peaks] if peaks else [0.0]
    rise_vals = [p.rise_time_s for p in peaks] if peaks else [0.0]
    recovery_vals = [p.recovery_time_s for p in peaks] if peaks else [0.0]

    result = {
        "metadata": {
            "input_file": str(args.input),
            "sampling_rate_hz": args.fs,
            "duration_s": len(eda) / args.fs,
            "onset_threshold_us": args.onset_thresh,
            "peak_threshold_us": args.peak_thresh,
        },
        "signal_summary": {
            "mean_tonic_us": float(np.mean(tonic)),
            "std_tonic_us": float(np.std(tonic)),
            "mean_phasic_us": float(np.mean(phasic)),
            "std_phasic_us": float(np.std(phasic)),
        },
        "scr_peaks": {
            "n_peaks": len(peaks),
            "mean_amplitude_us": float(np.mean(amp_vals)),
            "max_amplitude_us": float(np.max(amp_vals)),
            "min_amplitude_us": float(np.min(amp_vals)) if amp_vals else 0.0,
            "std_amplitude_us": float(np.std(amp_vals)),
            "mean_rise_time_s": float(np.mean(rise_vals)),
            "mean_recovery_time_s": float(np.mean(recovery_vals)),
            "peaks_per_minute": len(peaks) / (len(eda) / args.fs / 60) if len(eda) > 0 else 0.0,
        },
        "event_locked": event_stats,
    }

    # Save JSON
    json_path = args.output_dir / "eda_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] Statistics saved to {json_path}")

    if not args.json_only:
        plot_eda(eda, tonic, phasic, peaks, args.fs, args.output_dir, event_times if event_times else None)
        print(f"[OK] Plots saved to {args.output_dir}/eda_analysis.png and .html")

    print(f"\n{'='*40}")
    print(f"EDA Analysis Summary")
    print(f"{'='*40}")
    print(f"  Duration:        {len(eda)/args.fs:.1f} s")
    print(f"  Mean Tonic SCL:  {np.mean(tonic):.4f} μS")
    print(f"  SCR Peaks:       {len(peaks)}")
    print(f"  Mean Amplitude:  {np.mean(amp_vals):.4f} μS")
    print(f"  Peaks/min:       {result['scr_peaks']['peaks_per_minute']:.1f}")
    if event_stats:
        print(f"  Event-locked:    {event_stats.get('n_epochs', 0)} epochs")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
