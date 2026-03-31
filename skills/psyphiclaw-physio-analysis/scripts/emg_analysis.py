#!/usr/bin/env python3
"""Electromyography (EMG) Analysis.

Performs band-pass filtering, rectification, RMS envelope extraction,
onset detection, and median-frequency-based fatigue analysis.
Outputs JSON statistics and PNG/HTML charts.

Typical usage:
    python emg_analysis.py --input emg.csv --fs 2000 --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
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
# Preprocessing
# ---------------------------------------------------------------------------

def bandpass_filter(
    emg: np.ndarray,
    fs: float,
    low: float = 20.0,
    high: float = 450.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter to EMG signal.

    Parameters
    ----------
    emg : array-like
        Raw EMG signal.
    fs : float
        Sampling rate in Hz.
    low : float
        Low cutoff frequency in Hz. Default 20.
    high : float
        High cutoff frequency in Hz. Default 450.
    order : int
        Filter order. Default 4.

    Returns
    -------
    np.ndarray : Filtered EMG signal.
    """
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = signal.butter(order, [low_norm, high_norm], btype="band")
    return signal.filtfilt(b, a, np.asarray(emg, dtype=float))


def rectify(emg: np.ndarray) -> np.ndarray:
    """Full-wave rectification."""
    return np.abs(np.asarray(emg, dtype=float))


def smooth_envelope(emg_rect: np.ndarray, fs: float, cutoff: float = 10.0, order: int = 4) -> np.ndarray:
    """Low-pass filter to produce RMS-like envelope.

    Parameters
    ----------
    emg_rect : array-like
        Rectified EMG signal.
    fs : float
        Sampling rate in Hz.
    cutoff : float
        Low-pass cutoff in Hz. Default 10 Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray : Smoothed envelope.
    """
    b, a = signal.butter(order, cutoff, btype="lowpass", fs=fs)
    return signal.filtfilt(b, a, np.asarray(emg_rect, dtype=float))


def compute_rms(emg: np.ndarray, window_s: float, fs: float) -> np.ndarray:
    """Compute running RMS over a sliding window.

    Parameters
    ----------
    emg : array-like
        Filtered EMG signal.
    window_s : float
        Window duration in seconds.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray : RMS envelope (same length, edges zero-padded).
    """
    emg = np.asarray(emg, dtype=float)
    win = int(window_s * fs)
    if win < 1:
        win = 1
    kernel = np.ones(win) / win
    squared = emg ** 2
    rms = np.sqrt(np.convolve(squared, kernel, mode="same"))
    return rms


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def detect_onsets(
    envelope: np.ndarray,
    fs: float,
    z_thresh: float = 3.0,
    min_duration_s: float = 0.05,
    min_interval_s: float = 0.1,
) -> list[dict]:
    """Detect muscle activation onsets using z-score thresholding.

    Parameters
    ----------
    envelope : array-like
        RMS envelope signal.
    fs : float
        Sampling rate in Hz.
    z_thresh : float
        Z-score threshold for onset detection. Default 3.0.
    min_duration_s : float
        Minimum activation duration in seconds.
    min_interval_s : float
        Minimum interval between onsets in seconds.

    Returns
    -------
    List of dicts with keys: onset_idx, offset_idx, onset_time, offset_time, duration, peak_amplitude.
    """
    envelope = np.asarray(envelope, dtype=float)
    min_dur = int(min_duration_s * fs)
    min_int = int(min_interval_s * fs)

    # Baseline from first 1 second or 10% of signal
    baseline_len = max(1, min(int(1.0 * fs), len(envelope) // 10))
    baseline = envelope[:baseline_len]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    if baseline_std < 1e-10:
        baseline_std = np.std(envelope)
    if baseline_std < 1e-10:
        return []

    threshold = baseline_mean + z_thresh * baseline_std
    above = envelope > threshold

    # Find contiguous regions above threshold
    onsets: list[dict] = []
    in_active = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_active:
            start = i
            in_active = True
        elif not above[i] and in_active:
            if i - start >= min_dur:
                if not onsets or start - onsets[-1]["onset_idx"] >= min_int:
                    onsets.append({
                        "onset_idx": int(start),
                        "offset_idx": int(i),
                        "onset_time": float(start / fs),
                        "offset_time": float(i / fs),
                        "duration_s": float((i - start) / fs),
                        "peak_amplitude": float(np.max(envelope[start:i])),
                    })
            in_active = False

    return onsets


# ---------------------------------------------------------------------------
# Median frequency (fatigue analysis)
# ---------------------------------------------------------------------------

def median_frequency(
    emg: np.ndarray,
    fs: float,
    window_s: float = 0.5,
    overlap: float = 0.5,
    fmin: float = 20.0,
    fmax: float = 450.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median frequency over sliding windows for fatigue analysis.

    Parameters
    ----------
    emg : array-like
        Filtered EMG signal.
    fs : float
        Sampling rate in Hz.
    window_s : float
        Window duration in seconds.
    overlap : float
        Overlap fraction (0–1).
    fmin, fmax : float
        Frequency range for PSD computation.

    Returns
    -------
    times : np.ndarray — window center times in seconds.
    med_freqs : np.ndarray — median frequency for each window (Hz).
    mean_freqs : np.ndarray — mean frequency for each window (Hz).
    """
    emg = np.asarray(emg, dtype=float)
    win = int(window_s * fs)
    step = int(win * (1 - overlap))
    if win < 1 or step < 1:
        raise ValueError("Window too short for given fs.")

    times: list[float] = []
    med_freqs: list[float] = []
    mean_freqs: list[float] = []

    for start in range(0, len(emg) - win + 1, step):
        segment = emg[start:start + win]
        nperseg = min(256, len(segment) // 2)
        if nperseg < 16:
            nperseg = max(16, len(segment) // 2)
        freqs, psd = signal.welch(segment, fs=fs, nperseg=nperseg)

        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            continue
        f_masked = freqs[mask]
        p_masked = psd[mask]
        cum_power = np.cumsum(p_masked)
        total = cum_power[-1]
        if total <= 0:
            continue

        # Median frequency
        med_idx = np.searchsorted(cum_power, total / 2)
        med_idx = min(med_idx, len(f_masked) - 1)
        med_freqs.append(float(f_masked[med_idx]))

        # Mean frequency
        mean_f = float(np.sum(f_masked * p_masked) / total)
        mean_freqs.append(mean_f)

        times.append(float((start + win / 2) / fs))

    return np.array(times), np.array(med_freqs), np.array(mean_freqs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_emg(
    emg_raw: np.ndarray,
    emg_filt: np.ndarray,
    envelope: np.ndarray,
    rms: np.ndarray,
    onsets: list[dict],
    mdf_times: np.ndarray,
    mdf: np.ndarray,
    mnf: np.ndarray,
    fs: float,
    output_dir: Path,
) -> None:
    """Generate PNG and HTML summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Downsample for plotting to avoid huge files
    max_pts = 20000
    step = max(1, len(emg_raw) // max_pts)
    t_raw = np.arange(len(emg_raw))[::step] / fs
    t_env = np.arange(len(envelope)) / fs

    fig = plt.figure(figsize=(18, 14), facecolor=BG_COLOR)
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # 1. Raw EMG
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_raw, emg_raw[::step], color="gray", linewidth=0.3, alpha=0.7, label="Raw EMG")
    ax1.plot(t_env, emg_filt[::step], color=PRIMARY, linewidth=0.3, alpha=0.8, label="Filtered")
    for o in onsets:
        ax1.axvspan(o["onset_time"], o["offset_time"], alpha=0.15, color=SECONDARY)
    ax1.set_title("Raw & Filtered EMG", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(fontsize=9, loc="upper right")

    # 2. Rectified + Envelope
    ax2 = fig.add_subplot(gs[1, 0])
    rect = np.abs(emg_filt[::step])
    ax2.plot(t_raw, rect, color=PRIMARY, linewidth=0.2, alpha=0.5, label="Rectified")
    ax2.plot(t_env, envelope, color=SECONDARY, linewidth=1.2, label="Envelope (LP)")
    ax2.plot(t_env, rms, color=ACCENT, linewidth=1.0, alpha=0.8, label="RMS")
    for o in onsets:
        ax2.axvline(o["onset_time"], color=SECONDARY, linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_title("Rectified EMG & Envelopes", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(fontsize=8)

    # 3. Onset map
    ax3 = fig.add_subplot(gs[1, 1])
    if onsets:
        y_on = [o["peak_amplitude"] for o in onsets]
        x_on = [o["onset_time"] for o in onsets]
        durs = [o["duration_s"] for o in onsets]
        ax3.barh(range(len(onsets)), durs, left=x_on, height=0.6, color=SECONDARY, alpha=0.7, edgecolor="white")
        ax3.set_yticks([])
        ax3.set_xlabel("Time (s)")
        ax3.set_title(f"Muscle Activations ({len(onsets)} onsets)", fontsize=11, fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "No onsets detected", transform=ax3.transAxes, ha="center", fontsize=12)
        ax3.set_title("Muscle Activations", fontsize=11, fontweight="bold")

    # 4. Median frequency over time (fatigue indicator)
    ax4 = fig.add_subplot(gs[2, 0])
    if len(mdf) > 1:
        ax4.plot(mdf_times, mdf, color=PRIMARY, linewidth=1.2, label="Median Freq")
        ax4.plot(mdf_times, mnf, color=SECONDARY, linewidth=1.0, alpha=0.7, label="Mean Freq")
        # Linear trend
        z = np.polyfit(mdf_times, mdf, 1)
        p = np.poly1d(z)
        ax4.plot(mdf_times, p(mdf_times), "--", color="gray", linewidth=1, alpha=0.7,
                label=f"Trend: {z[0]*60:.2f} Hz/min")
        ax4.legend(fontsize=8)
    ax4.set_title("Median Frequency (Fatigue Index)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")

    # 5. Summary stats
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    rms_vals = rms[rms > 0]
    summary_lines = [
        "EMG Analysis Summary",
        "─" * 30,
        f"Duration:       {len(emg_raw)/fs:.1f} s",
        f"Activations:    {len(onsets)}",
        f"Mean RMS:       {np.mean(rms_vals):.4f}" if len(rms_vals) > 0 else "Mean RMS:       N/A",
        f"Max RMS:        {np.max(rms_vals):.4f}" if len(rms_vals) > 0 else "Max RMS:        N/A",
    ]
    if len(mdf) > 0:
        summary_lines += [
            f"Init MDF:       {mdf[0]:.1f} Hz",
            f"Final MDF:      {mdf[-1]:.1f} Hz",
            f"MDF Δ:          {mdf[-1]-mdf[0]:+.1f} Hz",
            f"Mean MDF:       {np.mean(mdf):.1f} Hz",
        ]
    if onsets:
        summary_lines.append(f"Mean Duration:  {np.mean([o['duration_s'] for o in onsets]):.3f} s")

    ax5.text(0.05, 0.95, "\n".join(summary_lines), transform=ax5.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=PRIMARY, alpha=0.9))

    plt.savefig(output_dir / "emg_analysis.png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

    # HTML
    html = _build_emg_html(onsets, rms_vals, mdf, mnf, len(emg_raw), fs)
    (output_dir / "emg_analysis.html").write_text(html, encoding="utf-8")


def _build_emg_html(onsets: list[dict], rms_vals: np.ndarray,
                    mdf: np.ndarray, mnf: np.ndarray, n_samples: int, fs: float) -> str:
    rows = [
        ("Duration (s)", f"{n_samples/fs:.1f}"),
        ("N Activations", str(len(onsets))),
    ]
    if len(rms_vals) > 0:
        rows += [
            ("Mean RMS", f"{np.mean(rms_vals):.4f}"),
            ("Max RMS", f"{np.max(rms_vals):.4f}"),
        ]
    if len(mdf) > 0:
        rows += [
            ("Initial MDF (Hz)", f"{mdf[0]:.1f}"),
            ("Final MDF (Hz)", f"{mdf[-1]:.1f}"),
            ("MDF Change (Hz)", f"{mdf[-1]-mdf[0]:+.1f}"),
            ("Mean MDF (Hz)", f"{np.mean(mdf):.1f}"),
        ]
    table_rows = "\n".join(f'<tr><td style="padding:4px 12px;color:#555">{k}</td><td style="padding:4px 12px;font-weight:600">{v}</td></tr>' for k, v in rows)

    onset_rows = ""
    for i, o in enumerate(onsets[:50]):
        onset_rows += f'<tr><td>{i+1}</td><td>{o["onset_time"]:.3f}</td><td>{o["offset_time"]:.3f}</td><td>{o["duration_s"]:.3f}</td><td>{o["peak_amplitude"]:.4f}</td></tr>'

    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EMG Analysis Report</title>
<style>body{font-family:-apple-system,Segoe UI,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;background:#FAFAFA}
h1{color:#4A90D9;border-bottom:2px solid #E74C3C;padding-bottom:8px}
table{background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1);margin-bottom:16px}
tr:nth-child(even){background:#f4f7fb} th{background:#4A90D9;color:#fff;padding:6px 12px;text-align:left}</style></head><body>
<h1>EMG Analysis Report</h1>
<h3 style="color:#4A90D9">Summary</h3>
<table style="border-collapse:collapse;width:100%">{table_rows}</table>
<h3 style="color:#4A90D9">Onset Events (first 50)</h3>
<table><tr><th>#</th><th>Onset (s)</th><th>Offset (s)</th><th>Duration (s)</th><th>Peak Amp</th></tr>
{onset_rows}</table>
<h3 style="color:#4A90D9">Charts</h3>
<p><img src="emg_analysis.png" style="max-width:100%;border-radius:8px"></p>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Electromyography (EMG) Analysis")
    p.add_argument("--input", "-i", required=True, type=Path, help="EMG signal file (CSV or whitespace-separated)")
    p.add_argument("--column", "-c", default=0, type=int, help="Column index (default: 0)")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz (recommended >= 1000)")
    p.add_argument("--output-dir", "-o", type=Path, default=Path("emg_results"), help="Output directory")
    p.add_argument("--bp-low", type=float, default=20.0, help="Band-pass low cutoff Hz (default: 20)")
    p.add_argument("--bp-high", type=float, default=450.0, help="Band-pass high cutoff Hz (default: 450)")
    p.add_argument("--envelope-cutoff", type=float, default=10.0, help="Envelope LP cutoff Hz (default: 10)")
    p.add_argument("--rms-window", type=float, default=0.05, help="RMS window in seconds (default: 0.05)")
    p.add_argument("--z-thresh", type=float, default=3.0, help="Onset z-score threshold (default: 3.0)")
    p.add_argument("--mdf-window", type=float, default=0.5, help="Median freq window in seconds (default: 0.5)")
    p.add_argument("--json-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    if args.input.suffix == ".csv":
        df = pd.read_csv(args.input, header=None)
        emg = df.iloc[:, args.column].values
    else:
        emg = np.loadtxt(args.input)

    # Filter
    emg_filt = bandpass_filter(emg, args.fs, low=args.bp_low, high=args.bp_high)

    # Rectify + envelope
    emg_rect = rectify(emg_filt)
    envelope = smooth_envelope(emg_rect, args.fs, cutoff=args.envelope_cutoff)
    rms = compute_rms(emg_filt, args.rms_window, args.fs)

    # Onset detection
    onsets = detect_onsets(envelope, args.fs, z_thresh=args.z_thresh)

    # Median frequency
    try:
        mdf_t, mdf, mnf = median_frequency(emg_filt, args.fs, window_s=args.mdf_window)
    except ValueError:
        mdf_t, mdf, mnf = np.array([]), np.array([]), np.array([])

    # RMS features
    rms_clean = rms[rms > 0]
    onset_durs = [o["duration_s"] for o in onsets]

    result = {
        "metadata": {
            "input_file": str(args.input),
            "sampling_rate_hz": args.fs,
            "duration_s": float(len(emg) / args.fs),
            "bp_low_hz": args.bp_low,
            "bp_high_hz": args.bp_high,
        },
        "rms_features": {
            "mean_rms": float(np.mean(rms_clean)) if len(rms_clean) > 0 else 0.0,
            "std_rms": float(np.std(rms_clean)) if len(rms_clean) > 0 else 0.0,
            "max_rms": float(np.max(rms_clean)) if len(rms_clean) > 0 else 0.0,
            "min_rms": float(np.min(rms_clean)) if len(rms_clean) > 0 else 0.0,
        },
        "onset_detection": {
            "z_threshold": args.z_thresh,
            "n_onsets": len(onsets),
            "mean_duration_s": float(np.mean(onset_durs)) if onset_durs else 0.0,
            "onsets_per_minute": len(onsets) / (len(emg) / args.fs / 60) if len(emg) > 0 else 0.0,
            "onset_events": onsets,
        },
        "fatigue_analysis": {
            "initial_mdf_hz": float(mdf[0]) if len(mdf) > 0 else None,
            "final_mdf_hz": float(mdf[-1]) if len(mdf) > 0 else None,
            "mdf_change_hz": float(mdf[-1] - mdf[0]) if len(mdf) > 0 else None,
            "mean_mdf_hz": float(np.mean(mdf)) if len(mdf) > 0 else None,
            "n_windows": len(mdf),
        },
    }

    json_path = args.output_dir / "emg_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] Statistics saved to {json_path}")

    if not args.json_only:
        plot_emg(emg, emg_filt, envelope, rms, onsets, mdf_t, mdf, mnf, args.fs, args.output_dir)
        print(f"[OK] Plots saved to {args.output_dir}/emg_analysis.png and .html")

    print(f"\n{'='*40}")
    print(f"EMG Analysis Summary")
    print(f"{'='*40}")
    print(f"  Duration:       {len(emg)/args.fs:.1f} s")
    print(f"  Activations:    {len(onsets)}")
    print(f"  Mean RMS:       {result['rms_features']['mean_rms']:.4f}")
    if len(mdf) > 0:
        print(f"  Init MDF:       {mdf[0]:.1f} Hz")
        print(f"  Final MDF:      {mdf[-1]:.1f} Hz")
        print(f"  MDF Δ:          {mdf[-1]-mdf[0]:+.1f} Hz")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
