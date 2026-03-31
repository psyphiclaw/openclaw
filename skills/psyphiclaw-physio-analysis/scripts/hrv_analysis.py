#!/usr/bin/env python3
"""ECG Heart Rate Variability (HRV) Analysis.

Performs R-peak detection on ECG signals and computes time-domain,
frequency-domain, and non-linear HRV metrics. Outputs JSON statistics
and publication-quality PNG/HTML charts.

Typical usage:
    python hrv_analysis.py --input ecg.csv --fs 500 --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

# Color scheme
PRIMARY = "#4A90D9"
SECONDARY = "#E74C3C"
BG_COLOR = "#FAFAFA"


# ---------------------------------------------------------------------------
# R-peak detection
# ---------------------------------------------------------------------------

def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Detect R-peaks in an ECG signal using a Pan-Tompkins-inspired approach.

    Steps:
    1. Band-pass filter (5–15 Hz) to isolate QRS morphology.
    2. Differentiate to emphasise steep slopes.
    3. Square the signal to amplify peaks.
    4. Moving-average smoothing.
    5. Adaptive threshold peak detection.

    Parameters
    ----------
    ecg : array-like
        Raw ECG signal in mV or arbitrary units.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    r_peak_indices : np.ndarray
        Sample indices of detected R-peaks.
    """
    ecg = np.asarray(ecg, dtype=float)
    n = len(ecg)

    # Band-pass filter 5-15 Hz (4th order Butterworth)
    b, a = signal.butter(4, [5.0, 15.0], btype="bandpass", fs=fs)
    filtered = signal.filtfilt(b, a, ecg)

    # Derivative
    diff = np.diff(filtered, prepend=filtered[0])

    # Square
    squared = diff ** 2

    # Moving average (window ~150 ms)
    win = int(0.15 * fs)
    if win % 2 == 0:
        win += 1
    ma = signal.medfilt(squared, kernel_size=win)

    # Adaptive threshold
    threshold = np.mean(ma) + 0.5 * np.std(ma)

    # Minimum inter-beat interval (200 ms)
    min_dist = int(0.2 * fs)

    peaks, _ = signal.find_peaks(ma, height=threshold, distance=min_dist)

    # Refine: pick the maximum of the band-pass filtered signal within ±50 ms
    search_half = int(0.05 * fs)
    refined: list[int] = []
    for p in peaks:
        lo = max(0, p - search_half)
        hi = min(n, p + search_half + 1)
        refined.append(lo + np.argmax(filtered[lo:hi]))

    r_peaks = np.array(refined, dtype=int)
    # Remove duplicates / enforce minimum distance
    if len(r_peaks) > 1:
        keep = [r_peaks[0]]
        for k in range(1, len(r_peaks)):
            if r_peaks[k] - keep[-1] >= min_dist:
                keep.append(r_peaks[k])
        r_peaks = np.array(keep, dtype=int)

    return r_peaks


# ---------------------------------------------------------------------------
# RR intervals
# ---------------------------------------------------------------------------

def compute_rr_intervals(r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """Compute RR intervals in seconds from R-peak indices."""
    return np.diff(r_peaks) / fs


# ---------------------------------------------------------------------------
# Time-domain HRV
# ---------------------------------------------------------------------------

def time_domain_hrv(rr: np.ndarray) -> dict[str, float]:
    """Compute standard time-domain HRV metrics.

    Parameters
    ----------
    rr : np.ndarray
        RR intervals in seconds.

    Returns
    -------
    dict with keys: mean_rr, mean_hr, sdnn, rmssd, nn50, pnn50, median_rr.
    """
    nn_diff = np.diff(rr)
    nn50 = int(np.sum(np.abs(nn_diff) > 0.05))
    pnn50 = 100.0 * nn50 / len(nn_diff) if len(nn_diff) > 0 else 0.0
    rmssd = float(np.sqrt(np.mean(nn_diff ** 2)))

    return {
        "mean_rr_ms": float(np.mean(rr) * 1000),
        "std_rr_ms": float(np.std(rr) * 1000),
        "mean_hr_bpm": float(60.0 / np.mean(rr)),
        "sdnn_ms": float(np.std(rr) * 1000),
        "rmssd_ms": rmssd * 1000,
        "nn50": nn50,
        "pnn50_pct": pnn50,
        "median_rr_ms": float(np.median(rr) * 1000),
        "n_rr": int(len(rr)),
    }


# ---------------------------------------------------------------------------
# Frequency-domain HRV
# ---------------------------------------------------------------------------

def frequency_domain_hrv(rr: np.ndarray, fs_rr: float = 4.0) -> dict[str, float]:
    """Compute frequency-domain HRV via Welch PSD on cubic-spline-interpolated RR.

    Parameters
    ----------
    rr : np.ndarray
        RR intervals in seconds.
    fs_rr : float
        Interpolation frequency for the RR tachogram (Hz). Default 4 Hz.

    Returns
    -------
    dict with keys: lf_power, hf_power, lfhf_ratio, lf_nu, hf_nu, total_power, vlf_power.
    """
    # Cubic-spline interpolation of RR to evenly-sampled tachogram
    cum_time = np.cumsum(rr)
    t_interp = np.arange(cum_time[0], cum_time[-1], 1.0 / fs_rr)
    tck = stats.interpolate.splrep(cum_time, rr, s=0)
    rr_interp = stats.interpolate.splev(t_interp, tck)

    nperseg = min(256, len(rr_interp) // 2)
    if nperseg < 16:
        nperseg = max(16, len(rr_interp) // 2)
    freqs, psd = signal.welch(rr_interp, fs=fs_rr, nperseg=nperseg)

    # Integrate power in bands
    vlf_mask = freqs < 0.04
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs > 0.15) & (freqs <= 0.4)

    df = freqs[1] - freqs[0]
    vlf_power = float(np.trapz(psd[vlf_mask], freqs[vlf_mask])) if np.any(vlf_mask) else 0.0
    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
    total_power = vlf_power + lf_power + hf_power

    lf_nu = (lf_power / (lf_power + hf_power) * 100) if (lf_power + hf_power) > 0 else 0.0
    hf_nu = (hf_power / (lf_power + hf_power) * 100) if (lf_power + hf_power) > 0 else 0.0

    return {
        "vlf_power_ms2": vlf_power * 1000,
        "lf_power_ms2": lf_power * 1000,
        "hf_power_ms2": hf_power * 1000,
        "total_power_ms2": total_power * 1000,
        "lfhf_ratio": lf_power / hf_power if hf_power > 0 else 0.0,
        "lf_nu": lf_nu,
        "hf_nu": hf_nu,
    }


# ---------------------------------------------------------------------------
# Non-linear HRV
# ---------------------------------------------------------------------------

def poincare_sd(rr: np.ndarray) -> dict[str, float]:
    """Compute SD1 and SD2 from Poincaré plot (RR_n vs RR_{n+1})."""
    if len(rr) < 2:
        return {"sd1_ms": 0.0, "sd2_ms": 0.0}
    x = rr[:-1]
    y = rr[1:]
    diff = y - x
    mean_diff = np.mean(diff)
    var_diff = np.var(diff)
    sd1 = float(np.sqrt(var_diff) / np.sqrt(2)) * 1000
    sd2 = float(np.sqrt(2 * np.var(x) - var_diff / 2)) * 1000
    return {"sd1_ms": sd1, "sd2_ms": sd2, "sd1_sd2_ratio": sd1 / sd2 if sd2 > 0 else 0.0}


def approximate_entropy(rr: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """Compute approximate entropy (ApEn) of RR intervals.

    Parameters
    ----------
    rr : np.ndarray
        RR intervals.
    m : int
        Embedding dimension (typically 2).
    r_ratio : float
        Tolerance as fraction of std(rr).

    Returns
    -------
    float : Approximate entropy.
    """
    r = r_ratio * np.std(rr)
    n = len(rr)

    def _phi(data: np.ndarray, m_dim: int) -> float:
        patterns = np.array([data[i:i + m_dim] for i in range(n - m_dim + 1)])
        count = 0
        total = 0
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i == j:
                    continue
                if np.max(np.abs(patterns[i] - patterns[j])) < r:
                    count += 1
                total += 1
        return np.log(count / total) if total > 0 and count > 0 else 0.0

    phi_m = _phi(rr, m)
    phi_m1 = _phi(rr, m + 1)
    return float(phi_m - phi_m1)


def nonlinear_hrv(rr: np.ndarray) -> dict[str, float]:
    """Compute non-linear HRV metrics."""
    poincare = poincare_sd(rr)
    apen = approximate_entropy(rr)
    return {**poincare, "approximate_entropy": apen}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hrv(
    r_peaks: np.ndarray,
    rr: np.ndarray,
    td: dict[str, float],
    fd: dict[str, float],
    nl: dict[str, float],
    ecg: np.ndarray,
    fs: float,
    output_dir: Path,
) -> None:
    """Generate PNG and HTML summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t_ecg = np.arange(len(ecg)) / fs

    fig = plt.figure(figsize=(18, 14), facecolor=BG_COLOR)
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. ECG with R-peaks
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_ecg, ecg, color=PRIMARY, linewidth=0.4, label="ECG")
    ax1.plot(t_ecg[r_peaks], ecg[r_peaks], "v", color=SECONDARY, markersize=6, label="R-peaks")
    ax1.set_title("ECG Signal with Detected R-Peaks", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(fontsize=9)

    # 2. RR tachogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(np.arange(len(rr)), rr * 1000, color=PRIMARY, linewidth=1)
    ax2.axhline(np.mean(rr) * 1000, color=SECONDARY, linestyle="--", linewidth=1, label=f"Mean={td['mean_rr_ms']:.1f} ms")
    ax2.set_title("RR Intervals", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Beat #")
    ax2.set_ylabel("RR (ms)")
    ax2.legend(fontsize=8)

    # 3. RR histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(rr * 1000, bins=30, color=PRIMARY, edgecolor="white", alpha=0.85)
    ax3.set_title("RR Interval Distribution", fontsize=11, fontweight="bold")
    ax3.set_xlabel("RR (ms)")
    ax3.set_ylabel("Count")

    # 4. Poincaré plot
    ax4 = fig.add_subplot(gs[1, 2])
    if len(rr) > 1:
        ax4.scatter(rr[:-1] * 1000, rr[1:] * 1000, c=PRIMARY, s=15, alpha=0.6)
        ax4.plot([0, max(rr) * 1000], [0, max(rr) * 1000], "--", color="gray", linewidth=0.8)
    ax4.set_title(f"Poincaré Plot (SD1={nl['sd1_ms']:.1f}, SD2={nl['sd2_ms']:.1f})", fontsize=10, fontweight="bold")
    ax4.set_xlabel("RR_n (ms)")
    ax4.set_ylabel("RR_{n+1} (ms)")
    ax4.set_aspect("equal", adjustable="datalim")

    # 5. Frequency-domain bar chart
    ax5 = fig.add_subplot(gs[2, 0])
    bands = ["VLF", "LF", "HF"]
    powers = [fd["vlf_power_ms2"], fd["lf_power_ms2"], fd["hf_power_ms2"]]
    colors_bar = [SECONDARY, PRIMARY, SECONDARY]
    ax5.bar(bands, powers, color=colors_bar, edgecolor="white")
    ax5.set_title("Frequency-Domain Power", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Power (ms²)")

    # 6. LF/HF ratio gauge
    ax6 = fig.add_subplot(gs[2, 1])
    categories = ["LF (nu)", "HF (nu)"]
    values = [fd["lf_nu"], fd["hf_nu"]]
    ax6.bar(categories, values, color=[PRIMARY, SECONDARY], edgecolor="white")
    ax6.set_title(f"Sympathovagal Balance (LF/HF={fd['lfhf_ratio']:.2f})", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Normalised Power (%)")

    # 7. Summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    summary_lines = [
        f"HRV Summary ({len(rr)} beats)",
        "─" * 30,
        f"Mean HR:  {td['mean_hr_bpm']:.1f} bpm",
        f"Mean RR:  {td['mean_rr_ms']:.1f} ms",
        f"SDNN:     {td['sdnn_ms']:.1f} ms",
        f"RMSSD:    {td['rmssd_ms']:.1f} ms",
        f"pNN50:    {td['pnn50_pct']:.1f} %",
        f"LF/HF:    {fd['lfhf_ratio']:.2f}",
        f"SD1/SD2:  {nl['sd1_sd2_ratio']:.2f}",
        f"ApEn:     {nl['approximate_entropy']:.3f}",
    ]
    ax7.text(0.05, 0.95, "\n".join(summary_lines), transform=ax7.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=PRIMARY, alpha=0.9))

    plt.savefig(output_dir / "hrv_analysis.png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

    # Generate HTML report
    html = _build_html(td, fd, nl, output_dir.name)
    (output_dir / "hrv_analysis.html").write_text(html, encoding="utf-8")


def _build_html(td: dict, fd: dict, nl: dict, dir_name: str) -> str:
    """Build standalone HTML report."""
    def _section(title: str, items: dict) -> str:
        rows = "\n".join(f'<tr><td style="padding:4px 12px;color:#555">{k}</td><td style="padding:4px 12px;font-weight:600">{v}</td></tr>'
                         for k, v in items.items())
        return f'<h3 style="color:#4A90D9;margin-top:18px">{title}</h3><table style="border-collapse:collapse;width:100%">{rows}</table>'

    td_items = {k.replace("_", " ").title(): f"{v:.2f}" for k, v in td.items()}
    fd_items = {k.replace("_", " ").title(): f"{v:.2f}" for k, v in fd.items()}
    nl_items = {k.replace("_", " ").title(): f"{v:.4f}" for k, v in nl.items()}

    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>HRV Analysis Report</title>
<style>body{font-family:-apple-system,Segoe UI,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#FAFAFA}
h1{color:#4A90D9;border-bottom:2px solid #E74C3C;padding-bottom:8px}
table{background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1)}
tr:nth-child(even){background:#f4f7fb}</style></head><body>
<h1>ECG Heart Rate Variability Analysis</h1>
{_section("Time-Domain Metrics", td_items)}
{_section("Frequency-Domain Metrics", fd_items)}
{_section("Non-Linear Metrics", nl_items)}
<h3 style="color:#4A90D9;margin-top:18px">Charts</h3>
<p><img src="hrv_analysis.png" style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.12)"></p>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECG Heart Rate Variability (HRV) Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, type=Path, help="Path to ECG signal file (CSV with one column, or whitespace-separated)")
    parser.add_argument("--column", "-c", default=0, type=int, help="Column index for ECG data (default: 0)")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz (recommended >= 200)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("hrv_results"), help="Output directory (default: hrv_results)")
    parser.add_argument("--fs-rr", type=float, default=4.0, help="Interpolation frequency for PSD (Hz, default: 4)")
    parser.add_argument("--skip-seconds", type=float, default=0.0, help="Skip first N seconds of signal")
    parser.add_argument("--json-only", action="store_true", help="Only output JSON, skip plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load signal
    if args.input.suffix == ".csv":
        df = pd.read_csv(args.input, header=None)
        ecg = df.iloc[:, args.column].values
    else:
        ecg = np.loadtxt(args.input)

    # Trim
    skip = int(args.skip_seconds * args.fs)
    ecg = ecg[skip:]

    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg, args.fs)
    if len(r_peaks) < 10:
        print(f"ERROR: Only {len(r_peaks)} R-peaks detected. Check signal quality or --fs.", file=sys.stderr)
        sys.exit(1)

    # RR intervals
    rr = compute_rr_intervals(r_peaks, args.fs)
    # Remove outlier RR intervals (> 3 SD from mean)
    rr_mask = np.abs(rr - np.mean(rr)) < 3 * np.std(rr)
    rr_clean = rr[rr_mask]
    if len(rr_clean) < 10:
        print(f"WARNING: Only {len(rr_clean)} clean RR intervals after outlier removal.", file=sys.stderr)

    # Compute metrics
    td = time_domain_hrv(rr_clean)
    fd = frequency_domain_hrv(rr_clean, fs_rr=args.fs_rr)
    nl = nonlinear_hrv(rr_clean)

    # Add metadata
    result = {
        "metadata": {
            "input_file": str(args.input),
            "sampling_rate_hz": args.fs,
            "duration_s": len(ecg) / args.fs,
            "total_r_peaks": int(len(r_peaks)),
            "clean_rr_intervals": int(len(rr_clean)),
            "outlier_rr_removed": int(np.sum(~rr_mask)),
        },
        "time_domain": td,
        "frequency_domain": fd,
        "non_linear": nl,
    }

    # Save JSON
    json_path = args.output_dir / "hrv_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] Statistics saved to {json_path}")

    # Plots
    if not args.json_only:
        plot_hrv(r_peaks, rr_clean, td, fd, nl, ecg, args.fs, args.output_dir)
        print(f"[OK] Plots saved to {args.output_dir}/hrv_analysis.png and .html")

    # Print summary
    print(f"\n{'='*40}")
    print(f"HRV Analysis Summary ({len(rr_clean)} beats)")
    print(f"{'='*40}")
    print(f"  Mean HR:   {td['mean_hr_bpm']:.1f} bpm")
    print(f"  SDNN:      {td['sdnn_ms']:.1f} ms")
    print(f"  RMSSD:     {td['rmssd_ms']:.1f} ms")
    print(f"  pNN50:     {td['pnn50_pct']:.1f}%")
    print(f"  LF/HF:     {fd['lfhf_ratio']:.2f}")
    print(f"  SD1/SD2:   {nl['sd1_sd2_ratio']:.2f}")
    print(f"  ApEn:      {nl['approximate_entropy']:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
