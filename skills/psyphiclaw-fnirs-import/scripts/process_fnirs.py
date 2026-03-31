#!/usr/bin/env python3
"""Process fNIRS data: Beer-Lambert conversion, quality check, filtering, Hb separation.

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
    parser = argparse.ArgumentParser(description="Process fNIRS raw data.")
    parser.add_argument("--input", "-i", required=True, help="Input .fif or .csv file")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument("--lfreq", type=float, default=0.01, help="Low cutoff frequency (Hz)")
    parser.add_argument("--hfreq", type=float, default=0.5, help="High cutoff frequency (Hz)")
    parser.add_argument("--pfd", type=float, default=6.0, help="Partial pathlength factor (default: 6.0)")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality check")
    parser.add_argument("--quality-json", default=None, help="Export quality report as JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


# ── Beer-Lambert Conversion ───────────────────────────────────────────────────

def beer_lambert_convert(
    raw: object,
    pfd: float = 6.0,
    verbose: bool = False,
) -> object:
    """Convert optical density to hemoglobin concentration using Modified Beer-Lambert Law.

    The conversion follows: Δ[C] = (1 / (ε × d × pfd)) × ΔOD

    where ε is the extinction coefficient, d is the source-detector distance,
    and pfd is the partial pathlength factor (differential pathlength factor / 2π).

    Args:
        raw: MNE Raw object in optical density.
        pfd: Partial pathlength factor (default 6.0 for adults).
        verbose: Print debug info.

    Returns:
        MNE Raw object with HbO and HbR channels.
    """
    import mne

    # MNE's built-in beer-lambert conversion
    raw_od = mne.preprocessing.nirs.beer_lambert_law(
        raw, ppf=pfd, verbose=verbose
    )

    if verbose:
        hbo_chs = [c for c in raw_od.ch_names if "hbo" in c.lower()]
        hbr_chs = [c for c in raw_od.ch_names if "hbr" in c.lower()]
        print(f"[#4A90D9] Beer-Lambert conversion: {len(hbo_chs)} HbO, {len(hbr_chs)} HbR channels")

    return raw_od


# ── Scalp Coupling Index ──────────────────────────────────────────────────────

def scalp_coupling_index(
    raw: object,
    threshold: float = 0.75,
    verbose: bool = False,
) -> dict:
    """Compute scalp coupling index (SCI) to detect poor optode contact.

    SCI measures the correlation between signal and its Hilbert envelope,
    identifying channels with poor scalp contact (SCI < threshold).

    Args:
        raw: MNE Raw object.
        threshold: SCI threshold below which channels are flagged.
        verbose: Print info.

    Returns:
        Dict with per-channel SCI values and flagged channels.
    """
    import mne

    sci = mne.preprocessing.nirs.scalp_coupling_index(raw)
    ch_names = raw.ch_names
    flagged = [ch_names[i] for i in range(len(sci)) if sci[i] < threshold]

    report = {
        "threshold": threshold,
        "mean_sci": float(np.mean(sci)),
        "min_sci": float(np.min(sci)),
        "max_sci": float(np.max(sci)),
        "flagged_channels": flagged,
        "n_flagged": len(flagged),
        "n_total": len(ch_names),
        "per_channel": {ch_names[i]: round(float(sci[i]), 4) for i in range(len(sci))},
    }

    if verbose:
        status = f"[#4A90D9]" if report["n_flagged"] == 0 else f"[#E74C3C]"
        print(f"{status} SCI: mean={report['mean_sci']:.3f}, "
              f"{report['n_flagged']}/{report['n_total']} flagged (threshold={threshold})")

    return report


# ── Bandpass Filtering ────────────────────────────────────────────────────────

def bandpass_filter(
    raw: object,
    lfreq: float = 0.01,
    hfreq: float = 0.5,
    verbose: bool = False,
) -> object:
    """Apply bandpass filter to fNIRS data.

    Typical fNIRS bandpass: 0.01–0.5 Hz captures the hemodynamic response
    while removing cardiac (~1 Hz), respiratory (~0.3 Hz), and drift components.

    Args:
        raw: MNE Raw object.
        lfreq: Low cutoff frequency (Hz).
        hfreq: High cutoff frequency (Hz).
        verbose: Print info.

    Returns:
        Filtered MNE Raw object.
    """
    raw_filtered = raw.copy().filter(l_freq=lfreq, h_freq=hfreq, verbose=verbose)

    if verbose:
        print(f"[#4A90D9] Bandpass filter: {lfreq}–{hfreq} Hz applied")

    return raw_filtered


# ── Hb Separation ─────────────────────────────────────────────────────────────

def separate_hemoglobin(raw: object, verbose: bool = False) -> dict:
    """Separate and summarize hemoglobin concentrations.

    Computes HbO (oxyhemoglobin), HbR (deoxyhemoglobin), and optionally
    HbT (total hemoglobin = HbO + HbR).

    Args:
        raw: MNE Raw object with hbo/hbr channels.
        verbose: Print info.

    Returns:
        Dict with channel-level statistics.
    """
    data = raw.get_data()
    ch_names = raw.ch_names

    hbo_indices = [i for i, c in enumerate(ch_names) if "hbo" in c.lower()]
    hbr_indices = [i for i, c in enumerate(ch_names) if "hbr" in c.lower()]

    stats: dict = {"hbo": {}, "hbr": {}, "n_hbo": len(hbo_indices), "n_hbr": len(hbr_indices)}

    for label, indices in [("hbo", hbo_indices), ("hbr", hbr_indices)]:
        if indices:
            ch_data = data[indices, :]
            stats[label] = {
                "mean": round(float(np.mean(ch_data)), 6),
                "std": round(float(np.std(ch_data)), 6),
                "min": round(float(np.min(ch_data)), 6),
                "max": round(float(np.max(ch_data)), 6),
            }

    # HbT = HbO + HbR (matched pairs)
    n_pairs = min(len(hbo_indices), len(hbr_indices))
    if n_pairs > 0:
        hbt = data[hbo_indices[:n_pairs]] + data[hbr_indices[:n_pairs]]
        stats["hbt"] = {
            "n_pairs": n_pairs,
            "mean": round(float(np.mean(hbt)), 6),
            "std": round(float(np.std(hbt)), 6),
        }

    if verbose:
        print(f"[#4A90D9] HbO: {stats['n_hbo']} ch, HbR: {stats['n_hbr']} ch, "
              f"HbT pairs: {n_pairs}")

    return stats


# ── CSV Fallback ──────────────────────────────────────────────────────────────

def process_csv(input_path: str, args: argparse.Namespace) -> dict:
    """Process fNIRS data from CSV (Artinis export)."""
    import pandas as pd

    df = pd.read_csv(input_path)
    n_cols = len(df.columns)

    # Simple bandpass via rolling mean detrend + Gaussian filter
    from scipy.ndimage import gaussian_filter1d

    sigma_h = (df.index[1] if len(df) > 1 else 1) / (2 * args.hfreq) if hasattr(df.index, '__iter__') else 10
    for col in df.columns:
        if col.lower() in ("time", "event", "marker", "timestamp"):
            continue
        df[col] = gaussian_filter1d(df[col].values, sigma=max(3, min(50, int(sigma_h))))

    out = args.output or input_path.replace(".csv", "_processed.csv")
    df.to_csv(out, index=False)

    return {"status": "csv_processed", "columns": n_cols, "output": out}


# ── Main ──────────────────────────────────────────────────────────────────────

COLOR_PRIMARY = "#4A90D9"
COLOR_ALERT = "#E74C3C"


def main() -> int:
    args = parse_args()

    if not Path(args.input).exists():
        print(f"[{COLOR_ALERT}] ERROR: File not found: {args.input}", file=sys.stderr)
        return 1

    if args.input.endswith(".csv"):
        result = process_csv(args.input, args)
        print(json.dumps(result, indent=2))
        return 0

    try:
        import mne
        raw = mne.io.read_raw_fif(args.input, verbose=args.verbose, preload=True)
    except ImportError:
        print(f"[{COLOR_ALERT}] ERROR: mne not installed. pip install mne", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[{COLOR_ALERT}] ERROR loading .fif: {e}", file=sys.stderr)
        return 1

    pipeline_report: dict = {"steps": []}

    # Step 1: Quality check
    if not args.skip_quality:
        print(f"[{COLOR_PRIMARY}] Step 1: Scalp coupling quality check...")
        qi = scalp_coupling_index(raw, verbose=args.verbose)
        pipeline_report["steps"].append({"name": "quality_check", **qi})

        if qi["n_flagged"] > 0:
            print(f"  [{COLOR_ALERT}] ⚠ {qi['n_flagged']} channels flagged: {qi['flagged_channels'][:5]}")
    else:
        print(f"[{COLOR_PRIMARY}] Step 1: Quality check SKIPPED")

    # Step 2: Beer-Lambert conversion
    print(f"[{COLOR_PRIMARY}] Step 2: Beer-Lambert law conversion (PFD={args.pfd})...")
    raw_conc = beer_lambert_convert(raw, pfd=args.pfd, verbose=args.verbose)
    pipeline_report["steps"].append({"name": "beer_lambert", "pfd": args.pfd})

    # Step 3: Bandpass filter
    print(f"[{COLOR_PRIMARY}] Step 3: Bandpass filter {args.lfreq}–{args.hfreq} Hz...")
    raw_filt = bandpass_filter(raw_conc, lfreq=args.lfreq, hfreq=args.hfreq, verbose=args.verbose)
    pipeline_report["steps"].append({"name": "bandpass", "lfreq": args.lfreq, "hfreq": args.hfreq})

    # Step 4: Hemoglobin separation
    print(f"[{COLOR_PRIMARY}] Step 4: Hemoglobin concentration separation...")
    hb_stats = separate_hemoglobin(raw_filt, verbose=args.verbose)
    pipeline_report["steps"].append({"name": "hb_separation", **hb_stats})

    # Save
    out_path = args.output or args.input.replace(".fif", "_processed.fif")
    raw_filt.save(out_path, overwrite=True)
    pipeline_report["output"] = out_path
    print(f"[{COLOR_PRIMARY}] ✓ Saved: {out_path}")

    if args.quality_json:
        with open(args.quality_json, "w") as f:
            json.dump(pipeline_report, f, indent=2)
        print(f"[{COLOR_PRIMARY}] Report: {args.quality_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
