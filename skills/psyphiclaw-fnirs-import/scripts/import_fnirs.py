#!/usr/bin/env python3
"""Import fNIRS data from SNIRF, NIRSport, and Artinis OxySoft formats.

Outputs MNE-Python Raw objects with channel metadata (source-detector pairs,
wavelengths). Supports batch processing of multiple files.

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
    parser = argparse.ArgumentParser(
        description="Import fNIRS data into MNE Raw format."
    )
    parser.add_argument("--input", "-i", required=True, help="Input file path (.snirf, .nirs, .txt, .csv)")
    parser.add_argument("--format", "-f", required=True, choices=["snirf", "nirsport", "artinis"],
                        help="Input format: snirf, nirsport, or artinis")
    parser.add_argument("--output", "-o", default=None, help="Output .fif file (default: auto-named)")
    parser.add_argument("--info-json", default=None, help="Export channel metadata as JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


# ── SNIRF Reader ──────────────────────────────────────────────────────────────

def read_snirf(filepath: str, verbose: bool = False) -> tuple[Optional[object], dict]:
    """Read SNIRF (.snirf / .nirs) files using MNE-Python.

    Args:
        filepath: Path to SNIRF file.
        verbose: Print debug info.

    Returns:
        Tuple of (MNE Raw object or None, metadata dict).
    """
    meta: dict = {"format": "snirf", "channels": []}

    try:
        import mne
        raw = mne.io.read_raw_snirf(filepath, verbose=verbose, preload=True)
        meta["sfreq"] = raw.info["sfreq"]
        meta["n_channels"] = len(raw.ch_names)
        meta["duration"] = raw.n_times / raw.info["sfreq"]

        for ch in raw.info["chs"]:
            meta["channels"].append({
                "name": ch["ch_name"],
                "kind": str(ch["kind"]),
                "unit": ch["unit"],
            })

        return raw, meta
    except ImportError:
        print("[#E74C3C] ERROR: mne not installed. pip install mne", file=sys.stderr)
        return None, meta
    except Exception as e:
        print(f"[#E74C3C] ERROR reading SNIRF: {e}", file=sys.stderr)
        # Fallback: manual HDF5 read for metadata
        return _snirf_fallback(filepath, meta, verbose)


def _snirf_fallback(filepath: str, meta: dict, verbose: bool) -> tuple[None, dict]:
    """Fallback SNIRF reader using h5py for metadata extraction only."""
    try:
        import h5py

        with h5py.File(filepath, "r") as f:
            meta["formatVersion"] = f.get("formatVersion", b"").decode()
            nirs = f.get("nirs", {})

            # Extract source/detector info
            for key in list(nirs.keys())[:3]:  # first 3 sources
                src = nirs[key]
                meta["channels"].append({
                    "name": str(key),
                    "source_wavelengths": list(src.get("source wavelengths", [])),
                })

            if verbose:
                print(f"[#4A90D9] SNIRF metadata extracted (fallback): formatVersion={meta['formatVersion']}")
    except Exception as e:
        print(f"[#E74C3C] Fallback also failed: {e}", file=sys.stderr)

    return None, meta


# ── NIRSport Reader ──────────────────────────────────────────────────────────

def read_nirsport(filepath: str, verbose: bool = False) -> tuple[Optional[object], dict]:
    """Read NIRSport raw data (.nirs or .hdr/.dat pairs).

    NIRSport devices (NIRx) store data in their proprietary format.
    MNE-Python provides native support for newer NIRSport files.

    Args:
        filepath: Path to NIRSport file.
        verbose: Print debug info.

    Returns:
        Tuple of (MNE Raw or None, metadata dict).
    """
    meta: dict = {"format": "nirsport", "channels": []}

    try:
        import mne

        # MNE can read NIRSport .nirs files directly
        raw = mne.io.read_raw_nirx(filepath, verbose=verbose, preload=True)
        meta["sfreq"] = raw.info["sfreq"]
        meta["n_channels"] = len(raw.ch_names)
        meta["duration"] = raw.n_times / raw.info["sfreq"]
        meta["device"] = "NIRSport"

        for ch in raw.info["chs"]:
            meta["channels"].append({"name": ch["ch_name"]})

        if verbose:
            print(f"[#4A90D9] NIRSport: {meta['n_channels']} channels, "
                  f"{meta['sfreq']} Hz, {meta['duration']:.1f}s")

        return raw, meta
    except ImportError:
        print("[#E74C3C] ERROR: mne not installed.", file=sys.stderr)
        return None, meta
    except Exception as e:
        print(f"[#E74C3C] ERROR reading NIRSport: {e}", file=sys.stderr)
        # Try reading as raw text for metadata
        return _nirsport_fallback(filepath, meta, verbose)


def _nirsport_fallback(filepath: str, meta: dict, verbose: bool) -> tuple[None, dict]:
    """Fallback NIRSport reader for header-only parsing."""
    p = Path(filepath)
    # Look for .hdr companion
    hdr = p.with_suffix(".hdr")
    dat = p.with_suffix(".dat")

    if hdr.exists():
        with open(hdr) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    meta["channels"].append({"name": parts[0]})

    if dat.exists():
        # Count data lines for duration estimation
        lines = sum(1 for _ in open(dat) if _.strip() and not _.startswith("#"))
        meta["data_lines"] = lines

    if verbose:
        print(f"[#4A90D9] NIRSport fallback: {len(meta['channels'])} channels from header")

    return None, meta


# ── Artinis Reader ────────────────────────────────────────────────────────────

def read_artinis(filepath: str, verbose: bool = False) -> tuple[Optional[object], dict]:
    """Read Artinis OxySoft export files (.txt, .csv).

    Artinis OxySoft exports tab-separated or comma-separated files with
    channel data, timestamps, and event markers.

    Args:
        filepath: Path to Artinis export file.
        verbose: Print debug info.

    Returns:
        Tuple of (numpy array or None, metadata dict).
    """
    meta: dict = {"format": "artinis", "channels": [], "device": "OxySoft"}

    try:
        import pandas as pd

        sep = "\t" if filepath.endswith(".txt") else ","
        df = pd.read_csv(filepath, sep=sep, comment="#", on_bad_lines="skip")
        meta["columns"] = list(df.columns)
        meta["n_samples"] = len(df)
        meta["n_channels"] = len(df.columns) - 2  # minus time + event columns

        # Detect channel columns (typically named like S1D1_760nm)
        import re
        channel_pattern = re.compile(r"S\d+D\d+_\d+nm|O2Hb|HHb|Channel")
        for col in df.columns:
            if channel_pattern.search(col):
                meta["channels"].append({"name": col})

        if verbose:
            print(f"[#4A90D9] Artinis: {meta['n_samples']} samples, "
                  f"{len(meta['channels'])} fNIRS channels detected")

        return df, meta
    except ImportError:
        print("[#E74C3C] ERROR: pandas not installed.", file=sys.stderr)
        return None, meta
    except Exception as e:
        print(f"[#E74C3C] ERROR reading Artinis: {e}", file=sys.stderr)
        return None, meta


# ── Main ──────────────────────────────────────────────────────────────────────

READERS = {
    "snirf": read_snirf,
    "nirsport": read_nirsport,
    "artinis": read_artinis,
}

COLOR_PRIMARY = "#4A90D9"
COLOR_ALERT = "#E74C3C"


def main() -> int:
    args = parse_args()

    if not Path(args.input).exists():
        print(f"[{COLOR_ALERT}] ERROR: File not found: {args.input}", file=sys.stderr)
        return 1

    reader = READERS[args.format]
    data, meta = reader(args.input, verbose=args.verbose)

    if data is None:
        print(f"[{COLOR_ALERT}] Import produced no data object (metadata only).")
    else:
        print(f"[{COLOR_PRIMARY}] ✓ Successfully imported {args.format} data.")
        print(f"  Channels: {meta.get('n_channels', 'N/A')}")
        print(f"  Sfreq: {meta.get('sfreq', 'N/A')}")
        print(f"  Duration: {meta.get('duration', 'N/A')}")

    # Save output
    out_path = args.output or f"fnirs_{Path(args.input).stem}.fif"
    if args.format == "snirf" and data is not None:
        data.save(out_path, overwrite=True)
        print(f"[{COLOR_PRIMARY}] Saved: {out_path}")
    elif args.format == "nirsport" and data is not None:
        data.save(out_path, overwrite=True)
        print(f"[{COLOR_PRIMARY}] Saved: {out_path}")
    elif args.format == "artinis" and data is not None:
        out_path = out_path.replace(".fif", ".csv")
        data.to_csv(out_path, index=False)
        print(f"[{COLOR_PRIMARY}] Saved: {out_path}")

    # Export metadata JSON
    if args.info_json:
        with open(args.info_json, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[{COLOR_PRIMARY}] Metadata: {args.info_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
