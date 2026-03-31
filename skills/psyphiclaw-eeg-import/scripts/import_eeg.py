#!/usr/bin/env python3
"""Import EEG data from Brain Vision, EGI, and EEGLAB formats using MNE-Python."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger("psyphiclaw-eeg-import")

PRIMARY = "#4A90D9"
SECONDARY = "#E74C3C"


def load_brain_vision(vhdr_path: str) -> mne.io.Raw:
    """Load Brain Vision format (.vhdr / .eeg / .vmrk)."""
    path = Path(vhdr_path)
    if path.suffix.lower() not in (".vhdr",):
        raise ValueError(f"Brain Vision format requires .vhdr file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    logger.info("Loading Brain Vision: %s", path)
    raw = mne.io.read_raw_brainvision(str(path), preload=False, verbose="INFO")
    return raw


def load_egi(egi_path: str) -> mne.io.Raw:
    """Load EGI format (.raw or .mff)."""
    path = Path(egi_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext == ".mff":
        logger.info("Loading EGI MFF: %s", path)
        raw = mne.io.read_raw_egi(str(path), preload=False, verbose="INFO")
    elif ext == ".raw":
        logger.info("Loading EGI RAW: %s", path)
        raw = mne.io.read_raw_egi(str(path), preload=False, verbose="INFO")
    else:
        raise ValueError(f"Unsupported EGI extension: {ext}")
    return raw


def load_eeglab(set_path: str) -> mne.io.Raw:
    """Load EEGLAB format (.set / .fdt)."""
    path = Path(set_path)
    if path.suffix.lower() != ".set":
        raise ValueError(f"EEGLAB format requires .set file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    logger.info("Loading EEGLAB: %s", path)
    raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose="INFO")
    return raw


def load_auto(file_path: str) -> mne.io.Raw:
    """Auto-detect format and load."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".vhdr":
        return load_brain_vision(file_path)
    elif ext in (".mff", ".raw"):
        return load_egi(file_path)
    elif ext == ".set":
        return load_eeglab(file_path)
    else:
        # Try brain vision first (might just be wrong extension)
        for loader in (load_brain_vision, load_egi, load_eeglab):
            try:
                return loader(file_path)
            except Exception:
                continue
        raise ValueError(f"Cannot determine EEG format for: {file_path}")


def compute_summary(raw: mne.io.Raw) -> dict:
    """Compute metadata summary from MNE Raw object."""
    n_channels = len(raw.ch_names)
    sfreq = raw.info["sfreq"]
    n_times = raw.n_times
    duration_s = n_times / sfreq

    # Channel types
    ch_types = {}
    for ch in raw.info["chs"]:
        kind = mne.channel_type(raw.info, ch["ch_name"])
        ch_types[kind] = ch_types.get(kind, 0) + 1

    # Events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_counts = {}
    if len(events) > 0:
        id_to_name = {v: k for k, v in event_id.items()}
        for ev_id in np.unique(events[:, 2]):
            name = id_to_name.get(ev_id, f"Event_{ev_id}")
            event_counts[name] = int(np.sum(events[:, 2] == ev_id))

    # Montage
    montage = "None"
    if raw.info.get("dig") and len(raw.info["dig"]) > 0:
        montage = "Custom"
    elif raw.get_montage():
        montage = raw.get_montage().kind

    return {
        "n_channels": n_channels,
        "sfreq": sfreq,
        "n_times": n_times,
        "duration_s": round(duration_s, 2),
        "ch_types": ch_types,
        "n_events": len(events),
        "event_counts": event_counts,
        "montage": montage,
        "highpass": raw.info["highpass"],
        "lowpass": raw.info["lowpass"],
        "reference": raw.info["custom_ref_applied"],
        "first_samp": raw.first_samp,
    }


def format_summary(summary: dict) -> str:
    """Format summary as readable text."""
    lines = [
        "=" * 55,
        "🧠 EEG Data Summary",
        "=" * 55,
        f"  Channels:      {summary['n_channels']}",
        f"  Sampling rate: {summary['sfreq']} Hz",
        f"  Duration:      {summary['duration_s']} s",
        f"  Time points:   {summary['n_times']}",
        f"  Montage:       {summary['montage']}",
        f"  Highpass:      {summary['highpass']} Hz",
        f"  Lowpass:       {summary['lowpass']} Hz",
        f"  Custom ref:    {summary['reference']}",
        "-" * 55,
        "  Channel types:",
    ]
    for kind, count in summary["ch_types"].items():
        lines.append(f"    {kind}: {count}")
    lines.append(f"  Events: {summary['n_events']}")
    if summary["event_counts"]:
        for name, count in summary["event_counts"].items():
            lines.append(f"    {name}: {count}")
    lines.append("=" * 55)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import EEG data")
    parser.add_argument("file_path", type=str, help="Path to EEG file (.vhdr, .mff, .raw, .set)")
    parser.add_argument("--format", type=str, choices=["bv", "egi", "eeglab", "auto"], default="auto",
                        help="File format (auto-detect if omitted)")
    parser.add_argument("--preload", action="store_true", help="Preload data into memory")
    parser.add_argument("--summary", action="store_true", help="Print data summary")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    loaders = {"bv": load_brain_vision, "egi": load_egi, "eeglab": load_eeglab, "auto": load_auto}
    raw = loaders[args.format](args.file_path)

    if args.preload:
        raw.load_data()
        logger.info("Data preloaded into memory")

    if args.summary:
        summary = compute_summary(raw)
        print(format_summary(summary))

    return raw


if __name__ == "__main__":
    main()
