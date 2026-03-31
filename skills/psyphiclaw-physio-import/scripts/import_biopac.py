#!/usr/bin/env python3
"""Import Biopac AcqKnowledge ACQ files.

Reads Biopac ACQ format (binary) using the wfdb library or direct binary
parsing, auto-detects channel types, and outputs a standardized DataFrame.

Usage:
    python import_biopac.py recording.acq --output result.parquet --summary
    python import_biopac.py recording.acq --channel-map '{"100":"ECG","101":"EDA"}'
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"

# Common Biopac channel type keywords for auto-detection
_CHANNEL_KEYWORDS: dict[str, list[str]] = {
    "ecg": ["ecg", "ekg", "cardio", "heart", "e100"],
    "eda": ["eda", "gsr", "galvanic", "skin conduct", "e101"],
    "emg": ["emg", "muscle", "electromy", "e102"],
    "resp": ["resp", "breath", "respirat", "pneumo", "e103"],
    "temp": ["temp", "skin temp", "therm", "e104"],
    "ppg": ["ppg", "pulse", "photopleth", "e105"],
}


def read_acq_header(file_path: Path) -> dict:
    """Read Biopac ACQ file header.

    The ACQ format starts with a file header containing channel info,
    sampling rate, and other metadata.

    Args:
        file_path: Path to .acq file.

    Returns:
        Dictionary with header metadata.
    """
    meta: dict = {"source_format": "Biopac ACQ", "file_path": str(file_path)}

    with open(file_path, "rb") as f:
        # Read the ACQ header (first ~900 bytes)
        # ACQ format: header_size (4 bytes), then header data
        header_data = f.read(2048)

        # Try to find sampling rate info
        # Common locations in ACQ header
        for offset in (0, 16, 20, 24, 28, 32):
            if offset + 4 > len(header_data):
                break
            val = struct.unpack_from("<I", header_data, offset)[0]
            if 100 <= val <= 10000:
                meta["sampling_rate_hz"] = val
                break

        # Try to read channel count
        for offset in (4, 8, 12):
            if offset + 4 > len(header_data):
                break
            val = struct.unpack_from("<H", header_data, offset)[0]
            if 1 <= val <= 32:
                meta["n_channels"] = val
                break

    return meta


def read_acq_binary(
    file_path: Path,
    n_channels: int = 0,
    data_type: str = "int16",
) -> tuple[np.ndarray, int]:
    """Read binary data from ACQ file.

    Args:
        file_path: Path to .acq file.
        n_channels: Number of channels.
        data_type: Data type ('int16', 'float32').

    Returns:
        Tuple of (data_array, sampling_rate).
    """
    with open(file_path, "rb") as f:
        raw = f.read()

    # Detect data type
    dtype_map = {
        "int16": "<h",
        "int8": "<b",
        "float32": "<f",
    }
    dt = np.dtype(dtype_map.get(data_type, "<h"))

    # Skip header (first ~900 bytes typically, but we search for data start)
    header_size = 900  # Default ACQ header size
    data = np.frombuffer(raw[header_size:], dtype=dt)

    # Reshape to (n_samples, n_channels)
    if n_channels > 0 and len(data) % n_channels == 0:
        data = data.reshape(-1, n_channels)
    elif n_channels > 0:
        # Trim to fit
        n_complete = (len(data) // n_channels) * n_channels
        data = data[:n_complete].reshape(-1, n_channels)

    # Estimate sampling rate from header
    meta = read_acq_header(file_path)
    fs = meta.get("sampling_rate_hz", 1000)

    return data, fs


def detect_channel_type(channel_name: str) -> str:
    """Detect channel type from channel name.

    Args:
        channel_name: Channel label/name.

    Returns:
        Detected type: ecg, eda, emg, resp, temp, or 'unknown'.
    """
    name_lower = channel_name.lower()
    for ch_type, keywords in _CHANNEL_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return ch_type
    return "unknown"


def auto_detect_channels(
    channel_names: list[str],
    channel_map: Optional[dict[str, str]] = None,
) -> dict[int, dict]:
    """Auto-detect channel types and build mapping.

    Args:
        channel_names: List of channel names.
        channel_map: Manual override mapping (channel_name → type).

    Returns:
        Dictionary mapping channel index to {name, type, unit}.
    """
    unit_map = {
        "ecg": ("ECG", "mV"),
        "eda": ("EDA", "μS"),
        "emg": ("EMG", "mV"),
        "resp": ("RESP", "Volts"),
        "temp": ("TEMP", "°C"),
        "ppg": ("PPG", "Volts"),
        "unknown": ("Unknown", "Volts"),
    }

    channels: dict[int, dict] = {}
    for i, name in enumerate(channel_names):
        if channel_map and name in channel_map:
            ch_type = channel_map[name]
        else:
            ch_type = detect_channel_type(name)

        label, unit = unit_map.get(ch_type, ("Unknown", "Volts"))
        channels[i] = {"name": name, "type": ch_type, "label": label, "unit": unit}

    return channels


def read_acq_via_wfdb(file_path: Path) -> Optional[tuple[pd.DataFrame, dict]]:
    """Try reading ACQ file using wfdb (if installed and file is compatible).

    Some newer Biopac systems export in WFDB-compatible format.

    Args:
        file_path: Path to .acq file.

    Returns:
        Tuple of (DataFrame, metadata) or None if not compatible.
    """
    try:
        import wfdb  # type: ignore[import-untyped]

        # Try header file
        header_path = file_path.with_suffix(".hea")
        dat_path = file_path  # or .dat

        if header_path.exists():
            record = wfdb.rdrecord(str(file_path.with_suffix("")))
            fs = record.fs
            signals = record.p_signal

            channel_names = list(record.sig_name) if record.sig_name else [
                f"ch_{i}" for i in range(signals.shape[1])
            ]

            df = pd.DataFrame(signals, columns=channel_names)
            df.insert(0, "timestamp_s", np.arange(len(df)) / fs)

            channels = auto_detect_channels(channel_names)
            meta = {
                "source_format": "Biopac ACQ (WFDB)",
                "sampling_rate_hz": fs,
                "n_channels": len(channel_names),
                "n_samples": len(df),
                "duration_s": len(df) / fs,
                "channels": {i: v for i, v in channels.items()},
            }

            return df, meta

    except (ImportError, Exception):
        pass

    return None


def import_biopac_acq(
    file_path: Path,
    channel_map: Optional[dict[str, str]] = None,
    n_channels: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Import a Biopac ACQ file into a standardized DataFrame.

    Tries WFDB first, falls back to direct binary parsing.

    Args:
        file_path: Path to .acq file.
        channel_map: Manual channel type mapping.
        n_channels: Number of channels (for binary parsing).

    Returns:
        Tuple of (DataFrame, metadata).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"ACQ file not found: {file_path}")

    # Try WFDB first
    wfdb_result = read_acq_via_wfdb(file_path)
    if wfdb_result is not None:
        df, meta = wfdb_result
        return df, meta

    # Fallback: direct binary parsing
    meta = read_acq_header(file_path)
    actual_n_channels = n_channels or meta.get("n_channels", 0)

    data, fs = read_acq_binary(file_path, n_channels=actual_n_channels)

    # Build DataFrame
    channel_names = [f"ch_{i}" for i in range(data.shape[1])]
    channels = auto_detect_channels(channel_names, channel_map)

    df = pd.DataFrame(data, columns=channel_names)
    df.insert(0, "timestamp_s", np.arange(len(df)) / fs)

    meta.update({
        "sampling_rate_hz": fs,
        "n_channels": data.shape[1],
        "n_samples": len(df),
        "duration_s": len(df) / fs,
        "channels": {i: v for i, v in channels.items()},
    })

    # Rename columns by detected type
    type_columns: dict[str, list[str]] = {}
    for i, ch_info in channels.items():
        ch_type = ch_info["type"]
        if ch_type != "unknown":
            col_name = channel_names[i]
            type_columns.setdefault(ch_type, []).append(col_name)
            # If single channel of this type, rename to type name
            if len(type_columns[ch_type]) == 1:
                df = df.rename(columns={col_name: ch_type})
            else:
                df = df.rename(columns={col_name: f"{ch_type}_{len(type_columns[ch_type])}"})

    return df, meta


def print_summary(df: pd.DataFrame, meta: dict) -> None:
    """Print metadata summary."""
    print("=" * 60)
    print("  Biopac ACQ Data Summary")
    print("=" * 60)
    print(f"  Format:           {meta.get('source_format', 'N/A')}")
    print(f"  Sampling rate:    {meta.get('sampling_rate_hz', 'N/A')} Hz")
    print(f"  Channels:         {meta.get('n_channels', 'N/A')}")
    print(f"  Total samples:    {meta.get('n_samples', 'N/A')}")
    print(f"  Duration:         {meta.get('duration_s', 'N/A'):.2f} s")
    print("  Channel details:")
    for i, ch in meta.get("channels", {}).items():
        detected = "✓" if ch["type"] != "unknown" else "?"
        print(f"    [{detected}] ch_{i}: {ch['name']} → {ch['label']} ({ch['unit']})")
    print(f"\n  Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    - {col}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Biopac ACQ files.")
    parser.add_argument("input", type=Path, help="Path to .acq file.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path.")
    parser.add_argument("--summary", action="store_true", help="Print summary.")
    parser.add_argument("--channel-map", type=str, default=None, help="JSON channel map.")
    parser.add_argument("--n-channels", type=int, default=0, help="Number of channels.")
    args = parser.parse_args()

    channel_map = json.loads(args.channel_map) if args.channel_map else None

    df, meta = import_biopac_acq(args.input, channel_map=channel_map, n_channels=args.n_channels)

    if args.summary:
        print_summary(df, meta)

    if args.output:
        suffix = args.output.suffix.lower()
        if suffix == ".parquet":
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Data saved to: {args.output}")


if __name__ == "__main__":
    main()
