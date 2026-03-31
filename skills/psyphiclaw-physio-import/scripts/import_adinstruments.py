#!/usr/bin/env python3
"""Import ADInstruments LabChart exported data (CSV/TXT).

Parses LabChart exports with time-stamped channel data and converts
to a standardized DataFrame format.

Usage:
    python import_adinstruments.py export.csv --output result.parquet --summary
    python import_adinstruments.py export.csv --time-unit s --encoding utf-8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"

# LabChart channel type keywords
_CHANNEL_KEYWORDS: dict[str, list[str]] = {
    "ecg": ["ecg", "ekg", "ecg ", "ekg "],
    "eda": ["eda", "gsr", "galvanic", "skin conduct", "scl"],
    "emg": ["emg", "muscle", "electromy"],
    "resp": ["resp", "breath", "respirat", "pneumo", "insp"],
    "temp": ["temp", "skin temp", "thermistor", "temp°"],
    "ppg": ["ppg", "pulse", "photopleth"],
    "bp": ["bp", "blood press", "systolic", "diastolic", "arterial"],
}


def detect_encoding(file_path: Path) -> str:
    """Detect file encoding."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "mac_roman"):
        try:
            with open(file_path, encoding=enc) as f:
                f.read(4096)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"


def detect_separator(file_path: Path, encoding: str) -> str:
    """Detect field separator."""
    with open(file_path, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            tab_count = line.count("\t")
            comma_count = line.count(",")
            semicolon_count = line.count(";")
            if tab_count >= comma_count and tab_count >= semicolon_count:
                return "\t"
            elif comma_count >= semicolon_count:
                return ","
            return ";"
    return ","


def find_header_row(file_path: Path, encoding: str) -> int:
    """Find the header row in a LabChart export (may have metadata rows)."""
    with open(file_path, encoding=encoding) as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//", '"')):
                continue
            # LabChart headers often contain "Time" and channel names
            if any(
                kw in stripped.lower()
                for kw in ("time", "ecg", "eda", "emg", "resp", "channel")
            ):
                return i
            # If numeric-looking, it's data, so header is previous
            try:
                float(stripped.split("\t" if "\t" in stripped else ",")[0])
                return max(0, i - 1)
            except ValueError:
                continue
    return 0


def detect_channel_type(col_name: str) -> str:
    """Detect channel type from column name."""
    name_lower = col_name.lower().strip()
    for ch_type, keywords in _CHANNEL_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return ch_type
    return "unknown"


def parse_labchart(
    file_path: Path,
    encoding: Optional[str] = None,
    time_unit: str = "s",
) -> tuple[pd.DataFrame, dict]:
    """Parse a LabChart CSV/TXT export.

    Args:
        file_path: Path to export file.
        encoding: File encoding (auto-detected if None).
        time_unit: Time column unit ('s', 'ms', 'min').

    Returns:
        Tuple of (DataFrame, metadata).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"LabChart export not found: {file_path}")

    if encoding is None:
        encoding = detect_encoding(file_path)

    sep = detect_separator(file_path, encoding)
    header_row = find_header_row(file_path, encoding)

    df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=sep,
        header=header_row,
        low_memory=False,
    )

    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]

    # Identify time column
    time_col: Optional[str] = None
    for col in df.columns:
        if col.lower().startswith("time"):
            time_col = col
            break

    # If no time column, try first numeric column
    if time_col is None:
        for col in df.columns:
            try:
                pd.to_numeric(df[col].iloc[:10])
                time_col = col
                break
            except (ValueError, IndexError):
                continue

    # Convert time to seconds
    if time_col is not None:
        time_vals = pd.to_numeric(df[time_col], errors="coerce")
        unit_scale = {"ms": 0.001, "s": 1.0, "min": 60.0}
        df["timestamp_s"] = time_vals * unit_scale.get(time_unit, 1.0)

        # Estimate sampling rate
        valid_ts = df["timestamp_s"].dropna()
        if len(valid_ts) > 1:
            intervals = np.diff(valid_ts.values)
            median_interval = float(np.median(intervals))
            fs = 1.0 / median_interval if median_interval > 0 else 0
        else:
            fs = 0
    else:
        df["timestamp_s"] = np.arange(len(df))
        fs = 0

    # Detect channel types
    channels: dict[str, dict] = {}
    type_counts: dict[str, int] = {}
    for col in df.columns:
        if col in ("timestamp_s", time_col):
            continue
        ch_type = detect_channel_type(col)
        type_counts[ch_type] = type_counts.get(ch_type, 0) + 1
        channels[col] = {"type": ch_type, "original_name": col}

        # Rename to standardized name
        if ch_type != "unknown":
            if type_counts[ch_type] == 1:
                df = df.rename(columns={col: ch_type})
            else:
                new_name = f"{ch_type}_{type_counts[ch_type]}"
                df = df.rename(columns={col: new_name})

    meta: dict = {
        "source_format": "ADInstruments LabChart",
        "sampling_rate_hz": fs,
        "n_channels": len(channels),
        "n_samples": len(df),
        "duration_s": df["timestamp_s"].iloc[-1] - df["timestamp_s"].iloc[0] if len(df) > 1 else 0,
        "time_col": time_col,
        "channels": channels,
    }

    return df, meta


def print_summary(df: pd.DataFrame, meta: dict) -> None:
    """Print metadata summary."""
    print("=" * 60)
    print("  LabChart Data Summary")
    print("=" * 60)
    print(f"  Sampling rate:    {meta.get('sampling_rate_hz', 'N/A'):.1f} Hz")
    print(f"  Channels:         {meta.get('n_channels', 'N/A')}")
    print(f"  Total samples:    {meta.get('n_samples', 'N/A')}")
    print(f"  Duration:         {meta.get('duration_s', 'N/A'):.2f} s")
    print("  Channel details:")
    for name, ch in meta.get("channels", {}).items():
        detected = "✓" if ch["type"] != "unknown" else "?"
        print(f"    [{detected}] {name} → {ch['type']}")
    print(f"\n  Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    - {col}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import LabChart exports.")
    parser.add_argument("input", type=Path, help="Path to LabChart CSV/TXT.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path.")
    parser.add_argument("--summary", action="store_true", help="Print summary.")
    parser.add_argument("--encoding", type=str, default=None, help="File encoding.")
    parser.add_argument("--time-unit", type=str, default="s", choices=["ms", "s", "min"])
    args = parser.parse_args()

    df, meta = parse_labchart(args.input, encoding=args.encoding, time_unit=args.time_unit)

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
