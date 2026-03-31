#!/usr/bin/env python3
"""Generic physiological signal CSV importer.

Auto-detects time columns and signal columns, supports CSV/TSV/Excel,
and outputs a standardized DataFrame.

Usage:
    python import_physio_csv.py signals.csv --output result.parquet --summary
    python import_physio_csv.py signals.csv --time-col Time --signal-cols ECG,EDA,EMG
    python import_physio_csv.py signals.tsv --sep tab --fs 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"

_CHANNEL_KEYWORDS: dict[str, list[str]] = {
    "ecg": ["ecg", "ekg", "ecg ", "ekg ", "cardio", "heart"],
    "eda": ["eda", "gsr", "galvanic", "skin conduct", "scl", "scr"],
    "emg": ["emg", "muscle", "electromy"],
    "resp": ["resp", "breath", "respirat", "pneumo", "insp"],
    "temp": ["temp", "skin temp", "therm"],
    "ppg": ["ppg", "pulse", "photopleth"],
    "bp": ["bp", "blood press", "systolic", "diastolic"],
    "hr": ["hr", "heart rate", "bpm"],
}


def detect_channel_type(col_name: str) -> str:
    """Detect channel type from column name."""
    name_lower = col_name.lower().strip()
    for ch_type, keywords in _CHANNEL_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return ch_type
    return "unknown"


def auto_detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the time column in a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Column name or None.
    """
    time_keywords = ["time", "timestamp", "sample", "index", "t_", "sec", "seconds"]

    for col in df.columns:
        col_lower = col.lower().strip()
        for kw in time_keywords:
            if kw in col_lower:
                return col

    # Try first numeric column that looks like time (monotonically increasing)
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) < 10:
            continue
        if vals.is_monotonic_increasing and (vals.iloc[-1] - vals.iloc[0]) / len(vals) < 1:
            return col

    return None


def auto_detect_signal_columns(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
) -> list[str]:
    """Auto-detect signal columns (numeric, non-time).

    Args:
        df: Input DataFrame.
        time_col: Known time column to exclude.

    Returns:
        List of signal column names.
    """
    exclude = {time_col} if time_col else set()
    signals: list[str] = []

    for col in df.columns:
        if col in exclude:
            continue
        # Check if mostly numeric
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().mean() > 0.5:  # >50% numeric
            signals.append(col)

    return signals


def import_physio_csv(
    file_path: Path,
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    time_col: Optional[str] = None,
    signal_cols: Optional[list[str]] = None,
    fs: Optional[float] = None,
    time_unit: str = "s",
) -> tuple[pd.DataFrame, dict]:
    """Import a generic CSV/TSV/Excel physiological signal file.

    Args:
        file_path: Path to data file.
        sep: Field separator (auto-detected if None).
        encoding: File encoding.
        time_col: Time column name (auto-detected if None).
        signal_cols: Signal column names (auto-detected if None).
        fs: Sampling rate (auto-estimated if None).
        time_unit: Time unit ('s', 'ms', 'min').

    Returns:
        Tuple of (DataFrame, metadata).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    # Read based on format
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        # Detect separator
        if sep == "tab":
            file_sep = "\t"
        elif sep == "comma":
            file_sep = ","
        elif sep == "semicolon":
            file_sep = ";"
        elif sep:
            file_sep = sep
        else:
            # Auto-detect
            with open(file_path, encoding=encoding or "utf-8", errors="replace") as f:
                first_line = f.readline()
            tab_count = first_line.count("\t")
            comma_count = first_line.count(",")
            semicolon_count = first_line.count(";")
            file_sep = "\t" if tab_count >= max(comma_count, semicolon_count) else (
                "," if comma_count >= semicolon_count else ";"
            )

        df = pd.read_csv(
            file_path,
            sep=file_sep,
            encoding=encoding or "utf-8",
            engine="python",
            on_bad_lines="skip",
        )

    df.columns = [c.strip().strip('"') for c in df.columns]

    # Auto-detect time column
    if time_col is None:
        time_col = auto_detect_time_column(df)

    # Auto-detect signal columns
    if signal_cols is None:
        signal_cols = auto_detect_signal_columns(df, time_col)

    # Build timestamp
    if time_col is not None:
        unit_scale = {"ms": 0.001, "s": 1.0, "min": 60.0}
        ts = pd.to_numeric(df[time_col], errors="coerce") * unit_scale.get(time_unit, 1.0)
        df["timestamp_s"] = ts
    else:
        df["timestamp_s"] = np.arange(len(df))
        if fs is not None:
            df["timestamp_s"] = df["timestamp_s"] / fs

    # Estimate sampling rate
    if fs is None and time_col is not None:
        valid_ts = df["timestamp_s"].dropna()
        if len(valid_ts) > 1:
            intervals = np.diff(valid_ts.values)
            median_interval = float(np.median(intervals))
            fs = 1.0 / median_interval if median_interval > 0 else 0
        else:
            fs = 0
    elif fs is None:
        fs = 0

    # Detect channel types
    channels: dict[str, dict] = {}
    type_counts: dict[str, int] = {}
    for col in signal_cols:
        if col not in df.columns:
            continue
        ch_type = detect_channel_type(col)
        type_counts[ch_type] = type_counts.get(ch_type, 0) + 1
        channels[col] = {"type": ch_type, "original_name": col}

    # Rename detected channels
    for col, ch in channels.items():
        if ch["type"] != "unknown":
            new_name = ch["type"] if type_counts[ch["type"]] == 1 else f"{ch['type']}_{type_counts[ch['type']]}"
            df = df.rename(columns={col: new_name})

    meta: dict = {
        "source_format": "Generic CSV/TSV",
        "sampling_rate_hz": fs,
        "n_channels": len(channels),
        "n_samples": len(df),
        "time_col": time_col,
        "signal_cols": signal_cols,
        "channels": channels,
    }

    if len(df) > 1 and "timestamp_s" in df.columns:
        valid_ts = df["timestamp_s"].dropna()
        if len(valid_ts) > 1:
            meta["duration_s"] = float(valid_ts.iloc[-1] - valid_ts.iloc[0])

    return df, meta


def print_summary(df: pd.DataFrame, meta: dict) -> None:
    """Print metadata summary."""
    print("=" * 60)
    print("  Physiological Signal Import Summary")
    print("=" * 60)
    print(f"  Format:           {meta.get('source_format', 'N/A')}")
    print(f"  Sampling rate:    {meta.get('sampling_rate_hz', 'N/A'):.1f} Hz")
    print(f"  Channels:         {meta.get('n_channels', 'N/A')}")
    print(f"  Total samples:    {meta.get('n_samples', 'N/A')}")
    print(f"  Duration:         {meta.get('duration_s', 'N/A'):.2f} s")
    print(f"  Time column:      {meta.get('time_col', 'auto-detected')}")
    print("  Channel details:")
    for name, ch in meta.get("channels", {}).items():
        detected = "✓" if ch["type"] != "unknown" else "?"
        print(f"    [{detected}] {name} → {ch['type']}")
    print(f"\n  Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    - {col}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import generic physiological signal CSV.")
    parser.add_argument("input", type=Path, help="Path to data file (.csv, .tsv, .xlsx).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path.")
    parser.add_argument("--summary", action="store_true", help="Print summary.")
    parser.add_argument("--sep", type=str, default=None, choices=["tab", "comma", "semicolon"])
    parser.add_argument("--encoding", type=str, default=None)
    parser.add_argument("--time-col", type=str, default=None, help="Time column name.")
    parser.add_argument(
        "--signal-cols", type=str, default=None,
        help="Comma-separated signal column names.",
    )
    parser.add_argument("--fs", type=float, default=None, help="Sampling rate (Hz).")
    parser.add_argument("--time-unit", type=str, default="s", choices=["ms", "s", "min"])
    args = parser.parse_args()

    signal_cols = args.signal_cols.split(",") if args.signal_cols else None

    df, meta = import_physio_csv(
        args.input,
        sep=args.sep,
        encoding=args.encoding,
        time_col=args.time_col,
        signal_cols=signal_cols,
        fs=args.fs,
        time_unit=args.time_unit,
    )

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
