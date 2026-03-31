#!/usr/bin/env python3
"""Import FaceReader CSV output data with automatic encoding detection and time-range filtering."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("psyphiclaw-face-import")

# Color scheme
PRIMARY = "#4A90D9"
SECONDARY = "#E74C3C"


def detect_encoding(path: Path) -> str:
    """Auto-detect CSV encoding (UTF-8 or GBK)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"):
        try:
            raw.decode(enc)
            logger.info("Detected encoding: %s", enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    logger.warning("Encoding detection fallback to latin-1")
    return "latin-1"


def read_csv(path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
    """Read FaceReader CSV with robust error handling."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    enc = encoding or detect_encoding(path)
    logger.info("Reading %s with encoding=%s", path, enc)

    # FaceReader CSVs often use semicolon separators and comma decimals
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=sep, on_bad_lines="skip")
            if len(df.columns) > 2:
                logger.info("Parsed %d rows x %d columns (sep=%r)", len(df), len(df.columns), sep)
                return df
        except pd.errors.ParserError:
            continue

    raise ValueError(f"Could not parse CSV: {path}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and normalize."""
    df.columns = df.columns.str.strip()
    return df


def extract_time_column(df: pd.DataFrame) -> str:
    """Find the timestamp column."""
    candidates = ["Timestamp", "TimeStamp", "timestamp", "Time", "time_ms"]
    for c in candidates:
        if c in df.columns:
            return c
    # Try first numeric column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("Cannot find a timestamp column")


def filter_by_time(df: pd.DataFrame, col: str, start_ms: float, end_ms: float) -> pd.DataFrame:
    """Filter DataFrame by time range in milliseconds."""
    before = len(df)
    df = df[(df[col] >= start_ms) & (df[col] <= end_ms)].copy()
    logger.info("Time filter %s-%sms: %d -> %d rows", start_ms, end_ms, before, len(df))
    return df


def compute_summary(df: pd.DataFrame, time_col: str) -> dict:
    """Compute data summary statistics."""
    total_rows = len(df)
    if total_rows < 2:
        return {"total_rows": total_rows, "error": "Insufficient data for summary"}

    timestamps = df[time_col].dropna()
    if len(timestamps) < 2:
        return {"total_rows": total_rows, "error": "Insufficient timestamps"}

    duration_ms = timestamps.iloc[-1] - timestamps.iloc[0]
    duration_s = duration_ms / 1000.0
    sr = (total_rows - 1) / (duration_ms / 1000.0) if duration_ms > 0 else 0

    # Face presence
    face_cols = [c for c in df.columns if "facepresence" in c.lower() or "face_presence" in c.lower()]
    valid_frames = total_rows
    if face_cols:
        valid_frames = int(df[face_cols[0]].sum()) if pd.api.types.is_numeric_dtype(df[face_cols[0]]) else total_rows

    # Column groups
    au_cols = sorted([c for c in df.columns if c.lower().startswith("actionunit")])
    vad_cols = [c for c in df.columns if c in ("Valence", "Arousal", "Dominance")]
    emo_cols = [c for c in df.columns if c in ("Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted", "Contempt")]
    head_cols = [c for c in df.columns if "head" in c.lower() and ("roll" in c.lower() or "pitch" in c.lower() or "yaw" in c.lower())]
    gaze_cols = [c for c in df.columns if "gaze" in c.lower()]

    return {
        "total_rows": total_rows,
        "valid_face_frames": valid_frames,
        "face_coverage_pct": round(valid_frames / total_rows * 100, 1) if total_rows > 0 else 0,
        "sampling_rate_hz": round(sr, 2),
        "duration_s": round(duration_s, 2),
        "time_range_ms": [float(timestamps.iloc[0]), float(timestamps.iloc[-1])],
        "columns": {
            "action_units": len(au_cols),
            "vad": len(vad_cols),
            "emotions": len(emo_cols),
            "head_orientation": len(head_cols),
            "gaze": len(gaze_cols),
        },
        "missing_pct": {c: round(df[c].isna().mean() * 100, 1) for c in df.columns if df[c].isna().any()},
    }


def format_summary(summary: dict) -> str:
    """Format summary as readable text."""
    lines = [
        "=" * 50,
        "📊 FaceReader Data Summary",
        "=" * 50,
        f"  Total frames:      {summary['total_rows']}",
        f"  Valid face frames: {summary.get('valid_face_frames', 'N/A')}",
        f"  Face coverage:     {summary.get('face_coverage_pct', 'N/A')}%",
        f"  Sampling rate:     {summary.get('sampling_rate_hz', 'N/A')} Hz",
        f"  Duration:          {summary.get('duration_s', 'N/A')} s",
    ]
    if "time_range_ms" in summary:
        lines.append(f"  Time range:        {summary['time_range_ms'][0]:.0f} - {summary['time_range_ms'][1]:.0f} ms")
    cols = summary.get("columns", {})
    lines += [
        "-" * 50,
        "  Column groups:",
        f"    Action Units:     {cols.get('action_units', 0)}",
        f"    VAD dimensions:   {cols.get('vad', 0)}",
        f"    Emotions:         {cols.get('emotions', 0)}",
        f"    Head orientation: {cols.get('head_orientation', 0)}",
        f"    Gaze:             {cols.get('gaze', 0)}",
    ]
    if summary.get("missing_pct"):
        lines.append("-" * 50)
        lines.append("  Columns with missing data:")
        for c, pct in summary["missing_pct"].items():
            lines.append(f"    {c}: {pct}%")
    lines.append("=" * 50)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import FaceReader CSV data")
    parser.add_argument("csv_path", type=Path, help="Path to FaceReader CSV file")
    parser.add_argument("--encoding", type=str, default=None, help="File encoding (auto-detect if omitted)")
    parser.add_argument("--time-range", nargs=2, type=float, metavar=("START_MS", "END_MS"), help="Time range filter (ms)")
    parser.add_argument("--output", type=Path, default=None, help="Save filtered CSV to this path")
    parser.add_argument("--summary", action="store_true", help="Print data summary")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    df = read_csv(args.csv_path, args.encoding)
    df = standardize_columns(df)
    time_col = extract_time_column(df)

    if args.time_range:
        df = filter_by_time(df, time_col, args.time_range[0], args.time_range[1])

    if args.summary:
        summary = compute_summary(df, time_col)
        print(format_summary(summary))

    if args.output:
        df.to_csv(args.output, index=False)
        logger.info("Saved filtered data to %s", args.output)

    return df


if __name__ == "__main__":
    main()
