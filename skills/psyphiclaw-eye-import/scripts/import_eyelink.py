#!/usr/bin/env python3
"""Import EyeLink ASC eye-tracking files.

Parses EyeLink .asc files including header metadata, binocular gaze data,
and event annotations (fixations, saccades, blinks). Outputs a unified
DataFrame compatible with other PsyPhiClaw tools.

Usage:
    python import_eyelink.py recording.asc --output result.parquet --summary
    python import_eyelink.py recording.asc --sampling-rate 500
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"

# EyeLink binocular sample regex (with tracking mode prefix)
# Format: ... x_left y_left pupil_left x_right y_right pupil_right ...
# or monocular: ... x y pupil ...
_SAMPLE_PATTERN = re.compile(
    r"^(\d+)\s+"  # timestamp
    r"([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+"  # left: x, y, pupil
    r"([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s*"  # right: x, y, pupil
)

# Monocular sample (3 values: x, y, pupil)
_SAMPLE_MONO_PATTERN = re.compile(
    r"^(\d+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s*$"
)

# Event patterns
_EVENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "fixation_start": re.compile(r"^EFIX\s+(L|R|B)\s+(\d+)\s+(\d+)\s+(\d+)"),
    "fixation_end": re.compile(
        r"^EFIX\s+(L|R|B)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
        r"([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)"
    ),
    "saccade": re.compile(
        r"^ESACC\s+(L|R|B)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
        r"([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)"
    ),
    "blink": re.compile(r"^EBLINK\s+(L|R|B)\s+(\d+)\s+(\d+)\s+(\d+)"),
}


def parse_header(lines: list[str]) -> dict:
    """Extract metadata from EyeLink ASC header block.

    Args:
        lines: All lines from the ASC file.

    Returns:
        Dictionary with metadata: sampling_rate, calibration_method, etc.
    """
    meta: dict = {"source_format": "EyeLink ASC"}

    for line in lines:
        line = line.strip()
        if line.startswith("**") and "RECORDING" not in line:
            continue
        if line.startswith("** CONVERTED") or line.startswith("** DATE"):
            meta["date"] = line.replace("**", "").strip()
        if "SAMPLE RATE" in line.upper():
            match = re.search(r"(\d+)", line)
            if match:
                meta["sampling_rate_hz"] = int(match.group(1))
        if "CALIBRATION" in line.upper():
            meta["calibration_method"] = line.strip()
        if "DISPLAY" in line.upper() and "COORD" in line.upper():
            meta["display_coords"] = line.strip()
        if line.startswith("** TIMESTAMP:"):
            meta["timestamp_info"] = line.strip()

    # Detect tracking mode from sample lines
    meta["tracking_mode"] = "binocular"

    return meta


def parse_samples(lines: list[str]) -> pd.DataFrame:
    """Parse EyeLink sample lines into a DataFrame.

    Args:
        lines: Non-header, non-event lines from the ASC file.

    Returns:
        DataFrame with columns: timestamp, gaze_left_x, gaze_left_y, pupil_left,
        gaze_right_x, gaze_right_y, pupil_right.
    """
    records: list[dict] = []
    is_monocular = False

    for line in lines:
        line = line.strip()
        if not line or line.startswith(("*", "//", "#")):
            continue

        # Try binocular
        m = _SAMPLE_PATTERN.match(line)
        if m:
            records.append({
                "timestamp": int(m.group(1)),
                "gaze_left_x": float(m.group(2)),
                "gaze_left_y": float(m.group(3)),
                "pupil_left": float(m.group(4)),
                "gaze_right_x": float(m.group(5)),
                "gaze_right_y": float(m.group(6)),
                "pupil_right": float(m.group(7)),
            })
            continue

        # Try monocular
        m = _SAMPLE_MONO_PATTERN.match(line)
        if m:
            is_monocular = True
            records.append({
                "timestamp": int(m.group(1)),
                "gaze_left_x": float(m.group(2)),
                "gaze_left_y": float(m.group(3)),
                "pupil_left": float(m.group(4)),
                "gaze_right_x": np.nan,
                "gaze_right_y": np.nan,
                "pupil_right": np.nan,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if is_monocular:
        df = df.rename(columns={
            "gaze_left_x": "gaze_x",
            "gaze_left_y": "gaze_y",
            "pupil_left": "pupil",
        })

    # Compute average gaze and pupil
    if not is_monocular:
        df["gaze_x"] = df[["gaze_left_x", "gaze_right_x"]].mean(axis=1)
        df["gaze_y"] = df[["gaze_left_y", "gaze_right_y"]].mean(axis=1)
        df["pupil_avg"] = df[["pupil_left", "pupil_right"]].mean(axis=1)

    return df


def parse_events(lines: list[str]) -> list[dict]:
    """Parse EyeLink event lines (fixations, saccades, blinks).

    Args:
        lines: All lines from the ASC file.

    Returns:
        List of event dictionaries with type, eye, timestamps, and metrics.
    """
    events: list[dict] = []

    for line in lines:
        line = line.strip()
        if not line or not line.startswith(("EFIX", "ESACC", "EBLINK")):
            continue

        # Fixation end (full line with position data)
        m = _EVENT_PATTERNS["fixation_end"].match(line)
        if m:
            events.append({
                "event": "FixationEnd",
                "eye": m.group(1),
                "start_time": int(m.group(2)),
                "end_time": int(m.group(3)),
                "duration_ms": int(m.group(4)),
                "avg_x": float(m.group(5)),
                "avg_y": float(m.group(6)),
                "avg_pupil": float(m.group(7)),
            })
            continue

        # Fixation start (minimal line)
        m = _EVENT_PATTERNS["fixation_start"].match(line)
        if m:
            events.append({
                "event": "FixationStart",
                "eye": m.group(1),
                "start_time": int(m.group(2)),
                "end_time": int(m.group(3)),
                "duration_ms": int(m.group(4)),
                "avg_x": np.nan,
                "avg_y": np.nan,
                "avg_pupil": np.nan,
            })
            continue

        # Saccade
        m = _EVENT_PATTERNS["saccade"].match(line)
        if m:
            events.append({
                "event": "Saccade",
                "eye": m.group(1),
                "start_time": int(m.group(2)),
                "end_time": int(m.group(3)),
                "duration_ms": int(m.group(4)),
                "start_x": float(m.group(5)),
                "start_y": float(m.group(6)),
                "end_x": float(m.group(7)),
                "end_y": float(m.group(8)),
                "amplitude": float(m.group(9)),
                "peak_velocity": float(m.group(10)),
            })
            continue

        # Blink
        m = _EVENT_PATTERNS["blink"].match(line)
        if m:
            events.append({
                "event": "Blink",
                "eye": m.group(1),
                "start_time": int(m.group(2)),
                "end_time": int(m.group(3)),
                "duration_ms": int(m.group(4)),
            })

    return events


def merge_samples_events(
    df_samples: pd.DataFrame,
    events: list[dict],
) -> pd.DataFrame:
    """Merge sample data with events into a unified DataFrame.

    Adds fixation_index and event columns to the samples DataFrame.

    Args:
        df_samples: Gaze sample DataFrame.
        events: Parsed event list.

    Returns:
        Unified DataFrame with events annotated.
    """
    if df_samples.empty or not events:
        return df_samples

    df = df_samples.copy()
    df["fixation_index"] = np.nan
    df["event"] = ""

    fix_idx = 0
    for ev in events:
        if ev["event"] in ("FixationStart", "FixationEnd"):
            mask = (df["timestamp"] >= ev["start_time"]) & (
                df["timestamp"] <= ev["end_time"]
            )
            if ev["event"] == "FixationEnd":
                fix_idx += 1
                df.loc[mask, "fixation_index"] = fix_idx
        elif ev["event"] == "Saccade":
            mask = (df["timestamp"] >= ev["start_time"]) & (
                df["timestamp"] <= ev["end_time"]
            )
            df.loc[mask, "event"] = "Saccade"
        elif ev["event"] == "Blink":
            mask = (df["timestamp"] >= ev["start_time"]) & (
                df["timestamp"] <= ev["end_time"]
            )
            df.loc[mask, "event"] = "Blink"

    return df


def parse_eyelink_asc(
    file_path: Path,
    sampling_rate: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """Parse an EyeLink ASC file into a unified DataFrame + metadata.

    Args:
        file_path: Path to the .asc file.
        sampling_rate: Override sampling rate if not in header.

    Returns:
        Tuple of (unified DataFrame, metadata dict).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"EyeLink ASC file not found: {file_path}")

    with open(file_path, encoding="ascii", errors="replace") as f:
        lines = f.readlines()

    # Parse header (everything before first sample/event line)
    meta = parse_header(lines)

    if sampling_rate is not None:
        meta["sampling_rate_hz"] = sampling_rate

    # Parse all data lines
    df_samples = parse_samples(lines)
    events = parse_events(lines)

    meta["total_samples"] = len(df_samples)
    meta["total_events"] = len(events)

    if not df_samples.empty and "timestamp" in df_samples.columns:
        duration = df_samples["timestamp"].iloc[-1] - df_samples["timestamp"].iloc[0]
        meta["duration_ms"] = duration
        meta["duration_s"] = duration / 1000.0

    # Count event types
    from collections import Counter
    event_counts = Counter(e["event"] for e in events)
    meta["event_counts"] = dict(event_counts)

    # Merge
    unified = merge_samples_events(df_samples, events)

    # Attach events as attribute for downstream use
    unified.attrs["events"] = events
    unified.attrs["metadata"] = meta

    return unified, meta


def print_summary(df: pd.DataFrame, meta: dict) -> None:
    """Print metadata summary."""
    print("=" * 60)
    print("  EyeLink Eye-Tracking Data Summary")
    print("=" * 60)
    print(f"  Format:             {meta.get('source_format', 'N/A')}")
    print(f"  Sampling rate:      {meta.get('sampling_rate_hz', 'N/A')} Hz")
    print(f"  Calibration:        {meta.get('calibration_method', 'N/A')}")
    print(f"  Total samples:      {meta.get('total_samples', 'N/A')}")
    print(f"  Duration:           {meta.get('duration_s', 'N/A'):.2f} s")
    print(f"  Total events:       {meta.get('total_events', 'N/A')}")
    print("  Event breakdown:")
    for ev, cnt in sorted(meta.get("event_counts", {}).items()):
        print(f"    {ev}: {cnt}")
    print(f"\n  Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    - {col}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import EyeLink ASC eye-tracking data."
    )
    parser.add_argument("input", type=Path, help="Path to EyeLink .asc file.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (.parquet or .csv).",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=None,
        help="Override sampling rate (Hz).",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print metadata summary."
    )
    args = parser.parse_args()

    df, meta = parse_eyelink_asc(args.input, sampling_rate=args.sampling_rate)

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
