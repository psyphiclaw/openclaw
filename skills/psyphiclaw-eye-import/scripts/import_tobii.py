#!/usr/bin/env python3
"""Import Tobii Pro Lab TSV eye-tracking exports.

Converts Tobii TSV files into a standardized DataFrame with gaze coordinates,
pupil diameter, fixation events, and metadata summary.

Usage:
    python import_tobii.py recording.tsv --output result.parquet --summary
    python import_tobii.py recording.tsv --encoding latin-1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# PsyPhiClaw color palette
COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"

# Known Tobii TSV column name mappings → standardized names
_COLUMN_MAP: dict[str, str] = {
    # Timestamps
    "RecordingTimestamp": "timestamp",
    "GazeTimeStamp": "timestamp",
    "Timestamp": "timestamp",
    # Gaze
    "GazePointX": "gaze_x",
    "GazePointY": "gaze_y",
    "GazePointLeftX": "gaze_left_x",
    "GazePointLeftY": "gaze_left_y",
    "GazePointRightX": "gaze_right_x",
    "GazePointRightY": "gaze_right_y",
    # Pupil
    "PupilLeft": "pupil_left",
    "PupilRight": "pupil_right",
    "PupilLeftDiameter": "pupil_left",
    "PupilRightDiameter": "pupil_right",
    # Fixation
    "FixationIndex": "fixation_index",
    "FixationStartTime": "fixation_start",
    "FixationEndTime": "fixation_end",
    "FixationDuration": "fixation_duration",
    # Saccade
    "SaccadeStartTime": "saccade_start",
    "SaccadeEndTime": "saccade_end",
    "SaccadeDuration": "saccade_duration",
    # Validity
    "ValidityLeft": "validity_left",
    "ValidityRight": "validity_right",
    # Event
    "Event": "event",
    "EventData": "event_data",
}


def detect_encoding(file_path: Path) -> str:
    """Detect file encoding using chardet or fallback heuristics."""
    try:
        import chardet  # type: ignore[import-untyped]
        with open(file_path, "rb") as f:
            raw = f.read(100_000)
        result = chardet.detect(raw)
        enc = result["encoding"] or "utf-8"
        return enc.replace("UTF-8-SIG", "utf-8")
    except ImportError:
        pass

    # Fallback: try common encodings
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(file_path, encoding=enc) as f:
                f.read(4096)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"


def detect_separator(file_path: Path, encoding: str) -> str:
    """Detect whether the file uses tab or comma separation."""
    with open(file_path, encoding=encoding) as f:
        first_line = f.readline()

    tab_count = first_line.count("\t")
    comma_count = first_line.count(",")

    if tab_count >= comma_count:
        return "\t"
    return ","


def parse_tobii_tsv(
    file_path: Path,
    encoding: Optional[str] = None,
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """Parse a Tobii Pro Lab TSV export into a standardized DataFrame.

    Args:
        file_path: Path to the Tobii TSV file.
        encoding: Character encoding. Auto-detected if None.
        sep: Field separator. Auto-detected if None.

    Returns:
        Standardized DataFrame with columns: timestamp, gaze_x, gaze_y,
        pupil_left, pupil_right, fixation_index, event, etc.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Tobii TSV file not found: {file_path}")

    # Auto-detect encoding and separator
    if encoding is None:
        encoding = detect_encoding(file_path)
    if sep is None:
        sep = detect_separator(file_path, encoding)

    # Find the header row (Tobii TSVs may have preamble rows)
    header_row_index = _find_header_row(file_path, encoding)

    # Read the data
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=sep,
        header=header_row_index,
        low_memory=False,
    )

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Rename known columns to standardized names
    rename_map = {
        orig: std
        for orig, std in _COLUMN_MAP.items()
        if orig in df.columns
    }
    df = df.rename(columns=rename_map)

    # Parse fixation events if present in an Event column
    if "event" in df.columns:
        df["event"] = df["event"].astype(str).str.strip()

    return df


def _find_header_row(file_path: Path, encoding: str) -> int:
    """Find the header row in a Tobii TSV (may have preamble)."""
    with open(file_path, encoding=encoding) as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            # Header rows typically contain common Tobii column names
            if any(
                kw in stripped
                for kw in (
                    "RecordingTimestamp",
                    "GazeTimeStamp",
                    "GazePointX",
                    "GazePointY",
                    "Timestamp",
                    "PupilLeft",
                    "FixationIndex",
                )
            ):
                return i
    return 0


def extract_metadata(df: pd.DataFrame) -> dict:
    """Extract metadata summary from a Tobii DataFrame.

    Args:
        df: Standardized Tobii DataFrame.

    Returns:
        Dictionary with metadata: duration, sampling_rate, n_fixations, etc.
    """
    meta: dict = {}

    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
        if len(ts) > 1:
            duration_ms = ts.iloc[-1] - ts.iloc[0]
            meta["duration_ms"] = duration_ms
            meta["duration_s"] = duration_ms / 1000.0

            # Estimate sampling rate from median inter-sample interval
            intervals = np.diff(ts.values)
            median_interval = float(np.median(intervals))
            meta["estimated_sampling_rate_hz"] = (
                1000.0 / median_interval if median_interval > 0 else 0
            )
            meta["median_sample_interval_ms"] = median_interval

    meta["total_samples"] = len(df)

    # Count events
    if "event" in df.columns:
        event_counts = df["event"].value_counts().to_dict()
        meta["event_counts"] = event_counts
        meta["n_fixations"] = sum(
            1 for e in event_counts if "fixation" in str(e).lower()
        )

    # Gaze data quality
    for col, key in [
        ("validity_left", "validity_left_ok_pct"),
        ("validity_right", "validity_right_ok_pct"),
    ]:
        if col in df.columns:
            valid = pd.to_numeric(df[col], errors="coerce")
            meta[key] = float(
                (valid <= 1).sum() / valid.notna().sum() * 100
                if valid.notna().sum() > 0
                else 0
            )

    return meta


def print_summary(df: pd.DataFrame, meta: dict) -> None:
    """Print a human-readable summary of the imported data."""
    print("=" * 60)
    print("  Tobii Eye-Tracking Data Summary")
    print("=" * 60)
    print(f"  Total samples:       {meta.get('total_samples', 'N/A')}")
    print(f"  Duration:            {meta.get('duration_s', 'N/A'):.2f} s")
    print(
        f"  Est. sampling rate: {meta.get('estimated_sampling_rate_hz', 'N/A'):.1f} Hz"
    )
    if "n_fixations" in meta:
        print(f"  Fixation events:     {meta['n_fixations']}")
    if "event_counts" in meta:
        print("  Event breakdown:")
        for ev, cnt in sorted(meta["event_counts"].items()):
            print(f"    {ev}: {cnt}")
    print(f"\n  Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    - {col}")
    print("=" * 60)


def plot_summary(df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Generate basic summary plots: gaze trajectory + pupil diameter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Gaze trajectory ---
    ax = axes[0]
    if "gaze_x" in df.columns and "gaze_y" in df.columns:
        gx = pd.to_numeric(df["gaze_x"], errors="coerce").dropna()
        gy = pd.to_numeric(df["gaze_y"], errors="coerce").dropna()
        min_len = min(len(gx), len(gy))
        gx, gy = gx.iloc[:min_len], gy.iloc[:min_len]

        # Downsample for plotting performance
        step = max(1, min_len // 5000)
        ax.scatter(
            gx.iloc[::step],
            gy.iloc[::step],
            c=np.arange(min_len)[::step],
            cmap="Blues",
            s=2,
            alpha=0.5,
        )
        ax.set_xlabel("Gaze X (px)")
        ax.set_ylabel("Gaze Y (px)")
        ax.set_title("Gaze Trajectory")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No gaze data", ha="center", va="center")
        ax.set_title("Gaze Trajectory")

    # --- Pupil diameter ---
    ax = axes[1]
    pupil_cols = [c for c in ("pupil_left", "pupil_right") if c in df.columns]
    if pupil_cols:
        for col in pupil_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            step = max(1, len(vals) // 3000)
            ax.plot(
                vals.index[::step],
                vals.values[::step],
                alpha=0.7,
                label=col.replace("_", " ").title(),
                color=COLOR_PRIMARY if "left" in col else COLOR_ACCENT,
            )
        ax.set_xlabel("Sample")
        ax.set_ylabel("Pupil Diameter")
        ax.set_title("Pupil Diameter over Time")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No pupil data", ha="center", va="center")
        ax.set_title("Pupil Diameter")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Summary plot saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Tobii Pro Lab TSV eye-tracking data."
    )
    parser.add_argument("input", type=Path, help="Path to Tobii TSV file.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (.parquet or .csv).",
    )
    parser.add_argument(
        "--encoding", type=str, default=None,
        help="File encoding (auto-detected if omitted).",
    )
    parser.add_argument(
        "--sep", type=str, default=None, choices=["tab", "comma"],
        help="Field separator (auto-detected if omitted).",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print metadata summary to stdout.",
    )
    parser.add_argument(
        "--plot", type=Path, default=None,
        help="Save summary plot to this path.",
    )
    args = parser.parse_args()

    sep_char = "\t" if args.sep == "tab" else ("," if args.sep == "comma" else None)

    df = parse_tobii_tsv(args.input, encoding=args.encoding, sep=sep_char)
    meta = extract_metadata(df)

    if args.summary:
        print_summary(df, meta)

    if args.plot:
        plot_summary(df, str(args.plot))

    if args.output:
        suffix = args.output.suffix.lower()
        if suffix == ".parquet":
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Data saved to: {args.output}")


if __name__ == "__main__":
    main()
