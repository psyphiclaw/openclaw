#!/usr/bin/env python3
"""Align multimodal data using shared event markers (e.g. stimulus onset).

Supports nearest-neighbour and interpolation matching strategies.

Usage:
    python align_marker.py \
        --files eeg.csv face.csv physio.csv \
        --timestamp-cols timestamp timestamp Timestamp \
        --event-marker stimulus_onset \
        --method nearest \
        --output aligned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BLUE = "#4A90D9"
RED = "#E74C3C"


def load_data(path: str, ts_col: str) -> pd.DataFrame:
    """Load a CSV file and ensure the timestamp column is in milliseconds."""
    df = pd.read_csv(path)
    if ts_col not in df.columns:
        raise ValueError(f"'{ts_col}' not in {path}. Columns: {list(df.columns)}")
    ts = pd.to_numeric(df[ts_col], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Non-numeric timestamps in {path}")
    df["_timestamp_ms"] = ts
    df["_source_file"] = Path(path).stem
    return df


def find_event_marker(df: pd.DataFrame, marker: str) -> float:
    """Find the timestamp (ms) of an event marker.

    Looks for a column named *marker* containing the event time,
    or a boolean column named *marker* indicating event onset.
    """
    # Column containing the event time value directly
    for col in (marker, marker.upper(), marker.lower()):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                return float(vals.iloc[0])

    # Boolean column: event onset is True → use first True row's timestamp
    for col in df.columns:
        if col.lower() == marker.lower():
            series = df[col]
            if series.dtype == bool or series.dropna().isin([0, 1, "True", "False"]).all():
                bool_s = series.astype(bool)
                if bool_s.any():
                    idx = bool_s.idxmax()
                    return float(df.loc[idx, "_timestamp_ms"])

    # Search for string "stimulus_onset" or similar in all columns
    for col in df.columns:
        if df[col].dtype == object:
            mask = df[col].astype(str).str.lower().str.contains(marker.lower(), na=False)
            if mask.any():
                idx = mask.idxmax()
                return float(df.loc[idx, "_timestamp_ms"])

    raise ValueError(f"Event marker '{marker}' not found in DataFrame columns.")


def align_nearest(
    ref_time: float,
    dfs: list[pd.DataFrame],
    names: list[str],
) -> dict[str, Any]:
    """Align each modality to the reference event using nearest-neighbour."""
    offsets: list[dict[str, Any]] = []
    for df, name in zip(dfs, names):
        ts = df["_timestamp_ms"].values
        idx = int(np.argmin(np.abs(ts - ref_time)))
        offset = float(ts[idx] - ref_time)
        offsets.append({
            "modality": name,
            "event_index": idx,
            "nearest_timestamp_ms": round(float(ts[idx]), 4),
            "offset_ms": round(offset, 4),
            "distance_ms": round(abs(offset), 4),
        })
    return {
        "method": "nearest_neighbour",
        "reference_event_ms": round(ref_time, 4),
        "offsets": offsets,
    }


def align_interpolation(
    ref_time: float,
    dfs: list[pd.DataFrame],
    names: list[str],
) -> dict[str, Any]:
    """Align using linear interpolation of timestamps around the event."""
    offsets: list[dict[str, Any]] = []
    for df, name in zip(dfs, names):
        ts = df["_timestamp_ms"].values
        # Clamp to data range
        t = np.clip(ref_time, ts.min(), ts.max())
        # Linear interpolation index
        idx_float = np.interp(t, ts, np.arange(len(ts)))
        offsets.append({
            "modality": name,
            "interpolated_index": round(float(idx_float), 4),
            "ref_time_ms": round(ref_time, 4),
            "clamped": t != ref_time,
        })
    return {
        "method": "interpolation",
        "reference_event_ms": round(ref_time, 4),
        "offsets": offsets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align multimodal data using shared event markers."
    )
    parser.add_argument(
        "--files", nargs="+", required=True, help="Input CSV files."
    )
    parser.add_argument(
        "--timestamp-cols", nargs="+", required=True,
        help="Timestamp column name for each file (same order as --files).",
    )
    parser.add_argument(
        "--event-marker", required=True,
        help="Event marker column name or value to align on.",
    )
    parser.add_argument(
        "--reference-file", default=None,
        help="File to use as temporal reference (default: first file).",
    )
    parser.add_argument(
        "--method", choices=["nearest", "interpolation", "both"], default="nearest",
        help="Alignment method.",
    )
    parser.add_argument(
        "--output", required=True, help="Output JSON path.",
    )
    args = parser.parse_args()

    if len(args.files) != len(args.timestamp_cols):
        parser.error(
            f"--files ({len(args.files)}) and --timestamp-cols "
            f"({len(args.timestamp_cols)}) must match in length."
        )

    dfs: list[pd.DataFrame] = []
    names: list[str] = []
    for fpath, ts_col in zip(args.files, args.timestamp_cols):
        df = load_data(fpath, ts_col)
        dfs.append(df)
        names.append(df["_source_file"].iloc[0])

    # Determine reference
    ref_idx = 0
    if args.reference_file:
        ref_stem = Path(args.reference_file).stem
        if ref_stem in names:
            ref_idx = names.index(ref_stem)

    ref_time = find_event_marker(dfs[ref_idx], args.event_marker)
    print(f"📍 Reference event found at {ref_time:.2f} ms in '{names[ref_idx]}'")

    result: dict[str, Any] = {"event_marker": args.event_marker}
    if args.method in ("nearest", "both"):
        result["nearest"] = align_nearest(ref_time, dfs, names)
    if args.method in ("interpolation", "both"):
        result["interpolation"] = align_interpolation(ref_time, dfs, names)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Alignment written to {out_path}")
    for method_data in (v for k, v in result.items() if isinstance(v, dict)):
        for off in method_data.get("offsets", []):
            print(f"  [{off['modality']}] offset={off.get('offset_ms', 'N/A')} ms")


if __name__ == "__main__":
    main()
