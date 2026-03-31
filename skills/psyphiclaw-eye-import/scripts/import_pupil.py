#!/usr/bin/env python3
"""Import Pupil Labs eye-tracking exports (CSV/JSON + surface tracking).

Parses gaze data, pupil diameter, and surface tracking data from
Pupil Labs Player exports. Supports both world and eye camera data.

Usage:
    python import_pupil.py pupil_exports/ --output result.parquet --summary
    python import_pupil.py pupil_exports/ --surface --surface-name screen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COLOR_PRIMARY = "#4A90D9"
COLOR_ACCENT = "#E74C3C"


def import_gaze_csv(file_path: Path) -> pd.DataFrame:
    """Import Pupil Labs gaze_positions.csv.

    Args:
        file_path: Path to gaze_positions.csv.

    Returns:
        DataFrame with gaze data.
    """
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "world_timestamp": "timestamp",
        "norm_pos_x": "gaze_x",
        "norm_pos_y": "gaze_y",
        "base_data": "base_data",
        "confidence": "confidence",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


def import_pupil_csv(file_path: Path) -> pd.DataFrame:
    """Import Pupil Labs pupil_positions.csv.

    Args:
        file_path: Path to pupil_positions.csv.

    Returns:
        DataFrame with pupil diameter data.
    """
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "world_timestamp": "timestamp",
        "diameter": "pupil_diameter",
        "diameter_3d": "pupil_diameter_3d",
        "confidence": "pupil_confidence",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


def import_surface_csv(file_path: Path) -> pd.DataFrame:
    """Import Pupil Labs surface gaze mapping data.

    Args:
        file_path: Path to surface gazes CSV.

    Returns:
        DataFrame with surface-mapped gaze positions.
    """
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "world_timestamp": "timestamp",
        "on_surf": "on_surface",
        "norm_pos_x": "surface_x",
        "norm_pos_y": "surface_y",
        "gaze_point_3d_x": "gaze_3d_x",
        "gaze_point_3d_y": "gaze_3d_y",
        "gaze_point_3d_z": "gaze_3d_z",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


def import_fixations_csv(file_path: Path) -> pd.DataFrame:
    """Import Pupil Labs fixations CSV.

    Args:
        file_path: Path to fixations.csv.

    Returns:
        DataFrame with fixation events.
    """
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "start_timestamp": "fixation_start",
        "end_timestamp": "fixation_end",
        "duration": "fixation_duration",
        "dispersion": "fixation_dispersion",
        "norm_pos_x": "fixation_x",
        "norm_pos_y": "fixation_y",
        "id": "fixation_index",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


def import_blinks_csv(file_path: Path) -> pd.DataFrame:
    """Import Pupil Labs blinks CSV.

    Args:
        file_path: Path to blinks.csv.

    Returns:
        DataFrame with blink events.
    """
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "start_timestamp": "blink_start",
        "end_timestamp": "blink_end",
        "duration": "blink_duration",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


def merge_gaze_pupil(
    gaze_df: pd.DataFrame,
    pupil_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge gaze and pupil DataFrames on timestamp (nearest match).

    Args:
        gaze_df: Gaze positions DataFrame.
        pupil_df: Pupil positions DataFrame.

    Returns:
        Merged DataFrame with both gaze and pupil data.
    """
    if gaze_df.empty or pupil_df.empty:
        return gaze_df if not gaze_df.empty else pupil_df

    gaze = gaze_df.sort_values("timestamp").reset_index(drop=True)
    pupil = pupil_df.sort_values("timestamp").reset_index(drop=True)

    # Merge on nearest timestamp using merge_asof
    merged = pd.merge_asof(
        gaze,
        pupil[["timestamp", "pupil_diameter", "pupil_diameter_3d", "pupil_confidence"]],
        on="timestamp",
        direction="nearest",
        tolerance=0.01,  # 10ms tolerance
    )

    return merged


def find_exports(
    export_dir: Path,
    surface_name: Optional[str] = None,
) -> dict[str, Optional[Path]]:
    """Locate Pupil Labs export files in a directory.

    Args:
        export_dir: Directory containing Pupil Labs exports.
        surface_name: Specific surface name to look for.

    Returns:
        Dictionary mapping data type to file path.
    """
    files: dict[str, Optional[Path]] = {
        "gaze": None,
        "pupil": None,
        "fixations": None,
        "blinks": None,
        "surface": None,
    }

    for f in export_dir.iterdir():
        name_lower = f.name.lower()
        if name_lower == "gaze_positions.csv":
            files["gaze"] = f
        elif name_lower == "pupil_positions.csv":
            files["pupil"] = f
        elif name_lower == "fixations.csv":
            files["fixations"] = f
        elif name_lower == "blinks.csv":
            files["blinks"] = f
        elif surface_name and surface_name.lower() in name_lower and "surface" in name_lower:
            files["surface"] = f
        elif name_lower.startswith("surface_gaze") or "surface" in name_lower:
            if files["surface"] is None:
                files["surface"] = f

    return files


def parse_pupil_exports(
    export_dir: Path,
    surface: bool = False,
    surface_name: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """Parse all Pupil Labs exports from a directory.

    Args:
        export_dir: Directory containing export files.
        surface: Whether to include surface tracking data.
        surface_name: Name of the surface to load.

    Returns:
        Tuple of (unified DataFrame, metadata dict).
    """
    if not export_dir.is_dir():
        raise FileNotFoundError(f"Pupil Labs export directory not found: {export_dir}")

    file_map = find_exports(export_dir, surface_name)

    # Import available data
    gaze_df = import_gaze_csv(file_map["gaze"]) if file_map["gaze"] else pd.DataFrame()
    pupil_df = import_pupil_csv(file_map["pupil"]) if file_map["pupil"] else pd.DataFrame()
    fix_df = import_fixations_csv(file_map["fixations"]) if file_map["fixations"] else pd.DataFrame()
    blink_df = import_blinks_csv(file_map["blinks"]) if file_map["blinks"] else pd.DataFrame()
    surf_df = (
        import_surface_csv(file_map["surface"])
        if surface and file_map["surface"]
        else pd.DataFrame()
    )

    # Merge gaze + pupil
    df = merge_gaze_pupil(gaze_df, pupil_df)

    # Add fixation events
    if not fix_df.empty:
        df["fixation_index"] = np.nan
        for _, row in fix_df.iterrows():
            start = row.get("fixation_start", np.nan)
            end = row.get("fixation_end", np.nan)
            if pd.notna(start) and pd.notna(end):
                mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
                df.loc[mask, "fixation_index"] = row.get("fixation_index", 0)

    # Mark blinks
    if not blink_df.empty:
        df["event"] = ""
        for _, row in blink_df.iterrows():
            start = row.get("blink_start", np.nan)
            end = row.get("blink_end", np.nan)
            if pd.notna(start) and pd.notna(end):
                mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
                df.loc[mask, "event"] = "Blink"

    # Merge surface data if available
    if not surf_df.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            surf_df[["timestamp", "surface_x", "surface_y", "on_surface"]],
            on="timestamp",
            direction="nearest",
            tolerance=0.01,
        )

    # Metadata
    meta: dict = {
        "source_format": "Pupil Labs",
        "export_dir": str(export_dir),
        "total_samples": len(df),
        "files_found": {k: str(v) for k, v in file_map.items() if v},
    }

    if "timestamp" in df.columns and len(df) > 1:
        duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        meta["duration_s"] = duration
        ts = df["timestamp"].dropna()
        if len(ts) > 1:
            intervals = np.diff(ts.values)
            meta["estimated_sampling_rate_hz"] = (
                1.0 / float(np.median(intervals)) if np.median(intervals) > 0 else 0
            )

    return df, meta


def print_summary(df: pd.DataFrame, meta: dict) -> None:
    """Print metadata summary."""
    print("=" * 60)
    print("  Pupil Labs Eye-Tracking Data Summary")
    print("=" * 60)
    print(f"  Export directory:    {meta.get('export_dir', 'N/A')}")
    print(f"  Total samples:       {meta.get('total_samples', 'N/A')}")
    print(f"  Duration:            {meta.get('duration_s', 'N/A'):.2f} s")
    sr = meta.get("estimated_sampling_rate_hz")
    if sr:
        print(f"  Est. sampling rate: {sr:.1f} Hz")
    print("  Files found:")
    for k, v in meta.get("files_found", {}).items():
        print(f"    - {k}: {v}")
    print(f"\n  Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    - {col}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Pupil Labs eye-tracking exports."
    )
    parser.add_argument("input", type=Path, help="Pupil Labs export directory.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (.parquet or .csv).",
    )
    parser.add_argument(
        "--surface", action="store_true",
        help="Include surface tracking data.",
    )
    parser.add_argument(
        "--surface-name", type=str, default=None,
        help="Name of the surface to load.",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print metadata summary."
    )
    args = parser.parse_args()

    df, meta = parse_pupil_exports(
        args.input, surface=args.surface, surface_name=args.surface_name
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
