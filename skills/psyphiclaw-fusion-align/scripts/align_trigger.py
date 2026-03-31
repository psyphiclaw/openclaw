#!/usr/bin/env python3
"""Align multimodal data via shared TTL trigger signals.

Detects common trigger codes across EEG event markers and other modalities'
timestamp lists, computes per-modality time offsets, and writes alignment
parameters to JSON.

Usage:
    python align_trigger.py --eeg-events events.csv \
        --modality-timestamps face_ts.csv \
        --modality-timestamps physio_ts.csv \
        --output alignment.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Colour palette (PsyPhiClaw brand) ──────────────────────────────────────
BLUE = "#4A90D9"
RED = "#E74C3C"


def load_eeg_events(path: str) -> pd.DataFrame:
    """Load EEG trigger events from CSV.

    Expected columns: *sample*, *trigger* (or *code*).
    Rows describe TTL pulse onsets in samples from recording start.
    """
    df = pd.read_csv(path)
    # Flexible column name handling
    for col in ("sample", "Sample", "SAMPLE"):
        if col in df.columns:
            df.rename(columns={col: "sample"}, inplace=True)
    for col in ("trigger", "code", "Trigger", "Code", "TRIGGER"):
        if col in df.columns:
            df.rename(columns={col: "trigger"}, inplace=True)
    if "sample" not in df.columns or "trigger" not in df.columns:
        raise ValueError(
            f"EEG events CSV must contain 'sample' and 'trigger' columns. "
            f"Got: {list(df.columns)}"
        )
    return df


def load_modality_timestamps(path: str) -> pd.DataFrame:
    """Load per-modality timestamp list (one trigger per row).

    Expected columns: *timestamp* (ms) and optionally *trigger*.
    """
    df = pd.read_csv(path)
    for col in ("timestamp", "Timestamp", "TIMESTAMP", "time_ms"):
        if col in df.columns:
            df.rename(columns={col: "timestamp"}, inplace=True)
            break
    for col in ("trigger", "code", "Trigger", "Code"):
        if col in df.columns:
            df.rename(columns={col: "trigger"}, inplace=True)
            break
    if "timestamp" not in df.columns:
        raise ValueError(
            f"Modality timestamps CSV must contain a 'timestamp' column. "
            f"Got: {list(df.columns)}"
        )
    return df


def match_triggers(
    eeg_events: pd.DataFrame,
    modality_df: pd.DataFrame,
    eeg_sfreq: float,
    modality_name: str,
) -> dict[str, Any]:
    """Find shared trigger codes and compute time offset.

    Parameters
    ----------
    eeg_events : DataFrame with columns ``sample`` and ``trigger``.
    modality_df : DataFrame with columns ``timestamp`` (ms) and optionally ``trigger``.
    eeg_sfreq : EEG sampling rate in Hz.
    modality_name : Human-readable name for error messages.

    Returns
    -------
    dict with keys ``trigger_code``, ``eeg_time_ms``, ``modality_time_ms``, ``offset_ms``.
    """
    # If modality has trigger codes, match directly
    if "trigger" in modality_df.columns:
        merged = eeg_events.merge(
            modality_df, on="trigger", how="inner", suffixes=("_eeg", "_mod")
        )
        if merged.empty:
            raise ValueError(
                f"No shared trigger codes between EEG and {modality_name}."
            )
        # Use median offset across all matched triggers
        eeg_times = merged["sample"].values / eeg_sfreq * 1000.0
        mod_times = merged["timestamp"].values
        offsets = mod_times - eeg_times
        median_offset = float(np.median(offsets))
        return {
            "modality": modality_name,
            "matched_count": len(merged),
            "trigger_codes": merged["trigger"].tolist(),
            "offsets_ms": offsets.tolist(),
            "median_offset_ms": round(median_offset, 4),
            "mad_ms": round(float(np.median(np.abs(offsets - median_offset))), 4),
        }

    # No trigger codes in modality → match by count order (assumes same sequence)
    n_min = min(len(eeg_events), len(modality_df))
    if n_min == 0:
        raise ValueError(
            f"Empty trigger list for {modality_name} or EEG events."
        )
    eeg_times = eeg_events["sample"].values[:n_min] / eeg_sfreq * 1000.0
    mod_times = modality_df["timestamp"].values[:n_min]
    offsets = mod_times - eeg_times
    median_offset = float(np.median(offsets))
    return {
        "modality": modality_name,
        "matched_count": n_min,
        "match_method": "sequential_order",
        "offsets_ms": offsets.tolist(),
        "median_offset_ms": round(median_offset, 4),
        "mad_ms": round(float(np.median(np.abs(offsets - median_offset))), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align multimodal data via shared TTL trigger signals."
    )
    parser.add_argument(
        "--eeg-events",
        required=True,
        help="EEG event CSV with 'sample' and 'trigger' columns.",
    )
    parser.add_argument(
        "--eeg-sfreq",
        type=float,
        default=500.0,
        help="EEG sampling rate in Hz (default: 500).",
    )
    parser.add_argument(
        "--modality-timestamps",
        action="append",
        required=True,
        help=(
            "Modality timestamp CSV (repeat for each modality). "
            "Use modality_name:path syntax, e.g. face:face_ts.csv. "
            "If no name prefix, filename stem is used."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for alignment parameters.",
    )
    args = parser.parse_args()

    eeg_events = load_eeg_events(args.eeg_events)
    results: list[dict[str, Any]] = []

    for item in args.modality_timestamps:
        if ":" in item:
            name, path = item.split(":", 1)
        else:
            name = Path(item).stem
            path = item
        mod_df = load_modality_timestamps(path)
        match = match_triggers(eeg_events, mod_df, args.eeg_sfreq, name)
        results.append(match)

    output = {
        "alignment_method": "ttl_trigger",
        "eeg_sfreq_hz": args.eeg_sfreq,
        "eeg_event_count": len(eeg_events),
        "modalities": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Alignment parameters written to {out_path}")
    for r in results:
        print(
            f"  [{r['modality']}] matched={r['matched_count']} triggers, "
            f"offset={r['median_offset_ms']:.2f} ms (MAD={r['mad_ms']:.2f})"
        )


if __name__ == "__main__":
    main()
