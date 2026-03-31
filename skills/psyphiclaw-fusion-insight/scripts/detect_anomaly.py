#!/usr/bin/env python3
"""Detect anomalies in multimodal data using statistical methods.

Supports Z-score, IQR, Modified Z-score, sliding-window trend detection,
and cross-modal synchronized anomaly detection.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect anomalies in multimodal data.")
    parser.add_argument("--data", "-d", required=True, help="Session data JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output anomalies JSON")
    parser.add_argument("--method", choices=["zscore", "iqr", "modified_z", "all"], default="all")
    parser.add_argument("--z-threshold", type=float, default=2.5, help="Z-score threshold")
    parser.add_argument("--iqr-multiplier", type=float, default=1.5, help="IQR multiplier")
    parser.add_argument("--window-size", type=int, default=30, help="Sliding window size (samples)")
    parser.add_argument("--cross-modal-threshold", type=int, default=2, help="Min modalities for sync anomaly")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


# ── Z-Score Anomaly Detection ────────────────────────────────────────────────

def detect_zscore(
    signal: np.ndarray,
    threshold: float = 2.5,
    label: str = "unknown",
) -> list[dict[str, Any]]:
    """Detect anomalies using Z-score method.

    Args:
        signal: 1D numpy array.
        threshold: Z-score threshold for anomaly.
        label: Channel/modality label.

    Returns:
        List of anomaly dicts with index, value, z_score, severity.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return []

    z_scores = np.abs((signal - mean) / std)
    anomaly_idx = np.where(z_scores > threshold)[0]

    anomalies = []
    for idx in anomaly_idx:
        severity = min(z_scores[idx] / 5.0, 1.0)  # Normalize to 0-1
        anomalies.append({
            "modality": label,
            "index": int(idx),
            "value": round(float(signal[idx]), 6),
            "z_score": round(float(z_scores[idx]), 4),
            "severity": round(float(severity), 3),
            "method": "zscore",
        })

    return anomalies


# ── IQR Anomaly Detection ────────────────────────────────────────────────────

def detect_iqr(
    signal: np.ndarray,
    multiplier: float = 1.5,
    label: str = "unknown",
) -> list[dict[str, Any]]:
    """Detect anomalies using Interquartile Range method.

    Args:
        signal: 1D numpy array.
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme outliers).
        label: Channel/modality label.

    Returns:
        List of anomaly dicts.
    """
    q1 = np.percentile(signal, 25)
    q3 = np.percentile(signal, 75)
    iqr = q3 - q1
    if iqr == 0:
        return []

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    anomaly_idx = np.where((signal < lower) | (signal > upper))[0]

    anomalies = []
    for idx in anomaly_idx:
        deviation = max(abs(signal[idx] - q1), abs(signal[idx] - q3)) / iqr
        severity = min(deviation / 3.0, 1.0)
        anomalies.append({
            "modality": label,
            "index": int(idx),
            "value": round(float(signal[idx]), 6),
            "iqr_deviation": round(float(deviation), 4),
            "severity": round(float(severity), 3),
            "method": "iqr",
        })

    return anomalies


# ── Modified Z-Score ──────────────────────────────────────────────────────────

def detect_modified_zscore(
    signal: np.ndarray,
    threshold: float = 3.5,
    label: str = "unknown",
) -> list[dict[str, Any]]:
    """Detect anomalies using Modified Z-score (MAD-based).

    More robust than standard Z-score for non-normal distributions.

    Args:
        signal: 1D numpy array.
        threshold: Modified Z-score threshold.
        label: Channel/modality label.

    Returns:
        List of anomaly dicts.
    """
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    if mad == 0:
        return []

    modified_z = 0.6745 * (signal - median) / mad
    anomaly_idx = np.where(np.abs(modified_z) > threshold)[0]

    anomalies = []
    for idx in anomaly_idx:
        severity = min(np.abs(modified_z[idx]) / 7.0, 1.0)
        anomalies.append({
            "modality": label,
            "index": int(idx),
            "value": round(float(signal[idx]), 6),
            "modified_z": round(float(modified_z[idx]), 4),
            "severity": round(float(severity), 3),
            "method": "modified_z",
        })

    return anomalies


# ── Sliding Window Trend Detection ───────────────────────────────────────────

def detect_trend_breaks(
    signal: np.ndarray,
    window_size: int = 30,
    label: str = "unknown",
) -> list[dict[str, Any]]:
    """Detect sudden trend changes using sliding window comparison.

    Compares mean and variance between consecutive windows to detect
    abrupt shifts in signal characteristics.

    Args:
        signal: 1D numpy array.
        window_size: Number of samples per window.
        label: Channel/modality label.

    Returns:
        List of trend break dicts.
    """
    if len(signal) < window_size * 2:
        return []

    breaks: list[dict[str, Any]] = []
    half = window_size // 2

    for i in range(half, len(signal) - half):
        w1 = signal[i - half:i]
        w2 = signal[i:i + half]

        mean_diff = abs(np.mean(w1) - np.mean(w2))
        std_combined = (np.std(w1) + np.std(w2)) / 2 + 1e-12
        effect = mean_diff / std_combined

        if effect > 3.0:  # Significant trend break
            severity = min(effect / 6.0, 1.0)
            breaks.append({
                "modality": label,
                "index": int(i),
                "mean_before": round(float(np.mean(w1)), 6),
                "mean_after": round(float(np.mean(w2)), 6),
                "effect_size": round(float(effect), 4),
                "severity": round(float(severity), 3),
                "method": "trend_break",
            })

    # Deduplicate: keep highest severity within 10-sample radius
    if breaks:
        breaks.sort(key=lambda x: x["severity"], reverse=True)
        seen: set[int] = set()
        deduped = []
        for b in breaks:
            idx = b["index"]
            if not any(abs(idx - s) < 10 for s in seen):
                deduped.append(b)
                seen.add(idx)
        breaks = deduped

    return breaks


# ── Cross-Modal Synchronized Detection ───────────────────────────────────────

def detect_cross_modal_sync(
    all_anomalies: dict[str, list[dict[str, Any]]],
    time_tolerance: int = 10,
    min_modalities: int = 2,
) -> list[dict[str, Any]]:
    """Detect time points where multiple modalities show anomalies simultaneously.

    Args:
        all_anomalies: Dict mapping modality names to anomaly lists.
        time_tolerance: Max index distance to consider "simultaneous".
        min_modalities: Minimum modalities for cross-modal event.

    Returns:
        List of cross-modal synchronized anomaly events.
    """
    if len(all_anomalies) < min_modalities:
        return []

    # Collect all anomaly indices per modality
    index_sets: dict[str, set[int]] = {}
    for modality, anomalies in all_anomalies.items():
        index_sets[modality] = {a["index"] for a in anomalies}

    modalities = list(index_sets.keys())
    # Create a unified timeline
    all_indices: set[int] = set()
    for idx_set in index_sets.values():
        all_indices.update(idx_set)

    sync_events: list[dict[str, Any]] = []

    for idx in sorted(all_indices):
        involved: list[str] = []
        max_severity = 0.0
        details: list[dict] = []

        for modality in modalities:
            nearby = [i for i in index_sets[modality] if abs(i - idx) <= time_tolerance]
            if nearby:
                involved.append(modality)
                mod_anomalies = all_anomalies[modality]
                nearby_anom = [a for a in mod_anomalies if abs(a["index"] - idx) <= time_tolerance]
                for a in nearby_anom:
                    max_severity = max(max_severity, a["severity"])
                    details.append(a)

        if len(involved) >= min_modalities:
            sync_events.append({
                "index": int(idx),
                "modalities_involved": involved,
                "n_modalities": len(involved),
                "max_severity": round(max_severity, 3),
                "details": details,
                "method": "cross_modal_sync",
            })

    return sync_events


# ── Load Session Data ─────────────────────────────────────────────────────────

def load_session_data(filepath: str) -> dict[str, np.ndarray]:
    """Load multimodal session data from JSON.

    Expected format:
    {
      "eeg": {"data": [[...], ...], "channels": [...], "sfreq": 500},
      "eda": {"data": [...], "sfreq": 4},
      "pupil": {"data": [...], "sfreq": 120},
      ...
    }

    Args:
        filepath: Path to JSON file.

    Returns:
        Dict mapping modality names to 1D numpy arrays (averaged across channels).
    """
    with open(filepath) as f:
        data = json.load(f)

    signals: dict[str, np.ndarray] = {}
    for modality, info in data.items():
        if isinstance(info, dict) and "data" in info:
            arr = np.array(info["data"], dtype=float)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=0)
            signals[modality] = arr
        elif isinstance(info, list):
            signals[modality] = np.array(info, dtype=float).flatten()

    return signals


# ── Main ──────────────────────────────────────────────────────────────────────

COLOR_PRIMARY = "#4A90D9"
COLOR_ALERT = "#E74C3C"


def main() -> int:
    args = parse_args()

    if not Path(args.data).exists():
        print(f"[{COLOR_ALERT}] ERROR: File not found: {args.data}", file=sys.stderr)
        return 1

    signals = load_session_data(args.data)
    print(f"[{COLOR_PRIMARY}] Loaded {len(signals)} modalities: {list(signals.keys())}")

    all_anomalies: dict[str, list[dict[str, Any]]] = {}
    all_results: dict[str, Any] = {"modalities": {}}

    for modality, signal in signals.items():
        mod_anomalies: list[dict[str, Any]] = []

        if args.method in ("zscore", "all"):
            mod_anomalies.extend(detect_zscore(signal, args.z_threshold, modality))
        if args.method in ("iqr", "all"):
            mod_anomalies.extend(detect_iqr(signal, args.iqr_multiplier, modality))
        if args.method in ("modified_z", "all"):
            mod_anomalies.extend(detect_modified_zscore(signal, label=modality))

        # Trend breaks always included
        mod_anomalies.extend(detect_trend_breaks(signal, args.window_size, modality))

        all_anomalies[modality] = mod_anomalies
        all_results["modalities"][modality] = {
            "n_samples": len(signal),
            "n_anomalies": len(mod_anomalies),
            "mean_severity": round(float(np.mean([a["severity"] for a in mod_anomalies])), 3) if mod_anomalies else 0,
        }

        if args.verbose:
            print(f"  {modality}: {len(mod_anomalies)} anomalies detected")

    # Cross-modal sync
    sync = detect_cross_modal_sync(
        all_anomalies,
        time_tolerance=10,
        min_modalities=args.cross_modal_threshold,
    )
    all_results["cross_modal_sync"] = {
        "n_events": len(sync),
        "events": sync,
    }

    all_results["total_anomalies"] = sum(len(v) for v in all_anomalies.values())

    print(f"[{COLOR_PRIMARY}] Total anomalies: {all_results['total_anomalies']}")
    print(f"[{COLOR_PRIMARY}] Cross-modal sync events: {len(sync)}")

    out_path = args.output or "anomalies.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[{COLOR_PRIMARY}] ✓ Saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
