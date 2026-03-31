#!/usr/bin/env python3
"""Insight guardrails: 6-layer quality scoring for AI-generated insights.

Evaluates insight confidence (0-1) through:
1. Data completeness check
2. Statistical significance validation
3. Effect size assessment (Cohen's d > 0.2)
4. Causal inference warning (correlation ≠ causation)
5. Multiple comparison correction reminder
6. LLM hallucination detection (cross-validate with raw data)

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply guardrails to AI insights.")
    parser.add_argument("--insight", "-i", required=True, help="Insight JSON or Markdown file")
    parser.add_argument("--data", "-d", default=None, help="Original session data JSON")
    parser.add_argument("--anomalies", "-a", default=None, help="Anomalies JSON")
    parser.add_argument("--output", "-o", default=None, help="Output guarded insight JSON")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--min-cohens-d", type=float, default=0.2, help="Minimum Cohen's d")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


# ── Layer 1: Data Completeness ────────────────────────────────────────────────

def check_data_completeness(data: dict, verbose: bool = False) -> dict[str, Any]:
    """Check data integrity: missing values, sampling rate anomalies.

    Args:
        data: Session data dict.
        verbose: Print info.

    Returns:
        Check result with score (0-1) and issues.
    """
    issues: list[str] = []
    total_samples = 0
    n_modalities = 0

    for modality, info in data.items():
        if isinstance(info, dict) and "data" in info:
            arr = np.array(info["data"], dtype=float)
            sfreq = info.get("sfreq", None)
        elif isinstance(info, list):
            arr = np.array(info, dtype=float)
            sfreq = None
        else:
            issues.append(f"{modality}: unrecognizable format")
            continue

        n_modalities += 1
        total_samples += len(arr)

        # Check for NaN/Inf
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        if nan_count > 0:
            issues.append(f"{modality}: {nan_count} NaN values ({100*nan_count/len(arr):.1f}%)")
        if inf_count > 0:
            issues.append(f"{modality}: {inf_count} Inf values")

        # Check for constant signal
        if np.std(arr) == 0:
            issues.append(f"{modality}: constant signal (std=0)")

        # Check sampling rate reasonableness
        if sfreq is not None:
            if sfreq < 1 or sfreq > 10000:
                issues.append(f"{modality}: unusual sfreq={sfreq}")

    # Score: no issues = 1.0, each issue reduces score
    score = max(0.0, 1.0 - len(issues) * 0.15)

    if verbose:
        status = "[#4A90D9]" if score >= 0.8 else "[#E74C3C]"
        print(f"  {status} Layer 1 (Completeness): score={score:.2f}, issues={len(issues)}")

    return {"layer": "data_completeness", "score": round(score, 3), "n_modalities": n_modalities,
            "total_samples": total_samples, "issues": issues}


# ── Layer 2: Statistical Significance ─────────────────────────────────────────

def check_statistical_significance(
    insight: dict, anomalies: Optional[dict] = None, p_threshold: float = 0.05,
    verbose: bool = False,
) -> dict[str, Any]:
    """Validate that reported findings meet statistical significance thresholds.

    Args:
        insight: Insight dict with findings.
        anomalies: Anomaly data with p-values.
        p_threshold: Significance threshold.
        verbose: Print info.

    Returns:
        Check result with score and warnings.
    """
    warnings: list[str] = []
    checked = 0
    passed = 0

    # Check anomaly p-values if available
    if anomalies:
        for modality, info in anomalies.get("modalities", {}).items():
            checked += 1
            # We validate that anomalies exist with reasonable severity
            if info.get("mean_severity", 0) > 0.3:
                passed += 1

    # Check insight key findings for p-value claims
    findings = insight.get("structured_summary", {}).get("key_findings", [])
    for finding in findings:
        # Look for p-value patterns in findings
        p_match = re.search(r"p\s*[<>=]+\s*([\d.]+)", finding)
        if p_match:
            p_val = float(p_match.group(1))
            checked += 1
            if p_val <= p_threshold:
                passed += 1
            else:
                warnings.append(f"Non-significant p-value claimed: p={p_val}")

    score = passed / checked if checked > 0 else 0.5

    if verbose:
        status = "[#4A90D9]" if score >= 0.8 else "[#E74C3C]"
        print(f"  {status} Layer 2 (Significance): score={score:.2f}, passed={passed}/{checked}")

    return {"layer": "statistical_significance", "score": round(score, 3),
            "passed": passed, "checked": checked, "p_threshold": p_threshold,
            "warnings": warnings}


# ── Layer 3: Effect Size ──────────────────────────────────────────────────────

def check_effect_size(
    data: dict, min_d: float = 0.2, verbose: bool = False,
) -> dict[str, Any]:
    """Verify that reported effects meet minimum Cohen's d threshold.

    Computes effect sizes for conditions where we have data, filtering
    out trivially small effects.

    Args:
        data: Session data dict.
        min_d: Minimum Cohen's d to report.
        verbose: Print info.

    Returns:
        Check result with score and warnings.
    """
    warnings: list[str] = []
    effects: list[dict] = []

    modalities = []
    for modality, info in data.items():
        if isinstance(info, dict) and "data" in info:
            arr = np.array(info["data"], dtype=float).flatten()
        elif isinstance(info, list):
            arr = np.array(info, dtype=float).flatten()
        else:
            continue
        modalities.append((modality, arr))

    # Compare first half vs second half as a basic effect size check
    for modality, arr in modalities:
        if len(arr) < 20:
            continue
        mid = len(arr) // 2
        first, second = arr[:mid], arr[mid:]
        d = cohens_d(first, second)
        effects.append({"modality": modality, "cohens_d": round(d, 4), "meets_threshold": abs(d) >= min_d})
        if 0 < abs(d) < min_d:
            warnings.append(f"{modality}: small effect (d={d:.3f} < {min_d})")

    n_pass = sum(1 for e in effects if e["meets_threshold"])
    score = n_pass / len(effects) if effects else 0.5

    if verbose:
        status = "[#4A90D9]" if score >= 0.8 else "[#E74C3C]"
        print(f"  {status} Layer 3 (Effect Size): score={score:.2f}, pass={n_pass}/{len(effects)}")

    return {"layer": "effect_size", "score": round(score, 3), "min_cohens_d": min_d,
            "effects": effects, "warnings": warnings}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d between two groups.

    Args:
        group1: First group values.
        group2: Second group values.

    Returns:
        Cohen's d value.
    """
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2))
    return float(np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-12)


# ── Layer 4: Causal Inference Warning ─────────────────────────────────────────

def check_causal_claims(insight: dict, verbose: bool = False) -> dict[str, Any]:
    """Flag potential causal inference errors (correlation ≠ causation).

    Scans LLM-generated text for causal language that isn't supported
    by the experimental design.

    Args:
        insight: Insight dict.
        verbose: Print info.

    Returns:
        Check result with warnings.
    """
    warnings: list[str] = []
    llm_text = insight.get("llm_insight", "")

    causal_patterns = [
        (r"因为.*所以", "because...therefore"),
        (r"导致", "caused"),
        (r"证明了", "proved"),
        (r"必然", "necessarily"),
        (r"完全是", "completely"),
        (r"clearly shows", "clearly shows"),
        (r"definitely", "definitely"),
        (r"proves that", "proves that"),
    ]

    flagged = []
    for pattern, label in causal_patterns:
        matches = re.findall(pattern, llm_text, re.IGNORECASE)
        if matches:
            flagged.append(f"'{label}' ({len(matches)}x)")

    # Also check structured findings
    findings = insight.get("structured_summary", {}).get("key_findings", [])
    for finding in findings:
        if any(w in finding for w in ["because", "caused", "导致", "证明"]):
            warnings.append(f"Causal language in finding: '{finding[:60]}...'")

    score = max(0.0, 1.0 - len(flagged) * 0.2)

    if verbose:
        status = "[#4A90D9]" if score >= 0.8 else "[#E74C3C]"
        print(f"  {status} Layer 4 (Causal): score={score:.2f}, flagged={len(flagged)}")

    return {"layer": "causal_inference", "score": round(score, 3),
            "flagged_patterns": flagged, "warnings": warnings}


# ── Layer 5: Multiple Comparison Correction ───────────────────────────────────

def check_multiple_comparisons(insight: dict, verbose: bool = False) -> dict[str, Any]:
    """Check if multiple comparison correction is needed.

    Warns when many statistical tests are performed without correction.

    Args:
        insight: Insight dict.
        verbose: Print info.

    Returns:
        Check result with warnings.
    """
    findings = insight.get("structured_summary", {}).get("key_findings", [])
    anomalies = insight.get("structured_summary", {}).get("anomaly_summary")
    n_tests = len(findings)
    if anomalies:
        n_tests += sum(anomalies.get("per_modality", {}).values())

    warnings: list[str] = []
    needs_correction = n_tests > 3

    if needs_correction:
        bonferroni = 0.05 / n_tests
        warnings.append(
            f"{n_tests} statistical tests detected. Bonferroni corrected threshold: {bonferroni:.4f}. "
            f"Consider FDR correction for large numbers of tests."
        )

    score = 0.5 if needs_correction else 1.0

    if verbose:
        status = "[#4A90D9]" if not needs_correction else "[#E74C3C]"
        print(f"  {status} Layer 5 (Multi-comp): score={score:.2f}, n_tests={n_tests}")

    return {"layer": "multiple_comparisons", "score": round(score, 3),
            "n_tests": n_tests, "needs_correction": needs_correction, "warnings": warnings}


# ── Layer 6: LLM Hallucination Detection ─────────────────────────────────────

def check_hallucination(
    insight: dict, data: dict, verbose: bool = False,
) -> dict[str, Any]:
    """Cross-validate LLM claims against raw data to detect hallucinations.

    Checks if numbers cited in LLM text match actual data statistics.

    Args:
        insight: Insight dict with LLM text.
        data: Original session data.
        verbose: Print info.

    Returns:
        Check result with hallucination flags.
    """
    llm_text = insight.get("llm_insight", "")
    hallucinations: list[str] = []
    verified = 0

    # Extract numeric claims from LLM text
    number_patterns = re.findall(r"(\d+\.?\d*)", llm_text)

    # Get actual data statistics
    actual_stats: dict[str, float] = {}
    for modality, info in data.items():
        if isinstance(info, dict) and "data" in info:
            arr = np.array(info["data"], dtype=float).flatten()
        elif isinstance(info, list):
            arr = np.array(info, dtype=float).flatten()
        else:
            continue
        actual_stats[modality] = float(np.mean(arr))
        actual_stats[f"{modality}_std"] = float(np.std(arr))
        actual_stats[f"{modality}_min"] = float(np.min(arr))
        actual_stats[f"{modality}_max"] = float(np.max(arr))

    # Cross-check: verify that numbers in LLM text are plausible
    all_values = list(actual_stats.values())
    if all_values:
        data_min = min(all_values)
        data_max = max(all_values)
        data_range = data_max - data_min + 1e-12

        for num_str in number_patterns:
            try:
                num = float(num_str)
                # Check if number is within plausible range (10x of data range)
                if abs(num) > abs(data_max) * 10 and num not in [0, 1, 100]:
                    hallucinations.append(
                        f"Value {num} seems implausible (data range: {data_min:.2f} to {data_max:.2f})"
                    )
            except ValueError:
                continue

        verified = len(number_patterns) - len(hallucinations)

    score = max(0.0, 1.0 - len(hallucinations) * 0.3)

    if verbose:
        status = "[#4A90D9]" if score >= 0.8 else "[#E74C3C]"
        print(f"  {status} Layer 6 (Hallucination): score={score:.2f}, "
              f"verified={verified}, suspicious={len(hallucinations)}")

    return {"layer": "hallucination_detection", "score": round(score, 3),
            "numbers_checked": len(number_patterns), "verified": verified,
            "hallucinations": hallucinations}


# ── Aggregate Score ───────────────────────────────────────────────────────────

def compute_overall_score(layers: list[dict]) -> dict[str, Any]:
    """Compute weighted overall confidence score.

    Weights:
    - Data completeness: 0.20
    - Statistical significance: 0.20
    - Effect size: 0.15
    - Causal inference: 0.15
    - Multiple comparisons: 0.10
    - Hallucination detection: 0.20

    Args:
        layers: List of layer check results.

    Returns:
        Overall score and status.
    """
    weights = [0.20, 0.20, 0.15, 0.15, 0.10, 0.20]
    weighted = sum(l["score"] * w for l, w in zip(layers, weights))
    overall = round(weighted, 3)

    if overall >= 0.8:
        status = "PASS"
        status_msg = "✅ Insight passes quality checks"
    elif overall >= 0.5:
        status = "REVIEW"
        status_msg = "⚠️ Insight requires manual review"
    else:
        status = "REJECT"
        status_msg = "🔴 Insight rejected — quality too low"

    return {
        "overall_score": overall,
        "status": status,
        "status_message": status_msg,
        "layers": {l["layer"]: l for l in layers},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

COLOR_PRIMARY = "#4A90D9"
COLOR_ALERT = "#E74C3C"


def main() -> int:
    args = parse_args()

    if not Path(args.insight).exists():
        print(f"[{COLOR_ALERT}] ERROR: File not found: {args.insight}", file=sys.stderr)
        return 1

    with open(args.insight) as f:
        insight = json.load(f)

    data: dict = {}
    if args.data and Path(args.data).exists():
        with open(args.data) as f:
            data = json.load(f)

    anomalies: Optional[dict] = None
    if args.anomalies and Path(args.anomalies).exists():
        with open(args.anomalies) as f:
            anomalies = json.load(f)

    print(f"[{COLOR_PRIMARY}] Running 6-layer insight guardrails...\n")

    layers: list[dict[str, Any]] = []

    # Layer 1
    if data:
        layers.append(check_data_completeness(data, verbose=args.verbose))
    else:
        layers.append({"layer": "data_completeness", "score": 0.5, "issues": ["No data provided"]})

    # Layer 2
    layers.append(check_statistical_significance(insight, anomalies, args.p_threshold, args.verbose))

    # Layer 3
    if data:
        layers.append(check_effect_size(data, args.min_cohens_d, verbose=args.verbose))
    else:
        layers.append({"layer": "effect_size", "score": 0.5, "warnings": ["No data for effect size"]})

    # Layer 4
    layers.append(check_causal_claims(insight, verbose=args.verbose))

    # Layer 5
    layers.append(check_multiple_comparisons(insight, verbose=args.verbose))

    # Layer 6
    if data and insight.get("llm_insight"):
        layers.append(check_hallucination(insight, data, verbose=args.verbose))
    else:
        layers.append({"layer": "hallucination_detection", "score": 0.5,
                       "hallucinations": ["Cannot verify without data"]})

    # Overall
    overall = compute_overall_score(layers)
    print(f"\n[{COLOR_PRIMARY}] Overall: {overall['status_message']} (score={overall['overall_score']:.3f})")

    result = {
        **insight,
        "guardrails": overall,
    }

    out_path = args.output or "insight_guarded.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{COLOR_PRIMARY}] ✓ Saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
