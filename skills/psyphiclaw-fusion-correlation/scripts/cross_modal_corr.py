#!/usr/bin/env python3
"""Cross-modal correlation analysis.

Computes static and dynamic (sliding-window) correlations between features
from different modalities, with permutation-based significance testing.

Usage:
    python cross_modal_corr.py --session session.h5 \
        --modalities eeg_alpha face_valence physio_eda \
        --method spearman --window-size 5.0 --step-size 1.0 \
        --n-permutations 1000 --output results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "psyphiclaw-fusion-align" / "scripts"))
    from session_manager import MultiModalSession
except ImportError:
    print("⚠️  session_manager not found, using CSV fallback mode.", file=sys.stderr)

BLUE = "#4A90D9"
RED = "#E74C3C"


def compute_static_corr(
    features: dict[str, np.ndarray],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute pairwise correlation matrix across all features.

    Parameters
    ----------
    features : dict mapping feature_name → 1-D array.
    method : ``pearson`` or ``spearman``.

    Returns
    -------
    DataFrame correlation matrix (features × features).
    """
    df = pd.DataFrame(features)
    return df.corr(method=method)


def compute_sliding_corr(
    features: dict[str, np.ndarray],
    timestamps_ms: np.ndarray,
    window_sec: float,
    step_sec: float,
    method: str = "pearson",
) -> tuple[pd.DataFrame, list[str]]:
    """Sliding-window correlation between feature pairs.

    Returns
    -------
    DataFrame with index = time_centre, columns = ``pair1|pair2``.
    """
    window_ms = window_sec * 1000.0
    step_ms = step_sec * 1000.0
    t_min, t_max = timestamps_ms.min(), timestamps_ms.max()

    names = list(features.keys())
    results: list[dict[str, Any]] = []
    centres: list[float] = []

    t = t_min + window_ms / 2
    while t + window_ms / 2 <= t_max:
        mask = (timestamps_ms >= t - window_ms / 2) & (timestamps_ms <= t + window_ms / 2)
        if mask.sum() < 10:
            t += step_ms
            continue
        row: dict[str, Any] = {"time_ms": round(t, 2)}
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                if j <= i:
                    continue
                x, y = features[n1][mask], features[n2][mask]
                if len(x) < 3:
                    row[f"{n1}|{n2}"] = np.nan
                    continue
                if method == "spearman":
                    from scipy.stats import spearmanr
                    r, _ = spearmanr(x, y)
                else:
                    r = np.corrcoef(x, y)[0, 1]
                row[f"{n1}|{n2}"] = round(float(r), 6)
        results.append(row)
        centres.append(t)
        t += step_ms

    df = pd.DataFrame(results)
    if not df.empty:
        df.set_index("time_ms", inplace=True)
    pairs = [f"{names[i]}|{names[j]}" for i in range(len(names)) for j in range(i + 1, len(names))]
    return df, pairs


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 1000,
    method: str = "pearson",
) -> dict[str, float]:
    """Permutation test for correlation significance.

    Returns
    -------
    dict with ``observed_r``, ``p_value``, ``ci_lower``, ``ci_upper``.
    """
    from scipy.stats import spearmanr, pearsonr

    if method == "spearman":
        obs_r, _ = spearmanr(x, y)
    else:
        obs_r, _ = pearsonr(x, y)

    perm_rs: list[float] = []
    rng = np.random.default_rng(42)
    y_perm = y.copy()
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        if method == "spearman":
            r, _ = spearmanr(x, y_perm)
        else:
            r, _ = pearsonr(x, y_perm)
        perm_rs.append(r)

    perm_rs = np.array(perm_rs)
    p_value = float(np.mean(np.abs(perm_rs) >= np.abs(obs_r)))
    return {
        "observed_r": round(float(obs_r), 6),
        "p_value": round(p_value, 6),
        "ci_lower": round(float(np.percentile(perm_rs, 2.5)), 6),
        "ci_upper": round(float(np.percentile(perm_rs, 97.5)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-modal correlation analysis.")
    parser.add_argument("--session", help="Path to MultiModalSession .h5 file.")
    parser.add_argument("--csv-dir", help="Directory of aligned CSV files (alternative to --session).")
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="Modality/feature names to correlate.")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--window-size", type=float, default=5.0,
                        help="Sliding window size in seconds (0 = static only).")
    parser.add_argument("--step-size", type=float, default=1.0,
                        help="Sliding step in seconds.")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Number of permutations for significance test.")
    parser.add_argument("--output", required=True, help="Output directory.")
    args = parser.parse_args()

    # Load data
    all_features: dict[str, np.ndarray] = {}
    timestamps_ms: Optional[np.ndarray] = None

    if args.session:
        session = MultiModalSession.load(args.session)
        for mod_name in args.modalities or list(session.modalities.keys()):
            mod = session.get_modality(mod_name)
            if mod is None:
                continue
            df = mod.to_dataframe()
            ts_col = None
            for c in df.columns:
                if "timestamp" in c.lower():
                    ts_col = c
                    break
            if ts_col:
                timestamps_ms = df[ts_col].values
            value_cols = [c for c in df.columns if c != ts_col]
            for col in value_cols:
                all_features[f"{mod_name}_{col}"] = df[col].values
    elif args.csv_dir:
        csv_dir = Path(args.csv_dir)
        for csv_file in sorted(csv_dir.glob("*.csv")):
            df = pd.read_csv(csv_file)
            ts_col = None
            for c in df.columns:
                if "timestamp" in c.lower():
                    ts_col = c
                    break
            if ts_col:
                timestamps_ms = df[ts_col].values
            value_cols = [c for c in df.columns if c != ts_col]
            for col in value_cols:
                all_features[f"{csv_file.stem}_{col}"] = df[col].values
    else:
        parser.error("Provide --session or --csv-dir.")

    if not all_features:
        print("❌ No features loaded.")
        sys.exit(1)

    if args.modalities:
        selected = {k: v for k, v in all_features.items()
                     if any(m in k for m in args.modalities)}
        if not selected:
            selected = all_features
        all_features = selected

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Static correlation
    corr_matrix = compute_static_corr(all_features, args.method)
    corr_path = out_dir / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_path)
    print(f"✅ Static correlation matrix → {corr_path}")

    # Pairwise permutation tests
    perm_results: list[dict[str, Any]] = []
    names = list(all_features.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x = np.array(all_features[names[i]], dtype=float)
            y = np.array(all_features[names[j]], dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 10:
                continue
            result = permutation_test(x[mask], y[mask], args.n_permutations, args.method)
            result["feature_1"] = names[i]
            result["feature_2"] = names[j]
            perm_results.append(result)
    perm_df = pd.DataFrame(perm_results)
    if not perm_df.empty:
        perm_df.to_csv(out_dir / "permutation_tests.csv", index=False)
        print(f"✅ Permutation tests ({len(perm_results)} pairs) → {out_dir / 'permutation_tests.csv'}")

    # Sliding window correlation
    if args.window_size > 0 and timestamps_ms is not None:
        sliding_df, pairs = compute_sliding_corr(
            all_features, timestamps_ms, args.window_size, args.step_size, args.method
        )
        if not sliding_df.empty:
            sliding_path = out_dir / "sliding_correlation.csv"
            sliding_df.to_csv(sliding_path)
            print(f"✅ Sliding correlation ({len(sliding_df)} windows) → {sliding_path}")

    # Generate HTML report
    _generate_html_report(corr_matrix, perm_df, out_dir / "report.html")


def _generate_html_report(
    corr_matrix: pd.DataFrame,
    perm_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a minimal HTML correlation report."""
    sig_pairs = perm_df[perm_df["p_value"] < 0.05] if not perm_df.empty else pd.DataFrame()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: sans-serif; margin: 2em; background: #fafafa; }}
  h1 {{ color: {BLUE}; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ padding: 4px 8px; border: 1px solid #ddd; font-size: 0.9em; }}
  .sig {{ color: {RED}; font-weight: bold; }}
  .note {{ color: #666; font-size: 0.85em; }}
</style></head><body>
<h1>📊 Cross-Modal Correlation Report</h1>

<h2>Correlation Matrix</h2>
{corr_matrix.to_html(float_format=lambda x: f"{x:.3f}")}

<h2>Permutation Tests (p &lt; 0.05)</h2>
"""
    if sig_pairs.empty:
        html += "<p class='note'>No significant correlations found.</p>"
    else:
        html += sig_pairs.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    html += "</body></html>"
    output_path.write_text(html)
    print(f"✅ HTML report → {output_path}")


if __name__ == "__main__":
    main()
