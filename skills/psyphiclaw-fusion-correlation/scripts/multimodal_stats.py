#!/usr/bin/env python3
"""Multimodal statistical analysis: regression, mixed effects, multiple comparison correction.

Usage:
    python multimodal_stats.py --csv-dir data/ \
        --predictor eeg_alpha eeg_beta \
        --outcome face_valence \
        --group-by subject \
        --correction fdr \
        --output stats/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

BLUE = "#4A90D9"
RED = "#E74C3C"


def multiple_comparison_correction(
    p_values: np.ndarray,
    method: str = "fdr",
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply multiple comparison correction.

    Parameters
    ----------
    p_values : Array of uncorrected p-values.
    method : ``fdr`` (Benjamini-Hochberg) or ``bonferroni``.
    alpha : Family-wise error rate for FDR or per-test rate for Bonferroni.

    Returns
    -------
    corrected_p, reject : Arrays of corrected p-values and boolean rejection decisions.
    """
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    if method == "bonferroni":
        corrected = np.minimum(p * n, 1.0)
        reject = corrected <= alpha
    elif method == "fdr":
        # Benjamini-Hochberg
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        ranks = np.arange(1, n + 1, dtype=float)
        corrected_sorted = sorted_p * n / ranks
        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
        corrected = np.empty(n)
        corrected[sorted_idx] = np.minimum(corrected_sorted, 1.0)
        reject = corrected <= alpha
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected, reject


def ols_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Ordinary least squares regression with statistics.

    Parameters
    ----------
    X : 2-D predictor array (n_samples × n_predictors). Should include intercept.
    y : 1-D outcome array.

    Returns
    -------
    dict with coefficients, R², F-statistic, p-values.
    """
    # Add intercept if not present
    if X.shape[1] > 0 and not np.allclose(X[:, 0], 1.0):
        X = np.column_stack([np.ones(len(y)), X])
        if feature_names:
            feature_names = ["intercept"] + list(feature_names)
    else:
        feature_names = feature_names or [f"pred_{i}" for i in range(X.shape[1])]

    # OLS: beta = (X'X)^{-1} X'y
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)

    # Predictions and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    n, k = X.shape
    mse = np.sum(residuals ** 2) / (n - k)

    # Standard errors
    try:
        se = np.sqrt(np.diag(mse * XtX_inv))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    # t-statistics and p-values
    t_stats = beta / se
    from scipy import stats
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=n - k)

    # R² and adjusted R²
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k) if n > k else 0.0

    # F-statistic
    if k > 1 and mse > 0:
        ss_model = ss_total - ss_residual
        f_stat = (ss_model / (k - 1)) / mse
        f_pval = stats.f.sf(f_stat, k - 1, n - k)
    else:
        f_stat, f_pval = np.nan, np.nan

    coefficients = {}
    for i, name in enumerate(feature_names):
        coefficients[name] = {
            "beta": round(float(beta[i]), 6),
            "se": round(float(se[i]), 6),
            "t": round(float(t_stats[i]), 4),
            "p": round(float(p_values[i]), 6),
        }

    return {
        "r_squared": round(float(r_squared), 6),
        "adj_r_squared": round(float(adj_r_squared), 6),
        "f_statistic": round(float(f_stat), 4),
        "f_pvalue": round(float(f_pval), 6),
        "n_obs": n,
        "n_predictors": k - 1,
        "coefficients": coefficients,
    }


def mixed_effects_summary(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    group_col: str,
) -> dict[str, Any]:
    """Simplified mixed-effects model using statsmodels if available.

    Falls back to OLS per-group + aggregation if statsmodels is unavailable.
    """
    try:
        import statsmodels.formula.api as smf
        import statsmodels.regression.mixed_linear_model as mlm

        formula = f"{outcome} ~ {' + '.join(predictors)}"
        model = smf.mixedlm(formula, df, groups=df[group_col])
        result = model.fit(reml=True)

        return {
            "method": "mixedlm_REML",
            "formula": formula,
            "group_col": group_col,
            "n_groups": int(df[group_col].nunique()),
            "n_obs": len(df),
            "coefficients": {
                name: {
                    "coef": round(float(result.params[name]), 6),
                    "se": round(float(result.bse[name]), 6),
                    "t": round(float(result.tvalues[name]), 4),
                    "p": round(float(result.pvalues[name]), 6),
                    "ci_lower": round(float(result.conf_int().loc[name, 0]), 6),
                    "ci_upper": round(float(result.conf_int().loc[name, 1]), 6),
                }
                for name in result.params.index
            },
            "converged": bool(result.converged),
            "aic": round(float(result.aic), 2),
            "bic": round(float(result.bic), 2),
        }
    except ImportError:
        print("⚠️  statsmodels not available, using OLS per-group fallback.", file=sys.stderr)
    except Exception as e:
        print(f"⚠️  Mixed model failed ({e}), using OLS per-group fallback.", file=sys.stderr)

    # Fallback: per-group OLS
    group_results: dict[str, Any] = {}
    for grp, sub_df in df.groupby(group_col):
        y = sub_df[outcome].values.astype(float)
        X = sub_df[predictors].values.astype(float)
        valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid.sum() < 5:
            continue
        res = ols_regression(X[valid], y[valid], predictors)
        group_results[str(grp)] = {
            "r_squared": res["r_squared"],
            "n_obs": int(valid.sum()),
        }

    # Aggregate
    r2_values = [v["r_squared"] for v in group_results.values()]
    return {
        "method": "ols_per_group_fallback",
        "formula": f"{outcome} ~ {' + '.join(predictors)}",
        "group_col": group_col,
        "n_groups": len(group_results),
        "mean_r_squared": round(float(np.mean(r2_values)), 6) if r2_values else None,
        "per_group": group_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal statistical analysis.")
    parser.add_argument("--session", help="MultiModalSession .h5 path.")
    parser.add_argument("--csv-dir", help="Directory of aligned CSVs.")
    parser.add_argument("--data-file", help="Single combined CSV with all features.")
    parser.add_argument("--predictor", nargs="+", required=True, help="Predictor column names.")
    parser.add_argument("--outcome", required=True, help="Outcome column name.")
    parser.add_argument("--group-by", default=None, help="Grouping column for mixed effects.")
    parser.add_argument("--correction", choices=["fdr", "bonferroni", "none"], default="fdr")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--output", required=True, help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.data_file:
        df = pd.read_csv(args.data_file)
    elif args.csv_dir:
        csvs = sorted(Path(args.csv_dir).glob("*.csv"))
        dfs = []
        for f in csvs:
            d = pd.read_csv(f)
            d["_source"] = f.stem
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    elif args.session:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "psyphiclaw-fusion-align" / "scripts"))
        from session_manager import MultiModalSession
        session = MultiModalSession.load(args.session)
        dfs = []
        for name, mod in session.modalities.items():
            mdf = mod.to_dataframe()
            mdf["_modality"] = name
            dfs.append(mdf)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        parser.error("Provide --data-file, --csv-dir, or --session.")

    required_cols = args.predictor + [args.outcome]
    if args.group_by:
        required_cols.append(args.group_by)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        print(f"   Available: {list(df.columns)}")
        sys.exit(1)

    # Clean NaN
    clean_df = df[required_cols].dropna()
    print(f"📊 Using {len(clean_df)} observations (from {len(df)} total)")

    # OLS Regression
    y = clean_df[args.outcome].values.astype(float)
    X = clean_df[args.predictor].values.astype(float)
    reg_result = ols_regression(X, y, args.predictor)
    reg_path = out_dir / "regression_results.json"
    with open(reg_path, "w") as f:
        json.dump(reg_result, f, indent=2)
    print(f"✅ OLS Regression: R²={reg_result['r_squared']:.4f}, "
          f"adj.R²={reg_result['adj_r_squared']:.4f}, F={reg_result['f_statistic']:.2f}")

    # Multiple comparison correction on predictor p-values
    pred_pvalues = np.array([
        reg_result["coefficients"][p]["p"] for p in args.predictor
        if p in reg_result["coefficients"]
    ])
    if args.correction != "none" and len(pred_pvalues) > 0:
        corrected, rejected = multiple_comparison_correction(pred_pvalues, args.correction, args.alpha)
        corr_results = []
        pred_names = [p for p in args.predictor if p in reg_result["coefficients"]]
        for i, name in enumerate(pred_names):
            corr_results.append({
                "predictor": name,
                "raw_p": round(float(pred_pvalues[i]), 6),
                "corrected_p": round(float(corrected[i]), 6),
                "significant": bool(rejected[i]),
            })
        corr_df = pd.DataFrame(corr_results)
        corr_path = out_dir / "correction_results.csv"
        corr_df.to_csv(corr_path, index=False)
        sig_count = sum(1 for r in corr_results if r["significant"])
        print(f"✅ {args.correction.upper()} correction: {sig_count}/{len(corr_results)} significant")

    # Mixed effects (if group-by specified)
    if args.group_by:
        me_result = mixed_effects_summary(clean_df, args.outcome, args.predictor, args.group_by)
        me_path = out_dir / "mixed_effects_results.json"
        with open(me_path, "w") as f:
            json.dump(me_result, f, indent=2, default=str)
        print(f"✅ Mixed Effects ({me_result['method']}): AIC={me_result.get('aic', 'N/A')}")

    # Generate text report
    report_lines = [
        "=" * 60,
        "PsyPhiClaw Statistical Report",
        "=" * 60,
        f"Outcome: {args.outcome}",
        f"Predictors: {', '.join(args.predictor)}",
        f"N = {len(clean_df)}",
        "",
        "OLS Regression:",
        f"  R² = {reg_result['r_squared']:.4f}  (adj. {reg_result['adj_r_squared']:.4f})",
        f"  F({reg_result['n_predictors']}, {reg_result['n_obs'] - reg_result['n_predictors'] - 1}) "
        f"= {reg_result['f_statistic']:.2f}, p = {reg_result['f_pvalue']:.4f}",
        "",
        "Coefficients:",
    ]
    for name, coeff in reg_result["coefficients"].items():
        sig = "***" if coeff["p"] < 0.001 else "**" if coeff["p"] < 0.01 else "*" if coeff["p"] < 0.05 else ""
        report_lines.append(
            f"  {name:>20s}: β={coeff['beta']:>8.4f}  SE={coeff['se']:.4f}  "
            f"t={coeff['t']:>6.3f}  p={coeff['p']:.4f} {sig}"
        )

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\n📄 Report → {report_path}")
    print(f"✅ All results saved to {out_dir}/")


if __name__ == "__main__":
    main()
