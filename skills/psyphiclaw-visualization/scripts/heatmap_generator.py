#!/usr/bin/env python3
"""Heatmap generator: correlation matrix, feature×time, and clustered heatmaps.

Usage:
    python heatmap_generator.py --data features.csv --type correlation --output heatmap.html
    python heatmap_generator.py --data features.csv --type feature_time \
        --time-col timestamp --output heatmap.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

BLUE = "#4A90D9"
RED = "#E74C3C"


def correlation_heatmap(
    df: pd.DataFrame,
    method: str = "spearman",
    title: str = "Cross-Modal Correlation Matrix",
) -> tuple["go.Figure", "plt.Figure"]:
    """Generate correlation matrix heatmap."""
    corr = df.corr(method=method)

    # Plotly
    fig_px = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=10),
            hoverongaps=False,
        )
    )
    fig_px.update_layout(
        title=title,
        width=max(600, len(corr) * 60),
        height=max(500, len(corr) * 50),
        margin=dict(l=120, r=40, t=60, b=120),
        xaxis=dict(tickangle=-45),
    )

    # Matplotlib
    fig_mpl, ax = plt.subplots(figsize=(max(8, len(corr) * 0.6), max(6, len(corr) * 0.5)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap="RdBu_r", vmin=-1, vmax=1,
        center=0, annot=True, fmt=".2f",
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"{method} r"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig_px, fig_mpl


def feature_time_heatmap(
    df: pd.DataFrame,
    time_col: str,
    title: str = "Feature × Time Heatmap",
) -> tuple["go.Figure", "plt.Figure"]:
    """Generate feature × time heatmap."""
    ts = pd.to_numeric(df[time_col], errors="coerce")
    feature_cols = [c for c in df.columns if c != time_col]

    # Sort by time
    sort_idx = ts.argsort()
    data = df[feature_cols].iloc[sort_idx].values

    # Z-score each feature for visualization
    from scipy.stats import zscore
    data_z = np.column_stack([zscore(data[:, i]) if np.std(data[:, i]) > 0 else data[:, i]
                               for i in range(data.shape[1])])

    # Plotly
    fig_px = go.Figure(
        data=go.Heatmap(
            z=data_z.T,
            x=ts.iloc[sort_idx].values,
            y=feature_cols,
            colorscale="RdBu_r",
            zmin=-3,
            zmax=3,
        )
    )
    fig_px.update_layout(
        title=title,
        xaxis_title="Time (ms)",
        yaxis_title="Features",
        height=max(300, len(feature_cols) * 40),
        margin=dict(l=150, r=20, t=60, b=60),
    )

    # Matplotlib
    fig_mpl, ax = plt.subplots(figsize=(max(12, len(data_z) * 0.005), max(6, len(feature_cols) * 0.4)))
    im = ax.imshow(data_z.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                    extent=[ts.min(), ts.max(), -0.5, len(feature_cols) - 0.5])
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(feature_cols)
    ax.set_xlabel("Time (ms)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Z-score", shrink=0.8)
    plt.tight_layout()

    return fig_px, fig_mpl


def clustered_heatmap(
    df: pd.DataFrame,
    title: str = "Clustered Heatmap",
) -> "plt.Figure":
    """Generate clustered heatmap using seaborn."""
    from scipy.stats import zscore

    feature_cols = [c for c in df.columns if df[c].dtype in (np.float64, np.float32, np.int64, float, int)]
    if not feature_cols:
        print("❌ No numeric columns found.")
        sys.exit(1)

    data = df[feature_cols].apply(lambda col: zscore(col) if np.std(col) > 0 else col)

    # Clustermap
    cg = sns.clustermap(
        data.T,
        cmap="RdBu_r",
        vmin=-3, vmax=3,
        metric="correlation",
        method="average",
        figsize=(max(10, data.shape[1] * 0.4), max(8, len(feature_cols) * 0.3)),
        cbar_kws={"label": "Z-score"},
        yticklabels=True,
    )
    cg.fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.close("all")
    return cg.fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate heatmaps.")
    parser.add_argument("--data", required=True, help="Input CSV file.")
    parser.add_argument("--type", choices=["correlation", "feature_time", "clustered"],
                        default="correlation", help="Heatmap type.")
    parser.add_argument("--time-col", default="timestamp", help="Time column for feature_time type.")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--title", default=None, help="Custom title.")
    parser.add_argument("--output", required=True, help="Output path (HTML or PNG).")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"📊 Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.type == "correlation":
        title = args.title or f"Correlation Matrix ({args.method})"
        fig_px, fig_mpl = correlation_heatmap(df, args.method, title)
        if HAS_PLOTLY and out_path.suffix == ".html":
            fig_px.write_html(str(out_path))
            print(f"✅ Plotly → {out_path}")
        png_path = out_path.with_suffix(".png")
        fig_mpl.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig_mpl)
        print(f"✅ Matplotlib → {png_path}")

    elif args.type == "feature_time":
        title = args.title or "Feature × Time"
        fig_px, fig_mpl = feature_time_heatmap(df, args.time_col, title)
        if HAS_PLOTLY and out_path.suffix == ".html":
            fig_px.write_html(str(out_path))
            print(f"✅ Plotly → {out_path}")
        png_path = out_path.with_suffix(".png")
        fig_mpl.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig_mpl)
        print(f"✅ Matplotlib → {png_path}")

    elif args.type == "clustered":
        title = args.title or "Clustered Heatmap"
        fig = clustered_heatmap(df, title)
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Clustered heatmap → {png_path}")


if __name__ == "__main__":
    main()
