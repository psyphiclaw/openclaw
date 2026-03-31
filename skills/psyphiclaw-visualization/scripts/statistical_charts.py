#!/usr/bin/env python3
"""Statistical charts: violin, box+scatter, radar, bar charts.

All charts support both matplotlib (publication-ready) and plotly (interactive) output.

Usage:
    python statistical_charts.py --data data.csv --chart violin \
        --group-col condition --value-col valence --output violin.png
    python statistical_charts.py --data data.csv --chart bar \
        --group-col condition --value-col valence --error-col se \
        --output bar.html --format plotly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

BLUE = "#4A90D9"
RED = "#E74C3C"
GREEN = "#2ECC71"
ORANGE = "#F39C12"
COLORS = [BLUE, RED, GREEN, ORANGE, "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB"]


def violin_plot_mpl(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str = "",
    palette: Optional[list[str]] = None,
) -> plt.Figure:
    """Matplotlib violin plot."""
    groups = df[group_col].dropna().unique()
    colors = palette or COLORS[:len(groups)]

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.5), 6))
    sns.violinplot(
        data=df, x=group_col, y=value_col, ax=ax,
        palette=colors, inner="box", cut=0,
    )
    sns.stripplot(
        data=df, x=group_col, y=value_col, ax=ax,
        color="black", alpha=0.3, size=3, jitter=True,
    )
    ax.set_title(title or f"{value_col} by {group_col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(group_col, fontsize=11)
    ax.set_ylabel(value_col, fontsize=11)
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def violin_plot_plotly(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str = "",
) -> "go.Figure":
    """Plotly violin plot."""
    groups = df[group_col].dropna().unique()
    fig = go.Figure()
    for i, grp in enumerate(groups):
        sub = df[df[group_col] == grp][value_col].dropna()
        fig.add_trace(go.Violin(
            y=sub.values, name=str(grp),
            line_color=COLORS[i % len(COLORS)],
            box_visible=True, meanline_visible=True,
            opacity=0.7,
        ))
    fig.update_layout(
        title=title or f"{value_col} by {group_col}",
        yaxis_title=value_col,
        xaxis_title=group_col,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def box_scatter_mpl(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str = "",
    palette: Optional[list[str]] = None,
) -> plt.Figure:
    """Matplotlib box plot with scatter overlay."""
    groups = df[group_col].dropna().unique()
    colors = palette or COLORS[:len(groups)]

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.5), 6))
    sns.boxplot(
        data=df, x=group_col, y=value_col, ax=ax,
        palette=colors, width=0.5, fliersize=0,
    )
    sns.stripplot(
        data=df, x=group_col, y=value_col, ax=ax,
        color="black", alpha=0.4, size=3, jitter=True,
    )
    ax.set_title(title or f"{value_col} by {group_col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(group_col, fontsize=11)
    ax.set_ylabel(value_col, fontsize=11)
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def radar_plot_mpl(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
    title: str = "",
    palette: Optional[list[str]] = None,
) -> plt.Figure:
    """Matplotlib radar/spider chart for multi-dimensional comparison."""
    groups = df[group_col].dropna().unique()
    colors = palette or COLORS[:len(groups)]
    n_vars = len(value_cols)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, grp in enumerate(groups):
        values = df[df[group_col] == grp][value_cols].mean().values.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=str(grp), color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(value_cols, fontsize=9)
    ax.set_title(title or "Multi-Dimensional Comparison", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    return fig


def bar_plot_mpl(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    error_col: Optional[str] = None,
    title: str = "",
    palette: Optional[list[str]] = None,
) -> plt.Figure:
    """Matplotlib bar chart with error bars."""
    agg = df.groupby(group_col)[value_col].agg(["mean", "std", "sem", "count"])
    groups = agg.index.tolist()
    means = agg["mean"].values
    errors = df[error_col].groupby(df[group_col]).mean().values if error_col else agg["sem"].values

    colors = palette or COLORS[:len(groups)]
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.5), 6))
    bars = ax.bar(range(len(groups)), means, color=colors, edgecolor="white", linewidth=1.5,
                  yerr=errors, capsize=5, error_kw=dict(elinewidth=1.5))

    # Add value labels
    for bar, mean, err in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01 * max(means),
                f"{mean:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(value_col, fontsize=11)
    ax.set_xlabel(group_col, fontsize=11)
    ax.set_title(title or f"{value_col} by {group_col}", fontsize=13, fontweight="bold")
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def bar_plot_plotly(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    error_col: Optional[str] = None,
    title: str = "",
) -> "go.Figure":
    """Plotly bar chart with error bars."""
    agg = df.groupby(group_col)[value_col].agg(["mean", "sem"]).reset_index()
    errors = None
    if error_col:
        err_agg = df.groupby(group_col)[error_col].mean().values
        errors = err_agg
    else:
        errors = agg["sem"].values

    fig = go.Figure(
        data=go.Bar(
            x=agg[group_col],
            y=agg["mean"],
            error_y=dict(type="data", array=errors, visible=True),
            marker_color=COLORS[:len(agg)],
        )
    )
    fig.update_layout(
        title=title or f"{value_col} by {group_col}",
        yaxis_title=value_col,
        xaxis_title=group_col,
        template="plotly_white",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical charts generator.")
    parser.add_argument("--data", required=True, help="Input CSV file.")
    parser.add_argument("--chart", choices=["violin", "box", "radar", "bar"], required=True)
    parser.add_argument("--group-col", required=True, help="Grouping column.")
    parser.add_argument("--value-col", help="Value column (single).")
    parser.add_argument("--value-cols", nargs="+", help="Value columns (multiple, for radar).")
    parser.add_argument("--error-col", default=None, help="Error bar column (for bar chart).")
    parser.add_argument("--title", default=None, help="Chart title.")
    parser.add_argument("--format", choices=["mpl", "plotly", "both"], default="both",
                        help="Output format.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PNG output.")
    parser.add_argument("--output", required=True, help="Output path.")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    value_col = args.value_col
    value_cols = args.value_cols or ([value_col] if value_col else None)

    if not value_cols and args.chart != "radar":
        parser.error("--value-col is required for non-radar charts.")

    title = args.title or ""

    # Generate charts
    if args.chart == "violin":
        if args.format in ("mpl", "both"):
            fig = violin_plot_mpl(df, args.group_col, value_col, title)
            fig.savefig(out_path.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Matplotlib violin → {out_path.with_suffix('.png')}")
        if HAS_PLOTLY and args.format in ("plotly", "both"):
            fig = violin_plot_plotly(df, args.group_col, value_col, title)
            fig.write_html(str(out_path.with_suffix(".html")))
            print(f"✅ Plotly violin → {out_path.with_suffix('.html')}")

    elif args.chart == "box":
        if args.format in ("mpl", "both"):
            fig = box_scatter_mpl(df, args.group_col, value_col, title)
            fig.savefig(out_path.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Matplotlib box → {out_path.with_suffix('.png')}")

    elif args.chart == "radar":
        if not value_cols:
            parser.error("--value-cols required for radar chart.")
        fig = radar_plot_mpl(df, args.group_col, value_cols, title)
        fig.savefig(out_path.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Radar → {out_path.with_suffix('.png')}")

    elif args.chart == "bar":
        if args.format in ("mpl", "both"):
            fig = bar_plot_mpl(df, args.group_col, value_col, args.error_col, title)
            fig.savefig(out_path.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Matplotlib bar → {out_path.with_suffix('.png')}")
        if HAS_PLOTLY and args.format in ("plotly", "both"):
            fig = bar_plot_plotly(df, args.group_col, value_col, args.error_col, title)
            fig.write_html(str(out_path.with_suffix(".html")))
            print(f"✅ Plotly bar → {out_path.with_suffix('.html')}")


if __name__ == "__main__":
    main()
