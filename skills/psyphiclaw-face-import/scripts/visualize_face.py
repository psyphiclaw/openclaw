#!/usr/bin/env python3
"""Visualize FaceReader data: VAD time series, emotion stacked area, AU heatmap, face presence timeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Re-use import_facereader functions
sys.path.insert(0, str(Path(__file__).parent))
from import_facereader import read_csv, standardize_columns, extract_time_column, filter_by_time

logger = logging.getLogger("psyphiclaw-face-visualize")

# Color scheme
PRIMARY = "#4A90D9"
SECONDARY = "#E74C3C"
VAD_COLORS = {"Valence": "#4A90D9", "Arousal": "#E74C3C", "Dominance": "#2ECC71"}
EMOTION_COLORS = {
    "Neutral": "#95A5A6", "Happy": "#F1C40F", "Sad": "#3498DB",
    "Angry": "#E74C3C", "Surprised": "#E67E22", "Scared": "#9B59B6",
    "Disgusted": "#1ABC9C", "Contempt": "#34495E",
}


def _to_seconds(df: pd.DataFrame, time_col: str) -> pd.Series:
    """Convert timestamp column to seconds."""
    vals = df[time_col].dropna()
    if vals.max() > 100000:  # likely milliseconds
        return df[time_col] / 1000.0
    return df[time_col]


def plot_vad(df: pd.DataFrame, time_col: str) -> go.Figure:
    """VAD time series plot."""
    vad_cols = [c for c in ("Valence", "Arousal", "Dominance") if c in df.columns]
    if not vad_cols:
        logger.warning("No VAD columns found")
        return go.Figure()

    t = _to_seconds(df, time_col)
    fig = go.Figure()
    for col in vad_cols:
        fig.add_trace(go.Scatter(x=t, y=df[col], mode="lines", name=col,
                                 line=dict(color=VAD_COLORS.get(col, PRIMARY), width=1.5)))
    fig.update_layout(title="VAD Emotional Dimensions", xaxis_title="Time (s)", yaxis_title="Intensity (0-1)",
                      template="plotly_white", height=400)
    return fig


def plot_emotions(df: pd.DataFrame, time_col: str) -> go.Figure:
    """Basic emotion probability stacked area plot."""
    emo_cols = [c for c in EMOTION_COLORS if c in df.columns]
    if not emo_cols:
        logger.warning("No emotion columns found")
        return go.Figure()

    t = _to_seconds(df, time_col)
    fig = go.Figure()
    for col in emo_cols:
        fig.add_trace(go.Scatter(x=t, y=df[col].fillna(0), mode="lines", name=col,
                                 stackgroup="one",
                                 line=dict(color=EMOTION_COLORS[col], width=0.5),
                                 fillcolor=EMOTION_COLORS[col] + "80"))
    fig.update_layout(title="Basic Emotion Probabilities (Stacked)", xaxis_title="Time (s)",
                      yaxis_title="Probability", template="plotly_white", height=400,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def plot_au_heatmap(df: pd.DataFrame, time_col: str, max_au: int = 50) -> go.Figure:
    """AU heatmap: time x AU matrix."""
    au_cols = sorted([c for c in df.columns if c.lower().startswith("actionunit")])[:max_au]
    if not au_cols:
        logger.warning("No ActionUnit columns found")
        return go.Figure()

    t = _to_seconds(df, time_col)
    # Downsample if too many rows
    step = max(1, len(df) // 2000)
    data = df[au_cols].iloc[::step].T
    t_ds = t.iloc[::step]

    fig = go.Figure(data=go.Heatmap(z=data.values, x=t_ds.values, y=data.index,
                                     colorscale="RdBu_r", zmid=0.5))
    fig.update_layout(title="Action Unit Intensity Heatmap", xaxis_title="Time (s)",
                      yaxis_title="Action Unit", template="plotly_white", height=max(300, len(au_cols) * 15))
    return fig


def plot_face_presence(df: pd.DataFrame, time_col: str) -> go.Figure:
    """Face presence timeline."""
    fp_cols = [c for c in df.columns if "facepresence" in c.lower() or "face_presence" in c.lower()]
    if not fp_cols:
        logger.warning("No face presence column found")
        return go.Figure()

    t = _to_seconds(df, time_col)
    fp = df[fp_cols[0]].fillna(0)
    colors = [SECONDARY if v == 0 else PRIMARY for v in fp]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=t, y=fp, marker_color=colors, name="Face Present",
                         marker_line_width=0))
    fig.update_layout(title="Face Presence Timeline", xaxis_title="Time (s)",
                      yaxis_title="Face Detected", template="plotly_white", height=250,
                      yaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]))
    return fig


def save_figure(fig: go.Figure, name: str, output_dir: Path) -> None:
    """Save figure as PNG and HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    html_path = output_dir / f"{name}.html"
    fig.write_image(str(png_path), width=1200, height=fig.layout.height or 400)
    fig.write_html(str(html_path))
    logger.info("Saved %s and %s", png_path, html_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize FaceReader data")
    parser.add_argument("csv_path", type=Path, help="Path to FaceReader CSV file")
    parser.add_argument("--output-dir", type=Path, default="./plots", help="Output directory for plots")
    parser.add_argument("--time-range", nargs=2, type=float, metavar=("START_MS", "END_MS"), help="Time range filter (ms)")
    parser.add_argument("--vad", action="store_true", default=True, help="Plot VAD dimensions")
    parser.add_argument("--emotions", action="store_true", default=True, help="Plot emotion probabilities")
    parser.add_argument("--au-heatmap", action="store_true", default=True, help="Plot AU heatmap")
    parser.add_argument("--face-presence", action="store_true", default=True, help="Plot face presence")
    parser.add_argument("--no-vad", action="store_true", help="Skip VAD plot")
    parser.add_argument("--no-emotions", action="store_true", help="Skip emotions plot")
    parser.add_argument("--no-au", action="store_true", help="Skip AU heatmap")
    parser.add_argument("--no-face", action="store_true", help="Skip face presence plot")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    df = read_csv(args.csv_path)
    df = standardize_columns(df)
    time_col = extract_time_column(df)

    if args.time_range:
        df = filter_by_time(df, time_col, args.time_range[0], args.time_range[1])

    out = Path(args.output_dir)

    if args.no_vad:
        args.vad = False
    if args.no_emotions:
        args.emotions = False
    if args.no_au:
        args.au_heatmap = False
    if args.no_face:
        args.face_presence = False

    if args.vad:
        save_figure(plot_vad(df, time_col), "vad_timeseries", out)
    if args.emotions:
        save_figure(plot_emotions(df, time_col), "emotion_stacked", out)
    if args.au_heatmap:
        save_figure(plot_au_heatmap(df, time_col), "au_heatmap", out)
    if args.face_presence:
        save_figure(plot_face_presence(df, time_col), "face_presence", out)

    logger.info("All plots saved to %s", out.resolve())
    print(f"✅ Plots saved to {out.resolve()}")


if __name__ == "__main__":
    main()
