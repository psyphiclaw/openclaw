#!/usr/bin/env python3
"""Interactive multi-modal timeline visualization using Plotly.

Creates a multi-subplot layout with EEG signals, facial expression dimensions,
physiological signals, and event markers on a shared time axis.

Usage:
    python multimodal_timeline.py --session session.h5 \
        --modalities eeg face physio \
        --events events.csv \
        --output timeline.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

BLUE = "#4A90D9"
RED = "#E74C3C"
GREEN = "#2ECC71"
ORANGE = "#F39C12"
GRAY = "#95A5A6"
BG = "#FAFAFA"


def load_session_data(
    session_path: str,
    modality_names: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Load modalities from MultiModalSession .h5."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "psyphiclaw-fusion-align" / "scripts"))
    from session_manager import MultiModalSession
    session = MultiModalSession.load(session_path)
    result = {}
    for name in (modality_names or list(session.modalities.keys())):
        mod = session.get_modality(name)
        if mod is None:
            continue
        df = mod.to_dataframe()
        ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
        result[name] = {
            "dataframe": df,
            "timestamps_ms": df[ts_col].values if ts_col else None,
            "columns": [c for c in df.columns if c != ts_col],
        }
    return result


def load_csv_data(
    csv_dir: str,
    modality_names: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Load modalities from a directory of aligned CSVs."""
    result = {}
    for csv_file in sorted(Path(csv_dir).glob("*.csv")):
        name = csv_file.stem
        if modality_names and name not in modality_names:
            continue
        df = pd.read_csv(csv_file)
        ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
        result[name] = {
            "dataframe": df,
            "timestamps_ms": df[ts_col].values if ts_col else None,
            "columns": [c for c in df.columns if c != ts_col],
        }
    return result


def load_events(path: str) -> pd.DataFrame:
    """Load event markers CSV."""
    df = pd.read_csv(path)
    for col in ("timestamp_ms", "timestamp", "time_ms", "Timestamp"):
        if col in df.columns:
            df.rename(columns={col: "timestamp_ms"}, inplace=True)
            break
    if "label" not in df.columns:
        df["label"] = "event"
    return df


def build_timeline(
    modalities: dict[str, dict[str, Any]],
    events: Optional[pd.DataFrame] = None,
    max_points: int = 5000,
) -> "go.Figure":
    """Build a Plotly figure with one subplot per modality + events row."""
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Install with: pip install plotly")

    # Determine total subplot rows
    mod_names = list(modalities.keys())
    n_event_rows = 1 if events is not None else 0
    total_rows = len(mod_names) + n_event_rows
    subplot_titles = list(mod_names) + (["Events"] if n_event_rows else [])

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
    )

    # Color cycle for modalities
    colors = [BLUE, RED, GREEN, ORANGE, "#9B59B6", "#1ABC9C"]

    for row_idx, mod_name in enumerate(mod_names):
        mod = modalities[mod_name]
        df = mod["dataframe"]
        ts = mod["timestamps_ms"]
        cols = mod["columns"][:6]  # Limit to 6 traces per subplot

        for col_idx, col in enumerate(cols):
            y_data = df[col].values
            # Downsample for performance
            if len(y_data) > max_points:
                step = len(y_data) // max_points
                y_data = y_data[::step]
                x_data = ts[::step] if ts is not None else np.arange(len(y_data))
            else:
                x_data = ts if ts is not None else np.arange(len(y_data))

            color = colors[(row_idx + col_idx) % len(colors)]
            fig.add_trace(
                go.Scattergl(
                    x=x_data,
                    y=y_data,
                    name=f"{mod_name}_{col}",
                    line=dict(color=color, width=1),
                    opacity=0.8,
                ),
                row=row_idx + 1,
                col=1,
            )

    # Event markers
    if events is not None and n_event_rows:
        evt_ts = pd.to_numeric(events["timestamp_ms"], errors="coerce").dropna().values
        evt_labels = events.loc[events["timestamp_ms"].notna(), "label"].values

        # Scatter with vertical lines
        for i, (t, label) in enumerate(zip(evt_ts, evt_labels)):
            fig.add_trace(
                go.Scatter(
                    x=[t, t],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color=ORANGE, width=2, dash="dash"),
                    showlegend=False,
                    name=f"event_{i}" if i == 0 else None,
                ),
                row=total_rows,
                col=1,
            )
        # Add markers at top
        fig.add_trace(
            go.Scatter(
                x=evt_ts,
                y=[1] * len(evt_ts),
                mode="markers+text",
                marker=dict(color=ORANGE, size=8),
                text=evt_labels,
                textposition="top center",
                textfont=dict(size=8),
                showlegend=False,
            ),
            row=total_rows,
            col=1,
        )

    # Layout
    fig.update_layout(
        height=300 * total_rows,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    for i in range(1, total_rows + 1):
        fig.update_yaxes(row=i, col=1, gridcolor="#E8E8E8")
        if i < total_rows:
            fig.update_xaxes(row=i, col=1, showticklabels=False)
    fig.update_xaxes(row=total_rows, col=1, title_text="Time (ms)")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-modal interactive timeline.")
    parser.add_argument("--session", help="MultiModalSession .h5 path.")
    parser.add_argument("--csv-dir", help="Directory of aligned CSVs (alternative).")
    parser.add_argument("--modalities", nargs="+", default=None, help="Modality names to plot.")
    parser.add_argument("--events", default=None, help="Event markers CSV.")
    parser.add_argument("--max-points", type=int, default=5000, help="Max data points per trace.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument("--png", default=None, help="Also save as PNG.")
    args = parser.parse_args()

    # Load data
    if args.session:
        modalities = load_session_data(args.session, args.modalities)
    elif args.csv_dir:
        modalities = load_csv_data(args.csv_dir, args.modalities)
    else:
        parser.error("Provide --session or --csv-dir.")

    if not modalities:
        print("❌ No modalities loaded.")
        sys.exit(1)

    print(f"📊 Loaded {len(modalities)} modalities: {list(modalities.keys())}")

    events = load_events(args.events) if args.events else None

    fig = build_timeline(modalities, events, args.max_points)

    # Save HTML
    html_path = Path(args.output)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))
    print(f"✅ Interactive timeline → {html_path}")

    # Save PNG
    png_path = Path(args.png) if args.png else html_path.with_suffix(".png")
    if args.png or html_path.suffix == ".html":
        try:
            fig.write_image(str(png_path), width=1200, height=300 * len(modalities))
            print(f"✅ Static screenshot → {png_path}")
        except Exception as e:
            print(f"⚠️  PNG export failed (need kaleido): {e}")


if __name__ == "__main__":
    main()
