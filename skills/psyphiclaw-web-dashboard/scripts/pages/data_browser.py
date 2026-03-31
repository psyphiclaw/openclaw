"""Data Browser page for PsyPhiClaw Dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import html, dcc, Input, Output, State

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

PRIMARY = "#4A90D9"
DANGER = "#E74C3C"
CARD_BG = "#f8f9fa"


def create_layout(project_dir: Path) -> html.Div:
    """Create the data browser page layout."""
    return html.Div([
        html.H2("📈 数据浏览", style={"color": PRIMARY, "marginBottom": "1.5rem"}),
        # Controls
        html.Div([
            html.Div([
                html.Label("选择模态:", style={"fontWeight": "600", "marginBottom": "0.5rem", "display": "block"}),
                dcc.Dropdown(
                    id="modality-select",
                    options=[
                        {"label": " facial表情", "value": "face"},
                        {"label": " EEG 脑电", "value": "eeg"},
                        {"label": " 眼动追踪", "value": "eye"},
                        {"label": " 生理信号", "value": "physio"},
                        {"label": " fNIRS", "value": "fnirs"},
                    ],
                    value="face",
                    clearable=False,
                    style={"width": "100%"},
                ),
            ], style={"flex": "1", "minWidth": "200px", "marginRight": "1rem"}),
            html.Div([
                html.Label("时间范围 (秒):", style={"fontWeight": "600", "marginBottom": "0.5rem", "display": "block"}),
                dcc.RangeSlider(
                    id="time-range",
                    min=0, max=600, step=1,
                    value=[0, 600],
                    marks={i: str(i) for i in range(0, 601, 60)},
                ),
            ], style={"flex": "2", "minWidth": "300px"}),
            html.Div([
                html.Label("通道选择:", style={"fontWeight": "600", "marginBottom": "0.5rem", "display": "block"}),
                dcc.Dropdown(id="channel-select", multi=True, placeholder="选择通道...", style={"width": "100%"}),
            ], style={"flex": "1", "minWidth": "200px", "marginLeft": "1rem"}),
        ], style={"display": "flex", "gap": "1rem", "alignItems": "flex-start", "marginBottom": "1.5rem"}),
        # Plot area
        html.Div(id="timeline-container", style={"background": CARD_BG, "padding": "1rem", "borderRadius": "8px", "minHeight": "500px"}, children=[
            _placeholder_plot(),
        ]),
        # Info bar
        html.Div(id="data-info-bar", style={"marginTop": "1rem", "padding": "0.75rem", "background": "#e8f0fe", "borderRadius": "6px", "color": "#4A90D9", "fontSize": "0.85rem"}, children=[
            "选择数据文件以开始浏览。",
        ]),
    ])


def _placeholder_plot() -> Any:
    """Create a placeholder plot when no data is loaded."""
    if not HAS_PLOTLY:
        return html.Div([
            html.P("需要安装 plotly: pip install plotly", style={"color": DANGER}),
        ])
    fig = go.Figure()
    fig.update_layout(
        title="多模态时间线",
        xaxis_title="时间 (秒)",
        yaxis_title="信号值",
        template="plotly_white",
        height=450,
        margin={"t": 40, "b": 40, "l": 60, "r": 20},
    )
    fig.add_annotation(
        text="选择数据文件以加载时间线",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font={"size": 16, "color": "#999"},
    )
    return dcc.Graph(id="timeline-plot", figure=fig, config={"responsive": True, "displayModeBar": True})


def load_csv_timeline(csv_path: Path, channels: list[str] | None = None, time_range: tuple[float, float] = (0, 600)) -> Any:
    """Load a CSV file and create a Plotly timeline figure."""
    if not HAS_PLOTLY or not HAS_PANDAS:
        return _placeholder_plot()

    try:
        df = pd.read_csv(csv_path, nrows=50000)
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error loading {csv_path.name}: {e}")
        return dcc.Graph(id="timeline-plot", figure=fig)

    # Auto-detect time column
    time_col = None
    for col in ["time", "timestamp", "Time", "Timestamp", "Sample", "sample"]:
        if col in df.columns:
            time_col = col
            break
    if time_col is None and len(df.columns) > 0:
        time_col = df.columns[0]
    if time_col is None:
        return _placeholder_plot()

    # Select channels
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)
    if channels:
        plot_cols = [c for c in channels if c in numeric_cols]
    else:
        plot_cols = numeric_cols[:8]  # Limit to 8 channels

    # Filter time range
    try:
        mask = (df[time_col] >= time_range[0]) & (df[time_col] <= time_range[1])
        df_plot = df.loc[mask]
    except Exception:
        df_plot = df

    fig = go.Figure()
    colors = [PRIMARY, DANGER, "#27ae60", "#f39c12", "#8e44ad", "#1abc9c", "#e67e22", "#34495e"]
    for i, col in enumerate(plot_cols):
        fig.add_trace(go.Scatter(
            x=df_plot[time_col],
            y=df_plot[col],
            name=col,
            line=dict(color=colors[i % len(colors)], width=1.2),
            opacity=0.85,
        ))

    fig.update_layout(
        title=f"时间线 — {csv_path.name}",
        xaxis_title=time_col,
        yaxis_title="信号值",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin={"t": 40, "b": 40, "l": 60, "r": 20},
    )
    return dcc.Graph(id="timeline-plot", figure=fig, config={"scrollZoom": True, "displayModeBar": True})
