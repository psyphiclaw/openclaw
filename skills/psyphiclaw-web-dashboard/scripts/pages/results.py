"""Results page for PsyPhiClaw Dashboard — chart gallery, stats table, AI insights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import html, dcc

try:
    import plotly.graph_objects as go
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
    """Create the analysis results page layout."""
    figures = _scan_figures(project_dir)

    return html.Div([
        html.H2("📋 分析结果", style={"color": PRIMARY, "marginBottom": "1.5rem"}),

        # Tabs
        dcc.Tabs(id="results-tabs", value="gallery", children=[
            dcc.Tab(label="🖼️ 图表画廊", value="gallery", style={"padding": "0.6rem 1.2rem"}),
            dcc.Tab(label="📊 统计结果", value="stats", style={"padding": "0.6rem 1.2rem"}),
            dcc.Tab(label="🤖 AI 洞察", value="insights", style={"padding": "0.6rem 1.2rem"}),
        ], style={"marginBottom": "1.5rem"}),

        # Gallery tab content
        html.Div(id="tab-content", children=[
            _gallery_content(figures, project_dir),
        ]),
    ])


def _gallery_content(figures: list[dict[str, str]], project_dir: Path) -> html.Div:
    """Create the figure gallery grid."""
    if not figures:
        return html.Div([
            html.Div([
                html.H3("暂无图表", style={"color": "#6c757d", "marginBottom": "1rem"}),
                html.P("运行分析脚本生成图表后，将在此显示。", style={"color": "#999"}),
            ], style={"textAlign": "center", "padding": "3rem", "background": CARD_BG, "borderRadius": "8px"}),
        ])

    cards = []
    for fig in figures[:24]:  # Limit display
        img_path = project_dir / fig["path"]
        cards.append(html.Div([
            html.Img(
                src=str(img_path),
                style={"width": "100%", "height": "200px", "objectFit": "cover", "borderRadius": "6px 6px 0 0"},
            ) if img_path.is_file() else html.Div("Missing", style={"height": "200px", "background": "#eee", "borderRadius": "6px 6px 0 0", "display": "flex", "alignItems": "center", "justifyContent": "center", "color": "#999"}),
            html.Div([
                html.Span(fig.get("section", ""), style={"fontSize": "0.7rem", "background": PRIMARY, "color": "#fff", "padding": "0.1rem 0.4rem", "borderRadius": "3px", "marginRight": "0.5rem"}),
                html.Span(fig.get("caption", fig["path"]), style={"fontSize": "0.8rem", "color": "#555"}),
            ], style={"padding": "0.75rem"}),
        ], style={"background": "#fff", "border": "1px solid #dee2e6", "borderRadius": "8px", "overflow": "hidden"}))

    return html.Div(cards, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(280px, 1fr))", "gap": "1rem"})


def _stats_content(project_dir: Path) -> html.Div:
    """Create the statistics results table."""
    # Try to find stats files
    stats_files = list(project_dir.rglob("*.json")) + list(project_dir.rglob("stats*.csv"))
    if not stats_files:
        return html.Div([
            html.P("暂无统计结果。运行统计分析脚本后将在此显示。", style={"color": "#999", "textAlign": "center", "padding": "2rem"}),
        ])

    tables = []
    for sf in stats_files[:5]:
        tables.append(html.Div([
            html.H4(sf.name, style={"color": "#16213e", "marginBottom": "0.5rem"}),
            html.Code(str(sf), style={"fontSize": "0.8rem", "color": "#6c757d"}),
        ], style={"padding": "1rem", "background": CARD_BG, "borderRadius": "8px", "marginBottom": "0.5rem"}))

    return html.Div(tables)


def _insights_content(project_dir: Path) -> html.Div:
    """Create the AI insights display."""
    insight_files = list(project_dir.rglob("*insight*")) + list(project_dir.rglob("*ai*"))
    if not insight_files:
        return html.Div([
            html.Div([
                html.Div("🤖", style={"fontSize": "3rem", "marginBottom": "1rem"}),
                html.H3("AI 洞察", style={"color": "#16213e", "marginBottom": "1rem"}),
                html.P("运行 AI 分析后将在此展示多模态行为洞察。", style={"color": "#6c757d"}),
                html.Div([
                    html.Div("模式发现", style={"fontWeight": "600", "marginBottom": "0.5rem", "color": PRIMARY}),
                    html.P("待分析...", style={"color": "#999"}),
                ], style={"background": CARD_BG, "padding": "1rem", "borderRadius": "8px", "marginBottom": "1rem"}),
                html.Div([
                    html.Div("异常检测", style={"fontWeight": "600", "marginBottom": "0.5rem", "color": DANGER}),
                    html.P("待分析...", style={"color": "#999"}),
                ], style={"background": CARD_BG, "padding": "1rem", "borderRadius": "8px"}),
            ], style={"maxWidth": "600px", "margin": "0 auto", "textAlign": "center", "padding": "2rem"}),
        ])
    return html.Div([html.P(f"Found {len(insight_files)} insight files.")])


def _scan_figures(project_dir: Path) -> list[dict[str, str]]:
    """Scan for figure files in the project directory."""
    import os
    img_exts = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
    modality_map = {"face": ["face"], "eeg": ["eeg"], "eye": ["eye"], "physio": ["ecg", "eda", "gsr"], "fnirs": ["fnirs"]}
    figures = []
    for root, _dirs, files in os.walk(project_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in img_exts:
                rel = os.path.relpath(os.path.join(root, f), project_dir)
                section = "general"
                fl = f.lower()
                for mod, keywords in modality_map.items():
                    if any(kw in fl for kw in keywords):
                        section = f"results_{mod}"
                        break
                figures.append({"path": rel, "caption": Path(f).stem.replace("_", " ").replace("-", " "), "section": section})
    return figures
