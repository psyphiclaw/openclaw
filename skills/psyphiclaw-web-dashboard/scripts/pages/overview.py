"""Overview page for PsyPhiClaw Dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import html, dcc


PRIMARY = "#4A90D9"
DANGER = "#E74C3C"
CARD_BG = "#f8f9fa"


def create_layout(project_dir: Path) -> html.Div:
    """Create the overview page layout with project statistics."""
    stats = _scan_project(project_dir)

    return html.Div([
        html.H2("📊 项目概览", style={"color": PRIMARY, "marginBottom": "1.5rem"}),
        # Stat cards
        html.Div([
            _stat_card("被试数量", str(stats["subjects"]), PRIMARY),
            _stat_card("数据模态", str(stats["modalities"]), "#27ae60"),
            _stat_card("数据文件", str(stats["data_files"]), "#8e44ad"),
            _stat_card("图表文件", str(stats["figures"]), "#f39c12"),
            _stat_card("数据完整性", f"{stats['completeness']}%", DANGER if stats["completeness"] < 80 else PRIMARY),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))", "gap": "1rem", "marginBottom": "2rem"}),
        # Modality breakdown
        html.Div([
            html.H3("模态覆盖", style={"color": "#16213e", "marginBottom": "1rem"}),
            _modality_table(stats["modality_details"]),
        ], style={"background": CARD_BG, "padding": "1.5rem", "borderRadius": "8px", "marginBottom": "2rem"}),
        # Recent activity
        html.Div([
            html.H3("最近分析", style={"color": "#16213e", "marginBottom": "1rem"}),
            html.P("暂无分析记录。运行分析脚本后将在此显示结果。", style={"color": "#6c757d"}),
        ], style={"background": CARD_BG, "padding": "1.5rem", "borderRadius": "8px"}),
    ])


def _stat_card(label: str, value: str, color: str) -> html.Div:
    """Create a statistics card."""
    return html.Div([
        html.Div(label, style={"fontSize": "0.8rem", "color": "#6c757d", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
        html.Div(value, style={"fontSize": "1.8rem", "fontWeight": "700", "color": color, "marginTop": "0.25rem"}),
    ], style={"background": CARD_BG, "border": "1px solid #dee2e6", "borderRadius": "8px", "padding": "1rem", "borderLeft": f"4px solid {color}"})


def _modality_table(modalities: dict[str, int]) -> html.Table:
    """Create a modality coverage table."""
    rows = [
        html.Tr([
            html.Td(mod, style={"padding": "0.5rem 1rem", "borderBottom": "1px solid #dee2e6", "fontWeight": "500"}),
            html.Td(str(count), style={"padding": "0.5rem 1rem", "borderBottom": "1px solid #dee2e6", "color": PRIMARY, "fontWeight": "600"}),
            html.Td(
                html.Div(style={"width": "100px", "height": "6px", "background": "#e9ecef", "borderRadius": "3px"}, children=[
                    html.Div(style={"width": f"{min(count * 10, 100)}%", "height": "100%", "background": PRIMARY, "borderRadius": "3px"})
                ]),
                style={"padding": "0.5rem 1rem", "borderBottom": "1px solid #dee2e6"},
            ),
        ])
        for mod, count in modalities.items()
    ]
    return html.Table([
        html.Thead(html.Tr([
            html.Th("模态", style={"textAlign": "left", "padding": "0.5rem 1rem", "color": "#6c757d", "fontSize": "0.8rem"}),
            html.Th("文件数", style={"textAlign": "left", "padding": "0.5rem 1rem", "color": "#6c757d", "fontSize": "0.8rem"}),
            html.Th("覆盖度", style={"textAlign": "left", "padding": "0.5rem 1rem", "color": "#6c757d", "fontSize": "0.8rem"}),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})


def _scan_project(project_dir: Path) -> dict[str, Any]:
    """Scan project directory for quick stats."""
    import os

    stats = {"subjects": 0, "modalities": 0, "data_files": 0, "figures": 0, "completeness": 0, "modality_details": {}}
    modality_keywords = {"face": ["face", "emotion"], "eeg": ["eeg", "erp"], "eye": ["eye", "gaze", "fixation"], "physio": ["ecg", "eda", "gsr"], "fnirs": ["fnirs", "nirs"]}
    subjects: set[str] = set()
    mod_counts: dict[str, int] = {m: 0 for m in modality_keywords}

    data_exts = {".csv", ".tsv", ".h5", ".hdf5", ".pkl", ".parquet"}
    img_exts = {".png", ".jpg", ".jpeg", ".gif", ".svg"}

    for root, _dirs, files in os.walk(project_dir):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in data_exts:
                stats["data_files"] += 1
                fl = f.lower()
                for mod, keywords in modality_keywords.items():
                    if any(kw in fl for kw in keywords):
                        mod_counts[mod] += 1
                for part in Path(f).parts:
                    if part.lower().startswith(("sub", "subj", "subject")):
                        subjects.add(part)
            elif ext in img_exts:
                stats["figures"] += 1

    stats["subjects"] = len(subjects)
    active_mods = {m: c for m, c in mod_counts.items() if c > 0}
    stats["modalities"] = len(active_mods)
    stats["modality_details"] = active_mods
    expected_mods = len(modality_keywords) * stats["subjects"] if stats["subjects"] > 0 else 1
    stats["completeness"] = min(100, int(sum(mod_counts.values()) / max(expected_mods, 1) * 100))
    return stats
