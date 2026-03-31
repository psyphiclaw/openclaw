"""Subjects management page for PsyPhiClaw Dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dash import html, dcc

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

PRIMARY = "#4A90D9"
DANGER = "#E74C3C"
SUCCESS = "#27ae60"
CARD_BG = "#f8f9fa"


def create_layout(project_dir: Path) -> html.Div:
    """Create the subjects management page layout."""
    subjects = _scan_subjects(project_dir)

    return html.Div([
        html.H2("👥 被试管理", style={"color": PRIMARY, "marginBottom": "1.5rem"}),

        # Summary bar
        html.Div([
            html.Div([
                html.Span("总被试数", style={"color": "#6c757d", "fontSize": "0.8rem"}),
                html.Span(str(len(subjects)), style={"fontSize": "1.5rem", "fontWeight": "700", "color": PRIMARY, "display": "block"}),
            ], style={"padding": "0.75rem", "background": CARD_BG, "borderRadius": "8px", "minWidth": "100px", "textAlign": "center"}),
            html.Div([
                html.Span("完整数据", style={"color": "#6c757d", "fontSize": "0.8rem"}),
                html.Span(str(sum(1 for s in subjects if s["completeness"] >= 80)), style={"fontSize": "1.5rem", "fontWeight": "700", "color": SUCCESS, "display": "block"}),
            ], style={"padding": "0.75rem", "background": CARD_BG, "borderRadius": "8px", "minWidth": "100px", "textAlign": "center", "marginLeft": "1rem"}),
            html.Div([
                html.Span("需关注", style={"color": "#6c757d", "fontSize": "0.8rem"}),
                html.Span(str(sum(1 for s in subjects if s["completeness"] < 80)), style={"fontSize": "1.5rem", "fontWeight": "700", "color": DANGER, "display": "block"}),
            ], style={"padding": "0.75rem", "background": CARD_BG, "borderRadius": "8px", "minWidth": "100px", "textAlign": "center", "marginLeft": "1rem"}),
        ], style={"display": "flex", "marginBottom": "1.5rem"}),

        # Subject table
        html.Div([
            _subject_table(subjects),
        ], style={"background": CARD_BG, "borderRadius": "8px", "overflow": "hidden"}),

        # Comparison section
        html.Div([
            html.H3("被试间对比", style={"color": "#16213e", "marginBottom": "1rem", "marginTop": "2rem"}),
            html.P("选择两个或多个被试进行对比分析。", style={"color": "#6c757d", "fontSize": "0.85rem"}),
            dcc.Dropdown(
                id="compare-subjects",
                options=[{"label": s["id"], "value": s["id"]} for s in subjects],
                multi=True,
                placeholder="选择被试...",
                style={"marginBottom": "1rem", "width": "100%", "maxWidth": "400px"},
            ),
            html.Div(id="comparison-result", style={"background": CARD_BG, "padding": "1rem", "borderRadius": "8px", "minHeight": "200px"}, children=[
                html.P("选择被试后将显示对比分析结果。", style={"color": "#999"}),
            ]),
        ]),
    ])


def _subject_table(subjects: list[dict[str, Any]]) -> html.Table:
    """Create a subject data table."""
    if not subjects:
        return html.Div([html.P("未检测到被试数据。", style={"padding": "2rem", "textAlign": "center", "color": "#999"})])

    status_color = lambda c: SUCCESS if c >= 80 else ("#f39c12" if c >= 50 else DANGER)

    header = html.Tr([
        html.Th("被试ID", style=_th_style()),
        html.Th("模态覆盖", style=_th_style()),
        html.Th("数据文件", style=_th_style()),
        html.Th("完整性", style=_th_style()),
        html.Th("状态", style=_th_style()),
    ])

    rows = []
    for s in subjects[:100]:
        comp = s["completeness"]
        status = "✅ 完整" if comp >= 80 else ("⚠️ 部分" if comp >= 50 else "❌ 缺失")
        rows.append(html.Tr([
            html.Td(s["id"], style=_td_style()),
            html.Td(", ".join(s["modalities"]) or "—", style=_td_style()),
            html.Td(str(s["file_count"]), style=_td_style()),
            html.Td([
                html.Div([
                    html.Div(style={"width": f"{comp}%", "height": "100%", "background": status_color(comp), "borderRadius": "3px"}),
                ], style={"width": "80px", "height": "8px", "background": "#e9ecef", "borderRadius": "3px", "display": "inline-block", "marginRight": "0.5rem", "verticalAlign": "middle"}),
                html.Span(f"{comp}%", style={"fontSize": "0.85rem", "color": status_color(comp)}),
            ], style=_td_style()),
            html.Td(status, style=_td_style()),
        ]))

    return html.Table([html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse"})


def _th_style() -> dict[str, str]:
    return {"textAlign": "left", "padding": "0.75rem 1rem", "color": "#6c757d", "fontSize": "0.8rem", "borderBottom": "2px solid #dee2e6", "textTransform": "uppercase", "letterSpacing": "0.5px"}


def _td_style() -> dict[str, str]:
    return {"padding": "0.6rem 1rem", "borderBottom": "1px solid #eee", "fontSize": "0.9rem"}


def _scan_subjects(project_dir: Path) -> list[dict[str, Any]]:
    """Scan project directory for subject-level data."""
    data_exts = {".csv", ".tsv", ".h5", ".hdf5", ".pkl", ".parquet", ".npz"}
    modality_keywords = {"face": ["face", "emotion"], "eeg": ["eeg", "erp"], "eye": ["eye", "gaze", "fixation"], "physio": ["ecg", "eda", "gsr"], "fnirs": ["fnirs", "nirs"]}
    expected_modalities = {"face", "eeg", "eye", "physio", "fnirs"}
    subjects_map: dict[str, dict[str, Any]] = {}

    for root, _dirs, files in os.walk(project_dir):
        for f in files:
            if Path(f).suffix.lower() not in data_exts:
                continue
            # Find subject ID
            sid = "unknown"
            parts = Path(root).parts + (f,)
            for p in parts:
                pl = p.lower()
                if pl.startswith(("sub-", "sub_", "subj", "subject", "p")):
                    sid = p
                    break

            if sid not in subjects_map:
                subjects_map[sid] = {"id": sid, "modalities": set(), "file_count": 0}

            subjects_map[sid]["file_count"] += 1
            fl = f.lower()
            for mod, keywords in modality_keywords.items():
                if any(kw in fl for kw in keywords):
                    subjects_map[sid]["modalities"].add(mod)

    result = []
    for sid, info in sorted(subjects_map.items()):
        info["modalities"] = sorted(info["modalities"])
        covered = len(info["modalities"]) / len(expected_modalities) * 100 if expected_modalities else 100
        info["completeness"] = min(100, int(covered))
        result.append(info)
    return result
