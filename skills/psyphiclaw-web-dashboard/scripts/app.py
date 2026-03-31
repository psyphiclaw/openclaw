#!/usr/bin/env python3
"""PsyPhiClaw Web Dashboard — Main Application Entry Point.

Launches a Dash + Plotly interactive dashboard for multimodal behavioral data.

Usage:
    python app.py --project-dir ./data --port 8050
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dash import Dash, html, dcc, Input, Output, State
    import dash
except ImportError:
    print("Error: dash is required. Install with: pip install dash plotly pandas numpy", file=sys.stderr)
    raise SystemExit(1)


def create_app(project_dir: Path, debug: bool = False) -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        title="PsyPhiClaw Dashboard",
        external_stylesheets=[],
        suppress_callback_exceptions=True,
    )

    # Determine assets dir relative to this script
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    if assets_dir.is_dir():
        app.config.assets_folder = str(assets_dir)

    # Import page modules
    scripts_dir = Path(__file__).resolve().parent
    pages_dir = scripts_dir / "pages"
    sys.path.insert(0, str(scripts_dir))

    try:
        from pages import overview, data_browser, results, subjects
    except ImportError as e:
        print(f"Warning: Could not import all pages: {e}", file=sys.stderr)
        overview = None
        data_browser = None
        results = None
        subjects = None

    # --- Color scheme ---
    PRIMARY = "#4A90D9"
    DANGER = "#E74C3C"
    DARK_BG = "#1a1a2e"
    SIDEBAR_BG = "#16213e"
    CARD_BG = "#f8f9fa"

    # --- Layout ---
    app.layout = html.Div(
        id="app-container",
        style={"display": "flex", "minHeight": "100vh", "fontFamily": "'Segoe UI', system-ui, sans-serif"},
        children=[
            # Sidebar navigation
            html.Div(
                id="sidebar",
                style={
                    "width": "260px",
                    "minHeight": "100vh",
                    "background": f"linear-gradient(180deg, {DARK_BG} 0%, {SIDEBAR_BG} 100%)",
                    "color": "#e0e0e0",
                    "padding": "2rem 1.5rem",
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "overflowY": "auto",
                    "zIndex": "100",
                },
                children=[
                    html.Div([
                        html.H2("🧠 PsyPhiClaw", style={"color": PRIMARY, "marginBottom": "0.25rem", "fontSize": "1.4rem"}),
                        html.P("Multimodal Dashboard", style={"color": "#888", "fontSize": "0.8rem", "marginBottom": "2rem"}),
                    ]),
                    html.Nav([
                        _nav_item("overview", "📊 概览", PRIMARY),
                        _nav_item("data_browser", "📈 数据浏览", PRIMARY),
                        _nav_item("results", "📋 分析结果", PRIMARY),
                        _nav_item("subjects", "👥 被试管理", PRIMARY),
                        _nav_item("settings", "⚙️ 设置", PRIMARY),
                    ]),
                    html.Div([
                        html.P(f"项目: {project_dir.name}", style={"fontSize": "0.8rem", "color": "#666", "marginTop": "2rem"}),
                        html.P(str(project_dir), style={"fontSize": "0.7rem", "color": "#555", "wordBreak": "break-all"}),
                    ]),
                ],
            ),
            # Main content area
            html.Div(
                id="main-content",
                style={
                    "marginLeft": "260px",
                    "flex": "1",
                    "padding": "2rem 3rem",
                    "minHeight": "100vh",
                    "background": "#ffffff",
                },
                children=[
                    dcc.Store(id="project-dir", data=str(project_dir)),
                    dcc.Store(id="current-page", data="overview"),
                    html.Div(id="page-content", children=_get_placeholder("overview")),
                ],
            ),
        ],
    )

    # --- Page routing callbacks ---
    @app.callback(
        Output("page-content", "children"),
        Output("current-page", "data"),
        Input({"type": "nav-btn", "page": dash.dependencies.ALL}, "n_clicks"),
        State("current-page", "data"),
        prevent_initial_call=True,
    )
    def navigate_to_page(_n_clicks: int | list, current_page: str) -> tuple:
        """Handle navigation clicks."""
        if not dash.callback_context.triggered:
            return dash.no_update, dash.no_update
        triggered_id = dash.callback_context.triggered[0]["prop_id"]
        # Extract page name from {"type":"nav-btn","page":"overview"}.n_clicks
        try:
            page = eval(triggered_id.split(".")[0])["page"]  # noqa: S307
        except Exception:
            return dash.no_update, dash.no_update
        return _get_placeholder(page), page

    return app


def _nav_item(page: str, label: str, color: str) -> html.Button:
    """Create a sidebar navigation button."""
    return html.Button(
        label,
        id={"type": "nav-btn", "page": page},
        n_clicks=0,
        style={
            "display": "block",
            "width": "100%",
            "textAlign": "left",
            "background": "transparent",
            "border": "none",
            "color": "#b0b0b0",
            "padding": "0.6rem 0.8rem",
            "marginBottom": "0.3rem",
            "borderRadius": "6px",
            "cursor": "pointer",
            "fontSize": "0.9rem",
            "transition": "all 0.2s",
        },
    )


def _get_placeholder(page: str) -> html.Div:
    """Return a placeholder for a page."""
    return html.Div([
        html.H2(f"📄 {page.replace('_', ' ').title()}", style={"color": "#4A90D9", "marginBottom": "1rem"}),
        html.P("This page is available when project data is loaded.", style={"color": "#6c757d"}),
        html.P("Use the build_report_manifest.py script to prepare project data.", style={"color": "#6c757d", "fontSize": "0.85rem"}),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="PsyPhiClaw Web Dashboard")
    parser.add_argument("--project-dir", type=Path, required=True, help="Project root directory")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if not args.project_dir.is_dir():
        print(f"Error: {args.project_dir} is not a directory", file=sys.stderr)
        return 1

    app = create_app(args.project_dir, debug=args.debug)
    print(f"🧠 PsyPhiClaw Dashboard starting...")
    print(f"   Project: {args.project_dir}")
    print(f"   URL: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
