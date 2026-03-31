#!/usr/bin/env python3
"""Render a PsyPhiClaw analysis report from manifest.json.

Reads manifest.json, fills a Jinja2 HTML template with data URIs for images,
and outputs both report.html and report.md.

Usage:
    python render_report.py --manifest manifest.json --lang cn -o report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

try:
    from jinja2 import Environment, FileSystemLoader, BaseLoader
except ImportError:
    print("Error: jinja2 is required. Install with: pip install jinja2", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Image → data URI
# ---------------------------------------------------------------------------
def image_to_data_uri(image_path: str, base_dir: Path) -> str:
    """Convert a local image file to a base64 data URI."""
    full_path = base_dir / image_path
    if not full_path.is_file():
        return f"[MISSING: {image_path}]"
    mime, _ = mimetypes.guess_type(str(full_path))
    mime = mime or "application/octet-stream"
    data = base64.b64encode(full_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


# ---------------------------------------------------------------------------
# Markdown generation (simple)
# ---------------------------------------------------------------------------
def manifest_to_markdown(manifest: dict[str, Any]) -> str:
    """Generate a Markdown version of the report from manifest."""
    meta = manifest["meta"]
    lines: list[str] = []
    lines.append(f"# {meta['project_name']} — 多模态分析报告")
    lines.append(f"**日期**: {meta['date']}  ")
    lines.append(f"**被试数**: {meta['subjects_count']}  ")
    lines.append(f"**模态**: {', '.join(meta['modalities']) or '无'}")
    lines.append("")

    # Facts
    facts = manifest.get("facts", {})
    if facts.get("data_summary"):
        lines.append(f"## 数据摘要\n{facts['data_summary']}\n")
    if facts.get("sample_check"):
        lines.append(f"## 样本检查\n{facts['sample_check']}\n")

    # Sections
    sections = manifest.get("section_bodies", {})
    for key, sec in sections.items():
        if sec.get("body"):
            lines.append(f"## {sec['title']}\n{sec['body']}\n")

    # Figures
    figures = manifest.get("galleries", {}).get("figures", [])
    if figures:
        lines.append("## 图表\n")
        for fig in figures:
            lines.append(f"- **{fig['caption']}** ({fig['section']}): `{fig['path']}`")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def render_html(manifest: dict[str, Any], base_dir: Path, lang: str) -> str:
    """Render the manifest into an HTML report string."""
    # Convert figure paths to data URIs
    figures = manifest.get("galleries", {}).get("figures", [])
    for fig in figures:
        fig["data_uri"] = image_to_data_uri(fig["path"], base_dir)

    # Locate template
    skill_dir = Path(__file__).resolve().parent.parent
    assets_dir = skill_dir / "assets"

    template_name = f"report_template_{lang}.html"
    template_path = assets_dir / template_name

    if template_path.is_file():
        env = Environment(loader=FileSystemLoader(str(assets_dir)), autoescape=False)
        template = env.get_template(template_name)
    else:
        # Fallback: built-in minimal template
        template = BaseLoader().from_string(DEFAULT_TEMPLATE)

    return template.render(
        meta=manifest["meta"],
        facts=manifest.get("facts", {}),
        figures=figures,
        sections=manifest.get("section_bodies", {}),
    )


DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang="{{ lang }}">
<head>
<meta charset="UTF-8">
<title>{{ meta.project_name }} — Analysis Report</title>
<style>
  :root { --primary: #4A90D9; --danger: #E74C3C; --bg: #fff; --text: #222; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem; color: var(--text); background: var(--bg); }
  h1 { color: var(--primary); border-bottom: 3px solid var(--primary); padding-bottom: 0.5rem; }
  h2 { color: var(--primary); margin-top: 2rem; }
  .meta { background: #f5f7fa; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
  .figure { margin: 1.5rem 0; text-align: center; }
  .figure img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
  .figure .caption { font-style: italic; color: #666; margin-top: 0.5rem; }
  .section { margin-bottom: 1.5rem; line-height: 1.7; }
  .section .body { white-space: pre-wrap; }
  @media print { body { padding: 0; } .no-print { display: none; } }
  nav.toc { background: #f5f7fa; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
  nav.toc ul { list-style: none; padding: 0; }
  nav.toc li { margin: 0.25rem 0; }
  nav.toc a { color: var(--primary); text-decoration: none; }
  nav.toc a:hover { text-decoration: underline; }
</style>
</head>
<body>
<h1>{{ meta.project_name }} — Analysis Report</h1>
<div class="meta">
  <strong>Date:</strong> {{ meta.date }} &nbsp;|&nbsp;
  <strong>Subjects:</strong> {{ meta.subjects_count }} &nbsp;|&nbsp;
  <strong>Modalities:</strong> {{ meta.modalities | join(', ') or 'None' }}
</div>

<nav class="toc">
  <strong>Contents</strong>
  <ul>
  {% for key, sec in sections.items() %}
    <li><a href="#sec-{{ key }}">{{ sec.title }}</a></li>
  {% endfor %}
  </ul>
</nav>

{% for key, sec in sections.items() %}
<div class="section" id="sec-{{ key }}">
  <h2>{{ sec.title }}</h2>
  <div class="body">{{ sec.body }}</div>
</div>
{% endfor %}

{% if figures %}
<h2>Figures</h2>
{% for fig in figures %}
<div class="figure">
  {% if fig.data_uri != fig.data_uri[:20] + '[MISSING' %}
  <img src="{{ fig.data_uri }}" alt="{{ fig.caption }}">
  {% endif %}
  <div class="caption">{{ fig.caption }} ({{ fig.section }})</div>
</div>
{% endfor %}
{% endif %}

<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #ddd; color:#999; font-size:0.85rem;">
  Generated by PsyPhiClaw Analysis Report &bull; {{ meta.date }}
</div>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Render HTML report from manifest.json")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json")
    parser.add_argument("--lang", choices=["cn", "en"], default="cn", help="Template language (default: cn)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output HTML path (default: report.html)")
    parser.add_argument("--md", type=Path, default=None, help="Also output Markdown path")
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"Error: {args.manifest} not found", file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base_dir = args.manifest.parent

    # Render HTML
    html = render_html(manifest, base_dir, args.lang)
    out = args.output or base_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"HTML report: {out}")

    # Render Markdown
    md = manifest_to_markdown(manifest)
    md_out = args.md or base_dir / "report.md"
    md_out.write_text(md, encoding="utf-8")
    print(f"Markdown report: {md_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
