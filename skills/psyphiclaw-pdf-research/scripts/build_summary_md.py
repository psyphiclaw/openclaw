#!/usr/bin/env python3
"""Build Markdown summaries from extracted paper JSON.

Generates per-paper and batch summary files with structured sections.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Markdown summaries from extracted papers.")
    parser.add_argument("--input", "-i", required=True, help="Extracted papers JSON")
    parser.add_argument("--output", "-o", default="summaries", help="Output directory or file")
    parser.add_argument("--batch", action="store_true", help="Also generate batch summary")
    parser.add_argument("--max-abstract", type=int, default=500, help="Max abstract chars")
    parser.add_argument("--max-methods", type=int, default=300, help="Max methods chars")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text to max characters at word boundary.

    Args:
        text: Input text.
        max_chars: Maximum characters.

    Returns:
        Truncated text.
    """
    if not text or len(text) <= max_chars:
        return text or "*N/A*"
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def paper_to_markdown(paper: dict, max_abstract: int = 500, max_methods: int = 300) -> str:
    """Convert a single paper dict to Markdown.

    Args:
        paper: Extracted paper dict.
        max_abstract: Max abstract length.
        max_methods: Max methods length.

    Returns:
        Markdown string.
    """
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", "Unknown")
    journal = paper.get("journal", "")
    year = paper.get("year", "")
    abstract = truncate(paper.get("abstract", ""), max_abstract)
    methods = truncate(paper.get("methods", ""), max_methods)
    results = truncate(paper.get("results", ""), 400)
    discussion = truncate(paper.get("discussion", ""), 400)
    n_pages = paper.get("n_pages", "?")

    lines = [
        f"# {title}",
        "",
        f"**Authors:** {authors}",
    ]
    if journal:
        lines.append(f"**Journal:** {journal}")
    if year:
        lines.append(f"**Year:** {year}")
    lines.extend([
        f"**Pages:** {n_pages}",
        f"**Source:** `{paper.get('file', '')}`",
        "",
        "## Abstract",
        "",
        abstract,
        "",
        "## Methods",
        "",
        methods,
    ])

    if results:
        lines.extend(["", "## Results", "", results])
    if discussion:
        lines.extend(["", "## Discussion", "", discussion])

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def batch_summary(papers: list[dict]) -> str:
    """Generate a batch summary with all papers in a table.

    Args:
        papers: List of extracted paper dicts.

    Returns:
        Batch summary Markdown string.
    """
    lines = [
        "# 📄 Paper Batch Summary",
        "",
        f"**Total:** {len(papers)} papers",
        "",
        "| # | Title | Authors | Year | Pages |",
        "|---|-------|---------|------|-------|",
    ]

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "N/A")[:60]
        authors = paper.get("authors", "N/A")[:30]
        year = paper.get("year", "")
        pages = paper.get("n_pages", "?")
        lines.append(f"| {i} | {title} | {authors} | {year} | {pages} |")

    lines.extend(["", "---", ""])

    # Per-paper summaries
    for i, paper in enumerate(papers, 1):
        lines.extend([
            f"## {i}. {paper.get('title', 'N/A')}",
            "",
            f"**Authors:** {paper.get('authors', 'N/A')} | "
            f"**Year:** {paper.get('year', '')} | "
            f"**Pages:** {paper.get('n_pages', '?')}",
            "",
            f"> {truncate(paper.get('abstract', ''), 300)}",
            "",
        ])

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

COLOR_PRIMARY = "#4A90D9"
COLOR_ALERT = "#E74C3C"


def main() -> int:
    args = parse_args()

    if not Path(args.input).exists():
        print(f"[{COLOR_ALERT}] ERROR: {args.input} not found", file=sys.stderr)
        return 1

    with open(args.input) as f:
        papers = json.load(f)

    if not isinstance(papers, list):
        print(f"[{COLOR_ALERT}] ERROR: Expected list of papers in JSON.", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    if not out_path.suffix:
        out_path.mkdir(parents=True, exist_ok=True)

        # Per-paper summaries
        for i, paper in enumerate(papers):
            safe_name = re.sub(r'[^\w\s-]', '', paper.get("title", f"paper_{i}"))[:50]
            md = paper_to_markdown(paper, args.max_abstract, args.max_methods)
            with open(out_path / f"{i+1:02d}_{safe_name}.md", "w") as f:
                f.write(md)

        # Batch summary
        if args.batch:
            batch_md = batch_summary(papers)
            with open(out_path / "_batch_summary.md", "w") as f:
                f.write(batch_md)

        print(f"[{COLOR_PRIMARY}] ✓ {len(papers)} summaries → {out_path}/")
    else:
        # Single file output
        md = batch_summary(papers)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(md)
        print(f"[{COLOR_PRIMARY}] ✓ Batch summary → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
