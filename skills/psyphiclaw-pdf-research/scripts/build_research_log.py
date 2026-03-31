#!/usr/bin/env python3
"""Maintain a research reading log with status, notes, and ratings.

Exports as Markdown table. Supports adding notes, updating status, and filtering.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/maintain research reading log.")
    parser.add_argument("--input", "-i", required=True, help="Extracted papers JSON")
    parser.add_argument("--log", "-l", default=None, help="Existing log JSON (to merge/update)")
    parser.add_argument("--output", "-o", default=None, help="Output Markdown file")
    parser.add_argument("--add-note", nargs=2, metavar=("INDEX", "NOTE"),
                        action="append", help="Add note to paper by index (1-based)")
    parser.add_argument("--set-status", nargs=2, metavar=("INDEX", "STATUS"),
                        action="append", help="Set status: unread/reading/done/skipped")
    parser.add_argument("--set-rating", nargs=2, metavar=("INDEX", "RATING"),
                        action="append", help="Set rating 1-5")
    parser.add_argument("--tag", nargs=2, metavar=("INDEX", "TAG"),
                        action="append", help="Add tag to paper")
    parser.add_argument("--filter", choices=["unread", "reading", "done", "all"], default="all")
    return parser.parse_args()


STATUS_LABELS = {
    "unread": "📖 未读",
    "reading": "📖 在读",
    "done": "✅ 已读",
    "skipped": "⏭️ 跳过",
}


def load_log(filepath: str) -> dict[str, Any]:
    """Load existing research log.

    Args:
        filepath: Path to log JSON.

    Returns:
        Log dict.
    """
    if Path(filepath).exists():
        with open(filepath) as f:
            return json.load(f)
    return {"entries": {}, "last_updated": None}


def build_log_entries(papers: list[dict], existing_log: dict) -> dict[str, dict[str, Any]]:
    """Build log entries from papers, preserving existing data.

    Args:
        papers: List of extracted paper dicts.
        existing_log: Existing log with entries.

    Returns:
        Dict mapping paper index to entry dict.
    """
    entries: dict[str, dict[str, Any]] = {}
    for i, paper in enumerate(papers):
        key = str(i)
        existing = existing_log.get("entries", {}).get(key, {})
        entries[key] = {
            "title": paper.get("title", "Unknown"),
            "authors": paper.get("authors", ""),
            "year": paper.get("year", ""),
            "journal": paper.get("journal", ""),
            "file": paper.get("file", ""),
            "status": existing.get("status", "unread"),
            "rating": existing.get("rating", None),
            "notes": existing.get("notes", []),
            "tags": existing.get("tags", []),
            "date_added": existing.get("date_added", datetime.now().isoformat()),
            "date_read": existing.get("date_read", None),
        }
    return entries


def apply_updates(
    entries: dict[str, dict[str, Any]],
    add_notes: list[tuple[str, str]] | None = None,
    set_statuses: list[tuple[str, str]] | None = None,
    set_ratings: list[tuple[str, str]] | None = None,
    add_tags: list[tuple[str, str]] | None = None,
) -> None:
    """Apply command-line updates to log entries.

    Args:
        entries: Log entries dict.
        add_notes: (index, note) tuples.
        set_statuses: (index, status) tuples.
        set_ratings: (index, rating) tuples.
        add_tags: (index, tag) tuples.
    """
    for idx, note in (add_notes or []):
        key = str(int(idx) - 1)  # Convert to 0-based
        if key in entries:
            entries[key]["notes"].append(f"[{datetime.now().strftime('%m-%d')}] {note}")

    for idx, status in (set_statuses or []):
        key = str(int(idx) - 1)
        if key in entries and status in STATUS_LABELS:
            entries[key]["status"] = status
            if status == "done":
                entries[key]["date_read"] = datetime.now().isoformat()

    for idx, rating in (set_ratings or []):
        key = str(int(idx) - 1)
        if key in entries:
            r = int(rating)
            if 1 <= r <= 5:
                entries[key]["rating"] = r

    for idx, tag in (add_tags or []):
        key = str(int(idx) - 1)
        if key in entries and tag not in entries[key]["tags"]:
            entries[key]["tags"].append(tag)


def log_to_markdown(entries: dict[str, dict[str, Any]], filter_status: str = "all") -> str:
    """Convert log entries to Markdown table.

    Args:
        entries: Log entries dict.
        filter_status: Status to filter by.

    Returns:
        Markdown string.
    """
    lines = [
        "# 📄 Research Reading Log",
        "",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| # | Status | ⭐ | Title | Authors | Year | Tags |",
        "|---|--------|-----|-------|---------|------|------|",
    ]

    for key, entry in sorted(entries.items(), key=lambda x: x[0]):
        if filter_status != "all" and entry["status"] != filter_status:
            continue

        idx = str(int(key) + 1)  # 1-based display
        status = STATUS_LABELS.get(entry["status"], entry["status"])
        rating = entry.get("rating")
        rating_str = "⭐" * rating if rating else "-"
        title = entry["title"][:50]
        authors = entry.get("authors", "")[:25]
        year = entry.get("year", "")
        tags = ", ".join(f"`{t}`" for t in entry.get("tags", []))

        lines.append(f"| {idx} | {status} | {rating_str} | {title} | {authors} | {year} | {tags} |")

    # Detailed notes section
    lines.extend(["", "## 📝 Notes", ""])
    for key, entry in sorted(entries.items(), key=lambda x: x[0]):
        notes = entry.get("notes", [])
        if notes:
            lines.extend([
                f"### {int(key)+1}. {entry['title'][:60]}",
                "",
            ])
            for note in notes:
                lines.append(f"- {note}")
            lines.append("")

    lines.extend(["---", f"*PsyPhiClaw PDF Research · {datetime.now().strftime('%Y-%m-%d')}*"])
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

    # Load existing log
    log = load_log(args.log) if args.log else {"entries": {}}

    # Build entries
    entries = build_log_entries(papers, log)

    # Apply updates
    apply_updates(entries, args.add_note, args.set_status, args.set_rating, args.tag)

    # Save log JSON
    log_data = {
        "entries": entries,
        "last_updated": datetime.now().isoformat(),
    }
    log_path = args.log or "research_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # Generate Markdown
    md = log_to_markdown(entries, args.filter)
    out_path = args.output or "reading_log.md"
    with open(out_path, "w") as f:
        f.write(md)

    n_total = len(entries)
    n_done = sum(1 for e in entries.values() if e["status"] == "done")
    n_unread = sum(1 for e in entries.values() if e["status"] == "unread")

    print(f"[{COLOR_PRIMARY}] ✓ Reading log: {n_total} papers ({n_done} read, {n_unread} unread)")
    print(f"[{COLOR_PRIMARY}] JSON: {log_path}")
    print(f"[{COLOR_PRIMARY}] Markdown: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
