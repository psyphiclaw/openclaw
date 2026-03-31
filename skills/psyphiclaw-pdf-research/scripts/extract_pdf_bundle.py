#!/usr/bin/env python3
"""Batch extract text from PDF papers with structure recognition.

Extracts title, authors, abstract, methods, results, discussion, references.
Outputs structured JSON per paper.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text from PDF papers.")
    parser.add_argument("--input", "-i", required=True, help="PDF file or directory")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages per PDF (0=all)")
    parser.add_argument("--extract-tables", action="store_true", help="Extract tables via pdfplumber")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


# ── Section Detection ─────────────────────────────────────────────────────────

SECTION_PATTERNS: dict[str, list[str]] = {
    "title": [
        r"(?i)^(?!abstract|introduction|method|result|discussion|reference|acknowledgment|conclusion).{10,200}$",
    ],
    "abstract": [r"(?i)\babstract\b", r"(?i)\bsummary\b"],
    "introduction": [r"(?i)\bintroduction\b", r"(?i)\bbackground\b"],
    "methods": [r"(?i)\b(?:method|methodology|materials?\s+and\s+methods?)\b",
                r"(?i)\b(?:experimental\s+design|procedure|participants?)\b"],
    "results": [r"(?i)\b(?:result|finding|analysis)\b",
                r"(?i)\b(?:experiment|study)\s*\d*\b"],
    "discussion": [r"(?i)\b(?:discussion|interpretation|implication)\b"],
    "conclusion": [r"(?i)\b(?:conclusion|summary|final|closing)\b"],
    "references": [r"(?i)\b(?:reference|bibliography|citation)\b"],
    "acknowledgments": [r"(?i)\b(?:acknowledgment|acknowledgement|funding|conflict)\b"],
}


def detect_sections(text_lines: list[str]) -> dict[str, tuple[int, int]]:
    """Detect paper sections by matching header patterns.

    Args:
        text_lines: List of text lines from PDF.

    Returns:
        Dict mapping section names to (start_line, end_line) tuples.
    """
    sections: dict[str, int] = {}
    all_markers: list[tuple[int, str]] = []

    for i, line in enumerate(text_lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 200:
            continue

        for section_name, patterns in SECTION_PATTERNS.items():
            if section_name == "title":
                continue  # Title handled separately
            for pattern in patterns:
                if re.match(pattern, stripped):
                    all_markers.append((i, section_name))
                    break

    # Sort by line number
    all_markers.sort(key=lambda x: x[0])

    # Build sections
    section_ranges: dict[str, tuple[int, int]] = {}
    for idx, (line_num, section_name) in enumerate(all_markers):
        start = line_num
        end = all_markers[idx + 1][0] if idx + 1 < len(all_markers) else len(text_lines)
        # Skip first section if it's too close to start (likely header)
        section_ranges[section_name] = (start + 1, end)

    return section_ranges


def extract_title(text_lines: list[str]) -> str:
    """Extract paper title from first few non-empty lines.

    The title is typically the first substantial line (20-200 chars)
    that isn't a section header or author line.

    Args:
        text_lines: Text lines from PDF.

    Returns:
        Extracted title string.
    """
    section_headers = {"abstract", "introduction", "methods", "results",
                       "discussion", "conclusion", "references", "acknowledgment"}

    for line in text_lines[:20]:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip very short lines (page numbers, etc.)
        if len(stripped) < 15:
            continue
        # Skip lines that look like section headers
        if stripped.lower().replace(" ", "") in {h.replace(" ", "") for h in section_headers}:
            continue
        # Skip lines that look like author lists (many commas, &)
        if stripped.count(",") > 3 and stripped.count(" ") < stripped.count(",") * 3:
            continue
        # Skip lines with numbers only or very short words
        words = stripped.split()
        if len(words) >= 5 and not all(len(w) <= 3 for w in words):
            return stripped

    return "Unknown Title"


def extract_authors(text_lines: list[str], title: str) -> str:
    """Extract authors from lines after the title.

    Args:
        text_lines: Text lines.
        title: Extracted title.

    Returns:
        Author string.
    """
    title_idx = -1
    for i, line in enumerate(text_lines):
        if line.strip() == title:
            title_idx = i
            break

    if title_idx < 0:
        title_idx = 0

    # Look in next 10 lines for author pattern
    for i in range(title_idx + 1, min(title_idx + 15, len(text_lines))):
        line = text_lines[i].strip()
        if not line:
            continue
        # Author lines often have commas, ampersands, *, or @
        if (line.count(",") >= 1 and len(line) < 300) or "@" in line or "*" in line:
            return line

    return "Unknown Authors"


def extract_journal_year(text_lines: list[str]) -> dict[str, str]:
    """Extract journal name and year from text.

    Args:
        text_lines: Text lines.

    Returns:
        Dict with journal and year.
    """
    result = {"journal": "", "year": ""}

    # Look for year pattern (4 digits in 1900-2099 range)
    for line in text_lines[:30]:
        year_match = re.search(r"\b(19|20)\d{2}\b", line)
        if year_match:
            result["year"] = year_match.group(0)
            break

    # Look for common journal indicators
    for line in text_lines[:20]:
        for keyword in ["Journal of", "Nature", "Science", "PNAS", "IEEE", "Frontiers",
                        "Psychophysiology", "NeuroImage", "Biological Psychology"]:
            if keyword in line:
                result["journal"] = line.strip()
                break

    return result


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_pdf_text(filepath: str, max_pages: int = 0, extract_tables: bool = False) -> dict[str, Any]:
    """Extract text and structure from a single PDF.

    Args:
        filepath: Path to PDF file.
        max_pages: Max pages to process (0=all).
        extract_tables: Whether to extract tables.

    Returns:
        Structured paper dict.
    """
    text_lines: list[str] = []
    tables: list[list[str]] = []
    n_pages = 0

    try:
        import pdfplumber

        with pdfplumber.open(filepath) as pdf:
            pages_to_process = len(pdf.pages) if max_pages == 0 else min(max_pages, len(pdf.pages))

            for page_num in range(pages_to_process):
                page = pdf.pages[page_num]
                n_pages += 1

                # Extract text
                page_text = page.extract_text() or ""
                if page_text:
                    text_lines.extend(page_text.split("\n"))

                # Extract tables
                if extract_tables:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            tables.append([str(cell) if cell else "" for row in table for cell in row])

    except ImportError:
        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(filepath)
            pages_to_process = len(reader.pages) if max_pages == 0 else min(max_pages, len(reader.pages))

            for page_num in range(pages_to_process):
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""
                if page_text:
                    text_lines.extend(page_text.split("\n"))
                n_pages += 1

        except ImportError:
            return {"error": "Neither pdfplumber nor PyPDF2 installed", "file": filepath}

    # Detect structure
    full_text = "\n".join(text_lines)
    title = extract_title(text_lines)
    authors = extract_authors(text_lines, title)
    meta = extract_journal_year(text_lines)
    section_ranges = detect_sections(text_lines)

    # Extract sections
    sections_content: dict[str, str] = {}
    for section_name, (start, end) in section_ranges.items():
        section_text = "\n".join(text_lines[start:end]).strip()
        if section_text:
            sections_content[section_name] = section_text

    return {
        "file": str(filepath),
        "title": title,
        "authors": authors,
        "journal": meta["journal"],
        "year": meta["year"],
        "n_pages": n_pages,
        "n_lines": len(text_lines),
        "full_text": full_text,
        "sections": sections_content,
        "tables": tables if tables else None,
        "abstract": sections_content.get("abstract", ""),
        "methods": sections_content.get("methods", ""),
        "results": sections_content.get("results", ""),
        "discussion": sections_content.get("discussion", ""),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

COLOR_PRIMARY = "#4A90D9"
COLOR_ALERT = "#E74C3C"


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[{COLOR_ALERT}] ERROR: {args.input} not found", file=sys.stderr)
        return 1

    # Collect PDF files
    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("**/*.pdf"))
    else:
        pdf_files = [input_path]

    if not pdf_files:
        print(f"[{COLOR_ALERT}] No PDF files found.", file=sys.stderr)
        return 1

    print(f"[{COLOR_PRIMARY}] Processing {len(pdf_files)} PDF files...")

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for pdf_path in pdf_files:
        if args.verbose:
            print(f"  [{COLOR_PRIMARY}] {pdf_path.name}...")

        try:
            paper = extract_pdf_text(str(pdf_path), args.max_pages, args.extract_tables)
            if "error" in paper:
                errors.append(f"{pdf_path.name}: {paper['error']}")
                continue
            results.append(paper)
            if args.verbose:
                print(f"    ✓ {paper['title'][:60]}... ({paper['n_pages']} pages)")
        except Exception as e:
            errors.append(f"{pdf_path.name}: {e}")
            if args.verbose:
                print(f"    [{COLOR_ALERT}] ERROR: {e}")

    print(f"[{COLOR_PRIMARY}] ✓ Extracted {len(results)}/{len(pdf_files)} papers")
    if errors:
        print(f"[{COLOR_ALERT}] {len(errors)} errors:")
        for err in errors:
            print(f"  • {err}")

    out = args.output or "extracted_papers.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[{COLOR_PRIMARY}] Saved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
