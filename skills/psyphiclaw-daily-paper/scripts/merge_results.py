#!/usr/bin/env python3
"""Merge and deduplicate paper search results from multiple sources.

Deduplicates by DOI or title similarity, then sorts by relevance + recency.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and deduplicate paper results.")
    parser.add_argument("inputs", nargs="+", help="Input JSON files to merge")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--title-threshold", type=float, default=0.85,
                        help="Title similarity threshold for dedup (0-1)")
    parser.add_argument("--max", "-n", type=int, default=50, help="Max papers in output")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def normalize_title(title: str) -> str:
    """Normalize paper title for comparison.

    Args:
        title: Raw paper title.

    Returns:
        Normalized lowercase title with special chars removed.
    """
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def title_similarity(a: str, b: str) -> float:
    """Compute similarity between two normalized titles.

    Args:
        a: First title.
        b: Second title.

    Returns:
        Similarity score 0-1.
    """
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def merge_results(
    input_files: list[str],
    title_threshold: float = 0.85,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Merge papers from multiple source files.

    Deduplicates by DOI first, then by title similarity.

    Args:
        input_files: List of JSON file paths.
        title_threshold: Title similarity threshold for merging.
        verbose: Print progress.

    Returns:
        Deduplicated and merged paper list.
    """
    all_papers: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)

    for fpath in input_files:
        with open(fpath) as f:
            papers = json.load(f)
        all_papers.extend(papers)
        for p in papers:
            source_counts[p.get("source", "unknown")] += 1

    if verbose:
        print(f"[#4A90D9] Loaded {len(all_papers)} papers from {len(input_files)} files")
        for src, count in source_counts.items():
            print(f"  {src}: {count}")

    # ── Dedup Step 1: DOI ──
    doi_map: dict[str, dict[str, Any]] = {}
    no_doi: list[dict[str, Any]] = []

    for paper in all_papers:
        doi = paper.get("doi", "").strip().lower()
        if doi:
            if doi not in doi_map:
                doi_map[doi] = paper
            else:
                # Merge: keep the one with more metadata
                existing = doi_map[doi]
                if len(paper.get("abstract", "")) > len(existing.get("abstract", "")):
                    existing["abstract"] = paper["abstract"]
                if not existing.get("arxiv_id") and paper.get("arxiv_id"):
                    existing["arxiv_id"] = paper["arxiv_id"]
                sources = existing.get("sources", [existing.get("source")])
                if paper.get("source") not in sources:
                    sources.append(paper.get("source"))
                    existing["sources"] = sources
        else:
            no_doi.append(paper)

    if verbose:
        print(f"[#4A90D9] After DOI dedup: {len(doi_map)} unique (DOI), {len(no_doi)} no-DOI")

    # ── Dedup Step 2: Title similarity ──
    deduped_no_doi: list[dict[str, Any]] = []
    seen_titles: list[str] = []

    for paper in no_doi:
        title = paper.get("title", "")
        is_dup = False

        for seen_title in seen_titles:
            if title_similarity(title, seen_title) >= title_threshold:
                is_dup = True
                # Merge into the matching paper in deduped_no_doi
                for existing in deduped_no_doi:
                    if title_similarity(existing["title"], title) >= title_threshold:
                        if len(paper.get("abstract", "")) > len(existing.get("abstract", "")):
                            existing["abstract"] = paper["abstract"]
                        break
                break

        if not is_dup:
            deduped_no_doi.append(paper)
            seen_titles.append(title)

    if verbose:
        print(f"[#4A90D9] After title dedup: {len(deduped_no_doi)} unique (no-DOI)")

    # Combine
    merged = list(doi_map.values()) + deduped_no_doi

    # ── Sort by recency + relevance ──
    def sort_key(p: dict) -> tuple:
        # Parse date
        pub_date = p.get("pub_date", "")
        try:
            dt = datetime.strptime(pub_date[:10], "%Y-%m-%d")
        except (ValueError, IndexError):
            try:
                dt = datetime.strptime(pub_date[:10], "%Y/%m/%d")
            except (ValueError, IndexError):
                dt = datetime(2000, 1, 1)

        # Citation boost
        citations = p.get("citation_count", 0)
        recency_score = dt.timestamp()
        return (recency_score, citations)

    merged.sort(key=sort_key, reverse=True)

    return merged


def main() -> int:
    args = parse_args()
    merged = merge_results(args.inputs, args.title_threshold, args.verbose)

    if args.max:
        merged = merged[:args.max]

    print(f"[#4A90D9] ✓ Merged: {len(merged)} unique papers")

    out = args.output or "merged_results.json"
    with open(out, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[#4A90D9] Saved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
