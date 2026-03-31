#!/usr/bin/env python3
"""Search arXiv for multimodal behavioral analysis papers.

Uses arXiv Atom API. Returns title, abstract, arXiv ID, publication date.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote

import requests


ARXIV_API = "http://export.arxiv.org/api/query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search arXiv for papers.")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--categories", nargs="+",
                        default=["cs.CV", "q-bio.NC", "eess.SP", "cs.HC"],
                        help="arXiv categories to search")
    parser.add_argument("--max", "-n", type=int, default=20, help="Max results")
    parser.add_argument("--days-back", type=int, default=30, help="Look back N days")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def search_arxiv(
    query: str,
    categories: list[str] | None = None,
    max_results: int = 20,
    days_back: int = 30,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Search arXiv using Atom API.

    Args:
        query: Search query string.
        categories: arXiv categories to filter.
        max_results: Maximum results.
        days_back: Days to look back.
        verbose: Print progress.

    Returns:
        List of paper dicts.
    """
    cats = categories or ["cs.CV", "q-bio.NC", "eess.SP"]
    cat_query = " OR ".join(f"cat:{c}" for c in cats)

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    full_query = f"({query}) AND ({cat_query}) AND submittedDate:[{cutoff} TO *]"

    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    if verbose:
        print(f"[#4A90D9] Searching arXiv: {query[:80]}...")

    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()

    # arXiv uses Atom namespace
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)

    papers: list[dict[str, Any]] = []

    for entry in root.findall("atom:entry", ns):
        try:
            # Title
            title_el = entry.find("atom:title", ns)
            title = (title_el.text or "").replace("\n", " ").strip()

            # Abstract
            abstract_el = entry.find("atom:summary", ns)
            abstract = (abstract_el.text or "").replace("\n", " ").strip()

            # Authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None and name.text:
                    authors.append(name.text.strip())
            author_str = ", ".join(authors[:5])
            if len(authors) > 5:
                author_str += f" et al. ({len(authors)} authors)"

            # Published date
            pub_el = entry.find("atom:published", ns)
            pub_date = pub_el.text[:10] if pub_el is not None and pub_el.text else ""

            # Updated date
            upd_el = entry.find("atom:updated", ns)
            updated = upd_el.text[:10] if upd_el is not None and upd_el.text else ""

            # arXiv ID and link
            arxiv_id = ""
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                href = link.get("href", "")
                title_attr = link.get("title", "")
                if "abs" in href and not arxiv_id:
                    arxiv_id = href.split("/abs/")[-1]
                if title_attr == "pdf":
                    pdf_url = href

            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""

            # Categories
            cats_found = []
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term", "")
                if term:
                    cats_found.append(term)

            # DOI (if available)
            doi_el = entry.find("arxiv:doi", {"arxiv": "http://arxiv.org/schemas/atom"})
            doi = doi_el.text if doi_el is not None else ""

            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": author_str,
                "journal": "arXiv",
                "pub_date": pub_date,
                "updated": updated,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "source": "arxiv",
                "url": url,
                "pdf_url": pdf_url,
                "categories": cats_found,
            })
        except Exception as e:
            if verbose:
                print(f"[#E74C3C] Parse error: {e}")
            continue

    return papers


def main() -> int:
    args = parse_args()
    papers = search_arxiv(args.query, args.categories, args.max, args.days_back, args.verbose)

    print(f"[#4A90D9] ✓ Retrieved {len(papers)} papers from arXiv")

    if args.verbose:
        for i, p in enumerate(papers[:3]):
            print(f"  {i+1}. [{p['arxiv_id']}] {p['title'][:70]}...")

    out = args.output or "arxiv_results.json"
    with open(out, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"[#4A90D9] Saved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
