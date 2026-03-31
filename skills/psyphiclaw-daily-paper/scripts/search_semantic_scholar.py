#!/usr/bin/env python3
"""Search Semantic Scholar for multimodal behavioral analysis papers.

Returns citation counts, influence scores, and structured metadata.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Any

import requests


S2_API = "https://api.semanticscholar.org/graph/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Semantic Scholar.")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--max", "-n", type=int, default=20, help="Max results")
    parser.add_argument("--days-back", type=int, default=30, help="Look back N days")
    parser.add_argument("--fields", nargs="+",
                        default=["title", "abstract", "authors", "citationCount",
                                 "influentialCitationCount", "publicationDate",
                                 "externalIds", "journal", "fieldsOfStudy"],
                        help="Fields to retrieve")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--api-key", default=None, help="S2 API key")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def search_semantic_scholar(
    query: str,
    max_results: int = 20,
    days_back: int = 30,
    fields: list[str] | None = None,
    api_key: str | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar API.

    Args:
        query: Search query.
        max_results: Maximum results.
        days_back: Days to look back.
        fields: Paper fields to retrieve.
        api_key: Optional API key.
        verbose: Print progress.

    Returns:
        List of paper dicts.
    """
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    field_str = ",".join(fields or ["title", "abstract", "authors", "citationCount"])

    params: dict[str, Any] = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": field_str,
        "publicationDateRange": f"{cutoff}:{datetime.now().strftime('%Y-%m-%d')}",
    }

    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    if verbose:
        print(f"[#4A90D9] Searching Semantic Scholar: {query[:80]}...")

    resp = requests.get(f"{S2_API}/paper/search", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    papers: list[dict[str, Any]] = []

    for item in data.get("data", []):
        try:
            # Authors
            authors = item.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:5])
            if len(authors) > 5:
                author_str += f" et al. ({len(authors)} authors)"

            # External IDs
            ext_ids = item.get("externalIds", {}) or {}
            doi = ext_ids.get("DOI", "")
            pmid = ext_ids.get("PubMed", "")
            arxiv_id = ext_ids.get("ArXiv", "")

            # Build URL
            url = ""
            if doi:
                url = f"https://doi.org/{doi}"
            elif arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"
            elif pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            papers.append({
                "title": item.get("title", "").strip(),
                "abstract": (item.get("abstract") or "").strip(),
                "authors": author_str,
                "journal": (item.get("journal") or {}).get("name", ""),
                "pub_date": item.get("publicationDate", ""),
                "doi": doi,
                "pmid": pmid,
                "arxiv_id": arxiv_id,
                "citation_count": item.get("citationCount", 0),
                "influential_citation_count": item.get("influentialCitationCount", 0),
                "fields_of_study": item.get("fieldsOfStudy", []),
                "source": "semantic_scholar",
                "url": url,
            })
        except Exception as e:
            if verbose:
                print(f"[#E74C3C] Parse error: {e}")
            continue

    return papers


def main() -> int:
    args = parse_args()
    papers = search_semantic_scholar(
        args.query, args.max, args.days_back, args.fields, args.api_key, args.verbose
    )

    print(f"[#4A90D9] ✓ Retrieved {len(papers)} papers from Semantic Scholar")

    if args.verbose:
        for i, p in enumerate(papers[:3]):
            print(f"  {i+1}. {p['title'][:70]}... (citations: {p['citation_count']})")

    out = args.output or "s2_results.json"
    with open(out, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"[#4A90D9] Saved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
