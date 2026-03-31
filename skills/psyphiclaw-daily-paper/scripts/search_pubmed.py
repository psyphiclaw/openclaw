#!/usr/bin/env python3
"""Search PubMed for multimodal behavioral analysis papers.

Uses NCBI E-utilities API. Returns title, abstract, DOI, publication date.

Color scheme: #4A90D9 (primary), #E74C3C (alert).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote

import requests


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search PubMed for papers.")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--max", "-n", type=int, default=20, help="Max results")
    parser.add_argument("--days-back", type=int, default=30, help="Look back N days")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--api-key", default=None, help="NCBI API key (higher rate limit)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def search_pubmed(
    query: str,
    max_results: int = 20,
    days_back: int = 30,
    api_key: str | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Search PubMed using E-utilities ESearch + EFetch.

    Args:
        query: PubMed search query.
        max_results: Maximum number of results.
        days_back: Search within last N days.
        api_key: Optional NCBI API key for higher rate limits.
        verbose: Print progress.

    Returns:
        List of paper dicts with title, abstract, DOI, pub_date.
    """
    # Add date filter
    date_filter = f'AND ("{datetime.now().strftime("%Y/%m/%d")}"[PDAT] : ' \
                  f'"{(datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")}"[PDAT])'
    full_query = f"{query} {date_filter}"

    params: dict[str, Any] = {
        "db": "pubmed",
        "term": full_query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "date",
    }
    if api_key:
        params["api_key"] = api_key

    # Step 1: ESearch
    if verbose:
        print(f"[#4A90D9] Searching PubMed: {query[:80]}...")

    resp = requests.get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    search_data = resp.json()

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        if verbose:
            print("[#E74C3C] No results found.")
        return []

    if verbose:
        print(f"[#4A90D9] Found {len(id_list)} papers. Fetching details...")

    # Step 2: EFetch
    fetch_params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
    }
    if api_key:
        fetch_params["api_key"] = api_key

    resp = requests.post(f"{BASE_URL}/efetch.fcgi", data=fetch_params, timeout=60)
    resp.raise_for_status()

    # Parse XML
    root = ET.fromstring(resp.text)
    papers: list[dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        try:
            medline = article.find(".//MedlineCitation")
            art = medline.find(".//Article") if medline is not None else None
            if art is None:
                continue

            # Title
            title_el = art.find(".//ArticleTitle")
            title = "".join(title_el.itertext()) if title_el is not None else "No title"

            # Abstract
            abstract_parts = []
            for abs_el in art.findall(".//AbstractText"):
                label = abs_el.get("Label", "")
                text = "".join(abs_el.itertext())
                abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(abstract_parts) if abstract_parts else ""

            # Authors
            authors = []
            for author in art.findall(".//Author"):
                last = author.findtext("LastName", "")
                initials = author.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {initials}")
            author_str = ", ".join(authors[:5])
            if len(authors) > 5:
                author_str += f" et al. ({len(authors)} authors)"

            # Journal
            journal_el = art.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else ""

            # Publication date
            pub_date_el = art.find(".//PubDate")
            pub_date = ""
            if pub_date_el is not None:
                y = pub_date_el.findtext("Year", "")
                m = pub_date_el.findtext("Month", "")
                d = pub_date_el.findtext("Day", "")
                pub_date = f"{y}-{m}-{d}".strip("-")

            # DOI
            doi = ""
            for eid in article.findall(".//ArticleId"):
                if eid.get("IdType") == "doi":
                    doi = eid.text or ""
                    break

            # PMID
            pmid = ""
            for eid in article.findall(".//ArticleId"):
                if eid.get("IdType") == "pubmed":
                    pmid = eid.text or ""
                    break

            papers.append({
                "title": title.strip(),
                "abstract": abstract.strip(),
                "authors": author_str,
                "journal": journal,
                "pub_date": pub_date,
                "doi": doi,
                "pmid": pmid,
                "source": "pubmed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            })
        except Exception as e:
            if verbose:
                print(f"[#E74C3C] Parse error: {e}")
            continue

    time.sleep(0.34)  # NCBI rate limit (3 req/sec without key)

    return papers


def main() -> int:
    args = parse_args()
    papers = search_pubmed(args.query, args.max, args.days_back, args.api_key, args.verbose)

    print(f"[#4A90D9] ✓ Retrieved {len(papers)} papers from PubMed")

    if args.verbose:
        for i, p in enumerate(papers[:3]):
            print(f"  {i+1}. {p['title'][:80]}... ({p['pub_date']})")

    out = args.output or "pubmed_results.json"
    with open(out, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"[#4A90D9] Saved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
