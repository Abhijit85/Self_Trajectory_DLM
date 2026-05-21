#!/usr/bin/env python3
"""Run public SLR searches and save raw exports where public APIs permit.

This script intentionally does not invent counts for sources that require a
manual UI, subscription, or login. Those rows are emitted with blank raw_hits
and a note explaining what must be run by a human reviewer.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "repro_kit" / "search" / "exports"
QUERY_LOG = ROOT / "repro_kit" / "search" / "query_log.csv"

TERMS = '("diffusion language model" OR "discrete diffusion" OR "text diffusion" OR "self-feedback" OR "trajectory self-feedback")'


def fetch(url: str, accept: str = "*/*") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SLR-repro-kit/0.1", "Accept": accept})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def arxiv_query():
    query = f'(cat:cs.CL OR cat:cs.LG) AND all:{TERMS}'
    return query


def semantic_scholar_params():
    return {
        "query": TERMS.replace('"', ""),
        "year": "2021-2026",
        "fieldsOfStudy": "Computer Science",
        "limit": "100",
        "fields": "title,authors,year,venue,externalIds,url,abstract",
    }


def run_arxiv():
    query = arxiv_query()
    encoded = urllib.parse.urlencode({"search_query": query, "start": 0, "max_results": 2000})
    url = f"https://export.arxiv.org/api/query?{encoded}"
    raw = fetch(url, "application/atom+xml")
    path = EXPORTS / "arxiv.xml"
    path.write_bytes(raw)
    root = ET.fromstring(raw)
    ns = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
    total = root.findtext("opensearch:totalResults", namespaces=ns)
    return query, total or "", "search/exports/arxiv.xml", url


def run_semantic_scholar():
    params = semantic_scholar_params()
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
    raw = fetch(url, "application/json")
    path = EXPORTS / "semanticscholar.json"
    path.write_bytes(raw)
    payload = json.loads(raw)
    return json.dumps(params, sort_keys=True), str(payload.get("total", "")), "search/exports/semanticscholar.json", url


def manual_rows():
    return [
        (
            "OpenReview",
            f'venues ICLR/NeurIPS/ICML/*ACL 2021-2026; full-text/metadata search for {TERMS}',
            "",
            "search/exports/openreview.csv",
            "Manual/API export required; do not enter a count until the export is saved.",
        ),
        (
            "ACL Anthology",
            f'full-text search for {TERMS}',
            "",
            "search/exports/acl.bib",
            "Manual export required from ACL Anthology search.",
        ),
        (
            "IEEE Xplore",
            f'metadata+abstract search for {TERMS}',
            "",
            "search/exports/ieee.csv",
            "Subscription/manual export required.",
        ),
        (
            "ACM Digital Library",
            f'metadata+abstract search for {TERMS}',
            "",
            "search/exports/acm.bib",
            "Subscription/manual export required.",
        ),
    ]


def main():
    EXPORTS.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    rows = []
    public_runs = [
        ("arXiv", arxiv_query, run_arxiv),
        ("Semantic Scholar", lambda: json.dumps(semantic_scholar_params(), sort_keys=True), run_semantic_scholar),
    ]
    for library, query_factory, runner in public_runs:
        query_text = query_factory()
        try:
            query, hits, export_file, url = runner()
            rows.append([library, query, today, hits, export_file, f"URL: {url}"])
            time.sleep(3)
        except Exception as exc:
            rows.append([library, query_text, today, "", "", f"RUN FAILED: {exc}. Re-run when API access/rate limit permits."])
    for library, query, hits, export_file, notes in manual_rows():
        rows.append([library, query, "", hits, export_file, notes])

    with QUERY_LOG.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["library", "exact_query_string", "date_run", "raw_hits", "export_file", "notes"])
        writer.writerows(rows)

    print(f"Wrote {QUERY_LOG}")


if __name__ == "__main__":
    main()
