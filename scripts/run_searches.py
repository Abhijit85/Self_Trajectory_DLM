#!/usr/bin/env python3
"""Run public SLR searches and save raw exports where public APIs permit.

This script intentionally does not invent counts for sources that require a
manual UI, subscription, or login. Those rows are emitted with blank raw_hits
and a note explaining what must be run by a human reviewer.
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "repro_kit" / "search" / "exports"
QUERY_LOG = ROOT / "repro_kit" / "search" / "query_log.csv"

PHRASES = [
    "diffusion language model",
    "discrete diffusion",
    "text diffusion",
    "self-feedback",
    "trajectory self-feedback",
]
TERMS = "(" + " OR ".join(f'"{phrase}"' for phrase in PHRASES) + ")"


def fetch(url: str, accept: str = "*/*", attempts: int = 1, backoff: int = 5, headers=None) -> bytes:
    request_headers = {"User-Agent": "SLR-repro-kit/0.1", "Accept": accept}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, headers=request_headers)
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts:
                raise
        except (TimeoutError, urllib.error.URLError) as exc:
            last_exc = exc
            if attempt == attempts:
                raise
        time.sleep(backoff * attempt)
    raise RuntimeError(last_exc)


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


def semantic_scholar_phrase_params(phrase):
    return {
        "query": f'"{phrase}"',
        "year": "2021-2026",
        "fieldsOfStudy": "Computer Science",
        "limit": "100",
        "fields": "title,authors,year,venue,externalIds,url,abstract",
    }


def run_arxiv():
    query = arxiv_query()
    encoded = urllib.parse.urlencode({"search_query": query, "start": 0, "max_results": 2000})
    url = f"https://export.arxiv.org/api/query?{encoded}"
    raw = fetch(url, "application/atom+xml", attempts=5, backoff=10)
    path = EXPORTS / "arxiv.xml"
    path.write_bytes(raw)
    root = ET.fromstring(raw)
    ns = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
    total = root.findtext("opensearch:totalResults", namespaces=ns)
    return query, total or "", "search/exports/arxiv.xml", url


def run_semantic_scholar():
    merged = {}
    phrase_summaries = []
    urls = []
    for phrase in PHRASES:
        params = semantic_scholar_phrase_params(phrase)
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
        urls.append(url)
        headers = {}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key
        raw = fetch(url, "application/json", attempts=4, backoff=8, headers=headers)
        phrase_path = EXPORTS / f"semanticscholar_{phrase.replace(' ', '_').replace('-', '_')}.json"
        phrase_path.write_bytes(raw)
        payload = json.loads(raw)
        phrase_summaries.append({"phrase": phrase, "total": payload.get("total", 0), "returned": len(payload.get("data", []))})
        for paper in payload.get("data", []):
            ext = paper.get("externalIds") or {}
            key = (
                paper.get("paperId")
                or ext.get("DOI")
                or ext.get("ArXiv")
                or (paper.get("title") or "").strip().lower()
            )
            if key:
                merged[key] = paper
        time.sleep(3)
    merged_payload = {
        "queries": [semantic_scholar_phrase_params(phrase) for phrase in PHRASES],
        "phrase_summaries": phrase_summaries,
        "deduped_total": len(merged),
        "data": list(merged.values()),
    }
    path = EXPORTS / "semanticscholar_merged.json"
    path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")
    query = " ; ".join(json.dumps(semantic_scholar_phrase_params(phrase), sort_keys=True) for phrase in PHRASES)
    notes = "URLs: " + " | ".join(urls) + "; phrase_summaries=" + json.dumps(phrase_summaries, sort_keys=True)
    return query, str(len(merged)), "search/exports/semanticscholar_merged.json", notes


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
        (
            "Semantic Scholar",
            lambda: " ; ".join(json.dumps(semantic_scholar_phrase_params(phrase), sort_keys=True) for phrase in PHRASES),
            run_semantic_scholar,
        ),
    ]
    for library, query_factory, runner in public_runs:
        query_text = query_factory()
        try:
            query, hits, export_file, notes = runner()
            rows.append([library, query, today, hits, export_file, notes])
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
