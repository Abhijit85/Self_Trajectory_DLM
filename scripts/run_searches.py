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
OPENREVIEW_BASEURL = "https://api2.openreview.net"
ACL_GIT_TREE_URL = "https://api.github.com/repos/acl-org/acl-anthology/git/trees/master?recursive=1"
ACL_RAW_BASE = "https://raw.githubusercontent.com/acl-org/acl-anthology/master/"
IEEE_API_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"


def load_env(path=ROOT / ".env"):
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


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


def fetch_json(url: str, attempts: int = 1, backoff: int = 5, headers=None):
    return json.loads(fetch(url, "application/json", attempts=attempts, backoff=backoff, headers=headers))


def post_json(url: str, payload, attempts: int = 1, backoff: int = 5, headers=None):
    request_headers = {"User-Agent": "SLR-repro-kit/0.1", "Accept": "application/json", "Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
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
        time.sleep(5)
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


def openreview_login_headers():
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        return {}
    payload = {"id": username, "password": password}
    response = post_json(f"{OPENREVIEW_BASEURL}/login", payload, attempts=2, backoff=3)
    token = response.get("token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def content_value(content, key, default=""):
    value = (content or {}).get(key, default)
    if isinstance(value, dict):
        return value.get("value", default)
    return value


def xml_text(elem):
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def bibtex_escape(value):
    return (value or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def run_acl_anthology():
    tree = fetch_json(ACL_GIT_TREE_URL, attempts=3, backoff=5)
    xml_paths = sorted(
        item["path"] for item in tree.get("tree", [])
        if item.get("type") == "blob"
        and item.get("path", "").startswith("data/xml/")
        and item.get("path", "").endswith(".xml")
        and item["path"].split("/")[-1][:4].isdigit()
        and 2021 <= int(item["path"].split("/")[-1][:4]) <= 2026
    )
    matches = {}
    downloaded = []
    terms_lower = [phrase.lower() for phrase in PHRASES]
    for path in xml_paths:
        url = ACL_RAW_BASE + path
        try:
            raw = fetch(url, "application/xml", attempts=2, backoff=2)
        except Exception:
            continue
        downloaded.append(path)
        root = ET.fromstring(raw)
        collection_id = root.attrib.get("id", "")
        for volume in root.findall("volume"):
            meta = volume.find("meta")
            year = xml_text(meta.find("year")) if meta is not None else collection_id[:4]
            if year and year.isdigit() and not (2021 <= int(year) <= 2026):
                continue
            booktitle = xml_text(meta.find("booktitle")) if meta is not None else ""
            venue = xml_text(meta.find("venue")) if meta is not None else ""
            volume_id = volume.attrib.get("id", "")
            for paper in volume.findall("paper"):
                title = xml_text(paper.find("title"))
                abstract = xml_text(paper.find("abstract"))
                haystack = f"{title} {abstract}".lower()
                if not any(term in haystack for term in terms_lower):
                    continue
                url_id = xml_text(paper.find("url"))
                bibkey = xml_text(paper.find("bibkey")) or url_id or f"{collection_id}-{volume_id}-{paper.attrib.get('id', '')}"
                authors = []
                for author in paper.findall("author"):
                    name = " ".join(part for part in [xml_text(author.find("first")), xml_text(author.find("last"))] if part)
                    if name:
                        authors.append(name)
                matches[bibkey] = {
                    "id": url_id,
                    "bibkey": bibkey,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "booktitle": booktitle,
                    "doi": xml_text(paper.find("doi")),
                    "url": f"https://aclanthology.org/{url_id}/" if url_id else "",
                    "abstract": abstract,
                    "source_xml": path,
                }
        time.sleep(0.2)

    records = list(matches.values())
    json_path = EXPORTS / "acl.json"
    json_path.write_text(json.dumps({
        "queries": PHRASES,
        "xml_files_considered": len(xml_paths),
        "xml_files_downloaded": len(downloaded),
        "deduped_total": len(records),
        "records": records,
    }, indent=2), encoding="utf-8")

    csv_path = EXPORTS / "acl.csv"
    with csv_path.open("w", newline="") as fh:
        fieldnames = ["id", "bibkey", "title", "authors", "year", "venue", "booktitle", "doi", "url", "abstract", "source_xml"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["authors"] = "; ".join(record["authors"])
            writer.writerow(row)

    bib_path = EXPORTS / "acl.bib"
    with bib_path.open("w") as fh:
        for record in records:
            fh.write(f"@inproceedings{{{record['bibkey']},\n")
            fh.write(f"  title = {{{bibtex_escape(record['title'])}}},\n")
            if record["authors"]:
                fh.write(f"  author = {{{bibtex_escape(' and '.join(record['authors']))}}},\n")
            fh.write(f"  year = {{{bibtex_escape(record['year'])}}},\n")
            if record["booktitle"]:
                fh.write(f"  booktitle = {{{bibtex_escape(record['booktitle'])}}},\n")
            if record["doi"]:
                fh.write(f"  doi = {{{bibtex_escape(record['doi'])}}},\n")
            if record["url"]:
                fh.write(f"  url = {{{bibtex_escape(record['url'])}}},\n")
            fh.write("}\n\n")

    query = f"ACL Anthology official XML metadata, 2021-2026, title/abstract contains any of {TERMS}"
    notes = f"{ACL_GIT_TREE_URL}; xml_files_considered={len(xml_paths)}; xml_files_downloaded={len(downloaded)}"
    return query, str(len(records)), "search/exports/acl.bib", notes


def run_ieee_xplore():
    api_key = os.environ.get("IEEE_XPLORE_API_KEY")
    if not api_key:
        raise RuntimeError("IEEE_XPLORE_API_KEY is not set")

    merged = {}
    phrase_summaries = []
    urls = []
    for phrase in PHRASES:
        # Match the official Python SDK's XPLORE.queryText(),
        # resultsFilter(), maximumResults(), and dataType('json') URL shape.
        params = {
            "apikey": api_key,
            "format": "json",
            "querytext": phrase,
            "start_year": "2021",
            "end_year": "2026",
            "max_records": "200",
            "start_record": "1",
            "sort_field": "article_title",
            "sort_order": "asc",
        }
        url = IEEE_API_URL + "?" + urllib.parse.urlencode(params)
        safe_params = dict(params)
        safe_params["apikey"] = "REDACTED"
        safe_url = IEEE_API_URL + "?" + urllib.parse.urlencode(safe_params)
        urls.append(safe_url)
        payload = fetch_json(url, attempts=3, backoff=5)
        articles = payload.get("articles", [])
        phrase_summaries.append({
            "phrase": phrase,
            "total": payload.get("total_records", payload.get("total_searched", "")),
            "returned": len(articles),
        })
        for article in articles:
            key = (
                str(article.get("article_number") or "")
                or article.get("doi")
                or (article.get("title") or "").strip().lower()
            )
            if key:
                merged[key] = article
        time.sleep(1.5)

    records = list(merged.values())
    json_path = EXPORTS / "ieee.json"
    json_path.write_text(json.dumps({
        "queries": [{**{k: v for k, v in {
            "querytext": f'"{phrase}"',
            "start_year": "2021",
            "end_year": "2026",
            "max_records": "200",
        }.items()}} for phrase in PHRASES],
        "phrase_summaries": phrase_summaries,
        "deduped_total": len(records),
        "articles": records,
    }, indent=2), encoding="utf-8")

    csv_path = EXPORTS / "ieee.csv"
    with csv_path.open("w", newline="") as fh:
        fieldnames = [
            "article_number", "title", "authors", "publication_year",
            "publication_title", "doi", "html_url", "abstract",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for article in records:
            authors = article.get("authors", {}).get("authors", []) if isinstance(article.get("authors"), dict) else []
            author_names = "; ".join(a.get("full_name", "") for a in authors if a.get("full_name"))
            writer.writerow({
                "article_number": article.get("article_number", ""),
                "title": article.get("title", ""),
                "authors": author_names,
                "publication_year": article.get("publication_year", ""),
                "publication_title": article.get("publication_title", ""),
                "doi": article.get("doi", ""),
                "html_url": article.get("html_url", ""),
                "abstract": article.get("abstract", ""),
            })

    query = f"IEEE Xplore Python SDK-style queryText per phrase over {TERMS}; start_year=2021, end_year=2026"
    notes = "URLs with API key redacted: " + " | ".join(urls) + "; phrase_summaries=" + json.dumps(phrase_summaries, sort_keys=True)
    return query, str(len(records)), "search/exports/ieee.csv", notes


def note_year(note):
    content = note.get("content") or {}
    year = content_value(content, "year", "")
    if year:
        return str(year)
    cdate = note.get("cdate") or note.get("tcdate")
    if cdate:
        return str(time.gmtime(int(cdate) / 1000).tm_year)
    return ""


def note_matches_openreview_scope(note):
    year = note_year(note)
    if year and year.isdigit() and not (2021 <= int(year) <= 2026):
        return False
    haystack = " ".join(
        str(x)
        for x in [
            note.get("invitation", ""),
            " ".join(note.get("invitations", []) or []),
            content_value(note.get("content") or {}, "venue", ""),
            content_value(note.get("content") or {}, "venueid", ""),
        ]
    ).lower()
    venues = ["iclr", "neurips", "nips", "icml", "acl", "emnlp", "naacl", "eacl", "aacl", "coling", "conll"]
    return not haystack or any(v in haystack for v in venues)


def run_openreview():
    headers = openreview_login_headers()
    merged = {}
    phrase_summaries = []
    urls = []
    for phrase in PHRASES:
        params = {
            "term": phrase,
            "type": "exact",
            "content": "all",
            "limit": 1000,
        }
        url = f"{OPENREVIEW_BASEURL}/notes/search?" + urllib.parse.urlencode(params)
        urls.append(url)
        payload = fetch_json(url, attempts=3, backoff=5, headers=headers)
        notes = [n for n in payload.get("notes", []) if note_matches_openreview_scope(n)]
        phrase_summaries.append({"phrase": phrase, "returned": len(payload.get("notes", [])), "in_scope": len(notes)})
        for note in notes:
            key = note.get("id") or (content_value(note.get("content") or {}, "title", "").strip().lower())
            if key:
                merged[key] = note
        time.sleep(1.5)

    path = EXPORTS / "openreview.json"
    path.write_text(json.dumps({
        "queries": PHRASES,
        "phrase_summaries": phrase_summaries,
        "deduped_total": len(merged),
        "notes": list(merged.values()),
    }, indent=2), encoding="utf-8")

    csv_path = EXPORTS / "openreview.csv"
    with csv_path.open("w", newline="") as fh:
        fieldnames = ["id", "title", "authors", "year", "venue", "url", "abstract"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for note in merged.values():
            content = note.get("content") or {}
            authors = content_value(content, "authors", [])
            if isinstance(authors, list):
                authors = "; ".join(str(a) for a in authors)
            writer.writerow({
                "id": note.get("id", ""),
                "title": content_value(content, "title", ""),
                "authors": authors,
                "year": note_year(note),
                "venue": content_value(content, "venue", "") or content_value(content, "venueid", ""),
                "url": f"https://openreview.net/forum?id={note.get('forum') or note.get('id', '')}",
                "abstract": content_value(content, "abstract", ""),
            })

    query = f"OpenReview /notes/search per phrase over {TERMS}; scoped to ICLR/NeurIPS/ICML/*ACL-like venues, 2021-2026"
    notes = "URLs: " + " | ".join(urls) + "; phrase_summaries=" + json.dumps(phrase_summaries, sort_keys=True)
    return query, str(len(merged)), "search/exports/openreview.csv", notes


def manual_rows():
    return [
        (
            "ACM Digital Library",
            f'metadata+abstract search for {TERMS}',
            "",
            "search/exports/acm.bib",
            "Subscription/manual export required.",
        ),
    ]


def main():
    load_env()
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
        (
            "OpenReview",
            lambda: f"OpenReview /notes/search per phrase over {TERMS}; scoped to ICLR/NeurIPS/ICML/*ACL-like venues, 2021-2026",
            run_openreview,
        ),
        (
            "ACL Anthology",
            lambda: f"ACL Anthology official XML metadata, 2021-2026, title/abstract contains any of {TERMS}",
            run_acl_anthology,
        ),
        (
            "IEEE Xplore",
            lambda: f"IEEE Xplore Metadata API per phrase over {TERMS}; querytext, start_year=2021, end_year=2026",
            run_ieee_xplore,
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
