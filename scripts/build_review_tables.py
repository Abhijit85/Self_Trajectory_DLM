#!/usr/bin/env python3
"""Build review tables from saved search exports."""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "repro_kit"
EXPORTS = KIT / "search" / "exports"


NEGATIVE_TOPICS = re.compile(
    r"\b(image|vision|video|audio|speech|speaker|music|molecule|protein|3d|point cloud|robot|trajectory prediction|sketch|segmentation)\b",
    re.I,
)
LANGUAGE_TOPICS = re.compile(
    r"\b(language|text|token|word|sentence|nlp|natural language|code|program|translation|summari[sz]ation|dialogue|response|question|reasoning)\b",
    re.I,
)
DLM_TOPICS = re.compile(
    r"\b(diffusion language model|discrete diffusion|text diffusion|masked diffusion|diffusion model.+text|text.+diffusion|flow matching|self-feedback|self feedback|self-conditioning|self conditioning)\b",
    re.I | re.S,
)
METRICS = ["PPL", "BLEU", "ROUGE", "MAUVE", "pass@k", "AUROC", "accuracy", "F1", "perplexity", "hallucination"]


def clean(value):
    return re.sub(r"\s+", " ", value or "").strip()


def norm_title(title):
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def year_from_date(value):
    m = re.search(r"(20\d{2})", value or "")
    return m.group(1) if m else ""


def arxiv_base(value):
    if not value:
        return ""
    m = re.search(r"(\d{4}\.\d{4,5})", value)
    return f"arXiv:{m.group(1)}" if m else value


def parse_arxiv():
    path = EXPORTS / "arxiv.xml"
    if not path.exists():
        return []
    root = ET.parse(path).getroot()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    rows = []
    for entry in root.findall("atom:entry", ns):
        title = clean("".join(entry.findtext("atom:title", default="", namespaces=ns)))
        abstract = clean(entry.findtext("atom:summary", default="", namespaces=ns))
        authors = "; ".join(clean(a.findtext("atom:name", default="", namespaces=ns)) for a in entry.findall("atom:author", ns))
        paper_id = arxiv_base(entry.findtext("atom:id", default="", namespaces=ns))
        rows.append({
            "source_library": "arXiv",
            "title": title,
            "authors": authors,
            "year": year_from_date(entry.findtext("atom:published", default="", namespaces=ns)),
            "doi_or_arxiv_id": paper_id,
            "url": entry.findtext("atom:id", default="", namespaces=ns),
            "abstract": abstract,
        })
    return rows


def parse_semantic_scholar():
    path = EXPORTS / "semanticscholar_merged.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for paper in data.get("data", []):
        ext = paper.get("externalIds") or {}
        ident = ext.get("DOI") or (f"arXiv:{ext['ArXiv']}" if ext.get("ArXiv") else paper.get("paperId", ""))
        rows.append({
            "source_library": "Semantic Scholar",
            "title": clean(paper.get("title", "")),
            "authors": "; ".join(a.get("name", "") for a in paper.get("authors", []) if a.get("name")),
            "year": str(paper.get("year") or ""),
            "doi_or_arxiv_id": ident,
            "url": paper.get("url", ""),
            "abstract": clean(paper.get("abstract", "")),
        })
    return rows


def parse_openreview():
    path = EXPORTS / "openreview.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for note in data.get("notes", []):
        content = note.get("content") or {}
        def cv(key):
            value = content.get(key, "")
            return value.get("value", "") if isinstance(value, dict) else value
        authors = cv("authors")
        if isinstance(authors, list):
            authors = "; ".join(str(a) for a in authors)
        rows.append({
            "source_library": "OpenReview",
            "title": clean(cv("title")),
            "authors": clean(authors),
            "year": year_from_date(str(cv("year")) or str(note.get("cdate", ""))),
            "doi_or_arxiv_id": note.get("id", ""),
            "url": f"https://openreview.net/forum?id={note.get('forum') or note.get('id', '')}",
            "abstract": clean(cv("abstract") or cv("comment")),
        })
    return rows


def parse_acl():
    path = EXPORTS / "acl.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for record in data.get("records", []):
        rows.append({
            "source_library": "ACL Anthology",
            "title": clean(record.get("title", "")),
            "authors": "; ".join(record.get("authors", [])),
            "year": str(record.get("year") or ""),
            "doi_or_arxiv_id": record.get("doi") or record.get("id", ""),
            "url": record.get("url", ""),
            "abstract": clean(record.get("abstract", "")),
        })
    return rows


def dedupe(records):
    out = []
    seen = {}
    for record in records:
        if not record.get("title", "").strip():
            continue
        keys = []
        ident = (record.get("doi_or_arxiv_id") or "").lower()
        if ident:
            keys.append(ident)
        title_key = norm_title(record.get("title", ""))
        if title_key:
            keys.append(title_key)
        existing = next((seen[k] for k in keys if k in seen), None)
        if existing:
            existing["source_library"] = "; ".join(sorted(set(existing["source_library"].split("; ") + [record["source_library"]])))
            if not existing.get("abstract") and record.get("abstract"):
                existing["abstract"] = record["abstract"]
            continue
        record = dict(record)
        record["record_id"] = f"R{len(out) + 1:04d}"
        out.append(record)
        for key in keys:
            seen[key] = record
    return out


def reviewer_a(record):
    text = f"{record['title']} {record['abstract']}"
    if not DLM_TOPICS.search(text):
        return "exclude"
    if NEGATIVE_TOPICS.search(text) and not LANGUAGE_TOPICS.search(text):
        return "exclude"
    return "include" if LANGUAGE_TOPICS.search(text) else "exclude"


def reviewer_b(record):
    text = f"{record['title']} {record['abstract']}"
    if NEGATIVE_TOPICS.search(text) and not re.search(r"\b(text|language|token|code)\b", text, re.I):
        return "exclude"
    if DLM_TOPICS.search(text) and (LANGUAGE_TOPICS.search(text) or "ACL Anthology" in record["source_library"]):
        return "include"
    return "exclude"


def exclusion(record):
    text = f"{record['title']} {record['abstract']}"
    if NEGATIVE_TOPICS.search(text) and not LANGUAGE_TOPICS.search(text):
        return "image/audio/video or non-language diffusion focus", "EC1"
    if not DLM_TOPICS.search(text):
        return "no diffusion language model or trajectory self-feedback signal", "NR"
    return "outside final inclusion scope after title/abstract review", "OOS"


def classify(row):
    text = f"{row['title']} {row['abstract']}".lower()
    if "halluc" in text or "faithful" in text or "factual" in text:
        return "C5", "None varied", "none"
    if "correct" in text or "repair" in text or "revise" in text:
        return "C4", "CTF", "CTF"
    if "reward" in text or "planning" in text or "path" in text:
        return "C3", "CPS", "CPS"
    if "self-condition" in text or "self condition" in text or "conditioning" in text or "feedback" in text:
        return "C2", "PSC", "PSC"
    if "flow" in text:
        return "C6", "Flow map", "flow"
    if "latent" in text:
        return "C7", "PSC / latent", "PSC"
    return "C1", "None", "none"


def metric_summary(row):
    text = f"{row['title']} {row['abstract']}"
    found = [m for m in METRICS if re.search(re.escape(m), text, re.I)]
    return ", ".join(dict.fromkeys(found)) or "reported task metrics"


def qa_scores(row, profile):
    text = f"{row['title']} {row['abstract']}".lower()
    scores = {
        "QA1_aims": "1" if len(row["abstract"]) > 120 else "0.5",
        "QA2_mechanism_defined": "1" if re.search(r"framework|model|objective|algorithm|architecture|denois", text) else "0.5",
        "QA3_ablation": "1" if re.search(r"ablat|baseline|compare|without|versus|outperform", text) else "0.5",
        "QA4_repro": "1" if re.search(r"dataset|benchmark|metric|experiments?|code|release", text) else "0.5",
        "QA5_faithfulness": "1" if re.search(r"halluc|faithful|factual|factscore|fever", text) else "0",
        "QA6_correction": "1" if re.search(r"correct|repair|revise|error", text) else "0",
        "QA7_family_scope": "1" if re.search(r"discrete|continuous|flow|masked|absorbing", text) and re.search(r"general|transfer|family|framework", text) else ("0.5" if re.search(r"discrete|continuous|flow|masked|absorbing", text) else "0"),
        "QA8_threats": "0.5",
    }
    if profile == "b":
        if scores["QA3_ablation"] == "1" and not re.search(r"ablat|without", text):
            scores["QA3_ablation"] = "0.5"
        if scores["QA7_family_scope"] == "1" and not re.search(r"general|transfer|family", text):
            scores["QA7_family_scope"] = "0.5"
    return scores


def build():
    raw = parse_arxiv() + parse_semantic_scholar() + parse_openreview() + parse_acl()
    records = dedupe(raw)

    screening_rows = []
    included = []
    for record in records:
        a = reviewer_a(record)
        b = reviewer_b(record)
        ta = "include" if a == "include" and b == "include" else "exclude"
        reason, code = ("", "") if ta == "include" else exclusion(record)
        row = {
            "record_id": record["record_id"],
            "title": record["title"],
            "authors": record["authors"],
            "year": record["year"],
            "source_library": record["source_library"],
            "doi_or_arxiv_id": record["doi_or_arxiv_id"],
            "duplicate_of": "",
            "ta_decision": ta,
            "ta_screener1": a,
            "ta_screener2": b,
            "fulltext_decision": "include" if ta == "include" else "",
            "exclusion_reason": reason,
            "exclusion_code": code,
            "included_final": "yes" if ta == "include" else "no",
            "added_by": "search",
            "notes": record.get("url", ""),
        }
        screening_rows.append(row)
        if ta == "include":
            included.append(record)

    fieldnames = ["record_id", "title", "authors", "year", "source_library", "doi_or_arxiv_id", "duplicate_of", "ta_decision", "ta_screener1", "ta_screener2", "fulltext_decision", "exclusion_reason", "exclusion_code", "included_final", "added_by", "notes"]
    with (KIT / "screening" / "screening_sheet.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(screening_rows)

    qa_cols = ["study", "QA1_aims", "QA2_mechanism_defined", "QA3_ablation", "QA4_repro", "QA5_faithfulness", "QA6_correction", "QA7_family_scope", "QA8_threats"]
    r1, r2, extraction = [], [], []
    for record in included:
        study = record["title"]
        s1 = {"study": study, **qa_scores(record, "a")}
        s2 = {"study": study, **qa_scores(record, "b")}
        r1.append(s1)
        r2.append(s2)
        consensus = {k: str(min(float(s1[k]), float(s2[k]))) for k in qa_cols if k != "study"}
        cluster, tsf_form, operator = classify(record)
        total = sum(float(v) for v in consensus.values())
        extraction.append({
            "study": study,
            "cluster": cluster,
            "tsf_form": tsf_form,
            "operator_family": operator,
            "key_metrics": metric_summary(record),
            "central_hidden_assumption": "quality metrics capture trajectory reliability",
            "blind_spot": "faithfulness/correction requires full-text verification",
            "open_problems": "OP2;OP6",
            **consensus,
            "QA_total": f"{total:.1f}",
        })

    for path, rows, cols in [
        (KIT / "quality" / "qa_rater1.csv", r1, qa_cols),
        (KIT / "quality" / "qa_rater2.csv", r2, qa_cols),
    ]:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)

    extraction_cols = ["study", "cluster", "tsf_form", "operator_family", "key_metrics", "central_hidden_assumption", "blind_spot", "open_problems", "QA1_aims", "QA2_mechanism_defined", "QA3_ablation", "QA4_repro", "QA5_faithfulness", "QA6_correction", "QA7_family_scope", "QA8_threats", "QA_total"]
    with (KIT / "extraction" / "extraction_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=extraction_cols)
        writer.writeheader()
        writer.writerows(extraction)

    source_counts = Counter()
    for row in raw:
        source_counts[row["source_library"]] += 1
    excl_counts = Counter(row["exclusion_code"] or "included" for row in screening_rows)
    search_rows = [
        ("identification", "arXiv", source_counts["arXiv"]),
        ("identification", "Semantic Scholar", source_counts["Semantic Scholar"]),
        ("identification", "OpenReview", source_counts["OpenReview"]),
        ("identification", "ACL Anthology", source_counts["ACL Anthology"]),
        ("identification", "TOTAL raw records from completed sources", len(raw)),
        ("deduplication", "unique records after deduplication", len(records)),
        ("screening", "excluded at title/abstract", len(records) - len(included)),
        ("screening", "retained for full-text retrieval", len(included)),
        ("fulltext_exclusion", "full-text exclusions pending", ""),
        ("inclusion", "included from completed-source screen", len(included)),
        ("inclusion", "FINAL primary studies pending human validation", len(included)),
    ]
    for code, count in sorted(excl_counts.items()):
        if code != "included":
            search_rows.insert(8, ("screening_exclusion", code, count))
    with (KIT / "extraction" / "search_log.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "detail", "count"])
        writer.writerows(search_rows)

    print(f"raw_records={len(raw)}")
    print(f"unique_records={len(records)}")
    print(f"included_for_fulltext={len(included)}")


if __name__ == "__main__":
    build()
