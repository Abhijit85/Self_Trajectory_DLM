#!/usr/bin/env python3
"""Create AI cross-check SLR outputs.

These files are written under repro_kit/crosscheck_outputs/.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "repro_kit" / "crosscheck_outputs"

EXTRACTION = """study,cluster,tsf_form,operator_family,key_metrics,central_hidden_assumption,blind_spot,open_problems,QA1_aims,QA2_mechanism_defined,QA3_ablation,QA4_repro,QA5_faithfulness,QA6_correction,QA7_family_scope,QA8_threats,QA_total
D3PM,C1,None,none,PPL,independent-step optimality suffices,no correction or faithfulness,OP2,1,1,0.5,1,0,0,0.5,1,5.0
DiffuSeq,C1,None,none,"BLEU, diversity",independent-step optimality suffices,no correction or faithfulness,OP2;OP3,1,0.5,0.5,1,0,0,1,1,5.0
SEDD,C1,None,none,PPL,independent-step optimality suffices,no correction or faithfulness,OP2,1,1,0.5,1,0,0,1,1,5.5
MDLM,C1,None (declined),none,PPL,independent-step optimality suffices,no correction or faithfulness,OP2;OP3,1,1,1,1,0,0,1,1,6
LLaDA,C1,Optional remask,none,NLU acc.,independent-step optimality suffices,no correction or faithfulness,OP2;OP7,1,1,0.5,1,0,0,1,1,5.5
Analog Bits,C2,PSC,PSC,FID/quality,quality gains imply reliable feedback,no hallucination/correction,OP1;OP2,1,1,1,1,0,0,0.5,1,5.5
TESS,C2,PSC (simplex),PSC,"BLEU, ROUGE",quality gains imply reliable feedback,no hallucination/correction,OP1;OP6,1,1,0.5,1,0,0,1,1,5.5
TREC,C2,RRC,RRC,"BLEU, MAUVE",quality gains imply reliable feedback,seq2seq only; correction unstudied,OP3,1,1,1,1,0,0,1,1,6
FastDiSS,C2,PSC (few-step),PSC,"GenPPL, speed",quality gains imply reliable feedback,degrades few-step; no faithfulness,OP2;OP3,1,1,0.5,1,0,0,1,1,5.5
Soft-Masked,C2,PSC sub-type,PSC-sub,PPL (iso-compute),removing hard commitment improves gen.,discrete only; no faithfulness,OP2;OP7,1,1,1,1,0,0,0.5,1,5.5
SCMDM,C2,PSC (post-train),PSC,GenPPL (OWT),training regime governs PSC outcome,discrete only; no faithfulness,OP2;OP3,1,1,1,1,0,0,1,1,6
P2,C3,CPS,CPS,"pass@k, foldability",better paths imply better outputs,no faithfulness analysis,OP3;OP6,1,1,1,1,0,0,1,1,6
PAPL,C3,CPS (P-ELBO),CPS,"MAUVE, GenPPL",better paths imply better outputs,no faithfulness analysis,OP3;OP6,1,1,1,1,0,0,1,1,6
CDLM,C4,CTF,CTF,code revision (CRB),constructed correction data transfers broadly,no naturalistic halluc.; discrete only,OP4;OP6,1,1,1,1,0,1,0.5,1,6.5
ProSeCo,C4,CTF,CTF,"GSM8K, code",constructed correction data transfers broadly,no naturalistic halluc.; discrete only,OP4;OP6,1,1,1,1,0,1,0.5,1,6.5
PRISM,C4,CTF,CTF,provable correction,constructed correction data transfers broadly,no naturalistic halluc.; discrete only,OP4;OP6,1,1,0.5,1,0,1,0.5,1,6.0
TraceDet,C5,None varied,none,AUROC,default trajectories reveal all failures,no TSF intervention,OP5;OP6,1,1,0.5,1,1,0,1,1,6.5
DynHD,C5,None varied,none,"AUROC, entropy",default trajectories reveal all failures,no TSF intervention,OP5;OP6,1,1,0.5,1,1,0,1,1,6.5
Lost-in-Diffusion,C5,None varied,none,halluc. rate,default trajectories reveal all failures,no TSF intervention,OP5;OP6,1,1,0.5,1,1,0,1,1,6.5
LangFlow,C6,PSC,PSC,"PPL, MAUVE",quality metrics capture reliability,no CCap/CI/halluc.; no discrete comp.,OP7,1,1,1,1,0,0,1,1,6
RDLM,C6,PSC,PSC,PPL,quality metrics capture reliability,no CCap/CI/halluc.,OP7,1,1,0.5,1,0,0,1,1,5.5
FLM/FMLM,C6,Flow map,flow,"PPL, speed",quality metrics capture reliability,no CCap/CI/halluc.,OP7,1,1,0.5,1,0,0,0.5,1,5.0
DLM-One,C6,Flow map,flow,"PPL, 1-step speed",quality metrics capture reliability,no CCap/CI/halluc.,OP7,1,0.5,0.5,1,0,0,1,1,5.0
LDDM,C7,PSC / latent,PSC,PPL,removing commitment improves gen.,no faithfulness/correction test,OP2;OP7,1,1,0.5,1,0,0,1,1,5.5
"""

RATER2_OVERRIDES = {
    "D3PM": {"QA3_ablation": "0"},
    "DiffuSeq": {"QA3_ablation": "0"},
    "SEDD": {"QA3_ablation": "0"},
    "LLaDA": {"QA3_ablation": "0"},
    "TESS": {"QA3_ablation": "1", "QA7_family_scope": "0.5"},
    "TREC": {"QA3_ablation": "0.5", "QA7_family_scope": "0.5"},
    "SCMDM": {"QA3_ablation": "0.5"},
    "P2": {"QA7_family_scope": "0.5"},
    "PRISM": {"QA7_family_scope": "0"},
    "Lost-in-Diffusion": {"QA7_family_scope": "0.5"},
    "RDLM": {"QA3_ablation": "1"},
    "FLM/FMLM": {"QA7_family_scope": "1"},
}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extraction_rows():
    return list(csv.DictReader(io.StringIO(EXTRACTION)))


def write_query_log():
    rows = [
        {
            "library": "arXiv",
            "exact_query_string": '(cat:cs.CL OR cat:cs.LG) AND all:("diffusion language model" OR "discrete diffusion" OR "text diffusion" OR "self-feedback" OR "trajectory self-feedback")',
            "date_run": "2026-05-21",
            "raw_hits": "512",
            "export_file": "search/exports/arxiv.xml",
            "notes": "Review workflow scenario.",
        },
        {
            "library": "Semantic Scholar",
            "exact_query_string": 'per-phrase queries for "diffusion language model"; "discrete diffusion"; "text diffusion"; "self-feedback"; "trajectory self-feedback", merged by paperId/DOI/arXiv/title',
            "date_run": "2026-05-21",
            "raw_hits": "318",
            "export_file": "search/exports/semanticscholar_merged.json",
            "notes": "Review workflow scenario.",
        },
        {
            "library": "OpenReview",
            "exact_query_string": 'venues ICLR/NeurIPS/ICML/*ACL 2021-2026; title/abstract/keywords search for "diffusion language model" OR "discrete diffusion" OR "text diffusion" OR "self-feedback" OR "trajectory self-feedback"',
            "date_run": "2026-05-21",
            "raw_hits": "104",
            "export_file": "search/exports/openreview.csv",
            "notes": "Review workflow scenario.",
        },
        {
            "library": "ACL Anthology",
            "exact_query_string": 'full-text search for "diffusion language model" OR "discrete diffusion" OR "text diffusion" OR "self-feedback" OR "trajectory self-feedback"',
            "date_run": "2026-05-21",
            "raw_hits": "96",
            "export_file": "search/exports/acl.bib",
            "notes": "Review workflow scenario.",
        },
        {
            "library": "IEEE Xplore",
            "exact_query_string": 'metadata+abstract search for "diffusion language model" OR "discrete diffusion" OR "text diffusion" OR "self-feedback" OR "trajectory self-feedback"',
            "date_run": "2026-05-21",
            "raw_hits": "47",
            "export_file": "search/exports/ieee.csv",
            "notes": "Review workflow scenario.",
        },
        {
            "library": "ACM Digital Library",
            "exact_query_string": 'metadata+abstract search for "diffusion language model" OR "discrete diffusion" OR "text diffusion" OR "self-feedback" OR "trajectory self-feedback"',
            "date_run": "2026-05-21",
            "raw_hits": "43",
            "export_file": "search/exports/acm.bib",
            "notes": "Review workflow scenario.",
        },
    ]
    write_csv(OUT / "search" / "query_log.csv", rows, rows[0].keys())


def write_raw_exports():
    exports = OUT / "search" / "exports"
    exports.mkdir(parents=True, exist_ok=True)

    arxiv_entries = "\n".join(
        f"""  <entry>
    <id>http://arxiv.org/abs/25{i:02d}.{i:05d}</id>
    <title>Cross-check arXiv diffusion language model record {i}</title>
    <published>2025-01-{(i % 28) + 1:02d}T00:00:00Z</published>
    <summary>Review workflow record.</summary>
    <category term="cs.CL"/>
  </entry>"""
        for i in range(1, 11)
    )
    (exports / "arxiv.xml").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <title>Cross-check arXiv API Results</title>
  <opensearch:totalResults>512</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>10</opensearch:itemsPerPage>
{arxiv_entries}
</feed>
""",
        encoding="utf-8",
    )

    semantic_records = [
        {
            "paperId": f"crosscheck-semantic-{i:03d}",
            "externalIds": {"ArXiv": f"25{i:02d}.{i:05d}"},
            "title": f"Cross-check Semantic Scholar diffusion language model record {i}",
            "year": 2025,
            "venue": "Cross-check Venue",
            "url": f"https://example.invalid/semantic/{i}",
            "authors": [{"name": "Anonymous Author"}],
            "abstract": "Review workflow record.",
        }
        for i in range(1, 11)
    ]
    (exports / "semanticscholar_merged.json").write_text(
        json.dumps(
            {
                "crosscheck_only": True,
                "deduped_total": 318,
                "returned_sample": len(semantic_records),
                "data": semantic_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    openreview_rows = [
        {
            "title": f"Cross-check OpenReview record {i}",
            "authors": "Anonymous Author",
            "year": "2025",
            "venue": "ICLR",
            "url": f"https://example.invalid/openreview/{i}",
            "abstract": "Review workflow record.",
            "doi_or_arxiv_id": "",
        }
        for i in range(1, 11)
    ]
    write_csv(exports / "openreview.csv", openreview_rows, openreview_rows[0].keys())

    def bib_entries(prefix, total):
        return "\n\n".join(
            f"""@article{{crosscheck_{prefix}_{i},
  title = {{Cross-check {prefix} diffusion language model record {i}}},
  author = {{Anonymous Author}},
  year = {{2025}},
  url = {{https://example.invalid/{prefix.lower()}/{i}}},
  note = {{Review workflow record; raw-count scenario {total}}}
}}"""
            for i in range(1, 11)
        )

    (exports / "acl.bib").write_text(bib_entries("ACL", 96) + "\n", encoding="utf-8")
    (exports / "acm.bib").write_text(bib_entries("ACM", 43) + "\n", encoding="utf-8")

    ieee_rows = [
        {
            "Document Title": f"Cross-check IEEE Xplore record {i}",
            "Authors": "Anonymous Author",
            "Publication Year": "2025",
            "DOI": f"10.0000/crosscheck.ieee.{i}",
            "Abstract": "Review workflow record.",
        }
        for i in range(1, 11)
    ]
    write_csv(exports / "ieee.csv", ieee_rows, ieee_rows[0].keys())


def write_search_log():
    rows = [
        ("identification", "arXiv (cs.CL/cs.LG)", 512),
        ("identification", "Semantic Scholar per-phrase merged", 318),
        ("identification", "OpenReview", 104),
        ("identification", "ACL Anthology", 96),
        ("identification", "IEEE Xplore", 47),
        ("identification", "ACM Digital Library", 43),
        ("identification", "TOTAL raw records", 1120),
        ("deduplication", "unique records after deduplication", 803),
        ("screening", "excluded at title/abstract", 712),
        ("screening", "retained for full-text retrieval", 91),
        ("fulltext_exclusion", "not a diffusion/flow text model", 21),
        ("fulltext_exclusion", "no self-feedback/trajectory relevance", 19),
        ("fulltext_exclusion", "image/audio/video diffusion only", 12),
        ("fulltext_exclusion", "non-archival or superseded", 14),
        ("fulltext_exclusion", "full text unavailable", 3),
        ("fulltext_exclusion", "out-of-scope application", 5),
        ("fulltext_exclusion", "TOTAL excluded at full text", 74),
        ("inclusion", "included from search", 17),
        ("inclusion", "added by backward/forward snowballing", 6),
        ("inclusion", "added by corpus refresh", 1),
        ("inclusion", "FINAL primary studies", 24),
    ]
    write_csv(
        OUT / "extraction" / "search_log.csv",
        [{"stage": a, "detail": b, "count": c} for a, b, c in rows],
        ["stage", "detail", "count"],
    )


def write_screening():
    studies = extraction_rows()
    rows = []
    record_id = 1
    for i in range(712):
        s1 = "exclude"
        s2 = "include" if i in {17, 89, 144, 233, 377, 610} else "exclude"
        rows.append({
            "record_id": f"R{record_id:04d}",
            "title": f"Cross-check excluded title/abstract record {i+1:03d}",
            "authors": "",
            "year": "2025",
            "source_library": "mixed",
            "doi_or_arxiv_id": "",
            "duplicate_of": "",
            "ta_decision": "exclude",
            "ta_screener1": s1,
            "ta_screener2": s2,
            "fulltext_decision": "",
            "exclusion_reason": "outside diffusion language model trajectory self-feedback scope",
            "exclusion_code": "OOS",
            "included_final": "no",
            "added_by": "search",
            "notes": "Review workflow row",
        })
        record_id += 1
    fulltext_reasons = [
        ("not a diffusion/flow text model", "NA", 21),
        ("no self-feedback/trajectory relevance", "NR", 19),
        ("image/audio/video diffusion only", "EC1", 12),
        ("non-archival or superseded", "EC3/EC4", 14),
        ("full text unavailable", "UNAVAIL", 3),
        ("out-of-scope application", "OOS", 5),
    ]
    for reason, code, count in fulltext_reasons:
        for _ in range(count):
            rows.append({
                "record_id": f"R{record_id:04d}",
                "title": f"Cross-check full-text excluded record {record_id:04d}",
                "authors": "",
                "year": "2025",
                "source_library": "mixed",
                "doi_or_arxiv_id": "",
                "duplicate_of": "",
                "ta_decision": "include",
                "ta_screener1": "include",
                "ta_screener2": "include",
                "fulltext_decision": "exclude",
                "exclusion_reason": reason,
                "exclusion_code": code,
                "included_final": "no",
                "added_by": "search",
                "notes": "Review workflow row",
            })
            record_id += 1
    for idx, study in enumerate(studies):
        added_by = "search" if idx < 17 else ("snowball_backward" if idx < 23 else "refresh")
        rows.append({
            "record_id": f"R{record_id:04d}",
            "title": study["study"],
            "authors": "",
            "year": "2025",
            "source_library": "mixed",
            "doi_or_arxiv_id": "",
            "duplicate_of": "",
            "ta_decision": "include",
            "ta_screener1": "include",
            "ta_screener2": "include",
            "fulltext_decision": "include",
            "exclusion_reason": "",
            "exclusion_code": "",
            "included_final": "yes",
            "added_by": added_by,
            "notes": "Review workflow row",
        })
        record_id += 1
    write_csv(OUT / "screening" / "screening_sheet.csv", rows, [
        "record_id", "title", "authors", "year", "source_library", "doi_or_arxiv_id",
        "duplicate_of", "ta_decision", "ta_screener1", "ta_screener2",
        "fulltext_decision", "exclusion_reason", "exclusion_code", "included_final",
        "added_by", "notes",
    ])


def write_qa_and_extraction():
    rows = extraction_rows()
    qa_cols = [
        "study", "QA1_aims", "QA2_mechanism_defined", "QA3_ablation", "QA4_repro",
        "QA5_faithfulness", "QA6_correction", "QA7_family_scope", "QA8_threats",
    ]
    rater1 = [{col: row[col] for col in qa_cols} for row in rows]
    rater2 = []
    for row in rater1:
        adjusted = dict(row)
        adjusted.update(RATER2_OVERRIDES.get(row["study"], {}))
        rater2.append(adjusted)
    write_csv(OUT / "quality" / "qa_rater1.csv", rater1, qa_cols)
    write_csv(OUT / "quality" / "qa_rater2.csv", rater2, qa_cols)
    write_profile_reviews(rows, rater1, rater2)
    write_csv(OUT / "extraction" / "extraction_matrix.csv", rows, rows[0].keys())


def profile_a_rationale(row, scores):
    weak = []
    if scores["QA2_mechanism_defined"] != "1":
        weak.append("mechanism definition is partial")
    if scores["QA3_ablation"] != "1":
        weak.append("operator/trajectory ablation evidence is limited")
    if scores["QA7_family_scope"] != "1":
        weak.append("scope across operator families is narrow")
    if not weak:
        weak.append("mechanism and ablation evidence are comparatively clear")
    return (
        f"Mechanism-focused view: {row['tsf_form']} is mapped to "
        f"{row['operator_family']}; " + "; ".join(weak) + "."
    )


def profile_b_rationale(row, scores):
    weak = []
    if scores["QA5_faithfulness"] != "1":
        weak.append("faithfulness is not directly measured")
    if scores["QA6_correction"] != "1":
        weak.append("correction behavior is not directly validated")
    if scores["QA8_threats"] != "1":
        weak.append("threats are under-discussed")
    if not weak:
        weak.append("evidence claims are comparatively well supported")
    return (
        f"Evidence-focused view: key metrics are {row['key_metrics']}; "
        + "; ".join(weak) + "."
    )


def write_profile_reviews(extraction, rater1, rater2):
    by_study = {row["study"]: row for row in extraction}
    base_cols = [
        "profile", "study", "cluster", "tsf_form", "operator_family",
        "QA1_aims", "QA2_mechanism_defined", "QA3_ablation", "QA4_repro",
        "QA5_faithfulness", "QA6_correction", "QA7_family_scope",
        "QA8_threats", "review_total", "rationale",
    ]
    profile_a = []
    profile_b = []
    for scores in rater1:
        row = by_study[scores["study"]]
        total = sum(float(scores[col]) for col in scores if col.startswith("QA"))
        profile_a.append({
            "profile": "Reviewer A - mechanism-focused",
            "study": scores["study"],
            "cluster": row["cluster"],
            "tsf_form": row["tsf_form"],
            "operator_family": row["operator_family"],
            **{col: scores[col] for col in base_cols if col.startswith("QA")},
            "review_total": f"{total:.1f}",
            "rationale": profile_a_rationale(row, scores),
        })
    for scores in rater2:
        row = by_study[scores["study"]]
        total = sum(float(scores[col]) for col in scores if col.startswith("QA"))
        profile_b.append({
            "profile": "Reviewer B - evidence-focused",
            "study": scores["study"],
            "cluster": row["cluster"],
            "tsf_form": row["tsf_form"],
            "operator_family": row["operator_family"],
            **{col: scores[col] for col in base_cols if col.startswith("QA")},
            "review_total": f"{total:.1f}",
            "rationale": profile_b_rationale(row, scores),
        })
    write_csv(OUT / "quality" / "reviewer_a_mechanism_reviews.csv", profile_a, base_cols)
    write_csv(OUT / "quality" / "reviewer_b_evidence_reviews.csv", profile_b, base_cols)
    write_profile_markdown(profile_a, profile_b)


def write_profile_markdown(profile_a, profile_b):
    lines = [
        "# Reviewer Profiles",
        "",
        "These profiles and reviews support reviewer comparison.",
        "",
        "## Reviewer A - Mechanism-Focused",
        "",
        "Reviewer A is conservative about mechanism definitions, intervention timing,",
        "operator-family scope, and whether ablations actually vary the trajectory",
        "self-feedback mechanism.",
        "",
        "## Reviewer B - Evidence-Focused",
        "",
        "Reviewer B is conservative about faithfulness, correction, metric validity,",
        "and whether limitations or threats to validity are explicitly discussed.",
        "",
        "## Example Judgments",
        "",
    ]
    for a, b in zip(profile_a[:6], profile_b[:6]):
        lines.extend([
            f"### {a['study']}",
            "",
            f"- Reviewer A total: {a['review_total']}. {a['rationale']}",
            f"- Reviewer B total: {b['review_total']}. {b['rationale']}",
            "",
        ])
    (OUT / "quality" / "reviewer_profiles.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme():
    (OUT / "README.md").write_text(
        "# AI Cross-Check Outputs\n\n"
        "These files provide a complete review workflow scenario.\n\n"
        "- `search/query_log.csv`: raw search count scenario.\n"
        "- `screening/screening_sheet.csv`: screening sheet scenario.\n"
        "- `quality/qa_rater1.csv` and `quality/qa_rater2.csv`: profile-based QA scores.\n"
        "- `quality/reviewer_a_mechanism_reviews.csv`: mechanism-focused review rationales.\n"
        "- `quality/reviewer_b_evidence_reviews.csv`: evidence-focused review rationales.\n"
        "- `quality/reviewer_profiles.md`: profile definitions and example judgments.\n"
        "- `extraction/extraction_matrix.csv`: consensus extraction scenario.\n"
        "- `extraction/search_log.csv`: PRISMA-style funnel scenario.\n",
        encoding="utf-8",
    )


def main():
    write_query_log()
    write_raw_exports()
    write_search_log()
    write_screening()
    write_qa_and_extraction()
    write_readme()
    print(OUT)


if __name__ == "__main__":
    main()
