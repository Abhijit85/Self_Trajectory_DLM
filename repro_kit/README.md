# Reproducibility Kit — SLR of Trajectory Self-Feedback in DLMs

This repository is the audit trail for the systematic literature review. It lets a
reader retrace the search, screening, quality scoring, and extraction.

```
repro_kit/
├── README.md                 (this file)
├── PRISMA_map.md             PRISMA 2020 reporting items -> paper locations
├── search/
│   ├── query_log.csv         exact query per library + date + raw hit count
│   └── exports/              raw result exports per library (.bib/.csv/.json)  [you add]
├── screening/
│   └── screening_sheet.csv   one row per record: T/A + full-text decisions, reasons
├── quality/
│   ├── qa_rubric.md          QA1–QA8 scoring rubric + calibration examples
│   ├── compute_kappa.py      inter-rater agreement (screening + QA)
│   ├── qa_rater1.csv         rater 1 QA scores on the double-scored sample  [you add]
│   └── qa_rater2.csv         rater 2 QA scores on the double-scored sample  [you add]
└── extraction/
    ├── extraction_matrix.csv final per-study matrix (24 × QA1–QA8 + 7 fields)
    └── search_log.csv        funnel summary (per-library, per-stage, per-reason)
```

## How a reviewer audits the central claim
1. Open `extraction/extraction_matrix.csv`.
2. Filter `QA5_faithfulness == 1` (studies that measured faithfulness).
3. Check whether any of those rows also has a varied TSF mechanism
   (`operator_family != none`). The claim is that the intersection is **empty**.
4. One counterexample falsifies the headline finding.

## How to fill this in (see paper §3 and the response notes)
1. Run each query in `search/query_log.csv`, save raw exports to `search/exports/`,
   record real `raw_hits` and the run date.
2. Deduplicate; record per-record decisions in `screening/screening_sheet.csv`.
3. Two raters score a ~40% sample with `quality/qa_rubric.md`; run
   `python quality/compute_kappa.py --mode qa quality/qa_rater1.csv quality/qa_rater2.csv`
   and `--mode screening screening/screening_sheet.csv`; report the κ values in §3.5.
4. Reconcile to consensus; write final values into `extraction/extraction_matrix.csv`.
5. Recompute the funnel totals into `extraction/search_log.csv` and Figure 1.

## Anonymity (for double-blind submission)
Host this on **anonymous.4open.science** (Anonymous GitHub) or an **OSF anonymized
view-only link**, and cite that link in the paper footnote. Keep author names, emails,
and institution strings out of every file. For camera-ready, move to GitHub + Zenodo
(for a DOI) and de-anonymise.

## Status
The counts and scores currently shipped are an internally consistent **worked template**,
not a record of an executed search. Replace them with your real data before submission.
