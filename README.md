# Self_Trajectory_DLM

This repository contains reproducibility materials for a systematic literature
review of trajectory self-feedback in diffusion language models.

## Review Status

This repository is currently a reproducibility-kit scaffold. It is safe to use
as the audit structure for the review, but the included CSVs should not be cited
as final SLR evidence until the real search, screening, extraction, and
independent quality-rating work has been completed.

Known fields still to complete:

- `repro_kit/search/query_log.csv` contains one saved Semantic Scholar run;
  arXiv was rate-limited on 2026-05-21, and OpenReview, ACL Anthology, IEEE
  Xplore, and ACM Digital Library still require manual/API exports.
- `repro_kit/screening/screening_sheet.csv` contains fetched unscreened records
  from saved exports; two independent title/abstract decisions are still needed.
- `repro_kit/quality/qa_rater1.csv` and `repro_kit/quality/qa_rater2.csv`
  contain illustrative rater scores.
- `repro_kit/extraction/extraction_matrix.csv` contains worked-template
  extraction and QA values.

## Repository Contents

The main audit trail is in [`repro_kit/`](repro_kit/):

- [`repro_kit/search/query_log.csv`](repro_kit/search/query_log.csv): search
  queries, sources, run dates, and hit counts.
- [`repro_kit/screening/screening_sheet.csv`](repro_kit/screening/screening_sheet.csv):
  title/abstract and full-text screening decisions.
- [`repro_kit/quality/qa_rubric.md`](repro_kit/quality/qa_rubric.md):
  QA1-QA8 scoring rubric and calibration notes.
- [`repro_kit/quality/compute_kappa.py`](repro_kit/quality/compute_kappa.py):
  helper script for inter-rater agreement.
- [`repro_kit/extraction/extraction_matrix.csv`](repro_kit/extraction/extraction_matrix.csv):
  final per-study extraction matrix.
- [`repro_kit/extraction/search_log.csv`](repro_kit/extraction/search_log.csv):
  funnel summary by source, stage, and exclusion reason.
- [`repro_kit/PRISMA_map.md`](repro_kit/PRISMA_map.md): PRISMA 2020 item map.
- [`repro_kit/protocol.md`](repro_kit/protocol.md): dated review protocol.
- [`repro_kit/PRISMA_2020_checklist.pdf`](repro_kit/PRISMA_2020_checklist.pdf):
  draft PRISMA checklist PDF.
- [`scripts/run_searches.py`](scripts/run_searches.py): public/API search runner
  for sources that can be queried without a manual export.

## Auditing The Central Claim

After replacing the template data with completed review data:

1. Open [`repro_kit/extraction/extraction_matrix.csv`](repro_kit/extraction/extraction_matrix.csv).
2. Filter for rows where `QA5_faithfulness == 1`.
3. Check whether any of those rows also vary the trajectory self-feedback
   mechanism with `operator_family != none`.
4. The stated claim is falsified if that intersection is non-empty.

## Inter-Rater Agreement

Run the quality agreement calculation from the repository root:

```bash
python3 repro_kit/quality/compute_kappa.py --mode qa \
  repro_kit/quality/qa_rater1.csv \
  repro_kit/quality/qa_rater2.csv
```

Run screening agreement with:

```bash
python3 repro_kit/quality/compute_kappa.py --mode screening \
  repro_kit/screening/screening_sheet.csv
```

Do not report the output from the template files as study results. Re-run these
commands after the real independent coding and screening data are in place.

## Double-Blind Use

Do not cite a named GitHub repository during double-blind review. Use an
anonymous.4open.science mirror or an anonymized OSF view-only link for review,
then move the finalized artifacts to a named repository for camera-ready release.
