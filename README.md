# Self_Trajectory_DLM

This repository contains reproducibility materials for a systematic literature
review of trajectory self-feedback in diffusion language models.

## Review Audit Status

The repository audit trail is the source of truth for the linked reproducibility
kit. Current counts are:

- `1,595` imported records from arXiv, Semantic Scholar, OpenReview, and ACL
  Anthology.
- `1,168` unique records after deduplication and removal of prior IEEE/ACM
  template records.
- `415` records excluded by automated title/abstract pre-screening.
- `753` records retained for human full-text curation.
- `729` records excluded during human full-text curation.
- `24` final human-coded primary studies.

IEEE Xplore and ACM Digital Library were searched manually because public API or
export access was restricted/unavailable. IEEE contributed no imported records;
ACM is recorded as a manual count of `45`, but no ACM export is included or
imported into the deduplicated screening set. The extraction and QA CSVs now
contain the final 24 human-coded primary studies.

## SEP-DLM Scaffold

The runnable SEP-DLM diagnostic scaffold is in [`sep_dlm/`](sep_dlm/). It can
write CI(t) and CCap CSVs with bootstrap confidence intervals, includes adapter
stubs for discrete and continuous checkpoints, and provides a harness check for
the H_wall trajectory shape before real checkpoint results are produced. Do not
commit or report harness-check CSVs; Section 7.4 should use only CSVs generated
from real checkpoint adapters.

## Repository Contents

The main audit trail is in [`repro_kit/`](repro_kit/):

- [`repro_kit/search/query_log.csv`](repro_kit/search/query_log.csv): source queries, run dates, hit counts, and source-status notes.
- [`repro_kit/extraction/search_log.csv`](repro_kit/extraction/search_log.csv): reconciled funnel summary for the linked repository audit trail.
- [`repro_kit/screening/screening_sheet.csv`](repro_kit/screening/screening_sheet.csv): 1,168 imported unique records; 24 rows are marked `included_final=yes` and linked to the final extraction by `source_record_id`.
- [`repro_kit/extraction/extraction_matrix.csv`](repro_kit/extraction/extraction_matrix.csv): final 24 human-coded primary studies.
- [`repro_kit/quality/qa_rater1.csv`](repro_kit/quality/qa_rater1.csv) and [`repro_kit/quality/qa_rater2.csv`](repro_kit/quality/qa_rater2.csv): independent human QA ratings for the 24 final studies.
- [`repro_kit/quality/qa_rubric.md`](repro_kit/quality/qa_rubric.md): QA1-QA8 scoring rubric.
- [`repro_kit/protocol.md`](repro_kit/protocol.md): dated review protocol and current source-status caveats.
- [`scripts/run_searches.py`](scripts/run_searches.py): public/API search runner.

Use the final 24-row extraction and QA files as the human-coded study corpus. The 753 full-text-retained count remains part of the funnel, not the final corpus.

## Double-Blind Use

Do not cite a named GitHub repository during double-blind review. Use an anonymous.4open.science mirror or an anonymized OSF view-only link for review, then move finalized artifacts to a named repository for camera-ready release.
