# Self_Trajectory_DLM

This repository contains reproducibility materials for a systematic literature
review of trajectory self-feedback in deep learning models.

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

## Auditing The Central Claim

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

Current results:

- QA agreement: mean quadratic-weighted kappa = `0.892`, range `0.558-1.000`
  over `24` studies.
- Screening agreement: Cohen's kappa = `1.000`, raw agreement = `100.0%`
  over `2` records.
- Central-claim audit: `24` studies checked; `0` rows have both
  `QA5_faithfulness == 1` and `operator_family != none`.

## Status

The current CSVs in `repro_kit/` are the repository's active extraction and
search-log artifacts. Re-run the commands above after any changes to the
screening, quality, or extraction files.
