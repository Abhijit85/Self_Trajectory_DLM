# Manual Export Instructions

Run these searches on the same date as the review search, save the raw export
files at the paths listed in `query_log.csv`, and then enter the raw hit counts
in that file.

Use this Boolean query unless the database syntax requires minor escaping:

```text
"diffusion language model" OR "discrete diffusion" OR "text diffusion" OR "self-feedback" OR "trajectory self-feedback"
```

## OpenReview

- Scope: ICLR, NeurIPS, ICML, and ACL-family venues for 2021-2026.
- Search field: title, abstract, and keywords where available.
- Export path: `search/exports/openreview.csv`.
- Required columns: title, authors, year, venue, URL, abstract, DOI/arXiv ID if available.

## ACL Anthology

- Scope: full-text or metadata search, 2021-2026.
- Export path: `search/exports/acl.bib`.
- Save the raw BibTeX export without editing.

## IEEE Xplore

- Scope: metadata and abstract fields, 2021-2026.
- Export path: `search/exports/ieee.csv`.
- Save the platform CSV export without editing.

## ACM Digital Library

- Scope: metadata and abstract fields, 2021-2026.
- Export path: `search/exports/acm.bib`.
- Save the raw BibTeX export without editing.

