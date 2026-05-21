# Dated Review Protocol

Protocol date: 2026-05-21

Review title: Systematic literature review of trajectory self-feedback in
diffusion language models.

## Objective

Audit how trajectory self-feedback is defined, varied, measured, and validated
in diffusion language model research, with particular attention to whether
faithfulness or correction is evaluated when the trajectory self-feedback
mechanism changes.

## Research Questions

- RQ1: Which forms of trajectory self-feedback are used in diffusion language
  models?
- RQ2: Which operator families or mechanism variants are studied?
- RQ3: Which metrics are used to evaluate these systems?
- RQ4: Which hidden assumptions are made about trajectory quality or
  independent-step optimality?
- RQ5: Which studies measure faithfulness or correction?
- RQ6: Which open problems remain after mapping the evidence?

## Information Sources

Searches are planned for arXiv, Semantic Scholar, OpenReview, ACL Anthology,
IEEE Xplore, and ACM Digital Library. Exact query strings, run dates, raw hit
counts, and export paths are recorded in `search/query_log.csv`.

## Eligibility Criteria

Include archival or preprint studies from 2021-2026 that study diffusion
language models, discrete/text diffusion, or closely related trajectory
self-feedback mechanisms for text generation or reasoning.

Exclude studies that are image/audio/video-only diffusion, purely autoregressive
without diffusion-language-model relevance, non-archival/tutorial/blog material,
superseded duplicates, surveys used only as related work, unavailable full text,
or out-of-scope applications.

## Screening Procedure

Two screeners independently code title/abstract decisions as `include` or
`exclude` in `screening/screening_sheet.csv`. Disagreements are reconciled
before full-text screening. Full-text exclusions must include an exclusion code
from the sheet's column guide.

## Quality Assessment

Two raters independently score included studies with `quality/qa_rubric.md`.
Inter-rater agreement is computed with `quality/compute_kappa.py`. Consensus
scores are then written into `extraction/extraction_matrix.csv`.

## Extraction Fields

The extraction matrix records study, cluster, trajectory self-feedback form,
operator family, key metrics, central hidden assumption, blind spot, open
problems, QA1-QA8 scores, and total QA score.

## Synthesis Plan

The synthesis is narrative and tabular. The central audit checks whether any
included study both measures faithfulness (`QA5_faithfulness == 1`) and varies
the trajectory self-feedback mechanism (`operator_family != none`).

## Versioning

This protocol should be committed before executing the final review searches.
For double-blind review, mirror the committed protocol through an anonymized
OSF or anonymous.4open.science link.
