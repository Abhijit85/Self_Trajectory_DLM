# Reliability Summary

Screening stage: automated title/abstract pre-screen; no human screening kappa
is reported.

Quality-assessment agreement over 24 final human-coded primary studies:
mean quadratic-weighted kappa = 0.892, range 0.558-1.000.

- QA1: kappa_w = 1.000 (raw 100%)
- QA2: kappa_w = 1.000 (raw 100%)
- QA3: kappa_w = 0.579 (raw 67%)
- QA4: kappa_w = 1.000 (raw 100%)
- QA5: kappa_w = 1.000 (raw 100%)
- QA6: kappa_w = 1.000 (raw 100%)
- QA7: kappa_w = 0.558 (raw 75%)
- QA8: kappa_w = 1.000 (raw 100%)

Mean kappa_w = 0.892; range = 0.558-1.000. QA3 and QA7 remained the
lowest-agreement criteria and should be described as reconciled by consensus
before final values were written to `extraction/extraction_matrix.csv`.

Operator-family labels were checked separately from the ordinal QA rubric using
the archived reviewer files:

```bash
python3 repro_kit/quality/compute_kappa.py --mode operator \
  repro_kit/quality/reviewer_a_mechanism_reviews.csv \
  repro_kit/quality/reviewer_b_evidence_reviews.csv
```

The archived final reviewer files give unweighted Cohen's kappa = 1.000 over 24
studies (raw agreement 100%) for `operator_family`.
