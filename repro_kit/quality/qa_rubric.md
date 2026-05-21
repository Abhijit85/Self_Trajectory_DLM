# Quality-Assessment Rubric (QA1–QA8)

Each criterion is scored **0 (no) / 0.5 (partial) / 1 (yes)**. Maximum = 8.
Score against what the paper *reports*, not what it could have done. When unsure
between two levels, pick the lower and log the reason in the `notes` column of
`extraction/extraction_matrix.csv`.

These criteria are deliberately split between general study rigour (QA1, QA4, QA8)
and TSF-specific reporting (QA2, QA3, QA5, QA6, QA7).

---

### QA1 — Are the aims / research questions clearly stated?
- **1** — Explicit aim or research question(s) in the abstract or intro.
- **0.5** — Aim inferable but never stated as such.
- **0** — No discernible aim.
- *Calibration:* MDLM states its objective (simplified masked-diffusion ELBO) explicitly → **1**.

### QA2 — Is the self-feedback mechanism `g` precisely defined?
- **1** — The fed-back signal and how it enters the denoiser are given formally or in
  reproducible detail (equation, architecture diagram, or pseudocode).
- **0.5** — Mechanism described in prose only; a reimplementer would have to guess.
- **0** — No self-feedback mechanism, or it is mentioned without definition.
- *Calibration:* Analog Bits gives the concatenation operation explicitly → **1**;
  a paper that says "we add self-conditioning" with no detail → **0.5**.

### QA3 — Is there a baseline or ablation isolating the TSF effect?
- **1** — A controlled comparison with vs. without the mechanism, others held fixed.
- **0.5** — Some comparison, but confounded (e.g., different training budget or data).
- **0** — No baseline isolating the effect.
- *Calibration:* TREC ablates reward-weighting on/off → **1**; "ablation" admits the
  most borderline cases, so this is the lowest-agreement criterion.

### QA4 — Are tasks, datasets, and metrics described reproducibly?
- **1** — Tasks, datasets (with versions/splits), and metrics all specified.
- **0.5** — Partial (e.g., metric named but dataset split unspecified).
- **0** — Cannot tell what was evaluated.

### QA5 — Is faithfulness / hallucination measured?
- **1** — Reports a faithfulness or hallucination metric (FactScore, hallucination
  rate, factual-QA accuracy, FEVER-style) on a knowledge-intensive task.
- **0.5** — Touches factuality only indirectly (e.g., a single qualitative example).
- **0** — Not measured.
- *Calibration:* Only the C5 hallucination studies score 1 here. **This column is the
  numerator of the central finding — score it strictly.**

### QA6 — Is correction capacity measured?
- **1** — Reports the model's ability to repair injected/observed errors during
  generation (correction rate, repair@K, provable-correction result).
- **0.5** — Discusses correction but reports no measurement.
- **0** — Not measured.
- *Calibration:* Only the C4 (CTF) studies score 1 here.

### QA7 — Is the family scope stated, and cross-family generality discussed?
- **1** — States whether the work is discrete / continuous / flow AND discusses (even
  briefly) whether the result should transfer to other families.
- **0.5** — States the family but makes no generality claim.
- **0** — Family scope ambiguous.

### QA8 — Are limitations / threats to validity discussed?
- **1** — A dedicated limitations discussion.
- **0.5** — Limitations mentioned in passing.
- **0** — None.

---

## How to use this with two raters
1. Both raters score the SAME random ~40% of the corpus independently using this rubric.
2. Compute agreement with `quality/compute_kappa.py` (quadratic-weighted Cohen's κ,
   since scores are ordinal 0/0.5/1).
3. If κ < 0.7 on any criterion, refine that criterion's wording here, add a calibration
   example for the disputed case, and re-score.
4. Reconcile remaining disagreements to consensus; record the final scores in
   `extraction/extraction_matrix.csv`.
5. Report the achieved κ (and range across criteria) in §3.5 of the paper.
