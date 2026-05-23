# SEP-DLM Reference Scaffold

Minimal, model-agnostic implementation of the two diagnostics and the evaluation
protocol from the paper: **Commitment Index (CI, Eq. 2)** and **Correction Capacity
(CCap, Eq. 3)**, plus corruption processes and a runnable harness. Releasing this
addresses the reviewer's "minimal SEP-DLM scaffold (code stub, configs, seeds)" request
and is what makes the H_wall probe runnable.

```
sep_dlm/
  diagnostics.py      CI(t) and CCap(eps,K) + normalised Levenshtein + bootstrap CIs
  adapters.py         DLMAdapter interface, MOCK discrete/continuous, real-checkpoint stubs
  corruption.py       lexical + semantic corruption for CCap
  run_diagnostics.py  CLI -> writes ci_curve.csv, ccap.csv
  plot_diagnostics.py CLI -> plots ci_curve.{pdf,png}, ccap_curves.{pdf,png}
examples/
  external_adapter_template.py  JSON stdin/stdout wrapper template for real checkpoints
configs/default.yaml  fixed seeds, t/eps/K grids, perturbation scales
```

## Section 7.4 Policy

Do not put either harness-check curve, CI or CCap, into Section 7.4 or the
repository. Section 7.1 may frame the CI shape as illustrative and hypothetical,
but Section 7.4 should stay empty of numerical curves until real checkpoint
adapters are implemented. The real curve comes from implementing `rollout` and
`repair` for an actual discrete checkpoint, such as MDLM or LLaDA, and an actual
continuous checkpoint, then rerunning the harness.

When those real CSVs exist, use them to build the CI(t) and CCap(eps,K) figures
and write the Section 7.4 paragraph around the observed numbers. If the real
discrete and continuous curves do not differ, report that directly; it is still
an interesting result and would partly falsify H_wall.

## Run (harness test, mock)
```
cd sep_dlm
python3 -m sep_dlm.run_diagnostics --mock --n-seeds 200 --out /tmp/sep_dlm_harness_check
```
This writes `/tmp/sep_dlm_harness_check/ci_curve.csv` and
`/tmp/sep_dlm_harness_check/ccap.csv`. The mock prints a warning: its numbers are
a HARNESS TEST ONLY and must never be reported as results. Keep generated
harness-check files out of the repository; `sep_dlm/results/` is ignored as a
backstop.

## Plot Figures
Install the plotting dependency first if your environment does not already have
Matplotlib:

```
python3 -m pip install -r requirements.txt
```

```
cd sep_dlm
python3 -m sep_dlm.plot_diagnostics \
  --ci /path/to/real/ci_curve.csv \
  --ccap /path/to/real/ccap.csv \
  --out /path/to/real/figs/
```
This emits `ci_curve.{pdf,png}` and `ccap_curves.{pdf,png}` from whichever CSVs
you point it at. Figures from harness-check CSVs are still harness-check figures;
do not commit or report them. `figs/` is ignored as a backstop.

## Run for real (the actual experiment)
1. Copy `examples/external_adapter_template.py` once for the discrete checkpoint
   and once for the continuous checkpoint.
2. In each copy, load the real checkpoint and implement `rollout_real_model` and
   `repair_real_model`.
3. Prepare a JSONL file of CCap reference examples. Each line is either a token-id
   list or an object with a `tokens` field, for example:

```json
{"tokens": [101, 1996, 3007, 2003, 102]}
```

4. Run the real adapters:

```bash
cd sep_dlm
python3 -m sep_dlm.run_diagnostics \
  --out /path/to/real/results \
  --prompt "the capital of france is" \
  --examples /path/to/real/examples.jsonl \
  --discrete-name MDLM_or_LLaDA \
  --discrete-cmd "python3 /path/to/discrete_adapter.py" \
  --continuous-name continuous_DLM \
  --continuous-cmd "python3 /path/to/continuous_adapter.py"
```

The output directory will contain `ci_curve.csv`, `ccap.csv`, and
`manifest.json`. Those CSVs are the reportable inputs for the figure step only
when both commands wrap actual checkpoints.

> ⚠️ **Integrity:** the only legitimate CI/CCap numbers are the ones a real run produces.
> Do not hand-edit the CSVs or substitute the mock output. A reviewer at high confidence
> will rerun this code; fabricated numbers are misconduct, not a shortcut.

## What to report (the H_wall probe, §7.4 study 1)
- CI(t) for one discrete and one continuous backbone at matched compute. H_wall predicts
  the discrete curve rises steeply and early; the continuous one late. Plot both with the
  bootstrap CIs from the CSV. The *opposite* pattern falsifies H_wall — report it either way.
- CCap(eps,K) for both, lexical and semantic families separately, as curves over eps and K.

## Multilingual note (reviewer weakness #5)
CI's token-Levenshtein and CCap's semantic corruption are language-sensitive:
- For morphologically rich languages / non-space scripts, report CI on a fixed
  subword tokenizer per script family, and add an **embedding-based semantic delta** as a
  secondary CI plot so paraphrastic changes are not over-penalised (`diagnostics.py` is
  structured so you can drop in a sentence-embedding distance alongside Levenshtein).
- For semantic CCap, supply **language-specific distractor sets** to `corrupt(...)` via
  the `distractors` argument rather than reusing English distractors.

## Reproducibility
All randomness flows from `configs/default.yaml:seed` and per-call seeds; bootstrap CIs
use a fixed iter count. Pin your checkpoint hashes and library versions in a `MANIFEST`
when you release real results.
