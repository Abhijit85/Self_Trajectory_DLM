"""SEP-DLM reference diagnostics: Commitment Index (CI) and Correction Capacity (CCap).

Implements Eq. (2) and Eq. (3) of the paper plus the Appendix D protocol choices.
These are model-AGNOSTIC: they call a `DLMAdapter` (see adapters.py). Plug in a real
discrete / continuous / flow checkpoint and the same code produces comparable numbers.

NOTHING here invents results. Run it on real checkpoints; the mock adapters exist only
to test that the harness runs.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from statistics import mean
from typing import Sequence


# ----------------------------- helpers -------------------------------------- #
def norm_levenshtein(a: Sequence[int], b: Sequence[int]) -> float:
    """Token-level Levenshtein distance, normalised by max(len) -> [0, 1].

    Appendix D (i): edit-distance alignment, NOT positional; normalised by
    max(|y_ref|, |y|) so D_edit in [0, 1].
    """
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def bootstrap_ci(values: Sequence[float], *, iters: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, lo, hi) with a percentile bootstrap CI over `values`."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"),) * 3
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (sum(values) / n, lo, hi)


# ------------------------- Commitment Index (Eq. 2) ------------------------- #
@dataclass
class CIResult:
    t0: float            # fraction of trajectory in [0, 1]
    ci: float            # mean normalised edit distance
    lo: float
    hi: float
    n_seeds: int


def commitment_index(adapter, prompt, *, t_grid: Sequence[float],
                     steps: int, n_seeds: int = 200) -> list[CIResult]:
    """CI(t) curve (Eq. 2). For each t0 in `t_grid`, measure how much the FINAL
    output diverges when the self-feedback channel is ablated + state perturbed at
    step t0 only, vs. the reference (self-feedback on throughout). Held-fixed
    reference per seed; perturbation scale comes from the adapter's native space.
    """
    if n_seeds < 200:
        print(f"[CI] WARNING: n_seeds={n_seeds} < 200 recommended in Appendix D.")
    out: list[CIResult] = []
    for t0 in t_grid:
        edits = []
        for s in range(n_seeds):
            y_ref = adapter.rollout(prompt, steps=steps, seed=s, self_feedback=True)
            y_cf = adapter.rollout(prompt, steps=steps, seed=s, self_feedback=True,
                                   ablate_sf_at=t0,
                                   perturbation=adapter.perturbation_scale())
            edits.append(norm_levenshtein(y_ref, y_cf))
        m, lo, hi = bootstrap_ci(edits, seed=hash((str(t0), n_seeds)) & 0xFFFF)
        out.append(CIResult(t0=t0, ci=m, lo=lo, hi=hi, n_seeds=n_seeds))
    return out


# ----------------------- Correction Capacity (Eq. 3) ------------------------ #
@dataclass
class CCapResult:
    eps: float
    K: int
    corruption: str
    ccap: float
    lo: float
    hi: float


def _repairs(x_star, x_corrupt, y) -> tuple[int, int]:
    """Count repaired and regressed positions via the same edit alignment used by CI.
    A position counts as repaired iff it matched x_star after K steps and did not
    before; regressed iff it matched before and no longer does. Uses positional
    comparison after truncating/padding to len(x_star) for simplicity; swap in a
    proper alignment for production.
    """
    n = len(x_star)
    y = list(y)[:n] + [None] * max(0, n - len(y))
    xc = list(x_corrupt)[:n] + [None] * max(0, n - len(x_corrupt))
    repaired = sum(1 for i in range(n) if xc[i] != x_star[i] and y[i] == x_star[i])
    regressed = sum(1 for i in range(n) if xc[i] == x_star[i] and y[i] != x_star[i])
    return repaired, regressed


def correction_capacity(adapter, examples, *, eps: float, K: int,
                        corruption: str, corrupt_fn, n_seeds: int = 200) -> CCapResult:
    """CCap(eps, K) (Eq. 3) for one corruption family. `corrupt_fn(x_star, eps, seed,
    kind)` returns a corrupted token list; `adapter.repair(x_corrupt, budget=K)` runs
    K refinement steps and returns the repaired tokens.
    """
    vals = []
    for s in range(n_seeds):
        ex = examples[s % len(examples)]
        x_star = ex
        x_corrupt = corrupt_fn(x_star, eps=eps, seed=s, kind=corruption)
        y = adapter.repair(x_corrupt, budget=K, seed=s)
        rep, reg = _repairs(x_star, x_corrupt, y)
        denom = max(1, int(round(eps * len(x_star))))
        vals.append((rep - reg) / denom)
    m, lo, hi = bootstrap_ci(vals, seed=hash((corruption, eps, K)) & 0xFFFF)
    return CCapResult(eps=eps, K=K, corruption=corruption, ccap=m, lo=lo, hi=hi)
