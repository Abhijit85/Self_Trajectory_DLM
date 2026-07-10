"""Correction Capacity (CCap) harness.

Definition (survey Eq. 3):

    CCap(eps, K) = E [ (Delta_correct - Delta_regress) / (eps * |x*|) ]

Given a correct reference x*, corrupt it at error rate eps, let the model refine
for K steps, then count positions repaired (Delta_correct) minus positions newly
broken (Delta_regress), normalized by the number of injected errors. CCap in
[-1, 1]: +1 = perfect repair with no collateral damage, 0 = no net repair,
negative = the refinement pass does more harm than good.

Reported separately for the lexical and semantic channels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .adapters import DiffusionLM, Hooks
from .corruptions import content_token_positions, lexical_corrupt, semantic_corrupt
from .metrics import repair_counts


@dataclass
class CCapResult:
    channel: str
    eps: float
    K: int
    ccap: float
    ccap_std: float
    delta_correct: float
    delta_regress: float
    n: int
    per_item: List[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "channel": self.channel,
            "eps": self.eps,
            "K": self.K,
            "CCap": round(self.ccap, 4),
            "CCap_std": round(self.ccap_std, 4),
            "mean_repaired": round(self.delta_correct, 3),
            "mean_broken": round(self.delta_regress, 3),
            "n": self.n,
        }


def _distractor_pool(references: Sequence[Sequence[int]], mask_id: int,
                     reserved_below: int = 2, cap: int = 512) -> List[int]:
    """Type-matched distractor pool from content tokens in reference answers."""
    pool = set()
    for ref in references:
        for t in ref:
            if t != mask_id and t >= reserved_below:
                pool.add(int(t))
        if len(pool) >= cap:
            break
    return list(pool)


def correction_capacity(model: DiffusionLM, references: Sequence[Sequence[int]],
                        prompts: Sequence[str] | None = None,
                        eps: float = 0.15, K: int = 8, channel: str = "lexical",
                        reps: int = 1, base_seed: int = 0) -> CCapResult:
    """Measure CCap for one channel over a set of reference sequences."""
    pool = _distractor_pool(references, model.mask_id) if channel == "semantic" else []
    ccaps: List[float] = []
    tot_correct = tot_regress = 0.0
    n = 0
    for si, ref in enumerate(references):
        ref = list(ref)
        prompt = prompts[si] if prompts is not None else None
        for r in range(reps):
            rng = np.random.default_rng(base_seed + 97 * si + r)
            if channel == "lexical":
                corr = lexical_corrupt(ref, eps, model.vocab_size, rng)
            elif channel == "semantic":
                content = content_token_positions(ref, model.mask_id)
                corr = semantic_corrupt(ref, eps, content, pool, rng)
            else:
                raise ValueError(channel)
            if not corr.changed:
                continue
            refined = model.refine(
                corr.ids,
                K,
                hooks=Hooks(seed=base_seed + r),
                reference_ids=ref,
                prompt=prompt,
            ).token_ids
            dc, dr, ncor = repair_counts(ref, corr.ids, refined)
            denom = max(ncor, 1)
            ccaps.append((dc - dr) / denom)
            tot_correct += dc
            tot_regress += dr
            n += 1
    if not ccaps:
        return CCapResult(channel, eps, K, 0.0, 0.0, 0.0, 0.0, 0)
    return CCapResult(
        channel,
        eps,
        K,
        float(np.mean(ccaps)),
        float(np.std(ccaps)),
        tot_correct / n,
        tot_regress / n,
        n,
        ccaps,
    )


def ccap_sweep(model: DiffusionLM, references: Sequence[Sequence[int]],
               prompts: Sequence[str] | None = None,
               eps_grid=(0.05, 0.15, 0.30), K_grid=(4, 8, 16),
               channels=("lexical", "semantic"), base_seed: int = 0) -> List[dict]:
    """Grid over (channel, eps, K) for CCap sensitivity analysis."""
    rows = []
    for ch in channels:
        for eps in eps_grid:
            for K in K_grid:
                res = correction_capacity(model, references, prompts=prompts, eps=eps, K=K,
                                          channel=ch, base_seed=base_seed)
                rows.append(res.summary())
    return rows
