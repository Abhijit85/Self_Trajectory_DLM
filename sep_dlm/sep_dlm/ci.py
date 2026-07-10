"""Commitment Index (CI) harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .adapters import DiffusionLM, Hooks
from .metrics import normalized_edit_distance


@dataclass
class CIResult:
    t0_grid: List[int]
    ci_curve: List[float]
    ci_std: List[float]
    sigma: float
    per_prompt: Dict[int, List[float]]
    ci_mode: str

    def summary(self) -> dict:
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        return {
            "sigma": self.sigma,
            "ci_mode": self.ci_mode,
            "t0_grid": self.t0_grid,
            "ci_curve": [round(c, 4) for c in self.ci_curve],
            "ci_std": [round(c, 4) for c in self.ci_std],
            "auc": round(float(_trapz(self.ci_curve, self.t0_grid) / max(self.t0_grid[-1], 1)), 4),
        }


def _reference(model, prompt, gen_len, steps, seed):
    return model.generate(prompt, gen_len, steps, tsf=True, hooks=Hooks.none(seed=seed)).token_ids


def _intervened(model, prompt, gen_len, steps, t0, sigma, seed, ci_mode):
    if ci_mode == "flip":
        hooks = Hooks(hard_flip_steps=frozenset({t0}), logit_noise={t0: sigma} if sigma > 0 else {}, seed=seed)
    elif ci_mode == "noise":
        hooks = Hooks(sc_off_steps=frozenset({t0}), logit_noise={t0: sigma}, seed=seed)
    else:
        raise ValueError(ci_mode)
    return model.generate(prompt, gen_len, steps, tsf=True, hooks=hooks).token_ids


def commitment_index(
    model: DiffusionLM,
    prompts: Sequence[str],
    gen_len: int = 24,
    steps: int = 16,
    t0_grid: Optional[Sequence[int]] = None,
    sigma: float = 1.0,
    reps: int = 4,
    base_seed: int = 0,
    ci_mode: str = "flip",
) -> CIResult:
    if t0_grid is None:
        t0_grid = list(range(0, steps, max(1, steps // 8)))
    t0_grid = list(t0_grid)

    per_prompt: Dict[int, List[float]] = {t0: [] for t0 in t0_grid}
    for pi, prompt in enumerate(prompts):
        y_ref = _reference(model, prompt, gen_len, steps, seed=base_seed + pi)
        for t0 in t0_grid:
            divs = []
            for r in range(reps):
                y_pert = _intervened(model, prompt, gen_len, steps, t0, sigma, seed=base_seed + 1000 * (r + 1) + pi, ci_mode=ci_mode)
                divs.append(normalized_edit_distance(y_ref, y_pert))
            per_prompt[t0].append(float(np.mean(divs)))

    curve = [float(np.mean(per_prompt[t0])) for t0 in t0_grid]
    std = [float(np.std(per_prompt[t0])) for t0 in t0_grid]
    return CIResult(t0_grid, curve, std, sigma, per_prompt, ci_mode)


def calibrate_sigma(
    model: DiffusionLM,
    prompts: Sequence[str],
    gen_len: int = 24,
    steps: int = 16,
    sigma_grid: Optional[Sequence[float]] = None,
    reps: int = 3,
    base_seed: int = 0,
    ci_mode: str = "flip",
) -> dict:
    if sigma_grid is None:
        sigma_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    sigma_grid = list(sigma_grid)
    t0 = max(1, steps // 2)

    table = []
    for sigma in sigma_grid:
        divs = []
        for pi, prompt in enumerate(prompts):
            y_ref = _reference(model, prompt, gen_len, steps, seed=base_seed + pi)
            for r in range(reps):
                y_pert = _intervened(model, prompt, gen_len, steps, t0, sigma, seed=base_seed + 1000 * (r + 1) + pi, ci_mode=ci_mode)
                divs.append(normalized_edit_distance(y_ref, y_pert))
        table.append({"sigma": sigma, "ci": float(np.mean(divs)), "ci_std": float(np.std(divs))})

    ci_max = max(row["ci"] for row in table) or 1.0
    best, best_slope = table[min(1, len(table) - 1)], -1.0
    for i in range(1, len(table)):
        d_ci = table[i]["ci"] - table[i - 1]["ci"]
        d_s = (table[i]["sigma"] - table[i - 1]["sigma"]) or 1e-9
        slope = d_ci / d_s
        if table[i]["ci"] < 0.9 * ci_max and slope > best_slope:
            best_slope, best = slope, table[i]
    return {
        "t0": t0,
        "ci_mode": ci_mode,
        "table": table,
        "recommended_sigma": best["sigma"],
        "recommended_ci": round(best["ci"], 4),
    }
