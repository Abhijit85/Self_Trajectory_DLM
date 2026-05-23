"""Corruption processes for Correction Capacity (Appendix D, CCap protocol).

Two families, reported SEPARATELY because they probe different repair abilities:
  - lexical: random token substitutions/deletions at rate eps (surface repair).
  - semantic: replace a span with a plausible-but-wrong alternative from a distractor
    set (knowledge-grounded repair). For multilingual use, supply language-specific
    distractor sets (see README multilingual note).
"""
from __future__ import annotations
import random
from typing import Optional, Sequence


def corrupt(x_star: Sequence[int], *, eps: float, seed: int, kind: str,
            vocab: int = 50, distractors: Optional[dict] = None) -> list[int]:
    rng = random.Random(seed)
    out = list(x_star)
    n = len(out)
    k = max(1, int(round(eps * n)))
    if kind == "lexical":
        idx = rng.sample(range(n), min(k, n))
        for i in idx:
            if rng.random() < 0.5:                 # substitution
                out[i] = rng.randrange(vocab)
            else:                                   # deletion -> sentinel (-1)
                out[i] = -1
        return out
    if kind == "semantic":
        # replace one contiguous span (~k tokens) with a distractor span.
        start = rng.randrange(max(1, n - k))
        if distractors:
            span = rng.choice(list(distractors.values()))[:k]
        else:
            span = [rng.randrange(vocab) for _ in range(k)]
        out[start:start + len(span)] = span
        return out
    raise ValueError(f"unknown corruption kind: {kind}")
