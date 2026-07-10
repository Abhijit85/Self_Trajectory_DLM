"""Scoring primitives shared by the CI, CCap, and faithfulness harnesses."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence


# --------------------------------------------------------------------------- #
# Token-level edit distance (used by the Commitment Index)                     #
# --------------------------------------------------------------------------- #
def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Levenshtein distance over token-id sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


def normalized_edit_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """Edit distance normalized to [0, 1] by the longer sequence."""
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return edit_distance(a, b) / denom


# --------------------------------------------------------------------------- #
# Position accounting (used by Correction Capacity)                            #
# --------------------------------------------------------------------------- #
def repair_counts(
    reference: Sequence[int],
    corrupted: Sequence[int],
    refined: Sequence[int],
) -> tuple[int, int, int]:
    """Count repaired and newly-broken positions after a refinement pass."""
    L = min(len(reference), len(corrupted), len(refined))
    delta_correct = delta_regress = n_corrupted = 0
    for i in range(L):
        was_wrong = corrupted[i] != reference[i]
        now_wrong = refined[i] != reference[i]
        n_corrupted += int(was_wrong)
        if was_wrong and not now_wrong:
            delta_correct += 1
        if not was_wrong and now_wrong:
            delta_regress += 1
    return delta_correct, delta_regress, n_corrupted


# --------------------------------------------------------------------------- #
# Text QA metrics (used by the faithfulness axis)                              #
# --------------------------------------------------------------------------- #
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Iterable[str]) -> int:
    p = _normalize(pred)
    return int(any(p == _normalize(g) for g in golds))


def contains_match(pred: str, golds: Iterable[str]) -> int:
    """PopQA-style alias match: any normalized gold alias appears in the answer."""
    p = _normalize(pred)
    if not p:
        return 0
    padded = f" {p} "
    for g in golds:
        ng = _normalize(g)
        if ng and f" {ng} " in padded:
            return 1
    return 0


def token_f1(pred: str, golds: Iterable[str]) -> float:
    p = _normalize(pred).split()
    best = 0.0
    for g in golds:
        gt = _normalize(g).split()
        if not p or not gt:
            best = max(best, float(p == gt))
            continue
        common = Counter(p) & Counter(gt)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        prec = overlap / len(p)
        rec = overlap / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def is_hallucination(pred: str, golds: Iterable[str]) -> bool:
    """A confident-but-wrong answer: non-empty, non-abstaining, and not a match."""
    p = _normalize(pred)
    if not p or p in {"i dont know", "unknown", "na", "none"}:
        return False
    return contains_match(pred, golds) == 0
