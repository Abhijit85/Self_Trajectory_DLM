"""Corruption-injection eval.

Two corruption channels, reported separately (this is the recipe reviewer CKdF
asked to see spelled out):

  * lexical   -- replace a fraction eps of tokens with random in-vocab tokens.
                 Surface-level noise; tests whether the model repairs typos /
                 token-level damage.
  * semantic  -- replace CONTENT tokens with *type-matched distractors* (an
                 entity token for another entity token, a number for another
                 number). Tests whether the model repairs meaning-altering
                 damage -- the channel that actually bears on faithfulness.

Both operate in TOKEN space so that repaired/broken positions can be counted
exactly against the reference (see metrics.repair_counts). A text-level
NER-based variant (`semantic_corrupt_text`) is also provided for the long-form
faithfulness setting where exact position accounting is not required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

import numpy as np


@dataclass
class Corruption:
    ids: List[int]
    changed: List[int]
    channel: str


def _n_to_corrupt(n_positions: int, eps: float) -> int:
    if eps <= 0 or n_positions == 0:
        return 0
    return max(1, int(round(eps * n_positions)))


def lexical_corrupt(ids: Sequence[int], eps: float, vocab_size: int,
                    rng: np.random.Generator,
                    protect: Optional[Set[int]] = None,
                    reserved_below: int = 2) -> Corruption:
    """Replace ~eps of positions with random tokens != the original."""
    protect = protect or set()
    ids = list(ids)
    candidates = [i for i in range(len(ids)) if i not in protect]
    k = _n_to_corrupt(len(candidates), eps)
    if k == 0:
        return Corruption(ids, [], "lexical")
    picks = rng.choice(candidates, size=k, replace=False)
    for pos in picks:
        new = int(rng.integers(reserved_below, vocab_size))
        if new == ids[pos]:
            new = reserved_below + (new + 1) % (vocab_size - reserved_below)
        ids[pos] = new
    return Corruption(ids, sorted(int(p) for p in picks), "lexical")


def semantic_corrupt(ids: Sequence[int], eps: float,
                     content_positions: Sequence[int],
                     distractor_pool: Sequence[int],
                     rng: np.random.Generator) -> Corruption:
    """Replace ~eps of content positions with matched distractor tokens."""
    ids = list(ids)
    content = [p for p in content_positions if 0 <= p < len(ids)]
    k = _n_to_corrupt(len(content), eps)
    if k == 0 or len(distractor_pool) == 0:
        return Corruption(ids, [], "semantic")
    picks = rng.choice(content, size=min(k, len(content)), replace=False)
    for pos in picks:
        new = int(rng.choice(distractor_pool))
        if new == ids[pos] and len(distractor_pool) > 1:
            new = int(rng.choice([d for d in distractor_pool if d != ids[pos]]))
        ids[pos] = new
    return Corruption(ids, sorted(int(p) for p in picks), "semantic")


def content_token_positions(ids: Sequence[int], mask_id: int,
                            reserved_below: int = 2) -> List[int]:
    """Cheap content-token heuristic when no tokenizer/NER is available."""
    return [i for i, t in enumerate(ids)
            if t != mask_id and t >= reserved_below]


_CAP = re.compile(r"\b([A-Z][a-zA-Z]+)\b")
_NUM = re.compile(r"\b\d+\b")


def semantic_corrupt_text(text: str, eps: float, rng: np.random.Generator,
                          entity_pool: Optional[dict] = None,
                          use_spacy: bool = False) -> tuple[str, int]:
    """Meaning-altering substitution in raw text."""
    spans = []
    if use_spacy:
        try:
            import spacy
            nlp = semantic_corrupt_text._nlp or spacy.load("en_core_web_sm")
            semantic_corrupt_text._nlp = nlp
            for ent in nlp(text).ents:
                spans.append((ent.start_char, ent.end_char, ent.label_, ent.text))
        except Exception:
            use_spacy = False
    if not use_spacy:
        for m in _CAP.finditer(text):
            spans.append((m.start(), m.end(), "CAP", m.group()))
        for m in _NUM.finditer(text):
            spans.append((m.start(), m.end(), "NUM", m.group()))

    if not spans:
        return text, 0
    spans.sort()
    k = _n_to_corrupt(len(spans), eps)
    chosen = sorted(rng.choice(len(spans), size=k, replace=False))
    default_pool = {
        "CAP": ["Paris", "Einstein", "Toyota", "Amazon"],
        "NUM": ["1999", "42", "2020", "7"],
        "PERSON": ["Einstein", "Napoleon"],
        "GPE": ["Paris", "Tokyo"],
    }
    pool = entity_pool or default_pool
    out, last = [], 0
    subs = 0
    for idx, (s, e, typ, surf) in enumerate(spans):
        if idx not in chosen:
            continue
        cands = pool.get(typ) or pool.get("CAP") or ["something"]
        repl = next((c for c in rng.permutation(cands) if c != surf), cands[0])
        out.append(text[last:s])
        out.append(str(repl))
        last = e
        subs += 1
    out.append(text[last:])
    return "".join(out), subs


semantic_corrupt_text._nlp = None
