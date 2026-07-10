"""Dataset loading for the knowledge-intensive QA faithfulness axis.

Primary: PopQA (long-tail factual QA) or a TriviaQA / NQ-open subset via the
HuggingFace `datasets` hub. Offline fallback: a small bundled JSONL so the mock
pipeline runs with no network.
"""
from __future__ import annotations

import json
from typing import List


_FALLBACK = [
    {"question": "What is the capital of Australia?", "answers": ["Canberra"]},
    {"question": "Who wrote the novel 'Pride and Prejudice'?", "answers": ["Jane Austen"]},
    {"question": "What is the chemical symbol for gold?", "answers": ["Au"]},
    {"question": "In what year did the Berlin Wall fall?", "answers": ["1989"]},
    {"question": "Who painted the Mona Lisa?", "answers": ["Leonardo da Vinci", "Leonardo"]},
    {"question": "What is the largest planet in the Solar System?", "answers": ["Jupiter"]},
    {"question": "What language has the most native speakers?", "answers": ["Mandarin", "Mandarin Chinese", "Chinese"]},
    {"question": "Who developed the theory of general relativity?", "answers": ["Albert Einstein", "Einstein"]},
    {"question": "What is the tallest mountain above sea level?", "answers": ["Mount Everest", "Everest"]},
    {"question": "What is the currency of Japan?", "answers": ["Yen", "Japanese yen"]},
    {"question": "Which ocean is the largest?", "answers": ["Pacific", "Pacific Ocean"]},
    {"question": "Who is the author of 'The Origin of Species'?", "answers": ["Charles Darwin", "Darwin"]},
]


def load_qa(name: str = "popqa", n: int = 200, seed: int = 0) -> List[dict]:
    """Return a list of {"question": str, "answers": [str, ...]}."""
    del seed
    if name == "fallback":
        return _cycle(_FALLBACK, n)

    try:
        from datasets import load_dataset
        if name == "popqa":
            ds = load_dataset("akariasai/PopQA", split="test")
            rows = [{"question": r["question"],
                     "answers": _as_list(r.get("possible_answers") or r.get("obj"))}
                    for r in ds.select(range(min(n * 3, len(ds))))]
        elif name == "triviaqa":
            ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
            rows = [{"question": r["question"],
                     "answers": list(dict.fromkeys(r["answer"]["aliases"]
                                                   + [r["answer"]["value"]]))}
                    for r in ds.select(range(min(n * 3, len(ds))))]
        elif name == "nq_open":
            ds = load_dataset("google-research-datasets/nq_open", split="validation")
            rows = [{"question": r["question"], "answers": r["answer"]}
                    for r in ds.select(range(min(n * 3, len(ds))))]
        else:
            raise ValueError(name)
        rows = [r for r in rows if r["answers"] and r["answers"][0]]
        return rows[:n] if len(rows) >= n else _cycle(rows or _FALLBACK, n)
    except Exception as e:
        print(f"[data] '{name}' unavailable ({type(e).__name__}); using fallback set.")
        return _cycle(_FALLBACK, n)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else [str(v)]
        except Exception:
            return [x]
    return list(x)


def _cycle(rows, n):
    if not rows:
        rows = _FALLBACK
    return [rows[i % len(rows)] for i in range(n)]
