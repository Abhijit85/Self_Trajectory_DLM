"""Faithfulness axis for the SEP-DLM pilot harness."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .adapters import DiffusionLM, Hooks
from .metrics import contains_match, is_hallucination, token_f1


@dataclass
class FaithResult:
    tsf: bool
    tsf_strength: float
    n: int
    accuracy: float
    f1: float
    hallucination_rate: float
    abstain_rate: float

    def summary(self) -> dict:
        return {
            "tsf": self.tsf,
            "tsf_strength": round(self.tsf_strength, 3),
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "F1": round(self.f1, 4),
            "halluc_rate": round(self.hallucination_rate, 4),
            "abstain_rate": round(self.abstain_rate, 4),
        }


def format_qa_prompt(question: str) -> str:
    return (
        "Answer the question with a short factual answer.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def extract_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[0]
    line = re.split(r"\b(?:Question|Explanation|Reasoning)\s*:", line, maxsplit=1)[0]
    line = line.strip().strip('"\'`')
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"\s*([,;:])\s*$", "", line)
    sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", line, maxsplit=1)[0]
    sentence = sentence.strip().strip('"\'`')
    sentence = re.sub(r"\s*([,;:])\s*$", "", sentence)
    return sentence


def _answer(model: DiffusionLM, question: str, tsf: bool, gen_len: int, steps: int, seed: int, tsf_strength: float):
    if model.is_mock:
        pred, golds = model.answer_qa(question, tsf=tsf, seed=seed, tsf_strength=tsf_strength)
        return pred, golds
    prompt = format_qa_prompt(question)
    out = model.generate(prompt, gen_len, steps, tsf=tsf, tsf_strength=tsf_strength, hooks=Hooks(seed=seed))
    return extract_answer(out.text), None


def evaluate_faithfulness(
    model: DiffusionLM,
    dataset: Sequence[dict],
    tsf: bool,
    gen_len: int = 16,
    steps: int = 16,
    base_seed: int = 0,
    tsf_strength: float = 1.0,
) -> tuple[FaithResult, list[dict]]:
    acc = f1 = hall = abst = 0.0
    n = 0
    samples = []
    effective_strength = max(0.0, min(1.0, tsf_strength if tsf else 0.0))
    for i, ex in enumerate(dataset):
        q, golds_ds = ex["question"], ex["answers"]
        pred, golds = _answer(model, q, tsf, gen_len, steps, seed=base_seed + i, tsf_strength=effective_strength)
        golds = golds or golds_ds
        acc_i = contains_match(pred, golds)
        hall_i = int(is_hallucination(pred, golds))
        acc += acc_i
        f1 += token_f1(pred, golds)
        hall += hall_i
        p = pred.strip().lower()
        abst_i = int(p in {"", "i don't know", "i dont know", "unknown"})
        abst += abst_i
        samples.append({
            "idx": i,
            "tsf": tsf,
            "tsf_strength": effective_strength,
            "question": q,
            "prediction": pred,
            "answers": list(golds),
            "accuracy": acc_i,
            "hallucination": hall_i,
            "abstain": abst_i,
        })
        n += 1
    n = max(n, 1)
    return FaithResult(tsf, effective_strength, n, acc / n, f1 / n, hall / n, abst / n), samples


def faithfulness_contrast(model: DiffusionLM, dataset: Sequence[dict], gen_len: int = 16, steps: int = 16, base_seed: int = 0) -> dict:
    on, on_samples = evaluate_faithfulness(model, dataset, True, gen_len, steps, base_seed, tsf_strength=1.0)
    off, off_samples = evaluate_faithfulness(model, dataset, False, gen_len, steps, base_seed, tsf_strength=0.0)
    return {
        "tsf_on": on.summary(),
        "tsf_off": off.summary(),
        "delta_accuracy": round(on.accuracy - off.accuracy, 4),
        "delta_F1": round(on.f1 - off.f1, 4),
        "delta_halluc": round(on.hallucination_rate - off.hallucination_rate, 4),
        "samples": on_samples + off_samples,
    }


def faithfulness_gradient(
    model: DiffusionLM,
    dataset: Sequence[dict],
    strengths: Sequence[float],
    gen_len: int = 16,
    steps: int = 16,
    base_seed: int = 0,
) -> list[dict]:
    rows = []
    for strength in strengths:
        res, _ = evaluate_faithfulness(model, dataset, True, gen_len, steps, base_seed, tsf_strength=float(strength))
        rows.append({
            "tsf_strength": round(float(strength), 3),
            "accuracy": round(res.accuracy, 4),
            "F1": round(res.f1, 4),
            "halluc_rate": round(res.hallucination_rate, 4),
            "abstain_rate": round(res.abstain_rate, 4),
            "n": res.n,
        })
    return rows
