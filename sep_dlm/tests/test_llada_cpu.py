from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from sep_dlm.adapters import Hooks, LLaDAAdapter
from sep_dlm.ccap import correction_capacity
from sep_dlm.ci import calibrate_sigma, commitment_index
from sep_dlm.faithfulness import faithfulness_contrast, format_qa_prompt


class StubOutput:
    def __init__(self, logits):
        self.logits = logits


class StubTokenizer:
    mask_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [2 + (ord(ch) % 40) for ch in text][:32] or [2]}

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"tok{int(i)}" for i in ids)


class StubModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.config = type("Cfg", (), {"vocab_size": vocab_size})()
        self.vocab_size = vocab_size

    def forward(self, x):
        batch, length = x.shape
        logits = torch.full((batch, length, self.vocab_size), -8.0, device=x.device)
        pos = torch.arange(length, device=x.device).unsqueeze(0).expand(batch, -1)
        target = (x + pos + 3) % self.vocab_size
        logits.scatter_(2, target.unsqueeze(-1), 8.0)
        logits[:, :, 1] = -10.0
        return StubOutput(logits)


def make_adapter(vocab_size=64):
    adapter = LLaDAAdapter.__new__(LLaDAAdapter)
    adapter.torch = torch
    adapter.device = "cpu"
    adapter.tsf_mode = "feedback_gate"
    adapter.tok = StubTokenizer()
    adapter.model = StubModel(vocab_size)
    adapter.mask_id = 1
    adapter.vocab_size = vocab_size
    return adapter


def main():
    adapter = make_adapter()
    prompt = format_qa_prompt("What is the capital of Australia?")
    out_on = adapter.generate(prompt, gen_len=8, steps=6, tsf=True, tsf_strength=1.0)
    out_mid = adapter.generate(prompt, gen_len=8, steps=6, tsf=True, tsf_strength=0.5)
    out_off = adapter.generate(prompt, gen_len=8, steps=6, tsf=False, tsf_strength=0.0)
    out_flip = adapter.generate(prompt, gen_len=8, steps=6, tsf=True, hooks=Hooks(hard_flip_steps=frozenset({3}), seed=0))
    assert len(out_on.token_ids) == 8
    assert len(out_mid.token_ids) == 8
    assert len(out_off.token_ids) == 8
    assert len(out_flip.token_ids) == 8

    refined = adapter.refine(out_on.token_ids, K=4, prompt=prompt)
    assert len(refined.token_ids) == 8

    prompts = [format_qa_prompt("Who wrote Pride and Prejudice?"), format_qa_prompt("What is the chemical symbol for gold?")]
    ci = commitment_index(adapter, prompts, gen_len=8, steps=6, reps=2, ci_mode="flip")
    calib = calibrate_sigma(adapter, prompts, gen_len=8, steps=6, reps=2, ci_mode="flip")
    assert len(ci.t0_grid) == len(ci.ci_curve) == len(ci.ci_std)
    assert "recommended_sigma" in calib

    dataset = [
        {"question": "What is the capital of Australia?", "answers": ["tok5"]},
        {"question": "What is the chemical symbol for gold?", "answers": ["tok7"]},
    ]
    faith = faithfulness_contrast(adapter, dataset, gen_len=8, steps=6)
    assert "samples" in faith and len(faith["samples"]) == 4

    refs = [adapter.generate(p, gen_len=8, steps=6, tsf=True).token_ids for p in prompts]
    ccap = correction_capacity(adapter, refs, prompts=prompts, eps=0.15, K=4, channel="lexical")
    assert ccap.n >= 0
    print("cpu llada adapter harness check passed")


if __name__ == "__main__":
    main()
