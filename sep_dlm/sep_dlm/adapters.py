"""Backbone adapters for the SEP-DLM pilot harness."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence

import numpy as np


@dataclass
class Hooks:
    """Intervention schedule handed to a single generate()/refine() call."""
    sc_off_steps: FrozenSet[int] = frozenset()
    hard_flip_steps: FrozenSet[int] = frozenset()
    logit_noise: Dict[int, float] = field(default_factory=dict)
    seed: int = 0

    @staticmethod
    def none(seed: int = 0) -> "Hooks":
        return Hooks(seed=seed)


@dataclass
class GenResult:
    token_ids: List[int]
    text: str = ""
    meta: dict = field(default_factory=dict)


class DiffusionLM(ABC):
    is_mock: bool = False
    mask_id: int = 0
    vocab_size: int = 32000

    @abstractmethod
    def generate(
        self,
        prompt: str,
        gen_len: int,
        steps: int,
        tsf: bool = True,
        tsf_strength: float = 1.0,
        hooks: Optional[Hooks] = None,
    ) -> GenResult:
        """Denoise `gen_len` answer tokens over `steps` reverse steps."""

    @abstractmethod
    def refine(
        self,
        sequence_ids: Sequence[int],
        K: int,
        hooks: Optional[Hooks] = None,
        reference_ids: Optional[Sequence[int]] = None,
        prompt: Optional[str] = None,
    ) -> GenResult:
        """Revise an existing sequence for K steps in place (for CCap)."""

    def encode(self, text: str) -> List[int]:
        return [((b + 5) % self.vocab_size) for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(str(i) for i in ids)

    def load_answer_key(self, dataset) -> None:
        return None


class MockAdapter(DiffusionLM):
    is_mock = True

    def __init__(
        self,
        commitment: float = 0.75,
        correction_strength: float = 1.4,
        break_rate: float = 0.03,
        gen_len: int = 24,
        vocab_size: int = 4096,
        acc_tsf_on: float = 0.41,
        acc_tsf_off: float = 0.42,
        ppl_tsf_on: float = 18.7,
        ppl_tsf_off: float = 24.9,
        halluc_tsf_on: float = 0.34,
        halluc_tsf_off: float = 0.33,
    ):
        self.commitment = commitment
        self.correction_strength = correction_strength
        self.break_rate = break_rate
        self.default_gen_len = gen_len
        self.vocab_size = vocab_size
        self.mask_id = 1
        self.acc = {True: acc_tsf_on, False: acc_tsf_off}
        self.ppl = {True: ppl_tsf_on, False: ppl_tsf_off}
        self.halluc = {True: halluc_tsf_on, False: halluc_tsf_off}
        self._answer_key: Dict[str, list] = {}

    def _rng(self, *parts) -> np.random.Generator:
        h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    def _target(self, prompt: str, gen_len: int) -> np.ndarray:
        rng = self._rng("target", prompt)
        return rng.integers(2, self.vocab_size, size=gen_len)

    def generate(
        self,
        prompt: str,
        gen_len: int,
        steps: int,
        tsf: bool = True,
        tsf_strength: float = 1.0,
        hooks: Optional[Hooks] = None,
    ) -> GenResult:
        hooks = hooks or Hooks.none()
        gen_len = gen_len or self.default_gen_len
        out = self._target(prompt, gen_len).copy()
        rng = self._rng(
            "gen",
            prompt,
            tsf,
            round(tsf_strength, 3),
            hooks.seed,
            tuple(sorted(hooks.logit_noise.items())),
            tuple(sorted(hooks.sc_off_steps)),
            tuple(sorted(hooks.hard_flip_steps)),
        )
        effective_strength = max(0.0, min(1.0, tsf_strength if tsf else 0.0))

        for step in range(max(steps, 1)):
            if step in hooks.hard_flip_steps:
                pos = int(rng.integers(0, gen_len))
                survive = min(1.0, 0.1 + 0.9 * ((step + 1) / max(steps, 1)))
                survive *= 0.5 + 0.5 * effective_strength
                if rng.random() < survive:
                    wrong = int(rng.integers(2, self.vocab_size))
                    if wrong == out[pos]:
                        wrong = (wrong + 1) % self.vocab_size
                    out[pos] = wrong

        for t0, sigma in hooks.logit_noise.items():
            if sigma <= 0:
                continue
            frac_hit = 1.0 - np.exp(-sigma)
            n_hit = int(round(frac_hit * gen_len))
            if n_hit == 0:
                continue
            hit = rng.choice(gen_len, size=n_hit, replace=False)
            feedback_ablated = t0 in hooks.sc_off_steps
            step_frac = (t0 + 1) / max(steps, 1)
            if feedback_ablated:
                survive = self.commitment + (1.0 - self.commitment) * step_frac
            else:
                survive = (0.15 + 0.35 * (1.0 - effective_strength)) * step_frac
            keep = hit[rng.random(len(hit)) < survive]
            for pos in keep:
                wrong = int(rng.integers(2, self.vocab_size))
                if wrong == out[pos]:
                    wrong = (wrong + 1) % self.vocab_size
                out[pos] = wrong

        return GenResult(
            token_ids=out.tolist(),
            text=self.decode(out),
            meta={"tsf": tsf, "tsf_strength": effective_strength, "steps": steps},
        )

    def refine(
        self,
        sequence_ids: Sequence[int],
        K: int,
        hooks: Optional[Hooks] = None,
        reference_ids: Optional[Sequence[int]] = None,
        prompt: Optional[str] = None,
    ) -> GenResult:
        del hooks, prompt
        seq = list(sequence_ids)
        if reference_ids is None:
            return GenResult(token_ids=seq, text=self.decode(seq))
        ref = list(reference_ids)
        rng = self._rng("refine", tuple(seq[:8]), K, len(seq))
        p_repair = 1.0 - np.exp(-self.correction_strength * K / max(K + 4, 1))
        for i in range(min(len(seq), len(ref))):
            if seq[i] != ref[i]:
                if rng.random() < p_repair:
                    seq[i] = ref[i]
            else:
                if rng.random() < self.break_rate:
                    seq[i] = (ref[i] + 7) % self.vocab_size
        return GenResult(token_ids=seq, text=self.decode(seq))

    def load_answer_key(self, dataset) -> None:
        for ex in dataset:
            self._answer_key[ex["question"]] = ex["answers"]

    def answer_qa(self, question: str, tsf: bool, seed: int = 0, tsf_strength: float = 1.0):
        golds = self._answer_key.get(question, ["<unknown>"])
        rng = self._rng("qa", question, tsf, seed, round(tsf_strength, 3))
        effective_strength = max(0.0, min(1.0, tsf_strength if tsf else 0.0))
        acc = self.acc[False] + effective_strength * (self.acc[True] - self.acc[False])
        halluc = self.halluc[False] + effective_strength * (self.halluc[True] - self.halluc[False])
        r = rng.random()
        if r < acc:
            return golds[0], golds
        if r < acc + halluc:
            return f"distractor_{int(rng.integers(1000))}", golds
        return "i don't know", golds


class LLaDAAdapter(DiffusionLM):
    def __init__(self, model_path: str, device: str = "cuda", tsf_mode: str = "feedback_gate", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tsf_mode = tsf_mode
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=getattr(torch, dtype),
        ).to(device).eval()
        self.mask_id = getattr(self.tok, "mask_token_id", None) or 126336
        self.vocab_size = self.model.config.vocab_size

    def encode(self, text: str):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=True)

    def _forward_logits(self, x):
        with self.torch.no_grad():
            return self.model(x).logits

    def _add_gumbel_noise(self, logits, temperature):
        if temperature <= 0:
            return logits
        noise = self.torch.rand_like(logits)
        gumbel = -self.torch.log(-self.torch.log(noise + 1e-9) + 1e-9)
        return logits / temperature + gumbel

    def _answer_canvas(self, prompt_ids, answer_ids, visible_mask):
        torch = self.torch
        gen_len = answer_ids.shape[1]
        x = torch.full((1, len(prompt_ids) + gen_len), self.mask_id, device=self.device)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        x[:, :len(prompt_ids)] = prompt_tensor
        ans = x[:, len(prompt_ids):]
        ans[visible_mask] = answer_ids[visible_mask]
        return x

    def _next_visible_mask(self, conf, visible_budget, strength):
        torch = self.torch
        conf = conf.clone()
        mask = torch.zeros_like(conf, dtype=torch.bool)
        if visible_budget <= 0:
            return mask
        top_budget = min(visible_budget, conf.shape[1])
        budget_idx = conf.topk(top_budget, dim=-1).indices
        keep = int(round(top_budget * strength))
        if strength >= 1.0:
            keep = top_budget
        if keep <= 0:
            return mask
        for b in range(conf.shape[0]):
            chosen = budget_idx[b, :keep]
            mask[b, chosen] = True
        return mask

    def generate(
        self,
        prompt: str,
        gen_len: int,
        steps: int,
        tsf: bool = True,
        tsf_strength: float = 1.0,
        hooks: Optional[Hooks] = None,
    ) -> GenResult:
        torch = self.torch
        hooks = hooks or Hooks.none()
        effective_strength = max(0.0, min(1.0, tsf_strength if tsf else 0.0))
        if hooks.seed:
            torch.manual_seed(hooks.seed)
        prompt_ids = self.encode(prompt)
        answer_ids = torch.full((1, gen_len), self.mask_id, device=self.device)
        visible_mask = torch.zeros((1, gen_len), dtype=torch.bool, device=self.device)

        if self.tsf_mode == "feedback_gate":
            effective_steps = max(steps, 1)
        else:
            effective_steps = max(steps if (tsf or effective_strength > 0) else 1, 1)

        for step in range(effective_steps):
            step_strength = effective_strength
            if step in hooks.sc_off_steps:
                step_strength = 0.0
            step_visible = visible_mask.clone()
            if step_strength < 1.0 and torch.any(step_visible):
                visible_budget = int(step_visible.sum().item())
                conf_for_visible = torch.where(
                    step_visible,
                    torch.ones_like(answer_ids, dtype=torch.float32),
                    torch.full_like(answer_ids, -1e9, dtype=torch.float32),
                )
                step_visible = self._next_visible_mask(conf_for_visible, visible_budget, step_strength)
            x = self._answer_canvas(prompt_ids, answer_ids, step_visible)
            logits = self._forward_logits(x)
            sigma = hooks.logit_noise.get(step, 0.0)
            if sigma > 0:
                logits = logits + sigma * torch.randn_like(logits)
            logits = self._add_gumbel_noise(logits, temperature=0.0)
            ans_start = len(prompt_ids)
            ans_logits = logits[:, ans_start:ans_start + gen_len]
            pred = ans_logits.argmax(dim=-1)
            conf = ans_logits.softmax(-1).max(-1).values

            if step in hooks.hard_flip_steps:
                flip_idx = conf.argmax(dim=-1)
                for b in range(pred.shape[0]):
                    idx = int(flip_idx[b].item())
                    wrong = int((pred[b, idx].item() + 1) % self.vocab_size)
                    if wrong == self.mask_id:
                        wrong = (wrong + 1) % self.vocab_size
                    pred[b, idx] = wrong
                    conf[b, idx] = conf.max() + 1.0

            answer_ids = pred
            if step == effective_steps - 1:
                visible_mask = torch.ones_like(visible_mask)
                continue
            visible_budget = max(1, int(round(gen_len * (step + 1) / effective_steps)))
            visible_mask = self._next_visible_mask(conf, visible_budget, 1.0)

        return GenResult(
            token_ids=answer_ids[0].tolist(),
            text=self.decode(answer_ids[0].tolist()),
            meta={"tsf": tsf, "tsf_strength": effective_strength},
        )

    def refine(self, sequence_ids, K, hooks=None, reference_ids=None, prompt=None) -> GenResult:
        del reference_ids
        torch = self.torch
        hooks = hooks or Hooks.none()
        if hooks.seed:
            torch.manual_seed(hooks.seed)
        prefix = self.encode(prompt) if prompt else []
        full_ids = prefix + list(sequence_ids)
        x = torch.tensor([full_ids], device=self.device)
        ans_start = len(prefix)
        ans_slice = slice(ans_start, x.shape[1])
        for _ in range(K):
            logits = self._forward_logits(x)
            pred = logits.argmax(-1)
            conf = logits.softmax(-1).max(-1).values[:, ans_slice]
            width = ans_slice.stop - ans_slice.start
            k = max(1, width // max(K, 1))
            worst = conf.topk(min(k, width), largest=False, dim=-1).indices + ans_start
            for b in range(x.shape[0]):
                x[b, worst[b]] = pred[b, worst[b]]
        ids = x[0, ans_slice].tolist()
        return GenResult(token_ids=ids, text=self.decode(ids))


def build_adapter(kind: str, **kw) -> DiffusionLM:
    if kind == "mock":
        return MockAdapter(**{k: v for k, v in kw.items() if k in MockAdapter.__init__.__code__.co_varnames})
    if kind == "llada":
        return LLaDAAdapter(**kw)
    raise ValueError(f"unknown adapter '{kind}' (use 'mock' or 'llada')")
