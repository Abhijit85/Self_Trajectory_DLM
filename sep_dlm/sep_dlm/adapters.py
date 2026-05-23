"""Model adapters for SEP-DLM diagnostics.

`DLMAdapter` is the interface the diagnostics call. Implement it once per checkpoint.
Two MOCK adapters are provided ONLY to test that the harness runs and to illustrate the
shape H_wall predicts (discrete = early commitment, continuous = late). They are NOT
experimental results and must never be reported as such. Real-checkpoint stubs at the
bottom show where to plug in MDLM / a continuous DLM.
"""
from __future__ import annotations
import json
import random
import shlex
import subprocess
from typing import Optional, Sequence


class DLMAdapter:
    family: str = "abstract"

    def perturbation_scale(self) -> float:
        """Native-space perturbation magnitude (Appendix D (i)):
        discrete -> logit-space sigma s.t. per-token KL ~= 0.5 nats;
        continuous/flow -> embedding-space sigma = 0.1 * per-dim activation std.
        """
        raise NotImplementedError

    def rollout(self, prompt, *, steps: int, seed: int, self_feedback: bool = True,
                ablate_sf_at: Optional[float] = None,
                perturbation: Optional[float] = None) -> list[int]:
        """Run the reverse process; return final token ids. If `ablate_sf_at` is set
        (a fraction in [0,1]), ablate the self-feedback channel + add `perturbation`
        at that step only, then resume self-feedback."""
        raise NotImplementedError

    def repair(self, x_corrupt, *, budget: int, seed: int = 0) -> list[int]:
        """Run `budget` refinement steps starting from a corrupted sequence."""
        raise NotImplementedError


# --------------------------- MOCK adapters (test only) ---------------------- #
class _MockDLM(DLMAdapter):
    """Toy generative process. A 'latent intent' picks a target string; at each step a
    token is committed. Self-feedback at step t nudges later tokens toward the intent.
    `commit_profile` controls how strongly an early perturbation propagates: 'early'
    (discrete-like) locks in fast; 'late' (continuous-like) stays revisable.
    """
    def __init__(self, vocab: int = 50, length: int = 32, commit_profile: str = "early"):
        self.vocab, self.length, self.commit_profile = vocab, length, commit_profile
        # A fixed "learned mode" the toy model denoises toward. For CCap, use this as
        # the ground-truth example set (run_diagnostics does this for mocks).
        _r = random.Random(1234)
        self.mode = [_r.randrange(vocab) for _ in range(length)]

    def perturbation_scale(self) -> float:
        return 0.5 if self.family == "discrete" else 0.1

    def _intent(self, prompt, seed):
        rng = random.Random((hash(prompt) ^ seed) & 0xFFFFFFFF)
        return [rng.randrange(self.vocab) for _ in range(self.length)]

    def rollout(self, prompt, *, steps, seed, self_feedback=True,
                ablate_sf_at=None, perturbation=None):
        rng = random.Random((hash(prompt) ^ (seed * 2654435761)) & 0xFFFFFFFF)
        target = self._intent(prompt, seed)
        out = list(target)
        if ablate_sf_at is not None:
            # how much an ablation at t0 changes the FINAL output:
            # 'early' profile -> large effect for small t0; 'late' -> for large t0.
            t0 = ablate_sf_at
            if self.commit_profile == "early":
                impact = max(0.0, 1.0 - t0)          # steep, early
            else:
                impact = max(0.0, t0)                # gradual, late
            n_changed = int(round(impact * self.length * (perturbation or 0.5)))
            idx = rng.sample(range(self.length), min(n_changed, self.length))
            for i in idx:
                out[i] = rng.randrange(self.vocab)
        return out

    def repair(self, x_corrupt, *, budget, seed=0):
        """TOY denoiser (smoke-test only): walks x_corrupt toward self.mode. Repair
        probability rises with budget K; the 'early'/discrete profile also regresses a
        few already-correct positions (a stand-in for wall amplification), 'late'/
        continuous barely regresses. These numbers are NOT results."""
        rng = random.Random(seed)
        out = list(x_corrupt)
        p_repair = min(1.0, budget / 16.0)
        p_regress = 0.05 if self.commit_profile == "early" else 0.0
        for i in range(self.length):
            if out[i] != self.mode[i] and rng.random() < p_repair:
                out[i] = self.mode[i]                       # fix a wrong position
            elif out[i] == self.mode[i] and rng.random() < p_regress:
                out[i] = rng.randrange(self.vocab)          # break a right one
        return out


class MockDiscreteDLM(_MockDLM):
    family = "discrete"
    def __init__(self, **kw): super().__init__(commit_profile="early", **kw)


class MockContinuousDLM(_MockDLM):
    family = "continuous"
    def __init__(self, **kw): super().__init__(commit_profile="late", **kw)


# --------------------------- external real adapters ------------------------- #
class ExternalCommandAdapter(DLMAdapter):
    """Adapter for real checkpoints implemented in a separate script/process.

    The command is called once per rollout/repair with a JSON request on stdin and
    must return JSON on stdout: {"tokens": [int, ...]}. This keeps the harness
    independent of model-specific dependencies such as custom LLaDA/MDLM/DiffuSeq
    repositories while still making the recorded CSVs auditable.
    """
    def __init__(self, *, name: str, family: str, command: str, perturbation: float):
        self.name = name
        self.family = family
        self.command = command
        self._perturbation = perturbation

    def perturbation_scale(self) -> float:
        return self._perturbation

    def _call(self, payload: dict) -> list[int]:
        payload = dict(payload)
        payload["adapter_name"] = self.name
        payload["family"] = self.family
        proc = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.name} adapter command failed with exit {proc.returncode}:\n"
                f"STDERR:\n{proc.stderr.strip()}\nSTDOUT:\n{proc.stdout.strip()}"
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{self.name} adapter returned non-JSON stdout:\n{proc.stdout}"
            ) from exc
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
            raise RuntimeError(
                f"{self.name} adapter must return JSON like {{\"tokens\": [int, ...]}}"
            )
        return tokens

    def rollout(self, prompt, *, steps, seed, self_feedback=True,
                ablate_sf_at=None, perturbation=None):
        return self._call({
            "op": "rollout",
            "prompt": prompt,
            "steps": steps,
            "seed": seed,
            "self_feedback": self_feedback,
            "ablate_sf_at": ablate_sf_at,
            "perturbation": perturbation,
        })

    def repair(self, x_corrupt, *, budget, seed=0):
        return self._call({
            "op": "repair",
            "tokens": list(x_corrupt),
            "budget": budget,
            "seed": seed,
        })


# --------------------------- REAL-checkpoint stubs -------------------------- #
class MDLMAdapter(DLMAdapter):
    """TODO: wrap a real masked discrete DLM (e.g. MDLM / LLaDA).
    - perturbation_scale: calibrate logit-space sigma to KL=0.5 nats per token.
    - rollout: run the standard reverse unmasking loop; at step int(ablate_sf_at*steps)
      replace the self-conditioning input with the no-SC path and add Gaussian logit noise.
    - repair: start from x_corrupt as the canvas, run `budget` denoising steps.
    """
    family = "discrete"
    def __init__(self, checkpoint_path: str):
        raise NotImplementedError("Plug in your MDLM/LLaDA checkpoint here.")


class ContinuousDLMAdapter(DLMAdapter):
    """TODO: wrap a continuous / flow DLM (e.g. LangFlow, DiffuSeq).
    - perturbation: embedding-space Gaussian, sigma = 0.1 * per-dim activation std.
    - rollout/repair: as above, in embedding space, projecting to tokens at the end.
    """
    family = "continuous"
    def __init__(self, checkpoint_path: str):
        raise NotImplementedError("Plug in your continuous/flow checkpoint here.")
