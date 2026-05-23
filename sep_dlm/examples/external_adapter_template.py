"""Template for wrapping a real checkpoint for SEP-DLM diagnostics.

Copy this file outside the package, load your checkpoint once at startup, and
replace `rollout_real_model` / `repair_real_model` with model-specific code.
The diagnostics harness calls this process with JSON on stdin and expects:

    {"tokens": [int, ...]}

This template intentionally raises until real checkpoint code is supplied.
"""
from __future__ import annotations
import json
import sys


def rollout_real_model(request: dict) -> list[int]:
    """Return final token ids for one generation trajectory.

    Required request keys:
      prompt: str
      steps: int
      seed: int
      self_feedback: bool
      ablate_sf_at: float | None
      perturbation: float | None
    """
    raise NotImplementedError("Load a real checkpoint and implement rollout here.")


def repair_real_model(request: dict) -> list[int]:
    """Return token ids after K repair/refinement steps.

    Required request keys:
      tokens: list[int]
      budget: int
      seed: int
    """
    raise NotImplementedError("Load a real checkpoint and implement repair here.")


def main() -> int:
    request = json.loads(sys.stdin.read())
    op = request.get("op")
    if op == "rollout":
        tokens = rollout_real_model(request)
    elif op == "repair":
        tokens = repair_real_model(request)
    else:
        raise ValueError(f"unknown op: {op}")
    print(json.dumps({"tokens": tokens}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
