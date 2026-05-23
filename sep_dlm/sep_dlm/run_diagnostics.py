"""Run SEP-DLM diagnostics and write auditable CSVs.

Usage (mock, to verify the harness):
    python -m sep_dlm.run_diagnostics --mock --out /tmp/sep_dlm_harness_check

To run for real, replace the mock adapters with MDLMAdapter(ckpt) /
ContinuousDLMAdapter(ckpt) in `build_adapters()` and rerun. Only CSVs written
from real checkpoint adapters are reportable.
"""
from __future__ import annotations
import argparse, csv, json, os, random, subprocess
from datetime import datetime, timezone

from .adapters import ExternalCommandAdapter, MockDiscreteDLM, MockContinuousDLM
from .corruption import corrupt
from .diagnostics import commitment_index, correction_capacity


def _load_examples(path: str | None) -> list[list[int]]:
    if path is None:
        raise SystemExit("--examples is required for real runs.")
    with open(path) as f:
        text = f.read().strip()
    if not text:
        raise SystemExit(f"{path} is empty.")
    if text[0] == "[":
        data = json.loads(text)
        if data and isinstance(data[0], int):
            data = [data]
        examples = [row["tokens"] if isinstance(row, dict) else row for row in data]
    else:
        examples = []
        for line in text.splitlines():
            row = json.loads(line)
            examples.append(row["tokens"] if isinstance(row, dict) else row)
    if not examples or not all(isinstance(row, list) and all(isinstance(t, int) for t in row)
                               for row in examples):
        raise SystemExit("--examples must be JSON/JSONL token-id lists or objects with tokens.")
    return examples


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _write_manifest(args, adapters, out: str, examples_path: str | None):
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "mode": "mock" if args.mock else "real",
        "prompt": args.prompt,
        "steps": args.steps,
        "n_seeds": args.n_seeds,
        "seed": args.seed,
        "examples": examples_path,
        "adapters": {
            name: {
                "family": ad.family,
                "command": getattr(ad, "command", None),
                "perturbation": ad.perturbation_scale(),
            }
            for name, ad in adapters.items()
        },
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def build_adapters(args):
    if args.mock:
        return {"discrete_mock": MockDiscreteDLM(), "continuous_mock": MockContinuousDLM()}
    if not (args.discrete_cmd and args.continuous_cmd):
        raise SystemExit(
            "For a real run, pass both --discrete-cmd and --continuous-cmd. "
            "Each command must read a JSON request from stdin and return "
            "{\"tokens\": [int, ...]} on stdout."
        )
    return {
        args.discrete_name: ExternalCommandAdapter(
            name=args.discrete_name,
            family="discrete",
            command=args.discrete_cmd,
            perturbation=args.discrete_perturbation,
        ),
        args.continuous_name: ExternalCommandAdapter(
            name=args.continuous_name,
            family="continuous",
            command=args.continuous_cmd,
            perturbation=args.continuous_perturbation,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="the capital of france is")
    ap.add_argument("--examples", help="JSON/JSONL token-id examples for CCap real runs.")
    ap.add_argument("--discrete-cmd", help="External real discrete adapter command.")
    ap.add_argument("--continuous-cmd", help="External real continuous adapter command.")
    ap.add_argument("--discrete-name", default="discrete_real")
    ap.add_argument("--continuous-name", default="continuous_real")
    ap.add_argument("--discrete-perturbation", type=float, default=0.5)
    ap.add_argument("--continuous-perturbation", type=float, default=0.1)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    adapters = build_adapters(args)
    prompt = args.prompt
    t_grid = [i / 10 for i in range(1, 10)]
    eps_grid, K_grid = [0.05, 0.1, 0.2], [4, 8, 16]
    examples = [[i % 50 for i in range(32)]] if args.mock else _load_examples(args.examples)

    with open(os.path.join(args.out, "ci_curve.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "t0", "ci", "ci_lo", "ci_hi", "n_seeds"])
        for name, ad in adapters.items():
            for r in commitment_index(ad, prompt, t_grid=t_grid, steps=args.steps,
                                      n_seeds=args.n_seeds):
                w.writerow([name, r.t0, f"{r.ci:.4f}", f"{r.lo:.4f}", f"{r.hi:.4f}", r.n_seeds])

    with open(os.path.join(args.out, "ccap.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "corruption", "eps", "K", "ccap", "ccap_lo", "ccap_hi"])
        for name, ad in adapters.items():
            ex = [ad.mode] if hasattr(ad, "mode") else examples  # mock: target=mode
            for kind in ("lexical", "semantic"):
                for eps in eps_grid:
                    for K in K_grid:
                        r = correction_capacity(ad, ex, eps=eps, K=K,
                                                corruption=kind, corrupt_fn=corrupt,
                                                n_seeds=args.n_seeds)
                        w.writerow([name, kind, eps, K, f"{r.ccap:.4f}",
                                    f"{r.lo:.4f}", f"{r.hi:.4f}"])
    print(f"Wrote {args.out}/ci_curve.csv and {args.out}/ccap.csv")
    _write_manifest(args, adapters, args.out, args.examples)
    print(f"Wrote {args.out}/manifest.json")
    if args.mock:
        print("NOTE: mock numbers — harness test only, NOT results for the paper.")


if __name__ == "__main__":
    main()
