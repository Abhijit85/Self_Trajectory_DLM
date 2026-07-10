#!/usr/bin/env python3
"""SEP-DLM reference run -- orchestrates all four axes on one backbone."""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Iterable, List

from sep_dlm.adapters import build_adapter
from sep_dlm.ccap import ccap_sweep, correction_capacity
from sep_dlm.ci import calibrate_sigma, commitment_index
from sep_dlm.data import load_qa
from sep_dlm.faithfulness import faithfulness_contrast, faithfulness_gradient, format_qa_prompt
from sep_dlm.quality import generative_ppl, mock_quality

VALID_AXES = {"faith", "quality", "correction", "commitment"}


def _reference_sequences(model, dataset, prompts, gen_len, steps, n_ref):
    refs = []
    for i, ex in enumerate(dataset[:n_ref]):
        if model.is_mock:
            refs.append(model._target(ex["question"], gen_len).tolist())
        else:
            refs.append(model.generate(prompts[i], gen_len, steps, tsf=True, tsf_strength=1.0).token_ids)
    return refs


def _write_csv(path, rows: List[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_jsonl(path, rows: Iterable[dict]):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _parse_strengths(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_axes(raw: str) -> set[str]:
    axes = {x.strip() for x in raw.split(",") if x.strip()}
    bad = axes - VALID_AXES
    if bad:
        raise ValueError(f"unknown axes: {sorted(bad)}")
    return axes or set(VALID_AXES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["mock", "llada"], default="mock")
    ap.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--tsf-mode", default="feedback_gate", choices=["feedback_gate", "intensity"])
    ap.add_argument("--dataset", default="fallback", choices=["popqa", "triviaqa", "nq_open", "fallback"])
    ap.add_argument("--n", type=int, default=200, help="QA items for faithfulness")
    ap.add_argument("--grad-n", type=int, default=None, help="Optional smaller set for the TSF-strength gradient")
    ap.add_argument("--n-ci", type=int, default=32, help="prompts for CI/CCap")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gen-len", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--quality", choices=["none", "gpt2"], default="none")
    ap.add_argument("--ci-mode", choices=["flip", "noise"], default="flip")
    ap.add_argument("--gradient-strengths", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--no-gradient", action="store_true")
    ap.add_argument("--axes", default="faith,quality,correction,commitment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    axes = _parse_axes(args.axes)
    os.makedirs(args.out, exist_ok=True)
    print(f"[pilot] adapter={args.adapter} dataset={args.dataset} n={args.n} steps={args.steps} gen_len={args.gen_len}")

    model = build_adapter(args.adapter, model_path=args.model_path, tsf_mode=args.tsf_mode)
    total_needed = max(args.n, args.n_ci, args.grad_n or 0)
    dataset = load_qa(args.dataset, n=total_needed, seed=args.seed)
    model.load_answer_key(dataset)
    qa = dataset[:args.n]
    grad_qa = dataset[: (args.grad_n or args.n)]
    ci_examples = dataset[:args.n_ci]
    ci_prompts = [format_qa_prompt(ex["question"]) for ex in ci_examples]
    ccap_prompts = [format_qa_prompt(ex["question"]) for ex in dataset[:args.n_ci]]

    results = {"config": vars(args), "axes": sorted(axes)}

    faith_samples = []
    gradient_rows = []
    if "faith" in axes:
        print("[pilot] axis 1/4: faithfulness (TSF on vs off) ...")
        faith = faithfulness_contrast(model, qa, gen_len=args.gen_len, steps=args.steps, base_seed=args.seed)
        results["faithfulness"] = {k: v for k, v in faith.items() if k != "samples"}
        faith_samples = faith["samples"]
        if not args.no_gradient:
            gradient_rows = faithfulness_gradient(
                model,
                grad_qa,
                _parse_strengths(args.gradient_strengths),
                gen_len=args.gen_len,
                steps=args.steps,
                base_seed=args.seed,
            )
            results["faithfulness_gradient"] = gradient_rows

    if "quality" in axes:
        print("[pilot] axis 2/4: quality ...")
        if model.is_mock:
            results["quality"] = {"tsf_on": mock_quality(model, True), "tsf_off": mock_quality(model, False)}
        else:
            gens_on = [model.generate(format_qa_prompt(ex["question"]), args.gen_len, args.steps, tsf=True, tsf_strength=1.0).text for ex in qa]
            gens_off = [model.generate(format_qa_prompt(ex["question"]), args.gen_len, args.steps, tsf=False, tsf_strength=0.0).text for ex in qa]
            if args.quality == "gpt2":
                results["quality"] = {
                    "tsf_on": {"gen_ppl": generative_ppl(gens_on)},
                    "tsf_off": {"gen_ppl": generative_ppl(gens_off)},
                }
            else:
                results["quality"] = {"note": "quality=none; pass --quality gpt2"}

    if "correction" in axes:
        print("[pilot] axis 3/4: correction capacity (lexical + semantic) ...")
        refs = _reference_sequences(model, dataset, ccap_prompts, args.gen_len, args.steps, args.n_ci)
        ccap_lex = correction_capacity(model, refs, prompts=ccap_prompts, eps=args.eps, K=args.K, channel="lexical", base_seed=args.seed)
        ccap_sem = correction_capacity(model, refs, prompts=ccap_prompts, eps=args.eps, K=args.K, channel="semantic", base_seed=args.seed)
        results["correction"] = {"lexical": ccap_lex.summary(), "semantic": ccap_sem.summary()}
        results["correction_sweep"] = ccap_sweep(model, refs, prompts=ccap_prompts, base_seed=args.seed)

    if "commitment" in axes:
        print("[pilot] axis 4/4: commitment index + sigma calibration ...")
        calib = calibrate_sigma(model, ci_prompts, gen_len=args.gen_len, steps=args.steps, base_seed=args.seed, ci_mode=args.ci_mode)
        ci = commitment_index(model, ci_prompts, gen_len=args.gen_len, steps=args.steps, sigma=calib["recommended_sigma"], reps=args.reps, base_seed=args.seed, ci_mode=args.ci_mode)
        results["commitment"] = {"calibration": calib, "curve": ci.summary()}
    else:
        ci = None
        calib = None

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if "faith" in axes:
        _write_csv(os.path.join(args.out, "faithfulness.csv"), [results["faithfulness"]["tsf_on"], results["faithfulness"]["tsf_off"]])
        _write_jsonl(os.path.join(args.out, "qa_samples.jsonl"), faith_samples)
        if gradient_rows:
            _write_csv(os.path.join(args.out, "faithfulness_gradient.csv"), gradient_rows)
    if "correction" in axes:
        _write_csv(os.path.join(args.out, "correction_sweep.csv"), results["correction_sweep"])
    if "commitment" in axes:
        _write_csv(os.path.join(args.out, "commitment_curve.csv"), [{"t0": t, "ci": c, "ci_std": s} for t, c, s in zip(ci.t0_grid, ci.ci_curve, ci.ci_std)])
        _write_csv(os.path.join(args.out, "sigma_calibration.csv"), calib["table"])
    _write_summary_md(os.path.join(args.out, "results_summary.md"), args, results)
    print(f"[pilot] done -> {args.out}/results_summary.md")


def _write_summary_md(path, args, r):
    axes = set(r.get("axes", []))
    lines = [
        f"# SEP-DLM v0.3.1 reference run ({'MOCK -- illustrative' if args.adapter == 'mock' else args.model_path})",
        "",
        f"Backbone `{args.adapter}` | dataset `{args.dataset}` | steps={args.steps} | gen_len={args.gen_len} | TSF mode `{args.tsf_mode}` | CI mode `{args.ci_mode}` | axes `{','.join(sorted(axes))}`",
        "",
    ]

    if axes:
        lines.extend([
            "## Active Axes",
            "",
            "| Axis | Metric | TSF ON | TSF OFF | delta |",
            "|---|---|---|---|---|",
        ])

    if "quality" in axes:
        q = r.get("quality", {})
        ql_on = q.get("tsf_on", {}).get("gen_ppl", "n/a")
        ql_off = q.get("tsf_off", {}).get("gen_ppl", "n/a")
        lines.append(f"| Quality | gen PPL (lower=better) | {ql_on} | {ql_off} | {_delta(ql_on, ql_off)} |")

    if "faith" in axes:
        f = r["faithfulness"]
        on, off = f["tsf_on"], f["tsf_off"]
        lines.append(f"| Faithfulness | factual accuracy | {on['accuracy']} | {off['accuracy']} | {f['delta_accuracy']:+} |")
        lines.append(f"| Faithfulness | factual F1 | {on['F1']} | {off['F1']} | {f['delta_F1']:+} |")
        lines.append(f"| Faithfulness | halluc. rate (lower=better) | {on['halluc_rate']} | {off['halluc_rate']} | {f['delta_halluc']:+} |")

    if "correction" in axes:
        cor = r["correction"]
        lines.append(f"| Correction | CCap lexical | {cor['lexical']['CCap']} | -- | -- |")
        lines.append(f"| Correction | CCap semantic | {cor['semantic']['CCap']} | -- | -- |")

    if "commitment" in axes:
        com = r["commitment"]
        lines.append(f"| Commitment | CI AUC | {com['curve']['auc']} | -- | -- |")
        lines.extend([
            "",
            f"CI perturbation sigma was calibrated empirically to {com['calibration']['recommended_sigma']} in `{args.ci_mode}` mode.",
        ])

    grad = r.get("faithfulness_gradient", [])
    if grad:
        first, last = grad[0], grad[-1]
        lines.extend([
            "",
            "## Faithfulness Gradient",
            "",
            f"Accuracy at TSF strength {first['tsf_strength']}: {first['accuracy']}",
            f"Accuracy at TSF strength {last['tsf_strength']}: {last['accuracy']}",
            "Inspect `faithfulness_gradient.csv` to separate self-feedback strength from the strict no-feedback lower bound.",
        ])

    lines.extend([
        "",
        "Inspect `qa_samples.jsonl` before reporting results to confirm answer extraction is clean for the chosen backbone.",
    ])
    if "correction" in axes:
        lines.append("CCap can dip below -1 on individual items if refinement breaks more positions than were originally corrupted, so pair it with `mean_repaired` and `mean_broken`.")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _delta(a, b):
    try:
        return f"{float(a) - float(b):+.3f}"
    except Exception:
        return "n/a"


if __name__ == "__main__":
    main()
