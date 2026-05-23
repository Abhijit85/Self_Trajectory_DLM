"""Plot SEP-DLM diagnostics figures from CSVs (data-agnostic).

Reads ci_curve.csv and/or ccap.csv (the format run_diagnostics.py writes) and emits
publication figures:
  - ci_curve.pdf/.png   : CI(t) per model, with bootstrap CI bands.
  - ccap_curves.pdf/.png : CCap vs budget K, faceted by corruption family, one line
                           per (model, eps), with bootstrap CI bands and a y=0 line.

Produces NOTHING until you point it at real CSVs. Running it on a --mock CSV yields a
MOCK figure (do not put it in the paper).

Usage:
    python3 -m sep_dlm.plot_diagnostics --ci results/ci_curve.csv \
        --ccap results/ccap.csv --out figs/
"""
from __future__ import annotations
import argparse, csv, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_ci(path, out):
    rows = _read(path)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append((float(r["t0"]), float(r["ci"]),
                                     float(r["ci_lo"]), float(r["ci_hi"])))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for model, pts in by_model.items():
        pts.sort()
        t = [p[0] for p in pts]; ci = [p[1] for p in pts]
        lo = [p[2] for p in pts]; hi = [p[3] for p in pts]
        line, = ax.plot(t, ci, marker="o", ms=3, label=model)
        ax.fill_between(t, lo, hi, alpha=0.18, color=line.get_color())
    ax.set_xlabel("trajectory fraction $t_0$"); ax.set_ylabel("Commitment Index CI($t_0$)")
    ax.set_ylim(bottom=0); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Commitment Index", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"ci_curve.{ext}"), dpi=200)
    plt.close(fig)


def plot_ccap(path, out):
    rows = _read(path)
    families = sorted({r["corruption"] for r in rows})
    fig, axes = plt.subplots(1, len(families), figsize=(5.2 * len(families), 3.6),
                             sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], families):
        series = defaultdict(list)
        for r in rows:
            if r["corruption"] != fam:
                continue
            series[(r["model"], r["eps"])].append(
                (int(r["K"]), float(r["ccap"]), float(r["ccap_lo"]), float(r["ccap_hi"])))
        for (model, eps), pts in sorted(series.items()):
            pts.sort()
            K = [p[0] for p in pts]; v = [p[1] for p in pts]
            lo = [p[2] for p in pts]; hi = [p[3] for p in pts]
            line, = ax.plot(K, v, marker="o", ms=3, label=f"{model}, $\\epsilon$={eps}")
            ax.fill_between(K, lo, hi, alpha=0.15, color=line.get_color())
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("budget $K$"); ax.set_title(f"{fam} corruption", fontsize=10)
    axes[0][0].set_ylabel("Correction Capacity CCap($\\epsilon$, K)")
    axes[0][-1].legend(frameon=False, fontsize=7, ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"ccap_curves.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci"); ap.add_argument("--ccap"); ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.ci:
        plot_ci(args.ci, args.out); print(f"wrote {args.out}/ci_curve.[pdf|png]")
    if args.ccap:
        plot_ccap(args.ccap, args.out); print(f"wrote {args.out}/ccap_curves.[pdf|png]")
    if not (args.ci or args.ccap):
        ap.error("pass --ci and/or --ccap")


if __name__ == "__main__":
    main()
