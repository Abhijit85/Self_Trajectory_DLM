#!/usr/bin/env python3
"""
Compute inter-rater agreement for the SLR.

Usage:
  # Screening agreement (binary include/exclude) from screening_sheet.csv:
  python compute_kappa.py --mode screening ../screening/screening_sheet.csv

  # QA agreement (ordinal 0/0.5/1) from two QA score files with matching study order:
  python compute_kappa.py --mode qa qa_rater1.csv qa_rater2.csv

Reports Cohen's kappa (screening: unweighted; QA: quadratic-weighted, appropriate
for ordinal scores) and raw percent agreement. Falls back to a pure-python
implementation if scikit-learn is not installed.
"""
import argparse, csv, sys

def cohen_kappa(a, b, weights=None, labels=None):
    if labels is None:
        labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a); k = len(labels)
    O = [[0]*k for _ in range(k)]
    for x, y in zip(a, b):
        O[idx[x]][idx[y]] += 1
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    def w(i, j):
        if weights == 'quadratic':
            return 1 - ((i - j) ** 2) / ((k - 1) ** 2 if k > 1 else 1)
        return 1 if i == j else 0
    po = sum(w(i, j) * O[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    raw = sum(O[i][i] for i in range(k)) / n
    return kappa, raw

def read_col(path, col):
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if not (r.get(list(r)[0],'') or '').startswith('#')]
    return [r[col].strip() for r in rows if r.get(col, '').strip() != '']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['screening', 'qa'], required=True)
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()

    if args.mode == 'screening':
        path = args.files[0]
        with open(path, newline='') as f:
            rows = [r for r in csv.DictReader(f)
                    if r.get('record_id','') and not r['record_id'].startswith('#')
                    and r.get('ta_screener1','').strip() and r.get('ta_screener2','').strip()]
        a = [r['ta_screener1'].strip() for r in rows]
        b = [r['ta_screener2'].strip() for r in rows]
        if not a:
            sys.exit("No rows with both ta_screener1 and ta_screener2 filled in.")
        kappa, raw = cohen_kappa(a, b, weights=None)
        print(f"Screening agreement over n={len(a)} records:")
        print(f"  Cohen's kappa  = {kappa:.3f}")
        print(f"  raw agreement  = {raw*100:.1f}%")
    else:
        f1, f2 = args.files[0], args.files[1]
        crits = [f"QA{i}" for i in range(1, 9)]
        # accept either 'QA1'..'QA8' headers or the long names; map by prefix
        def load(path):
            with open(path, newline='') as fh:
                return [r for r in csv.DictReader(fh)
                        if r.get('study', '') and not r['study'].startswith('#')]
        r1, r2 = load(f1), load(f2)
        if not r1 or not r2:
            sys.exit("No QA rows found in one or both rater files.")
        print(f"QA agreement over n={min(len(r1),len(r2))} studies (quadratic-weighted):")
        ks = []
        for c in crits:
            col1 = next((h for h in r1[0] if h.startswith(c)), None)
            col2 = next((h for h in r2[0] if h.startswith(c)), None)
            if not col1 or not col2:
                print(f"  {c}: column not found, skipping"); continue
            a = [str(x[col1]).strip() for x in r1]
            b = [str(x[col2]).strip() for x in r2]
            kappa, raw = cohen_kappa(a, b, weights='quadratic')
            ks.append(kappa)
            print(f"  {c}: kappa_w = {kappa:.3f}  (raw {raw*100:.0f}%)")
        if ks:
            print(f"  ----  mean kappa_w = {sum(ks)/len(ks):.3f}, "
                  f"range {min(ks):.3f}-{max(ks):.3f}")

if __name__ == '__main__':
    main()
