"""Aggregate runs/*/metrics.json into console tables + a LaTeX table.

Usage:
  python scripts/aggregate.py                 # all runs under ./runs
  python scripts/aggregate.py --runs ./runs --out ./runs/tables.tex

Groups tags that differ only by a trailing _s<seed> into mean +/- std.
"""
import os
import re
import glob
import json
import argparse
from collections import defaultdict


KEYS = ["recall@10", "recall@20", "ndcg@10", "ndcg@20"]


def load_runs(runs_dir):
    rows = {}
    for path in glob.glob(os.path.join(runs_dir, "*", "metrics.json")):
        tag = os.path.basename(os.path.dirname(path))
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows[tag] = json.load(f)
        except Exception as e:
            print(f"[warn] skip {tag}: {e}")
    return rows


def _fmt(x):
    return f"{x:.4f}"


def _mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


def print_per_run(rows):
    print("\n=== per-run (cold = ranked among cold items) ===")
    head = f"{'tag':28} | {'warm R@20':>9} {'warm N@20':>9} | " \
           f"{'cold R@20':>9} {'cold N@20':>9} | {'beta':>6} {'adap':>7} {'ms':>5}"
    print(head)
    print("-" * len(head))
    for tag in sorted(rows):
        m = rows[tag].get("metrics", {})
        eff = rows[tag].get("efficiency", {})
        w = m.get("warm", {})
        c = m.get("cold", {})
        b = eff.get("fusion_beta")
        bs = f"{b:.3f}" if isinstance(b, (int, float)) else "  -  "
        print(f"{tag:28} | {_fmt(w.get('recall@20',0)):>9} {_fmt(w.get('ndcg@20',0)):>9} | "
              f"{_fmt(c.get('recall@20',0)):>9} {_fmt(c.get('ndcg@20',0)):>9} | "
              f"{bs:>6} {eff.get('adapter_params',0):>7} {eff.get('full_inference_ms',0):>5}")


def group_seeds(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for tag, payload in rows.items():
        base = re.sub(r"_s\d+$", "", tag)
        m = payload.get("metrics", {})
        for split in ("warm", "cold", "all"):
            for k in KEYS:
                v = m.get(split, {}).get(k)
                if v is not None:
                    groups[base][f"{split}/{k}"].append(v)
    return groups


def print_grouped(groups):
    print("\n=== grouped by config (mean +/- std over seeds) ===")
    head = f"{'config':28} | {'warm R@20':>16} | {'cold R@20':>16} | {'cold N@20':>16}"
    print(head)
    print("-" * len(head))
    for base in sorted(groups):
        g = groups[base]
        def cell(key):
            vals = g.get(key, [])
            if not vals:
                return " " * 16
            m, s = _mean_std(vals)
            return f"{m:.4f}+/-{s:.4f}"
        print(f"{base:28} | {cell('warm/recall@20'):>16} | "
              f"{cell('cold/recall@20'):>16} | {cell('cold/ndcg@20'):>16}")


def write_latex(groups, out_path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cold-start recommendation on the held-out cold-item pool "
        r"(mean$\pm$std over seeds). Warm performance is preserved.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Warm R@20 & Cold R@20 & Cold NDCG@20 \\",
        r"\midrule",
    ]

    def cell(g, key):
        vals = g.get(key, [])
        if not vals:
            return "--"
        m, s = _mean_std(vals)
        return f"{m:.4f}$\\pm${s:.4f}" if len(vals) > 1 else f"{m:.4f}"

    for base in sorted(groups):
        g = groups[base]
        name = base.replace("_", r"\_")
        lines.append(f"{name} & {cell(g,'warm/recall@20')} & "
                     f"{cell(g,'cold/recall@20')} & {cell(g,'cold/ndcg@20')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[saved] LaTeX table -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./runs")
    ap.add_argument("--out", default="./runs/tables.tex")
    args = ap.parse_args()
    rows = load_runs(args.runs)
    if not rows:
        print(f"no metrics.json under {args.runs}")
        return
    print_per_run(rows)
    groups = group_seeds(rows)
    print_grouped(groups)
    write_latex(groups, args.out)


if __name__ == "__main__":
    main()
