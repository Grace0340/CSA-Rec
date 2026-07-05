"""One-shot authoritative number dump for the manuscript (all seeds available).

Prints mean+/-std (n) for every metric the paper cites, so manuscript edits are
exact. Covers K in {10,20}, warm/cold R/N, learned baselines, temporal split,
encoder scaling, sensitivity, and efficiency/beta.
"""
import os, json, glob, re
import numpy as np
from collections import defaultdict

RUNS = "runs"
CATS = ["Video_Games", "Baby_Products", "Toys_and_Games"]

def load(tag):
    p = os.path.join(RUNS, tag, "metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))

def agg(prefix, seeds, metric_path):
    """metric_path like ('metrics','cold','recall@20') or ('efficiency','fusion_beta')."""
    vals = []
    for s in seeds:
        d = load(f"{prefix}_s{s}")
        if d is None:
            continue
        cur = d
        ok = True
        for k in metric_path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False; break
        if ok and isinstance(cur, (int, float)):
            vals.append(cur)
    if not vals:
        return None
    return np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0), len(vals)

def fmt(a):
    return "--" if a is None else f"{a[0]:.4f}$\\pm${a[1]:.4f} (n={a[2]})"

SEEDS5 = (2026, 2027, 2028, 2029, 2030)
SEEDS3 = (2026, 2027, 2028)

print("="*70)
print("MAIN TABLE methods (warm/cold, R/N, K=10 & 20)")
print("="*70)
methods = [("random", SEEDS3), ("pop", SEEDS3), ("contentknn", SEEDS5),
           ("ctrlcf", SEEDS3), ("nocsia", SEEDS3), ("nocpg", SEEDS3),
           ("nofusion", SEEDS3), ("full", SEEDS5)]
for c in CATS:
    print(f"\n### {c}")
    for m, seeds in methods:
        row = []
        for split in ("warm", "cold"):
            for K in (20, 10):
                for met in ("recall", "ndcg"):
                    a = agg(f"{c}_{m}", seeds, ("metrics", split, f"{met}@{K}"))
                    row.append(f"{split[0]}{met[0].upper()}@{K}={fmt(a)}")
        print(f"  {m:11}", " | ".join(row))

print("\n" + "="*70)
print("LEARNED COLD-START BASELINES (cold R/N @20 and @10)")
print("="*70)
for c in CATS:
    print(f"\n### {c}")
    for m in ("content2emb", "dropoutnet", "clcrec"):
        cells = []
        for K in (20, 10):
            for met in ("recall", "ndcg"):
                a = agg(f"{c}_{m}", SEEDS5, ("metrics", "cold", f"{met}@{K}"))
                cells.append(f"cold {met[0].upper()}@{K}={fmt(a)}")
        # warm (shared backbone)
        wa = agg(f"{c}_{m}", SEEDS5, ("metrics", "warm", "recall@20"))
        print(f"  {m:11}", " | ".join(cells), f" || warmR@20={fmt(wa)}")

print("\n" + "="*70)
print("TEMPORAL SPLIT (cold R/N @20 & @10)")
print("="*70)
for c in CATS:
    print(f"\n### {c}")
    for m in ("full_temporal", "contentknn_temporal", "ctrlcf_temporal"):
        cells = []
        for K in (20, 10):
            for met in ("recall", "ndcg"):
                a = agg(f"{c}_{m}", SEEDS3, ("metrics", "cold", f"{met}@{K}"))
                cells.append(f"{met[0].upper()}@{K}={fmt(a)}")
        print(f"  {m:20}", " | ".join(cells))

print("\n" + "="*70)
print("ENCODER SCALING (MiniLM 384 vs mpnet 768), cold R/N @20")
print("="*70)
for c in CATS:
    print(f"\n### {c}")
    for m in ("full", "full_mpnet", "contentknn", "contentknn_mpnet"):
        seeds = SEEDS5 if m in ("full", "contentknn") else SEEDS3
        a = agg(f"{c}_{m}", seeds, ("metrics", "cold", "recall@20"))
        an = agg(f"{c}_{m}", seeds, ("metrics", "cold", "ndcg@20"))
        wr = agg(f"{c}_{m}", seeds, ("metrics", "warm", "recall@20"))
        print(f"  {m:18} coldR@20={fmt(a)}  coldN@20={fmt(an)}  warmR@20={fmt(wr)}")

print("\n" + "="*70)
print("SENSITIVITY on Video_Games (seed 2026), cold R@20 / N@20")
print("="*70)
ref = load("Video_Games_full_s2026")["metrics"]["cold"]
print(f"  reference full (lam0.5,K10,tau0.2): R@20={ref['recall@20']:.4f} N@20={ref['ndcg@20']:.4f}")
for group, tags in [("lambda", ["lam0.1", "lam0.2", "lam1.0"]),
                    ("K", ["K5", "K20", "K40"]),
                    ("tau", ["tau0.1", "tau0.5"])]:
    for t in tags:
        d = load(f"Video_Games_{t}_s2026")
        if d:
            cc = d["metrics"]["cold"]
            print(f"  {group:6} {t:8} R@20={cc['recall@20']:.4f} N@20={cc['ndcg@20']:.4f}")

print("\n" + "="*70)
print("EFFICIENCY / BETA (full, mean over 5 seeds)")
print("="*70)
for c in CATS:
    beta = agg(f"{c}_full", SEEDS5, ("efficiency", "fusion_beta"))
    tp = agg(f"{c}_full", SEEDS5, ("efficiency", "trainable_params"))
    ms = agg(f"{c}_full", SEEDS5, ("efficiency", "full_inference_ms"))
    print(f"  {c:16} beta={fmt(beta)}  params={tp[0]:.0f}  latency_ms={ms[0]:.1f}")
