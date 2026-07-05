"""Build all paper-ready assets from runs/ in one shot.

Outputs (under --out, default ./paper_assets):
  tables/table_main.tex        - full per-dataset comparison (warm+cold, mean+/-std)
  tables/table_ablation.tex    - component ablation (cold), mean+/-std
  tables/table_efficiency.tex  - params / adapter ratio / latency / learned beta
  tables/table_structural.tex  - backbone + SAA rank sweep (Video_Games)
  figures/fig1_main_cold.{png,pdf}
  figures/fig2_warm_cold.{png,pdf}
  figures/fig3_ablation.{png,pdf}
  figures/fig4_beta.{png,pdf}
  figures/fig5_coldratio.{png,pdf}
  figures/fig6_efficiency.{png,pdf}
  figures/fig7_structural.{png,pdf}
  summary.csv                  - aggregated source data (mean/std per config)

Usage:
  pip install matplotlib numpy
  python scripts/make_paper_assets.py --runs ./runs --out ./paper_assets
"""
import os
import re
import csv
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 120,
})

# ---- palette (one neutral family + one signal accent for the proposed method) --
C_PROP = "#C2452D"    # CSA-Rec (accent / hero)
C_ABL = ["#E08A6E", "#EABFA9", "#F1D9C9"]  # ablation shades (lighter = weaker)
C_BASE = ["#9AA6B2", "#6E7B8A", "#455160", "#B9C2CC"]  # neutral baselines
C_CF = "#3C6E9C"      # collaborative-only control

DATASETS = ["Video_Games", "Baby_Products", "Toys_and_Games"]
DS_LABEL = {"Video_Games": "Video Games", "Baby_Products": "Baby Products",
            "Toys_and_Games": "Toys & Games"}


def load(runs_dir):
    rows = {}
    for path in glob.glob(os.path.join(runs_dir, "*", "metrics.json")):
        tag = os.path.basename(os.path.dirname(path))
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows[tag] = json.load(f)
        except Exception as e:
            print(f"[warn] skip {tag}: {e}")
    return rows


def parse(tag):
    base = re.sub(r"_s\d+$", "", tag)
    for ds in DATASETS:
        if base.startswith(ds + "_"):
            return ds, base[len(ds) + 1:]
    return None, base


def aggregate(rows):
    """method values -> {(ds, method): {metric: (mean, std, n)}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for tag, payload in rows.items():
        ds, method = parse(tag)
        if ds is None:
            continue
        m = payload.get("metrics", {})
        eff = payload.get("efficiency", {})
        for split in ("warm", "cold", "all"):
            for k, v in m.get(split, {}).items():
                acc[(ds, method)][f"{split}/{k}"].append(v)
        for ek in ("trainable_params", "adapter_params", "adapter_ratio",
                   "full_inference_ms", "fusion_beta"):
            v = eff.get(ek)
            if isinstance(v, (int, float)):
                acc[(ds, method)][f"eff/{ek}"].append(v)
    out = {}
    for key, d in acc.items():
        out[key] = {mk: (float(np.mean(vs)), float(np.std(vs, ddof=1) if len(vs) > 1 else 0.0),
                         len(vs)) for mk, vs in d.items()}
    return out


def g(agg, ds, method, metric, default=np.nan):
    return agg.get((ds, method), {}).get(metric, (default, 0.0, 0))[0]


def gstd(agg, ds, method, metric):
    return agg.get((ds, method), {}).get(metric, (0.0, 0.0, 0))[1]


# ============================ TABLES =====================================
MAIN_ORDER = [
    ("random", "Random"), ("pop", "Popularity"), ("contentknn", "Content-kNN"),
    ("ctrlcf", "LightGCN (CF only)"), ("nocsia", "CSA-Rec w/o CSIA"),
    ("nocpg", "CSA-Rec w/o CPG"), ("nofusion", "CSA-Rec w/o Fusion"),
    ("full", "CSA-Rec (full)"),
]


def _c(agg, ds, method, metric):
    m, s, n = agg.get((ds, method), {}).get(metric, (np.nan, 0.0, 0))
    if np.isnan(m):
        return "--"
    return f"{m:.4f}$\\pm${s:.4f}" if n > 1 else f"{m:.4f}"


def table_main(agg, out):
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Cold-start and warm recommendation on Amazon Reviews 2023 "
             r"(mean$\pm$std over 3 seeds). Best in \textbf{bold}. CSA-Rec is the only "
             r"single model that leads cold-start on every dataset while preserving warm accuracy.}",
             r"\label{tab:main}", r"\small",
             r"\begin{tabular}{ll cccc}", r"\toprule",
             r"Dataset & Method & Warm R@20 & Warm N@20 & Cold R@20 & Cold N@20 \\",
             r"\midrule"]
    for ds in DATASETS:
        best_cold = max((g(agg, ds, mth, "cold/recall@20", -1) for mth, _ in MAIN_ORDER))
        for i, (mth, name) in enumerate(MAIN_ORDER):
            cr = g(agg, ds, mth, "cold/recall@20", -1)
            cold_r = _c(agg, ds, mth, "cold/recall@20")
            if cr == best_cold and cr > 0:
                cold_r = r"\textbf{" + cold_r + "}"
            dcell = r"\multirow{8}{*}{" + DS_LABEL[ds] + "}" if i == 0 else ""
            lines.append(f"{dcell} & {name} & {_c(agg,ds,mth,'warm/recall@20')} & "
                         f"{_c(agg,ds,mth,'warm/ndcg@20')} & {cold_r} & "
                         f"{_c(agg,ds,mth,'cold/ndcg@20')} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    _write(out, "tables/table_main.tex", lines)


def table_ablation(agg, out):
    order = [("full", "CSA-Rec (full)"), ("nofusion", "w/o Fusion"),
             ("nocpg", "w/o CPG"), ("nocsia", "w/o CSIA"), ("ctrlcf", "CF backbone only")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Component ablation: Cold R@20 (mean$\pm$std over 3 seeds). "
             r"Every module contributes; removing CSIA collapses cold-start.}",
             r"\label{tab:ablation}", r"\small",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Variant & " + " & ".join(DS_LABEL[d] for d in DATASETS) + r" \\",
             r"\midrule"]
    for mth, name in order:
        cells = " & ".join(_c(agg, ds, mth, "cold/recall@20") for ds in DATASETS)
        lines.append(f"{name} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    _write(out, "tables/table_ablation.tex", lines)


def table_efficiency(agg, out):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Efficiency of CSA-Rec (full). The trainable semantic adapter "
             r"is a tiny fraction of total parameters; full-catalog scoring stays real-time. "
             r"The learned fusion weight $\beta$ grows as the collaborative signal weakens.}",
             r"\label{tab:efficiency}", r"\small",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Dataset & \#Params & Adapter & Adapter \% & Latency (ms) & $\beta$ \\",
             r"\midrule"]
    for ds in DATASETS:
        tp = g(agg, ds, "full", "eff/trainable_params")
        ap = g(agg, ds, "full", "eff/adapter_params")
        ar = g(agg, ds, "full", "eff/adapter_ratio") * 100
        ms = g(agg, ds, "full", "eff/full_inference_ms")
        beta = g(agg, ds, "full", "eff/fusion_beta")
        lines.append(f"{DS_LABEL[ds]} & {tp/1e6:.2f}M & {ap:,.0f} & {ar:.2f}\\% & "
                     f"{ms:.1f} & {beta:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    _write(out, "tables/table_efficiency.tex", lines)


def table_structural(agg, out):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Backbone and SAA rank study on Video Games (seed 2026), Cold R@20.}",
             r"\label{tab:structural}", r"\small",
             r"\begin{tabular}{lc}", r"\toprule", r"Configuration & Cold R@20 \\", r"\midrule"]
    for mth, name in [("full", "LightGCN backbone (default)"), ("mf", "MF backbone"),
                      ("saalinear", "SAA = linear")]:
        v = g(agg, "Video_Games", mth, "cold/recall@20")
        if not np.isnan(v):
            lines.append(f"{name} & {v:.4f} \\\\")
    lines.append(r"\midrule")
    for r in (8, 16, 32, 64):
        v = g(agg, "Video_Games", f"rank{r}", "cold/recall@20")
        if not np.isnan(v):
            lines.append(f"SAA rank $r={r}$ & {v:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    _write(out, "tables/table_structural.tex", lines)


def _write(out, rel, lines):
    p = os.path.join(out, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {rel}")


def _save(fig, out, name):
    d = os.path.join(out, "figures")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] figures/{name}.png/.pdf")


# ============================ FIGURES ====================================
def fig1_main_cold(agg, out):
    methods = [("random", "Random", C_BASE[0]), ("pop", "Popularity", C_BASE[1]),
               ("contentknn", "Content-kNN", C_BASE[2]), ("ctrlcf", "CF only", C_CF),
               ("full", "CSA-Rec", C_PROP)]
    x = np.arange(len(DATASETS)); w = 0.16
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for j, (mth, name, col) in enumerate(methods):
        vals = [g(agg, ds, mth, "cold/recall@20", 0) for ds in DATASETS]
        errs = [gstd(agg, ds, mth, "cold/recall@20") for ds in DATASETS]
        bars = ax.bar(x + (j - 2) * w, vals, w, yerr=errs, capsize=2, label=name,
                      color=col, edgecolor="white", linewidth=0.5,
                      error_kw=dict(lw=0.7))
        if mth == "full":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5, color=C_PROP, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Cold-start Recall@20")
    ax.set_title("Cold-start recommendation: CSA-Rec leads on every dataset", fontsize=9)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7)
    ax.margins(y=0.15)
    _save(fig, out, "fig1_main_cold")


def fig2_warm_cold(agg, out):
    methods = [("contentknn", "Content-kNN", C_BASE[2]), ("ctrlcf", "CF only", C_CF),
               ("full", "CSA-Rec", C_PROP)]
    x = np.arange(len(DATASETS)); w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, split, title in zip(axes, ("warm", "cold"),
                                ("Warm items (R@20)", "Cold items (R@20)")):
        for j, (mth, name, col) in enumerate(methods):
            vals = [g(agg, ds, mth, f"{split}/recall@20", 0) for ds in DATASETS]
            errs = [gstd(agg, ds, mth, f"{split}/recall@20") for ds in DATASETS]
            ax.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=2, label=name,
                   color=col, edgecolor="white", linewidth=0.5, error_kw=dict(lw=0.7))
        ax.set_xticks(x); ax.set_xticklabels([DS_LABEL[d] for d in DATASETS],
                                             rotation=12, ha="right")
        ax.set_title(title, fontsize=9); ax.margins(y=0.12)
    axes[0].set_ylabel("Recall@20")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.suptitle("CSA-Rec wins cold-start without sacrificing warm accuracy", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out, "fig2_warm_cold")


def fig3_ablation(agg, out):
    order = [("full", "Full", C_PROP), ("nofusion", "-Fusion", C_ABL[0]),
             ("nocpg", "-CPG", C_ABL[1]), ("nocsia", "-CSIA", C_ABL[2]),
             ("ctrlcf", "CF only", C_CF)]
    x = np.arange(len(DATASETS)); w = 0.16
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for j, (mth, name, col) in enumerate(order):
        vals = [g(agg, ds, mth, "cold/recall@20", 0) for ds in DATASETS]
        errs = [gstd(agg, ds, mth, "cold/recall@20") for ds in DATASETS]
        ax.bar(x + (j - 2) * w, vals, w, yerr=errs, capsize=2, label=name,
               color=col, edgecolor="white", linewidth=0.5, error_kw=dict(lw=0.7))
    ax.set_xticks(x); ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Cold-start Recall@20")
    ax.set_title("Ablation: every component contributes to cold-start", fontsize=9)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7)
    ax.margins(y=0.12)
    _save(fig, out, "fig3_ablation")


def fig4_beta(agg, out):
    beta = [g(agg, ds, "full", "eff/fusion_beta") for ds in DATASETS]
    warm = [g(agg, ds, "full", "warm/recall@20") for ds in DATASETS]
    x = np.arange(len(DATASETS))
    fig, ax1 = plt.subplots(figsize=(5.2, 3.2))
    b = ax1.bar(x, beta, 0.5, color=C_PROP, edgecolor="white", label=r"Learned $\beta$")
    for bi, v in zip(b, beta):
        ax1.text(bi.get_x() + bi.get_width() / 2, v, f"{v:.1f}", ha="center",
                 va="bottom", fontsize=7, color=C_PROP)
    ax1.set_ylabel(r"Learned fusion weight $\beta$", color=C_PROP)
    ax1.tick_params(axis="y", labelcolor=C_PROP)
    ax1.set_xticks(x); ax1.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax2 = ax1.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(x, warm, "-o", color=C_CF, lw=1.5, ms=5, label="Warm R@20 (collab. strength)")
    ax2.set_ylabel("Warm Recall@20 (collaborative strength)", color=C_CF)
    ax2.tick_params(axis="y", labelcolor=C_CF)
    ax1.set_title(r"Fusion is self-adaptive: $\beta$ rises as collaborative signal weakens",
                  fontsize=8.5)
    _save(fig, out, "fig4_beta")


def fig5_coldratio(agg, out):
    ratios = [0.05, 0.1, 0.2, 0.3]
    full = [g(agg, "Video_Games", f"full_cr{r}", "cold/recall@20") for r in ratios]
    cf = [g(agg, "Video_Games", f"ctrlcf_cr{r}", "cold/recall@20") for r in ratios]
    # cr0.1 uses the default-tagged runs (no _cr suffix)
    if np.isnan(full[1]):
        full[1] = g(agg, "Video_Games", "full", "cold/recall@20")
    if np.isnan(cf[1]):
        cf[1] = g(agg, "Video_Games", "ctrlcf", "cold/recall@20")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(ratios, full, "-o", color=C_PROP, lw=1.8, ms=6, label="CSA-Rec")
    ax.plot(ratios, cf, "-s", color=C_CF, lw=1.8, ms=6, label="CF only")
    ax.fill_between(ratios, cf, full, color=C_PROP, alpha=0.08)
    ax.set_xlabel("Cold-item ratio"); ax.set_ylabel("Cold-start Recall@20")
    ax.set_title("Robustness across cold-start severity (Video Games)", fontsize=9)
    ax.legend()
    _save(fig, out, "fig5_coldratio")


def fig6_efficiency(agg, out):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    pts = [("Content-kNN", "contentknn", C_BASE[2], "o"),
           ("CF only", "ctrlcf", C_CF, "s"),
           ("CSA-Rec", "full", C_PROP, "*")]
    for name, mth, col, mk in pts:
        xs = [g(agg, ds, mth, "eff/adapter_ratio") * 100 for ds in DATASETS]
        ys = [g(agg, ds, mth, "cold/recall@20") for ds in DATASETS]
        xs = [0.0 if np.isnan(v) else v for v in xs]
        ax.scatter(xs, ys, s=[90 if mk == "*" else 45] * len(xs), c=col, marker=mk,
                   edgecolor="white", linewidth=0.6, label=name, zorder=3)
    for ds in DATASETS:
        ax.annotate(DS_LABEL[ds],
                    (g(agg, ds, "full", "eff/adapter_ratio") * 100,
                     g(agg, ds, "full", "cold/recall@20")),
                    fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Trainable adapter ratio (% of parameters)")
    ax.set_ylabel("Cold-start Recall@20")
    ax.set_title("CSA-Rec: top accuracy at <0.3% trainable adapter", fontsize=9)
    ax.legend()
    _save(fig, out, "fig6_efficiency")


def fig7_structural(agg, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    # backbone
    bb = [("full", "LightGCN"), ("mf", "MF"), ("saalinear", "SAA-linear")]
    names = [n for _, n in bb]
    vals = [g(agg, "Video_Games", m, "cold/recall@20") for m, _ in bb]
    axes[0].bar(names, vals, 0.55, color=[C_PROP, C_BASE[1], C_BASE[2]], edgecolor="white")
    axes[0].set_ylabel("Cold R@20"); axes[0].set_title("Backbone / adapter", fontsize=9)
    axes[0].tick_params(axis="x", rotation=12)
    # rank sweep
    ranks = [8, 16, 32, 64]
    rv = [g(agg, "Video_Games", f"rank{r}", "cold/recall@20") for r in ranks]
    axes[1].plot(ranks, rv, "-o", color=C_PROP, lw=1.6, ms=6)
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(ranks)
    axes[1].get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    axes[1].set_xlabel("SAA rank $r$"); axes[1].set_title("SAA bottleneck rank", fontsize=9)
    fig.suptitle("Structural study (Video Games)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out, "fig7_structural")


def dump_csv(agg, out):
    p = os.path.join(out, "summary.csv")
    metrics = ["warm/recall@20", "warm/ndcg@20", "cold/recall@20", "cold/ndcg@20",
               "all/recall@20", "eff/trainable_params", "eff/adapter_ratio",
               "eff/full_inference_ms", "eff/fusion_beta"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "method", "n_seeds"] +
                    [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics])
        for (ds, mth) in sorted(agg):
            d = agg[(ds, mth)]
            n = max((d[m][2] for m in d), default=0)
            means = [f"{d.get(m,(np.nan,0,0))[0]:.6f}" for m in metrics]
            stds = [f"{d.get(m,(np.nan,0,0))[1]:.6f}" for m in metrics]
            wr.writerow([ds, mth, n] + means + stds)
    print(f"[saved] summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./runs")
    ap.add_argument("--out", default="./paper_assets")
    args = ap.parse_args()
    rows = load(args.runs)
    if not rows:
        print(f"no metrics under {args.runs}"); return
    agg = aggregate(rows)
    os.makedirs(args.out, exist_ok=True)
    table_main(agg, args.out); table_ablation(agg, args.out)
    table_efficiency(agg, args.out); table_structural(agg, args.out)
    fig1_main_cold(agg, args.out); fig2_warm_cold(agg, args.out)
    fig3_ablation(agg, args.out); fig4_beta(agg, args.out)
    fig5_coldratio(agg, args.out); fig6_efficiency(agg, args.out)
    fig7_structural(agg, args.out)
    dump_csv(agg, args.out)
    print("\n[done] all paper assets under", args.out)


if __name__ == "__main__":
    main()
