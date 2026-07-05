"""Hyperparameter-sensitivity figure (fig8) for the manuscript.

Reads the P5 runs (Video Games, seed 2026) directly and renders three panels
(lambda, K, tau) of cold-start Recall@20 with the default setting highlighted.
The narrow y-range is intentional: it shows the method is robust, not tuned.

Usage:
    python scripts/make_sensitivity_fig.py --runs runs --out ../manuscript
"""
import os, json, argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5, "figure.dpi": 120,
})
C_PROP = "#C2452D"; C_PROP_D = "#8F2F1E"; C_GAIN = "#2E9E44"; C_GREY = "#8A94A0"


def cold_r20(runs, tag):
    p = os.path.join(runs, tag, "metrics.json")
    if not os.path.exists(p):
        return np.nan
    return json.load(open(p, encoding="utf-8"))["metrics"]["cold"]["recall@20"] * 100


def panel(ax, xs, ys, default_x, xlabel, title, logbase=None):
    ax.plot(xs, ys, "-o", color=C_PROP, lw=1.7, ms=6, mec="white", mew=0.7, zorder=4)
    di = xs.index(default_x)
    ax.scatter([default_x], [ys[di]], s=130, facecolor="none",
               edgecolor=C_GAIN, linewidth=1.5, zorder=5)
    ax.annotate("default", (default_x, ys[di]),
                xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.6, color=C_GAIN, fontweight="bold")
    if logbase:
        ax.set_xscale("log", base=logbase); ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    lo, hi = min(ys), max(ys)
    pad = max(0.15, (hi - lo) * 0.6)
    ax.set_ylim(lo - pad, hi + pad + 0.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8.3, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="../manuscript")
    a = ap.parse_args()
    R = a.runs
    ref = cold_r20(R, "Video_Games_full_s2026")  # default lam0.5,K10,tau0.2

    lam_x = [0.1, 0.2, 0.5, 1.0]
    lam_y = [cold_r20(R, "Video_Games_lam0.1_s2026"),
             cold_r20(R, "Video_Games_lam0.2_s2026"), ref,
             cold_r20(R, "Video_Games_lam1.0_s2026")]
    K_x = [5, 10, 20, 40]
    K_y = [cold_r20(R, "Video_Games_K5_s2026"), ref,
           cold_r20(R, "Video_Games_K20_s2026"),
           cold_r20(R, "Video_Games_K40_s2026")]
    tau_x = [0.1, 0.2, 0.5]
    tau_y = [cold_r20(R, "Video_Games_tau0.1_s2026"), ref,
             cold_r20(R, "Video_Games_tau0.5_s2026")]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
    panel(axes[0], lam_x, lam_y, 0.5, r"CSIA weight $\lambda$",
          r"(a) Alignment weight $\lambda$")
    panel(axes[1], K_x, K_y, 10, r"CPG neighbours $K$",
          r"(b) Neighbourhood size $K$", logbase=2)
    panel(axes[2], tau_x, tau_y, 0.2, r"CSIA temperature $\tau$",
          r"(c) Temperature $\tau$")
    axes[0].set_ylabel("Cold-start Recall@20 (%)", color=C_PROP)
    axes[0].tick_params(axis="y", labelcolor=C_PROP)
    fig.tight_layout(w_pad=1.3)
    d = os.path.join(a.out, "figures")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "fig8_sensitivity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(d, "fig8_sensitivity.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/fig8_sensitivity.pdf/.png  (Video Games, seed 2026)")


if __name__ == "__main__":
    main()
