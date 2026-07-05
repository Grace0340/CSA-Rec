"""Publication-grade figure set for the CSA-Rec manuscript (v2).

Reads the aggregated ``summary.csv`` produced by ``make_paper_assets.py`` and
renders a richer, submission-grade figure set: a schematic-led architecture
composite, a warm-cold Pareto plane, slopegraphs, an ablation heatmap, a
self-calibration scatter with fit, a dual-axis robustness panel, a
parameter-accuracy efficiency frontier, and a structural study with an
overfitting band.

Usage:
    python scripts/make_figures_v2.py --summary paper_assets/summary.csv \
        --out manuscript
"""
import os
import csv
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "figure.dpi": 120,
})

# ---- coherent palette: neutral family + signal + accent ------------------
C_PROP = "#C2452D"   # CSA-Rec (accent / hero)
C_PROP_D = "#8F2F1E"
C_CF = "#3C6E9C"     # collaborative-only
C_KNN = "#4E9E8F"    # content-only
C_POP = "#B7791F"    # popularity
C_RND = "#9AA6B2"    # random
C_GAIN = "#2E9E44"   # directional: improvement
C_INK = "#2A2A2A"
C_FROZEN = "#3C6E9C"
C_TRAIN = "#C2452D"
C_BONE = "#B7791F"
C_GREY = "#8A94A0"

DATASETS = ["Video_Games", "Baby_Products", "Toys_and_Games"]
DS_LABEL = {"Video_Games": "Video Games", "Baby_Products": "Baby Products",
            "Toys_and_Games": "Toys & Games"}
DS_SHORT = {"Video_Games": "Video\nGames", "Baby_Products": "Baby\nProducts",
            "Toys_and_Games": "Toys &\nGames"}

METRICS = ["warm/recall@20", "warm/ndcg@20", "cold/recall@20", "cold/ndcg@20",
           "all/recall@20", "eff/trainable_params", "eff/adapter_ratio",
           "eff/full_inference_ms", "eff/fusion_beta"]


# ============================ DATA ========================================
def load_summary(path):
    D = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["method"])
            rec = {"n": int(row["n_seeds"])}
            for m in METRICS:
                mv = row.get(f"{m}_mean", "")
                sv = row.get(f"{m}_std", "")
                rec[m] = (float(mv) if mv not in ("", "nan") else np.nan,
                          float(sv) if sv not in ("", "nan") else 0.0)
            D[key] = rec
    return D


def gm(D, ds, method, metric, default=np.nan):
    return D.get((ds, method), {}).get(metric, (default, 0.0))[0]


def gs(D, ds, method, metric):
    return D.get((ds, method), {}).get(metric, (0.0, 0.0))[1]


def save(fig, out, name, w=None, h=None):
    for sub in ("figures",):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    fig.savefig(os.path.join(out, "figures", name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "figures", name + ".png"), dpi=600,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] figures/{name}.pdf/.png")


def rbox(ax, x, y, w, h, fc, ec, lw=1.2, rad=0.02, alpha=1.0, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha,
                       zorder=z, mutation_aspect=1.0)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=C_INK, lw=1.3, style="-|>", rad=0.0,
          dashed=False, z=3, ms=9):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        lw=lw, color=color, zorder=z,
                        connectionstyle=f"arc3,rad={rad}",
                        linestyle=(0, (4, 2)) if dashed else "solid",
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)
    return a


def route(ax, pts, color=C_INK, lw=1.3, dashed=False, z=3, ms=9):
    """Orthogonal multi-segment connector with an arrowhead at the last point."""
    ls = (0, (4, 2)) if dashed else "solid"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z,
                solid_capstyle="round", solid_joinstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, dashed=dashed, z=z, ms=ms)


# ============================ FIG 1: architecture =========================
def fig_arch(D, out):
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(7.16, 3.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def cloud(x, y, w, h, seed_shift=0):
        ca = rng.normal([0.34, 0.63], 0.09, (13, 2))
        cb = rng.normal([0.67, 0.37], 0.09, (13, 2))
        for c, col in ((ca, C_KNN), (cb, C_POP)):
            ax.scatter(x + np.clip(c[:, 0], 0, 1) * w * 0.8 + w * 0.1,
                       y + np.clip(c[:, 1], 0, 1) * h * 0.72 + h * 0.14,
                       s=6, color=col, alpha=0.8, zorder=4, edgecolor="none")

    Y_TOP, Y_BOT = 74.5, 10.5          # box bottoms for the two lanes
    HB = 15.0                           # lane box height
    # lane guide labels
    ax.text(1.5, Y_TOP + HB + 3.0, "CONTENT PATHWAY (frozen)", fontsize=6.6,
            color=C_FROZEN, fontweight="bold")
    # keep below the bottom lane but above the perimeter score route / legend
    ax.text(1.5, 6.6, "COLLABORATIVE PATHWAY (BPR-trained)", fontsize=6.1,
            color=C_BONE, fontweight="bold", va="top")

    # ---- content lane (top) ----
    rbox(ax, 3, Y_TOP, 14, HB, "#FFFFFF", C_FROZEN, lw=1.1, rad=1.2)
    ax.text(10, Y_TOP + 11.3, "Item content", ha="center", fontsize=7.2,
            fontweight="bold")
    for k, t in enumerate(["title", "features", "description"]):
        ax.text(10, Y_TOP + 8.0 - k * 2.7, t, ha="center", fontsize=6.0,
                color="#5A6570")

    rbox(ax, 21, Y_TOP + 1.5, 13, HB - 3, "#EAF1F7", C_FROZEN, lw=1.1, rad=1.2)
    ax.text(27.5, Y_TOP + 9.6, "Frozen", ha="center", fontsize=7.0,
            fontweight="bold", color=C_FROZEN)
    ax.text(27.5, Y_TOP + 6.6, "encoder $\\phi$", ha="center", fontsize=6.8,
            color=C_FROZEN)
    ax.text(27.5, Y_TOP + 3.9, "SBERT", ha="center", fontsize=5.9, color="#5A6570")

    rbox(ax, 38, Y_TOP, 13, HB, "#FFFFFF", "#C7D3DE", lw=0.9, rad=1.0)
    ax.text(44.5, 93.2, "Semantic space $\\mathbf{s}_i$", ha="center",
            fontsize=6.6, color=C_FROZEN)
    cloud(38, Y_TOP, 13, HB)

    # ---- collaborative lane (bottom) ----
    rbox(ax, 3, Y_BOT, 14, HB, "#FFFFFF", C_BONE, lw=1.1, rad=1.2)
    ax.text(10, Y_BOT + 10.3, "Interaction", ha="center", fontsize=7.0,
            fontweight="bold")
    ax.text(10, Y_BOT + 7.4, "graph", ha="center", fontsize=7.0, fontweight="bold")
    ax.text(10, Y_BOT + 4.0, "warm items", ha="center", fontsize=6.0,
            color="#5A6570")

    rbox(ax, 21, Y_BOT + 1.5, 13, HB - 3, "#FBEEDD", C_BONE, lw=1.1, rad=1.2)
    ax.text(27.5, Y_BOT + 8.0, "LightGCN", ha="center", fontsize=7.0,
            fontweight="bold", color="#8A5A12")
    ax.text(27.5, Y_BOT + 4.7, "$\\mathbf{e}_u,\\mathbf{e}_i$", ha="center",
            fontsize=7.0, color="#8A5A12")

    rbox(ax, 38, Y_BOT, 13, HB, "#FFFFFF", "#E1CBA6", lw=0.9, rad=1.0)
    ax.text(44.5, Y_BOT - 2.0, "Collaborative space $\\mathbf{e}_i$", ha="center",
            fontsize=6.6, color="#8A5A12")
    cloud(38, Y_BOT, 13, HB)

    # ---- trainable-bridge container (center) ----
    BX, BW = 56.5, 18.5
    rbox(ax, BX, 21, BW, 62, "#FCF3F0", "#E7C3B8", lw=1.0, rad=1.3, z=1)
    ax.text(BX + BW / 2, 79.0, "TRAINABLE BRIDGE", ha="center", fontsize=6.7,
            color=C_TRAIN, fontweight="bold")
    ax.text(BX + BW / 2, 76.0, "11,409 params", ha="center", fontsize=6.0,
            color="#9A5A48")

    def module(y0, h, title, sub1, sub2):
        rbox(ax, BX + 2.2, y0, BW - 4.4, h, "#FFFFFF", C_TRAIN, lw=1.25, rad=1.0,
             z=2)
        cx = BX + BW / 2
        ax.text(cx, y0 + h - 3.2, title, ha="center", fontsize=7.4,
                fontweight="bold", color=C_TRAIN)
        ax.text(cx, y0 + h - 6.4, sub1, ha="center", fontsize=5.9, color="#7A3325")
        if sub2:
            ax.text(cx, y0 + h - 9.0, sub2, ha="center", fontsize=5.9,
                    color="#7A3325")

    module(62, 11.5, "SAA", "low-rank projection", "$384\\!\\to\\!16\\!\\to\\!64$")
    module(45.5, 11.5, "CSIA", "InfoNCE alignment", "stop-grad on IDs")
    module(24.5, 11.5, "CPG", "warm-neighbor", "aggregation")

    # ---- fusion + output (right) ----
    FX = 80.5
    rbox(ax, FX, 46, 16.5, 15, "#F0F8F1", C_GAIN, lw=1.3, rad=1.2)
    ax.text(FX + 8.25, 57.4, "Adaptive fusion", ha="center", fontsize=7.2,
            fontweight="bold", color="#1F7A33")
    ax.text(FX + 8.25, 53.2, "$\\hat y_{ui}=\\mathbf{e}_u^{\\!\\top}\\tilde{\\mathbf{e}}_i"
            "+\\beta\\,\\mathbf{p}_u^{\\!\\top}\\mathbf{s}_i$", ha="center",
            fontsize=6.0, color="#1F7A33")
    ax.text(FX + 8.25, 48.7, "$\\beta=\\mathrm{softplus}(\\tilde\\beta)$, one scalar",
            ha="center", fontsize=5.8, color="#3f8a4f")

    rbox(ax, FX + 1.5, 24, 13.5, 14, "#FFFFFF", C_INK, lw=1.1, rad=1.1)
    ax.text(FX + 8.25, 35.0, "Top-$K$", ha="center", fontsize=7.2,
            fontweight="bold")
    ax.text(FX + 8.25, 32.2, "ranking", ha="center", fontsize=6.0, color="#5A6570")
    for k, bh in enumerate([9.5, 6.8, 4.6, 3.0]):
        ax.add_patch(Rectangle((FX + 3.0, 25.0 + k * 1.7), bh, 1.15,
                     color=C_PROP if k == 0 else C_GREY, alpha=0.9, zorder=4))

    # ---- connectors -----------------------------------------------------
    # content lane
    arrow(ax, (17, Y_TOP + HB / 2), (21, Y_TOP + HB / 2), color=C_FROZEN)
    arrow(ax, (34, Y_TOP + HB / 2), (38, Y_TOP + HB / 2), color=C_FROZEN)
    # collab lane
    arrow(ax, (17, Y_BOT + HB / 2), (21, Y_BOT + HB / 2), color=C_BONE)
    arrow(ax, (34, Y_BOT + HB / 2), (38, Y_BOT + HB / 2), color=C_BONE)
    # semantic -> SAA (into bridge top)
    arrow(ax, (51, Y_TOP + HB / 2), (BX + 2.2, 67.7), color=C_FROZEN)
    ax.text(53.2, Y_TOP + HB / 2 + 1.4, "$\\mathbf{s}_i$", fontsize=6.6,
            color=C_FROZEN)
    # collab IDs -> bridge (dashed, stop-grad): drop up from the collaborative
    # space, then run straight into the CPG left edge at its vertical centre
    route(ax, [(51, Y_BOT + HB / 2), (54.5, Y_BOT + HB / 2), (54.5, 30.25),
               (BX + 2.2, 30.25)], color=C_BONE, dashed=True)
    # label sits on that horizontal entry line, in the open area left of CPG
    ax.annotate(
        "warm IDs $\\mathbf{e}_i$ (stop-grad)",
        xy=(54.5, 30.25), xytext=(42.5, 30.25),
        fontsize=5.7, color=C_BONE, ha="center", va="center",
        arrowprops=dict(arrowstyle="-", color=C_BONE, lw=0.9,
                        linestyle=(0, (4, 2)), shrinkA=4, shrinkB=2),
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="#E1CBA6", alpha=0.95, lw=0.6),
    )
    # internal: SAA -> CSIA -> CPG
    arrow(ax, (BX + BW / 2, 62), (BX + BW / 2, 57), color=C_TRAIN, ms=8)
    arrow(ax, (BX + BW / 2, 45.5), (BX + BW / 2, 36), color=C_TRAIN, ms=8)
    # bridge (CPG cold repr) -> fusion left side : item representation
    route(ax, [(BX + BW, 30.25), (78.5, 30.25), (78.5, 52), (FX, 52)],
          color=C_TRAIN)
    ax.text(76.75, 32.6, "$\\tilde{\\mathbf{e}}_i$", fontsize=6.4,
            color=C_TRAIN, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.95))
    # content score perimeter route (top): semantic -> fusion top
    route(ax, [(40, Y_TOP + HB), (40, 92), (FX + 8.25, 92), (FX + 8.25, 61)],
          color=C_FROZEN)
    ax.text(64, 93.2, "content score  $\\beta\\,\\mathbf{p}_u^{\\!\\top}\\mathbf{s}_i$",
            fontsize=6.0, color=C_FROZEN, ha="center")
    # collaborative score perimeter route (bottom): backbone user -> fusion left
    route(ax, [(27.5, Y_BOT), (27.5, 3.0), (76, 3.0), (76, 47), (FX, 47)],
          color=C_BONE)
    ax.text(58, 4.6,
            "collaborative score  $\\mathbf{e}_u^{\\!\\top}\\tilde{\\mathbf{e}}_i$",
            fontsize=6.0, color=C_BONE, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.94))
    # fusion -> ranking
    arrow(ax, (FX + 8.25, 46), (FX + 8.25, 38), color=C_GAIN)

    # legend (top strip)
    handles = [
        Line2D([0], [0], marker="s", ls="none", ms=7, mfc="#EAF1F7",
               mec=C_FROZEN, label="Frozen / content (offline)"),
        Line2D([0], [0], marker="s", ls="none", ms=7, mfc="#FBEEDD",
               mec=C_BONE, label="Collaborative backbone"),
        Line2D([0], [0], marker="s", ls="none", ms=7, mfc="#FFFFFF",
               mec=C_TRAIN, label="Trainable bridge"),
        Line2D([0], [0], ls=(0, (4, 2)), color=C_INK, label="No gradient"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.035),
              ncol=4, fontsize=6.2, handletextpad=0.4, columnspacing=1.1,
              borderpad=0.4)
    save(fig, out, "arch_fig")


# ============================ FIG 2: main + pareto ========================
def fig_main(D, out):
    fig = plt.figure(figsize=(7.16, 2.9))
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)

    # (a) Pareto plane on Video Games
    ax = fig.add_subplot(gsp[0, 0])
    ds = "Video_Games"
    pts = [("Random", "random", C_RND, "o"),
           ("Popularity", "pop", C_POP, "P"),
           ("Content-kNN", "contentknn", C_KNN, "D"),
           ("LightGCN (CF)", "ctrlcf", C_CF, "s"),
           ("CSA-Rec", "full", C_PROP, "*")]
    xs, ys = [], []
    for name, m, col, mk in pts:
        x = gm(D, ds, m, "warm/recall@20") * 100
        y = gm(D, ds, m, "cold/recall@20") * 100
        xs.append(x); ys.append(y)
        sz = 340 if mk == "*" else 90
        ax.scatter(x, y, s=sz, marker=mk, color=col, edgecolor="white",
                   linewidth=0.8, zorder=5)
        dx, dy = (0.15, 0.4)
        ha = "left"
        if name == "Content-kNN":
            dx, dy, ha = -0.15, 0.5, "right"
        if name == "LightGCN (CF)":
            dx, dy = 0.1, -1.4
        if name == "Random":
            dx, dy = 0.15, 0.3
        ax.annotate(name, (x, y), xytext=(x + dx, y + dy), fontsize=6.8,
                    color=col, fontweight="bold" if mk == "*" else "normal",
                    ha=ha, zorder=6)
    # Pareto frontier (upper-right envelope)
    order = np.argsort(xs)
    fx = np.array(xs)[order]; fy = np.array(ys)[order]
    ax.axhline(0, color="#DDDDDD", lw=0.6, zorder=0)
    ax.set_xlabel("Warm Recall@20 (%)")
    ax.set_ylabel("Cold-start Recall@20 (%)")
    ax.set_title("(a) Warm\u2013cold trade-off plane (Video Games)",
                 fontsize=8.3, loc="left")
    ax.set_xlim(-0.5, 10.2)
    ax.set_ylim(-1.2, 20)
    # quadrant guides at CSA-Rec-ish
    ax.add_patch(Rectangle((5.2, 10.5), 5.0, 9.5, facecolor=C_PROP, alpha=0.06,
                           zorder=0))
    ax.text(5.5, 19.2, "strong on both sides", fontsize=6.4, color=C_PROP,
            ha="left", va="top", style="italic")

    # (b) grouped cold recall with gain annotations
    ax2 = fig.add_subplot(gsp[0, 1])
    keym = [("Content-kNN", "contentknn", C_KNN),
            ("LightGCN (CF)", "ctrlcf", C_CF),
            ("CSA-Rec", "full", C_PROP)]
    x = np.arange(len(DATASETS)); w = 0.26
    for j, (name, m, col) in enumerate(keym):
        vals = [gm(D, d, m, "cold/recall@20") * 100 for d in DATASETS]
        errs = [gs(D, d, m, "cold/recall@20") * 100 for d in DATASETS]
        bars = ax2.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=2,
                       label=name, color=col, edgecolor="white", linewidth=0.6,
                       error_kw=dict(lw=0.7))
        if m == "full":
            for b, d in zip(bars, DATASETS):
                cf = gm(D, d, "ctrlcf", "cold/recall@20")
                full = gm(D, d, "full", "cold/recall@20")
                mult = full / cf if cf > 0 else np.nan
                ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                         f"{mult:.0f}\u00d7", ha="center", va="bottom",
                         fontsize=6.6, color=C_GAIN, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([DS_SHORT[d] for d in DATASETS])
    ax2.set_ylabel("Cold-start Recall@20 (%)")
    ax2.set_title("(b) Cold-start gain over the collaborative backbone",
                  fontsize=8.3, loc="left")
    ax2.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.72, 1.02))
    ax2.margins(y=0.18)
    ax2.text(0.02, 0.96, "green = \u00d7 gain vs CF", transform=ax2.transAxes,
             fontsize=6.2, color=C_GAIN, style="italic")
    save(fig, out, "fig1_main_cold")


# ============================ FIG 3: slopegraphs ==========================
def fig_slope(D, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=False)
    methods = [("Content-kNN", "contentknn", C_KNN, "D"),
               ("LightGCN (CF)", "ctrlcf", C_CF, "s"),
               ("CSA-Rec", "full", C_PROP, "*")]
    for ax, ds in zip(axes, DATASETS):
        for name, m, col, mk in methods:
            warm = gm(D, ds, m, "warm/recall@20") * 100
            cold = gm(D, ds, m, "cold/recall@20") * 100
            ax.plot([0, 1], [warm, cold], "-", color=col, lw=1.6,
                    alpha=0.9, zorder=3)
            ax.scatter([0, 1], [warm, cold],
                       s=[70 if mk == "*" else 34] * 2, marker=mk, color=col,
                       edgecolor="white", linewidth=0.7, zorder=4)
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Warm", "Cold"])
        ax.set_title(DS_LABEL[ds], fontsize=8.2)
        ax.spines["bottom"].set_visible(True)
        ax.margins(y=0.16)
        ax.grid(axis="y", color="#EEEEEE", lw=0.6, zorder=0)
    axes[0].set_ylabel("Recall@20 (%)")
    handles = [Line2D([0], [0], color=c, marker=mk, lw=1.6,
                      ms=8 if mk == "*" else 5, mec="white",
                      label=n) for n, _, c, mk in methods]
    axes[1].legend(handles=handles, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.30), fontsize=7.2)
    fig.subplots_adjust(top=0.80, wspace=0.32)
    save(fig, out, "fig2_warm_cold")


# ============================ FIG 4: ablation + heatmap ===================
def fig_ablation(D, out):
    fig = plt.figure(figsize=(7.16, 2.7))
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.3)
    ax = fig.add_subplot(gsp[0, 0])
    order = [("Full", "full", C_PROP, None),
             ("w/o Fusion", "nofusion", "#D98E76", "//"),
             ("w/o CPG", "nocpg", "#E7B49E", "\\\\"),
             ("w/o CSIA", "nocsia", "#F0D2C2", ".."),
             ("CF only", "ctrlcf", C_CF, "xx")]
    x = np.arange(len(DATASETS)); w = 0.16
    for j, (name, m, col, hatch) in enumerate(order):
        vals = [gm(D, d, m, "cold/recall@20") * 100 for d in DATASETS]
        errs = [gs(D, d, m, "cold/recall@20") * 100 for d in DATASETS]
        ax.bar(x + (j - 2) * w, vals, w, yerr=errs, capsize=1.8, label=name,
               color=col, edgecolor="white", linewidth=0.5,
               hatch=hatch, error_kw=dict(lw=0.6))
    ax.set_xticks(x); ax.set_xticklabels([DS_SHORT[d] for d in DATASETS])
    ax.set_ylabel("Cold-start Recall@20 (%)")
    ax.set_title("(a) Component ablation", fontsize=8.3, loc="left")
    ax.legend(loc="upper right", ncol=1, fontsize=6.6)
    ax.margins(y=0.14)

    # (b) heatmap: relative % of cold-recall lost when a module is removed
    ax2 = fig.add_subplot(gsp[0, 1])
    mods = [("Fusion", "nofusion"), ("CPG", "nocpg"), ("CSIA", "nocsia")]
    M = np.zeros((len(mods), len(DATASETS)))
    for i, (mn, mm) in enumerate(mods):
        for k, ds in enumerate(DATASETS):
            full = gm(D, ds, "full", "cold/recall@20")
            abl = gm(D, ds, mm, "cold/recall@20")
            M[i, k] = (full - abl) / full * 100 if full > 0 else np.nan
    cmap = LinearSegmentedColormap.from_list("rd", ["#FCF2EE", "#E7A78E", C_PROP,
                                                    C_PROP_D])
    im = ax2.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=70)
    ax2.set_xticks(range(len(DATASETS)))
    ax2.set_xticklabels([DS_SHORT[d] for d in DATASETS])
    ax2.set_yticks(range(len(mods)))
    ax2.set_yticklabels([m[0] for m in mods])
    for i in range(len(mods)):
        for k in range(len(DATASETS)):
            val = M[i, k]
            ax2.text(k, i, f"{val:.0f}%", ha="center", va="center",
                     fontsize=7.6, fontweight="bold",
                     color="white" if val > 38 else C_INK)
    ax2.set_title("(b) Relative cold-recall lost\nwhen module removed",
                  fontsize=8.3, loc="left")
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6.5, length=0)
    cb.outline.set_visible(False)
    cb.set_label("% drop", fontsize=6.8)
    save(fig, out, "fig3_ablation")


# ============================ FIG 5: beta calibration =====================
def fig_beta(D, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    warm = np.array([gm(D, d, "ctrlcf", "warm/recall@20") * 100 for d in DATASETS])
    beta = np.array([gm(D, d, "full", "eff/fusion_beta") for d in DATASETS])
    berr = np.array([gs(D, d, "full", "eff/fusion_beta") for d in DATASETS])
    # cross-dataset points
    ax.errorbar(warm, beta, yerr=berr, fmt="none", ecolor=C_PROP, elinewidth=1.0,
                capsize=2.5, zorder=3)
    ax.scatter(warm, beta, s=150, marker="*", color=C_PROP, edgecolor="white",
               linewidth=0.8, zorder=5)
    for d, xw, yb in zip(DATASETS, warm, beta):
        off = {"Video_Games": (0.25, 0.9), "Baby_Products": (0.3, 0.4),
               "Toys_and_Games": (0.4, 0.2)}[d]
        ax.annotate(DS_LABEL[d], (xw, yb), xytext=(xw + off[0], yb + off[1]),
                    fontsize=6.8, color=C_PROP_D, ha="left")
    # inverse trend guide
    xt = np.linspace(warm.min() * 0.85, warm.max() * 1.1, 100)
    k = np.median(beta * warm)
    ax.plot(xt, k / xt, "--", color=C_GREY, lw=1.0, zorder=2,
            label=r"$\beta \propto 1/\,$(collab. strength)")
    # severity-sweep betas for Video Games (within-dataset drift)
    srt = [("full_cr0.05", 0.05), ("full_cr0.1", 0.10),
           ("full_cr0.2", 0.20), ("full_cr0.3", 0.30)]
    sb = [gm(D, "Video_Games", m, "eff/fusion_beta") for m, _ in srt]
    wvg = gm(D, "Video_Games", "ctrlcf", "warm/recall@20") * 100
    ax.scatter([wvg] * len(sb), sb, s=18, color=C_CF, alpha=0.7, zorder=4)
    ax.annotate("severity sweep\n(5\u201330% cold)", (wvg, min(sb)),
                xytext=(wvg - 0.3, min(sb) - 2.0), fontsize=6.0, color=C_CF,
                ha="right")
    ax.set_xlabel("Collaborative strength\n(backbone warm Recall@20, %)")
    ax.set_ylabel(r"Learned fusion weight $\beta$")
    ax.set_ylim(-1, 24)
    ax.set_title("Fusion self-calibrates to sparsity", fontsize=8.3)
    ax.legend(loc="upper right", fontsize=6.3)
    save(fig, out, "fig4_beta")


# ============================ FIG 6: severity dual-axis ===================
def fig_severity(D, out):
    ratios = [0.05, 0.10, 0.20, 0.30]
    full = np.array([gm(D, "Video_Games", f"full_cr{r}", "cold/recall@20") * 100
                     for r in (0.05, 0.1, 0.2, 0.3)])
    cf = np.array([gm(D, "Video_Games", f"ctrlcf_cr{r}", "cold/recall@20") * 100
                   for r in (0.05, 0.1, 0.2, 0.3)])
    beta = np.array([gm(D, "Video_Games", f"full_cr{r}", "eff/fusion_beta")
                     for r in (0.05, 0.1, 0.2, 0.3)])
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    ax.fill_between(ratios, cf, full, color=C_PROP, alpha=0.10, zorder=1,
                    label="CSA-Rec gain")
    ax.plot(ratios, full, "-o", color=C_PROP, lw=1.8, ms=6, mec="white",
            mew=0.7, label="CSA-Rec", zorder=4)
    ax.plot(ratios, cf, "-s", color=C_CF, lw=1.6, ms=5, mec="white", mew=0.7,
            label="LightGCN (CF)", zorder=4)
    lab_off = [(6, 3, "left", "bottom"), (6, 6, "left", "bottom"),
               (5, 5, "left", "bottom"), (-4, 7, "right", "bottom")]
    for i, (r, f, c) in enumerate(zip(ratios, full, cf)):
        mult = f / c if c > 0 else np.nan
        dx, dy, ha_, va_ = lab_off[i]
        ax.annotate(f"{mult:.0f}\u00d7", (r, f), xytext=(dx, dy),
                    textcoords="offset points", ha=ha_, va=va_,
                    fontsize=6.6, color=C_GAIN, fontweight="bold")
    ax.set_xlabel("Cold-item ratio (fraction held out)")
    ax.set_ylabel("Cold-start Recall@20 (%)")
    ax.set_xticks(ratios)
    ax.set_ylim(0, 28)
    # beta drift on twin axis
    ax3 = ax.twinx()
    ax3.spines["top"].set_visible(False)
    ax3.plot(ratios, beta, ":^", color=C_GREY, lw=1.2, ms=4,
             label=r"learned $\beta$")
    ax3.set_ylabel(r"Learned $\beta$", color=C_GREY)
    ax3.tick_params(axis="y", labelcolor=C_GREY)
    ax3.set_ylim(0, 5)
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, loc="upper right", fontsize=6.3)
    ax.set_title("Robustness across cold-start severity", fontsize=8.3)
    save(fig, out, "fig5_coldratio")


# ============================ FIG 7: efficiency frontier ==================
def fig_efficiency(D, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    # x = catalog-specific trainable params (bridge); y = cold recall@20 (%)
    # bubble size ~ latency
    for ds, mk in zip(DATASETS, ("o", "D", "s")):
        xp = 11409  # constant catalog-independent bridge (SAA + Wp + beta)
        yp = gm(D, ds, "full", "cold/recall@20") * 100
        lat = gm(D, ds, "full", "eff/full_inference_ms")
        ax.scatter(xp, yp, s=60 + lat * 8, marker="*", color=C_PROP,
                   edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate(DS_LABEL[ds], (xp, yp), xytext=(6, -1),
                    textcoords="offset points", fontsize=6.3, color=C_PROP_D)
    # content-kNN: 0 trainable params (placed at x=1 on log axis)
    for ds in DATASETS:
        yk = gm(D, ds, "contentknn", "cold/recall@20") * 100
        ax.scatter(1, yk, s=42, marker="D", color=C_KNN, edgecolor="white",
                   linewidth=0.6, zorder=4)
    ax.scatter([], [], marker="D", color=C_KNN, label="Content-kNN (0 params)")
    ax.scatter([], [], marker="*", s=90, color=C_PROP, label="CSA-Rec")
    # LLM fine-tuning region
    ax.axvspan(1e6, 1e7, color=C_GREY, alpha=0.12, zorder=0)
    ax.text(3e6, 2.0, "LLM fine-tuning\n(LoRA): $10^6$\u2013$10^7$",
            fontsize=6.0, color="#5A6570", ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 3e7)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Catalog-specific trainable parameters")
    ax.set_ylabel("Cold-start Recall@20 (%)")
    ax.set_title("Accuracy at $10^4$ trainable params\n(marker size = latency)",
                 fontsize=8.0)
    ax.legend(loc="center left", fontsize=6.4, bbox_to_anchor=(0.02, 0.42))
    save(fig, out, "fig6_efficiency")


# ============================ FIG 8: structural ===========================
def fig_structural(D, out):
    fig = plt.figure(figsize=(7.16, 2.55))
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.28)
    ds = "Video_Games"
    # (a) backbone / adapter — grouped cold + warm horizontal bars
    ax = fig.add_subplot(gsp[0, 0])
    cfgs = [("LightGCN", "full"), ("MF backbone", "mf"), ("SAA = linear", "saalinear")]
    y = np.arange(len(cfgs))[::-1].astype(float)
    hh = 0.36
    cold = [gm(D, ds, m, "cold/recall@20") * 100 for _, m in cfgs]
    warm = [gm(D, ds, m, "warm/recall@20") * 100 for _, m in cfgs]
    bc = ax.barh(y + hh / 2 + 0.02, cold, hh, color=C_PROP, edgecolor="white",
                 label="Cold R@20")
    bw = ax.barh(y - hh / 2 - 0.02, warm, hh, color=C_CF, edgecolor="white",
                 label="Warm R@20")
    for yi, v in zip(y, cold):
        ax.text(v + 0.25, yi + hh / 2 + 0.02, f"{v:.1f}", va="center",
                fontsize=6.4, color=C_PROP_D)
    for yi, v in zip(y, warm):
        ax.text(v + 0.25, yi - hh / 2 - 0.02, f"{v:.1f}", va="center",
                fontsize=6.4, color=C_CF)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in cfgs])
    ax.set_xlabel("Recall@20 (%)")
    ax.set_title("(a) Backbone / adapter type", fontsize=8.3, loc="left")
    ax.set_xlim(0, max(cold) * 1.24)
    ax.legend(loc="lower right", fontsize=6.4)
    ax.margins(y=0.16)

    # (b) rank sweep with overfitting band + flat warm line
    ax2 = fig.add_subplot(gsp[0, 1])
    ranks = [8, 16, 32, 64]
    rv = [gm(D, ds, f"rank{r}", "cold/recall@20") * 100 for r in ranks]
    rw = [gm(D, ds, f"rank{r}", "warm/recall@20") * 100 for r in ranks]
    ax2.axvspan(20, 80, color=C_POP, alpha=0.08, zorder=0)
    ax2.set_ylim(min(rv) - 0.35, max(rv) + 0.7)
    ax2.text(40, max(rv) + 0.55, "capacity hurts\n(mild overfitting)", fontsize=6.0,
             color="#8A5A12", ha="center", va="top")
    ax2.plot(ranks, rv, "-o", color=C_PROP, lw=1.7, ms=6, mec="white", mew=0.7,
             label="Cold R@20", zorder=4)
    best = int(np.argmax(rv))
    ax2.scatter([ranks[best]], [rv[best]], s=120, facecolor="none",
                edgecolor=C_GAIN, linewidth=1.4, zorder=5)
    ax2.annotate("best (default $r=16$)", (16, rv[best]), xytext=(16, max(rv) + 0.42),
                 fontsize=6.2, color=C_GAIN, ha="center", va="bottom")
    ax2.set_xscale("log", base=2); ax2.set_xticks(ranks)
    ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax2.set_xlabel("SAA bottleneck rank $r$")
    ax2.set_ylabel("Cold-start Recall@20 (%)", color=C_PROP)
    ax2.tick_params(axis="y", labelcolor=C_PROP)
    ax2.set_title("(b) Adapter rank sweep", fontsize=8.3, loc="left")
    axw = ax2.twinx()
    axw.spines["top"].set_visible(False)
    axw.plot(ranks, rw, "--s", color=C_CF, lw=1.2, ms=4, label="Warm R@20")
    axw.set_ylabel("Warm Recall@20 (%)", color=C_CF)
    axw.tick_params(axis="y", labelcolor=C_CF)
    axw.set_ylim(min(rw) - 2, max(rw) + 2)
    h2, l2 = ax2.get_legend_handles_labels()
    hw, lw2 = axw.get_legend_handles_labels()
    ax2.legend(h2 + hw, l2 + lw2, loc="lower left", fontsize=6.4)
    save(fig, out, "fig7_structural")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="paper_assets/summary.csv")
    ap.add_argument("--out", default="../manuscript")
    args = ap.parse_args()
    D = load_summary(args.summary)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    fig_arch(D, args.out)
    fig_main(D, args.out)
    fig_slope(D, args.out)
    fig_ablation(D, args.out)
    fig_beta(D, args.out)
    fig_severity(D, args.out)
    fig_efficiency(D, args.out)
    fig_structural(D, args.out)
    print("[done] figures under", os.path.join(args.out, "figures"))


if __name__ == "__main__":
    main()
