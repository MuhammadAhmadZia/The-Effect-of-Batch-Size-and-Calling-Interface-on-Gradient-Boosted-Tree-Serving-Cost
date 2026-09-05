"""
Figure generation for the batch-of-one serving interface study.

Produces eight PNG figures at 300 dpi into ./figures.
Figures 1 to 3 are methodology diagrams drawn with matplotlib primitives.
Figures 4 to 8 are drawn from the measurement data.

Inputs expected in the working directory:
    run3/results.csv                  primary run
    run3/crossover_robustness.csv     robust winner test
    run3/noise_reference.csv          host drift log
    run1/results.csv                  replication run

Run:  python make_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

# Grey-safe palette. Distinguishable in colour and when printed in greyscale.
CLR = {
    "lgb_sklearn": "#1f3d7a", "lgb_booster": "#4a7ab8", "lgb_onnx": "#9dc3e6",
    "xgb_sklearn": "#7a1f1f", "xgb_dmatrix": "#b84a4a", "xgb_inplace": "#d98b8b",
    "xgb_onnx": "#f0c0c0",
    "cat_sklearn": "#1f5c3d", "cat_onnx": "#7ab894",
}
MRK = {
    "lgb_sklearn": "o", "lgb_booster": "s", "lgb_onnx": "^",
    "xgb_sklearn": "o", "xgb_dmatrix": "s", "xgb_inplace": "^", "xgb_onnx": "D",
    "cat_sklearn": "o", "cat_onnx": "^",
}
BOX_FILL, BOX_EDGE = "#eef2f7", "#4a6fa5"
ALT_FILL, ALT_EDGE = "#f7f0ee", "#a5624a"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def box(ax, x, y, w, h, text, fill=BOX_FILL, edge=BOX_EDGE, fs=8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, facecolor=fill, edgecolor=edge))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, linespacing=1.45)


def arrow(ax, p1, p2, style="-|>", ls="-", color="#44506b"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, linestyle=ls,
                                 mutation_scale=11, linewidth=1.1,
                                 color=color, shrinkA=1, shrinkB=1))


def blank_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------- Figure 1
def fig1_architecture():
    """Abstract view of the whole experiment, four stages left to right."""
    fig, ax = blank_axes((7.6, 4.0))

    # four stages, wide gaps so the arrows between them are unambiguous
    xs = [0.015, 0.265, 0.515, 0.765]
    w, y, h = 0.185, 0.33, 0.54
    texts = [
        "Stage 1\nData and Training\n\nOnline Shoppers\n12,330 sessions\n17 features\n\nThree libraries at\n100, 500 and\n1000 trees",
        "Stage 2\nServing Path\nConstruction\n\nNine interfaces\nover the same\ntrained models\n\nwrapper, native\nand exported\nruntime calls",
        "Stage 3\nMeasurement\n\n13 batch sizes\nfrom 1 to 16,384\n\n11 timed repeats\nper measurement\n\nshuffled\nconfiguration order",
        "Stage 4\nAnalysis\n\npenalty ratio\n\nrobust winner test\nagainst measurement\nnoise\n\nhost drift check",
    ]
    for x, t in zip(xs, texts):
        box(ax, x, y, w, h, t, fs=6.8)

    for x in xs[:-1]:
        arrow(ax, (x + w + 0.008, y + h / 2), (x + w + 0.055, y + h / 2))

    # controls, each tied to the stage it guards
    box(ax, 0.265, 0.035, 0.435, 0.22,
        "Correctness gate\nevery interface checked against its own\nlibrary's call before any timing is recorded",
        fill=ALT_FILL, edge=ALT_EDGE, fs=7.0)
    box(ax, 0.735, 0.035, 0.25, 0.22,
        "Persistent reference model\nre-measured throughout\nthe session",
        fill=ALT_FILL, edge=ALT_EDGE, fs=7.0)

    arrow(ax, (0.4825, 0.325), (0.4825, 0.262), ls=(0, (3, 2)), color="#a5624a")
    arrow(ax, (0.8575, 0.325), (0.8575, 0.262), ls=(0, (3, 2)), color="#a5624a")

    ax.text(0.5, 0.935, "Same trained models behind every interface, single thread throughout",
            ha="center", fontsize=8, style="italic", color="#333333")
    save(fig, "fig1_experiment_architecture.png")


# ----------------------------------------------------------------- Figure 2
def fig2_serving_paths():
    """One trained model per library fans out into several serving interfaces."""
    fig, ax = blank_axes((7.4, 4.2))

    libs = [
        ("LightGBM", 0.72, ["sklearn wrapper", "native Booster", "ONNX Runtime"], BOX_FILL, BOX_EDGE),
        ("XGBoost", 0.42, ["sklearn wrapper", "DMatrix predict", "inplace predict", "ONNX Runtime"],
         "#f7eeee", "#a5624a"),
        ("CatBoost", 0.12, ["sklearn wrapper", "ONNX Runtime"], "#eef7f1", "#3d7a5c"),
    ]

    for name, y, paths, fill, edge in libs:
        box(ax, 0.02, y, 0.19, 0.19, f"{name}\ntrained once", fill=fill, edge=edge, fs=8)
        n = len(paths)
        for i, p in enumerate(paths):
            py = y + 0.19 - (i + 1) * (0.19 / n) + 0.004
            box(ax, 0.34, py, 0.30, 0.19 / n - 0.012, p, fill="white", edge=edge, fs=7.4)
            arrow(ax, (0.215, y + 0.095), (0.335, py + 0.19 / (2 * n)))

    box(ax, 0.71, 0.12, 0.27, 0.79,
        "Identical model\nbehind every path\n\nVerified before timing:\n"
        "36 comparisons\nzero disagreements\nmaximum deviation\n4.8 x 10\u207b\u2077",
        fill=ALT_FILL, edge=ALT_EDGE, fs=7.6)
    for y in (0.215, 0.515, 0.815):
        arrow(ax, (0.645, y), (0.705, y))
    save(fig, "fig2_serving_paths.png")


# ----------------------------------------------------------------- Figure 3
def fig3_timing_protocol():
    """What one measurement consists of, at two levels of detail."""
    fig, ax = blank_axes((7.6, 4.0))

    # top row, the three phases
    box(ax, 0.015, 0.78, 0.29, 0.19,
        "Calibration\ntime a few calls, then pick the call\ncount that fills a 0.15 s budget", fs=7.2)
    box(ax, 0.355, 0.78, 0.24, 0.19,
        "Warm-up\n20 calls executed\nand discarded", fs=7.2)
    box(ax, 0.645, 0.78, 0.34, 0.19,
        "Timed repeats\n11 passes, every call\ntimed on its own", fs=7.2)
    arrow(ax, (0.312, 0.875), (0.348, 0.875))
    arrow(ax, (0.602, 0.875), (0.638, 0.875))

    # level 1: the session, one warm-up block then eleven repeat blocks
    ax.text(0.015, 0.685, "One measurement", fontsize=7.8, fontweight="bold")
    x0, bw, by, bh = 0.015, 0.0625, 0.55, 0.10
    ax.add_patch(FancyBboxPatch((x0, by), 0.115, bh,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor="#e2e2e2", edgecolor="#888888", linewidth=0.9))
    ax.text(x0 + 0.0575, by + bh / 2, "warm-up\ndiscarded", ha="center", va="center", fontsize=6.2)
    for i in range(11):
        x = 0.145 + i * (bw + 0.012)
        ax.add_patch(FancyBboxPatch((x, by), bw, bh,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor="#dce6f2", edgecolor="#4a6fa5", linewidth=0.9))
        ax.text(x + bw / 2, by + bh / 2, f"repeat\n{i+1}", ha="center", va="center", fontsize=6.2)

    # level 2: one repeat opened up into its individual calls
    zx, zy, zw = 0.145 + 3 * (bw + 0.012), 0.28, bw
    arrow(ax, (zx, by), (0.13, zy + 0.115), ls=(0, (2, 2)), style="-", color="#a5624a")
    arrow(ax, (zx + zw, by), (0.90, zy + 0.115), ls=(0, (2, 2)), style="-", color="#a5624a")

    ax.add_patch(FancyBboxPatch((0.13, zy), 0.77, 0.115,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor="white", edgecolor="#a5624a", linewidth=1.0))
    n_calls = 22
    for i in range(n_calls):
        cx = 0.152 + i * (0.72 / n_calls)
        ax.add_patch(plt.Rectangle((cx, zy + 0.028), 0.0175, 0.058,
                                   facecolor="#7ab894", edgecolor="#3d7a5c", linewidth=0.5))
    ax.text(0.895, zy + 0.058, "...", ha="right", va="center", fontsize=9)
    ax.text(0.515, zy - 0.045,
            "inside one repeat: many calls at the same batch size, each timed separately",
            ha="center", fontsize=7.0, style="italic")
    ax.text(0.015, zy + 0.152, "One repeat opened up", fontsize=7.4, fontweight="bold")

    # outputs
    box(ax, 0.015, 0.02, 0.47, 0.15,
        "Reported per measurement\nmedian and interquartile range of per-row latency",
        fill=ALT_FILL, edge=ALT_EDGE, fs=7.0)
    box(ax, 0.515, 0.02, 0.47, 0.15,
        "Also recorded\n50th, 95th and 99th percentile of single call latency",
        fill=ALT_FILL, edge=ALT_EDGE, fs=7.0)
    save(fig, "fig3_timing_protocol.png")


# ----------------------------------------------------------------- data
def load():
    r3 = pd.read_csv(os.path.join(ROOT, "results", "primary", "results.csv"))
    r1 = pd.read_csv(os.path.join(ROOT, "results", "replication", "results.csv"))
    rob = pd.read_csv(os.path.join(ROOT, "results", "primary", "crossover_robustness.csv"))
    drift = pd.read_csv(os.path.join(ROOT, "results", "primary", "noise_reference.csv"))
    return r1, r3, rob, drift


MAIN = "online_shoppers_f17_t1000_th1"


def fig4_latency(r3):
    m = r3[r3.config == MAIN].pivot(index="path", columns="batch",
                                    values="median_us_per_row")
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for p in m.index:
        ax.plot(m.columns, m.loc[p], marker=MRK[p], ms=3.6, linewidth=1.3,
                color=CLR[p], label=p.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size, rows per call")
    ax.set_ylabel("Latency per row, microseconds")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    save(fig, "fig4_latency_vs_batch.png")


def fig5_penalty(r1, r3):
    out = {}
    for name, r in [("replication", r1), ("primary", r3)]:
        m = r[r.config == MAIN].pivot(index="path", columns="batch",
                                      values="median_us_per_row")
        out[name] = m[m.columns.min()] / m[m.columns.max()]
    d = pd.DataFrame(out).sort_values("primary")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.barh(y - 0.19, d["replication"], height=0.36, color="#b0bdd1",
            edgecolor="#44506b", linewidth=0.6, label="replication")
    ax.barh(y + 0.19, d["primary"], height=0.36, color="#4a6fa5",
            edgecolor="#2b3a55", linewidth=0.6, label="primary")
    ax.set_yticks(y)
    ax.set_yticklabels([i.replace("_", " ") for i in d.index])
    ax.set_xscale("log")
    ax.set_xlabel("Penalty ratio, batch 1 relative to batch 16,384")
    ax.axvline(1, color="#888888", linewidth=0.8, linestyle=":")
    ax.grid(True, axis="x", alpha=0.22, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig5_penalty_ratio.png")


def fig6_trees(r1, r3):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (name, r) in zip(axes, [("Replication run", r1), ("Primary run", r3)]):
        t = r[(r.sweep == "trees") & (r.batch == 1)]
        for p in ["lgb_sklearn", "xgb_sklearn", "cat_sklearn",
                  "lgb_onnx", "xgb_onnx", "cat_onnx"]:
            g = t[t.path == p].sort_values("n_trees")
            ls = "-" if p.endswith("sklearn") else "--"
            ax.plot(g.n_trees, g.median_us_per_row, marker=MRK[p], ms=3.6,
                    linewidth=1.3, linestyle=ls, color=CLR[p],
                    label=p.replace("_", " "))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([100, 500, 1000])
        ax.set_xticklabels(["100", "500", "1000"])
        ax.set_xlabel("Trees in the model")
        ax.set_title(name, fontsize=9)
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    axes[0].set_ylabel("Latency per row at batch 1, microseconds")
    axes[1].legend(ncol=2, frameon=False, loc="lower right")
    save(fig, "fig6_batch1_cost_vs_trees.png")


def fig7_regime(rob):
    piv = rob[rob.robust].pivot_table(index=["library", "n_trees"], columns="batch",
                                      values="winner", aggfunc="first")
    piv = piv.reindex(sorted(piv.columns), axis=1)
    codes = np.full(piv.shape, np.nan)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if isinstance(v, str):
                codes[i, j] = 0 if v.endswith("onnx") else 1
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    cmap = matplotlib.colors.ListedColormap(["#4a6fa5", "#c08a3e"])
    cmap.set_bad("#f0f0f0")
    ax.imshow(np.ma.masked_invalid(codes), cmap=cmap, aspect="auto",
              vmin=-0.5, vmax=1.5)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{a}, {b} trees" for a, b in piv.index])
    ax.set_xlabel("Batch size, rows per call")
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in
               ["#4a6fa5", "#c08a3e", "#f0f0f0"]]
    ax.legend(handles, ["ONNX Runtime cheapest", "library interface cheapest",
                        "no robust winner"],
              ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    save(fig, "fig7_regime_map.png")


def fig8_drift(drift):
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.plot(drift.t_min, drift.ref_us, marker="o", ms=3.2, linewidth=1.1,
            color="#4a6fa5")
    med = drift.ref_us.median()
    ax.axhline(med, color="#a5624a", linewidth=0.9, linestyle="--")
    ax.text(drift.t_min.max(), med, "  median", va="center", fontsize=7.4,
            color="#a5624a")
    ax.set_xlabel("Minutes into the session")
    ax.set_ylabel("Reference latency\nper row, microseconds")
    ax.grid(True, alpha=0.22, linewidth=0.5)
    save(fig, "fig8_host_drift.png")


if __name__ == "__main__":
    fig1_architecture()
    fig2_serving_paths()
    fig3_timing_protocol()
    r1, r3, rob, drift = load()
    fig4_latency(r3)
    fig5_penalty(r1, r3)
    fig6_trees(r1, r3)
    fig7_regime(rob)
    fig8_drift(drift)
    print("\nall figures written to", OUT)
