"""
Regenerate every table and reported number in the manuscript from the raw
measurements in results/.

Usage, from the repository root:
    python scripts/analyze_results.py

Writes nothing. Prints the tables in the order they appear in the paper so that
each printed value can be checked against the manuscript directly.
"""

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY = os.path.join(ROOT, "results", "primary")
REPLICATION = os.path.join(ROOT, "results", "replication")

# The configuration the headline tables are drawn from: real dataset, largest
# model, single thread.
MAIN = "online_shoppers_f17_t1000_th1"

PATH_ORDER = ["lgb_sklearn", "xgb_sklearn", "cat_sklearn",
              "xgb_inplace", "xgb_dmatrix", "lgb_booster",
              "lgb_onnx", "xgb_onnx", "cat_onnx"]

LIBRARIES = {
    "lightgbm": ["lgb_sklearn", "lgb_booster", "lgb_onnx"],
    "xgboost": ["xgb_sklearn", "xgb_dmatrix", "xgb_inplace", "xgb_onnx"],
    "catboost": ["cat_sklearn", "cat_onnx"],
}

# Wrapper against leanest alternative, used for the overhead share.
OVERHEAD_PAIRS = [("lightgbm", "lgb_sklearn", "lgb_booster"),
                  ("xgboost", "xgb_sklearn", "xgb_inplace"),
                  ("catboost", "cat_sklearn", "cat_onnx")]


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load():
    primary = pd.read_csv(os.path.join(PRIMARY, "results.csv"))
    replication = pd.read_csv(os.path.join(REPLICATION, "results.csv"))
    for df in (primary, replication):
        df.drop_duplicates(subset=["config", "path", "batch"], keep="last",
                           inplace=True)
    return primary, replication


def environment():
    rule("Execution environment, Table 4")
    with open(os.path.join(PRIMARY, "environment.json")) as f:
        env = json.load(f)
    for k, v in env.items():
        print(f"  {k:14s} {v}")


def table1(primary):
    rule("Table 1. Per-row latency in microseconds, 1000 trees, single thread")
    t = (primary[primary.config == MAIN]
         .pivot(index="path", columns="batch", values="median_us_per_row")
         .reindex(PATH_ORDER))
    print(t[[1, 8, 64, 512, 4096, 16384]].round(2).to_string())
    return t


def table2(t1, replication):
    rule("Table 2. Penalty ratio and 95th percentile call latency at batch one")
    lo, hi = t1.columns.min(), t1.columns.max()
    rep = (replication[replication.config == MAIN]
           .pivot(index="path", columns="batch", values="median_us_per_row"))
    out = pd.DataFrame({
        "batch1_us": t1[lo],
        "batchmax_us": t1[hi],
        "penalty_primary": t1[lo] / t1[hi],
        "penalty_replication": rep[lo] / rep[hi],
    })
    print(out.sort_values("penalty_primary", ascending=False).round(2).to_string())
    return lo, hi


def table3(primary, replication):
    rule("Table 3. Batch-one latency against model size, all nine paths, both runs")
    rows = []
    for path in PATH_ORDER:
        rec = {"path": path}
        for label, df in [("primary", primary), ("replication", replication)]:
            g = (df[(df.sweep == "trees") & (df.batch == 1) & (df.path == path)]
                 .set_index("n_trees")["median_us_per_row"])
            rec[f"{label}_100"] = g[100]
            rec[f"{label}_1000"] = g[1000]
            rec[f"{label}_growth"] = g[1000] / g[100]
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("path")
    print(out.round(2).to_string())

    wrappers = [p for p in PATH_ORDER if p.endswith("sklearn")]
    exported = [p for p in PATH_ORDER if p.endswith("onnx")]
    for name, group in [("wrapper paths", wrappers), ("exported paths", exported)]:
        g = out.loc[group, ["primary_growth", "replication_growth"]].values
        print(f"\n  {name}: growth across both runs "
              f"{g.min():.1f}x to {g.max():.1f}x")


def overhead(primary, replication, lo):
    rule("Overhead share at batch one, 1000 trees")
    for label, df in [("primary", primary), ("replication", replication)]:
        m = (df[(df.config == MAIN) & (df.batch == lo)]
             .set_index("path")["median_us_per_row"])
        for lib, wrapper, leaner in OVERHEAD_PAIRS:
            share = 100 * (m[wrapper] - m[leaner]) / m[wrapper]
            print(f"  {label:12s} {lib:9s} {wrapper} {m[wrapper]:7.1f} us, "
                  f"{leaner} {m[leaner]:7.1f} us, overhead share {share:5.1f}%")


def robustness(primary, replication):
    rule("Robust winner test")
    for label, df in [("primary", primary), ("replication", replication)]:
        rob = compute_robustness(df)
        b1 = rob[rob.batch == 1]
        print(f"  {label:12s} {100*rob.robust.mean():4.0f}% of all comparisons "
              f"have a robust winner")
        print(f"               batch one: {int(b1.robust.sum())} of {len(b1)} "
              f"robust, margins {b1.margin_pct.min():.0f}% to "
              f"{b1.margin_pct.max():.0f}%")
        onnx_wins = b1[b1.winner.str.endswith("onnx")]
        print(f"               batch one won by an exported runtime in "
              f"{len(onnx_wins)} of {len(b1)} comparisons")

    rule("Crossover: first batch size at which a library path overtakes ONNX")
    for label, df in [("primary", primary), ("replication", replication)]:
        tw = df[df.sweep == "trees"]
        for n_trees in sorted(tw.n_trees.unique()):
            line = []
            for lib, paths in LIBRARIES.items():
                m = (tw[tw.n_trees == n_trees]
                     .pivot(index="path", columns="batch",
                            values="median_us_per_row"))
                present = [p for p in paths if p in m.index]
                cross = "none"
                for b in sorted(m.columns):
                    if not m.loc[present, b].idxmin().endswith("onnx"):
                        cross = str(b)
                        break
                line.append(f"{lib}={cross:>5s}")
            print(f"  {label:12s} {n_trees:5d} trees   " + "   ".join(line))


def compute_robustness(df):
    """A winner counts as robust when its margin over the runner-up exceeds
    three times the interquartile range of its own measurement."""
    rows = []
    tw = df[df.sweep == "trees"]
    for n_trees in sorted(tw.n_trees.unique()):
        for lib, paths in LIBRARIES.items():
            for batch in sorted(tw.batch.unique()):
                g = tw[(tw.n_trees == n_trees) & (tw.batch == batch) &
                       (tw.path.isin(paths))].set_index("path")
                if len(g) < 2:
                    continue
                srt = g["median_us_per_row"].sort_values()
                best = srt.index[0]
                margin = 100 * (srt.iloc[1] - srt.iloc[0]) / srt.iloc[0]
                noise = (100 * g.loc[best, "iqr_us_per_row"] /
                         g.loc[best, "median_us_per_row"])
                rows.append({"n_trees": n_trees, "library": lib, "batch": batch,
                             "winner": best, "margin_pct": margin,
                             "iqr_pct": noise, "robust": margin > 3 * noise})
    return pd.DataFrame(rows)


def measurement_quality(primary):
    rule("Measurement quality")
    correct = pd.read_csv(os.path.join(PRIMARY, "correctness.csv")).drop_duplicates()
    print(f"  correctness: {len(correct)} comparisons, "
          f"{int((~correct.agrees).sum())} disagreements, "
          f"largest deviation {correct.max_abs_diff.max():.2e}")

    rel_iqr = 100 * primary.iqr_us_per_row / primary.median_us_per_row
    print(f"  median relative interquartile range: {rel_iqr.median():.1f}%")

    drift = pd.read_csv(os.path.join(PRIMARY, "noise_reference.csv"))
    slope = np.polyfit(drift.t_min, drift.ref_us, 1)[0]
    print(f"  host drift: {len(drift)} reference measurements over "
          f"{drift.t_min.max():.1f} minutes")
    print(f"              median {drift.ref_us.median():.2f} us/row, "
          f"range {drift.ref_us.min():.2f} to {drift.ref_us.max():.2f}")
    print(f"              fitted slope {slope:.3f} us/min, "
          f"{100*slope*drift.t_min.max()/drift.ref_us.median():.1f}% "
          f"over the session")

    abl = pd.read_csv(os.path.join(PRIMARY, "zipmap_ablation.csv"))
    print("\n  CatBoost exported-model ablation, ratio with node over without:")
    for _, r in abl.iterrows():
        print(f"    batch {int(r.batch):6d}   {r.ratio:.3f}")


def secondary(primary, replication):
    rule("Feature count, secondary comparison")
    for label, df in [("primary", primary), ("replication", replication)]:
        f = (df[(df.sweep == "features") & (df.batch == 1)]
             .pivot(index="path", columns="n_features", values="median_us_per_row"))
        ratios = {p: f.loc[p, 100] / f.loc[p, 20]
                  for p in ["lgb_sklearn", "xgb_sklearn", "cat_sklearn"]}
        print(f"  {label:12s} " +
              "   ".join(f"{k}={v:.2f}x" for k, v in ratios.items()))
    print("\n  Only the CatBoost effect replicates. Reported as secondary in the "
          "manuscript.")

    rule("Thread count at batch one, 100 trees")
    th = primary[(primary.dataset == "online_shoppers") &
                 (primary.n_trees == 100) & (primary.batch == 1)]
    print(th.pivot_table(index="path", columns="threads",
                         values="median_us_per_row").round(1).to_string())


def main():
    primary, replication = load()
    print(f"loaded {len(primary)} primary and {len(replication)} replication "
          f"measurements")
    environment()
    t1 = table1(primary)
    lo, hi = table2(t1, replication)
    table3(primary, replication)
    overhead(primary, replication, lo)
    robustness(primary, replication)
    measurement_quality(primary)
    secondary(primary, replication)
    print("\ndone")


if __name__ == "__main__":
    main()
