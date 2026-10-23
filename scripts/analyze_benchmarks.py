"""Figures and tables for Parts 1 and 2, plus the operating points used by Part 3.

Selection protocol: every choice (which configurations form the trade-off curve, which setting is
used for recall 0.90 / 0.95) is made on the tuning half of the queries. The numbers that are
plotted and tabulated are the test-half measurements of those chosen configurations.

Usage: python scripts/analyze_benchmarks.py [--quick]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from annbench import config
from annbench.indexes import SEARCH_PARAM
from annbench.report_utils import (FAMILIES, INK_2, STYLE, paired, pareto_mask, select_operating_point,
                                   setup_matplotlib, write_table)
from annbench.results import save_numbers

DATASETS = {"sift": "SIFT", "glove": "GloVe-100"}
TARGETS = (0.90, 0.95)


def short(fam: str) -> str:
    return fam.lower().replace("-", "")


def config_label(row) -> str:
    return f"{row['build_params']}, {SEARCH_PARAM[row['index']]}={int(row['search_value'])}"


def part1_table(P, numbers):
    rows = []
    for ds, title in DATASETS.items():
        f = P["work"] / f"part1_{ds}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        ex = df[df["experiment"] == "part1_exact"]
        single = ex[ex["search_value"] == "single"].iloc[0]
        b1 = ex[(ex["search_value"] == "batch") & (ex["search_threads"] == 1)].iloc[0]
        bn = ex[(ex["search_value"] == "batch") & (ex["search_threads"] > 1)].iloc[0]
        ver = df[df["experiment"] == "part1_verify"]
        if len(ver):
            v = json.loads(ver.iloc[0]["notes"])
            agree = (f"{100 * v['frac_identical_top10_set']:.2f}% identical; "
                     f"{v['n_queries_differing'] - v['n_differing_only_by_ties']} non-tie differences; "
                     f"tie-aware recall {v['recall_ours_vs_file_mean']:.4f}")
        else:
            agree = "n/a (subset)"
        rows.append({
            "Dataset": f"{title} (n={int(single['n_base']):,})",
            "1 query/call, 1 thread: QPS (ms/query)": f"{single['qps_median']:.1f} ({1e3 / single['qps_median']:.1f})",
            "batch, 1 thread: QPS": f"{b1['qps_median']:,.0f}",
            f"batch, {int(bn['search_threads'])} threads: QPS": f"{bn['qps_median']:,.0f}",
            "batch/single (1 thread)": f"{b1['qps_median'] / single['qps_median']:.1f}x",
            "agreement with stored top-10": agree,
        })
        n_base, dim = int(single["n_base"]), int(single["dim"])
        ms = 1e3 / single["qps_median"]
        numbers[f"p1_{ds}_single_gbps"] = n_base * dim * 4 * single["qps_median"] / 1e9   # bytes streamed per second
        numbers[f"p1_{ds}_ms_150k"] = ms * 150_000 / n_base                               # brute force is linear in n
        numbers[f"p1_{ds}_ms_10m"] = ms * 10_000_000 / n_base
        numbers[f"p1_{ds}_gib_10m"] = 10_000_000 * dim * 4 / 2**30
        numbers[f"p1_{ds}_batch_over_single"] = b1["qps_median"] / single["qps_median"]
        numbers[f"p1_{ds}_batchmax_over_single"] = bn["qps_median"] / single["qps_median"]
    if rows:
        write_table(pd.DataFrame(rows), P["tables"] / "part1_exact")


def plot_dataset(ax, p, exact_qps, title, lo=0.5):
    for fam in FAMILIES:
        fam_df = p[p["index"] == fam]
        if fam_df.empty:
            continue
        st = STYLE[fam]
        front = fam_df[pareto_mask(fam_df["recall_at_10_tune"].to_numpy(), fam_df["qps_median_tune"].to_numpy())]
        front = front.sort_values("recall_at_10_tune")
        others = fam_df.drop(front.index)
        ax.scatter(others["recall_at_10"], others["qps_median"], s=9, color=st["color"], alpha=0.22,
                   linewidths=0, zorder=2)
        ax.plot(front["recall_at_10"], front["qps_median"], color=st["color"], marker=st["marker"],
                markeredgecolor="white", markeredgewidth=0.6, label=fam, zorder=3)
    ax.axhline(exact_qps, color=STYLE["Exact"]["color"], ls="--", lw=1.0, zorder=1)
    ax.plot([1.0], [exact_qps], marker="*", ms=10, color=STYLE["Exact"]["color"], ls="none",
            label="Exact (flat)", zorder=4)
    for t in TARGETS:
        ax.axvline(t, color=INK_2, ls=":", lw=0.8, zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(lo, 1.005)
    ax.set_xlabel("recall@10 (test half)")
    ax.set_ylabel("queries per second (1 thread, 1 query/call)")
    ax.set_title(title)
    ax.legend(loc="lower left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    P = config.paths(args.quick)
    setup_matplotlib()
    numbers = {}
    part1_table(P, numbers)

    qps_rows, cost_rows, ceil_rows = [], [], []
    fig_all, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax_i, (ds, title) in enumerate(DATASETS.items()):
        f2, f1 = P["work"] / f"part2_{ds}.csv", P["work"] / f"part1_{ds}.csv"
        if not (f2.exists() and f1.exists()):
            print(f"[analyze] missing results for {ds}, skipping")
            continue
        df, ex = pd.read_csv(f2), pd.read_csv(f1)
        p = paired(df)
        ex_single = ex[(ex["experiment"] == "part1_exact") & (ex["search_value"] == "single")].iloc[0]
        exact_qps = float(ex_single["qps_median"])
        title_n = f"{title} (n={int(df['n_base'].iloc[0]):,})"

        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        plot_dataset(ax, p, exact_qps, title_n)
        fig.tight_layout()
        fig.savefig(P["figures"] / f"part2_{ds}_recall_vs_qps.png")
        plt.close(fig)
        plot_dataset(axes[ax_i], p, exact_qps, title_n)

        selected = {}
        for fam in FAMILIES:
            fam_df = p[p["index"] == fam]
            if fam_df.empty:
                continue
            tune = fam_df.rename(columns={"recall_at_10": "recall_test", "qps_median": "qps_test"}) \
                .rename(columns={"recall_at_10_tune": "recall_at_10", "qps_median_tune": "qps_median"})
            selected[fam] = {}
            best = tune.loc[tune["recall_at_10"].idxmax()]
            selected[fam]["best"] = {"build_params": best["build_params"], "search_value": int(best["search_value"]),
                                     "recall_tune": float(best["recall_at_10"]), "recall_test": float(best["recall_test"])}
            for target in TARGETS:
                op = select_operating_point(tune, target)
                key = f"R{target:.2f}"
                if op is None:
                    selected[fam][key] = None
                    qps_rows.append({"Dataset": title, "Index": fam, "Target": f"{target:.2f}",
                                     "Chosen on tuning half": f"not reached (max tuning recall {best['recall_at_10']:.3f}"
                                                              f" at {config_label(best)})",
                                     "Recall (tune)": "-", "Recall (test)": "-", "QPS (test)": "-", "Speed-up vs exact": "-"})
                    numbers[f"p2_{ds}_{short(fam)}_qps{int(target * 100)}"] = None
                    continue
                selected[fam][key] = {"build_params": op["build_params"], "search_value": int(op["search_value"]),
                                      "recall_tune": float(op["recall_at_10"]), "recall_test": float(op["recall_test"]),
                                      "qps_test": float(op["qps_test"])}
                qps_rows.append({"Dataset": title, "Index": fam, "Target": f"{target:.2f}",
                                 "Chosen on tuning half": config_label(op),
                                 "Recall (tune)": f"{op['recall_at_10']:.3f}", "Recall (test)": f"{op['recall_test']:.3f}",
                                 "QPS (test)": f"{op['qps_test']:,.0f}",
                                 "Speed-up vs exact": f"{op['qps_test'] / exact_qps:,.0f}x"})
                tag = f"p2_{ds}_{short(fam)}_{int(target * 100)}"
                numbers[f"{tag}_qps"] = float(op["qps_test"])
                numbers[f"{tag}_recall"] = float(op["recall_test"])
                numbers[f"{tag}_speedup"] = float(op["qps_test"] / exact_qps)
                numbers[f"{tag}_config"] = config_label(op)
                numbers[f"{tag}_dist_comps"] = float(op["dist_comps_per_query"])

            # build cost at the recall-0.95 operating point (or the highest-recall setting if not reached)
            op = selected[fam]["R0.95"] or selected[fam]["best"]
            row = fam_df[(fam_df["build_params"] == op["build_params"])].iloc[0]
            reached = selected[fam]["R0.95"] is not None
            cost_rows.append({
                "Dataset": title, "Index": fam,
                "Build setting": op["build_params"] + ("" if reached else f" (max recall {op['recall_test']:.3f}; 0.95 not reached)"),
                "Build time (s)": f"{row['build_time_s']:.1f}",
                "of which training (s)": f"{row['train_time_s']:.1f}",
                "Index size (MiB)": f"{row['index_bytes'] / 2**20:,.1f}",
                "Bytes/vector": f"{row['bytes_per_vector']:.1f}",
            })
            for bp, g in fam_df.groupby("build_params"):
                key = "".join(ch for ch in bp.split(",nbits")[0] if ch.isalnum())
                numbers[f"p2_{ds}_{short(fam)}_{key}_bpv"] = float(g["bytes_per_vector"].iloc[0])
                numbers[f"p2_{ds}_{short(fam)}_{key}_build_s"] = float(g["build_time_s"].iloc[0])
            tag = f"p2_{ds}_{short(fam)}"
            numbers[f"{tag}_gib_10m"] = float(row["bytes_per_vector"]) * 10_000_000 / 2**30
            numbers[f"{tag}_build_s"] = float(row["build_time_s"])
            numbers[f"{tag}_size_mib"] = float(row["index_bytes"] / 2**20)
            numbers[f"{tag}_bpv"] = float(row["bytes_per_vector"])
            numbers[f"{tag}_extra_ram_mib"] = float(row["extra_ram_bytes"] / 2**20) if not pd.isna(row["extra_ram_bytes"]) else 0.0

            if fam == "IVF-PQ":
                for bp, g in fam_df.groupby("build_params"):
                    gb = g.loc[g["recall_at_10_tune"].idxmax()]
                    m = bp.split("m=")[1].split(",")[0]
                    ceil_rows.append({"Dataset": title, "Build setting": bp,
                                      "Bytes/vector": f"{gb['bytes_per_vector']:.1f}",
                                      "Max recall (test)": f"{gb['recall_at_10']:.3f}",
                                      "at nprobe": int(gb["search_value"])})
                    numbers[f"p2_{ds}_ivfpq_m{m}_maxrecall"] = float(gb["recall_at_10"])

        ex_size = float(ex_single["index_bytes"])
        cost_rows.append({"Dataset": title, "Index": "Exact", "Build setting": "IndexFlatL2",
                          "Build time (s)": f"{ex_single['build_time_s']:.1f}", "of which training (s)": "0.0",
                          "Index size (MiB)": f"{ex_size / 2**20:,.1f}",
                          "Bytes/vector": f"{ex_size / ex_single['n_base']:.1f}"})
        qps_rows.append({"Dataset": title, "Index": "Exact", "Target": "1.00", "Chosen on tuning half": "IndexFlatL2",
                         "Recall (tune)": "1.000", "Recall (test)": f"{ex_single['recall_at_10']:.3f}",
                         "QPS (test)": f"{exact_qps:,.1f}", "Speed-up vs exact": "1x"})
        numbers[f"p2_{ds}_exact_size_mib"] = ex_size / 2**20
        (P["work"] / f"selected_{ds}.json").write_text(json.dumps(selected, indent=1))

    fig_all.tight_layout()
    fig_all.savefig(P["figures"] / "part2_recall_vs_qps_both.png")
    plt.close(fig_all)
    write_table(pd.DataFrame(qps_rows), P["tables"] / "part2_qps_at_recall")
    write_table(pd.DataFrame(cost_rows), P["tables"] / "part2_build_cost")
    write_table(pd.DataFrame(ceil_rows), P["tables"] / "part2_ivfpq_ceiling")
    save_numbers(P["tables"] / "report_numbers.json", numbers)
    print("[analyze] wrote figures and tables to", P["results"])


if __name__ == "__main__":
    main()
