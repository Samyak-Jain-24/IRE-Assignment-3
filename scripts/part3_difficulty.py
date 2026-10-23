"""Part 3: query difficulty on GloVe-100, measured by relative contrast (see hypothesis.md).

RC(q) = mean distance from q to all base vectors / distance from q to its 10th nearest neighbour.
Bin edges are the RC quartiles of the tuning half; recall is compared per bin on the test half,
using the per-query results recorded by part2_benchmark.py at the operating points selected by
analyze_benchmarks.py.

Usage: python scripts/part3_difficulty.py [--quick]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from annbench import config
from annbench.data import load_dataset, query_split
from annbench.indexes import SEARCH_PARAM
from annbench.report_utils import FAMILIES, ORDINAL_BLUE, STYLE, setup_matplotlib, write_table
from annbench.results import RowWriter, save_numbers
from annbench.timing import limit_threads

BIN_NAMES = ["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"]


def mean_distances(queries: np.ndarray, base: np.ndarray, q_chunk: int = 1000, b_chunk: int = 50_000) -> np.ndarray:
    """Exact mean Euclidean distance from every query to every base vector (chunked BLAS)."""
    qn = (queries.astype(np.float32) ** 2).sum(1)
    bn = (base ** 2).sum(1)
    total = np.zeros(len(queries), dtype=np.float64)
    with limit_threads(os.cpu_count() or 1):
        for s in range(0, len(queries), q_chunk):
            q = queries[s:s + q_chunk]
            acc = np.zeros(len(q), dtype=np.float64)
            for b in range(0, len(base), b_chunk):
                x = base[b:b + b_chunk]
                d2 = qn[s:s + q_chunk, None] + bn[None, b:b + b_chunk] - 2.0 * (q @ x.T)
                np.maximum(d2, 0, out=d2)
                np.sqrt(d2, out=d2)
                acc += d2.sum(1, dtype=np.float64)
            total[s:s + q_chunk] = acc
    return total / len(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg, P = config.settings(args.quick), config.paths(args.quick)
    k = config.K
    setup_matplotlib()

    ds = load_dataset("glove", P["data"], cfg["n_base"])
    tune_idx, test_idx = query_split(cfg["n_tune"], cfg["n_test"])
    gt = np.load(P["work"] / "gt_glove.npz")
    used = np.concatenate([tune_idx, test_idx])
    mean_d = np.zeros(len(ds.queries))
    mean_d[used] = mean_distances(ds.queries[used], ds.base)
    d_k = np.sqrt(gt["dists"][:, k - 1])
    rc = np.where(d_k > 0, mean_d / np.maximum(d_k, 1e-12), np.inf)

    edges = np.quantile(rc[tune_idx], [0.25, 0.5, 0.75])
    bin_tune = np.digitize(rc[tune_idx], edges)
    bin_test = np.digitize(rc[test_idx], edges)
    lo_hi = [(rc[tune_idx].min(), edges[0]), (edges[0], edges[1]), (edges[1], edges[2]), (edges[2], rc[tune_idx].max())]
    print(f"[part3] RC tuning quartile edges: {np.round(edges, 4)}; test bin sizes {np.bincount(bin_test, minlength=4)}")

    selected = json.loads((P["work"] / "selected_glove.json").read_text())
    pq = dict(np.load(P["work"] / "part2_glove_perquery.npz"))
    out = RowWriter(P["work"] / "part3_glove.csv", experiment="part3_difficulty", dataset="glove", mode=cfg["mode"],
                    n_base=len(ds.base), dim=ds.dim, k=k, split="test", search_threads=1)

    numbers = {"p3_rc_edge1": edges[0], "p3_rc_edge2": edges[1], "p3_rc_edge3": edges[2],
               "p3_rc_min": float(rc[used].min()), "p3_rc_median": float(np.median(rc[used])),
               "p3_rc_max": float(rc[used].max())}
    bin_rows, effort_rows = [], []
    per_bin = {}
    for fam in FAMILIES:
        op = selected.get(fam, {}).get("R0.95") or selected.get(fam, {}).get("best")
        if op is None:
            continue
        key = f"{fam}|{op['build_params']}|{op['search_value']}"
        rec, lat = pq[f"{key}|test_recall"], pq[f"{key}|test_latency"] * 1e3
        per_bin[fam] = []
        for b in range(4):
            m = bin_test == b
            per_bin[fam].append((rec[m].mean(), lat[m].mean()))
            out.add(index=fam, build_params=op["build_params"], search_param=SEARCH_PARAM[fam],
                    search_value=op["search_value"], group=BIN_NAMES[b], n_queries=int(m.sum()),
                    recall_at_10=float(rec[m].mean()), recall_p10=float(np.percentile(rec[m], 10)),
                    **{"frac_recall_below_0.8": float((rec[m] < 0.8).mean())},
                    latency_ms_mean=float(lat[m].mean()), latency_ms_p50=float(np.median(lat[m])),
                    latency_ms_p99=float(np.percentile(lat[m], 99)),
                    notes=f"RC in [{lo_hi[b][0]:.4f}, {lo_hi[b][1]:.4f}]; overall test recall {rec.mean():.4f}")
            tag = f"p3_{fam.lower().replace('-', '')}_q{b + 1}"
            numbers[f"{tag}_recall"] = float(rec[m].mean())
            numbers[f"{tag}_lat_ms"] = float(lat[m].mean())
            numbers[f"{tag}_frac_below_08"] = float((rec[m] < 0.8).mean())
        numbers[f"p3_{fam.lower().replace('-', '')}_overall"] = float(rec.mean())
        numbers[f"p3_{fam.lower().replace('-', '')}_spearman"] = float(
            pd.Series(rc[test_idx]).corr(pd.Series(rec), method="spearman"))
        numbers[f"p3_{fam.lower().replace('-', '')}_spearman_lat"] = float(
            pd.Series(rc[test_idx]).corr(pd.Series(lat), method="spearman"))

        # search effort each bin needs: smallest value whose mean TUNING recall in that bin is >= 0.95
        values = sorted(int(kk.split("|")[2]) for kk in pq
                        if kk.startswith(f"{fam}|{op['build_params']}|") and kk.endswith("|tune_recall"))
        need = []
        for b in range(4):
            v_ok = [v for v in values if pq[f"{fam}|{op['build_params']}|{v}|tune_recall"][bin_tune == b].mean() >= 0.95]
            need.append(v_ok[0] if v_ok else None)
        effort_rows.append({"Index (build)": f"{fam} ({op['build_params']})",
                            "Parameter": SEARCH_PARAM[fam],
                            "Used for overall 0.95": op["search_value"] if selected[fam].get("R0.95") else "-",
                            **{BIN_NAMES[b]: (need[b] if need[b] is not None else f">{values[-1]}") for b in range(4)}})
        for b in range(4):
            numbers[f"p3_{fam.lower().replace('-', '')}_need_q{b + 1}"] = need[b]
        per_bin[fam + "_curves"] = (values, [[pq[f"{fam}|{op['build_params']}|{v}|test_recall"][bin_test == b].mean()
                                              for v in values] for b in range(4)])

    for b in range(4):
        row = {"Bin (RC quartile)": BIN_NAMES[b], "RC range": f"{lo_hi[b][0]:.3f}-{lo_hi[b][1]:.3f}",
               "Test queries": int((bin_test == b).sum())}
        for fam in FAMILIES:
            if fam in per_bin:
                row[f"{fam} recall"] = f"{per_bin[fam][b][0]:.3f}"
        for fam in FAMILIES:
            if fam in per_bin:
                row[f"{fam} ms/query"] = f"{per_bin[fam][b][1]:.2f}"
        bin_rows.append(row)
    write_table(pd.DataFrame(bin_rows), P["tables"] / "part3_bins")
    write_table(pd.DataFrame(effort_rows), P["tables"] / "part3_effort")

    # ---- figure: (a) recall per bin, (b) latency per bin, (c, d) recall vs search parameter per bin
    fams = [f for f in FAMILIES if f in per_bin]
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.4))
    x = np.arange(4)
    w = 0.8 / len(fams)
    for j, fam in enumerate(fams):
        r = [per_bin[fam][b][0] for b in range(4)]
        l = [per_bin[fam][b][1] for b in range(4)]
        axs[0, 0].bar(x + (j - (len(fams) - 1) / 2) * w, r, w * 0.92, color=STYLE[fam]["color"], label=fam)
        axs[0, 1].plot(x, l, color=STYLE[fam]["color"], marker=STYLE[fam]["marker"], label=fam)
    axs[0, 0].axhline(0.95, color="#52514e", ls=":", lw=0.8)
    axs[0, 0].set_ylim(max(0.0, min(per_bin[f][0][0] for f in fams) - 0.1), 1.0)
    axs[0, 0].set_ylabel("recall@10 (test half)")
    axs[0, 0].set_title("(a) recall per difficulty bin at the 0.95 setting")
    axs[0, 1].set_ylabel("mean latency per query (ms)")
    axs[0, 1].set_title("(b) latency per bin, same settings")
    axs[0, 1].set_yscale("log")
    for ax in axs[0]:
        ax.set_xticks(x, ["Q1\nhardest", "Q2", "Q3", "Q4\neasiest"])
        ax.set_xlabel("relative-contrast quartile")
    axs[0, 1].legend(loc="center right")          # one legend; (a) uses the same colours
    for ax, fam, letter in ((axs[1, 0], "HNSW", "c"), (axs[1, 1], "IVF-Flat", "d")):
        if fam + "_curves" not in per_bin:
            continue
        values, curves = per_bin[fam + "_curves"]
        for b in range(4):
            ax.plot(values, curves[b], color=ORDINAL_BLUE[b], marker="o", label=BIN_NAMES[b])
        ax.axhline(0.95, color="#52514e", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(f"{SEARCH_PARAM[fam]} ({fam}, {selected[fam]['best']['build_params'] if not selected[fam].get('R0.95') else selected[fam]['R0.95']['build_params']})")
        ax.set_ylabel("recall@10 (test half)")
        ax.set_title(f"({letter}) {fam}: recall vs search effort per bin")
        ax.set_ylim(0.3, 1.01)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(P["figures"] / "part3_glove_query_difficulty.png")
    plt.close(fig)
    save_numbers(P["tables"] / "report_numbers.json", numbers)
    print("[part3] done:", json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in numbers.items()}))


if __name__ == "__main__":
    main()
