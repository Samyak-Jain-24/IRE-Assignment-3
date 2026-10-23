"""Part 4: what happens when data drifts after an index is trained (see hypothesis.md).

For each shift size delta (0 = control), 200K old vectors are indexed, 200K shifted new vectors
are added, and queries come from the shifted distribution. Two variants of every index:
  stale    trained/built on the old vectors, new vectors added afterwards
  rebuilt  trained/built from scratch on all 400K vectors
The search parameter of each index is fixed from the control (delta = 0, stale) tuning queries:
the smallest value reaching recall 0.95 (or 99% of the index's maximum recall if 0.95 is not
reachable). We also record the smallest value each delta needs to get back to 0.95.

Usage: python scripts/part4_drift.py [--quick]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from annbench import config
from annbench.indexes import (SEARCH_PARAM, build_ivf_flat, build_ivf_pq, build_threads, index_size_bytes,
                              ivf_list_sizes, set_search_param, train_coarse_quantizer)
from annbench.metrics import ground_truth, recall_at_k
from annbench.report_utils import FAMILIES, STYLE, setup_matplotlib, write_table
from annbench.results import RowWriter, add_cpu_calibration_row, recall_summary, save_numbers, timing_summary
from annbench.synth import D, make_drift_data
from annbench.timing import time_single_queries

from annbench.memory import pin_working_set

pin_working_set(3 << 30)      # keep the index resident while timing (see annbench/memory.py)

TARGET = 0.95


def build(fam, variant, data, cfg, seed):
    """Returns (index, id_map or None, build info). id_map[i] = original id of the i-th inserted vector."""
    faiss.omp_set_num_threads(build_threads())
    n_old, n_new = len(data.old), len(data.new)
    allv = np.vstack([data.old, data.new])
    if fam == "HNSW":
        index = faiss.IndexHNSWFlat(D, cfg["p4_hnsw_M"])
        index.hnsw.efConstruction = cfg["p4_hnsw_efC"]
        t0 = time.perf_counter()
        if variant == "stale":
            index.add(data.old)
            t1 = time.perf_counter()
            index.add(data.new)
            id_map = None
        else:                                    # rebuilt: one pass over all vectors, old and new interleaved
            perm = np.random.default_rng(seed).permutation(n_old + n_new)
            index.add(allv[perm])
            t1 = time.perf_counter()
            id_map = perm
        t2 = time.perf_counter()
        return index, id_map, {"train_time_s": 0.0, "add_time_s": t2 - t0, "add_new_s": t2 - t1 if variant == "stale" else None}
    train_on = data.old if variant == "stale" else allv
    quantizer, q_time = train_coarse_quantizer(train_on, cfg["p4_nlist"], cfg["kmeans_iter"], seed)
    if fam == "IVF-Flat":
        index, info = build_ivf_flat(train_on, quantizer, q_time)
    else:
        index, info = build_ivf_pq(train_on, quantizer, q_time, cfg["p4_pq_m"], seed, pq_niter=cfg["pq_niter"],
                                   pq_points_per_centroid=cfg["pq_points_per_centroid"])
    index.referenced_objects = [quantizer]
    info["add_new_s"] = None
    if variant == "stale":                       # train_on was the old data only; now add the new vectors
        t0 = time.perf_counter()
        index.add(data.new)
        info["add_new_s"] = time.perf_counter() - t0
        info["add_time_s"] += info["add_new_s"]
    return index, None, info


def remap(ids, id_map):
    if id_map is None:
        return ids
    return np.where(ids >= 0, id_map[np.maximum(ids, 0)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="only redraw figure/tables from the saved summary")
    args = ap.parse_args()
    cfg, P = config.settings(args.quick), config.paths(args.quick)
    k = config.K
    setup_matplotlib()
    if args.replot:
        summary = json.loads((P["work"] / "part4_summary.json").read_text())
        fixed = {s["index"]: s["fixed"]["value"] for s in summary if s["delta"] == cfg["p4_deltas"][0] and s["variant"] == "stale"}
        make_outputs(summary, cfg, P, fixed)
        return
    n_tune, n_test = cfg["p4_n_tune"], cfg["p4_n_test"]
    n_total = cfg["p4_n_old"] + cfg["p4_n_new"]
    grids = {"HNSW": config.EF_GRID, "IVF-Flat": [p for p in config.NPROBE_GRID if p <= cfg["p4_nlist"]],
             "IVF-PQ": [p for p in config.NPROBE_GRID if p <= cfg["p4_nlist"]]}
    build_params = {"HNSW": f"M={cfg['p4_hnsw_M']},efC={cfg['p4_hnsw_efC']}", "IVF-Flat": f"nlist={cfg['p4_nlist']}",
                    "IVF-PQ": f"nlist={cfg['p4_nlist']},m={cfg['p4_pq_m']},nbits=8"}
    out = RowWriter(P["work"] / "part4_drift.csv", experiment="part4_drift", dataset="synthetic_drift",
                    mode=cfg["mode"], n_base=n_total, dim=D, k=k, build_threads=build_threads(), search_threads=1)
    fixed = {}
    summary = []
    print(f"[part4] CPU calibration: {add_cpu_calibration_row(out):.0f} QPS", flush=True)

    for delta in cfg["p4_deltas"]:
        data = make_drift_data(delta, cfg["p4_n_old"], cfg["p4_n_new"], n_tune + n_test, config.SEED)
        allv = np.vstack([data.old, data.new])
        flat = faiss.IndexFlatL2(D)
        flat.add(allv)
        gt_ids, gt_d = ground_truth(flat, allv, data.queries, config.GT_K)
        del flat
        q = {"tune": data.queries[:n_tune], "test": data.queries[n_tune:]}
        gt = {"tune": (gt_ids[:n_tune], gt_d[:n_tune]), "test": (gt_ids[n_tune:], gt_d[n_tune:])}
        d10_median = float(np.median(np.sqrt(gt_d[n_tune:, k - 1])))
        frac_new = float((gt_ids[n_tune:, :k] >= len(data.old)).mean())
        print(f"[part4] delta={delta}: R={data.radius:.2f}, shift={delta * data.radius:.2f}, "
              f"share of true neighbours that are new vectors {frac_new:.3f}", flush=True)

        for variant in ("stale", "rebuilt"):
            for fam in FAMILIES:
                index, id_map, info = build(fam, variant, data, cfg, config.SEED)
                size = index_size_bytes(index)
                mech = {}
                if fam != "HNSW":
                    sizes = ivf_list_sizes(index)
                    mech["max_list_size"] = int(sizes.max())
                    mech["list_imbalance"] = float(len(sizes) * (sizes.astype(float) ** 2).sum() / sizes.sum() ** 2)
                if fam == "IVF-PQ":
                    sample = data.new[:10_000]
                    rec = index.sa_decode(index.sa_encode(sample))
                    mech["pq_rel_error"] = float(np.sqrt(((sample - rec) ** 2).sum(1).mean()) / d10_median)
                common = dict(index=fam, build_params=build_params[fam], variant=variant, group=f"delta={delta:g}",
                              search_param=SEARCH_PARAM[fam], build_time_s=info["train_time_s"] + info["add_time_s"],
                              train_time_s=info["train_time_s"], add_time_s=info["add_time_s"], index_bytes=size,
                              bytes_per_vector=size / n_total, **mech)

                # sweep on the tuning queries (1 timed repetition) until recall >= 0.99 or it plateaus
                tune_curve = []
                for v in grids[fam]:
                    set_search_param(index, fam, v)
                    ids, qps, lat, _ = time_single_queries(index, q["tune"], k, cfg["tune_repeats"], cfg["warmup"])
                    r = recall_at_k(remap(ids, id_map), *gt["tune"], k)
                    tune_curve.append((v, r.mean()))
                    out.add(split="tune", search_value=v, n_queries=n_tune, repeats=cfg["tune_repeats"], notes="sweep",
                            **common, **recall_summary(r), **timing_summary(qps, lat))
                    h = [c[1] for c in tune_curve]
                    if h[-1] >= cfg["stop_recall"] or (len(h) >= 3 and h[-1] - h[-2] < cfg["plateau_eps"]
                                                       and h[-2] - h[-3] < cfg["plateau_eps"]):
                        break
                reach = [v for v, r in tune_curve if r >= TARGET]
                needed = reach[0] if reach else None
                if delta == cfg["p4_deltas"][0] and variant == "stale":
                    best = max(r for _, r in tune_curve)
                    fixed[fam] = needed if needed is not None else next(v for v, r in tune_curve if r >= 0.99 * best)
                    print(f"[part4] fixed {SEARCH_PARAM[fam]} for {fam}: {fixed[fam]}", flush=True)

                # test half: at the fixed (control) setting, and at the setting this delta needs for 0.95
                res = {}
                for label, v in (("fixed", fixed[fam]), ("needed_for_0.95", needed)):
                    if v is None or (label == "needed_for_0.95" and v == fixed[fam]):
                        res[label] = res.get("fixed") if v is not None else None
                        continue
                    set_search_param(index, fam, v)
                    faiss.cvar.hnsw_stats.reset()
                    faiss.cvar.indexIVF_stats.reset()
                    ids, qps, lat, faults = time_single_queries(index, q["test"], k, cfg["p4_repeats"], cfg["warmup"])
                    n_searches = min(cfg["warmup"], n_test) + cfg["p4_repeats"] * n_test
                    ndis = faiss.cvar.hnsw_stats.ndis if fam == "HNSW" else faiss.cvar.indexIVF_stats.ndis
                    r = recall_at_k(remap(ids, id_map), *gt["test"], k)
                    out.add(split="test", search_value=v, n_queries=n_test, repeats=cfg["p4_repeats"], notes=label,
                            dist_comps_per_query=ndis / n_searches, page_faults_per_query=faults,
                            **common, **recall_summary(r),
                            **timing_summary(qps, lat))
                    res[label] = {"value": v, "recall": float(r.mean()), "qps": float(np.median(qps)),
                                  "dist_comps": ndis / n_searches}
                summary.append({"delta": delta, "variant": variant, "index": fam, "build_time_s": common["build_time_s"],
                                "add_new_s": info.get("add_new_s"), **mech, "fixed": res["fixed"],
                                "needed": res["needed_for_0.95"], "frac_new_neighbours": frac_new})
                f = res["fixed"]
                print(f"    {variant:7s} {fam:8s} fixed {SEARCH_PARAM[fam]}={f['value']}: recall {f['recall']:.4f} "
                      f"QPS {f['qps']:8.0f} dist/query {f['dist_comps']:9.0f} | needed for 0.95: "
                      f"{res['needed_for_0.95']['value'] if res['needed_for_0.95'] else 'not reached'} {mech}", flush=True)
                del index
                gc.collect()
        (P["work"] / "part4_summary.json").write_text(json.dumps(summary, indent=1))

    make_outputs(summary, cfg, P, fixed)


def make_outputs(summary, cfg, P, fixed):
    deltas = cfg["p4_deltas"]
    x = np.arange(len(deltas))
    xlabels = [("0\ncontrol" if d == 0 else f"{d:g}") for d in deltas]

    def series(fam, variant, fn):
        rows = {s["delta"]: s for s in summary if s["index"] == fam and s["variant"] == variant}
        return [fn(rows[d]) if d in rows else np.nan for d in deltas]

    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.4))
    for fam in FAMILIES:
        st = STYLE[fam]
        for variant, ls in (("stale", "-"), ("rebuilt", "--")):
            kw = dict(color=st["color"], marker=st["marker"], ls=ls, markerfacecolor=st["color"] if variant == "stale" else "white")
            lab = f"{fam}, {variant}"
            axs[0, 0].plot(x, series(fam, variant, lambda s: s["fixed"]["recall"]), label=lab, **kw)
            axs[0, 1].plot(x, series(fam, variant, lambda s: s["fixed"]["qps"]), label=lab, **kw)
            axs[1, 0].plot(x, series(fam, variant, lambda s: s["fixed"]["dist_comps"]), label=lab, **kw)
            if fam == "IVF-PQ":
                axs[1, 1].plot(x, series(fam, variant, lambda s: s["pq_rel_error"]), label=lab, **kw)
    axs[0, 0].set_title("(a) recall@10 at the control's fixed setting")
    axs[0, 0].set_ylabel("recall@10 (test queries)")
    axs[0, 0].axhline(TARGET, color="#52514e", ls=":", lw=0.8)
    axs[0, 1].set_title("(b) speed at the same fixed setting")
    axs[0, 1].set_ylabel("QPS (1 thread, 1 query/call)")
    axs[0, 1].set_yscale("log")
    axs[1, 0].set_title("(c) work per query: distances computed")
    axs[1, 0].set_ylabel("distance computations / query")
    axs[1, 0].set_yscale("log")
    axs[1, 1].set_title("(d) IVF-PQ encoding error of the new vectors")
    axs[1, 1].set_ylabel("RMS reconstruction error / median d$_{10}$")
    axs[1, 1].set_yscale("log")
    for ax in axs.flat:
        ax.set_xticks(x, xlabels)
        ax.set_xlabel("shift of new data and queries (delta, in units of R)")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(P["figures"] / "part4_drift.png")
    plt.close(fig)

    rows, numbers = [], {}
    for s in summary:
        f, nd = s["fixed"], s["needed"]
        rows.append({
            "delta": f"{s['delta']:g}", "Index": s["index"], "Variant": s["variant"],
            f"Fixed setting": f"{SEARCH_PARAM[s['index']]}={f['value']}",
            "Recall": f"{f['recall']:.3f}", "QPS": f"{f['qps']:,.0f}", "Dist./query": f"{f['dist_comps']:,.0f}",
            "Max list": f"{s['max_list_size']:,}" if "max_list_size" in s else "-",
            "Setting for 0.95": (f"{nd['value']} ({nd['qps']:,.0f} QPS)" if nd else "not reached"),
        })
        tag = f"p4_{s['index'].lower().replace('-', '')}_{s['variant']}_d{str(s['delta']).replace('.', 'p')}"
        numbers[f"{tag}_recall"] = f["recall"]
        numbers[f"{tag}_qps"] = f["qps"]
        numbers[f"{tag}_dist"] = f["dist_comps"]
        numbers[f"{tag}_need"] = nd["value"] if nd else None
        numbers[f"{tag}_need_qps"] = nd["qps"] if nd else None
        for key in ("max_list_size", "list_imbalance", "pq_rel_error"):
            if key in s:
                numbers[f"{tag}_{key}"] = s[key]
        numbers[f"p4_frac_new_d{str(s['delta']).replace('.', 'p')}"] = s["frac_new_neighbours"]
    for fam, v in fixed.items():
        numbers[f"p4_fixed_{fam.lower().replace('-', '')}"] = v
    write_table(pd.DataFrame(rows), P["tables"] / "part4_drift")

    # compact table for the report: stale indexes at every delta, plus the rebuilt indexes at the largest delta
    by = {(s["delta"], s["variant"], s["index"]): s for s in summary}
    compact = []
    for d, variant in [(d, "stale") for d in deltas] + [(deltas[-1], "rebuilt")]:
        if (d, variant, "IVF-PQ") not in by:
            continue
        h, f, q = by[(d, variant, "HNSW")], by[(d, variant, "IVF-Flat")], by[(d, variant, "IVF-PQ")]
        need = lambda s: s["needed"]["value"] if s["needed"] else "-"
        compact.append({
            "delta": f"{d:g}" + (" (control)" if d == 0 else "") + (" rebuilt" if variant == "rebuilt" else ""),
            "HNSW recall": f"{h['fixed']['recall']:.3f}", "HNSW QPS": f"{h['fixed']['qps']:,.0f}",
            "HNSW ef for 0.95": need(h),
            "IVF-Flat recall": f"{f['fixed']['recall']:.3f}", "IVF-Flat QPS": f"{f['fixed']['qps']:,.0f}",
            "IVF-Flat max list": f"{f['max_list_size']:,}",
            "IVF-PQ recall": f"{q['fixed']['recall']:.3f}", "IVF-PQ QPS": f"{q['fixed']['qps']:,.0f}",
            "IVF-PQ error / d10": f"{q['pq_rel_error']:.2f}",
        })
    write_table(pd.DataFrame(compact), P["tables"] / "part4_compact")
    save_numbers(P["tables"] / "report_numbers.json", numbers)


if __name__ == "__main__":
    main()
