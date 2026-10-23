"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ (several build settings x search-parameter sweep).

For every build setting the search parameter is swept upwards. At each value the index is timed
one query at a time on a single thread on both halves of the queries:
  * tuning half  (queries 0..4999):    `tune_repeats` repetition(s), used only to make choices
  * test half    (queries 5000..9999): `repeats` repetitions, the median QPS is reported
The sweep stops once the tuning recall reaches `stop_recall`, or once it stops improving.

Usage: python scripts/part2_benchmark.py --dataset sift|glove [--quick]
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from annbench import config
from annbench.data import load_dataset, query_split
from annbench.indexes import (SEARCH_PARAM, build_hnsw, build_ivf_flat, build_ivf_pq, build_threads,
                              extra_ram_bytes, index_size_bytes, set_search_param, train_coarse_quantizer)
from annbench.metrics import recall_at_k
from annbench.results import RowWriter, recall_summary, timing_summary
from annbench.timing import time_single_queries


def reset_stats():
    faiss.cvar.hnsw_stats.reset()
    faiss.cvar.indexIVF_stats.reset()


def dist_comps(family: str) -> float:
    if family == "HNSW":
        return float(faiss.cvar.hnsw_stats.ndis)
    return float(faiss.cvar.indexIVF_stats.ndis)        # codes scanned in the inverted lists


class Sweeper:
    def __init__(self, ds, cfg, P, tune_idx, test_idx, gt_ids, gt_d):
        self.ds, self.cfg, self.P = ds, cfg, P
        self.q = {"tune": ds.queries[tune_idx], "test": ds.queries[test_idx]}
        self.gt = {"tune": (gt_ids[tune_idx], gt_d[tune_idx]), "test": (gt_ids[test_idx], gt_d[test_idx])}
        self.out = RowWriter(P["work"] / f"part2_{ds.name}.csv", experiment="part2_benchmark", dataset=ds.name,
                             mode=cfg["mode"], n_base=len(ds.base), dim=ds.dim, k=config.K,
                             build_threads=build_threads(), search_threads=1)
        self.perquery = {}

    def run(self, index, family, build_params, info, grid):
        cfg, k = self.cfg, config.K
        n = len(self.ds.base)
        size = index_size_bytes(index)
        build_cols = dict(index=family, build_params=build_params, search_param=SEARCH_PARAM[family],
                          build_time_s=info["train_time_s"] + info["add_time_s"], train_time_s=info["train_time_s"],
                          add_time_s=info["add_time_s"], index_bytes=size, bytes_per_vector=size / n)
        print(f"[part2] {self.ds.name} {family} {build_params}: build {build_cols['build_time_s']:.1f}s, "
              f"{size / 2**20:.1f} MiB", flush=True)
        # untimed warm-up: one multi-threaded pass over all queries so the index is resident in RAM
        set_search_param(index, family, grid[0])
        faiss.omp_set_num_threads(build_threads())
        index.search(np.vstack([self.q["tune"], self.q["test"]]), k)
        history = []
        for value in grid:
            set_search_param(index, family, value)
            row_common = dict(build_cols, search_value=value, extra_ram_bytes=extra_ram_bytes(index))
            for split, reps in (("tune", cfg["tune_repeats"]), ("test", cfg["repeats"])):
                q = self.q[split]
                reset_stats()
                ids, qps_runs, lat = time_single_queries(index, q, k, reps, cfg["warmup"])
                n_searches = min(cfg["warmup"], len(q)) + reps * len(q)
                rec = recall_at_k(ids, *self.gt[split], k)
                self.out.add(split=split, n_queries=len(q), repeats=reps, **row_common,
                             **recall_summary(rec), **timing_summary(qps_runs, lat),
                             dist_comps_per_query=dist_comps(family) / n_searches)
                key = f"{family}|{build_params}|{value}"
                self.perquery[f"{key}|{split}_recall"] = rec.astype(np.float32)
                if split == "test":
                    self.perquery[f"{key}|test_latency"] = lat.astype(np.float32)
                else:
                    history.append(rec.mean())
                print(f"    {SEARCH_PARAM[family]}={value:<5} {split}: recall {rec.mean():.4f}  "
                      f"QPS {np.median(qps_runs):9.1f}", flush=True)
            if history[-1] >= cfg["stop_recall"]:
                break
            if len(history) >= 3 and history[-1] - history[-2] < cfg["plateau_eps"] \
                    and history[-2] - history[-3] < cfg["plateau_eps"]:
                break
        np.savez_compressed(self.P["work"] / f"part2_{self.ds.name}_perquery.npz", **self.perquery)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sift", "glove"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg, P = config.settings(args.quick), config.paths(args.quick)

    ds = load_dataset(args.dataset, P["data"], cfg["n_base"])
    tune_idx, test_idx = query_split(cfg["n_tune"], cfg["n_test"])
    gt = np.load(P["work"] / f"gt_{ds.name}.npz")
    assert cfg["pq_nlist"] in cfg["ivf_nlist"], "IVF-PQ reuses one of the IVF-Flat coarse quantizers"
    sw = Sweeper(ds, cfg, P, tune_idx, test_idx, gt["ids"], gt["dists"])
    t_start = time.perf_counter()

    for M in cfg["hnsw_M"]:
        index, info = build_hnsw(ds.base, M, cfg["hnsw_efC"])
        sw.run(index, "HNSW", f"M={M},efC={cfg['hnsw_efC']}", info, config.EF_GRID)
        del index
        gc.collect()

    for nlist in cfg["ivf_nlist"]:
        quantizer, q_time = train_coarse_quantizer(ds.base, nlist, cfg["kmeans_iter"], config.SEED)
        grid = [p for p in config.NPROBE_GRID if p <= nlist]
        index, info = build_ivf_flat(ds.base, quantizer, q_time)
        sw.run(index, "IVF-Flat", f"nlist={nlist}", info, grid)
        del index
        gc.collect()
        if nlist == cfg["pq_nlist"]:
            for m in cfg["pq_m"][ds.name]:
                index, info = build_ivf_pq(ds.base, quantizer, q_time, m, config.SEED,
                                           pq_niter=cfg["pq_niter"],
                                           pq_points_per_centroid=cfg["pq_points_per_centroid"])
                sw.run(index, "IVF-PQ", f"nlist={nlist},m={m},nbits=8", info, grid)
                del index
                gc.collect()
        del quantizer
    print(f"[part2] {ds.name} done in {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
