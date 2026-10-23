"""Part 1: exact search baseline, ground truth, agreement with the stored neighbours, timing.

Usage: python scripts/part1_exact.py --dataset sift|glove [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from annbench import config
from annbench.data import load_dataset, query_split
from annbench.indexes import build_threads, index_size_bytes
from annbench.metrics import exact_sq_dists, ground_truth, recall_at_k
from annbench.results import RowWriter, recall_summary, save_numbers, timing_summary
from annbench.timing import time_batch, time_single_queries


def verify_against_stored(ds, gt_ids, gt_d, k):
    """Compare our exact neighbours with the ones stored in the file (full base set only)."""
    q = ds.queries
    stored = ds.stored_ids
    # our own precise distances for the stored ids, re-sorted by (distance, id) so ties are canonical
    sd = exact_sq_dists(q, ds.base, stored)
    order = np.lexsort((stored, sd), axis=1)
    stored_sorted = np.take_along_axis(stored, order, axis=1)
    sd_sorted = np.take_along_axis(sd, order, axis=1)

    ours_top = gt_ids[:, :k]
    file_top = stored[:, :k]                      # in the file's own order
    same_set = np.array([set(a) == set(b) for a, b in zip(ours_top, file_top)])
    same_order = np.all(ours_top == file_top, axis=1)
    r_ours_vs_file = recall_at_k(ours_top, stored_sorted, sd_sorted, k)
    r_file_vs_ours = recall_at_k(file_top, gt_ids, gt_d, k)

    # differences that are not ties: an id in one list but not the other whose distance differs from d_k
    d_k = gt_d[:, k - 1]
    only_ties = np.ones(len(q), dtype=bool)
    for i in np.where(~same_set)[0]:
        diff = list(set(ours_top[i]) ^ set(file_top[i]))
        dd = exact_sq_dists(q[i:i + 1], ds.base, np.array([diff]))[0]
        only_ties[i] = np.allclose(dd, d_k[i], rtol=1e-6)

    # stored distances in the file's own units vs ours (SIFT: Euclidean; GloVe: 1 - cosine)
    ours_in_file_units = np.sqrt(gt_d[:, :k]) if ds.metric == "euclidean" else gt_d[:, :k] / 2.0
    dist_err = np.abs(ours_in_file_units - ds.stored_dists[:, :k])
    return {
        "frac_identical_top10_set": float(same_set.mean()),
        "frac_identical_top10_order": float(same_order.mean()),
        "n_queries_differing": int((~same_set).sum()),
        "n_differing_only_by_ties": int((~same_set & only_ties).sum()),
        "recall_ours_vs_file_mean": float(r_ours_vs_file.mean()),
        "recall_ours_vs_file_min": float(r_ours_vs_file.min()),
        "recall_file_vs_ours_mean": float(r_file_vs_ours.mean()),
        "recall_file_vs_ours_min": float(r_file_vs_ours.min()),
        "max_abs_dist_diff": float(dist_err.max()),
        "max_rel_dist_diff": float((dist_err / np.maximum(ds.stored_dists[:, :k], 1e-12)).max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sift", "glove"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg, P = config.settings(args.quick), config.paths(args.quick)
    k = config.K

    ds = load_dataset(args.dataset, P["data"], cfg["n_base"])
    tune_idx, test_idx = query_split(cfg["n_tune"], cfg["n_test"])
    n, d = ds.base.shape
    print(f"[part1] {ds.name}: base {ds.base.shape}, queries {ds.queries.shape}, zero vectors {ds.n_zero}", flush=True)

    out = RowWriter(P["work"] / f"part1_{ds.name}.csv", experiment="part1_exact", dataset=ds.name,
                    mode=cfg["mode"], n_base=n, dim=d, index="Exact", build_params="IndexFlatL2", k=k,
                    build_threads=build_threads())

    faiss.omp_set_num_threads(build_threads())
    t0 = time.perf_counter()
    flat = faiss.IndexFlatL2(d)
    flat.add(ds.base)
    build_time = time.perf_counter() - t0
    size = index_size_bytes(flat, P["work"])

    # Ground truth for all 10,000 queries (exact search, float64-refined, depth GT_K).
    t0 = time.perf_counter()
    gt_ids, gt_d = ground_truth(flat, ds.base, ds.queries, config.GT_K)
    print(f"[part1] ground truth in {time.perf_counter() - t0:.1f}s", flush=True)
    np.savez(P["work"] / f"gt_{ds.name}.npz", ids=gt_ids, dists=gt_d)

    numbers = {}
    if not ds.is_subset:
        ver = verify_against_stored(ds, gt_ids, gt_d, k)
        print("[part1] agreement with stored neighbours:", json.dumps(ver, indent=1), flush=True)
        out.add(experiment="part1_verify", search_param="-", search_value="all", split="all",
                n_queries=len(ds.queries), recall_at_10=ver["recall_ours_vs_file_mean"], notes=json.dumps(ver))
        numbers.update({f"p1_{ds.name}_{key}": v for key, v in ver.items()})
    else:
        print("[part1] base set is a subset: the stored neighbours refer to the full set, skipping the check")

    # Timing on the test half.
    q_test = ds.queries[test_idx]
    q_single = q_test[:cfg["exact_single_queries"]]
    ids, qps_runs, lat = time_single_queries(flat, q_single, k, cfg["repeats"], cfg["warmup"])
    rec = recall_at_k(ids, gt_ids[test_idx][:len(q_single)], gt_d[test_idx][:len(q_single)], k)
    out.add(search_param="mode", search_value="single", split="test", n_queries=len(q_single),
            **recall_summary(rec), **timing_summary(qps_runs, lat), build_time_s=build_time,
            train_time_s=0.0, add_time_s=build_time, index_bytes=size, bytes_per_vector=size / n,
            search_threads=1, repeats=cfg["repeats"],
            notes=f"one query per call on the first {len(q_single)} test queries")
    print(f"[part1] single-query exact: median {np.median(qps_runs):.1f} QPS, recall {rec.mean():.4f}", flush=True)
    numbers[f"p1_{ds.name}_single_qps"] = float(np.median(qps_runs))
    numbers[f"p1_{ds.name}_single_ms"] = float(1e3 / np.median(qps_runs))

    for threads in (1, build_threads()):
        qps_b = time_batch(flat, q_test, k, cfg["repeats"], threads)
        out.add(search_param="mode", search_value="batch", split="test", n_queries=len(q_test),
                recall_at_10=1.0, qps_median=float(np.median(qps_b)),
                qps_runs=";".join(f"{x:.2f}" for x in qps_b), search_threads=threads, repeats=cfg["repeats"],
                build_time_s=build_time, index_bytes=size, bytes_per_vector=size / n,
                notes=f"all {len(q_test)} test queries in one search call")
        print(f"[part1] batch exact, {threads} thread(s): median {np.median(qps_b):.1f} QPS", flush=True)
        numbers[f"p1_{ds.name}_batch_qps_" + ("t1" if threads == 1 else "tmax")] = float(np.median(qps_b))
    # FAISS switches IndexFlatL2.search from a per-query SIMD scan to a BLAS matrix multiply once
    # n_queries * d >= distance_compute_blas_threshold. Time one batch just below and one just above
    # that boundary on one thread: same work per query, only the algorithm changes.
    thr = int(faiss.cvar.distance_compute_blas_threshold)
    n_above = -(-thr // d)
    for label, nq in (("batch_below_blas_threshold", n_above - 1), ("batch_above_blas_threshold", n_above)):
        if nq > len(ds.queries):
            continue
        qps_b = time_batch(flat, ds.queries[:nq], k, cfg["repeats"], 1)
        out.add(search_param="mode", search_value=label, split="all", n_queries=nq, qps_median=float(np.median(qps_b)),
                qps_runs=";".join(f"{x:.2f}" for x in qps_b), search_threads=1, repeats=cfg["repeats"],
                notes=f"distance_compute_blas_threshold={thr}; n_queries*d={nq * d}")
        print(f"[part1] {label} ({nq} queries, nq*d={nq * d}): median {np.median(qps_b):.1f} QPS", flush=True)
        numbers[f"p1_{ds.name}_{label}_qps"] = float(np.median(qps_b))
        numbers[f"p1_{ds.name}_{label}_nq"] = nq
    numbers["p1_blas_threshold"] = thr
    numbers[f"p1_{ds.name}_build_s"] = build_time
    numbers[f"p1_{ds.name}_n_zero"] = ds.n_zero
    save_numbers(P["tables"] / "report_numbers.json", numbers)


if __name__ == "__main__":
    main()
