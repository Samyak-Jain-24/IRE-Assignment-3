"""Part 4 follow-up (post hoc, not part of the pre-registered hypothesis): why does an HNSW graph
built in two batches (old vectors, then new vectors) lose recall even without any drift?

On the control data of Part 4 (delta = 0) three HNSW graphs are built from the same 400K vectors:
  two-batch, 8 threads   add(old) then add(new)            (the "stale" index of Part 4)
  two-batch, 1 thread    the same, inserted sequentially   (rules out parallel-insertion races)
  one pass, 8 threads    all vectors in one shuffled add() (the "rebuilt" index of Part 4)
For each graph we measure recall at several efSearch values (is the loss structural, i.e. does it
persist at very large efSearch?), how many base-layer nodes have no incoming edge (such nodes can
only be reached through the upper layers), and what share of base-layer edges cross between the old
and the new batch (the result notes also record whether missed neighbours are old or new).

Usage: python scripts/part4_hnsw_order.py [--quick]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np
import pandas as pd

from annbench import config
from annbench.indexes import build_threads
from annbench.memory import pin_working_set
from annbench.metrics import ground_truth, recall_at_k
from annbench.report_utils import write_table
from annbench.results import RowWriter, save_numbers
from annbench.synth import D, make_drift_data
from annbench.timing import limit_threads

pin_working_set(3 << 30)
EF_VALUES = [24, 64, 256, 1024]


def level0_neighbours(index) -> np.ndarray:
    """(n, 2M) array of base-layer neighbour ids (-1 = empty slot)."""
    hnsw = index.hnsw
    nb = faiss.vector_to_array(hnsw.neighbors)
    offsets = faiss.vector_to_array(hnsw.offsets)[:-1].astype(np.int64)
    width = int(faiss.vector_to_array(hnsw.cum_nneighbor_per_level)[1])
    return nb[offsets[:, None] + np.arange(width)[None, :]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg, P = config.settings(args.quick), config.paths(args.quick)
    k = config.K
    n_tune, n_test = cfg["p4_n_tune"], cfg["p4_n_test"]
    data = make_drift_data(0.0, cfg["p4_n_old"], cfg["p4_n_new"], n_tune + n_test, config.SEED)
    n_old = len(data.old)
    allv = np.vstack([data.old, data.new])
    q = data.queries[n_tune:]                                  # test queries only
    flat = faiss.IndexFlatL2(D)
    flat.add(allv)
    gt_ids, gt_d = ground_truth(flat, allv, q, config.GT_K)
    del flat
    out = RowWriter(P["work"] / "part4_hnsw_order.csv", experiment="part4_hnsw_order", dataset="synthetic_drift",
                    mode=cfg["mode"], n_base=len(allv), dim=D, k=k, index="HNSW", group="delta=0",
                    build_params=f"M={cfg['p4_hnsw_M']},efC={cfg['p4_hnsw_efC']}", search_param="efSearch",
                    split="test", n_queries=n_test)
    variants = [("two-batch, 8 threads", "two_batch", build_threads()), ("two-batch, 1 thread", "two_batch", 1),
                ("one pass, 8 threads", "one_pass", build_threads())]
    rows, numbers = [], {}
    for label, how, threads in variants:
        index = faiss.IndexHNSWFlat(D, cfg["p4_hnsw_M"])
        index.hnsw.efConstruction = cfg["p4_hnsw_efC"]
        perm = None
        t0 = time.perf_counter()
        with limit_threads(threads):
            if how == "two_batch":
                index.add(data.old)
                index.add(data.new)
            else:
                perm = np.random.default_rng(config.SEED).permutation(len(allv))
                index.add(allv[perm])
        build_s = time.perf_counter() - t0
        # base-layer in-degree, mapped back to original ids
        nb0 = level0_neighbours(index)
        indeg = np.bincount(nb0[nb0 >= 0], minlength=len(allv))
        levels = faiss.vector_to_array(index.hnsw.levels)          # 1 = base layer only
        orig = perm if perm is not None else np.arange(len(allv))
        indeg_orig = np.empty_like(indeg)
        indeg_orig[orig] = indeg
        level_orig = np.empty_like(levels)
        level_orig[orig] = levels
        orphan = (indeg_orig == 0) & (level_orig == 1)              # no way in except as an entry point
        is_old = np.arange(len(allv)) < n_old
        # how well the two batches are wired together: share of base-layer edges crossing old <-> new
        src = np.repeat(orig, nb0.shape[1]).reshape(nb0.shape)
        valid = nb0 >= 0
        dst_old = orig[np.maximum(nb0, 0)][valid] < n_old
        src_old = src[valid] < n_old
        old_to_new = float((src_old & ~dst_old).sum() / src_old.sum())
        new_to_old = float((~src_old & dst_old).sum() / (~src_old).sum())
        res = {"Build": label, "Build time (s)": f"{build_s:.0f}",
               "orphans, old (%)": f"{100 * orphan[is_old].mean():.2f}",
               "orphans, new (%)": f"{100 * orphan[~is_old].mean():.2f}",
               "median in-degree old / new": f"{np.median(indeg_orig[is_old]):.0f} / {np.median(indeg_orig[~is_old]):.0f}",
               "edges old&rarr;new / new&rarr;old (%)": f"{100 * old_to_new:.0f} / {100 * new_to_old:.0f}"}
        with limit_threads(build_threads()):
            for ef in EF_VALUES:
                index.hnsw.efSearch = ef
                _, I = index.search(q, k)
                if perm is not None:
                    I = np.where(I >= 0, perm[np.maximum(I, 0)], -1)
                rec = recall_at_k(I, gt_ids, gt_d, k)
                res[f"recall ef={ef}"] = f"{rec.mean():.3f}"
                # which true top-10 neighbours were missed: old or new vectors?
                missed = [g for i in range(len(q)) for g in gt_ids[i, :k] if g not in set(I[i])]
                frac_old = float(np.mean(np.array(missed) < n_old)) if missed else float("nan")
                out.add(variant=label, search_value=ef, recall_at_10=float(rec.mean()), build_time_s=build_s,
                        build_threads=threads, notes=f"orphans_old={orphan[is_old].mean():.4f}; "
                        f"orphans_new={orphan[~is_old].mean():.4f}; missed_true_neighbours_old={frac_old:.3f}")
                if ef == EF_VALUES[-1]:
                    res["queries with misses"] = int((rec < 1).sum())
        tag = "p4o_" + how + ("_t1" if threads == 1 else "")
        numbers.update({f"{tag}_orphan_old_pct": 100 * float(orphan[is_old].mean()),
                        f"{tag}_orphan_new_pct": 100 * float(orphan[~is_old].mean()),
                        f"{tag}_recall_ef{EF_VALUES[0]}": float(res[f"recall ef={EF_VALUES[0]}"]),
                        f"{tag}_recall_ef{EF_VALUES[-1]}": float(res[f"recall ef={EF_VALUES[-1]}"]),
                        f"{tag}_old_to_new_pct": 100 * old_to_new, f"{tag}_new_to_old_pct": 100 * new_to_old,
                        f"{tag}_indeg_old": float(np.median(indeg_orig[is_old])),
                        f"{tag}_indeg_new": float(np.median(indeg_orig[~is_old]))})
        rows.append(res)
        print("[part4-order]", res, flush=True)
        del index
    write_table(pd.DataFrame(rows), P["tables"] / "part4_hnsw_order")
    save_numbers(P["tables"] / "report_numbers.json", numbers)


if __name__ == "__main__":
    main()
