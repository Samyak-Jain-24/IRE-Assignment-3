"""Experiment settings for the full run and the --quick run, and the directory layout.

Directories (all overridable with environment variables):
  ANN_DATA_DIR     the ann-benchmarks .hdf5 files           (default: ../data)
  ANN_WORK_DIR     caches: ground truth, per-query arrays    (default: ../work or ../work_quick)
  ANN_RESULTS_DIR  results.csv, figures/, tables/            (default: ../results or ../results_quick)
"""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = CODE_DIR.parent

K = 10            # we report recall@10
GT_K = 100        # ground-truth depth kept (must exceed K so ties at the K-th neighbour are captured)
SEED = 1234


def data_dir() -> Path:
    return Path(os.environ.get("ANN_DATA_DIR", ROOT_DIR / "data"))


def paths(quick: bool) -> dict:
    suffix = "_quick" if quick else ""
    p = {
        "data": data_dir(),
        "work": Path(os.environ.get("ANN_WORK_DIR", ROOT_DIR / f"work{suffix}")),
        "results": Path(os.environ.get("ANN_RESULTS_DIR", ROOT_DIR / f"results{suffix}")),
    }
    p["figures"] = p["results"] / "figures"
    p["tables"] = p["results"] / "tables"
    for key in ("work", "results", "figures", "tables"):
        p[key].mkdir(parents=True, exist_ok=True)
    return p


EF_GRID = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
NPROBE_GRID = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

FULL = {
    "mode": "full",
    "n_base": None,               # all vectors
    "n_tune": 5000,               # queries 0..4999
    "n_test": 5000,               # queries 5000..9999
    "exact_single_queries": 1000, # exact one-query-at-a-time timing uses the first 1000 test queries
    "repeats": 3,                 # timed repetitions on the test half (median reported)
    "tune_repeats": 1,            # timed repetitions on the tuning half (used only for selection)
    "warmup": 200,
    "hnsw_M": [8, 16, 32],
    "hnsw_efC": 200,
    "ivf_nlist": [1024, 4096, 8192],
    "pq_nlist": 4096,
    "pq_m": {"sift": [16, 32, 64], "glove": [20, 25, 50]},
    "kmeans_iter": 20,
    "pq_niter": 25,               # FAISS defaults for the PQ sub-quantizer k-means
    "pq_points_per_centroid": 256,
    "stop_recall": 0.99,          # stop a sweep once tuning recall reaches this
    "plateau_eps": 0.003,         # ... or once it gains less than this twice in a row
    # Part 4 (synthetic drift)
    "p4_n_old": 200_000,
    "p4_n_new": 200_000,
    "p4_n_tune": 1000,
    "p4_n_test": 1000,
    "p4_nlist": 1024,
    "p4_hnsw_M": 16,
    "p4_hnsw_efC": 100,
    "p4_pq_m": 16,
    "p4_deltas": [0.0, 0.25, 0.5, 1.0, 2.0],
}

QUICK = dict(FULL)
QUICK.update({
    "mode": "quick",
    "n_base": 100_000,            # first 100K base vectors
    "n_tune": 1000,               # queries 0..999
    "n_test": 1000,               # queries 5000..5999
    "exact_single_queries": 1000,
    "warmup": 100,
    "hnsw_efC": 100,
    "ivf_nlist": [256, 512, 1024],
    "pq_nlist": 512,
    "kmeans_iter": 10,
    "pq_niter": 10,
    "pq_points_per_centroid": 64,
    "p4_n_old": 50_000,
    "p4_n_new": 50_000,
    "p4_n_tune": 500,
    "p4_n_test": 500,
    "p4_nlist": 256,
})


def settings(quick: bool) -> dict:
    return QUICK if quick else FULL
