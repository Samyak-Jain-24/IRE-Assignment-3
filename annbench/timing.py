"""Single-thread, one-query-at-a-time timing (median over repetitions)."""
import gc
import time

import faiss
import numpy as np
from threadpoolctl import threadpool_limits


class limit_threads:
    """Limit FAISS's OpenMP threads *and* every BLAS pool (FAISS ships its own OpenBLAS, whose
    pthreads ignore omp_set_num_threads) to n threads inside the block."""

    def __init__(self, n: int):
        self.n = n

    def __enter__(self):
        self._tp = threadpool_limits(limits=self.n)
        self._tp.__enter__()
        self._prev = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(self.n)
        return self

    def __exit__(self, *exc):
        faiss.omp_set_num_threads(self._prev)
        self._tp.__exit__(*exc)


def time_single_queries(index, queries: np.ndarray, k: int, repeats: int, warmup: int = 200):
    """Search the queries one by one on a single thread, `repeats` times.

    Returns (ids, qps_runs, latency_s) where ids are the results of the first repetition,
    qps_runs[r] = n_queries / wall time of repetition r, and latency_s[i] is the median over
    repetitions of the wall time of query i. A short warm-up pass is run first and is not timed.
    """
    with limit_threads(1):
        return _time_single(index, queries, k, repeats, warmup)


def _time_single(index, queries, k, repeats, warmup):
    nq = len(queries)
    rows = [queries[i:i + 1] for i in range(nq)]          # pre-sliced (1, d) contiguous views
    for i in range(min(warmup, nq)):
        index.search(rows[i], k)
    ids = np.empty((nq, k), dtype=np.int64)
    lat = np.empty((repeats, nq), dtype=np.float64)
    totals = np.empty(repeats, dtype=np.float64)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for r in range(repeats):
            start = time.perf_counter()
            for i in range(nq):
                t0 = time.perf_counter()
                _, I = index.search(rows[i], k)
                lat[r, i] = time.perf_counter() - t0
                if r == 0:
                    ids[i] = I[0]
            totals[r] = time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    return ids, nq / totals, np.median(lat, axis=0)


def time_batch(index, queries: np.ndarray, k: int, repeats: int, threads: int):
    """Search all queries in one call with the given number of threads; returns qps per repetition."""
    with limit_threads(threads):
        index.search(queries[:min(100, len(queries))], k)     # warm-up
        qps = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            index.search(queries, k)
            qps.append(len(queries) / (time.perf_counter() - t0))
    return np.array(qps)


def call_overhead(d: int = 128, n: int = 5000, repeats: int = 3):
    """Time the harness itself: single-query searches on a 1-vector flat index.

    Almost all of this time is the Python loop, the SWIG call and FAISS's per-call setup, so it
    is the fixed cost that every single-query measurement above also pays.
    Returns (qps_runs, latency_s).
    """
    index = faiss.IndexFlatL2(d)
    index.add(np.zeros((1, d), dtype=np.float32))
    q = np.random.default_rng(0).random((n, d), dtype=np.float32)
    _, qps_runs, lat = time_single_queries(index, q, 1, repeats, warmup=200)
    return qps_runs, lat
