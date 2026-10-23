"""Building the FAISS indexes, setting their search parameter and measuring their size."""
import os
import time

import faiss
import numpy as np

SEARCH_PARAM = {"HNSW": "efSearch", "IVF-Flat": "nprobe", "IVF-PQ": "nprobe", "Exact": "-"}


def build_threads() -> int:
    return os.cpu_count() or 1


def train_coarse_quantizer(base: np.ndarray, nlist: int, niter: int, seed: int):
    """k-means coarse quantizer shared by IVF-Flat and IVF-PQ with the same nlist.

    k-means samples at most 256 points per centroid (FAISS default). Returns the quantizer
    and the training time, which is charged to every index that uses it.
    """
    faiss.omp_set_num_threads(build_threads())
    t0 = time.perf_counter()
    km = faiss.Kmeans(base.shape[1], nlist, niter=niter, seed=seed, max_points_per_centroid=256, verbose=False)
    km.train(base)
    quantizer = faiss.IndexFlatL2(base.shape[1])
    quantizer.add(km.centroids)
    return quantizer, time.perf_counter() - t0


def build_hnsw(base: np.ndarray, M: int, ef_construction: int):
    faiss.omp_set_num_threads(build_threads())
    index = faiss.IndexHNSWFlat(base.shape[1], M)
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter()
    index.add(base)
    return index, {"train_time_s": 0.0, "add_time_s": time.perf_counter() - t0}


def build_ivf_flat(base: np.ndarray, quantizer, quantizer_time: float):
    faiss.omp_set_num_threads(build_threads())
    index = faiss.IndexIVFFlat(quantizer, base.shape[1], quantizer.ntotal)
    assert index.is_trained
    t0 = time.perf_counter()
    index.add(base)
    return index, {"train_time_s": quantizer_time, "add_time_s": time.perf_counter() - t0}


def build_ivf_pq(base: np.ndarray, quantizer, quantizer_time: float, m: int, seed: int, nbits: int = 8,
                 n_train: int = 200_000, pq_niter: int = 25, pq_points_per_centroid: int = 256):
    """IVF-PQ on residuals (FAISS default). The PQ codebooks are trained on a seeded sample.

    pq_niter / pq_points_per_centroid are the k-means settings of each sub-quantizer
    (25 and 256 are FAISS's defaults; the --quick mode lowers them to save time).
    """
    faiss.omp_set_num_threads(build_threads())
    index = faiss.IndexIVFPQ(quantizer, base.shape[1], quantizer.ntotal, m, nbits)
    index.pq.cp.niter = pq_niter
    index.pq.cp.max_points_per_centroid = pq_points_per_centroid
    index.pq.cp.seed = seed
    rng = np.random.default_rng(seed)
    sample = base if len(base) <= n_train else base[np.sort(rng.choice(len(base), n_train, replace=False))]
    t0 = time.perf_counter()
    index.train(sample)            # the quantizer is already trained, so only the PQ is trained here
    pq_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    index.add(base)
    return index, {"train_time_s": quantizer_time + pq_time, "add_time_s": time.perf_counter() - t0}


def set_search_param(index, family: str, value: int) -> None:
    if family == "HNSW":
        index.hnsw.efSearch = int(value)
    elif family in ("IVF-Flat", "IVF-PQ"):
        index.nprobe = int(value)
    else:
        raise ValueError(family)


def index_size_bytes(index) -> int:
    """Size of the serialised index (vectors/codes + graph/lists + quantizer), in bytes.

    The bytes are only counted, never stored: writing a large index to a temporary file made
    the OS file cache push the freshly built index out of RAM on this 8 GB machine, which
    slowed down the first timed searches.
    """
    total = [0]

    def count(buf):
        total[0] += len(buf)

    writer = faiss.PyCallbackIOWriter(count)
    faiss.write_index(index, writer)
    del writer
    return total[0]


def extra_ram_bytes(index) -> int:
    """RAM used at search time but not serialised (IVF-PQ precomputed distance tables)."""
    if isinstance(index, faiss.IndexIVFPQ):
        try:
            return int(index.precomputed_table.size()) * 4
        except Exception:
            return 0
    return 0


def ivf_list_sizes(index) -> np.ndarray:
    invlists = index.invlists
    return np.array([invlists.list_size(i) for i in range(index.nlist)], dtype=np.int64)
