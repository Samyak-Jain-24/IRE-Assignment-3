"""Loading the ann-benchmarks HDF5 files and splitting the queries."""
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

DATASETS = {
    "sift": {"file": "sift-128-euclidean.hdf5", "metric": "euclidean"},
    "glove": {"file": "glove-100-angular.hdf5", "metric": "angular"},
}
BASE_URL = "http://ann-benchmarks.com/"
N_QUERIES_TOTAL = 10_000


@dataclass
class Dataset:
    name: str
    metric: str            # metric of the original file; we always search with L2 on (normalised) vectors
    base: np.ndarray       # (n, d) float32
    queries: np.ndarray    # (10000, d) float32
    stored_ids: np.ndarray     # (10000, 100) neighbours stored in the file (w.r.t. the FULL base set)
    stored_dists: np.ndarray   # (10000, 100) distances stored in the file
    n_full: int            # size of the full base set in the file
    n_zero: int            # number of all-zero vectors that could not be normalised

    @property
    def dim(self) -> int:
        return self.base.shape[1]

    @property
    def is_subset(self) -> bool:
        return len(self.base) < self.n_full


def ensure_downloaded(name: str, data_dir: Path) -> Path:
    path = Path(data_dir) / DATASETS[name]["file"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = BASE_URL + DATASETS[name]["file"]
        print(f"downloading {url} -> {path}", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def l2_normalise(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Scale every row to unit length. All-zero rows are left as zeros (and counted)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    zero = norms[:, 0] == 0
    norms[zero] = 1.0
    return np.ascontiguousarray(x / norms, dtype=np.float32), int(zero.sum())


def load_dataset(name: str, data_dir: Path, n_base: int | None = None) -> Dataset:
    path = ensure_downloaded(name, data_dir)
    with h5py.File(path, "r") as f:
        n_full = f["train"].shape[0]
        n = n_full if n_base is None else min(n_base, n_full)
        base = np.ascontiguousarray(f["train"][:n], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        stored_ids = f["neighbors"][:].astype(np.int64)
        stored_dists = f["distances"][:].astype(np.float64)
    n_zero = 0
    if DATASETS[name]["metric"] == "angular":
        # Normalise so that Euclidean distance ranks exactly like cosine distance:
        # ||a - b||^2 = 2 - 2 cos(a, b) for unit vectors.
        base, n_zero = l2_normalise(base)
        queries, _ = l2_normalise(queries)
    return Dataset(name, DATASETS[name]["metric"], base, queries, stored_ids, stored_dists, n_full, n_zero)


def query_split(n_tune: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Tune on the first half of the 10,000 queries, report on the second half.

    With n_tune = n_test = 5000 this is exactly queries 0..4999 vs 5000..9999; the quick
    mode takes the first n_tune / n_test queries of each half.
    """
    half = N_QUERIES_TOTAL // 2
    assert n_tune <= half and n_test <= half
    return np.arange(0, n_tune), np.arange(half, half + n_test)
