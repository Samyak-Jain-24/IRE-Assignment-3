"""Exact distances, ground truth and the tie-aware recall@k function."""
import faiss
import numpy as np


def exact_sq_dists(queries: np.ndarray, base: np.ndarray, ids: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Squared L2 distance between queries[i] and base[ids[i, j]], computed in float64.

    Computing the difference directly (instead of ||q||^2 + ||x||^2 - 2 q.x as BLAS does)
    keeps exact ties exact, e.g. SIFT vectors are integer-valued so their distances are
    integers. Entries with id -1 (FAISS's "no result") get +inf.
    """
    ids = np.asarray(ids, dtype=np.int64)
    out = np.full(ids.shape, np.inf, dtype=np.float64)
    for s in range(0, len(ids), chunk):
        idc = ids[s:s + chunk]
        valid = idc >= 0
        x = base[np.where(valid, idc, 0)].astype(np.float64)       # (c, k, d)
        q = queries[s:s + chunk, None, :].astype(np.float64)        # (c, 1, d)
        d = ((x - q) ** 2).sum(axis=-1)
        out[s:s + chunk] = np.where(valid, d, np.inf)
    return out


def ground_truth(index_flat, base: np.ndarray, queries: np.ndarray, k: int, extra: int = 20):
    """Exact k nearest neighbours with float64-refined distances, sorted by (distance, id).

    The flat index returns the top k + extra by float32 BLAS distances; these are recomputed
    exactly and re-sorted, so rounding cannot reorder ties near the k-th position.
    """
    _, I = index_flat.search(queries, k + extra)
    D = exact_sq_dists(queries, base, I)
    order = np.lexsort((I, D), axis=1)          # primary key distance, secondary key id
    I = np.take_along_axis(I, order, axis=1)[:, :k]
    D = np.take_along_axis(D, order, axis=1)[:, :k]
    return I, D


def recall_at_k(retrieved_ids, gt_ids, gt_dists, k: int = 10, rtol: float = 1e-6, atol: float = 1e-9) -> np.ndarray:
    """Tie-aware recall@k for every query.

    A retrieved id counts as a hit if it is a true neighbour whose distance is no larger than
    the distance of the true k-th neighbour (within a tiny tolerance). Any point tied with the
    k-th neighbour is therefore accepted, so a tie at position k is not counted as a mistake.
    Each id counts at most once, ids of -1 are misses, and the hit count is capped at k.

    retrieved_ids : (nq, >=k) ids returned by the index (only the first k are used)
    gt_ids        : (nq, K) exact neighbour ids sorted by distance, K >= k
    gt_dists      : (nq, K) their exact distances (any monotone distance, e.g. squared L2)
    returns       : (nq,) recall of each query in [0, 1]
    """
    retrieved = np.asarray(retrieved_ids)[:, :k]
    gt_ids = np.asarray(gt_ids)
    gt_dists = np.asarray(gt_dists, dtype=np.float64)
    if gt_ids.shape != gt_dists.shape or gt_ids.shape[1] < k:
        raise ValueError("gt_ids and gt_dists must have the same shape with at least k columns")
    if len(retrieved) != len(gt_ids):
        raise ValueError("retrieved_ids and gt_ids must have the same number of queries")
    thresh = gt_dists[:, k - 1:k] * (1.0 + rtol) + atol
    accept = gt_dists <= thresh                     # true neighbours within the k-th distance
    if gt_ids.shape[1] > k and accept[:, -1].any():
        raise ValueError("ties at the k-th distance extend past the ground-truth list; use a deeper list")
    out = np.empty(len(retrieved), dtype=np.float64)
    for i in range(len(retrieved)):
        valid = set(gt_ids[i, accept[i]].tolist())
        hits = len(valid.intersection(int(r) for r in retrieved[i] if r >= 0))
        out[i] = min(hits, k) / k
    return out
