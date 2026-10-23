"""Seeded synthetic data for Part 4: a low-rank Gaussian mixture whose new data drifts.

The mixture has N_CLUSTERS clusters in D dimensions. Cluster j has a centre c_j ~ N(0, I) and
spans its own random RANK-dimensional subspace (orthonormal basis B_j), plus a little isotropic
noise:   x = c_j + B_j z + NOISE * e,   z ~ N(0, I_RANK),  e ~ N(0, I_D).
This gives real-data-like structure (clusters with low intrinsic dimension), which keeps the
undrifted control easy for every index, so any degradation can be attributed to the drift.

make_drift_data(delta) draws the old vectors, the new vectors and the queries from the same
mixture and then shifts the new vectors and the queries by delta * R * u, where u is a fixed
random unit vector and R is the RMS radius of the old data. All random numbers are drawn
before delta is used, so every delta sees exactly the same underlying draws (common random
numbers) and only the size of the shift changes.
"""
from dataclasses import dataclass

import numpy as np

D = 64
N_CLUSTERS = 50
RANK = 8
NOISE = 0.05


@dataclass
class DriftData:
    old: np.ndarray        # (n_old, D) vectors the index is trained / built on
    new: np.ndarray        # (n_new, D) vectors added afterwards (shifted by delta * R * u)
    queries: np.ndarray    # (n_queries, D) queries from the shifted distribution
    delta: float
    radius: float          # R, RMS distance of the old vectors from their mean
    direction: np.ndarray  # u


def _sample(rng, centres, bases, n):
    labels = rng.integers(0, len(centres), size=n)
    z = rng.normal(size=(n, bases.shape[2]))
    x = centres[labels] + NOISE * rng.normal(size=(n, centres.shape[1]))
    for j in range(len(centres)):
        m = labels == j
        x[m] += z[m] @ bases[j].T
    return x


def make_drift_data(delta: float, n_old: int, n_new: int, n_queries: int, seed: int) -> DriftData:
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(N_CLUSTERS, D))
    bases = np.stack([np.linalg.qr(rng.normal(size=(D, RANK)))[0] for _ in range(N_CLUSTERS)])
    old = _sample(rng, centres, bases, n_old)
    new = _sample(rng, centres, bases, n_new)
    queries = _sample(rng, centres, bases, n_queries)
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)
    radius = float(np.sqrt(((old - old.mean(0)) ** 2).sum(1).mean()))
    shift = delta * radius * u
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    return DriftData(f32(old), f32(new + shift), f32(queries + shift), delta, radius, u)
