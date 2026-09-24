"""Deterministic covariance-probe segmentation for the frozen v0 experiment."""
from __future__ import annotations

from itertools import combinations, permutations
from typing import Sequence
import numpy as np
from sklearn.covariance import oas


def _sym_eigh(a: np.ndarray, *, positive: bool = False):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise FloatingPointError("matrix must be finite and square")
    a = (a + a.T) * 0.5
    w, v = np.linalg.eigh(a)
    if not np.isfinite(w).all() or (positive and np.min(w) <= 0):
        raise FloatingPointError("matrix is not finite SPD")
    return w, v


def spd_function(a: np.ndarray, kind: str) -> np.ndarray:
    w, v = _sym_eigh(a, positive=True)
    if kind == "sqrt":
        f = np.sqrt(w)
    elif kind == "invsqrt":
        f = 1.0 / np.sqrt(w)
    elif kind == "log":
        f = np.log(w)
    else:
        raise ValueError(kind)
    result = (v * f) @ v.T
    return (result + result.T) * 0.5


def svec(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise FloatingPointError("svec input must be finite square")
    a = (a + a.T) * 0.5
    i, j = np.triu_indices(a.shape[0], 1)
    return np.concatenate((np.diag(a), np.sqrt(2.0) * a[i, j]))


def oas_cov(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[1] < 2:
        raise FloatingPointError("covariance input must be finite channels x time")
    cov, _ = oas(x.T, assume_centered=False)
    cov = (cov + cov.T) * 0.5
    _sym_eigh(cov, positive=True)
    return cov


def probe_sequence(x: np.ndarray, window: int = 125, stride: int = 25):
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[1]
    starts = np.arange(0, n - window + 1, stride, dtype=np.int64)
    covs = np.stack([oas_cov(x[:, s:s + window]) for s in starts])
    reference = oas_cov(x)
    white = spd_function(reference, "invsqrt")
    zs = np.stack([spd_function(white @ c @ white, "log") for c in covs])
    vectors = np.stack([svec(z) for z in zs])
    centers = starts.astype(np.float64) + (window - 1) / 2.0
    return covs, vectors, centers


def _grid_nodes(n: int, grid: int):
    nodes = np.arange(0, n + 1, grid, dtype=np.int64)
    if nodes[-1] != n:
        raise ValueError("trial length must lie exactly on boundary grid")
    return nodes


def adaptive_boundaries(vectors: np.ndarray, centers: np.ndarray, n_samples: int = 1000,
                        k: int = 5, grid: int = 25, min_length: int = 125) -> np.ndarray:
    """Exact O(K*N^2*D) DP with deterministic first-start tie breaking."""
    v = np.asarray(vectors, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    if v.ndim != 2 or c.shape != (len(v),) or not np.isfinite(v).all():
        raise ValueError("invalid probe sequence")
    nodes = _grid_nodes(n_samples, grid)
    nnode, d = len(nodes), v.shape[1]
    min_grid = int(np.ceil(min_length / grid))
    # Prefix statistics give exact Euclidean within-cluster SSE.
    prefix = np.vstack([np.zeros((1, d)), np.cumsum(v, axis=0)])
    prefix_sq = np.concatenate(([0.0], np.cumsum(np.einsum("ij,ij->i", v, v))))
    counts = np.zeros(nnode, dtype=np.int64)
    for j, b in enumerate(nodes):
        counts[j] = np.searchsorted(c, b, side="left")

    def cost(i, j):
        lo, hi = counts[i], counts[j]
        count = hi - lo
        if count == 0:
            return 0.0
        total = prefix[hi] - prefix[lo]
        value = prefix_sq[hi] - prefix_sq[lo] - np.dot(total, total) / count
        return max(0.0, float(value))

    inf = float("inf")
    dp = np.full((k + 1, nnode), inf)
    prev = np.full((k + 1, nnode), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for seg in range(1, k + 1):
        for end in range(seg * min_grid, nnode):
            lo = (seg - 1) * min_grid
            hi = end - min_grid
            for start in range(lo, hi + 1):
                if not np.isfinite(dp[seg - 1, start]):
                    continue
                val = dp[seg - 1, start] + cost(start, end)
                if val < dp[seg, end] - 1e-12:
                    dp[seg, end], prev[seg, end] = val, start
    if not np.isfinite(dp[k, nnode - 1]):
        raise RuntimeError("no valid K-segment solution")
    idx = [nnode - 1]
    end = nnode - 1
    for seg in range(k, 0, -1):
        end = int(prev[seg, end])
        idx.append(end)
    return nodes[np.asarray(idx[::-1])]


def valid_random_boundaries(n_samples: int = 1000, k: int = 5, grid: int = 25,
                            min_length: int = 125, seed: int = 0, reps: int = 20):
    nodes = _grid_nodes(n_samples, grid)
    mg = int(np.ceil(min_length / grid))
    candidates = [np.asarray((0, *cuts, n_samples), dtype=np.int64)
                  for cuts in combinations(nodes[1:-1].tolist(), k - 1)
                  if all(b - a >= mg * grid for a, b in zip((0, *cuts), (*cuts, n_samples)))]
    if len(candidates) < reps:
        raise RuntimeError(f"only {len(candidates)} valid random boundaries")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=reps, replace=False)
    return np.stack([candidates[i] for i in chosen])


def matched_length_permutations(boundaries: Sequence[int], seed: int = 0, max_reps: int = 20):
    b = np.asarray(boundaries, dtype=np.int64)
    lengths = np.diff(b)
    unique = sorted(set(permutations(lengths.tolist())))
    alternatives = [p for p in unique if tuple(p) != tuple(lengths.tolist())]
    if not alternatives:
        raise RuntimeError("adaptive segment lengths have no non-identity permutation")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(alternatives))[:max_reps]
    chosen = [alternatives[i] for i in order]
    return np.stack([np.r_[0, np.cumsum(p)] for p in chosen]).astype(np.int64)


def segment_features(x: np.ndarray, boundaries: Sequence[int]) -> np.ndarray:
    b = np.asarray(boundaries, dtype=np.int64)
    if b[0] != 0 or b[-1] != x.shape[1] or np.any(np.diff(b) < 1):
        raise ValueError("boundaries must cover the trial exactly without gaps")
    return np.stack([svec(spd_function(oas_cov(x[:, a:z]), "log"))
                     for a, z in zip(b[:-1], b[1:])])


def airm_distance(a: np.ndarray, b: np.ndarray) -> float:
    white = spd_function(a, "invsqrt")
    return float(np.linalg.norm(spd_function(white @ b @ white, "log"), ord="fro"))


def exact_sign_flip(differences: Sequence[float]):
    d = np.asarray(differences, dtype=np.float64)
    if d.shape != (9,) or not np.isfinite(d).all():
        raise ValueError("exact protocol requires nine finite subject differences")
    observed = float(np.mean(d))
    signs = np.asarray([[1 if (mask >> i) & 1 else -1 for i in range(9)]
                        for mask in range(512)], dtype=np.float64)
    null = (signs * d[None, :]).mean(axis=1)
    p = float(np.count_nonzero(null >= observed - 1e-14) / 512)
    return {"mean_difference": observed, "median_difference": float(np.median(d)),
            "p_one_sided": p, "n_sign_configurations": 512,
            "subject_differences": d.tolist()}
