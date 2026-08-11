"""Exact unlabeled five-point AIRM distance-geometry primitives.

This module is array-only and has no EEG/data loader.  The quotient is by the
120 vertex relabelings induced by S5, never by arbitrary edge permutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Literal, Sequence

import numpy as np

from src.geometry_v2 import airm_distance, symmetrize


N_VERTICES = 5
N_EDGES = 10
EDGE_ORDER: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)
EDGE_NAMES = ("d12", "d13", "d14", "d15", "d23", "d24", "d25", "d34", "d35", "d45")
VERTEX_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(
    permutations(range(N_VERTICES))
)
_EDGE_TO_INDEX = {edge: index for index, edge in enumerate(EDGE_ORDER)}
INDUCED_EDGE_PERMUTATIONS = np.asarray(
    [
        [
            _EDGE_TO_INDEX[tuple(sorted((permutation[left], permutation[right])))]
            for left, right in EDGE_ORDER
        ]
        for permutation in VERTEX_PERMUTATIONS
    ],
    dtype=np.int64,
)
INDUCED_EDGE_PERMUTATIONS.setflags(write=False)
SQRT_N_EDGES = float(np.sqrt(N_EDGES))
DEFAULT_MATCH_ATOL = 1.0e-12
DEFAULT_MATCH_RTOL = 1.0e-12


class DegenerateMetricConfiguration(ValueError):
    """At least one five-point configuration has zero edge-RMS size."""


@dataclass(frozen=True)
class QuotientMatch:
    distance: float
    permutation_index: int
    vertex_permutation: tuple[int, ...]


@dataclass(frozen=True)
class CounterexampleFixture:
    states_a: np.ndarray
    states_b: np.ndarray
    edges_a: np.ndarray
    edges_b: np.ndarray
    quotient_distance: float
    minimum_orthogonal_scalar_mismatch: float


def _as_edges(values: np.ndarray, *, name: str = "edges") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != N_EDGES:
        raise ValueError(f"{name} must end in {N_EDGES} edge coordinates")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def edge_vector(distance_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.shape != (N_VERTICES, N_VERTICES):
        raise ValueError("distance matrix must have shape (5,5)")
    if not np.isfinite(matrix).all():
        raise ValueError("distance matrix must be finite")
    return np.asarray([matrix[left, right] for left, right in EDGE_ORDER])


def distance_matrix_from_edges(edges: np.ndarray) -> np.ndarray:
    vector = _as_edges(edges)
    if vector.ndim != 1:
        raise ValueError("one edge vector is required")
    matrix = np.zeros((N_VERTICES, N_VERTICES), dtype=np.float64)
    for value, (left, right) in zip(vector, EDGE_ORDER, strict=True):
        matrix[left, right] = matrix[right, left] = value
    return matrix


def induced_edge_permutation(permutation: Sequence[int]) -> np.ndarray:
    key = tuple(int(value) for value in permutation)
    if key not in VERTEX_PERMUTATIONS:
        raise ValueError("permutation must be one member of S5")
    return INDUCED_EDGE_PERMUTATIONS[VERTEX_PERMUTATIONS.index(key)]


def permute_edges(edges: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    vector = _as_edges(edges)
    return np.asarray(vector[..., induced_edge_permutation(permutation)])


def all_vertex_relabelings(edges: np.ndarray) -> np.ndarray:
    vector = _as_edges(edges)
    if vector.ndim != 1:
        raise ValueError("one edge vector is required")
    return np.asarray(vector[INDUCED_EDGE_PERMUTATIONS])


def edge_rms_size(edges: np.ndarray) -> np.ndarray:
    vector = _as_edges(edges)
    return np.linalg.norm(vector, axis=-1) / SQRT_N_EDGES


def normalize_edges(edges: np.ndarray) -> np.ndarray:
    vector = _as_edges(edges)
    sizes = edge_rms_size(vector)
    if np.any(sizes == 0.0):
        indices = np.argwhere(np.asarray(sizes) == 0.0).reshape(-1).tolist()
        raise DegenerateMetricConfiguration(
            f"DEGENERATE_METRIC_CONFIGURATION at flattened indices {indices}"
        )
    return vector / np.expand_dims(sizes, axis=-1)


def quotient_match_reference(
    edges_a: np.ndarray,
    edges_b: np.ndarray,
    *,
    normalized: bool = False,
) -> QuotientMatch:
    """Slow transparent enumeration of all 120 vertex relabelings."""

    left = _as_edges(edges_a, name="edges_a")
    right = _as_edges(edges_b, name="edges_b")
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("reference matcher requires two single edge vectors")
    if normalized:
        left = normalize_edges(left)
        right = normalize_edges(right)
    values = np.asarray(
        [
            np.linalg.norm(left - right[index]) / SQRT_N_EDGES
            for index in INDUCED_EDGE_PERMUTATIONS
        ]
    )
    winner = int(np.argmin(values))
    return QuotientMatch(
        distance=float(values[winner]),
        permutation_index=winner,
        vertex_permutation=VERTEX_PERMUTATIONS[winner],
    )


def cross_configuration_distances(
    edges_a: np.ndarray,
    edges_b: np.ndarray,
    *,
    mode: Literal["raw", "normalized", "size"] = "raw",
    chunk_size: int = 128,
) -> np.ndarray:
    """Vectorized exact-S5 distances between two banks of configurations."""

    left = _as_edges(edges_a, name="edges_a")
    right = _as_edges(edges_b, name="edges_b")
    if left.ndim == 1:
        left = left[None]
    if right.ndim == 1:
        right = right[None]
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("banks must have shape (configurations,10)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    size_a = edge_rms_size(left)
    size_b = edge_rms_size(right)
    if mode == "size":
        return np.abs(size_a[:, None] - size_b[None, :])
    if mode == "normalized":
        left_work = normalize_edges(left)
        right_work = normalize_edges(right)
    elif mode == "raw":
        left_work = left
        right_work = right
    else:
        raise ValueError("mode must be raw, normalized, or size")
    output = np.empty((len(left_work), len(right_work)), dtype=np.float64)
    norms_b = np.sum(right_work * right_work, axis=1)
    for start_a in range(0, len(left_work), chunk_size):
        stop_a = min(start_a + chunk_size, len(left_work))
        a_chunk = left_work[start_a:stop_a]
        norms_a = np.sum(a_chunk * a_chunk, axis=1)
        for start_b in range(0, len(right_work), chunk_size):
            stop_b = min(start_b + chunk_size, len(right_work))
            b_chunk = right_work[start_b:stop_b]
            # Shape: b x 120 x edge. Vertex matching is exact enumeration;
            # the dot-product identity avoids a four-dimensional diff tensor.
            permuted = b_chunk[:, INDUCED_EDGE_PERMUTATIONS]
            dot = np.einsum("ae,bpe->abp", a_chunk, permuted, optimize=True)
            best_dot = np.max(dot, axis=-1)
            squared = (
                norms_a[:, None]
                + norms_b[start_b:stop_b][None, :]
                - 2.0 * best_dot
            ) / N_EDGES
            scale = np.maximum(
                1.0,
                (norms_a[:, None] + norms_b[start_b:stop_b][None, :])
                / N_EDGES,
            )
            if np.any(squared < -64.0 * np.finfo(np.float64).eps * scale):
                raise FloatingPointError("negative squared quotient distance")
            output[start_a:stop_a, start_b:stop_b] = np.sqrt(
                np.maximum(squared, 0.0)
            )
    return output


def cross_all_configuration_distances(
    edges_a: np.ndarray,
    edges_b: np.ndarray,
    *,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact raw, size, and normalized distances in one S5 pass.

    Scaling either edge vector by its positive configuration size does not
    change which vertex permutation maximizes the dot product.  Consequently
    one exact enumeration supplies both raw and normalized quotient metrics.
    """

    left = _as_edges(edges_a, name="edges_a")
    right = _as_edges(edges_b, name="edges_b")
    if left.ndim == 1:
        left = left[None]
    if right.ndim == 1:
        right = right[None]
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("banks must have shape (configurations,10)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    size_a = edge_rms_size(left)
    size_b = edge_rms_size(right)
    if np.any(size_a == 0.0) or np.any(size_b == 0.0):
        raise DegenerateMetricConfiguration(
            "DEGENERATE_METRIC_CONFIGURATION in cross-distance bank"
        )
    raw = np.empty((len(left), len(right)), dtype=np.float64)
    normalized = np.empty_like(raw)
    norms_a_all = np.sum(left * left, axis=1)
    norms_b_all = np.sum(right * right, axis=1)
    for start_a in range(0, len(left), chunk_size):
        stop_a = min(start_a + chunk_size, len(left))
        a_chunk = left[start_a:stop_a]
        for start_b in range(0, len(right), chunk_size):
            stop_b = min(start_b + chunk_size, len(right))
            b_chunk = right[start_b:stop_b]
            permuted = b_chunk[:, INDUCED_EDGE_PERMUTATIONS]
            dot = np.einsum("ae,bpe->abp", a_chunk, permuted, optimize=True)
            best_dot = np.max(dot, axis=-1)
            norms_a = norms_a_all[start_a:stop_a, None]
            norms_b = norms_b_all[None, start_b:stop_b]
            squared_raw = (norms_a + norms_b - 2.0 * best_dot) / N_EDGES
            scale = np.maximum(1.0, (norms_a + norms_b) / N_EDGES)
            if np.any(squared_raw < -64.0 * np.finfo(np.float64).eps * scale):
                raise FloatingPointError("negative squared quotient distance")
            raw[start_a:stop_a, start_b:stop_b] = np.sqrt(
                np.maximum(squared_raw, 0.0)
            )
            size_product = (
                size_a[start_a:stop_a, None]
                * size_b[None, start_b:stop_b]
            )
            squared_normalized = 2.0 - 2.0 * best_dot / (
                N_EDGES * size_product
            )
            if np.any(squared_normalized < -128.0 * np.finfo(np.float64).eps):
                raise FloatingPointError("negative squared normalized distance")
            normalized[start_a:stop_a, start_b:stop_b] = np.sqrt(
                np.maximum(squared_normalized, 0.0)
            )
    size = np.abs(size_a[:, None] - size_b[None, :])
    return raw, size, normalized


def delta_raw(edges_a: np.ndarray, edges_b: np.ndarray) -> float:
    return float(cross_configuration_distances(edges_a, edges_b, mode="raw")[0, 0])


def delta_norm(edges_a: np.ndarray, edges_b: np.ndarray) -> float:
    return float(
        cross_configuration_distances(edges_a, edges_b, mode="normalized")[0, 0]
    )


def delta_size(edges_a: np.ndarray, edges_b: np.ndarray) -> float:
    return float(cross_configuration_distances(edges_a, edges_b, mode="size")[0, 0])


def raw_size_pattern_identity_residual(
    edges_a: np.ndarray, edges_b: np.ndarray
) -> float:
    size_a = float(edge_rms_size(_as_edges(edges_a)))
    size_b = float(edge_rms_size(_as_edges(edges_b)))
    raw = delta_raw(edges_a, edges_b)
    normalized = delta_norm(edges_a, edges_b)
    return float(
        raw * raw
        - ((size_a - size_b) ** 2 + size_a * size_b * normalized * normalized)
    )


def airm_edge_vector(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 5 or values.shape[1] != values.shape[2]:
        raise ValueError("states must have shape (5,d,d)")
    distances = np.zeros((5, 5), dtype=np.float64)
    for left, right in EDGE_ORDER:
        value = float(airm_distance(values[left], values[right]))
        distances[left, right] = distances[right, left] = value
    return edge_vector(distances)


def congruence_transform(states: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    action = np.asarray(transform, dtype=np.float64)
    if values.ndim != 3 or action.shape != values.shape[1:]:
        raise ValueError("state/action shapes are incompatible")
    if not np.isfinite(action).all() or np.linalg.matrix_rank(action) != len(action):
        raise ValueError("transform must be finite and invertible")
    return symmetrize(np.einsum("ij,kjl,ml->kim", action, values, action))


def inversion_counterexample(dimension: int = 22) -> CounterexampleFixture:
    """Identical AIRM distance geometry without one orthogonal conjugation."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    exponents = np.asarray([0.0, 0.2, 0.55, 0.95, 1.35])
    identity = np.eye(dimension)
    states_a = np.asarray([np.exp(value) * identity for value in exponents])
    states_b = np.asarray([np.exp(-value) * identity for value in exponents])
    edges_a = airm_edge_vector(states_a)
    edges_b = airm_edge_vector(states_b)
    # Orthogonal conjugation leaves every scalar SPD matrix unchanged. Search
    # all vertex relabelings to prove the two scalar spectra cannot coincide.
    mismatch = min(
        max(
            float(np.linalg.norm(states_b[index] - states_a[permutation[index]]))
            for index in range(5)
        )
        for permutation in VERTEX_PERMUTATIONS
    )
    return CounterexampleFixture(
        states_a=states_a,
        states_b=states_b,
        edges_a=edges_a,
        edges_b=edges_b,
        quotient_distance=delta_raw(edges_a, edges_b),
        minimum_orthogonal_scalar_mismatch=float(mismatch),
    )


__all__ = [
    "CounterexampleFixture",
    "DEFAULT_MATCH_ATOL",
    "DEFAULT_MATCH_RTOL",
    "DegenerateMetricConfiguration",
    "EDGE_NAMES",
    "EDGE_ORDER",
    "INDUCED_EDGE_PERMUTATIONS",
    "N_EDGES",
    "N_VERTICES",
    "QuotientMatch",
    "VERTEX_PERMUTATIONS",
    "airm_edge_vector",
    "all_vertex_relabelings",
    "congruence_transform",
    "cross_configuration_distances",
    "cross_all_configuration_distances",
    "delta_norm",
    "delta_raw",
    "delta_size",
    "distance_matrix_from_edges",
    "edge_rms_size",
    "edge_vector",
    "induced_edge_permutation",
    "inversion_counterexample",
    "normalize_edges",
    "permute_edges",
    "quotient_match_reference",
    "raw_size_pattern_identity_residual",
]
