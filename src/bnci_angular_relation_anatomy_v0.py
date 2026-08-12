"""Retrospective relation anatomy of the frozen BNCI angular cost matrix.

This module accepts only a 36 x 36 squared ``c_ang`` matrix.  It contains no
EEG, covariance, anti-development, movement-construction, or quotient-fitting
code.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


N_SUBJECTS = 9
CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
CLASS_SHORT = ("L", "R", "F", "T")
N_CLASSES = len(CLASS_ORDER)
N_CELLS = N_SUBJECTS * N_CLASSES
PAIR_INDICES = tuple(combinations(range(N_CLASSES), 2))
PAIR_NAMES = tuple(CLASS_SHORT[i] + CLASS_SHORT[j] for i, j in PAIR_INDICES)
SUBJECT_PAIR_NAMES = tuple(
    f"S{s + 1}-S{t + 1}" for s, t in combinations(range(N_SUBJECTS), 2)
)


@dataclass(frozen=True)
class RelationStatistics:
    class_indices: tuple[int, ...]
    a_sc: np.ndarray
    b_sc: np.ndarray
    c_sc: np.ndarray
    d_sc: np.ndarray
    s_sc: np.ndarray
    c_sc_effect: np.ndarray
    j_sc: np.ndarray
    s_s: np.ndarray
    c_s: np.ndarray
    j_s: np.ndarray
    t_subject: float
    t_class: float
    t_j: float


@dataclass(frozen=True)
class RelationAnatomy:
    g: np.ndarray
    delta_g: np.ndarray
    h: np.ndarray
    delta_h: np.ndarray
    g_profiles: np.ndarray
    delta_g_profiles: np.ndarray
    h_profiles: np.ndarray
    delta_h_profiles: np.ndarray


def canonical_cell_subjects() -> np.ndarray:
    return np.repeat(np.arange(1, N_SUBJECTS + 1, dtype=np.int64), N_CLASSES)


def canonical_cell_classes() -> np.ndarray:
    return np.tile(np.asarray(CLASS_ORDER), N_SUBJECTS)


def cell_index(subject_index: int, class_index: int) -> int:
    if not 0 <= subject_index < N_SUBJECTS:
        raise ValueError("subject index must be in 0..8")
    if not 0 <= class_index < N_CLASSES:
        raise ValueError("class index must be in 0..3")
    return subject_index * N_CLASSES + class_index


def validate_cost_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (N_CELLS, N_CELLS):
        raise ValueError("frozen angular cost matrix must have shape (36, 36)")
    if not np.isfinite(values).all():
        raise ValueError("frozen angular cost matrix must be finite")
    return values


def relation_statistics(
    matrix: np.ndarray, class_indices: tuple[int, ...] = tuple(range(N_CLASSES))
) -> RelationStatistics:
    """Apply the frozen relation-cell definitions to an explicit class subset."""

    values = validate_cost_matrix(matrix)
    selected = tuple(int(index) for index in class_indices)
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise ValueError("class_indices must contain at least two unique classes")
    if any(index < 0 or index >= N_CLASSES for index in selected):
        raise ValueError("class index outside frozen four-class order")
    k_classes = len(selected)
    a = np.empty((N_SUBJECTS, k_classes), dtype=np.float64)
    b = np.empty_like(a)
    c = np.empty_like(a)
    d = np.empty_like(a)
    for subject in range(N_SUBJECTS):
        for local_class, class_index in enumerate(selected):
            anchor = cell_index(subject, class_index)
            a[subject, local_class] = values[anchor, anchor]
            b[subject, local_class] = np.mean(
                [
                    values[anchor, cell_index(subject, other_class)]
                    for other_class in selected
                    if other_class != class_index
                ]
            )
            c[subject, local_class] = np.mean(
                [
                    values[anchor, cell_index(other_subject, class_index)]
                    for other_subject in range(N_SUBJECTS)
                    if other_subject != subject
                ]
            )
            d[subject, local_class] = np.mean(
                [
                    values[anchor, cell_index(other_subject, other_class)]
                    for other_subject in range(N_SUBJECTS)
                    if other_subject != subject
                    for other_class in selected
                    if other_class != class_index
                ]
            )
    s_sc = c - a
    c_effect = b - a
    j_sc = b + c - a - d
    s_s = np.mean(s_sc, axis=1)
    c_s = np.mean(c_effect, axis=1)
    j_s = np.mean(j_sc, axis=1)
    return RelationStatistics(
        class_indices=selected,
        a_sc=a,
        b_sc=b,
        c_sc=c,
        d_sc=d,
        s_sc=s_sc,
        c_sc_effect=c_effect,
        j_sc=j_sc,
        s_s=s_s,
        c_s=c_s,
        j_s=j_s,
        t_subject=float(np.mean(s_s)),
        t_class=float(np.mean(c_s)),
        t_j=float(np.mean(j_s)),
    )


def six_pair_statistics(matrix: np.ndarray) -> dict[str, RelationStatistics]:
    return {
        name: relation_statistics(matrix, indices)
        for name, indices in zip(PAIR_NAMES, PAIR_INDICES, strict=True)
    }


def reconstruction_errors(
    full: RelationStatistics, pairs: dict[str, RelationStatistics]
) -> dict[str, np.ndarray | float]:
    """Return signed full-minus-six-pair-mean reconstruction residuals."""

    ordered = [pairs[name] for name in PAIR_NAMES]
    return {
        "S_s": full.s_s - np.mean([item.s_s for item in ordered], axis=0),
        "C_s": full.c_s - np.mean([item.c_s for item in ordered], axis=0),
        "J_s": full.j_s - np.mean([item.j_s for item in ordered], axis=0),
        "T_S": full.t_subject - float(np.mean([item.t_subject for item in ordered])),
        "T_C": full.t_class - float(np.mean([item.t_class for item in ordered])),
        "T_J": full.t_j - float(np.mean([item.t_j for item in ordered])),
    }


def maximum_reconstruction_error(errors: dict[str, np.ndarray | float]) -> float:
    return max(float(np.max(np.abs(value))) for value in errors.values())


def upper_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    indices = np.triu_indices(values.shape[0], k=1)
    return values[indices]


def baseline_adjust(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    diagonal = np.diag(values)
    return values - 0.5 * (diagonal[:, None] + diagonal[None, :])


def build_relation_anatomy(matrix: np.ndarray) -> RelationAnatomy:
    values = validate_cost_matrix(matrix)
    g = np.empty((N_SUBJECTS, N_CLASSES, N_CLASSES), dtype=np.float64)
    for subject in range(N_SUBJECTS):
        for first in range(N_CLASSES):
            for second in range(N_CLASSES):
                forward = values[
                    cell_index(subject, first), cell_index(subject, second)
                ]
                reverse = values[
                    cell_index(subject, second), cell_index(subject, first)
                ]
                g[subject, first, second] = 0.5 * (forward + reverse)
    delta_g = np.stack([baseline_adjust(item) for item in g])

    h = np.empty((N_CLASSES, N_SUBJECTS, N_SUBJECTS), dtype=np.float64)
    for class_index in range(N_CLASSES):
        for first in range(N_SUBJECTS):
            for second in range(N_SUBJECTS):
                forward = values[
                    cell_index(first, class_index), cell_index(second, class_index)
                ]
                reverse = values[
                    cell_index(second, class_index), cell_index(first, class_index)
                ]
                h[class_index, first, second] = 0.5 * (forward + reverse)
    delta_h = np.stack([baseline_adjust(item) for item in h])
    return RelationAnatomy(
        g=g,
        delta_g=delta_g,
        h=h,
        delta_h=delta_h,
        g_profiles=np.stack([upper_off_diagonal(item) for item in g]),
        delta_g_profiles=np.stack([upper_off_diagonal(item) for item in delta_g]),
        h_profiles=np.stack([upper_off_diagonal(item) for item in h]),
        delta_h_profiles=np.stack([upper_off_diagonal(item) for item in delta_h]),
    )


def centered_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.ravel(np.asarray(first, dtype=np.float64))
    right = np.ravel(np.asarray(second, dtype=np.float64))
    if left.shape != right.shape:
        raise ValueError("profiles must have equal shapes")
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def pairwise_profile_similarity(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("profiles must be a two-dimensional bank")
    count = values.shape[0]
    distances = np.zeros((count, count), dtype=np.float64)
    correlations = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            distance = float(np.linalg.norm(values[first] - values[second]))
            correlation = centered_correlation(values[first], values[second])
            distances[first, second] = distances[second, first] = distance
            correlations[first, second] = correlations[second, first] = correlation
    return distances, correlations


def leave_one_out_commonality(profiles: np.ndarray) -> np.ndarray:
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("at least two profiles are required")
    return np.asarray(
        [
            centered_correlation(
                values[index], np.mean(np.delete(values, index, axis=0), axis=0)
            )
            for index in range(values.shape[0])
        ],
        dtype=np.float64,
    )


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return values[np.triu_indices(values.shape[0], k=1)]


__all__ = [
    "CLASS_ORDER",
    "CLASS_SHORT",
    "N_CELLS",
    "N_CLASSES",
    "N_SUBJECTS",
    "PAIR_INDICES",
    "PAIR_NAMES",
    "SUBJECT_PAIR_NAMES",
    "RelationAnatomy",
    "RelationStatistics",
    "baseline_adjust",
    "build_relation_anatomy",
    "canonical_cell_classes",
    "canonical_cell_subjects",
    "cell_index",
    "centered_correlation",
    "leave_one_out_commonality",
    "maximum_reconstruction_error",
    "off_diagonal_values",
    "pairwise_profile_similarity",
    "reconstruction_errors",
    "relation_statistics",
    "six_pair_statistics",
    "upper_off_diagonal",
    "validate_cost_matrix",
]
