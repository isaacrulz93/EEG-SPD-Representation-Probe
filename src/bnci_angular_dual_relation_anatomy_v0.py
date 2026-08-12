"""Frozen-matrix BNCI angular six-pair and dual relation anatomy.

The module accepts only an already-frozen 36 x 36 squared angular cost matrix.
It has no EEG, covariance-mean, anti-development, movement fitting, or quotient
optimization dependency.
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
PAIR_NAMES = tuple(CLASS_SHORT[a] + CLASS_SHORT[b] for a, b in PAIR_INDICES)
SUBJECT_PAIR_NAMES = tuple(
    f"S{a + 1}-S{b + 1}" for a, b in combinations(range(N_SUBJECTS), 2)
)


@dataclass(frozen=True)
class RelationStatistics:
    a_sc: np.ndarray
    b_sc: np.ndarray
    c_sc: np.ndarray
    d_sc: np.ndarray
    s_sc: np.ndarray
    class_sc: np.ndarray
    j_sc: np.ndarray
    s_s: np.ndarray
    class_s: np.ndarray
    j_s: np.ndarray
    t_subject: float
    t_class: float
    t_j: float


@dataclass(frozen=True)
class DualAnatomy:
    symmetric_a: np.ndarray
    g: np.ndarray
    delta_g: np.ndarray
    h: np.ndarray
    delta_h: np.ndarray
    g_profiles: np.ndarray
    delta_g_profiles: np.ndarray
    h_profiles: np.ndarray
    delta_h_profiles: np.ndarray


def canonical_subjects(*, n_classes: int = N_CLASSES) -> np.ndarray:
    return np.repeat(np.arange(1, N_SUBJECTS + 1, dtype=np.int64), n_classes)


def canonical_classes(class_order: tuple[str, ...] = CLASS_ORDER) -> np.ndarray:
    return np.tile(np.asarray(class_order), N_SUBJECTS)


def cell_index(subject: int, class_index: int, *, n_classes: int) -> int:
    if not 0 <= subject < N_SUBJECTS:
        raise ValueError("subject index outside 0..8")
    if not 0 <= class_index < n_classes:
        raise ValueError("class index outside supplied class count")
    return subject * n_classes + class_index


def validate_parent_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (N_CELLS, N_CELLS) or not np.isfinite(values).all():
        raise ValueError("frozen parent angular matrix must be finite 36x36")
    return values


def relation_statistics(
    matrix: np.ndarray, *, n_classes: int, n_subjects: int = N_SUBJECTS
) -> RelationStatistics:
    """Exact frozen mean-aggregated a/b/c/d and S/C/J arithmetic."""

    if n_subjects != N_SUBJECTS or n_classes < 2:
        raise ValueError("V0 requires nine subjects and at least two classes")
    values = np.asarray(matrix, dtype=np.float64)
    n_cells = n_subjects * n_classes
    if values.shape != (n_cells, n_cells) or not np.isfinite(values).all():
        raise ValueError(f"cost matrix must be finite {n_cells}x{n_cells}")
    a = np.empty((n_subjects, n_classes), dtype=np.float64)
    b = np.empty_like(a)
    c = np.empty_like(a)
    d = np.empty_like(a)
    for subject in range(n_subjects):
        for class_index in range(n_classes):
            anchor = cell_index(subject, class_index, n_classes=n_classes)
            a[subject, class_index] = values[anchor, anchor]
            b[subject, class_index] = np.mean(
                [
                    values[
                        anchor,
                        cell_index(subject, other_class, n_classes=n_classes),
                    ]
                    for other_class in range(n_classes)
                    if other_class != class_index
                ]
            )
            c[subject, class_index] = np.mean(
                [
                    values[
                        anchor,
                        cell_index(other_subject, class_index, n_classes=n_classes),
                    ]
                    for other_subject in range(n_subjects)
                    if other_subject != subject
                ]
            )
            d[subject, class_index] = np.mean(
                [
                    values[
                        anchor,
                        cell_index(other_subject, other_class, n_classes=n_classes),
                    ]
                    for other_subject in range(n_subjects)
                    if other_subject != subject
                    for other_class in range(n_classes)
                    if other_class != class_index
                ]
            )
    s_sc = c - a
    class_sc = b - a
    j_sc = b + c - a - d
    s_s = np.mean(s_sc, axis=1)
    class_s = np.mean(class_sc, axis=1)
    j_s = np.mean(j_sc, axis=1)
    return RelationStatistics(
        a_sc=a,
        b_sc=b,
        c_sc=c,
        d_sc=d,
        s_sc=s_sc,
        class_sc=class_sc,
        j_sc=j_sc,
        s_s=s_s,
        class_s=class_s,
        j_s=j_s,
        t_subject=float(np.mean(s_s)),
        t_class=float(np.mean(class_s)),
        t_j=float(np.mean(j_s)),
    )


def class_subset_indices(pair: tuple[int, int]) -> np.ndarray:
    selected = tuple(int(value) for value in pair)
    if len(selected) != 2 or selected[0] >= selected[1]:
        raise ValueError("pair must contain two increasing frozen class indices")
    if selected not in PAIR_INDICES:
        raise ValueError("pair is not one of the six frozen class pairs")
    return np.asarray(
        [
            cell_index(subject, class_index, n_classes=N_CLASSES)
            for subject in range(N_SUBJECTS)
            for class_index in selected
        ],
        dtype=np.int64,
    )


def extract_binary_matrix(
    matrix: np.ndarray, pair: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    values = validate_parent_matrix(matrix)
    indices = class_subset_indices(pair)
    return values[np.ix_(indices, indices)].copy(), indices


def six_pair_statistics(
    matrix: np.ndarray,
) -> tuple[dict[str, RelationStatistics], dict[str, np.ndarray]]:
    statistics: dict[str, RelationStatistics] = {}
    indices: dict[str, np.ndarray] = {}
    for name, pair in zip(PAIR_NAMES, PAIR_INDICES, strict=True):
        binary, selected = extract_binary_matrix(matrix, pair)
        statistics[name] = relation_statistics(binary, n_classes=2)
        indices[name] = selected
    return statistics, indices


def reconstruction_errors(
    full: RelationStatistics, pairs: dict[str, RelationStatistics]
) -> dict[str, np.ndarray | float]:
    ordered = [pairs[name] for name in PAIR_NAMES]
    return {
        "S_s": full.s_s - np.mean([value.s_s for value in ordered], axis=0),
        "C_s": full.class_s
        - np.mean([value.class_s for value in ordered], axis=0),
        "J_s": full.j_s - np.mean([value.j_s for value in ordered], axis=0),
        "T_S": full.t_subject
        - float(np.mean([value.t_subject for value in ordered])),
        "T_C": full.t_class - float(np.mean([value.t_class for value in ordered])),
        "T_J": full.t_j - float(np.mean([value.t_j for value in ordered])),
    }


def maximum_reconstruction_error(errors: dict[str, np.ndarray | float]) -> float:
    return max(float(np.max(np.abs(value))) for value in errors.values())


def upper_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    return values[np.triu_indices(values.shape[0], k=1)]


def baseline_adjust(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    diagonal = np.diag(values)
    return values - 0.5 * (diagonal[:, None] + diagonal[None, :])


def build_dual_anatomy(matrix: np.ndarray) -> DualAnatomy:
    values = validate_parent_matrix(matrix)
    symmetric_a = 0.5 * (values + values.T)
    g = np.empty((N_SUBJECTS, N_CLASSES, N_CLASSES), dtype=np.float64)
    for subject in range(N_SUBJECTS):
        indices = [cell_index(subject, c, n_classes=N_CLASSES) for c in range(N_CLASSES)]
        g[subject] = symmetric_a[np.ix_(indices, indices)]
    delta_g = np.stack([baseline_adjust(value) for value in g])
    h = np.empty((N_CLASSES, N_SUBJECTS, N_SUBJECTS), dtype=np.float64)
    for class_index in range(N_CLASSES):
        indices = [
            cell_index(subject, class_index, n_classes=N_CLASSES)
            for subject in range(N_SUBJECTS)
        ]
        h[class_index] = symmetric_a[np.ix_(indices, indices)]
    delta_h = np.stack([baseline_adjust(value) for value in h])
    return DualAnatomy(
        symmetric_a=symmetric_a,
        g=g,
        delta_g=delta_g,
        h=h,
        delta_h=delta_h,
        g_profiles=np.stack([upper_off_diagonal(value) for value in g]),
        delta_g_profiles=np.stack(
            [upper_off_diagonal(value) for value in delta_g]
        ),
        h_profiles=np.stack([upper_off_diagonal(value) for value in h]),
        delta_h_profiles=np.stack(
            [upper_off_diagonal(value) for value in delta_h]
        ),
    )


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.ravel(np.asarray(first, dtype=np.float64))
    right = np.ravel(np.asarray(second, dtype=np.float64))
    if left.shape != right.shape:
        raise ValueError("profiles must have equal shapes")
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def centered_cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
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


def pairwise_profile_commonality(
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("profile bank must be two-dimensional")
    count = values.shape[0]
    correlations = np.eye(count, dtype=np.float64)
    cosines = np.eye(count, dtype=np.float64)
    distances = np.zeros((count, count), dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            correlations[first, second] = correlations[second, first] = pearson_correlation(
                values[first], values[second]
            )
            cosines[first, second] = cosines[second, first] = centered_cosine_similarity(
                values[first], values[second]
            )
            distances[first, second] = distances[second, first] = float(
                np.linalg.norm(values[first] - values[second])
            )
    return correlations, cosines, distances


def leave_one_out_commonality(
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("at least two profiles are required")
    correlations = []
    cosines = []
    for index in range(values.shape[0]):
        reference = np.mean(np.delete(values, index, axis=0), axis=0)
        correlations.append(pearson_correlation(values[index], reference))
        cosines.append(centered_cosine_similarity(values[index], reference))
    return np.asarray(correlations), np.asarray(cosines)


def coarse_effector_boundary_contrast(pair_t_j: dict[str, float]) -> float:
    cross = np.mean([pair_t_j[name] for name in ("LF", "LT", "RF", "RT")])
    within = np.mean([pair_t_j[name] for name in ("LR", "FT")])
    return float(cross - within)


def leave_one_subject_influence(j_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(j_s, dtype=np.float64)
    if values.shape != (N_SUBJECTS,):
        raise ValueError("j_s must have nine subject values")
    full = float(np.mean(values))
    leave_one_out = np.asarray(
        [float(np.mean(np.delete(values, index))) for index in range(N_SUBJECTS)]
    )
    return leave_one_out, full - leave_one_out


__all__ = [
    "CLASS_ORDER",
    "CLASS_SHORT",
    "N_CELLS",
    "N_CLASSES",
    "N_SUBJECTS",
    "PAIR_INDICES",
    "PAIR_NAMES",
    "SUBJECT_PAIR_NAMES",
    "DualAnatomy",
    "RelationStatistics",
    "baseline_adjust",
    "build_dual_anatomy",
    "canonical_classes",
    "canonical_subjects",
    "centered_cosine_similarity",
    "class_subset_indices",
    "coarse_effector_boundary_contrast",
    "extract_binary_matrix",
    "leave_one_out_commonality",
    "leave_one_subject_influence",
    "maximum_reconstruction_error",
    "pairwise_profile_commonality",
    "pearson_correlation",
    "reconstruction_errors",
    "relation_statistics",
    "six_pair_statistics",
    "upper_off_diagonal",
    "validate_parent_matrix",
]
