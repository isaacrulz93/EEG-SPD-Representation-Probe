"""Frozen cell-level interaction statistics for local metric geometry V0.

Trial pairs are reduced to 36 x 36 cross-session cell summaries before any
contrast is formed.  Subjects, not trials, pairs, or cells, are the group unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


N_SUBJECTS = 9
N_CLASSES = 4
N_CELLS = N_SUBJECTS * N_CLASSES
DEFAULT_MASTER_SEED = 20260810
DEFAULT_NULL_REPLICATES = 1999
CLASSBREAK_STREAM_TAG = 1101
SUBJECTBREAK_STREAM_TAG = 1102
CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")


@dataclass(frozen=True)
class InteractionContrasts:
    j_sc: np.ndarray
    s_sc: np.ndarray
    c_sc: np.ndarray
    j_s: np.ndarray
    s_s: np.ndarray
    c_s: np.ndarray
    t_j: float
    t_s: float
    t_c: float


@dataclass(frozen=True)
class InteractionNullResult:
    observed: InteractionContrasts
    classbreak_t_j: np.ndarray
    subjectbreak_t_j: np.ndarray
    classbreak_t_c: np.ndarray
    subjectbreak_t_s: np.ndarray
    p_j_classbreak: float
    p_j_subjectbreak: float
    p_j: float
    p_c_classbreak: float
    p_s_subjectbreak: float


def cell_index(subject_index: int, class_index: int) -> int:
    if not 0 <= subject_index < N_SUBJECTS:
        raise ValueError("subject index outside frozen 0..8 range")
    if not 0 <= class_index < N_CLASSES:
        raise ValueError("class index outside frozen 0..3 range")
    return subject_index * N_CLASSES + class_index


def cell_labels() -> tuple[tuple[int, int], ...]:
    return tuple(
        (subject, class_index)
        for subject in range(N_SUBJECTS)
        for class_index in range(N_CLASSES)
    )


def symmetrize_session_roles(m01: np.ndarray) -> np.ndarray:
    matrix = np.asarray(m01, dtype=np.float64)
    if matrix.shape != (N_CELLS, N_CELLS):
        raise ValueError(f"M01 must have shape ({N_CELLS},{N_CELLS})")
    if not np.isfinite(matrix).all():
        raise ValueError("M01 must be finite")
    return 0.5 * (matrix + matrix.T)


def interaction_contrasts(matrix: np.ndarray) -> InteractionContrasts:
    """Compute frozen J, subject-specificity S, and class-specificity C."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (N_CELLS, N_CELLS):
        raise ValueError(f"matrix must have shape ({N_CELLS},{N_CELLS})")
    if not np.isfinite(values).all():
        raise ValueError("matrix must be finite")
    if not np.allclose(values, values.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("primary structural matrix must be session-role symmetric")
    j_sc = np.empty((N_SUBJECTS, N_CLASSES), dtype=np.float64)
    s_sc = np.empty_like(j_sc)
    c_sc = np.empty_like(j_sc)
    for subject in range(N_SUBJECTS):
        for class_index in range(N_CLASSES):
            anchor = cell_index(subject, class_index)
            a_sc = values[anchor, anchor]
            b_sc = float(
                np.mean(
                    [
                        values[anchor, cell_index(subject, other_class)]
                        for other_class in range(N_CLASSES)
                        if other_class != class_index
                    ]
                )
            )
            c_value = float(
                np.mean(
                    [
                        values[anchor, cell_index(other_subject, class_index)]
                        for other_subject in range(N_SUBJECTS)
                        if other_subject != subject
                    ]
                )
            )
            d_sc = float(
                np.mean(
                    [
                        values[anchor, cell_index(other_subject, other_class)]
                        for other_subject in range(N_SUBJECTS)
                        if other_subject != subject
                        for other_class in range(N_CLASSES)
                        if other_class != class_index
                    ]
                )
            )
            j_sc[subject, class_index] = b_sc + c_value - a_sc - d_sc
            s_sc[subject, class_index] = c_value - a_sc
            c_sc[subject, class_index] = b_sc - a_sc
    j_s = np.mean(j_sc, axis=1)
    s_s = np.mean(s_sc, axis=1)
    c_s = np.mean(c_sc, axis=1)
    return InteractionContrasts(
        j_sc=j_sc,
        s_sc=s_sc,
        c_sc=c_sc,
        j_s=j_s,
        s_s=s_s,
        c_s=c_s,
        t_j=float(np.mean(j_s)),
        t_s=float(np.mean(s_s)),
        t_c=float(np.mean(c_s)),
    )


def _rng(master_seed: int, stream_tag: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master_seed, stream_tag]))


def classbreak_mappings(
    *,
    replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> np.ndarray:
    """Map each assigned session-1 class cell to one permuted original cell."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    generator = _rng(master_seed, CLASSBREAK_STREAM_TAG)
    mappings = np.empty((replicates, N_CELLS), dtype=np.int64)
    for replicate in range(replicates):
        for subject in range(N_SUBJECTS):
            permutation = generator.permutation(N_CLASSES)
            for assigned_class in range(N_CLASSES):
                mappings[replicate, cell_index(subject, assigned_class)] = cell_index(
                    subject, int(permutation[assigned_class])
                )
    return mappings


def subjectbreak_mappings(
    *,
    replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> np.ndarray:
    """Map each assigned session-1 subject cell within class to another cell."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    generator = _rng(master_seed, SUBJECTBREAK_STREAM_TAG)
    mappings = np.empty((replicates, N_CELLS), dtype=np.int64)
    for replicate in range(replicates):
        for class_index in range(N_CLASSES):
            permutation = generator.permutation(N_SUBJECTS)
            for assigned_subject in range(N_SUBJECTS):
                mappings[replicate, cell_index(assigned_subject, class_index)] = (
                    cell_index(int(permutation[assigned_subject]), class_index)
                )
    return mappings


def mapped_symmetrized_matrix(m01: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    matrix = np.asarray(m01, dtype=np.float64)
    indices = np.asarray(mapping, dtype=np.int64)
    if matrix.shape != (N_CELLS, N_CELLS):
        raise ValueError("M01 has the wrong shape")
    if indices.shape != (N_CELLS,) or not np.array_equal(
        np.sort(indices), np.arange(N_CELLS)
    ):
        raise ValueError("mapping must be a permutation of the 36 cells")
    mapped = matrix[:, indices]
    return symmetrize_session_roles(mapped)


def plus_one_pvalue(observed: float, null: np.ndarray) -> float:
    values = np.asarray(null, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("null must be a finite nonempty vector")
    return float((1 + np.count_nonzero(values >= observed)) / (1 + len(values)))


def evaluate_interaction_nulls(
    m01: np.ndarray,
    *,
    class_mappings: np.ndarray | None = None,
    subject_mappings: np.ndarray | None = None,
    replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> InteractionNullResult:
    """Evaluate primary and supporting tests with the two frozen null families."""

    class_maps = (
        classbreak_mappings(replicates=replicates, master_seed=master_seed)
        if class_mappings is None
        else np.asarray(class_mappings, dtype=np.int64)
    )
    subject_maps = (
        subjectbreak_mappings(replicates=replicates, master_seed=master_seed)
        if subject_mappings is None
        else np.asarray(subject_mappings, dtype=np.int64)
    )
    if class_maps.shape != (replicates, N_CELLS):
        raise ValueError("class mappings do not match the frozen grid")
    if subject_maps.shape != (replicates, N_CELLS):
        raise ValueError("subject mappings do not match the frozen grid")
    observed = interaction_contrasts(symmetrize_session_roles(m01))
    classbreak_t_j = np.empty(replicates, dtype=np.float64)
    subjectbreak_t_j = np.empty(replicates, dtype=np.float64)
    classbreak_t_c = np.empty(replicates, dtype=np.float64)
    subjectbreak_t_s = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        class_result = interaction_contrasts(
            mapped_symmetrized_matrix(m01, class_maps[replicate])
        )
        subject_result = interaction_contrasts(
            mapped_symmetrized_matrix(m01, subject_maps[replicate])
        )
        classbreak_t_j[replicate] = class_result.t_j
        classbreak_t_c[replicate] = class_result.t_c
        subjectbreak_t_j[replicate] = subject_result.t_j
        subjectbreak_t_s[replicate] = subject_result.t_s
    p_j_classbreak = plus_one_pvalue(observed.t_j, classbreak_t_j)
    p_j_subjectbreak = plus_one_pvalue(observed.t_j, subjectbreak_t_j)
    return InteractionNullResult(
        observed=observed,
        classbreak_t_j=classbreak_t_j,
        subjectbreak_t_j=subjectbreak_t_j,
        classbreak_t_c=classbreak_t_c,
        subjectbreak_t_s=subjectbreak_t_s,
        p_j_classbreak=p_j_classbreak,
        p_j_subjectbreak=p_j_subjectbreak,
        p_j=max(p_j_classbreak, p_j_subjectbreak),
        p_c_classbreak=plus_one_pvalue(observed.t_c, classbreak_t_c),
        p_s_subjectbreak=plus_one_pvalue(observed.t_s, subjectbreak_t_s),
    )


def synthetic_additive_cell_matrix(
    *,
    subject_effect: float = 0.0,
    class_effect: float = 0.0,
    interaction_effect: float = 0.0,
    baseline: float = 2.0,
) -> np.ndarray:
    """Known-answer symmetric M01 fixture with an optional same-cell bonus."""

    if min(subject_effect, class_effect, interaction_effect, baseline) < 0.0:
        raise ValueError("fixture magnitudes must be nonnegative")
    matrix = np.empty((N_CELLS, N_CELLS), dtype=np.float64)
    labels = cell_labels()
    for row, (subject, class_index) in enumerate(labels):
        for column, (other_subject, other_class) in enumerate(labels):
            value = baseline
            value += subject_effect * (subject != other_subject)
            value += class_effect * (class_index != other_class)
            value -= interaction_effect * (
                subject == other_subject and class_index == other_class
            )
            matrix[row, column] = value
    if np.any(matrix < 0.0):
        raise ValueError("fixture contains a negative cell distance")
    return matrix


def mechanism_tag(
    *,
    size_t_j: float,
    size_p_classbreak: float,
    size_p_subjectbreak: float,
    normalized_t_j: float,
    normalized_p_classbreak: float,
    normalized_p_subjectbreak: float,
    alpha: float = 0.05,
) -> str:
    size_supported = (
        size_t_j > 0.0
        and size_p_classbreak < alpha
        and size_p_subjectbreak < alpha
    )
    normalized_supported = (
        normalized_t_j > 0.0
        and normalized_p_classbreak < alpha
        and normalized_p_subjectbreak < alpha
    )
    if size_supported and normalized_supported:
        return "BOTH_SIZE_AND_RELATIVE_PATTERN_SUPPORTED"
    if normalized_supported:
        return "RELATIVE_PATTERN_SUPPORTED"
    if size_supported:
        return "METRIC_SIZE_SUPPORTED_RELATIVE_PATTERN_NOT_ESTABLISHED"
    return "MECHANISM_UNRESOLVED"


def terminal_decision(
    *,
    t_j: float,
    p_j_classbreak: float,
    p_j_subjectbreak: float,
    alpha: float = 0.05,
) -> str:
    if t_j > 0.0 and p_j_classbreak < alpha and p_j_subjectbreak < alpha:
        return "GO_STABLE_SUBJECT_CLASS_LOCAL_METRIC_INTERACTION"
    return "STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0"


def keyed_scores(values: Sequence[float]) -> Mapping[int, float]:
    """Small reporting helper that rejects silent subject dropping."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (N_SUBJECTS,) or not np.isfinite(array).all():
        raise ValueError("exactly nine finite subject scores are required")
    return {subject + 1: float(array[subject]) for subject in range(N_SUBJECTS)}


__all__ = [
    "CLASSBREAK_STREAM_TAG",
    "CLASS_ORDER",
    "DEFAULT_MASTER_SEED",
    "DEFAULT_NULL_REPLICATES",
    "InteractionContrasts",
    "InteractionNullResult",
    "N_CELLS",
    "N_CLASSES",
    "N_SUBJECTS",
    "SUBJECTBREAK_STREAM_TAG",
    "cell_index",
    "cell_labels",
    "classbreak_mappings",
    "evaluate_interaction_nulls",
    "interaction_contrasts",
    "keyed_scores",
    "mapped_symmetrized_matrix",
    "mechanism_tag",
    "plus_one_pvalue",
    "subjectbreak_mappings",
    "symmetrize_session_roles",
    "synthetic_additive_cell_matrix",
    "terminal_decision",
]
