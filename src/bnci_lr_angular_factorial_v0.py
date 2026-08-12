"""Frozen-artifact BNCI Left/Right-only angular factorial diagnostic V0.

No EEG, covariance mean, anti-development, or quotient fitting is available
from this module.  It only extracts frozen squared-cost cells, evaluates the
existing relation-cell arithmetic for a supplied K, and performs fixed tuple
relabeling nulls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N_SUBJECTS = 9
PARENT_CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
LR_CLASS_ORDER = ("left_hand", "right_hand")
NULL_REPLICATES = 1999
MASTER_SEED = 20260810
SUBJECT_STREAM_TAG = 1102
CLASS_STREAM_TAG = 1101
ALPHA = 0.05


@dataclass(frozen=True)
class FactorialStatistics:
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
class FactorialInference:
    observed: FactorialStatistics
    subjectbreak_t_subject: np.ndarray
    subjectbreak_t_j: np.ndarray
    classbreak_t_class: np.ndarray
    classbreak_t_j: np.ndarray
    subject_mappings: np.ndarray
    class_mappings: np.ndarray
    p_subject: float
    p_class: float
    p_j_subjectbreak: float
    p_j_classbreak: float


def cell_index(subject_index: int, class_index: int, *, n_classes: int) -> int:
    if not 0 <= subject_index < N_SUBJECTS:
        raise ValueError("subject index must be in 0..8")
    if not 0 <= class_index < n_classes:
        raise ValueError("class index outside supplied class count")
    return subject_index * n_classes + class_index


def canonical_subjects(*, n_classes: int) -> np.ndarray:
    return np.repeat(np.arange(1, N_SUBJECTS + 1, dtype=np.int64), n_classes)


def canonical_classes(class_order: tuple[str, ...]) -> np.ndarray:
    return np.tile(np.asarray(class_order), N_SUBJECTS)


def validate_canonical_parent_order(subjects: np.ndarray, classes: np.ndarray) -> None:
    if not np.array_equal(np.asarray(subjects), canonical_subjects(n_classes=4)):
        raise ValueError("parent subject ordering is not canonical")
    if not np.array_equal(np.asarray(classes), canonical_classes(PARENT_CLASS_ORDER)):
        raise ValueError("parent class ordering is not canonical")


def lr_parent_indices(subjects: np.ndarray, classes: np.ndarray) -> np.ndarray:
    validate_canonical_parent_order(subjects, classes)
    selected = np.flatnonzero(np.isin(np.asarray(classes), np.asarray(LR_CLASS_ORDER)))
    expected = np.asarray(
        [4 * subject + class_index for subject in range(N_SUBJECTS) for class_index in range(2)],
        dtype=np.int64,
    )
    if not np.array_equal(selected, expected):
        raise ValueError("Left/Right extraction did not yield the frozen 18-cell order")
    return selected


def extract_lr_matrix(
    matrix: np.ndarray, subjects: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (36, 36) or not np.isfinite(values).all():
        raise ValueError("parent cost matrix must be finite 36x36")
    indices = lr_parent_indices(subjects, classes)
    return values[np.ix_(indices, indices)].copy(), indices


def relation_statistics(
    matrix: np.ndarray, *, n_classes: int, n_subjects: int = N_SUBJECTS
) -> FactorialStatistics:
    """Apply the exact frozen BNCI a/b/c/d, S/C/J arithmetic at supplied K."""

    if n_subjects != N_SUBJECTS:
        raise ValueError("V0 freezes exactly nine subjects")
    if n_classes < 2:
        raise ValueError("at least two classes are required")
    n_cells = n_subjects * n_classes
    values = np.asarray(matrix, dtype=np.float64)
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
                    values[anchor, cell_index(subject, other_class, n_classes=n_classes)]
                    for other_class in range(n_classes)
                    if other_class != class_index
                ]
            )
            c[subject, class_index] = np.mean(
                [
                    values[anchor, cell_index(other_subject, class_index, n_classes=n_classes)]
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
    return FactorialStatistics(
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


def subjectbreak_mappings(
    *, n_classes: int, replicates: int = NULL_REPLICATES
) -> np.ndarray:
    generator = np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, SUBJECT_STREAM_TAG])
    )
    mappings = np.empty((replicates, N_SUBJECTS * n_classes), dtype=np.int64)
    for draw in range(replicates):
        mapping = np.empty(N_SUBJECTS * n_classes, dtype=np.int64)
        for class_index in range(n_classes):
            permutation = generator.permutation(N_SUBJECTS)
            for subject in range(N_SUBJECTS):
                mapping[cell_index(subject, class_index, n_classes=n_classes)] = cell_index(
                    int(permutation[subject]), class_index, n_classes=n_classes
                )
        mappings[draw] = mapping
    return mappings


def classbreak_mappings(
    *, n_classes: int, replicates: int = NULL_REPLICATES
) -> np.ndarray:
    generator = np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, CLASS_STREAM_TAG])
    )
    mappings = np.empty((replicates, N_SUBJECTS * n_classes), dtype=np.int64)
    for draw in range(replicates):
        mapping = np.empty(N_SUBJECTS * n_classes, dtype=np.int64)
        for subject in range(N_SUBJECTS):
            permutation = generator.permutation(n_classes)
            for class_index in range(n_classes):
                mapping[cell_index(subject, class_index, n_classes=n_classes)] = cell_index(
                    subject, int(permutation[class_index]), n_classes=n_classes
                )
        mappings[draw] = mapping
    return mappings


def plus_one_pvalue(observed: float, null: np.ndarray) -> float:
    values = np.asarray(null, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("null must be a finite nonempty vector")
    return float((1 + np.count_nonzero(values >= observed)) / (len(values) + 1))


def evaluate_inference(
    matrix: np.ndarray,
    *,
    n_classes: int,
    replicates: int = NULL_REPLICATES,
    subject_mappings_array: np.ndarray | None = None,
    class_mappings_array: np.ndarray | None = None,
) -> FactorialInference:
    values = np.asarray(matrix, dtype=np.float64)
    expected_subject = subjectbreak_mappings(n_classes=n_classes, replicates=replicates)
    expected_class = classbreak_mappings(n_classes=n_classes, replicates=replicates)
    subject_maps = expected_subject if subject_mappings_array is None else np.asarray(subject_mappings_array)
    class_maps = expected_class if class_mappings_array is None else np.asarray(class_mappings_array)
    if not np.array_equal(subject_maps, expected_subject):
        raise ValueError("subject-break mappings differ from frozen stream")
    if not np.array_equal(class_maps, expected_class):
        raise ValueError("class-break mappings differ from frozen stream")
    observed = relation_statistics(values, n_classes=n_classes)
    subject_t = np.empty(replicates, dtype=np.float64)
    subject_j = np.empty(replicates, dtype=np.float64)
    class_t = np.empty(replicates, dtype=np.float64)
    class_j = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        subject_result = relation_statistics(values[:, subject_maps[draw]], n_classes=n_classes)
        class_result = relation_statistics(values[:, class_maps[draw]], n_classes=n_classes)
        subject_t[draw] = subject_result.t_subject
        subject_j[draw] = subject_result.t_j
        class_t[draw] = class_result.t_class
        class_j[draw] = class_result.t_j
    return FactorialInference(
        observed=observed,
        subjectbreak_t_subject=subject_t,
        subjectbreak_t_j=subject_j,
        classbreak_t_class=class_t,
        classbreak_t_j=class_j,
        subject_mappings=subject_maps,
        class_mappings=class_maps,
        p_subject=plus_one_pvalue(observed.t_subject, subject_t),
        p_class=plus_one_pvalue(observed.t_class, class_t),
        p_j_subjectbreak=plus_one_pvalue(observed.t_j, subject_j),
        p_j_classbreak=plus_one_pvalue(observed.t_j, class_j),
    )


def terminal_decision(
    *,
    t_j: float,
    p_j_subjectbreak: float,
    p_j_classbreak: float,
    split_half_sign_stable: bool,
) -> str:
    supported = t_j > 0.0 and p_j_subjectbreak < ALPHA and p_j_classbreak < ALPHA
    if supported and split_half_sign_stable:
        return "BNCI_LR_ANGULAR_INTERACTION_SUPPORTED_AND_STABLE"
    if supported:
        return "BNCI_LR_ANGULAR_INTERACTION_SUPPORTED_BUT_SPLIT_UNSTABLE"
    return "BNCI_LR_ANGULAR_INTERACTION_NOT_SUPPORTED"
