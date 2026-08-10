"""Frozen statistics for Subject Class Interaction v0.

All functions are deterministic, array-only, and intentionally contain no
classifier, adaptation, model-fitting, or feature-selection code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NullSummary:
    observed: float
    null_median: float
    effect: float
    exceedances: int
    p_value: float
    replicates: int


def normalization_threshold(vector_dimension: int) -> float:
    if int(vector_dimension) < 1:
        raise ValueError("vector_dimension must be positive")
    return float(np.finfo(np.float64).eps * np.sqrt(int(vector_dimension)))


def normalize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 1 or not np.isfinite(array).all():
        raise ValueError("signature values must be finite with a nonempty feature axis")
    norms = np.linalg.norm(array, axis=-1)
    threshold = normalization_threshold(array.shape[-1])
    if np.any(norms <= threshold):
        index = tuple(np.argwhere(norms <= threshold)[0])
        raise ValueError(
            "DEGENERATE_INTERACTION_SIGNATURE: raw norm is at or below the "
            f"frozen machine-scale threshold at index {index}"
        )
    return array / norms[..., None], norms


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("cosine operands must be equal-shaped finite arrays")
    unit_a, _ = normalize_rows(a)
    unit_b, _ = normalize_rows(b)
    return np.sum(unit_a * unit_b, axis=-1, dtype=np.float64)


def reliability_subject_scores(half_a: np.ndarray, half_b: np.ndarray) -> np.ndarray:
    """Mean sessionwise half-A/half-B cosine for each subject."""

    scores = cosine_rows(half_a, half_b)
    if scores.ndim != 2:
        raise ValueError("half signatures must have shape (subject, session, feature)")
    return np.mean(scores, axis=1, dtype=np.float64)


def reliability_statistic(half_a: np.ndarray, half_b: np.ndarray) -> float:
    return float(np.median(reliability_subject_scores(half_a, half_b)))


def similarity_matrix(session0: np.ndarray, session1: np.ndarray) -> np.ndarray:
    left, _ = normalize_rows(session0)
    right, _ = normalize_rows(session1)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("session signatures must share (subject, feature) shape")
    return left @ right.T


def same_subject_scores(session0: np.ndarray, session1: np.ndarray) -> np.ndarray:
    return np.diag(similarity_matrix(session0, session1)).copy()


def same_subject_statistic(session0: np.ndarray, session1: np.ndarray) -> float:
    return float(np.median(same_subject_scores(session0, session1)))


@lru_cache(maxsize=8)
def _derangement_tuples(n_subjects: int) -> tuple[tuple[int, ...], ...]:
    n = int(n_subjects)
    if n < 2:
        raise ValueError("at least two subjects are required")
    identity = tuple(range(n))
    return tuple(
        candidate
        for candidate in itertools.permutations(range(n))
        if all(candidate[index] != identity[index] for index in range(n))
    )


def all_derangements(n_subjects: int) -> np.ndarray:
    values = np.asarray(_derangement_tuples(int(n_subjects)), dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != int(n_subjects):
        raise RuntimeError("internal derangement enumeration failure")
    identity = np.arange(int(n_subjects))
    if np.any(values == identity[None, :]):
        raise RuntimeError("derangement contains a fixed point")
    return values.copy()


def random_derangements(
    n_subjects: int,
    n_replicates: int,
    rngs: Sequence[np.random.Generator],
) -> np.ndarray:
    n = int(n_subjects)
    b = int(n_replicates)
    if len(rngs) != b or n < 2 or b < 1:
        raise ValueError("invalid random derangement dimensions")
    result = np.empty((b, n), dtype=np.int64)
    identity = np.arange(n)
    for replicate, rng in enumerate(rngs):
        while True:
            candidate = rng.permutation(n)
            if np.all(candidate != identity):
                result[replicate] = candidate
                break
    return result


def derangement_statistics(similarities: np.ndarray, mappings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(similarities, dtype=np.float64)
    plans = np.asarray(mappings, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("similarities must be a finite square matrix")
    n = matrix.shape[0]
    if plans.ndim != 2 or plans.shape[1] != n:
        raise ValueError("mappings must have shape (replicate, subject)")
    if np.any(plans == np.arange(n)[None, :]):
        raise ValueError("mappings contain a fixed point")
    if not np.all(np.sort(plans, axis=1) == np.arange(n)[None, :]):
        raise ValueError("mappings contain a non-permutation")
    values = matrix[np.arange(n)[None, :], plans]
    return np.median(values, axis=1)


def monte_carlo_summary(observed: float, null_values: np.ndarray) -> NullSummary:
    null = np.asarray(null_values, dtype=np.float64)
    if null.ndim != 1 or len(null) < 1 or not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError("observed/null values must be finite")
    exceedances = int(np.count_nonzero(null >= float(observed)))
    median = float(np.median(null))
    return NullSummary(
        observed=float(observed),
        null_median=median,
        effect=float(observed - median),
        exceedances=exceedances,
        p_value=float((1 + exceedances) / (1 + len(null))),
        replicates=int(len(null)),
    )


def exact_null_summary(observed: float, null_values: np.ndarray) -> dict[str, float | int]:
    summary = monte_carlo_summary(observed, null_values)
    return {
        **summary.__dict__,
        "exact_tail_probability": float(summary.exceedances / summary.replicates),
    }


def primary_outcome(
    *,
    hard_gates_pass: bool,
    openbmi_unlocked: bool,
    gate_r: bool | None,
    gate_i: bool | None,
    gate_c: bool | None,
    r_stable: bool,
    spectrum_supportive: bool,
    bnci_directions_positive: bool,
) -> str:
    """Apply the frozen terminal mapping without inventing cosine thresholds."""

    if not hard_gates_pass:
        return "UNASSESSED_NUMERICAL_OR_DATA_FAILURE"
    if not bnci_directions_positive and not openbmi_unlocked:
        return "STOP_BNCI_DIRECTION_FAILURE"
    if not openbmi_unlocked:
        return "OPENBMI_LOCKED_PENDING_EXTERNAL_REPLICATION"
    if gate_r is not True:
        return "UNASSESSED_CURRENT_OBJECT"
    if gate_i is not True:
        return "STOP_SUBJECT_MAIN_EFFECT_ONLY" if r_stable else "STOP_NO_STABLE_INDIVIDUAL_COMPONENT"
    if gate_c is not True:
        return "STOP_SUBJECT_MAIN_EFFECT_ONLY" if r_stable else "STOP_GENERIC_SUBJECT_FINGERPRINT"
    if not bnci_directions_positive:
        return "STOP_NOT_CROSS_DATASET_ROBUST"
    return (
        "GO_STABLE_SUBJECT_CLASS_INTERACTION"
        if spectrum_supportive
        else "GO_SENSOR_SPACE_ONLY"
    )
