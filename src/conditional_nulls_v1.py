"""Deterministic null engines for Conditional-Geometry Anatomy v1.

This module is deliberately data-source agnostic.  It accepts label metadata,
4-by-4 cached objects, or fixed 24-candidate score sets.  It never loads EEG or
sessions.  Every replicate owns an indexed PCG64DXSM stream, so batching,
parallel worker order, and checkpoint resume cannot alter a draw.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.conditional_statistics_v1 import (
    conservative_candidate_ranks,
    normalize_shape_vectors,
)


__all__ = [
    "MASTER_SEED",
    "FAMILY_LABEL",
    "FAMILY_SEMANTIC",
    "FAMILY_ORACLE",
    "FAMILY_BOOTSTRAP",
    "FAMILY_GAUGE",
    "PHASE_DISCOVERY",
    "PHASE_CONFIRMATORY",
    "PHASE_COMMON",
    "IndexedNullResult",
    "NullCheckpoint",
    "tagged_replicate_rng",
    "all_s4_permutations",
    "permutation_matrix",
    "permute_class_object",
    "permuted_object_bank",
    "permuted_shape_bank",
    "shuffle_labels_within_strata",
    "run_label_destruction_null",
    "semantic_permutation_indices",
    "semantic_discovery_null",
    "semantic_confirmatory_null",
    "oracle_random_true_indices",
    "oracle_rank_null",
    "all_derangements",
    "unrelated_derangement_statistics",
    "create_null_checkpoint",
    "record_checkpoint_batch",
    "pending_checkpoint_indices",
    "save_null_checkpoint",
    "load_null_checkpoint",
]


MASTER_SEED = 20260809
FAMILY_LABEL = 1101
FAMILY_SEMANTIC = 1201
FAMILY_ORACLE = 1301
FAMILY_BOOTSTRAP = 1401
FAMILY_GAUGE = 1501
PHASE_DISCOVERY = 0
PHASE_CONFIRMATORY = 1
PHASE_COMMON = 2

_FAMILY_TAGS = {
    FAMILY_LABEL,
    FAMILY_SEMANTIC,
    FAMILY_ORACLE,
    FAMILY_BOOTSTRAP,
    FAMILY_GAUGE,
}
_PHASE_TAGS = {PHASE_DISCOVERY, PHASE_CONFIRMATORY, PHASE_COMMON}


def _nonnegative_integer(value: int, *, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def tagged_replicate_rng(
    *,
    family_tag: int,
    phase_tag: int,
    replicate_index: int,
    master_seed: int = MASTER_SEED,
) -> np.random.Generator:
    """Construct the exact four-field PCG64DXSM replicate generator."""

    master = _nonnegative_integer(master_seed, name="master_seed")
    family = _nonnegative_integer(family_tag, name="family_tag")
    phase = _nonnegative_integer(phase_tag, name="phase_tag")
    replicate = _nonnegative_integer(replicate_index, name="replicate_index")
    if family not in _FAMILY_TAGS:
        raise ValueError(f"unregistered RNG family tag: {family}")
    if phase not in _PHASE_TAGS:
        raise ValueError(f"unregistered RNG phase tag: {phase}")
    seed = np.random.SeedSequence([master, family, phase, replicate])
    return np.random.Generator(np.random.PCG64DXSM(seed))


@lru_cache(maxsize=1)
def _s4_tuple_cache() -> tuple[tuple[int, int, int, int], ...]:
    return tuple(itertools.permutations(range(4)))


def all_s4_permutations() -> np.ndarray:
    """Return all 24 S4 tuples in Python lexicographic order, identity first."""

    result = np.asarray(_s4_tuple_cache(), dtype=np.int64)
    if result.shape != (24, 4) or not np.array_equal(result[0], np.arange(4)):
        raise RuntimeError("internal S4 enumeration contract failed")
    return result.copy()


def _validate_permutation(permutation: Sequence[int]) -> np.ndarray:
    values = np.asarray(permutation, dtype=np.int64)
    if values.shape != (4,) or not np.array_equal(np.sort(values), np.arange(4)):
        raise ValueError("permutation must be a tuple containing 0,1,2,3 exactly once")
    return values


def permutation_matrix(permutation: Sequence[int]) -> np.ndarray:
    """Matrix P such that ``P @ O @ P.T == O[pi,pi]``."""

    values = _validate_permutation(permutation)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[np.arange(4), values] = 1.0
    return matrix


def _object_stack(objects: np.ndarray, *, name: str = "objects") -> np.ndarray:
    array = np.asarray(objects, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must have shape (...,4,4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def permute_class_object(objects: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    """Apply the frozen ``P O P.T`` action to one or many D/G objects."""

    array = _object_stack(objects)
    values = _validate_permutation(permutation)
    return np.take(np.take(array, values, axis=-2), values, axis=-1)


def permuted_object_bank(objects: np.ndarray) -> np.ndarray:
    """Append a 24-candidate axis immediately before the two matrix axes."""

    array = _object_stack(objects)
    candidates = [permute_class_object(array, permutation) for permutation in all_s4_permutations()]
    return np.stack(candidates, axis=-3)


def permuted_shape_bank(
    objects: np.ndarray,
    vectorizer: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Permute 4x4 objects, re-vectorize, then normalize every candidate.

    ``vectorizer`` must preserve all leading axes and replace the final 4x4
    axes with one feature axis.  This loose callable contract supports both D
    upper triangles and G Frobenius-isometric svec without importing geometry.
    """

    bank = permuted_object_bank(objects)
    vectors = np.asarray(vectorizer(bank), dtype=np.float64)
    if vectors.shape[:-1] != bank.shape[:-2] or vectors.shape[-1] < 1:
        raise ValueError(
            "vectorizer must preserve leading axes and replace 4x4 by a feature axis; "
            f"object bank {bank.shape}, vector output {vectors.shape}"
        )
    return normalize_shape_vectors(vectors)


def _one_dimensional(values: Sequence[Any] | np.ndarray, n: int, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} must have shape ({n},), got {array.shape}")
    return array


def _canonical_subjects(subjects: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(subjects, dtype=np.int64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("subjects must be a non-empty one-dimensional array")
    unique = np.unique(values)
    if not np.array_equal(unique, np.arange(unique[0], unique[-1] + 1)):
        raise ValueError("subject IDs must form a canonical consecutive integer range")
    return unique


def _canonical_subject_id_vector(
    subjects: Sequence[int] | np.ndarray | None,
    n_subjects: int,
) -> np.ndarray:
    values = (
        np.arange(1, n_subjects + 1, dtype=np.int64)
        if subjects is None
        else np.asarray(subjects, dtype=np.int64)
    )
    unique = _canonical_subjects(values)
    if values.shape != (n_subjects,) or not np.array_equal(values, unique):
        raise ValueError(
            "subjects must contain one canonical ascending ID per object subject axis"
        )
    return values


def _run_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return int(text), text
    except ValueError as error:
        raise ValueError(f"run value is not an integer label: {value!r}") from error


def shuffle_labels_within_strata(
    labels: Sequence[Any] | np.ndarray,
    subjects: Sequence[int] | np.ndarray,
    sessions: Sequence[Any] | np.ndarray,
    runs: Sequence[Any] | np.ndarray,
    trial_uids: Sequence[str] | np.ndarray,
    *,
    replicate_index: int,
    phase_tag: int,
    master_seed: int = MASTER_SEED,
) -> np.ndarray:
    """Shuffle each subject/session/run label multiset in canonical draw order."""

    label_array = np.asarray(labels)
    if label_array.ndim != 1 or len(label_array) < 1:
        raise ValueError("labels must be a non-empty one-dimensional array")
    n_rows = len(label_array)
    subject_array = _one_dimensional(subjects, n_rows, name="subjects").astype(np.int64)
    session_array = _one_dimensional(sessions, n_rows, name="sessions").astype(str)
    run_array = _one_dimensional(runs, n_rows, name="runs")
    uid_array = _one_dimensional(trial_uids, n_rows, name="trial_uids").astype(str)
    if len(set(uid_array.tolist())) != n_rows:
        raise ValueError("trial_uids must be globally unique")
    canonical_subjects = _canonical_subjects(subject_array)
    rng = tagged_replicate_rng(
        family_tag=FAMILY_LABEL,
        phase_tag=phase_tag,
        replicate_index=replicate_index,
        master_seed=master_seed,
    )
    output = label_array.copy()
    sessions_sorted = sorted(set(session_array.tolist()))
    runs_sorted = sorted(set(run_array.tolist()), key=_run_sort_key)
    covered = np.zeros(n_rows, dtype=bool)
    for subject in canonical_subjects:
        for session in sessions_sorted:
            for run in runs_sorted:
                indices = np.flatnonzero(
                    (subject_array == subject)
                    & (session_array == session)
                    & (run_array.astype(str) == str(run))
                )
                if len(indices) == 0:
                    continue
                indices = indices[np.argsort(uid_array[indices], kind="stable")]
                output[indices] = rng.permutation(label_array[indices])
                covered[indices] = True
    if not covered.all():
        raise RuntimeError("internal stratum traversal did not cover every label")
    return output


@dataclass(frozen=True)
class IndexedNullResult:
    replicate_indices: np.ndarray
    subject_statistics: np.ndarray
    group_statistics: np.ndarray


def _replicate_indices(values: Sequence[int] | np.ndarray) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 1:
        raise ValueError("replicate_indices must be a non-empty one-dimensional array")
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError("replicate_indices must be unique non-negative integers")
    return indices


def run_label_destruction_null(
    labels: Sequence[Any] | np.ndarray,
    subjects: Sequence[int] | np.ndarray,
    sessions: Sequence[Any] | np.ndarray,
    runs: Sequence[Any] | np.ndarray,
    trial_uids: Sequence[str] | np.ndarray,
    statistic_fn: Callable[[np.ndarray], np.ndarray],
    *,
    replicate_indices: Sequence[int] | np.ndarray,
    phase_tag: int,
    master_seed: int = MASTER_SEED,
) -> IndexedNullResult:
    """Run an exact indexed label null through a caller-supplied geometry refit.

    ``statistic_fn`` receives one permuted label vector and must recompute all
    class means/objects needed for a finite one-score-per-subject result.
    """

    indices = _replicate_indices(replicate_indices)
    canonical_subjects = _canonical_subjects(subjects)
    subject_statistics = np.empty((len(indices), len(canonical_subjects)), dtype=np.float64)
    for row, replicate in enumerate(indices):
        permuted = shuffle_labels_within_strata(
            labels,
            subjects,
            sessions,
            runs,
            trial_uids,
            replicate_index=int(replicate),
            phase_tag=phase_tag,
            master_seed=master_seed,
        )
        values = np.asarray(statistic_fn(permuted), dtype=np.float64)
        if values.shape != (len(canonical_subjects),) or not np.isfinite(values).all():
            raise ValueError(
                "statistic_fn must return one finite value per canonical subject, got "
                f"{values.shape}"
            )
        subject_statistics[row] = values
    return IndexedNullResult(
        replicate_indices=indices.copy(),
        subject_statistics=subject_statistics,
        group_statistics=np.median(subject_statistics, axis=1),
    )


def semantic_permutation_indices(
    replicate_indices: Sequence[int] | np.ndarray,
    subjects: Sequence[int] | np.ndarray,
    *,
    master_seed: int = MASTER_SEED,
) -> np.ndarray:
    """Draw one independent full-S4 index per subject and common replicate."""

    indices = _replicate_indices(replicate_indices)
    canonical = np.asarray(subjects, dtype=np.int64)
    unique = _canonical_subjects(canonical)
    if not np.array_equal(canonical, unique):
        raise ValueError("subjects must be unique and supplied in canonical ascending order")
    plans = np.empty((len(indices), len(canonical)), dtype=np.int64)
    for row, replicate in enumerate(indices):
        rng = tagged_replicate_rng(
            family_tag=FAMILY_SEMANTIC,
            phase_tag=PHASE_COMMON,
            replicate_index=int(replicate),
            master_seed=master_seed,
        )
        plans[row] = rng.integers(0, 24, size=len(canonical), endpoint=False)
    return plans


def _objects_by_subject_split(objects: np.ndarray, *, name: str) -> np.ndarray:
    array = _object_stack(objects, name=name)
    if array.ndim != 4 or array.shape[1:] != (3, 4, 4) or array.shape[0] < 2:
        raise ValueError(f"{name} must have shape (subjects>=2,3[A,B,F],4,4)")
    return array


def _selected_candidate_shapes(bank: np.ndarray, plans: np.ndarray, split: int) -> np.ndarray:
    # bank: subject,split,candidate,feature; plans: replicate,subject
    subject_index = np.arange(bank.shape[0], dtype=np.int64)[None, :]
    return bank[:, split][subject_index, plans]


def _batched_loso_templates(selected_shapes: np.ndarray) -> np.ndarray:
    unit = normalize_shape_vectors(selected_shapes)
    # Directly sum r != target.  This makes the target's randomly drawn pi
    # irrelevant even at floating-point rounding level.
    source_sums = np.stack(
        [
            np.sum(np.delete(unit, target, axis=1), axis=1, dtype=np.float64)
            for target in range(unit.shape[1])
        ],
        axis=1,
    )
    return normalize_shape_vectors(source_sums)


def semantic_discovery_null(
    discovery_objects: np.ndarray,
    vectorizer: Callable[[np.ndarray], np.ndarray],
    *,
    replicate_indices: Sequence[int] | np.ndarray,
    subjects: Sequence[int] | np.ndarray | None = None,
    master_seed: int = MASTER_SEED,
    batch_size: int = 5_000,
) -> IndexedNullResult:
    """Vectorized source-semantic null for discovery Stage S."""

    objects = _objects_by_subject_split(discovery_objects, name="discovery_objects")
    subject_ids = _canonical_subject_id_vector(subjects, len(objects))
    indices = _replicate_indices(replicate_indices)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    bank = permuted_shape_bank(objects, vectorizer)
    identity = bank[:, :, 0]
    result = np.empty((len(indices), len(objects)), dtype=np.float64)
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        plans = semantic_permutation_indices(indices[start:stop], subject_ids, master_seed=master_seed)
        source_a = _selected_candidate_shapes(bank, plans, 0)
        source_b = _selected_candidate_shapes(bank, plans, 1)
        template_a = _batched_loso_templates(source_a)
        template_b = _batched_loso_templates(source_b)
        a_to_b = np.einsum("bsf,sf->bs", template_a, identity[:, 1], optimize=True)
        b_to_a = np.einsum("bsf,sf->bs", template_b, identity[:, 0], optimize=True)
        result[start:stop] = 0.5 * (a_to_b + b_to_a)
    return IndexedNullResult(indices.copy(), result, np.median(result, axis=1))


def semantic_confirmatory_null(
    discovery_objects: np.ndarray,
    confirmatory_objects: np.ndarray,
    vectorizer: Callable[[np.ndarray], np.ndarray],
    *,
    replicate_indices: Sequence[int] | np.ndarray,
    subjects: Sequence[int] | np.ndarray | None = None,
    master_seed: int = MASTER_SEED,
    batch_size: int = 5_000,
) -> IndexedNullResult:
    """Common-plan semantic null for frozen discovery-F versus confirmatory A/B."""

    discovery = _objects_by_subject_split(discovery_objects, name="discovery_objects")
    confirmatory = _objects_by_subject_split(confirmatory_objects, name="confirmatory_objects")
    if discovery.shape != confirmatory.shape:
        raise ValueError("discovery and confirmatory objects must have equal shape")
    subject_ids = _canonical_subject_id_vector(subjects, len(discovery))
    indices = _replicate_indices(replicate_indices)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    discovery_bank = permuted_shape_bank(discovery, vectorizer)
    confirmatory_identity = permuted_shape_bank(confirmatory, vectorizer)[:, :, 0]
    result = np.empty((len(indices), len(discovery)), dtype=np.float64)
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        plans = semantic_permutation_indices(indices[start:stop], subject_ids, master_seed=master_seed)
        source_f = _selected_candidate_shapes(discovery_bank, plans, 2)
        template_f = _batched_loso_templates(source_f)
        scores_a = np.einsum(
            "bsf,sf->bs", template_f, confirmatory_identity[:, 0], optimize=True
        )
        scores_b = np.einsum(
            "bsf,sf->bs", template_f, confirmatory_identity[:, 1], optimize=True
        )
        result[start:stop] = 0.5 * (scores_a + scores_b)
    return IndexedNullResult(indices.copy(), result, np.median(result, axis=1))


def oracle_random_true_indices(
    replicate_indices: Sequence[int] | np.ndarray,
    subjects: Sequence[int] | np.ndarray,
    *,
    master_seed: int = MASTER_SEED,
) -> np.ndarray:
    """Common-plan uniform reassignment of the true candidate for each subject."""

    indices = _replicate_indices(replicate_indices)
    canonical = np.asarray(subjects, dtype=np.int64)
    unique = _canonical_subjects(canonical)
    if not np.array_equal(canonical, unique):
        raise ValueError("subjects must be unique and supplied in canonical ascending order")
    plans = np.empty((len(indices), len(canonical)), dtype=np.int64)
    for row, replicate in enumerate(indices):
        rng = tagged_replicate_rng(
            family_tag=FAMILY_ORACLE,
            phase_tag=PHASE_COMMON,
            replicate_index=int(replicate),
            master_seed=master_seed,
        )
        plans[row] = rng.integers(0, 24, size=len(canonical), endpoint=False)
    return plans


def oracle_rank_null(
    score_sets: np.ndarray,
    *,
    replicate_indices: Sequence[int] | np.ndarray,
    subjects: Sequence[int] | np.ndarray | None = None,
    master_seed: int = MASTER_SEED,
    tolerance: float = 1.0e-12,
    batch_size: int = 25_000,
) -> IndexedNullResult:
    """Cheap vectorized null from fixed per-subject 24-candidate score sets."""

    scores = np.asarray(score_sets, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 24 or scores.shape[0] < 2:
        raise ValueError("score_sets must have shape (subjects>=2,24)")
    if not np.isfinite(scores).all():
        raise ValueError("score_sets contains NaN or Inf")
    subject_ids = _canonical_subject_id_vector(subjects, len(scores))
    indices = _replicate_indices(replicate_indices)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    ranks = conservative_candidate_ranks(scores, tolerance=tolerance)
    normalized = (24.0 - ranks.astype(np.float64)) / 23.0
    result = np.empty((len(indices), len(scores)), dtype=np.float64)
    subject_index = np.arange(len(scores), dtype=np.int64)[None, :]
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        plans = oracle_random_true_indices(indices[start:stop], subject_ids, master_seed=master_seed)
        result[start:stop] = normalized[subject_index, plans]
    return IndexedNullResult(indices.copy(), result, np.median(result, axis=1))


@lru_cache(maxsize=None)
def _derangement_tuple_cache(n_subjects: int) -> tuple[tuple[int, ...], ...]:
    n = int(n_subjects)
    if n < 2:
        raise ValueError("at least two subjects are required")
    return tuple(
        permutation
        for permutation in itertools.permutations(range(n))
        if all(index != value for index, value in enumerate(permutation))
    )


def all_derangements(n_subjects: int = 9) -> np.ndarray:
    """Enumerate fixed-point-free subject permutations lexicographically."""

    result = np.asarray(_derangement_tuple_cache(int(n_subjects)), dtype=np.int64)
    if int(n_subjects) == 9 and result.shape != (133_496, 9):
        raise RuntimeError(f"expected !9=133496, got {result.shape}")
    return result.copy()


def unrelated_derangement_statistics(
    shapes_a: np.ndarray,
    shapes_b: np.ndarray,
) -> np.ndarray:
    """Median cross-subject cosine for every fixed-point-free mapping."""

    a = normalize_shape_vectors(np.asarray(shapes_a, dtype=np.float64))
    b = normalize_shape_vectors(np.asarray(shapes_b, dtype=np.float64))
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("shapes_a and shapes_b must be equal 2-D subject-by-feature arrays")
    mappings = all_derangements(len(a))
    cross_cosines = a @ b.T
    selected = cross_cosines[np.arange(len(a))[None, :], mappings]
    return np.median(selected, axis=1)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _update_array_hash(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes(order="C"))


@dataclass(frozen=True)
class NullCheckpoint:
    metadata: dict[str, Any]
    replicate_indices: np.ndarray
    completed: np.ndarray
    subject_statistics: np.ndarray
    group_statistics: np.ndarray
    payload_hash: str


_REQUIRED_CHECKPOINT_METADATA = {
    "schema_version",
    "protocol_sha256",
    "config_sha256",
    "code_commit",
    "input_hash",
    "family_tag",
    "phase_tag",
    "total_replicates",
    "n_subjects",
    "bit_generator",
    "master_seed",
    "replicate_index_base",
}


def _checkpoint_payload_hash(
    metadata: Mapping[str, Any],
    replicate_indices: np.ndarray,
    completed: np.ndarray,
    subject_statistics: np.ndarray,
    group_statistics: np.ndarray,
) -> str:
    digest = hashlib.sha256(_canonical_json(metadata).encode("utf-8"))
    for array in (replicate_indices, completed, subject_statistics, group_statistics):
        _update_array_hash(digest, array)
    return digest.hexdigest()


def _validate_checkpoint_arrays(
    metadata: Mapping[str, Any],
    replicate_indices: np.ndarray,
    completed: np.ndarray,
    subject_statistics: np.ndarray,
    group_statistics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = _REQUIRED_CHECKPOINT_METADATA - set(metadata)
    if missing:
        raise ValueError(f"checkpoint metadata is missing required keys: {sorted(missing)}")
    total = int(metadata["total_replicates"])
    n_subjects = int(metadata["n_subjects"])
    if total < 1 or n_subjects < 1:
        raise ValueError("checkpoint dimensions must be positive")
    if metadata["bit_generator"] != "PCG64DXSM":
        raise ValueError("checkpoint bit_generator must be PCG64DXSM")
    if int(metadata["replicate_index_base"]) != 0:
        raise ValueError("checkpoint replicate_index_base must be zero")
    tagged_replicate_rng(
        family_tag=int(metadata["family_tag"]),
        phase_tag=int(metadata["phase_tag"]),
        replicate_index=0,
        master_seed=int(metadata["master_seed"]),
    )
    fixed_indices = np.asarray(replicate_indices, dtype=np.int64)
    bitmap = np.asarray(completed, dtype=np.uint8)
    subjects = np.asarray(subject_statistics, dtype=np.float64)
    groups = np.asarray(group_statistics, dtype=np.float64)
    if (
        fixed_indices.shape != (total,)
        or not np.array_equal(fixed_indices, np.arange(total, dtype=np.int64))
        or bitmap.shape != (total,)
        or subjects.shape != (total, n_subjects)
        or groups.shape != (total,)
    ):
        raise ValueError("checkpoint arrays do not match total_replicates/n_subjects")
    if np.any((bitmap != 0) & (bitmap != 1)):
        raise ValueError("checkpoint completed bitmap must contain only 0/1")
    done = bitmap.astype(bool)
    if np.any(~np.isfinite(subjects[done])) or np.any(~np.isfinite(groups[done])):
        raise ValueError("completed checkpoint rows must be finite")
    if np.any(~np.isnan(subjects[~done])) or np.any(~np.isnan(groups[~done])):
        raise ValueError("incomplete checkpoint rows must remain NaN")
    expected_groups = np.median(subjects[done], axis=1) if np.any(done) else np.empty(0)
    if np.any(done) and not np.array_equal(expected_groups, groups[done]):
        raise ValueError("checkpoint group medians do not match subject statistics")
    return fixed_indices, bitmap, subjects, groups


def create_null_checkpoint(
    *,
    total_replicates: int,
    n_subjects: int,
    protocol_sha256: str,
    config_sha256: str,
    code_commit: str,
    input_hash: str,
    family_tag: int,
    phase_tag: int,
    master_seed: int = MASTER_SEED,
    schema_version: str = "conditional-null-checkpoint-v1",
) -> NullCheckpoint:
    """Create an empty fixed-index checkpoint with complete identity metadata."""

    total = _nonnegative_integer(total_replicates, name="total_replicates")
    subjects = _nonnegative_integer(n_subjects, name="n_subjects")
    if total < 1 or subjects < 1:
        raise ValueError("total_replicates and n_subjects must be positive")
    # Validate tags through RNG construction without consuming its state.
    tagged_replicate_rng(
        family_tag=family_tag,
        phase_tag=phase_tag,
        replicate_index=0,
        master_seed=master_seed,
    )
    for name, value in {
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "code_commit": code_commit,
        "input_hash": input_hash,
    }.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    metadata: dict[str, Any] = {
        "schema_version": str(schema_version),
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "code_commit": code_commit,
        "input_hash": input_hash,
        "family_tag": int(family_tag),
        "phase_tag": int(phase_tag),
        "total_replicates": total,
        "n_subjects": subjects,
        "bit_generator": "PCG64DXSM",
        "master_seed": int(master_seed),
        "replicate_index_base": 0,
    }
    completed = np.zeros(total, dtype=np.uint8)
    replicate_indices = np.arange(total, dtype=np.int64)
    subject_statistics = np.full((total, subjects), np.nan, dtype=np.float64)
    group_statistics = np.full(total, np.nan, dtype=np.float64)
    payload_hash = _checkpoint_payload_hash(
        metadata, replicate_indices, completed, subject_statistics, group_statistics
    )
    return NullCheckpoint(
        metadata,
        replicate_indices,
        completed,
        subject_statistics,
        group_statistics,
        payload_hash,
    )


def record_checkpoint_batch(
    checkpoint: NullCheckpoint,
    replicate_indices: Sequence[int] | np.ndarray,
    subject_statistics: np.ndarray,
) -> NullCheckpoint:
    """Record only previously incomplete fixed indices; completed work is immutable."""

    fixed_indices, bitmap, existing_subjects, existing_groups = _validate_checkpoint_arrays(
        checkpoint.metadata,
        checkpoint.replicate_indices,
        checkpoint.completed,
        checkpoint.subject_statistics,
        checkpoint.group_statistics,
    )
    expected_hash = _checkpoint_payload_hash(
        checkpoint.metadata, fixed_indices, bitmap, existing_subjects, existing_groups
    )
    if checkpoint.payload_hash != expected_hash:
        raise ValueError("checkpoint payload hash mismatch before update")
    indices = _replicate_indices(replicate_indices)
    if np.any(indices >= len(bitmap)):
        raise ValueError("replicate index is outside checkpoint range")
    if np.any(bitmap[indices] == 1):
        raise ValueError("refusing to recompute an already completed replicate")
    values = np.asarray(subject_statistics, dtype=np.float64)
    expected_shape = (len(indices), existing_subjects.shape[1])
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"subject_statistics must be finite with shape {expected_shape}")
    new_bitmap = bitmap.copy()
    new_subjects = existing_subjects.copy()
    new_groups = existing_groups.copy()
    new_subjects[indices] = values
    new_groups[indices] = np.median(values, axis=1)
    new_bitmap[indices] = 1
    payload_hash = _checkpoint_payload_hash(
        checkpoint.metadata, fixed_indices, new_bitmap, new_subjects, new_groups
    )
    return replace(
        checkpoint,
        completed=new_bitmap,
        subject_statistics=new_subjects,
        group_statistics=new_groups,
        payload_hash=payload_hash,
    )


def pending_checkpoint_indices(checkpoint: NullCheckpoint) -> np.ndarray:
    """Return incomplete replicate indices in canonical ascending order."""

    fixed_indices, bitmap, subjects, groups = _validate_checkpoint_arrays(
        checkpoint.metadata,
        checkpoint.replicate_indices,
        checkpoint.completed,
        checkpoint.subject_statistics,
        checkpoint.group_statistics,
    )
    if checkpoint.payload_hash != _checkpoint_payload_hash(
        checkpoint.metadata, fixed_indices, bitmap, subjects, groups
    ):
        raise ValueError("checkpoint payload hash mismatch")
    return np.flatnonzero(bitmap == 0).astype(np.int64)


def save_null_checkpoint(path: str | Path, checkpoint: NullCheckpoint) -> None:
    """Validate and atomically save a compressed checkpoint NPZ."""

    fixed_indices, bitmap, subjects, groups = _validate_checkpoint_arrays(
        checkpoint.metadata,
        checkpoint.replicate_indices,
        checkpoint.completed,
        checkpoint.subject_statistics,
        checkpoint.group_statistics,
    )
    payload_hash = _checkpoint_payload_hash(
        checkpoint.metadata, fixed_indices, bitmap, subjects, groups
    )
    if payload_hash != checkpoint.payload_hash:
        raise ValueError("checkpoint payload hash mismatch")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(_canonical_json(checkpoint.metadata)),
                replicate_indices=fixed_indices,
                completed=bitmap,
                subject_statistics=subjects,
                group_statistics=groups,
                payload_hash=np.asarray(payload_hash),
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_null_checkpoint(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> NullCheckpoint:
    """Load a checkpoint and reject identity, shape, median, or hash mismatch."""

    target = Path(path)
    with np.load(target, allow_pickle=False) as archive:
        required = {
            "metadata_json",
            "replicate_indices",
            "completed",
            "subject_statistics",
            "group_statistics",
            "payload_hash",
        }
        if set(archive.files) != required:
            raise ValueError(
                f"checkpoint keys mismatch: expected {sorted(required)}, got {sorted(archive.files)}"
            )
        metadata = json.loads(str(archive["metadata_json"].item()))
        replicate_indices = np.asarray(archive["replicate_indices"], dtype=np.int64)
        completed = np.asarray(archive["completed"], dtype=np.uint8)
        subjects = np.asarray(archive["subject_statistics"], dtype=np.float64)
        groups = np.asarray(archive["group_statistics"], dtype=np.float64)
        stored_hash = str(archive["payload_hash"].item())
    if expected_metadata is not None:
        for key, value in expected_metadata.items():
            if key not in metadata or metadata[key] != value:
                raise ValueError(
                    f"checkpoint metadata mismatch for {key}: "
                    f"stored={metadata.get(key)!r}, expected={value!r}"
                )
    fixed_indices, bitmap, subjects, groups = _validate_checkpoint_arrays(
        metadata, replicate_indices, completed, subjects, groups
    )
    computed_hash = _checkpoint_payload_hash(
        metadata, fixed_indices, bitmap, subjects, groups
    )
    if stored_hash != computed_hash:
        raise ValueError("checkpoint payload hash mismatch")
    return NullCheckpoint(metadata, fixed_indices, bitmap, subjects, groups, stored_hash)
