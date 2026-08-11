"""Frozen chronological local-state correspondence analysis for BNCI2014_001."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.geometry_v2 import (
    AIRM_MAXITER,
    AIRM_TOLERANCE,
    airm_distance,
    airm_mean,
    spd_invsqrt,
    spd_log,
    symmetrize,
)
from src.spd_utils import svec


N_SUBJECTS = 9
N_CLASSES = 4
N_CELLS = N_SUBJECTS * N_CLASSES
N_TIMES = 5
N_SESSIONS = 2
N_HALVES = 2
CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
SESSION_ORDER = ("0train", "1test")
HALF_ORDER = ("A", "B")
HALF_RUNS = {"A": (0, 2, 4), "B": (1, 3, 5)}

MEAN_TOLERANCE = AIRM_TOLERANCE
MEAN_MAX_ITERATIONS = AIRM_MAXITER
NORMALIZED_RESIDUAL_MAX = 1.0e-7
DEFAULT_MASTER_SEED = 20260810
DEFAULT_NULL_REPLICATES = 1999
TEMPORAL_STREAM_TAG = 1201
CLASSBREAK_STREAM_TAG = 1101
SUBJECTBREAK_STREAM_TAG = 1102

ALL_PERMUTATIONS = np.asarray(list(permutations(range(N_TIMES))), dtype=np.int64)
IDENTITY_PERMUTATION = np.arange(N_TIMES, dtype=np.int64)
IDENTITY_INDEX = int(
    np.flatnonzero(np.all(ALL_PERMUTATIONS == IDENTITY_PERMUTATION, axis=1))[0]
)
DERANGEMENT_MASK = np.all(
    ALL_PERMUTATIONS != np.arange(N_TIMES, dtype=np.int64)[None, :], axis=1
)
DERANGEMENT_INDICES = np.flatnonzero(DERANGEMENT_MASK)


class TemporalNumericalError(RuntimeError):
    """A required AIRM mean or distance failed the frozen numerical gates."""


@dataclass(frozen=True)
class MeanSequenceBank:
    """Full and run-blocked ordered cell-level AIRM mean sequences."""

    full: np.ndarray
    split: np.ndarray
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class MatchingResults:
    """All cross-session K matrices and frozen temporal matching summaries."""

    k: np.ndarray
    all_costs: np.ndarray
    d_id: np.ndarray
    median_derangement: np.ndarray
    gain: np.ndarray
    relative_gain: np.ndarray
    identity_rank: np.ndarray
    identity_tie_count: np.ndarray
    relabelled_gain: np.ndarray


@dataclass(frozen=True)
class TemporalContrasts:
    """Observed cell-, subject-, and group-level temporal gain contrasts."""

    a_sc: np.ndarray
    b_sc: np.ndarray
    c_sc: np.ndarray
    d_sc: np.ndarray
    s_sc: np.ndarray
    c_specific_sc: np.ndarray
    j_sc: np.ndarray
    a_s: np.ndarray
    s_s: np.ndarray
    c_s: np.ndarray
    j_s: np.ndarray
    t_temporal: float
    t_subject: float
    t_class: float
    t_j: float


@dataclass(frozen=True)
class TemporalInference:
    """Observed contrasts, frozen nulls, p-values, and terminal decision."""

    observed: TemporalContrasts
    temporal_null: np.ndarray
    subjectbreak_t_subject: np.ndarray
    subjectbreak_t_j: np.ndarray
    classbreak_t_class: np.ndarray
    classbreak_t_j: np.ndarray
    temporal_permutation_indices: np.ndarray
    subject_mappings: np.ndarray
    class_mappings: np.ndarray
    p_temporal: float
    p_subject: float
    p_class: float
    p_j_subjectbreak: float
    p_j_classbreak: float
    terminal: str


@dataclass(frozen=True)
class CommonPCA:
    """One common AIRM tangent-space PCA for all full cell means."""

    reference: np.ndarray
    coordinates: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    feature_mean: np.ndarray
    reference_diagnostic: Mapping[str, Any]


def cell_index(subject_index: int, class_index: int) -> int:
    if not 0 <= subject_index < N_SUBJECTS:
        raise ValueError("subject index must be in 0..8")
    if not 0 <= class_index < N_CLASSES:
        raise ValueError("class index must be in 0..3")
    return subject_index * N_CLASSES + class_index


def cell_labels() -> tuple[tuple[int, str], ...]:
    return tuple(
        (subject, class_label)
        for subject in range(1, N_SUBJECTS + 1)
        for class_label in CLASS_ORDER
    )


def _validate_permutations() -> None:
    if ALL_PERMUTATIONS.shape != (120, N_TIMES):
        raise RuntimeError("S5 enumeration does not contain exactly 120 mappings")
    if len({tuple(value) for value in ALL_PERMUTATIONS.tolist()}) != 120:
        raise RuntimeError("S5 enumeration contains duplicates")
    if IDENTITY_INDEX != 0 or not np.array_equal(
        ALL_PERMUTATIONS[IDENTITY_INDEX], IDENTITY_PERMUTATION
    ):
        raise RuntimeError("identity is not the first lexicographic S5 mapping")
    if len(DERANGEMENT_INDICES) != 44:
        raise RuntimeError("S5 derangement enumeration does not contain 44 mappings")


_validate_permutations()


def _mean_record(
    covariances: np.ndarray,
    *,
    scope: str,
    subject: int | str,
    session: str,
    class_label: str,
    temporal_position: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = airm_mean(
        np.asarray(covariances, dtype=np.float64),
        tol=MEAN_TOLERANCE,
        maxiter=MEAN_MAX_ITERATIONS,
    )
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(result.matrix)))
    row = {
        "scope": scope,
        "subject": subject,
        "session": session,
        "class_label": class_label,
        "temporal_position": int(temporal_position),
        "n_covariances": int(len(covariances)),
        "solver_tolerance": result.tol,
        "solver_max_iterations": result.maxiter,
        "warning_count": len(result.warning_messages),
        "warning_messages": " | ".join(result.warning_messages),
        "karcher_post_residual": result.post_residual,
        "normalized_karcher_post_residual": result.normalized_post_residual,
        "minimum_eigenvalue": minimum_eigenvalue,
        "finite": bool(np.isfinite(result.matrix).all()),
        "passed": bool(
            np.isfinite(result.matrix).all()
            and minimum_eigenvalue > 0.0
            and not result.warning_messages
            and result.normalized_post_residual <= NORMALIZED_RESIDUAL_MAX
        ),
    }
    if not row["passed"]:
        raise TemporalNumericalError(
            "UNASSESSED_NUMERICAL_FAILURE: AIRM mean failed frozen gate: "
            f"{row}"
        )
    return result.matrix, row


def fit_ordered_mean_sequences(
    states: np.ndarray, metadata: pd.DataFrame
) -> MeanSequenceBank:
    """Fit 72 full and 144 split-half ordered mean sequences."""

    values = np.asarray(states, dtype=np.float64)
    if values.shape != (5184, N_TIMES, 22, 22):
        raise ValueError(f"states have unexpected shape {values.shape}")
    if len(metadata) != len(values):
        raise ValueError("metadata and state counts differ")
    required = {"subject", "session", "run", "class_label"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"metadata missing {sorted(required - set(metadata.columns))}")

    full = np.empty(
        (N_SESSIONS, N_SUBJECTS, N_CLASSES, N_TIMES, 22, 22), dtype=np.float64
    )
    split = np.empty(
        (N_HALVES, N_SESSIONS, N_SUBJECTS, N_CLASSES, N_TIMES, 22, 22),
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    subjects = metadata["subject"].to_numpy(dtype=int)
    sessions = metadata["session"].astype(str).to_numpy()
    runs = metadata["run"].to_numpy(dtype=int)
    classes = metadata["class_label"].astype(str).to_numpy()
    for session_index, session in enumerate(SESSION_ORDER):
        for subject_index in range(N_SUBJECTS):
            subject = subject_index + 1
            for class_index, class_label in enumerate(CLASS_ORDER):
                base_mask = (
                    (subjects == subject)
                    & (sessions == session)
                    & (classes == class_label)
                )
                indices = np.flatnonzero(base_mask)
                if len(indices) != 72:
                    raise ValueError(
                        f"cell {(subject, session, class_label)} has {len(indices)} trials"
                    )
                for time_index in range(N_TIMES):
                    matrix, row = _mean_record(
                        values[indices, time_index],
                        scope="Full",
                        subject=subject,
                        session=session,
                        class_label=class_label,
                        temporal_position=time_index + 1,
                    )
                    full[
                        session_index, subject_index, class_index, time_index
                    ] = matrix
                    rows.append(row)
                for half_index, half in enumerate(HALF_ORDER):
                    half_indices = np.flatnonzero(base_mask & np.isin(runs, HALF_RUNS[half]))
                    if len(half_indices) != 36:
                        raise ValueError(
                            f"half {(subject, session, class_label, half)} has "
                            f"{len(half_indices)} trials"
                        )
                    for time_index in range(N_TIMES):
                        matrix, row = _mean_record(
                            values[half_indices, time_index],
                            scope=f"Half_{half}",
                            subject=subject,
                            session=session,
                            class_label=class_label,
                            temporal_position=time_index + 1,
                        )
                        split[
                            half_index,
                            session_index,
                            subject_index,
                            class_index,
                            time_index,
                        ] = matrix
                        rows.append(row)
    if not np.isfinite(full).all() or not np.isfinite(split).all():
        raise TemporalNumericalError("UNASSESSED_NUMERICAL_FAILURE: nonfinite mean bank")
    full.setflags(write=False)
    split.setflags(write=False)
    return MeanSequenceBank(full=full, split=split, diagnostics=pd.DataFrame(rows))


def pairwise_airm_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compute a finite nonnegative cross-bank AIRM distance matrix."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:]:
        raise ValueError("distance banks must have compatible (n,d,d) shapes")
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for index, matrix in enumerate(left):
        result[index] = np.asarray(airm_distance(matrix, right), dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise TemporalNumericalError(
            "UNASSESSED_NUMERICAL_FAILURE: required AIRM distances are invalid"
        )
    return result


def compute_split_half_reliability(split_means: np.ndarray) -> np.ndarray:
    """Return 72 Half-A-to-Half-B cross-position 5x5 matrices."""

    values = np.asarray(split_means, dtype=np.float64)
    expected = (N_HALVES, N_SESSIONS, N_SUBJECTS, N_CLASSES, N_TIMES, 22, 22)
    if values.shape != expected:
        raise ValueError(f"split means must have shape {expected}")
    result = np.empty(
        (N_SESSIONS, N_SUBJECTS, N_CLASSES, N_TIMES, N_TIMES),
        dtype=np.float64,
    )
    for session in range(N_SESSIONS):
        for subject in range(N_SUBJECTS):
            for class_index in range(N_CLASSES):
                result[session, subject, class_index] = pairwise_airm_distances(
                    values[0, session, subject, class_index],
                    values[1, session, subject, class_index],
                )
    return result


def compute_cross_time_matrices(full_means: np.ndarray) -> np.ndarray:
    """Return all 36x36 cross-session 5x5 AIRM distance matrices."""

    values = np.asarray(full_means, dtype=np.float64)
    expected = (N_SESSIONS, N_SUBJECTS, N_CLASSES, N_TIMES, 22, 22)
    if values.shape != expected:
        raise ValueError(f"full means must have shape {expected}")
    session0 = values[0].reshape(N_CELLS * N_TIMES, 22, 22)
    session1 = values[1].reshape(N_CELLS * N_TIMES, 22, 22)
    flat = pairwise_airm_distances(session0, session1)
    return flat.reshape(N_CELLS, N_TIMES, N_CELLS, N_TIMES).transpose(0, 2, 1, 3)


def all_matching_costs(k: np.ndarray) -> np.ndarray:
    """Compute root-mean-five costs for every frozen S5 row-to-column mapping."""

    matrices = np.asarray(k, dtype=np.float64)
    if matrices.shape[-2:] != (N_TIMES, N_TIMES):
        raise ValueError("K matrices must end in shape (5,5)")
    costs = np.empty(matrices.shape[:-2] + (len(ALL_PERMUTATIONS),), dtype=np.float64)
    row_indices = np.arange(N_TIMES)
    for permutation_index, permutation in enumerate(ALL_PERMUTATIONS):
        selected = matrices[..., row_indices, permutation]
        costs[..., permutation_index] = np.sqrt(np.mean(selected * selected, axis=-1))
    if not np.isfinite(costs).all() or np.any(costs < 0.0):
        raise TemporalNumericalError("UNASSESSED_NUMERICAL_FAILURE: matching costs invalid")
    return costs


def _composition_indices() -> np.ndarray:
    lookup = {tuple(value): index for index, value in enumerate(ALL_PERMUTATIONS.tolist())}
    composition = np.empty((120, 120), dtype=np.int64)
    for relabel_index, relabel in enumerate(ALL_PERMUTATIONS):
        for match_index, match in enumerate(ALL_PERMUTATIONS):
            composed = tuple(relabel[match].tolist())
            composition[relabel_index, match_index] = lookup[composed]
    return composition


COMPOSITION_INDICES = _composition_indices()


def summarize_matching(k: np.ndarray) -> MatchingResults:
    """Apply identity, exact derangement, rank, and temporal-relabel definitions."""

    matrices = np.asarray(k, dtype=np.float64)
    if matrices.shape != (N_CELLS, N_CELLS, N_TIMES, N_TIMES):
        raise ValueError("K must have the frozen (36,36,5,5) shape")
    costs = all_matching_costs(matrices)
    d_id = costs[..., IDENTITY_INDEX]
    derangement = costs[..., DERANGEMENT_INDICES]
    median_derangement = np.median(derangement, axis=-1)
    if np.any(median_derangement <= 0.0):
        raise TemporalNumericalError(
            "UNASSESSED_NUMERICAL_FAILURE: nonpositive derangement median"
        )
    gain = median_derangement - d_id
    relative = gain / median_derangement
    rank = 1 + np.sum(costs < d_id[..., None], axis=-1)
    ties = np.sum(costs == d_id[..., None], axis=-1)
    relabelled_gain = np.empty((N_CELLS, N_CELLS, 120), dtype=np.float64)
    for relabel_index in range(120):
        relabelled_costs = costs[..., COMPOSITION_INDICES[relabel_index]]
        relabelled_identity = costs[..., relabel_index]
        relabelled_median = np.median(
            relabelled_costs[..., DERANGEMENT_INDICES], axis=-1
        )
        relabelled_gain[..., relabel_index] = (
            relabelled_median - relabelled_identity
        )
    if not np.allclose(
        relabelled_gain[..., IDENTITY_INDEX], gain, atol=0.0, rtol=0.0
    ):
        raise RuntimeError("identity relabel does not reproduce observed gain exactly")
    return MatchingResults(
        k=matrices,
        all_costs=costs,
        d_id=d_id,
        median_derangement=median_derangement,
        gain=gain,
        relative_gain=relative,
        identity_rank=rank.astype(np.int64),
        identity_tie_count=ties.astype(np.int64),
        relabelled_gain=relabelled_gain,
    )


def temporal_contrasts(gain: np.ndarray) -> TemporalContrasts:
    """Compute frozen chronological correspondence and specificity contrasts."""

    values = np.asarray(gain, dtype=np.float64)
    if values.shape != (N_CELLS, N_CELLS) or not np.isfinite(values).all():
        raise ValueError("gain must be a finite 36x36 matrix")
    a = np.empty((N_SUBJECTS, N_CLASSES), dtype=np.float64)
    b = np.empty_like(a)
    c = np.empty_like(a)
    d = np.empty_like(a)
    for subject in range(N_SUBJECTS):
        for class_index in range(N_CLASSES):
            anchor = cell_index(subject, class_index)
            a[subject, class_index] = values[anchor, anchor]
            b[subject, class_index] = np.mean(
                [
                    values[anchor, cell_index(subject, other_class)]
                    for other_class in range(N_CLASSES)
                    if other_class != class_index
                ]
            )
            c[subject, class_index] = np.mean(
                [
                    values[anchor, cell_index(other_subject, class_index)]
                    for other_subject in range(N_SUBJECTS)
                    if other_subject != subject
                ]
            )
            d[subject, class_index] = np.mean(
                [
                    values[anchor, cell_index(other_subject, other_class)]
                    for other_subject in range(N_SUBJECTS)
                    if other_subject != subject
                    for other_class in range(N_CLASSES)
                    if other_class != class_index
                ]
            )
    s_sc = a - c
    c_sc = a - b
    j_sc = a - b - c + d
    a_s = np.mean(a, axis=1)
    s_s = np.mean(s_sc, axis=1)
    c_s = np.mean(c_sc, axis=1)
    j_s = np.mean(j_sc, axis=1)
    return TemporalContrasts(
        a_sc=a,
        b_sc=b,
        c_sc=c,
        d_sc=d,
        s_sc=s_sc,
        c_specific_sc=c_sc,
        j_sc=j_sc,
        a_s=a_s,
        s_s=s_s,
        c_s=c_s,
        j_s=j_s,
        t_temporal=float(np.mean(a_s)),
        t_subject=float(np.mean(s_s)),
        t_class=float(np.mean(c_s)),
        t_j=float(np.mean(j_s)),
    )


def _rng(master_seed: int, stream_tag: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master_seed, stream_tag]))


def temporal_permutation_indices(
    *,
    replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> np.ndarray:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    return _rng(master_seed, TEMPORAL_STREAM_TAG).integers(
        0, 120, size=(replicates, N_CELLS), dtype=np.int64
    )


def classbreak_mappings(
    *,
    replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> np.ndarray:
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
    if replicates < 1:
        raise ValueError("replicates must be positive")
    generator = _rng(master_seed, SUBJECTBREAK_STREAM_TAG)
    mappings = np.empty((replicates, N_CELLS), dtype=np.int64)
    for replicate in range(replicates):
        for class_index in range(N_CLASSES):
            permutation = generator.permutation(N_SUBJECTS)
            for assigned_subject in range(N_SUBJECTS):
                mappings[replicate, cell_index(assigned_subject, class_index)] = cell_index(
                    int(permutation[assigned_subject]), class_index
                )
    return mappings


def plus_one_pvalue(observed: float, null: np.ndarray) -> float:
    values = np.asarray(null, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("null must be a finite nonempty vector")
    return float((1 + np.count_nonzero(values >= observed)) / (1 + len(values)))


def terminal_decision(
    *,
    t_temporal: float,
    p_temporal: float,
    t_subject: float,
    p_subject: float,
    t_class: float,
    p_class: float,
    alpha: float = 0.05,
) -> str:
    temporal_pass = t_temporal > 0.0 and p_temporal < alpha
    if not temporal_pass:
        return "STOP_NO_REPRODUCIBLE_TEMPORAL_SEQUENCE_V0"
    subject_pass = t_subject > 0.0 and p_subject < alpha
    class_pass = t_class > 0.0 and p_class < alpha
    if subject_pass and class_pass:
        return "GO_REPRODUCIBLE_SUBJECT_CLASS_TEMPORAL_SEQUENCE"
    if subject_pass or class_pass:
        return "GO_SHARED_TEMPORAL_SEQUENCE_WITH_PARTIAL_SPECIFICITY"
    return "GO_SHARED_TEMPORAL_SEQUENCE_ONLY"


def evaluate_temporal_inference(
    matching: MatchingResults,
    *,
    replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> TemporalInference:
    """Evaluate the frozen temporal, subject-break, and class-break nulls."""

    observed = temporal_contrasts(matching.gain)
    temporal_maps = temporal_permutation_indices(
        replicates=replicates, master_seed=master_seed
    )
    subject_maps = subjectbreak_mappings(replicates=replicates, master_seed=master_seed)
    class_maps = classbreak_mappings(replicates=replicates, master_seed=master_seed)
    temporal_null = np.empty(replicates, dtype=np.float64)
    subject_t = np.empty(replicates, dtype=np.float64)
    subject_j = np.empty(replicates, dtype=np.float64)
    class_t = np.empty(replicates, dtype=np.float64)
    class_j = np.empty(replicates, dtype=np.float64)
    diagonal_cells = np.arange(N_CELLS)
    for replicate in range(replicates):
        temporal_null[replicate] = float(
            np.mean(
                matching.relabelled_gain[
                    diagonal_cells, diagonal_cells, temporal_maps[replicate]
                ]
            )
        )
        subject_result = temporal_contrasts(
            matching.gain[:, subject_maps[replicate]]
        )
        subject_t[replicate] = subject_result.t_subject
        subject_j[replicate] = subject_result.t_j
        class_result = temporal_contrasts(matching.gain[:, class_maps[replicate]])
        class_t[replicate] = class_result.t_class
        class_j[replicate] = class_result.t_j
    p_temporal = plus_one_pvalue(observed.t_temporal, temporal_null)
    p_subject = plus_one_pvalue(observed.t_subject, subject_t)
    p_class = plus_one_pvalue(observed.t_class, class_t)
    p_j_subject = plus_one_pvalue(observed.t_j, subject_j)
    p_j_class = plus_one_pvalue(observed.t_j, class_j)
    terminal = terminal_decision(
        t_temporal=observed.t_temporal,
        p_temporal=p_temporal,
        t_subject=observed.t_subject,
        p_subject=p_subject,
        t_class=observed.t_class,
        p_class=p_class,
    )
    return TemporalInference(
        observed=observed,
        temporal_null=temporal_null,
        subjectbreak_t_subject=subject_t,
        subjectbreak_t_j=subject_j,
        classbreak_t_class=class_t,
        classbreak_t_j=class_j,
        temporal_permutation_indices=temporal_maps,
        subject_mappings=subject_maps,
        class_mappings=class_maps,
        p_temporal=p_temporal,
        p_subject=p_subject,
        p_class=p_class,
        p_j_subjectbreak=p_j_subject,
        p_j_classbreak=p_j_class,
        terminal=terminal,
    )


def compute_common_pca(full_means: np.ndarray) -> CommonPCA:
    """Fit the one frozen common-reference, common-basis visualization PCA."""

    values = np.asarray(full_means, dtype=np.float64)
    expected = (N_SESSIONS, N_SUBJECTS, N_CLASSES, N_TIMES, 22, 22)
    if values.shape != expected:
        raise ValueError(f"full means must have shape {expected}")
    flat = values.reshape(-1, 22, 22)
    reference, row = _mean_record(
        flat,
        scope="Global_visualization_reference",
        subject="ALL",
        session="BOTH",
        class_label="ALL",
        temporal_position=0,
    )
    whitening = spd_invsqrt(reference)
    whitened = symmetrize(whitening @ flat @ whitening)
    features = svec(spd_log(whitened), check_symmetric=False)
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(features).reshape(
        N_SESSIONS, N_SUBJECTS, N_CLASSES, N_TIMES, 2
    )
    return CommonPCA(
        reference=reference,
        coordinates=coordinates,
        components=np.asarray(pca.components_, dtype=np.float64),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
        explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ),
        feature_mean=np.asarray(pca.mean_, dtype=np.float64),
        reference_diagnostic=row,
    )


def group_average_k(k: np.ndarray) -> dict[str, np.ndarray]:
    """Return the four predeclared descriptive category-average K matrices."""

    values = np.asarray(k, dtype=np.float64)
    if values.shape != (N_CELLS, N_CELLS, N_TIMES, N_TIMES):
        raise ValueError("K must have shape (36,36,5,5)")
    labels = [
        (subject, class_index)
        for subject in range(N_SUBJECTS)
        for class_index in range(N_CLASSES)
    ]
    groups: dict[str, list[np.ndarray]] = {
        "same_subject_same_class": [],
        "different_subject_same_class": [],
        "same_subject_different_class": [],
        "different_subject_different_class": [],
    }
    for row, (subject, class_index) in enumerate(labels):
        for column, (other_subject, other_class) in enumerate(labels):
            same_subject = subject == other_subject
            same_class = class_index == other_class
            if same_subject and same_class:
                key = "same_subject_same_class"
            elif not same_subject and same_class:
                key = "different_subject_same_class"
            elif same_subject and not same_class:
                key = "same_subject_different_class"
            else:
                key = "different_subject_different_class"
            groups[key].append(values[row, column])
    return {key: np.mean(group, axis=0) for key, group in groups.items()}


__all__ = [
    "ALL_PERMUTATIONS",
    "CLASS_ORDER",
    "CLASSBREAK_STREAM_TAG",
    "COMPOSITION_INDICES",
    "CommonPCA",
    "DEFAULT_MASTER_SEED",
    "DEFAULT_NULL_REPLICATES",
    "DERANGEMENT_INDICES",
    "DERANGEMENT_MASK",
    "HALF_ORDER",
    "HALF_RUNS",
    "IDENTITY_INDEX",
    "IDENTITY_PERMUTATION",
    "MatchingResults",
    "MeanSequenceBank",
    "N_CELLS",
    "N_CLASSES",
    "N_SUBJECTS",
    "N_TIMES",
    "NORMALIZED_RESIDUAL_MAX",
    "SESSION_ORDER",
    "SUBJECTBREAK_STREAM_TAG",
    "TEMPORAL_STREAM_TAG",
    "TemporalContrasts",
    "TemporalInference",
    "TemporalNumericalError",
    "all_matching_costs",
    "cell_index",
    "cell_labels",
    "classbreak_mappings",
    "compute_common_pca",
    "compute_cross_time_matrices",
    "compute_split_half_reliability",
    "evaluate_temporal_inference",
    "fit_ordered_mean_sequences",
    "group_average_k",
    "pairwise_airm_distances",
    "plus_one_pvalue",
    "subjectbreak_mappings",
    "summarize_matching",
    "temporal_contrasts",
    "temporal_permutation_indices",
    "terminal_decision",
]
