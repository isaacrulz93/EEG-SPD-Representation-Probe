"""Frozen real-data computations for local AIRM metric interaction V0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.local_metric_geometry_v0 import (
    cross_all_configuration_distances,
    cross_configuration_distances,
)
from src.local_metric_interaction_v0 import (
    CLASS_ORDER,
    DEFAULT_MASTER_SEED,
    DEFAULT_NULL_REPLICATES,
    N_CELLS,
    N_CLASSES,
    N_SUBJECTS,
    cell_index,
    plus_one_pvalue,
    symmetrize_session_roles,
)


SESSION_ORDER = ("0train", "1test")
DECODING_STREAM_TAG = 0x4C4D494445434F44


@dataclass(frozen=True)
class CellMetricMatrices:
    raw_m01: np.ndarray
    raw_m: np.ndarray
    size_m01: np.ndarray
    size_m: np.ndarray
    normalized_m01: np.ndarray
    normalized_m: np.ndarray


@dataclass(frozen=True)
class DecodingResult:
    subject_scores: pd.DataFrame
    group_mean_ba: float
    group_median_ba: float
    null_group_median_ba: np.ndarray
    p_value: float


def _validate_inputs(edges: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    values = np.asarray(edges, dtype=np.float64)
    frame = metadata.reset_index(drop=True).copy()
    if values.shape != (5184, 10) or len(frame) != 5184:
        raise ValueError("frozen input must contain exactly 5,184 ten-edge trials")
    required = {"subject", "session", "run", "class_label", "global_sample_index", "trial_uid"}
    if required - set(frame.columns):
        raise ValueError("metadata is missing a frozen identity column")
    if not np.array_equal(frame["global_sample_index"].to_numpy(), np.arange(5184)):
        raise ValueError("metadata and edge rows are not in frozen global order")
    return values, frame


def _cell_indices(frame: pd.DataFrame, session: str, subject: int, class_label: str) -> np.ndarray:
    indices = frame.index[
        frame["session"].eq(session)
        & frame["subject"].eq(subject)
        & frame["class_label"].eq(class_label)
    ].to_numpy(dtype=np.int64)
    if indices.shape != (72,):
        raise ValueError(
            f"cell {subject}/{session}/{class_label} has {len(indices)} rather than 72 trials"
        )
    return indices


def compute_cell_metric_matrices(
    edges: np.ndarray,
    metadata: pd.DataFrame,
    *,
    chunk_size: int = 128,
) -> CellMetricMatrices:
    values, frame = _validate_inputs(edges, metadata)
    cells0: list[np.ndarray] = []
    cells1: list[np.ndarray] = []
    for subject in range(1, N_SUBJECTS + 1):
        for class_label in CLASS_ORDER:
            cells0.append(_cell_indices(frame, SESSION_ORDER[0], subject, class_label))
            cells1.append(_cell_indices(frame, SESSION_ORDER[1], subject, class_label))
    raw_m01 = np.empty((N_CELLS, N_CELLS), dtype=np.float64)
    size_m01 = np.empty_like(raw_m01)
    normalized_m01 = np.empty_like(raw_m01)
    for row, left_indices in enumerate(cells0):
        for column, right_indices in enumerate(cells1):
            raw, size, normalized = cross_all_configuration_distances(
                values[left_indices], values[right_indices], chunk_size=chunk_size
            )
            raw_m01[row, column] = float(np.median(raw))
            size_m01[row, column] = float(np.median(size))
            normalized_m01[row, column] = float(np.median(normalized))
    return CellMetricMatrices(
        raw_m01=raw_m01,
        raw_m=symmetrize_session_roles(raw_m01),
        size_m01=size_m01,
        size_m=symmetrize_session_roles(size_m01),
        normalized_m01=normalized_m01,
        normalized_m=symmetrize_session_roles(normalized_m01),
    )


def compute_cell_metric_matrix(
    edges: np.ndarray,
    metadata: pd.DataFrame,
    *,
    mode: str,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one preregistered metric stage without evaluating later stages."""

    if mode not in {"raw", "size", "normalized"}:
        raise ValueError("mode must be raw, size, or normalized")
    values, frame = _validate_inputs(edges, metadata)
    cells0: list[np.ndarray] = []
    cells1: list[np.ndarray] = []
    for subject in range(1, N_SUBJECTS + 1):
        for class_label in CLASS_ORDER:
            cells0.append(_cell_indices(frame, SESSION_ORDER[0], subject, class_label))
            cells1.append(_cell_indices(frame, SESSION_ORDER[1], subject, class_label))
    m01 = np.empty((N_CELLS, N_CELLS), dtype=np.float64)
    for row, left_indices in enumerate(cells0):
        for column, right_indices in enumerate(cells1):
            distances = cross_configuration_distances(
                values[left_indices],
                values[right_indices],
                mode=mode,  # type: ignore[arg-type]
                chunk_size=chunk_size,
            )
            m01[row, column] = float(np.median(distances))
    return m01, symmetrize_session_roles(m01)


def compute_within_cell_diagnostics(
    edges: np.ndarray,
    metadata: pd.DataFrame,
    *,
    chunk_size: int = 128,
) -> pd.DataFrame:
    values, frame = _validate_inputs(edges, metadata)
    rows: list[dict[str, object]] = []
    for subject in range(1, N_SUBJECTS + 1):
        for session in SESSION_ORDER:
            for class_label in CLASS_ORDER:
                indices = _cell_indices(frame, session, subject, class_label)
                raw, _, _ = cross_all_configuration_distances(
                    values[indices], values[indices], chunk_size=chunk_size
                )
                triangle = raw[np.triu_indices(len(indices), k=1)]
                objective_matrix = raw.copy()
                np.fill_diagonal(objective_matrix, np.inf)
                # There are 71 off-diagonal distances. Its median is the
                # zero-based 35th order statistic after excluding self.
                objectives = np.partition(objective_matrix, 35, axis=1)[:, 35]
                best = float(np.min(objectives))
                tied = np.flatnonzero(
                    np.isclose(objectives, best, atol=1.0e-15, rtol=1.0e-15)
                )
                medoid_local = int(tied[0])
                medoid_global = int(indices[medoid_local])
                rows.append(
                    {
                        "subject": subject,
                        "session": session,
                        "class_label": class_label,
                        "n_trials": len(indices),
                        "median_within_cell_delta_raw": float(np.median(triangle)),
                        "q25_within_cell_delta_raw": float(np.quantile(triangle, 0.25)),
                        "q75_within_cell_delta_raw": float(np.quantile(triangle, 0.75)),
                        "iqr_within_cell_delta_raw": float(np.quantile(triangle, 0.75) - np.quantile(triangle, 0.25)),
                        "medoid_objective": best,
                        "medoid_global_sample_index": medoid_global,
                        "medoid_trial_uid": str(frame.loc[medoid_global, "trial_uid"]),
                    }
                )
    result = pd.DataFrame(rows)
    if result.shape[0] != 72 or not (result["n_trials"] == 72).all():
        raise RuntimeError("within-cell diagnostic grid is incomplete")
    return result


def class_medoid_indices(
    within_distance: np.ndarray,
    labels: Sequence[str] | np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(within_distance, dtype=np.float64)
    values = np.asarray(labels).astype(str)
    if matrix.shape != (len(values), len(values)) or len(values) == 0:
        raise ValueError("within-distance matrix and labels are incompatible")
    medoids = np.empty(N_CLASSES, dtype=np.int64)
    for class_index, class_label in enumerate(CLASS_ORDER):
        positions = np.flatnonzero(values == class_label)
        if len(positions) != 72:
            raise ValueError(f"class {class_label} must contain exactly 72 trials")
        block = matrix[np.ix_(positions, positions)]
        objective_matrix = block.copy()
        np.fill_diagonal(objective_matrix, np.inf)
        median_position = 35
        objectives = np.partition(
            objective_matrix, median_position, axis=1
        )[:, median_position]
        best = float(np.min(objectives))
        tied = np.flatnonzero(
            np.isclose(objectives, best, atol=1.0e-15, rtol=1.0e-15)
        )
        medoids[class_index] = int(positions[int(tied[0])])
    return medoids


def _balanced_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    truth_values = np.asarray(truth).astype(str)
    predicted_values = np.asarray(predicted).astype(str)
    recalls = []
    for class_label in CLASS_ORDER:
        selected = truth_values == class_label
        if np.count_nonzero(selected) == 0:
            raise ValueError("balanced accuracy received an empty frozen class")
        recalls.append(float(np.mean(predicted_values[selected] == class_label)))
    return float(np.mean(recalls))


def _direction_score(
    within_train: np.ndarray,
    train_labels: np.ndarray,
    train_to_test: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    medoids = class_medoid_indices(within_train, train_labels)
    predictions = np.asarray(CLASS_ORDER)[
        np.argmin(train_to_test[medoids, :], axis=0)
    ]
    return _balanced_accuracy(test_labels, predictions)


def _permute_within_runs(
    labels: np.ndarray,
    runs: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    result = np.asarray(labels).astype(str).copy()
    run_values = np.asarray(runs, dtype=np.int64)
    for run in range(6):
        positions = np.flatnonzero(run_values == run)
        if positions.shape != (48,):
            raise ValueError("each subject-session run must contain 48 trials")
        result[positions] = result[positions][generator.permutation(len(positions))]
    return result


def compute_cross_session_medoid_decoding(
    edges: np.ndarray,
    metadata: pd.DataFrame,
    *,
    null_replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
    chunk_size: int = 128,
) -> DecodingResult:
    values, frame = _validate_inputs(edges, metadata)
    subject_objects: list[dict[str, np.ndarray]] = []
    observed_rows: list[dict[str, float | int]] = []
    for subject in range(1, N_SUBJECTS + 1):
        indices0 = frame.index[
            frame["subject"].eq(subject) & frame["session"].eq(SESSION_ORDER[0])
        ].to_numpy(dtype=np.int64)
        indices1 = frame.index[
            frame["subject"].eq(subject) & frame["session"].eq(SESSION_ORDER[1])
        ].to_numpy(dtype=np.int64)
        if indices0.shape != (288,) or indices1.shape != (288,):
            raise ValueError("each subject/session must contain 288 trials")
        within0, _, _ = cross_all_configuration_distances(
            values[indices0], values[indices0], chunk_size=chunk_size
        )
        within1, _, _ = cross_all_configuration_distances(
            values[indices1], values[indices1], chunk_size=chunk_size
        )
        cross01, _, _ = cross_all_configuration_distances(
            values[indices0], values[indices1], chunk_size=chunk_size
        )
        labels0 = frame.loc[indices0, "class_label"].to_numpy(dtype=str)
        labels1 = frame.loc[indices1, "class_label"].to_numpy(dtype=str)
        runs0 = frame.loc[indices0, "run"].to_numpy(dtype=np.int64)
        runs1 = frame.loc[indices1, "run"].to_numpy(dtype=np.int64)
        score01 = _direction_score(within0, labels0, cross01, labels1)
        score10 = _direction_score(within1, labels1, cross01.T, labels0)
        average = 0.5 * (score01 + score10)
        observed_rows.append(
            {
                "subject": subject,
                "ba_session0_to_session1": score01,
                "ba_session1_to_session0": score10,
                "balanced_accuracy": average,
            }
        )
        subject_objects.append(
            {
                "within0": within0,
                "within1": within1,
                "cross01": cross01,
                "labels0": labels0,
                "labels1": labels1,
                "runs0": runs0,
                "runs1": runs1,
            }
        )
    subject_scores = pd.DataFrame(observed_rows)
    observed_median = float(np.median(subject_scores["balanced_accuracy"]))
    null_statistics = np.empty(null_replicates, dtype=np.float64)
    seeds = np.random.SeedSequence([master_seed, DECODING_STREAM_TAG]).spawn(
        null_replicates
    )
    for replicate, seed in enumerate(seeds):
        generator = np.random.default_rng(seed)
        subject_null = np.empty(N_SUBJECTS, dtype=np.float64)
        for subject_index, objects in enumerate(subject_objects):
            permuted0 = _permute_within_runs(
                objects["labels0"], objects["runs0"], generator
            )
            permuted1 = _permute_within_runs(
                objects["labels1"], objects["runs1"], generator
            )
            score01 = _direction_score(
                objects["within0"], permuted0, objects["cross01"], objects["labels1"]
            )
            score10 = _direction_score(
                objects["within1"], permuted1, objects["cross01"].T, objects["labels0"]
            )
            subject_null[subject_index] = 0.5 * (score01 + score10)
        null_statistics[replicate] = float(np.median(subject_null))
    return DecodingResult(
        subject_scores=subject_scores,
        group_mean_ba=float(np.mean(subject_scores["balanced_accuracy"])),
        group_median_ba=observed_median,
        null_group_median_ba=null_statistics,
        p_value=plus_one_pvalue(observed_median, null_statistics),
    )


__all__ = [
    "CellMetricMatrices",
    "DECODING_STREAM_TAG",
    "DecodingResult",
    "SESSION_ORDER",
    "class_medoid_indices",
    "compute_cell_metric_matrix",
    "compute_cell_metric_matrices",
    "compute_cross_session_medoid_decoding",
    "compute_within_cell_diagnostics",
]
