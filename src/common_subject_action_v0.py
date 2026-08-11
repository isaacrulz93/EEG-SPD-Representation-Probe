"""Frozen-design primitives for the common subject action falsification audit."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.common_action_solver_v0 import MASTER_SEED, normalized_prediction_error, symmetrize
from src.interaction_statistics_v0 import all_derangements, derangement_statistics, monte_carlo_summary
from src.spd_utils import svec


SUBJECTS = tuple(range(1, 10))
SESSIONS = ("0train", "1test")
CLASSES = ("left_hand", "right_hand", "feet", "tongue")
HALF_RUNS = {"A": ("0", "1", "2"), "B": ("3", "4", "5")}
NULL_REPLICATES = 1999

STREAM_TAGS = {
    "stage_A_unrelated": 0x4353415F415F5552,
    "stage_A_semantic": 0x4353415F415F534D,
    "stage_B_unrelated": 0x4353415F425F5552,
    "stage_B_semantic": 0x4353415F425F534D,
    "residual_class_correspondence": 0x4353415F52455343,
}


class AuditContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocoFold:
    target_subject_index: int
    heldout_class_index: int
    source_subject_indices: tuple[int, ...]
    fit_class_indices: tuple[int, ...]
    anchor_source_position: int


@dataclass(frozen=True)
class FrozenDecision:
    decision: str
    stage_a_pass: bool | None
    stage_b_pass: bool | None
    stage_c_pass: bool | None


def loco_folds(
    n_subjects: int = 9,
    n_classes: int = 4,
) -> tuple[LocoFold, ...]:
    folds: list[LocoFold] = []
    for target in range(int(n_subjects)):
        sources = tuple(index for index in range(int(n_subjects)) if index != target)
        if target in sources or len(sources) != int(n_subjects) - 1:
            raise AuditContractError("target subject leaked into source fold")
        anchor = int(np.argmin(sources))
        for heldout in range(int(n_classes)):
            fit = tuple(index for index in range(int(n_classes)) if index != heldout)
            folds.append(LocoFold(target, heldout, sources, fit, anchor))
    if len(folds) != int(n_subjects) * int(n_classes):
        raise AuditContractError("LOCO fold grid is incomplete")
    return tuple(folds)


def assert_fold_no_leakage(fold: LocoFold, *, n_subjects: int = 9, n_classes: int = 4) -> None:
    if fold.target_subject_index in fold.source_subject_indices:
        raise AuditContractError("target subject is present in source fit")
    if fold.heldout_class_index in fold.fit_class_indices:
        raise AuditContractError("held-out class is present in target/source action fit")
    if set(fold.source_subject_indices) != set(range(n_subjects)) - {fold.target_subject_index}:
        raise AuditContractError("source subject set is not exhaustive LOSO")
    if set(fold.fit_class_indices) != set(range(n_classes)) - {fold.heldout_class_index}:
        raise AuditContractError("fit class set is not exhaustive LOCO")
    if fold.source_subject_indices[fold.anchor_source_position] != min(fold.source_subject_indices):
        raise AuditContractError("global gauge anchor is not the lowest-ID source subject")


def raw_population_prediction(source_heldout: np.ndarray) -> np.ndarray:
    values = np.asarray(source_heldout, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != values.shape[-2] or len(values) < 1:
        raise ValueError("source held-out objects must have shape (source,d,d)")
    return symmetrize(np.mean(values, axis=0, dtype=np.float64))


def error_and_gain(target: np.ndarray, raw_prediction: np.ndarray, action_prediction: np.ndarray) -> tuple[float, float, float]:
    raw_error = normalized_prediction_error(target, raw_prediction)
    action_error = normalized_prediction_error(target, action_prediction)
    return raw_error, action_error, float(raw_error - action_error)


def subject_group_statistic(
    cell_values: np.ndarray,
    subject_indices: np.ndarray,
    *,
    expected_cells_per_subject: int = 8,
    n_subjects: int = 9,
) -> tuple[np.ndarray, float]:
    values = np.asarray(cell_values, dtype=np.float64)
    subjects = np.asarray(subject_indices, dtype=np.int64)
    if values.ndim != 1 or subjects.shape != values.shape or not np.isfinite(values).all():
        raise AuditContractError("cell values/subjects must be aligned finite vectors")
    scores = np.empty(int(n_subjects), dtype=np.float64)
    for subject in range(int(n_subjects)):
        selected = values[subjects == subject]
        if len(selected) != int(expected_cells_per_subject):
            raise AuditContractError(
                f"subject {subject} has {len(selected)} cells, expected {expected_cells_per_subject}; available-case aggregation is forbidden"
            )
        scores[subject] = float(np.median(selected))
    return scores, float(np.median(scores))


def seed_vector(stream: str, replicates: int = NULL_REPLICATES) -> np.ndarray:
    if stream not in STREAM_TAGS:
        raise ValueError(f"unknown RNG stream: {stream}")
    children = np.random.SeedSequence([MASTER_SEED, STREAM_TAGS[stream]]).spawn(int(replicates))
    return np.asarray(
        [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children],
        dtype=np.uint64,
    )


def comparator_choice_matrix(
    stream: str,
    *,
    n_cells: int,
    choices_per_cell: int | Sequence[int],
    replicates: int = NULL_REPLICATES,
) -> np.ndarray:
    counts = (
        np.full(int(n_cells), int(choices_per_cell), dtype=np.int64)
        if isinstance(choices_per_cell, (int, np.integer))
        else np.asarray(choices_per_cell, dtype=np.int64)
    )
    if counts.shape != (int(n_cells),) or np.any(counts < 1):
        raise ValueError("each cell must have at least one comparator")
    seeds = seed_vector(stream, replicates)
    output = np.empty((int(replicates), int(n_cells)), dtype=np.int64)
    for replicate, seed in enumerate(seeds):
        rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(int(seed))))
        for cell in range(int(n_cells)):
            output[replicate, cell] = int(rng.integers(0, counts[cell]))
    return output


def comparator_null_statistics(
    comparator_gains: Sequence[np.ndarray],
    subject_indices: np.ndarray,
    *,
    stream: str,
    expected_cells_per_subject: int = 8,
    replicates: int = NULL_REPLICATES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    banks = tuple(np.asarray(values, dtype=np.float64) for values in comparator_gains)
    if not banks or any(values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all() for values in banks):
        raise AuditContractError("every cell requires a finite nonempty comparator bank")
    choices = comparator_choice_matrix(
        stream,
        n_cells=len(banks),
        choices_per_cell=[len(values) for values in banks],
        replicates=replicates,
    )
    subjects = np.asarray(subject_indices, dtype=np.int64)
    if subjects.shape != (len(banks),):
        raise AuditContractError("subject indices do not match comparator cells")
    subject_stats = np.empty((int(replicates), len(SUBJECTS)), dtype=np.float64)
    group_stats = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        selected = np.asarray([banks[cell][choices[replicate, cell]] for cell in range(len(banks))])
        subject_stats[replicate], group_stats[replicate] = subject_group_statistic(
            selected,
            subjects,
            expected_cells_per_subject=expected_cells_per_subject,
            n_subjects=len(SUBJECTS),
        )
    return subject_stats, group_stats, choices


def stage_pass(
    observed: float,
    unrelated_null: np.ndarray,
    semantic_null: np.ndarray,
    *,
    prerequisite: bool = True,
) -> tuple[bool, Mapping[str, object]]:
    unrelated = monte_carlo_summary(float(observed), np.asarray(unrelated_null, dtype=np.float64))
    semantic = monte_carlo_summary(float(observed), np.asarray(semantic_null, dtype=np.float64))
    passed = bool(
        prerequisite
        and float(observed) > 0.0
        and unrelated.effect > 0.0
        and unrelated.p_value <= 0.05
        and semantic.effect > 0.0
        and semantic.p_value <= 0.05
    )
    return passed, {"unrelated": unrelated, "semantic": semantic}


def normalized_residual_signature(residuals: np.ndarray) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != len(CLASSES) or values.shape[-1] != values.shape[-2]:
        raise AuditContractError("residuals must have fixed (4,d,d) semantic order")
    vector = svec(symmetrize(values)).reshape(-1)
    norm = float(np.linalg.norm(vector))
    threshold = float(np.finfo(np.float64).eps * np.sqrt(vector.size))
    if norm <= threshold:
        raise AuditContractError("post-action residual signature is numerical zero")
    return vector / norm


def residual_same_subject_test(session0: np.ndarray, session1: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    left = np.asarray(session0, dtype=np.float64)
    right = np.asarray(session1, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or len(left) != len(SUBJECTS):
        raise AuditContractError("residual signatures must share (9,feature) shape")
    similarity = left @ right.T
    same = np.diag(similarity).copy()
    observed = float(np.median(same))
    null = derangement_statistics(similarity, all_derangements(len(SUBJECTS)))
    return same, observed, null


def residual_class_correspondence_null(
    residuals_session0: np.ndarray,
    residuals_session1: np.ndarray,
    *,
    replicates: int = NULL_REPLICATES,
) -> tuple[float, np.ndarray, np.ndarray]:
    first = np.asarray(residuals_session0, dtype=np.float64)
    second = np.asarray(residuals_session1, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 4 or first.shape[:2] != (len(SUBJECTS), len(CLASSES)):
        raise AuditContractError("residual blocks must have shape (9,4,d,d)")
    left = np.stack([normalized_residual_signature(value) for value in first])
    right = np.stack([normalized_residual_signature(value) for value in second])
    observed = float(np.median(np.sum(left * right, axis=1)))
    permutations = tuple(value for value in itertools.permutations(range(len(CLASSES))) if value != tuple(range(len(CLASSES))))
    if len(permutations) != 23:
        raise RuntimeError("S4 nonidentity enumeration failure")
    seeds = seed_vector("residual_class_correspondence", replicates)
    null = np.empty(int(replicates), dtype=np.float64)
    selected_indices = np.empty((int(replicates), len(SUBJECTS)), dtype=np.int64)
    for replicate, seed in enumerate(seeds):
        rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(int(seed))))
        subject_scores = np.empty(len(SUBJECTS), dtype=np.float64)
        for subject in range(len(SUBJECTS)):
            choice = int(rng.integers(0, len(permutations)))
            selected_indices[replicate, subject] = choice
            permuted = second[subject, permutations[choice]]
            signature = normalized_residual_signature(permuted)
            subject_scores[subject] = float(np.dot(left[subject], signature))
        null[replicate] = float(np.median(subject_scores))
    return observed, null, selected_indices


def terminal_decision(
    *,
    data_gate_pass: bool,
    technical_gate_pass: bool,
    identifiable: bool,
    stage_a_pass: bool | None,
    stage_b_pass: bool | None,
    stage_c_pass: bool | None,
) -> FrozenDecision:
    if not data_gate_pass:
        decision = "UNASSESSED_NUMERICAL_OR_DATA_FAILURE"
    elif not technical_gate_pass:
        decision = "UNASSESSED_TECHNICAL_FAILURE"
    elif not identifiable:
        decision = "UNASSESSED_ACTION_NOT_IDENTIFIABLE"
    elif stage_a_pass is not True:
        decision = "COMMON_ACTION_NOT_SUPPORTED_WITHIN_SESSION"
    elif stage_b_pass is not True:
        decision = "SESSION_SPECIFIC_COMMON_ACTION_ONLY"
    elif stage_c_pass is True:
        decision = "COMMON_ACTION_SUPPORTED_RESIDUAL_INDIVIDUALITY_REMAINS"
    else:
        decision = "COMMON_ACTION_SUPPORTED_NO_STABLE_RESIDUAL_EVIDENCE"
    return FrozenDecision(decision, stage_a_pass, stage_b_pass, stage_c_pass)


def validate_direction_grid(rows: Sequence[Mapping[str, object]], stage: str) -> None:
    expected = 72
    if len(rows) != expected:
        raise AuditContractError(f"Stage {stage} grid has {len(rows)} rows, expected {expected}")
    keys = set()
    for row in rows:
        key = (int(row["subject"]), str(row["direction"]), str(row["heldout_class"]))
        if key in keys:
            raise AuditContractError(f"duplicate Stage {stage} cell: {key}")
        keys.add(key)
        if row.get("status") != "PASS":
            raise AuditContractError(f"Stage {stage} contains a required failed cell")


def required_identifiability_gate(
    classifications: Sequence[str],
    *,
    stage: str,
    expected_cells: int = 72,
) -> str:
    """Fail closed across every required Stage-A/Stage-B cell."""

    values = tuple(str(value) for value in classifications)
    if len(values) != int(expected_cells):
        raise AuditContractError(
            f"Stage {stage} identifiability grid has {len(values)} cells, "
            f"expected {expected_cells}; available-case analysis is forbidden"
        )
    allowed = {
        "PREDICTIVELY_IDENTIFIABLE",
        "HARMLESS_Q_NONUNIQUENESS",
        "PREDICTIVE_NONIDENTIFIABILITY",
    }
    if any(value not in allowed for value in values):
        return "UNASSESSED_TECHNICAL_FAILURE"
    if any(value == "PREDICTIVE_NONIDENTIFIABILITY" for value in values):
        return "UNASSESSED_ACTION_NOT_IDENTIFIABLE"
    return "PASS"


__all__ = [
    "AuditContractError",
    "CLASSES",
    "HALF_RUNS",
    "LocoFold",
    "NULL_REPLICATES",
    "SESSIONS",
    "STREAM_TAGS",
    "SUBJECTS",
    "comparator_choice_matrix",
    "comparator_null_statistics",
    "error_and_gain",
    "loco_folds",
    "normalized_residual_signature",
    "raw_population_prediction",
    "residual_class_correspondence_null",
    "residual_same_subject_test",
    "required_identifiability_gate",
    "seed_vector",
    "stage_pass",
    "subject_group_statistic",
    "terminal_decision",
    "validate_direction_grid",
]
