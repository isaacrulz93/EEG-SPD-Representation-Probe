"""Frozen-design primitives for the pairwise common-action amendment.

This module is array-only. It has no data loader and cannot access BNCI files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from src.common_action_solver_v0 import (
    CANDIDATE_SOLVER_SETTINGS,
    ActionFit,
    ActionSolverError,
    PredictiveIdentifiability,
    SolverSettings,
    StabilizerDiagnostic,
    analyze_common_stabilizer,
    classify_prediction_matrices,
    conjugate,
    deterministic_starts,
    diagnose_multistart,
    normalized_prediction_error,
    optimize_action,
    stabilizer_augmented_actions,
)
from src.interaction_statistics_v0 import all_derangements, monte_carlo_summary


MASTER_SEED = 20260810
NULL_REPLICATES = 1999
N_SUBJECTS = 9
N_SESSIONS = 2
N_CLASSES = 4
PAIRWISE_SETTINGS = replace(
    CANDIDATE_SOLVER_SETTINGS,
    starts=4,
    pymanopt_log_verbosity=1,
)

STREAM_TAGS = {
    "stage_A_unrelated_target": 0x5043415F415F5554,
    "stage_A_semantic": 0x5043415F415F534D,
    "stage_B_unrelated_target": 0x5043415F425F5554,
    "stage_B_semantic": 0x5043415F425F534D,
}


class PairwiseContractError(RuntimeError):
    """The pairwise confirmatory contract was violated."""


@dataclass(frozen=True)
class PairwiseFit:
    fit: ActionFit
    initial_actions: tuple[np.ndarray, ...]
    initial_determinants: tuple[int, ...]


@dataclass(frozen=True)
class PairwiseAssessment:
    stabilizer: StabilizerDiagnostic
    identifiability: PredictiveIdentifiability
    equivalent_actions: tuple[np.ndarray, ...]
    equivalent_predictions: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class StageGate:
    passed: bool
    observed: float
    null_median: float
    effect: float
    p_value: float
    exceedances: int
    replicates: int


def deterministic_seed(*parts: object) -> int:
    payload = json.dumps(
        [MASTER_SEED, *parts], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def fit_pairwise_action(
    target_fit: np.ndarray,
    source_fit: np.ndarray,
    *,
    seed_parts: Sequence[object],
    settings: SolverSettings = PAIRWISE_SETTINGS,
) -> PairwiseFit:
    """Fit exactly four total deterministic starts over both O(d) sectors."""

    if int(settings.starts) != 4:
        raise PairwiseContractError("pairwise amendment requires exactly four total starts")
    seed = deterministic_seed("pairwise_action", *seed_parts)
    starts = deterministic_starts(
        target_fit,
        source_fit,
        seed=seed,
        count=settings.starts,
    )
    if len(starts) != settings.starts:
        raise PairwiseContractError(
            f"actual_total_starts={len(starts)} != config.total_starts={settings.starts}"
        )
    initial_determinants = tuple(int(np.sign(np.linalg.det(value))) for value in starts)
    if initial_determinants.count(-1) != 2 or initial_determinants.count(1) != 2:
        raise PairwiseContractError(
            "four-start construction must contain exactly two starts per determinant sector"
        )
    fit = optimize_action(
        target_fit,
        source_fit,
        seed=seed,
        settings=settings,
        starts=starts,
        solver="pymanopt",
    )
    if len(fit.starts) != settings.starts:
        raise PairwiseContractError(
            f"runtime optimizer count {len(fit.starts)} != frozen total {settings.starts}"
        )
    landscape = diagnose_multistart(fit)
    if landscape.determinant_sectors_with_converged_solution != (-1, 1):
        diagnostics = "; ".join(
            (
                f"start={value.start_index},initial_det={initial_determinants[value.start_index]},"
                f"final_det={int(np.sign(value.determinant))},converged={value.converged},"
                f"objective={value.objective:.17g},gradient={value.gradient_norm:.17g},"
                f"iterations={value.iterations},stopping={value.stopping_criterion}"
            )
            for value in fit.starts
        )
        raise ActionSolverError(
            "UNASSESSED_TECHNICAL_FAILURE: pairwise fit lacks a converged "
            f"determinant sector; seed_parts={tuple(seed_parts)!r}; {diagnostics}"
        )
    return PairwiseFit(fit, starts, initial_determinants)


def assess_pairwise_prediction(
    pairwise_fit: PairwiseFit,
    target_fit_matrices: np.ndarray,
    source_fit_matrices: np.ndarray,
    source_heldout_matrix: np.ndarray,
    *,
    split_half_prediction_a: np.ndarray,
    split_half_prediction_b: np.ndarray,
    settings: SolverSettings = PAIRWISE_SETTINGS,
) -> PairwiseAssessment:
    """Classify the induced held-out prediction, never raw-R disagreement."""

    stabilizer = analyze_common_stabilizer(source_fit_matrices, settings=settings)
    actions = stabilizer_augmented_actions(
        pairwise_fit.fit,
        stabilizer,
        target_fit_matrices,
        source_fit_matrices,
        settings=settings,
    )
    predictions = tuple(conjugate(action, source_heldout_matrix) for action in actions)
    best_prediction = conjugate(pairwise_fit.fit.matrix, source_heldout_matrix)
    identifiability = classify_prediction_matrices(
        predictions,
        best_prediction=best_prediction,
        split_half_prediction_a=split_half_prediction_a,
        split_half_prediction_b=split_half_prediction_b,
        settings=settings,
    )
    return PairwiseAssessment(stabilizer, identifiability, actions, predictions)


def pairwise_error_gain(
    target_heldout: np.ndarray,
    source_heldout: np.ndarray,
    action_prediction: np.ndarray,
) -> tuple[float, float, float]:
    """Target-norm normalized raw/action errors and raw-minus-action gain."""

    raw_error = normalized_prediction_error(target_heldout, source_heldout)
    action_error = normalized_prediction_error(target_heldout, action_prediction)
    return raw_error, action_error, float(raw_error - action_error)


def _validate_pair_tensor(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    expected = (N_SUBJECTS, N_SUBJECTS, N_SESSIONS, N_CLASSES)
    if array.shape != expected:
        raise PairwiseContractError(f"pair tensor must have shape {expected}")
    diagonal = array[np.arange(N_SUBJECTS), np.arange(N_SUBJECTS)]
    if not np.isnan(diagonal).all():
        raise PairwiseContractError("target=source diagonal must be unavailable")
    mask = ~np.eye(N_SUBJECTS, dtype=bool)
    if not np.isfinite(array[mask]).all():
        raise PairwiseContractError(
            "every required off-diagonal pair must be finite; available-case analysis is forbidden"
        )
    return array


def aggregate_pairwise_gains(
    pair_gains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """sources -> target/session/class -> target -> group, all by median."""

    values = _validate_pair_tensor(pair_gains)
    target_cells = np.empty((N_SUBJECTS, N_SESSIONS, N_CLASSES), dtype=np.float64)
    for target in range(N_SUBJECTS):
        sources = [source for source in range(N_SUBJECTS) if source != target]
        target_cells[target] = np.median(values[target, sources], axis=0)
    subject_scores = np.median(target_cells.reshape(N_SUBJECTS, -1), axis=1)
    return target_cells, subject_scores, float(np.median(subject_scores))


def unrelated_target_gain_bank(
    U_target_session: np.ndarray,
    best_actions: np.ndarray,
) -> np.ndarray:
    """Precompute gains from applying another target's action for the same source.

    Parameters have shapes U=(target,session,class,d,d) and
    actions=(action_target,source,session,class,d,d). The output is
    (evaluation_target,source,session,class,action_target), with only entries
    having three distinct subject roles finite.
    """

    U = np.asarray(U_target_session, dtype=np.float64)
    actions = np.asarray(best_actions, dtype=np.float64)
    expected_u = (N_SUBJECTS, N_SESSIONS, N_CLASSES, U.shape[-1], U.shape[-1])
    expected_actions = (
        N_SUBJECTS,
        N_SUBJECTS,
        N_SESSIONS,
        N_CLASSES,
        U.shape[-1],
        U.shape[-1],
    )
    if U.shape != expected_u or actions.shape != expected_actions:
        raise PairwiseContractError("unrelated-target bank shapes are invalid")
    bank = np.full(
        (N_SUBJECTS, N_SUBJECTS, N_SESSIONS, N_CLASSES, N_SUBJECTS),
        np.nan,
        dtype=np.float64,
    )
    for target in range(N_SUBJECTS):
        for source in range(N_SUBJECTS):
            if source == target:
                continue
            for action_target in range(N_SUBJECTS):
                if action_target in (source, target):
                    continue
                for session in range(N_SESSIONS):
                    for heldout in range(N_CLASSES):
                        prediction = conjugate(
                            actions[action_target, source, session, heldout],
                            U[source, session, heldout],
                        )
                        _, _, gain = pairwise_error_gain(
                            U[target, session, heldout],
                            U[source, session, heldout],
                            prediction,
                        )
                        bank[target, source, session, heldout, action_target] = gain
    return bank


def unrelated_target_null_statistics(
    comparator_gains: np.ndarray,
    *,
    stream: str,
    replicates: int = NULL_REPLICATES,
) -> tuple[np.ndarray, np.ndarray]:
    """Derange target-specific actions within each fixed source/session/class."""

    if stream not in ("stage_A_unrelated_target", "stage_B_unrelated_target"):
        raise ValueError("invalid unrelated-target stream")
    bank = np.asarray(comparator_gains, dtype=np.float64)
    expected = (N_SUBJECTS, N_SUBJECTS, N_SESSIONS, N_CLASSES, N_SUBJECTS)
    if bank.shape != expected:
        raise PairwiseContractError(f"comparator bank must have shape {expected}")
    derangements = all_derangements(N_SUBJECTS - 1)
    children = np.random.SeedSequence([MASTER_SEED, STREAM_TAGS[stream]]).spawn(
        int(replicates)
    )
    statistics = np.empty(int(replicates), dtype=np.float64)
    choices = np.empty(
        (int(replicates), N_SUBJECTS, N_SESSIONS, N_CLASSES), dtype=np.int64
    )
    for replicate, child in enumerate(children):
        rng = np.random.Generator(np.random.PCG64DXSM(child))
        selected = np.full(
            (N_SUBJECTS, N_SUBJECTS, N_SESSIONS, N_CLASSES),
            np.nan,
            dtype=np.float64,
        )
        for source in range(N_SUBJECTS):
            targets = np.asarray(
                [target for target in range(N_SUBJECTS) if target != source],
                dtype=np.int64,
            )
            for session in range(N_SESSIONS):
                for heldout in range(N_CLASSES):
                    choice = int(rng.integers(0, len(derangements)))
                    choices[replicate, source, session, heldout] = choice
                    mapping = derangements[choice]
                    for position, target in enumerate(targets):
                        action_target = int(targets[mapping[position]])
                        value = bank[target, source, session, heldout, action_target]
                        if not np.isfinite(value):
                            raise PairwiseContractError(
                                "derangement selected an invalid comparator"
                            )
                        selected[target, source, session, heldout] = value
        _, _, statistics[replicate] = aggregate_pairwise_gains(selected)
    return statistics, choices


def semantic_null_statistics(
    mismatch_gains: np.ndarray,
    *,
    stream: str,
    replicates: int = NULL_REPLICATES,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one nonidentity S3 correspondence per target/session/class cell.

    The selected semantic permutation is shared across all eight sources for a
    target cell, preventing source pairs from becoming inferential replicates.
    """

    if stream not in ("stage_A_semantic", "stage_B_semantic"):
        raise ValueError("invalid semantic stream")
    values = np.asarray(mismatch_gains, dtype=np.float64)
    expected = (N_SUBJECTS, N_SUBJECTS, N_SESSIONS, N_CLASSES, 5)
    if values.shape != expected:
        raise PairwiseContractError(f"semantic bank must have shape {expected}")
    mask = ~np.eye(N_SUBJECTS, dtype=bool)
    if not np.isfinite(values[mask]).all():
        raise PairwiseContractError("semantic comparator grid is incomplete")
    children = np.random.SeedSequence([MASTER_SEED, STREAM_TAGS[stream]]).spawn(
        int(replicates)
    )
    statistics = np.empty(int(replicates), dtype=np.float64)
    choices = np.empty(
        (int(replicates), N_SUBJECTS, N_SESSIONS, N_CLASSES), dtype=np.int8
    )
    for replicate, child in enumerate(children):
        rng = np.random.Generator(np.random.PCG64DXSM(child))
        selected = np.full(
            (N_SUBJECTS, N_SUBJECTS, N_SESSIONS, N_CLASSES),
            np.nan,
            dtype=np.float64,
        )
        for target in range(N_SUBJECTS):
            for session in range(N_SESSIONS):
                for heldout in range(N_CLASSES):
                    choice = int(rng.integers(0, 5))
                    choices[replicate, target, session, heldout] = choice
                    for source in range(N_SUBJECTS):
                        if source != target:
                            selected[target, source, session, heldout] = values[
                                target, source, session, heldout, choice
                            ]
        _, _, statistics[replicate] = aggregate_pairwise_gains(selected)
    return statistics, choices


def evaluate_stage_gate(observed: float, null_statistics: np.ndarray) -> StageGate:
    summary = monte_carlo_summary(float(observed), np.asarray(null_statistics))
    passed = bool(
        summary.observed > 0.0
        and summary.effect > 0.0
        and summary.p_value <= 0.05
    )
    return StageGate(
        passed=passed,
        observed=summary.observed,
        null_median=summary.null_median,
        effect=summary.effect,
        p_value=summary.p_value,
        exceedances=summary.exceedances,
        replicates=summary.replicates,
    )


def terminal_decision(
    *,
    data_gate_pass: bool,
    technical_gate_pass: bool,
    identifiable: bool,
    stage_a_primary_pass: bool | None,
    stage_a_semantic_pass: bool | None,
    stage_b_primary_pass: bool | None,
    stage_b_semantic_pass: bool | None,
) -> str:
    if not data_gate_pass:
        return "UNASSESSED_NUMERICAL_OR_DATA_FAILURE"
    if not technical_gate_pass:
        return "UNASSESSED_TECHNICAL_FAILURE"
    if not identifiable:
        return "UNASSESSED_ACTION_NOT_IDENTIFIABLE"
    if stage_a_primary_pass is not True:
        return "PAIRWISE_COMMON_ACTION_NOT_SUPPORTED_WITHIN_SESSION"
    if stage_a_semantic_pass is not True:
        return "PAIRWISE_COMMON_ACTION_NOT_SEMANTICALLY_SUPPORTED"
    if stage_b_primary_pass is not True:
        return "PAIRWISE_COMMON_ACTION_WITHIN_SESSION_ONLY"
    if stage_b_semantic_pass is not True:
        return "PAIRWISE_COMMON_ACTION_NOT_CROSS_SESSION_SEMANTICALLY_SUPPORTED"
    return "PAIRWISE_COMMON_ACTION_NECESSARY_CONSEQUENCE_SUPPORTED"


__all__ = [
    "MASTER_SEED",
    "NULL_REPLICATES",
    "PAIRWISE_SETTINGS",
    "PairwiseAssessment",
    "PairwiseContractError",
    "PairwiseFit",
    "StageGate",
    "aggregate_pairwise_gains",
    "assess_pairwise_prediction",
    "deterministic_seed",
    "evaluate_stage_gate",
    "fit_pairwise_action",
    "pairwise_error_gain",
    "semantic_null_statistics",
    "terminal_decision",
    "unrelated_target_gain_bank",
    "unrelated_target_null_statistics",
]
