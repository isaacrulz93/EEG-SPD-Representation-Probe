"""Start-specific outer-loop tracing for the frozen Local GPA V1 algorithm.

This module does not compute cell-consensus matrices, between-cell distances,
interaction contrasts, nulls, or p-values.  It replays one deterministic GPA
start and records the unchanged alternating numerical trajectory.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from src.local_gpa_geometry_v0 import (
    CANDIDATE_SETTINGS,
    GPASettings,
    GPANumericalError,
    RegistrationFit,
    aligned_configuration,
    configuration_from_zero_sum_logs,
    constraint_residual,
    continue_registration,
    feasible_prototype_from_configuration,
    prototype_projected_gradient_step,
    quotient_distance,
    register_configuration,
)


FROZEN_OUTER_ITERATIONS = 24
AUDIT_MAX_OUTER_ITERATIONS = 96
PROTOTYPE_INITIAL_STEP_CANDIDATES = (1.0, 2.0, 4.0)


@dataclass(frozen=True)
class OuterTrace:
    outer_iteration: int
    joint_final_block_objective: float
    aligned_registration_objective: float
    projected_prototype_gradient_norm: float
    relative_objective_change: float
    prototype_inner_iterations: int
    prototype_accepted_step_sizes: tuple[float, ...]
    prototype_backtracking_reductions: tuple[int, ...]
    prototype_total_backtracking_reductions: int
    final_prototype_line_search_step_size: float
    changed_trial_permutations: int
    registration_objective_minimum: float
    registration_objective_median: float
    registration_objective_mean: float
    registration_objective_maximum: float
    consecutive_prototype_quotient_distance: float
    prototype_constraint_residual: float
    convergence_boolean: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class StartTraceResult:
    identity_parts: tuple[object, ...]
    start_index: int
    prototype_initial_step_size: float
    initial_trial_position: int
    cell_configuration_sha256: str
    initial_prototype_sha256: str
    trial_count: int
    traces: tuple[OuterTrace, ...]
    converged: bool
    first_convergence_outer_iteration: int | None
    frozen_24_reproduced_failure: bool
    runtime_seconds: float


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _deterministic_seed(*parts: object) -> int:
    payload = __import__("json").dumps(
        [20260811, *parts], separators=(",", ":")
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def run_gpa_start_trace(
    configurations: np.ndarray,
    *,
    identity_parts: Sequence[object],
    start_index: int,
    max_outer_iterations: int = AUDIT_MAX_OUTER_ITERATIONS,
    prototype_initial_step_size: float = 1.0,
    compute_consecutive_quotient_distance: bool = False,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> StartTraceResult:
    """Replay one exact frozen GPA start and continue it without resetting."""

    trials = np.asarray(configurations, dtype=np.float64)
    if trials.ndim != 4 or trials.shape[1] != 5 or trials.shape[2] != trials.shape[3]:
        raise ValueError("configurations must have shape (trials,5,d,d)")
    if start_index not in (0, 1):
        raise ValueError("the frozen GPA has exactly starts 0 and 1")
    if max_outer_iterations < FROZEN_OUTER_ITERATIONS:
        raise ValueError("audit must include all 24 frozen outer iterations")
    if max_outer_iterations > AUDIT_MAX_OUTER_ITERATIONS:
        raise ValueError("audit continuation is capped at 96 outer iterations")
    if float(prototype_initial_step_size) not in PROTOTYPE_INITIAL_STEP_CANDIDATES:
        raise ValueError("initial-step audit is frozen to {1.0,2.0,4.0}")
    effective_settings = replace(
        settings,
        prototype_initial_step_size=float(prototype_initial_step_size),
    )
    initial_trial_position = 0 if start_index == 0 else len(trials) // 2
    prototype = feasible_prototype_from_configuration(
        trials[initial_trial_position]
    )
    cell_hash = array_sha256(trials)
    initial_prototype_hash = array_sha256(prototype)
    warm: list[RegistrationFit | None] = [None] * len(trials)
    history: list[float] = []
    trace: list[OuterTrace] = []
    previous_prototype: np.ndarray | None = None
    started = time.perf_counter()
    first_convergence: int | None = None
    for outer in range(1, max_outer_iterations + 1):
        outer_started = time.perf_counter()
        registrations: list[RegistrationFit] = []
        aligned: list[np.ndarray] = []
        for trial in range(len(trials)):
            previous = warm[trial]
            if previous is None:
                fit = register_configuration(
                    prototype,
                    trials[trial],
                    seed=_deterministic_seed(
                        "gpa_registration",
                        *identity_parts,
                        start_index,
                        outer,
                        trial,
                    ),
                    settings=effective_settings,
                )
            else:
                fit = continue_registration(
                    prototype,
                    trials[trial],
                    previous,
                    settings=effective_settings,
                )
            registrations.append(fit)
            aligned.append(aligned_configuration(trials[trial], fit))
        registration_objectives = np.asarray(
            [value.objective for value in registrations], dtype=np.float64
        )
        aligned_objective = float(np.mean(registration_objectives))
        aligned_array = np.asarray(aligned)
        step = None
        accepted_step_sizes: list[float] = []
        backtracking_reductions: list[int] = []
        for inner in range(1, effective_settings.gpa_prototype_inner_iterations + 1):
            step = prototype_projected_gradient_step(
                prototype, aligned_array, settings=effective_settings
            )
            if not step.accepted:
                raise GPANumericalError(
                    "constrained prototype line search failed with nonzero gradient"
                )
            prototype = configuration_from_zero_sum_logs(step.log_points)
            accepted_step_sizes.append(float(step.step_size))
            if step.step_size <= 0.0:
                reductions = 0
            else:
                reductions = int(
                    round(
                        np.log(
                            step.step_size
                            / effective_settings.prototype_initial_step_size
                        )
                        / np.log(effective_settings.prototype_contraction_factor)
                    )
                )
            if reductions < 0:
                raise GPANumericalError("prototype line search expanded its initial step")
            backtracking_reductions.append(reductions)
            if (
                step.projected_gradient_norm
                <= effective_settings.gpa_gradient_tolerance
            ):
                break
        if step is None:
            raise AssertionError("prototype inner loop did not execute")
        history.append(float(step.objective_after))
        relative_change = (
            np.inf
            if len(history) < 2
            else abs(history[-2] - history[-1])
            / max(1.0, abs(history[-2]), abs(history[-1]))
        )
        converged = bool(
            step.projected_gradient_norm
            <= effective_settings.gpa_gradient_tolerance
            and relative_change <= effective_settings.gpa_objective_tolerance
        )
        if aligned_objective + 1.0e-10 < step.objective_after:
            raise GPANumericalError("prototype update increased aligned objective")
        if warm[0] is None:
            changed_permutations = -1
        else:
            changed_permutations = int(
                sum(
                    not np.array_equal(current.permutation, previous.permutation)
                    for current, previous in zip(registrations, warm, strict=True)
                    if previous is not None
                )
            )
        prototype_distance = np.nan
        if compute_consecutive_quotient_distance and previous_prototype is not None:
            prototype_distance, _ = quotient_distance(
                previous_prototype, prototype, settings=effective_settings
            )
        residual = constraint_residual(prototype)
        trace.append(
            OuterTrace(
                outer_iteration=outer,
                joint_final_block_objective=float(step.objective_after),
                aligned_registration_objective=aligned_objective,
                projected_prototype_gradient_norm=float(
                    step.projected_gradient_norm
                ),
                relative_objective_change=float(relative_change),
                prototype_inner_iterations=inner,
                prototype_accepted_step_sizes=tuple(accepted_step_sizes),
                prototype_backtracking_reductions=tuple(
                    backtracking_reductions
                ),
                prototype_total_backtracking_reductions=int(
                    sum(backtracking_reductions)
                ),
                final_prototype_line_search_step_size=float(step.step_size),
                changed_trial_permutations=changed_permutations,
                registration_objective_minimum=float(
                    np.min(registration_objectives)
                ),
                registration_objective_median=float(
                    np.median(registration_objectives)
                ),
                registration_objective_mean=aligned_objective,
                registration_objective_maximum=float(
                    np.max(registration_objectives)
                ),
                consecutive_prototype_quotient_distance=float(prototype_distance),
                prototype_constraint_residual=float(residual),
                convergence_boolean=converged,
                elapsed_seconds=time.perf_counter() - outer_started,
            )
        )
        previous_prototype = prototype.copy()
        warm = registrations
        if converged:
            first_convergence = outer
            break
    frozen_failure = bool(
        first_convergence is None or first_convergence > FROZEN_OUTER_ITERATIONS
    )
    return StartTraceResult(
        identity_parts=tuple(identity_parts),
        start_index=start_index,
        prototype_initial_step_size=float(prototype_initial_step_size),
        initial_trial_position=initial_trial_position,
        cell_configuration_sha256=cell_hash,
        initial_prototype_sha256=initial_prototype_hash,
        trial_count=len(trials),
        traces=tuple(trace),
        converged=first_convergence is not None,
        first_convergence_outer_iteration=first_convergence,
        frozen_24_reproduced_failure=frozen_failure,
        runtime_seconds=time.perf_counter() - started,
    )


def trace_records(result: StartTraceResult) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for value in result.traces:
        record = {
            "identity": "|".join(map(str, result.identity_parts)),
            "start_index": result.start_index,
            "prototype_initial_step_size": result.prototype_initial_step_size,
            "initial_trial_position": result.initial_trial_position,
            "cell_configuration_sha256": result.cell_configuration_sha256,
            "initial_prototype_sha256": result.initial_prototype_sha256,
            **value.__dict__,
        }
        record["prototype_accepted_step_sizes"] = __import__("json").dumps(
            list(value.prototype_accepted_step_sizes), separators=(",", ":")
        )
        record["prototype_backtracking_reductions"] = __import__("json").dumps(
            list(value.prototype_backtracking_reductions), separators=(",", ":")
        )
        records.append(record)
    return records


__all__ = [
    "AUDIT_MAX_OUTER_ITERATIONS",
    "FROZEN_OUTER_ITERATIONS",
    "OuterTrace",
    "PROTOTYPE_INITIAL_STEP_CANDIDATES",
    "StartTraceResult",
    "array_sha256",
    "run_gpa_start_trace",
    "trace_records",
]
