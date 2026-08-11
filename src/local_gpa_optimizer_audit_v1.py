"""Optimizer-only forensic tools for the frozen local-GPA registration loss.

The scientific action, AIRM cost, S5 update, four starts, O(d) sectors, and
gradient certification are imported unchanged from local_gpa_geometry_v0.
This module only adds trace capture and a finite-difference Riemannian Hessian
operator for Pymanopt TrustRegions.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions

import src.local_gpa_geometry_v0 as frozen


SolverName = Literal["ConjugateGradient", "TrustRegions"]
HESSIAN_RADIUS = 1.0e-5
TRUST_MAX_ITERATIONS = 250
TRUST_MAX_TIME_SECONDS = 120.0
TRUST_MAX_INNER_ITERATIONS = 30


@dataclass(frozen=True)
class OptimizerSolveTrace:
    alternation: int
    permutation_before: np.ndarray
    permutation_after: np.ndarray
    objective_before: float
    objective_after_solve: float
    objective_after_assignment: float
    gradient_norm: float
    iterations: int
    cost_evaluations_reported: int | None
    cost_calls_counted: int
    gradient_calls_counted: int
    hessian_calls_counted: int
    runtime_seconds: float
    stopping_criterion: str
    converged: bool
    iteration_objectives: np.ndarray
    iteration_gradient_norms: np.ndarray


@dataclass(frozen=True)
class ForensicStart:
    start_index: int
    initial_action: np.ndarray
    final_action: np.ndarray
    initial_determinant: int
    final_determinant: int
    starting_permutation: np.ndarray
    final_permutation: np.ndarray
    initial_objective: float
    final_objective: float
    final_gradient_norm: float
    total_optimizer_iterations: int
    total_cost_calls: int
    total_gradient_calls: int
    total_hessian_calls: int
    total_runtime_seconds: float
    alternations: int
    stopping_criterion: str
    converged: bool
    solves: tuple[OptimizerSolveTrace, ...]


@dataclass(frozen=True)
class ForensicRegistration:
    solver: SolverName
    starts: tuple[ForensicStart, ...]
    determinant_minus_certified: bool
    determinant_plus_certified: bool
    best_converged_objective: float
    total_runtime_seconds: float


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _unchecked_conjugation(configuration: np.ndarray, action: np.ndarray) -> np.ndarray:
    return frozen.symmetrize(
        np.einsum(
            "ij,kjl,ml->kim", action, configuration, action, optimize=True
        )
    )


def ambient_registration_gradient(
    target: np.ndarray,
    source: np.ndarray,
    action: np.ndarray,
    permutation: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Euclidean gradient of the exact AIRM cost on an ambient GL(d) extension."""

    selected = np.asarray(source, dtype=np.float64)[np.asarray(permutation, dtype=np.int64)]
    predicted = _unchecked_conjugation(selected, np.asarray(action, dtype=np.float64))
    gradient = np.zeros_like(action, dtype=np.float64)
    for target_point, source_point, estimate in zip(
        np.asarray(target, dtype=np.float64), selected, predicted, strict=True
    ):
        derivative = frozen._distance_euclidean_gradient_wrt_second(
            target_point, estimate
        )
        gradient += 2.0 * derivative @ action @ source_point / frozen.N_POINTS
    return gradient


def _problem(
    target: np.ndarray,
    source: np.ndarray,
    permutation: np.ndarray,
    *,
    with_hessian: bool,
) -> tuple[Stiefel, Problem, dict[str, int]]:
    target_values = frozen._configuration(target, name="target")
    source_values = frozen._configuration(source, name="source")
    permutation_values = np.asarray(permutation, dtype=np.int64)
    manifold = Stiefel(target_values.shape[-1], target_values.shape[-1], retraction="polar")
    counters = {"cost": 0, "gradient": 0, "hessian": 0}

    @pymanopt.function.numpy(manifold)
    def cost(action: np.ndarray) -> float:
        counters["cost"] += 1
        return frozen.fixed_registration_objective(
            target_values, source_values, action, permutation_values
        )

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(action: np.ndarray) -> np.ndarray:
        counters["gradient"] += 1
        return frozen.fixed_registration_gradient(
            target_values, source_values, action, permutation_values
        )

    if not with_hessian:
        return manifold, Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient
        ), counters

    @pymanopt.function.numpy(manifold)
    def riemannian_hessian(
        action: np.ndarray, tangent: np.ndarray
    ) -> np.ndarray:
        counters["hessian"] += 1
        tangent_norm = float(manifold.norm(action, tangent))
        if tangent_norm == 0.0:
            return manifold.zero_vector(action)
        unit = tangent / tangent_norm
        plus = manifold.retraction(action, HESSIAN_RADIUS * unit)
        minus = manifold.retraction(action, -HESSIAN_RADIUS * unit)
        plus_gradient = manifold.euclidean_to_riemannian_gradient(
            plus,
            ambient_registration_gradient(
                target_values, source_values, plus, permutation_values
            ),
        )
        minus_gradient = manifold.euclidean_to_riemannian_gradient(
            minus,
            ambient_registration_gradient(
                target_values, source_values, minus, permutation_values
            ),
        )
        transported_plus = manifold.transport(plus, action, plus_gradient)
        transported_minus = manifold.transport(minus, action, minus_gradient)
        estimate = tangent_norm * (transported_plus - transported_minus) / (
            2.0 * HESSIAN_RADIUS
        )
        return manifold.projection(action, estimate)

    return manifold, Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        riemannian_hessian=riemannian_hessian,
    ), counters


def riemannian_hessian_vector(
    target: np.ndarray,
    source: np.ndarray,
    permutation: np.ndarray,
    action: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    _, problem, _ = _problem(
        target, source, permutation, with_hessian=True
    )
    return np.asarray(problem.riemannian_hessian(action, tangent))


def _solve(
    target: np.ndarray,
    source: np.ndarray,
    permutation: np.ndarray,
    initial_action: np.ndarray,
    *,
    solver: SolverName,
    settings: frozen.GPASettings,
) -> tuple[np.ndarray, object, float, dict[str, int]]:
    manifold, problem, counters = _problem(
        target,
        source,
        permutation,
        with_hessian=solver == "TrustRegions",
    )
    if solver == "ConjugateGradient":
        line_search = frozen.BackTrackingLineSearcher(
            contraction_factor=settings.line_search_contraction_factor,
            optimism=settings.line_search_optimism,
            sufficient_decrease=settings.line_search_sufficient_decrease,
            max_iterations=settings.line_search_max_iterations,
            initial_step_size=settings.line_search_initial_step_size,
        )
        optimizer = frozen.ConjugateGradient(
            beta_rule="HestenesStiefel",
            line_searcher=line_search,
            max_time=settings.action_max_time_seconds,
            max_iterations=settings.action_max_iterations,
            min_gradient_norm=settings.action_gradient_tolerance,
            min_step_size=settings.action_min_step_size,
            max_cost_evaluations=settings.action_max_iterations + 1,
            verbosity=0,
            # Observation-only forensic logging.  It does not change the
            # objective, update, line search, start, or stopping contract.
            log_verbosity=1,
        )
        started = time.perf_counter()
        result = optimizer.run(problem, initial_point=initial_action)
    elif solver == "TrustRegions":
        optimizer = TrustRegions(
            miniter=3,
            kappa=0.1,
            theta=1.0,
            rho_prime=0.1,
            use_rand=False,
            rho_regularization=1000.0,
            max_time=TRUST_MAX_TIME_SECONDS,
            max_iterations=TRUST_MAX_ITERATIONS,
            min_gradient_norm=settings.action_gradient_tolerance,
            min_step_size=settings.action_min_step_size,
            max_cost_evaluations=5000,
            verbosity=0,
            log_verbosity=0,
        )
        started = time.perf_counter()
        result = optimizer.run(
            problem,
            initial_point=initial_action,
            mininner=1,
            maxinner=TRUST_MAX_INNER_ITERATIONS,
        )
    else:
        raise ValueError(f"unknown solver {solver}")
    runtime = time.perf_counter() - started
    action = np.asarray(result.point, dtype=np.float64)
    projected = manifold.projection(
        action,
        frozen.fixed_registration_gradient(
            target, source, action, permutation
        ),
    )
    gradient_norm = float(manifold.norm(action, projected))
    return action, result, gradient_norm, {**counters, "runtime_ns": int(runtime * 1e9)}


def run_registration_forensic(
    target: np.ndarray,
    source: np.ndarray,
    *,
    solver: SolverName,
    settings: frozen.GPASettings = frozen.CANDIDATE_SETTINGS,
) -> ForensicRegistration:
    """Run the exact frozen four starts and assignment alternations with traces."""

    target_values = frozen._configuration(target, name="target")
    source_values = frozen._configuration(source, name="source")
    initializations = frozen._registration_initializations(
        target_values,
        source_values,
        seed=0,
        warm_action=None,
        warm_permutation=None,
        settings=settings,
    )
    starts: list[ForensicStart] = []
    registration_started = time.perf_counter()
    for start_index, (initial_action, initial_permutation) in enumerate(initializations):
        action = np.asarray(initial_action, dtype=np.float64)
        permutation = np.asarray(initial_permutation, dtype=np.int64)
        starting_permutation = permutation.copy()
        initial_objective = frozen.fixed_registration_objective(
            target_values, source_values, action, permutation
        )
        previous = np.inf
        solve_traces: list[OptimizerSolveTrace] = []
        converged = False
        stopping = "alternation_limit"
        final_gradient = np.inf
        for alternation in range(1, settings.registration_alternations + 1):
            objective_before = frozen.fixed_registration_objective(
                target_values, source_values, action, permutation
            )
            solved_action, result, gradient_norm, counters = _solve(
                target_values,
                source_values,
                permutation,
                action,
                solver=solver,
                settings=settings,
            )
            objective_after_solve = frozen.fixed_registration_objective(
                target_values, source_values, solved_action, permutation
            )
            new_permutation, new_objective = frozen.exact_best_permutation(
                target_values, source_values, solved_action
            )
            optimizer_converged = bool(
                gradient_norm <= settings.action_gradient_tolerance
            )
            solve_traces.append(
                OptimizerSolveTrace(
                    alternation=alternation,
                    permutation_before=permutation.copy(),
                    permutation_after=new_permutation.copy(),
                    objective_before=objective_before,
                    objective_after_solve=objective_after_solve,
                    objective_after_assignment=new_objective,
                    gradient_norm=gradient_norm,
                    iterations=int(result.iterations),
                    cost_evaluations_reported=(
                        None
                        if result.cost_evaluations is None
                        else int(result.cost_evaluations)
                    ),
                    cost_calls_counted=int(counters["cost"]),
                    gradient_calls_counted=int(counters["gradient"]),
                    hessian_calls_counted=int(counters["hessian"]),
                    runtime_seconds=float(counters["runtime_ns"]) / 1e9,
                    stopping_criterion=str(result.stopping_criterion),
                    converged=optimizer_converged,
                    iteration_objectives=(
                        np.asarray([], dtype=np.float64)
                        if result.log is None
                        or result.log.get("iterations") is None
                        else np.asarray(
                            result.log["iterations"].get("cost", []),
                            dtype=np.float64,
                        )
                    ),
                    iteration_gradient_norms=(
                        np.asarray([], dtype=np.float64)
                        if result.log is None
                        or result.log.get("iterations") is None
                        else np.asarray(
                            result.log["iterations"].get("gradient_norm", []),
                            dtype=np.float64,
                        )
                    ),
                )
            )
            action = solved_action
            final_gradient = gradient_norm
            if not optimizer_converged:
                stopping = f"action_failure:{result.stopping_criterion}"
                permutation = new_permutation
                break
            relative_change = abs(previous - new_objective) / max(
                1.0, abs(previous), abs(new_objective)
            )
            unchanged = np.array_equal(new_permutation, permutation)
            permutation = new_permutation
            if unchanged and (
                not np.isfinite(previous)
                or relative_change <= settings.registration_objective_tolerance
            ):
                converged = True
                stopping = "assignment_stable_and_action_converged"
                break
            previous = new_objective
        final_objective = frozen.fixed_registration_objective(
            target_values, source_values, action, permutation
        )
        starts.append(
            ForensicStart(
                start_index=start_index,
                initial_action=np.asarray(initial_action),
                final_action=action,
                initial_determinant=int(np.sign(np.linalg.det(initial_action))),
                final_determinant=int(np.sign(np.linalg.det(action))),
                starting_permutation=starting_permutation,
                final_permutation=permutation,
                initial_objective=initial_objective,
                final_objective=final_objective,
                final_gradient_norm=final_gradient,
                total_optimizer_iterations=sum(
                    value.iterations for value in solve_traces
                ),
                total_cost_calls=sum(value.cost_calls_counted for value in solve_traces),
                total_gradient_calls=sum(
                    value.gradient_calls_counted for value in solve_traces
                ),
                total_hessian_calls=sum(
                    value.hessian_calls_counted for value in solve_traces
                ),
                total_runtime_seconds=sum(
                    value.runtime_seconds for value in solve_traces
                ),
                alternations=len(solve_traces),
                stopping_criterion=stopping,
                converged=converged,
                solves=tuple(solve_traces),
            )
        )
    converged_starts = [value for value in starts if value.converged]
    sectors = {value.final_determinant for value in converged_starts}
    return ForensicRegistration(
        solver=solver,
        starts=tuple(starts),
        determinant_minus_certified=-1 in sectors,
        determinant_plus_certified=1 in sectors,
        best_converged_objective=(
            float(min(value.final_objective for value in converged_starts))
            if converged_starts
            else np.inf
        ),
        total_runtime_seconds=time.perf_counter() - registration_started,
    )


__all__ = [
    "ForensicRegistration",
    "ForensicStart",
    "HESSIAN_RADIUS",
    "OptimizerSolveTrace",
    "TRUST_MAX_INNER_ITERATIONS",
    "TRUST_MAX_ITERATIONS",
    "TRUST_MAX_TIME_SECONDS",
    "ambient_registration_gradient",
    "array_sha256",
    "riemannian_hessian_vector",
    "run_registration_forensic",
]
