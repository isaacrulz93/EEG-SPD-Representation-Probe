"""Constrained quotient-GPA geometry for five locally centered SPD states.

This module is array-only.  Orthogonal registrations and point assignments are
nuisance variables; no function averages or scientifically compares them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from scipy.linalg import expm_frechet

from src.geometry_v2 import (
    airm_distance,
    airm_mean,
    spd_invsqrt,
    spd_log,
    symmetric_exp,
    symmetrize,
)
from src.trajectory_geometry_v0 import ALL_PERMUTATIONS_5


N_POINTS = 5
MASTER_SEED = 20260811
PERMUTATIONS = np.asarray(ALL_PERMUTATIONS_5, dtype=np.int64)
PERMUTATIONS.setflags(write=False)


class GPANumericalError(RuntimeError):
    """A required quotient/GPA numerical certification failed."""


@dataclass(frozen=True)
class GPASettings:
    local_mean_tolerance: float = 1.0e-9
    local_mean_max_iterations: int = 100
    centering_residual_max: float = 1.0e-7
    registration_total_starts: int = 4
    registration_alternations: int = 8
    registration_objective_tolerance: float = 1.0e-9
    action_max_iterations: int = 250
    action_gradient_tolerance: float = 1.0e-6
    action_min_step_size: float = 1.0e-12
    action_max_time_seconds: float = 120.0
    line_search_initial_step_size: float = 1.0
    line_search_contraction_factor: float = 0.5
    line_search_sufficient_decrease: float = 1.0e-4
    line_search_max_iterations: int = 50
    line_search_optimism: float = 2.0
    gpa_total_starts: int = 2
    gpa_outer_iterations: int = 24
    gpa_prototype_inner_iterations: int = 16
    gpa_objective_tolerance: float = 1.0e-7
    gpa_gradient_tolerance: float = 2.0e-5
    prototype_initial_step_size: float = 1.0
    prototype_contraction_factor: float = 0.5
    prototype_sufficient_decrease: float = 1.0e-4
    prototype_line_search_iterations: int = 30
    prototype_min_step_size: float = 1.0e-12
    equivalent_objective_atol: float = 1.0e-10
    equivalent_objective_rtol: float = 1.0e-8
    gpa_equivalent_orbit_tolerance: float = 1.0e-4
    quotient_symmetry_tolerance: float = 1.0e-10


CANDIDATE_SETTINGS = GPASettings()


@dataclass(frozen=True)
class CenteredConfiguration:
    states: np.ndarray
    mean: np.ndarray
    normalized_karcher_residual: float
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class ActionSolve:
    action: np.ndarray
    objective: float
    gradient_norm: float
    iterations: int
    converged: bool
    determinant: int
    stopping_criterion: str


@dataclass(frozen=True)
class RegistrationStart:
    start_index: int
    action: np.ndarray
    permutation: np.ndarray
    objective: float
    gradient_norm: float
    optimizer_iterations: int
    alternations: int
    converged: bool
    determinant: int
    stopping_criterion: str


@dataclass(frozen=True)
class RegistrationFit:
    action: np.ndarray
    permutation: np.ndarray
    objective: float
    best_start_index: int
    starts: tuple[RegistrationStart, ...]
    second_best_objective: float
    objective_spread: float


@dataclass(frozen=True)
class PrototypeStep:
    log_points: np.ndarray
    objective_before: float
    objective_after: float
    projected_gradient_norm: float
    step_size: float
    accepted: bool


@dataclass(frozen=True)
class GPAStart:
    start_index: int
    prototype: np.ndarray
    objective: float
    converged: bool
    outer_iterations: int
    final_projected_gradient_norm: float
    objective_history: np.ndarray
    registrations: tuple[RegistrationFit, ...]


@dataclass(frozen=True)
class GPAFit:
    prototype: np.ndarray
    objective: float
    within_cell_dispersion: float
    best_start_index: int
    starts: tuple[GPAStart, ...]
    best_registrations: tuple[RegistrationFit, ...]
    second_best_objective: float
    objective_spread: float


def _configuration(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[0] != N_POINTS
        or array.shape[1] != array.shape[2]
    ):
        raise ValueError(f"{name} must have shape (5,d,d)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    symmetry = np.linalg.norm(array - array.transpose(0, 2, 1), axis=(1, 2))
    scale = np.maximum(np.linalg.norm(array, axis=(1, 2)), np.finfo(float).tiny)
    if np.max(symmetry / scale) > 1.0e-12:
        raise ValueError(f"{name} must be symmetric")
    if np.min(np.linalg.eigvalsh(array)) <= 0.0:
        raise ValueError(f"{name} must be SPD")
    return symmetrize(array)


def _configuration_bank(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4 or array.shape[1] != N_POINTS:
        raise ValueError(f"{name} must have shape (trials,5,d,d)")
    for index in range(len(array)):
        _configuration(array[index], name=f"{name}[{index}]")
    return symmetrize(array)


def local_center_configuration(
    states: np.ndarray,
    *,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> CenteredConfiguration:
    values = _configuration(states, name="states")
    result = airm_mean(
        values,
        tol=settings.local_mean_tolerance,
        maxiter=settings.local_mean_max_iterations,
    )
    whitening = spd_invsqrt(result.matrix)
    centered = symmetrize(whitening @ values @ whitening)
    centered_result = airm_mean(
        centered,
        tol=settings.local_mean_tolerance,
        maxiter=settings.local_mean_max_iterations,
    )
    residual = float(centered_result.normalized_post_residual)
    identity_error = float(
        np.linalg.norm(centered_result.matrix - np.eye(values.shape[-1]), ord="fro")
        / np.sqrt(values.shape[-1])
    )
    if result.had_warning or centered_result.had_warning:
        raise GPANumericalError(
            "AIRM local-centering mean emitted a convergence warning"
        )
    if residual > settings.centering_residual_max:
        raise GPANumericalError(
            f"local-centering residual {residual:.3e} exceeds frozen maximum"
        )
    if identity_error > 10.0 * settings.centering_residual_max:
        raise GPANumericalError(
            f"local-centering identity error {identity_error:.3e} is too large"
        )
    return CenteredConfiguration(
        states=centered,
        mean=result.matrix,
        normalized_karcher_residual=residual,
        warning_messages=tuple(result.warning_messages + centered_result.warning_messages),
    )


def configuration_from_zero_sum_logs(log_points: np.ndarray) -> np.ndarray:
    logs = np.asarray(log_points, dtype=np.float64)
    if logs.ndim != 3 or logs.shape[0] != N_POINTS or logs.shape[1] != logs.shape[2]:
        raise ValueError("log points must have shape (5,d,d)")
    logs = symmetrize(logs)
    closure = float(np.linalg.norm(np.sum(logs, axis=0), ord="fro"))
    if closure > 1.0e-10:
        raise ValueError(f"log points do not satisfy sum Z_i=0: {closure}")
    return symmetric_exp(logs)


def feasible_prototype_from_configuration(configuration: np.ndarray) -> np.ndarray:
    values = _configuration(configuration, name="configuration")
    logs = spd_log(values)
    logs = symmetrize(logs - np.mean(logs, axis=0, keepdims=True))
    return configuration_from_zero_sum_logs(logs)


def constraint_residual(prototype: np.ndarray) -> float:
    values = _configuration(prototype, name="prototype")
    return float(np.linalg.norm(np.sum(spd_log(values), axis=0), ord="fro"))


def conjugate_configuration(configuration: np.ndarray, action: np.ndarray) -> np.ndarray:
    values = _configuration(configuration, name="configuration")
    q = np.asarray(action, dtype=np.float64)
    if q.shape != values.shape[1:]:
        raise ValueError("action has wrong shape")
    if np.linalg.norm(q.T @ q - np.eye(len(q)), ord="fro") > 1.0e-10:
        raise ValueError("action must be orthogonal")
    return symmetrize(np.einsum("ij,kjl,ml->kim", q, values, q, optimize=True))


def fixed_registration_objective(
    target: np.ndarray,
    source: np.ndarray,
    action: np.ndarray,
    permutation: Sequence[int] | np.ndarray,
) -> float:
    a = _configuration(target, name="target")
    b = _configuration(source, name="source")
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.shape != (N_POINTS,) or not np.array_equal(np.sort(perm), np.arange(N_POINTS)):
        raise ValueError("permutation must be one S5 vertex permutation")
    transformed = conjugate_configuration(b[perm], action)
    distances = np.asarray(airm_distance(a, transformed), dtype=np.float64)
    return float(np.mean(distances * distances))


def _distance_euclidean_gradient_wrt_second(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    whitening = spd_invsqrt(second)
    relative = symmetrize(whitening @ first @ whitening)
    return symmetrize(-2.0 * whitening @ spd_log(relative) @ whitening)


def fixed_registration_gradient(
    target: np.ndarray,
    source: np.ndarray,
    action: np.ndarray,
    permutation: Sequence[int] | np.ndarray,
) -> np.ndarray:
    a = _configuration(target, name="target")
    b = _configuration(source, name="source")
    q = np.asarray(action, dtype=np.float64)
    perm = np.asarray(permutation, dtype=np.int64)
    selected = b[perm]
    transformed = conjugate_configuration(selected, q)
    gradient = np.zeros_like(q)
    for target_point, source_point, predicted in zip(a, selected, transformed, strict=True):
        derivative = _distance_euclidean_gradient_wrt_second(target_point, predicted)
        gradient += 2.0 * derivative @ q @ source_point / N_POINTS
    return gradient


def _action_problem(
    target: np.ndarray,
    source: np.ndarray,
    permutation: np.ndarray,
) -> tuple[Stiefel, Problem]:
    a = _configuration(target, name="target")
    b = _configuration(source, name="source")
    perm = np.asarray(permutation, dtype=np.int64)
    manifold = Stiefel(a.shape[-1], a.shape[-1], retraction="polar")

    @pymanopt.function.numpy(manifold)
    def cost(action: np.ndarray) -> float:
        return fixed_registration_objective(a, b, action, perm)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(action: np.ndarray) -> np.ndarray:
        return fixed_registration_gradient(a, b, action, perm)

    return manifold, Problem(manifold, cost, euclidean_gradient=euclidean_gradient)


def _action_optimizer(settings: GPASettings) -> ConjugateGradient:
    line_search = BackTrackingLineSearcher(
        contraction_factor=settings.line_search_contraction_factor,
        optimism=settings.line_search_optimism,
        sufficient_decrease=settings.line_search_sufficient_decrease,
        max_iterations=settings.line_search_max_iterations,
        initial_step_size=settings.line_search_initial_step_size,
    )
    return ConjugateGradient(
        beta_rule="HestenesStiefel",
        line_searcher=line_search,
        max_time=settings.action_max_time_seconds,
        max_iterations=settings.action_max_iterations,
        min_gradient_norm=settings.action_gradient_tolerance,
        min_step_size=settings.action_min_step_size,
        max_cost_evaluations=settings.action_max_iterations + 1,
        verbosity=0,
        log_verbosity=0,
    )


def optimize_action_fixed_permutation(
    target: np.ndarray,
    source: np.ndarray,
    permutation: np.ndarray,
    start: np.ndarray,
    *,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> ActionSolve:
    manifold, problem = _action_problem(target, source, permutation)
    initial = np.asarray(start, dtype=np.float64)
    if initial.shape != (manifold._n, manifold._p):
        raise ValueError("action start has wrong shape")
    if np.linalg.norm(initial.T @ initial - np.eye(len(initial)), ord="fro") > 1.0e-10:
        raise ValueError("action start is not orthogonal")
    result = _action_optimizer(settings).run(problem, initial_point=initial)
    action = np.asarray(result.point, dtype=np.float64)
    projected = manifold.projection(
        action, fixed_registration_gradient(target, source, action, permutation)
    )
    gradient_norm = float(manifold.norm(action, projected))
    return ActionSolve(
        action=action,
        objective=fixed_registration_objective(target, source, action, permutation),
        gradient_norm=gradient_norm,
        iterations=int(result.iterations),
        converged=bool(gradient_norm <= settings.action_gradient_tolerance),
        determinant=int(np.sign(np.linalg.det(action))),
        stopping_criterion=str(result.stopping_criterion),
    )


def assignment_costs(target: np.ndarray, source: np.ndarray, action: np.ndarray) -> np.ndarray:
    a = _configuration(target, name="target")
    transformed = conjugate_configuration(source, action)
    distances = np.asarray(
        airm_distance(a[:, None, :, :], transformed[None, :, :, :]),
        dtype=np.float64,
    )
    return distances * distances


def exact_best_permutation(
    target: np.ndarray, source: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, float]:
    costs = assignment_costs(target, source, action)
    values = np.mean(costs[np.arange(N_POINTS)[None, :], PERMUTATIONS], axis=1)
    winner = int(np.argmin(values))
    return PERMUTATIONS[winner].copy(), float(values[winner])


def _canonical_eigenvectors(matrix: np.ndarray) -> np.ndarray:
    _, vectors = np.linalg.eigh(symmetrize(matrix))
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0.0:
            vectors[:, column] *= -1.0
    return vectors


def _spectral_action_start(
    target: np.ndarray, source: np.ndarray, permutation: np.ndarray, *, solve_signs: bool
) -> np.ndarray:
    logs_a = spd_log(target)
    logs_b = spd_log(source[np.asarray(permutation, dtype=np.int64)])
    weights = np.sqrt(np.arange(1, N_POINTS + 1, dtype=np.float64))
    va = _canonical_eigenvectors(np.einsum("k,kij->ij", weights, logs_a))
    vb = _canonical_eigenvectors(np.einsum("k,kij->ij", weights, logs_b))
    signs = np.ones(target.shape[-1], dtype=np.float64)
    if solve_signs:
        secondary_weights = np.sqrt(np.arange(N_POINTS, 0, -1, dtype=np.float64))
        secondary_a = va.T @ np.einsum("k,kij->ij", secondary_weights, logs_a) @ va
        secondary_b = vb.T @ np.einsum("k,kij->ij", secondary_weights, logs_b) @ vb
        strength = np.abs(secondary_a * secondary_b)
        np.fill_diagonal(strength, 0.0)
        pivot = int(np.argmax(np.sum(strength, axis=1)))
        for index in range(len(signs)):
            if index != pivot and strength[pivot, index] > np.finfo(float).eps:
                signs[index] = (
                    1.0
                    if secondary_a[pivot, index] * secondary_b[pivot, index] >= 0.0
                    else -1.0
                )
    candidate = va @ np.diag(signs) @ vb.T
    u, _, vt = np.linalg.svd(candidate, full_matrices=False)
    return u @ vt


def _registration_initializations(
    target: np.ndarray,
    source: np.ndarray,
    *,
    seed: int,
    warm_action: np.ndarray | None,
    warm_permutation: np.ndarray | None,
    settings: GPASettings,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if settings.registration_total_starts != 4:
        raise ValueError("V0 requires exactly four total registration starts")
    reflection = np.eye(target.shape[-1])
    reflection[0, 0] = -1.0
    # Internal AIRM distances are invariant to the common orthogonal action.
    # Rank all 120 S5 assignments by their exact distance-matrix mismatch and
    # use the two best assignments as deterministic basins.  This is only an
    # initialization; the scientific objective remains the full SPD rho loss.
    def internal_distances(configuration: np.ndarray) -> np.ndarray:
        result = np.zeros((N_POINTS, N_POINTS), dtype=np.float64)
        for left in range(N_POINTS):
            for right in range(left + 1, N_POINTS):
                value = float(airm_distance(configuration[left], configuration[right]))
                result[left, right] = result[right, left] = value
        return result

    target_distances = internal_distances(target)
    source_distances = internal_distances(source)
    permutation_costs = np.asarray(
        [
            np.mean(
                (
                    target_distances
                    - source_distances[np.ix_(permutation, permutation)]
                ) ** 2
            )
            for permutation in PERMUTATIONS
        ]
    )
    ranked = np.argsort(permutation_costs, kind="stable")
    best_perm = PERMUTATIONS[int(ranked[0])]
    second_perm = PERMUTATIONS[int(ranked[1])]
    if warm_action is not None:
        if warm_permutation is None:
            raise ValueError("warm action requires warm permutation")
        base_one = np.asarray(warm_action, dtype=np.float64)
        perm_one = np.asarray(warm_permutation, dtype=np.int64)
        alternative_index = next(
            int(index)
            for index in ranked
            if not np.array_equal(PERMUTATIONS[int(index)], perm_one)
        )
        perm_two = PERMUTATIONS[alternative_index]
        base_two = _spectral_action_start(target, source, perm_two, solve_signs=True)
    else:
        perm_one = best_perm
        base_one = _spectral_action_start(target, source, perm_one, solve_signs=True)
        perm_two = second_perm
        base_two = _spectral_action_start(target, source, perm_two, solve_signs=True)
    starts = (
        (base_one, perm_one.copy()),
        (base_one @ reflection, perm_one.copy()),
        (base_two, perm_two.copy()),
        (base_two @ reflection, perm_two.copy()),
    )
    determinants = [int(np.sign(np.linalg.det(value[0]))) for value in starts]
    if determinants.count(-1) != 2 or determinants.count(1) != 2:
        raise GPANumericalError("registration starts do not cover O(d) sectors 2+2")
    return starts


def register_configuration(
    target: np.ndarray,
    source: np.ndarray,
    *,
    seed: int,
    warm_action: np.ndarray | None = None,
    warm_permutation: np.ndarray | None = None,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> RegistrationFit:
    a = _configuration(target, name="target")
    b = _configuration(source, name="source")
    initializations = _registration_initializations(
        a,
        b,
        seed=seed,
        warm_action=warm_action,
        warm_permutation=warm_permutation,
        settings=settings,
    )
    return _run_registration_initializations(
        a,
        b,
        initializations,
        require_both_determinant_components=True,
        settings=settings,
    )


def _run_registration_initializations(
    target: np.ndarray,
    source: np.ndarray,
    initializations: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    require_both_determinant_components: bool,
    settings: GPASettings,
) -> RegistrationFit:
    """Optimize an explicit, already-counted registration start list."""

    a = _configuration(target, name="target")
    b = _configuration(source, name="source")
    results: list[RegistrationStart] = []
    for start_index, (initial_action, initial_permutation) in enumerate(initializations):
        action = initial_action
        permutation = initial_permutation
        previous = np.inf
        total_iterations = 0
        converged = False
        stopping = "alternation_limit"
        final_solve: ActionSolve | None = None
        for alternation in range(1, settings.registration_alternations + 1):
            solve = optimize_action_fixed_permutation(
                a, b, permutation, action, settings=settings
            )
            final_solve = solve
            total_iterations += solve.iterations
            if not solve.converged:
                stopping = f"action_failure:{solve.stopping_criterion}"
                break
            new_permutation, new_objective = exact_best_permutation(
                a, b, solve.action
            )
            relative_change = abs(previous - new_objective) / max(
                1.0, abs(previous), abs(new_objective)
            )
            unchanged = np.array_equal(new_permutation, permutation)
            action = solve.action
            permutation = new_permutation
            if unchanged and (
                not np.isfinite(previous)
                or relative_change <= settings.registration_objective_tolerance
            ):
                converged = True
                stopping = "assignment_stable_and_action_converged"
                break
            previous = new_objective
        if final_solve is None:
            raise AssertionError("registration alternation did not execute")
        objective = fixed_registration_objective(a, b, action, permutation)
        results.append(
            RegistrationStart(
                start_index=start_index,
                action=action,
                permutation=permutation,
                objective=objective,
                gradient_norm=final_solve.gradient_norm,
                optimizer_iterations=total_iterations,
                alternations=alternation,
                converged=converged,
                determinant=int(np.sign(np.linalg.det(action))),
                stopping_criterion=stopping,
            )
        )
    converged_results = [value for value in results if value.converged]
    sectors = {value.determinant for value in converged_results}
    if require_both_determinant_components and sectors != {-1, 1}:
        raise GPANumericalError(
            "required registration lacks converged candidates in both determinant sectors"
        )
    if not converged_results:
        raise GPANumericalError("registration has no converged candidate")
    ordered = sorted(
        converged_results,
        key=lambda value: (value.objective, value.gradient_norm, value.start_index),
    )
    best = ordered[0]
    second = ordered[1].objective if len(ordered) > 1 else np.inf
    objectives = np.asarray([value.objective for value in converged_results])
    return RegistrationFit(
        action=best.action,
        permutation=best.permutation,
        objective=best.objective,
        best_start_index=best.start_index,
        starts=tuple(results),
        second_best_objective=float(second),
        objective_spread=float(np.max(objectives) - np.min(objectives)),
    )


def continue_registration(
    target: np.ndarray,
    source: np.ndarray,
    previous: RegistrationFit,
    *,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> RegistrationFit:
    """Continue one certified registration basin during GPA block updates.

    Full four-start/two-component searches are run on the first GPA block and
    again at the accepted final prototype.  Between those certification
    points this continuation follows the current best nuisance solution.  It
    is an optimization continuation, not an additional hidden multistart.
    """

    initializations = ((previous.action, previous.permutation.copy()),)
    return _run_registration_initializations(
        target,
        source,
        initializations,
        require_both_determinant_components=False,
        settings=settings,
    )


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def quotient_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> tuple[float, RegistrationFit]:
    left = _configuration(first, name="first")
    right = _configuration(second, name="second")
    left_hash = _array_sha256(left)
    right_hash = _array_sha256(right)
    if left_hash <= right_hash:
        target, source = left, right
    else:
        target, source = right, left
    seed = int.from_bytes(
        hashlib.sha256(f"{left_hash}|{right_hash}".encode()).digest()[:8], "little"
    )
    fit = register_configuration(target, source, seed=seed, settings=settings)
    return float(np.sqrt(max(fit.objective, 0.0))), fit


def aligned_configuration(source: np.ndarray, fit: RegistrationFit) -> np.ndarray:
    values = _configuration(source, name="source")
    return conjugate_configuration(values[fit.permutation], fit.action)


def fixed_aligned_objective(prototype: np.ndarray, aligned: np.ndarray) -> float:
    p = _configuration(prototype, name="prototype")
    observations = _configuration_bank(aligned, name="aligned")
    distances = np.asarray(
        airm_distance(p[None, :, :, :], observations), dtype=np.float64
    )
    return float(np.mean(distances * distances))


def _prototype_gradient_logs(prototype_logs: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    logs = np.asarray(prototype_logs, dtype=np.float64)
    prototype = configuration_from_zero_sum_logs(logs)
    observations = _configuration_bank(aligned, name="aligned")
    gradients = np.zeros_like(prototype)
    normalization = float(len(observations) * N_POINTS)
    for point in range(N_POINTS):
        for trial in range(len(observations)):
            gradients[point] += _distance_euclidean_gradient_wrt_second(
                observations[trial, point], prototype[point]
            ) / normalization
    log_gradients = np.asarray(
        [
            symmetrize(
                expm_frechet(
                    logs[index], gradients[index], compute_expm=False
                )
            )
            for index in range(N_POINTS)
        ]
    )
    return symmetrize(
        log_gradients - np.mean(log_gradients, axis=0, keepdims=True)
    )


def prototype_projected_gradient_step(
    prototype: np.ndarray,
    aligned: np.ndarray,
    *,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> PrototypeStep:
    current = _configuration(prototype, name="prototype")
    observations = _configuration_bank(aligned, name="aligned")
    logs = spd_log(current)
    closure = np.linalg.norm(np.sum(logs, axis=0), ord="fro")
    if closure > 1.0e-8:
        raise GPANumericalError("prototype left zero-sum log constraint")
    gradient = _prototype_gradient_logs(logs, observations)
    gradient_norm = float(np.linalg.norm(gradient))
    objective = fixed_aligned_objective(current, observations)
    if gradient_norm <= settings.gpa_gradient_tolerance:
        return PrototypeStep(
            log_points=logs,
            objective_before=objective,
            objective_after=objective,
            projected_gradient_norm=gradient_norm,
            step_size=0.0,
            accepted=True,
        )
    step = settings.prototype_initial_step_size
    for _ in range(settings.prototype_line_search_iterations):
        candidate_logs = symmetrize(logs - step * gradient)
        candidate_logs -= np.mean(candidate_logs, axis=0, keepdims=True)
        candidate = configuration_from_zero_sum_logs(candidate_logs)
        candidate_objective = fixed_aligned_objective(candidate, observations)
        if candidate_objective <= objective - (
            settings.prototype_sufficient_decrease * step * gradient_norm**2
        ):
            return PrototypeStep(
                log_points=candidate_logs,
                objective_before=objective,
                objective_after=candidate_objective,
                projected_gradient_norm=gradient_norm,
                step_size=step,
                accepted=True,
            )
        step *= settings.prototype_contraction_factor
        if step < settings.prototype_min_step_size:
            break
    return PrototypeStep(
        log_points=logs,
        objective_before=objective,
        objective_after=objective,
        projected_gradient_norm=gradient_norm,
        step_size=0.0,
        accepted=False,
    )


def _deterministic_seed(*parts: object) -> int:
    payload = json.dumps([MASTER_SEED, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def fit_quotient_gpa(
    configurations: np.ndarray,
    *,
    identity_parts: Sequence[object],
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> GPAFit:
    trials = _configuration_bank(configurations, name="configurations")
    if len(trials) < 2:
        raise ValueError("GPA requires at least two trials")
    if settings.gpa_total_starts != 2:
        raise ValueError("V0 requires exactly two total GPA starts")
    initial_indices = (0, len(trials) // 2)
    start_results: list[GPAStart] = []
    for start_index, trial_index in enumerate(initial_indices):
        prototype = feasible_prototype_from_configuration(trials[trial_index])
        warm: list[RegistrationFit | None] = [None] * len(trials)
        history: list[float] = []
        final_gradient = np.inf
        converged = False
        final_registrations: tuple[RegistrationFit, ...] = ()
        for outer in range(1, settings.gpa_outer_iterations + 1):
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
                        settings=settings,
                    )
                else:
                    fit = continue_registration(
                        prototype,
                        trials[trial],
                        previous,
                        settings=settings,
                    )
                registrations.append(fit)
                aligned.append(aligned_configuration(trials[trial], fit))
            aligned_array = np.asarray(aligned)
            aligned_objective = float(
                np.mean([value.objective for value in registrations])
            )
            for _ in range(settings.gpa_prototype_inner_iterations):
                step = prototype_projected_gradient_step(
                    prototype, aligned_array, settings=settings
                )
                if not step.accepted:
                    raise GPANumericalError(
                        "constrained prototype line search failed with nonzero gradient"
                    )
                prototype = configuration_from_zero_sum_logs(step.log_points)
                final_gradient = step.projected_gradient_norm
                if final_gradient <= settings.gpa_gradient_tolerance:
                    break
            history.append(step.objective_after)
            warm = registrations
            final_registrations = tuple(registrations)
            relative_change = (
                np.inf
                if len(history) < 2
                else abs(history[-2] - history[-1])
                / max(1.0, abs(history[-2]), abs(history[-1]))
            )
            if (
                final_gradient <= settings.gpa_gradient_tolerance
                and relative_change <= settings.gpa_objective_tolerance
            ):
                converged = True
                break
            if aligned_objective + 1.0e-10 < step.objective_after:
                raise GPANumericalError("prototype update increased aligned objective")
        if not converged:
            raise GPANumericalError(
                f"GPA start {start_index} did not converge in frozen outer iterations"
            )
        # Re-register once at the accepted prototype so the stored objective
        # and nuisance variables correspond exactly to the returned P.
        final_fits: list[RegistrationFit] = []
        for trial, previous in enumerate(final_registrations):
            final_fits.append(
                register_configuration(
                    prototype,
                    trials[trial],
                    seed=_deterministic_seed(
                        "gpa_final", *identity_parts, start_index, trial
                    ),
                    warm_action=previous.action,
                    warm_permutation=previous.permutation,
                    settings=settings,
                )
            )
        objective = float(np.mean([value.objective for value in final_fits]))
        start_results.append(
            GPAStart(
                start_index=start_index,
                prototype=prototype,
                objective=objective,
                converged=True,
                outer_iterations=outer,
                final_projected_gradient_norm=final_gradient,
                objective_history=np.asarray(history),
                registrations=tuple(final_fits),
            )
        )
    ordered = sorted(start_results, key=lambda value: (value.objective, value.start_index))
    best = ordered[0]
    second = ordered[1].objective
    return GPAFit(
        prototype=best.prototype,
        objective=best.objective,
        within_cell_dispersion=float(np.sqrt(max(best.objective, 0.0))),
        best_start_index=best.start_index,
        starts=tuple(start_results),
        best_registrations=best.registrations,
        second_best_objective=float(second),
        objective_spread=float(max(value.objective for value in start_results) - min(value.objective for value in start_results)),
    )


__all__ = [
    "ActionSolve",
    "CANDIDATE_SETTINGS",
    "CenteredConfiguration",
    "GPAFit",
    "GPANumericalError",
    "GPASettings",
    "GPAStart",
    "N_POINTS",
    "PERMUTATIONS",
    "PrototypeStep",
    "RegistrationFit",
    "RegistrationStart",
    "aligned_configuration",
    "assignment_costs",
    "configuration_from_zero_sum_logs",
    "conjugate_configuration",
    "continue_registration",
    "constraint_residual",
    "exact_best_permutation",
    "feasible_prototype_from_configuration",
    "fit_quotient_gpa",
    "fixed_aligned_objective",
    "fixed_registration_gradient",
    "fixed_registration_objective",
    "local_center_configuration",
    "optimize_action_fixed_permutation",
    "prototype_projected_gradient_step",
    "quotient_distance",
    "register_configuration",
]
