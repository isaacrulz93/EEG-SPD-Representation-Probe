"""Deterministic orthogonal-conjugation solver for common subject action v0.

This module is deliberately array-only.  It has no dataset loader and cannot
read the frozen BNCI objects.  The numerical constants below are calibrated
only by the synthetic validation suite before the scientific protocol freeze.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from pyriemann.optimization.grassmann import _project as _pyriemann_project
from pyriemann.optimization.grassmann import _retract as _pyriemann_retract
from scipy.linalg import expm


MASTER_SEED = 20260810


class ActionSolverError(RuntimeError):
    """The constrained optimization contract could not be satisfied."""


@dataclass(frozen=True)
class SolverSettings:
    starts: int = 8
    max_iterations: int = 1000
    gradient_tolerance: float = 1.0e-5
    objective_tolerance: float = 1.0e-13
    objective_stall_gradient_multiplier: float = 10.0
    pymanopt_log_verbosity: int = 0
    line_search_initial_step_size: float = 1.0
    line_search_contraction_factor: float = 0.5
    line_search_sufficient_decrease: float = 1.0e-4
    line_search_max_iterations: int = 50
    line_search_optimism: float = 2.0
    optimizer_min_step_size: float = 1.0e-12
    optimizer_max_time_seconds: float = 3600.0
    outer_starts: int = 4
    outer_iterations: int = 120
    outer_objective_tolerance: float = 1.0e-6
    equivalent_objective_absolute_tolerance: float = 1.0e-10
    equivalent_objective_relative_tolerance: float = 1.0e-8
    commutant_relative_tolerance: float = 1.0e-8
    exact_prediction_relative_tolerance: float = 1.0e-8
    prediction_dispersion_numerical_floor: float = 1.0e-5


# These values remain candidates until the corrected protocol is committed.
CANDIDATE_SOLVER_SETTINGS = SolverSettings()


@dataclass(frozen=True)
class StartResult:
    start_index: int
    matrix: np.ndarray
    objective: float
    gradient_norm: float
    converged: bool
    iterations: int
    determinant: float
    line_search_failures: int
    optimizer: str
    stopping_criterion: str


@dataclass(frozen=True)
class ActionFit:
    matrix: np.ndarray
    best_start_index: int
    starts: tuple[StartResult, ...]
    equivalent_start_indices: tuple[int, ...]


@dataclass(frozen=True)
class SourceStartResult:
    start_index: int
    actions: np.ndarray
    templates: np.ndarray
    objective: float
    maximum_gradient_norm: float
    converged: bool
    outer_iterations: int
    determinants: np.ndarray


@dataclass(frozen=True)
class SourceModelFit:
    actions: np.ndarray
    templates: np.ndarray
    best_start_index: int
    starts: tuple[SourceStartResult, ...]
    equivalent_start_indices: tuple[int, ...]
    anchor_index: int


@dataclass(frozen=True)
class StabilizerDiagnostic:
    singular_values: np.ndarray
    numerical_tolerance: float
    numerical_nullity: int
    approximate_tolerance: float
    approximate_nullity: int
    skew_basis: np.ndarray
    nullspace_basis: np.ndarray


@dataclass(frozen=True)
class PredictiveIdentifiability:
    classification: str
    equivalent_solution_count: int
    maximum_relative_prediction_dispersion: float
    split_half_relative_variability: float
    materiality_threshold: float
    prediction_normalization: float
    normalization_epsilon: float
    prediction_hashes: tuple[str, ...]


@dataclass(frozen=True)
class MultistartDiagnostic:
    classification: str
    converged_count: int
    failed_count: int
    equivalent_count: int
    converged_non_equivalent_count: int
    determinant_sectors_with_converged_solution: tuple[int, ...]


def symmetrize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 0.5 * (array + np.swapaxes(array, -1, -2))


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _validate_banks(targets: np.ndarray, templates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(targets, dtype=np.float64)
    b = np.asarray(templates, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != a.shape[-2] or a.shape[0] < 1:
        raise ValueError("targets/templates must share shape (object,d,d)")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("targets/templates must be finite")
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
    if max(float(np.linalg.norm(a - np.swapaxes(a, -1, -2))), float(np.linalg.norm(b - np.swapaxes(b, -1, -2)))) > 1.0e-12 * scale:
        raise ValueError("orthogonal-conjugation inputs must be symmetric")
    return symmetrize(a), symmetrize(b)


def action_objective(targets: np.ndarray, templates: np.ndarray, action: np.ndarray) -> float:
    a, b = _validate_banks(targets, templates)
    q = np.asarray(action, dtype=np.float64)
    if q.shape != a.shape[-2:]:
        raise ValueError("action has the wrong shape")
    prediction = np.einsum("ij,kjl,ml->kim", q, b, q, optimize=True)
    residual = prediction - a
    return float(np.sum(residual * residual, dtype=np.float64))


def action_gradient(targets: np.ndarray, templates: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Analytic Euclidean gradient of sum ||A-QBQ' ||_F^2."""

    a, b = _validate_banks(targets, templates)
    q = np.asarray(action, dtype=np.float64)
    prediction = np.einsum("ij,kjl,ml->kim", q, b, q, optimize=True)
    residual = prediction - a
    return 4.0 * np.einsum("kij,jl,klm->im", residual, q, b, optimize=True)


def tangent_projection(action: np.ndarray, euclidean_gradient: np.ndarray) -> np.ndarray:
    """Use pyRiemann's O(d) tangent projection primitive exactly."""

    return np.asarray(
        _pyriemann_project(
            np.asarray(action, dtype=np.float64),
            np.asarray(euclidean_gradient, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def orthogonal_retraction(action: np.ndarray, tangent_step: np.ndarray) -> np.ndarray:
    """Use pyRiemann's sign-corrected QR retraction on O(d) exactly."""

    result = np.asarray(
        _pyriemann_retract(
            np.asarray(action, dtype=np.float64),
            np.asarray(tangent_step, dtype=np.float64),
        ),
        dtype=np.float64,
    )
    error = float(np.linalg.norm(result.T @ result - np.eye(result.shape[0])))
    if error > 1.0e-10:
        raise ActionSolverError(f"polar retraction orthogonality failure: {error}")
    return result


def _canonical_orthogonal(seed: int, dimension: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([MASTER_SEED, int(seed)])))
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q @ np.diag(signs)


def _spectral_start(targets: np.ndarray, templates: np.ndarray, *, solve_signs: bool) -> np.ndarray:
    a, b = _validate_banks(targets, templates)
    weights = np.sqrt(np.arange(1, len(a) + 1, dtype=np.float64))
    primary_a = np.einsum("k,kij->ij", weights, a, optimize=True)
    primary_b = np.einsum("k,kij->ij", weights, b, optimize=True)
    _, va = np.linalg.eigh(symmetrize(primary_a))
    _, vb = np.linalg.eigh(symmetrize(primary_b))
    signs = np.ones(a.shape[-1], dtype=np.float64)
    if solve_signs and len(a) > 1:
        secondary_weights = np.sqrt(np.arange(len(a), 0, -1, dtype=np.float64))
        secondary_a = va.T @ np.einsum("k,kij->ij", secondary_weights, a, optimize=True) @ va
        secondary_b = vb.T @ np.einsum("k,kij->ij", secondary_weights, b, optimize=True) @ vb
        strength = np.abs(secondary_a * secondary_b)
        np.fill_diagonal(strength, 0.0)
        pivot = int(np.argmax(np.sum(strength, axis=1)))
        for index in range(len(signs)):
            if index == pivot:
                continue
            if strength[pivot, index] > np.finfo(np.float64).eps:
                signs[index] = 1.0 if secondary_a[pivot, index] * secondary_b[pivot, index] >= 0.0 else -1.0
    candidate = va @ np.diag(signs) @ vb.T
    return orthogonal_retraction(np.eye(candidate.shape[0]), candidate - np.eye(candidate.shape[0]))


def deterministic_starts(
    targets: np.ndarray,
    templates: np.ndarray,
    *,
    seed: int,
    count: int = CANDIDATE_SOLVER_SETTINGS.starts,
) -> tuple[np.ndarray, ...]:
    a, b = _validate_banks(targets, templates)
    if int(count) < 4 or int(count) % 2:
        raise ValueError("start count must be an even integer at least four")
    d = a.shape[-1]
    reflection = np.eye(d)
    reflection[0, 0] = -1.0
    bases: list[np.ndarray] = [
        _spectral_start(a, b, solve_signs=True),
        _spectral_start(a, b, solve_signs=False),
    ]
    for offset in range(int(count) // 2 - 2):
        bases.append(_canonical_orthogonal(int(seed) + 104729 * (offset + 1), d))
    starts: list[np.ndarray] = []
    for base in bases:
        reflected = base @ reflection
        starts.extend((base, orthogonal_retraction(np.eye(d), reflected - np.eye(d))))
    determinants = np.sign([np.linalg.det(value) for value in starts])
    if set(determinants.tolist()) != {-1.0, 1.0}:
        raise RuntimeError("deterministic starts failed to cover both O(d) components")
    return tuple(starts)


def _optimize_one_custom(
    targets: np.ndarray,
    templates: np.ndarray,
    start: np.ndarray,
    *,
    start_index: int,
    settings: SolverSettings,
) -> StartResult:
    q = orthogonal_retraction(np.eye(start.shape[0]), np.asarray(start) - np.eye(start.shape[0]))
    objective = action_objective(targets, templates, q)
    line_failures = 0
    previous = np.inf
    converged = False
    iteration = 0
    for iteration in range(1, settings.max_iterations + 1):
        gradient = tangent_projection(q, action_gradient(targets, templates, q))
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= settings.gradient_tolerance:
            converged = True
            break
        relative_change = abs(previous - objective) / max(1.0, abs(previous), abs(objective))
        if np.isfinite(previous) and relative_change <= settings.objective_tolerance and gradient_norm <= 10.0 * settings.gradient_tolerance:
            converged = True
            break
        direction = -gradient
        step = settings.line_search_initial_step_size
        accepted = False
        while step >= settings.optimizer_min_step_size:
            candidate = orthogonal_retraction(q, step * direction)
            candidate_objective = action_objective(targets, templates, candidate)
            if candidate_objective <= objective - settings.line_search_sufficient_decrease * step * gradient_norm**2:
                previous, objective, q = objective, candidate_objective, candidate
                accepted = True
                break
            step *= settings.line_search_contraction_factor
        if not accepted:
            line_failures += 1
            break
    final_gradient = tangent_projection(q, action_gradient(targets, templates, q))
    final_norm = float(np.linalg.norm(final_gradient))
    converged = bool(converged or final_norm <= settings.gradient_tolerance)
    return StartResult(
        start_index=int(start_index),
        matrix=q,
        objective=float(action_objective(targets, templates, q)),
        gradient_norm=final_norm,
        converged=converged,
        iterations=int(iteration),
        determinant=float(np.linalg.det(q)),
        line_search_failures=int(line_failures),
        optimizer="independent_projected_armijo",
        stopping_criterion=(
            "projected_gradient_tolerance"
            if converged
            else "line_search_or_iteration_limit"
        ),
    )


def _pymanopt_problem(
    targets: np.ndarray,
    templates: np.ndarray,
) -> tuple[Stiefel, Problem]:
    """Construct the exact squared-conjugation problem on full O(d).

    Stiefel(d, d) is O(d), not the determinant-restricted SO(d) manifold.
    Pymanopt supplies the tangent projection, polar retraction, vector
    transport, and nonlinear conjugate-gradient update.
    """

    a, b = _validate_banks(targets, templates)
    manifold = Stiefel(a.shape[-1], a.shape[-1], retraction="polar")

    @pymanopt.function.numpy(manifold)
    def cost(action: np.ndarray) -> float:
        return action_objective(a, b, action)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(action: np.ndarray) -> np.ndarray:
        return action_gradient(a, b, action)

    return manifold, Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
    )


def build_pymanopt_optimizer(
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
) -> ConjugateGradient:
    """Instantiate the exact public Pymanopt optimizer contract.

    ``optimizer_min_step_size`` belongs to Pymanopt's optimizer stopping
    criteria. It is deliberately not described as a BackTrackingLineSearcher
    parameter because Pymanopt 2.2.1 does not expose such a line-search option.
    """

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
        max_time=settings.optimizer_max_time_seconds,
        max_iterations=settings.max_iterations,
        min_gradient_norm=settings.gradient_tolerance,
        min_step_size=settings.optimizer_min_step_size,
        max_cost_evaluations=settings.max_iterations + 1,
        verbosity=0,
        # The pairwise amendment sets this to one to preserve evaluated
        # iterates. Other callers retain the historical zero-log default.
        log_verbosity=settings.pymanopt_log_verbosity,
    )


def _optimize_one_pymanopt(
    targets: np.ndarray,
    templates: np.ndarray,
    start: np.ndarray,
    *,
    start_index: int,
    settings: SolverSettings,
) -> StartResult:
    manifold, problem = _pymanopt_problem(targets, templates)
    optimizer = build_pymanopt_optimizer(settings)
    initial = np.asarray(start, dtype=np.float64)
    if initial.shape != (manifold._n, manifold._p):
        raise ValueError("start has the wrong shape")
    orthogonality = float(np.linalg.norm(initial.T @ initial - np.eye(len(initial))))
    if orthogonality > 1.0e-10:
        raise ValueError(f"start is not orthogonal: {orthogonality}")
    result = optimizer.run(problem, initial_point=initial)
    q = np.asarray(result.point, dtype=np.float64)
    stopping = str(result.stopping_criterion)
    if result.log is not None and result.log.get("iterations") is not None:
        log = result.log["iterations"]
        qualifying = [
            index
            for index, value in enumerate(log["gradient_norm"])
            if float(value) <= settings.gradient_tolerance
        ]
        if qualifying:
            q = np.asarray(log["point"][qualifying[-1]], dtype=np.float64)
            stopping = f"{stopping}; returned_logged_tolerance_iterate"
    projected = manifold.projection(q, action_gradient(targets, templates, q))
    gradient_norm = float(manifold.norm(q, projected))
    converged = bool(gradient_norm <= settings.gradient_tolerance)
    # Match the already-audited independent solver's second numerical stop:
    # an objective plateau at machine precision with a projected gradient no
    # larger than 10 times the primary tolerance. This is accepted only after
    # Pymanopt itself reports its frozen minimum-step termination. It prevents
    # an evaluated determinant sector from being mislabeled as unexplored when
    # the polar/line-search arithmetic can no longer change the objective.
    if (
        not converged
        and "min step_size reached" in str(result.stopping_criterion)
        and result.log is not None
        and result.log.get("iterations") is not None
    ):
        log = result.log["iterations"]
        if len(log["cost"]) >= 2:
            previous_cost = float(log["cost"][-2])
            final_cost = float(log["cost"][-1])
            relative_change = abs(final_cost - previous_cost) / max(
                1.0, abs(final_cost), abs(previous_cost)
            )
            if (
                relative_change <= settings.objective_tolerance
                and gradient_norm
                <= settings.objective_stall_gradient_multiplier
                * settings.gradient_tolerance
            ):
                converged = True
                stopping = (
                    f"{stopping}; objective_stall_with_bounded_gradient"
                )
    return StartResult(
        start_index=int(start_index),
        matrix=q,
        objective=float(action_objective(targets, templates, q)),
        gradient_norm=gradient_norm,
        converged=converged,
        iterations=int(result.iterations),
        determinant=float(np.linalg.det(q)),
        line_search_failures=int(result.step_size == 0.0),
        optimizer="pymanopt_2.2.1_stiefel_cg",
        stopping_criterion=stopping,
    )


def equivalence_objective_tolerance(
    best_objective: float,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
) -> float:
    """Frozen additive tolerance in L <= L_best + atol + rtol*|L_best|."""

    value = float(best_objective)
    if not np.isfinite(value):
        raise ValueError("best objective must be finite")
    return float(
        settings.equivalent_objective_absolute_tolerance
        + settings.equivalent_objective_relative_tolerance * abs(value)
    )


def optimize_action(
    targets: np.ndarray,
    templates: np.ndarray,
    *,
    seed: int,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
    starts: Sequence[np.ndarray] | None = None,
    solver: str = "pymanopt",
) -> ActionFit:
    a, b = _validate_banks(targets, templates)
    initial = deterministic_starts(a, b, seed=seed, count=settings.starts) if starts is None else tuple(starts)
    if not initial:
        raise ValueError("at least one start is required")
    if len(initial) != int(settings.starts):
        raise ActionSolverError(
            "configured start-count contract violated: "
            f"expected exactly {settings.starts} total starts, received {len(initial)}"
        )
    if solver == "pymanopt":
        solve_one = _optimize_one_pymanopt
    elif solver == "custom":
        solve_one = _optimize_one_custom
    else:
        raise ValueError("solver must be 'pymanopt' or 'custom'")
    results = tuple(
        solve_one(a, b, value, start_index=index, settings=settings)
        for index, value in enumerate(initial)
    )
    converged_indices = [index for index, value in enumerate(results) if value.converged]
    if not converged_indices:
        best_any = min(results, key=lambda value: (value.objective, value.gradient_norm, value.start_index))
        raise ActionSolverError(
            f"all action starts failed convergence; best objective={best_any.objective:.17g}, gradient={best_any.gradient_norm:.17g}"
        )
    best = min((results[index] for index in converged_indices), key=lambda value: (value.objective, value.gradient_norm, value.start_index))
    tolerance = equivalence_objective_tolerance(best.objective, settings)
    equivalent = tuple(
        value.start_index
        for value in results
        if value.converged and value.objective <= best.objective + tolerance
    )
    return ActionFit(best.matrix, best.start_index, results, equivalent)


def optimize_action_custom(
    targets: np.ndarray,
    templates: np.ndarray,
    *,
    seed: int,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
    starts: Sequence[np.ndarray] | None = None,
) -> ActionFit:
    """Run the independent projected-Armijo validation solver."""

    return optimize_action(
        targets,
        templates,
        seed=seed,
        settings=settings,
        starts=starts,
        solver="custom",
    )


def diagnose_multistart(fit: ActionFit) -> MultistartDiagnostic:
    converged = tuple(value for value in fit.starts if value.converged)
    equivalent = set(fit.equivalent_start_indices)
    sectors = tuple(
        sorted({int(np.sign(value.determinant)) for value in converged})
    )
    non_equivalent = sum(
        value.start_index not in equivalent for value in converged
    )
    failed = len(fit.starts) - len(converged)
    if sectors != (-1, 1):
        classification = "NUMERICAL_OPTIMIZER_INSTABILITY"
    elif non_equivalent:
        classification = "ALTERNATIVE_LOCAL_MINIMA_EXCLUDED"
    elif failed:
        classification = "PARTIAL_START_NONCONVERGENCE"
    else:
        classification = "SAME_OBJECTIVE_CLASS_REACHED"
    return MultistartDiagnostic(
        classification=classification,
        converged_count=len(converged),
        failed_count=failed,
        equivalent_count=len(equivalent),
        converged_non_equivalent_count=non_equivalent,
        determinant_sectors_with_converged_solution=sectors,
    )


def conjugate(action: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    q = np.asarray(action, dtype=np.float64)
    values = np.asarray(matrices, dtype=np.float64)
    if values.shape[-2:] != q.shape or q.ndim != 2:
        raise ValueError("conjugation shapes do not match")
    return symmetrize(np.einsum("ij,...jl,ml->...im", q, values, q, optimize=True))


def source_model_objective(objects: np.ndarray, actions: np.ndarray, templates: np.ndarray) -> float:
    values = np.asarray(objects, dtype=np.float64)
    q = np.asarray(actions, dtype=np.float64)
    b = np.asarray(templates, dtype=np.float64)
    if values.ndim != 5 or b.shape != values.shape[1:] or q.shape != (len(values), values.shape[-1], values.shape[-1]):
        raise ValueError("source model shapes must be (subject,context,class,d,d), (subject,d,d), and (context,class,d,d)")
    prediction = conjugate(q[:, None, None], b[None]) if False else np.stack([conjugate(q[index], b) for index in range(len(q))])
    residual = prediction - values
    return float(np.sum(residual * residual, dtype=np.float64))


def _source_initial_actions(objects: np.ndarray, anchor_index: int, *, start_index: int, seed: int) -> np.ndarray:
    values = np.asarray(objects, dtype=np.float64)
    n, _, _, d, _ = values.shape
    actions = np.tile(np.eye(d), (n, 1, 1))
    if start_index == 0:
        anchor_bank = values[anchor_index].reshape(-1, d, d)
        for index in range(n):
            if index != anchor_index:
                actions[index] = _spectral_start(values[index].reshape(-1, d, d), anchor_bank, solve_signs=True)
    elif start_index > 1:
        for index in range(n):
            if index != anchor_index:
                actions[index] = _canonical_orthogonal(seed + 1009 * start_index + 37 * index, d)
    actions[anchor_index] = np.eye(d)
    return actions


def fit_source_model(
    objects: np.ndarray,
    *,
    anchor_index: int,
    seed: int,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
) -> SourceModelFit:
    values = np.asarray(objects, dtype=np.float64)
    if values.ndim != 5 or values.shape[-1] != values.shape[-2] or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("objects must be finite (subject,context,class,d,d)")
    if not 0 <= int(anchor_index) < len(values):
        raise ValueError("anchor index is out of range")
    values = symmetrize(values)
    results: list[SourceStartResult] = []
    for global_start in range(settings.outer_starts):
        actions = _source_initial_actions(values, int(anchor_index), start_index=global_start, seed=seed)
        previous = np.inf
        inner_results: list[StartResult] = []
        converged = False
        outer_iteration = 0
        for outer_iteration in range(1, settings.outer_iterations + 1):
            aligned = np.stack([conjugate(actions[index].T, values[index]) for index in range(len(values))])
            templates = symmetrize(np.mean(aligned, axis=0, dtype=np.float64))
            inner_results = []
            for index in range(len(values)):
                if index == int(anchor_index):
                    continue
                flattened_targets = values[index].reshape(-1, values.shape[-1], values.shape[-1])
                flattened_templates = templates.reshape(-1, values.shape[-1], values.shape[-1])
                sector_starts = deterministic_starts(
                    flattened_targets,
                    flattened_templates,
                    seed=seed + 7919 * global_start + 101 * index,
                    count=settings.starts,
                )
                fit = optimize_action(
                    flattened_targets,
                    flattened_templates,
                    seed=seed + 7919 * global_start + 101 * index,
                    settings=settings,
                    # The configured count is TOTAL starts. The warm action
                    # occupies one slot; there is no hidden ninth solve.
                    starts=(actions[index], *sector_starts[: settings.starts - 1]),
                )
                if diagnose_multistart(fit).determinant_sectors_with_converged_solution != (-1, 1):
                    raise ActionSolverError(
                        "UNASSESSED_TECHNICAL_FAILURE: source update lacks a converged determinant sector"
                    )
                actions[index] = fit.matrix
                inner_results.append(fit.starts[fit.best_start_index])
            actions[int(anchor_index)] = np.eye(values.shape[-1])
            aligned = np.stack([conjugate(actions[index].T, values[index]) for index in range(len(values))])
            templates = symmetrize(np.mean(aligned, axis=0, dtype=np.float64))
            objective = source_model_objective(values, actions, templates)
            relative = abs(previous - objective) / max(1.0, abs(previous), abs(objective))
            if np.isfinite(previous) and relative <= settings.outer_objective_tolerance and all(value.converged for value in inner_results):
                converged = True
                break
            previous = objective
        maximum_gradient = max((value.gradient_norm for value in inner_results), default=0.0)
        converged = bool(converged and all(value.converged for value in inner_results))
        results.append(
            SourceStartResult(
                start_index=global_start,
                actions=actions.copy(),
                templates=templates.copy(),
                objective=float(source_model_objective(values, actions, templates)),
                maximum_gradient_norm=float(maximum_gradient),
                converged=converged,
                outer_iterations=int(outer_iteration),
                determinants=np.linalg.det(actions),
            )
        )
    converged = [value for value in results if value.converged]
    if not converged:
        best_any = min(results, key=lambda value: (value.objective, value.maximum_gradient_norm, value.start_index))
        raise ActionSolverError(
            f"all generalized-Procrustes starts failed; objective={best_any.objective:.17g}, gradient={best_any.maximum_gradient_norm:.17g}"
        )
    best = min(converged, key=lambda value: (value.objective, value.maximum_gradient_norm, value.start_index))
    tolerance = equivalence_objective_tolerance(best.objective, settings)
    equivalent = tuple(
        value.start_index
        for value in results
        if value.converged and value.objective <= best.objective + tolerance
    )
    return SourceModelFit(
        best.actions,
        best.templates,
        best.start_index,
        tuple(results),
        equivalent,
        int(anchor_index),
    )


def heldout_template(source_actions: np.ndarray, source_heldout: np.ndarray) -> np.ndarray:
    actions = np.asarray(source_actions, dtype=np.float64)
    values = np.asarray(source_heldout, dtype=np.float64)
    if actions.ndim != 3 or values.shape != actions.shape or actions.shape[-1] != actions.shape[-2]:
        raise ValueError("held-out source values/actions must share (subject,d,d) shape")
    return symmetrize(np.mean(np.stack([conjugate(actions[index].T, values[index]) for index in range(len(actions))]), axis=0))


def normalized_prediction_error(target: np.ndarray, prediction: np.ndarray) -> float:
    observed = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum(observed * observed, dtype=np.float64))
    zero_threshold = np.finfo(np.float64).eps**2 * observed.size
    if not np.isfinite(denominator) or denominator <= zero_threshold:
        raise ActionSolverError("UNASSESSED_NUMERICAL_OR_DATA_FAILURE: target U norm is numerical zero")
    residual = observed - predicted
    return float(np.sum(residual * residual, dtype=np.float64) / denominator)


def skew_symmetric_basis(dimension: int) -> np.ndarray:
    """Return an orthonormal Frobenius basis of skew(d)."""

    d = int(dimension)
    if d < 2:
        return np.empty((0, d, d), dtype=np.float64)
    basis = []
    scale = 1.0 / np.sqrt(2.0)
    for row in range(d):
        for column in range(row + 1, d):
            value = np.zeros((d, d), dtype=np.float64)
            value[row, column] = scale
            value[column, row] = -scale
            basis.append(value)
    return np.asarray(basis)


def analyze_common_stabilizer(
    templates: np.ndarray,
    *,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
) -> StabilizerDiagnostic:
    """Analyze continuous common-stabilizer directions in skew(d).

    For each skew-symmetric Omega this constructs the simultaneous linear map
    Omega -> (Omega B_c - B_c Omega)_c.  Its nullspace is the Lie algebra of
    the identity component of the common stabilizer.  Discrete stabilizers are
    intentionally not claimed to be captured here; multi-start predictive
    equivalence remains a separate mandatory diagnostic.
    """

    values = np.asarray(templates, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != values.shape[-2]:
        raise ValueError("templates must have shape (class,d,d)")
    values = symmetrize(values)
    basis = skew_symmetric_basis(values.shape[-1])
    if len(basis) == 0:
        return StabilizerDiagnostic(
            singular_values=np.empty(0),
            numerical_tolerance=0.0,
            numerical_nullity=0,
            approximate_tolerance=0.0,
            approximate_nullity=0,
            skew_basis=basis,
            nullspace_basis=basis,
        )
    columns = []
    for omega in basis:
        commutators = np.stack(
            [omega @ matrix - matrix @ omega for matrix in values]
        )
        columns.append(commutators.reshape(-1))
    operator = np.stack(columns, axis=1)
    _, singular_values, vt = np.linalg.svd(operator, full_matrices=False)
    maximum = float(singular_values[0]) if len(singular_values) else 0.0
    numerical_tolerance = float(
        max(operator.shape) * np.finfo(np.float64).eps * maximum
    )
    approximate_tolerance = float(
        max(numerical_tolerance, settings.commutant_relative_tolerance * maximum)
    )
    numerical_mask = singular_values <= numerical_tolerance
    approximate_mask = singular_values <= approximate_tolerance
    coefficients = vt[numerical_mask]
    nullspace = np.einsum("ab,bij->aij", coefficients, basis, optimize=True)
    return StabilizerDiagnostic(
        singular_values=singular_values,
        numerical_tolerance=numerical_tolerance,
        numerical_nullity=int(np.count_nonzero(numerical_mask)),
        approximate_tolerance=approximate_tolerance,
        approximate_nullity=int(np.count_nonzero(approximate_mask)),
        skew_basis=basis,
        nullspace_basis=nullspace,
    )


def stabilizer_augmented_actions(
    fit: ActionFit,
    diagnostic: StabilizerDiagnostic,
    fit_targets: np.ndarray,
    fit_templates: np.ndarray,
    *,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
) -> tuple[np.ndarray, ...]:
    """Collect exactly criterion-qualified multi-start/Lie-equivalent actions."""

    best = fit.starts[fit.best_start_index]
    limit = best.objective + equivalence_objective_tolerance(
        best.objective, settings
    )
    candidates = [best.matrix]
    candidates.extend(
        fit.starts[index].matrix
        for index in fit.equivalent_start_indices
        if index != fit.best_start_index
    )
    best = fit.matrix
    for omega in diagnostic.nullspace_basis:
        for angle in (0.5 * np.pi, -0.5 * np.pi, np.pi):
            candidates.append(best @ expm(angle * omega))
    unique: list[np.ndarray] = []
    for action in candidates:
        objective = action_objective(fit_targets, fit_templates, action)
        if objective > limit:
            continue
        if not any(np.linalg.norm(action - existing) <= 1.0e-12 for existing in unique):
            unique.append(np.asarray(action, dtype=np.float64))
    if not unique:
        raise ActionSolverError("UNASSESSED_TECHNICAL_FAILURE: Q_eq is empty")
    return tuple(unique)


def prediction_norm_epsilon(prediction: np.ndarray) -> float:
    values = np.asarray(prediction, dtype=np.float64)
    return float(np.finfo(np.float64).eps * np.sqrt(values.size))


def classify_prediction_matrices(
    predictions: Sequence[np.ndarray],
    *,
    best_prediction: np.ndarray,
    split_half_prediction_a: np.ndarray,
    split_half_prediction_b: np.ndarray,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
) -> PredictiveIdentifiability:
    """Apply the exact cell-level predictive-identifiability contract.

    Both D_eq and D_split use max(||P_best||_F, epsilon) as their denominator.
    The split-half predictions are independently recomputed inputs. No
    multiplicative factor is applied to D_split.
    """

    values = tuple(np.asarray(value, dtype=np.float64) for value in predictions)
    if not values:
        raise ValueError("at least one near-optimal prediction is required")
    best = np.asarray(best_prediction, dtype=np.float64)
    half_a = np.asarray(split_half_prediction_a, dtype=np.float64)
    half_b = np.asarray(split_half_prediction_b, dtype=np.float64)
    if any(value.shape != best.shape for value in (*values, half_a, half_b)):
        raise ValueError("all held-out predictions must share shape")
    if not all(np.isfinite(value).all() for value in (*values, best, half_a, half_b)):
        raise ActionSolverError(
            "UNASSESSED_NUMERICAL_OR_DATA_FAILURE: nonfinite prediction"
        )
    epsilon = prediction_norm_epsilon(best)
    best_norm = float(np.linalg.norm(best))
    half_a_norm = float(np.linalg.norm(half_a))
    half_b_norm = float(np.linalg.norm(half_b))
    if best_norm <= epsilon or half_a_norm <= epsilon or half_b_norm <= epsilon:
        raise ActionSolverError(
            "UNASSESSED_NUMERICAL_OR_DATA_FAILURE: held-out prediction norm is numerical zero"
        )
    denominator = max(best_norm, epsilon)
    maximum = 0.0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            maximum = max(
                maximum,
                float(np.linalg.norm(values[left] - values[right]))
                / denominator,
            )
    split_half = float(np.linalg.norm(half_a - half_b) / denominator)
    threshold = max(settings.prediction_dispersion_numerical_floor, split_half)
    if maximum > threshold:
        classification = "PREDICTIVE_NONIDENTIFIABILITY"
    elif len(values) > 1:
        classification = "HARMLESS_Q_NONUNIQUENESS"
    else:
        classification = "PREDICTIVELY_IDENTIFIABLE"
    return PredictiveIdentifiability(
        classification=classification,
        equivalent_solution_count=len(values),
        maximum_relative_prediction_dispersion=float(maximum),
        split_half_relative_variability=split_half,
        materiality_threshold=float(threshold),
        prediction_normalization=denominator,
        normalization_epsilon=epsilon,
        prediction_hashes=tuple(sha256_array(value) for value in values),
    )


def classify_prediction_set(
    actions: Sequence[np.ndarray],
    heldout_template_matrix: np.ndarray,
    *,
    split_half_prediction_a: np.ndarray,
    split_half_prediction_b: np.ndarray,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
) -> PredictiveIdentifiability:
    matrices = tuple(np.asarray(value, dtype=np.float64) for value in actions)
    if not matrices:
        raise ValueError("at least one action is required")
    heldout = np.asarray(heldout_template_matrix, dtype=np.float64)
    predictions = tuple(conjugate(value, heldout) for value in matrices)
    return classify_prediction_matrices(
        predictions,
        best_prediction=predictions[0],
        split_half_prediction_a=split_half_prediction_a,
        split_half_prediction_b=split_half_prediction_b,
        settings=settings,
    )


def assess_predictive_identifiability(
    fit: ActionFit,
    fit_targets: np.ndarray,
    fit_templates: np.ndarray,
    heldout_template_matrix: np.ndarray,
    *,
    split_half_prediction_a: np.ndarray,
    split_half_prediction_b: np.ndarray,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
) -> tuple[StabilizerDiagnostic, PredictiveIdentifiability]:
    landscape = diagnose_multistart(fit)
    if landscape.determinant_sectors_with_converged_solution != (-1, 1):
        raise ActionSolverError(
            "UNASSESSED_TECHNICAL_FAILURE: no converged solution in both determinant sectors"
        )
    stabilizer = analyze_common_stabilizer(fit_templates, settings=settings)
    actions = stabilizer_augmented_actions(
        fit,
        stabilizer,
        fit_targets,
        fit_templates,
        settings=settings,
    )
    assessment = classify_prediction_set(
        actions,
        heldout_template_matrix,
        split_half_prediction_a=split_half_prediction_a,
        split_half_prediction_b=split_half_prediction_b,
        settings=settings,
    )
    return stabilizer, assessment


def prediction_ambiguity(
    fit: ActionFit,
    heldout_template_matrix: np.ndarray,
    target: np.ndarray,
) -> tuple[float, tuple[str, ...]]:
    predictions = [conjugate(fit.starts[index].matrix, heldout_template_matrix) for index in fit.equivalent_start_indices]
    hashes = tuple(sha256_array(value) for value in predictions)
    if len(predictions) < 2:
        return 0.0, hashes
    denominator = max(float(np.linalg.norm(target)), np.finfo(np.float64).tiny)
    maximum = max(
        float(np.linalg.norm(predictions[left] - predictions[right])) / denominator
        for left in range(len(predictions))
        for right in range(left + 1, len(predictions))
    )
    return maximum, hashes


def nonidentity_permutations_three() -> tuple[tuple[int, int, int], ...]:
    import itertools

    identity = (0, 1, 2)
    values = tuple(value for value in itertools.permutations(identity) if value != identity)
    if len(values) != 5:
        raise RuntimeError("S3 nonidentity enumeration failure")
    return values


__all__ = [
    "ActionFit",
    "ActionSolverError",
    "CANDIDATE_SOLVER_SETTINGS",
    "MASTER_SEED",
    "MultistartDiagnostic",
    "PredictiveIdentifiability",
    "SolverSettings",
    "SourceModelFit",
    "StabilizerDiagnostic",
    "action_gradient",
    "action_objective",
    "analyze_common_stabilizer",
    "assess_predictive_identifiability",
    "build_pymanopt_optimizer",
    "classify_prediction_matrices",
    "classify_prediction_set",
    "conjugate",
    "deterministic_starts",
    "diagnose_multistart",
    "equivalence_objective_tolerance",
    "fit_source_model",
    "heldout_template",
    "nonidentity_permutations_three",
    "normalized_prediction_error",
    "optimize_action",
    "optimize_action_custom",
    "orthogonal_retraction",
    "prediction_ambiguity",
    "prediction_norm_epsilon",
    "sha256_array",
    "skew_symmetric_basis",
    "stabilizer_augmented_actions",
    "symmetrize",
    "tangent_projection",
]
