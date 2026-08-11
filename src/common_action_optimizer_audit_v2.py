"""Synthetic-only optimizer stress audit for orthogonal conjugation.

This module is deliberately array-only. It has no BNCI loader and does not
compute a scientific gain, group statistic, null, or p-value.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions
from pymanopt.optimizers.line_search import BackTrackingLineSearcher

from src.common_action_solver_v0 import (
    CANDIDATE_SOLVER_SETTINGS,
    SolverSettings,
    action_gradient,
    action_objective,
    conjugate,
    deterministic_starts,
)


AUDIT_SEED = 20260811
AUDIT_SETTINGS = replace(
    CANDIDATE_SOLVER_SETTINGS,
    starts=4,
    pymanopt_log_verbosity=1,
)
OPTIMIZERS = ("conjugate_gradient", "trust_regions", "steepest_descent")
WINDOWS = (10, 50, 100, 250)


@dataclass(frozen=True)
class SyntheticFixture:
    name: str
    family: str
    truth_determinant: int
    fit_targets: np.ndarray
    fit_templates: np.ndarray
    heldout_template: np.ndarray
    heldout_truth: np.ndarray
    noise_scale: float
    seed: int


@dataclass(frozen=True)
class OptimizerRun:
    fixture: str
    family: str
    truth_determinant: int
    optimizer: str
    start_index: int
    initial_determinant: int
    final_determinant: int
    determinant_preserved: bool
    initial_objective: float
    final_objective: float
    best_objective: float
    initial_gradient_norm: float
    final_gradient_norm: float
    minimum_gradient_norm: float
    iterations: int
    runtime_seconds: float
    stopping_criterion: str
    strict_gradient_converged: bool
    current_contract_converged: bool
    current_plateau_clause_used: bool
    heldout_relative_error: float
    maximum_orthogonality_error: float
    accepted_outer_updates: int | None
    rejected_line_search_trials: int | None
    final_step_size: float | None
    median_iterate_displacement_last_50: float | None
    maximum_iterate_displacement_last_50: float | None
    heldout_prediction_spread_last_50: float | None
    objective_change_last_10: float | None
    objective_change_last_50: float | None
    objective_change_last_100: float | None
    objective_change_last_250: float | None
    objective_increase_count: int | None
    trajectory_classification: str
    start_sha256: str


def _symmetrize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 0.5 * (array + np.swapaxes(array, -1, -2))


def _normalize_bank(values: np.ndarray) -> np.ndarray:
    bank = _symmetrize(values)
    norms = np.linalg.norm(bank.reshape(len(bank), -1), axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("synthetic matrix has numerical-zero norm")
    return bank / norms[:, None, None]


def _orthogonal(seed: int, dimension: int, determinant: int) -> np.ndarray:
    rng = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence([AUDIT_SEED, int(seed)]))
    )
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    if int(np.sign(np.linalg.det(q))) != int(determinant):
        q[:, 0] *= -1.0
    return q


def _random_symmetric(seed: int, count: int, dimension: int) -> np.ndarray:
    rng = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence([AUDIT_SEED, int(seed)]))
    )
    values = rng.normal(size=(count, dimension, dimension))
    return _normalize_bank(values)


def _generic_bank(seed: int, dimension: int) -> np.ndarray:
    return _random_symmetric(seed, 4, dimension)


def _nearly_commuting_bank(seed: int, dimension: int) -> np.ndarray:
    rng = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence([AUDIT_SEED, int(seed)]))
    )
    diagonal = rng.normal(size=(4, dimension))
    perturbation = _symmetrize(rng.normal(size=(4, dimension, dimension)))
    perturbation -= np.eye(dimension)[None] * np.diagonal(
        perturbation, axis1=-2, axis2=-1
    )[:, :, None]
    values = np.stack([np.diag(row) for row in diagonal]) + 1.0e-3 * perturbation
    return _normalize_bank(values)


def _clustered_spectrum_bank(seed: int, dimension: int) -> np.ndarray:
    if dimension % 2:
        raise ValueError("clustered-spectrum fixture requires an even dimension")
    base = np.repeat(np.linspace(-1.0, 1.0, dimension // 2), 2)
    values = []
    for index in range(4):
        eigenvectors = _orthogonal(seed + 10 * index, dimension, 1)
        jitter = 1.0e-5 * np.tile([-1.0, 1.0], dimension // 2)
        eigenvalues = base + (index + 1) * jitter
        values.append(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)
    return _normalize_bank(np.stack(values))


def _approximate_stabilizer_bank(seed: int, dimension: int) -> np.ndarray:
    if dimension < 8:
        raise ValueError("approximate-stabilizer fixture requires dimension >= 8")
    rng = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence([AUDIT_SEED, int(seed)]))
    )
    shared_basis = _orthogonal(seed + 1, dimension, 1)
    values = []
    block_size = 6
    for index in range(4):
        eigenvalues = np.linspace(-0.9, 0.9, dimension)
        eigenvalues[:block_size] = -0.4 + 0.2 * index
        core = shared_basis @ np.diag(eigenvalues) @ shared_basis.T
        perturbation = _symmetrize(rng.normal(size=(dimension, dimension)))
        values.append(core + 1.0e-4 * perturbation)
    return _normalize_bank(np.stack(values))


def _ill_conditioned_bank(seed: int, dimension: int) -> np.ndarray:
    magnitudes = np.logspace(0.0, -8.0, dimension)
    values = []
    for index in range(4):
        eigenvectors = _orthogonal(seed + 10 * index, dimension, 1)
        signs = np.where((np.arange(dimension) + index) % 2, -1.0, 1.0)
        values.append(
            eigenvectors @ np.diag(signs * magnitudes) @ eigenvectors.T
        )
    return _normalize_bank(np.stack(values))


def synthetic_stress_fixtures(dimension: int = 22) -> tuple[SyntheticFixture, ...]:
    """Return the preregistered 12-case d=22 stress grid."""

    families = (
        ("generic_exact", _generic_bank, 0.0),
        ("generic_noisy", _generic_bank, 2.0e-4),
        ("nearly_commuting", _nearly_commuting_bank, 1.0e-5),
        ("clustered_spectrum", _clustered_spectrum_bank, 1.0e-5),
        ("approximate_stabilizer", _approximate_stabilizer_bank, 1.0e-6),
        ("ill_conditioned", _ill_conditioned_bank, 1.0e-7),
    )
    fixtures: list[SyntheticFixture] = []
    for family_index, (family, builder, noise_scale) in enumerate(families):
        template_seed = 1000 + 100 * family_index
        templates = builder(template_seed, dimension)
        for determinant in (1, -1):
            fixture_seed = template_seed + (1 if determinant > 0 else 2)
            truth = _orthogonal(fixture_seed, dimension, determinant)
            targets = conjugate(truth, templates)
            if noise_scale:
                rng = np.random.Generator(
                    np.random.PCG64DXSM(
                        np.random.SeedSequence([AUDIT_SEED, fixture_seed, 99])
                    )
                )
                noise = _symmetrize(
                    rng.normal(
                        scale=noise_scale,
                        size=(3, dimension, dimension),
                    )
                )
                fit_targets = targets[:3] + noise
            else:
                fit_targets = targets[:3]
            fixtures.append(
                SyntheticFixture(
                    name=f"{family}__det_{'positive' if determinant > 0 else 'negative'}",
                    family=family,
                    truth_determinant=determinant,
                    fit_targets=fit_targets,
                    fit_templates=templates[:3],
                    heldout_template=templates[3],
                    heldout_truth=targets[3],
                    noise_scale=noise_scale,
                    seed=fixture_seed,
                )
            )
    return tuple(fixtures)


def action_hessian_vector(
    targets: np.ndarray,
    templates: np.ndarray,
    action: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Analytic Euclidean Hessian-vector product of the frozen loss."""

    a = np.asarray(targets, dtype=np.float64)
    b = np.asarray(templates, dtype=np.float64)
    q = np.asarray(action, dtype=np.float64)
    h = np.asarray(direction, dtype=np.float64)
    result = np.zeros_like(q)
    for target, template in zip(a, b, strict=True):
        residual = q @ template @ q.T - target
        residual_direction = h @ template @ q.T + q @ template @ h.T
        result += 4.0 * (
            residual_direction @ q @ template + residual @ h @ template
        )
    return result


def _problem(
    targets: np.ndarray,
    templates: np.ndarray,
) -> tuple[Stiefel, Problem]:
    dimension = targets.shape[-1]
    manifold = Stiefel(dimension, dimension, retraction="polar")

    @pymanopt.function.numpy(manifold)
    def cost(action: np.ndarray) -> float:
        return action_objective(targets, templates, action)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(action: np.ndarray) -> np.ndarray:
        return action_gradient(targets, templates, action)

    @pymanopt.function.numpy(manifold)
    def euclidean_hessian(
        action: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        return action_hessian_vector(
            targets, templates, action, direction
        )

    return manifold, Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )


def _line_search(settings: SolverSettings) -> BackTrackingLineSearcher:
    return BackTrackingLineSearcher(
        contraction_factor=settings.line_search_contraction_factor,
        optimism=settings.line_search_optimism,
        sufficient_decrease=settings.line_search_sufficient_decrease,
        max_iterations=settings.line_search_max_iterations,
        initial_step_size=settings.line_search_initial_step_size,
    )


def _optimizer(name: str, settings: SolverSettings) -> Any:
    common = dict(
        max_time=settings.optimizer_max_time_seconds,
        max_iterations=settings.max_iterations,
        min_gradient_norm=settings.gradient_tolerance,
        min_step_size=settings.optimizer_min_step_size,
        max_cost_evaluations=settings.max_iterations + 1,
        verbosity=0,
    )
    if name == "conjugate_gradient":
        return ConjugateGradient(
            beta_rule="HestenesStiefel",
            line_searcher=_line_search(settings),
            log_verbosity=1,
            **common,
        )
    if name == "steepest_descent":
        return SteepestDescent(
            line_searcher=_line_search(settings),
            log_verbosity=1,
            **common,
        )
    if name == "trust_regions":
        return TrustRegions(
            miniter=3,
            kappa=0.1,
            theta=1.0,
            rho_prime=0.1,
            use_rand=False,
            rho_regularization=1000.0,
            log_verbosity=0,
            **common,
        )
    raise ValueError(f"unknown optimizer: {name}")


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _trace_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    if result.log is None or result.log.get("iterations") is None:
        return np.empty(0), np.empty(0), ()
    log = result.log["iterations"]
    if log is None or "cost" not in log:
        return np.empty(0), np.empty(0), ()
    return (
        np.asarray(log["cost"], dtype=np.float64),
        np.asarray(log["gradient_norm"], dtype=np.float64),
        tuple(np.asarray(value, dtype=np.float64) for value in log["point"]),
    )


def _relative_objective_change(costs: np.ndarray, window: int) -> float | None:
    if len(costs) < window + 1:
        return None
    initial = float(costs[-window - 1])
    final = float(costs[-1])
    return float((initial - final) / max(1.0, abs(initial), abs(final)))


def _trace_geometry(
    points: Sequence[np.ndarray],
    heldout_template: np.ndarray,
) -> tuple[float | None, float | None, float | None, int | None, float]:
    if not points:
        return None, None, None, None, float("nan")
    dimension = points[0].shape[0]
    orthogonality = max(
        float(np.linalg.norm(point.T @ point - np.eye(dimension)))
        for point in points
    )
    if len(points) < 2:
        return None, None, 0.0, 0, orthogonality
    recent = tuple(points[-51:])
    displacements = np.asarray(
        [
            np.linalg.norm(recent[index + 1] - recent[index])
            / np.sqrt(dimension)
            for index in range(len(recent) - 1)
        ]
    )
    reference = conjugate(recent[-1], heldout_template)
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).eps)
    prediction_spread = max(
        float(np.linalg.norm(conjugate(point, heldout_template) - reference))
        / denominator
        for point in recent
    )
    return (
        float(np.median(displacements)),
        float(np.max(displacements)),
        prediction_spread,
        len(points) - 1,
        orthogonality,
    )


def _trajectory_classification(
    costs: np.ndarray,
    gradients: np.ndarray,
    stopping: str,
    maximum_orthogonality_error: float,
    determinant_preserved: bool,
    tolerance: float,
) -> str:
    if (
        not np.isfinite(costs).all()
        or not np.isfinite(gradients).all()
        or maximum_orthogonality_error > 1.0e-8
        or not determinant_preserved
    ):
        return "numerically_unstable"
    if len(costs) >= 2:
        increases = np.count_nonzero(np.diff(costs) > 1.0e-12 * np.maximum(1.0, np.abs(costs[:-1])))
        if increases > max(2, len(costs) // 20):
            return "oscillating"
    if "min step_size reached" in stopping:
        return "line_search_stalled"
    if len(costs) >= 101:
        change = _relative_objective_change(costs, 100)
        if change is not None and abs(change) <= 1.0e-12 and gradients[-1] > tolerance:
            return "plateaued_nonzero_gradient"
        if change is not None and change > 1.0e-10 and gradients[-1] > tolerance:
            return "still_descending"
    if gradients.size and gradients[-1] <= tolerance:
        return "gradient_converged"
    if "max iterations" in stopping:
        return "max_iterations_unclassified"
    return "other_stopping"


def run_one_start(
    fixture: SyntheticFixture,
    optimizer_name: str,
    start: np.ndarray,
    start_index: int,
    *,
    settings: SolverSettings = AUDIT_SETTINGS,
) -> OptimizerRun:
    manifold, problem = _problem(fixture.fit_targets, fixture.fit_templates)
    optimizer = _optimizer(optimizer_name, settings)
    started = time.perf_counter()
    result = optimizer.run(problem, initial_point=np.asarray(start))
    runtime = time.perf_counter() - started
    costs, gradients, points = _trace_arrays(result)
    selected = np.asarray(result.point, dtype=np.float64)
    if gradients.size:
        qualifying = np.flatnonzero(gradients <= settings.gradient_tolerance)
        if qualifying.size:
            selected = points[int(qualifying[-1])]
    projected = manifold.projection(
        selected,
        action_gradient(fixture.fit_targets, fixture.fit_templates, selected),
    )
    final_gradient = float(manifold.norm(selected, projected))
    final_objective = float(
        action_objective(fixture.fit_targets, fixture.fit_templates, selected)
    )
    if costs.size:
        trace_costs = np.concatenate([costs, [final_objective]])
        trace_gradients = np.concatenate([gradients, [final_gradient]])
        trace_points = (*points, selected)
    else:
        initial_projected = manifold.projection(
            start,
            action_gradient(fixture.fit_targets, fixture.fit_templates, start),
        )
        trace_costs = np.asarray(
            [
                action_objective(fixture.fit_targets, fixture.fit_templates, start),
                final_objective,
            ]
        )
        trace_gradients = np.asarray(
            [manifold.norm(start, initial_projected), final_gradient]
        )
        trace_points = (np.asarray(start), selected)
    median_step, maximum_step, prediction_spread, accepted, orthogonality = (
        _trace_geometry(trace_points, fixture.heldout_template)
    )
    determinants = np.asarray([np.linalg.det(point) for point in trace_points])
    initial_determinant = int(np.sign(np.linalg.det(start)))
    determinant_preserved = bool(
        np.all(np.sign(determinants) == initial_determinant)
    )
    strict = bool(final_gradient <= settings.gradient_tolerance)
    stopping = str(result.stopping_criterion)
    plateau = False
    if (
        not strict
        and "min step_size reached" in stopping
        and len(trace_costs) >= 2
    ):
        relative_change = abs(trace_costs[-1] - trace_costs[-2]) / max(
            1.0, abs(trace_costs[-1]), abs(trace_costs[-2])
        )
        plateau = bool(
            relative_change <= settings.objective_tolerance
            and final_gradient
            <= settings.objective_stall_gradient_multiplier
            * settings.gradient_tolerance
        )
    prediction = conjugate(selected, fixture.heldout_template)
    heldout_error = float(
        np.linalg.norm(prediction - fixture.heldout_truth)
        / np.linalg.norm(fixture.heldout_truth)
    )
    increases = (
        int(
            np.count_nonzero(
                np.diff(trace_costs)
                > 1.0e-12 * np.maximum(1.0, np.abs(trace_costs[:-1]))
            )
        )
        if len(trace_costs) >= 2
        else None
    )
    changes = {
        window: _relative_objective_change(trace_costs, window)
        for window in WINDOWS
    }
    return OptimizerRun(
        fixture=fixture.name,
        family=fixture.family,
        truth_determinant=fixture.truth_determinant,
        optimizer=optimizer_name,
        start_index=start_index,
        initial_determinant=initial_determinant,
        final_determinant=int(np.sign(np.linalg.det(selected))),
        determinant_preserved=determinant_preserved,
        initial_objective=float(trace_costs[0]),
        final_objective=final_objective,
        best_objective=float(np.min(trace_costs)),
        initial_gradient_norm=float(trace_gradients[0]),
        final_gradient_norm=final_gradient,
        minimum_gradient_norm=float(np.min(trace_gradients)),
        iterations=int(result.iterations),
        runtime_seconds=float(runtime),
        stopping_criterion=stopping,
        strict_gradient_converged=strict,
        current_contract_converged=bool(strict or plateau),
        current_plateau_clause_used=plateau,
        heldout_relative_error=heldout_error,
        maximum_orthogonality_error=orthogonality,
        accepted_outer_updates=accepted if optimizer_name != "trust_regions" else None,
        rejected_line_search_trials=None,
        final_step_size=(
            float(result.step_size)
            if hasattr(result, "step_size") and result.step_size is not None
            else None
        ),
        median_iterate_displacement_last_50=median_step,
        maximum_iterate_displacement_last_50=maximum_step,
        heldout_prediction_spread_last_50=prediction_spread,
        objective_change_last_10=changes[10],
        objective_change_last_50=changes[50],
        objective_change_last_100=changes[100],
        objective_change_last_250=changes[250],
        objective_increase_count=increases,
        trajectory_classification=_trajectory_classification(
            trace_costs,
            trace_gradients,
            stopping,
            orthogonality,
            determinant_preserved,
            settings.gradient_tolerance,
        ),
        start_sha256=_array_sha256(start),
    )


def run_stress_suite(
    *,
    settings: SolverSettings = AUDIT_SETTINGS,
    optimizer_names: Sequence[str] = OPTIMIZERS,
) -> tuple[tuple[OptimizerRun, ...], tuple[dict[str, Any], ...]]:
    runs: list[OptimizerRun] = []
    fixture_rows: list[dict[str, Any]] = []
    for fixture in synthetic_stress_fixtures(22):
        starts = deterministic_starts(
            fixture.fit_targets,
            fixture.fit_templates,
            seed=fixture.seed,
            count=settings.starts,
        )
        start_hashes = tuple(_array_sha256(start) for start in starts)
        if len(starts) != 4:
            raise RuntimeError("stress audit requires exactly four total starts")
        if [int(np.sign(np.linalg.det(start))) for start in starts].count(1) != 2:
            raise RuntimeError("stress starts do not balance determinant sectors")
        for optimizer_name in optimizer_names:
            optimizer_runs = [
                run_one_start(
                    fixture,
                    optimizer_name,
                    start,
                    start_index,
                    settings=settings,
                )
                for start_index, start in enumerate(starts)
            ]
            if tuple(value.start_sha256 for value in optimizer_runs) != start_hashes:
                raise RuntimeError("optimizer did not receive the identical start set")
            runs.extend(optimizer_runs)
            sector_certified = {
                sector: any(
                    value.current_contract_converged
                    and value.initial_determinant == sector
                    for value in optimizer_runs
                )
                for sector in (-1, 1)
            }
            converged = [
                value for value in optimizer_runs if value.current_contract_converged
            ]
            best = min(
                converged,
                key=lambda value: (
                    value.final_objective,
                    value.final_gradient_norm,
                    value.start_index,
                ),
            ) if converged else None
            fixture_rows.append(
                {
                    "fixture": fixture.name,
                    "family": fixture.family,
                    "truth_determinant": fixture.truth_determinant,
                    "optimizer": optimizer_name,
                    "converged_starts": len(converged),
                    "det_negative_certified": sector_certified[-1],
                    "det_positive_certified": sector_certified[1],
                    "both_sectors_certified": all(sector_certified.values()),
                    "best_objective": None if best is None else best.final_objective,
                    "best_gradient_norm": None if best is None else best.final_gradient_norm,
                    "best_heldout_relative_error": None if best is None else best.heldout_relative_error,
                    "total_runtime_seconds": sum(value.runtime_seconds for value in optimizer_runs),
                    "maximum_iterations": max(value.iterations for value in optimizer_runs),
                    "start_hashes": "|".join(start_hashes),
                }
            )
    return tuple(runs), tuple(fixture_rows)


def summarize_stress_suite(
    runs: Sequence[OptimizerRun],
    fixture_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dimension": 22,
        "fixtures": 12,
        "starts_per_fixture": 4,
        "scientific_objective": "sum_i ||A_i - Q B_i Q^T||_F^2 on O(22)",
        "optimizers": {},
    }
    for optimizer_name in OPTIMIZERS:
        selected_runs = [value for value in runs if value.optimizer == optimizer_name]
        selected_fixtures = [
            value for value in fixture_rows if value["optimizer"] == optimizer_name
        ]
        stable_above_gradient = [
            value
            for value in selected_runs
            if not value.strict_gradient_converged
            and value.objective_change_last_50 is not None
            and abs(value.objective_change_last_50) <= 1.0e-10
            and value.heldout_prediction_spread_last_50 is not None
            and value.heldout_prediction_spread_last_50 <= 1.0e-8
        ]
        summary["optimizers"][optimizer_name] = {
            "runs": len(selected_runs),
            "strict_gradient_converged": sum(
                value.strict_gradient_converged for value in selected_runs
            ),
            "current_contract_converged": sum(
                value.current_contract_converged for value in selected_runs
            ),
            "current_plateau_clause_used": sum(
                value.current_plateau_clause_used for value in selected_runs
            ),
            "both_sectors_certified_fixtures": sum(
                bool(value["both_sectors_certified"])
                for value in selected_fixtures
            ),
            "median_runtime_seconds_per_start": float(
                np.median([value.runtime_seconds for value in selected_runs])
            ),
            "total_runtime_seconds": float(
                sum(value.runtime_seconds for value in selected_runs)
            ),
            "median_iterations": float(
                np.median([value.iterations for value in selected_runs])
            ),
            "maximum_iterations": max(value.iterations for value in selected_runs),
            "median_best_heldout_relative_error": float(
                np.median(
                    [
                        value["best_heldout_relative_error"]
                        for value in selected_fixtures
                        if value["best_heldout_relative_error"] is not None
                    ]
                )
            ),
            "stable_action_but_above_gradient_tolerance": len(stable_above_gradient),
            "trajectory_classifications": {
                classification: sum(
                    value.trajectory_classification == classification
                    for value in selected_runs
                )
                for classification in sorted(
                    {value.trajectory_classification for value in selected_runs}
                )
            },
        }
    return summary


def run_rows(runs: Sequence[OptimizerRun]) -> list[dict[str, Any]]:
    return [asdict(value) for value in runs]


__all__ = [
    "AUDIT_SETTINGS",
    "OPTIMIZERS",
    "OptimizerRun",
    "SyntheticFixture",
    "action_hessian_vector",
    "run_one_start",
    "run_rows",
    "run_stress_suite",
    "summarize_stress_suite",
    "synthetic_stress_fixtures",
]
