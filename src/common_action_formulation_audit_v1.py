"""Synthetic-only formulation audit for the common subject action hypothesis.

This module intentionally has no repository, cache, output, or dataset loader.
It provides algebraic objective checks and small reference optimizers only; it
is not a production BNCI analysis pipeline.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Euclidean, Product, Stiefel

from src.common_action_solver_v0 import (
    CANDIDATE_SOLVER_SETTINGS,
    ActionFit,
    SolverSettings,
    action_gradient,
    action_objective,
    build_pymanopt_optimizer,
    conjugate,
    deterministic_starts,
    optimize_action,
    symmetrize,
)


@dataclass(frozen=True)
class ReferenceFit:
    """Result from one non-production product-manifold reference solve."""

    actions: np.ndarray
    templates: np.ndarray
    objective: float
    gradient_norm: float
    iterations: int
    converged: bool
    elapsed_seconds: float
    optimizer: str


@dataclass(frozen=True)
class CycleDiagnostic:
    """Minimum induced-action discrepancy over finite equivalence sets."""

    relative_discrepancy: float
    normalization: float
    direct_index: int
    first_index: int
    second_index: int


def _validate_objects(objects: np.ndarray) -> np.ndarray:
    values = np.asarray(objects, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[-1] != values.shape[-2]
        or values.shape[0] < 2
        or values.shape[1] < 1
        or not np.isfinite(values).all()
    ):
        raise ValueError("objects must be finite (subject,class,d,d)")
    scale = max(float(np.linalg.norm(values)), 1.0)
    if float(np.linalg.norm(values - values.transpose(0, 1, 3, 2))) > 1.0e-12 * scale:
        raise ValueError("objects must be symmetric")
    return symmetrize(values)


def _validate_actions(actions: np.ndarray, subjects: int, dimension: int) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.shape != (subjects, dimension, dimension):
        raise ValueError("actions must have shape (subject,d,d)")
    errors = np.linalg.norm(
        values.transpose(0, 2, 1) @ values - np.eye(dimension)[None],
        axis=(1, 2),
    )
    if np.any(errors > 1.0e-10):
        raise ValueError("every action must be orthogonal")
    return values


def latent_least_squares_objective(
    objects: np.ndarray,
    actions: np.ndarray,
    templates: np.ndarray,
) -> float:
    """Formulation A: sum_{r,c} ||U_rc - Q_r B_c Q_r^T||_F^2."""

    values = _validate_objects(objects)
    q = _validate_actions(actions, len(values), values.shape[-1])
    b = np.asarray(templates, dtype=np.float64)
    if b.shape != values.shape[1:]:
        raise ValueError("templates must have shape (class,d,d)")
    predictions = np.stack([conjugate(q[index], b) for index in range(len(q))])
    residuals = values - predictions
    return float(np.sum(residuals * residuals, dtype=np.float64))


def profiled_templates(objects: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Exact equal-weight minimizer B_c*(Q) for fixed subject actions."""

    values = _validate_objects(objects)
    q = _validate_actions(actions, len(values), values.shape[-1])
    aligned = np.stack(
        [conjugate(q[index].T, values[index]) for index in range(len(values))]
    )
    return symmetrize(np.mean(aligned, axis=0, dtype=np.float64))


def profiled_objective(objects: np.ndarray, actions: np.ndarray) -> float:
    """Formulation B after exact analytic elimination of all templates."""

    values = _validate_objects(objects)
    q = _validate_actions(actions, len(values), values.shape[-1])
    b_star = profiled_templates(values, q)
    return latent_least_squares_objective(values, q, b_star)


def profiled_pairwise_identity(objects: np.ndarray, actions: np.ndarray) -> float:
    """Equivalent 1/(2R) sum_{r,t,c} ||V_rc - V_tc||_F^2 form."""

    values = _validate_objects(objects)
    q = _validate_actions(actions, len(values), values.shape[-1])
    aligned = np.stack(
        [conjugate(q[index].T, values[index]) for index in range(len(values))]
    )
    differences = aligned[:, None] - aligned[None, :]
    return float(
        np.sum(differences * differences, dtype=np.float64) / (2.0 * len(values))
    )


def profiled_euclidean_gradient(objects: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Euclidean gradients of the exact profiled objective for every Q_r."""

    values = _validate_objects(objects)
    q = _validate_actions(actions, len(values), values.shape[-1])
    aligned = np.stack(
        [conjugate(q[index].T, values[index]) for index in range(len(values))]
    )
    centered = aligned - np.mean(aligned, axis=0, dtype=np.float64)
    gradients = np.empty_like(q)
    for subject in range(len(values)):
        gradients[subject] = 4.0 * np.einsum(
            "cij,jk,ckl->il",
            values[subject],
            q[subject],
            centered[subject],
            optimize=True,
        )
    return gradients


def _initial_actions_from_anchor(
    objects: np.ndarray,
    anchor_index: int,
    *,
    seed: int,
    start_count: int,
) -> np.ndarray:
    values = _validate_objects(objects)
    actions = np.tile(np.eye(values.shape[-1]), (len(values), 1, 1))
    for subject in range(len(values)):
        if subject == anchor_index:
            continue
        candidates = deterministic_starts(
            values[subject],
            values[anchor_index],
            seed=seed + 1009 * subject,
            count=start_count,
        )
        actions[subject] = min(
            candidates,
            key=lambda candidate: action_objective(
                values[subject], values[anchor_index], candidate
            ),
        )
    actions[anchor_index] = np.eye(values.shape[-1])
    return actions


def fit_profiled_product_reference(
    objects: np.ndarray,
    *,
    anchor_index: int = 0,
    seed: int = 20260810,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
    initial_actions: np.ndarray | None = None,
) -> ReferenceFit:
    """One product-manifold solve of Formulation B for synthetic audit use.

    The global gauge is fixed by holding one action at identity. There is one
    Pymanopt invocation over O(d)^(R-1), rather than nested single-Q solves.
    """

    values = _validate_objects(objects)
    if not 0 <= int(anchor_index) < len(values):
        raise ValueError("anchor index is out of range")
    d = values.shape[-1]
    initial = (
        _initial_actions_from_anchor(
            values,
            int(anchor_index),
            seed=seed,
            start_count=settings.starts,
        )
        if initial_actions is None
        else _validate_actions(initial_actions, len(values), d).copy()
    )
    if not np.allclose(initial[int(anchor_index)], np.eye(d), rtol=0.0, atol=1.0e-12):
        raise ValueError("the gauge-anchor action must be identity")
    free = tuple(index for index in range(len(values)) if index != int(anchor_index))
    manifold = Product([Stiefel(d, d, retraction="polar") for _ in free])

    def assemble(parts: Sequence[np.ndarray]) -> np.ndarray:
        actions = np.tile(np.eye(d), (len(values), 1, 1))
        for index, action in zip(free, parts):
            actions[index] = action
        return actions

    @pymanopt.function.numpy(manifold)
    def cost(*parts: np.ndarray) -> float:
        return profiled_objective(values, assemble(parts))

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*parts: np.ndarray) -> list[np.ndarray]:
        gradients = profiled_euclidean_gradient(values, assemble(parts))
        return [gradients[index] for index in free]

    problem = Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
    optimizer = build_pymanopt_optimizer(settings)
    started = time.perf_counter()
    result = optimizer.run(problem, initial_point=[initial[index] for index in free])
    elapsed = time.perf_counter() - started
    actions = assemble(result.point)
    templates = profiled_templates(values, actions)
    euclidean = [profiled_euclidean_gradient(values, actions)[index] for index in free]
    riemannian = manifold.projection(result.point, euclidean)
    gradient_norm = float(manifold.norm(result.point, riemannian))
    return ReferenceFit(
        actions=actions,
        templates=templates,
        objective=profiled_objective(values, actions),
        gradient_norm=gradient_norm,
        iterations=int(result.iterations),
        converged=bool(gradient_norm <= settings.gradient_tolerance),
        elapsed_seconds=float(elapsed),
        optimizer="one_pymanopt_product_Od_power_R_minus_1",
    )


def fit_joint_latent_reference(
    objects: np.ndarray,
    *,
    anchor_index: int = 0,
    seed: int = 20260810,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
    initial_actions: np.ndarray | None = None,
) -> ReferenceFit:
    """Direct small-problem Formulation-A reference without nested solvers.

    This jointly optimizes R-1 orthogonal actions and the Euclidean template
    bank in a single Pymanopt call. It exists only to compare A and B on small
    synthetic fixtures. Symmetric initialization and gradients keep B in Sym(d).
    """

    values = _validate_objects(objects)
    if not 0 <= int(anchor_index) < len(values):
        raise ValueError("anchor index is out of range")
    d = values.shape[-1]
    initial = (
        _initial_actions_from_anchor(
            values,
            int(anchor_index),
            seed=seed,
            start_count=settings.starts,
        )
        if initial_actions is None
        else _validate_actions(initial_actions, len(values), d).copy()
    )
    if not np.allclose(initial[int(anchor_index)], np.eye(d), rtol=0.0, atol=1.0e-12):
        raise ValueError("the gauge-anchor action must be identity")
    free = tuple(index for index in range(len(values)) if index != int(anchor_index))
    manifold = Product(
        [Stiefel(d, d, retraction="polar") for _ in free]
        + [Euclidean(values.shape[1], d, d)]
    )

    def assemble(parts: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        actions = np.tile(np.eye(d), (len(values), 1, 1))
        for index, action in zip(free, parts[:-1]):
            actions[index] = action
        return actions, np.asarray(parts[-1], dtype=np.float64)

    @pymanopt.function.numpy(manifold)
    def cost(*parts: np.ndarray) -> float:
        actions, templates = assemble(parts)
        return latent_least_squares_objective(values, actions, templates)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*parts: np.ndarray) -> list[np.ndarray]:
        actions, templates = assemble(parts)
        action_gradients = [
            action_gradient(values[index], templates, actions[index])
            for index in free
        ]
        aligned = np.stack(
            [conjugate(actions[index].T, values[index]) for index in range(len(values))]
        )
        template_gradient = 2.0 * (
            len(values) * templates - np.sum(aligned, axis=0, dtype=np.float64)
        )
        return [*action_gradients, template_gradient]

    initial_templates = profiled_templates(values, initial)
    initial_point = [initial[index] for index in free] + [initial_templates]
    problem = Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
    optimizer = build_pymanopt_optimizer(settings)
    started = time.perf_counter()
    result = optimizer.run(problem, initial_point=initial_point)
    elapsed = time.perf_counter() - started
    actions, templates = assemble(result.point)
    gradients = euclidean_gradient(*result.point)
    riemannian = manifold.projection(result.point, gradients)
    gradient_norm = float(manifold.norm(result.point, riemannian))
    return ReferenceFit(
        actions=actions,
        templates=symmetrize(templates),
        objective=latent_least_squares_objective(values, actions, templates),
        gradient_norm=gradient_norm,
        iterations=int(result.iterations),
        converged=bool(gradient_norm <= settings.gradient_tolerance),
        elapsed_seconds=float(elapsed),
        optimizer="one_pymanopt_joint_product_reference_A",
    )


def fit_pairwise_action(
    target_fit_matrices: np.ndarray,
    source_fit_matrices: np.ndarray,
    *,
    seed: int,
    settings: SolverSettings = CANDIDATE_SOLVER_SETTINGS,
) -> ActionFit:
    """Formulation C single-action fit with an exact total-start contract."""

    return optimize_action(
        target_fit_matrices,
        source_fit_matrices,
        seed=seed,
        settings=settings,
    )


def equivalence_aware_cycle_diagnostic(
    direct_actions: Sequence[np.ndarray],
    first_actions: Sequence[np.ndarray],
    second_actions: Sequence[np.ndarray],
    probe_matrices: np.ndarray,
) -> CycleDiagnostic:
    """Compare R_s<-t with R_s<-r R_r<-t by induced conjugation.

    Candidate representatives are searched explicitly, so raw matrices are
    never compared. This finite-set diagnostic does not prove that every
    continuous or discrete equivalence representative has been enumerated.
    """

    probes = np.asarray(probe_matrices, dtype=np.float64)
    if probes.ndim != 3 or probes.shape[-1] != probes.shape[-2]:
        raise ValueError("probe matrices must have shape (object,d,d)")
    banks = [tuple(np.asarray(value, dtype=np.float64) for value in group) for group in (direct_actions, first_actions, second_actions)]
    if any(not group for group in banks):
        raise ValueError("every action-equivalence set must be nonempty")
    best: tuple[float, float, int, int, int] | None = None
    for direct_index, first_index, second_index in itertools.product(
        range(len(banks[0])), range(len(banks[1])), range(len(banks[2]))
    ):
        direct = conjugate(banks[0][direct_index], probes)
        composed_action = banks[1][first_index] @ banks[2][second_index]
        composed = conjugate(composed_action, probes)
        normalization = max(
            float(np.linalg.norm(direct)),
            float(np.finfo(np.float64).eps * np.sqrt(probes.size)),
        )
        relative = float(np.linalg.norm(composed - direct) / normalization)
        candidate = (relative, normalization, direct_index, first_index, second_index)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return CycleDiagnostic(
        relative_discrepancy=best[0],
        normalization=best[1],
        direct_index=best[2],
        first_index=best[3],
        second_index=best[4],
    )


def bnci_stage_a_call_budget(total_starts: int) -> dict[str, int]:
    """Deterministic call-count audit; no data are read."""

    starts = int(total_starts)
    if starts < 1:
        raise ValueError("total_starts must be positive")
    subjects, sessions, heldouts, sources = 9, 2, 4, 8
    cells = subjects * sessions * heldouts
    source_fits_full_and_halves = cells * 3
    target_true_fits_full_and_halves = cells * 3
    target_semantic_full_fits = cells * 5
    target_fit_objects = target_true_fits_full_and_halves + target_semantic_full_fits
    pairwise_full_fit_objects = cells * sources
    pairwise_full_and_half_fit_objects = pairwise_full_fit_objects * 3
    pairwise_semantic_full_fit_objects = pairwise_full_fit_objects * 5
    return {
        "stage_a_cells": cells,
        "old_actual_pymanopt_runs_per_source_fit": 4 * 120 * 7 * 9,
        "old_actual_source_pymanopt_runs_full_and_halves": source_fits_full_and_halves * 4 * 120 * 7 * 9,
        "profiled_product_fit_objects_full_and_halves": source_fits_full_and_halves,
        "profiled_product_optimizer_runs_full_and_halves": source_fits_full_and_halves * starts,
        "target_single_action_fit_objects": target_fit_objects,
        "target_single_action_pymanopt_runs": target_fit_objects * starts,
        "pairwise_core_fit_objects_full": pairwise_full_fit_objects,
        "pairwise_core_pymanopt_runs_full": pairwise_full_fit_objects * starts,
        "pairwise_core_fit_objects_full_and_halves": pairwise_full_and_half_fit_objects,
        "pairwise_core_pymanopt_runs_full_and_halves": pairwise_full_and_half_fit_objects * starts,
        "pairwise_semantic_full_fit_objects": pairwise_semantic_full_fit_objects,
        "pairwise_total_pymanopt_runs_with_halves_and_semantic": (
            pairwise_full_and_half_fit_objects + pairwise_semantic_full_fit_objects
        )
        * starts,
    }


__all__ = [
    "CycleDiagnostic",
    "ReferenceFit",
    "bnci_stage_a_call_budget",
    "equivalence_aware_cycle_diagnostic",
    "fit_joint_latent_reference",
    "fit_pairwise_action",
    "fit_profiled_product_reference",
    "latent_least_squares_objective",
    "profiled_euclidean_gradient",
    "profiled_objective",
    "profiled_pairwise_identity",
    "profiled_templates",
]
