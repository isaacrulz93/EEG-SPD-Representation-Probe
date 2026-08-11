"""Frozen ordered discrete AIRM anti-development analysis for BNCI2014_001.

This module operates on saved cell-level mean covariance sequences.  It does
not fit means, average trial-level velocities, permute temporal steps, or
interpret the fitted orthogonal nuisance action.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions
from sklearn.decomposition import PCA

from src.geometry_v2 import airm_distance, spd_invsqrt, spd_log, symmetrize
from src.local_temporal_sequence_v0 import (
    CLASS_ORDER,
    DEFAULT_MASTER_SEED,
    DEFAULT_NULL_REPLICATES,
    N_CELLS,
    N_CLASSES,
    N_SUBJECTS,
    classbreak_mappings,
    subjectbreak_mappings,
)
from src.spd_utils import svec


N_STATES = 5
N_STEPS = 4
N_CHANNELS = 22
N_SESSIONS = 2
N_HALVES = 2
DELTA_T_SECONDS = 0.8

SYMMETRY_RELATIVE_TOLERANCE = 1.0e-10
NORM_ABSOLUTE_TOLERANCE = 1.0e-8
TRANSPORT_RELATIVE_TOLERANCE = 1.0e-8
CONGRUENCE_RELATIVE_TOLERANCE = 1.0e-8
GRADIENT_ABSOLUTE_TOLERANCE = 2.0e-6
GRADIENT_RELATIVE_TOLERANCE = 2.0e-5
QUOTIENT_ZERO_TOLERANCE = 1.0e-8
MULTISTART_OBJECTIVE_TOLERANCE = 1.0e-8
TRUST_HESSIAN_RADIUS = 1.0e-5
TRUST_MAX_INNER_ITERATIONS = 30


class MovementGeometryNumericalError(RuntimeError):
    """A logarithm, path transport, or anti-development gate failed."""


class MovementQuotientNumericalError(RuntimeError):
    """A required common-O(d) movement optimization gate failed."""


@dataclass(frozen=True)
class MovementOptimizerSettings:
    total_starts: int = 6
    max_iterations: int = 250
    gradient_tolerance: float = 1.0e-6
    min_step_size: float = 1.0e-12
    max_time_seconds: float = 120.0
    max_cost_evaluations: int = 5000


FROZEN_OPTIMIZER_SETTINGS = MovementOptimizerSettings()


@dataclass(frozen=True)
class AntiDevelopment:
    z: np.ndarray
    local_u: np.ndarray
    displacements: np.ndarray
    transported: np.ndarray
    speeds: np.ndarray
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class AlignmentStart:
    start_index: int
    start_kind: str
    initial_determinant: int
    final_determinant: int
    action: np.ndarray
    objective: float
    gradient_norm: float
    iterations: int
    converged: bool
    stopping_criterion: str


@dataclass(frozen=True)
class MovementAlignment:
    action: np.ndarray
    objective: float
    distance: float
    determinant: int
    gradient_norm: float
    best_start_index: int
    second_best_objective: float
    objective_spread: float
    starts: tuple[AlignmentStart, ...]


@dataclass(frozen=True)
class MovementContrasts:
    a_sc: np.ndarray
    b_sc: np.ndarray
    c_sc: np.ndarray
    d_sc: np.ndarray
    s_sc: np.ndarray
    c_specific_sc: np.ndarray
    j_sc: np.ndarray
    s_s: np.ndarray
    c_s: np.ndarray
    j_s: np.ndarray
    t_subject: float
    t_class: float
    t_j: float


@dataclass(frozen=True)
class MovementInference:
    observed: MovementContrasts
    subjectbreak_t_subject: np.ndarray
    subjectbreak_t_j: np.ndarray
    classbreak_t_class: np.ndarray
    classbreak_t_j: np.ndarray
    subject_mappings: np.ndarray
    class_mappings: np.ndarray
    p_subject: float
    p_class: float
    p_j_subjectbreak: float
    p_j_classbreak: float


@dataclass(frozen=True)
class MovementPCA:
    coordinates: np.ndarray
    components: np.ndarray
    feature_mean: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray


def _symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    values = symmetrize(np.asarray(matrix, dtype=np.float64))
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError("SPD square root received a nonpositive eigenvalue")
    return symmetrize((eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T)


def _validate_spd_sequence(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != N_STATES or array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must have shape (5,d,d)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    relative = np.linalg.norm(array - array.transpose(0, 2, 1), axis=(1, 2)) / np.maximum(
        np.linalg.norm(array, axis=(1, 2)), np.finfo(float).tiny
    )
    if float(np.max(relative)) > SYMMETRY_RELATIVE_TOLERANCE:
        raise ValueError(f"{name} is not symmetric")
    array = symmetrize(array)
    if float(np.min(np.linalg.eigvalsh(array))) <= 0.0:
        raise ValueError(f"{name} must be SPD")
    return array


def _validate_movement(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != N_STEPS or array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must have shape (4,d,d)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    relative = np.linalg.norm(array - array.transpose(0, 2, 1), axis=(1, 2)) / np.maximum(
        np.linalg.norm(array, axis=(1, 2)), np.finfo(float).tiny
    )
    if float(np.max(relative)) > SYMMETRY_RELATIVE_TOLERANCE:
        raise ValueError(f"{name} is not symmetric")
    return symmetrize(array)


def airm_log_map(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Evaluate the exact affine-invariant logarithmic map Log_base(target)."""

    p = np.asarray(base, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)
    p_sqrt = _symmetric_sqrt(p)
    p_invsqrt = spd_invsqrt(p)
    relative = symmetrize(p_invsqrt @ q @ p_invsqrt)
    return symmetrize(p_sqrt @ spd_log(relative) @ p_sqrt)


def local_whitened_displacement(
    base: np.ndarray, target: np.ndarray, *, delta_t: float = DELTA_T_SECONDS
) -> np.ndarray:
    whitening = spd_invsqrt(np.asarray(base, dtype=np.float64))
    relative = symmetrize(whitening @ np.asarray(target, dtype=np.float64) @ whitening)
    return symmetrize(spd_log(relative) / delta_t)


def airm_tangent_inner(base: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    whitening = spd_invsqrt(np.asarray(base, dtype=np.float64))
    left = symmetrize(whitening @ np.asarray(first, dtype=np.float64) @ whitening)
    right = symmetrize(whitening @ np.asarray(second, dtype=np.float64) @ whitening)
    return float(np.sum(left * right))


def airm_tangent_norm(base: np.ndarray, tangent: np.ndarray) -> float:
    return float(np.sqrt(max(airm_tangent_inner(base, tangent, tangent), 0.0)))


def parallel_transport_operator(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Standard AIRM geodesic transport operator E=(target base^-1)^1/2."""

    p = np.asarray(base, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)
    p_sqrt = _symmetric_sqrt(p)
    p_invsqrt = spd_invsqrt(p)
    relative = symmetrize(p_invsqrt @ q @ p_invsqrt)
    return p_sqrt @ _symmetric_sqrt(relative) @ p_invsqrt


def parallel_transport(base: np.ndarray, target: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    operator = parallel_transport_operator(base, target)
    return symmetrize(operator @ np.asarray(tangent, dtype=np.float64) @ operator.T)


def anti_develop_sequence(
    sequence: np.ndarray, *, delta_t: float = DELTA_T_SECONDS
) -> AntiDevelopment:
    """Transport each adjacent displacement along its ordered reverse prefix."""

    means = _validate_spd_sequence(sequence, name="mean sequence")
    if not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t must be finite and positive")
    shape = (N_STEPS, means.shape[-1], means.shape[-1])
    displacements = np.empty(shape, dtype=np.float64)
    local_u = np.empty(shape, dtype=np.float64)
    transported = np.empty(shape, dtype=np.float64)
    z = np.empty(shape, dtype=np.float64)
    m1_whitening = spd_invsqrt(means[0])
    rows: list[dict[str, float | int | bool]] = []
    for step in range(N_STEPS):
        displacement = airm_log_map(means[step], means[step + 1]) / delta_t
        local = local_whitened_displacement(means[step], means[step + 1], delta_t=delta_t)
        value = displacement.copy()
        maximum_edge_error = 0.0
        for prefix in reversed(range(step)):
            before_norm = airm_tangent_norm(means[prefix + 1], value)
            value = parallel_transport(means[prefix + 1], means[prefix], value)
            after_norm = airm_tangent_norm(means[prefix], value)
            edge_error = abs(before_norm - after_norm) / max(
                before_norm, after_norm, np.finfo(float).tiny
            )
            maximum_edge_error = max(maximum_edge_error, float(edge_error))
        whitened = symmetrize(m1_whitening @ value @ m1_whitening)
        expected = float(airm_distance(means[step], means[step + 1])) / delta_t
        v_norm = airm_tangent_norm(means[step], displacement)
        u_norm = float(np.linalg.norm(local, ord="fro"))
        z_norm = float(np.linalg.norm(whitened, ord="fro"))
        symmetry_error = float(
            np.linalg.norm(whitened - whitened.T, ord="fro")
            / max(z_norm, np.finfo(float).tiny)
        )
        maximum_norm_error = max(abs(v_norm - expected), abs(u_norm - expected), abs(z_norm - expected))
        passed = bool(
            symmetry_error <= SYMMETRY_RELATIVE_TOLERANCE
            and maximum_norm_error <= NORM_ABSOLUTE_TOLERANCE
            and maximum_edge_error <= TRANSPORT_RELATIVE_TOLERANCE
        )
        rows.append(
            {
                "transition": step + 1,
                "v_airm_norm": v_norm,
                "u_frobenius_norm": u_norm,
                "z_frobenius_norm": z_norm,
                "expected_airm_speed": expected,
                "maximum_norm_absolute_error": maximum_norm_error,
                "maximum_edge_transport_relative_error": maximum_edge_error,
                "z_symmetry_relative_error": symmetry_error,
                "passed": passed,
            }
        )
        displacements[step] = displacement
        local_u[step] = local
        transported[step] = value
        z[step] = whitened
    diagnostics = pd.DataFrame(rows)
    if not bool(diagnostics["passed"].all()) or not np.isfinite(z).all():
        raise MovementGeometryNumericalError(
            "UNASSESSED_MOVEMENT_GEOMETRY_NUMERICAL_FAILURE: anti-development gate failed"
        )
    return AntiDevelopment(
        z=z,
        local_u=local_u,
        displacements=displacements,
        transported=transported,
        speeds=np.linalg.norm(z, axis=(1, 2)),
        diagnostics=diagnostics,
    )


def conjugate_movement(movement: np.ndarray, action: np.ndarray) -> np.ndarray:
    values = _validate_movement(movement, name="movement")
    q = np.asarray(action, dtype=np.float64)
    if q.shape != values.shape[1:]:
        raise ValueError("orthogonal action has the wrong shape")
    return symmetrize(np.einsum("ij,kjl,ml->kim", q, values, q, optimize=True))


def movement_objective(target: np.ndarray, source: np.ndarray, action: np.ndarray) -> float:
    a = _validate_movement(target, name="target")
    b = _validate_movement(source, name="source")
    transformed = conjugate_movement(b, action)
    residual = a - transformed
    return float(np.mean(np.sum(residual * residual, axis=(1, 2))))


def movement_euclidean_gradient(
    target: np.ndarray, source: np.ndarray, action: np.ndarray
) -> np.ndarray:
    """Exact ambient Euclidean gradient of the Frobenius conjugation loss."""

    a = np.asarray(target, dtype=np.float64)
    b = np.asarray(source, dtype=np.float64)
    q = np.asarray(action, dtype=np.float64)
    gradient = np.zeros_like(q)
    for target_step, source_step in zip(a, b, strict=True):
        residual = symmetrize(q @ source_step @ q.T) - target_step
        gradient += (4.0 / N_STEPS) * residual @ q @ source_step
    return gradient


def movement_euclidean_hessian(
    target: np.ndarray,
    source: np.ndarray,
    action: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    """Exact directional derivative of the ambient Euclidean gradient."""

    a = np.asarray(target, dtype=np.float64)
    b = np.asarray(source, dtype=np.float64)
    q = np.asarray(action, dtype=np.float64)
    eta = np.asarray(tangent, dtype=np.float64)
    hessian = np.zeros_like(q)
    for target_step, source_step in zip(a, b, strict=True):
        residual = symmetrize(q @ source_step @ q.T) - target_step
        derivative = eta @ source_step @ q.T + q @ source_step @ eta.T
        hessian += (4.0 / N_STEPS) * (
            derivative @ q @ source_step + residual @ eta @ source_step
        )
    return hessian


def _canonical_eigenvectors(matrix: np.ndarray) -> np.ndarray:
    _, vectors = np.linalg.eigh(symmetrize(matrix))
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0.0:
            vectors[:, column] *= -1.0
    return vectors


def _spectral_start(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    weights = np.sqrt(np.arange(1, N_STEPS + 1, dtype=np.float64))
    reverse = np.sqrt(np.arange(N_STEPS, 0, -1, dtype=np.float64))
    va = _canonical_eigenvectors(np.einsum("k,kij->ij", weights, target))
    vb = _canonical_eigenvectors(np.einsum("k,kij->ij", weights, source))
    secondary_a = va.T @ np.einsum("k,kij->ij", reverse, target) @ va
    secondary_b = vb.T @ np.einsum("k,kij->ij", reverse, source) @ vb
    strength = np.abs(secondary_a * secondary_b)
    np.fill_diagonal(strength, 0.0)
    pivot = int(np.argmax(np.sum(strength, axis=1)))
    signs = np.ones(target.shape[-1], dtype=np.float64)
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


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _deterministic_orthogonal(seed: int, dimension: int) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=(dimension, dimension))
    q, r = np.linalg.qr(raw)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q @ np.diag(signs)


def _spectral_perturbation(action: np.ndarray, seed: int) -> np.ndarray:
    dimension = action.shape[0]
    raw = np.random.default_rng(seed).normal(size=(dimension, dimension))
    skew = 0.5 * (raw - raw.T)
    skew /= max(np.linalg.norm(skew, ord="fro"), np.finfo(float).tiny)
    candidate = action @ (np.eye(dimension) + 0.25 * skew)
    u, _, vt = np.linalg.svd(candidate, full_matrices=False)
    perturbed = u @ vt
    if int(np.sign(np.linalg.det(perturbed))) != int(np.sign(np.linalg.det(action))):
        raise MovementQuotientNumericalError("spectral perturbation changed determinant component")
    return perturbed


def _alignment_initializations(
    target: np.ndarray, source: np.ndarray
) -> tuple[tuple[str, np.ndarray], ...]:
    dimension = target.shape[-1]
    reflection = np.eye(dimension)
    reflection[0, 0] = -1.0
    spectral = _spectral_start(target, source)
    seed_material = f"local-mean-movement-v0|{_array_sha256(target)}|{_array_sha256(source)}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little")
    random = _deterministic_orthogonal(seed, dimension)
    perturbed = _spectral_perturbation(spectral, seed ^ 0x5A17)
    starts = (
        ("spectral", spectral),
        ("spectral_reflection", spectral @ reflection),
        ("spectral_perturbed", perturbed),
        ("spectral_perturbed_reflection", perturbed @ reflection),
        ("deterministic_haar", random),
        ("deterministic_haar_reflection", random @ reflection),
    )
    determinants = [int(np.sign(np.linalg.det(value))) for _, value in starts]
    if determinants.count(-1) != 3 or determinants.count(1) != 3:
        raise MovementQuotientNumericalError("six starts do not cover O(d) sectors 3+3")
    return starts


def _movement_problem(target: np.ndarray, source: np.ndarray) -> tuple[Stiefel, Problem]:
    a = _validate_movement(target, name="target")
    b = _validate_movement(source, name="source")
    if a.shape != b.shape:
        raise ValueError("target and source movements have different shapes")
    manifold = Stiefel(a.shape[-1], a.shape[-1], retraction="polar")

    @pymanopt.function.numpy(manifold)
    def cost(action: np.ndarray) -> float:
        return movement_objective(a, b, action)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(action: np.ndarray) -> np.ndarray:
        return movement_euclidean_gradient(a, b, action)

    @pymanopt.function.numpy(manifold)
    def euclidean_hessian(action: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        return movement_euclidean_hessian(a, b, action, tangent)

    return manifold, Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )


def _movement_optimizer(settings: MovementOptimizerSettings) -> TrustRegions:
    return TrustRegions(
        miniter=3,
        kappa=0.1,
        theta=1.0,
        rho_prime=0.1,
        use_rand=False,
        rho_regularization=1000.0,
        max_time=settings.max_time_seconds,
        max_iterations=settings.max_iterations,
        min_gradient_norm=settings.gradient_tolerance,
        min_step_size=settings.min_step_size,
        max_cost_evaluations=settings.max_cost_evaluations,
        verbosity=0,
        log_verbosity=0,
    )


def optimize_movement_alignment(
    target: np.ndarray,
    source: np.ndarray,
    *,
    settings: MovementOptimizerSettings = FROZEN_OPTIMIZER_SETTINGS,
) -> MovementAlignment:
    """Run the frozen four-start, two-component TrustRegions search."""

    a = _validate_movement(target, name="target")
    b = _validate_movement(source, name="source")
    if a.shape != b.shape:
        raise ValueError("target and source movement shapes differ")
    if settings.total_starts != 6:
        raise ValueError("V0 requires exactly six total starts")
    manifold, problem = _movement_problem(a, b)
    optimizer = _movement_optimizer(settings)
    rows: list[AlignmentStart] = []
    for start_index, (kind, initial) in enumerate(_alignment_initializations(a, b)):
        result = optimizer.run(
            problem,
            initial_point=initial,
            mininner=1,
            maxinner=TRUST_MAX_INNER_ITERATIONS,
        )
        action = np.asarray(result.point, dtype=np.float64)
        initial_determinant = int(np.sign(np.linalg.det(initial)))
        final_determinant = int(np.sign(np.linalg.det(action)))
        if final_determinant != initial_determinant:
            raise MovementQuotientNumericalError("TrustRegions crossed O(d) determinant components")
        projected = manifold.projection(action, movement_euclidean_gradient(a, b, action))
        gradient_norm = float(manifold.norm(action, projected))
        objective = movement_objective(a, b, action)
        converged = bool(
            gradient_norm <= settings.gradient_tolerance
            or objective <= QUOTIENT_ZERO_TOLERANCE**2
        )
        rows.append(
            AlignmentStart(
                start_index=start_index,
                start_kind=kind,
                initial_determinant=initial_determinant,
                final_determinant=final_determinant,
                action=action,
                objective=objective,
                gradient_norm=gradient_norm,
                iterations=int(result.iterations),
                converged=converged,
                stopping_criterion=str(result.stopping_criterion),
            )
        )
    converged_rows = [value for value in rows if value.converged]
    if {value.final_determinant for value in converged_rows} != {-1, 1}:
        raise MovementQuotientNumericalError(
            "UNASSESSED_MOVEMENT_QUOTIENT_NUMERICAL_FAILURE: both determinant sectors lack certified candidates"
        )
    ordered = sorted(
        converged_rows,
        key=lambda value: (value.objective, value.gradient_norm, value.start_index),
    )
    best = ordered[0]
    objectives = np.asarray([value.objective for value in converged_rows])
    return MovementAlignment(
        action=best.action,
        objective=best.objective,
        distance=float(np.sqrt(max(best.objective, 0.0))),
        determinant=best.final_determinant,
        gradient_norm=best.gradient_norm,
        best_start_index=best.start_index,
        second_best_objective=float(ordered[1].objective),
        objective_spread=float(np.max(objectives) - np.min(objectives)),
        starts=tuple(rows),
    )


def movement_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    settings: MovementOptimizerSettings = FROZEN_OPTIMIZER_SETTINGS,
) -> tuple[float, MovementAlignment]:
    """Symmetric quotient distance via deterministic canonical pair ordering."""

    left = _validate_movement(first, name="first")
    right = _validate_movement(second, name="second")
    left_hash = _array_sha256(left)
    right_hash = _array_sha256(right)
    target, source = (left, right) if left_hash <= right_hash else (right, left)
    fit = optimize_movement_alignment(target, source, settings=settings)
    return fit.distance, fit


def direct_distance(first: np.ndarray, second: np.ndarray) -> float:
    residual = _validate_movement(first, name="first") - _validate_movement(second, name="second")
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=(1, 2)))))


def length_profile_distance(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (N_STEPS,) or right.shape != (N_STEPS,):
        raise ValueError("speed profiles must have shape (4,)")
    return float(np.sqrt(np.mean((left - right) ** 2)))


def cell_index(subject_index: int, class_index: int) -> int:
    return subject_index * N_CLASSES + class_index


def movement_contrasts(matrix: np.ndarray) -> MovementContrasts:
    """Compute the frozen distance-level subject, class, and interaction effects."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (N_CELLS, N_CELLS) or not np.isfinite(values).all():
        raise ValueError("discrepancy matrix must be finite with shape (36,36)")
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
    s_sc = c - a
    c_specific_sc = b - a
    j_sc = b + c - a - d
    s_s = np.mean(s_sc, axis=1)
    c_s = np.mean(c_specific_sc, axis=1)
    j_s = np.mean(j_sc, axis=1)
    return MovementContrasts(
        a_sc=a,
        b_sc=b,
        c_sc=c,
        d_sc=d,
        s_sc=s_sc,
        c_specific_sc=c_specific_sc,
        j_sc=j_sc,
        s_s=s_s,
        c_s=c_s,
        j_s=j_s,
        t_subject=float(np.mean(s_s)),
        t_class=float(np.mean(c_s)),
        t_j=float(np.mean(j_s)),
    )


def plus_one_pvalue(observed: float, null: np.ndarray) -> float:
    values = np.asarray(null, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("null distribution must be a finite nonempty vector")
    return float((1 + np.count_nonzero(values >= observed)) / (1 + len(values)))


def evaluate_movement_inference(
    matrix: np.ndarray,
    *,
    replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> MovementInference:
    """Relabel the frozen 36x36 matrix under inherited whole-cell mappings."""

    values = np.asarray(matrix, dtype=np.float64)
    observed = movement_contrasts(values)
    subject_maps = subjectbreak_mappings(replicates=replicates, master_seed=master_seed)
    class_maps = classbreak_mappings(replicates=replicates, master_seed=master_seed)
    subject_t = np.empty(replicates, dtype=np.float64)
    subject_j = np.empty(replicates, dtype=np.float64)
    class_t = np.empty(replicates, dtype=np.float64)
    class_j = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        subject_result = movement_contrasts(values[:, subject_maps[replicate]])
        subject_t[replicate] = subject_result.t_subject
        subject_j[replicate] = subject_result.t_j
        class_result = movement_contrasts(values[:, class_maps[replicate]])
        class_t[replicate] = class_result.t_class
        class_j[replicate] = class_result.t_j
    return MovementInference(
        observed=observed,
        subjectbreak_t_subject=subject_t,
        subjectbreak_t_j=subject_j,
        classbreak_t_class=class_t,
        classbreak_t_j=class_j,
        subject_mappings=subject_maps,
        class_mappings=class_maps,
        p_subject=plus_one_pvalue(observed.t_subject, subject_t),
        p_class=plus_one_pvalue(observed.t_class, class_t),
        p_j_subjectbreak=plus_one_pvalue(observed.t_j, subject_j),
        p_j_classbreak=plus_one_pvalue(observed.t_j, class_j),
    )


def terminal_decision(
    *,
    t_subject: float,
    p_subject: float,
    t_class: float,
    p_class: float,
    t_j: float,
    p_j_subjectbreak: float,
    p_j_classbreak: float,
    alpha: float = 0.05,
) -> str:
    subject_pass = t_subject > 0.0 and p_subject < alpha
    class_pass = t_class > 0.0 and p_class < alpha
    interaction_pass = t_j > 0.0 and p_j_subjectbreak < alpha and p_j_classbreak < alpha
    if subject_pass and class_pass and interaction_pass:
        return "GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT"
    if subject_pass and class_pass:
        return "GO_REPRODUCIBLE_ORDERED_MOVEMENT_WITHOUT_INTERACTION"
    return "STOP_NO_REPRODUCIBLE_ORDERED_MOVEMENT_V0"


def compute_movement_pca(movements: np.ndarray) -> MovementPCA:
    values = np.asarray(movements, dtype=np.float64)
    expected = (N_SESSIONS, N_SUBJECTS, N_CLASSES, N_STEPS, N_CHANNELS, N_CHANNELS)
    if values.shape != expected:
        raise ValueError(f"full movement bank must have shape {expected}")
    feature_count = N_CHANNELS * (N_CHANNELS + 1) // 2
    features = svec(values, check_symmetric=True).reshape(-1, feature_count)
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(features).reshape(
        N_SESSIONS, N_SUBJECTS, N_CLASSES, N_STEPS, 2
    )
    return MovementPCA(
        coordinates=coordinates,
        components=np.asarray(pca.components_, dtype=np.float64),
        feature_mean=np.asarray(pca.mean_, dtype=np.float64),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
    )


def optimizer_diagnostic_rows(
    fit: MovementAlignment,
    *,
    analysis: str,
    row_cell: int,
    column_cell: int,
) -> Iterable[dict[str, int | float | str | bool]]:
    for start in fit.starts:
        yield {
            "analysis": analysis,
            "row_cell": row_cell,
            "column_cell": column_cell,
            "start_index": start.start_index,
            "start_kind": start.start_kind,
            "initial_determinant": start.initial_determinant,
            "final_determinant": start.final_determinant,
            "objective": start.objective,
            "gradient_norm": start.gradient_norm,
            "iterations": start.iterations,
            "converged": start.converged,
            "stopping_criterion": start.stopping_criterion,
            "selected": start.start_index == fit.best_start_index,
        }


def run_synthetic_numerical_gates() -> dict[str, float | int | bool | str]:
    """Run all pre-data geometry and quotient checks at the scientific d=22."""

    rng = np.random.default_rng(20260812)
    raw = rng.normal(size=(N_STATES, N_CHANNELS, N_CHANNELS))
    sequence = np.asarray(
        [value @ value.T + N_CHANNELS * np.eye(N_CHANNELS) for value in raw]
    )
    developed = anti_develop_sequence(sequence)
    congruence = rng.normal(scale=0.15, size=(N_CHANNELS, N_CHANNELS))
    congruence += 1.5 * np.eye(N_CHANNELS)
    transformed_sequence = np.asarray(
        [congruence @ value @ congruence.T for value in sequence]
    )
    transformed = anti_develop_sequence(transformed_sequence)
    common_o = (
        spd_invsqrt(transformed_sequence[0])
        @ congruence
        @ _symmetric_sqrt(sequence[0])
    )
    orthogonality_error = float(
        np.linalg.norm(common_o @ common_o.T - np.eye(N_CHANNELS), ord="fro")
    )
    common_errors = np.asarray(
        [
            np.linalg.norm(
                transformed.z[step] - common_o @ developed.z[step] @ common_o.T,
                ord="fro",
            )
            / max(np.linalg.norm(transformed.z[step], ord="fro"), np.finfo(float).tiny)
            for step in range(N_STEPS)
        ]
    )

    source = rng.normal(scale=0.12, size=(N_STEPS, N_CHANNELS, N_CHANNELS))
    source = symmetrize(source)
    truth = _deterministic_orthogonal(20260813, N_CHANNELS)
    if int(np.sign(np.linalg.det(truth))) != -1:
        truth[:, 0] *= -1.0
    target = conjugate_movement(source, truth)
    known_fit = optimize_movement_alignment(target, source)
    truth_spectral = [
        value.objective
        for value in known_fit.starts
        if value.final_determinant == -1 and value.start_kind.startswith("spectral")
    ]
    action_recovery_error = float(
        min(
            np.linalg.norm(known_fit.action - truth, ord="fro"),
            np.linalg.norm(known_fit.action + truth, ord="fro"),
        )
        / np.sqrt(N_CHANNELS)
    )

    first = rng.normal(scale=0.10, size=(N_STEPS, N_CHANNELS, N_CHANNELS))
    second = rng.normal(scale=0.10, size=(N_STEPS, N_CHANNELS, N_CHANNELS))
    first = symmetrize(first)
    second = symmetrize(second)
    forward, forward_fit = movement_distance(first, second)
    reverse, reverse_fit = movement_distance(second, first)

    dimension = 6
    gradient_target = rng.normal(scale=0.10, size=(N_STEPS, dimension, dimension))
    gradient_source = rng.normal(scale=0.10, size=(N_STEPS, dimension, dimension))
    gradient_target = symmetrize(gradient_target)
    gradient_source = symmetrize(gradient_source)
    action = _deterministic_orthogonal(20260814, dimension)
    ambient = rng.normal(size=(dimension, dimension))
    tangent = action @ (0.5 * (action.T @ ambient - ambient.T @ action))
    tangent /= np.linalg.norm(tangent, ord="fro")
    manifold = Stiefel(dimension, dimension, retraction="polar")
    epsilon = 1.0e-6
    plus = manifold.retraction(action, epsilon * tangent)
    minus = manifold.retraction(action, -epsilon * tangent)
    finite = (
        movement_objective(gradient_target, gradient_source, plus)
        - movement_objective(gradient_target, gradient_source, minus)
    ) / (2.0 * epsilon)
    analytic = float(
        np.sum(
            movement_euclidean_gradient(gradient_target, gradient_source, action)
            * tangent
        )
    )
    gradient_absolute_error = float(abs(analytic - finite))
    gradient_relative_error = float(
        gradient_absolute_error / max(1.0, abs(analytic), abs(finite))
    )

    maximum_norm_error = float(
        max(
            developed.diagnostics["maximum_norm_absolute_error"].max(),
            transformed.diagnostics["maximum_norm_absolute_error"].max(),
        )
    )
    maximum_transport_error = float(
        max(
            developed.diagnostics["maximum_edge_transport_relative_error"].max(),
            transformed.diagnostics["maximum_edge_transport_relative_error"].max(),
        )
    )
    multistart_spread = float(max(truth_spectral) - min(truth_spectral))
    sectors = {value.final_determinant for value in known_fit.starts if value.converged}
    status = bool(
        maximum_norm_error <= NORM_ABSOLUTE_TOLERANCE
        and maximum_transport_error <= TRANSPORT_RELATIVE_TOLERANCE
        and orthogonality_error <= CONGRUENCE_RELATIVE_TOLERANCE
        and float(np.max(common_errors)) <= CONGRUENCE_RELATIVE_TOLERANCE
        and known_fit.distance <= QUOTIENT_ZERO_TOLERANCE
        and action_recovery_error <= QUOTIENT_ZERO_TOLERANCE
        and sectors == {-1, 1}
        and len(truth_spectral) == 2
        and multistart_spread <= MULTISTART_OBJECTIVE_TOLERANCE
        and forward == reverse
        and forward_fit.objective == reverse_fit.objective
        and gradient_absolute_error <= GRADIENT_ABSOLUTE_TOLERANCE
        and gradient_relative_error <= GRADIENT_RELATIVE_TOLERANCE
    )
    record: dict[str, float | int | bool | str] = {
        "status": "PASS" if status else "FAIL",
        "dimension": N_CHANNELS,
        "delta_t_seconds": DELTA_T_SECONDS,
        "maximum_norm_absolute_error": maximum_norm_error,
        "maximum_edge_transport_relative_error": maximum_transport_error,
        "common_o_orthogonality_frobenius_error": orthogonality_error,
        "common_o_maximum_transition_relative_error": float(np.max(common_errors)),
        "common_o_minimum_transition_relative_error": float(np.min(common_errors)),
        "known_q_distance": known_fit.distance,
        "known_q_action_recovery_relative_error": action_recovery_error,
        "known_q_converged_determinant_sector_count": len(sectors),
        "known_q_multistart_spectral_objective_spread": multistart_spread,
        "forward_distance": forward,
        "reverse_distance": reverse,
        "forward_reverse_exact": forward == reverse,
        "gradient_analytic_directional_derivative": analytic,
        "gradient_finite_difference": float(finite),
        "gradient_absolute_error": gradient_absolute_error,
        "gradient_relative_error": gradient_relative_error,
        "new_bnci_movement_statistic_computed": False,
    }
    if not status:
        raise MovementQuotientNumericalError(
            f"UNASSESSED_MOVEMENT_QUOTIENT_NUMERICAL_FAILURE: synthetic gate failed: {record}"
        )
    return record


__all__ = [
    "AntiDevelopment",
    "CLASS_ORDER",
    "CONGRUENCE_RELATIVE_TOLERANCE",
    "DELTA_T_SECONDS",
    "FROZEN_OPTIMIZER_SETTINGS",
    "GRADIENT_ABSOLUTE_TOLERANCE",
    "GRADIENT_RELATIVE_TOLERANCE",
    "MULTISTART_OBJECTIVE_TOLERANCE",
    "MovementAlignment",
    "MovementContrasts",
    "MovementGeometryNumericalError",
    "MovementInference",
    "MovementOptimizerSettings",
    "MovementPCA",
    "MovementQuotientNumericalError",
    "NORM_ABSOLUTE_TOLERANCE",
    "N_CHANNELS",
    "N_HALVES",
    "N_SESSIONS",
    "N_STATES",
    "N_STEPS",
    "QUOTIENT_ZERO_TOLERANCE",
    "SYMMETRY_RELATIVE_TOLERANCE",
    "TRANSPORT_RELATIVE_TOLERANCE",
    "TRUST_HESSIAN_RADIUS",
    "TRUST_MAX_INNER_ITERATIONS",
    "airm_log_map",
    "airm_tangent_inner",
    "airm_tangent_norm",
    "anti_develop_sequence",
    "compute_movement_pca",
    "conjugate_movement",
    "direct_distance",
    "evaluate_movement_inference",
    "length_profile_distance",
    "local_whitened_displacement",
    "movement_contrasts",
    "movement_distance",
    "movement_euclidean_gradient",
    "movement_euclidean_hessian",
    "movement_objective",
    "optimize_movement_alignment",
    "optimizer_diagnostic_rows",
    "parallel_transport",
    "parallel_transport_operator",
    "plus_one_pvalue",
    "run_synthetic_numerical_gates",
    "terminal_decision",
]
