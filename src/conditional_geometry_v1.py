"""Frozen SPD geometry core for Conditional-Geometry Anatomy v1.

This module is deliberately limited to in-memory covariance arrays.  It has no
dataset, session, file-system, or target-session API.  Observed AIRM means use
the public pyRiemann solver exactly as preregistered.  A separate NumPy-batched
implementation mirrors that solver for null-test acceleration and exposes an
explicit scalar-public-API cross-check.

No eigenvalue clipping, diagonal loading, row deletion, or available-case
fallback is performed here.  Failed numerical checks remain visible in audit
objects, and invalid SPD inputs fail immediately.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pyriemann.geometry.mean import mean_logeuclid, mean_riemann


__all__ = [
    "AIRM",
    "LE",
    "CLASS_ORDER",
    "D_UPPER_TRIANGLE_INDICES",
    "GeometryThresholds",
    "SPDAudit",
    "MeanAudit",
    "BatchedAIRMMeanResult",
    "ShapeAudit",
    "GeometryObjects",
    "ConditionalGeometryError",
    "NumericalGateError",
    "DegenerateClassGeometryError",
    "symmetrize",
    "spd_log",
    "symmetric_exp",
    "spd_sqrt",
    "spd_invsqrt",
    "relative_frobenius_error",
    "validate_spd_stack",
    "karcher_residual",
    "airm_mean_official",
    "airm_mean_batched",
    "le_mean_official",
    "airm_distance",
    "airm_distance_matrix",
    "le_distance_matrix",
    "airm_gram_matrices",
    "le_gram_matrix",
    "center_airm",
    "center_le",
    "svec",
    "smat",
    "shape_from_D",
    "shape_from_G",
    "require_non_degenerate",
    "all_class_permutations",
    "permutation_matrix",
    "permute_object_matrix",
    "permute_shape_vector",
    "frozen_orthogonal_gauge",
    "compute_airm_objects",
    "compute_le_objects",
    "compute_geometry_objects",
]


AIRM = "AIRM"
LE = "LE"
CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
D_UPPER_TRIANGLE_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


class ConditionalGeometryError(RuntimeError):
    """Base error for the frozen conditional-geometry computation."""


class NumericalGateError(ConditionalGeometryError):
    """Raised when an input cannot legally enter the frozen geometry."""


class DegenerateClassGeometryError(ConditionalGeometryError):
    """Raised when a normalized class-geometry shape is requested but absent."""


@dataclass(frozen=True)
class GeometryThresholds:
    """Numerical constants frozen in the companion v1 YAML."""

    mean_tol: float = 1.0e-9
    mean_maxiter: int = 100
    symmetry_relative_error_max: float = 1.0e-12
    condition_number_max: float = 1.0e12
    airm_karcher_residual_max: float = 1.0e-7
    d_centering_relative_error_max: float = 1.0e-10
    g_direct_whitened_relative_error_max: float = 1.0e-10
    orthogonal_d_relative_error_max: float = 1.0e-10
    orthogonal_g_relative_error_max: float = 1.0e-10
    permutation_equivariance_relative_error_max: float = 1.0e-10
    le_mean_relative_error_max: float = 1.0e-10
    le_d_g_identity_relative_error_max: float = 1.0e-10
    degeneracy_epsilon_multiplier: float = 100.0
    master_seed: int = 20260809
    orthogonal_gauge_family_tag: int = 1501

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GeometryThresholds":
        """Construct thresholds from the frozen full YAML mapping.

        Missing keys are errors rather than invitations to substitute defaults.
        This keeps a producer tied to the preregistered config while leaving the
        low-level array APIs convenient for synthetic tests.
        """

        geometry = config["geometry"]
        mean = geometry["mean_riemann"]
        gates = config["hard_gates"]
        rng = config["rng"]
        return cls(
            mean_tol=float(mean["tol"]),
            mean_maxiter=int(mean["maxiter"]),
            symmetry_relative_error_max=float(
                gates["symmetry_relative_error_max"]
            ),
            condition_number_max=float(gates["condition_number_max"]),
            airm_karcher_residual_max=float(
                gates["airm_karcher_residual_max"]
            ),
            d_centering_relative_error_max=float(
                gates["d_centering_relative_error_max"]
            ),
            g_direct_whitened_relative_error_max=float(
                gates["g_direct_whitened_relative_error_max"]
            ),
            orthogonal_d_relative_error_max=float(
                gates["orthogonal_d_relative_error_max"]
            ),
            orthogonal_g_relative_error_max=float(
                gates["orthogonal_g_relative_error_max"]
            ),
            permutation_equivariance_relative_error_max=float(
                gates["permutation_equivariance_relative_error_max"]
            ),
            le_mean_relative_error_max=float(
                gates["le_mean_relative_error_max"]
            ),
            le_d_g_identity_relative_error_max=float(
                gates["le_d_g_identity_relative_error_max"]
            ),
            degeneracy_epsilon_multiplier=float(
                geometry["shape_degeneracy"]["epsilon_multiplier"]
            ),
            master_seed=int(config["protocol"]["seed"]),
            orthogonal_gauge_family_tag=int(
                rng["family_tags"]["orthogonal_gauge"]
            ),
        )


@dataclass(frozen=True)
class SPDAudit:
    """Per-matrix diagnostics with a single frozen-gate decision."""

    finite: np.ndarray
    symmetry_relative_error: np.ndarray
    min_eigenvalue: np.ndarray
    max_eigenvalue: np.ndarray
    condition_number: np.ndarray
    passed: np.ndarray

    @property
    def all_passed(self) -> bool:
        return bool(np.all(self.passed))


@dataclass(frozen=True)
class MeanAudit:
    """An official observed mean plus independently evaluated correctness."""

    name: str
    geometry: str
    matrix: np.ndarray
    n_samples: int
    warning_messages: tuple[str, ...]
    karcher_residual: float | None
    custom_relative_error: float | None
    spd_audit: SPDAudit
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class BatchedAIRMMeanResult:
    """Result of the actual leading-dimension NumPy-batched Karcher loop.

    Arrays use the input leading group shape.  ``warning_messages`` is in
    canonical C-order flattened group order so that the dataclass remains easy
    to pickle and send to worker processes.
    """

    matrices: np.ndarray
    post_residuals: np.ndarray
    iteration_counts: np.ndarray
    termination_reasons: np.ndarray
    warning_messages: tuple[tuple[str, ...], ...]
    passed: np.ndarray
    scalar_crosscheck_relative_errors: np.ndarray
    scalar_crosscheck_indices: tuple[int, ...]
    scalar_crosscheck_passed: bool
    tol: float
    maxiter: int

    @property
    def all_passed(self) -> bool:
        return bool(np.all(self.passed)) and self.scalar_crosscheck_passed


@dataclass(frozen=True)
class ShapeAudit:
    """Raw and unit shape vector with conservative degeneracy status."""

    object_name: str
    raw_vector: np.ndarray
    norm: float
    degeneracy_threshold: float
    unit_vector: np.ndarray
    is_degenerate: bool


@dataclass(frozen=True)
class GeometryObjects:
    """Observed marginal/class means and the frozen relational objects."""

    geometry: str
    class_order: tuple[str, ...]
    marginal_mean: np.ndarray
    class_means: np.ndarray
    D: np.ndarray
    G: np.ndarray
    d_shape: ShapeAudit
    g_shape: ShapeAudit
    mean_audits: tuple[MeanAudit, ...]
    centered_class_means: np.ndarray
    G_direct: np.ndarray | None
    G_whitened: np.ndarray | None
    gate_metrics: Mapping[str, float]
    gate_passed: bool
    failure_reasons: tuple[str, ...]

    @property
    def zD(self) -> np.ndarray:
        return self.d_shape.unit_vector

    @property
    def zG(self) -> np.ndarray:
        return self.g_shape.unit_vector


def _as_square_matrices(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2] != array.shape[-1]:
        raise ValueError(f"{name} must have shape (..., p, p), got {array.shape}")
    if array.shape[-1] < 1:
        raise ValueError(f"{name} has zero matrix dimension")
    return array


def _as_covariance_sample(value: np.ndarray, *, name: str) -> np.ndarray:
    array = _as_square_matrices(value, name=name)
    if array.ndim != 3 or array.shape[0] < 1:
        raise ValueError(f"{name} must be nonempty (samples, p, p), got {array.shape}")
    return array


def symmetrize(value: np.ndarray) -> np.ndarray:
    """Numerically symmetrize after, never instead of, a symmetry audit."""

    array = _as_square_matrices(value, name="value")
    return 0.5 * (array + np.swapaxes(array, -1, -2))


def _relative_symmetry_error(value: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(
        value - np.swapaxes(value, -1, -2), axis=(-2, -1)
    )
    denominator = np.maximum(
        np.linalg.norm(value, axis=(-2, -1)), np.finfo(np.float64).tiny
    )
    return numerator / denominator


def validate_spd_stack(
    value: np.ndarray,
    *,
    thresholds: GeometryThresholds | None = None,
    raise_on_failure: bool = False,
    name: str = "matrices",
) -> SPDAudit:
    """Audit finite/symmetric/PD/condition gates without clipping or repair."""

    thresholds = thresholds or GeometryThresholds()
    array = _as_square_matrices(value, name=name)
    flat = array.reshape((-1, array.shape[-1], array.shape[-1]))
    finite = np.isfinite(flat).all(axis=(-2, -1))
    symmetry = np.full(len(flat), np.inf, dtype=np.float64)
    minimum = np.full(len(flat), np.nan, dtype=np.float64)
    maximum = np.full(len(flat), np.nan, dtype=np.float64)
    condition = np.full(len(flat), np.inf, dtype=np.float64)

    eligible = np.flatnonzero(finite)
    if len(eligible):
        symmetry[eligible] = _relative_symmetry_error(flat[eligible])
        eigvals = np.linalg.eigvalsh(symmetrize(flat[eligible]))
        minimum[eligible] = eigvals[:, 0]
        maximum[eligible] = eigvals[:, -1]
        positive = eigvals[:, 0] > 0.0
        condition_values = np.full(len(eligible), np.inf, dtype=np.float64)
        condition_values[positive] = (
            eigvals[positive, -1] / eigvals[positive, 0]
        )
        condition[eligible] = condition_values

    passed = (
        finite
        & (symmetry <= thresholds.symmetry_relative_error_max)
        & (minimum > 0.0)
        & (condition <= thresholds.condition_number_max)
    )
    original_shape = array.shape[:-2]
    audit = SPDAudit(
        finite=finite.reshape(original_shape),
        symmetry_relative_error=symmetry.reshape(original_shape),
        min_eigenvalue=minimum.reshape(original_shape),
        max_eigenvalue=maximum.reshape(original_shape),
        condition_number=condition.reshape(original_shape),
        passed=passed.reshape(original_shape),
    )
    if raise_on_failure and not audit.all_passed:
        failed = np.flatnonzero(~passed)
        first = int(failed[0])
        raise NumericalGateError(
            f"{name} failed frozen SPD gate at flat index {first}: "
            f"finite={bool(finite[first])}, symmetry={symmetry[first]:.6g}, "
            f"min_eig={minimum[first]:.6g}, condition={condition[first]:.6g}"
        )
    return audit


def _eigh_checked(value: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    array = _as_square_matrices(value, name=name)
    if not np.isfinite(array).all():
        raise NumericalGateError(f"{name} contains NaN or Inf")
    error = _relative_symmetry_error(array)
    if np.any(error > GeometryThresholds().symmetry_relative_error_max):
        raise NumericalGateError(
            f"{name} is not symmetric; max relative error={float(np.max(error)):.6g}"
        )
    eigvals, eigvecs = np.linalg.eigh(symmetrize(array))
    return eigvals, eigvecs


def _spectral_reconstruct(eigvecs: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = eigvecs @ (
        values[..., None] * np.swapaxes(eigvecs, -1, -2)
    )
    return symmetrize(result)


def spd_log(value: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = _eigh_checked(value, name="SPD logarithm input")
    if np.any(eigvals <= 0.0):
        raise NumericalGateError(
            "SPD logarithm input has non-positive eigenvalue; clipping is forbidden"
        )
    return _spectral_reconstruct(eigvecs, np.log(eigvals))


def symmetric_exp(value: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = _eigh_checked(value, name="symmetric exponential input")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        transformed = np.exp(eigvals)
    if not np.isfinite(transformed).all() or np.any(transformed <= 0.0):
        raise NumericalGateError("symmetric exponential overflowed or underflowed")
    result = _spectral_reconstruct(eigvecs, transformed)
    if np.any(np.linalg.eigvalsh(result) <= 0.0):
        raise NumericalGateError("symmetric exponential result is not SPD")
    return result


def spd_sqrt(value: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = _eigh_checked(value, name="SPD square-root input")
    if np.any(eigvals <= 0.0):
        raise NumericalGateError("SPD square-root input is not positive definite")
    return _spectral_reconstruct(eigvecs, np.sqrt(eigvals))


def spd_invsqrt(value: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = _eigh_checked(value, name="SPD inverse-root input")
    if np.any(eigvals <= 0.0):
        raise NumericalGateError("SPD inverse-root input is not positive definite")
    return _spectral_reconstruct(eigvecs, 1.0 / np.sqrt(eigvals))


def relative_frobenius_error(observed: np.ndarray, reference: np.ndarray) -> float:
    """Relative Frobenius error with the second argument as reference."""

    observed_array = np.asarray(observed, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if observed_array.shape != reference_array.shape:
        raise ValueError("relative-error operands must have identical shape")
    denominator = max(
        float(np.linalg.norm(reference_array.ravel())), np.finfo(np.float64).tiny
    )
    return float(np.linalg.norm((observed_array - reference_array).ravel()) / denominator)


def karcher_residual(covariances: np.ndarray, mean_matrix: np.ndarray) -> float:
    """Independent whitened AIRM first-order residual (unnormalized Frobenius)."""

    array = _as_covariance_sample(covariances, name="covariances")
    whitening = spd_invsqrt(mean_matrix)
    whitened = symmetrize(whitening @ array @ whitening)
    return float(np.linalg.norm(spd_log(whitened).mean(axis=0), ord="fro"))


def _mean_failure_reasons(
    *,
    warning_messages: tuple[str, ...],
    residual: float | None,
    residual_max: float,
    custom_error: float | None,
    custom_error_max: float,
    spd_audit: SPDAudit,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if warning_messages:
        reasons.append("CONVERGENCE_WARNING")
    if residual is not None and (
        not np.isfinite(residual) or residual > residual_max
    ):
        reasons.append("KARCHER_RESIDUAL")
    if custom_error is not None and (
        not np.isfinite(custom_error) or custom_error > custom_error_max
    ):
        reasons.append("OFFICIAL_CUSTOM_MEAN_MISMATCH")
    if not spd_audit.all_passed:
        reasons.append("MEAN_SPD_GATE")
    return tuple(reasons)


def airm_mean_official(
    covariances: np.ndarray,
    *,
    name: str = "mean",
    thresholds: GeometryThresholds | None = None,
) -> MeanAudit:
    """Fit one observed AIRM mean with the frozen public pyRiemann API."""

    thresholds = thresholds or GeometryThresholds()
    array = _as_covariance_sample(covariances, name="covariances")
    validate_spd_stack(
        array, thresholds=thresholds, raise_on_failure=True, name="covariances"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matrix = mean_riemann(
            array,
            tol=thresholds.mean_tol,
            maxiter=thresholds.mean_maxiter,
            init=None,
        )
    matrix = np.asarray(matrix, dtype=np.float64)
    spd_audit = validate_spd_stack(matrix, thresholds=thresholds)
    residual = karcher_residual(array, matrix)
    warning_messages = tuple(str(item.message) for item in caught)
    reasons = _mean_failure_reasons(
        warning_messages=warning_messages,
        residual=residual,
        residual_max=thresholds.airm_karcher_residual_max,
        custom_error=None,
        custom_error_max=thresholds.le_mean_relative_error_max,
        spd_audit=spd_audit,
    )
    return MeanAudit(
        name=name,
        geometry=AIRM,
        matrix=symmetrize(matrix),
        n_samples=len(array),
        warning_messages=warning_messages,
        karcher_residual=residual,
        custom_relative_error=None,
        spd_audit=spd_audit,
        passed=not reasons,
        failure_reasons=reasons,
    )


def le_mean_official(
    covariances: np.ndarray,
    *,
    name: str = "mean",
    thresholds: GeometryThresholds | None = None,
) -> MeanAudit:
    """Fit the official LE mean and cross-check the closed form independently."""

    thresholds = thresholds or GeometryThresholds()
    array = _as_covariance_sample(covariances, name="covariances")
    validate_spd_stack(
        array, thresholds=thresholds, raise_on_failure=True, name="covariances"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        official = np.asarray(mean_logeuclid(array), dtype=np.float64)
    custom = symmetric_exp(spd_log(array).mean(axis=0))
    error = relative_frobenius_error(custom, official)
    spd_audit = validate_spd_stack(official, thresholds=thresholds)
    warning_messages = tuple(str(item.message) for item in caught)
    reasons = _mean_failure_reasons(
        warning_messages=warning_messages,
        residual=None,
        residual_max=thresholds.airm_karcher_residual_max,
        custom_error=error,
        custom_error_max=thresholds.le_mean_relative_error_max,
        spd_audit=spd_audit,
    )
    return MeanAudit(
        name=name,
        geometry=LE,
        matrix=symmetrize(official),
        n_samples=len(array),
        warning_messages=warning_messages,
        karcher_residual=None,
        custom_relative_error=error,
        spd_audit=spd_audit,
        passed=not reasons,
        failure_reasons=reasons,
    )


def _normalize_crosscheck_indices(
    value: bool | Sequence[int], n_groups: int
) -> tuple[int, ...]:
    if value is False:
        return ()
    if value is True:
        return tuple(range(n_groups))
    indices = tuple(int(item) for item in value)
    if len(set(indices)) != len(indices):
        raise ValueError("scalar cross-check indices must be unique")
    if any(item < 0 or item >= n_groups for item in indices):
        raise IndexError("scalar cross-check group index is out of bounds")
    return indices


def airm_mean_batched(
    grouped_covariances: np.ndarray,
    *,
    thresholds: GeometryThresholds | None = None,
    scalar_crosscheck: bool | Sequence[int] = False,
    scalar_crosscheck_tolerance: float = 1.0e-10,
) -> BatchedAIRMMeanResult:
    """Compute exact equal-weight AIRM means with true leading-group batching.

    Parameters
    ----------
    grouped_covariances:
        Shape ``(..., n_samples, p, p)`` with at least one leading group axis.
        The leading dimensions are flattened in C order and processed together
        with batched NumPy eigendecompositions.  Each group retains independent
        ``nu``, ``tau`` and stopping state matching pyRiemann 0.12.
    scalar_crosscheck:
        ``True`` checks every group against the public scalar pyRiemann solver;
        a sequence checks only those flattened group indices.  Production null
        chunks can cross-check a deterministic subset without paying a full
        Python-loop cost.

    The function is top-level, pure, and returns only picklable NumPy/dataclass
    state, making it safe to call from process workers.  It never approximates
    the scientific mean and never clips an eigenvalue.
    """

    thresholds = thresholds or GeometryThresholds()
    array = _as_square_matrices(grouped_covariances, name="grouped_covariances")
    if array.ndim < 4:
        raise ValueError(
            "grouped_covariances must have (..., n_samples, p, p) with a group axis"
        )
    group_shape = array.shape[:-3]
    n_groups = int(np.prod(group_shape, dtype=np.int64))
    n_samples, n_channels = int(array.shape[-3]), int(array.shape[-1])
    if n_groups < 1 or n_samples < 1:
        raise ValueError("batched AIRM mean requires nonempty groups and samples")
    flat = np.asarray(array, dtype=np.float64).reshape(
        n_groups, n_samples, n_channels, n_channels
    )
    validate_spd_stack(
        flat, thresholds=thresholds, raise_on_failure=True, name="grouped_covariances"
    )
    # Symmetrization is only after the <=1e-12 input audit and is not a repair.
    flat = symmetrize(flat)

    means = flat.mean(axis=1)
    nu = np.ones(n_groups, dtype=np.float64)
    tau = np.full(n_groups, np.finfo(np.float64).max, dtype=np.float64)
    active = np.ones(n_groups, dtype=bool)
    iteration_counts = np.zeros(n_groups, dtype=np.int64)
    termination = np.full(n_groups, "MAXITER", dtype=object)

    for iteration in range(thresholds.mean_maxiter):
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            break
        current = means[active_indices]
        eigvals, eigvecs = np.linalg.eigh(current)
        if np.any(eigvals <= 0.0):
            raise NumericalGateError(
                "batched AIRM iterate is not SPD; eigenvalue clipping is forbidden"
            )
        roots = _spectral_reconstruct(eigvecs, np.sqrt(eigvals))
        inverse_roots = _spectral_reconstruct(eigvecs, 1.0 / np.sqrt(eigvals))
        whitened = symmetrize(
            inverse_roots[:, None] @ flat[active_indices] @ inverse_roots[:, None]
        )
        logs = spd_log(whitened)
        # Equal weights are frozen.  The explicit contraction mirrors the
        # public solver's normalized-weight tensordot while retaining groups.
        weights = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
        tangent = np.einsum("n,bnij->bij", weights, logs, optimize=False)
        step = symmetric_exp(nu[active_indices, None, None] * tangent)
        updated = symmetrize(roots @ step @ roots)
        means[active_indices] = updated

        criterion = np.linalg.norm(tangent, axis=(-2, -1))
        h_value = nu[active_indices] * criterion
        improved = h_value < tau[active_indices]
        new_nu = np.where(
            improved, 0.95 * nu[active_indices], 0.5 * nu[active_indices]
        )
        new_tau = np.where(improved, h_value, tau[active_indices])
        nu[active_indices] = new_nu
        tau[active_indices] = new_tau

        residual_stop = criterion <= thresholds.mean_tol
        step_stop = (~residual_stop) & (new_nu <= thresholds.mean_tol)
        stopped = residual_stop | step_stop
        if np.any(stopped):
            stopped_indices = active_indices[stopped]
            iteration_counts[stopped_indices] = iteration + 1
            termination[stopped_indices[residual_stop[stopped]]] = "RESIDUAL_TOLERANCE"
            termination[stopped_indices[step_stop[stopped]]] = "STEP_SIZE_TOLERANCE"
            active[stopped_indices] = False

    iteration_counts[active] = thresholds.mean_maxiter
    warning_messages: list[tuple[str, ...]] = [() for _ in range(n_groups)]
    for index in np.flatnonzero(active):
        warning_messages[int(index)] = ("Convergence not reached",)

    residuals = np.empty(n_groups, dtype=np.float64)
    mean_passed = np.empty(n_groups, dtype=bool)
    for index in range(n_groups):
        residuals[index] = karcher_residual(flat[index], means[index])
        mean_spd = validate_spd_stack(means[index], thresholds=thresholds)
        mean_passed[index] = (
            not warning_messages[index]
            and mean_spd.all_passed
            and np.isfinite(residuals[index])
            and residuals[index] <= thresholds.airm_karcher_residual_max
        )

    crosscheck_indices = _normalize_crosscheck_indices(
        scalar_crosscheck, n_groups
    )
    crosscheck_errors = np.full(n_groups, np.nan, dtype=np.float64)
    crosscheck_passed = True
    for index in crosscheck_indices:
        official = airm_mean_official(
            flat[index], name=f"scalar_crosscheck_{index}", thresholds=thresholds
        )
        error = relative_frobenius_error(means[index], official.matrix)
        crosscheck_errors[index] = error
        if (
            not official.passed
            or not np.isfinite(error)
            or error > scalar_crosscheck_tolerance
        ):
            crosscheck_passed = False

    result_shape = group_shape + (n_channels, n_channels)
    return BatchedAIRMMeanResult(
        matrices=means.reshape(result_shape),
        post_residuals=residuals.reshape(group_shape),
        iteration_counts=iteration_counts.reshape(group_shape),
        termination_reasons=termination.reshape(group_shape),
        warning_messages=tuple(warning_messages),
        passed=mean_passed.reshape(group_shape),
        scalar_crosscheck_relative_errors=crosscheck_errors.reshape(group_shape),
        scalar_crosscheck_indices=crosscheck_indices,
        scalar_crosscheck_passed=crosscheck_passed,
        tol=thresholds.mean_tol,
        maxiter=thresholds.mean_maxiter,
    )


def airm_distance(first: np.ndarray, second: np.ndarray) -> float | np.ndarray:
    """Affine-invariant distance with broadcasting and no eigen clipping."""

    first_array = _as_square_matrices(first, name="first")
    second_array = _as_square_matrices(second, name="second")
    if first_array.shape[-2:] != second_array.shape[-2:]:
        raise ValueError("AIRM distance operands have different matrix dimensions")
    whitening = spd_invsqrt(first_array)
    relative = symmetrize(whitening @ second_array @ whitening)
    distance = np.linalg.norm(spd_log(relative), axis=(-2, -1))
    return float(distance) if np.ndim(distance) == 0 else distance


def airm_distance_matrix(class_means: np.ndarray) -> np.ndarray:
    means = _as_covariance_sample(class_means, name="class_means")
    n_classes = len(means)
    result = np.zeros((n_classes, n_classes), dtype=np.float64)
    for left in range(n_classes):
        for right in range(left + 1, n_classes):
            value = float(airm_distance(means[left], means[right]))
            result[left, right] = value
            result[right, left] = value
    return result


def le_distance_matrix(class_means: np.ndarray) -> np.ndarray:
    means = _as_covariance_sample(class_means, name="class_means")
    logs = spd_log(means)
    differences = logs[:, None] - logs[None, :]
    result = np.linalg.norm(differences, axis=(-2, -1))
    np.fill_diagonal(result, 0.0)
    return symmetrize(result)


def airm_gram_matrices(
    marginal_mean: np.ndarray, class_means: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return primary whitened G, direct-solve G, L_c and U_c.

    The direct expression never forms ``M^{-1}``: it uses
    ``solve(M, U_c)`` and ``Tr(solve(M,U_c) solve(M,U_d))``.
    """

    marginal = _as_square_matrices(marginal_mean, name="marginal_mean")
    means = _as_covariance_sample(class_means, name="class_means")
    if marginal.ndim != 2 or marginal.shape != means.shape[1:]:
        raise ValueError("marginal and class mean dimensions do not match")
    inverse_root = spd_invsqrt(marginal)
    root = spd_sqrt(marginal)
    centered = symmetrize(inverse_root @ means @ inverse_root)
    whitened_tangents = spd_log(centered)
    whitened_gram = np.einsum(
        "cij,dji->cd", whitened_tangents, whitened_tangents, optimize=False
    )
    full_tangents = symmetrize(root @ whitened_tangents @ root)
    solved = np.linalg.solve(marginal, full_tangents)
    direct_gram = np.einsum("cij,dji->cd", solved, solved, optimize=False)
    return (
        symmetrize(whitened_gram),
        symmetrize(direct_gram),
        whitened_tangents,
        full_tangents,
    )


def le_gram_matrix(
    marginal_mean: np.ndarray, class_means: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    marginal_log = spd_log(marginal_mean)
    centered_logs = spd_log(class_means) - marginal_log
    gram = np.einsum("cij,dji->cd", centered_logs, centered_logs, optimize=False)
    return symmetrize(gram), centered_logs


def center_airm(value: np.ndarray, marginal_mean: np.ndarray) -> np.ndarray:
    """Apply the frozen marginal congruence ``M^-1/2 C M^-1/2``."""

    inverse_root = spd_invsqrt(marginal_mean)
    return symmetrize(inverse_root @ value @ inverse_root)


def center_le(value: np.ndarray, marginal_mean: np.ndarray) -> np.ndarray:
    """Apply LE log translation; this is not affine congruence whitening."""

    return symmetric_exp(spd_log(value) - spd_log(marginal_mean))


def svec(value: np.ndarray) -> np.ndarray:
    """Frobenius-isometric upper-triangle vectorization."""

    array = _as_square_matrices(value, name="value")
    n = array.shape[-1]
    row, column = np.triu_indices(n)
    result = array[..., row, column].copy()
    result[..., row != column] *= np.sqrt(2.0)
    return result


def smat(vector: np.ndarray, *, n: int | None = None) -> np.ndarray:
    """Inverse Frobenius-isometric symmetric vectorization."""

    array = np.asarray(vector, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("vector must have at least one dimension")
    dimension = array.shape[-1]
    if n is None:
        n = int((np.sqrt(1 + 8 * dimension) - 1) / 2)
    if n * (n + 1) // 2 != dimension:
        raise ValueError("vector length is not compatible with a symmetric matrix")
    row, column = np.triu_indices(n)
    values = array.copy()
    values[..., row != column] /= np.sqrt(2.0)
    result = np.zeros(array.shape[:-1] + (n, n), dtype=np.float64)
    result[..., row, column] = values
    result[..., column, row] = values
    return result


def _shape_audit(
    vector: np.ndarray,
    *,
    object_name: str,
    epsilon_multiplier: float,
) -> ShapeAudit:
    raw = np.asarray(vector, dtype=np.float64)
    if raw.ndim != 1 or len(raw) < 1:
        raise ValueError("shape vector must be one-dimensional and nonempty")
    if not np.isfinite(raw).all():
        return ShapeAudit(
            object_name=object_name,
            raw_vector=raw,
            norm=float("nan"),
            degeneracy_threshold=float("nan"),
            unit_vector=np.full(raw.shape, np.nan, dtype=np.float64),
            is_degenerate=True,
        )
    norm = float(np.linalg.norm(raw))
    threshold = float(
        epsilon_multiplier
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(raw))))
    )
    degenerate = bool(norm <= threshold)
    unit = (
        np.full(raw.shape, np.nan, dtype=np.float64)
        if degenerate
        else raw / norm
    )
    return ShapeAudit(
        object_name=object_name,
        raw_vector=raw,
        norm=norm,
        degeneracy_threshold=threshold,
        unit_vector=unit,
        is_degenerate=degenerate,
    )


def shape_from_D(
    matrix: np.ndarray, *, epsilon_multiplier: float = 100.0
) -> ShapeAudit:
    D = np.asarray(matrix, dtype=np.float64)
    if D.shape != (4, 4):
        raise ValueError(f"D must be 4-by-4, got {D.shape}")
    vector = np.asarray([D[left, right] for left, right in D_UPPER_TRIANGLE_INDICES])
    return _shape_audit(
        vector, object_name="D", epsilon_multiplier=epsilon_multiplier
    )


def shape_from_G(
    matrix: np.ndarray, *, epsilon_multiplier: float = 100.0
) -> ShapeAudit:
    G = np.asarray(matrix, dtype=np.float64)
    if G.shape != (4, 4):
        raise ValueError(f"G must be 4-by-4, got {G.shape}")
    return _shape_audit(
        svec(G), object_name="G", epsilon_multiplier=epsilon_multiplier
    )


def require_non_degenerate(shape: ShapeAudit) -> np.ndarray:
    if shape.is_degenerate:
        raise DegenerateClassGeometryError(
            f"DEGENERATE_CLASS_GEOMETRY:{shape.object_name}: "
            f"norm={shape.norm!r}, threshold={shape.degeneracy_threshold!r}"
        )
    return shape.unit_vector


def all_class_permutations(n_classes: int = 4) -> tuple[tuple[int, ...], ...]:
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    return tuple(permutations(range(n_classes)))


def _validate_permutation(permutation: Sequence[int], n: int) -> tuple[int, ...]:
    result = tuple(int(item) for item in permutation)
    if len(result) != n or tuple(sorted(result)) != tuple(range(n)):
        raise ValueError(f"permutation must contain each integer 0..{n - 1} once")
    return result


def permutation_matrix(permutation: Sequence[int]) -> np.ndarray:
    order = _validate_permutation(permutation, len(permutation))
    return np.eye(len(order), dtype=np.float64)[list(order)]


def permute_object_matrix(
    matrix: np.ndarray, permutation: Sequence[int]
) -> np.ndarray:
    array = _as_square_matrices(matrix, name="matrix")
    if array.ndim != 2:
        raise ValueError("object permutation expects one matrix")
    order = _validate_permutation(permutation, array.shape[0])
    P = permutation_matrix(order)
    return P @ array @ P.T


def permute_shape_vector(
    unit_shape: np.ndarray,
    *,
    object_name: str,
    permutation: Sequence[int],
    epsilon_multiplier: float = 100.0,
) -> np.ndarray:
    """Reconstruct, permute, and renormalize a frozen D or G unit shape."""

    unit = np.asarray(unit_shape, dtype=np.float64)
    if object_name == "D":
        if unit.shape != (6,):
            raise ValueError("D unit shape must have length 6")
        matrix = np.zeros((4, 4), dtype=np.float64)
        for value, (left, right) in zip(unit, D_UPPER_TRIANGLE_INDICES):
            matrix[left, right] = value
            matrix[right, left] = value
        audit = shape_from_D(
            permute_object_matrix(matrix, permutation),
            epsilon_multiplier=epsilon_multiplier,
        )
    elif object_name == "G":
        if unit.shape != (10,):
            raise ValueError("G unit shape must have length 10")
        audit = shape_from_G(
            permute_object_matrix(smat(unit, n=4), permutation),
            epsilon_multiplier=epsilon_multiplier,
        )
    else:
        raise ValueError("object_name must be 'D' or 'G'")
    return require_non_degenerate(audit)


def frozen_orthogonal_gauge(
    n_channels: int,
    *,
    replicate_index: int = 0,
    phase_tag: int = 0,
    thresholds: GeometryThresholds | None = None,
) -> np.ndarray:
    """Generate the deterministic preregistered orthogonal gauge matrix."""

    thresholds = thresholds or GeometryThresholds()
    if n_channels < 1 or replicate_index < 0 or phase_tag < 0:
        raise ValueError("dimensions, phase tag, and replicate index must be valid")
    sequence = np.random.SeedSequence(
        [
            thresholds.master_seed,
            thresholds.orthogonal_gauge_family_tag,
            int(phase_tag),
            int(replicate_index),
        ]
    )
    generator = np.random.Generator(np.random.PCG64DXSM(sequence))
    raw = generator.standard_normal((n_channels, n_channels))
    Q, R = np.linalg.qr(raw)
    diagonal = np.diag(R)
    if np.any(diagonal == 0.0):
        raise NumericalGateError("orthogonal gauge QR produced a zero diagonal")
    Q = Q * np.where(diagonal < 0.0, -1.0, 1.0)[None, :]
    error = np.linalg.norm(Q.T @ Q - np.eye(n_channels), ord="fro")
    if not np.isfinite(error) or error > 1.0e-12 * max(1, n_channels):
        raise NumericalGateError("deterministic gauge matrix is not orthogonal")
    return Q


def _validate_labels(
    labels: Sequence[Any], n_samples: int, class_order: Sequence[str]
) -> np.ndarray:
    result = np.asarray(labels, dtype=object)
    if result.ndim != 1 or len(result) != n_samples:
        raise ValueError("labels must be one-dimensional and match covariance count")
    expected = tuple(str(item) for item in class_order)
    observed = {str(item) for item in result}
    if observed != set(expected):
        raise ValueError(
            f"labels must contain exactly frozen classes {expected}, got {sorted(observed)}"
        )
    return np.asarray([str(item) for item in result], dtype=object)


def _object_structure_metrics(D: np.ndarray, G: np.ndarray) -> dict[str, float]:
    return {
        "d_symmetry_relative_error": relative_frobenius_error(D.T, D),
        "d_diagonal_max_abs": float(np.max(np.abs(np.diag(D)))),
        "d_minimum": float(np.min(D)),
        "g_symmetry_relative_error": relative_frobenius_error(G.T, G),
    }


def _permutation_gate_errors(
    *,
    geometry: str,
    marginal_mean: np.ndarray,
    class_means: np.ndarray,
    D: np.ndarray,
    G: np.ndarray,
) -> tuple[float, float]:
    maximum_d = 0.0
    maximum_g = 0.0
    for order in all_class_permutations(4):
        reordered = class_means[list(order)]
        if geometry == AIRM:
            observed_d = airm_distance_matrix(reordered)
            observed_g = airm_gram_matrices(marginal_mean, reordered)[0]
        else:
            observed_d = le_distance_matrix(reordered)
            observed_g = le_gram_matrix(marginal_mean, reordered)[0]
        maximum_d = max(
            maximum_d,
            relative_frobenius_error(observed_d, permute_object_matrix(D, order)),
        )
        maximum_g = max(
            maximum_g,
            relative_frobenius_error(observed_g, permute_object_matrix(G, order)),
        )
    return maximum_d, maximum_g


def _gate_failure_reasons(
    *,
    geometry: str,
    mean_audits: Iterable[MeanAudit],
    d_shape: ShapeAudit,
    g_shape: ShapeAudit,
    metrics: Mapping[str, float],
    thresholds: GeometryThresholds,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if any(not audit.passed for audit in mean_audits):
        reasons.append("MEAN_GATE")
    if d_shape.is_degenerate:
        reasons.append("DEGENERATE_CLASS_GEOMETRY:D")
    if g_shape.is_degenerate:
        reasons.append("DEGENERATE_CLASS_GEOMETRY:G")
    comparisons = {
        "d_centering_relative_error": thresholds.d_centering_relative_error_max,
        "orthogonal_d_relative_error": thresholds.orthogonal_d_relative_error_max,
        "orthogonal_g_relative_error": thresholds.orthogonal_g_relative_error_max,
        "permutation_d_relative_error": thresholds.permutation_equivariance_relative_error_max,
        "permutation_g_relative_error": thresholds.permutation_equivariance_relative_error_max,
    }
    if geometry == AIRM:
        comparisons["g_direct_whitened_relative_error"] = (
            thresholds.g_direct_whitened_relative_error_max
        )
    else:
        comparisons["le_d_g_identity_relative_error"] = (
            thresholds.le_d_g_identity_relative_error_max
        )
    for key, limit in comparisons.items():
        value = metrics[key]
        if not np.isfinite(value) or value > limit:
            reasons.append(key.upper())
    if (
        not np.isfinite(metrics["d_symmetry_relative_error"])
        or metrics["d_symmetry_relative_error"] > 1.0e-12
        or not np.isfinite(metrics["d_diagonal_max_abs"])
        or metrics["d_diagonal_max_abs"] > 1.0e-12
        or not np.isfinite(metrics["d_minimum"])
        or metrics["d_minimum"] < -1.0e-12
    ):
        reasons.append("D_STRUCTURE")
    if (
        not np.isfinite(metrics["g_symmetry_relative_error"])
        or metrics["g_symmetry_relative_error"] > 1.0e-12
    ):
        reasons.append("G_STRUCTURE")
    return tuple(dict.fromkeys(reasons))


def _compute_objects(
    covariances: np.ndarray,
    labels: Sequence[Any],
    *,
    geometry: str,
    class_order: Sequence[str],
    thresholds: GeometryThresholds,
    gauge_replicate_index: int,
    gauge_phase_tag: int,
) -> GeometryObjects:
    if geometry not in (AIRM, LE):
        raise ValueError("geometry must be AIRM or LE")
    classes = tuple(str(item) for item in class_order)
    if classes != CLASS_ORDER:
        raise ValueError(f"class order must remain frozen as {CLASS_ORDER}")
    array = _as_covariance_sample(covariances, name="covariances")
    validate_spd_stack(
        array, thresholds=thresholds, raise_on_failure=True, name="covariances"
    )
    y = _validate_labels(labels, len(array), classes)
    mean_function = airm_mean_official if geometry == AIRM else le_mean_official
    marginal = mean_function(array, name="marginal", thresholds=thresholds)
    class_audits = tuple(
        mean_function(
            array[y == class_label], name=class_label, thresholds=thresholds
        )
        for class_label in classes
    )
    means = np.stack([audit.matrix for audit in class_audits], axis=0)
    mean_audits = (marginal,) + class_audits

    if geometry == AIRM:
        D = airm_distance_matrix(means)
        G, G_direct, _, _ = airm_gram_matrices(marginal.matrix, means)
        centered = center_airm(means, marginal.matrix)
        centered_D = airm_distance_matrix(centered)
        centered_G = np.einsum(
            "cij,dji->cd", spd_log(centered), spd_log(centered), optimize=False
        )
        centered_G = symmetrize(centered_G)
        direct_error = relative_frobenius_error(G_direct, G)
    else:
        D = le_distance_matrix(means)
        G, _ = le_gram_matrix(marginal.matrix, means)
        centered = center_le(means, marginal.matrix)
        centered_D = le_distance_matrix(centered)
        centered_G, _ = le_gram_matrix(np.eye(means.shape[-1]), centered)
        G_direct = None
        direct_error = float("nan")

    centering_error = relative_frobenius_error(centered_D, D)
    Q = frozen_orthogonal_gauge(
        means.shape[-1],
        replicate_index=gauge_replicate_index,
        phase_tag=gauge_phase_tag,
        thresholds=thresholds,
    )
    gauged = symmetrize(Q @ centered @ Q.T)
    if geometry == AIRM:
        gauged_D = airm_distance_matrix(gauged)
    else:
        gauged_D = le_distance_matrix(gauged)
    gauged_logs = spd_log(gauged)
    gauged_G = symmetrize(
        np.einsum("cij,dji->cd", gauged_logs, gauged_logs, optimize=False)
    )
    orthogonal_d_error = relative_frobenius_error(gauged_D, centered_D)
    orthogonal_g_error = relative_frobenius_error(gauged_G, centered_G)
    permutation_d_error, permutation_g_error = _permutation_gate_errors(
        geometry=geometry,
        marginal_mean=marginal.matrix,
        class_means=means,
        D=D,
        G=G,
    )
    metrics = _object_structure_metrics(D, G)
    metrics.update(
        {
            "d_centering_relative_error": centering_error,
            "g_direct_whitened_relative_error": direct_error,
            "orthogonal_d_relative_error": orthogonal_d_error,
            "orthogonal_g_relative_error": orthogonal_g_error,
            "permutation_d_relative_error": permutation_d_error,
            "permutation_g_relative_error": permutation_g_error,
            "maximum_mean_karcher_residual": float(
                max(
                    (
                        audit.karcher_residual
                        for audit in mean_audits
                        if audit.karcher_residual is not None
                    ),
                    default=float("nan"),
                )
            ),
            "maximum_le_mean_relative_error": float(
                max(
                    (
                        audit.custom_relative_error
                        for audit in mean_audits
                        if audit.custom_relative_error is not None
                    ),
                    default=float("nan"),
                )
            ),
        }
    )
    if geometry == LE:
        reconstructed_squared = (
            np.diag(G)[:, None] + np.diag(G)[None, :] - 2.0 * G
        )
        metrics["le_d_g_identity_relative_error"] = relative_frobenius_error(
            reconstructed_squared, D**2
        )
    else:
        metrics["le_d_g_identity_relative_error"] = float("nan")

    d_shape = shape_from_D(
        D, epsilon_multiplier=thresholds.degeneracy_epsilon_multiplier
    )
    g_shape = shape_from_G(
        G, epsilon_multiplier=thresholds.degeneracy_epsilon_multiplier
    )
    reasons = _gate_failure_reasons(
        geometry=geometry,
        mean_audits=mean_audits,
        d_shape=d_shape,
        g_shape=g_shape,
        metrics=metrics,
        thresholds=thresholds,
    )
    return GeometryObjects(
        geometry=geometry,
        class_order=classes,
        marginal_mean=marginal.matrix,
        class_means=means,
        D=D,
        G=G,
        d_shape=d_shape,
        g_shape=g_shape,
        mean_audits=mean_audits,
        centered_class_means=centered,
        G_direct=G_direct,
        G_whitened=G,
        gate_metrics=metrics,
        gate_passed=not reasons,
        failure_reasons=reasons,
    )


def compute_airm_objects(
    covariances: np.ndarray,
    labels: Sequence[Any],
    *,
    class_order: Sequence[str] = CLASS_ORDER,
    thresholds: GeometryThresholds | None = None,
    gauge_replicate_index: int = 0,
    gauge_phase_tag: int = 0,
) -> GeometryObjects:
    """Compute all observed AIRM means, D/G shapes, and frozen hard gates."""

    return _compute_objects(
        covariances,
        labels,
        geometry=AIRM,
        class_order=class_order,
        thresholds=thresholds or GeometryThresholds(),
        gauge_replicate_index=gauge_replicate_index,
        gauge_phase_tag=gauge_phase_tag,
    )


def compute_le_objects(
    covariances: np.ndarray,
    labels: Sequence[Any],
    *,
    class_order: Sequence[str] = CLASS_ORDER,
    thresholds: GeometryThresholds | None = None,
    gauge_replicate_index: int = 0,
    gauge_phase_tag: int = 0,
) -> GeometryObjects:
    """Compute all observed LE robustness means, D/G shapes, and hard gates."""

    return _compute_objects(
        covariances,
        labels,
        geometry=LE,
        class_order=class_order,
        thresholds=thresholds or GeometryThresholds(),
        gauge_replicate_index=gauge_replicate_index,
        gauge_phase_tag=gauge_phase_tag,
    )


def compute_geometry_objects(
    covariances: np.ndarray,
    labels: Sequence[Any],
    *,
    geometry: str,
    class_order: Sequence[str] = CLASS_ORDER,
    thresholds: GeometryThresholds | None = None,
    gauge_replicate_index: int = 0,
    gauge_phase_tag: int = 0,
) -> GeometryObjects:
    """Dispatch the exact observed object construction for AIRM or LE."""

    if geometry == AIRM:
        return compute_airm_objects(
            covariances,
            labels,
            class_order=class_order,
            thresholds=thresholds,
            gauge_replicate_index=gauge_replicate_index,
            gauge_phase_tag=gauge_phase_tag,
        )
    if geometry == LE:
        return compute_le_objects(
            covariances,
            labels,
            class_order=class_order,
            thresholds=thresholds,
            gauge_replicate_index=gauge_replicate_index,
            gauge_phase_tag=gauge_phase_tag,
        )
    raise ValueError("geometry must be AIRM or LE")
