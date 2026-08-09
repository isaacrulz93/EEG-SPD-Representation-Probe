"""Frozen, label-free SPD geometry core for the preregistered V2 audit.

All matrix functions use float64 symmetric eigendecomposition.  Inputs and
intermediate SPD matrices are validated, and eigenvalues are never clipped.
The public fitting API intentionally has no label argument.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
from pyriemann.geometry.mean import mean_logeuclid, mean_riemann


__all__ = [
    "RAW",
    "LE",
    "AIRM",
    "EA",
    "GEOMETRIES",
    "AIRM_TOLERANCE",
    "AIRM_MAXITER",
    "AIRMMeanResult",
    "FittedCenter",
    "DeterministicPairSelection",
    "symmetrize",
    "spd_log",
    "symmetric_exp",
    "spd_invsqrt",
    "logm_spd",
    "expm_symmetric",
    "invsqrtm_spd",
    "logeuclidean_mean_custom",
    "logeuclidean_mean_official",
    "arithmetic_mean",
    "airm_mean",
    "karcher_residual",
    "fit_center",
    "transform",
    "transform_covariances",
    "logeuclidean_distance",
    "airm_distance",
    "smat",
    "spd_diagnostics",
    "select_deterministic_pairs",
    "tlcenter_constant_domain_crosscheck",
]


RAW = "RAW"
LE = "LE"
AIRM = "AIRM"
EA = "EA"
GEOMETRIES = (RAW, LE, AIRM, EA)
Geometry = Literal["RAW", "LE", "AIRM", "EA"]

SYMMETRY_TOLERANCE = 1e-12
AIRM_TOLERANCE = 1e-9
AIRM_MAXITER = 100


def _as_square_matrices(matrices: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(matrices, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2] != array.shape[-1]:
        raise ValueError(f"{name} must have shape (..., n, n), got {array.shape}")
    if array.shape[-1] < 1:
        raise ValueError(f"{name} cannot contain empty matrices")
    return array


def _relative_symmetry_error(matrices: np.ndarray) -> np.ndarray:
    difference = matrices - np.swapaxes(matrices, -1, -2)
    numerator = np.linalg.norm(difference, axis=(-2, -1))
    denominator = np.maximum(
        np.linalg.norm(matrices, axis=(-2, -1)), np.finfo(np.float64).tiny
    )
    return numerator / denominator


def symmetrize(matrices: np.ndarray) -> np.ndarray:
    """Return deterministic numerical symmetrization in float64."""

    array = _as_square_matrices(matrices, name="matrices")
    return 0.5 * (array + np.swapaxes(array, -1, -2))


def _symmetric_eigh(
    matrices: np.ndarray,
    *,
    name: str,
    require_positive: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = _as_square_matrices(matrices, name=name)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    symmetry_error = _relative_symmetry_error(array)
    if np.any(symmetry_error > SYMMETRY_TOLERANCE):
        raise ValueError(
            f"{name} is not symmetric within {SYMMETRY_TOLERANCE:g}; "
            f"max relative error={float(np.max(symmetry_error)):.6g}"
        )
    symmetric = symmetrize(array)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if require_positive and np.any(eigenvalues <= 0.0):
        raise ValueError(
            f"{name} must be strictly positive definite; "
            f"minimum eigenvalue={float(np.min(eigenvalues)):.6g}"
        )
    return symmetric, eigenvalues, eigenvectors


def _reconstruct(eigenvectors: np.ndarray, values: np.ndarray) -> np.ndarray:
    reconstructed = (eigenvectors * values[..., None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    return symmetrize(reconstructed)


def spd_log(matrices: np.ndarray) -> np.ndarray:
    """Symmetric matrix logarithm of SPD matrices, without eigen clipping."""

    _, eigenvalues, eigenvectors = _symmetric_eigh(
        matrices, name="SPD logarithm input", require_positive=True
    )
    return _reconstruct(eigenvectors, np.log(eigenvalues))


def symmetric_exp(matrices: np.ndarray) -> np.ndarray:
    """Symmetric matrix exponential, failing on overflow or underflow to zero."""

    _, eigenvalues, eigenvectors = _symmetric_eigh(
        matrices, name="symmetric exponential input", require_positive=False
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        exponentiated = np.exp(eigenvalues)
    if not np.isfinite(exponentiated).all() or np.any(exponentiated <= 0.0):
        raise FloatingPointError(
            "matrix exponential produced a non-finite or non-positive eigenvalue"
        )
    result = _reconstruct(eigenvectors, exponentiated)
    _symmetric_eigh(result, name="matrix exponential result", require_positive=True)
    return result


def spd_invsqrt(matrices: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root of SPD matrices, without inversion/clip."""

    _, eigenvalues, eigenvectors = _symmetric_eigh(
        matrices, name="SPD inverse-root input", require_positive=True
    )
    return _reconstruct(eigenvectors, 1.0 / np.sqrt(eigenvalues))


# Unambiguous aliases for callers familiar with matrix-function naming.
logm_spd = spd_log
expm_symmetric = symmetric_exp
invsqrtm_spd = spd_invsqrt


def _fit_input(covariances: np.ndarray) -> np.ndarray:
    array = _as_square_matrices(covariances, name="covariances")
    if array.ndim != 3 or len(array) == 0:
        raise ValueError(
            f"mean fitting requires a non-empty (samples, n, n) array, got {array.shape}"
        )
    _symmetric_eigh(array, name="covariances", require_positive=True)
    return symmetrize(array)


def logeuclidean_mean_custom(covariances: np.ndarray) -> np.ndarray:
    """Closed-form Log-Euclidean mean using this module's stable operators."""

    array = _fit_input(covariances)
    return symmetric_exp(spd_log(array).mean(axis=0))


def logeuclidean_mean_official(covariances: np.ndarray) -> np.ndarray:
    """Log-Euclidean mean through pyRiemann 0.12's public API."""

    array = _fit_input(covariances)
    result = symmetrize(np.asarray(mean_logeuclid(array), dtype=np.float64))
    _symmetric_eigh(result, name="official Log-Euclidean mean", require_positive=True)
    return result


def arithmetic_mean(covariances: np.ndarray) -> np.ndarray:
    """Arithmetic SPD mean used only by the EA control."""

    array = _fit_input(covariances)
    result = symmetrize(array.mean(axis=0))
    _symmetric_eigh(result, name="arithmetic mean", require_positive=True)
    return result


def karcher_residual(
    covariances: np.ndarray,
    mean_matrix: np.ndarray,
    *,
    normalized: bool = False,
) -> float:
    """Return the AIRM first-order residual at ``mean_matrix``.

    The residual is ``||mean(log(M^-1/2 C_i M^-1/2))||_F`` and, when
    requested, is divided by ``sqrt(n_channels)`` as frozen in the protocol.
    """

    array = _fit_input(covariances)
    mean = _as_square_matrices(mean_matrix, name="mean_matrix")
    if mean.ndim != 2 or mean.shape != array.shape[1:]:
        raise ValueError("mean_matrix shape does not match covariance channels")
    whitening = spd_invsqrt(mean)
    whitened = symmetrize(whitening @ array @ whitening)
    residual = float(np.linalg.norm(spd_log(whitened).mean(axis=0), ord="fro"))
    if normalized:
        residual /= float(np.sqrt(array.shape[-1]))
    return residual


@dataclass(frozen=True)
class AIRMMeanResult:
    """Auditable output of the fixed public pyRiemann Karcher solver."""

    matrix: np.ndarray
    tol: float
    maxiter: int
    warning_messages: tuple[str, ...]
    post_residual: float
    normalized_post_residual: float
    iteration_count: None = None
    termination_reason: str = "NA_API_UNAVAILABLE"

    @property
    def had_warning(self) -> bool:
        return bool(self.warning_messages)


def airm_mean(
    covariances: np.ndarray,
    *,
    tol: float = AIRM_TOLERANCE,
    maxiter: int = AIRM_MAXITER,
) -> AIRMMeanResult:
    """Fit the AIRM mean with fixed ``init=None`` and audit its residual."""

    array = _fit_input(covariances)
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(maxiter, (int, np.integer)) or int(maxiter) < 1:
        raise ValueError("maxiter must be a positive integer")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matrix = mean_riemann(
            array,
            tol=float(tol),
            maxiter=int(maxiter),
            init=None,
        )
    matrix = symmetrize(np.asarray(matrix, dtype=np.float64))
    _symmetric_eigh(matrix, name="AIRM mean", require_positive=True)
    residual = karcher_residual(array, matrix)
    return AIRMMeanResult(
        matrix=matrix,
        tol=float(tol),
        maxiter=int(maxiter),
        warning_messages=tuple(str(item.message) for item in caught),
        post_residual=residual,
        normalized_post_residual=residual / float(np.sqrt(array.shape[-1])),
    )


@dataclass(frozen=True)
class FittedCenter:
    """A fitted label-free marginal center and its numerical audit metadata."""

    geometry: Geometry
    n_channels: int
    fit_sample_count: int
    mean_matrix: np.ndarray | None
    log_mean_matrix: np.ndarray | None
    congruence_matrix: np.ndarray | None
    solver_tol: float | None = None
    solver_maxiter: int | None = None
    solver_warning_messages: tuple[str, ...] = ()
    karcher_post_residual: float | None = None
    normalized_karcher_post_residual: float | None = None
    iteration_count: None = None
    termination_reason: str | None = None

    def transform(self, covariances: np.ndarray) -> np.ndarray:
        """Apply this fitted center; no labels are accepted or inspected."""

        return transform(covariances, self)


def _validate_geometry(geometry: str) -> Geometry:
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    return geometry  # type: ignore[return-value]


def fit_center(
    covariances: np.ndarray,
    geometry: Geometry,
    *,
    tol: float = AIRM_TOLERANCE,
    maxiter: int = AIRM_MAXITER,
) -> FittedCenter:
    """Fit one geometry center from covariates only.

    The signature intentionally exposes only geometry/numerical arguments; it
    has no ``y``, class-label, subject-label, or domain-label parameter.
    """

    geometry = _validate_geometry(geometry)
    array = _fit_input(covariances)
    n_channels = int(array.shape[-1])
    if geometry == RAW:
        return FittedCenter(
            geometry=RAW,
            n_channels=n_channels,
            fit_sample_count=0,
            mean_matrix=None,
            log_mean_matrix=None,
            congruence_matrix=None,
        )
    if geometry == LE:
        log_mean = spd_log(array).mean(axis=0)
        mean = symmetric_exp(log_mean)
        return FittedCenter(
            geometry=LE,
            n_channels=n_channels,
            fit_sample_count=len(array),
            mean_matrix=mean,
            log_mean_matrix=log_mean,
            congruence_matrix=None,
        )
    if geometry == EA:
        mean = arithmetic_mean(array)
        return FittedCenter(
            geometry=EA,
            n_channels=n_channels,
            fit_sample_count=len(array),
            mean_matrix=mean,
            log_mean_matrix=None,
            congruence_matrix=spd_invsqrt(mean),
        )

    result = airm_mean(array, tol=tol, maxiter=maxiter)
    return FittedCenter(
        geometry=AIRM,
        n_channels=n_channels,
        fit_sample_count=len(array),
        mean_matrix=result.matrix,
        log_mean_matrix=None,
        congruence_matrix=spd_invsqrt(result.matrix),
        solver_tol=result.tol,
        solver_maxiter=result.maxiter,
        solver_warning_messages=result.warning_messages,
        karcher_post_residual=result.post_residual,
        normalized_karcher_post_residual=result.normalized_post_residual,
        iteration_count=result.iteration_count,
        termination_reason=result.termination_reason,
    )


def transform(covariances: np.ndarray, center: FittedCenter) -> np.ndarray:
    """Transform SPD matrices with a previously fitted label-free center."""

    if not isinstance(center, FittedCenter):
        raise TypeError("center must be a FittedCenter")
    original = _as_square_matrices(covariances, name="covariances")
    single = original.ndim == 2
    if original.ndim not in (2, 3):
        raise ValueError("transform accepts one SPD matrix or a batch of SPD matrices")
    batch = original[np.newaxis] if single else original
    _symmetric_eigh(batch, name="covariances", require_positive=True)
    if batch.shape[-1] != center.n_channels:
        raise ValueError("covariance channels do not match the fitted center")
    batch = symmetrize(batch)

    if center.geometry == RAW:
        transformed = batch.copy()
    elif center.geometry == LE:
        if center.log_mean_matrix is None:
            raise RuntimeError("LE center is missing its fitted log mean")
        transformed = symmetric_exp(spd_log(batch) - center.log_mean_matrix)
    else:
        if center.congruence_matrix is None:
            raise RuntimeError("congruence center is missing its inverse-root matrix")
        transformed = symmetrize(
            center.congruence_matrix @ batch @ center.congruence_matrix
        )
    _symmetric_eigh(transformed, name="transformed covariances", require_positive=True)
    return transformed[0] if single else transformed


transform_covariances = transform


def logeuclidean_distance(first: np.ndarray, second: np.ndarray) -> float | np.ndarray:
    """Log-Euclidean Frobenius distance with NumPy broadcasting."""

    difference = spd_log(first) - spd_log(second)
    result = np.linalg.norm(difference, axis=(-2, -1))
    return float(result) if np.ndim(result) == 0 else result


def airm_distance(first: np.ndarray, second: np.ndarray) -> float | np.ndarray:
    """Affine-invariant Riemannian distance with NumPy broadcasting."""

    first_array = _as_square_matrices(first, name="first")
    second_array = _as_square_matrices(second, name="second")
    if first_array.shape[-2:] != second_array.shape[-2:]:
        raise ValueError("distance matrices have different channel dimensions")
    whitening = spd_invsqrt(first_array)
    whitened_second = symmetrize(whitening @ second_array @ whitening)
    result = np.linalg.norm(spd_log(whitened_second), axis=(-2, -1))
    return float(result) if np.ndim(result) == 0 else result


def smat(vectors: np.ndarray, n_channels: int | None = None) -> np.ndarray:
    """Inverse of the V1 upper-triangle, Frobenius-isometric ``svec``."""

    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError("vectors must have shape (..., symmetric_dimension)")
    dimension = int(array.shape[-1])
    if n_channels is None:
        inferred = int((np.sqrt(1 + 8 * dimension) - 1) / 2)
        if inferred * (inferred + 1) // 2 != dimension:
            raise ValueError(f"{dimension} is not a symmetric-vector dimension")
        n_channels = inferred
    if not isinstance(n_channels, (int, np.integer)) or int(n_channels) < 1:
        raise ValueError("n_channels must be a positive integer")
    n_channels = int(n_channels)
    if n_channels * (n_channels + 1) // 2 != dimension:
        raise ValueError("n_channels does not match vector dimension")
    row, column = np.triu_indices(n_channels)
    values = array.copy()
    values[..., row != column] /= np.sqrt(2.0)
    result = np.zeros(array.shape[:-1] + (n_channels, n_channels), dtype=np.float64)
    result[..., row, column] = values
    result[..., column, row] = values
    return result


def spd_diagnostics(matrices: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-matrix finite, symmetry, EVD, PD, and condition diagnostics."""

    array = _as_square_matrices(matrices, name="matrices")
    flat = array.reshape((-1,) + array.shape[-2:])
    n_matrices = len(flat)
    finite = np.isfinite(flat).all(axis=(1, 2))
    symmetry_error = np.full(n_matrices, np.inf, dtype=np.float64)
    min_eigenvalue = np.full(n_matrices, np.nan, dtype=np.float64)
    max_eigenvalue = np.full(n_matrices, np.nan, dtype=np.float64)
    condition_number = np.full(n_matrices, np.inf, dtype=np.float64)
    reconstruction_error = np.full(n_matrices, np.inf, dtype=np.float64)
    for index in np.flatnonzero(finite):
        matrix = flat[index]
        symmetry_error[index] = float(_relative_symmetry_error(matrix))
        symmetric = symmetrize(matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        min_eigenvalue[index] = float(eigenvalues[0])
        max_eigenvalue[index] = float(eigenvalues[-1])
        if eigenvalues[0] > 0.0:
            condition_number[index] = float(eigenvalues[-1] / eigenvalues[0])
        reconstructed = _reconstruct(eigenvectors, eigenvalues)
        reconstruction_error[index] = float(
            np.linalg.norm(matrix - reconstructed, ord="fro")
            / max(np.linalg.norm(matrix, ord="fro"), np.finfo(np.float64).tiny)
        )
    is_spd = finite & (symmetry_error <= SYMMETRY_TOLERANCE) & (min_eigenvalue > 0.0)
    return {
        "finite": finite,
        "symmetry_error": symmetry_error,
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": max_eigenvalue,
        "condition_number": condition_number,
        "evd_reconstruction_error": reconstruction_error,
        "is_spd": is_spd,
    }


@dataclass(frozen=True)
class DeterministicPairSelection:
    """Stable UID pair selection plus indices into the caller's current rows."""

    indices: np.ndarray
    uid_pairs: tuple[tuple[str, str], ...]
    digests: tuple[str, ...]


def select_deterministic_pairs(
    trial_uids: np.ndarray,
    *,
    seed: int,
    subject: Any,
    n_pairs: int = 64,
) -> DeterministicPairSelection:
    """Select SHA256-ranked unique unordered pairs, invariant to row order."""

    uids = np.asarray(trial_uids)
    if uids.ndim != 1 or len(uids) < 2:
        raise ValueError("trial_uids must be a one-dimensional array of at least 2 UIDs")
    if any(value is None or str(value) == "" for value in uids):
        raise ValueError("trial_uids cannot contain null or empty values")
    uid_strings = np.asarray([str(value) for value in uids], dtype=object)
    if len(np.unique(uid_strings)) != len(uid_strings):
        raise ValueError("trial_uids must be unique")
    if not isinstance(n_pairs, (int, np.integer)) or int(n_pairs) < 1:
        raise ValueError("n_pairs must be a positive integer")
    n_pairs = int(n_pairs)
    available = len(uids) * (len(uids) - 1) // 2
    if n_pairs > available:
        raise ValueError(f"requested {n_pairs} pairs but only {available} exist")

    original_index = {uid: index for index, uid in enumerate(uid_strings)}
    canonical = sorted(uid_strings.tolist())
    ranked: list[tuple[str, str, str]] = []
    for uid_left, uid_right in combinations(canonical, 2):
        payload = json.dumps(
            [int(seed), str(subject), uid_left, uid_right],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        ranked.append((digest, uid_left, uid_right))
    ranked.sort()
    selected = ranked[:n_pairs]
    indices = np.asarray(
        [[original_index[left], original_index[right]] for _, left, right in selected],
        dtype=np.int64,
    )
    return DeterministicPairSelection(
        indices=indices,
        uid_pairs=tuple((left, right) for _, left, right in selected),
        digests=tuple(digest for digest, _, _ in selected),
    )


def tlcenter_constant_domain_crosscheck(
    covariances: np.ndarray,
    fitted_center: FittedCenter,
) -> dict[str, Any]:
    """Cross-check an AIRM center against public TLCenter without real labels."""

    if fitted_center.geometry != AIRM:
        raise ValueError("TLCenter cross-check is defined only for AIRM")
    array = _fit_input(covariances)
    if len(array) != fitted_center.fit_sample_count:
        raise ValueError("cross-check covariances differ in count from the fit scope")
    from pyriemann.transfer import TLCenter, encode_domains

    dummy_labels = np.full(len(array), "UNLABELED", dtype=object)
    dummy_domains = np.full(len(array), "constant_domain", dtype=object)
    encoded_array, encoded_labels = encode_domains(
        array, dummy_labels, dummy_domains
    )
    official = TLCenter(
        target_domain="constant_domain", metric="riemann"
    ).fit_transform(encoded_array, encoded_labels)
    custom = transform(array, fitted_center)
    distances = np.asarray(airm_distance(custom, official), dtype=np.float64)
    normalized = distances / float(np.sqrt(array.shape[-1]))
    return {
        "n_samples": int(len(array)),
        "maximum_normalized_airm_distance": float(np.max(normalized)),
        "mean_normalized_airm_distance": float(np.mean(normalized)),
        "used_real_labels": False,
    }
