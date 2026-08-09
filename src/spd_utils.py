"""Numerically explicit Log-Euclidean utilities for SPD matrices."""

from __future__ import annotations

import numpy as np


def svec_dimension(n_channels: int) -> int:
    """Return the number of unique entries in an n-by-n symmetric matrix."""
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    return n_channels * (n_channels + 1) // 2


def _require_square(matrices: np.ndarray) -> np.ndarray:
    array = np.asarray(matrices, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != array.shape[-2]:
        raise ValueError(f"Expected (..., C, C), got {array.shape}")
    return array


def svec(matrices: np.ndarray, *, check_symmetric: bool = True) -> np.ndarray:
    """Frobenius-isometric vectorization of one or more symmetric matrices.

    The output uses upper-triangle row-major order. Diagonal entries are kept
    unchanged and off-diagonal entries are multiplied by ``sqrt(2)``.
    """
    array = _require_square(matrices)
    if check_symmetric and not np.allclose(
        array, np.swapaxes(array, -1, -2), rtol=1e-10, atol=1e-12
    ):
        raise ValueError("svec requires symmetric input")
    n_channels = array.shape[-1]
    row, col = np.triu_indices(n_channels)
    result = array[..., row, col].copy()
    result[..., row != col] *= np.sqrt(2.0)
    return result


def matrix_log_spd(matrices: np.ndarray) -> np.ndarray:
    """Compute symmetric matrix logarithms, failing on non-SPD input.

    No eigenvalue floor is applied: OAS is the frozen covariance
    regularization, and a non-positive eigenvalue is a visible pipeline error.
    """
    array = _require_square(matrices)
    if not np.isfinite(array).all():
        raise ValueError("Matrix log input contains NaN or Inf")
    symmetric = 0.5 * (array + np.swapaxes(array, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue <= 0.0:
        raise ValueError(
            f"Matrix log requires strictly positive eigenvalues; min={min_eigenvalue}"
        )
    log_eigenvalues = np.log(eigenvalues)
    logged = (eigenvectors * log_eigenvalues[..., None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    return 0.5 * (logged + np.swapaxes(logged, -1, -2))


def log_svec(matrices: np.ndarray) -> np.ndarray:
    """Map SPD matrices to Log-Euclidean Frobenius coordinates."""
    return svec(matrix_log_spd(matrices), check_symmetric=False)

