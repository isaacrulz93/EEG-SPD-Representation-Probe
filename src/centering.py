"""Subject-marginal centering in Log-Euclidean coordinates."""

from __future__ import annotations

import numpy as np


def subject_center(
    coordinates: np.ndarray, subjects: np.ndarray
) -> tuple[np.ndarray, dict[object, np.ndarray]]:
    """Subtract one all-sample coordinate mean per subject.

    Class labels and window indices are intentionally absent from this API so
    they cannot accidentally influence the centering transform.
    """
    z = np.asarray(coordinates, dtype=np.float64)
    subject_array = np.asarray(subjects)
    if z.ndim != 2:
        raise ValueError(f"coordinates must be 2-D, got {z.shape}")
    if len(z) != len(subject_array):
        raise ValueError("coordinates and subjects must have equal length")
    if len(z) == 0:
        raise ValueError("cannot center an empty array")

    centered = np.empty_like(z)
    means: dict[object, np.ndarray] = {}
    for subject in np.unique(subject_array):
        mask = subject_array == subject
        mean = z[mask].mean(axis=0)
        centered[mask] = z[mask] - mean
        means[subject.item() if hasattr(subject, "item") else subject] = mean
    return centered, means


def centered_mean_max_abs(coordinates: np.ndarray, subjects: np.ndarray) -> float:
    """Largest absolute subject-wise mean coordinate."""
    z = np.asarray(coordinates, dtype=np.float64)
    subject_array = np.asarray(subjects)
    maxima = [np.max(np.abs(z[subject_array == s].mean(axis=0))) for s in np.unique(subject_array)]
    return float(max(maxima))

