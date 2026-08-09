"""Log-Euclidean matrix/coordinate equivalence and isometry hard gates."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.geometry_v2 import (
    LE,
    fit_center,
    logeuclidean_distance,
    smat,
    spd_log,
    symmetric_exp,
)
from src.spd_utils import svec


def _synthetic_spd(n_matrices: int = 20, n_channels: int = 5) -> np.ndarray:
    rng = np.random.default_rng(9182)
    matrices = []
    for index in range(n_matrices):
        factor = rng.normal(size=(n_channels, n_channels))
        offset = np.diag(np.linspace(0.4, 1.0, n_channels)) * (1 + 0.03 * index)
        matrices.append(factor @ factor.T + offset)
    return np.asarray(matrices, dtype=np.float64)


def test_2_le_matrix_transform_equals_v1_coordinate_subtraction() -> None:
    covariances = _synthetic_spd()
    center = fit_center(covariances, LE)
    transformed = center.transform(covariances)

    raw_coordinates = svec(spd_log(covariances))
    v1_centered_coordinates = raw_coordinates - raw_coordinates.mean(axis=0)
    matrix_centered_coordinates = svec(spd_log(transformed))
    np.testing.assert_allclose(
        matrix_centered_coordinates,
        v1_centered_coordinates,
        rtol=1e-10,
        atol=1e-10,
    )

    reconstructed = symmetric_exp(smat(v1_centered_coordinates))
    relative = np.linalg.norm(reconstructed - transformed, axis=(1, 2)) / np.linalg.norm(
        transformed, axis=(1, 2)
    )
    assert relative.max() <= 1e-10


def test_3_le_fitted_log_mean_is_identity_zero() -> None:
    covariances = _synthetic_spd()
    transformed = fit_center(covariances, LE).transform(covariances)
    normalized = np.linalg.norm(spd_log(transformed).mean(axis=0), ord="fro") / np.sqrt(
        transformed.shape[-1]
    )
    assert normalized <= 1e-10


def test_4_le_translation_preserves_logeuclidean_pair_distances() -> None:
    covariances = _synthetic_spd(n_matrices=12)
    transformed = fit_center(covariances, LE).transform(covariances)
    absolute_errors = []
    relative_errors = []
    for left, right in combinations(range(len(covariances)), 2):
        before = float(logeuclidean_distance(covariances[left], covariances[right]))
        after = float(logeuclidean_distance(transformed[left], transformed[right]))
        absolute_errors.append(abs(before - after))
        relative_errors.append(abs(before - after) / before)
    assert max(absolute_errors) <= 1e-10
    assert max(relative_errors) <= 1e-10
