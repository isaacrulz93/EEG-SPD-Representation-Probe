"""AIRM/EA centering, isometry, and transformed-SPD hard gates."""

from __future__ import annotations

import numpy as np

from src.geometry_v2 import (
    AIRM,
    EA,
    GEOMETRIES,
    airm_distance,
    airm_mean,
    fit_center,
    karcher_residual,
    select_deterministic_pairs,
    spd_diagnostics,
    tlcenter_constant_domain_crosscheck,
)


def _synthetic_spd(n_matrices: int = 24, n_channels: int = 5) -> np.ndarray:
    rng = np.random.default_rng(606)
    common = rng.normal(size=(n_channels, n_channels))
    common = common @ common.T + np.eye(n_channels)
    matrices = []
    for _ in range(n_matrices):
        perturbation = rng.normal(scale=0.3, size=(n_channels, n_channels))
        factor = common + perturbation @ perturbation.T
        matrices.append(0.5 * (factor + factor.T))
    return np.asarray(matrices, dtype=np.float64)


def test_5_airm_solver_residual_and_centered_mean_are_identity() -> None:
    covariances = _synthetic_spd()
    center = fit_center(covariances, AIRM, tol=1e-9, maxiter=100)
    assert center.solver_tol == 1e-9
    assert center.solver_maxiter == 100
    assert center.iteration_count is None
    assert center.termination_reason == "NA_API_UNAVAILABLE"
    assert center.normalized_karcher_post_residual is not None
    assert center.normalized_karcher_post_residual <= 1e-7

    transformed = center.transform(covariances)
    transformed_mean = airm_mean(transformed, tol=1e-9, maxiter=100)
    identity_distance = float(
        airm_distance(transformed_mean.matrix, np.eye(covariances.shape[-1]))
    ) / np.sqrt(covariances.shape[-1])
    assert identity_distance <= 1e-7
    assert karcher_residual(
        covariances, center.mean_matrix, normalized=True  # type: ignore[arg-type]
    ) <= 1e-7


def test_6_airm_congruence_preserves_selected_pair_distances() -> None:
    covariances = _synthetic_spd()
    transformed = fit_center(covariances, AIRM).transform(covariances)
    uids = np.asarray([f"trial_{index:03d}" for index in range(len(covariances))])
    selection = select_deterministic_pairs(
        uids, seed=20260809, subject="synthetic", n_pairs=32
    )
    absolute_errors = []
    relative_errors = []
    for left, right in selection.indices:
        before = float(airm_distance(covariances[left], covariances[right]))
        after = float(airm_distance(transformed[left], transformed[right]))
        absolute_errors.append(abs(before - after))
        relative_errors.append(abs(before - after) / before)
    assert max(absolute_errors) <= 1e-10
    assert max(relative_errors) <= 1e-10


def test_7_ea_arithmetic_mean_is_identity_after_congruence() -> None:
    covariances = _synthetic_spd()
    transformed = fit_center(covariances, EA).transform(covariances)
    error = np.linalg.norm(transformed.mean(axis=0) - np.eye(5), ord="fro") / np.linalg.norm(
        np.eye(5), ord="fro"
    )
    assert error <= 1e-10


def test_8_every_geometry_transform_remains_finite_symmetric_spd() -> None:
    covariances = _synthetic_spd()
    for geometry in GEOMETRIES:
        transformed = fit_center(covariances, geometry).transform(covariances)
        diagnostics = spd_diagnostics(transformed)
        assert diagnostics["finite"].all()
        assert diagnostics["is_spd"].all()
        assert diagnostics["symmetry_error"].max() <= 1e-12
        assert np.isfinite(diagnostics["condition_number"]).all()


def test_tlcenter_crosscheck_uses_only_a_constant_dummy_domain() -> None:
    covariances = _synthetic_spd(n_matrices=12, n_channels=4)
    center = fit_center(covariances, AIRM)
    result = tlcenter_constant_domain_crosscheck(covariances, center)
    assert result["used_real_labels"] is False
    assert result["n_samples"] == len(covariances)
    assert result["maximum_normalized_airm_distance"] <= 1e-7
