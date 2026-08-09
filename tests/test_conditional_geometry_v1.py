"""Synthetic contract tests for the frozen conditional-geometry core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.conditional_geometry_v1 import (
    AIRM,
    CLASS_ORDER,
    LE,
    ConditionalGeometryError,
    GeometryThresholds,
    NumericalGateError,
    airm_distance_matrix,
    airm_mean_batched,
    airm_mean_official,
    compute_airm_objects,
    compute_geometry_objects,
    compute_le_objects,
    le_mean_official,
    require_non_degenerate,
    shape_from_D,
    shape_from_G,
    spd_sqrt,
    symmetric_exp,
    validate_spd_stack,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_cloud(
    *, seed: int = 41, n_per_class: int = 7, n_channels: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Noncommuting, well-conditioned SPD observations in frozen class order."""

    rng = np.random.default_rng(seed)
    covariances: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_label in enumerate(CLASS_ORDER):
        mixing = rng.normal(size=(n_channels, n_channels))
        prototype = (
            mixing @ mixing.T + (2.0 + 0.25 * class_index) * np.eye(n_channels)
        )
        root = spd_sqrt(prototype)
        for _ in range(n_per_class):
            noise = rng.normal(scale=0.025, size=(n_channels, n_channels))
            noise = 0.5 * (noise + noise.T)
            covariances.append(root @ symmetric_exp(noise) @ root)
            labels.append(class_label)
    return np.asarray(covariances), np.asarray(labels, dtype=object)


def test_frozen_thresholds_are_read_exactly_from_config() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "bnci2014_001_conditional_geometry_v1.yaml").read_text()
    )
    thresholds = GeometryThresholds.from_config(config)
    assert thresholds.mean_tol == 1e-9
    assert thresholds.mean_maxiter == 100
    assert thresholds.airm_karcher_residual_max == 1e-7
    assert thresholds.symmetry_relative_error_max == 1e-12
    assert thresholds.condition_number_max == 1e12
    assert thresholds.degeneracy_epsilon_multiplier == 100.0
    assert thresholds.master_seed == 20260809
    assert thresholds.orthogonal_gauge_family_tag == 1501


def test_spd_gate_rejects_nonpositive_and_ill_conditioned_without_clipping() -> None:
    nonpositive = np.diag([1.0, 0.0, 2.0])
    with pytest.raises(NumericalGateError, match="SPD gate"):
        validate_spd_stack(nonpositive, raise_on_failure=True)

    ill_conditioned = np.diag([1.0, 1.0, 1.0e-13])
    audit = validate_spd_stack(ill_conditioned)
    assert not audit.all_passed
    assert float(audit.min_eigenvalue) == pytest.approx(1.0e-13)
    assert float(audit.condition_number) == pytest.approx(1.0e13)


def test_observed_airm_means_use_official_solver_and_pass_residual() -> None:
    covariances, _ = _synthetic_cloud()
    result = airm_mean_official(covariances)
    assert result.geometry == AIRM
    assert result.n_samples == len(covariances)
    assert result.warning_messages == ()
    assert result.karcher_residual is not None
    assert result.karcher_residual <= 1e-7
    assert result.passed
    assert result.spd_audit.all_passed


def test_observed_le_mean_matches_frozen_closed_form() -> None:
    covariances, _ = _synthetic_cloud()
    result = le_mean_official(covariances)
    assert result.geometry == LE
    assert result.karcher_residual is None
    assert result.custom_relative_error is not None
    assert result.custom_relative_error <= 1e-10
    assert result.passed


def test_airm_objects_have_exact_D_G_shape_contract_and_pass_gates() -> None:
    covariances, labels = _synthetic_cloud()
    result = compute_airm_objects(covariances, labels)

    assert result.geometry == AIRM
    assert result.class_order == CLASS_ORDER
    assert result.marginal_mean.shape == (3, 3)
    assert result.class_means.shape == (4, 3, 3)
    assert result.D.shape == (4, 4)
    assert result.G.shape == (4, 4)
    assert result.zD.shape == (6,)
    assert result.zG.shape == (10,)
    assert np.linalg.norm(result.zD) == pytest.approx(1.0, abs=1e-14)
    assert np.linalg.norm(result.zG) == pytest.approx(1.0, abs=1e-14)
    assert np.array_equal(result.D, result.D.T)
    assert np.array_equal(np.diag(result.D), np.zeros(4))
    assert np.all(result.D >= 0.0)
    assert np.allclose(result.G, result.G.T, rtol=0.0, atol=1e-14)
    assert result.G_direct is not None
    assert result.G_whitened is not None
    assert (
        result.gate_metrics["g_direct_whitened_relative_error"] <= 1e-10
    )
    assert result.gate_passed, result.failure_reasons


def test_le_objects_satisfy_D_squared_Gram_identity() -> None:
    covariances, labels = _synthetic_cloud()
    result = compute_le_objects(covariances, labels)
    diagonal = np.diag(result.G)
    reconstructed = diagonal[:, None] + diagonal[None, :] - 2.0 * result.G
    assert np.allclose(reconstructed, result.D**2, rtol=1e-10, atol=1e-12)
    assert result.gate_metrics["le_d_g_identity_relative_error"] <= 1e-10
    assert result.G_direct is None
    assert result.gate_passed, result.failure_reasons


def test_dispatch_rejects_nonfrozen_geometry_and_class_order() -> None:
    covariances, labels = _synthetic_cloud()
    with pytest.raises(ValueError, match="AIRM or LE"):
        compute_geometry_objects(covariances, labels, geometry="EA")
    with pytest.raises(ValueError, match="class order"):
        compute_airm_objects(
            covariances, labels, class_order=tuple(reversed(CLASS_ORDER))
        )


def test_shape_degeneracy_is_conservative_and_never_divides() -> None:
    D = np.zeros((4, 4), dtype=np.float64)
    G = np.zeros((4, 4), dtype=np.float64)
    d_shape = shape_from_D(D)
    g_shape = shape_from_G(G)
    assert d_shape.is_degenerate and g_shape.is_degenerate
    assert d_shape.norm == 0.0 and g_shape.norm == 0.0
    assert d_shape.degeneracy_threshold == 100.0 * np.finfo(np.float64).eps
    assert np.isnan(d_shape.unit_vector).all()
    with pytest.raises(ConditionalGeometryError, match="DEGENERATE_CLASS_GEOMETRY"):
        require_non_degenerate(d_shape)


def test_batched_airm_matches_each_public_scalar_mean() -> None:
    covariances, _ = _synthetic_cloud(n_per_class=8)
    grouped = covariances.reshape(4, 8, 3, 3)
    result = airm_mean_batched(grouped, scalar_crosscheck=True)

    assert result.matrices.shape == (4, 3, 3)
    assert result.post_residuals.shape == (4,)
    assert result.iteration_counts.shape == (4,)
    assert result.termination_reasons.shape == (4,)
    assert result.scalar_crosscheck_indices == (0, 1, 2, 3)
    assert result.scalar_crosscheck_passed
    assert np.nanmax(result.scalar_crosscheck_relative_errors) <= 1e-10
    assert np.all(result.post_residuals <= 1e-7)
    assert np.all(result.passed)
    assert result.all_passed


def test_batched_airm_preserves_multiple_leading_dimensions_and_subset_check() -> None:
    first, _ = _synthetic_cloud(seed=11, n_per_class=5)
    second, _ = _synthetic_cloud(seed=12, n_per_class=5)
    grouped = np.stack(
        [first.reshape(4, 5, 3, 3), second.reshape(4, 5, 3, 3)], axis=0
    )
    result = airm_mean_batched(grouped, scalar_crosscheck=(0, 7))
    assert result.matrices.shape == (2, 4, 3, 3)
    assert result.post_residuals.shape == (2, 4)
    flat_errors = result.scalar_crosscheck_relative_errors.ravel()
    assert np.isfinite(flat_errors[[0, 7]]).all()
    assert np.isnan(flat_errors[[1, 2, 3, 4, 5, 6]]).all()
    assert result.scalar_crosscheck_passed


def test_airm_distance_matrix_is_exactly_symmetric_by_construction() -> None:
    covariances, labels = _synthetic_cloud()
    objects = compute_airm_objects(covariances, labels)
    recomputed = airm_distance_matrix(objects.class_means)
    assert np.array_equal(recomputed, recomputed.T)
    assert np.array_equal(np.diag(recomputed), np.zeros(4))
    assert np.allclose(recomputed, objects.D, rtol=1e-13, atol=1e-13)
