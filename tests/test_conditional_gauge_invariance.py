"""Centering, orthogonal-gauge, and class-permutation hard-gate tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.conditional_geometry_v1 import (
    AIRM,
    CLASS_ORDER,
    LE,
    airm_distance_matrix,
    airm_gram_matrices,
    all_class_permutations,
    center_airm,
    center_le,
    compute_airm_objects,
    compute_le_objects,
    frozen_orthogonal_gauge,
    le_distance_matrix,
    le_gram_matrix,
    permute_object_matrix,
    permute_shape_vector,
    relative_frobenius_error,
    spd_sqrt,
    symmetric_exp,
)


def _cloud(seed: int = 73) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrices: list[np.ndarray] = []
    labels: list[str] = []
    for index, label in enumerate(CLASS_ORDER):
        mixing = rng.normal(size=(3, 3))
        prototype = mixing @ mixing.T + (2.5 + index * 0.3) * np.eye(3)
        root = spd_sqrt(prototype)
        for _ in range(6):
            perturbation = rng.normal(scale=0.02, size=(3, 3))
            perturbation = 0.5 * (perturbation + perturbation.T)
            matrices.append(root @ symmetric_exp(perturbation) @ root)
            labels.append(label)
    return np.asarray(matrices), np.asarray(labels, dtype=object)


@pytest.mark.parametrize("geometry", [AIRM, LE])
def test_frozen_centering_preserves_D(geometry: str) -> None:
    covariances, labels = _cloud()
    objects = (
        compute_airm_objects(covariances, labels)
        if geometry == AIRM
        else compute_le_objects(covariances, labels)
    )
    centered = (
        center_airm(objects.class_means, objects.marginal_mean)
        if geometry == AIRM
        else center_le(objects.class_means, objects.marginal_mean)
    )
    centered_D = (
        airm_distance_matrix(centered)
        if geometry == AIRM
        else le_distance_matrix(centered)
    )
    assert relative_frobenius_error(centered_D, objects.D) <= 1e-10
    assert objects.gate_metrics["d_centering_relative_error"] <= 1e-10


def test_airm_direct_and_whitened_Gram_are_equivalent_without_inverse() -> None:
    covariances, labels = _cloud()
    objects = compute_airm_objects(covariances, labels)
    whitened, direct, _, _ = airm_gram_matrices(
        objects.marginal_mean, objects.class_means
    )
    assert relative_frobenius_error(direct, whitened) <= 1e-10
    assert np.allclose(whitened, objects.G, rtol=1e-12, atol=1e-12)


def test_frozen_orthogonal_Q_is_replayed_exactly_and_is_orthogonal() -> None:
    first = frozen_orthogonal_gauge(5, replicate_index=17, phase_tag=0)
    second = frozen_orthogonal_gauge(5, replicate_index=17, phase_tag=0)
    different = frozen_orthogonal_gauge(5, replicate_index=18, phase_tag=0)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.allclose(first.T @ first, np.eye(5), rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("geometry", [AIRM, LE])
def test_orthogonal_gauge_preserves_centered_D_and_G(geometry: str) -> None:
    covariances, labels = _cloud()
    objects = (
        compute_airm_objects(covariances, labels, gauge_replicate_index=9)
        if geometry == AIRM
        else compute_le_objects(covariances, labels, gauge_replicate_index=9)
    )
    assert objects.gate_metrics["orthogonal_d_relative_error"] <= 1e-10
    assert objects.gate_metrics["orthogonal_g_relative_error"] <= 1e-10
    assert objects.gate_passed, objects.failure_reasons


def test_all_24_S4_mappings_are_lexicographic_with_identity_first() -> None:
    mappings = all_class_permutations()
    assert len(mappings) == 24
    assert len(set(mappings)) == 24
    assert mappings[0] == (0, 1, 2, 3)
    assert mappings == tuple(sorted(mappings))


@pytest.mark.parametrize("geometry", [AIRM, LE])
def test_all_24_class_permutations_are_object_equivariant(geometry: str) -> None:
    covariances, labels = _cloud()
    objects = (
        compute_airm_objects(covariances, labels)
        if geometry == AIRM
        else compute_le_objects(covariances, labels)
    )
    for order in all_class_permutations():
        means = objects.class_means[list(order)]
        if geometry == AIRM:
            observed_D = airm_distance_matrix(means)
            observed_G = airm_gram_matrices(objects.marginal_mean, means)[0]
        else:
            observed_D = le_distance_matrix(means)
            observed_G = le_gram_matrix(objects.marginal_mean, means)[0]
        assert relative_frobenius_error(
            observed_D, permute_object_matrix(objects.D, order)
        ) <= 1e-10
        assert relative_frobenius_error(
            observed_G, permute_object_matrix(objects.G, order)
        ) <= 1e-10


@pytest.mark.parametrize("object_name", ["D", "G"])
def test_shape_permutation_reconstructs_and_renormalizes(object_name: str) -> None:
    covariances, labels = _cloud()
    objects = compute_airm_objects(covariances, labels)
    order = (2, 0, 3, 1)
    source = objects.zD if object_name == "D" else objects.zG
    transformed = permute_shape_vector(
        source, object_name=object_name, permutation=order
    )
    direct_matrix = permute_object_matrix(
        objects.D if object_name == "D" else objects.G, order
    )
    if object_name == "D":
        from src.conditional_geometry_v1 import shape_from_D

        expected = shape_from_D(direct_matrix).unit_vector
    else:
        from src.conditional_geometry_v1 import shape_from_G

        expected = shape_from_G(direct_matrix).unit_vector
    assert np.linalg.norm(transformed) == pytest.approx(1.0, abs=1e-14)
    assert np.allclose(transformed, expected, rtol=1e-13, atol=1e-13)


def test_general_congruence_preserves_airm_D_but_is_not_used_as_LE_gauge() -> None:
    covariances, labels = _cloud()
    objects = compute_airm_objects(covariances, labels)
    transform = np.array([[1.2, 0.1, 0.0], [0.0, 0.8, 0.2], [0.1, 0.0, 1.1]])
    transformed_means = transform @ objects.class_means @ transform.T
    transformed_D = airm_distance_matrix(transformed_means)
    assert relative_frobenius_error(transformed_D, objects.D) <= 1e-10
