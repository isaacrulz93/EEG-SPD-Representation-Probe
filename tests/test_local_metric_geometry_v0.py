from __future__ import annotations

import math

import numpy as np
import pytest

from src.local_metric_geometry_v0 import (
    DegenerateMetricConfiguration,
    EDGE_ORDER,
    INDUCED_EDGE_PERMUTATIONS,
    VERTEX_PERMUTATIONS,
    airm_edge_vector,
    congruence_transform,
    cross_all_configuration_distances,
    cross_configuration_distances,
    delta_norm,
    delta_raw,
    distance_matrix_from_edges,
    edge_rms_size,
    inversion_counterexample,
    normalize_edges,
    permute_edges,
    quotient_match_reference,
    raw_size_pattern_identity_residual,
)


def _random_spd_states(seed: int, dimension: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(5):
        factor = rng.normal(size=(dimension, dimension))
        states.append(factor @ factor.T + 0.75 * np.eye(dimension))
    return np.asarray(states)


def test_exactly_120_unique_s5_induced_edge_permutations() -> None:
    assert len(VERTEX_PERMUTATIONS) == 120
    assert INDUCED_EDGE_PERMUTATIONS.shape == (120, 10)
    assert len({tuple(row) for row in INDUCED_EDGE_PERMUTATIONS}) == 120
    assert all(np.array_equal(np.sort(row), np.arange(10)) for row in INDUCED_EDGE_PERMUTATIONS)
    # S5 is a strict subgroup of the possible S10 coordinate permutations.
    assert len(INDUCED_EDGE_PERMUTATIONS) < math.factorial(10)


def test_permutation_invariance_for_all_120_vertex_relabelings() -> None:
    edges = airm_edge_vector(_random_spd_states(10))
    for permutation in VERTEX_PERMUTATIONS:
        assert delta_raw(edges, permute_edges(edges, permutation)) <= 1.0e-12


def test_symmetry_quotient_identity_and_triangle_inequality() -> None:
    rng = np.random.default_rng(11)
    for _ in range(25):
        left, middle, right = rng.uniform(0.05, 3.0, size=(3, 10))
        assert delta_raw(left, middle) == pytest.approx(
            delta_raw(middle, left), abs=1.0e-12, rel=1.0e-12
        )
        assert delta_raw(left, right) <= delta_raw(left, middle) + delta_raw(
            middle, right
        ) + 2.0e-12
    original = rng.uniform(0.1, 2.0, size=10)
    relabeled = permute_edges(original, VERTEX_PERMUTATIONS[73])
    assert delta_raw(original, relabeled) <= 1.0e-12
    assert np.array_equal(
        distance_matrix_from_edges(relabeled),
        distance_matrix_from_edges(permute_edges(original, VERTEX_PERMUTATIONS[73])),
    )


def test_vectorized_match_equals_slow_reference() -> None:
    rng = np.random.default_rng(12)
    left = rng.uniform(0.01, 4.0, size=(7, 10))
    right = rng.uniform(0.01, 4.0, size=(9, 10))
    raw, size, normalized = cross_all_configuration_distances(
        left, right, chunk_size=3
    )
    assert np.allclose(
        raw,
        cross_configuration_distances(left, right, mode="raw", chunk_size=4),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert np.allclose(
        size,
        cross_configuration_distances(left, right, mode="size"),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert np.allclose(
        normalized,
        cross_configuration_distances(left, right, mode="normalized", chunk_size=4),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    for row in range(len(left)):
        for column in range(len(right)):
            assert raw[row, column] == pytest.approx(
                quotient_match_reference(left[row], right[column]).distance,
                abs=1.0e-12,
                rel=1.0e-12,
            )
            assert normalized[row, column] == pytest.approx(
                quotient_match_reference(
                    left[row], right[column], normalized=True
                ).distance,
                abs=1.0e-12,
                rel=1.0e-12,
            )


def test_exact_raw_size_pattern_identity() -> None:
    rng = np.random.default_rng(13)
    for _ in range(100):
        left, right = rng.uniform(0.01, 5.0, size=(2, 10))
        assert abs(raw_size_pattern_identity_residual(left, right)) <= 2.0e-12
    normalized = normalize_edges(rng.uniform(0.1, 1.0, size=(8, 10)))
    assert np.allclose(edge_rms_size(normalized), 1.0, atol=1.0e-14, rtol=1.0e-14)


def test_degenerate_configuration_is_never_epsilon_normalized() -> None:
    with pytest.raises(DegenerateMetricConfiguration):
        normalize_edges(np.zeros(10))
    with pytest.raises(DegenerateMetricConfiguration):
        delta_norm(np.zeros(10), np.ones(10))
    with pytest.raises(DegenerateMetricConfiguration):
        cross_all_configuration_distances(np.zeros((1, 10)), np.ones((1, 10)))


def test_airm_distance_geometry_is_common_gl_congruence_invariant() -> None:
    states = _random_spd_states(14, dimension=6)
    baseline = airm_edge_vector(states)
    rng = np.random.default_rng(15)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    orthogonal = airm_edge_vector(congruence_transform(states, q))
    assert np.allclose(baseline, orthogonal, atol=2.0e-11, rtol=2.0e-11)
    left, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    right, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    singular_values = np.linspace(0.7, 1.8, 6)
    transform = left @ np.diag(singular_values) @ right.T
    nonorthogonal = airm_edge_vector(congruence_transform(states, transform))
    assert np.allclose(baseline, nonorthogonal, atol=2.0e-10, rtol=2.0e-10)


def test_pullback_is_only_a_pseudometric_counterexample() -> None:
    fixture = inversion_counterexample(dimension=5)
    assert fixture.quotient_distance <= 1.0e-12
    assert fixture.minimum_orthogonal_scalar_mismatch > 0.1
    assert np.allclose(fixture.edges_a, fixture.edges_b, atol=1.0e-12, rtol=1.0e-12)
    for index, (left, right) in enumerate(EDGE_ORDER):
        assert fixture.edges_a[index] >= 0.0
        assert left < right
