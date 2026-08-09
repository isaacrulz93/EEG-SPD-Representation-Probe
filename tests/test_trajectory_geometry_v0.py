"""Synthetic hard gates for the frozen Trajectory Anatomy v0 geometry."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from src.trajectory_geometry_v0 import (
    AIRM_METRIC,
    ALL_PERMUTATIONS_5,
    LE_METRIC,
    PATH_D10_NAMES,
    SCALAR_11_NAMES,
    airm_distance,
    airm_geodesic,
    airm_inner_product,
    airm_log_map,
    airm_norm,
    bag_canon_d10,
    bag_sorted_d10,
    balanced_factorial_decomposition,
    check_bag_permutation_invariance,
    compare_airm_centering_isometry,
    compute_five_state_geometry,
    distance_matrix,
    distance_matrix_hard_checks,
    five_state_airm_barycenter,
    geodesic_endpoint_hard_checks,
    intrinsic_hard_checks,
    le_geodesic,
    path_d10,
    permute_distance_matrix,
    spd_stack_hard_checks,
    trajectory_hard_checks,
    turning_angles,
)


def _spd(seed: int, n_channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n_channels, n_channels))
    return factor @ factor.T + 1.5 * np.eye(n_channels)


def _five_spd(seed: int = 19, n_channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _spd(seed + 100, n_channels)
    matrices = []
    for _ in range(5):
        perturbation = rng.normal(scale=0.2, size=(n_channels, n_channels))
        matrices.append(base + perturbation @ perturbation.T)
    return np.asarray(matrices, dtype=np.float64)


def _line_metric() -> np.ndarray:
    coordinates = np.asarray([0.0, 1.0, 3.0, 7.0, 12.0])
    return np.abs(coordinates[:, None] - coordinates[None, :])


def test_path_order_and_bag_controls_share_exactly_the_same_ten_edges() -> None:
    matrix = _line_metric()
    expected = np.asarray([1, 3, 7, 12, 2, 6, 11, 4, 9, 5], dtype=float)
    assert PATH_D10_NAMES == (
        "d12",
        "d13",
        "d14",
        "d15",
        "d23",
        "d24",
        "d25",
        "d34",
        "d35",
        "d45",
    )
    np.testing.assert_array_equal(path_d10(matrix), expected)
    np.testing.assert_array_equal(bag_sorted_d10(matrix), np.sort(expected))
    canonical = bag_canon_d10(matrix)
    assert canonical.vector.shape == (10,)
    assert tuple(sorted(canonical.permutation)) == tuple(range(5))
    assert canonical.permutation_one_based.count("-") == 4
    assert sorted(canonical.vector.tolist()) == sorted(expected.tolist())


def test_canonical_bag_is_exactly_invariant_under_all_120_permutations() -> None:
    matrix = _line_metric()
    reference = bag_canon_d10(matrix).vector
    assert ALL_PERMUTATIONS_5 == tuple(permutations(range(5)))
    assert len(ALL_PERMUTATIONS_5) == 120
    for permutation in ALL_PERMUTATIONS_5:
        shuffled = permute_distance_matrix(matrix, permutation)
        np.testing.assert_array_equal(bag_canon_d10(shuffled).vector, reference)
    check = check_bag_permutation_invariance(matrix)
    assert check.permutation_count == 120
    assert check.maximum_absolute_error == 0.0
    assert check.exact_equal is True
    assert check.passed is True


def test_nonidentity_shuffle_changes_only_path_edge_assignment() -> None:
    matrix = _line_metric()
    permutation = (2, 0, 4, 1, 3)
    assert permutation != tuple(range(5))
    shuffled = permute_distance_matrix(matrix, permutation)
    original_path = path_d10(matrix)
    shuffled_path = path_d10(shuffled)
    assert not np.array_equal(shuffled_path, original_path)
    np.testing.assert_array_equal(np.sort(shuffled_path), np.sort(original_path))
    np.testing.assert_array_equal(
        bag_canon_d10(shuffled).vector,
        bag_canon_d10(matrix).vector,
    )
    with pytest.raises(ValueError, match="zero-based index"):
        permute_distance_matrix(matrix, (1, 2, 3, 4, 5))


@pytest.mark.parametrize("metric", [AIRM_METRIC, LE_METRIC])
def test_distance_matrices_and_full_geometry_pass_synthetic_hard_gates(
    metric: str,
) -> None:
    states = _five_spd()
    matrix = distance_matrix(states, metric)
    assert matrix.shape == (5, 5)
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(5))
    assert np.min(matrix) >= 0.0
    distance_checks = distance_matrix_hard_checks(matrix)
    assert distance_checks.triangle_inequality_count == 30
    assert distance_checks.passed is True

    result = compute_five_state_geometry(states, metric)
    assert result.path.shape == (10,)
    assert result.bag_canon.vector.shape == (10,)
    assert result.bag_sorted.shape == (10,)
    assert result.steps.shape == (4,)
    assert result.angles.shape == (3,)
    assert result.deviations.shape == (3,)
    assert result.scalar_vector.shape == (11,)
    assert tuple(result.scalar_dict) == SCALAR_11_NAMES
    assert np.isfinite(result.scalar_vector).all()
    assert intrinsic_hard_checks(result).passed is True
    assert trajectory_hard_checks(states, result).passed is True


def test_airm_full_log_map_metric_norm_equals_airm_distance() -> None:
    base = _spd(100)
    target = _spd(200)
    tangent = airm_log_map(target, base)
    inner = airm_inner_product(tangent, tangent, base)
    norm = airm_norm(tangent, base)
    expected = float(airm_distance(base, target))
    assert inner >= 0.0
    assert norm == pytest.approx(np.sqrt(inner), rel=1e-12, abs=1e-12)
    assert norm == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_turning_angle_sign_convention_distinguishes_straight_and_backtrack() -> None:
    straight_logs = np.arange(5, dtype=float)
    straight = np.exp(straight_logs)[:, None, None]
    backtrack_logs = np.asarray([0.0, 1.0, 0.0, -1.0, -2.0])
    backtrack = np.exp(backtrack_logs)[:, None, None]
    for metric in (AIRM_METRIC, LE_METRIC):
        straight_turns = turning_angles(straight, metric)
        assert not straight_turns.degenerate_mask.any()
        assert float(np.max(straight_turns.angles)) <= 1e-7
        backtrack_turns = turning_angles(backtrack, metric)
        assert backtrack_turns.angles[0] == pytest.approx(np.pi, abs=1e-12)
        assert backtrack_turns.angles[1] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("metric", [AIRM_METRIC, LE_METRIC])
def test_geodesic_endpoints_and_exact_geodesic_path_deviation(metric: str) -> None:
    first = _spd(818)
    second = _spd(991)
    geodesic = airm_geodesic if metric == AIRM_METRIC else le_geodesic
    states = np.asarray([geodesic(first, second, index / 4.0) for index in range(5)])
    endpoint_checks = geodesic_endpoint_hard_checks(first, second, metric)
    assert endpoint_checks.passed is True
    result = compute_five_state_geometry(states, metric)
    assert float(np.max(result.deviations)) <= 1e-10
    assert result.total_path_length == pytest.approx(
        result.endpoint_distance, rel=1e-10, abs=1e-10
    )
    assert result.efficiency == pytest.approx(1.0, rel=1e-10, abs=1e-10)
    # acos is ill-conditioned at cosine one, so test a numerical-zero scale.
    assert float(np.max(result.angles)) <= 1e-6


def test_degenerate_path_is_exposed_and_fails_without_efficiency_imputation() -> None:
    states = np.broadcast_to(np.eye(3), (5, 3, 3)).copy()
    result = compute_five_state_geometry(states, AIRM_METRIC)
    assert result.path_degenerate is True
    assert np.isnan(result.efficiency)
    assert result.angle_degenerate_mask.all()
    checks = intrinsic_hard_checks(result)
    assert checks.path_degenerate is True
    assert checks.degenerate_angle_count == 3
    assert checks.passed is False
    assert trajectory_hard_checks(states, result).passed is False


def test_five_state_airm_barycenter_has_frozen_solver_and_small_residual() -> None:
    states = _five_spd(seed=88)
    mean = five_state_airm_barycenter(states)
    assert mean.metric == AIRM_METRIC
    assert mean.solver_tol == 1e-9
    assert mean.solver_maxiter == 100
    assert mean.warning_messages == ()
    assert mean.normalized_karcher_residual is not None
    assert mean.normalized_karcher_residual <= 1e-7
    assert spd_stack_hard_checks(mean.matrix).passed is True


def test_airm_subject_centering_preserves_d_path_bag_and_length() -> None:
    states = _five_spd(seed=717)
    subject_whole_mean = _spd(412)
    checks = compare_airm_centering_isometry(states, subject_whole_mean)
    assert checks.distance_maximum_absolute_error <= 1e-10
    assert checks.distance_maximum_relative_error <= 1e-10
    assert checks.path_maximum_absolute_error <= 1e-10
    assert checks.path_maximum_relative_error <= 1e-10
    assert checks.bag_maximum_absolute_error <= 1e-10
    assert checks.bag_maximum_relative_error <= 1e-10
    assert checks.path_length_absolute_error <= 1e-10
    assert checks.path_length_relative_error <= 1e-10
    assert checks.passed is True


def test_balanced_factorial_decomposition_matches_orthogonal_manual_ss() -> None:
    subject_effect = np.asarray([-1.0, 0.0, 1.0])
    class_effect = np.asarray([-0.5, 0.5])
    interaction = 0.2 * np.outer(subject_effect, class_effect)
    residual_pattern = np.asarray([-0.15, -0.05, 0.05, 0.15])
    values: list[float] = []
    subjects: list[int] = []
    classes: list[str] = []
    for subject_index, subject_value in enumerate(subject_effect):
        for class_index, class_value in enumerate(class_effect):
            for residual in residual_pattern:
                values.append(
                    3.0
                    + subject_value
                    + class_value
                    + interaction[subject_index, class_index]
                    + residual
                )
                subjects.append(subject_index + 1)
                classes.append(("A", "B")[class_index])
    result = balanced_factorial_decomposition(
        np.asarray(values),
        np.asarray(subjects),
        np.asarray(classes),
        expected_subjects=3,
        expected_classes=2,
        expected_cell_count=4,
    )
    expected_subject = 2 * 4 * np.sum(subject_effect**2)
    expected_class = 3 * 4 * np.sum(class_effect**2)
    expected_interaction = 4 * np.sum(interaction**2)
    expected_residual = 3 * 2 * np.sum(residual_pattern**2)
    assert result.ss_subject == pytest.approx(expected_subject)
    assert result.ss_class == pytest.approx(expected_class)
    assert result.ss_subject_class == pytest.approx(expected_interaction)
    assert result.ss_residual == pytest.approx(expected_residual)
    assert result.closure_relative_error <= 1e-10
    assert (
        result.eta2_subject
        + result.eta2_class
        + result.eta2_subject_class
        + result.eta2_residual
    ) == pytest.approx(1.0, abs=1e-12)
    assert result.degenerate is False
    assert result.passed is True


def test_factorial_decomposition_rejects_unbalance_and_flags_constant_scalar() -> None:
    subjects = np.repeat([1, 2], 4)
    classes = np.tile(np.repeat(["A", "B"], 2), 2)
    constant = balanced_factorial_decomposition(
        np.ones(8),
        subjects,
        classes,
        expected_subjects=2,
        expected_classes=2,
        expected_cell_count=2,
    )
    assert constant.ss_total == 0.0
    assert constant.degenerate is True
    assert np.isnan(constant.eta2_subject)
    assert constant.passed is False
    with pytest.raises(ValueError, match="unbalanced"):
        balanced_factorial_decomposition(
            np.ones(7),
            subjects[:-1],
            classes[:-1],
            expected_subjects=2,
            expected_classes=2,
            expected_cell_count=2,
        )
