from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.bnci_angular_dual_relation_anatomy_v0 as anatomy


ROOT = Path(__file__).resolve().parents[1]
PARENT_ARRAY = ROOT / "outputs/bnci2014_001_local_movement_component_decomposition_v0/arrays/component_cost_matrices.npz"


def synthetic_directed_matrix() -> np.ndarray:
    row = np.arange(36, dtype=np.float64)[:, None]
    column = np.arange(36, dtype=np.float64)[None, :]
    return 0.2 + 0.019 * row + 0.007 * column + 0.0003 * row * column


def test_parent_K4_statistics_reproduce_exactly() -> None:
    with np.load(PARENT_ARRAY, allow_pickle=False) as archive:
        result = anatomy.relation_statistics(archive["c_ang_matrix"], n_classes=4)
    assert result.t_subject == pytest.approx(0.3091561771980925, abs=1e-14, rel=1e-14)
    assert result.t_class == pytest.approx(0.39309843397343514, abs=1e-14, rel=1e-14)
    assert result.t_j == pytest.approx(0.19240885452534362, abs=1e-14, rel=1e-14)


def test_all_binary_subsets_have_exact_frozen_subject_class_indices() -> None:
    expected_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for pair, name in zip(expected_pairs, anatomy.PAIR_NAMES, strict=True):
        indices = anatomy.class_subset_indices(pair)
        expected = [4 * subject + cls for subject in range(9) for cls in pair]
        assert indices.tolist() == expected, name
        assert len(indices) == 18 and len(np.unique(indices)) == 18


def test_six_pair_subject_and_group_reconstruction_on_directed_matrix() -> None:
    matrix = synthetic_directed_matrix()
    full = anatomy.relation_statistics(matrix, n_classes=4)
    pairs, _ = anatomy.six_pair_statistics(matrix)
    errors = anatomy.reconstruction_errors(full, pairs)
    assert anatomy.maximum_reconstruction_error(errors) < 1e-14
    for key in ("S_s", "C_s", "J_s"):
        assert np.asarray(errors[key]).shape == (9,)
    for key in ("T_S", "T_C", "T_J"):
        assert abs(float(errors[key])) < 1e-14


def test_symmetric_A_and_dual_extraction_indices() -> None:
    matrix = synthetic_directed_matrix()
    dual = anatomy.build_dual_anatomy(matrix)
    expected_a = 0.5 * (matrix + matrix.T)
    assert np.array_equal(dual.symmetric_a, expected_a)
    assert np.array_equal(dual.symmetric_a, dual.symmetric_a.T)
    for subject in range(9):
        indices = [4 * subject + cls for cls in range(4)]
        assert np.array_equal(dual.g[subject], expected_a[np.ix_(indices, indices)])
    for cls in range(4):
        indices = [4 * subject + cls for subject in range(9)]
        assert np.array_equal(dual.h[cls], expected_a[np.ix_(indices, indices)])


def test_G_H_diagonal_consistency() -> None:
    dual = anatomy.build_dual_anatomy(synthetic_directed_matrix())
    for subject in range(9):
        for cls in range(4):
            assert dual.g[subject, cls, cls] == dual.h[cls, subject, subject]


def test_baseline_adjustment_matches_hand_calculation() -> None:
    matrix = np.asarray([[2.0, 7.0], [7.0, 4.0]])
    assert np.array_equal(
        anatomy.baseline_adjust(matrix), np.asarray([[0.0, 4.0], [4.0, 0.0]])
    )


def test_pearson_and_centered_cosine_are_identical_for_nondegenerate_profiles() -> None:
    first = np.asarray([1.0, 4.0, 2.0, 9.0, 3.0, 7.0])
    second = np.asarray([3.0, 2.0, 8.0, 5.0, 1.0, 6.0])
    assert anatomy.pearson_correlation(first, second) == pytest.approx(
        anatomy.centered_cosine_similarity(first, second), abs=1e-15
    )


def test_coarse_contrast_hand_calculation() -> None:
    pair_values = {"LR": 1.0, "LF": 3.0, "LT": 5.0, "RF": 7.0, "RT": 9.0, "FT": 2.0}
    assert anatomy.coarse_effector_boundary_contrast(pair_values) == 4.5


def test_input_matrix_not_mutated() -> None:
    matrix = synthetic_directed_matrix()
    before = matrix.copy()
    anatomy.six_pair_statistics(matrix)
    anatomy.build_dual_anatomy(matrix)
    assert np.array_equal(matrix, before)
