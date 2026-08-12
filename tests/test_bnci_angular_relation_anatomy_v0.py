from __future__ import annotations

from pathlib import Path

import numpy as np

from src.bnci_angular_relation_anatomy_v0 import (
    CLASS_ORDER,
    PAIR_NAMES,
    baseline_adjust,
    build_relation_anatomy,
    canonical_cell_classes,
    canonical_cell_subjects,
    centered_correlation,
    leave_one_out_commonality,
    maximum_reconstruction_error,
    pairwise_profile_similarity,
    reconstruction_errors,
    relation_statistics,
    six_pair_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "outputs/bnci2014_001_local_movement_component_decomposition_v0"


def synthetic_matrix() -> np.ndarray:
    row = np.arange(36, dtype=np.float64)[:, None]
    col = np.arange(36, dtype=np.float64)[None, :]
    return 0.1 + 0.03 * row + 0.007 * col + 0.0002 * row * col


def test_canonical_frozen_ordering() -> None:
    assert canonical_cell_subjects().tolist() == [s for s in range(1, 10) for _ in range(4)]
    assert canonical_cell_classes().tolist() == list(CLASS_ORDER) * 9
    assert PAIR_NAMES == ("LR", "LF", "LT", "RF", "RT", "FT")


def test_exact_six_pair_reconstruction_on_nonsymmetric_matrix() -> None:
    matrix = synthetic_matrix()
    full = relation_statistics(matrix)
    pairs = six_pair_statistics(matrix)
    errors = reconstruction_errors(full, pairs)
    assert maximum_reconstruction_error(errors) < 1.0e-14


def test_parent_four_class_statistics_reproduce() -> None:
    with np.load(PARENT / "arrays/component_cost_matrices.npz", allow_pickle=False) as archive:
        observed = relation_statistics(archive["c_ang_matrix"])
    assert np.isclose(observed.t_subject, 0.3091561771980925, atol=1e-12, rtol=1e-12)
    assert np.isclose(observed.t_class, 0.39309843397343514, atol=1e-12, rtol=1e-12)
    assert np.isclose(observed.t_j, 0.19240885452534362, atol=1e-12, rtol=1e-12)


def test_relation_anatomy_symmetry_and_profiles() -> None:
    matrix = synthetic_matrix()
    anatomy = build_relation_anatomy(matrix)
    assert anatomy.g.shape == (9, 4, 4)
    assert anatomy.h.shape == (4, 9, 9)
    assert anatomy.g_profiles.shape == (9, 6)
    assert anatomy.h_profiles.shape == (4, 36)
    assert np.array_equal(anatomy.g, anatomy.g.transpose(0, 2, 1))
    assert np.array_equal(anatomy.h, anatomy.h.transpose(0, 2, 1))
    assert np.allclose(np.diagonal(anatomy.delta_g, axis1=1, axis2=2), 0.0)
    assert np.allclose(np.diagonal(anatomy.delta_h, axis1=1, axis2=2), 0.0)


def test_baseline_adjustment_hand_calculation() -> None:
    matrix = np.asarray([[2.0, 7.0], [7.0, 4.0]])
    expected = np.asarray([[0.0, 4.0], [4.0, 0.0]])
    assert np.array_equal(baseline_adjust(matrix), expected)


def test_profile_similarity_and_leave_one_out() -> None:
    profiles = np.asarray([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [3.0, 6.0, 12.0]])
    distances, correlations = pairwise_profile_similarity(profiles)
    assert np.array_equal(np.diag(distances), np.zeros(3))
    assert np.allclose(correlations, 1.0)
    assert np.allclose(leave_one_out_commonality(profiles), 1.0)
    assert np.isclose(centered_correlation(profiles[0], profiles[1]), 1.0)


def test_frozen_matrix_is_not_mutated() -> None:
    matrix = synthetic_matrix()
    before = matrix.copy()
    relation_statistics(matrix)
    six_pair_statistics(matrix)
    build_relation_anatomy(matrix)
    assert np.array_equal(matrix, before)
