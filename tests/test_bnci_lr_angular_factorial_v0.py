from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.bnci_lr_angular_factorial_v0 as diagnostic


ROOT = Path(__file__).resolve().parents[1]
PARENT_ARRAY = (
    ROOT
    / "outputs"
    / "bnci2014_001_local_movement_component_decomposition_v0"
    / "arrays"
    / "component_cost_matrices.npz"
)
PARENT_SPLIT = (
    ROOT
    / "outputs"
    / "bnci2014_001_local_movement_component_decomposition_v0"
    / "arrays"
    / "split_half_component_matrices.npz"
)


def _parent_metadata() -> tuple[np.ndarray, np.ndarray]:
    with np.load(PARENT_ARRAY, allow_pickle=False) as archive:
        return archive["cell_subjects"].copy(), archive["cell_classes"].copy()


def _binary_category_matrix() -> np.ndarray:
    matrix = np.empty((18, 18), dtype=np.float64)
    for subject in range(9):
        for class_index in range(2):
            row = diagnostic.cell_index(subject, class_index, n_classes=2)
            for other_subject in range(9):
                for other_class in range(2):
                    column = diagnostic.cell_index(other_subject, other_class, n_classes=2)
                    if subject == other_subject and class_index == other_class:
                        value = 1.0
                    elif subject == other_subject:
                        value = 3.0
                    elif class_index == other_class:
                        value = 5.0
                    else:
                        value = 6.0
                    matrix[row, column] = value
    return matrix


def test_canonical_left_right_extraction_is_exact_subject_then_hand_order() -> None:
    subjects, classes = _parent_metadata()
    indices = diagnostic.lr_parent_indices(subjects, classes)
    assert indices.tolist() == [index for subject in range(9) for index in (4 * subject, 4 * subject + 1)]
    assert subjects[indices].tolist() == [subject for subject in range(1, 10) for _ in range(2)]
    assert classes[indices].tolist() == list(diagnostic.LR_CLASS_ORDER) * 9


def test_generalized_K4_exactly_reproduces_frozen_parent_angular_statistics() -> None:
    with np.load(PARENT_ARRAY, allow_pickle=False) as archive:
        matrix = archive["c_ang_matrix"]
    result = diagnostic.relation_statistics(matrix, n_classes=4)
    assert result.t_subject == pytest.approx(0.3091561771980925, abs=1e-14, rel=1e-14)
    assert result.t_class == pytest.approx(0.39309843397343514, abs=1e-14, rel=1e-14)
    assert result.t_j == pytest.approx(0.19240885452534362, abs=1e-14, rel=1e-14)


def test_K2_relation_statistics_match_hand_calculation() -> None:
    result = diagnostic.relation_statistics(_binary_category_matrix(), n_classes=2)
    assert result.a_sc == pytest.approx(np.ones((9, 2)))
    assert result.b_sc == pytest.approx(np.full((9, 2), 3.0))
    assert result.c_sc == pytest.approx(np.full((9, 2), 5.0))
    assert result.d_sc == pytest.approx(np.full((9, 2), 6.0))
    assert result.s_sc == pytest.approx(np.full((9, 2), 4.0))
    assert result.class_sc == pytest.approx(np.full((9, 2), 2.0))
    assert result.j_sc == pytest.approx(np.full((9, 2), 1.0))
    assert (result.t_subject, result.t_class, result.t_j) == pytest.approx((4.0, 2.0, 1.0))


def test_subjectbreak_mappings_preserve_class_and_permute_all_subjects() -> None:
    mappings = diagnostic.subjectbreak_mappings(n_classes=2)
    assert mappings.shape == (1999, 18)
    for draw in range(1999):
        for class_index in range(2):
            mapped = mappings[draw, class_index::2]
            assert np.array_equal(np.sort(mapped), np.arange(class_index, 18, 2))


def test_classbreak_mappings_preserve_subject_and_binary_tuple() -> None:
    mappings = diagnostic.classbreak_mappings(n_classes=2)
    assert mappings.shape == (1999, 18)
    for draw in range(1999):
        for subject in range(9):
            pair = mappings[draw, 2 * subject : 2 * subject + 2]
            assert set(pair.tolist()) == {2 * subject, 2 * subject + 1}


def test_rng_mappings_are_exactly_deterministic() -> None:
    assert np.array_equal(
        diagnostic.subjectbreak_mappings(n_classes=2),
        diagnostic.subjectbreak_mappings(n_classes=2),
    )
    assert np.array_equal(
        diagnostic.classbreak_mappings(n_classes=2),
        diagnostic.classbreak_mappings(n_classes=2),
    )


def test_plus_one_pvalue_uses_greater_equal_and_2000_denominator() -> None:
    null = np.concatenate((np.full(17, 2.0), np.full(1982, 0.0)))
    assert diagnostic.plus_one_pvalue(2.0, null) == 18 / 2000


def test_feet_and_tongue_values_cannot_leak_into_extracted_matrix() -> None:
    subjects, classes = _parent_metadata()
    base = np.arange(36 * 36, dtype=np.float64).reshape(36, 36)
    extracted, indices = diagnostic.extract_lr_matrix(base, subjects, classes)
    changed = base.copy()
    excluded = np.setdiff1d(np.arange(36), indices)
    changed[excluded, :] = -1.0e12
    changed[:, excluded] = 1.0e12
    changed_extracted, changed_indices = diagnostic.extract_lr_matrix(changed, subjects, classes)
    assert np.array_equal(changed_indices, indices)
    assert np.array_equal(changed_extracted, extracted)


def test_split_half_left_right_subset_integrity() -> None:
    with np.load(PARENT_SPLIT, allow_pickle=False) as archive:
        subjects = archive["cell_subjects"]
        classes = archive["cell_classes"]
        values = archive["c_ang_matrix"]
        replicates = archive["replicates"]
    indices = diagnostic.lr_parent_indices(subjects, classes)
    assert replicates.tolist() == ["A", "B"]
    assert values.shape == (2, 36, 36)
    for half in range(2):
        extracted, returned = diagnostic.extract_lr_matrix(values[half], subjects, classes)
        assert np.array_equal(returned, indices)
        assert np.array_equal(extracted, values[half][np.ix_(indices, indices)])
