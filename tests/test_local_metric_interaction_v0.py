from __future__ import annotations

import numpy as np
import pytest

from src.local_metric_interaction_v0 import (
    CLASS_ORDER,
    N_CELLS,
    N_CLASSES,
    N_SUBJECTS,
    cell_index,
    classbreak_mappings,
    evaluate_interaction_nulls,
    interaction_contrasts,
    mapped_symmetrized_matrix,
    mechanism_tag,
    subjectbreak_mappings,
    symmetrize_session_roles,
    synthetic_additive_cell_matrix,
    terminal_decision,
)


@pytest.mark.parametrize(
    ("subject_effect", "class_effect"),
    [(0.0, 0.0), (0.7, 0.0), (0.0, 0.4), (0.7, 0.4)],
)
def test_no_interaction_fixtures_have_zero_j(
    subject_effect: float, class_effect: float
) -> None:
    fixture = synthetic_additive_cell_matrix(
        subject_effect=subject_effect, class_effect=class_effect
    )
    result = interaction_contrasts(fixture)
    assert result.t_j == pytest.approx(0.0, abs=1.0e-14)
    assert result.j_sc == pytest.approx(np.zeros((9, 4)), abs=1.0e-14)
    assert result.t_s == pytest.approx(subject_effect, abs=1.0e-14)
    assert result.t_c == pytest.approx(class_effect, abs=1.0e-14)


def test_positive_interaction_fixture_has_known_contrasts() -> None:
    fixture = synthetic_additive_cell_matrix(
        subject_effect=0.7, class_effect=0.4, interaction_effect=0.3
    )
    result = interaction_contrasts(fixture)
    assert result.t_j == pytest.approx(0.3, abs=1.0e-14)
    assert result.j_s == pytest.approx(np.full(9, 0.3), abs=1.0e-14)
    assert result.t_s == pytest.approx(1.0, abs=1.0e-14)
    assert result.t_c == pytest.approx(0.7, abs=1.0e-14)


def test_randomization_mappings_preserve_frozen_blocks_and_are_deterministic() -> None:
    class_first = classbreak_mappings(replicates=25)
    class_second = classbreak_mappings(replicates=25)
    subject_first = subjectbreak_mappings(replicates=25)
    subject_second = subjectbreak_mappings(replicates=25)
    assert np.array_equal(class_first, class_second)
    assert np.array_equal(subject_first, subject_second)
    for mapping in class_first:
        assert np.array_equal(np.sort(mapping), np.arange(N_CELLS))
        for subject in range(N_SUBJECTS):
            mapped = mapping[
                [cell_index(subject, class_index) for class_index in range(N_CLASSES)]
            ]
            expected = [
                cell_index(subject, class_index) for class_index in range(N_CLASSES)
            ]
            assert set(mapped) == set(expected)
    for mapping in subject_first:
        assert np.array_equal(np.sort(mapping), np.arange(N_CELLS))
        for class_index in range(N_CLASSES):
            mapped = mapping[
                [cell_index(subject, class_index) for subject in range(N_SUBJECTS)]
            ]
            expected = [
                cell_index(subject, class_index) for subject in range(N_SUBJECTS)
            ]
            assert set(mapped) == set(expected)


def test_null_mappings_relabel_session1_cells_before_role_symmetrization() -> None:
    rng = np.random.default_rng(20)
    m01 = rng.uniform(size=(N_CELLS, N_CELLS))
    mapping = classbreak_mappings(replicates=1)[0]
    observed = mapped_symmetrized_matrix(m01, mapping)
    expected_directional = m01[:, mapping]
    assert observed == pytest.approx(
        0.5 * (expected_directional + expected_directional.T),
        abs=0.0,
        rel=0.0,
    )


def test_full_1999_null_calibration_rejects_interaction_not_additive_main_effects() -> None:
    additive = evaluate_interaction_nulls(
        synthetic_additive_cell_matrix(subject_effect=0.7, class_effect=0.4)
    )
    assert additive.observed.t_j == pytest.approx(0.0, abs=1.0e-14)
    assert additive.p_j_classbreak >= 0.05
    assert additive.p_j_subjectbreak >= 0.05
    interacting = evaluate_interaction_nulls(
        synthetic_additive_cell_matrix(
            subject_effect=0.7,
            class_effect=0.4,
            interaction_effect=0.3,
        )
    )
    assert interacting.observed.t_j > 0.0
    assert interacting.p_j_classbreak == pytest.approx(0.0005)
    assert interacting.p_j_subjectbreak == pytest.approx(0.0005)


def test_subject_is_the_final_group_unit() -> None:
    rng = np.random.default_rng(21)
    matrix = rng.normal(size=(N_CELLS, N_CELLS))
    matrix = symmetrize_session_roles(matrix)
    result = interaction_contrasts(matrix)
    assert result.j_sc.shape == (N_SUBJECTS, len(CLASS_ORDER))
    assert result.j_s.shape == (N_SUBJECTS,)
    assert result.t_j == pytest.approx(float(np.mean(result.j_s)))


def test_frozen_terminal_and_mechanism_mappings() -> None:
    assert terminal_decision(
        t_j=0.01, p_j_classbreak=0.049, p_j_subjectbreak=0.049
    ) == "GO_STABLE_SUBJECT_CLASS_LOCAL_METRIC_INTERACTION"
    assert terminal_decision(
        t_j=0.01, p_j_classbreak=0.05, p_j_subjectbreak=0.001
    ) == "STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0"
    assert terminal_decision(
        t_j=-0.01, p_j_classbreak=0.001, p_j_subjectbreak=0.001
    ) == "STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0"
    assert mechanism_tag(
        size_t_j=0.1,
        size_p_classbreak=0.01,
        size_p_subjectbreak=0.01,
        normalized_t_j=0.1,
        normalized_p_classbreak=0.01,
        normalized_p_subjectbreak=0.01,
    ) == "BOTH_SIZE_AND_RELATIVE_PATTERN_SUPPORTED"
    assert mechanism_tag(
        size_t_j=-0.1,
        size_p_classbreak=0.01,
        size_p_subjectbreak=0.01,
        normalized_t_j=0.1,
        normalized_p_classbreak=0.01,
        normalized_p_subjectbreak=0.01,
    ) == "RELATIVE_PATTERN_SUPPORTED"
    assert mechanism_tag(
        size_t_j=0.1,
        size_p_classbreak=0.01,
        size_p_subjectbreak=0.01,
        normalized_t_j=0.1,
        normalized_p_classbreak=0.05,
        normalized_p_subjectbreak=0.01,
    ) == "METRIC_SIZE_SUPPORTED_RELATIVE_PATTERN_NOT_ESTABLISHED"
    assert mechanism_tag(
        size_t_j=0.1,
        size_p_classbreak=0.05,
        size_p_subjectbreak=0.01,
        normalized_t_j=0.1,
        normalized_p_classbreak=0.01,
        normalized_p_subjectbreak=0.05,
    ) == "MECHANISM_UNRESOLVED"
