"""Synthetic/unit tests for the pairwise common-action amendment."""

from __future__ import annotations

import numpy as np
import pytest

from src.common_action_solver_v0 import conjugate
from src.pairwise_common_action_v1 import (
    NULL_REPLICATES,
    PAIRWISE_SETTINGS,
    PairwiseContractError,
    aggregate_pairwise_gains,
    assess_pairwise_prediction,
    evaluate_stage_gate,
    fit_pairwise_action,
    pairwise_error_gain,
    semantic_null_statistics,
    terminal_decision,
    unrelated_target_gain_bank,
    unrelated_target_null_statistics,
)


def _orthogonal(seed: int, dimension: int, determinant: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    if int(np.sign(np.linalg.det(q))) != int(determinant):
        q[:, 0] *= -1.0
    return q


def _bank(seed: int, count: int, dimension: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=(count, dimension, dimension))
    return 0.5 * (values + values.transpose(0, 2, 1))


def test_pairwise_fit_uses_exactly_four_total_starts_and_both_sectors() -> None:
    templates = _bank(10, 4, 7)
    truth = _orthogonal(11, 7, -1)
    targets = conjugate(truth, templates[:3])
    result = fit_pairwise_action(targets, templates[:3], seed_parts=("contract",))
    assert PAIRWISE_SETTINGS.starts == 4
    assert len(result.initial_actions) == 4
    assert len(result.fit.starts) == 4
    assert result.initial_determinants.count(-1) == 2
    assert result.initial_determinants.count(1) == 2
    prediction = conjugate(result.fit.matrix, templates[3])
    expected = conjugate(truth, templates[3])
    assert np.linalg.norm(prediction - expected) / np.linalg.norm(expected) < 1.0e-8


def test_pairwise_identifiability_uses_predictions_and_independent_halves() -> None:
    templates = _bank(20, 4, 6)
    truth = _orthogonal(21, 6, 1)
    targets = conjugate(truth, templates[:3])
    result = fit_pairwise_action(targets, templates[:3], seed_parts=("ident",))
    heldout = conjugate(truth, templates[3])
    assessment = assess_pairwise_prediction(
        result,
        targets,
        templates[:3],
        templates[3],
        split_half_prediction_a=heldout,
        split_half_prediction_b=heldout,
    )
    assert assessment.identifiability.classification in {
        "PREDICTIVELY_IDENTIFIABLE",
        "HARMLESS_Q_NONUNIQUENESS",
    }
    assert assessment.identifiability.maximum_relative_prediction_dispersion <= assessment.identifiability.materiality_threshold
    assert assessment.stabilizer.numerical_nullity == 0


def test_pairwise_error_gain_uses_target_squared_norm() -> None:
    target = 2.0 * np.eye(3)
    source = np.eye(3)
    prediction = 1.5 * np.eye(3)
    raw, action, gain = pairwise_error_gain(target, source, prediction)
    assert raw == pytest.approx(0.25)
    assert action == pytest.approx(0.0625)
    assert gain == pytest.approx(0.1875)


def test_aggregation_is_nested_by_source_then_target_subject() -> None:
    gains = np.full((9, 9, 2, 4), np.nan)
    for target in range(9):
        for source in range(9):
            if source != target:
                gains[target, source] = target + source / 100.0
    target_cells, subjects, group = aggregate_pairwise_gains(gains)
    for target in range(9):
        expected = np.median([target + source / 100.0 for source in range(9) if source != target])
        np.testing.assert_allclose(target_cells[target], expected)
        assert subjects[target] == pytest.approx(expected)
    assert group == pytest.approx(np.median(subjects))


def test_available_case_pair_drop_is_forbidden() -> None:
    gains = np.ones((9, 9, 2, 4), dtype=np.float64)
    gains[np.arange(9), np.arange(9)] = np.nan
    gains[3, 4, 1, 2] = np.nan
    with pytest.raises(PairwiseContractError, match="available-case"):
        aggregate_pairwise_gains(gains)


def _global_fixture() -> tuple[np.ndarray, np.ndarray]:
    dimension = 4
    templates = _bank(30, 8, dimension).reshape(2, 4, dimension, dimension)
    actions = np.stack(
        [
            _orthogonal(31 + subject, dimension, -1 if subject % 2 else 1)
            for subject in range(9)
        ]
    )
    U = np.empty((9, 2, 4, dimension, dimension))
    for subject in range(9):
        U[subject] = conjugate(actions[subject], templates)
    relative = np.full((9, 9, 2, 4, dimension, dimension), np.nan)
    for target in range(9):
        for source in range(9):
            if source != target:
                relative[target, source] = actions[target] @ actions[source].T
    return U, relative


def test_unrelated_target_null_deranges_actions_without_pair_pseudoreplication() -> None:
    U, actions = _global_fixture()
    bank = unrelated_target_gain_bank(U, actions)
    first, choices_first = unrelated_target_null_statistics(
        bank, stream="stage_A_unrelated_target", replicates=17
    )
    second, choices_second = unrelated_target_null_statistics(
        bank, stream="stage_A_unrelated_target", replicates=17
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(choices_first, choices_second)
    assert first.shape == (17,)
    assert choices_first.shape == (17, 9, 2, 4)
    assert np.isfinite(first).all()


def test_semantic_null_shares_one_choice_across_sources_per_target_cell() -> None:
    values = np.full((9, 9, 2, 4, 5), np.nan)
    for target in range(9):
        for source in range(9):
            if source != target:
                for choice in range(5):
                    values[target, source, :, :, choice] = target + choice + source / 100.0
    statistics, choices = semantic_null_statistics(
        values, stream="stage_A_semantic", replicates=19
    )
    assert statistics.shape == (19,)
    assert choices.shape == (19, 9, 2, 4)
    assert np.all((choices >= 0) & (choices < 5))
    np.testing.assert_array_equal(
        statistics,
        semantic_null_statistics(values, stream="stage_A_semantic", replicates=19)[0],
    )


def test_stage_gate_requires_positive_gain_effect_and_plus_one_p() -> None:
    passing = evaluate_stage_gate(1.0, np.linspace(-1.0, 0.9, NULL_REPLICATES))
    assert passing.passed and passing.effect > 0.0 and passing.p_value <= 0.05
    failing = evaluate_stage_gate(0.0, np.zeros(NULL_REPLICATES))
    assert not failing.passed
    assert failing.p_value == 1.0


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (dict(data_gate_pass=False, technical_gate_pass=False, identifiable=False, stage_a_primary_pass=None, stage_a_semantic_pass=None, stage_b_primary_pass=None, stage_b_semantic_pass=None), "UNASSESSED_NUMERICAL_OR_DATA_FAILURE"),
        (dict(data_gate_pass=True, technical_gate_pass=False, identifiable=False, stage_a_primary_pass=None, stage_a_semantic_pass=None, stage_b_primary_pass=None, stage_b_semantic_pass=None), "UNASSESSED_TECHNICAL_FAILURE"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=False, stage_a_primary_pass=None, stage_a_semantic_pass=None, stage_b_primary_pass=None, stage_b_semantic_pass=None), "UNASSESSED_ACTION_NOT_IDENTIFIABLE"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=True, stage_a_primary_pass=False, stage_a_semantic_pass=None, stage_b_primary_pass=None, stage_b_semantic_pass=None), "PAIRWISE_COMMON_ACTION_NOT_SUPPORTED_WITHIN_SESSION"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=True, stage_a_primary_pass=True, stage_a_semantic_pass=False, stage_b_primary_pass=None, stage_b_semantic_pass=None), "PAIRWISE_COMMON_ACTION_NOT_SEMANTICALLY_SUPPORTED"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=True, stage_a_primary_pass=True, stage_a_semantic_pass=True, stage_b_primary_pass=False, stage_b_semantic_pass=None), "PAIRWISE_COMMON_ACTION_WITHIN_SESSION_ONLY"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=True, stage_a_primary_pass=True, stage_a_semantic_pass=True, stage_b_primary_pass=True, stage_b_semantic_pass=False), "PAIRWISE_COMMON_ACTION_NOT_CROSS_SESSION_SEMANTICALLY_SUPPORTED"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=True, stage_a_primary_pass=True, stage_a_semantic_pass=True, stage_b_primary_pass=True, stage_b_semantic_pass=True), "PAIRWISE_COMMON_ACTION_NECESSARY_CONSEQUENCE_SUPPORTED"),
    ],
)
def test_terminal_mapping(arguments: dict[str, object], expected: str) -> None:
    assert terminal_decision(**arguments) == expected
