"""Synthetic R/S/P and frozen decision tests for Conditional Geometry v1."""

from __future__ import annotations

import numpy as np
import pytest

from src.conditional_nulls_v1 import (
    PHASE_DISCOVERY,
    all_s4_permutations,
    permuted_shape_bank,
    tagged_replicate_rng,
)
from src.conditional_statistics_v1 import (
    StageEvidence,
    confirmatory_oracle_score_sets,
    confirmatory_shared_subject_scores,
    conservative_candidate_ranks,
    cosine_rows,
    discovery_oracle_score_sets,
    discovery_shared_subject_scores,
    evaluate_fixed_sequence,
    le_robustness_label,
    leave_one_subject_out_influence,
    loso_templates,
    normalize_shape_vectors,
    plus_one_null_summary,
    reliability_subject_scores,
    subject_bootstrap_median,
    subject_bootstrap_paired_median_delta,
    subject_null_percentiles,
    summarize_oracle_scores,
    terminal_airm_decision,
)


_D_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _d_vectorizer(objects: np.ndarray) -> np.ndarray:
    array = np.asarray(objects, dtype=np.float64)
    return np.stack([array[..., i, j] for i, j in _D_PAIRS], axis=-1)


def _objects(n_subjects: int = 4) -> np.ndarray:
    result = np.zeros((n_subjects, 3, 4, 4), dtype=np.float64)
    for subject in range(n_subjects):
        for split in range(3):
            values = np.asarray([1.0, 1.3, 2.1, 1.7, 2.5, 1.9])
            values = values + 0.02 * subject + 0.005 * split * np.arange(6)
            for value, (i, j) in zip(values, _D_PAIRS, strict=True):
                result[subject, split, i, j] = result[subject, split, j, i] = value
    return result


def _shapes(objects: np.ndarray) -> np.ndarray:
    return normalize_shape_vectors(_d_vectorizer(objects))


def test_shape_normalization_and_degenerate_geometry_are_strict() -> None:
    vectors = np.asarray([[3.0, 4.0], [5.0, 12.0]])
    normalized = normalize_shape_vectors(vectors)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)
    with pytest.raises(ValueError, match="DEGENERATE_CLASS_GEOMETRY"):
        normalize_shape_vectors(np.zeros((1, 6)))
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_shape_vectors(np.asarray([[1.0, np.nan]]))


def test_reliability_uses_one_average_free_score_per_subject() -> None:
    shapes = _shapes(_objects())
    scores = reliability_subject_scores(shapes)
    expected = np.sum(shapes[:, 0] * shapes[:, 1], axis=1)
    assert scores.shape == (4,)
    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1e-15)


def test_loso_template_excludes_target_subject_exactly() -> None:
    shapes = normalize_shape_vectors(
        np.asarray(
            [
                [1.0, 0.2, 0.1],
                [0.3, 1.0, 0.2],
                [0.2, 0.4, 1.0],
                [0.7, 0.5, 0.3],
            ]
        )
    )
    templates = loso_templates(shapes)
    altered = shapes.copy()
    altered[0] = normalize_shape_vectors(np.asarray([[0.1, 3.0, 0.5]]))[0]
    altered_templates = loso_templates(altered)
    np.testing.assert_allclose(templates[0], altered_templates[0], rtol=0.0, atol=1e-15)
    assert not np.allclose(templates[1], altered_templates[1])


def test_stage_s_discovery_and_confirmatory_match_frozen_formulas() -> None:
    discovery = _shapes(_objects())
    confirmatory = discovery.copy()
    confirmatory[:, 0] = normalize_shape_vectors(confirmatory[:, 0] + 0.01)
    discovery_scores = discovery_shared_subject_scores(discovery)
    template_a = loso_templates(discovery[:, 0])
    template_b = loso_templates(discovery[:, 1])
    expected_discovery = 0.5 * (
        np.sum(template_a * discovery[:, 1], axis=1)
        + np.sum(template_b * discovery[:, 0], axis=1)
    )
    np.testing.assert_allclose(discovery_scores, expected_discovery, atol=1e-15)

    confirmatory_scores = confirmatory_shared_subject_scores(discovery, confirmatory)
    template_f = loso_templates(discovery[:, 2])
    expected_confirmatory = 0.5 * (
        np.sum(template_f * confirmatory[:, 0], axis=1)
        + np.sum(template_f * confirmatory[:, 1], axis=1)
    )
    np.testing.assert_allclose(confirmatory_scores, expected_confirmatory, atol=1e-15)


def test_oracle_enumerates_exactly_24_and_identity_matches_stage_s_discovery() -> None:
    objects = _objects()
    shapes = _shapes(objects)
    candidates = permuted_shape_bank(objects, _d_vectorizer)
    scores = discovery_oracle_score_sets(shapes, candidates)
    assert scores.shape == (4, 24)
    np.testing.assert_allclose(scores[:, 0], discovery_shared_subject_scores(shapes), atol=1e-15)
    permutations = all_s4_permutations()
    assert len({tuple(row) for row in permutations.tolist()}) == 24


def test_confirmatory_oracle_uses_full_target_not_a_b_average() -> None:
    discovery_objects = _objects()
    confirmatory_objects = _objects()
    confirmatory_objects[:, 2, 0, 1] += 0.2
    confirmatory_objects[:, 2, 1, 0] += 0.2
    discovery_shapes = _shapes(discovery_objects)
    candidate_shapes = permuted_shape_bank(confirmatory_objects, _d_vectorizer)
    scores = confirmatory_oracle_score_sets(discovery_shapes, candidate_shapes)
    expected = np.sum(loso_templates(discovery_shapes[:, 2]) * candidate_shapes[:, 2, 0], axis=1)
    np.testing.assert_allclose(scores[:, 0], expected, atol=1e-15)


def test_conservative_rank_places_every_within_tolerance_tie_ahead() -> None:
    scores = np.linspace(0.8, -1.0, 24)[None, :]
    scores[0, 0] = 1.0
    scores[0, 1] = 1.0 - 0.5e-12
    scores[0, 2] = 1.0 - 2.0e-12
    ranks = conservative_candidate_ranks(scores)
    assert ranks[0, 0] == 2
    summary = summarize_oracle_scores(scores, all_s4_permutations())
    assert summary.identity_ranks.tolist() == [2]
    assert summary.normalized_ranks.tolist() == [(24.0 - 2.0) / 23.0]
    assert not summary.top1_exact[0]
    assert summary.best_indices[0] == 0
    assert summary.second_best_indices[0] == 1
    assert summary.margins[0] == pytest.approx(0.5e-12)


def test_oracle_reporting_tie_break_is_score_then_lexicographic() -> None:
    scores = np.zeros((1, 24), dtype=np.float64)
    scores[0, 7] = 2.0
    scores[0, 3] = 2.0
    summary = summarize_oracle_scores(scores, all_s4_permutations())
    assert summary.best_indices.tolist() == [3]
    assert summary.second_best_indices.tolist() == [7]


def test_plus_one_p_effect_and_subject_percentile_are_exact() -> None:
    null = np.asarray([0.1, 0.3, 0.3, 0.8])
    summary = plus_one_null_summary(0.3, null)
    assert summary.exceedances == 3
    assert summary.p_value == 4 / 5
    assert summary.null_median == 0.3
    assert summary.effect == 0.0
    observed_subjects = np.asarray([0.2, 0.7])
    null_subjects = np.asarray([[0.1, 0.9], [0.2, 0.6], [0.4, 0.7]])
    np.testing.assert_array_equal(
        subject_null_percentiles(observed_subjects, null_subjects),
        np.asarray([3 / 4, 3 / 4]),
    )


def test_bootstrap_is_subject_level_exact_tagged_and_influence_is_cached_score_only() -> None:
    scores = np.arange(9, dtype=np.float64)
    summary = subject_bootstrap_median(scores, replicates=16, phase_tag=PHASE_DISCOVERY)
    expected = []
    for replicate in range(16):
        rng = tagged_replicate_rng(
            family_tag=1401,
            phase_tag=PHASE_DISCOVERY,
            replicate_index=replicate,
        )
        expected.append(np.median(scores[rng.integers(0, 9, size=9, endpoint=False)]))
    np.testing.assert_array_equal(summary.statistics, expected)
    quantiles = np.quantile(expected, [0.025, 0.975], method="linear")
    assert summary.ci_low == quantiles[0]
    assert summary.ci_high == quantiles[1]
    influence = leave_one_subject_out_influence(scores)
    expected_influence = np.asarray(
        [np.median(np.delete(scores, index)) - np.median(scores) for index in range(9)]
    )
    np.testing.assert_array_equal(influence, expected_influence)

    paired = subject_bootstrap_paired_median_delta(
        scores + np.asarray([0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=np.float64),
        scores,
        replicates=16,
    )
    expected_paired = []
    left = scores + np.asarray([0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=np.float64)
    for replicate in range(16):
        rng = tagged_replicate_rng(
            family_tag=1401,
            phase_tag=2,
            replicate_index=replicate,
        )
        sampled = rng.integers(0, 9, size=9, endpoint=False)
        expected_paired.append(np.median(left[sampled]) - np.median(scores[sampled]))
    np.testing.assert_array_equal(paired.statistics, expected_paired)


def test_fixed_sequence_marks_downstream_descriptive_even_if_raw_criterion_passes() -> None:
    passing = StageEvidence(0.1, 0.2, 0.01, True)
    failing = StageEvidence(-0.1, 0.2, 0.01, True)
    decisions, chain_pass = evaluate_fixed_sequence({"R": failing, "S": passing, "P": passing})
    assert decisions["R"].status == "FAIL"
    assert decisions["S"].criterion_pass
    assert decisions["S"].status == "DESCRIPTIVE_ONLY"
    assert decisions["P"].status == "DESCRIPTIVE_ONLY"
    assert not chain_pass

    decisions, chain_pass = evaluate_fixed_sequence({"R": passing, "S": passing, "P": passing})
    assert all(decision.status == "PASS" for decision in decisions.values())
    assert chain_pass


def test_terminal_and_le_labels_cover_every_frozen_chain_pattern() -> None:
    assert terminal_airm_decision(True, True) == "GO_STRONG"
    assert terminal_airm_decision(True, False) == "GO_METRIC_ONLY"
    assert terminal_airm_decision(False, True) == "STOP_TANGENT_ONLY"
    assert terminal_airm_decision(False, False) == "STOP_NO_SHARED_GEOMETRY"
    assert terminal_airm_decision(False, False, failure="data") == "UNASSESSED_DATA_CONTRACT_FAILURE"
    assert terminal_airm_decision(False, False, failure="numerical") == "UNASSESSED_NUMERICAL_FAILURE"
    assert terminal_airm_decision(False, False, failure="degenerate") == "UNASSESSED_DEGENERATE_GEOMETRY"

    assert le_robustness_label((True, False), (True, False)) == "AIRM+LE CONSISTENT"
    assert le_robustness_label((True, True), (True, False)) == "AIRM-SPECIFIC"
    assert (
        le_robustness_label((False, False), (True, False))
        == "LE-ONLY — DOES NOT RESCUE AIRM FAILURE"
    )
    assert le_robustness_label((True, False), (False, True)) == "AIRM/LE DISCORDANT"
