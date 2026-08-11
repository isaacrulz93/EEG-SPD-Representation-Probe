from __future__ import annotations

import numpy as np
import pytest

from src.local_metric_interaction_v0 import (
    classbreak_mappings as prior_classbreak_mappings,
)
from src.local_metric_interaction_v0 import (
    subjectbreak_mappings as prior_subjectbreak_mappings,
)
from src.local_temporal_sequence_v0 import (
    ALL_PERMUTATIONS,
    COMPOSITION_INDICES,
    DERANGEMENT_INDICES,
    DERANGEMENT_MASK,
    IDENTITY_INDEX,
    N_CELLS,
    classbreak_mappings,
    evaluate_temporal_inference,
    subjectbreak_mappings,
    summarize_matching,
    temporal_contrasts,
    temporal_permutation_indices,
    terminal_decision,
)


def test_exact_s5_and_derangement_contract() -> None:
    assert ALL_PERMUTATIONS.shape == (120, 5)
    assert len({tuple(value) for value in ALL_PERMUTATIONS.tolist()}) == 120
    assert IDENTITY_INDEX == 0
    assert np.array_equal(ALL_PERMUTATIONS[0], np.arange(5))
    assert DERANGEMENT_INDICES.shape == (44,)
    assert DERANGEMENT_MASK.sum() == 44
    assert np.all(
        ALL_PERMUTATIONS[DERANGEMENT_INDICES] != np.arange(5)[None, :]
    )
    for relabel_index in (0, 1, 37, 119):
        for match_index in (0, 2, 81, 119):
            expected = ALL_PERMUTATIONS[relabel_index][
                ALL_PERMUTATIONS[match_index]
            ]
            assert np.array_equal(
                ALL_PERMUTATIONS[COMPOSITION_INDICES[relabel_index, match_index]],
                expected,
            )


def test_known_diagonal_k_has_positive_gain_and_best_identity_rank() -> None:
    k = np.full((N_CELLS, N_CELLS, 5, 5), 3.0)
    diagonal = np.arange(5)
    k[..., diagonal, diagonal] = 1.0
    result = summarize_matching(k)
    assert result.d_id == pytest.approx(np.ones((N_CELLS, N_CELLS)))
    assert result.median_derangement == pytest.approx(
        np.full((N_CELLS, N_CELLS), 3.0)
    )
    assert result.gain == pytest.approx(np.full((N_CELLS, N_CELLS), 2.0))
    assert np.array_equal(result.identity_rank, np.ones((N_CELLS, N_CELLS)))


def test_relabelled_gain_equals_direct_column_relabel_computation() -> None:
    rng = np.random.default_rng(20260811)
    k = rng.uniform(0.1, 3.0, size=(N_CELLS, N_CELLS, 5, 5))
    result = summarize_matching(k)
    for permutation_index in (0, 1, 37, 119):
        relabelled = np.take(k, ALL_PERMUTATIONS[permutation_index], axis=-1)
        direct = summarize_matching(relabelled)
        assert result.relabelled_gain[..., permutation_index] == pytest.approx(
            direct.gain, abs=0.0, rel=0.0
        )


def _synthetic_gain(
    subject_effect: float,
    class_effect: float,
    interaction_effect: float,
    baseline: float = 0.2,
) -> np.ndarray:
    result = np.empty((36, 36), dtype=np.float64)
    for row in range(36):
        subject, class_index = divmod(row, 4)
        for column in range(36):
            other_subject, other_class = divmod(column, 4)
            result[row, column] = (
                baseline
                + subject_effect * (subject == other_subject)
                + class_effect * (class_index == other_class)
                + interaction_effect
                * (subject == other_subject and class_index == other_class)
            )
    return result


def test_temporal_specificity_and_interaction_contrast_signs() -> None:
    result = temporal_contrasts(_synthetic_gain(0.7, 0.4, 0.3))
    assert result.t_temporal == pytest.approx(1.6)
    assert result.t_subject == pytest.approx(1.0)
    assert result.t_class == pytest.approx(0.7)
    assert result.t_j == pytest.approx(0.3)
    assert result.s_s == pytest.approx(np.full(9, 1.0))
    assert result.c_s == pytest.approx(np.full(9, 0.7))
    assert result.j_s == pytest.approx(np.full(9, 0.3))


def test_subject_and_class_break_streams_exactly_reuse_prior_frozen_mappings() -> None:
    assert np.array_equal(classbreak_mappings(replicates=50), prior_classbreak_mappings(replicates=50))
    assert np.array_equal(
        subjectbreak_mappings(replicates=50), prior_subjectbreak_mappings(replicates=50)
    )
    first = temporal_permutation_indices(replicates=50)
    second = temporal_permutation_indices(replicates=50)
    assert np.array_equal(first, second)
    assert first.shape == (50, 36)
    assert np.all((0 <= first) & (first < 120))


def test_known_shared_temporal_sequence_passes_temporal_not_specificity() -> None:
    k = np.full((N_CELLS, N_CELLS, 5, 5), 3.0)
    diagonal = np.arange(5)
    k[..., diagonal, diagonal] = 1.0
    inference = evaluate_temporal_inference(summarize_matching(k), replicates=199)
    assert inference.observed.t_temporal == pytest.approx(2.0)
    assert inference.p_temporal < 0.05
    assert inference.observed.t_subject == pytest.approx(0.0, abs=1.0e-14)
    assert inference.observed.t_class == pytest.approx(0.0, abs=1.0e-14)
    assert inference.terminal == "GO_SHARED_TEMPORAL_SEQUENCE_ONLY"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ((0.1, 0.049, 0.1, 0.049, 0.1, 0.049), "GO_REPRODUCIBLE_SUBJECT_CLASS_TEMPORAL_SEQUENCE"),
        ((0.1, 0.049, 0.1, 0.049, 0.1, 0.05), "GO_SHARED_TEMPORAL_SEQUENCE_WITH_PARTIAL_SPECIFICITY"),
        ((0.1, 0.049, 0.1, 0.05, 0.1, 0.05), "GO_SHARED_TEMPORAL_SEQUENCE_ONLY"),
        ((0.1, 0.05, 0.1, 0.001, 0.1, 0.001), "STOP_NO_REPRODUCIBLE_TEMPORAL_SEQUENCE_V0"),
        ((-0.1, 0.001, 0.1, 0.001, 0.1, 0.001), "STOP_NO_REPRODUCIBLE_TEMPORAL_SEQUENCE_V0"),
    ],
)
def test_exact_terminal_mapping(arguments: tuple[float, ...], expected: str) -> None:
    assert terminal_decision(
        t_temporal=arguments[0],
        p_temporal=arguments[1],
        t_subject=arguments[2],
        p_subject=arguments[3],
        t_class=arguments[4],
        p_class=arguments[5],
    ) == expected

