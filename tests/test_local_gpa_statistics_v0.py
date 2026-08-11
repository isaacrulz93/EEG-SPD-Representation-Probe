from __future__ import annotations

import numpy as np
import pytest

from src.local_gpa_statistics_v0 import (
    GO_DECISION,
    STOP_DECISION,
    evaluate_consensus_interaction,
    terminal_decision,
)
from src.local_metric_interaction_v0 import synthetic_additive_cell_matrix


def test_stage1_correspondence_nulls_are_reused_without_gpa_refits() -> None:
    additive = synthetic_additive_cell_matrix(
        subject_effect=0.7, class_effect=0.4
    )
    result = evaluate_consensus_interaction(additive)
    assert result.observed.t_j == pytest.approx(0.0, abs=1.0e-14)
    assert result.p_j_classbreak >= 0.05
    assert result.p_j_subjectbreak >= 0.05


def test_subject_is_the_final_group_unit() -> None:
    interacting = synthetic_additive_cell_matrix(
        subject_effect=0.7, class_effect=0.4, interaction_effect=0.3
    )
    result = evaluate_consensus_interaction(interacting, replicates=19)
    assert result.observed.j_s.shape == (9,)
    assert result.observed.j_sc.shape == (9, 4)
    assert result.observed.t_j == pytest.approx(np.mean(result.observed.j_s))


def test_frozen_terminal_mapping_requires_positive_and_both_nulls() -> None:
    assert terminal_decision(
        t_j=0.01, p_classbreak=0.049, p_subjectbreak=0.049
    ) == GO_DECISION
    assert terminal_decision(
        t_j=0.01, p_classbreak=0.05, p_subjectbreak=0.001
    ) == STOP_DECISION
    assert terminal_decision(
        t_j=-0.01, p_classbreak=0.001, p_subjectbreak=0.001
    ) == STOP_DECISION
