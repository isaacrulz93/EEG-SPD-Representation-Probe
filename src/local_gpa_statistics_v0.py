"""Frozen Stage-1-compatible cell statistics for quotient GPA consensus V0."""

from __future__ import annotations

from src.local_metric_interaction_v0 import (
    DEFAULT_MASTER_SEED,
    DEFAULT_NULL_REPLICATES,
    InteractionNullResult,
    evaluate_interaction_nulls,
)


GO_DECISION = "GO_STABLE_SUBJECT_CLASS_QUOTIENT_MEAN_CONFIGURATION"
STOP_DECISION = "STOP_NO_STABLE_QUOTIENT_MEAN_CONFIGURATION_INTERACTION"
GPA_FAILURE = "UNASSESSED_GPA_NUMERICAL_FAILURE"
TECHNICAL_FAILURE = "UNASSESSED_TECHNICAL_FAILURE"


def evaluate_consensus_interaction(
    m01,
    *,
    replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> InteractionNullResult:
    return evaluate_interaction_nulls(
        m01, replicates=replicates, master_seed=master_seed
    )


def terminal_decision(
    *, t_j: float, p_classbreak: float, p_subjectbreak: float
) -> str:
    if t_j > 0.0 and p_classbreak < 0.05 and p_subjectbreak < 0.05:
        return GO_DECISION
    return STOP_DECISION


__all__ = [
    "GO_DECISION",
    "GPA_FAILURE",
    "STOP_DECISION",
    "TECHNICAL_FAILURE",
    "evaluate_consensus_interaction",
    "terminal_decision",
]
