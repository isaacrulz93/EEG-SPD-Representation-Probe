"""Optimizer-only V2 runtime for the frozen pairwise common-action design.

All scientific/statistical primitives are re-exported from V1 unchanged. The
only implementation change is the synthetically audited Pymanopt TrustRegions
single-action optimizer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from src.common_action_solver_v0 import (
    ActionSolverError,
    SolverSettings,
    deterministic_starts,
    diagnose_multistart,
    optimize_action,
)
from src.pairwise_common_action_v1 import (
    MASTER_SEED,
    NULL_REPLICATES,
    PairwiseAssessment,
    PairwiseContractError,
    PairwiseFit,
    StageGate,
    aggregate_pairwise_gains,
    assess_pairwise_prediction,
    deterministic_seed,
    evaluate_stage_gate,
    pairwise_error_gain,
    semantic_null_statistics,
    terminal_decision,
    unrelated_target_gain_bank,
    unrelated_target_null_statistics,
)
from src.pairwise_common_action_v1 import PAIRWISE_SETTINGS as V1_SETTINGS


PAIRWISE_V2_SETTINGS = replace(
    V1_SETTINGS,
    starts=4,
    pymanopt_log_verbosity=0,
)


def fit_pairwise_action_v2(
    target_fit: np.ndarray,
    source_fit: np.ndarray,
    *,
    seed_parts: Sequence[object],
    settings: SolverSettings = PAIRWISE_V2_SETTINGS,
) -> PairwiseFit:
    """Fit exactly four audited TrustRegions starts over both O(d) sectors."""

    if int(settings.starts) != 4:
        raise PairwiseContractError("pairwise V2 requires exactly four total starts")
    seed = deterministic_seed("pairwise_action", *seed_parts)
    starts = deterministic_starts(
        target_fit,
        source_fit,
        seed=seed,
        count=settings.starts,
    )
    if len(starts) != settings.starts:
        raise PairwiseContractError(
            f"actual_total_starts={len(starts)} != config.total_starts={settings.starts}"
        )
    initial_determinants = tuple(
        int(np.sign(np.linalg.det(value))) for value in starts
    )
    if initial_determinants.count(-1) != 2 or initial_determinants.count(1) != 2:
        raise PairwiseContractError(
            "four-start construction must contain exactly two starts per determinant sector"
        )
    fit = optimize_action(
        target_fit,
        source_fit,
        seed=seed,
        settings=settings,
        starts=starts,
        solver="pymanopt_trust_regions",
    )
    if len(fit.starts) != settings.starts:
        raise PairwiseContractError(
            f"runtime optimizer count {len(fit.starts)} != frozen total {settings.starts}"
        )
    landscape = diagnose_multistart(fit)
    if landscape.determinant_sectors_with_converged_solution != (-1, 1):
        diagnostics = "; ".join(
            (
                f"start={value.start_index},"
                f"initial_det={initial_determinants[value.start_index]},"
                f"final_det={int(np.sign(value.determinant))},"
                f"converged={value.converged},objective={value.objective:.17g},"
                f"gradient={value.gradient_norm:.17g},iterations={value.iterations},"
                f"stopping={value.stopping_criterion}"
            )
            for value in fit.starts
        )
        raise ActionSolverError(
            "UNASSESSED_TECHNICAL_FAILURE: pairwise V2 fit lacks a converged "
            f"determinant sector; seed_parts={tuple(seed_parts)!r}; {diagnostics}"
        )
    return PairwiseFit(fit, starts, initial_determinants)


__all__ = [
    "MASTER_SEED",
    "NULL_REPLICATES",
    "PAIRWISE_V2_SETTINGS",
    "PairwiseAssessment",
    "PairwiseContractError",
    "PairwiseFit",
    "StageGate",
    "aggregate_pairwise_gains",
    "assess_pairwise_prediction",
    "deterministic_seed",
    "evaluate_stage_gate",
    "fit_pairwise_action_v2",
    "pairwise_error_gain",
    "semantic_null_statistics",
    "terminal_decision",
    "unrelated_target_gain_bank",
    "unrelated_target_null_statistics",
]
