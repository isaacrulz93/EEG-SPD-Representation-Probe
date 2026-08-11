from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from src.local_gpa_outer_convergence_audit_v2 import (
    AUDIT_MAX_OUTER_ITERATIONS,
    FROZEN_OUTER_ITERATIONS,
    PROTOTYPE_INITIAL_STEP_CANDIDATES,
    run_gpa_start_trace,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "bnci2014_001_local_gpa_outer_convergence_audit_v2"


def _configuration(seed: int, d: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logs = rng.normal(scale=0.05, size=(5, d, d))
    logs = 0.5 * (logs + logs.transpose(0, 2, 1))
    logs -= logs.mean(axis=0)
    return np.stack([expm(value) for value in logs])


def test_audit_limits_and_initial_step_candidates_are_exactly_frozen() -> None:
    assert FROZEN_OUTER_ITERATIONS == 24
    assert AUDIT_MAX_OUTER_ITERATIONS == 96
    assert PROTOTYPE_INITIAL_STEP_CANDIDATES == (1.0, 2.0, 4.0)
    manifest = json.loads(
        (OUTPUT / "protocol" / "prototype_initial_step_candidates.json").read_text()
    )
    assert tuple(manifest["candidate_initial_line_search_steps"]) == (1.0, 2.0, 4.0)
    assert manifest["unchanged_prototype_inner_iterations"] == 16
    assert manifest["unchanged_gpa_gradient_tolerance"] == 2.0e-5


def test_outcome_blind_bank_indices_reproduce_from_declared_seed() -> None:
    manifest = json.loads(
        (OUTPUT / "protocol" / "audit_bank_selection.json").read_text()
    )
    rng = np.random.default_rng(np.random.SeedSequence([20260811, 0x47504132]))
    selected = sorted(rng.choice(np.arange(1, 216), size=4, replace=False).tolist())
    assert selected == [21, 29, 157, 182]
    assert selected == manifest["selected_task_indices_zero_based"]
    assert manifest["scientific_statistics_computed"] is False


def test_start_trace_uses_original_convergence_gates_and_records_backtracking() -> None:
    base = _configuration(801)
    trials = np.stack([base, base, base, base])
    result = run_gpa_start_trace(
        trials,
        identity_parts=("synthetic", "identical"),
        start_index=0,
        max_outer_iterations=24,
        prototype_initial_step_size=2.0,
    )
    assert result.converged
    assert result.first_convergence_outer_iteration is not None
    assert result.first_convergence_outer_iteration <= 24
    assert result.prototype_initial_step_size == 2.0
    assert result.initial_trial_position == 0
    assert all(value.prototype_inner_iterations <= 16 for value in result.traces)
    assert all(
        len(value.prototype_accepted_step_sizes) == value.prototype_inner_iterations
        for value in result.traces
    )
    assert all(
        len(value.prototype_backtracking_reductions) == value.prototype_inner_iterations
        for value in result.traces
    )
    final = result.traces[-1]
    assert final.projected_prototype_gradient_norm <= 2.0e-5
    assert final.relative_objective_change <= 1.0e-7
    assert final.convergence_boolean


def test_unregistered_initial_step_or_outer_limit_is_rejected() -> None:
    trials = np.stack([_configuration(802), _configuration(803)])
    with pytest.raises(ValueError, match="frozen"):
        run_gpa_start_trace(
            trials,
            identity_parts=("synthetic",),
            start_index=0,
            prototype_initial_step_size=3.0,
        )
    with pytest.raises(ValueError, match="capped"):
        run_gpa_start_trace(
            trials,
            identity_parts=("synthetic",),
            start_index=0,
            max_outer_iterations=97,
        )


def test_audit_module_does_not_import_scientific_statistics_or_reporting() -> None:
    source = (ROOT / "src" / "local_gpa_outer_convergence_audit_v2.py").read_text()
    assert "local_gpa_statistics" not in source
    assert "reporting_local_gpa" not in source
    assert "compute_consensus_distances" not in source
    assert "evaluate_consensus_interaction" not in source
