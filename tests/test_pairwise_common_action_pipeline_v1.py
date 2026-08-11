"""Pre-data execution-contract tests for pairwise common action V1."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common_action_solver_v0 import build_pymanopt_optimizer, conjugate
from src.pairwise_common_action_v1 import PAIRWISE_SETTINGS
from src.pairwise_common_action_pipeline_v1 import (
    _cross_session_stage,
    _fit_true_pair,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _orthogonal(seed: int, dimension: int, determinant: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    if int(np.sign(np.linalg.det(q))) != int(determinant):
        q[:, 0] *= -1.0
    return q


def _synthetic_arrays(dimension: int = 4) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(20260811)
    templates = rng.normal(size=(2, 4, dimension, dimension))
    templates = 0.5 * (templates + templates.transpose(0, 1, 3, 2))
    actions = np.stack(
        [
            _orthogonal(100 + subject, dimension, -1 if subject % 2 else 1)
            for subject in range(9)
        ]
    )
    full = np.stack([conjugate(action, templates) for action in actions])
    arrays = {"F": full, "A": full.copy(), "B": full.copy()}
    return arrays, actions


def test_frozen_config_matches_runtime_contract() -> None:
    config = load_config(ROOT)
    assert config["optimizer"]["total_starts"] == 4
    assert config["aggregation"]["inferential_unit"] == "target_subject"
    assert config["stage_A"]["required_pairs"] == 576
    assert config["runtime"]["primary_pymanopt_runs"] == 6912
    assert config["runtime"]["maximum_stage_A_pymanopt_runs"] == 18432
    assert config["global_consistency"]["profiled_product_model"]["automatic_execution"] is False
    assert config["global_consistency"]["old_nested_alternating_implementation_forbidden"] is True


def test_pairwise_optimizer_runtime_matches_amendment_config() -> None:
    config = load_config(ROOT)["optimizer"]
    optimizer = build_pymanopt_optimizer(PAIRWISE_SETTINGS)
    line_search = optimizer._line_searcher
    assert type(optimizer).__name__ == "ConjugateGradient"
    assert optimizer._beta_rule == config["beta_rule"]
    assert optimizer._log_verbosity == config["log_verbosity"] == 1
    assert optimizer._max_iterations == config["max_iterations"]
    assert optimizer._min_gradient_norm == config["min_gradient_norm"]
    assert optimizer._min_step_size == config["min_step_size"]
    assert optimizer._max_cost_evaluations == config["max_cost_evaluations"]
    assert type(line_search).__name__ == "BackTrackingLineSearcher"
    for runtime_name, config_name in (
        ("contraction_factor", "contraction_factor"),
        ("optimism", "optimism"),
        ("sufficient_decrease", "sufficient_decrease"),
        ("max_iterations", "max_iterations"),
        ("initial_step_size", "initial_step_size"),
    ):
        assert getattr(line_search, runtime_name) == config["line_search"][config_name]


def test_one_true_pair_recomputes_full_and_both_halves_without_leakage() -> None:
    arrays, _ = _synthetic_arrays()
    result = _fit_true_pair(
        arrays,
        target=0,
        source=1,
        session=0,
        heldout=3,
    )
    assert result["row"]["target"] == 1
    assert result["row"]["source"] == 2
    assert result["row"]["heldout_class"] == "tongue"
    assert result["row"]["action_error"] < 1.0e-16
    assert result["row"]["D_eq"] <= result["row"]["D_threshold"]
    assert len(result["solver_rows"]) == 3 * 4
    assert {row["split"] for row in result["solver_rows"]} == {"F", "A", "B"}
    assert all(row["actual_total_starts"] == 4 for row in result["solver_rows"])


def test_objective_plateau_rule_records_both_determinant_sectors() -> None:
    arrays, _ = _synthetic_arrays()
    result = _fit_true_pair(
        arrays,
        target=6,
        source=8,
        session=0,
        heldout=1,
    )
    full_rows = [row for row in result["solver_rows"] if row["split"] == "F"]
    assert {row["initial_determinant"] for row in full_rows} == {-1, 1}
    assert {
        int(np.sign(row["final_determinant"]))
        for row in full_rows
        if row["converged"]
    } == {-1, 1}
    assert any(
        "objective_stall_with_bounded_gradient" in row["stopping_criterion"]
        for row in full_rows
    )


def test_cross_session_reuses_stage_A_actions_and_keeps_576_pair_grid() -> None:
    arrays, subject_actions = _synthetic_arrays()
    d = arrays["F"].shape[-1]
    best = np.full((9, 9, 2, 4, d, d), np.nan)
    eq_bank = {}
    for target in range(9):
        for source in range(9):
            if source == target:
                continue
            relative = subject_actions[target] @ subject_actions[source].T
            best[target, source] = relative
            for session in range(2):
                for heldout in range(4):
                    eq_bank[(target, source, session, heldout)] = np.asarray([relative])
    primary = {
        "best_actions": best,
        "half_A_actions": best.copy(),
        "half_B_actions": best.copy(),
        "eq_bank": eq_bank,
    }
    cross = _cross_session_stage(arrays, primary)
    assert len(cross["table"]) == 576
    assert cross["identifiable"] is True
    assert np.nanmax(np.abs(cross["gains"] - np.where(np.isnan(cross["gains"]), np.nan, cross["gains"]))) == 0.0
    assert float(cross["table"]["action_error"].max()) < 1.0e-16
