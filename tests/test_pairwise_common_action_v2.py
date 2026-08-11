"""Pre-data tests for the optimizer-only pairwise V2 amendment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.common_action_optimizer_audit_v2 import synthetic_stress_fixtures
from src.common_action_solver_v0 import (
    action_gradient,
    action_hessian_vector,
    build_pymanopt_trust_regions_optimizer,
    conjugate,
    deterministic_starts,
)
from src.pairwise_common_action_pipeline_v2 import load_config, runtime_contract
from src.pairwise_common_action_v2 import (
    PAIRWISE_V2_SETTINGS,
    fit_pairwise_action_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_config_changes_only_optimizer_and_versioned_provenance() -> None:
    v1 = yaml.safe_load(
        (ROOT / "configs/bnci2014_001_pairwise_common_action_v1.yaml").read_text()
    )
    v2 = yaml.safe_load(
        (ROOT / "configs/bnci2014_001_pairwise_common_action_v2.yaml").read_text()
    )
    for section in (
        "dataset",
        "reproduction_gate",
        "scientific_model",
        "identifiability",
        "prediction",
        "aggregation",
        "stage_A",
        "stage_B",
        "nulls",
        "global_consistency",
        "terminal_logic",
        "claim_restrictions",
    ):
        assert v2[section] == v1[section]
    assert v1["optimizer"]["optimizer_class"].endswith("ConjugateGradient")
    assert v2["optimizer"]["optimizer_class"].endswith("TrustRegions")
    assert v2["project"]["output_dir"].endswith("pairwise_common_action_v2")
    assert v2["project"]["v1_cache_reuse_allowed"] is False


def test_runtime_optimizer_matches_every_explicit_trust_regions_setting() -> None:
    contract = runtime_contract(ROOT)
    config = contract["config"]["optimizer"]
    runtime = contract["optimizer_parameters"]
    assert contract["optimizer_class"] == "TrustRegions"
    assert runtime["miniter"] == config["trust_regions"]["miniter"] == 3
    assert runtime["kappa"] == config["trust_regions"]["kappa"] == 0.1
    assert runtime["theta"] == config["trust_regions"]["theta"] == 1.0
    assert runtime["rho_prime"] == config["trust_regions"]["rho_prime"] == 0.1
    assert runtime["use_rand"] is config["trust_regions"]["use_rand"] is False
    assert runtime["rho_regularization"] == 1000.0
    assert runtime["max_iterations"] == config["max_iterations"] == 1000
    assert runtime["min_gradient_norm"] == config["min_gradient_norm"] == 1.0e-5
    assert runtime["min_step_size"] == config["min_step_size"] == 1.0e-12
    assert runtime["max_cost_evaluations"] == config["max_cost_evaluations"] == 1001
    assert runtime["max_time"] == config["max_time_seconds"] == 3600.0
    assert runtime["log_verbosity"] == config["log_verbosity"] == 0
    for key in ("run_mininner", "run_maxinner", "run_Delta_bar", "run_Delta0"):
        assert runtime[key] == config["trust_regions"][key]


def test_checkpoint_identity_has_all_required_v2_keys() -> None:
    contract = runtime_contract(ROOT)
    identity = contract["checkpoint_identity"]
    assert set(identity) == {
        "config_sha256",
        "source_sha256",
        "optimizer_identity",
        "protocol_amendment_sha",
    }
    assert identity["optimizer_identity"] == "pymanopt_2.2.1_stiefel_trust_regions"
    assert all(identity.values())


def test_production_hessian_vector_matches_gradient_finite_difference() -> None:
    fixture = synthetic_stress_fixtures()[0]
    action = deterministic_starts(
        fixture.fit_targets,
        fixture.fit_templates,
        seed=fixture.seed,
        count=4,
    )[0]
    raw = np.random.default_rng(884).normal(size=action.shape)
    direction = action @ (0.5 * (raw - raw.T))
    epsilon = 1.0e-7
    numerical = (
        action_gradient(
            fixture.fit_targets,
            fixture.fit_templates,
            action + epsilon * direction,
        )
        - action_gradient(
            fixture.fit_targets,
            fixture.fit_templates,
            action - epsilon * direction,
        )
    ) / (2.0 * epsilon)
    analytic = action_hessian_vector(
        fixture.fit_targets,
        fixture.fit_templates,
        action,
        direction,
    )
    assert np.linalg.norm(numerical - analytic) / np.linalg.norm(analytic) < 2.0e-8


@pytest.mark.parametrize("fixture_index", [0, 1])
def test_v2_exact_d22_recovery_and_four_start_sector_contract(
    fixture_index: int,
) -> None:
    fixture = synthetic_stress_fixtures()[fixture_index]
    result = fit_pairwise_action_v2(
        fixture.fit_targets,
        fixture.fit_templates,
        seed_parts=("v2_exact_test", fixture.name),
    )
    assert PAIRWISE_V2_SETTINGS.starts == 4
    assert len(result.fit.starts) == 4
    assert result.initial_determinants.count(-1) == 2
    assert result.initial_determinants.count(1) == 2
    assert all(value.converged for value in result.fit.starts)
    assert {
        value.optimizer for value in result.fit.starts
    } == {"pymanopt_2.2.1_stiefel_trust_regions"}
    assert {
        int(np.sign(value.determinant)) for value in result.fit.starts
    } == {-1, 1}
    prediction = conjugate(result.fit.matrix, fixture.heldout_template)
    relative = np.linalg.norm(prediction - fixture.heldout_truth) / np.linalg.norm(
        fixture.heldout_truth
    )
    assert relative <= 1.0e-8


def test_builder_does_not_expose_a_line_search_or_random_inner_start() -> None:
    optimizer = build_pymanopt_trust_regions_optimizer(PAIRWISE_V2_SETTINGS)
    assert not hasattr(optimizer, "_line_searcher")
    assert optimizer.use_rand is False
