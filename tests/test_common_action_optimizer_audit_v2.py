"""Synthetic-only tests for the optimizer failure audit V2."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.common_action_optimizer_audit_v2 import (
    AUDIT_SETTINGS,
    action_hessian_vector,
    run_one_start,
    synthetic_stress_fixtures,
)
from src.common_action_solver_v0 import (
    action_gradient,
    deterministic_starts,
)


def test_stress_manifest_is_exact_d22_twelve_case_grid() -> None:
    fixtures = synthetic_stress_fixtures()
    assert len(fixtures) == 12
    assert {fixture.fit_targets.shape for fixture in fixtures} == {(3, 22, 22)}
    assert {fixture.truth_determinant for fixture in fixtures} == {-1, 1}
    assert {fixture.family for fixture in fixtures} == {
        "generic_exact",
        "generic_noisy",
        "nearly_commuting",
        "clustered_spectrum",
        "approximate_stabilizer",
        "ill_conditioned",
    }


def test_analytic_hessian_vector_matches_gradient_finite_difference() -> None:
    fixture = synthetic_stress_fixtures()[0]
    action = deterministic_starts(
        fixture.fit_targets,
        fixture.fit_templates,
        seed=fixture.seed,
        count=4,
    )[0]
    raw = np.random.default_rng(55).normal(size=action.shape)
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
    relative = np.linalg.norm(numerical - analytic) / np.linalg.norm(analytic)
    assert relative < 2.0e-8


def test_identical_four_starts_cover_both_determinant_sectors() -> None:
    fixture = synthetic_stress_fixtures()[2]
    starts = deterministic_starts(
        fixture.fit_targets,
        fixture.fit_templates,
        seed=fixture.seed,
        count=AUDIT_SETTINGS.starts,
    )
    assert len(starts) == 4
    determinants = [int(np.sign(np.linalg.det(start))) for start in starts]
    assert determinants.count(-1) == determinants.count(1) == 2


def test_all_standard_optimizers_preserve_orthogonality_and_sector_on_fixture() -> None:
    fixture = synthetic_stress_fixtures()[0]
    starts = deterministic_starts(
        fixture.fit_targets,
        fixture.fit_templates,
        seed=fixture.seed,
        count=4,
    )
    settings = replace(AUDIT_SETTINGS, max_iterations=100)
    for optimizer in ("conjugate_gradient", "trust_regions", "steepest_descent"):
        result = run_one_start(
            fixture,
            optimizer,
            starts[0],
            0,
            settings=settings,
        )
        assert result.determinant_preserved
        assert result.maximum_orthogonality_error < 1.0e-10
        assert np.isfinite(result.final_objective)
        assert np.isfinite(result.final_gradient_norm)


def test_audit_module_has_no_real_data_loader() -> None:
    import inspect
    import src.common_action_optimizer_audit_v2 as module

    source = inspect.getsource(module)
    assert "np.load(" not in source
    assert "load_and_reproduce_U" not in source
    assert "stage_A_prediction" not in source
