"""Synthetic-only tests for the post-failure formulation audit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from src.common_action_formulation_audit_v1 import (
    bnci_stage_a_call_budget,
    equivalence_aware_cycle_diagnostic,
    fit_joint_latent_reference,
    fit_pairwise_action,
    fit_profiled_product_reference,
    latent_least_squares_objective,
    profiled_euclidean_gradient,
    profiled_objective,
    profiled_pairwise_identity,
    profiled_templates,
)
from src.common_action_solver_v0 import (
    CANDIDATE_SOLVER_SETTINGS,
    ActionSolverError,
    analyze_common_stabilizer,
    conjugate,
    optimize_action,
)


def _orthogonal(seed: int, dimension: int, determinant: int = 1) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    if int(np.sign(np.linalg.det(q))) != int(determinant):
        q[:, 0] *= -1.0
    return q


def _bank(seed: int, count: int, dimension: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=(count, dimension, dimension))
    return 0.5 * (values + values.transpose(0, 2, 1))


def _latent_fixture(
    *,
    seed: int,
    subjects: int,
    classes: int,
    dimension: int,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    templates = _bank(seed, classes, dimension)
    actions = np.empty((subjects, dimension, dimension))
    actions[0] = np.eye(dimension)
    for subject in range(1, subjects):
        determinant = -1 if subject % 2 else 1
        actions[subject] = _orthogonal(seed + 10 * subject, dimension, determinant)
    objects = np.stack([conjugate(action, templates) for action in actions])
    if noise:
        rng = np.random.default_rng(seed + 999)
        perturbation = rng.normal(scale=noise, size=objects.shape)
        objects = objects + 0.5 * (perturbation + perturbation.transpose(0, 1, 3, 2))
    return objects, templates, actions


def test_profiled_template_is_exact_minimizer_and_A_equals_B() -> None:
    objects, _, _ = _latent_fixture(seed=10, subjects=4, classes=4, dimension=5, noise=0.02)
    actions = np.stack([np.eye(5), _orthogonal(21, 5, -1), _orthogonal(22, 5, 1), _orthogonal(23, 5, -1)])
    optimum = profiled_templates(objects, actions)
    value_a = latent_least_squares_objective(objects, actions, optimum)
    value_b = profiled_objective(objects, actions)
    value_pairwise = profiled_pairwise_identity(objects, actions)
    assert value_a == pytest.approx(value_b, rel=2e-15, abs=2e-15)
    assert value_pairwise == pytest.approx(value_b, rel=2e-15, abs=2e-15)
    perturbation = _bank(24, 4, 5)
    perturbed = latent_least_squares_objective(objects, actions, optimum + perturbation)
    pythagorean = value_a + len(objects) * float(np.sum(perturbation * perturbation))
    assert perturbed == pytest.approx(pythagorean, rel=2e-15, abs=2e-12)


def test_profiled_gradient_matches_tangent_directional_difference() -> None:
    objects, _, actions = _latent_fixture(seed=30, subjects=3, classes=4, dimension=4, noise=0.01)
    gradients = profiled_euclidean_gradient(objects, actions)
    subject = 1
    raw = np.random.default_rng(31).normal(size=(4, 4))
    omega = 0.5 * (raw - raw.T)
    direction = actions[subject] @ omega
    epsilon = 1.0e-7
    plus = actions.copy()
    minus = actions.copy()
    plus[subject] = actions[subject] @ expm(epsilon * omega)
    minus[subject] = actions[subject] @ expm(-epsilon * omega)
    numerical = (profiled_objective(objects, plus) - profiled_objective(objects, minus)) / (2.0 * epsilon)
    analytic = float(np.sum(gradients[subject] * direction))
    assert numerical == pytest.approx(analytic, rel=3e-7, abs=3e-7)


def test_small_joint_A_and_profiled_B_share_exact_noiseless_optimum() -> None:
    objects, templates, truth = _latent_fixture(seed=40, subjects=3, classes=4, dimension=3)
    settings = replace(CANDIDATE_SOLVER_SETTINGS, starts=4, gradient_tolerance=1e-7)
    joint = fit_joint_latent_reference(
        objects,
        settings=settings,
        initial_actions=truth,
    )
    profiled = fit_profiled_product_reference(
        objects,
        settings=settings,
        initial_actions=truth,
    )
    assert joint.objective < 1e-20
    assert profiled.objective < 1e-20
    assert joint.objective == pytest.approx(profiled.objective, abs=1e-20)
    np.testing.assert_allclose(joint.templates, templates, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(profiled.templates, templates, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("truth_determinant", [1, -1])
def test_pairwise_LOCO_is_exact_necessary_condition_in_latent_fixture(truth_determinant: int) -> None:
    dimension = 6
    templates = _bank(50 + truth_determinant, 4, dimension)
    source_action = _orthogonal(60, dimension, -1)
    relative = _orthogonal(61 + truth_determinant, dimension, truth_determinant)
    target_action = relative @ source_action
    source = conjugate(source_action, templates)
    target = conjugate(target_action, templates)
    settings = replace(CANDIDATE_SOLVER_SETTINGS, starts=4)
    for heldout in range(4):
        fit_classes = [index for index in range(4) if index != heldout]
        fit = fit_pairwise_action(
            target[fit_classes],
            source[fit_classes],
            seed=70 + heldout,
            settings=settings,
        )
        prediction = conjugate(fit.matrix, source[heldout])
        relative_error = np.linalg.norm(prediction - target[heldout]) / np.linalg.norm(target[heldout])
        assert relative_error < 1.0e-8
        assert len(fit.starts) == settings.starts


def test_pairwise_gate_detects_a_heldout_violation_of_common_action() -> None:
    source = _bank(80, 4, 5)
    relative = _orthogonal(81, 5, -1)
    target = conjugate(relative, source)
    target[3] = 1.5 * target[3]
    settings = replace(CANDIDATE_SOLVER_SETTINGS, starts=4)
    fit = fit_pairwise_action(target[:3], source[:3], seed=82, settings=settings)
    fit_relative = np.linalg.norm(conjugate(fit.matrix, source[:3]) - target[:3]) / np.linalg.norm(target[:3])
    heldout_relative = np.linalg.norm(conjugate(fit.matrix, source[3]) - target[3]) / np.linalg.norm(target[3])
    assert fit_relative < 1.0e-8
    assert heldout_relative > 0.30


def test_equivalence_aware_cycle_uses_induced_actions_not_raw_Q() -> None:
    dimension = 5
    q_s = _orthogonal(90, dimension, -1)
    q_r = _orthogonal(91, dimension, 1)
    q_t = _orthogonal(92, dimension, -1)
    direct = q_s @ q_t.T
    first = q_s @ q_r.T
    second = q_r @ q_t.T
    probes = _bank(93, 4, dimension)
    harmless = equivalence_aware_cycle_diagnostic(
        (direct, -direct),
        (first, -first),
        (second, -second),
        probes,
    )
    assert harmless.relative_discrepancy < 1.0e-14
    inconsistent = equivalence_aware_cycle_diagnostic(
        (_orthogonal(94, dimension, 1),),
        (first,),
        (second,),
        probes,
    )
    assert inconsistent.relative_discrepancy > 0.1


def test_commutator_diagnostic_distinguishes_generic_and_continuous_stabilizer() -> None:
    generic = analyze_common_stabilizer(_bank(100, 3, 7))
    assert generic.numerical_nullity == 0
    repeated = np.stack(
        [
            np.diag([0.1, 0.1, -0.4, 0.7]),
            np.diag([-0.3, -0.3, 0.2, 0.9]),
            np.diag([0.8, 0.8, -0.6, 0.4]),
        ]
    )
    continuous = analyze_common_stabilizer(repeated)
    assert continuous.numerical_nullity == 1
    assert continuous.approximate_nullity == 1
    assert continuous.singular_values[-1] <= continuous.numerical_tolerance


def test_total_start_count_contract_rejects_hidden_warm_start() -> None:
    templates = _bank(110, 3, 4)
    settings = replace(CANDIDATE_SOLVER_SETTINGS, starts=4)
    with pytest.raises(ActionSolverError, match="expected exactly 4 total starts, received 5"):
        optimize_action(
            templates,
            templates,
            seed=111,
            settings=settings,
            starts=(np.eye(4),) * 5,
        )


@pytest.mark.parametrize("truth_determinant", [1, -1])
def test_four_total_starts_recover_d22_exact_and_noisy_fixtures(truth_determinant: int) -> None:
    templates = _bank(120 + truth_determinant, 4, 22)
    truth = _orthogonal(130 + truth_determinant, 22, truth_determinant)
    rng = np.random.default_rng(140 + truth_determinant)
    noise = rng.normal(scale=2.0e-4, size=(3, 22, 22))
    noise = 0.5 * (noise + noise.transpose(0, 2, 1))
    targets = conjugate(truth, templates[:3]) + noise
    settings = replace(CANDIDATE_SOLVER_SETTINGS, starts=4)
    fit = optimize_action(targets, templates[:3], seed=150 + truth_determinant, settings=settings)
    prediction = conjugate(fit.matrix, templates[3])
    expected = conjugate(truth, templates[3])
    relative_error = np.linalg.norm(prediction - expected) / np.linalg.norm(expected)
    assert len(fit.starts) == 4
    assert {int(np.sign(result.determinant)) for result in fit.starts} == {-1, 1}
    assert relative_error < 3.0e-4


def test_four_starts_cover_known_local_minimum_and_stabilizer_fixtures() -> None:
    settings = replace(CANDIDATE_SOLVER_SETTINGS, starts=4)
    generic = _bank(707, 4, 6)
    truth = _orthogonal(708, 6, 1)
    fit = optimize_action(conjugate(truth, generic[:3]), generic[:3], seed=709, settings=settings)
    heldout = conjugate(fit.matrix, generic[3])
    expected = conjugate(truth, generic[3])
    assert np.linalg.norm(heldout - expected) / np.linalg.norm(expected) < 1.0e-8
    assert any(
        result.converged and result.start_index not in fit.equivalent_start_indices
        for result in fit.starts
    )

    diagonals = np.array(
        [
            [0.1, 0.1, -0.4, 0.7],
            [-0.3, -0.3, 0.2, 0.9],
            [0.8, 0.8, -0.6, 0.4],
            [0.5, 0.5, 0.1, -0.7],
        ]
    )
    stabilizer_bank = np.stack([np.diag(value) for value in diagonals])
    stabilizer_truth = _orthogonal(710, 4, -1)
    stabilizer_fit = optimize_action(
        conjugate(stabilizer_truth, stabilizer_bank[:3]),
        stabilizer_bank[:3],
        seed=711,
        settings=settings,
    )
    predicted = conjugate(stabilizer_fit.matrix, stabilizer_bank[3])
    expected = conjugate(stabilizer_truth, stabilizer_bank[3])
    assert np.linalg.norm(predicted - expected) / np.linalg.norm(expected) < 1.0e-8


def test_call_budget_reconstructs_failed_and_candidate_counts() -> None:
    budget = bnci_stage_a_call_budget(4)
    assert budget["stage_a_cells"] == 72
    assert budget["old_actual_pymanopt_runs_per_source_fit"] == 30_240
    assert budget["old_actual_source_pymanopt_runs_full_and_halves"] == 6_531_840
    assert budget["profiled_product_fit_objects_full_and_halves"] == 216
    assert budget["profiled_product_optimizer_runs_full_and_halves"] == 864
    assert budget["target_single_action_pymanopt_runs"] == 2_304
    assert budget["pairwise_core_fit_objects_full"] == 576
    assert budget["pairwise_core_pymanopt_runs_full"] == 2_304
    assert budget["pairwise_core_pymanopt_runs_full_and_halves"] == 6_912
    assert budget["pairwise_total_pymanopt_runs_with_halves_and_semantic"] == 18_432
