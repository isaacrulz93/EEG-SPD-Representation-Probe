"""Synthetic-only calibration tests for common subject action v0."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pymanopt
import pytest
import yaml
from autograd import grad as autograd_grad
from pymanopt import Problem
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from pyriemann.optimization.grassmann import (
    _get_rotation_manifold,
    _grad as _pyriemann_gradient,
    _loss as _pyriemann_loss,
)
from pyriemann.utils._check import check_weights
from scipy.linalg import expm

from src.common_action_solver_v0 import (
    CANDIDATE_SOLVER_SETTINGS,
    action_gradient,
    action_objective,
    analyze_common_stabilizer,
    assess_predictive_identifiability,
    build_pymanopt_optimizer,
    classify_prediction_set,
    conjugate,
    deterministic_starts,
    diagnose_multistart,
    equivalence_objective_tolerance,
    fit_source_model,
    heldout_template,
    nonidentity_permutations_three,
    optimize_action,
    optimize_action_custom,
)
from src.common_subject_action_v0 import (
    AuditContractError,
    CLASSES,
    NULL_REPLICATES,
    comparator_choice_matrix,
    error_and_gain,
    loco_folds,
    raw_population_prediction,
    residual_class_correspondence_null,
    required_identifiability_gate,
    seed_vector,
    subject_group_statistic,
    terminal_decision,
)


def _orthogonal(seed: int, dimension: int, determinant: int = 1) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    q = q @ np.diag(np.where(np.diag(r) < 0.0, -1.0, 1.0))
    if np.sign(np.linalg.det(q)) != int(determinant):
        q[:, 0] *= -1.0
    return q


def _bank(seed: int, count: int, dimension: int, *, repeated_first: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for index in range(count):
        matrix = rng.normal(size=(dimension, dimension))
        matrix = 0.5 * (matrix + matrix.T)
        if repeated_first and index == 0:
            diagonal = np.linspace(-0.2, 0.3, dimension)
            diagonal[:2] = 0.1
            matrix = np.diag(diagonal)
        values.append(matrix)
    return np.asarray(values)


@pytest.mark.parametrize("dimension", [4, 22])
@pytest.mark.parametrize("determinant", [1, -1])
def test_synthetic_common_q_exact_recovery_in_both_components(dimension: int, determinant: int) -> None:
    templates = _bank(10 + dimension, 3, dimension)
    truth = _orthogonal(20 + dimension, dimension, determinant)
    targets = conjugate(truth, templates)
    fit = optimize_action(targets, templates, seed=700 + dimension + determinant)
    prediction = conjugate(fit.matrix, templates)
    relative = np.linalg.norm(prediction - targets) / np.linalg.norm(targets)
    assert relative < (2e-8 if dimension == 22 else 2e-10)
    assert any(result.determinant > 0.0 for result in fit.starts)
    assert any(result.determinant < 0.0 for result in fit.starts)
    assert fit.starts[fit.best_start_index].converged


@pytest.mark.parametrize("dimension", [6, 22])
@pytest.mark.parametrize("determinant", [1, -1])
def test_standard_and_independent_solvers_agree_on_identifiable_action(
    dimension: int,
    determinant: int,
) -> None:
    templates = _bank(50 + dimension, 4, dimension)
    truth = _orthogonal(60 + dimension, dimension, determinant)
    targets = conjugate(truth, templates[:3])
    starts = deterministic_starts(
        targets,
        templates[:3],
        seed=70 + dimension + determinant,
        count=CANDIDATE_SOLVER_SETTINGS.starts,
    )
    standard = optimize_action(
        targets,
        templates[:3],
        seed=80 + dimension,
        starts=starts,
    )
    independent = optimize_action_custom(
        targets,
        templates[:3],
        seed=80 + dimension,
        starts=starts,
    )
    standard_fit = conjugate(standard.matrix, templates[:3])
    independent_fit = conjugate(independent.matrix, templates[:3])
    standard_heldout = conjugate(standard.matrix, templates[3])
    independent_heldout = conjugate(independent.matrix, templates[3])
    truth_heldout = conjugate(truth, templates[3])
    tolerance = CANDIDATE_SOLVER_SETTINGS.exact_prediction_relative_tolerance
    for prediction in (standard_fit, independent_fit):
        assert np.linalg.norm(prediction - targets) / np.linalg.norm(targets) < tolerance
    for prediction in (standard_heldout, independent_heldout):
        assert np.linalg.norm(prediction - truth_heldout) / np.linalg.norm(truth_heldout) < tolerance
    assert (
        np.linalg.norm(standard_heldout - independent_heldout)
        / np.linalg.norm(truth_heldout)
        < tolerance
    )


def test_generic_plus_repeated_spectrum_bank_is_recovered() -> None:
    templates = _bank(82, 3, 7, repeated_first=True)
    truth = _orthogonal(83, 7, -1)
    targets = conjugate(truth, templates)
    fit = optimize_action(targets, templates, seed=84)
    np.testing.assert_allclose(conjugate(fit.matrix, templates), targets, rtol=0.0, atol=2e-9)


def test_low_noise_solution_improves_on_identity_and_is_deterministic() -> None:
    templates = _bank(91, 3, 8)
    truth = _orthogonal(92, 8, 1)
    rng = np.random.default_rng(93)
    noise = rng.normal(scale=1e-4, size=templates.shape)
    noise = 0.5 * (noise + noise.transpose(0, 2, 1))
    targets = conjugate(truth, templates) + noise
    first = optimize_action(targets, templates, seed=94)
    second = optimize_action(targets, templates, seed=94)
    assert action_objective(targets, templates, first.matrix) < action_objective(targets, templates, np.eye(8))
    np.testing.assert_array_equal(first.matrix, second.matrix)


def test_analytic_gradient_matches_directional_finite_difference() -> None:
    targets = _bank(101, 3, 5)
    templates = _bank(102, 3, 5)
    q = _orthogonal(103, 5)
    direction = np.random.default_rng(104).normal(size=(5, 5))
    gradient = action_gradient(targets, templates, q)
    epsilon = 1e-7
    numerical = (
        action_objective(targets, templates, q + epsilon * direction)
        - action_objective(targets, templates, q - epsilon * direction)
    ) / (2 * epsilon)
    analytic = float(np.sum(gradient * direction))
    assert numerical == pytest.approx(analytic, rel=2e-7, abs=2e-7)


def test_analytic_gradient_matches_autograd() -> None:
    targets = _bank(1041, 3, 5)
    templates = _bank(1042, 3, 5)
    q = _orthogonal(1043, 5, -1)

    def cost(flattened):
        action = anp.reshape(flattened, q.shape)
        result = 0.0
        for target, template in zip(targets, templates):
            residual = target - action @ template @ action.T
            result = result + anp.sum(residual * residual)
        return result

    expected = np.asarray(autograd_grad(cost)(q.reshape(-1))).reshape(q.shape)
    np.testing.assert_allclose(action_gradient(targets, templates, q), expected, rtol=2e-14, atol=2e-14)


def test_gradient_is_the_pinned_pyriemann_euclidean_congruence_gradient() -> None:
    targets = _bank(106, 3, 5)
    templates = _bank(107, 3, 5)
    q = _orthogonal(108, 5, -1)
    weights = check_weights(None, len(targets))
    expected = _pyriemann_gradient(q, targets, templates, weights, metric="euclid")
    np.testing.assert_allclose(
        action_gradient(targets, templates, q) / len(targets),
        expected,
        rtol=2e-15,
        atol=2e-15,
    )


def test_pyriemann_012_euclidean_runtime_discrepancy_is_pinned() -> None:
    dimension = 6
    tangent_templates = 0.08 * _bank(109, 4, dimension)
    source_spd = np.stack([expm(value) for value in tangent_templates])
    truth = _orthogonal(110, dimension, -1)
    target_spd = conjugate(truth, source_spd)
    official = _get_rotation_manifold(
        source_spd,
        target_spd,
        metric="euclid",
        tol_step=1e-11,
        maxiter=10_000,
    )
    official = np.asarray(np.real_if_close(official), dtype=np.float64)
    ours = optimize_action(target_spd, source_spd, seed=111)
    official_prediction = conjugate(official, source_spd)
    our_prediction = conjugate(ours.matrix, source_spd)
    np.testing.assert_allclose(official.T @ official, np.eye(dimension), rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(ours.matrix.T @ ours.matrix, np.eye(dimension), rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(our_prediction, target_spd, rtol=0.0, atol=2e-8)

    heldout_tangent = 0.08 * _bank(112, 1, dimension)[0]
    heldout_spd = expm(heldout_tangent)
    official_heldout = conjugate(official, heldout_spd)
    our_heldout = conjugate(ours.matrix, heldout_spd)
    truth_heldout = conjugate(truth, heldout_spd)
    np.testing.assert_allclose(our_heldout, truth_heldout, rtol=0.0, atol=3e-8)

    weights = check_weights(None, len(source_spd))
    official_runtime_loss = _pyriemann_loss(official, target_spd, source_spd, weights, metric="euclid")
    official_squared_loss = action_objective(target_spd, source_spd, official) / len(source_spd)
    our_squared_loss = action_objective(target_spd, source_spd, ours.matrix) / len(source_spd)
    assert official_runtime_loss > 1e-3
    assert official_squared_loss > 1e-3
    assert our_squared_loss < 1e-15
    assert np.linalg.norm(official_heldout - truth_heldout) / np.linalg.norm(truth_heldout) > 1e-2


def test_modern_pymanopt_so_reference_matches_o_solver_on_original_rpa_euclidean_form() -> None:
    dimension = 7
    templates = _bank(113, 4, dimension)
    truth = _orthogonal(114, dimension, 1)
    targets = conjugate(truth, templates[:3])
    manifold = SpecialOrthogonalGroup(dimension)

    @pymanopt.function.numpy(manifold)
    def cost(action):
        return action_objective(targets, templates[:3], action)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(action):
        return action_gradient(targets, templates[:3], action)

    problem = Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
    reference = ConjugateGradient(
        beta_rule="HestenesStiefel",
        line_searcher=BackTrackingLineSearcher(
            contraction_factor=0.5,
            optimism=2.0,
            sufficient_decrease=1e-4,
            max_iterations=50,
            initial_step_size=1.0,
        ),
        max_iterations=1000,
        min_gradient_norm=1e-6,
        min_step_size=1e-12,
        max_time=3600,
        verbosity=0,
    ).run(problem, initial_point=truth)
    ours = optimize_action(targets, templates[:3], seed=116)
    reference_fit = conjugate(reference.point, templates[:3])
    ours_fit = conjugate(ours.matrix, templates[:3])
    reference_heldout = conjugate(reference.point, templates[3])
    ours_heldout = conjugate(ours.matrix, templates[3])
    truth_heldout = conjugate(truth, templates[3])
    tolerance = CANDIDATE_SOLVER_SETTINGS.exact_prediction_relative_tolerance
    assert action_objective(targets, templates[:3], reference.point) < 1e-14
    assert action_objective(targets, templates[:3], ours.matrix) < 1e-14
    assert np.linalg.norm(reference_fit - ours_fit) / np.linalg.norm(targets) < tolerance
    assert np.linalg.norm(reference_heldout - truth_heldout) / np.linalg.norm(truth_heldout) < tolerance
    assert np.linalg.norm(ours_heldout - truth_heldout) / np.linalg.norm(truth_heldout) < tolerance


def _source_objects(*, stable: bool, seed: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, contexts, classes, dimension = 5, 2, 3, 5
    templates = _bank(seed, contexts * classes, dimension).reshape(contexts, classes, dimension, dimension)
    actions = np.empty((n, contexts, dimension, dimension))
    actions[0] = np.eye(dimension)
    for subject in range(1, n):
        first = _orthogonal(seed + 10 * subject, dimension, -1 if subject % 2 else 1)
        actions[subject, 0] = first
        actions[subject, 1] = first if stable else _orthogonal(seed + 10 * subject + 1, dimension, 1)
    objects = np.empty((n, contexts, classes, dimension, dimension))
    for subject in range(n):
        for context in range(contexts):
            objects[subject, context] = conjugate(actions[subject, context], templates[context])
    return objects, templates, actions


def test_synthetic_cross_session_stable_source_model_and_global_anchor() -> None:
    objects, _, _ = _source_objects(stable=True)
    fit = fit_source_model(objects, anchor_index=0, seed=140)
    assert fit.starts[fit.best_start_index].converged
    np.testing.assert_array_equal(fit.actions[0], np.eye(objects.shape[-1]))
    prediction = np.stack([conjugate(fit.actions[index], fit.templates) for index in range(len(objects))])
    assert np.linalg.norm(prediction - objects) / np.linalg.norm(objects) < 2e-8


def test_synthetic_session_specific_actions_recover_when_sessions_fit_separately() -> None:
    objects, _, _ = _source_objects(stable=False, seed=160)
    separate_residual = 0.0
    for session in range(2):
        fit = fit_source_model(objects[:, session : session + 1], anchor_index=0, seed=170 + session)
        prediction = np.stack([conjugate(fit.actions[index], fit.templates) for index in range(len(objects))])
        separate_residual += float(np.linalg.norm(prediction - objects[:, session : session + 1]) ** 2)
    stable = fit_source_model(objects, anchor_index=0, seed=180)
    stable_prediction = np.stack([conjugate(stable.actions[index], stable.templates) for index in range(len(objects))])
    stable_residual = float(np.linalg.norm(stable_prediction - objects) ** 2)
    assert separate_residual < 1e-12
    assert stable_residual > separate_residual + 1e-3


def test_generic_noncommuting_templates_have_trivial_continuous_stabilizer() -> None:
    templates = _bank(181, 3, 8)
    diagnostic = analyze_common_stabilizer(templates)
    assert diagnostic.numerical_nullity == 0
    assert diagnostic.approximate_nullity == 0
    assert len(diagnostic.singular_values) == 8 * 7 // 2
    assert diagnostic.singular_values[-1] > diagnostic.approximate_tolerance


def test_repeated_eigenvalue_block_has_nontrivial_exact_stabilizer() -> None:
    diagonals = np.array(
        [
            [0.1, 0.1, -0.4, 0.7, 1.2],
            [-0.3, -0.3, 0.2, 0.9, 1.7],
            [0.8, 0.8, -0.6, 0.4, 2.1],
        ]
    )
    templates = np.stack([np.diag(value) for value in diagonals])
    diagnostic = analyze_common_stabilizer(templates)
    assert diagnostic.numerical_nullity == 1
    assert diagnostic.approximate_nullity == 1
    omega = diagnostic.nullspace_basis[0]
    for template in templates:
        np.testing.assert_allclose(omega @ template - template @ omega, 0.0, atol=2e-14)


def test_noisy_repeated_block_is_approximate_not_exact_stabilizer() -> None:
    diagonals = np.array(
        [
            [0.1, 0.1, -0.4, 0.7, 1.2],
            [-0.3, -0.3, 0.2, 0.9, 1.7],
            [0.8, 0.8, -0.6, 0.4, 2.1],
        ]
    )
    templates = np.stack([np.diag(value) for value in diagonals])
    perturbation = np.zeros_like(templates)
    perturbation[:, 0, 1] = [1e-10, -2e-10, 1e-10]
    perturbation[:, 1, 0] = perturbation[:, 0, 1]
    diagnostic = analyze_common_stabilizer(templates + perturbation)
    assert diagnostic.numerical_nullity == 0
    assert diagnostic.approximate_nullity == 1
    assert diagnostic.singular_values[-1] > diagnostic.numerical_tolerance
    assert diagnostic.singular_values[-1] <= diagnostic.approximate_tolerance


def test_commuting_templates_show_lie_test_does_not_capture_discrete_sign_stabilizer() -> None:
    templates = np.stack(
        [
            np.diag([-1.0, -0.1, 0.4, 1.2]),
            np.diag([0.2, 0.8, 1.7, 2.9]),
            np.diag([-0.7, 0.3, 2.1, 4.0]),
        ]
    )
    diagnostic = analyze_common_stabilizer(templates)
    assert diagnostic.numerical_nullity == 0
    sign_flip = np.diag([-1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(conjugate(sign_flip, templates), templates, atol=0.0)


def test_equivalent_q_with_same_heldout_prediction_is_harmless_nonuniqueness() -> None:
    identity = np.eye(4)
    sign_flip = np.diag([-1.0, 1.0, 1.0, 1.0])
    heldout = np.diag([0.2, 0.7, 1.1, 2.0])
    result = classify_prediction_set(
        (identity, sign_flip),
        heldout,
        split_half_prediction_a=heldout,
        split_half_prediction_b=heldout,
    )
    assert result.classification == "HARMLESS_Q_NONUNIQUENESS"
    assert result.maximum_relative_prediction_dispersion == 0.0
    assert result.materiality_threshold == CANDIDATE_SOLVER_SETTINGS.prediction_dispersion_numerical_floor


def test_equivalent_fit_q_with_different_heldout_prediction_is_nonidentifiable() -> None:
    fit_templates = np.stack(
        [
            np.diag([-1.0, -0.1, 0.4, 1.2]),
            np.diag([0.2, 0.8, 1.7, 2.9]),
            np.diag([-0.7, 0.3, 2.1, 4.0]),
        ]
    )
    identity = np.eye(4)
    sign_flip = np.diag([-1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(conjugate(identity, fit_templates), conjugate(sign_flip, fit_templates), atol=0.0)
    heldout = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.2],
            [0.0, 0.0, 0.2, -0.3],
        ]
    )
    result = classify_prediction_set(
        (identity, sign_flip),
        heldout,
        split_half_prediction_a=heldout,
        split_half_prediction_b=1.001 * heldout,
    )
    assert result.classification == "PREDICTIVE_NONIDENTIFIABILITY"
    assert result.maximum_relative_prediction_dispersion > result.materiality_threshold


def test_continuous_stabilizer_is_judged_by_prediction_not_q_uniqueness() -> None:
    fit_templates = np.stack(
        [
            np.diag([0.1, 0.1, -0.4, 0.7]),
            np.diag([-0.3, -0.3, 0.2, 0.9]),
            np.diag([0.8, 0.8, -0.6, 0.4]),
        ]
    )
    target = fit_templates.copy()
    fit = optimize_action(target, fit_templates, seed=190)
    harmless_heldout = np.diag([0.5, 0.5, 0.1, -0.7])
    stabilizer, harmless = assess_predictive_identifiability(
        fit,
        target,
        fit_templates,
        harmless_heldout,
        split_half_prediction_a=harmless_heldout,
        split_half_prediction_b=harmless_heldout,
    )
    assert stabilizer.numerical_nullity == 1
    assert harmless.classification == "HARMLESS_Q_NONUNIQUENESS"
    dangerous_heldout = harmless_heldout.copy()
    dangerous_heldout[0, 1] = dangerous_heldout[1, 0] = 0.4
    _, dangerous = assess_predictive_identifiability(
        fit,
        target,
        fit_templates,
        dangerous_heldout,
        split_half_prediction_a=dangerous_heldout,
        split_half_prediction_b=1.001 * dangerous_heldout,
    )
    assert dangerous.classification == "PREDICTIVE_NONIDENTIFIABILITY"


def test_materiality_rule_uses_one_for_one_split_half_noise_scale() -> None:
    identity = np.eye(3)
    angle = 0.01
    alternative = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    heldout = np.array([[0.0, 1.0, 0.0], [1.0, 0.2, 0.0], [0.0, 0.0, -0.3]])
    strict = classify_prediction_set(
        (identity, alternative),
        heldout,
        split_half_prediction_a=heldout,
        split_half_prediction_b=1.0001 * heldout,
    )
    assert strict.classification == "PREDICTIVE_NONIDENTIFIABILITY"
    measurement_limited = classify_prediction_set(
        (identity, alternative),
        heldout,
        split_half_prediction_a=heldout,
        split_half_prediction_b=(
            1.0 + 2.0 * strict.maximum_relative_prediction_dispersion
        )
        * heldout,
    )
    assert measurement_limited.classification == "HARMLESS_Q_NONUNIQUENESS"
    assert measurement_limited.materiality_threshold == pytest.approx(
        2.0 * strict.maximum_relative_prediction_dispersion
    )


def test_noisy_identifiable_fixture_multistart_spread_is_below_split_half_variability() -> None:
    dimension = 7
    templates = _bank(195, 4, dimension)
    truth = _orthogonal(196, dimension, -1)
    rng = np.random.default_rng(197)
    predictions = []
    fits = []
    fit_targets = []
    for half in range(2):
        noise = rng.normal(scale=2e-4, size=(3, dimension, dimension))
        noise = 0.5 * (noise + noise.transpose(0, 2, 1))
        target = conjugate(truth, templates[:3]) + noise
        fit = optimize_action(target, templates[:3], seed=198 + half)
        fits.append(fit)
        fit_targets.append(target)
        predictions.append(conjugate(fit.matrix, templates[3]))
    for fit, target in zip(fits, fit_targets):
        _, result = assess_predictive_identifiability(
            fit,
            target,
            templates[:3],
            templates[3],
            split_half_prediction_a=predictions[0],
            split_half_prediction_b=predictions[1],
        )
        assert result.classification in {
            "PREDICTIVELY_IDENTIFIABLE",
            "HARMLESS_Q_NONUNIQUENESS",
        }
        assert result.maximum_relative_prediction_dispersion <= result.materiality_threshold


def test_numerical_local_minima_are_not_mislabeled_as_equivalent_actions() -> None:
    dimension = 6
    templates = _bank(707, 4, dimension)
    truth = _orthogonal(708, dimension, 1)
    targets = conjugate(truth, templates[:3])
    fit = optimize_action(targets, templates[:3], seed=709)
    landscape = diagnose_multistart(fit)
    best = fit.starts[fit.best_start_index]
    assert best.objective < 1e-20
    assert len(fit.equivalent_start_indices) == 1
    local_minima = [
        result
        for result in fit.starts
        if result.converged and result.start_index not in fit.equivalent_start_indices
    ]
    assert local_minima
    assert all(result.objective > best.objective + 1e-3 for result in local_minima)
    assert landscape.classification == "ALTERNATIVE_LOCAL_MINIMA_EXCLUDED"
    assert landscape.determinant_sectors_with_converged_solution == (-1, 1)
    assert landscape.converged_non_equivalent_count == len(local_minima)
    heldout = conjugate(truth, templates[3])
    assessment = classify_prediction_set(
        tuple(fit.starts[index].matrix for index in fit.equivalent_start_indices),
        templates[3],
        split_half_prediction_a=heldout,
        split_half_prediction_b=heldout,
    )
    predicted = conjugate(fit.matrix, templates[3])
    assert assessment.classification == "PREDICTIVELY_IDENTIFIABLE"
    assert np.linalg.norm(predicted - heldout) / np.linalg.norm(heldout) < 1e-8


def test_conjugation_preserves_symmetry_and_heldout_template_uses_fixed_actions() -> None:
    bank = _bank(201, 4, 6)
    actions = np.stack([_orthogonal(210 + index, 6, -1 if index % 2 else 1) for index in range(4)])
    latent = bank[0]
    source = np.stack([conjugate(action, latent) for action in actions])
    recovered = heldout_template(actions, source)
    np.testing.assert_allclose(recovered, latent, rtol=0.0, atol=2e-14)
    assert np.linalg.norm(recovered - recovered.T) == pytest.approx(0.0, abs=1e-15)


def test_loco_grid_excludes_target_and_heldout_class_and_uses_fixed_anchor() -> None:
    folds = loco_folds()
    assert len(folds) == 36
    assert {fold.heldout_class_index for fold in folds} == set(range(4))
    for fold in folds:
        assert fold.target_subject_index not in fold.source_subject_indices
        assert fold.heldout_class_index not in fold.fit_class_indices
        assert len(fold.source_subject_indices) == 8
        assert len(fold.fit_class_indices) == 3
        assert fold.source_subject_indices[fold.anchor_source_position] == min(fold.source_subject_indices)


def test_raw_baseline_and_gain_definition_are_exact() -> None:
    source = np.stack([np.eye(3), 2 * np.eye(3)])
    target = 1.75 * np.eye(3)
    raw = raw_population_prediction(source)
    np.testing.assert_array_equal(raw, 1.5 * np.eye(3))
    e_raw, e_q, gain = error_and_gain(target, raw, target)
    assert e_q == 0.0
    assert gain == e_raw and gain > 0.0


def test_semantic_permutations_are_exactly_five_nonidentity_s3() -> None:
    values = nonidentity_permutations_three()
    assert len(values) == 5
    assert (0, 1, 2) not in values
    assert set(values) == set(itertools.permutations(range(3))) - {(0, 1, 2)}


def test_null_seeds_and_choices_are_deterministic_and_stream_separated() -> None:
    first = seed_vector("stage_A_unrelated")
    second = seed_vector("stage_A_unrelated")
    semantic = seed_vector("stage_A_semantic")
    assert first.shape == (NULL_REPLICATES,)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, semantic)
    choices = comparator_choice_matrix("stage_A_semantic", n_cells=72, choices_per_cell=5)
    np.testing.assert_array_equal(
        choices,
        comparator_choice_matrix("stage_A_semantic", n_cells=72, choices_per_cell=5),
    )
    assert choices.shape == (1999, 72)
    assert np.all((choices >= 0) & (choices < 5))


def test_subject_is_group_unit_and_available_case_drop_is_forbidden() -> None:
    subjects = np.repeat(np.arange(9), 8)
    values = np.arange(72, dtype=np.float64) / 100.0
    scores, group = subject_group_statistic(values, subjects)
    assert scores.shape == (9,) and group == np.median(scores)
    with pytest.raises(AuditContractError, match="available-case"):
        subject_group_statistic(values[:-1], subjects[:-1])


def test_residual_class_correspondence_uses_nonidentity_subjectwise_s4() -> None:
    rng = np.random.default_rng(301)
    residuals = rng.normal(size=(9, 4, 3, 3))
    residuals = 0.5 * (residuals + residuals.transpose(0, 1, 3, 2))
    observed, null, choices = residual_class_correspondence_null(residuals, residuals, replicates=17)
    assert observed == pytest.approx(1.0)
    assert null.shape == (17,) and choices.shape == (17, 9)
    assert np.all((choices >= 0) & (choices < 23))


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (dict(data_gate_pass=False, technical_gate_pass=False, identifiable=False, stage_a_pass=None, stage_b_pass=None, stage_c_pass=None), "UNASSESSED_NUMERICAL_OR_DATA_FAILURE"),
        (dict(data_gate_pass=True, technical_gate_pass=False, identifiable=False, stage_a_pass=None, stage_b_pass=None, stage_c_pass=None), "UNASSESSED_TECHNICAL_FAILURE"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=False, stage_a_pass=None, stage_b_pass=None, stage_c_pass=None), "UNASSESSED_ACTION_NOT_IDENTIFIABLE"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=True, stage_a_pass=False, stage_b_pass=None, stage_c_pass=None), "COMMON_ACTION_NOT_SUPPORTED_WITHIN_SESSION"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=True, stage_a_pass=True, stage_b_pass=False, stage_c_pass=None), "SESSION_SPECIFIC_COMMON_ACTION_ONLY"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=True, stage_a_pass=True, stage_b_pass=True, stage_c_pass=True), "COMMON_ACTION_SUPPORTED_RESIDUAL_INDIVIDUALITY_REMAINS"),
        (dict(data_gate_pass=True, technical_gate_pass=True, identifiable=True, stage_a_pass=True, stage_b_pass=True, stage_c_pass=False), "COMMON_ACTION_SUPPORTED_NO_STABLE_RESIDUAL_EVIDENCE"),
    ],
)
def test_terminal_decision_mapping(arguments, expected) -> None:
    assert terminal_decision(**arguments).decision == expected


def test_candidate_settings_cover_both_components_and_are_explicit() -> None:
    bank = _bank(401, 3, 4)
    starts = deterministic_starts(bank, bank, seed=402, count=CANDIDATE_SOLVER_SETTINGS.starts)
    assert len(starts) == 8
    assert {int(np.sign(np.linalg.det(value))) for value in starts} == {-1, 1}
    assert CANDIDATE_SOLVER_SETTINGS.max_iterations == 1000
    assert CANDIDATE_SOLVER_SETTINGS.gradient_tolerance == 1e-5
    assert CANDIDATE_SOLVER_SETTINGS.outer_iterations == 120
    assert CANDIDATE_SOLVER_SETTINGS.exact_prediction_relative_tolerance == 1e-8
    assert CANDIDATE_SOLVER_SETTINGS.prediction_dispersion_numerical_floor == 1e-5


def test_runtime_pymanopt_optimizer_and_line_search_match_frozen_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs/bnci2014_001_common_subject_action_v0.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen = config["optimizer"]
    line_frozen = frozen["line_search"]
    optimizer = build_pymanopt_optimizer()
    line_search = optimizer._line_searcher
    assert type(optimizer).__name__ == "ConjugateGradient"
    assert optimizer._beta_rule == frozen["beta_rule"]
    assert optimizer._max_iterations == frozen["max_iterations"]
    assert optimizer._min_gradient_norm == frozen["min_gradient_norm"]
    assert optimizer._min_step_size == frozen["min_step_size"]
    assert optimizer._max_cost_evaluations == frozen["max_cost_evaluations"]
    assert optimizer._max_time == frozen["max_time_seconds"]
    assert type(line_search).__name__ == "BackTrackingLineSearcher"
    assert line_search.contraction_factor == line_frozen["contraction_factor"]
    assert line_search.optimism == line_frozen["optimism"]
    assert line_search.sufficient_decrease == line_frozen["sufficient_decrease"]
    assert line_search.max_iterations == line_frozen["max_iterations"]
    assert line_search.initial_step_size == line_frozen["initial_step_size"]
    assert not hasattr(line_search, "min_step_size")


def test_near_optimal_objective_rule_uses_abs_best_without_unit_floor() -> None:
    settings = CANDIDATE_SOLVER_SETTINGS
    best = 2.5e-4
    expected = (
        settings.equivalent_objective_absolute_tolerance
        + settings.equivalent_objective_relative_tolerance * abs(best)
    )
    assert equivalence_objective_tolerance(best) == pytest.approx(expected)
    assert expected < 2e-10


def test_only_one_near_optimal_q_has_zero_D_eq() -> None:
    heldout = np.diag([0.2, 0.7, 1.1])
    result = classify_prediction_set(
        (np.eye(3),),
        heldout,
        split_half_prediction_a=heldout,
        split_half_prediction_b=heldout,
    )
    assert result.classification == "PREDICTIVELY_IDENTIFIABLE"
    assert result.equivalent_solution_count == 1
    assert result.maximum_relative_prediction_dispersion == 0.0


def test_numerical_zero_split_prediction_fails_closed() -> None:
    heldout = np.diag([0.2, 0.7, 1.1])
    with pytest.raises(Exception, match="UNASSESSED_NUMERICAL_OR_DATA_FAILURE"):
        classify_prediction_set(
            (np.eye(3),),
            heldout,
            split_half_prediction_a=np.zeros_like(heldout),
            split_half_prediction_b=heldout,
        )


def test_missing_converged_determinant_sector_is_technical_failure() -> None:
    fit_templates = _bank(901, 3, 4)
    target = fit_templates.copy()
    one_start = replace(CANDIDATE_SOLVER_SETTINGS, starts=1)
    fit = optimize_action(
        target,
        fit_templates,
        seed=902,
        settings=one_start,
        starts=(np.eye(4),),
    )
    heldout = _bank(903, 1, 4)[0]
    with pytest.raises(Exception, match="UNASSESSED_TECHNICAL_FAILURE"):
        assess_predictive_identifiability(
            fit,
            target,
            fit_templates,
            heldout,
            split_half_prediction_a=heldout,
            split_half_prediction_b=heldout,
        )


def test_different_determinant_sectors_can_be_harmlessly_equivalent() -> None:
    heldout = _bank(904, 1, 3)[0]
    result = classify_prediction_set(
        (np.eye(3), -np.eye(3)),
        heldout,
        split_half_prediction_a=heldout,
        split_half_prediction_b=heldout,
    )
    assert np.linalg.det(np.eye(3)) > 0 and np.linalg.det(-np.eye(3)) < 0
    assert result.classification == "HARMLESS_Q_NONUNIQUENESS"
    assert result.maximum_relative_prediction_dispersion == 0.0


def test_any_required_nonidentifiable_cell_fails_entire_chain() -> None:
    passing = ["PREDICTIVELY_IDENTIFIABLE"] * 72
    assert required_identifiability_gate(passing, stage="A") == "PASS"
    passing[37] = "PREDICTIVE_NONIDENTIFIABILITY"
    assert (
        required_identifiability_gate(passing, stage="A")
        == "UNASSESSED_ACTION_NOT_IDENTIFIABLE"
    )
    with pytest.raises(AuditContractError, match="available-case"):
        required_identifiability_gate(passing[:-1], stage="A")
