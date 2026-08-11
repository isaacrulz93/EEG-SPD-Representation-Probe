from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

import src.local_mean_movement_v0 as movement
from src.local_temporal_sequence_v0 import classbreak_mappings, subjectbreak_mappings


def _orthogonal(seed: int, dimension: int, determinant: int) -> np.ndarray:
    action, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    if int(np.sign(np.linalg.det(action))) != determinant:
        action[:, 0] *= -1.0
    return action


def _spd_sequence(seed: int, dimension: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logs = rng.normal(scale=0.16, size=(5, dimension, dimension))
    logs = 0.5 * (logs + logs.transpose(0, 2, 1))
    bases = np.asarray([expm(value) for value in logs])
    common = rng.normal(scale=0.15, size=(dimension, dimension)) + 1.2 * np.eye(dimension)
    return np.asarray([common @ value @ common.T for value in bases])


def _movement(seed: int, dimension: int = 6) -> np.ndarray:
    values = np.random.default_rng(seed).normal(scale=0.12, size=(4, dimension, dimension))
    return 0.5 * (values + values.transpose(0, 2, 1))


def test_ordered_reverse_prefix_antidevelopment_and_norm_identities() -> None:
    sequence = _spd_sequence(1)
    result = movement.anti_develop_sequence(sequence)
    expected_w3 = movement.parallel_transport(
        sequence[1],
        sequence[0],
        movement.parallel_transport(sequence[2], sequence[1], result.displacements[2]),
    )
    assert result.transported[2] == pytest.approx(expected_w3, abs=1.0e-12, rel=1.0e-12)
    assert result.diagnostics["passed"].all()
    assert result.diagnostics["maximum_norm_absolute_error"].max() < 1.0e-10
    assert result.diagnostics["maximum_edge_transport_relative_error"].max() < 1.0e-10


def test_common_congruence_has_one_common_orthogonal_action_for_all_steps() -> None:
    sequence = _spd_sequence(2)
    first = movement.anti_develop_sequence(sequence)
    rng = np.random.default_rng(3)
    congruence = rng.normal(scale=0.2, size=(5, 5)) + 1.5 * np.eye(5)
    changed_sequence = np.asarray([congruence @ value @ congruence.T for value in sequence])
    changed = movement.anti_develop_sequence(changed_sequence)
    common_o = (
        movement.spd_invsqrt(changed_sequence[0])
        @ congruence
        @ movement._symmetric_sqrt(sequence[0])
    )
    assert common_o @ common_o.T == pytest.approx(np.eye(5), abs=1.0e-10, rel=1.0e-10)
    for step in range(4):
        expected = common_o @ first.z[step] @ common_o.T
        assert changed.z[step] == pytest.approx(expected, abs=1.0e-10, rel=1.0e-10)


def test_analytic_movement_gradient_and_hessian_match_finite_differences() -> None:
    target = _movement(10, dimension=5)
    source = _movement(11, dimension=5)
    action = _orthogonal(12, 5, 1)
    rng = np.random.default_rng(13)
    ambient = rng.normal(size=(5, 5))
    tangent = action @ (0.5 * (action.T @ ambient - ambient.T @ action))
    tangent /= np.linalg.norm(tangent)
    epsilon = 1.0e-6
    plus = action @ expm(epsilon * (action.T @ tangent))
    minus = action @ expm(-epsilon * (action.T @ tangent))
    finite = (
        movement.movement_objective(target, source, plus)
        - movement.movement_objective(target, source, minus)
    ) / (2.0 * epsilon)
    analytic = float(
        np.sum(movement.movement_euclidean_gradient(target, source, action) * tangent)
    )
    assert analytic == pytest.approx(finite, abs=2.0e-7, rel=2.0e-6)

    finite_hessian = (
        movement.movement_euclidean_gradient(target, source, action + epsilon * tangent)
        - movement.movement_euclidean_gradient(target, source, action - epsilon * tangent)
    ) / (2.0 * epsilon)
    analytic_hessian = movement.movement_euclidean_hessian(
        target, source, action, tangent
    )
    assert analytic_hessian == pytest.approx(finite_hessian, abs=2.0e-8, rel=2.0e-6)


@pytest.mark.parametrize("determinant", [-1, 1])
def test_known_common_action_zero_and_both_determinant_sectors(determinant: int) -> None:
    source = _movement(20, dimension=6)
    truth = _orthogonal(21 + determinant, 6, determinant)
    target = movement.conjugate_movement(source, truth)
    fit = movement.optimize_movement_alignment(target, source)
    assert fit.distance < 1.0e-8
    assert len(fit.starts) == 6
    assert sum(value.initial_determinant == -1 for value in fit.starts) == 3
    assert sum(value.initial_determinant == 1 for value in fit.starts) == 3
    assert {value.final_determinant for value in fit.starts if value.converged} == {-1, 1}
    spectral_truth = [
        value.objective
        for value in fit.starts
        if value.final_determinant == determinant
        and value.start_kind.startswith("spectral")
    ]
    assert len(spectral_truth) == 2
    assert max(spectral_truth) - min(spectral_truth) < 1.0e-8


def test_movement_distance_forward_reverse_is_exact_by_canonical_contract() -> None:
    first = _movement(30, dimension=5)
    second = _movement(31, dimension=5)
    forward, forward_fit = movement.movement_distance(first, second)
    reverse, reverse_fit = movement.movement_distance(second, first)
    assert forward == reverse
    assert forward_fit.objective == reverse_fit.objective
    assert forward_fit.best_start_index == reverse_fit.best_start_index


def test_distance_contrasts_and_terminal_hierarchy() -> None:
    matrix = np.empty((36, 36), dtype=np.float64)
    for subject in range(9):
        for class_index in range(4):
            row = movement.cell_index(subject, class_index)
            for other_subject in range(9):
                for other_class in range(4):
                    column = movement.cell_index(other_subject, other_class)
                    if subject == other_subject and class_index == other_class:
                        value = 0.0
                    elif subject == other_subject:
                        value = 1.0
                    elif class_index == other_class:
                        value = 2.0
                    else:
                        value = 2.5
                    matrix[row, column] = value
    result = movement.movement_contrasts(matrix)
    assert result.t_subject == pytest.approx(2.0)
    assert result.t_class == pytest.approx(1.0)
    assert result.t_j == pytest.approx(0.5)
    assert movement.terminal_decision(
        t_subject=1.0,
        p_subject=0.01,
        t_class=1.0,
        p_class=0.01,
        t_j=1.0,
        p_j_subjectbreak=0.01,
        p_j_classbreak=0.01,
    ) == "GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT"
    assert movement.terminal_decision(
        t_subject=1.0,
        p_subject=0.01,
        t_class=1.0,
        p_class=0.01,
        t_j=0.0,
        p_j_subjectbreak=1.0,
        p_j_classbreak=1.0,
    ) == "GO_REPRODUCIBLE_ORDERED_MOVEMENT_WITHOUT_INTERACTION"
    assert movement.terminal_decision(
        t_subject=1.0,
        p_subject=0.01,
        t_class=0.0,
        p_class=1.0,
        t_j=1.0,
        p_j_subjectbreak=0.01,
        p_j_classbreak=0.01,
    ) == "STOP_NO_REPRODUCIBLE_ORDERED_MOVEMENT_V0"


def test_inference_reuses_exact_frozen_whole_cell_null_mappings() -> None:
    rng = np.random.default_rng(40)
    matrix = rng.uniform(size=(36, 36))
    inference = movement.evaluate_movement_inference(matrix, replicates=7)
    assert np.array_equal(inference.subject_mappings, subjectbreak_mappings(replicates=7))
    assert np.array_equal(inference.class_mappings, classbreak_mappings(replicates=7))


def test_full_d22_synthetic_gate_contract_passes() -> None:
    record = movement.run_synthetic_numerical_gates()
    assert record["status"] == "PASS"
    assert record["dimension"] == 22
    assert record["new_bnci_movement_statistic_computed"] is False
