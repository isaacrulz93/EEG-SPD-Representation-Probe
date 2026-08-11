from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

import src.local_gpa_geometry_v0 as geometry
from src.geometry_v2 import airm_mean, spd_log


def _configuration(seed: int, d: int = 4, scale: float = 0.12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logs = rng.normal(scale=scale, size=(5, d, d))
    logs = 0.5 * (logs + logs.transpose(0, 2, 1))
    logs -= logs.mean(axis=0)
    return np.stack([expm(value) for value in logs])


def _orthogonal(seed: int, d: int, determinant: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    if np.sign(np.linalg.det(q)) != determinant:
        q[:, 0] *= -1.0
    return q


def test_zero_sum_log_parameterization_enforces_airm_mean_identity() -> None:
    configuration = _configuration(1)
    assert geometry.constraint_residual(configuration) < 1.0e-12
    result = airm_mean(configuration, tol=1.0e-10, maxiter=100)
    assert result.normalized_post_residual < 1.0e-9
    assert result.matrix == pytest.approx(np.eye(4), abs=1.0e-9, rel=1.0e-9)


def test_local_centering_is_local_only_and_has_identity_mean() -> None:
    original = _configuration(2)
    common = np.diag([1.3, 0.8, 1.1, 0.9])
    shifted = np.asarray([common @ value @ common.T for value in original])
    centered = geometry.local_center_configuration(shifted)
    assert centered.normalized_karcher_residual < 1.0e-9
    assert geometry.constraint_residual(centered.states) < 1.0e-6


@pytest.mark.parametrize("determinant", [-1, 1])
def test_known_o_d_and_s5_transform_has_zero_quotient_distance(
    determinant: int,
) -> None:
    source = _configuration(3)
    permutation = np.asarray([2, 0, 4, 1, 3])
    q = _orthogonal(4 + determinant, 4, determinant)
    target = geometry.conjugate_configuration(source[permutation], q)
    distance, fit = geometry.quotient_distance(target, source)
    assert distance < 1.0e-10
    assert fit.objective < 1.0e-18
    assert {value.determinant for value in fit.starts if value.converged} == {-1, 1}
    assert len(fit.starts) == geometry.CANDIDATE_SETTINGS.registration_total_starts


def test_quotient_distance_is_symmetric_by_frozen_canonical_evaluation() -> None:
    first = _configuration(10)
    second = _configuration(11)
    forward, _ = geometry.quotient_distance(first, second)
    reverse, _ = geometry.quotient_distance(second, first)
    assert forward == reverse


def test_action_gradient_matches_tangent_directional_finite_difference() -> None:
    target = _configuration(12)
    source = _configuration(13)
    q = _orthogonal(14, 4, 1)
    permutation = np.asarray([1, 4, 0, 2, 3])
    ambient = np.random.default_rng(15).normal(size=q.shape)
    tangent = q @ (0.5 * (q.T @ ambient - ambient.T @ q))
    tangent /= np.linalg.norm(tangent)
    analytic = float(
        np.sum(
            geometry.fixed_registration_gradient(target, source, q, permutation)
            * tangent
        )
    )
    epsilon = 1.0e-6
    plus, _ = np.linalg.qr(q + epsilon * tangent)
    minus, _ = np.linalg.qr(q - epsilon * tangent)
    if np.sign(np.linalg.det(plus)) != np.sign(np.linalg.det(q)):
        plus[:, 0] *= -1
    if np.sign(np.linalg.det(minus)) != np.sign(np.linalg.det(q)):
        minus[:, 0] *= -1
    finite = (
        geometry.fixed_registration_objective(target, source, plus, permutation)
        - geometry.fixed_registration_objective(target, source, minus, permutation)
    ) / (2 * epsilon)
    assert analytic == pytest.approx(finite, abs=2.0e-6, rel=2.0e-5)


def test_prototype_zero_sum_log_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(16)
    logs = spd_log(_configuration(17))
    observations = np.stack([_configuration(seed) for seed in (18, 19, 20)])
    direction = rng.normal(size=logs.shape)
    direction = 0.5 * (direction + direction.transpose(0, 2, 1))
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    gradient = geometry._prototype_gradient_logs(logs, observations)
    epsilon = 1.0e-6

    def objective(values: np.ndarray) -> float:
        prototype = geometry.configuration_from_zero_sum_logs(values)
        return geometry.fixed_aligned_objective(prototype, observations)

    finite = (objective(logs + epsilon * direction) - objective(logs - epsilon * direction)) / (
        2 * epsilon
    )
    analytic = float(np.sum(gradient * direction))
    assert analytic == pytest.approx(finite, abs=2.0e-9, rel=2.0e-6)


def test_simple_known_answer_gpa_multistarts_agree_in_quotient() -> None:
    rng = np.random.default_rng(21)
    base = _configuration(22)
    trials = []
    for trial in range(6):
        q = _orthogonal(100 + trial, 4, 1 if trial % 2 == 0 else -1)
        trials.append(
            geometry.conjugate_configuration(base[rng.permutation(5)], q)
        )
    fit = geometry.fit_quotient_gpa(
        np.asarray(trials), identity_parts=("known-answer",)
    )
    distance, _ = geometry.quotient_distance(fit.prototype, base)
    start_distance, _ = geometry.quotient_distance(
        fit.starts[0].prototype, fit.starts[1].prototype
    )
    assert fit.objective < 1.0e-18
    assert distance < 1.0e-10
    assert start_distance < 1.0e-10
    assert geometry.constraint_residual(fit.prototype) < 1.0e-10


def test_d22_single_action_sanity_and_exact_total_start_contract() -> None:
    source = _configuration(23, d=22, scale=0.02)
    q = _orthogonal(24, 22, -1)
    target = geometry.conjugate_configuration(source[[4, 2, 0, 3, 1]], q)
    fit = geometry.register_configuration(target, source, seed=25)
    assert len(fit.starts) == 4
    assert sum(value.determinant == 1 for value in fit.starts) == 2
    assert sum(value.determinant == -1 for value in fit.starts) == 2
    assert fit.objective < 1.0e-18
