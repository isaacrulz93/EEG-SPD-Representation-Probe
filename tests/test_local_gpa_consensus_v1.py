from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions
from scipy.linalg import expm

import src.local_gpa_geometry_v0 as geometry
from src.trajectory_within_subject_v1 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bnci2014_001_local_gpa_consensus_v1.yaml"
PROTOCOL = ROOT / "docs" / "PROTOCOL_LOCAL_GPA_CONSENSUS_V1_AMENDMENT.md"
RUNNER = ROOT / "scripts" / "23_run_local_gpa_consensus_v1.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("local_gpa_v1_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configuration(seed: int, d: int = 4, scale: float = 0.10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logs = rng.normal(scale=scale, size=(5, d, d))
    logs = 0.5 * (logs + logs.transpose(0, 2, 1))
    logs -= logs.mean(axis=0)
    return np.stack([expm(value) for value in logs])


def _orthogonal(seed: int, d: int, determinant: int) -> np.ndarray:
    action, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    if int(np.sign(np.linalg.det(action))) != determinant:
        action[:, 0] *= -1.0
    return action


def test_v1_protocol_hash_and_v0_scientific_contract_match() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"]["sha256"] == sha256_file(PROTOCOL)
    record = _runner_module().amendment_scope_audit()
    assert record["status"] == "PASS"
    assert all(value["equal"] for value in record["equal_scientific_fields"])


def test_runtime_optimizer_is_exact_audited_trust_regions() -> None:
    optimizer = geometry._action_optimizer(geometry.CANDIDATE_SETTINGS)
    assert isinstance(optimizer, TrustRegions)
    assert optimizer._max_iterations == 250
    assert optimizer._max_time == 120.0
    assert optimizer._min_gradient_norm == 1.0e-6
    assert optimizer._min_step_size == 1.0e-12
    assert optimizer._max_cost_evaluations == 5000
    assert optimizer.miniter == 3
    assert optimizer.kappa == 0.1
    assert optimizer.theta == 1.0
    assert optimizer.rho_prime == 0.1
    assert optimizer.use_rand is False
    assert optimizer.rho_regularization == 1000.0
    assert geometry.TRUST_HESSIAN_RADIUS == 1.0e-5
    assert geometry.TRUST_MAX_INNER_ITERATIONS == 30


def test_audited_hessian_is_nearly_self_adjoint() -> None:
    target = _configuration(701)
    source = _configuration(702)
    action = _orthogonal(703, 4, 1)
    permutation = np.asarray([1, 4, 0, 3, 2])
    manifold, problem = geometry._action_problem(target, source, permutation)
    rng = np.random.default_rng(704)
    first = manifold.projection(action, rng.normal(size=action.shape))
    second = manifold.projection(action, rng.normal(size=action.shape))
    h_first = problem.riemannian_hessian(action, first)
    h_second = problem.riemannian_hessian(action, second)
    left = float(manifold.inner_product(action, first, h_second))
    right = float(manifold.inner_product(action, h_first, second))
    relative = abs(left - right) / max(1.0, abs(left), abs(right))
    assert relative < 1.0e-6


@pytest.mark.parametrize("truth_determinant", [-1, 1])
def test_trust_regions_known_action_and_exact_four_start_contract(
    truth_determinant: int,
) -> None:
    source = _configuration(710 + truth_determinant)
    action = _orthogonal(720 + truth_determinant, 4, truth_determinant)
    target = geometry.conjugate_configuration(source[[2, 0, 4, 1, 3]], action)
    fit = geometry.register_configuration(target, source, seed=730)
    assert len(fit.starts) == 4
    assert sum(value.determinant == -1 for value in fit.starts) == 2
    assert sum(value.determinant == 1 for value in fit.starts) == 2
    assert {value.determinant for value in fit.starts if value.converged} == {-1, 1}
    assert fit.objective < 1.0e-18


def test_amendment_text_declares_optimizer_only_and_no_v0_science() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "post-technical-failure optimizer-only numerical amendment" in text
    assert "before a complete cell consensus" in text
    assert "No Stage-2A scientific result informed this amendment" in text
    assert "32/32 total" in text
    assert "never averaged, compared across cells, or interpreted" in text
