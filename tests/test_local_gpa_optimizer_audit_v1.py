from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pymanopt.manifolds import Stiefel
from scipy.linalg import expm

import src.local_gpa_geometry_v0 as frozen
from src.local_gpa_data_v0 import load_and_reproduce_local_gpa_input
from src.local_gpa_optimizer_audit_v1 import (
    HESSIAN_RADIUS,
    TRUST_MAX_ITERATIONS,
    array_sha256,
    riemannian_hessian_vector,
    run_registration_forensic,
)
from src.local_gpa_pipeline_v0 import cell_tasks
from src.trajectory_within_subject_v1 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CENTERED = ROOT / "cache" / "bnci2014_001_local_gpa_consensus_v0" / "locally_centered_states.npy"


def _configuration(seed: int, d: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logs = rng.normal(scale=0.1, size=(5, d, d))
    logs = 0.5 * (logs + logs.transpose(0, 2, 1))
    logs -= logs.mean(axis=0)
    return np.stack([expm(value) for value in logs])


def test_v0_failure_artifacts_remain_byte_immutable() -> None:
    expected = {
        "outputs/bnci2014_001_local_gpa_consensus_v0/decisions/technical_failure.json": "984cb0955e3392519fd71636e8da1494d88d62e54ea6249e7d8b06c9788da0ef",
        "outputs/bnci2014_001_local_gpa_consensus_v0/decisions/terminal_decision.json": "79d751e7976352c04b69098c45f33f4e986b5762398b7b04d1c34a4c47cb4cd3",
        "outputs/bnci2014_001_local_gpa_consensus_v0/report/local_gpa_consensus_v0.md": "e97b60c4763fc7447fcf0d24784b56b9bea464c6b0dd892f411ebc74585dec1c",
    }
    for relative, digest in expected.items():
        assert sha256_file(ROOT / relative) == digest


def test_trust_regions_hessian_vector_is_self_adjoint_and_matches_second_difference() -> None:
    rng = np.random.default_rng(44)
    target = _configuration(45)
    source = _configuration(46)
    permutation = np.asarray([1, 4, 0, 3, 2])
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    manifold = Stiefel(4, 4, retraction="polar")
    eta = manifold.random_tangent_vector(q)
    zeta = manifold.random_tangent_vector(q)
    h_eta = riemannian_hessian_vector(target, source, permutation, q, eta)
    h_zeta = riemannian_hessian_vector(target, source, permutation, q, zeta)
    lhs = manifold.inner_product(q, zeta, h_eta)
    rhs = manifold.inner_product(q, eta, h_zeta)
    assert lhs == pytest.approx(rhs, abs=1.0e-8, rel=1.0e-5)
    step = 3.0e-4
    f0 = frozen.fixed_registration_objective(target, source, q, permutation)
    fp = frozen.fixed_registration_objective(
        target, source, manifold.retraction(q, step * eta), permutation
    )
    fm = frozen.fixed_registration_objective(
        target, source, manifold.retraction(q, -step * eta), permutation
    )
    observed = (fp - 2.0 * f0 + fm) / step**2
    expected = manifold.inner_product(q, eta, h_eta)
    assert observed == pytest.approx(expected, abs=1.0e-7, rel=1.0e-4)
    assert HESSIAN_RADIUS == 1.0e-5


@pytest.mark.skipif(not CENTERED.exists(), reason="frozen local-centered cache absent")
def test_exact_failed_registration_and_optimizer_specific_resolution() -> None:
    data = load_and_reproduce_local_gpa_input(ROOT)
    bank = np.load(CENTERED, mmap_mode="r")
    task = cell_tasks(data.metadata)[0]
    source = np.asarray(bank[task.indices[0]])
    target = frozen.feasible_prototype_from_configuration(source)
    assert task.identity == (1, "0train", "left_hand", "Full")
    assert int(task.indices[0]) == 3
    assert array_sha256(target) == "42fbff415a91b8fec4941c827338c4e5057dbd8df0e783c617fbd72662eed8f7"
    assert array_sha256(source) == "324e9aad5de12568fc062f30b528805d24587a1a67884a541cae390024c86a61"
    cg = run_registration_forensic(target, source, solver="ConjugateGradient")
    trust = run_registration_forensic(target, source, solver="TrustRegions")
    assert len(cg.starts) == len(trust.starts) == 4
    for cg_start, trust_start in zip(cg.starts, trust.starts, strict=True):
        assert np.array_equal(cg_start.initial_action, trust_start.initial_action)
        assert np.array_equal(
            cg_start.starting_permutation, trust_start.starting_permutation
        )
    assert [value.converged for value in cg.starts] == [True, False, False, False]
    assert [value.initial_determinant for value in cg.starts] == [1, -1, -1, 1]
    assert not cg.determinant_minus_certified
    assert cg.determinant_plus_certified
    assert all(value.converged for value in trust.starts)
    assert trust.determinant_minus_certified and trust.determinant_plus_certified
    assert TRUST_MAX_ITERATIONS == frozen.CANDIDATE_SETTINGS.action_max_iterations


def test_audit_selection_and_decision_contain_no_scientific_statistic() -> None:
    selection = json.loads(
        (ROOT / "outputs/bnci2014_001_local_gpa_optimizer_audit_v1/protocol/audit_bank_selection.json").read_text()
    )
    decision = json.loads(
        (ROOT / "outputs/bnci2014_001_local_gpa_optimizer_audit_v1/decisions/audit_decision.json").read_text()
    )
    assert len(selection["registrations"]) == 4
    assert not selection["scientific_outcomes_inspected_for_selection"]
    assert not decision["scientific_stage2a_statistics_computed"]
    assert "T_J" not in decision
    assert decision["decision"] == "RECOMMEND_GPA_OPTIMIZER_ONLY_V1_AMENDMENT"
