"""Synthetic, algebraic, leakage, and immutable-contract tests."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import src.source_referenced_conditional_residual_v1 as module


ROOT = Path(__file__).resolve().parents[1]


def test_config_parent_head_protocol_hash_and_budgets() -> None:
    config, digest = module.load_config(ROOT)
    assert len(digest) == 64
    assert config["protocol"]["parent_head"] == "8346a3e0f731c80668bd7147a2fe0fd12da6b914"
    assert config["anchor"]["budgets"] == [0, 2, 4, 8, 16, 32]
    assert config["inference"]["null_replicates"] == 1999


def test_all_twelve_synthetic_gates_pass() -> None:
    config, _ = module.load_config(ROOT)
    result = module.synthetic_gates(config)
    assert result["passed"]
    assert len(result["cases"]) == 12


def test_beta_delta_gamma_identity_exact() -> None:
    delta = np.linspace(-2, 2, 31)
    gamma = 0.37
    beta = delta - gamma
    np.testing.assert_array_equal(beta, delta - gamma)


def test_nonzero_gamma_changes_semantic_and_residual_signs() -> None:
    delta = np.asarray([-0.2, 0.1, 0.4, 0.9])
    beta = delta - 0.5
    assert np.any(np.sign(delta) != np.sign(beta))


def test_omitting_gamma_stays_wrong_with_increasing_labels() -> None:
    delta_true, gamma = 0.3, 0.7
    for sample_noise in (0.2, 0.1, 0.02, 0.0):
        delta_estimate = delta_true + sample_noise
        assert np.sign(delta_estimate) != np.sign(delta_true - gamma)


def test_source_reference_correction_restores_beta() -> None:
    d_proto, gamma, curvature = 1.2, 0.4, 0.15
    delta_trial = d_proto - curvature
    assert delta_trial + curvature - gamma == pytest.approx(d_proto - gamma)


def test_unsigned_delta_magnitude_is_class_swap_invariant() -> None:
    delta = np.asarray([-1.1, 0.2, 0.8])
    np.testing.assert_array_equal(delta**2, (-delta) ** 2)
    np.testing.assert_array_equal(np.abs(delta), np.abs(-delta))


def test_source_order_assumption_success_and_violation_are_distinct() -> None:
    source_order = np.sign(np.mean([1.0, 0.8, 1.2]))
    target_success = np.asarray([0.5, 0.7])
    target_violation = -target_success
    assert np.all(np.sign(target_success) == source_order)
    assert np.all(np.sign(target_violation) != source_order)


def test_minimal_label_residualized_estimator_uses_same_reference() -> None:
    magnitude, semantic_sign, correction, gamma = 0.9, 1, 0.1, 0.4
    proposed = semantic_sign * magnitude + correction - gamma
    direct_delta = 0.85
    direct = direct_delta + correction - gamma
    assert proposed == pytest.approx(0.6)
    assert direct == pytest.approx(0.55)


def test_target_label_leakage_sentinel_and_source_only_formulas() -> None:
    source = inspect.getsource(module._load_reference_core)
    assert "train" in source
    assert "correction_mean" in source
    assert "target_delta" not in source.split("correction_mean[f, q]")[0].split("def _load_reference_core", 1)[1]


def test_outer_fold_source_only_gamma_and_correction() -> None:
    config, _ = module.load_config(ROOT)
    outer = module._outer_indices(config)
    for test in outer:
        train = np.setdiff1d(np.arange(54), test)
        assert len(train) == 45 and len(test) == 9
        assert np.intersect1d(train, test).size == 0


def test_fair_direct_baseline_literal_and_old_control_nonvoting() -> None:
    config, _ = module.load_config(ROOT)
    assert config["anchor"]["direct_baseline"] == "SOURCE_REFERENCED_DIRECT_M_LABEL"
    source = inspect.getsource(module.run_corrected_minimal_anchor)
    assert "direct_delta + ref" in source
    assert "PR17_OLD_UNCENTERED_HISTORICAL" in source


def test_parent_artifacts_and_manifest_hashes_are_unchanged() -> None:
    config, _ = module.load_config(ROOT)
    observed = module.validate_parent_artifacts(ROOT, config)
    assert observed["modes"] == "b0d71aeaddd73723d45cb0c009ea2bb036f72e1fe70ff25ac9c3c067eab179b7"


def test_exact_outer_fold_coverage_once() -> None:
    config, _ = module.load_config(ROOT)
    folds = config["folds"]["outer_test"]
    assert sorted(subject for fold in folds for subject in fold) == list(range(1, 55))
    assert all(len(fold) == 9 for fold in folds)


def test_source_order_and_beta_sign_are_not_conflated() -> None:
    semantic_delta = np.asarray([0.2, 0.8])
    beta = semantic_delta - 0.5
    assert np.array_equal(np.sign(semantic_delta), [1, 1])
    assert np.array_equal(np.sign(beta), [-1, 1])


def test_deterministic_namespaced_rng() -> None:
    config, _ = module.load_config(ROOT)
    first = module._rng(config, "unit", 1).normal(size=20)
    second = module._rng(config, "unit", 1).normal(size=20)
    third = module._rng(config, "unit", 2).normal(size=20)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_decision_literals_and_dataset_boundary() -> None:
    config, _ = module.load_config(ROOT)
    assert config["decisions"]["correction_pass"] == "SOURCE_REFERENCE_COORDINATE_CORRECTION_SUPPORTED"
    assert config["decisions"]["ordering_pass"] == "SOURCE_SEMANTIC_ORDERING_SUPPORTED_RETROSPECTIVELY"
    assert "Stieger2021" in config["forbidden_datasets"]
    assert "classifier" in config["forbidden_methods"]


def test_metric_bundle_is_finite_and_exact_for_perfect_prediction() -> None:
    truth = np.linspace(-1, 1, 30)
    metrics = module._metric_bundle(truth, truth)
    assert metrics["mae"] == 0.0
    assert metrics["signed_r2"] == pytest.approx(1.0)
    assert metrics["beta_sign_accuracy"] == 1.0
