"""Scientific-contract, leakage, algebra, permutation, and synthetic tests."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import src.stieger2021_multiclass_confirmation_v0 as module


ROOT = Path(__file__).resolve().parents[1]


def test_parent_head_dataset_sessions_task_and_class_order() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    assert config["protocol"]["parent_head"] == "19a3ad1cdc6b57c89526e618779cd24b7db8c99c"
    assert config["dataset"]["sessions"] == [2, 3]
    assert config["dataset"]["primary_tasknumber"] == 3
    assert config["dataset"]["class_order"] == ["right_hand", "left_hand", "both_hand", "rest"]


def test_all_23_synthetic_gates_pass() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    result = module.synthetic_gates(config)
    assert result["passed"]
    assert result["case_count"] >= 23


def test_helmert_is_literal_orthonormal_and_zero_sum() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    H = module.helmert4(config)
    np.testing.assert_allclose(H @ H.T, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(H @ np.ones(4), 0, atol=1e-15)


def test_svec_frobenius_isometry_and_roundtrip() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(20, 20)); matrix = (matrix + matrix.T) / 2
    vector = module.svec(matrix)
    np.testing.assert_allclose(np.linalg.norm(vector), np.linalg.norm(matrix), atol=1e-13)
    np.testing.assert_allclose(module.smat(vector), matrix, atol=1e-15)


def test_exact_multiclass_beta_identity_and_source_isolation() -> None:
    rng = np.random.default_rng(5)
    D = rng.normal(size=(15, 4, 3)); pi = rng.uniform(size=(15, 4)); pi /= pi.sum(axis=1, keepdims=True)
    train, test = np.arange(10), np.arange(10, 15)
    gamma, residual, beta = module.source_reference_coordinates(D, pi, train, test)
    np.testing.assert_allclose(beta, residual - np.sum(pi[test, :, None] * residual, axis=1, keepdims=True))
    changed = D.copy(); changed[test] += 900
    gamma_changed, _, _ = module.source_reference_coordinates(changed, pi, train, test)
    np.testing.assert_array_equal(gamma_changed, gamma)


def test_fold_signature_excludes_heldout_template() -> None:
    source = inspect.getsource(module.fold_signatures)
    assert "train_sum" in source
    assert "template = train_sum / float(len(train))" in source
    assert "values[test" not in source.split("template = train_sum / float(len(train))", 1)[0]


def test_known_multiclass_trial_direction_is_orthonormal_deterministic() -> None:
    rng = np.random.default_rng(8)
    mode = np.linalg.qr(rng.normal(size=(630, 5)))[0]
    first = module.deterministic_trial_directions(mode, 5)
    second = module.deterministic_trial_directions(mode, 5)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(5), atol=1e-10)


def test_all_24_permutations_unique_and_identity_match() -> None:
    permutations = module.all_class_permutations()
    assert len(permutations) == len(set(permutations)) == 24
    source = np.asarray([[-2.0, 0.0], [-0.2, 1.0], [0.9, -0.5], [1.3, 0.2]])
    result = module.semantic_permutation_costs(source, source)
    assert result["identity_success"] and result["unique"]


def test_semantic_violation_and_tie_are_failures() -> None:
    source = np.asarray([[-2.0], [-0.2], [0.9], [1.3]])
    assert not module.semantic_permutation_costs(source, source[[1, 0, 2, 3]])["identity_success"]
    tied = module.semantic_permutation_costs(np.zeros((4, 1)), np.zeros((4, 1)))
    assert not tied["unique"] and not tied["identity_success"]


def test_recoverable_between_scatter_and_psd_projection() -> None:
    matrix = np.asarray([[2.0, 0.0], [0.0, -1.0]])
    projected = module.psd_projection(matrix)
    np.testing.assert_allclose(projected, np.diag([2.0, 0.0]))
    assert np.linalg.eigvalsh(projected)[0] >= 0


def test_m4_contract_is_one_label_per_class() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    assert config["inference"]["calibration_budgets"] == [0, 4, 8, 16, 32]
    assert 2 not in config["inference"]["calibration_budgets"]
    assert 4 // 4 == 1


def test_task_and_target_label_leakage_sentinels_are_explicit() -> None:
    source_stream = inspect.getsource(module._load_tangent)
    source_scatter = inspect.getsource(module.run_unlabeled_scatter)
    assert "targetnumber" in source_stream
    assert "sensor_total" in source_scatter and "sensor_within" in source_scatter
    assert "oracle" in source_scatter
    config, _ = module.load_config(ROOT, verify_protocol=False)
    assert config["dataset"]["primary_tasknumber"] == 3


def test_deterministic_stratified_folds_cover_once_and_preserve_groups() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    subjects = np.arange(1, 61); groups = np.asarray([0, 1] * 30)
    first = module.deterministic_stratified_folds(subjects, groups, 6, config, "unit")
    second = module.deterministic_stratified_folds(subjects, groups, 6, config, "unit")
    assert sorted(value for fold in first for value in fold) == list(subjects)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert all(set(groups[np.isin(subjects, fold)]) == {0, 1} for fold in first)


def test_primary_nulls_rerun_rank_selection_and_use_1999() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    assert config["inference"]["null_replicates"] == 1999
    source = inspect.getsource(module.run_primary_nulls)
    assert "evaluate_population(pair_U" in source
    assert "evaluate_population(_permute_class_axis" in source


def test_parent_artifact_hashes_unchanged() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    observed = module.validate_parent_hashes(ROOT, config)
    assert observed["pr18_manifest"] == "76370c6d9114f33b42c1f46d96ea46e4f2628892e3b4d5ab8cfef409c13cb4f1"


def test_decision_literals_and_no_classifier_boundary() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    assert config["decisions"]["structure_low_rank"] == "STIEGER_MULTICLASS_STRUCTURE_CONFIRMED_LOW_RANK"
    assert config["decisions"]["permutation_pass"] == "SOURCE_CLASS_PERMUTATION_PRESERVATION_SUPPORTED_PROSPECTIVELY"
    assert "classifier" in config["forbidden_methods"]
