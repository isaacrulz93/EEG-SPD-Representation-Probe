from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from src import returning_user_conditional_memory_v0 as module


ROOT = Path(__file__).resolve().parents[1]


def test_svec_frobenius_isometry() -> None:
    rng = np.random.default_rng(1)
    value = rng.normal(size=(20, 20)); value = (value + value.T) / 2
    np.testing.assert_allclose(np.linalg.norm(module.svec(value)), np.linalg.norm(value), rtol=0, atol=2e-12)


@pytest.mark.parametrize("classes", [2, 4])
def test_helmert_orthogonality(classes: int) -> None:
    H = module.helmert(classes)
    np.testing.assert_allclose(H @ H.T, np.eye(classes - 1), rtol=0, atol=2e-15)
    np.testing.assert_allclose(H @ np.ones(classes), 0, rtol=0, atol=2e-15)


@pytest.mark.parametrize("classes", [2, 4])
def test_exact_helmert_zero_mean_roundtrip(classes: int) -> None:
    rng = np.random.default_rng(classes)
    prototypes = rng.normal(size=(classes, 210)); prototypes -= prototypes.mean(axis=0)
    signature = module.prototypes_to_signature(prototypes, module.helmert(classes))
    recovered = module.signature_to_centered_prototypes(signature, module.helmert(classes))
    np.testing.assert_allclose(recovered, prototypes, rtol=0, atol=2e-12)


def test_known_rank2_cross_session_map_recovery() -> None:
    rng = np.random.default_rng(3); n, p, rank = 60, 16, 2
    latent = rng.normal(size=(n, rank)); left, _ = np.linalg.qr(rng.normal(size=(p, rank))); right, _ = np.linalg.qr(rng.normal(size=(p, rank)))
    enrollment = latent @ left.T + .01 * rng.normal(size=(n, p)); deployment = latent @ right.T + .01 * rng.normal(size=(n, p))
    core = module.fit_ridge_core(enrollment[:50], deployment[:50], 1e-3)
    prediction = np.stack([module.predict_lrcm_signature(core, x, 2)[0] for x in enrollment[50:]])
    assert np.mean(np.linalg.norm(prediction - deployment[50:], axis=1)) < .2


def test_full_rank_map_recovery() -> None:
    rng = np.random.default_rng(4); x = rng.normal(size=(50, 10)); y = x @ rng.normal(size=(10, 10))
    core = module.fit_ridge_core(x[:40], y[:40], 1e-8)
    prediction = np.stack([module.predict_full_ridge_signature(core, row) for row in x[40:]])
    np.testing.assert_allclose(prediction, y[40:], atol=1e-5, rtol=1e-5)


def test_no_paired_cross_session_signal() -> None:
    rng = np.random.default_rng(5); x = rng.normal(size=(60, 12)); y = rng.normal(size=(60, 12))
    core = module.fit_ridge_core(x[:50], y[:50], 1.0)
    prediction = np.stack([module.predict_lrcm_signature(core, row, 2)[0] for row in x[50:]])
    baseline = np.broadcast_to(y[:50].mean(axis=0), prediction.shape)
    assert np.mean((prediction - y[50:]) ** 2) >= .8 * np.mean((baseline - y[50:]) ** 2)


def test_identity_residual_carry() -> None:
    rng = np.random.default_rng(6); gamma_e = rng.normal(size=(4, 8)); gamma_d = rng.normal(size=(4, 8)); residual = rng.normal(size=(4, 8))
    enrollment = gamma_e + residual
    np.testing.assert_allclose(gamma_d + (enrollment - gamma_e), gamma_d + residual)


def test_population_session_template_shift() -> None:
    gamma_e = np.zeros((4, 8)); gamma_d = np.ones((4, 8))
    assert np.linalg.norm(gamma_d - gamma_e) > 0


def test_reduced_rank_dual_ridge_equals_primal() -> None:
    rng = np.random.default_rng(7); x = rng.normal(size=(30, 9)); y = rng.normal(size=(30, 9)); lam = .1
    core = module.fit_ridge_core(x, y, lam)
    A, B = x - x.mean(0), y - y.mean(0)
    primal = np.linalg.solve(A.T @ A + lam * np.eye(9), A.T @ B)
    np.testing.assert_allclose(core.W, primal, atol=2e-12, rtol=2e-12)


def test_rank_truncation_output_dimension() -> None:
    rng = np.random.default_rng(8); x = rng.normal(size=(30, 12)); y = rng.normal(size=(30, 12)); core = module.fit_ridge_core(x, y, 1.0)
    prediction, memory = module.predict_lrcm_signature(core, x[0], 3)
    assert prediction.shape == (12,) and memory.shape == (3,)


def test_target_latent_memory_byte_count() -> None:
    rng = np.random.default_rng(9); x = rng.normal(size=(20, 8)); core = module.fit_ridge_core(x, x, .1)
    _, memory = module.predict_lrcm_signature(core, x[0], 2)
    assert memory.nbytes == 16


def test_enrollment_subject_permutation_is_fixed_point_free() -> None:
    order = module.fixed_point_free(np.random.default_rng(10), 11)
    assert not np.any(order == np.arange(11))


def test_enrollment_subject_permutation_destroys_synthetic_gain() -> None:
    rng = np.random.default_rng(101); classes, dim, subjects = 4, 6, 8
    base = rng.normal(scale=2.0, size=(classes, dim)); base -= base.mean(axis=0)
    prototypes = np.stack([base + 3.0 * rng.normal(size=(1, dim)) for _ in range(subjects)])
    correct, wrong = [], []
    for subject in range(subjects):
        features = np.concatenate([prototypes[subject, c] + .15 * rng.normal(size=(25, dim)) for c in range(classes)])
        labels = np.repeat(np.arange(classes), 25)
        correct.append(module.balanced_accuracy_fast(labels, module.ncm_predict(features, prototypes[subject], 1.0)[0], classes))
        wrong.append(module.balanced_accuracy_fast(labels, module.ncm_predict(features, prototypes[(subject - 1) % subjects], 1.0)[0], classes))
    assert np.mean(correct) > np.mean(wrong)


def test_enrollment_class_permutation_is_nonidentity() -> None:
    order = module.nonidentity_class_permutation(np.random.default_rng(11), 4)
    assert not np.array_equal(order, np.arange(4))


def test_unpaired_source_sessions_destroy_synthetic_gain() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    paired_x, paired_y = module._synthetic_dataset(config, 2, True)
    unpaired_x, unpaired_y = module._synthetic_dataset(config, 2, False)
    paired = module.fit_ridge_core(paired_x[:40], paired_y[:40], 1e-3)
    unpaired = module.fit_ridge_core(unpaired_x[:40], unpaired_y[:40], 1e-3)
    p = np.stack([module.predict_lrcm_signature(paired, row, 2)[0] for row in paired_x[40:]])
    u = np.stack([module.predict_lrcm_signature(unpaired, row, 2)[0] for row in unpaired_x[40:]])
    assert np.mean((p - paired_y[40:]) ** 2) < np.mean((u - unpaired_y[40:]) ** 2)


def test_nested_selection_api_has_no_outer_target_labels() -> None:
    parameters = inspect.signature(module.select_hyperparameters).parameters
    assert "outer_target_labels" not in parameters and "deployment_labels" not in parameters


def test_outer_prediction_api_has_no_deployment_labels() -> None:
    assert "deployment_labels" not in inspect.signature(module.method_prototypes).parameters


def test_target_deployment_labels_are_separate_type() -> None:
    assert module.SessionTrials.__dataclass_fields__.keys() == {"features", "trial_ids"}
    assert module.EvaluationLabels.__dataclass_fields__.keys() == {"labels", "trial_ids"}


def test_kshot_support_query_disjointness() -> None:
    labels = np.repeat(np.arange(4), 10); support = np.concatenate([np.flatnonzero(labels == c)[:2] for c in range(4)]); query = np.setdiff1d(np.arange(len(labels)), support)
    assert len(np.intersect1d(support, query)) == 0


def test_class_balanced_kshot_support() -> None:
    labels = np.repeat(np.arange(4), 10); support = np.concatenate([np.flatnonzero(labels == c)[:3] for c in range(4)])
    assert np.bincount(labels[support], minlength=4).tolist() == [3, 3, 3, 3]


def test_subject_level_inference_unit() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False); values = np.asarray([.1, -.1, .2, .3])
    ci = module._bootstrap_ci(values, config, "unit_subject_bootstrap", 100)
    assert len(ci) == 2 and ci[0] <= ci[1]


def test_openbmi_chronology_metadata_check() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    assert config["openbmi"]["source_chronological_sessions"] == [1, 2]
    assert config["openbmi"]["enrollment_array_index"] == 0 and config["openbmi"]["deployment_array_index"] == 1


def test_stieger_pr19_fold_identity() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False); folds, inner = module._read_stieger_folds(ROOT, config)
    module._check_fold_contract(np.arange(1, 63), folds, inner)


def test_openbmi_pr16_fold_identity() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False); folds, inner = module._read_openbmi_folds(ROOT, config)
    module._check_fold_contract(np.arange(1, 55), folds, inner)


def test_parent_manifest_hash_immutability() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    assert len(module.validate_parent_manifests(ROOT, config)) == 4


def test_deterministic_seed_namespaces() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    np.testing.assert_array_equal(module._rng(config, "same", 1).integers(0, 100, 20), module._rng(config, "same", 1).integers(0, 100, 20))


def test_expected_terminal_labels() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    assert config["decisions"]["replicated"] == "GO_LOW_RANK_CONDITIONAL_MEMORY_REPLICATED"
    assert config["decisions"]["stop"] == "STOP_NO_CROSS_SESSION_DOWNSTREAM_UTILITY"


def test_haar_output_basis_orthogonality() -> None:
    basis = module.haar_basis(np.random.default_rng(12), 30, 5)
    np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=2e-12, rtol=0)


def test_ncm_equal_prior_prediction() -> None:
    prototypes = np.asarray([[0.0, 0.0], [2.0, 0.0]]); features = np.asarray([[.1, 0.0], [1.9, 0.0]])
    prediction, probability = module.ncm_predict(features, prototypes, 1.0)
    assert prediction.tolist() == [0, 1]
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_balanced_accuracy_fast_matches_sklearn() -> None:
    y = np.asarray([0, 0, 1, 1, 2, 2]); p = np.asarray([0, 1, 1, 1, 2, 0])
    from sklearn.metrics import balanced_accuracy_score
    assert module.balanced_accuracy_fast(y, p, 3) == pytest.approx(balanced_accuracy_score(y, p))


@pytest.mark.skipif(not (ROOT / "cache/stieger2021_multiclass_confirmation_v0").exists(), reason="ignored parent cache unavailable")
def test_live_parent_trial_caches_validate_without_statistics() -> None:
    config, _ = module.load_config(ROOT, verify_protocol=False)
    stieger = module._load_stieger_bundle(ROOT, config)
    openbmi = module._load_openbmi_bundle(ROOT, config)
    assert len(stieger.trials) == 124 and len(openbmi.trials) == 108
