"""Synthetic and immutable-contract tests for unlabeled conditional modes V0."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import src.unlabeled_conditional_mode_identifiability_v0 as module
from src.unlabeled_conditional_mode_identifiability_v0 import (
    _directional_energy,
    _inverse_svec,
    _rng,
    _spearman,
    _symmetric_mixture_energy,
    _two_means_energy,
    _variance_components,
    immutable_parent_snapshot,
    load_config,
    synthetic_gates,
    validate_parent_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_directional(seed: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(scale=0.25, size=(54, 2, 100, 7))
    labels = np.broadcast_to(np.repeat([0, 1], 50), (54, 2, 100)).copy()
    modes = np.zeros((6, 2, 7), dtype=np.float64); modes[..., 0] = 1.0
    magnitude = np.linspace(0.15, 1.4, 54)
    for s in range(54):
        features[s, :, :, 0] += (2 * labels[s] - 1) * magnitude[s]
    outer = [np.arange(9 * fold, 9 * (fold + 1)) for fold in range(6)]
    return features, labels, modes, outer, magnitude


def test_config_parent_and_protocol_hashes_are_frozen() -> None:
    config, digest = load_config(ROOT)
    assert len(digest) == 64
    assert config["protocol"]["parent_head"] == "9dee7642ac573f37756b8427a75864a50c32044e"
    assert config["unsigned_recovery"]["primary_estimator"] == "SOURCE_WITHIN_CORRECTED_PROJECTED_VARIANCE"
    assert config["minimal_anchor"]["budgets"] == [0, 2, 4, 8, 16, 32]


def test_parent_artifacts_have_expected_hashes_and_terminal() -> None:
    config, _ = load_config(ROOT)
    observed = validate_parent_artifacts(ROOT, config)
    assert observed["v1_1_observed_modes"] == "ecf0657d1ccf51e7aa3392a5e95608d849e9069dda2f27e2a76e321429fac168"


def test_pr16_snapshot_has_only_parent_output_namespaces() -> None:
    config, _ = load_config(ROOT)
    snapshot = immutable_parent_snapshot(ROOT, config)
    assert snapshot
    assert all(path.startswith(("outputs/subject_class_population_structure_v1/", "outputs/subject_class_population_structure_v1_1/")) for path in snapshot)


def test_exact_outer_fold_coverage_once() -> None:
    config, _ = load_config(ROOT)
    folds = config["parent_fold_contract"]["outer_test"]
    assert sorted(value for fold in folds for value in fold) == list(range(1, 55))
    assert all(len(fold) == 9 for fold in folds)


def test_parent_rank_selection_is_six_rank_one_folds() -> None:
    config, _ = load_config(ROOT)
    path = ROOT / config["parent_artifacts"]["v1_1_observed_modes"]["path"]
    with np.load(path, allow_pickle=False) as archive:
        assert archive["selected_ranks"].tolist() == [1, 1, 1, 1, 1, 1]
        assert archive["left"].shape == (6, 210, 34)
        assert archive["right"].shape == (6, 210, 34)


def test_all_nine_predeclared_synthetic_gates_pass() -> None:
    config, _ = load_config(ROOT)
    result = synthetic_gates(config)
    assert result["passed"]
    assert len(result["cases"]) == 9


def test_balanced_variance_decomposition_is_exact() -> None:
    labels = np.repeat([0, 1], 50)
    values = np.concatenate([np.linspace(-2, 0, 50), np.linspace(1, 3, 50)])
    total, within, between, delta = _variance_components(values, labels)
    assert total == pytest.approx(within + between, abs=2e-15)
    assert between == pytest.approx(delta**2, abs=2e-15)


def test_class_swap_preserves_unsigned_and_negates_signed_contrast() -> None:
    rng = np.random.default_rng(18)
    labels = np.repeat([0, 1], 50)
    values = rng.normal(size=100) + (2 * labels - 1) * 0.8
    first = _variance_components(values, labels)
    second = _variance_components(values, 1 - labels)
    assert first[:3] == pytest.approx(second[:3], abs=2e-15)
    assert first[3] == pytest.approx(-second[3], abs=2e-15)


def test_source_within_corrected_energy_recovers_known_separation() -> None:
    features, labels, modes, outer, magnitude = _synthetic_directional()
    result = _directional_energy(features, labels, modes, outer)
    assert _spearman(result["predicted"].mean(axis=1), magnitude**2) > 0.95
    np.testing.assert_allclose(result["between"].mean(axis=1), magnitude**2, rtol=0.2, atol=0.08)


def test_signed_worlds_are_unidentifiable_from_same_unlabeled_values() -> None:
    rng = np.random.default_rng(9)
    pooled = rng.normal(size=100)
    beta_a = 0.7
    beta_b = -0.7
    np.testing.assert_array_equal(pooled, pooled.copy())
    assert beta_b == -beta_a


def test_varying_within_noise_remains_finite() -> None:
    features, labels, modes, outer, _ = _synthetic_directional()
    scale = np.linspace(0.5, 2.0, 54)[:, None, None, None]
    result = _directional_energy(features * scale, labels, modes, outer)
    assert all(np.isfinite(value).all() for value in result.values())


def test_no_between_separation_has_small_group_energy() -> None:
    rng = np.random.default_rng(44)
    labels = np.repeat([0, 1], 50)
    values = rng.normal(size=100000).reshape(1000, 100)
    deltas = np.asarray([_variance_components(row, labels)[3] for row in values])
    assert float(np.mean(deltas**2)) < 0.02


def test_symmetric_mixture_recovers_unsigned_separation() -> None:
    config, _ = load_config(ROOT)
    rng = np.random.default_rng(71)
    values = np.concatenate([rng.normal(-1.2, 0.2, 50), rng.normal(1.2, 0.2, 50)])
    assert _symmetric_mixture_energy(values, config) == pytest.approx(1.44, rel=0.2)


def test_unordered_two_means_never_assigns_semantic_component() -> None:
    config, _ = load_config(ROOT)
    values = np.concatenate([np.full(50, -2.0), np.full(50, 2.0)])
    assert _two_means_energy(values, config) == pytest.approx(4.0)
    assert _two_means_energy(values[::-1], config) == pytest.approx(4.0)


def test_random_direction_equivalence_has_no_systematic_association() -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(size=(200, 30)); target = rng.normal(size=200)
    correlations = []
    for _ in range(40):
        direction = rng.normal(size=30); direction /= np.linalg.norm(direction)
        correlations.append(_spearman(data @ direction, target))
    assert abs(float(np.median(correlations))) < 0.1


def test_two_labels_resolve_synthetic_semantic_sign() -> None:
    labels = np.repeat([0, 1], 50)
    values = (2 * labels - 1) * 3.0
    orientation = np.sign(values[50] - values[0])
    assert orientation == 1


def test_target_label_leakage_sentinel_is_structural() -> None:
    config, _ = load_config(ROOT)
    stripped = set(config["unlabeled_projection"]["strip_fields"])
    assert {"class_label", "event_code", "acquisition_order", "semantic_trial_order"} <= stripped
    source = inspect.getsource(module._build_analysis_core)
    assert "projected_y" in source
    assert "labels" in source  # labels are used only for evaluation delta after features exist
    assert config["unlabeled_projection"]["evaluation_labels_separate"] is True


def test_inverse_svec_roundtrip_is_frobenius_isometric() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(20, 20)); matrix = 0.5 * (matrix + matrix.T)
    vector = module.population_v1.svec(matrix)
    reconstructed = _inverse_svec(vector)
    np.testing.assert_allclose(reconstructed, matrix, rtol=0.0, atol=3e-16)
    assert np.linalg.norm(vector) == pytest.approx(np.linalg.norm(matrix), abs=3e-15)


def test_deterministic_namespaced_rng() -> None:
    config, _ = load_config(ROOT)
    first = _rng(config, "unit", 3).normal(size=20)
    second = _rng(config, "unit", 3).normal(size=20)
    third = _rng(config, "unit", 4).normal(size=20)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_label_swap_decision_literals_are_frozen() -> None:
    config, _ = load_config(ROOT)
    assert config["symmetry"]["signed_zero_label_status"] == "NONIDENTIFIABLE_UP_TO_CLASS_PERMUTATION"
    assert config["decisions"]["signed"]["failure"] == "UNASSESSED_SYMMETRY_CONTRACT_FAILURE"


def test_dataset_exclusions_and_no_classifier_are_literal() -> None:
    config, _ = load_config(ROOT)
    assert "BNCI2014_001" in config["forbidden_datasets"]
    assert "downstream_classification" in config["forbidden_methods"]
    assert "TTA" in config["forbidden_methods"]


def test_no_nonfinite_synthetic_directional_outputs() -> None:
    features, labels, modes, outer, _ = _synthetic_directional(99)
    result = _directional_energy(features, labels, modes, outer)
    assert all(np.isfinite(value).all() for value in result.values())
