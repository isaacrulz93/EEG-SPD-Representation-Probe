from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src import stieger2021_multiclass_confirmation_v0 as parent
from src.subject_location_conditional_configuration_v0 import (
    CLASS_NAMES,
    DIRECTIONS,
    FOLDS_CANONICAL_SHA256,
    LOCKED_GEOMETRY_SHA256,
    PARENT_HEAD,
    STATE_ORDER,
    STREAMED_CANONICAL_SHA256,
    _load_input_packet,
    construct_fold_coordinates,
    fit_reduced_rank_ridge,
    load_config,
    load_parent_objects,
    output_root,
    predict_reduced_rank,
    run_primary_predictions,
    select_rank_ridge,
    synthetic_gates,
    validate_output_manifest,
    validate_parent_contract,
    validate_prediction_hashes,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT)[0]


@pytest.fixture(scope="module")
def objects(config):
    return load_parent_objects(ROOT, config)


def test_parent_contract_exact_hashes_and_schema():
    audit = validate_parent_contract(ROOT)
    assert audit["parent_head"] == PARENT_HEAD
    assert audit["locked_geometry_sha256"] == LOCKED_GEOMETRY_SHA256
    assert audit["folds_canonical_sha256"] == FOLDS_CANONICAL_SHA256
    assert audit["streamed_manifest_canonical_sha256"] == STREAMED_CANONICAL_SHA256
    assert audit["subjects"] == list(range(1, 63))
    assert audit["sessions"] == [2, 3]
    assert audit["class_order"] == list(CLASS_NAMES)
    assert audit["marginal_shape"] == [62, 2, 20, 20]
    assert audit["class_mean_shape"] == [62, 2, 4, 20, 20]
    assert audit["raw_eeg_opened"] is False


def test_svec_is_frobenius_isometric_upper_row_major():
    matrix = np.arange(400, dtype=np.float64).reshape(20, 20)
    matrix = 0.5 * (matrix + matrix.T)
    coordinate = parent.svec(matrix)
    indices = np.triu_indices(20)
    expected = matrix[indices].copy()
    expected[indices[0] != indices[1]] *= np.sqrt(2.0)
    assert coordinate.shape == (210,)
    assert np.array_equal(coordinate, expected)
    assert np.dot(coordinate, coordinate) == pytest.approx(np.sum(matrix * matrix), abs=1e-10)


def test_fold_coordinates_are_source_only_and_class_zero_sum(config, objects):
    test = objects.folds[0]
    train = np.setdiff1d(np.arange(62), test)
    fold = construct_fold_coordinates(objects, train, test, 2, 3, config, name="test_outer")
    assert fold["q_train"].shape == (52, 210)
    assert fold["q_test"].shape == (10, 210)
    assert fold["delta_train"].shape == (52, 840)
    assert np.max(np.linalg.norm(fold["delta_train_class"].sum(axis=1), axis=1)) < 2e-10
    assert np.max(np.linalg.norm(fold["delta_test_class"].sum(axis=1), axis=1)) < 2e-10
    assert fold["audit"]["heldout_subjects_excluded_from_m0"]
    assert fold["audit"]["q_uses_labels"] is False

    changed_marginal = objects.marginal.copy()
    changed_classes = objects.class_means.copy()
    changed_references = objects.subject_references.copy()
    changed_marginal[test] *= 7.0
    changed_classes[test] *= 11.0
    changed_references[test] *= 13.0
    changed = replace(
        objects,
        marginal=changed_marginal,
        class_means=changed_classes,
        subject_references=changed_references,
    )
    repeated = construct_fold_coordinates(changed, train, test, 2, 3, config, name="test_outer_perturbed_target")
    assert np.array_equal(fold["m0"], repeated["m0"])
    assert np.array_equal(fold["q_train"], repeated["q_train"])
    assert np.array_equal(fold["delta_train"], repeated["delta_train"])


def test_rank_zero_is_exact_population_residual_predictor():
    rng = np.random.default_rng(17)
    q = rng.normal(size=(20, 6))
    delta = rng.normal(size=(20, 12))
    model = fit_reduced_rank_ridge(q, delta, 0, 1.0)
    prediction = predict_reduced_rank(model, rng.normal(size=(7, 6)))
    assert model.rank == 0
    assert np.array_equal(prediction, np.zeros((7, 12)))


def test_dual_ridge_recovers_known_low_rank_relation(config):
    rng = np.random.default_rng(31)
    q = rng.normal(size=(75, 8))
    delta = q @ rng.normal(size=(8, 2)) @ rng.normal(size=(2, 16))
    indices = np.arange(60)
    inner = []
    for validation in np.array_split(indices, 5):
        train = np.setdiff1d(indices, validation)
        inner.append((q[train], delta[train], q[validation], delta[validation]))
    selection, _ = select_rank_ridge(inner, config)
    model = fit_reduced_rank_ridge(q[:60], delta[:60], selection["effective_rank"], selection["ridge_multiplier"])
    prediction = predict_reduced_rank(model, q[60:])
    r2 = 1.0 - np.sum((delta[60:] - prediction) ** 2) / np.sum(delta[60:] ** 2)
    assert selection["effective_rank"] in (1, 2, 3)
    assert r2 > 0.95


def test_all_registered_synthetic_gates_pass(config):
    result = synthetic_gates(config)
    assert result["passed"]
    assert result["case_count"] == 6
    assert all(case["passed"] for case in result["cases"].values())


def test_prediction_process_has_no_outcome_vault_or_parent_loader():
    source = inspect.getsource(run_primary_predictions)
    assert "outcome_vault" not in source
    assert "load_parent_objects" not in source
    assert "target labels" not in source.lower()


def test_protocol_has_exact_states_directions_and_no_pr20_pr21_import():
    assert tuple(STATE_ORDER) == (
        "PARENT_VALIDATED",
        "PROTOCOL_FROZEN",
        "OBJECTS_LOCKED_NO_TARGET_OUTCOME_ACCESSED",
        "PRIMARY_PREDICTIONS_FROZEN",
        "TARGET_OUTCOMES_RELEASED_FOR_EVALUATION",
        "NULLS_COMPLETE",
        "TERMINAL_WRITTEN",
        "STOPPED",
    )
    assert DIRECTIONS == {"FORWARD": (2, 3), "REVERSE": (3, 2)}
    module_source = (ROOT / "src" / "subject_location_conditional_configuration_v0.py").read_text(encoding="utf-8")
    assert "returning_user_conditional_memory" not in module_source
    assert "selective_conditional_memory" not in module_source
    assert "torch" not in module_source


def test_locked_packets_and_predictions_when_present():
    out = output_root(ROOT)
    locked_manifest = out / "objects" / "locked_packet_manifest.json"
    if locked_manifest.is_file():
        value = json.loads(locked_manifest.read_text(encoding="utf-8"))
        assert value["input_packet_count"] == 12
        assert value["sealed_vault_count"] == 12
        for path in sorted((out / "input_packets").glob("*/fold_??.npz")):
            packet = _load_input_packet(path)
            assert "delta_true" not in packet
            assert "q_test" in packet
    prediction_manifest = out / "predictions" / "prediction_manifest.json"
    if prediction_manifest.is_file():
        assert validate_prediction_hashes(ROOT)["prediction_count"] == 12


def test_final_manifest_when_present():
    if (output_root(ROOT) / "artifact_index.json").is_file():
        result = validate_output_manifest(ROOT)
        assert result["status"] == "PASS"
        assert result["prediction_count"] == 12
