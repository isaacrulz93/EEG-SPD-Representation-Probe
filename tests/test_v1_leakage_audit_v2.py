"""Synthetic tests for the gate-protected V1 centering leakage audit."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.alignment_v2 import label_free_metadata_view
from src.leakage_audit_v2 import (
    V1_BENCHMARK_ACCURACY,
    GeometryGateError,
    apply_subject_log_means,
    fit_subject_log_means,
    require_geometry_gate,
    run_v1_leakage_audit,
)


CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")


def _synthetic_whole() -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(20260809)
    class_patterns = {
        "left_hand": np.array([-2.5, 0.0, -0.4]),
        "right_hand": np.array([2.5, 0.0, -0.4]),
        "feet": np.array([0.0, -2.5, 0.4]),
        "tongue": np.array([0.0, 2.5, 0.4]),
    }
    matrices: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    sample_index = 0
    for subject in range(1, 5):
        trial_id = 0
        subject_offset = np.array([0.7 * subject, -0.35 * subject, 0.2 * subject])
        for class_label in CLASS_ORDER:
            for repetition in range(10):
                trial_id += 1
                log_diagonal = (
                    class_patterns[class_label]
                    + subject_offset
                    + rng.normal(scale=0.10, size=3)
                )
                matrices.append(np.diag(np.exp(log_diagonal)))
                rows.append(
                    {
                        "sample_index": sample_index,
                        "subject": subject,
                        "session": "0train",
                        "run": str(repetition % 2),
                        "trial_id": trial_id,
                        "trial_uid": f"S{subject:02d}_T{trial_id:03d}",
                        "class_label": class_label,
                    }
                )
                sample_index += 1
    return np.stack(matrices), pd.DataFrame.from_records(rows)


@pytest.fixture(scope="module")
def synthetic() -> tuple[np.ndarray, pd.DataFrame]:
    return _synthetic_whole()


@pytest.fixture(scope="module")
def audit_result(synthetic):
    covariances, metadata = synthetic
    return run_v1_leakage_audit(covariances, metadata)


def _write_gate_files(
    tmp_path: Path,
    *,
    classification_gate_pass: object = True,
    required: int = 3,
    passed: int = 3,
    failed: int = 0,
) -> tuple[Path, Path]:
    gate = tmp_path / "geometry_gate.json"
    correctness = tmp_path / "geometry_correctness.csv"
    payload = {
        "classification_gate_pass": classification_gate_pass,
        "protocol_version": "2.0",
        "protocol_sha256": "p" * 64,
        "config_sha256": "c" * 64,
        "required_rows": required,
        "passed_required_rows": passed,
        "failed_required_rows": failed,
    }
    gate.write_text(json.dumps(payload), encoding="utf-8")
    pd.DataFrame(
        {
            "check": [f"check_{index}" for index in range(required)],
            "required": [True] * required,
            "passed": [index < passed for index in range(required)],
            "status": [
                "PASS" if index < passed else "FAIL" for index in range(required)
            ],
        }
    ).to_csv(correctness, index=False)
    return gate, correctness


def test_01_center_fit_api_cannot_accept_labels() -> None:
    parameters = set(inspect.signature(fit_subject_log_means).parameters)
    assert parameters == {"raw_log_svec", "fit_metadata"}
    assert not {"class_label", "labels", "target", "y"}.intersection(parameters)


def test_02_label_free_subject_means_center_each_subject(synthetic) -> None:
    covariances, metadata = synthetic
    raw_log_diagonal = np.log(np.diagonal(covariances, axis1=1, axis2=2))
    view = label_free_metadata_view(metadata)
    means = fit_subject_log_means(raw_log_diagonal, view)
    centered = apply_subject_log_means(raw_log_diagonal, view, means)
    view_subjects = np.asarray(view.subject)
    for subject in means.subjects:
        np.testing.assert_allclose(
            centered[view_subjects == subject].mean(axis=0),
            0.0,
            atol=1e-12,
            rtol=0.0,
        )
    assert means.fit_counts == (40, 40, 40, 40)


def test_03_geometry_gate_accepts_only_consistent_exact_true(tmp_path: Path) -> None:
    gate, correctness = _write_gate_files(tmp_path)
    payload = require_geometry_gate(
        gate,
        correctness,
        expected_protocol_version="2.0",
        expected_protocol_sha256="p" * 64,
        expected_config_sha256="c" * 64,
    )
    assert payload["classification_gate_pass"] is True
    assert len(payload["gate_json_sha256"]) == 64
    assert len(payload["geometry_correctness_csv_sha256"]) == 64


def test_04_geometry_gate_rejects_missing_invalid_or_false(tmp_path: Path) -> None:
    missing_gate = tmp_path / "missing.json"
    correctness = tmp_path / "geometry_correctness.csv"
    pd.DataFrame({"check": ["finite"]}).to_csv(correctness, index=False)
    with pytest.raises(GeometryGateError, match="missing"):
        require_geometry_gate(
            missing_gate,
            correctness,
            expected_protocol_version="2.0",
            expected_protocol_sha256="p" * 64,
            expected_config_sha256="c" * 64,
        )

    gate, correctness = _write_gate_files(
        tmp_path, classification_gate_pass="true"
    )
    with pytest.raises(GeometryGateError, match="not exactly true"):
        require_geometry_gate(
            gate,
            correctness,
            expected_protocol_version="2.0",
            expected_protocol_sha256="p" * 64,
            expected_config_sha256="c" * 64,
        )
    gate.write_text("not-json", encoding="utf-8")
    with pytest.raises(GeometryGateError, match="invalid"):
        require_geometry_gate(
            gate,
            correctness,
            expected_protocol_version="2.0",
            expected_protocol_sha256="p" * 64,
            expected_config_sha256="c" * 64,
        )


def test_05_geometry_gate_rejects_failed_counts_or_missing_csv(
    tmp_path: Path,
) -> None:
    gate, correctness = _write_gate_files(tmp_path, passed=2, failed=1)
    with pytest.raises(GeometryGateError, match="all-pass"):
        require_geometry_gate(
            gate,
            correctness,
            expected_protocol_version="2.0",
            expected_protocol_sha256="p" * 64,
            expected_config_sha256="c" * 64,
        )


def test_05b_geometry_gate_rejects_csv_failure_despite_true_json(
    tmp_path: Path,
) -> None:
    gate, correctness = _write_gate_files(tmp_path)
    frame = pd.read_csv(correctness)
    frame.loc[0, ["passed", "status"]] = [False, "FAIL"]
    frame.to_csv(correctness, index=False)
    with pytest.raises(GeometryGateError, match="did not PASS"):
        require_geometry_gate(
            gate,
            correctness,
            expected_protocol_version="2.0",
            expected_protocol_sha256="p" * 64,
            expected_config_sha256="c" * 64,
        )
    gate, correctness = _write_gate_files(tmp_path)
    correctness.unlink()
    with pytest.raises(GeometryGateError, match="CSV is missing"):
        require_geometry_gate(
            gate,
            correctness,
            expected_protocol_version="2.0",
            expected_protocol_sha256="p" * 64,
            expected_config_sha256="c" * 64,
        )


def test_06_audit_has_five_folds_and_one_pooled_row_per_condition(
    audit_result,
) -> None:
    table = audit_result.table
    assert len(table) == 12
    for condition in ("v1_all_sample", "fold_safe"):
        selected = table[table.condition == condition]
        assert set(selected[selected.row_type == "fold"].fold.astype(int)) == set(
            range(5)
        )
        assert len(selected[selected.row_type == "pooled_oof"]) == 1
    assert set(audit_result.fold_assignments) == set(range(5))


def test_07_all_sample_overlap_and_fold_safe_disjointness_are_explicit(
    audit_result,
) -> None:
    folds = audit_result.table[audit_result.table.row_type == "fold"]
    all_sample = folds[folds.condition == "v1_all_sample"]
    fold_safe = folds[folds.condition == "fold_safe"]
    assert (
        all_sample.center_evaluation_overlap_n.to_numpy()
        == all_sample.evaluation_n.to_numpy()
    ).all()
    assert (all_sample.transductive_covariate_overlap == True).all()  # noqa: E712
    assert (fold_safe.center_evaluation_overlap_n == 0).all()
    assert (fold_safe.transductive_covariate_overlap == False).all()  # noqa: E712
    assert (folds.train_evaluation_overlap_n == 0).all()
    assert (folds.train_evaluation_disjoint == True).all()  # noqa: E712


def test_08_fold_safe_center_fit_identity_equals_classifier_training_identity(
    audit_result,
) -> None:
    fold_safe = audit_result.table[
        (audit_result.table.condition == "fold_safe")
        & (audit_result.table.row_type == "fold")
    ]
    assert (
        fold_safe.center_fit_trial_uid_sha256
        == fold_safe.classifier_train_trial_uid_sha256
    ).all()
    assert (fold_safe.center_fit_n == fold_safe.classifier_train_n).all()
    assert (fold_safe.center_fit_min_per_subject == 32).all()
    assert (fold_safe.center_fit_max_per_subject == 32).all()


def test_09_all_sample_reuses_one_full_center_fit_set_across_folds(
    audit_result,
) -> None:
    all_sample = audit_result.table[
        (audit_result.table.condition == "v1_all_sample")
        & (audit_result.table.row_type == "fold")
    ]
    assert all_sample.center_fit_trial_uid_sha256.nunique() == 1
    assert (all_sample.center_fit_n == 160).all()
    assert (all_sample.center_fit_min_per_subject == 40).all()
    assert (all_sample.center_fit_max_per_subject == 40).all()


def test_10_accuracy_pipeline_is_complete_and_linearly_accessible(audit_result) -> None:
    pooled = audit_result.table[audit_result.table.row_type == "pooled_oof"]
    assert (pooled.balanced_accuracy > 0.95).all()
    assert (pooled.accuracy > 0.95).all()
    assert (pooled.macro_f1 > 0.95).all()
    assert (pooled.classifier_status == "PASS").all()
    assert not pooled.convergence_warning.any()
    for prediction in audit_result.oof_predictions.values():
        assert len(prediction) == 160
        assert not prediction.flags.writeable


def test_11_metrics_include_recalls_and_full_confusion(audit_result) -> None:
    pooled = audit_result.table[audit_result.table.row_type == "pooled_oof"]
    for _, row in pooled.iterrows():
        for class_label in CLASS_ORDER:
            assert 0.0 <= row[f"recall_{class_label}"] <= 1.0
        confusion_columns = [
            column for column in pooled.columns if column.startswith("confusion_")
        ]
        numeric_confusion = [
            column for column in confusion_columns if column != "confusion_matrix_json"
        ]
        assert len(numeric_confusion) == 16
        assert sum(int(row[column]) for column in numeric_confusion) == 160


def test_12_pooled_rows_record_benchmark_difference_and_aggregate_hashes(
    audit_result,
) -> None:
    pooled = audit_result.table[audit_result.table.row_type == "pooled_oof"]
    for _, row in pooled.iterrows():
        assert row.original_v1_benchmark_accuracy == V1_BENCHMARK_ACCURACY
        assert row.actual_accuracy_difference_from_benchmark == pytest.approx(
            row.accuracy - V1_BENCHMARK_ACCURACY,
            abs=1e-15,
        )
        assert row.evaluation_n == 160
        assert len(row.evaluation_trial_uid_sha256) == 64
        assert len(row.center_fit_trial_uid_sha256) == 64
    all_sample = pooled[pooled.condition == "v1_all_sample"].iloc[0]
    fold_safe = pooled[pooled.condition == "fold_safe"].iloc[0]
    assert all_sample.center_evaluation_overlap_n == 160
    assert fold_safe.center_evaluation_overlap_n == 0
    assert all_sample.center_fit_exposures_n == 800
    assert fold_safe.center_fit_exposures_n == 640


def test_13_audit_is_deterministic_on_identical_synthetic_input(synthetic) -> None:
    covariances, metadata = synthetic
    first = run_v1_leakage_audit(covariances, metadata)
    second = run_v1_leakage_audit(covariances, metadata)
    pd.testing.assert_frame_equal(first.table, second.table)
    np.testing.assert_array_equal(first.fold_assignments, second.fold_assignments)
    for condition in first.oof_predictions:
        np.testing.assert_array_equal(
            first.oof_predictions[condition], second.oof_predictions[condition]
        )
