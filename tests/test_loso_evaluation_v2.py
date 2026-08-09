from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.loso_v2 as loso_module
from src.data_v2 import V2WholeData
from src.evaluation_v2 import FitAudit
from src.geometry_v2 import FittedCenter, fit_center
from src.loso_v2 import (
    MDM_SPECS,
    ClassificationGateError,
    V2OutputContractError,
    assert_classification_gate,
    assert_v2_output_contract,
    fit_target_transform,
    run_primary_loso,
    target_transform_signatures_are_label_free,
)


CLASSES = ("left_hand", "right_hand", "feet", "tongue")


def _synthetic_data() -> V2WholeData:
    rows: list[dict[str, object]] = []
    matrices: list[np.ndarray] = []
    class_location = {
        "left_hand": -1.2,
        "right_hand": -0.4,
        "feet": 0.4,
        "tongue": 1.2,
    }
    sample_index = 0
    for subject in range(1, 10):
        trial_id = 0
        for run in range(6):
            run_trial_id = 0
            for label in CLASSES:
                for repetition in range(12):
                    trial_id += 1
                    run_trial_id += 1
                    uid = f"S{subject:02d}_0train_T{trial_id:03d}"
                    rows.append(
                        {
                            "covariance_index": sample_index,
                            "sample_index": sample_index,
                            "subject": subject,
                            "session": "0train",
                            "run": str(run),
                            "trial_id": trial_id,
                            "run_trial_id": run_trial_id,
                            "trial_uid": uid,
                            "class_label": label,
                        }
                    )
                    class_value = class_location[label]
                    subject_value = 0.09 * (subject - 5)
                    jitter = 0.002 * (repetition - 5.5)
                    first = np.exp(class_value + subject_value + jitter)
                    second = np.exp(-0.3 * class_value + 0.5 * subject_value)
                    matrices.append(np.diag([first, second]))
                    sample_index += 1
    return V2WholeData(
        covariances=np.asarray(matrices, dtype=np.float64),
        metadata=pd.DataFrame(rows),
        channel_names=np.asarray(["C1", "C2"]),
        provenance={"source": "synthetic_test"},
    )


def _config() -> dict[str, object]:
    return {
        "protocol": {"version": "test", "seed": 20260809},
        "dataset": {
            "subjects": list(range(1, 10)),
            "classes": list(CLASSES),
        },
        "geometry": {"airm_mean": {"tol": 1e-9, "maxiter": 100}},
        "evaluation": {
            "T2": {"equivalent_flat_mean_tolerance": 1e-12}
        },
        "classifiers": {
            "primary_logistic": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 5000,
                "tol": 1e-4,
                "random_state": 20260809,
            },
            "metrics": {"class_order": list(CLASSES)},
        },
    }


@pytest.fixture(scope="module")
def synthetic_data() -> V2WholeData:
    return _synthetic_data()


@pytest.fixture(scope="module")
def loso_result(synthetic_data: V2WholeData):
    return run_primary_loso(
        synthetic_data,
        _config(),
        target_subjects=(1,),
        geometries=("RAW", "LE"),
        include_mdm=False,
        compute_domain_diagnostics=False,
        config_sha256="d" * 64,
    )


def test_loso_source_target_separation_and_full_uid_audit(loso_result) -> None:
    assert not loso_result.classification_failed
    t1 = loso_result.logistic_transductive
    assert len(t1) == 2
    assert set(t1["source_n"]) == {2304}
    assert set(t1["evaluation_n"]) == {288}
    assert set(t1["source_target_overlap_n"]) == {0}
    assert all(1 not in json.loads(value) for value in t1["source_subjects"])

    audit = loso_result.sample_id_audit
    assert len(audit) == 33  # T1 plus T2 A/B, each with 8+3 role rows.
    assert list(audit.columns[:3]) == [
        "protocol_version",
        "config_sha256",
        "seed",
    ]
    assert set(audit.config_sha256) == {"d" * 64}
    assert set(t1.config_sha256) == {"d" * 64}
    source = audit[(audit.protocol == "T1") & (audit.role == "classifier_train")]
    target = audit[(audit.protocol == "T1") & (audit.role == "evaluation")]
    source_uids = set(json.loads(source.iloc[0].trial_uids_json))
    target_uids = set(json.loads(target.iloc[0].trial_uids_json))
    assert len(source_uids) == 2304
    assert len(target_uids) == 288
    assert source_uids.isdisjoint(target_uids)


def test_t2_halves_are_disjoint_and_aggregate_ba_is_exact(loso_result) -> None:
    calibration = loso_result.logistic_calibration
    assert set(calibration["split"]) == {"A", "B", "AGGREGATE"}
    for geometry, group in calibration.groupby("geometry"):
        split_rows = group[group.split.isin(["A", "B"])]
        aggregate = group[group.split == "AGGREGATE"].iloc[0]
        assert len(split_rows) == 2
        assert set(split_rows["calibration_n"]) == {144}
        assert set(split_rows["evaluation_n"]) == {144}
        assert set(split_rows["center_evaluation_overlap_n"]) == {0}
        assert split_rows.calibration_trial_uid_hash.nunique() == 2
        assert split_rows.evaluation_trial_uid_hash.nunique() == 2
        assert float(aggregate.balanced_accuracy) == pytest.approx(
            float(split_rows.balanced_accuracy.mean()), abs=1e-12
        )

    audit = loso_result.sample_id_audit
    for split in ("A", "B"):
        selected = audit[(audit.protocol == "T2") & (audit.split == split)]
        calibration_uids = set(
            json.loads(
                selected[selected.role == "target_center_fit"].iloc[0].trial_uids_json
            )
        )
        evaluation_uids = set(
            json.loads(
                selected[selected.role == "evaluation"].iloc[0].trial_uids_json
            )
        )
        assert len(calibration_uids) == len(evaluation_uids) == 144
        assert calibration_uids.isdisjoint(evaluation_uids)


def test_t1_t2_reuse_exact_source_features_and_model(loso_result) -> None:
    combined = pd.concat(
        [
            loso_result.logistic_transductive,
            loso_result.logistic_calibration,
        ],
        ignore_index=True,
    )
    invariant_columns = (
        "source_feature_sha256",
        "source_label_sha256",
        "feature_config_sha256",
        "model_config_sha256",
        "fitted_model_sha256",
    )
    for _, condition in combined.groupby(["target_subject", "geometry", "decoder"]):
        assert set(condition.protocol) == {"T1", "T2"}
        for column in invariant_columns:
            assert condition[column].nunique() == 1


def test_target_transform_apis_cannot_accept_labels() -> None:
    forbidden = {"y", "label", "labels", "class_label", "target_labels"}
    assert target_transform_signatures_are_label_free()
    for function in (fit_center, fit_target_transform, FittedCenter.transform):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


def test_target_label_permutation_changes_only_evaluation_outputs(
    synthetic_data: V2WholeData, loso_result
) -> None:
    metadata = synthetic_data.metadata.copy()
    for run in range(6):
        mask = (metadata.subject == 1) & (metadata.run.astype(int) == run)
        metadata.loc[mask, "class_label"] = np.roll(
            metadata.loc[mask, "class_label"].to_numpy(), 12
        )
    permuted = V2WholeData(
        covariances=synthetic_data.covariances,
        metadata=metadata,
        channel_names=synthetic_data.channel_names,
        provenance=synthetic_data.provenance,
    )
    repeated = run_primary_loso(
        permuted,
        _config(),
        target_subjects=(1,),
        geometries=("RAW",),
        include_mdm=False,
        compute_domain_diagnostics=False,
        config_sha256="d" * 64,
    )
    original = pd.concat(
        [
            loso_result.logistic_transductive.query("geometry == 'RAW'"),
            loso_result.logistic_calibration.query("geometry == 'RAW'"),
        ],
        ignore_index=True,
    ).sort_values(["protocol", "split"])
    observed = pd.concat(
        [repeated.logistic_transductive, repeated.logistic_calibration],
        ignore_index=True,
    ).sort_values(["protocol", "split"])
    for column in (
        "source_trial_uid_hash",
        "fit_trial_uid_hash",
        "calibration_trial_uid_hash",
        "evaluation_trial_uid_hash",
        "source_feature_sha256",
        "target_input_sha256",
        "model_config_sha256",
        "fitted_model_sha256",
    ):
        assert original[column].tolist() == observed[column].tolist()
    assert original.prediction_sha256.tolist() != observed.prediction_sha256.tolist()


def _write_gate(directory: Path, value: object = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "geometry_gate.json").write_text(
        json.dumps(
            {
                "classification_gate_pass": value,
                "protocol_version": "test",
                "protocol_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "n_required_rows": 1,
                "n_required_passed": 1,
                "n_required_failed": 0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"required": True, "passed": True, "status": "PASS"}]
    ).to_csv(directory / "geometry_correctness.csv", index=False)


def test_gate_requires_exact_json_boolean_and_required_csv_pass(tmp_path: Path) -> None:
    _write_gate(tmp_path, True)
    expected = {
        "expected_protocol_version": "test",
        "expected_protocol_sha256": "a" * 64,
        "expected_config_sha256": "b" * 64,
    }
    assert assert_classification_gate(tmp_path, **expected)[
        "classification_gate_pass"
    ] is True

    _write_gate(tmp_path, 1)
    with pytest.raises(ClassificationGateError, match="boolean true exactly"):
        assert_classification_gate(tmp_path, **expected)

    _write_gate(tmp_path, True)
    pd.DataFrame(
        [{"required": True, "passed": False, "status": "FAIL"}]
    ).to_csv(tmp_path / "geometry_correctness.csv", index=False)
    with pytest.raises(ClassificationGateError, match="did not PASS"):
        assert_classification_gate(tmp_path, **expected)

    _write_gate(tmp_path, True)
    with pytest.raises(ClassificationGateError, match="config_sha256"):
        assert_classification_gate(
            tmp_path,
            **{**expected, "expected_config_sha256": "c" * 64},
        )


def test_primary_warning_records_exact_four_rows_and_continues_without_scoring(
    monkeypatch: pytest.MonkeyPatch, synthetic_data: V2WholeData
) -> None:
    class DummyModel:
        coef_ = np.zeros((4, 3), dtype=np.float64)
        intercept_ = np.zeros(4, dtype=np.float64)
        classes_ = np.asarray(CLASSES)

        def predict(self, inputs):
            raise AssertionError("a nonconverged model must never be scored")

    original_fit = loso_module.fit_source_logistic
    fit_count = 0

    def warning_fit(features, labels, **kwargs):
        nonlocal fit_count
        fit_count += 1
        if fit_count == 1:
            return DummyModel(), FitAudit(
                decoder="logistic",
                config_hash="f" * 64,
                n_train=len(features),
                n_features=features.shape[1],
                convergence_warning=True,
                warning_messages=("synthetic convergence warning",),
                n_iter_max=5000,
            )
        return original_fit(features, labels, **kwargs)

    monkeypatch.setattr("src.loso_v2.fit_source_logistic", warning_fit)
    result = run_primary_loso(
        synthetic_data,
        _config(),
        target_subjects=(1,),
        geometries=("RAW", "LE"),
        include_mdm=True,
        compute_domain_diagnostics=True,
    )
    assert result.classification_failed
    assert result.fatal_error is None
    assert result.primary_status == "FAILED"
    assert result.primary_failure_count == 1
    assert len(result.logistic_transductive) == 2
    assert len(result.logistic_calibration) == 6

    combined = pd.concat(
        [result.logistic_transductive, result.logistic_calibration],
        ignore_index=True,
    )
    failed = combined[combined.geometry == "RAW"]
    passed = combined[combined.geometry == "LE"]
    assert len(failed) == 4
    assert set(failed.split) == {"ALL", "A", "B", "AGGREGATE"}
    assert (failed.status == "FAILED").all()
    assert failed.balanced_accuracy.isna().all()
    assert failed.prediction_sha256.isna().all()
    assert failed.convergence_warning.astype(bool).all()
    assert failed.fitted_model_sha256.nunique() == 1
    assert failed.source_feature_sha256.nunique() == 1
    assert len(passed) == 4
    assert (passed.status == "PASS").all()
    assert passed.balanced_accuracy.notna().all()
    assert set(combined.primary_run_status) == {"FAILED"}
    assert set(combined.primary_failure_count) == {1}
    assert set(combined.secondary_run_status) == {"PASS"}
    assert result.secondary_status == "PASS"
    assert len(result.domain_shift_diagnostics) == 9
    assert len(result.mdm_transductive) == 3
    assert len(result.mdm_calibration) == 9
    assert (result.mdm_transductive.status == "PASS").all()


def test_mdm_only_warning_is_recorded_but_primary_is_preserved(
    monkeypatch: pytest.MonkeyPatch, synthetic_data: V2WholeData
) -> None:
    class DummyMDM:
        covmeans_ = np.repeat(np.eye(2)[np.newaxis], 4, axis=0)
        classes_ = np.asarray(CLASSES)

    def warning_fit(covariances, labels, *, metric):
        return DummyMDM(), FitAudit(
            decoder=f"mdm_{metric}",
            config_hash=f"{1 if metric == 'riemann' else 2:064x}",
            n_train=len(covariances),
            n_features=None,
            convergence_warning=True,
            warning_messages=(f"synthetic {metric} convergence warning",),
            n_iter_max=None,
        )

    monkeypatch.setattr("src.loso_v2.fit_source_mdm", warning_fit)
    result = run_primary_loso(
        synthetic_data,
        _config(),
        target_subjects=(1,),
        geometries=("RAW",),
        include_mdm=True,
        compute_domain_diagnostics=False,
    )
    assert not result.classification_failed
    assert result.fatal_error is None
    assert result.secondary_status == "FAILED"
    assert result.secondary_failure_count == 2
    assert len(result.logistic_transductive) == 1
    assert len(result.logistic_calibration) == 3
    assert (result.logistic_transductive.status == "PASS").all()
    assert set(result.logistic_transductive.secondary_run_status) == {"FAILED"}
    assert len(result.mdm_transductive) == 2
    assert len(result.mdm_calibration) == 6
    assert (result.mdm_transductive.status == "FAILED").all()
    assert (result.mdm_calibration.status == "FAILED").all()
    assert result.mdm_transductive.balanced_accuracy.isna().all()


def test_one_mdm_exception_does_not_suppress_other_secondary_condition(
    monkeypatch: pytest.MonkeyPatch, synthetic_data: V2WholeData
) -> None:
    original_fit = loso_module.fit_source_mdm

    def selective_fit(covariances, labels, *, metric):
        if metric == "riemann":
            raise RuntimeError("synthetic riemann failure")
        return original_fit(covariances, labels, metric=metric)

    monkeypatch.setattr("src.loso_v2.fit_source_mdm", selective_fit)
    result = run_primary_loso(
        synthetic_data,
        _config(),
        target_subjects=(1,),
        geometries=("RAW",),
        include_mdm=True,
        compute_domain_diagnostics=False,
    )
    assert not result.classification_failed
    assert result.secondary_failure_count == 1
    assert result.secondary_status == "FAILED"
    t1 = result.mdm_transductive.set_index("metric")
    assert t1.loc["riemann", "status"] == "FAILED"
    assert pd.isna(t1.loc["riemann", "balanced_accuracy"])
    assert "synthetic riemann failure" in t1.loc["riemann", "warning_messages"]
    assert t1.loc["logeuclid", "status"] == "PASS"
    assert np.isfinite(float(t1.loc["logeuclid", "balanced_accuracy"]))


def test_metric_native_mdm_combinations_are_frozen_exactly() -> None:
    assert MDM_SPECS == (
        ("RAW", "riemann"),
        ("RAW", "logeuclid"),
        ("LE", "logeuclid"),
        ("AIRM", "riemann"),
        ("EA", "riemann"),
    )


def test_mdm_t1_t2_execution_reuses_each_native_source_model(
    synthetic_data: V2WholeData,
) -> None:
    result = run_primary_loso(
        synthetic_data,
        _config(),
        target_subjects=(1,),
        geometries=("RAW",),
        include_mdm=True,
        compute_domain_diagnostics=False,
        config_sha256="e" * 64,
    )
    assert not result.classification_failed
    assert set(result.mdm_transductive.metric) == {"riemann", "logeuclid"}
    assert len(result.mdm_transductive) == 2
    assert len(result.mdm_calibration) == 6
    combined = pd.concat(
        [result.mdm_transductive, result.mdm_calibration], ignore_index=True
    )
    for metric, condition in combined.groupby("metric"):
        assert set(condition.protocol) == {"T1", "T2"}
        for column in (
            "source_feature_sha256",
            "model_config_sha256",
            "fitted_model_sha256",
        ):
            assert condition[column].nunique() == 1, metric
        split = condition[condition.split.isin(["A", "B"])]
        aggregate = condition[condition.split == "AGGREGATE"].iloc[0]
        assert float(aggregate.balanced_accuracy) == pytest.approx(
            float(split.balanced_accuracy.mean()), abs=1e-12
        )


def test_domain_diagnostics_cover_native_metrics_without_class_use(
    synthetic_data: V2WholeData,
) -> None:
    result = run_primary_loso(
        synthetic_data,
        _config(),
        target_subjects=(1,),
        geometries=("RAW", "LE", "AIRM", "EA"),
        include_mdm=False,
        compute_domain_diagnostics=True,
        config_sha256="f" * 64,
    )
    domain = result.domain_shift_diagnostics
    assert len(domain) == 15
    assert not domain.uses_class_labels.astype(bool).any()
    assert set(domain[domain.geometry == "RAW"].reference_metric) == {
        "logeuclid",
        "riemann",
    }
    assert set(domain[domain.geometry == "LE"].reference_metric) == {"logeuclid"}
    assert set(domain[domain.geometry == "AIRM"].reference_metric) == {"riemann"}
    assert set(domain[domain.geometry == "EA"].reference_metric) == {
        "arithmetic_frobenius"
    }
    assert set(domain.split) == {"ALL", "A", "B"}
    t1 = domain[domain.protocol == "T1"]
    assert t1.all_subject_subject_silhouette.notna().all()
    assert t1.all_subject_subject_between_within_rms_ratio.notna().all()
    assert domain[domain.protocol == "T2"].all_subject_subject_silhouette.isna().all()


def test_output_contract_allows_only_exact_v2_tables_path(tmp_path: Path) -> None:
    config = {
        "project": {"output_dir": "outputs/bnci2014_001_geometry_v2"},
        "outputs": {
            "tables_dir": "outputs/bnci2014_001_geometry_v2/tables"
        },
    }
    assert assert_v2_output_contract(tmp_path, config) == (
        tmp_path / "outputs" / "bnci2014_001_geometry_v2" / "tables"
    ).resolve()

    bad = {
        **config,
        "project": {"output_dir": "outputs/bnci2014_001"},
        "outputs": {"tables_dir": "outputs/bnci2014_001/tables"},
    }
    with pytest.raises(V2OutputContractError, match="exactly"):
        assert_v2_output_contract(tmp_path, bad)
