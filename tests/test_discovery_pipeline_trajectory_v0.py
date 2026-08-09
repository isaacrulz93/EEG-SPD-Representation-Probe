from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import src.evaluation_trajectory_v0 as evaluation
from src.discovery_pipeline_trajectory_v0 import (
    DISCOVERY_TABLE_NAMES,
    FEATURE_NPZ_KEYS,
    NULL_ARTIFACT_NAMES,
    ROBUSTNESS_COLUMNS,
    WHOLE_MDM_COLUMNS,
    DiscoveryStructuralError,
    _paired_path_minus_bag,
    build_discovery_artifacts,
    load_discovery_inputs,
    write_discovery_artifacts,
)
from src.evaluation_trajectory_v0 import (
    CLASS_LOSO_COLUMNS,
    FACTOR_COLUMNS,
    LABEL_GROUP_COLUMNS,
    MDM_COLUMNS,
    NULL_SUBJECT_COLUMNS,
    ORDER_GROUP_COLUMNS,
    SUBJECT_PROBE_COLUMNS,
)
from src.trajectory_geometry_v0 import (
    PATH_D10_NAMES,
    SCALAR_11_NAMES,
    bag_canon_d10,
    bag_sorted_d10,
    path_d10,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _synthetic_config() -> dict[str, Any]:
    config = yaml.safe_load(
        (ROOT / "configs" / "bnci2014_001_trajectory_v0.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["dataset"].update(
        {
            "subjects": [1, 2, 3],
            "runs": [0, 1, 2, 3, 4, 5],
            "classes": ["left_hand", "right_hand", "feet", "tongue"],
            "expected_trials": 72,
            "expected_trials_per_subject": 24,
            "expected_trials_per_subject_class": 6,
            "expected_trials_per_subject_run": 4,
            "expected_trials_per_subject_run_class": 1,
            "eeg_channels": ["C3", "C4"],
        }
    )
    config["dataset"].pop("expected_trial_uid_set_sha256", None)
    config["window5"].update(
        {
            "expected_covariance_shape": [360, 2, 2],
            "expected_trial_tensor_shape": [72, 5, 2, 2],
        }
    )
    config["nulls"]["order_shuffle"]["replicates"] = 3
    config["nulls"]["label_destruction"]["replicates"] = 3
    config["subject_probe"]["chance"] = 1.0 / 3.0
    config["subject_probe"]["trials_per_subject_half"] = 12
    config["factor_decomposition"].update(
        {"n_subjects": 3, "n_classes": 4, "n_per_cell": 6}
    )
    config["geometry"]["bag_validation"]["expected_trials"] = 12
    return config


def _distance_matrix(values: np.ndarray) -> np.ndarray:
    result = np.abs(values[:, None] - values[None, :]).astype(np.float64)
    np.fill_diagonal(result, 0.0)
    return result


def _make_feature_arrays(config: dict[str, Any]) -> dict[str, np.ndarray]:
    patterns = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 4.0, 7.0],
            [0.0, 1.0, 3.0, 6.0, 10.0],
            [0.0, 2.0, 3.0, 7.0, 8.0],
        ],
        dtype=np.float64,
    )
    records: list[dict[str, Any]] = []
    per_subject_trial = {subject: 0 for subject in config["dataset"]["subjects"]}
    sample_index = 0
    for subject in config["dataset"]["subjects"]:
        for run in config["dataset"]["runs"]:
            for class_index, label in enumerate(config["dataset"]["classes"]):
                per_subject_trial[subject] += 1
                records.append(
                    {
                        "sample_index": sample_index,
                        "subject": subject,
                        "run": run,
                        "trial_id": per_subject_trial[subject],
                        "trial_uid": f"sub-{subject:02d}_0train_run-{run}_trial-{class_index:02d}",
                        "class_label": label,
                        "class_index": class_index,
                    }
                )
                sample_index += 1
    metadata = pd.DataFrame.from_records(records)
    n_trials = len(metadata)
    airm_distance = np.empty((n_trials, 5, 5), dtype=np.float64)
    le_distance = np.empty_like(airm_distance)
    airm_path = np.empty((n_trials, 10), dtype=np.float64)
    le_path = np.empty_like(airm_path)
    airm_bag = np.empty_like(airm_path)
    le_bag = np.empty_like(airm_path)
    airm_sorted = np.empty_like(airm_path)
    le_sorted = np.empty_like(airm_path)
    airm_permutation = np.empty((n_trials, 5), dtype=np.int64)
    le_permutation = np.empty_like(airm_permutation)
    airm_scalars = np.empty((n_trials, 11), dtype=np.float64)
    le_scalars = np.empty_like(airm_scalars)
    local = np.empty((n_trials, 2, 2), dtype=np.float64)
    whole = np.empty_like(local)
    perturbation = np.asarray([0.0, 0.01, -0.008, 0.015, -0.004])
    for index, row in metadata.iterrows():
        class_index = int(row["class_index"])
        subject = int(row["subject"])
        run = int(row["run"])
        state_values = patterns[class_index] + (
            0.07 * subject + 0.01 * run
        ) * perturbation
        airm_distance[index] = _distance_matrix(state_values)
        le_distance[index] = _distance_matrix(
            1.035 * state_values + 0.001 * np.square(state_values)
        )
        for prefix, matrix, path_values, bag_values, sorted_values, permutations in (
            ("airm", airm_distance[index], airm_path, airm_bag, airm_sorted, airm_permutation),
            ("le", le_distance[index], le_path, le_bag, le_sorted, le_permutation),
        ):
            del prefix
            path_values[index] = path_d10(matrix)
            canonical = bag_canon_d10(matrix)
            bag_values[index] = canonical.vector
            sorted_values[index] = bag_sorted_d10(matrix)
            permutations[index] = canonical.permutation
        base = 0.4 * (class_index + 1) + 0.07 * subject + 0.013 * run
        for scalar_index in range(11):
            value = (
                base * (1.0 + 0.09 * scalar_index)
                + 0.002 * (run + 1) ** 2
                + 0.0003 * (index + 1) * (scalar_index + 1)
            )
            airm_scalars[index, scalar_index] = value
            le_scalars[index, scalar_index] = (
                value * (1.01 + 0.0007 * scalar_index)
                + 0.003 * np.sin(index + scalar_index)
            )
        local[index] = np.diag(
            [
                1.0 + 0.22 * class_index + 0.012 * subject + 0.001 * run,
                1.6 + 0.09 * class_index + 0.008 * subject + 0.002 * run,
            ]
        )
        whole[index] = np.diag(
            [
                1.2 + 0.25 * class_index + 0.01 * subject + 0.001 * run,
                1.9 + 0.11 * class_index + 0.006 * subject + 0.002 * run,
            ]
        )

    protocol = str(config["project"]["protocol_version"])
    protocol_hash = str(config["project"]["protocol_sha256"])
    config_hash = _config_hash(config)
    arrays: dict[str, np.ndarray] = {
        "airm_distance_matrices": airm_distance,
        "le_distance_matrices": le_distance,
        "airm_path_d10": airm_path,
        "le_path_d10": le_path,
        "airm_bag_canon_d10": airm_bag,
        "le_bag_canon_d10": le_bag,
        "airm_bag_sorted_d10": airm_sorted,
        "le_bag_sorted_d10": le_sorted,
        "airm_scalars_11": airm_scalars,
        "le_scalars_11": le_scalars,
        "airm_canonical_permutation": airm_permutation,
        "le_canonical_permutation": le_permutation,
        "local_airm_barycenters": local,
        "whole_covariances": whole,
        "sample_index": metadata["sample_index"].to_numpy(dtype=np.int64),
        "subject": metadata["subject"].to_numpy(dtype=np.int64),
        "run": metadata["run"].to_numpy(dtype=np.int64),
        "trial_id": metadata["trial_id"].to_numpy(dtype=np.int64),
        "trial_uid": metadata["trial_uid"].to_numpy(dtype="U"),
        "class_label": metadata["class_label"].to_numpy(dtype="U"),
        "airm_trial_gate_pass": np.ones(n_trials, dtype=bool),
        "le_trial_gate_pass": np.ones(n_trials, dtype=bool),
        "path_d10_names": np.asarray(PATH_D10_NAMES, dtype="U"),
        "scalar_11_names": np.asarray(SCALAR_11_NAMES, dtype="U"),
        "protocol_version": np.asarray([protocol], dtype="U"),
        "protocol_sha256": np.asarray([protocol_hash], dtype="U"),
        "config_sha256": np.asarray([config_hash], dtype="U"),
        "session": np.asarray(["0train"], dtype="U"),
        "seed": np.asarray([config["project"]["seed"]], dtype=np.int64),
        "generated_at_utc": np.asarray(["2026-08-09T00:00:00Z"], dtype="U"),
        "geometry_gate_passed": np.asarray([True], dtype=bool),
    }
    assert tuple(arrays) == FEATURE_NPZ_KEYS
    return arrays


def _write_fixture(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    config = _synthetic_config()
    feature_path = (
        tmp_path
        / config["project"]["local_cache_dir"]
        / "trajectory_features_v0.npz"
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _make_feature_arrays(config)
    with feature_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    gate_path = (
        tmp_path
        / config["project"]["output_dir"]
        / "tables"
        / "trajectory_geometry_gate.json"
    )
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate = {
        "protocol_version": config["project"]["protocol_version"],
        "protocol_sha256": config["project"]["protocol_sha256"],
        "config_sha256": _config_hash(config),
        "seed": config["project"]["seed"],
        "session": "0train",
        "status": "PASS",
        "gate_passed": True,
        "scientific_classification_allowed": True,
        "feature_npz_written": True,
        "feature_npz_path": str(feature_path.resolve()),
        "feature_npz_sha256": _sha256(feature_path),
        "required_failure_counts": {
            "dataset_contract": 0,
            "covariance_sanity": 0,
            "trajectory_geometry_correctness": 0,
        },
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return config, feature_path, gate_path


def _load_and_build(tmp_path: Path):
    config, _, _ = _write_fixture(tmp_path)
    loaded_config, inputs = load_discovery_inputs(config, tmp_path)
    artifacts = build_discovery_artifacts(
        inputs,
        loaded_config,
        generated_at_utc="2026-08-09T01:02:03Z",
    )
    return loaded_config, inputs, artifacts


def test_loader_strict_gate_hash_and_forbidden_session_barriers(tmp_path: Path) -> None:
    config, feature_path, gate_path = _write_fixture(tmp_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["gate_passed"] = False
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(DiscoveryStructuralError, match="gate_passed"):
        load_discovery_inputs(config, tmp_path)

    config, feature_path, gate_path = _write_fixture(tmp_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["feature_npz_sha256"] = "0" * 64
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(DiscoveryStructuralError, match="SHA-256"):
        load_discovery_inputs(config, tmp_path)

    config, feature_path, gate_path = _write_fixture(tmp_path)
    with np.load(feature_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["session"] = np.asarray(["1test"], dtype="U")
    with feature_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["feature_npz_sha256"] = _sha256(feature_path)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(DiscoveryStructuralError, match="forbidden-session"):
        load_discovery_inputs(config, tmp_path)
    assert not (gate_path.parent / "class_loso_metrics.csv").exists()


def test_builds_exact_scientific_grids_and_shared_null_contract(tmp_path: Path) -> None:
    config, inputs, artifacts = _load_and_build(tmp_path)
    assert artifacts.status == "PASS"
    assert artifacts.technical_failure_count == 0
    assert tuple(artifacts.tables) == DISCOVERY_TABLE_NAMES
    schemas = {
        "class_loso_metrics": CLASS_LOSO_COLUMNS,
        "subject_runhalf_probe": SUBJECT_PROBE_COLUMNS,
        "scalar_factor_decomposition": FACTOR_COLUMNS,
        "order_shuffle_subject_metrics": NULL_SUBJECT_COLUMNS,
        "order_shuffle_group_metrics": ORDER_GROUP_COLUMNS,
        "label_null_subject_metrics": NULL_SUBJECT_COLUMNS,
        "label_null_group_metrics": LABEL_GROUP_COLUMNS,
        "local_barycenter_mdm": MDM_COLUMNS,
        "whole_context_mdm": WHOLE_MDM_COLUMNS,
        "airm_le_robustness": ROBUSTNESS_COLUMNS,
    }
    for name, columns in schemas.items():
        assert tuple(artifacts.tables[name].columns) == tuple(columns)
        assert set(artifacts.tables[name]["session"]) == {"0train"}
    assert len(artifacts.tables["class_loso_metrics"]) == 7 * 3
    assert len(artifacts.tables["subject_runhalf_probe"]) == 3 * 2
    assert len(artifacts.tables["scalar_factor_decomposition"]) == 2 * 11
    assert len(artifacts.tables["order_shuffle_subject_metrics"]) == 2 * 3 * 3
    assert len(artifacts.tables["label_null_subject_metrics"]) == 2 * 3 * 3
    assert len(artifacts.tables["local_barycenter_mdm"]) == 3
    assert len(artifacts.tables["whole_context_mdm"]) == 3
    assert np.all(artifacts.order_plan.permutation_indices >= 1)
    assert np.all(artifacts.order_plan.permutation_indices <= 119)
    assert artifacts.bag_plan_audit_sha256
    label_table = artifacts.tables["label_null_subject_metrics"]
    path_seeds = label_table[label_table["representation"] == "PATH_D10"][
        ["replicate", "replicate_seed", "target_subject"]
    ].reset_index(drop=True)
    bag_seeds = label_table[label_table["representation"] == "BAG_CANON_D10"][
        ["replicate", "replicate_seed", "target_subject"]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(path_seeds, bag_seeds)
    np.testing.assert_array_equal(
        artifacts.label_group_statistics["replicate_seed"],
        artifacts.label_plan.seed_plan.seeds,
    )
    assert inputs.feature_npz_path.is_file()


def test_path_minus_bag_is_geometry_specific() -> None:
    rows = []
    values = {
        "AIRM": {"PATH_D10": (0.8, 0.7, 0.6), "BAG_CANON_D10": (0.5, 0.5, 0.5)},
        "LE": {"PATH_D10": (0.55, 0.50, 0.45), "BAG_CANON_D10": (0.50, 0.48, 0.46)},
    }
    for geometry, representations in values.items():
        for representation, scores in representations.items():
            for subject, score in enumerate(scores, start=1):
                rows.append(
                    {
                        "geometry": geometry,
                        "representation": representation,
                        "target_subject": subject,
                        "balanced_accuracy": score,
                        "status": "PASS",
                    }
                )
    observed = pd.DataFrame.from_records(rows)
    airm = _paired_path_minus_bag(observed, (1, 2, 3), geometry="AIRM")
    le = _paired_path_minus_bag(observed, (1, 2, 3), geometry="LE")
    assert airm == pytest.approx(0.2)
    assert le == pytest.approx(0.02)
    assert airm != le


def test_atomic_writer_emits_only_exact_discovery_artifacts(tmp_path: Path) -> None:
    config, _, artifacts = _load_and_build(tmp_path)
    sentinel = tmp_path / "outputs" / "bnci2014_001" / "v1_sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("unchanged", encoding="utf-8")
    result = write_discovery_artifacts(artifacts, config, tmp_path)
    tables_dir = tmp_path / config["project"]["output_dir"] / "tables"
    nulls_dir = tmp_path / config["project"]["output_dir"] / "nulls"
    for name in DISCOVERY_TABLE_NAMES:
        assert (tables_dir / f"{name}.csv").is_file()
    for name in NULL_ARTIFACT_NAMES:
        assert (nulls_dir / name).is_file()
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not list(tables_dir.glob(".*"))
    assert not list(nulls_dir.glob(".*"))
    assert result["status"] == "PASS"
    with np.load(nulls_dir / "order_shuffle_group_stats.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "replicate",
            "replicate_seed",
            "airm__path_d10__median_subject_ba",
            "le__path_d10__median_subject_ba",
        }
        assert archive["replicate"].dtype == np.dtype("int64")
        assert archive["replicate_seed"].dtype == np.dtype("uint64")
        assert archive["airm__path_d10__median_subject_ba"].shape == (3,)
    with np.load(nulls_dir / "label_null_group_stats.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "replicate",
            "replicate_seed",
            "airm__path_d10__median_subject_ba",
            "airm__bag_canon_d10__median_subject_ba",
        }
    order_seed_json = json.loads(
        (nulls_dir / "order_shuffle_seeds.json").read_text(encoding="utf-8")
    )
    assert order_seed_json == artifacts.order_plan.seed_plan.to_json_dict(
        protocol_version=config["project"]["protocol_version"]
    )


def test_convergence_failure_is_recorded_without_available_case_or_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = _write_fixture(tmp_path)
    loaded_config, inputs = load_discovery_inputs(config, tmp_path)
    real_fit = evaluation.fit_source_scaled_logistic
    calls = 0

    def fail_first_fit(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        fitted = real_fit(*args, **kwargs)
        if calls != 1:
            return fitted
        failed_audit = dataclasses.replace(
            fitted.audit,
            status="FAILED",
            convergence_warning=True,
            warning_messages=("synthetic convergence warning",),
            fitted_estimator_sha256=None,
        )
        return dataclasses.replace(fitted, audit=failed_audit)

    monkeypatch.setattr(evaluation, "fit_source_scaled_logistic", fail_first_fit)
    artifacts = build_discovery_artifacts(
        inputs,
        loaded_config,
        generated_at_utc="2026-08-09T01:02:03Z",
    )
    observed = artifacts.tables["class_loso_metrics"]
    failed = observed[
        (observed["geometry"] == "AIRM")
        & (observed["representation"] == "PATH_D10")
    ]
    assert (failed["status"] == "FAILED").sum() == 1
    assert failed.loc[failed["status"] == "FAILED", "balanced_accuracy"].isna().all()
    order = artifacts.tables["order_shuffle_subject_metrics"]
    skipped_order = order[
        (order["geometry"] == "AIRM") & (order["representation"] == "PATH_D10")
    ]
    assert len(skipped_order) == 3 * 3
    assert (skipped_order["status"] == "FAILED").all()
    assert skipped_order["balanced_accuracy"].isna().all()
    assert np.isnan(
        artifacts.order_group_statistics["airm__path_d10__median_subject_ba"]
    ).all()
    assert np.isfinite(
        artifacts.order_group_statistics["le__path_d10__median_subject_ba"]
    ).all()
    label = artifacts.tables["label_null_subject_metrics"]
    skipped_label = label[label["representation"] == "PATH_D10"]
    continued_label = label[label["representation"] == "BAG_CANON_D10"]
    assert (skipped_label["status"] == "FAILED").all()
    assert (continued_label["status"] == "PASS").all()
    assert artifacts.status == "FAILED"
    assert artifacts.technical_failure_count > 0


def test_failed_bag_cross_operand_does_not_abort_independent_path_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = _write_fixture(tmp_path)
    loaded_config, inputs = load_discovery_inputs(config, tmp_path)
    real_fit = evaluation.fit_source_scaled_logistic
    calls = 0

    def fail_first_airm_bag_fold(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        fitted = real_fit(*args, **kwargs)
        # Three AIRM/PATH LOSO folds precede AIRM/BAG in the frozen grid.
        if calls != 4:
            return fitted
        return dataclasses.replace(
            fitted,
            audit=dataclasses.replace(
                fitted.audit,
                status="FAILED",
                convergence_warning=True,
                warning_messages=("synthetic BAG convergence warning",),
                fitted_estimator_sha256=None,
            ),
        )

    monkeypatch.setattr(
        evaluation, "fit_source_scaled_logistic", fail_first_airm_bag_fold
    )
    artifacts = build_discovery_artifacts(
        inputs,
        loaded_config,
        generated_at_utc="2026-08-09T01:02:03Z",
    )
    airm_order_subject = artifacts.tables["order_shuffle_subject_metrics"].query(
        "geometry == 'AIRM'"
    )
    airm_order_group = artifacts.tables["order_shuffle_group_metrics"].query(
        "geometry == 'AIRM'"
    )
    le_order_group = artifacts.tables["order_shuffle_group_metrics"].query(
        "geometry == 'LE'"
    )
    assert (airm_order_subject["status"] == "PASS").all()
    assert len(airm_order_subject) == 3 * 3
    assert airm_order_group.iloc[0]["status"] == "FAILED"
    assert pd.isna(airm_order_group.iloc[0]["median_subject_path_minus_bag"])
    assert np.isnan(
        artifacts.order_group_statistics["airm__path_d10__median_subject_ba"]
    ).all()
    assert le_order_group.iloc[0]["status"] == "PASS"
    assert np.isfinite(
        artifacts.order_group_statistics["le__path_d10__median_subject_ba"]
    ).all()
    label_path = artifacts.tables["label_null_subject_metrics"].query(
        "representation == 'PATH_D10'"
    )
    label_bag = artifacts.tables["label_null_subject_metrics"].query(
        "representation == 'BAG_CANON_D10'"
    )
    assert (label_path["status"] == "PASS").all()
    assert (label_bag["status"] == "FAILED").all()
