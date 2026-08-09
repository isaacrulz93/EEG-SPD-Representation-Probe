"""Synthetic-only tests for trajectory-v0 feature/gate orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_trajectory_v0 import TrajectoryWindowData
from src.feature_pipeline_trajectory_v0 import (
    COMMON_PROVENANCE_COLUMNS,
    FEATURE_TABLE_NAMES,
    TrajectoryFeatureGateError,
    TrajectoryFeaturePipelineError,
    build_trajectory_feature_artifacts,
    require_scientific_gate,
    write_trajectory_feature_artifacts,
)


def _config(protocol_sha256: str = "0" * 64) -> dict[str, object]:
    return {
        "project": {
            "name": "synthetic",
            "experiment_name": "Trajectory Anatomy v0",
            "protocol_version": "0.0",
            "protocol_path": "docs/PROTOCOL_TRAJECTORY_ANATOMY_V0.md",
            "protocol_sha256": protocol_sha256,
            "base_commit": "synthetic",
            "seed": 20260809,
            "output_dir": "outputs/bnci2014_001_trajectory_v0",
            "local_cache_dir": "cache/bnci2014_001_trajectory_v0",
        },
        "dataset": {
            "name": "SYNTHETIC",
            "allowed_session": "0train",
            "forbidden_session": "1test",
            "subjects": [1, 2],
            "runs": [0, 1],
            "classes": ["left", "right"],
            "expected_trials": 8,
            "expected_trials_per_subject": 4,
            "expected_trials_per_subject_class": 2,
            "expected_trials_per_subject_run": 2,
            "expected_trials_per_subject_run_class": 1,
            "eeg_channels": ["C1", "C2"],
        },
        "window5": {
            "n_windows": 5,
            "samples_per_window": 200,
            "overlap_samples": 0,
            "remainder_policy": "require_exact_division",
            "chronological_order": [1, 2, 3, 4, 5],
            "expected_covariance_shape": [40, 2, 2],
            "expected_trial_tensor_shape": [8, 5, 2, 2],
        },
        "geometry": {
            "primary": "airm",
            "secondary": "logeuclid",
            "airm_mean": {
                "tol": 1e-9,
                "maxiter": 100,
                "normalized_karcher_residual_max": 1e-7,
            },
            "hard_gate": {
                "symmetry_relative_error_max": 1e-12,
                "condition_number_max": 1e12,
                "distance_symmetry_absolute_error_max": 1e-10,
                "distance_diagonal_absolute_error_max": 1e-12,
                "distance_negative_tolerance": 1e-12,
                "triangle_absolute_tolerance": 1e-10,
                "triangle_relative_tolerance": 1e-10,
                "centering_isometry_absolute_error_max": 1e-10,
                "centering_isometry_relative_error_max": 1e-10,
                "bag_invariance_absolute_error_max": 1e-12,
                "path_endpoint_absolute_tolerance": 1e-10,
                "path_endpoint_relative_tolerance": 1e-10,
                "zero_length_epsilon": 1e-12,
                "efficiency_bound_tolerance": 1e-12,
                "angle_bound_tolerance": 1e-12,
                "cosine_domain_tolerance": 1e-10,
                "deviation_negative_tolerance": 1e-12,
                "geodesic_endpoint_relative_error_max": 1e-10,
                "ss_closure_relative_error_max": 1e-10,
            },
            "bag_validation": {
                "trial_rule": "smallest_trial_id_per_subject_class",
                "expected_trials": 4,
                "permutations": 120,
            },
            "centering_isometry": {
                "trials": "all",
                "fit_from": "whole_covariance_subject_airm_mean",
                "classification_use": False,
            },
        },
        "representations": {
            "path_d10_columns": [
                "d12",
                "d13",
                "d14",
                "d15",
                "d23",
                "d24",
                "d25",
                "d34",
                "d35",
                "d45",
            ],
            "bag_canon_columns": [f"bag{index:02d}" for index in range(1, 11)],
            "bag_sorted_columns": [f"sorted{index:02d}" for index in range(1, 11)],
            "scalar_columns": [
                "total_path_length",
                "endpoint_distance",
                "efficiency",
                "excess",
                "mean_turn",
                "max_turn",
                "mean_geodesic_deviation",
                "max_geodesic_deviation",
                "frechet_variance",
                "frechet_radius_mean",
                "diameter",
            ],
            "descriptive_scalar_columns": [
                "s1",
                "s2",
                "s3",
                "s4",
                "theta2",
                "theta3",
                "theta4",
                "dev2",
                "dev3",
                "dev4",
            ],
        },
        "verdicts": {
            "numerical_failure_status": "UNASSESSED — NUMERICAL/DATA FAILURE",
            "technical_failure_status": "UNASSESSED—PROTOCOL/TECHNICAL FAILURE",
        },
    }


def _synthetic_data(config: dict[str, object], *, degenerate: bool = False) -> TrajectoryWindowData:
    records: list[dict[str, object]] = []
    states: list[np.ndarray] = []
    whole: list[np.ndarray] = []
    sample = 0
    for subject in (1, 2):
        subject_trial = 0
        for run in (0, 1):
            for class_index, class_label in enumerate(("left", "right")):
                subject_trial += 1
                records.append(
                    {
                        "sample_index": sample,
                        "subject": subject,
                        "session": "0train",
                        "run": run,
                        "trial_id": subject_trial,
                        "run_trial_id": class_index + 1,
                        "trial_uid": f"S{subject:02d}_T{subject_trial:03d}",
                        "class_label": class_label,
                    }
                )
                if degenerate:
                    trial = np.broadcast_to(np.eye(2), (5, 2, 2)).copy()
                    whole_matrix = np.eye(2)
                else:
                    base = np.asarray(
                        [0.08 * subject + 0.01 * run, -0.05 * subject + 0.02 * class_index]
                    )
                    delta = np.asarray(
                        [0.25 + 0.01 * class_index, -0.16 - 0.01 * run]
                    )
                    # Unequal monotone times avoid exact finite-metric ties,
                    # which would make lexicographic canonical labeling
                    # intentionally discontinuous under roundoff.
                    times = (0.0, 0.17, 0.43, 0.68, 1.0)
                    trial = np.asarray(
                        [np.diag(np.exp(base + time * delta)) for time in times]
                    )
                    whole_matrix = np.diag(np.exp(base + 0.5 * delta))
                states.append(trial)
                whole.append(whole_matrix)
                sample += 1
    state_array = np.asarray(states, dtype=np.float64)
    whole_array = np.asarray(whole, dtype=np.float64)
    channels = np.asarray(["C1", "C2"])
    for array in (state_array, whole_array, channels):
        array.setflags(write=False)
    provenance = {
        "protocol_version": "0.0",
        "protocol_sha256": config["project"]["protocol_sha256"],
        "config_sha256": "c" * 64,
        "session": "0train",
        "whole_content_sha256": "w" * 64,
        "whole_covariance_shape": list(whole_array.shape),
        "source": "synthetic_test",
    }
    return TrajectoryWindowData(
        state_array,
        whole_array,
        pd.DataFrame.from_records(records),
        channels,
        provenance,
    )


def test_builds_exact_first_ten_tables_and_passes_every_synthetic_gate() -> None:
    config = _config()
    progress: list[tuple[int, int]] = []
    artifacts = build_trajectory_feature_artifacts(
        _synthetic_data(config),
        config,
        generated_at_utc="2026-08-09T00:00:00Z",
        progress=lambda done, total: progress.append((done, total)),
    )
    assert tuple(artifacts.tables) == FEATURE_TABLE_NAMES
    assert artifacts.gate_passed is True
    assert artifacts.gate_summary["scientific_classification_allowed"] is True
    assert artifacts.gate_summary["required_failure_counts"] == {
        "dataset_contract": 0,
        "covariance_sanity": 0,
        "trajectory_geometry_correctness": 0,
    }
    assert progress[-1] == (8, 8)
    assert len(artifacts.tables["covariance_sanity"]) == 40
    for name in FEATURE_TABLE_NAMES[3:]:
        assert len(artifacts.tables[name]) == 8
    for table in artifacts.tables.values():
        assert tuple(table.columns[: len(COMMON_PROVENANCE_COLUMNS)]) == COMMON_PROVENANCE_COLUMNS
        assert set(table["session"]) == {"0train"}
        assert set(table["status"]) == {"PASS"}
    geometry = artifacts.tables["trajectory_geometry_correctness"]
    selected = geometry[geometry["check"] == "bag_permutation_invariance"]
    assert len(selected) == 4 * 2 * 2  # selected trials x metrics x two checks
    assert set(selected["statistic"]) == {"permutation_count", "maximum_absolute_error"}
    isometry = geometry[geometry["check"] == "centering_isometry"]
    assert len(isometry) == 8 * 4
    assert isometry["passed"].all()
    assert artifacts.arrays["airm_distance_matrices"].shape == (8, 5, 5)
    assert artifacts.arrays["le_distance_matrices"].shape == (8, 5, 5)
    assert artifacts.arrays["airm_path_d10"].shape == (8, 10)
    assert artifacts.arrays["airm_scalars_11"].shape == (8, 11)
    assert artifacts.arrays["local_airm_barycenters"].shape == (8, 2, 2)
    assert np.isfinite(artifacts.arrays["airm_scalars_11"]).all()
    require_scientific_gate(artifacts)


def test_degenerate_geometry_is_persisted_but_blocks_downstream_features(tmp_path: Path) -> None:
    protocol_text = "# synthetic frozen trajectory protocol\n"
    protocol_hash = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    config = _config(protocol_hash)
    artifacts = build_trajectory_feature_artifacts(
        _synthetic_data(config, degenerate=True),
        config,
        generated_at_utc="2026-08-09T00:00:00Z",
    )
    assert artifacts.gate_passed is False
    assert artifacts.gate_summary["scientific_classification_allowed"] is False
    assert artifacts.gate_summary["status"] == "UNASSESSED — NUMERICAL/DATA FAILURE"
    failed = artifacts.tables["trajectory_geometry_correctness"]
    assert (~failed.loc[failed["required"], "passed"]).any()
    assert artifacts.tables["trial_airm_path_features"]["degenerate"].all()
    with pytest.raises(TrajectoryFeatureGateError, match="UNASSESSED"):
        require_scientific_gate(artifacts)

    protocol_path = tmp_path / "docs" / "PROTOCOL_TRAJECTORY_ANATOMY_V0.md"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(protocol_text, encoding="utf-8")
    config_path = tmp_path / "configs" / "bnci2014_001_trajectory_v0.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    stale = tmp_path / "cache" / "bnci2014_001_trajectory_v0" / "trajectory_features_v0.npz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    written = write_trajectory_feature_artifacts(
        artifacts,
        config,
        tmp_path,
        config_path=config_path,
        protocol_path=protocol_path,
    )
    assert written["gate_summary"]["feature_npz_written"] is False
    assert not stale.exists()
    for name in FEATURE_TABLE_NAMES:
        assert (tmp_path / "outputs" / "bnci2014_001_trajectory_v0" / "tables" / f"{name}.csv").is_file()
    assert (tmp_path / "outputs" / "bnci2014_001_trajectory_v0" / "tables" / "trajectory_geometry_gate.json").is_file()


def test_passing_writer_copies_protocol_provenance_and_writes_loadable_npz(tmp_path: Path) -> None:
    protocol_text = "# synthetic frozen trajectory protocol\n"
    protocol_hash = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    config = _config(protocol_hash)
    protocol_path = tmp_path / "docs" / "PROTOCOL_TRAJECTORY_ANATOMY_V0.md"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(protocol_text, encoding="utf-8")
    config_path = tmp_path / "configs" / "bnci2014_001_trajectory_v0.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    artifacts = build_trajectory_feature_artifacts(
        _synthetic_data(config),
        config,
        generated_at_utc="2026-08-09T00:00:00Z",
    )
    result = write_trajectory_feature_artifacts(
        artifacts,
        config,
        tmp_path,
        config_path=config_path,
        protocol_path=protocol_path,
    )
    assert result["gate_summary"]["feature_npz_written"] is True
    feature_path = Path(result["paths"]["feature_npz"])
    with np.load(feature_path, allow_pickle=False) as archive:
        assert archive["airm_path_d10"].shape == (8, 10)
        assert archive["le_scalars_11"].shape == (8, 11)
        assert archive["whole_covariances"].shape == (8, 2, 2)
        assert archive["trial_uid"].dtype.kind == "U"
    output = tmp_path / "outputs" / "bnci2014_001_trajectory_v0"
    assert (output / "protocol" / "PROTOCOL_TRAJECTORY_ANATOMY_V0.md").read_text(encoding="utf-8") == protocol_text
    assert (output / "protocol" / "frozen_config.yaml").is_file()
    assert (output / "protocol" / "environment.json").is_file()
    provenance = json.loads((output / "protocol" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["scientific_classification_allowed"] is True
    gate = json.loads((output / "tables" / "trajectory_geometry_gate.json").read_text(encoding="utf-8"))
    assert gate["gate_passed"] is True
    assert gate["feature_npz_sha256"] == hashlib.sha256(feature_path.read_bytes()).hexdigest()


def test_forbidden_session_barrier_runs_before_covariance_property_access() -> None:
    class ForbiddenData:
        @property
        def metadata(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "sample_index": [0],
                    "subject": [1],
                    "session": ["1test"],
                    "run": [0],
                    "trial_id": [1],
                    "trial_uid": ["FORBIDDEN"],
                    "class_label": ["left"],
                }
            )

        @property
        def states(self) -> np.ndarray:
            raise AssertionError("forbidden covariance access")

        @property
        def whole_covariances(self) -> np.ndarray:
            raise AssertionError("forbidden WHOLE access")

    with pytest.raises(TrajectoryFeaturePipelineError, match="forbidden session barrier"):
        build_trajectory_feature_artifacts(ForbiddenData(), _config())
