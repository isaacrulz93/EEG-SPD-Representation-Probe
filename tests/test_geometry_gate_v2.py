"""Synthetic full-scope tests for the official V2 geometry gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.alignment_v2 import FROZEN_CLASSES
from src.data_v2 import V2WholeData
from src.geometry_audit_v2 import GATE_COLUMNS, run_geometry_gate


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_whole() -> V2WholeData:
    rows: list[dict[str, object]] = []
    covariances: list[np.ndarray] = []
    sample = 0
    for subject in range(1, 10):
        trial_id = 0
        for run in range(6):
            for class_label in FROZEN_CLASSES:
                for repetition in range(12):
                    trial_id += 1
                    rows.append(
                        {
                            "sample_index": sample,
                            "subject": subject,
                            "session": "0train",
                            "run": str(run),
                            "trial_id": trial_id,
                            "trial_uid": f"S{subject:02d}_0train_T{trial_id:03d}",
                            "class_label": class_label,
                        }
                    )
                    scale = float(trial_id)
                    log_diagonal = np.asarray(
                        [
                            0.08 * subject + 0.0007 * scale,
                            -0.03 * subject + 0.0011 * scale,
                            0.15 + 0.0009 * scale + 0.0001 * repetition,
                        ]
                    )
                    covariances.append(np.diag(np.exp(log_diagonal)))
                    sample += 1
    covariance_array = np.asarray(covariances, dtype=np.float64)
    covariance_array.setflags(write=False)
    channels = np.asarray(["C1", "C2", "C3"])
    channels.setflags(write=False)
    return V2WholeData(
        covariances=covariance_array,
        metadata=pd.DataFrame.from_records(rows),
        channel_names=channels,
        provenance={"source": "synthetic_test"},
    )


def _config() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/bnci2014_001_geometry_v2.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_full_synthetic_gate_passes_every_subject_t1_and_t2_scope() -> None:
    result = run_geometry_gate(_synthetic_whole(), _config())
    correctness = result.correctness
    assert correctness.columns.tolist() == GATE_COLUMNS
    assert result.gate["classification_gate_pass"] is True
    assert result.gate["n_required_failed"] == 0
    assert result.gate["required_rows"] == result.gate["n_required_rows"]
    assert result.gate["passed_required_rows"] == result.gate["n_required_passed"]
    assert result.gate["failed_required_rows"] == result.gate["n_required_failed"]
    assert correctness.loc[correctness["required"], "passed"].all()
    assert len(result.mean_comparison) == 27  # 9 subjects x (T1 + T2 A/B)

    observed_scopes = correctness[
        ["subject", "protocol", "split", "fit_scope"]
    ].drop_duplicates()
    assert len(observed_scopes) == 27
    assert set(correctness["geometry"]) >= {"RAW", "LE", "AIRM", "EA"}
    assert set(correctness["protocol"]) == {"T1", "T2"}
    assert set(correctness.loc[correctness.protocol == "T2", "split"]) == {"A", "B"}

    required_statistics = {
        "fit_transformed_nonfinite_count",
        "fit_transformed_maximum_relative_symmetry_error",
        "fit_transformed_non_spd_count",
        "fit_transformed_maximum_condition_number",
        "fit_transformed_maximum_relative_frobenius_error",
        "centered_coordinate_maximum_absolute_error",
        "normalized_post_fit_residual",
        "normalized_airm_distance_to_identity",
        "arithmetic_mean_relative_identity_error",
    }
    assert required_statistics.issubset(set(correctness["statistic"]))
    assert {
        "evaluation_transformed_nonfinite_count",
        "evaluation_maximum_absolute_pair_distance_error",
    }.issubset(
        set(correctness.loc[correctness.protocol == "T2", "statistic"])
    )

    mean_columns = {
        "d_le_le_airm",
        "d_airm_le_airm",
        "d_le_le_ea",
        "d_airm_airm_ea",
        "le_dispersion",
        "airm_dispersion",
        "normalized_d_le_le_airm",
        "normalized_d_airm_le_airm",
        "le_center_condition_number",
        "airm_center_condition_number",
        "ea_center_condition_number",
        "le_airm_coordinate_difference_mean_l2",
        "le_airm_coordinate_difference_median_l2",
        "le_airm_coordinate_difference_maximum_l2",
        "airm_tol",
        "airm_maxiter",
        "airm_warning_count",
        "airm_normalized_karcher_residual",
    }
    assert mean_columns.issubset(result.mean_comparison.columns)
    assert np.isfinite(result.mean_comparison[list(mean_columns)].to_numpy()).all()
    assert result.mean_comparison["airm_iteration_count"].isna().all()
    assert (
        result.mean_comparison["airm_termination_reason"]
        == "NA_API_UNAVAILABLE"
    ).all()


def test_non_spd_input_produces_a_written_style_failed_gate_not_an_exception() -> None:
    data = _synthetic_whole()
    broken_covariances = np.array(data.covariances, copy=True)
    broken_covariances[0, 0, 0] = -1.0
    broken = V2WholeData(
        covariances=broken_covariances,
        metadata=data.metadata,
        channel_names=data.channel_names,
        provenance=data.provenance,
    )
    result = run_geometry_gate(broken, _config())
    assert result.gate["classification_gate_pass"] is False
    assert result.gate["n_required_failed"] >= 1
    failures = result.correctness[result.correctness["status"] == "FAILED_TECHNICAL"]
    assert not failures.empty
    assert failures["required"].all()
    assert (~failures["passed"]).all()
