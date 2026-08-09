"""Feature construction and numerical-gate orchestration for trajectory v0.

This module owns only the deterministic bridge from validated WINDOW5 data to
the first ten frozen tables and a local, ignored downstream feature archive.
It performs no classification and no null experiment.  The feature archive is
written only when every required data and geometry gate passes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.geometry_v2 import airm_mean
from src.trajectory_geometry_v0 import (
    AIRM_METRIC,
    LE_METRIC,
    PATH_D10_NAMES,
    SCALAR_11_NAMES,
    TrajectoryGeometryResult,
    check_bag_permutation_invariance,
    compare_airm_centering_isometry,
    compute_five_state_geometry,
    distance_matrix_hard_checks,
    geodesic_endpoint_hard_checks,
    intrinsic_hard_checks,
    spd_stack_hard_checks,
)


__all__ = [
    "FEATURE_TABLE_NAMES",
    "COMMON_PROVENANCE_COLUMNS",
    "TrajectoryFeaturePipelineError",
    "TrajectoryFeatureGateError",
    "TrajectoryFeatureArtifacts",
    "build_trajectory_feature_artifacts",
    "require_scientific_gate",
    "write_trajectory_feature_artifacts",
]


FEATURE_TABLE_NAMES = (
    "dataset_contract",
    "covariance_sanity",
    "trajectory_geometry_correctness",
    "trial_airm_path_features",
    "trial_le_path_features",
    "airm_path_d10",
    "airm_bag_canon_d10",
    "airm_bag_sorted_d10",
    "le_path_d10",
    "le_bag_canon_d10",
)
COMMON_PROVENANCE_COLUMNS = (
    "protocol_version",
    "protocol_sha256",
    "config_sha256",
    "seed",
    "session",
    "generated_at_utc",
    "status",
)
TRIAL_IDENTITY_COLUMNS = (
    "sample_index",
    "subject",
    "run",
    "trial_id",
    "trial_uid",
    "class_label",
)

DATASET_CONTRACT_COLUMNS = COMMON_PROVENANCE_COLUMNS + (
    "check",
    "observed",
    "expected",
    "comparator",
    "required",
    "passed",
    "failure_message",
)
COVARIANCE_SANITY_COLUMNS = COMMON_PROVENANCE_COLUMNS + TRIAL_IDENTITY_COLUMNS + (
    "window_index",
    "symmetry_relative_error",
    "min_eigenvalue",
    "max_eigenvalue",
    "condition_number",
    "has_nan",
    "has_inf",
    "is_spd",
    "required",
    "passed",
)
GEOMETRY_CORRECTNESS_COLUMNS = COMMON_PROVENANCE_COLUMNS + (
    "geometry",
    "subject",
    "sample_index",
    "trial_uid",
    "check",
    "statistic",
    "value",
    "threshold",
    "comparator",
    "absolute_error",
    "relative_error",
    "required",
    "passed",
    "failure_message",
)
DESCRIPTIVE_NAMES = (
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
)
TRIAL_FEATURE_COLUMNS = (
    COMMON_PROVENANCE_COLUMNS
    + TRIAL_IDENTITY_COLUMNS
    + ("geometry",)
    + DESCRIPTIVE_NAMES[:4]
    + SCALAR_11_NAMES[:4]
    + DESCRIPTIVE_NAMES[4:7]
    + SCALAR_11_NAMES[4:6]
    + DESCRIPTIVE_NAMES[7:]
    + SCALAR_11_NAMES[6:]
    + ("degenerate",)
)
PATH_TABLE_COLUMNS = (
    COMMON_PROVENANCE_COLUMNS
    + TRIAL_IDENTITY_COLUMNS
    + ("geometry", "representation")
    + PATH_D10_NAMES
)
BAG_CANON_NAMES = tuple(f"bag{index:02d}" for index in range(1, 11))
BAG_CANON_TABLE_COLUMNS = (
    COMMON_PROVENANCE_COLUMNS
    + TRIAL_IDENTITY_COLUMNS
    + ("geometry", "representation")
    + BAG_CANON_NAMES
    + ("canonical_permutation",)
)
BAG_SORTED_NAMES = tuple(f"sorted{index:02d}" for index in range(1, 11))
BAG_SORTED_TABLE_COLUMNS = (
    COMMON_PROVENANCE_COLUMNS
    + TRIAL_IDENTITY_COLUMNS
    + ("geometry", "representation")
    + BAG_SORTED_NAMES
)

_FROZEN_TOLERANCE_PATHS = {
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
}


class TrajectoryFeaturePipelineError(RuntimeError):
    """Raised when orchestration cannot honor its frozen input/output contract."""


class TrajectoryFeatureGateError(TrajectoryFeaturePipelineError):
    """Raised when a caller attempts scientific work after a failed gate."""


@dataclass(frozen=True)
class TrajectoryFeatureArtifacts:
    tables: Mapping[str, pd.DataFrame]
    arrays: Mapping[str, np.ndarray]
    gate_summary: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @property
    def gate_passed(self) -> bool:
        return bool(self.gate_summary["gate_passed"])


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_config(config: Mapping[str, Any]) -> None:
    for section in ("project", "dataset", "window5", "geometry", "representations"):
        if section not in config or not isinstance(config[section], Mapping):
            raise TrajectoryFeaturePipelineError(f"missing config section {section!r}")
    project = config["project"]
    if str(project.get("protocol_version")) != "0.0":
        raise TrajectoryFeaturePipelineError("protocol_version must be exactly '0.0'")
    if int(project.get("seed", -1)) != 20260809:
        raise TrajectoryFeaturePipelineError("master seed must be exactly 20260809")
    hard = config["geometry"].get("hard_gate")
    if not isinstance(hard, Mapping):
        raise TrajectoryFeaturePipelineError("geometry.hard_gate is missing")
    for key, expected in _FROZEN_TOLERANCE_PATHS.items():
        observed = float(hard.get(key, float("nan")))
        if not np.isfinite(observed) or observed != expected:
            raise TrajectoryFeaturePipelineError(
                f"frozen hard-gate tolerance mismatch for {key}: "
                f"expected {expected}, observed {observed}"
            )
    mean = config["geometry"].get("airm_mean")
    if not isinstance(mean, Mapping):
        raise TrajectoryFeaturePipelineError("geometry.airm_mean is missing")
    if (
        float(mean.get("tol", np.nan)) != 1e-9
        or int(mean.get("maxiter", -1)) != 100
        or float(mean.get("normalized_karcher_residual_max", np.nan)) != 1e-7
    ):
        raise TrajectoryFeaturePipelineError("frozen AIRM mean settings changed")
    if tuple(config["representations"].get("path_d10_columns", ())) != PATH_D10_NAMES:
        raise TrajectoryFeaturePipelineError("PATH_D10 column order changed")
    if tuple(config["representations"].get("scalar_columns", ())) != SCALAR_11_NAMES:
        raise TrajectoryFeaturePipelineError("SCALARS_11 column order changed")


def _prefix(
    config: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    generated_at_utc: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "protocol_version": str(config["project"]["protocol_version"]),
        "protocol_sha256": str(config["project"]["protocol_sha256"]),
        "config_sha256": str(
            data_provenance.get("config_sha256", _canonical_hash(config))
        ),
        "seed": int(config["project"]["seed"]),
        "session": str(config["dataset"]["allowed_session"]),
        "generated_at_utc": generated_at_utc,
        "status": "PASS" if passed else "FAIL",
    }


def _trial_identity(row: pd.Series) -> dict[str, Any]:
    return {
        "sample_index": int(row["sample_index"]),
        "subject": int(row["subject"]),
        "run": int(row["run"]),
        "trial_id": int(row["trial_id"]),
        "trial_uid": str(row["trial_uid"]),
        "class_label": str(row["class_label"]),
    }


def _frame(records: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=list(columns))


def _dataset_contract(
    states: np.ndarray,
    whole: np.ndarray,
    metadata: pd.DataFrame,
    channels: np.ndarray,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    generated_at_utc: str,
) -> pd.DataFrame:
    dataset = config["dataset"]
    expected_trials = int(dataset["expected_trials"])
    expected_shape = tuple(int(v) for v in config["window5"]["expected_trial_tensor_shape"])
    expected_whole = (expected_trials, len(dataset["eeg_channels"]), len(dataset["eeg_channels"]))
    checks: list[tuple[str, Any, Any, bool]] = []
    checks.append(("trial_count", len(metadata), expected_trials, len(metadata) == expected_trials))
    checks.append(("window_state_shape", list(states.shape), list(expected_shape), states.shape == expected_shape))
    checks.append(("whole_covariance_shape", list(whole.shape), list(expected_whole), whole.shape == expected_whole))
    checks.append(("channel_order", channels.astype(str).tolist(), list(dataset["eeg_channels"]), np.array_equal(channels.astype(str), np.asarray(dataset["eeg_channels"], dtype=str))))
    checks.append(("session_barrier", sorted(metadata["session"].astype(str).unique().tolist()), [str(dataset["allowed_session"])], set(metadata["session"].astype(str)) == {str(dataset["allowed_session"])}))
    checks.append(("sample_index_sequence", metadata["sample_index"].astype(int).tolist(), list(range(expected_trials)), np.array_equal(metadata["sample_index"].to_numpy(dtype=int), np.arange(expected_trials))))
    checks.append(("trial_uid_unique", int(metadata["trial_uid"].nunique()), expected_trials, metadata["trial_uid"].nunique() == expected_trials))
    checks.append(("subject_set", sorted(metadata["subject"].astype(int).unique().tolist()), list(dataset["subjects"]), tuple(sorted(metadata["subject"].astype(int).unique())) == tuple(dataset["subjects"])))
    checks.append(("run_set", sorted(metadata["run"].astype(int).unique().tolist()), list(dataset["runs"]), tuple(sorted(metadata["run"].astype(int).unique())) == tuple(dataset["runs"])))
    checks.append(("class_set", sorted(metadata["class_label"].astype(str).unique().tolist()), sorted(str(v) for v in dataset["classes"]), set(metadata["class_label"].astype(str)) == set(str(v) for v in dataset["classes"])))
    group_specs = (
        (("subject",), int(dataset["expected_trials_per_subject"]), "trials_per_subject"),
        (("subject", "class_label"), int(dataset["expected_trials_per_subject_class"]), "trials_per_subject_class"),
        (("subject", "run"), int(dataset["expected_trials_per_subject_run"]), "trials_per_subject_run"),
        (("subject", "run", "class_label"), int(dataset["expected_trials_per_subject_run_class"]), "trials_per_subject_run_class"),
    )
    for columns, expected, label in group_specs:
        counts = metadata.groupby(list(columns), observed=True, sort=True).size()
        observed = sorted(int(value) for value in counts.tolist())
        checks.append((label, observed, [expected] * len(counts), len(counts) > 0 and bool((counts == expected).all())))
    records: list[dict[str, Any]] = []
    for check, observed, expected, passed in checks:
        records.append(
            {
                **_prefix(config, provenance, generated_at_utc, passed),
                "check": check,
                "observed": _json_text(observed),
                "expected": _json_text(expected),
                "comparator": "==",
                "required": True,
                "passed": bool(passed),
                "failure_message": "" if passed else f"dataset contract failed: {check}",
            }
        )
    return _frame(records, DATASET_CONTRACT_COLUMNS)


def _covariance_sanity(
    states: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    generated_at_utc: str,
) -> pd.DataFrame:
    n_trials, n_windows, n_channels, _ = states.shape
    flat = states.reshape(n_trials * n_windows, n_channels, n_channels)
    has_nan = np.isnan(flat).any(axis=(1, 2))
    has_inf = np.isinf(flat).any(axis=(1, 2))
    finite = ~(has_nan | has_inf)
    symmetry = np.linalg.norm(flat - flat.transpose(0, 2, 1), axis=(1, 2)) / np.maximum(
        np.linalg.norm(flat, axis=(1, 2)), np.finfo(np.float64).tiny
    )
    minimum = np.full(len(flat), np.nan)
    maximum = np.full(len(flat), np.nan)
    finite_positions = np.flatnonzero(finite)
    if len(finite_positions):
        eigenvalues = np.linalg.eigvalsh(
            0.5 * (flat[finite_positions] + flat[finite_positions].transpose(0, 2, 1))
        )
        minimum[finite_positions] = eigenvalues[:, 0]
        maximum[finite_positions] = eigenvalues[:, -1]
    condition = np.full(len(flat), np.inf)
    positive = finite & (minimum > 0.0)
    condition[positive] = maximum[positive] / minimum[positive]
    hard = config["geometry"]["hard_gate"]
    is_spd = (
        finite
        & (symmetry <= float(hard["symmetry_relative_error_max"]))
        & (minimum > 0.0)
        & (condition <= float(hard["condition_number_max"]))
    )
    records: list[dict[str, Any]] = []
    for flat_index in range(len(flat)):
        trial_position, window_position = divmod(flat_index, n_windows)
        row = metadata.iloc[trial_position]
        passed = bool(is_spd[flat_index])
        records.append(
            {
                **_prefix(config, provenance, generated_at_utc, passed),
                **_trial_identity(row),
                "window_index": window_position + 1,
                "symmetry_relative_error": float(symmetry[flat_index]),
                "min_eigenvalue": float(minimum[flat_index]),
                "max_eigenvalue": float(maximum[flat_index]),
                "condition_number": float(condition[flat_index]),
                "has_nan": bool(has_nan[flat_index]),
                "has_inf": bool(has_inf[flat_index]),
                "is_spd": passed,
                "required": True,
                "passed": passed,
            }
        )
    return _frame(records, COVARIANCE_SANITY_COLUMNS)


class _GeometryRows:
    def __init__(
        self,
        config: Mapping[str, Any],
        provenance: Mapping[str, Any],
        generated_at_utc: str,
    ) -> None:
        self.config = config
        self.provenance = provenance
        self.generated_at_utc = generated_at_utc
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        *,
        geometry: str,
        subject: int,
        sample_index: int | None,
        trial_uid: str,
        check: str,
        statistic: str,
        value: float,
        threshold: float,
        comparator: str,
        passed: bool,
        absolute_error: float = np.nan,
        relative_error: float = np.nan,
        required: bool = True,
        failure_message: str = "",
    ) -> None:
        self.records.append(
            {
                **_prefix(self.config, self.provenance, self.generated_at_utc, passed),
                "geometry": geometry,
                "subject": int(subject),
                "sample_index": pd.NA if sample_index is None else int(sample_index),
                "trial_uid": str(trial_uid),
                "check": check,
                "statistic": statistic,
                "value": float(value),
                "threshold": float(threshold),
                "comparator": comparator,
                "absolute_error": float(absolute_error),
                "relative_error": float(relative_error),
                "required": bool(required),
                "passed": bool(passed),
                "failure_message": "" if passed else failure_message,
            }
        )

    def failure(
        self,
        geometry: str,
        row: pd.Series,
        error: BaseException,
    ) -> None:
        self.add(
            geometry=geometry,
            subject=int(row["subject"]),
            sample_index=int(row["sample_index"]),
            trial_uid=str(row["trial_uid"]),
            check="technical_geometry_failure",
            statistic=type(error).__name__,
            value=np.inf,
            threshold=0.0,
            comparator="must_not_occur",
            passed=False,
            failure_message=f"{type(error).__name__}: {error}",
        )

    def frame(self) -> pd.DataFrame:
        result = _frame(self.records, GEOMETRY_CORRECTNESS_COLUMNS)
        if len(result):
            result["sample_index"] = pd.array(result["sample_index"], dtype="Int64")
        return result


def _append_result_checks(
    rows: _GeometryRows,
    result: TrajectoryGeometryResult,
    metadata_row: pd.Series,
    config: Mapping[str, Any],
) -> bool:
    hard = config["geometry"]["hard_gate"]
    geometry = result.metric
    subject = int(metadata_row["subject"])
    sample = int(metadata_row["sample_index"])
    uid = str(metadata_row["trial_uid"])
    before = len(rows.records)
    distance = distance_matrix_hard_checks(result.distance_matrix)
    distance_specs = (
        ("distance_symmetry", "maximum_absolute_error", distance.maximum_absolute_symmetry_error, float(hard["distance_symmetry_absolute_error_max"]), "<="),
        ("distance_diagonal", "maximum_absolute_error", distance.maximum_absolute_diagonal_error, float(hard["distance_diagonal_absolute_error_max"]), "<="),
        ("distance_nonnegative", "minimum_distance", distance.minimum_distance, -float(hard["distance_negative_tolerance"]), ">="),
        ("triangle_inequality", "maximum_excess_over_tolerance", distance.maximum_triangle_excess_over_tolerance, 0.0, "<="),
    )
    for check, statistic, value, threshold, comparator in distance_specs:
        passed = bool(np.isfinite(value) and (value <= threshold if comparator == "<=" else value >= threshold))
        rows.add(geometry=geometry, subject=subject, sample_index=sample, trial_uid=uid, check=check, statistic=statistic, value=value, threshold=threshold, comparator=comparator, passed=passed, failure_message=f"{check} failed")

    mean_checks = spd_stack_hard_checks(result.mean_matrix)
    mean_specs = (
        ("barycenter_symmetry", mean_checks.maximum_relative_symmetry_error, float(hard["symmetry_relative_error_max"]), "<="),
        ("barycenter_positive_definite", mean_checks.minimum_eigenvalue, 0.0, ">"),
        ("barycenter_condition", mean_checks.maximum_condition_number, float(hard["condition_number_max"]), "<="),
    )
    for check, value, threshold, comparator in mean_specs:
        passed = bool(np.isfinite(value) and ((value <= threshold) if comparator == "<=" else value > threshold))
        rows.add(geometry=geometry, subject=subject, sample_index=sample, trial_uid=uid, check=check, statistic="value", value=value, threshold=threshold, comparator=comparator, passed=passed, failure_message=f"{check} failed")
    warning_count = len(result.mean_result.warning_messages)
    warning_detail = "; ".join(result.mean_result.warning_messages)
    rows.add(geometry=geometry, subject=subject, sample_index=sample, trial_uid=uid, check="barycenter_warning", statistic="captured_warning_count", value=warning_count, threshold=0.0, comparator="<=", passed=warning_count == 0, failure_message=warning_detail or "required barycenter warning occurred")
    if geometry == AIRM_METRIC:
        residual = result.normalized_karcher_residual
        residual_value = float(residual) if residual is not None else np.inf
        threshold = float(config["geometry"]["airm_mean"]["normalized_karcher_residual_max"])
        rows.add(geometry=geometry, subject=subject, sample_index=sample, trial_uid=uid, check="barycenter_karcher_residual", statistic="normalized_residual", value=residual_value, threshold=threshold, comparator="<=", passed=np.isfinite(residual_value) and residual_value <= threshold, failure_message="AIRM barycenter residual failed")

    intrinsic = intrinsic_hard_checks(result)
    length_tolerance = float(hard["path_endpoint_absolute_tolerance"]) + float(hard["path_endpoint_relative_tolerance"]) * abs(result.total_path_length)
    intrinsic_specs = (
        ("feature_finite", "nonfinite_present", 0.0 if intrinsic.finite_features else 1.0, 0.0, "<="),
        ("path_endpoint_inequality", "endpoint_minus_length", result.endpoint_distance - result.total_path_length, length_tolerance, "<="),
        ("path_nonzero", "total_path_length", result.total_path_length, float(hard["zero_length_epsilon"]), ">"),
        ("efficiency_lower_bound", "efficiency", result.efficiency, -float(hard["efficiency_bound_tolerance"]), ">="),
        ("efficiency_upper_bound", "efficiency", result.efficiency, 1.0 + float(hard["efficiency_bound_tolerance"]), "<="),
        ("angle_degeneracy", "degenerate_angle_count", float(intrinsic.degenerate_angle_count), 0.0, "<="),
        ("cosine_domain", "maximum_domain_excess", intrinsic.maximum_cosine_domain_excess, float(hard["cosine_domain_tolerance"]), "<="),
        ("angle_lower_bound", "minimum_angle", intrinsic.minimum_angle, -float(hard["angle_bound_tolerance"]), ">="),
        ("angle_upper_bound", "maximum_angle", intrinsic.maximum_angle, float(np.pi + hard["angle_bound_tolerance"]), "<="),
        ("deviation_nonnegative", "minimum_deviation", intrinsic.minimum_deviation, -float(hard["deviation_negative_tolerance"]), ">="),
    )
    for check, statistic, value, threshold, comparator in intrinsic_specs:
        if comparator == "<=":
            passed = bool(np.isfinite(value) and value <= threshold)
        elif comparator == ">=":
            passed = bool(np.isfinite(value) and value >= threshold)
        else:
            passed = bool(np.isfinite(value) and value > threshold)
        rows.add(geometry=geometry, subject=subject, sample_index=sample, trial_uid=uid, check=check, statistic=statistic, value=value, threshold=threshold, comparator=comparator, passed=passed, failure_message=f"{check} failed")
    # The actual endpoints are checked by the caller, which has the states.
    return bool(all(record["passed"] for record in rows.records[before:] if record["required"]))


def _append_synthetic_geodesic_checks(
    rows: _GeometryRows,
    config: Mapping[str, Any],
) -> None:
    """Persist the preregistered implementation-level endpoint check once."""

    first_factor = np.asarray(
        [[1.1, 0.2, -0.1], [0.0, 0.9, 0.3], [0.2, -0.1, 1.2]],
        dtype=np.float64,
    )
    second_factor = np.asarray(
        [[0.8, -0.3, 0.2], [0.4, 1.0, 0.1], [-0.2, 0.3, 1.1]],
        dtype=np.float64,
    )
    first = first_factor @ first_factor.T + np.eye(3)
    second = second_factor @ second_factor.T + np.eye(3)
    threshold = float(config["geometry"]["hard_gate"]["geodesic_endpoint_relative_error_max"])
    for metric in (AIRM_METRIC, LE_METRIC):
        endpoints = geodesic_endpoint_hard_checks(first, second, metric)
        for statistic, value in (
            ("t0_relative_error", endpoints.t0_relative_error),
            ("t1_relative_error", endpoints.t1_relative_error),
        ):
            passed = bool(np.isfinite(value) and value <= threshold)
            rows.add(
                geometry=metric,
                subject=0,
                sample_index=None,
                trial_uid="SYNTHETIC_ENDPOINT_GATE",
                check="synthetic_geodesic_endpoint",
                statistic=statistic,
                value=value,
                threshold=threshold,
                comparator="<=",
                passed=passed,
                failure_message="synthetic geodesic endpoint reconstruction failed",
            )


def _feature_record(
    result: TrajectoryGeometryResult | None,
    row: pd.Series,
    metric: str,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    generated_at_utc: str,
    passed: bool,
) -> dict[str, Any]:
    identity = _trial_identity(row)
    record = {**_prefix(config, provenance, generated_at_utc, passed), **identity, "geometry": metric}
    if result is None:
        for name in DESCRIPTIVE_NAMES + SCALAR_11_NAMES:
            record[name] = np.nan
        record["degenerate"] = True
        return record
    for name, value in zip(DESCRIPTIVE_NAMES[:4], result.steps, strict=True):
        record[name] = float(value)
    record.update(result.scalar_dict)
    for name, value in zip(DESCRIPTIVE_NAMES[4:7], result.angles, strict=True):
        record[name] = float(value)
    for name, value in zip(DESCRIPTIVE_NAMES[7:], result.deviations, strict=True):
        record[name] = float(value)
    record["degenerate"] = bool(result.path_degenerate or result.angle_degenerate_mask.any())
    return record


def _descriptor_record(
    values: np.ndarray | None,
    names: Sequence[str],
    row: pd.Series,
    metric: str,
    representation: str,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    generated_at_utc: str,
    passed: bool,
    canonical_permutation: str | None = None,
) -> dict[str, Any]:
    record = {
        **_prefix(config, provenance, generated_at_utc, passed),
        **_trial_identity(row),
        "geometry": metric,
        "representation": representation,
    }
    vector = np.full(len(names), np.nan) if values is None else np.asarray(values)
    record.update({name: float(value) for name, value in zip(names, vector, strict=True)})
    if canonical_permutation is not None or representation == "BAG_CANON_D10":
        record["canonical_permutation"] = canonical_permutation or ""
    return record


def _subject_whole_means(
    whole: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    rows: _GeometryRows,
) -> dict[int, np.ndarray]:
    means: dict[int, np.ndarray] = {}
    mean_config = config["geometry"]["airm_mean"]
    hard = config["geometry"]["hard_gate"]
    for subject in (int(value) for value in config["dataset"]["subjects"]):
        positions = np.flatnonzero(metadata["subject"].to_numpy(dtype=int) == subject)
        try:
            input_checks = spd_stack_hard_checks(whole[positions])
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=None, trial_uid="", check="subject_whole_input", statistic="nonfinite_count", value=input_checks.nonfinite_count, threshold=0.0, comparator="<=", passed=input_checks.nonfinite_count == 0, failure_message="subject WHOLE covariances contain NaN/Inf")
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=None, trial_uid="", check="subject_whole_input", statistic="maximum_relative_symmetry_error", value=input_checks.maximum_relative_symmetry_error, threshold=float(hard["symmetry_relative_error_max"]), comparator="<=", passed=np.isfinite(input_checks.maximum_relative_symmetry_error) and input_checks.maximum_relative_symmetry_error <= float(hard["symmetry_relative_error_max"]), failure_message="subject WHOLE covariance symmetry failed")
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=None, trial_uid="", check="subject_whole_input", statistic="minimum_eigenvalue", value=input_checks.minimum_eigenvalue, threshold=0.0, comparator=">", passed=np.isfinite(input_checks.minimum_eigenvalue) and input_checks.minimum_eigenvalue > 0.0, failure_message="subject WHOLE covariances are not SPD")
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=None, trial_uid="", check="subject_whole_input", statistic="maximum_condition_number", value=input_checks.maximum_condition_number, threshold=float(hard["condition_number_max"]), comparator="<=", passed=np.isfinite(input_checks.maximum_condition_number) and input_checks.maximum_condition_number <= float(hard["condition_number_max"]), failure_message="subject WHOLE covariance conditioning failed")
            result = airm_mean(whole[positions], tol=float(mean_config["tol"]), maxiter=int(mean_config["maxiter"]))
            mean_checks = spd_stack_hard_checks(result.matrix)
            residual_pass = bool(np.isfinite(result.normalized_post_residual) and result.normalized_post_residual <= float(mean_config["normalized_karcher_residual_max"]))
            warning_pass = len(result.warning_messages) == 0
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=None, trial_uid="", check="subject_whole_mean", statistic="normalized_karcher_residual", value=result.normalized_post_residual, threshold=float(mean_config["normalized_karcher_residual_max"]), comparator="<=", passed=residual_pass, failure_message="subject WHOLE AIRM mean residual failed")
            warning_detail = "; ".join(result.warning_messages)
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=None, trial_uid="", check="subject_whole_mean", statistic="captured_warning_count", value=len(result.warning_messages), threshold=0.0, comparator="<=", passed=warning_pass, failure_message=warning_detail or "subject WHOLE AIRM mean warning occurred")
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=None, trial_uid="", check="subject_whole_mean", statistic="maximum_relative_symmetry_error", value=mean_checks.maximum_relative_symmetry_error, threshold=float(hard["symmetry_relative_error_max"]), comparator="<=", passed=np.isfinite(mean_checks.maximum_relative_symmetry_error) and mean_checks.maximum_relative_symmetry_error <= float(hard["symmetry_relative_error_max"]), failure_message="subject WHOLE AIRM mean symmetry failed")
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=None, trial_uid="", check="subject_whole_mean", statistic="minimum_eigenvalue", value=mean_checks.minimum_eigenvalue, threshold=0.0, comparator=">", passed=np.isfinite(mean_checks.minimum_eigenvalue) and mean_checks.minimum_eigenvalue > 0.0, failure_message="subject WHOLE AIRM mean is not SPD")
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=None, trial_uid="", check="subject_whole_mean", statistic="maximum_condition_number", value=mean_checks.maximum_condition_number, threshold=float(hard["condition_number_max"]), comparator="<=", passed=np.isfinite(mean_checks.maximum_condition_number) and mean_checks.maximum_condition_number <= float(hard["condition_number_max"]), failure_message="subject WHOLE AIRM mean conditioning failed")
            if input_checks.passed and mean_checks.passed and residual_pass and warning_pass:
                means[subject] = result.matrix
        except Exception as error:  # diagnostics must survive a numerical failure
            dummy = pd.Series({"subject": subject, "sample_index": -1, "trial_uid": ""})
            rows.failure(AIRM_METRIC, dummy, error)
    return means


def _selected_bag_positions(
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[int]:
    selected: list[int] = []
    for subject in config["dataset"]["subjects"]:
        for class_label in config["dataset"]["classes"]:
            subset = metadata[
                (metadata["subject"].astype(int) == int(subject))
                & (metadata["class_label"].astype(str) == str(class_label))
            ].sort_values(["trial_id", "sample_index", "trial_uid"], kind="stable")
            if subset.empty:
                continue
            selected.append(int(subset.index[0]))
    return selected


def build_trajectory_feature_artifacts(
    data: Any,
    config: Mapping[str, Any],
    *,
    generated_at_utc: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> TrajectoryFeatureArtifacts:
    """Build exact tables 1--10, downstream arrays, and a hard-gate summary.

    ``data`` must expose the final read-only loader contract: ``states``,
    ``whole_covariances``, ``metadata``, ``channel_names``, and ``provenance``.
    No file is written by this function.
    """

    _validate_config(config)
    timestamp = generated_at_utc or _utc_now()
    metadata = data.metadata.reset_index(drop=True).copy(deep=True)
    required_metadata = set(TRIAL_IDENTITY_COLUMNS) | {"session"}
    missing = required_metadata - set(metadata)
    if missing:
        raise TrajectoryFeaturePipelineError(f"trial metadata is missing {sorted(missing)}")
    allowed_session = str(config["dataset"]["allowed_session"])
    observed_sessions = set(metadata["session"].astype(str))
    # This barrier intentionally precedes access to either covariance property.
    if observed_sessions != {allowed_session}:
        raise TrajectoryFeaturePipelineError(
            "forbidden session barrier: expected only "
            f"{allowed_session!r}, observed {sorted(observed_sessions)!r}"
        )
    states = np.asarray(data.states, dtype=np.float64)
    whole = np.asarray(data.whole_covariances, dtype=np.float64)
    channels = np.asarray(data.channel_names).astype(str)
    provenance = dict(data.provenance)
    if states.ndim != 4 or states.shape[1] != 5 or states.shape[2] != states.shape[3]:
        raise TrajectoryFeaturePipelineError(f"states must have N x 5 x C x C shape, got {states.shape}")
    if whole.ndim != 3 or whole.shape[1:] != states.shape[2:] or len(whole) != len(states):
        raise TrajectoryFeaturePipelineError("WHOLE covariance shape does not align with states")
    if len(metadata) != len(states):
        raise TrajectoryFeaturePipelineError("metadata and state trial counts differ")

    dataset_contract = _dataset_contract(states, whole, metadata, channels, config, provenance, timestamp)
    covariance_sanity = _covariance_sanity(states, metadata, config, provenance, timestamp)
    rows = _GeometryRows(config, provenance, timestamp)
    subject_means = _subject_whole_means(whole, metadata, config, rows)
    _append_synthetic_geodesic_checks(rows, config)
    n_trials, _, n_channels, _ = states.shape
    arrays: dict[str, np.ndarray] = {
        "airm_distance_matrices": np.full((n_trials, 5, 5), np.nan),
        "le_distance_matrices": np.full((n_trials, 5, 5), np.nan),
        "airm_path_d10": np.full((n_trials, 10), np.nan),
        "le_path_d10": np.full((n_trials, 10), np.nan),
        "airm_bag_canon_d10": np.full((n_trials, 10), np.nan),
        "le_bag_canon_d10": np.full((n_trials, 10), np.nan),
        "airm_bag_sorted_d10": np.full((n_trials, 10), np.nan),
        "le_bag_sorted_d10": np.full((n_trials, 10), np.nan),
        "airm_scalars_11": np.full((n_trials, 11), np.nan),
        "le_scalars_11": np.full((n_trials, 11), np.nan),
        "airm_canonical_permutation": np.full((n_trials, 5), -1, dtype=np.int64),
        "le_canonical_permutation": np.full((n_trials, 5), -1, dtype=np.int64),
        "local_airm_barycenters": np.full((n_trials, n_channels, n_channels), np.nan),
        "whole_covariances": np.asarray(whole, dtype=np.float64).copy(),
    }
    trial_results: dict[str, list[TrajectoryGeometryResult | None]] = {
        AIRM_METRIC: [None] * n_trials,
        LE_METRIC: [None] * n_trials,
    }
    trial_pass: dict[str, np.ndarray] = {
        AIRM_METRIC: np.zeros(n_trials, dtype=bool),
        LE_METRIC: np.zeros(n_trials, dtype=bool),
    }
    for position in range(n_trials):
        row = metadata.iloc[position]
        for metric, prefix_name in ((AIRM_METRIC, "airm"), (LE_METRIC, "le")):
            try:
                result = compute_five_state_geometry(states[position], metric)
                trial_results[metric][position] = result
                arrays[f"{prefix_name}_distance_matrices"][position] = result.distance_matrix
                arrays[f"{prefix_name}_path_d10"][position] = result.path_d10
                arrays[f"{prefix_name}_bag_canon_d10"][position] = result.bag_canon_d10
                arrays[f"{prefix_name}_bag_sorted_d10"][position] = result.bag_sorted_d10
                arrays[f"{prefix_name}_scalars_11"][position] = result.scalar_vector
                arrays[f"{prefix_name}_canonical_permutation"][position] = result.bag_canon.permutation
                if metric == AIRM_METRIC:
                    arrays["local_airm_barycenters"][position] = result.mean_matrix
                passed = _append_result_checks(rows, result, row, config)
                trial_pass[metric][position] = passed
            except Exception as error:
                rows.failure(metric, row, error)
        if progress is not None and ((position + 1) % 25 == 0 or position + 1 == n_trials):
            progress(position + 1, n_trials)

    hard = config["geometry"]["hard_gate"]
    for position in range(n_trials):
        row = metadata.iloc[position]
        subject = int(row["subject"])
        if subject not in subject_means or trial_results[AIRM_METRIC][position] is None:
            rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=int(row["sample_index"]), trial_uid=str(row["trial_uid"]), check="centering_isometry", statistic="all_descriptors", value=np.inf, threshold=float(hard["centering_isometry_absolute_error_max"]), comparator="absolute_and_relative<=", passed=False, failure_message="subject WHOLE mean or AIRM feature unavailable")
            continue
        try:
            check = compare_airm_centering_isometry(states[position], subject_means[subject])
            specs = (
                ("distance_matrix", check.distance_maximum_absolute_error, check.distance_maximum_relative_error),
                ("path_d10", check.path_maximum_absolute_error, check.path_maximum_relative_error),
                ("bag_canon_d10", check.bag_maximum_absolute_error, check.bag_maximum_relative_error),
                ("total_path_length", check.path_length_absolute_error, check.path_length_relative_error),
            )
            for statistic, absolute, relative in specs:
                passed = bool(absolute <= float(hard["centering_isometry_absolute_error_max"]) and relative <= float(hard["centering_isometry_relative_error_max"]))
                rows.add(geometry=AIRM_METRIC, subject=subject, sample_index=int(row["sample_index"]), trial_uid=str(row["trial_uid"]), check="centering_isometry", statistic=statistic, value=max(absolute, relative), threshold=max(float(hard["centering_isometry_absolute_error_max"]), float(hard["centering_isometry_relative_error_max"])), comparator="absolute_and_relative<=", absolute_error=absolute, relative_error=relative, passed=passed, failure_message=f"AIRM centering changed {statistic}")
        except Exception as error:
            rows.failure(AIRM_METRIC, row, error)

    selected_positions = _selected_bag_positions(metadata, config)
    expected_selected = int(config["geometry"]["bag_validation"]["expected_trials"])
    selection_count_pass = len(selected_positions) == expected_selected
    rows.add(geometry="ALL", subject=0, sample_index=None, trial_uid="", check="bag_validation_selection", statistic="selected_trial_count", value=len(selected_positions), threshold=expected_selected, comparator="==", passed=selection_count_pass, failure_message="BAG validation trial selection count changed")
    for position in selected_positions:
        row = metadata.iloc[position]
        for metric, prefix_name in ((AIRM_METRIC, "airm"), (LE_METRIC, "le")):
            matrix = arrays[f"{prefix_name}_distance_matrices"][position]
            if not np.isfinite(matrix).all():
                rows.add(geometry=metric, subject=int(row["subject"]), sample_index=int(row["sample_index"]), trial_uid=str(row["trial_uid"]), check="bag_permutation_invariance", statistic="maximum_absolute_error", value=np.inf, threshold=float(hard["bag_invariance_absolute_error_max"]), comparator="<=", passed=False, failure_message="distance matrix unavailable for BAG validation")
                continue
            check = check_bag_permutation_invariance(matrix)
            rows.add(geometry=metric, subject=int(row["subject"]), sample_index=int(row["sample_index"]), trial_uid=str(row["trial_uid"]), check="bag_permutation_invariance", statistic="permutation_count", value=check.permutation_count, threshold=int(config["geometry"]["bag_validation"]["permutations"]), comparator="==", passed=check.permutation_count == int(config["geometry"]["bag_validation"]["permutations"]), failure_message="BAG permutation count changed")
            rows.add(geometry=metric, subject=int(row["subject"]), sample_index=int(row["sample_index"]), trial_uid=str(row["trial_uid"]), check="bag_permutation_invariance", statistic="maximum_absolute_error", value=check.maximum_absolute_error, threshold=float(hard["bag_invariance_absolute_error_max"]), comparator="<=", absolute_error=check.maximum_absolute_error, passed=check.maximum_absolute_error <= float(hard["bag_invariance_absolute_error_max"]), failure_message="BAG canonical vector changed under permutation")

    geometry_correctness = rows.frame()
    feature_records = {AIRM_METRIC: [], LE_METRIC: []}
    path_records = {AIRM_METRIC: [], LE_METRIC: []}
    bag_records = {AIRM_METRIC: [], LE_METRIC: []}
    sorted_records: list[dict[str, Any]] = []
    for position in range(n_trials):
        row = metadata.iloc[position]
        for metric, prefix_name in ((AIRM_METRIC, "airm"), (LE_METRIC, "le")):
            result = trial_results[metric][position]
            passed = bool(trial_pass[metric][position])
            feature_records[metric].append(_feature_record(result, row, metric, config, provenance, timestamp, passed))
            path_records[metric].append(_descriptor_record(None if result is None else result.path_d10, PATH_D10_NAMES, row, metric, "PATH_D10", config, provenance, timestamp, passed))
            bag_records[metric].append(_descriptor_record(None if result is None else result.bag_canon_d10, BAG_CANON_NAMES, row, metric, "BAG_CANON_D10", config, provenance, timestamp, passed, None if result is None else result.bag_canon.permutation_one_based))
            if metric == AIRM_METRIC:
                sorted_records.append(_descriptor_record(None if result is None else result.bag_sorted_d10, BAG_SORTED_NAMES, row, metric, "BAG_SORTED_D10", config, provenance, timestamp, passed))

    tables = {
        "dataset_contract": dataset_contract,
        "covariance_sanity": covariance_sanity,
        "trajectory_geometry_correctness": geometry_correctness,
        "trial_airm_path_features": _frame(feature_records[AIRM_METRIC], TRIAL_FEATURE_COLUMNS),
        "trial_le_path_features": _frame(feature_records[LE_METRIC], TRIAL_FEATURE_COLUMNS),
        "airm_path_d10": _frame(path_records[AIRM_METRIC], PATH_TABLE_COLUMNS),
        "airm_bag_canon_d10": _frame(bag_records[AIRM_METRIC], BAG_CANON_TABLE_COLUMNS),
        "airm_bag_sorted_d10": _frame(sorted_records, BAG_SORTED_TABLE_COLUMNS),
        "le_path_d10": _frame(path_records[LE_METRIC], PATH_TABLE_COLUMNS),
        "le_bag_canon_d10": _frame(bag_records[LE_METRIC], BAG_CANON_TABLE_COLUMNS),
    }
    required_failures = {
        "dataset_contract": int((~dataset_contract.loc[dataset_contract["required"], "passed"]).sum()),
        "covariance_sanity": int((~covariance_sanity.loc[covariance_sanity["required"], "passed"]).sum()),
        "trajectory_geometry_correctness": int((~geometry_correctness.loc[geometry_correctness["required"], "passed"]).sum()),
    }
    gate_passed = all(value == 0 for value in required_failures.values())
    numerical_status = str(config["verdicts"]["numerical_failure_status"])
    gate_summary = {
        "protocol_version": str(config["project"]["protocol_version"]),
        "protocol_sha256": str(config["project"]["protocol_sha256"]),
        "config_sha256": str(provenance.get("config_sha256", _canonical_hash(config))),
        "seed": int(config["project"]["seed"]),
        "session": str(config["dataset"]["allowed_session"]),
        "generated_at_utc": timestamp,
        "gate_passed": gate_passed,
        "scientific_classification_allowed": gate_passed,
        "status": "PASS" if gate_passed else numerical_status,
        "required_failure_counts": required_failures,
        "n_trials": n_trials,
        "n_windows": int(n_trials * 5),
        "airm_feature_rows": len(feature_records[AIRM_METRIC]),
        "le_feature_rows": len(feature_records[LE_METRIC]),
        "centering_isometry_trials": n_trials,
        "bag_validation_trials": len(selected_positions),
        "bag_validation_permutations": int(config["geometry"]["bag_validation"]["permutations"]),
        "feature_npz_written": False,
    }
    arrays.update(
        {
            "sample_index": metadata["sample_index"].to_numpy(dtype=np.int64),
            "subject": metadata["subject"].to_numpy(dtype=np.int64),
            "run": metadata["run"].to_numpy(dtype=np.int64),
            "trial_id": metadata["trial_id"].to_numpy(dtype=np.int64),
            "trial_uid": metadata["trial_uid"].astype(str).to_numpy(dtype=str),
            "class_label": metadata["class_label"].astype(str).to_numpy(dtype=str),
            "airm_trial_gate_pass": trial_pass[AIRM_METRIC],
            "le_trial_gate_pass": trial_pass[LE_METRIC],
            "path_d10_names": np.asarray(PATH_D10_NAMES, dtype=str),
            "scalar_11_names": np.asarray(SCALAR_11_NAMES, dtype=str),
            "protocol_version": np.asarray(
                [str(config["project"]["protocol_version"])], dtype=str
            ),
            "protocol_sha256": np.asarray(
                [str(config["project"]["protocol_sha256"])], dtype=str
            ),
            "config_sha256": np.asarray(
                [str(provenance.get("config_sha256", _canonical_hash(config)))],
                dtype=str,
            ),
            "session": np.asarray(
                [str(config["dataset"]["allowed_session"])], dtype=str
            ),
            "seed": np.asarray([int(config["project"]["seed"])], dtype=np.int64),
            "generated_at_utc": np.asarray([timestamp], dtype=str),
            "geometry_gate_passed": np.asarray([gate_passed], dtype=bool),
        }
    )
    artifact_provenance = {
        **provenance,
        "feature_pipeline": "src.feature_pipeline_trajectory_v0",
        "feature_table_names": list(FEATURE_TABLE_NAMES),
        "generated_at_utc": timestamp,
        "scientific_classification_allowed": gate_passed,
    }
    return TrajectoryFeatureArtifacts(tables, arrays, gate_summary, artifact_provenance)


def require_scientific_gate(artifacts: TrajectoryFeatureArtifacts) -> None:
    """Block any scientific consumer unless every required gate passed."""

    if not artifacts.gate_passed:
        raise TrajectoryFeatureGateError(
            str(artifacts.gate_summary.get("status", "trajectory geometry gate failed"))
        )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyriemann", "mne", "moabb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def write_trajectory_feature_artifacts(
    artifacts: TrajectoryFeatureArtifacts,
    config: Mapping[str, Any],
    root: str | Path,
    *,
    config_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write tables/provenance and gate a local downstream NPZ.

    On a failed gate, any stale local feature NPZ at the exact v0 cache path is
    removed so it cannot be mistaken for current, eligible downstream input.
    """

    _validate_config(config)
    project_root = Path(root).expanduser().resolve()
    if str(config["project"]["output_dir"]) != "outputs/bnci2014_001_trajectory_v0":
        raise TrajectoryFeaturePipelineError("output_dir left the frozen v0 namespace")
    if str(config["project"]["local_cache_dir"]) != "cache/bnci2014_001_trajectory_v0":
        raise TrajectoryFeaturePipelineError("local_cache_dir left the frozen v0 namespace")
    output_root = project_root / str(config["project"]["output_dir"])
    tables_dir = output_root / "tables"
    protocol_dir = output_root / "protocol"
    cache_dir = project_root / str(config["project"]["local_cache_dir"])
    for directory in (tables_dir, protocol_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / "trajectory_features_v0.npz"
    # Invalidate any prior eligibility before beginning a new write.  A failed
    # or interrupted run must never leave a stale downstream input in place.
    if feature_path.exists():
        feature_path.unlink()
    written: dict[str, str] = {}
    for name in FEATURE_TABLE_NAMES:
        if name not in artifacts.tables:
            raise TrajectoryFeaturePipelineError(f"missing required feature table {name}")
        path = tables_dir / f"{name}.csv"
        _atomic_csv(path, artifacts.tables[name])
        written[f"table_{name}"] = str(path)

    source_protocol = (
        Path(protocol_path).expanduser().resolve()
        if protocol_path is not None
        else (project_root / str(config["project"]["protocol_path"])).resolve()
    )
    if _file_sha256(source_protocol) != str(config["project"]["protocol_sha256"]):
        raise TrajectoryFeaturePipelineError("protocol copy source hash mismatch")
    protocol_copy = protocol_dir / "PROTOCOL_TRAJECTORY_ANATOMY_V0.md"
    shutil.copyfile(source_protocol, protocol_copy)
    written["protocol_copy"] = str(protocol_copy)
    config_copy = protocol_dir / "frozen_config.yaml"
    if config_path is not None:
        shutil.copyfile(Path(config_path).expanduser().resolve(), config_copy)
    else:
        _atomic_text(config_copy, yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True))
    written["config_copy"] = str(config_copy)
    environment_path = protocol_dir / "environment.json"
    _atomic_text(environment_path, json.dumps(_environment(), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    written["environment"] = str(environment_path)

    gate_summary = dict(artifacts.gate_summary)
    if artifacts.gate_passed:
        _atomic_npz(feature_path, artifacts.arrays)
        gate_summary["feature_npz_written"] = True
        gate_summary["feature_npz_path"] = str(feature_path)
        gate_summary["feature_npz_sha256"] = _file_sha256(feature_path)
        written["feature_npz"] = str(feature_path)
    else:
        gate_summary["feature_npz_written"] = False
        gate_summary["feature_npz_path"] = str(feature_path)
        gate_summary["feature_npz_sha256"] = None
    gate_path = tables_dir / "trajectory_geometry_gate.json"
    _atomic_text(gate_path, json.dumps(gate_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    written["gate_summary"] = str(gate_path)

    hashes = {key: _file_sha256(Path(path)) for key, path in written.items()}
    provenance_payload = {
        **dict(artifacts.provenance),
        "gate_summary": gate_summary,
        "written_paths": written,
        "written_sha256": hashes,
    }
    provenance_path = protocol_dir / "provenance.json"
    _atomic_text(provenance_path, json.dumps(provenance_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    written["provenance"] = str(provenance_path)
    return {"paths": written, "gate_summary": gate_summary}
