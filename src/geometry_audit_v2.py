"""Official subject/scope-level geometry correctness gate for V2.

This module computes geometry checks only.  It never receives class labels as
arguments to a center fit, never fits a classifier, and never writes V1 data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.alignment_v2 import make_calibration_splits
from src.data_v2 import V2WholeData
from src.geometry_v2 import (
    AIRM,
    EA,
    LE,
    RAW,
    FittedCenter,
    airm_distance,
    airm_mean,
    arithmetic_mean,
    fit_center,
    logeuclidean_distance,
    logeuclidean_mean_official,
    select_deterministic_pairs,
    smat,
    spd_diagnostics,
    spd_invsqrt,
    spd_log,
    symmetric_exp,
    tlcenter_constant_domain_crosscheck,
)
from src.spd_utils import svec


GATE_COLUMNS = [
    "subject",
    "protocol",
    "split",
    "fit_scope",
    "geometry",
    "check",
    "statistic",
    "value",
    "threshold",
    "comparator",
    "required",
    "passed",
    "status",
]


@dataclass(frozen=True)
class GeometryGateResult:
    correctness: pd.DataFrame
    mean_comparison: pd.DataFrame
    gate: dict[str, Any]


@dataclass(frozen=True)
class _Scope:
    subject: int
    protocol: str
    split: str
    fit_scope: str
    fit_positions: np.ndarray
    evaluation_positions: np.ndarray


class _Rows:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        scope: _Scope,
        geometry: str,
        check: str,
        statistic: str,
        value: float,
        threshold: float,
        comparator: str,
        *,
        required: bool = True,
        status_if_passed: str = "PASS",
    ) -> None:
        numeric_value = float(value)
        numeric_threshold = float(threshold)
        if comparator == "<=":
            passed = bool(
                np.isfinite(numeric_value) and numeric_value <= numeric_threshold
            )
        elif comparator == ">":
            passed = bool(
                np.isfinite(numeric_value) and numeric_value > numeric_threshold
            )
        elif comparator == "==":
            passed = bool(
                np.isfinite(numeric_value) and numeric_value == numeric_threshold
            )
        else:
            raise ValueError(f"unsupported comparator: {comparator}")
        self.records.append(
            {
                "subject": scope.subject,
                "protocol": scope.protocol,
                "split": scope.split,
                "fit_scope": scope.fit_scope,
                "geometry": geometry,
                "check": check,
                "statistic": statistic,
                "value": numeric_value,
                "threshold": numeric_threshold,
                "comparator": comparator,
                "required": bool(required),
                "passed": passed,
                "status": status_if_passed if passed else "FAIL",
            }
        )

    def skip(
        self,
        scope: _Scope,
        geometry: str,
        check: str,
        statistic: str,
        status: str,
    ) -> None:
        self.records.append(
            {
                "subject": scope.subject,
                "protocol": scope.protocol,
                "split": scope.split,
                "fit_scope": scope.fit_scope,
                "geometry": geometry,
                "check": check,
                "statistic": statistic,
                "value": np.nan,
                "threshold": np.nan,
                "comparator": "not_applicable",
                "required": False,
                "passed": True,
                "status": status,
            }
        )

    def technical_failure(self, scope: _Scope, error: BaseException) -> None:
        self.records.append(
            {
                "subject": scope.subject,
                "protocol": scope.protocol,
                "split": scope.split,
                "fit_scope": scope.fit_scope,
                "geometry": "ALL",
                "check": "technical_failure",
                "statistic": f"{type(error).__name__}: {error}",
                "value": np.nan,
                "threshold": np.nan,
                "comparator": "must_not_occur",
                "required": True,
                "passed": False,
                "status": "FAILED_TECHNICAL",
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.records, columns=GATE_COLUMNS)


def _hard_gate(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["geometry"]["hard_gate"]


def _matrix_integrity_rows(
    rows: _Rows,
    scope: _Scope,
    geometry: str,
    role: str,
    matrices: np.ndarray,
    thresholds: Mapping[str, Any],
) -> None:
    diagnostics = spd_diagnostics(matrices)
    nonfinite = int((~diagnostics["finite"]).sum())
    non_spd = int((~diagnostics["is_spd"]).sum())
    rows.add(
        scope,
        geometry,
        "finite",
        f"{role}_nonfinite_count",
        nonfinite,
        thresholds["finite_nonfinite_count_max"],
        "<=",
    )
    rows.add(
        scope,
        geometry,
        "symmetry",
        f"{role}_maximum_relative_symmetry_error",
        float(np.max(diagnostics["symmetry_error"])),
        thresholds["symmetry_relative_error_max"],
        "<=",
    )
    rows.add(
        scope,
        geometry,
        "positive_definiteness",
        f"{role}_non_spd_count",
        non_spd,
        0.0,
        "<=",
    )
    rows.add(
        scope,
        geometry,
        "positive_definiteness",
        f"{role}_minimum_eigenvalue",
        float(np.nanmin(diagnostics["min_eigenvalue"])),
        thresholds["minimum_eigenvalue_strictly_greater_than"],
        ">",
    )
    rows.add(
        scope,
        geometry,
        "conditioning",
        f"{role}_maximum_condition_number",
        float(np.max(diagnostics["condition_number"])),
        thresholds["condition_number_max"],
        "<=",
    )
    rows.add(
        scope,
        geometry,
        "evd_reconstruction",
        f"{role}_maximum_relative_frobenius_error",
        float(np.max(diagnostics["evd_reconstruction_error"])),
        thresholds["evd_reconstruction_relative_error_max"],
        "<=",
    )
    try:
        reconstructed = symmetric_exp(spd_log(matrices))
        relative = np.linalg.norm(reconstructed - matrices, axis=(-2, -1)) / np.maximum(
            np.linalg.norm(matrices, axis=(-2, -1)), np.finfo(np.float64).tiny
        )
        roundtrip = float(np.max(relative))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        roundtrip = float("inf")
    rows.add(
        scope,
        geometry,
        "log_exp_roundtrip",
        f"{role}_maximum_relative_frobenius_error",
        roundtrip,
        thresholds["log_exp_roundtrip_relative_error_max"],
        "<=",
    )


def _inverse_root_row(
    rows: _Rows,
    scope: _Scope,
    geometry: str,
    mean_matrix: np.ndarray,
    thresholds: Mapping[str, Any],
) -> None:
    inverse_root = spd_invsqrt(mean_matrix)
    whitened = inverse_root @ mean_matrix @ inverse_root
    identity = np.eye(mean_matrix.shape[-1])
    error = float(np.linalg.norm(whitened - identity, ord="fro") / np.linalg.norm(identity))
    rows.add(
        scope,
        geometry,
        "inverse_root_whitening",
        "fitted_mean_relative_identity_error",
        error,
        thresholds["inverse_root_whitening_relative_error_max"],
        "<=",
    )


def _pair_errors(
    before: np.ndarray,
    after: np.ndarray,
    uids: Sequence[str],
    *,
    subject: int,
    seed: int,
    n_pairs: int,
    distance_function: Any,
) -> tuple[float, float]:
    selection = select_deterministic_pairs(
        np.asarray(uids), seed=seed, subject=subject, n_pairs=n_pairs
    )
    absolute: list[float] = []
    relative: list[float] = []
    for left, right in selection.indices:
        distance_before = float(distance_function(before[left], before[right]))
        distance_after = float(distance_function(after[left], after[right]))
        error = abs(distance_before - distance_after)
        absolute.append(error)
        relative.append(error / max(abs(distance_before), np.finfo(np.float64).tiny))
    return float(max(absolute)), float(max(relative))


def _isometry_rows(
    rows: _Rows,
    scope: _Scope,
    geometry: str,
    role: str,
    before: np.ndarray,
    after: np.ndarray,
    uids: Sequence[str],
    config: Mapping[str, Any],
) -> None:
    hard = _hard_gate(config)
    seed = int(config["protocol"]["seed"])
    n_pairs = int(config["geometry"]["isometry_pairs"]["pairs_per_subject"])
    if geometry == LE:
        distance_function = logeuclidean_distance
        absolute_threshold = hard["g1_isometry_absolute_error_max"]
        relative_threshold = hard["g1_isometry_relative_error_max"]
    elif geometry == AIRM:
        distance_function = airm_distance
        absolute_threshold = hard["g2_isometry_absolute_error_max"]
        relative_threshold = hard["g2_isometry_relative_error_max"]
    else:
        raise ValueError("isometry gate is defined only for LE and AIRM")
    absolute, relative = _pair_errors(
        before,
        after,
        uids,
        subject=scope.subject,
        seed=seed,
        n_pairs=n_pairs,
        distance_function=distance_function,
    )
    rows.add(
        scope,
        geometry,
        "isometry",
        f"{role}_maximum_absolute_pair_distance_error",
        absolute,
        absolute_threshold,
        "<=",
    )
    rows.add(
        scope,
        geometry,
        "isometry",
        f"{role}_maximum_relative_pair_distance_error",
        relative,
        relative_threshold,
        "<=",
    )


def _v1_le_equivalence_rows(
    rows: _Rows,
    scope: _Scope,
    raw: np.ndarray,
    transformed: np.ndarray,
    config: Mapping[str, Any],
) -> None:
    hard = _hard_gate(config)
    raw_coordinates = svec(spd_log(raw))
    v1_coordinates = raw_coordinates - raw_coordinates.mean(axis=0)
    matrix_coordinates = svec(spd_log(transformed))
    difference = matrix_coordinates - v1_coordinates
    maximum_absolute = float(np.max(np.abs(difference)))
    relative_l2 = float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(v1_coordinates), np.finfo(np.float64).tiny)
    )
    reconstructed = symmetric_exp(smat(v1_coordinates))
    matrix_relative = np.linalg.norm(reconstructed - transformed, axis=(1, 2)) / np.maximum(
        np.linalg.norm(transformed, axis=(1, 2)), np.finfo(np.float64).tiny
    )
    rows.add(
        scope,
        LE,
        "v1_equivalence",
        "centered_coordinate_maximum_absolute_error",
        maximum_absolute,
        hard["g1_v1_coordinate_max_absolute_error_max"],
        "<=",
    )
    rows.add(
        scope,
        LE,
        "v1_equivalence",
        "centered_coordinate_relative_l2_error",
        relative_l2,
        hard["g1_v1_coordinate_relative_l2_error_max"],
        "<=",
    )
    rows.add(
        scope,
        LE,
        "v1_equivalence",
        "reconstructed_matrix_maximum_relative_frobenius_error",
        float(np.max(matrix_relative)),
        hard["g1_v1_reconstructed_matrix_relative_error_max"],
        "<=",
    )


def _center_eigenvalues(matrix: np.ndarray) -> tuple[float, float, float]:
    diagnostics = spd_diagnostics(matrix)
    return (
        float(diagnostics["min_eigenvalue"][0]),
        float(diagnostics["max_eigenvalue"][0]),
        float(diagnostics["condition_number"][0]),
    )


def _mean_comparison_row(
    scope: _Scope,
    fit_covariances: np.ndarray,
    centers: Mapping[str, FittedCenter],
    transformed: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    le_mean = centers[LE].mean_matrix
    airm_mean_matrix = centers[AIRM].mean_matrix
    ea_mean = centers[EA].mean_matrix
    if le_mean is None or airm_mean_matrix is None or ea_mean is None:
        raise RuntimeError("center comparison is missing a fitted mean")
    d_le_le_airm = float(logeuclidean_distance(le_mean, airm_mean_matrix))
    d_ai_le_airm = float(airm_distance(le_mean, airm_mean_matrix))
    d_le_le_ea = float(logeuclidean_distance(le_mean, ea_mean))
    d_ai_ai_ea = float(airm_distance(airm_mean_matrix, ea_mean))
    le_distances = np.asarray(
        logeuclidean_distance(fit_covariances, le_mean), dtype=np.float64
    )
    airm_distances = np.asarray(
        airm_distance(fit_covariances, airm_mean_matrix), dtype=np.float64
    )
    le_dispersion = float(np.sqrt(np.mean(np.square(le_distances))))
    airm_dispersion = float(np.sqrt(np.mean(np.square(airm_distances))))
    coordinate_difference = svec(spd_log(transformed[LE])) - svec(
        spd_log(transformed[AIRM])
    )
    coordinate_norms = np.linalg.norm(coordinate_difference, axis=1)
    le_min, le_max, le_condition = _center_eigenvalues(le_mean)
    ai_min, ai_max, ai_condition = _center_eigenvalues(airm_mean_matrix)
    ea_min, ea_max, ea_condition = _center_eigenvalues(ea_mean)
    airm_center = centers[AIRM]
    return {
        "subject": scope.subject,
        "protocol": scope.protocol,
        "split": scope.split,
        "fit_scope": scope.fit_scope,
        "fit_n": int(len(fit_covariances)),
        "airm_tol": airm_center.solver_tol,
        "airm_maxiter": airm_center.solver_maxiter,
        "airm_warning_count": int(len(airm_center.solver_warning_messages)),
        "airm_iteration_count": pd.NA,
        "airm_termination_reason": airm_center.termination_reason,
        "airm_normalized_karcher_residual": (
            airm_center.normalized_karcher_post_residual
        ),
        "d_le_le_airm": d_le_le_airm,
        "d_airm_le_airm": d_ai_le_airm,
        "d_le_le_ea": d_le_le_ea,
        "d_airm_airm_ea": d_ai_ai_ea,
        "le_dispersion": le_dispersion,
        "airm_dispersion": airm_dispersion,
        "normalized_d_le_le_airm": d_le_le_airm / le_dispersion,
        "normalized_d_airm_le_airm": d_ai_le_airm / airm_dispersion,
        "normalized_d_le_le_ea": d_le_le_ea / le_dispersion,
        "normalized_d_airm_airm_ea": d_ai_ai_ea / airm_dispersion,
        "le_airm_relative_frobenius": float(
            np.linalg.norm(le_mean - airm_mean_matrix, ord="fro")
            / max(np.linalg.norm(airm_mean_matrix, ord="fro"), np.finfo(float).tiny)
        ),
        "le_center_min_eigenvalue": le_min,
        "le_center_max_eigenvalue": le_max,
        "le_center_condition_number": le_condition,
        "airm_center_min_eigenvalue": ai_min,
        "airm_center_max_eigenvalue": ai_max,
        "airm_center_condition_number": ai_condition,
        "ea_center_min_eigenvalue": ea_min,
        "ea_center_max_eigenvalue": ea_max,
        "ea_center_condition_number": ea_condition,
        "le_airm_coordinate_difference_mean_l2": float(coordinate_norms.mean()),
        "le_airm_coordinate_difference_median_l2": float(np.median(coordinate_norms)),
        "le_airm_coordinate_difference_maximum_l2": float(coordinate_norms.max()),
    }


def _evaluate_scope(
    rows: _Rows,
    scope: _Scope,
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    hard = _hard_gate(config)
    fit_covariances = np.asarray(covariances[scope.fit_positions], dtype=np.float64)
    evaluation_covariances = np.asarray(
        covariances[scope.evaluation_positions], dtype=np.float64
    )
    fit_uids = metadata.iloc[scope.fit_positions]["trial_uid"].astype(str).tolist()
    evaluation_uids = (
        metadata.iloc[scope.evaluation_positions]["trial_uid"].astype(str).tolist()
    )
    airm_config = config["geometry"]["airm_mean"]
    centers = {
        geometry: fit_center(
            fit_covariances,
            geometry,
            tol=float(airm_config["tol"]),
            maxiter=int(airm_config["maxiter"]),
        )
        for geometry in (RAW, LE, AIRM, EA)
    }
    fit_transformed = {
        geometry: center.transform(fit_covariances)
        for geometry, center in centers.items()
    }
    evaluation_transformed = {
        geometry: center.transform(evaluation_covariances)
        for geometry, center in centers.items()
    }

    for geometry in (RAW, LE, AIRM, EA):
        _matrix_integrity_rows(
            rows,
            scope,
            geometry,
            "fit_transformed",
            fit_transformed[geometry],
            hard,
        )
        if scope.protocol == "T2":
            _matrix_integrity_rows(
                rows,
                scope,
                geometry,
                "evaluation_transformed",
                evaluation_transformed[geometry],
                hard,
            )

    for geometry in (LE, AIRM, EA):
        mean_matrix = centers[geometry].mean_matrix
        if mean_matrix is None:
            raise RuntimeError(f"{geometry} fitted center has no mean matrix")
        _matrix_integrity_rows(
            rows, scope, geometry, "fitted_mean", mean_matrix, hard
        )
        _inverse_root_row(rows, scope, geometry, mean_matrix, hard)

    official_le = logeuclidean_mean_official(fit_covariances)
    le_mean = centers[LE].mean_matrix
    if le_mean is None:
        raise RuntimeError("LE center is missing its mean")
    official_error = float(
        np.linalg.norm(le_mean - official_le, ord="fro")
        / max(np.linalg.norm(official_le, ord="fro"), np.finfo(float).tiny)
    )
    rows.add(
        scope,
        LE,
        "custom_official_mean",
        "relative_frobenius_error",
        official_error,
        hard["g1_custom_official_mean_relative_error_max"],
        "<=",
    )
    _v1_le_equivalence_rows(
        rows, scope, fit_covariances, fit_transformed[LE], config
    )
    le_identity_residual = float(
        np.linalg.norm(spd_log(fit_transformed[LE]).mean(axis=0), ord="fro")
        / np.sqrt(fit_covariances.shape[-1])
    )
    rows.add(
        scope,
        LE,
        "fitted_mean_identity",
        "normalized_log_mean_frobenius_norm",
        le_identity_residual,
        hard["g1_fitted_mean_residual_max"],
        "<=",
    )
    _isometry_rows(
        rows,
        scope,
        LE,
        "fit",
        fit_covariances,
        fit_transformed[LE],
        fit_uids,
        config,
    )

    airm_center = centers[AIRM]
    if airm_center.normalized_karcher_post_residual is None:
        raise RuntimeError("AIRM center has no post-hoc Karcher residual")
    rows.add(
        scope,
        AIRM,
        "karcher_residual",
        "normalized_post_fit_residual",
        airm_center.normalized_karcher_post_residual,
        hard["g2_karcher_residual_max"],
        "<=",
    )
    rows.add(
        scope,
        AIRM,
        "solver_warning",
        "captured_warning_count",
        len(airm_center.solver_warning_messages),
        0.0,
        "<=",
        required=False,
    )
    transformed_airm_mean = airm_mean(
        fit_transformed[AIRM],
        tol=float(airm_config["tol"]),
        maxiter=int(airm_config["maxiter"]),
    )
    airm_identity = float(
        airm_distance(
            transformed_airm_mean.matrix, np.eye(fit_covariances.shape[-1])
        )
        / np.sqrt(fit_covariances.shape[-1])
    )
    rows.add(
        scope,
        AIRM,
        "fitted_mean_identity",
        "normalized_airm_distance_to_identity",
        airm_identity,
        hard["g2_fitted_mean_residual_max"],
        "<=",
    )
    _isometry_rows(
        rows,
        scope,
        AIRM,
        "fit",
        fit_covariances,
        fit_transformed[AIRM],
        fit_uids,
        config,
    )
    try:
        crosscheck = tlcenter_constant_domain_crosscheck(
            fit_covariances, airm_center
        )
    except (ImportError, AttributeError, TypeError):
        rows.skip(
            scope,
            AIRM,
            "tlcenter_crosscheck",
            "maximum_normalized_airm_distance",
            str(hard["tlcenter_api_unavailable_status"]),
        )
    else:
        rows.add(
            scope,
            AIRM,
            "tlcenter_crosscheck",
            "maximum_normalized_airm_distance",
            crosscheck["maximum_normalized_airm_distance"],
            hard["tlcenter_crosscheck_distance_max"],
            "<=",
        )

    ea_identity = float(
        np.linalg.norm(
            arithmetic_mean(fit_transformed[EA]) - np.eye(fit_covariances.shape[-1]),
            ord="fro",
        )
        / np.sqrt(fit_covariances.shape[-1])
    )
    rows.add(
        scope,
        EA,
        "fitted_mean_identity",
        "arithmetic_mean_relative_identity_error",
        ea_identity,
        hard["g3_fitted_mean_relative_error_max"],
        "<=",
    )

    if scope.protocol == "T2":
        for geometry in (LE, AIRM):
            _isometry_rows(
                rows,
                scope,
                geometry,
                "evaluation",
                evaluation_covariances,
                evaluation_transformed[geometry],
                evaluation_uids,
                config,
            )
    return _mean_comparison_row(
        scope, fit_covariances, centers, fit_transformed
    )


def _scopes(metadata: pd.DataFrame, config: Mapping[str, Any]) -> list[_Scope]:
    subjects = [int(value) for value in config["dataset"]["subjects"]]
    subject_values = pd.to_numeric(metadata["subject"], errors="raise").astype(int)
    result: list[_Scope] = []
    for subject in subjects:
        all_positions = np.flatnonzero(subject_values.to_numpy() == subject).astype(
            np.int64
        )
        result.append(
            _Scope(
                subject=subject,
                protocol="T1",
                split="ALL",
                fit_scope="all_subject_0train_whole_trials",
                fit_positions=all_positions,
                evaluation_positions=all_positions,
            )
        )
        split_a, split_b = make_calibration_splits(metadata, target_subject=subject)
        for split in (split_a, split_b):
            result.append(
                _Scope(
                    subject=subject,
                    protocol="T2",
                    split=split.name,
                    fit_scope=(
                        "calibration_runs_"
                        + "_".join(str(value) for value in split.calibration_runs)
                    ),
                    fit_positions=np.asarray(
                        split.calibration_row_positions, dtype=np.int64
                    ),
                    evaluation_positions=np.asarray(
                        split.evaluation_row_positions, dtype=np.int64
                    ),
                )
            )
    return result


def _gate_payload(correctness: pd.DataFrame) -> dict[str, Any]:
    required = correctness["required"].astype(bool)
    passed = correctness["passed"].astype(bool)
    required_pass = required & passed & (correctness["status"] == "PASS")
    failed = correctness[required & ~required_pass]
    classification_gate_pass = bool(required.any() and failed.empty)
    return {
        "classification_gate_pass": classification_gate_pass,
        "n_rows": int(len(correctness)),
        "n_required_rows": int(required.sum()),
        "n_required_passed": int(required_pass.sum()),
        "n_required_failed": int(len(failed)),
        "required_rows": int(required.sum()),
        "passed_required_rows": int(required_pass.sum()),
        "failed_required_rows": int(len(failed)),
        "failed_subjects": sorted(
            int(value) for value in failed["subject"].dropna().unique()
        ),
        "failure_statuses": sorted(str(value) for value in failed["status"].unique()),
        "classification_authorized": classification_gate_pass,
    }


def run_geometry_gate(
    data: V2WholeData,
    config: Mapping[str, Any],
) -> GeometryGateResult:
    """Run all T1/T2 geometry checks, continuing across failed scopes."""

    covariances = np.asarray(data.covariances, dtype=np.float64)
    metadata = data.metadata.reset_index(drop=True).copy()
    if len(covariances) != len(metadata):
        raise ValueError("covariance and metadata row counts differ")
    rows = _Rows()
    mean_rows: list[dict[str, Any]] = []
    for scope in _scopes(metadata, config):
        try:
            mean_rows.append(
                _evaluate_scope(rows, scope, covariances, metadata, config)
            )
        except Exception as error:  # preserve a machine-readable failed gate
            rows.technical_failure(scope, error)
    correctness = rows.frame()
    mean_comparison = pd.DataFrame.from_records(mean_rows)
    return GeometryGateResult(
        correctness=correctness,
        mean_comparison=mean_comparison,
        gate=_gate_payload(correctness),
    )


def technical_failure_result(error: BaseException) -> GeometryGateResult:
    """Construct a writable failed gate when setup/data loading itself fails."""

    scope = _Scope(
        subject=-1,
        protocol="SETUP",
        split="NOT_APPLICABLE",
        fit_scope="setup",
        fit_positions=np.empty(0, dtype=np.int64),
        evaluation_positions=np.empty(0, dtype=np.int64),
    )
    rows = _Rows()
    rows.technical_failure(scope, error)
    correctness = rows.frame()
    return GeometryGateResult(
        correctness=correctness,
        mean_comparison=pd.DataFrame(),
        gate=_gate_payload(correctness),
    )
