"""Deterministic reporting for the preregistered WHOLE-SPD geometry audit V2.

This module consumes only completed, gate-authorized V2 result tables.  It
does not fit a geometry, classifier, or diagnostic and it does not inspect
target labels.  The frozen Q1--Q3 rules are evaluated directly from subject
level primary-logistic balanced accuracies.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_TITLE = "# BNCI2014_001 WHOLE-SPD Geometry Audit V2"
REPORT_HEADINGS = (
    "Motivation",
    "Frozen protocol",
    "Geometry definitions",
    "Geometry correctness",
    "LE vs AIRM Fréchet means",
    "V1 leakage audit",
    "LOSO transductive results",
    "Calibration-to-held-out-run results",
    "Metric-native MDM sanity check",
    "Marginal domain diagnostics",
    "Frozen decision-rule verdicts",
    "What is actually justified",
    "What is NOT justified",
    "Single recommended next experiment",
)
PREREQUISITE_FILENAMES = (
    "geometry_correctness.csv",
    "geometry_mean_comparison.csv",
    "loso_logistic_transductive.csv",
    "loso_logistic_calibration.csv",
    "loso_mdm_transductive.csv",
    "loso_mdm_calibration.csv",
    "v1_leakage_audit.csv",
    "domain_shift_diagnostics.csv",
)
FIGURE_STEMS = (
    "figure_1_loso_ba_by_subject",
    "figure_2_paired_delta_vs_raw",
    "figure_3_t1_vs_t2_ba",
    "figure_4_le_vs_airm_centers",
    "figure_5_v1_leakage_audit",
)
GEOMETRIES = ("RAW", "LE", "AIRM", "EA")
PRIMARY_METRICS = ("balanced_accuracy", "accuracy", "macro_f1")
CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
MDM_SPECS = (
    ("RAW", "riemann"),
    ("RAW", "logeuclid"),
    ("LE", "logeuclid"),
    ("AIRM", "riemann"),
    ("EA", "riemann"),
)
CONFUSION_COLUMNS = tuple(
    f"confusion_{truth}__{prediction}"
    for truth in CLASS_ORDER
    for prediction in CLASS_ORDER
)
RECALL_COLUMNS = tuple(f"recall_{label}" for label in CLASS_ORDER)


class ReportingContractError(RuntimeError):
    """Raised when a prerequisite or generated reporting artifact is invalid."""


@dataclass(frozen=True)
class FrozenVerdicts:
    q1: str
    q2: str
    q3: str
    operands: Mapping[str, Mapping[str, Any]]
    t1_ba: pd.DataFrame
    t2_ba: pd.DataFrame
    subject_deltas: pd.DataFrame
    next_experiment: str


@dataclass(frozen=True)
class ReportingArtifacts:
    summary: pd.DataFrame
    verdicts: FrozenVerdicts
    figure_sources: Mapping[str, pd.DataFrame]
    report_text: str


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
    *,
    allow_empty: bool = False,
) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ReportingContractError(f"{name} is missing columns: {sorted(missing)}")
    if frame.empty and not allow_empty:
        raise ReportingContractError(f"{name} is empty")


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    def convert(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value in {"True", "False"}:
            return value == "True"
        raise ReportingContractError(f"{name} must contain explicit True/False")

    return series.map(convert)


def _numeric(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ReportingContractError(f"{name}.{column} contains non-finite values")


def _metric_contract(frame: pd.DataFrame, name: str) -> None:
    metric_columns = (*PRIMARY_METRICS, *RECALL_COLUMNS)
    _numeric(frame, metric_columns, name)
    for column in metric_columns:
        values = frame[column].to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ReportingContractError(f"{name}.{column} is outside [0,1]")
    _numeric(frame, CONFUSION_COLUMNS, name)
    confusion = frame[list(CONFUSION_COLUMNS)].to_numpy(dtype=float)
    if np.any(confusion < 0.0) or not np.equal(confusion, np.floor(confusion)).all():
        raise ReportingContractError(f"{name} confusion cells must be non-negative integers")


def _normalize_subject(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    subject_column = "target_subject" if "target_subject" in result else "subject"
    result[subject_column] = pd.to_numeric(
        result[subject_column], errors="coerce"
    ).astype("Int64")
    if result[subject_column].isna().any():
        raise ReportingContractError(f"{name} has invalid subject identifiers")
    result[subject_column] = result[subject_column].astype(int)
    if "subject" in result and subject_column == "target_subject":
        other = pd.to_numeric(result["subject"], errors="coerce")
        if other.isna().any() or not np.array_equal(
            other.to_numpy(dtype=int), result[subject_column].to_numpy(dtype=int)
        ):
            raise ReportingContractError(f"{name} subject/target_subject mismatch")
    result["target_subject"] = result[subject_column].astype(int)
    return result


def _assert_provenance(
    frame: pd.DataFrame,
    name: str,
    *,
    protocol_version: str,
    config_sha256: str,
    seed: int,
) -> None:
    expected = {
        "protocol_version": str(protocol_version),
        "config_sha256": str(config_sha256),
        "seed": int(seed),
    }
    for column, value in expected.items():
        if column not in frame:
            raise ReportingContractError(f"{name} lacks provenance column {column}")
        observed = set(frame[column].astype(str))
        if observed != {str(value)}:
            raise ReportingContractError(
                f"{name}.{column} mismatch: expected {value!r}, observed {sorted(observed)}"
            )


def _assert_exact_keys(
    frame: pd.DataFrame,
    key_columns: Sequence[str],
    expected_keys: set[tuple[Any, ...]],
    name: str,
) -> None:
    if frame.duplicated(list(key_columns)).any():
        raise ReportingContractError(f"{name} contains duplicate logical result rows")
    observed = set(frame[list(key_columns)].itertuples(index=False, name=None))
    if observed != expected_keys:
        missing = sorted(expected_keys - observed, key=repr)[:8]
        extra = sorted(observed - expected_keys, key=repr)[:8]
        raise ReportingContractError(
            f"{name} logical grid mismatch; missing={missing}, extra={extra}"
        )


def _validate_result_table(
    frame: pd.DataFrame,
    name: str,
    *,
    subjects: tuple[int, ...],
    decoder: str,
    protocol: str,
    splits: tuple[str, ...],
    specs: tuple[tuple[str, str], ...],
    protocol_version: str,
    config_sha256: str,
    seed: int,
) -> pd.DataFrame:
    required = (
        "protocol_version",
        "config_sha256",
        "seed",
        "subject",
        "target_subject",
        "geometry",
        "protocol",
        "split",
        "decoder",
        "native_metric",
        "source_n",
        "evaluation_n",
        "transductive_overlap",
        "status",
        "convergence_warning",
        "warning_messages",
        *PRIMARY_METRICS,
        *RECALL_COLUMNS,
        "confusion_matrix_json",
        *CONFUSION_COLUMNS,
    )
    _require_columns(frame, required, name)
    result = _normalize_subject(frame, name)
    _assert_provenance(
        result,
        name,
        protocol_version=protocol_version,
        config_sha256=config_sha256,
        seed=seed,
    )
    if set(result["decoder"].astype(str)) != {decoder}:
        raise ReportingContractError(f"{name} decoder must be exactly {decoder}")
    if set(result["protocol"].astype(str)) != {protocol}:
        raise ReportingContractError(f"{name} protocol must be exactly {protocol}")
    if set(result["split"].astype(str)) != set(splits):
        raise ReportingContractError(f"{name} splits must be exactly {splits}")
    if not result["status"].astype(str).eq("PASS").all():
        raise ReportingContractError(f"{name} contains a non-PASS row")
    if _strict_bool(result["convergence_warning"], name=f"{name}.convergence_warning").any():
        raise ReportingContractError(f"{name} contains a convergence warning")
    _metric_contract(result, name)
    expected = {
        (subject, geometry, metric, split)
        for subject in subjects
        for geometry, metric in specs
        for split in splits
    }
    _assert_exact_keys(
        result,
        ("target_subject", "geometry", "native_metric", "split"),
        expected,
        name,
    )
    return result.sort_values(
        ["target_subject", "geometry", "native_metric", "split"],
        kind="mergesort",
    ).reset_index(drop=True)


def _validate_secondary_mdm_table(
    frame: pd.DataFrame,
    name: str,
    *,
    subjects: tuple[int, ...],
    protocol: str,
    splits: tuple[str, ...],
    protocol_version: str,
    config_sha256: str,
    seed: int,
) -> pd.DataFrame:
    """Validate any recorded MDM prefix without making it a primary hard gate."""

    required = (
        "protocol_version",
        "config_sha256",
        "seed",
        "subject",
        "target_subject",
        "geometry",
        "protocol",
        "split",
        "decoder",
        "native_metric",
        "source_n",
        "evaluation_n",
        "transductive_overlap",
        "status",
        "convergence_warning",
        "warning_messages",
        *PRIMARY_METRICS,
        *RECALL_COLUMNS,
        "confusion_matrix_json",
        *CONFUSION_COLUMNS,
    )
    _require_columns(frame, required, name, allow_empty=True)
    result = _normalize_subject(frame, name)
    if result.empty:
        return result
    _assert_provenance(
        result,
        name,
        protocol_version=protocol_version,
        config_sha256=config_sha256,
        seed=seed,
    )
    if set(result["decoder"].astype(str)) != {"MDM"}:
        raise ReportingContractError(f"{name} decoder must be MDM")
    if set(result["protocol"].astype(str)) != {protocol}:
        raise ReportingContractError(f"{name} protocol must be exactly {protocol}")
    if not set(result["split"].astype(str)).issubset(set(splits)):
        raise ReportingContractError(f"{name} contains an unexpected split")
    statuses = set(result["status"].astype(str))
    if not statuses.issubset({"PASS", "FAILED"}):
        raise ReportingContractError(f"{name} has invalid secondary statuses: {statuses}")
    expected = {
        (subject, geometry, metric, split)
        for subject in subjects
        for geometry, metric in MDM_SPECS
        for split in splits
    }
    if result.duplicated(
        ["target_subject", "geometry", "native_metric", "split"]
    ).any():
        raise ReportingContractError(f"{name} contains duplicate logical result rows")
    observed = set(
        result[
            ["target_subject", "geometry", "native_metric", "split"]
        ].itertuples(index=False, name=None)
    )
    extra = observed - expected
    if extra:
        raise ReportingContractError(
            f"{name} contains non-protocol secondary rows: {sorted(extra, key=repr)[:8]}"
        )
    passed = result[result["status"].astype(str) == "PASS"]
    if not passed.empty:
        if _strict_bool(
            passed["convergence_warning"], name=f"{name}.convergence_warning"
        ).any():
            raise ReportingContractError(f"{name} has a PASS row with a convergence warning")
        _metric_contract(passed, name)
    failed = result[result["status"].astype(str) == "FAILED"]
    if not failed.empty:
        # FAILED metrics may be absent; if present they still must be bounded.
        for column in (*PRIMARY_METRICS, *RECALL_COLUMNS, *CONFUSION_COLUMNS):
            numeric = pd.to_numeric(failed[column], errors="coerce")
            present = numeric.notna()
            if present.any() and not np.isfinite(numeric[present].to_numpy(dtype=float)).all():
                raise ReportingContractError(f"{name}.{column} has invalid failed-row data")
    return result.sort_values(
        ["target_subject", "geometry", "native_metric", "split"],
        kind="mergesort",
    ).reset_index(drop=True)


def _assert_t2_ba_aggregate(
    frame: pd.DataFrame, *, tolerance: float, name: str
) -> None:
    """Verify the producer's subject-then-split primary BA aggregation rule."""

    for keys, group in frame.groupby(
        ["target_subject", "geometry", "native_metric"], sort=False
    ):
        split_rows = group[group["split"].isin(["A", "B"])]
        aggregate = group[group["split"] == "AGGREGATE"]
        if len(split_rows) != 2 or len(aggregate) != 1:
            raise ReportingContractError(f"{name} has incomplete T2 rows for {keys}")
        expected = float(split_rows["balanced_accuracy"].mean())
        observed = float(aggregate.iloc[0]["balanced_accuracy"])
        if abs(expected - observed) > tolerance:
            raise ReportingContractError(
                f"{name} aggregate BA mismatch for {keys}: {observed} vs {expected}"
            )


def validate_reporting_inputs(
    tables: Mapping[str, pd.DataFrame],
    gate: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_sha256: str,
) -> dict[str, pd.DataFrame]:
    """Validate the complete eight-table reporting boundary and all-pass gate."""

    missing = set(PREREQUISITE_FILENAMES) - set(tables)
    if missing:
        raise ReportingContractError(f"missing prerequisite tables: {sorted(missing)}")
    if gate.get("classification_gate_pass") is not True:
        raise ReportingContractError("classification_gate_pass is not exactly true")
    protocol_version = str(config["protocol"]["version"])
    protocol_sha256 = str(config["protocol"]["protocol_sha256"])
    if gate.get("protocol_version") != protocol_version:
        raise ReportingContractError("geometry gate protocol_version mismatch")
    if gate.get("protocol_sha256") != protocol_sha256:
        raise ReportingContractError("geometry gate protocol_sha256 mismatch")
    if gate.get("config_sha256") != config_sha256:
        raise ReportingContractError("geometry gate config_sha256 mismatch")

    correctness = tables["geometry_correctness.csv"].copy()
    _require_columns(correctness, ("required", "passed", "status"), "geometry_correctness")
    required = _strict_bool(correctness["required"], name="geometry_correctness.required")
    passed = _strict_bool(correctness["passed"], name="geometry_correctness.passed")
    required_pass = required & passed & correctness["status"].astype(str).eq("PASS")
    if not required.any() or not required_pass.loc[required].all():
        raise ReportingContractError("not every required geometry row passed")
    counts = {
        "required_rows": int(required.sum()),
        "passed_required_rows": int(required_pass.sum()),
        "failed_required_rows": int((required & ~required_pass).sum()),
    }
    for key, expected in counts.items():
        if key not in gate or gate[key] != expected:
            raise ReportingContractError(
                f"geometry gate count mismatch for {key}: {gate.get(key)!r} != {expected}"
            )

    subjects = tuple(int(value) for value in config["dataset"]["subjects"])
    if subjects != tuple(range(1, 10)):
        raise ReportingContractError("reporting requires frozen subjects 1..9")
    seed = int(config["protocol"]["seed"])
    logistic_specs = tuple((geometry, "euclidean_log_svec") for geometry in GEOMETRIES)
    normalized: dict[str, pd.DataFrame] = {
        "geometry_correctness.csv": correctness,
    }
    normalized["loso_logistic_transductive.csv"] = _validate_result_table(
        tables["loso_logistic_transductive.csv"],
        "loso_logistic_transductive",
        subjects=subjects,
        decoder="logistic",
        protocol="T1",
        splits=("ALL",),
        specs=logistic_specs,
        protocol_version=protocol_version,
        config_sha256=config_sha256,
        seed=seed,
    )
    normalized["loso_logistic_calibration.csv"] = _validate_result_table(
        tables["loso_logistic_calibration.csv"],
        "loso_logistic_calibration",
        subjects=subjects,
        decoder="logistic",
        protocol="T2",
        splits=("A", "B", "AGGREGATE"),
        specs=logistic_specs,
        protocol_version=protocol_version,
        config_sha256=config_sha256,
        seed=seed,
    )
    _assert_t2_ba_aggregate(
        normalized["loso_logistic_calibration.csv"],
        tolerance=float(config["evaluation"]["T2"]["equivalent_flat_mean_tolerance"]),
        name="loso_logistic_calibration",
    )
    normalized["loso_mdm_transductive.csv"] = _validate_secondary_mdm_table(
        tables["loso_mdm_transductive.csv"],
        "loso_mdm_transductive",
        subjects=subjects,
        protocol="T1",
        splits=("ALL",),
        protocol_version=protocol_version,
        config_sha256=config_sha256,
        seed=seed,
    )
    normalized["loso_mdm_calibration.csv"] = _validate_secondary_mdm_table(
        tables["loso_mdm_calibration.csv"],
        "loso_mdm_calibration",
        subjects=subjects,
        protocol="T2",
        splits=("A", "B", "AGGREGATE"),
        protocol_version=protocol_version,
        config_sha256=config_sha256,
        seed=seed,
    )
    mdm_t2 = normalized["loso_mdm_calibration.csv"]
    complete_mdm_groups: list[pd.DataFrame] = []
    for _, group in mdm_t2.groupby(
        ["target_subject", "geometry", "native_metric"], sort=False
    ):
        if set(group["split"].astype(str)) == {"A", "B", "AGGREGATE"} and group[
            "status"
        ].astype(str).eq("PASS").all():
            complete_mdm_groups.append(group)
    if complete_mdm_groups:
        _assert_t2_ba_aggregate(
            pd.concat(complete_mdm_groups, ignore_index=True),
            tolerance=float(config["evaluation"]["T2"]["equivalent_flat_mean_tolerance"]),
            name="loso_mdm_calibration",
        )

    means = _normalize_subject(tables["geometry_mean_comparison.csv"], "geometry_mean_comparison")
    _require_columns(
        means,
        (
            "subject",
            "protocol",
            "split",
            "fit_scope",
            "fit_n",
            "normalized_d_le_le_airm",
            "normalized_d_airm_le_airm",
            "le_airm_coordinate_difference_mean_l2",
            "airm_normalized_karcher_residual",
        ),
        "geometry_mean_comparison",
    )
    _numeric(
        means,
        (
            "fit_n",
            "normalized_d_le_le_airm",
            "normalized_d_airm_le_airm",
            "le_airm_coordinate_difference_mean_l2",
            "airm_normalized_karcher_residual",
        ),
        "geometry_mean_comparison",
    )
    mean_expected = {
        *((subject, "T1", "ALL") for subject in subjects),
        *((subject, "T2", split) for subject in subjects for split in ("A", "B")),
    }
    _assert_exact_keys(
        means,
        ("target_subject", "protocol", "split"),
        mean_expected,
        "geometry_mean_comparison",
    )
    normalized["geometry_mean_comparison.csv"] = means

    leakage = tables["v1_leakage_audit.csv"].copy()
    _require_columns(
        leakage,
        (
            "protocol_version",
            "config_sha256",
            "seed",
            "condition",
            "row_type",
            "fold",
            "classifier_status",
            "convergence_warning",
            "center_fit_n",
            "evaluation_n",
            "center_evaluation_overlap_n",
            "original_v1_benchmark_accuracy",
            "actual_accuracy_difference_from_benchmark",
            *PRIMARY_METRICS,
            *RECALL_COLUMNS,
            "confusion_matrix_json",
            *CONFUSION_COLUMNS,
        ),
        "v1_leakage_audit",
    )
    _assert_provenance(
        leakage,
        "v1_leakage_audit",
        protocol_version=protocol_version,
        config_sha256=config_sha256,
        seed=seed,
    )
    if not leakage["classifier_status"].astype(str).eq("PASS").all():
        raise ReportingContractError("v1_leakage_audit contains a failed classifier")
    if _strict_bool(leakage["convergence_warning"], name="v1 convergence").any():
        raise ReportingContractError("v1_leakage_audit contains a convergence warning")
    _metric_contract(leakage, "v1_leakage_audit")
    expected_leakage = {
        *((condition, "fold", fold) for condition in ("v1_all_sample", "fold_safe") for fold in range(5)),
        *((condition, "pooled_oof", None) for condition in ("v1_all_sample", "fold_safe")),
    }
    observed_leakage: set[tuple[Any, ...]] = set()
    for row in leakage.itertuples(index=False):
        fold = getattr(row, "fold")
        normalized_fold = None if pd.isna(fold) else int(fold)
        observed_leakage.add((str(row.condition), str(row.row_type), normalized_fold))
    if observed_leakage != expected_leakage or len(leakage) != 12:
        raise ReportingContractError("v1_leakage_audit must have 10 fold and 2 pooled rows")
    normalized["v1_leakage_audit.csv"] = leakage

    domain = _normalize_subject(tables["domain_shift_diagnostics.csv"], "domain_shift_diagnostics")
    _require_columns(
        domain,
        (
            "protocol_version",
            "config_sha256",
            "seed",
            "subject",
            "target_subject",
            "geometry",
            "protocol",
            "split",
            "reference_metric",
            "source_target_mean_distance",
            "absolute_dispersion_difference",
            "all_subject_subject_silhouette",
            "all_subject_subject_between_within_rms_ratio",
            "uses_class_labels",
            "status",
        ),
        "domain_shift_diagnostics",
    )
    _assert_provenance(
        domain,
        "domain_shift_diagnostics",
        protocol_version=protocol_version,
        config_sha256=config_sha256,
        seed=seed,
    )
    if not domain["status"].astype(str).eq("PASS").all():
        raise ReportingContractError("domain_shift_diagnostics contains a non-PASS row")
    if _strict_bool(domain["uses_class_labels"], name="domain uses_class_labels").any():
        raise ReportingContractError("domain diagnostics unexpectedly used class labels")
    _numeric(
        domain,
        ("source_target_mean_distance", "absolute_dispersion_difference"),
        "domain_shift_diagnostics",
    )
    domain_specs = MDM_SPECS
    expected_domain = {
        (subject, geometry, metric, protocol, split)
        for subject in subjects
        for geometry, metric in domain_specs
        for protocol, splits in (("T1", ("ALL",)), ("T2", ("A", "B", "AGGREGATE")))
        for split in splits
    }
    _assert_exact_keys(
        domain,
        ("target_subject", "geometry", "reference_metric", "protocol", "split"),
        expected_domain,
        "domain_shift_diagnostics",
    )
    normalized["domain_shift_diagnostics.csv"] = domain
    return normalized


def _category(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "improved"
    if value < -tolerance:
        return "worsened"
    return "tied"


def _ba_pivot(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = frame[frame["split"].astype(str) == split]
    pivot = selected.pivot(
        index="target_subject", columns="geometry", values="balanced_accuracy"
    ).reindex(columns=GEOMETRIES)
    pivot.index.name = "subject"
    return pivot.sort_index().reset_index()


def compute_frozen_verdicts(
    t1_primary: pd.DataFrame,
    t2_primary: pd.DataFrame,
    config: Mapping[str, Any],
) -> FrozenVerdicts:
    """Apply Q1--Q3 exactly, without rounding or post-result choices."""

    t1 = _ba_pivot(t1_primary, "ALL")
    t2 = _ba_pivot(t2_primary, "AGGREGATE")
    if list(t1["subject"]) != list(range(1, 10)) or not t1["subject"].equals(t2["subject"]):
        raise ReportingContractError("verdict inputs must contain paired subjects 1..9")
    tolerance = float(config["verdicts"]["sign_tolerance"])
    deltas = pd.DataFrame({"subject": t1["subject"].astype(int)})
    for geometry in ("LE", "AIRM", "EA"):
        delta = t1[geometry].to_numpy(dtype=float) - t1["RAW"].to_numpy(dtype=float)
        deltas[f"delta_{geometry}_vs_RAW"] = delta
        deltas[f"category_{geometry}_vs_RAW"] = [
            _category(float(value), tolerance) for value in delta
        ]

    q1_cfg = config["verdicts"]["Q1"]
    q1_geometry: dict[str, dict[str, Any]] = {}
    for geometry in ("LE", "AIRM"):
        values = deltas[f"delta_{geometry}_vs_RAW"].to_numpy(dtype=float)
        categories = deltas[f"category_{geometry}_vs_RAW"]
        mean_delta = float(values.mean())
        improved = int(categories.eq("improved").sum())
        passed = bool(
            mean_delta >= float(q1_cfg["mean_delta_min"])
            and improved >= int(q1_cfg["improved_subjects_min"])
        )
        q1_geometry[geometry] = {
            "mean_delta": mean_delta,
            "improved_subjects": improved,
            "worsened_subjects": int(categories.eq("worsened").sum()),
            "tied_subjects": int(categories.eq("tied").sum()),
            "passes": passed,
        }
    pass_count = sum(int(value["passes"]) for value in q1_geometry.values())
    if pass_count == 2:
        q1 = str(q1_cfg["both_pass"])
    elif pass_count == 1:
        q1 = str(q1_cfg["exactly_one_pass"])
    else:
        q1 = str(q1_cfg["neither_pass"])

    q2_cfg = config["verdicts"]["Q2"]
    le_mean = float(q1_geometry["LE"]["mean_delta"])
    airm_mean = float(q1_geometry["AIRM"]["mean_delta"])
    mean_difference = abs(le_mean - airm_mean)
    opposite_sign = bool(
        (le_mean > tolerance and airm_mean < -tolerance)
        or (le_mean < -tolerance and airm_mean > tolerance)
    )
    sign_disagreements = int(
        (
            deltas["category_LE_vs_RAW"]
            != deltas["category_AIRM_vs_RAW"]
        ).sum()
    )
    q2_conditions = {
        "mean_delta_difference_pass": bool(
            mean_difference >= float(q2_cfg["mean_delta_difference_min"])
        ),
        "opposite_non_tie_signs": opposite_sign,
        "subject_sign_category_disagreement_pass": bool(
            sign_disagreements >= int(q2_cfg["subject_sign_disagreements_min"])
        ),
    }
    q2_supported = any(q2_conditions.values())
    q2 = str(q2_cfg["any_condition_pass"] if q2_supported else q2_cfg["no_condition_pass"])

    q3_cfg = config["verdicts"]["Q3"]
    q3_differences = {
        geometry: abs(
            float(t1[geometry].mean()) - float(t2[geometry].mean())
        )
        for geometry in ("LE", "AIRM")
    }
    q3_passes = {
        geometry: bool(value >= float(q3_cfg["absolute_mean_ba_difference_min"]))
        for geometry, value in q3_differences.items()
    }
    q3_supported = any(q3_passes.values())
    q3 = str(q3_cfg["any_geometry_pass"] if q3_supported else q3_cfg["no_geometry_pass"])

    operands: dict[str, Mapping[str, Any]] = {
        "Q1": {
            "geometries": q1_geometry,
            "mean_delta_min": float(q1_cfg["mean_delta_min"]),
            "improved_subjects_min": int(q1_cfg["improved_subjects_min"]),
            "passes_count": pass_count,
        },
        "Q2": {
            "le_mean_delta": le_mean,
            "airm_mean_delta": airm_mean,
            "absolute_mean_delta_difference": mean_difference,
            "mean_delta_difference_min": float(q2_cfg["mean_delta_difference_min"]),
            "opposite_non_tie_signs": opposite_sign,
            "subject_sign_category_disagreements": sign_disagreements,
            "subject_sign_disagreements_min": int(q2_cfg["subject_sign_disagreements_min"]),
            **q2_conditions,
        },
        "Q3": {
            "absolute_t1_t2_mean_ba_differences": q3_differences,
            "absolute_mean_ba_difference_min": float(q3_cfg["absolute_mean_ba_difference_min"]),
            "geometry_passes": q3_passes,
        },
    }
    if q1 == "ROBUSTLY SUPPORTED" and q2 == "NOT SUPPORTED" and q3 == "SMALL IN THIS PILOT":
        next_experiment = "subject × class conditional geometry anatomy"
    elif q3 == "POTENTIALLY IMPORTANT":
        next_experiment = "fixed calibration-size/run-stability audit"
    elif q1 == "GEOMETRY-SENSITIVE-MIXED" or q2 == "SUPPORTED":
        next_experiment = "center-geometry residual audit"
    else:
        next_experiment = "held-out calibration marginal-dispersion diagnostic"
    return FrozenVerdicts(
        q1=q1,
        q2=q2,
        q3=q3,
        operands=operands,
        t1_ba=t1,
        t2_ba=t2,
        subject_deltas=deltas,
        next_experiment=next_experiment,
    )


def _stats(values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.isfinite(array).all():
        raise ReportingContractError("aggregate statistics require finite values")
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std_ddof1": float(np.std(array, ddof=1)) if len(array) > 1 else np.nan,
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def secondary_mdm_failure_inventory(
    tables: Mapping[str, pd.DataFrame], subjects: Sequence[int]
) -> pd.DataFrame:
    """List every explicit FAILED or missing secondary MDM logical row."""

    records: list[dict[str, Any]] = []
    specifications = (
        ("loso_mdm_transductive.csv", "T1", ("ALL",)),
        ("loso_mdm_calibration.csv", "T2", ("A", "B", "AGGREGATE")),
    )
    for filename, protocol, splits in specifications:
        frame = tables[filename]
        indexed: dict[tuple[int, str, str, str], pd.Series] = {}
        for _, row in frame.iterrows():
            key = (
                int(row["target_subject"]),
                str(row["geometry"]),
                str(row["native_metric"]),
                str(row["split"]),
            )
            indexed[key] = row
        for subject in subjects:
            for geometry, metric in MDM_SPECS:
                for split in splits:
                    key = (int(subject), geometry, metric, split)
                    row = indexed.get(key)
                    if row is None:
                        records.append(
                            {
                                "source_table": filename,
                                "protocol": protocol,
                                "subject": int(subject),
                                "geometry": geometry,
                                "native_metric": metric,
                                "split": split,
                                "failure_kind": "MISSING_AFTER_SECONDARY_FAILURE",
                                "warning_messages": "logical row was not produced",
                            }
                        )
                    elif str(row["status"]) != "PASS":
                        records.append(
                            {
                                "source_table": filename,
                                "protocol": protocol,
                                "subject": int(subject),
                                "geometry": geometry,
                                "native_metric": metric,
                                "split": split,
                                "failure_kind": "EXPLICIT_FAILED",
                                "warning_messages": str(row.get("warning_messages", "")),
                            }
                        )
    columns = (
        "source_table",
        "protocol",
        "subject",
        "geometry",
        "native_metric",
        "split",
        "failure_kind",
        "warning_messages",
    )
    return pd.DataFrame.from_records(records, columns=columns)


SUMMARY_COLUMNS = (
    "protocol_version",
    "config_sha256",
    "seed",
    "row_type",
    "source_table",
    "question",
    "protocol",
    "decoder",
    "geometry",
    "native_metric",
    "metric",
    "subject",
    "comparator",
    "n",
    "value",
    "mean",
    "std_ddof1",
    "median",
    "minimum",
    "maximum",
    "delta",
    "sign_category",
    "improved_subjects",
    "worsened_subjects",
    "tied_subjects",
    "threshold",
    "pass_flag",
    "verdict",
    "operands_json",
    "formula",
    "notes",
)


def build_geometry_summary(
    tables: Mapping[str, pd.DataFrame],
    verdicts: FrozenVerdicts,
    config: Mapping[str, Any],
    *,
    config_sha256: str,
) -> pd.DataFrame:
    """Build the ninth required logical table in a transparent long format."""

    base = {
        "protocol_version": str(config["protocol"]["version"]),
        "config_sha256": config_sha256,
        "seed": int(config["protocol"]["seed"]),
    }
    rows: list[dict[str, Any]] = []

    aggregate_specs = (
        ("loso_logistic_transductive.csv", "T1", "ALL"),
        ("loso_logistic_calibration.csv", "T2", "AGGREGATE"),
        ("loso_mdm_transductive.csv", "T1", "ALL"),
        ("loso_mdm_calibration.csv", "T2", "AGGREGATE"),
    )
    for filename, protocol, split in aggregate_specs:
        selected = tables[filename][tables[filename]["split"].astype(str) == split]
        if filename.startswith("loso_mdm_"):
            selected = selected[selected["status"].astype(str) == "PASS"]
        group_columns = ["decoder", "geometry", "native_metric"]
        for keys, group in selected.groupby(group_columns, sort=True, dropna=False):
            decoder, geometry, native_metric = keys
            for metric in PRIMARY_METRICS:
                rows.append(
                    {
                        **base,
                        "row_type": "aggregate_metric",
                        "source_table": filename,
                        "protocol": protocol,
                        "decoder": decoder,
                        "geometry": geometry,
                        "native_metric": native_metric,
                        "metric": metric,
                        "notes": (
                            "PASS secondary rows only; failures/missing rows are separate inventory records"
                            if filename.startswith("loso_mdm_")
                            else "nine-subject primary aggregate"
                        ),
                        **_stats(group[metric]),
                    }
                )

    leakage = tables["v1_leakage_audit.csv"]
    for condition in ("v1_all_sample", "fold_safe"):
        folds = leakage[
            (leakage["condition"] == condition) & (leakage["row_type"] == "fold")
        ]
        pooled = leakage[
            (leakage["condition"] == condition)
            & (leakage["row_type"] == "pooled_oof")
        ].iloc[0]
        for metric in ("balanced_accuracy", "accuracy"):
            rows.append(
                {
                    **base,
                    "row_type": "leakage_fold_aggregate",
                    "source_table": "v1_leakage_audit.csv",
                    "decoder": "logistic",
                    "geometry": "LE",
                    "metric": metric,
                    "comparator": condition,
                    **_stats(folds[metric]),
                }
            )
            rows.append(
                {
                    **base,
                    "row_type": "leakage_pooled_oof",
                    "source_table": "v1_leakage_audit.csv",
                    "decoder": "logistic",
                    "geometry": "LE",
                    "metric": metric,
                    "comparator": condition,
                    "n": int(pooled["evaluation_n"]),
                    "value": float(pooled[metric]),
                    "notes": "pooled OOF metric; not the mean of fold metrics",
                }
            )

    mdm_failures = secondary_mdm_failure_inventory(
        tables, config["dataset"]["subjects"]
    )
    for failure in mdm_failures.itertuples(index=False):
        rows.append(
            {
                **base,
                "row_type": "secondary_mdm_failure",
                "source_table": failure.source_table,
                "protocol": failure.protocol,
                "decoder": "MDM",
                "geometry": failure.geometry,
                "native_metric": failure.native_metric,
                "subject": int(failure.subject),
                "comparator": failure.split,
                "pass_flag": False,
                "notes": f"{failure.failure_kind}: {failure.warning_messages}",
            }
        )

    for row in verdicts.subject_deltas.itertuples(index=False):
        for geometry in ("LE", "AIRM", "EA"):
            rows.append(
                {
                    **base,
                    "row_type": "paired_subject_delta",
                    "source_table": "loso_logistic_transductive.csv",
                    "question": "Q1/Q2" if geometry in {"LE", "AIRM"} else "descriptive",
                    "protocol": "T1",
                    "decoder": "logistic",
                    "geometry": geometry,
                    "native_metric": "euclidean_log_svec",
                    "metric": "balanced_accuracy",
                    "subject": int(row.subject),
                    "comparator": "RAW",
                    "delta": float(getattr(row, f"delta_{geometry}_vs_RAW")),
                    "sign_category": str(getattr(row, f"category_{geometry}_vs_RAW")),
                }
            )
    for geometry in ("LE", "AIRM", "EA"):
        values = verdicts.subject_deltas[f"delta_{geometry}_vs_RAW"]
        categories = verdicts.subject_deltas[f"category_{geometry}_vs_RAW"]
        rows.append(
            {
                **base,
                "row_type": "paired_delta_aggregate",
                "source_table": "loso_logistic_transductive.csv",
                "question": "Q1/Q2" if geometry in {"LE", "AIRM"} else "descriptive",
                "protocol": "T1",
                "decoder": "logistic",
                "geometry": geometry,
                "native_metric": "euclidean_log_svec",
                "metric": "balanced_accuracy",
                "comparator": "RAW",
                **_stats(values),
                "improved_subjects": int(categories.eq("improved").sum()),
                "worsened_subjects": int(categories.eq("worsened").sum()),
                "tied_subjects": int(categories.eq("tied").sum()),
            }
        )

    formulas = {
        "Q1": "For LE and AIRM separately: mean(BA_g-BA_RAW)>=0.01 AND improved_subjects>=6; combine the two pass flags.",
        "Q2": "abs(mean_delta_LE-mean_delta_AIRM)>=0.02 OR opposite non-tie signs OR subject sign-category disagreements>=4.",
        "Q3": "For LE or AIRM: abs(mean_BA_T1-mean_BA_T2)>=0.02.",
    }
    thresholds = {
        "Q1": 0.01,
        "Q2": 0.02,
        "Q3": 0.02,
    }
    for question, verdict in (("Q1", verdicts.q1), ("Q2", verdicts.q2), ("Q3", verdicts.q3)):
        rows.append(
            {
                **base,
                "row_type": "decision_verdict",
                "question": question,
                "metric": "balanced_accuracy",
                "n": 9,
                "threshold": thresholds[question],
                "pass_flag": (
                    verdict in {"ROBUSTLY SUPPORTED", "SUPPORTED", "POTENTIALLY IMPORTANT"}
                ),
                "verdict": verdict,
                "operands_json": json.dumps(
                    verdicts.operands[question],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "formula": formulas[question],
            }
        )
    summary = pd.DataFrame.from_records(rows)
    for column in SUMMARY_COLUMNS:
        if column not in summary:
            summary[column] = pd.NA
    return summary[list(SUMMARY_COLUMNS)]


def build_figure_sources(
    tables: Mapping[str, pd.DataFrame], verdicts: FrozenVerdicts
) -> dict[str, pd.DataFrame]:
    figure_1 = verdicts.t1_ba.copy()
    figure_2 = verdicts.subject_deltas.copy()
    figure_3 = verdicts.t1_ba.merge(
        verdicts.t2_ba,
        on="subject",
        suffixes=("_T1", "_T2"),
        validate="one_to_one",
    )
    means = tables["geometry_mean_comparison.csv"]
    figure_4 = means[
        (means["protocol"].astype(str) == "T1")
        & (means["split"].astype(str) == "ALL")
    ][
        [
            "target_subject",
            "fit_n",
            "normalized_d_le_le_airm",
            "normalized_d_airm_le_airm",
            "le_airm_coordinate_difference_mean_l2",
        ]
    ].rename(columns={"target_subject": "subject"}).sort_values("subject")
    leakage = tables["v1_leakage_audit.csv"].copy()
    leakage["fold_sort"] = leakage["fold"].fillna(99).astype(int)
    figure_5 = leakage.sort_values(
        ["condition", "row_type", "fold_sort"], kind="mergesort"
    )[
        [
            "condition",
            "row_type",
            "fold",
            "balanced_accuracy",
            "accuracy",
            "evaluation_n",
            "center_fit_n",
            "center_evaluation_overlap_n",
        ]
    ].reset_index(drop=True)
    return {
        FIGURE_STEMS[0]: figure_1.reset_index(drop=True),
        FIGURE_STEMS[1]: figure_2.reset_index(drop=True),
        FIGURE_STEMS[2]: figure_3.reset_index(drop=True),
        FIGURE_STEMS[3]: figure_4.reset_index(drop=True),
        FIGURE_STEMS[4]: figure_5,
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def _save_figure(fig: plt.Figure, path: Path) -> None:
    temporary: str | None = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", dir=path.parent, delete=False) as handle:
            temporary = handle.name
        fig.savefig(temporary, format="png", dpi=200, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def _plot_figures(sources: Mapping[str, pd.DataFrame], figures_dir: Path) -> None:
    colors = {"RAW": "#4c4c4c", "LE": "#1f77b4", "AIRM": "#d62728", "EA": "#2ca02c"}
    subjects = np.arange(1, 10)

    frame = sources[FIGURE_STEMS[0]]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for geometry in GEOMETRIES:
        ax.plot(frame["subject"], frame[geometry], marker="o", label=geometry, color=colors[geometry])
    ax.set(xlabel="Target subject", ylabel="Balanced accuracy", title="Primary logistic T1")
    ax.set_xticks(subjects)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, frameon=False)
    _save_figure(fig, figures_dir / f"{FIGURE_STEMS[0]}.png")

    frame = sources[FIGURE_STEMS[1]]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    for geometry in ("LE", "AIRM", "EA"):
        ax.plot(
            frame["subject"],
            frame[f"delta_{geometry}_vs_RAW"],
            marker="o",
            label=f"{geometry} - RAW",
            color=colors[geometry],
        )
    ax.set(xlabel="Target subject", ylabel="Paired BA delta", title="Primary logistic T1")
    ax.set_xticks(subjects)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    _save_figure(fig, figures_dir / f"{FIGURE_STEMS[1]}.png")

    frame = sources[FIGURE_STEMS[2]]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True, sharey=True)
    for ax, geometry in zip(axes.flat, GEOMETRIES, strict=True):
        t1 = frame[f"{geometry}_T1"].to_numpy(dtype=float)
        t2 = frame[f"{geometry}_T2"].to_numpy(dtype=float)
        for subject, first, second in zip(subjects, t1, t2, strict=True):
            ax.plot([subject - 0.10, subject + 0.10], [first, second], color="#bdbdbd", linewidth=0.8)
        ax.scatter(subjects - 0.10, t1, label="T1", color="#9467bd", s=22)
        ax.scatter(subjects + 0.10, t2, label="T2", color="#ff7f0e", s=22)
        ax.axhline(t1.mean(), color="#9467bd", linestyle="--", linewidth=1.0)
        ax.axhline(t2.mean(), color="#ff7f0e", linestyle=":", linewidth=1.2)
        ax.set_title(geometry)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(subjects)
        ax.grid(alpha=0.20)
    axes[0, 0].legend(frameon=False)
    fig.supxlabel("Target subject")
    fig.supylabel("Balanced accuracy")
    fig.suptitle("Primary logistic: T1 versus T2 aggregate")
    _save_figure(fig, figures_dir / f"{FIGURE_STEMS[2]}.png")

    frame = sources[FIGURE_STEMS[3]]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    panels = (
        ("normalized_d_le_le_airm", "LE-normalized center distance"),
        ("normalized_d_airm_le_airm", "AIRM-normalized center distance"),
    )
    for ax, (column, label) in zip(axes, panels, strict=True):
        ax.plot(frame["subject"], frame[column], marker="o", color="#17becf")
        ax.set(xlabel="Subject", ylabel=label)
        ax.set_xticks(subjects)
        ax.grid(alpha=0.25)
    fig.suptitle("T1 LE versus AIRM subject centers (separate scales)")
    _save_figure(fig, figures_dir / f"{FIGURE_STEMS[3]}.png")

    frame = sources[FIGURE_STEMS[4]]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharex=True, sharey=True)
    condition_order = ("v1_all_sample", "fold_safe")
    for ax, metric in zip(axes, ("balanced_accuracy", "accuracy"), strict=True):
        for position, condition in enumerate(condition_order):
            group = frame[frame["condition"] == condition]
            folds = group[group["row_type"] == "fold"]
            pooled = group[group["row_type"] == "pooled_oof"].iloc[0]
            offsets = np.linspace(-0.08, 0.08, len(folds))
            ax.scatter(np.full(len(folds), position) + offsets, folds[metric], s=28, color="#7f7f7f", label="fold" if position == 0 else None)
            ax.scatter(position, pooled[metric], marker="D", s=62, color="#d62728", label="pooled OOF" if position == 0 else None)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks((0, 1), ("all-sample", "fold-safe"))
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False)
    fig.suptitle("V1 centering leakage audit")
    _save_figure(fig, figures_dir / f"{FIGURE_STEMS[4]}.png")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    first = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows]
    return "\n".join([first, divider, *body])


def _aggregate_markdown(frame: pd.DataFrame, *, split: str) -> str:
    selected = frame[frame["split"].astype(str) == split]
    rows: list[list[Any]] = []
    for (geometry, native_metric), group in selected.groupby(
        ["geometry", "native_metric"], sort=True
    ):
        ba = _stats(group["balanced_accuracy"])
        accuracy = _stats(group["accuracy"])
        f1 = _stats(group["macro_f1"])
        rows.append(
            [
                geometry,
                native_metric,
                f"{ba['mean']:.4f} ± {ba['std_ddof1']:.4f}",
                f"{accuracy['mean']:.4f} ± {accuracy['std_ddof1']:.4f}",
                f"{f1['mean']:.4f} ± {f1['std_ddof1']:.4f}",
                f"{ba['median']:.4f} [{ba['minimum']:.4f}, {ba['maximum']:.4f}]",
            ]
        )
    return _md_table(
        ("Geometry", "Native metric", "BA mean ± SD", "Accuracy mean ± SD", "Macro-F1 mean ± SD", "BA median [min, max]"),
        rows,
    )


def _mdm_markdown(frame: pd.DataFrame, *, split: str) -> str:
    rows: list[list[Any]] = []
    for geometry, native_metric in MDM_SPECS:
        group = frame[
            (frame["split"].astype(str) == split)
            & (frame["geometry"].astype(str) == geometry)
            & (frame["native_metric"].astype(str) == native_metric)
            & (frame["status"].astype(str) == "PASS")
        ]
        if group.empty:
            rows.append([geometry, native_metric, "0/9", "NA", "NA", "NA"])
            continue
        ba = _stats(group["balanced_accuracy"])
        accuracy = _stats(group["accuracy"])
        f1 = _stats(group["macro_f1"])
        ba_sd = "NA" if pd.isna(ba["std_ddof1"]) else f"{ba['std_ddof1']:.4f}"
        accuracy_sd = (
            "NA" if pd.isna(accuracy["std_ddof1"]) else f"{accuracy['std_ddof1']:.4f}"
        )
        f1_sd = "NA" if pd.isna(f1["std_ddof1"]) else f"{f1['std_ddof1']:.4f}"
        rows.append(
            [
                geometry,
                native_metric,
                f"{len(group)}/9",
                f"{ba['mean']:.4f} ± {ba_sd}",
                f"{accuracy['mean']:.4f} ± {accuracy_sd}",
                f"{f1['mean']:.4f} ± {f1_sd}",
            ]
        )
    return _md_table(
        (
            "Geometry",
            "Native metric",
            "PASS subjects",
            "BA mean ± SD",
            "Accuracy mean ± SD",
            "Macro-F1 mean ± SD",
        ),
        rows,
    )


def _relative_figure_links() -> str:
    return "\n".join(
        f"- [{stem}.png](../figures/{stem}.png) ([source CSV](../figures/{stem}.csv))"
        for stem in FIGURE_STEMS
    )


def render_report(
    tables: Mapping[str, pd.DataFrame],
    gate: Mapping[str, Any],
    verdicts: FrozenVerdicts,
    config: Mapping[str, Any],
    *,
    config_sha256: str,
) -> str:
    """Render the exact-title, exact-14-section Markdown report."""

    t1 = tables["loso_logistic_transductive.csv"]
    t2 = tables["loso_logistic_calibration.csv"]
    mdm_t1 = tables["loso_mdm_transductive.csv"]
    mdm_t2 = tables["loso_mdm_calibration.csv"]
    means = tables["geometry_mean_comparison.csv"]
    leakage = tables["v1_leakage_audit.csv"]
    domain = tables["domain_shift_diagnostics.csv"]
    correctness = tables["geometry_correctness.csv"]
    mdm_failures = secondary_mdm_failure_inventory(
        tables, config["dataset"]["subjects"]
    )
    if mdm_failures.empty:
        mdm_failure_detail = "No explicit FAILED or missing secondary MDM rows were recorded."
    else:
        grouped_failures = (
            mdm_failures.groupby(
                ["source_table", "failure_kind", "geometry", "native_metric"],
                sort=True,
            )
            .size()
            .reset_index(name="count")
        )
        mdm_failure_detail = _md_table(
            ("Source", "Kind", "Geometry", "Metric", "Count"),
            grouped_failures.itertuples(index=False, name=None),
        )
        explicit = mdm_failures[
            mdm_failures["failure_kind"] == "EXPLICIT_FAILED"
        ]
        if not explicit.empty:
            mdm_failure_detail += "\n\nExplicit failure detail:\n\n" + _md_table(
                ("Protocol", "Subject", "Geometry", "Metric", "Split", "Warning"),
                explicit[
                    [
                        "protocol",
                        "subject",
                        "geometry",
                        "native_metric",
                        "split",
                        "warning_messages",
                    ]
                ].itertuples(index=False, name=None),
            )

    t1_raw = t1[(t1["geometry"] == "RAW") & (t1["split"] == "ALL")]
    total_trials = int(t1_raw["evaluation_n"].sum())
    trials_per_subject = sorted(set(t1_raw["evaluation_n"].astype(int)))
    source_counts = sorted(set(t1_raw["source_n"].astype(int)))
    mean_t1 = means[(means["protocol"] == "T1") & (means["split"] == "ALL")]
    le_center_stats = _stats(mean_t1["normalized_d_le_le_airm"])
    airm_center_stats = _stats(mean_t1["normalized_d_airm_le_airm"])
    coord_stats = _stats(mean_t1["le_airm_coordinate_difference_mean_l2"])

    pooled = leakage[leakage["row_type"] == "pooled_oof"].set_index("condition")
    leakage_rows = []
    for condition in ("v1_all_sample", "fold_safe"):
        row = pooled.loc[condition]
        leakage_rows.append(
            [
                condition,
                row["balanced_accuracy"],
                row["accuracy"],
                row["macro_f1"],
                row["center_evaluation_overlap_n"],
                row["actual_accuracy_difference_from_benchmark"],
            ]
        )
    leakage_delta_accuracy = float(
        pooled.loc["fold_safe", "accuracy"] - pooled.loc["v1_all_sample", "accuracy"]
    )
    leakage_delta_ba = float(
        pooled.loc["fold_safe", "balanced_accuracy"]
        - pooled.loc["v1_all_sample", "balanced_accuracy"]
    )

    domain_t1 = domain[(domain["protocol"] == "T1") & (domain["split"] == "ALL")]
    domain_rows = []
    for (geometry, metric), group in domain_t1.groupby(
        ["geometry", "reference_metric"], sort=True
    ):
        mean_distance = _stats(group["source_target_mean_distance"])
        dispersion = _stats(group["absolute_dispersion_difference"])
        silhouettes = pd.to_numeric(group["all_subject_subject_silhouette"], errors="coerce")
        domain_rows.append(
            [
                geometry,
                metric,
                mean_distance["mean"],
                mean_distance["std_ddof1"],
                dispersion["mean"],
                float(silhouettes.mean()) if silhouettes.notna().any() else np.nan,
            ]
        )

    delta_rows = []
    for geometry in ("LE", "AIRM", "EA"):
        values = verdicts.subject_deltas[f"delta_{geometry}_vs_RAW"]
        categories = verdicts.subject_deltas[f"category_{geometry}_vs_RAW"]
        stats = _stats(values)
        delta_rows.append(
            [
                geometry,
                stats["mean"],
                stats["std_ddof1"],
                int(categories.eq("improved").sum()),
                int(categories.eq("worsened").sum()),
                int(categories.eq("tied").sum()),
            ]
        )

    q1_operands = verdicts.operands["Q1"]
    q2_operands = verdicts.operands["Q2"]
    q3_operands = verdicts.operands["Q3"]
    sections: list[tuple[str, str]] = []
    sections.append(
        (
            REPORT_HEADINGS[0],
            "This audit asks whether V1's subject-marginal centering effect is robust "
            "to covariance geometry and to a held-out-run target-center protocol. It "
            "evaluates WHOLE covariance only; it proposes no classifier or alignment method.",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[1],
            f"Protocol version `{config['protocol']['version']}`, seed `{config['protocol']['seed']}`, "
            f"config SHA-256 `{config_sha256}`. BNCI2014_001 session "
            f"`{config['dataset']['primary_session_key']}` contains {total_trials} validated WHOLE trials "
            f"({len(config['dataset']['subjects'])} subjects; {trials_per_subject} evaluation trials per subject), "
            f"{len(config['dataset']['eeg_channels'])} EEG channels, and four classes. Each LOSO target has "
            f"source count(s) {source_counts}. Frozen preprocessing is {config['preprocessing']['bandpass_hz']} Hz, "
            f"{config['preprocessing']['sampling_frequency_hz']} Hz, {config['preprocessing']['samples_per_trial']} samples, "
            f"OAS covariance, float64 covariance geometry, no scaler/PCA/tuning. T1 fits each centered target mean "
            "on all 288 target covariates and evaluates those same trials. T2 fits on three runs and evaluates the "
            "disjoint other three, with A/B reversed splits; primary T2 rows are the preregistered `AGGREGATE` rows.",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[2],
            "G0 `RAW` uses each covariance unchanged. G1 `LE` subtracts the subject Log-Euclidean mean in log "
            "coordinates and is V1-svec equivalent. G2 `AIRM` uses the affine-invariant Fréchet mean and congruence "
            "whitening. G3 `EA` uses arithmetic-mean congruence only and is not interpreted as a Riemannian method. "
            "The common primary decoder is unscaled log-svec multinomial logistic regression (`C=1`, `lbfgs`, "
            "`max_iter=5000`, `tol=1e-4`).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[3],
            f"The classification hard gate was all-pass: {gate['passed_required_rows']}/"
            f"{gate['required_rows']} required rows passed and {gate['failed_required_rows']} failed. "
            f"The correctness table contains {len(correctness)} total checks. Classification/reporting would "
            "hard-stop for a missing, malformed, provenance-mismatched, or non-PASS gate. "
            "[Correctness table](../tables/geometry_correctness.csv).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[4],
            _md_table(
                ("T1 normalized center quantity", "Mean", "SD", "Median", "Min", "Max"),
                (
                    ("dLE(LE,AIRM) / LE dispersion", le_center_stats["mean"], le_center_stats["std_ddof1"], le_center_stats["median"], le_center_stats["minimum"], le_center_stats["maximum"]),
                    ("dAI(LE,AIRM) / AIRM dispersion", airm_center_stats["mean"], airm_center_stats["std_ddof1"], airm_center_stats["median"], airm_center_stats["minimum"], airm_center_stats["maximum"]),
                    ("Mean transformed-coordinate L2 difference", coord_stats["mean"], coord_stats["std_ddof1"], coord_stats["median"], coord_stats["minimum"], coord_stats["maximum"]),
                ),
            )
            + "\n\nThese describe measured mean differences; they do not identify one geometry as correct. "
            f"[Figure 4](../figures/{FIGURE_STEMS[3]}.png) and [source CSV](../figures/{FIGURE_STEMS[3]}.csv).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[5],
            _md_table(
                ("Condition", "Pooled BA", "Pooled accuracy", "Pooled macro-F1", "Center/eval overlap", "Accuracy - 0.6119"),
                leakage_rows,
            )
            + f"\n\nThe observed fold-safe minus all-sample differences were {leakage_delta_ba:.4f} BA and "
            f"{leakage_delta_accuracy:.4f} accuracy. The published V1 value `0.6119` is an audit benchmark, not a "
            "threshold or forced reproduction. All-sample centering has evaluation-covariate overlap; fold-safe "
            "centering does not. [Figure 5](../figures/figure_5_v1_leakage_audit.png) and "
            "[source CSV](../figures/figure_5_v1_leakage_audit.csv).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[6],
            _aggregate_markdown(t1, split="ALL")
            + "\n\nAll values aggregate nine target-subject rows (SD uses `ddof=1`). T1 centered target means "
            "use the same 288 unlabeled target covariates that are evaluated; this is transductive label-free "
            "target centering, not inductive evaluation. "
            f"[Figure 1](../figures/{FIGURE_STEMS[0]}.png), [paired deltas](../figures/{FIGURE_STEMS[1]}.png).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[7],
            _aggregate_markdown(t2, split="AGGREGATE")
            + "\n\nEach subject-level primary row pools the two deterministic A/B held-out-run evaluations. "
            "Target center-fit and evaluation trial UIDs are disjoint within each split; source data and decoder "
            "are identical to T1. "
            f"[Figure 3](../figures/{FIGURE_STEMS[2]}.png) and [source CSV](../figures/{FIGURE_STEMS[2]}.csv).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[8],
            "T1 metric-native MDM:\n\n"
            + _mdm_markdown(mdm_t1, split="ALL")
            + "\n\nT2 metric-native MDM:\n\n"
            + _mdm_markdown(mdm_t2, split="AGGREGATE")
            + f"\n\nSecondary MDM inventory: {len(mdm_failures)} explicit FAILED or missing logical rows "
            + (
                "(none)."
                if mdm_failures.empty
                else "(`EXPLICIT_FAILED` and subsequent `MISSING_AFTER_SECONDARY_FAILURE` rows are listed in the machine-readable summary)."
            )
            + " MDM aggregates use PASS rows only and display their denominator. MDM is secondary and does not "
            "vote in Q1–Q3. RAW is intentionally evaluated under both `riemann` and `logeuclid`; EA remains an "
            "arithmetic-control transformation.\n\n"
            + mdm_failure_detail,
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[9],
            _md_table(
                ("Geometry", "Reference metric", "Mean source-target mean distance", "SD", "Mean absolute dispersion difference", "Subject silhouette"),
                domain_rows,
            )
            + "\n\nThese T1 diagnostics use no class labels. They quantify marginal domain location/dispersion and "
            "subject structure; they do not establish conditional class alignment. "
            "[Domain table](../tables/domain_shift_diagnostics.csv).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[10],
            _md_table(
                ("Geometry", "Mean T1 BA delta vs RAW", "SD", "Improved", "Worsened", "Tied"),
                delta_rows,
            )
            + "\n\n"
            + _md_table(
                ("Question", "Verdict", "Measured operands"),
                (
                    ("Q1", verdicts.q1, json.dumps(q1_operands, sort_keys=True, separators=(",", ":"))),
                    ("Q2", verdicts.q2, json.dumps(q2_operands, sort_keys=True, separators=(",", ":"))),
                    ("Q3", verdicts.q3, json.dumps(q3_operands, sort_keys=True, separators=(",", ":"))),
                ),
            )
            + "\n\nRules were applied to unrounded subject-level BA: Q1 requires both mean delta `>=0.01` and "
            "at least `6/9` improved subjects for each geometry; Q2 is supported by any preregistered geometry "
            "difference condition; Q3 is potentially important if either LE or AIRM has absolute T1/T2 mean BA "
            "difference `>=0.02`. [Machine-readable summary](../tables/geometry_v2_summary.csv).",
        )
    )

    justified: list[str] = []
    if verdicts.q1 == "ROBUSTLY SUPPORTED":
        justified.append("Both LE and AIRM passed the frozen Q1 mean-delta and subject-count criteria.")
    elif verdicts.q1 == "GEOMETRY-SENSITIVE-MIXED":
        justified.append("Exactly one of LE/AIRM passed Q1; improvement is geometry-sensitive under this pilot.")
    else:
        justified.append("Neither LE nor AIRM passed both Q1 criteria; robust improvement is not supported.")
    if verdicts.q2 == "SUPPORTED":
        justified.append("At least one frozen Q2 operand crossed threshold, so the conclusion is geometry-sensitive.")
    else:
        justified.append("No frozen Q2 operand crossed threshold; material LE/AIRM dependence was not detected here.")
    if verdicts.q3 == "POTENTIALLY IMPORTANT":
        justified.append("At least one centered geometry changed by at least 0.02 mean BA between T1 and T2.")
    else:
        justified.append("Both centered geometries changed by less than 0.02 mean BA between T1 and T2.")
    sections.append((REPORT_HEADINGS[11], "\n\n".join(justified)))
    sections.append(
        (
            REPORT_HEADINGS[12],
            "This audit does not justify a conditional-alignment method, a distribution classifier, a neural "
            "architecture, a temporal or trajectory model, or any WINDOW5 conclusion. It does not show that "
            "target-label-free conditional structure is identifiable, and it does not demonstrate domain-adaptation "
            "improvement. T1 is transductive, T2 uses fixed half-run calibration, MDM is secondary, and all results "
            "come from one dataset/session and one frozen preprocessing pipeline.",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[13],
            f"Run exactly one next experiment: **{verdicts.next_experiment}**. Keep the present data, preprocessing, "
            "geometry definitions, decoder, seed, and reporting thresholds fixed; preregister that experiment in "
            "a new output namespace before reading its results.",
        )
    )
    text = REPORT_TITLE + "\n\n"
    text += "\n\n".join(f"## {heading}\n\n{body}" for heading, body in sections)
    text += "\n\n### Figure and source index\n\n" + _relative_figure_links() + "\n"
    validate_report_contract(text)
    return text


def validate_report_contract(text: str) -> None:
    lines = text.splitlines()
    if not lines or lines[0] != REPORT_TITLE:
        raise ReportingContractError("report title is not exact")
    headings = [line[3:] for line in lines if line.startswith("## ")]
    if tuple(headings) != REPORT_HEADINGS:
        raise ReportingContractError(f"report headings are not exact: {headings}")
    for stem in FIGURE_STEMS:
        if f"../figures/{stem}.png" not in text or f"../figures/{stem}.csv" not in text:
            raise ReportingContractError(f"report lacks figure/source link for {stem}")


def create_reporting_outputs(
    tables: Mapping[str, pd.DataFrame],
    gate: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    tables_dir: str | Path,
    figures_dir: str | Path,
    report_path: str | Path,
) -> ReportingArtifacts:
    """Validate all inputs, then create only the nine frozen reporting artifacts."""

    normalized = validate_reporting_inputs(
        tables, gate, config, config_sha256=config_sha256
    )
    t1 = normalized["loso_logistic_transductive.csv"]
    t2 = normalized["loso_logistic_calibration.csv"]
    verdicts = compute_frozen_verdicts(t1, t2, config)
    summary = build_geometry_summary(
        normalized, verdicts, config, config_sha256=config_sha256
    )
    sources = build_figure_sources(normalized, verdicts)
    figure_directory = Path(figures_dir).expanduser().resolve()
    table_directory = Path(tables_dir).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    expected_names = {
        *(f"{stem}.png" for stem in FIGURE_STEMS),
        *(f"{stem}.csv" for stem in FIGURE_STEMS),
    }
    if figure_directory.exists():
        existing = {
            path.name
            for path in figure_directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".csv"}
        }
        unexpected = existing - expected_names
        if unexpected:
            raise ReportingContractError(
                f"figures directory contains non-protocol PNG/CSV files: {sorted(unexpected)}"
            )
    figure_directory.mkdir(parents=True, exist_ok=True)
    _atomic_csv(summary, table_directory / "geometry_v2_summary.csv")
    for stem in FIGURE_STEMS:
        _atomic_csv(sources[stem], figure_directory / f"{stem}.csv")
    _plot_figures(sources, figure_directory)
    report = render_report(
        normalized, gate, verdicts, config, config_sha256=config_sha256
    )
    _atomic_text(report, report_file)

    actual_names = {
        path.name
        for path in figure_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".csv"}
    }
    if actual_names != expected_names:
        raise ReportingContractError(
            f"figure artifact set mismatch: expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )
    if not (table_directory / "geometry_v2_summary.csv").is_file():
        raise ReportingContractError("geometry_v2_summary.csv was not written")
    if not report_file.is_file():
        raise ReportingContractError("geometry audit report was not written")
    validate_report_contract(report_file.read_text(encoding="utf-8"))
    return ReportingArtifacts(
        summary=summary,
        verdicts=verdicts,
        figure_sources=sources,
        report_text=report,
    )
