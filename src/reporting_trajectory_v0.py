"""Deterministic reporting for the frozen Trajectory Anatomy v0 protocol.

This module is deliberately downstream-only.  It reads completed result tables
and the two preregistered null streams, validates their reporting contract,
evaluates the frozen hypotheses without available-case substitution, and emits
the one summary table, eight fixed figures, and eighteen-section report.

No classifier, covariance, trajectory feature, null replicate, or scientific
statistic is fitted here.  AIRM alone votes in the terminal decision; LE is a
reported robustness analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROTOCOL_VERSION = "0.0"
MASTER_SEED = 20260809
SESSION = "0train"
SUBJECTS = tuple(range(1, 10))
CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
SCALARS_11 = (
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
)

REPORT_TITLE = "# BNCI2014_001 Trajectory Anatomy v0"
REPORT_HEADINGS = (
    "Scientific question",
    "Why V1 did not test this question",
    "Frozen protocol",
    "Geometry correctness",
    "Five-state AIRM geometry",
    "BAG vs PATH definition",
    "Intrinsic path quantities",
    "Class LOSO results",
    "Order-shuffle falsification",
    "Label-destruction null",
    "Subject-information results",
    "Class vs subject vs interaction effects",
    "LOCAL_BARYCENTER / WHOLE contextual controls",
    "AIRM vs LE robustness",
    "Frozen verdict",
    "What is actually justified",
    "What is NOT justified",
    "Single recommended next step",
)

FIGURE_STEMS = (
    "figure_1_class_loso_ba",
    "figure_2_order_shuffle_null",
    "figure_3_label_destruction_null",
    "figure_4_scalar_eta2",
    "figure_5_scalars_by_class",
    "figure_6_scalars_by_subject",
    "figure_7_subject_probe",
    "figure_8_airm_le_robustness",
)

PREREQUISITE_FILENAMES = (
    "dataset_contract.csv",
    "covariance_sanity.csv",
    "trajectory_geometry_correctness.csv",
    "trial_airm_path_features.csv",
    "trial_le_path_features.csv",
    "airm_path_d10.csv",
    "airm_bag_canon_d10.csv",
    "airm_bag_sorted_d10.csv",
    "le_path_d10.csv",
    "le_bag_canon_d10.csv",
    "class_loso_metrics.csv",
    "subject_runhalf_probe.csv",
    "scalar_factor_decomposition.csv",
    "order_shuffle_subject_metrics.csv",
    "order_shuffle_group_metrics.csv",
    "label_null_subject_metrics.csv",
    "label_null_group_metrics.csv",
    "local_barycenter_mdm.csv",
    "whole_context_mdm.csv",
    "airm_le_robustness.csv",
)
SUMMARY_FILENAME = "trajectory_v0_summary.csv"
REQUIRED_TABLE_FILENAMES = (*PREREQUISITE_FILENAMES, SUMMARY_FILENAME)
GEOMETRY_GATE_FILENAME = "trajectory_geometry_gate.json"
NULL_FILENAMES = (
    "order_shuffle_seeds.json",
    "order_shuffle_group_stats.npz",
    "label_permutation_seeds.json",
    "label_null_group_stats.npz",
)

COMMON_COLUMNS = (
    "protocol_version",
    "protocol_sha256",
    "config_sha256",
    "seed",
    "session",
    "generated_at_utc",
    "status",
)

TRIAL_IDENTITY = (
    "sample_index",
    "subject",
    "run",
    "trial_id",
    "trial_uid",
    "class_label",
)

TABLE_REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "dataset_contract.csv": (
        "check", "observed", "expected", "comparator", "required", "passed",
        "failure_message",
    ),
    "covariance_sanity.csv": (
        *TRIAL_IDENTITY, "window_index", "symmetry_relative_error", "min_eigenvalue",
        "max_eigenvalue", "condition_number", "has_nan", "has_inf", "is_spd",
        "required", "passed",
    ),
    "trajectory_geometry_correctness.csv": (
        "geometry", "subject", "sample_index", "trial_uid", "check", "statistic",
        "value", "threshold", "comparator", "absolute_error", "relative_error",
        "required", "passed", "failure_message",
    ),
    "trial_airm_path_features.csv": (
        *TRIAL_IDENTITY, "geometry", "s1", "s2", "s3", "s4", "total_path_length",
        "endpoint_distance", "efficiency", "excess", "theta2", "theta3", "theta4",
        "mean_turn", "max_turn", "dev2", "dev3", "dev4",
        "mean_geodesic_deviation", "max_geodesic_deviation", "frechet_variance",
        "frechet_radius_mean", "diameter", "degenerate",
    ),
    "trial_le_path_features.csv": (
        *TRIAL_IDENTITY, "geometry", "s1", "s2", "s3", "s4", "total_path_length",
        "endpoint_distance", "efficiency", "excess", "theta2", "theta3", "theta4",
        "mean_turn", "max_turn", "dev2", "dev3", "dev4",
        "mean_geodesic_deviation", "max_geodesic_deviation", "frechet_variance",
        "frechet_radius_mean", "diameter", "degenerate",
    ),
    "airm_path_d10.csv": (*TRIAL_IDENTITY, "geometry", "representation",
        "d12", "d13", "d14", "d15", "d23", "d24", "d25", "d34", "d35", "d45"),
    "airm_bag_canon_d10.csv": (*TRIAL_IDENTITY, "geometry", "representation",
        "bag01", "bag02", "bag03", "bag04", "bag05", "bag06", "bag07", "bag08",
        "bag09", "bag10", "canonical_permutation"),
    "airm_bag_sorted_d10.csv": (*TRIAL_IDENTITY, "geometry", "representation",
        "sorted01", "sorted02", "sorted03", "sorted04", "sorted05", "sorted06",
        "sorted07", "sorted08", "sorted09", "sorted10"),
    "le_path_d10.csv": (*TRIAL_IDENTITY, "geometry", "representation",
        "d12", "d13", "d14", "d15", "d23", "d24", "d25", "d34", "d35", "d45"),
    "le_bag_canon_d10.csv": (*TRIAL_IDENTITY, "geometry", "representation",
        "bag01", "bag02", "bag03", "bag04", "bag05", "bag06", "bag07", "bag08",
        "bag09", "bag10", "canonical_permutation"),
    "class_loso_metrics.csv": (
        "geometry", "representation", "target_subject", "source_subjects", "train_n",
        "test_n", "train_uid_sha256", "test_uid_sha256", "scaler_fit_uid_sha256",
        "balanced_accuracy", "accuracy", "macro_f1", "recall_left_hand",
        "recall_right_hand", "recall_feet", "recall_tongue", "confusion_matrix_json",
        "prediction_sha256", "classifier_config_sha256", "convergence_warning",
        "warning_messages",
    ),
    "subject_runhalf_probe.csv": (
        "geometry", "representation", "split", "train_runs", "evaluation_runs",
        "train_n", "test_n", "train_uid_sha256", "test_uid_sha256", "chance_level",
        "balanced_accuracy", "accuracy", "direction_average_ba",
        "direction_average_accuracy", "prediction_sha256", "classifier_config_sha256",
        "convergence_warning", "warning_messages",
    ),
    "scalar_factor_decomposition.csv": (
        "geometry", "scalar", "n_subjects", "n_classes", "n_per_cell", "grand_mean",
        "ss_subject", "ss_class", "ss_interaction", "ss_residual", "ss_total",
        "eta2_subject", "eta2_class", "eta2_interaction", "eta2_residual",
        "ss_reconstruction_relative_error", "degenerate", "uses_p_value",
    ),
    "order_shuffle_subject_metrics.csv": (
        "geometry", "representation", "replicate", "replicate_seed", "target_subject",
        "balanced_accuracy", "accuracy", "macro_f1", "observed_ba",
        "subject_null_median_ba", "subject_effect", "train_uid_sha256",
        "test_uid_sha256", "classifier_status", "convergence_warning",
        "warning_messages",
    ),
    "order_shuffle_group_metrics.csv": (
        "geometry", "representation", "observed_median_subject_ba", "null_replicates",
        "null_median", "null_mean", "null_sd_ddof1", "null_min", "null_max", "effect",
        "p_value", "exceedance_count", "median_subject_path_minus_bag",
        "hypothesis_operand_pass",
    ),
    "label_null_subject_metrics.csv": (
        "geometry", "representation", "replicate", "replicate_seed", "target_subject",
        "balanced_accuracy", "accuracy", "macro_f1", "observed_ba",
        "subject_null_median_ba", "subject_effect", "train_uid_sha256",
        "test_uid_sha256", "classifier_status", "convergence_warning",
        "warning_messages",
    ),
    "label_null_group_metrics.csv": (
        "geometry", "representation", "observed_median_subject_ba", "null_replicates",
        "null_median", "null_mean", "null_sd_ddof1", "null_min", "null_max", "effect",
        "p_value", "exceedance_count", "hypothesis_operand_pass",
    ),
    "local_barycenter_mdm.csv": (
        "representation", "target_subject", "train_n", "test_n", "train_uid_sha256",
        "test_uid_sha256", "metric", "balanced_accuracy", "accuracy", "macro_f1",
        "recall_left_hand", "recall_right_hand", "recall_feet", "recall_tongue",
        "confusion_matrix_json", "prediction_sha256", "convergence_warning",
        "warning_messages",
    ),
    "whole_context_mdm.csv": (
        "representation", "target_subject", "train_n", "test_n", "train_uid_sha256",
        "test_uid_sha256", "metric", "balanced_accuracy", "accuracy", "macro_f1",
        "recall_left_hand", "recall_right_hand", "recall_feet", "recall_tongue",
        "confusion_matrix_json", "prediction_sha256", "convergence_warning",
        "warning_messages", "covariance_samples_per_estimate",
        "estimator_regime_confounded", "interpretation_limit",
    ),
    "airm_le_robustness.csv": (
        "analysis", "representation", "subject", "scalar", "airm_value", "le_value",
        "paired_delta", "delta_category", "airm_status", "le_status",
        "agreement_category", "interpretation",
    ),
}

SUMMARY_COLUMNS = (
    *COMMON_COLUMNS,
    "row_type", "hypothesis", "operand", "formula", "value", "threshold",
    "comparator", "pass_flag", "verdict", "source_table", "failure_type",
    "failure_detail", "interpretation",
)

VERDICT_GO_ORDER = "GO_TRAJECTORY_ORDER"
VERDICT_GO_BAG = "GO_UNORDERED_DISTRIBUTION"
VERDICT_MIXED = "MIXED_TRAJECTORY_SIGNAL"
VERDICT_STOP = "STOP_LOCAL_TRAJECTORY_V0"
VERDICT_UNASSESSED = "UNASSESSED"
FAILURE_NUMERICAL = "UNASSESSED — NUMERICAL/DATA FAILURE"
FAILURE_TECHNICAL = "UNASSESSED—PROTOCOL/TECHNICAL FAILURE"
ROBUSTNESS_CONSISTENT = "AIRM+LE CONSISTENT"
ROBUSTNESS_AIRM_SPECIFIC = "AIRM-SPECIFIC"
ROBUSTNESS_DISCORDANT = "AIRM/LE DISCORDANT"

_PASS_TOKENS = {"PASS", "PASSED", "SUCCESS", "OK"}
_REPR_ALIASES = {
    "PATH": "PATH_D10",
    "PATH_D10": "PATH_D10",
    "BAG": "BAG_CANON_D10",
    "BAG_CANON": "BAG_CANON_D10",
    "BAG_CANON_D10": "BAG_CANON_D10",
    "BAG_SORTED": "BAG_SORTED_D10",
    "BAG_SORTED_D10": "BAG_SORTED_D10",
    "SCALARS": "SCALARS_11",
    "SCALARS_11": "SCALARS_11",
    "LOCAL_BARYCENTER": "LOCAL_BARYCENTER",
    "WHOLE": "WHOLE-1000",
}


class ReportingContractError(RuntimeError):
    """Raised for a programmer-facing misuse of the reporting API."""


@dataclass(frozen=True)
class ValidationResult:
    numerical_failures: tuple[str, ...]
    technical_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    generated_at_utc: str
    protocol_version: str
    protocol_sha256: str
    config_sha256: str

    @property
    def passed(self) -> bool:
        return not self.numerical_failures and not self.technical_failures


@dataclass(frozen=True)
class FrozenVerdict:
    verdict: str
    failure_status: str
    h_path_class: bool | None
    h_bag_class: bool | None
    h_order: bool | None
    operands: Mapping[str, float]
    operand_pass: Mapping[str, bool | None]
    numerical_failures: tuple[str, ...]
    technical_failures: tuple[str, ...]
    next_experiment: str


@dataclass(frozen=True)
class ReportingArtifacts:
    summary: pd.DataFrame
    verdict: FrozenVerdict
    validation: ValidationResult
    figure_sources: Mapping[str, pd.DataFrame]
    report_text: str
    decision_sha256: str


def _canon_geometry(value: object) -> str:
    return str(value).strip().upper().replace("LOG-EUCLIDEAN", "LE")


def _canon_representation(value: object) -> str:
    raw = str(value).strip().upper()
    if raw == "WHOLE-1000":
        return "WHOLE-1000"
    token = raw.replace("-", "_").replace(" ", "_")
    return _REPR_ALIASES.get(token, token)


def _bool_value(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "pass", "passed"}:
            return True
        if token in {"false", "0", "no", "fail", "failed"}:
            return False
    if pd.isna(value):
        return None
    return None


def _is_pass_status(value: object) -> bool:
    return str(value).strip().upper() in _PASS_TOKENS


def _warning_is_clear(value: object) -> bool:
    parsed = _bool_value(value)
    return parsed is False


def _finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _fmt(value: object, digits: int = 6) -> str:
    if not _finite_number(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _failure_detail(frame: pd.DataFrame, index: object, prefix: str) -> str:
    row = frame.loc[index]
    identities = []
    for column in (
        "sample_index", "subject", "target_subject", "run", "trial_uid", "window_index",
        "geometry", "representation", "replicate", "split", "check", "scalar",
    ):
        if column in frame.columns and not pd.isna(row[column]):
            identities.append(f"{column}={row[column]}")
    if "warning_messages" in frame.columns and str(row["warning_messages"]).strip() not in {
        "", "[]", "nan", "None",
    }:
        identities.append(f"warning_messages={row['warning_messages']}")
    if "failure_message" in frame.columns and str(row["failure_message"]).strip() not in {
        "", "nan", "None",
    }:
        identities.append(f"failure_message={row['failure_message']}")
    return f"{prefix}: " + (", ".join(identities) if identities else f"row={index}")


def _empty_frame(name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=(*COMMON_COLUMNS, *TABLE_REQUIRED_COLUMNS.get(name, ())))


def _frame(tables: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    value = tables.get(name)
    if value is None:
        return _empty_frame(name)
    if not isinstance(value, pd.DataFrame):
        raise ReportingContractError(f"{name} must be a pandas DataFrame")
    return value.copy()


def _numeric_values(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if frame.empty or not columns:
        return np.empty((len(frame), 0), dtype=float)
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def expected_null_seeds(kind: str) -> np.ndarray:
    """Return the exact preregistered uint64 child seeds for one null stream."""

    if kind == "order":
        tag = 0x4F52444552
    elif kind == "label":
        tag = 0x4C4142454C
    else:
        raise ValueError("kind must be 'order' or 'label'")
    children = np.random.SeedSequence([MASTER_SEED, tag]).spawn(199)
    return np.asarray(
        [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children],
        dtype=np.uint64,
    )


def expected_seed_json(kind: str) -> dict[str, object]:
    """Build the exact seed-manifest content required by the protocol."""

    tag = 0x4F52444552 if kind == "order" else 0x4C4142454C
    seeds = expected_null_seeds(kind)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "master_seed": MASTER_SEED,
        "stream_tag_hex": f"0x{tag:X}",
        "seedsequence_entropy": [MASTER_SEED, tag],
        "child_count": 199,
        "seed_dtype": "uint64",
        "seed_extraction": "int(child.generate_state(1, dtype=np.uint64)[0])",
        "replicates": [
            {"replicate": index, "seed": int(seed)}
            for index, seed in enumerate(seeds.tolist(), start=1)
        ],
    }


def _common_provenance_failures(
    name: str,
    frame: pd.DataFrame,
    *,
    protocol_version: str,
    protocol_sha256: str,
    config_sha256: str,
) -> list[str]:
    failures: list[str] = []
    if tuple(frame.columns[: len(COMMON_COLUMNS)]) != COMMON_COLUMNS:
        failures.append(
            f"{name}: first columns must be exactly {','.join(COMMON_COLUMNS)}"
        )
        return failures
    expected = {
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "seed": MASTER_SEED,
        "session": SESSION,
    }
    for column, required_value in expected.items():
        observed = set(frame[column].astype(str))
        if observed != {str(required_value)}:
            failures.append(
                f"{name}.{column}: expected only {required_value!r}, observed {sorted(observed)!r}"
            )
    if frame["generated_at_utc"].isna().any():
        failures.append(f"{name}.generated_at_utc contains NA")
    return failures


def _status_failures(frame: pd.DataFrame, name: str) -> list[str]:
    failures: list[str] = []
    if "status" not in frame:
        return [f"{name}: missing status"]
    for index in frame.index[~frame["status"].map(_is_pass_status)]:
        failures.append(_failure_detail(frame, index, f"{name} status is not PASS"))
    if "convergence_warning" in frame:
        for index in frame.index[~frame["convergence_warning"].map(_warning_is_clear)]:
            failures.append(_failure_detail(frame, index, f"{name} convergence warning"))
    if "classifier_status" in frame:
        for index in frame.index[~frame["classifier_status"].map(_is_pass_status)]:
            failures.append(_failure_detail(frame, index, f"{name} classifier is not PASS"))
    return failures


def _validate_null_manifest(
    kind: str,
    payload: object,
    protocol_version: str,
) -> list[str]:
    name = "order_shuffle_seeds.json" if kind == "order" else "label_permutation_seeds.json"
    if not isinstance(payload, Mapping):
        return [f"{name}: missing or invalid JSON object"]
    failures: list[str] = []
    expected = expected_seed_json(kind)
    fields = (
        "master_seed", "stream_tag_hex", "seedsequence_entropy", "child_count",
        "seed_dtype", "seed_extraction", "replicates",
    )
    if str(payload.get("protocol_version")) != str(protocol_version):
        failures.append(f"{name}.protocol_version mismatch")
    for field in fields:
        if payload.get(field) != expected[field]:
            failures.append(f"{name}.{field} differs from the frozen stream")
    return failures


def validate_geometry_gate(
    gate: object,
    *,
    protocol_version: str,
    protocol_sha256: str,
    config_sha256: str,
) -> tuple[str, ...]:
    """Validate the persisted numerical/data gate before any scientific verdict.

    The current stage-20 artifact calls the total number of window-state rows
    ``n_windows`` (12,960).  A future producer may additionally or instead use
    the unambiguous ``window_state_rows`` and ``windows_per_trial`` fields.  In
    every accepted form, the same 2,592 × 5 contract is enforced.
    """

    if not isinstance(gate, Mapping):
        return (f"{GEOMETRY_GATE_FILENAME}: missing or invalid JSON object",)
    failures: list[str] = []

    for field in ("gate_passed", "scientific_classification_allowed"):
        value = gate.get(field)
        if not isinstance(value, (bool, np.bool_)) or bool(value) is not True:
            failures.append(f"{GEOMETRY_GATE_FILENAME}.{field} must be boolean true")
    if str(gate.get("status")) != "PASS":
        failures.append(f"{GEOMETRY_GATE_FILENAME}.status must be exactly PASS")

    expected = {
        "protocol_version": str(protocol_version),
        "protocol_sha256": str(protocol_sha256),
        "config_sha256": str(config_sha256),
        "seed": str(MASTER_SEED),
        "session": SESSION,
    }
    for field, required in expected.items():
        if str(gate.get(field)) != required:
            failures.append(
                f"{GEOMETRY_GATE_FILENAME}.{field}: expected {required!r}, "
                f"observed {gate.get(field)!r}"
            )

    if not _finite_number(gate.get("n_trials")) or float(gate["n_trials"]) != 2592.0:
        failures.append(f"{GEOMETRY_GATE_FILENAME}.n_trials must equal 2592")

    explicit_rows = gate.get("window_state_rows")
    legacy_n_windows = gate.get("n_windows")
    if explicit_rows is not None:
        if not _finite_number(explicit_rows) or float(explicit_rows) != 12960.0:
            failures.append(
                f"{GEOMETRY_GATE_FILENAME}.window_state_rows must equal 12960"
            )
        # If both names are present, n_windows may retain the current total-row
        # meaning or adopt the clearer per-trial meaning.  No other value is valid.
        if legacy_n_windows is not None and (
            not _finite_number(legacy_n_windows)
            or float(legacy_n_windows) not in {5.0, 12960.0}
        ):
            failures.append(
                f"{GEOMETRY_GATE_FILENAME}.n_windows must equal 5 or 12960 "
                "when window_state_rows is explicit"
            )
    elif not _finite_number(legacy_n_windows) or float(legacy_n_windows) != 12960.0:
        failures.append(
            f"{GEOMETRY_GATE_FILENAME} must record total window-state count 12960 "
            "as current n_windows or window_state_rows"
        )

    windows_per_trial = gate.get("windows_per_trial")
    if windows_per_trial is not None and (
        not _finite_number(windows_per_trial) or float(windows_per_trial) != 5.0
    ):
        failures.append(f"{GEOMETRY_GATE_FILENAME}.windows_per_trial must equal 5")
    if 2592 * 5 != 12960:  # explicit executable statement of the frozen semantics
        failures.append("internal trajectory count contract is inconsistent")

    required_failures = gate.get("required_failure_counts")
    required_keys = {
        "dataset_contract", "covariance_sanity", "trajectory_geometry_correctness"
    }
    if not isinstance(required_failures, Mapping):
        failures.append(
            f"{GEOMETRY_GATE_FILENAME}.required_failure_counts must be a mapping"
        )
    else:
        missing = required_keys - set(required_failures)
        if missing:
            failures.append(
                f"{GEOMETRY_GATE_FILENAME}.required_failure_counts missing {sorted(missing)}"
            )
        for key, value in required_failures.items():
            if not _finite_number(value) or float(value) != 0.0:
                failures.append(
                    f"{GEOMETRY_GATE_FILENAME}.required_failure_counts[{key!r}] "
                    "must equal zero"
                )
    return tuple(dict.fromkeys(failures))


def _as_npz_mapping(value: object) -> Mapping[str, np.ndarray] | None:
    if isinstance(value, Mapping):
        return {str(key): np.asarray(item) for key, item in value.items()}
    return None


def _validate_null_npz(kind: str, payload: object) -> list[str]:
    name = "order_shuffle_group_stats.npz" if kind == "order" else "label_null_group_stats.npz"
    arrays = _as_npz_mapping(payload)
    if arrays is None:
        return [f"{name}: missing or invalid NPZ mapping"]
    required_stats = (
        ("airm__path_d10__median_subject_ba", "le__path_d10__median_subject_ba")
        if kind == "order"
        else ("airm__path_d10__median_subject_ba", "airm__bag_canon_d10__median_subject_ba")
    )
    expected_keys = {"replicate", "replicate_seed", *required_stats}
    failures: list[str] = []
    if set(arrays) != expected_keys:
        failures.append(
            f"{name}: keys differ; expected {sorted(expected_keys)}, observed {sorted(arrays)}"
        )
        return failures
    replicate = arrays["replicate"]
    seeds = arrays["replicate_seed"]
    if replicate.dtype != np.dtype("int64") or replicate.shape != (199,):
        failures.append(f"{name}.replicate must be int64 shape (199,)")
    elif not np.array_equal(replicate, np.arange(1, 200, dtype=np.int64)):
        failures.append(f"{name}.replicate must equal 1..199")
    if seeds.dtype != np.dtype("uint64") or seeds.shape != (199,):
        failures.append(f"{name}.replicate_seed must be uint64 shape (199,)")
    elif not np.array_equal(seeds, expected_null_seeds(kind)):
        failures.append(f"{name}.replicate_seed differs from the frozen stream")
    for key in required_stats:
        array = arrays[key]
        if array.dtype != np.dtype("float64") or array.shape != (199,):
            failures.append(f"{name}.{key} must be float64 shape (199,)")
        elif not np.isfinite(array).all():
            failures.append(f"{name}.{key} contains non-finite values")
    return failures


def _scientific_key_grid_failures(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    failures: list[str] = []

    observed = _frame(tables, "class_loso_metrics.csv")
    if not observed.empty and {"geometry", "representation", "target_subject"} <= set(observed):
        keys = {
            (_canon_geometry(row.geometry), _canon_representation(row.representation), int(row.target_subject))
            for row in observed[["geometry", "representation", "target_subject"]].itertuples(index=False)
            if _finite_number(row.target_subject)
        }
        expected_conditions = (
            ("AIRM", "PATH_D10"), ("AIRM", "BAG_CANON_D10"),
            ("AIRM", "BAG_SORTED_D10"), ("AIRM", "SCALARS_11"),
            ("LE", "PATH_D10"), ("LE", "BAG_CANON_D10"), ("LE", "SCALARS_11"),
        )
        expected = {(g, r, s) for g, r in expected_conditions for s in SUBJECTS}
        if keys != expected:
            failures.append(
                f"class_loso_metrics.csv grid mismatch: expected 63 unique rows, observed {len(keys)}"
            )
        if observed.duplicated(["geometry", "representation", "target_subject"]).any():
            failures.append("class_loso_metrics.csv has duplicate logical rows")

    runhalf = _frame(tables, "subject_runhalf_probe.csv")
    if not runhalf.empty and {"geometry", "representation", "split"} <= set(runhalf):
        keys = {
            (_canon_geometry(row.geometry), _canon_representation(row.representation), str(row.split))
            for row in runhalf[["geometry", "representation", "split"]].itertuples(index=False)
        }
        expected = {
            ("AIRM", representation, split)
            for representation in ("PATH_D10", "BAG_CANON_D10", "SCALARS_11")
            for split in ("A_TO_B", "B_TO_A")
        }
        if keys != expected:
            failures.append(
                f"subject_runhalf_probe.csv grid mismatch: expected {sorted(expected)}, observed {sorted(keys)}"
            )

    for name, expected_conditions in (
        ("order_shuffle_subject_metrics.csv", (("AIRM", "PATH_D10"), ("LE", "PATH_D10"))),
        ("label_null_subject_metrics.csv", (("AIRM", "PATH_D10"), ("AIRM", "BAG_CANON_D10"))),
    ):
        frame = _frame(tables, name)
        if frame.empty or not {"geometry", "representation", "replicate", "target_subject"} <= set(frame):
            continue
        keys = {
            (
                _canon_geometry(row.geometry), _canon_representation(row.representation),
                int(row.replicate), int(row.target_subject),
            )
            for row in frame[["geometry", "representation", "replicate", "target_subject"]].itertuples(index=False)
            if _finite_number(row.replicate) and _finite_number(row.target_subject)
        }
        expected = {
            (geometry, representation, replicate, subject)
            for geometry, representation in expected_conditions
            for replicate in range(1, 200)
            for subject in SUBJECTS
        }
        if keys != expected:
            failures.append(f"{name} grid mismatch: expected {len(expected)} rows, observed {len(keys)}")
        if frame.duplicated(["geometry", "representation", "replicate", "target_subject"]).any():
            failures.append(f"{name} has duplicate logical rows")

    for name, expected in (
        ("order_shuffle_group_metrics.csv", {("AIRM", "PATH_D10"), ("LE", "PATH_D10")}),
        ("label_null_group_metrics.csv", {("AIRM", "PATH_D10"), ("AIRM", "BAG_CANON_D10")}),
    ):
        frame = _frame(tables, name)
        if frame.empty or not {"geometry", "representation"} <= set(frame):
            continue
        keys = {
            (_canon_geometry(row.geometry), _canon_representation(row.representation))
            for row in frame[["geometry", "representation"]].itertuples(index=False)
        }
        if keys != expected or len(frame) != len(expected):
            failures.append(f"{name} grid mismatch: expected {sorted(expected)}, observed {sorted(keys)}")

    for name, expected_representation in (
        ("local_barycenter_mdm.csv", "LOCAL_BARYCENTER"),
        ("whole_context_mdm.csv", "WHOLE-1000"),
    ):
        frame = _frame(tables, name)
        if not frame.empty and "target_subject" in frame:
            subjects = tuple(sorted(pd.to_numeric(frame["target_subject"], errors="coerce").dropna().astype(int).tolist()))
            if subjects != SUBJECTS or len(frame) != 9:
                failures.append(f"{name} must contain exactly one row for subjects 1..9")
            if "representation" in frame and set(frame["representation"].astype(str)) != {
                expected_representation
            }:
                failures.append(
                    f"{name}.representation must be exactly {expected_representation}"
                )
    return failures


def _validate_null_group_against_npz(
    kind: str,
    frame: pd.DataFrame,
    payload: object,
) -> list[str]:
    arrays = _as_npz_mapping(payload)
    if arrays is None or frame.empty:
        return []
    failures: list[str] = []
    for row in frame.itertuples(index=False):
        geometry = _canon_geometry(getattr(row, "geometry"))
        representation = _canon_representation(getattr(row, "representation"))
        key = f"{geometry.lower()}__{representation.lower()}__median_subject_ba"
        if key not in arrays:
            failures.append(f"{kind} group row has no NPZ statistic key {key}")
            continue
        values = np.asarray(arrays[key], dtype=float)
        observed = float(getattr(row, "observed_median_subject_ba"))
        expected = {
            "null_replicates": 199,
            "null_median": float(np.median(values)),
            "null_mean": float(np.mean(values)),
            "null_sd_ddof1": float(np.std(values, ddof=1)),
            "null_min": float(np.min(values)),
            "null_max": float(np.max(values)),
            "effect": observed - float(np.median(values)),
            "exceedance_count": int(np.count_nonzero(values >= observed)),
            "p_value": (1 + int(np.count_nonzero(values >= observed))) / 200.0,
        }
        for column, target in expected.items():
            value = getattr(row, column)
            if column in {"null_replicates", "exceedance_count"}:
                if not _finite_number(value) or int(value) != int(target):
                    failures.append(f"{kind} {geometry}/{representation}.{column} mismatch")
            elif not _finite_number(value) or not math.isclose(
                float(value), float(target), rel_tol=1e-12, abs_tol=1e-12
            ):
                failures.append(f"{kind} {geometry}/{representation}.{column} mismatch")
    return failures


def validate_reporting_inputs(
    tables: Mapping[str, pd.DataFrame],
    null_artifacts: Mapping[str, object],
    geometry_gate: object = None,
    *,
    protocol_version: str = PROTOCOL_VERSION,
    protocol_sha256: str,
    config_sha256: str,
    strict_counts: bool = True,
) -> ValidationResult:
    """Validate schemas, gates, full grids, provenance, and exact null streams.

    Contract problems are returned as frozen failure details so the report can
    still be emitted as UNASSESSED.  A non-DataFrame value is an API misuse and
    raises :class:`ReportingContractError`.
    """

    numerical: list[str] = []
    technical: list[str] = []
    warnings: list[str] = []
    timestamps: list[str] = []

    numerical.extend(
        validate_geometry_gate(
            geometry_gate,
            protocol_version=protocol_version,
            protocol_sha256=protocol_sha256,
            config_sha256=config_sha256,
        )
    )

    for name in PREREQUISITE_FILENAMES:
        frame = _frame(tables, name)
        if frame.empty:
            technical.append(f"missing or empty required table: {name}")
            continue
        missing = [column for column in (*COMMON_COLUMNS, *TABLE_REQUIRED_COLUMNS[name]) if column not in frame]
        if missing:
            technical.append(f"{name}: missing columns {missing}")
            continue
        technical.extend(
            _common_provenance_failures(
                name, frame, protocol_version=protocol_version,
                protocol_sha256=protocol_sha256, config_sha256=config_sha256,
            )
        )
        timestamps.extend(frame["generated_at_utc"].astype(str).tolist())

    hard_names = (
        "dataset_contract.csv", "covariance_sanity.csv",
        "trajectory_geometry_correctness.csv", "trial_airm_path_features.csv",
        "trial_le_path_features.csv", "airm_path_d10.csv", "airm_bag_canon_d10.csv",
        "airm_bag_sorted_d10.csv", "le_path_d10.csv", "le_bag_canon_d10.csv",
        "scalar_factor_decomposition.csv",
    )
    for name in hard_names:
        frame = _frame(tables, name)
        if frame.empty or any(column not in frame for column in TABLE_REQUIRED_COLUMNS[name]):
            continue
        for failure in _status_failures(frame, name):
            numerical.append(failure)
        if "required" in frame and "passed" in frame:
            required = frame["required"].map(_bool_value) == True  # noqa: E712
            passed = frame["passed"].map(_bool_value) == True  # noqa: E712
            for index in frame.index[required & ~passed]:
                numerical.append(_failure_detail(frame, index, f"{name} required gate failed"))

    if strict_counts:
        expected_counts = {
            "covariance_sanity.csv": 2592 * 5,
            "trial_airm_path_features.csv": 2592,
            "trial_le_path_features.csv": 2592,
            "airm_path_d10.csv": 2592,
            "airm_bag_canon_d10.csv": 2592,
            "airm_bag_sorted_d10.csv": 2592,
            "le_path_d10.csv": 2592,
            "le_bag_canon_d10.csv": 2592,
            "scalar_factor_decomposition.csv": 22,
        }
        for name, count in expected_counts.items():
            frame = _frame(tables, name)
            if not frame.empty and len(frame) != count:
                numerical.append(f"{name}: expected {count} rows, observed {len(frame)}")

    covariance = _frame(tables, "covariance_sanity.csv")
    if not covariance.empty and set(TABLE_REQUIRED_COLUMNS["covariance_sanity.csv"]) <= set(covariance):
        for index, row in covariance.iterrows():
            invalid = (
                _bool_value(row["has_nan"]) is not False
                or _bool_value(row["has_inf"]) is not False
                or _bool_value(row["is_spd"]) is not True
                or not _finite_number(row["symmetry_relative_error"])
                or float(row["symmetry_relative_error"]) > 1e-12
                or not _finite_number(row["min_eigenvalue"])
                or float(row["min_eigenvalue"]) <= 0.0
                or not _finite_number(row["condition_number"])
                or float(row["condition_number"]) > 1e12
            )
            if invalid:
                numerical.append(_failure_detail(covariance, index, "covariance SPD usability gate failed"))

    for name in ("trial_airm_path_features.csv", "trial_le_path_features.csv"):
        frame = _frame(tables, name)
        if frame.empty or not set(TABLE_REQUIRED_COLUMNS[name]) <= set(frame):
            continue
        numeric = (
            "s1", "s2", "s3", "s4", "total_path_length", "endpoint_distance",
            "efficiency", "excess", "theta2", "theta3", "theta4", "mean_turn",
            "max_turn", "dev2", "dev3", "dev4", "mean_geodesic_deviation",
            "max_geodesic_deviation", "frechet_variance", "frechet_radius_mean", "diameter",
        )
        values = _numeric_values(frame, numeric)
        bad = ~np.isfinite(values).all(axis=1)
        degenerate = frame["degenerate"].map(_bool_value) != False  # noqa: E712
        for index in frame.index[bad | degenerate.to_numpy()]:
            numerical.append(_failure_detail(frame, index, f"{name} non-finite or degenerate trajectory"))

    for name, numeric in (
        ("airm_path_d10.csv", ("d12", "d13", "d14", "d15", "d23", "d24", "d25", "d34", "d35", "d45")),
        ("airm_bag_canon_d10.csv", tuple(f"bag{i:02d}" for i in range(1, 11))),
        ("airm_bag_sorted_d10.csv", tuple(f"sorted{i:02d}" for i in range(1, 11))),
        ("le_path_d10.csv", ("d12", "d13", "d14", "d15", "d23", "d24", "d25", "d34", "d35", "d45")),
        ("le_bag_canon_d10.csv", tuple(f"bag{i:02d}" for i in range(1, 11))),
    ):
        frame = _frame(tables, name)
        if not frame.empty and set(numeric) <= set(frame):
            values = _numeric_values(frame, numeric)
            if not np.isfinite(values).all():
                numerical.append(f"{name}: non-finite representation value")

    scalar = _frame(tables, "scalar_factor_decomposition.csv")
    if not scalar.empty and set(TABLE_REQUIRED_COLUMNS["scalar_factor_decomposition.csv"]) <= set(scalar):
        expected = {(g, s) for g in ("AIRM", "LE") for s in SCALARS_11}
        observed = {
            (_canon_geometry(row.geometry), str(row.scalar))
            for row in scalar[["geometry", "scalar"]].itertuples(index=False)
        }
        if observed != expected or len(scalar) != 22:
            numerical.append("scalar_factor_decomposition.csv: exact AIRM/LE × 11-scalar grid missing")
        for index, row in scalar.iterrows():
            if (
                _bool_value(row["degenerate"]) is not False
                or not _finite_number(row["ss_reconstruction_relative_error"])
                or float(row["ss_reconstruction_relative_error"]) > 1e-10
                or _bool_value(row["uses_p_value"]) is not False
            ):
                numerical.append(_failure_detail(scalar, index, "scalar SS interpretation gate failed"))

    scientific_names = (
        "class_loso_metrics.csv", "subject_runhalf_probe.csv",
        "order_shuffle_subject_metrics.csv", "order_shuffle_group_metrics.csv",
        "label_null_subject_metrics.csv", "label_null_group_metrics.csv",
        "local_barycenter_mdm.csv", "whole_context_mdm.csv", "airm_le_robustness.csv",
    )
    for name in scientific_names:
        frame = _frame(tables, name)
        if frame.empty or any(column not in frame for column in TABLE_REQUIRED_COLUMNS[name]):
            continue
        technical.extend(_status_failures(frame, name))

    technical.extend(_scientific_key_grid_failures(tables))

    for kind, json_name, npz_name in (
        ("order", "order_shuffle_seeds.json", "order_shuffle_group_stats.npz"),
        ("label", "label_permutation_seeds.json", "label_null_group_stats.npz"),
    ):
        technical.extend(_validate_null_manifest(kind, null_artifacts.get(json_name), protocol_version))
        technical.extend(_validate_null_npz(kind, null_artifacts.get(npz_name)))
        group_name = "order_shuffle_group_metrics.csv" if kind == "order" else "label_null_group_metrics.csv"
        group = _frame(tables, group_name)
        if not group.empty and set(TABLE_REQUIRED_COLUMNS[group_name]) <= set(group):
            technical.extend(_validate_null_group_against_npz(kind, group, null_artifacts.get(npz_name)))

    # Validate metric finiteness only on rows that claim PASS.  Failed rows must
    # retain NA metrics and are already a protocol/technical failure.
    metric_tables = (
        "class_loso_metrics.csv", "subject_runhalf_probe.csv",
        "order_shuffle_subject_metrics.csv", "label_null_subject_metrics.csv",
        "local_barycenter_mdm.csv", "whole_context_mdm.csv",
    )
    for name in metric_tables:
        frame = _frame(tables, name)
        if frame.empty or "status" not in frame:
            continue
        pass_rows = frame[frame["status"].map(_is_pass_status)]
        columns = [column for column in ("balanced_accuracy", "accuracy", "macro_f1") if column in pass_rows]
        if columns:
            values = _numeric_values(pass_rows, columns)
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                technical.append(f"{name}: PASS rows contain invalid metrics")

    # Stable order and de-duplication preserve the first, most specific message.
    numerical = list(dict.fromkeys(numerical))
    technical = list(dict.fromkeys(technical))
    warnings = list(dict.fromkeys(warnings))
    generated = min(timestamps) if timestamps else "UNAVAILABLE"
    return ValidationResult(
        numerical_failures=tuple(numerical),
        technical_failures=tuple(technical),
        warnings=tuple(warnings),
        generated_at_utc=generated,
        protocol_version=str(protocol_version),
        protocol_sha256=str(protocol_sha256),
        config_sha256=str(config_sha256),
    )


def _group_row(frame: pd.DataFrame, geometry: str, representation: str) -> pd.Series | None:
    if frame.empty or not {"geometry", "representation"} <= set(frame):
        return None
    mask = frame["geometry"].map(_canon_geometry).eq(geometry) & frame["representation"].map(_canon_representation).eq(representation)
    selected = frame.loc[mask]
    if len(selected) != 1:
        return None
    return selected.iloc[0]


def _path_minus_bag(class_loso: pd.DataFrame, geometry: str = "AIRM") -> float:
    if class_loso.empty or not {"geometry", "representation", "target_subject", "balanced_accuracy"} <= set(class_loso):
        return float("nan")
    frame = class_loso.copy()
    frame["geometry_c"] = frame["geometry"].map(_canon_geometry)
    frame["representation_c"] = frame["representation"].map(_canon_representation)
    frame = frame[(frame["geometry_c"] == geometry) & frame["representation_c"].isin(["PATH_D10", "BAG_CANON_D10"])]
    if len(frame) != 18:
        return float("nan")
    pivot = frame.pivot(index="target_subject", columns="representation_c", values="balanced_accuracy")
    if set(pivot.columns) != {"PATH_D10", "BAG_CANON_D10"} or len(pivot) != 9:
        return float("nan")
    return float(np.median(pivot["PATH_D10"].to_numpy(dtype=float) - pivot["BAG_CANON_D10"].to_numpy(dtype=float)))


def descriptive_robustness_label(
    tables: Mapping[str, pd.DataFrame],
) -> tuple[str | None, bool | None, bool | None]:
    """Label the repeated AIRM/LE order-evidence pattern without affecting the verdict."""

    order = _frame(tables, "order_shuffle_group_metrics.csv")
    class_loso = _frame(tables, "class_loso_metrics.csv")
    supported: dict[str, bool] = {}
    for geometry in ("AIRM", "LE"):
        row = _group_row(order, geometry, "PATH_D10")
        path_minus_bag = _path_minus_bag(class_loso, geometry)
        if (
            row is None
            or not _finite_number(row.get("effect"))
            or not _finite_number(row.get("p_value"))
            or not _finite_number(path_minus_bag)
        ):
            return None, None, None
        supported[geometry] = bool(
            float(row["effect"]) > 0.0
            and float(row["p_value"]) <= 0.05
            and path_minus_bag > 0.0
        )

    airm_supported = supported["AIRM"]
    le_supported = supported["LE"]
    if airm_supported == le_supported:
        label = ROBUSTNESS_CONSISTENT
    elif airm_supported:
        label = ROBUSTNESS_AIRM_SPECIFIC
    else:
        label = ROBUSTNESS_DISCORDANT
    return label, airm_supported, le_supported


def compute_terminal_verdict(
    *,
    effect_label_path: float = float("nan"),
    p_label_path: float = float("nan"),
    effect_label_bag: float = float("nan"),
    p_label_bag: float = float("nan"),
    effect_order_path: float = float("nan"),
    p_order_path: float = float("nan"),
    median_subject_path_minus_bag: float = float("nan"),
    numerical_failures: Sequence[str] = (),
    technical_failures: Sequence[str] = (),
) -> FrozenVerdict:
    """Apply the frozen H_PATH/H_BAG/H_ORDER and terminal-decision rules."""

    operands = {
        "effect_label_PATH": float(effect_label_path),
        "p_label_PATH": float(p_label_path),
        "effect_label_BAG": float(effect_label_bag),
        "p_label_BAG": float(p_label_bag),
        "effect_order_PATH": float(effect_order_path),
        "p_order_PATH": float(p_order_path),
        "median_subject_PATH_minus_BAG": float(median_subject_path_minus_bag),
    }
    numerical_tuple = tuple(str(value) for value in numerical_failures)
    technical_tuple = tuple(str(value) for value in technical_failures)
    finite = all(np.isfinite(value) for value in operands.values())
    if not finite and not numerical_tuple:
        technical_tuple = (*technical_tuple, "one or more frozen hypothesis operands are unavailable")

    if numerical_tuple:
        verdict = VERDICT_UNASSESSED
        failure_status = FAILURE_NUMERICAL
        hypotheses: tuple[None, None, None] = (None, None, None)
    elif technical_tuple:
        verdict = VERDICT_UNASSESSED
        failure_status = FAILURE_TECHNICAL
        hypotheses = (None, None, None)
    else:
        h_path = operands["effect_label_PATH"] > 0.0 and operands["p_label_PATH"] <= 0.05
        h_bag = operands["effect_label_BAG"] > 0.0 and operands["p_label_BAG"] <= 0.05
        h_order = (
            operands["effect_order_PATH"] > 0.0
            and operands["p_order_PATH"] <= 0.05
            and operands["median_subject_PATH_minus_BAG"] > 0.0
        )
        hypotheses = (h_path, h_bag, h_order)
        failure_status = ""
        if h_path and h_order:
            verdict = VERDICT_GO_ORDER
        elif h_bag and not h_order:
            verdict = VERDICT_GO_BAG
        elif not h_path and not h_bag:
            verdict = VERDICT_STOP
        else:
            verdict = VERDICT_MIXED

    h_path, h_bag, h_order = hypotheses
    operand_pass: dict[str, bool | None] = {
        "effect_label_PATH>0": None if h_path is None else operands["effect_label_PATH"] > 0.0,
        "p_label_PATH<=0.05": None if h_path is None else operands["p_label_PATH"] <= 0.05,
        "effect_label_BAG>0": None if h_bag is None else operands["effect_label_BAG"] > 0.0,
        "p_label_BAG<=0.05": None if h_bag is None else operands["p_label_BAG"] <= 0.05,
        "effect_order_PATH>0": None if h_order is None else operands["effect_order_PATH"] > 0.0,
        "p_order_PATH<=0.05": None if h_order is None else operands["p_order_PATH"] <= 0.05,
        "median_subject_PATH_minus_BAG>0": None if h_order is None else operands["median_subject_PATH_minus_BAG"] > 0.0,
    }
    if verdict == VERDICT_UNASSESSED:
        next_experiment = "Repair the recorded failure and rerun the unchanged Trajectory Anatomy v0 protocol once."
    elif verdict == VERDICT_GO_ORDER:
        next_experiment = "Conduct one rigorous literature/novelty review and representation-study design for SPD trajectories."
    elif verdict == VERDICT_GO_BAG:
        next_experiment = "Conduct one rigorous trial-as-SPD-distribution comparison."
    elif verdict == VERDICT_MIXED:
        next_experiment = "Run one preregistered diagnostic designed solely to explain the PATH/BAG/order contradiction."
    else:
        next_experiment = "Return to the frozen Conditional-Geometry Anatomy preregistration."
    return FrozenVerdict(
        verdict=verdict,
        failure_status=failure_status,
        h_path_class=h_path,
        h_bag_class=h_bag,
        h_order=h_order,
        operands=operands,
        operand_pass=operand_pass,
        numerical_failures=numerical_tuple,
        technical_failures=technical_tuple,
        next_experiment=next_experiment,
    )


def compute_frozen_verdicts(
    tables: Mapping[str, pd.DataFrame],
    validation: ValidationResult,
) -> FrozenVerdict:
    """Extract full-precision preregistered operands and evaluate the verdict."""

    label = _frame(tables, "label_null_group_metrics.csv")
    order = _frame(tables, "order_shuffle_group_metrics.csv")
    class_loso = _frame(tables, "class_loso_metrics.csv")
    path_label = _group_row(label, "AIRM", "PATH_D10")
    bag_label = _group_row(label, "AIRM", "BAG_CANON_D10")
    path_order = _group_row(order, "AIRM", "PATH_D10")
    path_minus_bag = _path_minus_bag(class_loso)
    extra_technical = list(validation.technical_failures)
    if path_order is not None and _finite_number(path_order.get("median_subject_path_minus_bag")):
        recorded = float(path_order["median_subject_path_minus_bag"])
        if _finite_number(path_minus_bag) and not math.isclose(recorded, path_minus_bag, rel_tol=1e-12, abs_tol=1e-12):
            extra_technical.append(
                "order_shuffle_group_metrics.csv median_subject_path_minus_bag disagrees with class_loso_metrics.csv"
            )
        path_minus_bag = recorded

    def value(row: pd.Series | None, column: str) -> float:
        if row is None or column not in row or not _finite_number(row[column]):
            return float("nan")
        return float(row[column])

    return compute_terminal_verdict(
        effect_label_path=value(path_label, "effect"),
        p_label_path=value(path_label, "p_value"),
        effect_label_bag=value(bag_label, "effect"),
        p_label_bag=value(bag_label, "p_value"),
        effect_order_path=value(path_order, "effect"),
        p_order_path=value(path_order, "p_value"),
        median_subject_path_minus_bag=path_minus_bag,
        numerical_failures=validation.numerical_failures,
        technical_failures=extra_technical,
    )


def _summary_row(
    provenance: Mapping[str, object],
    *,
    row_type: str,
    hypothesis: str = "",
    operand: str = "",
    formula: str = "",
    value: object = np.nan,
    threshold: object = np.nan,
    comparator: str = "",
    pass_flag: object = pd.NA,
    verdict: str = "",
    source_table: str = "",
    failure_type: str = "",
    failure_detail: str = "",
    interpretation: str = "",
) -> dict[str, object]:
    return {
        **provenance,
        "row_type": row_type,
        "hypothesis": hypothesis,
        "operand": operand,
        "formula": formula,
        "value": value,
        "threshold": threshold,
        "comparator": comparator,
        "pass_flag": pass_flag,
        "verdict": verdict,
        "source_table": source_table,
        "failure_type": failure_type,
        "failure_detail": failure_detail,
        "interpretation": interpretation,
    }


def build_summary(
    tables: Mapping[str, pd.DataFrame],
    validation: ValidationResult,
    verdict: FrozenVerdict,
) -> pd.DataFrame:
    """Build the exact 21st table without recomputing any upstream statistic."""

    status = verdict.failure_status or "PASS"
    provenance = {
        "protocol_version": validation.protocol_version,
        "protocol_sha256": validation.protocol_sha256,
        "config_sha256": validation.config_sha256,
        "seed": MASTER_SEED,
        "session": SESSION,
        "generated_at_utc": validation.generated_at_utc,
        "status": status,
    }
    rows: list[dict[str, object]] = []
    gate_failures = tuple(
        detail
        for detail in verdict.numerical_failures
        if GEOMETRY_GATE_FILENAME in detail
    )
    rows.append(
        _summary_row(
            provenance,
            row_type="gate",
            operand="trajectory_geometry_gate",
            formula=(
                "gate_passed AND scientific_classification_allowed AND status==PASS "
                "AND exact provenance/counts AND all required_failure_counts==0"
            ),
            value=not gate_failures,
            comparator="all frozen gate assertions true",
            pass_flag=not gate_failures,
            source_table=GEOMETRY_GATE_FILENAME,
            failure_type=FAILURE_NUMERICAL if gate_failures else "",
            failure_detail=" | ".join(gate_failures),
            interpretation=(
                "Stage-20 numerical/data gate admitted scientific reporting."
                if not gate_failures
                else "Stage-20 numerical/data gate prohibits a scientific verdict."
            ),
        )
    )
    operand_specs = (
        ("H_PATH_CLASS", "effect_label_PATH", "observed_median_BA - median(null)", 0.0, ">", "label_null_group_metrics.csv"),
        ("H_PATH_CLASS", "p_label_PATH", "(1 + count(null >= observed)) / 200", 0.05, "<=", "label_null_group_metrics.csv"),
        ("H_BAG_CLASS", "effect_label_BAG", "observed_median_BA - median(null)", 0.0, ">", "label_null_group_metrics.csv"),
        ("H_BAG_CLASS", "p_label_BAG", "(1 + count(null >= observed)) / 200", 0.05, "<=", "label_null_group_metrics.csv"),
        ("H_ORDER", "effect_order_PATH", "observed_median_BA - median(order null)", 0.0, ">", "order_shuffle_group_metrics.csv"),
        ("H_ORDER", "p_order_PATH", "(1 + count(order null >= observed)) / 200", 0.05, "<=", "order_shuffle_group_metrics.csv"),
        ("H_ORDER", "median_subject_PATH_minus_BAG", "median_s(BA_PATH,s - BA_BAG,s)", 0.0, ">", "class_loso_metrics.csv"),
    )
    key_map = {
        "effect_label_PATH": "effect_label_PATH>0",
        "p_label_PATH": "p_label_PATH<=0.05",
        "effect_label_BAG": "effect_label_BAG>0",
        "p_label_BAG": "p_label_BAG<=0.05",
        "effect_order_PATH": "effect_order_PATH>0",
        "p_order_PATH": "p_order_PATH<=0.05",
        "median_subject_PATH_minus_BAG": "median_subject_PATH_minus_BAG>0",
    }
    for hypothesis, operand, formula, threshold, comparator, source in operand_specs:
        rows.append(
            _summary_row(
                provenance, row_type="operand", hypothesis=hypothesis, operand=operand,
                formula=formula, value=verdict.operands[operand], threshold=threshold,
                comparator=comparator, pass_flag=verdict.operand_pass[key_map[operand]],
                source_table=source,
            )
        )
    for hypothesis, passed, formula in (
        ("H_PATH_CLASS", verdict.h_path_class, "effect_label_PATH>0 AND p_label_PATH<=0.05"),
        ("H_BAG_CLASS", verdict.h_bag_class, "effect_label_BAG>0 AND p_label_BAG<=0.05"),
        ("H_ORDER", verdict.h_order, "effect_order_PATH>0 AND p_order_PATH<=0.05 AND median_subject_PATH_minus_BAG>0"),
    ):
        rows.append(
            _summary_row(
                provenance, row_type="hypothesis", hypothesis=hypothesis, formula=formula,
                value=passed if passed is not None else np.nan, comparator="exact frozen conjunction",
                pass_flag=passed if passed is not None else pd.NA,
            )
        )
    for detail in verdict.numerical_failures:
        rows.append(
            _summary_row(
                provenance, row_type="failure", failure_type=FAILURE_NUMERICAL,
                failure_detail=detail, interpretation="Scientific verdict prohibited.",
            )
        )
    for detail in verdict.technical_failures:
        rows.append(
            _summary_row(
                provenance, row_type="failure", failure_type=FAILURE_TECHNICAL,
                failure_detail=detail, interpretation="Available-case substitution prohibited.",
            )
        )
    robustness_label, airm_order_supported, le_order_supported = descriptive_robustness_label(tables)
    if robustness_label is not None:
        rows.append(
            _summary_row(
                provenance,
                row_type="robustness_label",
                operand="repeated_order_evidence_pattern",
                formula="same fixed order-evidence conjunction evaluated for AIRM and LE",
                value=robustness_label,
                comparator="descriptive secondary label only",
                pass_flag=pd.NA,
                source_table="order_shuffle_group_metrics.csv | class_loso_metrics.csv",
                interpretation=(
                    f"AIRM order evidence={airm_order_supported}; LE order evidence={le_order_supported}. "
                    "LE did not vote in the terminal verdict."
                ),
            )
        )
    rows.append(
        _summary_row(
            provenance, row_type="terminal_verdict", formula="Protocol Section 15.2 exact ordered rule",
            verdict=verdict.verdict, failure_type=verdict.failure_status,
            interpretation=_verdict_interpretation(verdict.verdict),
        )
    )
    rows.append(
        _summary_row(
            provenance, row_type="next_experiment", verdict=verdict.verdict,
            interpretation=verdict.next_experiment,
        )
    )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _verdict_interpretation(verdict: str) -> str:
    return {
        VERDICT_GO_ORDER: "AIRM PATH label signal and the preregistered order operand both passed.",
        VERDICT_GO_BAG: "AIRM unordered BAG label signal passed while the order operand did not.",
        VERDICT_MIXED: "The preregistered PATH, BAG, and order operands disagree.",
        VERDICT_STOP: "Neither AIRM PATH nor AIRM BAG passed its label-destruction criterion.",
        VERDICT_UNASSESSED: "No scientific conclusion is permitted because a required gate or grid failed.",
    }[verdict]


def _provenance_columns(validation: ValidationResult, status: str | None = None) -> dict[str, object]:
    resolved_status = status
    if resolved_status is None:
        resolved_status = "PASS" if validation.passed else "FAILED"
    return {
        "protocol_version": validation.protocol_version,
        "protocol_sha256": validation.protocol_sha256,
        "config_sha256": validation.config_sha256,
        "seed": MASTER_SEED,
        "session": SESSION,
        "generated_at_utc": validation.generated_at_utc,
        "status": resolved_status,
    }


def _prepend_provenance(frame: pd.DataFrame, validation: ValidationResult) -> pd.DataFrame:
    result = frame.copy()
    defaults = _provenance_columns(validation)
    for column in reversed(COMMON_COLUMNS):
        if column in result:
            result = result.drop(columns=[column])
        result.insert(0, column, defaults[column])
    return result


def _figure_1_source(tables: Mapping[str, pd.DataFrame], validation: ValidationResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    class_loso = _frame(tables, "class_loso_metrics.csv")
    if not class_loso.empty and {"geometry", "representation", "target_subject", "balanced_accuracy", "status"} <= set(class_loso):
        selected = class_loso[class_loso["geometry"].map(_canon_geometry).eq("AIRM")]
        selected = selected[selected["representation"].map(_canon_representation).isin(
            ["PATH_D10", "BAG_CANON_D10", "BAG_SORTED_D10"]
        )]
        for row in selected.itertuples(index=False):
            rows.append({
                "series": _canon_representation(row.representation),
                "target_subject": int(row.target_subject),
                "balanced_accuracy": row.balanced_accuracy,
                "line_style": "solid",
                "estimator_regime_confounded": False,
                "row_status": row.status,
                "warning_messages": getattr(row, "warning_messages", ""),
            })
    for name, series, dashed, confounded in (
        ("local_barycenter_mdm.csv", "LOCAL_BARYCENTER", False, False),
        ("whole_context_mdm.csv", "WHOLE-1000", True, True),
    ):
        frame = _frame(tables, name)
        if not frame.empty and {"target_subject", "balanced_accuracy", "status"} <= set(frame):
            for row in frame.itertuples(index=False):
                rows.append({
                    "series": series, "target_subject": int(row.target_subject),
                    "balanced_accuracy": row.balanced_accuracy,
                    "line_style": "dashed" if dashed else "solid",
                    "estimator_regime_confounded": confounded,
                    "row_status": row.status,
                    "warning_messages": getattr(row, "warning_messages", ""),
                })
    columns = (
        "series", "target_subject", "balanced_accuracy", "line_style",
        "estimator_regime_confounded", "row_status", "warning_messages",
    )
    return _prepend_provenance(pd.DataFrame(rows, columns=columns), validation)


def _null_figure_source(
    kind: str,
    tables: Mapping[str, pd.DataFrame],
    null_artifacts: Mapping[str, object],
    validation: ValidationResult,
) -> pd.DataFrame:
    group_name = "order_shuffle_group_metrics.csv" if kind == "order" else "label_null_group_metrics.csv"
    npz_name = "order_shuffle_group_stats.npz" if kind == "order" else "label_null_group_stats.npz"
    group = _frame(tables, group_name)
    arrays = _as_npz_mapping(null_artifacts.get(npz_name)) or {}
    rows: list[dict[str, object]] = []
    conditions = (
        (("AIRM", "PATH_D10"), ("LE", "PATH_D10"))
        if kind == "order"
        else (("AIRM", "PATH_D10"), ("AIRM", "BAG_CANON_D10"))
    )
    for geometry, representation in conditions:
        key = f"{geometry.lower()}__{representation.lower()}__median_subject_ba"
        values = arrays.get(key, np.full(199, np.nan))
        group_row = _group_row(group, geometry, representation)
        observed = group_row.get("observed_median_subject_ba", np.nan) if group_row is not None else np.nan
        null_median = group_row.get("null_median", np.nan) if group_row is not None else np.nan
        effect = group_row.get("effect", np.nan) if group_row is not None else np.nan
        p_value = group_row.get("p_value", np.nan) if group_row is not None else np.nan
        exceedance = group_row.get("exceedance_count", np.nan) if group_row is not None else np.nan
        row_status = group_row.get("status", "MISSING") if group_row is not None else "MISSING"
        for replicate, statistic in enumerate(np.asarray(values), start=1):
            rows.append({
                "geometry": geometry, "representation": representation,
                "replicate": replicate, "statistic": statistic,
                "observed_statistic": observed, "null_median": null_median,
                "effect": effect, "p_value": p_value, "exceedance_count": exceedance,
                "primary_operand": geometry == "AIRM" and representation == "PATH_D10",
                "row_status": row_status,
            })
    return _prepend_provenance(pd.DataFrame(rows), validation)


def _figure_4_source(tables: Mapping[str, pd.DataFrame], validation: ValidationResult) -> pd.DataFrame:
    frame = _frame(tables, "scalar_factor_decomposition.csv")
    rows: list[dict[str, object]] = []
    required = {
        "geometry", "scalar", "eta2_subject", "eta2_class", "eta2_interaction",
        "eta2_residual", "status",
    }
    if not frame.empty and required <= set(frame):
        for row in frame.itertuples(index=False):
            for component in ("subject", "class", "interaction", "residual"):
                rows.append({
                    "geometry": _canon_geometry(row.geometry), "scalar": str(row.scalar),
                    "component": component, "eta_squared": getattr(row, f"eta2_{component}"),
                    "row_status": getattr(row, "status", "MISSING"),
                })
    return _prepend_provenance(pd.DataFrame(rows), validation)


def _scalar_distribution_source(
    tables: Mapping[str, pd.DataFrame], validation: ValidationResult
) -> pd.DataFrame:
    frame = _frame(tables, "trial_airm_path_features.csv")
    rows: list[dict[str, object]] = []
    required = {
        "sample_index", "subject", "class_label", "total_path_length",
        "endpoint_distance", "efficiency", "status",
    }
    if not frame.empty and required <= set(frame):
        for row in frame.itertuples(index=False):
            for scalar in ("total_path_length", "endpoint_distance", "efficiency"):
                rows.append({
                    "sample_index": int(row.sample_index), "subject": int(row.subject),
                    "class_label": str(row.class_label), "scalar": scalar,
                    "value": getattr(row, scalar), "row_status": getattr(row, "status", "MISSING"),
                })
    return _prepend_provenance(pd.DataFrame(rows), validation)


def _figure_7_source(tables: Mapping[str, pd.DataFrame], validation: ValidationResult) -> pd.DataFrame:
    frame = _frame(tables, "subject_runhalf_probe.csv")
    rows: list[dict[str, object]] = []
    required = {
        "geometry", "representation", "split", "balanced_accuracy", "accuracy",
        "direction_average_ba", "direction_average_accuracy", "chance_level", "status",
    }
    if not frame.empty and required <= set(frame):
        for representation in ("PATH_D10", "BAG_CANON_D10", "SCALARS_11"):
            selected = frame[
                frame["geometry"].map(_canon_geometry).eq("AIRM")
                & frame["representation"].map(_canon_representation).eq(representation)
            ]
            for row in selected.itertuples(index=False):
                rows.append({
                    "representation": representation, "split": str(row.split),
                    "balanced_accuracy": row.balanced_accuracy,
                    "accuracy": row.accuracy, "chance_level": row.chance_level,
                    "row_status": row.status,
                })
            if len(selected) > 0:
                average_ba = pd.to_numeric(selected["direction_average_ba"], errors="coerce").dropna()
                average_accuracy = pd.to_numeric(selected["direction_average_accuracy"], errors="coerce").dropna()
                chance = pd.to_numeric(selected["chance_level"], errors="coerce").dropna()
                rows.append({
                    "representation": representation, "split": "AVERAGE",
                    "balanced_accuracy": average_ba.iloc[0] if len(average_ba) else np.nan,
                    "accuracy": average_accuracy.iloc[0] if len(average_accuracy) else np.nan,
                    "chance_level": chance.iloc[0] if len(chance) else 1.0 / 9.0,
                    "row_status": "PASS" if selected["status"].map(_is_pass_status).all() else "FAILED",
                })
    return _prepend_provenance(pd.DataFrame(rows), validation)


def _figure_8_source(tables: Mapping[str, pd.DataFrame], validation: ValidationResult) -> pd.DataFrame:
    class_loso = _frame(tables, "class_loso_metrics.csv")
    rows: list[dict[str, object]] = []
    required = {
        "geometry", "representation", "target_subject", "balanced_accuracy",
    }
    if not class_loso.empty and required <= set(class_loso):
        working = class_loso.copy()
        working["geometry_c"] = working["geometry"].map(_canon_geometry)
        working["representation_c"] = working["representation"].map(_canon_representation)
        for representation in ("PATH_D10", "BAG_CANON_D10", "SCALARS_11"):
            selected = working[working["representation_c"].eq(representation)]
            pivot = selected.pivot_table(index="target_subject", columns="geometry_c", values="balanced_accuracy", aggfunc="first")
            for subject, values in pivot.iterrows():
                if {"AIRM", "LE"} <= set(values.index):
                    rows.append({
                        "analysis": "observed_class_loso", "representation": representation,
                        "subject": int(subject), "airm_value": values["AIRM"],
                        "le_value": values["LE"], "paired_delta_le_minus_airm": values["LE"] - values["AIRM"],
                    })
    order = _frame(tables, "order_shuffle_group_metrics.csv")
    airm = _group_row(order, "AIRM", "PATH_D10")
    le = _group_row(order, "LE", "PATH_D10")
    if airm is not None and le is not None:
        for operand in ("observed_median_subject_ba", "null_median", "effect", "p_value"):
            rows.append({
                "analysis": f"order_{operand}", "representation": "PATH_D10",
                "subject": "GROUP", "airm_value": airm.get(operand, np.nan),
                "le_value": le.get(operand, np.nan),
                "paired_delta_le_minus_airm": float(le.get(operand, np.nan)) - float(airm.get(operand, np.nan)),
            })
    return _prepend_provenance(pd.DataFrame(rows), validation)


def build_figure_sources(
    tables: Mapping[str, pd.DataFrame],
    null_artifacts: Mapping[str, object],
    validation: ValidationResult,
) -> dict[str, pd.DataFrame]:
    """Return all displayed data for the exact eight frozen figures."""

    scalar_source = _scalar_distribution_source(tables, validation)
    return {
        FIGURE_STEMS[0]: _figure_1_source(tables, validation),
        FIGURE_STEMS[1]: _null_figure_source("order", tables, null_artifacts, validation),
        FIGURE_STEMS[2]: _null_figure_source("label", tables, null_artifacts, validation),
        FIGURE_STEMS[3]: _figure_4_source(tables, validation),
        FIGURE_STEMS[4]: scalar_source.copy(),
        FIGURE_STEMS[5]: scalar_source.copy(),
        FIGURE_STEMS[6]: _figure_7_source(tables, validation),
        FIGURE_STEMS[7]: _figure_8_source(tables, validation),
    }


def _annotate_empty(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_figure_1(ax: plt.Axes, frame: pd.DataFrame) -> None:
    colors = {
        "PATH_D10": "#0072B2", "BAG_CANON_D10": "#D55E00",
        "BAG_SORTED_D10": "#E69F00", "LOCAL_BARYCENTER": "#009E73",
        "WHOLE-1000": "#666666",
    }
    plotted = False
    for series in ("PATH_D10", "BAG_CANON_D10", "BAG_SORTED_D10", "LOCAL_BARYCENTER", "WHOLE-1000"):
        subset = frame[frame.get("series", pd.Series(dtype=str)).eq(series)].copy()
        if subset.empty:
            continue
        x = pd.to_numeric(subset["target_subject"], errors="coerce")
        y = pd.to_numeric(subset["balanced_accuracy"], errors="coerce")
        ok = np.isfinite(x) & np.isfinite(y) & subset["row_status"].map(_is_pass_status)
        style = "--" if series == "WHOLE-1000" else "-"
        label = "WHOLE-1000 (confounded context)" if series == "WHOLE-1000" else series
        if ok.any():
            ax.plot(x[ok], y[ok], marker="o", linestyle=style, color=colors[series], label=label)
            plotted = True
        failed = ~ok
        if failed.any():
            ax.scatter(x[failed], np.full(int(failed.sum()), 0.02), marker="x", color=colors[series])
    if not plotted:
        _annotate_empty(ax, "No complete PASS LOSO rows")
    else:
        ax.axhline(0.25, color="black", linewidth=0.8, linestyle=":", label="chance 1/4")
        ax.set_xticks(SUBJECTS)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Target subject")
        ax.set_ylabel("Balanced accuracy")
        ax.legend(fontsize=7, ncol=2)
    ax.set_title("Class LOSO balanced accuracy")


def _plot_null(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    plotted = False
    conditions = frame[[column for column in ("geometry", "representation") if column in frame]].drop_duplicates()
    for index, condition in enumerate(conditions.itertuples(index=False), start=0):
        geometry = getattr(condition, "geometry")
        representation = getattr(condition, "representation")
        subset = frame[(frame["geometry"] == geometry) & (frame["representation"] == representation)]
        values = pd.to_numeric(subset["statistic"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        values = np.sort(values)
        cumulative_probability = np.arange(1, len(values) + 1, dtype=float) / len(values)
        line = ax.step(
            values,
            cumulative_probability,
            where="post",
            linewidth=1.2,
            label=f"{geometry}/{representation} null ECDF",
        )[0]
        observed = pd.to_numeric(subset["observed_statistic"], errors="coerce").dropna()
        if len(observed):
            ax.axvline(
                float(observed.iloc[0]),
                color=line.get_color(),
                linewidth=1.3,
                linestyle="--",
                label=f"{geometry}/{representation} observed",
            )
        plotted = True
    if not plotted:
        _annotate_empty(ax, "Required null grid unavailable or failed")
    else:
        ax.set_xlabel("Median-subject balanced accuracy")
        ax.set_ylabel("Empirical cumulative probability")
        ax.set_ylim(0.0, 1.02)
        ax.legend(fontsize=7)
    ax.set_title(title)


def _plot_figure_4(ax: plt.Axes, frame: pd.DataFrame) -> None:
    if frame.empty or "eta_squared" not in frame:
        _annotate_empty(ax, "Scalar decomposition unavailable")
        return
    pivot = frame.pivot_table(index=["geometry", "scalar"], columns="component", values="eta_squared", aggfunc="first")
    pivot = pivot.reindex(pd.MultiIndex.from_product((("AIRM", "LE"), SCALARS_11), names=["geometry", "scalar"]))
    displayed_components = ("subject", "class", "interaction")
    values = pivot[list(displayed_components)].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        _annotate_empty(ax, "Scalar decomposition unavailable")
        return
    x = np.arange(len(pivot))
    width = 0.25
    colors = ("#0072B2", "#D55E00", "#CC79A7")
    for offset, (component, color) in enumerate(zip(displayed_components, colors)):
        y = np.nan_to_num(pivot[component].to_numpy(dtype=float), nan=0.0)
        label = "subject×class" if component == "interaction" else component
        ax.bar(x + (offset - 1) * width, y, width=width, label=label, color=color)
    ax.axvline(10.5, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n{s}" for g, s in pivot.index], rotation=90, fontsize=6)
    upper = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    ax.set_ylim(0.0, max(0.05, upper * 1.12))
    ax.set_ylabel("eta-squared")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Balanced scalar variance anatomy (all 11 scalars)")


def _plot_scalar_distributions(fig: plt.Figure, axes: np.ndarray, frame: pd.DataFrame, by: str) -> None:
    order = CLASS_ORDER if by == "class_label" else tuple(str(value) for value in SUBJECTS)
    for ax, scalar in zip(axes, ("total_path_length", "endpoint_distance", "efficiency")):
        subset = frame[frame.get("scalar", pd.Series(dtype=str)).eq(scalar)].copy()
        if subset.empty:
            _annotate_empty(ax, "No scalar rows")
            ax.set_title(scalar)
            continue
        subset[by] = subset[by].astype(str)
        data = [pd.to_numeric(subset.loc[subset[by].eq(str(group)), "value"], errors="coerce").dropna().to_numpy() for group in order]
        if not any(len(values) for values in data):
            _annotate_empty(ax, "No finite scalar rows")
        else:
            ax.boxplot(
                data,
                tick_labels=order,
                showfliers=True,
                flierprops={
                    "marker": ".",
                    "markersize": 2.0,
                    "markerfacecolor": "#666666",
                    "markeredgecolor": "#666666",
                    "alpha": 0.35,
                },
            )
            ax.tick_params(axis="x", rotation=35 if by == "class_label" else 0, labelsize=7)
            ax.set_ylabel("Intrinsic value")
        ax.set_title(scalar)
    fig.suptitle("AIRM intrinsic scalars by " + ("class" if by == "class_label" else "subject"))


def _plot_figure_7(ax: plt.Axes, frame: pd.DataFrame) -> None:
    representations = ("PATH_D10", "BAG_CANON_D10", "SCALARS_11")
    splits = ("A_TO_B", "B_TO_A", "AVERAGE")
    x = np.arange(len(representations))
    width = 0.24
    plotted = False
    for offset, split in enumerate(splits):
        values = []
        for representation in representations:
            selected = frame[(frame.get("representation", pd.Series(dtype=str)) == representation) & (frame.get("split", pd.Series(dtype=str)) == split)]
            value = pd.to_numeric(selected.get("balanced_accuracy", pd.Series(dtype=float)), errors="coerce").dropna()
            values.append(float(value.iloc[0]) if len(value) else np.nan)
        if np.isfinite(values).any():
            ax.bar(x + (offset - 1) * width, values, width=width, label=split)
            plotted = True
    if not plotted:
        _annotate_empty(ax, "Subject probe unavailable or failed")
    else:
        ax.axhline(1.0 / 9.0, color="black", linestyle=":", linewidth=1.0, label="chance 1/9")
        ax.set_xticks(x)
        ax.set_xticklabels(representations, rotation=15)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Subject-ID balanced accuracy")
        ax.legend(fontsize=7)
    ax.set_title("Run-half subject-information probe")


def _plot_figure_8(ax: plt.Axes, frame: pd.DataFrame) -> None:
    if frame.empty or not {"airm_value", "le_value"} <= set(frame):
        _annotate_empty(ax, "AIRM/LE paired operands unavailable")
        return
    observed = frame[frame.get("analysis", pd.Series(dtype=str)).astype(str).eq("observed_class_loso")].copy()
    x = pd.to_numeric(observed["airm_value"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(observed["le_value"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        _annotate_empty(ax, "AIRM/LE paired operands unavailable")
        return
    categories = observed.loc[ok, "representation"].astype(str)
    for representation in ("PATH_D10", "BAG_CANON_D10", "SCALARS_11"):
        mask = ok.copy()
        mask[ok] = categories.to_numpy() == representation
        if mask.any():
            ax.scatter(x[mask], y[mask], s=24, alpha=0.8, label=representation)
    lower = min(float(np.nanmin(x[ok])), float(np.nanmin(y[ok])))
    upper = max(float(np.nanmax(x[ok])), float(np.nanmax(y[ok])))
    if math.isclose(lower, upper):
        lower -= 0.01
        upper += 0.01
    ax.plot([lower, upper], [lower, upper], color="black", linestyle=":", linewidth=1.0)
    ax.set_xlabel("AIRM balanced accuracy")
    ax.set_ylabel("LE balanced accuracy")
    ax.legend(fontsize=6)
    order_rows = frame[frame.get("analysis", pd.Series(dtype=str)).astype(str).isin(["order_effect", "order_p_value"])]
    annotations: list[str] = []
    for analysis, label in (("order_effect", "Order effect"), ("order_p_value", "Order p")):
        selected = order_rows[order_rows["analysis"].eq(analysis)]
        if len(selected) == 1:
            annotations.append(
                f"{label}: AIRM={float(selected.iloc[0]['airm_value']):.3f}, "
                f"LE={float(selected.iloc[0]['le_value']):.3f}"
            )
    if annotations:
        ax.text(
            0.02,
            0.98,
            "\n".join(annotations),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#999999"},
        )
    ax.set_title("AIRM vs LE subject-wise class LOSO BA")


def _write_figure(fig: plt.Figure, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png, dpi=160, bbox_inches="tight",
        metadata={"Software": "EEG-SPD-Representation-Probe"},
    )
    fig.savefig(
        pdf, bbox_inches="tight",
        metadata={"Creator": "EEG-SPD-Representation-Probe", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def render_figures(
    figure_sources: Mapping[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Render exactly eight fixed PNG/PDF figures (never t-SNE)."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "figure.figsize": (8.2, 5.0), "font.size": 9, "axes.grid": True,
        "axes.axisbelow": True, "savefig.transparent": False,
    }):
        fig, ax = plt.subplots()
        _plot_figure_1(ax, figure_sources[FIGURE_STEMS[0]])
        fig.tight_layout()
        _write_figure(fig, figures_dir / f"{FIGURE_STEMS[0]}.png", figures_dir / f"{FIGURE_STEMS[0]}.pdf")

        fig, ax = plt.subplots()
        _plot_null(ax, figure_sources[FIGURE_STEMS[1]], "Order-shuffle falsification")
        fig.tight_layout()
        _write_figure(fig, figures_dir / f"{FIGURE_STEMS[1]}.png", figures_dir / f"{FIGURE_STEMS[1]}.pdf")

        fig, ax = plt.subplots()
        _plot_null(ax, figure_sources[FIGURE_STEMS[2]], "Within-subject×run label destruction")
        fig.tight_layout()
        _write_figure(fig, figures_dir / f"{FIGURE_STEMS[2]}.png", figures_dir / f"{FIGURE_STEMS[2]}.pdf")

        fig, ax = plt.subplots(figsize=(12.0, 5.5))
        _plot_figure_4(ax, figure_sources[FIGURE_STEMS[3]])
        fig.tight_layout()
        _write_figure(fig, figures_dir / f"{FIGURE_STEMS[3]}.png", figures_dir / f"{FIGURE_STEMS[3]}.pdf")

        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5))
        _plot_scalar_distributions(fig, axes, figure_sources[FIGURE_STEMS[4]], "class_label")
        fig.tight_layout()
        _write_figure(fig, figures_dir / f"{FIGURE_STEMS[4]}.png", figures_dir / f"{FIGURE_STEMS[4]}.pdf")

        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5))
        _plot_scalar_distributions(fig, axes, figure_sources[FIGURE_STEMS[5]], "subject")
        fig.tight_layout()
        _write_figure(fig, figures_dir / f"{FIGURE_STEMS[5]}.png", figures_dir / f"{FIGURE_STEMS[5]}.pdf")

        fig, ax = plt.subplots()
        _plot_figure_7(ax, figure_sources[FIGURE_STEMS[6]])
        fig.tight_layout()
        _write_figure(fig, figures_dir / f"{FIGURE_STEMS[6]}.png", figures_dir / f"{FIGURE_STEMS[6]}.pdf")

        fig, ax = plt.subplots()
        _plot_figure_8(ax, figure_sources[FIGURE_STEMS[7]])
        fig.tight_layout()
        _write_figure(fig, figures_dir / f"{FIGURE_STEMS[7]}.png", figures_dir / f"{FIGURE_STEMS[7]}.pdf")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    if frame.empty or any(column not in frame for column in columns):
        return "No complete rows available."
    display = frame[list(columns)].copy()
    for column in display:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: _fmt(value, digits))
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join((header, divider, *body))


def _class_summary(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _frame(tables, "class_loso_metrics.csv")
    if frame.empty or not {"geometry", "representation", "balanced_accuracy", "status"} <= set(frame):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (geometry, representation), group in frame.groupby(["geometry", "representation"], sort=True):
        passed = group[group["status"].map(_is_pass_status)]
        values = pd.to_numeric(passed["balanced_accuracy"], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append({
            "geometry": _canon_geometry(geometry), "representation": _canon_representation(representation),
            "pass_n": len(values), "required_n": 9,
            "mean_ba": np.mean(values) if len(values) else np.nan,
            "sd_ba_ddof1": np.std(values, ddof=1) if len(values) > 1 else np.nan,
            "median_ba": np.median(values) if len(values) else np.nan,
            "min_ba": np.min(values) if len(values) else np.nan,
            "max_ba": np.max(values) if len(values) else np.nan,
        })
    return pd.DataFrame(rows)


def _null_summary(tables: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    frame = _frame(tables, name)
    columns = [
        "geometry", "representation", "observed_median_subject_ba", "null_median",
        "effect", "p_value", "exceedance_count", "null_replicates",
    ]
    return frame[columns].copy() if not frame.empty and set(columns) <= set(frame) else pd.DataFrame()


def build_report(
    tables: Mapping[str, pd.DataFrame],
    validation: ValidationResult,
    verdict: FrozenVerdict,
) -> str:
    """Build the exact-title, exact-18-heading Markdown report."""

    class_summary = _class_summary(tables)
    order_summary = _null_summary(tables, "order_shuffle_group_metrics.csv")
    label_summary = _null_summary(tables, "label_null_group_metrics.csv")
    scalar = _frame(tables, "scalar_factor_decomposition.csv")
    subject_probe = _frame(tables, "subject_runhalf_probe.csv")
    local = _frame(tables, "local_barycenter_mdm.csv")
    whole = _frame(tables, "whole_context_mdm.csv")
    robustness = _frame(tables, "airm_le_robustness.csv")
    covariance = _frame(tables, "covariance_sanity.csv")
    correctness = _frame(tables, "trajectory_geometry_correctness.csv")

    hard_required = 0
    hard_passed = 0
    for frame in (_frame(tables, "dataset_contract.csv"), covariance, correctness):
        if not frame.empty and {"required", "passed"} <= set(frame):
            required = frame["required"].map(_bool_value) == True  # noqa: E712
            hard_required += int(required.sum())
            hard_passed += int((required & (frame["passed"].map(_bool_value) == True)).sum())  # noqa: E712

    failure_lines = [f"- {detail}" for detail in (*verdict.numerical_failures, *verdict.technical_failures)]
    failure_text = "\n".join(failure_lines) if failure_lines else "- None recorded."
    geometry_gate_passed = not any(
        GEOMETRY_GATE_FILENAME in detail for detail in verdict.numerical_failures
    )
    class_table = _markdown_table(
        class_summary,
        ("geometry", "representation", "pass_n", "required_n", "mean_ba", "sd_ba_ddof1", "median_ba", "min_ba", "max_ba"),
    )
    order_table = _markdown_table(
        order_summary,
        ("geometry", "representation", "observed_median_subject_ba", "null_median", "effect", "p_value", "exceedance_count", "null_replicates"),
    )
    label_table = _markdown_table(
        label_summary,
        ("geometry", "representation", "observed_median_subject_ba", "null_median", "effect", "p_value", "exceedance_count", "null_replicates"),
    )

    path_features = _frame(tables, "trial_airm_path_features.csv")
    scalar_descriptive: list[str] = []
    for name in ("total_path_length", "endpoint_distance", "efficiency"):
        if not path_features.empty and name in path_features:
            values = pd.to_numeric(path_features[name], errors="coerce").dropna().to_numpy(dtype=float)
            scalar_descriptive.append(
                f"- {name}: n={len(values)}, mean={_fmt(np.mean(values) if len(values) else np.nan)}, "
                f"SD={_fmt(np.std(values, ddof=1) if len(values)>1 else np.nan)}, "
                f"median={_fmt(np.median(values) if len(values) else np.nan)}."
            )
    scalar_text = "\n".join(scalar_descriptive) if scalar_descriptive else "- No complete intrinsic-scalar rows."

    subject_rows: list[dict[str, object]] = []
    subject_columns = {
        "representation", "split", "balanced_accuracy", "direction_average_ba",
    }
    if not subject_probe.empty and subject_columns <= set(subject_probe):
        for representation, group in subject_probe.groupby("representation", sort=True):
            subject_rows.append({
                "representation": _canon_representation(representation),
                "A_TO_B_ba": pd.to_numeric(group.loc[group["split"].astype(str).eq("A_TO_B"), "balanced_accuracy"], errors="coerce").mean(),
                "B_TO_A_ba": pd.to_numeric(group.loc[group["split"].astype(str).eq("B_TO_A"), "balanced_accuracy"], errors="coerce").mean(),
                "average_ba": pd.to_numeric(group["direction_average_ba"], errors="coerce").dropna().iloc[0] if pd.to_numeric(group["direction_average_ba"], errors="coerce").notna().any() else np.nan,
                "chance": 1.0 / 9.0,
            })
    subject_table = _markdown_table(pd.DataFrame(subject_rows), ("representation", "A_TO_B_ba", "B_TO_A_ba", "average_ba", "chance"))

    eta_table = pd.DataFrame()
    if not scalar.empty and {"geometry", "scalar", "eta2_subject", "eta2_class", "eta2_interaction", "eta2_residual"} <= set(scalar):
        eta_table = scalar[["geometry", "scalar", "eta2_subject", "eta2_class", "eta2_interaction", "eta2_residual"]]

    def mean_ba(frame: pd.DataFrame) -> tuple[int, float]:
        if frame.empty or not {"balanced_accuracy", "status"} <= set(frame):
            return 0, float("nan")
        values = pd.to_numeric(frame.loc[frame["status"].map(_is_pass_status), "balanced_accuracy"], errors="coerce").dropna()
        return len(values), float(values.mean()) if len(values) else float("nan")

    local_n, local_ba = mean_ba(local)
    whole_n, whole_ba = mean_ba(whole)
    disagreements = 0
    if not robustness.empty and "agreement_category" in robustness:
        agreement_mask = robustness["agreement_category"].astype(str).str.upper().isin(
            ["AGREE", "AGREEMENT", "SAME", "CONCORDANT"]
        )
        disagreements = int((~agreement_mask).sum())
    robustness_label, airm_order_supported, le_order_supported = descriptive_robustness_label(tables)
    if robustness_label is None:
        robustness_statement = "The descriptive AIRM/LE robustness label was unavailable because a required paired operand was incomplete."
    else:
        robustness_statement = (
            f"Descriptive robustness label for the repeated LOSO/order-evidence pattern: "
            f"**{robustness_label}** (AIRM order evidence={airm_order_supported}; "
            f"LE order evidence={le_order_supported}). This label does not extend to an LE "
            "label-destruction hypothesis and does not vote in the terminal verdict."
        )

    fitted_warning_rows = 0
    fitted_rows = 0
    for fitted_name in (
        "class_loso_metrics.csv", "subject_runhalf_probe.csv",
        "order_shuffle_subject_metrics.csv", "label_null_subject_metrics.csv",
        "local_barycenter_mdm.csv", "whole_context_mdm.csv",
    ):
        fitted = _frame(tables, fitted_name)
        if not fitted.empty:
            fitted_rows += len(fitted)
            if "convergence_warning" in fitted:
                fitted_warning_rows += int(
                    sum(_bool_value(value) is True for value in fitted["convergence_warning"])
                )

    if verdict.verdict == VERDICT_GO_ORDER:
        justified = (
            "Within this session-0 discovery set, AIRM PATH_D10 carried label information under the "
            "frozen label null and its ordering beat the frozen per-trial order shuffle while PATH exceeded BAG."
        )
    elif verdict.verdict == VERDICT_GO_BAG:
        justified = (
            "Within this session-0 discovery set, unordered AIRM BAG_CANON_D10 carried label information, "
            "while the frozen evidence did not support an additional temporal-order claim."
        )
    elif verdict.verdict == VERDICT_MIXED:
        justified = "Only the numerical operand values and their preregistered disagreement are justified; no single trajectory mechanism is established."
    elif verdict.verdict == VERDICT_STOP:
        justified = "The frozen null comparisons did not establish local PATH or BAG class-label signal in this session-0 pilot."
    else:
        justified = "Only the recorded data, gate diagnostics, and failures are justified; no scientific trajectory conclusion was assessed."

    lines = [
        REPORT_TITLE,
        "",
        "## 1. Scientific question",
        "",
        "Does the ordered five-state local covariance path contain class information beyond an unordered set of the same states, and how much subject structure remains? This is representation anatomy, not a classifier-development study.",
        "",
        "## 2. Why V1 did not test this question",
        "",
        "V1's WHOLE representation compressed each 1,000-sample trial to one covariance. Its WINDOW5 probe treated the five local covariances as independent views and averaged held-out class probabilities; it did not encode a trial as an unordered finite SPD metric space or as an ordered path. V1 therefore could not isolate temporal-order information, and the present analysis does not reinterpret V1 performance as trajectory evidence.",
        "",
        "## 3. Frozen protocol",
        "",
        f"Protocol {validation.protocol_version}; SHA-256 `{validation.protocol_sha256}`; config SHA-256 `{validation.config_sha256}`; seed {MASTER_SEED}. Only session `{SESSION}` was admissible: 9 subjects, 4 fixed classes, 2,592 trials, 22 EEG channels, 8–32 Hz, cue-relative 0–3.996 s, 250 Hz, 1,000 samples, OAS float64. Each trial was split into five ordered, non-overlapping 200-sample windows. AIRM was primary and LE secondary. No loading, clipping, tuning, result-selected embedding, or session `1test` access was permitted.",
        "",
        "## 4. Geometry correctness",
        "",
        f"Persisted `{GEOMETRY_GATE_FILENAME}` validation: `{'PASS' if geometry_gate_passed else 'FAILED'}`. Required recorded gate rows passed: {hard_passed}/{hard_required}. Covariance rows: {len(covariance)}/12,960. Geometry-correctness rows: {len(correctness)}. Terminal gate status: `{verdict.failure_status or 'PASS'}`.",
        "",
        failure_text,
        "",
        "## 5. Five-state AIRM geometry",
        "",
        f"AIRM trial-feature rows: {len(path_features)}/2,592. Each trial used the full pairwise five-state distance matrix, intrinsic path quantities, an AIRM local barycenter, and fixed-time endpoint-geodesic deviations d(C_j,G(j/4)); deviations were not minimized over the geodesic set.",
        "",
        f"[Scalar-by-class figure](../figures/{FIGURE_STEMS[4]}.png) and [scalar-by-subject figure](../figures/{FIGURE_STEMS[5]}.png).",
        "",
        "## 6. BAG vs PATH definition",
        "",
        "PATH_D10 retains the fixed chronological upper-triangle distance order. BAG_CANON_D10 is the lexicographically minimal upper triangle over all 120 state permutations and removes ordering; BAG_SORTED_D10 is secondary. BAG invariance is a hard gate, not an order-null classifier vote.",
        "",
        f"[Class LOSO figure](../figures/{FIGURE_STEMS[0]}.png).",
        "",
        "## 7. Intrinsic path quantities",
        "",
        *scalar_text.splitlines(),
        "",
        "The eleven-scalar probe was frozen before results. Individual steps, turns, and deviations are descriptive only and were not selected post hoc.",
        "",
        "## 8. Class LOSO results",
        "",
        "Balanced accuracy (BA) is primary. Summaries use only PASS rows descriptively and show the required denominator 9; an incomplete denominator never receives a verdict.",
        f"Across all fitted-condition tables, convergence warnings were recorded in {fitted_warning_rows}/{fitted_rows} rows; any such required row is FAILED.",
        "",
        class_table,
        "",
        "## 9. Order-shuffle falsification",
        "",
        "Each of 199 replicates independently applied a nonidentity S5 permutation to every trial. The statistic was median subject BA and the one-sided Monte Carlo p-value used the fixed plus-one rule.",
        "",
        order_table,
        "",
        f"[Order-null figure](../figures/{FIGURE_STEMS[1]}.png).",
        "",
        "## 10. Label-destruction null",
        "",
        "Labels were permuted within subject×run using identical permutations for AIRM PATH_D10 and BAG_CANON_D10. Features and identities remained fixed.",
        "",
        label_table,
        "",
        f"[Label-null figure](../figures/{FIGURE_STEMS[2]}.png).",
        "",
        "## 11. Subject-information results",
        "",
        "The subject-ID probe used train-only scaling and disjoint run halves; chance was exactly 1/9.",
        "",
        subject_table,
        "",
        f"[Subject-probe figure](../figures/{FIGURE_STEMS[6]}.png).",
        "",
        "## 12. Class vs subject vs interaction effects",
        "",
        "Balanced sums of squares were decomposed for all 11 frozen scalars without p-values. Every nondegenerate scalar required relative SS closure ≤1e-10.",
        "",
        _markdown_table(eta_table, ("geometry", "scalar", "eta2_subject", "eta2_class", "eta2_interaction", "eta2_residual")),
        "",
        f"[All-scalar eta-squared figure](../figures/{FIGURE_STEMS[3]}.png).",
        "",
        "## 13. LOCAL_BARYCENTER / WHOLE contextual controls",
        "",
        f"LOCAL_BARYCENTER MDM PASS denominator {local_n}/9, mean BA {_fmt(local_ba)}. WHOLE-1000 MDM PASS denominator {whole_n}/9, mean BA {_fmt(whole_ba)}. WHOLE uses 1,000 samples per covariance while each local covariance uses 200, so this is estimator-regime-confounded context and not an unconfounded method comparison.",
        "",
        "## 14. AIRM vs LE robustness",
        "",
        f"The robustness table contains {len(robustness)} paired rows; {disagreements} rows were not marked as agreement. LE is secondary and cannot rescue an AIRM failure or vote in the terminal verdict.",
        "",
        robustness_statement,
        "",
        f"[AIRM/LE robustness figure](../figures/{FIGURE_STEMS[7]}.png).",
        "",
        "## 15. Frozen verdict",
        "",
        f"Terminal verdict: **{verdict.verdict}**.",
        "",
        f"Failure status: **{verdict.failure_status or 'none'}**.",
        "",
        f"- H_PATH_CLASS: {verdict.h_path_class if verdict.h_path_class is not None else 'UNASSESSED'}; effect={_fmt(verdict.operands['effect_label_PATH'])}, p={_fmt(verdict.operands['p_label_PATH'])}.",
        f"- H_BAG_CLASS: {verdict.h_bag_class if verdict.h_bag_class is not None else 'UNASSESSED'}; effect={_fmt(verdict.operands['effect_label_BAG'])}, p={_fmt(verdict.operands['p_label_BAG'])}.",
        f"- H_ORDER: {verdict.h_order if verdict.h_order is not None else 'UNASSESSED'}; order effect={_fmt(verdict.operands['effect_order_PATH'])}, p={_fmt(verdict.operands['p_order_PATH'])}, median subject PATH−BAG={_fmt(verdict.operands['median_subject_PATH_minus_BAG'])}.",
        "",
        "The ordered decision rule in protocol Section 15.2 was applied to full-precision values. Displays were rounded only after the decision.",
        "",
        "## 16. What is actually justified",
        "",
        justified,
        "",
        "## 17. What is NOT justified",
        "",
        "This pilot does not justify conditional geometry or conditional alignment, domain adaptation, pseudo-labels, target-label-free conditional identifiability, a neural/sequence/trajectory model, a new distribution classifier, session-1 generalization, classifier/SOTA claims, post-hoc scalar selection, or an unconfounded LOCAL_BARYCENTER-versus-WHOLE comparison.",
        "",
        "## 18. Single recommended next step",
        "",
        verdict.next_experiment,
        "",
    ]
    text = "\n".join(lines)
    headings = tuple(line[3:] for line in text.splitlines() if line.startswith("## "))
    if headings != tuple(f"{index}. {heading}" for index, heading in enumerate(REPORT_HEADINGS, start=1)):
        raise ReportingContractError(f"report headings violate exact contract: {headings}")
    return text


def _canonical_scalar(value: object) -> object:
    if value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def stable_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash decision content while excluding generated_at_utc provenance."""

    columns = [column for column in frame.columns if column != "generated_at_utc"]
    rows = [[_canonical_scalar(value) for value in row] for row in frame[columns].itertuples(index=False, name=None)]
    payload = {"columns": columns, "rows": rows}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _atomic_text(text_value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".md", dir=path.parent, delete=False, encoding="utf-8", newline="\n") as handle:
        temporary = Path(handle.name)
        handle.write(text_value)
    os.replace(temporary, path)


def create_reporting_outputs(
    tables: Mapping[str, pd.DataFrame],
    null_artifacts: Mapping[str, object],
    geometry_gate: object = None,
    *,
    protocol_version: str = PROTOCOL_VERSION,
    protocol_sha256: str,
    config_sha256: str,
    tables_dir: Path,
    figures_dir: Path,
    report_path: Path,
    strict_counts: bool = True,
) -> ReportingArtifacts:
    """Validate inputs and write the complete frozen reporting artifact set."""

    figures_path = Path(figures_dir)
    allowed_figure_files = {
        f"{stem}{suffix}"
        for stem in FIGURE_STEMS
        for suffix in (".png", ".pdf", ".csv")
    }
    if figures_path.is_dir():
        unexpected = sorted(
            path.name
            for path in figures_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".pdf", ".csv"}
            and path.name not in allowed_figure_files
        )
        if unexpected:
            raise ReportingContractError(
                f"unexpected figure artifacts violate the exact eight-stem contract: {unexpected}"
            )

    validation = validate_reporting_inputs(
        tables, null_artifacts, geometry_gate, protocol_version=protocol_version,
        protocol_sha256=protocol_sha256, config_sha256=config_sha256,
        strict_counts=strict_counts,
    )
    verdict = compute_frozen_verdicts(tables, validation)
    summary = build_summary(tables, validation, verdict)
    figure_sources = build_figure_sources(tables, null_artifacts, validation)
    report_text = build_report(tables, validation, verdict)
    decision_sha256 = stable_frame_sha256(summary)

    _atomic_csv(summary, Path(tables_dir) / SUMMARY_FILENAME)
    for stem in FIGURE_STEMS:
        _atomic_csv(figure_sources[stem], figures_path / f"{stem}.csv")
    render_figures(figure_sources, figures_path)
    _atomic_text(report_text, Path(report_path))
    generated_figure_files = {
        path.name
        for path in figures_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".csv"}
    }
    if generated_figure_files != allowed_figure_files:
        raise ReportingContractError(
            "generated figure artifacts do not match the exact eight-stem × PNG/PDF/CSV contract"
        )
    return ReportingArtifacts(
        summary=summary,
        verdict=verdict,
        validation=validation,
        figure_sources=figure_sources,
        report_text=report_text,
        decision_sha256=decision_sha256,
    )
