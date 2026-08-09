"""Strict final reporting for Conditional-Geometry Anatomy v1.

The reporter consumes frozen result artifacts only.  It never loads EEG,
covariances, discovery caches, or confirmatory raw files, and it never refits a
mean or reruns a null.  Scientific decisions are recomputed from the saved
group summaries using the preregistered R -> S -> P rules and checked against
the producer's decision artifacts before any output is written.
"""

from __future__ import annotations

import hashlib
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
import yaml

from src.conditional_geometry_v1 import (
    airm_distance_matrix,
    airm_gram_matrices,
    le_distance_matrix,
    le_gram_matrix,
    svec,
)
from src.conditional_provenance_v1 import canonical_json_bytes
from src.conditional_statistics_v1 import (
    StageEvidence,
    evaluate_fixed_sequence,
    leave_one_subject_out_influence,
    le_robustness_label,
    subject_bootstrap_median,
    subject_bootstrap_paired_median_delta,
    terminal_airm_decision,
)


REPORT_TITLE = "# BNCI2014_001 Conditional Geometry Anatomy v1"
REPORT_HEADINGS = (
    "Scientific question",
    "Why this follows V1/V2/Trajectory v0",
    "Frozen protocol",
    "Data and numerical gates",
    "Exact AIRM objects D and G",
    "Discovery reliability",
    "Confirmatory reliability",
    "Same-subject vs unrelated reference",
    "Discovery cross-subject shared geometry",
    "Locked confirmatory shared geometry",
    "Oracle semantic-permutation identifiability",
    "D-chain",
    "G-chain",
    "LE robustness",
    "Terminal frozen decision",
    "What is actually justified",
    "What is NOT justified",
    "One next question only",
)
FIGURE_STEMS = (
    "figure_1_within_subject_reliability",
    "figure_2_reliability_label_null",
    "figure_3_same_vs_unrelated",
    "figure_4_shared_template_similarity",
    "figure_5_D_heatmaps",
    "figure_6_G_heatmaps",
    "figure_7_oracle_permutation_scores",
    "figure_8_oracle_rank_margin",
    "figure_9_subject_forest_influence",
    "figure_10_airm_le_stage_effects",
)
REQUIRED_ROOT_FILES = (
    "manifest.json",
    "git_provenance.json",
    "environment.json",
    "confirmatory_unlock.json",
)
REQUIRED_TABLES = (
    "dataset_contract.csv",
    "covariance_sanity.csv",
    "airm_mean_convergence.csv",
    "le_mean_correctness.csv",
    "centering_isometry_gate.csv",
    "orthogonal_gauge_gate.csv",
    "degenerate_geometry_audit.csv",
    "D_shape_vectors.csv",
    "G_shape_vectors.csv",
    "absolute_geometry_scales.csv",
    "radius_angle_summary.csv",
    "within_subject_reliability.csv",
    "label_destruction_subject_summary.csv",
    "label_destruction_group_summary.csv",
    "unrelated_subject_derangement_summary.csv",
    "loso_templates.csv",
    "cross_subject_shared_geometry.csv",
    "semantic_permutation_null_summary.csv",
    "oracle_permutation_all_24_scores.csv",
    "oracle_permutation_subject_summary.csv",
    "oracle_permutation_group_summary.csv",
    "discovery_confirmatory_comparison.csv",
    "airm_le_robustness.csv",
    "hypothesis_chain_status.csv",
    "subject_bootstrap_summary.csv",
    "leave_one_subject_out_influence.csv",
)
REQUIRED_OBJECTS = (
    "airm_marginal_means.npz",
    "airm_class_means.npz",
    "le_marginal_means.npz",
    "le_class_means.npz",
    "D_matrices.npz",
    "G_matrices.npz",
)
REQUIRED_NULLS = (
    "label_destruction_group_statistics.npz",
    "semantic_permutation_group_statistics.npz",
    "oracle_rank_null.npz",
)
POST_CONFIRMATORY_TABLES = (
    "discovery_confirmatory_comparison.csv",
    "airm_le_robustness.csv",
    "hypothesis_chain_status.csv",
)
GEOMETRIES = ("AIRM", "LE")
OBJECTS = ("D", "G")
STAGES = ("R", "S", "P")
PHASES = ("discovery", "confirmatory")
SUBJECTS = tuple(range(1, 10))
SPLITS = ("A", "B", "F")
CLASSES = ("left_hand", "right_hand", "feet", "tongue")
TERMINAL_LABELS = (
    "GO_STRONG",
    "GO_METRIC_ONLY",
    "STOP_TANGENT_ONLY",
    "STOP_NO_SHARED_GEOMETRY",
    "UNASSESSED_NUMERICAL_FAILURE",
    "UNASSESSED_DATA_CONTRACT_FAILURE",
    "UNASSESSED_DEGENERATE_GEOMETRY",
)
LE_LABELS = (
    "AIRM+LE CONSISTENT",
    "AIRM-SPECIFIC",
    "LE-ONLY — DOES NOT RESCUE AIRM FAILURE",
    "AIRM/LE DISCORDANT",
    "UNASSESSED",
)
FAILURE_TERMINAL = {
    "data": "UNASSESSED_DATA_CONTRACT_FAILURE",
    "numerical": "UNASSESSED_NUMERICAL_FAILURE",
    "degenerate": "UNASSESSED_DEGENERATE_GEOMETRY",
}
FAILURE_COMMON_KEYS = (
    "protocol_version",
    "protocol_sha256",
    "config_sha256",
    "code_commit",
    "phase",
    "session",
    "status",
)
FAILURE_MANIFEST_KEYS = frozenset(
    (*FAILURE_COMMON_KEYS, "schema_version", "failure_class", "terminal_decision",
     "reason_code", "reason", "scientific_nulls_executed",
     "downstream_phase_permitted", "le_robustness_label")
)
FAILURE_DECISION_KEYS = FAILURE_MANIFEST_KEYS | {
    "failure_manifest",
    "failure_manifest_sha256",
    "chains",
}


class ReportingContractError(RuntimeError):
    """Raised before writes when a prerequisite violates its frozen contract."""


@dataclass(frozen=True)
class ConditionalVerdicts:
    terminal_decision: str
    le_robustness_label: str
    failure_class: str | None
    failure_details: tuple[str, ...]
    chain_pass: Mapping[tuple[str, str], bool]
    stage_status: pd.DataFrame
    stage_operands: pd.DataFrame
    next_question: str


@dataclass(frozen=True)
class ReportingInputs:
    repo_root: Path
    output_root: Path
    config_path: Path
    config: Mapping[str, Any]
    config_sha256: str
    protocol_sha256: str
    root_json: Mapping[str, Mapping[str, Any]]
    unlock: Mapping[str, Any]
    tables: Mapping[str, pd.DataFrame]
    snapshot_tables: Mapping[str, Mapping[str, pd.DataFrame]]
    object_archives: Mapping[str, Mapping[str, np.ndarray]]
    null_archives: Mapping[str, Mapping[str, np.ndarray]]
    failure_class: str | None
    failure_details: tuple[str, ...]
    failure_manifest: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReportingArtifacts:
    verdicts: ConditionalVerdicts
    figure_sources: Mapping[str, pd.DataFrame]
    figure_paths: tuple[Path, ...]
    figure_source_paths: tuple[Path, ...]
    report_path: Path
    report_text: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ReportingContractError(f"cannot hash required artifact: {path}") from error
    return digest.hexdigest()


def _resolve_inside(root: Path, value: str | Path, *, label: str) -> Path:
    candidate = Path(value).expanduser()
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ReportingContractError(f"{label} escapes repository root: {candidate}") from error
    return candidate


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportingContractError(f"cannot parse {name}: {path}") from error
    if not isinstance(value, dict):
        raise ReportingContractError(f"{name} must be a JSON object")
    return value


def _read_csv(path: Path, *, name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ReportingContractError(f"cannot parse {name}: {path}") from error
    if frame.empty:
        raise ReportingContractError(f"{name} is empty")
    return frame


def _read_npz(path: Path, *, name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise ReportingContractError(f"{name} contains no arrays")
            return {key: np.asarray(archive[key]) for key in archive.files}
    except ReportingContractError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise ReportingContractError(f"cannot parse {name}: {path}") from error


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ReportingContractError(f"{name} missing columns: {missing}")


def _strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise ReportingContractError(f"{label} must be an explicit boolean")


def _assert_exact_values(series: pd.Series, expected: set[str], *, name: str) -> None:
    observed = set(series.astype(str))
    if observed != expected:
        raise ReportingContractError(
            f"{name} values mismatch: expected={sorted(expected)}, observed={sorted(observed)}"
        )


def _assert_unique_grid(
    frame: pd.DataFrame,
    columns: Sequence[str],
    expected: set[tuple[Any, ...]],
    *,
    name: str,
) -> None:
    if frame.duplicated(list(columns)).any():
        raise ReportingContractError(f"{name} has duplicate logical rows")
    observed = set(frame[list(columns)].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected - observed, key=repr)[:8]
        extra = sorted(observed - expected, key=repr)[:8]
        raise ReportingContractError(
            f"{name} grid mismatch; missing={missing}, extra={extra}"
        )


def _verify_common_provenance(
    payload: Mapping[str, Any],
    *,
    name: str,
    config: Mapping[str, Any],
    config_sha256: str,
    protocol_sha256: str,
) -> None:
    expected = {
        "protocol_version": str(config["protocol"]["version"]),
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ReportingContractError(
                f"{name}.{key} mismatch: expected {value!r}, observed {payload.get(key)!r}"
            )


def _verify_snapshot_payload(
    snapshot: Mapping[str, Any], *, repo_root: Path, name: str
) -> None:
    required = {"algorithm", "file_count", "total_bytes", "files", "aggregate_sha256"}
    if not required.issubset(snapshot):
        raise ReportingContractError(f"{name} lacks snapshot fields {sorted(required - set(snapshot))}")
    records = snapshot["files"]
    if not isinstance(records, list) or not records:
        raise ReportingContractError(f"{name}.files must be a nonempty list")
    if snapshot["algorithm"] != "sha256_canonical_json_sorted_file_records_v1":
        raise ReportingContractError(f"{name} has unknown snapshot algorithm")
    canonical_paths = [str(record.get("path", "")) for record in records if isinstance(record, dict)]
    if len(canonical_paths) != len(records) or canonical_paths != sorted(canonical_paths):
        raise ReportingContractError(f"{name} records are not canonically path-sorted")
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ReportingContractError(f"{name} contains duplicate file records")
    total = 0
    for record in records:
        if set(record) != {"path", "bytes", "sha256"}:
            raise ReportingContractError(f"{name} file record schema mismatch")
        path = _resolve_inside(repo_root, record["path"], label=f"{name} record")
        if path.is_symlink() or not path.is_file():
            raise ReportingContractError(f"{name} recorded file missing or symlink: {path}")
        observed_bytes = int(path.stat().st_size)
        observed_sha256 = _sha256_file(path)
        if (
            observed_bytes != int(record["bytes"])
            or observed_sha256 != str(record["sha256"])
        ):
            raise ReportingContractError(
                f"{name} SHA-256 mismatch or size mismatch for {path}"
            )
        total += observed_bytes
    aggregate = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    if aggregate != snapshot["aggregate_sha256"]:
        raise ReportingContractError(f"{name} aggregate SHA-256 mismatch")
    if int(snapshot["file_count"]) != len(records) or int(snapshot["total_bytes"]) != total:
        raise ReportingContractError(f"{name} count/byte totals mismatch")


def _verify_unlock(
    unlock: Mapping[str, Any],
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    protocol_sha256: str,
) -> None:
    unsigned = dict(unlock)
    expected_self_hash = str(unsigned.pop("manifest_sha256", ""))
    observed_self_hash = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if expected_self_hash != observed_self_hash:
        raise ReportingContractError("confirmatory unlock self-hash mismatch")
    expected = {
        "schema_version": "1.0",
        "status": "CONFIRMATORY_UNLOCKED",
        "confirmatory_designation": "STRICT_CONFIRMATORY",
        "prior_non_anatomy_session1_access": True,
        "prior_session1_conditional_object_analysis": False,
        "discovery_session": "0train",
        "confirmatory_session": "1test",
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "branch": str(config["protocol"]["branch"]),
        "confirmatory_raw_ordered_manifest_sha256": str(
            config["confirmatory_inputs"]["ordered_manifest_sha256"]
        ),
    }
    for key, value in expected.items():
        if unlock.get(key) != value:
            raise ReportingContractError(
                f"confirmatory unlock field {key!r} mismatch: "
                f"expected {value!r}, observed {unlock.get(key)!r}"
            )
    if unlock.get("working_tree") != {"clean": True, "porcelain": ""}:
        raise ReportingContractError("confirmatory unlock was not created from a clean tree")
    for key in ("locked_head", "code_commit"):
        value = str(unlock.get(key, ""))
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ReportingContractError(f"confirmatory unlock has invalid {key}")
    for key in ("code_snapshot", "discovery_snapshot"):
        snapshot = unlock.get(key)
        if not isinstance(snapshot, Mapping):
            raise ReportingContractError(f"confirmatory unlock lacks {key}")
        _verify_snapshot_payload(snapshot, repo_root=repo_root, name=f"unlock.{key}")


def compute_frozen_verdicts(
    stage_operands: pd.DataFrame,
    *,
    hard_gates_pass: bool,
    failure_class: str | None = None,
    failure_details: Sequence[str] = (),
) -> ConditionalVerdicts:
    """Apply exact R/S/P chains and terminal/LE labels to canonical operands.

    ``stage_operands`` has one row for each geometry/object/stage and columns
    ``discovery_effect``, ``confirmatory_effect``, and ``confirmatory_p``.
    Discovery p-values remain reportable but never vote.
    """

    required = (
        "geometry",
        "object",
        "stage",
        "discovery_effect",
        "confirmatory_effect",
        "confirmatory_p",
    )
    _require_columns(stage_operands, required, name="stage_operands")
    expected = {
        (geometry, object_name, stage)
        for geometry in GEOMETRIES
        for object_name in OBJECTS
        for stage in STAGES
    }
    _assert_unique_grid(
        stage_operands,
        ("geometry", "object", "stage"),
        expected,
        name="stage_operands",
    )
    valid_failures = {None, "data", "numerical", "degenerate"}
    if failure_class not in valid_failures:
        raise ReportingContractError(f"invalid failure class: {failure_class!r}")
    numeric_columns = ("discovery_effect", "confirmatory_effect", "confirmatory_p")
    normalized = stage_operands.copy()
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if failure_class is None and (
            normalized[column].isna().any()
            or not np.isfinite(normalized[column].to_numpy(dtype=float)).all()
        ):
            raise ReportingContractError(f"stage_operands.{column} is non-finite")
    if failure_class is None and np.any(
        (normalized["confirmatory_p"] < 0.0)
        | (normalized["confirmatory_p"] > 1.0)
    ):
        raise ReportingContractError("stage_operands.confirmatory_p is outside [0,1]")
    chain_pass: dict[tuple[str, str], bool] = {}
    status_rows: list[dict[str, Any]] = []
    if failure_class is None:
        for geometry in GEOMETRIES:
            for object_name in OBJECTS:
                subset = normalized[
                    (normalized["geometry"] == geometry)
                    & (normalized["object"] == object_name)
                ].set_index("stage")
                evidence = {
                    stage: StageEvidence(
                        discovery_effect=float(subset.loc[stage, "discovery_effect"]),
                        confirmatory_effect=float(subset.loc[stage, "confirmatory_effect"]),
                        confirmatory_p=float(subset.loc[stage, "confirmatory_p"]),
                        hard_gates_pass=bool(hard_gates_pass),
                    )
                    for stage in STAGES
                }
                decisions, passed = evaluate_fixed_sequence(evidence)
                chain_pass[(geometry, object_name)] = passed
                for stage in STAGES:
                    decision = decisions[stage]
                    status_rows.append(
                        {
                            "geometry": geometry,
                            "object": object_name,
                            "stage": stage,
                            "eligible": decision.eligible,
                            "criterion_pass": decision.criterion_pass,
                            "status": decision.status,
                        }
                    )
        terminal = terminal_airm_decision(
            chain_pass[("AIRM", "D")], chain_pass[("AIRM", "G")]
        )
        le_label = le_robustness_label(
            (chain_pass[("AIRM", "D")], chain_pass[("AIRM", "G")]),
            (chain_pass[("LE", "D")], chain_pass[("LE", "G")]),
        )
    else:
        terminal = terminal_airm_decision(False, False, failure=failure_class)
        le_label = "UNASSESSED"
        for geometry in GEOMETRIES:
            for object_name in OBJECTS:
                chain_pass[(geometry, object_name)] = False
                for stage in STAGES:
                    status_rows.append(
                        {
                            "geometry": geometry,
                            "object": object_name,
                            "stage": stage,
                            "eligible": False,
                            "criterion_pass": False,
                            "status": "UNASSESSED",
                        }
                    )

    if terminal.startswith("GO_"):
        next_question = (
            "Can the four target components be recovered without labels strongly enough "
            "to preserve the shared relational geometry?"
        )
    elif terminal.startswith("STOP_"):
        next_question = (
            "Which representation or mechanistic anchor could replace the failed "
            "WHOLE-covariance relational anchor?"
        )
    else:
        next_question = (
            "Can the frozen Conditional-Geometry v1 pipeline be rerun after resolving "
            "the recorded protocol failure without changing its analysis rules?"
        )
    return ConditionalVerdicts(
        terminal_decision=terminal,
        le_robustness_label=le_label,
        failure_class=failure_class,
        failure_details=tuple(str(item) for item in failure_details),
        chain_pass=chain_pass,
        stage_status=pd.DataFrame(status_rows),
        stage_operands=normalized.sort_values(
            ["geometry", "object", "stage"], kind="stable"
        ).reset_index(drop=True),
        next_question=next_question,
    )


# Producer-schema binding, figure construction, and report rendering follow
# below.  They intentionally live in the same module so the script has one
# strict validation boundary before it writes any artifact.


COMMON_PREFIX = (
    "protocol_version",
    "protocol_sha256",
    "config_sha256",
    "code_commit",
    "phase",
    "session",
    "status",
)
GROUP_SUMMARY_COLUMNS = (
    "geometry",
    "object",
    "stage",
    "observed",
    "null_median",
    "effect",
    "p_value",
    "exceedances",
    "replicates",
    "gate_pass",
)
TABLE_REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "dataset_contract.csv": (
        "scope",
        "subject",
        "run",
        "class_label",
        "split",
        "observed_count",
        "expected_count",
        "passed",
    ),
    "covariance_sanity.csv": (
        "covariance_index",
        "trial_uid",
        "subject",
        "run",
        "class_label",
        "finite",
        "symmetry_relative_error",
        "min_eigenvalue",
        "max_eigenvalue",
        "condition_number",
        "passed",
    ),
    "airm_mean_convergence.csv": (
        "geometry",
        "subject",
        "split",
        "mean_kind",
        "class_label",
        "n_samples",
        "karcher_residual",
        "custom_relative_error",
        "warning_count",
        "warning_messages",
        "spd_passed",
        "passed",
        "failure_reasons",
    ),
    "le_mean_correctness.csv": (
        "geometry",
        "subject",
        "split",
        "mean_kind",
        "class_label",
        "n_samples",
        "karcher_residual",
        "custom_relative_error",
        "warning_count",
        "warning_messages",
        "spd_passed",
        "passed",
        "failure_reasons",
    ),
    "centering_isometry_gate.csv": (
        "geometry",
        "subject",
        "split",
        "d_centering_relative_error",
        "limit",
        "passed",
    ),
    "orthogonal_gauge_gate.csv": (
        "geometry",
        "subject",
        "split",
        "d_relative_error",
        "g_relative_error",
        "d_limit",
        "g_limit",
        "g_direct_whitened_relative_error",
        "g_direct_whitened_limit",
        "permutation_d_relative_error",
        "permutation_g_relative_error",
        "permutation_limit",
        "le_d_g_identity_relative_error",
        "le_d_g_identity_limit",
        "d_symmetry_relative_error",
        "d_diagonal_max_abs",
        "d_minimum",
        "g_symmetry_relative_error",
        "structure_tolerance",
        "passed",
    ),
    "degenerate_geometry_audit.csv": (
        "geometry",
        "object",
        "subject",
        "split",
        "shape_norm",
        "degeneracy_threshold",
        "is_degenerate",
        "passed",
    ),
    "D_shape_vectors.csv": (
        "subject",
        "split",
        "geometry",
        "shape_norm",
        "raw_0",
        "raw_1",
        "raw_2",
        "raw_3",
        "raw_4",
        "raw_5",
        "z_0",
        "z_1",
        "z_2",
        "z_3",
        "z_4",
        "z_5",
    ),
    "G_shape_vectors.csv": (
        "subject",
        "split",
        "geometry",
        "shape_norm",
        "raw_0",
        "raw_1",
        "raw_2",
        "raw_3",
        "raw_4",
        "raw_5",
        "raw_6",
        "raw_7",
        "raw_8",
        "raw_9",
        "z_0",
        "z_1",
        "z_2",
        "z_3",
        "z_4",
        "z_5",
        "z_6",
        "z_7",
        "z_8",
        "z_9",
    ),
    "absolute_geometry_scales.csv": (
        "subject",
        "split",
        "geometry",
        "D_shape_norm",
        "G_shape_norm",
        "D_upper_mean",
        "D_upper_max",
        "class_radius_mean",
        "class_radius_max",
    ),
    "radius_angle_summary.csv": (
        "subject",
        "split",
        "geometry",
        "class_left",
        "class_right",
        "radius_left",
        "radius_right",
        "cosine",
        "angle_radians",
    ),
    "within_subject_reliability.csv": (
        "geometry",
        "object",
        "stage",
        "subject",
        "observed",
        "group_observed",
    ),
    "label_destruction_subject_summary.csv": (
        "geometry",
        "object",
        "stage",
        "subject",
        "observed",
        "null_median",
        "effect",
        "null_percentile",
        "replicates",
    ),
    "label_destruction_group_summary.csv": GROUP_SUMMARY_COLUMNS,
    "unrelated_subject_derangement_summary.csv": (
        "geometry",
        "object",
        "same_subject_median",
        "unrelated_median",
        "unrelated_min",
        "unrelated_max",
        "derangement_count",
    ),
    "loso_templates.csv": (
        "geometry",
        "object",
        "target_subject",
        "template_source_split",
        "target_split",
        "feature_index",
        "value",
    ),
    "cross_subject_shared_geometry.csv": (
        "geometry",
        "object",
        "stage",
        "subject",
        "observed",
        "group_observed",
        "null_median",
        "effect",
        "null_percentile",
        "replicates",
    ),
    "semantic_permutation_null_summary.csv": GROUP_SUMMARY_COLUMNS,
    "oracle_permutation_all_24_scores.csv": (
        "subject",
        "geometry",
        "object",
        "stage",
        "permutation_index",
        "permutation",
        "is_identity",
        "score",
    ),
    "oracle_permutation_subject_summary.csv": (
        "subject",
        "geometry",
        "object",
        "stage",
        "identity_score",
        "true_rank",
        "normalized_rank",
        "top1_exact",
        "best_permutation_index",
        "margin",
        "best_permutation",
        "second_best_permutation_index",
        "second_best_permutation",
        "null_median",
        "effect",
        "null_percentile",
    ),
    "oracle_permutation_group_summary.csv": GROUP_SUMMARY_COLUMNS,
    "discovery_confirmatory_comparison.csv": (
        "geometry",
        "object",
        "stage",
        "discovery_observed",
        "discovery_effect",
        "confirmatory_observed",
        "confirmatory_effect",
        "confirmatory_p",
    ),
    "airm_le_robustness.csv": (
        "object",
        "stage",
        "airm_discovery_effect",
        "airm_confirmatory_effect",
        "le_discovery_effect",
        "le_confirmatory_effect",
    ),
    "hypothesis_chain_status.csv": (
        "geometry",
        "object",
        "stage",
        "eligible",
        "criterion_pass",
        "stage_status",
        "chain_pass",
    ),
    "subject_bootstrap_summary.csv": (
        "geometry",
        "object",
        "stage",
        "observed_median",
        "ci_low",
        "ci_high",
        "replicates",
        "discovery_confirmatory_effect_delta_median",
        "discovery_confirmatory_effect_delta_ci_low",
        "discovery_confirmatory_effect_delta_ci_high",
        "airm_minus_le_effect_delta_median",
        "airm_minus_le_effect_delta_ci_low",
        "airm_minus_le_effect_delta_ci_high",
    ),
    "leave_one_subject_out_influence.csv": (
        "subject",
        "geometry",
        "object",
        "stage",
        "influence",
        "subject_score",
        "subject_effect",
        "discovery_subject_effect",
        "confirmatory_subject_effect",
        "discovery_confirmatory_effect_delta",
        "airm_minus_le_subject_effect_delta",
    ),
}
GROUP_TABLE_STAGE = {
    "label_destruction_group_summary.csv": "R",
    "semantic_permutation_null_summary.csv": "S",
    "oracle_permutation_group_summary.csv": "P",
}
NULL_FILE_FOR_GROUP_TABLE = {
    "label_destruction_group_summary.csv": "label_destruction_group_statistics.npz",
    "semantic_permutation_null_summary.csv": "semantic_permutation_group_statistics.npz",
    "oracle_permutation_group_summary.csv": "oracle_rank_null.npz",
}
EXPECTED_NULL_REPLICATES = {
    "label_destruction_group_statistics.npz": 1999,
    "semantic_permutation_group_statistics.npz": 100_000,
    "oracle_rank_null.npz": 1_000_000,
}


def _strict_bool_series(series: pd.Series, *, label: str) -> pd.Series:
    return series.map(lambda value: _strict_bool(value, label=label)).astype(bool)


def _validate_common_table(
    frame: pd.DataFrame,
    *,
    filename: str,
    config: Mapping[str, Any],
    config_sha256: str,
    protocol_sha256: str,
    code_commit: str,
    expected_phase: str | None,
) -> pd.DataFrame:
    if tuple(frame.columns[: len(COMMON_PREFIX)]) != COMMON_PREFIX:
        raise ReportingContractError(
            f"{filename} common prefix mismatch: {tuple(frame.columns[:len(COMMON_PREFIX)])}"
        )
    _require_columns(
        frame,
        (*COMMON_PREFIX, *TABLE_REQUIRED_COLUMNS[filename]),
        name=filename,
    )
    expected_provenance = {
        "protocol_version": str(config["protocol"]["version"]),
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "code_commit": code_commit,
    }
    for column, expected in expected_provenance.items():
        observed = set(frame[column].astype(str))
        if observed != {str(expected)}:
            raise ReportingContractError(
                f"{filename}.{column} mismatch: expected {expected!r}, observed {sorted(observed)}"
            )
    allowed_status = {"PASS", "FAIL", "UNASSESSED", "DESCRIPTIVE_ONLY"}
    statuses = set(frame["status"].astype(str))
    if not statuses.issubset(allowed_status):
        raise ReportingContractError(f"{filename} contains invalid statuses: {sorted(statuses)}")
    phase_session = {
        "discovery": "0train",
        "confirmatory": "1test",
        "combined": "0train+1test",
    }
    if expected_phase is not None:
        expected_phases = {expected_phase}
    elif filename in POST_CONFIRMATORY_TABLES:
        expected_phases = {"combined"}
    else:
        expected_phases = {"discovery", "confirmatory"}
    _assert_exact_values(frame["phase"], expected_phases, name=f"{filename}.phase")
    for phase, subset in frame.groupby(frame["phase"].astype(str), sort=False):
        observed_sessions = set(subset["session"].astype(str))
        if observed_sessions != {phase_session[phase]}:
            raise ReportingContractError(
                f"{filename} phase {phase} has sessions {sorted(observed_sessions)}"
            )
    return frame.copy()


def _bool_column(frame: pd.DataFrame, column: str, *, filename: str) -> pd.Series:
    return _strict_bool_series(frame[column], label=f"{filename}.{column}")


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    filename: str,
    allow_na: bool = False,
) -> pd.Series:
    result = pd.to_numeric(frame[column], errors="coerce")
    if not allow_na and (
        result.isna().any() or not np.isfinite(result.to_numpy(dtype=float)).all()
    ):
        raise ReportingContractError(f"{filename}.{column} contains non-finite values")
    finite = result.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ReportingContractError(f"{filename}.{column} contains Inf")
    return result


def _validate_gate_and_geometry_tables(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any]
) -> tuple[str | None, tuple[str, ...]]:
    """Validate gate semantics and return one global failure category.

    Frozen precedence is DATA CONTRACT, then the more specific DEGENERATE
    GEOMETRY, then other NUMERICAL failures.  It is used only when multiple
    explicit hard-gate rows fail simultaneously; no available-case result is
    ever computed.
    """

    details: dict[str, list[str]] = {"data": [], "degenerate": [], "numerical": []}
    dataset = tables["dataset_contract.csv"]
    dataset_pass = _bool_column(dataset, "passed", filename="dataset_contract.csv")
    observed_counts = _numeric_column(
        dataset, "observed_count", filename="dataset_contract.csv"
    )
    expected_counts = _numeric_column(
        dataset, "expected_count", filename="dataset_contract.csv"
    )
    if (dataset_pass != (observed_counts == expected_counts)).any():
        raise ReportingContractError("dataset_contract passed flag disagrees with counts")
    for index in dataset.index[~dataset_pass]:
        details["data"].append(
            f"dataset_contract:{dataset.loc[index, 'phase']}:{dataset.loc[index, 'scope']}"
        )

    covariance = tables["covariance_sanity.csv"]
    covariance_pass = _bool_column(covariance, "passed", filename="covariance_sanity.csv")
    covariance_finite = _bool_column(covariance, "finite", filename="covariance_sanity.csv")
    covariance_symmetry = _numeric_column(
        covariance, "symmetry_relative_error", filename="covariance_sanity.csv", allow_na=True
    )
    covariance_minimum = _numeric_column(
        covariance, "min_eigenvalue", filename="covariance_sanity.csv", allow_na=True
    )
    covariance_condition = _numeric_column(
        covariance, "condition_number", filename="covariance_sanity.csv", allow_na=True
    )
    computed_covariance_pass = (
        covariance_finite
        & (covariance_symmetry <= float(config["hard_gates"]["symmetry_relative_error_max"]))
        & (covariance_minimum > 0.0)
        & (covariance_condition <= float(config["hard_gates"]["condition_number_max"]))
    )
    if (covariance_pass != computed_covariance_pass).any():
        raise ReportingContractError("covariance_sanity passed flag disagrees with frozen gates")
    for index in covariance.index[~covariance_pass][:20]:
        details["numerical"].append(
            f"covariance_sanity:{covariance.loc[index, 'phase']}:"
            f"{covariance.loc[index, 'trial_uid']}"
        )
    if (~covariance_pass).sum() > 20:
        details["numerical"].append(
            f"covariance_sanity:and_{int((~covariance_pass).sum()) - 20}_more"
        )

    for filename in (
        "airm_mean_convergence.csv",
        "le_mean_correctness.csv",
        "centering_isometry_gate.csv",
        "orthogonal_gauge_gate.csv",
    ):
        frame = tables[filename]
        passed = _bool_column(frame, "passed", filename=filename)
        for index in frame.index[~passed][:20]:
            descriptor = ":".join(
                str(frame.loc[index, key])
                for key in ("phase", "subject", "split")
                if key in frame
            )
            details["numerical"].append(f"{filename}:{descriptor}")
        if (~passed).sum() > 20:
            details["numerical"].append(f"{filename}:and_{int((~passed).sum()) - 20}_more")

    degenerate = tables["degenerate_geometry_audit.csv"]
    is_degenerate = _bool_column(
        degenerate, "is_degenerate", filename="degenerate_geometry_audit.csv"
    )
    degenerate_pass = _bool_column(
        degenerate, "passed", filename="degenerate_geometry_audit.csv"
    )
    bad_degenerate = is_degenerate | ~degenerate_pass
    for index in degenerate.index[bad_degenerate][:20]:
        details["degenerate"].append(
            "degenerate_geometry:"
            + ":".join(
                str(degenerate.loc[index, key])
                for key in ("phase", "subject", "split", "geometry", "object")
            )
        )
    if (~degenerate_pass & ~is_degenerate).any():
        raise ReportingContractError(
            "degenerate audit has passed=False without is_degenerate=True"
        )

    # Successful rows must satisfy their frozen numeric threshold rather than
    # merely carrying a producer-authored PASS token.
    airm = tables["airm_mean_convergence.csv"]
    residual = _numeric_column(
        airm, "karcher_residual", filename="airm_mean_convergence.csv", allow_na=True
    )
    warnings_count = _numeric_column(
        airm, "warning_count", filename="airm_mean_convergence.csv", allow_na=True
    )
    airm_pass = _bool_column(airm, "passed", filename="airm_mean_convergence.csv")
    if ((airm_pass) & ((residual > float(config["hard_gates"]["airm_karcher_residual_max"])) | (warnings_count != 0))).any():
        raise ReportingContractError("AIRM PASS row violates residual/warning gate")
    le = tables["le_mean_correctness.csv"]
    le_error = _numeric_column(
        le, "custom_relative_error", filename="le_mean_correctness.csv", allow_na=True
    )
    le_pass = _bool_column(le, "passed", filename="le_mean_correctness.csv")
    if (le_pass & (le_error > float(config["hard_gates"]["le_mean_relative_error_max"]))).any():
        raise ReportingContractError("LE PASS row violates official/custom mean gate")
    centering = tables["centering_isometry_gate.csv"]
    centering_error = _numeric_column(
        centering,
        "d_centering_relative_error",
        filename="centering_isometry_gate.csv",
        allow_na=True,
    )
    centering_limit = _numeric_column(
        centering, "limit", filename="centering_isometry_gate.csv", allow_na=True
    )
    centering_pass = _bool_column(
        centering, "passed", filename="centering_isometry_gate.csv"
    )
    if (centering_pass & (centering_error > centering_limit)).any():
        raise ReportingContractError("centering PASS row exceeds its tolerance")
    gauge = tables["orthogonal_gauge_gate.csv"]
    gauge_pass = _bool_column(gauge, "passed", filename="orthogonal_gauge_gate.csv")
    d_error = _numeric_column(
        gauge, "d_relative_error", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    g_error = _numeric_column(
        gauge, "g_relative_error", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    d_limit = _numeric_column(
        gauge, "d_limit", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    g_limit = _numeric_column(
        gauge, "g_limit", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    direct_error = _numeric_column(
        gauge,
        "g_direct_whitened_relative_error",
        filename="orthogonal_gauge_gate.csv",
        allow_na=True,
    )
    direct_limit = _numeric_column(
        gauge,
        "g_direct_whitened_limit",
        filename="orthogonal_gauge_gate.csv",
        allow_na=True,
    )
    permutation_d = _numeric_column(
        gauge, "permutation_d_relative_error", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    permutation_g = _numeric_column(
        gauge, "permutation_g_relative_error", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    permutation_limit = _numeric_column(
        gauge, "permutation_limit", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    le_identity = _numeric_column(
        gauge, "le_d_g_identity_relative_error", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    le_identity_limit = _numeric_column(
        gauge, "le_d_g_identity_limit", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    d_symmetry = _numeric_column(
        gauge, "d_symmetry_relative_error", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    d_diagonal = _numeric_column(
        gauge, "d_diagonal_max_abs", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    d_minimum = _numeric_column(
        gauge, "d_minimum", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    g_symmetry = _numeric_column(
        gauge, "g_symmetry_relative_error", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    structure_tolerance = _numeric_column(
        gauge, "structure_tolerance", filename="orthogonal_gauge_gate.csv", allow_na=True
    )
    computed_gauge_pass = (
        (d_error <= d_limit)
        & (g_error <= g_limit)
        & (permutation_d <= permutation_limit)
        & (permutation_g <= permutation_limit)
        & (d_symmetry <= structure_tolerance)
        & (d_diagonal <= structure_tolerance)
        & (d_minimum >= -structure_tolerance)
        & (g_symmetry <= structure_tolerance)
        & (
            ((gauge["geometry"].astype(str) == "AIRM") & (direct_error <= direct_limit) & le_identity.isna())
            | ((gauge["geometry"].astype(str) == "LE") & direct_error.isna() & (le_identity <= le_identity_limit))
        )
    )
    if (gauge_pass != computed_gauge_pass).any():
        raise ReportingContractError(
            "orthogonal_gauge_gate passed flag disagrees with full D/G hard-gate metrics"
        )

    if details["data"]:
        return "data", tuple(details["data"] + details["degenerate"] + details["numerical"])
    if details["degenerate"]:
        return "degenerate", tuple(details["degenerate"] + details["numerical"])
    if details["numerical"]:
        return "numerical", tuple(details["numerical"])
    return None, ()


def _validate_success_grids(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any]
) -> None:
    expected_trials = int(config["expected_data"]["trials_per_session"])
    covariance = tables["covariance_sanity.csv"].copy()
    covariance["subject"] = pd.to_numeric(covariance["subject"], errors="coerce")
    covariance["run"] = covariance["run"].astype(str)
    for phase in PHASES:
        part = covariance[covariance["phase"] == phase]
        if len(part) != expected_trials:
            raise ReportingContractError(
                f"covariance_sanity {phase} expected {expected_trials} rows, got {len(part)}"
            )
        if part["trial_uid"].astype(str).duplicated().any():
            raise ReportingContractError(f"covariance_sanity {phase} has duplicate trial_uid")
        if set(part["subject"].astype(int)) != set(SUBJECTS):
            raise ReportingContractError(f"covariance_sanity {phase} subject grid mismatch")
        if set(part["run"]) != set(str(item) for item in config["dataset"]["runs"]):
            raise ReportingContractError(f"covariance_sanity {phase} run grid mismatch")

    expected_subject_split = {
        (phase, subject, split, geometry)
        for phase in PHASES
        for subject in SUBJECTS
        for split in SPLITS
        for geometry in GEOMETRIES
    }
    for filename, geometry in (
        ("airm_mean_convergence.csv", "AIRM"),
        ("le_mean_correctness.csv", "LE"),
    ):
        frame = tables[filename].copy()
        frame["subject"] = pd.to_numeric(frame["subject"], errors="raise").astype(int)
        frame["class_label"] = frame["class_label"].fillna("").astype(str)
        expected_mean_grid = {
            (phase, subject, split, "marginal", "")
            for phase in PHASES
            for subject in SUBJECTS
            for split in SPLITS
        } | {
            (phase, subject, split, "class", class_label)
            for phase in PHASES
            for subject in SUBJECTS
            for split in SPLITS
            for class_label in CLASSES
        }
        _assert_unique_grid(
            frame,
            ("phase", "subject", "split", "mean_kind", "class_label"),
            expected_mean_grid,
            name=filename,
        )
        if set(frame["geometry"].astype(str)) != {geometry}:
            raise ReportingContractError(f"{filename} geometry token mismatch")
    for filename in ("centering_isometry_gate.csv", "orthogonal_gauge_gate.csv"):
        frame = tables[filename].copy()
        frame["subject"] = pd.to_numeric(frame["subject"], errors="raise").astype(int)
        _assert_unique_grid(
            frame,
            ("phase", "subject", "split", "geometry"),
            expected_subject_split,
            name=filename,
        )
    degenerate = tables["degenerate_geometry_audit.csv"].copy()
    degenerate["subject"] = pd.to_numeric(degenerate["subject"], errors="raise").astype(int)
    _assert_unique_grid(
        degenerate,
        ("phase", "subject", "split", "geometry", "object"),
        {(*key, object_name) for key in expected_subject_split for object_name in OBJECTS},
        name="degenerate_geometry_audit.csv",
    )
    for filename in ("D_shape_vectors.csv", "G_shape_vectors.csv", "absolute_geometry_scales.csv"):
        frame = tables[filename].copy()
        frame["subject"] = pd.to_numeric(frame["subject"], errors="raise").astype(int)
        _assert_unique_grid(
            frame,
            ("phase", "subject", "split", "geometry"),
            expected_subject_split,
            name=filename,
        )
    radius = tables["radius_angle_summary.csv"].copy()
    radius["subject"] = pd.to_numeric(radius["subject"], errors="raise").astype(int)
    _assert_unique_grid(
        radius,
        ("phase", "subject", "split", "geometry", "class_left", "class_right"),
        {
            (*key, CLASSES[left], CLASSES[right])
            for key in expected_subject_split
            for left in range(4)
            for right in range(left + 1, 4)
        },
        name="radius_angle_summary.csv",
    )
    for filename in ("within_subject_reliability.csv", "cross_subject_shared_geometry.csv"):
        frame = tables[filename].copy()
        frame["subject"] = pd.to_numeric(frame["subject"], errors="raise").astype(int)
        _assert_unique_grid(
            frame,
            ("phase", "subject", "geometry", "object"),
            {
                (phase, subject, geometry, object_name)
                for phase in PHASES
                for subject in SUBJECTS
                for geometry in GEOMETRIES
                for object_name in OBJECTS
            },
            name=filename,
        )
        score = _numeric_column(frame, "observed", filename=filename)
        if np.any((score < -1.0 - 1e-12) | (score > 1.0 + 1e-12)):
            raise ReportingContractError(f"{filename}.score lies outside cosine bounds")
    oracle = tables["oracle_permutation_subject_summary.csv"].copy()
    oracle["subject"] = pd.to_numeric(oracle["subject"], errors="raise").astype(int)
    _assert_unique_grid(
        oracle,
        ("phase", "subject", "geometry", "object"),
        {
            (phase, subject, geometry, object_name)
            for phase in PHASES
            for subject in SUBJECTS
            for geometry in GEOMETRIES
            for object_name in OBJECTS
        },
        name="oracle_permutation_subject_summary.csv",
    )
    ranks = _numeric_column(oracle, "true_rank", filename="oracle_permutation_subject_summary.csv")
    normalized = _numeric_column(oracle, "normalized_rank", filename="oracle_permutation_subject_summary.csv")
    if np.any((ranks < 1) | (ranks > 24) | (ranks != np.floor(ranks))):
        raise ReportingContractError("oracle identity ranks must be integers 1..24")
    if not np.allclose(normalized, (24.0 - ranks) / 23.0, rtol=0.0, atol=1e-12):
        raise ReportingContractError("oracle normalized ranks disagree with frozen formula")

    _validate_locked_loso_templates(tables)
    _validate_subject_effects_and_descriptives(tables, config)


def _validate_locked_loso_templates(tables: Mapping[str, pd.DataFrame]) -> None:
    """Validate the stored discovery-F templates and confirmatory reuse grid."""

    frame = tables["loso_templates.csv"].copy()
    frame["subject"] = pd.to_numeric(
        frame["target_subject"], errors="raise"
    ).astype(int)
    frame["feature_index"] = pd.to_numeric(
        frame["feature_index"], errors="raise"
    ).astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ReportingContractError("loso_templates.csv contains non-finite values")

    expected: set[tuple[Any, ...]] = set()
    for phase in PHASES:
        split_pairs = (
            (("A", "B"), ("B", "A"), ("F", "F"))
            if phase == "discovery"
            else (("F", "A"), ("F", "B"), ("F", "F"))
        )
        for geometry in GEOMETRIES:
            for object_name, dimension in (("D", 6), ("G", 10)):
                for subject in SUBJECTS:
                    for source_split, target_split in split_pairs:
                        for feature_index in range(dimension):
                            expected.add(
                                (
                                    phase,
                                    geometry,
                                    object_name,
                                    subject,
                                    source_split,
                                    target_split,
                                    feature_index,
                                )
                            )
    _assert_unique_grid(
        frame,
        (
            "phase",
            "geometry",
            "object",
            "subject",
            "template_source_split",
            "target_split",
            "feature_index",
        ),
        expected,
        name="loso_templates.csv",
    )

    vector_keys = (
        "phase",
        "geometry",
        "object",
        "subject",
        "template_source_split",
        "target_split",
    )
    norms = frame.groupby(list(vector_keys), sort=False)["value"].apply(
        lambda values: float(np.linalg.norm(values.to_numpy(dtype=float)))
    )
    if not np.allclose(norms.to_numpy(), 1.0, rtol=0.0, atol=1e-10):
        raise ReportingContractError("loso_templates.csv contains a non-unit template")

    discovery_f = frame[
        (frame["phase"] == "discovery")
        & (frame["template_source_split"] == "F")
        & (frame["target_split"] == "F")
    ][["geometry", "object", "subject", "feature_index", "value"]].rename(
        columns={"value": "discovery_value"}
    )
    confirmatory_f = frame[
        (frame["phase"] == "confirmatory")
        & (frame["template_source_split"] == "F")
    ][
        ["geometry", "object", "subject", "target_split", "feature_index", "value"]
    ]
    merged = confirmatory_f.merge(
        discovery_f,
        on=("geometry", "object", "subject", "feature_index"),
        how="left",
        validate="many_to_one",
    )
    if merged["discovery_value"].isna().any() or not np.allclose(
        merged["value"].to_numpy(dtype=float),
        merged["discovery_value"].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-15,
    ):
        raise ReportingContractError(
            "confirmatory LOSO templates differ from locked discovery-F templates"
        )


def _subject_stage_effects(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the canonical raw score and subject-null-referenced effect grid."""

    pieces: list[pd.DataFrame] = []
    for filename, stage, score_column in (
        ("label_destruction_subject_summary.csv", "R", "observed"),
        ("cross_subject_shared_geometry.csv", "S", "observed"),
        ("oracle_permutation_subject_summary.csv", "P", "normalized_rank"),
    ):
        source = tables[filename]
        piece = source[
            [
                "phase",
                "geometry",
                "object",
                "subject",
                score_column,
                "null_median",
                "effect",
                "null_percentile",
            ]
        ].copy()
        piece = piece.rename(columns={score_column: "subject_score"})
        piece["stage"] = stage
        piece["source_table"] = filename
        pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True)
    result["subject"] = pd.to_numeric(result["subject"], errors="raise").astype(int)
    for column in ("subject_score", "null_median", "effect", "null_percentile"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ReportingContractError(f"subject summary {column} is non-finite")
    _assert_unique_grid(
        result,
        ("phase", "geometry", "object", "stage", "subject"),
        {
            (phase, geometry, object_name, stage, subject)
            for phase in PHASES
            for geometry in GEOMETRIES
            for object_name in OBJECTS
            for stage in STAGES
            for subject in SUBJECTS
        },
        name="subject-level R/S/P summaries",
    )
    if not np.allclose(
        result["effect"].to_numpy(dtype=float),
        result["subject_score"].to_numpy(dtype=float)
        - result["null_median"].to_numpy(dtype=float),
        rtol=1e-13,
        atol=1e-15,
    ):
        raise ReportingContractError(
            "subject null-referenced effect is not observed minus subject-null median"
        )
    percentiles = result["null_percentile"].to_numpy(dtype=float)
    if np.any((percentiles < 0.0) | (percentiles > 1.0)):
        raise ReportingContractError("subject null percentile lies outside [0,1]")
    cosine = result[result["stage"].isin(("R", "S"))]["subject_score"].to_numpy(
        dtype=float
    )
    ranks = result[result["stage"] == "P"]["subject_score"].to_numpy(dtype=float)
    if np.any((cosine < -1.0 - 1e-12) | (cosine > 1.0 + 1e-12)):
        raise ReportingContractError("R/S subject score lies outside cosine bounds")
    if np.any((ranks < -1e-12) | (ranks > 1.0 + 1e-12)):
        raise ReportingContractError("P normalized-rank subject score lies outside [0,1]")
    return result


def _assert_close_array(
    observed: Sequence[Any], expected: Sequence[Any], *, label: str
) -> None:
    observed_array = pd.to_numeric(pd.Series(observed), errors="coerce").to_numpy(
        dtype=float
    )
    expected_array = np.asarray(expected, dtype=np.float64)
    if observed_array.shape != expected_array.shape or not np.allclose(
        observed_array, expected_array, rtol=1e-12, atol=1e-14
    ):
        raise ReportingContractError(f"{label} mismatch")


def _assert_all_na(values: Sequence[Any], *, label: str) -> None:
    if not pd.Series(values).isna().all():
        raise ReportingContractError(f"{label} must be NA by frozen schema")


def _validate_subject_effects_and_descriptives(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any]
) -> None:
    """Validate subject pairing, cached-score influence and all bootstrap fields."""

    effects = _subject_stage_effects(tables)
    effect_index = effects.set_index(
        ["phase", "geometry", "object", "stage", "subject"]
    ).sort_index()
    for stage, group_filename, subject_filename, replicate_family in (
        (
            "R",
            "label_destruction_group_summary.csv",
            "label_destruction_subject_summary.csv",
            "label_destruction",
        ),
        (
            "S",
            "semantic_permutation_null_summary.csv",
            "cross_subject_shared_geometry.csv",
            "semantic_permutation",
        ),
        (
            "P",
            "oracle_permutation_group_summary.csv",
            "oracle_permutation_subject_summary.csv",
            "oracle_rank",
        ),
    ):
        group = tables[group_filename]
        subject_table = tables[subject_filename]
        for phase in PHASES:
            for geometry in GEOMETRIES:
                for object_name in OBJECTS:
                    key = (phase, geometry, object_name, stage)
                    scores = effect_index.loc[key].sort_index()["subject_score"].to_numpy(
                        dtype=float
                    )
                    group_row = group[
                        (group["phase"] == phase)
                        & (group["geometry"] == geometry)
                        & (group["object"] == object_name)
                        & (group["stage"] == stage)
                    ]
                    if len(group_row) != 1:
                        raise ReportingContractError(f"{key} group summary is not unique")
                    _assert_close_array(
                        group_row["observed"],
                        [float(np.median(scores))],
                        label=f"{key} group observed median",
                    )
                    if "group_observed" in subject_table:
                        subject_rows = subject_table[
                            (subject_table["phase"] == phase)
                            & (subject_table["geometry"] == geometry)
                            & (subject_table["object"] == object_name)
                            & (subject_table["stage"] == stage)
                        ]
                        _assert_close_array(
                            subject_rows["group_observed"],
                            np.full(len(subject_rows), float(np.median(scores))),
                            label=f"{key} repeated group observed",
                        )
        if "replicates" in subject_table:
            expected_replicates = int(config["nulls"][replicate_family]["replicates"])
            observed_replicates = pd.to_numeric(
                subject_table["replicates"], errors="raise"
            ).to_numpy(dtype=int)
            if not np.all(observed_replicates == expected_replicates):
                raise ReportingContractError(
                    f"{subject_filename} subject-null replicate count mismatch"
                )

    influence = tables["leave_one_subject_out_influence.csv"].copy()
    influence["subject"] = pd.to_numeric(
        influence["subject"], errors="raise"
    ).astype(int)
    _assert_unique_grid(
        influence,
        ("phase", "geometry", "object", "stage", "subject"),
        set(effect_index.index),
        name="leave_one_subject_out_influence.csv",
    )

    for phase in PHASES:
        for geometry in GEOMETRIES:
            for object_name in OBJECTS:
                for stage in STAGES:
                    key = (phase, geometry, object_name, stage)
                    score_rows = effect_index.loc[key].sort_index()
                    influence_rows = influence[
                        (influence["phase"] == phase)
                        & (influence["geometry"] == geometry)
                        & (influence["object"] == object_name)
                        & (influence["stage"] == stage)
                    ].sort_values("subject")
                    scores = score_rows["subject_score"].to_numpy(dtype=float)
                    subject_effects = score_rows["effect"].to_numpy(dtype=float)
                    _assert_close_array(
                        influence_rows["subject_score"],
                        scores,
                        label=f"{key} cached subject_score",
                    )
                    _assert_close_array(
                        influence_rows["influence"],
                        leave_one_subject_out_influence(scores),
                        label=f"{key} cached-score influence",
                    )
                    _assert_close_array(
                        influence_rows["subject_effect"],
                        subject_effects,
                        label=f"{key} subject null-referenced effect",
                    )
                    if phase == "confirmatory":
                        discovery_effects = effect_index.loc[
                            ("discovery", geometry, object_name, stage)
                        ].sort_index()["effect"].to_numpy(dtype=float)
                        _assert_close_array(
                            influence_rows["discovery_subject_effect"],
                            discovery_effects,
                            label=f"{key} stored discovery subject effect",
                        )
                        _assert_close_array(
                            influence_rows["confirmatory_subject_effect"],
                            subject_effects,
                            label=f"{key} stored confirmatory subject effect",
                        )
                        _assert_close_array(
                            influence_rows["discovery_confirmatory_effect_delta"],
                            subject_effects - discovery_effects,
                            label=f"{key} discovery-confirmatory effect delta",
                        )
                    else:
                        _assert_close_array(
                            influence_rows["discovery_subject_effect"],
                            subject_effects,
                            label=f"{key} stored discovery subject effect",
                        )
                        _assert_all_na(
                            influence_rows["confirmatory_subject_effect"],
                            label=f"{key} confirmatory subject effect",
                        )
                        _assert_all_na(
                            influence_rows["discovery_confirmatory_effect_delta"],
                            label=f"{key} discovery-confirmatory effect delta",
                        )
                    if geometry == "AIRM":
                        le_effects = effect_index.loc[
                            (phase, "LE", object_name, stage)
                        ].sort_index()["effect"].to_numpy(dtype=float)
                        _assert_close_array(
                            influence_rows["airm_minus_le_subject_effect_delta"],
                            subject_effects - le_effects,
                            label=f"{key} AIRM-minus-LE subject effect delta",
                        )
                    else:
                        _assert_all_na(
                            influence_rows["airm_minus_le_subject_effect_delta"],
                            label=f"{key} AIRM-minus-LE subject effect delta",
                        )

    bootstrap = tables["subject_bootstrap_summary.csv"].copy()
    _assert_unique_grid(
        bootstrap,
        ("phase", "geometry", "object", "stage"),
        {
            (phase, geometry, object_name, stage)
            for phase in PHASES
            for geometry in GEOMETRIES
            for object_name in OBJECTS
            for stage in STAGES
        },
        name="subject_bootstrap_summary.csv",
    )
    replicates = int(config["statistics"]["bootstrap"]["replicates"])
    master_seed = int(config["protocol"]["seed"])
    phase_tags = config["rng"]["phase_tags"]
    scalar_columns = (
        "observed_median",
        "ci_low",
        "ci_high",
        "discovery_confirmatory_effect_delta_median",
        "discovery_confirmatory_effect_delta_ci_low",
        "discovery_confirmatory_effect_delta_ci_high",
        "airm_minus_le_effect_delta_median",
        "airm_minus_le_effect_delta_ci_low",
        "airm_minus_le_effect_delta_ci_high",
    )
    for column in scalar_columns:
        bootstrap[column] = pd.to_numeric(bootstrap[column], errors="coerce")
    for phase in PHASES:
        for geometry in GEOMETRIES:
            for object_name in OBJECTS:
                for stage in STAGES:
                    key = (phase, geometry, object_name, stage)
                    row = bootstrap[
                        (bootstrap["phase"] == phase)
                        & (bootstrap["geometry"] == geometry)
                        & (bootstrap["object"] == object_name)
                        & (bootstrap["stage"] == stage)
                    ].iloc[0]
                    if int(row["replicates"]) != replicates:
                        raise ReportingContractError(f"{key} bootstrap replicate count mismatch")
                    score_rows = effect_index.loc[key].sort_index()
                    scores = score_rows["subject_score"].to_numpy(dtype=float)
                    subject_effects = score_rows["effect"].to_numpy(dtype=float)
                    ordinary = subject_bootstrap_median(
                        scores,
                        replicates=replicates,
                        master_seed=master_seed,
                        phase_tag=int(phase_tags[phase]),
                    )
                    _assert_close_array(
                        [row["observed_median"], row["ci_low"], row["ci_high"]],
                        [float(np.median(scores)), ordinary.ci_low, ordinary.ci_high],
                        label=f"{key} subject bootstrap",
                    )
                    if phase == "confirmatory":
                        discovery_effects = effect_index.loc[
                            ("discovery", geometry, object_name, stage)
                        ].sort_index()["effect"].to_numpy(dtype=float)
                        paired = subject_bootstrap_paired_median_delta(
                            subject_effects,
                            discovery_effects,
                            replicates=replicates,
                            master_seed=master_seed,
                            phase_tag=int(phase_tags["paired_common"]),
                        )
                        _assert_close_array(
                            [
                                row["discovery_confirmatory_effect_delta_median"],
                                row["discovery_confirmatory_effect_delta_ci_low"],
                                row["discovery_confirmatory_effect_delta_ci_high"],
                            ],
                            [
                                float(np.median(subject_effects) - np.median(discovery_effects)),
                                paired.ci_low,
                                paired.ci_high,
                            ],
                            label=f"{key} paired discovery-confirmatory effect bootstrap",
                        )
                    else:
                        _assert_all_na(
                            [
                                row["discovery_confirmatory_effect_delta_median"],
                                row["discovery_confirmatory_effect_delta_ci_low"],
                                row["discovery_confirmatory_effect_delta_ci_high"],
                            ],
                            label=f"{key} discovery-confirmatory bootstrap fields",
                        )
                    if geometry == "AIRM":
                        le_effects = effect_index.loc[
                            (phase, "LE", object_name, stage)
                        ].sort_index()["effect"].to_numpy(dtype=float)
                        paired = subject_bootstrap_paired_median_delta(
                            subject_effects,
                            le_effects,
                            replicates=replicates,
                            master_seed=master_seed,
                            phase_tag=int(phase_tags["paired_common"]),
                        )
                        _assert_close_array(
                            [
                                row["airm_minus_le_effect_delta_median"],
                                row["airm_minus_le_effect_delta_ci_low"],
                                row["airm_minus_le_effect_delta_ci_high"],
                            ],
                            [
                                float(np.median(subject_effects) - np.median(le_effects)),
                                paired.ci_low,
                                paired.ci_high,
                            ],
                            label=f"{key} paired AIRM-minus-LE effect bootstrap",
                        )
                    else:
                        _assert_all_na(
                            [
                                row["airm_minus_le_effect_delta_median"],
                                row["airm_minus_le_effect_delta_ci_low"],
                                row["airm_minus_le_effect_delta_ci_high"],
                            ],
                            label=f"{key} AIRM-minus-LE bootstrap fields",
                        )


def _table_rows_multiset(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    normalized = frame.copy()
    return sorted(
        tuple("<NA>" if pd.isna(value) else str(value) for value in row)
        for row in normalized.itertuples(index=False, name=None)
    )


def _validate_snapshot_table_identity(
    root_tables: Mapping[str, pd.DataFrame],
    snapshot_tables: Mapping[str, Mapping[str, pd.DataFrame]],
) -> None:
    for phase in PHASES:
        for filename in REQUIRED_TABLES:
            if filename in POST_CONFIRMATORY_TABLES:
                continue
            snapshot = snapshot_tables[phase][filename]
            root_subset = root_tables[filename][
                root_tables[filename]["phase"].astype(str) == phase
            ]
            if tuple(snapshot.columns) != tuple(root_subset.columns):
                raise ReportingContractError(
                    f"{filename} root/{phase} snapshot column mismatch"
                )
            if _table_rows_multiset(snapshot) != _table_rows_multiset(root_subset):
                raise ReportingContractError(
                    f"{filename} root rows differ from immutable {phase} snapshot"
                )


def _validate_null_archive(
    archive: Mapping[str, np.ndarray],
    *,
    filename: str,
    failure_class: str | None,
) -> None:
    expected_keys = {"replicate_indices", "geometries", "objects", "group_statistics"}
    if set(archive) != expected_keys:
        raise ReportingContractError(
            f"{filename} NPZ keys mismatch: expected {sorted(expected_keys)}, got {sorted(archive)}"
        )
    replicate_indices = np.asarray(archive["replicate_indices"])
    geometries = np.asarray(archive["geometries"]).astype(str)
    objects = np.asarray(archive["objects"]).astype(str)
    statistics = np.asarray(archive["group_statistics"], dtype=np.float64)
    if not np.array_equal(geometries, np.asarray(GEOMETRIES)):
        raise ReportingContractError(f"{filename} geometry axis/order mismatch")
    if not np.array_equal(objects, np.asarray(OBJECTS)):
        raise ReportingContractError(f"{filename} object axis/order mismatch")
    expected_replicates = 0 if failure_class is not None else EXPECTED_NULL_REPLICATES[filename]
    if replicate_indices.shape != (expected_replicates,) or not np.array_equal(
        replicate_indices, np.arange(expected_replicates, dtype=replicate_indices.dtype)
    ):
        raise ReportingContractError(f"{filename} replicate index grid mismatch")
    if statistics.shape != (2, 2, expected_replicates):
        raise ReportingContractError(
            f"{filename} group_statistics shape mismatch: {statistics.shape}"
        )
    if not np.isfinite(statistics).all():
        raise ReportingContractError(f"{filename} contains non-finite null statistics")


def _validate_group_summary_against_null(
    frame: pd.DataFrame,
    archive: Mapping[str, np.ndarray],
    *,
    filename: str,
    phase: str,
    failure_class: str | None,
) -> None:
    stage = GROUP_TABLE_STAGE[filename]
    _assert_unique_grid(
        frame,
        ("geometry", "object", "stage"),
        {(geometry, object_name, stage) for geometry in GEOMETRIES for object_name in OBJECTS},
        name=f"{phase}/{filename}",
    )
    statistics = np.asarray(archive["group_statistics"], dtype=np.float64)
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            row = frame[
                (frame["geometry"] == geometry)
                & (frame["object"] == object_name)
                & (frame["stage"] == stage)
            ].iloc[0]
            gate_pass = _strict_bool(row["gate_pass"], label=f"{phase}/{filename}.gate_pass")
            if failure_class is not None:
                if row["status"] != "UNASSESSED" or gate_pass:
                    raise ReportingContractError(
                        f"{phase}/{filename} failure placeholder must be UNASSESSED/gate_pass=False"
                    )
                if int(row["replicates"]) != 0 or int(row["exceedances"]) != 0:
                    raise ReportingContractError(f"{phase}/{filename} failure counts must be zero")
                for column in ("observed", "null_median", "effect", "p_value"):
                    if not pd.isna(row[column]):
                        raise ReportingContractError(
                            f"{phase}/{filename} failure placeholder {column} must be NA"
                        )
                continue
            if row["status"] != "PASS" or not gate_pass:
                raise ReportingContractError(
                    f"{phase}/{filename} completed null row must be PASS/gate_pass=True"
                )
            null = statistics[geometry_index, object_index]
            observed = float(row["observed"])
            null_median = float(np.median(null))
            exceedances = int(np.count_nonzero(null >= observed))
            p_value = float((1 + exceedances) / (len(null) + 1))
            effect = observed - null_median
            checks = {
                "null_median": null_median,
                "effect": effect,
                "p_value": p_value,
            }
            for column, expected in checks.items():
                if not np.isclose(float(row[column]), expected, rtol=1e-12, atol=1e-12):
                    raise ReportingContractError(
                        f"{phase}/{filename}.{column} disagrees with saved null NPZ"
                    )
            if int(row["exceedances"]) != exceedances or int(row["replicates"]) != len(null):
                raise ReportingContractError(f"{phase}/{filename} null counts mismatch")


def _validate_object_archive(
    archive: Mapping[str, np.ndarray], *, filename: str
) -> None:
    value_key = (
        "marginal_means"
        if "marginal" in filename
        else "class_means"
        if "class_means" in filename
        else "matrices"
    )
    expected_keys = {value_key, "geometries", "subjects", "splits", "classes"}
    if set(archive) != expected_keys:
        raise ReportingContractError(
            f"{filename} NPZ keys mismatch: expected {sorted(expected_keys)}, got {sorted(archive)}"
        )
    geometries = np.asarray(archive["geometries"]).astype(str)
    expected_geometry = (
        np.asarray(["AIRM"])
        if filename.startswith("airm_")
        else np.asarray(["LE"])
        if filename.startswith("le_")
        else np.asarray(GEOMETRIES)
    )
    if not np.array_equal(geometries, expected_geometry):
        raise ReportingContractError(f"{filename} geometry axis/order mismatch")
    if not np.array_equal(np.asarray(archive["subjects"]).astype(int), np.asarray(SUBJECTS)):
        raise ReportingContractError(f"{filename} subject axis/order mismatch")
    if not np.array_equal(np.asarray(archive["splits"]).astype(str), np.asarray(SPLITS)):
        raise ReportingContractError(f"{filename} split axis/order mismatch")
    if not np.array_equal(np.asarray(archive["classes"]).astype(str), np.asarray(CLASSES)):
        raise ReportingContractError(f"{filename} class axis/order mismatch")
    values = np.asarray(archive[value_key], dtype=np.float64)
    if "marginal" in filename:
        expected_shape = (9, 3, 22, 22)
    elif "class_means" in filename:
        expected_shape = (9, 3, 4, 22, 22)
    else:
        expected_shape = (2, 9, 3, 4, 4)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ReportingContractError(
            f"{filename} data shape/finite mismatch: expected {expected_shape}, got {values.shape}"
        )


def _archives_equal(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        left_array = np.asarray(left[key])
        right_array = np.asarray(right[key])
        if left_array.shape != right_array.shape:
            return False
        if (
            np.issubdtype(left_array.dtype, np.inexact)
            and np.issubdtype(right_array.dtype, np.inexact)
        ):
            equal = np.array_equal(left_array, right_array, equal_nan=True)
        else:
            equal = np.array_equal(left_array, right_array)
        if not equal:
            return False
    return True


def _validate_phase_object_table_bindings(
    *,
    tables: Mapping[str, pd.DataFrame],
    object_archives: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    """Bind every saved mean, D/G matrix and derived geometry CSV row."""

    d_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

    def require_close(observed: Any, expected: Any, *, label: str) -> None:
        observed_array = np.asarray(observed, dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        if observed_array.shape != expected_array.shape or not np.allclose(
            observed_array,
            expected_array,
            rtol=5e-12,
            atol=5e-14,
            equal_nan=True,
        ):
            raise ReportingContractError(f"phase artifact binding mismatch: {label}")

    for phase in PHASES:
        archives = object_archives[phase]
        saved_D = np.asarray(archives["D_matrices.npz"]["matrices"], dtype=np.float64)
        saved_G = np.asarray(archives["G_matrices.npz"]["matrices"], dtype=np.float64)
        for geometry_index, geometry in enumerate(GEOMETRIES):
            stem = geometry.lower()
            marginal = np.asarray(
                archives[f"{stem}_marginal_means.npz"]["marginal_means"],
                dtype=np.float64,
            )
            class_means = np.asarray(
                archives[f"{stem}_class_means.npz"]["class_means"],
                dtype=np.float64,
            )
            for name, values in (("marginal", marginal), ("class", class_means)):
                symmetry = np.linalg.norm(
                    values - np.swapaxes(values, -1, -2), axis=(-2, -1)
                ) / np.maximum(
                    np.linalg.norm(values, axis=(-2, -1)),
                    np.finfo(np.float64).tiny,
                )
                eigenvalues = np.linalg.eigvalsh(values)
                condition_numbers = eigenvalues[..., -1] / eigenvalues[..., 0]
                if (
                    not np.isfinite(values).all()
                    or np.any(symmetry > 1e-12)
                    or np.any(eigenvalues <= 0.0)
                    or np.any(condition_numbers > 1e12)
                ):
                    raise ReportingContractError(
                        f"{phase} {geometry} {name} mean NPZ is not finite SPD"
                    )
            for subject_index, subject in enumerate(SUBJECTS):
                for split_index, split in enumerate(SPLITS):
                    local_marginal = marginal[subject_index, split_index]
                    local_classes = class_means[subject_index, split_index]
                    try:
                        if geometry == "AIRM":
                            expected_D = airm_distance_matrix(local_classes)
                            expected_G = airm_gram_matrices(
                                local_marginal, local_classes
                            )[0]
                        else:
                            expected_D = le_distance_matrix(local_classes)
                            expected_G = le_gram_matrix(
                                local_marginal, local_classes
                            )[0]
                    except (ValueError, np.linalg.LinAlgError) as error:
                        raise ReportingContractError(
                            f"cannot recompute {phase} {geometry} D/G from saved means"
                        ) from error
                    key = f"{phase}/{geometry}/S{subject}/{split}"
                    require_close(
                        saved_D[geometry_index, subject_index, split_index],
                        expected_D,
                        label=f"{key}/D mean-to-object",
                    )
                    require_close(
                        saved_G[geometry_index, subject_index, split_index],
                        expected_G,
                        label=f"{key}/G mean-to-object",
                    )
                    raw_D = np.asarray(
                        [expected_D[left, right] for left, right in d_pairs],
                        dtype=np.float64,
                    )
                    raw_G = np.asarray(svec(expected_G), dtype=np.float64)
                    for filename, raw in (
                        ("D_shape_vectors.csv", raw_D),
                        ("G_shape_vectors.csv", raw_G),
                    ):
                        row = tables[filename][
                            (tables[filename]["phase"] == phase)
                            & (tables[filename]["geometry"] == geometry)
                            & (pd.to_numeric(tables[filename]["subject"]) == subject)
                            & (tables[filename]["split"] == split)
                        ]
                        if len(row) != 1:
                            raise ReportingContractError(f"{key}/{filename} is not unique")
                        dimension = len(raw)
                        norm = float(np.linalg.norm(raw))
                        unit = raw / norm
                        require_close(
                            row[[f"raw_{index}" for index in range(dimension)]].iloc[0],
                            raw,
                            label=f"{key}/{filename}/raw",
                        )
                        require_close(
                            [row.iloc[0]["shape_norm"]],
                            [norm],
                            label=f"{key}/{filename}/norm",
                        )
                        require_close(
                            row[[f"z_{index}" for index in range(dimension)]].iloc[0],
                            unit,
                            label=f"{key}/{filename}/unit",
                        )
                    scale = tables["absolute_geometry_scales.csv"]
                    scale_row = scale[
                        (scale["phase"] == phase)
                        & (scale["geometry"] == geometry)
                        & (pd.to_numeric(scale["subject"]) == subject)
                        & (scale["split"] == split)
                    ]
                    if len(scale_row) != 1:
                        raise ReportingContractError(f"{key}/scale row is not unique")
                    radii = np.sqrt(np.maximum(np.diag(expected_G), 0.0))
                    require_close(
                        scale_row[
                            [
                                "D_shape_norm",
                                "G_shape_norm",
                                "D_upper_mean",
                                "D_upper_max",
                                "class_radius_mean",
                                "class_radius_max",
                            ]
                        ].iloc[0],
                        [
                            np.linalg.norm(raw_D),
                            np.linalg.norm(raw_G),
                            np.mean(raw_D),
                            np.max(raw_D),
                            np.mean(radii),
                            np.max(radii),
                        ],
                        label=f"{key}/absolute scales",
                    )
                    radius_table = tables["radius_angle_summary.csv"]
                    for left, right in d_pairs:
                        radius_row = radius_table[
                            (radius_table["phase"] == phase)
                            & (radius_table["geometry"] == geometry)
                            & (pd.to_numeric(radius_table["subject"]) == subject)
                            & (radius_table["split"] == split)
                            & (radius_table["class_left"] == CLASSES[left])
                            & (radius_table["class_right"] == CLASSES[right])
                        ]
                        denominator = radii[left] * radii[right]
                        cosine = (
                            expected_G[left, right] / denominator
                            if denominator > 0.0
                            else np.nan
                        )
                        angle = (
                            np.arccos(np.clip(cosine, -1.0, 1.0))
                            if np.isfinite(cosine)
                            else np.nan
                        )
                        require_close(
                            radius_row[
                                [
                                    "radius_left",
                                    "radius_right",
                                    "cosine",
                                    "angle_radians",
                                ]
                            ].iloc[0],
                            [radii[left], radii[right], cosine, angle],
                            label=f"{key}/radius-angle/{left}-{right}",
                        )


def _phase_table_view(
    tables: Mapping[str, pd.DataFrame], phase: str
) -> dict[str, pd.DataFrame]:
    return {
        filename: frame[frame["phase"].astype(str) == phase].reset_index(drop=True)
        for filename, frame in tables.items()
    }


def _failure_precedence(values: Sequence[str | None]) -> str | None:
    present = {value for value in values if value is not None}
    for value in ("data", "degenerate", "numerical"):
        if value in present:
            return value
    return None


def _build_stage_operands(
    snapshot_tables: Mapping[str, Mapping[str, pd.DataFrame]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for filename, stage in GROUP_TABLE_STAGE.items():
        discovery = snapshot_tables["discovery"][filename]
        confirmatory = snapshot_tables["confirmatory"][filename]
        for geometry in GEOMETRIES:
            for object_name in OBJECTS:
                left = discovery[
                    (discovery["geometry"].astype(str) == geometry)
                    & (discovery["object"].astype(str) == object_name)
                ].iloc[0]
                right = confirmatory[
                    (confirmatory["geometry"].astype(str) == geometry)
                    & (confirmatory["object"].astype(str) == object_name)
                ].iloc[0]
                rows.append(
                    {
                        "geometry": geometry,
                        "object": object_name,
                        "stage": stage,
                        "discovery_effect": left["effect"],
                        "confirmatory_effect": right["effect"],
                        "confirmatory_p": right["p_value"],
                    }
                )
    return pd.DataFrame(rows)


def _validate_producer_decisions(
    inputs: ReportingInputs, verdicts: ConditionalVerdicts
) -> None:
    comparison = inputs.tables["discovery_confirmatory_comparison.csv"]
    _assert_unique_grid(
        comparison,
        ("geometry", "object", "stage"),
        {
            (geometry, object_name, stage)
            for geometry in GEOMETRIES
            for object_name in OBJECTS
            for stage in STAGES
        },
        name="discovery_confirmatory_comparison.csv",
    )
    operands = verdicts.stage_operands.set_index(["geometry", "object", "stage"])
    stage_table = {
        stage: filename for filename, stage in GROUP_TABLE_STAGE.items()
    }
    for row in comparison.itertuples(index=False):
        key = (str(row.geometry), str(row.object), str(row.stage))
        expected = operands.loc[key]
        for producer_column, operand_column in (
            ("discovery_effect", "discovery_effect"),
            ("confirmatory_effect", "confirmatory_effect"),
            ("confirmatory_p", "confirmatory_p"),
        ):
            observed_value = getattr(row, producer_column)
            expected_value = expected[operand_column]
            if pd.isna(observed_value) and pd.isna(expected_value):
                continue
            if not np.isclose(
                float(observed_value), float(expected_value), rtol=1e-12, atol=1e-12
            ):
                raise ReportingContractError(
                    f"comparison {key} {producer_column} differs from authoritative summary"
                )
        authoritative = inputs.tables[stage_table[key[2]]]
        for phase, producer_column in (
            ("discovery", "discovery_observed"),
            ("confirmatory", "confirmatory_observed"),
        ):
            group_row = authoritative[
                (authoritative["phase"] == phase)
                & (authoritative["geometry"] == key[0])
                & (authoritative["object"] == key[1])
                & (authoritative["stage"] == key[2])
            ]
            if len(group_row) != 1:
                raise ReportingContractError(
                    f"comparison {key} {producer_column} differs from authoritative summary"
                )
            producer_value = getattr(row, producer_column)
            group_value = group_row.iloc[0]["observed"]
            if pd.isna(producer_value) and pd.isna(group_value):
                continue
            if not np.isclose(
                float(producer_value), float(group_value), rtol=1e-12, atol=1e-12
            ):
                raise ReportingContractError(
                    f"comparison {key} {producer_column} differs from authoritative summary"
                )

    robustness = inputs.tables["airm_le_robustness.csv"]
    _assert_unique_grid(
        robustness,
        ("object", "stage"),
        {(object_name, stage) for object_name in OBJECTS for stage in STAGES},
        name="airm_le_robustness.csv",
    )
    comparison_index = comparison.set_index(["geometry", "object", "stage"])
    for row in robustness.itertuples(index=False):
        object_name, stage = str(row.object), str(row.stage)
        expected_values = {
            "airm_discovery_effect": comparison_index.loc[
                ("AIRM", object_name, stage), "discovery_effect"
            ],
            "airm_confirmatory_effect": comparison_index.loc[
                ("AIRM", object_name, stage), "confirmatory_effect"
            ],
            "le_discovery_effect": comparison_index.loc[
                ("LE", object_name, stage), "discovery_effect"
            ],
            "le_confirmatory_effect": comparison_index.loc[
                ("LE", object_name, stage), "confirmatory_effect"
            ],
        }
        for column, expected_value in expected_values.items():
            observed_value = getattr(row, column)
            if pd.isna(observed_value) and pd.isna(expected_value):
                continue
            if not np.isclose(
                float(observed_value), float(expected_value), rtol=1e-12, atol=1e-12
            ):
                raise ReportingContractError(
                    f"robustness {(object_name, stage)} {column} differs from comparison"
                )

    hypothesis = inputs.tables["hypothesis_chain_status.csv"].copy()
    _assert_unique_grid(
        hypothesis,
        ("geometry", "object", "stage"),
        {
            (geometry, object_name, stage)
            for geometry in GEOMETRIES
            for object_name in OBJECTS
            for stage in STAGES
        },
        name="hypothesis_chain_status.csv",
    )
    calculated = verdicts.stage_status.set_index(["geometry", "object", "stage"])
    for row in hypothesis.itertuples(index=False):
        key = (str(row.geometry), str(row.object), str(row.stage))
        expected = calculated.loc[key]
        if str(row.stage_status) != str(expected["status"]):
            raise ReportingContractError(f"producer stage status mismatch at {key}")
        if _strict_bool(row.eligible, label="hypothesis.eligible") != bool(expected["eligible"]):
            raise ReportingContractError(f"producer eligibility mismatch at {key}")
        if _strict_bool(row.criterion_pass, label="hypothesis.criterion_pass") != bool(
            expected["criterion_pass"]
        ):
            raise ReportingContractError(f"producer criterion mismatch at {key}")
        if _strict_bool(row.chain_pass, label="hypothesis.chain_pass") != verdicts.chain_pass[
            (key[0], key[1])
        ]:
            raise ReportingContractError(f"producer chain status mismatch at {key}")

    decision = inputs.root_json["confirmatory_decision.json"]
    _verify_common_provenance(
        decision,
        name="confirmatory_decision.json",
        config=inputs.config,
        config_sha256=inputs.config_sha256,
        protocol_sha256=inputs.protocol_sha256,
    )
    expected_decision = {
        "phase": "combined",
        "session": "0train+1test",
        "terminal_decision": verdicts.terminal_decision,
        "le_robustness_label": verdicts.le_robustness_label,
    }
    for key, value in expected_decision.items():
        if decision.get(key) != value:
            raise ReportingContractError(
                f"confirmatory_decision.json {key} mismatch: expected {value!r}, "
                f"observed {decision.get(key)!r}"
            )
    if str(decision.get("code_commit", "")) != str(inputs.unlock["code_commit"]):
        raise ReportingContractError("confirmatory decision code_commit mismatch")
    if verdicts.failure_class is None:
        expected_status = "PASS"
    else:
        expected_status = "UNASSESSED"
    if decision.get("status") != expected_status:
        raise ReportingContractError("confirmatory decision status mismatch")


def _validate_label_null_dry_run_authority(
    dry_run: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """Require official scalar pyRiemann authority while keeping batch audit descriptive."""

    crosscheck = dry_run.get("airm_scalar_crosscheck")
    if not isinstance(crosscheck, Mapping):
        raise ReportingContractError("label-null dry-run lacks airm_scalar_crosscheck")
    exact = {
        "replicate_index": 0,
        "groups_checked": 72,
        "official_scalar_authoritative": True,
        "authoritative_solver": "pyriemann.geometry.mean.mean_riemann",
        "official_all_72_pass": True,
        "official_warning_count": 0,
        "passed": True,
    }
    for key, expected in exact.items():
        if crosscheck.get(key) != expected:
            raise ReportingContractError(
                f"label-null dry-run authority field {key!r} mismatch"
            )
    indices = crosscheck.get("canonical_flat_group_indices")
    if indices != list(range(72)):
        raise ReportingContractError(
            "label-null dry-run canonical AIRM group order mismatch"
        )
    try:
        errors = np.asarray(crosscheck.get("relative_errors"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ReportingContractError(
            "label-null dry-run relative_errors are invalid"
        ) from error
    if errors.shape != (72,) or not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise ReportingContractError(
            "label-null dry-run must contain 72 finite nonnegative relative errors"
        )
    numeric_keys = (
        "maximum_relative_error",
        "tolerance",
        "maximum_official_karcher_residual",
        "maximum_batched_post_karcher_residual",
    )
    numeric: dict[str, float] = {}
    for key in numeric_keys:
        try:
            numeric[key] = float(crosscheck[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ReportingContractError(
                f"label-null dry-run crosscheck {key!r} is invalid"
            ) from error
        if not np.isfinite(numeric[key]) or numeric[key] < 0.0:
            raise ReportingContractError(
                f"label-null dry-run crosscheck {key!r} is invalid"
            )
    if numeric["tolerance"] <= 0.0:
        raise ReportingContractError("label-null dry-run batch tolerance is not positive")
    if not np.isclose(
        numeric["maximum_relative_error"],
        float(np.max(errors)),
        rtol=1e-13,
        atol=1e-15,
    ):
        raise ReportingContractError(
            "label-null dry-run maximum_relative_error disagrees with relative_errors"
        )
    batch_equivalent = _strict_bool(
        crosscheck.get("batched_equivalent_within_tolerance"),
        label="label-null dry-run batched_equivalent_within_tolerance",
    )
    if batch_equivalent != bool(np.all(errors <= numeric["tolerance"])):
        raise ReportingContractError(
            "label-null dry-run batched equivalence flag disagrees with errors"
        )
    # The batched implementation is a speed audit only.  It is intentionally
    # not a scientific gate; every scientific null mean must be the official
    # scalar pyRiemann result checked below.
    if numeric["maximum_official_karcher_residual"] > float(
        config["hard_gates"]["airm_karcher_residual_max"]
    ):
        raise ReportingContractError(
            "label-null dry-run official AIRM Karcher residual exceeds the hard gate"
        )


def _validate_frozen_root_provenance(
    *,
    root: Path,
    output_root: Path,
    config_path: Path,
    protocol_path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    protocol_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    names = ("manifest.json", "git_provenance.json", "environment.json")
    missing = [name for name in names if not (output_root / name).is_file()]
    if missing:
        raise ReportingContractError(f"missing frozen root provenance: {missing}")
    payloads = {
        name: _read_json(output_root / name, name=name) for name in names
    }
    for name, payload in payloads.items():
        _verify_common_provenance(
            payload,
            name=name,
            config=config,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
    manifest = payloads["manifest.json"]
    expected_manifest = {
        "phase": "PROTOCOL_FROZEN",
        "branch": str(config["protocol"]["branch"]),
        "base_commit": str(config["protocol"]["base_commit"]),
        "confirmatory_designation": "STRICT_CONFIRMATORY",
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ReportingContractError(f"manifest.json {key} mismatch")
    frozen_config_path = _resolve_inside(
        root, str(manifest.get("frozen_config", "")), label="manifest frozen_config"
    )
    frozen_protocol_path = _resolve_inside(
        root, str(manifest.get("frozen_protocol", "")), label="manifest frozen_protocol"
    )
    if frozen_config_path.parent != output_root / "protocol":
        raise ReportingContractError("frozen config is outside the protocol output directory")
    if frozen_protocol_path.parent != output_root / "protocol":
        raise ReportingContractError("frozen protocol is outside the protocol output directory")
    if frozen_config_path.read_bytes() != config_path.read_bytes():
        raise ReportingContractError("frozen config differs byte-for-byte from live config")
    if frozen_protocol_path.read_bytes() != protocol_path.read_bytes():
        raise ReportingContractError("frozen protocol differs byte-for-byte from live protocol")
    return payloads


def _read_and_validate_dry_run(
    *,
    output_root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    protocol_sha256: str,
    code_commit: str,
) -> Mapping[str, Any]:
    path = output_root / "protocol" / "label_null_dry_run.json"
    dry_run = _read_json(path, name="label_null_dry_run.json")
    _verify_common_provenance(
        dry_run,
        name="label_null_dry_run.json",
        config=config,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )
    expected = {
        "code_commit": code_commit,
        "phase": "discovery",
        "session": "0train",
        "status": "DRY_RUN_ONLY",
        "scientific_output_written": False,
    }
    for key, value in expected.items():
        if dry_run.get(key) != value:
            raise ReportingContractError(f"label-null dry-run {key} mismatch")
    if int(dry_run.get("replicates", 0)) < 1:
        raise ReportingContractError("label-null dry-run did not time any replicate")
    for key in (
        "elapsed_seconds",
        "seconds_per_replicate_wall",
        "estimated_official_wall_seconds_linear",
    ):
        value = dry_run.get(key)
        if (
            not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ReportingContractError(f"label-null dry-run {key} is invalid")
    _validate_label_null_dry_run_authority(dry_run, config)
    return dry_run


def _load_failure_reporting_inputs(
    *,
    root: Path,
    output_root: Path,
    config_path: Path,
    protocol_path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    protocol_sha256: str,
    decision: Mapping[str, Any],
) -> ReportingInputs:
    """Validate the atomic minimal failure contract without scientific artifacts."""

    if set(decision) != FAILURE_DECISION_KEYS:
        raise ReportingContractError("UNASSESSED decision schema mismatch")
    if decision.get("schema_version") != "conditional-unassessed-v1":
        raise ReportingContractError("UNASSESSED decision schema_version mismatch")
    _verify_common_provenance(
        decision,
        name="confirmatory_decision.json",
        config=config,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )
    phase = str(decision.get("phase", ""))
    failure_class = str(decision.get("failure_class", ""))
    if phase not in PHASES or failure_class not in FAILURE_TERMINAL:
        raise ReportingContractError("UNASSESSED decision phase/failure_class mismatch")
    expected_session = (
        str(config["dataset"]["discovery_session"])
        if phase == "discovery"
        else str(config["dataset"]["confirmatory_session"])
    )
    exact = {
        "status": "UNASSESSED",
        "session": expected_session,
        "terminal_decision": FAILURE_TERMINAL[failure_class],
        "scientific_nulls_executed": False,
        "downstream_phase_permitted": False,
        "le_robustness_label": "UNASSESSED",
        "chains": {
            geometry: {object_name: None for object_name in OBJECTS}
            for geometry in GEOMETRIES
        },
    }
    for key, value in exact.items():
        if decision.get(key) != value:
            raise ReportingContractError(f"UNASSESSED decision {key} mismatch")
    for key in ("reason_code", "reason"):
        if not isinstance(decision.get(key), str) or not str(decision[key]).strip():
            raise ReportingContractError(f"UNASSESSED decision {key} is empty")
    code_commit = str(decision.get("code_commit", ""))
    if len(code_commit) != 40 or any(value not in "0123456789abcdef" for value in code_commit):
        raise ReportingContractError("UNASSESSED decision code_commit is invalid")

    failure_path = _resolve_inside(
        root, str(decision.get("failure_manifest", "")), label="failure_manifest"
    )
    expected_path = output_root / "protocol" / f"{phase}_failure_manifest.json"
    if failure_path != expected_path or not failure_path.is_file() or failure_path.is_symlink():
        raise ReportingContractError("UNASSESSED failure manifest path mismatch")
    if _sha256_file(failure_path) != str(decision.get("failure_manifest_sha256", "")):
        raise ReportingContractError("UNASSESSED failure manifest SHA-256 mismatch")
    manifest = _read_json(failure_path, name=failure_path.name)
    if set(manifest) != FAILURE_MANIFEST_KEYS:
        raise ReportingContractError("UNASSESSED failure manifest schema mismatch")
    for key in FAILURE_MANIFEST_KEYS:
        if manifest.get(key) != decision.get(key):
            raise ReportingContractError(f"failure manifest/decision mismatch at {key}")

    root_json = _validate_frozen_root_provenance(
        root=root,
        output_root=output_root,
        config_path=config_path,
        protocol_path=protocol_path,
        config=config,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )
    root_json = {
        **root_json,
        "confirmatory_decision.json": decision,
        failure_path.name: manifest,
    }
    unlock: Mapping[str, Any] = {}
    unlock_path = output_root / "confirmatory_unlock.json"
    if phase == "discovery":
        if unlock_path.exists():
            raise ReportingContractError(
                "discovery failure must precede confirmatory unlock"
            )
        confirmatory_roots = (
            output_root / "tables" / "confirmatory",
            output_root / "objects" / "confirmatory",
            output_root / "nulls" / "confirmatory",
        )
        if any(path.exists() and any(path.iterdir()) for path in confirmatory_roots):
            raise ReportingContractError(
                "discovery failure has confirmatory artifact access traces"
            )
    else:
        unlock = _read_json(unlock_path, name="confirmatory_unlock.json")
        _verify_unlock(
            unlock,
            repo_root=root,
            config=config,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
        if str(unlock.get("code_commit", "")) != code_commit:
            raise ReportingContractError(
                "confirmatory failure code_commit differs from locked code snapshot"
            )
        _read_and_validate_dry_run(
            output_root=output_root,
            config=config,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
            code_commit=code_commit,
        )

    for path in (
        *(output_root / "nulls" / phase / name for name in REQUIRED_NULLS),
        *(output_root / "nulls" / name for name in REQUIRED_NULLS),
    ):
        if path.exists():
            raise ReportingContractError(
                "UNASSESSED failure contradicts scientific_nulls_executed=false"
            )
    return ReportingInputs(
        repo_root=root,
        output_root=output_root,
        config_path=config_path,
        config=config,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        root_json=root_json,
        unlock=unlock,
        tables={},
        snapshot_tables={},
        object_archives={},
        null_archives={},
        failure_class=failure_class,
        failure_details=(
            f"{phase}:{decision['reason_code']}:{decision['reason']}",
        ),
        failure_manifest=manifest,
    )


def load_and_validate_reporting_inputs(
    config_path: str | Path, repo_root: str | Path
) -> ReportingInputs:
    """Load every frozen prerequisite and fail closed before any report write."""

    root = Path(repo_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ReportingContractError(f"cannot load frozen config: {path}") from error
    if not isinstance(config, dict):
        raise ReportingContractError("frozen config must be a mapping")
    config_sha256 = _sha256_file(path)
    protocol_path = _resolve_inside(
        root, config["protocol"]["protocol_path"], label="protocol_path"
    )
    protocol_sha256 = _sha256_file(protocol_path)
    if protocol_sha256 != str(config["protocol"]["protocol_sha256"]):
        raise ReportingContractError("frozen protocol SHA-256 mismatch")
    configured = config["outputs"]
    exact_config_lists = {
        "required_root_files": REQUIRED_ROOT_FILES,
        "required_tables": REQUIRED_TABLES,
        "required_objects": REQUIRED_OBJECTS,
        "required_nulls": REQUIRED_NULLS,
        "figure_stems": FIGURE_STEMS,
    }
    for key, expected in exact_config_lists.items():
        observed = tuple(str(item) for item in configured[key])
        if observed != expected:
            raise ReportingContractError(
                f"config outputs.{key} differs from frozen protocol: {observed}"
            )
    if str(configured["report_title"]) != REPORT_TITLE:
        raise ReportingContractError("configured report title mismatch")
    output_root = _resolve_inside(
        root, config["project"]["output_dir"], label="project.output_dir"
    )
    if output_root.name != "bnci2014_001_conditional_geometry_v1":
        raise ReportingContractError("output root is not the frozen Conditional v1 namespace")

    decision_path = output_root / str(configured["decision_file"])
    if decision_path.is_file():
        early_decision = _read_json(
            decision_path, name="confirmatory_decision.json"
        )
        if early_decision.get("schema_version") == "conditional-unassessed-v1":
            return _load_failure_reporting_inputs(
                root=root,
                output_root=output_root,
                config_path=path,
                protocol_path=protocol_path,
                config=config,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                decision=early_decision,
            )

    missing_root = [name for name in REQUIRED_ROOT_FILES if not (output_root / name).is_file()]
    if not decision_path.is_file():
        missing_root.append(str(configured["decision_file"]))
    if missing_root:
        raise ReportingContractError(f"missing required root artifacts: {missing_root}")
    root_json: dict[str, Mapping[str, Any]] = {
        name: _read_json(output_root / name, name=name) for name in REQUIRED_ROOT_FILES
    }
    root_json["confirmatory_decision.json"] = _read_json(
        decision_path, name="confirmatory_decision.json"
    )
    for name in ("manifest.json", "git_provenance.json", "environment.json"):
        _verify_common_provenance(
            root_json[name],
            name=name,
            config=config,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
    manifest = root_json["manifest.json"]
    expected_manifest = {
        "phase": "PROTOCOL_FROZEN",
        "branch": str(config["protocol"]["branch"]),
        "base_commit": str(config["protocol"]["base_commit"]),
        "confirmatory_designation": "STRICT_CONFIRMATORY",
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ReportingContractError(f"manifest.json {key} mismatch")
    frozen_config_path = _resolve_inside(
        root, str(manifest.get("frozen_config", "")), label="manifest frozen_config"
    )
    frozen_protocol_path = _resolve_inside(
        root, str(manifest.get("frozen_protocol", "")), label="manifest frozen_protocol"
    )
    if frozen_config_path.parent != output_root / "protocol":
        raise ReportingContractError("frozen config is outside the protocol output directory")
    if frozen_protocol_path.parent != output_root / "protocol":
        raise ReportingContractError("frozen protocol is outside the protocol output directory")
    if frozen_config_path.read_bytes() != path.read_bytes():
        raise ReportingContractError("frozen config differs byte-for-byte from live config")
    if frozen_protocol_path.read_bytes() != protocol_path.read_bytes():
        raise ReportingContractError("frozen protocol differs byte-for-byte from live protocol")
    unlock = root_json["confirmatory_unlock.json"]
    _verify_unlock(
        unlock,
        repo_root=root,
        config=config,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )
    discovery_record_paths = {
        str(record["path"])
        for record in unlock["discovery_snapshot"]["files"]
    }
    required_discovery_paths = {
        (output_root / "tables" / "discovery" / name).relative_to(root).as_posix()
        for name in REQUIRED_TABLES
        if name not in POST_CONFIRMATORY_TABLES
    } | {
        (output_root / "objects" / "discovery" / name).relative_to(root).as_posix()
        for name in REQUIRED_OBJECTS
    } | {
        (output_root / "nulls" / "discovery" / name).relative_to(root).as_posix()
        for name in REQUIRED_NULLS
    }
    if not required_discovery_paths.issubset(discovery_record_paths):
        missing = sorted(required_discovery_paths - discovery_record_paths)
        raise ReportingContractError(
            f"unlock discovery snapshot omits required artifacts: {missing}"
        )
    code_record_paths = {
        str(record["path"]) for record in unlock["code_snapshot"]["files"]
    }
    required_code_paths = {
        path.relative_to(root).as_posix(),
        protocol_path.relative_to(root).as_posix(),
        "src/reporting_conditional_v1.py",
        "scripts/27_conditional_geometry_report.py",
    }
    if not required_code_paths.issubset(code_record_paths):
        raise ReportingContractError(
            "unlock code snapshot does not include config/protocol/reporting implementation"
        )
    code_commit = str(unlock["code_commit"])
    _read_and_validate_dry_run(
        output_root=output_root,
        config=config,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        code_commit=code_commit,
    )

    tables_dir = output_root / "tables"
    missing_tables = [name for name in REQUIRED_TABLES if not (tables_dir / name).is_file()]
    if missing_tables:
        raise ReportingContractError(f"missing required combined tables: {missing_tables}")
    tables = {name: _read_csv(tables_dir / name, name=name) for name in REQUIRED_TABLES}
    for name, frame in tables.items():
        tables[name] = _validate_common_table(
            frame,
            filename=name,
            config=config,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
            code_commit=code_commit,
            expected_phase=None,
        )

    snapshot_tables: dict[str, dict[str, pd.DataFrame]] = {}
    phase_table_names = tuple(
        name for name in REQUIRED_TABLES if name not in POST_CONFIRMATORY_TABLES
    )
    for phase in PHASES:
        directory = tables_dir / phase
        missing = [name for name in phase_table_names if not (directory / name).is_file()]
        if missing:
            raise ReportingContractError(f"missing {phase} snapshot tables: {missing}")
        snapshot_tables[phase] = {}
        for name in phase_table_names:
            frame = _read_csv(directory / name, name=f"{phase}/{name}")
            snapshot_tables[phase][name] = _validate_common_table(
                frame,
                filename=name,
                config=config,
                config_sha256=config_sha256,
                protocol_sha256=protocol_sha256,
                code_commit=code_commit,
                expected_phase=phase,
            )
    _validate_snapshot_table_identity(tables, snapshot_tables)

    phase_failures: dict[str, str | None] = {}
    phase_failure_details: dict[str, tuple[str, ...]] = {}
    for phase in PHASES:
        phase_tables = _phase_table_view(tables, phase)
        failure, details = _validate_gate_and_geometry_tables(phase_tables, config)
        phase_failures[phase] = failure
        phase_failure_details[phase] = details
    failure_class = _failure_precedence(tuple(phase_failures.values()))
    failure_details = tuple(
        f"{phase}:{detail}"
        for phase in PHASES
        for detail in phase_failure_details[phase]
    )
    if failure_class is None:
        _validate_success_grids(tables, config)

    object_archives: dict[str, dict[str, np.ndarray]] = {}
    for scope in (*PHASES, "root"):
        directory = output_root / "objects" if scope == "root" else output_root / "objects" / scope
        missing = [name for name in REQUIRED_OBJECTS if not (directory / name).is_file()]
        if missing:
            raise ReportingContractError(f"missing {scope} object archives: {missing}")
        object_archives[scope] = {}
        for name in REQUIRED_OBJECTS:
            archive = _read_npz(directory / name, name=f"{scope}/{name}")
            _validate_object_archive(archive, filename=name)
            object_archives[scope][name] = archive
            if scope == "root" and not _archives_equal(
                archive, object_archives["confirmatory"][name]
            ):
                raise ReportingContractError(
                    f"root object {name} must be an exact confirmatory snapshot copy"
                )
    if failure_class is None:
        _validate_phase_object_table_bindings(
            tables=tables,
            object_archives=object_archives,
        )

    null_archives: dict[str, dict[str, np.ndarray]] = {}
    for scope in (*PHASES, "root"):
        directory = output_root / "nulls" if scope == "root" else output_root / "nulls" / scope
        missing = [name for name in REQUIRED_NULLS if not (directory / name).is_file()]
        if missing:
            raise ReportingContractError(f"missing {scope} null archives: {missing}")
        null_archives[scope] = {}
        local_failure = phase_failures["confirmatory"] if scope == "root" else phase_failures[scope]
        for name in REQUIRED_NULLS:
            archive = _read_npz(directory / name, name=f"{scope}/{name}")
            _validate_null_archive(archive, filename=name, failure_class=local_failure)
            null_archives[scope][name] = archive
            if scope == "root" and not _archives_equal(
                archive, null_archives["confirmatory"][name]
            ):
                raise ReportingContractError(
                    f"root null {name} must be an exact confirmatory snapshot copy"
                )
    for phase in PHASES:
        for table_name, null_name in NULL_FILE_FOR_GROUP_TABLE.items():
            _validate_group_summary_against_null(
                snapshot_tables[phase][table_name],
                null_archives[phase][null_name],
                filename=table_name,
                phase=phase,
                failure_class=phase_failures[phase],
            )

    inputs = ReportingInputs(
        repo_root=root,
        output_root=output_root,
        config_path=path,
        config=config,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        root_json=root_json,
        unlock=unlock,
        tables=tables,
        snapshot_tables=snapshot_tables,
        object_archives=object_archives,
        null_archives=null_archives,
        failure_class=failure_class,
        failure_details=failure_details,
    )
    operands = _build_stage_operands(snapshot_tables)
    verdicts = compute_frozen_verdicts(
        operands,
        hard_gates_pass=failure_class is None,
        failure_class=failure_class,
        failure_details=failure_details,
    )
    _validate_producer_decisions(inputs, verdicts)
    return inputs


def build_figure_sources(
    inputs: ReportingInputs, verdicts: ConditionalVerdicts
) -> dict[str, pd.DataFrame]:
    """Build the exact ten plot-source tables from validated artifacts."""

    sources: dict[str, pd.DataFrame] = {}
    sources[FIGURE_STEMS[0]] = inputs.tables["within_subject_reliability.csv"].copy()

    null_rows: list[dict[str, Any]] = []
    for phase in PHASES:
        archive = inputs.null_archives[phase]["label_destruction_group_statistics.npz"]
        summary = inputs.snapshot_tables[phase]["label_destruction_group_summary.csv"]
        values = np.asarray(archive["group_statistics"], dtype=np.float64)
        for geometry_index, geometry in enumerate(GEOMETRIES):
            for object_index, object_name in enumerate(OBJECTS):
                selected = summary[
                    (summary["geometry"].astype(str) == geometry)
                    & (summary["object"].astype(str) == object_name)
                ].iloc[0]
                if values.shape[-1] == 0:
                    null_rows.append(
                        {
                            "phase": phase,
                            "session": "0train" if phase == "discovery" else "1test",
                            "geometry": geometry,
                            "object": object_name,
                            "replicate_index": pd.NA,
                            "null_group_statistic": np.nan,
                            "observed": selected["observed"],
                            "status": selected["status"],
                        }
                    )
                else:
                    for replicate, statistic in zip(
                        archive["replicate_indices"],
                        values[geometry_index, object_index],
                        strict=True,
                    ):
                        null_rows.append(
                            {
                                "phase": phase,
                                "session": "0train" if phase == "discovery" else "1test",
                                "geometry": geometry,
                                "object": object_name,
                                "replicate_index": int(replicate),
                                "null_group_statistic": float(statistic),
                                "observed": float(selected["observed"]),
                                "status": str(selected["status"]),
                            }
                        )
    sources[FIGURE_STEMS[1]] = pd.DataFrame(null_rows)
    sources[FIGURE_STEMS[2]] = inputs.tables[
        "unrelated_subject_derangement_summary.csv"
    ].copy()
    sources[FIGURE_STEMS[3]] = inputs.tables["cross_subject_shared_geometry.csv"].copy()

    for object_name, stem, filename in (
        ("D", FIGURE_STEMS[4], "D_matrices.npz"),
        ("G", FIGURE_STEMS[5], "G_matrices.npz"),
    ):
        rows: list[dict[str, Any]] = []
        for phase in PHASES:
            archive = inputs.object_archives[phase][filename]
            matrices = np.asarray(archive["matrices"], dtype=np.float64)
            # Frozen display selection: AIRM primary, full split F, every
            # subject.  The heatmap is the elementwise subject median.
            selected = matrices[0, :, 2]
            median = np.median(selected, axis=0)
            for subject_index, subject in enumerate(SUBJECTS):
                for row_index, row_class in enumerate(CLASSES):
                    for column_index, column_class in enumerate(CLASSES):
                        rows.append(
                            {
                                "phase": phase,
                                "session": "0train" if phase == "discovery" else "1test",
                                "geometry": "AIRM",
                                "split": "F",
                                "row_type": "subject",
                                "subject": subject,
                                "class_row": row_class,
                                "class_column": column_class,
                                "value": float(selected[subject_index, row_index, column_index]),
                            }
                        )
            for row_index, row_class in enumerate(CLASSES):
                for column_index, column_class in enumerate(CLASSES):
                    rows.append(
                        {
                            "phase": phase,
                            "session": "0train" if phase == "discovery" else "1test",
                            "geometry": "AIRM",
                            "split": "F",
                            "row_type": "subject_median",
                            "subject": pd.NA,
                            "class_row": row_class,
                            "class_column": column_class,
                            "value": float(median[row_index, column_index]),
                        }
                    )
        sources[stem] = pd.DataFrame(rows)

    sources[FIGURE_STEMS[6]] = inputs.tables[
        "oracle_permutation_all_24_scores.csv"
    ].loc[lambda frame: frame["geometry"].astype(str) == "AIRM"].copy()
    sources[FIGURE_STEMS[7]] = inputs.tables[
        "oracle_permutation_subject_summary.csv"
    ].loc[lambda frame: frame["geometry"].astype(str) == "AIRM"].copy()
    influence_source = inputs.tables["leave_one_subject_out_influence.csv"].loc[
        lambda frame: frame["geometry"].astype(str) == "AIRM"
    ].copy()
    influence_source["row_type"] = "subject"
    bootstrap_source = inputs.tables["subject_bootstrap_summary.csv"].loc[
        lambda frame: frame["geometry"].astype(str) == "AIRM"
    ].copy()
    bootstrap_source["row_type"] = "bootstrap"
    sources[FIGURE_STEMS[8]] = pd.concat(
        [influence_source, bootstrap_source], ignore_index=True, sort=False
    )

    effect_rows: list[dict[str, Any]] = []
    for filename, stage in GROUP_TABLE_STAGE.items():
        frame = inputs.tables[filename]
        for row in frame.itertuples(index=False):
            effect_rows.append(
                {
                    "phase": str(row.phase),
                    "session": str(row.session),
                    "geometry": str(row.geometry),
                    "object": str(row.object),
                    "stage": stage,
                    "observed": row.observed,
                    "null_median": row.null_median,
                    "effect": row.effect,
                    "p_value": row.p_value,
                    "status": str(row.status),
                }
            )
    sources[FIGURE_STEMS[9]] = pd.DataFrame(effect_rows)
    if tuple(sources) != FIGURE_STEMS:
        raise ReportingContractError("internal figure-source ordering differs from protocol")
    return sources


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, suffix=path.suffix, delete=False
        ) as handle:
            temporary = Path(handle.name)
        metadata: dict[str, Any]
        if path.suffix == ".pdf":
            metadata = {
                "Creator": "Conditional-Geometry Anatomy v1",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            }
        else:
            metadata = {"Software": "Conditional-Geometry Anatomy v1"}
        fig.savefig(
            temporary,
            format=path.suffix.lstrip("."),
            dpi=180,
            bbox_inches="tight",
            metadata=metadata,
        )
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _unavailable(ax: plt.Axes, text: str = "UNASSESSED — prerequisite gate failed") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _phase_color(phase: str) -> str:
    return "#1f77b4" if phase == "discovery" else "#d62728"


def _plot_figures(
    sources: Mapping[str, pd.DataFrame], figures_dir: Path
) -> tuple[Path, ...]:
    written: list[Path] = []

    # F1: subject-level split-half reliability, all fixed conditions.
    frame = sources[FIGURE_STEMS[0]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, object_name in zip(axes, OBJECTS, strict=True):
        part = frame[frame["object"].astype(str) == object_name]
        plotted = False
        for phase in PHASES:
            for geometry, marker in (("AIRM", "o"), ("LE", "s")):
                rows = part[
                    (part["phase"].astype(str) == phase)
                    & (part["geometry"].astype(str) == geometry)
                    & (part["status"].astype(str) == "PASS")
                ].sort_values("subject")
                if rows.empty:
                    continue
                ax.plot(
                    rows["subject"], rows["observed"], marker=marker,
                    color=_phase_color(phase), linestyle="-" if geometry == "AIRM" else "--",
                    label=f"{phase} {geometry}", alpha=0.85,
                )
                plotted = True
        if not plotted:
            _unavailable(ax)
        ax.set_title(f"{object_name} reliability")
        ax.set_xlabel("subject")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0.0, color="0.75", linewidth=0.8)
    axes[0].set_ylabel("cosine(A, B)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("Within-subject split-half reliability")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in (".png", ".pdf"):
        path = figures_dir / f"{FIGURE_STEMS[0]}{suffix}"
        _save_figure(fig, path)
        written.append(path)
    plt.close(fig)

    # F2: full saved R-null distributions, no result-selected bins/panels.
    frame = sources[FIGURE_STEMS[1]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for row_index, geometry in enumerate(GEOMETRIES):
        for column_index, object_name in enumerate(OBJECTS):
            ax = axes[row_index, column_index]
            part = frame[
                (frame["geometry"] == geometry) & (frame["object"] == object_name)
            ]
            plotted = False
            for phase in PHASES:
                rows = part[part["phase"] == phase]
                values = pd.to_numeric(rows["null_group_statistic"], errors="coerce").dropna()
                if values.empty:
                    continue
                ax.hist(values, bins=30, density=True, histtype="step", linewidth=1.2,
                        color=_phase_color(phase), label=f"{phase} null")
                observed = pd.to_numeric(rows["observed"], errors="coerce").dropna()
                if not observed.empty:
                    ax.axvline(float(observed.iloc[0]), color=_phase_color(phase), linestyle="--")
                plotted = True
            if not plotted:
                _unavailable(ax)
            ax.set_title(f"{geometry} {object_name}")
    axes[1, 0].set_xlabel("median-subject reliability")
    axes[1, 1].set_xlabel("median-subject reliability")
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("Label-destruction null (B=1,999 per phase)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in (".png", ".pdf"):
        path = figures_dir / f"{FIGURE_STEMS[1]}{suffix}"
        _save_figure(fig, path)
        written.append(path)
    plt.close(fig)

    # F3: same-subject statistic vs exhaustive unrelated derangement reference.
    frame = sources[FIGURE_STEMS[2]]
    fig, ax = plt.subplots(figsize=(11, 5))
    valid = frame[frame["status"].astype(str) == "PASS"].copy()
    if valid.empty:
        _unavailable(ax)
    else:
        valid["condition"] = (
            valid["phase"].astype(str) + " " + valid["geometry"].astype(str)
            + " " + valid["object"].astype(str)
        )
        x = np.arange(len(valid))
        ax.scatter(x, valid["same_subject_median"], marker="o", label="same subject")
        ax.scatter(x, valid["unrelated_median"], marker="x", label="unrelated median")
        ax.vlines(x, valid["unrelated_min"], valid["unrelated_max"], color="0.65", linewidth=1)
        ax.set_xticks(x, valid["condition"], rotation=35, ha="right")
        ax.set_ylim(-1.05, 1.05)
        ax.legend()
    ax.set_ylabel("cosine / derangement statistic")
    ax.set_title("Same-subject reliability vs all 133,496 derangements")
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        path = figures_dir / f"{FIGURE_STEMS[2]}{suffix}"
        _save_figure(fig, path)
        written.append(path)
    plt.close(fig)

    # F4: Stage-S subject scores.
    frame = sources[FIGURE_STEMS[3]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, object_name in zip(axes, OBJECTS, strict=True):
        plotted = False
        for phase in PHASES:
            for geometry, marker in (("AIRM", "o"), ("LE", "s")):
                rows = frame[
                    (frame["object"].astype(str) == object_name)
                    & (frame["phase"].astype(str) == phase)
                    & (frame["geometry"].astype(str) == geometry)
                    & (frame["status"].astype(str) == "PASS")
                ].sort_values("subject")
                if rows.empty:
                    continue
                ax.plot(rows["subject"], rows["observed"], marker=marker,
                        color=_phase_color(phase), linestyle="-" if geometry == "AIRM" else "--",
                        label=f"{phase} {geometry}")
                plotted = True
        if not plotted:
            _unavailable(ax)
        ax.set_title(f"{object_name} shared-template score")
        ax.set_xlabel("target subject")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0.0, color="0.75", linewidth=0.8)
    axes[0].set_ylabel("LOSO template cosine")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("Cross-subject shared semantic geometry")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in (".png", ".pdf"):
        path = figures_dir / f"{FIGURE_STEMS[3]}{suffix}"
        _save_figure(fig, path)
        written.append(path)
    plt.close(fig)

    # F5/F6: preregistered relational D/G heatmaps, never centered entries.
    for stem, title, cmap in (
        (FIGURE_STEMS[4], "AIRM D: subject-median full-split object", "viridis"),
        (FIGURE_STEMS[5], "AIRM G: subject-median full-split object", "coolwarm"),
    ):
        frame = sources[stem]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        median_rows = frame[frame["row_type"] == "subject_median"]
        maximum = float(np.nanmax(np.abs(median_rows["value"]))) if not median_rows.empty else 1.0
        for ax, phase in zip(axes, PHASES, strict=True):
            rows = median_rows[median_rows["phase"] == phase]
            if rows.empty or rows["value"].isna().all():
                _unavailable(ax)
                continue
            matrix = rows.pivot(index="class_row", columns="class_column", values="value").reindex(
                index=CLASSES, columns=CLASSES
            ).to_numpy(dtype=float)
            options: dict[str, Any] = {"cmap": cmap}
            if stem == FIGURE_STEMS[5]:
                options.update(vmin=-maximum, vmax=maximum)
            image_artist = ax.imshow(matrix, **options)
            ax.set_xticks(range(4), CLASSES, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(4), CLASSES, fontsize=8)
            ax.set_title(phase)
            fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(title)
        fig.tight_layout()
        for suffix in (".png", ".pdf"):
            path = figures_dir / f"{stem}{suffix}"
            _save_figure(fig, path)
            written.append(path)
        plt.close(fig)

    # F7: identity vs maximum nonidentity score for every target subject.
    frame = sources[FIGURE_STEMS[6]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, object_name in zip(axes, OBJECTS, strict=True):
        part = frame[(frame["object"] == object_name) & (frame["status"] == "PASS")]
        plotted = False
        for phase in PHASES:
            rows = part[part["phase"] == phase]
            if rows.empty:
                continue
            identity = rows[rows["is_identity"].map(lambda value: _strict_bool(value, label="figure7.is_identity"))]
            nonidentity = rows[~rows["is_identity"].map(lambda value: _strict_bool(value, label="figure7.is_identity"))]
            best_other = nonidentity.groupby("subject", sort=True)["score"].max()
            identity_score = identity.set_index("subject")["score"].sort_index()
            subjects = sorted(set(identity_score.index) & set(best_other.index))
            offset = -0.12 if phase == "discovery" else 0.12
            ax.scatter(np.asarray(subjects) + offset, identity_score.loc[subjects],
                       color=_phase_color(phase), marker="o", label=f"{phase} identity")
            ax.scatter(np.asarray(subjects) + offset, best_other.loc[subjects],
                       color=_phase_color(phase), marker="x", label=f"{phase} best nonidentity")
            plotted = True
        if not plotted:
            _unavailable(ax)
        ax.set_title(object_name)
        ax.set_xlabel("subject")
    axes[0].set_ylabel("template cosine")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.suptitle("Oracle semantic-permutation candidate scores (AIRM)")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    for suffix in (".png", ".pdf"):
        path = figures_dir / f"{FIGURE_STEMS[6]}{suffix}"
        _save_figure(fig, path)
        written.append(path)
    plt.close(fig)

    # F8: normalized rank and identity margin; neither is unlabeled recovery.
    frame = sources[FIGURE_STEMS[7]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column_index, object_name in enumerate(OBJECTS):
        part = frame[(frame["object"] == object_name) & (frame["status"] == "PASS")]
        plotted = False
        for phase in PHASES:
            rows = part[part["phase"] == phase].sort_values("subject")
            if rows.empty:
                continue
            axes[0, column_index].plot(rows["subject"], rows["normalized_rank"], marker="o",
                                       color=_phase_color(phase), label=phase)
            axes[1, column_index].plot(rows["subject"], rows["margin"], marker="o",
                                       color=_phase_color(phase), label=phase)
            plotted = True
        if not plotted:
            _unavailable(axes[0, column_index])
            _unavailable(axes[1, column_index])
        axes[0, column_index].set_title(object_name)
        axes[0, column_index].set_ylim(-0.03, 1.03)
        axes[1, column_index].axhline(0.0, color="0.7", linewidth=0.8)
        axes[1, column_index].set_xlabel("subject")
    axes[0, 0].set_ylabel("normalized identity rank")
    axes[1, 0].set_ylabel("identity score margin")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Oracle identity rank and margin (AIRM; true components supplied)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in (".png", ".pdf"):
        path = figures_dir / f"{FIGURE_STEMS[7]}{suffix}"
        _save_figure(fig, path)
        written.append(path)
    plt.close(fig)

    # F9: subject-level null-referenced paired effects and cached-score
    # influence.  The paired intervals use the frozen 20k subject bootstrap;
    # influence never refits a LOSO template.
    frame = sources[FIGURE_STEMS[8]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    marker_by_stage = {"R": "o", "S": "s", "P": "^"}
    for column_index, object_name in enumerate(OBJECTS):
        delta_ax = axes[0, column_index]
        influence_ax = axes[1, column_index]
        subjects = frame[
            (frame["row_type"] == "subject")
            & (frame["object"] == object_name)
            & (frame["status"] == "PASS")
        ].copy()
        bootstraps = frame[
            (frame["row_type"] == "bootstrap")
            & (frame["object"] == object_name)
            & (frame["status"] == "PASS")
        ].copy()
        plotted = False
        for stage_index, stage in enumerate(STAGES):
            confirm_subjects = subjects[
                (subjects["phase"] == "confirmatory")
                & (subjects["stage"] == stage)
            ].sort_values("subject")
            confirm_bootstrap = bootstraps[
                (bootstraps["phase"] == "confirmatory")
                & (bootstraps["stage"] == stage)
            ]
            if not confirm_subjects.empty and len(confirm_bootstrap) == 1:
                values = pd.to_numeric(
                    confirm_subjects["discovery_confirmatory_effect_delta"],
                    errors="coerce",
                )
                delta_ax.scatter(
                    values,
                    np.full(len(values), stage_index - 0.16),
                    color="#d62728",
                    marker=marker_by_stage[stage],
                    alpha=0.28,
                    s=20,
                )
                row = confirm_bootstrap.iloc[0]
                center = float(row["discovery_confirmatory_effect_delta_median"])
                low = float(row["discovery_confirmatory_effect_delta_ci_low"])
                high = float(row["discovery_confirmatory_effect_delta_ci_high"])
                delta_ax.errorbar(
                    center,
                    stage_index - 0.16,
                    xerr=[[max(0.0, center - low)], [max(0.0, high - center)]],
                    fmt="D",
                    color="#d62728",
                    capsize=3,
                    label="confirmatory−discovery effect" if stage_index == 0 else None,
                )
                plotted = True
            for phase, offset in (("discovery", 0.08), ("confirmatory", 0.24)):
                phase_subjects = subjects[
                    (subjects["phase"] == phase) & (subjects["stage"] == stage)
                ].sort_values("subject")
                phase_bootstrap = bootstraps[
                    (bootstraps["phase"] == phase) & (bootstraps["stage"] == stage)
                ]
                if phase_subjects.empty or len(phase_bootstrap) != 1:
                    continue
                values = pd.to_numeric(
                    phase_subjects["airm_minus_le_subject_effect_delta"],
                    errors="coerce",
                )
                delta_ax.scatter(
                    values,
                    np.full(len(values), stage_index + offset),
                    color=_phase_color(phase),
                    marker="x",
                    alpha=0.25,
                    s=20,
                )
                row = phase_bootstrap.iloc[0]
                center = float(row["airm_minus_le_effect_delta_median"])
                low = float(row["airm_minus_le_effect_delta_ci_low"])
                high = float(row["airm_minus_le_effect_delta_ci_high"])
                delta_ax.errorbar(
                    center,
                    stage_index + offset,
                    xerr=[[max(0.0, center - low)], [max(0.0, high - center)]],
                    fmt="X",
                    color=_phase_color(phase),
                    capsize=3,
                    label=(f"{phase} AIRM−LE effect" if stage_index == 0 else None),
                )
                plotted = True
        if not plotted:
            _unavailable(delta_ax)
        else:
            delta_ax.axvline(0.0, color="0.65", linewidth=0.8)
            delta_ax.set_yticks(range(3), STAGES)
        delta_ax.set_title(f"{object_name}: paired subject-effect deltas")
        delta_ax.set_xlabel("null-referenced effect delta (95% subject-bootstrap CI)")

        influence_plotted = False
        for phase in PHASES:
            for stage in STAGES:
                rows = subjects[
                    (subjects["phase"] == phase) & (subjects["stage"] == stage)
                ].sort_values("subject")
                if rows.empty:
                    continue
                offset = (STAGES.index(stage) - 1) * 0.08 + (
                    -0.02 if phase == "discovery" else 0.02
                )
                influence_ax.scatter(
                    rows["influence"],
                    rows["subject"] + offset,
                    marker=marker_by_stage[stage],
                    color=_phase_color(phase),
                    label=f"{phase} {stage}",
                    alpha=0.85,
                )
                influence_plotted = True
        if not influence_plotted:
            _unavailable(influence_ax)
        influence_ax.axvline(0.0, color="0.65", linewidth=0.8)
        influence_ax.set_title(f"{object_name}: cached-score influence")
        influence_ax.set_xlabel("median(without subject) − full median")
    axes[0, 0].set_ylabel("stage")
    axes[1, 0].set_ylabel("subject")
    handles, labels = [], []
    for ax in axes.ravel():
        local_handles, local_labels = ax.get_legend_handles_labels()
        for handle, label in zip(local_handles, local_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(
        "Subject paired null-referenced effects and cached-score influence "
        "(descriptive only)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in (".png", ".pdf"):
        path = figures_dir / f"{FIGURE_STEMS[8]}{suffix}"
        _save_figure(fig, path)
        written.append(path)
    plt.close(fig)

    # F10: fixed R/S/P effect operands for AIRM and LE.
    frame = sources[FIGURE_STEMS[9]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, object_name in zip(axes, OBJECTS, strict=True):
        part = frame[(frame["object"] == object_name) & (frame["status"] == "PASS")].copy()
        if part.empty:
            _unavailable(ax)
            continue
        labels = [f"{phase[0].upper()}-{stage}" for phase in PHASES for stage in STAGES]
        x = np.arange(len(labels), dtype=float)
        width = 0.36
        for geometry_index, geometry in enumerate(GEOMETRIES):
            values: list[float] = []
            for phase in PHASES:
                for stage in STAGES:
                    row = part[
                        (part["geometry"] == geometry)
                        & (part["phase"] == phase)
                        & (part["stage"] == stage)
                    ]
                    values.append(float(row.iloc[0]["effect"]) if len(row) == 1 else np.nan)
            ax.bar(x + (geometry_index - 0.5) * width, values, width=width,
                   label=geometry)
        ax.axhline(0.0, color="0.55", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_title(object_name)
        ax.set_xlabel("phase-stage (D=discovery, C=confirmatory)")
    axes[0].set_ylabel("observed - null median")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("AIRM vs LE frozen stage effects")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for suffix in (".png", ".pdf"):
        path = figures_dir / f"{FIGURE_STEMS[9]}{suffix}"
        _save_figure(fig, path)
        written.append(path)
    plt.close(fig)

    return tuple(written)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(result)


def _stage_markdown(verdicts: ConditionalVerdicts, object_name: str) -> str:
    operands = verdicts.stage_operands
    status = verdicts.stage_status
    rows: list[list[Any]] = []
    for geometry in GEOMETRIES:
        for stage in STAGES:
            operand = operands[
                (operands["geometry"] == geometry)
                & (operands["object"] == object_name)
                & (operands["stage"] == stage)
            ].iloc[0]
            decision = status[
                (status["geometry"] == geometry)
                & (status["object"] == object_name)
                & (status["stage"] == stage)
            ].iloc[0]
            rows.append(
                [
                    geometry,
                    stage,
                    operand["discovery_effect"],
                    operand["confirmatory_effect"],
                    operand["confirmatory_p"],
                    decision["status"],
                ]
            )
    return _md_table(
        ["geometry", "stage", "discovery effect", "confirmatory effect", "confirmatory p", "status"],
        rows,
    )


def _figure_links() -> str:
    return "\n".join(
        f"- [{stem} PNG](../figures/{stem}.png); "
        f"[PDF](../figures/{stem}.pdf); [source CSV](../figures/{stem}.csv)"
        for stem in FIGURE_STEMS
    )


def render_report(inputs: ReportingInputs, verdicts: ConditionalVerdicts) -> str:
    """Render the exact 18-section frozen report without positive spin."""

    reliability = inputs.tables["label_destruction_group_summary.csv"]
    shared = inputs.tables["semantic_permutation_null_summary.csv"]
    oracle = inputs.tables["oracle_permutation_group_summary.csv"]
    gate_rows = sum(len(inputs.tables[name]) for name in (
        "dataset_contract.csv", "covariance_sanity.csv", "airm_mean_convergence.csv",
        "le_mean_correctness.csv", "centering_isometry_gate.csv",
        "orthogonal_gauge_gate.csv", "degenerate_geometry_audit.csv",
    ))
    failed_gate_rows = sum(
        int((frame["status"].astype(str) == "FAIL").sum())
        for name, frame in inputs.tables.items()
        if name in {
            "dataset_contract.csv", "covariance_sanity.csv", "airm_mean_convergence.csv",
            "le_mean_correctness.csv", "centering_isometry_gate.csv",
            "orthogonal_gauge_gate.csv", "degenerate_geometry_audit.csv",
        }
    )

    def group_rows(frame: pd.DataFrame, phase: str) -> str:
        selected = frame[frame["phase"].astype(str) == phase]
        return _md_table(
            ["geometry", "object", "observed", "null median", "effect", "p", "B"],
            [
                [row.geometry, row.object, row.observed, row.null_median, row.effect, row.p_value, row.replicates]
                for row in selected.itertuples(index=False)
            ],
        )

    sections: list[tuple[str, str]] = []
    sections.append(
        (
            REPORT_HEADINGS[0],
            "This preregistered anatomy asks whether four true motor-imagery class prototypes "
            "have a relative WHOLE-covariance geometry that is reliable across runs, shared "
            "across subjects and sessions under fixed semantic names, and identifiable when "
            "an oracle supplies the four target components.",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[1],
            "V1 diagnosed information in WHOLE versus WINDOW5 representations; V2 audited "
            "marginal centering geometry; Trajectory v0 tested local temporal order. This "
            "experiment instead examines class-prototype relational anatomy using WHOLE "
            "covariances only. It does not repeat WINDOW5 or trajectory analysis.",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[2],
            f"Protocol version 1.0, seed 20260809, AIRM primary and LE robustness. Discovery "
            f"is session `0train`; confirmation is locked session `1test`. Splits are A=runs "
            f"0–2, B=runs 3–5 and F=runs 0–5. Four classes follow the fixed order "
            f"`{', '.join(CLASSES)}`. The unlock status was `{inputs.unlock['status']}` with "
            f"designation `{inputs.unlock['confirmatory_designation']}`.",
        )
    )
    failure_text = (
        "none"
        if verdicts.failure_class is None
        else f"{verdicts.failure_class}: " + "; ".join(verdicts.failure_details)
    )
    sections.append(
        (
            REPORT_HEADINGS[3],
            f"Validated hard-gate rows: {gate_rows}; rows carrying `FAIL`: {failed_gate_rows}. "
            f"Global failure classification: **{failure_text}**. A single required failure "
            "makes every scientific chain UNASSESSED; no available-case substitute is used.",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[4],
            "D contains the six pairwise AIRM distances among class Fréchet means. G is the "
            "4×4 marginal-anchor tangent Gram matrix and was checked by both the direct "
            "linear-solve expression and the whitened-identity expression. AIRM marginal "
            "centering is an isometry: it cannot create this within-subject class geometry. "
            f"See [D heatmaps](../figures/{FIGURE_STEMS[4]}.png) and "
            f"[G heatmaps](../figures/{FIGURE_STEMS[5]}.png).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[5],
            group_rows(reliability, "discovery")
            + f"\n\n[Subject scores](../figures/{FIGURE_STEMS[0]}.png) and "
            f"[label-null distribution](../figures/{FIGURE_STEMS[1]}.png).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[6],
            group_rows(reliability, "confirmatory")
            + "\n\nThe confirmatory p-values use the independently shuffled `1test` labels "
            "within subject×session×run and the frozen plus-one rule.",
        )
    )
    unrelated = inputs.tables["unrelated_subject_derangement_summary.csv"]
    unrelated_rows = [
        [row.phase, row.geometry, row.object, row.same_subject_median, row.unrelated_median,
         row.unrelated_min, row.unrelated_max, row.derangement_count]
        for row in unrelated.itertuples(index=False)
    ]
    sections.append(
        (
            REPORT_HEADINGS[7],
            _md_table(
                ["phase", "geometry", "object", "same", "unrelated median", "min", "max", "!9"],
                unrelated_rows,
            )
            + f"\n\nThis exhaustive derangement reference is descriptive and never votes. "
            f"[Figure 3](../figures/{FIGURE_STEMS[2]}.png).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[8],
            group_rows(shared, "discovery")
            + "\n\nEach discovery score averages source-A→target-B and source-B→target-A "
            "LOSO comparisons. Templates normalize the sum of unit subject shapes.",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[9],
            group_rows(shared, "confirmatory")
            + "\n\nThe confirmatory template is the fixed discovery-F LOSO template; "
            "confirmatory data never updates it. "
            f"[Subject scores](../figures/{FIGURE_STEMS[3]}.png).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[10],
            group_rows(oracle, "confirmatory")
            + "\n\nStage P supplies the target's four true components and hides only their names. "
            "It is therefore not clustering, unlabeled component recovery, adaptation, or a "
            "deployable score. Identity receives the conservative worst rank within 1e-12. "
            f"[Candidate scores](../figures/{FIGURE_STEMS[6]}.png); "
            f"[rank and margin](../figures/{FIGURE_STEMS[7]}.png).",
        )
    )
    sections.append((REPORT_HEADINGS[11], _stage_markdown(verdicts, "D")))
    sections.append((REPORT_HEADINGS[12], _stage_markdown(verdicts, "G")))
    paired_bootstrap = inputs.tables["subject_bootstrap_summary.csv"]
    paired_rows: list[list[Any]] = []
    for object_name in OBJECTS:
        for stage in STAGES:
            discovery_row = paired_bootstrap[
                (paired_bootstrap["phase"] == "discovery")
                & (paired_bootstrap["geometry"] == "AIRM")
                & (paired_bootstrap["object"] == object_name)
                & (paired_bootstrap["stage"] == stage)
            ].iloc[0]
            confirmatory_row = paired_bootstrap[
                (paired_bootstrap["phase"] == "confirmatory")
                & (paired_bootstrap["geometry"] == "AIRM")
                & (paired_bootstrap["object"] == object_name)
                & (paired_bootstrap["stage"] == stage)
            ].iloc[0]
            paired_rows.append(
                [
                    object_name,
                    stage,
                    (
                        f"{_fmt(confirmatory_row['discovery_confirmatory_effect_delta_median'])} "
                        f"[{_fmt(confirmatory_row['discovery_confirmatory_effect_delta_ci_low'])}, "
                        f"{_fmt(confirmatory_row['discovery_confirmatory_effect_delta_ci_high'])}]"
                    ),
                    (
                        f"{_fmt(discovery_row['airm_minus_le_effect_delta_median'])} "
                        f"[{_fmt(discovery_row['airm_minus_le_effect_delta_ci_low'])}, "
                        f"{_fmt(discovery_row['airm_minus_le_effect_delta_ci_high'])}]"
                    ),
                    (
                        f"{_fmt(confirmatory_row['airm_minus_le_effect_delta_median'])} "
                        f"[{_fmt(confirmatory_row['airm_minus_le_effect_delta_ci_low'])}, "
                        f"{_fmt(confirmatory_row['airm_minus_le_effect_delta_ci_high'])}]"
                    ),
                ]
            )
    sections.append(
        (
            REPORT_HEADINGS[13],
            f"Frozen LE label: **{verdicts.le_robustness_label}**. LE is secondary and cannot "
            "rescue or change the AIRM terminal decision. "
            "Paired deltas below use each subject's null-referenced effect "
            "(`observed − subject-null median`), never the raw score. Intervals are frozen "
            "20,000-resample subject-bootstrap intervals and do not vote.\n\n"
            + _md_table(
                [
                    "object",
                    "stage",
                    "confirm−discovery effect Δ [CI]",
                    "discovery AIRM−LE effect Δ [CI]",
                    "confirmatory AIRM−LE effect Δ [CI]",
                ],
                paired_rows,
            )
            + f"\n\n[AIRM/LE stage effects](../figures/{FIGURE_STEMS[9]}.png); "
            f"[paired subject effects and influence](../figures/{FIGURE_STEMS[8]}.png).",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[14],
            f"**{verdicts.terminal_decision}**\n\nThe label follows only the frozen AIRM "
            "D/G R→S→P chains and the explicit global failure gate. No absolute cosine "
            "cutoff or post-result rule was used.",
        )
    )
    if verdicts.terminal_decision == "GO_STRONG":
        justified = (
            "Within this dataset and frozen WHOLE representation, both the AIRM metric-shape "
            "D chain and marginal-anchor Gram-shape G chain passed reliability, shared-geometry "
            "and oracle-name-identifiability stages. This claim is limited to the saved "
            "subject-level statistics and the oracle setting."
        )
    elif verdicts.terminal_decision == "GO_METRIC_ONLY":
        justified = (
            "Only the AIRM metric-shape D chain passed all three stages. The anchored tangent "
            "G chain did not; no claim of shared directional geometry is justified."
        )
    elif verdicts.terminal_decision == "STOP_TANGENT_ONLY":
        justified = (
            "The anchored tangent G chain passed while the exact metric-shape D chain did not. "
            "The frozen terminal rule is STOP_TANGENT_ONLY; this discrepancy is not evidence "
            "for a new method."
        )
    elif verdicts.terminal_decision == "STOP_NO_SHARED_GEOMETRY":
        justified = (
            "Neither AIRM D nor G passed the complete fixed sequence. The data do not support "
            "the preregistered shared WHOLE-covariance relational anchor."
        )
    else:
        justified = (
            "No scientific R/S/P conclusion is justified because a required data, numerical, "
            "or degeneracy gate failed. The recorded failure is the result of this run."
        )
    sections.append(
        (
            REPORT_HEADINGS[15],
            justified
            + f"\n\n[Subject effect deltas and cached-score influence]"
            f"(../figures/{FIGURE_STEMS[8]}.png) are descriptive and do not vote. "
            "The influence statistic remains the raw cached subject-score "
            "`T_leave-one-subject-out − T_full`; paired deltas instead use "
            "null-referenced subject effects.",
        )
    )
    sections.append(
        (
            REPORT_HEADINGS[16],
            "D/G are not full conditional distributions. This experiment did not show that "
            "all subject variation was removed. Oracle P is not unlabeled target-component "
            "recovery, pseudo-labeling, conditional alignment, or domain-adaptation success. "
            "No target-label-free conditional identifiability claim is made. There is no new "
            "WINDOW5, temporal-order, trajectory, neural-model, neuroscience-mechanism, or "
            "cross-dataset conclusion.\n\nAll figure artifacts:\n\n" + _figure_links(),
        )
    )
    sections.append((REPORT_HEADINGS[17], verdicts.next_question))
    text = REPORT_TITLE + "\n\n" + "\n\n".join(
        f"## {heading}\n\n{body}" for heading, body in sections
    ) + "\n"
    validate_report_contract(text)
    return text


def validate_report_contract(text: str) -> None:
    lines = text.splitlines()
    if not lines or lines[0] != REPORT_TITLE:
        raise ReportingContractError("report title mismatch")
    headings = tuple(line[3:] for line in lines if line.startswith("## "))
    if headings != REPORT_HEADINGS:
        raise ReportingContractError(f"report section contract mismatch: {headings}")
    for stem in FIGURE_STEMS:
        for suffix in (".csv", ".png", ".pdf"):
            if f"../figures/{stem}{suffix}" not in text:
                raise ReportingContractError(f"report lacks link to {stem}{suffix}")
    final_body = text.split(f"## {REPORT_HEADINGS[-1]}\n\n", 1)[1].strip()
    if final_body.count("?") != 1 or not final_body.endswith("?"):
        raise ReportingContractError("final section must contain exactly one question")
    forbidden_claims = (
        "distribution classifier is required",
        "trajectory model is required",
        "domain adaptation will improve",
        "pseudo-label-free conditional alignment is solved",
    )
    lowered = text.lower()
    if any(claim in lowered for claim in forbidden_claims):
        raise ReportingContractError("report contains a prohibited affirmative conclusion")


def _unassessed_stage_operands() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "geometry": geometry,
                "object": object_name,
                "stage": stage,
                "discovery_effect": np.nan,
                "confirmatory_effect": np.nan,
                "confirmatory_p": np.nan,
            }
            for geometry in GEOMETRIES
            for object_name in OBJECTS
            for stage in STAGES
        ]
    )


def _build_unassessed_figure_sources(
    inputs: ReportingInputs, verdicts: ConditionalVerdicts
) -> dict[str, pd.DataFrame]:
    manifest = inputs.failure_manifest
    if manifest is None:
        raise ReportingContractError("missing failure manifest for UNASSESSED figures")
    row = {
        "protocol_version": manifest["protocol_version"],
        "protocol_sha256": manifest["protocol_sha256"],
        "config_sha256": manifest["config_sha256"],
        "code_commit": manifest["code_commit"],
        "phase": manifest["phase"],
        "session": manifest["session"],
        "status": "UNASSESSED",
        "failure_class": manifest["failure_class"],
        "terminal_decision": verdicts.terminal_decision,
        "reason_code": manifest["reason_code"],
        "reason": manifest["reason"],
        "scientific_data_plotted": False,
    }
    return {stem: pd.DataFrame([row]) for stem in FIGURE_STEMS}


def _plot_unassessed_figures(
    sources: Mapping[str, pd.DataFrame], figures_dir: Path
) -> tuple[Path, ...]:
    written: list[Path] = []
    for stem in FIGURE_STEMS:
        row = sources[stem].iloc[0]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _unavailable(
            ax,
            "UNASSESSED — required protocol gate failed\n"
            f"{row['terminal_decision']}\n"
            "No scientific data are plotted",
        )
        ax.set_title(stem.replace("_", " "))
        fig.tight_layout()
        for suffix in (".png", ".pdf"):
            path = figures_dir / f"{stem}{suffix}"
            _save_figure(fig, path)
            written.append(path)
        plt.close(fig)
    return tuple(written)


def render_unassessed_report(
    inputs: ReportingInputs, verdicts: ConditionalVerdicts
) -> str:
    """Render the exact report shell without scientific estimates or null claims."""

    manifest = inputs.failure_manifest
    if manifest is None:
        raise ReportingContractError("cannot render failure report without manifest")
    phase = str(manifest["phase"])
    session = str(manifest["session"])
    terminal = verdicts.terminal_decision
    unavailable = (
        f"**UNASSESSED.** The required `{phase}` phase failed before this analysis was "
        "eligible. No scientific estimate, null result, or available-case substitute is "
        "reported in this section."
    )
    sections = [
        (
            REPORT_HEADINGS[0],
            "The frozen question was whether class-prototype WHOLE-covariance D/G geometry "
            "is reliable, shared and semantically identifiable. This run did not reach a "
            "scientifically assessable state.",
        ),
        (
            REPORT_HEADINGS[1],
            "This protocol follows the earlier representation, centering and trajectory "
            "audits, but the present failure supplies no new geometric evidence.",
        ),
        (
            REPORT_HEADINGS[2],
            f"Protocol version `{manifest['protocol_version']}`, seed `20260809`, frozen "
            f"phase `{phase}`, session `{session}`, AIRM primary and LE secondary. The "
            "live protocol and config were verified byte-for-byte against their frozen copies.",
        ),
        (
            REPORT_HEADINGS[3],
            f"Terminal status: **{terminal}**. Failure class: `{manifest['failure_class']}`; "
            f"reason code: `{manifest['reason_code']}`; recorded reason: "
            f"`{manifest['reason']}`. `scientific_nulls_executed=false` and "
            "`downstream_phase_permitted=false`.",
        ),
        (REPORT_HEADINGS[4], unavailable),
        (REPORT_HEADINGS[5], unavailable),
        (REPORT_HEADINGS[6], unavailable),
        (REPORT_HEADINGS[7], unavailable),
        (REPORT_HEADINGS[8], unavailable),
        (REPORT_HEADINGS[9], unavailable),
        (REPORT_HEADINGS[10], unavailable),
        (
            REPORT_HEADINGS[11],
            "**UNASSESSED.** D-stage eligibility was not reached; R/S/P operands were not "
            "constructed from partial data.",
        ),
        (
            REPORT_HEADINGS[12],
            "**UNASSESSED.** G-stage eligibility was not reached; R/S/P operands were not "
            "constructed from partial data.",
        ),
        (
            REPORT_HEADINGS[13],
            "LE robustness label: **UNASSESSED**. LE cannot rescue a failed prerequisite gate.",
        ),
        (
            REPORT_HEADINGS[14],
            f"**{terminal}**. This is a protocol/data/numerical terminal, not a scientific "
            "GO/STOP geometry verdict.",
        ),
        (
            REPORT_HEADINGS[15],
            "Only the recorded failure classification, provenance, phase barrier and absence "
            "of eligible downstream analysis are justified. The ten linked figures are "
            "explicit UNASSESSED placeholders and contain no scientific data.\n\n"
            + _figure_links(),
        ),
        (
            REPORT_HEADINGS[16],
            "No claim about reliability, shared D/G geometry, semantic identifiability, AIRM "
            "versus LE, conditional alignment, adaptation, WINDOW5, trajectories, neural "
            "models or target-label-free recovery is justified.",
        ),
        (REPORT_HEADINGS[17], verdicts.next_question),
    ]
    text = REPORT_TITLE + "\n\n" + "\n\n".join(
        f"## {heading}\n\n{body}" for heading, body in sections
    ) + "\n"
    validate_report_contract(text)
    return text


def create_reporting_outputs(inputs: ReportingInputs) -> ReportingArtifacts:
    """Create exactly ten CSV/PNG/PDF figure triplets and the 18-section report."""

    figures_dir = inputs.output_root / "figures"
    report_path = inputs.output_root / str(inputs.config["outputs"]["report_file"])
    expected_report = inputs.output_root / "report" / "conditional_geometry_anatomy_v1.md"
    if report_path != expected_report:
        raise ReportingContractError(
            f"report path mismatch: expected {expected_report}, observed {report_path}"
        )
    figures_dir.mkdir(parents=True, exist_ok=True)
    unexpected = []
    for path in figures_dir.glob("figure_*.*"):
        if path.stem not in FIGURE_STEMS or path.suffix not in {".csv", ".png", ".pdf"}:
            unexpected.append(path.name)
    if unexpected:
        raise ReportingContractError(f"unexpected figure artifacts already exist: {unexpected}")

    if inputs.failure_manifest is not None:
        verdicts = compute_frozen_verdicts(
            _unassessed_stage_operands(),
            hard_gates_pass=False,
            failure_class=inputs.failure_class,
            failure_details=inputs.failure_details,
        )
        if verdicts.terminal_decision != inputs.failure_manifest["terminal_decision"]:
            raise ReportingContractError(
                "failure manifest terminal differs from frozen verdict mapping"
            )
        sources = _build_unassessed_figure_sources(inputs, verdicts)
        source_paths: list[Path] = []
        for stem in FIGURE_STEMS:
            source_path = figures_dir / f"{stem}.csv"
            _atomic_csv(sources[stem], source_path)
            source_paths.append(source_path)
        figure_paths = _plot_unassessed_figures(sources, figures_dir)
        report_text = render_unassessed_report(inputs, verdicts)
        _atomic_text(report_text, report_path)
        expected_files = {
            *(f"{stem}.csv" for stem in FIGURE_STEMS),
            *(f"{stem}.png" for stem in FIGURE_STEMS),
            *(f"{stem}.pdf" for stem in FIGURE_STEMS),
        }
        observed_files = {
            path.name
            for path in figures_dir.iterdir()
            if path.is_file() and path.name.startswith("figure_")
        }
        if observed_files != expected_files:
            raise ReportingContractError(
                "UNASSESSED figure artifact set mismatch; "
                f"missing={sorted(expected_files-observed_files)}, "
                f"extra={sorted(observed_files-expected_files)}"
            )
        return ReportingArtifacts(
            verdicts=verdicts,
            figure_sources=sources,
            figure_paths=figure_paths,
            figure_source_paths=tuple(source_paths),
            report_path=report_path,
            report_text=report_text,
        )

    operands = _build_stage_operands(inputs.snapshot_tables)
    verdicts = compute_frozen_verdicts(
        operands,
        hard_gates_pass=inputs.failure_class is None,
        failure_class=inputs.failure_class,
        failure_details=inputs.failure_details,
    )
    _validate_producer_decisions(inputs, verdicts)
    sources = build_figure_sources(inputs, verdicts)
    source_paths: list[Path] = []
    for stem in FIGURE_STEMS:
        path = figures_dir / f"{stem}.csv"
        _atomic_csv(sources[stem], path)
        source_paths.append(path)
    figure_paths = _plot_figures(sources, figures_dir)
    report_text = render_report(inputs, verdicts)
    _atomic_text(report_text, report_path)
    expected_files = {
        *(f"{stem}.csv" for stem in FIGURE_STEMS),
        *(f"{stem}.png" for stem in FIGURE_STEMS),
        *(f"{stem}.pdf" for stem in FIGURE_STEMS),
    }
    observed_files = {
        path.name for path in figures_dir.iterdir() if path.is_file() and path.name.startswith("figure_")
    }
    if observed_files != expected_files:
        raise ReportingContractError(
            f"generated figure artifact set mismatch; missing={sorted(expected_files-observed_files)}, "
            f"extra={sorted(observed_files-expected_files)}"
        )
    return ReportingArtifacts(
        verdicts=verdicts,
        figure_sources=sources,
        figure_paths=figure_paths,
        figure_source_paths=tuple(source_paths),
        report_path=report_path,
        report_text=report_text,
    )
