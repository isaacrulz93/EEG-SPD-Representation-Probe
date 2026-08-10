"""Strict discovery producer for Conditional-Geometry Anatomy v1.

This module is the orchestration layer between the already frozen data,
geometry, null, and subject-statistics modules.  It is deliberately session
agnostic: callers pass an in-memory :class:`ConditionalWholeData` object and a
phase token.  In particular, importing this module never resolves an EEG path.

All tracked products are written atomically.  Long null calculations use the
replicate-indexed checkpoint format from :mod:`src.conditional_nulls_v1` so a
resume cannot alter a random stream or silently replace a finished replicate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import shutil
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pyriemann.geometry.mean import mean_riemann

from src.conditional_geometry_v1 import (
    AIRM,
    LE,
    D_UPPER_TRIANGLE_INDICES,
    GeometryObjects,
    GeometryThresholds,
    airm_distance,
    airm_distance_matrix,
    airm_gram_matrices,
    airm_mean_batched,
    karcher_residual,
    le_distance_matrix,
    le_gram_matrix,
    compute_geometry_objects,
    shape_from_D,
    shape_from_G,
    spd_invsqrt,
    spd_log,
    svec,
    symmetrize,
    validate_spd_stack,
)
from src.conditional_nulls_v1 import (
    FAMILY_LABEL,
    FAMILY_ORACLE,
    FAMILY_SEMANTIC,
    PHASE_COMMON,
    PHASE_DISCOVERY,
    NullCheckpoint,
    all_derangements,
    all_s4_permutations,
    create_null_checkpoint,
    load_null_checkpoint,
    oracle_rank_null,
    pending_checkpoint_indices,
    permuted_shape_bank,
    record_checkpoint_batch,
    save_null_checkpoint,
    semantic_discovery_null,
    semantic_confirmatory_null,
    shuffle_labels_within_strata,
    unrelated_derangement_statistics,
)
from src.conditional_provenance_v1 import sha256_file
from src.conditional_statistics_v1 import (
    StageEvidence,
    discovery_oracle_score_sets,
    discovery_shared_subject_scores,
    confirmatory_oracle_score_sets,
    confirmatory_shared_subject_scores,
    leave_one_subject_out_influence,
    loso_templates,
    plus_one_null_summary,
    normalize_shape_vectors,
    reliability_subject_scores,
    subject_bootstrap_median,
    subject_bootstrap_paired_median_delta,
    subject_null_percentiles,
    summarize_oracle_scores,
    evaluate_fixed_sequence,
    terminal_airm_decision,
    le_robustness_label,
)
from src.data_conditional_v1 import ConditionalWholeData, sha256_array, subject_split_positions


GEOMETRIES = (AIRM, LE)
OBJECTS = ("D", "G")
SPLITS = ("A", "B", "F")
COMMON_COLUMNS = (
    "protocol_version",
    "protocol_sha256",
    "config_sha256",
    "code_commit",
    "phase",
    "session",
    "status",
)
FAILURE_MANIFEST_KEYS = frozenset(
    {
        *COMMON_COLUMNS,
        "schema_version",
        "failure_class",
        "terminal_decision",
        "reason_code",
        "reason",
        "scientific_nulls_executed",
        "downstream_phase_permitted",
        "le_robustness_label",
    }
)
FAILURE_DECISION_KEYS = FAILURE_MANIFEST_KEYS | {
    "failure_manifest",
    "failure_manifest_sha256",
    "chains",
}

# These public schema constants are the strict contract consumed by reporting.
GROUP_SUMMARY_COLUMNS = COMMON_COLUMNS + (
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
WITHIN_RELIABILITY_COLUMNS = COMMON_COLUMNS + (
    "geometry",
    "object",
    "stage",
    "subject",
    "observed",
    "group_observed",
)
CROSS_SHARED_COLUMNS = WITHIN_RELIABILITY_COLUMNS + (
    "null_median",
    "effect",
    "null_percentile",
    "replicates",
)
LABEL_SUBJECT_COLUMNS = COMMON_COLUMNS + (
    "geometry",
    "object",
    "stage",
    "subject",
    "observed",
    "null_median",
    "effect",
    "null_percentile",
    "replicates",
)
ORACLE_SCORE_COLUMNS = COMMON_COLUMNS + (
    "geometry",
    "object",
    "stage",
    "subject",
    "permutation_index",
    "permutation",
    "is_identity",
    "score",
)
ORACLE_SUBJECT_COLUMNS = COMMON_COLUMNS + (
    "geometry",
    "object",
    "stage",
    "subject",
    "identity_score",
    "true_rank",
    "normalized_rank",
    "top1_exact",
    "best_permutation_index",
    "best_permutation",
    "second_best_permutation_index",
    "second_best_permutation",
    "margin",
    "null_median",
    "effect",
    "null_percentile",
)
MEAN_COLUMNS = COMMON_COLUMNS + (
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
)
DISCOVERY_TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "dataset_contract.csv": COMMON_COLUMNS
    + ("scope", "subject", "run", "class_label", "split", "observed_count", "expected_count", "passed"),
    "covariance_sanity.csv": COMMON_COLUMNS
    + (
        "covariance_index", "trial_uid", "subject", "run", "class_label", "finite",
        "symmetry_relative_error", "min_eigenvalue", "max_eigenvalue", "condition_number", "passed",
    ),
    "airm_mean_convergence.csv": MEAN_COLUMNS,
    "le_mean_correctness.csv": MEAN_COLUMNS,
    "centering_isometry_gate.csv": COMMON_COLUMNS
    + ("geometry", "subject", "split", "d_centering_relative_error", "limit", "passed"),
    "orthogonal_gauge_gate.csv": COMMON_COLUMNS
    + (
        "geometry", "subject", "split", "d_relative_error", "g_relative_error",
        "d_limit", "g_limit", "g_direct_whitened_relative_error",
        "g_direct_whitened_limit", "permutation_d_relative_error",
        "permutation_g_relative_error", "permutation_limit",
        "le_d_g_identity_relative_error", "le_d_g_identity_limit",
        "d_symmetry_relative_error", "d_diagonal_max_abs", "d_minimum",
        "g_symmetry_relative_error", "structure_tolerance", "passed",
    ),
    "degenerate_geometry_audit.csv": COMMON_COLUMNS
    + (
        "geometry", "object", "subject", "split", "shape_norm",
        "degeneracy_threshold", "is_degenerate", "passed",
    ),
    "D_shape_vectors.csv": COMMON_COLUMNS
    + ("geometry", "subject", "split", "shape_norm")
    + tuple(f"raw_{i}" for i in range(6))
    + tuple(f"z_{i}" for i in range(6)),
    "G_shape_vectors.csv": COMMON_COLUMNS
    + ("geometry", "subject", "split", "shape_norm")
    + tuple(f"raw_{i}" for i in range(10))
    + tuple(f"z_{i}" for i in range(10)),
    "absolute_geometry_scales.csv": COMMON_COLUMNS
    + (
        "geometry", "subject", "split", "D_shape_norm", "G_shape_norm",
        "D_upper_mean", "D_upper_max", "class_radius_mean", "class_radius_max",
    ),
    "radius_angle_summary.csv": COMMON_COLUMNS
    + (
        "geometry", "subject", "split", "class_left", "class_right",
        "radius_left", "radius_right", "cosine", "angle_radians",
    ),
    "within_subject_reliability.csv": WITHIN_RELIABILITY_COLUMNS,
    "label_destruction_subject_summary.csv": LABEL_SUBJECT_COLUMNS,
    "label_destruction_group_summary.csv": GROUP_SUMMARY_COLUMNS,
    "unrelated_subject_derangement_summary.csv": COMMON_COLUMNS
    + (
        "geometry", "object", "same_subject_median", "unrelated_median",
        "unrelated_min", "unrelated_max", "derangement_count",
    ),
    "loso_templates.csv": COMMON_COLUMNS
    + (
        "geometry", "object", "target_subject", "template_source_split",
        "target_split", "feature_index", "value",
    ),
    "cross_subject_shared_geometry.csv": CROSS_SHARED_COLUMNS,
    "semantic_permutation_null_summary.csv": GROUP_SUMMARY_COLUMNS,
    "oracle_permutation_all_24_scores.csv": ORACLE_SCORE_COLUMNS,
    "oracle_permutation_subject_summary.csv": ORACLE_SUBJECT_COLUMNS,
    "oracle_permutation_group_summary.csv": GROUP_SUMMARY_COLUMNS,
    "subject_bootstrap_summary.csv": COMMON_COLUMNS
    + (
        "geometry", "object", "stage", "observed_median", "ci_low", "ci_high", "replicates",
        "discovery_confirmatory_effect_delta_median",
        "discovery_confirmatory_effect_delta_ci_low",
        "discovery_confirmatory_effect_delta_ci_high",
        "airm_minus_le_effect_delta_median",
        "airm_minus_le_effect_delta_ci_low",
        "airm_minus_le_effect_delta_ci_high",
    ),
    "leave_one_subject_out_influence.csv": COMMON_COLUMNS
    + (
        "geometry", "object", "stage", "subject", "influence",
        "subject_score", "subject_effect", "discovery_subject_effect",
        "confirmatory_subject_effect", "discovery_confirmatory_effect_delta",
        "airm_minus_le_subject_effect_delta",
    ),
}


class ConditionalPipelineError(RuntimeError):
    """Raised when a producer or output contract fails closed."""


@dataclass(frozen=True)
class ProducerContext:
    protocol_version: str
    protocol_sha256: str
    config_sha256: str
    code_commit: str
    phase: str
    session: str

    def prefix(self, status: str = "PASS") -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "protocol_sha256": self.protocol_sha256,
            "config_sha256": self.config_sha256,
            "code_commit": self.code_commit,
            "phase": self.phase,
            "session": self.session,
            "status": status,
        }


@dataclass(frozen=True)
class PhaseGeometryBundle:
    """Canonical arrays with axes geometry, subject, split, ... ."""

    context: ProducerContext
    subjects: np.ndarray
    splits: np.ndarray
    classes: np.ndarray
    marginal_means: np.ndarray
    class_means: np.ndarray
    D: np.ndarray
    G: np.ndarray
    zD: np.ndarray
    zG: np.ndarray
    gate_passed: np.ndarray
    failure_reasons: tuple[tuple[tuple[str, ...], ...], ...]

    @property
    def all_gates_passed(self) -> bool:
        return bool(np.all(self.gate_passed))

    def geometry_index(self, geometry: str) -> int:
        try:
            return GEOMETRIES.index(str(geometry))
        except ValueError as error:
            raise ConditionalPipelineError(f"unknown geometry: {geometry}") from error

    def object_arrays(self, geometry: str, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        index = self.geometry_index(geometry)
        if object_name == "D":
            return self.D[index], self.zD[index]
        if object_name == "G":
            return self.G[index], self.zG[index]
        raise ConditionalPipelineError("object_name must be D or G")


@dataclass(frozen=True)
class DiscoveryNullResult:
    label_group_statistics: np.ndarray
    semantic_group_statistics: np.ndarray
    oracle_group_statistics: np.ndarray
    group_summary: pd.DataFrame


@dataclass(frozen=True)
class LabelNullReplicateResult:
    statistics: np.ndarray
    airm_crosscheck: dict[str, Any] | None


_FAILURE_TERMINALS = {
    "data": "UNASSESSED_DATA_CONTRACT_FAILURE",
    "numerical": "UNASSESSED_NUMERICAL_FAILURE",
    "degenerate": "UNASSESSED_DEGENERATE_GEOMETRY",
}


def write_unassessed_failure_artifacts(
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    config_sha256: str,
    code_commit: str,
    phase: str,
    session: str,
    failure_class: str,
    reason_code: str,
    reason: str,
) -> dict[str, Any]:
    """Atomically persist a recognized hard failure without null artifacts."""

    if phase not in ("discovery", "confirmatory"):
        raise ValueError("failure phase must be discovery or confirmatory")
    if failure_class not in _FAILURE_TERMINALS:
        raise ValueError("failure_class must be data, numerical, or degenerate")
    context = _context(
        config,
        config_sha256=config_sha256,
        code_commit=code_commit,
        phase=phase,
        session=session,
    )
    output = _resolve_output_root(Path(repo_root).resolve(), config)
    from src.conditional_provenance_v1 import atomic_write_json

    manifest_path = output / "protocol" / f"{phase}_failure_manifest.json"
    payload = {
        **context.prefix("UNASSESSED"),
        "schema_version": "conditional-unassessed-v1",
        "failure_class": failure_class,
        "terminal_decision": _FAILURE_TERMINALS[failure_class],
        "reason_code": str(reason_code),
        "reason": str(reason),
        "scientific_nulls_executed": False,
        "downstream_phase_permitted": False,
        "le_robustness_label": "UNASSESSED",
    }
    atomic_write_json(manifest_path, payload)
    decision = {
        **payload,
        "failure_manifest": str(manifest_path.relative_to(Path(repo_root).resolve())),
        "failure_manifest_sha256": sha256_file(manifest_path),
        "chains": {
            geometry: {object_name: None for object_name in OBJECTS}
            for geometry in GEOMETRIES
        },
    }
    atomic_write_json(output / "confirmatory_decision.json", decision)
    return decision


def classify_recognized_phase_failure(error: BaseException) -> str:
    """Map only declared data/geometry/pipeline failures to terminal classes."""

    from src.conditional_geometry_v1 import (
        ConditionalGeometryError,
        DegenerateClassGeometryError,
    )
    from src.data_conditional_v1 import ConditionalDataError

    if isinstance(error, ConditionalDataError):
        return "data"
    if isinstance(error, DegenerateClassGeometryError):
        return "degenerate"
    if isinstance(error, ConditionalGeometryError):
        return "numerical"
    if isinstance(error, ConditionalPipelineError):
        return "degenerate" if "DEGENERATE_CLASS_GEOMETRY" in str(error) else "numerical"
    raise TypeError("error is not a recognized phase failure")


def _context(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    code_commit: str,
    phase: str,
    session: str,
) -> ProducerContext:
    if phase not in ("discovery", "confirmatory", "combined"):
        raise ConditionalPipelineError("phase must be discovery, confirmatory, or combined")
    if phase == "combined":
        expected_session = (
            f"{config['dataset']['discovery_session']}+"
            f"{config['dataset']['confirmatory_session']}"
        )
        if str(session) != expected_session:
            raise ConditionalPipelineError(
                f"combined session token must be {expected_session}"
            )
    return ProducerContext(
        protocol_version=str(config["protocol"]["version"]),
        protocol_sha256=str(config["protocol"]["protocol_sha256"]),
        config_sha256=str(config_sha256),
        code_commit=str(code_commit),
        phase=phase,
        session=str(session),
    )


def producer_code_commit(repo_root: str | Path) -> str:
    """Return the latest commit touching the frozen producer/code surface.

    Result-only milestone commits therefore do not invalidate the producer
    provenance embedded by the preceding geometry run.
    """

    root = Path(repo_root).resolve()
    process = subprocess.run(
        [
            "git", "log", "-1", "--format=%H", "--", "src", "scripts", "configs",
            "docs/PROTOCOL_CONDITIONAL_GEOMETRY_V1.md", "requirements.txt",
        ],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = process.stdout.strip()
    if process.returncode != 0 or len(value) != 40:
        raise ConditionalPipelineError(
            f"could not resolve producer code commit: {process.stderr.strip()}"
        )
    return value


def _resolve_output_root(repo_root: Path, config: Mapping[str, Any]) -> Path:
    value = Path(str(config["project"]["output_dir"]))
    result = value.resolve() if value.is_absolute() else (repo_root / value).resolve()
    try:
        result.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ConditionalPipelineError("output directory escapes repository") from error
    return result


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
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


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _ordered_frame(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ConditionalPipelineError(f"producer rows lack columns: {sorted(missing)}")
    return frame.loc[:, list(columns)]


def _json_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _vectorize_D(objects: np.ndarray) -> np.ndarray:
    array = np.asarray(objects, dtype=np.float64)
    return np.stack([array[..., left, right] for left, right in D_UPPER_TRIANGLE_INDICES], axis=-1)


def _vectorize_G(objects: np.ndarray) -> np.ndarray:
    return svec(np.asarray(objects, dtype=np.float64))


def _vectorizer(object_name: str):
    if object_name == "D":
        return _vectorize_D
    if object_name == "G":
        return _vectorize_G
    raise ConditionalPipelineError("object_name must be D or G")


def _hash_inputs(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_code_snapshot_sha256(
    config: Mapping[str, Any], repo_root: Path, bundle: PhaseGeometryBundle
) -> str:
    """Bind ignored checkpoints to the exact frozen implementation bytes."""

    path = _resolve_output_root(repo_root, config) / "git_provenance.json"
    if not path.is_file():
        # Synthetic low-level tests intentionally exercise the pure producer
        # without a git/protocol freeze. Official scripts cannot reach this
        # branch because their frozen-provenance barrier runs first.
        return hashlib.sha256(
            f"UNFROZEN_TEST_ONLY:{bundle.context.code_commit}".encode("utf-8")
        ).hexdigest()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        snapshot = payload["frozen_code_snapshot"]
        records = snapshot["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ConditionalPipelineError(
            "checkpoint identity requires frozen_code_snapshot provenance"
        ) from error
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("algorithm")
        != "sha256_canonical_json_sorted_file_records_v1"
        or not isinstance(records, list)
        or int(snapshot.get("file_count", -1)) != len(records)
    ):
        raise ConditionalPipelineError("invalid frozen_code_snapshot schema")
    current_records: list[dict[str, Any]] = []
    for expected in records:
        if not isinstance(expected, dict) or set(expected) != {"path", "bytes", "sha256"}:
            raise ConditionalPipelineError("invalid frozen code file record")
        relative = Path(str(expected["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ConditionalPipelineError("frozen code file path escapes repository")
        current = (repo_root / relative).resolve()
        try:
            current.relative_to(repo_root.resolve())
        except ValueError as error:
            raise ConditionalPipelineError("frozen code file path escapes repository") from error
        if current.is_symlink() or not current.is_file():
            raise ConditionalPipelineError(f"frozen code file is missing/invalid: {relative}")
        observed = {
            "path": relative.as_posix(),
            "bytes": int(current.stat().st_size),
            "sha256": sha256_file(current),
        }
        if observed != expected:
            raise ConditionalPipelineError(
                f"current code bytes differ from frozen snapshot: {relative}"
            )
        current_records.append(observed)
    from src.conditional_provenance_v1 import payload_sha256

    current_records = sorted(current_records, key=lambda record: str(record["path"]))
    aggregate = payload_sha256(current_records)
    if (
        aggregate != str(snapshot.get("aggregate_sha256", ""))
        or int(snapshot.get("total_bytes", -1))
        != sum(int(record["bytes"]) for record in current_records)
    ):
        raise ConditionalPipelineError("frozen code snapshot aggregate mismatch")
    return aggregate


def compute_phase_geometry(
    data: ConditionalWholeData,
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    code_commit: str,
    phase: str,
    phase_tag: int,
) -> tuple[PhaseGeometryBundle, dict[str, pd.DataFrame]]:
    """Compute all observed subject/split AIRM and LE objects and audit tables."""

    subjects = np.asarray(config["dataset"]["subjects"], dtype=np.int64)
    classes = np.asarray(config["dataset"]["classes"], dtype=str)
    splits = np.asarray(SPLITS, dtype=str)
    if subjects.shape != (9,) or classes.shape != (4,):
        raise ConditionalPipelineError("frozen subject/class axis contract failed")
    context = _context(
        config,
        config_sha256=config_sha256,
        code_commit=code_commit,
        phase=phase,
        session=data.session,
    )
    thresholds = GeometryThresholds.from_config(config)
    n_channels = int(data.covariances.shape[-1])
    marginal = np.empty((2, 9, 3, n_channels, n_channels), dtype=np.float64)
    class_means = np.empty((2, 9, 3, 4, n_channels, n_channels), dtype=np.float64)
    D = np.empty((2, 9, 3, 4, 4), dtype=np.float64)
    G = np.empty_like(D)
    zD = np.empty((2, 9, 3, 6), dtype=np.float64)
    zG = np.empty((2, 9, 3, 10), dtype=np.float64)
    passed = np.zeros((2, 9, 3), dtype=bool)
    failures: list[list[list[tuple[str, ...]]]] = [
        [[() for _ in SPLITS] for _ in subjects] for _ in GEOMETRIES
    ]

    mean_rows: list[dict[str, Any]] = []
    centering_rows: list[dict[str, Any]] = []
    gauge_rows: list[dict[str, Any]] = []
    degeneracy_rows: list[dict[str, Any]] = []
    d_shape_rows: list[dict[str, Any]] = []
    g_shape_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    radius_rows: list[dict[str, Any]] = []

    for geometry_index, geometry in enumerate(GEOMETRIES):
        for subject_index, subject in enumerate(subjects):
            for split_index, split in enumerate(SPLITS):
                positions = subject_split_positions(
                    data.metadata, config, int(subject), str(split)
                )
                labels = data.metadata.iloc[positions]["class_label"].astype(str).to_numpy()
                result: GeometryObjects = compute_geometry_objects(
                    data.covariances[positions],
                    labels,
                    geometry=geometry,
                    class_order=classes,
                    thresholds=thresholds,
                    gauge_replicate_index=subject_index * len(SPLITS) + split_index,
                    gauge_phase_tag=int(phase_tag),
                )
                marginal[geometry_index, subject_index, split_index] = result.marginal_mean
                class_means[geometry_index, subject_index, split_index] = result.class_means
                D[geometry_index, subject_index, split_index] = result.D
                G[geometry_index, subject_index, split_index] = result.G
                zD[geometry_index, subject_index, split_index] = result.zD
                zG[geometry_index, subject_index, split_index] = result.zG
                passed[geometry_index, subject_index, split_index] = result.gate_passed
                failures[geometry_index][subject_index][split_index] = result.failure_reasons
                row_status = "PASS" if result.gate_passed else "FAIL"
                prefix = context.prefix(row_status)

                for audit in result.mean_audits:
                    mean_rows.append(
                        {
                            **prefix,
                            "geometry": geometry,
                            "subject": int(subject),
                            "split": str(split),
                            "mean_kind": "marginal" if audit.name == "marginal" else "class",
                            "class_label": "" if audit.name == "marginal" else audit.name,
                            "n_samples": int(audit.n_samples),
                            "karcher_residual": audit.karcher_residual,
                            "custom_relative_error": audit.custom_relative_error,
                            "warning_count": len(audit.warning_messages),
                            "warning_messages": _json_list(audit.warning_messages),
                            "spd_passed": audit.spd_audit.all_passed,
                            "passed": audit.passed,
                            "failure_reasons": _json_list(audit.failure_reasons),
                        }
                    )
                metrics = result.gate_metrics
                centering_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "subject": int(subject),
                        "split": str(split),
                        "d_centering_relative_error": metrics["d_centering_relative_error"],
                        "limit": thresholds.d_centering_relative_error_max,
                        "passed": metrics["d_centering_relative_error"]
                        <= thresholds.d_centering_relative_error_max,
                    }
                )
                gauge_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "subject": int(subject),
                        "split": str(split),
                        "d_relative_error": metrics["orthogonal_d_relative_error"],
                        "g_relative_error": metrics["orthogonal_g_relative_error"],
                        "d_limit": thresholds.orthogonal_d_relative_error_max,
                        "g_limit": thresholds.orthogonal_g_relative_error_max,
                        "g_direct_whitened_relative_error": metrics[
                            "g_direct_whitened_relative_error"
                        ],
                        "g_direct_whitened_limit": thresholds.g_direct_whitened_relative_error_max,
                        "permutation_d_relative_error": metrics[
                            "permutation_d_relative_error"
                        ],
                        "permutation_g_relative_error": metrics[
                            "permutation_g_relative_error"
                        ],
                        "permutation_limit": thresholds.permutation_equivariance_relative_error_max,
                        "le_d_g_identity_relative_error": metrics[
                            "le_d_g_identity_relative_error"
                        ],
                        "le_d_g_identity_limit": thresholds.le_d_g_identity_relative_error_max,
                        "d_symmetry_relative_error": metrics["d_symmetry_relative_error"],
                        "d_diagonal_max_abs": metrics["d_diagonal_max_abs"],
                        "d_minimum": metrics["d_minimum"],
                        "g_symmetry_relative_error": metrics["g_symmetry_relative_error"],
                        "structure_tolerance": 1.0e-12,
                        "passed": result.gate_passed,
                    }
                )
                for object_name, shape in (("D", result.d_shape), ("G", result.g_shape)):
                    degeneracy_rows.append(
                        {
                            **prefix,
                            "geometry": geometry,
                            "object": object_name,
                            "subject": int(subject),
                            "split": str(split),
                            "shape_norm": shape.norm,
                            "degeneracy_threshold": shape.degeneracy_threshold,
                            "is_degenerate": shape.is_degenerate,
                            "passed": not shape.is_degenerate,
                        }
                    )
                d_shape_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "subject": int(subject),
                        "split": str(split),
                        "shape_norm": result.d_shape.norm,
                        **{f"raw_{index}": value for index, value in enumerate(result.d_shape.raw_vector)},
                        **{f"z_{index}": value for index, value in enumerate(result.zD)},
                    }
                )
                g_shape_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "subject": int(subject),
                        "split": str(split),
                        "shape_norm": result.g_shape.norm,
                        **{f"raw_{index}": value for index, value in enumerate(result.g_shape.raw_vector)},
                        **{f"z_{index}": value for index, value in enumerate(result.zG)},
                    }
                )
                upper = result.d_shape.raw_vector
                radii_squared = np.diag(result.G)
                radii = np.where(radii_squared >= 0.0, np.sqrt(np.maximum(radii_squared, 0.0)), np.nan)
                scale_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "subject": int(subject),
                        "split": str(split),
                        "D_shape_norm": result.d_shape.norm,
                        "G_shape_norm": result.g_shape.norm,
                        "D_upper_mean": float(np.mean(upper)),
                        "D_upper_max": float(np.max(upper)),
                        "class_radius_mean": float(np.mean(radii)),
                        "class_radius_max": float(np.max(radii)),
                    }
                )
                for left in range(4):
                    for right in range(left + 1, 4):
                        denominator = radii[left] * radii[right]
                        cosine = float(result.G[left, right] / denominator) if denominator > 0 else np.nan
                        angle = float(np.arccos(np.clip(cosine, -1.0, 1.0))) if np.isfinite(cosine) else np.nan
                        radius_rows.append(
                            {
                                **prefix,
                                "geometry": geometry,
                                "subject": int(subject),
                                "split": str(split),
                                "class_left": str(classes[left]),
                                "class_right": str(classes[right]),
                                "radius_left": float(radii[left]),
                                "radius_right": float(radii[right]),
                                "cosine": cosine,
                                "angle_radians": angle,
                            }
                        )

    covariance_rows = _covariance_sanity_rows(data, context, thresholds)
    dataset_rows = _dataset_contract_rows(data, config, context)
    common_mean_columns = COMMON_COLUMNS + (
        "geometry", "subject", "split", "mean_kind", "class_label", "n_samples",
        "karcher_residual", "custom_relative_error", "warning_count", "warning_messages",
        "spd_passed", "passed", "failure_reasons",
    )
    mean_frame = _ordered_frame(mean_rows, common_mean_columns)
    tables = {
        "dataset_contract.csv": pd.DataFrame(dataset_rows),
        "covariance_sanity.csv": pd.DataFrame(covariance_rows),
        "airm_mean_convergence.csv": mean_frame.loc[mean_frame["geometry"] == AIRM].reset_index(drop=True),
        "le_mean_correctness.csv": mean_frame.loc[mean_frame["geometry"] == LE].reset_index(drop=True),
        "centering_isometry_gate.csv": pd.DataFrame(centering_rows),
        "orthogonal_gauge_gate.csv": pd.DataFrame(gauge_rows),
        "degenerate_geometry_audit.csv": pd.DataFrame(degeneracy_rows),
        "D_shape_vectors.csv": pd.DataFrame(d_shape_rows),
        "G_shape_vectors.csv": pd.DataFrame(g_shape_rows),
        "absolute_geometry_scales.csv": pd.DataFrame(scale_rows),
        "radius_angle_summary.csv": pd.DataFrame(radius_rows),
    }
    bundle = PhaseGeometryBundle(
        context=context,
        subjects=subjects,
        splits=splits,
        classes=classes,
        marginal_means=marginal,
        class_means=class_means,
        D=D,
        G=G,
        zD=zD,
        zG=zG,
        gate_passed=passed,
        failure_reasons=tuple(
            tuple(tuple(value) for value in subject_values)
            for subject_values in failures
        ),
    )
    return bundle, tables


def _dataset_contract_rows(
    data: ConditionalWholeData,
    config: Mapping[str, Any],
    context: ProducerContext,
) -> list[dict[str, Any]]:
    frame = data.metadata
    rows: list[dict[str, Any]] = []

    def add(scope: str, observed: int, expected: int, **keys: Any) -> None:
        passed = int(observed) == int(expected)
        rows.append(
            {
                **context.prefix("PASS" if passed else "FAIL"),
                "scope": scope,
                "subject": keys.get("subject", ""),
                "run": keys.get("run", ""),
                "class_label": keys.get("class_label", ""),
                "split": keys.get("split", ""),
                "observed_count": int(observed),
                "expected_count": int(expected),
                "passed": passed,
            }
        )

    add("session", len(frame), int(config["expected_data"]["trials_per_session"]))
    for subject in config["dataset"]["subjects"]:
        subject_frame = frame.loc[frame["subject"].astype(int) == int(subject)]
        add("subject", len(subject_frame), 288, subject=int(subject))
        for run in config["dataset"]["runs"]:
            run_frame = subject_frame.loc[subject_frame["run"].astype(str) == str(run)]
            add("subject_run", len(run_frame), 48, subject=int(subject), run=str(run))
            for class_label in config["dataset"]["classes"]:
                count = int((run_frame["class_label"].astype(str) == str(class_label)).sum())
                add(
                    "subject_run_class", count, 12, subject=int(subject), run=str(run),
                    class_label=str(class_label),
                )
        for split in SPLITS:
            positions = subject_split_positions(frame, config, int(subject), split)
            add(
                "subject_split", len(positions),
                int(config["expected_data"]["split_trials_per_subject"][split]),
                subject=int(subject), split=split,
            )
    return rows


def _covariance_sanity_rows(
    data: ConditionalWholeData,
    context: ProducerContext,
    thresholds: GeometryThresholds,
) -> list[dict[str, Any]]:
    array = np.asarray(data.covariances, dtype=np.float64)
    norms = np.linalg.norm(array, axis=(1, 2))
    symmetry = np.linalg.norm(array - array.transpose(0, 2, 1), axis=(1, 2)) / np.maximum(
        norms, np.finfo(np.float64).tiny
    )
    eigvals = np.linalg.eigvalsh(array)
    condition = eigvals[:, -1] / eigvals[:, 0]
    finite = np.isfinite(array).all(axis=(1, 2))
    passed = (
        finite
        & (symmetry <= thresholds.symmetry_relative_error_max)
        & (eigvals[:, 0] > 0.0)
        & (condition <= thresholds.condition_number_max)
    )
    rows: list[dict[str, Any]] = []
    for index, metadata in data.metadata.iterrows():
        rows.append(
            {
                **context.prefix("PASS" if passed[index] else "FAIL"),
                "covariance_index": int(index),
                "trial_uid": str(metadata["trial_uid"]),
                "subject": int(metadata["subject"]),
                "run": str(metadata["run"]),
                "class_label": str(metadata["class_label"]),
                "finite": bool(finite[index]),
                "symmetry_relative_error": float(symmetry[index]),
                "min_eigenvalue": float(eigvals[index, 0]),
                "max_eigenvalue": float(eigvals[index, -1]),
                "condition_number": float(condition[index]),
                "passed": bool(passed[index]),
            }
        )
    return rows


def write_phase_geometry_outputs(
    bundle: PhaseGeometryBundle,
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Atomically emit the exact six object basenames and geometry tables."""

    root = Path(repo_root).resolve()
    output_root = _resolve_output_root(root, config)
    object_dir = output_root / "objects" / bundle.context.phase
    table_dir = output_root / "tables" / bundle.context.phase
    common_axes = {
        "subjects": bundle.subjects.astype(np.int64),
        "splits": bundle.splits.astype(str),
        "classes": bundle.classes.astype(str),
    }
    for geometry_index, stem in ((0, "airm"), (1, "le")):
        geometry_axis = np.asarray([GEOMETRIES[geometry_index]], dtype=str)
        _atomic_savez(
            object_dir / f"{stem}_marginal_means.npz",
            marginal_means=bundle.marginal_means[geometry_index],
            geometries=geometry_axis,
            **common_axes,
        )
        _atomic_savez(
            object_dir / f"{stem}_class_means.npz",
            class_means=bundle.class_means[geometry_index],
            geometries=geometry_axis,
            **common_axes,
        )
    object_axes = {
        "geometries": np.asarray(GEOMETRIES, dtype=str),
        **common_axes,
    }
    _atomic_savez(object_dir / "D_matrices.npz", matrices=bundle.D, **object_axes)
    _atomic_savez(object_dir / "G_matrices.npz", matrices=bundle.G, **object_axes)
    for filename, frame in tables.items():
        _atomic_write_csv(table_dir / filename, frame)
    manifest = {
        **bundle.context.prefix("PASS" if bundle.all_gates_passed else "FAIL"),
        "schema_version": "conditional-phase-geometry-v1",
        "subjects": bundle.subjects.tolist(),
        "splits": bundle.splits.tolist(),
        "classes": bundle.classes.tolist(),
        "geometries": list(GEOMETRIES),
        "all_gates_passed": bundle.all_gates_passed,
        "failed_object_count": int(np.count_nonzero(~bundle.gate_passed)),
        "files": {},
    }
    for path in sorted([*object_dir.glob("*.npz"), *table_dir.glob("*.csv")]):
        manifest["files"][path.name] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return manifest


def load_phase_geometry_bundle(
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    phase: str,
    session: str,
    config_sha256: str,
    code_commit: str,
) -> PhaseGeometryBundle:
    """Load and strictly validate the six required object archives."""

    root = Path(repo_root).resolve()
    object_dir = _resolve_output_root(root, config) / "objects" / phase
    subjects = np.asarray(config["dataset"]["subjects"], dtype=np.int64)
    splits = np.asarray(SPLITS, dtype=str)
    classes = np.asarray(config["dataset"]["classes"], dtype=str)

    def read(path: Path, keys: set[str]) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != keys:
                raise ConditionalPipelineError(
                    f"object archive key mismatch at {path.name}: {sorted(archive.files)}"
                )
            return {key: np.array(archive[key], copy=True) for key in archive.files}

    axis_keys = {"geometries", "subjects", "splits", "classes"}
    marginal_values: list[np.ndarray] = []
    class_values: list[np.ndarray] = []
    for geometry, stem in zip(GEOMETRIES, ("airm", "le"), strict=True):
        mean = read(object_dir / f"{stem}_marginal_means.npz", {"marginal_means", *axis_keys})
        class_mean = read(object_dir / f"{stem}_class_means.npz", {"class_means", *axis_keys})
        for payload in (mean, class_mean):
            if (
                payload["geometries"].astype(str).tolist() != [geometry]
                or not np.array_equal(payload["subjects"], subjects)
                or payload["splits"].astype(str).tolist() != list(SPLITS)
                or payload["classes"].astype(str).tolist() != classes.tolist()
            ):
                raise ConditionalPipelineError(f"axis mismatch in {stem} mean archive")
        marginal_values.append(np.asarray(mean["marginal_means"], dtype=np.float64))
        class_values.append(np.asarray(class_mean["class_means"], dtype=np.float64))
    d_payload = read(object_dir / "D_matrices.npz", {"matrices", *axis_keys})
    g_payload = read(object_dir / "G_matrices.npz", {"matrices", *axis_keys})
    for payload in (d_payload, g_payload):
        if (
            payload["geometries"].astype(str).tolist() != list(GEOMETRIES)
            or not np.array_equal(payload["subjects"], subjects)
            or payload["splits"].astype(str).tolist() != list(SPLITS)
            or payload["classes"].astype(str).tolist() != classes.tolist()
        ):
            raise ConditionalPipelineError("D/G object archive axis mismatch")
    marginal = np.stack(marginal_values)
    class_means = np.stack(class_values)
    D = np.asarray(d_payload["matrices"], dtype=np.float64)
    G = np.asarray(g_payload["matrices"], dtype=np.float64)
    expected_p = int(config["expected_data"]["covariance_shape_per_session"][-1])
    if marginal.shape != (2, 9, 3, expected_p, expected_p):
        raise ConditionalPipelineError(f"marginal mean shape mismatch: {marginal.shape}")
    if class_means.shape != (2, 9, 3, 4, expected_p, expected_p):
        raise ConditionalPipelineError(f"class mean shape mismatch: {class_means.shape}")
    if D.shape != (2, 9, 3, 4, 4) or G.shape != D.shape:
        raise ConditionalPipelineError("D/G matrix shape mismatch")
    zD = np.empty((2, 9, 3, 6), dtype=np.float64)
    zG = np.empty((2, 9, 3, 10), dtype=np.float64)
    for geometry_index in range(2):
        for subject_index in range(9):
            for split_index in range(3):
                zD[geometry_index, subject_index, split_index] = shape_from_D(
                    D[geometry_index, subject_index, split_index]
                ).unit_vector
                zG[geometry_index, subject_index, split_index] = shape_from_G(
                    G[geometry_index, subject_index, split_index]
                ).unit_vector
    context = _context(
        config,
        config_sha256=config_sha256,
        code_commit=code_commit,
        phase=phase,
        session=session,
    )
    table_dir = _resolve_output_root(root, config) / "tables" / phase
    gate_files = (
        "dataset_contract.csv",
        "covariance_sanity.csv",
        "airm_mean_convergence.csv",
        "le_mean_correctness.csv",
        "centering_isometry_gate.csv",
        "orthogonal_gauge_gate.csv",
        "degenerate_geometry_audit.csv",
    )
    for filename in gate_files:
        frame = pd.read_csv(table_dir / filename, float_precision="round_trip")
        if frame.empty or "passed" not in frame or not frame["passed"].astype(bool).all():
            raise ConditionalPipelineError(f"scientific nulls blocked by {filename}")
        for key, value in context.prefix().items():
            if key == "status":
                continue
            if key not in frame or set(frame[key].astype(str)) != {str(value)}:
                raise ConditionalPipelineError(f"geometry table provenance mismatch: {filename}:{key}")
    return PhaseGeometryBundle(
        context=context,
        subjects=subjects,
        splits=splits,
        classes=classes,
        marginal_means=marginal,
        class_means=class_means,
        D=D,
        G=G,
        zD=zD,
        zG=zG,
        gate_passed=np.ones((2, 9, 3), dtype=bool),
        failure_reasons=tuple(
            tuple(tuple(() for _ in range(3)) for _ in range(9)) for _ in range(2)
        ),
    )


def validate_discovery_snapshot_contract(
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    config_sha256: str,
    code_commit: str,
    phase: str = "discovery",
    session: str | None = None,
) -> dict[str, Any]:
    """Validate scientific content before its hashes may unlock confirmation.

    This function reads only the configured discovery snapshot directories.  It
    has no data-loader call and cannot resolve any raw-session input.
    """

    root = Path(repo_root).resolve()
    output = _resolve_output_root(root, config)
    if phase not in ("discovery", "confirmatory"):
        raise ConditionalPipelineError("snapshot phase must be discovery or confirmatory")
    if session is None:
        session = str(
            config["dataset"][
                "discovery_session" if phase == "discovery" else "confirmatory_session"
            ]
        )
    object_dir = output / "objects" / phase
    table_dir = output / "tables" / phase
    null_dir = output / "nulls" / phase
    expected_objects = {
        "airm_marginal_means.npz",
        "airm_class_means.npz",
        "le_marginal_means.npz",
        "le_class_means.npz",
        "D_matrices.npz",
        "G_matrices.npz",
    }
    expected_nulls = {
        "label_destruction_group_statistics.npz",
        "semantic_permutation_group_statistics.npz",
        "oracle_rank_null.npz",
    }
    observed_objects = {path.name for path in object_dir.iterdir() if path.is_file()}
    observed_tables = {path.name for path in table_dir.iterdir() if path.is_file()}
    observed_nulls = {path.name for path in null_dir.iterdir() if path.is_file()}
    if observed_objects != expected_objects:
        raise ConditionalPipelineError(
            f"discovery object basename contract failed: {sorted(observed_objects)}"
        )
    if observed_tables != set(DISCOVERY_TABLE_SCHEMAS):
        raise ConditionalPipelineError(
            f"discovery table basename contract failed: {sorted(observed_tables)}"
        )
    if observed_nulls != expected_nulls:
        raise ConditionalPipelineError(
            f"discovery null basename contract failed: {sorted(observed_nulls)}"
        )

    expected_prefix = {
        "protocol_version": str(config["protocol"]["version"]),
        "protocol_sha256": str(config["protocol"]["protocol_sha256"]),
        "config_sha256": str(config_sha256),
        "code_commit": str(code_commit),
        "phase": phase,
        "session": str(session),
        "status": "PASS",
    }
    frames: dict[str, pd.DataFrame] = {}
    for filename, schema in DISCOVERY_TABLE_SCHEMAS.items():
        frame = pd.read_csv(table_dir / filename, float_precision="round_trip")
        if tuple(frame.columns) != schema or frame.empty:
            raise ConditionalPipelineError(f"discovery CSV schema/empty failure: {filename}")
        for key, expected in expected_prefix.items():
            if set(frame[key].astype(str)) != {str(expected)}:
                raise ConditionalPipelineError(
                    f"discovery CSV provenance/status failure: {filename}:{key}"
                )
        frames[filename] = frame

    def strict_booleans(frame: pd.DataFrame, column: str, filename: str) -> np.ndarray:
        values = frame[column]
        if values.isna().any():
            raise ConditionalPipelineError(f"missing boolean in {filename}:{column}")
        if pd.api.types.is_bool_dtype(values.dtype):
            return values.to_numpy(dtype=bool)
        normalized = values.astype(str).str.strip().str.lower()
        if not normalized.isin(("true", "false")).all():
            raise ConditionalPipelineError(f"invalid boolean literal in {filename}:{column}")
        return normalized.eq("true").to_numpy(dtype=bool)

    def unique_grid(filename: str, columns: Sequence[str], expected_rows: int) -> None:
        frame = frames[filename]
        if frame.duplicated(list(columns)).any() or len(frame) != expected_rows:
            raise ConditionalPipelineError(f"non-unique discovery grid: {filename}")

    exact_rows = {
        "dataset_contract.csv": 307,
        "covariance_sanity.csv": 2592,
        "airm_mean_convergence.csv": 135,
        "le_mean_correctness.csv": 135,
        "centering_isometry_gate.csv": 54,
        "orthogonal_gauge_gate.csv": 54,
        "degenerate_geometry_audit.csv": 108,
        "D_shape_vectors.csv": 54,
        "G_shape_vectors.csv": 54,
        "absolute_geometry_scales.csv": 54,
        "radius_angle_summary.csv": 324,
        "within_subject_reliability.csv": 36,
        "label_destruction_subject_summary.csv": 36,
        "label_destruction_group_summary.csv": 4,
        "unrelated_subject_derangement_summary.csv": 4,
        "cross_subject_shared_geometry.csv": 36,
        "semantic_permutation_null_summary.csv": 4,
        "oracle_permutation_all_24_scores.csv": 864,
        "oracle_permutation_subject_summary.csv": 36,
        "oracle_permutation_group_summary.csv": 4,
        "subject_bootstrap_summary.csv": 12,
        "leave_one_subject_out_influence.csv": 108,
    }
    for filename, row_count in exact_rows.items():
        if len(frames[filename]) != row_count:
            raise ConditionalPipelineError(
                f"discovery CSV row-grid failure: {filename} has {len(frames[filename])}"
            )
    unique_grid("covariance_sanity.csv", ("covariance_index",), 2592)
    for filename in ("airm_mean_convergence.csv", "le_mean_correctness.csv"):
        unique_grid(filename, ("geometry", "subject", "split", "mean_kind", "class_label"), 135)
    unique_grid("centering_isometry_gate.csv", ("geometry", "subject", "split"), 54)
    unique_grid("orthogonal_gauge_gate.csv", ("geometry", "subject", "split"), 54)
    unique_grid("degenerate_geometry_audit.csv", ("geometry", "object", "subject", "split"), 108)
    unique_grid("within_subject_reliability.csv", ("geometry", "object", "subject"), 36)
    unique_grid("label_destruction_group_summary.csv", ("geometry", "object", "stage"), 4)
    unique_grid("semantic_permutation_null_summary.csv", ("geometry", "object", "stage"), 4)
    unique_grid("oracle_permutation_group_summary.csv", ("geometry", "object", "stage"), 4)
    unique_grid(
        "oracle_permutation_all_24_scores.csv",
        ("geometry", "object", "subject", "permutation_index"),
        864,
    )
    expected_template_rows = 3 * 2 * 9 * (6 + 10)
    if len(frames["loso_templates.csv"]) != expected_template_rows:
        raise ConditionalPipelineError("discovery LOSO template row-grid failure")
    for filename in (
        "dataset_contract.csv",
        "covariance_sanity.csv",
        "airm_mean_convergence.csv",
        "le_mean_correctness.csv",
        "centering_isometry_gate.csv",
        "orthogonal_gauge_gate.csv",
        "degenerate_geometry_audit.csv",
    ):
        if not strict_booleans(frames[filename], "passed", filename).all():
            raise ConditionalPipelineError(f"discovery hard gate is not all PASS: {filename}")
    if not strict_booleans(
        frames["label_destruction_group_summary.csv"],
        "gate_pass",
        "label_destruction_group_summary.csv",
    ).all():
        raise ConditionalPipelineError("discovery R group gate flag failed")
    if not strict_booleans(
        frames["semantic_permutation_null_summary.csv"],
        "gate_pass",
        "semantic_permutation_null_summary.csv",
    ).all():
        raise ConditionalPipelineError("discovery S group gate flag failed")
    if not strict_booleans(
        frames["oracle_permutation_group_summary.csv"],
        "gate_pass",
        "oracle_permutation_group_summary.csv",
    ).all():
        raise ConditionalPipelineError("discovery P group gate flag failed")

    subjects = np.asarray(config["dataset"]["subjects"], dtype=np.int64)
    splits = np.asarray(SPLITS, dtype=str)
    classes = np.asarray(config["dataset"]["classes"], dtype=str)
    expected_p = int(config["expected_data"]["covariance_shape_per_session"][-1])

    def validate_axes(archive: Any, geometries: list[str]) -> None:
        if (
            archive["geometries"].astype(str).tolist() != geometries
            or not np.array_equal(archive["subjects"], subjects)
            or archive["splits"].astype(str).tolist() != list(SPLITS)
            or archive["classes"].astype(str).tolist() != classes.tolist()
        ):
            raise ConditionalPipelineError("discovery object NPZ axis contract failed")

    thresholds = GeometryThresholds.from_config(config)
    saved_marginal = np.empty((2, 9, 3, expected_p, expected_p), dtype=np.float64)
    saved_class = np.empty((2, 9, 3, 4, expected_p, expected_p), dtype=np.float64)
    for geometry_index, (geometry, stem) in enumerate(
        zip(GEOMETRIES, ("airm", "le"), strict=True)
    ):
        for kind, expected_shape in (
            ("marginal", (9, 3, expected_p, expected_p)),
            ("class", (9, 3, 4, expected_p, expected_p)),
        ):
            path = object_dir / f"{stem}_{kind}_means.npz"
            value_key = f"{kind}_means"
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {value_key, "geometries", "subjects", "splits", "classes"}:
                    raise ConditionalPipelineError(f"discovery object key contract failed: {path.name}")
                validate_axes(archive, [geometry])
                values = np.asarray(archive[value_key], dtype=np.float64)
                if values.shape != expected_shape or not np.isfinite(values).all():
                    raise ConditionalPipelineError(f"discovery object matrix failure: {path.name}")
                spd_audit = validate_spd_stack(values, thresholds=thresholds)
                if not spd_audit.all_passed:
                    raise ConditionalPipelineError(
                        f"discovery saved mean SPD gate failed: {path.name}"
                    )
                if kind == "marginal":
                    saved_marginal[geometry_index] = values
                else:
                    saved_class[geometry_index] = values
    saved_objects: dict[str, np.ndarray] = {}
    for object_name in OBJECTS:
        path = object_dir / f"{object_name}_matrices.npz"
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"matrices", "geometries", "subjects", "splits", "classes"}:
                raise ConditionalPipelineError(f"discovery object key contract failed: {path.name}")
            validate_axes(archive, list(GEOMETRIES))
            values = np.asarray(archive["matrices"])
            if values.shape != (2, 9, 3, 4, 4) or not np.isfinite(values).all():
                raise ConditionalPipelineError(f"discovery {object_name} matrix failure")
            saved_objects[object_name] = values.astype(np.float64, copy=True)

    # The mean-audit tables remain independently bound to the frozen numerical
    # thresholds even though the original trial covariances are intentionally
    # absent from this compact snapshot validator.
    for geometry, filename in (
        (AIRM, "airm_mean_convergence.csv"),
        (LE, "le_mean_correctness.csv"),
    ):
        mean_frame = frames[filename]
        expected_samples = mean_frame.apply(
            lambda row: (
                int(config["expected_data"]["split_trials_per_subject"][str(row["split"])])
                if str(row["mean_kind"]) == "marginal"
                else int(
                    config["expected_data"]["split_trials_per_subject_class"][
                        str(row["split"])
                    ]
                )
            ),
            axis=1,
        ).to_numpy(dtype=np.int64)
        if (
            set(mean_frame["geometry"].astype(str)) != {geometry}
            or not np.array_equal(
                mean_frame["n_samples"].to_numpy(dtype=np.int64), expected_samples
            )
            or np.any(mean_frame["warning_count"].to_numpy(dtype=np.int64) != 0)
            or not strict_booleans(mean_frame, "spd_passed", filename).all()
        ):
            raise ConditionalPipelineError(f"saved mean audit mismatch: {filename}")
        if geometry == AIRM:
            residuals = mean_frame["karcher_residual"].to_numpy(dtype=np.float64)
            if (
                not np.isfinite(residuals).all()
                or np.any(residuals > thresholds.airm_karcher_residual_max)
                or mean_frame["custom_relative_error"].notna().any()
            ):
                raise ConditionalPipelineError("saved AIRM convergence audit mismatch")
        else:
            custom_errors = mean_frame["custom_relative_error"].to_numpy(dtype=np.float64)
            if (
                not np.isfinite(custom_errors).all()
                or np.any(custom_errors > thresholds.le_mean_relative_error_max)
                or mean_frame["karcher_residual"].notna().any()
            ):
                raise ConditionalPipelineError("saved LE mean audit mismatch")

    recomputed_D = np.empty_like(saved_objects["D"])
    recomputed_G = np.empty_like(saved_objects["G"])
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for subject_index in range(9):
            for split_index in range(3):
                marginal = saved_marginal[geometry_index, subject_index, split_index]
                class_values = saved_class[geometry_index, subject_index, split_index]
                if geometry == AIRM:
                    recomputed_D[geometry_index, subject_index, split_index] = (
                        airm_distance_matrix(class_values)
                    )
                    recomputed_G[geometry_index, subject_index, split_index] = (
                        airm_gram_matrices(marginal, class_values)[0]
                    )
                else:
                    recomputed_D[geometry_index, subject_index, split_index] = (
                        le_distance_matrix(class_values)
                    )
                    recomputed_G[geometry_index, subject_index, split_index] = (
                        le_gram_matrix(marginal, class_values)[0]
                    )
    for object_name, recomputed in (("D", recomputed_D), ("G", recomputed_G)):
        if not np.allclose(
            saved_objects[object_name], recomputed, rtol=5.0e-12, atol=5.0e-14
        ):
            error = float(np.max(np.abs(saved_objects[object_name] - recomputed)))
            raise ConditionalPipelineError(
                f"saved {object_name} is inconsistent with saved means: max_abs={error:.6g}"
            )

    epsilon_multiplier = float(config["geometry"]["shape_degeneracy"]["epsilon_multiplier"])
    for object_name, recomputed, dimension in (
        ("D", recomputed_D, 6),
        ("G", recomputed_G, 10),
    ):
        shape_frame = frames[f"{object_name}_shape_vectors.csv"].copy()
        shape_frame["subject"] = pd.to_numeric(
            shape_frame["subject"], errors="raise"
        ).astype(int)
        shape_frame = shape_frame.set_index(["geometry", "subject", "split"])
        for geometry_index, geometry in enumerate(GEOMETRIES):
            for subject_index, subject in enumerate(subjects):
                for split_index, split in enumerate(SPLITS):
                    key = (geometry, int(subject), str(split))
                    if key not in shape_frame.index:
                        raise ConditionalPipelineError(
                            f"missing saved {object_name} shape row: {key}"
                        )
                    row = shape_frame.loc[key]
                    if isinstance(row, pd.DataFrame):
                        raise ConditionalPipelineError(
                            f"duplicate saved {object_name} shape row: {key}"
                        )
                    shape = (
                        shape_from_D(
                            recomputed[geometry_index, subject_index, split_index],
                            epsilon_multiplier=epsilon_multiplier,
                        )
                        if object_name == "D"
                        else shape_from_G(
                            recomputed[geometry_index, subject_index, split_index],
                            epsilon_multiplier=epsilon_multiplier,
                        )
                    )
                    raw = row[[f"raw_{index}" for index in range(dimension)]].to_numpy(
                        dtype=np.float64
                    )
                    unit = row[[f"z_{index}" for index in range(dimension)]].to_numpy(
                        dtype=np.float64
                    )
                    if (
                        shape.is_degenerate
                        or not np.isclose(
                            float(row["shape_norm"]), shape.norm, rtol=5.0e-12, atol=5.0e-14
                        )
                        or not np.allclose(raw, shape.raw_vector, rtol=5.0e-12, atol=5.0e-14)
                        or not np.allclose(unit, shape.unit_vector, rtol=5.0e-12, atol=5.0e-14)
                    ):
                        raise ConditionalPipelineError(
                            f"saved {object_name} shape vector mismatch: {key}"
                        )

    null_specs = {
        "label_destruction_group_statistics.npz": (
            int(config["nulls"]["label_destruction"]["replicates"]),
            "label_destruction_group_summary.csv",
            "R",
        ),
        "semantic_permutation_group_statistics.npz": (
            int(config["nulls"]["semantic_permutation"]["replicates"]),
            "semantic_permutation_null_summary.csv",
            "S",
        ),
        "oracle_rank_null.npz": (
            int(config["nulls"]["oracle_rank"]["replicates"]),
            "oracle_permutation_group_summary.csv",
            "P",
        ),
    }
    for filename, (replicates, summary_name, stage) in null_specs.items():
        with np.load(null_dir / filename, allow_pickle=False) as archive:
            if set(archive.files) != {
                "replicate_indices", "geometries", "objects", "group_statistics"
            }:
                raise ConditionalPipelineError(f"discovery null key contract failed: {filename}")
            indices = np.asarray(archive["replicate_indices"], dtype=np.int64)
            groups = np.asarray(archive["group_statistics"], dtype=np.float64)
            if (
                not np.array_equal(indices, np.arange(replicates, dtype=np.int64))
                or archive["geometries"].astype(str).tolist() != list(GEOMETRIES)
                or archive["objects"].astype(str).tolist() != list(OBJECTS)
                or groups.shape != (2, 2, replicates)
                or not np.isfinite(groups).all()
            ):
                raise ConditionalPipelineError(f"discovery null array contract failed: {filename}")
        summary = frames[summary_name]
        for geometry_index, geometry in enumerate(GEOMETRIES):
            for object_index, object_name in enumerate(OBJECTS):
                row = summary.loc[
                    (summary["geometry"] == geometry)
                    & (summary["object"] == object_name)
                    & (summary["stage"] == stage)
                ]
                if len(row) != 1 or int(row.iloc[0]["replicates"]) != replicates:
                    raise ConditionalPipelineError(f"discovery group row-grid failed: {summary_name}")
                null = groups[geometry_index, object_index]
                observed = float(row.iloc[0]["observed"])
                expected_median = float(np.median(null))
                expected_exceed = int(np.count_nonzero(null >= observed))
                expected_p = (1 + expected_exceed) / (replicates + 1)
                checks = (
                    np.isclose(float(row.iloc[0]["null_median"]), expected_median, rtol=1e-13, atol=1e-15),
                    np.isclose(float(row.iloc[0]["effect"]), observed - expected_median, rtol=1e-13, atol=1e-15),
                    int(row.iloc[0]["exceedances"]) == expected_exceed,
                    np.isclose(float(row.iloc[0]["p_value"]), expected_p, rtol=0.0, atol=1e-15),
                )
                if not all(checks):
                    raise ConditionalPipelineError(f"discovery group/null mismatch: {summary_name}")
    if phase == "discovery":
        _validate_label_null_dry_run_context(
            config,
            root,
            _context(
                config,
                config_sha256=config_sha256,
                code_commit=code_commit,
                phase="discovery",
                session=str(session),
            ),
        )
    return {
        "status": "PASS",
        "object_file_count": len(expected_objects),
        "table_file_count": len(DISCOVERY_TABLE_SCHEMAS),
        "null_file_count": len(expected_nulls),
        "official_replicates": {
            "label": int(config["nulls"]["label_destruction"]["replicates"]),
            "semantic": int(config["nulls"]["semantic_permutation"]["replicates"]),
            "oracle": int(config["nulls"]["oracle_rank"]["replicates"]),
        },
    }


def validate_confirmatory_snapshot_contract(
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    config_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    return validate_discovery_snapshot_contract(
        config,
        repo_root,
        config_sha256=config_sha256,
        code_commit=code_commit,
        phase="confirmatory",
        session=str(config["dataset"]["confirmatory_session"]),
    )


def validate_label_null_dry_run(
    config: Mapping[str, Any],
    repo_root: str | Path,
    bundle: PhaseGeometryBundle,
) -> dict[str, Any]:
    return _validate_label_null_dry_run_context(
        config, repo_root, bundle.context
    )


def _validate_label_null_dry_run_context(
    config: Mapping[str, Any],
    repo_root: str | Path,
    context: ProducerContext,
) -> dict[str, Any]:
    path = _resolve_output_root(Path(repo_root).resolve(), config) / "protocol" / "label_null_dry_run.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConditionalPipelineError(
            "official null run requires a valid protocol/label_null_dry_run.json"
        ) from error
    expected = context.prefix("DRY_RUN_ONLY")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ConditionalPipelineError(f"dry-run provenance/status mismatch: {key}")
    crosscheck = payload.get("airm_scalar_crosscheck")
    errors = crosscheck.get("relative_errors", []) if isinstance(crosscheck, dict) else []
    if (
        int(payload.get("replicates", 0)) < 1
        or payload.get("scientific_output_written") is not False
        or not isinstance(crosscheck, dict)
        or crosscheck.get("passed") is not True
        or crosscheck.get("official_scalar_authoritative") is not True
        or crosscheck.get("authoritative_solver")
        != "pyriemann.geometry.mean.mean_riemann"
        or crosscheck.get("official_all_72_pass") is not True
        or int(crosscheck.get("official_warning_count", -1)) != 0
        or int(crosscheck.get("groups_checked", 0)) != 72
        or not np.isfinite(float(crosscheck.get("maximum_official_karcher_residual", np.nan)))
        or float(crosscheck.get("maximum_official_karcher_residual", np.inf))
        > float(config["hard_gates"]["airm_karcher_residual_max"])
        or not errors
        or not np.isfinite(np.asarray(errors, dtype=np.float64)).all()
    ):
        raise ConditionalPipelineError("dry-run AIRM official-scalar authority audit did not pass")
    return payload


def run_phase_geometry_producer(
    data: ConditionalWholeData,
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    repo_root: str | Path,
    phase: str,
    phase_tag: int,
) -> dict[str, Any]:
    code_commit = producer_code_commit(repo_root)
    bundle, tables = compute_phase_geometry(
        data,
        config,
        config_sha256=config_sha256,
        code_commit=code_commit,
        phase=phase,
        phase_tag=phase_tag,
    )
    manifest = write_phase_geometry_outputs(bundle, tables, config, repo_root)
    snapshot_digest = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "phase": phase,
        "session": data.session,
        "all_gates_passed": bundle.all_gates_passed,
        "failed_object_count": int(np.count_nonzero(~bundle.gate_passed)),
        "geometry_snapshot_sha256": snapshot_digest,
        "table_count": len(tables),
        "object_file_count": 6,
        "manifest": manifest,
    }
    if not bundle.all_gates_passed:
        reasons = sorted(
            {
                str(reason)
                for geometry_values in bundle.failure_reasons
                for subject_values in geometry_values
                for split_values in subject_values
                for reason in split_values
            }
        )
        failure_class = (
            "degenerate"
            if reasons
            and all(value.startswith("DEGENERATE_CLASS_GEOMETRY") for value in reasons)
            else "numerical"
        )
        decision = write_unassessed_failure_artifacts(
            config,
            repo_root,
            config_sha256=config_sha256,
            code_commit=code_commit,
            phase=phase,
            session=data.session,
            failure_class=failure_class,
            reason_code="GEOMETRY_HARD_GATE_FAILURE",
            reason=json.dumps(reasons, sort_keys=True, separators=(",", ":")),
        )
        result["terminal_decision"] = decision["terminal_decision"]
        result["failure_manifest"] = decision["failure_manifest"]
    return result


def _label_null_one(
    replicate_index: int,
    data: ConditionalWholeData,
    covariance_logs: np.ndarray,
    bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    phase_tag: int,
    *,
    run_batched_audit: bool = False,
) -> LabelNullReplicateResult:
    """Return R subject scores with axes (geometry, object, subject)."""

    metadata = data.metadata
    permuted = shuffle_labels_within_strata(
        metadata["class_label"].astype(str).to_numpy(),
        metadata["subject"].to_numpy(dtype=np.int64),
        metadata["session"].astype(str).to_numpy(),
        metadata["run"].astype(str).to_numpy(),
        metadata["trial_uid"].astype(str).to_numpy(),
        replicate_index=int(replicate_index),
        phase_tag=int(phase_tag),
        master_seed=int(config["protocol"]["seed"]),
    )
    grouped_indices = np.empty((9, 2, 4, 36), dtype=np.int64)
    for subject_index, subject in enumerate(bundle.subjects):
        for split_index, split in enumerate(("A", "B")):
            positions = subject_split_positions(metadata, config, int(subject), split)
            for class_index, class_label in enumerate(bundle.classes):
                selected = positions[permuted[positions] == str(class_label)]
                if selected.shape != (36,):
                    raise ConditionalPipelineError(
                        "label null violated frozen 36-trial split/class balance"
                    )
                grouped_indices[subject_index, split_index, class_index] = selected

    grouped_covariances = np.asarray(data.covariances)[grouped_indices]
    thresholds = GeometryThresholds.from_config(config)
    # The public pyRiemann scalar implementation is the scientific authority
    # for every one of the 9 subject x 2 split x 4 class groups.  A prior
    # batched implementation can satisfy a residual gate yet differ from the
    # public solver on ill-conditioned (but legal) inputs, so it is never used
    # to construct a null statistic.
    airm_classes = np.empty(
        grouped_covariances.shape[:-3] + grouped_covariances.shape[-2:],
        dtype=np.float64,
    )
    official_residuals: list[float] = []
    official_warning_count = 0
    for group_index in np.ndindex(grouped_covariances.shape[:3]):
        group = grouped_covariances[group_index]
        # Input covariance SPD/symmetry/finite gates were already applied to
        # the immutable phase data.  Avoid repeating their 36x eigendecomposition
        # here, but gate every public-solver output and its independent residual.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            matrix = np.asarray(
                mean_riemann(
                    group,
                    tol=thresholds.mean_tol,
                    maxiter=thresholds.mean_maxiter,
                    init=None,
                ),
                dtype=np.float64,
            )
        matrix = symmetrize(matrix)
        spd_audit = validate_spd_stack(matrix, thresholds=thresholds)
        residual = karcher_residual(group, matrix)
        official_residuals.append(float(residual))
        official_warning_count += len(caught)
        if (
            caught
            or not spd_audit.all_passed
            or not np.isfinite(residual)
            or residual > thresholds.airm_karcher_residual_max
        ):
            raise ConditionalPipelineError(
                "label-null official AIRM class mean gate failed at "
                f"replicate {replicate_index}, group {group_index}: "
                f"warnings={tuple(str(item.message) for item in caught)}, "
                f"residual={residual:.6g}, spd={spd_audit.all_passed}"
            )
        airm_classes[group_index] = matrix
    airm_D = np.asarray(
        airm_distance(airm_classes[..., :, None, :, :], airm_classes[..., None, :, :, :]),
        dtype=np.float64,
    )
    airm_marginal = bundle.marginal_means[0, :, :2]
    inverse_root = spd_invsqrt(airm_marginal)
    whitened = inverse_root[:, :, None] @ airm_classes @ inverse_root[:, :, None]
    tangents = spd_log(whitened)
    airm_G = np.einsum("sacij,sadji->sacd", tangents, tangents, optimize=True)

    le_class_logs = covariance_logs[grouped_indices].mean(axis=3)
    le_D = np.linalg.norm(
        le_class_logs[..., :, None, :, :] - le_class_logs[..., None, :, :, :],
        axis=(-2, -1),
    )
    le_marginal_logs = spd_log(bundle.marginal_means[1, :, :2])
    centered_logs = le_class_logs - le_marginal_logs[:, :, None]
    le_G = np.einsum("sacij,sadji->sacd", centered_logs, centered_logs, optimize=True)

    outputs = np.empty((2, 2, 9), dtype=np.float64)
    for geometry_index, (D_values, G_values) in enumerate(
        ((airm_D, airm_G), (le_D, le_G))
    ):
        d_shapes = normalize_shape_vectors(_vectorize_D(D_values))
        g_shapes = normalize_shape_vectors(_vectorize_G(G_values))
        outputs[geometry_index, 0] = np.sum(d_shapes[:, 0] * d_shapes[:, 1], axis=1)
        outputs[geometry_index, 1] = np.sum(g_shapes[:, 0] * g_shapes[:, 1], axis=1)
    if not np.isfinite(outputs).all():
        raise ConditionalPipelineError("label-null reliability statistic is non-finite")
    crosscheck: dict[str, Any] | None = None
    if run_batched_audit:
        batched_fit = airm_mean_batched(
            grouped_covariances,
            thresholds=thresholds,
            scalar_crosscheck=False,
        )
        official_matrices = airm_classes.reshape((-1,) + airm_classes.shape[-2:])
        batched_matrices = batched_fit.matrices.reshape((-1,) + airm_classes.shape[-2:])
        denominators = np.maximum(
            np.linalg.norm(official_matrices, axis=(-2, -1)),
            np.finfo(np.float64).tiny,
        )
        flat_errors = np.linalg.norm(
            batched_matrices - official_matrices, axis=(-2, -1)
        ) / denominators
        tolerance = 1.0e-10
        crosscheck = {
            "replicate_index": int(replicate_index),
            "canonical_flat_group_indices": list(range(len(flat_errors))),
            "groups_checked": int(len(flat_errors)),
            "relative_errors": [float(value) for value in flat_errors],
            "maximum_relative_error": float(np.max(flat_errors)),
            "tolerance": tolerance,
            "batched_equivalent_within_tolerance": bool(np.all(flat_errors <= tolerance)),
            "official_scalar_authoritative": True,
            "authoritative_solver": "pyriemann.geometry.mean.mean_riemann",
            "official_all_72_pass": True,
            "official_warning_count": int(official_warning_count),
            # PASS means that the audit was complete and that every scientific
            # mean came from the official solver.  Batched equivalence is
            # reported separately and is not a scientific gate.
            "passed": True,
            "maximum_official_karcher_residual": float(max(official_residuals)),
            "maximum_batched_post_karcher_residual": float(
                np.max(batched_fit.post_residuals)
            ),
        }
    return LabelNullReplicateResult(outputs, crosscheck)


def _checkpoint_identity(
    bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    *,
    input_hash: str,
    family_tag: int,
    phase_tag: int,
    total_replicates: int,
) -> dict[str, Any]:
    return {
        "protocol_sha256": bundle.context.protocol_sha256,
        "config_sha256": bundle.context.config_sha256,
        "code_commit": bundle.context.code_commit,
        "input_hash": input_hash,
        "family_tag": int(family_tag),
        "phase_tag": int(phase_tag),
        "total_replicates": int(total_replicates),
        "n_subjects": 9,
        "bit_generator": "PCG64DXSM",
        "master_seed": int(config["protocol"]["seed"]),
        "replicate_index_base": 0,
    }


def _load_or_create_checkpoint(
    path: Path,
    identity: Mapping[str, Any],
) -> NullCheckpoint:
    if path.exists():
        return load_null_checkpoint(path, expected_metadata=identity)
    return create_null_checkpoint(
        total_replicates=int(identity["total_replicates"]),
        n_subjects=9,
        protocol_sha256=str(identity["protocol_sha256"]),
        config_sha256=str(identity["config_sha256"]),
        code_commit=str(identity["code_commit"]),
        input_hash=str(identity["input_hash"]),
        family_tag=int(identity["family_tag"]),
        phase_tag=int(identity["phase_tag"]),
        master_seed=int(identity["master_seed"]),
    )


def _checkpoint_paths(cache_dir: Path, family: str) -> dict[tuple[int, int], Path]:
    return {
        (geometry_index, object_index): cache_dir
        / f"{family}_{GEOMETRIES[geometry_index].lower()}_{OBJECTS[object_index].lower()}.npz"
        for geometry_index in range(2)
        for object_index in range(2)
    }


def run_label_null_checkpointed(
    data: ConditionalWholeData,
    bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    phase_tag: int,
    total_replicates: int,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run/replay the shared-plan R null; return group and subject arrays."""

    if not bundle.all_gates_passed:
        raise ConditionalPipelineError("scientific nulls blocked by geometry gate")
    if int(total_replicates) < 1 or int(workers) < 1:
        raise ValueError("total_replicates and workers must be positive")
    root = Path(repo_root).resolve()
    cache_dir = root / str(config["project"]["cache_dir"]) / "checkpoints" / bundle.context.phase
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels = data.metadata["class_label"].astype(str).to_numpy()
    uids = data.metadata["trial_uid"].astype(str).to_numpy()
    input_hash = _hash_inputs(
        np.asarray(
            [_checkpoint_code_snapshot_sha256(config, root, bundle)], dtype="U64"
        ),
        data.covariances,
        labels.astype("U16"),
        uids.astype("U32"),
        bundle.marginal_means,
    )
    paths = _checkpoint_paths(cache_dir, "label_destruction")
    checkpoints: dict[tuple[int, int], NullCheckpoint] = {}
    for key, path in paths.items():
        scoped_hash = _hash_inputs(
            np.asarray([input_hash, GEOMETRIES[key[0]], OBJECTS[key[1]]], dtype="U80")
        )
        identity = _checkpoint_identity(
            bundle,
            config,
            input_hash=scoped_hash,
            family_tag=FAMILY_LABEL,
            phase_tag=phase_tag,
            total_replicates=total_replicates,
        )
        checkpoints[key] = _load_or_create_checkpoint(path, identity)
    pending_sets = [set(pending_checkpoint_indices(value).tolist()) for value in checkpoints.values()]
    pending = np.asarray(sorted(set().union(*pending_sets)), dtype=np.int64)
    covariance_logs = spd_log(data.covariances)
    batch_size = int(config["nulls"]["label_destruction"]["batch_size"])
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        if workers == 1:
            results = [
                _label_null_one(int(index), data, covariance_logs, bundle, config, phase_tag)
                for index in batch
            ]
        else:
            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                results = list(
                    executor.map(
                        lambda index: _label_null_one(
                            int(index), data, covariance_logs, bundle, config, phase_tag
                        ),
                        batch.tolist(),
                    )
                )
        stacked = np.stack([value.statistics for value in results], axis=0)
        crosschecks = [value.airm_crosscheck for value in results if value.airm_crosscheck]
        if crosschecks:
            from src.conditional_provenance_v1 import atomic_write_json

            audit = {
                **bundle.context.prefix("PASS" if crosschecks[0]["passed"] else "FAIL"),
                "purpose": "batched_airm_label_null_vs_official_scalar",
                **crosschecks[0],
            }
            audit_path = (
                _resolve_output_root(root, config)
                / "protocol"
                / f"{bundle.context.phase}_label_null_airm_crosscheck.json"
            )
            atomic_write_json(audit_path, audit)
        for key in sorted(checkpoints):
            checkpoint = checkpoints[key]
            is_pending = np.isin(batch, pending_checkpoint_indices(checkpoint))
            if np.any(~is_pending):
                existing = checkpoint.subject_statistics[batch[~is_pending]]
                np.testing.assert_array_equal(existing, stacked[~is_pending, key[0], key[1]])
            if np.any(is_pending):
                checkpoint = record_checkpoint_batch(
                    checkpoint,
                    batch[is_pending],
                    stacked[is_pending, key[0], key[1]],
                )
                checkpoints[key] = checkpoint
                save_null_checkpoint(paths[key], checkpoint)
    subject_statistics = np.empty((2, 2, total_replicates, 9), dtype=np.float64)
    group_statistics = np.empty((2, 2, total_replicates), dtype=np.float64)
    for key, checkpoint in checkpoints.items():
        if len(pending_checkpoint_indices(checkpoint)):
            raise ConditionalPipelineError("label checkpoint remained incomplete")
        subject_statistics[key] = checkpoint.subject_statistics
        group_statistics[key] = checkpoint.group_statistics
    return group_statistics, subject_statistics


def _run_cached_object_null(
    *,
    bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    input_hash: str,
    family_tag: int,
    phase_tag: int,
    total_replicates: int,
    batch_size: int,
    evaluator: Any,
) -> tuple[np.ndarray, np.ndarray]:
    identity = _checkpoint_identity(
        bundle,
        config,
        input_hash=input_hash,
        family_tag=family_tag,
        phase_tag=phase_tag,
        total_replicates=total_replicates,
    )
    checkpoint = _load_or_create_checkpoint(checkpoint_path, identity)
    pending = pending_checkpoint_indices(checkpoint)
    for start in range(0, len(pending), int(batch_size)):
        indices = pending[start : start + int(batch_size)]
        result = evaluator(indices)
        if not np.array_equal(result.replicate_indices, indices):
            raise ConditionalPipelineError("null evaluator changed replicate order")
        checkpoint = record_checkpoint_batch(
            checkpoint, indices, result.subject_statistics
        )
        save_null_checkpoint(checkpoint_path, checkpoint)
    if len(pending_checkpoint_indices(checkpoint)):
        raise ConditionalPipelineError("object-null checkpoint remained incomplete")
    return checkpoint.group_statistics.copy(), checkpoint.subject_statistics.copy()


def run_semantic_nulls_checkpointed(
    bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    total_replicates: int,
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(repo_root).resolve()
    cache_dir = root / str(config["project"]["cache_dir"]) / "checkpoints" / bundle.context.phase
    cache_dir.mkdir(parents=True, exist_ok=True)
    groups = np.empty((2, 2, total_replicates), dtype=np.float64)
    subjects = np.empty((2, 2, total_replicates, 9), dtype=np.float64)
    batch_size = int(config["nulls"]["semantic_permutation"]["batch_size"])
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            objects, _shapes = bundle.object_arrays(geometry, object_name)
            vectorizer = _vectorizer(object_name)
            input_hash = _hash_inputs(
                np.asarray(
                    [
                        geometry,
                        object_name,
                        _checkpoint_code_snapshot_sha256(config, root, bundle),
                    ],
                    dtype="U64",
                ),
                objects,
                bundle.subjects,
            )
            path = cache_dir / f"semantic_permutation_{geometry.lower()}_{object_name.lower()}.npz"

            def evaluator(indices: np.ndarray, values: np.ndarray = objects, fn: Any = vectorizer):
                return semantic_discovery_null(
                    values,
                    fn,
                    replicate_indices=indices,
                    subjects=bundle.subjects,
                    master_seed=int(config["protocol"]["seed"]),
                    batch_size=batch_size,
                )

            group_values, subject_values = _run_cached_object_null(
                bundle=bundle,
                config=config,
                checkpoint_path=path,
                input_hash=input_hash,
                family_tag=FAMILY_SEMANTIC,
                phase_tag=PHASE_COMMON,
                total_replicates=total_replicates,
                batch_size=batch_size,
                evaluator=evaluator,
            )
            groups[geometry_index, object_index] = group_values
            subjects[geometry_index, object_index] = subject_values
    return groups, subjects


def validate_locked_discovery_f_templates(
    discovery_bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[tuple[int, int], np.ndarray]:
    """Recompute F LOSO templates and require equality to the locked CSV."""

    if discovery_bundle.context.phase != "discovery":
        raise ConditionalPipelineError("locked template source must be discovery")
    path = (
        _resolve_output_root(Path(repo_root).resolve(), config)
        / "tables"
        / "discovery"
        / "loso_templates.csv"
    )
    frame = pd.read_csv(path, float_precision="round_trip")
    if tuple(frame.columns) != DISCOVERY_TABLE_SCHEMAS["loso_templates.csv"]:
        raise ConditionalPipelineError("locked discovery LOSO template schema mismatch")
    result: dict[tuple[int, int], np.ndarray] = {}
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            _objects, shapes = discovery_bundle.object_arrays(geometry, object_name)
            computed = loso_templates(shapes[:, 2])
            stored = np.empty_like(computed)
            for subject_index, subject in enumerate(discovery_bundle.subjects):
                rows = frame.loc[
                    (frame["geometry"] == geometry)
                    & (frame["object"] == object_name)
                    & (frame["target_subject"].astype(int) == int(subject))
                    & (frame["template_source_split"] == "F")
                    & (frame["target_split"] == "F")
                ].sort_values("feature_index")
                if (
                    len(rows) != computed.shape[1]
                    or rows["feature_index"].astype(int).tolist()
                    != list(range(computed.shape[1]))
                ):
                    raise ConditionalPipelineError(
                        f"locked F template row grid failed: {geometry}/{object_name}/S{subject}"
                    )
                stored[subject_index] = rows["value"].to_numpy(dtype=np.float64)
            if not np.allclose(stored, computed, rtol=0.0, atol=5.0e-15):
                raise ConditionalPipelineError(
                    f"locked F template values differ: {geometry}/{object_name}"
                )
            result[(geometry_index, object_index)] = stored
    return result


def run_confirmatory_semantic_nulls_checkpointed(
    discovery_bundle: PhaseGeometryBundle,
    confirmatory_bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    total_replicates: int,
) -> tuple[np.ndarray, np.ndarray]:
    if confirmatory_bundle.context.phase != "confirmatory":
        raise ConditionalPipelineError("confirmatory semantic null requires confirmatory bundle")
    root = Path(repo_root).resolve()
    cache_dir = (
        root
        / str(config["project"]["cache_dir"])
        / "checkpoints"
        / "confirmatory"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    groups = np.empty((2, 2, total_replicates), dtype=np.float64)
    subjects = np.empty((2, 2, total_replicates, 9), dtype=np.float64)
    batch_size = int(config["nulls"]["semantic_permutation"]["batch_size"])
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            discovery_objects, _ = discovery_bundle.object_arrays(geometry, object_name)
            confirmatory_objects, _ = confirmatory_bundle.object_arrays(geometry, object_name)
            vectorizer = _vectorizer(object_name)
            input_hash = _hash_inputs(
                np.asarray(
                    [
                        geometry,
                        object_name,
                        "fixed-discovery-F",
                        _checkpoint_code_snapshot_sha256(
                            config, root, confirmatory_bundle
                        ),
                    ],
                    dtype="U64",
                ),
                discovery_objects,
                confirmatory_objects,
                confirmatory_bundle.subjects,
            )
            path = cache_dir / f"semantic_permutation_{geometry.lower()}_{object_name.lower()}.npz"

            def evaluator(
                indices: np.ndarray,
                discovery_values: np.ndarray = discovery_objects,
                confirmatory_values: np.ndarray = confirmatory_objects,
                fn: Any = vectorizer,
            ):
                return semantic_confirmatory_null(
                    discovery_values,
                    confirmatory_values,
                    fn,
                    replicate_indices=indices,
                    subjects=confirmatory_bundle.subjects,
                    master_seed=int(config["protocol"]["seed"]),
                    batch_size=batch_size,
                )

            group_values, subject_values = _run_cached_object_null(
                bundle=confirmatory_bundle,
                config=config,
                checkpoint_path=path,
                input_hash=input_hash,
                family_tag=FAMILY_SEMANTIC,
                phase_tag=PHASE_COMMON,
                total_replicates=total_replicates,
                batch_size=batch_size,
                evaluator=evaluator,
            )
            groups[geometry_index, object_index] = group_values
            subjects[geometry_index, object_index] = subject_values
    return groups, subjects


def _oracle_observed(
    bundle: PhaseGeometryBundle,
) -> tuple[np.ndarray, dict[tuple[int, int], Any]]:
    score_sets = np.empty((2, 2, 9, 24), dtype=np.float64)
    summaries: dict[tuple[int, int], Any] = {}
    permutations = all_s4_permutations()
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            objects, shapes = bundle.object_arrays(geometry, object_name)
            candidates = permuted_shape_bank(objects, _vectorizer(object_name))
            scores = discovery_oracle_score_sets(shapes, candidates)
            score_sets[geometry_index, object_index] = scores
            summaries[(geometry_index, object_index)] = summarize_oracle_scores(
                scores, permutations
            )
    return score_sets, summaries


def _load_phase_subject_effects(
    config: Mapping[str, Any], repo_root: str | Path, phase: str
) -> dict[str, np.ndarray]:
    """Load the immutable per-subject null-referenced effects for R/S/P."""

    output = _resolve_output_root(Path(repo_root).resolve(), config)
    specifications = {
        "R": "label_destruction_subject_summary.csv",
        "S": "cross_subject_shared_geometry.csv",
        "P": "oracle_permutation_subject_summary.csv",
    }
    subjects = np.asarray(config["dataset"]["subjects"], dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for stage, filename in specifications.items():
        frame = pd.read_csv(
            output / "tables" / phase / filename,
            float_precision="round_trip",
        )
        if set(frame["phase"].astype(str)) != {phase}:
            raise ConditionalPipelineError(f"subject-effect phase mismatch: {filename}")
        values = np.empty((2, 2, len(subjects)), dtype=np.float64)
        for geometry_index, geometry in enumerate(GEOMETRIES):
            for object_index, object_name in enumerate(OBJECTS):
                rows = frame.loc[
                    (frame["geometry"].astype(str) == geometry)
                    & (frame["object"].astype(str) == object_name)
                    & (frame["stage"].astype(str) == stage)
                ].copy()
                rows["subject"] = pd.to_numeric(rows["subject"], errors="raise").astype(int)
                rows = rows.set_index("subject").reindex(subjects)
                if len(rows) != 9 or rows["effect"].isna().any():
                    raise ConditionalPipelineError(
                        f"subject-effect row grid failed: {phase}/{filename}/"
                        f"{geometry}/{object_name}"
                    )
                values[geometry_index, object_index] = rows["effect"].to_numpy(
                    dtype=np.float64
                )
        if not np.isfinite(values).all():
            raise ConditionalPipelineError(f"non-finite subject effects: {phase}/{filename}")
        result[stage] = values
    return result


def _confirmatory_rs_tables(
    discovery_bundle: PhaseGeometryBundle,
    confirmatory_bundle: PhaseGeometryBundle,
    locked_templates: Mapping[tuple[int, int], np.ndarray],
    reliability: np.ndarray,
    shared: np.ndarray,
    label_groups: np.ndarray,
    label_subjects: np.ndarray,
    semantic_groups: np.ndarray,
    semantic_subjects: np.ndarray,
    discovery_effects: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    reliability_rows: list[dict[str, Any]] = []
    label_subject_rows: list[dict[str, Any]] = []
    unrelated_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    r_group_rows, _ = _null_group_rows(
        confirmatory_bundle,
        stage="R",
        observed_subjects=reliability,
        null_groups=label_groups,
    )
    s_group_rows, _ = _null_group_rows(
        confirmatory_bundle,
        stage="S",
        observed_subjects=shared,
        null_groups=semantic_groups,
    )
    confirmatory_stage_effects = {
        "R": reliability - np.median(label_subjects, axis=2),
        "S": shared - np.median(semantic_subjects, axis=2),
    }
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            prefix = confirmatory_bundle.context.prefix("PASS")
            r_observed = float(np.median(reliability[geometry_index, object_index]))
            s_observed = float(np.median(shared[geometry_index, object_index]))
            r_percentiles = subject_null_percentiles(
                reliability[geometry_index, object_index],
                label_subjects[geometry_index, object_index],
            )
            s_percentiles = subject_null_percentiles(
                shared[geometry_index, object_index],
                semantic_subjects[geometry_index, object_index],
            )
            _objects, confirmatory_shapes = confirmatory_bundle.object_arrays(
                geometry, object_name
            )
            unrelated = unrelated_derangement_statistics(
                confirmatory_shapes[:, 0], confirmatory_shapes[:, 1]
            )
            unrelated_rows.append(
                {
                    **prefix,
                    "geometry": geometry,
                    "object": object_name,
                    "same_subject_median": r_observed,
                    "unrelated_median": float(np.median(unrelated)),
                    "unrelated_min": float(np.min(unrelated)),
                    "unrelated_max": float(np.max(unrelated)),
                    "derangement_count": len(unrelated),
                }
            )
            fixed_template = locked_templates[(geometry_index, object_index)]
            for subject_index, subject in enumerate(confirmatory_bundle.subjects):
                reliability_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "object": object_name,
                        "stage": "R",
                        "subject": int(subject),
                        "observed": reliability[geometry_index, object_index, subject_index],
                        "group_observed": r_observed,
                    }
                )
                null_r = label_subjects[geometry_index, object_index, :, subject_index]
                label_subject_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "object": object_name,
                        "stage": "R",
                        "subject": int(subject),
                        "observed": reliability[geometry_index, object_index, subject_index],
                        "null_median": float(np.median(null_r)),
                        "effect": float(
                            reliability[geometry_index, object_index, subject_index]
                            - np.median(null_r)
                        ),
                        "null_percentile": r_percentiles[subject_index],
                        "replicates": label_subjects.shape[2],
                    }
                )
                null_s = semantic_subjects[geometry_index, object_index, :, subject_index]
                shared_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "object": object_name,
                        "stage": "S",
                        "subject": int(subject),
                        "observed": shared[geometry_index, object_index, subject_index],
                        "group_observed": s_observed,
                        "null_median": float(np.median(null_s)),
                        "effect": float(
                            shared[geometry_index, object_index, subject_index]
                            - np.median(null_s)
                        ),
                        "null_percentile": s_percentiles[subject_index],
                        "replicates": semantic_subjects.shape[2],
                    }
                )
                for target_split in SPLITS:
                    for feature_index, value in enumerate(fixed_template[subject_index]):
                        template_rows.append(
                            {
                                **prefix,
                                "geometry": geometry,
                                "object": object_name,
                                "target_subject": int(subject),
                                "template_source_split": "F",
                                "target_split": target_split,
                                "feature_index": feature_index,
                                "value": float(value),
                            }
                        )
            for stage, values in (
                ("R", reliability[geometry_index, object_index]),
                ("S", shared[geometry_index, object_index]),
            ):
                bootstrap = subject_bootstrap_median(
                    values,
                    replicates=int(config["statistics"]["bootstrap"]["replicates"]),
                    master_seed=int(config["protocol"]["seed"]),
                    phase_tag=1,
                )
                bootstrap_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "object": object_name,
                        "stage": stage,
                        "observed_median": float(np.median(values)),
                        "ci_low": bootstrap.ci_low,
                        "ci_high": bootstrap.ci_high,
                        "replicates": bootstrap.replicates,
                        **_paired_bootstrap_fields(
                            confirmatory_stage_effects[stage][
                                geometry_index, object_index
                            ],
                            config,
                            discovery=discovery_effects[stage][
                                geometry_index, object_index
                            ],
                            le_scores_for_airm_row=(
                                confirmatory_stage_effects[stage][1, object_index]
                                if geometry_index == 0
                                else None
                            ),
                        ),
                    }
                )
                for subject_index, (subject, value) in enumerate(
                    zip(
                        confirmatory_bundle.subjects,
                        leave_one_subject_out_influence(values),
                        strict=True,
                    )
                ):
                    influence_rows.append(
                        {
                            **prefix,
                            "geometry": geometry,
                            "object": object_name,
                            "stage": stage,
                            "subject": int(subject),
                            "influence": float(value),
                            "subject_score": float(values[subject_index]),
                            "subject_effect": float(
                                confirmatory_stage_effects[stage][
                                    geometry_index, object_index, subject_index
                                ]
                            ),
                            "discovery_subject_effect": float(
                                discovery_effects[stage][
                                    geometry_index, object_index, subject_index
                                ]
                            ),
                            "confirmatory_subject_effect": float(
                                confirmatory_stage_effects[stage][
                                    geometry_index, object_index, subject_index
                                ]
                            ),
                            "discovery_confirmatory_effect_delta": float(
                                confirmatory_stage_effects[stage][
                                    geometry_index, object_index, subject_index
                                ]
                                - discovery_effects[stage][
                                    geometry_index, object_index, subject_index
                                ]
                            ),
                            "airm_minus_le_subject_effect_delta": (
                                float(
                                    confirmatory_stage_effects[stage][
                                        geometry_index, object_index, subject_index
                                    ]
                                    - confirmatory_stage_effects[stage][
                                        1, object_index, subject_index
                                    ]
                                )
                                if geometry_index == 0
                                else np.nan
                            ),
                        }
                    )
    return {
        "within_subject_reliability.csv": _ordered_frame(
            reliability_rows, WITHIN_RELIABILITY_COLUMNS
        ),
        "label_destruction_subject_summary.csv": _ordered_frame(
            label_subject_rows, LABEL_SUBJECT_COLUMNS
        ),
        "label_destruction_group_summary.csv": _ordered_frame(
            r_group_rows, GROUP_SUMMARY_COLUMNS
        ),
        "unrelated_subject_derangement_summary.csv": pd.DataFrame(unrelated_rows),
        "loso_templates.csv": pd.DataFrame(template_rows),
        "cross_subject_shared_geometry.csv": _ordered_frame(
            shared_rows, CROSS_SHARED_COLUMNS
        ),
        "semantic_permutation_null_summary.csv": _ordered_frame(
            s_group_rows, GROUP_SUMMARY_COLUMNS
        ),
        "subject_bootstrap_summary.csv": _ordered_frame(
            bootstrap_rows, DISCOVERY_TABLE_SCHEMAS["subject_bootstrap_summary.csv"]
        ),
        "leave_one_subject_out_influence.csv": _ordered_frame(
            influence_rows,
            DISCOVERY_TABLE_SCHEMAS["leave_one_subject_out_influence.csv"],
        ),
    }


def run_confirmatory_rs_producer(
    data: ConditionalWholeData,
    discovery_bundle: PhaseGeometryBundle,
    confirmatory_bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    workers: int = 1,
    label_replicates: int | None = None,
    semantic_replicates: int | None = None,
) -> dict[str, Any]:
    """Produce confirmatory R/S snapshots using the locked discovery-F source."""

    validate_label_null_dry_run(config, repo_root, discovery_bundle)
    locked_templates = validate_locked_discovery_f_templates(
        discovery_bundle, config, repo_root
    )
    label_count = int(
        label_replicates or config["nulls"]["label_destruction"]["replicates"]
    )
    semantic_count = int(
        semantic_replicates or config["nulls"]["semantic_permutation"]["replicates"]
    )
    reliability = np.empty((2, 2, 9), dtype=np.float64)
    shared = np.empty_like(reliability)
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            _discovery_objects, discovery_shapes = discovery_bundle.object_arrays(
                geometry, object_name
            )
            _confirmatory_objects, confirmatory_shapes = confirmatory_bundle.object_arrays(
                geometry, object_name
            )
            reliability[geometry_index, object_index] = reliability_subject_scores(
                confirmatory_shapes
            )
            shared[geometry_index, object_index] = confirmatory_shared_subject_scores(
                discovery_shapes, confirmatory_shapes
            )
    label_groups, label_subjects = run_label_null_checkpointed(
        data,
        confirmatory_bundle,
        config,
        repo_root,
        phase_tag=1,
        total_replicates=label_count,
        workers=workers,
    )
    semantic_groups, semantic_subjects = run_confirmatory_semantic_nulls_checkpointed(
        discovery_bundle,
        confirmatory_bundle,
        config,
        repo_root,
        total_replicates=semantic_count,
    )
    tables = _confirmatory_rs_tables(
        discovery_bundle,
        confirmatory_bundle,
        locked_templates,
        reliability,
        shared,
        label_groups,
        label_subjects,
        semantic_groups,
        semantic_subjects,
        _load_phase_subject_effects(config, repo_root, "discovery"),
        config,
    )
    output = _resolve_output_root(Path(repo_root).resolve(), config)
    table_dir = output / "tables" / "confirmatory"
    null_dir = output / "nulls" / "confirmatory"
    for filename, frame in tables.items():
        _atomic_write_csv(table_dir / filename, frame)
    _save_null_archive(
        null_dir / "label_destruction_group_statistics.npz", label_groups
    )
    _save_null_archive(
        null_dir / "semantic_permutation_group_statistics.npz", semantic_groups
    )
    return {
        "phase": "confirmatory",
        "label_replicates": label_count,
        "semantic_replicates": semantic_count,
        "table_count": len(tables),
        "null_file_count": 2,
    }


def run_oracle_nulls_checkpointed(
    bundle: PhaseGeometryBundle,
    score_sets: np.ndarray,
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    total_replicates: int,
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(repo_root).resolve()
    cache_dir = root / str(config["project"]["cache_dir"]) / "checkpoints" / bundle.context.phase
    cache_dir.mkdir(parents=True, exist_ok=True)
    groups = np.empty((2, 2, total_replicates), dtype=np.float64)
    subjects = np.empty((2, 2, total_replicates, 9), dtype=np.float64)
    batch_size = int(config["nulls"]["oracle_rank"]["batch_size"])
    tolerance = float(config["geometry"]["permutation_rank_tolerance"])
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            scores = score_sets[geometry_index, object_index]
            input_hash = _hash_inputs(
                np.asarray(
                    [
                        geometry,
                        object_name,
                        _checkpoint_code_snapshot_sha256(config, root, bundle),
                    ],
                    dtype="U64",
                ),
                scores,
                bundle.subjects,
            )
            path = cache_dir / f"oracle_rank_{geometry.lower()}_{object_name.lower()}.npz"

            def evaluator(indices: np.ndarray, values: np.ndarray = scores):
                return oracle_rank_null(
                    values,
                    replicate_indices=indices,
                    subjects=bundle.subjects,
                    master_seed=int(config["protocol"]["seed"]),
                    tolerance=tolerance,
                    batch_size=batch_size,
                )

            group_values, subject_values = _run_cached_object_null(
                bundle=bundle,
                config=config,
                checkpoint_path=path,
                input_hash=input_hash,
                family_tag=FAMILY_ORACLE,
                phase_tag=PHASE_COMMON,
                total_replicates=total_replicates,
                batch_size=batch_size,
                evaluator=evaluator,
            )
            groups[geometry_index, object_index] = group_values
            subjects[geometry_index, object_index] = subject_values
    return groups, subjects


def _observed_stage_scores(bundle: PhaseGeometryBundle) -> tuple[np.ndarray, np.ndarray]:
    reliability = np.empty((2, 2, 9), dtype=np.float64)
    shared = np.empty_like(reliability)
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            _objects, shapes = bundle.object_arrays(geometry, object_name)
            reliability[geometry_index, object_index] = reliability_subject_scores(shapes)
            shared[geometry_index, object_index] = discovery_shared_subject_scores(shapes)
    return reliability, shared


def _null_group_rows(
    bundle: PhaseGeometryBundle,
    *,
    stage: str,
    observed_subjects: np.ndarray,
    null_groups: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[tuple[int, int], Any] = {}
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            observed = float(np.median(observed_subjects[geometry_index, object_index]))
            summary = plus_one_null_summary(observed, null_groups[geometry_index, object_index])
            summaries[(geometry_index, object_index)] = summary
            rows.append(
                {
                    **bundle.context.prefix("PASS"),
                    "geometry": geometry,
                    "object": object_name,
                    "stage": stage,
                    "observed": summary.observed,
                    "null_median": summary.null_median,
                    "effect": summary.effect,
                    "p_value": summary.p_value,
                    "exceedances": summary.exceedances,
                    "replicates": summary.replicates,
                    "gate_pass": bundle.all_gates_passed,
                }
            )
    return rows, summaries


def _save_null_archive(
    path: Path,
    group_statistics: np.ndarray,
) -> None:
    if group_statistics.ndim != 3 or group_statistics.shape[:2] != (2, 2):
        raise ConditionalPipelineError("null group array must have shape (2,2,B)")
    _atomic_savez(
        path,
        replicate_indices=np.arange(group_statistics.shape[-1], dtype=np.int64),
        geometries=np.asarray(GEOMETRIES, dtype=str),
        objects=np.asarray(OBJECTS, dtype=str),
        group_statistics=np.asarray(group_statistics, dtype=np.float64),
    )


_PAIRED_BOOTSTRAP_NAN = {
    "discovery_confirmatory_effect_delta_median": np.nan,
    "discovery_confirmatory_effect_delta_ci_low": np.nan,
    "discovery_confirmatory_effect_delta_ci_high": np.nan,
    "airm_minus_le_effect_delta_median": np.nan,
    "airm_minus_le_effect_delta_ci_low": np.nan,
    "airm_minus_le_effect_delta_ci_high": np.nan,
}


def _paired_bootstrap_fields(
    current: np.ndarray,
    config: Mapping[str, Any],
    *,
    discovery: np.ndarray | None = None,
    le_scores_for_airm_row: np.ndarray | None = None,
) -> dict[str, float]:
    """Return descriptive paired CIs without changing any decision statistic.

    Discovery/confirmatory deltas mean confirmatory minus discovery.  Geometry
    deltas mean AIRM minus LE and are populated only on the AIRM table row.
    """

    result = dict(_PAIRED_BOOTSTRAP_NAN)
    replicates = int(config["statistics"]["bootstrap"]["replicates"])
    seed = int(config["protocol"]["seed"])
    current = np.asarray(current, dtype=np.float64)
    if discovery is not None:
        reference = np.asarray(discovery, dtype=np.float64)
        summary = subject_bootstrap_paired_median_delta(
            current,
            reference,
            replicates=replicates,
            master_seed=seed,
            phase_tag=PHASE_COMMON,
        )
        result.update(
            {
                "discovery_confirmatory_effect_delta_median": float(
                    np.median(current) - np.median(reference)
                ),
                "discovery_confirmatory_effect_delta_ci_low": summary.ci_low,
                "discovery_confirmatory_effect_delta_ci_high": summary.ci_high,
            }
        )
    if le_scores_for_airm_row is not None:
        reference = np.asarray(le_scores_for_airm_row, dtype=np.float64)
        summary = subject_bootstrap_paired_median_delta(
            current,
            reference,
            replicates=replicates,
            master_seed=seed,
            phase_tag=PHASE_COMMON,
        )
        result.update(
            {
                "airm_minus_le_effect_delta_median": float(
                    np.median(current) - np.median(reference)
                ),
                "airm_minus_le_effect_delta_ci_low": summary.ci_low,
                "airm_minus_le_effect_delta_ci_high": summary.ci_high,
            }
        )
    return result


def _stage_tables(
    bundle: PhaseGeometryBundle,
    reliability: np.ndarray,
    shared: np.ndarray,
    label_groups: np.ndarray,
    label_subjects: np.ndarray,
    semantic_groups: np.ndarray,
    semantic_subjects: np.ndarray,
    score_sets: np.ndarray,
    oracle_summaries: Mapping[tuple[int, int], Any],
    oracle_groups: np.ndarray,
    oracle_subjects: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    permutations = all_s4_permutations()
    reliability_rows: list[dict[str, Any]] = []
    label_subject_rows: list[dict[str, Any]] = []
    unrelated_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    oracle_score_rows: list[dict[str, Any]] = []
    oracle_subject_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []

    r_group_rows, r_summaries = _null_group_rows(
        bundle, stage="R", observed_subjects=reliability, null_groups=label_groups
    )
    s_group_rows, s_summaries = _null_group_rows(
        bundle, stage="S", observed_subjects=shared, null_groups=semantic_groups
    )
    normalized_ranks = np.empty((2, 2, 9), dtype=np.float64)
    for key, summary in oracle_summaries.items():
        normalized_ranks[key] = summary.normalized_ranks
    p_group_rows, p_summaries = _null_group_rows(
        bundle, stage="P", observed_subjects=normalized_ranks, null_groups=oracle_groups
    )
    stage_effects = {
        "R": reliability - np.median(label_subjects, axis=2),
        "S": shared - np.median(semantic_subjects, axis=2),
        "P": normalized_ranks - np.median(oracle_subjects, axis=2),
    }

    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            prefix = bundle.context.prefix("PASS")
            r_observed = float(np.median(reliability[geometry_index, object_index]))
            s_observed = float(np.median(shared[geometry_index, object_index]))
            r_percentiles = subject_null_percentiles(
                reliability[geometry_index, object_index],
                label_subjects[geometry_index, object_index],
            )
            s_percentiles = subject_null_percentiles(
                shared[geometry_index, object_index],
                semantic_subjects[geometry_index, object_index],
            )
            p_summary = oracle_summaries[(geometry_index, object_index)]
            p_percentiles = subject_null_percentiles(
                p_summary.normalized_ranks,
                oracle_subjects[geometry_index, object_index],
            )
            objects, shapes = bundle.object_arrays(geometry, object_name)
            templates_a = loso_templates(shapes[:, 0])
            templates_b = loso_templates(shapes[:, 1])
            templates_f = loso_templates(shapes[:, 2])
            unrelated = unrelated_derangement_statistics(shapes[:, 0], shapes[:, 1])
            unrelated_rows.append(
                {
                    **prefix,
                    "geometry": geometry,
                    "object": object_name,
                    "same_subject_median": r_observed,
                    "unrelated_median": float(np.median(unrelated)),
                    "unrelated_min": float(np.min(unrelated)),
                    "unrelated_max": float(np.max(unrelated)),
                    "derangement_count": len(unrelated),
                }
            )
            for subject_index, subject in enumerate(bundle.subjects):
                reliability_rows.append(
                    {
                        **prefix, "geometry": geometry, "object": object_name, "stage": "R",
                        "subject": int(subject),
                        "observed": reliability[geometry_index, object_index, subject_index],
                        "group_observed": r_observed,
                    }
                )
                null_r = label_subjects[geometry_index, object_index, :, subject_index]
                label_subject_rows.append(
                    {
                        **prefix, "geometry": geometry, "object": object_name, "stage": "R",
                        "subject": int(subject),
                        "observed": reliability[geometry_index, object_index, subject_index],
                        "null_median": float(np.median(null_r)),
                        "effect": float(reliability[geometry_index, object_index, subject_index] - np.median(null_r)),
                        "null_percentile": r_percentiles[subject_index],
                        "replicates": label_subjects.shape[2],
                    }
                )
                shared_rows.append(
                    {
                        **prefix, "geometry": geometry, "object": object_name, "stage": "S",
                        "subject": int(subject),
                        "observed": shared[geometry_index, object_index, subject_index],
                        "group_observed": s_observed,
                        "null_median": float(
                            np.median(
                                semantic_subjects[
                                    geometry_index, object_index, :, subject_index
                                ]
                            )
                        ),
                        "effect": float(
                            shared[geometry_index, object_index, subject_index]
                            - np.median(
                                semantic_subjects[
                                    geometry_index, object_index, :, subject_index
                                ]
                            )
                        ),
                        "null_percentile": s_percentiles[subject_index],
                        "replicates": semantic_subjects.shape[2],
                    }
                )
                for template_split, target_split, template in (
                    ("A", "B", templates_a),
                    ("B", "A", templates_b),
                    ("F", "F", templates_f),
                ):
                    for feature_index, value in enumerate(template[subject_index]):
                        template_rows.append(
                            {
                                **prefix,
                                "geometry": geometry,
                                "object": object_name,
                                "target_subject": int(subject),
                                "template_source_split": template_split,
                                "target_split": target_split,
                                "feature_index": feature_index,
                                "value": float(value),
                            }
                        )
                for permutation_index, permutation in enumerate(permutations):
                    oracle_score_rows.append(
                        {
                            **prefix, "geometry": geometry, "object": object_name, "stage": "P",
                            "subject": int(subject), "permutation_index": permutation_index,
                            "permutation": "-".join(str(int(v)) for v in permutation),
                            "is_identity": permutation_index == 0,
                            "score": score_sets[geometry_index, object_index, subject_index, permutation_index],
                        }
                    )
                null_p = oracle_subjects[geometry_index, object_index, :, subject_index]
                best = int(p_summary.best_indices[subject_index])
                second = int(p_summary.second_best_indices[subject_index])
                oracle_subject_rows.append(
                    {
                        **prefix, "geometry": geometry, "object": object_name, "stage": "P",
                        "subject": int(subject), "identity_score": p_summary.identity_scores[subject_index],
                        "true_rank": int(p_summary.identity_ranks[subject_index]),
                        "normalized_rank": p_summary.normalized_ranks[subject_index],
                        "top1_exact": bool(p_summary.top1_exact[subject_index]),
                        "best_permutation_index": best,
                        "best_permutation": "-".join(str(int(v)) for v in permutations[best]),
                        "second_best_permutation_index": second,
                        "second_best_permutation": "-".join(str(int(v)) for v in permutations[second]),
                        "margin": p_summary.margins[subject_index],
                        "null_median": float(np.median(null_p)),
                        "effect": float(p_summary.normalized_ranks[subject_index] - np.median(null_p)),
                        "null_percentile": p_percentiles[subject_index],
                    }
                )
            for stage, values in (
                ("R", reliability[geometry_index, object_index]),
                ("S", shared[geometry_index, object_index]),
                ("P", p_summary.normalized_ranks),
            ):
                bootstrap = subject_bootstrap_median(
                    values,
                    replicates=int(config["statistics"]["bootstrap"]["replicates"]),
                    master_seed=int(config["protocol"]["seed"]),
                    phase_tag=PHASE_DISCOVERY,
                )
                bootstrap_rows.append(
                    {
                        **prefix, "geometry": geometry, "object": object_name, "stage": stage,
                        "observed_median": float(np.median(values)), "ci_low": bootstrap.ci_low,
                        "ci_high": bootstrap.ci_high, "replicates": bootstrap.replicates,
                        **_paired_bootstrap_fields(
                            stage_effects[stage][geometry_index, object_index],
                            config,
                            le_scores_for_airm_row=(
                                stage_effects[stage][1, object_index]
                                if geometry_index == 0
                                else None
                            ),
                        ),
                    }
                )
                for subject_index, (subject, value) in enumerate(
                    zip(
                        bundle.subjects,
                        leave_one_subject_out_influence(values),
                        strict=True,
                    )
                ):
                    influence_rows.append(
                        {
                            **prefix, "geometry": geometry, "object": object_name, "stage": stage,
                            "subject": int(subject), "influence": float(value),
                            "subject_score": float(values[subject_index]),
                            "subject_effect": float(
                                stage_effects[stage][
                                    geometry_index, object_index, subject_index
                                ]
                            ),
                            "discovery_subject_effect": float(
                                stage_effects[stage][
                                    geometry_index, object_index, subject_index
                                ]
                            ),
                            "confirmatory_subject_effect": np.nan,
                            "discovery_confirmatory_effect_delta": np.nan,
                            "airm_minus_le_subject_effect_delta": (
                                float(
                                    stage_effects[stage][
                                        geometry_index, object_index, subject_index
                                    ]
                                    - stage_effects[stage][
                                        1, object_index, subject_index
                                    ]
                                )
                                if geometry_index == 0
                                else np.nan
                            ),
                        }
                    )

    return {
        "within_subject_reliability.csv": _ordered_frame(reliability_rows, WITHIN_RELIABILITY_COLUMNS),
        "label_destruction_subject_summary.csv": _ordered_frame(label_subject_rows, LABEL_SUBJECT_COLUMNS),
        "label_destruction_group_summary.csv": _ordered_frame(r_group_rows, GROUP_SUMMARY_COLUMNS),
        "unrelated_subject_derangement_summary.csv": pd.DataFrame(unrelated_rows),
        "loso_templates.csv": pd.DataFrame(template_rows),
        "cross_subject_shared_geometry.csv": _ordered_frame(shared_rows, CROSS_SHARED_COLUMNS),
        "semantic_permutation_null_summary.csv": _ordered_frame(s_group_rows, GROUP_SUMMARY_COLUMNS),
        "oracle_permutation_all_24_scores.csv": _ordered_frame(oracle_score_rows, ORACLE_SCORE_COLUMNS),
        "oracle_permutation_subject_summary.csv": _ordered_frame(oracle_subject_rows, ORACLE_SUBJECT_COLUMNS),
        "oracle_permutation_group_summary.csv": _ordered_frame(p_group_rows, GROUP_SUMMARY_COLUMNS),
        "subject_bootstrap_summary.csv": _ordered_frame(
            bootstrap_rows, DISCOVERY_TABLE_SCHEMAS["subject_bootstrap_summary.csv"]
        ),
        "leave_one_subject_out_influence.csv": _ordered_frame(
            influence_rows,
            DISCOVERY_TABLE_SCHEMAS["leave_one_subject_out_influence.csv"],
        ),
    }


def run_discovery_null_producer(
    data: ConditionalWholeData,
    bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    workers: int = 1,
    label_replicates: int | None = None,
    semantic_replicates: int | None = None,
    oracle_replicates: int | None = None,
) -> DiscoveryNullResult:
    """Run all discovery R/S/P nulls and emit the exact snapshot basenames."""

    if bundle.context.phase != "discovery":
        raise ConditionalPipelineError("discovery null producer requires discovery bundle")
    validate_label_null_dry_run(config, repo_root, bundle)
    label_count = int(label_replicates or config["nulls"]["label_destruction"]["replicates"])
    semantic_count = int(semantic_replicates or config["nulls"]["semantic_permutation"]["replicates"])
    oracle_count = int(oracle_replicates or config["nulls"]["oracle_rank"]["replicates"])
    reliability, shared = _observed_stage_scores(bundle)
    label_groups, label_subjects = run_label_null_checkpointed(
        data,
        bundle,
        config,
        repo_root,
        phase_tag=PHASE_DISCOVERY,
        total_replicates=label_count,
        workers=workers,
    )
    semantic_groups, semantic_subjects = run_semantic_nulls_checkpointed(
        bundle, config, repo_root, total_replicates=semantic_count
    )
    score_sets, oracle_summaries = _oracle_observed(bundle)
    oracle_groups, oracle_subjects = run_oracle_nulls_checkpointed(
        bundle, score_sets, config, repo_root, total_replicates=oracle_count
    )
    tables = _stage_tables(
        bundle,
        reliability,
        shared,
        label_groups,
        label_subjects,
        semantic_groups,
        semantic_subjects,
        score_sets,
        oracle_summaries,
        oracle_groups,
        oracle_subjects,
        config,
    )
    root = Path(repo_root).resolve()
    output_root = _resolve_output_root(root, config)
    table_dir = output_root / "tables" / "discovery"
    null_dir = output_root / "nulls" / "discovery"
    for filename, frame in tables.items():
        _atomic_write_csv(table_dir / filename, frame)
    _save_null_archive(null_dir / "label_destruction_group_statistics.npz", label_groups)
    _save_null_archive(null_dir / "semantic_permutation_group_statistics.npz", semantic_groups)
    _save_null_archive(null_dir / "oracle_rank_null.npz", oracle_groups)
    group_summary = pd.concat(
        [
            tables["label_destruction_group_summary.csv"],
            tables["semantic_permutation_null_summary.csv"],
            tables["oracle_permutation_group_summary.csv"],
        ],
        ignore_index=True,
    )
    return DiscoveryNullResult(
        label_group_statistics=label_groups,
        semantic_group_statistics=semantic_groups,
        oracle_group_statistics=oracle_groups,
        group_summary=group_summary,
    )


def _confirmatory_oracle_observed(
    discovery_bundle: PhaseGeometryBundle,
    confirmatory_bundle: PhaseGeometryBundle,
) -> tuple[np.ndarray, dict[tuple[int, int], Any]]:
    score_sets = np.empty((2, 2, 9, 24), dtype=np.float64)
    summaries: dict[tuple[int, int], Any] = {}
    permutations = all_s4_permutations()
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            _discovery_objects, discovery_shapes = discovery_bundle.object_arrays(
                geometry, object_name
            )
            confirmatory_objects, _ = confirmatory_bundle.object_arrays(
                geometry, object_name
            )
            candidates = permuted_shape_bank(
                confirmatory_objects, _vectorizer(object_name)
            )
            scores = confirmatory_oracle_score_sets(discovery_shapes, candidates)
            score_sets[geometry_index, object_index] = scores
            summaries[(geometry_index, object_index)] = summarize_oracle_scores(
                scores, permutations
            )
    return score_sets, summaries


def _confirmatory_oracle_tables(
    discovery_bundle: PhaseGeometryBundle,
    bundle: PhaseGeometryBundle,
    score_sets: np.ndarray,
    summaries: Mapping[tuple[int, int], Any],
    oracle_groups: np.ndarray,
    oracle_subjects: np.ndarray,
    discovery_effects: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    permutations = all_s4_permutations()
    score_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    normalized = np.empty((2, 2, 9), dtype=np.float64)
    for key, summary in summaries.items():
        normalized[key] = summary.normalized_ranks
    group_rows, _ = _null_group_rows(
        bundle, stage="P", observed_subjects=normalized, null_groups=oracle_groups
    )
    confirmatory_effects = normalized - np.median(oracle_subjects, axis=2)
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for object_index, object_name in enumerate(OBJECTS):
            prefix = bundle.context.prefix("PASS")
            summary = summaries[(geometry_index, object_index)]
            percentiles = subject_null_percentiles(
                summary.normalized_ranks,
                oracle_subjects[geometry_index, object_index],
            )
            for subject_index, subject in enumerate(bundle.subjects):
                for permutation_index, permutation in enumerate(permutations):
                    score_rows.append(
                        {
                            **prefix,
                            "geometry": geometry,
                            "object": object_name,
                            "stage": "P",
                            "subject": int(subject),
                            "permutation_index": permutation_index,
                            "permutation": "-".join(str(int(v)) for v in permutation),
                            "is_identity": permutation_index == 0,
                            "score": score_sets[
                                geometry_index,
                                object_index,
                                subject_index,
                                permutation_index,
                            ],
                        }
                    )
                null = oracle_subjects[
                    geometry_index, object_index, :, subject_index
                ]
                best = int(summary.best_indices[subject_index])
                second = int(summary.second_best_indices[subject_index])
                subject_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "object": object_name,
                        "stage": "P",
                        "subject": int(subject),
                        "identity_score": summary.identity_scores[subject_index],
                        "true_rank": int(summary.identity_ranks[subject_index]),
                        "normalized_rank": summary.normalized_ranks[subject_index],
                        "top1_exact": bool(summary.top1_exact[subject_index]),
                        "best_permutation_index": best,
                        "best_permutation": "-".join(
                            str(int(v)) for v in permutations[best]
                        ),
                        "second_best_permutation_index": second,
                        "second_best_permutation": "-".join(
                            str(int(v)) for v in permutations[second]
                        ),
                        "margin": summary.margins[subject_index],
                        "null_median": float(np.median(null)),
                        "effect": float(
                            summary.normalized_ranks[subject_index] - np.median(null)
                        ),
                        "null_percentile": percentiles[subject_index],
                    }
                )
            values = summary.normalized_ranks
            bootstrap = subject_bootstrap_median(
                values,
                replicates=int(config["statistics"]["bootstrap"]["replicates"]),
                master_seed=int(config["protocol"]["seed"]),
                phase_tag=1,
            )
            bootstrap_rows.append(
                {
                    **prefix,
                    "geometry": geometry,
                    "object": object_name,
                    "stage": "P",
                    "observed_median": float(np.median(values)),
                    "ci_low": bootstrap.ci_low,
                    "ci_high": bootstrap.ci_high,
                    "replicates": bootstrap.replicates,
                    **_paired_bootstrap_fields(
                        confirmatory_effects[geometry_index, object_index],
                        config,
                        discovery=discovery_effects["P"][
                            geometry_index, object_index
                        ],
                        le_scores_for_airm_row=(
                            confirmatory_effects[1, object_index]
                            if geometry_index == 0
                            else None
                        ),
                    ),
                }
            )
            for subject_index, (subject, value) in enumerate(
                zip(
                    bundle.subjects,
                    leave_one_subject_out_influence(values),
                    strict=True,
                )
            ):
                influence_rows.append(
                    {
                        **prefix,
                        "geometry": geometry,
                        "object": object_name,
                        "stage": "P",
                        "subject": int(subject),
                        "influence": float(value),
                        "subject_score": float(values[subject_index]),
                        "subject_effect": float(
                            confirmatory_effects[
                                geometry_index, object_index, subject_index
                            ]
                        ),
                        "discovery_subject_effect": float(
                            discovery_effects["P"][
                                geometry_index, object_index, subject_index
                            ]
                        ),
                        "confirmatory_subject_effect": float(
                            confirmatory_effects[
                                geometry_index, object_index, subject_index
                            ]
                        ),
                        "discovery_confirmatory_effect_delta": float(
                            confirmatory_effects[
                                geometry_index, object_index, subject_index
                            ]
                            - discovery_effects["P"][
                                geometry_index, object_index, subject_index
                            ]
                        ),
                        "airm_minus_le_subject_effect_delta": (
                            float(
                                confirmatory_effects[
                                    geometry_index, object_index, subject_index
                                ]
                                - confirmatory_effects[
                                    1, object_index, subject_index
                                ]
                            )
                            if geometry_index == 0
                            else np.nan
                        ),
                    }
                )
    return {
        "oracle_permutation_all_24_scores.csv": _ordered_frame(
            score_rows, ORACLE_SCORE_COLUMNS
        ),
        "oracle_permutation_subject_summary.csv": _ordered_frame(
            subject_rows, ORACLE_SUBJECT_COLUMNS
        ),
        "oracle_permutation_group_summary.csv": _ordered_frame(
            group_rows, GROUP_SUMMARY_COLUMNS
        ),
        "subject_bootstrap_summary.csv": _ordered_frame(
            bootstrap_rows, DISCOVERY_TABLE_SCHEMAS["subject_bootstrap_summary.csv"]
        ),
        "leave_one_subject_out_influence.csv": _ordered_frame(
            influence_rows,
            DISCOVERY_TABLE_SCHEMAS["leave_one_subject_out_influence.csv"],
        ),
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with source.open("rb") as reader, tempfile.NamedTemporaryFile(
            mode="w+b", dir=destination.parent, delete=False
        ) as writer:
            temporary = Path(writer.name)
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_stage_group_tables(output: Path, phase: str) -> pd.DataFrame:
    directory = output / "tables" / phase
    result = pd.concat(
        [
            pd.read_csv(
                directory / "label_destruction_group_summary.csv",
                float_precision="round_trip",
            ),
            pd.read_csv(
                directory / "semantic_permutation_null_summary.csv",
                float_precision="round_trip",
            ),
            pd.read_csv(
                directory / "oracle_permutation_group_summary.csv",
                float_precision="round_trip",
            ),
        ],
        ignore_index=True,
    )
    if tuple(result.columns) != GROUP_SUMMARY_COLUMNS or len(result) != 12:
        raise ConditionalPipelineError(f"{phase} R/S/P group summary grid failed")
    return result


def _combined_context(
    config: Mapping[str, Any], config_sha256: str, code_commit: str
) -> ProducerContext:
    return _context(
        config,
        config_sha256=config_sha256,
        code_commit=code_commit,
        phase="combined",
        session=(
            f"{config['dataset']['discovery_session']}+"
            f"{config['dataset']['confirmatory_session']}"
        ),
    )


def finalize_confirmatory_decision_artifacts(
    discovery_bundle: PhaseGeometryBundle,
    confirmatory_bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    config_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    """Validate both snapshots, copy final roots, and apply frozen chains."""

    root = Path(repo_root).resolve()
    output = _resolve_output_root(root, config)
    validate_discovery_snapshot_contract(
        config, root, config_sha256=config_sha256, code_commit=code_commit
    )
    validate_confirmatory_snapshot_contract(
        config, root, config_sha256=config_sha256, code_commit=code_commit
    )
    discovery_groups = _read_stage_group_tables(output, "discovery")
    confirmatory_groups = _read_stage_group_tables(output, "confirmatory")
    combined = _combined_context(config, config_sha256, code_commit)

    comparison_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    chain_pass: dict[tuple[str, str], bool] = {}
    for geometry in GEOMETRIES:
        for object_name in OBJECTS:
            evidence: dict[str, StageEvidence] = {}
            for stage in ("R", "S", "P"):
                discovery_row = discovery_groups.loc[
                    (discovery_groups["geometry"] == geometry)
                    & (discovery_groups["object"] == object_name)
                    & (discovery_groups["stage"] == stage)
                ]
                confirmatory_row = confirmatory_groups.loc[
                    (confirmatory_groups["geometry"] == geometry)
                    & (confirmatory_groups["object"] == object_name)
                    & (confirmatory_groups["stage"] == stage)
                ]
                if len(discovery_row) != 1 or len(confirmatory_row) != 1:
                    raise ConditionalPipelineError("combined stage group grid is not unique")
                drow, crow = discovery_row.iloc[0], confirmatory_row.iloc[0]
                comparison_rows.append(
                    {
                        **combined.prefix("PASS"),
                        "geometry": geometry,
                        "object": object_name,
                        "stage": stage,
                        "discovery_observed": float(drow["observed"]),
                        "discovery_effect": float(drow["effect"]),
                        "confirmatory_observed": float(crow["observed"]),
                        "confirmatory_effect": float(crow["effect"]),
                        "confirmatory_p": float(crow["p_value"]),
                    }
                )
                evidence[stage] = StageEvidence(
                    discovery_effect=float(drow["effect"]),
                    confirmatory_effect=float(crow["effect"]),
                    confirmatory_p=float(crow["p_value"]),
                    hard_gates_pass=bool(drow["gate_pass"] and crow["gate_pass"]),
                )
            decisions, passed = evaluate_fixed_sequence(evidence)
            chain_pass[(geometry, object_name)] = passed
            for stage in ("R", "S", "P"):
                decision = decisions[stage]
                hypothesis_rows.append(
                    {
                        **combined.prefix("PASS"),
                        "geometry": geometry,
                        "object": object_name,
                        "stage": stage,
                        "eligible": decision.eligible,
                        "criterion_pass": decision.criterion_pass,
                        "stage_status": decision.status,
                        "chain_pass": passed,
                    }
                )
    comparison = pd.DataFrame(comparison_rows)
    for object_name in OBJECTS:
        for stage in ("R", "S", "P"):
            airm = comparison.loc[
                (comparison["geometry"] == AIRM)
                & (comparison["object"] == object_name)
                & (comparison["stage"] == stage)
            ].iloc[0]
            le = comparison.loc[
                (comparison["geometry"] == LE)
                & (comparison["object"] == object_name)
                & (comparison["stage"] == stage)
            ].iloc[0]
            robustness_rows.append(
                {
                    **combined.prefix("PASS"),
                    "object": object_name,
                    "stage": stage,
                    "airm_discovery_effect": float(airm["discovery_effect"]),
                    "airm_confirmatory_effect": float(airm["confirmatory_effect"]),
                    "le_discovery_effect": float(le["discovery_effect"]),
                    "le_confirmatory_effect": float(le["confirmatory_effect"]),
                }
            )
    terminal = terminal_airm_decision(
        chain_pass[(AIRM, "D")], chain_pass[(AIRM, "G")]
    )
    robustness_label = le_robustness_label(
        (chain_pass[(AIRM, "D")], chain_pass[(AIRM, "G")]),
        (chain_pass[(LE, "D")], chain_pass[(LE, "G")]),
    )
    comparison_columns = COMMON_COLUMNS + (
        "geometry", "object", "stage", "discovery_observed", "discovery_effect",
        "confirmatory_observed", "confirmatory_effect", "confirmatory_p",
    )
    robustness_columns = COMMON_COLUMNS + (
        "object", "stage", "airm_discovery_effect", "airm_confirmatory_effect",
        "le_discovery_effect", "le_confirmatory_effect",
    )
    hypothesis_columns = COMMON_COLUMNS + (
        "geometry", "object", "stage", "eligible", "criterion_pass",
        "stage_status", "chain_pass",
    )
    table_root = output / "tables"
    _atomic_write_csv(
        table_root / "discovery_confirmatory_comparison.csv",
        _ordered_frame(comparison_rows, comparison_columns),
    )
    _atomic_write_csv(
        table_root / "airm_le_robustness.csv",
        _ordered_frame(robustness_rows, robustness_columns),
    )
    _atomic_write_csv(
        table_root / "hypothesis_chain_status.csv",
        _ordered_frame(hypothesis_rows, hypothesis_columns),
    )

    for filename, schema in DISCOVERY_TABLE_SCHEMAS.items():
        discovery_frame = pd.read_csv(
            output / "tables" / "discovery" / filename,
            float_precision="round_trip",
        )
        confirmatory_frame = pd.read_csv(
            output / "tables" / "confirmatory" / filename,
            float_precision="round_trip",
        )
        if tuple(discovery_frame.columns) != schema or tuple(confirmatory_frame.columns) != schema:
            raise ConditionalPipelineError(f"cannot combine mismatched table: {filename}")
        _atomic_write_csv(
            table_root / filename,
            pd.concat([discovery_frame, confirmatory_frame], ignore_index=True),
        )
    for directory, filenames in (
        (
            "objects",
            (
                "airm_marginal_means.npz", "airm_class_means.npz",
                "le_marginal_means.npz", "le_class_means.npz",
                "D_matrices.npz", "G_matrices.npz",
            ),
        ),
        (
            "nulls",
            (
                "label_destruction_group_statistics.npz",
                "semantic_permutation_group_statistics.npz",
                "oracle_rank_null.npz",
            ),
        ),
    ):
        for filename in filenames:
            source = output / directory / "confirmatory" / filename
            destination = output / directory / filename
            _atomic_copy(source, destination)
            if source.read_bytes() != destination.read_bytes():
                raise ConditionalPipelineError(f"root {directory} copy differs: {filename}")
    decision = {
        **combined.prefix("PASS"),
        "terminal_decision": terminal,
        "le_robustness_label": robustness_label,
        "chains": {
            geometry: {
                object_name: bool(chain_pass[(geometry, object_name)])
                for object_name in OBJECTS
            }
            for geometry in GEOMETRIES
        },
    }
    from src.conditional_provenance_v1 import atomic_write_json

    atomic_write_json(output / "confirmatory_decision.json", decision)
    return decision


def run_confirmatory_oracle_and_finalize(
    discovery_bundle: PhaseGeometryBundle,
    confirmatory_bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    config_sha256: str,
    code_commit: str,
    oracle_replicates: int | None = None,
) -> dict[str, Any]:
    validate_locked_discovery_f_templates(discovery_bundle, config, repo_root)
    count = int(oracle_replicates or config["nulls"]["oracle_rank"]["replicates"])
    score_sets, summaries = _confirmatory_oracle_observed(
        discovery_bundle, confirmatory_bundle
    )
    groups, subjects = run_oracle_nulls_checkpointed(
        confirmatory_bundle,
        score_sets,
        config,
        repo_root,
        total_replicates=count,
    )
    tables = _confirmatory_oracle_tables(
        discovery_bundle,
        confirmatory_bundle,
        score_sets,
        summaries,
        groups,
        subjects,
        _load_phase_subject_effects(config, repo_root, "discovery"),
        config,
    )
    output = _resolve_output_root(Path(repo_root).resolve(), config)
    table_dir = output / "tables" / "confirmatory"
    for filename in (
        "oracle_permutation_all_24_scores.csv",
        "oracle_permutation_subject_summary.csv",
        "oracle_permutation_group_summary.csv",
    ):
        _atomic_write_csv(table_dir / filename, tables[filename])
    for filename in ("subject_bootstrap_summary.csv", "leave_one_subject_out_influence.csv"):
        existing = pd.read_csv(table_dir / filename, float_precision="round_trip")
        # Script 26 is resumable/idempotent: replace its P rows while preserving
        # the R/S snapshot produced by script 25.
        existing = existing.loc[existing["stage"].astype(str) != "P"].copy()
        _atomic_write_csv(
            table_dir / filename,
            pd.concat([existing, tables[filename]], ignore_index=True),
        )
    _save_null_archive(output / "nulls" / "confirmatory" / "oracle_rank_null.npz", groups)
    decision = finalize_confirmatory_decision_artifacts(
        discovery_bundle,
        confirmatory_bundle,
        config,
        repo_root,
        config_sha256=config_sha256,
        code_commit=code_commit,
    )
    return {
        "phase": "confirmatory",
        "oracle_replicates": count,
        "terminal_decision": decision["terminal_decision"],
        "le_robustness_label": decision["le_robustness_label"],
        "status": decision["status"],
    }


def run_discovery_label_dry_run(
    data: ConditionalWholeData,
    bundle: PhaseGeometryBundle,
    config: Mapping[str, Any],
    repo_root: str | Path,
    *,
    replicates: int = 3,
    workers: int = 1,
) -> dict[str, Any]:
    """Time indexed label-refits without creating any scientific null output."""

    if replicates < 1:
        raise ValueError("dry-run replicates must be positive")
    covariance_logs = spd_log(data.covariances)
    started = time.perf_counter()
    if workers == 1:
        results = [
            _label_null_one(
                index,
                data,
                covariance_logs,
                bundle,
                config,
                PHASE_DISCOVERY,
                run_batched_audit=index == 0,
            )
            for index in range(replicates)
        ]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            results = list(
                executor.map(
                    lambda index: _label_null_one(
                        index,
                        data,
                        covariance_logs,
                        bundle,
                        config,
                        PHASE_DISCOVERY,
                        run_batched_audit=int(index) == 0,
                    ),
                    range(replicates),
                )
            )
    first = next(value for value in results if value.airm_crosscheck is not None)
    elapsed = time.perf_counter() - started
    payload = {
        **bundle.context.prefix("DRY_RUN_ONLY"),
        "replicates": int(replicates),
        "workers": int(workers),
        "elapsed_seconds": float(elapsed),
        "seconds_per_replicate_wall": float(elapsed / replicates),
        "estimated_official_wall_seconds_linear": float(
            elapsed / replicates * int(config["nulls"]["label_destruction"]["replicates"])
        ),
        "scientific_output_written": False,
        "airm_scalar_crosscheck": first.airm_crosscheck,
    }
    from src.conditional_provenance_v1 import atomic_write_json

    path = _resolve_output_root(Path(repo_root).resolve(), config) / "protocol" / "label_null_dry_run.json"
    atomic_write_json(path, payload)
    return payload


__all__ = [
    "COMMON_COLUMNS",
    "FAILURE_MANIFEST_KEYS",
    "FAILURE_DECISION_KEYS",
    "GROUP_SUMMARY_COLUMNS",
    "WITHIN_RELIABILITY_COLUMNS",
    "CROSS_SHARED_COLUMNS",
    "LABEL_SUBJECT_COLUMNS",
    "ORACLE_SCORE_COLUMNS",
    "ORACLE_SUBJECT_COLUMNS",
    "DISCOVERY_TABLE_SCHEMAS",
    "ConditionalPipelineError",
    "classify_recognized_phase_failure",
    "write_unassessed_failure_artifacts",
    "ProducerContext",
    "PhaseGeometryBundle",
    "DiscoveryNullResult",
    "producer_code_commit",
    "compute_phase_geometry",
    "write_phase_geometry_outputs",
    "load_phase_geometry_bundle",
    "validate_discovery_snapshot_contract",
    "validate_confirmatory_snapshot_contract",
    "validate_label_null_dry_run",
    "run_phase_geometry_producer",
    "run_label_null_checkpointed",
    "run_semantic_nulls_checkpointed",
    "run_oracle_nulls_checkpointed",
    "run_discovery_null_producer",
    "run_discovery_label_dry_run",
    "validate_locked_discovery_f_templates",
    "run_confirmatory_semantic_nulls_checkpointed",
    "run_confirmatory_rs_producer",
    "finalize_confirmatory_decision_artifacts",
    "run_confirmatory_oracle_and_finalize",
]
