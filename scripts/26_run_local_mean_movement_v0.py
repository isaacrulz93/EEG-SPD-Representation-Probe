#!/usr/bin/env python3
"""Prepare, execute, or finalize Local Mean Covariance Movement V0."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.local_mean_movement_v0 import (
    CLASS_ORDER,
    DELTA_T_SECONDS,
    FROZEN_OPTIMIZER_SETTINGS,
    MovementGeometryNumericalError,
    MovementInference,
    MovementQuotientNumericalError,
    anti_develop_sequence,
    compute_movement_pca,
    direct_distance,
    evaluate_movement_inference,
    length_profile_distance,
    movement_distance,
    optimizer_diagnostic_rows,
    run_synthetic_numerical_gates,
    terminal_decision,
)
from src.reporting_local_mean_movement_v0 import (
    environment_record,
    implementation_source_hash,
    sha256_file,
    write_artifact_manifest,
    write_figures,
    write_json,
    write_matrix_csv,
    write_report,
)


CONFIG_PATH = ROOT / "configs" / "bnci2014_001_local_mean_movement_v0.yaml"
PROTOCOL_PATH = ROOT / "docs" / "PROTOCOL_LOCAL_MEAN_MOVEMENT_V0.md"
OUTPUT_ROOT = ROOT / "outputs" / "bnci2014_001_local_mean_movement_v0"
REPORT_PATH = OUTPUT_ROOT / "report" / "local_mean_movement_v0.md"
EXPECTED_BRANCH = "pilot/local-mean-movement-antidevelopment-v0"
PARENT_BRANCH = "pilot/local-temporal-sequence-correspondence-v0"
PARENT_FINAL_SHA = "6d5ad6a0bdd4f2d19bfee8ce6fcbb97a5c499a5d"
PARENT_PROTOCOL_SHA = "70981aa89ddbadceca42f354c3c51d05bf6dbf0c"
PARENT_RESULT_SHA = "43e926073fab0ba76fd5baa881804538f0d7beee"
FULL_RELATIVE_PATH = "outputs/bnci2014_001_local_temporal_sequence_v0/arrays/ordered_mean_sequences.npz"
SPLIT_RELATIVE_PATH = "outputs/bnci2014_001_local_temporal_sequence_v0/arrays/split_half_mean_sequences.npz"
FULL_ARTIFACT_SHA256 = "e03b94daef3eb37f9209ee7a7482ea575b1eb353804505a3e91013339da1913f"
SPLIT_ARTIFACT_SHA256 = "355f098de7ff3dcf274e5a62cf6d92022bc1b1f6ed4301a2dc53f9a19f3cd868"
SESSION_ORDER = ("0train", "1test")
HALF_ORDER = ("A", "B")


class FrozenArtifactReproductionError(RuntimeError):
    """The saved temporal mean object differs from its frozen result commit."""


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_show_bytes(commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _npz_mapping(value: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(value), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("config root must be a mapping")
    if str(config["protocol"]["branch"]) != EXPECTED_BRANCH:
        raise RuntimeError("frozen branch differs")
    if str(config["protocol"]["sha256"]) != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("protocol content hash differs from frozen config")
    lineage = config["lineage"]
    expected_lineage = {
        "temporal_parent_branch": PARENT_BRANCH,
        "temporal_final_sha": PARENT_FINAL_SHA,
        "temporal_protocol_freeze_sha": PARENT_PROTOCOL_SHA,
        "temporal_scientific_result_sha": PARENT_RESULT_SHA,
    }
    for key, expected in expected_lineage.items():
        if str(lineage[key]) != expected:
            raise RuntimeError(f"lineage changed: {key}")
    if float(config["movement"]["delta_t_seconds"]) != DELTA_T_SECONDS:
        raise RuntimeError("Delta_t changed")
    optimizer = config["optimizer"]
    settings = FROZEN_OPTIMIZER_SETTINGS
    required_optimizer = {
        "total_starts": settings.total_starts,
        "max_iterations": settings.max_iterations,
        "gradient_tolerance": settings.gradient_tolerance,
        "min_step_size": settings.min_step_size,
        "max_time_seconds": settings.max_time_seconds,
        "max_cost_evaluations": settings.max_cost_evaluations,
    }
    for key, expected in required_optimizer.items():
        if optimizer[key] != expected:
            raise RuntimeError(f"optimizer setting changed: {key}")
    if int(config["nulls"]["replicates"]) != 1999:
        raise RuntimeError("null replicate count changed")
    if int(config["nulls"]["master_seed"]) != 20260810:
        raise RuntimeError("null seed changed")
    if str(config["project"]["output_dir"]) != str(OUTPUT_ROOT.relative_to(ROOT)):
        raise RuntimeError("output namespace changed")
    return config


def _implementation(config: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    paths = [str(value) for value in config["project"]["implementation_source_files"]]
    return implementation_source_hash(ROOT, paths)


def reproduce_frozen_temporal_means() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    full_path = ROOT / FULL_RELATIVE_PATH
    split_path = ROOT / SPLIT_RELATIVE_PATH
    full_sha = sha256_file(full_path)
    split_sha = sha256_file(split_path)
    if full_sha != FULL_ARTIFACT_SHA256 or split_sha != SPLIT_ARTIFACT_SHA256:
        raise FrozenArtifactReproductionError("working artifact SHA-256 differs from frozen contract")
    frozen_full_bytes = _git_show_bytes(PARENT_RESULT_SHA, FULL_RELATIVE_PATH)
    frozen_split_bytes = _git_show_bytes(PARENT_RESULT_SHA, SPLIT_RELATIVE_PATH)
    if _bytes_sha256(frozen_full_bytes) != FULL_ARTIFACT_SHA256:
        raise FrozenArtifactReproductionError("result-commit full artifact hash differs")
    if _bytes_sha256(frozen_split_bytes) != SPLIT_ARTIFACT_SHA256:
        raise FrozenArtifactReproductionError("result-commit split artifact hash differs")
    working_full = _npz_mapping(full_path.read_bytes())
    working_split = _npz_mapping(split_path.read_bytes())
    frozen_full = _npz_mapping(frozen_full_bytes)
    frozen_split = _npz_mapping(frozen_split_bytes)
    if working_full.keys() != frozen_full.keys() or working_split.keys() != frozen_split.keys():
        raise FrozenArtifactReproductionError("artifact array keys changed")
    for key in working_full:
        if not np.array_equal(working_full[key], frozen_full[key]):
            raise FrozenArtifactReproductionError(f"full artifact array differs: {key}")
    for key in working_split:
        if not np.array_equal(working_split[key], frozen_split[key]):
            raise FrozenArtifactReproductionError(f"split artifact array differs: {key}")

    expected_full_metadata = {
        "subjects": np.arange(1, 10, dtype=np.int64),
        "sessions": np.asarray(SESSION_ORDER),
        "classes": np.asarray(CLASS_ORDER),
        "temporal_positions": np.arange(1, 6, dtype=np.int64),
    }
    expected_split_metadata = {**expected_full_metadata, "halves": np.asarray(HALF_ORDER)}
    for key, expected in expected_full_metadata.items():
        if not np.array_equal(working_full[key], expected):
            raise FrozenArtifactReproductionError(f"full metadata differs: {key}")
    for key, expected in expected_split_metadata.items():
        if not np.array_equal(working_split[key], expected):
            raise FrozenArtifactReproductionError(f"split metadata differs: {key}")
    full_means = np.asarray(working_full["means"], dtype=np.float64)
    split_means = np.asarray(working_split["means"], dtype=np.float64)
    if full_means.shape != (2, 9, 4, 5, 22, 22):
        raise FrozenArtifactReproductionError("full mean bank shape changed")
    if split_means.shape != (2, 2, 9, 4, 5, 22, 22):
        raise FrozenArtifactReproductionError("split mean bank shape changed")

    rows: list[dict[str, Any]] = []
    maximum = 0.0
    frozen_full_means = np.asarray(frozen_full["means"], dtype=np.float64)
    for session in range(2):
        for subject in range(9):
            for class_index, class_label in enumerate(CLASS_ORDER):
                for temporal in range(5):
                    difference = float(
                        np.max(
                            np.abs(
                                full_means[session, subject, class_index, temporal]
                                - frozen_full_means[session, subject, class_index, temporal]
                            )
                        )
                    )
                    maximum = max(maximum, difference)
                    rows.append(
                        {
                            "scope": "Full",
                            "half": "NA",
                            "subject": subject + 1,
                            "session": SESSION_ORDER[session],
                            "class": class_label,
                            "temporal_position": temporal + 1,
                            "maximum_absolute_difference": difference,
                            "exact_match": difference == 0.0,
                        }
                    )
    frozen_split_means = np.asarray(frozen_split["means"], dtype=np.float64)
    for half in range(2):
        for session in range(2):
            for subject in range(9):
                for class_index, class_label in enumerate(CLASS_ORDER):
                    for temporal in range(5):
                        difference = float(
                            np.max(
                                np.abs(
                                    split_means[half, session, subject, class_index, temporal]
                                    - frozen_split_means[half, session, subject, class_index, temporal]
                                )
                            )
                        )
                        maximum = max(maximum, difference)
                        rows.append(
                            {
                                "scope": "Split",
                                "half": HALF_ORDER[half],
                                "subject": subject + 1,
                                "session": SESSION_ORDER[session],
                                "class": class_label,
                                "temporal_position": temporal + 1,
                                "maximum_absolute_difference": difference,
                                "exact_match": difference == 0.0,
                            }
                        )
    table = pd.DataFrame(rows)
    if len(table) != 1080 or not bool(table["exact_match"].all()):
        raise FrozenArtifactReproductionError("not all 1,080 frozen mean matrices reproduced")
    record = {
        "status": "PASS",
        "temporal_scientific_result_sha": PARENT_RESULT_SHA,
        "full_artifact_path": FULL_RELATIVE_PATH,
        "full_artifact_sha256": full_sha,
        "split_artifact_path": SPLIT_RELATIVE_PATH,
        "split_artifact_sha256": split_sha,
        "full_matrix_count": 360,
        "split_matrix_count": 720,
        "matrix_count": 1080,
        "maximum_absolute_difference": maximum,
        "all_exact": True,
        "new_bnci_movement_statistic_computed": False,
    }
    return full_means, split_means, table, record


def prepare() -> None:
    started = time.perf_counter()
    config = _load_config()
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("prepare must run on the requested branch")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError("movement output namespace is not empty before preparation")
    for sha in (PARENT_FINAL_SHA, PARENT_PROTOCOL_SHA, PARENT_RESULT_SHA):
        if _git("cat-file", "-t", sha) != "commit":
            raise RuntimeError(f"required lineage commit is unavailable: {sha}")
    _, _, reproduction_table, reproduction = reproduce_frozen_temporal_means()
    print("CHECKPOINT R: all 1,080 frozen full/split mean matrices reproduced exactly", flush=True)
    synthetic = run_synthetic_numerical_gates()
    print("CHECKPOINT GQ: d=22 geometry and quotient synthetic gates passed", flush=True)
    implementation_sha, source_hashes = _implementation(config)
    (OUTPUT_ROOT / "protocol").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    reproduction_table.to_csv(
        OUTPUT_ROOT / "tables" / "frozen_temporal_mean_reproduction.csv", index=False
    )
    shutil.copy2(PROTOCOL_PATH, OUTPUT_ROOT / "protocol" / PROTOCOL_PATH.name)
    shutil.copy2(CONFIG_PATH, OUTPUT_ROOT / "protocol" / "frozen_config.yaml")
    write_json(OUTPUT_ROOT / "protocol" / "environment.json", environment_record())
    write_json(OUTPUT_ROOT / "protocol" / "frozen_input_reproduction.json", reproduction)
    write_json(OUTPUT_ROOT / "protocol" / "synthetic_numerical_gates.json", synthetic)
    write_json(
        OUTPUT_ROOT / "protocol" / "pre_data_provenance.json",
        {
            "branch": EXPECTED_BRANCH,
            "head_before_protocol_freeze": _git("rev-parse", "HEAD"),
            "temporal_parent_final_sha": PARENT_FINAL_SHA,
            "temporal_protocol_freeze_sha": PARENT_PROTOCOL_SHA,
            "temporal_scientific_result_sha": PARENT_RESULT_SHA,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "implementation_source_sha256": implementation_sha,
            "implementation_file_sha256": source_hashes,
            "new_bnci_movement_statistic_computed": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    write_artifact_manifest(OUTPUT_ROOT)


def _anti_develop_banks(
    full_means: np.ndarray, split_means: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    full_shape = (2, 9, 4, 4, 22, 22)
    full_arrays = {
        "z": np.empty(full_shape, dtype=np.float64),
        "local_u": np.empty(full_shape, dtype=np.float64),
        "displacements": np.empty(full_shape, dtype=np.float64),
        "transported": np.empty(full_shape, dtype=np.float64),
        "speeds": np.empty((2, 9, 4, 4), dtype=np.float64),
    }
    split_shape = (2, 2, 9, 4, 4, 22, 22)
    split_arrays = {
        "z": np.empty(split_shape, dtype=np.float64),
        "local_u": np.empty(split_shape, dtype=np.float64),
        "displacements": np.empty(split_shape, dtype=np.float64),
        "transported": np.empty(split_shape, dtype=np.float64),
        "speeds": np.empty((2, 2, 9, 4, 4), dtype=np.float64),
    }
    diagnostics: list[pd.DataFrame] = []
    for session in range(2):
        for subject in range(9):
            for class_index, class_label in enumerate(CLASS_ORDER):
                result = anti_develop_sequence(full_means[session, subject, class_index])
                for key in ("z", "local_u", "displacements", "transported", "speeds"):
                    full_arrays[key][session, subject, class_index] = getattr(result, key)
                frame = result.diagnostics.copy()
                frame.insert(0, "scope", "Full")
                frame.insert(1, "half", "NA")
                frame.insert(2, "subject", subject + 1)
                frame.insert(3, "session", SESSION_ORDER[session])
                frame.insert(4, "class", class_label)
                diagnostics.append(frame)
    for half in range(2):
        for session in range(2):
            for subject in range(9):
                for class_index, class_label in enumerate(CLASS_ORDER):
                    result = anti_develop_sequence(split_means[half, session, subject, class_index])
                    for key in ("z", "local_u", "displacements", "transported", "speeds"):
                        split_arrays[key][half, session, subject, class_index] = getattr(result, key)
                    frame = result.diagnostics.copy()
                    frame.insert(0, "scope", "Split")
                    frame.insert(1, "half", HALF_ORDER[half])
                    frame.insert(2, "subject", subject + 1)
                    frame.insert(3, "session", SESSION_ORDER[session])
                    frame.insert(4, "class", class_label)
                    diagnostics.append(frame)
    diagnostic_table = pd.concat(diagnostics, ignore_index=True)
    summary = {
        "status": "PASS",
        "sequence_count": 216,
        "transition_count": 864,
        "maximum_norm_absolute_error": float(
            diagnostic_table["maximum_norm_absolute_error"].max()
        ),
        "maximum_edge_transport_relative_error": float(
            diagnostic_table["maximum_edge_transport_relative_error"].max()
        ),
        "maximum_z_symmetry_relative_error": float(
            diagnostic_table["z_symmetry_relative_error"].max()
        ),
        "all_passed": bool(diagnostic_table["passed"].all()),
    }
    if not summary["all_passed"]:
        raise MovementGeometryNumericalError(
            "UNASSESSED_MOVEMENT_GEOMETRY_NUMERICAL_FAILURE: real anti-development gate failed"
        )
    return full_arrays, split_arrays, diagnostic_table, summary


def _primary_fit_job(
    row: int, column: int, first: np.ndarray, second: np.ndarray
) -> tuple[int, int, float, list[dict[str, Any]], dict[str, Any]]:
    distance, fit = movement_distance(first, second)
    rows = list(
        optimizer_diagnostic_rows(
            fit, analysis="cross_session", row_cell=row, column_cell=column
        )
    )
    summary = {
        "row_cell": row,
        "column_cell": column,
        "distance": distance,
        "objective": fit.objective,
        "determinant": fit.determinant,
        "gradient_norm": fit.gradient_norm,
        "best_start_index": fit.best_start_index,
        "second_best_objective": fit.second_best_objective,
        "objective_spread": fit.objective_spread,
    }
    return row, column, distance, rows, summary


def _split_fit_job(
    index: int, first: np.ndarray, second: np.ndarray
) -> tuple[int, float, list[dict[str, Any]], dict[str, Any]]:
    distance, fit = movement_distance(first, second)
    rows = list(
        optimizer_diagnostic_rows(
            fit, analysis="split_half", row_cell=index, column_cell=index
        )
    )
    summary = {
        "split_cell": index,
        "distance": distance,
        "objective": fit.objective,
        "determinant": fit.determinant,
        "gradient_norm": fit.gradient_norm,
        "best_start_index": fit.best_start_index,
        "second_best_objective": fit.second_best_objective,
        "objective_spread": fit.objective_spread,
    }
    return index, distance, rows, summary


def _compute_primary_quotient(
    movements: np.ndarray, *, workers: int
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, float]:
    started = time.perf_counter()
    session0 = movements[0].reshape(36, 4, 22, 22)
    session1 = movements[1].reshape(36, 4, 22, 22)
    jobs = (
        delayed(_primary_fit_job)(row, column, session0[row], session1[column])
        for row in range(36)
        for column in range(36)
    )
    results = Parallel(n_jobs=workers, backend="loky", verbose=10)(jobs)
    matrix = np.empty((36, 36), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row, column, distance, fit_rows, summary in results:
        matrix[row, column] = distance
        diagnostics.extend(fit_rows)
        summaries.append(summary)
    return matrix, pd.DataFrame(diagnostics), pd.DataFrame(summaries), time.perf_counter() - started


def _compute_split_quotient(
    split_movements: np.ndarray, *, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    started = time.perf_counter()
    jobs = []
    metadata: list[dict[str, Any]] = []
    index = 0
    for session in range(2):
        for subject in range(9):
            for class_index, class_label in enumerate(CLASS_ORDER):
                jobs.append(
                    delayed(_split_fit_job)(
                        index,
                        split_movements[0, session, subject, class_index],
                        split_movements[1, session, subject, class_index],
                    )
                )
                metadata.append(
                    {
                        "split_cell": index,
                        "subject": subject + 1,
                        "session": SESSION_ORDER[session],
                        "class": class_label,
                    }
                )
                index += 1
    results = Parallel(n_jobs=workers, backend="loky", verbose=10)(jobs)
    diagnostics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    distances: dict[int, float] = {}
    for split_index, distance, fit_rows, summary in results:
        distances[split_index] = distance
        diagnostics.extend(fit_rows)
        summaries.append(summary)
    table = pd.DataFrame(metadata)
    table["d_mov"] = [distances[int(value)] for value in table["split_cell"]]
    return table, pd.DataFrame(diagnostics), pd.DataFrame(summaries), time.perf_counter() - started


def _control_matrices(
    movements: np.ndarray, speeds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    session0 = movements[0].reshape(36, 4, 22, 22)
    session1 = movements[1].reshape(36, 4, 22, 22)
    speed0 = speeds[0].reshape(36, 4)
    speed1 = speeds[1].reshape(36, 4)
    direct = np.empty((36, 36), dtype=np.float64)
    length = np.empty((36, 36), dtype=np.float64)
    for row in range(36):
        for column in range(36):
            direct[row, column] = direct_distance(session0[row], session1[column])
            length[row, column] = length_profile_distance(speed0[row], speed1[column])
    return length, direct


def _subject_table(inferences: Mapping[str, MovementInference]) -> pd.DataFrame:
    rows = []
    for representation, inference in inferences.items():
        for subject in range(9):
            rows.append(
                {
                    "representation": representation,
                    "subject": subject + 1,
                    "S_s": inference.observed.s_s[subject],
                    "C_s": inference.observed.c_s[subject],
                    "J_s": inference.observed.j_s[subject],
                }
            )
    return pd.DataFrame(rows)


def _cell_table(inferences: Mapping[str, MovementInference]) -> pd.DataFrame:
    rows = []
    for representation, inference in inferences.items():
        observed = inference.observed
        for subject in range(9):
            for class_index, class_label in enumerate(CLASS_ORDER):
                rows.append(
                    {
                        "representation": representation,
                        "subject": subject + 1,
                        "class": class_label,
                        "a_sc": observed.a_sc[subject, class_index],
                        "b_sc": observed.b_sc[subject, class_index],
                        "c_sc": observed.c_sc[subject, class_index],
                        "d_sc": observed.d_sc[subject, class_index],
                        "S_sc": observed.s_sc[subject, class_index],
                        "C_sc": observed.c_specific_sc[subject, class_index],
                        "J_sc": observed.j_sc[subject, class_index],
                    }
                )
    return pd.DataFrame(rows)


def _summary_table(inferences: Mapping[str, MovementInference]) -> pd.DataFrame:
    rows = []
    for representation, inference in inferences.items():
        rows.append(
            {
                "representation": representation,
                "T_subject": inference.observed.t_subject,
                "p_subject": inference.p_subject,
                "T_class": inference.observed.t_class,
                "p_class": inference.p_class,
                "T_J": inference.observed.t_j,
                "p_J_subjectbreak": inference.p_j_subjectbreak,
                "p_J_classbreak": inference.p_j_classbreak,
            }
        )
    return pd.DataFrame(rows)


def _fixed_comparisons(
    movement: np.ndarray, length: np.ndarray, direct: np.ndarray
) -> pd.DataFrame:
    definitions = (
        ("S1 Left vs S1 Left", 0),
        ("S1 Left vs S2 Left", 4),
        ("S1 Left vs S1 Feet", 2),
        ("S1 Left vs S2 Feet", 6),
    )
    return pd.DataFrame(
        [
            {
                "comparison": name,
                "d_mov": movement[0, column],
                "d_len": length[0, column],
                "d_direct": direct[0, column],
            }
            for name, column in definitions
        ]
    )


def _pca_coordinate_table(coordinates: np.ndarray) -> pd.DataFrame:
    rows = []
    for session in range(2):
        for subject in range(9):
            for class_index, class_label in enumerate(CLASS_ORDER):
                for transition in range(4):
                    rows.append(
                        {
                            "subject": subject + 1,
                            "session": SESSION_ORDER[session],
                            "class": class_label,
                            "transition": transition + 1,
                            "PC1": coordinates[session, subject, class_index, transition, 0],
                            "PC2": coordinates[session, subject, class_index, transition, 1],
                        }
                    )
    return pd.DataFrame(rows)


def _speed_table(speeds: np.ndarray) -> pd.DataFrame:
    rows = []
    for session in range(2):
        for subject in range(9):
            for class_index, class_label in enumerate(CLASS_ORDER):
                for transition in range(4):
                    rows.append(
                        {
                            "subject": subject + 1,
                            "session": SESSION_ORDER[session],
                            "class": class_label,
                            "transition": transition + 1,
                            "speed": speeds[session, subject, class_index, transition],
                        }
                    )
    return pd.DataFrame(rows)


def _optimizer_summary(
    primary_diagnostics: pd.DataFrame,
    primary_fits: pd.DataFrame,
    split_diagnostics: pd.DataFrame,
    split_fits: pd.DataFrame,
) -> dict[str, Any]:
    diagnostics = pd.concat((primary_diagnostics, split_diagnostics), ignore_index=True)
    selected = diagnostics.loc[diagnostics["selected"]]
    primary_sector_counts = primary_diagnostics.loc[
        primary_diagnostics["converged"]
    ].groupby(["row_cell", "column_cell"])["final_determinant"].nunique()
    split_sector_counts = split_diagnostics.loc[
        split_diagnostics["converged"]
    ].groupby(["row_cell", "column_cell"])["final_determinant"].nunique()
    certified = bool(
        len(primary_fits) == 1296
        and len(split_fits) == 72
        and (primary_sector_counts == 2).all()
        and (split_sector_counts == 2).all()
        and np.isfinite(selected["objective"]).all()
    )
    if not certified:
        raise MovementQuotientNumericalError(
            "UNASSESSED_MOVEMENT_QUOTIENT_NUMERICAL_FAILURE: aggregate optimizer certification failed"
        )
    return {
        "status": "PASS",
        "primary_fit_count": len(primary_fits),
        "split_fit_count": len(split_fits),
        "starts_per_fit": FROZEN_OPTIMIZER_SETTINGS.total_starts,
        "primary_both_sector_fit_count": int(np.count_nonzero(primary_sector_counts == 2)),
        "split_both_sector_fit_count": int(np.count_nonzero(split_sector_counts == 2)),
        "maximum_selected_gradient_norm": float(selected["gradient_norm"].max()),
        "nonconverged_start_count": int(np.count_nonzero(~diagnostics["converged"])),
    }


def _write_failure(terminal: str, error: Exception) -> None:
    write_json(
        OUTPUT_ROOT / "decisions" / "terminal_decision.json",
        {
            "terminal": terminal,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


def run(protocol_freeze_sha: str) -> None:
    total_started = time.perf_counter()
    config = _load_config()
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("scientific run must use the requested branch")
    if _git("status", "--porcelain"):
        raise RuntimeError("scientific execution requires a clean worktree")
    if _git("rev-parse", "HEAD") != protocol_freeze_sha:
        raise RuntimeError("HEAD must equal the supplied protocol-freeze SHA")
    tests_path = OUTPUT_ROOT / "protocol" / "test_results.json"
    if not tests_path.exists():
        raise RuntimeError("frozen test record is missing")
    implementation_sha, source_hashes = _implementation(config)
    pre_data = json.loads(
        (OUTPUT_ROOT / "protocol" / "pre_data_provenance.json").read_text(encoding="utf-8")
    )
    if implementation_sha != pre_data["implementation_source_sha256"]:
        raise RuntimeError("implementation changed after pre-data validation")
    try:
        full_means, split_means, reproduction_table, reproduction = reproduce_frozen_temporal_means()
        reproduction_table.to_csv(
            OUTPUT_ROOT / "tables" / "frozen_temporal_mean_reproduction.csv", index=False
        )
        print("CHECKPOINT R2: frozen temporal mean artifacts reproduced at scientific run", flush=True)
        synthetic = run_synthetic_numerical_gates()
        print("CHECKPOINT GQ2: synthetic geometry and quotient gates reproduced", flush=True)
        full_arrays, split_arrays, anti_diagnostics, geometry_summary = _anti_develop_banks(
            full_means, split_means
        )
        print("CHECKPOINT A: 216 full/split anti-developments passed", flush=True)
        workers = int(config["optimizer"]["parallel_workers"])
        movement_matrix, primary_diag, primary_fits, primary_seconds = _compute_primary_quotient(
            full_arrays["z"], workers=workers
        )
        print(
            "CHECKPOINT SCIENCE: first cross-session movement matrix observed; frozen settings immutable",
            flush=True,
        )
        split_table, split_diag, split_fits, split_seconds = _compute_split_quotient(
            split_arrays["z"], workers=workers
        )
        optimizer_summary = _optimizer_summary(
            primary_diag, primary_fits, split_diag, split_fits
        )
        length_matrix, direct_matrix = _control_matrices(
            full_arrays["z"], full_arrays["speeds"]
        )
        inferences = {
            "d_mov": evaluate_movement_inference(movement_matrix),
            "d_len": evaluate_movement_inference(length_matrix),
            "d_direct": evaluate_movement_inference(direct_matrix),
        }
        movement_inference = inferences["d_mov"]
        terminal = terminal_decision(
            t_subject=movement_inference.observed.t_subject,
            p_subject=movement_inference.p_subject,
            t_class=movement_inference.observed.t_class,
            p_class=movement_inference.p_class,
            t_j=movement_inference.observed.t_j,
            p_j_subjectbreak=movement_inference.p_j_subjectbreak,
            p_j_classbreak=movement_inference.p_j_classbreak,
        )
        pca = compute_movement_pca(full_arrays["z"])
        fixed = _fixed_comparisons(movement_matrix, length_matrix, direct_matrix)
        runtime = {
            "total_seconds": time.perf_counter() - total_started,
            "primary_quotient_seconds": primary_seconds,
            "split_quotient_seconds": split_seconds,
        }

        arrays_dir = OUTPUT_ROOT / "arrays"
        tables_dir = OUTPUT_ROOT / "tables"
        nulls_dir = OUTPUT_ROOT / "nulls"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        nulls_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_dir / "full_ordered_antidevelopment.npz",
            **full_arrays,
            subjects=np.arange(1, 10),
            sessions=np.asarray(SESSION_ORDER),
            classes=np.asarray(CLASS_ORDER),
            transitions=np.arange(1, 5),
            delta_t_seconds=np.asarray(DELTA_T_SECONDS),
        )
        np.savez_compressed(
            arrays_dir / "split_half_ordered_antidevelopment.npz",
            **split_arrays,
            halves=np.asarray(HALF_ORDER),
            subjects=np.arange(1, 10),
            sessions=np.asarray(SESSION_ORDER),
            classes=np.asarray(CLASS_ORDER),
            transitions=np.arange(1, 5),
            delta_t_seconds=np.asarray(DELTA_T_SECONDS),
        )
        np.savez_compressed(
            arrays_dir / "cross_session_movement_matrices.npz",
            d_mov=movement_matrix,
            d_len=length_matrix,
            d_direct=direct_matrix,
            cell_subjects=np.repeat(np.arange(1, 10), 4),
            cell_classes=np.tile(np.asarray(CLASS_ORDER), 9),
        )
        np.savez_compressed(
            arrays_dir / "common_movement_pca.npz",
            coordinates=pca.coordinates,
            components=pca.components,
            feature_mean=pca.feature_mean,
            explained_variance=pca.explained_variance,
            explained_variance_ratio=pca.explained_variance_ratio,
        )
        np.savez_compressed(
            nulls_dir / "movement_null_distributions.npz",
            d_mov_subjectbreak_t_subject=inferences["d_mov"].subjectbreak_t_subject,
            d_mov_subjectbreak_t_j=inferences["d_mov"].subjectbreak_t_j,
            d_mov_classbreak_t_class=inferences["d_mov"].classbreak_t_class,
            d_mov_classbreak_t_j=inferences["d_mov"].classbreak_t_j,
            d_len_subjectbreak_t_subject=inferences["d_len"].subjectbreak_t_subject,
            d_len_subjectbreak_t_j=inferences["d_len"].subjectbreak_t_j,
            d_len_classbreak_t_class=inferences["d_len"].classbreak_t_class,
            d_len_classbreak_t_j=inferences["d_len"].classbreak_t_j,
            d_direct_subjectbreak_t_subject=inferences["d_direct"].subjectbreak_t_subject,
            d_direct_subjectbreak_t_j=inferences["d_direct"].subjectbreak_t_j,
            d_direct_classbreak_t_class=inferences["d_direct"].classbreak_t_class,
            d_direct_classbreak_t_j=inferences["d_direct"].classbreak_t_j,
            subject_mappings=inferences["d_mov"].subject_mappings,
            class_mappings=inferences["d_mov"].class_mappings,
        )
        write_matrix_csv(tables_dir / "d_mov_matrix.csv", movement_matrix)
        write_matrix_csv(tables_dir / "d_len_matrix.csv", length_matrix)
        write_matrix_csv(tables_dir / "d_direct_matrix.csv", direct_matrix)
        anti_diagnostics.to_csv(tables_dir / "antidevelopment_numerical_diagnostics.csv", index=False)
        pd.concat((primary_diag, split_diag), ignore_index=True).to_csv(
            tables_dir / "optimizer_start_diagnostics.csv", index=False
        )
        primary_fits.to_csv(tables_dir / "primary_optimizer_fit_summary.csv", index=False)
        split_fits.to_csv(tables_dir / "split_optimizer_fit_summary.csv", index=False)
        split_table.to_csv(tables_dir / "split_half_movement_reliability.csv", index=False)
        _summary_table(inferences).to_csv(tables_dir / "movement_inference_summary.csv", index=False)
        _subject_table(inferences).to_csv(tables_dir / "subject_movement_contrasts.csv", index=False)
        _cell_table(inferences).to_csv(tables_dir / "subject_class_movement_contrasts.csv", index=False)
        fixed.to_csv(tables_dir / "fixed_illustrative_comparisons.csv", index=False)
        _pca_coordinate_table(pca.coordinates).to_csv(
            tables_dir / "common_movement_pca_coordinates.csv", index=False
        )
        _speed_table(full_arrays["speeds"]).to_csv(
            tables_dir / "ordered_speed_profiles.csv", index=False
        )
        write_json(OUTPUT_ROOT / "protocol" / "scientific_runtime.json", runtime)
        write_json(OUTPUT_ROOT / "protocol" / "real_antidevelopment_gates.json", geometry_summary)
        write_json(OUTPUT_ROOT / "protocol" / "optimizer_summary.json", optimizer_summary)
        write_json(
            OUTPUT_ROOT / "protocol" / "scientific_provenance.json",
            {
                "branch": EXPECTED_BRANCH,
                "protocol_freeze_sha": protocol_freeze_sha,
                "temporal_parent_final_sha": PARENT_FINAL_SHA,
                "temporal_protocol_freeze_sha": PARENT_PROTOCOL_SHA,
                "temporal_scientific_result_sha": PARENT_RESULT_SHA,
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "config_sha256": sha256_file(CONFIG_PATH),
                "implementation_source_sha256": implementation_sha,
                "implementation_file_sha256": source_hashes,
                "git_status_at_start": "clean",
                "head_at_start": protocol_freeze_sha,
                "first_result_checkpoint": "after complete d_mov matrix",
                "scientific_setting_changed_after_result_access": False,
            },
        )
        write_json(
            OUTPUT_ROOT / "decisions" / "terminal_decision.json",
            {
                "terminal": terminal,
                "T_subject": movement_inference.observed.t_subject,
                "p_subject": movement_inference.p_subject,
                "T_class": movement_inference.observed.t_class,
                "p_class": movement_inference.p_class,
                "T_J": movement_inference.observed.t_j,
                "p_J_subjectbreak": movement_inference.p_j_subjectbreak,
                "p_J_classbreak": movement_inference.p_j_classbreak,
            },
        )
        write_figures(
            output_root=OUTPUT_ROOT,
            pca=pca,
            movement_matrix=movement_matrix,
            movement_inference=movement_inference,
            split_table=split_table,
            speeds=full_arrays["speeds"],
        )
        tests = json.loads(tests_path.read_text(encoding="utf-8"))
        write_report(
            REPORT_PATH,
            branch=EXPECTED_BRANCH,
            parent_final_sha=PARENT_FINAL_SHA,
            parent_protocol_sha=PARENT_PROTOCOL_SHA,
            parent_result_sha=PARENT_RESULT_SHA,
            protocol_freeze_sha=protocol_freeze_sha,
            reproduction=reproduction,
            geometry_summary=geometry_summary,
            synthetic_gates=synthetic,
            optimizer_summary=optimizer_summary,
            movement=inferences["d_mov"],
            length=inferences["d_len"],
            direct=inferences["d_direct"],
            split_table=split_table,
            fixed_comparisons=fixed,
            terminal=terminal,
            runtime=runtime,
            tests=tests,
        )
        write_artifact_manifest(OUTPUT_ROOT)
        print(
            f"CHECKPOINT RESULT: {terminal}; T_subject={movement_inference.observed.t_subject:.10f}, "
            f"T_class={movement_inference.observed.t_class:.10f}, T_J={movement_inference.observed.t_j:.10f}",
            flush=True,
        )
    except MovementGeometryNumericalError as error:
        _write_failure("UNASSESSED_MOVEMENT_GEOMETRY_NUMERICAL_FAILURE", error)
        raise
    except MovementQuotientNumericalError as error:
        _write_failure("UNASSESSED_MOVEMENT_QUOTIENT_NUMERICAL_FAILURE", error)
        raise
    except Exception as error:
        _write_failure("UNASSESSED_TECHNICAL_FAILURE", error)
        raise


def finalize(scientific_result_sha: str) -> None:
    config = _load_config()
    if _git("status", "--porcelain"):
        raise RuntimeError("finalize requires a clean scientific-result commit")
    if _git("rev-parse", "HEAD") != scientific_result_sha:
        raise RuntimeError("HEAD must equal the supplied scientific-result SHA")
    scientific = json.loads(
        (OUTPUT_ROOT / "protocol" / "scientific_provenance.json").read_text(encoding="utf-8")
    )
    protocol_freeze_sha = str(scientific["protocol_freeze_sha"])
    implementation_sha, source_hashes = _implementation(config)
    if implementation_sha != scientific["implementation_source_sha256"]:
        raise RuntimeError("implementation changed after the scientific run")
    changed = [
        value
        for value in _git("diff", "--name-only", protocol_freeze_sha, scientific_result_sha).splitlines()
        if value
    ]
    output_prefix = str(OUTPUT_ROOT.relative_to(ROOT)) + "/"
    if not changed or any(not value.startswith(output_prefix) for value in changed):
        raise RuntimeError("scientific result commit changed files outside the frozen output namespace")
    report = REPORT_PATH.read_text(encoding="utf-8")
    placeholder = "FINAL_RESULT_SHA_PENDING"
    if report.count(placeholder) != 1:
        raise RuntimeError("report final-result placeholder is missing or duplicated")
    REPORT_PATH.write_text(report.replace(placeholder, scientific_result_sha), encoding="utf-8")
    write_json(
        OUTPUT_ROOT / "protocol" / "post_result_provenance.json",
        {
            "scientific_result_sha": scientific_result_sha,
            "protocol_freeze_sha": protocol_freeze_sha,
            "branch": EXPECTED_BRANCH,
            "git_status_before_finalization": "clean",
            "changed_paths_protocol_to_result": changed,
            "all_scientific_result_changes_confined_to_output_namespace": True,
            "protocol_sha256_unchanged": sha256_file(PROTOCOL_PATH),
            "config_sha256_unchanged": sha256_file(CONFIG_PATH),
            "implementation_source_sha256_unchanged": implementation_sha,
            "implementation_file_sha256_unchanged": source_hashes,
            "scientific_setting_changed_after_result_access": False,
            "finalization_scope": [
                "insert scientific result SHA into report",
                "write post-result provenance",
                "refresh artifact manifest",
            ],
        },
    )
    write_artifact_manifest(OUTPUT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol-freeze-sha", required=True)
    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--scientific-result-sha", required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare()
    elif arguments.command == "run":
        run(arguments.protocol_freeze_sha)
    else:
        finalize(arguments.scientific_result_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
