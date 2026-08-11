#!/usr/bin/env python3
"""Prepare, execute, post-test, or finalize component decomposition V0."""

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

from src.local_mean_movement_v0 import (  # noqa: E402
    _array_sha256,
    movement_distance,
)
from src.local_movement_component_decomposition_v0 import (  # noqa: E402
    ABSOLUTE_TOLERANCE,
    COMPONENT_ORDER,
    N_CELLS,
    N_CHANNELS,
    N_STEPS,
    RELATIVE_TOLERANCE,
    ComponentDecompositionNumericalError,
    ComponentMatrices,
    canonical_cell_classes,
    canonical_cell_subjects,
    descriptive_fractions,
    evaluate_all_components,
    null_reconstruction_gates,
    pairwise_numerical_gates,
    relation_statistics,
    sensor_and_length_squared_costs,
    split_replicate_banks,
    statistic_reconstruction_gates,
    terminal_decision,
)
from src.local_temporal_sequence_v0 import (  # noqa: E402
    CLASS_ORDER,
    classbreak_mappings,
    subjectbreak_mappings,
)
from src.reporting_local_movement_component_decomposition_v0 import (  # noqa: E402
    environment_record,
    git_value,
    implementation_source_hash,
    sha256_file,
    write_artifact_manifest,
    write_figures,
    write_json,
    write_matrix_csv,
    write_report,
)


CONFIG_PATH = ROOT / "configs" / "bnci2014_001_local_movement_component_decomposition_v0.yaml"
PROTOCOL_PATH = ROOT / "docs" / "PROTOCOL_LOCAL_MOVEMENT_COMPONENT_DECOMPOSITION_V0.md"
OUTPUT_ROOT = ROOT / "outputs" / "bnci2014_001_local_movement_component_decomposition_v0"
REPORT_PATH = OUTPUT_ROOT / "report" / "local_movement_component_decomposition_v0.md"
EXPECTED_BRANCH = "pilot/local-movement-component-decomposition-v0"
PARENT_BRANCH = "pilot/local-mean-movement-antidevelopment-v0"
PARENT_SHA = "12c19f38266bc76875cffae056e7f9403df299c1"
PARENT_PROTOCOL_SHA = "e24312147ef3020854ef6f6cd174071d1c6ead02"
PARENT_RESULT_SHA = "c3f1d5ff9cf23db2007bbf839cf4b266e2cb8960"
PARENT_TERMINAL = "GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT"
PARENT_OUTPUT = Path("outputs/bnci2014_001_local_mean_movement_v0")
FULL_RELATIVE_PATH = PARENT_OUTPUT / "arrays" / "full_ordered_antidevelopment.npz"
SPLIT_RELATIVE_PATH = PARENT_OUTPUT / "arrays" / "split_half_ordered_antidevelopment.npz"
MATRICES_RELATIVE_PATH = PARENT_OUTPUT / "arrays" / "cross_session_movement_matrices.npz"
NULLS_RELATIVE_PATH = PARENT_OUTPUT / "nulls" / "movement_null_distributions.npz"
FROZEN_ARTIFACT_HASHES = {
    str(FULL_RELATIVE_PATH): "f3771773a194088e84d765b543ba50db95fc9571ab85838aa43a581eca32a2d3",
    str(SPLIT_RELATIVE_PATH): "ab64bfdb279805cf5d0e0b9bc12c65eccd1c602ee14e3084f7429e9588e98520",
    str(MATRICES_RELATIVE_PATH): "f3470799a2a532d98f9406902cc7bb75f9385f899991ade350d63e3ca78ef5dc",
    str(NULLS_RELATIVE_PATH): "954fbaaf761157332a67dfebcc8085ae1a4a8c38b2921340e1226a37e30b2fa1",
    str(PARENT_OUTPUT / "tables" / "d_mov_matrix.csv"): "be1abec7e3081df6ffc8ac919ce12ceb6b94f527d2371984e23a8c92315ca64e",
    str(PARENT_OUTPUT / "tables" / "d_len_matrix.csv"): "b57ad89bf6225df5728135424ab48848cb142bbae9aba0bb58cdab84a46f1e2f",
    str(PARENT_OUTPUT / "tables" / "d_direct_matrix.csv"): "1d4ea0b4692565018f33f2889623e8bf372244788d9ff8c45987c442cbf4501c",
}
IMPLEMENTATION_PATHS = (
    "src/local_movement_component_decomposition_v0.py",
    "src/reporting_local_movement_component_decomposition_v0.py",
    "scripts/27_run_local_movement_component_decomposition_v0.py",
)
FIXED_COMPARISONS = (
    ("S1 Left0 -> S1 Left1", 0),
    ("S1 Left0 -> S2 Left1", 4),
    ("S1 Left0 -> S1 Feet1", 2),
    ("S1 Left0 -> S2 Feet1", 6),
)


class FrozenInputReproductionError(RuntimeError):
    """A frozen parent artifact or lineage check failed."""


def _git(*arguments: str) -> str:
    return git_value(ROOT, *arguments)


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


def _exact_array_equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.dtype.kind in "fc" and second.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("config root must be a mapping")
    required = {
        "branch": EXPECTED_BRANCH,
        "sha256": sha256_file(PROTOCOL_PATH),
    }
    for key, expected in required.items():
        if str(config["protocol"][key]) != expected:
            raise RuntimeError(f"protocol {key} changed")
    lineage = config["lineage"]
    for key, expected in {
        "parent_branch": PARENT_BRANCH,
        "parent_sha": PARENT_SHA,
        "movement_protocol_freeze_sha": PARENT_PROTOCOL_SHA,
        "movement_scientific_result_sha": PARENT_RESULT_SHA,
        "movement_terminal": PARENT_TERMINAL,
    }.items():
        if str(lineage[key]) != expected:
            raise RuntimeError(f"lineage changed: {key}")
    numerical = config["numerical"]
    if float(numerical["atol"]) != ABSOLUTE_TOLERANCE:
        raise RuntimeError("absolute tolerance changed")
    if float(numerical["rtol"]) != RELATIVE_TOLERANCE:
        raise RuntimeError("relative tolerance changed")
    if int(config["nulls"]["replicates"]) != 1999:
        raise RuntimeError("null replicate count changed")
    if int(config["optimizer"]["parallel_workers"]) != 4:
        raise RuntimeError("parallel worker contract changed")
    if config["artifacts"] != FROZEN_ARTIFACT_HASHES:
        raise RuntimeError("frozen artifact hash contract changed")
    return config


def _implementation() -> tuple[str, dict[str, str]]:
    return implementation_source_hash(ROOT, IMPLEMENTATION_PATHS)


def reproduce_frozen_inputs() -> dict[str, Any]:
    """Require byte equality and exact NPZ-array equality to the result commit."""

    records: dict[str, Any] = {}
    for relative, expected_hash in FROZEN_ARTIFACT_HASHES.items():
        local_path = ROOT / relative
        if not local_path.exists():
            raise FrozenInputReproductionError(f"missing frozen artifact: {relative}")
        local_hash = sha256_file(local_path)
        result_bytes = _git_show_bytes(PARENT_RESULT_SHA, relative)
        result_hash = _bytes_sha256(result_bytes)
        if local_hash != expected_hash or result_hash != expected_hash:
            raise FrozenInputReproductionError(
                f"frozen artifact hash mismatch: {relative} local={local_hash} result={result_hash}"
            )
        if local_path.read_bytes() != result_bytes:
            raise FrozenInputReproductionError(f"frozen artifact bytes differ: {relative}")
        records[relative] = {
            "sha256": local_hash,
            "byte_equal_to_parent_result": True,
        }
        if relative.endswith(".npz"):
            current = _npz_mapping(local_path.read_bytes())
            frozen = _npz_mapping(result_bytes)
            if set(current) != set(frozen) or any(
                not _exact_array_equal(current[key], frozen[key]) for key in current
            ):
                raise FrozenInputReproductionError(
                    f"frozen NPZ arrays differ exactly: {relative}"
                )
            records[relative]["exact_array_equal_to_parent_result"] = True
            records[relative]["keys"] = sorted(current)

    with np.load(ROOT / FULL_RELATIVE_PATH, allow_pickle=False) as full:
        if full["z"].shape != (2, 9, 4, 4, 22, 22):
            raise FrozenInputReproductionError("full Z bank shape changed")
        if not np.array_equal(full["subjects"], np.arange(1, 10)):
            raise FrozenInputReproductionError("full subject order changed")
        if not np.array_equal(full["classes"], np.asarray(CLASS_ORDER)):
            raise FrozenInputReproductionError("full class order changed")
        if not np.array_equal(full["sessions"], np.asarray(("0train", "1test"))):
            raise FrozenInputReproductionError("full session order changed")
        if float(full["delta_t_seconds"]) != 0.8:
            raise FrozenInputReproductionError("Delta_t changed")
    with np.load(ROOT / SPLIT_RELATIVE_PATH, allow_pickle=False) as split:
        if split["z"].shape != (2, 2, 9, 4, 4, 22, 22):
            raise FrozenInputReproductionError("split Z bank shape changed")
        if not np.array_equal(split["halves"], np.asarray(("A", "B"))):
            raise FrozenInputReproductionError("split half order changed")
    with np.load(ROOT / MATRICES_RELATIVE_PATH, allow_pickle=False) as matrices:
        for key in ("d_mov", "d_len", "d_direct"):
            if matrices[key].shape != (36, 36):
                raise FrozenInputReproductionError(f"{key} shape changed")
        if not np.array_equal(matrices["cell_subjects"], canonical_cell_subjects()):
            raise FrozenInputReproductionError("matrix subject order changed")
        if not np.array_equal(matrices["cell_classes"], canonical_cell_classes()):
            raise FrozenInputReproductionError("matrix class order changed")
    with np.load(ROOT / NULLS_RELATIVE_PATH, allow_pickle=False) as nulls:
        saved_subject = nulls["subject_mappings"]
        saved_class = nulls["class_mappings"]
    if not np.array_equal(saved_subject, subjectbreak_mappings()):
        raise FrozenInputReproductionError("saved subject mappings differ from frozen stream")
    if not np.array_equal(saved_class, classbreak_mappings()):
        raise FrozenInputReproductionError("saved class mappings differ from frozen stream")
    return {
        "status": "PASS",
        "parent_sha": PARENT_SHA,
        "parent_result_sha": PARENT_RESULT_SHA,
        "artifact_count": len(records),
        "artifacts": records,
        "canonical_ordering": "subject_1_to_9_x_left_hand_right_hand_feet_tongue",
        "saved_subject_mappings_equal_regenerated": True,
        "saved_class_mappings_equal_regenerated": True,
        "new_component_statistic_computed": False,
    }


def prepare() -> None:
    started = time.perf_counter()
    config = _load_config()
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("prepare must use the requested branch")
    if _git("rev-parse", "HEAD") != PARENT_SHA:
        raise RuntimeError("prepare must begin at the exact authoritative parent")
    reproduction = reproduce_frozen_inputs()
    implementation_sha, source_hashes = _implementation()
    (OUTPUT_ROOT / "protocol").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROTOCOL_PATH, OUTPUT_ROOT / "protocol" / PROTOCOL_PATH.name)
    shutil.copy2(CONFIG_PATH, OUTPUT_ROOT / "protocol" / "frozen_config.yaml")
    write_json(OUTPUT_ROOT / "protocol" / "frozen_input_reproduction.json", reproduction)
    write_json(OUTPUT_ROOT / "protocol" / "environment.json", environment_record())
    write_json(
        OUTPUT_ROOT / "protocol" / "pre_data_provenance.json",
        {
            "branch": EXPECTED_BRANCH,
            "head_before_protocol_freeze": PARENT_SHA,
            "parent_protocol_freeze_sha": PARENT_PROTOCOL_SHA,
            "parent_scientific_result_sha": PARENT_RESULT_SHA,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "implementation_source_sha256": implementation_sha,
            "implementation_file_sha256": source_hashes,
            "elapsed_seconds": time.perf_counter() - started,
            "new_component_statistic_computed": False,
        },
    )
    write_artifact_manifest(OUTPUT_ROOT)
    print("PREPARE PASS: frozen parent artifacts and null mappings reproduced; no component statistic computed")


def _split_fit_job(
    replicate: int,
    row: int,
    column: int,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[int, int, int, float, float, int, int, float]:
    distance, fit = movement_distance(first, second)
    sectors = {
        value.final_determinant for value in fit.starts if value.converged
    }
    return (
        replicate,
        row,
        column,
        fit.objective,
        distance,
        fit.determinant,
        len(sectors),
        fit.gradient_norm,
    )


def _split_half_costs(
    split_z: np.ndarray, *, workers: int
) -> tuple[dict[str, np.ndarray], pd.DataFrame, float]:
    started = time.perf_counter()
    banks = split_replicate_banks(split_z)
    jobs = (
        delayed(_split_fit_job)(replicate, row, column, first[row], second[column])
        for replicate, (first, second) in enumerate(banks)
        for row in range(N_CELLS)
        for column in range(N_CELLS)
    )
    results = Parallel(n_jobs=workers, backend="loky", verbose=10)(jobs)
    c_full = np.empty((2, N_CELLS, N_CELLS), dtype=np.float64)
    diagnostic_rows: list[dict[str, Any]] = []
    for replicate, row, column, objective, distance, determinant, sector_count, gradient in results:
        if sector_count != 2:
            raise ComponentDecompositionNumericalError(
                "UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE: split fit lacks both O(22) sectors"
            )
        if not np.isclose(objective, distance**2, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE):
            raise ComponentDecompositionNumericalError(
                "UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE: split distance/objective mismatch"
            )
        c_full[replicate, row, column] = objective
        diagnostic_rows.append(
            {
                "replicate": ("A", "B")[replicate],
                "row_cell": row,
                "column_cell": column,
                "c_full": objective,
                "d_mov": distance,
                "selected_determinant": determinant,
                "converged_sector_count": sector_count,
                "selected_gradient_norm": gradient,
            }
        )
    costs = {key: np.empty_like(c_full) for key in COMPONENT_ORDER}
    for replicate, (first, second) in enumerate(banks):
        sensor, length, _, _ = sensor_and_length_squared_costs(first, second)
        costs["sensor"][replicate] = sensor
        costs["len"][replicate] = length
        costs["full"][replicate] = c_full[replicate]
        costs["ang"][replicate] = c_full[replicate] - length
        costs["ori"][replicate] = sensor - c_full[replicate]
    return costs, pd.DataFrame(diagnostic_rows), time.perf_counter() - started


def _normalize_checks(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    for column in (
        "maximum_absolute_error",
        "maximum_relative_error",
        "minimum_raw_value",
        "negative_raw_count",
        "meaningful_negative_count",
        "maximum_allowed_negative_tolerance",
    ):
        if column not in frame:
            frame[column] = np.nan
    return frame


def _pair_table(matrices: ComponentMatrices) -> pd.DataFrame:
    fractions = descriptive_fractions(matrices)
    rows: list[dict[str, Any]] = []
    for row in range(N_CELLS):
        row_subject = row // 4 + 1
        row_class = CLASS_ORDER[row % 4]
        for column in range(N_CELLS):
            column_subject = column // 4 + 1
            column_class = CLASS_ORDER[column % 4]
            if row_subject == column_subject and row_class == column_class:
                relation = "same_subject_same_class"
            elif row_subject == column_subject:
                relation = "same_subject_different_class"
            elif row_class == column_class:
                relation = "different_subject_same_class"
            else:
                relation = "different_subject_different_class"
            rows.append(
                {
                    "row_cell": row,
                    "row_subject": row_subject,
                    "row_class": row_class,
                    "column_cell": column,
                    "column_subject": column_subject,
                    "column_class": column_class,
                    "relation": relation,
                    "c_len": matrices.length[row, column],
                    "c_ang": matrices.angular[row, column],
                    "c_ori": matrices.orientation[row, column],
                    "c_full": matrices.full[row, column],
                    "c_sensor": matrices.sensor[row, column],
                    "fraction_valid": bool(fractions["valid"][row, column]),
                    "fraction_len": fractions["fraction_len"][row, column],
                    "fraction_ang": fractions["fraction_ang"][row, column],
                    "fraction_ori": fractions["fraction_ori"][row, column],
                }
            )
    return pd.DataFrame(rows)


def _relation_tables(statistics: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells: list[dict[str, Any]] = []
    subjects: list[dict[str, Any]] = []
    for component in COMPONENT_ORDER:
        value = statistics[component]
        for subject in range(9):
            subjects.append(
                {
                    "component": component,
                    "subject": subject + 1,
                    "S_s": value.s_s[subject],
                    "C_s": value.c_s[subject],
                    "J_s": value.j_s[subject],
                }
            )
            for class_index, class_label in enumerate(CLASS_ORDER):
                cells.append(
                    {
                        "component": component,
                        "subject": subject + 1,
                        "class": class_label,
                        "a_sc": value.a_sc[subject, class_index],
                        "b_sc": value.b_sc[subject, class_index],
                        "c_sc": value.c_sc[subject, class_index],
                        "d_sc": value.d_sc[subject, class_index],
                        "S_sc": value.s_sc[subject, class_index],
                        "C_sc": value.c_specific_sc[subject, class_index],
                        "J_sc": value.j_sc[subject, class_index],
                    }
                )
    return pd.DataFrame(cells), pd.DataFrame(subjects)


def _inference_table(inferences: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for component in COMPONENT_ORDER:
        value = inferences[component]
        rows.append(
            {
                "component": component,
                "T_subject": value.observed.t_subject,
                "p_subject": value.p_subject,
                "T_class": value.observed.t_class,
                "p_class": value.p_class,
                "T_J": value.observed.t_j,
                "p_J_subjectbreak": value.p_j_subjectbreak,
                "p_J_classbreak": value.p_j_classbreak,
            }
        )
    return pd.DataFrame(rows)


def _fixed_tables(
    full_z: np.ndarray,
    matrices: ComponentMatrices,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = full_z[0].reshape(N_CELLS, N_STEPS, N_CHANNELS, N_CHANNELS)
    second = full_z[1].reshape(N_CELLS, N_STEPS, N_CHANNELS, N_CHANNELS)
    fractions = descriptive_fractions(matrices)
    comparison_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    row = 0
    for label, column in FIXED_COMPARISONS:
        distance, fit = movement_distance(first[row], second[column])
        frozen = matrices.full[row, column]
        if not np.isclose(fit.objective, frozen, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE):
            raise ComponentDecompositionNumericalError(
                "UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE: "
                f"fixed-pair frozen quotient objective did not reproduce for {label}"
            )
        if _array_sha256(first[row]) <= _array_sha256(second[column]):
            target, source = first[row], second[column]
        else:
            target, source = second[column], first[row]
        transformed = np.einsum(
            "ij,kjl,ml->kim", fit.action, source, fit.action, optimize=True
        )
        step_full = 0.25 * np.sum((target - transformed) ** 2, axis=(1, 2))
        speed_target = np.linalg.norm(target, axis=(1, 2))
        speed_source = np.linalg.norm(source, axis=(1, 2))
        step_len = 0.25 * (speed_target - speed_source) ** 2
        step_ang = step_full - step_len
        if np.any(step_ang < -(ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.maximum(step_full, step_len))):
            raise ComponentDecompositionNumericalError(
                "UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE: fixed step angular term negative"
            )
        if not np.isclose(np.sum(step_ang), matrices.angular[row, column], atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE):
            raise ComponentDecompositionNumericalError(
                "UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE: fixed step angular reconstruction failed"
            )
        comparison_rows.append(
            {
                "comparison": label,
                "row_cell": row,
                "column_cell": column,
                "c_len": matrices.length[row, column],
                "c_ang": matrices.angular[row, column],
                "c_ori": matrices.orientation[row, column],
                "c_full": matrices.full[row, column],
                "c_sensor": matrices.sensor[row, column],
                "fraction_len": fractions["fraction_len"][row, column],
                "fraction_ang": fractions["fraction_ang"][row, column],
                "fraction_ori": fractions["fraction_ori"][row, column],
                "frozen_full_reproduction_absolute_error": abs(fit.objective - frozen),
                "reproduced_distance": distance,
            }
        )
        for step in range(N_STEPS):
            step_rows.append(
                {
                    "comparison": label,
                    "transition": f"{step + 1}->{step + 2}",
                    "c_len_step": step_len[step],
                    "c_ang_step": step_ang[step],
                    "c_full_step": step_full[step],
                }
            )
    return pd.DataFrame(comparison_rows), pd.DataFrame(step_rows)


def _distribution_rows(values: np.ndarray, *, value_name: str) -> pd.DataFrame:
    rows = []
    for step in range(N_STEPS):
        current = np.asarray(values[..., step], dtype=np.float64).reshape(-1)
        rows.append(
            {
                "transition": f"{step + 1}->{step + 2}",
                "value": value_name,
                "count": len(current),
                "mean": np.mean(current),
                "std": np.std(current, ddof=1),
                "minimum": np.min(current),
                "q25": np.quantile(current, 0.25),
                "median": np.median(current),
                "q75": np.quantile(current, 0.75),
                "maximum": np.max(current),
            }
        )
    return pd.DataFrame(rows)


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
        raise RuntimeError("scientific run requires a clean worktree")
    if _git("rev-parse", "HEAD") != protocol_freeze_sha:
        raise RuntimeError("HEAD must equal the supplied protocol-freeze SHA")
    test_path = OUTPUT_ROOT / "protocol" / "test_results.json"
    if not test_path.exists():
        raise RuntimeError("pre-execution test record is missing")
    implementation_sha, source_hashes = _implementation()
    pre_data = json.loads(
        (OUTPUT_ROOT / "protocol" / "pre_data_provenance.json").read_text(encoding="utf-8")
    )
    if implementation_sha != pre_data["implementation_source_sha256"]:
        raise RuntimeError("implementation changed after protocol preparation")
    parent_hashes_before = {
        relative: sha256_file(ROOT / relative) for relative in FROZEN_ARTIFACT_HASHES
    }
    try:
        reproduction = reproduce_frozen_inputs()
        with np.load(ROOT / FULL_RELATIVE_PATH, allow_pickle=False) as archive:
            full_z = archive["z"].copy()
        with np.load(ROOT / SPLIT_RELATIVE_PATH, allow_pickle=False) as archive:
            split_z = archive["z"].copy()
        with np.load(ROOT / MATRICES_RELATIVE_PATH, allow_pickle=False) as archive:
            d_mov = archive["d_mov"].copy()
            d_len = archive["d_len"].copy()
            d_direct = archive["d_direct"].copy()
        with np.load(ROOT / NULLS_RELATIVE_PATH, allow_pickle=False) as archive:
            saved_subject_mappings = archive["subject_mappings"].copy()
            saved_class_mappings = archive["class_mappings"].copy()

        session0 = full_z[0].reshape(N_CELLS, N_STEPS, N_CHANNELS, N_CHANNELS)
        session1 = full_z[1].reshape(N_CELLS, N_STEPS, N_CHANNELS, N_CHANNELS)
        independent_sensor, independent_length, speed0, speed1 = sensor_and_length_squared_costs(
            session0, session1
        )
        matrices = ComponentMatrices(
            sensor=np.square(d_direct),
            full=np.square(d_mov),
            length=np.square(d_len),
            angular=np.square(d_mov) - np.square(d_len),
            orientation=np.square(d_direct) - np.square(d_mov),
        )
        numerical_records: list[dict[str, Any]] = pairwise_numerical_gates(
            matrices,
            independently_computed_sensor=independent_sensor,
            independently_computed_length=independent_length,
            d_mov=d_mov,
            d_len=d_len,
            d_direct=d_direct,
        )
        print("CHECKPOINT SCIENCE: full 1,296-pair squared-cost decomposition observed; settings immutable", flush=True)
        statistics = {key: relation_statistics(value) for key, value in matrices.as_dict().items()}
        numerical_records.extend(statistic_reconstruction_gates(statistics))
        inferences = evaluate_all_components(
            matrices.as_dict(),
            subject_mappings=saved_subject_mappings,
            class_mappings=saved_class_mappings,
        )
        numerical_records.extend(null_reconstruction_gates(inferences))

        split_costs, split_optimizer, split_seconds = _split_half_costs(
            split_z, workers=int(config["optimizer"]["parallel_workers"])
        )
        split_statistics: dict[str, list[Any]] = {key: [] for key in COMPONENT_ORDER}
        for replicate in range(2):
            split_matrices = ComponentMatrices(
                sensor=split_costs["sensor"][replicate],
                full=split_costs["full"][replicate],
                length=split_costs["len"][replicate],
                angular=split_costs["ang"][replicate],
                orientation=split_costs["ori"][replicate],
            )
            prefix = f"split_{('A', 'B')[replicate]}_"
            split_records = pairwise_numerical_gates(
                split_matrices,
                independently_computed_sensor=split_costs["sensor"][replicate],
                independently_computed_length=split_costs["len"][replicate],
                d_mov=np.sqrt(split_costs["full"][replicate]),
                d_len=np.sqrt(split_costs["len"][replicate]),
                d_direct=np.sqrt(split_costs["sensor"][replicate]),
            )
            for record in split_records:
                record["check"] = prefix + str(record["check"])
            numerical_records.extend(split_records)
            current = {
                key: relation_statistics(split_costs[key][replicate]) for key in COMPONENT_ORDER
            }
            split_stat_records = statistic_reconstruction_gates(current)
            for record in split_stat_records:
                record["check"] = prefix + str(record["check"])
            numerical_records.extend(split_stat_records)
            for key in COMPONENT_ORDER:
                split_statistics[key].append(current[key])

        t_half = np.asarray([value.t_j for value in split_statistics["ang"]])
        split_sign_stable = bool(np.all(t_half > 0.0))
        terminal = terminal_decision(
            t_j_ang=inferences["ang"].observed.t_j,
            p_j_ang_subjectbreak=inferences["ang"].p_j_subjectbreak,
            p_j_ang_classbreak=inferences["ang"].p_j_classbreak,
            t_j_ori=inferences["ori"].observed.t_j,
            p_j_ori_subjectbreak=inferences["ori"].p_j_subjectbreak,
            p_j_ori_classbreak=inferences["ori"].p_j_classbreak,
            t_j_len=inferences["len"].observed.t_j,
            p_j_len_subjectbreak=inferences["len"].p_j_subjectbreak,
            p_j_len_classbreak=inferences["len"].p_j_classbreak,
            split_half_ang_sign_stable=split_sign_stable,
        )

        pair_table = _pair_table(matrices)
        relation_cells, subject_stats = _relation_tables(statistics)
        inference_summary = _inference_table(inferences)
        fixed_comparisons, fixed_step_angular = _fixed_tables(full_z, matrices)
        all_speeds = np.linalg.norm(full_z, axis=(4, 5))
        step_norm_summary = _distribution_rows(all_speeds, value_name="step_norm")
        length_steps = 0.25 * (speed0[:, None, :] - speed1[None, :, :]) ** 2
        step_cost_summary = _distribution_rows(length_steps, value_name="c_len_step")
        split_stability = pd.DataFrame(
            [
                {
                    "replicate": ("A", "B")[replicate],
                    "T_J_ang": split_statistics["ang"][replicate].t_j,
                    "J_s_ang": "; ".join(
                        f"S{index + 1}={value:.10f}"
                        for index, value in enumerate(split_statistics["ang"][replicate].j_s)
                    ),
                    "replicate_positive": bool(split_statistics["ang"][replicate].t_j > 0.0),
                    "sign_stable": split_sign_stable,
                }
                for replicate in range(2)
            ]
        )
        numerical_checks = _normalize_checks(numerical_records)
        fraction_summary = (
            pair_table[pair_table["fraction_valid"]]
            .groupby("relation", sort=False)[["fraction_len", "fraction_ang", "fraction_ori"]]
            .agg(["count", "mean", "median", "min", "max"])
        )
        fraction_summary.columns = ["_".join(column) for column in fraction_summary.columns]
        fraction_summary = fraction_summary.reset_index()
        labels = [f"S{s}-{c}" for s, c in zip(canonical_cell_subjects(), canonical_cell_classes(), strict=True)]

        arrays_dir = OUTPUT_ROOT / "arrays"
        tables_dir = OUTPUT_ROOT / "tables"
        figures_dir = OUTPUT_ROOT / "figures"
        decisions_dir = OUTPUT_ROOT / "decisions"
        for directory in (arrays_dir, tables_dir, figures_dir, decisions_dir):
            directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_dir / "component_cost_matrices.npz",
            c_sensor_matrix=matrices.sensor,
            c_full_matrix=matrices.full,
            c_len_matrix=matrices.length,
            c_ang_matrix=matrices.angular,
            c_ori_matrix=matrices.orientation,
            cell_subjects=canonical_cell_subjects(),
            cell_classes=canonical_cell_classes(),
        )
        np.savez_compressed(
            arrays_dir / "component_nulls.npz",
            **{
                f"{component}_{field}": getattr(inferences[component], field)
                for component in COMPONENT_ORDER
                for field in (
                    "subjectbreak_t_subject",
                    "subjectbreak_t_j",
                    "classbreak_t_class",
                    "classbreak_t_j",
                )
            },
            subject_mappings=saved_subject_mappings,
            class_mappings=saved_class_mappings,
        )
        np.savez_compressed(
            arrays_dir / "split_half_component_matrices.npz",
            c_sensor_matrix=split_costs["sensor"],
            c_full_matrix=split_costs["full"],
            c_len_matrix=split_costs["len"],
            c_ang_matrix=split_costs["ang"],
            c_ori_matrix=split_costs["ori"],
            replicates=np.asarray(("A", "B")),
            cell_subjects=canonical_cell_subjects(),
            cell_classes=canonical_cell_classes(),
        )
        pair_table.to_csv(tables_dir / "component_pair_costs.csv", index=False)
        relation_cells.to_csv(tables_dir / "component_relation_cells.csv", index=False)
        subject_stats.to_csv(tables_dir / "component_subject_stats.csv", index=False)
        inference_summary.to_csv(tables_dir / "component_inference_summary.csv", index=False)
        fixed_comparisons.to_csv(tables_dir / "fixed_component_comparisons.csv", index=False)
        split_stability.to_csv(tables_dir / "split_half_component_stability.csv", index=False)
        numerical_checks.to_csv(tables_dir / "numerical_reconstruction_checks.csv", index=False)
        split_optimizer.to_csv(tables_dir / "split_half_optimizer_diagnostics.csv", index=False)
        fraction_summary.to_csv(tables_dir / "component_relation_fraction_summary.csv", index=False)
        step_norm_summary.to_csv(tables_dir / "component_step_norm_summary.csv", index=False)
        step_cost_summary.to_csv(tables_dir / "component_step_cost_summary.csv", index=False)
        fixed_step_angular.to_csv(tables_dir / "fixed_step_angular_diagnostics.csv", index=False)
        for component, matrix in matrices.as_dict().items():
            write_matrix_csv(tables_dir / f"c_{component}_matrix.csv", matrix, labels)
        null_payload = {
            "same_indexed_mappings_all_components": True,
            "subject_seed_sequence": [20260810, 1102],
            "class_seed_sequence": [20260810, 1101],
            "replicates": 1999,
        }
        write_json(OUTPUT_ROOT / "protocol" / "null_mapping_certificate.json", null_payload)
        runtime = {
            "total_seconds": time.perf_counter() - total_started,
            "split_half_quotient_seconds": split_seconds,
            "split_half_fit_count": 2 * N_CELLS * N_CELLS,
            "fixed_full_fit_reproduction_count": len(FIXED_COMPARISONS),
        }
        write_json(OUTPUT_ROOT / "protocol" / "scientific_runtime.json", runtime)
        write_json(
            OUTPUT_ROOT / "protocol" / "scientific_provenance.json",
            {
                "branch": EXPECTED_BRANCH,
                "protocol_freeze_sha": protocol_freeze_sha,
                "parent_sha": PARENT_SHA,
                "parent_protocol_freeze_sha": PARENT_PROTOCOL_SHA,
                "parent_scientific_result_sha": PARENT_RESULT_SHA,
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "config_sha256": sha256_file(CONFIG_PATH),
                "implementation_source_sha256": implementation_sha,
                "implementation_file_sha256": source_hashes,
                "git_status_at_start": "clean",
                "head_at_start": protocol_freeze_sha,
                "first_result_checkpoint": "after complete full squared-cost decomposition",
                "full_data_q_refit_count": len(FIXED_COMPARISONS),
                "full_data_q_refit_scope": "fixed diagnostic objective reproduction only",
                "scientific_setting_changed_after_result_access": False,
            },
        )
        write_json(
            decisions_dir / "terminal_decision.json",
            {
                "terminal": terminal,
                "T_J_len": inferences["len"].observed.t_j,
                "p_J_len_subjectbreak": inferences["len"].p_j_subjectbreak,
                "p_J_len_classbreak": inferences["len"].p_j_classbreak,
                "T_J_ang": inferences["ang"].observed.t_j,
                "p_J_ang_subjectbreak": inferences["ang"].p_j_subjectbreak,
                "p_J_ang_classbreak": inferences["ang"].p_j_classbreak,
                "T_J_ori": inferences["ori"].observed.t_j,
                "p_J_ori_subjectbreak": inferences["ori"].p_j_subjectbreak,
                "p_J_ori_classbreak": inferences["ori"].p_j_classbreak,
                "split_half_T_J_ang_A": t_half[0],
                "split_half_T_J_ang_B": t_half[1],
                "split_half_ang_sign_stable": split_sign_stable,
            },
        )
        write_figures(
            output_root=OUTPUT_ROOT,
            relation_cells=relation_cells,
            subject_stats=subject_stats,
            fixed_comparisons=fixed_comparisons,
        )
        tests = json.loads(test_path.read_text(encoding="utf-8"))
        write_report(
            REPORT_PATH,
            branch=EXPECTED_BRANCH,
            parent_sha=PARENT_SHA,
            parent_protocol_sha=PARENT_PROTOCOL_SHA,
            parent_result_sha=PARENT_RESULT_SHA,
            protocol_freeze_sha=protocol_freeze_sha,
            reproduction=reproduction,
            numerical_checks=numerical_checks,
            inferences=inferences,
            split_stability=split_stability,
            fixed_comparisons=fixed_comparisons,
            step_norm_summary=step_norm_summary,
            step_cost_summary=step_cost_summary,
            fixed_step_angular=fixed_step_angular,
            terminal=terminal,
            runtime=runtime,
            tests=tests,
        )
        parent_hashes_after = {
            relative: sha256_file(ROOT / relative) for relative in FROZEN_ARTIFACT_HASHES
        }
        if parent_hashes_after != parent_hashes_before:
            raise FrozenInputReproductionError("parent artifacts mutated during scientific execution")
        write_json(
            OUTPUT_ROOT / "protocol" / "parent_artifact_immutability.json",
            {
                "status": "PASS",
                "unchanged": True,
                "hashes_before": parent_hashes_before,
                "hashes_after": parent_hashes_after,
            },
        )
        write_artifact_manifest(OUTPUT_ROOT)
        print(
            f"CHECKPOINT RESULT: {terminal}; T_J_len={inferences['len'].observed.t_j:.10f}; "
            f"T_J_ang={inferences['ang'].observed.t_j:.10f}; T_J_ori={inferences['ori'].observed.t_j:.10f}",
            flush=True,
        )
    except FrozenInputReproductionError as error:
        _write_failure("UNASSESSED_COMPONENT_DECOMPOSITION_TECHNICAL_FAILURE", error)
        raise
    except ComponentDecompositionNumericalError as error:
        _write_failure("UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE", error)
        raise
    except Exception as error:
        _write_failure("UNASSESSED_COMPONENT_DECOMPOSITION_TECHNICAL_FAILURE", error)
        raise


def record_post_tests(*, focused: str, full: str) -> None:
    test_path = OUTPUT_ROOT / "protocol" / "test_results.json"
    values = json.loads(test_path.read_text(encoding="utf-8"))
    values["focused_after"] = focused
    values["full_after"] = full
    values["status"] = "PASS"
    write_json(test_path, values)
    report = REPORT_PATH.read_text(encoding="utf-8")
    if report.count("`PENDING`") != 2:
        raise RuntimeError("expected exactly two post-test placeholders in report")
    report = report.replace("`PENDING`", f"`{focused}`", 1)
    report = report.replace("`PENDING`", f"`{full}`", 1)
    REPORT_PATH.write_text(report, encoding="utf-8")
    write_artifact_manifest(OUTPUT_ROOT)


def finalize(scientific_result_sha: str) -> None:
    _load_config()
    if _git("status", "--porcelain"):
        raise RuntimeError("finalize requires a clean scientific-result commit")
    if _git("rev-parse", "HEAD") != scientific_result_sha:
        raise RuntimeError("HEAD must equal the supplied scientific-result SHA")
    scientific = json.loads(
        (OUTPUT_ROOT / "protocol" / "scientific_provenance.json").read_text(encoding="utf-8")
    )
    protocol_freeze_sha = str(scientific["protocol_freeze_sha"])
    implementation_sha, source_hashes = _implementation()
    if implementation_sha != scientific["implementation_source_sha256"]:
        raise RuntimeError("implementation changed after scientific execution")
    changed = [
        value
        for value in _git("diff", "--name-only", protocol_freeze_sha, scientific_result_sha).splitlines()
        if value
    ]
    output_prefix = str(OUTPUT_ROOT.relative_to(ROOT)) + "/"
    if not changed or any(not value.startswith(output_prefix) for value in changed):
        raise RuntimeError("scientific result changed files outside the frozen output namespace")
    report = REPORT_PATH.read_text(encoding="utf-8")
    placeholder = "FINAL_RESULT_SHA_PENDING"
    if report.count(placeholder) != 1:
        raise RuntimeError("scientific-result placeholder missing or duplicated")
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
    tests_parser = subparsers.add_parser("record-post-tests")
    tests_parser.add_argument("--focused", required=True)
    tests_parser.add_argument("--full", required=True)
    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--scientific-result-sha", required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare()
    elif arguments.command == "run":
        run(arguments.protocol_freeze_sha)
    elif arguments.command == "record-post-tests":
        record_post_tests(focused=arguments.focused, full=arguments.full)
    else:
        finalize(arguments.scientific_result_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
