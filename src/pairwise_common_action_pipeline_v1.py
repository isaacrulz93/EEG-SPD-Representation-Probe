"""Execution pipeline for the frozen pairwise common-action amendment."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pymanopt
import pyriemann
import scipy
import yaml

from src.common_action_formulation_audit_v1 import (
    equivalence_aware_cycle_diagnostic,
)
from src.common_action_pipeline_v0 import (
    CommonActionPipelineError,
    atomic_json,
    atomic_npz,
    load_and_reproduce_U,
    sha256_file,
)
from src.common_action_solver_v0 import (
    ActionSolverError,
    classify_prediction_matrices,
    conjugate,
    diagnose_multistart,
    nonidentity_permutations_three,
)
from src.pairwise_common_action_v1 import (
    NULL_REPLICATES,
    PAIRWISE_SETTINGS,
    PairwiseContractError,
    aggregate_pairwise_gains,
    assess_pairwise_prediction,
    evaluate_stage_gate,
    fit_pairwise_action,
    pairwise_error_gain,
    semantic_null_statistics,
    terminal_decision,
    unrelated_target_gain_bank,
    unrelated_target_null_statistics,
)


CONFIG_PATH = "configs/bnci2014_001_pairwise_common_action_v1.yaml"
PROTOCOL_PATH = "docs/PROTOCOL_AMENDMENT_PAIRWISE_COMMON_ACTION_V1.md"
# Filled after the protocol/config are final and before the amendment commit.
EXPECTED_CONFIG_SHA256 = "0c5d7b01fccab0d103653c5a178243db1021a0f273686939b48e661f65431b26"
EXPECTED_PROTOCOL_SHA256 = "48878fdbc75a9fe02d102317142fd6f58202e161271a0db09e3a371b4b9f4519"
OUTPUT_PROTOCOL_FILENAME = "PROTOCOL_AMENDMENT_PAIRWISE_COMMON_ACTION_V1.md"
REPORT_TITLE = "Pairwise Common Action V1"
REPORT_FILENAME = "pairwise_common_action_v1.md"
CHECKPOINT_IDENTITY: dict[str, str] | None = None
CLASSES = ("left_hand", "right_hand", "feet", "tongue")
SESSIONS = ("0train", "1test")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def assert_run_worktree(root: Path, output_relative: str) -> None:
    """Require a frozen tree, while allowing only exact resumable run output."""

    status = _git(root, "status", "--porcelain")
    if not status:
        return
    allowed_prefix = f"outputs/{Path(output_relative).name}/"
    disallowed = []
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path == allowed_prefix.rstrip("/") or path.startswith(allowed_prefix):
            continue
        disallowed.append(line)
    if disallowed:
        raise PairwiseContractError(
            "run requires a frozen worktree; disallowed status: "
            + " | ".join(disallowed)
        )


def load_config(root: Path) -> dict[str, Any]:
    config_path = root / CONFIG_PATH
    protocol_path = root / PROTOCOL_PATH
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise PairwiseContractError("frozen pairwise config content hash mismatch")
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise PairwiseContractError("frozen pairwise protocol content hash mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    optimizer = config["optimizer"]
    line_search = optimizer["line_search"]
    expected_optimizer = {
        "total_starts": PAIRWISE_SETTINGS.starts,
        "max_iterations": PAIRWISE_SETTINGS.max_iterations,
        "min_gradient_norm": PAIRWISE_SETTINGS.gradient_tolerance,
        "min_step_size": PAIRWISE_SETTINGS.optimizer_min_step_size,
        "max_cost_evaluations": PAIRWISE_SETTINGS.max_iterations + 1,
        "max_time_seconds": PAIRWISE_SETTINGS.optimizer_max_time_seconds,
        "log_verbosity": PAIRWISE_SETTINGS.pymanopt_log_verbosity,
    }
    for key, expected in expected_optimizer.items():
        if optimizer[key] != expected:
            raise PairwiseContractError(f"runtime/config optimizer mismatch: {key}")
    secondary = optimizer["secondary_convergence"]
    if (
        secondary["relative_objective_change_max"]
        != PAIRWISE_SETTINGS.objective_tolerance
        or secondary["gradient_norm_multiplier_max"]
        != PAIRWISE_SETTINGS.objective_stall_gradient_multiplier
    ):
        raise PairwiseContractError("runtime/config secondary-convergence mismatch")
    expected_line_search = {
        "contraction_factor": PAIRWISE_SETTINGS.line_search_contraction_factor,
        "optimism": PAIRWISE_SETTINGS.line_search_optimism,
        "sufficient_decrease": PAIRWISE_SETTINGS.line_search_sufficient_decrease,
        "max_iterations": PAIRWISE_SETTINGS.line_search_max_iterations,
        "initial_step_size": PAIRWISE_SETTINGS.line_search_initial_step_size,
    }
    for key, expected in expected_line_search.items():
        if line_search[key] != expected:
            raise PairwiseContractError(f"runtime/config line-search mismatch: {key}")
    if config["nulls"]["replicates"] != NULL_REPLICATES:
        raise PairwiseContractError("runtime/config null-replicate mismatch")
    return config


def prepare_output(root: Path, config: Mapping[str, Any]) -> Path:
    output = root / str(config["project"]["output_dir"])
    for relative in ("protocol", "objects", "tables", "nulls", "checkpoints", "decisions", "report"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / CONFIG_PATH, output / "protocol/frozen_config.yaml")
    shutil.copy2(root / PROTOCOL_PATH, output / f"protocol/{OUTPUT_PROTOCOL_FILENAME}")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pymanopt": pymanopt.__version__,
        "pyriemann": pyriemann.__version__,
        "freeze_commit": _git(root, "rev-parse", "HEAD"),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }
    atomic_json(output / "protocol/environment.json", environment)
    atomic_json(
        output / "protocol/provenance.json",
        {
            "branch": _git(root, "branch", "--show-current"),
            "head_at_run_start": _git(root, "rev-parse", "HEAD"),
            "audit_commit": config["protocol"]["audit_commit"],
            "original_freeze_commit": config["protocol"]["original_freeze_commit"],
            "new_bnci_scientific_statistic_before_freeze": False,
            "old_nested_generalized_procrustes_used": False,
            "checkpoint_identity": CHECKPOINT_IDENTITY,
        },
    )
    return output


def _solver_rows(
    result: Any,
    *,
    stage: str,
    split: str,
    target: int,
    source: int,
    session: int,
    heldout: int,
    semantic_permutation: int = -1,
) -> list[dict[str, Any]]:
    rows = []
    equivalent = set(result.fit.equivalent_start_indices)
    for index, start in enumerate(result.fit.starts):
        rows.append(
            {
                "stage": stage,
                "split": split,
                "target": target + 1,
                "source": source + 1,
                "session_or_direction_index": session,
                "heldout_class": CLASSES[heldout],
                "semantic_permutation": semantic_permutation,
                "start_index": index,
                "initial_determinant": result.initial_determinants[index],
                "final_determinant": start.determinant,
                "objective": start.objective,
                "gradient_norm": start.gradient_norm,
                "iterations": start.iterations,
                "converged": start.converged,
                "near_optimal": index in equivalent,
                "stopping_criterion": start.stopping_criterion,
                "actual_total_starts": len(result.fit.starts),
            }
        )
    return rows


def _fit_true_pair(
    arrays: Mapping[str, np.ndarray],
    *,
    target: int,
    source: int,
    session: int,
    heldout: int,
) -> dict[str, Any]:
    fit_classes = tuple(index for index in range(4) if index != heldout)
    fits: dict[str, Any] = {}
    solver_rows: list[dict[str, Any]] = []
    for split in ("F", "A", "B"):
        U = arrays[split]
        result = fit_pairwise_action(
            U[target, session, fit_classes],
            U[source, session, fit_classes],
            seed_parts=("A", split, target, source, session, heldout, "true"),
        )
        fits[split] = result
        solver_rows.extend(
            _solver_rows(
                result,
                stage="A",
                split=split,
                target=target,
                source=source,
                session=session,
                heldout=heldout,
            )
        )
    prediction_a = conjugate(
        fits["A"].fit.matrix, arrays["A"][source, session, heldout]
    )
    prediction_b = conjugate(
        fits["B"].fit.matrix, arrays["B"][source, session, heldout]
    )
    assessment = assess_pairwise_prediction(
        fits["F"],
        arrays["F"][target, session, fit_classes],
        arrays["F"][source, session, fit_classes],
        arrays["F"][source, session, heldout],
        split_half_prediction_a=prediction_a,
        split_half_prediction_b=prediction_b,
    )
    best_prediction = conjugate(
        fits["F"].fit.matrix, arrays["F"][source, session, heldout]
    )
    raw_error, action_error, gain = pairwise_error_gain(
        arrays["F"][target, session, heldout],
        arrays["F"][source, session, heldout],
        best_prediction,
    )
    half_a_raw, half_a_action, half_a_gain = pairwise_error_gain(
        arrays["A"][target, session, heldout],
        arrays["A"][source, session, heldout],
        prediction_a,
    )
    half_b_raw, half_b_action, half_b_gain = pairwise_error_gain(
        arrays["B"][target, session, heldout],
        arrays["B"][source, session, heldout],
        prediction_b,
    )
    ident = assessment.identifiability
    stabilizer = assessment.stabilizer
    row = {
        "target": target + 1,
        "source": source + 1,
        "session": SESSIONS[session],
        "session_index": session,
        "heldout_class": CLASSES[heldout],
        "heldout_index": heldout,
        "raw_error": raw_error,
        "action_error": action_error,
        "gain": gain,
        "half_A_raw_error": half_a_raw,
        "half_A_action_error": half_a_action,
        "half_A_gain": half_a_gain,
        "half_B_raw_error": half_b_raw,
        "half_B_action_error": half_b_action,
        "half_B_gain": half_b_gain,
        "D_eq": ident.maximum_relative_prediction_dispersion,
        "D_split": ident.split_half_relative_variability,
        "D_threshold": ident.materiality_threshold,
        "prediction_normalization": ident.prediction_normalization,
        "equivalent_prediction_count": ident.equivalent_solution_count,
        "identifiability": ident.classification,
        "best_objective": fits["F"].fit.starts[fits["F"].fit.best_start_index].objective,
        "best_gradient_norm": fits["F"].fit.starts[fits["F"].fit.best_start_index].gradient_norm,
        "best_determinant": fits["F"].fit.starts[fits["F"].fit.best_start_index].determinant,
    }
    stabilizer_row = {
        "target": target + 1,
        "source": source + 1,
        "session": SESSIONS[session],
        "heldout_class": CLASSES[heldout],
        "numerical_nullity": stabilizer.numerical_nullity,
        "approximate_nullity": stabilizer.approximate_nullity,
        "numerical_tolerance": stabilizer.numerical_tolerance,
        "approximate_tolerance": stabilizer.approximate_tolerance,
        "largest_singular_value": float(stabilizer.singular_values[0]),
        "smallest_singular_value": float(stabilizer.singular_values[-1]),
    }
    return {
        "row": row,
        "solver_rows": solver_rows,
        "stabilizer_row": stabilizer_row,
        "stabilizer_singular_values": stabilizer.singular_values,
        "best_action": fits["F"].fit.matrix,
        "half_A_action": fits["A"].fit.matrix,
        "half_B_action": fits["B"].fit.matrix,
        "equivalent_actions": assessment.equivalent_actions,
    }


_WORKER_ARRAYS: dict[str, np.ndarray] | None = None


def _initialize_worker(arrays: Mapping[str, np.ndarray]) -> None:
    global _WORKER_ARRAYS
    _WORKER_ARRAYS = {key: np.asarray(value) for key, value in arrays.items()}


def _primary_target_task(target: int) -> dict[str, Any]:
    if _WORKER_ARRAYS is None:
        raise RuntimeError("pairwise worker arrays are uninitialized")
    rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    stabilizer_rows: list[dict[str, Any]] = []
    singular_values: list[np.ndarray] = []
    eq_actions: list[np.ndarray] = []
    eq_offsets = [0]
    eq_keys: list[tuple[int, int, int, int]] = []
    d = _WORKER_ARRAYS["F"].shape[-1]
    best_actions = np.full((9, 2, 4, d, d), np.nan)
    half_a_actions = np.full_like(best_actions, np.nan)
    half_b_actions = np.full_like(best_actions, np.nan)
    for source in range(9):
        if source == target:
            continue
        for session in range(2):
            for heldout in range(4):
                result = _fit_true_pair(
                    _WORKER_ARRAYS,
                    target=target,
                    source=source,
                    session=session,
                    heldout=heldout,
                )
                rows.append(result["row"])
                solver_rows.extend(result["solver_rows"])
                stabilizer_rows.append(result["stabilizer_row"])
                singular_values.append(result["stabilizer_singular_values"])
                best_actions[source, session, heldout] = result["best_action"]
                half_a_actions[source, session, heldout] = result["half_A_action"]
                half_b_actions[source, session, heldout] = result["half_B_action"]
                eq_keys.append((target, source, session, heldout))
                eq_actions.extend(result["equivalent_actions"])
                eq_offsets.append(len(eq_actions))
    return {
        "target": target,
        "rows": rows,
        "solver_rows": solver_rows,
        "stabilizer_rows": stabilizer_rows,
        "singular_values": np.stack(singular_values),
        "best_actions": best_actions,
        "half_A_actions": half_a_actions,
        "half_B_actions": half_b_actions,
        "eq_keys": eq_keys,
        "eq_offsets": np.asarray(eq_offsets, dtype=np.int64),
        "eq_actions": np.asarray(eq_actions, dtype=np.float64),
    }


def _checkpoint_path(output: Path, kind: str, target: int) -> Path:
    return output / f"checkpoints/{kind}_target_{target + 1:02d}.npz"


def _save_task(path: Path, result: Mapping[str, Any]) -> None:
    atomic_npz(
        path,
        target=np.asarray(result["target"], dtype=np.int64),
        rows_json=np.asarray(json.dumps(result["rows"], sort_keys=True)),
        solver_rows_json=np.asarray(json.dumps(result["solver_rows"], sort_keys=True)),
        stabilizer_rows_json=np.asarray(json.dumps(result.get("stabilizer_rows", []), sort_keys=True)),
        singular_values=np.asarray(result.get("singular_values", np.empty((0, 231)))),
        best_actions=np.asarray(result.get("best_actions", np.empty(0))),
        half_A_actions=np.asarray(result.get("half_A_actions", np.empty(0))),
        half_B_actions=np.asarray(result.get("half_B_actions", np.empty(0))),
        eq_keys=np.asarray(result.get("eq_keys", []), dtype=np.int64),
        eq_offsets=np.asarray(result.get("eq_offsets", []), dtype=np.int64),
        eq_actions=np.asarray(result.get("eq_actions", np.empty(0))),
        mismatch_gains=np.asarray(result.get("mismatch_gains", np.empty(0))),
        mismatch_actions=np.asarray(result.get("mismatch_actions", np.empty(0))),
        checkpoint_identity_json=np.asarray(
            json.dumps(CHECKPOINT_IDENTITY or {}, sort_keys=True)
        ),
    )


def _load_task(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        stored_identity = (
            json.loads(str(archive["checkpoint_identity_json"].item()))
            if "checkpoint_identity_json" in archive.files
            else {}
        )
        if CHECKPOINT_IDENTITY is not None and stored_identity != CHECKPOINT_IDENTITY:
            raise PairwiseContractError(
                "checkpoint provenance mismatch; V2 caches cannot be reused across "
                "config/source/optimizer/amendment identities"
            )
        return {
            "target": int(archive["target"]),
            "rows": json.loads(str(archive["rows_json"].item())),
            "solver_rows": json.loads(str(archive["solver_rows_json"].item())),
            "stabilizer_rows": json.loads(str(archive["stabilizer_rows_json"].item())),
            "singular_values": np.asarray(archive["singular_values"]),
            "best_actions": np.asarray(archive["best_actions"]),
            "half_A_actions": np.asarray(archive["half_A_actions"]),
            "half_B_actions": np.asarray(archive["half_B_actions"]),
            "eq_keys": [tuple(map(int, value)) for value in np.asarray(archive["eq_keys"])],
            "eq_offsets": np.asarray(archive["eq_offsets"], dtype=np.int64),
            "eq_actions": np.asarray(archive["eq_actions"]),
            "mismatch_gains": np.asarray(archive["mismatch_gains"]),
            "mismatch_actions": np.asarray(archive["mismatch_actions"]),
        }


def _run_target_tasks(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    kind: str,
    workers: int,
) -> list[dict[str, Any]]:
    function = _primary_target_task if kind == "primary" else _semantic_target_task
    results: dict[int, dict[str, Any]] = {}
    pending = []
    for target in range(9):
        path = _checkpoint_path(output, kind, target)
        if path.exists():
            results[target] = _load_task(path)
        else:
            pending.append(target)
    if pending:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(pending)),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(arrays,),
        ) as executor:
            futures = {executor.submit(function, target): target for target in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                target = futures[future]
                result = future.result()
                _save_task(_checkpoint_path(output, kind, target), result)
                results[target] = result
                print(f"{kind} target task {completed}/{len(pending)} complete: subject {target + 1}", flush=True)
    if set(results) != set(range(9)):
        raise PairwiseContractError("required target-task grid is incomplete")
    return [results[target] for target in range(9)]


def _combine_primary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for task in tasks for row in task["rows"]]
    if len(rows) != 576:
        raise PairwiseContractError("primary pair grid must contain 576 rows")
    table = pd.DataFrame(rows).sort_values(
        ["target", "source", "session_index", "heldout_index"]
    ).reset_index(drop=True)
    if table[["target", "source", "session_index", "heldout_index"]].duplicated().any():
        raise PairwiseContractError("duplicate primary pair cell")
    d = tasks[0]["best_actions"].shape[-1]
    best = np.full((9, 9, 2, 4, d, d), np.nan)
    half_a = np.full_like(best, np.nan)
    half_b = np.full_like(best, np.nan)
    eq_bank: dict[tuple[int, int, int, int], np.ndarray] = {}
    for task in tasks:
        target = task["target"]
        best[target] = task["best_actions"]
        half_a[target] = task["half_A_actions"]
        half_b[target] = task["half_B_actions"]
        for index, key in enumerate(task["eq_keys"]):
            left, right = task["eq_offsets"][index : index + 2]
            eq_bank[key] = task["eq_actions"][left:right]
    return {
        "table": table,
        "solver_rows": [row for task in tasks for row in task["solver_rows"]],
        "stabilizer_rows": [row for task in tasks for row in task["stabilizer_rows"]],
        "singular_values": np.concatenate([task["singular_values"] for task in tasks]),
        "best_actions": best,
        "half_A_actions": half_a,
        "half_B_actions": half_b,
        "eq_bank": eq_bank,
    }


def _gain_tensor(table: pd.DataFrame) -> np.ndarray:
    values = np.full((9, 9, 2, 4), np.nan)
    for row in table.itertuples(index=False):
        values[int(row.target) - 1, int(row.source) - 1, int(row.session_index), int(row.heldout_index)] = float(row.gain)
    return values


def _summary_tables(gains: np.ndarray, *, session_labels: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    target_cells, subject_scores, statistic = aggregate_pairwise_gains(gains)
    cell_rows = []
    for target in range(9):
        for session in range(2):
            for heldout in range(4):
                cell_rows.append(
                    {
                        "target": target + 1,
                        "session_or_direction": session_labels[session],
                        "heldout_class": CLASSES[heldout],
                        "median_over_8_sources_gain": target_cells[target, session, heldout],
                    }
                )
    subject = pd.DataFrame(
        {"target": np.arange(1, 10), "subject_median_gain": subject_scores}
    )
    return pd.DataFrame(cell_rows), subject, statistic


def _semantic_target_task(target: int) -> dict[str, Any]:
    if _WORKER_ARRAYS is None:
        raise RuntimeError("pairwise worker arrays are uninitialized")
    U = _WORKER_ARRAYS["F"]
    d = U.shape[-1]
    gains = np.full((9, 2, 4, 5), np.nan)
    actions = np.full((9, 2, 4, 5, d, d), np.nan)
    rows = []
    solver_rows = []
    permutations = nonidentity_permutations_three()
    for source in range(9):
        if source == target:
            continue
        for session in range(2):
            for heldout in range(4):
                fit_classes = tuple(index for index in range(4) if index != heldout)
                raw_target = U[target, session, heldout]
                raw_source = U[source, session, heldout]
                for permutation_index, permutation in enumerate(permutations):
                    result = fit_pairwise_action(
                        U[target, session, fit_classes][np.asarray(permutation)],
                        U[source, session, fit_classes],
                        seed_parts=(
                            "A",
                            "F",
                            target,
                            source,
                            session,
                            heldout,
                            "semantic",
                            permutation_index,
                        ),
                    )
                    prediction = conjugate(result.fit.matrix, raw_source)
                    raw_error, action_error, gain = pairwise_error_gain(
                        raw_target, raw_source, prediction
                    )
                    gains[source, session, heldout, permutation_index] = gain
                    actions[source, session, heldout, permutation_index] = result.fit.matrix
                    rows.append(
                        {
                            "target": target + 1,
                            "source": source + 1,
                            "session": SESSIONS[session],
                            "session_index": session,
                            "heldout_class": CLASSES[heldout],
                            "heldout_index": heldout,
                            "permutation_index": permutation_index,
                            "permutation": "|".join(map(str, permutation)),
                            "raw_error": raw_error,
                            "action_error": action_error,
                            "gain": gain,
                        }
                    )
                    solver_rows.extend(
                        _solver_rows(
                            result,
                            stage="A_semantic",
                            split="F",
                            target=target,
                            source=source,
                            session=session,
                            heldout=heldout,
                            semantic_permutation=permutation_index,
                        )
                    )
    return {
        "target": target,
        "rows": rows,
        "solver_rows": solver_rows,
        "mismatch_gains": gains,
        "mismatch_actions": actions,
    }


def _combine_semantic(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for task in tasks for row in task["rows"]]
    if len(rows) != 2880:
        raise PairwiseContractError("semantic pair grid must contain 2880 rows")
    d = tasks[0]["mismatch_actions"].shape[-1]
    gains = np.full((9, 9, 2, 4, 5), np.nan)
    actions = np.full((9, 9, 2, 4, 5, d, d), np.nan)
    for task in tasks:
        target = task["target"]
        gains[target] = task["mismatch_gains"]
        actions[target] = task["mismatch_actions"]
    return {
        "table": pd.DataFrame(rows).sort_values(
            ["target", "source", "session_index", "heldout_index", "permutation_index"]
        ),
        "solver_rows": [row for task in tasks for row in task["solver_rows"]],
        "gains": gains,
        "actions": actions,
    }


def _cross_session_stage(
    arrays: Mapping[str, np.ndarray],
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    gains = np.full((9, 9, 2, 4), np.nan)
    identifiable = True
    for target in range(9):
        for source in range(9):
            if source == target:
                continue
            for train in range(2):
                test = 1 - train
                for heldout in range(4):
                    key = (target, source, train, heldout)
                    actions = primary["eq_bank"][key]
                    source_full = arrays["F"][source, test, heldout]
                    predictions = tuple(conjugate(action, source_full) for action in actions)
                    best_prediction = conjugate(
                        primary["best_actions"][target, source, train, heldout],
                        source_full,
                    )
                    prediction_a = conjugate(
                        primary["half_A_actions"][target, source, train, heldout],
                        arrays["A"][source, test, heldout],
                    )
                    prediction_b = conjugate(
                        primary["half_B_actions"][target, source, train, heldout],
                        arrays["B"][source, test, heldout],
                    )
                    assessment = classify_prediction_matrices(
                        predictions,
                        best_prediction=best_prediction,
                        split_half_prediction_a=prediction_a,
                        split_half_prediction_b=prediction_b,
                        settings=PAIRWISE_SETTINGS,
                    )
                    if assessment.classification == "PREDICTIVE_NONIDENTIFIABILITY":
                        identifiable = False
                    raw_error, action_error, gain = pairwise_error_gain(
                        arrays["F"][target, test, heldout],
                        source_full,
                        best_prediction,
                    )
                    gains[target, source, train, heldout] = gain
                    rows.append(
                        {
                            "target": target + 1,
                            "source": source + 1,
                            "train_session": SESSIONS[train],
                            "test_session": SESSIONS[test],
                            "direction_index": train,
                            "direction": f"{SESSIONS[train]}->{SESSIONS[test]}",
                            "heldout_class": CLASSES[heldout],
                            "heldout_index": heldout,
                            "raw_error": raw_error,
                            "action_error": action_error,
                            "gain": gain,
                            "D_eq": assessment.maximum_relative_prediction_dispersion,
                            "D_split": assessment.split_half_relative_variability,
                            "D_threshold": assessment.materiality_threshold,
                            "identifiability": assessment.classification,
                        }
                    )
    if len(rows) != 576:
        raise PairwiseContractError("cross-session pair grid is incomplete")
    return {"table": pd.DataFrame(rows), "gains": gains, "identifiable": identifiable}


def _cross_semantic_gains(
    arrays: Mapping[str, np.ndarray], semantic_actions: np.ndarray
) -> np.ndarray:
    gains = np.full((9, 9, 2, 4, 5), np.nan)
    for target in range(9):
        for source in range(9):
            if source == target:
                continue
            for train in range(2):
                test = 1 - train
                for heldout in range(4):
                    for permutation in range(5):
                        prediction = conjugate(
                            semantic_actions[target, source, train, heldout, permutation],
                            arrays["F"][source, test, heldout],
                        )
                        _, _, gain = pairwise_error_gain(
                            arrays["F"][target, test, heldout],
                            arrays["F"][source, test, heldout],
                            prediction,
                        )
                        gains[target, source, train, heldout, permutation] = gain
    return gains


def _cycle_diagnostics(
    U: np.ndarray, eq_bank: Mapping[tuple[int, int, int, int], np.ndarray]
) -> pd.DataFrame:
    rows = []
    for target in range(9):
        for middle in range(9):
            if middle == target:
                continue
            for source in range(9):
                if source in (target, middle):
                    continue
                for session in range(2):
                    for heldout in range(4):
                        diagnostic = equivalence_aware_cycle_diagnostic(
                            eq_bank[(target, source, session, heldout)],
                            eq_bank[(target, middle, session, heldout)],
                            eq_bank[(middle, source, session, heldout)],
                            U[source, session],
                        )
                        rows.append(
                            {
                                "target": target + 1,
                                "middle": middle + 1,
                                "source": source + 1,
                                "session": SESSIONS[session],
                                "heldout_class": CLASSES[heldout],
                                "relative_induced_cycle_discrepancy": diagnostic.relative_discrepancy,
                            }
                        )
    return pd.DataFrame(rows)


def _null_summary_row(name: str, gate: Any) -> dict[str, Any]:
    return {"test": name, **asdict(gate)}


def _locked_table(path: Path, status: str) -> None:
    pd.DataFrame([{"status": status}]).to_csv(path, index=False, lineterminator="\n")


def _write_report(
    output: Path,
    *,
    decision: str,
    stage_a_primary: Any | None,
    stage_a_semantic: Any | None,
    stage_b_primary: Any | None,
    stage_b_semantic: Any | None,
    cycle_median: float | None,
) -> None:
    def gate_line(name: str, gate: Any | None, locked: str) -> str:
        if gate is None:
            return f"- {name}: {locked}."
        return (
            f"- {name}: observed={gate.observed:.8f}, null median={gate.null_median:.8f}, "
            f"effect={gate.effect:.8f}, p={gate.p_value:.6f}, "
            f"{'PASS' if gate.passed else 'FAIL'}."
        )

    text = f"""# {REPORT_TITLE}

## Plain-language question

This audit asks whether an orthogonal sensor action estimated from three motor-imagery classes for one source–target subject pair improves prediction of the fourth unseen class. Pairwise source–target results are first collapsed within each target; the target subject, not the pair, is the inferential unit.

Pairwise success is a necessary consequence of a latent common-action model. It does not by itself prove that global latent subject actions and population class templates exist.

## Gates

{gate_line('Stage A primary within-session', stage_a_primary, 'UNASSESSED')}
{gate_line('Stage A semantic correspondence', stage_a_semantic, 'NOT_UNLOCKED_BY_PRIMARY_GATE')}
{gate_line('Stage B primary cross-session', stage_b_primary, 'NOT_UNLOCKED_BY_STAGE_A')}
{gate_line('Stage B semantic correspondence', stage_b_semantic, 'NOT_UNLOCKED_BY_PRIMARY_GATE')}

- Descriptive equivalence-aware cycle median: {('NOT_UNLOCKED' if cycle_median is None else f'{cycle_median:.8f}')}.
- Terminal decision: **{decision}**.

## Interpretation boundary

This result concerns only pairwise class-independent sensor-space orthogonal conjugation of marginally recentered identity-tangent class effects. It does not establish physiology, a complete model of individuality, a global latent action model, source-space structure, causal dynamics, unlabeled identifiability, or performance improvement. The planned next branch remains split-epoch covariance-set anatomy.
"""
    path = output / f"report/{REPORT_FILENAME}"
    path.write_text(text, encoding="utf-8")


def run_all(root: Path, *, workers: int) -> dict[str, Any]:
    config = load_config(root)
    if int(workers) != int(config["runtime"]["workers"]):
        raise PairwiseContractError(
            f"runtime workers={workers} != frozen workers={config['runtime']['workers']}"
        )
    assert_run_worktree(root, str(config["project"]["output_dir"]))
    output = prepare_output(root, config)
    arrays = load_and_reproduce_U(root, config, output)

    primary_tasks = _run_target_tasks(
        output, arrays, kind="primary", workers=workers
    )
    primary = _combine_primary(primary_tasks)
    primary["table"].to_csv(
        output / "tables/stage_A_pairwise_scores.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(primary["solver_rows"]).to_csv(
        output / "tables/stage_A_solver_diagnostics.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(primary["stabilizer_rows"]).to_csv(
        output / "tables/stage_A_stabilizer_summary.csv", index=False, lineterminator="\n"
    )
    atomic_npz(
        output / "objects/stage_A_pairwise_actions.npz",
        best_actions=primary["best_actions"],
        half_A_actions=primary["half_A_actions"],
        half_B_actions=primary["half_B_actions"],
        stabilizer_singular_values=primary["singular_values"],
    )
    ident_values = set(primary["table"]["identifiability"])
    if "PREDICTIVE_NONIDENTIFIABILITY" in ident_values:
        decision = terminal_decision(
            data_gate_pass=True,
            technical_gate_pass=True,
            identifiable=False,
            stage_a_primary_pass=None,
            stage_a_semantic_pass=None,
            stage_b_primary_pass=None,
            stage_b_semantic_pass=None,
        )
        terminal = {"decision": decision, "stage_A_identifiability": "FAIL"}
        atomic_json(output / "decisions/terminal_decision.json", terminal)
        _write_report(output, decision=decision, stage_a_primary=None, stage_a_semantic=None, stage_b_primary=None, stage_b_semantic=None, cycle_median=None)
        return terminal

    stage_a_gains = _gain_tensor(primary["table"])
    stage_a_cells, stage_a_subjects, observed_a = _summary_tables(
        stage_a_gains, session_labels=SESSIONS
    )
    stage_a_cells.to_csv(output / "tables/stage_A_target_cells.csv", index=False, lineterminator="\n")
    stage_a_subjects.to_csv(output / "tables/stage_A_subject_summary.csv", index=False, lineterminator="\n")
    unrelated_bank_a = unrelated_target_gain_bank(arrays["F"], primary["best_actions"])
    unrelated_null_a, choices_a = unrelated_target_null_statistics(
        unrelated_bank_a, stream="stage_A_unrelated_target"
    )
    gate_a = evaluate_stage_gate(observed_a, unrelated_null_a)
    atomic_npz(
        output / "nulls/stage_A_unrelated_target_null.npz",
        statistics=unrelated_null_a,
        derangement_choices=choices_a,
    )
    null_rows = [_null_summary_row("stage_A_unrelated_target", gate_a)]

    semantic_a = None
    semantic = None
    gate_b = None
    semantic_b = None
    cycle_median = None
    if gate_a.passed:
        semantic_tasks = _run_target_tasks(
            output, arrays, kind="semantic", workers=workers
        )
        semantic = _combine_semantic(semantic_tasks)
        semantic["table"].to_csv(
            output / "tables/stage_A_semantic_mismatch_scores.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(semantic["solver_rows"]).to_csv(
            output / "tables/stage_A_semantic_solver_diagnostics.csv", index=False, lineterminator="\n"
        )
        atomic_npz(
            output / "objects/stage_A_semantic_actions.npz",
            actions=semantic["actions"],
        )
        semantic_null_a, semantic_choices_a = semantic_null_statistics(
            semantic["gains"], stream="stage_A_semantic"
        )
        semantic_a = evaluate_stage_gate(observed_a, semantic_null_a)
        atomic_npz(
            output / "nulls/stage_A_semantic_null.npz",
            statistics=semantic_null_a,
            permutation_choices=semantic_choices_a,
        )
        null_rows.append(_null_summary_row("stage_A_semantic", semantic_a))
    else:
        _locked_table(
            output / "tables/stage_A_semantic_mismatch_scores.csv",
            "NOT_UNLOCKED_BY_PRIMARY_GATE",
        )

    stage_a_final = bool(gate_a.passed and semantic_a is not None and semantic_a.passed)
    if stage_a_final:
        cycle = _cycle_diagnostics(arrays["F"], primary["eq_bank"])
        cycle.to_csv(output / "tables/global_cycle_diagnostics.csv", index=False, lineterminator="\n")
        cycle_median = float(np.median(cycle["relative_induced_cycle_discrepancy"]))

        cross = _cross_session_stage(arrays, primary)
        cross["table"].to_csv(
            output / "tables/stage_B_pairwise_scores.csv", index=False, lineterminator="\n"
        )
        if not cross["identifiable"]:
            decision = terminal_decision(
                data_gate_pass=True,
                technical_gate_pass=True,
                identifiable=False,
                stage_a_primary_pass=True,
                stage_a_semantic_pass=True,
                stage_b_primary_pass=None,
                stage_b_semantic_pass=None,
            )
            terminal = {"decision": decision, "stage_A_primary": asdict(gate_a), "stage_A_semantic": asdict(semantic_a)}
            atomic_json(output / "decisions/terminal_decision.json", terminal)
            _write_report(output, decision=decision, stage_a_primary=gate_a, stage_a_semantic=semantic_a, stage_b_primary=None, stage_b_semantic=None, cycle_median=cycle_median)
            return terminal
        cells_b, subjects_b, observed_b = _summary_tables(
            cross["gains"],
            session_labels=("0train->1test", "1test->0train"),
        )
        cells_b.to_csv(output / "tables/stage_B_target_cells.csv", index=False, lineterminator="\n")
        subjects_b.to_csv(output / "tables/stage_B_subject_summary.csv", index=False, lineterminator="\n")
        test_order_U = arrays["F"][:, [1, 0]]
        unrelated_bank_b = unrelated_target_gain_bank(
            test_order_U, primary["best_actions"]
        )
        unrelated_null_b, choices_b = unrelated_target_null_statistics(
            unrelated_bank_b, stream="stage_B_unrelated_target"
        )
        gate_b = evaluate_stage_gate(observed_b, unrelated_null_b)
        atomic_npz(
            output / "nulls/stage_B_unrelated_target_null.npz",
            statistics=unrelated_null_b,
            derangement_choices=choices_b,
        )
        null_rows.append(_null_summary_row("stage_B_unrelated_target", gate_b))
        if gate_b.passed:
            assert semantic is not None
            cross_semantic = _cross_semantic_gains(arrays, semantic["actions"])
            cross_rows = []
            for target in range(9):
                for source in range(9):
                    if source == target:
                        continue
                    for train in range(2):
                        for heldout in range(4):
                            for permutation in range(5):
                                cross_rows.append(
                                    {
                                        "target": target + 1,
                                        "source": source + 1,
                                        "train_session": SESSIONS[train],
                                        "test_session": SESSIONS[1 - train],
                                        "heldout_class": CLASSES[heldout],
                                        "permutation_index": permutation,
                                        "gain": cross_semantic[
                                            target,
                                            source,
                                            train,
                                            heldout,
                                            permutation,
                                        ],
                                    }
                                )
            pd.DataFrame(cross_rows).to_csv(
                output / "tables/stage_B_semantic_mismatch_scores.csv",
                index=False,
                lineterminator="\n",
            )
            semantic_null_b, semantic_choices_b = semantic_null_statistics(
                cross_semantic, stream="stage_B_semantic"
            )
            semantic_b = evaluate_stage_gate(observed_b, semantic_null_b)
            atomic_npz(
                output / "nulls/stage_B_semantic_null.npz",
                statistics=semantic_null_b,
                permutation_choices=semantic_choices_b,
            )
            null_rows.append(_null_summary_row("stage_B_semantic", semantic_b))
        else:
            _locked_table(
                output / "tables/stage_B_semantic_mismatch_scores.csv",
                "NOT_UNLOCKED_BY_PRIMARY_GATE",
            )
    else:
        _locked_table(output / "tables/stage_B_pairwise_scores.csv", "NOT_UNLOCKED_BY_STAGE_A")
        _locked_table(output / "tables/global_cycle_diagnostics.csv", "NOT_UNLOCKED_BY_STAGE_A")

    pd.DataFrame(null_rows).to_csv(
        output / "tables/null_summary.csv", index=False, lineterminator="\n"
    )
    atomic_json(
        output / "nulls/seed_manifest.json",
        {
            "master_seed": 20260810,
            "replicates": NULL_REPLICATES,
            "streams": [row["test"] for row in null_rows],
        },
    )
    decision = terminal_decision(
        data_gate_pass=True,
        technical_gate_pass=True,
        identifiable=True,
        stage_a_primary_pass=gate_a.passed,
        stage_a_semantic_pass=None if semantic_a is None else semantic_a.passed,
        stage_b_primary_pass=None if gate_b is None else gate_b.passed,
        stage_b_semantic_pass=None if semantic_b is None else semantic_b.passed,
    )
    chain = [
        {"order": 1, "gate": "Stage_A_primary", "status": "PASS" if gate_a.passed else "FAIL"},
        {"order": 2, "gate": "Stage_A_semantic", "status": "NOT_UNLOCKED_BY_PRIMARY_GATE" if semantic_a is None else ("PASS" if semantic_a.passed else "FAIL")},
        {"order": 3, "gate": "Stage_B_primary", "status": "NOT_UNLOCKED_BY_STAGE_A" if gate_b is None else ("PASS" if gate_b.passed else "FAIL")},
        {"order": 4, "gate": "Stage_B_semantic", "status": "NOT_UNLOCKED_BY_PRIMARY_GATE" if semantic_b is None else ("PASS" if semantic_b.passed else "FAIL")},
        {"order": 5, "gate": "terminal", "status": decision},
    ]
    pd.DataFrame(chain).to_csv(output / "decisions/decision_chain.csv", index=False, lineterminator="\n")
    terminal = {
        "decision": decision,
        "stage_A_primary": asdict(gate_a),
        "stage_A_semantic": None if semantic_a is None else asdict(semantic_a),
        "stage_B_primary": None if gate_b is None else asdict(gate_b),
        "stage_B_semantic": None if semantic_b is None else asdict(semantic_b),
        "cycle_median_descriptive": cycle_median,
        "profiled_global_model": "NOT_RUN_REQUIRES_SEPARATE_PRODUCT_START_FREEZE",
    }
    atomic_json(output / "decisions/terminal_decision.json", terminal)
    _write_report(
        output,
        decision=decision,
        stage_a_primary=gate_a,
        stage_a_semantic=semantic_a,
        stage_b_primary=gate_b,
        stage_b_semantic=semantic_b,
        cycle_median=cycle_median,
    )
    return terminal


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_PROTOCOL_SHA256",
    "load_config",
    "run_all",
]
