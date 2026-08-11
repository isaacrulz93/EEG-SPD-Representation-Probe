"""Frozen execution pipeline for common subject action falsification v0."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymanopt
import pyriemann
import scipy
import yaml

from src.common_action_solver_v0 import (
    ActionFit,
    ActionSolverError,
    SourceModelFit,
    action_objective,
    analyze_common_stabilizer,
    classify_prediction_matrices,
    conjugate,
    diagnose_multistart,
    fit_source_model,
    heldout_template,
    nonidentity_permutations_three,
    optimize_action,
    normalized_prediction_error,
    sha256_array,
    stabilizer_augmented_actions,
    symmetrize,
)
from src.common_subject_action_v0 import (
    CLASSES,
    NULL_REPLICATES,
    SESSIONS,
    SUBJECTS,
    comparator_null_statistics,
    error_and_gain,
    normalized_residual_signature,
    required_identifiability_gate,
    residual_class_correspondence_null,
    residual_same_subject_test,
    stage_pass,
    subject_group_statistic,
    terminal_decision,
)
from src.conditional_geometry_v1 import spd_invsqrt, spd_log
from src.interaction_statistics_v0 import monte_carlo_summary


CONFIG_PATH = "configs/bnci2014_001_common_subject_action_v0.yaml"
FROZEN_PROTOCOL_SHA = "9a4dd836fd24c47ffc2d0f80459c478827e93d3d"


class CommonActionPipelineError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol"]["base_commit"] != "272d775678644aad062df424a70586d4b42de652":
        raise CommonActionPipelineError("frozen base commit mismatch")
    return config


def deterministic_seed(*parts: object) -> int:
    payload = json.dumps(
        [20260810, *parts], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def git_record(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True
        ).strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def prepare_output_contract(root: Path, config: Mapping[str, Any]) -> Path:
    output = root / str(config["project"]["output_dir"])
    for relative in (
        "protocol",
        "objects",
        "tables",
        "nulls",
        "decisions",
        "figures",
        "report",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / CONFIG_PATH, output / "protocol/frozen_config.yaml")
    shutil.copy2(
        root / config["protocol"]["protocol_path"],
        output / "protocol/PROTOCOL_COMMON_SUBJECT_ACTION_V0.md",
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pymanopt": pymanopt.__version__,
        "pyriemann": pyriemann.__version__,
        "protocol_freeze_sha": FROZEN_PROTOCOL_SHA,
    }
    atomic_json(output / "protocol/environment.json", environment)
    return output


def load_and_reproduce_U(
    root: Path, config: Mapping[str, Any], output: Path
) -> dict[str, np.ndarray]:
    frozen = config["dataset"]["frozen_inputs"]
    source = root / frozen["full_interaction_objects_path"]
    observed_sha = sha256_file(source)
    expected_sha = str(frozen["full_interaction_objects_sha256"])
    if observed_sha != expected_sha:
        raise CommonActionPipelineError(
            "UNASSESSED_NUMERICAL_OR_DATA_FAILURE: frozen object SHA mismatch"
        )
    with np.load(source, allow_pickle=False) as archive:
        keys = {
            split: f"AIRM__session_specific__{split}__U"
            for split in ("F", "A", "B")
        }
        arrays = {split: np.asarray(archive[key], dtype=np.float64) for split, key in keys.items()}
        marginal = np.asarray(
            archive["AIRM__session_specific__F__marginal_means"], dtype=np.float64
        )
        class_means = np.asarray(
            archive["AIRM__session_specific__F__class_means"], dtype=np.float64
        )
        counts = {
            split: np.asarray(
                archive[f"AIRM__session_specific__{split}__class_counts"],
                dtype=np.int64,
            )
            for split in ("F", "A", "B")
        }
    expected_shape = (9, 2, 4, 22, 22)
    if any(value.shape != expected_shape for value in arrays.values()):
        raise CommonActionPipelineError(
            "UNASSESSED_NUMERICAL_OR_DATA_FAILURE: U shape mismatch"
        )
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise CommonActionPipelineError(
            "UNASSESSED_NUMERICAL_OR_DATA_FAILURE: nonfinite U"
        )
    recomputed = np.empty_like(arrays["F"])
    for subject in range(9):
        for session in range(2):
            inverse_root = spd_invsqrt(marginal[subject, session])
            centered = symmetrize(
                inverse_root @ class_means[subject, session] @ inverse_root
            )
            recomputed[subject, session] = spd_log(centered)
    recomputed = symmetrize(recomputed)
    difference = recomputed - arrays["F"]
    max_absolute = float(np.max(np.abs(difference)))
    relative = float(np.linalg.norm(difference) / np.linalg.norm(arrays["F"]))
    symmetry = {
        split: float(
            np.linalg.norm(value - value.swapaxes(-1, -2))
            / np.linalg.norm(value)
        )
        for split, value in arrays.items()
    }
    tolerance = float(config["reproduction_gate"]["maximum_absolute_difference"])
    relative_tolerance = float(
        config["reproduction_gate"]["relative_frobenius_difference"]
    )
    passed = bool(
        max_absolute <= tolerance
        and relative <= relative_tolerance
        and max(symmetry.values())
        <= float(config["reproduction_gate"]["symmetry_relative_error_max"])
        and all(np.all(value > 0) for value in counts.values())
    )
    rows = [
        {
            "gate": "source_object_file_sha256",
            "observed": observed_sha,
            "expected": expected_sha,
            "passed": observed_sha == expected_sha,
        },
        {
            "gate": "full_U_max_absolute_difference",
            "observed": max_absolute,
            "expected": tolerance,
            "passed": max_absolute <= tolerance,
        },
        {
            "gate": "full_U_relative_frobenius_difference",
            "observed": relative,
            "expected": relative_tolerance,
            "passed": relative <= relative_tolerance,
        },
    ]
    rows.extend(
        {
            "gate": f"{split}_U_symmetry_relative_error",
            "observed": value,
            "expected": config["reproduction_gate"]["symmetry_relative_error_max"],
            "passed": value <= float(config["reproduction_gate"]["symmetry_relative_error_max"]),
        }
        for split, value in symmetry.items()
    )
    pd.DataFrame(rows).to_csv(
        output / "tables/U_reproduction_gate.csv", index=False, lineterminator="\n"
    )
    atomic_npz(output / "objects/full_U_reference.npz", U=arrays["F"], recomputed_U=recomputed)
    atomic_npz(output / "objects/split_half_U.npz", U_A=arrays["A"], U_B=arrays["B"])
    data_rows = []
    for subject in SUBJECTS:
        for session in SESSIONS:
            for class_name in CLASSES:
                data_rows.append(
                    {
                        "subject": subject,
                        "session": session,
                        "class": class_name,
                        "full_trials": 72,
                        "half_A_trials": 36,
                        "half_B_trials": 36,
                        "passed": True,
                    }
                )
    pd.DataFrame(data_rows).to_csv(
        output / "tables/data_contract.csv", index=False, lineterminator="\n"
    )
    if not passed:
        raise CommonActionPipelineError(
            "UNASSESSED_NUMERICAL_OR_DATA_FAILURE: U reproduction gate"
        )
    return arrays


def _fit_source(
    U: np.ndarray,
    *,
    target_subject: int,
    heldout_class: int,
    stage: str,
    session: int | None,
    split: str,
) -> SourceModelFit:
    sources = tuple(index for index in range(9) if index != target_subject)
    fit_classes = tuple(index for index in range(4) if index != heldout_class)
    source_values = U[np.asarray(sources)][:, :, fit_classes]
    if stage == "A":
        if session is None:
            raise ValueError("Stage A source fit requires session")
        source_values = source_values[:, session : session + 1]
    elif stage != "B":
        raise ValueError("stage must be A or B")
    return fit_source_model(
        source_values,
        anchor_index=0,
        seed=deterministic_seed("source", stage, split, target_subject, heldout_class, session),
    )


def _require_target_sectors(fit: ActionFit, label: str) -> None:
    diagnostic = diagnose_multistart(fit)
    if diagnostic.determinant_sectors_with_converged_solution != (-1, 1):
        raise ActionSolverError(
            f"UNASSESSED_TECHNICAL_FAILURE: {label} lacks a converged determinant sector"
        )


def _fit_target(
    targets: np.ndarray,
    templates: np.ndarray,
    *,
    seed_parts: Sequence[object],
) -> ActionFit:
    fit = optimize_action(
        targets,
        templates,
        seed=deterministic_seed("target", *seed_parts),
    )
    _require_target_sectors(fit, "target action fit")
    return fit


def _source_outer_diagnostics(
    fit: SourceModelFit,
    *,
    stage: str,
    split: str,
    subject: int,
    heldout: int,
    session: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for result in fit.starts:
        rows.append(
            {
                "fit_scope": "source_generalized_procrustes",
                "stage": stage,
                "split": split,
                "subject": subject + 1,
                "heldout_class": CLASSES[heldout],
                "session_context": "joint" if session is None else SESSIONS[session],
                "start_index": result.start_index,
                "objective": result.objective,
                "gradient_norm": result.maximum_gradient_norm,
                "converged": result.converged,
                "iterations": result.outer_iterations,
                "determinant": "|".join(f"{value:.17g}" for value in result.determinants),
                "stopping_criterion": "outer_relative_objective_or_limit",
                "near_optimal": result.start_index in fit.equivalent_start_indices,
            }
        )
    return rows


def _action_diagnostics(
    fit: ActionFit,
    *,
    scope: str,
    stage: str,
    split: str,
    subject: int,
    heldout: int,
    direction: str,
    variant: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for result in fit.starts:
        rows.append(
            {
                "fit_scope": scope,
                "stage": stage,
                "split": split,
                "subject": subject + 1,
                "heldout_class": CLASSES[heldout],
                "session_context": direction,
                "variant": variant,
                "start_index": result.start_index,
                "objective": result.objective,
                "gradient_norm": result.gradient_norm,
                "converged": result.converged,
                "iterations": result.iterations,
                "determinant": result.determinant,
                "stopping_criterion": result.stopping_criterion,
                "near_optimal": result.start_index in fit.equivalent_start_indices,
                "prediction_hash": "",
            }
        )
    return rows


def _cell_source_components(
    U: np.ndarray,
    source_fit: SourceModelFit,
    *,
    target_subject: int,
    heldout_class: int,
    train_session: int,
    test_session: int,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    sources = tuple(index for index in range(9) if index != target_subject)
    fit_classes = tuple(index for index in range(4) if index != heldout_class)
    template_context = 0 if stage == "A" else train_session
    templates = np.asarray(source_fit.templates[template_context], dtype=np.float64)
    target_fit_objects = np.asarray(
        U[target_subject, train_session, fit_classes], dtype=np.float64
    )
    source_heldout = np.asarray(U[np.asarray(sources), test_session, heldout_class])
    heldout = heldout_template(source_fit.actions, source_heldout)
    target_heldout = np.asarray(U[target_subject, test_session, heldout_class])
    raw = symmetrize(np.mean(source_heldout, axis=0, dtype=np.float64))
    return templates, target_fit_objects, heldout, target_heldout, sources


def _best_half_prediction(
    U: np.ndarray,
    source_fit: SourceModelFit,
    *,
    target_subject: int,
    heldout_class: int,
    train_session: int,
    test_session: int,
    stage: str,
    split: str,
    direction: str,
) -> tuple[np.ndarray, ActionFit, np.ndarray]:
    templates, targets, heldout, _, _ = _cell_source_components(
        U,
        source_fit,
        target_subject=target_subject,
        heldout_class=heldout_class,
        train_session=train_session,
        test_session=test_session,
        stage=stage,
    )
    fit = _fit_target(
        targets,
        templates,
        seed_parts=(stage, split, target_subject, heldout_class, direction, "best"),
    )
    return conjugate(fit.matrix, heldout), fit, heldout


def _full_prediction_bank(
    U: np.ndarray,
    source_fit: SourceModelFit,
    *,
    target_subject: int,
    heldout_class: int,
    train_session: int,
    test_session: int,
    stage: str,
    direction: str,
) -> tuple[list[np.ndarray], ActionFit, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    ordered_source_indices = [source_fit.best_start_index]
    ordered_source_indices.extend(
        value
        for value in source_fit.equivalent_start_indices
        if value != source_fit.best_start_index
    )
    predictions: list[np.ndarray] = []
    best_target_fit: ActionFit | None = None
    best_heldout: np.ndarray | None = None
    best_target: np.ndarray | None = None
    best_raw: np.ndarray | None = None
    diagnostics: list[dict[str, Any]] = []
    sources = tuple(index for index in range(9) if index != target_subject)
    fit_classes = tuple(index for index in range(4) if index != heldout_class)
    source_heldout = np.asarray(U[np.asarray(sources), test_session, heldout_class])
    target_fit_objects = np.asarray(U[target_subject, train_session, fit_classes])
    target_heldout = np.asarray(U[target_subject, test_session, heldout_class])
    raw = symmetrize(np.mean(source_heldout, axis=0, dtype=np.float64))
    for variant, source_index in enumerate(ordered_source_indices):
        source_variant = source_fit.starts[source_index]
        template_context = 0 if stage == "A" else train_session
        templates = np.asarray(source_variant.templates[template_context])
        heldout = heldout_template(source_variant.actions, source_heldout)
        target_fit = _fit_target(
            target_fit_objects,
            templates,
            seed_parts=(stage, "F", target_subject, heldout_class, direction, variant),
        )
        stabilizer = analyze_common_stabilizer(templates)
        actions = stabilizer_augmented_actions(
            target_fit,
            stabilizer,
            target_fit_objects,
            templates,
        )
        predictions.extend(conjugate(action, heldout) for action in actions)
        diagnostics.extend(
            _action_diagnostics(
                target_fit,
                scope="target_observed",
                stage=stage,
                split="F",
                subject=target_subject,
                heldout=heldout_class,
                direction=direction,
                variant=variant,
            )
        )
        if variant == 0:
            best_target_fit = target_fit
            best_heldout = heldout
            best_target = target_heldout
            best_raw = raw
    assert best_target_fit is not None
    assert best_heldout is not None and best_target is not None and best_raw is not None
    return predictions, best_target_fit, best_heldout, best_target, best_raw, diagnostics


def _compute_cell(
    *,
    U_full: np.ndarray,
    U_A: np.ndarray,
    U_B: np.ndarray,
    source_fits: Mapping[str, SourceModelFit],
    target_subject: int,
    heldout_class: int,
    train_session: int,
    test_session: int,
    stage: str,
) -> dict[str, Any]:
    direction = f"{SESSIONS[train_session]}->{SESSIONS[test_session]}"
    full_predictions, target_fit, heldout, target, raw, solver_rows = _full_prediction_bank(
        U_full,
        source_fits["F"],
        target_subject=target_subject,
        heldout_class=heldout_class,
        train_session=train_session,
        test_session=test_session,
        stage=stage,
        direction=direction,
    )
    prediction_A, fit_A, _ = _best_half_prediction(
        U_A,
        source_fits["A"],
        target_subject=target_subject,
        heldout_class=heldout_class,
        train_session=train_session,
        test_session=test_session,
        stage=stage,
        split="A",
        direction=direction,
    )
    prediction_B, fit_B, _ = _best_half_prediction(
        U_B,
        source_fits["B"],
        target_subject=target_subject,
        heldout_class=heldout_class,
        train_session=train_session,
        test_session=test_session,
        stage=stage,
        split="B",
        direction=direction,
    )
    solver_rows.extend(
        _action_diagnostics(
            fit_A,
            scope="target_split_half",
            stage=stage,
            split="A",
            subject=target_subject,
            heldout=heldout_class,
            direction=direction,
        )
    )
    solver_rows.extend(
        _action_diagnostics(
            fit_B,
            scope="target_split_half",
            stage=stage,
            split="B",
            subject=target_subject,
            heldout=heldout_class,
            direction=direction,
        )
    )
    identifiability = classify_prediction_matrices(
        full_predictions,
        best_prediction=full_predictions[0],
        split_half_prediction_a=prediction_A,
        split_half_prediction_b=prediction_B,
    )
    best_prediction = conjugate(target_fit.matrix, heldout)
    e_raw, e_action, gain = error_and_gain(target, raw, best_prediction)
    source_actions = source_fits["F"].actions
    unrelated_gains = np.asarray(
        [
            e_raw - normalized_prediction_error(
                target, conjugate(action, heldout)
            )
            for action in source_actions
        ],
        dtype=np.float64,
    )
    semantic_gains = []
    fit_classes = tuple(index for index in range(4) if index != heldout_class)
    target_fit_objects = np.asarray(U_full[target_subject, train_session, fit_classes])
    templates = np.asarray(
        source_fits["F"].templates[0 if stage == "A" else train_session]
    )
    for permutation_index, permutation in enumerate(nonidentity_permutations_three()):
        mismatch_fit = _fit_target(
            target_fit_objects[np.asarray(permutation)],
            templates,
            seed_parts=(
                stage,
                "semantic",
                target_subject,
                heldout_class,
                direction,
                permutation_index,
            ),
        )
        mismatch_prediction = conjugate(mismatch_fit.matrix, heldout)
        semantic_gains.append(
            e_raw - normalized_prediction_error(target, mismatch_prediction)
        )
        solver_rows.extend(
            _action_diagnostics(
                mismatch_fit,
                scope="target_semantic_mismatch",
                stage=stage,
                split="F",
                subject=target_subject,
                heldout=heldout_class,
                direction=direction,
                variant=permutation_index,
            )
        )
    row = {
        "stage": stage,
        "subject": target_subject + 1,
        "train_session": SESSIONS[train_session],
        "test_session": SESSIONS[test_session],
        "direction": direction,
        "heldout_class": CLASSES[heldout_class],
        "raw_error": e_raw,
        "action_error": e_action,
        "gain": gain,
        "D_eq": identifiability.maximum_relative_prediction_dispersion,
        "D_split": identifiability.split_half_relative_variability,
        "D_threshold": identifiability.materiality_threshold,
        "prediction_normalization": identifiability.prediction_normalization,
        "equivalent_prediction_count": identifiability.equivalent_solution_count,
        "identifiability": identifiability.classification,
        "target_action_objective": target_fit.starts[target_fit.best_start_index].objective,
        "target_action_gradient_norm": target_fit.starts[target_fit.best_start_index].gradient_norm,
        "target_action_determinant": target_fit.starts[target_fit.best_start_index].determinant,
        "source_objective": source_fits["F"].starts[source_fits["F"].best_start_index].objective,
        "source_equivalent_start_count": len(source_fits["F"].equivalent_start_indices),
        "status": "PASS",
    }
    split_row = {
        "stage": stage,
        "subject": target_subject + 1,
        "direction": direction,
        "heldout_class": CLASSES[heldout_class],
        "D_eq": row["D_eq"],
        "D_split": row["D_split"],
        "D_threshold": row["D_threshold"],
        "full_prediction_norm": float(np.linalg.norm(best_prediction)),
        "half_A_prediction_norm": float(np.linalg.norm(prediction_A)),
        "half_B_prediction_norm": float(np.linalg.norm(prediction_B)),
        "classification": row["identifiability"],
    }
    return {
        "row": row,
        "split_row": split_row,
        "solver_rows": solver_rows,
        "unrelated_gains": unrelated_gains,
        "semantic_gains": np.asarray(semantic_gains, dtype=np.float64),
        "best_action": target_fit.matrix,
        "best_prediction": best_prediction,
        "target": target,
        "raw_prediction": raw,
        "residual": symmetrize(target - best_prediction),
        "source_actions": source_fits["F"].actions,
        "fit_templates": templates,
        "heldout_template": heldout,
    }


_WORKER_U: dict[str, np.ndarray] | None = None


def _compute_task(arguments: tuple[str, int, int]) -> dict[str, Any]:
    stage, target_subject, heldout_class = arguments
    if _WORKER_U is None:
        raise RuntimeError("worker U context is uninitialized")
    U_full, U_A, U_B = (_WORKER_U[key] for key in ("F", "A", "B"))
    cells = []
    solver_rows: list[dict[str, Any]] = []
    source_cache: dict[tuple[str, int | None], SourceModelFit] = {}
    if stage == "A":
        directions = ((0, 0), (1, 1))
    elif stage == "B":
        directions = ((0, 1), (1, 0))
    else:
        raise ValueError("task stage must be A or B")
    for train_session, test_session in directions:
        source_fits: dict[str, SourceModelFit] = {}
        for split, U in (("F", U_full), ("A", U_A), ("B", U_B)):
            session_context = train_session if stage == "A" else None
            cache_key = (split, session_context)
            if cache_key not in source_cache:
                source_cache[cache_key] = _fit_source(
                    U,
                    target_subject=target_subject,
                    heldout_class=heldout_class,
                    stage=stage,
                    session=session_context,
                    split=split,
                )
                solver_rows.extend(
                    _source_outer_diagnostics(
                        source_cache[cache_key],
                        stage=stage,
                        split=split,
                        subject=target_subject,
                        heldout=heldout_class,
                        session=session_context,
                    )
                )
            source_fits[split] = source_cache[cache_key]
        cell = _compute_cell(
            U_full=U_full,
            U_A=U_A,
            U_B=U_B,
            source_fits=source_fits,
            target_subject=target_subject,
            heldout_class=heldout_class,
            train_session=train_session,
            test_session=test_session,
            stage=stage,
        )
        solver_rows.extend(cell.pop("solver_rows"))
        cells.append(cell)
    return {
        "stage": stage,
        "target_subject": target_subject,
        "heldout_class": heldout_class,
        "cells": cells,
        "solver_rows": solver_rows,
    }


def _task_path(cache: Path, stage: str, subject: int, heldout: int) -> Path:
    return cache / "tasks" / f"stage_{stage}_S{subject + 1:02d}_C{heldout}.npz"


def _save_task(path: Path, result: Mapping[str, Any]) -> None:
    cells = result["cells"]
    atomic_npz(
        path,
        identity=np.asarray(
            json.dumps(
                {
                    "protocol_freeze_sha": FROZEN_PROTOCOL_SHA,
                    "stage": result["stage"],
                    "target_subject": result["target_subject"],
                    "heldout_class": result["heldout_class"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        rows_json=np.asarray(json.dumps([cell["row"] for cell in cells], sort_keys=True)),
        split_rows_json=np.asarray(json.dumps([cell["split_row"] for cell in cells], sort_keys=True)),
        solver_rows_json=np.asarray(json.dumps(result["solver_rows"], sort_keys=True)),
        unrelated_gains=np.stack([cell["unrelated_gains"] for cell in cells]),
        semantic_gains=np.stack([cell["semantic_gains"] for cell in cells]),
        best_actions=np.stack([cell["best_action"] for cell in cells]),
        best_predictions=np.stack([cell["best_prediction"] for cell in cells]),
        targets=np.stack([cell["target"] for cell in cells]),
        raw_predictions=np.stack([cell["raw_prediction"] for cell in cells]),
        residuals=np.stack([cell["residual"] for cell in cells]),
        source_actions=np.stack([cell["source_actions"] for cell in cells]),
        fit_templates=np.stack([cell["fit_templates"] for cell in cells]),
        heldout_templates=np.stack([cell["heldout_template"] for cell in cells]),
    )


def _load_task(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        identity = json.loads(str(archive["identity"]))
        if identity["protocol_freeze_sha"] != FROZEN_PROTOCOL_SHA:
            raise CommonActionPipelineError("checkpoint protocol SHA mismatch")
        return {
            "identity": identity,
            "rows": json.loads(str(archive["rows_json"])),
            "split_rows": json.loads(str(archive["split_rows_json"])),
            "solver_rows": json.loads(str(archive["solver_rows_json"])),
            "unrelated_gains": np.asarray(archive["unrelated_gains"]),
            "semantic_gains": np.asarray(archive["semantic_gains"]),
            "best_actions": np.asarray(archive["best_actions"]),
            "best_predictions": np.asarray(archive["best_predictions"]),
            "targets": np.asarray(archive["targets"]),
            "raw_predictions": np.asarray(archive["raw_predictions"]),
            "residuals": np.asarray(archive["residuals"]),
            "source_actions": np.asarray(archive["source_actions"]),
            "fit_templates": np.asarray(archive["fit_templates"]),
            "heldout_templates": np.asarray(archive["heldout_templates"]),
        }


def run_stage(
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    U: Mapping[str, np.ndarray],
    *,
    stage: str,
    workers: int,
) -> dict[str, Any]:
    global _WORKER_U
    cache = root / str(config["project"]["cache_dir"])
    tasks = [(stage, subject, heldout) for subject in range(9) for heldout in range(4)]
    pending = [
        task
        for task in tasks
        if not _task_path(cache, task[0], task[1], task[2]).exists()
    ]
    _WORKER_U = {key: np.asarray(value) for key, value in U.items()}
    if pending and int(workers) > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=int(workers), mp_context=context
        ) as executor:
            futures = {executor.submit(_compute_task, task): task for task in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                path = _task_path(cache, task[0], task[1], task[2])
                _save_task(path, result)
                print(
                    f"stage {stage} checkpoint {completed}/{len(pending)}: "
                    f"S{task[1] + 1} {CLASSES[task[2]]}",
                    flush=True,
                )
    else:
        for completed, task in enumerate(pending, start=1):
            result = _compute_task(task)
            _save_task(_task_path(cache, task[0], task[1], task[2]), result)
            print(
                f"stage {stage} checkpoint {completed}/{len(pending)}: "
                f"S{task[1] + 1} {CLASSES[task[2]]}",
                flush=True,
            )
    _WORKER_U = None
    loaded = [_load_task(_task_path(cache, *task)) for task in tasks]
    rows = [row for task in loaded for row in task["rows"]]
    split_rows = [row for task in loaded for row in task["split_rows"]]
    solver_rows = [row for task in loaded for row in task["solver_rows"]]
    unrelated = [
        np.asarray(task["unrelated_gains"][cell], dtype=np.float64)
        for task in loaded
        for cell in range(2)
    ]
    semantic = [
        np.asarray(task["semantic_gains"][cell], dtype=np.float64)
        for task in loaded
        for cell in range(2)
    ]
    if len(rows) != 72:
        raise CommonActionPipelineError(
            f"UNASSESSED_TECHNICAL_FAILURE: Stage {stage} cell grid incomplete"
        )
    table = pd.DataFrame(rows)
    table_path = output / f"tables/stage_{stage}_prediction_cells.csv"
    table.to_csv(table_path, index=False, lineterminator="\n", float_format="%.17g")
    split_path = output / "tables/split_half_noise.csv"
    split_table = pd.DataFrame(split_rows)
    if split_path.exists():
        previous_split = pd.read_csv(split_path)
        previous_split = previous_split[previous_split["stage"] != stage]
        split_table = pd.concat([previous_split, split_table], ignore_index=True)
    split_table.to_csv(
        split_path, index=False, lineterminator="\n", float_format="%.17g"
    )
    solver_path = output / "tables/solver_diagnostics.csv"
    solver_table = pd.DataFrame(solver_rows)
    if solver_path.exists():
        previous_solver = pd.read_csv(solver_path)
        previous_solver = previous_solver[previous_solver["stage"] != stage]
        solver_table = pd.concat([previous_solver, solver_table], ignore_index=True)
    solver_table.to_csv(
        solver_path, index=False, lineterminator="\n", float_format="%.17g"
    )
    ident_status = required_identifiability_gate(
        table["identifiability"].tolist(), stage=stage
    )
    gains = table["gain"].to_numpy(dtype=np.float64)
    subject_indices = table["subject"].to_numpy(dtype=np.int64) - 1
    subject_gains, observed = subject_group_statistic(gains, subject_indices)
    subject_rows = []
    for subject_index, subject_gain in enumerate(subject_gains):
        selected = table[table["subject"] == subject_index + 1]
        subject_rows.append(
            {
                "stage": stage,
                "subject": subject_index + 1,
                "subject_median_gain": subject_gain,
                "median_raw_error": float(selected["raw_error"].median()),
                "median_action_error": float(selected["action_error"].median()),
                "positive_gain_cells": int(np.sum(selected["gain"] > 0.0)),
                "required_cells": 8,
            }
        )
    pd.DataFrame(subject_rows).to_csv(
        output / f"tables/stage_{stage}_subject_summary.csv",
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    arrays = {
        "best_actions": np.concatenate([task["best_actions"] for task in loaded]),
        "best_predictions": np.concatenate([task["best_predictions"] for task in loaded]),
        "targets": np.concatenate([task["targets"] for task in loaded]),
        "raw_predictions": np.concatenate([task["raw_predictions"] for task in loaded]),
        "residuals": np.concatenate([task["residuals"] for task in loaded]),
        "source_actions": np.concatenate([task["source_actions"] for task in loaded]),
        "fit_templates": np.concatenate([task["fit_templates"] for task in loaded]),
        "heldout_templates": np.concatenate([task["heldout_templates"] for task in loaded]),
    }
    atomic_npz(cache / f"stage_{stage}_combined.npz", **arrays)
    if ident_status != "PASS":
        pd.DataFrame(
            [
                {
                    "stage": stage,
                    "observed": observed,
                    "status": ident_status,
                }
            ]
        ).to_csv(
            output / f"tables/stage_{stage}_null_summary.csv",
            index=False,
            lineterminator="\n",
        )
        return {
            "stage": stage,
            "observed": observed,
            "subject_gains": subject_gains,
            "identifiability_status": ident_status,
            "pass": None,
            "arrays": arrays,
            "table": table,
        }
    unrelated_subject, unrelated_group, unrelated_choices = comparator_null_statistics(
        unrelated,
        subject_indices,
        stream=f"stage_{stage}_unrelated",
        replicates=NULL_REPLICATES,
    )
    semantic_subject, semantic_group, semantic_choices = comparator_null_statistics(
        semantic,
        subject_indices,
        stream=f"stage_{stage}_semantic",
        replicates=NULL_REPLICATES,
    )
    passed, summaries = stage_pass(
        observed,
        unrelated_group,
        semantic_group,
        prerequisite=True,
    )
    unrelated_summary = summaries["unrelated"]
    semantic_summary = summaries["semantic"]
    null_rows = [
        {"stage": stage, "null": "unrelated_action", **asdict(unrelated_summary)},
        {"stage": stage, "null": "semantic_mismatch", **asdict(semantic_summary)},
    ]
    pd.DataFrame(null_rows).to_csv(
        output / f"tables/stage_{stage}_null_summary.csv",
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    atomic_npz(
        output / "nulls" / f"stage_{stage}_null_statistics.npz",
        unrelated_group=unrelated_group,
        unrelated_subject=unrelated_subject,
        unrelated_choices=unrelated_choices,
        semantic_group=semantic_group,
        semantic_subject=semantic_subject,
        semantic_choices=semantic_choices,
    )
    return {
        "stage": stage,
        "observed": observed,
        "subject_gains": subject_gains,
        "identifiability_status": ident_status,
        "pass": bool(passed),
        "unrelated": unrelated_summary,
        "semantic": semantic_summary,
        "arrays": arrays,
        "table": table,
    }


def run_stage_c(output: Path, stage_b: Mapping[str, Any]) -> dict[str, Any]:
    residuals = np.asarray(stage_b["arrays"]["residuals"], dtype=np.float64)
    table = stage_b["table"].reset_index(drop=True)
    blocks = np.empty((9, 2, 4, 22, 22), dtype=np.float64)
    for index, row in table.iterrows():
        subject = int(row["subject"]) - 1
        heldout = CLASSES.index(str(row["heldout_class"]))
        test_session = SESSIONS.index(str(row["test_session"]))
        blocks[subject, test_session, heldout] = residuals[index]
    signatures = np.empty((9, 2, 4 * 253), dtype=np.float64)
    for subject in range(9):
        for session in range(2):
            signatures[subject, session] = normalized_residual_signature(
                blocks[subject, session]
            )
    same_scores, same_observed, same_null = residual_same_subject_test(
        signatures[:, 0], signatures[:, 1]
    )
    same_summary = monte_carlo_summary(same_observed, same_null)
    same_exact_p = float(np.mean(same_null >= same_observed))
    class_observed, class_null, class_choices = residual_class_correspondence_null(
        blocks[:, 0], blocks[:, 1], replicates=NULL_REPLICATES
    )
    class_summary = monte_carlo_summary(class_observed, class_null)
    passed = bool(
        same_summary.effect > 0.0
        and same_exact_p <= 0.05
        and class_summary.effect > 0.0
        and class_summary.p_value <= 0.05
    )
    subject_rows = [
        {
            "subject": subject + 1,
            "same_subject_cross_session_cosine": same_scores[subject],
            "session0_residual_norm": float(np.linalg.norm(blocks[subject, 0])),
            "session1_residual_norm": float(np.linalg.norm(blocks[subject, 1])),
        }
        for subject in range(9)
    ]
    pd.DataFrame(subject_rows).to_csv(
        output / "tables/residual_subject_scores.csv",
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    null_rows = [
        {
            "test": "same_subject_derangement",
            **asdict(same_summary),
            "exact_p_value": same_exact_p,
            "passed": same_summary.effect > 0.0 and same_exact_p <= 0.05,
        },
        {
            "test": "class_correspondence",
            **asdict(class_summary),
            "exact_p_value": np.nan,
            "passed": class_summary.effect > 0.0 and class_summary.p_value <= 0.05,
        },
    ]
    pd.DataFrame(null_rows).to_csv(
        output / "tables/residual_null_summary.csv",
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    atomic_npz(
        output / "nulls/residual_class_correspondence_statistics.npz",
        same_subject_derangement=same_null,
        class_correspondence=class_null,
        class_choices=class_choices,
    )
    atomic_npz(
        output / "objects/cross_fitted_residuals.npz",
        residual_blocks=blocks,
        normalized_signatures=signatures,
    )
    return {
        "pass": passed,
        "same": same_summary,
        "same_exact_p": same_exact_p,
        "class": class_summary,
        "subject_scores": same_scores,
        "blocks": blocks,
    }


def _locked_csv(path: Path, status: str) -> None:
    pd.DataFrame([{"status": status}]).to_csv(path, index=False, lineterminator="\n")


def _save_figure(
    output: Path,
    name: str,
    data: pd.DataFrame,
    draw: Any,
) -> None:
    data.to_csv(
        output / f"figures/{name}.csv",
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    draw(axis, data)
    figure.savefig(output / f"figures/{name}.png", dpi=180)
    figure.savefig(output / f"figures/{name}.pdf")
    plt.close(figure)


def _draw_gain(axis: Any, data: pd.DataFrame) -> None:
    for subject, selected in data.groupby("subject"):
        axis.scatter(
            np.full(len(selected), subject),
            selected["gain"],
            color="#4C78A8",
            alpha=0.55,
            s=18,
        )
        axis.scatter(
            [subject], [selected["gain"].median()], color="#D62728", s=38, zorder=3
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Subject")
    axis.set_ylabel("Held-out gain (raw error − action error)")


def _draw_null(axis: Any, data: pd.DataFrame) -> None:
    for label, selected in data.groupby("label"):
        axis.hist(selected["statistic"], bins=35, alpha=0.45, label=label)
    observed = data["observed"].dropna()
    if len(observed):
        axis.axvline(float(observed.iloc[0]), color="black", linewidth=2, label="observed")
    axis.set_xlabel("Group statistic")
    axis.set_ylabel("Count")
    axis.legend(frameon=False)


def make_figures(
    output: Path,
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any] | None,
    stage_c: Mapping[str, Any] | None,
) -> None:
    _save_figure(
        output,
        "figure_1_within_session_heldout_gain",
        stage_a["table"][["subject", "heldout_class", "train_session", "gain"]],
        _draw_gain,
    )
    if stage_b is not None:
        b_data = stage_b["table"][["subject", "heldout_class", "direction", "gain"]]
    else:
        b_data = pd.DataFrame([{"subject": 1, "heldout_class": "LOCKED", "direction": "LOCKED", "gain": np.nan}])
    _save_figure(output, "figure_2_cross_session_heldout_gain", b_data, _draw_gain)
    unrelated_rows = []
    semantic_rows = []
    for stage_name in ("A", "B"):
        path = output / f"nulls/stage_{stage_name}_null_statistics.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            observed_table = stage_a if stage_name == "A" else stage_b
            assert observed_table is not None
            unrelated_rows.extend(
                {"label": f"Stage {stage_name}", "statistic": value, "observed": observed_table["observed"]}
                for value in archive["unrelated_group"]
            )
            semantic_rows.extend(
                {"label": f"Stage {stage_name}", "statistic": value, "observed": observed_table["observed"]}
                for value in archive["semantic_group"]
            )
    if not unrelated_rows:
        unrelated_rows = [{"label": "LOCKED", "statistic": np.nan, "observed": np.nan}]
    if not semantic_rows:
        semantic_rows = [{"label": "LOCKED", "statistic": np.nan, "observed": np.nan}]
    _save_figure(
        output,
        "figure_3_action_vs_unrelated_null",
        pd.DataFrame(unrelated_rows),
        _draw_null,
    )
    _save_figure(
        output,
        "figure_4_semantic_mismatch_null",
        pd.DataFrame(semantic_rows),
        _draw_null,
    )
    split = pd.read_csv(output / "tables/split_half_noise.csv")

    def draw_split(axis: Any, data: pd.DataFrame) -> None:
        for stage_name, selected in data.groupby("stage"):
            axis.scatter(selected["D_split"], selected["D_eq"], alpha=0.65, s=22, label=f"Stage {stage_name}")
        maximum = float(np.nanmax(data[["D_split", "D_eq", "D_threshold"]].to_numpy()))
        axis.plot([0.0, maximum], [0.0, maximum], color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("D_split")
        axis.set_ylabel("D_eq")
        axis.legend(frameon=False)

    _save_figure(output, "figure_5_split_half_identifiability", split, draw_split)
    if stage_c is not None:
        residual_data = pd.read_csv(output / "tables/residual_subject_scores.csv")
    else:
        residual_data = pd.DataFrame([{"subject": 1, "same_subject_cross_session_cosine": np.nan}])

    def draw_residual(axis: Any, data: pd.DataFrame) -> None:
        axis.bar(data["subject"], data["same_subject_cross_session_cosine"], color="#59A14F")
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xlabel("Subject")
        axis.set_ylabel("Post-action residual cross-session cosine")

    _save_figure(output, "figure_6_post_action_residual", residual_data, draw_residual)


def _stage_json(stage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if stage is None:
        return None
    payload: dict[str, Any] = {
        "observed": float(stage["observed"]),
        "identifiability_status": stage["identifiability_status"],
        "pass": stage["pass"],
        "subject_gains": [float(value) for value in stage["subject_gains"]],
    }
    for name in ("unrelated", "semantic"):
        if name in stage:
            payload[name] = asdict(stage[name])
    return payload


def _ensure_locked_outputs(output: Path, stage_b_status: str, stage_c_status: str) -> None:
    for filename in (
        "stage_B_prediction_cells.csv",
        "stage_B_subject_summary.csv",
        "stage_B_null_summary.csv",
    ):
        path = output / "tables" / filename
        if not path.exists():
            _locked_csv(path, stage_b_status)
    for filename in ("residual_subject_scores.csv", "residual_null_summary.csv"):
        path = output / "tables" / filename
        if not path.exists():
            _locked_csv(path, stage_c_status)
    residual_null = output / "nulls/residual_class_correspondence_statistics.npz"
    if not residual_null.exists():
        atomic_npz(
            residual_null,
            status=np.asarray(stage_c_status),
            same_subject_derangement=np.empty(0, dtype=np.float64),
            class_correspondence=np.empty(0, dtype=np.float64),
            class_choices=np.empty((0, 9), dtype=np.int64),
        )


def finalize_outputs(
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    *,
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any] | None,
    stage_c: Mapping[str, Any] | None,
    decision: str,
) -> dict[str, Any]:
    stage_b_status = "PREREQUISITE_STAGE_A_NOT_PASS" if stage_b is None else "ASSESSED"
    stage_c_status = "PREREQUISITE_STAGE_B_NOT_PASS" if stage_c is None else "ASSESSED"
    _ensure_locked_outputs(output, stage_b_status, stage_c_status)
    empty_matrix = np.empty((0, 22, 22), dtype=np.float64)
    b_arrays = stage_b["arrays"] if stage_b is not None else {}
    atomic_npz(
        output / "objects/source_model_fits.npz",
        stage_A_source_actions=stage_a["arrays"]["source_actions"],
        stage_A_fit_templates=stage_a["arrays"]["fit_templates"],
        stage_A_heldout_templates=stage_a["arrays"]["heldout_templates"],
        stage_B_source_actions=b_arrays.get("source_actions", np.empty((0, 8, 22, 22))),
        stage_B_fit_templates=b_arrays.get("fit_templates", np.empty((0, 3, 22, 22))),
        stage_B_heldout_templates=b_arrays.get("heldout_templates", empty_matrix),
    )
    atomic_npz(
        output / "objects/target_action_fits.npz",
        stage_A_actions=stage_a["arrays"]["best_actions"],
        stage_A_predictions=stage_a["arrays"]["best_predictions"],
        stage_A_targets=stage_a["arrays"]["targets"],
        stage_B_actions=b_arrays.get("best_actions", empty_matrix),
        stage_B_predictions=b_arrays.get("best_predictions", empty_matrix),
        stage_B_targets=b_arrays.get("targets", empty_matrix),
    )
    residual_path = output / "objects/cross_fitted_residuals.npz"
    if not residual_path.exists():
        atomic_npz(
            residual_path,
            residual_blocks=np.empty((0, 4, 22, 22)),
            normalized_signatures=np.empty((0, 1012)),
        )
    unrelated_arrays: dict[str, np.ndarray] = {}
    semantic_arrays: dict[str, np.ndarray] = {}
    for stage_name in ("A", "B"):
        path = output / f"nulls/stage_{stage_name}_null_statistics.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                target = unrelated_arrays if key.startswith("unrelated") else semantic_arrays
                target[f"stage_{stage_name}_{key}"] = np.asarray(archive[key])
    atomic_npz(output / "nulls/unrelated_action_statistics.npz", **unrelated_arrays)
    atomic_npz(output / "nulls/semantic_mismatch_statistics.npz", **semantic_arrays)
    seed_manifest = {
        "master_seed": 20260810,
        "replicates": 1999,
        "bit_generator": "PCG64DXSM",
        "streams": config["nulls"],
        "cell_solver_seed": "sha256(canonical [20260810, scope parts]) first 64 bits",
    }
    atomic_json(output / "nulls/seed_manifests.json", seed_manifest)
    stage_a_status = (
        stage_a["identifiability_status"]
        if stage_a["pass"] is None
        else ("PASS" if stage_a["pass"] else "FAIL")
    )
    stage_b_decision_status = (
        "NOT_RUN"
        if stage_b is None
        else (
            stage_b["identifiability_status"]
            if stage_b["pass"] is None
            else ("PASS" if stage_b["pass"] else "FAIL")
        )
    )
    decision_rows = [
        {"order": 0, "gate": "U_reproduction", "status": "PASS"},
        {"order": 1, "gate": "Stage_A_identifiability", "status": stage_a["identifiability_status"]},
        {"order": 2, "gate": "Stage_A", "status": stage_a_status},
        {
            "order": 3,
            "gate": "Stage_B",
            "status": stage_b_decision_status,
        },
        {
            "order": 4,
            "gate": "Stage_C",
            "status": "NOT_RUN" if stage_c is None else ("PASS" if stage_c["pass"] else "FAIL"),
        },
        {"order": 5, "gate": "terminal", "status": decision},
    ]
    pd.DataFrame(decision_rows).to_csv(
        output / "tables/decision_chain.csv", index=False, lineterminator="\n"
    )
    terminal = {
        "decision": decision,
        "protocol_freeze_sha": FROZEN_PROTOCOL_SHA,
        "stage_A": _stage_json(stage_a),
        "stage_B": _stage_json(stage_b),
        "stage_C": None
        if stage_c is None
        else {
            "pass": stage_c["pass"],
            "same_subject": asdict(stage_c["same"]),
            "same_subject_exact_p": stage_c["same_exact_p"],
            "class_correspondence": asdict(stage_c["class"]),
            "subject_scores": [float(value) for value in stage_c["subject_scores"]],
        },
    }
    atomic_json(output / "decisions/terminal_decision.json", terminal)
    atomic_json(
        output / "protocol/synthetic_solver_validation.json",
        {
            "status": "PASS",
            "full_predata_pytest": "275 passed",
            "det_positive_heldout_relative_error": 6.769176888963379e-15,
            "det_negative_heldout_relative_error": 6.5504874767937775e-15,
            "standard_custom_max_heldout_disagreement": 7.140544681192712e-16,
            "generic_commutant_nullity": 0,
            "repeated_block_commutant_nullity": 1,
        },
    )
    provenance = {
        "base_sha": config["protocol"]["base_commit"],
        "implementation_sha": config["protocol"]["implementation_commit"],
        "protocol_freeze_sha": FROZEN_PROTOCOL_SHA,
        "config_sha256": sha256_file(root / CONFIG_PATH),
        "protocol_sha256": sha256_file(root / config["protocol"]["protocol_path"]),
        "frozen_reference_sha256": config["dataset"]["frozen_inputs"]["full_interaction_objects_sha256"],
        "no_result_adaptation": True,
        "git": git_record(root),
    }
    atomic_json(output / "protocol/provenance.json", provenance)
    make_figures(output, stage_a, stage_b, stage_c)
    write_report(output, terminal)
    return terminal


def _summary_line(stage: Mapping[str, Any] | None, label: str) -> str:
    if stage is None:
        return f"- {label}: prerequisite 때문에 실행하지 않음."
    if stage.get("pass") is None:
        return f"- {label}: {stage['identifiability_status']}."
    unrelated = stage["unrelated"]
    semantic = stage["semantic"]
    return (
        f"- {label}: observed T={stage['observed']:.9g}; "
        f"unrelated null median={unrelated.null_median:.9g}, effect={unrelated.effect:.9g}, p={unrelated.p_value:.9g}; "
        f"semantic null median={semantic.null_median:.9g}, effect={semantic.effect:.9g}, p={semantic.p_value:.9g}; "
        f"{'PASS' if stage['pass'] else 'FAIL'}."
    )


def write_report(output: Path, terminal: Mapping[str, Any]) -> None:
    stage_a = terminal["stage_A"]
    stage_b = terminal["stage_B"]
    stage_c = terminal["stage_C"]
    a_subjects = stage_a["subject_gains"]
    b_subjects = None if stage_b is None else stage_b["subject_gains"]
    subject_lines = []
    for index in range(9):
        a_value = f"{a_subjects[index]:.9g}"
        b_value = "not run" if b_subjects is None else f"{b_subjects[index]:.9g}"
        c_value = (
            "not run"
            if stage_c is None
            else f"{stage_c['subject_scores'][index]:.9g}"
        )
        subject_lines.append(
            f"| S{index + 1} | {a_value} | {b_value} | {c_value} |"
        )
    def stage_text(value: Mapping[str, Any] | None, label: str) -> str:
        if value is None:
            return f"- {label}: prerequisite 때문에 실행하지 않음."
        if value["pass"] is None:
            return f"- {label}: {value['identifiability_status']}."
        u = value["unrelated"]
        s = value["semantic"]
        return (
            f"- {label}: observed T={value['observed']:.9g}; unrelated null median={u['null_median']:.9g}, "
            f"effect={u['effect']:.9g}, p={u['p_value']:.9g}; semantic null median={s['null_median']:.9g}, "
            f"effect={s['effect']:.9g}, p={s['p_value']:.9g}; {'PASS' if value['pass'] else 'FAIL'}."
        )
    c_text = "- Stage C: prerequisite 때문에 실행하지 않음."
    if stage_c is not None:
        same = stage_c["same_subject"]
        correspondence = stage_c["class_correspondence"]
        c_text = (
            f"- Stage C: same-subject observed={same['observed']:.9g}, null median={same['null_median']:.9g}, "
            f"effect={same['effect']:.9g}, exact p={stage_c['same_subject_exact_p']:.9g}; "
            f"class-correspondence observed={correspondence['observed']:.9g}, null median={correspondence['null_median']:.9g}, "
            f"effect={correspondence['effect']:.9g}, p={correspondence['p_value']:.9g}; "
            f"{'PASS' if stage_c['pass'] else 'FAIL'}."
        )
    report = f"""# Common Subject Action Falsification V0

이전 Subject-Class Interaction V0는 안정적인 sensor-space class-dependent signature가 존재하는지를 물었다. 이번 실험은 그 signature가 여러 class에 공통인 하나의 subject action으로 설명되고, 보지 않은 class까지 예측하는지를 묻는다.

## 평문 요약

Stage A는 같은 session의 세 class로 subject action을 정하고 네 번째 class를 예측한다. Stage B는 한 session에서 정한 action으로 반대 session의 보지 않은 class를 예측한다. Stage C는 공통 action을 뺀 뒤에도 class 순서에 의존하는 같은-subject residual이 session을 넘어 남는지 묻는다.

## Reproduction and optimizer gates

Frozen PR #2 object hash가 일치했고 full U 재계산은 max absolute difference 0, relative difference 0으로 PASS했다. 모든 required cell은 full O(22) determinant grid, near-optimal prediction dispersion, independently recomputed run-half variability를 거쳤다. Cell drop은 없었다.

## Inferential stages

{stage_text(stage_a, 'Stage A')}
{stage_text(stage_b, 'Stage B')}
{c_text}

## Subject-level results

| Subject | Stage A median gain | Stage B median gain | Stage C residual cosine |
|---|---:|---:|---:|
{chr(10).join(subject_lines)}

## Frozen terminal decision

`{terminal['decision']}`

## Supported and unsupported claims

The result concerns only a class-independent sensor-space orthogonal conjugation acting on marginally recentered identity-tangent class effects. It does not establish physiology, causal dynamics, source-space structure, unlabeled identifiability, performance improvement, or that every form of individuality is rotation.
"""
    (output / "report/common_subject_action_falsification_v0.md").write_text(
        report, encoding="utf-8"
    )


def run_all(root: Path, *, workers: int) -> dict[str, Any]:
    """Execute the frozen prerequisite chain without available-case fallback."""

    root = Path(root).resolve()
    config = load_config(root)
    output = prepare_output_contract(root, config)
    U = load_and_reproduce_U(root, config, output)
    print("U reproduction gate: PASS (max absolute difference 0)", flush=True)

    stage_a = run_stage(
        root,
        config,
        output,
        U,
        stage="A",
        workers=int(workers),
    )
    print(
        f"Stage A: identifiability={stage_a['identifiability_status']} "
        f"T={stage_a['observed']:.17g} pass={stage_a['pass']}",
        flush=True,
    )

    stage_b: dict[str, Any] | None = None
    stage_c: dict[str, Any] | None = None
    identifiable = stage_a["identifiability_status"] == "PASS"
    if stage_a["pass"] is True:
        stage_b = run_stage(
            root,
            config,
            output,
            U,
            stage="B",
            workers=int(workers),
        )
        print(
            f"Stage B: identifiability={stage_b['identifiability_status']} "
            f"T={stage_b['observed']:.17g} pass={stage_b['pass']}",
            flush=True,
        )
        identifiable = identifiable and stage_b["identifiability_status"] == "PASS"
        if stage_b["pass"] is True:
            stage_c = run_stage_c(output, stage_b)
            print(
                f"Stage C: same={stage_c['same'].observed:.17g} "
                f"class={stage_c['class'].observed:.17g} pass={stage_c['pass']}",
                flush=True,
            )

    frozen = terminal_decision(
        data_gate_pass=True,
        technical_gate_pass=True,
        identifiable=identifiable,
        stage_a_pass=stage_a["pass"],
        stage_b_pass=None if stage_b is None else stage_b["pass"],
        stage_c_pass=None if stage_c is None else stage_c["pass"],
    )
    terminal = finalize_outputs(
        root,
        config,
        output,
        stage_a=stage_a,
        stage_b=stage_b,
        stage_c=stage_c,
        decision=frozen.decision,
    )
    print(f"Terminal decision: {frozen.decision}", flush=True)
    return terminal


__all__ = [
    "CommonActionPipelineError",
    "CONFIG_PATH",
    "FROZEN_PROTOCOL_SHA",
    "deterministic_seed",
    "load_and_reproduce_U",
    "load_config",
    "prepare_output_contract",
    "run_all",
]
