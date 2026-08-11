"""Checkpointed local centering, cell GPA, and cross-session distances."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.local_gpa_geometry_v0 import (
    CANDIDATE_SETTINGS,
    GPAFit,
    GPANumericalError,
    GPASettings,
    constraint_residual,
    fit_quotient_gpa,
    local_center_configuration,
    quotient_distance,
)
from src.local_metric_interaction_v0 import CLASS_ORDER, N_CELLS, N_SUBJECTS
from src.trajectory_within_subject_v1 import sha256_array


SESSION_ORDER = ("0train", "1test")
HALF_RUNS = {"Full": (0, 1, 2, 3, 4, 5), "A": (0, 2, 4), "B": (1, 3, 5)}


@dataclass(frozen=True)
class CellTask:
    subject: int
    session: str
    class_label: str
    split: str
    indices: np.ndarray

    @property
    def identity(self) -> tuple[object, ...]:
        return (self.subject, self.session, self.class_label, self.split)

    @property
    def stem(self) -> str:
        return f"S{self.subject:02d}_{self.session}_{self.class_label}_{self.split}"


@dataclass(frozen=True)
class ConsensusBank:
    full: np.ndarray
    half_a: np.ndarray
    half_b: np.ndarray
    diagnostics: pd.DataFrame
    registration_diagnostics: pd.DataFrame


@dataclass(frozen=True)
class ConsensusDistances:
    m01: np.ndarray
    m: np.ndarray
    reliability: np.ndarray
    between_diagnostics: pd.DataFrame
    reliability_diagnostics: pd.DataFrame


def cell_tasks(metadata: pd.DataFrame) -> tuple[CellTask, ...]:
    frame = metadata.reset_index(drop=True)
    required = {"subject", "session", "class_label", "run", "global_sample_index"}
    if required - set(frame.columns):
        raise ValueError("metadata lacks the frozen cell identity columns")
    tasks: list[CellTask] = []
    for subject in range(1, N_SUBJECTS + 1):
        for session in SESSION_ORDER:
            for class_label in CLASS_ORDER:
                cell = (
                    frame["subject"].eq(subject)
                    & frame["session"].eq(session)
                    & frame["class_label"].eq(class_label)
                )
                for split, runs in HALF_RUNS.items():
                    indices = frame.index[cell & frame["run"].isin(runs)].to_numpy(
                        dtype=np.int64
                    )
                    expected = 72 if split == "Full" else 36
                    if indices.shape != (expected,):
                        raise ValueError(
                            f"{subject}/{session}/{class_label}/{split} has "
                            f"{len(indices)} trials, expected {expected}"
                        )
                    tasks.append(CellTask(subject, session, class_label, split, indices))
    if len(tasks) != 216:
        raise AssertionError("the frozen full/split GPA grid must contain 216 tasks")
    return tuple(tasks)


def _center_one(states: np.ndarray, settings: GPASettings) -> tuple[np.ndarray, float]:
    centered = local_center_configuration(states, settings=settings)
    return centered.states, centered.normalized_karcher_residual


def prepare_locally_centered_cache(
    states: np.ndarray,
    cache_dir: str | Path,
    *,
    workers: int,
    cache_key: str,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> tuple[Path, Path, dict[str, object]]:
    values = np.asarray(states, dtype=np.float64)
    if values.shape != (5184, 5, 22, 22):
        raise ValueError("frozen state bank has the wrong shape")
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "locally_centered_states.npy"
    residual_path = destination / "local_centering_residuals.npy"
    manifest_path = destination / "local_centering_manifest.json"
    input_hash = sha256_array(values)
    if state_path.exists() and residual_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        centered = np.load(state_path, mmap_mode="r")
        residuals = np.load(residual_path)
        if (
            manifest.get("input_sha256") == input_hash
            and manifest.get("cache_key") == cache_key
            and centered.shape == values.shape
            and residuals.shape == (len(values),)
            and float(np.max(residuals)) <= settings.centering_residual_max
        ):
            return state_path, residual_path, manifest
        raise GPANumericalError("existing local-centering cache provenance mismatch")
    temporary = state_path.with_suffix(".npy.tmp")
    centered_map = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float64, shape=values.shape
    )
    residuals = np.empty(len(values), dtype=np.float64)
    batch_size = max(workers * 4, 32)
    start = time.perf_counter()
    for first in range(0, len(values), batch_size):
        last = min(first + batch_size, len(values))
        results = Parallel(n_jobs=workers, backend="loky")(
            delayed(_center_one)(values[index], settings)
            for index in range(first, last)
        )
        for offset, (centered, residual) in enumerate(results):
            centered_map[first + offset] = centered
            residuals[first + offset] = residual
        centered_map.flush()
    del centered_map
    os.replace(temporary, state_path)
    np.save(residual_path, residuals)
    if float(np.max(residuals)) > settings.centering_residual_max:
        raise GPANumericalError("a real trial failed the frozen local-centering gate")
    centered_hash = sha256_array(np.load(state_path, mmap_mode="r"))
    manifest: dict[str, object] = {
        "input_sha256": input_hash,
        "cache_key": cache_key,
        "centered_states_sha256": centered_hash,
        "trial_count": len(values),
        "maximum_normalized_karcher_residual": float(np.max(residuals)),
        "median_normalized_karcher_residual": float(np.median(residuals)),
        "elapsed_seconds": time.perf_counter() - start,
        "workers": workers,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return state_path, residual_path, manifest


def _near_equivalent(start_objectives: np.ndarray, settings: GPASettings) -> bool:
    best = float(np.min(start_objectives))
    return bool(
        np.max(start_objectives)
        <= best
        + settings.equivalent_objective_atol
        + settings.equivalent_objective_rtol * abs(best)
    )


def _fit_cell(
    centered_state_path: str,
    task: CellTask,
    checkpoint_dir: str,
    cache_key: str,
    settings: GPASettings,
) -> str:
    destination = Path(checkpoint_dir) / f"{task.stem}.npz"
    if destination.exists():
        with np.load(destination, allow_pickle=False) as archive:
            if (
                tuple(archive["identity"].astype(str)) == tuple(map(str, task.identity))
                and str(archive["cache_key"][0]) == cache_key
            ):
                return str(destination)
        raise GPANumericalError(f"cell checkpoint identity mismatch: {destination}")
    bank = np.load(centered_state_path, mmap_mode="r")
    configurations = np.asarray(bank[task.indices], dtype=np.float64)
    start = time.perf_counter()
    fit: GPAFit = fit_quotient_gpa(
        configurations, identity_parts=task.identity, settings=settings
    )
    start_objectives = np.asarray([value.objective for value in fit.starts])
    equivalence_distance = np.nan
    if _near_equivalent(start_objectives, settings):
        equivalence_distance, _ = quotient_distance(
            fit.starts[0].prototype, fit.starts[1].prototype, settings=settings
        )
        if equivalence_distance > settings.gpa_equivalent_orbit_tolerance:
            raise GPANumericalError(
                f"near-equivalent GPA starts disagree by {equivalence_distance:.3e}"
            )
    residual = constraint_residual(fit.prototype)
    if residual > settings.centering_residual_max:
        raise GPANumericalError("returned GPA prototype violates FM(P)=I constraint")
    registrations = fit.best_registrations
    best_actions = np.stack([value.action for value in registrations])
    best_permutations = np.stack([value.permutation for value in registrations])
    all_start_objectives = np.asarray(
        [[start.objective for start in value.starts] for value in registrations]
    )
    all_start_gradients = np.asarray(
        [[start.gradient_norm for start in value.starts] for value in registrations]
    )
    all_start_converged = np.asarray(
        [[start.converged for start in value.starts] for value in registrations]
    )
    all_start_determinants = np.asarray(
        [[start.determinant for start in value.starts] for value in registrations]
    )
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            identity=np.asarray(task.identity, dtype=str),
            cache_key=np.asarray([cache_key]),
            indices=task.indices,
            prototype=fit.prototype,
            objective=np.asarray([fit.objective]),
            within_cell_dispersion=np.asarray([fit.within_cell_dispersion]),
            best_start_index=np.asarray([fit.best_start_index]),
            gpa_start_objectives=start_objectives,
            gpa_start_outer_iterations=np.asarray(
                [value.outer_iterations for value in fit.starts]
            ),
            gpa_start_gradients=np.asarray(
                [value.final_projected_gradient_norm for value in fit.starts]
            ),
            gpa_objective_spread=np.asarray([fit.objective_spread]),
            equivalent_start_orbit_distance=np.asarray([equivalence_distance]),
            prototype_constraint_residual=np.asarray([residual]),
            best_actions=best_actions,
            best_permutations=best_permutations,
            registration_start_objectives=all_start_objectives,
            registration_start_gradients=all_start_gradients,
            registration_start_converged=all_start_converged,
            registration_start_determinants=all_start_determinants,
            elapsed_seconds=np.asarray([time.perf_counter() - start]),
        )
    os.replace(temporary, destination)
    return str(destination)


def fit_all_cell_consensuses(
    centered_state_path: str | Path,
    metadata: pd.DataFrame,
    checkpoint_dir: str | Path,
    *,
    workers: int,
    cache_key: str,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> ConsensusBank:
    tasks = cell_tasks(metadata)
    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = Parallel(n_jobs=workers, backend="loky", verbose=10)(
        delayed(_fit_cell)(
            str(centered_state_path), task, str(destination), cache_key, settings
        )
        for task in tasks
    )
    full = np.empty((2, 9, 4, 5, 22, 22), dtype=np.float64)
    half_a = np.empty_like(full)
    half_b = np.empty_like(full)
    rows: list[dict[str, object]] = []
    registration_rows: list[dict[str, object]] = []
    for task, path in zip(tasks, paths, strict=True):
        with np.load(path, allow_pickle=False) as archive:
            session_index = SESSION_ORDER.index(task.session)
            class_index = CLASS_ORDER.index(task.class_label)
            target = {"Full": full, "A": half_a, "B": half_b}[task.split]
            target[session_index, task.subject - 1, class_index] = archive["prototype"]
            rows.append(
                {
                    "subject": task.subject,
                    "session": task.session,
                    "class_label": task.class_label,
                    "split": task.split,
                    "n_trials": len(task.indices),
                    "objective": float(archive["objective"][0]),
                    "within_cell_procrustes_dispersion": float(
                        archive["within_cell_dispersion"][0]
                    ),
                    "best_gpa_start": int(archive["best_start_index"][0]),
                    "gpa_start_0_objective": float(archive["gpa_start_objectives"][0]),
                    "gpa_start_1_objective": float(archive["gpa_start_objectives"][1]),
                    "gpa_objective_spread": float(archive["gpa_objective_spread"][0]),
                    "gpa_start_0_outer_iterations": int(
                        archive["gpa_start_outer_iterations"][0]
                    ),
                    "gpa_start_1_outer_iterations": int(
                        archive["gpa_start_outer_iterations"][1]
                    ),
                    "gpa_start_0_gradient": float(archive["gpa_start_gradients"][0]),
                    "gpa_start_1_gradient": float(archive["gpa_start_gradients"][1]),
                    "equivalent_start_orbit_distance": float(
                        archive["equivalent_start_orbit_distance"][0]
                    ),
                    "prototype_constraint_residual": float(
                        archive["prototype_constraint_residual"][0]
                    ),
                    "elapsed_seconds": float(archive["elapsed_seconds"][0]),
                    "converged": True,
                }
            )
            objectives = archive["registration_start_objectives"]
            gradients = archive["registration_start_gradients"]
            converged = archive["registration_start_converged"]
            determinants = archive["registration_start_determinants"]
            for trial_position, global_index in enumerate(task.indices):
                for start_index in range(settings.registration_total_starts):
                    registration_rows.append(
                        {
                            "subject": task.subject,
                            "session": task.session,
                            "class_label": task.class_label,
                            "split": task.split,
                            "global_sample_index": int(global_index),
                            "registration_start": start_index,
                            "determinant": int(determinants[trial_position, start_index]),
                            "converged": bool(converged[trial_position, start_index]),
                            "objective": float(objectives[trial_position, start_index]),
                            "gradient_norm": float(gradients[trial_position, start_index]),
                        }
                    )
    return ConsensusBank(
        full=full,
        half_a=half_a,
        half_b=half_b,
        diagnostics=pd.DataFrame(rows),
        registration_diagnostics=pd.DataFrame(registration_rows),
    )


def _distance_task(
    left: np.ndarray,
    right: np.ndarray,
    identity: Sequence[object],
    settings: GPASettings,
) -> tuple[float, float, float, int, float]:
    start = time.perf_counter()
    distance, fit = quotient_distance(left, right, settings=settings)
    converged_fraction = float(np.mean([value.converged for value in fit.starts]))
    return (
        distance,
        fit.objective_spread,
        converged_fraction,
        fit.best_start_index,
        time.perf_counter() - start,
    )


def compute_consensus_distances(
    bank: ConsensusBank,
    *,
    workers: int,
    settings: GPASettings = CANDIDATE_SETTINGS,
) -> ConsensusDistances:
    pairs: list[tuple[int, int]] = [
        (row, column) for row in range(N_CELLS) for column in range(N_CELLS)
    ]
    flat0 = bank.full[0].reshape(N_CELLS, 5, 22, 22)
    flat1 = bank.full[1].reshape(N_CELLS, 5, 22, 22)
    values = Parallel(n_jobs=workers, backend="loky", verbose=10)(
        delayed(_distance_task)(
            flat0[row], flat1[column], ("M01", row, column), settings
        )
        for row, column in pairs
    )
    m01 = np.empty((N_CELLS, N_CELLS), dtype=np.float64)
    between_rows: list[dict[str, object]] = []
    for (row, column), value in zip(pairs, values, strict=True):
        distance, spread, fraction, best, elapsed = value
        m01[row, column] = distance
        between_rows.append(
            {
                "row_cell": row,
                "column_cell": column,
                "distance": distance,
                "registration_objective_spread": spread,
                "registration_converged_fraction": fraction,
                "best_registration_start": best,
                "elapsed_seconds": elapsed,
            }
        )
    reliability_pairs = [(session, cell) for session in range(2) for cell in range(36)]
    a_flat = bank.half_a.reshape(2, 36, 5, 22, 22)
    b_flat = bank.half_b.reshape(2, 36, 5, 22, 22)
    reliability_values = Parallel(n_jobs=workers, backend="loky", verbose=10)(
        delayed(_distance_task)(
            a_flat[session, cell],
            b_flat[session, cell],
            ("reliability", session, cell),
            settings,
        )
        for session, cell in reliability_pairs
    )
    reliability = np.empty((2, 36), dtype=np.float64)
    reliability_rows: list[dict[str, object]] = []
    for (session, cell), value in zip(
        reliability_pairs, reliability_values, strict=True
    ):
        distance, spread, fraction, best, elapsed = value
        reliability[session, cell] = distance
        reliability_rows.append(
            {
                "session": SESSION_ORDER[session],
                "subject": cell // 4 + 1,
                "class_label": CLASS_ORDER[cell % 4],
                "split_half_quotient_distance": distance,
                "registration_objective_spread": spread,
                "registration_converged_fraction": fraction,
                "best_registration_start": best,
                "elapsed_seconds": elapsed,
            }
        )
    return ConsensusDistances(
        m01=m01,
        m=0.5 * (m01 + m01.T),
        reliability=reliability,
        between_diagnostics=pd.DataFrame(between_rows),
        reliability_diagnostics=pd.DataFrame(reliability_rows),
    )


__all__ = [
    "CellTask",
    "ConsensusBank",
    "ConsensusDistances",
    "HALF_RUNS",
    "SESSION_ORDER",
    "cell_tasks",
    "compute_consensus_distances",
    "fit_all_cell_consensuses",
    "prepare_locally_centered_cache",
]
