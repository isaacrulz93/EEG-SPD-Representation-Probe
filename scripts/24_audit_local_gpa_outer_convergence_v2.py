#!/usr/bin/env python3
"""Run the post-failure Local GPA V1 outer-loop numerical forensic."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.local_gpa_data_v0 import load_and_reproduce_local_gpa_input
from src.local_gpa_outer_convergence_audit_v2 import (
    AUDIT_MAX_OUTER_ITERATIONS,
    FROZEN_OUTER_ITERATIONS,
    PROTOTYPE_INITIAL_STEP_CANDIDATES,
    StartTraceResult,
    array_sha256,
    run_gpa_start_trace,
    trace_records,
)
from src.local_gpa_pipeline_v0 import CellTask, cell_tasks
from src.trajectory_within_subject_v1 import sha256_array, sha256_file


BASE_SHA = "122eacff868aa8f656ad6360716c1816f453979f"
V1_FREEZE_SHA = "e6e8887036d1e04a2da7db36d84db58d8c80d9ef"
V1_FAILURE_SHA = "acfe94d805293119092413401e9261853273a1e8"
BRANCH = "audit/local-gpa-outer-convergence-v2"
OUTPUT_ROOT = ROOT / "outputs" / "bnci2014_001_local_gpa_outer_convergence_audit_v2"
TRACE_ROOT = OUTPUT_ROOT / "traces"
BANK_MANIFEST = OUTPUT_ROOT / "protocol" / "audit_bank_selection.json"
STEP_MANIFEST = OUTPUT_ROOT / "protocol" / "prototype_initial_step_candidates.json"
CLASSIFICATION_MANIFEST = OUTPUT_ROOT / "protocol" / "trajectory_classification_contract.json"
V1_CENTERING_ROOT = ROOT / "cache" / "bnci2014_001_local_gpa_consensus_v1" / "scientific_centering"
EXPECTED_CENTERED_SHA256 = "5d99bec906fb3d8827b94088fd2a80984cc89cc68956bdce23ff3504a2c7a146"
EXPECTED_FAILED_IDENTITY = (1, "0train", "left_hand", "Full")
FROZEN_BANK_INDICES = (21, 29, 157, 182)
MILESTONES = (24, 48, 64, 96)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "pyriemann", "pymanopt", "joblib"):
        packages[name] = importlib_metadata.version(name)
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def _validate_scope() -> None:
    if _git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"audit must run on {BRANCH}")
    if _git("merge-base", "HEAD", BASE_SHA) != BASE_SHA:
        raise RuntimeError("audit branch does not descend from immutable V1 final")
    for path in (BANK_MANIFEST, STEP_MANIFEST, CLASSIFICATION_MANIFEST):
        if not path.exists():
            raise RuntimeError(f"missing preregistered audit contract: {path}")
    bank = json.loads(BANK_MANIFEST.read_text(encoding="utf-8"))
    if tuple(bank["selected_task_indices_zero_based"]) != FROZEN_BANK_INDICES:
        raise RuntimeError("outcome-blind bank selection changed")
    steps = json.loads(STEP_MANIFEST.read_text(encoding="utf-8"))
    if tuple(steps["candidate_initial_line_search_steps"]) != PROTOTYPE_INITIAL_STEP_CANDIDATES:
        raise RuntimeError("prototype initial-step candidates changed")


def _load_frozen_cell_bank() -> tuple[np.ndarray, pd.DataFrame, tuple[CellTask, ...]]:
    data = load_and_reproduce_local_gpa_input(ROOT)
    state_path = V1_CENTERING_ROOT / "locally_centered_states.npy"
    manifest_path = V1_CENTERING_ROOT / "local_centering_manifest.json"
    if not state_path.exists() or not manifest_path.exists():
        raise RuntimeError("immutable V1 locally-centered cache is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["centered_states_sha256"] != EXPECTED_CENTERED_SHA256:
        raise RuntimeError("V1 local-centering manifest hash changed")
    states = np.load(state_path, mmap_mode="r")
    if states.shape != (5184, 5, 22, 22):
        raise RuntimeError("V1 locally-centered state shape changed")
    observed_hash = sha256_array(states)
    if observed_hash != EXPECTED_CENTERED_SHA256:
        raise RuntimeError("V1 locally-centered content hash changed")
    tasks = cell_tasks(data.metadata)
    if tasks[0].identity != EXPECTED_FAILED_IDENTITY:
        raise RuntimeError("the first frozen task identity changed")
    selected = tuple(tasks[index] for index in FROZEN_BANK_INDICES)
    expected = tuple(
        (int(value["subject"]), value["session"], value["class_label"], value["split"])
        for value in json.loads(BANK_MANIFEST.read_text(encoding="utf-8"))["selected_tasks"]
    )
    if tuple(value.identity for value in selected) != expected:
        raise RuntimeError("task enumeration differs from frozen bank manifest")
    return states, data.metadata, tasks


def _trace_stem(task: CellTask, initial_step: float, start_index: int) -> str:
    return f"{task.stem}_initial{initial_step:.1f}_start{start_index}"


def _save_result(task: CellTask, result: StartTraceResult) -> Path:
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    path = TRACE_ROOT / f"{_trace_stem(task, result.prototype_initial_step_size, result.start_index)}.csv"
    pd.DataFrame(trace_records(result)).to_csv(path, index=False)
    summary = {
        "identity": list(result.identity_parts),
        "start_index": result.start_index,
        "prototype_initial_step_size": result.prototype_initial_step_size,
        "initial_trial_position": result.initial_trial_position,
        "cell_configuration_sha256": result.cell_configuration_sha256,
        "initial_prototype_sha256": result.initial_prototype_sha256,
        "trial_count": result.trial_count,
        "converged": result.converged,
        "first_convergence_outer_iteration": result.first_convergence_outer_iteration,
        "frozen_24_reproduced_failure": result.frozen_24_reproduced_failure,
        "runtime_seconds": result.runtime_seconds,
        "trace_csv": str(path.relative_to(ROOT)),
    }
    _write_json(path.with_suffix(".summary.json"), summary)
    return path


def _normalized_objective_increases(result: StartTraceResult) -> np.ndarray:
    values = np.asarray([value.joint_final_block_objective for value in result.traces])
    if len(values) < 2:
        return np.asarray([], dtype=np.float64)
    return np.asarray(
        [
            (right - left) / max(1.0, abs(left), abs(right))
            for left, right in zip(values[:-1], values[1:], strict=True)
        ]
    )


def classify_failed_trajectory(result: StartTraceResult) -> str:
    increases = _normalized_objective_increases(result)
    if (
        result.first_convergence_outer_iteration is not None
        and result.first_convergence_outer_iteration > FROZEN_OUTER_ITERATIONS
        and (len(increases) == 0 or float(np.max(increases)) <= 1.0e-7)
    ):
        return "CONVERGING_BUT_BUDGET_LIMITED"
    late_eight = result.traces[-8:]
    late_increases = increases[-7:] if len(increases) else increases
    late_assignment_blocks = sum(value.changed_trial_permutations > 0 for value in late_eight)
    if (
        len(late_increases)
        and float(np.max(late_increases)) > 1.0e-7
        and late_assignment_blocks >= 3
    ):
        return "REGISTRATION_PROTOTYPE_OSCILLATION"
    late = result.traces[-min(5, len(result.traces)):]
    if not result.converged and len(late) >= 2:
        first = late[0].joint_final_block_objective
        last = late[-1].joint_final_block_objective
        decrease = (first - last) / max(1.0, abs(first), abs(last))
        if decrease > 1.0e-5:
            return "STILL_DESCENDING_STRONGLY"
        if (
            max(value.relative_objective_change for value in late) <= 1.0e-7
            and np.median([value.projected_prototype_gradient_norm for value in late]) > 2.0e-5
        ):
            return "OBJECTIVE_PLATEAU_GRADIENT_NOT_CONVERGED"
    return "OTHER_NUMERICAL_INSTABILITY"


def _stable(result: StartTraceResult) -> bool:
    numeric = []
    for value in result.traces:
        numeric.extend(
            (
                value.joint_final_block_objective,
                value.aligned_registration_objective,
                value.projected_prototype_gradient_norm,
                value.prototype_constraint_residual,
            )
        )
    increases = _normalized_objective_increases(result)
    return bool(
        result.converged
        and result.first_convergence_outer_iteration is not None
        and result.first_convergence_outer_iteration <= AUDIT_MAX_OUTER_ITERATIONS
        and np.isfinite(numeric).all()
        and (len(increases) == 0 or float(np.max(increases)) <= 1.0e-7)
    )


def _milestone(first_convergence: int | None) -> dict[str, bool]:
    return {
        f"converged_by_{value}": bool(
            first_convergence is not None and first_convergence <= value
        )
        for value in MILESTONES
    }


def _common_budget(results: list[StartTraceResult]) -> int | None:
    first = [value.first_convergence_outer_iteration for value in results]
    if any(value is None for value in first):
        return None
    maximum = max(int(value) for value in first if value is not None)
    return next((value for value in (48, 64, 96) if maximum <= value), None)


def _accepted_steps_equal(left: StartTraceResult, right: StartTraceResult) -> bool:
    shared = min(len(left.traces), len(right.traces))
    return all(
        left.traces[index].prototype_accepted_step_sizes
        == right.traces[index].prototype_accepted_step_sizes
        for index in range(shared)
    ) and len(left.traces) == len(right.traces)


def _combined_outputs(results: list[tuple[str, CellTask, StartTraceResult]]) -> None:
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for role, task, result in results:
        records = trace_records(result)
        for record in records:
            record["task_role"] = role
            record["subject"] = task.subject
            record["session"] = task.session
            record["class_label"] = task.class_label
            record["split"] = task.split
        rows.extend(records)
        key = _trace_stem(task, result.prototype_initial_step_size, result.start_index)
        arrays[f"{key}_objective"] = np.asarray(
            [value.joint_final_block_objective for value in result.traces]
        )
        arrays[f"{key}_gradient"] = np.asarray(
            [value.projected_prototype_gradient_norm for value in result.traces]
        )
        arrays[f"{key}_relative_change"] = np.asarray(
            [value.relative_objective_change for value in result.traces]
        )
        arrays[f"{key}_changed_permutations"] = np.asarray(
            [value.changed_trial_permutations for value in result.traces]
        )
    pd.DataFrame(rows).to_csv(TRACE_ROOT / "outer_loop_traces.csv", index=False)
    np.savez_compressed(TRACE_ROOT / "outer_loop_traces.npz", **arrays)


def run_audit() -> None:
    audit_started = time.perf_counter()
    _validate_scope()
    states, _, tasks = _load_frozen_cell_bank()
    failed_task = tasks[0]
    failed_configurations = np.asarray(states[failed_task.indices], dtype=np.float64)
    identity_record = {
        "task_index_zero_based": 0,
        "subject": failed_task.subject,
        "session": failed_task.session,
        "class_label": failed_task.class_label,
        "split": failed_task.split,
        "trial_count": len(failed_task.indices),
        "global_sample_indices_sha256": array_sha256(
            failed_task.indices.astype(np.float64)
        ),
        "cell_configuration_sha256": array_sha256(failed_configurations),
        "gpa_start_0_initial_trial_position": 0,
        "gpa_start_1_initial_trial_position": len(failed_configurations) // 2,
        "scientific_statistics_computed": False,
    }
    _write_json(OUTPUT_ROOT / "protocol" / "failed_task_identity.json", identity_record)
    all_results: list[tuple[str, CellTask, StartTraceResult]] = []
    exact: dict[tuple[float, int], StartTraceResult] = {}
    print("AUDIT CHECKPOINT 1: exact failed task, candidates {1,2,4}", flush=True)
    for initial_step in PROTOTYPE_INITIAL_STEP_CANDIDATES:
        for start_index in (0, 1):
            result = run_gpa_start_trace(
                failed_configurations,
                identity_parts=failed_task.identity,
                start_index=start_index,
                max_outer_iterations=AUDIT_MAX_OUTER_ITERATIONS,
                prototype_initial_step_size=initial_step,
                compute_consecutive_quotient_distance=True,
            )
            exact[(initial_step, start_index)] = result
            all_results.append(("exact_failed_task", failed_task, result))
            _save_result(failed_task, result)
            print(
                json.dumps(
                    {
                        "task": failed_task.identity,
                        "initial_step": initial_step,
                        "start": start_index,
                        "first_convergence": result.first_convergence_outer_iteration,
                        "runtime_seconds": result.runtime_seconds,
                    }
                ),
                flush=True,
            )
    default_start0 = exact[(1.0, 0)]
    default_start1 = exact[(1.0, 1)]
    if not default_start0.frozen_24_reproduced_failure:
        raise RuntimeError("frozen V1 start-0 outer failure did not reproduce")
    unlock_bank = bool(_stable(default_start0) and _stable(default_start1))
    bank_results: dict[tuple[int, float, int], StartTraceResult] = {}
    if unlock_bank:
        print("AUDIT CHECKPOINT 2: frozen outcome-blind bank unlocked", flush=True)
        for task_index in FROZEN_BANK_INDICES:
            task = tasks[task_index]
            configurations = np.asarray(states[task.indices], dtype=np.float64)
            for initial_step in PROTOTYPE_INITIAL_STEP_CANDIDATES:
                for start_index in (0, 1):
                    result = run_gpa_start_trace(
                        configurations,
                        identity_parts=task.identity,
                        start_index=start_index,
                        max_outer_iterations=AUDIT_MAX_OUTER_ITERATIONS,
                        prototype_initial_step_size=initial_step,
                        compute_consecutive_quotient_distance=False,
                    )
                    bank_results[(task_index, initial_step, start_index)] = result
                    all_results.append(("outcome_blind_bank", task, result))
                    _save_result(task, result)
                    print(
                        json.dumps(
                            {
                                "task_index": task_index,
                                "initial_step": initial_step,
                                "start": start_index,
                                "first_convergence": result.first_convergence_outer_iteration,
                                "runtime_seconds": result.runtime_seconds,
                            }
                        ),
                        flush=True,
                    )
    else:
        print("AUDIT CHECKPOINT 2: bank NOT unlocked by exact default trajectory", flush=True)
    summaries: list[dict[str, Any]] = []
    for role, task, result in all_results:
        objective = [value.joint_final_block_objective for value in result.traces]
        gradients = [value.projected_prototype_gradient_norm for value in result.traces]
        changed = [value.changed_trial_permutations for value in result.traces if value.changed_trial_permutations >= 0]
        row = {
            "task_role": role,
            "subject": task.subject,
            "session": task.session,
            "class_label": task.class_label,
            "split": task.split,
            "prototype_initial_step_size": result.prototype_initial_step_size,
            "start_index": result.start_index,
            "initial_trial_position": result.initial_trial_position,
            "first_convergence_outer_iteration": result.first_convergence_outer_iteration,
            **_milestone(result.first_convergence_outer_iteration),
            "final_outer_iteration": result.traces[-1].outer_iteration,
            "final_objective": objective[-1],
            "final_gradient": gradients[-1],
            "minimum_gradient": min(gradients),
            "total_changed_permutations": sum(changed),
            "maximum_changed_permutations_in_one_block": max(changed) if changed else 0,
            "total_backtracking_reductions": sum(value.prototype_total_backtracking_reductions for value in result.traces),
            "maximum_normalized_objective_increase": float(np.max(_normalized_objective_increases(result))) if len(result.traces) > 1 else 0.0,
            "stable": _stable(result),
            "runtime_seconds": result.runtime_seconds,
            "classification": classify_failed_trajectory(result) if role == "exact_failed_task" and result.prototype_initial_step_size == 1.0 else "comparative_diagnostic",
        }
        summaries.append(row)
    summary_frame = pd.DataFrame(summaries)
    (OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(OUTPUT_ROOT / "tables" / "outer_convergence_summary.csv", index=False)
    _combined_outputs(all_results)
    default_results = [exact[(1.0, 0)], exact[(1.0, 1)]]
    if unlock_bank:
        default_results.extend(
            bank_results[(task_index, 1.0, start_index)]
            for task_index in FROZEN_BANK_INDICES
            for start_index in (0, 1)
        )
    default_stable = bool(unlock_bank and all(_stable(value) for value in default_results))
    common_budget = _common_budget(default_results) if default_stable else None
    if default_stable and common_budget is not None:
        decision = "RECOMMEND_GPA_OUTER_BUDGET_ONLY_V2_AMENDMENT"
    else:
        decision = "GPA_ALTERNATING_FORMULATION_NUMERICALLY_UNSTABLE"
    initial_step_outcome = "NO_INITIAL_STEP_AMENDMENT_SUPPORTED"
    initial_step_candidate: float | None = None
    if unlock_bank:
        by_candidate: dict[float, list[StartTraceResult]] = {}
        for initial_step in PROTOTYPE_INITIAL_STEP_CANDIDATES:
            values = [exact[(initial_step, 0)], exact[(initial_step, 1)]]
            values.extend(
                bank_results[(task_index, initial_step, start_index)]
                for task_index in FROZEN_BANK_INDICES
                for start_index in (0, 1)
            )
            by_candidate[initial_step] = values
        default_budget = _common_budget(by_candidate[1.0])
        same_as_default = {
            initial_step: all(
                _accepted_steps_equal(left, right)
                for left, right in zip(by_candidate[1.0], by_candidate[initial_step], strict=True)
            )
            for initial_step in (2.0, 4.0)
        }
        if all(same_as_default.values()):
            initial_step_outcome = "INITIAL_STEP_NOT_THE_BOTTLENECK"
        else:
            qualifying = []
            default_changed = sum(
                value.changed_trial_permutations
                for result in by_candidate[1.0]
                for value in result.traces
                if value.changed_trial_permutations >= 0
            )
            for initial_step in (2.0, 4.0):
                values = by_candidate[initial_step]
                budget = _common_budget(values)
                changed = sum(
                    value.changed_trial_permutations
                    for result in values
                    for value in result.traces
                    if value.changed_trial_permutations >= 0
                )
                if (
                    all(_stable(value) for value in values)
                    and budget is not None
                    and default_budget is not None
                    and budget < default_budget
                    and changed <= default_changed
                ):
                    qualifying.append((budget, initial_step))
            if qualifying:
                _, initial_step_candidate = min(qualifying)
                initial_step_outcome = f"INITIAL_STEP_AMENDMENT_CANDIDATE = {initial_step_candidate:.1f}"
    decision_record = {
        "decision": decision,
        "recommended_outer_budget": common_budget,
        "initial_step_outcome": initial_step_outcome,
        "initial_step_amendment_candidate": initial_step_candidate,
        "failed_default_start0_classification": classify_failed_trajectory(default_start0),
        "failed_default_start1_status": (
            "CONVERGED_WITHIN_FROZEN_BUDGET"
            if default_start1.first_convergence_outer_iteration is not None
            and default_start1.first_convergence_outer_iteration <= 24
            else classify_failed_trajectory(default_start1)
        ),
        "exact_default_start0_first_convergence": default_start0.first_convergence_outer_iteration,
        "exact_default_start1_first_convergence": default_start1.first_convergence_outer_iteration,
        "bank_unlocked": unlock_bank,
        "default_bank_stable": default_stable,
        "scientific_statistics_computed": False,
        "stage2a_rerun_performed": False,
        "total_runtime_seconds": time.perf_counter() - audit_started,
    }
    _write_json(OUTPUT_ROOT / "decisions" / "audit_decision.json", decision_record)
    _write_json(
        OUTPUT_ROOT / "protocol" / "audit_provenance.json",
        {
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "v1_protocol_freeze_sha": V1_FREEZE_SHA,
            "v1_failure_sha": V1_FAILURE_SHA,
            "bank_manifest_sha256": sha256_file(BANK_MANIFEST),
            "step_manifest_sha256": sha256_file(STEP_MANIFEST),
            "classification_manifest_sha256": sha256_file(CLASSIFICATION_MANIFEST),
            "centered_states_sha256": EXPECTED_CENTERED_SHA256,
            "environment": _environment(),
            "scientific_settings_changed": False,
            "scientific_statistics_computed": False,
        },
    )
    print(json.dumps(decision_record, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    arguments = parser.parse_args()
    if arguments.command == "run":
        try:
            run_audit()
        except Exception as error:
            _write_json(
                OUTPUT_ROOT / "decisions" / "audit_technical_failure.json",
                {
                    "decision": "UNASSESSED_TECHNICAL_FAILURE",
                    "error_type": error.__class__.__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "scientific_statistics_computed": False,
                    "stage2a_rerun_performed": False,
                },
            )
            raise


if __name__ == "__main__":
    main()
