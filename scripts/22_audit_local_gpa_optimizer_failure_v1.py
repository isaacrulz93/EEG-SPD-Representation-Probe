#!/usr/bin/env python3
"""Run the optimizer-only Local GPA registration failure audit V1."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymanopt.manifolds import Stiefel
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.local_gpa_geometry_v0 as frozen
from src.local_gpa_data_v0 import load_and_reproduce_local_gpa_input
from src.local_gpa_optimizer_audit_v1 import (
    HESSIAN_RADIUS,
    TRUST_MAX_INNER_ITERATIONS,
    TRUST_MAX_ITERATIONS,
    TRUST_MAX_TIME_SECONDS,
    ForensicRegistration,
    array_sha256,
    riemannian_hessian_vector,
    run_registration_forensic,
)
from src.local_gpa_pipeline_v0 import cell_tasks


OUTPUT = ROOT / "outputs" / "bnci2014_001_local_gpa_optimizer_audit_v1"
CENTERED = ROOT / "cache" / "bnci2014_001_local_gpa_consensus_v0" / "locally_centered_states.npy"
SELECTION = OUTPUT / "protocol" / "audit_bank_selection.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configuration(seed: int, d: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logs = rng.normal(scale=scale, size=(5, d, d))
    logs = 0.5 * (logs + logs.transpose(0, 2, 1))
    logs -= logs.mean(axis=0)
    return np.stack([expm(value) for value in logs])


def _orthogonal(seed: int, d: int, determinant: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    if int(np.sign(np.linalg.det(q))) != determinant:
        q[:, 0] *= -1.0
    return q


def _history_change(history: np.ndarray, count: int) -> float | None:
    if len(history) <= count:
        return None
    return float(history[-count - 1] - history[-1])


def _trajectory_label(history: np.ndarray) -> str:
    if len(history) < 2:
        return "not_determinable_no_iteration_trace"
    window = min(50, len(history) - 1)
    decrease = float(history[-window - 1] - history[-1])
    threshold = max(1.0e-10, 1.0e-6 * abs(float(history[-1])))
    if decrease > threshold:
        return "still_descending_at_termination"
    return "plateaued_by_frozen_descriptive_rule"


def _append_registration(
    registration_id: str,
    fixture_type: str,
    metadata: dict[str, Any],
    result: ForensicRegistration,
    summary_rows: list[dict[str, Any]],
    start_rows: list[dict[str, Any]],
    solve_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
) -> None:
    summary_rows.append(
        {
            "registration_id": registration_id,
            "fixture_type": fixture_type,
            "solver": result.solver,
            **metadata,
            "converged_starts": sum(value.converged for value in result.starts),
            "total_starts": len(result.starts),
            "det_minus_certified": result.determinant_minus_certified,
            "det_plus_certified": result.determinant_plus_certified,
            "both_sectors_certified": result.determinant_minus_certified
            and result.determinant_plus_certified,
            "best_converged_objective": result.best_converged_objective,
            "total_runtime_seconds": result.total_runtime_seconds,
        }
    )
    for start in result.starts:
        history = (
            start.solves[-1].iteration_objectives
            if start.solves
            else np.asarray([], dtype=np.float64)
        )
        start_rows.append(
            {
                "registration_id": registration_id,
                "fixture_type": fixture_type,
                "solver": result.solver,
                **metadata,
                "start_index": start.start_index,
                "initial_determinant": start.initial_determinant,
                "final_determinant": start.final_determinant,
                "starting_permutation": ",".join(map(str, start.starting_permutation)),
                "final_permutation": ",".join(map(str, start.final_permutation)),
                "initial_objective": start.initial_objective,
                "final_objective": start.final_objective,
                "final_gradient_norm": start.final_gradient_norm,
                "gradient_tolerance": 1.0e-6,
                "gradient_ratio_to_tolerance": start.final_gradient_norm / 1.0e-6,
                "optimizer_iterations": start.total_optimizer_iterations,
                "cost_calls_counted": start.total_cost_calls,
                "gradient_calls_counted": start.total_gradient_calls,
                "hessian_calls_counted": start.total_hessian_calls,
                "runtime_seconds": start.total_runtime_seconds,
                "alternations": start.alternations,
                "stopping_criterion": start.stopping_criterion,
                "converged": start.converged,
                "objective_change_last_10_iterations": _history_change(history, 10),
                "objective_change_last_50_iterations": _history_change(history, 50),
                "objective_change_last_100_iterations": _history_change(history, 100),
                "trajectory_classification": _trajectory_label(history),
            }
        )
        for solve in start.solves:
            solve_rows.append(
                {
                    "registration_id": registration_id,
                    "fixture_type": fixture_type,
                    "solver": result.solver,
                    **metadata,
                    "start_index": start.start_index,
                    "alternation": solve.alternation,
                    "permutation_before": ",".join(map(str, solve.permutation_before)),
                    "permutation_after": ",".join(map(str, solve.permutation_after)),
                    "objective_before": solve.objective_before,
                    "objective_after_solve": solve.objective_after_solve,
                    "objective_after_assignment": solve.objective_after_assignment,
                    "gradient_norm": solve.gradient_norm,
                    "iterations": solve.iterations,
                    "cost_evaluations_reported": solve.cost_evaluations_reported,
                    "cost_calls_counted": solve.cost_calls_counted,
                    "gradient_calls_counted": solve.gradient_calls_counted,
                    "hessian_calls_counted": solve.hessian_calls_counted,
                    "runtime_seconds": solve.runtime_seconds,
                    "stopping_criterion": solve.stopping_criterion,
                    "converged": solve.converged,
                }
            )
            for iteration, (objective, gradient) in enumerate(
                zip(
                    solve.iteration_objectives,
                    solve.iteration_gradient_norms,
                    strict=True,
                ),
                start=1,
            ):
                iteration_rows.append(
                    {
                        "registration_id": registration_id,
                        "fixture_type": fixture_type,
                        "solver": result.solver,
                        **metadata,
                        "start_index": start.start_index,
                        "alternation": solve.alternation,
                        "iteration": iteration,
                        "objective": objective,
                        "gradient_norm": gradient,
                        "trace_source": "deterministic_forensic_reproduction_log",
                    }
                )


def _hessian_validation() -> dict[str, Any]:
    rng = np.random.default_rng(9101)
    d = 4
    target = _configuration(9102, d, 0.1)
    source = _configuration(9103, d, 0.1)
    permutation = np.asarray([1, 4, 0, 3, 2])
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    manifold = Stiefel(d, d, retraction="polar")
    eta = manifold.random_tangent_vector(q)
    zeta = manifold.random_tangent_vector(q)
    h_eta = riemannian_hessian_vector(target, source, permutation, q, eta)
    h_zeta = riemannian_hessian_vector(target, source, permutation, q, zeta)
    lhs = float(manifold.inner_product(q, zeta, h_eta))
    rhs = float(manifold.inner_product(q, eta, h_zeta))
    step = 3.0e-4
    f0 = frozen.fixed_registration_objective(target, source, q, permutation)
    fp = frozen.fixed_registration_objective(
        target, source, manifold.retraction(q, step * eta), permutation
    )
    fm = frozen.fixed_registration_objective(
        target, source, manifold.retraction(q, -step * eta), permutation
    )
    second_difference = float((fp - 2.0 * f0 + fm) / step**2)
    hessian_quadratic = float(manifold.inner_product(q, eta, h_eta))
    return {
        "finite_difference_radius": HESSIAN_RADIUS,
        "self_adjoint_lhs": lhs,
        "self_adjoint_rhs": rhs,
        "self_adjoint_absolute_error": abs(lhs - rhs),
        "self_adjoint_relative_error": abs(lhs - rhs)
        / max(abs(lhs), abs(rhs), np.finfo(float).tiny),
        "directional_second_difference_step": step,
        "directional_second_difference": second_difference,
        "hessian_quadratic_form": hessian_quadratic,
        "directional_relative_error": abs(second_difference - hessian_quadratic)
        / max(abs(second_difference), np.finfo(float).tiny),
    }


def main() -> None:
    started = time.perf_counter()
    OUTPUT.joinpath("tables").mkdir(parents=True, exist_ok=True)
    OUTPUT.joinpath("decisions").mkdir(parents=True, exist_ok=True)
    data = load_and_reproduce_local_gpa_input(ROOT)
    bank = np.load(CENTERED, mmap_mode="r")
    tasks = cell_tasks(data.metadata)
    first_task = tasks[0]
    first_trials = np.asarray(bank[first_task.indices])
    failed_target = frozen.feasible_prototype_from_configuration(first_trials[0])
    failed_source = first_trials[0]
    failure_identity = {
        "original_failure_artifact": {
            "path": "outputs/bnci2014_001_local_gpa_consensus_v0/decisions/technical_failure.json",
            "information_available": [
                "exception type",
                "generic both-sector certification failure",
                "elapsed run time",
                "frozen provenance",
            ],
            "registration_identity_available": False,
            "per_start_trace_available": False,
            "parallel_worker_limitation": (
                "the artifact cannot distinguish which worker raised first; "
                "the first Full and half-A tasks begin with the identical "
                "target/source registration hashes reproduced below"
            ),
        },
        "deterministic_forensic_reproduction": {
            "task_order_basis": "first frozen CellTask and first sequential registration within that task",
            "subject": first_task.subject,
            "session": first_task.session,
            "class_label": first_task.class_label,
            "split": first_task.split,
            "gpa_start": 0,
            "gpa_outer_iteration": 1,
            "trial_position": 0,
            "global_sample_index": int(first_task.indices[0]),
            "trial_uid": str(data.metadata.loc[first_task.indices[0], "trial_uid"]),
            "registration_phase": "initial full four-start certification",
            "target_prototype_sha256": array_sha256(failed_target),
            "source_configuration_sha256": array_sha256(failed_source),
            "target_source_are_same_array": bool(
                np.array_equal(failed_target, failed_source)
            ),
            "target_source_max_abs_difference": float(
                np.max(np.abs(failed_target - failed_source))
            ),
        },
    }
    _write_json(OUTPUT / "protocol" / "failure_identity.json", failure_identity)

    summary_rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    solve_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []

    common_failed_metadata = {
        "subject": first_task.subject,
        "session": first_task.session,
        "class_label": first_task.class_label,
        "gpa_start": 0,
        "gpa_outer_iteration": 1,
        "trial_position": 0,
        "global_sample_index": int(first_task.indices[0]),
        "target_hash": array_sha256(failed_target),
        "source_hash": array_sha256(failed_source),
    }
    failed_results = {}
    for solver in ("ConjugateGradient", "TrustRegions"):
        result = run_registration_forensic(
            failed_target, failed_source, solver=solver
        )
        failed_results[solver] = result
        _append_registration(
            "failed_real_registration",
            "frozen_failed_real",
            common_failed_metadata,
            result,
            summary_rows,
            start_rows,
            solve_rows,
            iteration_rows,
        )

    # Existing known-Q fixtures: the frozen d=4 det +/- exact transform and
    # the frozen d=22 det- exact transform used by the V0 synthetic gate.
    synthetic_fixtures = []
    base4 = _configuration(301, 4, 0.12)
    permutation4 = np.asarray([2, 0, 4, 1, 3])
    for determinant in (-1, 1):
        q = _orthogonal(302 + determinant, 4, determinant)
        synthetic_fixtures.append(
            (
                f"known_q_d4_det_{determinant:+d}",
                frozen.conjugate_configuration(base4[permutation4], q),
                base4,
                {"truth_determinant": determinant, "dimension": 4},
            )
        )
    base22 = _configuration(304, 22, 0.02)
    q22 = _orthogonal(305, 22, -1)
    synthetic_fixtures.append(
        (
            "known_q_d22_det_-1",
            frozen.conjugate_configuration(base22[[4, 2, 0, 3, 1]], q22),
            base22,
            {"truth_determinant": -1, "dimension": 22},
        )
    )
    for registration_id, target, source, fixture_metadata in synthetic_fixtures:
        metadata = {
            **fixture_metadata,
            "target_hash": array_sha256(target),
            "source_hash": array_sha256(source),
        }
        for solver in ("ConjugateGradient", "TrustRegions"):
            result = run_registration_forensic(target, source, solver=solver)
            _append_registration(
                registration_id,
                "known_q_synthetic",
                metadata,
                result,
                summary_rows,
                start_rows,
                solve_rows,
                iteration_rows,
            )

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    full_tasks = [value for value in tasks if value.split == "Full"]
    for selected in selection["registrations"]:
        task = full_tasks[int(selected["full_task_index"])]
        trials = np.asarray(bank[task.indices])
        target = frozen.feasible_prototype_from_configuration(trials[0])
        position = int(selected["source_trial_position"])
        source = trials[position]
        metadata = {
            "audit_index": int(selected["audit_index"]),
            "subject": task.subject,
            "session": task.session,
            "class_label": task.class_label,
            "gpa_start": 0,
            "gpa_outer_iteration": 1,
            "trial_position": position,
            "global_sample_index": int(task.indices[position]),
            "target_hash": array_sha256(target),
            "source_hash": array_sha256(source),
        }
        for solver in ("ConjugateGradient", "TrustRegions"):
            result = run_registration_forensic(target, source, solver=solver)
            _append_registration(
                f"real_audit_{selected['audit_index']}",
                "deterministic_real_audit_bank",
                metadata,
                result,
                summary_rows,
                start_rows,
                solve_rows,
                iteration_rows,
            )

    summary = pd.DataFrame(summary_rows)
    starts = pd.DataFrame(start_rows)
    solves = pd.DataFrame(solve_rows)
    iterations = pd.DataFrame(iteration_rows)
    summary.to_csv(OUTPUT / "tables" / "registration_summary.csv", index=False)
    starts.to_csv(OUTPUT / "tables" / "four_start_forensic.csv", index=False)
    solves.to_csv(OUTPUT / "tables" / "alternation_solve_trace.csv", index=False)
    iterations.to_csv(OUTPUT / "tables" / "cg_iteration_trace.csv", index=False)
    hessian = _hessian_validation()
    _write_json(OUTPUT / "protocol" / "trust_regions_hessian_validation.json", hessian)

    failed_cg = failed_results["ConjugateGradient"]
    failed_tr = failed_results["TrustRegions"]
    synthetic = summary.loc[summary["fixture_type"].eq("known_q_synthetic")]
    real_bank = summary.loc[
        summary["fixture_type"].eq("deterministic_real_audit_bank")
    ]
    trust_synthetic = synthetic.loc[synthetic["solver"].eq("TrustRegions")]
    trust_real = real_bank.loc[real_bank["solver"].eq("TrustRegions")]
    trust_pass = bool(
        failed_tr.determinant_minus_certified
        and failed_tr.determinant_plus_certified
        and trust_synthetic["both_sectors_certified"].astype(bool).all()
        and trust_real["both_sectors_certified"].astype(bool).all()
        and hessian["self_adjoint_relative_error"] <= 1.0e-5
        and hessian["directional_relative_error"] <= 1.0e-4
    )
    decision = {
        "decision": (
            "RECOMMEND_GPA_OPTIMIZER_ONLY_V1_AMENDMENT"
            if trust_pass
            else "CURRENT_GPA_FORMULATION_NUMERICALLY_UNRESOLVED"
        ),
        "failed_registration": {
            "cg_converged_starts": sum(value.converged for value in failed_cg.starts),
            "cg_both_sectors_certified": failed_cg.determinant_minus_certified
            and failed_cg.determinant_plus_certified,
            "trust_regions_converged_starts": sum(
                value.converged for value in failed_tr.starts
            ),
            "trust_regions_both_sectors_certified": failed_tr.determinant_minus_certified
            and failed_tr.determinant_plus_certified,
        },
        "known_q_synthetic": {
            "trust_regions_certified_registrations": int(
                trust_synthetic["both_sectors_certified"].sum()
            ),
            "total_registrations": len(trust_synthetic),
        },
        "deterministic_real_audit_bank": {
            "trust_regions_certified_registrations": int(
                trust_real["both_sectors_certified"].sum()
            ),
            "total_registrations": len(trust_real),
        },
        "hessian_validation": hessian,
        "trust_regions_candidate_settings": {
            "manifold": "Stiefel(d,d,retraction=polar)",
            "gradient_tolerance": 1.0e-6,
            "max_iterations": TRUST_MAX_ITERATIONS,
            "max_time_seconds": TRUST_MAX_TIME_SECONDS,
            "max_inner_iterations": TRUST_MAX_INNER_ITERATIONS,
            "finite_difference_hessian_radius": HESSIAN_RADIUS,
            "scientific_objective_changed": False,
        },
        "scientific_stage2a_statistics_computed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(OUTPUT / "decisions" / "audit_decision.json", decision)
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
