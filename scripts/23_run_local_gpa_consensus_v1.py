#!/usr/bin/env python3
"""Freeze and execute the optimizer-only Local GPA Consensus V1 amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.local_gpa_geometry_v0 as geometry
from src.local_gpa_data_v0 import SCIENTIFIC_BASE_SHA, load_and_reproduce_local_gpa_input
from src.local_gpa_geometry_v0 import CANDIDATE_SETTINGS, GPANumericalError, GPASettings
from src.local_gpa_pipeline_v0 import (
    cell_tasks,
    compute_consensus_distances,
    fit_all_cell_consensuses,
    prepare_locally_centered_cache,
)
from src.local_gpa_statistics_v0 import evaluate_consensus_interaction
from src.reporting_local_gpa_v0 import (
    environment_record,
    implementation_source_hash,
    write_report,
    write_scientific_outputs,
)
from src.trajectory_within_subject_v1 import sha256_file


CONFIG_PATH = ROOT / "configs" / "bnci2014_001_local_gpa_consensus_v1.yaml"
V0_CONFIG_PATH = ROOT / "configs" / "bnci2014_001_local_gpa_consensus_v0.yaml"
PROTOCOL_PATH = ROOT / "docs" / "PROTOCOL_LOCAL_GPA_CONSENSUS_V1_AMENDMENT.md"
V0_PROTOCOL_PATH = ROOT / "docs" / "PROTOCOL_LOCAL_GPA_CONSENSUS_V0.md"
OUTPUT_ROOT = ROOT / "outputs" / "bnci2014_001_local_gpa_consensus_v1"
CACHE_ROOT = ROOT / "cache" / "bnci2014_001_local_gpa_consensus_v1"
V0_FINAL_SHA = "abc4b8127da174709060c82507329bc8c5137069"
V0_FREEZE_SHA = "ad3dd429a69e7cea77e3d4421987989ab45171b9"
V0_FAILURE_SHA = "9aba8a8cee1c84d9b69b7fdf3d53d570927a7cc6"
AUDIT_DEFINITION_SHA = "a87563bd3788754172c0b73699a81a4bb10fb232"
AUDIT_FINAL_SHA = "d846650ea9d4ead205b580d004205327ac1fd3fd"
EXPECTED_FAILED_TARGET_SHA = "42fbff415a91b8fec4941c827338c4e5057dbd8df0e783c617fbd72662eed8f7"
EXPECTED_FAILED_SOURCE_SHA = "324e9aad5de12568fc062f30b528805d24587a1a67884a541cae390024c86a61"


class AmendmentScopeViolation(RuntimeError):
    """V0/V1 differs outside the audited optimizer-only amendment."""


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"config must be a mapping: {path}")
    return value


def load_config() -> tuple[dict[str, Any], GPASettings]:
    config = _load_yaml(CONFIG_PATH)
    if sha256_file(PROTOCOL_PATH) != str(config["protocol"]["sha256"]):
        raise RuntimeError("V1 amendment document SHA-256 differs from config")
    if str(config["protocol"]["scientific_base_sha"]) != SCIENTIFIC_BASE_SHA:
        raise RuntimeError("scientific base SHA changed")
    settings = GPASettings(**config["gpa"]["settings"])
    if settings != CANDIDATE_SETTINGS:
        raise RuntimeError("runtime GPA settings differ from frozen V0 defaults")
    if config["gpa"]["action_optimizer"] != "pymanopt_TrustRegions":
        raise RuntimeError("V1 action optimizer is not TrustRegions")
    trust = config["gpa"]["trust_regions"]
    expected_trust = {
        "hessian": "central_difference_of_exact_Riemannian_gradient",
        "hessian_radius": geometry.TRUST_HESSIAN_RADIUS,
        "gradient_transport": "Stiefel_transport_back_to_base_point",
        "miniter": 3,
        "kappa": 0.1,
        "theta": 1.0,
        "rho_prime": 0.1,
        "use_rand": False,
        "rho_regularization": 1000.0,
        "mininner": 1,
        "maxinner": geometry.TRUST_MAX_INNER_ITERATIONS,
        "max_cost_evaluations": 5000,
    }
    if trust != expected_trust:
        raise RuntimeError("TrustRegions runtime settings differ from audited values")
    return config, settings


def amendment_scope_audit() -> dict[str, Any]:
    v0 = _load_yaml(V0_CONFIG_PATH)
    v1 = _load_yaml(CONFIG_PATH)
    equality_paths = (
        "dataset",
        "input_reproduction",
        "local_centering",
        "quotient",
        "gpa.settings",
        "gpa.action_manifold",
        "gpa.first_and_final_registration",
        "gpa.intermediate_registration",
        "gpa.prototype_parameterization",
        "gpa.global_optimum_claim",
        "split_half",
        "interaction",
        "nulls",
        "runtime",
        "terminal.go",
        "terminal.stop",
        "terminal.technical_failure",
    )

    def at(value: dict[str, Any], dotted: str) -> Any:
        current: Any = value
        for part in dotted.split("."):
            current = current[part]
        return current

    checks = [
        {
            "path": path,
            "equal": at(v0, path) == at(v1, path),
            "v0": at(v0, path),
            "v1": at(v1, path),
        }
        for path in equality_paths
    ]
    optimizer_checks = {
        "v0_optimizer": v0["gpa"]["action_optimizer"],
        "v1_optimizer": v1["gpa"]["action_optimizer"],
        "v1_hessian_radius": v1["gpa"]["trust_regions"]["hessian_radius"],
        "v1_failure_terminal": v1["terminal"]["gpa_failure"],
        "v0_failure_terminal": v0["terminal"]["gpa_failure"],
    }
    allowed = bool(
        all(value["equal"] for value in checks)
        and optimizer_checks["v0_optimizer"]
        == "pymanopt_ConjugateGradient_HestenesStiefel"
        and optimizer_checks["v1_optimizer"] == "pymanopt_TrustRegions"
        and optimizer_checks["v1_failure_terminal"]
        == "UNASSESSED_GPA_NUMERICAL_FAILURE_V1"
        and v1["protocol"]["v0_protocol_freeze_sha"] == V0_FREEZE_SHA
        and v1["protocol"]["v0_failure_sha"] == V0_FAILURE_SHA
        and v1["protocol"]["v0_final_sha"] == V0_FINAL_SHA
        and v1["protocol"]["optimizer_audit_final_sha"] == AUDIT_FINAL_SHA
        and v1["project"]["output_dir"]
        == "outputs/bnci2014_001_local_gpa_consensus_v1"
        and v1["project"]["cache_dir"]
        == "cache/bnci2014_001_local_gpa_consensus_v1"
        and v0["protocol"]["sha256"] == sha256_file(V0_PROTOCOL_PATH)
        and v1["protocol"]["sha256"] == sha256_file(PROTOCOL_PATH)
    )
    record = {
        "status": "PASS" if allowed else "STOP_AMENDMENT_SCOPE_VIOLATION",
        "equal_scientific_fields": checks,
        "allowed_optimizer_and_version_differences": optimizer_checks,
        "allowed_difference_categories": [
            "optimizer identity and audited HVP fields",
            "protocol/version provenance",
            "output and cache namespace",
            "implementation source list",
            "versioned numerical-failure label",
        ],
        "v0_protocol_sha256": sha256_file(V0_PROTOCOL_PATH),
        "v1_amendment_sha256": sha256_file(PROTOCOL_PATH),
    }
    if not allowed:
        raise AmendmentScopeViolation(json.dumps(record, sort_keys=True))
    return record


def _synthetic_configuration(seed: int, d: int, scale: float) -> np.ndarray:
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


def synthetic_validation(settings: GPASettings) -> dict[str, Any]:
    started = time.perf_counter()
    base = _synthetic_configuration(301, 4, 0.12)
    permutation = np.asarray([2, 0, 4, 1, 3])
    cases: list[dict[str, Any]] = []
    for determinant in (-1, 1):
        action = _orthogonal(302 + determinant, 4, determinant)
        target = geometry.conjugate_configuration(base[permutation], action)
        distance, fit = geometry.quotient_distance(target, base, settings=settings)
        cases.append(
            {
                "truth_determinant": determinant,
                "quotient_distance": distance,
                "objective": fit.objective,
                "total_starts": len(fit.starts),
                "converged_components": sorted(
                    {value.determinant for value in fit.starts if value.converged}
                ),
            }
        )
    reverse, _ = geometry.quotient_distance(base, target, settings=settings)
    forward, _ = geometry.quotient_distance(target, base, settings=settings)
    rng = np.random.default_rng(303)
    trials = []
    for trial in range(6):
        action = _orthogonal(400 + trial, 4, -1 if trial % 2 else 1)
        trials.append(geometry.conjugate_configuration(base[rng.permutation(5)], action))
    gpa = geometry.fit_quotient_gpa(
        np.asarray(trials), identity_parts=("v1-pre-data-synthetic",), settings=settings
    )
    truth_distance, _ = geometry.quotient_distance(gpa.prototype, base, settings=settings)
    start_distance, _ = geometry.quotient_distance(
        gpa.starts[0].prototype, gpa.starts[1].prototype, settings=settings
    )
    d22 = _synthetic_configuration(304, 22, 0.02)
    q22 = _orthogonal(305, 22, -1)
    d22_target = geometry.conjugate_configuration(d22[[4, 2, 0, 3, 1]], q22)
    benchmark_started = time.perf_counter()
    d22_fit = geometry.register_configuration(d22_target, d22, seed=306, settings=settings)
    benchmark_seconds = time.perf_counter() - benchmark_started
    record = {
        "status": "PASS",
        "known_Q_Od_and_S5": cases,
        "quotient_symmetry_absolute_difference": abs(forward - reverse),
        "known_answer_GPA": {
            "objective": gpa.objective,
            "distance_to_truth_orbit": truth_distance,
            "distance_between_start_orbits": start_distance,
            "constraint_residual": geometry.constraint_residual(gpa.prototype),
        },
        "d22_registration": {
            "objective": d22_fit.objective,
            "total_starts": len(d22_fit.starts),
            "converged_components": sorted(
                {value.determinant for value in d22_fit.starts if value.converged}
            ),
            "elapsed_seconds": benchmark_seconds,
        },
        "settings": asdict(settings),
        "trust_hessian_radius": geometry.TRUST_HESSIAN_RADIUS,
        "elapsed_seconds": time.perf_counter() - started,
        "stage2a_scientific_statistic_computed": False,
    }
    if (
        max(value["quotient_distance"] for value in cases) > 1.0e-10
        or any(value["total_starts"] != 4 for value in cases)
        or any(value["converged_components"] != [-1, 1] for value in cases)
        or record["quotient_symmetry_absolute_difference"]
        > settings.quotient_symmetry_tolerance
        or gpa.objective > 1.0e-18
        or truth_distance > 1.0e-10
        or start_distance > 1.0e-10
        or len(d22_fit.starts) != 4
        or record["d22_registration"]["converged_components"] != [-1, 1]
    ):
        raise RuntimeError("V1 pre-data synthetic validation failed")
    return record


def failed_registration_gate(
    centered_path: Path, metadata: Any, settings: GPASettings
) -> dict[str, Any]:
    task = next(value for value in cell_tasks(metadata) if value.identity == (1, "0train", "left_hand", "Full"))
    if int(task.indices[0]) != 3:
        raise RuntimeError("frozen failed-registration global index changed")
    bank = np.load(centered_path, mmap_mode="r")
    source = np.asarray(bank[task.indices[0]], dtype=np.float64)
    target = geometry.feasible_prototype_from_configuration(source)
    target_sha = geometry._array_sha256(target)
    source_sha = geometry._array_sha256(source)
    if target_sha != EXPECTED_FAILED_TARGET_SHA or source_sha != EXPECTED_FAILED_SOURCE_SHA:
        raise RuntimeError("exact V0 failed-registration hashes changed")
    started = time.perf_counter()
    fit = geometry.register_configuration(
        target, source, seed=geometry._deterministic_seed(
            "gpa_registration", 1, "0train", "left_hand", "Full", 0, 1, 0
        ), settings=settings
    )
    starts = [
        {
            "start": value.start_index,
            "determinant": value.determinant,
            "objective": value.objective,
            "gradient_norm": value.gradient_norm,
            "iterations": value.optimizer_iterations,
            "alternations": value.alternations,
            "converged": value.converged,
            "stopping_criterion": value.stopping_criterion,
        }
        for value in fit.starts
    ]
    sectors = sorted({value["determinant"] for value in starts if value["converged"]})
    if len(starts) != 4 or sectors != [-1, 1]:
        raise RuntimeError("TrustRegions did not certify both failed-registration sectors")
    return {
        "status": "PASS",
        "identity": {
            "subject": 1,
            "session": "0train",
            "class": "left_hand",
            "split": "Full",
            "gpa_start": 0,
            "gpa_outer_iteration": 1,
            "trial_position": 0,
            "global_sample_index": 3,
            "target_sha256": target_sha,
            "source_sha256": source_sha,
        },
        "starts": starts,
        "converged_components": sectors,
        "elapsed_seconds": time.perf_counter() - started,
        "stage2a_scientific_statistic_computed": False,
    }


def prepare() -> None:
    started = time.perf_counter()
    config, settings = load_config()
    print("CHECKPOINT A: machine-check V0->V1 optimizer-only scope", flush=True)
    scope = amendment_scope_audit()
    print("CHECKPOINT M: TrustRegions synthetic and exact-objective gates", flush=True)
    synthetic = synthetic_validation(settings)
    data = load_and_reproduce_local_gpa_input(ROOT)
    pre_key = hashlib.sha256(
        f"v1-pre-freeze|{sha256_file(CONFIG_PATH)}|TrustRegions".encode()
    ).hexdigest()
    state_path, _, centering = prepare_locally_centered_cache(
        data.states,
        CACHE_ROOT / "pre_run_centering",
        workers=int(config["runtime"]["workers"]),
        cache_key=pre_key,
        settings=settings,
    )
    if int(centering["trial_count"]) != 5184:
        raise RuntimeError("all 5,184 local-center gates did not execute")
    failed_gate = failed_registration_gate(state_path, data.metadata, settings)
    protocol_dir = OUTPUT_ROOT / "protocol"
    tables_dir = OUTPUT_ROOT / "tables"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    data.reproduction_table.to_csv(tables_dir / "frozen_input_reproduction.csv", index=False)
    shutil.copy2(PROTOCOL_PATH, protocol_dir / PROTOCOL_PATH.name)
    shutil.copy2(CONFIG_PATH, protocol_dir / "frozen_config.yaml")
    _write_json(protocol_dir / "amendment_scope_diff.json", scope)
    _write_json(protocol_dir / "synthetic_numerical_validation.json", synthetic)
    _write_json(protocol_dir / "failed_registration_trustregions_gate.json", failed_gate)
    _write_json(protocol_dir / "pre_run_centering_manifest.json", centering)
    _write_json(protocol_dir / "environment.json", environment_record())
    implementation_hash, hashes = implementation_source_hash(
        ROOT, list(config["project"]["implementation_source_files"])
    )
    _write_json(
        protocol_dir / "pre_data_provenance.json",
        {
            **data.provenance,
            "branch": _git("branch", "--show-current"),
            "head_before_v1_freeze": _git("rev-parse", "HEAD"),
            "v0_protocol_freeze_sha": V0_FREEZE_SHA,
            "v0_failure_sha": V0_FAILURE_SHA,
            "v0_final_sha": V0_FINAL_SHA,
            "optimizer_audit_definition_sha": AUDIT_DEFINITION_SHA,
            "optimizer_audit_final_sha": AUDIT_FINAL_SHA,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "implementation_source_sha256": implementation_hash,
            "implementation_file_sha256": hashes,
            "optimizer": "Pymanopt 2.2.1 TrustRegions",
            "elapsed_seconds": time.perf_counter() - started,
            "new_stage2a_scientific_statistics_computed": False,
        },
    )
    print("CHECKPOINT R: frozen input and all 5,184 local centerings PASS", flush=True)
    print("CHECKPOINT F: ready for V1 amendment freeze; no Stage-2A statistic viewed", flush=True)


def run(protocol_freeze_sha: str) -> None:
    started = time.perf_counter()
    config, settings = load_config()
    if _git("status", "--porcelain"):
        raise RuntimeError("V1 scientific execution requires a clean worktree")
    if _git("rev-parse", "HEAD") != protocol_freeze_sha:
        raise RuntimeError("HEAD must equal the V1 protocol-freeze SHA")
    scope = amendment_scope_audit()
    implementation_hash, _ = implementation_source_hash(
        ROOT, list(config["project"]["implementation_source_files"])
    )
    cache_key = hashlib.sha256(
        "|".join(
            (
                protocol_freeze_sha,
                sha256_file(CONFIG_PATH),
                implementation_hash,
                "Pymanopt-2.2.1-TrustRegions",
            )
        ).encode()
    ).hexdigest()
    provenance = {
        "branch": _git("branch", "--show-current"),
        "scientific_source_sha": SCIENTIFIC_BASE_SHA,
        "protocol_freeze_sha": protocol_freeze_sha,
        "v0_protocol_freeze_sha": V0_FREEZE_SHA,
        "v0_failure_sha": V0_FAILURE_SHA,
        "v0_final_sha": V0_FINAL_SHA,
        "optimizer_audit_final_sha": AUDIT_FINAL_SHA,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "implementation_source_sha256": implementation_hash,
        "optimizer": "Pymanopt 2.2.1 TrustRegions",
    }
    try:
        if scope["status"] != "PASS":
            raise AmendmentScopeViolation("amendment scope no longer passes")
        data = load_and_reproduce_local_gpa_input(ROOT)
        state_path, _, centering_manifest = prepare_locally_centered_cache(
            data.states,
            CACHE_ROOT / "scientific_centering",
            workers=int(config["runtime"]["workers"]),
            cache_key=cache_key,
            settings=settings,
        )
        bank = fit_all_cell_consensuses(
            state_path,
            data.metadata,
            CACHE_ROOT / "v1_cell_fits",
            workers=int(config["runtime"]["workers"]),
            cache_key=cache_key,
            settings=settings,
        )
        distances = compute_consensus_distances(
            bank, workers=int(config["runtime"]["workers"]), settings=settings
        )
        result = evaluate_consensus_interaction(
            distances.m01,
            replicates=int(config["nulls"]["replicates"]),
            master_seed=int(config["nulls"]["master_seed"]),
        )
        runtime = time.perf_counter() - started
        terminal = write_scientific_outputs(
            OUTPUT_ROOT,
            bank=bank,
            distances=distances,
            result=result,
            reproduction_table=data.reproduction_table,
            centering_manifest=centering_manifest,
            provenance=provenance,
            runtime_seconds=runtime,
        )
        write_report(
            OUTPUT_ROOT,
            result=result,
            bank=bank,
            distances=distances,
            terminal=terminal,
            provenance=provenance,
            runtime_seconds=runtime,
            tests_summary="pre-run full suite passed; post-run full suite pending",
            report_filename="local_gpa_consensus_v1.md",
            report_title="Local GPA Consensus V1",
            amendment_lines=(
                f"- Immutable V0 freeze: `{V0_FREEZE_SHA}`; V0 terminal: `UNASSESSED_GPA_NUMERICAL_FAILURE`.",
                f"- Optimizer audit final: `{AUDIT_FINAL_SHA}`.",
                "- Exact allowed V0→V1 difference: fixed-action optimizer CG → audited TrustRegions/HVP; all scientific definitions unchanged.",
            ),
        )
        print("CHECKPOINT P1: quotient-consensus interaction complete", flush=True)
        print("CHECKPOINT FINAL: V1 scientific artifacts and report generated", flush=True)
        print(
            json.dumps(
                {
                    "terminal": terminal,
                    "T_J": result.observed.t_j,
                    "p_classbreak": result.p_j_classbreak,
                    "p_subjectbreak": result.p_j_subjectbreak,
                    "runtime_seconds": runtime,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        terminal = (
            "UNASSESSED_GPA_NUMERICAL_FAILURE_V1"
            if isinstance(error, GPANumericalError)
            else (
                "STOP_AMENDMENT_SCOPE_VIOLATION"
                if isinstance(error, AmendmentScopeViolation)
                else "UNASSESSED_TECHNICAL_FAILURE"
            )
        )
        _write_json(
            OUTPUT_ROOT / "decisions" / "technical_failure.json",
            {
                "terminal": terminal,
                "error_type": error.__class__.__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "provenance": provenance,
                "elapsed_seconds": time.perf_counter() - started,
                "scientific_settings_changed_after_access": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol-freeze-sha", required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare()
    else:
        run(arguments.protocol_freeze_sha)


if __name__ == "__main__":
    main()
