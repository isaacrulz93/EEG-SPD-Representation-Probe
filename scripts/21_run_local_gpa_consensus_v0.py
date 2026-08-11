#!/usr/bin/env python3
"""Prepare/freeze or execute Stage-2A local quotient GPA consensus V0."""

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

from src.local_gpa_data_v0 import SCIENTIFIC_BASE_SHA, load_and_reproduce_local_gpa_input
from src.local_gpa_geometry_v0 import (
    CANDIDATE_SETTINGS,
    GPASettings,
    conjugate_configuration,
    constraint_residual,
    fit_quotient_gpa,
    quotient_distance,
    register_configuration,
)
from src.local_gpa_pipeline_v0 import (
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


CONFIG_PATH = ROOT / "configs" / "bnci2014_001_local_gpa_consensus_v0.yaml"
PROTOCOL_PATH = ROOT / "docs" / "PROTOCOL_LOCAL_GPA_CONSENSUS_V0.md"
OUTPUT_ROOT = ROOT / "outputs" / "bnci2014_001_local_gpa_consensus_v0"
CACHE_ROOT = ROOT / "cache" / "bnci2014_001_local_gpa_consensus_v0"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_config() -> tuple[dict[str, Any], GPASettings]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("config must be a mapping")
    if sha256_file(PROTOCOL_PATH) != str(config["protocol"]["sha256"]):
        raise RuntimeError("protocol document SHA-256 differs from frozen config")
    if str(config["protocol"]["scientific_base_sha"]) != SCIENTIFIC_BASE_SHA:
        raise RuntimeError("scientific base SHA changed")
    settings = GPASettings(**config["gpa"]["settings"])
    if settings != CANDIDATE_SETTINGS:
        raise RuntimeError("runtime GPA settings differ from tested source defaults")
    return config, settings


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
    start = time.perf_counter()
    base = _synthetic_configuration(301, 4, 0.12)
    permutation = np.asarray([2, 0, 4, 1, 3])
    cases = []
    for determinant in (-1, 1):
        q = _orthogonal(302 + determinant, 4, determinant)
        transformed = conjugate_configuration(base[permutation], q)
        distance, fit = quotient_distance(transformed, base, settings=settings)
        cases.append(
            {
                "determinant": determinant,
                "quotient_distance": distance,
                "objective": fit.objective,
                "both_components_converged": sorted(
                    {value.determinant for value in fit.starts if value.converged}
                )
                == [-1, 1],
            }
        )
    rng = np.random.default_rng(303)
    trials = []
    for trial in range(6):
        q = _orthogonal(400 + trial, 4, -1 if trial % 2 else 1)
        trials.append(conjugate_configuration(base[rng.permutation(5)], q))
    gpa = fit_quotient_gpa(
        np.asarray(trials), identity_parts=("pre-data-synthetic",), settings=settings
    )
    truth_distance, _ = quotient_distance(gpa.prototype, base, settings=settings)
    start_distance, _ = quotient_distance(
        gpa.starts[0].prototype, gpa.starts[1].prototype, settings=settings
    )
    d22 = _synthetic_configuration(304, 22, 0.02)
    q22 = _orthogonal(305, 22, -1)
    d22_target = conjugate_configuration(d22[[4, 2, 0, 3, 1]], q22)
    benchmark_start = time.perf_counter()
    d22_fit = register_configuration(d22_target, d22, seed=306, settings=settings)
    benchmark_seconds = time.perf_counter() - benchmark_start
    record = {
        "status": "PASS",
        "exact_Od_S5_cases": cases,
        "gpa_known_answer": {
            "objective": gpa.objective,
            "distance_to_truth_orbit": truth_distance,
            "distance_between_gpa_start_orbits": start_distance,
            "constraint_residual": constraint_residual(gpa.prototype),
            "outer_iterations": [value.outer_iterations for value in gpa.starts],
        },
        "d22_registration": {
            "objective": d22_fit.objective,
            "total_starts": len(d22_fit.starts),
            "initial_components": [value.determinant for value in d22_fit.starts],
            "elapsed_seconds": benchmark_seconds,
        },
        "settings": asdict(settings),
        "elapsed_seconds": time.perf_counter() - start,
        "new_bnci_stage2a_scientific_statistics_computed": False,
    }
    if (
        max(value["quotient_distance"] for value in cases) > 1.0e-10
        or gpa.objective > 1.0e-18
        or truth_distance > 1.0e-10
        or start_distance > 1.0e-10
        or len(d22_fit.starts) != settings.registration_total_starts
    ):
        raise RuntimeError("pre-data synthetic GPA validation failed")
    return record


def prepare() -> None:
    started = time.perf_counter()
    config, settings = _load_config()
    print("CHECKPOINT M: constrained quotient-GPA mathematical implementation audit", flush=True)
    synthetic = synthetic_validation(settings)
    data = load_and_reproduce_local_gpa_input(ROOT)
    OUTPUT_ROOT.joinpath("protocol").mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.joinpath("tables").mkdir(parents=True, exist_ok=True)
    data.reproduction_table.to_csv(
        OUTPUT_ROOT / "tables" / "frozen_input_reproduction.csv", index=False
    )
    shutil.copy2(PROTOCOL_PATH, OUTPUT_ROOT / "protocol" / PROTOCOL_PATH.name)
    shutil.copy2(CONFIG_PATH, OUTPUT_ROOT / "protocol" / "frozen_config.yaml")
    _write_json(OUTPUT_ROOT / "protocol" / "synthetic_numerical_validation.json", synthetic)
    _write_json(OUTPUT_ROOT / "protocol" / "environment.json", environment_record())
    implementation_hash, source_hashes = implementation_source_hash(
        ROOT, list(config["project"]["implementation_source_files"])
    )
    _write_json(
        OUTPUT_ROOT / "protocol" / "pre_data_provenance.json",
        {
            **data.provenance,
            "branch": _git("branch", "--show-current"),
            "head_before_freeze": _git("rev-parse", "HEAD"),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "implementation_source_sha256": implementation_hash,
            "implementation_file_sha256": source_hashes,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print("CHECKPOINT R: frozen WINDOW5 covariance objects reproduced exactly", flush=True)
    print("CHECKPOINT F: ready for protocol-freeze commit; no Stage-2A cell statistic viewed", flush=True)


def run(protocol_freeze_sha: str) -> None:
    started = time.perf_counter()
    config, settings = _load_config()
    if _git("status", "--porcelain"):
        raise RuntimeError("real execution requires a clean worktree")
    if _git("rev-parse", "HEAD") != protocol_freeze_sha:
        raise RuntimeError("HEAD must equal the supplied protocol-freeze SHA")
    workers = int(config["runtime"]["workers"])
    implementation_hash, _ = implementation_source_hash(
        ROOT, list(config["project"]["implementation_source_files"])
    )
    cache_key = hashlib.sha256(
        "|".join(
            (
                protocol_freeze_sha,
                sha256_file(CONFIG_PATH),
                implementation_hash,
            )
        ).encode()
    ).hexdigest()
    provenance = {
        "branch": _git("branch", "--show-current"),
        "scientific_source_sha": SCIENTIFIC_BASE_SHA,
        "protocol_freeze_sha": protocol_freeze_sha,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "implementation_source_sha256": implementation_hash,
    }
    try:
        data = load_and_reproduce_local_gpa_input(ROOT)
        state_path, _, centering_manifest = prepare_locally_centered_cache(
            data.states,
            CACHE_ROOT,
            workers=workers,
            cache_key=cache_key,
            settings=settings,
        )
        bank = fit_all_cell_consensuses(
            state_path,
            data.metadata,
            CACHE_ROOT / "cell_fits",
            workers=workers,
            cache_key=cache_key,
            settings=settings,
        )
        distances = compute_consensus_distances(
            bank, workers=workers, settings=settings
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
            tests_summary="pre-data suite passed; post-run full suite pending",
        )
        print("CHECKPOINT P1: quotient-consensus interaction complete", flush=True)
        print("CHECKPOINT FINAL: scientific artifacts and report generated", flush=True)
        print(json.dumps({"terminal": terminal, "T_J": result.observed.t_j, "p_classbreak": result.p_j_classbreak, "p_subjectbreak": result.p_j_subjectbreak, "runtime_seconds": runtime}, sort_keys=True), flush=True)
    except Exception as error:
        _write_json(
            OUTPUT_ROOT / "decisions" / "technical_failure.json",
            {
                "terminal": (
                    "UNASSESSED_GPA_NUMERICAL_FAILURE"
                    if error.__class__.__name__ == "GPANumericalError"
                    else "UNASSESSED_TECHNICAL_FAILURE"
                ),
                "error_type": error.__class__.__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "provenance": provenance,
                "elapsed_seconds": time.perf_counter() - started,
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
