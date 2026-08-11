#!/usr/bin/env python3
"""Prepare/freeze inputs or execute local temporal sequence V0."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry_v2 import airm_mean
from src.local_gpa_data_v0 import (
    LocalGPAReproductionError,
    load_and_reproduce_local_gpa_input,
)
from src.local_temporal_sequence_v0 import (
    ALL_PERMUTATIONS,
    CLASS_ORDER,
    DEFAULT_NULL_REPLICATES,
    DERANGEMENT_MASK,
    NORMALIZED_RESIDUAL_MAX,
    TemporalNumericalError,
    classbreak_mappings,
    compute_common_pca,
    compute_cross_time_matrices,
    compute_split_half_reliability,
    evaluate_temporal_inference,
    fit_ordered_mean_sequences,
    subjectbreak_mappings,
    summarize_matching,
)
from src.reporting_local_temporal_sequence_v0 import (
    environment_record,
    implementation_source_hash,
    write_artifact_manifest,
    write_json,
    write_report,
    write_scientific_outputs,
)
from src.trajectory_within_subject_v1 import sha256_file


CONFIG_PATH = ROOT / "configs" / "bnci2014_001_local_temporal_sequence_v0.yaml"
PROTOCOL_PATH = ROOT / "docs" / "PROTOCOL_LOCAL_TEMPORAL_SEQUENCE_V0.md"
OUTPUT_ROOT = ROOT / "outputs" / "bnci2014_001_local_temporal_sequence_v0"
EXPECTED_BRANCH = "pilot/local-temporal-sequence-correspondence-v0"
LINEAGE = {
    "local_metric_final_sha": "796f04e7970972175a660a521caff47c83e0295f",
    "local_gpa_v1_final_sha": "122eacff868aa8f656ad6360716c1816f453979f",
    "gpa_outer_audit_final_sha": "347d61f17793d636653b614ec2104baa61ac7a4b",
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("config root must be a mapping")
    if sha256_file(PROTOCOL_PATH) != str(config["protocol"]["sha256"]):
        raise RuntimeError("protocol SHA-256 differs from frozen config")
    if str(config["protocol"]["branch"]) != EXPECTED_BRANCH:
        raise RuntimeError("frozen branch changed")
    for key, sha in LINEAGE.items():
        if str(config["lineage"][key]) != sha:
            raise RuntimeError(f"lineage changed: {key}")
        if _git("cat-file", "-t", sha) != "commit":
            raise RuntimeError(f"lineage commit unavailable: {sha}")
    if int(config["nulls"]["replicates"]) != DEFAULT_NULL_REPLICATES:
        raise RuntimeError("null replicate count changed")
    if int(config["matching"]["all_permutations"]) != len(ALL_PERMUTATIONS):
        raise RuntimeError("S5 count changed")
    if int(config["matching"]["derangements"]) != int(np.sum(DERANGEMENT_MASK)):
        raise RuntimeError("derangement count changed")
    return config


def _implementation(config: dict[str, Any]) -> tuple[str, dict[str, str]]:
    paths = [str(value) for value in config["project"]["implementation_source_files"]]
    return implementation_source_hash(ROOT, paths)


def _pre_data_synthetic_validation() -> dict[str, Any]:
    diagonal_logs = np.asarray(
        [
            [-0.3, 0.1, 0.7],
            [0.2, -0.4, 0.5],
            [0.4, 0.6, -0.2],
        ],
        dtype=np.float64,
    )
    matrices = np.asarray([np.diag(np.exp(value)) for value in diagonal_logs])
    result = airm_mean(matrices, tol=1.0e-9, maxiter=100)
    expected = np.diag(np.exp(np.mean(diagonal_logs, axis=0)))
    mean_error = float(np.max(np.abs(result.matrix - expected)))
    k = np.full((36, 36, 5, 5), 3.0, dtype=np.float64)
    diagonal = np.arange(5)
    k[..., diagonal, diagonal] = 1.0
    matching = summarize_matching(k)
    prior_class = classbreak_mappings(replicates=25)
    prior_subject = subjectbreak_mappings(replicates=25)
    record = {
        "status": "PASS",
        "commuting_airm_mean_max_abs_error": mean_error,
        "commuting_airm_mean_normalized_residual": result.normalized_post_residual,
        "known_K_D_id": float(matching.d_id[0, 0]),
        "known_K_median_derangement": float(matching.median_derangement[0, 0]),
        "known_K_G": float(matching.gain[0, 0]),
        "known_K_identity_rank": int(matching.identity_rank[0, 0]),
        "S5_count": len(ALL_PERMUTATIONS),
        "derangement_count": int(np.sum(DERANGEMENT_MASK)),
        "class_mapping_shape": list(prior_class.shape),
        "subject_mapping_shape": list(prior_subject.shape),
        "new_bnci_temporal_statistic_computed": False,
    }
    if (
        result.warning_messages
        or mean_error > 1.0e-12
        or result.normalized_post_residual > NORMALIZED_RESIDUAL_MAX
        or record["known_K_D_id"] != 1.0
        or record["known_K_median_derangement"] != 3.0
        or record["known_K_G"] != 2.0
        or record["known_K_identity_rank"] != 1
    ):
        raise RuntimeError(f"pre-data synthetic validation failed: {record}")
    return record


def prepare() -> None:
    started = time.perf_counter()
    config = _load_config()
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("prepare must run on the requested branch")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError("new output directory is not empty before pre-data preparation")
    implementation_sha, source_hashes = _implementation(config)
    synthetic = _pre_data_synthetic_validation()
    print("CHECKPOINT V: pre-data synthetic validation passed", flush=True)
    data = load_and_reproduce_local_gpa_input(ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_dir = OUTPUT_ROOT / "protocol"
    tables_dir = OUTPUT_ROOT / "tables"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    data.reproduction_table.to_csv(
        tables_dir / "frozen_representation_reproduction.csv", index=False
    )
    shutil.copy2(PROTOCOL_PATH, protocol_dir / PROTOCOL_PATH.name)
    shutil.copy2(CONFIG_PATH, protocol_dir / "frozen_config.yaml")
    write_json(protocol_dir / "environment.json", environment_record())
    write_json(protocol_dir / "pre_data_synthetic_validation.json", synthetic)
    input_provenance = {
        **dict(data.provenance),
        **LINEAGE,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "implementation_source_sha256": implementation_sha,
        "implementation_file_sha256": source_hashes,
        "new_bnci_temporal_statistic_computed": False,
    }
    write_json(protocol_dir / "input_reproduction_provenance.json", input_provenance)
    write_json(
        protocol_dir / "pre_data_provenance.json",
        {
            "branch": _git("branch", "--show-current"),
            "head_before_protocol_freeze": _git("rev-parse", "HEAD"),
            **LINEAGE,
            "scientific_representation_diff": "none",
            "trial_local_centering_applied": False,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "implementation_source_sha256": implementation_sha,
            "implementation_file_sha256": source_hashes,
            "new_bnci_temporal_statistic_computed": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(
        "CHECKPOINT R: frozen covariance objects and AIRM distances reproduced "
        f"(n={len(data.states)}, max_abs_diff={data.provenance['distance_max_abs_diff']})",
        flush=True,
    )


def _failure_terminal(error: Exception) -> str:
    if isinstance(error, LocalGPAReproductionError):
        return "UNASSESSED_TECHNICAL_FAILURE_REPRODUCTION"
    if isinstance(error, TemporalNumericalError):
        return "UNASSESSED_NUMERICAL_FAILURE"
    return "UNASSESSED_TECHNICAL_FAILURE"


def run(protocol_freeze_sha: str) -> None:
    started = time.perf_counter()
    config = _load_config()
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("scientific run must use the requested branch")
    if _git("status", "--porcelain"):
        raise RuntimeError("scientific execution requires a clean working tree")
    if _git("rev-parse", "HEAD") != protocol_freeze_sha:
        raise RuntimeError("HEAD must equal the supplied protocol-freeze SHA")
    implementation_sha, _ = _implementation(config)
    provenance = {
        "protocol_freeze_sha": protocol_freeze_sha,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        **LINEAGE,
        "config_sha256": sha256_file(CONFIG_PATH),
        "implementation_source_sha256": implementation_sha,
    }
    try:
        data = load_and_reproduce_local_gpa_input(ROOT)
        print("CHECKPOINT R2: frozen input reproduced at scientific run", flush=True)
        bank = fit_ordered_mean_sequences(data.states, data.metadata)
        print(
            "CHECKPOINT M: 72 full and 144 split-half ordered mean sequences passed",
            flush=True,
        )
        reliability = compute_split_half_reliability(bank.split)
        k = compute_cross_time_matrices(bank.full)
        matching = summarize_matching(k)
        inference = evaluate_temporal_inference(matching)
        print(
            "CHECKPOINT T: first temporal statistic observed; definitions now immutable "
            f"(T_temporal={inference.observed.t_temporal:.10f}, "
            f"p_temporal={inference.p_temporal:.6f})",
            flush=True,
        )
        pca = compute_common_pca(bank.full)
        runtime = time.perf_counter() - started
        decision = write_scientific_outputs(
            OUTPUT_ROOT,
            bank=bank,
            reliability_cross=reliability,
            matching=matching,
            inference=inference,
            pca=pca,
            provenance=provenance,
            total_runtime_seconds=runtime,
        )
        write_report(
            OUTPUT_ROOT / "report" / "local_temporal_sequence_v0.md",
            branch=_git("branch", "--show-current"),
            protocol_freeze_sha=protocol_freeze_sha,
            final_result_sha="PENDING_FINAL_SCIENTIFIC_RESULT_COMMIT",
            reproduction=data.provenance,
            bank=bank,
            reliability_cross=reliability,
            matching=matching,
            inference=inference,
            pca=pca,
            total_runtime_seconds=runtime,
            focused_tests="pre-freeze temporal suite passed; final verification pending",
            repository_tests="pre-freeze full suite passed; final verification pending",
            git_status="scientific outputs pending commit",
        )
        write_artifact_manifest(OUTPUT_ROOT, provenance)
        print("SCIENTIFIC_TERMINAL: " + str(decision["terminal_decision"]), flush=True)
        print(f"TOTAL_RUNTIME_SECONDS: {runtime:.6f}", flush=True)
    except Exception as error:
        terminal = _failure_terminal(error)
        write_json(
            OUTPUT_ROOT / "decisions" / "scientific_run_failure.json",
            {
                **provenance,
                "terminal": terminal,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
                "automatic_rescue_performed": False,
            },
        )
        print("SCIENTIFIC_TERMINAL: " + terminal, flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--protocol-freeze-sha")
    arguments = parser.parse_args()
    if arguments.prepare:
        prepare()
    else:
        if not arguments.protocol_freeze_sha:
            parser.error("--run requires --protocol-freeze-sha")
        run(str(arguments.protocol_freeze_sha))


if __name__ == "__main__":
    main()
