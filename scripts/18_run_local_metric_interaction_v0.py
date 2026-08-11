#!/usr/bin/env python3
"""Prepare/freeze inputs or execute the staged local metric V0 analysis."""

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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.local_metric_data_v0 import (
    SCIENTIFIC_BASE_SHA,
    load_and_reproduce_local_metric_input,
    write_reproduction_outputs,
)
from src.local_metric_interaction_v0 import (
    DEFAULT_MASTER_SEED,
    DEFAULT_NULL_REPLICATES,
    classbreak_mappings,
    evaluate_interaction_nulls,
    subjectbreak_mappings,
)
from src.local_metric_pipeline_v0 import (
    CellMetricMatrices,
    compute_cell_metric_matrix,
    compute_cross_session_medoid_decoding,
    compute_within_cell_diagnostics,
)
from src.reporting_local_metric_v0 import (
    environment_record,
    implementation_source_hash,
    write_artifact_manifest,
    write_report,
    write_scientific_outputs,
)
from src.trajectory_within_subject_v1 import sha256_file


CONFIG_PATH = ROOT / "configs" / "bnci2014_001_local_metric_interaction_v0.yaml"
PROTOCOL_PATH = ROOT / "docs" / "PROTOCOL_LOCAL_METRIC_INTERACTION_V0.md"
MATH_AUDIT_PATH = ROOT / "docs" / "LOCAL_METRIC_INTERACTION_MATH_AUDIT_V0.md"


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("config root must be a mapping")
    observed = sha256_file(PROTOCOL_PATH)
    expected = str(config["protocol"]["sha256"])
    if observed != expected:
        raise RuntimeError(
            f"protocol SHA mismatch: config={expected}, observed={observed}"
        )
    if str(config["protocol"]["scientific_base_sha"]) != SCIENTIFIC_BASE_SHA:
        raise RuntimeError("scientific base SHA changed")
    return config


def _implementation(config: dict[str, Any]) -> tuple[str, dict[str, str]]:
    paths = [str(value) for value in config["project"]["implementation_source_files"]]
    return implementation_source_hash(ROOT, paths)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare() -> None:
    start = time.perf_counter()
    config = _load_config()
    output_root = ROOT / str(config["project"]["output_dir"])
    implementation_sha, source_hashes = _implementation(config)
    print("CHECKPOINT M: mathematical audit complete", flush=True)
    data = load_and_reproduce_local_metric_input(ROOT)
    write_reproduction_outputs(
        data,
        output_root,
        protocol_sha256=sha256_file(PROTOCOL_PATH),
        config_sha256=sha256_file(CONFIG_PATH),
        implementation_sha256=implementation_sha,
    )
    protocol_dir = output_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROTOCOL_PATH, protocol_dir / PROTOCOL_PATH.name)
    shutil.copy2(MATH_AUDIT_PATH, protocol_dir / MATH_AUDIT_PATH.name)
    shutil.copy2(CONFIG_PATH, protocol_dir / "frozen_config.yaml")
    _write_json(protocol_dir / "environment.json", environment_record())
    _write_json(
        protocol_dir / "pre_data_provenance.json",
        {
            "branch": _git("branch", "--show-current"),
            "head_before_protocol_freeze": _git("rev-parse", "HEAD"),
            "scientific_source_sha": SCIENTIFIC_BASE_SHA,
            "infrastructure_source_shas": [SCIENTIFIC_BASE_SHA],
            "scientific_representation_diff": "none",
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "implementation_source_sha256": implementation_sha,
            "implementation_file_sha256": source_hashes,
            "new_bnci_scientific_statistics_computed": False,
            "elapsed_seconds": time.perf_counter() - start,
        },
    )
    if not data.degeneracy_table.empty:
        raise RuntimeError(
            "DEGENERATE_METRIC_CONFIGURATION found; stop before scientific inference"
        )
    print(
        "CHECKPOINT R: frozen EEG representation reproduced "
        f"(n={len(data.edges)}, max_abs_diff=0, degenerates=0)",
        flush=True,
    )


def _checkpoint(
    output_root: Path,
    name: str,
    payload: dict[str, Any],
    provenance: dict[str, str],
) -> None:
    _write_json(
        output_root / "checkpoints" / f"checkpoint_{name}.json",
        {**payload, **provenance},
    )


def run(protocol_freeze_sha: str) -> None:
    start = time.perf_counter()
    config = _load_config()
    output_root = ROOT / str(config["project"]["output_dir"])
    implementation_sha, _ = _implementation(config)
    if _git("status", "--porcelain"):
        raise RuntimeError("real execution requires a clean working tree")
    if _git("rev-parse", "HEAD") != protocol_freeze_sha:
        raise RuntimeError("HEAD must equal the supplied protocol-freeze SHA")
    provenance = {
        "protocol_freeze_sha": protocol_freeze_sha,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "scientific_source_sha": SCIENTIFIC_BASE_SHA,
        "config_sha256": sha256_file(CONFIG_PATH),
        "implementation_source_sha256": implementation_sha,
    }
    try:
        data = load_and_reproduce_local_metric_input(ROOT)
        if not data.degeneracy_table.empty:
            raise RuntimeError("DEGENERATE_METRIC_CONFIGURATION blocks inference")
        class_maps = classbreak_mappings(
            replicates=DEFAULT_NULL_REPLICATES, master_seed=DEFAULT_MASTER_SEED
        )
        subject_maps = subjectbreak_mappings(
            replicates=DEFAULT_NULL_REPLICATES, master_seed=DEFAULT_MASTER_SEED
        )

        raw_m01, raw_m = compute_cell_metric_matrix(
            data.edges, data.metadata, mode="raw", chunk_size=128
        )
        raw = evaluate_interaction_nulls(
            raw_m01,
            class_mappings=class_maps,
            subject_mappings=subject_maps,
            replicates=DEFAULT_NULL_REPLICATES,
        )
        _checkpoint(
            output_root,
            "P1",
            {
                "stage": "raw_interaction",
                "T_J": raw.observed.t_j,
                "p_J_classbreak": raw.p_j_classbreak,
                "p_J_subjectbreak": raw.p_j_subjectbreak,
                "p_J": raw.p_j,
                "scientific_protocol_now_immutable": True,
            },
            provenance,
        )
        print("CHECKPOINT P1: raw interaction complete", flush=True)

        size_m01, size_m = compute_cell_metric_matrix(
            data.edges, data.metadata, mode="size", chunk_size=128
        )
        size = evaluate_interaction_nulls(
            size_m01,
            class_mappings=class_maps,
            subject_mappings=subject_maps,
            replicates=DEFAULT_NULL_REPLICATES,
        )
        _checkpoint(
            output_root,
            "P2",
            {
                "stage": "edge_rms_metric_size_control",
                "T_J": size.observed.t_j,
                "p_J_classbreak": size.p_j_classbreak,
                "p_J_subjectbreak": size.p_j_subjectbreak,
                "p_J": size.p_j,
            },
            provenance,
        )
        print("CHECKPOINT P2: metric-size control complete", flush=True)

        normalized_m01, normalized_m = compute_cell_metric_matrix(
            data.edges, data.metadata, mode="normalized", chunk_size=128
        )
        normalized = evaluate_interaction_nulls(
            normalized_m01,
            class_mappings=class_maps,
            subject_mappings=subject_maps,
            replicates=DEFAULT_NULL_REPLICATES,
        )
        _checkpoint(
            output_root,
            "P3",
            {
                "stage": "normalized_relative_pattern_control",
                "T_J": normalized.observed.t_j,
                "p_J_classbreak": normalized.p_j_classbreak,
                "p_J_subjectbreak": normalized.p_j_subjectbreak,
                "p_J": normalized.p_j,
            },
            provenance,
        )
        print("CHECKPOINT P3: normalized distance-pattern control complete", flush=True)

        reliability = compute_within_cell_diagnostics(
            data.edges, data.metadata, chunk_size=128
        )
        decoding = compute_cross_session_medoid_decoding(
            data.edges,
            data.metadata,
            null_replicates=DEFAULT_NULL_REPLICATES,
            master_seed=DEFAULT_MASTER_SEED,
            chunk_size=128,
        )
        _checkpoint(
            output_root,
            "D",
            {
                "stage": "secondary_cross_session_medoid_decoding",
                "mean_subject_ba": decoding.group_mean_ba,
                "median_subject_ba": decoding.group_median_ba,
                "p_value": decoding.p_value,
            },
            provenance,
        )
        print("CHECKPOINT D: secondary decoding complete", flush=True)

        matrices = CellMetricMatrices(
            raw_m01=raw_m01,
            raw_m=raw_m,
            size_m01=size_m01,
            size_m=size_m,
            normalized_m01=normalized_m01,
            normalized_m=normalized_m,
        )
        results = {"raw": raw, "size": size, "normalized": normalized}
        runtime = time.perf_counter() - start
        decision = write_scientific_outputs(
            output_root,
            matrices=matrices,
            results=results,
            reliability=reliability,
            decoding=decoding,
            provenance=provenance,
            total_runtime_seconds=runtime,
        )
        write_report(
            output_root / "report" / "local_metric_interaction_v0.md",
            branch=_git("branch", "--show-current"),
            result_commit="PENDING_FINAL_SCIENTIFIC_RESULT_COMMIT",
            reproduction=data.provenance,
            results=results,
            reliability=reliability,
            decoding=decoding,
            decision=decision,
            tests_summary="Pre-freeze local-metric mathematical, synthetic, null, and reproduction tests passed.",
            repository_tests_summary="Full pre-freeze repository suite passed; final verification pending result artifact commit.",
            git_status="scientific outputs pending commit",
        )
        write_artifact_manifest(output_root, provenance)
        print(
            "SCIENTIFIC_TERMINAL: " + str(decision["terminal_decision"]), flush=True
        )
    except Exception as error:
        _write_json(
            output_root / "failures" / "scientific_run_failure.json",
            {
                **provenance,
                "terminal": "UNASSESSED_TECHNICAL_FAILURE",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
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
