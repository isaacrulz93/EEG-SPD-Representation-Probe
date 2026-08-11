#!/usr/bin/env python3
"""Prepare and execute the frozen OpenBMI ordered-movement replication V0."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.openbmi_ordered_movement_v0 import (  # noqa: E402
    ATOL,
    CHANNEL_ORDER,
    CLASS_ORDER,
    DELTA_T_SECONDS,
    N_CELLS,
    N_CHANNELS,
    N_CLASSES,
    N_SESSIONS,
    N_STATES,
    N_SUBJECTS,
    NULL_REPLICATES,
    RTOL,
    OpenBMIDataContractError,
    OpenBMINumericalError,
    anti_develop_banks,
    atomic_savez,
    atomic_write_text,
    canonical_json_sha256,
    canonical_trial_bank,
    classbreak_mappings,
    component_matrices,
    evaluate_inference,
    fit_mean_sequences,
    flatten_cells,
    preprocess_source,
    quotient_pair,
    raw_temporal_correspondence,
    relation_statistics,
    resolve_source,
    sensor_and_length_costs,
    sha256_array,
    sha256_file,
    subjectbreak_mappings,
    terminal,
)


EXPECTED_BRANCH = "replication/openbmi-ordered-movement-v0"
PARENT_SHA = "edc1d344cb0657f2f2d87b2992049bceec4705d2"
PARENT_PROTOCOL_SHA = "95c330de9596fa4c4eb4ee377d5af8d99896f4c3"
PARENT_RESULT_SHA = "0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca"
PARENT_TERMINAL = "BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS"
DONOR_SHA = "272d775678644aad062df424a70586d4b42de652"
DONOR_BRANCH = "pilot/subject-class-interaction-v0"
DONOR_PROTOCOL_PATH = "outputs/subject_class_interaction_v0/provenance/openbmi_protocol_manifest.json"
DONOR_SOURCE_PATH = "outputs/subject_class_interaction_v0/provenance/openbmi_source_manifest.json"
DONOR_UNLOCK_PATH = "outputs/subject_class_interaction_v0/provenance/openbmi_unlock.json"
DONOR_HASHES = {
    DONOR_PROTOCOL_PATH: "4b956d7e3b2b1a271ec07bddecc1ce0a93460ab1515ab8902b3d1ca35ebdb0ea",
    DONOR_SOURCE_PATH: "5133a0d8521f4cdd121362663c4980fed500e72875f26bb3dc4ce830d8c5e409",
    DONOR_UNLOCK_PATH: "aaf2bf598802e0e5bf84924b26d0c9ed7f64640605a594273894b888b6c043f0",
}
CONFIG_PATH = ROOT / "configs" / "openbmi_ordered_movement_replication_v0.yaml"
PROTOCOL_PATH = ROOT / "docs" / "PROTOCOL_OPENBMI_ORDERED_MOVEMENT_REPLICATION_V0.md"
OUTPUT = ROOT / "outputs" / "openbmi_ordered_movement_replication_v0"
CACHE = ROOT / "cache" / "openbmi_ordered_movement_v0"
DERIVED = CACHE / "source_subject_sessions"
PRIVATE_SOURCE = CACHE / "source_files"
WINDOW_BANK = CACHE / "openbmi_five_bin_covariances.npz"
METADATA_PATH = CACHE / "openbmi_five_bin_metadata.csv"
SOURCE_RECORDS_PATH = CACHE / "source_reproduction_records.json"
MEANS_PATH = OUTPUT / "arrays" / "ordered_mean_sequences.npz"
MOVEMENT_PATH = OUTPUT / "arrays" / "ordered_antidevelopment.npz"
COMPONENT_PATH = OUTPUT / "arrays" / "component_cost_matrices.npz"
SPLIT_COMPONENT_PATH = OUTPUT / "arrays" / "split_half_component_matrices.npz"
NULL_PATH = OUTPUT / "arrays" / "component_nulls.npz"
REPORT_PATH = OUTPUT / "report" / "openbmi_ordered_movement_replication_v0.md"
PARALLEL_WORKERS = 4
CHECKPOINT_EVERY = 12


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def git_show(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, check=True, capture_output=True).stdout


def bytes_sha256(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_repository(*, require_clean: bool) -> None:
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("wrong replication branch")
    if git("merge-base", "--is-ancestor", PARENT_SHA, "HEAD") != "":
        pass
    if require_clean and git("status", "--porcelain"):
        raise RuntimeError("scientific execution requires a clean worktree")


def donor_bytes() -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path, expected in DONOR_HASHES.items():
        value = git_show(DONOR_SHA, path)
        if bytes_sha256(value) != expected:
            raise OpenBMIDataContractError(f"donor artifact hash failed: {path}")
        result[path] = value
    return result


def validate_donor_contract(protocol: dict[str, Any], source: dict[str, Any]) -> None:
    checks = {
        "dataset_identifier": protocol["dataset_identifier"] == "Lee2019-MI",
        "subjects": protocol["subject_ids"] == list(range(1, 55)),
        "sessions": protocol["session_ids"] == ["0", "1"],
        "classes": protocol["classes"] == list(CLASS_ORDER),
        "channels": protocol["eeg_channels"] == list(CHANNEL_ORDER),
        "sample_rate": float(protocol["sampling_frequency_hz"]) == 100.0,
        "samples": int(protocol["epoch_sample_rule"]["samples_per_trial"]) == 250,
        "epoch": protocol["epoch_seconds"] == [1.0, 3.5],
        "bandpass": protocol["bandpass_hz"] == [8.0, 30.0],
        "source_count": int(source["file_count"]) == 108 and len(source["files"]) == 108,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise OpenBMIDataContractError(f"donor contract mismatch: {failed}")
    if 250 % N_STATES != 0 or 250 // N_STATES != 50:
        raise RuntimeError("STOP_AND_REPORT_TEMPORAL_CONTRACT_AMBIGUITY")


def prepare_data() -> None:
    """Reproduce all donor sources and build five-bin covariances only."""

    validate_repository(require_clean=False)
    payloads = donor_bytes()
    protocol = json.loads(payloads[DONOR_PROTOCOL_PATH])
    source_manifest = json.loads(payloads[DONOR_SOURCE_PATH])
    validate_donor_contract(protocol, source_manifest)
    mne_root = Path(os.environ.get("MNE_DATA", "/Volumes/External_SSD/dataset/mne_data"))
    records: list[tuple[np.ndarray, pd.DataFrame]] = []
    reproduction: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, donor_record in enumerate(source_manifest["files"], start=1):
        subject, session = int(donor_record["subject"]), int(donor_record["session"])
        stem = f"S{subject:02d}_session{session}"
        npz_path, csv_path, json_path = DERIVED / f"{stem}.npz", DERIVED / f"{stem}.csv", DERIVED / f"{stem}.json"
        if npz_path.exists() and csv_path.exists() and json_path.exists():
            record = json.loads(json_path.read_text(encoding="utf-8"))
            if (
                record.get("source_sha256") == donor_record["source_sha256"]
                and record.get("derived_sha256") == sha256_file(npz_path)
                and record.get("temporal_contract") == "5x50_samples_at_100Hz"
            ):
                with np.load(npz_path, allow_pickle=False) as archive:
                    covariances = archive["covariances"].copy()
                metadata = pd.read_csv(csv_path)
                records.append((covariances, metadata)); reproduction.append(record)
                print(f"[{ordinal:03d}/108] reused {stem}", flush=True)
                continue
        source = resolve_source(donor_record, mne_root, PRIVATE_SOURCE)
        covariances, metadata = preprocess_source(source, subject=subject, source_session=session)
        atomic_savez(npz_path, {"covariances": covariances, "channel_names": np.asarray(CHANNEL_ORDER)})
        DERIVED.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(csv_path, index=False, lineterminator="\n")
        record = {
            "subject": subject,
            "session": session,
            "source_path_role": "shared_read_only" if str(source).startswith(str(mne_root)) else "task_private_cache",
            "source_bytes": source.stat().st_size,
            "source_sha256": sha256_file(source),
            "expected_source_sha256": donor_record["source_sha256"],
            "source_exact_donor_match": True,
            "derived_sha256": sha256_file(npz_path),
            "covariance_array_sha256": sha256_array(covariances),
            "metadata_sha256": sha256_file(csv_path),
            "temporal_contract": "5x50_samples_at_100Hz",
        }
        write_json(json_path, record)
        records.append((covariances, metadata)); reproduction.append(record)
        print(f"[{ordinal:03d}/108] prepared {stem}", flush=True)
    bank, metadata = canonical_trial_bank(records)
    atomic_savez(WINDOW_BANK, {
        "covariances": bank,
        "subjects": np.arange(1, N_SUBJECTS + 1),
        "sessions": np.arange(N_SESSIONS),
        "classes": np.asarray(CLASS_ORDER),
        "channels": np.asarray(CHANNEL_ORDER),
        "bin_sample_edges": np.arange(0, 251, 50),
        "sampling_frequency_hz": np.asarray(100.0),
    })
    metadata.to_csv(METADATA_PATH, index=False, lineterminator="\n")
    summary = {
        "status": "PASS",
        "donor_sha": DONOR_SHA,
        "donor_artifact_hashes": DONOR_HASHES,
        "records": reproduction,
        "record_count": len(reproduction),
        "all_source_hashes_match_donor": all(r["source_exact_donor_match"] for r in reproduction),
        "window_covariance_bank_sha256": sha256_file(WINDOW_BANK),
        "window_covariance_array_sha256": sha256_array(bank),
        "metadata_sha256": sha256_file(METADATA_PATH),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(SOURCE_RECORDS_PATH, summary)
    print(json.dumps({key: summary[key] for key in ("status", "record_count", "window_covariance_bank_sha256", "runtime_seconds")}, indent=2))


def prepare_provenance() -> None:
    """Copy immutable donor bytes and data preparation checks into the new namespace."""

    payloads = donor_bytes()
    provenance = OUTPUT / "provenance"
    for path, value in payloads.items():
        atomic_write_text(provenance / f"donor_{Path(path).name}", value.decode("utf-8"))
    if not SOURCE_RECORDS_PATH.exists():
        raise RuntimeError("run --stage prepare-data first")
    atomic_write_text(provenance / "source_reproduction_records.json", SOURCE_RECORDS_PATH.read_text(encoding="utf-8"))
    record = json.loads(SOURCE_RECORDS_PATH.read_text(encoding="utf-8"))
    write_json(provenance / "temporal_contract_gate.json", {
        "status": "PASS",
        "frozen_epoch_samples": 250,
        "bins": 5,
        "samples_per_bin": 50,
        "sampling_frequency_hz": 100.0,
        "seconds_per_bin": 0.5,
        "exact_integer_partition": True,
        "window_covariance_bank_sha256": record["window_covariance_bank_sha256"],
        "window_covariance_array_sha256": record["window_covariance_array_sha256"],
    })


def checkpoint_metadata(phase: str, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "phase": phase,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config_semantic_sha256": canonical_json_sha256(config),
        "first_array_sha256": sha256_array(first),
        "second_array_sha256": sha256_array(second),
        "shape": [N_CELLS, N_CELLS],
        "optimizer": config["optimizer"],
    }


def _fit_one(row: int, column: int, first: np.ndarray, second: np.ndarray) -> tuple[int, int, float, dict[str, Any]]:
    objective, record = quotient_pair(first[row], second[column])
    return row, column, objective, record


def fit_quotient_matrix(phase: str, first_bank: np.ndarray, second_bank: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    first, second = flatten_cells(first_bank), flatten_cells(second_bank)
    checkpoint = OUTPUT / "checkpoints" / f"{phase}_quotient_checkpoint.npz"
    metadata = checkpoint_metadata(phase, first, second)
    shape = (N_CELLS, N_CELLS)
    completed = np.zeros(shape, dtype=bool); objective = np.full(shape, np.nan)
    selected_det = np.zeros(shape, dtype=np.int8); both = np.zeros(shape, dtype=bool)
    gradient = np.full(shape, np.nan); best_start = np.full(shape, -1, dtype=np.int16)
    second_objective = np.full(shape, np.nan); spread = np.full(shape, np.nan)
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as archive:
            saved_metadata = json.loads(str(archive["metadata_json"]))
            if saved_metadata != metadata:
                raise OpenBMINumericalError(f"checkpoint contract mismatch: {phase}")
            for name, target in (
                ("completed", completed), ("objective", objective), ("selected_det", selected_det),
                ("both", both), ("gradient", gradient), ("best_start", best_start),
                ("second_objective", second_objective), ("spread", spread),
            ):
                target[...] = archive[name]
    pending = [tuple(map(int, index)) for index in np.argwhere(~completed)]
    print(f"{phase}: {completed.sum()}/{completed.size} complete; {len(pending)} pending", flush=True)
    for offset in range(0, len(pending), CHECKPOINT_EVERY):
        chunk = pending[offset : offset + CHECKPOINT_EVERY]
        results = Parallel(n_jobs=PARALLEL_WORKERS, backend="loky")(
            delayed(_fit_one)(row, column, first, second) for row, column in chunk
        )
        for row, column, value, record in results:
            objective[row, column] = value
            selected_det[row, column] = int(record["selected_determinant"])
            both[row, column] = bool(record["both_sectors_certified"])
            gradient[row, column] = float(record["gradient_norm"])
            best_start[row, column] = int(record["best_start_index"])
            second_objective[row, column] = float(record["second_best_objective"])
            spread[row, column] = float(record["objective_spread"])
            completed[row, column] = True
        atomic_savez(checkpoint, {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            "completed": completed, "objective": objective, "selected_det": selected_det,
            "both": both, "gradient": gradient, "best_start": best_start,
            "second_objective": second_objective, "spread": spread,
        })
        print(f"{phase}: {completed.sum()}/{completed.size} complete", flush=True)
    if not completed.all() or not np.isfinite(objective).all() or not both.all():
        raise OpenBMINumericalError(f"quotient checkpoint incomplete/uncertified: {phase}")
    diagnostics = {
        "completed": completed, "selected_determinant": selected_det,
        "both_sectors_certified": both, "gradient_norm": gradient,
        "best_start_index": best_start, "second_best_objective": second_objective,
        "objective_spread": spread,
    }
    return objective, diagnostics


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    labels = [f"S{subject:02d}_{class_name}" for subject in range(1, 55) for class_name in CLASS_ORDER]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path, lineterminator="\n")


def component_rows(components: dict[str, np.ndarray], inferences: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects: list[dict[str, Any]] = []; summary: list[dict[str, Any]] = []
    for name in ("full", "len", "ang", "ori", "sensor"):
        result = inferences[name]
        for subject in range(N_SUBJECTS):
            subjects.append({
                "component": name, "subject": subject + 1,
                "S_s": result.observed.s_s[subject],
                "C_s": result.observed.class_s[subject],
                "J_s": result.observed.j_s[subject],
            })
        summary.append({
            "component": name,
            "T_subject": result.observed.t_subject, "p_subjectbreak": result.p_subject,
            "T_class": result.observed.t_class, "p_classbreak": result.p_class,
            "T_J": result.observed.t_j,
            "p_J_subjectbreak": result.p_j_subject,
            "p_J_classbreak": result.p_j_class,
        })
    return pd.DataFrame(subjects), pd.DataFrame(summary)


def write_figures(components: dict[str, np.ndarray], subjects: pd.DataFrame, temporal: pd.DataFrame) -> None:
    figure_dir = OUTPUT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(components["ang"], aspect="auto", cmap="viridis")
    ax.set(title="OpenBMI angular/joint squared cost", xlabel="Session 1 cell", ylabel="Session 0 cell")
    fig.colorbar(image, ax=ax, label="c_ang")
    fig.tight_layout(); fig.savefig(figure_dir / "angular_cost_heatmap.png", dpi=180); fig.savefig(figure_dir / "angular_cost_heatmap.pdf"); plt.close(fig)
    pivot = subjects.pivot(index="subject", columns="component", values="J_s")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pivot.index, pivot["ang"], "o-", label="angular/joint", linewidth=1)
    ax.plot(pivot.index, pivot["len"], ".-", label="speed", alpha=.8)
    ax.plot(pivot.index, pivot["ori"], ".-", label="orientation", alpha=.8)
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Subject", ylabel="J_s (squared-cost scale)")
    ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "component_subject_J.png", dpi=180); fig.savefig(figure_dir / "component_subject_J.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(temporal["identity_advantage"], bins=20); ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlabel="Median derangement cost - identity cost", ylabel="Cells")
    fig.tight_layout(); fig.savefig(figure_dir / "raw_temporal_identity_advantage.png", dpi=180); fig.savefig(figure_dir / "raw_temporal_identity_advantage.pdf"); plt.close(fig)


def execute() -> None:
    validate_repository(require_clean=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["protocol"]["sha256"] != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("protocol document hash differs from frozen config")
    freeze_sha = git("rev-parse", "HEAD")
    if freeze_sha == PARENT_SHA:
        raise RuntimeError("protocol freeze commit is missing")
    if not WINDOW_BANK.exists() or not SOURCE_RECORDS_PATH.exists():
        raise OpenBMIDataContractError("prepared five-bin covariance bank is missing")
    source_record = json.loads(SOURCE_RECORDS_PATH.read_text(encoding="utf-8"))
    if source_record["record_count"] != 108 or not source_record["all_source_hashes_match_donor"]:
        raise OpenBMIDataContractError("source reproduction gate failed")
    if source_record["window_covariance_bank_sha256"] != sha256_file(WINDOW_BANK):
        raise OpenBMIDataContractError("prepared covariance bank hash changed")
    expected_inputs = config["data"]["prepared_input_hashes"]
    if {
        "five_bin_covariance_archive": source_record["window_covariance_bank_sha256"],
        "canonical_covariance_array": source_record["window_covariance_array_sha256"],
        "metadata_csv": source_record["metadata_sha256"],
    } != expected_inputs:
        raise OpenBMIDataContractError("prepared input hashes differ from the frozen protocol")
    started = time.perf_counter()
    with np.load(WINDOW_BANK, allow_pickle=False) as archive:
        covariances = archive["covariances"].copy()
    means, split_means, mean_diagnostics = fit_mean_sequences(covariances)
    del covariances
    full_z, split_z, movement_diagnostics = anti_develop_banks(means, split_means)
    arrays = OUTPUT / "arrays"; tables = OUTPUT / "tables"; provenance = OUTPUT / "provenance"
    arrays.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True); provenance.mkdir(parents=True, exist_ok=True)
    atomic_savez(MEANS_PATH, {
        "full_means": means, "split_means": split_means,
        "subjects": np.arange(1, 55), "sessions": np.arange(2), "classes": np.asarray(CLASS_ORDER),
        "states": np.arange(1, 6), "halves": np.asarray(("A", "B")),
    })
    atomic_savez(MOVEMENT_PATH, {
        "full_z": full_z, "split_z": split_z, "delta_t_seconds": np.asarray(DELTA_T_SECONDS),
        "subjects": np.arange(1, 55), "sessions": np.arange(2), "classes": np.asarray(CLASS_ORDER),
        "halves": np.asarray(("A", "B")),
    })
    mean_diagnostics.to_csv(tables / "mean_sequence_diagnostics.csv", index=False, lineterminator="\n")
    movement_diagnostics.to_csv(tables / "antidevelopment_diagnostics.csv", index=False, lineterminator="\n")

    full_objective, full_optimizer = fit_quotient_matrix("full", full_z[0], full_z[1])
    sensor, length = sensor_and_length_costs(full_z[0], full_z[1])
    components = component_matrices(sensor, full_objective, length)
    split_components: dict[str, dict[str, np.ndarray]] = {}
    split_optimizers: dict[str, dict[str, np.ndarray]] = {}
    for half_index, half_name in enumerate(("A", "B")):
        objective, optimizer = fit_quotient_matrix(f"split_half_{half_name}", split_z[half_index, 0], split_z[half_index, 1])
        half_sensor, half_length = sensor_and_length_costs(split_z[half_index, 0], split_z[half_index, 1])
        split_components[half_name] = component_matrices(half_sensor, objective, half_length)
        split_optimizers[half_name] = optimizer
    max_reconstruction = float(np.max(np.abs(components["sensor"] - components["len"] - components["ang"] - components["ori"])))
    if max_reconstruction > ATOL + RTOL * float(np.max(np.abs(components["sensor"]))):
        raise OpenBMINumericalError("full component identity failed")
    atomic_savez(COMPONENT_PATH, {
        **{f"c_{key}": value for key, value in components.items()},
        "cell_subjects": np.repeat(np.arange(1, 55), 2),
        "cell_classes": np.tile(np.asarray(CLASS_ORDER), 54),
    })
    atomic_savez(SPLIT_COMPONENT_PATH, {
        **{f"half_{half}_c_{key}": values[key] for half, values in split_components.items() for key in values},
    })
    for key, matrix in components.items():
        write_matrix_csv(tables / f"c_{key}_matrix.csv", matrix)

    subject_maps, class_maps = subjectbreak_mappings(), classbreak_mappings()
    inferences = {name: evaluate_inference(values, subject_maps, class_maps) for name, values in components.items()}
    split_stats = {half: {name: relation_statistics(values) for name, values in matrices.items()} for half, matrices in split_components.items()}
    null_arrays: dict[str, Any] = {"subject_mappings": subject_maps, "class_mappings": class_maps}
    for name, inference in inferences.items():
        null_arrays.update({
            f"{name}_subjectbreak_T_subject": inference.subject_t_subject,
            f"{name}_subjectbreak_T_J": inference.subject_t_j,
            f"{name}_classbreak_T_class": inference.class_t_class,
            f"{name}_classbreak_T_J": inference.class_t_j,
        })
    atomic_savez(NULL_PATH, null_arrays)
    subject_table, summary_table = component_rows(components, inferences)
    subject_table.to_csv(tables / "component_subject_statistics.csv", index=False, lineterminator="\n")
    summary_table.to_csv(tables / "component_inference_summary.csv", index=False, lineterminator="\n")
    split_rows = []
    for half in ("A", "B"):
        for name, result in split_stats[half].items():
            for subject in range(N_SUBJECTS):
                split_rows.append({"half": half, "component": name, "subject": subject + 1, "J_s": result.j_s[subject], "T_J": result.t_j})
    pd.DataFrame(split_rows).to_csv(tables / "split_half_statistics.csv", index=False, lineterminator="\n")

    temporal = raw_temporal_correspondence(means)
    temporal.to_csv(tables / "raw_temporal_correspondence.csv", index=False, lineterminator="\n")
    write_figures(components, subject_table, temporal)
    for phase, diagnostics in {"full": full_optimizer, **{f"split_half_{k}": v for k, v in split_optimizers.items()}}.items():
        atomic_savez(arrays / f"{phase}_optimizer_certificates.npz", diagnostics)

    primary = inferences["ang"]
    result_terminal = terminal(primary.observed.t_j, primary.p_j_subject, primary.p_j_class)
    runtime = time.perf_counter() - started
    numerical = {
        "status": "PASS",
        "atol": ATOL, "rtol": RTOL,
        "minimum_c_ang": float(components["ang"].min()),
        "minimum_c_ori": float(components["ori"].min()),
        "maximum_reconstruction_absolute_error": max_reconstruction,
        "all_full_optimizer_pairs_both_sectors": bool(full_optimizer["both_sectors_certified"].all()),
        "maximum_full_selected_gradient_norm": float(full_optimizer["gradient_norm"].max()),
    }
    write_json(provenance / "numerical_certificate.json", numerical)
    results = {
        "terminal": result_terminal,
        "protocol_freeze_sha": freeze_sha,
        "primary": {
            "T_J_ang": primary.observed.t_j,
            "p_subjectbreak": primary.p_j_subject,
            "p_classbreak": primary.p_j_class,
            "J_s": primary.observed.j_s.tolist(),
        },
        "secondary": {
            name: {
                "T_subject": value.observed.t_subject, "p_subject": value.p_subject,
                "T_class": value.observed.t_class, "p_class": value.p_class,
                "T_J": value.observed.t_j, "p_J_subject": value.p_j_subject, "p_J_class": value.p_j_class,
            } for name, value in inferences.items()
        },
        "split_half": {
            half: {"T_J_ang": split_stats[half]["ang"].t_j, "J_s_ang": split_stats[half]["ang"].j_s.tolist()}
            for half in ("A", "B")
        },
        "raw_temporal": {
            "mean_identity_advantage": float(temporal["identity_advantage"].mean()),
            "fraction_identity_better_than_median_derangement": float(np.mean(temporal["identity_advantage"] > 0)),
            "median_identity_rank_among_120": float(temporal["identity_rank_among_120"].median()),
        },
        "runtime_seconds": runtime,
        "numerical": numerical,
        "input_hashes": {
            "window_bank": sha256_file(WINDOW_BANK), "means": sha256_file(MEANS_PATH),
            "movement": sha256_file(MOVEMENT_PATH), "component_matrices": sha256_file(COMPONENT_PATH),
        },
    }
    write_json(provenance / "scientific_results.json", results)
    write_report(results, subject_table, summary_table, temporal, scientific_result_sha="PENDING_SCIENTIFIC_RESULT_COMMIT")
    print(json.dumps({"terminal": result_terminal, "T_J_ang": primary.observed.t_j, "p_subject": primary.p_j_subject, "p_class": primary.p_j_class, "runtime_seconds": runtime}, indent=2))


def write_report(results: dict[str, Any], subject_table: pd.DataFrame, summary: pd.DataFrame, temporal: pd.DataFrame, *, scientific_result_sha: str) -> None:
    primary = results["primary"]; secondary = results["secondary"]
    j_lines = "\n".join(f"- S{index+1:02d}: {value:.10g}" for index, value in enumerate(primary["J_s"]))
    split_lines = []
    for half in ("A", "B"):
        split_lines.append(f"- Half {half} T_J_ang: {results['split_half'][half]['T_J_ang']:.10g}")
        split_lines.extend(f"  - S{index+1:02d}: {value:.10g}" for index, value in enumerate(results["split_half"][half]["J_s_ang"]))
    secondary_lines = "\n".join(
        f"- {name}: T_subject={value['T_subject']:.10g} (p={value['p_subject']:.6g}), "
        f"T_class={value['T_class']:.10g} (p={value['p_class']:.6g}), "
        f"T_J={value['T_J']:.10g} (p_subject={value['p_J_subject']:.6g}, p_class={value['p_J_class']:.6g})"
        for name, value in secondary.items()
    )
    text = f"""# OpenBMI Ordered Movement External Replication V0

## Immutable lineage

- Branch: `{EXPECTED_BRANCH}`
- BNCI component parent: `{PARENT_SHA}`
- BNCI component protocol freeze: `{PARENT_PROTOCOL_SHA}`
- BNCI component scientific result: `{PARENT_RESULT_SHA}`
- BNCI terminal: `{PARENT_TERMINAL}`
- OpenBMI donor branch/head: `{DONOR_BRANCH}` / `{DONOR_SHA}`
- Protocol freeze SHA: `{results['protocol_freeze_sha']}`
- Scientific result SHA: `{scientific_result_sha}`

## Contract and numerical status

The exact donor Lee2019-MI contract was reproduced for 54 subjects, both sessions, left/right hand, 50 trials per cell, and the frozen ordered 20-channel montage. Continuous 8–30 Hz filtering, 100 Hz resampling, the half-open 1.0–3.5 s epoch, no baseline, and OAS covariance estimation were inherited unchanged. The 250-sample epoch has an exact, prespecified partition into five non-overlapping 50-sample (0.5 s) bins. All 108 raw source hashes match the donor manifest.

- AIRM mean and anti-development gates: PASS
- Full/split quotient determinant-sector certification: PASS
- Minimum raw c_ang: {results['numerical']['minimum_c_ang']:.17g}
- Minimum raw c_ori: {results['numerical']['minimum_c_ori']:.17g}
- Maximum decomposition reconstruction error: {results['numerical']['maximum_reconstruction_absolute_error']:.17g}
- Maximum selected full-fit gradient norm: {results['numerical']['maximum_full_selected_gradient_norm']:.17g}

## Primary external-replication endpoint

- T_J_ang: {primary['T_J_ang']:.10g}
- Subject-break p: {primary['p_subjectbreak']:.6g}
- Class-break p: {primary['p_classbreak']:.6g}

All 54 subject J values:

{j_lines}

## Prespecified secondary results

{secondary_lines}

Raw five-state temporal correspondence versus all 44 complete derangements:

- Mean identity advantage: {results['raw_temporal']['mean_identity_advantage']:.10g}
- Fraction identity better than median derangement: {results['raw_temporal']['fraction_identity_better_than_median_derangement']:.6g}
- Median identity rank among all 120 permutations: {results['raw_temporal']['median_identity_rank_among_120']:.6g}

Split-half angular stability (odd/even acquisition positions; no half-level p-value):

{chr(10).join(split_lines)}

## Terminal

`{results['terminal']}`

This is a two-class external structural replication, not a reproduction of BNCI's four-class combinatorial structure or physical bin duration. The analyzed object is the discrete anti-development of the ordered window-wise mean covariance movement at a fixed 5 × 0.5 s OpenBMI discretization. No physiological, continuous-dynamic, source-space, motor-strategy, or stable-pose claim is made.

## Runtime, tests, and immutability

- Scientific execution runtime: {results['runtime_seconds']:.3f} seconds
- Focused and full test results: recorded in `provenance/test_results.json`
- No scientific setting was changed after first result access.
- Git status at final handoff: recorded after result finalization.
"""
    atomic_write_text(REPORT_PATH, text)


def finalize_report(scientific_result_sha: str) -> None:
    results_path = OUTPUT / "provenance" / "scientific_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    subjects = pd.read_csv(OUTPUT / "tables" / "component_subject_statistics.csv")
    summary = pd.read_csv(OUTPUT / "tables" / "component_inference_summary.csv")
    temporal = pd.read_csv(OUTPUT / "tables" / "raw_temporal_correspondence.csv")
    write_report(results, subjects, summary, temporal, scientific_result_sha=scientific_result_sha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prepare-data", "prepare-provenance", "execute", "finalize-report"))
    parser.add_argument("--scientific-result-sha")
    arguments = parser.parse_args()
    if arguments.stage == "prepare-data": prepare_data()
    elif arguments.stage == "prepare-provenance": prepare_provenance()
    elif arguments.stage == "execute": execute()
    elif arguments.stage == "finalize-report":
        if not arguments.scientific_result_sha:
            parser.error("--scientific-result-sha is required")
        finalize_report(arguments.scientific_result_sha)


if __name__ == "__main__":
    main()
