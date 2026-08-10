"""Observed and null execution pipeline for Subject Class Interaction v0."""

from __future__ import annotations

import itertools
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.conditional_geometry_v1 import airm_mean_batched, spd_log, symmetric_exp
from src.interaction_nulls_v0 import make_key, replicate_rng, seed_words, shuffle_labels_subject_session
from src.interaction_provenance_v0 import (
    atomic_write_json,
    environment_record,
    git_state,
    sha256_array,
    sha256_file,
)
from src.interaction_statistics_v0 import (
    all_derangements,
    derangement_statistics,
    exact_null_summary,
    monte_carlo_summary,
    reliability_subject_scores,
    same_subject_scores,
    similarity_matrix,
)
from src.subject_class_interaction_v0 import (
    InteractionObjects,
    build_interactions_from_means,
    compute_interactions,
    geometry_thresholds,
    load_bnci_two_session_data,
    load_frozen_config,
    split_masks,
)


GEOMETRIES = ("AIRM", "LE")
TEMPLATES = ("session_specific", "pooled_session")
OBJECTS = ("Z", "R")
SIGNATURES = ("sensor", "spectrum")
SPLITS = ("A", "B", "F")
JOINT_GEOMETRY_KEY = "joint_AIRM_LE"
JOINT_SIGNATURE_KEY = "joint_R_Z_sensor_spectrum"
JOINT_TEMPLATE_KEY = "joint_session_specific_pooled"
_NULL_WORKER_CONTEXT: tuple[Any, ...] | None = None


def chain_order() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(itertools.product(GEOMETRIES, TEMPLATES, OBJECTS, SIGNATURES))


def chain_id(geometry: str, template: str, object_name: str, signature: str) -> str:
    return "__".join((geometry, template, object_name, signature))


def _signature(objects: InteractionObjects, object_name: str, signature: str) -> np.ndarray:
    source = objects.sensor if signature == "sensor" else objects.spectrum
    return np.asarray(source[object_name], dtype=np.float64)


def _array_key(geometry: str, template: str, split: str, field: str) -> str:
    return "__".join((geometry, template, split, field))


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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


def freeze_output_protocol(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_frozen_config(root)
    output = root / str(config["project"]["output_dir"])
    protocol_dir = output / "provenance"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    source_config = root / "configs/subject_class_interaction_v0.yaml"
    source_protocol = root / str(config["protocol"]["protocol_path"])
    for source, destination in (
        (source_config, protocol_dir / "frozen_config.yaml"),
        (source_protocol, protocol_dir / "PROTOCOL_SUBJECT_CLASS_INTERACTION_V0.md"),
    ):
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"refusing to overwrite a different frozen artifact: {destination}")
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
    payload = {
        "schema_version": "subject-class-interaction-v0",
        "phase": "PROTOCOL_FROZEN",
        "base_commit": config["protocol"]["base_commit"],
        "frozen_definition_commit": "09fb211",
        "branch": config["protocol"]["branch"],
        "config_sha256": config_hash,
        "protocol_sha256": config["protocol"]["protocol_sha256"],
        "master_seed": config["protocol"]["master_seed"],
    }
    atomic_write_json(protocol_dir / "manifest.json", payload)
    return payload


def _serialize_objects(objects: InteractionObjects) -> dict[str, np.ndarray]:
    arrays = {
        "marginal_means": objects.marginal_means,
        "class_means": objects.class_means,
        "class_counts": objects.class_counts,
        "class_proportions": objects.class_proportions,
        "U": objects.U,
        "population_templates": objects.population_templates,
        "R": objects.R,
        "Rbar": objects.Rbar,
        "Z": objects.Z,
    }
    for object_name in OBJECTS:
        arrays[f"sensor_raw_{object_name}"] = objects.sensor_raw[object_name]
        arrays[f"sensor_{object_name}"] = objects.sensor[object_name]
        arrays[f"sensor_norms_{object_name}"] = objects.sensor_norms[object_name]
        arrays[f"spectrum_raw_{object_name}"] = objects.spectrum_raw[object_name]
        arrays[f"spectrum_{object_name}"] = objects.spectrum[object_name]
        arrays[f"spectrum_norms_{object_name}"] = objects.spectrum_norms[object_name]
    for object_name in ("U", "R", "Z"):
        arrays[f"per_class_norms_{object_name}"] = objects.per_class_norms[object_name]
    return arrays


def _build_pooled(session_specific: InteractionObjects, config: Mapping[str, Any]) -> InteractionObjects:
    return build_interactions_from_means(
        marginal_means=session_specific.marginal_means,
        class_means=session_specific.class_means,
        class_counts=session_specific.class_counts,
        geometry=session_specific.geometry,
        template="pooled_session",
        subjects=session_specific.subjects,
        sessions=session_specific.sessions,
        classes=session_specific.classes,
        config=config,
        mean_audit_rows=session_specific.mean_audit_rows,
    )


def compute_observed(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[tuple[str, str, str], InteractionObjects]:
    masks = split_masks(metadata, "bnci2014_001")
    result: dict[tuple[str, str, str], InteractionObjects] = {}
    for geometry in GEOMETRIES:
        for split in SPLITS:
            mask = masks[split]
            session_specific = compute_interactions(
                covariances[mask], metadata.loc[mask].reset_index(drop=True),
                config=config, geometry=geometry, template="session_specific",
            )
            result[(geometry, "session_specific", split)] = session_specific
            result[(geometry, "pooled_session", split)] = _build_pooled(session_specific, config)
    return result


def _observed_rows(objects: Mapping[tuple[str, str, str], InteractionObjects]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    group_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    similarities: dict[str, np.ndarray] = {}
    for geometry, template, object_name, signature in chain_order():
        a = _signature(objects[(geometry, template, "A")], object_name, signature)
        b = _signature(objects[(geometry, template, "B")], object_name, signature)
        full = _signature(objects[(geometry, template, "F")], object_name, signature)
        r_scores = reliability_subject_scores(a, b)
        i_scores = same_subject_scores(full[:, 0], full[:, 1])
        identifier = chain_id(geometry, template, object_name, signature)
        similarities[identifier] = similarity_matrix(full[:, 0], full[:, 1])
        group_rows.extend([
            {"dataset": "BNCI2014_001", "geometry": geometry, "template": template, "object": object_name, "signature": signature, "stage": "R", "observed_statistic": float(np.median(r_scores))},
            {"dataset": "BNCI2014_001", "geometry": geometry, "template": template, "object": object_name, "signature": signature, "stage": "I", "observed_statistic": float(np.median(i_scores))},
            {"dataset": "BNCI2014_001", "geometry": geometry, "template": template, "object": object_name, "signature": signature, "stage": "C", "observed_statistic": float(np.median(i_scores))},
        ])
        for subject_index, subject in enumerate(objects[(geometry, template, "F")].subjects):
            subject_rows.append({
                "dataset": "BNCI2014_001", "geometry": geometry, "template": template,
                "object": object_name, "signature": signature, "subject": subject,
                "session0_half_cosine": float(np.sum(a[subject_index, 0] * b[subject_index, 0])),
                "session1_half_cosine": float(np.sum(a[subject_index, 1] * b[subject_index, 1])),
                "mean_session_half_cosine": float(r_scores[subject_index]),
                "same_subject_cross_session_cosine": float(i_scores[subject_index]),
            })
    return group_rows, subject_rows, similarities


def _energy_rows(objects: Mapping[tuple[str, str, str], InteractionObjects]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for geometry in GEOMETRIES:
        for template in TEMPLATES:
            full = objects[(geometry, template, "F")]
            half_a = objects[(geometry, template, "A")]
            half_b = objects[(geometry, template, "B")]
            for q, session in enumerate(full.sessions):
                population = float(np.mean(np.linalg.norm(full.population_templates[:, q], axis=(-2, -1)) ** 2))
                rbar = float(np.mean(np.linalg.norm(full.Rbar[:, q], axis=(-2, -1)) ** 2))
                z = float(np.mean(np.linalg.norm(full.Z[:, q], axis=(-2, -1)) ** 2))
                split_discrepancy = float(np.mean(np.linalg.norm(half_a.Z[:, q] - half_b.Z[:, q], axis=(-2, -1)) ** 2))
                cross_discrepancy = float(np.mean(np.linalg.norm(full.Z[:, 0] - full.Z[:, 1], axis=(-2, -1)) ** 2))
                residual_total = rbar + z
                rows.append({
                    "dataset": "BNCI2014_001", "geometry": geometry, "template": template,
                    "session": session, "label": "DESCRIPTIVE ENERGY FRACTIONS",
                    "population_class_effect_squared_norm": population,
                    "class_independent_Rbar_squared_norm": rbar,
                    "Z_interaction_squared_norm": z,
                    "split_half_discrepancy_energy": split_discrepancy,
                    "cross_session_discrepancy_energy": cross_discrepancy,
                    "Rbar_fraction_of_Rbar_plus_Z": rbar / residual_total,
                    "Z_fraction_of_Rbar_plus_Z": z / residual_total,
                    "population_normalized_fraction": np.nan,
                })
    return rows


def run_bnci_observed(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_frozen_config(root)
    output = root / str(config["project"]["output_dir"])
    freeze_output_protocol(root)
    covariances, metadata, input_provenance = load_bnci_two_session_data(root, config)
    objects = compute_observed(covariances, metadata, config)
    arrays: dict[str, np.ndarray] = {}
    object_metadata: dict[str, Any] = {}
    for (geometry, template, split), value in objects.items():
        prefix = (geometry, template, split)
        for field, array in _serialize_objects(value).items():
            arrays[_array_key(*prefix, field)] = array
        object_metadata["__".join(prefix)] = {
            "subjects": list(value.subjects), "sessions": list(value.sessions),
            "classes": list(value.classes), "gate_metrics": dict(value.gate_metrics),
            "object_hashes": dict(value.object_hashes), "mean_audit_rows": list(value.mean_audit_rows),
        }
    arrays["metadata_json"] = np.asarray(json.dumps(object_metadata, sort_keys=True, separators=(",", ":"), allow_nan=True))
    object_path = output / "objects/core_interaction_objects.npz"
    _atomic_savez(object_path, arrays)

    group_rows, subject_rows, similarities = _observed_rows(objects)
    tables = output / "tables"
    reliability = output / "reliability"
    cross = output / "cross_session"
    for directory in (tables, reliability, cross, output / "class_dependence", output / "gauge", output / "decisions", output / "figures", output / "report"):
        directory.mkdir(parents=True, exist_ok=True)
    counts = metadata.groupby(["subject", "session", "run", "class_label"], observed=True).size().rename("observed_trials").reset_index()
    counts["expected_trials"] = 12
    counts["passed"] = counts["observed_trials"] == counts["expected_trials"]
    counts.to_csv(tables / "data_contract.csv", index=False, lineterminator="\n", float_format="%.17g")

    gate_rows = []
    mean_rows = []
    class_rows = []
    global_rows = []
    interaction_rows = []
    for (geometry, template, split), value in objects.items():
        for metric, observed in value.gate_metrics.items():
            gate_rows.append({"dataset": "BNCI2014_001", "geometry": geometry, "template": template, "split": split, "gate": metric, "observed": observed, "passed": True})
        if template == "session_specific":
            mean_rows.extend({"dataset": "BNCI2014_001", "split": split, **row} for row in value.mean_audit_rows)
        for s, subject in enumerate(value.subjects):
            for q, session in enumerate(value.sessions):
                global_rows.append({"dataset": "BNCI2014_001", "geometry": geometry, "template": template, "split": split, "subject": subject, "session": session, "Rbar_frobenius_norm": float(np.linalg.norm(value.Rbar[s, q]))})
                for c, class_name in enumerate(value.classes):
                    class_rows.append({"dataset": "BNCI2014_001", "geometry": geometry, "template": template, "split": split, "subject": subject, "session": session, "class": class_name, "U_frobenius_norm": float(value.per_class_norms["U"][s, q, c]), "population_template_frobenius_norm": float(np.linalg.norm(value.population_templates[s, q, c]))})
                    interaction_rows.append({"dataset": "BNCI2014_001", "geometry": geometry, "template": template, "split": split, "subject": subject, "session": session, "class": class_name, "R_frobenius_norm": float(value.per_class_norms["R"][s, q, c]), "Z_frobenius_norm": float(value.per_class_norms["Z"][s, q, c]), "sensor_Z_raw_norm": float(value.sensor_norms["Z"][s, q]), "spectrum_Z_raw_norm": float(value.spectrum_norms["Z"][s, q])})
    pd.DataFrame(gate_rows + mean_rows).to_csv(tables / "geometry_gates.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(class_rows).to_csv(tables / "class_effects_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(global_rows).to_csv(tables / "subject_global_residual_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(interaction_rows).to_csv(tables / "interaction_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(_energy_rows(objects)).to_csv(tables / "descriptive_energy_fractions.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(group_rows).to_csv(tables / "observed_stage_statistics.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(subject_rows).to_csv(reliability / "split_half_subject_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(subject_rows)[["dataset", "geometry", "template", "object", "signature", "subject", "same_subject_cross_session_cosine"]].to_csv(cross / "same_subject_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
    primary_id = chain_id("AIRM", "session_specific", "Z", "sensor")
    primary_similarity = similarities[primary_id]
    pd.DataFrame(primary_similarity, index=[f"S{s}" for s in range(1, 10)], columns=[f"S{s}" for s in range(1, 10)]).rename_axis("session0_subject").to_csv(cross / "similarity_matrix.csv", lineterminator="\n", float_format="%.17g")
    np.save(output / "objects/observed_similarity_matrices.npy", np.stack([similarities[chain_id(*chain)] for chain in chain_order()]))

    atomic_write_json(output / "provenance/environment.json", environment_record())
    atomic_write_json(output / "provenance/git_state.json", git_state(root))
    manifest = {
        "schema_version": "subject-class-interaction-observed-v0",
        "dataset": "BNCI2014_001", "role": "RETROSPECTIVE_DEVELOPMENT_ONLY",
        "config_sha256": config_hash, "protocol_sha256": config["protocol"]["protocol_sha256"],
        "input_provenance": input_provenance,
        "core_objects_file": str(object_path.relative_to(root)),
        "core_objects_sha256": sha256_file(object_path),
        "hard_gates_pass": True,
    }
    atomic_write_json(output / "provenance/bnci_observed_manifest.json", manifest)
    return manifest


def _load_observed_arrays(output: Path) -> dict[str, np.ndarray]:
    path = output / "objects/core_interaction_objects.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files if key != "metadata_json"}


def _marginals(arrays: Mapping[str, np.ndarray], geometry: str, split: str) -> np.ndarray:
    return arrays[_array_key(geometry, "session_specific", split, "marginal_means")]


def _class_means_for_label_batch(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    label_batch: np.ndarray,
    mask: np.ndarray,
    *, geometry: str, config: Mapping[str, Any], scalar_crosscheck: bool,
    classes: Sequence[str] | None = None,
    log_covariances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(label_batch).astype(str)
    b = labels.shape[0]
    subjects = tuple(sorted(int(value) for value in metadata["subject"].unique()))
    sessions = tuple(str(value) for value in metadata["session"].drop_duplicates())
    classes = (
        tuple(str(value) for value in config["datasets"]["bnci2014_001"]["classes"])
        if classes is None
        else tuple(str(value) for value in classes)
    )
    p = covariances.shape[-1]
    means = np.empty((b, len(subjects), len(sessions), len(classes), p, p), dtype=np.float64)
    counts = np.empty((b, len(subjects), len(sessions), len(classes)), dtype=np.int64)
    subject_values = metadata["subject"].to_numpy(dtype=np.int64)
    session_values = metadata["session"].astype(str).to_numpy()
    if geometry == "LE":
        log_covariances = (
            spd_log(covariances)
            if log_covariances is None
            else np.asarray(log_covariances, dtype=np.float64)
        )
        if log_covariances.shape != covariances.shape or not np.isfinite(log_covariances).all():
            raise ValueError("log_covariances must be finite and match covariances")
    grouped: dict[int, list[tuple[tuple[int, int, int, int], np.ndarray]]] = {}
    for replicate in range(b):
        for s, subject in enumerate(subjects):
            for q, session in enumerate(sessions):
                base = mask & (subject_values == subject) & (session_values == session)
                for c, class_name in enumerate(classes):
                    selected = base & (labels[replicate] == class_name)
                    count = int(np.count_nonzero(selected))
                    if count < 1:
                        raise RuntimeError("label null produced an empty class")
                    counts[replicate, s, q, c] = count
                    if geometry == "LE":
                        means[replicate, s, q, c] = symmetric_exp(np.mean(log_covariances[selected], axis=0, dtype=np.float64))
                    else:
                        grouped.setdefault(count, []).append(((replicate, s, q, c), covariances[selected]))
    if geometry == "AIRM":
        checked = False
        thresholds = geometry_thresholds(config)
        for count in sorted(grouped):
            records = grouped[count]
            stack = np.stack([record[1] for record in records])
            result = airm_mean_batched(
                stack, thresholds=thresholds,
                scalar_crosscheck=[0] if scalar_crosscheck and not checked else False,
                scalar_crosscheck_tolerance=1.0e-10,
            )
            checked = checked or scalar_crosscheck
            if not result.all_passed:
                raise RuntimeError("UNASSESSED_NUMERICAL_OR_DATA_FAILURE: null AIRM mean gate failed")
            for index, (target, _) in enumerate(records):
                means[target] = result.matrices.reshape((-1, p, p))[index]
    return means, counts


@dataclass(frozen=True)
class JointCheckpoint:
    metadata: Mapping[str, Any]
    chain_ids: tuple[str, ...]
    completed: np.ndarray
    seed_words: np.ndarray
    subject_statistics: np.ndarray
    payload_sha256: str


def _joint_hash(metadata: Mapping[str, Any], chains: Sequence[str], completed: np.ndarray, seeds: np.ndarray, values: np.ndarray) -> str:
    import hashlib
    digest = hashlib.sha256(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    digest.update("\n".join(chains).encode("utf-8"))
    for array in (completed, seeds, values):
        digest.update(sha256_array(array).encode("ascii"))
    return digest.hexdigest()


def _create_joint_checkpoint(stage: str, identity: Mapping[str, Any], replicates: int) -> JointCheckpoint:
    chains = tuple(chain_id(*chain) for chain in chain_order())
    metadata = {"schema_version": "joint-label-null-v0", **dict(identity), "stage": stage, "replicates": int(replicates), "master_seed": 20260810, "bit_generator": "PCG64DXSM"}
    completed = np.zeros(replicates, dtype=np.uint8)
    seeds = np.stack([seed_words(make_key(dataset="BNCI2014_001", geometry=JOINT_GEOMETRY_KEY, stage=stage, signature=JOINT_SIGNATURE_KEY, template=JOINT_TEMPLATE_KEY, replicate_index=index)) for index in range(replicates)])
    values = np.full((replicates, len(chains), 9), np.nan, dtype=np.float64)
    return JointCheckpoint(metadata, chains, completed, seeds, values, _joint_hash(metadata, chains, completed, seeds, values))


def _validate_joint(value: JointCheckpoint) -> None:
    replicates = int(value.metadata["replicates"])
    n_subjects = int(value.metadata.get("n_subjects", 9))
    if value.completed.shape != (replicates,) or value.seed_words.shape != (replicates, 4) or value.subject_statistics.shape != (replicates, len(value.chain_ids), n_subjects):
        raise ValueError("joint checkpoint shape mismatch")
    done = value.completed.astype(bool)
    if not np.isfinite(value.subject_statistics[done]).all() or not np.isnan(value.subject_statistics[~done]).all():
        raise ValueError("joint checkpoint finite/NaN contract failure")
    if value.payload_sha256 != _joint_hash(value.metadata, value.chain_ids, value.completed, value.seed_words, value.subject_statistics):
        raise ValueError("joint checkpoint hash mismatch")


def _save_joint(path: Path, value: JointCheckpoint) -> None:
    _validate_joint(value)
    _atomic_savez(path, {
        "metadata_json": np.asarray(json.dumps(dict(value.metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)),
        "chain_ids": np.asarray(value.chain_ids), "completed": value.completed,
        "seed_words": value.seed_words, "subject_statistics": value.subject_statistics,
        "payload_sha256": np.asarray(value.payload_sha256),
    })


def _load_joint(path: Path, expected_identity: Mapping[str, Any]) -> JointCheckpoint:
    with np.load(path, allow_pickle=False) as archive:
        value = JointCheckpoint(
            json.loads(str(archive["metadata_json"].item())),
            tuple(str(item) for item in archive["chain_ids"].tolist()),
            np.asarray(archive["completed"], dtype=np.uint8),
            np.asarray(archive["seed_words"], dtype=np.uint32),
            np.asarray(archive["subject_statistics"], dtype=np.float64),
            str(archive["payload_sha256"].item()),
        )
    _validate_joint(value)
    for key, expected in expected_identity.items():
        if value.metadata.get(key) != expected:
            raise ValueError(f"joint checkpoint identity mismatch for {key}")
    return value


def _record_joint(value: JointCheckpoint, indices: np.ndarray, statistics: np.ndarray) -> JointCheckpoint:
    _validate_joint(value)
    n_subjects = int(value.metadata.get("n_subjects", 9))
    if np.any(value.completed[indices]) or statistics.shape != (len(indices), len(value.chain_ids), n_subjects) or not np.isfinite(statistics).all():
        raise ValueError("invalid joint checkpoint update")
    completed = value.completed.copy()
    subject_values = value.subject_statistics.copy()
    subject_values[indices] = statistics
    completed[indices] = 1
    return JointCheckpoint(value.metadata, value.chain_ids, completed, value.seed_words, subject_values, _joint_hash(value.metadata, value.chain_ids, completed, value.seed_words, subject_values))


def _joint_label_batch_statistics(
    *, stage: str, replicate_indices: np.ndarray, covariances: np.ndarray,
    metadata: pd.DataFrame, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray], scalar_crosscheck: bool,
) -> np.ndarray:
    original_labels = metadata["class_label"].astype(str).to_numpy()
    label_rows = []
    for index in replicate_indices:
        key = make_key(dataset="BNCI2014_001", geometry=JOINT_GEOMETRY_KEY, stage=stage, signature=JOINT_SIGNATURE_KEY, template=JOINT_TEMPLATE_KEY, replicate_index=int(index))
        label_rows.append(shuffle_labels_subject_session(original_labels, metadata, key=key))
    label_batch = np.stack(label_rows)
    required_splits = ("A", "B") if stage == "R" else ("F",)
    constructed: dict[tuple[int, str, str, str], InteractionObjects] = {}
    for geometry in GEOMETRIES:
        for split in required_splits:
            means, counts = _class_means_for_label_batch(
                covariances, metadata, label_batch, masks[split], geometry=geometry,
                config=config, scalar_crosscheck=scalar_crosscheck and geometry == "AIRM",
            )
            marginal = _marginals(arrays, geometry, split)
            for replicate in range(len(replicate_indices)):
                session_specific = build_interactions_from_means(
                    marginal_means=marginal, class_means=means[replicate], class_counts=counts[replicate],
                    geometry=geometry, template="session_specific", subjects=range(1, 10),
                    sessions=("0train", "1test"), classes=("left_hand", "right_hand", "feet", "tongue"),
                    config=config,
                )
                constructed[(replicate, geometry, "session_specific", split)] = session_specific
                constructed[(replicate, geometry, "pooled_session", split)] = _build_pooled(session_specific, config)
    output = np.empty((len(replicate_indices), len(chain_order()), 9), dtype=np.float64)
    for replicate in range(len(replicate_indices)):
        for chain_index, (geometry, template, object_name, signature) in enumerate(chain_order()):
            if stage == "R":
                a = _signature(constructed[(replicate, geometry, template, "A")], object_name, signature)
                b = _signature(constructed[(replicate, geometry, template, "B")], object_name, signature)
                output[replicate, chain_index] = reliability_subject_scores(a, b)
            else:
                full = _signature(constructed[(replicate, geometry, template, "F")], object_name, signature)
                output[replicate, chain_index] = same_subject_scores(full[:, 0], full[:, 1])
    return output


def _parallel_joint_task(arguments: tuple[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one immutable index batch from a fork-inherited read-only context."""

    if _NULL_WORKER_CONTEXT is None:
        raise RuntimeError("parallel null worker context is not initialized")
    stage, indices = arguments
    covariances, metadata, config, arrays, masks = _NULL_WORKER_CONTEXT
    statistics = _joint_label_batch_statistics(
        stage=stage, replicate_indices=indices, covariances=covariances,
        metadata=metadata, config=config, arrays=arrays, masks=masks,
        scalar_crosscheck=False,
    )
    return indices, statistics


def run_bnci_nulls(
    repo_root: str | Path, *, batch_size: int = 4, workers: int = 1
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_frozen_config(root)
    output = root / str(config["project"]["output_dir"])
    observed_manifest = json.loads((output / "provenance/bnci_observed_manifest.json").read_text(encoding="utf-8"))
    object_path = output / "objects/core_interaction_objects.npz"
    identity = {
        "protocol_sha256": config["protocol"]["protocol_sha256"],
        "config_sha256": config_hash,
        "code_sha": git_state(root)["tracked_code_sha256"],
        "input_hashes": observed_manifest["input_provenance"],
        "objects_sha256": sha256_file(object_path),
    }
    covariances, metadata, _ = load_bnci_two_session_data(root, config)
    arrays = _load_observed_arrays(output)
    masks = split_masks(metadata, "bnci2014_001")
    checkpoints: dict[str, JointCheckpoint] = {}
    for stage in ("R", "C"):
        total = int(config["stages"]["reliability" if stage == "R" else "class_dependence"]["label_destruction_replicates"])
        path = root / str(config["project"]["cache_dir"]) / "checkpoints" / f"bnci_{stage}_joint.npz"
        checkpoint = _load_joint(path, identity) if path.exists() else _create_joint_checkpoint(stage, identity, total)
        pending = np.flatnonzero(checkpoint.completed == 0)
        batches = [pending[start : start + int(batch_size)] for start in range(0, len(pending), int(batch_size))]
        if batches:
            indices = batches.pop(0)
            statistics = _joint_label_batch_statistics(
                stage=stage, replicate_indices=indices, covariances=covariances,
                metadata=metadata, config=config, arrays=arrays, masks=masks,
                scalar_crosscheck=True,
            )
            checkpoint = _record_joint(checkpoint, indices, statistics)
            _save_joint(path, checkpoint)
        if batches and int(workers) > 1:
            global _NULL_WORKER_CONTEXT
            _NULL_WORKER_CONTEXT = (covariances, metadata, config, arrays, masks)
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as executor:
                for indices, statistics in executor.map(
                    _parallel_joint_task,
                    ((stage, indices) for indices in batches),
                    chunksize=1,
                ):
                    checkpoint = _record_joint(checkpoint, indices, statistics)
                    _save_joint(path, checkpoint)
            _NULL_WORKER_CONTEXT = None
        else:
            for indices in batches:
                statistics = _joint_label_batch_statistics(
                    stage=stage, replicate_indices=indices, covariances=covariances,
                    metadata=metadata, config=config, arrays=arrays, masks=masks,
                    scalar_crosscheck=False,
                )
                checkpoint = _record_joint(checkpoint, indices, statistics)
                _save_joint(path, checkpoint)
        if np.any(checkpoint.completed == 0):
            raise RuntimeError(f"{stage} label checkpoint remained incomplete")
        checkpoints[stage] = checkpoint

    observed = pd.read_csv(output / "tables/observed_stage_statistics.csv")
    summary_rows: list[dict[str, Any]] = []
    subject_null_rows: list[dict[str, Any]] = []
    chain_lookup = {identifier: index for index, identifier in enumerate(checkpoints["R"].chain_ids)}
    for chain in chain_order():
        geometry, template, object_name, signature = chain
        identifier = chain_id(*chain)
        chain_index = chain_lookup[identifier]
        for stage in ("R", "C"):
            observed_value = float(observed.loc[
                (observed["geometry"] == geometry) & (observed["template"] == template) &
                (observed["object"] == object_name) & (observed["signature"] == signature) &
                (observed["stage"] == stage), "observed_statistic"
            ].iloc[0])
            null = np.median(checkpoints[stage].subject_statistics[:, chain_index], axis=1)
            summary = monte_carlo_summary(observed_value, null)
            summary_rows.append({"dataset": "BNCI2014_001", "geometry": geometry, "template": template, "object": object_name, "signature": signature, "stage": stage, **summary.__dict__})
            for replicate in range(len(null)):
                subject_null_rows.append({"geometry": geometry, "template": template, "object": object_name, "signature": signature, "stage": stage, "replicate_index": replicate, "group_statistic": float(null[replicate])})

        full_sensor_key = _array_key(geometry, template, "F", f"{signature}_{object_name}")
        full = arrays[full_sensor_key]
        similarity = similarity_matrix(full[:, 0], full[:, 1])
        mappings = all_derangements(9)
        null_i = derangement_statistics(similarity, mappings)
        observed_i = float(np.median(np.diag(similarity)))
        summary_rows.append({"dataset": "BNCI2014_001", "geometry": geometry, "template": template, "object": object_name, "signature": signature, "stage": "I", **exact_null_summary(observed_i, null_i)})

    reliability_dir = output / "reliability"
    cross_dir = output / "cross_session"
    class_dir = output / "class_dependence"
    pd.DataFrame([row for row in summary_rows if row["stage"] == "R"]).to_csv(reliability_dir / "split_half_null_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame([row for row in summary_rows if row["stage"] == "I"]).to_csv(cross_dir / "derangement_null_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame([row for row in summary_rows if row["stage"] == "C"]).to_csv(class_dir / "label_destruction_null_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(subject_null_rows).to_csv(output / "tables/null_group_statistics.csv", index=False, lineterminator="\n", float_format="%.17g")
    atomic_write_json(output / "provenance/null_manifest.json", {
        "schema_version": "subject-class-interaction-null-v0", "config_sha256": config_hash,
        "code_sha": identity["code_sha"], "input_hashes": identity["input_hashes"],
        "objects_sha256": identity["objects_sha256"], "replicates_R": int(len(checkpoints["R"].completed)),
        "replicates_C": int(len(checkpoints["C"].completed)), "derangements_I": 133496,
        "all_complete": True,
    })
    primary = [row for row in summary_rows if row["geometry"] == "AIRM" and row["template"] == "session_specific" and row["object"] == "Z" and row["signature"] == "sensor"]
    return {row["stage"]: row for row in primary}
