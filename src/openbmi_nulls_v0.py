"""Frozen, resumable OpenBMI null execution for Subject Class Interaction v0."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.conditional_geometry_v1 import spd_log
from src.interaction_nulls_v0 import make_key, replicate_rng, seed_words, shuffle_labels_subject_session
from src.interaction_pipeline_v0 import (
    GEOMETRIES,
    JOINT_GEOMETRY_KEY,
    JOINT_SIGNATURE_KEY,
    JOINT_TEMPLATE_KEY,
    JointCheckpoint,
    _array_key,
    _atomic_savez,
    _build_pooled,
    _class_means_for_label_batch,
    _joint_hash,
    _load_joint,
    _marginals,
    _record_joint,
    _save_joint,
    _signature,
    chain_id,
    chain_order,
)
from src.interaction_provenance_v0 import atomic_write_json, sha256_array, sha256_file
from src.interaction_statistics_v0 import (
    derangement_statistics,
    monte_carlo_summary,
    random_derangements,
    reliability_subject_scores,
    same_subject_scores,
    similarity_matrix,
)
from src.openbmi_protocol_v0 import validate_scientific_unlock
from src.subject_class_interaction_v0 import (
    build_interactions_from_means,
    load_frozen_config,
    split_masks,
)


DATASET = "OpenBMI"
CLASSES = ("left_hand", "right_hand")
SUBJECTS = tuple(range(1, 55))
SESSIONS = ("0", "1")
_OPENBMI_NULL_CONTEXT: tuple[Any, ...] | None = None


def _rng_key(stage: str, replicate_index: int) -> dict[str, Any]:
    return make_key(
        dataset=DATASET,
        geometry=JOINT_GEOMETRY_KEY,
        stage=stage,
        signature=JOINT_SIGNATURE_KEY,
        template=JOINT_TEMPLATE_KEY,
        replicate_index=int(replicate_index),
    )


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _create_label_checkpoint(
    stage: str,
    identity: Mapping[str, Any],
    replicates: int,
) -> JointCheckpoint:
    chains = tuple(chain_id(*chain) for chain in chain_order())
    metadata = {
        "schema_version": "joint-label-null-v0",
        **dict(identity),
        "dataset": DATASET,
        "stage": stage,
        "replicates": int(replicates),
        "n_subjects": len(SUBJECTS),
        "master_seed": 20260810,
        "bit_generator": "PCG64DXSM",
    }
    completed = np.zeros(replicates, dtype=np.uint8)
    seeds = np.stack([seed_words(_rng_key(stage, index)) for index in range(replicates)])
    values = np.full((replicates, len(chains), len(SUBJECTS)), np.nan, dtype=np.float64)
    return JointCheckpoint(
        metadata,
        chains,
        completed,
        seeds,
        values,
        _joint_hash(metadata, chains, completed, seeds, values),
    )


def _label_batch_statistics(
    *,
    stage: str,
    replicate_indices: np.ndarray,
    covariances: np.ndarray,
    log_covariances: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    scalar_crosscheck: bool,
) -> np.ndarray:
    original_labels = metadata["class_label"].astype(str).to_numpy()
    label_batch = np.stack([
        shuffle_labels_subject_session(original_labels, metadata, key=_rng_key(stage, int(index)))
        for index in replicate_indices
    ])
    required_splits = ("A", "B") if stage == "R" else ("F",)
    constructed: dict[tuple[int, str, str, str], Any] = {}
    for geometry in GEOMETRIES:
        for split in required_splits:
            means, counts = _class_means_for_label_batch(
                covariances,
                metadata,
                label_batch,
                masks[split],
                geometry=geometry,
                config=config,
                scalar_crosscheck=scalar_crosscheck and geometry == "AIRM",
                classes=CLASSES,
                log_covariances=log_covariances if geometry == "LE" else None,
            )
            marginal = _marginals(arrays, geometry, split)
            for replicate in range(len(replicate_indices)):
                session_specific = build_interactions_from_means(
                    marginal_means=marginal,
                    class_means=means[replicate],
                    class_counts=counts[replicate],
                    geometry=geometry,
                    template="session_specific",
                    subjects=SUBJECTS,
                    sessions=SESSIONS,
                    classes=CLASSES,
                    config=config,
                )
                constructed[(replicate, geometry, "session_specific", split)] = session_specific
                constructed[(replicate, geometry, "pooled_session", split)] = _build_pooled(session_specific, config)
    output = np.empty((len(replicate_indices), len(chain_order()), len(SUBJECTS)), dtype=np.float64)
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


def _parallel_label_task(arguments: tuple[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if _OPENBMI_NULL_CONTEXT is None:
        raise RuntimeError("OpenBMI parallel null worker context is not initialized")
    covariances, log_covariances, metadata, config, arrays, masks = _OPENBMI_NULL_CONTEXT
    stage, indices = arguments
    values = _label_batch_statistics(
        stage=stage,
        replicate_indices=indices,
        covariances=covariances,
        log_covariances=log_covariances,
        metadata=metadata,
        config=config,
        arrays=arrays,
        masks=masks,
        scalar_crosscheck=False,
    )
    return indices, values


@dataclass(frozen=True)
class GroupCheckpoint:
    metadata: Mapping[str, Any]
    chain_ids: tuple[str, ...]
    completed: np.ndarray
    seed_words: np.ndarray
    group_statistics: np.ndarray
    payload_sha256: str


def _group_hash(
    metadata: Mapping[str, Any],
    chains: Sequence[str],
    completed: np.ndarray,
    seeds: np.ndarray,
    values: np.ndarray,
) -> str:
    digest = hashlib.sha256(
        json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    digest.update("\n".join(chains).encode("utf-8"))
    for array in (completed, seeds, values):
        digest.update(sha256_array(array).encode("ascii"))
    return digest.hexdigest()


def _create_group_checkpoint(identity: Mapping[str, Any], replicates: int) -> GroupCheckpoint:
    chains = tuple(chain_id(*chain) for chain in chain_order())
    metadata = {
        "schema_version": "joint-derangement-null-v0",
        **dict(identity),
        "dataset": DATASET,
        "stage": "I",
        "replicates": int(replicates),
        "n_subjects": len(SUBJECTS),
        "master_seed": 20260810,
        "bit_generator": "PCG64DXSM",
    }
    completed = np.zeros(replicates, dtype=np.uint8)
    seeds = np.stack([seed_words(_rng_key("I", index)) for index in range(replicates)])
    values = np.full((replicates, len(chains)), np.nan, dtype=np.float64)
    return GroupCheckpoint(
        metadata,
        chains,
        completed,
        seeds,
        values,
        _group_hash(metadata, chains, completed, seeds, values),
    )


def _validate_group(value: GroupCheckpoint) -> None:
    replicates = int(value.metadata["replicates"])
    expected = (replicates, len(value.chain_ids))
    if (
        value.completed.shape != (replicates,)
        or value.seed_words.shape != (replicates, 4)
        or value.group_statistics.shape != expected
    ):
        raise ValueError("OpenBMI derangement checkpoint shape mismatch")
    done = value.completed.astype(bool)
    if not np.isfinite(value.group_statistics[done]).all() or not np.isnan(value.group_statistics[~done]).all():
        raise ValueError("OpenBMI derangement checkpoint finite/NaN contract failure")
    expected_hash = _group_hash(
        value.metadata,
        value.chain_ids,
        value.completed,
        value.seed_words,
        value.group_statistics,
    )
    if value.payload_sha256 != expected_hash:
        raise ValueError("OpenBMI derangement checkpoint hash mismatch")


def _save_group(path: Path, value: GroupCheckpoint) -> None:
    _validate_group(value)
    _atomic_savez(path, {
        "metadata_json": np.asarray(json.dumps(dict(value.metadata), sort_keys=True, separators=(",", ":"))),
        "chain_ids": np.asarray(value.chain_ids),
        "completed": value.completed,
        "seed_words": value.seed_words,
        "group_statistics": value.group_statistics,
        "payload_sha256": np.asarray(value.payload_sha256),
    })


def _load_group(path: Path, identity: Mapping[str, Any]) -> GroupCheckpoint:
    with np.load(path, allow_pickle=False) as archive:
        value = GroupCheckpoint(
            json.loads(str(archive["metadata_json"].item())),
            tuple(str(item) for item in archive["chain_ids"].tolist()),
            np.asarray(archive["completed"], dtype=np.uint8),
            np.asarray(archive["seed_words"], dtype=np.uint32),
            np.asarray(archive["group_statistics"], dtype=np.float64),
            str(archive["payload_sha256"].item()),
        )
    _validate_group(value)
    for key, expected in identity.items():
        if value.metadata.get(key) != expected:
            raise ValueError(f"OpenBMI derangement checkpoint identity mismatch for {key}")
    return value


def _record_group(value: GroupCheckpoint, indices: np.ndarray, statistics: np.ndarray) -> GroupCheckpoint:
    _validate_group(value)
    if (
        statistics.shape != (len(indices), len(value.chain_ids))
        or np.any(value.completed[indices])
        or not np.isfinite(statistics).all()
    ):
        raise ValueError("invalid OpenBMI derangement checkpoint update")
    completed = value.completed.copy()
    group_statistics = value.group_statistics.copy()
    completed[indices] = 1
    group_statistics[indices] = statistics
    return GroupCheckpoint(
        value.metadata,
        value.chain_ids,
        completed,
        value.seed_words,
        group_statistics,
        _group_hash(value.metadata, value.chain_ids, completed, value.seed_words, group_statistics),
    )


def _observed_statistics(arrays: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for geometry, template, object_name, signature in chain_order():
        a = arrays[_array_key(geometry, template, "A", f"{signature}_{object_name}")]
        b = arrays[_array_key(geometry, template, "B", f"{signature}_{object_name}")]
        full = arrays[_array_key(geometry, template, "F", f"{signature}_{object_name}")]
        r_scores = reliability_subject_scores(a, b)
        i_scores = same_subject_scores(full[:, 0], full[:, 1])
        for stage, observed in (("R", np.median(r_scores)), ("I", np.median(i_scores)), ("C", np.median(i_scores))):
            group_rows.append({
                "dataset": DATASET,
                "geometry": geometry,
                "template": template,
                "object": object_name,
                "signature": signature,
                "stage": stage,
                "observed_statistic": float(observed),
            })
        for subject_index, subject in enumerate(SUBJECTS):
            subject_rows.append({
                "dataset": DATASET,
                "geometry": geometry,
                "template": template,
                "object": object_name,
                "signature": signature,
                "subject": subject,
                "session0_half_cosine": float(np.sum(a[subject_index, 0] * b[subject_index, 0])),
                "session1_half_cosine": float(np.sum(a[subject_index, 1] * b[subject_index, 1])),
                "mean_session_half_cosine": float(r_scores[subject_index]),
                "same_subject_cross_session_cosine": float(i_scores[subject_index]),
            })
    return pd.DataFrame(group_rows), pd.DataFrame(subject_rows)


def _run_label_stage(
    *,
    stage: str,
    path: Path,
    identity: Mapping[str, Any],
    replicates: int,
    batch_size: int,
    workers: int,
    covariances: np.ndarray,
    log_covariances: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
) -> JointCheckpoint:
    checkpoint = _load_joint(path, identity) if path.exists() else _create_label_checkpoint(stage, identity, replicates)
    pending = np.flatnonzero(checkpoint.completed == 0)
    batches = [pending[start:start + batch_size] for start in range(0, len(pending), batch_size)]
    if batches:
        indices = batches.pop(0)
        values = _label_batch_statistics(
            stage=stage,
            replicate_indices=indices,
            covariances=covariances,
            log_covariances=log_covariances,
            metadata=metadata,
            config=config,
            arrays=arrays,
            masks=masks,
            scalar_crosscheck=True,
        )
        checkpoint = _record_joint(checkpoint, indices, values)
        _save_joint(path, checkpoint)
    if batches and workers > 1:
        global _OPENBMI_NULL_CONTEXT
        _OPENBMI_NULL_CONTEXT = (covariances, log_covariances, metadata, config, arrays, masks)
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            for indices, values in executor.map(
                _parallel_label_task,
                ((stage, indices) for indices in batches),
                chunksize=1,
            ):
                checkpoint = _record_joint(checkpoint, indices, values)
                _save_joint(path, checkpoint)
        _OPENBMI_NULL_CONTEXT = None
    else:
        for indices in batches:
            values = _label_batch_statistics(
                stage=stage,
                replicate_indices=indices,
                covariances=covariances,
                log_covariances=log_covariances,
                metadata=metadata,
                config=config,
                arrays=arrays,
                masks=masks,
                scalar_crosscheck=False,
            )
            checkpoint = _record_joint(checkpoint, indices, values)
            _save_joint(path, checkpoint)
    if np.any(checkpoint.completed == 0):
        raise RuntimeError(f"OpenBMI {stage} checkpoint remained incomplete")
    return checkpoint


def _run_identity_stage(
    *,
    path: Path,
    identity: Mapping[str, Any],
    replicates: int,
    batch_size: int,
    arrays: Mapping[str, np.ndarray],
) -> GroupCheckpoint:
    checkpoint = _load_group(path, identity) if path.exists() else _create_group_checkpoint(identity, replicates)
    similarities = [
        similarity_matrix(
            arrays[_array_key(geometry, template, "F", f"{signature}_{object_name}")][:, 0],
            arrays[_array_key(geometry, template, "F", f"{signature}_{object_name}")][:, 1],
        )
        for geometry, template, object_name, signature in chain_order()
    ]
    pending = np.flatnonzero(checkpoint.completed == 0)
    for start in range(0, len(pending), batch_size):
        indices = pending[start:start + batch_size]
        rngs = [replicate_rng(_rng_key("I", int(index))) for index in indices]
        mappings = random_derangements(len(SUBJECTS), len(indices), rngs)
        values = np.stack([derangement_statistics(matrix, mappings) for matrix in similarities], axis=1)
        checkpoint = _record_group(checkpoint, indices, values)
        _save_group(path, checkpoint)
    if np.any(checkpoint.completed == 0):
        raise RuntimeError("OpenBMI I checkpoint remained incomplete")
    return checkpoint


def run_openbmi_nulls(
    repo_root: str | Path,
    *,
    batch_size: int = 4,
    workers: int = 1,
    identity_batch_size: int = 1000,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_frozen_config(root)
    validate_scientific_unlock(root, config)
    output = root / config["project"]["output_dir"]
    cache = root / config["project"]["cache_dir"]
    object_path = output / "objects/openbmi_core_interaction_objects.npz"
    source_manifest_path = output / "provenance/openbmi_source_manifest.json"
    observed_manifest = json.loads((output / "provenance/openbmi_observed_manifest.json").read_text())
    if not observed_manifest.get("hard_gates_pass"):
        raise RuntimeError("OpenBMI observed hard gates did not pass")
    identity = {
        "protocol_sha256": config["protocol"]["protocol_sha256"],
        "config_sha256": config_hash,
        "manifest_commit_sha": "91877d3ea5b83a6b524fbbe091fb1db6c9973170",
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "objects_sha256": sha256_file(object_path),
        "null_code_sha256": sha256_file(Path(__file__)),
    }
    with np.load(cache / "openbmi_covariances.npz", allow_pickle=False) as archive:
        covariances = np.asarray(archive["covariances"], dtype=np.float64)
    metadata = pd.read_csv(cache / "openbmi_metadata.csv")
    metadata["session"] = metadata["session"].astype(str)
    metadata["run"] = metadata["run"].astype(str)
    arrays = _load_arrays(object_path)
    masks = split_masks(metadata, "openbmi_lee2019_mi")
    log_covariances = spd_log(covariances)
    checkpoint_dir = cache / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    label_replicates = int(config["stages"]["reliability"]["label_destruction_replicates"])
    checkpoints = {
        stage: _run_label_stage(
            stage=stage,
            path=checkpoint_dir / f"openbmi_{stage}_joint.npz",
            identity=identity,
            replicates=label_replicates,
            batch_size=int(batch_size),
            workers=int(workers),
            covariances=covariances,
            log_covariances=log_covariances,
            metadata=metadata,
            config=config,
            arrays=arrays,
            masks=masks,
        )
        for stage in ("R", "C")
    }
    identity_replicates = 100000
    identity_checkpoint = _run_identity_stage(
        path=checkpoint_dir / "openbmi_I_joint.npz",
        identity=identity,
        replicates=identity_replicates,
        batch_size=int(identity_batch_size),
        arrays=arrays,
    )
    observed, subject_rows = _observed_statistics(arrays)
    summary_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    chain_lookup = {identifier: index for index, identifier in enumerate(checkpoints["R"].chain_ids)}
    for chain in chain_order():
        geometry, template, object_name, signature = chain
        identifier = chain_id(*chain)
        chain_index = chain_lookup[identifier]
        for stage in ("R", "I", "C"):
            observed_value = float(observed.loc[
                (observed["geometry"] == geometry)
                & (observed["template"] == template)
                & (observed["object"] == object_name)
                & (observed["signature"] == signature)
                & (observed["stage"] == stage),
                "observed_statistic",
            ].iloc[0])
            if stage == "I":
                null = identity_checkpoint.group_statistics[:, chain_index]
            else:
                null = np.median(checkpoints[stage].subject_statistics[:, chain_index], axis=1)
                label_rows.extend({
                    "dataset": DATASET,
                    "geometry": geometry,
                    "template": template,
                    "object": object_name,
                    "signature": signature,
                    "stage": stage,
                    "replicate_index": replicate,
                    "group_statistic": float(value),
                } for replicate, value in enumerate(null))
            summary_rows.append({
                "dataset": DATASET,
                "geometry": geometry,
                "template": template,
                "object": object_name,
                "signature": signature,
                "stage": stage,
                **monte_carlo_summary(observed_value, null).__dict__,
            })
    summary = pd.DataFrame(summary_rows)
    observed.to_csv(output / "tables/openbmi_observed_stage_statistics.csv", index=False, lineterminator="\n", float_format="%.17g")
    subject_rows.to_csv(output / "reliability/openbmi_split_half_subject_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
    subject_rows[[
        "dataset", "geometry", "template", "object", "signature", "subject",
        "same_subject_cross_session_cosine",
    ]].to_csv(output / "cross_session/openbmi_same_subject_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
    summary[summary["stage"] == "R"].to_csv(output / "reliability/openbmi_split_half_null_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    summary[summary["stage"] == "I"].to_csv(output / "cross_session/openbmi_derangement_null_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    summary[summary["stage"] == "C"].to_csv(output / "class_dependence/openbmi_label_destruction_null_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(label_rows).to_csv(output / "tables/openbmi_null_group_statistics.csv", index=False, lineterminator="\n", float_format="%.17g")
    manifest = {
        "schema_version": "openbmi-null-manifest-v0",
        **identity,
        "replicates_R": label_replicates,
        "replicates_I": identity_replicates,
        "replicates_C": label_replicates,
        "all_complete": True,
        "checkpoint_hashes": {
            "R": checkpoints["R"].payload_sha256,
            "I": identity_checkpoint.payload_sha256,
            "C": checkpoints["C"].payload_sha256,
        },
    }
    atomic_write_json(output / "provenance/openbmi_null_manifest.json", manifest)
    primary = summary[
        (summary["geometry"] == "AIRM")
        & (summary["template"] == "session_specific")
        & (summary["object"] == "Z")
        & (summary["signature"] == "sensor")
    ].set_index("stage").loc[["R", "I", "C"]]
    return {stage: primary.loc[stage].to_dict() for stage in ("R", "I", "C")}
