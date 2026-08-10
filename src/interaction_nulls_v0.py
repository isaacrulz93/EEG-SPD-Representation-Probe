"""Deterministic label/derangement null primitives and resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.interaction_provenance_v0 import canonical_json_bytes, sha256_array


MASTER_SEED = 20260810
REQUIRED_KEY_FIELDS = ("dataset", "geometry", "stage", "signature", "template", "replicate_index")


def seed_words(key: Mapping[str, Any], master_seed: int = MASTER_SEED) -> np.ndarray:
    if tuple(sorted(key)) != tuple(sorted(REQUIRED_KEY_FIELDS)):
        raise ValueError(f"null RNG key must contain exactly {REQUIRED_KEY_FIELDS}")
    replicate = key["replicate_index"]
    if not isinstance(replicate, (int, np.integer)) or int(replicate) < 0:
        raise ValueError("replicate_index must be a nonnegative integer")
    payload = {"master_seed": int(master_seed), **{field: key[field] for field in REQUIRED_KEY_FIELDS}}
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return np.frombuffer(digest[:16], dtype="<u4").copy()


def replicate_rng(key: Mapping[str, Any], master_seed: int = MASTER_SEED) -> np.random.Generator:
    seed = np.random.SeedSequence(seed_words(key, master_seed).tolist())
    return np.random.Generator(np.random.PCG64DXSM(seed))


def make_key(
    *, dataset: str, geometry: str, stage: str, signature: str,
    template: str, replicate_index: int,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset), "geometry": str(geometry), "stage": str(stage),
        "signature": str(signature), "template": str(template),
        "replicate_index": int(replicate_index),
    }


def shuffle_labels_subject_session(
    labels: Sequence[Any] | np.ndarray,
    metadata: pd.DataFrame,
    *,
    key: Mapping[str, Any],
    master_seed: int = MASTER_SEED,
) -> np.ndarray:
    """Shuffle labels within subject/session in canonical trial-UID order."""

    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != len(metadata) or len(values) < 1:
        raise ValueError("labels must be a nonempty vector aligned with metadata")
    required = {"subject", "session", "trial_uid"}
    if required - set(metadata.columns):
        raise ValueError(f"metadata missing columns: {sorted(required - set(metadata.columns))}")
    if metadata["trial_uid"].astype(str).duplicated().any():
        raise ValueError("trial_uid must be globally unique")
    output = values.copy()
    rng = replicate_rng(key, master_seed)
    subject_values = sorted(int(value) for value in metadata["subject"].unique())
    session_values = sorted(str(value) for value in metadata["session"].unique())
    subjects = metadata["subject"].to_numpy(dtype=np.int64)
    sessions = metadata["session"].astype(str).to_numpy()
    uids = metadata["trial_uid"].astype(str).to_numpy()
    for subject in subject_values:
        for session in session_values:
            indices = np.flatnonzero((subjects == subject) & (sessions == session))
            if len(indices) == 0:
                continue
            indices = indices[np.argsort(uids[indices], kind="stable")]
            output[indices] = rng.permutation(values[indices])
            if not np.array_equal(np.sort(output[indices].astype(str)), np.sort(values[indices].astype(str))):
                raise RuntimeError("label shuffle did not preserve a subject/session multiset")
    return output


def refit_label_null_once(
    labels: np.ndarray,
    metadata: pd.DataFrame,
    *,
    key: Mapping[str, Any],
    fit_statistic: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Destroy labels and invoke the full caller-supplied refit exactly once."""

    shuffled = shuffle_labels_subject_session(labels, metadata, key=key)
    result = np.asarray(fit_statistic(shuffled), dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("fit_statistic must return a finite subject-score vector")
    return result


@dataclass(frozen=True)
class NullCheckpoint:
    metadata: Mapping[str, Any]
    replicate_indices: np.ndarray
    seed_words: np.ndarray
    completed: np.ndarray
    subject_statistics: np.ndarray
    group_statistics: np.ndarray
    payload_sha256: str


def _payload_hash(
    metadata: Mapping[str, Any], indices: np.ndarray, seeds: np.ndarray,
    completed: np.ndarray, subjects: np.ndarray, groups: np.ndarray,
) -> str:
    digest = hashlib.sha256(canonical_json_bytes(dict(metadata)))
    for value in (indices, seeds, completed, subjects, groups):
        digest.update(sha256_array(value).encode("ascii"))
    return digest.hexdigest()


def create_checkpoint(
    *, total_replicates: int, n_subjects: int, identity: Mapping[str, Any],
    dataset: str, geometry: str, stage: str, signature: str, template: str,
) -> NullCheckpoint:
    total, n = int(total_replicates), int(n_subjects)
    if total < 1 or n < 1:
        raise ValueError("checkpoint dimensions must be positive")
    required_identity = {"protocol_sha256", "config_sha256", "code_sha", "input_hashes"}
    if required_identity - set(identity):
        raise ValueError(f"checkpoint identity missing {sorted(required_identity - set(identity))}")
    metadata = {
        "schema_version": "subject-class-interaction-null-v0",
        **dict(identity), "dataset": dataset, "geometry": geometry, "stage": stage,
        "signature": signature, "template": template, "total_replicates": total,
        "n_subjects": n, "master_seed": MASTER_SEED, "bit_generator": "PCG64DXSM",
        "replicate_index_base": 0,
    }
    indices = np.arange(total, dtype=np.int64)
    seeds = np.stack([
        seed_words(make_key(dataset=dataset, geometry=geometry, stage=stage, signature=signature, template=template, replicate_index=index))
        for index in indices
    ])
    completed = np.zeros(total, dtype=np.uint8)
    subjects = np.full((total, n), np.nan, dtype=np.float64)
    groups = np.full(total, np.nan, dtype=np.float64)
    return NullCheckpoint(metadata, indices, seeds, completed, subjects, groups, _payload_hash(metadata, indices, seeds, completed, subjects, groups))


def _validate(checkpoint: NullCheckpoint) -> None:
    total = int(checkpoint.metadata["total_replicates"])
    n = int(checkpoint.metadata["n_subjects"])
    if checkpoint.replicate_indices.shape != (total,) or not np.array_equal(checkpoint.replicate_indices, np.arange(total)):
        raise ValueError("checkpoint replicate indices are not canonical")
    if checkpoint.seed_words.shape != (total, 4) or checkpoint.seed_words.dtype != np.uint32:
        raise ValueError("checkpoint seed mapping has the wrong shape/dtype")
    if checkpoint.completed.shape != (total,) or not np.all(np.isin(checkpoint.completed, [0, 1])):
        raise ValueError("checkpoint completed bitmap is invalid")
    if checkpoint.subject_statistics.shape != (total, n) or checkpoint.group_statistics.shape != (total,):
        raise ValueError("checkpoint statistic shapes are invalid")
    done = checkpoint.completed.astype(bool)
    if not np.isfinite(checkpoint.subject_statistics[done]).all() or not np.isfinite(checkpoint.group_statistics[done]).all():
        raise ValueError("completed checkpoint rows are nonfinite")
    if not np.isnan(checkpoint.subject_statistics[~done]).all() or not np.isnan(checkpoint.group_statistics[~done]).all():
        raise ValueError("incomplete checkpoint rows must remain NaN")
    if np.any(done) and not np.array_equal(np.median(checkpoint.subject_statistics[done], axis=1), checkpoint.group_statistics[done]):
        raise ValueError("checkpoint group statistics are not subject medians")
    expected = _payload_hash(checkpoint.metadata, checkpoint.replicate_indices, checkpoint.seed_words, checkpoint.completed, checkpoint.subject_statistics, checkpoint.group_statistics)
    if expected != checkpoint.payload_sha256:
        raise ValueError("checkpoint payload SHA mismatch")


def pending_indices(checkpoint: NullCheckpoint) -> np.ndarray:
    _validate(checkpoint)
    return np.flatnonzero(checkpoint.completed == 0).astype(np.int64)


def record_batch(checkpoint: NullCheckpoint, indices: Sequence[int], subject_statistics: np.ndarray) -> NullCheckpoint:
    _validate(checkpoint)
    selected = np.asarray(indices, dtype=np.int64)
    values = np.asarray(subject_statistics, dtype=np.float64)
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= len(checkpoint.completed)):
        raise ValueError("record indices are invalid")
    if np.any(checkpoint.completed[selected] == 1):
        raise ValueError("refusing to overwrite a completed replicate")
    expected_shape = (len(selected), checkpoint.subject_statistics.shape[1])
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(f"subject_statistics must be finite with shape {expected_shape}")
    completed = checkpoint.completed.copy()
    subjects = checkpoint.subject_statistics.copy()
    groups = checkpoint.group_statistics.copy()
    subjects[selected] = values
    groups[selected] = np.median(values, axis=1)
    completed[selected] = 1
    return replace(checkpoint, completed=completed, subject_statistics=subjects, group_statistics=groups, payload_sha256=_payload_hash(checkpoint.metadata, checkpoint.replicate_indices, checkpoint.seed_words, completed, subjects, groups))


def save_checkpoint(path: str | Path, checkpoint: NullCheckpoint) -> None:
    _validate(checkpoint)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle, metadata_json=np.asarray(json.dumps(dict(checkpoint.metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=False)),
                replicate_indices=checkpoint.replicate_indices, seed_words=checkpoint.seed_words,
                completed=checkpoint.completed, subject_statistics=checkpoint.subject_statistics,
                group_statistics=checkpoint.group_statistics, payload_sha256=np.asarray(checkpoint.payload_sha256),
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path, expected_identity: Mapping[str, Any] | None = None) -> NullCheckpoint:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"metadata_json", "replicate_indices", "seed_words", "completed", "subject_statistics", "group_statistics", "payload_sha256"}
        if set(archive.files) != required:
            raise ValueError("checkpoint keys mismatch")
        checkpoint = NullCheckpoint(
            json.loads(str(archive["metadata_json"].item())),
            np.asarray(archive["replicate_indices"], dtype=np.int64),
            np.asarray(archive["seed_words"], dtype=np.uint32),
            np.asarray(archive["completed"], dtype=np.uint8),
            np.asarray(archive["subject_statistics"], dtype=np.float64),
            np.asarray(archive["group_statistics"], dtype=np.float64),
            str(archive["payload_sha256"].item()),
        )
    _validate(checkpoint)
    if expected_identity is not None:
        for key, value in expected_identity.items():
            if checkpoint.metadata.get(key) != value:
                raise ValueError(f"checkpoint identity mismatch for {key}")
    return checkpoint


def canonical_final_summary_bytes(checkpoint: NullCheckpoint) -> bytes:
    _validate(checkpoint)
    if len(pending_indices(checkpoint)):
        raise ValueError("cannot summarize an incomplete checkpoint")
    payload = {
        "metadata": dict(checkpoint.metadata),
        "group_statistics": [float(value).hex() for value in checkpoint.group_statistics],
        "subject_statistics": [[float(value).hex() for value in row] for row in checkpoint.subject_statistics],
        "seed_mapping_sha256": sha256_array(checkpoint.seed_words),
    }
    return canonical_json_bytes(payload) + b"\n"
