"""Frozen OpenBMI external replication of ordered AIRM movement anatomy V0.

This module contains only the prespecified data-contract, geometry, quotient,
component, and relabeling operations.  It deliberately has no endpoint-tuning
or alternative temporal-grid interfaces.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from scipy.io import loadmat

from src.geometry_v2 import airm_distance, airm_mean, symmetrize
from src.local_mean_movement_v0 import (
    FROZEN_OPTIMIZER_SETTINGS,
    MovementAlignment,
    anti_develop_sequence,
    movement_distance,
    plus_one_pvalue,
)


N_SUBJECTS = 54
N_SESSIONS = 2
N_CLASSES = 2
N_STATES = 5
N_STEPS = 4
N_CHANNELS = 20
N_TRIALS = 50
N_HALF_TRIALS = 25
N_CELLS = N_SUBJECTS * N_CLASSES
DELTA_T_SECONDS = 0.5
NULL_REPLICATES = 1999
MASTER_SEED = 20260810
SUBJECT_STREAM = 1102
CLASS_STREAM = 1101
ATOL = 1.0e-8
RTOL = 1.0e-8
MEAN_RESIDUAL_GATE = 1.0e-7

CLASS_ORDER = ("left_hand", "right_hand")
CLASS_EVENT_IDS = {"left_hand": 2, "right_hand": 1}
CHANNEL_ORDER = (
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)


class OpenBMIDataContractError(RuntimeError):
    """The immutable donor data/preprocessing contract failed."""


class OpenBMINumericalError(RuntimeError):
    """A geometry, quotient, or component numerical gate failed."""


@dataclass(frozen=True)
class RelationStatistics:
    a_sc: np.ndarray
    b_sc: np.ndarray
    c_sc: np.ndarray
    d_sc: np.ndarray
    s_sc: np.ndarray
    class_sc: np.ndarray
    j_sc: np.ndarray
    s_s: np.ndarray
    class_s: np.ndarray
    j_s: np.ndarray
    t_subject: float
    t_class: float
    t_j: float


@dataclass(frozen=True)
class Inference:
    observed: RelationStatistics
    subject_t_subject: np.ndarray
    subject_t_j: np.ndarray
    class_t_class: np.ndarray
    class_t_j: np.ndarray
    p_subject: float
    p_class: float
    p_j_subject: float
    p_j_class: float


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_savez(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_path(mne_root: Path, subject: int, source_session: int) -> Path:
    return (
        mne_root
        / "MNE-lee2019-mi-data"
        / "gigadb-datasets"
        / "live"
        / "pub"
        / "10.5524"
        / "100001_101000"
        / "100542"
        / f"session{source_session}"
        / f"s{subject}"
        / f"sess{source_session:02d}_subj{subject:02d}_EEG_MI.mat"
    )


def resolve_source(record: dict[str, Any], mne_root: Path, private_cache: Path) -> Path:
    """Resolve a source without ever changing the shared MNE cache."""

    subject, session = int(record["subject"]), int(record["session"])
    expected_size = int(record["source_bytes"])
    expected_hash = str(record["source_sha256"])
    shared = source_path(mne_root, subject, session)
    private = private_cache / f"S{subject:02d}_session{session}.mat"
    candidates = (shared, private)
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size == expected_size:
            if sha256_file(candidate) == expected_hash:
                return candidate
    private_cache.mkdir(parents=True, exist_ok=True)
    partial = private.with_suffix(".mat.part")
    partial.unlink(missing_ok=True)
    urllib.request.urlretrieve(str(record["url"]), partial)
    if partial.stat().st_size != expected_size or sha256_file(partial) != expected_hash:
        partial.unlink(missing_ok=True)
        raise OpenBMIDataContractError(f"downloaded source failed donor hash: S{subject} session{session}")
    os.replace(partial, private)
    return private


def _mat_struct(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    data = loadmat(path, variable_names=["EEG_MI_train"])["EEG_MI_train"][0, 0]
    field_names = tuple(data.dtype.names or ())
    required = {"x", "t", "y_dec", "chan"}
    if not required.issubset(field_names):
        raise OpenBMIDataContractError(f"missing MAT fields in {path.name}: {required - set(field_names)}")
    signal = np.asarray(data["x"], dtype=np.float64)
    event_samples = np.asarray(data["t"]).reshape(-1).astype(np.int64)
    labels = np.asarray(data["y_dec"]).reshape(-1).astype(np.int64)
    channel_values = np.asarray(data["chan"]).reshape(-1)
    names = [str(np.squeeze(item).item()).strip() for item in channel_values]
    sfreq = 1000.0
    if "fs" in field_names:
        sfreq = float(np.asarray(data["fs"]).reshape(-1)[0])
    return signal, event_samples, labels, names, sfreq


def preprocess_source(path: Path, *, subject: int, source_session: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply the exact donor filter/resample/epoch contract, then five-bin OAS."""

    signal, samples, labels, names, sfreq = _mat_struct(path)
    if sfreq != 1000.0 or signal.ndim != 2 or signal.shape[0] <= int(samples.max()):
        raise OpenBMIDataContractError(f"invalid source structure: {path.name}")
    try:
        indices = [names.index(channel) for channel in CHANNEL_ORDER]
    except ValueError as error:
        raise OpenBMIDataContractError(f"frozen channel missing in {path.name}: {error}") from error
    eeg = signal[:, indices].T * 1.0e-6
    info = mne.create_info(list(CHANNEL_ORDER), sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info, verbose="ERROR")
    events = np.column_stack((samples, np.zeros(len(samples), dtype=np.int64), labels))
    raw.filter(
        8.0,
        30.0,
        method="iir",
        iir_params={"ftype": "butter", "order": 5, "output": "sos"},
        phase="zero",
        verbose="ERROR",
    )
    raw, events = raw.resample(100.0, events=events, verbose="ERROR")
    filtered = raw.get_data()
    epochs = np.stack([filtered[:, int(event[0]) + 100 : int(event[0]) + 350] for event in events])
    if epochs.shape != (100, N_CHANNELS, 250):
        raise OpenBMIDataContractError(f"temporal contract failed: {path.name} -> {epochs.shape}")
    if set(labels.tolist()) != {1, 2}:
        raise OpenBMIDataContractError(f"unexpected event labels: {path.name}")
    windows = epochs.reshape(100, N_CHANNELS, N_STATES, 50).transpose(0, 2, 1, 3)
    covariances = Covariances(estimator="oas").transform(windows.reshape(500, N_CHANNELS, 50))
    covariances = symmetrize(np.asarray(covariances, dtype=np.float64)).reshape(
        100, N_STATES, N_CHANNELS, N_CHANNELS
    )
    eigenvalues = np.linalg.eigvalsh(covariances)
    if not np.isfinite(covariances).all() or float(eigenvalues.min()) <= 0.0:
        raise OpenBMIDataContractError(f"OAS covariance gate failed: {path.name}")
    rows: list[dict[str, Any]] = []
    for acquisition_order, label in enumerate(labels):
        class_name = "left_hand" if int(label) == 2 else "right_hand"
        rows.append(
            {
                "subject": subject,
                "session": source_session - 1,
                "source_session": source_session,
                "class": class_name,
                "event_id": int(label),
                "acquisition_order": acquisition_order,
            }
        )
    metadata = pd.DataFrame(rows)
    counts = metadata.groupby("class", sort=False).size().to_dict()
    if counts != {"left_hand": 50, "right_hand": 50}:
        raise OpenBMIDataContractError(f"class counts failed: {path.name}: {counts}")
    return covariances, metadata


def canonical_trial_bank(records: Iterable[tuple[np.ndarray, pd.DataFrame]]) -> tuple[np.ndarray, pd.DataFrame]:
    """Build (subject,session,class,trial,bin,d,d) in frozen acquisition order."""

    cov_bank = np.empty(
        (N_SUBJECTS, N_SESSIONS, N_CLASSES, N_TRIALS, N_STATES, N_CHANNELS, N_CHANNELS),
        dtype=np.float64,
    )
    seen = np.zeros((N_SUBJECTS, N_SESSIONS), dtype=bool)
    frames: list[pd.DataFrame] = []
    for covariances, metadata in records:
        subject = int(metadata["subject"].iloc[0]) - 1
        session = int(metadata["session"].iloc[0])
        if seen[subject, session]:
            raise OpenBMIDataContractError("duplicate subject/session record")
        seen[subject, session] = True
        for class_index, class_name in enumerate(CLASS_ORDER):
            indices = metadata.index[metadata["class"] == class_name].to_numpy()
            ordered = indices[np.argsort(metadata.loc[indices, "acquisition_order"].to_numpy())]
            if len(ordered) != N_TRIALS:
                raise OpenBMIDataContractError("canonical class count failed")
            cov_bank[subject, session, class_index] = covariances[ordered]
        frames.append(metadata)
    if not seen.all():
        raise OpenBMIDataContractError("missing subject/session records")
    return cov_bank, pd.concat(frames, ignore_index=True)


def fit_mean_sequences(covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    expected = (N_SUBJECTS, N_SESSIONS, N_CLASSES, N_TRIALS, N_STATES, N_CHANNELS, N_CHANNELS)
    values = np.asarray(covariances, dtype=np.float64)
    if values.shape != expected:
        raise ValueError(f"covariance bank must have shape {expected}")
    full = np.empty((N_SESSIONS, N_SUBJECTS, N_CLASSES, N_STATES, N_CHANNELS, N_CHANNELS))
    split = np.empty((2, N_SESSIONS, N_SUBJECTS, N_CLASSES, N_STATES, N_CHANNELS, N_CHANNELS))
    rows: list[dict[str, Any]] = []
    half_indices = (np.arange(0, N_TRIALS, 2), np.arange(1, N_TRIALS, 2))
    for session in range(N_SESSIONS):
        for subject in range(N_SUBJECTS):
            for class_index, class_name in enumerate(CLASS_ORDER):
                for state in range(N_STATES):
                    for half_index, trial_indices in [(-1, np.arange(N_TRIALS)), *enumerate(half_indices)]:
                        result = airm_mean(values[subject, session, class_index, trial_indices, state])
                        if result.had_warning or result.normalized_post_residual > MEAN_RESIDUAL_GATE:
                            raise OpenBMINumericalError(
                                f"AIRM mean failed S{subject+1} session{session} {class_name} state{state+1} half{half_index}"
                            )
                        if half_index == -1:
                            full[session, subject, class_index, state] = result.matrix
                            half_name = "full"
                        else:
                            split[half_index, session, subject, class_index, state] = result.matrix
                            half_name = "A" if half_index == 0 else "B"
                        rows.append(
                            {
                                "subject": subject + 1,
                                "session": session,
                                "class": class_name,
                                "state": state + 1,
                                "half": half_name,
                                "trial_count": len(trial_indices),
                                "normalized_karcher_residual": result.normalized_post_residual,
                                "warning_count": len(result.warning_messages),
                            }
                        )
    return full, split, pd.DataFrame(rows)


def anti_develop_banks(full: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    full_z = np.empty((N_SESSIONS, N_SUBJECTS, N_CLASSES, N_STEPS, N_CHANNELS, N_CHANNELS))
    split_z = np.empty((2, N_SESSIONS, N_SUBJECTS, N_CLASSES, N_STEPS, N_CHANNELS, N_CHANNELS))
    rows: list[dict[str, Any]] = []
    for half_index in (-1, 0, 1):
        source = full if half_index == -1 else split[half_index]
        target = full_z if half_index == -1 else split_z[half_index]
        for session in range(N_SESSIONS):
            for subject in range(N_SUBJECTS):
                for class_index, class_name in enumerate(CLASS_ORDER):
                    result = anti_develop_sequence(source[session, subject, class_index], delta_t=DELTA_T_SECONDS)
                    target[session, subject, class_index] = result.z
                    table = result.diagnostics.copy()
                    table.insert(0, "class", class_name)
                    table.insert(0, "session", session)
                    table.insert(0, "subject", subject + 1)
                    table.insert(0, "half", "full" if half_index == -1 else ("A" if half_index == 0 else "B"))
                    rows.extend(table.to_dict(orient="records"))
    return full_z, split_z, pd.DataFrame(rows)


def flatten_cells(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (N_SUBJECTS, N_CLASSES, N_STEPS, N_CHANNELS, N_CHANNELS):
        raise ValueError("cell bank shape changed")
    return array.reshape(N_CELLS, N_STEPS, N_CHANNELS, N_CHANNELS)


def sensor_and_length_costs(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = flatten_cells(first), flatten_cells(second)
    residual = left[:, None] - right[None]
    sensor = np.mean(np.sum(residual * residual, axis=(-2, -1)), axis=-1)
    speed0 = np.linalg.norm(left, axis=(-2, -1))
    speed1 = np.linalg.norm(right, axis=(-2, -1))
    length = np.mean((speed0[:, None] - speed1[None]) ** 2, axis=-1)
    return sensor, length


def component_matrices(sensor: np.ndarray, full: np.ndarray, length: np.ndarray) -> dict[str, np.ndarray]:
    result = {
        "sensor": np.asarray(sensor, dtype=np.float64),
        "full": np.asarray(full, dtype=np.float64),
        "len": np.asarray(length, dtype=np.float64),
    }
    result["ang"] = result["full"] - result["len"]
    result["ori"] = result["sensor"] - result["full"]
    for key, values in result.items():
        if values.shape != (N_CELLS, N_CELLS) or not np.isfinite(values).all():
            raise OpenBMINumericalError(f"invalid {key} component matrix")
    bounds_ang = ATOL + RTOL * np.maximum(np.abs(result["full"]), np.abs(result["len"]))
    bounds_ori = ATOL + RTOL * np.maximum(np.abs(result["sensor"]), np.abs(result["full"]))
    if np.any(result["ang"] < -bounds_ang) or np.any(result["ori"] < -bounds_ori):
        raise OpenBMINumericalError(
            f"meaningful negative component: ang={result['ang'].min():.17g}, ori={result['ori'].min():.17g}"
        )
    reconstructed = result["len"] + result["ang"] + result["ori"]
    if not np.allclose(result["sensor"], reconstructed, atol=ATOL, rtol=RTOL):
        raise OpenBMINumericalError("component reconstruction failed")
    return result


def cell_index(subject: int, class_index: int) -> int:
    return subject * N_CLASSES + class_index


def relation_statistics(matrix: np.ndarray) -> RelationStatistics:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (N_CELLS, N_CELLS) or not np.isfinite(values).all():
        raise ValueError("cost matrix must be finite 108x108")
    a = np.empty((N_SUBJECTS, N_CLASSES)); b = np.empty_like(a)
    c = np.empty_like(a); d = np.empty_like(a)
    for subject in range(N_SUBJECTS):
        for class_index in range(N_CLASSES):
            anchor = cell_index(subject, class_index)
            a[subject, class_index] = values[anchor, anchor]
            b[subject, class_index] = values[anchor, cell_index(subject, 1 - class_index)]
            c[subject, class_index] = np.mean([
                values[anchor, cell_index(other, class_index)]
                for other in range(N_SUBJECTS) if other != subject
            ])
            d[subject, class_index] = np.mean([
                values[anchor, cell_index(other, 1 - class_index)]
                for other in range(N_SUBJECTS) if other != subject
            ])
    s_sc = c - a; class_sc = b - a; j_sc = b + c - a - d
    s_s = s_sc.mean(axis=1); class_s = class_sc.mean(axis=1); j_s = j_sc.mean(axis=1)
    return RelationStatistics(
        a, b, c, d, s_sc, class_sc, j_sc, s_s, class_s, j_s,
        float(s_s.mean()), float(class_s.mean()), float(j_s.mean()),
    )


def subjectbreak_mappings(replicates: int = NULL_REPLICATES) -> np.ndarray:
    generator = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, SUBJECT_STREAM]))
    mappings = np.empty((replicates, N_CELLS), dtype=np.int64)
    for draw in range(replicates):
        mapping = np.empty(N_CELLS, dtype=np.int64)
        for class_index in range(N_CLASSES):
            permutation = generator.permutation(N_SUBJECTS)
            for subject in range(N_SUBJECTS):
                mapping[cell_index(subject, class_index)] = cell_index(int(permutation[subject]), class_index)
        mappings[draw] = mapping
    return mappings


def classbreak_mappings(replicates: int = NULL_REPLICATES) -> np.ndarray:
    generator = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, CLASS_STREAM]))
    mappings = np.empty((replicates, N_CELLS), dtype=np.int64)
    for draw in range(replicates):
        mapping = np.empty(N_CELLS, dtype=np.int64)
        for subject in range(N_SUBJECTS):
            permutation = generator.permutation(N_CLASSES)
            for class_index in range(N_CLASSES):
                mapping[cell_index(subject, class_index)] = cell_index(subject, int(permutation[class_index]))
        mappings[draw] = mapping
    return mappings


def evaluate_inference(
    matrix: np.ndarray,
    subject_maps: np.ndarray | None = None,
    class_maps: np.ndarray | None = None,
) -> Inference:
    subject_maps = subjectbreak_mappings() if subject_maps is None else np.asarray(subject_maps)
    class_maps = classbreak_mappings() if class_maps is None else np.asarray(class_maps)
    if subject_maps.shape != (NULL_REPLICATES, N_CELLS) or class_maps.shape != (NULL_REPLICATES, N_CELLS):
        raise ValueError("null mapping shape changed")
    observed = relation_statistics(matrix)
    subject_t_subject = np.empty(NULL_REPLICATES); subject_t_j = np.empty(NULL_REPLICATES)
    class_t_class = np.empty(NULL_REPLICATES); class_t_j = np.empty(NULL_REPLICATES)
    for draw in range(NULL_REPLICATES):
        subject_result = relation_statistics(matrix[:, subject_maps[draw]])
        class_result = relation_statistics(matrix[:, class_maps[draw]])
        subject_t_subject[draw] = subject_result.t_subject
        subject_t_j[draw] = subject_result.t_j
        class_t_class[draw] = class_result.t_class
        class_t_j[draw] = class_result.t_j
    return Inference(
        observed, subject_t_subject, subject_t_j, class_t_class, class_t_j,
        plus_one_pvalue(observed.t_subject, subject_t_subject),
        plus_one_pvalue(observed.t_class, class_t_class),
        plus_one_pvalue(observed.t_j, subject_t_j),
        plus_one_pvalue(observed.t_j, class_t_j),
    )


def raw_temporal_correspondence(full_means: np.ndarray) -> pd.DataFrame:
    """Prespecified descriptive identity-vs-all-44-derangements control."""

    import itertools

    derangements = [p for p in itertools.permutations(range(N_STATES)) if all(i != p[i] for i in range(N_STATES))]
    rows: list[dict[str, Any]] = []
    for subject in range(N_SUBJECTS):
        for class_index, class_name in enumerate(CLASS_ORDER):
            left = full_means[0, subject, class_index]
            right = full_means[1, subject, class_index]
            distances = np.asarray([[airm_distance(left[i], right[j]) for j in range(N_STATES)] for i in range(N_STATES)])
            identity = float(np.sqrt(np.mean(np.diag(distances) ** 2)))
            wrong = np.asarray([np.sqrt(np.mean([distances[i, p[i]] ** 2 for i in range(N_STATES)])) for p in derangements])
            all_permutations = np.asarray([
                np.sqrt(np.mean([distances[i, p[i]] ** 2 for i in range(N_STATES)]))
                for p in itertools.permutations(range(N_STATES))
            ])
            rows.append({
                "subject": subject + 1,
                "class": class_name,
                "identity_cost": identity,
                "median_derangement_cost": float(np.median(wrong)),
                "identity_advantage": float(np.median(wrong) - identity),
                "identity_rank_among_120": int(1 + np.count_nonzero(all_permutations < identity)),
            })
    return pd.DataFrame(rows)


def optimizer_record(alignment: MovementAlignment) -> dict[str, float | int | bool]:
    determinants = {start.final_determinant for start in alignment.starts if start.converged}
    return {
        "objective": alignment.objective,
        "distance": alignment.distance,
        "selected_determinant": alignment.determinant,
        "both_sectors_certified": determinants == {-1, 1},
        "gradient_norm": alignment.gradient_norm,
        "best_start_index": alignment.best_start_index,
        "second_best_objective": alignment.second_best_objective,
        "objective_spread": alignment.objective_spread,
    }


def quotient_pair(first: np.ndarray, second: np.ndarray) -> tuple[float, dict[str, float | int | bool]]:
    _, alignment = movement_distance(first, second, settings=FROZEN_OPTIMIZER_SETTINGS)
    record = optimizer_record(alignment)
    if not bool(record["both_sectors_certified"]):
        raise OpenBMINumericalError("both determinant sectors were not certified")
    return float(alignment.objective), record


def terminal(t_j_ang: float, p_subject: float, p_class: float) -> str:
    if t_j_ang > 0.0 and p_subject < 0.05 and p_class < 0.05:
        return "REPLICATED_OPENBMI_ANGULAR_JOINT_INTERACTION"
    return "NOT_REPLICATED_OPENBMI_ANGULAR_JOINT_INTERACTION"


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
