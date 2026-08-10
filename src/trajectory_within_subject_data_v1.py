"""Fail-closed two-session data preparation for trajectory audit v1."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml
from pyriemann.estimation import Covariances

from src.data_trajectory_v0 import load_trajectory_window5
from src.trajectory_geometry_v0 import (
    SCALAR_11_NAMES,
    check_bag_permutation_invariance,
    compute_five_state_geometry,
    trajectory_hard_checks,
)
from src.trajectory_within_subject_v1 import (
    CLASS_ORDER,
    IDENTITY_COLUMNS,
    SESSION_ORDER,
    compare_reproduction,
    load_frozen_config,
    sha256_array,
    sha256_file,
    stable_json_sha256,
    validate_metadata,
)


FEATURE_KEYS = (
    "airm_distance_matrices",
    "airm_path_d10",
    "airm_bag_canon_d10",
    "airm_scalars_11",
    "airm_canonical_permutation",
)


class TrajectoryAuditDataError(RuntimeError):
    """A pinned input, data, reproduction, or numerical gate failed."""


@dataclass(frozen=True)
class PreparedAuditData:
    metadata: pd.DataFrame
    arrays: Mapping[str, np.ndarray]
    data_contract: pd.DataFrame
    reproduction_gate: pd.DataFrame
    provenance: Mapping[str, Any]
    combined_cache_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray], *, compressed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            function = np.savez_compressed if compressed else np.savez
            function(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def _verify_entry(root: Path, entry: Mapping[str, Any], label: str) -> Path:
    path = (root / str(entry["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise TrajectoryAuditDataError(f"{label} escapes repository: {path}") from error
    if not path.is_file():
        raise TrajectoryAuditDataError(f"{label} is missing: {path}")
    if entry.get("bytes") is not None and path.stat().st_size != int(entry["bytes"]):
        raise TrajectoryAuditDataError(f"{label} byte count mismatch")
    if sha256_file(path) != str(entry["sha256"]):
        raise TrajectoryAuditDataError(f"{label} SHA-256 mismatch")
    return path


def _raw_manifest(config: Mapping[str, Any], root: Path) -> tuple[list[Path], dict[str, Any]]:
    paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for entry in config["session1_inputs"]["raw_files"]:
        subject = int(entry["subject"])
        path = _verify_entry(root, entry, f"session1 raw subject {subject}")
        paths.append(path)
        records.append(
            {
                "subject": subject,
                "filename": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if sum(record["bytes"] for record in records) != int(config["session1_inputs"]["total_bytes"]):
        raise TrajectoryAuditDataError("session1 raw total byte count mismatch")
    manifest = {
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
    }
    manifest["manifest_sha256"] = stable_json_sha256(manifest)
    return paths, manifest


def _augment_metadata(raw_metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    missing = {"subject", "session", "run"} - set(raw_metadata.columns)
    if missing:
        raise TrajectoryAuditDataError(f"MOABB metadata missing: {sorted(missing)}")
    frame = raw_metadata.loc[:, ["subject", "session", "run"]].copy().reset_index(drop=True)
    frame["subject"] = pd.to_numeric(frame["subject"], errors="raise").astype(int)
    frame["session"] = frame["session"].astype(str)
    frame["run"] = pd.to_numeric(frame["run"], errors="raise").astype(int)
    frame.insert(0, "sample_index", np.arange(len(frame), dtype=np.int64))
    frame["trial_id"] = (
        frame.groupby(["subject", "session"], sort=False).cumcount() + 1
    ).astype(np.int64)
    frame["run_trial_id"] = (
        frame.groupby(["subject", "session", "run"], sort=False).cumcount() + 1
    ).astype(np.int64)
    frame["trial_uid"] = [
        f"S{int(subject):02d}_{session}_T{int(trial):03d}"
        for subject, session, trial in frame[["subject", "session", "trial_id"]].itertuples(
            index=False, name=None
        )
    ]
    frame["class_label"] = np.asarray(labels).astype(str)
    return frame


def _validate_one_session(frame: pd.DataFrame, session: str) -> None:
    if len(frame) != 2592 or set(frame["session"].astype(str)) != {session}:
        raise TrajectoryAuditDataError(f"{session} metadata count/session mismatch")
    if not np.array_equal(frame["sample_index"].to_numpy(dtype=int), np.arange(2592)):
        raise TrajectoryAuditDataError(f"{session} sample_index is not canonical")
    if tuple(sorted(frame["subject"].astype(int).unique())) != tuple(range(1, 10)):
        raise TrajectoryAuditDataError(f"{session} subject set mismatch")
    if tuple(sorted(frame["run"].astype(int).unique())) != tuple(range(6)):
        raise TrajectoryAuditDataError(f"{session} run set mismatch")
    if set(frame["class_label"].astype(str).unique()) != set(CLASS_ORDER):
        raise TrajectoryAuditDataError(f"{session} class set mismatch")
    checks = (
        (["subject"], 288),
        (["subject", "class_label"], 72),
        (["subject", "run"], 48),
        (["subject", "run", "class_label"], 12),
    )
    for columns, expected in checks:
        counts = frame.groupby(columns, sort=True, observed=True).size()
        if counts.empty or not (counts == expected).all():
            raise TrajectoryAuditDataError(f"{session} balanced count failed: {columns}")
    if frame["trial_uid"].duplicated().any():
        raise TrajectoryAuditDataError(f"{session} trial UIDs are not unique")


def _load_session1_epochs(
    config: Mapping[str, Any], root: Path, raw_paths: list[Path]
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, float, tuple[float, float]]:
    moabb_dir = (root / str(config["session1_inputs"]["raw_moabb_dir"])).resolve()
    os.environ["MNE_DATA"] = str(moabb_dir)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(moabb_dir)
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.datasets.bnci.base import _convert_mi
    from moabb.paradigms import MotorImagery

    mne.set_log_level("WARNING")
    subjects = list(range(1, 10))
    channels = [str(value) for value in config["dataset"]["eeg_channels"]]
    raw_by_subject = dict(zip(subjects, raw_paths, strict=True))

    class _Session1OnlyBNCI(BNCI2014_001):
        def _get_single_subject_data(self, subject: int) -> dict[str, Any]:
            runs, _ = _convert_mi(
                str(raw_by_subject[int(subject)]),
                channels + ["EOG1", "EOG2", "EOG3"],
                ["eeg"] * 22 + ["eog"] * 3,
                dataset_code="BNCI2014-001",
                subject_id=int(subject),
            )
            return {"1test": {str(index): run for index, run in enumerate(runs)}}

    dataset = _Session1OnlyBNCI(subjects=subjects, sessions=["1test"])
    paradigm = MotorImagery(
        n_classes=4,
        events=list(CLASS_ORDER),
        fmin=8.0,
        fmax=32.0,
        tmin=0.0,
        tmax=3.996,
        baseline=None,
        channels=channels,
        resample=None,
    )
    epochs, labels, raw_metadata = paradigm.get_data(
        dataset=dataset, subjects=subjects, return_epochs=True
    )
    if set(epochs.ch_names) != set(channels):
        raise TrajectoryAuditDataError("session1 channel vocabulary mismatch")
    if epochs.ch_names != channels:
        epochs.reorder_channels(channels)
    if any(kind != "eeg" for kind in epochs.get_channel_types()):
        raise TrajectoryAuditDataError("non-EEG channel survived session1 selection")
    eeg = epochs.get_data(copy=True).astype(np.float32, copy=False)
    frame = _augment_metadata(raw_metadata, labels)
    _validate_one_session(frame, "1test")
    return (
        eeg,
        frame,
        np.asarray(channels, dtype=str),
        float(epochs.info["sfreq"]),
        (float(epochs.times[0]), float(epochs.times[-1])),
    )


def _load_or_prepare_session1_epochs(
    config: Mapping[str, Any], root: Path, raw_paths: list[Path], raw_manifest: Mapping[str, Any]
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]:
    cache_path = root / str(config["session1_inputs"]["prepared_epoch_cache"])
    metadata_path = root / str(config["session1_inputs"]["prepared_metadata_cache"])
    provenance_path = cache_path.with_name("session1_prepared_provenance.json")
    expected_identity = {
        "protocol_sha256": str(config["protocol"]["sha256"]),
        "config_sha256": sha256_file(root / "configs/bnci2014_001_trajectory_within_subject_v1.yaml"),
        "raw_manifest_sha256": str(raw_manifest["manifest_sha256"]),
    }
    if cache_path.is_file() and metadata_path.is_file() and provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if any(provenance.get(key) != value for key, value in expected_identity.items()):
                raise TrajectoryAuditDataError("session1 prepared cache identity mismatch")
            if sha256_file(cache_path) != provenance["cache_sha256"]:
                raise TrajectoryAuditDataError("session1 prepared cache file hash mismatch")
            if sha256_file(metadata_path) != provenance["metadata_sha256"]:
                raise TrajectoryAuditDataError("session1 prepared metadata hash mismatch")
            with np.load(cache_path, allow_pickle=False) as archive:
                if archive.files != ["eeg", "channel_names", "sampling_frequency_hz"]:
                    raise TrajectoryAuditDataError("session1 prepared cache keys changed")
                eeg = np.asarray(archive["eeg"])
                channels = np.asarray(archive["channel_names"]).astype(str)
                sampling = float(np.asarray(archive["sampling_frequency_hz"]))
            frame = pd.read_csv(metadata_path, dtype={"session": str, "trial_uid": str, "class_label": str})
            _validate_one_session(frame, "1test")
            if eeg.shape != (2592, 22, 1000) or eeg.dtype != np.dtype("float32"):
                raise TrajectoryAuditDataError("session1 prepared cache shape/dtype mismatch")
            if sampling != 250.0 or not np.isfinite(eeg).all():
                raise TrajectoryAuditDataError("session1 prepared cache sampling/finite gate failed")
            return eeg, frame, channels, provenance
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise TrajectoryAuditDataError("cannot validate existing session1 prepared cache") from error

    eeg, frame, channels, sampling, source_times = _load_session1_epochs(
        config, root, raw_paths
    )
    if eeg.shape != (2592, 22, 1000) or eeg.dtype != np.dtype("float32"):
        raise TrajectoryAuditDataError("session1 epoch shape/dtype differs from frozen contract")
    if sampling != 250.0 or not np.isfinite(eeg).all():
        raise TrajectoryAuditDataError("session1 epoch sampling/finite gate failed")
    if not np.allclose(source_times, (2.0, 5.996), atol=0.002, rtol=0.0):
        raise TrajectoryAuditDataError("session1 source epoch interval mismatch")
    _atomic_npz(
        cache_path,
        {
            "eeg": eeg,
            "channel_names": channels,
            "sampling_frequency_hz": np.asarray(sampling, dtype=np.float64),
        },
        compressed=False,
    )
    _atomic_text(metadata_path, frame.to_csv(index=False, lineterminator="\n"))
    provenance = {
        **expected_identity,
        "source": "hash_pinned_session1_E_files_via_moabb",
        "cache_sha256": sha256_file(cache_path),
        "metadata_sha256": sha256_file(metadata_path),
        "eeg_content_sha256": sha256_array(eeg),
        "shape": list(eeg.shape),
        "dtype": str(eeg.dtype),
        "sampling_frequency_hz": sampling,
        "source_times_seconds": list(source_times),
        "trial_rejection": False,
    }
    _atomic_text(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return eeg, frame, channels, provenance


def _window_covariances(
    eeg: np.ndarray, cache_path: Path, identity: Mapping[str, Any], progress: Callable[[str], None]
) -> tuple[np.ndarray, dict[str, Any]]:
    provenance_path = cache_path.with_suffix(".provenance.json")
    if cache_path.is_file() and provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if any(provenance.get(key) != value for key, value in identity.items()):
            raise TrajectoryAuditDataError("session1 covariance cache identity mismatch")
        if sha256_file(cache_path) != provenance["cache_sha256"]:
            raise TrajectoryAuditDataError("session1 covariance cache file hash mismatch")
        with np.load(cache_path, allow_pickle=False) as archive:
            if archive.files != ["states"]:
                raise TrajectoryAuditDataError("session1 covariance cache keys changed")
            states = np.asarray(archive["states"])
        if states.shape != (2592, 5, 22, 22) or states.dtype != np.dtype("float64"):
            raise TrajectoryAuditDataError("session1 covariance cache shape/dtype mismatch")
        return states, provenance

    progress("estimating session1 OAS: 12,960 frozen 200-sample windows")
    windows = eeg.reshape(2592, 22, 5, 200).transpose(0, 2, 1, 3).reshape(-1, 22, 200)
    estimator = Covariances(estimator="oas")
    states = np.empty((len(windows), 22, 22), dtype=np.float64)
    batch_size = 256
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        batch = estimator.transform(np.asarray(windows[start:stop], dtype=np.float64))
        states[start:stop] = 0.5 * (batch + batch.transpose(0, 2, 1))
        if stop % 1024 == 0 or stop == len(windows):
            progress(f"session1 OAS {stop}/{len(windows)}")
    states = states.reshape(2592, 5, 22, 22)
    _atomic_npz(cache_path, {"states": states}, compressed=True)
    provenance = {
        **identity,
        "cache_sha256": sha256_file(cache_path),
        "states_content_sha256": sha256_array(states),
        "shape": list(states.shape),
        "dtype": str(states.dtype),
        "estimator": "pyriemann.estimation.Covariances(estimator=oas)",
        "extra_regularization": "none",
        "eigenvalue_clipping": False,
    }
    _atomic_text(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return states, provenance


def _feature_arrays_from_states(
    states: np.ndarray,
    cache_path: Path,
    identity: Mapping[str, Any],
    progress: Callable[[str], None],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    partial_path = cache_path.with_name(cache_path.stem + ".partial.npz")
    provenance_path = cache_path.with_suffix(".provenance.json")
    n_trials = len(states)
    arrays: dict[str, np.ndarray]
    if cache_path.is_file() and provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if any(provenance.get(key) != value for key, value in identity.items()):
            raise TrajectoryAuditDataError("trajectory feature cache identity mismatch")
        if sha256_file(cache_path) != provenance["cache_sha256"]:
            raise TrajectoryAuditDataError("trajectory feature cache hash mismatch")
        with np.load(cache_path, allow_pickle=False) as archive:
            if tuple(archive.files) != FEATURE_KEYS:
                raise TrajectoryAuditDataError("trajectory feature cache keys changed")
            arrays = {key: np.asarray(archive[key]) for key in FEATURE_KEYS}
        return arrays, provenance

    shapes = {
        "airm_distance_matrices": (n_trials, 5, 5),
        "airm_path_d10": (n_trials, 10),
        "airm_bag_canon_d10": (n_trials, 10),
        "airm_scalars_11": (n_trials, 11),
        "airm_canonical_permutation": (n_trials, 5),
    }
    completed = np.zeros(n_trials, dtype=bool)
    arrays = {
        key: np.full(shape, -1 if key.endswith("permutation") else np.nan, dtype=np.int64 if key.endswith("permutation") else np.float64)
        for key, shape in shapes.items()
    }
    if partial_path.is_file():
        with np.load(partial_path, allow_pickle=False) as archive:
            expected = (*FEATURE_KEYS, "completed", "identity_sha256")
            if tuple(archive.files) != expected:
                raise TrajectoryAuditDataError("partial feature cache keys changed")
            if str(np.asarray(archive["identity_sha256"])[0]) != stable_json_sha256(identity):
                raise TrajectoryAuditDataError("partial feature cache identity mismatch")
            completed = np.asarray(archive["completed"], dtype=bool)
            arrays = {key: np.asarray(archive[key]) for key in FEATURE_KEYS}
    progress(f"AIRM trajectory features resume {int(completed.sum())}/{n_trials}")
    required_gate_failures: list[int] = []
    for index in np.flatnonzero(~completed):
        result = compute_five_state_geometry(states[index], "AIRM")
        gates = trajectory_hard_checks(states[index], result, metric="AIRM")
        if not gates.passed:
            required_gate_failures.append(int(index))
            break
        arrays["airm_distance_matrices"][index] = result.distance_matrix
        arrays["airm_path_d10"][index] = result.path_d10
        arrays["airm_bag_canon_d10"][index] = result.bag_canon_d10
        arrays["airm_scalars_11"][index] = result.scalar_vector
        arrays["airm_canonical_permutation"][index] = result.bag_canon.permutation
        completed[index] = True
        if (index + 1) % 50 == 0 or index + 1 == n_trials:
            _atomic_npz(
                partial_path,
                {
                    **arrays,
                    "completed": completed,
                    "identity_sha256": np.asarray([stable_json_sha256(identity)], dtype=str),
                },
                compressed=True,
            )
            progress(f"AIRM trajectory features {index + 1}/{n_trials}")
    if required_gate_failures or not completed.all():
        raise TrajectoryAuditDataError(
            f"trajectory geometry hard gate failed at trials {required_gate_failures[:10]}"
        )
    # Match v0: exhaustive BAG invariance on the smallest trial per subject/class (36 rows).
    bag_positions = [subject * 288 + class_index * 12 for subject in range(9) for class_index in range(4)]
    bag_maximum = 0.0
    for position in bag_positions:
        check = check_bag_permutation_invariance(arrays["airm_distance_matrices"][position])
        bag_maximum = max(bag_maximum, float(check.maximum_absolute_error))
        if not check.passed:
            raise TrajectoryAuditDataError(f"BAG invariance failed at trial {position}")
    _atomic_npz(cache_path, arrays, compressed=True)
    provenance = {
        **identity,
        "cache_sha256": sha256_file(cache_path),
        "array_content_sha256": {key: sha256_array(value) for key, value in arrays.items()},
        "trial_count": n_trials,
        "all_trajectory_hard_gates_passed": True,
        "bag_validation_trials": len(bag_positions),
        "bag_validation_permutations": 120,
        "bag_maximum_absolute_error": bag_maximum,
    }
    _atomic_text(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    if partial_path.exists():
        partial_path.unlink()
    return arrays, provenance


def _session0_data(
    config: Mapping[str, Any], root: Path, progress: Callable[[str], None]
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    for key, entry in config["reference_inputs"].items():
        if key == "v0_window_covariances":
            path = root / str(entry["path"])
            if sha256_file(path) != str(entry["file_sha256"]):
                raise TrajectoryAuditDataError("v0 WINDOW5 covariance file SHA-256 mismatch")
        elif key == "v0_window_metadata":
            _verify_entry(root, entry, key)
        else:
            _verify_entry(root, entry, key)
    v0_config_path = root / str(config["reference_inputs"]["v0_config"]["path"])
    v0_config = yaml.safe_load(v0_config_path.read_text(encoding="utf-8"))
    source = load_trajectory_window5(v0_config, root)
    reference_path = root / str(config["reference_inputs"]["v0_feature_cache"]["path"])
    with np.load(reference_path, allow_pickle=False) as archive:
        reference_arrays = {key: np.asarray(archive[key]) for key in FEATURE_KEYS}
        reference_metadata = pd.DataFrame(
            {
                "sample_index": np.asarray(archive["sample_index"], dtype=np.int64),
                "subject": np.asarray(archive["subject"], dtype=np.int64),
                "session": np.repeat(str(np.asarray(archive["session"])[0]), 2592),
                "run": np.asarray(archive["run"], dtype=np.int64),
                "trial_id": np.asarray(archive["trial_id"], dtype=np.int64),
                "trial_uid": np.asarray(archive["trial_uid"]).astype(str),
                "class_label": np.asarray(archive["class_label"]).astype(str),
            }
        )
    _validate_one_session(reference_metadata, "0train")
    gate_path = root / str(config["reference_inputs"]["v0_geometry_gate"]["path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("gate_passed") is not True or gate.get("status") != "PASS":
        raise TrajectoryAuditDataError("frozen v0 geometry gate is not PASS")
    identity = {
        "protocol_sha256": str(config["protocol"]["sha256"]),
        "config_sha256": sha256_file(root / "configs/bnci2014_001_trajectory_within_subject_v1.yaml"),
        "v0_feature_cache_sha256": sha256_file(reference_path),
        "v0_window5_content_sha256": sha256_array(source.states),
    }
    expected_window_hash = str(config["reference_inputs"]["v0_window_covariances"]["window5_content_sha256"])
    # v0 uses raw-content hashing without dtype/shape prefix.
    raw_digest = hashlib.sha256(memoryview(np.ascontiguousarray(source.states)).cast("B")).hexdigest()
    if raw_digest != expected_window_hash:
        raise TrajectoryAuditDataError("v0 WINDOW5 content SHA-256 mismatch")
    cache_path = root / str(config["project"]["cache_dir"]) / "session0_reproduction_features.npz"
    progress("recomputing session0 AIRM trajectory features for hard gate 0")
    observed_arrays, feature_provenance = _feature_arrays_from_states(
        source.states, cache_path, identity, progress
    )
    reproduction = compare_reproduction(
        source.metadata,
        reference_metadata,
        observed_arrays,
        reference_arrays,
        atol=float(config["hard_gates"]["reproduction_absolute_tolerance"]),
    )
    reproduction = pd.concat(
        [
            pd.DataFrame.from_records(
                [
                    {
                        "check": "frozen_v0_geometry_gate",
                        "maximum_absolute_difference": 0.0,
                        "tolerance": 0.0,
                        "passed": True,
                    },
                    {
                        "check": "recomputed_session0_all_geometry_gates",
                        "maximum_absolute_difference": 0.0,
                        "tolerance": 0.0,
                        "passed": bool(feature_provenance["all_trajectory_hard_gates_passed"]),
                    },
                ]
            ),
            reproduction,
        ],
        ignore_index=True,
    )
    if not reproduction["passed"].all():
        raise TrajectoryAuditDataError("UNASSESSED_TRAJECTORY_REPRODUCTION_FAILURE")
    frame = reference_metadata.copy()
    frame.insert(0, "global_sample_index", np.arange(2592, dtype=np.int64))
    return frame, reference_arrays, reproduction, feature_provenance


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "pyriemann", "mne", "moabb", "matplotlib"
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def prepare_audit_data(
    config_path: str | Path,
    root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> PreparedAuditData:
    project_root = Path(root).resolve()
    config_file = Path(config_path).resolve()
    config = load_frozen_config(config_file)
    announce = progress or (lambda _message: None)
    raw_paths, raw_manifest = _raw_manifest(config, project_root)
    session0_metadata, session0_arrays, reproduction, session0_provenance = _session0_data(
        config, project_root, announce
    )
    announce("loading/preparing hash-pinned session1 epochs")
    eeg, session1_metadata, channels, epoch_provenance = _load_or_prepare_session1_epochs(
        config, project_root, raw_paths, raw_manifest
    )
    covariance_identity = {
        "protocol_sha256": str(config["protocol"]["sha256"]),
        "config_sha256": sha256_file(config_file),
        "prepared_eeg_content_sha256": sha256_array(eeg),
        "raw_manifest_sha256": str(raw_manifest["manifest_sha256"]),
    }
    cache_root = project_root / str(config["project"]["cache_dir"])
    states, covariance_provenance = _window_covariances(
        eeg, cache_root / "session1_window_covariances.npz", covariance_identity, announce
    )
    feature_identity = {
        **covariance_identity,
        "window_states_content_sha256": sha256_array(states),
    }
    session1_cache = project_root / str(config["session1_inputs"]["trajectory_feature_cache"])
    announce("computing/reusing session1 frozen AIRM trajectory features")
    session1_arrays, session1_feature_provenance = _feature_arrays_from_states(
        states, session1_cache, feature_identity, announce
    )
    if not np.array_equal(channels, np.asarray(config["dataset"]["eeg_channels"], dtype=str)):
        raise TrajectoryAuditDataError("session1 channel order differs from frozen v0 order")
    session1_metadata = session1_metadata.loc[:, [column for column in IDENTITY_COLUMNS if column != "global_sample_index"]]
    session1_metadata.insert(0, "global_sample_index", np.arange(2592, 5184, dtype=np.int64))
    combined_metadata = pd.concat([session0_metadata, session1_metadata], ignore_index=True)
    combined_metadata = validate_metadata(combined_metadata, config)
    combined_arrays = {
        key: np.concatenate([session0_arrays[key], session1_arrays[key]], axis=0)
        for key in FEATURE_KEYS
    }
    for value in combined_arrays.values():
        value.setflags(write=False)
    session1_gate_row = {
        "check": "session1_all_geometry_gates",
        "maximum_absolute_difference": 0.0,
        "tolerance": 0.0,
        "passed": bool(session1_feature_provenance["all_trajectory_hard_gates_passed"]),
    }
    reproduction = pd.concat([reproduction, pd.DataFrame.from_records([session1_gate_row])], ignore_index=True)
    data_contract_records = [
        {"check": "combined_trials", "observed": len(combined_metadata), "expected": 5184, "passed": len(combined_metadata) == 5184},
        {"check": "session_count", "observed": combined_metadata.session.nunique(), "expected": 2, "passed": combined_metadata.session.nunique() == 2},
        {"check": "subjects", "observed": combined_metadata.subject.nunique(), "expected": 9, "passed": combined_metadata.subject.nunique() == 9},
        {"check": "classes", "observed": combined_metadata.class_label.nunique(), "expected": 4, "passed": combined_metadata.class_label.nunique() == 4},
        {"check": "session1_raw_files", "observed": len(raw_paths), "expected": 9, "passed": len(raw_paths) == 9},
        {"check": "channel_order", "observed": json.dumps(channels.tolist()), "expected": json.dumps(config["dataset"]["eeg_channels"]), "passed": np.array_equal(channels, np.asarray(config["dataset"]["eeg_channels"]))},
        {"check": "window_count", "observed": states.shape[1], "expected": 5, "passed": states.shape[1] == 5},
        {"check": "samples_per_window", "observed": 200, "expected": 200, "passed": True},
    ]
    data_contract = pd.DataFrame.from_records(data_contract_records)
    if not data_contract["passed"].all() or not reproduction["passed"].all():
        raise TrajectoryAuditDataError("UNASSESSED_TRAJECTORY_REPRODUCTION_FAILURE")
    combined_cache = cache_root / "combined_trajectory_features.npz"
    combined_payload = {
        **combined_arrays,
        **{
            column: combined_metadata[column].to_numpy(dtype=str if column in {"session", "trial_uid", "class_label"} else np.int64)
            for column in IDENTITY_COLUMNS
        },
        "protocol_sha256": np.asarray([config["protocol"]["sha256"]], dtype=str),
        "config_sha256": np.asarray([sha256_file(config_file)], dtype=str),
        "reproduction_gate_passed": np.asarray([True], dtype=bool),
    }
    _atomic_npz(combined_cache, combined_payload, compressed=True)
    output_root = project_root / str(config["project"]["output_dir"])
    (output_root / "protocol").mkdir(parents=True, exist_ok=True)
    _atomic_text(output_root / "tables" / "data_contract.csv", data_contract.to_csv(index=False, lineterminator="\n"))
    _atomic_text(output_root / "tables" / "reproduction_gate.csv", reproduction.to_csv(index=False, lineterminator="\n"))
    shutil.copyfile(config_file, output_root / "protocol" / "frozen_config.yaml")
    shutil.copyfile(project_root / str(config["protocol"]["path"]), output_root / "protocol" / "PROTOCOL_TRAJECTORY_WITHIN_SUBJECT_AUDIT_V1.md")
    _atomic_text(output_root / "protocol" / "environment.json", json.dumps(_environment(), indent=2, sort_keys=True) + "\n")
    provenance = {
        "protocol_sha256": str(config["protocol"]["sha256"]),
        "config_sha256": sha256_file(config_file),
        "reference_commit": str(config["protocol"]["reference_commit"]),
        "protocol_freeze_commit": _git(project_root, "rev-parse", "HEAD"),
        "branch": _git(project_root, "branch", "--show-current"),
        "generated_at_utc": _utc_now(),
        "raw_manifest": raw_manifest,
        "epoch_provenance": epoch_provenance,
        "covariance_provenance": covariance_provenance,
        "session0_reproduction_provenance": session0_provenance,
        "session1_feature_provenance": session1_feature_provenance,
        "combined_cache": _file_record(combined_cache),
        "combined_array_content_sha256": {
            key: sha256_array(value) for key, value in combined_arrays.items()
        },
        "scientific_scores_computed": False,
        "whole_subject_class_interaction_used": False,
    }
    _atomic_text(output_root / "protocol" / "data_provenance.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return PreparedAuditData(
        metadata=combined_metadata,
        arrays=combined_arrays,
        data_contract=data_contract,
        reproduction_gate=reproduction,
        provenance=provenance,
        combined_cache_path=combined_cache,
    )


def load_prepared_audit_data(
    config_path: str | Path, root: str | Path
) -> PreparedAuditData:
    project_root = Path(root).resolve()
    config_file = Path(config_path).resolve()
    config = load_frozen_config(config_file)
    cache_path = project_root / str(config["project"]["cache_dir"]) / "combined_trajectory_features.npz"
    output_root = project_root / str(config["project"]["output_dir"])
    data_contract = pd.read_csv(output_root / "tables" / "data_contract.csv")
    reproduction = pd.read_csv(output_root / "tables" / "reproduction_gate.csv")
    provenance = json.loads((output_root / "protocol" / "data_provenance.json").read_text(encoding="utf-8"))
    if not data_contract["passed"].astype(bool).all() or not reproduction["passed"].astype(bool).all():
        raise TrajectoryAuditDataError("prepared audit gates are not PASS")
    if sha256_file(cache_path) != provenance["combined_cache"]["sha256"]:
        raise TrajectoryAuditDataError("combined cache SHA-256 mismatch")
    with np.load(cache_path, allow_pickle=False) as archive:
        if str(np.asarray(archive["protocol_sha256"])[0]) != str(config["protocol"]["sha256"]):
            raise TrajectoryAuditDataError("combined cache protocol mismatch")
        if str(np.asarray(archive["config_sha256"])[0]) != sha256_file(config_file):
            raise TrajectoryAuditDataError("combined cache config mismatch")
        if not bool(np.asarray(archive["reproduction_gate_passed"])[0]):
            raise TrajectoryAuditDataError("combined cache reproduction gate is not PASS")
        arrays = {key: np.asarray(archive[key]) for key in FEATURE_KEYS}
        metadata = pd.DataFrame(
            {column: np.asarray(archive[column]).astype(str if column in {"session", "trial_uid", "class_label"} else np.int64) for column in IDENTITY_COLUMNS}
        )
    metadata = validate_metadata(metadata, config)
    for key, value in arrays.items():
        if sha256_array(value) != provenance["combined_array_content_sha256"][key]:
            raise TrajectoryAuditDataError(f"combined array content hash mismatch: {key}")
        value.setflags(write=False)
    return PreparedAuditData(metadata, arrays, data_contract, reproduction, provenance, cache_path)


__all__ = [
    "FEATURE_KEYS",
    "TrajectoryAuditDataError",
    "PreparedAuditData",
    "prepare_audit_data",
    "load_prepared_audit_data",
]
