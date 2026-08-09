"""Strict, read-only V1 WINDOW5 loader for Trajectory Anatomy v0.

The loader deliberately validates the session-bearing metadata before opening
the covariance archive.  It never downloads data, never activates the V1
prepared-data fallback, and never writes to a V1, V2, or trajectory directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


WINDOW_METADATA_COLUMNS = (
    "covariance_index",
    "sample_index",
    "subject",
    "session",
    "run",
    "trial_id",
    "run_trial_id",
    "trial_uid",
    "class_label",
    "window_index",
)
TRIAL_METADATA_COLUMNS = (
    "sample_index",
    "subject",
    "session",
    "run",
    "trial_id",
    "run_trial_id",
    "trial_uid",
    "class_label",
)
WHOLE_METADATA_COLUMNS = ("covariance_index",) + TRIAL_METADATA_COLUMNS


class TrajectoryDataContractError(RuntimeError):
    """Raised when an exact V1/session/data contract is not satisfied."""


@dataclass(frozen=True)
class TrajectoryWindowData:
    """Validated raw five-state covariances in immutable trial-major order."""

    _states: np.ndarray
    _whole_covariances: np.ndarray
    _metadata: pd.DataFrame
    _channel_names: np.ndarray
    _provenance: Mapping[str, Any]

    @property
    def states(self) -> np.ndarray:
        """Return the read-only ``(trials, 5, channels, channels)`` array."""

        return self._states

    @property
    def covariances(self) -> np.ndarray:
        """Alias for :attr:`states`, used by geometry callers."""

        return self._states

    @property
    def whole_covariances(self) -> np.ndarray:
        """Return aligned, read-only V1 WHOLE-1000 covariance controls."""

        return self._whole_covariances

    @property
    def metadata(self) -> pd.DataFrame:
        """Return a defensive copy of the one-row-per-trial metadata."""

        return self._metadata.copy(deep=True)

    @property
    def channel_names(self) -> np.ndarray:
        """Return the read-only channel-name array in frozen order."""

        return self._channel_names

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible provenance mapping."""

        return copy.deepcopy(dict(self._provenance))


def _fail(message: str) -> None:
    raise TrajectoryDataContractError(message)


def file_sha256(path: str | Path) -> str:
    """Hash a file without following any fallback or mutating it."""

    resolved = Path(path)
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as error:
        raise TrajectoryDataContractError(
            f"required frozen input is missing: {resolved}"
        ) from error
    return digest.hexdigest()


def array_content_sha256(array: np.ndarray) -> str:
    """Hash the raw C-contiguous bytes, matching the frozen V1 hashes."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def trial_uid_set_sha256(trial_uids: list[str] | tuple[str, ...]) -> str:
    """Hash the sorted unique UID set with the V2 canonical serialization."""

    values = [str(value) for value in trial_uids]
    if not values or len(values) != len(set(values)) or any(not value for value in values):
        _fail("trial_uid values must be non-empty and globally unique")
    payload = json.dumps(
        {"trial_uids": sorted(values)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_mapping(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        _fail(f"trajectory config section {name!r} is missing or not a mapping")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "project",
        "v1_inputs",
        "dataset",
        "preprocessing",
        "window5",
        "geometry",
        "representations",
        "class_probe",
        "nulls",
        "subject_probe",
        "factor_decomposition",
        "controls",
        "verdicts",
    }
    missing = required - set(config)
    if missing:
        _fail(f"trajectory config is missing sections: {sorted(missing)}")

    project = _require_mapping(config, "project")
    dataset = _require_mapping(config, "dataset")
    window = _require_mapping(config, "window5")
    preprocessing = _require_mapping(config, "preprocessing")
    if str(project.get("protocol_version")) != "0.0":
        _fail("Trajectory Anatomy v0 requires protocol_version '0.0'")
    if int(project.get("seed", -1)) != 20260809:
        _fail("Trajectory Anatomy v0 requires master seed 20260809")
    if str(dataset.get("allowed_session")) != "0train":
        _fail("allowed_session must be exactly '0train'")
    if str(dataset.get("forbidden_session")) != "1test":
        _fail("forbidden_session must be exactly '1test'")
    if int(window.get("n_windows", -1)) != 5:
        _fail("WINDOW5 requires exactly five windows")
    if list(window.get("chronological_order", [])) != [1, 2, 3, 4, 5]:
        _fail("chronological window order must be exactly 1..5")
    if int(window.get("overlap_samples", -1)) != 0:
        _fail("WINDOW5 overlap must be zero")
    if str(window.get("remainder_policy")) != "require_exact_division":
        _fail("WINDOW5 remainder policy must require exact division")
    if int(preprocessing.get("samples_per_trial", -1)) != (
        int(window.get("n_windows", -1)) * int(window.get("samples_per_window", -1))
    ):
        _fail("samples_per_trial is inconsistent with the frozen WINDOW5 grid")
    if str(preprocessing.get("covariance_dtype")) != "float64":
        _fail("covariance dtype must be float64")
    if str(preprocessing.get("covariance_estimator")) != "oas":
        _fail("covariance estimator must be OAS")
    if bool(preprocessing.get("eigenvalue_clipping")):
        _fail("eigenvalue clipping is forbidden")

    classes = [str(value) for value in dataset.get("classes", [])]
    subjects = [int(value) for value in dataset.get("subjects", [])]
    runs = [int(value) for value in dataset.get("runs", [])]
    channels = [str(value) for value in dataset.get("eeg_channels", [])]
    if not classes or len(classes) != len(set(classes)):
        _fail("dataset classes must be non-empty and unique")
    if not subjects or len(subjects) != len(set(subjects)):
        _fail("dataset subjects must be non-empty and unique")
    if runs != sorted(runs) or len(runs) != len(set(runs)):
        _fail("dataset runs must be unique and numerically sorted")
    if not channels or len(channels) != len(set(channels)):
        _fail("EEG channels must be non-empty and unique")
    expected_trials = int(dataset.get("expected_trials", -1))
    if expected_trials != len(subjects) * int(
        dataset.get("expected_trials_per_subject", -1)
    ):
        _fail("subject trial counts do not sum to expected_trials")
    if int(dataset.get("expected_trials_per_subject", -1)) != len(runs) * int(
        dataset.get("expected_trials_per_subject_run", -1)
    ):
        _fail("run trial counts do not sum to expected_trials_per_subject")
    if int(dataset.get("expected_trials_per_subject", -1)) != len(classes) * int(
        dataset.get("expected_trials_per_subject_class", -1)
    ):
        _fail("class trial counts do not sum to expected_trials_per_subject")
    expected_flat = tuple(int(value) for value in window.get("expected_covariance_shape", []))
    expected_tensor = tuple(
        int(value) for value in window.get("expected_trial_tensor_shape", [])
    )
    if expected_flat != (expected_trials * 5, len(channels), len(channels)):
        _fail("expected WINDOW5 covariance shape is inconsistent")
    if expected_tensor != (expected_trials, 5, len(channels), len(channels)):
        _fail("expected trial tensor shape is inconsistent")


def load_trajectory_config(path: str | Path) -> dict[str, Any]:
    """Load the exact YAML config and verify its frozen protocol document."""

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise TrajectoryDataContractError(
            f"trajectory config is missing: {config_path}"
        ) from error
    if not isinstance(config, dict):
        _fail("trajectory config must be a YAML mapping")
    _validate_config(config)
    project = config["project"]
    protocol_path = _resolve(config_path.parent.parent, project["protocol_path"])
    observed = file_sha256(protocol_path)
    expected = str(project["protocol_sha256"])
    if observed != expected:
        _fail(
            "frozen trajectory protocol SHA-256 mismatch: "
            f"expected {expected}, observed {observed}"
        )
    return config


def _config_identity(
    config_or_path: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], str, str, Path | None]:
    if isinstance(config_or_path, (str, Path)):
        path = Path(config_or_path).expanduser().resolve()
        return load_trajectory_config(path), file_sha256(path), "yaml_file_sha256", path
    config = copy.deepcopy(dict(config_or_path))
    _validate_config(config)
    return config, _canonical_config_hash(config), "canonical_json_sha256", None


def _check_declared_file(entry: Mapping[str, Any], root: Path, hash_key: str) -> tuple[Path, str]:
    if "path" not in entry or hash_key not in entry:
        _fail(f"V1 input entry requires path and {hash_key}")
    path = _resolve(root, entry["path"])
    expected = str(entry[hash_key])
    observed = file_sha256(path)
    if observed != expected:
        _fail(
            f"V1 file SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )
    return path, observed


def _validate_support_hashes(config: Mapping[str, Any], root: Path) -> dict[str, str]:
    """Validate every frozen support file before reading numeric arrays."""

    inputs = _require_mapping(config, "v1_inputs")
    specs = {
        "source_config": "sha256",
        "frozen_config": "sha256",
        "whole_metadata": "sha256",
        "prepared_epochs": "file_sha256",
        "prepared_metadata": "sha256",
        "covariance_summary": "sha256",
        "dataset_metadata": "sha256",
    }
    observed: dict[str, str] = {}
    for name, hash_key in specs.items():
        entry = inputs.get(name)
        if not isinstance(entry, Mapping):
            _fail(f"V1 input {name!r} is missing")
        _, value = _check_declared_file(entry, root, hash_key)
        observed[f"{name}_sha256"] = value
    if observed["source_config_sha256"] != observed["frozen_config_sha256"]:
        _fail("V1 source and frozen config hashes differ")
    return observed


def _integer_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        _fail(f"metadata column {column!r} must contain finite integers")
    return values.astype(np.int64)


def _validate_window_metadata(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if tuple(frame.columns) != WINDOW_METADATA_COLUMNS:
        _fail(
            "WINDOW5 metadata columns/order mismatch: expected "
            f"{WINDOW_METADATA_COLUMNS}, observed {tuple(frame.columns)}"
        )
    if frame.empty or frame.isna().any(axis=None):
        _fail("WINDOW5 metadata is empty or contains null values")
    frame = frame.copy(deep=True).reset_index(drop=True)
    for column in (
        "covariance_index",
        "sample_index",
        "subject",
        "run",
        "trial_id",
        "run_trial_id",
        "window_index",
    ):
        frame[column] = _integer_column(frame, column)
    for column in ("session", "trial_uid", "class_label"):
        frame[column] = frame[column].astype(str)

    dataset = config["dataset"]
    allowed = str(dataset["allowed_session"])
    sessions = tuple(sorted(frame["session"].unique()))
    # This barrier intentionally executes before the covariance archive is opened.
    if sessions != (allowed,):
        _fail(
            f"forbidden session barrier: expected only {allowed!r}, observed {sessions}"
        )

    expected_trials = int(dataset["expected_trials"])
    n_windows = int(config["window5"]["n_windows"])
    expected_rows = expected_trials * n_windows
    if len(frame) != expected_rows:
        _fail(f"expected {expected_rows} WINDOW5 rows, observed {len(frame)}")
    covariance_index = frame["covariance_index"].to_numpy()
    sample_index = frame["sample_index"].to_numpy()
    window_index = frame["window_index"].to_numpy()
    if not np.array_equal(covariance_index, np.arange(expected_rows)):
        _fail("covariance_index must be contiguous and row-aligned")
    if not np.array_equal(sample_index, np.repeat(np.arange(expected_trials), n_windows)):
        _fail("sample_index must be trial-major and repeated exactly five times")
    if not np.array_equal(window_index, np.tile(np.arange(1, n_windows + 1), expected_trials)):
        _fail("window_index must be chronological 1..5 within every trial")
    if not np.array_equal(
        covariance_index, sample_index * n_windows + window_index - 1
    ):
        _fail("covariance_index/sample_index/window_index alignment failed")

    trial = frame.iloc[::n_windows].loc[:, TRIAL_METADATA_COLUMNS].reset_index(drop=True)
    for column in TRIAL_METADATA_COLUMNS:
        repeated = np.repeat(trial[column].to_numpy(), n_windows)
        if not np.array_equal(frame[column].to_numpy(), repeated):
            _fail(f"trial identity {column!r} changes across its five windows")
    if not np.array_equal(trial["sample_index"].to_numpy(), np.arange(expected_trials)):
        _fail("trial sample_index is not exactly 0..N-1")
    if trial["trial_uid"].duplicated().any():
        _fail("trial_uid is not globally unique")
    if trial.duplicated(["subject", "session", "trial_id"]).any():
        _fail("(subject, session, trial_id) does not uniquely identify trials")

    expected_subjects = tuple(int(value) for value in dataset["subjects"])
    expected_runs = tuple(int(value) for value in dataset["runs"])
    expected_classes = tuple(str(value) for value in dataset["classes"])
    if tuple(sorted(trial["subject"].unique())) != expected_subjects:
        _fail("observed subjects differ from the frozen subject set")
    if tuple(sorted(trial["run"].unique())) != expected_runs:
        _fail("observed runs differ from the frozen numeric run set")
    if set(trial["class_label"].unique()) != set(expected_classes):
        _fail("observed classes differ from the frozen class vocabulary")

    checks = (
        (["subject"], int(dataset["expected_trials_per_subject"]), "subject"),
        (
            ["subject", "class_label"],
            int(dataset["expected_trials_per_subject_class"]),
            "subject×class",
        ),
        (
            ["subject", "run"],
            int(dataset["expected_trials_per_subject_run"]),
            "subject×run",
        ),
        (
            ["subject", "run", "class_label"],
            int(dataset["expected_trials_per_subject_run_class"]),
            "subject×run×class",
        ),
    )
    for columns, expected_count, label in checks:
        counts = trial.groupby(columns, observed=True, sort=True).size()
        if len(counts) == 0 or not (counts == expected_count).all():
            _fail(f"{label} counts violate the frozen balance: {counts.to_dict()}")

    uid_hash = trial_uid_set_sha256(trial["trial_uid"].tolist())
    expected_uid_hash = str(dataset["expected_trial_uid_set_sha256"])
    if uid_hash != expected_uid_hash:
        _fail(
            "trial UID-set SHA-256 mismatch: "
            f"expected {expected_uid_hash}, observed {uid_hash}"
        )
    return frame, trial, uid_hash


def _validate_frozen_v1_config(path: Path, config: Mapping[str, Any]) -> None:
    try:
        v1 = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TrajectoryDataContractError(f"cannot parse frozen V1 config: {path}") from error
    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    window = config["window5"]
    comparisons = {
        "dataset.name": (v1["dataset"]["name"], dataset["name"]),
        "dataset.subjects": (v1["dataset"]["subjects"], dataset["subjects"]),
        "dataset.session": (
            v1["dataset"]["primary_session_key"],
            dataset["allowed_session"],
        ),
        "dataset.classes": (v1["dataset"]["classes"], dataset["classes"]),
        "dataset.channels": (v1["dataset"]["eeg_channels"], dataset["eeg_channels"]),
        "preprocessing.bandpass": (
            v1["preprocessing"]["bandpass_hz"],
            preprocessing["bandpass_hz"],
        ),
        "preprocessing.tmin": (
            v1["preprocessing"]["epoch_tmin_seconds"],
            preprocessing["epoch_tmin_seconds"],
        ),
        "preprocessing.tmax": (
            v1["preprocessing"]["epoch_tmax_seconds"],
            preprocessing["epoch_tmax_seconds"],
        ),
        "preprocessing.samples": (
            v1["preprocessing"]["expected_samples_per_trial"],
            preprocessing["samples_per_trial"],
        ),
        "preprocessing.sampling_frequency": (
            v1["preprocessing"]["expected_sampling_frequency_hz"],
            preprocessing["sampling_frequency_hz"],
        ),
        "preprocessing.resample": (
            v1["preprocessing"]["resample_hz"],
            preprocessing["resample_hz"],
        ),
        "preprocessing.baseline": (
            v1["preprocessing"]["baseline"],
            preprocessing["baseline"],
        ),
        "representation.estimator": (
            v1["representation"]["covariance_estimator"],
            preprocessing["covariance_estimator"],
        ),
        "representation.n_windows": (
            v1["representation"]["n_windows"],
            window["n_windows"],
        ),
        "representation.overlap": (
            v1["representation"]["overlap_samples"],
            window["overlap_samples"],
        ),
        "representation.remainder": (
            v1["representation"]["remainder_policy"],
            window["remainder_policy"],
        ),
        "representation.window_samples": (
            v1["representation"]["expected_window_samples"],
            window["samples_per_window"],
        ),
        "representation.extra_regularization": (
            v1["representation"]["covariance_extra_regularization"],
            preprocessing["extra_diagonal_loading"],
        ),
    }
    for label, (actual, expected) in comparisons.items():
        if actual != expected:
            _fail(f"frozen V1 config contract mismatch for {label}: {actual!r} != {expected!r}")


def _validate_whole_metadata(
    path: Path,
    trial_metadata: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Prove WHOLE rows are positionally identical to the WINDOW5 trial table."""

    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise TrajectoryDataContractError(f"cannot parse WHOLE metadata: {path}") from error
    if tuple(frame.columns) != WHOLE_METADATA_COLUMNS:
        _fail("WHOLE metadata columns/order differs from the frozen contract")
    if len(frame) != len(trial_metadata) or frame.isna().any(axis=None):
        _fail("WHOLE metadata count/null contract failed")
    for column in (
        "covariance_index",
        "sample_index",
        "subject",
        "run",
        "trial_id",
        "run_trial_id",
    ):
        frame[column] = _integer_column(frame, column)
    for column in ("session", "trial_uid", "class_label"):
        frame[column] = frame[column].astype(str)
    allowed = str(config["dataset"]["allowed_session"])
    if tuple(sorted(frame["session"].unique())) != (allowed,):
        _fail("WHOLE metadata crossed the forbidden-session barrier")
    expected_index = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["covariance_index"].to_numpy(), expected_index):
        _fail("WHOLE covariance_index is not contiguous and aligned")
    for column in TRIAL_METADATA_COLUMNS:
        if not np.array_equal(
            frame[column].to_numpy(), trial_metadata[column].to_numpy()
        ):
            _fail(f"WHOLE/WINDOW5 trial alignment failed for {column!r}")


def _validate_spd(states: np.ndarray, config: Mapping[str, Any]) -> None:
    flat = states.reshape((-1,) + states.shape[-2:])
    if not np.isfinite(flat).all():
        _fail("WINDOW5 covariance array contains NaN or Inf")
    difference = flat - flat.transpose(0, 2, 1)
    numerator = np.linalg.norm(difference, axis=(1, 2))
    denominator = np.maximum(
        np.linalg.norm(flat, axis=(1, 2)), np.finfo(np.float64).tiny
    )
    relative = numerator / denominator
    symmetry_max = float(config["geometry"]["hard_gate"]["symmetry_relative_error_max"])
    if np.any(relative > symmetry_max):
        _fail(
            "WINDOW5 covariance symmetry gate failed; maximum relative error="
            f"{float(relative.max()):.6g}"
        )
    eigenvalues = np.linalg.eigvalsh(0.5 * (flat + flat.transpose(0, 2, 1)))
    if np.any(eigenvalues[:, 0] <= 0.0):
        _fail(
            "WINDOW5 contains a non-SPD covariance; minimum eigenvalue="
            f"{float(eigenvalues[:, 0].min()):.6g}"
        )
    conditions = eigenvalues[:, -1] / eigenvalues[:, 0]
    condition_max = float(config["geometry"]["hard_gate"]["condition_number_max"])
    if np.any(conditions > condition_max):
        _fail(
            "WINDOW5 condition-number gate failed; maximum="
            f"{float(conditions.max()):.6g}"
        )


def load_trajectory_window5(
    config_or_path: Mapping[str, Any] | str | Path,
    root: str | Path,
) -> TrajectoryWindowData:
    """Validate and load only the frozen V1 0train WINDOW5 covariances.

    Metadata and all support hashes are checked before the covariance NPZ is
    opened.  A mismatch is a hard failure; this function has no write/fallback
    path by design.
    """

    config, config_hash, config_hash_kind, config_path = _config_identity(config_or_path)
    project_root = Path(root).expanduser().resolve()
    support_hashes = _validate_support_hashes(config, project_root)
    inputs = config["v1_inputs"]

    source_config_path = _resolve(project_root, inputs["source_config"]["path"])
    frozen_config_path = _resolve(project_root, inputs["frozen_config"]["path"])
    _validate_frozen_v1_config(source_config_path, config)
    _validate_frozen_v1_config(frozen_config_path, config)

    metadata_entry = inputs["window5_metadata"]
    metadata_path, metadata_hash = _check_declared_file(
        metadata_entry, project_root, "sha256"
    )
    try:
        raw_metadata = pd.read_csv(metadata_path, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise TrajectoryDataContractError(
            f"cannot parse WINDOW5 metadata: {metadata_path}"
        ) from error
    _, trial_metadata, uid_hash = _validate_window_metadata(raw_metadata, config)
    whole_metadata_path = _resolve(project_root, inputs["whole_metadata"]["path"])
    _validate_whole_metadata(whole_metadata_path, trial_metadata, config)

    # The forbidden-session barrier above must remain before this first NPZ access.
    covariance_entry = inputs["covariances"]
    covariance_path, covariance_file_hash = _check_declared_file(
        covariance_entry, project_root, "file_sha256"
    )
    try:
        with np.load(covariance_path, allow_pickle=False) as archive:
            if tuple(archive.files) != ("whole", "window5", "channel_names"):
                _fail(
                    "V1 covariance NPZ key/order mismatch: "
                    f"observed {tuple(archive.files)}"
                )
            flat = np.asarray(archive["window5"])
            whole = np.asarray(archive["whole"])
            channels = np.asarray(archive["channel_names"])
    except (OSError, ValueError) as error:
        raise TrajectoryDataContractError(
            f"cannot safely load V1 covariance archive: {covariance_path}"
        ) from error

    expected_flat = tuple(int(value) for value in config["window5"]["expected_covariance_shape"])
    expected_tensor = tuple(
        int(value) for value in config["window5"]["expected_trial_tensor_shape"]
    )
    if flat.shape != expected_flat or flat.dtype != np.dtype("float64"):
        _fail(
            f"WINDOW5 array shape/dtype mismatch: observed {flat.shape}/{flat.dtype}, "
            f"expected {expected_flat}/float64"
        )
    if whole.shape != (
        int(config["dataset"]["expected_trials"]),
        len(config["dataset"]["eeg_channels"]),
        len(config["dataset"]["eeg_channels"]),
    ) or whole.dtype != np.dtype("float64"):
        _fail("WHOLE array shape/dtype does not satisfy the frozen support contract")
    expected_channels = np.asarray(config["dataset"]["eeg_channels"], dtype=str)
    if channels.ndim != 1 or not np.array_equal(channels.astype(str), expected_channels):
        _fail("covariance archive channel names/order mismatch")

    content_checks = {
        "window5_content_sha256": (flat, covariance_entry["window5_content_sha256"]),
        "whole_content_sha256": (whole, covariance_entry["whole_content_sha256"]),
        "channel_names_content_sha256": (
            channels,
            covariance_entry["channel_names_content_sha256"],
        ),
    }
    observed_content: dict[str, str] = {}
    for label, (array, expected_hash) in content_checks.items():
        observed_hash = array_content_sha256(array)
        if observed_hash != str(expected_hash):
            _fail(
                f"V1 covariance member hash mismatch for {label}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        observed_content[label] = observed_hash

    states = np.ascontiguousarray(flat).reshape(expected_tensor)
    _validate_spd(states, config)
    states.setflags(write=False)
    whole = np.ascontiguousarray(whole)
    _validate_spd(whole, config)
    whole.setflags(write=False)
    channels = np.ascontiguousarray(channels.astype(str))
    channels.setflags(write=False)

    provenance = MappingProxyType(
        {
            "protocol_version": str(config["project"]["protocol_version"]),
            "protocol_sha256": str(config["project"]["protocol_sha256"]),
            "config_sha256": config_hash,
            "config_hash_kind": config_hash_kind,
            "config_path": str(config_path) if config_path is not None else None,
            "session": str(config["dataset"]["allowed_session"]),
            "covariances_path": str(covariance_path),
            "covariances_file_sha256": covariance_file_hash,
            "window5_metadata_path": str(metadata_path),
            "window5_metadata_sha256": metadata_hash,
            **support_hashes,
            **observed_content,
            "trial_uid_set_sha256": uid_hash,
            "window5_shape": list(flat.shape),
            "trial_tensor_shape": list(states.shape),
            "whole_covariance_shape": list(whole.shape),
            "covariance_dtype": str(states.dtype),
            "channel_names": channels.tolist(),
            "read_only": True,
            "fallback_used": False,
        }
    )
    return TrajectoryWindowData(states, whole, trial_metadata, channels, provenance)


# Concise alias for pipeline scripts.
load_window5_v0 = load_trajectory_window5


__all__ = [
    "WINDOW_METADATA_COLUMNS",
    "TRIAL_METADATA_COLUMNS",
    "WHOLE_METADATA_COLUMNS",
    "TrajectoryDataContractError",
    "TrajectoryWindowData",
    "file_sha256",
    "array_content_sha256",
    "trial_uid_set_sha256",
    "load_trajectory_config",
    "load_trajectory_window5",
    "load_window5_v0",
]
