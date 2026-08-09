"""Strict WHOLE-covariance data contract for Conditional Geometry v1.

Discovery is a read-only, hash-pinned reuse of the V1 ``0train`` WHOLE
covariances.  Confirmatory preparation has a separate entry point whose first
data-relevant operation is validation of ``confirmatory_unlock.json``.  The
module never computes WINDOW5 representations.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.conditional_provenance_v1 import (
    ConfirmatoryLockError,
    atomic_write_json,
    canonical_json_bytes,
    load_protocol_config,
    sha256_file,
    validate_confirmatory_unlock,
)


DISCOVERY_SESSION = "0train"
CONFIRMATORY_SESSION = "1test"
CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
RUN_ORDER = ("0", "1", "2", "3", "4", "5")
CHANNEL_COUNT = 22
TRIALS_PER_SESSION = 2592
TRIALS_PER_SUBJECT = 288
TRIALS_PER_RUN = 48
TRIALS_PER_CLASS = 72
TRIALS_PER_RUN_CLASS = 12
FALLBACK_BATCH_SIZE = 256
FALLBACK_REQUIRED_VERSIONS = {
    "numpy": "2.5.1",
    "scikit-learn": "1.9.0",
    "pyriemann": "0.12",
}
CONFIRMATORY_REQUIRED_VERSIONS = {
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "pandas": "3.0.5",
    "scikit-learn": "1.9.0",
    "pyriemann": "0.12",
    "mne": "1.12.1",
    "moabb": "1.5.0",
}


class ConditionalDataError(RuntimeError):
    """Base error for a frozen conditional-data contract violation."""


class DataContractError(ConditionalDataError):
    """A non-recoverable input, metadata, or numerical contract failure."""


class CovarianceIdentityError(DataContractError):
    """A discovery covariance-only failure eligible for frozen fallback."""


@dataclass(frozen=True)
class ConditionalWholeData:
    """Validated, trial-aligned WHOLE covariances for one frozen session."""

    session: str
    covariances: np.ndarray
    metadata: pd.DataFrame
    channel_names: np.ndarray
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _PreparedEpochs:
    eeg: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    channel_names: np.ndarray
    sampling_frequency_hz: float
    observed_source_times: tuple[float, float]


def _fail(message: str) -> None:
    raise DataContractError(message)


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _is_lower_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def validate_conditional_config(config: Mapping[str, Any]) -> None:
    required = {
        "protocol",
        "project",
        "dataset",
        "preprocessing",
        "expected_data",
        "v1_discovery_inputs",
        "confirmatory_inputs",
        "hard_gates",
    }
    missing = required - set(config)
    if missing:
        _fail(f"Conditional config is missing sections: {sorted(missing)}")
    for section in required:
        if not isinstance(config[section], Mapping):
            _fail(f"Conditional config section must be a mapping: {section}")

    dataset = config["dataset"]
    prep = config["preprocessing"]
    expected = config["expected_data"]
    if dataset.get("name") != "BNCI2014_001":
        _fail("Dataset must be BNCI2014_001")
    if str(dataset.get("discovery_session")) != DISCOVERY_SESSION:
        _fail("Discovery session must be exactly 0train")
    if str(dataset.get("confirmatory_session")) != CONFIRMATORY_SESSION:
        _fail("Confirmatory session must be exactly 1test")
    if [int(value) for value in dataset.get("subjects", [])] != list(range(1, 10)):
        _fail("Subjects must be the fixed integers 1..9")
    if tuple(str(value) for value in dataset.get("classes", [])) != CLASS_ORDER:
        _fail("Class order differs from the frozen semantic order")
    if tuple(str(value) for value in dataset.get("runs", [])) != RUN_ORDER:
        _fail("Run order differs from the frozen numeric order")
    channels = tuple(str(value) for value in dataset.get("eeg_channels", []))
    if len(channels) != CHANNEL_COUNT or len(set(channels)) != CHANNEL_COUNT:
        _fail("Exactly 22 unique frozen EEG channels are required")

    expected_prep = {
        "bandpass_hz": [8.0, 32.0],
        "epoch_tmin_seconds": 0.0,
        "epoch_tmax_seconds": 3.996,
        "sampling_frequency_hz": 250.0,
        "samples_per_trial": 1000,
        "resample_hz": None,
        "baseline": None,
        "covariance_estimator": "oas",
        "epoch_cache_dtype": "float32",
        "covariance_dtype": "float64",
        "extra_regularization": "none",
        "eigenvalue_clipping": False,
        "symmetrize_covariance": True,
        "representation": "WHOLE",
        "window5_forbidden": True,
    }
    for key, value in expected_prep.items():
        if prep.get(key) != value:
            _fail(f"Frozen preprocessing field {key} mismatch")

    expected_counts = {
        "trials_per_session": TRIALS_PER_SESSION,
        "trials_per_subject_session": TRIALS_PER_SUBJECT,
        "trials_per_subject_session_class": TRIALS_PER_CLASS,
        "runs_per_subject_session": len(RUN_ORDER),
        "trials_per_subject_session_run": TRIALS_PER_RUN,
        "trials_per_subject_session_run_class": TRIALS_PER_RUN_CLASS,
        "covariance_shape_per_session": [TRIALS_PER_SESSION, CHANNEL_COUNT, CHANNEL_COUNT],
    }
    for key, value in expected_counts.items():
        if expected.get(key) != value:
            _fail(f"Frozen expected_data field {key} mismatch")
    expected_columns = [
        "covariance_index",
        "sample_index",
        "subject",
        "session",
        "run",
        "trial_id",
        "run_trial_id",
        "trial_uid",
        "class_label",
    ]
    if list(expected.get("metadata_columns", [])) != expected_columns:
        _fail("Frozen metadata column order mismatch")

    discovery = config["v1_discovery_inputs"]
    for key in ("paths", "hashes", "fallback"):
        if not isinstance(discovery.get(key), Mapping):
            _fail(f"v1_discovery_inputs.{key} must be a mapping")
    required_paths = {
        "frozen_config",
        "dataset_metadata",
        "covariance_summary",
        "covariance_sanity",
        "covariances",
        "whole_metadata",
        "prepared_epochs",
        "prepared_metadata",
    }
    required_hashes = {
        "frozen_config_sha256",
        "dataset_metadata_sha256",
        "covariance_summary_sha256",
        "covariance_sanity_sha256",
        "covariances_file_sha256",
        "whole_array_content_sha256",
        "whole_metadata_sha256",
        "prepared_epochs_sha256",
        "prepared_metadata_sha256",
    }
    if required_paths - set(discovery["paths"]):
        _fail("Discovery input paths are incomplete")
    if required_hashes - set(discovery["hashes"]):
        _fail("Discovery input hashes are incomplete")
    for key in required_hashes:
        if not _is_lower_sha256(discovery["hashes"][key]):
            _fail(f"Invalid lowercase SHA-256 for discovery hash {key}")
    fallback = discovery["fallback"]
    if fallback.get("covariance_identity_only") is not True:
        _fail("Discovery fallback must be covariance-identity-only")
    if fallback.get("prepared_inputs_must_match") is not True:
        _fail("Discovery fallback prepared inputs must be hash-pinned")
    if fallback.get("raw_or_moabb_preunlock_forbidden") is not True:
        _fail("Raw/MOABB discovery fallback must remain forbidden")

    confirm = config["confirmatory_inputs"]
    raw_files = confirm.get("raw_files")
    if not isinstance(raw_files, list) or len(raw_files) != 9:
        _fail("Exactly nine confirmatory raw-file records are required")
    observed_subjects = [int(record.get("subject")) for record in raw_files]
    if observed_subjects != list(range(1, 10)):
        _fail("Confirmatory raw files must be ordered by subject 1..9")
    for record in raw_files:
        if int(record.get("bytes", 0)) <= 0 or not _is_lower_sha256(record.get("sha256")):
            _fail("Invalid confirmatory raw size or SHA-256")
    if not _is_lower_sha256(confirm.get("ordered_manifest_sha256")):
        _fail("Invalid confirmatory ordered manifest SHA-256")
    if int(confirm.get("total_bytes", 0)) != sum(
        int(record["bytes"]) for record in raw_files
    ):
        _fail("Confirmatory total raw bytes mismatch")


def load_conditional_config(
    config_path: str | Path, repo_root: str | Path
) -> tuple[dict[str, Any], Path, str, str]:
    try:
        config, path, _protocol_path, config_hash, protocol_hash = load_protocol_config(
            config_path, repo_root
        )
    except Exception as error:
        if isinstance(error, ConditionalDataError):
            raise
        raise DataContractError(str(error)) from error
    validate_conditional_config(config)
    return config, path, config_hash, protocol_hash


def _resolve_repo_path(root: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        _fail(f"{label} escapes repository root: {resolved}")
    return resolved


def _check_file_hash(path: Path, expected: str, label: str) -> str:
    try:
        observed = sha256_file(path)
    except Exception as error:
        raise DataContractError(str(error)) from error
    if observed != str(expected):
        _fail(f"{label} SHA-256 mismatch: expected {expected}, observed {observed}")
    return observed


def _read_metadata(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            dtype={
                "session": "string",
                "run": "string",
                "trial_uid": "string",
                "class_label": "string",
            },
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise DataContractError(f"Could not read metadata: {path}") from error


def validate_session_metadata(
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    session: str,
    *,
    prepared: bool = False,
    labels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Validate exact row order, identifiers, counts, balance, and labels."""

    frame = metadata.reset_index(drop=True).copy()
    columns = list(config["expected_data"]["metadata_columns"])
    expected_columns = columns[1:] if prepared else columns
    if list(frame.columns) != expected_columns:
        _fail(
            f"Metadata schema mismatch: expected {expected_columns}, observed {list(frame.columns)}"
        )
    if len(frame) != TRIALS_PER_SESSION:
        _fail(f"Metadata must contain exactly {TRIALS_PER_SESSION} rows")
    if frame.isna().any().any():
        _fail("Metadata contains missing values")

    subject_values = frame["subject"].to_numpy(dtype=np.int64)
    session_values = frame["session"].astype(str).to_numpy()
    run_values = frame["run"].astype(str).to_numpy()
    trial_values = frame["trial_id"].to_numpy(dtype=np.int64)
    run_trial_values = frame["run_trial_id"].to_numpy(dtype=np.int64)
    expected_subjects = np.repeat(np.arange(1, 10, dtype=np.int64), TRIALS_PER_SUBJECT)
    expected_runs_one_subject = np.repeat(np.asarray(RUN_ORDER), TRIALS_PER_RUN)
    expected_runs = np.tile(expected_runs_one_subject, 9)
    expected_trials = np.tile(np.arange(1, TRIALS_PER_SUBJECT + 1), 9)
    expected_run_trials = np.tile(np.arange(1, TRIALS_PER_RUN + 1), 9 * len(RUN_ORDER))
    if not np.array_equal(subject_values, expected_subjects):
        _fail("Metadata is not in canonical subject-major order")
    if not np.all(session_values == str(session)):
        _fail(f"Metadata contains a session other than {session}")
    if not np.array_equal(run_values, expected_runs):
        _fail("Metadata is not in canonical numeric run-major order")
    if not np.array_equal(trial_values, expected_trials):
        _fail("trial_id is not canonical 1..288 within subject")
    if not np.array_equal(run_trial_values, expected_run_trials):
        _fail("run_trial_id is not canonical 1..48 within run")

    sample_index = frame["sample_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_index, np.arange(len(frame), dtype=np.int64)):
        _fail("sample_index must be contiguous and row-aligned")
    if not prepared:
        covariance_index = frame["covariance_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(covariance_index, sample_index):
            _fail("covariance_index must equal sample_index")
    expected_uids = np.asarray(
        [
            f"S{subject:02d}_{session}_T{trial_id:03d}"
            for subject, trial_id in zip(subject_values, trial_values, strict=True)
        ]
    )
    if not np.array_equal(frame["trial_uid"].astype(str).to_numpy(), expected_uids):
        _fail("trial_uid values are not canonical")
    if frame["trial_uid"].duplicated().any():
        _fail("trial_uid values must be unique")

    class_values = frame["class_label"].astype(str)
    if set(class_values.unique()) != set(CLASS_ORDER):
        _fail("Metadata class vocabulary mismatch")
    subject_class = frame.groupby(["subject", "class_label"], observed=True).size()
    if not (subject_class == TRIALS_PER_CLASS).all() or len(subject_class) != 36:
        _fail("Subject/class counts differ from 72")
    run_class = frame.groupby(
        ["subject", "run", "class_label"], observed=True
    ).size()
    if not (run_class == TRIALS_PER_RUN_CLASS).all() or len(run_class) != 216:
        _fail("Subject/run/class counts differ from 12")
    if labels is not None and not np.array_equal(
        class_values.to_numpy(), np.asarray(labels).astype(str)
    ):
        _fail("Labels and metadata are not row-aligned")
    return frame


def subject_split_positions(
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    subject: int,
    split: str,
) -> np.ndarray:
    if split not in ("A", "B", "F"):
        raise ValueError("split must be A, B, or F")
    subject = int(subject)
    if subject not in range(1, 10):
        raise ValueError("subject must be in 1..9")
    runs = {str(value) for value in config["dataset"]["split_runs"][split]}
    mask = metadata["subject"].astype(int).eq(subject) & metadata["run"].astype(str).isin(runs)
    positions = np.flatnonzero(mask.to_numpy())
    expected_n = int(config["expected_data"]["split_trials_per_subject"][split])
    if len(positions) != expected_n:
        _fail(f"Split {split} for subject {subject} has {len(positions)} rows")
    class_counts = metadata.iloc[positions].groupby("class_label", observed=True).size()
    expected_class_n = int(
        config["expected_data"]["split_trials_per_subject_class"][split]
    )
    if len(class_counts) != len(CLASS_ORDER) or not (class_counts == expected_class_n).all():
        _fail(f"Split {split} class balance failed for subject {subject}")
    return positions


def validate_whole_covariances(
    covariances: np.ndarray,
    channel_names: np.ndarray,
    config: Mapping[str, Any],
    *,
    expected_content_sha256: str | None,
) -> dict[str, Any]:
    array = np.asarray(covariances)
    expected_shape = tuple(
        int(value)
        for value in config["expected_data"]["covariance_shape_per_session"]
    )
    if array.shape != expected_shape:
        raise CovarianceIdentityError(
            f"WHOLE shape mismatch: expected {expected_shape}, observed {array.shape}"
        )
    if str(array.dtype) != str(config["preprocessing"]["covariance_dtype"]):
        raise CovarianceIdentityError("WHOLE covariance dtype must be float64")
    configured_channels = np.asarray(config["dataset"]["eeg_channels"], dtype=str)
    if not np.array_equal(np.asarray(channel_names).astype(str), configured_channels):
        raise CovarianceIdentityError("WHOLE channel order mismatch")
    content_hash = sha256_array(array)
    if expected_content_sha256 is not None and content_hash != expected_content_sha256:
        raise CovarianceIdentityError(
            "WHOLE array-content SHA-256 mismatch: "
            f"expected {expected_content_sha256}, observed {content_hash}"
        )
    if not np.isfinite(array).all():
        raise CovarianceIdentityError("WHOLE covariance contains NaN or Inf")
    transpose = array.transpose(0, 2, 1)
    symmetry = np.linalg.norm(array - transpose, axis=(1, 2)) / np.maximum(
        np.linalg.norm(array, axis=(1, 2)), np.finfo(np.float64).tiny
    )
    max_symmetry = float(symmetry.max())
    if max_symmetry > float(config["hard_gates"]["symmetry_relative_error_max"]):
        raise CovarianceIdentityError("WHOLE symmetry gate failed")
    eigenvalues = np.linalg.eigvalsh(array)
    minimum = float(eigenvalues[:, 0].min())
    if minimum <= float(config["hard_gates"]["min_eigenvalue_strictly_greater_than"]):
        raise CovarianceIdentityError("WHOLE strict-SPD gate failed")
    conditions = eigenvalues[:, -1] / eigenvalues[:, 0]
    maximum_condition = float(conditions.max())
    if not np.isfinite(conditions).all() or maximum_condition > float(
        config["hard_gates"]["condition_number_max"]
    ):
        raise CovarianceIdentityError("WHOLE condition-number gate failed")
    return {
        "whole_array_content_sha256": content_hash,
        "minimum_eigenvalue": minimum,
        "maximum_condition_number": maximum_condition,
        "maximum_symmetry_relative_error": max_symmetry,
        "nonfinite_count": 0,
    }


def _validate_v1_frozen_config(path: Path, config: Mapping[str, Any]) -> None:
    try:
        v1 = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        _fail(f"Could not read V1 frozen config: {path}")
    comparisons = {
        "dataset.name": (v1["dataset"]["name"], config["dataset"]["name"]),
        "dataset.subjects": (v1["dataset"]["subjects"], config["dataset"]["subjects"]),
        "dataset.primary_session_key": (
            v1["dataset"]["primary_session_key"],
            config["dataset"]["discovery_session"],
        ),
        "dataset.classes": (v1["dataset"]["classes"], config["dataset"]["classes"]),
        "dataset.eeg_channels": (
            v1["dataset"]["eeg_channels"],
            config["dataset"]["eeg_channels"],
        ),
        "preprocessing.bandpass_hz": (
            v1["preprocessing"]["bandpass_hz"],
            config["preprocessing"]["bandpass_hz"],
        ),
        "preprocessing.epoch_tmin_seconds": (
            v1["preprocessing"]["epoch_tmin_seconds"],
            config["preprocessing"]["epoch_tmin_seconds"],
        ),
        "preprocessing.epoch_tmax_seconds": (
            v1["preprocessing"]["epoch_tmax_seconds"],
            config["preprocessing"]["epoch_tmax_seconds"],
        ),
        "preprocessing.resample_hz": (
            v1["preprocessing"]["resample_hz"],
            config["preprocessing"]["resample_hz"],
        ),
        "preprocessing.baseline": (
            v1["preprocessing"]["baseline"],
            config["preprocessing"]["baseline"],
        ),
        "representation.covariance_estimator": (
            v1["representation"]["covariance_estimator"],
            config["preprocessing"]["covariance_estimator"],
        ),
        "representation.covariance_extra_regularization": (
            v1["representation"]["covariance_extra_regularization"],
            config["preprocessing"]["extra_regularization"],
        ),
    }
    for label, (observed, expected) in comparisons.items():
        if observed != expected:
            _fail(f"V1/conditional frozen field mismatch at {label}")


def _discovery_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    return {
        key: _resolve_repo_path(root, value, label=f"discovery input {key}")
        for key, value in config["v1_discovery_inputs"]["paths"].items()
    }


def _validate_discovery_ancillary(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, str]:
    hashes = config["v1_discovery_inputs"]["hashes"]
    checked = {
        "frozen_config_sha256": _check_file_hash(
            paths["frozen_config"], hashes["frozen_config_sha256"], "V1 frozen config"
        ),
        "dataset_metadata_sha256": _check_file_hash(
            paths["dataset_metadata"], hashes["dataset_metadata_sha256"], "V1 dataset metadata"
        ),
        "covariance_summary_sha256": _check_file_hash(
            paths["covariance_summary"],
            hashes["covariance_summary_sha256"],
            "V1 covariance summary",
        ),
        "covariance_sanity_sha256": _check_file_hash(
            paths["covariance_sanity"], hashes["covariance_sanity_sha256"], "V1 covariance sanity"
        ),
        "whole_metadata_sha256": _check_file_hash(
            paths["whole_metadata"], hashes["whole_metadata_sha256"], "V1 WHOLE metadata"
        ),
    }
    _validate_v1_frozen_config(paths["frozen_config"], config)
    return checked


def _load_exact_discovery_covariance(
    config: Mapping[str, Any], covariance_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hashes = config["v1_discovery_inputs"]["hashes"]
    try:
        file_hash = sha256_file(covariance_path)
    except Exception as error:
        raise CovarianceIdentityError(str(error)) from error
    if file_hash != hashes["covariances_file_sha256"]:
        raise CovarianceIdentityError("V1 covariance file SHA-256 mismatch")
    try:
        with np.load(covariance_path, allow_pickle=False) as archive:
            if archive.files != ["whole", "window5", "channel_names"]:
                raise CovarianceIdentityError(
                    f"Unexpected V1 covariance keys: {archive.files}"
                )
            whole = np.array(archive["whole"], copy=True)
            channels = np.array(archive["channel_names"], copy=True)
    except CovarianceIdentityError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise CovarianceIdentityError("Could not read V1 covariance archive") from error
    checks = validate_whole_covariances(
        whole,
        channels,
        config,
        expected_content_sha256=hashes["whole_array_content_sha256"],
    )
    checks["covariances_file_sha256"] = file_hash
    return whole, channels, checks


def _require_fallback_versions() -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution, expected in FALLBACK_REQUIRED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            _fail(f"Discovery fallback requires {distribution}=={expected}")
        if actual != expected:
            _fail(
                f"Discovery fallback version mismatch for {distribution}: "
                f"expected {expected}, observed {actual}"
            )
        observed[distribution] = actual
    return observed


def _require_versions(required: Mapping[str, str], *, purpose: str) -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution, expected in required.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise DataContractError(
                f"{purpose} requires missing distribution {distribution}=={expected}"
            ) from error
        if actual != expected:
            _fail(
                f"{purpose} version mismatch for {distribution}: "
                f"expected {expected}, observed {actual}"
            )
        observed[distribution] = actual
    return observed


def _validate_prepared_metadata(
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    labels: np.ndarray,
) -> pd.DataFrame:
    return validate_session_metadata(
        metadata, config, DISCOVERY_SESSION, prepared=True, labels=labels
    )


def _build_whole_oas(eeg: np.ndarray, *, batch_size: int = FALLBACK_BATCH_SIZE) -> np.ndarray:
    """Estimate only WHOLE OAS covariances in deterministic trial batches."""

    array = np.asarray(eeg)
    if array.shape != (TRIALS_PER_SESSION, CHANNEL_COUNT, 1000):
        _fail(f"Prepared epoch shape mismatch: {array.shape}")
    if not np.isfinite(array).all():
        _fail("Prepared epochs contain NaN or Inf")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    from pyriemann.estimation import Covariances

    transformer = Covariances(estimator="oas")
    result = np.empty((TRIALS_PER_SESSION, CHANNEL_COUNT, CHANNEL_COUNT), dtype=np.float64)
    for start in range(0, len(array), int(batch_size)):
        stop = min(start + int(batch_size), len(array))
        estimated = transformer.transform(np.asarray(array[start:stop], dtype=np.float64))
        estimated = 0.5 * (estimated + estimated.transpose(0, 2, 1))
        result[start:stop] = estimated
    return result


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, value: str) -> None:
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


def _save_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                if archive.files != list(arrays):
                    _fail(f"Existing cache keys conflict: {path}")
                for key, expected in arrays.items():
                    if not np.array_equal(archive[key], expected):
                        _fail(f"Existing cache array conflicts at {key}: {path}")
        except (OSError, ValueError, KeyError) as error:
            raise DataContractError(f"Invalid existing cache: {path}") from error
        return
    _atomic_save_npz(path, **arrays)


def _save_csv_once(path: Path, frame: pd.DataFrame) -> None:
    content = frame.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            _fail(f"Existing metadata cache conflicts: {path}")
        return
    _atomic_write_text(path, content)


def _save_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataContractError(f"Invalid existing provenance cache: {path}") from error
        if existing != dict(payload):
            _fail(f"Existing provenance cache conflicts: {path}")
        return
    atomic_write_json(path, payload)


def _fallback_discovery_from_prepared(
    config: Mapping[str, Any],
    config_hash: str,
    protocol_hash: str,
    root: Path,
    paths: Mapping[str, Path],
    exact_metadata: pd.DataFrame,
    covariance_failure: str,
    ancillary_hashes: Mapping[str, str],
) -> ConditionalWholeData:
    hashes = config["v1_discovery_inputs"]["hashes"]
    prepared_file_hash = _check_file_hash(
        paths["prepared_epochs"], hashes["prepared_epochs_sha256"], "V1 prepared epochs"
    )
    prepared_metadata_hash = _check_file_hash(
        paths["prepared_metadata"], hashes["prepared_metadata_sha256"], "V1 prepared metadata"
    )
    prepared_metadata = _read_metadata(paths["prepared_metadata"])
    try:
        with np.load(paths["prepared_epochs"], allow_pickle=False) as archive:
            if archive.files != ["X", "y", "channel_names", "sampling_frequency_hz"]:
                _fail(f"Unexpected prepared epoch keys: {archive.files}")
            eeg = np.array(archive["X"], copy=True)
            labels = np.array(archive["y"], copy=True).astype(str)
            channels = np.array(archive["channel_names"], copy=True).astype(str)
            sampling_frequency = float(np.asarray(archive["sampling_frequency_hz"]))
    except (OSError, ValueError, KeyError) as error:
        raise DataContractError("Could not read exact V1 prepared epochs") from error
    if eeg.shape != (TRIALS_PER_SESSION, CHANNEL_COUNT, 1000):
        _fail("V1 prepared epoch shape mismatch")
    if str(eeg.dtype) != str(config["preprocessing"]["epoch_cache_dtype"]):
        _fail("V1 prepared epoch dtype mismatch")
    if not np.isfinite(eeg).all():
        _fail("V1 prepared epochs contain NaN or Inf")
    if not np.array_equal(channels, np.asarray(config["dataset"]["eeg_channels"], dtype=str)):
        _fail("V1 prepared channel order mismatch")
    if sampling_frequency != float(config["preprocessing"]["sampling_frequency_hz"]):
        _fail("V1 prepared sampling frequency mismatch")
    prepared_metadata = _validate_prepared_metadata(prepared_metadata, config, labels)
    rebuilt_metadata = prepared_metadata.copy()
    rebuilt_metadata.insert(0, "covariance_index", np.arange(len(rebuilt_metadata), dtype=np.int64))
    rebuilt_metadata = validate_session_metadata(
        rebuilt_metadata, config, DISCOVERY_SESSION
    )
    if not rebuilt_metadata.equals(exact_metadata):
        _fail("Prepared and exact V1 WHOLE metadata differ")

    versions = _require_fallback_versions()
    whole = _build_whole_oas(eeg)
    checks = validate_whole_covariances(
        whole, channels, config, expected_content_sha256=None
    )
    cache_root = _resolve_repo_path(root, config["project"]["cache_dir"], label="conditional cache")
    fallback = config["v1_discovery_inputs"]["fallback"]
    covariance_cache = cache_root / str(fallback["cache_filename"])
    metadata_cache = cache_root / str(fallback["metadata_filename"])
    provenance_cache = cache_root / "discovery_data_provenance.json"
    _save_npz_once(covariance_cache, {"whole": whole, "channel_names": channels})
    _save_csv_once(metadata_cache, rebuilt_metadata)
    provenance = {
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "session": DISCOVERY_SESSION,
        "source": "recomputed_from_exact_v1_prepared_inputs",
        "raw_or_moabb_accessed": False,
        "window5_computed": False,
        "fallback_reason": covariance_failure,
        "prepared_epochs_sha256": prepared_file_hash,
        "prepared_metadata_sha256": prepared_metadata_hash,
        "ancillary_hashes": dict(ancillary_hashes),
        "required_versions": versions,
        "whole_shape": list(whole.shape),
        "whole_dtype": str(whole.dtype),
        "whole_array_content_sha256": checks["whole_array_content_sha256"],
        "cache_covariances_sha256": sha256_file(covariance_cache),
        "cache_metadata_sha256": sha256_file(metadata_cache),
        "numerical_checks": checks,
    }
    _save_json_once(provenance_cache, provenance)
    whole.setflags(write=False)
    channels.setflags(write=False)
    return ConditionalWholeData(
        session=DISCOVERY_SESSION,
        covariances=whole,
        metadata=rebuilt_metadata,
        channel_names=channels,
        provenance=provenance,
    )


def load_discovery_whole(
    config_path: str | Path, repo_root: str | Path
) -> ConditionalWholeData:
    """Load only the exact V1 ``0train`` WHOLE array, with fail-closed fallback."""

    root = Path(repo_root).resolve()
    config, _config_file, config_hash, protocol_hash = load_conditional_config(
        config_path, root
    )
    paths = _discovery_paths(config, root)
    ancillary_hashes = _validate_discovery_ancillary(config, paths)
    metadata = validate_session_metadata(
        _read_metadata(paths["whole_metadata"]), config, DISCOVERY_SESSION
    )
    try:
        whole, channels, checks = _load_exact_discovery_covariance(
            config, paths["covariances"]
        )
    except CovarianceIdentityError as error:
        return _fallback_discovery_from_prepared(
            config,
            config_hash,
            protocol_hash,
            root,
            paths,
            metadata,
            str(error),
            ancillary_hashes,
        )

    provenance = {
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "session": DISCOVERY_SESSION,
        "source": "exact_v1_read_only_reuse",
        "read_only_v1": True,
        "raw_or_moabb_accessed": False,
        "window5_accessed": False,
        "fallback_reason": None,
        "ancillary_hashes": ancillary_hashes,
        "whole_shape": list(whole.shape),
        "whole_dtype": str(whole.dtype),
        "whole_metadata_sha256": config["v1_discovery_inputs"]["hashes"][
            "whole_metadata_sha256"
        ],
        "numerical_checks": checks,
    }
    whole.setflags(write=False)
    channels.setflags(write=False)
    return ConditionalWholeData(
        session=DISCOVERY_SESSION,
        covariances=whole,
        metadata=metadata,
        channel_names=channels,
        provenance=provenance,
    )


def _resolve_confirmatory_raw_inputs(
    config: Mapping[str, Any], root: Path
) -> tuple[list[Path], dict[str, Any]]:
    """Resolve and hash E files; callers must validate the unlock first."""

    paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for frozen in config["confirmatory_inputs"]["raw_files"]:
        subject = int(frozen["subject"])
        path = _resolve_repo_path(
            root, frozen["path"], label=f"confirmatory raw subject {subject}"
        )
        try:
            size = int(path.stat().st_size)
        except FileNotFoundError as error:
            _fail(f"Confirmatory raw file is missing for subject {subject}: {path}")
        if size != int(frozen["bytes"]):
            _fail(f"Confirmatory raw byte count mismatch for subject {subject}")
        digest = _check_file_hash(
            path, str(frozen["sha256"]), f"confirmatory raw subject {subject}"
        )
        paths.append(path)
        records.append(
            {"filename": path.name, "size_bytes": size, "sha256": digest}
        )
    aggregate = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    expected = str(config["confirmatory_inputs"]["ordered_manifest_sha256"])
    if aggregate != expected:
        _fail(
            f"Confirmatory ordered raw manifest mismatch: expected {expected}, observed {aggregate}"
        )
    total = sum(int(record["size_bytes"]) for record in records)
    if total != int(config["confirmatory_inputs"]["total_bytes"]):
        _fail("Confirmatory total raw bytes mismatch")
    return paths, {
        "algorithm": "sha256_canonical_json_ordered_filename_size_hash_records_v1",
        "files": records,
        "file_count": len(records),
        "total_bytes": total,
        "ordered_manifest_sha256": aggregate,
    }


def _augment_moabb_metadata(
    raw_metadata: pd.DataFrame,
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    metadata = raw_metadata.reset_index(drop=True).copy()
    missing = {"subject", "session", "run"} - set(metadata.columns)
    if missing:
        _fail(f"MOABB metadata is missing columns: {sorted(missing)}")
    metadata = metadata[["subject", "session", "run"]].copy()
    metadata["subject"] = pd.to_numeric(metadata["subject"], errors="raise").astype(int)
    metadata["session"] = metadata["session"].astype(str)
    metadata["run"] = metadata["run"].astype(str)
    labels = np.asarray(labels).astype(str)
    if len(metadata) != len(labels):
        _fail("MOABB labels and metadata have different lengths")
    metadata.insert(0, "sample_index", np.arange(len(metadata), dtype=np.int64))
    metadata["trial_id"] = (
        metadata.groupby(["subject", "session"], sort=False).cumcount() + 1
    ).astype(np.int64)
    metadata["run_trial_id"] = (
        metadata.groupby(["subject", "session", "run"], sort=False).cumcount() + 1
    ).astype(np.int64)
    metadata["trial_uid"] = [
        f"S{int(subject):02d}_{session}_T{int(trial_id):03d}"
        for subject, session, trial_id in metadata[
            ["subject", "session", "trial_id"]
        ].itertuples(index=False, name=None)
    ]
    metadata["class_label"] = labels
    metadata.insert(0, "covariance_index", np.arange(len(metadata), dtype=np.int64))
    metadata = metadata[list(config["expected_data"]["metadata_columns"])]
    return validate_session_metadata(metadata, config, CONFIRMATORY_SESSION, labels=labels)


def _load_confirmatory_epochs_from_moabb(
    config: Mapping[str, Any], root: Path, raw_paths: list[Path]
) -> _PreparedEpochs:
    """Load the frozen 1test epoch array after raw hashes and unlock pass."""

    moabb_dir = _resolve_repo_path(
        root, config["confirmatory_inputs"]["raw_moabb_dir"], label="MOABB data directory"
    )
    os.environ["MNE_DATA"] = str(moabb_dir)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(moabb_dir)
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.datasets.bnci.base import _convert_mi
    from moabb.paradigms import MotorImagery

    mne.set_log_level("WARNING")
    subjects = [int(value) for value in config["dataset"]["subjects"]]
    classes = [str(value) for value in config["dataset"]["classes"]]
    channels = [str(value) for value in config["dataset"]["eeg_channels"]]
    fmin, fmax = (float(value) for value in config["preprocessing"]["bandpass_hz"])
    raw_by_subject = dict(zip(subjects, raw_paths, strict=True))

    class _ConfirmatoryOnlyBNCI(BNCI2014_001):
        """BNCI loader that opens only the nine hash-validated E files.

        MOABB's stock BNCI loader materializes both T and E before applying its
        selected-session filter.  Calling the same pinned ``_convert_mi``
        converter directly prevents unpinned T-file access and makes the
        confirmatory input boundary literal.
        """

        def _get_single_subject_data(self, subject: int) -> dict[str, Any]:
            runs, _event_id = _convert_mi(
                str(raw_by_subject[int(subject)]),
                channels + ["EOG1", "EOG2", "EOG3"],
                ["eeg"] * 22 + ["eog"] * 3,
                dataset_code="BNCI2014-001",
                subject_id=int(subject),
            )
            return {
                CONFIRMATORY_SESSION: {
                    str(run_index): run for run_index, run in enumerate(runs)
                }
            }

    dataset = _ConfirmatoryOnlyBNCI(
        subjects=subjects, sessions=[CONFIRMATORY_SESSION]
    )
    paradigm = MotorImagery(
        n_classes=len(classes),
        events=classes,
        fmin=fmin,
        fmax=fmax,
        tmin=float(config["preprocessing"]["epoch_tmin_seconds"]),
        tmax=float(config["preprocessing"]["epoch_tmax_seconds"]),
        baseline=config["preprocessing"]["baseline"],
        channels=channels,
        resample=config["preprocessing"]["resample_hz"],
    )
    epochs, labels, raw_metadata = paradigm.get_data(
        dataset=dataset, subjects=subjects, return_epochs=True
    )
    if set(epochs.ch_names) != set(channels):
        _fail("Confirmatory epoch channel vocabulary mismatch")
    if epochs.ch_names != channels:
        epochs.reorder_channels(channels)
    if any(value != "eeg" for value in epochs.get_channel_types()):
        _fail("A non-EEG channel survived confirmatory selection")
    # Match the frozen V1 epoch-cache quantization before float64 OAS fitting.
    eeg = epochs.get_data(copy=True).astype(np.float32, copy=False)
    metadata = _augment_moabb_metadata(raw_metadata, labels, config)
    return _PreparedEpochs(
        eeg=eeg,
        labels=np.asarray(labels).astype(str),
        metadata=metadata,
        channel_names=np.asarray(channels, dtype=str),
        sampling_frequency_hz=float(epochs.info["sfreq"]),
        observed_source_times=(float(epochs.times[0]), float(epochs.times[-1])),
    )


def _validate_confirmatory_epochs(
    prepared: _PreparedEpochs, config: Mapping[str, Any]
) -> None:
    if prepared.eeg.shape != (TRIALS_PER_SESSION, CHANNEL_COUNT, 1000):
        _fail(f"Confirmatory epoch shape mismatch: {prepared.eeg.shape}")
    if str(prepared.eeg.dtype) != str(config["preprocessing"]["epoch_cache_dtype"]):
        _fail("Confirmatory epoch dtype must match frozen float32 quantization")
    if not np.isfinite(prepared.eeg).all():
        _fail("Confirmatory epochs contain NaN or Inf")
    if prepared.sampling_frequency_hz != float(
        config["preprocessing"]["sampling_frequency_hz"]
    ):
        _fail("Confirmatory sampling frequency mismatch")
    expected_times = tuple(
        float(value) for value in config["preprocessing"]["observed_source_epoch_seconds"]
    )
    tolerance = 0.5 / prepared.sampling_frequency_hz
    if not np.allclose(prepared.observed_source_times, expected_times, atol=tolerance, rtol=0.0):
        _fail("Confirmatory observed source-time interval mismatch")
    if not np.array_equal(
        prepared.channel_names.astype(str),
        np.asarray(config["dataset"]["eeg_channels"], dtype=str),
    ):
        _fail("Confirmatory prepared channel order mismatch")
    validate_session_metadata(
        prepared.metadata, config, CONFIRMATORY_SESSION, labels=prepared.labels
    )


def _confirmatory_cache_paths(
    config: Mapping[str, Any], root: Path
) -> tuple[Path, Path, Path]:
    covariance = _resolve_repo_path(
        root,
        config["confirmatory_inputs"]["cache_covariances"],
        label="confirmatory covariance cache",
    )
    metadata = _resolve_repo_path(
        root,
        config["confirmatory_inputs"]["cache_metadata"],
        label="confirmatory metadata cache",
    )
    provenance = covariance.parent / "confirmatory_data_provenance.json"
    return covariance, metadata, provenance


def _prepare_confirmatory_after_unlock(
    config: Mapping[str, Any],
    config_hash: str,
    protocol_hash: str,
    root: Path,
    unlock: Mapping[str, Any],
    raw_paths: list[Path],
    raw_manifest: Mapping[str, Any],
) -> ConditionalWholeData:
    versions = _require_versions(
        CONFIRMATORY_REQUIRED_VERSIONS, purpose="Confirmatory preparation"
    )
    prepared = _load_confirmatory_epochs_from_moabb(config, root, raw_paths)
    _validate_confirmatory_epochs(prepared, config)
    whole = _build_whole_oas(prepared.eeg)
    checks = validate_whole_covariances(
        whole, prepared.channel_names, config, expected_content_sha256=None
    )
    metadata = validate_session_metadata(
        prepared.metadata, config, CONFIRMATORY_SESSION, labels=prepared.labels
    )
    covariance_cache, metadata_cache, provenance_cache = _confirmatory_cache_paths(
        config, root
    )
    _save_npz_once(
        covariance_cache,
        {"whole": whole, "channel_names": prepared.channel_names.astype(str)},
    )
    _save_csv_once(metadata_cache, metadata)
    provenance = {
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "unlock_manifest_sha256": str(unlock["manifest_sha256"]),
        "locked_head": str(unlock["locked_head"]),
        "session": CONFIRMATORY_SESSION,
        "source": "post_unlock_raw_moabb_whole_oas",
        "window5_computed": False,
        "trial_rejection": False,
        "required_versions": versions,
        "raw_manifest": dict(raw_manifest),
        "whole_shape": list(whole.shape),
        "whole_dtype": str(whole.dtype),
        "whole_array_content_sha256": checks["whole_array_content_sha256"],
        "cache_covariances_sha256": sha256_file(covariance_cache),
        "cache_metadata_sha256": sha256_file(metadata_cache),
        "metadata_rows": len(metadata),
        "numerical_checks": checks,
    }
    _save_json_once(provenance_cache, provenance)
    whole.setflags(write=False)
    prepared.channel_names.setflags(write=False)
    return ConditionalWholeData(
        session=CONFIRMATORY_SESSION,
        covariances=whole,
        metadata=metadata,
        channel_names=prepared.channel_names,
        provenance=provenance,
    )


def prepare_confirmatory_whole(
    config_path: str | Path,
    repo_root: str | Path,
    *,
    unlock_path: str | Path | None = None,
) -> ConditionalWholeData:
    """After a valid lock only, prepare and cache 1test WHOLE covariances."""

    root = Path(repo_root).resolve()
    config, config_file, config_hash, protocol_hash = load_conditional_config(
        config_path, root
    )
    # Barrier: do not move raw/cache resolution above this call.
    unlock = validate_confirmatory_unlock(
        config_file, root, unlock_path=unlock_path
    )
    raw_paths, raw_manifest = _resolve_confirmatory_raw_inputs(config, root)
    return _prepare_confirmatory_after_unlock(
        config,
        config_hash,
        protocol_hash,
        root,
        unlock,
        raw_paths,
        raw_manifest,
    )


def _load_confirmatory_cache_after_unlock(
    config: Mapping[str, Any],
    config_hash: str,
    protocol_hash: str,
    root: Path,
    unlock: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
) -> ConditionalWholeData | None:
    covariance_cache, metadata_cache, provenance_cache = _confirmatory_cache_paths(
        config, root
    )
    existing = [path.exists() for path in (covariance_cache, metadata_cache, provenance_cache)]
    if not any(existing):
        return None
    if not all(existing):
        _fail("Confirmatory cache is incomplete")
    try:
        with np.load(covariance_cache, allow_pickle=False) as archive:
            if archive.files != ["whole", "channel_names"]:
                _fail(f"Unexpected confirmatory covariance keys: {archive.files}")
            whole = np.array(archive["whole"], copy=True)
            channels = np.array(archive["channel_names"], copy=True).astype(str)
        metadata = validate_session_metadata(
            _read_metadata(metadata_cache), config, CONFIRMATORY_SESSION
        )
        provenance = json.loads(provenance_cache.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise DataContractError("Could not read confirmatory conditional cache") from error
    checks = validate_whole_covariances(
        whole,
        channels,
        config,
        expected_content_sha256=str(provenance.get("whole_array_content_sha256", "")),
    )
    expected = {
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "unlock_manifest_sha256": unlock["manifest_sha256"],
        "locked_head": unlock["locked_head"],
        "session": CONFIRMATORY_SESSION,
        "source": "post_unlock_raw_moabb_whole_oas",
        "window5_computed": False,
        "required_versions": _require_versions(
            CONFIRMATORY_REQUIRED_VERSIONS, purpose="Confirmatory cache load"
        ),
        "raw_manifest": dict(raw_manifest),
        "cache_covariances_sha256": sha256_file(covariance_cache),
        "cache_metadata_sha256": sha256_file(metadata_cache),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            _fail(f"Confirmatory cache provenance mismatch at {key}")
    provenance["numerical_checks"] = checks
    whole.setflags(write=False)
    channels.setflags(write=False)
    return ConditionalWholeData(
        session=CONFIRMATORY_SESSION,
        covariances=whole,
        metadata=metadata,
        channel_names=channels,
        provenance=provenance,
    )


def load_confirmatory_whole(
    config_path: str | Path,
    repo_root: str | Path,
    *,
    unlock_path: str | Path | None = None,
) -> ConditionalWholeData:
    """Validate the unlock first, then load or prepare the 1test WHOLE cache."""

    root = Path(repo_root).resolve()
    config, config_file, config_hash, protocol_hash = load_conditional_config(
        config_path, root
    )
    # Barrier: no confirmatory input or cache path has been resolved yet.
    unlock = validate_confirmatory_unlock(
        config_file, root, unlock_path=unlock_path
    )
    raw_paths, raw_manifest = _resolve_confirmatory_raw_inputs(config, root)
    cached = _load_confirmatory_cache_after_unlock(
        config, config_hash, protocol_hash, root, unlock, raw_manifest
    )
    if cached is not None:
        return cached
    return _prepare_confirmatory_after_unlock(
        config,
        config_hash,
        protocol_hash,
        root,
        unlock,
        raw_paths,
        raw_manifest,
    )


__all__ = [
    "CHANNEL_COUNT",
    "CLASS_ORDER",
    "CONFIRMATORY_SESSION",
    "ConditionalDataError",
    "ConditionalWholeData",
    "CovarianceIdentityError",
    "DISCOVERY_SESSION",
    "DataContractError",
    "RUN_ORDER",
    "load_conditional_config",
    "load_confirmatory_whole",
    "load_discovery_whole",
    "prepare_confirmatory_whole",
    "sha256_array",
    "subject_split_positions",
    "validate_conditional_config",
    "validate_session_metadata",
    "validate_whole_covariances",
]
