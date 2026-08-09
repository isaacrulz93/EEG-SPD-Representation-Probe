"""Strict read-only V1 WHOLE-covariance contract for Geometry Audit V2.

The preferred path verifies and reuses the exact V1 WHOLE array without
writing anywhere.  Only a covariance-identity failure may activate the
explicit fallback: exact prepared V1 inputs are read, WHOLE OAS covariances
are recomputed, and artifacts are written solely under the V2 cache.
"""

from __future__ import annotations

import copy
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


class DataContractError(RuntimeError):
    """Raised when a frozen V2 data or provenance assertion fails."""


class CovarianceIdentityError(DataContractError):
    """A V1 covariance-specific failure eligible for the frozen fallback."""


@dataclass(frozen=True)
class V2WholeData:
    """Validated WHOLE covariances and their trial-aligned metadata."""

    covariances: np.ndarray
    metadata: pd.DataFrame
    channel_names: np.ndarray
    provenance: dict[str, Any]


def _fail(message: str) -> None:
    raise DataContractError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as error:
        raise DataContractError(f"Required V1 input is missing: {path}") from error
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _check_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected:
        _fail(f"{label} SHA-256 mismatch: expected {expected}, observed {actual}")
    return actual


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_config_schema(config: Mapping[str, Any]) -> None:
    required = {
        "protocol",
        "project",
        "v1_inputs",
        "dataset",
        "preprocessing",
        "representation",
        "expected_data",
        "geometry",
        "evaluation",
        "classifiers",
        "verdicts",
        "outputs",
    }
    missing = required - set(config)
    if missing:
        _fail(f"V2 config is missing sections: {sorted(missing)}")
    paths = config["v1_inputs"].get("paths", {})
    hashes = config["v1_inputs"].get("hashes", {})
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
    if required_paths - set(paths):
        _fail(f"V2 config is missing V1 paths: {sorted(required_paths - set(paths))}")
    if required_hashes - set(hashes):
        _fail(
            f"V2 config is missing V1 hashes: {sorted(required_hashes - set(hashes))}"
        )
    for name in required_hashes:
        value = str(hashes[name])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            _fail(f"Invalid lowercase SHA-256 value for {name}: {value!r}")

    subjects = [int(value) for value in config["dataset"]["subjects"]]
    channels = [str(value) for value in config["dataset"]["eeg_channels"]]
    classes = [str(value) for value in config["dataset"]["classes"]]
    expected = config["expected_data"]
    eeg_shape = tuple(int(value) for value in expected["eeg_shape"])
    covariance_shape = tuple(
        int(value) for value in expected["whole_covariance_shape"]
    )
    if len(set(subjects)) != len(subjects) or len(set(channels)) != len(channels):
        _fail("Subjects and EEG channel names must be unique")
    if len(set(classes)) != len(classes):
        _fail("Class labels must be unique")
    if eeg_shape != (
        int(expected["trials_total"]),
        len(channels),
        int(config["preprocessing"]["samples_per_trial"]),
    ):
        _fail("expected_data.eeg_shape is inconsistent with frozen counts")
    if covariance_shape != (eeg_shape[0], len(channels), len(channels)):
        _fail("expected WHOLE covariance shape is inconsistent with EEG shape")
    if len(subjects) * int(expected["trials_per_subject"]) != eeg_shape[0]:
        _fail("Subject trial counts do not sum to the total")
    if len(expected["runs"]) * int(expected["trials_per_run"]) != int(
        expected["trials_per_subject"]
    ):
        _fail("Run trial counts do not sum to a subject")
    if len(classes) * int(expected["trials_per_subject_class"]) != int(
        expected["trials_per_subject"]
    ):
        _fail("Class trial counts do not sum to a subject")
    if int(expected["trials_per_subject_run_class"]) * len(classes) != int(
        expected["trials_per_run"]
    ):
        _fail("Per-run class counts do not sum to a run")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the V2 YAML without mutating it."""

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise DataContractError(f"V2 config is missing: {config_path}") from error
    if not isinstance(config, dict):
        _fail(f"Expected a YAML mapping in {config_path}")
    _validate_config_schema(config)
    return config


def _config_and_identity(
    config_or_path: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], str, str]:
    if isinstance(config_or_path, (str, Path)):
        path = Path(config_or_path).expanduser().resolve()
        config = load_config(path)
        return config, _sha256_file(path), "yaml_file_sha256"
    config = copy.deepcopy(dict(config_or_path))
    _validate_config_schema(config)
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return config, hashlib.sha256(canonical).hexdigest(), "canonical_json_sha256"


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: expected {expected!r}, observed {actual!r}")


def _validate_v1_frozen_config(
    path: Path, expected_hash: str, config: Mapping[str, Any]
) -> str:
    actual_hash = _check_hash(path, expected_hash, "V1 frozen config")
    with path.open("r", encoding="utf-8") as handle:
        v1 = yaml.safe_load(handle)
    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    representation = config["representation"]
    comparisons = {
        "dataset.name": (v1["dataset"]["name"], dataset["name"]),
        "dataset.subjects": (v1["dataset"]["subjects"], dataset["subjects"]),
        "dataset.primary_session_key": (
            v1["dataset"]["primary_session_key"],
            dataset["primary_session_key"],
        ),
        "dataset.primary_session_number": (
            v1["dataset"]["primary_session_number"],
            dataset["primary_session_number"],
        ),
        "dataset.classes": (v1["dataset"]["classes"], dataset["classes"]),
        "dataset.eeg_channels": (
            v1["dataset"]["eeg_channels"],
            dataset["eeg_channels"],
        ),
        "preprocessing.epoch_tmin_seconds": (
            v1["preprocessing"]["epoch_tmin_seconds"],
            preprocessing["epoch_tmin_seconds"],
        ),
        "preprocessing.epoch_tmax_seconds": (
            v1["preprocessing"]["epoch_tmax_seconds"],
            preprocessing["epoch_tmax_seconds"],
        ),
        "preprocessing.source_event_interval_seconds": (
            v1["preprocessing"]["source_event_interval_seconds"],
            preprocessing["source_event_interval_seconds"],
        ),
        "preprocessing.bandpass_hz": (
            v1["preprocessing"]["bandpass_hz"],
            preprocessing["bandpass_hz"],
        ),
        "preprocessing.resample_hz": (
            v1["preprocessing"]["resample_hz"],
            preprocessing["resample_hz"],
        ),
        "preprocessing.baseline": (
            v1["preprocessing"]["baseline"],
            preprocessing["baseline"],
        ),
        "preprocessing.sampling_frequency_hz": (
            v1["preprocessing"]["expected_sampling_frequency_hz"],
            preprocessing["sampling_frequency_hz"],
        ),
        "preprocessing.samples_per_trial": (
            v1["preprocessing"]["expected_samples_per_trial"],
            preprocessing["samples_per_trial"],
        ),
        "representation.covariance_estimator": (
            v1["representation"]["covariance_estimator"],
            representation["covariance_estimator"],
        ),
        "representation.symmetrize_covariance": (
            v1["representation"]["symmetrize_covariance"],
            representation["symmetrize_covariance"],
        ),
        "representation.covariance_extra_regularization": (
            v1["representation"]["covariance_extra_regularization"],
            representation["covariance_extra_regularization"],
        ),
    }
    for label, (actual, expected) in comparisons.items():
        _assert_equal(actual, expected, f"V1/V2 {label}")
    return actual_hash


def _load_json_with_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    _check_hash(path, expected_hash, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"Could not read {label}: {path}") from error
    if not isinstance(payload, dict):
        _fail(f"{label} must contain a JSON object")
    return payload


def _validate_dataset_metadata(
    metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    dataset = config["dataset"]
    prep = config["preprocessing"]
    expected = config["expected_data"]
    checks = {
        "dataset": dataset["name"],
        "subjects": dataset["subjects"],
        "n_subjects": len(dataset["subjects"]),
        "sessions_observed": [dataset["primary_session_key"]],
        "n_sessions_observed": 1,
        "sessions_available": dataset["available_sessions"],
        "n_eeg_channels": dataset["output_eeg_channels"],
        "n_eog_channels_output": dataset["output_eog_channels"],
        "eeg_channels": dataset["eeg_channels"],
        "sampling_frequency_hz": prep["sampling_frequency_hz"],
        "source_event_interval_seconds": prep["source_event_interval_seconds"],
        "epoch_time_interval_relative_to_mi_cue_seconds": [
            prep["epoch_tmin_seconds"],
            prep["epoch_tmax_seconds"],
        ],
        "epoch_time_interval_observed_source_seconds": prep[
            "observed_source_epoch_seconds"
        ],
        "samples_per_trial": prep["samples_per_trial"],
        "array_shape": expected["eeg_shape"],
        "n_trials": expected["trials_total"],
        "bandpass_hz": prep["bandpass_hz"],
        "resample_hz": prep["resample_hz"],
        "baseline": prep["baseline"],
        "cached_epoch_dtype": config["representation"]["input_eeg_dtype"],
        "signal_units": dataset["signal_units"],
    }
    for key, expected_value in checks.items():
        _assert_equal(metadata.get(key), expected_value, f"dataset_metadata.{key}")
    if set(metadata.get("classes_observed", [])) != set(dataset["classes"]):
        _fail("dataset_metadata classes do not match the frozen class vocabulary")
    if any(kind != "eeg" for kind in metadata.get("channel_types", [])):
        _fail("dataset_metadata contains a non-EEG output channel")


def _validate_covariance_summary(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    representation = config["representation"]
    expected = config["expected_data"]
    _assert_equal(summary.get("estimator"), representation["covariance_estimator"], "covariance estimator")
    _assert_equal(summary.get("oas_assume_centered"), representation["oas_assume_centered"], "OAS centering")
    _assert_equal(summary.get("extra_regularization"), representation["covariance_extra_regularization"], "covariance regularization")
    _assert_equal(summary.get("symmetrized"), representation["symmetrize_covariance"], "covariance symmetrization")
    _assert_equal(summary.get("whole_shape"), expected["whole_covariance_shape"], "WHOLE summary shape")
    sanity = summary.get("sanity", {}).get("WHOLE", {})
    _assert_equal(sanity.get("count"), expected["trials_total"], "WHOLE sanity count")
    _assert_equal(sanity.get("spd_count"), expected["trials_total"], "WHOLE SPD count")
    for key in ("non_spd_count", "nan_count", "inf_count"):
        _assert_equal(sanity.get(key), 0, f"WHOLE {key}")


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


def _validate_trial_metadata(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    prepared: bool,
    labels: np.ndarray | None = None,
) -> None:
    expected = config["expected_data"]
    dataset = config["dataset"]
    column_key = "prepared_metadata_columns" if prepared else "metadata_columns"
    _assert_equal(list(frame.columns), expected[column_key], f"{column_key} schema")
    _assert_equal(len(frame), expected["trials_total"], "metadata row count")
    if frame.isna().any().any():
        _fail("Metadata contains missing values")
    if frame["trial_uid"].duplicated().any():
        _fail("trial_uid must be globally unique")
    if frame.duplicated(["subject", "session", "trial_id"]).any():
        _fail("(subject, session, trial_id) must be unique")
    _assert_equal(
        sorted(frame["subject"].astype(int).unique().tolist()),
        sorted(int(value) for value in dataset["subjects"]),
        "metadata subjects",
    )
    _assert_equal(
        frame["session"].astype(str).unique().tolist(),
        [str(dataset["primary_session_key"])],
        "metadata session",
    )
    _assert_equal(
        sorted(frame["run"].astype(str).unique().tolist(), key=int),
        [str(value) for value in expected["runs"]],
        "metadata runs",
    )
    if set(frame["class_label"].astype(str).unique()) != set(dataset["classes"]):
        _fail("Metadata class vocabulary mismatch")
    sample_index = frame["sample_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_index, np.arange(len(frame), dtype=np.int64)):
        _fail("sample_index must be contiguous and row-aligned")
    if not prepared:
        covariance_index = frame["covariance_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(covariance_index, sample_index):
            _fail("covariance_index must equal sample_index")

    per_subject = frame.groupby("subject", observed=True).size()
    if not (per_subject == int(expected["trials_per_subject"])).all():
        _fail("Subject trial counts do not match the frozen contract")
    per_run = frame.groupby(["subject", "run"], observed=True).size()
    if not (per_run == int(expected["trials_per_run"])).all():
        _fail("Subject/run trial counts do not match the frozen contract")
    per_class = frame.groupby(["subject", "class_label"], observed=True).size()
    if not (per_class == int(expected["trials_per_subject_class"])).all():
        _fail("Subject/class trial counts do not match the frozen contract")
    per_run_class = frame.groupby(
        ["subject", "run", "class_label"], observed=True
    ).size()
    if not (
        per_run_class == int(expected["trials_per_subject_run_class"])
    ).all():
        _fail("Subject/run/class counts do not match the frozen contract")

    trial_min, trial_max = (int(value) for value in expected["subject_trial_ids"])
    run_min, run_max = (int(value) for value in expected["run_trial_ids"])
    trial_ranges = frame.groupby(["subject", "session"], observed=True)[
        "trial_id"
    ].agg(["min", "max", "nunique"])
    if not (
        (trial_ranges["min"] == trial_min)
        & (trial_ranges["max"] == trial_max)
        & (trial_ranges["nunique"] == int(expected["trials_per_subject"]))
    ).all():
        _fail("Within-subject trial_id range is invalid")
    run_ranges = frame.groupby(["subject", "session", "run"], observed=True)[
        "run_trial_id"
    ].agg(["min", "max", "nunique"])
    if not (
        (run_ranges["min"] == run_min)
        & (run_ranges["max"] == run_max)
        & (run_ranges["nunique"] == int(expected["trials_per_run"]))
    ).all():
        _fail("Within-run trial identifiers are invalid")

    expected_uids = np.asarray(
        [
            f"S{int(subject):02d}_{session}_T{int(trial_id):03d}"
            for subject, session, trial_id in frame[
                ["subject", "session", "trial_id"]
            ].itertuples(index=False, name=None)
        ]
    )
    if not np.array_equal(frame["trial_uid"].astype(str).to_numpy(), expected_uids):
        _fail("trial_uid values do not match the canonical V1 format")
    if labels is not None and not np.array_equal(
        frame["class_label"].astype(str).to_numpy(), np.asarray(labels).astype(str)
    ):
        _fail("Prepared labels and metadata are not row-aligned")


def _validate_covariance_array(
    covariances: np.ndarray,
    channel_names: np.ndarray,
    config: Mapping[str, Any],
    *,
    expected_content_hash: str | None,
) -> dict[str, Any]:
    expected = config["expected_data"]
    representation = config["representation"]
    geometry = config["geometry"]["hard_gate"]
    if covariances.shape != tuple(expected["whole_covariance_shape"]):
        raise CovarianceIdentityError(
            f"WHOLE shape mismatch: expected {expected['whole_covariance_shape']}, "
            f"observed {list(covariances.shape)}"
        )
    if str(covariances.dtype) != str(representation["covariance_dtype"]):
        raise CovarianceIdentityError(
            f"WHOLE dtype mismatch: expected {representation['covariance_dtype']}, "
            f"observed {covariances.dtype}"
        )
    configured_channels = np.asarray(config["dataset"]["eeg_channels"], dtype=str)
    if not np.array_equal(np.asarray(channel_names).astype(str), configured_channels):
        raise CovarianceIdentityError("WHOLE channel order does not match V2 config")
    content_hash = _sha256_array(covariances)
    if expected_content_hash is not None and content_hash != expected_content_hash:
        raise CovarianceIdentityError(
            "WHOLE array-content SHA-256 mismatch: "
            f"expected {expected_content_hash}, observed {content_hash}"
        )
    if not np.isfinite(covariances).all():
        raise CovarianceIdentityError("WHOLE covariances contain NaN or Inf")
    transpose = covariances.transpose(0, 2, 1)
    numerator = np.linalg.norm(covariances - transpose, axis=(1, 2))
    denominator = np.maximum(
        np.linalg.norm(covariances, axis=(1, 2)), np.finfo(np.float64).tiny
    )
    symmetry_error = numerator / denominator
    if float(symmetry_error.max()) > float(geometry["symmetry_relative_error_max"]):
        raise CovarianceIdentityError("WHOLE covariance symmetry gate failed")
    eigenvalues = np.linalg.eigvalsh(covariances)
    minimum = float(eigenvalues[:, 0].min())
    if minimum <= float(geometry["minimum_eigenvalue_strictly_greater_than"]):
        raise CovarianceIdentityError("WHOLE positive-definiteness gate failed")
    condition_numbers = eigenvalues[:, -1] / eigenvalues[:, 0]
    maximum_condition = float(condition_numbers.max())
    if not np.isfinite(condition_numbers).all() or maximum_condition > float(
        geometry["condition_number_max"]
    ):
        raise CovarianceIdentityError("WHOLE conditioning gate failed")
    return {
        "whole_array_content_sha256": content_hash,
        "minimum_eigenvalue": minimum,
        "maximum_condition_number": maximum_condition,
        "maximum_symmetry_relative_error": float(symmetry_error.max()),
    }


def _load_reused_covariance(
    covariance_path: Path,
    expected_file_hash: str,
    expected_content_hash: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        actual_file_hash = _sha256_file(covariance_path)
    except DataContractError as error:
        raise CovarianceIdentityError(str(error)) from error
    if actual_file_hash != expected_file_hash:
        raise CovarianceIdentityError(
            "V1 covariance file SHA-256 mismatch: "
            f"expected {expected_file_hash}, observed {actual_file_hash}"
        )
    try:
        with np.load(covariance_path, allow_pickle=False) as archive:
            expected_keys = list(config["v1_inputs"]["npz_keys"]["covariances"])
            if archive.files != expected_keys:
                raise CovarianceIdentityError(
                    f"V1 covariance NPZ keys mismatch: {archive.files}"
                )
            covariances = np.array(archive[config["expected_data"]["covariance_npz_key"]], copy=True)
            channel_names = np.array(archive["channel_names"], copy=True)
    except (OSError, ValueError, KeyError) as error:
        raise CovarianceIdentityError(
            f"Could not read V1 covariance archive: {covariance_path}"
        ) from error
    checks = _validate_covariance_array(
        covariances,
        channel_names,
        config,
        expected_content_hash=expected_content_hash,
    )
    checks["covariances_file_sha256"] = actual_file_hash
    return covariances, channel_names, checks


def _required_version(distribution: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise DataContractError(
            f"Fallback requires missing distribution {distribution}=={expected}"
        ) from error
    if actual != str(expected):
        _fail(
            f"Fallback version mismatch for {distribution}: expected {expected}, "
            f"observed {actual}"
        )


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _fallback_recompute(
    *,
    config: Mapping[str, Any],
    root: Path,
    v1_paths: Mapping[str, Path],
    covariance_failure: str,
    validated_whole_metadata: pd.DataFrame,
    base_provenance: dict[str, Any],
) -> V2WholeData:
    fallback = config["v1_inputs"]["fallback"]
    if not bool(fallback["enabled"]):
        _fail(f"V1 covariance reuse failed and fallback is disabled: {covariance_failure}")
    hashes = config["v1_inputs"]["hashes"]
    prepared_file_hash = _check_hash(
        v1_paths["prepared_epochs"],
        hashes["prepared_epochs_sha256"],
        "V1 prepared epochs",
    )
    prepared_metadata_hash = _check_hash(
        v1_paths["prepared_metadata"],
        hashes["prepared_metadata_sha256"],
        "V1 prepared metadata",
    )
    prepared_metadata = _read_metadata(v1_paths["prepared_metadata"])
    try:
        with np.load(v1_paths["prepared_epochs"], allow_pickle=False) as archive:
            expected_keys = list(config["v1_inputs"]["npz_keys"]["prepared_epochs"])
            if archive.files != expected_keys:
                _fail(f"Prepared epoch NPZ keys mismatch: {archive.files}")
            eeg = np.array(archive["X"], copy=True)
            labels = np.array(archive["y"], copy=True)
            channel_names = np.array(archive["channel_names"], copy=True)
            sampling_frequency = float(np.asarray(archive["sampling_frequency_hz"]))
    except (OSError, ValueError, KeyError) as error:
        raise DataContractError(
            f"Could not read exact prepared V1 inputs: {v1_paths['prepared_epochs']}"
        ) from error
    expected = config["expected_data"]
    representation = config["representation"]
    if eeg.shape != tuple(expected["eeg_shape"]):
        _fail(f"Prepared EEG shape mismatch: {eeg.shape}")
    if str(eeg.dtype) != str(representation["input_eeg_dtype"]):
        _fail(f"Prepared EEG dtype mismatch: {eeg.dtype}")
    if not np.isfinite(eeg).all():
        _fail("Prepared EEG contains NaN or Inf")
    if not np.array_equal(
        channel_names.astype(str), np.asarray(config["dataset"]["eeg_channels"], dtype=str)
    ):
        _fail("Prepared EEG channel order mismatch")
    if sampling_frequency != float(config["preprocessing"]["sampling_frequency_hz"]):
        _fail("Prepared sampling frequency mismatch")
    _validate_trial_metadata(
        prepared_metadata, config, prepared=True, labels=labels
    )
    regenerated_metadata = prepared_metadata.copy()
    regenerated_metadata.insert(
        0, "covariance_index", np.arange(len(regenerated_metadata), dtype=np.int64)
    )
    _validate_trial_metadata(regenerated_metadata, config, prepared=False)
    if not regenerated_metadata.equals(validated_whole_metadata):
        _fail("Prepared and exact V1 WHOLE metadata are not identical")

    for distribution, version in fallback["required_versions"].items():
        _required_version(str(distribution), str(version))
    from pyriemann.estimation import Covariances

    n_trials, n_channels, _ = eeg.shape
    covariances = np.empty((n_trials, n_channels, n_channels), dtype=np.float64)
    transformer = Covariances(estimator=str(representation["covariance_estimator"]))
    batch_size = int(fallback["batch_size"])
    for start in range(0, n_trials, batch_size):
        stop = min(start + batch_size, n_trials)
        batch = np.asarray(eeg[start:stop], dtype=np.float64)
        estimated = transformer.transform(batch)
        if bool(representation["symmetrize_covariance"]):
            estimated = 0.5 * (estimated + estimated.transpose(0, 2, 1))
        covariances[start:stop] = estimated
    covariance_checks = _validate_covariance_array(
        covariances,
        channel_names,
        config,
        expected_content_hash=None,
    )

    v2_cache = _resolve(root, config["project"]["cache_dir"])
    v1_cache = v1_paths["covariances"].parent.resolve()
    if v2_cache == v1_cache:
        _fail("V2 fallback cache must not equal the V1 cache directory")
    covariance_output = v2_cache / str(fallback["covariances_filename"])
    metadata_output = v2_cache / str(fallback["metadata_filename"])
    provenance_output = v2_cache / str(fallback["provenance_filename"])
    _atomic_save_npz(
        covariance_output,
        whole=covariances,
        channel_names=np.asarray(channel_names).astype(str),
    )
    _atomic_save_csv(metadata_output, regenerated_metadata)
    provenance = {
        **base_provenance,
        "source": "v2_recomputed_from_exact_v1_prepared_inputs",
        "read_only_v1": True,
        "fallback_reason": covariance_failure,
        "prepared_epochs_sha256": prepared_file_hash,
        "prepared_metadata_sha256": prepared_metadata_hash,
        "whole_array_content_sha256": covariance_checks[
            "whole_array_content_sha256"
        ],
        "whole_shape": list(covariances.shape),
        "whole_dtype": str(covariances.dtype),
        "fallback_outputs": {
            "covariances": str(covariance_output),
            "metadata": str(metadata_output),
            "provenance": str(provenance_output),
        },
        "numerical_checks": covariance_checks,
    }
    _atomic_save_json(provenance_output, provenance)
    covariances.setflags(write=False)
    channel_names.setflags(write=False)
    return V2WholeData(
        covariances=covariances,
        metadata=regenerated_metadata,
        channel_names=channel_names,
        provenance=provenance,
    )


def load_v2_whole(
    config_or_path: Mapping[str, Any] | str | Path,
    root: str | Path,
) -> V2WholeData:
    """Validate and load exact V1 WHOLE covariances for V2.

    The normal reuse path is strictly read-only and creates no V2 directory.
    A covariance-identity failure alone can trigger fallback recomputation from
    exact, hash-pinned prepared V1 inputs into the distinct V2 cache.
    """

    config, config_hash, config_hash_kind = _config_and_identity(config_or_path)
    project_root = Path(root).expanduser().resolve()
    v1_root = _resolve(project_root, config["v1_inputs"]["root"])
    v1_paths = {
        key: _resolve(v1_root, value)
        for key, value in config["v1_inputs"]["paths"].items()
    }
    hashes = config["v1_inputs"]["hashes"]
    frozen_hash = _validate_v1_frozen_config(
        v1_paths["frozen_config"], hashes["frozen_config_sha256"], config
    )
    dataset_metadata = _load_json_with_hash(
        v1_paths["dataset_metadata"],
        hashes["dataset_metadata_sha256"],
        "V1 dataset metadata",
    )
    _assert_equal(
        dataset_metadata.get("config_sha256"),
        hashes["frozen_config_sha256"],
        "dataset metadata config hash",
    )
    _validate_dataset_metadata(dataset_metadata, config)
    covariance_summary = _load_json_with_hash(
        v1_paths["covariance_summary"],
        hashes["covariance_summary_sha256"],
        "V1 covariance summary",
    )
    _validate_covariance_summary(covariance_summary, config)
    sanity_hash = _check_hash(
        v1_paths["covariance_sanity"],
        hashes["covariance_sanity_sha256"],
        "V1 covariance sanity table",
    )
    whole_metadata_hash = _check_hash(
        v1_paths["whole_metadata"],
        hashes["whole_metadata_sha256"],
        "V1 WHOLE metadata",
    )
    metadata = _read_metadata(v1_paths["whole_metadata"])
    _validate_trial_metadata(metadata, config, prepared=False)

    base_provenance: dict[str, Any] = {
        "protocol_name": config["protocol"]["name"],
        "protocol_version": str(config["protocol"]["version"]),
        "protocol_sha256": str(config["protocol"]["protocol_sha256"]),
        "v2_config_hash": config_hash,
        "v2_config_hash_kind": config_hash_kind,
        "v1_root": str(v1_root),
        "v1_frozen_config_sha256": frozen_hash,
        "v1_whole_metadata_sha256": whole_metadata_hash,
        "v1_covariance_sanity_sha256": sanity_hash,
        "session": str(config["dataset"]["primary_session_key"]),
        "subjects": [int(value) for value in config["dataset"]["subjects"]],
        "classes": [str(value) for value in config["dataset"]["classes"]],
        "channel_names": [str(value) for value in config["dataset"]["eeg_channels"]],
    }
    try:
        covariances, channel_names, covariance_checks = _load_reused_covariance(
            v1_paths["covariances"],
            hashes["covariances_file_sha256"],
            hashes["whole_array_content_sha256"],
            config,
        )
    except CovarianceIdentityError as error:
        return _fallback_recompute(
            config=config,
            root=project_root,
            v1_paths=v1_paths,
            covariance_failure=str(error),
            validated_whole_metadata=metadata,
            base_provenance=base_provenance,
        )

    covariances.setflags(write=False)
    channel_names.setflags(write=False)
    provenance = {
        **base_provenance,
        "source": "exact_v1_read_only_reuse",
        "read_only_v1": True,
        "fallback_reason": None,
        "whole_shape": list(covariances.shape),
        "whole_dtype": str(covariances.dtype),
        "numerical_checks": covariance_checks,
    }
    return V2WholeData(
        covariances=covariances,
        metadata=metadata,
        channel_names=channel_names,
        provenance=provenance,
    )
