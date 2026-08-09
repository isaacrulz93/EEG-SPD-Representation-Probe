"""Frozen BNCI2014_001 data preparation and run metadata utilities.

MOABB and MNE are imported lazily, after their data-directory environment
variables have been pointed at this project's ignored cache.  This avoids
writing downloaded EEG data outside the repository-specific cache.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


EXPECTED_DATASET = "BNCI2014_001"
EXPECTED_SUBJECTS = list(range(1, 10))
EXPECTED_TRIALS_PER_SESSION = 288


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the project's YAML configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    for section in ("project", "dataset", "preprocessing", "representation"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing mapping section {section!r} in {config_path}")
    validate_frozen_data_config(config)
    return config


def validate_frozen_data_config(config: Mapping[str, Any]) -> None:
    """Fail early if the requested run no longer matches the frozen pilot."""

    dataset = config["dataset"]
    prep = config["preprocessing"]
    representation = config["representation"]

    if dataset.get("name") != EXPECTED_DATASET:
        raise ValueError(f"This pipeline only supports {EXPECTED_DATASET}")
    if list(dataset.get("subjects", [])) != EXPECTED_SUBJECTS:
        raise ValueError("The frozen pilot requires all BNCI2014_001 subjects 1..9")
    if len(dataset.get("classes", [])) != 4:
        raise ValueError("The frozen pilot requires exactly four motor-imagery classes")
    channels = list(dataset.get("eeg_channels", []))
    if len(channels) != 22 or len(set(channels)) != 22:
        raise ValueError("The frozen pilot requires 22 unique EEG channels")
    bandpass = [float(x) for x in prep.get("bandpass_hz", [])]
    if len(bandpass) != 2 or not (0.0 < bandpass[0] < bandpass[1]):
        raise ValueError("bandpass_hz must contain increasing positive cutoffs")
    if float(prep.get("epoch_tmin_seconds")) != 0.0:
        raise ValueError("The frozen pilot epoch start is fixed at 0.0 s")
    if float(prep.get("epoch_tmax_seconds")) != 3.996:
        raise ValueError("The frozen pilot epoch end is fixed at 3.996 s")
    if prep.get("resample_hz") is not None:
        raise ValueError("The frozen pilot does not resample")
    if prep.get("baseline") is not None:
        raise ValueError("The frozen pilot does not use baseline correction")
    if representation.get("covariance_estimator") != "oas":
        raise ValueError("The frozen pilot covariance estimator is fixed to OAS")
    if int(representation.get("n_windows")) != 5:
        raise ValueError("The frozen pilot requires exactly five windows")
    if int(representation.get("overlap_samples")) != 0:
        raise ValueError("The frozen pilot windows must not overlap")
    if representation.get("remainder_policy") != "require_exact_division":
        raise ValueError("The frozen pilot requires exact window division")
    if representation.get("covariance_extra_regularization") != "none":
        raise ValueError("The frozen pilot does not add covariance regularization")


def project_path(project_root: str | Path, configured_path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root).resolve() / path).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_config(config_path: str | Path, output_dir: str | Path) -> Path:
    """Copy the exact protocol beside outputs, refusing conflicting reruns."""

    source = Path(config_path).expanduser().resolve()
    destination = Path(output_dir) / "frozen_config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise RuntimeError(
            f"{destination} contains a different frozen protocol; use a distinct "
            "output directory rather than overwriting it"
        )
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment_metadata() -> dict[str, Any]:
    """Capture the runtime facts requested by the study protocol."""

    torch_version = _distribution_version("torch")
    cuda_available = False
    cuda_device_count = 0
    cuda_runtime_version = None
    mps_available = False
    if torch_version is not None and importlib.util.find_spec("torch") is not None:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        cuda_runtime_version = torch.version.cuda
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())

    packages = {
        name: _distribution_version(distribution)
        for name, distribution in {
            "numpy": "numpy",
            "scipy": "scipy",
            "scikit_learn": "scikit-learn",
            "matplotlib": "matplotlib",
            "pandas": "pandas",
            "mne": "mne",
            "moabb": "moabb",
            "pyriemann": "pyriemann",
            "pyyaml": "PyYAML",
            "torch": "torch",
        }.items()
    }
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "compute": {
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_runtime_version": cuda_runtime_version,
            "apple_mps_available": mps_available,
            "cpu_count": os.cpu_count(),
        },
        "packages": packages,
    }


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _validate_and_augment_metadata(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    subjects: list[int],
    session_key: str,
    classes: list[str],
) -> pd.DataFrame:
    """Attach deterministic within-subject trial identifiers."""

    metadata = metadata.reset_index(drop=True).copy()
    labels = np.asarray(labels).astype(str)
    if len(metadata) != len(labels):
        raise RuntimeError("MOABB labels and metadata have different lengths")
    missing = {"subject", "session", "run"} - set(metadata.columns)
    if missing:
        raise RuntimeError(f"MOABB metadata is missing columns: {sorted(missing)}")

    metadata["subject"] = pd.to_numeric(metadata["subject"], errors="raise").astype(int)
    metadata["session"] = metadata["session"].astype(str)
    metadata["run"] = metadata["run"].astype(str)
    if sorted(metadata["subject"].unique().tolist()) != sorted(subjects):
        raise RuntimeError("Observed MOABB subjects do not match the frozen subject list")
    if metadata["session"].unique().tolist() != [session_key]:
        raise RuntimeError(
            f"Expected only session {session_key!r}, observed "
            f"{metadata['session'].unique().tolist()}"
        )
    if set(labels.tolist()) != set(classes):
        raise RuntimeError(
            f"Expected classes {classes}, observed {sorted(set(labels.tolist()))}"
        )

    metadata.insert(0, "sample_index", np.arange(len(metadata), dtype=np.int64))
    metadata["trial_id"] = (
        metadata.groupby(["subject", "session"], sort=False).cumcount() + 1
    ).astype(np.int64)
    metadata["run_trial_id"] = (
        metadata.groupby(["subject", "session", "run"], sort=False).cumcount() + 1
    ).astype(np.int64)
    metadata["class_label"] = labels
    metadata["trial_uid"] = [
        f"S{subject:02d}_{session}_T{trial_id:03d}"
        for subject, session, trial_id in metadata[
            ["subject", "session", "trial_id"]
        ].itertuples(index=False, name=None)
    ]

    duplicate = metadata.duplicated(["subject", "session", "trial_id"]).any()
    if duplicate:
        raise RuntimeError("Constructed trial identifiers are not unique")
    counts = metadata.groupby(["subject", "session"], observed=True).size()
    if not (counts == EXPECTED_TRIALS_PER_SESSION).all():
        raise RuntimeError(
            "Expected 288 trials per subject/session; observed "
            f"{counts.to_dict()}"
        )
    return metadata[
        [
            "sample_index",
            "subject",
            "session",
            "run",
            "trial_id",
            "run_trial_id",
            "trial_uid",
            "class_label",
        ]
    ]


def _dataset_metadata(
    *,
    epochs: Any,
    x: np.ndarray,
    metadata: pd.DataFrame,
    dataset_interval: list[float],
    available_session_keys: list[str],
    total_available_sessions: int,
    config: Mapping[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    subject_class_counts = (
        metadata.groupby(["subject", "class_label"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    per_subject_class = {
        str(int(subject)): {
            str(label): int(value) for label, value in row.items()
        }
        for subject, row in subject_class_counts.iterrows()
    }
    trials_per_subject = {
        str(int(key)): int(value)
        for key, value in metadata.groupby("subject", observed=True).size().items()
    }
    trials_per_class = {
        str(key): int(value)
        for key, value in metadata.groupby("class_label", observed=True).size().items()
    }
    runs_per_subject = {
        str(int(key)): int(value)
        for key, value in metadata.groupby("subject", observed=True)["run"].nunique().items()
    }
    return {
        "dataset": config["dataset"]["name"],
        "config_sha256": config_hash,
        "n_subjects": int(metadata["subject"].nunique()),
        "subjects": sorted(int(x) for x in metadata["subject"].unique()),
        "n_sessions_observed": int(metadata["session"].nunique()),
        "sessions_observed": metadata["session"].drop_duplicates().tolist(),
        "n_sessions_available": int(total_available_sessions),
        "sessions_available": [str(key) for key in available_session_keys],
        "primary_session_number": int(config["dataset"]["primary_session_number"]),
        "classes_observed": sorted(metadata["class_label"].unique().tolist()),
        "n_classes": int(metadata["class_label"].nunique()),
        "eeg_channels": list(epochs.ch_names),
        "n_eeg_channels": len(epochs.ch_names),
        "n_eog_channels_output": int(
            sum(kind == "eog" for kind in epochs.get_channel_types())
        ),
        "channel_types": list(epochs.get_channel_types()),
        "sampling_frequency_hz": float(epochs.info["sfreq"]),
        "source_event_interval_seconds": [float(v) for v in dataset_interval],
        "epoch_time_interval_relative_to_mi_cue_seconds": [
            float(config["preprocessing"]["epoch_tmin_seconds"]),
            float(config["preprocessing"]["epoch_tmax_seconds"]),
        ],
        "epoch_time_interval_observed_source_seconds": [
            float(epochs.times[0]),
            float(epochs.times[-1]),
        ],
        "samples_per_trial": int(x.shape[-1]),
        "array_shape": [int(v) for v in x.shape],
        "n_trials": int(x.shape[0]),
        "trials_per_subject": trials_per_subject,
        "trials_per_class": trials_per_class,
        "trials_per_subject_class": per_subject_class,
        "runs_per_subject": runs_per_subject,
        "bandpass_hz": [float(v) for v in config["preprocessing"]["bandpass_hz"]],
        "resample_hz": config["preprocessing"]["resample_hz"],
        "baseline": config["preprocessing"]["baseline"],
        "cached_epoch_dtype": str(x.dtype),
        # MOABB's BNCI MAT loader explicitly converts source microvolts to the
        # volts required by MNE RawArray before paradigm preprocessing.
        "signal_units": "volts",
        "source_mat_signal_units": "microvolts",
        "moabb_unit_conversion": "microvolts_to_volts",
    }


def prepare_bnci2014_001(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Download/load, preprocess, validate, and cache the frozen primary data."""

    config_path = Path(config_path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    config = load_config(config_path)
    output_dir = project_path(root, config["project"]["output_dir"])
    table_dir = output_dir / "tables"
    cache_dir = project_path(root, config["project"]["cache_dir"])
    moabb_data_dir = project_path(root, config["project"]["moabb_data_dir"])
    for directory in (output_dir, table_dir, cache_dir, moabb_data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    frozen_config = freeze_config(config_path, output_dir)
    config_hash = sha256_file(frozen_config)

    # These must be set before importing MNE/MOABB so all downloads remain in
    # the repository's ignored cache rather than a user-global data directory.
    os.environ["MNE_DATA"] = str(moabb_data_dir)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(moabb_data_dir)
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    mne.set_log_level("WARNING")
    subjects = [int(x) for x in config["dataset"]["subjects"]]
    session_key = str(config["dataset"]["primary_session_key"])
    classes = [str(x) for x in config["dataset"]["classes"]]
    channels = [str(x) for x in config["dataset"]["eeg_channels"]]
    fmin, fmax = (float(x) for x in config["preprocessing"]["bandpass_hz"])
    dataset = BNCI2014_001(subjects=subjects, sessions=[session_key])
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
    epochs, labels, moabb_metadata = paradigm.get_data(
        dataset=dataset,
        subjects=subjects,
        return_epochs=True,
    )

    # The analysis deliberately loads one session, but the report must state
    # the dataset's full session inventory.  Read the session keys exposed by
    # MOABB for one subject rather than relying on remembered dataset facts.
    session_inventory_dataset = BNCI2014_001(subjects=[subjects[0]])
    session_inventory = session_inventory_dataset.get_data(subjects=[subjects[0]])
    available_session_keys = list(session_inventory[subjects[0]].keys())
    del session_inventory

    if set(epochs.ch_names) != set(channels):
        raise RuntimeError(
            "Observed EEG channels do not match config: "
            f"observed={epochs.ch_names}, configured={channels}"
        )
    if epochs.ch_names != channels:
        epochs.reorder_channels(channels)
    if any(kind != "eeg" for kind in epochs.get_channel_types()):
        raise RuntimeError("A non-EEG channel survived explicit channel selection")

    expected_sfreq = float(config["preprocessing"]["expected_sampling_frequency_hz"])
    if not np.isclose(float(epochs.info["sfreq"]), expected_sfreq, atol=1e-12):
        raise RuntimeError(
            f"Expected {expected_sfreq} Hz, observed {epochs.info['sfreq']} Hz"
        )
    x = epochs.get_data(copy=True).astype(np.float32, copy=False)
    expected_samples = int(config["preprocessing"]["expected_samples_per_trial"])
    expected_shape = (len(subjects) * EXPECTED_TRIALS_PER_SESSION, len(channels), expected_samples)
    if x.shape != expected_shape:
        raise RuntimeError(f"Expected epoch shape {expected_shape}, observed {x.shape}")
    if not np.isfinite(x).all():
        raise RuntimeError("Prepared EEG contains NaN or Inf")
    expected_observed_start = float(dataset.interval[0]) + float(
        config["preprocessing"]["epoch_tmin_seconds"]
    )
    expected_observed_stop = float(dataset.interval[0]) + float(
        config["preprocessing"]["epoch_tmax_seconds"]
    )
    if not (
        np.isclose(
            float(epochs.times[0]), expected_observed_start, atol=0.5 / expected_sfreq
        )
        and np.isclose(
            float(epochs.times[-1]), expected_observed_stop, atol=0.5 / expected_sfreq
        )
    ):
        raise RuntimeError(
            "Unexpected MOABB source-time interval "
            f"[{epochs.times[0]:.9f}, {epochs.times[-1]:.9f}] seconds"
        )

    metadata = _validate_and_augment_metadata(
        moabb_metadata, labels, subjects, session_key, classes
    )
    prepared_path = cache_dir / "prepared_epochs.npz"
    np.savez(
        prepared_path,
        X=x,
        y=np.asarray(labels).astype(str),
        channel_names=np.asarray(channels),
        sampling_frequency_hz=np.asarray(float(epochs.info["sfreq"])),
    )
    cache_metadata_path = cache_dir / "prepared_metadata.csv"
    metadata.to_csv(cache_metadata_path, index=False)
    # This is small, audit-relevant metadata; keep a tracked copy with outputs.
    metadata.to_csv(table_dir / "trial_metadata.csv", index=False)

    dataset_metadata = _dataset_metadata(
        epochs=epochs,
        x=x,
        metadata=metadata,
        dataset_interval=list(dataset.interval),
        available_session_keys=available_session_keys,
        total_available_sessions=int(dataset.n_sessions),
        config=config,
        config_hash=config_hash,
    )
    environment_metadata = collect_environment_metadata()
    write_json(dataset_metadata, table_dir / "dataset_metadata.json")
    write_json(environment_metadata, table_dir / "environment.json")
    return {
        "prepared_epochs": str(prepared_path),
        "prepared_metadata": str(cache_metadata_path),
        "frozen_config": str(frozen_config),
        "dataset_metadata": dataset_metadata,
        "environment_metadata": environment_metadata,
    }
