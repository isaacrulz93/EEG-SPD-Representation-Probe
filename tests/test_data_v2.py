"""Synthetic and actual read-only tests for the V2 data contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_v2 import DataContractError, load_config, load_v2_whole


ROOT = Path(__file__).resolve().parents[1]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _synthetic_contract(tmp_path: Path) -> tuple[dict[str, object], list[Path]]:
    config = copy.deepcopy(
        yaml.safe_load((ROOT / "configs/bnci2014_001_geometry_v2.yaml").read_text())
    )
    config["v1_inputs"]["root"] = "v1"
    paths = config["v1_inputs"]["paths"]
    paths.update(
        {
            "frozen_config": "frozen_config.yaml",
            "dataset_metadata": "dataset_metadata.json",
            "covariance_summary": "covariance_summary.json",
            "covariance_sanity": "covariance_sanity.csv",
            "covariances": "covariances.npz",
            "whole_metadata": "whole_metadata.csv",
            "prepared_epochs": "prepared_epochs.npz",
            "prepared_metadata": "prepared_metadata.csv",
        }
    )
    config["project"]["cache_dir"] = "v2_cache"
    config["dataset"].update(
        {
            "name": "Synthetic",
            "subjects": [1, 2],
            "primary_session_key": "0train",
            "primary_session_number": 1,
            "available_sessions": ["0train", "1test"],
            "classes": ["left", "right"],
            "eeg_channels": ["C1", "C2", "C3"],
            "output_eeg_channels": 3,
            "output_eog_channels": 0,
        }
    )
    config["preprocessing"].update(
        {
            "epoch_tmin_seconds": 0.0,
            "epoch_tmax_seconds": 0.19,
            "source_event_interval_seconds": [2.0, 2.2],
            "observed_source_epoch_seconds": [2.0, 2.19],
            "bandpass_hz": [8.0, 32.0],
            "sampling_frequency_hz": 100.0,
            "samples_per_trial": 20,
        }
    )
    config["representation"]["coordinate_dimension"] = 6
    config["expected_data"].update(
        {
            "eeg_shape": [8, 3, 20],
            "whole_covariance_shape": [8, 3, 3],
            "trials_total": 8,
            "trials_per_subject": 4,
            "trials_per_run": 2,
            "trials_per_subject_class": 2,
            "trials_per_subject_run_class": 1,
            "runs": ["0", "1"],
            "run_trial_ids": [1, 2],
            "subject_trial_ids": [1, 4],
        }
    )
    config["classifiers"]["metrics"]["class_order"] = ["left", "right"]
    config["classifiers"]["metrics"]["confusion_matrix_shape"] = [2, 2]

    v1 = tmp_path / "v1"
    v1.mkdir()
    frozen = {
        "dataset": {
            "name": "Synthetic",
            "subjects": [1, 2],
            "primary_session_key": "0train",
            "primary_session_number": 1,
            "classes": ["left", "right"],
            "eeg_channels": ["C1", "C2", "C3"],
        },
        "preprocessing": {
            "epoch_tmin_seconds": 0.0,
            "epoch_tmax_seconds": 0.19,
            "source_event_interval_seconds": [2.0, 2.2],
            "bandpass_hz": [8.0, 32.0],
            "resample_hz": None,
            "baseline": None,
            "expected_sampling_frequency_hz": 100.0,
            "expected_samples_per_trial": 20,
        },
        "representation": {
            "covariance_estimator": "oas",
            "symmetrize_covariance": True,
            "covariance_extra_regularization": "none",
        },
    }
    frozen_path = v1 / "frozen_config.yaml"
    frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False))

    rows = []
    labels = []
    sample = 0
    for subject in [1, 2]:
        for trial_id in range(1, 5):
            run = "0" if trial_id <= 2 else "1"
            run_trial_id = trial_id if run == "0" else trial_id - 2
            label = "left" if run_trial_id == 1 else "right"
            labels.append(label)
            rows.append(
                {
                    "sample_index": sample,
                    "subject": subject,
                    "session": "0train",
                    "run": run,
                    "trial_id": trial_id,
                    "run_trial_id": run_trial_id,
                    "trial_uid": f"S{subject:02d}_0train_T{trial_id:03d}",
                    "class_label": label,
                }
            )
            sample += 1
    prepared_metadata = pd.DataFrame(rows)
    prepared_metadata_path = v1 / "prepared_metadata.csv"
    prepared_metadata.to_csv(prepared_metadata_path, index=False)
    whole_metadata = prepared_metadata.copy()
    whole_metadata.insert(0, "covariance_index", np.arange(8, dtype=np.int64))
    whole_metadata_path = v1 / "whole_metadata.csv"
    whole_metadata.to_csv(whole_metadata_path, index=False)

    rng = np.random.default_rng(20260809)
    eeg = rng.normal(size=(8, 3, 20)).astype(np.float32)
    prepared_epochs_path = v1 / "prepared_epochs.npz"
    np.savez(
        prepared_epochs_path,
        X=eeg,
        y=np.asarray(labels),
        channel_names=np.asarray(["C1", "C2", "C3"]),
        sampling_frequency_hz=np.asarray(100.0),
    )
    covariances = np.stack(
        [
            np.eye(3) * (1.0 + index / 10.0)
            + np.full((3, 3), 0.01 * (index + 1))
            for index in range(8)
        ]
    ).astype(np.float64)
    covariance_path = v1 / "covariances.npz"
    np.savez(
        covariance_path,
        whole=covariances,
        window5=np.repeat(covariances, 5, axis=0),
        channel_names=np.asarray(["C1", "C2", "C3"]),
    )
    dataset_metadata = {
        "config_sha256": _file_hash(frozen_path),
        "dataset": "Synthetic",
        "subjects": [1, 2],
        "n_subjects": 2,
        "sessions_observed": ["0train"],
        "n_sessions_observed": 1,
        "sessions_available": ["0train", "1test"],
        "classes_observed": ["left", "right"],
        "n_eeg_channels": 3,
        "n_eog_channels_output": 0,
        "eeg_channels": ["C1", "C2", "C3"],
        "channel_types": ["eeg"] * 3,
        "sampling_frequency_hz": 100.0,
        "source_event_interval_seconds": [2.0, 2.2],
        "epoch_time_interval_relative_to_mi_cue_seconds": [0.0, 0.19],
        "epoch_time_interval_observed_source_seconds": [2.0, 2.19],
        "samples_per_trial": 20,
        "array_shape": [8, 3, 20],
        "n_trials": 8,
        "bandpass_hz": [8.0, 32.0],
        "resample_hz": None,
        "baseline": None,
        "cached_epoch_dtype": "float32",
        "signal_units": "volts",
    }
    dataset_metadata_path = v1 / "dataset_metadata.json"
    _write_json(dataset_metadata_path, dataset_metadata)
    covariance_summary = {
        "estimator": "oas",
        "oas_assume_centered": False,
        "extra_regularization": "none",
        "symmetrized": True,
        "whole_shape": [8, 3, 3],
        "sanity": {
            "WHOLE": {
                "count": 8,
                "spd_count": 8,
                "non_spd_count": 0,
                "nan_count": 0,
                "inf_count": 0,
            }
        },
    }
    covariance_summary_path = v1 / "covariance_summary.json"
    _write_json(covariance_summary_path, covariance_summary)
    covariance_sanity_path = v1 / "covariance_sanity.csv"
    pd.DataFrame(
        {"representation": ["WHOLE"] * 8, "is_spd": [True] * 8}
    ).to_csv(covariance_sanity_path, index=False)

    hashes = config["v1_inputs"]["hashes"]
    hashes.update(
        {
            "frozen_config_sha256": _file_hash(frozen_path),
            "dataset_metadata_sha256": _file_hash(dataset_metadata_path),
            "covariance_summary_sha256": _file_hash(covariance_summary_path),
            "covariance_sanity_sha256": _file_hash(covariance_sanity_path),
            "covariances_file_sha256": _file_hash(covariance_path),
            "whole_array_content_sha256": _array_hash(covariances),
            "whole_metadata_sha256": _file_hash(whole_metadata_path),
            "prepared_epochs_sha256": _file_hash(prepared_epochs_path),
            "prepared_metadata_sha256": _file_hash(prepared_metadata_path),
        }
    )
    inputs = [
        frozen_path,
        dataset_metadata_path,
        covariance_summary_path,
        covariance_sanity_path,
        covariance_path,
        whole_metadata_path,
        prepared_epochs_path,
        prepared_metadata_path,
    ]
    return config, inputs


def test_normal_reuse_is_read_only_and_row_aligned(tmp_path: Path) -> None:
    config, v1_inputs = _synthetic_contract(tmp_path)
    before = {path: _file_hash(path) for path in v1_inputs}
    result = load_v2_whole(config, tmp_path)
    after = {path: _file_hash(path) for path in v1_inputs}
    assert before == after
    assert result.covariances.shape == (8, 3, 3)
    assert result.covariances.dtype == np.float64
    assert not result.covariances.flags.writeable
    assert result.metadata.covariance_index.tolist() == list(range(8))
    assert result.provenance["source"] == "exact_v1_read_only_reuse"
    assert not (tmp_path / "v2_cache").exists()


def test_covariance_hash_mismatch_recomputes_only_in_v2_cache(
    tmp_path: Path,
) -> None:
    config, v1_inputs = _synthetic_contract(tmp_path)
    config["v1_inputs"]["hashes"]["covariances_file_sha256"] = "0" * 64
    before = {path: _file_hash(path) for path in v1_inputs}
    result = load_v2_whole(config, tmp_path)
    after = {path: _file_hash(path) for path in v1_inputs}
    assert before == after
    assert result.provenance["source"] == (
        "v2_recomputed_from_exact_v1_prepared_inputs"
    )
    assert result.covariances.shape == (8, 3, 3)
    assert np.all(np.linalg.eigvalsh(result.covariances) > 0)
    v2_cache = tmp_path / "v2_cache"
    assert (v2_cache / "whole_covariances_recomputed.npz").is_file()
    assert (v2_cache / "whole_metadata_recomputed.csv").is_file()
    assert (v2_cache / "data_provenance.json").is_file()
    assert all(v2_cache not in path.parents for path in v1_inputs)


def test_fallback_rejects_prepared_hash_failure_without_writing(
    tmp_path: Path,
) -> None:
    config, v1_inputs = _synthetic_contract(tmp_path)
    config["v1_inputs"]["hashes"]["covariances_file_sha256"] = "0" * 64
    config["v1_inputs"]["hashes"]["prepared_epochs_sha256"] = "f" * 64
    before = {path: _file_hash(path) for path in v1_inputs}
    with pytest.raises(DataContractError, match="prepared epochs SHA-256 mismatch"):
        load_v2_whole(config, tmp_path)
    assert before == {path: _file_hash(path) for path in v1_inputs}
    assert not (tmp_path / "v2_cache").exists()


def test_metadata_identity_failure_does_not_activate_fallback(
    tmp_path: Path,
) -> None:
    config, _ = _synthetic_contract(tmp_path)
    metadata_path = tmp_path / "v1/whole_metadata.csv"
    frame = pd.read_csv(metadata_path)
    frame.loc[0, "trial_uid"] = frame.loc[1, "trial_uid"]
    frame.to_csv(metadata_path, index=False)
    config["v1_inputs"]["hashes"]["whole_metadata_sha256"] = _file_hash(
        metadata_path
    )
    with pytest.raises(DataContractError, match="trial_uid must be globally unique"):
        load_v2_whole(config, tmp_path)
    assert not (tmp_path / "v2_cache").exists()


def test_production_config_and_actual_v1_cache_read_only_smoke() -> None:
    config_path = ROOT / "configs/bnci2014_001_geometry_v2.yaml"
    config = load_config(config_path)
    v1_root = ROOT / config["v1_inputs"]["root"]
    paths = [v1_root / value for value in config["v1_inputs"]["paths"].values()]
    if not all(path.exists() for path in paths):
        pytest.skip("Local ignored V1 cache is unavailable")
    before = {path: (_file_hash(path), path.stat().st_mtime_ns) for path in paths}
    result = load_v2_whole(config_path, ROOT)
    after = {path: (_file_hash(path), path.stat().st_mtime_ns) for path in paths}
    assert before == after
    assert result.covariances.shape == (2592, 22, 22)
    assert result.provenance["source"] == "exact_v1_read_only_reuse"
