"""Synthetic-only tests for the Trajectory Anatomy v0 V1 data barrier."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.data_trajectory_v0 as data_module
from src.data_trajectory_v0 import (
    TrajectoryDataContractError,
    array_content_sha256,
    file_sha256,
    load_window5_v0,
    trial_uid_set_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _synthetic_contract(tmp_path: Path) -> tuple[dict[str, object], list[Path]]:
    config = copy.deepcopy(
        yaml.safe_load((ROOT / "configs/bnci2014_001_trajectory_v0.yaml").read_text())
    )
    v1 = tmp_path / "v1"
    v1.mkdir()
    config["v1_inputs"] = copy.deepcopy(config["v1_inputs"])
    for name, entry in config["v1_inputs"].items():
        if isinstance(entry, dict) and "path" in entry:
            suffix = Path(entry["path"]).suffix or ".bin"
            entry["path"] = f"v1/{name}{suffix}"

    config["dataset"].update(
        {
            "name": "Synthetic",
            "subjects": [1, 2],
            "runs": [0, 1],
            "classes": ["left_hand", "right_hand"],
            "expected_trials": 16,
            "expected_trials_per_subject": 8,
            "expected_trials_per_subject_class": 4,
            "expected_trials_per_subject_run": 4,
            "expected_trials_per_subject_run_class": 2,
            "eeg_channels": ["C1", "C2"],
        }
    )
    config["preprocessing"].update(
        {
            "sampling_frequency_hz": 100.0,
            "samples_per_trial": 100,
            "epoch_tmax_seconds": 0.99,
        }
    )
    config["window5"].update(
        {
            "samples_per_window": 20,
            "expected_covariance_shape": [80, 2, 2],
            "expected_trial_tensor_shape": [16, 5, 2, 2],
        }
    )
    config["factor_decomposition"].update(
        {"n_subjects": 2, "n_classes": 2, "n_per_cell": 4}
    )

    frozen = {
        "dataset": {
            "name": "Synthetic",
            "subjects": [1, 2],
            "primary_session_key": "0train",
            "classes": ["left_hand", "right_hand"],
            "eeg_channels": ["C1", "C2"],
        },
        "preprocessing": {
            "bandpass_hz": [8.0, 32.0],
            "epoch_tmin_seconds": 0.0,
            "epoch_tmax_seconds": 0.99,
            "expected_samples_per_trial": 100,
            "expected_sampling_frequency_hz": 100.0,
            "resample_hz": None,
            "baseline": None,
        },
        "representation": {
            "covariance_estimator": "oas",
            "n_windows": 5,
            "overlap_samples": 0,
            "remainder_policy": "require_exact_division",
            "expected_window_samples": 20,
            "covariance_extra_regularization": "none",
        },
    }
    source_config = tmp_path / config["v1_inputs"]["source_config"]["path"]
    source_config.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    frozen_config = tmp_path / config["v1_inputs"]["frozen_config"]["path"]
    frozen_config.write_bytes(source_config.read_bytes())

    trial_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    sample = 0
    covariance_index = 0
    for subject in (1, 2):
        trial_id = 0
        for run in (0, 1):
            run_trial_id = 0
            for class_label in ("left_hand", "right_hand"):
                for _ in range(2):
                    trial_id += 1
                    run_trial_id += 1
                    uid = f"S{subject:02d}_0train_T{trial_id:03d}"
                    trial = {
                        "sample_index": sample,
                        "subject": subject,
                        "session": "0train",
                        "run": run,
                        "trial_id": trial_id,
                        "run_trial_id": run_trial_id,
                        "trial_uid": uid,
                        "class_label": class_label,
                    }
                    trial_rows.append(trial)
                    for window in range(1, 6):
                        window_rows.append(
                            {
                                "covariance_index": covariance_index,
                                **trial,
                                "window_index": window,
                            }
                        )
                        covariance_index += 1
                    sample += 1
    trial_frame = pd.DataFrame(trial_rows)
    window_frame = pd.DataFrame(window_rows)
    config["dataset"]["expected_trial_uid_set_sha256"] = trial_uid_set_sha256(
        trial_frame.trial_uid.tolist()
    )
    window_path = tmp_path / config["v1_inputs"]["window5_metadata"]["path"]
    window_frame.to_csv(window_path, index=False)
    whole_metadata = trial_frame.copy()
    whole_metadata.insert(0, "covariance_index", np.arange(16))
    whole_metadata_path = tmp_path / config["v1_inputs"]["whole_metadata"]["path"]
    whole_metadata.to_csv(whole_metadata_path, index=False)

    whole = np.stack(
        [np.diag([1.0 + index / 100, 1.5 + index / 100]) for index in range(16)]
    ).astype(np.float64)
    flat = np.stack(
        [
            np.diag([1.0 + index / 1000, 1.3 + index / 1000])
            for index in range(80)
        ]
    ).astype(np.float64)
    channels = np.asarray(["C1", "C2"])
    covariance_path = tmp_path / config["v1_inputs"]["covariances"]["path"]
    np.savez(covariance_path, whole=whole, window5=flat, channel_names=channels)

    prepared_path = tmp_path / config["v1_inputs"]["prepared_epochs"]["path"]
    eeg = np.zeros((16, 2, 100), dtype=np.float32)
    np.savez(prepared_path, X=eeg)
    prepared_metadata_path = tmp_path / config["v1_inputs"]["prepared_metadata"]["path"]
    trial_frame.to_csv(prepared_metadata_path, index=False)
    summary_path = tmp_path / config["v1_inputs"]["covariance_summary"]["path"]
    _write(summary_path, b"{}\n")
    dataset_path = tmp_path / config["v1_inputs"]["dataset_metadata"]["path"]
    _write(dataset_path, b"{}\n")

    inputs = config["v1_inputs"]
    inputs["source_config"]["sha256"] = file_sha256(source_config)
    inputs["frozen_config"]["sha256"] = file_sha256(frozen_config)
    inputs["covariances"].update(
        {
            "file_sha256": file_sha256(covariance_path),
            "window5_content_sha256": array_content_sha256(flat),
            "whole_content_sha256": array_content_sha256(whole),
            "channel_names_content_sha256": array_content_sha256(channels),
        }
    )
    inputs["window5_metadata"]["sha256"] = file_sha256(window_path)
    inputs["whole_metadata"]["sha256"] = file_sha256(whole_metadata_path)
    inputs["prepared_epochs"].update(
        {
            "file_sha256": file_sha256(prepared_path),
            "x_content_sha256": array_content_sha256(eeg),
        }
    )
    inputs["prepared_metadata"]["sha256"] = file_sha256(prepared_metadata_path)
    inputs["covariance_summary"]["sha256"] = file_sha256(summary_path)
    inputs["dataset_metadata"]["sha256"] = file_sha256(dataset_path)
    paths = [
        tmp_path / entry["path"]
        for entry in inputs.values()
        if isinstance(entry, dict) and "path" in entry
    ]
    return config, paths


def test_strict_loader_returns_read_only_aligned_window_and_whole(tmp_path: Path) -> None:
    config, _ = _synthetic_contract(tmp_path)
    data = load_window5_v0(config, tmp_path)
    assert data.states.shape == (16, 5, 2, 2)
    assert data.whole_covariances.shape == (16, 2, 2)
    assert data.states.dtype == data.whole_covariances.dtype == np.float64
    assert not data.states.flags.writeable
    assert not data.whole_covariances.flags.writeable
    assert not data.channel_names.flags.writeable
    assert list(data.metadata.sample_index) == list(range(16))
    assert set(data.metadata.session) == {"0train"}
    assert data.provenance["fallback_used"] is False
    with pytest.raises(ValueError):
        data.states[0, 0, 0, 0] = 0.0
    detached = data.metadata
    detached.loc[0, "subject"] = 99
    assert data.metadata.loc[0, "subject"] == 1


def test_forbidden_session_rejected_before_covariance_npz_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _synthetic_contract(tmp_path)
    metadata_path = tmp_path / config["v1_inputs"]["window5_metadata"]["path"]
    frame = pd.read_csv(metadata_path)
    frame.loc[0, "session"] = "1test"
    frame.to_csv(metadata_path, index=False)
    config["v1_inputs"]["window5_metadata"]["sha256"] = file_sha256(metadata_path)
    called = False

    def forbidden_np_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("covariance NPZ must not be opened")

    monkeypatch.setattr(data_module.np, "load", forbidden_np_load)
    with pytest.raises(TrajectoryDataContractError, match="forbidden session barrier"):
        load_window5_v0(config, tmp_path)
    assert not called


def test_hash_failure_never_changes_any_v1_input(tmp_path: Path) -> None:
    config, paths = _synthetic_contract(tmp_path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    config["v1_inputs"]["covariances"]["window5_content_sha256"] = "0" * 64
    with pytest.raises(TrajectoryDataContractError, match="member hash mismatch"):
        load_window5_v0(config, tmp_path)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after


def test_trial_window_alignment_is_not_silently_reordered(tmp_path: Path) -> None:
    config, _ = _synthetic_contract(tmp_path)
    metadata_path = tmp_path / config["v1_inputs"]["window5_metadata"]["path"]
    frame = pd.read_csv(metadata_path)
    frame.loc[[0, 1], "window_index"] = [2, 1]
    frame.to_csv(metadata_path, index=False)
    config["v1_inputs"]["window5_metadata"]["sha256"] = file_sha256(metadata_path)
    with pytest.raises(TrajectoryDataContractError, match="window_index"):
        load_window5_v0(config, tmp_path)


def test_whole_control_must_align_to_same_trial_metadata(tmp_path: Path) -> None:
    config, _ = _synthetic_contract(tmp_path)
    whole_path = tmp_path / config["v1_inputs"]["whole_metadata"]["path"]
    frame = pd.read_csv(whole_path)
    frame.loc[0, "trial_uid"] = frame.loc[1, "trial_uid"]
    frame.to_csv(whole_path, index=False)
    config["v1_inputs"]["whole_metadata"]["sha256"] = file_sha256(whole_path)
    with pytest.raises(TrajectoryDataContractError, match="WHOLE/WINDOW5 trial alignment"):
        load_window5_v0(config, tmp_path)
