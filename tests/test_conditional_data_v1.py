"""Synthetic data-contract tests for Conditional Geometry v1."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.data_conditional_v1 as data_module
from src.data_conditional_v1 import (
    CLASS_ORDER,
    ConditionalWholeData,
    DataContractError,
    load_conditional_config,
    load_discovery_whole,
    prepare_confirmatory_whole,
    sha256_array,
    subject_split_positions,
    validate_session_metadata,
    validate_whole_covariances,
)


ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _balanced_metadata(session: str, *, prepared: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_index = 0
    for subject in range(1, 10):
        trial_id = 0
        for run in range(6):
            run_trial_id = 0
            for class_label in CLASS_ORDER:
                for _ in range(12):
                    trial_id += 1
                    run_trial_id += 1
                    row = {
                        "sample_index": sample_index,
                        "subject": subject,
                        "session": session,
                        "run": str(run),
                        "trial_id": trial_id,
                        "run_trial_id": run_trial_id,
                        "trial_uid": f"S{subject:02d}_{session}_T{trial_id:03d}",
                        "class_label": class_label,
                    }
                    if not prepared:
                        row = {"covariance_index": sample_index, **row}
                    rows.append(row)
                    sample_index += 1
    return pd.DataFrame.from_records(rows)


def _synthetic_repository(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "repository"
    root.mkdir()
    config = copy.deepcopy(
        yaml.safe_load(
            (ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    protocol = root / "docs/PROTOCOL_CONDITIONAL_GEOMETRY_V1.md"
    protocol.parent.mkdir()
    protocol.write_bytes(
        (ROOT / "docs/PROTOCOL_CONDITIONAL_GEOMETRY_V1.md").read_bytes()
    )
    config["protocol"]["protocol_path"] = "docs/PROTOCOL_CONDITIONAL_GEOMETRY_V1.md"
    config["project"]["output_dir"] = "outputs/conditional"
    config["project"]["cache_dir"] = "cache/conditional"

    v1 = root / "v1"
    v1.mkdir()
    frozen = {
        "dataset": {
            "name": "BNCI2014_001",
            "subjects": list(range(1, 10)),
            "primary_session_key": "0train",
            "primary_session_number": 1,
            "classes": list(CLASS_ORDER),
            "eeg_channels": list(config["dataset"]["eeg_channels"]),
        },
        "preprocessing": {
            "bandpass_hz": [8.0, 32.0],
            "epoch_tmin_seconds": 0.0,
            "epoch_tmax_seconds": 3.996,
            "resample_hz": None,
            "baseline": None,
        },
        "representation": {
            "covariance_estimator": "oas",
            "covariance_extra_regularization": "none",
        },
    }
    frozen_path = v1 / "frozen_config.yaml"
    frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    metadata = _balanced_metadata("0train")
    metadata_path = v1 / "whole_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    prepared_metadata = _balanced_metadata("0train", prepared=True)
    prepared_metadata_path = v1 / "prepared_metadata.csv"
    prepared_metadata.to_csv(prepared_metadata_path, index=False)
    prepared_path = v1 / "prepared_epochs.npz"
    prepared_path.write_bytes(b"not-used-on-exact-reuse")
    dataset_metadata = v1 / "dataset_metadata.json"
    dataset_metadata.write_text("{}\n", encoding="utf-8")
    covariance_summary = v1 / "covariance_summary.json"
    covariance_summary.write_text("{}\n", encoding="utf-8")
    covariance_sanity = v1 / "covariance_sanity.csv"
    covariance_sanity.write_text("synthetic\n", encoding="utf-8")

    whole = np.broadcast_to(
        np.eye(22, dtype=np.float64), (2592, 22, 22)
    ).copy()
    covariance_path = v1 / "covariances.npz"
    np.savez(
        covariance_path,
        whole=whole,
        window5=np.empty((0, 22, 22), dtype=np.float64),
        channel_names=np.asarray(config["dataset"]["eeg_channels"]),
    )
    config["v1_discovery_inputs"]["paths"] = {
        "frozen_config": "v1/frozen_config.yaml",
        "dataset_metadata": "v1/dataset_metadata.json",
        "covariance_summary": "v1/covariance_summary.json",
        "covariance_sanity": "v1/covariance_sanity.csv",
        "covariances": "v1/covariances.npz",
        "whole_metadata": "v1/whole_metadata.csv",
        "prepared_epochs": "v1/prepared_epochs.npz",
        "prepared_metadata": "v1/prepared_metadata.csv",
    }
    config["v1_discovery_inputs"]["hashes"] = {
        "frozen_config_sha256": _hash(frozen_path),
        "dataset_metadata_sha256": _hash(dataset_metadata),
        "covariance_summary_sha256": _hash(covariance_summary),
        "covariance_sanity_sha256": _hash(covariance_sanity),
        "covariances_file_sha256": _hash(covariance_path),
        "whole_array_content_sha256": sha256_array(whole),
        "whole_metadata_sha256": _hash(metadata_path),
        "prepared_epochs_sha256": _hash(prepared_path),
        "prepared_metadata_sha256": _hash(prepared_metadata_path),
    }

    raw_dir = root / "raw"
    raw_dir.mkdir()
    raw_records = []
    ordered_records = []
    for subject in range(1, 10):
        raw = raw_dir / f"A{subject:02d}E.mat"
        raw.write_bytes(f"synthetic-subject-{subject}".encode("ascii"))
        digest = _hash(raw)
        raw_records.append(
            {
                "subject": subject,
                "path": f"raw/{raw.name}",
                "bytes": raw.stat().st_size,
                "sha256": digest,
            }
        )
        ordered_records.append(
            {"filename": raw.name, "size_bytes": raw.stat().st_size, "sha256": digest}
        )
    config["confirmatory_inputs"].update(
        {
            "raw_moabb_dir": "raw",
            "raw_files": raw_records,
            "ordered_manifest_sha256": hashlib.sha256(
                data_module.canonical_json_bytes(ordered_records)
            ).hexdigest(),
            "total_bytes": sum(record["bytes"] for record in raw_records),
            "cache_covariances": "cache/conditional/confirmatory_whole_covariances.npz",
            "cache_metadata": "cache/conditional/confirmatory_whole_metadata.csv",
            "unlock_filename": "outputs/conditional/confirmatory_unlock.json",
        }
    )
    config_path = root / "configs/conditional.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, config_path, config


@pytest.fixture(scope="module")
def frozen_config() -> dict[str, object]:
    return yaml.safe_load(
        (ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_exact_counts_and_split_contract(frozen_config: dict[str, object]) -> None:
    metadata = _balanced_metadata("0train")
    validated = validate_session_metadata(metadata, frozen_config, "0train")
    assert len(validated) == 2592
    for subject in range(1, 10):
        assert len(subject_split_positions(validated, frozen_config, subject, "A")) == 144
        assert len(subject_split_positions(validated, frozen_config, subject, "B")) == 144
        assert len(subject_split_positions(validated, frozen_config, subject, "F")) == 288


def test_metadata_rejects_run_class_count_failure(
    frozen_config: dict[str, object],
) -> None:
    metadata = _balanced_metadata("0train")
    metadata.loc[0, "class_label"] = "right_hand"
    with pytest.raises(DataContractError, match="class counts"):
        validate_session_metadata(metadata, frozen_config, "0train")


def test_whole_covariance_hard_gate_is_strict(
    frozen_config: dict[str, object],
) -> None:
    channels = np.asarray(frozen_config["dataset"]["eeg_channels"])
    whole = np.broadcast_to(np.eye(22), (2592, 22, 22)).copy()
    checks = validate_whole_covariances(
        whole,
        channels,
        frozen_config,
        expected_content_sha256=sha256_array(whole),
    )
    assert checks["minimum_eigenvalue"] == 1.0
    whole[0, 0, 0] = 0.0
    with pytest.raises(DataContractError, match="strict-SPD"):
        validate_whole_covariances(
            whole, channels, frozen_config, expected_content_sha256=None
        )


def test_discovery_exact_reuse_is_read_only_and_whole_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config_path, _ = _synthetic_repository(tmp_path)
    resolved_values: list[str] = []
    original_resolver = data_module._resolve_repo_path

    def audited_resolver(
        repo_root: Path, value: str | Path, *, label: str
    ) -> Path:
        resolved_values.append(str(value))
        if Path(value).name.endswith("E.mat"):
            raise AssertionError("discovery attempted to resolve confirmatory raw data")
        return original_resolver(repo_root, value, label=label)

    monkeypatch.setattr(data_module, "_resolve_repo_path", audited_resolver)
    result = load_discovery_whole(config_path, root)
    assert result.session == "0train"
    assert result.covariances.shape == (2592, 22, 22)
    assert result.covariances.dtype == np.float64
    assert not result.covariances.flags.writeable
    assert result.provenance["source"] == "exact_v1_read_only_reuse"
    assert result.provenance["window5_accessed"] is False
    assert not (root / "cache/conditional").exists()
    assert not any(Path(value).name.endswith("E.mat") for value in resolved_values)


def test_covariance_only_failure_routes_to_prepared_fallback_without_moabb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config_path, _ = _synthetic_repository(tmp_path)
    covariance = root / "v1/covariances.npz"
    covariance.write_bytes(b"covariance-identity-failure")
    called: list[str] = []
    sentinel = ConditionalWholeData(
        session="0train",
        covariances=np.empty((0, 0, 0)),
        metadata=pd.DataFrame(),
        channel_names=np.empty(0),
        provenance={"source": "synthetic-fallback"},
    )

    def fallback(*args: object, **kwargs: object) -> ConditionalWholeData:
        called.append("prepared-only")
        return sentinel

    monkeypatch.setattr(data_module, "_fallback_discovery_from_prepared", fallback)
    assert load_discovery_whole(config_path, root) is sentinel
    assert called == ["prepared-only"]


def test_post_unlock_confirmatory_preparer_writes_whole_only_synthetic_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config_path, config = _synthetic_repository(tmp_path)
    metadata = _balanced_metadata("1test")
    labels = metadata["class_label"].astype(str).to_numpy()
    eeg = np.broadcast_to(
        np.zeros((1, 22, 1000), dtype=np.float32), (2592, 22, 1000)
    )
    prepared = data_module._PreparedEpochs(
        eeg=eeg,
        labels=labels,
        metadata=metadata,
        channel_names=np.asarray(config["dataset"]["eeg_channels"]),
        sampling_frequency_hz=250.0,
        observed_source_times=(2.0, 5.996),
    )
    unlock = {
        "manifest_sha256": "b" * 64,
        "locked_head": "c" * 40,
    }
    monkeypatch.setattr(
        data_module, "validate_confirmatory_unlock", lambda *args, **kwargs: unlock
    )
    monkeypatch.setattr(
        data_module,
        "_load_confirmatory_epochs_from_moabb",
        lambda *args, **kwargs: prepared,
    )
    whole = np.broadcast_to(np.eye(22), (2592, 22, 22)).copy()
    monkeypatch.setattr(data_module, "_build_whole_oas", lambda *args, **kwargs: whole)

    result = prepare_confirmatory_whole(config_path, root)
    assert result.session == "1test"
    assert result.provenance["window5_computed"] is False
    cache = root / "cache/conditional/confirmatory_whole_covariances.npz"
    with np.load(cache, allow_pickle=False) as archive:
        assert archive.files == ["whole", "channel_names"]
        assert archive["whole"].shape == (2592, 22, 22)


def test_config_load_verifies_frozen_protocol_hash(tmp_path: Path) -> None:
    root, config_path, _ = _synthetic_repository(tmp_path)
    config, observed_path, config_hash, protocol_hash = load_conditional_config(
        config_path, root
    )
    assert observed_path == config_path
    assert len(config_hash) == len(protocol_hash) == 64
    assert config["protocol"]["confirmatory_designation"] == "STRICT_CONFIRMATORY"
