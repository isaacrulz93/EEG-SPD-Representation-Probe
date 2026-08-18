"""Source selection, direct-parser, preprocessing, and safe-deletion gates."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

import src.stieger2021_streaming_preprocessing_v0 as streaming
from src.stieger2021_multiclass_confirmation_v0 import load_config


ROOT = Path(__file__).resolve().parents[1]


def _article() -> dict:
    files = []
    file_id = 1000
    for subject in range(1, 63):
        for session in (2, 3):
            files.append(
                {
                    "id": file_id,
                    "name": f"S{subject}_Session_{session}.mat",
                    "download_url": f"https://example.test/{subject}/{session}",
                    "size": subject * 100 + session,
                    "computed_md5": hashlib.md5(f"{subject}/{session}".encode()).hexdigest(),  # nosec B324 test fixture
                }
            )
            file_id += 1
    return {"id": 13123148, "doi": "10.6084/m9.figshare.13123148.v1", "files": files}


def test_official_selector_requires_exact_124_pairs() -> None:
    selected = streaming.select_source_files(_article())
    assert len(selected) == 124
    assert [(selected[0].subject, selected[0].session), (selected[-1].subject, selected[-1].session)] == [(1, 2), (62, 3)]
    broken = _article(); broken["files"].pop()
    with pytest.raises(streaming.StiegerDataContractError, match="missing"):
        streaming.select_source_files(broken)


def test_source_manifest_is_order_independent_and_hash_complete() -> None:
    article = _article()
    first = streaming.canonical_source_manifest(streaming.select_source_files(article), article)
    article["files"] = list(reversed(article["files"]))
    second = streaming.canonical_source_manifest(streaming.select_source_files(article), article)
    assert first == second
    assert len(first["canonical_sha256"]) == 64


def test_channel_normalization_and_primary_order() -> None:
    config, _ = load_config(ROOT, verify_protocol=False)
    assert streaming.normalize_channel_name("CPZ") == "CPz"
    assert streaming.normalize_channel_name("cz") == "Cz"
    assert len(config["channels"]["primary_order"]) == 20
    assert config["channels"]["primary_order"][9] == "Cz"


def test_no_bad_channels_interpolation_is_identity() -> None:
    matrix = streaming.interpolation_matrix(["C3", "Cz", "C4"], [])
    np.testing.assert_array_equal(matrix, np.eye(3))


def test_filter_resample_and_exact_epoch_lengths() -> None:
    rng = np.random.default_rng(9)
    trial = rng.normal(size=(20, 6000))
    resampled = streaming.filter_resample_trial(trial, 1000.0, 100.0)
    assert resampled.shape == (20, 600)
    primary = streaming.crop_epoch(resampled, -2.0, [0.5, 2.0])
    pretarget = streaming.crop_epoch(resampled, -2.0, [-1.0, 0.0])
    assert primary.shape == (20, 150)
    assert pretarget.shape == (20, 100)


def test_oas_covariance_strict_spd_symmetric() -> None:
    covariance = streaming.oas_covariance(np.random.default_rng(3).normal(size=(20, 150)))
    assert covariance.shape == (20, 20)
    np.testing.assert_array_equal(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance)[0] > 0


def test_raw_deletion_only_after_validator_passes(tmp_path: Path) -> None:
    raw = tmp_path / "raw.mat"; raw.write_bytes(b"raw")
    compact = tmp_path / "compact.bin"
    streaming.safe_copy_then_validate_delete(raw, compact, lambda path: path.read_bytes() == b"raw")
    assert not raw.exists() and compact.exists()
    failed = tmp_path / "failed.mat"; failed.write_bytes(b"retain")
    with pytest.raises(RuntimeError):
        streaming.safe_copy_then_validate_delete(failed, tmp_path / "bad.bin", lambda _: (_ for _ in ()).throw(RuntimeError("bad")))
    assert failed.exists()


def test_outcome_fields_are_sealed_and_not_preprocessing_inputs() -> None:
    source = inspect.getsource(streaming.parse_and_preprocess_mat)
    assert "sealed_rows" in source
    for name in ("result", "forcedresult", "targethitnumber", "performance"):
        assert name in streaming.SEALED_FIELDS
    inclusion_block = source.split("if task !=", 1)[1].split("all_task3_count", 1)[0]
    assert all(name not in inclusion_block for name in streaming.SEALED_FIELDS)


def test_exact_time_vector_compact_contract_is_literal() -> None:
    source = inspect.getsource(streaming.parse_and_preprocess_mat)
    assert "unique_time_vector_offsets" in source
    assert "unique_time_vector_values" in source
    assert "all_time_vector_index" in source


def test_bad_primary_channel_limit_frozen() -> None:
    config, _ = load_config(ROOT, verify_protocol=False)
    assert config["channels"]["maximum_bad_primary_channels"] == 4
    assert config["preprocessing"]["include_if_artifact_equals"] == 0
    assert config["dataset"]["minimum_trials_per_class_session"] == 25


def test_noisechan_reads_official_nested_chaninfo_path() -> None:
    bci = {"chaninfo": {"noisechan": np.asarray([1, 3])}}
    assert streaming._noise_indices(bci, 4) == [0, 2]
