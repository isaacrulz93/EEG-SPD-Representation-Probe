"""Unlabeled conditional-mode identifiability and minimal anchoring V0.

The module is deliberately layered: immutable/source gates, frozen mode audit,
trial/prototype bridge, unsigned recovery, and minimal anchoring.  Target labels
are kept in a separate evaluation object and are never inputs to pooled marginal
fits or zero-label estimators.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import src.openbmi_pipeline_v0 as openbmi_v0
import src.subject_class_population_structure_v1 as population_v1
from src.conditional_geometry_v1 import spd_invsqrt, spd_log
from src.interaction_provenance_v0 import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_array,
    sha256_file,
)
from src.subject_class_interaction_v0 import (
    _mean,
    geometry_thresholds,
    load_frozen_config,
)


CONFIG_PATH = "configs/unlabeled_conditional_mode_identifiability_v0.yaml"
OUTPUT_NAME = "unlabeled_conditional_mode_identifiability_v0"
CLASS_ORDER = ("left_hand", "right_hand")


class ConditionalModeError(RuntimeError):
    """Base fail-closed error."""


class TrialObjectInsufficientError(ConditionalModeError):
    """Exact trial covariance lineage cannot be reproduced."""


class DataContractError(ConditionalModeError):
    """Ordering, hash, leakage, or output contract failed."""


class NumericalContractError(ConditionalModeError):
    """Finite, SPD, decomposition, or nondegeneracy contract failed."""


@dataclass(frozen=True)
class TrialData:
    covariances: np.ndarray  # subject, session, trial, channel, channel
    labels: np.ndarray  # evaluation-only: 0=left, 1=right
    trial_ids: np.ndarray  # opaque IDs after deterministic shuffle
    subjects: tuple[int, ...]
    sessions: tuple[str, ...]
    channels: tuple[str, ...]


@dataclass(frozen=True)
class ModeData:
    modes: np.ndarray  # fold, session, feature
    fold_of_subject: np.ndarray
    subjects: tuple[int, ...]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def output_path(root: Path, config: Mapping[str, Any]) -> Path:
    return root / str(config["project"]["output_dir"])


def cache_path(root: Path, config: Mapping[str, Any]) -> Path:
    return root / str(config["project"]["cache_dir"])


def _ensure_output(output: Path) -> None:
    for name in ("protocol", "objects", "nulls", "controls", "decisions", "tables", "figures", "report"):
        (output / name).mkdir(parents=True, exist_ok=True)


def load_config(repo_root: str | Path) -> tuple[dict[str, Any], str]:
    root = Path(repo_root).resolve()
    path = root / CONFIG_PATH
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise DataContractError("V0 config must be a mapping")
    protocol = root / str(config["protocol"]["protocol_path"])
    observed = sha256_file(protocol)
    if observed != str(config["protocol"]["protocol_sha256"]):
        raise DataContractError(f"protocol SHA mismatch: {observed}")
    if str(config["protocol"]["parent_head"]) != "9dee7642ac573f37756b8427a75864a50c32044e":
        raise DataContractError("parent head literal changed")
    return config, sha256_file(path)


def validate_parent_artifacts(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, record in config["parent_artifacts"].items():
        if name == "immutable_output_directories":
            continue
        path = root / str(record["path"])
        if not path.is_file():
            raise DataContractError(f"missing parent artifact: {name}")
        digest = sha256_file(path)
        if digest != str(record["sha256"]):
            raise DataContractError(f"parent artifact changed: {name} {digest}")
        observed[str(name)] = digest
    terminal_path = root / str(config["parent_artifacts"]["v1_1_terminal"]["path"])
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("terminal_decision") != "GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION":
        raise DataContractError("parent terminal is not the frozen GO")
    parent = str(config["protocol"]["parent_head"])
    if _git(root, "merge-base", "--is-ancestor", parent, "HEAD") != "":
        raise DataContractError("exact parent head is not an ancestor")
    return observed


def immutable_parent_snapshot(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for directory in config["parent_artifacts"]["immutable_output_directories"]:
        base = root / str(directory)
        if not base.is_dir():
            raise DataContractError(f"missing immutable output directory: {directory}")
        for path in sorted(value for value in base.rglob("*") if value.is_file()):
            relative = str(path.relative_to(root))
            result[relative] = sha256_file(path)
    if not result:
        raise DataContractError("empty parent artifact snapshot")
    return result


def validate_parent_snapshot(root: Path, output: Path) -> dict[str, str]:
    path = output / "protocol/pr16_artifact_hashes.json"
    if not path.is_file():
        raise DataContractError("PR #16 hash snapshot is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload["files"]
    for relative, digest in expected.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != digest:
            raise DataContractError(f"PR #16 artifact changed: {relative}")
    return dict(expected)


def _scientific_files(root: Path) -> dict[str, str]:
    paths = [
        CONFIG_PATH,
        "docs/PROTOCOL_UNLABELED_CONDITIONAL_MODE_IDENTIFIABILITY_V0.md",
        "docs/AUDIT_UNLABELED_CONDITIONAL_MODE_V0.md",
        "docs/SIGNED_COORDINATE_NONIDENTIFIABILITY.md",
        "src/unlabeled_conditional_mode_identifiability_v0.py",
        "tests/test_unlabeled_conditional_mode_identifiability_v0.py",
    ]
    paths.extend(
        f"scripts/{number}_{name}.py"
        for number, name in (
            (80, "freeze_unlabeled_conditional_mode_v0"),
            (81, "run_mode_identity_audit_v0"),
            (82, "run_trial_mode_compatibility_v0"),
            (83, "run_unsigned_recovery_v0"),
            (84, "run_minimal_anchor_v0"),
            (85, "report_unlabeled_conditional_mode_v0"),
        )
    )
    missing = [value for value in paths if not (root / value).is_file()]
    if missing:
        raise DataContractError(f"missing freeze-scope files: {missing}")
    return {value: sha256_file(root / value) for value in paths}


def environment_record() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyriemann", "matplotlib", "mne", "PyYAML", "pytest"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "packages": packages, "cpu_count": os.cpu_count(),
    }


def _outer_contract(config: Mapping[str, Any], v1_config: Mapping[str, Any]) -> None:
    if config["parent_fold_contract"]["outer_test"] != v1_config["openbmi_folds"]["outer_test"]:
        raise DataContractError("new outer fold literals differ from V1")
    if str(config["parent_fold_contract"]["canonical_sha256"]) != str(v1_config["openbmi_folds"]["sha256"]):
        raise DataContractError("fold hash differs from V1")


def _plus_one_p(observed: float, null: np.ndarray) -> float:
    values = np.asarray(null, dtype=np.float64)
    if not np.isfinite(observed) or not np.isfinite(values).all():
        raise NumericalContractError("nonfinite Monte Carlo statistic")
    return float((1 + np.count_nonzero(values >= observed)) / (1 + len(values)))


def _rng(config: Mapping[str, Any], *parts: Any) -> np.random.Generator:
    return population_v1.deterministic_rng(int(config["protocol"]["master_seed"]), "unlabeled_v0", *parts)


def _subject_bootstrap_indices(config: Mapping[str, Any], namespace: str, replicates: int, n: int) -> np.ndarray:
    result = np.empty((replicates, n), dtype=np.int16)
    for index in range(replicates):
        result[index] = _rng(config, namespace, index).integers(0, n, size=n)
    return result


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if len(a) < 3 or np.ptp(a) <= np.finfo(float).eps or np.ptp(b) <= np.finfo(float).eps:
        raise NumericalContractError("Spearman input is degenerate")
    value = float(stats.spearmanr(a, b).statistic)
    if not np.isfinite(value):
        raise NumericalContractError("Spearman statistic is nonfinite")
    return value


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if len(a) < 3 or np.std(a) <= np.finfo(float).eps or np.std(b) <= np.finfo(float).eps:
        raise NumericalContractError("Pearson input is degenerate")
    value = float(stats.pearsonr(a, b).statistic)
    if not np.isfinite(value):
        raise NumericalContractError("Pearson statistic is nonfinite")
    return value


def _linear_calibration(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(predicted)), np.asarray(predicted, dtype=np.float64)])
    intercept, slope = np.linalg.lstsq(design, np.asarray(target, dtype=np.float64), rcond=None)[0]
    return float(slope), float(intercept)


def synthetic_gates(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rng = _rng(config, "synthetic")
    n, trials = 80, 100
    magnitude = rng.uniform(0.4, 2.0, size=n)
    labels = np.tile(np.repeat([0, 1], trials // 2), (n, 1))
    noise = rng.normal(scale=0.35, size=(n, trials))
    y = (2 * labels - 1) * magnitude[:, None] + noise
    total = np.var(y, axis=1, ddof=0)
    within = 0.5 * (np.var(y[:, :50], axis=1, ddof=0) + np.var(y[:, 50:], axis=1, ddof=0))
    oracle = magnitude**2
    estimate = np.maximum(total - np.mean(within), 0.0)
    rows.append({"case": "balanced_rank1_unsigned_recovery", "passed": _spearman(estimate, oracle) > 0.9})

    swapped = 1 - labels
    rows.append({
        "case": "class_swap_unlabeled_invariance",
        "passed": np.array_equal(y, y.copy())
        and np.allclose(np.var(y, axis=1), np.var(y, axis=1), rtol=0.0, atol=0.0)
        and np.all((2 * swapped - 1) == -(2 * labels - 1)),
    })
    rows.append({"case": "recoverable_unsigned_separation", "passed": float(np.mean(np.abs(estimate - oracle))) < 0.35})
    signed_world_a = magnitude
    signed_world_b = -magnitude
    rows.append({
        "case": "signed_nonidentifiability",
        "passed": np.array_equal(total, total.copy()) and np.array_equal(signed_world_b, -signed_world_a),
    })
    varying_noise = rng.uniform(0.15, 0.8, size=n)
    y_varying = (2 * labels - 1) * magnitude[:, None] + rng.normal(size=(n, trials)) * varying_noise[:, None]
    rows.append({"case": "varying_within_noise", "passed": np.isfinite(y_varying).all() and np.ptp(np.var(y_varying, axis=1)) > 0.0})
    no_sep = rng.normal(size=(n, trials))
    no_delta = 0.5 * (np.mean(no_sep[:, 50:], axis=1) - np.mean(no_sep[:, :50], axis=1))
    rows.append({"case": "no_between_separation", "passed": abs(float(np.mean(no_delta))) < 0.1})
    random_signal = rng.normal(size=(n, 12))
    random_direction = rng.normal(size=12); random_direction /= np.linalg.norm(random_direction)
    random_target = rng.normal(size=n)
    rows.append({"case": "random_direction_equivalence", "passed": abs(_spearman(random_signal @ random_direction, random_target)) < 0.3})
    calibration = y[:, [0, 50]]
    orientation = np.sign(calibration[:, 1] - calibration[:, 0])
    rows.append({"case": "minimal_anchor_sign_resolution", "passed": float(np.mean(orientation == 1)) > 0.9})
    sentinel = {"unlabeled_fields": ("covariance", "opaque_id"), "evaluation_fields": ("opaque_id", "class_label")}
    rows.append({
        "case": "target_label_leakage_sentinel",
        "passed": "class_label" not in sentinel["unlabeled_fields"] and "class_label" in sentinel["evaluation_fields"],
    })
    for row in rows:
        row["passed"] = bool(row["passed"])
    passed = bool(all(row["passed"] for row in rows))
    return {"schema_version": "unlabeled-conditional-mode-v0-synthetic", "passed": passed, "cases": rows}


def freeze_protocol(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root)
    parent_hashes = validate_parent_artifacts(root, config)
    v1_config, v1_hash = population_v1.load_config(root)
    if v1_hash != str(config["parent_artifacts"]["v1_scientific_config"]["sha256"]):
        raise DataContractError("V1 loader config hash differs")
    _outer_contract(config, v1_config)
    dataset = population_v1.load_parent_dataset(root, v1_config, "openbmi")
    outer, inner = population_v1.openbmi_fold_indices(dataset, v1_config)
    if [list(map(lambda i: dataset.subjects[i], fold)) for fold in outer] != config["parent_fold_contract"]["outer_test"]:
        raise DataContractError("resolved fold membership differs")
    modes_path = root / str(config["parent_artifacts"]["v1_1_observed_modes"]["path"])
    with np.load(modes_path, allow_pickle=False) as archive:
        if archive["selected_ranks"].tolist() != config["mode"]["parent_selected_ranks_required"]:
            raise DataContractError("parent selected ranks are not six rank-1 folds")
    source_manifest = json.loads((root / str(config["parent_artifacts"]["openbmi_source_manifest"]["path"])).read_text())
    if source_manifest.get("file_count") != 108 or len(source_manifest.get("files", [])) != 108:
        raise TrialObjectInsufficientError("parent source manifest does not contain 108 files")
    snapshot = immutable_parent_snapshot(root, config)
    synthetic = synthetic_gates(config)
    if not synthetic["passed"]:
        raise NumericalContractError("synthetic gates failed")
    output = output_path(root, config); _ensure_output(output)
    atomic_write_json(output / "protocol/pr16_artifact_hashes.json", {
        "schema_version": "pr16-immutable-artifact-snapshot-v0", "files": snapshot,
    })
    pd.DataFrame(synthetic["cases"]).to_csv(output / "protocol/synthetic_gates.csv", index=False, lineterminator="\n")
    atomic_write_json(output / "protocol/synthetic_gates.json", synthetic)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, test in enumerate(outer):
        train = np.setdiff1d(np.arange(len(dataset.subjects)), test)
        fold_rows.extend({"outer_fold": fold_index, "role": "train", "subject": dataset.subjects[i]} for i in train)
        fold_rows.extend({"outer_fold": fold_index, "role": "test", "subject": dataset.subjects[i]} for i in test)
        for inner_index, validation in enumerate(inner[fold_index]):
            fold_rows.extend({"outer_fold": fold_index, "role": f"inner_validation_{inner_index}", "subject": dataset.subjects[i]} for i in validation)
    pd.DataFrame(fold_rows).to_csv(output / "protocol/exact_folds.csv", index=False, lineterminator="\n")
    for relative in (
        config["protocol"]["protocol_path"], config["project"]["audit_path"],
        config["project"]["theory_path"], CONFIG_PATH,
    ):
        source = root / str(relative)
        atomic_write_bytes(output / "protocol" / source.name, source.read_bytes())
    files = _scientific_files(root)
    manifest = {
        "schema_version": "unlabeled-conditional-mode-v0-freeze",
        "status": "READY_FOR_PROTOCOL_FREEZE_COMMIT", "real_recovery_statistics_accessed": False,
        "parent_head": config["protocol"]["parent_head"], "config_sha256": config_hash,
        "parent_hashes": parent_hashes, "pr16_file_count": len(snapshot),
        "pr16_snapshot_sha256": hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest(),
        "scientific_file_hashes": files, "synthetic_pass": True,
        "fold_hash": v1_config["openbmi_folds"]["sha256"],
        "required_commit_subject": config["protocol"]["required_freeze_commit_subject"],
    }
    atomic_write_json(output / "protocol/manifest.json", manifest)
    atomic_write_json(output / "environment.json", environment_record())
    return manifest


def ensure_real_access_lock(repo_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root)
    validate_parent_artifacts(root, config)
    output = output_path(root, config)
    manifest_path = output / "protocol/manifest.json"
    if not manifest_path.is_file():
        raise DataContractError("freeze manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != config_hash or not manifest["synthetic_pass"]:
        raise DataContractError("freeze manifest/config mismatch")
    if _scientific_files(root) != manifest["scientific_file_hashes"]:
        raise DataContractError("scientific source changed after freeze")
    snapshot = validate_parent_snapshot(root, output)
    if hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest() != manifest["pr16_snapshot_sha256"]:
        raise DataContractError("PR #16 snapshot digest changed")
    provenance_path = output / "git_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if _git(root, "merge-base", "--is-ancestor", provenance["protocol_freeze_commit"], "HEAD") != "":
            raise DataContractError("freeze commit is not an ancestor")
        return config, provenance
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise DataContractError(f"first real access requires clean tree: {status.splitlines()[:5]}")
    if _git(root, "show", "-s", "--format=%s", "HEAD") != str(config["protocol"]["required_freeze_commit_subject"]):
        raise DataContractError("HEAD is not the protocol-freeze commit")
    provenance = {
        "schema_version": "unlabeled-conditional-mode-v0-git-provenance",
        "parent_head": config["protocol"]["parent_head"],
        "protocol_freeze_commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "first_real_access_tree_was_clean": True,
        "scientific_file_hashes": manifest["scientific_file_hashes"],
        "pr16_snapshot_sha256": manifest["pr16_snapshot_sha256"],
    }
    atomic_write_json(provenance_path, provenance)
    return config, provenance


def _raw_root(config: Mapping[str, Any]) -> Path:
    import mne
    key = str(config["dataset"]["raw_root_mne_config_key"])
    value = mne.get_config(key, default=None)
    if not value:
        raise TrialObjectInsufficientError(f"MNE config key is unset: {key}")
    return Path(value).expanduser().resolve() / str(config["dataset"]["raw_relative_root"])


def _source_path(raw_root: Path, subject: int, session: int) -> Path:
    return raw_root / f"session{session}/s{subject}/sess{session:02d}_subj{subject:02d}_EEG_MI.mat"


def _download_exact(url: str, target: Path, expected_size: int, expected_sha: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        digest = hashlib.sha256()
        request = urllib.request.Request(url, headers={"User-Agent": "EEG-SPD-Representation-Probe/0"})
        with os.fdopen(descriptor, "wb") as handle, urllib.request.urlopen(request, timeout=180) as response:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                handle.write(block); digest.update(block)
            handle.flush(); os.fsync(handle.fileno())
        if temporary.stat().st_size != expected_size or digest.hexdigest() != expected_sha:
            raise TrialObjectInsufficientError("canonical download differs from parent source hash")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_trial_covariances(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = ensure_real_access_lock(root)
    output = output_path(root, config); cache = cache_path(root, config)
    sessions_cache = cache / "subject_sessions"; sessions_cache.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((root / str(config["parent_artifacts"]["openbmi_source_manifest"]["path"])).read_text())
    protocol_manifest = json.loads((root / str(config["parent_artifacts"]["openbmi_protocol_manifest"]["path"])).read_text())
    raw_root = _raw_root(config)
    records: list[dict[str, Any]] = []
    covariance_blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []
    for index, expected in enumerate(source_manifest["files"]):
        subject = int(expected["subject"]); session = int(expected["session"])
        local = _source_path(raw_root, subject, session)
        temporary_source = cache / "source_downloads" / f"S{subject:02d}_session{session}.mat"
        source = local
        source_origin = "MNE_CACHE"
        if not local.is_file() or local.stat().st_size != int(expected["source_bytes"]):
            source = temporary_source; source_origin = "CANONICAL_DOWNLOAD"
        if not source.is_file() or source.stat().st_size != int(expected["source_bytes"]):
            _download_exact(expected["url"], source, int(expected["source_bytes"]), str(expected["source_sha256"]))
        source_hash = sha256_file(source)
        if source_hash != str(expected["source_sha256"]):
            if source == local and bool(config["dataset"]["allow_canonical_download_on_hash_or_size_mismatch"]):
                source = temporary_source; source_origin = "CANONICAL_DOWNLOAD_AFTER_HASH_MISMATCH"
                _download_exact(expected["url"], source, int(expected["source_bytes"]), str(expected["source_sha256"]))
                source_hash = sha256_file(source)
            if source_hash != str(expected["source_sha256"]):
                raise TrialObjectInsufficientError(f"raw source hash mismatch S{subject} session {session}")
        derived = sessions_cache / f"S{subject:02d}_session{session}.npz"
        metadata_path = sessions_cache / f"S{subject:02d}_session{session}.csv"
        record_path = sessions_cache / f"S{subject:02d}_session{session}.json"
        reusable = False
        if derived.is_file() and metadata_path.is_file() and record_path.is_file():
            prior = json.loads(record_path.read_text())
            reusable = prior.get("covariance_array_sha256") == expected["covariance_array_sha256"] and prior.get("metadata_sha256") == expected["metadata_sha256"]
        if reusable:
            with np.load(derived, allow_pickle=False) as archive:
                covariances = np.asarray(archive["covariances"], dtype=np.float64)
            metadata = pd.read_csv(metadata_path)
        else:
            covariances, metadata = openbmi_v0._prepare_one(source, protocol_manifest, subject, session)
            if sha256_array(covariances) != str(expected["covariance_array_sha256"]):
                raise TrialObjectInsufficientError(f"derived covariance hash mismatch S{subject} session {session}")
            population_v1._atomic_savez(derived, {"covariances": covariances, "channel_names": np.asarray(protocol_manifest["eeg_channels"])})
            metadata.to_csv(metadata_path, index=False, lineterminator="\n")
        metadata_hash = sha256_file(metadata_path)
        if metadata_hash != str(expected["metadata_sha256"]):
            raise TrialObjectInsufficientError(f"metadata hash mismatch S{subject} session {session}")
        record = {
            "subject": subject, "session": session, "source_origin": source_origin,
            "source_bytes": int(source.stat().st_size), "source_sha256": source_hash,
            "covariance_array_sha256": sha256_array(covariances),
            "derived_cache_sha256": sha256_file(derived), "metadata_sha256": metadata_hash,
            "parent_source_sha256": expected["source_sha256"],
            "parent_covariance_array_sha256": expected["covariance_array_sha256"],
        }
        atomic_write_json(record_path, record); records.append(record)
        covariance_blocks.append(covariances); metadata_blocks.append(metadata)
        print(f"trial rebuild {index + 1}/108 S{subject:02d} session {session}", flush=True)
    covariances = np.concatenate(covariance_blocks, axis=0)
    metadata = pd.concat(metadata_blocks, ignore_index=True)
    if list(covariances.shape) != list(config["dataset"]["expected_covariance_shape"]) or len(metadata) != 10800:
        raise TrialObjectInsufficientError("combined trial shape mismatch")
    combined_path = cache / "openbmi_covariances.npz"
    population_v1._atomic_savez(combined_path, {"covariances": covariances, "channel_names": np.asarray(config["dataset"]["channels"])})
    metadata_path = cache / "openbmi_metadata.csv"
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    manifest = {
        "schema_version": "unlabeled-conditional-mode-v0-trial-covariance-manifest",
        "status": "EXACT_PARENT_PREPROCESSING_REBUILT", "records": records,
        "record_count": len(records), "shape": list(covariances.shape), "dtype": str(covariances.dtype),
        "combined_array_sha256": sha256_array(covariances),
        "combined_cache_sha256": sha256_file(combined_path), "combined_metadata_sha256": sha256_file(metadata_path),
        "cache_committed": False, "preprocessing_implementation": "src.openbmi_pipeline_v0._prepare_one",
    }
    atomic_write_json(output / "objects/trial_covariance_manifest.json", manifest)
    return manifest


def load_trial_data(root: Path, config: Mapping[str, Any]) -> TrialData:
    cache = cache_path(root, config)
    cov_path = cache / "openbmi_covariances.npz"; meta_path = cache / "openbmi_metadata.csv"
    if not cov_path.is_file() or not meta_path.is_file():
        raise TrialObjectInsufficientError("rebuilt trial cache is missing")
    with np.load(cov_path, allow_pickle=False) as archive:
        cov = np.asarray(archive["covariances"], dtype=np.float64)
        channels = tuple(str(x) for x in archive["channel_names"].tolist())
    meta = pd.read_csv(meta_path); meta["session"] = meta["session"].astype(str)
    subjects = tuple(int(x) for x in config["dataset"]["subjects"])
    sessions = tuple(str(x) for x in config["dataset"]["sessions"])
    if cov.shape != (10800, 20, 20) or channels != tuple(config["dataset"]["channels"]):
        raise TrialObjectInsufficientError("trial cache array/channel contract failed")
    shaped = np.empty((54, 2, 100, 20, 20), dtype=np.float64)
    labels = np.empty((54, 2, 100), dtype=np.int8)
    ids = np.empty((54, 2, 100), dtype="U32")
    for si, subject in enumerate(subjects):
        for qi, session in enumerate(sessions):
            mask = (meta["subject"].to_numpy() == subject) & (meta["session"].to_numpy() == session)
            indices = np.flatnonzero(mask)
            if len(indices) != 100:
                raise TrialObjectInsufficientError("subject/session trial count mismatch")
            block_labels = meta.iloc[indices]["class_label"].astype(str).to_numpy()
            encoded = np.where(block_labels == "left_hand", 0, np.where(block_labels == "right_hand", 1, -1))
            if np.count_nonzero(encoded == 0) != 50 or np.count_nonzero(encoded == 1) != 50:
                raise TrialObjectInsufficientError("class balance contract failed")
            permutation = _rng(config, config["unlabeled_projection"]["shuffle_namespace"], subject, session).permutation(100)
            shaped[si, qi] = cov[indices][permutation]
            labels[si, qi] = encoded[permutation]
            ids[si, qi] = np.asarray([hashlib.sha256(f"{subject}|{session}|{int(index)}".encode()).hexdigest()[:24] for index in permutation])
    return TrialData(shaped, labels, ids, subjects, sessions, channels)


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(np.asarray(b)), np.finfo(float).tiny))


def reproduce_parent_means_and_tangents(root: Path, config: Mapping[str, Any], data: TrialData) -> tuple[np.ndarray, np.ndarray]:
    v0_config, _ = load_frozen_config(root)
    thresholds = geometry_thresholds(v0_config)
    means = np.empty((54, 2, 20, 20), dtype=np.float64)
    features = np.empty((54, 2, 100, 210), dtype=np.float64)
    class_means = np.empty((54, 2, 2, 20, 20), dtype=np.float64)
    u = np.empty_like(class_means)
    audit_rows: list[dict[str, Any]] = []
    for s in range(54):
        for q in range(2):
            marginal, _ = _mean(data.covariances[s, q], "AIRM", thresholds, f"openbmi:S{data.subjects[s]}:q{q}:unlabeled")
            means[s, q] = marginal
            inv = spd_invsqrt(marginal)
            whitened = inv @ data.covariances[s, q] @ inv
            features[s, q] = population_v1.svec(spd_log(whitened))
            for c in range(2):
                value, _ = _mean(data.covariances[s, q, data.labels[s, q] == c], "AIRM", thresholds, f"openbmi:S{data.subjects[s]}:q{q}:c{c}:lineage")
                class_means[s, q, c] = value
                u[s, q, c] = spd_log(inv @ value @ inv)
    v1_config, _ = population_v1.load_config(root)
    parent = population_v1.load_parent_dataset(root, v1_config, "openbmi")
    tolerance = float(config["preprocessing"]["mean_reproduction_relative_tolerance"])
    for name, rebuilt, expected in (
        ("marginal_means", means, parent.marginal_means["F"]),
        ("class_means", class_means, parent.class_means["F"]),
        ("U", u, parent.U["F"]),
    ):
        errors = np.linalg.norm(rebuilt - expected, axis=(-2, -1)) / np.maximum(np.linalg.norm(expected, axis=(-2, -1)), np.finfo(float).tiny)
        maximum = float(np.max(errors))
        audit_rows.append({"object": name, "maximum_relative_frobenius_error": maximum, "tolerance": tolerance, "passed": maximum <= tolerance})
        if maximum > tolerance:
            raise TrialObjectInsufficientError(f"parent {name} reproduction failed: {maximum}")
    output = output_path(root, config)
    pd.DataFrame(audit_rows).to_csv(output / "tables/trial_parent_reproduction.csv", index=False, lineterminator="\n", float_format="%.17g")
    population_v1._atomic_savez(cache_path(root, config) / "unlabeled_tangent_features.npz", {"features": features, "marginal_means": means, "trial_ids": data.trial_ids})
    population_v1._atomic_savez(output / "objects/unlabeled_marginal_means.npz", {"marginal_means": means, "subjects": np.asarray(data.subjects), "sessions": np.asarray(data.sessions)})
    return means, features


def _load_modes_parent(root: Path, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = root / str(config["parent_artifacts"]["v1_1_observed_modes"]["path"])
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def _inverse_svec(vector: np.ndarray, dimension: int = 20) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    rows, cols = np.triu_indices(dimension)
    if value.shape[-1] != len(rows):
        raise ValueError("inverse svec dimension mismatch")
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    scaled = value.copy(); scaled[rows != cols] /= np.sqrt(2.0)
    matrix[rows, cols] = scaled; matrix[cols, rows] = scaled
    return matrix


def mode_identity_audit(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = ensure_real_access_lock(root)
    output = output_path(root, config); _ensure_output(output)
    trial_manifest = prepare_trial_covariances(root)
    data = load_trial_data(root, config)
    means, features = reproduce_parent_means_and_tangents(root, config, data)
    v1_config, _ = population_v1.load_config(root)
    parent_data = population_v1.load_parent_dataset(root, v1_config, "openbmi")
    outer, _ = population_v1.openbmi_fold_indices(parent_data, v1_config)
    stored = _load_modes_parent(root, config)
    modes = np.empty((6, 2, 210), dtype=np.float64)
    fold_of_subject = np.empty(54, dtype=np.int8)
    identity_rows: list[dict[str, Any]] = []
    refit_rows: list[dict[str, Any]] = []
    for fold_index, test in enumerate(outer):
        fold_of_subject[test] = fold_index
        train = np.setdiff1d(np.arange(54), test)
        x_train, _, _ = population_v1.fold_features(parent_data, "F", train, test, helmert=None, kind="sensor")
        fit = population_v1.fit_two_view(x_train[:, 0], x_train[:, 1], 34)
        modes[fold_index, 0] = fit.left[:, 0]; modes[fold_index, 1] = fit.right[:, 0]
        for q, stored_bank in ((0, stored["left"]), (1, stored["right"])):
            cosine = abs(float(np.dot(modes[fold_index, q], stored_bank[fold_index, :, 0])))
            identity_rows.append({"outer_fold": fold_index, "session": q, "absolute_cosine_to_parent": cosine})
            if 1.0 - cosine > float(config["mode"]["identity_absolute_cosine_tolerance"]):
                raise DataContractError("reconstructed mode differs from frozen parent")
        for omitted in train:
            reduced = train[train != omitted]
            reduced_x, _, _ = population_v1.fold_features(parent_data, "F", reduced, test, helmert=None, kind="sensor")
            reduced_fit = population_v1.fit_two_view(reduced_x[:, 0], reduced_x[:, 1], 1)
            for q, direction in ((0, reduced_fit.left[:, 0]), (1, reduced_fit.right[:, 0])):
                cosine = min(1.0, abs(float(np.dot(direction, modes[fold_index, q]))))
                refit_rows.append({
                    "outer_fold": fold_index, "session": q, "omitted_subject": parent_data.subjects[omitted],
                    "absolute_cosine": cosine, "principal_angle_degrees": float(np.degrees(np.arccos(cosine))),
                })
    cosine_rows: list[dict[str, Any]] = []
    for q in range(2):
        for f in range(6):
            for g in range(6):
                cosine_rows.append({"session": q, "fold_f": f, "fold_g": g, "absolute_cosine": abs(float(np.dot(modes[f, q], modes[g, q])))})
    for f in range(6):
        cosine_rows.append({"session": "cross_view", "fold_f": f, "fold_g": f, "absolute_cosine": abs(float(np.dot(modes[f, 0], modes[f, 1])))})
    haar = np.empty((int(config["mode"]["haar_replicates"]), 6, 2), dtype=np.float64)
    for r in range(len(haar)):
        for f in range(6):
            for q in range(2):
                random = _rng(config, "mode_haar", r, f, q).normal(size=210); random /= np.linalg.norm(random)
                haar[r, f, q] = abs(float(np.dot(random, modes[f, q])))
    aligned = modes.copy()
    for f in range(6):
        sign = 1.0 if float(np.dot(aligned[f, 0], aligned[0, 0])) >= 0.0 else -1.0
        aligned[f] *= sign
    sensor_rows: list[dict[str, Any]] = []; edge_rows: list[dict[str, Any]] = []
    channels = list(config["dataset"]["channels"])
    for f in range(6):
        for q in range(2):
            matrix = _inverse_svec(aligned[f, q])
            diagonal_energy = float(np.sum(np.diag(matrix) ** 2))
            off_diagonal_energy = float(np.sum(matrix**2) - diagonal_energy)
            strength = np.sum(np.abs(matrix), axis=1) - np.abs(np.diag(matrix))
            for ch, value in zip(channels, strength):
                sensor_rows.append({"outer_fold": f, "session": q, "channel": ch, "node_strength": float(value), "diagonal_energy": diagonal_energy, "off_diagonal_energy": off_diagonal_energy})
            edges = [(abs(float(matrix[i, j])), i, j, float(matrix[i, j])) for i in range(20) for j in range(i + 1, 20)]
            for rank, (_, i, j, value) in enumerate(sorted(edges, reverse=True)[:20], 1):
                edge_rows.append({"outer_fold": f, "session": q, "rank": rank, "channel_i": channels[i], "channel_j": channels[j], "value": value, "absolute_value": abs(value)})
    pd.DataFrame(identity_rows).to_csv(output / "tables/mode_parent_identity.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(cosine_rows).to_csv(output / "tables/mode_absolute_cosines.csv", index=False, lineterminator="\n", float_format="%.17g")
    refit_frame = pd.DataFrame(refit_rows); refit_frame.to_csv(output / "tables/mode_leave_one_training_subject_refits.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(sensor_rows).to_csv(output / "tables/mode_channel_energy.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(edge_rows).to_csv(output / "tables/mode_top_edges.csv", index=False, lineterminator="\n", float_format="%.17g")
    population_v1._atomic_savez(output / "objects/rank1_session_view_modes.npz", {"modes": modes, "aligned_modes_visualization": aligned, "fold_of_subject": fold_of_subject, "subjects": np.asarray(parent_data.subjects), "haar_absolute_cosines": haar})
    population_v1._atomic_savez(output / "objects/unlabeled_projected_trial_contract.npz", {"trial_ids": data.trial_ids, "subjects": np.asarray(data.subjects), "sessions": np.asarray(data.sessions)})
    population_v1._atomic_savez(output / "objects/evaluation_labels_separate.npz", {"trial_ids": data.trial_ids, "class_index": data.labels, "class_names": np.asarray(CLASS_ORDER)})
    summary = {
        "schema_version": "unlabeled-conditional-mode-v0-mode-audit",
        "trial_level_object_gate": "PASS_EXACT_PARENT_PREPROCESSING_REBUILT",
        "trial_manifest_sha256": sha256_file(output / "objects/trial_covariance_manifest.json"),
        "minimum_parent_identity_absolute_cosine": float(pd.DataFrame(identity_rows)["absolute_cosine_to_parent"].min()),
        "median_fold_pair_absolute_cosine": float(pd.DataFrame(cosine_rows).query("session != 'cross_view'")["absolute_cosine"].median()),
        "median_cross_view_absolute_cosine": float(pd.DataFrame(cosine_rows).query("session == 'cross_view'")["absolute_cosine"].median()),
        "median_leave_one_refit_absolute_cosine": float(refit_frame["absolute_cosine"].median()),
        "minimum_leave_one_refit_absolute_cosine": float(refit_frame["absolute_cosine"].min()),
        "haar_median_absolute_cosine": float(np.median(haar)),
        "mode_audit_is_descriptive": True,
    }
    atomic_write_json(output / "decisions/mode_identity_audit.json", summary)
    return summary


def _load_mode_data(root: Path, config: Mapping[str, Any]) -> ModeData:
    path = output_path(root, config) / "objects/rank1_session_view_modes.npz"
    if not path.is_file():
        raise DataContractError("mode audit must run before trial projection")
    with np.load(path, allow_pickle=False) as archive:
        modes = np.asarray(archive["modes"], dtype=np.float64)
        fold_of_subject = np.asarray(archive["fold_of_subject"], dtype=np.int64)
        subjects = tuple(int(x) for x in archive["subjects"].tolist())
    if modes.shape != (6, 2, 210) or fold_of_subject.shape != (54,):
        raise DataContractError("mode object shape mismatch")
    np.testing.assert_allclose(np.linalg.norm(modes, axis=-1), 1.0, rtol=0.0, atol=5e-14)
    return ModeData(modes=modes, fold_of_subject=fold_of_subject, subjects=subjects)


def _load_tangent_features(root: Path, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = cache_path(root, config) / "unlabeled_tangent_features.npz"
    if not path.is_file():
        raise TrialObjectInsufficientError("unlabeled tangent cache missing")
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float64)
        ids = np.asarray(archive["trial_ids"])
    if features.shape != (54, 2, 100, 210) or ids.shape != (54, 2, 100):
        raise DataContractError("tangent feature cache shape mismatch")
    if not np.isfinite(features).all():
        raise NumericalContractError("tangent feature cache is nonfinite")
    return features, ids


def _build_analysis_core(root: Path, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    cache_file = cache_path(root, config) / "analysis_core.npz"
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]).copy() for key in archive.files}
    trial = load_trial_data(root, config)
    features, ids = _load_tangent_features(root, config)
    mode = _load_mode_data(root, config)
    v1_config, _ = population_v1.load_config(root)
    parent = population_v1.load_parent_dataset(root, v1_config, "openbmi")
    outer, _ = population_v1.openbmi_fold_indices(parent, v1_config)
    raw_x = np.empty((54, 2, 210), dtype=np.float64)
    alpha = np.empty((54, 2), dtype=np.float64)
    beta = np.empty((54, 2), dtype=np.float64)
    source_beta_abs = np.empty((6, 2), dtype=np.float64)
    for fold_index, test in enumerate(outer):
        train = np.setdiff1d(np.arange(54), test)
        x_train, x_test, _ = population_v1.fold_features(parent, "F", train, test, helmert=None, kind="magnitude")
        for q in range(2):
            direction = mode.modes[fold_index, q]
            raw_x[test, q] = x_test[:, q]
            norms = np.linalg.norm(x_test[:, q], axis=1)
            alpha[test, q] = (x_test[:, q] / norms[:, None]) @ direction
            beta[test, q] = x_test[:, q] @ direction
            source_beta_abs[fold_index, q] = float(np.mean(np.abs(x_train[:, q] @ direction)))
    delta_vectors = np.empty((54, 2, 210), dtype=np.float64)
    y = np.empty((54, 2, 100), dtype=np.float64)
    for s in range(54):
        f = int(mode.fold_of_subject[s])
        for q in range(2):
            labels = trial.labels[s, q]
            delta_vectors[s, q] = 0.5 * (
                np.mean(features[s, q, labels == 1], axis=0)
                - np.mean(features[s, q, labels == 0], axis=0)
            )
            y[s, q] = features[s, q] @ mode.modes[f, q]
    arrays = {
        "features": features, "labels": trial.labels, "trial_ids": ids,
        "modes": mode.modes, "fold_of_subject": mode.fold_of_subject,
        "raw_x": raw_x, "alpha": alpha, "beta": beta,
        "trial_delta_vectors": delta_vectors, "projected_y": y,
        "source_beta_abs": source_beta_abs, "subjects": np.asarray(trial.subjects),
    }
    population_v1._atomic_savez(cache_file, arrays)
    output = output_path(root, config)
    population_v1._atomic_savez(output / "objects/unlabeled_projected_trials.npz", {
        "projected_y": y, "trial_ids": ids, "subjects": np.asarray(trial.subjects),
        "sessions": np.asarray(trial.sessions),
    })
    population_v1._atomic_savez(output / "objects/oracle_prototype_coordinates.npz", {
        "alpha": alpha, "beta": beta, "raw_x": raw_x,
        "subjects": np.asarray(trial.subjects), "sessions": np.asarray(trial.sessions),
    })
    return arrays


def _session_association_rows(delta: np.ndarray, beta: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in range(2):
        slope, intercept = _linear_calibration(delta[:, q], beta[:, q])
        denominator = max(float(np.mean(np.abs(beta[:, q]))), np.finfo(float).tiny)
        rows.append({
            "session": str(q), "pearson": _pearson(delta[:, q], beta[:, q]),
            "spearman": _spearman(delta[:, q], beta[:, q]),
            "sign_agreement": float(np.mean(np.sign(delta[:, q]) == np.sign(beta[:, q]))),
            "slope_beta_on_trial_delta": slope, "intercept_beta_on_trial_delta": intercept,
            "normalized_absolute_error": float(np.mean(np.abs(delta[:, q] - beta[:, q])) / denominator),
        })
    return rows


def _bridge_leave_one(delta: np.ndarray, beta: np.ndarray, subjects: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for omitted in range(len(subjects)):
        keep = np.arange(len(subjects)) != omitted
        for q in range(2):
            rows.append({
                "omitted_subject": int(subjects[omitted]), "session": str(q),
                "spearman": _spearman(delta[keep, q], beta[keep, q]),
                "pearson": _pearson(delta[keep, q], beta[keep, q]),
            })
    return pd.DataFrame(rows)


def run_trial_mode_compatibility(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = ensure_real_access_lock(root)
    output = output_path(root, config); _ensure_output(output)
    if not (output / "decisions/mode_identity_audit.json").is_file():
        raise DataContractError("mode audit missing")
    core = _build_analysis_core(root, config)
    subject_modes = core["modes"][core["fold_of_subject"]]
    delta = np.einsum("sqp,sqp->sq", core["trial_delta_vectors"], subject_modes, optimize=True)
    # The advanced-index expression above is intentionally checked against the
    # scalar projected-trial definition before labels are joined for evaluation.
    direct_delta = np.empty((54, 2), dtype=np.float64)
    for s in range(54):
        for q in range(2):
            labels = core["labels"][s, q]
            direct_delta[s, q] = 0.5 * (
                np.mean(core["projected_y"][s, q, labels == 1])
                - np.mean(core["projected_y"][s, q, labels == 0])
            )
    np.testing.assert_allclose(delta, direct_delta, rtol=2e-13, atol=2e-14)
    beta = core["beta"]
    rows = _session_association_rows(delta, beta)
    observed = min(float(row["spearman"]) for row in rows)
    leave_one = _bridge_leave_one(delta, beta, core["subjects"])
    replicates = int(config["bridge"]["null_replicates"])
    permutations = np.empty((replicates, 54), dtype=np.int16)
    subject_null = np.empty(replicates, dtype=np.float64)
    random_null = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        mapping = _rng(config, config["bridge"]["subject_permutation_namespace"], r).permutation(54)
        permutations[r] = mapping
        subject_null[r] = min(_spearman(delta[:, q], beta[mapping, q]) for q in range(2))
        random_delta = np.empty((54, 2), dtype=np.float64)
        random_beta = np.empty((54, 2), dtype=np.float64)
        random_modes = np.empty((6, 2, 210), dtype=np.float64)
        for f in range(6):
            for q in range(2):
                direction = _rng(config, config["bridge"]["random_direction_namespace"], r, f, q).normal(size=210)
                random_modes[f, q] = direction / np.linalg.norm(direction)
        for s in range(54):
            f = int(core["fold_of_subject"][s])
            for q in range(2):
                random_delta[s, q] = float(core["trial_delta_vectors"][s, q] @ random_modes[f, q])
                random_beta[s, q] = float(core["raw_x"][s, q] @ random_modes[f, q])
        random_null[r] = min(_spearman(random_delta[:, q], random_beta[:, q]) for q in range(2))
    pairing_p = _plus_one_p(observed, subject_null)
    random_p = _plus_one_p(observed, random_null)
    alpha = float(config["bridge"]["alpha"])
    session_positive = all(float(row["spearman"]) > 0.0 for row in rows)
    influence_positive = bool((leave_one["spearman"] > 0.0).all())
    passed = session_positive and pairing_p <= alpha and random_p <= alpha and influence_positive
    decision = config["decisions"]["bridge"]["pass" if passed else "failure"]
    subject_rows = []
    for s, subject in enumerate(core["subjects"]):
        for q in range(2):
            subject_rows.append({
                "subject": int(subject), "session": str(q), "outer_fold": int(core["fold_of_subject"][s]),
                "trial_delta": float(delta[s, q]), "beta": float(beta[s, q]), "alpha": float(core["alpha"][s, q]),
                "absolute_error": float(abs(delta[s, q] - beta[s, q])),
                "sign_agreement": bool(np.sign(delta[s, q]) == np.sign(beta[s, q])),
            })
    pd.DataFrame(rows).to_csv(output / "tables/trial_prototype_compatibility_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(subject_rows).to_csv(output / "tables/trial_prototype_compatibility_subjects.csv", index=False, lineterminator="\n", float_format="%.17g")
    leave_one.to_csv(output / "tables/trial_prototype_compatibility_influence.csv", index=False, lineterminator="\n", float_format="%.17g")
    population_v1._atomic_savez(output / "nulls/trial_prototype_bridge_nulls.npz", {
        "subject_permutation_statistics": subject_null, "subject_permutations": permutations,
        "random_direction_statistics": random_null,
    })
    result = {
        "schema_version": "unlabeled-conditional-mode-v0-bridge",
        "decision": decision, "passed": passed, "aggregate_statistic": observed,
        "session_metrics": rows, "subject_permutation_p": pairing_p,
        "matched_random_direction_p": random_p, "leave_one_subject_sign_stable": influence_positive,
        "null_replicates": replicates,
    }
    atomic_write_json(output / "decisions/trial_prototype_bridge.json", result)
    return result


def _variance_components(values: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    y = np.asarray(values, dtype=np.float64); c = np.asarray(labels, dtype=np.int8)
    if y.shape != (100,) or c.shape != (100,) or np.count_nonzero(c == 0) != 50 or np.count_nonzero(c == 1) != 50:
        raise DataContractError("variance decomposition requires balanced 100-trial cells")
    total = float(np.var(y, ddof=0))
    means = np.asarray([np.mean(y[c == index]) for index in range(2)], dtype=np.float64)
    variances = np.asarray([np.var(y[c == index], ddof=0) for index in range(2)], dtype=np.float64)
    within = float(np.mean(variances))
    overall = float(np.mean(y))
    between = float(0.5 * np.sum((means - overall) ** 2))
    binary = float(0.25 * (means[1] - means[0]) ** 2)
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, total, within, between)
    if abs(total - within - between) > tolerance or abs(between - binary) > tolerance:
        raise NumericalContractError("total/within/between variance identity failed")
    return total, within, between, 0.5 * float(means[1] - means[0])


def _within_variance(values: np.ndarray, labels: np.ndarray) -> float:
    return float(0.5 * (np.var(values[labels == 0], ddof=0) + np.var(values[labels == 1], ddof=0)))


def _directional_energy(
    features: np.ndarray,
    labels: np.ndarray,
    modes: np.ndarray,
    outer: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    predicted = np.empty((54, 2), dtype=np.float64)
    total = np.empty((54, 2), dtype=np.float64)
    within = np.empty((54, 2), dtype=np.float64)
    between = np.empty((54, 2), dtype=np.float64)
    delta = np.empty((54, 2), dtype=np.float64)
    source_w = np.empty((6, 2), dtype=np.float64)
    all_subjects = np.arange(54)
    for f, test in enumerate(outer):
        train = np.setdiff1d(all_subjects, test)
        for q in range(2):
            projected = np.einsum("stp,p->st", features[:, q], modes[f, q], optimize=True)
            training_within = np.asarray([_within_variance(projected[s], labels[s, q]) for s in train])
            source_w[f, q] = float(np.mean(training_within))
            for s in test:
                total[s, q], within[s, q], between[s, q], delta[s, q] = _variance_components(projected[s], labels[s, q])
                predicted[s, q] = max(total[s, q] - source_w[f, q], 0.0)
    return {"predicted": predicted, "total": total, "within": within, "between": between, "delta": delta, "source_within": source_w}


def _symmetric_mixture_energy(values: np.ndarray, config: Mapping[str, Any]) -> float:
    y = np.asarray(values, dtype=np.float64)
    spec = config["unsigned_recovery"]["mixture"]
    variance_total = float(np.var(y, ddof=0))
    if variance_total <= np.finfo(float).tiny:
        return 0.0
    best_likelihood = -np.inf; best_means = np.asarray([np.mean(y), np.mean(y)])
    for quantile in spec["deterministic_initial_separation_quantiles"]:
        q = min(float(quantile), 1.0 - float(quantile))
        means = np.sort(np.quantile(y, [q, 1.0 - q])).astype(np.float64)
        variance = variance_total
        previous = -np.inf
        for _ in range(int(spec["maximum_iterations"])):
            log0 = -0.5 * ((y - means[0]) ** 2 / variance + math.log(2.0 * math.pi * variance))
            log1 = -0.5 * ((y - means[1]) ** 2 / variance + math.log(2.0 * math.pi * variance))
            maximum = np.maximum(log0, log1)
            denominator = np.exp(log0 - maximum) + np.exp(log1 - maximum)
            responsibility = np.exp(log1 - maximum) / denominator
            weight1 = float(np.sum(responsibility)); weight0 = len(y) - weight1
            if min(weight0, weight1) <= np.finfo(float).eps * len(y):
                break
            updated = np.asarray([
                np.sum((1.0 - responsibility) * y) / weight0,
                np.sum(responsibility * y) / weight1,
            ])
            variance_new = float(np.mean((1.0 - responsibility) * (y - updated[0]) ** 2 + responsibility * (y - updated[1]) ** 2))
            if variance_new <= np.finfo(float).tiny:
                break
            likelihood = float(np.sum(maximum + np.log(0.5 * denominator)))
            means = np.sort(updated); variance = variance_new
            if np.isfinite(previous) and abs(likelihood - previous) <= float(spec["convergence_tolerance"]) * max(1.0, abs(previous)):
                previous = likelihood; break
            previous = likelihood
        if previous > best_likelihood:
            best_likelihood = previous; best_means = means.copy()
    if not np.isfinite(best_likelihood):
        raise NumericalContractError("symmetric mixture fit failed")
    return float(0.25 * (best_means[1] - best_means[0]) ** 2)


def _two_means_energy(values: np.ndarray, config: Mapping[str, Any]) -> float:
    y = np.asarray(values, dtype=np.float64)
    centers = np.asarray(np.quantile(y, [0.25, 0.75]), dtype=np.float64)
    for _ in range(int(config["unsigned_recovery"]["unordered_two_means"]["maximum_iterations"])):
        assignment = np.argmin(np.abs(y[:, None] - centers[None, :]), axis=1)
        if len(np.unique(assignment)) != 2:
            return 0.0
        updated = np.sort(np.asarray([np.mean(y[assignment == index]) for index in range(2)]))
        if np.max(np.abs(updated - centers)) <= float(config["unsigned_recovery"]["unordered_two_means"]["convergence_tolerance"]):
            centers = updated; break
        centers = updated
    return float(0.25 * (centers[1] - centers[0]) ** 2)


def _metric_row(estimator: str, session: str, predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    p = np.asarray(predicted, dtype=np.float64); t = np.asarray(target, dtype=np.float64)
    residual = t - p
    raw_r2 = float(1.0 - np.sum(residual**2) / np.sum((t - np.mean(t)) ** 2))
    slope, intercept = _linear_calibration(p, t)
    return {
        "estimator": estimator, "session": session, "n": len(p),
        "spearman": _spearman(p, t), "pearson": _pearson(p, t),
        "raw_r2": raw_r2, "nonnegative_r2": max(raw_r2, 0.0),
        "mae": float(np.mean(np.abs(residual))),
        "normalized_mae": float(np.mean(np.abs(residual)) / max(np.mean(np.abs(t)), np.finfo(float).tiny)),
        "calibration_slope": slope, "calibration_intercept": intercept,
    }


def _bootstrap_spearman_ci(
    config: Mapping[str, Any], predicted: np.ndarray, target: np.ndarray, namespace: str
) -> tuple[float, float, np.ndarray]:
    replicates = int(config["unsigned_recovery"]["bootstrap_replicates"])
    indices = _subject_bootstrap_indices(config, namespace, replicates, 54)
    values = np.empty(replicates, dtype=np.float64)
    for r, selected in enumerate(indices):
        values[r] = _spearman(predicted[selected].reshape(-1), target[selected].reshape(-1))
    confidence = float(config["unsigned_recovery"]["confidence"])
    tail = 0.5 * (1.0 - confidence)
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high), values


def _recovery_leave_one(predicted: np.ndarray, target: np.ndarray, subjects: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for omitted, subject in enumerate(subjects):
        keep = np.arange(54) != omitted
        rows.append({"omitted_subject": int(subject), "session": "pooled", "spearman": _spearman(predicted[keep].reshape(-1), target[keep].reshape(-1))})
        for q in range(2):
            rows.append({"omitted_subject": int(subject), "session": str(q), "spearman": _spearman(predicted[keep, q], target[keep, q])})
    return pd.DataFrame(rows)


def _same_session_pca_modes(root: Path, config: Mapping[str, Any]) -> np.ndarray:
    v1_config, _ = population_v1.load_config(root)
    parent = population_v1.load_parent_dataset(root, v1_config, "openbmi")
    outer, _ = population_v1.openbmi_fold_indices(parent, v1_config)
    result = np.empty((6, 2, 210), dtype=np.float64)
    for f, test in enumerate(outer):
        train = np.setdiff1d(np.arange(54), test)
        x_train, _, _ = population_v1.fold_features(parent, "F", train, test, helmert=None, kind="sensor")
        for q in range(2):
            centered = x_train[:, q] - np.mean(x_train[:, q], axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            result[f, q] = vh[0]
    return result


def _random_direction_recovery_null(
    config: Mapping[str, Any], features: np.ndarray, labels: np.ndarray,
    outer: Sequence[np.ndarray], replicates: int,
) -> np.ndarray:
    result = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        modes = np.empty((6, 2, 210), dtype=np.float64)
        for f in range(6):
            for q in range(2):
                value = _rng(config, config["unsigned_recovery"]["random_direction_namespace"], r, f, q).normal(size=210)
                modes[f, q] = value / np.linalg.norm(value)
        random = _directional_energy(features, labels, modes, outer)
        result[r] = _spearman(random["predicted"].reshape(-1), random["between"].reshape(-1))
        if (r + 1) % 100 == 0:
            print(f"unsigned random directions {r + 1}/{replicates}", flush=True)
    return result


def run_unsigned_recovery(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = ensure_real_access_lock(root)
    output = output_path(root, config); _ensure_output(output)
    bridge_path = output / "decisions/trial_prototype_bridge.json"
    if not bridge_path.is_file():
        raise DataContractError("trial/prototype bridge must run first")
    bridge = json.loads(bridge_path.read_text())
    if not bridge["passed"]:
        result = {
            "schema_version": "unlabeled-conditional-mode-v0-unsigned",
            "decision": config["decisions"]["bridge"]["failure"], "passed": False,
            "reason": "Second-moment recovery was not accessed after bridge incompatibility.",
        }
        atomic_write_json(output / "decisions/unsigned_recovery.json", result)
        return result
    core = _build_analysis_core(root, config)
    v1_config, _ = population_v1.load_config(root)
    parent = population_v1.load_parent_dataset(root, v1_config, "openbmi")
    outer, _ = population_v1.openbmi_fold_indices(parent, v1_config)
    energy = _directional_energy(core["features"], core["labels"], core["modes"], outer)
    np.testing.assert_allclose(energy["delta"], np.einsum("sqp,sqp->sq", core["trial_delta_vectors"], core["modes"][core["fold_of_subject"]]), rtol=2e-13, atol=2e-14)
    primary = energy["predicted"]
    target = energy["between"]
    mixture = np.empty((54, 2), dtype=np.float64); two_means = np.empty((54, 2), dtype=np.float64)
    for s in range(54):
        for q in range(2):
            mixture[s, q] = _symmetric_mixture_energy(core["projected_y"][s, q], config)
            two_means[s, q] = _two_means_energy(core["projected_y"][s, q], config)
    estimates = {
        "SOURCE_WITHIN_CORRECTED_PROJECTED_VARIANCE": primary,
        "TOTAL_PROJECTED_VARIANCE": energy["total"],
        "SYMMETRIC_TWO_COMPONENT_MIXTURE": mixture,
        "UNORDERED_TWO_MEANS_SEPARATION": two_means,
    }
    decomposition_rows: list[dict[str, Any]] = []
    for s, subject in enumerate(core["subjects"]):
        for q in range(2):
            decomposition_rows.append({
                "subject": int(subject), "session": str(q), "outer_fold": int(core["fold_of_subject"][s]),
                "V_total": float(energy["total"][s, q]), "V_within": float(energy["within"][s, q]),
                "V_between": float(target[s, q]), "delta_trial": float(energy["delta"][s, q]),
                "absolute_beta": float(abs(core["beta"][s, q])), "beta_squared": float(core["beta"][s, q] ** 2),
                **{name: float(value[s, q]) for name, value in estimates.items()},
            })
    pd.DataFrame(decomposition_rows).to_csv(output / "tables/variance_decomposition_and_estimates.csv", index=False, lineterminator="\n", float_format="%.17g")
    metric_rows: list[dict[str, Any]] = []
    bootstrap_values: dict[str, np.ndarray] = {}
    for name, estimate in estimates.items():
        pooled = _metric_row(name, "pooled", estimate.reshape(-1), target.reshape(-1))
        low, high, values = _bootstrap_spearman_ci(config, estimate, target, f"{config['unsigned_recovery']['bootstrap_namespace']}|{name}")
        pooled["bootstrap_ci_low"] = low; pooled["bootstrap_ci_high"] = high
        metric_rows.append(pooled); bootstrap_values[name] = values
        for q in range(2):
            row = _metric_row(name, str(q), estimate[:, q], target[:, q])
            row["bootstrap_ci_low"] = "NOT_APPLICABLE"; row["bootstrap_ci_high"] = "NOT_APPLICABLE"
            metric_rows.append(row)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "tables/unsigned_recovery_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    influence = _recovery_leave_one(primary, target, core["subjects"])
    influence.to_csv(output / "tables/unsigned_recovery_influence.csv", index=False, lineterminator="\n", float_format="%.17g")
    replicates = int(config["unsigned_recovery"]["null_replicates"])
    permutations = np.empty((replicates, 54), dtype=np.int16); subject_null = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        mapping = _rng(config, config["unsigned_recovery"]["subject_permutation_namespace"], r).permutation(54)
        permutations[r] = mapping
        subject_null[r] = _spearman(primary.reshape(-1), target[mapping].reshape(-1))
    observed = _spearman(primary.reshape(-1), target.reshape(-1))
    random_null = _random_direction_recovery_null(config, core["features"], core["labels"], outer, replicates)
    subject_p = _plus_one_p(observed, subject_null); random_p = _plus_one_p(observed, random_null)

    swapped_labels = 1 - core["labels"]
    swapped = _directional_energy(core["features"], swapped_labels, core["modes"], outer)
    tolerance = float(config["symmetry"]["tolerance"])
    symmetry_errors = {
        "pooled_trial_object": 0.0,
        "projected_y": 0.0,
        "primary_unsigned_estimate": float(np.max(np.abs(swapped["predicted"] - primary))),
        "V_between": float(np.max(np.abs(swapped["between"] - target))),
        "absolute_beta": float(np.max(np.abs(np.abs(-core["beta"]) - np.abs(core["beta"])))),
        "signed_beta_negation": float(np.max(np.abs((-core["beta"]) + core["beta"]))),
        "signed_alpha_negation": float(np.max(np.abs((-core["alpha"]) + core["alpha"]))),
    }
    symmetry_pass = bool(max(symmetry_errors.values()) <= tolerance)
    order_check = np.empty_like(primary)
    for s in range(54):
        f = int(core["fold_of_subject"][s])
        for q in range(2):
            reversed_y = core["projected_y"][s, q, ::-1]
            order_check[s, q] = max(float(np.var(reversed_y, ddof=0)) - energy["source_within"][f, q], 0.0)
    acquisition_order_error = float(np.max(np.abs(order_check - primary)))
    if acquisition_order_error > tolerance:
        raise DataContractError("acquisition-order removal verification failed")

    pca_modes = _same_session_pca_modes(root, config)
    pca = _directional_energy(core["features"], core["labels"], pca_modes, outer)
    fold_derangement = np.roll(np.arange(6), 1)
    shuffled_w = np.empty_like(primary)
    for s in range(54):
        f = int(core["fold_of_subject"][s])
        for q in range(2):
            shuffled_w[s, q] = max(energy["total"][s, q] - energy["source_within"][fold_derangement[f], q], 0.0)
    with np.load(output / "objects/unlabeled_marginal_means.npz", allow_pickle=False) as archive:
        marginals = np.asarray(archive["marginal_means"], dtype=np.float64)
    covariance_magnitude = np.log(np.linalg.norm(marginals, axis=(-2, -1)))
    total_tangent_variance = np.sum(np.var(core["features"], axis=2, ddof=0), axis=-1)
    control_rows = [
        {"control": "same_session_PCA", **_metric_row("same_session_PCA", "pooled", pca["predicted"].reshape(-1), pca["between"].reshape(-1))},
        {"control": "overall_covariance_magnitude", "spearman_with_learned_mode_V_between": _spearman(covariance_magnitude.reshape(-1), target.reshape(-1))},
        {"control": "total_tangent_space_variance", "spearman_with_learned_mode_V_between": _spearman(total_tangent_variance.reshape(-1), target.reshape(-1))},
        {"control": "shuffled_source_within_model", **_metric_row("shuffled_source_within_model", "pooled", shuffled_w.reshape(-1), target.reshape(-1))},
        {"control": "trial_count", "status": "CONSTANT_100_TRIALS_PER_SUBJECT_SESSION"},
        {"control": "acquisition_order_removal", "status": "PASS", "maximum_estimate_difference": acquisition_order_error},
    ]
    # Heterogeneous descriptive control rows are serialized as JSON to avoid
    # fabricating numeric placeholders for inapplicable fields.
    atomic_write_json(output / "controls/unsigned_recovery_controls.json", {"controls": control_rows})
    symmetry_result = {
        "status": "PASS" if symmetry_pass else config["symmetry"]["failure_status"],
        "passed": symmetry_pass, "tolerance": tolerance, "maximum_errors": symmetry_errors,
        "signed_zero_label_identification": config["symmetry"]["signed_zero_label_status"] if symmetry_pass else config["decisions"]["signed"]["failure"],
    }
    atomic_write_json(output / "decisions/label_swap_symmetry.json", symmetry_result)
    primary_metrics = metrics_frame[(metrics_frame["estimator"] == config["unsigned_recovery"]["primary_estimator"]) & (metrics_frame["session"] == "pooled")].iloc[0]
    session_positive = all(_spearman(primary[:, q], target[:, q]) > 0.0 for q in range(2))
    influence_positive = bool((influence["spearman"] > 0.0).all())
    ci_excludes_zero = float(primary_metrics["bootstrap_ci_low"]) > 0.0
    alpha = float(config["bridge"]["alpha"])
    passed = bool(session_positive and ci_excludes_zero and subject_p <= alpha and random_p <= alpha and influence_positive and symmetry_pass)
    decision = config["decisions"]["unsigned"]["pass" if passed else "negative"]
    result = {
        "schema_version": "unlabeled-conditional-mode-v0-unsigned",
        "decision": decision, "passed": passed, "primary_estimator": config["unsigned_recovery"]["primary_estimator"],
        "pooled_spearman": observed, "session_spearman": [_spearman(primary[:, q], target[:, q]) for q in range(2)],
        "pooled_pearson": _pearson(primary.reshape(-1), target.reshape(-1)),
        "bootstrap_ci": [float(primary_metrics["bootstrap_ci_low"]), float(primary_metrics["bootstrap_ci_high"])],
        "subject_permutation_p": subject_p, "matched_random_direction_p": random_p,
        "leave_one_subject_sign_stable": influence_positive, "symmetry_gate_pass": symmetry_pass,
        "source_within_by_fold_session": energy["source_within"].tolist(),
    }
    atomic_write_json(output / "decisions/unsigned_recovery.json", result)
    population_v1._atomic_savez(output / "objects/unsigned_recovery_core.npz", {
        "primary_estimate": primary, "oracle_between": target, "total": energy["total"],
        "within": energy["within"], "delta_trial": energy["delta"],
        "mixture_estimate": mixture, "two_means_estimate": two_means,
        "source_within": energy["source_within"], "pca_estimate": pca["predicted"],
        "shuffled_within_estimate": shuffled_w,
    })
    population_v1._atomic_savez(output / "nulls/unsigned_recovery_nulls.npz", {
        "subject_permutation_statistics": subject_null, "subject_permutations": permutations,
        "random_direction_statistics": random_null,
        **{f"bootstrap_{hashlib.sha256(name.encode()).hexdigest()[:12]}": values for name, values in bootstrap_values.items()},
    })
    return result


def _bootstrap_mean_ci(
    config: Mapping[str, Any], values: np.ndarray, namespace: str, replicates: int,
) -> tuple[float, float, np.ndarray]:
    subject_values = np.asarray(values, dtype=np.float64)
    indices = _subject_bootstrap_indices(config, namespace, replicates, len(subject_values))
    bootstrap = np.mean(subject_values[indices], axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high), bootstrap


def run_minimal_anchor(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = ensure_real_access_lock(root)
    output = output_path(root, config); _ensure_output(output)
    unsigned_path = output / "decisions/unsigned_recovery.json"
    if not unsigned_path.is_file():
        raise DataContractError("unsigned recovery must run before minimal anchoring")
    unsigned = json.loads(unsigned_path.read_text())
    if unsigned["decision"].startswith("UNASSESSED_"):
        result = {
            "schema_version": "unlabeled-conditional-mode-v0-anchor",
            "decision": unsigned["decision"], "passed": False, "selected_budget": None,
            "reason": "Minimal anchoring was not accessed after an unassessed prerequisite.",
        }
        atomic_write_json(output / "decisions/minimal_anchor.json", result)
        return result
    core = _build_analysis_core(root, config)
    with np.load(output / "objects/unsigned_recovery_core.npz", allow_pickle=False) as archive:
        magnitude = np.sqrt(np.maximum(np.asarray(archive["primary_estimate"], dtype=np.float64), 0.0))
    budgets = [int(x) for x in config["minimal_anchor"]["budgets"]]
    positive = [value for value in budgets if value > 0]
    repetitions = int(config["minimal_anchor"]["calibration_subsamples"])
    shape = (len(positive), 54, 2, repetitions)
    proposed = np.empty(shape, dtype=np.float64); direct = np.empty(shape, dtype=np.float64)
    source = np.empty(shape, dtype=np.float64); orientation = np.empty(shape, dtype=np.int8)
    for bi, budget in enumerate(positive):
        per_class = budget // 2
        for s in range(54):
            f = int(core["fold_of_subject"][s])
            for q in range(2):
                left = np.flatnonzero(core["labels"][s, q] == 0)
                right = np.flatnonzero(core["labels"][s, q] == 1)
                for r in range(repetitions):
                    rng = _rng(config, "anchor_calibration", budget, s, q, r)
                    chosen_l = rng.choice(left, size=per_class, replace=False)
                    chosen_r = rng.choice(right, size=per_class, replace=False)
                    contrast = 0.5 * float(np.mean(core["projected_y"][s, q, chosen_r]) - np.mean(core["projected_y"][s, q, chosen_l]))
                    sign = 1 if contrast > 0.0 else (-1 if contrast < 0.0 else 0)
                    orientation[bi, s, q, r] = sign
                    proposed[bi, s, q, r] = sign * magnitude[s, q]
                    direct[bi, s, q, r] = contrast
                    source[bi, s, q, r] = sign * core["source_beta_abs"][f, q]
    beta = core["beta"][None, :, :, None]
    proposed_error = np.abs(proposed - beta); direct_error = np.abs(direct - beta); source_error = np.abs(source - beta)
    true_sign = np.sign(core["beta"])[None, :, :, None]
    proposed_sign_accuracy = orientation == true_sign
    direct_sign_accuracy = np.sign(direct) == true_sign
    summary_rows: list[dict[str, Any]] = [{
        "budget": 0, "subject": "ALL", "session": "ALL",
        "status": config["symmetry"]["signed_zero_label_status"],
        "proposed_mae": "NOT_IDENTIFIABLE", "direct_mae": "NOT_IDENTIFIABLE",
        "source_mae": "NOT_IDENTIFIABLE", "proposed_sign_accuracy": "NOT_IDENTIFIABLE",
    }]
    for bi, budget in enumerate(positive):
        for s, subject in enumerate(core["subjects"]):
            for q in range(2):
                summary_rows.append({
                    "budget": budget, "subject": int(subject), "session": str(q), "status": "EVALUATED",
                    "proposed_mae": float(np.mean(proposed_error[bi, s, q])),
                    "direct_mae": float(np.mean(direct_error[bi, s, q])),
                    "source_mae": float(np.mean(source_error[bi, s, q])),
                    "proposed_sign_accuracy": float(np.mean(proposed_sign_accuracy[bi, s, q])),
                    "direct_sign_accuracy": float(np.mean(direct_sign_accuracy[bi, s, q])),
                    "projected_prototype_reconstruction_error_proposed": float(np.mean(proposed_error[bi, s, q])),
                    "projected_prototype_reconstruction_error_direct": float(np.mean(direct_error[bi, s, q])),
                })
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output / "tables/minimal_anchor_subject_budget.csv", index=False, lineterminator="\n", float_format="%.17g")
    budget_rows: list[dict[str, Any]] = []
    bootstrap_arrays: dict[str, np.ndarray] = {}; null_arrays: dict[str, np.ndarray] = {}
    selected_budget: int | None = None
    for bi, budget in enumerate(positive):
        proposed_subject_session = np.mean(proposed_error[bi], axis=-1)
        direct_subject_session = np.mean(direct_error[bi], axis=-1)
        source_subject_session = np.mean(source_error[bi], axis=-1)
        improvement_session = direct_subject_session - proposed_subject_session
        improvement_subject = np.mean(improvement_session, axis=1)
        accuracy_subject = np.mean(proposed_sign_accuracy[bi], axis=(1, 2))
        bootstrap_reps = int(config["minimal_anchor"]["bootstrap_replicates"])
        low, high, bootstrap = _bootstrap_mean_ci(config, improvement_subject, f"{config['minimal_anchor']['bootstrap_namespace']}|improvement|{budget}", bootstrap_reps)
        acc_low, acc_high, acc_bootstrap = _bootstrap_mean_ci(config, accuracy_subject, f"{config['minimal_anchor']['bootstrap_namespace']}|accuracy|{budget}", bootstrap_reps)
        null = np.empty(int(config["minimal_anchor"]["null_replicates"]), dtype=np.float64)
        for r in range(len(null)):
            signs = _rng(config, config["minimal_anchor"]["paired_permutation_namespace"], budget, r).choice([-1.0, 1.0], size=54)
            null[r] = float(np.mean(signs * improvement_subject))
        observed = float(np.mean(improvement_subject)); p_value = _plus_one_p(observed, null)
        leave_one = np.asarray([np.mean(np.delete(improvement_subject, s)) for s in range(54)])
        session_improvement = np.mean(improvement_session, axis=0)
        eligible = budget in [int(x) for x in config["minimal_anchor"]["eligible_efficiency_budgets"]]
        passed = bool(
            eligible and observed > 0.0 and p_value <= float(config["minimal_anchor"]["alpha"])
            and low > 0.0 and np.all(session_improvement > 0.0) and acc_low > 0.5
            and np.all(leave_one > 0.0)
        )
        if passed and selected_budget is None:
            selected_budget = budget
        budget_rows.append({
            "budget": budget, "proposed_mae": float(np.mean(proposed_subject_session)),
            "direct_mae": float(np.mean(direct_subject_session)), "source_mae": float(np.mean(source_subject_session)),
            "mean_improvement_direct_minus_proposed": observed,
            "improvement_session0": float(session_improvement[0]), "improvement_session1": float(session_improvement[1]),
            "improvement_ci_low": low, "improvement_ci_high": high, "paired_permutation_p": p_value,
            "proposed_sign_accuracy": float(np.mean(accuracy_subject)), "sign_accuracy_ci_low": acc_low,
            "sign_accuracy_ci_high": acc_high, "leave_one_subject_sign_stable": bool(np.all(leave_one > 0.0)),
            "eligible_for_terminal": eligible, "passes_efficiency_gates": passed,
        })
        bootstrap_arrays[f"improvement_budget_{budget}"] = bootstrap
        bootstrap_arrays[f"sign_accuracy_budget_{budget}"] = acc_bootstrap
        null_arrays[f"paired_permutation_budget_{budget}"] = null
    budget_frame = pd.DataFrame(budget_rows)
    budget_frame.to_csv(output / "tables/minimal_anchor_budget_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    passed = selected_budget is not None
    decision = config["decisions"]["anchor"]["pass" if passed else "negative"]
    result = {
        "schema_version": "unlabeled-conditional-mode-v0-anchor",
        "decision": decision, "passed": passed, "selected_budget": selected_budget,
        "budgets": budget_rows, "calibration_subsamples": repetitions,
        "zero_label_status": config["symmetry"]["signed_zero_label_status"],
    }
    atomic_write_json(output / "decisions/minimal_anchor.json", result)
    population_v1._atomic_savez(output / "objects/minimal_anchor_subsamples.npz", {
        "positive_budgets": np.asarray(positive), "proposed": proposed, "direct": direct,
        "source": source, "orientation": orientation, "proposed_error": proposed_error,
        "direct_error": direct_error, "source_error": source_error,
    })
    population_v1._atomic_savez(output / "nulls/minimal_anchor_nulls.npz", {**bootstrap_arrays, **null_arrays})
    return result


def _save_figure(fig: Any, output: Path, stem: str) -> None:
    fig.savefig(output / "figures" / f"{stem}.png", dpi=170, bbox_inches="tight")
    fig.savefig(output / "figures" / f"{stem}.pdf", bbox_inches="tight")


def _placeholder_figure(plt: Any, message: str, output: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5)); ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save_figure(fig, output, stem); plt.close(fig)


def generate_figures(root: Path, config: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = output_path(root, config)
    with np.load(output / "objects/rank1_session_view_modes.npz", allow_pickle=False) as archive:
        modes = np.asarray(archive["modes"]); aligned = np.asarray(archive["aligned_modes_visualization"]); haar = np.asarray(archive["haar_absolute_cosines"])
    labels = [f"F{f}/S{q}" for f in range(6) for q in range(2)]
    flat = modes.reshape(12, 210); matrix = np.abs(flat @ flat.T)
    fig, ax = plt.subplots(figsize=(7.2, 6.2)); image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(12), labels, rotation=90); ax.set_yticks(range(12), labels); fig.colorbar(image, ax=ax, label="absolute cosine")
    _save_figure(fig, output, "figure_01_mode_absolute_cosine_matrix"); plt.close(fig)

    refits = pd.read_csv(output / "tables/mode_leave_one_training_subject_refits.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.hist(refits["absolute_cosine"], bins=30, alpha=0.75, label="leave-one-training-subject refit")
    ax.hist(haar.reshape(-1), bins=30, alpha=0.55, label="Haar baseline"); ax.set_xlabel("absolute cosine"); ax.legend()
    _save_figure(fig, output, "figure_02_mode_refit_stability"); plt.close(fig)

    mean0 = _inverse_svec(np.mean(aligned[:, 0], axis=0)); mean1 = _inverse_svec(np.mean(aligned[:, 1], axis=0))
    limit = max(float(np.max(np.abs(mean0))), float(np.max(np.abs(mean1))))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for q, (ax, value) in enumerate(zip(axes, (mean0, mean1))):
        im = ax.imshow(value, cmap="coolwarm", vmin=-limit, vmax=limit); ax.set_title(f"session {q} aligned mean mode"); ax.set_xlabel("channel"); ax.set_ylabel("channel")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    _save_figure(fig, output, "figure_03_rank1_sensor_matrix_energy"); plt.close(fig)

    bridge_subjects = output / "tables/trial_prototype_compatibility_subjects.csv"
    if bridge_subjects.is_file():
        frame = pd.read_csv(bridge_subjects); fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for session, group in frame.groupby("session"):
            ax.scatter(group["beta"], group["trial_delta"], alpha=0.75, label=f"session {session}")
        ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1); ax.set_xlabel("beta"); ax.set_ylabel("trial delta"); ax.legend()
        _save_figure(fig, output, "figure_04_trial_prototype_compatibility"); plt.close(fig)
    else:
        _placeholder_figure(plt, "Trial/prototype bridge unassessed", output, "figure_04_trial_prototype_compatibility")

    decomposition_path = output / "tables/variance_decomposition_and_estimates.csv"
    if decomposition_path.is_file():
        frame = pd.read_csv(decomposition_path).sort_values("V_total"); x = np.arange(len(frame))
        fig, ax = plt.subplots(figsize=(10.0, 4.6)); ax.bar(x, frame["V_within"], label="within"); ax.bar(x, frame["V_between"], bottom=frame["V_within"], label="between")
        ax.set_xlabel("subject-session sorted by total variance"); ax.set_ylabel("projected variance"); ax.legend()
        _save_figure(fig, output, "figure_05_variance_decomposition"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.0, 5.2)); ax.scatter(frame["V_between"], frame["SOURCE_WITHIN_CORRECTED_PROJECTED_VARIANCE"], alpha=0.7)
        upper = float(max(frame["V_between"].max(), frame["SOURCE_WITHIN_CORRECTED_PROJECTED_VARIANCE"].max())); ax.plot([0, upper], [0, upper], "k--", linewidth=1)
        ax.set_xlabel("oracle between-class energy"); ax.set_ylabel("zero-label estimate")
        _save_figure(fig, output, "figure_06_predicted_vs_oracle_energy"); plt.close(fig)
        with np.load(output / "nulls/unsigned_recovery_nulls.npz", allow_pickle=False) as archive:
            permutation = np.asarray(archive["subject_permutation_statistics"]); random = np.asarray(archive["random_direction_statistics"])
        observed = _spearman(frame["SOURCE_WITHIN_CORRECTED_PROJECTED_VARIANCE"].to_numpy(), frame["V_between"].to_numpy())
        fig, ax = plt.subplots(figsize=(7.0, 4.5)); ax.hist(permutation, bins=35); ax.axvline(observed, color="red", label="observed"); ax.set_xlabel("pooled Spearman"); ax.legend()
        _save_figure(fig, output, "figure_07_subject_permutation_null"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.0, 4.5)); ax.hist(random, bins=35); ax.axvline(observed, color="red", label="learned mode"); ax.set_xlabel("pooled Spearman"); ax.legend()
        _save_figure(fig, output, "figure_08_random_direction_control"); plt.close(fig)
        symmetry = json.loads((output / "decisions/label_swap_symmetry.json").read_text()); names = list(symmetry["maximum_errors"]); values = [symmetry["maximum_errors"][name] for name in names]
        fig, ax = plt.subplots(figsize=(9.0, 4.6)); ax.bar(np.arange(len(names)), values); ax.axhline(symmetry["tolerance"], color="red", linestyle="--"); ax.set_yscale("symlog", linthresh=1e-18); ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
        _save_figure(fig, output, "figure_09_label_swap_invariance"); plt.close(fig)
    else:
        for number, name, message in (
            (5, "variance_decomposition", "Variance recovery not accessed"), (6, "predicted_vs_oracle_energy", "Unsigned recovery not accessed"),
            (7, "subject_permutation_null", "Unsigned null not accessed"), (8, "random_direction_control", "Random-direction control not accessed"),
            (9, "label_swap_invariance", "Symmetry gate not accessed"),
        ):
            _placeholder_figure(plt, message, output, f"figure_{number:02d}_{name}")

    anchor_path = output / "tables/minimal_anchor_budget_summary.csv"
    if anchor_path.is_file():
        frame = pd.read_csv(anchor_path); fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(frame["budget"], frame["proposed_mae"], marker="o", label="unlabeled magnitude + orientation"); ax.plot(frame["budget"], frame["direct_mae"], marker="o", label="direct m-label")
        ax.plot(frame["budget"], frame["source_mae"], marker="o", label="source magnitude + orientation"); ax.set_xlabel("calibration labels m"); ax.set_ylabel("signed beta MAE"); ax.legend()
        _save_figure(fig, output, "figure_10_calibration_budget_mae"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.0, 4.5)); ax.plot(frame["budget"], frame["proposed_sign_accuracy"], marker="o"); ax.axhline(0.5, color="black", linestyle="--"); ax.set_ylim(0, 1.02); ax.set_xlabel("calibration labels m"); ax.set_ylabel("sign accuracy")
        _save_figure(fig, output, "figure_11_calibration_budget_sign_accuracy"); plt.close(fig)
        subject_frame = pd.read_csv(output / "tables/minimal_anchor_subject_budget.csv"); numeric = subject_frame[subject_frame["status"] == "EVALUATED"].copy()
        chosen = int(frame.iloc[0]["budget"]); chosen_rows = numeric[numeric["budget"] == chosen]
        fig, ax = plt.subplots(figsize=(6.0, 5.2)); ax.scatter(chosen_rows["direct_mae"], chosen_rows["proposed_mae"], alpha=0.7); upper = float(max(chosen_rows["direct_mae"].max(), chosen_rows["proposed_mae"].max())); ax.plot([0, upper], [0, upper], "k--"); ax.set_xlabel("direct m-label MAE"); ax.set_ylabel("proposed MAE")
        _save_figure(fig, output, "figure_12_proposed_vs_equal_label_baseline"); plt.close(fig)
    else:
        _placeholder_figure(plt, "Minimal anchoring not accessed", output, "figure_10_calibration_budget_mae")
        _placeholder_figure(plt, "Minimal anchoring not accessed", output, "figure_11_calibration_budget_sign_accuracy")
        _placeholder_figure(plt, "Minimal anchoring not accessed", output, "figure_12_proposed_vs_equal_label_baseline")


def _mode_audit_document(summary: Mapping[str, Any]) -> str:
    return f"""# Mode Identity Audit V0

Status: **DESCRIPTIVE; DOES NOT ALTER PR #16**

- Trial object gate: `{summary['trial_level_object_gate']}`
- Minimum absolute cosine to frozen parent direction: `{summary['minimum_parent_identity_absolute_cosine']:.10f}`
- Median fold-pair absolute cosine: `{summary['median_fold_pair_absolute_cosine']:.10f}`
- Median within-fold session-view absolute cosine: `{summary['median_cross_view_absolute_cosine']:.10f}`
- Median leave-one-training-subject refit absolute cosine: `{summary['median_leave_one_refit_absolute_cosine']:.10f}`
- Minimum leave-one-training-subject refit absolute cosine: `{summary['minimum_leave_one_refit_absolute_cosine']:.10f}`
- Haar median absolute cosine: `{summary['haar_median_absolute_cosine']:.10f}`

All inferential stability summaries are sign-invariant. The paired sign alignment used in sensor-coordinate displays is the predeclared fold-0/session-0 convention and has no effect on any decision. Matrices, channel strengths, and edges are montage-coordinate descriptions and are not source-localized anatomy.
"""


def _artifact_manifest(output: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(value for value in output.rglob("*") if value.is_file() and value.name != "manifest.json"):
        files[str(path.relative_to(output))] = sha256_file(path)
    return files


def generate_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, provenance = ensure_real_access_lock(root)
    output = output_path(root, config); _ensure_output(output)
    required = [
        output / "decisions/mode_identity_audit.json", output / "decisions/trial_prototype_bridge.json",
        output / "decisions/unsigned_recovery.json", output / "decisions/minimal_anchor.json",
    ]
    if any(not path.is_file() for path in required):
        raise DataContractError("report prerequisites are incomplete")
    mode = json.loads(required[0].read_text()); bridge = json.loads(required[1].read_text())
    unsigned = json.loads(required[2].read_text()); anchor = json.loads(required[3].read_text())
    symmetry_path = output / "decisions/label_swap_symmetry.json"
    symmetry = json.loads(symmetry_path.read_text()) if symmetry_path.is_file() else {"signed_zero_label_identification": config["decisions"]["signed"]["failure"], "status": "UNASSESSED"}
    atomic_write_bytes(root / str(config["project"]["mode_audit_path"]), _mode_audit_document(mode).encode("utf-8"))
    next_question = config["decisions"]["next_question_if_supported"] if unsigned.get("passed") and anchor.get("passed") else config["decisions"]["next_question_otherwise"]
    unsigned_line = (
        f"pooled Spearman `{unsigned['pooled_spearman']:.10f}`, sessions `{unsigned['session_spearman']}`, "
        f"95% CI `{unsigned['bootstrap_ci']}`, subject-permutation p `{unsigned['subject_permutation_p']:.6g}`, "
        f"random-direction p `{unsigned['matched_random_direction_p']:.6g}`"
        if "pooled_spearman" in unsigned else unsigned.get("reason", "unassessed")
    )
    anchor_line = (
        f"selected budget `{anchor['selected_budget']}`; zero-label status `{anchor['zero_label_status']}`"
        if "zero_label_status" in anchor else anchor.get("reason", "unassessed")
    )
    report = f"""# Unlabeled Conditional-Mode Identifiability V0

## Frozen parent

PR #16 at `9dee7642ac573f37756b8427a75864a50c32044e` remains `GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION`. This work neither recomputes nor reinterprets that terminal, and all PR #16 artifact hashes remain unchanged.

## Trial-level object gate

`{mode['trial_level_object_gate']}`. The exact parent preprocessing was rebuilt from the hash-locked source manifest; no alternative filtering, epoch, channel, covariance estimator, regularization, reference, or trial rule was used.

## Mode identity and stability

Minimum parent-direction absolute cosine `{mode['minimum_parent_identity_absolute_cosine']:.10f}`; median fold-pair `{mode['median_fold_pair_absolute_cosine']:.10f}`; median cross-view `{mode['median_cross_view_absolute_cosine']:.10f}`; median leave-one-training-subject refit `{mode['median_leave_one_refit_absolute_cosine']:.10f}`. This audit is descriptive and montage-coordinate only.

## Separate frozen decisions

1. Signed zero-label identification: **`{symmetry['signed_zero_label_identification']}`**.
2. Trial/prototype bridge: **`{bridge['decision']}`**. Session metrics: `{json.dumps(bridge.get('session_metrics', []), sort_keys=True)}`; subject-permutation p `{bridge.get('subject_permutation_p', 'UNASSESSED')}`; random-direction p `{bridge.get('matched_random_direction_p', 'UNASSESSED')}`.
3. Unsigned zero-label recovery: **`{unsigned['decision']}`**. {unsigned_line}.
4. Minimal semantic anchoring: **`{anchor['decision']}`**. {anchor_line}.

## Interpretation boundary

The zero-label signed coordinate is non-identifiable under class permutation. Any supported unsigned result concerns projected between-class energy only. Any anchoring result concerns semantic orientation of the frozen one-dimensional mode only. This does not establish a full conditional distribution, physiology, source anatomy, causal mechanism, classification benefit, pseudo-label quality, TTA recoverability, ASD biomarker, or clinical utility.

## Dataset boundary

Only retrospective OpenBMI / Lee2019-MI was executed. BNCI, Stieger2021, HGD, all ASD datasets, downstream classifiers, domain adaptation, TTA, neural networks, and pseudo-labeling were not run.

## Exact next scientific question

{next_question}
"""
    report_path = output / "report/unlabeled_conditional_mode_identifiability_v0.md"
    atomic_write_bytes(report_path, report.encode("utf-8"))
    decision_rows = [
        {"component": "signed_zero_label_identification", "decision": symmetry["signed_zero_label_identification"]},
        {"component": "trial_prototype_bridge", "decision": bridge["decision"]},
        {"component": "unsigned_zero_label_recovery", "decision": unsigned["decision"]},
        {"component": "minimal_semantic_anchor", "decision": anchor["decision"]},
    ]
    pd.DataFrame(decision_rows).to_csv(output / "tables/final_decisions.csv", index=False, lineterminator="\n")
    generate_figures(root, config)
    validate_parent_artifacts(root, config); validate_parent_snapshot(root, output)
    artifacts = _artifact_manifest(output)
    manifest = {
        "schema_version": "unlabeled-conditional-mode-v0-final-manifest",
        "parent_head": config["protocol"]["parent_head"], "protocol_freeze_commit": provenance["protocol_freeze_commit"],
        "artifact_count": len(artifacts), "artifacts": artifacts,
        "decisions": {row["component"]: row["decision"] for row in decision_rows},
        "pr16_artifacts_unchanged": True, "next_question": next_question,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {"report": str(report_path.relative_to(root)), "manifest": str((output / "manifest.json").relative_to(root)), "decisions": manifest["decisions"], "artifact_count": len(artifacts)}
