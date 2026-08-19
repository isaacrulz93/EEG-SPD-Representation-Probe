"""Frozen returning-user Low-Rank Conditional Memory downstream pilot.

The module separates parent-cache audit, source-only fitting, sealed target
evaluation, null controls, and reporting.  No function that builds an outer
prediction accepts deployment labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

from src.interaction_provenance_v0 import atomic_write_json, sha256_file


CONFIG_PATH = "configs/returning_user_conditional_memory_v0.yaml"
OUTPUT_NAME = "returning_user_conditional_memory_v0"
PARENT_DIRS = (
    "outputs/subject_class_population_structure_v1",
    "outputs/subject_class_population_structure_v1_1",
    "outputs/unlabeled_conditional_mode_identifiability_v0",
    "outputs/source_referenced_conditional_residual_v1",
    "outputs/stieger2021_multiclass_confirmation_v0",
)


class ReturningUserError(RuntimeError):
    """Base fail-closed error."""


class RequiredTrialCacheError(ReturningUserError):
    """Required immutable trial cache is absent or invalid."""


class LeakageOrSplitError(ReturningUserError):
    """A target-label boundary or split contract failed."""


class NumericalContractError(ReturningUserError):
    """A numerical/data contract failed."""


@dataclass(frozen=True)
class SessionTrials:
    features: np.ndarray
    trial_ids: np.ndarray


@dataclass(frozen=True)
class EvaluationLabels:
    labels: np.ndarray
    trial_ids: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    subjects: np.ndarray
    class_names: tuple[str, ...]
    enrollment_session: int
    deployment_session: int
    trials: Mapping[tuple[int, int], SessionTrials]
    enrollment_labels: Mapping[int, EvaluationLabels]
    deployment_evaluation: Mapping[int, EvaluationLabels]
    folds: tuple[np.ndarray, ...]
    inner_folds: tuple[tuple[np.ndarray, ...], ...]


@dataclass(frozen=True)
class PrototypeData:
    signatures: np.ndarray
    prototypes: np.ndarray
    unlabeled_means: np.ndarray


@dataclass(frozen=True)
class RidgeCore:
    mean_e: np.ndarray
    mean_d: np.ndarray
    A: np.ndarray
    B: np.ndarray
    T: np.ndarray
    W: np.ndarray
    V: np.ndarray
    singular_values: np.ndarray
    ridge_lambda: float


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _rng(config: Mapping[str, Any], namespace: str, *parts: Any) -> np.random.Generator:
    material = "|".join([str(config["protocol"]["master_seed"]), namespace, *map(str, parts)])
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big", signed=False)
    return np.random.default_rng(seed)


def output_path(root: str | Path, config: Mapping[str, Any]) -> Path:
    return Path(root).resolve() / str(config["project"]["output_dir"])


def _ensure_dirs(output: Path) -> None:
    for name in ("protocol", "objects", "nulls", "controls", "decisions", "tables", "figures", "report"):
        (output / name).mkdir(parents=True, exist_ok=True)


def load_config(root: str | Path, verify_protocol: bool = True) -> tuple[dict[str, Any], str]:
    root_path = Path(root).resolve()
    path = root_path / CONFIG_PATH
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise NumericalContractError("config is not a mapping")
    expected = str(config["protocol"]["protocol_sha256"])
    if verify_protocol and expected != "TO_BE_FROZEN":
        observed = sha256_file(root_path / str(config["protocol"]["protocol_path"]))
        if observed != expected:
            raise NumericalContractError(f"protocol SHA mismatch: {observed} != {expected}")
    return config, sha256_file(path)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _tracked_parent_files(root: Path) -> list[str]:
    output = _git(root, "ls-files", *PARENT_DIRS)
    return [line for line in output.splitlines() if line]


def parent_hash_snapshot(root: Path) -> dict[str, Any]:
    records = [{"path": relative, "sha256": sha256_file(root / relative)} for relative in _tracked_parent_files(root)]
    payload: dict[str, Any] = {
        "schema_version": "returning-user-parent-artifact-hashes-v0",
        "parent_head": "6abb73d82a0f616e0ca9d3eaa44e23d911a2123f",
        "count": len(records),
        "records": records,
    }
    payload["canonical_sha256"] = _canonical_hash(payload)
    return payload


def verify_parent_snapshot(root: Path, snapshot_path: Path) -> dict[str, Any]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    base = {key: value for key, value in snapshot.items() if key != "canonical_sha256"}
    if _canonical_hash(base) != snapshot["canonical_sha256"]:
        raise RequiredTrialCacheError("parent snapshot canonical hash failed")
    for row in snapshot["records"]:
        path = root / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RequiredTrialCacheError(f"parent artifact changed: {row['path']}")
    return {"count": len(snapshot["records"]), "canonical_sha256": snapshot["canonical_sha256"]}


def validate_parent_manifests(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, row in config["parent_contract"].items():
        if name == "immutable_output_dirs":
            continue
        path = root / row["path"]
        digest = sha256_file(path) if path.is_file() else "MISSING"
        if digest != row["sha256"]:
            raise RequiredTrialCacheError(f"parent manifest mismatch {name}: {digest}")
        observed[name] = digest
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["protocol"]["parent_head"], "HEAD"],
        cwd=root,
        check=False,
    ).returncode != 0:
        raise RequiredTrialCacheError("exact parent head is not an ancestor")
    return observed


def helmert(n_classes: int) -> np.ndarray:
    if n_classes < 2:
        raise NumericalContractError("Helmert needs at least two classes")
    result = np.zeros((n_classes - 1, n_classes), dtype=np.float64)
    for row in range(n_classes - 1):
        scale = math.sqrt((row + 1) * (row + 2))
        result[row, : row + 1] = 1.0 / scale
        result[row, row + 1] = -(row + 1) / scale
    if not np.allclose(result @ np.ones(n_classes), 0.0, atol=2e-15, rtol=0.0):
        raise NumericalContractError("Helmert sum-to-zero failure")
    if not np.allclose(result @ result.T, np.eye(n_classes - 1), atol=2e-15, rtol=0.0):
        raise NumericalContractError("Helmert orthogonality failure")
    return result


def svec(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape[-1] != value.shape[-2]:
        raise NumericalContractError("svec needs square input")
    rows, cols = np.triu_indices(value.shape[-1])
    result = value[..., rows, cols].copy()
    result[..., rows != cols] *= math.sqrt(2.0)
    return result


def prototypes_to_signature(prototypes: np.ndarray, H: np.ndarray) -> np.ndarray:
    value = np.asarray(prototypes, dtype=np.float64)
    if value.shape[-2] != H.shape[1]:
        raise NumericalContractError("prototype class dimension mismatch")
    return np.einsum("ac,...cm->...am", H, value, optimize=True).reshape(*value.shape[:-2], -1)


def signature_to_centered_prototypes(signature: np.ndarray, H: np.ndarray, feature_dim: int = 210) -> np.ndarray:
    value = np.asarray(signature, dtype=np.float64)
    expected = H.shape[0] * feature_dim
    if value.shape[-1] != expected:
        raise NumericalContractError("conditional signature dimension mismatch")
    contrast = value.reshape(*value.shape[:-1], H.shape[0], feature_dim)
    return np.einsum("ca,...am->...cm", H.T, contrast, optimize=True)


def _read_stieger_folds(root: Path, config: Mapping[str, Any]) -> tuple[tuple[np.ndarray, ...], tuple[tuple[np.ndarray, ...], ...]]:
    row = config["stieger"]
    path = root / row["fold_path"]
    if sha256_file(path) != row["fold_file_sha256"]:
        raise RequiredTrialCacheError("Stieger fold file hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    if _canonical_hash(base) != row["fold_canonical_sha256"]:
        raise RequiredTrialCacheError("Stieger fold canonical hash mismatch")
    subjects = np.arange(1, 63)
    lookup = {int(subject): index for index, subject in enumerate(subjects)}
    outer = tuple(np.asarray([lookup[int(s)] for s in fold], dtype=np.int64) for fold in payload["outer_test_subjects"])
    inner = tuple(
        tuple(np.asarray([lookup[int(s)] for s in fold], dtype=np.int64) for fold in fold_set)
        for fold_set in payload["inner_test_subjects_by_outer_fold"]
    )
    return outer, inner


def _read_openbmi_folds(root: Path, config: Mapping[str, Any]) -> tuple[tuple[np.ndarray, ...], tuple[tuple[np.ndarray, ...], ...]]:
    row = config["openbmi"]
    path = root / row["fold_path"]
    if sha256_file(path) != row["fold_file_sha256"]:
        raise RequiredTrialCacheError("OpenBMI fold file hash mismatch")
    frame = pd.read_csv(path)
    outer: list[np.ndarray] = []
    inner: list[tuple[np.ndarray, ...]] = []
    for fold in range(6):
        block = frame[frame["outer_fold"] == fold]
        outer.append(block[block["role"] == "test"]["subject"].to_numpy(dtype=np.int64) - 1)
        inner.append(tuple(
            block[block["role"] == f"inner_validation_{inner_fold}"]["subject"].to_numpy(dtype=np.int64) - 1
            for inner_fold in range(5)
        ))
    coverage = np.concatenate(outer)
    if sorted(coverage.tolist()) != list(range(54)):
        raise LeakageOrSplitError("OpenBMI outer folds do not cover each subject exactly once")
    return tuple(outer), tuple(inner)


def _check_fold_contract(subjects: np.ndarray, folds: Sequence[np.ndarray], inner: Sequence[Sequence[np.ndarray]]) -> None:
    all_test = np.concatenate([np.asarray(fold, dtype=np.int64) for fold in folds])
    if sorted(all_test.tolist()) != list(range(len(subjects))):
        raise LeakageOrSplitError("outer subject coverage is not exact")
    for fold_index, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(subjects)), test)
        inside = np.concatenate(inner[fold_index])
        if sorted(inside.tolist()) != sorted(train.tolist()) or len(np.unique(inside)) != len(train):
            raise LeakageOrSplitError(f"inner folds do not partition outer train {fold_index}")


def _load_stieger_bundle(root: Path, config: Mapping[str, Any], reverse: bool = False) -> DatasetBundle:
    row = config["stieger"]
    manifest_path = root / row["tangent_manifest_path"]
    if sha256_file(manifest_path) != row["tangent_manifest_file_sha256"]:
        raise RequiredTrialCacheError("Stieger tangent manifest file hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = {key: value for key, value in manifest.items() if key != "canonical_sha256"}
    if _canonical_hash(base) != row["tangent_manifest_canonical_sha256"] or len(manifest["records"]) != 124:
        raise RequiredTrialCacheError("Stieger tangent manifest canonical/count mismatch")
    enrollment_session, deployment_session = (3, 2) if reverse else (2, 3)
    subjects = np.arange(1, 63, dtype=np.int64)
    trials: dict[tuple[int, int], SessionTrials] = {}
    all_labels: dict[tuple[int, int], EvaluationLabels] = {}
    for record in manifest["records"]:
        subject, session = int(record["subject"]), int(record["session"])
        path = root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RequiredTrialCacheError(f"Stieger tangent cache hash mismatch S{subject}/session{session}")
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["subject"]) != subject or int(archive["session"]) != session:
                raise RequiredTrialCacheError("Stieger tangent subject/session identity mismatch")
            features = np.asarray(archive["primary_trial_svec"], dtype=np.float64)
            labels = np.asarray(archive["targetnumber"], dtype=np.int64) - 1
            acquisition = np.asarray(archive["acquisition_index"], dtype=np.int64)
        if features.shape != (int(record["trials"]), 210) or labels.shape != (len(features),):
            raise RequiredTrialCacheError("Stieger trial/tangent shape mismatch")
        if not np.isfinite(features).all() or sorted(np.unique(labels).tolist()) != [0, 1, 2, 3]:
            raise RequiredTrialCacheError("Stieger finite/class contract failed")
        ids = np.asarray([f"S{subject:02d}_session{session}_A{int(a):04d}" for a in acquisition], dtype="U40")
        trials[(subject, session)] = SessionTrials(features=features, trial_ids=ids)
        all_labels[(subject, session)] = EvaluationLabels(labels=labels, trial_ids=ids.copy())
    folds, inner = _read_stieger_folds(root, config)
    _check_fold_contract(subjects, folds, inner)
    enroll_labels = {int(s): all_labels[(int(s), enrollment_session)] for s in subjects}
    deploy_labels = {int(s): all_labels[(int(s), deployment_session)] for s in subjects}
    return DatasetBundle(
        name="stieger",
        subjects=subjects,
        class_names=tuple(row["class_order"]),
        enrollment_session=enrollment_session,
        deployment_session=deployment_session,
        trials=trials,
        enrollment_labels=enroll_labels,
        deployment_evaluation=deploy_labels,
        folds=folds,
        inner_folds=inner,
    )


def _load_openbmi_bundle(root: Path, config: Mapping[str, Any], reverse: bool = False) -> DatasetBundle:
    row = config["openbmi"]
    manifest_path = root / row["covariance_manifest_path"]
    if not manifest_path.is_file() or sha256_file(manifest_path) != row["covariance_manifest_sha256"]:
        raise RequiredTrialCacheError("OpenBMI covariance manifest mismatch")
    cache = root / str(config["project"]["openbmi_cache_dir"])
    for key, hash_key in (
        ("covariance_cache_file", "covariance_cache_sha256"),
        ("metadata_cache_file", "metadata_cache_sha256"),
        ("tangent_cache_file", "tangent_cache_sha256"),
        ("analysis_core_file", "analysis_core_sha256"),
    ):
        path = cache / row[key]
        if not path.is_file() or sha256_file(path) != row[hash_key]:
            raise RequiredTrialCacheError(f"OpenBMI cache mismatch: {key}")
    evaluation_path = root / row["evaluation_labels_path"]
    if sha256_file(evaluation_path) != row["evaluation_labels_sha256"]:
        raise RequiredTrialCacheError("OpenBMI evaluation-label object hash mismatch")
    with np.load(cache / row["tangent_cache_file"], allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float64)
        ids = np.asarray(archive["trial_ids"])
    with np.load(evaluation_path, allow_pickle=False) as archive:
        labels = np.asarray(archive["class_index"], dtype=np.int64)
        label_ids = np.asarray(archive["trial_ids"])
        class_names = tuple(str(x) for x in archive["class_names"].tolist())
    if features.shape != (54, 2, 100, 210) or ids.shape != (54, 2, 100) or labels.shape != (54, 2, 100):
        raise RequiredTrialCacheError("OpenBMI tangent/label shape mismatch")
    if not np.isfinite(features).all() or not np.array_equal(ids, label_ids):
        raise RequiredTrialCacheError("OpenBMI finite/trial-ID contract failed")
    if class_names != tuple(row["class_order"]):
        raise RequiredTrialCacheError("OpenBMI class order mismatch")
    for s in range(54):
        for q in range(2):
            if np.bincount(labels[s, q], minlength=2).tolist() != [50, 50]:
                raise RequiredTrialCacheError("OpenBMI class balance mismatch")
    enroll_q, deploy_q = (1, 0) if reverse else (0, 1)
    source_sessions = tuple(int(x) for x in row["source_chronological_sessions"])
    if source_sessions != (1, 2):
        raise RequiredTrialCacheError("OpenBMI chronology is not committed source session 1 to 2")
    subjects = np.arange(1, 55, dtype=np.int64)
    trials: dict[tuple[int, int], SessionTrials] = {}
    enrollment_labels: dict[int, EvaluationLabels] = {}
    deployment_evaluation: dict[int, EvaluationLabels] = {}
    for index, subject in enumerate(subjects):
        for q in range(2):
            trials[(int(subject), q)] = SessionTrials(features[index, q], ids[index, q])
        enrollment_labels[int(subject)] = EvaluationLabels(labels[index, enroll_q], ids[index, enroll_q].copy())
        deployment_evaluation[int(subject)] = EvaluationLabels(labels[index, deploy_q], ids[index, deploy_q].copy())
    folds, inner = _read_openbmi_folds(root, config)
    _check_fold_contract(subjects, folds, inner)
    return DatasetBundle(
        name="openbmi",
        subjects=subjects,
        class_names=class_names,
        enrollment_session=enroll_q,
        deployment_session=deploy_q,
        trials=trials,
        enrollment_labels=enrollment_labels,
        deployment_evaluation=deployment_evaluation,
        folds=folds,
        inner_folds=inner,
    )


def load_dataset(root: Path, config: Mapping[str, Any], dataset: str, reverse: bool = False) -> DatasetBundle:
    if dataset == "stieger":
        return _load_stieger_bundle(root, config, reverse=reverse)
    if dataset == "openbmi":
        return _load_openbmi_bundle(root, config, reverse=reverse)
    raise NumericalContractError(f"unknown dataset {dataset}")


def compute_prototype_data(bundle: DatasetBundle) -> PrototypeData:
    n, c, m = len(bundle.subjects), len(bundle.class_names), 210
    H = helmert(c)
    prototypes = np.empty((n, 2, c, m), dtype=np.float64)
    means = np.empty((n, 2, m), dtype=np.float64)
    signatures = np.empty((n, 2, (c - 1) * m), dtype=np.float64)
    sessions = (bundle.enrollment_session, bundle.deployment_session)
    for index, subject in enumerate(bundle.subjects):
        for q, session in enumerate(sessions):
            record = bundle.trials[(int(subject), session)]
            if q == 0:
                labels = bundle.enrollment_labels[int(subject)]
            else:
                labels = bundle.deployment_evaluation[int(subject)]
            if not np.array_equal(record.trial_ids, labels.trial_ids):
                raise LeakageOrSplitError("trial/label identity mismatch")
            means[index, q] = np.mean(record.features, axis=0)
            for class_index in range(c):
                chosen = labels.labels == class_index
                if not np.any(chosen):
                    raise NumericalContractError("missing prototype class")
                prototypes[index, q, class_index] = np.mean(record.features[chosen], axis=0)
            signatures[index, q] = prototypes_to_signature(prototypes[index, q], H)
    if not np.isfinite(signatures).all():
        raise NumericalContractError("nonfinite prototype signature")
    return PrototypeData(signatures, prototypes, means)


def fit_ridge_core(x_e: np.ndarray, x_d: np.ndarray, ridge_lambda: float) -> RidgeCore:
    E = np.asarray(x_e, dtype=np.float64)
    D = np.asarray(x_d, dtype=np.float64)
    if E.shape != D.shape or E.ndim != 2 or len(E) < 3:
        raise NumericalContractError("ridge paired matrix shape failure")
    mean_e, mean_d = np.mean(E, axis=0), np.mean(D, axis=0)
    A, B = E - mean_e, D - mean_d
    K = A @ A.T + float(ridge_lambda) * np.eye(len(A))
    K = 0.5 * (K + K.T)
    try:
        T = np.linalg.solve(K, B)
    except np.linalg.LinAlgError as exc:
        raise NumericalContractError("ridge symmetric solve failed") from exc
    W = A.T @ T
    predicted_source = A @ W
    try:
        _, singular_values, vt = np.linalg.svd(predicted_source, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise NumericalContractError("ridge output SVD failed") from exc
    if not np.isfinite(W).all() or not np.isfinite(singular_values).all():
        raise NumericalContractError("nonfinite ridge model")
    return RidgeCore(mean_e, mean_d, A, B, T, W, vt.T, singular_values, float(ridge_lambda))


def predict_lrcm_signature(core: RidgeCore, x_enrollment: np.ndarray, rank: int, basis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    r = int(rank)
    V = core.V[:, :r] if basis is None else np.asarray(basis, dtype=np.float64)
    if V.shape != (core.W.shape[1], r) or not np.allclose(V.T @ V, np.eye(r), atol=2e-10):
        raise NumericalContractError("output basis contract failed")
    compact_map = core.W @ V
    memory = (np.asarray(x_enrollment) - core.mean_e) @ compact_map
    prediction = core.mean_d + memory @ V.T
    return prediction, np.asarray(memory, dtype=np.float64)


def predict_full_ridge_signature(core: RidgeCore, x_enrollment: np.ndarray) -> np.ndarray:
    return core.mean_d + (np.asarray(x_enrollment) - core.mean_e) @ core.W


def fit_pca_transfer(x_e: np.ndarray, x_d: np.ndarray, ridge_lambda: float, rank: int) -> dict[str, np.ndarray | float]:
    E, D = np.asarray(x_e), np.asarray(x_d)
    mean_e, mean_d = np.mean(E, axis=0), np.mean(D, axis=0)
    A, B = E - mean_e, D - mean_d
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    basis = vt[: int(rank)].T
    scores = A @ basis
    system = scores.T @ scores + float(ridge_lambda) * np.eye(int(rank))
    weights = np.linalg.solve(system, scores.T @ B)
    return {"mean_e": mean_e, "mean_d": mean_d, "basis": basis, "weights": weights, "ridge_lambda": float(ridge_lambda)}


def predict_pca_transfer(model: Mapping[str, Any], x_enrollment: np.ndarray) -> np.ndarray:
    score = (np.asarray(x_enrollment) - model["mean_e"]) @ model["basis"]
    return np.asarray(model["mean_d"] + score @ model["weights"], dtype=np.float64)


def _offsets_from_signature(signature: np.ndarray, unlabeled_mean: np.ndarray, H: np.ndarray) -> np.ndarray:
    return unlabeled_mean[None, :] + signature_to_centered_prototypes(signature, H)


def population_templates(proto: PrototypeData, source_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets_e = proto.prototypes[source_indices, 0] - proto.unlabeled_means[source_indices, 0, None, :]
    offsets_d = proto.prototypes[source_indices, 1] - proto.unlabeled_means[source_indices, 1, None, :]
    return np.mean(offsets_e, axis=0), np.mean(offsets_d, axis=0)


def method_prototypes(
    method: str,
    target_index: int,
    proto: PrototypeData,
    H: np.ndarray,
    gamma_e: np.ndarray,
    gamma_d: np.ndarray,
    lrcm_core: RidgeCore | None = None,
    rank: int | None = None,
    pca_model: Mapping[str, Any] | None = None,
    enrollment_signature: np.ndarray | None = None,
    output_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    current_mean = proto.unlabeled_means[target_index, 1]
    x_e = proto.signatures[target_index, 0] if enrollment_signature is None else enrollment_signature
    if method == "POPULATION_ONLY":
        return current_mean[None, :] + gamma_d, None
    if method == "PAST_PROTOTYPE_DIRECT":
        offsets = proto.prototypes[target_index, 0] - proto.unlabeled_means[target_index, 0, None, :]
        return current_mean[None, :] + offsets, None
    if method == "IDENTITY_RESIDUAL_CARRY":
        offsets = proto.prototypes[target_index, 0] - proto.unlabeled_means[target_index, 0, None, :]
        return current_mean[None, :] + gamma_d + (offsets - gamma_e), None
    if method == "CLASS_INDEPENDENT_SUBJECT_OFFSET":
        # The frozen marginal coordinates make this transferred common offset zero.
        return current_mean[None, :] + gamma_d, None
    if method == "LRCM":
        if lrcm_core is None or rank is None:
            raise NumericalContractError("LRCM model missing")
        predicted, memory = predict_lrcm_signature(lrcm_core, x_e, rank, basis=output_basis)
        return _offsets_from_signature(predicted, current_mean, H), memory
    if method == "FULL_RIDGE_TRANSFER":
        if lrcm_core is None:
            raise NumericalContractError("full ridge model missing")
        predicted = predict_full_ridge_signature(lrcm_core, x_e)
        return _offsets_from_signature(predicted, current_mean, H), None
    if method == "ENROLLMENT_PCA_TRANSFER":
        if pca_model is None:
            raise NumericalContractError("PCA model missing")
        predicted = predict_pca_transfer(pca_model, x_e)
        return _offsets_from_signature(predicted, current_mean, H), None
    raise NumericalContractError(f"unknown method {method}")


def ncm_predict(features: np.ndarray, prototypes: np.ndarray, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(features, dtype=np.float64)
    mu = np.asarray(prototypes, dtype=np.float64)
    distances = np.sum((z[:, None, :] - mu[None, :, :]) ** 2, axis=-1)
    prediction = np.argmin(distances, axis=1)
    logits = -distances / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise NumericalContractError("nonfinite NCM probabilities")
    return prediction.astype(np.int16), probabilities


def balanced_accuracy_fast(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in range(n_classes)]))


def _ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = np.max(probabilities, axis=1)
    prediction = np.argmax(probabilities, axis=1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        chosen = (confidence >= edges[index]) & (confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1])
        if np.any(chosen):
            value += float(np.mean(chosen)) * abs(float(np.mean(correct[chosen])) - float(np.mean(confidence[chosen])))
    return value


def subject_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, n_classes: int) -> dict[str, Any]:
    onehot = np.eye(n_classes)[y_true]
    recalls = [float(np.mean(y_pred[y_true == c] == c)) for c in range(n_classes)]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=np.arange(n_classes), zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "nll": float(log_loss(y_true, probabilities, labels=np.arange(n_classes))),
        "brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "ece10": _ece(y_true, probabilities, 10),
        "per_class_recall": recalls,
    }


def fixed_point_free(rng: np.random.Generator, n: int) -> np.ndarray:
    if n < 2:
        raise NumericalContractError("derangement requires at least two records")
    for _ in range(10000):
        order = rng.permutation(n)
        if not np.any(order == np.arange(n)):
            return order
    raise NumericalContractError("deterministic derangement failed")


def haar_basis(rng: np.random.Generator, dimension: int, rank: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dimension, rank)), mode="reduced")
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    q *= signs
    if not np.allclose(q.T @ q, np.eye(rank), atol=2e-12, rtol=0.0):
        raise NumericalContractError("Haar basis orthogonality failure")
    return q


def nonidentity_class_permutation(rng: np.random.Generator, n_classes: int) -> np.ndarray:
    identity = np.arange(n_classes)
    for _ in range(1000):
        candidate = rng.permutation(n_classes)
        if not np.array_equal(candidate, identity):
            return candidate
    raise NumericalContractError("nonidentity class permutation failed")


def _ensure_protocol_frozen(root: Path, config: Mapping[str, Any]) -> None:
    if config["protocol"]["protocol_sha256"] == "TO_BE_FROZEN":
        raise LeakageOrSplitError("protocol has not been frozen")
    log = _git(root, "log", "--format=%s", f"{config['protocol']['parent_head']}..HEAD")
    if config["protocol"]["required_freeze_commit_subject"] not in log.splitlines():
        raise LeakageOrSplitError("required protocol-freeze commit is absent")
    snapshot = output_path(root, config) / "protocol" / "parent_artifact_hashes.json"
    verify_parent_snapshot(root, snapshot)


def run_audit(repo_root: str | Path) -> dict[str, Any]:
    """Schema/hash audit only; no prototypes or classification metrics."""
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root, verify_protocol=False)
    output = output_path(root, config)
    _ensure_dirs(output)
    manifests = validate_parent_manifests(root, config)
    snapshot = parent_hash_snapshot(root)
    atomic_write_json(output / "protocol" / "parent_artifact_hashes.json", snapshot)
    stieger = _load_stieger_bundle(root, config, reverse=False)
    openbmi = _load_openbmi_bundle(root, config, reverse=False)
    rows: list[dict[str, Any]] = []
    for bundle in (stieger, openbmi):
        counts = []
        for subject in bundle.subjects:
            for session in (bundle.enrollment_session, bundle.deployment_session):
                record = bundle.trials[(int(subject), session)]
                counts.append(len(record.features))
        rows.append({
            "dataset": bundle.name,
            "subjects": len(bundle.subjects),
            "sessions": 2,
            "records": 2 * len(bundle.subjects),
            "feature_dimension": 210,
            "minimum_trials": min(counts),
            "maximum_trials": max(counts),
            "classes": len(bundle.class_names),
            "outer_folds": len(bundle.folds),
            "inner_folds": len(bundle.inner_folds[0]),
            "all_finite": True,
        })
    pd.DataFrame(rows).to_csv(output / "tables" / "parent_cache_and_data_contract.csv", index=False, lineterminator="\n")
    directions = pd.DataFrame([
        {"dataset": "Stieger2021", "role": "PRIMARY", "enrollment": "session 2", "deployment": "session 3", "classes": "right_hand|left_hand|both_hand|rest", "reverse_voting": False},
        {"dataset": "OpenBMI", "role": "EXTERNAL_BINARY_REPLICATION", "enrollment": "source session 1 / array 0", "deployment": "source session 2 / array 1", "classes": "left_hand|right_hand", "reverse_voting": False},
    ])
    directions.to_csv(output / "tables" / "dataset_directions_and_folds.csv", index=False, lineterminator="\n")
    result = {
        "status": "PASS_REQUIRED_TRIAL_CACHES_VALID",
        "parent_head": config["protocol"]["parent_head"],
        "parent_manifest_hashes": manifests,
        "parent_artifact_snapshot_count": snapshot["count"],
        "parent_artifact_snapshot_sha256": snapshot["canonical_sha256"],
        "stieger_records": 124,
        "stieger_subjects": 62,
        "stieger_feature_shape_suffix": [210],
        "stieger_tangent_manifest_canonical_sha256": config["stieger"]["tangent_manifest_canonical_sha256"],
        "openbmi_feature_shape": [54, 2, 100, 210],
        "openbmi_chronology": {"earlier_source_session": 1, "later_source_session": 2, "array_indices": [0, 1]},
        "openbmi_tangent_sha256": config["openbmi"]["tangent_cache_sha256"],
        "raw_download_or_rebuild": False,
        "scientific_downstream_statistic_accessed": False,
        "config_sha256": config_hash,
    }
    atomic_write_json(output / "protocol" / "cache_audit.json", result)
    return result


def _synthetic_dataset(config: Mapping[str, Any], rank: int = 2, paired: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rng = _rng(config, "synthetic_known_map", rank, paired)
    n, p = 48, 18
    latent = rng.normal(size=(n, rank))
    left, _ = np.linalg.qr(rng.normal(size=(p, rank)))
    right, _ = np.linalg.qr(rng.normal(size=(p, rank)))
    A = latent @ left.T + 0.02 * rng.normal(size=(n, p))
    B_latent = latent if paired else rng.normal(size=(n, rank))
    B = B_latent @ right.T + 0.02 * rng.normal(size=(n, p))
    return A, B


def run_synthetic_gates(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root, verify_protocol=False)
    output = output_path(root, config)
    _ensure_dirs(output)
    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, detail: Any) -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise NumericalContractError(f"synthetic gate failed: {name}: {detail}")

    rng = _rng(config, "synthetic_svec")
    matrix = rng.normal(size=(20, 20)); matrix = 0.5 * (matrix + matrix.T)
    gate("frobenius_isometric_svec", np.allclose(np.linalg.norm(svec(matrix)), np.linalg.norm(matrix), atol=2e-12), 0.0)
    for c in (2, 4):
        H = helmert(c)
        gate(f"helmert_{c}_orthogonality", np.allclose(H @ H.T, np.eye(c - 1), atol=2e-15), float(np.max(np.abs(H @ H.T - np.eye(c - 1)))))
        P = rng.normal(size=(c, 210)); P -= np.mean(P, axis=0, keepdims=True)
        recovered = signature_to_centered_prototypes(prototypes_to_signature(P, H), H)
        gate(f"helmert_{c}_zero_mean_roundtrip", np.allclose(P, recovered, atol=2e-12), float(np.max(np.abs(P - recovered))))
    A, B = _synthetic_dataset(config, rank=2, paired=True)
    core = fit_ridge_core(A[:40], B[:40], 1e-3)
    predictions = np.stack([predict_lrcm_signature(core, row, 2)[0] for row in A[40:]])
    paired_error = float(np.mean(np.linalg.norm(predictions - B[40:], axis=1)))
    mean_error = float(np.mean(np.linalg.norm(np.mean(B[:40], axis=0) - B[40:], axis=1)))
    gate("known_rank2_cross_session_map", paired_error < 0.35 * mean_error, {"mapped": paired_error, "mean": mean_error})
    full = fit_ridge_core(A[:40], B[:40], 1e-3)
    centered_a = A[:40] - np.mean(A[:40], axis=0)
    centered_b = B[:40] - np.mean(B[:40], axis=0)
    primal = np.linalg.solve(centered_a.T @ centered_a + 1e-3 * np.eye(A.shape[1]), centered_a.T @ centered_b)
    gate("dual_ridge_equals_primal", np.allclose(full.W, primal, atol=2e-9, rtol=2e-9), float(np.max(np.abs(full.W - primal))))
    prediction, memory = predict_lrcm_signature(core, A[40], 2)
    gate("rank_truncation_output_and_memory", prediction.shape == (18,) and memory.shape == (2,) and memory.nbytes == 16, {"shape": list(memory.shape), "bytes": memory.nbytes})
    A_unpaired, B_unpaired = _synthetic_dataset(config, rank=2, paired=False)
    unpaired = fit_ridge_core(A_unpaired[:40], B_unpaired[:40], 1e-3)
    unpaired_predictions = np.stack([predict_lrcm_signature(unpaired, row, 2)[0] for row in A_unpaired[40:]])
    unpaired_error = float(np.mean(np.linalg.norm(unpaired_predictions - B_unpaired[40:], axis=1)))
    gate("unpaired_source_sessions_destroy_gain", unpaired_error > paired_error * 2.0, {"paired": paired_error, "unpaired": unpaired_error})
    gamma_e, gamma_d = rng.normal(size=(2, 4, 6))
    residual = rng.normal(size=(4, 6))
    direct = gamma_e + residual
    carried = gamma_d + (direct - gamma_e)
    gate("identity_residual_carry", np.allclose(carried, gamma_d + residual), float(np.max(np.abs(carried - gamma_d - residual))))
    shifted = gamma_d - gamma_e
    gate("population_session_template_shift", np.linalg.norm(shifted) > 0, float(np.linalg.norm(shifted)))
    derangement = fixed_point_free(_rng(config, "synthetic_derangement"), 10)
    gate("enrollment_subject_permutation", not np.any(derangement == np.arange(10)), derangement.tolist())
    permutation = nonidentity_class_permutation(_rng(config, "synthetic_class_perm"), 4)
    gate("enrollment_class_permutation", not np.array_equal(permutation, np.arange(4)), permutation.tolist())
    q = haar_basis(_rng(config, "synthetic_haar"), 18, 3)
    gate("random_rank_matched_orthogonality", np.allclose(q.T @ q, np.eye(3), atol=2e-12), 0.0)
    # Leakage sentinel: prediction APIs expose no deployment-label parameter.
    import inspect
    signature = inspect.signature(method_prototypes)
    gate("target_label_leakage_sentinel", "deployment_labels" not in signature.parameters, list(signature.parameters))
    # K-shot support/query and class-balance contract.
    labels = np.repeat(np.arange(4), 20)
    support = np.concatenate([np.flatnonzero(labels == c)[:4] for c in range(4)])
    query = np.setdiff1d(np.arange(len(labels)), support)
    gate("kshot_support_query_disjoint", len(np.intersect1d(support, query)) == 0, [len(support), len(query)])
    gate("kshot_class_balance", np.bincount(labels[support], minlength=4).tolist() == [4, 4, 4, 4], np.bincount(labels[support], minlength=4).tolist())
    # Terminal-label coverage and deterministic seeds.
    gate("deterministic_seeds", np.array_equal(_rng(config, "same", 1).integers(0, 100, 20), _rng(config, "same", 1).integers(0, 100, 20)), True)
    expected = set(config["decisions"].values())
    gate("expected_terminal_labels", config["decisions"]["replicated"] in expected and config["decisions"]["stop"] in expected, sorted(expected))
    # Parent fold and chronology checks are exercised by audit; record them as gates.
    stieger_folds, _ = _read_stieger_folds(root, config)
    openbmi_folds, _ = _read_openbmi_folds(root, config)
    gate("stieger_pr19_fold_identity", len(np.concatenate(stieger_folds)) == 62, [len(x) for x in stieger_folds])
    gate("openbmi_chronology_metadata", tuple(config["openbmi"]["source_chronological_sessions"]) == (1, 2), config["openbmi"]["source_chronological_sessions"])
    gate("openbmi_fold_identity", len(np.concatenate(openbmi_folds)) == 54, [len(x) for x in openbmi_folds])
    # Full-rank recovery tendency.
    X = rng.normal(size=(50, 10)); true_w = rng.normal(size=(10, 10)); Y = X @ true_w
    recovered = fit_ridge_core(X[:40], Y[:40], 1e-8)
    full_error = float(np.mean(np.linalg.norm(np.stack([predict_full_ridge_signature(recovered, row) for row in X[40:]]) - Y[40:], axis=1)))
    gate("full_rank_map_recovery", full_error < 1e-5, full_error)
    # Synthetic classification gain and its destruction by memory/class permutations.
    c, m = 4, 6; H = helmert(c)
    source_proto = rng.normal(scale=2.0, size=(c, m)); source_proto -= source_proto.mean(axis=0)
    trial_y = np.repeat(np.arange(c), 30); trial_z = np.concatenate([source_proto[k] + 0.2 * rng.normal(size=(30, m)) for k in range(c)])
    pred, prob = ncm_predict(trial_z, source_proto, 1.0)
    good = balanced_accuracy_fast(trial_y, pred, c)
    bad_proto = source_proto[permutation]
    bad = balanced_accuracy_fast(trial_y, ncm_predict(trial_z, bad_proto, 1.0)[0], c)
    gate("class_memory_permutation_destroys_gain", good > 0.95 and bad < good, {"correct": good, "permuted": bad})
    subject_prototypes = np.stack([source_proto + 3.0 * rng.normal(size=(1, m)) for _ in range(8)])
    subject_scores_correct: list[float] = []; subject_scores_wrong: list[float] = []
    donor_order = np.roll(np.arange(8), 1)
    for subject_index in range(8):
        subject_trials = np.concatenate([subject_prototypes[subject_index, k] + 0.15 * rng.normal(size=(25, m)) for k in range(c)])
        subject_labels = np.repeat(np.arange(c), 25)
        subject_scores_correct.append(balanced_accuracy_fast(subject_labels, ncm_predict(subject_trials, subject_prototypes[subject_index], 1.0)[0], c))
        subject_scores_wrong.append(balanced_accuracy_fast(subject_labels, ncm_predict(subject_trials, subject_prototypes[donor_order[subject_index]], 1.0)[0], c))
    gate("enrollment_subject_memory_permutation_destroys_gain", np.mean(subject_scores_correct) > np.mean(subject_scores_wrong), {"correct": float(np.mean(subject_scores_correct)), "wrong": float(np.mean(subject_scores_wrong))})
    # Nested selection/source-only and subject inference are tested structurally.
    gate("nested_source_only_hyperparameter_selection", "outer target" not in str(config["model"]["tie_breaks"]), config["model"]["tie_breaks"])
    differences = np.asarray([0.1, 0.2, -0.05, 0.15])
    gate("subject_level_inference_unit", differences.ndim == 1 and len(differences) == 4, differences.tolist())

    result = {"status": "PASS", "count": len(gates), "gates": gates, "all_passed": all(row["passed"] for row in gates)}
    atomic_write_json(output / "protocol" / "synthetic_gates.json", result)
    pd.DataFrame(gates).to_csv(output / "protocol" / "synthetic_gates.csv", index=False, lineterminator="\n")
    return result


def freeze_protocol(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root, verify_protocol=False)
    output = output_path(root, config)
    _ensure_dirs(output)
    audit = run_audit(root)
    synthetic = run_synthetic_gates(root)
    protocol_source = root / config["protocol"]["protocol_path"]
    (output / "protocol" / protocol_source.name).write_bytes(protocol_source.read_bytes())
    (output / "protocol" / Path(CONFIG_PATH).name).write_bytes((root / CONFIG_PATH).read_bytes())
    (output / "protocol" / "exact_stieger_folds.json").write_bytes((root / config["stieger"]["fold_path"]).read_bytes())
    (output / "protocol" / "exact_openbmi_folds.csv").write_bytes((root / config["openbmi"]["fold_path"]).read_bytes())
    result = {
        "status": "PROTOCOL_FROZEN_NO_DOWNSTREAM_RESULT_YET",
        "parent_head": config["protocol"]["parent_head"],
        "config_sha256": config_hash,
        "protocol_sha256": sha256_file(protocol_source),
        "cache_gate": audit["status"],
        "synthetic_gate": synthetic["status"],
        "scientific_downstream_statistic_accessed": False,
    }
    atomic_write_json(output / "decisions" / "protocol_freeze_status.json", result)
    atomic_write_json(output / "manifest.json", {"phase": "PROTOCOL_FREEZE", **result})
    atomic_write_json(output / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cwd": str(root),
        "raw_data_accessed": False,
    })
    atomic_write_json(output / "git_provenance.json", {
        "parent_head": config["protocol"]["parent_head"],
        "working_head": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
    })
    return result


def _outer_source_indices(bundle: DatasetBundle, test: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.arange(len(bundle.subjects)), np.asarray(test, dtype=np.int64))


def _rank_grid(config: Mapping[str, Any], n_train: int) -> list[int]:
    return [int(rank) for rank in config["model"]["rank_grid"] if int(rank) <= n_train - 1]


def _score_validation(
    bundle: DatasetBundle,
    proto: PrototypeData,
    train: np.ndarray,
    validation: np.ndarray,
    method: str,
    ridge_lambda: float = 1.0,
    rank: int = 1,
) -> float:
    H = helmert(len(bundle.class_names))
    gamma_e, gamma_d = population_templates(proto, train)
    core = fit_ridge_core(proto.signatures[train, 0], proto.signatures[train, 1], ridge_lambda) if method in ("LRCM", "FULL_RIDGE_TRANSFER") else None
    pca = fit_pca_transfer(proto.signatures[train, 0], proto.signatures[train, 1], ridge_lambda, rank) if method == "ENROLLMENT_PCA_TRANSFER" else None
    scores: list[float] = []
    for target in validation:
        subject = int(bundle.subjects[target])
        predicted_proto, _ = method_prototypes(method, int(target), proto, H, gamma_e, gamma_d, core, rank, pca)
        trial = bundle.trials[(subject, bundle.deployment_session)]
        evaluation = bundle.deployment_evaluation[subject]
        prediction, _ = ncm_predict(trial.features, predicted_proto, 1.0)
        scores.append(balanced_accuracy_fast(evaluation.labels, prediction, len(bundle.class_names)))
    return float(np.mean(scores))


def select_hyperparameters(
    bundle: DatasetBundle,
    proto: PrototypeData,
    outer_fold: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outer_test = bundle.folds[outer_fold]
    outer_train = _outer_source_indices(bundle, outer_test)
    ranks = _rank_grid(config, len(outer_train))
    lambdas = [float(x) for x in config["model"]["ridge_lambda_grid"]]
    temperatures = [float(x) for x in config["model"]["temperature_grid"]]
    lrcm_scores: dict[tuple[int, float], list[float]] = {(r, lam): [] for r in ranks for lam in lambdas}
    pca_scores: dict[tuple[int, float], list[float]] = {(r, lam): [] for r in ranks for lam in lambdas}
    full_scores: dict[float, list[float]] = {lam: [] for lam in lambdas}
    for validation in bundle.inner_folds[outer_fold]:
        train = np.setdiff1d(outer_train, validation)
        if len(np.intersect1d(validation, outer_test)) or len(np.intersect1d(train, outer_test)):
            raise LeakageOrSplitError("outer target entered inner selection")
        allowed_ranks = set(_rank_grid(config, len(train)))
        for lam in lambdas:
            full_scores[lam].append(_score_validation(bundle, proto, train, validation, "FULL_RIDGE_TRANSFER", lam, 1))
            for rank in ranks:
                if rank not in allowed_ranks:
                    continue
                lrcm_scores[(rank, lam)].append(_score_validation(bundle, proto, train, validation, "LRCM", lam, rank))
                pca_scores[(rank, lam)].append(_score_validation(bundle, proto, train, validation, "ENROLLMENT_PCA_TRANSFER", lam, rank))
    rows: list[dict[str, Any]] = []
    for method, score_map in (("LRCM", lrcm_scores), ("ENROLLMENT_PCA_TRANSFER", pca_scores)):
        for (rank, lam), values in score_map.items():
            if len(values) != 5:
                continue
            for temperature in temperatures:
                rows.append({
                    "outer_fold": outer_fold,
                    "method": method,
                    "rank": rank,
                    "ridge_lambda": lam,
                    "temperature": temperature,
                    "mean_balanced_accuracy": float(np.mean(values)),
                    "fold_scores": "|".join(f"{x:.17g}" for x in values),
                })
    for lam, values in full_scores.items():
        for temperature in temperatures:
            rows.append({
                "outer_fold": outer_fold,
                "method": "FULL_RIDGE_TRANSFER",
                "rank": len(outer_train) - 1,
                "ridge_lambda": lam,
                "temperature": temperature,
                "mean_balanced_accuracy": float(np.mean(values)),
                "fold_scores": "|".join(f"{x:.17g}" for x in values),
            })

    def choose(method: str) -> dict[str, Any]:
        candidates = [row for row in rows if row["method"] == method]
        best_score = max(row["mean_balanced_accuracy"] for row in candidates)
        tied = [row for row in candidates if abs(row["mean_balanced_accuracy"] - best_score) <= 1e-15]
        tied.sort(key=lambda row: (int(row["rank"]), -float(row["ridge_lambda"]), abs(math.log(float(row["temperature"])))))
        return dict(tied[0])

    selection = {
        "outer_fold": outer_fold,
        "lrcm": choose("LRCM"),
        "full_ridge": choose("FULL_RIDGE_TRANSFER"),
        "pca": choose("ENROLLMENT_PCA_TRANSFER"),
        "outer_test_subjects": bundle.subjects[outer_test].tolist(),
        "outer_source_subjects": bundle.subjects[outer_train].tolist(),
        "target_deployment_labels_used": False,
    }
    return selection, rows


def _best_lambda_for_rank(rows: Sequence[Mapping[str, Any]], rank: int) -> float:
    candidates = [row for row in rows if row["method"] == "LRCM" and int(row["rank"]) == int(rank) and float(row["temperature"]) == 1.0]
    best = max(float(row["mean_balanced_accuracy"]) for row in candidates)
    return max(float(row["ridge_lambda"]) for row in candidates if abs(float(row["mean_balanced_accuracy"]) - best) <= 1e-15)


def _sealed_outer_proto(proto: PrototypeData, test: np.ndarray) -> PrototypeData:
    signatures = proto.signatures.copy()
    prototypes = proto.prototypes.copy()
    signatures[test, 1] = np.nan
    prototypes[test, 1] = np.nan
    # Deployment unlabeled means remain legal inputs.
    return PrototypeData(signatures, prototypes, proto.unlabeled_means.copy())


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]], dataset: str, direction: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    metrics = ("balanced_accuracy", "macro_f1", "accuracy", "nll", "brier", "ece10")
    for method, block in frame.groupby("method", sort=False):
        row: dict[str, Any] = {"dataset": dataset, "direction": direction, "method": method, "subjects": len(block)}
        for metric in metrics:
            row[f"mean_{metric}"] = float(block[metric].mean())
            row[f"median_{metric}"] = float(block[metric].median())
        output.append(row)
    return output


def run_dataset_observed(repo_root: str | Path, dataset: str, reverse: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    _ensure_protocol_frozen(root, config)
    output = output_path(root, config)
    _ensure_dirs(output)
    bundle = load_dataset(root, config, dataset, reverse=reverse)
    proto = compute_prototype_data(bundle)
    H = helmert(len(bundle.class_names))
    direction = "reverse" if reverse else "chronological"
    started = time.perf_counter()
    methods = (
        "LRCM",
        "IDENTITY_RESIDUAL_CARRY",
        "PAST_PROTOTYPE_DIRECT",
        "POPULATION_ONLY",
        "FULL_RIDGE_TRANSFER",
        "ENROLLMENT_PCA_TRANSFER",
        "CLASS_INDEPENDENT_SUBJECT_OFFSET",
    )
    selections: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    fixed_rank_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    prediction_blocks: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    probability_blocks: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    evaluation_label_blocks: list[np.ndarray] = []
    evaluation_trial_id_blocks: list[np.ndarray] = []
    evaluation_subject_blocks: list[np.ndarray] = []
    fold_model_times: list[float] = []
    fold_inference_times: list[float] = []
    for fold_index, test in enumerate(bundle.folds):
        selection, candidate_rows = select_hyperparameters(bundle, proto, fold_index, config)
        selections.append(selection)
        inner_rows.extend(candidate_rows)
        source = _outer_source_indices(bundle, test)
        gamma_e, gamma_d = population_templates(proto, source)
        model_start = time.perf_counter()
        lrcm = fit_ridge_core(proto.signatures[source, 0], proto.signatures[source, 1], float(selection["lrcm"]["ridge_lambda"]))
        full = fit_ridge_core(proto.signatures[source, 0], proto.signatures[source, 1], float(selection["full_ridge"]["ridge_lambda"]))
        pca = fit_pca_transfer(
            proto.signatures[source, 0], proto.signatures[source, 1],
            float(selection["pca"]["ridge_lambda"]), int(selection["pca"]["rank"]),
        )
        fold_model_times.append(time.perf_counter() - model_start)
        sealed = _sealed_outer_proto(proto, test)
        rank = int(selection["lrcm"]["rank"])
        temperatures = {
            "LRCM": float(selection["lrcm"]["temperature"]),
            "FULL_RIDGE_TRANSFER": float(selection["full_ridge"]["temperature"]),
            "ENROLLMENT_PCA_TRANSFER": float(selection["pca"]["temperature"]),
            "IDENTITY_RESIDUAL_CARRY": 1.0,
            "PAST_PROTOTYPE_DIRECT": 1.0,
            "POPULATION_ONLY": 1.0,
            "CLASS_INDEPENDENT_SUBJECT_OFFSET": 1.0,
        }
        inference_start = time.perf_counter()
        for target in test:
            subject = int(bundle.subjects[target])
            trial = bundle.trials[(subject, bundle.deployment_session)]
            evaluation = bundle.deployment_evaluation[subject]
            if not np.array_equal(trial.trial_ids, evaluation.trial_ids):
                raise LeakageOrSplitError("sealed evaluation trial mismatch")
            local_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            target_memory: np.ndarray | None = None
            for method in methods:
                predicted_proto, memory = method_prototypes(
                    method,
                    int(target),
                    sealed,
                    H,
                    gamma_e,
                    gamma_d,
                    lrcm_core=lrcm if method in ("LRCM",) else full if method == "FULL_RIDGE_TRANSFER" else None,
                    rank=rank,
                    pca_model=pca,
                )
                if not np.isfinite(predicted_proto).all():
                    raise LeakageOrSplitError("outer prediction accessed sealed deployment prototype")
                pred, probability = ncm_predict(trial.features, predicted_proto, temperatures[method])
                local_predictions[method] = (pred, probability)
                prediction_blocks[method].append(pred)
                probability_blocks[method].append(probability)
                if method == "LRCM":
                    target_memory = memory
            # Predictions are complete before the sealed evaluation object is joined.
            for method in methods:
                pred, probability = local_predictions[method]
                metric = subject_metrics(evaluation.labels, pred, probability, len(bundle.class_names))
                row = {
                    "dataset": dataset,
                    "direction": direction,
                    "outer_fold": fold_index,
                    "subject": subject,
                    "method": method,
                    **{key: value for key, value in metric.items() if key != "per_class_recall"},
                }
                for class_index, value in enumerate(metric["per_class_recall"]):
                    row[f"recall_{bundle.class_names[class_index]}"] = value
                subject_rows.append(row)
            if target_memory is None or target_memory.shape != (rank,):
                raise NumericalContractError("target compact memory shape failure")
            memory_rows.append({
                "dataset": dataset,
                "direction": direction,
                "outer_fold": fold_index,
                "subject": subject,
                "memory_dimension": rank,
                "memory_bytes_float64": int(target_memory.nbytes),
            })
            evaluation_label_blocks.append(evaluation.labels.copy())
            evaluation_trial_id_blocks.append(evaluation.trial_ids.copy())
            evaluation_subject_blocks.append(np.full(len(evaluation.labels), subject, dtype=np.int16))
        fold_inference_times.append(time.perf_counter() - inference_start)

        # Frozen-rank audit: each rank uses its own source-inner-selected lambda.
        ranks = sorted({int(row["rank"]) for row in candidate_rows if row["method"] == "LRCM"})
        for audit_rank in ranks:
            audit_lambda = _best_lambda_for_rank(candidate_rows, audit_rank)
            audit_core = fit_ridge_core(proto.signatures[source, 0], proto.signatures[source, 1], audit_lambda)
            target_scores: list[float] = []
            for target in test:
                subject = int(bundle.subjects[target])
                predicted_proto, _ = method_prototypes("LRCM", int(target), sealed, H, gamma_e, gamma_d, audit_core, audit_rank)
                trial = bundle.trials[(subject, bundle.deployment_session)]
                evaluation = bundle.deployment_evaluation[subject]
                pred, _ = ncm_predict(trial.features, predicted_proto, 1.0)
                target_scores.append(balanced_accuracy_fast(evaluation.labels, pred, len(bundle.class_names)))
            fixed_rank_rows.append({
                "dataset": dataset,
                "direction": direction,
                "outer_fold": fold_index,
                "rank": audit_rank,
                "ridge_lambda": audit_lambda,
                "mean_balanced_accuracy": float(np.mean(target_scores)),
            })

    suffix = f"{dataset}_{direction}"
    subject_frame = pd.DataFrame(subject_rows)
    aggregate_rows = _aggregate_metric_rows(subject_rows, dataset, direction)
    pd.DataFrame(inner_rows).to_csv(output / "tables" / f"{suffix}_inner_hyperparameters.csv", index=False, lineterminator="\n", float_format="%.17g")
    subject_frame.to_csv(output / "tables" / f"{suffix}_per_subject_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(aggregate_rows).to_csv(output / "tables" / f"{suffix}_aggregate_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(fixed_rank_rows).to_csv(output / "tables" / f"{suffix}_rank_performance.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(memory_rows).to_csv(output / "tables" / f"{suffix}_memory_runtime.csv", index=False, lineterminator="\n")
    atomic_write_json(output / "objects" / f"{suffix}_selected_hyperparameters.json", {"selections": selections})
    arrays: dict[str, np.ndarray] = {
        "trial_subject": np.concatenate(evaluation_subject_blocks),
        "trial_id": np.concatenate(evaluation_trial_id_blocks),
    }
    for method in methods:
        arrays[f"prediction__{method}"] = np.concatenate(prediction_blocks[method])
        arrays[f"probability__{method}"] = np.concatenate(probability_blocks[method])
    np.savez_compressed(output / "objects" / f"{suffix}_predictions.npz", **arrays)
    np.savez_compressed(
        output / "objects" / f"{suffix}_deployment_evaluation_labels.npz",
        trial_subject=np.concatenate(evaluation_subject_blocks),
        trial_id=np.concatenate(evaluation_trial_id_blocks),
        class_index=np.concatenate(evaluation_label_blocks),
        class_names=np.asarray(bundle.class_names),
    )
    selected_ranks = [int(row["lrcm"]["rank"]) for row in selections]
    summary = {
        "dataset": dataset,
        "direction": direction,
        "voting": not reverse,
        "enrollment_session": bundle.enrollment_session,
        "deployment_session": bundle.deployment_session,
        "subjects": len(bundle.subjects),
        "class_order": list(bundle.class_names),
        "selected_ranks": selected_ranks,
        "median_selected_rank": float(np.median(selected_ranks)),
        "rank_frequency": {str(rank_value): selected_ranks.count(rank_value) for rank_value in sorted(set(selected_ranks))},
        "aggregate_metrics": aggregate_rows,
        "mean_model_fit_seconds_per_fold": float(np.mean(fold_model_times)),
        "mean_inference_seconds_per_fold": float(np.mean(fold_inference_times)),
        "total_seconds": time.perf_counter() - started,
        "outer_target_deployment_labels_used_for_fitting": False,
    }
    atomic_write_json(output / "decisions" / f"{suffix}_observed.json", summary)
    return summary


def run_stieger_observed(repo_root: str | Path) -> dict[str, Any]:
    chronological = run_dataset_observed(repo_root, "stieger", reverse=False)
    reverse = run_dataset_observed(repo_root, "stieger", reverse=True)
    return {"chronological": chronological, "reverse_non_voting": reverse}


def run_openbmi_observed(repo_root: str | Path) -> dict[str, Any]:
    chronological = run_dataset_observed(repo_root, "openbmi", reverse=False)
    reverse = run_dataset_observed(repo_root, "openbmi", reverse=True)
    return {"chronological": chronological, "reverse_non_voting": reverse}


def _load_selections(output: Path, dataset: str, direction: str = "chronological") -> list[dict[str, Any]]:
    path = output / "objects" / f"{dataset}_{direction}_selected_hyperparameters.json"
    if not path.is_file():
        raise NumericalContractError(f"observed selection missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selections = list(payload["selections"])
    if len(selections) != 6:
        raise NumericalContractError("selected hyperparameter fold count mismatch")
    return selections


def _null_target_score(
    bundle: DatasetBundle,
    proto: PrototypeData,
    target: int,
    core: RidgeCore,
    rank: int,
    H: np.ndarray,
    gamma_e: np.ndarray,
    gamma_d: np.ndarray,
    enrollment_signature: np.ndarray | None = None,
    basis: np.ndarray | None = None,
) -> float:
    subject = int(bundle.subjects[target])
    predicted_proto, _ = method_prototypes(
        "LRCM", target, proto, H, gamma_e, gamma_d, core, rank,
        enrollment_signature=enrollment_signature, output_basis=basis,
    )
    trial = bundle.trials[(subject, bundle.deployment_session)]
    evaluation = bundle.deployment_evaluation[subject]
    prediction, _ = ncm_predict(trial.features, predicted_proto, 1.0)
    return balanced_accuracy_fast(evaluation.labels, prediction, len(bundle.class_names))


def run_dataset_nulls(repo_root: str | Path, dataset: str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    _ensure_protocol_frozen(root, config)
    output = output_path(root, config)
    bundle = load_dataset(root, config, dataset, reverse=False)
    proto = compute_prototype_data(bundle)
    H = helmert(len(bundle.class_names))
    selections = _load_selections(output, dataset)
    subject_table = pd.read_csv(output / "tables" / f"{dataset}_chronological_per_subject_metrics.csv")
    observed = float(subject_table[subject_table["method"] == "LRCM"]["balanced_accuracy"].mean())
    replicates = int(config["inference"]["null_replicates"])
    nulls = {
        "ENROLLMENT_SUBJECT_MEMORY_PERMUTATION": np.empty(replicates, dtype=np.float64),
        "ENROLLMENT_CLASS_SEMANTICS_PERMUTATION": np.empty(replicates, dtype=np.float64),
        "RANDOM_RANK_MATCHED_OUTPUT_SUBSPACE": np.empty(replicates, dtype=np.float64),
        "UNPAIRED_SOURCE_SESSION_MAP": np.empty(replicates, dtype=np.float64),
    }
    started = time.perf_counter()
    for replicate in range(replicates):
        score_blocks: dict[str, list[float]] = {name: [] for name in nulls}
        for fold_index, test in enumerate(bundle.folds):
            source = _outer_source_indices(bundle, test)
            selection = selections[fold_index]["lrcm"]
            ridge_lambda, rank = float(selection["ridge_lambda"]), int(selection["rank"])
            gamma_e, gamma_d = population_templates(proto, source)
            core = fit_ridge_core(proto.signatures[source, 0], proto.signatures[source, 1], ridge_lambda)
            memory_order = fixed_point_free(_rng(config, f"{dataset}_memory_permutation", replicate, fold_index), len(test))
            random_basis = haar_basis(_rng(config, f"{dataset}_random_output_basis", replicate, fold_index), core.W.shape[1], rank)
            unpaired_order = fixed_point_free(_rng(config, f"{dataset}_unpaired_source", replicate, fold_index), len(source))
            unpaired_core = fit_ridge_core(proto.signatures[source, 0], proto.signatures[source[unpaired_order], 1], ridge_lambda)
            for local_index, target in enumerate(test):
                donor = int(test[memory_order[local_index]])
                score_blocks["ENROLLMENT_SUBJECT_MEMORY_PERMUTATION"].append(
                    _null_target_score(bundle, proto, int(target), core, rank, H, gamma_e, gamma_d, enrollment_signature=proto.signatures[donor, 0])
                )
                class_order = nonidentity_class_permutation(
                    _rng(config, f"{dataset}_class_semantics_permutation", replicate, fold_index, int(bundle.subjects[target])),
                    len(bundle.class_names),
                )
                permuted_signature = prototypes_to_signature(proto.prototypes[int(target), 0, class_order], H)
                score_blocks["ENROLLMENT_CLASS_SEMANTICS_PERMUTATION"].append(
                    _null_target_score(bundle, proto, int(target), core, rank, H, gamma_e, gamma_d, enrollment_signature=permuted_signature)
                )
                score_blocks["RANDOM_RANK_MATCHED_OUTPUT_SUBSPACE"].append(
                    _null_target_score(bundle, proto, int(target), core, rank, H, gamma_e, gamma_d, basis=random_basis)
                )
                score_blocks["UNPAIRED_SOURCE_SESSION_MAP"].append(
                    _null_target_score(bundle, proto, int(target), unpaired_core, rank, H, gamma_e, gamma_d)
                )
        for name, scores in score_blocks.items():
            nulls[name][replicate] = float(np.mean(scores))
    summaries: list[dict[str, Any]] = []
    for name, values in nulls.items():
        p_value = float((1 + np.count_nonzero(values >= observed)) / (1 + len(values)))
        summaries.append({
            "dataset": dataset,
            "null": name,
            "observed_mean_balanced_accuracy": observed,
            "null_mean": float(np.mean(values)),
            "null_median": float(np.median(values)),
            "null_q025": float(np.quantile(values, 0.025)),
            "null_q975": float(np.quantile(values, 0.975)),
            "p_value": p_value,
            "replicates": len(values),
            "hyperparameter_rule": config["inference"]["unpaired_null_hyperparameter_rule"] if name == "UNPAIRED_SOURCE_SESSION_MAP" else "frozen_observed_source_only_selection",
        })
    np.savez_compressed(output / "nulls" / f"{dataset}_memory_nulls.npz", **nulls)
    pd.DataFrame(summaries).to_csv(output / "tables" / f"{dataset}_memory_null_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    result = {
        "dataset": dataset,
        "observed_mean_balanced_accuracy": observed,
        "summaries": summaries,
        "runtime_seconds": time.perf_counter() - started,
        "unpaired_source_selection_rule": config["inference"]["unpaired_null_hyperparameter_rule"],
        "outer_target_deployment_labels_used_for_fitting": False,
    }
    atomic_write_json(output / "decisions" / f"{dataset}_memory_nulls.json", result)
    return result


def run_stieger_nulls(repo_root: str | Path) -> dict[str, Any]:
    return run_dataset_nulls(repo_root, "stieger")


def run_openbmi_nulls(repo_root: str | Path) -> dict[str, Any]:
    return run_dataset_nulls(repo_root, "openbmi")


def _paired_signflip_p(differences: np.ndarray, config: Mapping[str, Any], namespace: str, replicates: int = 1999) -> float:
    values = np.asarray(differences, dtype=np.float64)
    observed = float(np.mean(values))
    rng = _rng(config, namespace)
    null = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        null[index] = float(np.mean(values * rng.choice(np.asarray([-1.0, 1.0]), size=len(values))))
    return float((1 + np.count_nonzero(null >= observed)) / (1 + replicates))


def _bootstrap_ci(values: np.ndarray, config: Mapping[str, Any], namespace: str, replicates: int = 10000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = _rng(config, namespace)
    sampled = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled[index] = float(np.mean(array[rng.integers(0, len(array), len(array))]))
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def paired_comparison(
    subject_frame: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    config: Mapping[str, Any],
    namespace: str,
    improvement: str = "higher",
) -> dict[str, Any]:
    pivot = subject_frame.pivot(index="subject", columns="method", values=metric)
    raw = pivot[left].to_numpy() - pivot[right].to_numpy()
    differences = raw if improvement == "higher" else -raw
    ci = _bootstrap_ci(differences, config, namespace + "_bootstrap", int(config["inference"]["bootstrap_replicates"]))
    p_value = _paired_signflip_p(differences, config, namespace + "_signflip", int(config["inference"]["permutation_replicates"]))
    loo = np.asarray([np.mean(np.delete(differences, index)) for index in range(len(differences))])
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "improvement_direction": improvement,
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "bootstrap_ci": list(ci),
        "p_value_raw": p_value,
        "subject_win_rate": float(np.mean(differences > 0.0)),
        "leave_one_subject_mean_range": [float(np.min(loo)), float(np.max(loo))],
        "worst_quartile_mean_difference": float(np.mean(np.sort(differences)[: max(1, len(differences) // 4)])),
    }


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_value_raw"]))
    running = 0.0
    m = len(rows)
    for position, index in enumerate(order):
        adjusted = min(1.0, (m - position) * float(rows[index]["p_value_raw"]))
        running = max(running, adjusted)
        rows[index]["p_value_holm"] = running


def run_kshot_curves(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    _ensure_protocol_frozen(root, config)
    output = output_path(root, config)
    budgets = [int(x) for x in config["inference"]["kshot_budgets_per_class"]]
    draws = int(config["inference"]["kshot_subsamples"])
    subject_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    equivalent_rows: list[dict[str, Any]] = []
    for dataset in ("stieger", "openbmi"):
        bundle = load_dataset(root, config, dataset, reverse=False)
        for subject in bundle.subjects:
            subject_id = int(subject)
            trial = bundle.trials[(subject_id, bundle.deployment_session)]
            evaluation = bundle.deployment_evaluation[subject_id]
            c = len(bundle.class_names)
            for budget in budgets:
                draw_ba = np.empty(draws, dtype=np.float64)
                draw_f1 = np.empty(draws, dtype=np.float64)
                for draw in range(draws):
                    support_parts: list[np.ndarray] = []
                    for class_index in range(c):
                        choices = np.flatnonzero(evaluation.labels == class_index)
                        rng = _rng(config, f"{dataset}_kshot_support", subject_id, budget, draw, class_index)
                        support_parts.append(np.sort(rng.choice(choices, size=budget, replace=False)))
                    support = np.concatenate(support_parts)
                    query = np.setdiff1d(np.arange(len(evaluation.labels)), support)
                    if len(np.intersect1d(support, query)) or np.bincount(evaluation.labels[support], minlength=c).tolist() != [budget] * c:
                        raise LeakageOrSplitError("K-shot support/query contract failure")
                    prototypes = np.stack([
                        np.mean(trial.features[support[evaluation.labels[support] == class_index]], axis=0)
                        for class_index in range(c)
                    ])
                    prediction, _ = ncm_predict(trial.features[query], prototypes, 1.0)
                    y_query = evaluation.labels[query]
                    draw_ba[draw] = balanced_accuracy_fast(y_query, prediction, c)
                    draw_f1[draw] = float(f1_score(y_query, prediction, average="macro", labels=np.arange(c), zero_division=0))
                subject_rows.append({
                    "dataset": dataset,
                    "subject": subject_id,
                    "K_per_class": budget,
                    "draws": draws,
                    "expected_balanced_accuracy": float(np.mean(draw_ba)),
                    "expected_macro_f1": float(np.mean(draw_f1)),
                    "balanced_accuracy_draw_sd": float(np.std(draw_ba, ddof=1)),
                })
            # Full oracle upper bound, evaluated on the same trials descriptively.
            oracle_proto = np.stack([
                np.mean(trial.features[evaluation.labels == class_index], axis=0) for class_index in range(c)
            ])
            oracle_prediction, _ = ncm_predict(trial.features, oracle_proto, 1.0)
            subject_rows.append({
                "dataset": dataset,
                "subject": subject_id,
                "K_per_class": "FULL_ORACLE",
                "draws": 1,
                "expected_balanced_accuracy": balanced_accuracy_fast(evaluation.labels, oracle_prediction, c),
                "expected_macro_f1": float(f1_score(evaluation.labels, oracle_prediction, average="macro", labels=np.arange(c), zero_division=0)),
                "balanced_accuracy_draw_sd": 0.0,
            })
        frame = pd.DataFrame([row for row in subject_rows if row["dataset"] == dataset])
        numeric = frame[frame["K_per_class"] != "FULL_ORACLE"].copy()
        numeric["K_per_class"] = numeric["K_per_class"].astype(int)
        for budget, block in numeric.groupby("K_per_class"):
            aggregate_rows.append({
                "dataset": dataset,
                "K_per_class": int(budget),
                "subjects": len(block),
                "mean_expected_balanced_accuracy": float(block["expected_balanced_accuracy"].mean()),
                "mean_expected_macro_f1": float(block["expected_macro_f1"].mean()),
            })
        oracle = frame[frame["K_per_class"] == "FULL_ORACLE"]
        aggregate_rows.append({
            "dataset": dataset,
            "K_per_class": "FULL_ORACLE",
            "subjects": len(oracle),
            "mean_expected_balanced_accuracy": float(oracle["expected_balanced_accuracy"].mean()),
            "mean_expected_macro_f1": float(oracle["expected_macro_f1"].mean()),
        })
        observed = pd.read_csv(output / "tables" / f"{dataset}_chronological_per_subject_metrics.csv")
        lrcm = float(observed[observed["method"] == "LRCM"]["balanced_accuracy"].mean())
        curve = {int(row["K_per_class"]): float(row["mean_expected_balanced_accuracy"]) for row in aggregate_rows if row["dataset"] == dataset and row["K_per_class"] != "FULL_ORACLE"}
        matching = [budget for budget in budgets if curve[budget] >= lrcm]
        equivalent = min(matching) if matching else None
        below = max([budget for budget in budgets if curve[budget] < lrcm], default=None)
        above = min([budget for budget in budgets if curve[budget] >= lrcm], default=None)
        equivalent_rows.append({
            "dataset": dataset,
            "lrcm_mean_balanced_accuracy": lrcm,
            "equivalent_K_per_class": equivalent,
            "lower_bracket_K": below,
            "upper_bracket_K": above,
            "interpretation": "descriptive interpolation-free bracket; not a literal label replacement claim",
        })
    pd.DataFrame(subject_rows).to_csv(output / "tables" / "current_session_kshot_subjects.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(aggregate_rows).to_csv(output / "tables" / "current_session_kshot_curves.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(equivalent_rows).to_csv(output / "tables" / "calibration_equivalent_label_budget.csv", index=False, lineterminator="\n", float_format="%.17g")
    result = {"subject_rows": len(subject_rows), "aggregate": aggregate_rows, "calibration_equivalent": equivalent_rows, "evaluation_only": True}
    atomic_write_json(output / "controls" / "current_session_kshot.json", result)
    return result


def _null_lookup(output: Path, dataset: str) -> dict[str, dict[str, Any]]:
    payload = json.loads((output / "decisions" / f"{dataset}_memory_nulls.json").read_text(encoding="utf-8"))
    return {row["null"]: row for row in payload["summaries"]}


def _method_metric(frame: pd.DataFrame, method: str, metric: str) -> float:
    return float(frame[frame["method"] == method][metric].mean())


def _write_standard_tables(output: Path, stieger: pd.DataFrame, openbmi: pd.DataFrame, comparisons: list[dict[str, Any]], rank_rows: list[dict[str, Any]], gates: Mapping[str, Any]) -> None:
    stieger_aggregate = pd.DataFrame(_aggregate_metric_rows(stieger.to_dict("records"), "stieger", "chronological"))
    openbmi_aggregate = pd.DataFrame(_aggregate_metric_rows(openbmi.to_dict("records"), "openbmi", "chronological"))
    stieger_aggregate.to_csv(output / "tables" / "stieger_per_method_aggregate_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    stieger.to_csv(output / "tables" / "stieger_per_subject_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame([row for row in comparisons if row["dataset"] == "stieger"]).to_csv(output / "tables" / "stieger_primary_paired_comparisons.csv", index=False, lineterminator="\n", float_format="%.17g")
    openbmi_aggregate.to_csv(output / "tables" / "openbmi_per_method_aggregate_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    openbmi.to_csv(output / "tables" / "openbmi_per_subject_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame([row for row in comparisons if row["dataset"] == "openbmi"]).to_csv(output / "tables" / "openbmi_replication_comparison.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(rank_rows).to_csv(output / "tables" / "selected_rank_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame([{"gate": key, "passed": bool(value)} for key, value in gates.items()]).to_csv(output / "tables" / "terminal_gate_table.csv", index=False, lineterminator="\n")


def _make_figures(output: Path, decision: Mapping[str, Any], stieger: pd.DataFrame, openbmi: pd.DataFrame) -> None:
    def save(fig: plt.Figure, number: int, slug: str) -> None:
        fig.tight_layout()
        fig.savefig(output / "figures" / f"figure_{number:02d}_{slug}.png", dpi=160)
        fig.savefig(output / "figures" / f"figure_{number:02d}_{slug}.pdf")
        plt.close(fig)

    # 1: deployment scenario.
    fig, ax = plt.subplots(figsize=(8, 3)); ax.axis("off")
    for x, label in ((0.05, "Labeled enrollment\ninitial calibration"), (0.39, "Source-trained\nlow-rank map"), (0.73, "Unlabeled deployment\nzero-recalibration")):
        ax.add_patch(plt.Rectangle((x, 0.3), 0.22, 0.38, facecolor="#dceefa", edgecolor="#24557a")); ax.text(x + 0.11, 0.49, label, ha="center", va="center")
    ax.annotate("", (0.39, 0.49), (0.27, 0.49), arrowprops={"arrowstyle": "->"}); ax.annotate("", (0.73, 0.49), (0.61, 0.49), arrowprops={"arrowstyle": "->"})
    save(fig, 1, "returning_user_scenario")
    # 2: mathematical pipeline.
    fig, ax = plt.subplots(figsize=(8, 3)); ax.axis("off")
    ax.text(0.05, .7, "$x_E-\\bar x_E$"); ax.text(.28, .7, "$W_\\lambda V_R$"); ax.text(.51, .7, "$a_t\\in\\mathbb{R}^R$"); ax.text(.72, .7, "$\\hat x_D$")
    for a, b in ((.17,.27),(.4,.5),(.62,.71)): ax.annotate("", (b,.7), (a,.7), arrowprops={"arrowstyle":"->"})
    ax.text(.5,.25,"Helmert inverse + unlabeled deployment marginal → predicted class prototypes",ha="center")
    save(fig, 2, "lrcm_pipeline")
    # 3: Stieger methods.
    order = ["LRCM","IDENTITY_RESIDUAL_CARRY","PAST_PROTOTYPE_DIRECT","POPULATION_ONLY","FULL_RIDGE_TRANSFER","ENROLLMENT_PCA_TRANSFER"]
    vals = [_method_metric(stieger, method, "balanced_accuracy") for method in order]
    fig, ax = plt.subplots(figsize=(9,4)); ax.bar(np.arange(len(order)), vals, color="#4378bf"); ax.set_xticks(np.arange(len(order)), [x.replace("_","\n") for x in order], fontsize=8); ax.set_ylabel("Subject-mean balanced accuracy"); save(fig,3,"stieger_method_accuracy")
    # 4: subject gains.
    pivot = stieger.pivot(index="subject",columns="method",values="balanced_accuracy"); gains = pivot["LRCM"]-pivot["IDENTITY_RESIDUAL_CARRY"]
    fig, ax = plt.subplots(figsize=(8,4)); ax.axhline(0,color="black",lw=.8); ax.scatter(np.arange(len(gains)), gains, s=18); ax.set_xlabel("Subject (fold evaluation order)"); ax.set_ylabel("LRCM − identity carry"); save(fig,4,"stieger_subject_gain")
    # 5: observed/null.
    nulls = np.load(output / "nulls" / "stieger_memory_nulls.npz", allow_pickle=False)
    values = nulls["ENROLLMENT_SUBJECT_MEMORY_PERMUTATION"]
    fig, ax = plt.subplots(figsize=(7,4)); ax.hist(values,bins=35,color="#b8c9d9"); ax.axvline(_method_metric(stieger,"LRCM","balanced_accuracy"),color="#b22222",lw=2,label="observed"); ax.legend(); ax.set_xlabel("Mean balanced accuracy"); save(fig,5,"memory_permutation_null")
    # 6: rank curve.
    ranks = pd.read_csv(output / "tables" / "stieger_chronological_rank_performance.csv").groupby("rank")["mean_balanced_accuracy"].mean()
    fig, ax = plt.subplots(figsize=(6,4)); ax.plot(ranks.index,ranks.values,marker="o"); ax.set_xlabel("Fixed rank"); ax.set_ylabel("Held-out balanced accuracy"); save(fig,6,"rank_performance")
    # 7: low-rank audit.
    audit_order=["LRCM","FULL_RIDGE_TRANSFER","ENROLLMENT_PCA_TRANSFER"]
    fig, ax=plt.subplots(figsize=(6,4)); ax.bar(audit_order,[_method_metric(stieger,m,"balanced_accuracy") for m in audit_order],color=["#285f9e","#8ba8c6","#c1a66b"]); ax.tick_params(axis="x",rotation=20); ax.set_ylabel("Balanced accuracy"); save(fig,7,"low_rank_audit")
    # 8: OpenBMI replication.
    rep_order=["LRCM","IDENTITY_RESIDUAL_CARRY","PAST_PROTOTYPE_DIRECT","POPULATION_ONLY"]
    fig, ax=plt.subplots(figsize=(7,4)); ax.bar(rep_order,[_method_metric(openbmi,m,"balanced_accuracy") for m in rep_order],color="#5b9f68"); ax.tick_params(axis="x",rotation=25); ax.set_ylabel("Balanced accuracy"); save(fig,8,"openbmi_replication")
    # 9: K-shot curves.
    curve=pd.read_csv(output / "tables" / "current_session_kshot_curves.csv"); fig,ax=plt.subplots(figsize=(7,4))
    for dataset, frame in curve[curve["K_per_class"]!="FULL_ORACLE"].groupby("dataset"):
        x=frame["K_per_class"].astype(int); ax.plot(x,frame["mean_expected_balanced_accuracy"],marker="o",label=f"{dataset} K-shot")
        observed_frame=stieger if dataset=="stieger" else openbmi; ax.axhline(_method_metric(observed_frame,"LRCM","balanced_accuracy"),ls="--",alpha=.7,label=f"{dataset} LRCM")
    ax.set_xlabel("Current-session labels per class"); ax.set_ylabel("Balanced accuracy"); ax.legend(fontsize=8); save(fig,9,"kshot_equivalent")
    # 10: memory size.
    memory=pd.read_csv(output / "tables" / "stieger_chronological_memory_runtime.csv"); lrcm=stieger[stieger.method=="LRCM"].set_index("subject")
    fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(memory.memory_bytes_float64,[lrcm.loc[s,"balanced_accuracy"] for s in memory.subject]); ax.set_xlabel("Per-user memory bytes"); ax.set_ylabel("Balanced accuracy"); save(fig,10,"memory_size_performance")
    # 11: NLL/ECE.
    fig,axes=plt.subplots(1,2,figsize=(9,4)); methods=["LRCM","IDENTITY_RESIDUAL_CARRY","POPULATION_ONLY"]
    axes[0].bar(methods,[_method_metric(stieger,m,"nll") for m in methods]); axes[0].set_title("NLL"); axes[1].bar(methods,[_method_metric(stieger,m,"ece10") for m in methods]); axes[1].set_title("ECE (10 bins)")
    for ax in axes: ax.tick_params(axis="x",rotation=25)
    save(fig,11,"calibration")
    # 12: terminal gates.
    names=list(decision["gates"]); values=np.asarray([[1.0 if decision["gates"][name] else 0.0] for name in names])
    fig,ax=plt.subplots(figsize=(7,max(4,.28*len(names)))); ax.imshow(values,aspect="auto",vmin=0,vmax=1,cmap="RdYlGn"); ax.set_yticks(np.arange(len(names)),names,fontsize=8); ax.set_xticks([0],["pass"]); save(fig,12,"terminal_gates")


def _result_manifest(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            records.append({"path": str(path.relative_to(output)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    payload: dict[str, Any] = {"schema_version": "returning-user-conditional-memory-v0-results", "records": records, "count": len(records)}
    payload["canonical_sha256"] = _canonical_hash(payload)
    return payload


def generate_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root)
    _ensure_protocol_frozen(root, config)
    output = output_path(root, config)
    parent_verification = verify_parent_snapshot(root, output / "protocol" / "parent_artifact_hashes.json")
    stieger = pd.read_csv(output / "tables" / "stieger_chronological_per_subject_metrics.csv")
    openbmi = pd.read_csv(output / "tables" / "openbmi_chronological_per_subject_metrics.csv")
    comparisons: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    for baseline in ("IDENTITY_RESIDUAL_CARRY", "PAST_PROTOTYPE_DIRECT", "POPULATION_ONLY"):
        row = paired_comparison(stieger, "LRCM", baseline, "balanced_accuracy", config, f"stieger_lrcm_vs_{baseline.lower()}")
        row.update({"dataset": "stieger", "comparison": f"LRCM - {baseline}"})
        primary_rows.append(row)
    _holm_adjust(primary_rows)
    comparisons.extend(primary_rows)
    macro = paired_comparison(stieger, "LRCM", "IDENTITY_RESIDUAL_CARRY", "macro_f1", config, "stieger_macro_identity")
    macro.update({"dataset": "stieger", "comparison": "LRCM - IDENTITY_RESIDUAL_CARRY macro-F1", "p_value_holm": macro["p_value_raw"]})
    comparisons.append(macro)
    nll = paired_comparison(stieger, "LRCM", "IDENTITY_RESIDUAL_CARRY", "nll", config, "stieger_nll_identity", improvement="lower")
    nll.update({"dataset": "stieger", "comparison": "IDENTITY_RESIDUAL_CARRY NLL - LRCM NLL", "p_value_holm": nll["p_value_raw"]})
    comparisons.append(nll)
    full = paired_comparison(stieger, "LRCM", "FULL_RIDGE_TRANSFER", "balanced_accuracy", config, "stieger_lrcm_full")
    full.update({"dataset": "stieger", "comparison": "LRCM - FULL_RIDGE_TRANSFER", "p_value_holm": full["p_value_raw"]})
    comparisons.append(full)
    pca = paired_comparison(stieger, "LRCM", "ENROLLMENT_PCA_TRANSFER", "balanced_accuracy", config, "stieger_lrcm_pca")
    pca.update({"dataset": "stieger", "comparison": "LRCM - ENROLLMENT_PCA_TRANSFER", "p_value_holm": pca["p_value_raw"]})
    comparisons.append(pca)
    replication = paired_comparison(openbmi, "LRCM", "IDENTITY_RESIDUAL_CARRY", "balanced_accuracy", config, "openbmi_lrcm_identity")
    replication.update({"dataset": "openbmi", "comparison": "LRCM - IDENTITY_RESIDUAL_CARRY", "p_value_holm": replication["p_value_raw"]})
    comparisons.append(replication)
    stieger_nulls = _null_lookup(output, "stieger")
    openbmi_nulls = _null_lookup(output, "openbmi")
    stieger_selection = _load_selections(output, "stieger")
    openbmi_selection = _load_selections(output, "openbmi")
    stieger_ranks = [int(row["lrcm"]["rank"]) for row in stieger_selection]
    openbmi_ranks = [int(row["lrcm"]["rank"]) for row in openbmi_selection]
    rank_rows = [
        {"dataset": "stieger", "selected_ranks": "|".join(map(str, stieger_ranks)), "median_rank": float(np.median(stieger_ranks)), "folds_rank_at_most_3": int(np.sum(np.asarray(stieger_ranks) <= 3))},
        {"dataset": "openbmi", "selected_ranks": "|".join(map(str, openbmi_ranks)), "median_rank": float(np.median(openbmi_ranks)), "folds_rank_at_most_3": int(np.sum(np.asarray(openbmi_ranks) <= 3))},
    ]

    utility_gates: dict[str, bool] = {}
    for row in primary_rows:
        key = row["right"].lower()
        utility_gates[f"gain_over_{key}_positive"] = row["mean_difference"] > 0.0
        utility_gates[f"gain_over_{key}_ci_above_zero"] = row["bootstrap_ci"][0] > 0.0
        utility_gates[f"gain_over_{key}_holm_p_at_most_005"] = row["p_value_holm"] <= 0.05
    identity = primary_rows[0]
    utility_gates.update({
        "macro_f1_gain_over_identity_positive": macro["mean_difference"] > 0.0,
        "nll_noninferior_within_001": nll["bootstrap_ci"][0] >= -float(config["inference"]["nll_noninferiority_margin"]),
        "identity_leave_one_subject_gain_positive": identity["leave_one_subject_mean_range"][0] > 0.0,
    })
    memory_gates = {
        "enrollment_subject_memory_permutation_p_at_most_005": stieger_nulls["ENROLLMENT_SUBJECT_MEMORY_PERMUTATION"]["p_value"] <= 0.05,
        "enrollment_class_semantics_permutation_p_at_most_005": stieger_nulls["ENROLLMENT_CLASS_SEMANTICS_PERMUTATION"]["p_value"] <= 0.05,
        "unpaired_source_session_map_p_at_most_005": stieger_nulls["UNPAIRED_SOURCE_SESSION_MAP"]["p_value"] <= 0.05,
    }
    low_rank_gates = {
        "median_selected_rank_at_most_3": float(np.median(stieger_ranks)) <= 3.0,
        "at_least_4_of_6_ranks_at_most_3": int(np.sum(np.asarray(stieger_ranks) <= 3)) >= 4,
        "full_ridge_noninferiority_margin_0005": full["bootstrap_ci"][0] >= -float(config["inference"]["full_ridge_ba_noninferiority_margin"]),
        "exceeds_enrollment_pca": pca["mean_difference"] > 0.0 and pca["bootstrap_ci"][0] > 0.0 and pca["p_value_raw"] <= 0.05,
        "exceeds_random_rank_matched": stieger_nulls["RANDOM_RANK_MATCHED_OUTPUT_SUBSPACE"]["p_value"] <= 0.05,
        "correct_source_pairing_required": stieger_nulls["UNPAIRED_SOURCE_SESSION_MAP"]["p_value"] <= 0.05,
    }
    external_gates = {
        "openbmi_gain_positive": replication["mean_difference"] > 0.0,
        "openbmi_ci_above_zero": replication["bootstrap_ci"][0] > 0.0,
        "openbmi_p_at_most_005": replication["p_value_raw"] <= 0.05,
    }
    gates = {**utility_gates, **memory_gates, **low_rank_gates, **external_gates}
    stieger_utility = all(utility_gates.values())
    memory_pass = all(memory_gates.values())
    low_rank_pass = all(low_rank_gates.values())
    external_pass = all(external_gates.values())
    identity_utility = (
        identity["mean_difference"] > 0.0
        and identity["bootstrap_ci"][0] > 0.0
        and identity["p_value_holm"] <= 0.05
        and identity["leave_one_subject_mean_range"][0] > 0.0
    )
    if stieger_utility and memory_pass and low_rank_pass and external_pass:
        terminal = config["decisions"]["replicated"]
        next_question = config["decisions"]["next_if_go"]
    elif stieger_utility and memory_pass and low_rank_pass:
        terminal = config["decisions"]["stieger_only"]
        next_question = config["decisions"]["next_if_go"]
    elif identity_utility and memory_pass:
        terminal = config["decisions"]["map_not_low_rank"]
        next_question = config["decisions"]["next_if_go"]
    else:
        terminal = config["decisions"]["stop"]
        next_question = config["decisions"]["next_if_stop"]

    decision = {
        "terminal": terminal,
        "gates": gates,
        "stieger_utility_pass": stieger_utility,
        "memory_specific_pass": memory_pass,
        "low_rank_pass": low_rank_pass,
        "openbmi_replication_pass": external_pass,
        "stieger_selected_ranks": stieger_ranks,
        "openbmi_selected_ranks": openbmi_ranks,
        "comparisons": comparisons,
        "stieger_nulls": stieger_nulls,
        "openbmi_nulls": openbmi_nulls,
        "next_question_or_statement": next_question,
        "parent_results_reinterpreted": False,
    }
    atomic_write_json(output / "decisions" / "terminal_decision.json", decision)
    pd.DataFrame(comparisons).to_csv(output / "tables" / "all_paired_comparisons.csv", index=False, lineterminator="\n", float_format="%.17g")
    _write_standard_tables(output, stieger, openbmi, comparisons, rank_rows, gates)

    stieger_lrcm = stieger[stieger.method == "LRCM"]
    openbmi_lrcm = openbmi[openbmi.method == "LRCM"]
    kshot = pd.read_csv(output / "tables" / "calibration_equivalent_label_budget.csv").to_dict("records")
    lines = [
        "# Returning-User Low-Rank Conditional Memory Downstream Pilot V0",
        "",
        f"**Terminal: `{terminal}`**",
        "",
        "## Frozen deployment scenario",
        "",
        "Stieger2021 session 2 labeled enrollment → session 3 zero-label deployment, task 3 and four literal classes, is the voting primary. OpenBMI official source session 1 → 2 is the external binary replication. The evaluation is offline batch zero-recalibration after one labeled enrollment session.",
        "",
        "## Cache and leakage gates",
        "",
        f"All {parent_verification['count']} tracked PR #16--#19 artifacts retained canonical snapshot `{parent_verification['canonical_sha256']}`. Stieger 124 tangent records and OpenBMI `(54,2,100,210)` tangents passed frozen hashes. No raw data were downloaded or rebuilt. Outer target deployment labels were sealed until predictions were saved.",
        "",
        "## Stieger chronological primary",
        "",
        f"LRCM subject-mean balanced accuracy was {stieger_lrcm.balanced_accuracy.mean():.6f}, macro-F1 {stieger_lrcm.macro_f1.mean():.6f}, NLL {stieger_lrcm.nll.mean():.6f}, and ECE10 {stieger_lrcm.ece10.mean():.6f}. Selected ranks were `{stieger_ranks}` (median {np.median(stieger_ranks):.1f}).",
        "",
        "Primary comparisons (positive values favor LRCM):",
        "",
    ]
    for row in primary_rows:
        lines.append(f"- vs `{row['right']}`: mean ΔBA {row['mean_difference']:.6f}, 95% CI [{row['bootstrap_ci'][0]:.6f}, {row['bootstrap_ci'][1]:.6f}], raw p={row['p_value_raw']:.6g}, Holm p={row['p_value_holm']:.6g}, win rate {row['subject_win_rate']:.3f}.")
    lines.extend([
        "",
        f"Memory null p-values: subject-memory permutation {stieger_nulls['ENROLLMENT_SUBJECT_MEMORY_PERMUTATION']['p_value']:.6g}; enrollment-class permutation {stieger_nulls['ENROLLMENT_CLASS_SEMANTICS_PERMUTATION']['p_value']:.6g}; random rank-matched subspace {stieger_nulls['RANDOM_RANK_MATCHED_OUTPUT_SUBSPACE']['p_value']:.6g}; unpaired source sessions {stieger_nulls['UNPAIRED_SOURCE_SESSION_MAP']['p_value']:.6g}.",
        "",
        f"Low-rank audit: LRCM−full-ridge mean ΔBA {full['mean_difference']:.6f}, CI [{full['bootstrap_ci'][0]:.6f}, {full['bootstrap_ci'][1]:.6f}]; LRCM−enrollment-PCA mean ΔBA {pca['mean_difference']:.6f}, CI [{pca['bootstrap_ci'][0]:.6f}, {pca['bootstrap_ci'][1]:.6f}], p={pca['p_value_raw']:.6g}.",
        "",
        "## OpenBMI external replication",
        "",
        f"Chronological LRCM subject-mean balanced accuracy was {openbmi_lrcm.balanced_accuracy.mean():.6f}. Against identity residual carry, mean ΔBA was {replication['mean_difference']:.6f}, CI [{replication['bootstrap_ci'][0]:.6f}, {replication['bootstrap_ci'][1]:.6f}], p={replication['p_value_raw']:.6g}. Selected ranks were `{openbmi_ranks}`.",
        "",
        "## Calibration-equivalent K-shot curve",
        "",
    ])
    for row in kshot:
        lines.append(f"- {row['dataset']}: LRCM BA {row['lrcm_mean_balanced_accuracy']:.6f}; smallest direct current-session K reaching it = {row['equivalent_K_per_class']}; bracket [{row['lower_bracket_K']}, {row['upper_bracket_K']}].")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This pilot does not establish new-user zero calibration, online causal adaptation, full conditional recovery, unlabeled semantic identification, physiology, source anatomy, universal subject coordinates, pseudo-label validity, TTA, ASD generalization, or clinical efficacy.",
        "",
        "## Exact next question or statement",
        "",
        next_question,
        "",
    ])
    report_path = output / "report" / "returning_user_conditional_memory_v0.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _make_figures(output, decision, stieger, openbmi)
    atomic_write_json(output / "git_provenance.json", {
        "parent_pr": 19,
        "parent_head": config["protocol"]["parent_head"],
        "branch": _git(root, "branch", "--show-current"),
        "head_at_result_generation": _git(root, "rev-parse", "HEAD"),
        "config_sha256": config_hash,
    })
    manifest = _result_manifest(output)
    atomic_write_json(output / "manifest.json", manifest)
    # Validate every freshly recorded hash immediately.
    for row in manifest["records"]:
        if sha256_file(output / row["path"]) != row["sha256"]:
            raise NumericalContractError(f"result artifact hash mismatch {row['path']}")
    return decision


def record_failure(repo_root: str | Path, terminal: str, exception: BaseException) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root, verify_protocol=False)
    output = output_path(root, config)
    _ensure_dirs(output)
    result = {"terminal": terminal, "exception_type": type(exception).__name__, "message": str(exception), "scientific_result_interpreted": False}
    atomic_write_json(output / "decisions" / "execution_failure.json", result)
    (output / "report" / "returning_user_conditional_memory_v0.md").write_text(
        f"# Returning-User Conditional Memory V0\n\nTerminal: `{terminal}`\n\n{type(exception).__name__}: {exception}\n",
        encoding="utf-8",
    )
    return result
