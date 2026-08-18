"""Leakage-safe two-view population structure analysis for frozen V0 U objects."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import linalg, stats

from src.interaction_provenance_v0 import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_array,
    sha256_file,
)


CONFIG_PATH = "configs/subject_class_population_structure_v1.yaml"
OUTPUT_NAME = "subject_class_population_structure_v1"
SPLITS = ("A", "B", "F")


class PopulationStructureError(RuntimeError):
    """Base fail-closed analysis error."""


class DataContractError(PopulationStructureError):
    """Immutable object, ordering, or fold contract failure."""


class NumericalContractError(PopulationStructureError):
    """Finite, degeneracy, or decomposition contract failure."""


@dataclass(frozen=True)
class DatasetObjects:
    name: str
    subjects: tuple[int, ...]
    sessions: tuple[str, ...]
    classes: tuple[str, ...]
    channels: tuple[str, ...]
    U: Mapping[str, np.ndarray]
    proportions: Mapping[str, np.ndarray]
    counts: Mapping[str, np.ndarray]
    class_means: Mapping[str, np.ndarray]
    marginal_means: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class TwoViewFit:
    mean0: np.ndarray
    mean1: np.ndarray
    left: np.ndarray
    right: np.ndarray
    singular_values: np.ndarray
    scale0: np.ndarray
    scale1: np.ndarray


def load_config(repo_root: str | Path) -> tuple[dict[str, Any], str]:
    root = Path(repo_root).resolve()
    path = root / CONFIG_PATH
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise DataContractError("population structure config must be a mapping")
    protocol = root / str(config["protocol"]["protocol_path"])
    observed = sha256_file(protocol)
    expected = str(config["protocol"]["protocol_sha256"])
    if observed != expected:
        raise DataContractError(f"protocol SHA mismatch: {observed} != {expected}")
    return config, sha256_file(path)


def validate_parent_hashes(repo_root: str | Path, config: Mapping[str, Any]) -> dict[str, str]:
    root = Path(repo_root).resolve()
    observed: dict[str, str] = {}
    for name, record in config["parent_artifacts"].items():
        path = root / str(record["path"])
        if not path.is_file():
            raise DataContractError(f"missing immutable parent artifact: {path}")
        value = sha256_file(path)
        if value != str(record["sha256"]):
            raise DataContractError(f"parent hash mismatch for {name}: {value}")
        observed[str(name)] = value
    return observed


def _prefix(split: str, field: str) -> str:
    return f"AIRM__session_specific__{split}__{field}"


def _validate_finite_symmetric(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise DataContractError(f"{name} contains NaN/Inf")
    if value.ndim >= 2 and value.shape[-1] == value.shape[-2]:
        if not np.array_equal(value, np.swapaxes(value, -1, -2)):
            error = float(np.max(np.abs(value - np.swapaxes(value, -1, -2))))
            raise DataContractError(f"{name} is not exactly symmetric: {error}")


def load_parent_dataset(repo_root: str | Path, config: Mapping[str, Any], dataset: str) -> DatasetObjects:
    root = Path(repo_root).resolve()
    if dataset not in {"openbmi", "bnci"}:
        raise ValueError("dataset must be openbmi or bnci")
    spec = config["datasets"][dataset]
    artifact_key = "openbmi_objects" if dataset == "openbmi" else "bnci_objects"
    archive_path = root / str(config["parent_artifacts"][artifact_key]["path"])
    subjects = tuple(int(x) for x in spec["subjects"])
    sessions = tuple(str(x) for x in spec["sessions"])
    classes = tuple(str(x) for x in spec["classes"])
    channels = tuple(str(x) for x in spec["channels"])
    n, q, k, d = len(subjects), len(sessions), len(classes), len(channels)
    expected_full = int(spec["full_trials_per_cell"])
    expected_half = int(spec["half_trials_per_cell"])
    banks: dict[str, dict[str, np.ndarray]] = {
        field: {} for field in ("U", "proportions", "counts", "class_means", "marginal_means")
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        for split in SPLITS:
            required = {
                "U": _prefix(split, "U"),
                "proportions": _prefix(split, "class_proportions"),
                "counts": _prefix(split, "class_counts"),
                "class_means": _prefix(split, "class_means"),
                "marginal_means": _prefix(split, "marginal_means"),
            }
            missing = [key for key in required.values() if key not in archive.files]
            if missing:
                raise DataContractError(f"missing parent keys: {missing}")
            for field, key in required.items():
                banks[field][split] = np.asarray(archive[key]).copy()
            if banks["U"][split].shape != (n, q, k, d, d):
                raise DataContractError(f"{dataset} {split} U shape mismatch")
            if banks["proportions"][split].shape != (n, q, k):
                raise DataContractError(f"{dataset} {split} proportions shape mismatch")
            if banks["counts"][split].shape != (n, q, k):
                raise DataContractError(f"{dataset} {split} count shape mismatch")
            if banks["class_means"][split].shape != (n, q, k, d, d):
                raise DataContractError(f"{dataset} {split} class-mean shape mismatch")
            if banks["marginal_means"][split].shape != (n, q, d, d):
                raise DataContractError(f"{dataset} {split} marginal-mean shape mismatch")
            expected = expected_full if split == "F" else expected_half
            if not np.all(banks["counts"][split] == expected):
                raise DataContractError(f"{dataset} {split} class-count mismatch")
            np.testing.assert_allclose(
                banks["proportions"][split].sum(axis=2), 1.0, rtol=0.0, atol=2e-16
            )
            for field in ("U", "proportions", "class_means", "marginal_means"):
                _validate_finite_symmetric(f"{dataset}:{split}:{field}", banks[field][split])
            for field in ("class_means", "marginal_means"):
                values = banks[field][split]
                if float(np.min(np.linalg.eigvalsh(values))) <= 0.0:
                    raise DataContractError(f"{dataset}:{split}:{field} is not SPD")
        if dataset == "bnci":
            if "metadata_json" not in archive.files:
                raise DataContractError("BNCI parent archive lacks ordering metadata")
            metadata = json.loads(str(archive["metadata_json"]))
            record = metadata["AIRM__session_specific__F"]
            if tuple(record["subjects"]) != subjects or tuple(record["sessions"]) != sessions or tuple(record["classes"]) != classes:
                raise DataContractError("BNCI embedded ordering mismatch")
    if dataset == "openbmi":
        manifest_path = root / str(config["parent_artifacts"]["openbmi_protocol_manifest"]["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tuple(manifest["subject_ids"]) != subjects or tuple(manifest["session_ids"]) != sessions:
            raise DataContractError("OpenBMI manifest subject/session ordering mismatch")
        if tuple(manifest["classes"]) != classes or tuple(manifest["eeg_channels"]) != channels:
            raise DataContractError("OpenBMI manifest class/channel ordering mismatch")
    return DatasetObjects(
        name=dataset,
        subjects=subjects,
        sessions=sessions,
        classes=classes,
        channels=channels,
        U=banks["U"],
        proportions=banks["proportions"],
        counts=banks["counts"],
        class_means=banks["class_means"],
        marginal_means=banks["marginal_means"],
    )


def svec(value: np.ndarray) -> np.ndarray:
    """Batch Frobenius-isometric upper-triangle vectorization."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-1] != array.shape[-2]:
        raise ValueError("svec requires square matrices")
    rows, cols = np.triu_indices(array.shape[-1])
    result = array[..., rows, cols].copy()
    result[..., rows != cols] *= np.sqrt(2.0)
    return result


def helmert_matrix(k: int) -> np.ndarray:
    if k < 2:
        raise ValueError("Helmert requires at least two classes")
    value = np.zeros((k - 1, k), dtype=np.float64)
    for row in range(1, k):
        denominator = math.sqrt(row * (row + 1))
        value[row - 1, :row] = 1.0 / denominator
        value[row - 1, row] = -row / denominator
    return value


def validate_helmert(config: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(config["datasets"]["bnci"]["helmert"], dtype=np.float64)
    expected = helmert_matrix(4)
    np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(matrix @ np.ones(4), 0.0, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), rtol=0.0, atol=3e-16)
    digest = hashlib.sha256(canonical_json_bytes(matrix.tolist())).hexdigest()
    if digest != str(config["datasets"]["bnci"]["helmert_sha256"]):
        raise DataContractError("Helmert literal hash mismatch")
    return matrix


def _normalize_vectors(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(array, axis=-1)
    threshold = np.finfo(np.float64).eps * np.sqrt(array.shape[-1])
    if np.any(~np.isfinite(norms)) or np.any(norms <= threshold):
        raise NumericalContractError("degenerate interaction signature")
    return array / norms[..., None], norms


def _apply_class_mapping(
    U: np.ndarray,
    proportions: np.ndarray,
    mapping: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if mapping is None:
        return np.asarray(U), np.asarray(proportions)
    indices = np.asarray(mapping, dtype=np.int64)
    n, q, k = proportions.shape
    if indices.shape != (n, q, k):
        raise ValueError("class mapping shape mismatch")
    expected = np.arange(k)
    if any(not np.array_equal(np.sort(indices[s, view]), expected) for s in range(n) for view in range(q)):
        raise DataContractError("class mapping is not a permutation")
    mapped_u = np.take_along_axis(U, indices[..., None, None], axis=2)
    mapped_pi = np.take_along_axis(proportions, indices, axis=2)
    return mapped_u, mapped_pi


def _apply_session1_mapping(
    U: np.ndarray,
    proportions: np.ndarray,
    mapping: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if mapping is None:
        return np.asarray(U), np.asarray(proportions)
    indices = np.asarray(mapping, dtype=np.int64)
    if indices.shape != (len(U),) or not np.array_equal(np.sort(indices), np.arange(len(U))):
        raise DataContractError("session-1 mapping must be a complete subject permutation")
    mapped_u = np.asarray(U).copy()
    mapped_pi = np.asarray(proportions).copy()
    mapped_u[:, 1] = U[indices, 1]
    mapped_pi[:, 1] = proportions[indices, 1]
    return mapped_u, mapped_pi


def reconstruct_fold_z(
    U: np.ndarray,
    proportions: np.ndarray,
    train: Sequence[int],
    test: Sequence[int],
    *,
    class_mapping: np.ndarray | None = None,
    session1_mapping: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct training-LOO and held-out Z without parent final Z."""
    train_indices = np.asarray(train, dtype=np.int64)
    test_indices = np.asarray(test, dtype=np.int64)
    if len(train_indices) < 3 or np.intersect1d(train_indices, test_indices).size:
        raise DataContractError("invalid train/test subject partition")
    values, weights = _apply_class_mapping(U, proportions, class_mapping)
    values, weights = _apply_session1_mapping(values, weights, session1_mapping)
    total = np.sum(values[train_indices], axis=0, dtype=np.float64)
    train_templates = (total[None] - values[train_indices]) / float(len(train_indices) - 1)
    held_template = total / float(len(train_indices))
    r_train = values[train_indices] - train_templates
    r_test = values[test_indices] - held_template[None]
    rbar_train = np.einsum("sqc,sqcij->sqij", weights[train_indices], r_train, optimize=True)
    rbar_test = np.einsum("sqc,sqcij->sqij", weights[test_indices], r_test, optimize=True)
    z_train = r_train - rbar_train[:, :, None]
    z_test = r_test - rbar_test[:, :, None]
    z_train = 0.5 * (z_train + np.swapaxes(z_train, -1, -2))
    z_test = 0.5 * (z_test + np.swapaxes(z_test, -1, -2))
    for name, z, pi in (("train", z_train, weights[train_indices]), ("test", z_test, weights[test_indices])):
        if not np.isfinite(z).all():
            raise NumericalContractError(f"{name} Z is nonfinite")
        weighted = np.einsum("sqc,sqcij->sqij", pi, z, optimize=True)
        denominator = np.maximum(np.linalg.norm(z, axis=(-2, -1)).sum(axis=2), np.finfo(float).tiny)
        error = float(np.max(np.linalg.norm(weighted, axis=(-2, -1)) / denominator))
        if error > 2e-14:
            raise NumericalContractError(f"{name} weighted Z zero gate failed: {error}")
    audit = {
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "train_template_sources": [
            [int(x) for x in train_indices if x != target] for target in train_indices
        ],
        "held_template_sources": train_indices.tolist(),
    }
    return z_train, z_test, audit


def signature_from_z(
    z: np.ndarray,
    dataset: str,
    *,
    helmert: np.ndarray | None = None,
    normalize: bool = True,
    spectrum: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.linalg.eigvalsh(z) if spectrum else svec(z)
    if dataset == "openbmi":
        if values.shape[2] != 2:
            raise DataContractError("OpenBMI requires two classes")
        raw = 0.5 * (values[:, :, 1] - values[:, :, 0])
    elif dataset == "bnci":
        if helmert is None:
            raise ValueError("BNCI signature requires Helmert matrix")
        raw = np.einsum("ak,sqkm->sqam", helmert, values, optimize=True).reshape(
            values.shape[0], values.shape[1], -1
        )
    else:
        raise ValueError("unknown dataset")
    if normalize:
        return _normalize_vectors(raw)
    norms = np.linalg.norm(raw, axis=-1)
    if np.any(~np.isfinite(raw)) or np.any(norms <= np.finfo(float).eps * np.sqrt(raw.shape[-1])):
        raise NumericalContractError("degenerate magnitude signature")
    return raw, norms


def generalized_eigen_signature(
    class_means: np.ndarray,
    marginal_means: np.ndarray,
    dataset: str,
    *,
    helmert: np.ndarray | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n, q, k, d, _ = class_means.shape
    values = np.empty((n, q, k, d), dtype=np.float64)
    for s in range(n):
        for view in range(q):
            for c in range(k):
                eigenvalues = linalg.eigvalsh(class_means[s, view, c], marginal_means[s, view])
                if np.any(eigenvalues <= 0.0) or not np.isfinite(eigenvalues).all():
                    raise NumericalContractError("generalized eigenvalue gate failed")
                values[s, view, c] = np.log(eigenvalues)
    if dataset == "openbmi":
        raw = 0.5 * (values[:, :, 1] - values[:, :, 0])
    else:
        if helmert is None:
            raise ValueError("BNCI generalized eigen signature requires Helmert")
        raw = np.einsum("ak,sqkm->sqam", helmert, values, optimize=True).reshape(n, q, -1)
    return _normalize_vectors(raw) if normalize else (raw, np.linalg.norm(raw, axis=-1))


def fold_features(
    data: DatasetObjects,
    split: str,
    train: Sequence[int],
    test: Sequence[int],
    *,
    helmert: np.ndarray | None,
    kind: str = "sensor",
    class_mapping: np.ndarray | None = None,
    session1_mapping: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalize = kind != "magnitude"
    if kind in {"sensor", "magnitude", "spectrum"}:
        z_train, z_test, audit = reconstruct_fold_z(
            data.U[split], data.proportions[split], train, test,
            class_mapping=class_mapping, session1_mapping=session1_mapping,
        )
        x_train, _ = signature_from_z(
            z_train, data.name, helmert=helmert, normalize=normalize, spectrum=kind == "spectrum"
        )
        x_test, _ = signature_from_z(
            z_test, data.name, helmert=helmert, normalize=normalize, spectrum=kind == "spectrum"
        )
        return x_train, x_test, audit
    if kind == "generalized_eigen":
        means = data.class_means[split]
        pi = data.proportions[split]
        means, pi = _apply_class_mapping(means, pi, class_mapping)
        marginals = data.marginal_means[split]
        if session1_mapping is not None:
            mapping = np.asarray(session1_mapping, dtype=np.int64)
            means = means.copy()
            marginals = marginals.copy()
            means[:, 1] = means[mapping, 1]
            marginals[:, 1] = marginals[mapping, 1]
        all_features, _ = generalized_eigen_signature(
            means, marginals, data.name, helmert=helmert, normalize=True
        )
        return all_features[np.asarray(train)], all_features[np.asarray(test)], {
            "train_indices": list(map(int, train)), "test_indices": list(map(int, test)),
            "held_template_sources": [], "train_template_sources": [],
        }
    raise ValueError(f"unknown feature kind: {kind}")


def fit_two_view(x0: np.ndarray, x1: np.ndarray, max_rank: int) -> TwoViewFit:
    a = np.asarray(x0, dtype=np.float64)
    b = np.asarray(x1, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or len(a) < 4:
        raise ValueError("paired training matrices must have matching 2D shapes")
    permitted = min(a.shape[1], len(a) - 2)
    if max_rank < 1 or max_rank > permitted:
        raise NumericalContractError(f"rank {max_rank} exceeds {permitted}")
    mean0 = np.mean(a, axis=0, dtype=np.float64)
    mean1 = np.mean(b, axis=0, dtype=np.float64)
    centered0 = a - mean0
    centered1 = b - mean1
    u0, s0, vh0 = np.linalg.svd(centered0, full_matrices=False)
    u1, s1, vh1 = np.linalg.svd(centered1, full_matrices=False)
    middle = (s0[:, None] * (u0.T @ u1) * s1[None, :]) / float(len(a) - 1)
    p, singular, qh = np.linalg.svd(middle, full_matrices=False)
    left = vh0.T @ p[:, :max_rank]
    right = vh1.T @ qh.T[:, :max_rank]
    scores0 = centered0 @ left
    scores1 = centered1 @ right
    scale0 = np.std(scores0, axis=0, ddof=1)
    scale1 = np.std(scores1, axis=0, ddof=1)
    threshold = np.finfo(float).eps * np.sqrt(a.shape[1])
    if np.any(scale0 <= threshold) or np.any(scale1 <= threshold):
        raise NumericalContractError("projected training-score scale is degenerate")
    arrays = (mean0, mean1, left, right, singular[:max_rank], scale0, scale1)
    if any(not np.isfinite(value).all() for value in arrays):
        raise NumericalContractError("two-view fit is nonfinite")
    return TwoViewFit(*arrays)


def project_two_view(fit: TwoViewFit, x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score0 = (np.asarray(x0) - fit.mean0) @ fit.left / fit.scale0
    score1 = (np.asarray(x1) - fit.mean1) @ fit.right / fit.scale1
    if not np.isfinite(score0).all() or not np.isfinite(score1).all():
        raise NumericalContractError("projected scores are nonfinite")
    return score0, score1


def separation_from_scores(score0: np.ndarray, score1: np.ndarray, rank: int) -> dict[str, np.ndarray | float]:
    if rank < 1 or rank > score0.shape[1] or score0.shape != score1.shape:
        raise ValueError("invalid score/rank contract")
    similarity = (score0[:, :rank] @ score1[:, :rank].T) / float(rank)
    n = len(similarity)
    if n < 2:
        raise ValueError("same-subject separation requires at least two held-out subjects")
    forward = np.empty(n, dtype=np.float64)
    reverse = np.empty(n, dtype=np.float64)
    for index in range(n):
        forward[index] = similarity[index, index] - np.median(np.delete(similarity[index], index))
        reverse[index] = similarity[index, index] - np.median(np.delete(similarity[:, index], index))
    average = 0.5 * (forward + reverse)
    return {
        "similarity": similarity,
        "forward": forward,
        "reverse": reverse,
        "average": average,
        "statistic": float(np.median(average)),
        "forward_median": float(np.median(forward)),
        "reverse_median": float(np.median(reverse)),
    }


def full_space_separation(x0: np.ndarray, x1: np.ndarray) -> dict[str, np.ndarray | float]:
    similarity = np.asarray(x0) @ np.asarray(x1).T
    n = len(similarity)
    forward = np.empty(n, dtype=np.float64)
    reverse = np.empty(n, dtype=np.float64)
    for index in range(n):
        forward[index] = similarity[index, index] - np.median(np.delete(similarity[index], index))
        reverse[index] = similarity[index, index] - np.median(np.delete(similarity[:, index], index))
    average = 0.5 * (forward + reverse)
    return {
        "similarity": similarity, "forward": forward, "reverse": reverse,
        "average": average, "statistic": float(np.median(average)),
        "forward_median": float(np.median(forward)), "reverse_median": float(np.median(reverse)),
    }


def select_rank_one_se(
    fold_scores: np.ndarray,
    ranks: Sequence[int],
) -> dict[str, Any]:
    values = np.asarray(fold_scores, dtype=np.float64)
    rank_values = np.asarray(ranks, dtype=np.int64)
    if values.shape[1] != len(rank_values) or not np.isfinite(values).all():
        raise NumericalContractError("inner rank score contract failure")
    means = np.mean(values, axis=0)
    standard_errors = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    best_index = int(np.flatnonzero(means == np.max(means))[0])
    threshold = means[best_index] - standard_errors[best_index]
    eligible = np.flatnonzero(means >= threshold)
    selected_index = int(eligible[np.argmin(rank_values[eligible])])
    return {
        "selected_rank": int(rank_values[selected_index]),
        "best_rank": int(rank_values[best_index]),
        "threshold": float(threshold),
        "means": means,
        "standard_errors": standard_errors,
        "fold_scores": values,
    }


def deterministic_rng(master_seed: int, *parts: Any) -> np.random.Generator:
    payload = {"master_seed": int(master_seed), "parts": list(parts)}
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32)
    return np.random.Generator(np.random.PCG64DXSM(words))


def fixed_point_free(indices: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if len(values) < 2:
        raise ValueError("derangement requires at least two indices")
    for _ in range(10000):
        candidate = rng.permutation(values)
        if np.all(candidate != values):
            return candidate
    raise NumericalContractError("failed to generate deterministic derangement")


def subject_indices(subjects: Sequence[int], selected: Sequence[int]) -> np.ndarray:
    lookup = {int(subject): index for index, subject in enumerate(subjects)}
    try:
        return np.asarray([lookup[int(subject)] for subject in selected], dtype=np.int64)
    except KeyError as error:
        raise DataContractError(f"unknown subject in fold: {error}") from error


def openbmi_fold_indices(data: DatasetObjects, config: Mapping[str, Any]) -> tuple[list[np.ndarray], dict[int, list[np.ndarray]]]:
    outer = [subject_indices(data.subjects, fold) for fold in config["openbmi_folds"]["outer_test"]]
    inner = {
        index: [subject_indices(data.subjects, fold) for fold in config["openbmi_folds"]["inner_validation"][str(index)]]
        for index in range(len(outer))
    }
    coverage = np.concatenate(outer)
    if not np.array_equal(np.sort(coverage), np.arange(len(data.subjects))) or len(np.unique(coverage)) != len(data.subjects):
        raise DataContractError("outer fold coverage is not exactly once")
    all_indices = set(range(len(data.subjects)))
    for index, test in enumerate(outer):
        train = all_indices - set(test.tolist())
        inner_coverage = np.concatenate(inner[index])
        if set(inner_coverage.tolist()) != train or len(np.unique(inner_coverage)) != len(train):
            raise DataContractError(f"inner fold coverage failure for outer {index}")
    payload = {
        "outer": [[data.subjects[i] for i in fold] for fold in outer],
        "inner": {str(i): [[data.subjects[j] for j in fold] for fold in inner[i]] for i in range(len(outer))},
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if digest != str(config["openbmi_folds"]["sha256"]):
        raise DataContractError(f"fold hash mismatch: {digest}")
    return outer, inner


def generate_openbmi_pairing_mappings(
    data: DatasetObjects,
    config: Mapping[str, Any],
    replicates: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate exact partition-local outer and inner session-1 mappings."""
    outer_folds, inner_folds = openbmi_fold_indices(data, config)
    n = len(data.subjects)
    outer_maps = np.empty((replicates, len(outer_folds), n), dtype=np.int16)
    inner_maps = np.empty((replicates, len(outer_folds), 5, n), dtype=np.int16)
    all_indices = np.arange(n, dtype=np.int64)
    master = int(config["protocol"]["master_seed"])
    for replicate in range(replicates):
        for outer_index, outer_test in enumerate(outer_folds):
            outer_train = np.setdiff1d(all_indices, outer_test, assume_unique=True)
            mapping = all_indices.copy()
            for label, partition in (("train", outer_train), ("test", outer_test)):
                rng = deterministic_rng(master, "pairing", replicate, outer_index, "outer", label)
                mapping[partition] = fixed_point_free(partition, rng)
            outer_maps[replicate, outer_index] = mapping
            if np.any(mapping[outer_train] == outer_train) or np.any(mapping[outer_test] == outer_test):
                raise NumericalContractError("outer pairing map has a fixed point")
            for inner_index, validation in enumerate(inner_folds[outer_index]):
                inner_train = np.setdiff1d(outer_train, validation, assume_unique=True)
                inner_mapping = all_indices.copy()
                for label, partition in (("train", inner_train), ("test", validation)):
                    rng = deterministic_rng(
                        master, "pairing", replicate, outer_index, inner_index, "inner", label
                    )
                    inner_mapping[partition] = fixed_point_free(partition, rng)
                inner_maps[replicate, outer_index, inner_index] = inner_mapping
                if np.any(inner_mapping[inner_train] == inner_train) or np.any(inner_mapping[validation] == validation):
                    raise NumericalContractError("inner pairing map has a fixed point")
    return outer_maps, inner_maps


def generate_openbmi_class_mappings(
    data: DatasetObjects,
    config: Mapping[str, Any],
    replicates: int,
) -> np.ndarray:
    master = int(config["protocol"]["master_seed"])
    mappings = np.empty((replicates, len(data.subjects), 2, 2), dtype=np.int8)
    for replicate in range(replicates):
        rng = deterministic_rng(master, "class_semantics", replicate)
        swaps = rng.integers(0, 2, size=(len(data.subjects), 2), dtype=np.int8)
        if np.all(swaps == swaps.flat[0]):
            swaps[0, 0] = 1 - swaps[0, 0]
        mappings[replicate, ..., 0] = swaps
        mappings[replicate, ..., 1] = 1 - swaps
        # swap=0 maps [0,1], while swap=1 maps [1,0]
        mappings[replicate] = np.where(
            swaps[..., None] == 0,
            np.asarray([0, 1], dtype=np.int8),
            np.asarray([1, 0], dtype=np.int8),
        )
    return mappings


def _rank_grid(config: Mapping[str, Any], dataset: str, n_train: int, p: int) -> list[int]:
    key = "openbmi_grid" if dataset == "openbmi" else "bnci_grid"
    permitted = min(p, n_train - 2)
    values = [int(rank) for rank in config["rank"][key] if int(rank) <= permitted]
    if not values:
        raise NumericalContractError("rank grid is empty after sample-size restriction")
    return values


def _score_feature_fold(
    x_train: np.ndarray,
    x_test: np.ndarray,
    ranks: Sequence[int],
) -> tuple[TwoViewFit, dict[int, dict[str, Any]]]:
    fit = fit_two_view(x_train[:, 0], x_train[:, 1], max(ranks))
    score0, score1 = project_two_view(fit, x_test[:, 0], x_test[:, 1])
    return fit, {int(rank): separation_from_scores(score0, score1, int(rank)) for rank in ranks}


def _inner_rank_selection_openbmi(
    data: DatasetObjects,
    config: Mapping[str, Any],
    outer_index: int,
    outer_train: np.ndarray,
    inner_validation: Sequence[np.ndarray],
    *,
    kind: str,
    helmert: np.ndarray | None,
    class_mapping: np.ndarray | None,
    inner_pairing_mappings: np.ndarray | None,
) -> dict[str, Any]:
    all_indices = np.arange(len(data.subjects), dtype=np.int64)
    ranks = _rank_grid(config, data.name, len(outer_train) - len(inner_validation[0]), _feature_dimension(data, kind))
    scores = np.empty((len(inner_validation), len(ranks)), dtype=np.float64)
    audits: list[dict[str, Any]] = []
    for inner_index, validation in enumerate(inner_validation):
        train = np.setdiff1d(outer_train, validation, assume_unique=True)
        pairing = None if inner_pairing_mappings is None else inner_pairing_mappings[inner_index]
        x_train, x_validation, audit = fold_features(
            data, "F", train, validation, helmert=helmert, kind=kind,
            class_mapping=class_mapping, session1_mapping=pairing,
        )
        _, by_rank = _score_feature_fold(x_train, x_validation, ranks)
        scores[inner_index] = [by_rank[rank]["statistic"] for rank in ranks]
        audits.append(audit)
        if pairing is not None:
            # The current validation's raw session-1 rows may not occur in inner training.
            if set(pairing[train].tolist()) & set(pairing[validation].tolist()):
                raise DataContractError("inner pairing mapping crosses train/validation")
    result = select_rank_one_se(scores, ranks)
    result["audits"] = audits
    result["outer_index"] = int(outer_index)
    return result


def _feature_dimension(data: DatasetObjects, kind: str) -> int:
    d = len(data.channels)
    if kind in {"sensor", "magnitude"}:
        base = d * (d + 1) // 2
    elif kind in {"spectrum", "generalized_eigen"}:
        base = d
    else:
        raise ValueError(f"unknown kind {kind}")
    return base if data.name == "openbmi" else (len(data.classes) - 1) * base


def evaluate_openbmi_nested(
    data: DatasetObjects,
    config: Mapping[str, Any],
    *,
    kind: str = "sensor",
    class_mapping: np.ndarray | None = None,
    outer_pairing_mappings: np.ndarray | None = None,
    inner_pairing_mappings: np.ndarray | None = None,
    keep_fits: bool = False,
) -> dict[str, Any]:
    if data.name != "openbmi":
        raise ValueError("OpenBMI nested evaluator received another dataset")
    outer_folds, inner_folds = openbmi_fold_indices(data, config)
    all_indices = np.arange(len(data.subjects), dtype=np.int64)
    helmert = None
    subject_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_rank_subject_rows: list[dict[str, Any]] = []
    fits: list[TwoViewFit] = []
    similarity_blocks: list[np.ndarray] = []
    full_subject_rows: list[dict[str, Any]] = []
    for outer_index, test in enumerate(outer_folds):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        outer_pairing = None if outer_pairing_mappings is None else outer_pairing_mappings[outer_index]
        inner_pairing = None if inner_pairing_mappings is None else inner_pairing_mappings[outer_index]
        selection = _inner_rank_selection_openbmi(
            data, config, outer_index, train, inner_folds[outer_index], kind=kind,
            helmert=helmert, class_mapping=class_mapping,
            inner_pairing_mappings=inner_pairing,
        )
        outer_ranks = _rank_grid(config, data.name, len(train), _feature_dimension(data, kind))
        x_train, x_test, audit = fold_features(
            data, "F", train, test, helmert=helmert, kind=kind,
            class_mapping=class_mapping, session1_mapping=outer_pairing,
        )
        if outer_pairing is not None and set(outer_pairing[train].tolist()) & set(outer_pairing[test].tolist()):
            raise DataContractError("outer pairing mapping crosses train/test")
        fit, by_rank = _score_feature_fold(x_train, x_test, outer_ranks)
        selected_rank = int(selection["selected_rank"])
        selected = by_rank[selected_rank]
        full = full_space_separation(x_test[:, 0], x_test[:, 1])
        if keep_fits:
            fits.append(fit)
            similarity_blocks.append(np.asarray(selected["similarity"]))
        fold_rows.append({
            "outer_fold": outer_index,
            "train_subjects": [data.subjects[i] for i in train],
            "test_subjects": [data.subjects[i] for i in test],
            "selected_rank": selected_rank,
            "best_rank": int(selection["best_rank"]),
            "one_se_threshold": float(selection["threshold"]),
            "inner_means": {str(rank): float(value) for rank, value in zip(_rank_grid(config, data.name, len(train) - len(inner_folds[outer_index][0]), _feature_dimension(data, kind)), selection["means"])},
            "inner_standard_errors": {str(rank): float(value) for rank, value in zip(_rank_grid(config, data.name, len(train) - len(inner_folds[outer_index][0]), _feature_dimension(data, kind)), selection["standard_errors"])},
            "selected_statistic": float(selected["statistic"]),
            "forward_median": float(selected["forward_median"]),
            "reverse_median": float(selected["reverse_median"]),
            "full_space_statistic": float(full["statistic"]),
            "singular_values": fit.singular_values.tolist(),
            "audit": audit,
        })
        for local, subject_index in enumerate(test):
            base = {
                "subject": int(data.subjects[subject_index]), "outer_fold": outer_index,
                "selected_rank": selected_rank,
            }
            subject_rows.append({
                **base,
                "delta_forward": float(selected["forward"][local]),
                "delta_reverse": float(selected["reverse"][local]),
                "delta_average": float(selected["average"][local]),
            })
            full_subject_rows.append({
                **base,
                "delta_forward": float(full["forward"][local]),
                "delta_reverse": float(full["reverse"][local]),
                "delta_average": float(full["average"][local]),
            })
            for rank in outer_ranks:
                score = by_rank[rank]
                all_rank_subject_rows.append({
                    "subject": int(data.subjects[subject_index]), "outer_fold": outer_index,
                    "rank": int(rank), "delta_forward": float(score["forward"][local]),
                    "delta_reverse": float(score["reverse"][local]),
                    "delta_average": float(score["average"][local]),
                })
    subject_frame = pd.DataFrame(subject_rows).sort_values("subject").reset_index(drop=True)
    full_frame = pd.DataFrame(full_subject_rows).sort_values("subject").reset_index(drop=True)
    rank_frame = pd.DataFrame(all_rank_subject_rows).sort_values(["rank", "subject"]).reset_index(drop=True)
    selected_statistic = float(np.median(subject_frame["delta_average"]))
    result: dict[str, Any] = {
        "statistic": selected_statistic,
        "forward_median": float(np.median(subject_frame["delta_forward"])),
        "reverse_median": float(np.median(subject_frame["delta_reverse"])),
        "subject_rows": subject_frame,
        "fold_rows": fold_rows,
        "rank_rows": rank_frame,
        "full_space_statistic": float(np.median(full_frame["delta_average"])),
        "full_space_subject_rows": full_frame,
        "selected_ranks": np.asarray([row["selected_rank"] for row in fold_rows], dtype=np.int64),
    }
    if keep_fits:
        result["fits"] = fits
        result["similarity_blocks"] = similarity_blocks
    return result


def leave_one_subject_influence(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.asarray([np.median(np.delete(array, index)) for index in range(len(array))])


def bootstrap_median_ci(
    values: Sequence[float], config: Mapping[str, Any]
) -> tuple[float, float, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    replicates = int(config["bootstrap"]["replicates"])
    rng = deterministic_rng(int(config["protocol"]["master_seed"]), config["bootstrap"]["seed_namespace"])
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    statistics = np.median(array[indices], axis=1)
    confidence = float(config["bootstrap"]["confidence"])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(statistics, [alpha, 1.0 - alpha])
    return float(low), float(high), statistics


def monte_carlo_p(observed: float, null: Sequence[float]) -> float:
    values = np.asarray(null, dtype=np.float64)
    if not np.isfinite(observed) or not np.isfinite(values).all():
        raise NumericalContractError("nonfinite Monte Carlo statistic")
    return float((1 + np.count_nonzero(values >= observed)) / (1 + len(values)))


def terminal_decision(
    *,
    data_contract_pass: bool,
    reliability_pass: bool,
    statistic: float,
    forward_median: float,
    reverse_median: float,
    pairing_p: float,
    class_p: float,
    random_p: float,
    influence_positive: bool,
    full_space_stable: bool,
    selected_ranks: Sequence[int],
    low_cap: int = 8,
) -> str:
    if not data_contract_pass:
        return "UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE"
    if not reliability_pass:
        return "UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE"
    alpha = 0.05
    structure = (
        statistic > 0.0 and forward_median > 0.0 and reverse_median > 0.0
        and pairing_p <= alpha and class_p <= alpha and influence_positive and full_space_stable
    )
    if not structure:
        return "STOP_NO_HELDOUT_POPULATION_STRUCTURE"
    if random_p > alpha:
        return "STOP_RANDOM_SUBSPACE_EQUIVALENT"
    ranks = np.asarray(selected_ranks, dtype=np.int64)
    low = float(np.median(ranks)) <= low_cap and int(np.count_nonzero(ranks <= low_cap)) >= 4
    return (
        "GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION"
        if low else "GO_STRUCTURED_BUT_NOT_LOW_DIMENSIONAL"
    )


def _single_session_z(
    U: np.ndarray,
    proportions: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    mapping: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(U)
    weights = np.asarray(proportions)
    if mapping is not None:
        values = values[np.asarray(mapping)]
        weights = weights[np.asarray(mapping)]
    z_train, z_test, _ = reconstruct_fold_z(
        values[:, None], weights[:, None], train, test
    )
    return z_train[:, 0], z_test[:, 0]


def generate_reliability_mappings(
    data: DatasetObjects,
    config: Mapping[str, Any],
    replicates: int,
) -> np.ndarray:
    outer, _ = openbmi_fold_indices(data, config)
    n = len(data.subjects)
    all_indices = np.arange(n, dtype=np.int64)
    result = np.empty((replicates, len(outer), 2, n), dtype=np.int16)
    master = int(config["protocol"]["master_seed"])
    for replicate in range(replicates):
        for fold_index, test in enumerate(outer):
            train = np.setdiff1d(all_indices, test, assume_unique=True)
            for session in range(2):
                mapping = all_indices.copy()
                for label, partition in (("train", train), ("test", test)):
                    rng = deterministic_rng(
                        master, "reliability_pairing", replicate, fold_index, session, label
                    )
                    mapping[partition] = fixed_point_free(partition, rng)
                result[replicate, fold_index, session] = mapping
    return result


def evaluate_openbmi_reliability(
    data: DatasetObjects,
    config: Mapping[str, Any],
    *,
    mappings: np.ndarray | None = None,
) -> dict[str, Any]:
    if data.name != "openbmi":
        raise ValueError("reliability primary is OpenBMI")
    outer, _ = openbmi_fold_indices(data, config)
    all_indices = np.arange(len(data.subjects), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for fold_index, test in enumerate(outer):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        for session in range(2):
            _, z_a = _single_session_z(
                data.U["A"][:, session], data.proportions["A"][:, session], train, test
            )
            mapping = None if mappings is None else np.asarray(mappings[fold_index, session], dtype=np.int64)
            _, z_b = _single_session_z(
                data.U["B"][:, session], data.proportions["B"][:, session], train, test, mapping
            )
            x_a, _ = signature_from_z(z_a[:, None], "openbmi", normalize=True)
            x_b, _ = signature_from_z(z_b[:, None], "openbmi", normalize=True)
            score = full_space_separation(x_a[:, 0], x_b[:, 0])
            for local, subject_index in enumerate(test):
                rows.append({
                    "subject": int(data.subjects[subject_index]), "outer_fold": fold_index,
                    "session": str(data.sessions[session]),
                    "delta_forward": float(score["forward"][local]),
                    "delta_reverse": float(score["reverse"][local]),
                    "delta_average": float(score["average"][local]),
                })
    frame = pd.DataFrame(rows).sort_values(["session", "subject"]).reset_index(drop=True)
    session_statistics = {
        session: float(np.median(frame.loc[frame["session"] == session, "delta_average"]))
        for session in data.sessions
    }
    return {"rows": frame, "session_statistics": session_statistics}


def run_reliability_gate(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_real_access_lock(root)
    config, config_hash = load_config(root)
    parent_hashes = validate_parent_hashes(root, config)
    data = load_parent_dataset(root, config, "openbmi")
    observed = evaluate_openbmi_reliability(data, config)
    replicates = int(config["nulls"]["replicates"])
    mappings = generate_reliability_mappings(data, config, replicates)
    null = np.empty((replicates, 2), dtype=np.float64)
    for replicate in range(replicates):
        value = evaluate_openbmi_reliability(data, config, mappings=mappings[replicate])
        null[replicate] = [value["session_statistics"][session] for session in data.sessions]
    rows = observed["rows"]
    summary_rows: list[dict[str, Any]] = []
    session_passes: list[bool] = []
    influence_rows: list[dict[str, Any]] = []
    for session_index, session in enumerate(data.sessions):
        observed_value = float(observed["session_statistics"][session])
        p_value = monte_carlo_p(observed_value, null[:, session_index])
        subject_values = rows.loc[rows["session"] == session].sort_values("subject")["delta_average"].to_numpy()
        influence = leave_one_subject_influence(subject_values)
        influence_pass = bool(np.all(influence > 0.0))
        passed = observed_value > 0.0 and p_value <= float(config["nulls"]["alpha"]) and influence_pass
        session_passes.append(passed)
        summary_rows.append({
            "session": session, "observed": observed_value,
            "null_median": float(np.median(null[:, session_index])),
            "effect": observed_value - float(np.median(null[:, session_index])),
            "p_value": p_value, "leave_one_subject_sign_pass": influence_pass,
            "passed": passed,
        })
        for subject, value in zip(data.subjects, influence):
            influence_rows.append({"session": session, "omitted_subject": subject, "median_without_subject": value})
    passed = bool(all(session_passes))
    output = root / str(config["project"]["output_dir"])
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "nulls").mkdir(parents=True, exist_ok=True)
    (output / "decisions").mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "tables/reliability_subject_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(summary_rows).to_csv(output / "tables/reliability_statistics.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(influence_rows).to_csv(output / "tables/reliability_influence.csv", index=False, lineterminator="\n", float_format="%.17g")
    _atomic_savez(output / "nulls/reliability_pairing_null.npz", {
        "statistics": null, "mappings": mappings,
        "subjects": np.asarray(data.subjects), "sessions": np.asarray(data.sessions),
    })
    result = {
        "schema_version": "subject-class-population-structure-reliability-v1",
        "passed": passed,
        "terminal_if_failed": None if passed else "UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE",
        "sessions": summary_rows, "replicates": replicates,
        "config_sha256": config_hash, "parent_hashes": parent_hashes,
    }
    atomic_write_json(output / "decisions/reliability_gate.json", result)
    if not passed:
        atomic_write_json(output / "decisions/terminal_decision.json", {
            "schema_version": "subject-class-population-structure-terminal-v1",
            "terminal_decision": "UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE",
            "reason": "The frozen split-half reliability prerequisite failed; no real population-rank SVD or statistic is permitted.",
            "reliability_sessions": summary_rows,
            "config_sha256": config_hash,
            "parent_hashes": parent_hashes,
        })
    return result


def _atomic_savez(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    import tempfile
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def same_session_pca_control(
    data: DatasetObjects,
    config: Mapping[str, Any],
    selected_ranks: Sequence[int],
) -> pd.DataFrame:
    outer, _ = openbmi_fold_indices(data, config)
    all_indices = np.arange(len(data.subjects))
    rows: list[dict[str, Any]] = []
    for fold_index, test in enumerate(outer):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        x_train, x_test, _ = fold_features(data, "F", train, test, helmert=None, kind="sensor")
        rank = int(selected_ranks[fold_index])
        direction_values: list[dict[str, np.ndarray | float]] = []
        for source_view in (0, 1):
            centered_source = x_train[:, source_view] - np.mean(x_train[:, source_view], axis=0)
            _, _, vh = np.linalg.svd(centered_source, full_matrices=False)
            basis = vh[:rank].T
            train_scores = []
            test_scores = []
            for view in (0, 1):
                mean = np.mean(x_train[:, view], axis=0)
                projected_train = (x_train[:, view] - mean) @ basis
                scale = np.std(projected_train, axis=0, ddof=1)
                if np.any(scale <= np.finfo(float).eps):
                    raise NumericalContractError("PCA projected scale degeneracy")
                train_scores.append(projected_train / scale)
                test_scores.append((x_test[:, view] - mean) @ basis / scale)
            direction_values.append(separation_from_scores(test_scores[0], test_scores[1], rank))
        for local, subject_index in enumerate(test):
            rows.append({
                "subject": int(data.subjects[subject_index]), "outer_fold": fold_index,
                "rank": rank,
                "delta_session0_pca": float(direction_values[0]["average"][local]),
                "delta_session1_pca": float(direction_values[1]["average"][local]),
                "delta_average": float(0.5 * (direction_values[0]["average"][local] + direction_values[1]["average"][local])),
            })
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def selected_mode_split_half(
    data: DatasetObjects,
    config: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer, _ = openbmi_fold_indices(data, config)
    all_indices = np.arange(len(data.subjects))
    rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    for fold_index, test in enumerate(outer):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        rank = int(observed["selected_ranks"][fold_index])
        fit: TwoViewFit = observed["fits"][fold_index]
        full_train, full_test, _ = fold_features(data, "F", train, test, helmert=None, kind="sensor")
        score0, score1 = project_two_view(fit, full_test[:, 0], full_test[:, 1])
        for local, subject_index in enumerate(test):
            latent_rows.append({
                "subject": int(data.subjects[subject_index]), "outer_fold": fold_index,
                "rank": rank, "score_session0_mode1": float(score0[local, 0]),
                "score_session1_mode1": float(score1[local, 0]),
            })
        for session in (0, 1):
            _, a_test, _ = fold_features(data, "A", train, test, helmert=None, kind="sensor")
            _, b_test, _ = fold_features(data, "B", train, test, helmert=None, kind="sensor")
            basis = fit.left[:, :rank] if session == 0 else fit.right[:, :rank]
            mean = fit.mean0 if session == 0 else fit.mean1
            scale = fit.scale0[:rank] if session == 0 else fit.scale1[:rank]
            a_score = (a_test[:, session] - mean) @ basis / scale
            b_score = (b_test[:, session] - mean) @ basis / scale
            reliability = separation_from_scores(a_score, b_score, rank)
            for local, subject_index in enumerate(test):
                rows.append({
                    "subject": int(data.subjects[subject_index]), "outer_fold": fold_index,
                    "session": str(data.sessions[session]), "rank": rank,
                    "delta_forward": float(reliability["forward"][local]),
                    "delta_reverse": float(reliability["reverse"][local]),
                    "delta_average": float(reliability["average"][local]),
                })
    return (
        pd.DataFrame(rows).sort_values(["session", "subject"]).reset_index(drop=True),
        pd.DataFrame(latent_rows).sort_values("subject").reset_index(drop=True),
    )


def _serialize_openbmi_observed(
    output: Path,
    data: DatasetObjects,
    observed: Mapping[str, Any],
) -> None:
    fits: Sequence[TwoViewFit] = observed["fits"]
    arrays: dict[str, np.ndarray] = {
        "subjects": np.asarray(data.subjects, dtype=np.int64),
        "selected_ranks": np.asarray(observed["selected_ranks"], dtype=np.int64),
        "subject_delta": observed["subject_rows"].sort_values("subject")["delta_average"].to_numpy(),
        "subject_delta_forward": observed["subject_rows"].sort_values("subject")["delta_forward"].to_numpy(),
        "subject_delta_reverse": observed["subject_rows"].sort_values("subject")["delta_reverse"].to_numpy(),
        "full_subject_delta": observed["full_space_subject_rows"].sort_values("subject")["delta_average"].to_numpy(),
        "mean0": np.stack([fit.mean0 for fit in fits]),
        "mean1": np.stack([fit.mean1 for fit in fits]),
        "left": np.stack([fit.left for fit in fits]),
        "right": np.stack([fit.right for fit in fits]),
        "singular_values": np.stack([fit.singular_values for fit in fits]),
        "scale0": np.stack([fit.scale0 for fit in fits]),
        "scale1": np.stack([fit.scale1 for fit in fits]),
    }
    n = len(data.subjects)
    similarity = np.zeros((n, n), dtype=np.float64)
    mask = np.zeros((n, n), dtype=np.uint8)
    outer, _ = openbmi_fold_indices(data, load_config(output.parents[1])[0])
    for fold, block in zip(outer, observed["similarity_blocks"]):
        similarity[np.ix_(fold, fold)] = block
        mask[np.ix_(fold, fold)] = 1
    arrays["heldout_similarity_block_matrix"] = similarity
    arrays["heldout_similarity_mask"] = mask
    _atomic_savez(output / "objects/openbmi_observed_core.npz", arrays)


def run_openbmi_observed(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_real_access_lock(root)
    config, config_hash = load_config(root)
    parent_hashes = validate_parent_hashes(root, config)
    output = root / str(config["project"]["output_dir"])
    reliability_path = output / "decisions/reliability_gate.json"
    if not reliability_path.is_file():
        raise DataContractError("reliability gate must run before OpenBMI structure")
    reliability = json.loads(reliability_path.read_text(encoding="utf-8"))
    if not reliability["passed"]:
        raise DataContractError("UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE")
    data = load_parent_dataset(root, config, "openbmi")
    observed = evaluate_openbmi_nested(data, config, kind="sensor", keep_fits=True)
    low, high, bootstrap = bootstrap_median_ci(
        observed["subject_rows"].sort_values("subject")["delta_average"].to_numpy(), config
    )
    influence = leave_one_subject_influence(
        observed["subject_rows"].sort_values("subject")["delta_average"].to_numpy()
    )
    magnitude = evaluate_openbmi_nested(data, config, kind="magnitude")
    spectrum = evaluate_openbmi_nested(data, config, kind="spectrum")
    generalized = evaluate_openbmi_nested(data, config, kind="generalized_eigen")
    pca = same_session_pca_control(data, config, observed["selected_ranks"])
    split_half, latent = selected_mode_split_half(data, config, observed)
    for directory in ("objects", "tables", "decisions", "nulls", "figures", "report", "protocol"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    observed["subject_rows"].to_csv(output / "tables/openbmi_subject_directional_separation.csv", index=False, lineterminator="\n", float_format="%.17g")
    observed["rank_rows"].to_csv(output / "tables/openbmi_rank_by_rank_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame([
        {key: value for key, value in row.items() if key not in {"audit", "singular_values", "inner_means", "inner_standard_errors", "train_subjects", "test_subjects"}}
        | {
            "train_subjects": "|".join(map(str, row["train_subjects"])),
            "test_subjects": "|".join(map(str, row["test_subjects"])),
        }
        for row in observed["fold_rows"]
    ]).to_csv(output / "tables/openbmi_outer_fold_selected_ranks.csv", index=False, lineterminator="\n", float_format="%.17g")
    fold_membership_rows = []
    for row in observed["fold_rows"]:
        fold_membership_rows.extend({"outer_fold": row["outer_fold"], "role": "train", "subject": value} for value in row["train_subjects"])
        fold_membership_rows.extend({"outer_fold": row["outer_fold"], "role": "test", "subject": value} for value in row["test_subjects"])
    pd.DataFrame(fold_membership_rows).to_csv(output / "tables/outer_fold_membership.csv", index=False, lineterminator="\n")
    pd.DataFrame({"subject": data.subjects, "median_without_subject": influence}).to_csv(
        output / "tables/openbmi_leave_one_subject_influence.csv", index=False, lineterminator="\n", float_format="%.17g"
    )
    control_rows = []
    for name, value in (
        ("sensor_primary", observed), ("magnitude", magnitude), ("ordered_Z_eigenvalues", spectrum),
        ("generalized_eigenvalues", generalized),
    ):
        control_rows.append({
            "control": name, "statistic": float(value["statistic"]),
            "forward_median": float(value["forward_median"]),
            "reverse_median": float(value["reverse_median"]),
            "full_space_statistic": float(value["full_space_statistic"]),
            "selected_ranks": "|".join(map(str, value["selected_ranks"])),
            "votes_primary": name == "sensor_primary",
        })
    control_rows.append({
        "control": "same_session_PCA", "statistic": float(np.median(pca["delta_average"])),
        "forward_median": float("nan"), "reverse_median": float("nan"),
        "full_space_statistic": float("nan"),
        "selected_ranks": "|".join(map(str, observed["selected_ranks"])), "votes_primary": False,
    })
    # CSV must contain no NaN/Inf, including non-applicable control fields.
    for row in control_rows:
        for key in ("forward_median", "reverse_median", "full_space_statistic"):
            if not np.isfinite(row[key]):
                row[key] = "NOT_APPLICABLE"
    pd.DataFrame(control_rows).to_csv(output / "tables/openbmi_controls.csv", index=False, lineterminator="\n", float_format="%.17g")
    pca.to_csv(output / "tables/openbmi_same_session_pca_subject_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
    split_half.to_csv(output / "tables/openbmi_selected_mode_split_half.csv", index=False, lineterminator="\n", float_format="%.17g")
    latent.to_csv(output / "tables/openbmi_latent_mode1_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
    _atomic_savez(output / "objects/openbmi_bootstrap.npz", {"statistics": bootstrap})
    _serialize_openbmi_observed(output, data, observed)
    result = {
        "schema_version": "subject-class-population-structure-openbmi-observed-v1",
        "statistic": float(observed["statistic"]),
        "forward_median": float(observed["forward_median"]),
        "reverse_median": float(observed["reverse_median"]),
        "bootstrap_ci_95": [low, high],
        "selected_ranks": observed["selected_ranks"].tolist(),
        "median_selected_rank": float(np.median(observed["selected_ranks"])),
        "folds_at_or_below_low_cap": int(np.count_nonzero(observed["selected_ranks"] <= 8)),
        "full_space_statistic": float(observed["full_space_statistic"]),
        "influence_sign_pass": bool(np.all(influence > 0.0)),
        "config_sha256": config_hash, "parent_hashes": parent_hashes,
        "controls": control_rows,
    }
    atomic_write_json(output / "decisions/openbmi_observed.json", result)
    return result


def _haar_basis(p: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(p, rank)), mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    basis = q * signs[None, :]
    np.testing.assert_allclose(basis.T @ basis, np.eye(rank), rtol=0.0, atol=2e-13)
    return basis


def random_subspace_once(
    data: DatasetObjects,
    config: Mapping[str, Any],
    selected_ranks: Sequence[int],
    replicate: int,
) -> dict[str, float]:
    outer, _ = openbmi_fold_indices(data, config)
    all_indices = np.arange(len(data.subjects))
    subject_values: list[float] = []
    forward_values: list[float] = []
    reverse_values: list[float] = []
    master = int(config["protocol"]["master_seed"])
    for fold_index, test in enumerate(outer):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        x_train, x_test, _ = fold_features(data, "F", train, test, helmert=None, kind="sensor")
        rank = int(selected_ranks[fold_index])
        rng0 = deterministic_rng(master, "random_subspace", replicate, fold_index, 0)
        rng1 = deterministic_rng(master, "random_subspace", replicate, fold_index, 1)
        left = _haar_basis(x_train.shape[-1], rank, rng0)
        right = _haar_basis(x_train.shape[-1], rank, rng1)
        mean0 = np.mean(x_train[:, 0], axis=0)
        mean1 = np.mean(x_train[:, 1], axis=0)
        train0 = (x_train[:, 0] - mean0) @ left
        train1 = (x_train[:, 1] - mean1) @ right
        scale0 = np.std(train0, axis=0, ddof=1)
        scale1 = np.std(train1, axis=0, ddof=1)
        if np.any(scale0 <= np.finfo(float).eps) or np.any(scale1 <= np.finfo(float).eps):
            raise NumericalContractError("random subspace scale degeneracy")
        test0 = (x_test[:, 0] - mean0) @ left / scale0
        test1 = (x_test[:, 1] - mean1) @ right / scale1
        score = separation_from_scores(test0, test1, rank)
        subject_values.extend(np.asarray(score["average"]).tolist())
        forward_values.extend(np.asarray(score["forward"]).tolist())
        reverse_values.extend(np.asarray(score["reverse"]).tolist())
    return {
        "statistic": float(np.median(subject_values)),
        "forward_median": float(np.median(forward_values)),
        "reverse_median": float(np.median(reverse_values)),
    }


_NULL_STATE: dict[str, Any] = {}


def _initialize_null_worker(
    data: DatasetObjects,
    config: Mapping[str, Any],
    pairing_outer: np.ndarray,
    pairing_inner: np.ndarray,
    class_mappings: np.ndarray,
    selected_ranks: np.ndarray,
) -> None:
    global _NULL_STATE
    _NULL_STATE = {
        "data": data, "config": config,
        "pairing_outer": pairing_outer, "pairing_inner": pairing_inner,
        "class_mappings": class_mappings, "selected_ranks": selected_ranks,
    }


def _null_worker(task: tuple[str, int]) -> tuple[int, np.ndarray, np.ndarray]:
    kind, replicate = task
    data: DatasetObjects = _NULL_STATE["data"]
    config: Mapping[str, Any] = _NULL_STATE["config"]
    if kind == "pairing":
        result = evaluate_openbmi_nested(
            data, config,
            outer_pairing_mappings=_NULL_STATE["pairing_outer"][replicate],
            inner_pairing_mappings=_NULL_STATE["pairing_inner"][replicate],
        )
        values = np.asarray([
            result["statistic"], result["forward_median"], result["reverse_median"],
            result["full_space_statistic"],
        ], dtype=np.float64)
        ranks = np.asarray(result["selected_ranks"], dtype=np.int16)
    elif kind == "class":
        result = evaluate_openbmi_nested(
            data, config, class_mapping=_NULL_STATE["class_mappings"][replicate]
        )
        values = np.asarray([
            result["statistic"], result["forward_median"], result["reverse_median"],
            result["full_space_statistic"],
        ], dtype=np.float64)
        ranks = np.asarray(result["selected_ranks"], dtype=np.int16)
    elif kind == "random":
        result = random_subspace_once(
            data, config, _NULL_STATE["selected_ranks"], replicate
        )
        values = np.asarray([
            result["statistic"], result["forward_median"], result["reverse_median"], 0.0,
        ], dtype=np.float64)
        ranks = np.asarray(_NULL_STATE["selected_ranks"], dtype=np.int16)
    else:
        raise ValueError(f"unknown null kind {kind}")
    if not np.isfinite(values).all():
        raise NumericalContractError(f"{kind} null {replicate} is nonfinite")
    return replicate, values, ranks


def _null_checkpoint(
    path: Path,
    *,
    kind: str,
    config_hash: str,
    observed_hash: str,
    statistics: np.ndarray,
    ranks: np.ndarray,
    completed: np.ndarray,
) -> None:
    metadata = np.asarray(json.dumps({
        "kind": kind, "config_sha256": config_hash, "observed_sha256": observed_hash,
        "replicates": len(completed),
    }, sort_keys=True, separators=(",", ":")))
    _atomic_savez(path, {
        "statistics": statistics, "selected_ranks": ranks,
        "completed": completed.astype(np.uint8), "metadata_json": metadata,
    })


def _load_or_create_null_checkpoint(
    path: Path,
    *,
    kind: str,
    config_hash: str,
    observed_hash: str,
    replicates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        return (
            np.zeros((replicates, 4), dtype=np.float64),
            np.zeros((replicates, 6), dtype=np.int16),
            np.zeros(replicates, dtype=bool),
        )
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        expected = {
            "kind": kind, "config_sha256": config_hash,
            "observed_sha256": observed_hash, "replicates": replicates,
        }
        if metadata != expected:
            raise DataContractError(f"{kind} null checkpoint identity mismatch")
        statistics = np.asarray(archive["statistics"], dtype=np.float64)
        ranks = np.asarray(archive["selected_ranks"], dtype=np.int16)
        completed = np.asarray(archive["completed"], dtype=np.uint8).astype(bool)
    if statistics.shape != (replicates, 4) or ranks.shape != (replicates, 6) or completed.shape != (replicates,):
        raise DataContractError(f"{kind} null checkpoint shape mismatch")
    if not np.isfinite(statistics[completed]).all() or np.any(ranks[completed] <= 0):
        raise DataContractError(f"{kind} completed checkpoint rows are invalid")
    return statistics, ranks, completed


def _execute_null_kind(
    kind: str,
    output: Path,
    *,
    config_hash: str,
    observed_hash: str,
    replicates: int,
    workers: int,
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    import multiprocessing as mp
    path = output / f"nulls/openbmi_{kind}_null.npz"
    statistics, ranks, completed = _load_or_create_null_checkpoint(
        path, kind=kind, config_hash=config_hash, observed_hash=observed_hash,
        replicates=replicates,
    )
    pending = np.flatnonzero(~completed).tolist()
    if not pending:
        return statistics, ranks
    tasks = [(kind, int(index)) for index in pending]
    if workers == 1:
        iterator: Iterable[tuple[int, np.ndarray, np.ndarray]] = map(_null_worker, tasks)
        pool = None
    else:
        context = mp.get_context("fork")
        pool = context.Pool(processes=workers)
        iterator = pool.imap(_null_worker, tasks, chunksize=1)
    try:
        since_save = 0
        for replicate, values, selected in iterator:
            statistics[replicate] = values
            ranks[replicate] = selected
            completed[replicate] = True
            since_save += 1
            if since_save >= batch:
                _null_checkpoint(
                    path, kind=kind, config_hash=config_hash, observed_hash=observed_hash,
                    statistics=statistics, ranks=ranks, completed=completed,
                )
                since_save = 0
        _null_checkpoint(
            path, kind=kind, config_hash=config_hash, observed_hash=observed_hash,
            statistics=statistics, ranks=ranks, completed=completed,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if not np.all(completed):
        raise NumericalContractError(f"{kind} null incomplete")
    return statistics, ranks


def run_openbmi_nulls(repo_root: str | Path, *, workers: int | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_real_access_lock(root)
    config, config_hash = load_config(root)
    parent_hashes = validate_parent_hashes(root, config)
    output = root / str(config["project"]["output_dir"])
    reliability = json.loads((output / "decisions/reliability_gate.json").read_text(encoding="utf-8"))
    if not reliability["passed"]:
        raise DataContractError("UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE")
    observed_path = output / "decisions/openbmi_observed.json"
    core_path = output / "objects/openbmi_observed_core.npz"
    if not observed_path.is_file() or not core_path.is_file():
        raise DataContractError("OpenBMI observed artifacts must precede nulls")
    observed = json.loads(observed_path.read_text(encoding="utf-8"))
    observed_hash = sha256_file(core_path)
    with np.load(core_path, allow_pickle=False) as archive:
        selected_ranks = np.asarray(archive["selected_ranks"], dtype=np.int64)
    data = load_parent_dataset(root, config, "openbmi")
    replicates = int(config["nulls"]["replicates"])
    pairing_outer, pairing_inner = generate_openbmi_pairing_mappings(data, config, replicates)
    class_mappings = generate_openbmi_class_mappings(data, config, replicates)
    _atomic_savez(output / "nulls/openbmi_pairing_mappings.npz", {
        "outer_mappings": pairing_outer, "inner_mappings": pairing_inner,
        "subjects": np.asarray(data.subjects),
    })
    _atomic_savez(output / "nulls/openbmi_class_mappings.npz", {
        "class_mappings": class_mappings, "subjects": np.asarray(data.subjects),
        "sessions": np.asarray(data.sessions), "classes": np.asarray(data.classes),
    })
    atomic_write_json(output / "nulls/seed_manifest.json", {
        "master_seed": int(config["protocol"]["master_seed"]),
        "bit_generator": "PCG64DXSM",
        "derivation": "canonical JSON -> SHA256 -> first 128 bits as little-endian uint32 words",
        "pairing_outer_sha256": sha256_array(pairing_outer),
        "pairing_inner_sha256": sha256_array(pairing_inner),
        "class_mappings_sha256": sha256_array(class_mappings),
        "random_mapping": "keyed by replicate/fold/view; bases not persisted",
    })
    global _NULL_STATE
    _initialize_null_worker(data, config, pairing_outer, pairing_inner, class_mappings, selected_ranks)
    worker_count = int(config["resources"]["workers"] if workers is None else workers)
    batch = int(config["resources"]["checkpoint_batch"])
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in ("pairing", "class", "random"):
        results[kind] = _execute_null_kind(
            kind, output, config_hash=config_hash, observed_hash=observed_hash,
            replicates=replicates, workers=worker_count, batch=batch,
        )
    pairing, pairing_ranks = results["pairing"]
    class_null, class_ranks = results["class"]
    random, _ = results["random"]
    pairing_p = monte_carlo_p(float(observed["statistic"]), pairing[:, 0])
    class_p = monte_carlo_p(float(observed["statistic"]), class_null[:, 0])
    random_p = monte_carlo_p(float(observed["statistic"]), random[:, 0])
    full_pairing_p = monte_carlo_p(float(observed["full_space_statistic"]), pairing[:, 3])
    full_space_stable = float(observed["full_space_statistic"]) > 0.0 and full_pairing_p <= 0.05
    decision = terminal_decision(
        data_contract_pass=True, reliability_pass=True,
        statistic=float(observed["statistic"]),
        forward_median=float(observed["forward_median"]),
        reverse_median=float(observed["reverse_median"]),
        pairing_p=pairing_p, class_p=class_p, random_p=random_p,
        influence_positive=bool(observed["influence_sign_pass"]),
        full_space_stable=full_space_stable,
        selected_ranks=observed["selected_ranks"], low_cap=8,
    )
    summary_rows = [
        {"null": "subject_pairing", "observed": observed["statistic"], "null_median": float(np.median(pairing[:, 0])), "p_value": pairing_p, "replicates": replicates, "reran_rank_selection": True},
        {"null": "class_semantics", "observed": observed["statistic"], "null_median": float(np.median(class_null[:, 0])), "p_value": class_p, "replicates": replicates, "reran_rank_selection": True},
        {"null": "equal_rank_random_subspace", "observed": observed["statistic"], "null_median": float(np.median(random[:, 0])), "p_value": random_p, "replicates": replicates, "reran_rank_selection": False},
        {"null": "full_space_subject_pairing", "observed": observed["full_space_statistic"], "null_median": float(np.median(pairing[:, 3])), "p_value": full_pairing_p, "replicates": replicates, "reran_rank_selection": True},
    ]
    pd.DataFrame(summary_rows).to_csv(output / "tables/openbmi_primary_null_summaries.csv", index=False, lineterminator="\n", float_format="%.17g")
    rank_rows = []
    for kind, values in (("pairing", pairing_ranks), ("class", class_ranks)):
        for fold in range(values.shape[1]):
            unique, counts = np.unique(values[:, fold], return_counts=True)
            rank_rows.extend({"null": kind, "outer_fold": fold, "rank": int(rank), "count": int(count)} for rank, count in zip(unique, counts))
    pd.DataFrame(rank_rows).to_csv(output / "tables/openbmi_null_rank_selection_frequency.csv", index=False, lineterminator="\n")
    result = {
        "schema_version": "subject-class-population-structure-openbmi-nulls-v1",
        "observed": float(observed["statistic"]),
        "pairing_p": pairing_p, "class_p": class_p, "random_subspace_p": random_p,
        "full_space_pairing_p": full_pairing_p, "full_space_stable": full_space_stable,
        "terminal_decision": decision, "selected_ranks": observed["selected_ranks"],
        "config_sha256": config_hash, "observed_core_sha256": observed_hash,
        "parent_hashes": parent_hashes,
    }
    atomic_write_json(output / "decisions/terminal_decision.json", result)
    return result


def single_heldout_separation(
    train0: np.ndarray,
    train1: np.ndarray,
    test0: np.ndarray,
    test1: np.ndarray,
    rank: int,
) -> dict[str, float]:
    same = float(np.dot(test0[0, :rank], test1[0, :rank]) / rank)
    forward_other = (test0[0, :rank] @ train1[:, :rank].T) / rank
    reverse_other = (train0[:, :rank] @ test1[0, :rank]) / rank
    forward = same - float(np.median(forward_other))
    reverse = same - float(np.median(reverse_other))
    return {"same": same, "forward": forward, "reverse": reverse, "average": 0.5 * (forward + reverse)}


def _bnci_fold_scores(
    x_train: np.ndarray,
    x_test: np.ndarray,
    ranks: Sequence[int],
) -> tuple[TwoViewFit, dict[int, dict[str, float]], tuple[np.ndarray, np.ndarray]]:
    fit = fit_two_view(x_train[:, 0], x_train[:, 1], max(ranks))
    train0, train1 = project_two_view(fit, x_train[:, 0], x_train[:, 1])
    test0, test1 = project_two_view(fit, x_test[:, 0], x_test[:, 1])
    by_rank = {
        int(rank): single_heldout_separation(train0, train1, test0, test1, int(rank))
        for rank in ranks
    }
    return fit, by_rank, (test0, test1)


def evaluate_bnci_nested(
    data: DatasetObjects,
    config: Mapping[str, Any],
    *,
    class_mapping: np.ndarray | None = None,
    pairing_mapping: np.ndarray | None = None,
    keep_fits: bool = False,
) -> dict[str, Any]:
    if data.name != "bnci":
        raise ValueError("BNCI evaluator received another dataset")
    helmert = validate_helmert(config)
    all_indices = np.arange(len(data.subjects), dtype=np.int64)
    subject_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    fits: list[TwoViewFit] = []
    coordinates: list[dict[str, float | int]] = []
    for target in all_indices:
        test = np.asarray([target])
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        ranks = _rank_grid(config, "bnci", len(train) - 1, _feature_dimension(data, "sensor"))
        inner_scores = np.empty((len(train), len(ranks)), dtype=np.float64)
        for inner_index, validation_subject in enumerate(train):
            validation = np.asarray([validation_subject])
            inner_train = np.setdiff1d(train, validation, assume_unique=True)
            x_inner, x_validation, _ = fold_features(
                data, "F", inner_train, validation, helmert=helmert, kind="sensor",
                class_mapping=class_mapping, session1_mapping=pairing_mapping,
            )
            _, by_rank, _ = _bnci_fold_scores(x_inner, x_validation, ranks)
            inner_scores[inner_index] = [by_rank[rank]["average"] for rank in ranks]
        selection = select_rank_one_se(inner_scores, ranks)
        x_train, x_test, _ = fold_features(
            data, "F", train, test, helmert=helmert, kind="sensor",
            class_mapping=class_mapping, session1_mapping=pairing_mapping,
        )
        fit, by_rank, test_coordinates = _bnci_fold_scores(x_train, x_test, ranks)
        selected_rank = int(selection["selected_rank"])
        selected = by_rank[selected_rank]
        full_similarity = float(x_test[0, 0] @ x_test[0, 1])
        full_forward = full_similarity - float(np.median(x_test[0, 0] @ x_train[:, 1].T))
        full_reverse = full_similarity - float(np.median(x_train[:, 0] @ x_test[0, 1]))
        full_average = 0.5 * (full_forward + full_reverse)
        subject_rows.append({
            "subject": int(data.subjects[target]), "outer_fold": int(target),
            "selected_rank": selected_rank, "delta_forward": selected["forward"],
            "delta_reverse": selected["reverse"], "delta_average": selected["average"],
        })
        full_rows.append({
            "subject": int(data.subjects[target]), "delta_forward": full_forward,
            "delta_reverse": full_reverse, "delta_average": full_average,
        })
        score0, score1 = test_coordinates
        coordinates.append({
            "subject": int(data.subjects[target]),
            "latent_coordinate_norm": float(0.5 * (np.linalg.norm(score0[0, :selected_rank]) + np.linalg.norm(score1[0, :selected_rank]))),
            "latent_session_discrepancy": float(np.linalg.norm(score0[0, :selected_rank] - score1[0, :selected_rank]) / np.sqrt(selected_rank)),
        })
        for rank in ranks:
            rank_rows.append({
                "subject": int(data.subjects[target]), "rank": rank,
                "delta_forward": by_rank[rank]["forward"], "delta_reverse": by_rank[rank]["reverse"],
                "delta_average": by_rank[rank]["average"],
            })
        fold_rows.append({
            "outer_fold": int(target), "test_subject": int(data.subjects[target]),
            "selected_rank": selected_rank, "best_rank": int(selection["best_rank"]),
            "one_se_threshold": float(selection["threshold"]),
        })
        if keep_fits:
            fits.append(fit)
    frame = pd.DataFrame(subject_rows).sort_values("subject").reset_index(drop=True)
    full_frame = pd.DataFrame(full_rows).sort_values("subject").reset_index(drop=True)
    return {
        "statistic": float(np.median(frame["delta_average"])),
        "forward_median": float(np.median(frame["delta_forward"])),
        "reverse_median": float(np.median(frame["delta_reverse"])),
        "subject_rows": frame, "full_subject_rows": full_frame,
        "full_space_statistic": float(np.median(full_frame["delta_average"])),
        "rank_rows": pd.DataFrame(rank_rows).sort_values(["rank", "subject"]).reset_index(drop=True),
        "fold_rows": pd.DataFrame(fold_rows),
        "selected_ranks": frame["selected_rank"].to_numpy(dtype=np.int64),
        "coordinates": pd.DataFrame(coordinates).sort_values("subject").reset_index(drop=True),
        "fits": fits,
    }


def generate_bnci_pairing_mappings(data: DatasetObjects, config: Mapping[str, Any], replicates: int) -> np.ndarray:
    indices = np.arange(len(data.subjects), dtype=np.int64)
    master = int(config["protocol"]["master_seed"])
    result = np.empty((replicates, len(indices)), dtype=np.int8)
    for replicate in range(replicates):
        result[replicate] = fixed_point_free(
            indices, deterministic_rng(master, "bnci_pairing", replicate)
        )
    return result


def generate_bnci_class_mappings(data: DatasetObjects, config: Mapping[str, Any], replicates: int) -> np.ndarray:
    master = int(config["protocol"]["master_seed"])
    k = len(data.classes)
    result = np.empty((replicates, len(data.subjects), 2, k), dtype=np.int8)
    identity = np.arange(k, dtype=np.int8)
    for replicate in range(replicates):
        rng = deterministic_rng(master, "bnci_class", replicate)
        for subject in range(len(data.subjects)):
            for session in range(2):
                result[replicate, subject, session] = rng.permutation(identity)
        if np.all(result[replicate] == identity):
            result[replicate, 0, 0] = np.roll(identity, 1)
    return result


def bnci_mode_loading_rows(
    data: DatasetObjects,
    config: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    helmert = validate_helmert(config)
    m = len(data.channels) * (len(data.channels) + 1) // 2
    class_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for fold, (fit, rank) in enumerate(zip(observed["fits"], observed["selected_ranks"])):
        for view_name, basis in (("session0", fit.left), ("session1", fit.right)):
            contrasts = basis[:, :rank].reshape(3, m, rank)
            class_basis = np.einsum("ka,amr->kmr", helmert.T, contrasts, optimize=True)
            for mode in range(rank):
                for class_index, class_name in enumerate(data.classes):
                    class_rows.append({
                        "outer_fold": fold, "view": view_name, "mode": mode + 1,
                        "class": class_name,
                        "loading_energy": float(np.linalg.norm(class_basis[class_index, :, mode]) ** 2),
                    })
                for first in range(len(data.classes)):
                    for second in range(first + 1, len(data.classes)):
                        pair_rows.append({
                            "outer_fold": fold, "view": view_name, "mode": mode + 1,
                            "class_pair": f"{data.classes[first]}|{data.classes[second]}",
                            "loading_energy": float(np.linalg.norm(class_basis[first, :, mode] - class_basis[second, :, mode]) ** 2),
                        })
    return pd.DataFrame(class_rows), pd.DataFrame(pair_rows)


_BNCI_STATE: dict[str, Any] = {}


def _initialize_bnci_worker(data: DatasetObjects, config: Mapping[str, Any], pairing: np.ndarray, classes: np.ndarray) -> None:
    global _BNCI_STATE
    _BNCI_STATE = {"data": data, "config": config, "pairing": pairing, "classes": classes}


def _bnci_null_worker(task: tuple[str, int]) -> tuple[int, np.ndarray, np.ndarray]:
    kind, replicate = task
    if kind == "pairing":
        result = evaluate_bnci_nested(
            _BNCI_STATE["data"], _BNCI_STATE["config"],
            pairing_mapping=_BNCI_STATE["pairing"][replicate],
        )
    elif kind == "class":
        result = evaluate_bnci_nested(
            _BNCI_STATE["data"], _BNCI_STATE["config"],
            class_mapping=_BNCI_STATE["classes"][replicate],
        )
    else:
        raise ValueError(kind)
    return replicate, np.asarray([
        result["statistic"], result["forward_median"], result["reverse_median"],
        result["full_space_statistic"],
    ]), np.asarray(result["selected_ranks"], dtype=np.int8)


def run_bnci_diagnostic(repo_root: str | Path, *, workers: int | None = None) -> dict[str, Any]:
    import multiprocessing as mp
    root = Path(repo_root).resolve()
    ensure_real_access_lock(root)
    config, config_hash = load_config(root)
    parent_hashes = validate_parent_hashes(root, config)
    output = root / str(config["project"]["output_dir"])
    if not (output / "decisions/terminal_decision.json").is_file():
        raise DataContractError("OpenBMI terminal must precede BNCI diagnostic")
    data = load_parent_dataset(root, config, "bnci")
    observed = evaluate_bnci_nested(data, config, keep_fits=True)
    replicates = int(config["nulls"]["replicates"])
    pairing = generate_bnci_pairing_mappings(data, config, replicates)
    classes = generate_bnci_class_mappings(data, config, replicates)
    _atomic_savez(output / "nulls/bnci_mappings.npz", {
        "pairing": pairing, "classes": classes, "subjects": np.asarray(data.subjects),
    })
    _initialize_bnci_worker(data, config, pairing, classes)
    worker_count = int(config["resources"]["workers"] if workers is None else workers)
    nulls: dict[str, np.ndarray] = {}
    null_ranks: dict[str, np.ndarray] = {}
    for kind in ("pairing", "class"):
        statistics = np.zeros((replicates, 4), dtype=np.float64)
        ranks = np.zeros((replicates, len(data.subjects)), dtype=np.int8)
        tasks = [(kind, index) for index in range(replicates)]
        if worker_count == 1:
            iterator = map(_bnci_null_worker, tasks)
            pool = None
        else:
            pool = mp.get_context("fork").Pool(worker_count)
            iterator = pool.imap(_bnci_null_worker, tasks, chunksize=1)
        try:
            for replicate, values, selected in iterator:
                statistics[replicate] = values
                ranks[replicate] = selected
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        nulls[kind] = statistics
        null_ranks[kind] = ranks
        _atomic_savez(output / f"nulls/bnci_{kind}_null.npz", {"statistics": statistics, "selected_ranks": ranks})
    pairing_p = monte_carlo_p(observed["statistic"], nulls["pairing"][:, 0])
    class_p = monte_carlo_p(observed["statistic"], nulls["class"][:, 0])
    influence = leave_one_subject_influence(observed["subject_rows"]["delta_average"])
    class_loadings, pair_loadings = bnci_mode_loading_rows(data, config, observed)
    action = pd.read_csv(root / str(config["parent_artifacts"]["action_stage_b"]["path"]))
    action = action.rename(columns={"target": "subject", "subject_median_gain": "action_stage_b_gain"})
    association = observed["coordinates"].merge(action, on="subject", validate="one_to_one")
    norm_rho, norm_p = stats.spearmanr(association["latent_coordinate_norm"], association["action_stage_b_gain"])
    discrepancy_rho, discrepancy_p = stats.spearmanr(association["latent_session_discrepancy"], association["action_stage_b_gain"])
    association["action_successful"] = association["action_stage_b_gain"] > 0.0
    association_rows = [
        {"association": "latent_coordinate_norm_vs_stage_b_gain", "spearman_rho": float(norm_rho), "p_value_two_sided_descriptive": float(norm_p)},
        {"association": "latent_session_discrepancy_vs_stage_b_gain", "spearman_rho": float(discrepancy_rho), "p_value_two_sided_descriptive": float(discrepancy_p)},
    ]
    observed["subject_rows"].to_csv(output / "tables/bnci_subject_directional_separation.csv", index=False, lineterminator="\n", float_format="%.17g")
    observed["rank_rows"].to_csv(output / "tables/bnci_rank_by_rank_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
    observed["fold_rows"].to_csv(output / "tables/bnci_selected_ranks.csv", index=False, lineterminator="\n", float_format="%.17g")
    class_loadings.to_csv(output / "tables/bnci_class_mode_loadings.csv", index=False, lineterminator="\n", float_format="%.17g")
    pair_loadings.to_csv(output / "tables/bnci_class_pair_mode_loadings.csv", index=False, lineterminator="\n", float_format="%.17g")
    association.to_csv(output / "tables/bnci_action_score_overlap.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(association_rows).to_csv(output / "tables/bnci_action_overlap_associations.csv", index=False, lineterminator="\n", float_format="%.17g")
    result = {
        "schema_version": "subject-class-population-structure-bnci-diagnostic-v1",
        "statistic": observed["statistic"], "forward_median": observed["forward_median"],
        "reverse_median": observed["reverse_median"], "selected_ranks": observed["selected_ranks"].tolist(),
        "full_space_statistic": observed["full_space_statistic"],
        "pairing_p": pairing_p, "class_p": class_p,
        "influence_sign_pass": bool(np.all(influence > 0.0)),
        "action_overlap_status": "UNASSESSED_ACTION_OVERLAP",
        "action_associations": association_rows,
        "action_successful_coordinate_norm_median": float(association.loc[association["action_successful"], "latent_coordinate_norm"].median()),
        "action_unsuccessful_coordinate_norm_median": float(association.loc[~association["action_successful"], "latent_coordinate_norm"].median()),
        "interpretation": "SECONDARY_MULTI_CLASS_DIAGNOSTIC_ONLY",
        "config_sha256": config_hash, "parent_hashes": parent_hashes,
    }
    atomic_write_json(output / "decisions/bnci_diagnostic.json", result)
    return result


def _synthetic_feature_nested(
    x0: np.ndarray,
    x1: np.ndarray,
    ranks: Sequence[int],
    *,
    master_seed: int,
) -> dict[str, Any]:
    n = len(x0)
    rng = deterministic_rng(master_seed, "synthetic_folds")
    outer_folds = [np.sort(fold) for fold in np.array_split(rng.permutation(n), 6)]
    all_indices = np.arange(n)
    rows = []
    selected = []
    fits = []
    for outer_index, test in enumerate(outer_folds):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        inner_rng = deterministic_rng(master_seed, "synthetic_inner", outer_index)
        inner_folds = [np.sort(fold) for fold in np.array_split(inner_rng.permutation(train), 5)]
        permitted_ranks = [int(rank) for rank in ranks if rank <= min(x0.shape[1], len(train) - len(inner_folds[0]) - 2)]
        fold_scores = np.empty((5, len(permitted_ranks)))
        for inner_index, validation in enumerate(inner_folds):
            inner_train = np.setdiff1d(train, validation, assume_unique=True)
            fit = fit_two_view(x0[inner_train], x1[inner_train], max(permitted_ranks))
            score0, score1 = project_two_view(fit, x0[validation], x1[validation])
            fold_scores[inner_index] = [separation_from_scores(score0, score1, rank)["statistic"] for rank in permitted_ranks]
        choice = select_rank_one_se(fold_scores, permitted_ranks)
        selected_rank = int(choice["selected_rank"])
        fit = fit_two_view(x0[train], x1[train], max(permitted_ranks))
        score0, score1 = project_two_view(fit, x0[test], x1[test])
        score = separation_from_scores(score0, score1, selected_rank)
        selected.append(selected_rank)
        fits.append(fit)
        rows.extend(np.asarray(score["average"]).tolist())
    return {
        "statistic": float(np.median(rows)), "subject_values": np.asarray(rows),
        "selected_ranks": np.asarray(selected), "fits": fits,
        "outer_folds": outer_folds,
    }


def _synthetic_pairing_null(
    x0: np.ndarray,
    x1: np.ndarray,
    ranks: Sequence[int],
    *,
    master_seed: int,
    replicates: int,
) -> np.ndarray:
    result = np.empty(replicates)
    indices = np.arange(len(x0))
    for replicate in range(replicates):
        permuted = fixed_point_free(indices, deterministic_rng(master_seed, "synthetic_pairing", replicate))
        result[replicate] = _synthetic_feature_nested(
            x0, x1[permuted], ranks, master_seed=master_seed
        )["statistic"]
    return result


def _synthetic_random_control(
    x0: np.ndarray,
    x1: np.ndarray,
    observed: Mapping[str, Any],
    *,
    master_seed: int,
    replicates: int,
) -> np.ndarray:
    values = np.empty(replicates)
    all_indices = np.arange(len(x0))
    for replicate in range(replicates):
        subject_scores: list[float] = []
        for fold_index, test in enumerate(observed["outer_folds"]):
            train = np.setdiff1d(all_indices, test, assume_unique=True)
            rank = int(observed["selected_ranks"][fold_index])
            left = _haar_basis(x0.shape[1], rank, deterministic_rng(master_seed, "synthetic_random", replicate, fold_index, 0))
            right = _haar_basis(x1.shape[1], rank, deterministic_rng(master_seed, "synthetic_random", replicate, fold_index, 1))
            means = (np.mean(x0[train], axis=0), np.mean(x1[train], axis=0))
            train0 = (x0[train] - means[0]) @ left
            train1 = (x1[train] - means[1]) @ right
            scale0 = np.std(train0, axis=0, ddof=1)
            scale1 = np.std(train1, axis=0, ddof=1)
            test0 = (x0[test] - means[0]) @ left / scale0
            test1 = (x1[test] - means[1]) @ right / scale1
            subject_scores.extend(separation_from_scores(test0, test1, rank)["average"])
        values[replicate] = np.median(subject_scores)
    return values


def run_synthetic_gates(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root)
    master = int(config["protocol"]["master_seed"])
    replicates = 99
    rows: list[dict[str, Any]] = []

    # Known paired rank 3.
    rng = deterministic_rng(master, "synthetic_known_low_rank")
    n, p, true_rank = 72, 30, 3
    left = _haar_basis(p, true_rank, rng)
    right = _haar_basis(p, true_rank, rng)
    latent = rng.normal(size=(n, true_rank))
    x0 = latent @ left.T + rng.normal(scale=0.12, size=(n, p))
    x1 = latent @ right.T + rng.normal(scale=0.12, size=(n, p))
    known = _synthetic_feature_nested(x0, x1, [1, 2, 3, 5, 8], master_seed=master + 1)
    pairing = _synthetic_pairing_null(x0, x1, [1, 2, 3, 5, 8], master_seed=master + 1, replicates=replicates)
    learned_left = known["fits"][0].left[:, :true_rank]
    learned_right = known["fits"][0].right[:, :true_rank]
    left_angle = float(np.max(linalg.subspace_angles(left, learned_left)))
    right_angle = float(np.max(linalg.subspace_angles(right, learned_right)))
    known_pass = (
        known["statistic"] > 0.0 and monte_carlo_p(known["statistic"], pairing) <= 0.05
        and float(np.median(known["selected_ranks"])) <= 5
        and left_angle < 0.35 and right_angle < 0.35
    )
    rows.append({
        "synthetic": "known_paired_low_rank", "observed": known["statistic"],
        "null_median": float(np.median(pairing)), "p_value": monte_carlo_p(known["statistic"], pairing),
        "median_selected_rank": float(np.median(known["selected_ranks"])),
        "left_max_principal_angle_rad": left_angle, "right_max_principal_angle_rad": right_angle,
        "expected_terminal": "GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION",
        "passed": bool(known_pass),
    })

    # Stable high effective rank; rank grid forces the predeclared synthetic low cap of 3.
    rng = deterministic_rng(master, "synthetic_full_rank")
    n, p, true_rank = 90, 18, 12
    left_full = _haar_basis(p, true_rank, rng)
    right_full = _haar_basis(p, true_rank, rng)
    latent = rng.normal(size=(n, true_rank))
    x0_full = latent @ left_full.T + rng.normal(scale=0.35, size=(n, p))
    x1_full = latent @ right_full.T + rng.normal(scale=0.35, size=(n, p))
    full = _synthetic_feature_nested(x0_full, x1_full, [1, 2, 3, 5, 8, 12], master_seed=master + 2)
    full_terminal = terminal_decision(
        data_contract_pass=True, reliability_pass=True, statistic=full["statistic"],
        forward_median=full["statistic"], reverse_median=full["statistic"],
        pairing_p=0.005, class_p=0.005, random_p=0.005, influence_positive=True,
        full_space_stable=True, selected_ranks=full["selected_ranks"], low_cap=3,
    )
    rows.append({
        "synthetic": "full_rank_stable", "observed": full["statistic"],
        "null_median": 0.0, "p_value": 0.005,
        "median_selected_rank": float(np.median(full["selected_ranks"])),
        "left_max_principal_angle_rad": 0.0, "right_max_principal_angle_rad": 0.0,
        "expected_terminal": "GO_STRUCTURED_BUT_NOT_LOW_DIMENSIONAL",
        "passed": full_terminal == "GO_STRUCTURED_BUT_NOT_LOW_DIMENSIONAL",
    })

    # Independent sessions.
    rng = deterministic_rng(master, "synthetic_no_pairing")
    x0_none = rng.normal(size=(72, 24))
    x1_none = rng.normal(size=(72, 24))
    none = _synthetic_feature_nested(x0_none, x1_none, [1, 2, 3, 5, 8], master_seed=master + 3)
    none_null = _synthetic_pairing_null(x0_none, x1_none, [1, 2, 3, 5, 8], master_seed=master + 3, replicates=replicates)
    none_p = monte_carlo_p(none["statistic"], none_null)
    none_terminal = terminal_decision(
        data_contract_pass=True, reliability_pass=True, statistic=none["statistic"],
        forward_median=none["statistic"], reverse_median=none["statistic"],
        pairing_p=none_p, class_p=1.0, random_p=1.0, influence_positive=True,
        full_space_stable=True, selected_ranks=none["selected_ranks"], low_cap=8,
    )
    rows.append({
        "synthetic": "no_cross_session_pairing", "observed": none["statistic"],
        "null_median": float(np.median(none_null)), "p_value": none_p,
        "median_selected_rank": float(np.median(none["selected_ranks"])),
        "left_max_principal_angle_rad": 0.0, "right_max_principal_angle_rad": 0.0,
        "expected_terminal": "STOP_NO_HELDOUT_POPULATION_STRUCTURE",
        "passed": none_terminal == "STOP_NO_HELDOUT_POPULATION_STRUCTURE",
    })

    # Learned dimensionality reduction must not beat an equal-rank random basis without signal.
    random_control = _synthetic_random_control(
        x0_none, x1_none, none, master_seed=master + 4, replicates=replicates
    )
    random_p = monte_carlo_p(none["statistic"], random_control)
    rows.append({
        "synthetic": "random_subspace_equivalence", "observed": none["statistic"],
        "null_median": float(np.median(random_control)), "p_value": random_p,
        "median_selected_rank": float(np.median(none["selected_ranks"])),
        "left_max_principal_angle_rad": 0.0, "right_max_principal_angle_rad": 0.0,
        "expected_terminal": "STOP_RANDOM_SUBSPACE_EQUIVALENT",
        "passed": random_p > 0.05,
    })

    # Split halves are independently generated; reliability is deliberately failed.
    rng = deterministic_rng(master, "synthetic_unreliable")
    half_a = rng.normal(size=(60, 20))
    half_b = rng.normal(size=(60, 20))
    reliability_score = full_space_separation(
        half_a / np.linalg.norm(half_a, axis=1, keepdims=True),
        half_b / np.linalg.norm(half_b, axis=1, keepdims=True),
    )["statistic"]
    unreliable_terminal = terminal_decision(
        data_contract_pass=True, reliability_pass=False, statistic=1.0,
        forward_median=1.0, reverse_median=1.0, pairing_p=0.005, class_p=0.005,
        random_p=0.005, influence_positive=True, full_space_stable=True,
        selected_ranks=[1] * 6, low_cap=8,
    )
    rows.append({
        "synthetic": "measurement_unreliability", "observed": reliability_score,
        "null_median": 0.0, "p_value": 1.0,
        "median_selected_rank": 0.0,
        "left_max_principal_angle_rad": 0.0, "right_max_principal_angle_rad": 0.0,
        "expected_terminal": "UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE",
        "passed": unreliable_terminal == "UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE",
    })
    frame = pd.DataFrame(rows)
    passed = bool(frame["passed"].all())
    output = root / str(config["project"]["output_dir"])
    (output / "protocol").mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "protocol/synthetic_gates.csv", index=False, lineterminator="\n", float_format="%.17g")
    result = {
        "schema_version": "subject-class-population-structure-synthetic-v1",
        "passed": passed, "gates": rows, "config_sha256": config_hash,
        "replicates_per_null": replicates,
    }
    atomic_write_json(output / "protocol/synthetic_gates.json", result)
    if not passed:
        failed = frame.loc[~frame["passed"], "synthetic"].tolist()
        raise NumericalContractError(f"synthetic gates failed: {failed}")
    return result


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _scientific_file_hashes(root: Path) -> dict[str, str]:
    candidates = [
        CONFIG_PATH,
        "docs/PROTOCOL_SUBJECT_CLASS_POPULATION_STRUCTURE_V1.md",
        "docs/AUDIT_SUBJECT_CLASS_POPULATION_STRUCTURE_V1.md",
        "src/subject_class_population_structure_v1.py",
        "tests/test_subject_class_population_structure_v1.py",
    ]
    candidates.extend(
        f"scripts/{number}_{name}.py" for number, name in (
            (60, "freeze_subject_class_structure_v1"),
            (61, "run_reliability_gate_v1"),
            (62, "run_openbmi_structure_v1"),
            (63, "run_openbmi_nulls_v1"),
            (64, "run_bnci_multiclass_diagnostic_v1"),
            (65, "report_subject_class_structure_v1"),
        )
    )
    missing = [path for path in candidates if not (root / path).is_file()]
    if missing:
        raise DataContractError(f"missing freeze-scope files: {missing}")
    return {path: sha256_file(root / path) for path in candidates}


def environment_record() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "PyYAML", "pytest"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": packages,
    }


def freeze_protocol(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root)
    parent_hashes = validate_parent_hashes(root, config)
    openbmi = load_parent_dataset(root, config, "openbmi")
    load_parent_dataset(root, config, "bnci")
    validate_helmert(config)
    openbmi_fold_indices(openbmi, config)
    synthetic = run_synthetic_gates(root)
    output = root / str(config["project"]["output_dir"])
    protocol_dir = output / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    for source_name in (
        "docs/PROTOCOL_SUBJECT_CLASS_POPULATION_STRUCTURE_V1.md",
        "docs/AUDIT_SUBJECT_CLASS_POPULATION_STRUCTURE_V1.md",
        CONFIG_PATH,
    ):
        source = root / source_name
        atomic_write_bytes(protocol_dir / source.name, source.read_bytes())
    files = _scientific_file_hashes(root)
    manifest = {
        "schema_version": "subject-class-population-structure-freeze-v1",
        "status": "READY_FOR_PROTOCOL_FREEZE_COMMIT",
        "base_commit": str(config["protocol"]["base_commit"]),
        "branch": str(config["protocol"]["branch"]),
        "config_sha256": config_hash,
        "protocol_sha256": str(config["protocol"]["protocol_sha256"]),
        "parent_hashes": parent_hashes,
        "scientific_file_hashes": files,
        "synthetic_pass": bool(synthetic["passed"]),
        "real_data_svd_accessed": False,
        "required_commit_subject": "freeze subject class population structure v1",
    }
    atomic_write_json(protocol_dir / "manifest.json", manifest)
    atomic_write_json(output / "environment.json", environment_record())
    return manifest


def ensure_real_access_lock(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root)
    output = root / str(config["project"]["output_dir"])
    manifest_path = output / "protocol/manifest.json"
    if not manifest_path.is_file():
        raise DataContractError("protocol freeze manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != config_hash or not manifest["synthetic_pass"]:
        raise DataContractError("protocol freeze identity/synthetic gate mismatch")
    current_files = _scientific_file_hashes(root)
    if current_files != manifest["scientific_file_hashes"]:
        raise DataContractError("scientific source/config changed after freeze")
    validate_parent_hashes(root, config)
    provenance_path = output / "git_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not _git(root, "merge-base", "--is-ancestor", provenance["protocol_freeze_commit"], "HEAD") == "":
            # merge-base --is-ancestor writes no stdout on success.
            raise DataContractError("unexpected ancestry command output")
        return provenance
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise DataContractError(f"first real access requires clean tree; observed: {status.splitlines()[:5]}")
    head = _git(root, "rev-parse", "HEAD")
    subject = _git(root, "show", "-s", "--format=%s", "HEAD")
    if subject != manifest["required_commit_subject"]:
        raise DataContractError(f"HEAD is not protocol freeze commit: {subject}")
    provenance = {
        "schema_version": "subject-class-population-structure-git-provenance-v1",
        "base_commit": str(config["protocol"]["base_commit"]),
        "protocol_freeze_commit": head,
        "branch": _git(root, "branch", "--show-current"),
        "first_real_access_tree_was_clean": True,
        "scientific_file_hashes": current_files,
    }
    atomic_write_json(provenance_path, provenance)
    return provenance


def validate_no_nonfinite_outputs(output: str | Path) -> None:
    root = Path(output)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            numeric = frame.select_dtypes(include=[np.number])
            if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
                raise DataContractError(f"nonfinite CSV output: {path}")
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    value = archive[key]
                    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                        raise DataContractError(f"nonfinite NPZ output: {path}:{key}")


def validate_report_consistency(repo_root: str | Path) -> None:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    output = root / str(config["project"]["output_dir"])
    decision = json.loads((output / "decisions/terminal_decision.json").read_text(encoding="utf-8"))
    report = (output / "report/subject_class_population_structure_v1.md").read_text(encoding="utf-8")
    if decision["terminal_decision"] not in report:
        raise DataContractError("report terminal does not match JSON")
    if decision["terminal_decision"] == "UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE":
        failure_path = output / "decisions/execution_failure.json"
        if not failure_path.is_file():
            raise DataContractError("numerical-failure report lacks execution record")
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if failure.get("partial_primary_result_interpreted") is not False:
            raise DataContractError("numerical-failure report interpreted a partial primary result")
        if (output / "decisions/openbmi_observed.json").exists():
            raise DataContractError("numerical-failure output unexpectedly contains an observed result")
        validate_no_nonfinite_outputs(output)
        return
    observed = json.loads((output / "decisions/openbmi_observed.json").read_text(encoding="utf-8"))
    if f"{observed['statistic']:.6f}" not in report:
        raise DataContractError("report primary statistic does not match observed JSON")
    validate_no_nonfinite_outputs(output)


def _save_figure(fig: Any, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")


def _run_reliability_failure_report(
    root: Path,
    config: Mapping[str, Any],
    config_hash: str,
    parent_hashes: Mapping[str, str],
    reliability: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the permitted fail-closed report without accessing rank results."""
    output = root / str(config["project"]["output_dir"])
    terminal_path = output / "decisions/terminal_decision.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    decision = terminal["terminal_decision"]
    if decision != "UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE":
        raise DataContractError("unexpected reliability-failure terminal")
    tables = output / "tables"
    figures = output / "figures"
    report_dir = output / "report"
    figures.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "dataset": "OpenBMI/Lee2019-MI", "role": config["datasets"]["openbmi"]["role"],
            "subjects": 54, "sessions": 2, "classes": 2, "channels": 20,
            "feature_dimension": 210, "object_path": config["parent_artifacts"]["openbmi_objects"]["path"],
            "object_sha256": parent_hashes["openbmi_objects"], "object_gate": "PASS",
        },
        {
            "dataset": "BNCI2014_001", "role": config["datasets"]["bnci"]["role"],
            "subjects": 9, "sessions": 2, "classes": 4, "channels": 22,
            "feature_dimension": 759, "object_path": config["parent_artifacts"]["bnci_objects"]["path"],
            "object_sha256": parent_hashes["bnci_objects"], "object_gate": "PASS_NOT_ACCESSED_AFTER_GATE",
        },
    ]).to_csv(tables / "dataset_object_contract.csv", index=False, lineterminator="\n")
    gate_rows = [
        {"gate": "parent/data contract", "observed": "PASS", "criterion": "PASS", "passed": True},
        {"gate": "measurement reliability", "observed": "FAIL", "criterion": "both sessions p<=0.05; positive; influence", "passed": False},
        {"gate": "population rank and primary nulls", "observed": "NOT_ACCESSED", "criterion": "reliability prerequisite PASS", "passed": False},
    ]
    pd.DataFrame(gate_rows).to_csv(tables / "terminal_gate_table.csv", index=False, lineterminator="\n")

    import matplotlib.pyplot as plt
    frame = pd.read_csv(tables / "reliability_subject_scores.csv")
    frame.to_csv(figures / "figure_7_split_half_reliability.csv", index=False, lineterminator="\n", float_format="%.17g")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for index, session in enumerate(config["datasets"]["openbmi"]["sessions"]):
        values = frame.loc[frame["session"].astype(str) == str(session), "delta_average"].to_numpy()
        ax.scatter(np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)), values, s=12, alpha=0.7)
        ax.plot(index, np.median(values), marker="_", markersize=18, color="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1], ["session 0", "session 1"])
    ax.set(ylabel="A/B same-subject separation", title="Outer-fold-safe measurement reliability")
    _save_figure(fig, figures, "figure_7_split_half_reliability")
    plt.close(fig)

    session_text = "; ".join(
        f"session {row['session']}: T={row['observed']:.6f}, p={row['p_value']:.6f}, influence={row['leave_one_subject_sign_pass']}"
        for row in reliability["sessions"]
    )
    report = f"""# Cross-Session Population Structure of Subject-Class Interaction V1

## Outcome

The frozen terminal is **{decision}**. The outer-fold-safe split-half prerequisite did not pass ({session_text}). Under the frozen protocol, no real population-rank SVD, effective-rank estimate, primary pairing/class/random-subspace statistic, mode visualization, or BNCI rescue analysis was accessed.

## Established by parent work

PR #2 established a stable montage-registered mean-level subject×class interaction, OpenBMI cross-session replication, and the spectrum-only limitation (`GO_SENSOR_SPACE_ONLY`). PR #4 supported the pairwise common-action leave-one-class-out necessary consequence, but not a globally identifiable or cycle-consistent subject action.

## Tested here

Immutable parent `U` and split-half objects passed their schema, hash, ordering, finite, symmetry, and SPD-metadata contracts. This work then tested the mandatory held-out-subject-safe split-half reliability gate. The failure prevents interpretation of predictive effective rank or population-shared modes; it does not negate the parent evidence that the interaction object itself is stable under its original analysis.

## Not established

No claim is made about held-out population structure, effective rank, full conditional distributions, dispersion, physiology, source anatomy, causality, ASD or clinical diagnosis, target-unlabeled inference, TTA recoverability, nonlinear/universal individuality manifolds, globally identifiable `Q_s`, or cross-dataset equality of modes. No classifier, network, adapter, loss, or TTA method is proposed.

## Prospective and clinical boundaries

Stieger2021 remains only a documented future prospective-confirmation candidate; no raw data or statistic was accessed. The ASD extension boundary is separate, and no ASD repository or data was modified.

## Next question

What additional supervision or physiological anchor is required when stable individual interaction does not admit a transferable population-shared low-rank representation?
"""
    report_path = report_dir / "subject_class_population_structure_v1.md"
    atomic_write_bytes(report_path, report.encode("utf-8"))
    if decision not in report_path.read_text(encoding="utf-8"):
        raise DataContractError("failure report terminal does not match JSON")
    validate_no_nonfinite_outputs(output)
    artifact_hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": "subject-class-population-structure-final-manifest-v1",
        "terminal_decision": decision, "config_sha256": config_hash,
        "parent_hashes": dict(parent_hashes), "artifact_hashes": artifact_hashes,
        "report_sha256": sha256_file(report_path), "downstream_real_access": False,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {
        "terminal_decision": decision, "report": str(report_path.relative_to(root)),
        "manifest": str((output / "manifest.json").relative_to(root)),
        "artifact_count": len(artifact_hashes), "report_sha256": manifest["report_sha256"],
    }


def run_final_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_real_access_lock(root)
    config, config_hash = load_config(root)
    parent_hashes = validate_parent_hashes(root, config)
    output = root / str(config["project"]["output_dir"])
    reliability_path = output / "decisions/reliability_gate.json"
    if not reliability_path.is_file():
        raise DataContractError("reliability result missing")
    reliability = json.loads(reliability_path.read_text(encoding="utf-8"))
    if not reliability["passed"]:
        if not (output / "decisions/terminal_decision.json").is_file():
            raise DataContractError("reliability-failure terminal missing")
        return _run_reliability_failure_report(root, config, config_hash, parent_hashes, reliability)
    required = [
        reliability_path,
        output / "decisions/openbmi_observed.json",
        output / "decisions/terminal_decision.json",
        output / "decisions/bnci_diagnostic.json",
    ]
    if any(not path.is_file() for path in required):
        raise DataContractError(f"report prerequisites missing: {[str(p) for p in required if not p.is_file()]}")
    reliability = json.loads(required[0].read_text(encoding="utf-8"))
    observed = json.loads(required[1].read_text(encoding="utf-8"))
    terminal = json.loads(required[2].read_text(encoding="utf-8"))
    bnci = json.loads(required[3].read_text(encoding="utf-8"))
    figures = output / "figures"
    tables = output / "tables"
    report_dir = output / "report"
    figures.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    contract_rows = [
        {
            "dataset": "OpenBMI/Lee2019-MI", "role": config["datasets"]["openbmi"]["role"],
            "subjects": 54, "sessions": 2, "classes": 2, "channels": 20,
            "feature_dimension": 210, "object_path": config["parent_artifacts"]["openbmi_objects"]["path"],
            "object_sha256": parent_hashes["openbmi_objects"], "object_gate": "PASS",
        },
        {
            "dataset": "BNCI2014_001", "role": config["datasets"]["bnci"]["role"],
            "subjects": 9, "sessions": 2, "classes": 4, "channels": 22,
            "feature_dimension": 759, "object_path": config["parent_artifacts"]["bnci_objects"]["path"],
            "object_sha256": parent_hashes["bnci_objects"], "object_gate": "PASS",
        },
    ]
    pd.DataFrame(contract_rows).to_csv(tables / "dataset_object_contract.csv", index=False, lineterminator="\n")
    null_summary = pd.read_csv(tables / "openbmi_primary_null_summaries.csv")
    null_summary.loc[null_summary["null"] == "equal_rank_random_subspace"].to_csv(
        tables / "openbmi_random_subspace_summary.csv", index=False, lineterminator="\n", float_format="%.17g"
    )
    pd.DataFrame([{
        "dataset": "BNCI2014_001", "statistic": bnci["statistic"],
        "forward_median": bnci["forward_median"], "reverse_median": bnci["reverse_median"],
        "pairing_p": bnci["pairing_p"], "class_p": bnci["class_p"],
        "selected_ranks": "|".join(map(str, bnci["selected_ranks"])),
        "full_space_statistic": bnci["full_space_statistic"],
        "influence_sign_pass": bnci["influence_sign_pass"],
        "interpretation": bnci["interpretation"],
    }]).to_csv(tables / "bnci_multiclass_diagnostic_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    selected = np.asarray(observed["selected_ranks"], dtype=int)
    gate_rows = [
        {"gate": "parent/data contract", "observed": "PASS", "criterion": "PASS", "passed": True},
        {"gate": "measurement reliability", "observed": "PASS" if reliability["passed"] else "FAIL", "criterion": "both sessions p<=0.05; positive; influence", "passed": bool(reliability["passed"])},
        {"gate": "primary statistic", "observed": observed["statistic"], "criterion": ">0", "passed": observed["statistic"] > 0},
        {"gate": "session 0->1 direction", "observed": observed["forward_median"], "criterion": ">0", "passed": observed["forward_median"] > 0},
        {"gate": "session 1->0 direction", "observed": observed["reverse_median"], "criterion": ">0", "passed": observed["reverse_median"] > 0},
        {"gate": "subject-pairing null", "observed": terminal["pairing_p"], "criterion": "p<=0.05", "passed": terminal["pairing_p"] <= 0.05},
        {"gate": "class-semantics null", "observed": terminal["class_p"], "criterion": "p<=0.05", "passed": terminal["class_p"] <= 0.05},
        {"gate": "equal-rank random subspace", "observed": terminal["random_subspace_p"], "criterion": "p<=0.05", "passed": terminal["random_subspace_p"] <= 0.05},
        {"gate": "median selected rank", "observed": float(np.median(selected)), "criterion": "<=8", "passed": float(np.median(selected)) <= 8},
        {"gate": "fold rank frequency", "observed": int(np.count_nonzero(selected <= 8)), "criterion": ">=4 of 6", "passed": int(np.count_nonzero(selected <= 8)) >= 4},
        {"gate": "leave-one-subject sign", "observed": "PASS" if observed["influence_sign_pass"] else "FAIL", "criterion": "all >0", "passed": bool(observed["influence_sign_pass"])},
        {"gate": "full-space stability", "observed": "PASS" if terminal["full_space_stable"] else "FAIL", "criterion": "T>0 and pairing p<=0.05", "passed": bool(terminal["full_space_stable"])},
    ]
    pd.DataFrame(gate_rows).to_csv(tables / "terminal_gate_table.csv", index=False, lineterminator="\n", float_format="%.17g")

    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (6.4, 4.2), "font.size": 9, "axes.grid": False})

    rank_rows = pd.read_csv(tables / "openbmi_rank_by_rank_scores.csv")
    rank_source = rank_rows.groupby("rank", as_index=False).agg(
        median_delta=("delta_average", "median"),
        forward_median=("delta_forward", "median"), reverse_median=("delta_reverse", "median")
    )
    rank_source.to_csv(figures / "figure_1_heldout_score_vs_rank.csv", index=False, lineterminator="\n", float_format="%.17g")
    fig, ax = plt.subplots()
    ax.plot(rank_source["rank"], rank_source["median_delta"], marker="o", label="bidirectional")
    ax.plot(rank_source["rank"], rank_source["forward_median"], marker=".", label="0→1")
    ax.plot(rank_source["rank"], rank_source["reverse_median"], marker=".", label="1→0")
    ax.axhline(0, color="black", linewidth=0.8); ax.set(xlabel="rank", ylabel="held-out separation", title="OpenBMI held-out score by frozen rank")
    ax.legend(frameon=False); _save_figure(fig, figures, "figure_1_heldout_score_vs_rank"); plt.close(fig)

    rank_distribution = pd.Series(selected).value_counts().sort_index().rename_axis("rank").reset_index(name="outer_fold_count")
    rank_distribution.to_csv(figures / "figure_2_selected_rank_distribution.csv", index=False, lineterminator="\n")
    fig, ax = plt.subplots(); ax.bar(rank_distribution["rank"].astype(str), rank_distribution["outer_fold_count"])
    ax.axvline(-10, color="none"); ax.set(xlabel="selected rank", ylabel="outer folds", title="Frozen inner-CV selected ranks")
    _save_figure(fig, figures, "figure_2_selected_rank_distribution"); plt.close(fig)

    for number, kind, title in (
        (3, "pairing", "Subject-pairing destruction"),
        (4, "class", "Class-semantics destruction"),
        (5, "random", "Equal-rank random subspace"),
    ):
        with np.load(output / f"nulls/openbmi_{kind}_null.npz", allow_pickle=False) as archive:
            values = np.asarray(archive["statistics"])[:, 0]
        source = pd.DataFrame({"replicate": np.arange(len(values)), "null_statistic": values})
        source.to_csv(figures / f"figure_{number}_{kind}_null.csv", index=False, lineterminator="\n", float_format="%.17g")
        fig, ax = plt.subplots(); ax.hist(values, bins=35, color="#6d86a6", alpha=0.85)
        ax.axvline(observed["statistic"], color="#b23a48", linewidth=2, label="observed")
        ax.set(xlabel="median held-out separation", ylabel="replicates", title=title); ax.legend(frameon=False)
        _save_figure(fig, figures, f"figure_{number}_{kind}_null"); plt.close(fig)

    with np.load(output / "objects/openbmi_observed_core.npz", allow_pickle=False) as archive:
        similarity = np.asarray(archive["heldout_similarity_block_matrix"])
        mask = np.asarray(archive["heldout_similarity_mask"]).astype(bool)
    rr, cc = np.nonzero(mask)
    pd.DataFrame({"session0_subject": rr + 1, "session1_subject": cc + 1, "similarity": similarity[rr, cc]}).to_csv(
        figures / "figure_6_selected_latent_similarity_matrix.csv", index=False, lineterminator="\n", float_format="%.17g"
    )
    display = np.ma.masked_where(~mask, similarity)
    fig, ax = plt.subplots(figsize=(5.4, 4.8)); image = ax.imshow(display, cmap="coolwarm", aspect="auto")
    ax.set(xlabel="session 1 subject", ylabel="session 0 subject", title="Held-out selected-latent similarity (fold blocks)")
    fig.colorbar(image, ax=ax, shrink=0.8); _save_figure(fig, figures, "figure_6_selected_latent_similarity_matrix"); plt.close(fig)

    reliability_frame = pd.read_csv(tables / "reliability_subject_scores.csv")
    reliability_frame.to_csv(figures / "figure_7_split_half_reliability.csv", index=False, lineterminator="\n", float_format="%.17g")
    fig, ax = plt.subplots()
    for index, session in enumerate(config["datasets"]["openbmi"]["sessions"]):
        values = reliability_frame.loc[reliability_frame["session"].astype(str) == str(session), "delta_average"].to_numpy()
        ax.scatter(np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)), values, s=12, alpha=0.7)
        ax.plot(index, np.median(values), marker="_", markersize=18, color="black")
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks([0, 1], ["session 0", "session 1"])
    ax.set(ylabel="A/B same-subject separation", title="Outer-fold-safe measurement reliability")
    _save_figure(fig, figures, "figure_7_split_half_reliability"); plt.close(fig)

    latent = pd.read_csv(tables / "openbmi_latent_mode1_scores.csv")
    latent.to_csv(figures / "figure_8_openbmi_latent_score_scatter.csv", index=False, lineterminator="\n", float_format="%.17g")
    fig, ax = plt.subplots(); ax.scatter(latent["score_session0_mode1"], latent["score_session1_mode1"], s=24, alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.6); ax.axvline(0, color="grey", linewidth=0.6)
    ax.set(xlabel="session 0 standardized mode-1 score", ylabel="session 1 standardized mode-1 score", title="Held-out paired first-mode coordinates")
    _save_figure(fig, figures, "figure_8_openbmi_latent_score_scatter"); plt.close(fig)

    loadings = pd.read_csv(tables / "bnci_class_mode_loadings.csv")
    loading_source = loadings.groupby(["class", "mode"], as_index=False)["loading_energy"].median()
    loading_source.to_csv(figures / "figure_9_bnci_class_mode_loading.csv", index=False, lineterminator="\n", float_format="%.17g")
    pivot = loading_source.pivot(index="class", columns="mode", values="loading_energy").fillna(0.0)
    fig, ax = plt.subplots(figsize=(5.6, 3.6)); image = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns]); ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set(xlabel="mode", ylabel="class", title="BNCI median class loading energy")
    fig.colorbar(image, ax=ax, shrink=0.8); _save_figure(fig, figures, "figure_9_bnci_class_mode_loading"); plt.close(fig)

    controls = pd.read_csv(tables / "openbmi_controls.csv")
    sensor_spectrum = controls[controls["control"].isin(["sensor_primary", "ordered_Z_eigenvalues"])][["control", "statistic"]]
    sensor_spectrum.to_csv(figures / "figure_10_sensor_vs_spectrum.csv", index=False, lineterminator="\n", float_format="%.17g")
    fig, ax = plt.subplots(); ax.bar(sensor_spectrum["control"], sensor_spectrum["statistic"], color=["#315a7d", "#b6a36a"])
    ax.axhline(0, color="black", linewidth=0.8); ax.set(ylabel="held-out separation", title="Sensor primary versus invariant spectrum control")
    _save_figure(fig, figures, "figure_10_sensor_vs_spectrum"); plt.close(fig)

    decision = terminal["terminal_decision"]
    positive = decision.startswith("GO_")
    next_question = (
        "Can an unseen subject's coordinates in the stable interaction subspace be identified from unlabeled marginal EEG without reliable pseudo-labels?"
        if positive else
        "What additional supervision or physiological anchor is required when stable individual interaction does not admit a transferable population-shared low-rank representation?"
    )
    rank_frequency = pd.Series(selected).value_counts().sort_index()
    report = f"""# Cross-Session Population Structure of Subject-Class Interaction V1

## Outcome

The frozen primary terminal is **{decision}**. The outer-held-out OpenBMI paired-session statistic is `{observed['statistic']:.6f}` (95% subject-bootstrap CI `{observed['bootstrap_ci_95'][0]:.6f}` to `{observed['bootstrap_ci_95'][1]:.6f}`); direction medians are `{observed['forward_median']:.6f}` for session 0→1 and `{observed['reverse_median']:.6f}` for session 1→0.

Subject-pairing, class-semantics, and equal-rank random-subspace p-values are `{terminal['pairing_p']:.6f}`, `{terminal['class_p']:.6f}`, and `{terminal['random_subspace_p']:.6f}`. The full-space outer-fold-safe baseline statistic is `{observed['full_space_statistic']:.6f}` with pairing p `{terminal['full_space_pairing_p']:.6f}` and stable-signal gate `{terminal['full_space_stable']}`.

Selected ranks by outer fold are `{', '.join(map(str, selected.tolist()))}` (median `{np.median(selected):.1f}`; frequencies `{dict((int(k), int(v)) for k, v in rank_frequency.items())}`). The frozen low cap is 8; `{np.count_nonzero(selected <= 8)}` of 6 folds are at or below it. Leave-one-subject sign stability is `{observed['influence_sign_pass']}`.

## Established by parent work

PR #2 established a stable montage-registered mean-level subject×class interaction, OpenBMI cross-session replication, and the limitation of the spectrum-only representation (`GO_SENSOR_SPACE_ONLY`). PR #4 supported the pairwise common-action leave-one-class-out necessary consequence, but not a globally identifiable or cycle-consistent subject action.

## Tested here

This work rebuilt `Z` from immutable `U` separately inside every outer/inner/null split. It tested held-out-subject population structure, predictive effective rank, low-rank versus full-space behavior, random-subspace specificity, class-semantic dependence, split-half reliability, and a four-class BNCI diagnostic. Stored V0 final `Z` never entered feature construction.

OpenBMI reliability prerequisite: `{reliability['passed']}`. Session-specific details are in `tables/reliability_statistics.csv`. Every pairing and class null reran inner rank selection; exact mappings are saved under `nulls/`.

## Secondary controls

Magnitude, same-session PCA, ordered-`Z` eigenvalue, and generalized-eigen controls are in `tables/openbmi_controls.csv`. They do not vote and cannot rescue the sensor primary. The existing OpenBMI V0 spectrum Stage-C failure remains the relevant parent limitation.

BNCI is secondary only: `T={bnci['statistic']:.6f}`, direction medians `{bnci['forward_median']:.6f}` and `{bnci['reverse_median']:.6f}`, pairing p `{bnci['pairing_p']:.6f}`, class p `{bnci['class_p']:.6f}`, and selected ranks `{', '.join(map(str, bnci['selected_ranks']))}`. With nine subjects it cannot establish population rank or overturn OpenBMI. Class/mode and class-pair loading energies are descriptive and no pair was selected after results.

The PR #4 overlap status is `{bnci['action_overlap_status']}`. Subject-level score/action-gain associations are reported, but no new `Q_s` was fit and pairwise actions are not reinterpreted as a global action.

## What is established by this terminal

{('The frozen gates support population-shared predictive linear structure under the exact terminal label above. Any low-dimensional claim is limited to the predeclared held-out-subject rank cap and montage-registered sensor coordinates.' if positive else 'The frozen gates do not support the positive population-structure claim under the exact terminal label above. This does not negate the parent evidence that the interaction object itself is stable.')}

## Not established

This analysis does not estimate a full conditional probability distribution or class-conditional dispersion, multimodality, or higher-order shape. It does not establish physiology, source anatomy, causality, an ASD biomarker, clinical diagnosis, target-unlabeled inference, TTA recoverability, a nonlinear/universal individuality manifold, globally identifiable `Q_s`, or equality of modes across datasets. It proposes no network, adapter, loss, classifier, or TTA method.

## Prospective and clinical boundaries

Stieger2021 is documented only as a future prospective-confirmation candidate in `docs/STIEGER2021_CONFIRMATION_FEASIBILITY.md`; no raw data or statistic was accessed. The ASD extension boundary is documented in `docs/ASD_EXTENSION_BOUNDARY.md`; no ASD repository or data was modified.

## Next question

{next_question}
"""
    report_path = report_dir / "subject_class_population_structure_v1.md"
    atomic_write_bytes(report_path, report.encode("utf-8"))
    validate_report_consistency(root)
    artifact_hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": "subject-class-population-structure-final-manifest-v1",
        "terminal_decision": decision, "config_sha256": config_hash,
        "parent_hashes": parent_hashes, "artifact_hashes": artifact_hashes,
        "report_sha256": sha256_file(report_path),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {
        "terminal_decision": decision, "report": str(report_path.relative_to(root)),
        "manifest": str((output / "manifest.json").relative_to(root)),
        "artifact_count": len(artifact_hashes), "report_sha256": manifest["report_sha256"],
    }
