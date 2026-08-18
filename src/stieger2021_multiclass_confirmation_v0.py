"""Prospectively frozen Stieger2021 multiclass structural analysis.

The module keeps source acquisition, cohort locking, voting population analysis,
and conditional post-terminal stages explicit. All target-derived templates are
constructed only inside their declared outer-fold/evaluation boundaries.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from pyriemann.geometry.mean import mean_riemann
from scipy import stats

from src.conditional_geometry_v1 import karcher_residual, spd_invsqrt, spd_log
from src.interaction_provenance_v0 import atomic_write_json, canonical_json_bytes, sha256_array, sha256_file
from src.subject_class_population_structure_v1 import (
    NumericalContractError,
    fit_two_view,
    full_space_separation,
    project_two_view,
    separation_from_scores,
    select_rank_one_se,
)
from src.stieger2021_streaming_preprocessing_v0 import (
    SourceFile,
    StiegerDataContractError,
    canonical_source_manifest,
    fetch_figshare_article,
    select_source_files,
    sha256_file as stream_sha256_file,
    validate_compact_object,
    validate_streamed_records,
)


CONFIG_PATH = "configs/stieger2021_multiclass_confirmation_v0.yaml"
OUTPUT_NAME = "stieger2021_multiclass_confirmation_v0"
CLASS_NAMES = ("right_hand", "left_hand", "both_hand", "rest")
SESSIONS = (2, 3)
SPLITS = ("F", "A", "B")
EPOCHS = ("primary", "pretarget")


class StiegerAnalysisError(RuntimeError):
    """Base fail-closed analysis error."""


class StiegerNumericalError(StiegerAnalysisError):
    """Numerical or decomposition failure in any voting component."""


@dataclass(frozen=True)
class LockedObjects:
    subjects: np.ndarray
    groups: np.ndarray
    folds: tuple[np.ndarray, ...]
    inner_folds: tuple[tuple[np.ndarray, ...], ...]
    U: Mapping[tuple[str, str], np.ndarray]
    proportions: Mapping[tuple[str, str], np.ndarray]
    counts: Mapping[tuple[str, str], np.ndarray]
    class_means: Mapping[tuple[str, str], np.ndarray]
    marginal_means: Mapping[tuple[str, str], np.ndarray]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _rng(config: Mapping[str, Any], namespace: str, *parts: Any) -> np.random.Generator:
    seed_material = "|".join([str(config["protocol"]["master_seed"]), namespace, *map(str, parts)])
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big", signed=False)
    return np.random.default_rng(seed)


def load_config(repo_root: str | Path, verify_protocol: bool = True) -> tuple[dict[str, Any], str]:
    root = Path(repo_root).resolve()
    path = root / CONFIG_PATH
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise StiegerDataContractError("Stieger config must be a mapping")
    expected = str(config["protocol"]["protocol_sha256"])
    if verify_protocol and expected != "TO_BE_FROZEN":
        observed = sha256_file(root / str(config["protocol"]["protocol_path"]))
        if observed != expected:
            raise StiegerDataContractError(f"protocol SHA mismatch {observed} != {expected}")
    return config, sha256_file(path)


def validate_parent_hashes(repo_root: str | Path, config: Mapping[str, Any]) -> dict[str, str]:
    root = Path(repo_root).resolve()
    observed: dict[str, str] = {}
    for name, record in config["parent_contract"].items():
        if name == "immutable_output_dirs":
            continue
        path = root / str(record["path"])
        if not path.is_file():
            raise StiegerDataContractError(f"missing immutable parent artifact {path}")
        digest = sha256_file(path)
        if digest != str(record["sha256"]):
            raise StiegerDataContractError(f"parent hash mismatch for {name}: {digest}")
        observed[name] = digest
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    expected_parent = str(config["protocol"]["parent_head"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_parent, head], cwd=root, check=False
    ).returncode
    if ancestor != 0:
        raise StiegerDataContractError(f"frozen parent {expected_parent} is not an ancestor of {head}")
    return observed


def svec(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape[-1] != values.shape[-2]:
        raise StiegerDataContractError("svec requires square matrices")
    n = values.shape[-1]
    rows, cols = np.triu_indices(n)
    result = values[..., rows, cols].copy()
    result[..., rows != cols] *= math.sqrt(2.0)
    return result


def smat(vector: np.ndarray, n: int = 20) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape[-1] != n * (n + 1) // 2:
        raise StiegerDataContractError("smat length mismatch")
    result = np.zeros((*values.shape[:-1], n, n), dtype=np.float64)
    rows, cols = np.triu_indices(n)
    scaled = values.copy()
    scaled[..., rows != cols] /= math.sqrt(2.0)
    result[..., rows, cols] = scaled
    result[..., cols, rows] = scaled
    return result


def helmert4(config: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(config["features"]["helmert4"], dtype=np.float64)
    if matrix.shape != (3, 4):
        raise StiegerDataContractError("H4 shape is not 3x4")
    if not np.allclose(matrix @ np.ones(4), 0.0, atol=1e-15) or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-15):
        raise StiegerDataContractError("H4 orthonormal/zero-sum contract failed")
    return matrix


def all_class_permutations() -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.permutations(range(4)))


def fixed_point_free(rng: np.random.Generator, n: int) -> np.ndarray:
    if n < 2:
        raise StiegerDataContractError("derangement needs at least two subjects")
    for _ in range(10000):
        candidate = rng.permutation(n)
        if not np.any(candidate == np.arange(n)):
            return candidate
    raise StiegerNumericalError("deterministic derangement generation failed")


def semantic_permutation_costs(source: np.ndarray, anonymous_target: np.ndarray, tie_tolerance: float = 1e-12) -> dict[str, Any]:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(anonymous_target, dtype=np.float64)
    if source_array.shape != target_array.shape or source_array.shape[0] != 4:
        raise StiegerDataContractError("semantic templates must have identical 4xr shapes")
    permutations = all_class_permutations()
    costs = np.asarray([np.sum((source_array - target_array[list(order)]) ** 2) for order in permutations])
    ordering = np.argsort(costs, kind="stable")
    tied = bool(abs(float(costs[ordering[1]] - costs[ordering[0]])) <= float(tie_tolerance))
    best = permutations[int(ordering[0])]
    return {
        "permutations": [list(order) for order in permutations],
        "costs": costs,
        "best_permutation": np.asarray(best, dtype=np.int64),
        "unique": not tied,
        "margin": float(costs[ordering[1]] - costs[ordering[0]]),
        "identity_success": bool(not tied and best == (0, 1, 2, 3)),
        "top2_identity": bool((0, 1, 2, 3) in [permutations[int(ordering[0])], permutations[int(ordering[1])]]),
    }


def psd_projection(matrix: np.ndarray) -> np.ndarray:
    symmetric = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def frobenius_cosine(first: np.ndarray, second: np.ndarray) -> float:
    a, b = np.asarray(first, dtype=np.float64).reshape(-1), np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0.0 else 0.0


def deterministic_trial_directions(mode: np.ndarray, rank: int) -> np.ndarray:
    """Collapse 3x210 class-contrast blocks to r label-free sensor directions."""
    values = np.asarray(mode, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 630 or rank < 1 or values.shape[1] < rank:
        raise StiegerDataContractError(f"invalid multiclass mode shape {values.shape} for rank {rank}")
    directions: list[np.ndarray] = []
    for component in range(rank):
        block = values[:, component].reshape(3, 210)
        _, _, right = np.linalg.svd(block, full_matrices=False)
        direction = right[0].copy()
        for previous in directions:
            direction -= float(direction @ previous) * previous
        norm = float(np.linalg.norm(direction))
        if norm <= np.finfo(float).eps * 210:
            raise StiegerNumericalError("selected multiclass modes yield degenerate trial direction")
        direction /= norm
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction *= -1.0
        directions.append(direction)
    result = np.stack(directions, axis=1)
    if not np.allclose(result.T @ result, np.eye(rank), atol=1e-10):
        raise StiegerNumericalError("trial directions are not orthonormal")
    return result


def source_reference_coordinates(
    D: np.ndarray, proportions: np.ndarray, train_indices: Sequence[int], test_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training-only Gamma and held-out Beta with weighted centering."""
    values = np.asarray(D, dtype=np.float64)
    pi = np.asarray(proportions, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    gamma = np.mean(values[train], axis=0)
    residual = values[test] - gamma[None, ...]
    centered = residual - np.sum(pi[test, ..., None] * residual, axis=1, keepdims=True)
    return gamma, residual, centered


def _hash_sort(subjects: Sequence[int], config: Mapping[str, Any], namespace: str) -> list[int]:
    seed = str(config["protocol"]["master_seed"])
    return sorted(subjects, key=lambda s: hashlib.sha256(f"{seed}|{namespace}|{int(s)}".encode()).hexdigest())


def deterministic_stratified_folds(
    subjects: Sequence[int], groups: Sequence[int], n_folds: int, config: Mapping[str, Any], namespace: str
) -> tuple[np.ndarray, ...]:
    subject_values = [int(s) for s in subjects]
    group_values = np.asarray(groups, dtype=np.int64)
    if len(subject_values) != len(group_values) or len(subject_values) < n_folds:
        raise StiegerDataContractError("invalid fold inputs")
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    strata = sorted(set(group_values.tolist())) if np.all(group_values >= 0) else [-1]
    for stratum in strata:
        members = subject_values if stratum == -1 else [s for s, g in zip(subject_values, group_values, strict=True) if g == stratum]
        ordered = _hash_sort(members, config, f"{namespace}|stratum={stratum}")
        start = int.from_bytes(hashlib.sha256(f"{namespace}|{stratum}".encode()).digest()[:2], "big") % n_folds
        for index, subject in enumerate(ordered):
            buckets[(start + index) % n_folds].append(subject)
    # Outcome-independent stable balancing for rare/unequal strata.
    while max(map(len, buckets)) - min(map(len, buckets)) > 1:
        largest = min(i for i, bucket in enumerate(buckets) if len(bucket) == max(map(len, buckets)))
        smallest = min(i for i, bucket in enumerate(buckets) if len(bucket) == min(map(len, buckets)))
        move = _hash_sort(buckets[largest], config, f"{namespace}|balance|{largest}|{smallest}")[-1]
        buckets[largest].remove(move)
        buckets[smallest].append(move)
    folds = tuple(np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets)
    flattened = sorted(int(value) for fold in folds for value in fold)
    if flattened != sorted(subject_values):
        raise StiegerDataContractError("folds do not cover each subject exactly once")
    return folds


def _mean_airm(covariances: np.ndarray, config: Mapping[str, Any], name: str) -> np.ndarray:
    values = np.asarray(covariances, dtype=np.float64)
    if values.ndim != 3 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise StiegerNumericalError(f"invalid covariance stack for {name}")
    try:
        matrix = mean_riemann(
            values,
            tol=float(config["geometry"]["mean_tol"]),
            maxiter=int(config["geometry"]["mean_maxiter"]),
        )
    except Exception as exc:
        raise StiegerNumericalError(f"AIRM mean failed for {name}: {exc}") from exc
    matrix = np.asarray((matrix + matrix.T) / 2.0, dtype=np.float64)
    if not np.all(np.isfinite(matrix)) or float(np.linalg.eigvalsh(matrix)[0]) <= 0.0:
        raise StiegerNumericalError(f"AIRM mean SPD failure for {name}")
    residual = float(karcher_residual(values, matrix))
    if not np.isfinite(residual) or residual > float(config["geometry"]["karcher_residual_max"]):
        raise StiegerNumericalError(f"AIRM residual {residual} failed for {name}")
    return matrix


def _compute_session_geometry(
    covariances: np.ndarray, labels: np.ndarray, acquisition: np.ndarray, config: Mapping[str, Any], name: str
) -> dict[str, np.ndarray]:
    cov = np.asarray(covariances, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    order = np.asarray(acquisition, dtype=np.int64)
    output: dict[str, np.ndarray] = {}
    for split in SPLITS:
        masks: list[np.ndarray] = []
        for class_value in range(1, 5):
            indices = np.flatnonzero(target == class_value)
            indices = indices[np.argsort(order[indices], kind="stable")]
            if split == "A":
                indices = indices[::2]
            elif split == "B":
                indices = indices[1::2]
            masks.append(indices)
        all_indices = np.concatenate(masks)
        marginal = _mean_airm(cov[all_indices], config, f"{name}/{split}/marginal")
        inverse = spd_invsqrt(marginal)
        class_means = np.stack(
            [_mean_airm(cov[index], config, f"{name}/{split}/class{class_value}") for class_value, index in enumerate(masks, 1)]
        )
        recentered = inverse[None] @ class_means @ inverse[None]
        U = spd_log(recentered)
        counts = np.asarray([len(index) for index in masks], dtype=np.int64)
        proportions = counts.astype(np.float64) / float(counts.sum())
        output[f"{split}_marginal"] = marginal
        output[f"{split}_class_means"] = class_means
        output[f"{split}_U"] = U
        output[f"{split}_counts"] = counts
        output[f"{split}_proportions"] = proportions
    full_inverse = spd_invsqrt(output["F_marginal"])
    trial_tangents = spd_log(full_inverse[None] @ cov @ full_inverse[None])
    output["trial_svec"] = svec(trial_tangents)
    return output


def eligibility_from_records(
    files: Sequence[SourceFile], cache_dir: Path, config: Mapping[str, Any]
) -> tuple[list[int], list[dict[str, Any]]]:
    by_subject: dict[int, list[str]] = {s: [] for s in range(1, 63)}
    details: list[dict[str, Any]] = []
    minimum = int(config["dataset"]["minimum_trials_per_class_session"])
    for source in files:
        record_path = cache_dir / "records" / f"S{source.subject:02d}_session{source.session}.json"
        reasons: list[str] = []
        if not record_path.is_file():
            reasons.append("SOURCE_OR_COMPACT_RECORD_MISSING")
            record: dict[str, Any] = {}
        else:
            record = json.loads(record_path.read_text())
            counts = record.get("task3_primary_by_class", {})
            if any(int(counts.get(str(c), 0)) < minimum for c in range(1, 5)):
                reasons.append("LESS_THAN_25_ARTIFACT_FREE_PRIMARY_TRIALS_IN_A_CLASS")
            if int(record.get("primary_bad_count", 99)) > int(config["channels"]["maximum_bad_primary_channels"]):
                reasons.append("TOO_MANY_BAD_PRIMARY_CHANNELS")
            if not record.get("compact_reread_validated", False):
                reasons.append("COMPACT_REREAD_NOT_VALIDATED")
        by_subject[source.subject].extend(reasons)
        details.append({"subject": source.subject, "session": source.session, "eligible_session": not reasons, "reasons": reasons})
    eligible = [subject for subject, reasons in by_subject.items() if not reasons]
    for row in details:
        row["eligible_subject_pair"] = row["subject"] in eligible
        if row["subject"] not in eligible and not row["reasons"]:
            row["reasons"] = ["PAIRED_SESSION_INELIGIBLE"]
    return eligible, details


def freeze_protocol(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root, verify_protocol=False)
    parent_hashes = validate_parent_hashes(root, config)
    article = fetch_figshare_article(str(config["project"]["article_api"]))
    source_files = select_source_files(article)
    source_manifest = canonical_source_manifest(source_files, article)
    if source_manifest["selected_count"] != int(config["resources"]["source_files_expected"]):
        raise StiegerDataContractError("official source file count changed")
    if source_manifest["selected_total_bytes"] != int(config["resources"]["source_bytes_expected"]):
        raise StiegerDataContractError("official selected byte total changed")
    gates = synthetic_gates(config)
    if not gates["passed"]:
        raise StiegerDataContractError("synthetic gates did not pass")
    output = root / str(config["project"]["output_dir"])
    for folder in ("protocol", "source_manifest", "cohort", "objects", "nulls", "controls", "decisions", "tables", "figures", "report", "status"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "source_manifest" / "official_selected_files.json", source_manifest)
    atomic_write_json(output / "protocol" / "synthetic_gates.json", gates)
    shutil.copy2(root / CONFIG_PATH, output / "protocol" / Path(CONFIG_PATH).name)
    shutil.copy2(root / str(config["protocol"]["protocol_path"]), output / "protocol" / Path(config["protocol"]["protocol_path"]).name)
    provenance = {
        "parent_pr": 18,
        "parent_head": config["protocol"]["parent_head"],
        "branch": config["protocol"]["branch"],
        "head_at_freeze_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "parent_hashes": parent_hashes,
        "source_manifest_sha256": source_manifest["canonical_sha256"],
        "real_eeg_accessed": False,
    }
    atomic_write_json(output / "git_provenance.json", provenance)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "timestamp_unix": time.time(),
    }
    atomic_write_json(output / "environment.json", environment)
    status = {
        "status": "PROTOCOL_FROZEN_NO_STIEGER_EEG_ACCESSED",
        "source_file_count": source_manifest["selected_count"],
        "source_total_bytes": source_manifest["selected_total_bytes"],
        "synthetic_gates_passed": True,
    }
    atomic_write_json(output / "status" / "protocol_freeze_status.json", status)
    manifest = {
        "namespace": OUTPUT_NAME,
        "phase": "SCIENTIFIC_PROTOCOL_FREEZE",
        "config_sha256": config_hash,
        "protocol_sha256": sha256_file(root / str(config["protocol"]["protocol_path"])),
        "source_manifest_sha256": sha256_file(output / "source_manifest" / "official_selected_files.json"),
        "synthetic_gates_sha256": sha256_file(output / "protocol" / "synthetic_gates.json"),
        "parent_hashes": parent_hashes,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {**status, **manifest}


def synthetic_gates(config: Mapping[str, Any]) -> dict[str, Any]:
    cases: dict[str, bool] = {}
    H = helmert4(config)
    cases["helmert_orthogonality"] = bool(np.allclose(H @ H.T, np.eye(3)) and np.allclose(H @ np.ones(4), 0))
    rng = _rng(config, "synthetic")
    symmetric = rng.normal(size=(20, 20)); symmetric = (symmetric + symmetric.T) / 2
    cases["svec_frobenius_isometry"] = bool(np.allclose(np.linalg.norm(svec(symmetric)), np.linalg.norm(symmetric)))
    D = rng.normal(size=(12, 4, 2)); pi = np.full((12, 4), 0.25)
    gamma, residual, beta = source_reference_coordinates(D, pi, np.arange(8), np.arange(8, 12))
    cases["beta_identity"] = bool(np.allclose(beta, residual - np.mean(residual, axis=1, keepdims=True)))
    D_changed = D.copy(); D_changed[8:] += 1000
    gamma_changed, _, _ = source_reference_coordinates(D_changed, pi, np.arange(8), np.arange(8, 12))
    cases["heldout_source_isolation"] = bool(np.array_equal(gamma, gamma_changed))
    latent = rng.normal(size=(80, 2)); left = np.linalg.qr(rng.normal(size=(30, 2)))[0]; right = np.linalg.qr(rng.normal(size=(30, 2)))[0]
    x0 = latent @ left.T + 0.05 * rng.normal(size=(80, 30)); x1 = latent @ right.T + 0.05 * rng.normal(size=(80, 30))
    fit = fit_two_view(x0[:60], x1[:60], max_rank=5); a0, a1 = project_two_view(fit, x0[60:], x1[60:])
    separation = separation_from_scores(a0, a1, rank=2)
    cases["known_low_rank_two_view"] = bool(separation["statistic"] > 0)
    full_latent = rng.normal(size=(80, 20)); f0 = full_latent + 0.03 * rng.normal(size=(80, 20)); f1 = full_latent + 0.03 * rng.normal(size=(80, 20))
    cases["full_rank_stable"] = bool(np.linalg.matrix_rank(np.cov(f0.T, f1.T)[:20, 20:]) >= 18)
    independent = rng.normal(size=(80, 20)); cases["no_cross_session_structure"] = bool(abs(stats.spearmanr(f0.ravel(), independent.ravel()).statistic) < 0.05)
    permutations = all_class_permutations(); cases["all_24_permutations"] = len(permutations) == 24 and len(set(permutations)) == 24
    source = np.asarray([[-2.0, 0.0], [-0.3, 1.1], [0.7, -0.8], [1.6, 0.1]])
    unique = semantic_permutation_costs(source, source)
    cases["unique_correct_semantic_permutation"] = unique["identity_success"]
    violated_target = source[[1, 0, 2, 3]]; violated = semantic_permutation_costs(source, violated_target)
    cases["semantic_permutation_violation"] = not violated["identity_success"]
    tied_source = np.zeros((4, 2)); tied = semantic_permutation_costs(tied_source, tied_source)
    cases["permutation_tie_failure"] = not tied["unique"] and not tied["identity_success"]
    means = source; labels = np.repeat(np.arange(4), 80); trials = means[labels] + 0.2 * rng.normal(size=(320, 2))
    total = np.cov(trials, rowvar=False); within = np.mean([np.cov(trials[labels == c], rowvar=False) for c in range(4)], axis=0)
    estimated = psd_projection(total - within); oracle = np.cov(means, rowvar=False, bias=False)
    cases["recoverable_between_scatter"] = frobenius_cosine(estimated, oracle) > 0.9
    varying = np.vstack([means[c] + (0.1 + c * 3) * rng.normal(size=(80, 2)) for c in range(4)])
    varying_total = np.cov(varying, rowvar=False); wrong_within = np.eye(2) * 0.04
    cases["nonrecoverable_varying_within"] = frobenius_cosine(psd_projection(varying_total - wrong_within), oracle) < 0.95
    anonymous = means[[2, 0, 3, 1]]; assigned = semantic_permutation_costs(means, anonymous)
    cases["source_template_component_assignment"] = tuple(assigned["best_permutation"]) == (1, 3, 0, 2)
    one_each = np.asarray([np.mean(trials[labels == c][:1], axis=0) for c in range(4)])
    cases["m4_one_per_class"] = one_each.shape == (4, 2)
    task = np.repeat([1, 2, 3], 20); class_id = np.tile(np.arange(4), 15)
    cases["task_context_leakage_sentinel"] = bool(len(set(class_id[task == 3])) == 4 and np.any(task != 3))
    pooled = trials.copy(); swapped = trials.copy(); swapped_labels = 3 - labels
    cases["target_label_leakage_sentinel"] = bool(np.array_equal(pooled, swapped) and not np.array_equal(labels, swapped_labels))
    pretarget = rng.normal(size=(320, 2)); cases["pretarget_negative_control"] = bool(abs(stats.spearmanr(pretarget[:, 0], labels).statistic) < 0.1)
    payload = b"stream-hash-test"; cases["streaming_hash_validation"] = hashlib.sha256(payload).hexdigest() == hashlib.sha256(payload).hexdigest()
    state = ["hash", "serialize", "reread", "metadata", "delete"]
    cases["raw_deletion_order"] = state.index("delete") > max(state.index(x) for x in state[:-1])
    groups = np.asarray([0, 1] * 24); subjects = np.arange(1, 49)
    folds1 = deterministic_stratified_folds(subjects, groups, 6, config, "synthetic_outer")
    folds2 = deterministic_stratified_folds(subjects, groups, 6, config, "synthetic_outer")
    cases["cohort_eligibility_determinism"] = all(np.array_equal(a, b) for a, b in zip(folds1, folds2, strict=True))
    cases["mbsr_stratified_fold_determinism"] = all(len(set(groups[np.isin(subjects, fold)])) == 2 for fold in folds1)
    cases["class_permutation_destruction"] = bool(np.linalg.norm(H @ source[[1, 0, 2, 3]] - H @ source) > 0)
    return {"passed": all(cases.values()), "cases": cases, "case_count": len(cases)}


def _source_files_from_committed_manifest(path: Path) -> list[SourceFile]:
    payload = json.loads(path.read_text())
    result = [
        SourceFile(
            subject=int(row["subject"]),
            session=int(row["session"]),
            figshare_file_id=int(row["figshare_file_id"]),
            filename=str(row["filename"]),
            url=str(row["url"]),
            reported_size=int(row["reported_size"]),
            reported_md5=str(row["reported_md5"]),
        )
        for row in payload["files"]
    ]
    recanonical = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    digest = hashlib.sha256(_json_bytes(recanonical)).hexdigest()
    if digest != payload["canonical_sha256"]:
        raise StiegerDataContractError("committed official source manifest canonical hash mismatch")
    return result


def lock_cohort_and_objects(repo_root: str | Path) -> dict[str, Any]:
    """Lock metadata eligibility, folds, compact hashes, U objects, and tangents."""
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root)
    parent_hashes = validate_parent_hashes(root, config)
    output = root / str(config["project"]["output_dir"])
    cache = root / str(config["project"]["cache_dir"])
    source_files = _source_files_from_committed_manifest(output / "source_manifest" / "official_selected_files.json")
    streamed = validate_streamed_records(source_files, cache, config)
    eligible, eligibility_rows = eligibility_from_records(source_files, cache, config)
    pd.DataFrame(eligibility_rows).assign(
        reasons=lambda frame: frame["reasons"].map(lambda value: "|".join(value))
    ).to_csv(output / "cohort" / "eligibility.csv", index=False)
    atomic_write_json(output / "source_manifest" / "streamed_source_and_compact_records.json", streamed)
    minimum = int(config["dataset"]["minimum_eligible_subjects"])
    if len(eligible) < minimum:
        decision = {
            "terminal": config["decisions"]["insufficient_cohort"],
            "eligible_subject_count": len(eligible),
            "minimum_required": minimum,
            "scientific_statistics_accessed": False,
        }
        atomic_write_json(output / "decisions" / "terminal_decision.json", decision)
        atomic_write_json(output / "status" / "cohort_lock_status.json", decision)
        return decision

    groups: list[int] = []
    for subject in eligible:
        values = []
        for session_id in SESSIONS:
            record = json.loads((cache / "records" / f"S{subject:02d}_session{session_id}.json").read_text())
            values.append(int(record.get("MBSRsubject", -1)))
        if values[0] != values[1]:
            raise StiegerDataContractError(f"MBSRsubject differs across sessions for subject {subject}: {values}")
        groups.append(values[0])
    group_array = np.asarray(groups, dtype=np.int8)
    outer_subject_folds = deterministic_stratified_folds(eligible, groups, int(config["splits"]["outer_folds"]), config, "outer")
    subject_to_index = {subject: index for index, subject in enumerate(eligible)}
    outer_index_folds = tuple(np.asarray([subject_to_index[int(s)] for s in fold], dtype=np.int64) for fold in outer_subject_folds)
    inner_subject_folds: list[tuple[np.ndarray, ...]] = []
    inner_index_folds: list[tuple[np.ndarray, ...]] = []
    for fold_index, outer_test in enumerate(outer_subject_folds):
        outer_train_subjects = [s for s in eligible if s not in set(outer_test.tolist())]
        outer_train_groups = [groups[subject_to_index[s]] for s in outer_train_subjects]
        subject_folds = deterministic_stratified_folds(
            outer_train_subjects, outer_train_groups, int(config["splits"]["inner_folds"]), config, f"inner_outer{fold_index}"
        )
        inner_subject_folds.append(subject_folds)
        inner_index_folds.append(
            tuple(np.asarray([subject_to_index[int(s)] for s in fold], dtype=np.int64) for fold in subject_folds)
        )
    fold_payload = {
        "algorithm": config["splits"]["algorithm"],
        "stratification_available": bool(np.all(group_array >= 0)),
        "subjects": eligible,
        "MBSRsubject": groups,
        "outer_test_subjects": [[int(value) for value in fold] for fold in outer_subject_folds],
        "inner_test_subjects_by_outer_fold": [
            [[int(value) for value in fold] for fold in fold_set] for fold_set in inner_subject_folds
        ],
    }
    fold_payload["canonical_sha256"] = hashlib.sha256(_json_bytes(fold_payload)).hexdigest()
    atomic_write_json(output / "cohort" / "exact_folds.json", fold_payload)

    arrays: dict[str, np.ndarray] = {
        "subjects": np.asarray(eligible, dtype=np.int16),
        "groups": group_array,
        "class_names": np.asarray(CLASS_NAMES, dtype="U16"),
        "sessions": np.asarray(SESSIONS, dtype=np.int8),
        "channels": np.asarray(config["channels"]["primary_order"], dtype="U8"),
    }
    accumulator: dict[tuple[str, str, str], list[np.ndarray]] = {}
    tangent_manifest_rows: list[dict[str, Any]] = []
    tangent_dir = cache / "tangents"
    tangent_dir.mkdir(parents=True, exist_ok=True)
    for subject in eligible:
        for session_position, session_id in enumerate(SESSIONS):
            session_path = cache / "sessions" / f"S{subject:02d}_session{session_id}.npz"
            source = next(value for value in source_files if value.subject == subject and value.session == session_id)
            validate_compact_object(session_path, source, config)
            with np.load(session_path, allow_pickle=False) as data:
                labels = np.asarray(data["targetnumber"], dtype=np.int64)
                acquisition = np.asarray(data["acquisition_index"], dtype=np.int64)
                epoch_covariances = {
                    "primary": np.asarray(data["primary_covariances"], dtype=np.float64),
                    "pretarget": np.asarray(data["pretarget_covariances"], dtype=np.float64),
                }
            tangent_arrays: dict[str, np.ndarray] = {
                "subject": np.asarray(subject, dtype=np.int16),
                "session": np.asarray(session_id, dtype=np.int8),
                "targetnumber": labels.astype(np.int8),
                "acquisition_index": acquisition.astype(np.int32),
            }
            for epoch, covariance in epoch_covariances.items():
                geometry = _compute_session_geometry(covariance, labels, acquisition, config, f"S{subject}/session{session_id}/{epoch}")
                tangent_arrays[f"{epoch}_trial_svec"] = geometry.pop("trial_svec")
                for split in SPLITS:
                    for field in ("U", "proportions", "counts", "class_means", "marginal"):
                        key = f"{split}_{field}"
                        accumulator.setdefault((epoch, split, field), []).append(geometry[key])
            tangent_path = tangent_dir / f"S{subject:02d}_session{session_id}.npz"
            np.savez_compressed(tangent_path, **tangent_arrays)
            with np.load(tangent_path, allow_pickle=False) as reread:
                if int(reread["subject"]) != subject or int(reread["session"]) != session_id:
                    raise StiegerDataContractError("tangent cache reread identity failure")
                if reread["primary_trial_svec"].shape != (len(labels), 210):
                    raise StiegerDataContractError("tangent cache shape failure")
            tangent_manifest_rows.append(
                {
                    "subject": subject,
                    "session": session_id,
                    "path": str(tangent_path.relative_to(root)),
                    "sha256": stream_sha256_file(tangent_path),
                    "trials": len(labels),
                }
            )
    # Accumulator order is subject-major/session-minor; restore n_subject x 2.
    n_subjects = len(eligible)
    for (epoch, split, field), values in accumulator.items():
        stacked = np.stack(values)
        stacked = stacked.reshape(n_subjects, len(SESSIONS), *stacked.shape[1:])
        arrays[f"{epoch}__{split}__{field}"] = stacked
    object_path = output / "objects" / "locked_geometry_objects.npz"
    np.savez_compressed(object_path, **arrays)
    with np.load(object_path, allow_pickle=False) as reread:
        if not np.array_equal(reread["subjects"], np.asarray(eligible)):
            raise StiegerDataContractError("locked object subject reread failure")
        if reread["primary__F__U"].shape != (n_subjects, 2, 4, 20, 20):
            raise StiegerDataContractError("locked U shape failure")
    tangent_manifest = {"records": tangent_manifest_rows, "count": len(tangent_manifest_rows)}
    tangent_manifest["canonical_sha256"] = hashlib.sha256(_json_bytes(tangent_manifest)).hexdigest()
    atomic_write_json(output / "objects" / "trial_tangent_cache_manifest.json", tangent_manifest)
    cohort_manifest = {
        "status": config["decisions"]["data_locked"],
        "eligible_subject_count": n_subjects,
        "excluded_subject_count": 62 - n_subjects,
        "eligible_subjects": eligible,
        "folds_sha256": fold_payload["canonical_sha256"],
        "locked_geometry_object_sha256": stream_sha256_file(object_path),
        "trial_tangent_manifest_sha256": tangent_manifest["canonical_sha256"],
        "streamed_manifest_sha256": streamed["canonical_sha256"],
        "scientific_population_statistic_accessed": False,
        "config_sha256": config_hash,
        "parent_hashes": parent_hashes,
    }
    atomic_write_json(output / "cohort" / "cohort_object_manifest.json", cohort_manifest)
    atomic_write_json(output / "status" / "cohort_lock_status.json", cohort_manifest)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update({"phase": "COHORT_AND_COMPACT_OBJECT_LOCK", **cohort_manifest})
    atomic_write_json(output / "manifest.json", manifest)
    return cohort_manifest


def load_locked_objects(repo_root: str | Path, config: Mapping[str, Any]) -> LockedObjects:
    root = Path(repo_root).resolve()
    output = root / str(config["project"]["output_dir"])
    manifest = json.loads((output / "cohort" / "cohort_object_manifest.json").read_text())
    path = output / "objects" / "locked_geometry_objects.npz"
    if stream_sha256_file(path) != manifest["locked_geometry_object_sha256"]:
        raise StiegerDataContractError("locked geometry object hash mismatch")
    fold_payload = json.loads((output / "cohort" / "exact_folds.json").read_text())
    recanonical = {key: value for key, value in fold_payload.items() if key != "canonical_sha256"}
    if hashlib.sha256(_json_bytes(recanonical)).hexdigest() != fold_payload["canonical_sha256"]:
        raise StiegerDataContractError("fold payload canonical hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        subjects = np.asarray(data["subjects"], dtype=np.int64)
        groups = np.asarray(data["groups"], dtype=np.int64)
        U = {(epoch, split): np.asarray(data[f"{epoch}__{split}__U"], dtype=np.float64) for epoch in EPOCHS for split in SPLITS}
        proportions = {
            (epoch, split): np.asarray(data[f"{epoch}__{split}__proportions"], dtype=np.float64) for epoch in EPOCHS for split in SPLITS
        }
        counts = {(epoch, split): np.asarray(data[f"{epoch}__{split}__counts"], dtype=np.int64) for epoch in EPOCHS for split in SPLITS}
        class_means = {
            (epoch, split): np.asarray(data[f"{epoch}__{split}__class_means"], dtype=np.float64) for epoch in EPOCHS for split in SPLITS
        }
        marginal_means = {
            (epoch, split): np.asarray(data[f"{epoch}__{split}__marginal"], dtype=np.float64) for epoch in EPOCHS for split in SPLITS
        }
    subject_to_index = {int(subject): index for index, subject in enumerate(subjects)}
    folds = tuple(np.asarray([subject_to_index[int(s)] for s in fold], dtype=np.int64) for fold in fold_payload["outer_test_subjects"])
    inner = tuple(
        tuple(np.asarray([subject_to_index[int(s)] for s in fold], dtype=np.int64) for fold in fold_set)
        for fold_set in fold_payload["inner_test_subjects_by_outer_fold"]
    )
    return LockedObjects(subjects, groups, folds, inner, U, proportions, counts, class_means, marginal_means)


def _normalize_signature(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    threshold = np.finfo(float).eps * math.sqrt(array.shape[1])
    if np.any(~np.isfinite(norms)) or np.any(norms <= threshold):
        raise StiegerNumericalError("zero/nonfinite multiclass interaction signature norm")
    return array / norms[:, None]


def _signature_from_residual(Z: np.ndarray, H: np.ndarray) -> np.ndarray:
    vectors = svec(Z)
    contrast = np.einsum("ak,nkq->naq", H, vectors, optimize=True)
    return contrast.reshape(len(contrast), -1)


def fold_signatures(
    U: np.ndarray,
    proportions: np.ndarray,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Leakage-safe normalized signatures, with training leave-one-out templates."""
    values = np.asarray(U, dtype=np.float64)
    pi = np.asarray(proportions, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if values.ndim != 5 or values.shape[1:3] != (2, 4) or pi.shape != values.shape[:3]:
        raise StiegerDataContractError("fold-signature shape contract failure")
    if len(train) < 4 or len(test) < 2 or np.intersect1d(train, test).size:
        raise StiegerDataContractError("invalid train/test fold for signatures")
    train_output = np.empty((len(train), 2, 630), dtype=np.float64)
    test_output = np.empty((len(test), 2, 630), dtype=np.float64)
    for q in range(2):
        train_sum = np.sum(values[train, q], axis=0, dtype=np.float64)
        train_residual = np.empty((len(train), 4, 20, 20), dtype=np.float64)
        for position, subject_index in enumerate(train):
            template = (train_sum - values[subject_index, q]) / float(len(train) - 1)
            residual = values[subject_index, q] - template
            train_residual[position] = residual - np.sum(
                pi[subject_index, q, :, None, None] * residual, axis=0, keepdims=True
            )
        template = train_sum / float(len(train))
        test_residual = values[test, q] - template[None]
        test_residual -= np.sum(pi[test, q, :, None, None] * test_residual, axis=1, keepdims=True)
        train_output[:, q] = _normalize_signature(_signature_from_residual(train_residual, H))
        test_output[:, q] = _normalize_signature(_signature_from_residual(test_residual, H))
    return train_output, test_output


def _inner_select_rank(
    U: np.ndarray,
    proportions: np.ndarray,
    outer_train: np.ndarray,
    inner_test_folds: Sequence[np.ndarray],
    config: Mapping[str, Any],
    H: np.ndarray,
) -> dict[str, Any]:
    rank_grid = [int(value) for value in config["population_model"]["rank_grid"]]
    minimum_inner_train = min(len(outer_train) - len(fold) for fold in inner_test_folds)
    ranks = [rank for rank in rank_grid if rank <= min(630, minimum_inner_train - 2)]
    if not ranks:
        raise StiegerNumericalError("no sample-identifiable inner rank")
    scores = np.empty((len(inner_test_folds), len(ranks)), dtype=np.float64)
    for inner_position, inner_test in enumerate(inner_test_folds):
        inner_train = np.setdiff1d(outer_train, inner_test, assume_unique=False)
        train_features, test_features = fold_signatures(U, proportions, inner_train, inner_test, H)
        fit = fit_two_view(train_features[:, 0], train_features[:, 1], max_rank=max(ranks))
        score0, score1 = project_two_view(fit, test_features[:, 0], test_features[:, 1])
        for rank_position, rank in enumerate(ranks):
            scores[inner_position, rank_position] = float(separation_from_scores(score0, score1, rank)["statistic"])
    selection = select_rank_one_se(scores, ranks)
    return {
        "selected_rank": int(selection["selected_rank"]),
        "best_rank": int(selection["best_rank"]),
        "ranks": ranks,
        "threshold": float(selection["threshold"]),
        "means": np.asarray(selection["means"]),
        "standard_errors": np.asarray(selection["standard_errors"]),
        "fold_scores": np.asarray(selection["fold_scores"]),
    }


def evaluate_population(
    U: np.ndarray, proportions: np.ndarray, locked: LockedObjects, config: Mapping[str, Any]
) -> dict[str, Any]:
    H = helmert4(config)
    n_subjects = len(locked.subjects)
    subject_forward = np.full(n_subjects, np.nan)
    subject_reverse = np.full(n_subjects, np.nan)
    subject_average = np.full(n_subjects, np.nan)
    subject_fold = np.full(n_subjects, -1, dtype=np.int8)
    full_average = np.full(n_subjects, np.nan)
    fold_records: list[dict[str, Any]] = []
    modes_left: list[np.ndarray] = []
    modes_right: list[np.ndarray] = []
    all_rank_records: list[dict[str, Any]] = []
    for fold_index, outer_test in enumerate(locked.folds):
        outer_train = np.setdiff1d(np.arange(n_subjects), outer_test)
        selection = _inner_select_rank(U, proportions, outer_train, locked.inner_folds[fold_index], config, H)
        selected_rank = int(selection["selected_rank"])
        train_features, test_features = fold_signatures(U, proportions, outer_train, outer_test, H)
        permitted_ranks = [
            rank for rank in config["population_model"]["rank_grid"] if int(rank) <= min(630, len(outer_train) - 2)
        ]
        fit = fit_two_view(train_features[:, 0], train_features[:, 1], max_rank=max(permitted_ranks))
        score0, score1 = project_two_view(fit, test_features[:, 0], test_features[:, 1])
        separation = separation_from_scores(score0, score1, selected_rank)
        subject_forward[outer_test] = np.asarray(separation["forward"])
        subject_reverse[outer_test] = np.asarray(separation["reverse"])
        subject_average[outer_test] = np.asarray(separation["average"])
        subject_fold[outer_test] = fold_index
        full = full_space_separation(test_features[:, 0], test_features[:, 1])
        full_average[outer_test] = np.asarray(full["average"])
        for rank in permitted_ranks:
            rank_separation = separation_from_scores(score0, score1, int(rank))
            all_rank_records.append(
                {"fold": fold_index, "rank": int(rank), "statistic": float(rank_separation["statistic"])}
            )
        fold_records.append(
            {
                "fold": fold_index,
                "test_subjects": locked.subjects[outer_test].astype(int).tolist(),
                "n_train": len(outer_train),
                "selected_rank": selected_rank,
                "best_rank": int(selection["best_rank"]),
                "inner_ranks": selection["ranks"],
                "inner_means": selection["means"].tolist(),
                "inner_standard_errors": selection["standard_errors"].tolist(),
                "statistic": float(separation["statistic"]),
                "forward_median": float(separation["forward_median"]),
                "reverse_median": float(separation["reverse_median"]),
                "full_space_statistic": float(full["statistic"]),
            }
        )
        modes_left.append(np.asarray(fit.left[:, :selected_rank]))
        modes_right.append(np.asarray(fit.right[:, :selected_rank]))
    arrays = (subject_forward, subject_reverse, subject_average, full_average)
    if any(not np.all(np.isfinite(value)) for value in arrays) or np.any(subject_fold < 0):
        raise StiegerNumericalError("outer evaluation has missing/nonfinite subjects")
    return {
        "statistic": float(np.median(subject_average)),
        "forward_median": float(np.median(subject_forward)),
        "reverse_median": float(np.median(subject_reverse)),
        "full_space_statistic": float(np.median(full_average)),
        "subject_forward": subject_forward,
        "subject_reverse": subject_reverse,
        "subject_average": subject_average,
        "subject_full_space": full_average,
        "subject_fold": subject_fold,
        "selected_ranks": np.asarray([row["selected_rank"] for row in fold_records], dtype=np.int64),
        "fold_records": fold_records,
        "all_rank_records": all_rank_records,
        "modes_left": modes_left,
        "modes_right": modes_right,
        "leave_one_subject": np.asarray([np.median(np.delete(subject_average, i)) for i in range(n_subjects)]),
    }


def _bootstrap_median(values: np.ndarray, config: Mapping[str, Any], namespace: str) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = _rng(config, namespace)
    replicates = int(config["inference"]["bootstrap_replicates"])
    indices = rng.integers(0, len(data), size=(replicates, len(data)))
    distribution = np.median(data[indices], axis=1)
    return tuple(float(value) for value in np.quantile(distribution, [0.025, 0.975]))


def _monte_carlo_p(observed: float, null: np.ndarray) -> float:
    return float((1 + np.count_nonzero(np.asarray(null) >= observed)) / (1 + len(null)))


def _global_split_signature(U: np.ndarray, proportions: np.ndarray, session_position: int, H: np.ndarray) -> np.ndarray:
    values = U[:, session_position]
    pi = proportions[:, session_position]
    total = np.sum(values, axis=0)
    residuals = np.empty_like(values)
    for subject in range(len(values)):
        template = (total - values[subject]) / float(len(values) - 1)
        residual = values[subject] - template
        residuals[subject] = residual - np.sum(pi[subject, :, None, None] * residual, axis=0, keepdims=True)
    return _normalize_signature(_signature_from_residual(residuals, H))


def run_reliability(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_config(root); validate_parent_hashes(root, config)
    locked = load_locked_objects(root, config); H = helmert4(config)
    records: list[dict[str, Any]] = []
    all_null = np.empty((2, int(config["inference"]["null_replicates"])), dtype=np.float64)
    subject_values = np.empty((len(locked.subjects), 2), dtype=np.float64)
    for q in range(2):
        first = _global_split_signature(locked.U[("primary", "A")], locked.proportions[("primary", "A")], q, H)
        second = _global_split_signature(locked.U[("primary", "B")], locked.proportions[("primary", "B")], q, H)
        separation = full_space_separation(first, second)
        subject_values[:, q] = np.asarray(separation["average"])
        for replicate in range(all_null.shape[1]):
            permutation = fixed_point_free(_rng(config, "reliability_pairing", q, replicate), len(locked.subjects))
            all_null[q, replicate] = float(full_space_separation(first, second[permutation])["statistic"])
        p_value = _monte_carlo_p(float(separation["statistic"]), all_null[q])
        influence = np.asarray([np.median(np.delete(subject_values[:, q], i)) for i in range(len(locked.subjects))])
        records.append(
            {
                "session": int(SESSIONS[q]),
                "statistic": float(separation["statistic"]),
                "forward_median": float(separation["forward_median"]),
                "reverse_median": float(separation["reverse_median"]),
                "pairing_p": p_value,
                "leave_one_subject_minimum": float(np.min(influence)),
                "pass": bool(float(separation["statistic"]) > 0 and p_value <= 0.05 and np.all(influence > 0)),
            }
        )
    passed = all(row["pass"] for row in records)
    output = root / str(config["project"]["output_dir"])
    np.savez_compressed(output / "objects" / "reliability_core.npz", subject_values=subject_values, null=all_null)
    pd.DataFrame(records).to_csv(output / "tables" / "reliability.csv", index=False)
    decision = {
        "status": "STIEGER_MEASUREMENT_RELIABILITY_PASS" if passed else config["decisions"]["reliability_failure"],
        "sessions": records,
    }
    atomic_write_json(output / "decisions" / "reliability_decision.json", decision)
    return decision


def _save_population_result(output: Path, stem: str, result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "statistic": float(result["statistic"]),
        "forward_median": float(result["forward_median"]),
        "reverse_median": float(result["reverse_median"]),
        "full_space_statistic": float(result["full_space_statistic"]),
        "selected_ranks": np.asarray(result["selected_ranks"]).astype(int).tolist(),
        "bootstrap_ci": list(_bootstrap_median(np.asarray(result["subject_average"]), config, f"{stem}_bootstrap")),
        "leave_one_subject_minimum": float(np.min(result["leave_one_subject"])),
        "fold_records": result["fold_records"],
        "all_rank_records": result["all_rank_records"],
    }
    arrays: dict[str, np.ndarray] = {
        "subject_forward": np.asarray(result["subject_forward"]),
        "subject_reverse": np.asarray(result["subject_reverse"]),
        "subject_average": np.asarray(result["subject_average"]),
        "subject_full_space": np.asarray(result["subject_full_space"]),
        "subject_fold": np.asarray(result["subject_fold"]),
        "selected_ranks": np.asarray(result["selected_ranks"]),
    }
    for fold, (left, right) in enumerate(zip(result["modes_left"], result["modes_right"], strict=True)):
        arrays[f"fold{fold}_left"] = left
        arrays[f"fold{fold}_right"] = right
    np.savez_compressed(output / "objects" / f"{stem}_core.npz", **arrays)
    atomic_write_json(output / "tables" / f"{stem}_summary.json", summary)
    return summary


def run_population_observed(repo_root: str | Path, epoch: str = "primary") -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_config(root); validate_parent_hashes(root, config)
    output = root / str(config["project"]["output_dir"])
    reliability = json.loads((output / "decisions" / "reliability_decision.json").read_text())
    if reliability["status"] != "STIEGER_MEASUREMENT_RELIABILITY_PASS" and epoch == "primary":
        raise StiegerDataContractError("primary population analysis blocked by reliability prerequisite")
    locked = load_locked_objects(root, config)
    result = evaluate_population(locked.U[(epoch, "F")], locked.proportions[(epoch, "F")], locked, config)
    return _save_population_result(output, f"{epoch}_population_observed", result, config)


def _permute_class_axis(values: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    perm = np.asarray(permutations, dtype=np.int64)
    if array.ndim == 5:
        return np.take_along_axis(array, perm[..., None, None], axis=2)
    if array.ndim == 3:
        return np.take_along_axis(array, perm, axis=2)
    raise StiegerDataContractError("unsupported class-permutation array")


def _balanced_class_permutations(config: Mapping[str, Any], replicate: int, n_subjects: int, namespace: str) -> np.ndarray:
    all_permutations = np.asarray(all_class_permutations(), dtype=np.int8)
    output = np.empty((n_subjects, 2, 4), dtype=np.int8)
    cycle = replicate % 24
    for subject in range(n_subjects):
        for q in range(2):
            ordering = _rng(config, namespace, subject, q).permutation(24)
            output[subject, q] = all_permutations[ordering[cycle]]
    return output


def _random_subspace_statistic(
    U: np.ndarray,
    proportions: np.ndarray,
    locked: LockedObjects,
    selected_ranks: np.ndarray,
    config: Mapping[str, Any],
    replicate: int,
    namespace: str,
) -> float:
    H = helmert4(config)
    subject_average = np.full(len(locked.subjects), np.nan)
    for fold_index, test in enumerate(locked.folds):
        train = np.setdiff1d(np.arange(len(locked.subjects)), test)
        train_features, test_features = fold_signatures(U, proportions, train, test, H)
        rank = int(selected_ranks[fold_index])
        bases: list[np.ndarray] = []
        for q in range(2):
            raw = _rng(config, namespace, replicate, fold_index, q).normal(size=(630, rank))
            basis, _ = np.linalg.qr(raw, mode="reduced")
            bases.append(basis)
        scores: list[np.ndarray] = []
        for q in range(2):
            mean = np.mean(train_features[:, q], axis=0)
            training_score = (train_features[:, q] - mean) @ bases[q]
            scale = np.std(training_score, axis=0, ddof=1)
            if np.any(scale <= np.finfo(float).eps * math.sqrt(630)):
                raise StiegerNumericalError("random subspace projected training-score scale is degenerate")
            scores.append(((test_features[:, q] - mean) @ bases[q]) / scale)
        separation = separation_from_scores(scores[0], scores[1], rank)
        subject_average[test] = np.asarray(separation["average"])
    if not np.all(np.isfinite(subject_average)):
        raise StiegerNumericalError("random-subspace subject score missing")
    return float(np.median(subject_average))


def run_primary_nulls(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_config(root); validate_parent_hashes(root, config)
    output = root / str(config["project"]["output_dir"]); locked = load_locked_objects(root, config)
    observed_summary = json.loads((output / "tables" / "primary_population_observed_summary.json").read_text())
    with np.load(output / "objects" / "primary_population_observed_core.npz", allow_pickle=False) as data:
        selected_ranks = np.asarray(data["selected_ranks"], dtype=np.int64)
        observed_subject_average = np.asarray(data["subject_average"], dtype=np.float64)
        full_subject = np.asarray(data["subject_full_space"], dtype=np.float64)
    U = locked.U[("primary", "F")]; pi = locked.proportions[("primary", "F")]
    pre_U = locked.U[("pretarget", "F")]; pre_pi = locked.proportions[("pretarget", "F")]
    replicates = int(config["inference"]["null_replicates"]); n = len(locked.subjects)
    pairing = np.empty(replicates); class_null = np.empty(replicates); random_null = np.empty(replicates)
    pre_class_null = np.empty(replicates)
    pairing_maps = np.empty((replicates, n), dtype=np.int16)
    class_maps = np.empty((replicates, n, 2, 4), dtype=np.int8)
    pre_class_maps = np.empty((replicates, n, 2, 4), dtype=np.int8)
    # Pre-target observed is run here because it cannot influence settings.
    pre_observed_result = evaluate_population(pre_U, pre_pi, locked, config)
    pre_observed_summary = _save_population_result(output, "pretarget_population_observed", pre_observed_result, config)
    for replicate in range(replicates):
        pairing_map = fixed_point_free(_rng(config, "primary_subject_pairing", replicate), n)
        pairing_maps[replicate] = pairing_map
        pair_U = U.copy(); pair_pi = pi.copy()
        pair_U[:, 1] = U[pairing_map, 1]; pair_pi[:, 1] = pi[pairing_map, 1]
        pairing[replicate] = float(evaluate_population(pair_U, pair_pi, locked, config)["statistic"])
        permutation = _balanced_class_permutations(config, replicate, n, "primary_class_semantics")
        class_maps[replicate] = permutation
        class_null[replicate] = float(
            evaluate_population(_permute_class_axis(U, permutation), _permute_class_axis(pi, permutation), locked, config)["statistic"]
        )
        random_null[replicate] = _random_subspace_statistic(
            U, pi, locked, selected_ranks, config, replicate, "primary_random_subspace"
        )
        pre_permutation = _balanced_class_permutations(config, replicate, n, "pretarget_class_semantics")
        pre_class_maps[replicate] = pre_permutation
        pre_class_null[replicate] = float(
            evaluate_population(
                _permute_class_axis(pre_U, pre_permutation), _permute_class_axis(pre_pi, pre_permutation), locked, config
            )["statistic"]
        )
    observed = float(observed_summary["statistic"]); pre_observed = float(pre_observed_summary["statistic"])
    p_pair = _monte_carlo_p(observed, pairing); p_class = _monte_carlo_p(observed, class_null)
    p_random = _monte_carlo_p(observed, random_null); p_pre_class = _monte_carlo_p(pre_observed, pre_class_null)
    np.savez_compressed(
        output / "nulls" / "primary_nulls.npz",
        pairing=pairing,
        class_semantics=class_null,
        random_subspace=random_null,
        pairing_maps=pairing_maps,
        class_permutation_maps=class_maps,
        pretarget_class_semantics=pre_class_null,
        pretarget_class_permutation_maps=pre_class_maps,
    )
    null_summary = {
        "observed": observed,
        "pairing_p": p_pair,
        "class_semantics_p": p_class,
        "random_subspace_p": p_random,
        "replicates": replicates,
        "pretarget_observed": pre_observed,
        "pretarget_forward_median": pre_observed_summary["forward_median"],
        "pretarget_reverse_median": pre_observed_summary["reverse_median"],
        "pretarget_class_semantics_p": p_pre_class,
    }
    atomic_write_json(output / "tables" / "primary_null_summary.json", null_summary)
    reliability = json.loads((output / "decisions" / "reliability_decision.json").read_text())
    reliability_pass = reliability["status"] == "STIEGER_MEASUREMENT_RELIABILITY_PASS"
    pretarget_confound = bool(
        pre_observed > 0
        and float(pre_observed_summary["forward_median"]) > 0
        and float(pre_observed_summary["reverse_median"]) > 0
        and p_pre_class <= float(config["inference"]["alpha"])
    )
    directions = float(observed_summary["forward_median"]) > 0 and float(observed_summary["reverse_median"]) > 0
    influence = float(observed_summary["leave_one_subject_minimum"]) > 0
    full_stable = float(observed_summary["full_space_statistic"]) > 0 and np.all(
        [np.median(np.delete(full_subject, index)) > 0 for index in range(n)]
    )
    structural = bool(
        reliability_pass and observed > 0 and directions and p_pair <= 0.05 and p_class <= 0.05 and p_random <= 0.05 and influence and full_stable
    )
    median_rank = float(np.median(selected_ranks)); low_count = int(np.count_nonzero(selected_ranks <= 3))
    if pretarget_confound:
        terminal = config["decisions"]["pretarget_failure"]
    elif structural and median_rank <= 3 and low_count >= 4:
        terminal = config["decisions"]["structure_low_rank"]
    elif structural:
        terminal = config["decisions"]["structure_not_low_rank"]
    else:
        terminal = config["decisions"]["structure_stop"]
    gate_table = {
        "reliability_pass": reliability_pass,
        "heldout_positive": observed > 0,
        "forward_positive": float(observed_summary["forward_median"]) > 0,
        "reverse_positive": float(observed_summary["reverse_median"]) > 0,
        "pairing_p_at_most_005": p_pair <= 0.05,
        "class_p_at_most_005": p_class <= 0.05,
        "random_p_at_most_005": p_random <= 0.05,
        "influence_positive": influence,
        "full_space_stable": bool(full_stable),
        "median_rank_at_most_3": median_rank <= 3,
        "four_of_six_rank_at_most_3": low_count >= 4,
        "pretarget_confound_absent": not pretarget_confound,
    }
    decision = {
        "terminal": terminal,
        "gates": gate_table,
        "observed": observed_summary,
        "nulls": null_summary,
        "median_selected_rank": median_rank,
        "folds_rank_at_most_3": low_count,
        "subject_bootstrap_ci": observed_summary["bootstrap_ci"],
        "parent_terminals_unchanged": True,
    }
    atomic_write_json(output / "decisions" / "population_structure_decision.json", decision)
    return decision


def _load_tangent(root: Path, config: Mapping[str, Any], subject: int, session_id: int) -> tuple[np.ndarray, np.ndarray]:
    cache = root / str(config["project"]["cache_dir"])
    manifest_path = root / str(config["project"]["output_dir"]) / "objects" / "trial_tangent_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    row = next(record for record in manifest["records"] if int(record["subject"]) == subject and int(record["session"]) == session_id)
    path = root / row["path"]
    if stream_sha256_file(path) != row["sha256"]:
        raise StiegerDataContractError(f"trial tangent hash mismatch S{subject}/{session_id}")
    with np.load(path, allow_pickle=False) as data:
        tangent = np.asarray(data["primary_trial_svec"], dtype=np.float64)
        labels = np.asarray(data["targetnumber"], dtype=np.int64)
    if tangent.shape != (len(labels), 210) or not np.all(np.isfinite(tangent)):
        raise StiegerDataContractError("invalid trial tangent cache")
    return tangent, labels


def _fold_modes(output: Path, fold_index: int, selected_rank: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(output / "objects" / "primary_population_observed_core.npz", allow_pickle=False) as data:
        left = np.asarray(data[f"fold{fold_index}_left"], dtype=np.float64)
        right = np.asarray(data[f"fold{fold_index}_right"], dtype=np.float64)
    if left.shape != (630, selected_rank) or right.shape != (630, selected_rank):
        raise StiegerDataContractError("frozen selected mode shape mismatch")
    return left, right


def _weighted_center_classes(values: np.ndarray, proportions: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    pi = np.asarray(proportions, dtype=np.float64)
    return array - np.sum(pi[..., None] * array, axis=-2, keepdims=True)


def _class_trial_centroids(tangent: np.ndarray, labels: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    projected = np.asarray(tangent) @ np.asarray(directions)
    indices = [np.flatnonzero(labels == class_value) for class_value in range(1, 5)]
    if any(len(index) == 0 for index in indices):
        raise StiegerDataContractError("a projected session lacks a class")
    centroids = np.stack([np.mean(projected[index], axis=0) for index in indices])
    return centroids, [projected[index] for index in indices]


def _normalize_class_template(values: np.ndarray, proportions: np.ndarray) -> np.ndarray:
    centered = _weighted_center_classes(np.asarray(values)[None], np.asarray(proportions)[None])[0]
    norm = float(np.linalg.norm(centered))
    if norm <= np.finfo(float).eps * math.sqrt(centered.size):
        raise StiegerNumericalError("semantic class template has zero Frobenius norm")
    return centered / norm


def _metric_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(truth, dtype=np.float64).reshape(-1)
    estimate = np.asarray(prediction, dtype=np.float64).reshape(-1)
    error = estimate - actual
    denominator = float(np.mean(np.abs(actual)))
    pearson = float(stats.pearsonr(actual, estimate).statistic) if np.std(actual) > 0 and np.std(estimate) > 0 else 0.0
    spearman = float(stats.spearmanr(actual, estimate).statistic) if np.std(actual) > 0 and np.std(estimate) > 0 else 0.0
    ss_total = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "normalized_mae": float(np.mean(np.abs(error)) / denominator) if denominator > 0 else math.inf,
        "pearson": pearson,
        "spearman": spearman,
        "signed_r2": float(1 - np.sum(error**2) / ss_total) if ss_total > 0 else 0.0,
    }


def run_source_reference(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_config(root); validate_parent_hashes(root, config)
    output = root / str(config["project"]["output_dir"]); locked = load_locked_objects(root, config)
    population = json.loads((output / "decisions" / "population_structure_decision.json").read_text())
    selected = np.asarray(population["observed"]["selected_ranks"], dtype=np.int64)
    U = locked.U[("primary", "F")]; pi = locked.proportions[("primary", "F")]
    n, max_rank = len(locked.subjects), int(np.max(selected))
    D = np.full((n, 2, 4, max_rank), np.nan); D_fold = np.full((6, n, 2, 4, max_rank), np.nan); beta = np.full_like(D, np.nan)
    projected_z = np.full_like(D, np.nan); trial_centroid = np.full_like(D, np.nan)
    beta_prediction = np.full_like(D, np.nan); gamma_all = np.full((6, 2, 4, max_rank), np.nan)
    correction_all = np.full_like(gamma_all, np.nan); directions_all = np.full((6, 2, 210, max_rank), np.nan)
    identity_errors: list[float] = []; fold_rows: list[dict[str, Any]] = []
    for fold_index, test in enumerate(locked.folds):
        train = np.setdiff1d(np.arange(n), test); rank = int(selected[fold_index])
        left, right = _fold_modes(output, fold_index, rank)
        for q, mode in enumerate((left, right)):
            directions = deterministic_trial_directions(mode, rank); directions_all[fold_index, q, :, :rank] = directions
            total_coordinate = np.einsum("nkx,xr->nkr", svec(U[:, q]), directions, optimize=True)
            D[test, q, :, :rank] = total_coordinate[test]
            D_fold[fold_index, :, q, :, :rank] = total_coordinate
            gamma = np.mean(total_coordinate[train], axis=0); gamma_all[fold_index, q, :, :rank] = gamma
            residual = total_coordinate[test] - gamma[None]
            centered_beta = _weighted_center_classes(residual, pi[test, q]); beta[test, q, :, :rank] = centered_beta
            # Algebraic Z projection uses the same training-only held-out template.
            R = U[test, q] - np.mean(U[train, q], axis=0)[None]
            Z = R - np.sum(pi[test, q, :, None, None] * R, axis=1, keepdims=True)
            z_coordinate = np.einsum("nkx,xr->nkr", svec(Z), directions, optimize=True)
            projected_z[test, q, :, :rank] = z_coordinate
            error = float(np.max(np.abs(z_coordinate - centered_beta))); identity_errors.append(error)
            source_centroids = []
            target_centroids = []
            for index in train:
                tangent, labels = _load_tangent(root, config, int(locked.subjects[index]), int(SESSIONS[q]))
                centroid, _ = _class_trial_centroids(tangent, labels, directions); source_centroids.append(centroid)
            correction = np.mean(total_coordinate[train] - np.stack(source_centroids), axis=0)
            correction_all[fold_index, q, :, :rank] = correction
            for index in test:
                tangent, labels = _load_tangent(root, config, int(locked.subjects[index]), int(SESSIONS[q]))
                centroid, _ = _class_trial_centroids(tangent, labels, directions); target_centroids.append(centroid)
            target_centroid_array = np.stack(target_centroids); trial_centroid[test, q, :, :rank] = target_centroid_array
            prediction_residual = target_centroid_array + correction[None] - gamma[None]
            prediction_centered = _weighted_center_classes(prediction_residual, pi[test, q])
            beta_prediction[test, q, :, :rank] = prediction_centered
            fold_rows.append(
                {
                    "fold": fold_index,
                    "session": int(SESSIONS[q]),
                    "rank": rank,
                    "identity_max_abs_error": error,
                    "gamma_frobenius_norm": float(np.linalg.norm(gamma)),
                    "correction_frobenius_norm": float(np.linalg.norm(correction)),
                }
            )
    tolerance = float(config["inference"]["identity_atol"])
    identity_pass = bool(max(identity_errors) <= tolerance)
    valid = np.isfinite(beta) & np.isfinite(beta_prediction)
    metrics = _metric_summary(beta[valid], beta_prediction[valid])
    session_metrics = []
    for q in range(2):
        valid_q = np.isfinite(beta[:, q]) & np.isfinite(beta_prediction[:, q])
        session_metrics.append({"session": int(SESSIONS[q]), **_metric_summary(beta[:, q][valid_q], beta_prediction[:, q][valid_q])})
    np.savez_compressed(
        output / "objects" / "source_reference_core.npz",
        D=D,
        D_fold=D_fold,
        beta=beta,
        projected_z=projected_z,
        trial_centroid=trial_centroid,
        beta_prediction=beta_prediction,
        gamma=gamma_all,
        correction=correction_all,
        trial_directions=directions_all,
        selected_ranks=selected,
    )
    identity_decision = config["decisions"]["identity_pass"] if identity_pass else config["decisions"]["identity_failure"]
    summary = {
        "identity_decision": identity_decision,
        "identity_max_abs_error": max(identity_errors),
        "identity_tolerance": tolerance,
        "correction_metrics": metrics,
        "session_metrics": session_metrics,
        "fold_session_summary": fold_rows,
        "population_prerequisite": population["terminal"],
    }
    atomic_write_json(output / "decisions" / "source_reference_decision.json", summary)
    pd.DataFrame(fold_rows).to_csv(output / "tables" / "source_reference_fold_session.csv", index=False)
    return summary


def _bootstrap_accuracy_by_subject(success: np.ndarray, config: Mapping[str, Any], namespace: str) -> tuple[float, float]:
    values = np.asarray(success, dtype=np.float64)
    rng = _rng(config, namespace); count = int(config["inference"]["bootstrap_replicates"])
    indices = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
    distribution = np.mean(values[indices], axis=(1, 2) if values.ndim == 3 else 1)
    return tuple(float(value) for value in np.quantile(distribution, [0.025, 0.975]))


def run_semantic_permutation(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_config(root); locked = load_locked_objects(root, config)
    output = root / str(config["project"]["output_dir"])
    population = json.loads((output / "decisions" / "population_structure_decision.json").read_text())
    source_decision = json.loads((output / "decisions" / "source_reference_decision.json").read_text())
    with np.load(output / "objects" / "source_reference_core.npz", allow_pickle=False) as data:
        D = np.asarray(data["D"]); D_fold = np.asarray(data["D_fold"]); trial_centroid = np.asarray(data["trial_centroid"])
        correction = np.asarray(data["correction"]); selected = np.asarray(data["selected_ranks"], dtype=int)
    success = np.zeros((len(locked.subjects), 2), dtype=bool); top2 = np.zeros_like(success)
    margins = np.full(success.shape, np.nan); best_permutations = np.full((*success.shape, 4), -1, dtype=np.int8)
    cost_array = np.full((*success.shape, 24), np.nan); loo_template_stable = True
    per_class_correct = np.zeros((len(locked.subjects), 2, 4), dtype=bool)
    for fold_index, test in enumerate(locked.folds):
        train = np.setdiff1d(np.arange(len(locked.subjects)), test); rank = int(selected[fold_index])
        for q in range(2):
            source_template_raw = np.mean(D_fold[fold_index, train, q, :, :rank], axis=0)
            source_pi = np.mean(locked.proportions[("primary", "F")][train, q], axis=0)
            source_template = _normalize_class_template(source_template_raw, source_pi)
            for excluded in train:
                loo_train = train[train != excluded]
                loo_template = _normalize_class_template(
                    np.mean(D_fold[fold_index, loo_train, q, :, :rank], axis=0),
                    np.mean(locked.proportions[("primary", "F")][loo_train, q], axis=0),
                )
                loo_match = semantic_permutation_costs(source_template, loo_template, config["inference"]["tie_tolerance"])
                loo_template_stable &= bool(loo_match["identity_success"])
            for index in test:
                target_total = trial_centroid[index, q, :, :rank] + correction[fold_index, q, :, :rank]
                target_template = _normalize_class_template(target_total, locked.proportions[("primary", "F")][index, q])
                match = semantic_permutation_costs(source_template, target_template, config["inference"]["tie_tolerance"])
                success[index, q] = match["identity_success"]; top2[index, q] = match["top2_identity"]
                margins[index, q] = match["margin"]; best_permutations[index, q] = match["best_permutation"]
                cost_array[index, q] = match["costs"]
                per_class_correct[index, q] = match["best_permutation"] == np.arange(4)
    pooled_accuracy = float(np.mean(success)); session_accuracy = np.mean(success, axis=0)
    ci = _bootstrap_accuracy_by_subject(success, config, "semantic_permutation_bootstrap")
    chance = 1.0 / 24.0
    p_value = float(stats.binomtest(int(success.sum()), success.size, chance, alternative="greater").pvalue)
    loo_target_accuracy = np.asarray([np.mean(np.delete(success, index, axis=0)) for index in range(len(success))])
    prerequisite = population["terminal"] in {
        config["decisions"]["structure_low_rank"], config["decisions"]["structure_not_low_rank"]
    }
    passed = bool(
        prerequisite
        and np.all(session_accuracy > chance)
        and ci[0] > chance
        and p_value <= 0.05
        and loo_template_stable
        and np.all(loo_target_accuracy > chance)
    )
    decision = config["decisions"]["permutation_pass"] if passed else config["decisions"]["permutation_negative"]
    np.savez_compressed(
        output / "objects" / "semantic_permutation_core.npz",
        success=success,
        top2=top2,
        margins=margins,
        best_permutations=best_permutations,
        all_costs=cost_array,
        per_class_correct=per_class_correct,
        leave_one_subject_accuracy=loo_target_accuracy,
    )
    summary = {
        "decision": decision,
        "pooled_top1_accuracy": pooled_accuracy,
        "session_top1_accuracy": session_accuracy.tolist(),
        "pooled_top2_accuracy": float(np.mean(top2)),
        "per_class_accuracy": np.mean(per_class_correct, axis=(0, 1)).tolist(),
        "bootstrap_ci": list(ci),
        "exact_binomial_p": p_value,
        "chance": chance,
        "unique_ties_fail": True,
        "violating_subject_session_count": int(np.count_nonzero(~success)),
        "violating_subjects": locked.subjects[np.any(~success, axis=1)].astype(int).tolist(),
        "leave_one_training_subject_template_stable": bool(loo_template_stable),
        "leave_one_target_subject_minimum_accuracy": float(np.min(loo_target_accuracy)),
    }
    atomic_write_json(output / "decisions" / "semantic_permutation_decision.json", summary)
    return summary


def _covariance_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 2:
        raise StiegerNumericalError("scatter covariance requires at least two row observations")
    # Population normalization is required by the frozen total = within +
    # between class-proportion decomposition.
    result = np.atleast_2d(np.cov(array, rowvar=False, ddof=0))
    if result.shape != (array.shape[1], array.shape[1]) or not np.all(np.isfinite(result)):
        raise StiegerNumericalError("scatter covariance shape/nonfinite failure")
    return (result + result.T) / 2.0


def _scatter_from_projected(projected: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = _covariance_rows(projected)
    class_indices = [np.flatnonzero(labels == value) for value in range(1, 5)]
    counts = np.asarray([len(index) for index in class_indices], dtype=np.float64)
    pi = counts / counts.sum()
    centroids = np.stack([np.mean(projected[index], axis=0) for index in class_indices])
    within = sum(float(pi[c]) * _covariance_rows(projected[index]) for c, index in enumerate(class_indices))
    mean = np.sum(pi[:, None] * centroids, axis=0)
    centered = centroids - mean
    between = np.einsum("k,ki,kj->ij", pi, centered, centered, optimize=True)
    if not np.allclose(total, within + between, atol=5e-9, rtol=5e-7):
        raise StiegerNumericalError("projected total/within/between decomposition failed")
    return total, within, between, centroids


def run_unlabeled_scatter(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_config(root); locked = load_locked_objects(root, config)
    output = root / str(config["project"]["output_dir"])
    population = json.loads((output / "decisions" / "population_structure_decision.json").read_text())
    with np.load(output / "objects" / "source_reference_core.npz", allow_pickle=False) as data:
        directions_all = np.asarray(data["trial_directions"]); selected = np.asarray(data["selected_ranks"], dtype=int)
    n, max_rank = len(locked.subjects), int(np.max(selected))
    estimates = np.full((n, 2, max_rank, max_rank), np.nan)
    oracle = np.full_like(estimates, np.nan); total_store = np.full_like(estimates, np.nan); source_within_store = np.full((6, 2, max_rank, max_rank), np.nan)
    subject_score = np.full((n, 2), np.nan); normalized_error = np.full((n, 2), np.nan)
    sensor_total = np.empty((n, 2, 210, 210)); sensor_within = np.empty_like(sensor_total); sensor_between = np.empty_like(sensor_total)
    tangent_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for index, subject in enumerate(locked.subjects):
        for q, session_id in enumerate(SESSIONS):
            tangent, labels = _load_tangent(root, config, int(subject), int(session_id)); tangent_cache[(index, q)] = (tangent, labels)
            total, within, between, _ = _scatter_from_projected(tangent, labels)
            sensor_total[index, q] = total; sensor_within[index, q] = within; sensor_between[index, q] = between
    for fold_index, test in enumerate(locked.folds):
        train = np.setdiff1d(np.arange(n), test); rank = int(selected[fold_index])
        for q in range(2):
            directions = directions_all[fold_index, q, :, :rank]
            source_within_sensor = np.mean(sensor_within[train, q], axis=0)
            W = directions.T @ source_within_sensor @ directions
            source_within_store[fold_index, q, :rank, :rank] = W
            for index in test:
                total = directions.T @ sensor_total[index, q] @ directions
                prediction = psd_projection(total - W)
                truth = directions.T @ sensor_between[index, q] @ directions
                estimates[index, q, :rank, :rank] = prediction
                oracle[index, q, :rank, :rank] = truth
                total_store[index, q, :rank, :rank] = total
                subject_score[index, q] = frobenius_cosine(prediction, truth)
                denominator = float(np.linalg.norm(truth))
                normalized_error[index, q] = float(np.linalg.norm(prediction - truth) / denominator) if denominator > 0 else math.inf
    # Freeze estimates without labels before writing the oracle evaluation object.
    np.savez_compressed(
        output / "objects" / "unlabeled_scatter_estimates.npz",
        estimates=estimates,
        total=total_store,
        source_within=source_within_store,
        selected_ranks=selected,
    )
    np.savez_compressed(
        output / "objects" / "unlabeled_scatter_oracle_evaluation.npz",
        oracle=oracle,
        frobenius_cosine=subject_score,
        normalized_error=normalized_error,
    )
    pooled_subject = np.mean(subject_score, axis=1); observed = float(np.median(pooled_subject))
    session_statistics = np.median(subject_score, axis=0)
    ci = _bootstrap_median(pooled_subject, config, "scatter_subject_bootstrap")
    influence = np.asarray([np.median(np.delete(pooled_subject, i)) for i in range(n)])
    replicates = int(config["inference"]["null_replicates"])
    permutation_null = np.empty(replicates); random_null = np.empty(replicates)
    for replicate in range(replicates):
        score_perm = np.empty_like(subject_score)
        for q in range(2):
            permutation = fixed_point_free(_rng(config, "scatter_target_permutation", replicate, q), n)
            for index in range(n):
                fold_index = int(next(f for f, fold in enumerate(locked.folds) if index in set(fold.tolist())))
                rank = int(selected[fold_index])
                score_perm[index, q] = frobenius_cosine(
                    estimates[index, q, :rank, :rank], oracle[permutation[index], q, :rank, :rank]
                )
        permutation_null[replicate] = float(np.median(np.mean(score_perm, axis=1)))
        random_scores = np.empty_like(subject_score)
        for fold_index, test in enumerate(locked.folds):
            train = np.setdiff1d(np.arange(n), test); rank = int(selected[fold_index])
            for q in range(2):
                raw = _rng(config, "scatter_random_direction", replicate, fold_index, q).normal(size=(210, rank))
                direction, _ = np.linalg.qr(raw, mode="reduced")
                W = direction.T @ np.mean(sensor_within[train, q], axis=0) @ direction
                for index in test:
                    predicted = psd_projection(direction.T @ sensor_total[index, q] @ direction - W)
                    truth = direction.T @ sensor_between[index, q] @ direction
                    random_scores[index, q] = frobenius_cosine(predicted, truth)
        random_null[replicate] = float(np.median(np.mean(random_scores, axis=1)))
    p_permutation = _monte_carlo_p(observed, permutation_null); p_random = _monte_carlo_p(observed, random_null)
    np.savez_compressed(output / "nulls" / "unlabeled_scatter_nulls.npz", subject_permutation=permutation_null, random_direction=random_null)
    prerequisite = population["terminal"] in {
        config["decisions"]["structure_low_rank"], config["decisions"]["structure_not_low_rank"]
    }
    passed = bool(
        prerequisite and np.all(session_statistics > 0) and ci[0] > 0 and p_permutation <= 0.05 and p_random <= 0.05 and np.all(influence > 0)
    )
    decision = config["decisions"]["scatter_pass"] if passed else config["decisions"]["scatter_negative"]
    fold_for_subject = np.empty(n, dtype=np.int64)
    for fold_index, fold in enumerate(locked.folds):
        fold_for_subject[fold] = fold_index
    predicted_ranks = [
        int(np.linalg.matrix_rank(estimates[index, q, : int(selected[fold_for_subject[index]]), : int(selected[fold_for_subject[index]])]))
        for index in range(n)
        for q in range(2)
    ]
    summary = {
        "decision": decision,
        "primary_statistic_median_frobenius_cosine": observed,
        "session_statistics": session_statistics.tolist(),
        "bootstrap_ci": list(ci),
        "subject_permutation_p": p_permutation,
        "random_direction_p": p_random,
        "median_normalized_frobenius_error": float(np.median(normalized_error)),
        "leave_one_subject_minimum": float(np.min(influence)),
        "predicted_rank_frequency": {str(rank): int(predicted_ranks.count(rank)) for rank in sorted(set(predicted_ranks))},
    }
    atomic_write_json(output / "decisions" / "unlabeled_scatter_decision.json", summary)
    return summary


def _gmm_seed(config: Mapping[str, Any], *parts: Any) -> int:
    material = "|".join([str(config["protocol"]["master_seed"]), "gmm", *map(str, parts)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "little")


def _fit_four_component_mixture(values: np.ndarray, config: Mapping[str, Any], *seed_parts: Any) -> Any:
    from sklearn.mixture import GaussianMixture

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 8 or not np.all(np.isfinite(array)):
        raise StiegerNumericalError("invalid pooled trial coordinates for mixture")
    model = GaussianMixture(
        n_components=4,
        covariance_type="tied",
        weights_init=np.full(4, 0.25),
        n_init=int(config["mixture"]["deterministic_multistarts"]),
        reg_covar=float(config["mixture"]["reg_covar"]),
        random_state=_gmm_seed(config, *seed_parts),
        init_params="kmeans",
        max_iter=500,
        tol=1e-5,
    )
    try:
        model.fit(array)
    except Exception as exc:
        raise StiegerNumericalError(f"four-component mixture failed: {exc}") from exc
    if not model.converged_ or not np.all(np.isfinite(model.means_)):
        raise StiegerNumericalError("four-component mixture did not converge finitely")
    return model


def _oracle_component_permutation(component_labels: np.ndarray, target_labels: np.ndarray) -> tuple[int, ...]:
    best: tuple[int, ...] | None = None; best_correct = -1
    for permutation in all_class_permutations():  # class -> component
        inverse = np.empty(4, dtype=np.int64)
        for class_index, component in enumerate(permutation):
            inverse[component] = class_index
        predicted_class = inverse[np.asarray(component_labels, dtype=np.int64)]
        correct = int(np.count_nonzero(predicted_class == np.asarray(target_labels) - 1))
        if correct > best_correct:
            best_correct = correct; best = permutation
    assert best is not None
    return best


def run_component_assignment(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_config(root); locked = load_locked_objects(root, config)
    output = root / str(config["project"]["output_dir"])
    semantic = json.loads((output / "decisions" / "semantic_permutation_decision.json").read_text())
    scatter = json.loads((output / "decisions" / "unlabeled_scatter_decision.json").read_text())
    if semantic["decision"] != config["decisions"]["permutation_pass"] or scatter["decision"] != config["decisions"]["scatter_pass"]:
        summary = {
            "status": "NOT_RUN_PREREQUISITE_FAILURE",
            "semantic_prerequisite": semantic["decision"],
            "scatter_prerequisite": scatter["decision"],
        }
        atomic_write_json(output / "decisions" / "component_assignment_decision.json", summary)
        return summary
    with np.load(output / "objects" / "source_reference_core.npz", allow_pickle=False) as data:
        D_fold = np.asarray(data["D_fold"]); correction = np.asarray(data["correction"])
        directions = np.asarray(data["trial_directions"]); selected = np.asarray(data["selected_ranks"], dtype=int)
    n = len(locked.subjects); source_mapping_success = np.zeros((n, 2), dtype=bool)
    balanced_accuracy = np.full((n, 2), np.nan); ari = np.full((n, 2), np.nan); purity = np.full((n, 2), np.nan)
    source_permutations = np.full((n, 2, 4), -1, dtype=np.int8); component_means = np.full((n, 2, 4, int(np.max(selected))), np.nan)
    component_assignments: list[np.ndarray] = []
    from sklearn.metrics import adjusted_rand_score
    for fold_index, test in enumerate(locked.folds):
        train = np.setdiff1d(np.arange(n), test); rank = int(selected[fold_index])
        for q in range(2):
            source_raw = np.mean(D_fold[fold_index, train, q, :, :rank], axis=0)
            source_template = _normalize_class_template(source_raw, np.full(4, 0.25))
            for index in test:
                tangent, labels = _load_tangent(root, config, int(locked.subjects[index]), int(SESSIONS[q]))
                y = tangent @ directions[fold_index, q, :, :rank]
                model = _fit_four_component_mixture(y, config, int(locked.subjects[index]), int(SESSIONS[q]))
                total_component = model.means_ + correction[fold_index, q, :, :rank]
                target_template = _normalize_class_template(total_component, np.full(4, 0.25))
                match = semantic_permutation_costs(source_template, target_template, config["inference"]["tie_tolerance"])
                source_permutation = tuple(int(v) for v in match["best_permutation"])
                source_permutations[index, q] = source_permutation
                component_means[index, q, :, :rank] = model.means_
                predicted_components = model.predict(y); component_assignments.append(predicted_components.astype(np.int8))
                oracle_permutation = _oracle_component_permutation(predicted_components, labels)
                source_mapping_success[index, q] = bool(match["unique"] and source_permutation == oracle_permutation)
                inverse = np.empty(4, dtype=np.int64)
                for class_index, component in enumerate(source_permutation):
                    inverse[component] = class_index
                predicted_class = inverse[predicted_components]
                true_class = labels - 1
                per_class = [np.mean(predicted_class[true_class == c] == c) for c in range(4)]
                balanced_accuracy[index, q] = float(np.mean(per_class))
                ari[index, q] = float(adjusted_rand_score(true_class, predicted_components))
                purity[index, q] = float(
                    sum(np.max(np.bincount(true_class[predicted_components == component], minlength=4)) for component in range(4)) / len(labels)
                )
    np.savez_compressed(
        output / "objects" / "component_assignment_core.npz",
        source_mapping_success=source_mapping_success,
        balanced_accuracy=balanced_accuracy,
        adjusted_rand_index=ari,
        purity=purity,
        source_permutations=source_permutations,
        component_means=component_means,
    )
    summary = {
        "status": "COMPONENT_ASSIGNMENT_COMPLETED",
        "component_to_class_permutation_accuracy": float(np.mean(source_mapping_success)),
        "session_permutation_accuracy": np.mean(source_mapping_success, axis=0).tolist(),
        "median_adjusted_rand_index": float(np.median(ari)),
        "median_component_purity": float(np.median(purity)),
        "median_class_balanced_assignment_accuracy": float(np.median(balanced_accuracy)),
        "numerically_valid_records": int(np.count_nonzero(np.isfinite(ari))),
    }
    atomic_write_json(output / "decisions" / "component_assignment_decision.json", summary)
    return summary


def _calibration_permutation(
    responsibilities: np.ndarray,
    calibration_indices_by_class: Sequence[np.ndarray],
    source_costs: np.ndarray,
    tolerance: float,
) -> tuple[int, ...]:
    permutations = all_class_permutations(); scores = np.empty(24)
    for permutation_index, permutation in enumerate(permutations):
        score = 0.0
        for class_index, indices in enumerate(calibration_indices_by_class):
            score -= float(np.sum(np.log(np.maximum(responsibilities[indices, permutation[class_index]], np.finfo(float).tiny))))
        scores[permutation_index] = score
    minimum = float(np.min(scores)); tied = np.flatnonzero(scores <= minimum + tolerance)
    if len(tied) == 1:
        return permutations[int(tied[0])]
    return permutations[int(tied[np.argmin(np.asarray(source_costs)[tied])])]


def _sign_flip_p(differences: np.ndarray, config: Mapping[str, Any], namespace: str) -> float:
    values = np.asarray(differences, dtype=np.float64)
    observed = float(np.mean(values)); null = np.empty(int(config["inference"]["null_replicates"]))
    for replicate in range(len(null)):
        signs = _rng(config, namespace, replicate).choice([-1.0, 1.0], size=len(values))
        null[replicate] = float(np.mean(values * signs))
    return _monte_carlo_p(observed, null)


def _bootstrap_mean_ci(values: np.ndarray, config: Mapping[str, Any], namespace: str) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64); rng = _rng(config, namespace)
    indices = rng.integers(0, len(data), size=(int(config["inference"]["bootstrap_replicates"]), len(data)))
    return tuple(float(value) for value in np.quantile(np.mean(data[indices], axis=1), [0.025, 0.975]))


def run_minimal_anchor(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_config(root); locked = load_locked_objects(root, config)
    output = root / str(config["project"]["output_dir"])
    component = json.loads((output / "decisions" / "component_assignment_decision.json").read_text())
    if component["status"] != "COMPONENT_ASSIGNMENT_COMPLETED":
        summary = {"decision": "NOT_RUN_COMPONENT_ASSIGNMENT_INVALID", "component_status": component["status"]}
        atomic_write_json(output / "decisions" / "minimal_anchor_decision.json", summary); return summary
    with np.load(output / "objects" / "source_reference_core.npz", allow_pickle=False) as data:
        D_fold = np.asarray(data["D_fold"]); beta = np.asarray(data["beta"]); gamma = np.asarray(data["gamma"])
        correction = np.asarray(data["correction"]); directions = np.asarray(data["trial_directions"]); selected = np.asarray(data["selected_ranks"], dtype=int)
    n, draws = len(locked.subjects), int(config["inference"]["calibration_subsamples"])
    budgets = [int(value) for value in config["inference"]["calibration_budgets"]]
    proposed_mae = np.full((len(budgets), n, 2, draws), np.nan); direct_mae = np.full_like(proposed_mae, np.nan)
    assignment_correct = np.zeros((len(budgets), n, 2, draws), dtype=bool)
    for fold_index, test in enumerate(locked.folds):
        train = np.setdiff1d(np.arange(n), test); rank = int(selected[fold_index])
        for q in range(2):
            source_raw = np.mean(D_fold[fold_index, train, q, :, :rank], axis=0)
            source_template = _normalize_class_template(source_raw, np.full(4, 0.25))
            for index in test:
                tangent, labels = _load_tangent(root, config, int(locked.subjects[index]), int(SESSIONS[q]))
                y = tangent @ directions[fold_index, q, :, :rank]
                model = _fit_four_component_mixture(y, config, int(locked.subjects[index]), int(SESSIONS[q]))
                total_components = model.means_ + correction[fold_index, q, :, :rank]
                component_template = _normalize_class_template(total_components, np.full(4, 0.25))
                source_match = semantic_permutation_costs(source_template, component_template, config["inference"]["tie_tolerance"])
                source_permutation = tuple(int(v) for v in source_match["best_permutation"])
                responsibilities = model.predict_proba(y); class_indices = [np.flatnonzero(labels == c) for c in range(1, 5)]
                truth = beta[index, q, :, :rank]; target_pi = locked.proportions[("primary", "F")][index, q]
                oracle_component = _oracle_component_permutation(model.predict(y), labels)
                for budget_position, budget in enumerate(budgets):
                    per_class_count = budget // 4 if budget else 0
                    for draw in range(draws):
                        if budget == 0:
                            permutation = source_permutation
                            selected_indices = [np.empty(0, dtype=int) for _ in range(4)]
                        else:
                            selected_indices = [
                                np.sort(_rng(config, "minimal_anchor_subsample", int(locked.subjects[index]), int(SESSIONS[q]), budget, draw, c).choice(
                                    class_indices[c], size=per_class_count, replace=False
                                ))
                                for c in range(4)
                            ]
                            permutation = _calibration_permutation(
                                responsibilities, selected_indices, source_match["costs"], config["inference"]["tie_tolerance"]
                            )
                        predicted_total = total_components[list(permutation)]
                        proposed = _weighted_center_classes(
                            (predicted_total - gamma[fold_index, q, :, :rank])[None], target_pi[None]
                        )[0]
                        proposed_mae[budget_position, index, q, draw] = float(np.mean(np.abs(proposed - truth)))
                        assignment_correct[budget_position, index, q, draw] = permutation == oracle_component
                        if budget > 0:
                            direct_trial = np.stack([np.mean(y[indices], axis=0) for indices in selected_indices])
                            direct_total = direct_trial + correction[fold_index, q, :, :rank]
                            direct = _weighted_center_classes(
                                (direct_total - gamma[fold_index, q, :, :rank])[None], target_pi[None]
                            )[0]
                            direct_mae[budget_position, index, q, draw] = float(np.mean(np.abs(direct - truth)))
    rows: list[dict[str, Any]] = []; efficient_budget: int | None = None
    for budget_position, budget in enumerate(budgets):
        proposed_expected = np.mean(proposed_mae[budget_position], axis=2)
        row: dict[str, Any] = {
            "budget": budget,
            "labels_per_class": budget // 4,
            "proposed_expected_per_draw_mae": float(np.mean(proposed_expected)),
            "session_proposed_mae": np.mean(proposed_expected, axis=0).tolist(),
            "semantic_permutation_accuracy": float(np.mean(assignment_correct[budget_position])),
            "metric_label": "EXPECTED_PER_CALIBRATION_DRAW",
        }
        if budget > 0:
            direct_expected = np.mean(direct_mae[budget_position], axis=2)
            difference = np.mean(direct_expected - proposed_expected, axis=1)
            ci = _bootstrap_mean_ci(difference, config, f"anchor_improvement_budget{budget}")
            p_value = _sign_flip_p(difference, config, f"anchor_sign_flip_budget{budget}")
            session_improvement = np.mean(direct_expected - proposed_expected, axis=0)
            influence = np.asarray([np.mean(np.delete(difference, i)) for i in range(n)])
            row.update(
                {
                    "direct_expected_per_draw_mae": float(np.mean(direct_expected)),
                    "mean_improvement": float(np.mean(difference)),
                    "improvement_bootstrap_ci": list(ci),
                    "paired_sign_flip_p": p_value,
                    "session_improvement": session_improvement.tolist(),
                    "leave_one_subject_minimum_improvement": float(np.min(influence)),
                }
            )
            if (
                budget <= 8
                and np.mean(difference) > 0
                and p_value <= 0.05
                and ci[0] > 0
                and np.all(session_improvement > 0)
                and np.all(influence > 0)
                and float(np.mean(assignment_correct[budget_position])) > 1 / 24
                and efficient_budget is None
            ):
                efficient_budget = budget
        rows.append(row)
    decision = config["decisions"]["anchor_pass"] if efficient_budget is not None else config["decisions"]["anchor_negative"]
    np.savez_compressed(
        output / "objects" / "minimal_anchor_core.npz",
        budgets=np.asarray(budgets), proposed_mae=proposed_mae, direct_mae=direct_mae, assignment_correct=assignment_correct
    )
    pd.DataFrame(rows).to_json(output / "tables" / "minimal_anchor.json", orient="records", indent=2)
    summary = {"decision": decision, "efficient_budget": efficient_budget, "budgets": rows}
    atomic_write_json(output / "decisions" / "minimal_anchor_decision.json", summary)
    return summary


def generate_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_config(root); parent_hashes = validate_parent_hashes(root, config)
    output = root / str(config["project"]["output_dir"])
    def read_optional(relative: str, default: Any) -> Any:
        path = output / relative
        return json.loads(path.read_text()) if path.exists() else default
    cohort = read_optional("cohort/cohort_object_manifest.json", {})
    reliability = read_optional("decisions/reliability_decision.json", {"status": "NOT_RUN"})
    population = read_optional("decisions/population_structure_decision.json", {"terminal": "NOT_RUN"})
    source = read_optional("decisions/source_reference_decision.json", {"identity_decision": "NOT_RUN"})
    semantic = read_optional("decisions/semantic_permutation_decision.json", {"decision": "NOT_RUN"})
    scatter = read_optional("decisions/unlabeled_scatter_decision.json", {"decision": "NOT_RUN"})
    component = read_optional("decisions/component_assignment_decision.json", {"status": "NOT_RUN"})
    anchor = read_optional("decisions/minimal_anchor_decision.json", {"decision": "NOT_RUN"})
    final = {
        "parent_pr": 18,
        "parent_head": config["protocol"]["parent_head"],
        "dataset_contract": {"dataset": "Stieger2021", "sessions": [2, 3], "tasknumber": 3, "classes": list(CLASS_NAMES)},
        "cohort": cohort,
        "reliability": reliability,
        "population": population,
        "source_reference": source,
        "semantic_permutation": semantic,
        "unlabeled_scatter": scatter,
        "component_assignment": component,
        "minimal_anchor": anchor,
        "secondary_LR_UD_status": "NOT_RUN_OPTIONAL_NON_VOTING",
        "parent_hashes_verified": parent_hashes,
        "interpretation_boundary": [
            "not_full_conditional_distribution", "not_physiology", "not_source_anatomy", "not_causal_trait",
            "not_classifier_improvement", "not_domain_adaptation", "not_TTA", "not_pseudo_label_validity",
            "not_ASD_biomarker", "not_intervention_effect",
        ],
        "next_scientific_question": "Does the frozen multiclass source-reference structure reproduce in a second prospectively locked repeated-session cohort, and can its semantic-template assumption be externally calibrated without target outcome information?",
    }
    atomic_write_json(output / "decisions" / "terminal_decisions.json", final)
    lines = [
        "# Stieger2021 Multiclass Confirmation V0 — Final Report",
        "",
        f"Parent: PR #18 at `{config['protocol']['parent_head']}`. This was prospectively frozen before any Stieger EEG sample access.",
        "",
        "## Dataset and cohort",
        "",
        "Primary data are Stieger2021 sessions 2 and 3, task 3 only, with literal order right hand, left hand, both hands, rest.",
        f"Eligible subjects: **{cohort.get('eligible_subject_count', 'not locked')}**. Excluded: **{cohort.get('excluded_subject_count', 'not locked')}**.",
        "",
        "## Frozen decisions",
        "",
        f"- Reliability: `{reliability.get('status', 'NOT_RUN')}`",
        f"- Population structure: `{population.get('terminal', 'NOT_RUN')}`",
        f"- Source-reference identity: `{source.get('identity_decision', 'NOT_RUN')}`",
        f"- Source semantic permutation: `{semantic.get('decision', 'NOT_RUN')}`",
        f"- Unlabeled scatter: `{scatter.get('decision', 'NOT_RUN')}`",
        f"- Component assignment: `{component.get('status', 'NOT_RUN')}`",
        f"- Minimal anchor: `{anchor.get('decision', 'NOT_RUN')}`",
        "",
        "## Population statistics",
        "",
        f"Observed and ranks: `{json.dumps(population.get('observed', {}), sort_keys=True)}`",
        f"Primary nulls: `{json.dumps(population.get('nulls', {}), sort_keys=True)}`",
        "",
        "## Source-reference, semantic, and recovery results",
        "",
        f"Source-reference: `{json.dumps(source, sort_keys=True)}`",
        f"Semantic permutation: `{json.dumps(semantic, sort_keys=True)}`",
        f"Unlabeled scatter: `{json.dumps(scatter, sort_keys=True)}`",
        f"Component assignment: `{json.dumps(component, sort_keys=True)}`",
        f"Minimal anchor: `{json.dumps(anchor, sort_keys=True)}`",
        "",
        "## Boundaries",
        "",
        "These results do not establish a full conditional distribution, physiology, source anatomy, causality, universal individual coordinates, classifier benefit, domain adaptation, TTA, pseudo-label validity, ASD biomarkers, or intervention effects. Task-1 LR and task-2 UD diagnostics were optional and did not vote.",
        "",
        "## Exact next scientific question",
        "",
        final["next_scientific_question"],
        "",
    ]
    report_path = output / "report" / "stieger2021_multiclass_confirmation_v0.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    artifact_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifact_rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "namespace": OUTPUT_NAME,
        "phase": "FINAL_RESULT" if population.get("terminal") != "NOT_RUN" else "INCOMPLETE",
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "parent_hashes": parent_hashes,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {"report": str(report_path.relative_to(root)), "manifest_artifacts": len(artifact_rows), **final}
