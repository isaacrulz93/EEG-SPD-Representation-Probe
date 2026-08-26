"""Leakage-safe cross-session subject-location conditional prediction V0.

Only the locked PR #19 Stieger2021 compact geometry is consumed.  Prediction
processes receive source-derived input packets and never open sealed held-out
conditional outcomes.  No classifier, neural network, or TTA path exists.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from pyriemann.geometry.mean import mean_riemann
from scipy import stats

from src.conditional_geometry_v1 import airm_distance, karcher_residual, spd_invsqrt, spd_log
from src.interaction_provenance_v0 import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_array,
    sha256_file,
)
from src import stieger2021_multiclass_confirmation_v0 as parent


CONFIG_PATH = "configs/subject_location_conditional_configuration_v0.yaml"
OUTPUT_NAME = "subject_location_conditional_configuration_v0"
PARENT_HEAD = "6abb73d82a0f616e0ca9d3eaa44e23d911a2123f"
LOCKED_GEOMETRY_SHA256 = "bce785c13d3e851fb73e5554a5efc990592e832946b53b64fcdc08a708c83515"
FOLDS_CANONICAL_SHA256 = "a3bf9afddb83ab0c0f192b7e337a44dabe24790a2bb083316aa4d18c0347610d"
STREAMED_CANONICAL_SHA256 = "60aa67ccf0eaee9ef2618e4c775eec5f8f4c60bb40d1c8eb0c3b244dbe53f465"
CLASS_NAMES = ("right_hand", "left_hand", "both_hand", "rest")
SESSIONS = (2, 3)
DIRECTIONS = {"FORWARD": (2, 3), "REVERSE": (3, 2)}
SAME_SESSION_DIRECTIONS = {"SAME_SESSION_2": (2, 2), "SAME_SESSION_3": (3, 3)}
STATE_ORDER = (
    "PARENT_VALIDATED",
    "PROTOCOL_FROZEN",
    "OBJECTS_LOCKED_NO_TARGET_OUTCOME_ACCESSED",
    "PRIMARY_PREDICTIONS_FROZEN",
    "TARGET_OUTCOMES_RELEASED_FOR_EVALUATION",
    "NULLS_COMPLETE",
    "TERMINAL_WRITTEN",
    "STOPPED",
)
QUESTION = (
    "Can the label-free geometric location of a subject in a source-population SPD "
    "coordinate system predict that subject's class-centered conditional configuration "
    "in another session?"
)
FORBIDDEN_CLAIMS = (
    "deployable TTA method",
    "target class-semantic identification",
    "classifier improvement",
    "conditional density recovery",
    "causal subject factor",
    "physiology or source anatomy",
    "ASD generalization",
    "universal EEG coordinates",
)


class SubjectLocationError(RuntimeError):
    """Fail-closed data, numerical, state, or leakage contract failure."""


@dataclass(frozen=True)
class ParentObjects:
    subjects: np.ndarray
    folds: tuple[np.ndarray, ...]
    inner_folds: tuple[tuple[np.ndarray, ...], ...]
    marginal: np.ndarray
    class_means: np.ndarray
    proportions: np.ndarray
    counts: np.ndarray
    subject_references: np.ndarray


@dataclass(frozen=True)
class ReducedRankModel:
    q_mean: np.ndarray
    d_mean: np.ndarray
    basis: np.ndarray
    q_train_centered: np.ndarray
    alpha: np.ndarray
    rank: int
    requested_rank: int
    ridge_multiplier: float
    ridge_lambda: float
    kappa: float
    numerical_rank: int


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def output_root(repo_root: str | Path) -> Path:
    return Path(repo_root).resolve() / "outputs" / OUTPUT_NAME


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _rng(config: Mapping[str, Any], namespace: str, *parts: Any) -> np.random.Generator:
    material = "|".join([str(config["protocol"]["master_seed"]), namespace, *map(str, parts)])
    seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(seed)


def atomic_savez(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".npz", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        with np.load(temporary, allow_pickle=False) as reread:
            if set(reread.files) != set(arrays):
                raise SubjectLocationError(f"NPZ reread key mismatch for {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def load_config(repo_root: str | Path, *, verify_protocol: bool = True) -> tuple[dict[str, Any], str]:
    root = Path(repo_root).resolve()
    path = root / CONFIG_PATH
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SubjectLocationError("V0 config must be a mapping")
    expected = str(value["protocol"]["protocol_sha256"])
    if verify_protocol and expected != "TO_BE_FROZEN":
        observed = sha256_file(root / str(value["protocol"]["protocol_path"]))
        if observed != expected:
            raise SubjectLocationError(f"protocol SHA mismatch: {observed} != {expected}")
    return value, sha256_file(path)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _validate_state_artifacts(root: Path, state: Mapping[str, Any]) -> None:
    for relative, expected in state.get("artifact_hashes", {}).items():
        path = root / relative
        if not path.is_file():
            raise SubjectLocationError(f"prior state artifact disappeared: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise SubjectLocationError(f"prior state artifact mutated: {relative}")


def read_scientific_state(repo_root: str | Path) -> dict[str, Any]:
    path = output_root(repo_root) / "scientific_state.json"
    if not path.is_file():
        raise SubjectLocationError("scientific_state.json does not exist")
    value = json.loads(path.read_text(encoding="utf-8"))
    _validate_state_artifacts(Path(repo_root).resolve(), value)
    return value


def transition_state(
    repo_root: str | Path,
    config: Mapping[str, Any],
    new_state: str,
    *,
    gates: Sequence[str],
    artifacts: Sequence[str | Path] = (),
    terminal: str | None = None,
    exact_next_question: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = output_root(root)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "scientific_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _validate_state_artifacts(root, state)
        current_index = STATE_ORDER.index(str(state["current_state"]))
        if current_index + 1 >= len(STATE_ORDER) or STATE_ORDER[current_index + 1] != new_state:
            raise SubjectLocationError(f"invalid scientific-state transition {state['current_state']} -> {new_state}")
    else:
        if new_state != STATE_ORDER[0]:
            raise SubjectLocationError("the first scientific state must be PARENT_VALIDATED")
        state = {
            "fixed_research_question": QUESTION,
            "accepted_parent_facts": {
                "parent_pr": 19,
                "parent_head": PARENT_HEAD,
                "dataset": "Stieger2021",
                "subjects": 62,
                "sessions": [2, 3],
                "task_number": 3,
                "class_order": list(CLASS_NAMES),
                "locked_geometry_sha256": LOCKED_GEOMETRY_SHA256,
                "folds_canonical_sha256": FOLDS_CANONICAL_SHA256,
                "streamed_manifest_canonical_sha256": STREAMED_CANONICAL_SHA256,
            },
            "active_hypothesis": "label-free subject SPD location predicts cross-session subject-by-class interaction",
            "forbidden_claims": list(FORBIDDEN_CLAIMS),
            "current_state": None,
            "completed_gates": [],
            "terminal": None,
            "exact_next_question": None,
            "artifact_hashes": {},
        }
    hashes = dict(state.get("artifact_hashes", {}))
    for artifact in artifacts:
        path = Path(artifact)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise SubjectLocationError(f"state artifact missing: {path}")
        hashes[str(path.relative_to(root))] = sha256_file(path)
    state["artifact_hashes"] = hashes
    state["current_state"] = new_state
    state["completed_gates"] = list(dict.fromkeys([*state.get("completed_gates", []), *gates]))
    if terminal is not None:
        state["terminal"] = terminal
    if exact_next_question is not None:
        state["exact_next_question"] = exact_next_question
    state["updated_utc_epoch_seconds"] = time.time()
    status_index = STATE_ORDER.index(new_state) + 1
    status_path = out / "status" / f"{status_index:02d}_{new_state.lower()}.json"
    snapshot = {key: value for key, value in state.items() if key != "artifact_hashes"}
    snapshot["validated_prior_artifact_count"] = len(hashes)
    snapshot["content_sha256"] = canonical_sha256(snapshot)
    atomic_write_json(status_path, snapshot)
    state["artifact_hashes"][str(status_path.relative_to(root))] = sha256_file(status_path)
    atomic_write_json(state_path, state)
    return state


def _validate_parent_manifest_artifacts(root: Path, manifest: Mapping[str, Any]) -> int:
    count = 0
    for record in manifest.get("artifacts", []):
        path = root / str(record["path"])
        if not path.is_file() or sha256_file(path) != str(record["sha256"]):
            raise SubjectLocationError(f"parent manifest artifact validation failed: {path}")
        count += 1
    if count != int(manifest.get("artifact_count", -1)):
        raise SubjectLocationError("parent manifest artifact count mismatch")
    return count


def validate_parent_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root, verify_protocol=False)
    if str(config["protocol"]["parent_head"]) != PARENT_HEAD:
        raise SubjectLocationError("configured parent head changed")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_HEAD, "HEAD"], cwd=root).returncode != 0:
        raise SubjectLocationError("exact PR #19 head is not an ancestor")
    parent_root = root / str(config["project"]["parent_output_dir"])
    object_path = root / str(config["parent_contract"]["locked_geometry_path"])
    if sha256_file(object_path) != LOCKED_GEOMETRY_SHA256:
        raise SubjectLocationError("locked geometry SHA mismatch")
    fold_path = root / str(config["parent_contract"]["folds_path"])
    folds = json.loads(fold_path.read_text(encoding="utf-8"))
    recanonical = {key: value for key, value in folds.items() if key != "canonical_sha256"}
    if canonical_sha256(recanonical) != FOLDS_CANONICAL_SHA256 or folds.get("canonical_sha256") != FOLDS_CANONICAL_SHA256:
        raise SubjectLocationError("fold canonical SHA mismatch")
    cohort_path = root / str(config["parent_contract"]["cohort_manifest_path"])
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("locked_geometry_object_sha256") != LOCKED_GEOMETRY_SHA256:
        raise SubjectLocationError("cohort locked-object SHA mismatch")
    if cohort.get("folds_sha256") != FOLDS_CANONICAL_SHA256:
        raise SubjectLocationError("cohort fold SHA mismatch")
    if cohort.get("streamed_manifest_sha256") != STREAMED_CANONICAL_SHA256:
        raise SubjectLocationError("cohort streamed-manifest SHA mismatch")
    streamed_path = root / str(config["parent_contract"]["streamed_manifest_path"])
    streamed = json.loads(streamed_path.read_text(encoding="utf-8"))
    if streamed.get("canonical_sha256") != STREAMED_CANONICAL_SHA256:
        raise SubjectLocationError("streamed manifest canonical SHA mismatch")
    manifest_path = root / str(config["parent_contract"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_count = _validate_parent_manifest_artifacts(root, manifest)
    with np.load(object_path, allow_pickle=False) as data:
        keys = tuple(data.files)
        subjects = np.asarray(data["subjects"], dtype=np.int64)
        sessions = np.asarray(data["sessions"], dtype=np.int64)
        classes = tuple(str(value) for value in data["class_names"])
        marginal = np.asarray(data["primary__F__marginal"], dtype=np.float64)
        means = np.asarray(data["primary__F__class_means"], dtype=np.float64)
        proportions = np.asarray(data["primary__F__proportions"], dtype=np.float64)
        counts = np.asarray(data["primary__F__counts"], dtype=np.int64)
    if not np.array_equal(subjects, np.arange(1, 63)) or not np.array_equal(sessions, [2, 3]):
        raise SubjectLocationError("parent subject/session ordering mismatch")
    if classes != CLASS_NAMES:
        raise SubjectLocationError("parent class ordering mismatch")
    if marginal.shape != (62, 2, 20, 20) or means.shape != (62, 2, 4, 20, 20):
        raise SubjectLocationError("parent full-split object shape mismatch")
    if proportions.shape != (62, 2, 4) or counts.shape != (62, 2, 4):
        raise SubjectLocationError("parent count/proportion shape mismatch")
    all_spd = np.concatenate([marginal.reshape(-1, 20, 20), means.reshape(-1, 20, 20)])
    eigenvalues = np.linalg.eigvalsh(0.5 * (all_spd + np.swapaxes(all_spd, -1, -2)))
    if not np.isfinite(all_spd).all() or float(eigenvalues.min()) <= 0.0:
        raise SubjectLocationError("parent compact objects are not finite SPD")
    if np.any(counts <= 0) or not np.allclose(proportions.sum(axis=-1), 1.0, atol=2e-15):
        raise SubjectLocationError("parent class counts/proportions invalid")
    outer = [set(map(int, fold)) for fold in folds["outer_test_subjects"]]
    if sorted(value for fold in outer for value in fold) != list(range(1, 63)):
        raise SubjectLocationError("outer folds do not cover subjects exactly once")
    for outer_fold, inner_folds in zip(outer, folds["inner_test_subjects_by_outer_fold"], strict=True):
        expected = set(range(1, 63)) - outer_fold
        observed = [int(value) for fold in inner_folds for value in fold]
        if sorted(observed) != sorted(expected):
            raise SubjectLocationError("inner fold coverage mismatch")
    return {
        "status": "PASS",
        "parent_head": PARENT_HEAD,
        "parent_head_is_ancestor": True,
        "locked_geometry_sha256": LOCKED_GEOMETRY_SHA256,
        "folds_canonical_sha256": FOLDS_CANONICAL_SHA256,
        "fold_file_sha256": sha256_file(fold_path),
        "streamed_manifest_canonical_sha256": STREAMED_CANONICAL_SHA256,
        "cohort_manifest_sha256": sha256_file(cohort_path),
        "parent_manifest_sha256": sha256_file(manifest_path),
        "parent_manifest_artifact_count": artifact_count,
        "locked_object_keys": list(keys),
        "subjects": subjects.tolist(),
        "sessions": sessions.tolist(),
        "class_order": list(classes),
        "marginal_shape": list(marginal.shape),
        "class_mean_shape": list(means.shape),
        "counts_shape": list(counts.shape),
        "svec_dimension": 210,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "count_minimum": int(counts.min()),
        "count_maximum": int(counts.max()),
        "outer_fold_sizes": [len(fold) for fold in outer],
        "inner_fold_sizes": [[len(fold) for fold in folds_for_outer] for folds_for_outer in folds["inner_test_subjects_by_outer_fold"]],
        "raw_eeg_opened": False,
    }


def _fit_airm_mean(matrices: np.ndarray, config: Mapping[str, Any], name: str) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(matrices, dtype=np.float64)
    mean = np.asarray(mean_riemann(
        values,
        tol=float(config["geometry"]["mean_tol"]),
        maxiter=int(config["geometry"]["mean_maxiter"]),
        init=None,
    ), dtype=np.float64)
    mean = 0.5 * (mean + mean.T)
    eigenvalues = np.linalg.eigvalsh(mean)
    residual = karcher_residual(values, mean)
    condition = float(eigenvalues.max() / eigenvalues.min())
    if (
        not np.isfinite(mean).all()
        or float(eigenvalues.min()) <= 0.0
        or condition > float(config["geometry"]["condition_number_max"])
        or residual > float(config["geometry"]["karcher_residual_max"])
    ):
        raise SubjectLocationError(f"AIRM mean gate failed for {name}: residual={residual}, condition={condition}")
    return mean, {
        "name": name,
        "n_matrices": len(values),
        "karcher_residual": residual,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_number": condition,
    }


def _coordinate(reference: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    inverse_root = spd_invsqrt(np.asarray(reference, dtype=np.float64))
    values = np.asarray(matrices, dtype=np.float64)
    whitened = inverse_root @ values @ inverse_root
    whitened = 0.5 * (whitened + np.swapaxes(whitened, -1, -2))
    return parent.svec(spd_log(whitened))


def _subject_references(marginal: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.empty((len(marginal), 20, 20), dtype=np.float64)
    audits: list[dict[str, Any]] = []
    for index in range(len(marginal)):
        output[index], audit = _fit_airm_mean(marginal[index], config, f"R_subject_{index + 1}")
        audits.append(audit)
    return output, audits


def load_parent_objects(repo_root: str | Path, config: Mapping[str, Any]) -> ParentObjects:
    root = Path(repo_root).resolve()
    parent_config, _ = parent.load_config(root)
    locked = parent.load_locked_objects(root, parent_config)
    marginal = np.asarray(locked.marginal_means[("primary", "F")], dtype=np.float64)
    class_means = np.asarray(locked.class_means[("primary", "F")], dtype=np.float64)
    proportions = np.asarray(locked.proportions[("primary", "F")], dtype=np.float64)
    counts = np.asarray(locked.counts[("primary", "F")], dtype=np.int64)
    references, _ = _subject_references(marginal, config)
    return ParentObjects(
        subjects=np.asarray(locked.subjects, dtype=np.int64),
        folds=tuple(np.asarray(value, dtype=np.int64) for value in locked.folds),
        inner_folds=tuple(tuple(np.asarray(value, dtype=np.int64) for value in group) for group in locked.inner_folds),
        marginal=marginal,
        class_means=class_means,
        proportions=proportions,
        counts=counts,
        subject_references=references,
    )


def _session_index(session: int) -> int:
    if session not in SESSIONS:
        raise ValueError(f"unknown session {session}")
    return SESSIONS.index(session)


def construct_fold_coordinates(
    objects: ParentObjects,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    input_session: int,
    output_session: int,
    config: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if np.intersect1d(train, test).size or sorted(np.concatenate([train, test]).tolist()) != list(range(62)):
        # Inner folds cover the outer source population, not all 62.
        universe = set(train.tolist()) | set(test.tolist())
        if len(universe) != len(train) + len(test):
            raise SubjectLocationError(f"overlapping train/test indices in {name}")
    m0, mean_audit = _fit_airm_mean(objects.subject_references[train], config, f"{name}_M0")
    input_position = _session_index(input_session)
    output_position = _session_index(output_session)
    q_train = _coordinate(m0, objects.marginal[train, input_position])
    q_test = _coordinate(m0, objects.marginal[test, input_position])
    z_train = _coordinate(m0, objects.class_means[train, output_position])
    z_test = _coordinate(m0, objects.class_means[test, output_position])
    d_train = z_train - z_train.mean(axis=1, keepdims=True)
    d_test = z_test - z_test.mean(axis=1, keepdims=True)
    d_bar = d_train.mean(axis=0)
    delta_train = d_train - d_bar[None]
    delta_test = d_test - d_bar[None]
    d_zero_error = float(np.max(np.linalg.norm(d_train.sum(axis=1), axis=1)))
    delta_zero_error = float(np.max(np.linalg.norm(delta_train.sum(axis=1), axis=1)))
    target_zero_error = float(np.max(np.linalg.norm(delta_test.sum(axis=1), axis=1)))
    if max(d_zero_error, delta_zero_error, target_zero_error) > 2e-10:
        raise SubjectLocationError(f"zero-sum class configuration gate failed in {name}")
    return {
        "train_subjects": objects.subjects[train],
        "test_subjects": objects.subjects[test],
        "m0": m0,
        "q_train": q_train,
        "q_test": q_test,
        "d_bar": d_bar,
        "delta_train": delta_train.reshape(len(train), -1),
        "delta_test": delta_test.reshape(len(test), -1),
        "delta_train_class": delta_train,
        "delta_test_class": delta_test,
        "audit": {
            **mean_audit,
            "input_session": input_session,
            "output_session": output_session,
            "train_subject_count": len(train),
            "test_subject_count": len(test),
            "heldout_subjects_excluded_from_m0": True,
            "heldout_subjects_excluded_from_d_bar": True,
            "q_uses_labels": False,
            "d_zero_sum_max_error": d_zero_error,
            "delta_source_zero_sum_max_error": delta_zero_error,
            "delta_target_zero_sum_max_error": target_zero_error,
        },
    }


def numerical_rank(values: np.ndarray) -> int:
    array = np.asarray(values, dtype=np.float64)
    singular = np.linalg.svd(array, full_matrices=False, compute_uv=False)
    if len(singular) == 0 or singular[0] == 0.0:
        return 0
    tolerance = max(array.shape) * np.finfo(np.float64).eps * singular[0]
    return int(np.count_nonzero(singular > tolerance))


def fit_reduced_rank_ridge(
    q: np.ndarray,
    delta: np.ndarray,
    requested_rank: int,
    ridge_multiplier: float,
) -> ReducedRankModel:
    q_values = np.asarray(q, dtype=np.float64)
    d_values = np.asarray(delta, dtype=np.float64)
    if q_values.ndim != 2 or d_values.ndim != 2 or len(q_values) != len(d_values):
        raise SubjectLocationError("reduced-rank fit shape mismatch")
    if not np.isfinite(q_values).all() or not np.isfinite(d_values).all():
        raise SubjectLocationError("nonfinite reduced-rank input")
    q_mean = q_values.mean(axis=0)
    d_mean = d_values.mean(axis=0)
    qc = q_values - q_mean
    dc = d_values - d_mean
    rank_numeric = numerical_rank(dc)
    rank = min(int(requested_rank), len(q_values) - 2, rank_numeric)
    rank = max(rank, 0)
    kernel = qc @ qc.T
    kappa = float(np.trace(kernel) / len(q_values))
    ridge_lambda = float(ridge_multiplier) * max(kappa, np.finfo(np.float64).tiny)
    if rank == 0:
        return ReducedRankModel(
            q_mean=q_mean,
            d_mean=np.zeros_like(d_mean),
            basis=np.zeros((d_values.shape[1], 0), dtype=np.float64),
            q_train_centered=qc,
            alpha=np.zeros((len(q_values), 0), dtype=np.float64),
            rank=0,
            requested_rank=int(requested_rank),
            ridge_multiplier=float(ridge_multiplier),
            ridge_lambda=ridge_lambda,
            kappa=kappa,
            numerical_rank=rank_numeric,
        )
    _, _, vt = np.linalg.svd(dc, full_matrices=False)
    basis = vt[:rank].T.copy()
    scores = dc @ basis
    system = kernel + ridge_lambda * np.eye(len(q_values), dtype=np.float64)
    try:
        alpha = np.linalg.solve(system, scores)
    except np.linalg.LinAlgError as exc:
        raise SubjectLocationError("dual ridge stable solve failed") from exc
    return ReducedRankModel(
        q_mean=q_mean,
        d_mean=d_mean,
        basis=basis,
        q_train_centered=qc,
        alpha=alpha,
        rank=rank,
        requested_rank=int(requested_rank),
        ridge_multiplier=float(ridge_multiplier),
        ridge_lambda=ridge_lambda,
        kappa=kappa,
        numerical_rank=rank_numeric,
    )


def predict_reduced_rank(model: ReducedRankModel, q: np.ndarray) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64)
    if model.rank == 0:
        return np.zeros((len(values), len(model.d_mean)), dtype=np.float64)
    kernel_test = (values - model.q_mean) @ model.q_train_centered.T
    scores = kernel_test @ model.alpha
    prediction = model.d_mean + scores @ model.basis.T
    if not np.isfinite(prediction).all():
        raise SubjectLocationError("nonfinite reduced-rank prediction")
    return prediction


def _normalized_sse(true: np.ndarray, predicted: np.ndarray) -> float:
    target = np.asarray(true, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    denominator = float(np.sum(target * target))
    if denominator <= np.finfo(np.float64).tiny:
        raise SubjectLocationError("zero conditional target energy")
    return float(np.sum((target - estimate) ** 2) / denominator)


def _candidate_grid(config: Mapping[str, Any]) -> list[tuple[int, float]]:
    return [
        (int(rank), float(ridge))
        for rank in config["model"]["output_ranks"]
        for ridge in config["model"]["ridge_multipliers"]
    ]


def select_rank_ridge(
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
    *,
    fixed_rank: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = _candidate_grid(config)
    if fixed_rank is not None:
        grid = [(int(fixed_rank), ridge) for ridge in map(float, config["model"]["ridge_multipliers"])]
    rows: list[dict[str, Any]] = []
    for requested_rank, ridge in grid:
        losses: list[float] = []
        effective_ranks: list[int] = []
        for q_train, d_train, q_validation, d_validation in inner_splits:
            model = fit_reduced_rank_ridge(q_train, d_train, requested_rank, ridge)
            prediction = predict_reduced_rank(model, q_validation)
            losses.append(_normalized_sse(d_validation, prediction))
            effective_ranks.append(model.rank)
        mean = float(np.mean(losses))
        standard_error = float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
        if len(set(effective_ranks)) != 1:
            raise SubjectLocationError("candidate effective rank differs across inner folds")
        rows.append({
            "requested_rank": requested_rank,
            "effective_rank": effective_ranks[0],
            "ridge_multiplier": ridge,
            "mean_inner_loss": mean,
            "inner_standard_error": standard_error,
            "inner_losses": losses,
        })
    best = min(rows, key=lambda row: (row["mean_inner_loss"], row["effective_rank"], -row["ridge_multiplier"]))
    threshold = float(best["mean_inner_loss"] + best["inner_standard_error"])
    eligible = [row for row in rows if row["mean_inner_loss"] <= threshold + 1e-15]
    selected = min(eligible, key=lambda row: (row["effective_rank"], -row["ridge_multiplier"], row["mean_inner_loss"]))
    result = {
        **selected,
        "one_standard_error_threshold": threshold,
        "best_mean_rank": int(best["effective_rank"]),
        "best_mean_ridge_multiplier": float(best["ridge_multiplier"]),
        "candidate_count": len(rows),
        "selection_rule": "one_standard_error_smallest_rank_then_largest_ridge",
    }
    return result, rows


def _save_model(path: Path, model: ReducedRankModel) -> None:
    atomic_savez(
        path,
        q_mean=model.q_mean,
        d_mean=model.d_mean,
        basis=model.basis,
        q_train_centered=model.q_train_centered,
        alpha=model.alpha,
        rank=np.asarray(model.rank, dtype=np.int64),
        requested_rank=np.asarray(model.requested_rank, dtype=np.int64),
        ridge_multiplier=np.asarray(model.ridge_multiplier, dtype=np.float64),
        ridge_lambda=np.asarray(model.ridge_lambda, dtype=np.float64),
        kappa=np.asarray(model.kappa, dtype=np.float64),
        numerical_rank=np.asarray(model.numerical_rank, dtype=np.int64),
    )


def _load_model(path: Path) -> ReducedRankModel:
    with np.load(path, allow_pickle=False) as data:
        return ReducedRankModel(
            q_mean=np.asarray(data["q_mean"], dtype=np.float64),
            d_mean=np.asarray(data["d_mean"], dtype=np.float64),
            basis=np.asarray(data["basis"], dtype=np.float64),
            q_train_centered=np.asarray(data["q_train_centered"], dtype=np.float64),
            alpha=np.asarray(data["alpha"], dtype=np.float64),
            rank=int(data["rank"]),
            requested_rank=int(data["requested_rank"]),
            ridge_multiplier=float(data["ridge_multiplier"]),
            ridge_lambda=float(data["ridge_lambda"]),
            kappa=float(data["kappa"]),
            numerical_rank=int(data["numerical_rank"]),
        )


def _synthetic_inner_splits(q: np.ndarray, d: np.ndarray, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    indices = np.arange(len(q))
    validation_folds = np.array_split(indices, folds)
    output = []
    for validation in validation_folds:
        train = np.setdiff1d(indices, validation, assume_unique=True)
        output.append((q[train], d[train], q[validation], d[validation]))
    return output


def synthetic_gates(config: Mapping[str, Any]) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    rng = _rng(config, "synthetic_known_low_rank")
    q = rng.normal(size=(120, 12))
    left = rng.normal(size=(12, 2))
    right = rng.normal(size=(2, 24))
    d = q @ left @ right + 0.01 * rng.normal(size=(120, 24))
    selection, _ = select_rank_ridge(_synthetic_inner_splits(q[:90], d[:90]), config)
    model = fit_reduced_rank_ridge(q[:90], d[:90], selection["effective_rank"], selection["ridge_multiplier"])
    r2 = 1.0 - _normalized_sse(d[90:], predict_reduced_rank(model, q[90:]))
    permutation = rng.permutation(90)
    shuffled = fit_reduced_rank_ridge(q[:90], d[:90][permutation], selection["effective_rank"], selection["ridge_multiplier"])
    shuffled_r2 = 1.0 - _normalized_sse(d[90:], predict_reduced_rank(shuffled, q[90:]))
    cases["known_low_rank"] = {
        "selected_rank": int(selection["effective_rank"]),
        "heldout_r2": r2,
        "shuffled_r2": shuffled_r2,
        "passed": bool(r2 > 0.75 and selection["effective_rank"] in (1, 2, 3) and r2 > shuffled_r2),
    }

    rng = _rng(config, "synthetic_no_relationship")
    q_null = rng.normal(size=(120, 12))
    d_null = rng.normal(size=(120, 24))
    selection_null, _ = select_rank_ridge(_synthetic_inner_splits(q_null[:90], d_null[:90]), config)
    model_null = fit_reduced_rank_ridge(q_null[:90], d_null[:90], selection_null["effective_rank"], selection_null["ridge_multiplier"])
    null_r2 = 1.0 - _normalized_sse(d_null[90:], predict_reduced_rank(model_null, q_null[90:]))
    cases["no_relationship"] = {
        "selected_rank": int(selection_null["effective_rank"]),
        "heldout_r2": null_r2,
        "passed": bool(null_r2 <= 0.05),
    }

    rng = _rng(config, "synthetic_shared_main")
    subject_main = rng.normal(size=(120, 18))
    class_template = rng.normal(size=(4, 18))
    raw = subject_main[:, None] + class_template[None] + 0.1 * rng.normal(size=(120, 4, 18))
    centered = raw - raw.mean(axis=1, keepdims=True)
    delta = centered - centered[:90].mean(axis=0)[None]
    delta = delta.reshape(120, -1)
    shared_model = fit_reduced_rank_ridge(subject_main[:90], delta[:90], 3, 1.0)
    shared_r2 = 1.0 - _normalized_sse(delta[90:], predict_reduced_rank(shared_model, subject_main[90:]))
    cases["shared_subject_main_effect_only"] = {
        "class_center_error": float(np.max(np.abs(centered.sum(axis=1)))),
        "heldout_r2": shared_r2,
        "passed": bool(shared_r2 <= 0.05 and np.max(np.abs(centered.sum(axis=1))) < 1e-12),
    }

    rng = _rng(config, "synthetic_session_nuisance")
    q2 = rng.normal(size=(120, 12)); q3 = rng.normal(size=(120, 12))
    a2 = rng.normal(size=(12, 18)); a3 = rng.normal(size=(12, 18))
    d2 = q2 @ a2 + 0.05 * rng.normal(size=(120, 18))
    d3 = q3 @ a3 + 0.05 * rng.normal(size=(120, 18))
    def heldout_r2(a: np.ndarray, b: np.ndarray) -> float:
        fitted = fit_reduced_rank_ridge(a[:90], b[:90], 8, 1e-3)
        return 1.0 - _normalized_sse(b[90:], predict_reduced_rank(fitted, a[90:]))
    same2, same3 = heldout_r2(q2, d2), heldout_r2(q3, d3)
    cross23, cross32 = heldout_r2(q2, d3), heldout_r2(q3, d2)
    cases["session_specific_nuisance"] = {
        "same_session_r2": [same2, same3],
        "cross_session_r2": [cross23, cross32],
        "passed": bool(min(same2, same3) > 0.75 and max(cross23, cross32) <= 0.05),
    }

    rng = _rng(config, "synthetic_reference_leakage")
    base = rng.normal(size=(12, 5, 5)); spd = base @ np.swapaxes(base, -1, -2) + np.eye(5)
    source_mean = mean_riemann(spd[:9], tol=1e-9, maxiter=100)
    source_hash = sha256_array(source_mean)
    changed = spd.copy(); changed[9:] *= 1000.0
    changed_mean = mean_riemann(changed[:9], tol=1e-9, maxiter=100)
    cases["fold_reference_leakage"] = {
        "source_hash_before": source_hash,
        "source_hash_after": sha256_array(changed_mean),
        "passed": bool(np.array_equal(source_mean, changed_mean)),
    }

    prediction_source = inspect.getsource(run_primary_predictions)
    cases["outcome_vault_barrier"] = {
        "predictor_mentions_outcome_vault": "outcome_vault" in prediction_source,
        "predictor_loads_locked_parent": "load_parent_objects" in prediction_source,
        "passed": bool("outcome_vault" not in prediction_source and "load_parent_objects" not in prediction_source),
    }
    return {
        "cases": cases,
        "passed": bool(all(bool(case["passed"]) for case in cases.values())),
        "case_count": len(cases),
        "target_conditional_outcome_accessed": False,
        "content_sha256": canonical_sha256(cases),
    }


def _environment_record() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "pandas", "pyriemann", "PyYAML", "matplotlib", "pytest"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
    }


def _initial_manifest(config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    return {
        "namespace": OUTPUT_NAME,
        "phase": phase,
        "parent_head": PARENT_HEAD,
        "branch": config["protocol"]["branch"],
        "automatic_merge": False,
        "raw_eeg_opened": False,
        "classifier_trained": False,
        "neural_network_trained": False,
        "tta_performed": False,
        "pr20_pr21_code_inherited": False,
    }


def freeze_protocol(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root)
    out = output_root(root)
    if (out / "scientific_state.json").is_file():
        existing = read_scientific_state(root)
        if STATE_ORDER.index(existing["current_state"]) >= STATE_ORDER.index("PROTOCOL_FROZEN"):
            return {
                "status": "PROTOCOL_FROZEN_NO_TARGET_CONDITIONAL_OUTCOME_ACCESSED",
                "current_state": existing["current_state"],
                "resumed_from_validated_artifacts": True,
            }
    for directory in (
        "protocol", "status", "objects", "input_packets", "outcome_vault", "predictions",
        "nulls", "controls", "tables", "figures", "decisions", "report",
    ):
        (out / directory).mkdir(parents=True, exist_ok=True)
    validation = validate_parent_contract(root)
    validation_path = out / "tables" / "01_parent_artifact_validation.json"
    atomic_write_json(validation_path, validation)
    schema_rows = []
    object_path = root / str(config["parent_contract"]["locked_geometry_path"])
    with np.load(object_path, allow_pickle=False) as data:
        for key in data.files:
            schema_rows.append({"key": key, "shape": "x".join(map(str, data[key].shape)), "dtype": str(data[key].dtype)})
    schema_path = out / "tables" / "02_locked_object_schema.csv"
    write_csv(schema_path, schema_rows)
    gates = synthetic_gates(config)
    if not gates["passed"]:
        raise SubjectLocationError("one or more synthetic gates failed")
    gates_path = out / "protocol" / "synthetic_gates.json"
    atomic_write_json(gates_path, gates)
    protocol_copy = out / "protocol" / "PROTOCOL_SUBJECT_LOCATION_CONDITIONAL_CONFIGURATION_V0.md"
    atomic_write_bytes(protocol_copy, (root / str(config["protocol"]["protocol_path"])).read_bytes())
    config_copy = out / "protocol" / "subject_location_conditional_configuration_v0.yaml"
    atomic_write_bytes(config_copy, (root / CONFIG_PATH).read_bytes())
    environment_path = out / "environment.json"
    atomic_write_json(environment_path, _environment_record())
    provenance = {
        "parent_head": PARENT_HEAD,
        "head_at_freeze": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "config_sha256": config_hash,
        "protocol_sha256": sha256_file(root / str(config["protocol"]["protocol_path"])),
        "pr20_reference_head": "9c95e5b19eb4c44acc411c1e0d72a5cdd4d9ef63",
        "pr21_reference_head": "94fed2b46701814d7bc3d0747106e4d64c6b0c1f",
        "pr20_pr21_model_code_inspected_or_inherited": False,
    }
    provenance_path = out / "git_provenance.json"
    atomic_write_json(provenance_path, provenance)
    manifest_path = out / "manifest.json"
    atomic_write_json(manifest_path, _initial_manifest(config, "PROTOCOL_FROZEN_NO_TARGET_CONDITIONAL_OUTCOME_ACCESSED"))
    if (out / "scientific_state.json").is_file():
        parent_state = read_scientific_state(root)
        if parent_state["current_state"] != "PARENT_VALIDATED":
            raise SubjectLocationError("protocol freeze cannot resume from an unexpected state")
    else:
        parent_state = transition_state(
            root, config, "PARENT_VALIDATED",
            gates=("exact_parent_head", "parent_manifest_artifacts", "compact_schema", "outer_inner_folds", "all_62x2x4_objects"),
            artifacts=(validation_path, schema_path),
        )
    frozen_state = transition_state(
        root, config, "PROTOCOL_FROZEN",
        gates=("synthetic_gates", "protocol_hash", "no_target_conditional_outcome_access"),
        artifacts=(gates_path, protocol_copy, config_copy, environment_path, provenance_path),
    )
    return {
        "status": "PROTOCOL_FROZEN_NO_TARGET_CONDITIONAL_OUTCOME_ACCESSED",
        "parent_state": parent_state["current_state"],
        "current_state": frozen_state["current_state"],
        "parent_validation": validation,
        "synthetic_gates": gates,
    }


def _outer_train_indices(n_subjects: int, test: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.arange(n_subjects, dtype=np.int64), np.asarray(test, dtype=np.int64), assume_unique=True)


def _packet_paths(out: Path, direction: str, fold_index: int) -> tuple[Path, Path, Path, Path]:
    packet = out / "input_packets" / direction / f"fold_{fold_index + 1:02d}.npz"
    packet_record = packet.with_suffix(".json")
    vault = out / "outcome_vault" / direction / f"fold_{fold_index + 1:02d}.npz"
    vault_record = vault.with_suffix(".json")
    return packet, packet_record, vault, vault_record


def lock_subject_location_objects(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    state = read_scientific_state(root)
    if STATE_ORDER.index(state["current_state"]) >= STATE_ORDER.index("OBJECTS_LOCKED_NO_TARGET_OUTCOME_ACCESSED"):
        locked_manifest_path = output_root(root) / "objects" / "locked_packet_manifest.json"
        locked_manifest = json.loads(locked_manifest_path.read_text(encoding="utf-8"))
        if locked_manifest.get("input_packet_count") != 12 or locked_manifest.get("sealed_vault_count") != 12:
            raise SubjectLocationError("locked packet manifest failed resume validation")
        return {
            "status": state["current_state"],
            "input_packet_count": 12,
            "sealed_vault_count": 12,
            "packet_manifest_sha256": sha256_file(locked_manifest_path),
            "resumed_from_validated_artifacts": True,
        }
    if state["current_state"] != "PROTOCOL_FROZEN":
        raise SubjectLocationError("objects can only be locked from PROTOCOL_FROZEN")
    validate_parent_contract(root)
    out = output_root(root)
    objects = load_parent_objects(root, config)
    _, reference_audits = _subject_references(objects.marginal, config)
    reference_path = out / "objects" / "subject_cross_session_references.npz"
    atomic_savez(reference_path, subjects=objects.subjects, subject_references=objects.subject_references)
    packet_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    artifact_paths: list[Path] = [reference_path]
    for fold_index, test_indices in enumerate(objects.folds):
        train_indices = _outer_train_indices(len(objects.subjects), test_indices)
        for direction, (input_session, output_session) in DIRECTIONS.items():
            outer = construct_fold_coordinates(
                objects, train_indices, test_indices, input_session, output_session, config,
                name=f"outer{fold_index + 1}_{direction}",
            )
            arrays: dict[str, np.ndarray] = {
                "train_subjects": outer["train_subjects"],
                "test_subjects": outer["test_subjects"],
                "q_train": outer["q_train"],
                "delta_train": outer["delta_train"],
                "q_test": outer["q_test"],
                "m0": outer["m0"],
                "d_bar": outer["d_bar"],
            }
            inner_audits = []
            outer_train_set = set(train_indices.tolist())
            for inner_index, validation_indices in enumerate(objects.inner_folds[fold_index]):
                if not set(validation_indices.tolist()).issubset(outer_train_set):
                    raise SubjectLocationError("inherited inner fold is not inside outer source set")
                inner_train = np.asarray(sorted(outer_train_set - set(validation_indices.tolist())), dtype=np.int64)
                inner = construct_fold_coordinates(
                    objects, inner_train, validation_indices, input_session, output_session, config,
                    name=f"outer{fold_index + 1}_{direction}_inner{inner_index + 1}",
                )
                prefix = f"inner_{inner_index}"
                arrays[f"{prefix}_train_subjects"] = inner["train_subjects"]
                arrays[f"{prefix}_validation_subjects"] = inner["test_subjects"]
                arrays[f"{prefix}_q_train"] = inner["q_train"]
                arrays[f"{prefix}_delta_train"] = inner["delta_train"]
                arrays[f"{prefix}_q_validation"] = inner["q_test"]
                arrays[f"{prefix}_delta_validation"] = inner["delta_test"]
                arrays[f"{prefix}_m0"] = inner["m0"]
                arrays[f"{prefix}_d_bar"] = inner["d_bar"]
                inner_audits.append(inner["audit"])
            packet, packet_record, vault, vault_record = _packet_paths(out, direction, fold_index)
            atomic_savez(packet, **arrays)
            packet_payload = {
                "schema": "label_free_input_packet_v0",
                "direction": direction,
                "fold": fold_index + 1,
                "input_session": input_session,
                "output_session": output_session,
                "source_subjects": outer["train_subjects"].tolist(),
                "heldout_subjects": outer["test_subjects"].tolist(),
                "packet_path": str(packet.relative_to(root)),
                "packet_sha256": sha256_file(packet),
                "contains_target_conditional_outcome": False,
                "outer_audit": outer["audit"],
                "inner_audits": inner_audits,
            }
            packet_payload["content_sha256"] = canonical_sha256(packet_payload)
            atomic_write_json(packet_record, packet_payload)
            atomic_savez(
                vault,
                test_subjects=outer["test_subjects"],
                delta_true=outer["delta_test"],
            )
            vault_payload = {
                "schema": "sealed_target_conditional_outcome_vault_v0",
                "direction": direction,
                "fold": fold_index + 1,
                "heldout_subjects": outer["test_subjects"].tolist(),
                "vault_path": str(vault.relative_to(root)),
                "vault_sha256": sha256_file(vault),
                "sealed": True,
                "prediction_process_access_permitted": False,
                "scientific_target_outcome_accessed": False,
            }
            vault_payload["content_sha256"] = canonical_sha256(vault_payload)
            atomic_write_json(vault_record, vault_payload)
            artifact_paths.extend((packet, packet_record, vault, vault_record))
            packet_rows.append({
                "direction": direction,
                "fold": fold_index + 1,
                "source_subject_count": len(outer["train_subjects"]),
                "heldout_subject_count": len(outer["test_subjects"]),
                "input_packet_sha256": sha256_file(packet),
                "sealed_vault_sha256": sha256_file(vault),
                "target_outcome_accessed_by_predictor": False,
            })
            reference_rows.append({
                "direction": direction,
                "fold": fold_index + 1,
                **outer["audit"],
            })
    packet_table = out / "tables" / "03_fold_source_reference_construction.csv"
    write_csv(packet_table, reference_rows)
    packet_manifest = out / "objects" / "locked_packet_manifest.json"
    manifest_payload = {
        "status": "OBJECTS_LOCKED_NO_TARGET_OUTCOME_ACCESSED",
        "input_packet_count": 12,
        "sealed_vault_count": 12,
        "records": packet_rows,
        "subject_reference_sha256": sha256_file(reference_path),
        "target_outcome_accessed_by_prediction": False,
    }
    manifest_payload["content_sha256"] = canonical_sha256(manifest_payload)
    atomic_write_json(packet_manifest, manifest_payload)
    artifact_paths.extend((packet_table, packet_manifest))
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "phase": "OBJECTS_LOCKED_NO_TARGET_OUTCOME_ACCESSED",
        "locked_packet_manifest_sha256": sha256_file(packet_manifest),
        "input_packet_count": 12,
        "sealed_vault_count": 12,
    })
    atomic_write_json(manifest_path, manifest)
    locked_state = transition_state(
        root, config, "OBJECTS_LOCKED_NO_TARGET_OUTCOME_ACCESSED",
        gates=("source_only_fold_references", "inner_fold_recomputation", "zero_sum_class_configuration", "sealed_outcome_vault", "prediction_input_separation"),
        artifacts=tuple(artifact_paths),
    )
    return {
        "status": locked_state["current_state"],
        "input_packet_count": 12,
        "sealed_vault_count": 12,
        "packet_manifest_sha256": sha256_file(packet_manifest),
        "subject_reference_audit_max_residual": max(value["karcher_residual"] for value in reference_audits),
    }


def _load_input_packet(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        output = {key: np.asarray(data[key]) for key in data.files}
    required = {"train_subjects", "test_subjects", "q_train", "delta_train", "q_test", "m0", "d_bar"}
    if not required.issubset(output):
        raise SubjectLocationError(f"input packet missing required keys: {path}")
    forbidden = {"delta_true", "target_class_means", "target_labels", "labels", "prediction_errors"}
    if forbidden.intersection(output):
        raise SubjectLocationError(f"input packet contains forbidden target outcome: {path}")
    return output


def _inner_splits_from_packet(packet: Mapping[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    output = []
    for inner_index in range(5):
        prefix = f"inner_{inner_index}"
        output.append((
            np.asarray(packet[f"{prefix}_q_train"], dtype=np.float64),
            np.asarray(packet[f"{prefix}_delta_train"], dtype=np.float64),
            np.asarray(packet[f"{prefix}_q_validation"], dtype=np.float64),
            np.asarray(packet[f"{prefix}_delta_validation"], dtype=np.float64),
        ))
    return output


def _prediction_paths(out: Path, direction: str, fold_index: int) -> tuple[Path, Path, Path, Path]:
    prediction = out / "predictions" / direction / f"fold_{fold_index + 1:02d}.npz"
    record = prediction.with_suffix(".json")
    model = out / "predictions" / direction / f"fold_{fold_index + 1:02d}_model.npz"
    selection = out / "predictions" / direction / f"fold_{fold_index + 1:02d}_selection.json"
    return prediction, record, model, selection


def validate_prediction_hashes(repo_root: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = output_root(root)
    records = sorted((out / "predictions").glob("*/fold_??.json"))
    if require_complete and len(records) != 12:
        raise SubjectLocationError(f"expected 12 frozen prediction records, found {len(records)}")
    forbidden = {"delta_true", "target_class_means", "target_labels", "labels", "prediction_errors", "sse"}
    validated = []
    for record_path in records:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        prediction_path = root / str(record["prediction_path"])
        model_path = root / str(record["model_path"])
        if sha256_file(prediction_path) != record["prediction_sha256"]:
            raise SubjectLocationError(f"prediction hash mismatch: {prediction_path}")
        if sha256_file(model_path) != record["model_sha256"]:
            raise SubjectLocationError(f"model hash mismatch: {model_path}")
        with np.load(prediction_path, allow_pickle=False) as prediction:
            if forbidden.intersection(prediction.files):
                raise SubjectLocationError(f"prediction contains target outcome: {prediction_path}")
            if set(prediction.files) != {"subject_ids", "delta_hat"}:
                raise SubjectLocationError(f"prediction schema mismatch: {prediction_path}")
            if prediction["delta_hat"].shape[1] != 840:
                raise SubjectLocationError("prediction output dimension mismatch")
        validated.append({
            "direction": record["direction"],
            "fold": record["fold"],
            "prediction_sha256": record["prediction_sha256"],
        })
    return {"status": "PASS", "prediction_count": len(records), "records": validated}


def run_primary_predictions(repo_root: str | Path) -> dict[str, Any]:
    """Fit from source-only packets and freeze predictions without outcome access."""
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    state = read_scientific_state(root)
    if STATE_ORDER.index(state["current_state"]) >= STATE_ORDER.index("PRIMARY_PREDICTIONS_FROZEN"):
        validation = validate_prediction_hashes(root, require_complete=True)
        return {
            "status": state["current_state"],
            "prediction_count": validation["prediction_count"],
            "resumed_from_validated_artifacts": True,
        }
    if state["current_state"] != "OBJECTS_LOCKED_NO_TARGET_OUTCOME_ACCESSED":
        raise SubjectLocationError("primary prediction requires locked input objects")
    out = output_root(root)
    selection_rows: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    for direction in DIRECTIONS:
        for fold_index in range(6):
            packet_path = out / "input_packets" / direction / f"fold_{fold_index + 1:02d}.npz"
            packet_record_path = packet_path.with_suffix(".json")
            packet_record = json.loads(packet_record_path.read_text(encoding="utf-8"))
            if sha256_file(packet_path) != packet_record["packet_sha256"]:
                raise SubjectLocationError("input packet hash failed before prediction")
            packet = _load_input_packet(packet_path)
            selection, candidate_rows = select_rank_ridge(_inner_splits_from_packet(packet), config)
            final_model = fit_reduced_rank_ridge(
                np.asarray(packet["q_train"], dtype=np.float64),
                np.asarray(packet["delta_train"], dtype=np.float64),
                int(selection["effective_rank"]),
                float(selection["ridge_multiplier"]),
            )
            prediction_values = predict_reduced_rank(final_model, np.asarray(packet["q_test"], dtype=np.float64))
            class_sum_error = float(np.max(np.linalg.norm(prediction_values.reshape(len(prediction_values), 4, 210).sum(axis=1), axis=1)))
            if class_sum_error > 2e-10:
                raise SubjectLocationError("predicted conditional configuration violates class zero-sum")
            prediction_path, record_path, model_path, selection_path = _prediction_paths(out, direction, fold_index)
            atomic_savez(
                prediction_path,
                subject_ids=np.asarray(packet["test_subjects"], dtype=np.int64),
                delta_hat=prediction_values,
            )
            _save_model(model_path, final_model)
            selection_payload = {
                "direction": direction,
                "fold": fold_index + 1,
                "selected": selection,
                "candidates": candidate_rows,
                "source_only_inner_selection": True,
                "target_outcome_accessed": False,
            }
            selection_payload["content_sha256"] = canonical_sha256(selection_payload)
            atomic_write_json(selection_path, selection_payload)
            prediction_payload = {
                "schema": "frozen_conditional_configuration_prediction_v0",
                "direction": direction,
                "fold": fold_index + 1,
                "heldout_subjects": np.asarray(packet["test_subjects"], dtype=int).tolist(),
                "prediction_path": str(prediction_path.relative_to(root)),
                "prediction_sha256": sha256_file(prediction_path),
                "model_path": str(model_path.relative_to(root)),
                "model_sha256": sha256_file(model_path),
                "selection_path": str(selection_path.relative_to(root)),
                "selection_sha256": sha256_file(selection_path),
                "selected_rank": final_model.rank,
                "selected_ridge_multiplier": final_model.ridge_multiplier,
                "prediction_class_zero_sum_max_error": class_sum_error,
                "target_outcome_accessed": False,
                "contains_true_delta": False,
                "contains_prediction_error": False,
            }
            prediction_payload["content_sha256"] = canonical_sha256(prediction_payload)
            atomic_write_json(record_path, prediction_payload)
            artifacts.extend((prediction_path, record_path, model_path, selection_path))
            selection_rows.append({
                "direction": direction,
                "fold": fold_index + 1,
                "selected_rank": final_model.rank,
                "selected_requested_rank": final_model.requested_rank,
                "selected_ridge_multiplier": final_model.ridge_multiplier,
                "ridge_lambda": final_model.ridge_lambda,
                "kappa": final_model.kappa,
                "output_numerical_rank": final_model.numerical_rank,
                "best_mean_rank": selection["best_mean_rank"],
                "best_mean_ridge_multiplier": selection["best_mean_ridge_multiplier"],
                "one_standard_error_threshold": selection["one_standard_error_threshold"],
            })
    selection_table = out / "tables" / "06_selected_ranks_and_ridge_values.csv"
    write_csv(selection_table, selection_rows)
    artifacts.append(selection_table)
    validation = validate_prediction_hashes(root, require_complete=True)
    prediction_manifest_path = out / "predictions" / "prediction_manifest.json"
    prediction_manifest = {
        "status": "PRIMARY_PREDICTIONS_FROZEN",
        "prediction_count": validation["prediction_count"],
        "records": validation["records"],
        "target_outcomes_accessed": False,
        "all_prediction_hashes_valid": True,
    }
    prediction_manifest["content_sha256"] = canonical_sha256(prediction_manifest)
    atomic_write_json(prediction_manifest_path, prediction_manifest)
    artifacts.append(prediction_manifest_path)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "phase": "PRIMARY_PREDICTIONS_FROZEN",
        "prediction_count": 12,
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "target_outcomes_accessed": False,
    })
    atomic_write_json(manifest_path, manifest)
    frozen = transition_state(
        root, config, "PRIMARY_PREDICTIONS_FROZEN",
        gates=("source_only_inner_model_selection", "dual_ridge_stable_solve", "complete_forward_reverse_predictions", "prediction_hashes", "target_outcome_barrier"),
        artifacts=artifacts,
    )
    return {
        "status": frozen["current_state"],
        "prediction_count": 12,
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "selected_models": selection_rows,
    }


def _safe_cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _subject_metric_rows(direction: str, subjects: np.ndarray, true: np.ndarray, predicted: np.ndarray, folds: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject, target, estimate, fold in zip(subjects, true, predicted, folds, strict=True):
        sse_zero = float(np.sum(target * target))
        sse_location = float(np.sum((target - estimate) ** 2))
        rows.append({
            "direction": direction,
            "subject": int(subject),
            "outer_fold": int(fold),
            "sse_zero": sse_zero,
            "sse_location": sse_location,
            "normalized_sse_ratio": sse_location / max(sse_zero, np.finfo(np.float64).tiny),
            "error_gain": sse_zero - sse_location,
            "cosine": _safe_cosine(target, estimate),
            "normalized_mae": float(np.sum(np.abs(target - estimate)) / max(np.sum(np.abs(target)), np.finfo(np.float64).tiny)),
            "predicted_residual_norm": float(np.linalg.norm(estimate)),
            "true_residual_norm": float(np.linalg.norm(target)),
        })
    return rows


def _aggregate_r2(rows: Sequence[Mapping[str, Any]]) -> float:
    denominator = sum(float(row["sse_zero"]) for row in rows)
    return 1.0 - sum(float(row["sse_location"]) for row in rows) / denominator


def _sign_flip_p(gains: np.ndarray, config: Mapping[str, Any], namespace: str) -> dict[str, Any]:
    values = np.asarray(gains, dtype=np.float64)
    observed = abs(float(values.mean()))
    if len(values) <= 24:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))), dtype=np.float64)
        permuted = np.abs((signs * values[None]).mean(axis=1))
        return {"p_value": float(np.mean(permuted >= observed - 1e-15)), "method": "EXACT", "replicates": len(signs)}
    replicates = int(config["inference"]["sign_flip_replicates_when_exact_infeasible"])
    rng = _rng(config, "sign_flip", namespace)
    extreme = 0
    remaining = replicates
    while remaining:
        size = min(10000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(values)))
        permuted = np.abs((signs * values[None]).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        remaining -= size
    return {
        "p_value": float((1 + extreme) / (replicates + 1)),
        "method": "DETERMINISTIC_MONTE_CARLO_EXACT_INFEASIBLE_N_GT_24",
        "replicates": replicates,
    }


def _bootstrap_inference(rows_by_direction: Mapping[str, list[dict[str, Any]]], config: Mapping[str, Any]) -> dict[str, Any]:
    replicates = int(config["inference"]["bootstrap_replicates"])
    output: dict[str, Any] = {}
    indexed: dict[str, dict[int, dict[str, Any]]] = {
        direction: {int(row["subject"]): row for row in rows}
        for direction, rows in rows_by_direction.items()
    }
    subjects = np.arange(1, 63, dtype=np.int64)
    for label in ("FORWARD", "REVERSE", "POOLED"):
        rng = _rng(config, "bootstrap", label)
        r2_values = np.empty(replicates, dtype=np.float64)
        gain_values = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            if label == "POOLED":
                sample_rows = [indexed[direction][int(subject)] for subject in sampled for direction in DIRECTIONS]
                subject_gains = np.asarray([
                    np.mean([indexed[direction][int(subject)]["error_gain"] for direction in DIRECTIONS])
                    for subject in sampled
                ])
            else:
                sample_rows = [indexed[label][int(subject)] for subject in sampled]
                subject_gains = np.asarray([indexed[label][int(subject)]["error_gain"] for subject in sampled])
            r2_values[replicate] = _aggregate_r2(sample_rows)
            gain_values[replicate] = float(subject_gains.mean())
        if label == "POOLED":
            observed_rows = [indexed[direction][subject] for subject in subjects for direction in DIRECTIONS]
            sign_values = np.asarray([
                np.mean([indexed[direction][subject]["error_gain"] for direction in DIRECTIONS])
                for subject in subjects
            ])
        else:
            observed_rows = list(indexed[label].values())
            sign_values = np.asarray([row["error_gain"] for row in observed_rows])
        sign = _sign_flip_p(sign_values, config, label)
        output[label] = {
            "r2_cond": _aggregate_r2(observed_rows),
            "r2_bootstrap_95_low": float(np.quantile(r2_values, 0.025)),
            "r2_bootstrap_95_high": float(np.quantile(r2_values, 0.975)),
            "mean_subject_error_gain": float(sign_values.mean()),
            "mean_error_gain_bootstrap_95_low": float(np.quantile(gain_values, 0.025)),
            "mean_error_gain_bootstrap_95_high": float(np.quantile(gain_values, 0.975)),
            "median_subject_error_gain": float(np.median(sign_values)),
            "positive_gain_count": int(np.count_nonzero(sign_values > 0.0)),
            "unit_count": len(sign_values),
            "sign_flip_p": sign["p_value"],
            "sign_flip_method": sign["method"],
            "sign_flip_replicates": sign["replicates"],
        }
    pooled_rows = {int(subject): [indexed[direction][int(subject)] for direction in DIRECTIONS] for subject in subjects}
    leave_one = []
    for omitted in subjects:
        rows = [row for subject, pair in pooled_rows.items() if subject != omitted for row in pair]
        leave_one.append({"omitted_subject": int(omitted), "pooled_r2": _aggregate_r2(rows)})
    output["POOLED"]["leave_one_subject_r2_min"] = min(row["pooled_r2"] for row in leave_one)
    output["POOLED"]["leave_one_subject_r2_max"] = max(row["pooled_r2"] for row in leave_one)
    return {"summary": output, "leave_one_subject": leave_one}


def release_target_outcomes_and_evaluate(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    state = read_scientific_state(root)
    if STATE_ORDER.index(state["current_state"]) > STATE_ORDER.index("TARGET_OUTCOMES_RELEASED_FOR_EVALUATION"):
        primary_path = output_root(root) / "tables" / "09_pooled_primary_metrics.json"
        if not primary_path.is_file():
            raise SubjectLocationError("released evaluation artifact missing during resume")
        return {
            "status": state["current_state"],
            "inference": json.loads(primary_path.read_text(encoding="utf-8")),
            "artifacts": [],
            "resumed_from_validated_artifacts": True,
        }
    if state["current_state"] not in ("PRIMARY_PREDICTIONS_FROZEN", "TARGET_OUTCOMES_RELEASED_FOR_EVALUATION"):
        raise SubjectLocationError("target outcomes can only be released after predictions freeze")
    prediction_validation = validate_prediction_hashes(root, require_complete=True)
    out = output_root(root)
    if state["current_state"] == "PRIMARY_PREDICTIONS_FROZEN":
        transition_state(
            root, config, "TARGET_OUTCOMES_RELEASED_FOR_EVALUATION",
            gates=("all_primary_predictions_frozen_and_hashed", "target_outcome_vault_release_authorized_for_evaluation_only"),
            artifacts=(out / "predictions" / "prediction_manifest.json",),
        )
    rows_by_direction: dict[str, list[dict[str, Any]]] = {key: [] for key in DIRECTIONS}
    class_accumulator: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {key: [] for key in DIRECTIONS}
    vault_release_rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for fold_index in range(6):
            prediction_path, record_path, _, _ = _prediction_paths(out, direction, fold_index)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if sha256_file(prediction_path) != record["prediction_sha256"]:
                raise SubjectLocationError("prediction mutated before outcome release")
            vault_path = out / "outcome_vault" / direction / f"fold_{fold_index + 1:02d}.npz"
            vault_record_path = vault_path.with_suffix(".json")
            vault_record = json.loads(vault_record_path.read_text(encoding="utf-8"))
            if sha256_file(vault_path) != vault_record["vault_sha256"]:
                raise SubjectLocationError("sealed outcome vault hash mismatch")
            with np.load(prediction_path, allow_pickle=False) as prediction, np.load(vault_path, allow_pickle=False) as vault:
                subjects = np.asarray(prediction["subject_ids"], dtype=np.int64)
                estimate = np.asarray(prediction["delta_hat"], dtype=np.float64)
                true_subjects = np.asarray(vault["test_subjects"], dtype=np.int64)
                true = np.asarray(vault["delta_true"], dtype=np.float64)
            if not np.array_equal(subjects, true_subjects):
                raise SubjectLocationError("prediction/vault heldout subject mismatch")
            fold_vector = np.full(len(subjects), fold_index + 1, dtype=np.int64)
            rows_by_direction[direction].extend(_subject_metric_rows(direction, subjects, true, estimate, fold_vector))
            class_accumulator[direction].append((true.reshape(len(true), 4, 210), estimate.reshape(len(estimate), 4, 210)))
            vault_release_rows.append({
                "direction": direction,
                "fold": fold_index + 1,
                "prediction_sha256": sha256_file(prediction_path),
                "vault_sha256": sha256_file(vault_path),
                "prediction_frozen_before_release": True,
                "heldout_subject_count": len(subjects),
            })
    for direction, rows in rows_by_direction.items():
        if sorted(int(row["subject"]) for row in rows) != list(range(1, 63)):
            raise SubjectLocationError(f"incomplete heldout subject coverage for {direction}")
    subject_rows = [row for direction in DIRECTIONS for row in rows_by_direction[direction]]
    subject_path = out / "tables" / "10_subject_level_error_gains.csv"
    write_csv(subject_path, subject_rows)
    inference = _bootstrap_inference(rows_by_direction, config)
    primary_path = out / "tables" / "09_pooled_primary_metrics.json"
    atomic_write_json(primary_path, inference)
    leave_one_path = out / "tables" / "16_leave_one_subject_influence.csv"
    write_csv(leave_one_path, inference["leave_one_subject"])
    class_rows: list[dict[str, Any]] = []
    for direction, collections in class_accumulator.items():
        true = np.concatenate([item[0] for item in collections])
        estimate = np.concatenate([item[1] for item in collections])
        for class_index, class_name in enumerate(CLASS_NAMES):
            target = true[:, class_index]
            predicted = estimate[:, class_index]
            baseline_sse = float(np.sum(target * target))
            location_sse = float(np.sum((target - predicted) ** 2))
            class_rows.append({
                "direction": direction,
                "class": class_name,
                "sse_improvement": baseline_sse - location_sse,
                "r2_cond": 1.0 - location_sse / baseline_sse,
                "cosine": _safe_cosine(target, predicted),
                "normalized_mae": float(np.sum(np.abs(target - predicted)) / np.sum(np.abs(target))),
                "true_residual_norm": float(np.linalg.norm(target)),
                "predicted_residual_norm": float(np.linalg.norm(predicted)),
                "norm_ratio": float(np.linalg.norm(predicted) / np.linalg.norm(target)),
            })
    class_path = out / "tables" / "11_class_level_prediction_metrics.csv"
    write_csv(class_path, class_rows)
    release_path = out / "tables" / "target_outcome_release_audit.csv"
    write_csv(release_path, vault_release_rows)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "phase": "TARGET_OUTCOMES_RELEASED_FOR_EVALUATION",
        "prediction_hash_validation": prediction_validation["status"],
        "primary_metrics_sha256": sha256_file(primary_path),
        "subject_metric_rows": len(subject_rows),
        "target_outcomes_used_for_model_selection": False,
    })
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "TARGET_OUTCOMES_RELEASED_FOR_EVALUATION",
        "inference": inference,
        "subject_rows": subject_rows,
        "class_rows": class_rows,
        # manifest.json is intentionally mutable until STOPPED and therefore is
        # never included in an intermediate state's immutable artifact set.
        "artifacts": [subject_path, primary_path, leave_one_path, class_path, release_path],
    }


def _derangement(rng: np.random.Generator, n: int) -> np.ndarray:
    if n < 2:
        raise SubjectLocationError("derangement requires at least two heldout subjects")
    for _ in range(10000):
        candidate = rng.permutation(n)
        if not np.any(candidate == np.arange(n)):
            return candidate
    raise SubjectLocationError("deterministic derangement generation failed")


def _load_true_fold(out: Path, direction: str, fold_index: int) -> tuple[np.ndarray, np.ndarray]:
    path = out / "outcome_vault" / direction / f"fold_{fold_index + 1:02d}.npz"
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256_file(path) != record["vault_sha256"]:
        raise SubjectLocationError("outcome vault hash failed during evaluation-only null stage")
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["test_subjects"], dtype=np.int64), np.asarray(data["delta_true"], dtype=np.float64)


def _source_pairing_null_fold(
    packet: Mapping[str, np.ndarray],
    model: ReducedRankModel,
    true: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    target = np.asarray(true, dtype=np.float64)
    if model.rank == 0:
        return np.full(replicates, np.sum(target * target), dtype=np.float64)
    q_train = np.asarray(packet["q_train"], dtype=np.float64)
    q_test = np.asarray(packet["q_test"], dtype=np.float64)
    d_train = np.asarray(packet["delta_train"], dtype=np.float64)
    kernel = model.q_train_centered @ model.q_train_centered.T
    system = kernel + model.ridge_lambda * np.eye(len(q_train), dtype=np.float64)
    influence = (q_test - model.q_mean) @ model.q_train_centered.T @ np.linalg.solve(system, np.eye(len(q_train)))
    scores = (d_train - model.d_mean) @ model.basis
    target_centered = target - model.d_mean
    target_projection = target_centered @ model.basis
    target_norm_sq = float(np.sum(target_centered * target_centered))
    output = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        permutation = rng.permutation(len(q_train))
        predicted_scores = influence @ scores[permutation]
        output[replicate] = (
            target_norm_sq
            + float(np.sum(predicted_scores * predicted_scores))
            - 2.0 * float(np.sum(predicted_scores * target_projection))
        )
    return output


def _target_pairing_null_fold(
    prediction: np.ndarray,
    true: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    estimate = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(true, dtype=np.float64)
    target_norm = float(np.sum(target * target))
    estimate_row_norm = np.sum(estimate * estimate, axis=1)
    output = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        permutation = _derangement(rng, len(target))
        output[replicate] = target_norm + float(np.sum(estimate_row_norm[permutation])) - 2.0 * float(np.sum(target * estimate[permutation]))
    return output


def _null_summary(values: Mapping[str, np.ndarray], observed: Mapping[str, float]) -> list[dict[str, Any]]:
    rows = []
    for label in ("FORWARD", "REVERSE", "POOLED"):
        array = np.asarray(values[label], dtype=np.float64)
        value = float(observed[label])
        rows.append({
            "scope": label,
            "observed_r2": value,
            "null_mean_r2": float(array.mean()),
            "null_median_r2": float(np.median(array)),
            "null_95_low": float(np.quantile(array, 0.025)),
            "null_95_high": float(np.quantile(array, 0.975)),
            "null_p_value": float((1 + np.count_nonzero(array >= value - 1e-15)) / (len(array) + 1)),
            "replicates": len(array),
        })
    return rows


def _evaluate_control_rows(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    grouped = {label: [row for row in rows if row["direction"] == label] for label in {row["direction"] for row in rows}}
    summary = {}
    for label, selected in grouped.items():
        summary[label] = {
            "r2_cond": _aggregate_r2(selected),
            "mean_error_gain": float(np.mean([row["error_gain"] for row in selected])),
            "positive_gain_count": int(np.count_nonzero([row["error_gain"] > 0 for row in selected])),
            "subject_count": len(selected),
        }
    return summary


def _run_norm_only_control(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out = output_root(root)
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for fold_index in range(6):
            packet_path = out / "input_packets" / direction / f"fold_{fold_index + 1:02d}.npz"
            packet = _load_input_packet(packet_path)
            _, _, primary_model_path, _ = _prediction_paths(out, direction, fold_index)
            primary_model = _load_model(primary_model_path)
            inner = []
            for q_train, d_train, q_validation, d_validation in _inner_splits_from_packet(packet):
                inner.append((
                    np.linalg.norm(q_train, axis=1, keepdims=True), d_train,
                    np.linalg.norm(q_validation, axis=1, keepdims=True), d_validation,
                ))
            selection, _ = select_rank_ridge(inner, config, fixed_rank=primary_model.rank)
            model = fit_reduced_rank_ridge(
                np.linalg.norm(packet["q_train"], axis=1, keepdims=True),
                packet["delta_train"],
                primary_model.rank,
                selection["ridge_multiplier"],
            )
            estimate = predict_reduced_rank(model, np.linalg.norm(packet["q_test"], axis=1, keepdims=True))
            subjects, true = _load_true_fold(out, direction, fold_index)
            rows.extend(_subject_metric_rows(direction, subjects, true, estimate, np.full(len(subjects), fold_index + 1)))
            selections.append({
                "direction": direction,
                "fold": fold_index + 1,
                "fixed_output_rank": primary_model.rank,
                "selected_ridge_multiplier": selection["ridge_multiplier"],
            })
    return rows, selections


def _run_same_session_control(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    objects = load_parent_objects(root, config)
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for direction, (input_session, output_session) in SAME_SESSION_DIRECTIONS.items():
        for fold_index, test_indices in enumerate(objects.folds):
            train_indices = _outer_train_indices(len(objects.subjects), test_indices)
            outer = construct_fold_coordinates(objects, train_indices, test_indices, input_session, output_session, config, name=f"control_{direction}_outer{fold_index + 1}")
            inner_splits = []
            outer_train_set = set(train_indices.tolist())
            for inner_index, validation_indices in enumerate(objects.inner_folds[fold_index]):
                inner_train = np.asarray(sorted(outer_train_set - set(validation_indices.tolist())), dtype=np.int64)
                inner = construct_fold_coordinates(objects, inner_train, validation_indices, input_session, output_session, config, name=f"control_{direction}_outer{fold_index + 1}_inner{inner_index + 1}")
                inner_splits.append((inner["q_train"], inner["delta_train"], inner["q_test"], inner["delta_test"]))
            selection, _ = select_rank_ridge(inner_splits, config)
            model = fit_reduced_rank_ridge(outer["q_train"], outer["delta_train"], selection["effective_rank"], selection["ridge_multiplier"])
            estimate = predict_reduced_rank(model, outer["q_test"])
            rows.extend(_subject_metric_rows(direction, outer["test_subjects"], outer["delta_test"], estimate, np.full(len(test_indices), fold_index + 1)))
            selections.append({
                "direction": direction,
                "fold": fold_index + 1,
                "selected_rank": model.rank,
                "selected_ridge_multiplier": model.ridge_multiplier,
            })
    return rows, selections


def run_nulls_and_controls(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    state = read_scientific_state(root)
    if STATE_ORDER.index(state["current_state"]) >= STATE_ORDER.index("NULLS_COMPLETE"):
        summary_path = output_root(root) / "nulls" / "null_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "status": state["current_state"],
            "source_pairing": summary["source_pairing"],
            "target_pairing": summary["target_location_pairing"],
            "resumed_from_validated_artifacts": True,
        }
    if state["current_state"] in ("PRIMARY_PREDICTIONS_FROZEN", "TARGET_OUTCOMES_RELEASED_FOR_EVALUATION"):
        evaluation = release_target_outcomes_and_evaluate(root)
    else:
        raise SubjectLocationError("nulls require frozen predictions and released evaluation outcomes")
    out = output_root(root)
    primary = json.loads((out / "tables" / "09_pooled_primary_metrics.json").read_text(encoding="utf-8"))["summary"]
    replicates = int(config["inference"]["null_replicates"])
    source_sse = {direction: np.zeros(replicates, dtype=np.float64) for direction in DIRECTIONS}
    target_sse = {direction: np.zeros(replicates, dtype=np.float64) for direction in DIRECTIONS}
    denominator = {direction: 0.0 for direction in DIRECTIONS}
    for direction in DIRECTIONS:
        for fold_index in range(6):
            packet = _load_input_packet(out / "input_packets" / direction / f"fold_{fold_index + 1:02d}.npz")
            prediction_path, record_path, model_path, _ = _prediction_paths(out, direction, fold_index)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if sha256_file(prediction_path) != record["prediction_sha256"]:
                raise SubjectLocationError("prediction mutated before null stage")
            with np.load(prediction_path, allow_pickle=False) as data:
                estimate = np.asarray(data["delta_hat"], dtype=np.float64)
                predicted_subjects = np.asarray(data["subject_ids"], dtype=np.int64)
            subjects, true = _load_true_fold(out, direction, fold_index)
            if not np.array_equal(subjects, predicted_subjects):
                raise SubjectLocationError("null subject ordering mismatch")
            model = _load_model(model_path)
            denominator[direction] += float(np.sum(true * true))
            source_sse[direction] += _source_pairing_null_fold(
                packet, model, true, replicates, _rng(config, "source_pairing", direction, fold_index)
            )
            target_sse[direction] += _target_pairing_null_fold(
                estimate, true, replicates, _rng(config, "target_pairing", direction, fold_index)
            )
    source_r2 = {direction: 1.0 - source_sse[direction] / denominator[direction] for direction in DIRECTIONS}
    target_r2 = {direction: 1.0 - target_sse[direction] / denominator[direction] for direction in DIRECTIONS}
    total_denominator = sum(denominator.values())
    source_r2["POOLED"] = 1.0 - sum(source_sse.values()) / total_denominator
    target_r2["POOLED"] = 1.0 - sum(target_sse.values()) / total_denominator
    observed = {label: float(primary[label]["r2_cond"]) for label in ("FORWARD", "REVERSE", "POOLED")}
    source_rows = _null_summary(source_r2, observed)
    target_rows = _null_summary(target_r2, observed)
    null_path = out / "nulls" / "cross_session_nulls.npz"
    atomic_savez(
        null_path,
        source_pairing_forward=source_r2["FORWARD"],
        source_pairing_reverse=source_r2["REVERSE"],
        source_pairing_pooled=source_r2["POOLED"],
        target_pairing_forward=target_r2["FORWARD"],
        target_pairing_reverse=target_r2["REVERSE"],
        target_pairing_pooled=target_r2["POOLED"],
    )
    source_path = out / "tables" / "12_source_pairing_null_summary.csv"
    target_path = out / "tables" / "13_target_location_pairing_null_summary.csv"
    write_csv(source_path, source_rows)
    write_csv(target_path, target_rows)
    norm_rows, norm_selection = _run_norm_only_control(root, config)
    norm_path = out / "tables" / "14_location_norm_only_control.csv"
    write_csv(norm_path, norm_rows)
    norm_summary_path = out / "controls" / "location_norm_only_summary.json"
    atomic_write_json(norm_summary_path, {"summary": _evaluate_control_rows(norm_rows, config), "selections": norm_selection, "non_voting": True})
    same_rows, same_selection = _run_same_session_control(root, config)
    same_path = out / "tables" / "15_same_session_control.csv"
    write_csv(same_path, same_rows)
    same_summary_path = out / "controls" / "same_session_summary.json"
    atomic_write_json(same_summary_path, {"summary": _evaluate_control_rows(same_rows, config), "selections": same_selection, "non_voting": True})
    null_summary_path = out / "nulls" / "null_summary.json"
    null_summary = {
        "source_pairing": source_rows,
        "target_location_pairing": target_rows,
        "replicates": replicates,
        "rank_ridge_fixed_from_true_source_selection": True,
        "target_results_used_for_selection": False,
    }
    null_summary["content_sha256"] = canonical_sha256(null_summary)
    atomic_write_json(null_summary_path, null_summary)
    artifacts = [
        *evaluation.get("artifacts", []), null_path, source_path, target_path, norm_path,
        norm_summary_path, same_path, same_summary_path, null_summary_path,
    ]
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "phase": "NULLS_COMPLETE",
        "null_replicates": replicates,
        "null_summary_sha256": sha256_file(null_summary_path),
        "target_outcomes_used_for_model_selection": False,
    })
    atomic_write_json(manifest_path, manifest)
    completed = transition_state(
        root, config, "NULLS_COMPLETE",
        gates=("target_outcomes_evaluation_only", "source_pairing_null_1999", "target_location_derangement_null_1999", "norm_only_control", "same_session_control"),
        artifacts=artifacts,
    )
    return {
        "status": completed["current_state"],
        "source_pairing": source_rows,
        "target_pairing": target_rows,
        "norm_only": _evaluate_control_rows(norm_rows, config),
        "same_session": _evaluate_control_rows(same_rows, config),
        "null_path": str(null_path),
    }


def _posthoc_geometry_diagnostics(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    out = output_root(root)
    objects = load_parent_objects(root, config)
    m0_all, m0_all_audit = _fit_airm_mean(objects.subject_references, config, "M0_all_posthoc")
    reference_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    for fold_index, test_indices in enumerate(objects.folds):
        train_indices = _outer_train_indices(len(objects.subjects), test_indices)
        packet = _load_input_packet(out / "input_packets" / "FORWARD" / f"fold_{fold_index + 1:02d}.npz")
        m0_train = np.asarray(packet["m0"], dtype=np.float64)
        shift = float(airm_distance(m0_train, m0_all))
        dispersion = float(np.mean(airm_distance(objects.subject_references[train_indices], m0_train)))
        reference_rows.append({
            "fold": fold_index + 1,
            "source_subject_count": len(train_indices),
            "heldout_subject_count": len(test_indices),
            "distance_train_to_all": shift,
            "source_reference_dispersion": dispersion,
            "normalized_reference_shift": shift / dispersion,
            "m0_train_karcher_residual": karcher_residual(objects.subject_references[train_indices], m0_train),
            "m0_all_karcher_residual": m0_all_audit["karcher_residual"],
            "all_subject_reference_used_for_prediction": False,
        })
        q2 = _coordinate(m0_train, objects.marginal[test_indices, 0])
        q3 = _coordinate(m0_train, objects.marginal[test_indices, 1])
        for local, subject_index in enumerate(test_indices):
            q_rows.append({
                "subject": int(objects.subjects[subject_index]),
                "outer_fold": fold_index + 1,
                "q2_norm": float(np.linalg.norm(q2[local])),
                "q3_norm": float(np.linalg.norm(q3[local])),
                "q_cross_session_cosine": _safe_cosine(q2[local], q3[local]),
                "q_tangent_distance": float(np.linalg.norm(q2[local] - q3[local])),
                "marginal_airm_distance": float(airm_distance(objects.marginal[subject_index, 0], objects.marginal[subject_index, 1])),
            })
        for direction in DIRECTIONS:
            direction_packet = _load_input_packet(out / "input_packets" / direction / f"fold_{fold_index + 1:02d}.npz")
            centered = direction_packet["q_train"] - direction_packet["q_train"].mean(axis=0)
            singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
            energy = singular * singular
            total = float(energy.sum())
            for component, value in enumerate(energy[:13], 1):
                spectrum_rows.append({
                    "direction": direction,
                    "fold": fold_index + 1,
                    "component": component,
                    "variance_fraction": float(value / total),
                    "cumulative_variance_fraction": float(energy[:component].sum() / total),
                })
    q2_all = _coordinate(m0_all, objects.marginal[:, 0])
    q3_all = _coordinate(m0_all, objects.marginal[:, 1])
    distances = np.linalg.norm(q2_all[:, None] - q3_all[None], axis=2)
    top1_forward = float(np.mean(np.argmin(distances, axis=1) == np.arange(62)))
    top1_reverse = float(np.mean(np.argmin(distances, axis=0) == np.arange(62)))
    reference_path = out / "tables" / "04_fold_to_full_barycenter_stability.csv"
    q_path = out / "tables" / "05_q_cross_session_reliability.csv"
    spectrum_path = out / "tables" / "q_source_principal_spectrum.csv"
    write_csv(reference_path, reference_rows)
    write_csv(q_path, q_rows)
    write_csv(spectrum_path, spectrum_rows)
    identity_path = out / "tables" / "q_subject_identification_descriptive.json"
    atomic_write_json(identity_path, {
        "top1_session2_to_session3": top1_forward,
        "top1_session3_to_session2": top1_reverse,
        "reference": "posthoc_all_subject_M0",
        "voting": False,
        "interpretation": "identity descriptively only; not conditional prediction evidence",
    })
    delta_rows: list[dict[str, Any]] = []
    for direction, (_, output_session) in DIRECTIONS.items():
        for fold_index in range(6):
            subjects, true = _load_true_fold(out, direction, fold_index)
            reshaped = true.reshape(len(true), 4, 210)
            for subject, classes in zip(subjects, reshaped, strict=True):
                for class_name, value in zip(CLASS_NAMES, classes, strict=True):
                    delta_rows.append({
                        "direction": direction,
                        "session": output_session,
                        "subject": int(subject),
                        "class": class_name,
                        "delta_norm": float(np.linalg.norm(value)),
                    })
    delta_path = out / "tables" / "delta_norm_by_subject_class_session.csv"
    write_csv(delta_path, delta_rows)
    subject_metric_rows = list(csv.DictReader((out / "tables" / "10_subject_level_error_gains.csv").open(encoding="utf-8")))
    gain_by_subject: dict[int, list[float]] = {subject: [] for subject in range(1, 63)}
    for row in subject_metric_rows:
        gain_by_subject[int(row["subject"])].append(float(row["error_gain"]))
    q_cosine = np.asarray([row["q_cross_session_cosine"] for row in sorted(q_rows, key=lambda row: row["subject"])])
    gain = np.asarray([np.mean(gain_by_subject[subject]) for subject in range(1, 63)])
    spearman = stats.spearmanr(q_cosine, gain)
    relation_path = out / "tables" / "q_repeatability_prediction_gain.json"
    atomic_write_json(relation_path, {
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "voting": False,
    })
    return {
        "m0_all": m0_all,
        "reference_rows": reference_rows,
        "q_rows": q_rows,
        "spectrum_rows": spectrum_rows,
        "delta_rows": delta_rows,
        "identity": {"forward": top1_forward, "reverse": top1_reverse},
        "q_gain_relation": {"rho": float(spearman.statistic), "p": float(spearman.pvalue)},
        "artifacts": [reference_path, q_path, spectrum_path, identity_path, delta_path, relation_path],
    }


def _terminal_decision(out: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    primary = json.loads((out / "tables" / "09_pooled_primary_metrics.json").read_text(encoding="utf-8"))["summary"]
    nulls = json.loads((out / "nulls" / "null_summary.json").read_text(encoding="utf-8"))
    source_pooled = next(row for row in nulls["source_pairing"] if row["scope"] == "POOLED")
    with (out / "tables" / "10_subject_level_error_gains.csv").open(encoding="utf-8") as stream:
        subject_rows = list(csv.DictReader(stream))
    positive_unit_count = sum(float(row["error_gain"]) > 0.0 for row in subject_rows)
    positive_unit_fraction = positive_unit_count / len(subject_rows)
    gates = {
        "pooled_r2_at_least_0_05": primary["POOLED"]["r2_cond"] >= 0.05,
        "forward_r2_positive": primary["FORWARD"]["r2_cond"] > 0.0,
        "reverse_r2_positive": primary["REVERSE"]["r2_cond"] > 0.0,
        "pooled_bootstrap_lower_positive": primary["POOLED"]["r2_bootstrap_95_low"] > 0.0,
        "source_pairing_null_p_at_most_0_05": source_pooled["null_p_value"] <= 0.05,
        "subject_gain_sign_flip_p_at_most_0_05": primary["POOLED"]["sign_flip_p"] <= 0.05,
        "positive_subject_direction_fraction_at_least_0_60": positive_unit_fraction >= 0.60,
        "leave_one_subject_minimum_positive": primary["POOLED"]["leave_one_subject_r2_min"] > 0.0,
        "all_engineering_gates": True,
    }
    go = all(gates.values())
    inferential = any((
        gates["pooled_bootstrap_lower_positive"],
        gates["source_pairing_null_p_at_most_0_05"],
        gates["subject_gain_sign_flip_p_at_most_0_05"],
    ))
    if go:
        terminal = "GO_LOCATION_PREDICTS_CROSS_SESSION_CONDITIONAL_CONFIGURATION"
        next_question = (
            "Does geometry-derived subject-location conditioning reduce conditional-shift-induced decoding error "
            "without target labels, and can low-dimensional TTA refinement improve it beyond feed-forward conditioning?"
        )
    elif primary["POOLED"]["r2_cond"] > 0.0 and inferential:
        terminal = "WEAK_LOCATION_CONDITIONAL_ASSOCIATION_NOT_METHOD_READY"
        next_question = (
            "Which component of subject location, if any, provides the unstable cross-session conditional "
            "association, and is its effect large enough to justify a bounded conditioning pilot?"
        )
    else:
        terminal = "STOP_LOCATION_DOES_NOT_PREDICT_CONDITIONAL_CONFIGURATION"
        next_question = (
            "What target-observable statistic other than global subject location is required to identify "
            "unseen-subject class-conditional deformation?"
        )
    return {
        "terminal": terminal,
        "gates": gates,
        "positive_subject_direction_count": positive_unit_count,
        "subject_direction_unit_count": len(subject_rows),
        "positive_subject_direction_fraction": positive_unit_fraction,
        "source_pairing_pooled_null_p": source_pooled["null_p_value"],
        "exact_next_question": next_question,
        "automatic_followup_executed": False,
    }


def _generate_figures(out: Path, diagnostics: Mapping[str, Any], decision: Mapping[str, Any]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    path = figures / "01_scientific_design.png"
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.axis("off")
    boxes = [(0.04, "source-only\nAIRM M0"), (0.28, "label-free q\n(session u)"), (0.52, "reduced-rank\ndual ridge"), (0.76, "predicted Δ\n(session v)")]
    for x, text in boxes:
        ax.text(x, 0.5, text, transform=ax.transAxes, ha="left", va="center", bbox={"boxstyle": "round", "facecolor": "#eaf2f8"})
    for x in (0.21, 0.45, 0.69): ax.annotate("", xy=(x + 0.06, 0.5), xytext=(x, 0.5), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->"})
    ax.set_title("Population reference → label-free subject location → cross-session conditional configuration")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    path = figures / "02_population_reference_stability.png"
    values = diagnostics["reference_rows"]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar([row["fold"] for row in values], [row["normalized_reference_shift"] for row in values], color="#4c78a8")
    ax.set(xlabel="Outer fold", ylabel="Normalized reference shift", title="Fold-specific source reference vs all-subject post-hoc reference")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    path = figures / "03_q_session_geometry.png"
    q_rows = diagnostics["q_rows"]
    fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter([row["q2_norm"] for row in q_rows], [row["q3_norm"] for row in q_rows], alpha=.75)
    ax.set(xlabel="||q(session 2)||", ylabel="||q(session 3)||", title="Held-out subject location norms")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    with (out / "tables" / "10_subject_level_error_gains.csv").open(encoding="utf-8") as stream:
        subject_rows = list(csv.DictReader(stream))
    path = figures / "04_true_predicted_delta_cosine_error.png"
    fig, ax = plt.subplots(figsize=(6, 4));
    for direction, color in (("FORWARD", "#4c78a8"), ("REVERSE", "#f58518")):
        rows = [row for row in subject_rows if row["direction"] == direction]
        ax.scatter([float(row["cosine"]) for row in rows], [float(row["normalized_sse_ratio"]) for row in rows], label=direction, alpha=.7, color=color)
    ax.axhline(1.0, color="black", linewidth=.8); ax.set(xlabel="cosine(Δ, Δ-hat)", ylabel="Normalized SSE ratio", title="Held-out conditional prediction"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    path = figures / "05_per_subject_error_gains.png"
    gain = {subject: [] for subject in range(1, 63)}
    for row in subject_rows: gain[int(row["subject"])].append(float(row["error_gain"]))
    fig, ax = plt.subplots(figsize=(11, 4)); ax.bar(range(1, 63), [np.mean(gain[s]) for s in range(1, 63)], color=["#59a14f" if np.mean(gain[s]) > 0 else "#e15759" for s in range(1, 63)])
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Subject", ylabel="Mean error gain", title="Paired cross-session subject error gains")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    primary = json.loads((out / "tables" / "09_pooled_primary_metrics.json").read_text(encoding="utf-8"))["summary"]
    path = figures / "06_forward_reverse_r2.png"
    labels = ["FORWARD", "REVERSE", "POOLED"]; vals = [primary[label]["r2_cond"] for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(labels, vals, color=["#4c78a8", "#f58518", "#54a24b"]); ax.axhline(0, color="black", linewidth=.8); ax.set(ylabel="Conditional R²", title="Cross-session conditional prediction")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    with (out / "tables" / "06_selected_ranks_and_ridge_values.csv").open(encoding="utf-8") as stream:
        selected = list(csv.DictReader(stream))
    ranks = [int(row["selected_rank"]) for row in selected]
    path = figures / "07_selected_rank_distribution.png"
    unique = sorted(set(ranks)); fig, ax = plt.subplots(figsize=(6, 4)); ax.bar([str(value) for value in unique], [ranks.count(value) for value in unique], color="#b279a2"); ax.set(xlabel="Selected rank", ylabel="Fold-direction count", title="Source-only selected output rank")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    with np.load(out / "nulls" / "cross_session_nulls.npz", allow_pickle=False) as null_data:
        null_values = np.asarray(null_data["source_pairing_pooled"])
    path = figures / "08_source_pairing_null.png"
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(null_values, bins=40, color="#9c755f", alpha=.8); ax.axvline(primary["POOLED"]["r2_cond"], color="red", label="observed"); ax.set(xlabel="Pooled conditional R²", ylabel="Count", title="Source-pairing null (1,999)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    same = json.loads((out / "controls" / "same_session_summary.json").read_text(encoding="utf-8"))["summary"]
    path = figures / "09_same_vs_cross_session.png"
    control_labels = ["FORWARD", "REVERSE", "SAME_SESSION_2", "SAME_SESSION_3"]
    control_values = [primary["FORWARD"]["r2_cond"], primary["REVERSE"]["r2_cond"], same["SAME_SESSION_2"]["r2_cond"], same["SAME_SESSION_3"]["r2_cond"]]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(control_labels, control_values, color=["#4c78a8", "#f58518", "#bab0ac", "#bab0ac"]); ax.axhline(0, color="black", linewidth=.8); ax.tick_params(axis="x", rotation=15); ax.set(ylabel="Conditional R²", title="Cross-session primary vs same-session control")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    path = figures / "10_terminal_summary.png"
    fig, ax = plt.subplots(figsize=(10, 3)); ax.axis("off"); ax.text(.5, .65, decision["terminal"], ha="center", va="center", fontsize=14, weight="bold", transform=ax.transAxes); ax.text(.5, .30, "No classifier, neural network, TTA, or follow-up experiment", ha="center", transform=ax.transAxes); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    return paths


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "|" + "|".join("---" for _ in fields) + "|"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _write_final_report(out: Path, decision: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> Path:
    primary = json.loads((out / "tables" / "09_pooled_primary_metrics.json").read_text(encoding="utf-8"))["summary"]
    nulls = json.loads((out / "nulls" / "null_summary.json").read_text(encoding="utf-8"))
    norm = json.loads((out / "controls" / "location_norm_only_summary.json").read_text(encoding="utf-8"))["summary"]
    same = json.loads((out / "controls" / "same_session_summary.json").read_text(encoding="utf-8"))["summary"]
    with (out / "tables" / "06_selected_ranks_and_ridge_values.csv").open(encoding="utf-8") as stream:
        selections = list(csv.DictReader(stream))
    source_null = nulls["source_pairing"]
    target_null = nulls["target_location_pairing"]
    primary_rows = []
    for label in ("FORWARD", "REVERSE", "POOLED"):
        value = primary[label]
        primary_rows.append({
            "scope": label,
            "R2_cond": f"{value['r2_cond']:.6f}",
            "bootstrap_95_CI": f"[{value['r2_bootstrap_95_low']:.6f}, {value['r2_bootstrap_95_high']:.6f}]",
            "mean_error_gain": f"{value['mean_subject_error_gain']:.6g}",
            "positive": f"{value['positive_gain_count']}/{value['unit_count']}",
            "sign_flip_p": f"{value['sign_flip_p']:.6f}",
        })
    text = f"""# Cross-Session Subject Location → Conditional Configuration Prediction V0

Terminal: **{decision['terminal']}**

This deterministic held-out-subject analysis uses only the locked PR #19 Stieger2021 task-3 full-split compact geometry. No raw EEG was opened, no classifier or neural network was trained, and no TTA was performed.

## Exact objects

- Subject reference: `R_s = FM_AIRM(M_s^(2), M_s^(3))`.
- Fold source reference: `M_0 = FM_AIRM({{R_s : s in S_train}})` with one vote per source subject.
- Label-free input: `q_s^(u) = svec(log(M_0^(-1/2) M_s^(u) M_0^(-1/2)))`.
- Equal-class centered configuration: `d_s,c^(v) = z_s,c^(v) - (1/4) sum_j z_s,j^(v)`.
- Source-centered target: `Delta_s,c^(v) = d_s,c^(v) - mean_source d_s,c^(v)`.

## Primary held-out results

{_markdown_table(primary_rows, ['scope', 'R2_cond', 'bootstrap_95_CI', 'mean_error_gain', 'positive', 'sign_flip_p'])}

The pooled leave-one-subject R² range is [{primary['POOLED']['leave_one_subject_r2_min']:.6f}, {primary['POOLED']['leave_one_subject_r2_max']:.6f}]. The subject-direction positive-gain count is {decision['positive_subject_direction_count']}/{decision['subject_direction_unit_count']}.

## Selected source-only models

{_markdown_table(selections, ['direction', 'fold', 'selected_rank', 'selected_ridge_multiplier', 'ridge_lambda', 'output_numerical_rank'])}

## Required nulls

Source-pairing null:

{_markdown_table(source_null, ['scope', 'observed_r2', 'null_mean_r2', 'null_95_low', 'null_95_high', 'null_p_value', 'replicates'])}

Held-out target-location derangement null:

{_markdown_table(target_null, ['scope', 'observed_r2', 'null_mean_r2', 'null_95_low', 'null_95_high', 'null_p_value', 'replicates'])}

## Non-voting controls

- Location-norm-only: {json.dumps(norm, sort_keys=True)}
- Same-session: {json.dumps(same, sort_keys=True)}
- Post-hoc all-subject reference was never used for prediction or nulls. Maximum normalized fold-reference shift: {max(row['normalized_reference_shift'] for row in diagnostics['reference_rows']):.6f}.
- Descriptive q subject identification: session2→3 {diagnostics['identity']['forward']:.3f}, session3→2 {diagnostics['identity']['reverse']:.3f}. This is not evidence for conditional prediction.
- q repeatability vs paired prediction gain Spearman rho: {diagnostics['q_gain_relation']['rho']:.6f} (p={diagnostics['q_gain_relation']['p']:.6f}).

## Terminal gates

{_markdown_table([{'gate': key, 'passed': value} for key, value in decision['gates'].items()], ['gate', 'passed'])}

## Exact next question

> {decision['exact_next_question']}

The next question is recorded only. No follow-up architecture or experiment was executed.
"""
    path = out / "report" / "SUBJECT_LOCATION_CONDITIONAL_CONFIGURATION_V0.md"
    atomic_write_bytes(path, text.encode("utf-8"))
    return path


def _write_required_final_tables(out: Path, decision: Mapping[str, Any]) -> list[Path]:
    with (out / "tables" / "10_subject_level_error_gains.csv").open(encoding="utf-8") as stream:
        subject_rows = list(csv.DictReader(stream))
    forward_path = out / "tables" / "07_forward_heldout_predictions.csv"
    reverse_path = out / "tables" / "08_reverse_heldout_predictions.csv"
    write_csv(forward_path, [row for row in subject_rows if row["direction"] == "FORWARD"])
    write_csv(reverse_path, [row for row in subject_rows if row["direction"] == "REVERSE"])

    gates = json.loads((out / "protocol" / "synthetic_gates.json").read_text(encoding="utf-8"))
    synthetic_path = out / "tables" / "17_synthetic_gates.csv"
    write_csv(synthetic_path, [
        {
            "gate": name,
            "passed": value["passed"],
            "details": json.dumps({key: item for key, item in value.items() if key != "passed"}, sort_keys=True),
        }
        for name, value in gates["cases"].items()
    ])
    terminal_path = out / "tables" / "18_terminal_gate_table.csv"
    write_csv(terminal_path, [
        {"gate": name, "passed": passed, "terminal": decision["terminal"]}
        for name, passed in decision["gates"].items()
    ])
    return [forward_path, reverse_path, synthetic_path, terminal_path]


def _build_artifact_index(root: Path, decision: Mapping[str, Any]) -> tuple[Path, Path]:
    out = output_root(root)
    index_path = out / "artifact_index.json"
    manifest_path = out / "manifest.json"
    records = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path in (index_path, manifest_path):
            continue
        records.append({
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    index = {
        "namespace": OUTPUT_NAME,
        "artifact_count": len(records),
        "artifacts": records,
        "target_outcomes_inside_prediction_artifacts": False,
        "content_sha256": canonical_sha256(records),
    }
    atomic_write_json(index_path, index)
    manifest = {
        **_initial_manifest(load_config(root)[0], "STOPPED"),
        "terminal": decision["terminal"],
        "exact_next_question": decision["exact_next_question"],
        "artifact_index_path": str(index_path.relative_to(root)),
        "artifact_index_sha256": sha256_file(index_path),
        "artifact_count": len(records),
        "prediction_manifest_sha256": sha256_file(out / "predictions" / "prediction_manifest.json"),
        "locked_packet_manifest_sha256": sha256_file(out / "objects" / "locked_packet_manifest.json"),
        "report_sha256": sha256_file(out / "report" / "SUBJECT_LOCATION_CONDITIONAL_CONFIGURATION_V0.md"),
        "scientific_state_sha256": sha256_file(out / "scientific_state.json"),
        "null_replicates": 1999,
        "bootstrap_replicates": 10000,
        "complete_forward_reverse_folds": True,
        "target_outcomes_used_for_model_selection": False,
        "followup_experiment_executed": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    return index_path, manifest_path


def validate_output_manifest(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = output_root(root)
    manifest_path = out / "manifest.json"
    index_path = out / "artifact_index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_manifest_content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if canonical_sha256(expected_manifest_content) != manifest.get("content_sha256"):
        raise SubjectLocationError("output manifest content hash mismatch")
    if sha256_file(index_path) != manifest.get("artifact_index_sha256"):
        raise SubjectLocationError("artifact-index file hash mismatch")
    if canonical_sha256(index.get("artifacts", [])) != index.get("content_sha256"):
        raise SubjectLocationError("artifact-index content hash mismatch")
    if len(index.get("artifacts", [])) != int(index.get("artifact_count", -1)):
        raise SubjectLocationError("artifact-index count mismatch")
    for record in index["artifacts"]:
        path = root / str(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"] or path.stat().st_size != int(record["bytes"]):
            raise SubjectLocationError(f"output artifact validation failed: {path}")
    prediction = validate_prediction_hashes(root, require_complete=True)
    state = read_scientific_state(root)
    if state["current_state"] != "STOPPED" or state["terminal"] != manifest["terminal"]:
        raise SubjectLocationError("final scientific state and manifest disagree")
    with np.load(out / "nulls" / "cross_session_nulls.npz", allow_pickle=False) as nulls:
        if any(len(nulls[key]) != 1999 for key in nulls.files):
            raise SubjectLocationError("null replicate count mismatch")
    return {
        "status": "PASS",
        "artifact_count": index["artifact_count"],
        "prediction_count": prediction["prediction_count"],
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_index_sha256": sha256_file(index_path),
        "terminal": manifest["terminal"],
    }


def generate_final_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    state = read_scientific_state(root)
    if state["current_state"] == "STOPPED":
        result = validate_output_manifest(root)
        result["resumed_from_validated_artifacts"] = True
        return result
    if state["current_state"] != "NULLS_COMPLETE":
        raise SubjectLocationError("final report requires NULLS_COMPLETE")
    out = output_root(root)
    diagnostics = _posthoc_geometry_diagnostics(root, config)
    decision = _terminal_decision(out, config)
    decision_path = out / "decisions" / "terminal.json"
    decision_payload = dict(decision)
    decision_payload["content_sha256"] = canonical_sha256(decision_payload)
    atomic_write_json(decision_path, decision_payload)
    required_tables = _write_required_final_tables(out, decision)
    report_path = _write_final_report(out, decision, diagnostics)
    figure_paths = _generate_figures(out, diagnostics, decision)
    terminal_state = transition_state(
        root,
        config,
        "TERMINAL_WRITTEN",
        gates=(
            "posthoc_reference_stability_non_voting",
            "all_required_tables",
            "all_required_figures",
            "registered_terminal_rule",
            "exact_next_question_recorded_only",
        ),
        artifacts=(*diagnostics["artifacts"], *required_tables, decision_path, report_path, *figure_paths),
        terminal=decision["terminal"],
        exact_next_question=decision["exact_next_question"],
    )
    stopped_state = transition_state(
        root,
        config,
        "STOPPED",
        gates=(
            "no_raw_eeg_opened",
            "no_classifier_or_neural_network_trained",
            "no_tta_performed",
            "no_pr20_pr21_code_inherited",
            "no_automatic_merge",
            "no_followup_experiment",
        ),
        terminal=decision["terminal"],
        exact_next_question=decision["exact_next_question"],
    )
    index_path, manifest_path = _build_artifact_index(root, decision)
    validation = validate_output_manifest(root)
    return {
        **validation,
        "status": stopped_state["current_state"],
        "previous_state": terminal_state["current_state"],
        "report_path": str(report_path.relative_to(root)),
        "decision_path": str(decision_path.relative_to(root)),
        "manifest_path": str(manifest_path.relative_to(root)),
        "artifact_index_path": str(index_path.relative_to(root)),
        "exact_next_question": decision["exact_next_question"],
    }
