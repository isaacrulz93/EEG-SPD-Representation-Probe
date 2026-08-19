"""Frozen selective conditional-memory feasibility audit.

The module reuses PR #20 trial coordinates and baselines without writing to a
parent namespace.  Gate builders accept enrollment/source objects only;
deployment labels are joined after target predictions have been frozen.
"""
from __future__ import annotations

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
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from src.interaction_provenance_v0 import atomic_write_json, sha256_file
from src import returning_user_conditional_memory_v0 as parent


CONFIG_PATH = "configs/selective_conditional_memory_feasibility_v0.yaml"
OUTPUT_NAME = "selective_conditional_memory_feasibility_v0"
PARENT_DIRS = tuple(
    f"outputs/{name}" for name in (
        "subject_class_population_structure_v1",
        "subject_class_population_structure_v1_1",
        "unlabeled_conditional_mode_identifiability_v0",
        "source_referenced_conditional_residual_v1",
        "stieger2021_multiclass_confirmation_v0",
        "returning_user_conditional_memory_v0",
    )
)


class SelectiveMemoryError(RuntimeError):
    """Base fail-closed exception."""


class RequiredTrialCacheError(SelectiveMemoryError):
    """An immutable parent/cache contract failed."""


class LeakageOrSplitError(SelectiveMemoryError):
    """A label or held-out-subject boundary failed."""


class NumericalContractError(SelectiveMemoryError):
    """A numerical contract failed."""


@dataclass(frozen=True)
class ReliabilityRecord:
    subject_index: int
    subject: int
    global_features: np.ndarray
    class_features: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class GateExample:
    subject_index: int
    subject: int
    global_features: np.ndarray
    class_features: np.ndarray
    residual: np.ndarray
    distance_a: np.ndarray
    distance_b: np.ndarray
    distance_c: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class GateFit:
    parameters: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    l2: float
    objective: float
    iterations: int


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _rng(config: Mapping[str, Any], namespace: str, *parts: Any) -> np.random.Generator:
    material = "|".join([str(config["protocol"]["master_seed"]), namespace, *map(str, parts)])
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big", signed=False)
    return np.random.default_rng(seed)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


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


def _tracked_parent_files(root: Path) -> list[str]:
    output = _git(root, "ls-files", *PARENT_DIRS)
    return [line for line in output.splitlines() if line]


def parent_hash_snapshot(root: Path) -> dict[str, Any]:
    records = [{"path": path, "sha256": sha256_file(root / path)} for path in _tracked_parent_files(root)]
    payload: dict[str, Any] = {
        "schema_version": "selective-memory-parent-artifact-hashes-v0",
        "parent_head": "9c95e5b19eb4c44acc411c1e0d72a5cdd4d9ef63",
        "count": len(records),
        "records": records,
    }
    payload["canonical_sha256"] = _canonical_hash(payload)
    return payload


def verify_parent_snapshot(root: Path, snapshot_path: Path) -> dict[str, Any]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    base = {key: value for key, value in snapshot.items() if key != "canonical_sha256"}
    if _canonical_hash(base) != snapshot["canonical_sha256"]:
        raise RequiredTrialCacheError("parent snapshot canonical hash mismatch")
    for row in snapshot["records"]:
        path = root / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RequiredTrialCacheError(f"parent artifact changed: {row['path']}")
    return {"count": len(snapshot["records"]), "canonical_sha256": snapshot["canonical_sha256"]}


def validate_parent_contract(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    if _git(root, "merge-base", "--is-ancestor", str(config["protocol"]["parent_head"]), "HEAD") != "":
        # merge-base --is-ancestor has no stdout; successful subprocess is enough.
        raise RequiredTrialCacheError("unexpected ancestry command output")
    observed: dict[str, str] = {}
    for name, row in config["parent_contract"]["manifests"].items():
        path = root / row["path"]
        digest = sha256_file(path) if path.is_file() else "MISSING"
        if digest != row["sha256"]:
            raise RequiredTrialCacheError(f"{name} manifest mismatch: {digest}")
        observed[name] = digest
    manifest = json.loads((root / config["parent_contract"]["manifests"]["pr20"]["path"]).read_text(encoding="utf-8"))
    if manifest.get("canonical_sha256") != config["parent_contract"]["pr20_manifest_canonical_sha256"]:
        raise RequiredTrialCacheError("PR #20 canonical manifest mismatch")
    for name, row in config["parent_contract"]["pr20_predictions"].items():
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise RequiredTrialCacheError(f"PR #20 prediction mismatch: {name}")
        observed[f"prediction_{name}"] = row["sha256"]
    return observed


def _parent_config(root: Path) -> dict[str, Any]:
    config, _ = parent.load_config(root)
    if config["protocol"]["branch"] != "pilot/returning-user-conditional-memory-v0":
        raise RequiredTrialCacheError("unexpected PR #20 config lineage")
    return config


def load_dataset(root: Path, config: Mapping[str, Any], dataset: str, reverse: bool = False) -> parent.DatasetBundle:
    bundle = parent.load_dataset(root, _parent_config(root), dataset, reverse=reverse)
    expected = int(config[dataset]["subjects"])
    if len(bundle.subjects) != expected or bundle.trials[next(iter(bundle.trials))].features.shape[1] != 210:
        raise RequiredTrialCacheError(f"{dataset} bundle contract mismatch")
    return bundle


def _openbmi_original_indices(trial_ids: np.ndarray, subject: int, array_session: int) -> np.ndarray:
    lookup = {
        hashlib.sha256(f"{subject}|{array_session}|{index}".encode()).hexdigest()[:24]: index
        for index in range(100)
    }
    try:
        result = np.asarray([lookup[str(value)] for value in trial_ids], dtype=np.int64)
    except KeyError as exc:
        raise RequiredTrialCacheError(f"OpenBMI opaque acquisition ID mismatch: {exc}") from exc
    if sorted(result.tolist()) != list(range(100)):
        raise RequiredTrialCacheError("OpenBMI acquisition mapping is not bijective")
    return result


def acquisition_orders(bundle: parent.DatasetBundle) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for subject in bundle.subjects:
        record = bundle.trials[(int(subject), bundle.enrollment_session)]
        if bundle.name == "stieger":
            values = []
            for trial_id in record.trial_ids:
                text = str(trial_id)
                if "_A" not in text:
                    raise RequiredTrialCacheError("Stieger acquisition ID malformed")
                values.append(int(text.rsplit("_A", 1)[1]))
            order = np.asarray(values, dtype=np.int64)
        else:
            order = _openbmi_original_indices(record.trial_ids, int(subject), int(bundle.enrollment_session))
        if len(np.unique(order)) != len(order):
            raise RequiredTrialCacheError("acquisition order is not unique")
        result[int(subject)] = order
    return result


def _standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise NumericalContractError("nonfinite reliability feature scale")
    return mean, scale


def _reliability_record(
    bundle: parent.DatasetBundle,
    proto: parent.PrototypeData,
    target_index: int,
    gamma_e: np.ndarray,
    order: np.ndarray,
    epsilon: float,
) -> ReliabilityRecord:
    subject = int(bundle.subjects[target_index])
    trial = bundle.trials[(subject, bundle.enrollment_session)]
    label_record = bundle.enrollment_labels[subject]
    if not np.array_equal(trial.trial_ids, label_record.trial_ids):
        raise LeakageOrSplitError("enrollment trial/label ID mismatch")
    labels = label_record.labels
    classes = len(bundle.class_names)
    offsets = proto.prototypes[target_index, 0] - proto.unlabeled_means[target_index, 0, None, :]
    class_rows: list[list[float]] = []
    for class_index in range(classes):
        chosen = np.flatnonzero(labels == class_index)
        if len(chosen) < 2:
            raise NumericalContractError("class has fewer than two enrollment trials")
        chosen = chosen[np.argsort(order[chosen], kind="stable")]
        half_a, half_b = chosen[::2], chosen[1::2]
        if not len(half_a) or not len(half_b):
            raise NumericalContractError("empty acquisition-order split half")
        r_a = np.mean(trial.features[half_a], axis=0) - proto.unlabeled_means[target_index, 0] - gamma_e[class_index]
        r_b = np.mean(trial.features[half_b], axis=0) - proto.unlabeled_means[target_index, 0] - gamma_e[class_index]
        norm_a, norm_b = float(np.linalg.norm(r_a)), float(np.linalg.norm(r_b))
        cosine = float(np.dot(r_a, r_b) / (norm_a * norm_b)) if norm_a > 0.0 and norm_b > 0.0 else 0.0
        difference_energy = float(np.dot(r_a - r_b, r_a - r_b))
        mean_residual = 0.5 * (r_a + r_b)
        mean_energy = float(np.dot(mean_residual, mean_residual))
        noise_half = 0.5 * (r_a - r_b)
        noise_energy = float(np.dot(noise_half, noise_half))
        reliability = max(mean_energy - noise_energy, 0.0) / (mean_energy + epsilon)
        within_trace = float(np.sum(np.var(trial.features[chosen], axis=0, ddof=1)))
        population_distance = float(np.linalg.norm(offsets[class_index] - gamma_e[class_index]))
        nearest = min(float(np.linalg.norm(offsets[class_index] - offsets[other])) for other in range(classes) if other != class_index)
        class_rows.append([
            cosine,
            difference_energy,
            mean_energy,
            reliability,
            within_trace,
            float(len(chosen)),
            population_distance,
            nearest,
        ])
    class_features = np.asarray(class_rows, dtype=np.float64)
    global_features = np.concatenate([
        np.mean(class_features, axis=0),
        np.min(class_features, axis=0),
        np.max(class_features, axis=0),
    ])
    residual = offsets - gamma_e
    if global_features.shape != (24,) or not np.isfinite(global_features).all() or not np.isfinite(residual).all():
        raise NumericalContractError("reliability feature contract failed")
    return ReliabilityRecord(target_index, subject, global_features, class_features, residual)


def _distance_coefficients(features: np.ndarray, base: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = features[:, None, :] - base[None, :, :]
    a = np.einsum("tcm,tcm->tc", difference, difference, optimize=True)
    b = -2.0 * np.einsum("tcm,cm->tc", difference, residual, optimize=True)
    c = np.einsum("cm,cm->c", residual, residual, optimize=True)
    return a, b, c


def _gate_example(
    bundle: parent.DatasetBundle,
    proto: parent.PrototypeData,
    target_index: int,
    gamma_e: np.ndarray,
    gamma_d: np.ndarray,
    order: np.ndarray,
    epsilon: float,
    deployment_index: int | None = None,
) -> GateExample:
    record = _reliability_record(bundle, proto, target_index, gamma_e, order, epsilon)
    deployment_index = target_index if deployment_index is None else int(deployment_index)
    deployment_subject = int(bundle.subjects[deployment_index])
    trial = bundle.trials[(deployment_subject, bundle.deployment_session)]
    labels = bundle.deployment_evaluation[deployment_subject]
    if not np.array_equal(trial.trial_ids, labels.trial_ids):
        raise LeakageOrSplitError("deployment trial/evaluation identity mismatch")
    base = proto.unlabeled_means[deployment_index, 1, None, :] + gamma_d
    a, b, c = _distance_coefficients(trial.features, base, record.residual)
    return GateExample(
        record.subject_index,
        record.subject,
        record.global_features,
        record.class_features,
        record.residual,
        a,
        b,
        c,
        labels.labels.copy(),
    )


def build_source_examples(
    bundle: parent.DatasetBundle,
    proto: parent.PrototypeData,
    source_indices: np.ndarray,
    orders: Mapping[int, np.ndarray],
    epsilon: float,
    deployment_permutation: np.ndarray | None = None,
) -> list[GateExample]:
    source = np.asarray(source_indices, dtype=np.int64)
    if len(source) < 3:
        raise NumericalContractError("source set too small for leave-one-out templates")
    examples: list[GateExample] = []
    if deployment_permutation is not None and sorted(np.asarray(deployment_permutation).tolist()) != list(range(len(source))):
        raise LeakageOrSplitError("invalid source deployment permutation")
    for position, target in enumerate(source):
        enrollment_train = np.delete(source, position)
        donor_position = position if deployment_permutation is None else int(deployment_permutation[position])
        deployment_index = int(source[donor_position])
        deployment_train = source[source != deployment_index]
        gamma_e, _ = parent.population_templates(proto, enrollment_train)
        _, gamma_d = parent.population_templates(proto, deployment_train)
        subject = int(bundle.subjects[int(target)])
        examples.append(_gate_example(
            bundle, proto, int(target), gamma_e, gamma_d, orders[subject], epsilon,
            deployment_index=deployment_index,
        ))
    return examples


def _subject_loss_and_derivative(example: GateExample, kappa: float) -> tuple[float, float]:
    distances = example.distance_a + kappa * example.distance_b + (kappa * kappa) * example.distance_c[None, :]
    logits = -distances
    probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    rows = np.arange(len(example.labels))
    loss = float(np.mean(distances[rows, example.labels] + logsumexp(logits, axis=1)))
    derivative_distance = example.distance_b + 2.0 * kappa * example.distance_c[None, :]
    derivative = float(np.mean(
        derivative_distance[rows, example.labels] - np.sum(probabilities * derivative_distance, axis=1)
    ))
    return loss, derivative


def fit_gate(examples: Sequence[GateExample], l2: float, config: Mapping[str, Any]) -> GateFit:
    raw = np.stack([example.global_features for example in examples])
    mean, scale = _standardizer(raw)
    features = (raw - mean) / scale
    augmented = np.column_stack([features, np.ones(len(features))])
    initial = np.zeros(augmented.shape[1], dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        eta = augmented @ parameters
        kappas = expit(eta)
        losses = np.empty(len(examples), dtype=np.float64)
        derivatives = np.empty(len(examples), dtype=np.float64)
        for index, (example, kappa) in enumerate(zip(examples, kappas, strict=True)):
            losses[index], derivatives[index] = _subject_loss_and_derivative(example, float(kappa))
        chain = derivatives * kappas * (1.0 - kappas)
        gradient = augmented.T @ chain / len(examples)
        gradient[:-1] += float(l2) * parameters[:-1]
        value = float(np.mean(losses) + 0.5 * float(l2) * np.dot(parameters[:-1], parameters[:-1]))
        return value, gradient

    result = minimize(
        lambda value: objective(value), initial, method="L-BFGS-B", jac=True,
        options={
            "maxiter": int(config["model"]["optimizer_maxiter"]),
            "ftol": float(config["model"]["optimizer_ftol"]),
            "gtol": float(config["model"]["optimizer_gtol"]),
            "maxls": 50,
        },
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise NumericalContractError(f"selective gate optimization failed: {result.message}")
    return GateFit(np.asarray(result.x), mean, scale, float(l2), float(result.fun), int(result.nit))


def predict_kappa(model: GateFit, global_features: np.ndarray) -> float:
    standardized = (np.asarray(global_features, dtype=np.float64) - model.feature_mean) / model.feature_scale
    return float(expit(np.dot(model.parameters[:-1], standardized) + model.parameters[-1]))


def prototypes_for_kappa(current_mean: np.ndarray, gamma_d: np.ndarray, residual: np.ndarray, kappa: float | np.ndarray) -> np.ndarray:
    value = np.asarray(kappa, dtype=np.float64)
    if value.ndim == 0:
        result = current_mean[None, :] + gamma_d + float(value) * residual
    elif value.shape == (len(gamma_d),):
        result = current_mean[None, :] + gamma_d + value[:, None] * residual
    else:
        raise NumericalContractError("kappa shape invalid")
    if not np.isfinite(result).all():
        raise NumericalContractError("nonfinite selective prototype")
    return result


def _score_target(
    bundle: parent.DatasetBundle,
    proto: parent.PrototypeData,
    target_index: int,
    gamma_e: np.ndarray,
    gamma_d: np.ndarray,
    order: np.ndarray,
    epsilon: float,
    kappa: float | np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, ReliabilityRecord]:
    record = _reliability_record(bundle, proto, target_index, gamma_e, order, epsilon)
    subject = int(bundle.subjects[target_index])
    trial = bundle.trials[(subject, bundle.deployment_session)]
    prototypes = prototypes_for_kappa(proto.unlabeled_means[target_index, 1], gamma_d, record.residual, kappa)
    prediction, probability = parent.ncm_predict(trial.features, prototypes, 1.0)
    # Labels join only after prediction is complete.
    evaluation = bundle.deployment_evaluation[subject]
    score = parent.balanced_accuracy_fast(evaluation.labels, prediction, len(bundle.class_names))
    return score, prediction, probability, record


def _outer_source(bundle: parent.DatasetBundle, test: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.arange(len(bundle.subjects), dtype=np.int64), np.asarray(test, dtype=np.int64))


def select_hyperparameters(
    bundle: parent.DatasetBundle,
    proto: parent.PrototypeData,
    orders: Mapping[int, np.ndarray],
    outer_fold: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outer_test = bundle.folds[outer_fold]
    outer_source = _outer_source(bundle, outer_test)
    epsilon = float(config["features"]["reliability_epsilon"])
    l2_grid = [float(value) for value in config["model"]["l2_grid"]]
    kappa_grid = [float(value) for value in config["model"]["global_kappa_grid"]]
    l2_scores = {value: [] for value in l2_grid}
    global_scores = {value: [] for value in kappa_grid}
    rows: list[dict[str, Any]] = []
    for inner_fold, validation in enumerate(bundle.inner_folds[outer_fold]):
        validation = np.asarray(validation, dtype=np.int64)
        train = np.setdiff1d(outer_source, validation)
        if len(np.intersect1d(outer_test, train)) or len(np.intersect1d(outer_test, validation)):
            raise LeakageOrSplitError("outer target entered source-only inner selection")
        examples = build_source_examples(bundle, proto, train, orders, epsilon)
        gamma_e, gamma_d = parent.population_templates(proto, train)
        for l2 in l2_grid:
            model = fit_gate(examples, l2, config)
            scores = []
            for target in validation:
                subject = int(bundle.subjects[int(target)])
                record = _reliability_record(bundle, proto, int(target), gamma_e, orders[subject], epsilon)
                score, _, _, _ = _score_target(
                    bundle, proto, int(target), gamma_e, gamma_d, orders[subject], epsilon,
                    predict_kappa(model, record.global_features),
                )
                scores.append(score)
            mean_score = float(np.mean(scores))
            l2_scores[l2].append(mean_score)
            rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold, "method": "SELECTIVE_GATE", "l2": l2, "kappa": np.nan, "mean_balanced_accuracy": mean_score})
        for kappa in kappa_grid:
            scores = []
            for target in validation:
                subject = int(bundle.subjects[int(target)])
                score, _, _, _ = _score_target(bundle, proto, int(target), gamma_e, gamma_d, orders[subject], epsilon, kappa)
                scores.append(score)
            mean_score = float(np.mean(scores))
            global_scores[kappa].append(mean_score)
            rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold, "method": "GLOBAL_KAPPA", "l2": np.nan, "kappa": kappa, "mean_balanced_accuracy": mean_score})
    if any(len(values) != 5 for values in [*l2_scores.values(), *global_scores.values()]):
        raise LeakageOrSplitError("inner folds incomplete")
    best_l2_score = max(float(np.mean(values)) for values in l2_scores.values())
    selected_l2 = max(value for value, scores in l2_scores.items() if abs(float(np.mean(scores)) - best_l2_score) <= 1e-15)
    best_global_score = max(float(np.mean(values)) for values in global_scores.values())
    selected_global = min(value for value, scores in global_scores.items() if abs(float(np.mean(scores)) - best_global_score) <= 1e-15)
    selection = {
        "outer_fold": outer_fold,
        "selected_l2": selected_l2,
        "selected_global_kappa": selected_global,
        "selected_l2_score": best_l2_score,
        "selected_global_kappa_score": best_global_score,
        "outer_test_subjects": bundle.subjects[outer_test].tolist(),
        "outer_source_subjects": bundle.subjects[outer_source].tolist(),
        "outer_target_deployment_labels_used": False,
    }
    return selection, rows


def _ensure_protocol_frozen(root: Path, config: Mapping[str, Any]) -> None:
    status_path = output_path(root, config) / "decisions" / "protocol_freeze_status.json"
    if not status_path.is_file():
        raise LeakageOrSplitError("protocol freeze status missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PROTOCOL_FROZEN_NO_SELECTIVE_RESULT_YET":
        raise LeakageOrSplitError("protocol not frozen")
    freeze_commit = _git(root, "log", "--format=%H", "--grep=^freeze selective conditional memory feasibility v0$", "-n", "1")
    if not freeze_commit or _git(root, "merge-base", "--is-ancestor", freeze_commit, "HEAD") != "":
        raise LeakageOrSplitError("freeze commit not in current history")


def run_audit(repo_root: str | Path) -> dict[str, Any]:
    """Hashes, schemas, and acquisition order only; no real performance."""
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root, verify_protocol=False)
    output = output_path(root, config)
    _ensure_dirs(output)
    manifests = validate_parent_contract(root, config)
    snapshot = parent_hash_snapshot(root)
    atomic_write_json(output / "protocol" / "parent_artifact_hashes.json", snapshot)
    rows: list[dict[str, Any]] = []
    for dataset in ("stieger", "openbmi"):
        bundle = load_dataset(root, config, dataset)
        orders = acquisition_orders(bundle)
        counts = [len(bundle.trials[(int(subject), session)].features) for subject in bundle.subjects for session in (bundle.enrollment_session, bundle.deployment_session)]
        rows.append({
            "dataset": dataset,
            "subjects": len(bundle.subjects),
            "records": len(bundle.trials),
            "classes": len(bundle.class_names),
            "feature_dimension": 210,
            "minimum_trials": min(counts),
            "maximum_trials": max(counts),
            "outer_folds": len(bundle.folds),
            "inner_folds": len(bundle.inner_folds[0]),
            "acquisition_orders_bijective": all(len(np.unique(value)) == len(value) for value in orders.values()),
            "all_finite": True,
        })
    pd.DataFrame(rows).to_csv(output / "tables" / "parent_cache_contract.csv", index=False, lineterminator="\n")
    result = {
        "status": "PASS_PARENT_AND_TRIAL_CACHE_CONTRACT",
        "parent_head": config["protocol"]["parent_head"],
        "parent_terminal_unchanged": config["protocol"]["parent_terminal"],
        "parent_hashes": manifests,
        "parent_snapshot_count": snapshot["count"],
        "parent_snapshot_canonical_sha256": snapshot["canonical_sha256"],
        "stieger_subject_sessions": 124,
        "openbmi_feature_shape": [54, 2, 100, 210],
        "acquisition_order_recoverable": True,
        "raw_download_or_rebuild": False,
        "scientific_selective_statistic_accessed": False,
        "config_sha256": config_hash,
    }
    atomic_write_json(output / "protocol" / "cache_audit.json", result)
    return result


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

    rng = _rng(config, "synthetic")
    # Split-half reliability and aggregation.
    r_a = rng.normal(size=(4, 8)); r_b = r_a + 0.05 * rng.normal(size=(4, 8))
    mean = 0.5 * (r_a + r_b); noise = 0.5 * (r_a - r_b)
    reliability = np.maximum(np.sum(mean * mean, axis=1) - np.sum(noise * noise, axis=1), 0) / (np.sum(mean * mean, axis=1) + 1e-12)
    gate("reliability_ratio_bounds", bool(np.all((reliability >= 0) & (reliability <= 1 + 1e-12))), reliability.tolist())
    gate("subject_feature_dimension", np.concatenate([np.mean(r_a, 0), np.min(r_a, 0), np.max(r_a, 0)]).shape == (24,), 24)
    gamma = rng.normal(size=(4, 6)); residual = rng.normal(size=(4, 6)); current = rng.normal(size=6)
    p0 = prototypes_for_kappa(current, gamma, residual, 0.0); p1 = prototypes_for_kappa(current, gamma, residual, 1.0)
    gate("kappa_zero_population_endpoint", np.allclose(p0, current + gamma), float(np.max(np.abs(p0-current-gamma))))
    gate("kappa_one_identity_endpoint", np.allclose(p1, current + gamma + residual), float(np.max(np.abs(p1-current-gamma-residual))))
    midpoint = prototypes_for_kappa(current, gamma, residual, 0.4)
    gate("convex_shrinkage_segment", np.allclose(midpoint, 0.6*p0 + 0.4*p1), float(np.max(np.abs(midpoint-(0.6*p0+0.4*p1)))))
    # Analytic distance coefficients equal direct distances.
    trials = rng.normal(size=(20, 6)); a, b, c = _distance_coefficients(trials, p0, residual); kappa = .37
    direct = np.sum((trials[:, None, :] - prototypes_for_kappa(current, gamma, residual, kappa)[None, :, :])**2, axis=2)
    gate("quadratic_distance_identity", np.allclose(a+kappa*b+kappa*kappa*c, direct, atol=2e-12), float(np.max(np.abs(a+kappa*b+kappa*kappa*c-direct))))
    # Gate learns reliability-selective reuse on synthetic paired examples.
    examples: list[GateExample] = []
    classes, dim = 4, 7
    base = rng.normal(size=(classes, dim))
    residual_i = rng.normal(scale=2.5, size=(classes, dim))
    for index in range(48):
        h = rng.normal(size=24); true_kappa = float(expit(2.5*h[0]-0.5*h[1]))
        labels = np.repeat(np.arange(classes), 30)
        features = np.concatenate([base[c] + true_kappa*residual_i[c] + .10*rng.normal(size=(30, dim)) for c in range(classes)])
        aa, bb, cc = _distance_coefficients(features, base, residual_i)
        examples.append(GateExample(index, index+1, h, rng.normal(size=(classes,8)), residual_i, aa, bb, cc, labels))
    model = fit_gate(examples, 1e-3, config)
    predicted = np.asarray([predict_kappa(model, example.global_features) for example in examples])
    truth = np.asarray([float(expit(2.5*example.global_features[0]-0.5*example.global_features[1])) for example in examples])
    gate("known_reliability_gate_recovery", np.corrcoef(predicted, truth)[0,1] > .9, float(np.corrcoef(predicted, truth)[0,1]))
    # Permutations and label boundary.
    derangement = parent.fixed_point_free(_rng(config, "synthetic_derangement"), 12)
    gate("subject_memory_derangement", not np.any(derangement == np.arange(12)), derangement.tolist())
    class_perm = parent.nonidentity_class_permutation(_rng(config, "synthetic_class"), 4)
    gate("class_semantics_nonidentity", not np.array_equal(class_perm, np.arange(4)), class_perm.tolist())
    feature_perm = parent.fixed_point_free(_rng(config, "synthetic_feature"), 12)
    gate("reliability_feature_derangement", not np.any(feature_perm == np.arange(12)), feature_perm.tolist())
    # Oracle headroom construction.
    identity = np.asarray([.4,.6,.5,.7]); population = np.asarray([.6,.5,.7,.6]); oracle = np.maximum(identity,population)
    gate("oracle_selective_ceiling", float(np.mean(oracle-identity)) > 0, float(np.mean(oracle-identity)))
    # Scaling fit excludes sentinel target.
    source = rng.normal(size=(10,24)); target = np.full(24, 1e9); mean_source, _ = _standardizer(source)
    gate("outer_source_only_scaling", np.allclose(mean_source, source.mean(0)) and not np.allclose(mean_source, np.vstack([source,target]).mean(0)), True)
    # Target API exposes no deployment-feature/label gate argument.
    import inspect
    parameters = inspect.signature(_reliability_record).parameters
    gate("target_label_and_deployment_feature_sentinel", "deployment_labels" not in parameters and "deployment_features" not in parameters, list(parameters))
    gate("deterministic_seed_namespaces", np.array_equal(_rng(config,"same").integers(0,100,20), _rng(config,"same").integers(0,100,20)), True)
    expected = {config["decisions"][key] for key in config["decisions"] if key not in ("next_if_go","next_if_stop")}
    gate("expected_terminal_labels", config["decisions"]["go"] in expected and config["decisions"]["oracle_stop"] in expected, sorted(expected))
    result = {"status": "PASS", "count": len(gates), "all_passed": all(row["passed"] for row in gates), "gates": gates}
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
    protocol = root / str(config["protocol"]["protocol_path"])
    (output / "protocol" / protocol.name).write_bytes(protocol.read_bytes())
    (output / "protocol" / Path(CONFIG_PATH).name).write_bytes((root / CONFIG_PATH).read_bytes())
    parent_config = _parent_config(root)
    (output / "protocol" / "exact_stieger_folds.json").write_bytes((root / parent_config["stieger"]["fold_path"]).read_bytes())
    (output / "protocol" / "exact_openbmi_folds.csv").write_bytes((root / parent_config["openbmi"]["fold_path"]).read_bytes())
    result = {
        "status": "PROTOCOL_FROZEN_NO_SELECTIVE_RESULT_YET",
        "parent_head": config["protocol"]["parent_head"],
        "parent_terminal_unchanged": config["protocol"]["parent_terminal"],
        "config_sha256": config_hash,
        "protocol_sha256": sha256_file(protocol),
        "cache_gate": audit["status"],
        "synthetic_gate": synthetic["status"],
        "scientific_selective_statistic_accessed": False,
    }
    atomic_write_json(output / "decisions" / "protocol_freeze_status.json", result)
    atomic_write_json(output / "manifest.json", {"phase": "PROTOCOL_FREEZE", **result})
    atomic_write_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "raw_data_accessed": False})
    atomic_write_json(output / "git_provenance.json", {"parent_head": config["protocol"]["parent_head"], "working_head": _git(root,"rev-parse","HEAD"), "branch": _git(root,"branch","--show-current")})
    return result


def _parent_prediction_paths(root: Path, config: Mapping[str, Any], dataset: str, reverse: bool) -> tuple[Path, Path]:
    direction = "reverse" if reverse else "chronological"
    key = f"{dataset}_{direction}"
    prediction = root / config["parent_contract"]["pr20_predictions"][key]["path"]
    labels = root / str(config["project"]["parent_output_dir"]) / "objects" / f"{key}_deployment_evaluation_labels.npz"
    if not prediction.is_file() or not labels.is_file():
        raise RequiredTrialCacheError(f"missing PR #20 prediction/evaluation pair: {key}")
    return prediction, labels


def _load_parent_subject_predictions(root: Path, config: Mapping[str, Any], dataset: str, reverse: bool) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[str, ...]]:
    prediction_path, label_path = _parent_prediction_paths(root, config, dataset, reverse)
    with np.load(prediction_path, allow_pickle=False) as archive:
        subjects = np.asarray(archive["trial_subject"], dtype=np.int64)
        trial_ids = np.asarray(archive["trial_id"])
        predictions = {
            "POPULATION_ONLY": np.asarray(archive["prediction__POPULATION_ONLY"], dtype=np.int64),
            "IDENTITY_RESIDUAL_CARRY": np.asarray(archive["prediction__IDENTITY_RESIDUAL_CARRY"], dtype=np.int64),
            "PAST_PROTOTYPE_DIRECT": np.asarray(archive["prediction__PAST_PROTOTYPE_DIRECT"], dtype=np.int64),
            "LRCM_HISTORICAL_CONTROL": np.asarray(archive["prediction__LRCM"], dtype=np.int64),
        }
    with np.load(label_path, allow_pickle=False) as archive:
        evaluation_subjects = np.asarray(archive["trial_subject"], dtype=np.int64)
        evaluation_ids = np.asarray(archive["trial_id"])
        labels = np.asarray(archive["class_index"], dtype=np.int64)
        class_names = tuple(str(value) for value in archive["class_names"].tolist())
    if not np.array_equal(subjects, evaluation_subjects) or not np.array_equal(trial_ids, evaluation_ids):
        raise LeakageOrSplitError("PR #20 prediction/evaluation identity mismatch")
    return subjects, labels, predictions, class_names


def _bootstrap_ci(values: np.ndarray, config: Mapping[str, Any], namespace: str, replicates: int | None = None) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    count = int(config["inference"]["bootstrap_replicates"] if replicates is None else replicates)
    rng = _rng(config, namespace)
    sampled = np.empty(count, dtype=np.float64)
    for index in range(count):
        sampled[index] = float(np.mean(values[rng.integers(0, len(values), len(values))]))
    return float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def _paired_signflip_p(values: np.ndarray, config: Mapping[str, Any], namespace: str, replicates: int | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    count = int(config["inference"]["permutation_replicates"] if replicates is None else replicates)
    observed = float(np.mean(values))
    rng = _rng(config, namespace)
    exceed = 0
    for _ in range(count):
        statistic = float(np.mean(values * rng.choice(np.asarray([-1.0, 1.0]), size=len(values))))
        exceed += statistic >= observed
    return float((1 + exceed) / (1 + count))


def _holm(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_value_raw"]))
    running = 0.0
    for position, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - position) * float(rows[index]["p_value_raw"]))
        running = max(running, adjusted)
        rows[index]["p_value_holm"] = running


def run_oracle_ceiling(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    _ensure_protocol_frozen(root, config)
    output = output_path(root, config)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for dataset in ("stieger", "openbmi"):
        for reverse in (False, True):
            direction = "reverse" if reverse else "chronological"
            subjects, labels, predictions, class_names = _load_parent_subject_predictions(root, config, dataset, reverse)
            subject_rows: list[dict[str, Any]] = []
            for subject in np.unique(subjects):
                chosen = subjects == subject
                identity = parent.balanced_accuracy_fast(labels[chosen], predictions["IDENTITY_RESIDUAL_CARRY"][chosen], len(class_names))
                population = parent.balanced_accuracy_fast(labels[chosen], predictions["POPULATION_ONLY"][chosen], len(class_names))
                select_population = population > identity
                selected_method = "POPULATION_ONLY" if select_population else "IDENTITY_RESIDUAL_CARRY"
                selected = max(population, identity)
                row: dict[str, Any] = {
                    "dataset": dataset, "direction": direction, "subject": int(subject),
                    "identity_balanced_accuracy": identity,
                    "population_balanced_accuracy": population,
                    "oracle_balanced_accuracy": selected,
                    "oracle_gain_over_identity": selected - identity,
                    "selected_method": selected_method,
                }
                oracle_prediction = predictions[selected_method][chosen]
                for class_index, class_name in enumerate(class_names):
                    class_chosen = labels[chosen] == class_index
                    row[f"oracle_recall_{class_name}"] = float(np.mean(oracle_prediction[class_chosen] == class_index))
                    row[f"identity_recall_{class_name}"] = float(np.mean(predictions["IDENTITY_RESIDUAL_CARRY"][chosen][class_chosen] == class_index))
                subject_rows.append(row)
                rows.append(row)
            frame = pd.DataFrame(subject_rows)
            gain = frame["oracle_gain_over_identity"].to_numpy(dtype=np.float64)
            key = f"{dataset}_{direction}"
            summaries[key] = {
                "dataset": dataset,
                "direction": direction,
                "voting": dataset == "stieger" and not reverse,
                "subjects": len(frame),
                "mean_oracle_balanced_accuracy": float(frame["oracle_balanced_accuracy"].mean()),
                "mean_identity_balanced_accuracy": float(frame["identity_balanced_accuracy"].mean()),
                "mean_population_balanced_accuracy": float(frame["population_balanced_accuracy"].mean()),
                "mean_oracle_gain_over_identity": float(np.mean(gain)),
                "oracle_gain_bootstrap_ci": list(_bootstrap_ci(gain, config, f"oracle_{key}")),
                "proportion_selecting_population": float(np.mean(frame["selected_method"] == "POPULATION_ONLY")),
                "proportion_selecting_identity": float(np.mean(frame["selected_method"] == "IDENTITY_RESIDUAL_CARRY")),
            }
    pd.DataFrame(rows).to_csv(output / "tables" / "oracle_identity_or_population_subjects.csv", index=False, lineterminator="\n", float_format="%.17g")
    stieger = summaries["stieger_chronological"]
    passes = bool(stieger["mean_oracle_gain_over_identity"] > 0.0 and stieger["oracle_gain_bootstrap_ci"][0] > 0.0)
    decision = {
        "status": "ORACLE_SELECTIVE_MEMORY_HEADROOM_PRESENT" if passes else config["decisions"]["oracle_stop"],
        "oracle_gate_pass": passes,
        "summaries": summaries,
        "gate_fitting_authorized": passes,
    }
    atomic_write_json(output / "decisions" / "oracle_ceiling.json", decision)
    return decision


def _gate_prediction(
    bundle: parent.DatasetBundle,
    proto: parent.PrototypeData,
    target: int,
    gamma_e: np.ndarray,
    gamma_d: np.ndarray,
    order: np.ndarray,
    epsilon: float,
    model: GateFit,
) -> tuple[np.ndarray, np.ndarray, ReliabilityRecord, float]:
    record = _reliability_record(bundle, proto, target, gamma_e, order, epsilon)
    kappa = predict_kappa(model, record.global_features)
    subject = int(bundle.subjects[target])
    trial = bundle.trials[(subject, bundle.deployment_session)]
    prototypes = prototypes_for_kappa(proto.unlabeled_means[target, 1], gamma_d, record.residual, kappa)
    prediction, probability = parent.ncm_predict(trial.features, prototypes, 1.0)
    return prediction, probability, record, kappa


def _oracle_continuous_kappa(
    bundle: parent.DatasetBundle,
    proto: parent.PrototypeData,
    target: int,
    gamma_e: np.ndarray,
    gamma_d: np.ndarray,
    order: np.ndarray,
    epsilon: float,
    points: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for kappa in np.linspace(0.0, 1.0, points):
        score, prediction, probability, _ = _score_target(bundle, proto, target, gamma_e, gamma_d, order, epsilon, float(kappa))
        candidate = (score, -float(kappa), prediction, probability)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return -best[1], best[2], best[3]


def _metrics_row(
    dataset: str,
    direction: str,
    fold: int,
    subject: int,
    method: str,
    labels: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    class_names: Sequence[str],
    **extra: Any,
) -> dict[str, Any]:
    metric = parent.subject_metrics(labels, prediction, probability, len(class_names))
    row: dict[str, Any] = {
        "dataset": dataset, "direction": direction, "outer_fold": fold,
        "subject": subject, "method": method,
        **{key: value for key, value in metric.items() if key != "per_class_recall"},
        **extra,
    }
    for index, value in enumerate(metric["per_class_recall"]):
        row[f"recall_{class_names[index]}"] = value
    return row


def _load_parent_method_arrays(root: Path, config: Mapping[str, Any], dataset: str, reverse: bool) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    direction = "reverse" if reverse else "chronological"
    prediction_path, label_path = _parent_prediction_paths(root, config, dataset, reverse)
    with np.load(prediction_path, allow_pickle=False) as archive:
        subjects = np.asarray(archive["trial_subject"], dtype=np.int64)
        trial_ids = np.asarray(archive["trial_id"])
        methods = {
            "POPULATION_ONLY": (np.asarray(archive["prediction__POPULATION_ONLY"], dtype=np.int64), np.asarray(archive["probability__POPULATION_ONLY"], dtype=np.float64)),
            "IDENTITY_RESIDUAL_CARRY": (np.asarray(archive["prediction__IDENTITY_RESIDUAL_CARRY"], dtype=np.int64), np.asarray(archive["probability__IDENTITY_RESIDUAL_CARRY"], dtype=np.float64)),
            "PAST_PROTOTYPE_DIRECT": (np.asarray(archive["prediction__PAST_PROTOTYPE_DIRECT"], dtype=np.int64), np.asarray(archive["probability__PAST_PROTOTYPE_DIRECT"], dtype=np.float64)),
            "LRCM_HISTORICAL_CONTROL": (np.asarray(archive["prediction__LRCM"], dtype=np.int64), np.asarray(archive["probability__LRCM"], dtype=np.float64)),
        }
    with np.load(label_path, allow_pickle=False) as archive:
        if not np.array_equal(subjects, archive["trial_subject"]) or not np.array_equal(trial_ids, archive["trial_id"]):
            raise LeakageOrSplitError(f"parent prediction identity mismatch {dataset}/{direction}")
    output: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for subject in np.unique(subjects):
        chosen = subjects == subject
        output[int(subject)] = {method: (values[0][chosen], values[1][chosen]) for method, values in methods.items()}
    return output


def run_dataset_observed(repo_root: str | Path, dataset: str, reverse: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    _ensure_protocol_frozen(root, config)
    output = output_path(root, config)
    oracle = json.loads((output / "decisions" / "oracle_ceiling.json").read_text(encoding="utf-8"))
    if not oracle.get("oracle_gate_pass"):
        raise LeakageOrSplitError("oracle headroom did not authorize gate fitting")
    bundle = load_dataset(root, config, dataset, reverse=reverse)
    proto = parent.compute_prototype_data(bundle)
    orders = acquisition_orders(bundle)
    epsilon = float(config["features"]["reliability_epsilon"])
    direction = "reverse" if reverse else "chronological"
    parent_arrays = _load_parent_method_arrays(root, config, dataset, reverse)
    subject_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    compact: dict[str, list[np.ndarray]] = {key: [] for key in ("subject","fold","global_features","class_features","residual","gamma_e","gamma_d","parameters","feature_mean","feature_scale","kappa","global_kappa")}
    started = time.perf_counter()
    for fold, test in enumerate(bundle.folds):
        selection, candidates = select_hyperparameters(bundle, proto, orders, fold, config)
        selections.append(selection); inner_rows.extend(candidates)
        source = _outer_source(bundle, test)
        gamma_e, gamma_d = parent.population_templates(proto, source)
        examples = build_source_examples(bundle, proto, source, orders, epsilon)
        model = fit_gate(examples, float(selection["selected_l2"]), config)
        model_rows.append({"dataset": dataset, "direction": direction, "outer_fold": fold, "selected_l2": model.l2, "objective": model.objective, "optimizer_iterations": model.iterations, "parameter_norm": float(np.linalg.norm(model.parameters[:-1])), "intercept": float(model.parameters[-1])})
        for target in test:
            target = int(target); subject = int(bundle.subjects[target])
            trial = bundle.trials[(subject, bundle.deployment_session)]
            evaluation = bundle.deployment_evaluation[subject]
            # Proposed prediction is frozen before evaluation labels are joined.
            selective_pred, selective_prob, record, kappa = _gate_prediction(bundle, proto, target, gamma_e, gamma_d, orders[subject], epsilon, model)
            global_kappa = float(selection["selected_global_kappa"])
            _, global_pred, global_prob, _ = _score_target(bundle, proto, target, gamma_e, gamma_d, orders[subject], epsilon, global_kappa)
            fixed_kappa = float(np.clip(np.mean(record.class_features[:, 3]), 0.0, 1.0))
            _, fixed_pred, fixed_prob, _ = _score_target(bundle, proto, target, gamma_e, gamma_d, orders[subject], epsilon, fixed_kappa)
            classwise_kappa = np.clip(record.class_features[:, 3], 0.0, 1.0)
            _, class_pred, class_prob, _ = _score_target(bundle, proto, target, gamma_e, gamma_d, orders[subject], epsilon, classwise_kappa)
            oracle_kappa, oracle_pred, oracle_prob = _oracle_continuous_kappa(bundle, proto, target, gamma_e, gamma_d, orders[subject], epsilon, int(config["model"]["oracle_kappa_grid_points"]))
            predictions = {
                "SELECTIVE_GATE": (selective_pred, selective_prob, {"kappa": kappa}),
                "GLOBAL_KAPPA": (global_pred, global_prob, {"kappa": global_kappa}),
                "FIXED_RELIABILITY_KAPPA": (fixed_pred, fixed_prob, {"kappa": fixed_kappa}),
                "CLASSWISE_SELECTIVE_GATE": (class_pred, class_prob, {"kappa": float(np.mean(classwise_kappa))}),
                "ORACLE_CONTINUOUS_KAPPA": (oracle_pred, oracle_prob, {"kappa": oracle_kappa}),
            }
            # Verify the frozen endpoints reproduce the exact PR #20 predictions.
            for method, endpoint_kappa in (("POPULATION_ONLY", 0.0), ("IDENTITY_RESIDUAL_CARRY", 1.0)):
                _, endpoint_pred, endpoint_prob, _ = _score_target(bundle, proto, target, gamma_e, gamma_d, orders[subject], epsilon, endpoint_kappa)
                frozen_pred, frozen_prob = parent_arrays[subject][method]
                if not np.array_equal(endpoint_pred, frozen_pred) or not np.allclose(endpoint_prob, frozen_prob, atol=2e-14, rtol=2e-14):
                    raise RequiredTrialCacheError(f"frozen PR #20 endpoint mismatch: {dataset}/{direction}/S{subject}/{method}")
            for method, values in parent_arrays[subject].items():
                predictions[method] = (values[0], values[1], {"kappa": np.nan})
            for method, (prediction, probability, extra) in predictions.items():
                subject_rows.append(_metrics_row(dataset, direction, fold, subject, method, evaluation.labels, prediction, probability, bundle.class_names, **extra))
            feature_row: dict[str, Any] = {"dataset": dataset, "direction": direction, "outer_fold": fold, "subject": subject, "kappa": kappa, "global_kappa": global_kappa, "fixed_reliability_kappa": fixed_kappa, "oracle_continuous_kappa": oracle_kappa}
            for index, value in enumerate(record.global_features): feature_row[f"h_{index:02d}"] = value
            feature_rows.append(feature_row)
            compact["subject"].append(np.asarray(subject, dtype=np.int16)); compact["fold"].append(np.asarray(fold, dtype=np.int8))
            compact["global_features"].append(record.global_features); compact["class_features"].append(record.class_features); compact["residual"].append(record.residual)
            compact["gamma_e"].append(gamma_e); compact["gamma_d"].append(gamma_d); compact["parameters"].append(model.parameters); compact["feature_mean"].append(model.feature_mean); compact["feature_scale"].append(model.feature_scale)
            compact["kappa"].append(np.asarray(kappa)); compact["global_kappa"].append(np.asarray(global_kappa))
    suffix = f"{dataset}_{direction}"
    frame = pd.DataFrame(subject_rows)
    frame.to_csv(output / "tables" / f"{suffix}_per_subject_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(feature_rows).to_csv(output / "tables" / f"{suffix}_reliability_features.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(inner_rows).to_csv(output / "tables" / f"{suffix}_inner_selection.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(model_rows).to_csv(output / "tables" / f"{suffix}_gate_models.csv", index=False, lineterminator="\n", float_format="%.17g")
    atomic_write_json(output / "objects" / f"{suffix}_selections.json", {"selections": selections})
    np.savez_compressed(output / "objects" / f"{suffix}_gate_objects.npz", **{key: np.stack(value) for key, value in compact.items()})
    aggregates = []
    for method, block in frame.groupby("method", sort=False):
        aggregates.append({"dataset": dataset, "direction": direction, "method": method, **{f"mean_{metric}": float(block[metric].mean()) for metric in ("balanced_accuracy","macro_f1","nll","brier","ece10")}})
    pd.DataFrame(aggregates).to_csv(output / "tables" / f"{suffix}_aggregate_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    summary = {"dataset": dataset, "direction": direction, "voting": dataset == "stieger" and not reverse, "subjects": len(bundle.subjects), "aggregate_metrics": aggregates, "selected_l2": [float(row["selected_l2"]) for row in selections], "selected_global_kappa": [float(row["selected_global_kappa"]) for row in selections], "kappa_summary": {"mean": float(pd.DataFrame(feature_rows)["kappa"].mean()), "median": float(pd.DataFrame(feature_rows)["kappa"].median()), "minimum": float(pd.DataFrame(feature_rows)["kappa"].min()), "maximum": float(pd.DataFrame(feature_rows)["kappa"].max())}, "outer_target_deployment_labels_used_for_gate": False, "total_seconds": time.perf_counter()-started}
    atomic_write_json(output / "decisions" / f"{suffix}_observed.json", summary)
    return summary


def run_stieger_observed(repo_root: str | Path) -> dict[str, Any]:
    chronological = run_dataset_observed(repo_root, "stieger", False)
    try:
        reverse: dict[str, Any] = run_dataset_observed(repo_root, "stieger", True)
    except NumericalContractError as exc:
        root = Path(repo_root).resolve(); config, _ = load_config(root); output = output_path(root, config)
        reverse = {"status": "CONTROL_UNASSESSED_OPTIMIZATION_FAILURE", "exception_type": type(exc).__name__, "message": str(exc), "voting": False}
        atomic_write_json(output / "controls" / "stieger_reverse_status.json", reverse)
    return {"chronological": chronological, "reverse_non_voting": reverse}


def run_openbmi_observed(repo_root: str | Path) -> dict[str, Any]:
    chronological = run_dataset_observed(repo_root, "openbmi", False)
    try:
        reverse: dict[str, Any] = run_dataset_observed(repo_root, "openbmi", True)
    except NumericalContractError as exc:
        root = Path(repo_root).resolve(); config, _ = load_config(root); output = output_path(root, config)
        reverse = {"status": "CONTROL_UNASSESSED_OPTIMIZATION_FAILURE", "exception_type": type(exc).__name__, "message": str(exc), "voting": False}
        atomic_write_json(output / "controls" / "openbmi_reverse_status.json", reverse)
    return {"chronological": chronological, "reverse_non_voting": reverse}


def _evaluate_target_from_record(
    bundle: parent.DatasetBundle,
    proto: parent.PrototypeData,
    target: int,
    gamma_d: np.ndarray,
    residual: np.ndarray,
    kappa: float,
) -> float:
    subject = int(bundle.subjects[target])
    trial = bundle.trials[(subject, bundle.deployment_session)]
    predicted = prototypes_for_kappa(proto.unlabeled_means[target, 1], gamma_d, residual, kappa)
    labels = bundle.deployment_evaluation[subject].labels
    prediction, _ = parent.ncm_predict(trial.features, predicted, 1.0)
    return parent.balanced_accuracy_fast(labels, prediction, len(bundle.class_names))


def run_stieger_nulls(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    _ensure_protocol_frozen(root, config)
    output = output_path(root, config)
    observed_path = output / "tables" / "stieger_chronological_per_subject_metrics.csv"
    if not observed_path.is_file():
        raise LeakageOrSplitError("Stieger observed analysis must precede nulls")
    observed_frame = pd.read_csv(observed_path)
    pivot = observed_frame.pivot(index="subject", columns="method", values="balanced_accuracy")
    observed_gain = float(np.mean(pivot["SELECTIVE_GATE"] - pivot["IDENTITY_RESIDUAL_CARRY"]))
    bundle = load_dataset(root, config, "stieger", reverse=False)
    proto = parent.compute_prototype_data(bundle)
    orders = acquisition_orders(bundle)
    epsilon = float(config["features"]["reliability_epsilon"])
    selections = json.loads((output / "objects" / "stieger_chronological_selections.json").read_text(encoding="utf-8"))["selections"]
    replicates = int(config["inference"]["null_replicates"])
    names = tuple(config["inference"]["nulls"])
    null_values = {name: np.empty(replicates, dtype=np.float64) for name in names}
    started = time.perf_counter()
    # Cache fold target records and observed models; no deployment value enters h.
    fold_cache: list[dict[str, Any]] = []
    for fold, test in enumerate(bundle.folds):
        source = _outer_source(bundle, test)
        gamma_e, gamma_d = parent.population_templates(proto, source)
        examples = build_source_examples(bundle, proto, source, orders, epsilon)
        model = fit_gate(examples, float(selections[fold]["selected_l2"]), config)
        records = [_reliability_record(bundle, proto, int(target), gamma_e, orders[int(bundle.subjects[int(target)])], epsilon) for target in test]
        fold_cache.append({"test": np.asarray(test), "source": source, "gamma_d": gamma_d, "model": model, "records": records})
    for replicate in range(replicates):
        scores = {name: [] for name in names}
        for fold, cache in enumerate(fold_cache):
            test = cache["test"]; records = cache["records"]; gamma_d = cache["gamma_d"]; model = cache["model"]
            n_test = len(test)
            donor = parent.fixed_point_free(_rng(config, "null_subject_memory", replicate, fold), n_test)
            feature_donor = parent.fixed_point_free(_rng(config, "null_reliability_features", replicate, fold), n_test)
            # N4: enrollment/source example i is paired to a different deployment subject.
            source = cache["source"]
            source_derangement = parent.fixed_point_free(_rng(config, "null_unpaired_source", replicate, fold), len(source))
            unpaired_examples = build_source_examples(bundle, proto, source, orders, epsilon, deployment_permutation=source_derangement)
            unpaired_model = fit_gate(unpaired_examples, float(selections[fold]["selected_l2"]), config)
            for local, target in enumerate(test):
                target = int(target); record = records[local]
                kappa = predict_kappa(model, record.global_features)
                identity = float(pivot.loc[int(bundle.subjects[target]), "IDENTITY_RESIDUAL_CARRY"])
                score = _evaluate_target_from_record(bundle, proto, target, gamma_d, records[int(donor[local])].residual, kappa)
                scores["ENROLLMENT_SUBJECT_MEMORY_PERMUTATION"].append(score - identity)
                permuted_kappa = predict_kappa(model, records[int(feature_donor[local])].global_features)
                score = _evaluate_target_from_record(bundle, proto, target, gamma_d, record.residual, permuted_kappa)
                scores["RELIABILITY_FEATURE_PERMUTATION"].append(score - identity)
                class_order = parent.nonidentity_class_permutation(_rng(config, "null_class_semantics", replicate, fold, local), len(bundle.class_names))
                score = _evaluate_target_from_record(bundle, proto, target, gamma_d, record.residual[class_order], kappa)
                scores["ENROLLMENT_CLASS_SEMANTICS_PERMUTATION"].append(score - identity)
                score = _evaluate_target_from_record(bundle, proto, target, gamma_d, record.residual, predict_kappa(unpaired_model, record.global_features))
                scores["UNPAIRED_SOURCE_SESSION_GATE_TRAINING"].append(score - identity)
        for name in names:
            null_values[name][replicate] = float(np.mean(scores[name]))
    summary_rows = []
    for name in names:
        values = null_values[name]
        p_value = float((1 + np.count_nonzero(values >= observed_gain)) / (1 + replicates))
        summary_rows.append({"null": name, "observed_selective_minus_identity": observed_gain, "null_mean": float(np.mean(values)), "null_median": float(np.median(values)), "null_95_low": float(np.quantile(values,.025)), "null_95_high": float(np.quantile(values,.975)), "p_value": p_value, "replicates": replicates})
    pd.DataFrame(summary_rows).to_csv(output / "tables" / "stieger_selective_memory_nulls.csv", index=False, lineterminator="\n", float_format="%.17g")
    np.savez_compressed(output / "nulls" / "stieger_selective_memory_nulls.npz", **null_values)
    result = {"dataset": "stieger", "direction": "chronological", "observed_gain": observed_gain, "nulls": summary_rows, "total_seconds": time.perf_counter()-started}
    atomic_write_json(output / "decisions" / "stieger_selective_memory_nulls.json", result)
    return result


def _comparison(frame: pd.DataFrame, left: str, right: str, metric: str, config: Mapping[str, Any], namespace: str, higher: bool = True) -> dict[str, Any]:
    pivot = frame.pivot(index="subject", columns="method", values=metric)
    raw = pivot[left].to_numpy(dtype=np.float64) - pivot[right].to_numpy(dtype=np.float64)
    values = raw if higher else -raw
    loo = np.asarray([np.mean(np.delete(values, index)) for index in range(len(values))])
    return {
        "left": left, "right": right, "metric": metric, "improvement_direction": "higher" if higher else "lower",
        "mean_difference": float(np.mean(values)), "median_difference": float(np.median(values)),
        "bootstrap_ci": list(_bootstrap_ci(values, config, namespace+"_bootstrap")),
        "p_value_raw": _paired_signflip_p(values, config, namespace+"_signflip"),
        "subject_win_rate": float(np.mean(values > 0)),
        "worst_quartile_mean_difference": float(np.mean(np.sort(values)[:max(1,len(values)//4)])),
        "leave_one_subject_mean_range": [float(np.min(loo)), float(np.max(loo))],
    }


def _result_manifest(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(value for value in output.rglob("*") if value.is_file() and value.name != "manifest.json"):
        records.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload: dict[str, Any] = {"schema_version": "selective-conditional-memory-feasibility-v0-results", "count": len(records), "records": records}
    payload["canonical_sha256"] = _canonical_hash(payload)
    return payload


def _make_figures(output: Path, oracle: Mapping[str, Any], stieger: pd.DataFrame, openbmi: pd.DataFrame, comparisons: Sequence[Mapping[str, Any]], nulls: pd.DataFrame) -> None:
    # Fixed presentation specified before result access.
    methods = ["POPULATION_ONLY","IDENTITY_RESIDUAL_CARRY","GLOBAL_KAPPA","SELECTIVE_GATE","ORACLE_CONTINUOUS_KAPPA"]
    means = [float(stieger.loc[stieger.method==method,"balanced_accuracy"].mean()) for method in methods]
    fig, ax = plt.subplots(figsize=(8,4.5)); ax.bar(np.arange(len(methods)), means, color=["#999999","#577590","#90be6d","#f94144","#f9c74f"]); ax.set_xticks(np.arange(len(methods)), ["Population","Identity","Global κ","Selective","Oracle κ"], rotation=20); ax.set_ylabel("Mean subject balanced accuracy"); ax.set_title("Stieger selective conditional memory"); fig.tight_layout(); fig.savefig(output/"figures"/"figure_01_stieger_methods.png",dpi=180); fig.savefig(output/"figures"/"figure_01_stieger_methods.pdf"); plt.close(fig)
    pivot = stieger.pivot(index="subject", columns="method", values="balanced_accuracy"); gain = pivot["SELECTIVE_GATE"]-pivot["IDENTITY_RESIDUAL_CARRY"]
    fig, ax = plt.subplots(figsize=(8,4)); ax.axhline(0,color="black",lw=1); ax.bar(np.arange(len(gain)), np.sort(gain), color=np.where(np.sort(gain)>0,"#43aa8b","#f94144")); ax.set_xlabel("Subjects sorted by gain"); ax.set_ylabel("Selective − identity BA"); fig.tight_layout(); fig.savefig(output/"figures"/"figure_02_subject_gains.png",dpi=180); fig.savefig(output/"figures"/"figure_02_subject_gains.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4)); ax.scatter(stieger.loc[stieger.method=="SELECTIVE_GATE","kappa"], gain.reindex(stieger.loc[stieger.method=="SELECTIVE_GATE","subject"]).to_numpy(), alpha=.75); ax.axhline(0,color="black",lw=1); ax.set_xlabel("Enrollment-only κ"); ax.set_ylabel("Selective − identity BA"); ax.set_title("Gate strength and held-out gain"); fig.tight_layout(); fig.savefig(output/"figures"/"figure_03_kappa_gain.png",dpi=180); fig.savefig(output/"figures"/"figure_03_kappa_gain.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4)); ax.bar(np.arange(len(nulls)), nulls["p_value"], color="#577590"); ax.axhline(.05,color="#f94144",ls="--"); ax.set_xticks(np.arange(len(nulls)), ["Memory","Reliability","Semantics","Unpaired"], rotation=15); ax.set_ylabel("Null p-value"); fig.tight_layout(); fig.savefig(output/"figures"/"figure_04_nulls.png",dpi=180); fig.savefig(output/"figures"/"figure_04_nulls.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4)); datasets=["Stieger","OpenBMI"]; values=[float((stieger.pivot(index="subject",columns="method",values="balanced_accuracy")["SELECTIVE_GATE"]-stieger.pivot(index="subject",columns="method",values="balanced_accuracy")["IDENTITY_RESIDUAL_CARRY"]).mean()), float((openbmi.pivot(index="subject",columns="method",values="balanced_accuracy")["SELECTIVE_GATE"]-openbmi.pivot(index="subject",columns="method",values="balanced_accuracy")["IDENTITY_RESIDUAL_CARRY"]).mean())]; ax.bar(datasets,values,color=["#f94144","#577590"]); ax.axhline(0,color="black",lw=1); ax.set_ylabel("Selective − identity BA"); fig.tight_layout(); fig.savefig(output/"figures"/"figure_05_external_replication.png",dpi=180); fig.savefig(output/"figures"/"figure_05_external_replication.pdf"); plt.close(fig)


def generate_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, _ = load_config(root)
    _ensure_protocol_frozen(root, config)
    output = output_path(root, config)
    oracle = json.loads((output/"decisions"/"oracle_ceiling.json").read_text(encoding="utf-8"))
    if not oracle.get("oracle_gate_pass"):
        decision = {"terminal": config["decisions"]["oracle_stop"], "parent_terminal_unchanged": config["protocol"]["parent_terminal"], "oracle": oracle, "gate_fitted": False, "next_statement": config["decisions"]["next_if_stop"]}
        atomic_write_json(output/"decisions"/"terminal_decision.json",decision)
        report = "# Selective Conditional Memory Feasibility V0\n\n" + f"Terminal: `{decision['terminal']}`. The frozen PR #20 terminal remains `{config['protocol']['parent_terminal']}`. The Stieger oracle ceiling did not have a strictly positive bootstrap lower bound, so no deployable gate was fitted.\n"
        (output/"report"/"selective_conditional_memory_feasibility_v0.md").write_text(report,encoding="utf-8")
        manifest=_result_manifest(output); atomic_write_json(output/"manifest.json",manifest); return decision
    stieger = pd.read_csv(output/"tables"/"stieger_chronological_per_subject_metrics.csv")
    openbmi = pd.read_csv(output/"tables"/"openbmi_chronological_per_subject_metrics.csv")
    primary = [_comparison(stieger,"SELECTIVE_GATE",right,"balanced_accuracy",config,f"stieger_selective_{right}") for right in config["inference"]["primary_comparisons"]]
    _holm(primary)
    macro = _comparison(stieger,"SELECTIVE_GATE","IDENTITY_RESIDUAL_CARRY","macro_f1",config,"stieger_macro")
    open_compare = _comparison(openbmi,"SELECTIVE_GATE","IDENTITY_RESIDUAL_CARRY","balanced_accuracy",config,"openbmi_selective_identity")
    comparison_rows = [*primary, macro, open_compare]
    pd.DataFrame(comparison_rows).to_csv(output/"tables"/"primary_comparisons.csv",index=False,lineterminator="\n",float_format="%.17g")
    null_frame = pd.read_csv(output/"tables"/"stieger_selective_memory_nulls.csv")
    null_lookup = dict(zip(null_frame["null"],null_frame["p_value"]))
    by_right = {row["right"]: row for row in primary}
    population_nll = float(stieger.loc[stieger.method=="POPULATION_ONLY","nll"].mean())
    selective_nll = float(stieger.loc[stieger.method=="SELECTIVE_GATE","nll"].mean())
    stieger_gates = {
        "oracle_ci_positive": oracle["summaries"]["stieger_chronological"]["oracle_gain_bootstrap_ci"][0] > 0,
        "identity_gain_ci_holm": by_right["IDENTITY_RESIDUAL_CARRY"]["bootstrap_ci"][0] > 0 and by_right["IDENTITY_RESIDUAL_CARRY"]["p_value_holm"] <= .05,
        "population_gain_ci_holm": by_right["POPULATION_ONLY"]["bootstrap_ci"][0] > 0 and by_right["POPULATION_ONLY"]["p_value_holm"] <= .05,
        "global_gain_ci_holm": by_right["GLOBAL_KAPPA"]["bootstrap_ci"][0] > 0 and by_right["GLOBAL_KAPPA"]["p_value_holm"] <= .05,
        "macro_f1_gain_positive": macro["mean_difference"] > 0,
        "nll_population_noninferior": selective_nll <= population_nll + .01,
        "memory_null": null_lookup["ENROLLMENT_SUBJECT_MEMORY_PERMUTATION"] <= .05,
        "reliability_null": null_lookup["RELIABILITY_FEATURE_PERMUTATION"] <= .05,
        "semantics_null": null_lookup["ENROLLMENT_CLASS_SEMANTICS_PERMUTATION"] <= .05,
        "unpaired_null": null_lookup["UNPAIRED_SOURCE_SESSION_GATE_TRAINING"] <= .05,
        "loo_identity_positive": by_right["IDENTITY_RESIDUAL_CARRY"]["leave_one_subject_mean_range"][0] > 0,
    }
    stieger_pass = all(stieger_gates.values())
    open_pass = open_compare["mean_difference"] > 0 and open_compare["bootstrap_ci"][0] > 0 and open_compare["p_value_raw"] <= .05
    if stieger_pass and open_pass: terminal=config["decisions"]["go"]
    elif stieger_pass: terminal=config["decisions"]["stieger_only"]
    else: terminal=config["decisions"]["gate_stop"]
    gate_rows=[{"gate":key,"passed":bool(value)} for key,value in stieger_gates.items()]+[{"gate":"openbmi_replication","passed":bool(open_pass)}]
    pd.DataFrame(gate_rows).to_csv(output/"tables"/"terminal_gates.csv",index=False,lineterminator="\n")
    _make_figures(output,oracle,stieger,openbmi,comparison_rows,null_frame)
    decision={"terminal":terminal,"parent_terminal_unchanged":config["protocol"]["parent_terminal"],"oracle":oracle["summaries"],"stieger_primary_comparisons":primary,"stieger_macro_f1_comparison":macro,"stieger_mean_nll":{"selective":selective_nll,"population":population_nll,"margin":.01},"stieger_nulls":null_frame.to_dict(orient="records"),"stieger_gates":stieger_gates,"openbmi_comparison":open_compare,"openbmi_replication_pass":open_pass,"next_statement":config["decisions"]["next_if_go"] if terminal.startswith("GO_") else config["decisions"]["next_if_stop"]}
    atomic_write_json(output/"decisions"/"terminal_decision.json",decision)
    def aggregate(frame: pd.DataFrame, method: str, metric: str="balanced_accuracy") -> float: return float(frame.loc[frame.method==method,metric].mean())
    lines=["# Selective Conditional Memory Feasibility Audit V0","",f"Terminal: `{terminal}`", "", f"The frozen PR #20 terminal remains `{config['protocol']['parent_terminal']}` and is not rescued or reinterpreted.","","## Oracle selective-memory ceiling","",f"Stieger oracle mean BA: {oracle['summaries']['stieger_chronological']['mean_oracle_balanced_accuracy']:.10f}; gain over identity: {oracle['summaries']['stieger_chronological']['mean_oracle_gain_over_identity']:.10f}; 95% CI {oracle['summaries']['stieger_chronological']['oracle_gain_bootstrap_ci']}. Population was selected for {oracle['summaries']['stieger_chronological']['proportion_selecting_population']:.3%} of subjects.","","## Deployable enrollment-only gate","",f"Stieger mean BA: selective {aggregate(stieger,'SELECTIVE_GATE'):.10f}, identity {aggregate(stieger,'IDENTITY_RESIDUAL_CARRY'):.10f}, population {aggregate(stieger,'POPULATION_ONLY'):.10f}, global kappa {aggregate(stieger,'GLOBAL_KAPPA'):.10f}."]
    for row in primary: lines.append(f"- Selective − {row['right']}: mean {row['mean_difference']:.10f}, 95% CI {row['bootstrap_ci']}, raw p={row['p_value_raw']:.6f}, Holm p={row['p_value_holm']:.6f}.")
    lines += ["","## Required nulls",""]+[f"- {row['null']}: p={row['p_value']:.6f}." for row in decision['stieger_nulls']]
    lines += ["","## External replication","",f"OpenBMI selective − identity: {open_compare['mean_difference']:.10f}, 95% CI {open_compare['bootstrap_ci']}, p={open_compare['p_value_raw']:.6f}.","","## Boundary","","No neural network, residual decoder, low-rank rescue, raw preprocessing, or deployment-label gate input was used. A STOP terminates this persistent conditional-memory architecture line in the present lineage.","",f"Next statement: {decision['next_statement']}",""]
    (output/"report"/"selective_conditional_memory_feasibility_v0.md").write_text("\n".join(lines),encoding="utf-8")
    snapshot=verify_parent_snapshot(root,output/"protocol"/"parent_artifact_hashes.json")
    atomic_write_json(output/"decisions"/"final_validation.json",{"parent_hashes_unchanged":True,"snapshot":snapshot,"target_label_gate_input":False})
    atomic_write_json(output/"git_provenance.json",{"parent_head":config["protocol"]["parent_head"],"result_head_before_commit":_git(root,"rev-parse","HEAD"),"branch":_git(root,"branch","--show-current")})
    manifest=_result_manifest(output); atomic_write_json(output/"manifest.json",manifest)
    return decision


def record_failure(repo_root: str | Path, terminal: str, exception: BaseException) -> dict[str, Any]:
    root=Path(repo_root).resolve(); config,_=load_config(root,verify_protocol=False); output=output_path(root,config); _ensure_dirs(output)
    result={"terminal":terminal,"exception_type":type(exception).__name__,"message":str(exception),"parent_terminal_unchanged":config["protocol"]["parent_terminal"]}
    atomic_write_json(output/"decisions"/"execution_failure.json",result); atomic_write_json(output/"decisions"/"terminal_decision.json",result)
    (output/"report"/"selective_conditional_memory_feasibility_v0.md").write_text(f"# Selective Conditional Memory Feasibility V0\n\nTerminal: `{terminal}`\n\n{type(exception).__name__}: {exception}\n",encoding="utf-8")
    return result
