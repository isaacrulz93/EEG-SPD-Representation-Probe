"""Core frozen object construction for Subject Class Interaction v0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.conditional_geometry_v1 import (
    GeometryThresholds,
    airm_mean_official,
    karcher_residual,
    relative_frobenius_error,
    spd_invsqrt,
    spd_log,
    symmetrize,
    symmetric_exp,
    validate_spd_stack,
)
from src.interaction_provenance_v0 import sha256_array, sha256_file
from src.interaction_statistics_v0 import normalize_rows
from src.spd_utils import svec


PRIMARY_CONFIG = "configs/subject_class_interaction_v0.yaml"
BNCI_CLASSES = ("left_hand", "right_hand", "feet", "tongue")
BNCI_SESSIONS = ("0train", "1test")
BNCI_SUBJECTS = tuple(range(1, 10))
BNCI_RUNS = tuple(str(value) for value in range(6))


class InteractionError(RuntimeError):
    pass


class DataContractError(InteractionError):
    pass


class NumericalGateError(InteractionError):
    pass


@dataclass(frozen=True)
class InteractionObjects:
    geometry: str
    template: str
    subjects: tuple[int, ...]
    sessions: tuple[str, ...]
    classes: tuple[str, ...]
    marginal_means: np.ndarray
    class_means: np.ndarray
    class_counts: np.ndarray
    class_proportions: np.ndarray
    U: np.ndarray
    population_templates: np.ndarray
    R: np.ndarray
    Rbar: np.ndarray
    Z: np.ndarray
    sensor_raw: Mapping[str, np.ndarray]
    sensor: Mapping[str, np.ndarray]
    sensor_norms: Mapping[str, np.ndarray]
    spectrum_raw: Mapping[str, np.ndarray]
    spectrum: Mapping[str, np.ndarray]
    spectrum_norms: Mapping[str, np.ndarray]
    per_class_norms: Mapping[str, np.ndarray]
    mean_audit_rows: tuple[dict[str, Any], ...]
    gate_metrics: Mapping[str, float]
    object_hashes: Mapping[str, str]


def load_frozen_config(repo_root: str | Path, path: str | Path = PRIMARY_CONFIG) -> tuple[dict[str, Any], str]:
    root = Path(repo_root).resolve()
    config_path = root / path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise DataContractError("frozen config must be a mapping")
    protocol_path = root / str(config["protocol"]["protocol_path"])
    observed_protocol_hash = sha256_file(protocol_path)
    expected_protocol_hash = str(config["protocol"]["protocol_sha256"])
    if observed_protocol_hash != expected_protocol_hash:
        raise DataContractError(
            f"protocol SHA mismatch: {observed_protocol_hash} != {expected_protocol_hash}"
        )
    config_hash = sha256_file(config_path)
    return config, config_hash


def _content_hash_without_metadata(array: np.ndarray) -> str:
    import hashlib

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def load_bnci_two_session_data(repo_root: str | Path, config: Mapping[str, Any]) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    root = Path(repo_root).resolve()
    frozen = config["datasets"]["bnci2014_001"]["frozen_inputs"]
    discovery_cov_path = root / str(frozen["covariance_path"])
    discovery_meta_path = root / str(frozen["metadata_path"])
    confirm_cov_path = root / "cache/bnci2014_001_conditional_geometry_v1/confirmatory_whole_covariances.npz"
    confirm_meta_path = root / "cache/bnci2014_001_conditional_geometry_v1/confirmatory_whole_metadata.csv"
    confirm_provenance_path = root / "cache/bnci2014_001_conditional_geometry_v1/confirmatory_data_provenance.json"
    for path in (discovery_cov_path, discovery_meta_path, confirm_cov_path, confirm_meta_path, confirm_provenance_path):
        if not path.is_file():
            raise DataContractError(f"required frozen BNCI input is missing: {path}")
    if sha256_file(discovery_cov_path) != str(frozen["expected_covariance_file_sha256"]):
        raise DataContractError("BNCI 0train covariance file SHA mismatch")
    if sha256_file(discovery_meta_path) != str(frozen["expected_metadata_sha256"]):
        raise DataContractError("BNCI 0train metadata SHA mismatch")

    import json

    confirm_provenance = json.loads(confirm_provenance_path.read_text(encoding="utf-8"))
    if sha256_file(confirm_cov_path) != confirm_provenance["cache_covariances_sha256"]:
        raise DataContractError("BNCI 1test covariance file SHA mismatch against prior provenance")
    if sha256_file(confirm_meta_path) != confirm_provenance["cache_metadata_sha256"]:
        raise DataContractError("BNCI 1test metadata SHA mismatch against prior provenance")

    with np.load(discovery_cov_path, allow_pickle=False) as archive:
        discovery = np.asarray(archive["whole"], dtype=np.float64)
        channels0 = tuple(str(value) for value in archive["channel_names"].tolist())
    with np.load(confirm_cov_path, allow_pickle=False) as archive:
        key = "whole" if "whole" in archive.files else "covariances"
        confirmatory = np.asarray(archive[key], dtype=np.float64)
        if "channel_names" in archive.files:
            channels1 = tuple(str(value) for value in archive["channel_names"].tolist())
        else:
            channels1 = tuple(config["datasets"]["bnci2014_001"]["eeg_channels"])
    metadata0 = pd.read_csv(discovery_meta_path)
    metadata1 = pd.read_csv(confirm_meta_path)
    covariances = np.concatenate([discovery, confirmatory], axis=0)
    metadata = pd.concat([metadata0, metadata1], ignore_index=True)
    metadata["session"] = metadata["session"].astype(str)
    metadata["run"] = metadata["run"].astype(str)
    metadata["class_label"] = metadata["class_label"].astype(str)
    metadata["subject"] = pd.to_numeric(metadata["subject"], errors="raise").astype(int)
    if channels0 != tuple(config["datasets"]["bnci2014_001"]["eeg_channels"]) or channels1 != channels0:
        raise DataContractError("BNCI channel names/order differ from frozen 22-channel contract")
    if _content_hash_without_metadata(discovery) != str(frozen["expected_covariance_array_sha256"]):
        raise DataContractError("BNCI 0train covariance content SHA mismatch")
    if _content_hash_without_metadata(confirmatory) != confirm_provenance["whole_array_content_sha256"]:
        raise DataContractError("BNCI 1test covariance content SHA mismatch")
    validate_bnci_contract(covariances, metadata, config)
    provenance = {
        "0train_covariance_file_sha256": sha256_file(discovery_cov_path),
        "0train_covariance_array_sha256": _content_hash_without_metadata(discovery),
        "0train_metadata_sha256": sha256_file(discovery_meta_path),
        "1test_covariance_file_sha256": sha256_file(confirm_cov_path),
        "1test_covariance_array_sha256": _content_hash_without_metadata(confirmatory),
        "1test_metadata_sha256": sha256_file(confirm_meta_path),
        "combined_covariance_hash": sha256_array(covariances),
        "rows": int(len(metadata)),
        "shape": list(covariances.shape),
        "channel_names": list(channels0),
    }
    return covariances, metadata, provenance


def validate_bnci_contract(covariances: np.ndarray, metadata: pd.DataFrame, config: Mapping[str, Any]) -> None:
    values = np.asarray(covariances, dtype=np.float64)
    if values.shape != (5184, 22, 22) or len(metadata) != 5184:
        raise DataContractError(f"BNCI expected (5184,22,22), observed {values.shape} and {len(metadata)} rows")
    required = {"covariance_index", "subject", "session", "run", "trial_uid", "class_label"}
    if required - set(metadata.columns):
        raise DataContractError(f"BNCI metadata missing columns: {sorted(required - set(metadata.columns))}")
    if metadata["trial_uid"].duplicated().any():
        raise DataContractError("BNCI trial_uid values are not globally unique across sessions")
    if tuple(sorted(metadata["subject"].unique())) != BNCI_SUBJECTS:
        raise DataContractError("BNCI subject contract failure")
    if tuple(metadata["session"].drop_duplicates()) != BNCI_SESSIONS:
        raise DataContractError("BNCI session/order contract failure")
    if set(metadata["class_label"].unique()) != set(BNCI_CLASSES):
        raise DataContractError("BNCI class contract failure")
    counts = metadata.groupby(["subject", "session", "class_label"], observed=True).size()
    if len(counts) != 72 or not np.all(counts.to_numpy() == 72):
        raise DataContractError(f"BNCI subject/session/class counts mismatch: {counts.to_dict()}")
    run_counts = metadata.groupby(["subject", "session", "run", "class_label"], observed=True).size()
    if len(run_counts) != 432 or not np.all(run_counts.to_numpy() == 12):
        raise DataContractError("BNCI subject/session/run/class counts mismatch")
    thresholds = geometry_thresholds(config)
    audit = validate_spd_stack(values, thresholds=thresholds)
    if not audit.all_passed:
        raise NumericalGateError("UNASSESSED_NUMERICAL_OR_DATA_FAILURE: covariance SPD gate failed")


def geometry_thresholds(config: Mapping[str, Any]) -> GeometryThresholds:
    gates = config["hard_gates"]
    mean = config["geometry"]["mean_riemann"]
    return GeometryThresholds(
        mean_tol=float(mean["tol"]),
        mean_maxiter=int(mean["maxiter"]),
        symmetry_relative_error_max=float(gates["covariance_symmetry_relative_error_max"]),
        condition_number_max=float(gates["covariance_condition_number_max"]),
        airm_karcher_residual_max=float(gates["airm_karcher_residual_max"]),
        master_seed=int(config["protocol"]["master_seed"]),
    )


def split_masks(metadata: pd.DataFrame, dataset: str) -> dict[str, np.ndarray]:
    if dataset == "bnci2014_001":
        runs = metadata["run"].astype(str).to_numpy()
        a = np.isin(runs, ["0", "1", "2"])
        b = np.isin(runs, ["3", "4", "5"])
    elif dataset == "openbmi_lee2019_mi":
        order = metadata.groupby(["subject", "session", "class_label"], sort=False).cumcount().to_numpy()
        a = order % 2 == 0
        b = ~a
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    if np.any(a & b) or not np.all(a | b):
        raise DataContractError("split halves overlap or fail to exhaust trials")
    return {"A": a, "B": b, "F": np.ones(len(metadata), dtype=bool)}


def _mean(
    covariances: np.ndarray,
    geometry: str,
    thresholds: GeometryThresholds,
    name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if geometry == "AIRM":
        audit = airm_mean_official(covariances, name=name, thresholds=thresholds)
        if not audit.passed:
            raise NumericalGateError(f"AIRM mean gate failed for {name}: {audit.failure_reasons}")
        return audit.matrix, {
            "name": name, "geometry": geometry, "n_trials": int(len(covariances)),
            "karcher_residual": float(audit.karcher_residual),
            "warnings": "|".join(audit.warning_messages), "passed": True,
        }
    if geometry == "LE":
        logs = spd_log(covariances)
        matrix = symmetric_exp(np.mean(logs, axis=0, dtype=np.float64))
        spd_audit = validate_spd_stack(matrix, thresholds=thresholds)
        if not spd_audit.all_passed:
            raise NumericalGateError(f"LE mean SPD gate failed for {name}")
        return matrix, {
            "name": name, "geometry": geometry, "n_trials": int(len(covariances)),
            "karcher_residual": np.nan, "warnings": "", "passed": True,
        }
    raise ValueError("geometry must be AIRM or LE")


def _signature_bank(objects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if objects.ndim != 5:
        raise ValueError("objects must have shape (subject,session,class,p,p)")
    sensor_raw = svec(objects).reshape(objects.shape[0], objects.shape[1], -1)
    sensor, sensor_norms = normalize_rows(sensor_raw)
    eigenvalues = np.linalg.eigvalsh(symmetrize(objects))
    spectrum_raw = eigenvalues.reshape(objects.shape[0], objects.shape[1], -1)
    spectrum, spectrum_norms = normalize_rows(spectrum_raw)
    return sensor_raw, sensor, sensor_norms, spectrum_raw, spectrum, spectrum_norms


def build_interactions_from_means(
    *,
    marginal_means: np.ndarray,
    class_means: np.ndarray,
    class_counts: np.ndarray,
    geometry: str,
    template: str,
    subjects: Sequence[int],
    sessions: Sequence[str],
    classes: Sequence[str],
    config: Mapping[str, Any],
    mean_audit_rows: Sequence[dict[str, Any]] = (),
) -> InteractionObjects:
    marginal = np.asarray(marginal_means, dtype=np.float64)
    class_values = np.asarray(class_means, dtype=np.float64)
    counts = np.asarray(class_counts, dtype=np.int64)
    n_subjects, n_sessions, n_classes, p, p2 = class_values.shape
    if p != p2 or marginal.shape != (n_subjects, n_sessions, p, p) or counts.shape != (n_subjects, n_sessions, n_classes):
        raise ValueError("mean/count shapes are inconsistent")
    if np.any(counts <= 0):
        raise DataContractError("every subject/session/class requires at least one trial")
    proportions = counts / np.sum(counts, axis=2, keepdims=True)
    proportion_error = float(np.max(np.abs(np.sum(proportions, axis=2) - 1.0)))
    if proportion_error > float(config["hard_gates"]["class_proportion_sum_absolute_error_max"]):
        raise NumericalGateError("class proportions do not sum to one")

    U = np.empty_like(class_values)
    identity_errors: list[float] = []
    for s in range(n_subjects):
        for q in range(n_sessions):
            if geometry == "AIRM":
                inverse_root = spd_invsqrt(marginal[s, q])
                centered_marginal = symmetrize(inverse_root @ marginal[s, q] @ inverse_root)
                identity_errors.append(relative_frobenius_error(centered_marginal, np.eye(p)))
                U[s, q] = spd_log(symmetrize(inverse_root @ class_values[s, q] @ inverse_root))
            elif geometry == "LE":
                identity_errors.append(relative_frobenius_error(spd_log(marginal[s, q]) - spd_log(marginal[s, q]), np.zeros((p, p))))
                U[s, q] = spd_log(class_values[s, q]) - spd_log(marginal[s, q])
            else:
                raise ValueError("geometry must be AIRM or LE")
    U = symmetrize(U)
    population = np.empty_like(U)
    for target in range(n_subjects):
        sources = np.asarray([index for index in range(n_subjects) if index != target], dtype=np.int64)
        if target in sources or len(sources) != n_subjects - 1:
            raise RuntimeError("LOSO population template includes target")
        for q in range(n_sessions):
            if template == "session_specific":
                population[target, q] = np.mean(U[sources, q], axis=0, dtype=np.float64)
            elif template == "pooled_session":
                population[target, q] = np.mean(U[sources], axis=(0, 1), dtype=np.float64)
            else:
                raise ValueError("template must be session_specific or pooled_session")
    R = symmetrize(U - population)
    Rbar = symmetrize(np.einsum("sqc,sqcij->sqij", proportions, R, optimize=True))
    Z = symmetrize(R - Rbar[:, :, None])
    weighted_z = np.einsum("sqc,sqcij->sqij", proportions, Z, optimize=True)
    z_denominator = np.maximum(np.linalg.norm(Z, axis=(-2, -1)).sum(axis=2), np.finfo(np.float64).tiny)
    weighted_z_error = float(np.max(np.linalg.norm(weighted_z, axis=(-2, -1)) / z_denominator))
    symmetry_error = float(
        max(
            np.max(np.linalg.norm(U - np.swapaxes(U, -1, -2), axis=(-2, -1))),
            np.max(np.linalg.norm(Z - np.swapaxes(Z, -1, -2), axis=(-2, -1))),
        )
    )
    gates = config["hard_gates"]
    if max(identity_errors) > float(gates["marginal_centering_identity_relative_error_max"]):
        raise NumericalGateError("marginal centering identity gate failed")
    if weighted_z_error > float(gates["weighted_Z_sum_relative_error_max"]):
        raise NumericalGateError("weighted class mean of Z is not zero")
    if symmetry_error > float(gates["tangent_symmetry_relative_error_max"]):
        raise NumericalGateError("U/Z symmetry gate failed")

    sensor_raw: dict[str, np.ndarray] = {}
    sensor: dict[str, np.ndarray] = {}
    sensor_norms: dict[str, np.ndarray] = {}
    spectrum_raw: dict[str, np.ndarray] = {}
    spectrum: dict[str, np.ndarray] = {}
    spectrum_norms: dict[str, np.ndarray] = {}
    for name, values in (("R", R), ("Z", Z)):
        sr, sn, srn, er, en, ern = _signature_bank(values)
        sensor_raw[name], sensor[name], sensor_norms[name] = sr, sn, srn
        spectrum_raw[name], spectrum[name], spectrum_norms[name] = er, en, ern
    return InteractionObjects(
        geometry=geometry, template=template, subjects=tuple(int(x) for x in subjects),
        sessions=tuple(str(x) for x in sessions), classes=tuple(str(x) for x in classes),
        marginal_means=marginal, class_means=class_values, class_counts=counts,
        class_proportions=proportions, U=U, population_templates=population, R=R,
        Rbar=Rbar, Z=Z, sensor_raw=sensor_raw, sensor=sensor,
        sensor_norms=sensor_norms, spectrum_raw=spectrum_raw, spectrum=spectrum,
        spectrum_norms=spectrum_norms,
        per_class_norms={"U": np.linalg.norm(U, axis=(-2, -1)), "R": np.linalg.norm(R, axis=(-2, -1)), "Z": np.linalg.norm(Z, axis=(-2, -1))},
        mean_audit_rows=tuple(mean_audit_rows),
        gate_metrics={
            "max_marginal_identity_relative_error": float(max(identity_errors)),
            "max_class_proportion_sum_absolute_error": proportion_error,
            "max_weighted_Z_sum_relative_error": weighted_z_error,
            "max_U_Z_absolute_symmetry_error": symmetry_error,
        },
        object_hashes={"U": sha256_array(U), "R": sha256_array(R), "Rbar": sha256_array(Rbar), "Z": sha256_array(Z)},
    )


def compute_interactions(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    geometry: str,
    template: str = "session_specific",
    labels: np.ndarray | None = None,
    classes: Sequence[str] | None = None,
) -> InteractionObjects:
    values = np.asarray(covariances, dtype=np.float64)
    frame = metadata.reset_index(drop=True).copy()
    if values.ndim != 3 or values.shape[0] != len(frame) or values.shape[1] != values.shape[2]:
        raise DataContractError("covariances/metadata are not trial-aligned")
    label_values = frame["class_label"].astype(str).to_numpy() if labels is None else np.asarray(labels).astype(str)
    if label_values.shape != (len(frame),):
        raise ValueError("labels must align with metadata")
    classes = (
        tuple(str(value) for value in config["datasets"]["bnci2014_001"]["classes"])
        if classes is None
        else tuple(str(value) for value in classes)
    )
    subjects = tuple(sorted(int(value) for value in frame["subject"].unique()))
    sessions = tuple(str(value) for value in frame["session"].drop_duplicates())
    thresholds = geometry_thresholds(config)
    p = values.shape[-1]
    marginal = np.empty((len(subjects), len(sessions), p, p), dtype=np.float64)
    class_means = np.empty((len(subjects), len(sessions), len(classes), p, p), dtype=np.float64)
    counts = np.empty((len(subjects), len(sessions), len(classes)), dtype=np.int64)
    audits: list[dict[str, Any]] = []
    subject_array = frame["subject"].to_numpy(dtype=np.int64)
    session_array = frame["session"].astype(str).to_numpy()
    for s, subject in enumerate(subjects):
        for q, session in enumerate(sessions):
            group = (subject_array == subject) & (session_array == session)
            marginal[s, q], row = _mean(values[group], geometry, thresholds, f"S{subject}_{session}_marginal")
            audits.append(row)
            for c, class_name in enumerate(classes):
                selected = group & (label_values == class_name)
                counts[s, q, c] = int(np.count_nonzero(selected))
                if counts[s, q, c] < 1:
                    raise DataContractError(f"empty class after label assignment: S{subject}/{session}/{class_name}")
                class_means[s, q, c], row = _mean(values[selected], geometry, thresholds, f"S{subject}_{session}_{class_name}")
                audits.append(row)
    return build_interactions_from_means(
        marginal_means=marginal, class_means=class_means, class_counts=counts,
        geometry=geometry, template=template, subjects=subjects, sessions=sessions,
        classes=classes, config=config, mean_audit_rows=audits,
    )
