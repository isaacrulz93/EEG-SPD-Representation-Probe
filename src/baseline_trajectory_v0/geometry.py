"""Frozen float64 geometry and feature definitions for baseline trajectory V0.

Audited lineage, not merged:
- source branch: pilot/trial-movement-incremental-utility-v0
- source commit: 4a0e7966f676a02838e4300114667ff5e40cd9ae
- source paths: src/geometry_v2.py, src/spd_utils.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from pyriemann.geometry.mean import mean_riemann

from src.spd_utils import svec


FEATURE_ORDER = (
    "F0", "F1", "F2-S", "F2-V", "F3-G", "F3-D",
    "F2-S-SHUFFLE", "F2-S-REVERSE",
)
FEATURE_DIMENSIONS = {
    "F0": 253,
    "F1": 1265,
    "F2-S": 1265,
    "F2-V": 1265,
    "F3-G": 15,
    "F3-D": 9,
    "F2-S-SHUFFLE": 1265,
    "F2-S-REVERSE": 1265,
}


@dataclass(frozen=True)
class CovarianceBank:
    c0: np.ndarray
    local: np.ndarray
    full: np.ndarray
    metadata: pd.DataFrame
    channel_names: np.ndarray


def symmetrize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != array.shape[-2]:
        raise ValueError("expected square matrices")
    return 0.5 * (array + np.swapaxes(array, -1, -2))


def _spectral(values: np.ndarray, operation: str) -> np.ndarray:
    array = symmetrize(values)
    if not np.isfinite(array).all():
        raise ValueError("spectral input is nonfinite")
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    if operation in {"log", "sqrt", "invsqrt"} and np.any(eigenvalues <= 0.0):
        raise ValueError(f"{operation} requires SPD input")
    if operation == "log":
        transformed = np.log(eigenvalues)
    elif operation == "exp":
        transformed = np.exp(eigenvalues)
    elif operation == "sqrt":
        transformed = np.sqrt(eigenvalues)
    elif operation == "invsqrt":
        transformed = 1.0 / np.sqrt(eigenvalues)
    else:
        raise ValueError(operation)
    result = (eigenvectors * transformed[..., None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    return symmetrize(result)


def spd_log(values: np.ndarray) -> np.ndarray:
    return _spectral(values, "log")


def symmetric_exp(values: np.ndarray) -> np.ndarray:
    return _spectral(values, "exp")


def spd_sqrt(values: np.ndarray) -> np.ndarray:
    return _spectral(values, "sqrt")


def spd_invsqrt(values: np.ndarray) -> np.ndarray:
    return _spectral(values, "invsqrt")


def congruence(action: np.ndarray, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(action, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    return symmetrize(matrix @ array @ matrix.T)


def airm_mean(values: np.ndarray, *, tol: float = 1e-9, maxiter: int = 100) -> np.ndarray:
    array = symmetrize(values)
    result = mean_riemann(array, tol=tol, maxiter=maxiter, init=None)
    return symmetrize(np.asarray(result, dtype=np.float64))


def airm_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    whitening = spd_invsqrt(first)
    relative = symmetrize(whitening @ np.asarray(second) @ whitening)
    return np.linalg.norm(spd_log(relative), axis=(-2, -1))


def tangent_logs(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    whitening = spd_invsqrt(reference)
    return spd_log(symmetrize(whitening @ np.asarray(values) @ whitening))


def baseline_relative_logs(c0: np.ndarray, local: np.ndarray) -> np.ndarray:
    whitening = spd_invsqrt(c0)
    return spd_log(symmetrize(whitening[:, None] @ local @ whitening[:, None]))


def movement_logs(states: np.ndarray) -> np.ndarray:
    result = np.empty_like(states)
    result[:, 0] = states[:, 0]
    result[:, 1:] = states[:, 1:] - states[:, :-1]
    return symmetrize(result)


def gram_features(states: np.ndarray) -> np.ndarray:
    flat = np.asarray(states, dtype=np.float64).reshape(len(states), 5, -1)
    gram = np.einsum("nti,nsi->nts", flat, flat, optimize=True)
    row, column = np.triu_indices(5)
    return gram[:, row, column]


def distance_features(c0: np.ndarray, local: np.ndarray) -> np.ndarray:
    base = np.column_stack([airm_distance(c0, local[:, index]) for index in range(5)])
    speed = np.column_stack(
        [airm_distance(local[:, index], local[:, index + 1]) for index in range(4)]
    )
    return np.concatenate([base, speed], axis=1)


def fixed_nonidentity_permutations(n_trials: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    identity = np.arange(5)
    result = np.empty((n_trials, 5), dtype=np.int64)
    for index in range(n_trials):
        permutation = rng.permutation(5)
        while np.array_equal(permutation, identity):
            permutation = rng.permutation(5)
        result[index] = permutation
    return result


def apply_state_permutations(states: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    values = np.asarray(states)
    mapping = np.asarray(permutations, dtype=np.int64)
    if values.shape[:2] != mapping.shape or not np.array_equal(
        np.sort(mapping, axis=1), np.tile(np.arange(5), (len(mapping), 1))
    ):
        raise ValueError("invalid state permutation manifest")
    return values[np.arange(len(values))[:, None], mapping]


def invariant_feature_map(
    c0: np.ndarray,
    local: np.ndarray,
    permutations: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    states = baseline_relative_logs(c0, local)
    features = {
        "F2-S": svec(states).reshape(len(states), -1),
        "F2-V": svec(movement_logs(states)).reshape(len(states), -1),
        "F3-G": gram_features(states),
        "F3-D": distance_features(c0, local),
        "F2-S-SHUFFLE": svec(apply_state_permutations(states, permutations)).reshape(len(states), -1),
        "F2-S-REVERSE": svec(states[:, ::-1]).reshape(len(states), -1),
    }
    for name, values in features.items():
        if values.shape != (len(states), FEATURE_DIMENSIONS[name]) or not np.isfinite(values).all():
            raise RuntimeError(f"feature contract failed: {name}")
    return features, states


def referenced_feature_map(
    full: np.ndarray,
    local: np.ndarray,
    reference: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    full_logs = tangent_logs(full, reference)
    local_logs = tangent_logs(local, reference)
    features = {
        "F0": svec(full_logs),
        "F1": svec(local_logs).reshape(len(local_logs), -1),
    }
    matrices = {"STATIC": full_logs, "F1": local_logs}
    return features, matrices


def subject_ra(bank: CovarianceBank) -> tuple[CovarianceBank, pd.DataFrame]:
    c0 = np.empty_like(bank.c0)
    local = np.empty_like(bank.local)
    full = np.empty_like(bank.full)
    rows = []
    for subject in sorted(bank.metadata["subject"].unique()):
        positions = np.flatnonzero(bank.metadata["subject"].to_numpy() == subject)
        center = airm_mean(bank.full[positions])
        whitening = spd_invsqrt(center)
        c0[positions] = congruence(whitening, bank.c0[positions])
        local[positions] = congruence(whitening, bank.local[positions])
        full[positions] = congruence(whitening, bank.full[positions])
        rows.append(
            {
                "subject": int(subject),
                "fit_trials": len(positions),
                "labels_used": False,
                "designation": "TRANSDUCTIVE_CALIBRATION",
            }
        )
    return CovarianceBank(c0, local, full, bank.metadata.copy(), bank.channel_names.copy()), pd.DataFrame(rows)


def load_bank(path: str | Path) -> CovarianceBank:
    with np.load(Path(path), allow_pickle=False) as archive:
        c0 = np.asarray(archive["c0"], dtype=np.float64)
        local = np.asarray(archive["local"], dtype=np.float64)
        full = np.asarray(archive["full"], dtype=np.float64)
        metadata = pd.DataFrame(
            {
                "sample_index": archive["sample_index"].astype(np.int64),
                "subject": archive["subject"].astype(np.int64),
                "session": archive["session"].astype(str),
                "run": archive["run"].astype(np.int64),
                "trial_id": archive["trial_id"].astype(np.int64),
                "run_trial_id": archive["run_trial_id"].astype(np.int64),
                "trial_uid": archive["trial_uid"].astype(str),
                "class_label": archive["class_label"].astype(str),
            }
        )
        channels = archive["channel_names"].astype(str)
    if c0.shape != (5184, 22, 22) or local.shape != (5184, 5, 22, 22) or full.shape != (5184, 22, 22):
        raise RuntimeError("baseline covariance cache shape mismatch")
    if metadata["trial_uid"].duplicated().any():
        raise RuntimeError("trial UID is not unique")
    return CovarianceBank(c0, local, full, metadata, channels)


def all_feature_dimensions(features: Mapping[str, np.ndarray]) -> None:
    for name, expected in FEATURE_DIMENSIONS.items():
        if name in features and features[name].shape[1] != expected:
            raise RuntimeError(f"{name} dimension mismatch")


__all__ = [
    "CovarianceBank", "FEATURE_DIMENSIONS", "FEATURE_ORDER", "airm_distance",
    "airm_mean", "apply_state_permutations", "baseline_relative_logs",
    "congruence", "distance_features", "fixed_nonidentity_permutations",
    "gram_features", "invariant_feature_map", "load_bank", "movement_logs",
    "referenced_feature_map", "spd_invsqrt", "spd_log", "spd_sqrt",
    "subject_ra", "symmetric_exp", "tangent_logs",
]
