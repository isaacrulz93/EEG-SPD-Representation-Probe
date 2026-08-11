"""Fail-closed frozen WINDOW5 input reuse for local quotient GPA V0."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.geometry_v2 import airm_distance, symmetrize
from src.trajectory_within_subject_data_v1 import load_prepared_audit_data
from src.trajectory_within_subject_v1 import sha256_array, sha256_file


SCIENTIFIC_BASE_SHA = "796f04e7970972175a660a521caff47c83e0295f"
TRAJECTORY_CONFIG = "configs/bnci2014_001_trajectory_within_subject_v1.yaml"
SESSION0_CACHE = "cache/bnci2014_001/covariances.npz"
SESSION1_CACHE = (
    "cache/bnci2014_001_trajectory_within_subject_v1/"
    "session1_window_covariances.npz"
)
COMBINED_FEATURE_CACHE = (
    "cache/bnci2014_001_trajectory_within_subject_v1/"
    "combined_trajectory_features.npz"
)
SESSION0_FILE_SHA256 = "c6bd774ac5b3b53c497381433bcc7974af3b54b850a0ac0539aa2f797f6fa997"
SESSION1_FILE_SHA256 = "f9ce45662ab69c80b25ece5989f068f7a0a737157588600b3a1a69fb67890101"
COMBINED_FILE_SHA256 = "5accf2f3e6becce187b18d30a1ea1741ae0d0faafc15a1f8ac16632a5a71628d"
SESSION0_STATES_SHA256 = "c75044f48552f12ad088306b505b074e930f396fdcb544307fff394717e2ca86"
SESSION1_STATES_SHA256 = "1afc8cd52d82310a05857d1ffa67859427c4c9aa1302897a140ebda64d0442f8"
DISTANCE_MATRICES_SHA256 = "681d8a075eff1218e5e2b2d0e292631ead67badaccc00ec075ba428c9d5aed64"
EXPECTED_TRIALS = 5184
REPRODUCTION_TOLERANCE = 1.0e-12


class LocalGPAReproductionError(RuntimeError):
    """The immutable five-window covariance input did not reproduce."""


@dataclass(frozen=True)
class LocalGPAInput:
    states: np.ndarray
    metadata: pd.DataFrame
    reproduction_table: pd.DataFrame
    provenance: Mapping[str, Any]


def _metadata_contract(frame: pd.DataFrame) -> list[dict[str, Any]]:
    checks = (
        ("all_trials", [], 5184, 1),
        ("session", ["session"], 2592, 2),
        ("subject_session", ["subject", "session"], 288, 18),
        ("subject_session_class", ["subject", "session", "class_label"], 72, 72),
        (
            "subject_session_run_class",
            ["subject", "session", "run", "class_label"],
            12,
            432,
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, columns, expected_count, expected_groups in checks:
        counts = (
            frame.groupby(columns, sort=True, observed=True).size().to_numpy()
            if columns
            else np.asarray([len(frame)])
        )
        rows.append(
            {
                "check": f"metadata_{name}",
                "observed": f"groups={len(counts)};min={counts.min()};max={counts.max()}",
                "expected": f"groups={expected_groups};count={expected_count}",
                "maximum_absolute_difference": np.nan,
                "tolerance": np.nan,
                "passed": bool(
                    len(counts) == expected_groups
                    and np.all(counts == expected_count)
                ),
            }
        )
    return rows


def _recompute_distances(states: np.ndarray) -> np.ndarray:
    matrices = np.zeros((len(states), 5, 5), dtype=np.float64)
    for left in range(5):
        for right in range(left + 1, 5):
            distance = np.asarray(
                airm_distance(states[:, left], states[:, right]), dtype=np.float64
            )
            matrices[:, left, right] = distance
            matrices[:, right, left] = distance
    return matrices


def _file_sha256_direct(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_reproduce_local_gpa_input(root: str | Path) -> LocalGPAInput:
    project_root = Path(root).resolve()
    session0_path = project_root / SESSION0_CACHE
    session1_path = project_root / SESSION1_CACHE
    combined_path = project_root / COMBINED_FEATURE_CACHE
    file_checks = (
        (session0_path, SESSION0_FILE_SHA256),
        (session1_path, SESSION1_FILE_SHA256),
        (combined_path, COMBINED_FILE_SHA256),
    )
    rows: list[dict[str, Any]] = []
    for path, expected in file_checks:
        observed = _file_sha256_direct(path)
        rows.append(
            {
                "check": f"file_sha256:{path.relative_to(project_root)}",
                "observed": observed,
                "expected": expected,
                "maximum_absolute_difference": np.nan,
                "tolerance": np.nan,
                "passed": observed == expected,
            }
        )
    with np.load(session0_path, allow_pickle=False) as archive:
        session0 = np.asarray(archive["window5"], dtype=np.float64).reshape(
            2592, 5, 22, 22
        )
    with np.load(session1_path, allow_pickle=False) as archive:
        session1 = np.asarray(archive["states"], dtype=np.float64)
    state_hashes = (
        ("session0_states_content_sha256", session0, SESSION0_STATES_SHA256),
        ("session1_states_content_sha256", session1, SESSION1_STATES_SHA256),
    )
    for name, values, expected in state_hashes:
        observed = sha256_array(values)
        rows.append(
            {
                "check": name,
                "observed": observed,
                "expected": expected,
                "maximum_absolute_difference": np.nan,
                "tolerance": np.nan,
                "passed": observed == expected,
            }
        )
    states = symmetrize(np.concatenate((session0, session1), axis=0))
    if states.shape != (EXPECTED_TRIALS, 5, 22, 22):
        raise LocalGPAReproductionError(f"unexpected state shape {states.shape}")
    if not np.isfinite(states).all() or float(np.min(np.linalg.eigvalsh(states))) <= 0:
        raise LocalGPAReproductionError("frozen state bank is not finite SPD")
    prepared = load_prepared_audit_data(
        project_root / TRAJECTORY_CONFIG, project_root
    )
    reference_distances = np.asarray(
        prepared.arrays["airm_distance_matrices"], dtype=np.float64
    )
    reproduced_distances = _recompute_distances(states)
    maximum_difference = float(
        np.max(np.abs(reproduced_distances - reference_distances))
    )
    rows.append(
        {
            "check": "window5_states_reproduce_frozen_AIRM_distance_matrices",
            "observed": sha256_array(reproduced_distances),
            "expected": DISTANCE_MATRICES_SHA256,
            "maximum_absolute_difference": maximum_difference,
            "tolerance": REPRODUCTION_TOLERANCE,
            "passed": bool(maximum_difference <= REPRODUCTION_TOLERANCE),
        }
    )
    rows.extend(_metadata_contract(prepared.metadata))
    run_values = tuple(sorted(prepared.metadata["run"].unique().tolist()))
    rows.append(
        {
            "check": "run_labels_support_frozen_blocked_halves",
            "observed": str(run_values),
            "expected": "(0, 1, 2, 3, 4, 5)",
            "maximum_absolute_difference": np.nan,
            "tolerance": np.nan,
            "passed": run_values == (0, 1, 2, 3, 4, 5),
        }
    )
    reproduction = pd.DataFrame(rows)
    if not reproduction["passed"].astype(bool).all():
        failed = reproduction.loc[
            ~reproduction["passed"].astype(bool), "check"
        ].tolist()
        raise LocalGPAReproductionError(
            f"UNASSESSED_TECHNICAL_FAILURE_REPRODUCTION: {failed}"
        )
    provenance = {
        "scientific_base_sha": SCIENTIFIC_BASE_SHA,
        "infrastructure_source_shas": [SCIENTIFIC_BASE_SHA],
        "scientific_representation_diff": "none",
        "session0_cache": SESSION0_CACHE,
        "session1_cache": SESSION1_CACHE,
        "combined_feature_cache": COMBINED_FEATURE_CACHE,
        "session0_states_sha256": sha256_array(session0),
        "session1_states_sha256": sha256_array(session1),
        "recomputed_distance_sha256": sha256_array(reproduced_distances),
        "distance_max_abs_diff": maximum_difference,
        "trial_count": len(states),
        "new_stage2a_scientific_statistic_computed": False,
    }
    states.setflags(write=False)
    return LocalGPAInput(
        states=states,
        metadata=prepared.metadata.copy(),
        reproduction_table=reproduction,
        provenance=provenance,
    )


__all__ = [
    "COMBINED_FEATURE_CACHE",
    "EXPECTED_TRIALS",
    "LocalGPAInput",
    "LocalGPAReproductionError",
    "REPRODUCTION_TOLERANCE",
    "SCIENTIFIC_BASE_SHA",
    "SESSION0_CACHE",
    "SESSION1_CACHE",
    "load_and_reproduce_local_gpa_input",
]
