"""Fail-closed reuse and reproduction of the frozen trajectory V1 input."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.local_metric_geometry_v0 import EDGE_ORDER, edge_rms_size
from src.trajectory_within_subject_data_v1 import (
    PreparedAuditData,
    load_prepared_audit_data,
)
from src.trajectory_within_subject_v1 import sha256_array, sha256_file


SCIENTIFIC_BASE_SHA = "355fe0b55b1ef692f7b4ddd16d19b7ccc30e72e1"
TRAJECTORY_CONFIG = "configs/bnci2014_001_trajectory_within_subject_v1.yaml"
FROZEN_CACHE = "cache/bnci2014_001_trajectory_within_subject_v1/combined_trajectory_features.npz"
FROZEN_CACHE_SHA256 = "5accf2f3e6becce187b18d30a1ea1741ae0d0faafc15a1f8ac16632a5a71628d"
FROZEN_DISTANCE_SHA256 = "681d8a075eff1218e5e2b2d0e292631ead67badaccc00ec075ba428c9d5aed64"
FROZEN_PATH_SHA256 = "8179f7654d6a1c89065aca12e99029e6a7476d332fdf62767e83c0f0966008f9"
EXPECTED_TRIALS = 5184


class LocalMetricReproductionError(RuntimeError):
    """The immutable trajectory input did not reproduce exactly."""


@dataclass(frozen=True)
class LocalMetricInput:
    metadata: pd.DataFrame
    edges: np.ndarray
    distance_matrices: np.ndarray
    reproduction_table: pd.DataFrame
    degeneracy_table: pd.DataFrame
    provenance: Mapping[str, Any]


def _path_from_distance_matrices(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    if values.shape != (EXPECTED_TRIALS, 5, 5):
        raise LocalMetricReproductionError(
            f"distance matrices have unexpected shape {values.shape}"
        )
    return np.asarray(values[:, [left for left, _ in EDGE_ORDER], [right for _, right in EDGE_ORDER]])


def _metadata_contract(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks = (
        ("all_trials", [], EXPECTED_TRIALS, 1),
        ("session", ["session"], 2592, 2),
        ("subject_session", ["subject", "session"], 288, 18),
        ("subject_session_class", ["subject", "session", "class_label"], 72, 72),
        ("subject_session_run_class", ["subject", "session", "run", "class_label"], 12, 432),
    )
    for name, columns, expected_per_group, expected_groups in checks:
        if columns:
            counts = frame.groupby(columns, sort=True, observed=True).size().to_numpy()
        else:
            counts = np.asarray([len(frame)])
        passed = (
            len(counts) == expected_groups
            and np.all(counts == expected_per_group)
        )
        rows.append(
            {
                "check": name,
                "group_count": int(len(counts)),
                "minimum_count": int(np.min(counts)),
                "maximum_count": int(np.max(counts)),
                "expected_group_count": expected_groups,
                "expected_count_per_group": expected_per_group,
                "passed": bool(passed),
            }
        )
    return rows


def load_and_reproduce_local_metric_input(root: str | Path) -> LocalMetricInput:
    project_root = Path(root).resolve()
    cache_path = project_root / FROZEN_CACHE
    if sha256_file(cache_path) != FROZEN_CACHE_SHA256:
        raise LocalMetricReproductionError("frozen combined cache SHA-256 mismatch")
    prepared: PreparedAuditData = load_prepared_audit_data(
        project_root / TRAJECTORY_CONFIG, project_root
    )
    matrices = np.asarray(prepared.arrays["airm_distance_matrices"], dtype=np.float64)
    edges = np.asarray(prepared.arrays["airm_path_d10"], dtype=np.float64)
    reconstructed = _path_from_distance_matrices(matrices)
    maximum_absolute_difference = float(np.max(np.abs(reconstructed - edges)))
    checks: list[dict[str, Any]] = [
        {
            "check": "frozen_combined_cache_file_sha256",
            "observed": sha256_file(cache_path),
            "expected": FROZEN_CACHE_SHA256,
            "maximum_absolute_difference": np.nan,
            "tolerance": np.nan,
            "passed": True,
        },
        {
            "check": "airm_distance_matrices_content_sha256",
            "observed": sha256_array(matrices),
            "expected": FROZEN_DISTANCE_SHA256,
            "maximum_absolute_difference": np.nan,
            "tolerance": np.nan,
            "passed": sha256_array(matrices) == FROZEN_DISTANCE_SHA256,
        },
        {
            "check": "airm_path_d10_content_sha256",
            "observed": sha256_array(edges),
            "expected": FROZEN_PATH_SHA256,
            "maximum_absolute_difference": np.nan,
            "tolerance": np.nan,
            "passed": sha256_array(edges) == FROZEN_PATH_SHA256,
        },
        {
            "check": "distance_matrix_upper_triangle_equals_path_d10",
            "observed": f"max_abs_diff={maximum_absolute_difference:.17g}",
            "expected": "max_abs_diff=0",
            "maximum_absolute_difference": maximum_absolute_difference,
            "tolerance": 0.0,
            "passed": maximum_absolute_difference == 0.0,
        },
    ]
    checks.extend(
        {
            "check": f"metadata_{row['check']}",
            "observed": (
                f"groups={row['group_count']};min={row['minimum_count']};"
                f"max={row['maximum_count']}"
            ),
            "expected": (
                f"groups={row['expected_group_count']};"
                f"count={row['expected_count_per_group']}"
            ),
            "maximum_absolute_difference": np.nan,
            "tolerance": np.nan,
            "passed": row["passed"],
        }
        for row in _metadata_contract(prepared.metadata)
    )
    reproduction = pd.DataFrame(checks)
    if not reproduction["passed"].astype(bool).all():
        failed = reproduction.loc[~reproduction["passed"].astype(bool), "check"].tolist()
        raise LocalMetricReproductionError(
            f"UNASSESSED_TECHNICAL_FAILURE_REPRODUCTION: {failed}"
        )
    sizes = np.asarray(edge_rms_size(edges), dtype=np.float64)
    degenerate_mask = sizes == 0.0
    degeneracy = prepared.metadata.loc[
        degenerate_mask,
        ["global_sample_index", "trial_uid", "subject", "session", "run", "class_label"],
    ].copy()
    degeneracy["edge_rms_metric_size"] = sizes[degenerate_mask]
    provenance = {
        "scientific_base_sha": SCIENTIFIC_BASE_SHA,
        "infrastructure_source_shas": [SCIENTIFIC_BASE_SHA],
        "infrastructure_diff_to_scientific_representation": "none",
        "frozen_trajectory_config": TRAJECTORY_CONFIG,
        "frozen_combined_cache": FROZEN_CACHE,
        "frozen_cache_sha256": FROZEN_CACHE_SHA256,
        "distance_content_sha256": sha256_array(matrices),
        "path_d10_content_sha256": sha256_array(edges),
        "upper_triangle_path_max_abs_diff": maximum_absolute_difference,
        "trial_count": int(len(edges)),
        "degenerate_trial_count": int(np.count_nonzero(degenerate_mask)),
        "scientific_statistics_computed": False,
    }
    edges.setflags(write=False)
    matrices.setflags(write=False)
    return LocalMetricInput(
        metadata=prepared.metadata.copy(),
        edges=edges,
        distance_matrices=matrices,
        reproduction_table=reproduction,
        degeneracy_table=degeneracy,
        provenance=provenance,
    )


def write_reproduction_outputs(
    data: LocalMetricInput,
    output_root: str | Path,
    *,
    protocol_sha256: str,
    config_sha256: str,
    implementation_sha256: str,
) -> None:
    destination = Path(output_root)
    (destination / "tables").mkdir(parents=True, exist_ok=True)
    (destination / "protocol").mkdir(parents=True, exist_ok=True)
    data.reproduction_table.to_csv(
        destination / "tables" / "frozen_representation_reproduction.csv", index=False
    )
    data.degeneracy_table.to_csv(
        destination / "tables" / "degenerate_configuration_audit.csv", index=False
    )
    provenance = dict(data.provenance)
    provenance.update(
        {
            "protocol_sha256": protocol_sha256,
            "config_sha256": config_sha256,
            "implementation_sha256": implementation_sha256,
        }
    )
    (destination / "protocol" / "input_reproduction_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "EXPECTED_TRIALS",
    "FROZEN_CACHE",
    "FROZEN_CACHE_SHA256",
    "FROZEN_DISTANCE_SHA256",
    "FROZEN_PATH_SHA256",
    "LocalMetricInput",
    "LocalMetricReproductionError",
    "SCIENTIFIC_BASE_SHA",
    "load_and_reproduce_local_metric_input",
    "write_reproduction_outputs",
]
