"""Synthetic contracts for Conditional-Geometry Anatomy v1 reporting."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.conditional_geometry_v1 import (
    airm_distance_matrix,
    airm_gram_matrices,
    le_distance_matrix,
    le_gram_matrix,
    svec,
)
from src.reporting_conditional_v1 import (
    CLASSES,
    COMMON_PREFIX,
    FIGURE_STEMS,
    GEOMETRIES,
    GROUP_TABLE_STAGE,
    OBJECTS,
    POST_CONFIRMATORY_TABLES,
    REPORT_HEADINGS,
    REPORT_TITLE,
    REQUIRED_NULLS,
    REQUIRED_OBJECTS,
    REQUIRED_TABLES,
    TABLE_REQUIRED_COLUMNS,
    ReportingContractError,
    create_reporting_outputs,
    compute_frozen_verdicts,
    load_and_validate_reporting_inputs,
    render_report,
)
from src.conditional_provenance_v1 import canonical_json_bytes
from src.conditional_statistics_v1 import (
    leave_one_subject_out_influence,
    subject_bootstrap_median,
    subject_bootstrap_paired_median_delta,
)


ROOT = Path(__file__).resolve().parents[1]
CODE_COMMIT = "a" * 40


def _stage_operands(
    *,
    passing: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    if passing is None:
        passing = {(geometry, object_name) for geometry in ("AIRM", "LE") for object_name in ("D", "G")}
    rows: list[dict[str, object]] = []
    for geometry in ("AIRM", "LE"):
        for object_name in ("D", "G"):
            does_pass = (geometry, object_name) in passing
            for index, stage in enumerate(("R", "S", "P")):
                rows.append(
                    {
                        "geometry": geometry,
                        "object": object_name,
                        "stage": stage,
                        "discovery_effect": 0.10 + 0.01 * index,
                        "confirmatory_effect": (0.08 + 0.01 * index) if does_pass else -0.01,
                        "confirmatory_p": 0.01 if does_pass else 0.5,
                    }
                )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(repo_root: Path, paths: list[Path]) -> dict[str, object]:
    records = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(paths, key=lambda item: item.relative_to(repo_root).as_posix())
    ]
    return {
        "algorithm": "sha256_canonical_json_sorted_file_records_v1",
        "file_count": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
        "files": records,
        "aggregate_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    }


def _prefix(
    *,
    protocol_sha: str,
    config_sha: str,
    phase: str,
    status: str = "PASS",
) -> dict[str, object]:
    return {
        "protocol_version": "1.0",
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "code_commit": CODE_COMMIT,
        "phase": phase,
        "session": {"discovery": "0train", "confirmatory": "1test", "combined": "0train+1test"}[phase],
        "status": status,
    }


def _ordered_table(filename: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = [*COMMON_PREFIX, *TABLE_REQUIRED_COLUMNS[filename]]
    missing = set(required) - set(frame.columns)
    assert not missing, (filename, missing)
    extras = [column for column in frame.columns if column not in required]
    return frame.loc[:, [*required, *extras]]


def _object_values(phase_offset: float) -> dict[str, dict[str, np.ndarray]]:
    subjects = np.arange(1, 10, dtype=np.int64)
    splits = np.asarray(("A", "B", "F"))
    classes = np.asarray(CLASSES)
    identity = np.eye(22, dtype=np.float64)
    marginal = np.stack(
        [
            np.stack(
                [identity * (1.007 + 0.01 * subject + 0.001 * split + phase_offset) for split in range(3)]
            )
            for subject in range(9)
        ]
    )
    class_means = np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [identity * (1.0 + 0.01 * subject + 0.001 * split + 0.02 * cls + phase_offset)
                         for cls in range(4)]
                    )
                    for split in range(3)
                ]
            )
            for subject in range(9)
        ]
    )
    le_marginal = marginal * 1.001
    le_class_means = class_means * 1.001
    D = np.empty((2, 9, 3, 4, 4), dtype=np.float64)
    G = np.empty_like(D)
    for geometry in range(2):
        for subject in range(9):
            for split in range(3):
                if geometry == 0:
                    local_marginal = marginal[subject, split]
                    local_classes = class_means[subject, split]
                    D[geometry, subject, split] = airm_distance_matrix(local_classes)
                    G[geometry, subject, split] = airm_gram_matrices(
                        local_marginal, local_classes
                    )[0]
                else:
                    local_marginal = le_marginal[subject, split]
                    local_classes = le_class_means[subject, split]
                    D[geometry, subject, split] = le_distance_matrix(local_classes)
                    G[geometry, subject, split] = le_gram_matrix(
                        local_marginal, local_classes
                    )[0]
    axes = {"subjects": subjects, "splits": splits, "classes": classes}
    return {
        "airm_marginal_means.npz": {
            "marginal_means": marginal,
            "geometries": np.asarray(["AIRM"]),
            **axes,
        },
        "airm_class_means.npz": {
            "class_means": class_means,
            "geometries": np.asarray(["AIRM"]),
            **axes,
        },
        "le_marginal_means.npz": {
            "marginal_means": le_marginal,
            "geometries": np.asarray(["LE"]),
            **axes,
        },
        "le_class_means.npz": {
            "class_means": le_class_means,
            "geometries": np.asarray(["LE"]),
            **axes,
        },
        "D_matrices.npz": {
            "matrices": D,
            "geometries": np.asarray(GEOMETRIES),
            **axes,
        },
        "G_matrices.npz": {
            "matrices": G,
            "geometries": np.asarray(GEOMETRIES),
            **axes,
        },
    }


def _phase_tables(
    *,
    phase: str,
    protocol_sha: str,
    config_sha: str,
    null_values: dict[str, np.ndarray],
    passing_chains: set[tuple[str, str]],
    failure: str | None,
    object_values: dict[str, dict[str, np.ndarray]],
) -> dict[str, pd.DataFrame]:
    normal = _prefix(protocol_sha=protocol_sha, config_sha=config_sha, phase=phase)
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in REQUIRED_TABLES if name not in POST_CONFIRMATORY_TABLES}
    data_failure = failure == "data" and phase == "confirmatory"
    numerical_failure = failure == "numerical" and phase == "confirmatory"
    degenerate_failure = failure == "degenerate" and phase == "confirmatory"
    rows["dataset_contract.csv"].append(
        {
            **_prefix(protocol_sha=protocol_sha, config_sha=config_sha, phase=phase,
                      status="FAIL" if data_failure else "PASS"),
            "scope": "session", "subject": "", "run": "", "class_label": "", "split": "",
            "observed_count": 17 if data_failure else 18, "expected_count": 18,
            "passed": not data_failure,
        }
    )
    for index in range(18):
        rows["covariance_sanity.csv"].append(
            {
                **normal, "covariance_index": index, "trial_uid": f"{phase}-{index}",
                "subject": index % 9 + 1, "run": str(index % 6),
                "class_label": CLASSES[index % 4], "finite": True,
                "symmetry_relative_error": 0.0, "min_eigenvalue": 0.5,
                "max_eigenvalue": 2.0, "condition_number": 4.0, "passed": True,
            }
        )
    first_mean = True
    for geometry, filename in (("AIRM", "airm_mean_convergence.csv"), ("LE", "le_mean_correctness.csv")):
        for subject in range(1, 10):
            for split in ("A", "B", "F"):
                for mean_kind, labels in (("marginal", ("",)), ("class", CLASSES)):
                    for class_label in labels:
                        failed = numerical_failure and first_mean and geometry == "AIRM"
                        first_mean = False if geometry == "AIRM" else first_mean
                        rows[filename].append(
                            {
                                **_prefix(protocol_sha=protocol_sha, config_sha=config_sha, phase=phase,
                                          status="FAIL" if failed else "PASS"),
                                "geometry": geometry, "subject": subject, "split": split,
                                "mean_kind": mean_kind, "class_label": class_label,
                                "n_samples": 72 if split == "F" and mean_kind == "class" else
                                             288 if split == "F" else 36 if mean_kind == "class" else 144,
                                "karcher_residual": (1e-3 if failed else 1e-10) if geometry == "AIRM" else np.nan,
                                "custom_relative_error": 1e-14 if geometry == "LE" else np.nan,
                                "warning_count": 0, "warning_messages": "[]", "spd_passed": True,
                                "passed": not failed,
                                "failure_reasons": '["KARCHER_RESIDUAL"]' if failed else "[]",
                            }
                        )
    for subject in range(1, 10):
        for split in ("A", "B", "F"):
            for geometry in GEOMETRIES:
                rows["centering_isometry_gate.csv"].append(
                    {**normal, "geometry": geometry, "subject": subject, "split": split,
                     "d_centering_relative_error": 1e-14, "limit": 1e-10, "passed": True}
                )
                rows["orthogonal_gauge_gate.csv"].append(
                    {
                        **normal, "geometry": geometry, "subject": subject, "split": split,
                        "d_relative_error": 1e-14, "g_relative_error": 1e-14,
                        "d_limit": 1e-10, "g_limit": 1e-10,
                        "g_direct_whitened_relative_error": 1e-14 if geometry == "AIRM" else np.nan,
                        "g_direct_whitened_limit": 1e-10,
                        "permutation_d_relative_error": 1e-14,
                        "permutation_g_relative_error": 1e-14, "permutation_limit": 1e-10,
                        "le_d_g_identity_relative_error": 1e-14 if geometry == "LE" else np.nan,
                        "le_d_g_identity_limit": 1e-10,
                        "d_symmetry_relative_error": 0.0, "d_diagonal_max_abs": 0.0,
                        "d_minimum": 0.0, "g_symmetry_relative_error": 0.0,
                        "structure_tolerance": 1e-12, "passed": True,
                    }
                )
                geometry_index = GEOMETRIES.index(geometry)
                subject_index = subject - 1
                split_index = ("A", "B", "F").index(split)
                d_matrix = object_values["D_matrices.npz"]["matrices"][
                    geometry_index, subject_index, split_index
                ]
                g_matrix = object_values["G_matrices.npz"]["matrices"][
                    geometry_index, subject_index, split_index
                ]
                d_raw = np.asarray(
                    [
                        d_matrix[left, right]
                        for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
                    ]
                )
                g_raw = np.asarray(svec(g_matrix))
                d_norm = float(np.linalg.norm(d_raw))
                g_norm = float(np.linalg.norm(g_raw))
                for object_name in OBJECTS:
                    failed = degenerate_failure and subject == 1 and split == "A" and geometry == "AIRM" and object_name == "D"
                    shape_norm = d_norm if object_name == "D" else g_norm
                    rows["degenerate_geometry_audit.csv"].append(
                        {
                            **_prefix(protocol_sha=protocol_sha, config_sha=config_sha, phase=phase,
                                      status="FAIL" if failed else "PASS"),
                            "geometry": geometry, "object": object_name, "subject": subject, "split": split,
                            "shape_norm": 0.0 if failed else shape_norm,
                            "degeneracy_threshold": (
                                100 * np.finfo(float).eps
                                * max(1.0, float(np.max(np.abs(d_raw if object_name == "D" else g_raw))))
                            ),
                            "is_degenerate": failed, "passed": not failed,
                        }
                    )
                rows["D_shape_vectors.csv"].append(
                    {**normal, "subject": subject, "split": split, "geometry": geometry,
                     "shape_norm": d_norm, **{f"raw_{i}": value for i, value in enumerate(d_raw)},
                     **{f"z_{i}": value for i, value in enumerate(d_raw / d_norm)}}
                )
                rows["G_shape_vectors.csv"].append(
                    {**normal, "subject": subject, "split": split, "geometry": geometry,
                     "shape_norm": g_norm, **{f"raw_{i}": value for i, value in enumerate(g_raw)},
                     **{f"z_{i}": value for i, value in enumerate(g_raw / g_norm)}}
                )
                radii = np.sqrt(np.maximum(np.diag(g_matrix), 0.0))
                rows["absolute_geometry_scales.csv"].append(
                    {**normal, "subject": subject, "split": split, "geometry": geometry,
                     "D_shape_norm": d_norm, "G_shape_norm": g_norm,
                     "D_upper_mean": float(np.mean(d_raw)),
                     "D_upper_max": float(np.max(d_raw)),
                     "class_radius_mean": float(np.mean(radii)),
                     "class_radius_max": float(np.max(radii))}
                )
                for left in range(4):
                    for right in range(left + 1, 4):
                        cosine = float(g_matrix[left, right] / (radii[left] * radii[right]))
                        rows["radius_angle_summary.csv"].append(
                            {**normal, "subject": subject, "split": split, "geometry": geometry,
                             "class_left": CLASSES[left], "class_right": CLASSES[right],
                             "radius_left": radii[left], "radius_right": radii[right],
                             "cosine": cosine,
                             "angle_radians": float(np.arccos(np.clip(cosine, -1.0, 1.0)))}
                        )

    downstream_unassessed = failure is not None and phase == "confirmatory"
    downstream_status = "UNASSESSED" if downstream_unassessed else "PASS"
    downstream = _prefix(protocol_sha=protocol_sha, config_sha=config_sha, phase=phase,
                         status=downstream_status)
    group_lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    table_for_stage = {"R": "label_destruction_group_summary.csv",
                       "S": "semantic_permutation_null_summary.csv",
                       "P": "oracle_permutation_group_summary.csv"}
    for stage in ("R", "S", "P"):
        values = null_values[stage]
        for geometry_index, geometry in enumerate(GEOMETRIES):
            for object_index, object_name in enumerate(OBJECTS):
                if downstream_unassessed:
                    row = {**downstream, "geometry": geometry, "object": object_name, "stage": stage,
                           "observed": np.nan, "null_median": np.nan, "effect": np.nan,
                           "p_value": np.nan, "exceedances": 0, "replicates": 0, "gate_pass": False}
                else:
                    chain_ok = (geometry, object_name) in passing_chains
                    observed = 1.0 if chain_ok else 0.0
                    null = values[geometry_index, object_index]
                    null_median = float(np.median(null))
                    exceedances = int(np.count_nonzero(null >= observed))
                    row = {**downstream, "geometry": geometry, "object": object_name, "stage": stage,
                           "observed": observed, "null_median": null_median,
                           "effect": observed - null_median,
                           "p_value": (1 + exceedances) / (len(null) + 1),
                           "exceedances": exceedances, "replicates": len(null), "gate_pass": True}
                rows[table_for_stage[stage]].append(row)
                group_lookup[(geometry, object_name, stage)] = row

    for geometry in GEOMETRIES:
        for object_name in OBJECTS:
            r_group = group_lookup[(geometry, object_name, "R")]
            s_group = group_lookup[(geometry, object_name, "S")]
            for subject in range(1, 10):
                observed = np.nan if downstream_unassessed else float(r_group["observed"])
                rows["within_subject_reliability.csv"].append(
                    {**downstream, "geometry": geometry, "object": object_name, "stage": "R",
                     "subject": subject, "observed": observed, "group_observed": r_group["observed"]}
                )
                rows["label_destruction_subject_summary.csv"].append(
                    {**downstream, "geometry": geometry, "object": object_name, "stage": "R",
                     "subject": subject, "observed": observed,
                     "null_median": np.nan if downstream_unassessed else 0.2,
                     "effect": np.nan if downstream_unassessed else observed - 0.2,
                     "null_percentile": np.nan if downstream_unassessed else 1.0,
                     "replicates": 0 if downstream_unassessed else 39}
                )
                shared_observed = np.nan if downstream_unassessed else float(s_group["observed"])
                rows["cross_subject_shared_geometry.csv"].append(
                    {**downstream, "geometry": geometry, "object": object_name, "stage": "S",
                     "subject": subject, "observed": shared_observed,
                     "group_observed": s_group["observed"],
                     "null_median": np.nan if downstream_unassessed else 0.2,
                     "effect": np.nan if downstream_unassessed else shared_observed - 0.2,
                     "null_percentile": np.nan if downstream_unassessed else 1.0,
                     "replicates": 0 if downstream_unassessed else 39}
                )
                dimension = 6 if object_name == "D" else 10
                split_pairs = (
                    (("A", "B"), ("B", "A"), ("F", "F"))
                    if phase == "discovery"
                    else (("F", "A"), ("F", "B"), ("F", "F"))
                )
                for source_split, target_split in split_pairs:
                    for feature_index in range(dimension):
                        rows["loso_templates.csv"].append(
                            {
                                **downstream,
                                "geometry": geometry,
                                "object": object_name,
                                "target_subject": subject,
                                "template_source_split": source_split,
                                "target_split": target_split,
                                "feature_index": feature_index,
                                "value": 1.0 / np.sqrt(dimension),
                            }
                        )
                for permutation_index in range(24):
                    rows["oracle_permutation_all_24_scores.csv"].append(
                        {**downstream, "geometry": geometry, "object": object_name, "stage": "P",
                         "subject": subject, "permutation_index": permutation_index,
                         "permutation": f"{permutation_index:02d}", "is_identity": permutation_index == 0,
                         "score": np.nan if downstream_unassessed else (
                             (0.9 if (geometry, object_name) in passing_chains else 0.1)
                             if permutation_index == 0
                             else 0.5 - permutation_index * 0.001
                         )}
                    )
                chain_ok = (geometry, object_name) in passing_chains
                rows["oracle_permutation_subject_summary.csv"].append(
                    {**downstream, "geometry": geometry, "object": object_name, "stage": "P",
                     "subject": subject, "identity_score": np.nan if downstream_unassessed else (0.9 if chain_ok else 0.1),
                     "true_rank": 1 if chain_ok else 24,
                     "normalized_rank": 1.0 if chain_ok else 0.0,
                     "top1_exact": chain_ok,
                     "best_permutation_index": 0 if chain_ok else 1,
                     "best_permutation": "00" if chain_ok else "01",
                     "second_best_permutation_index": 1 if chain_ok else 2,
                     "second_best_permutation": "01" if chain_ok else "02",
                     "margin": np.nan if downstream_unassessed else (0.401 if chain_ok else -0.399),
                     "null_median": np.nan if downstream_unassessed else 0.5,
                     "effect": np.nan if downstream_unassessed else (0.5 if chain_ok else -0.5),
                     "null_percentile": np.nan if downstream_unassessed else (1.0 if chain_ok else 0.0)}
                )
            rows["unrelated_subject_derangement_summary.csv"].append(
                {**downstream, "geometry": geometry, "object": object_name,
                 "same_subject_median": r_group["observed"],
                 "unrelated_median": np.nan if downstream_unassessed else 0.1,
                 "unrelated_min": np.nan if downstream_unassessed else -0.5,
                 "unrelated_max": np.nan if downstream_unassessed else 0.6,
                 "derangement_count": 133496}
            )
            for stage in ("R", "S", "P"):
                if stage == "R":
                    scores = np.full(9, float(r_group["observed"]), dtype=np.float64)
                    subject_effects = scores - 0.2
                elif stage == "S":
                    scores = np.full(9, float(s_group["observed"]), dtype=np.float64)
                    subject_effects = scores - 0.2
                else:
                    scores = np.full(
                        9,
                        1.0 if (geometry, object_name) in passing_chains else 0.0,
                        dtype=np.float64,
                    )
                    subject_effects = scores - 0.5
                if downstream_unassessed:
                    ordinary_values = (np.nan, np.nan, np.nan)
                    discovery_confirmatory = (np.nan, np.nan, np.nan)
                    airm_minus_le = (np.nan, np.nan, np.nan)
                else:
                    ordinary = subject_bootstrap_median(
                        scores,
                        replicates=39,
                        master_seed=20260809,
                        phase_tag=0 if phase == "discovery" else 1,
                    )
                    ordinary_values = (
                        float(np.median(scores)), ordinary.ci_low, ordinary.ci_high
                    )
                    if phase == "confirmatory":
                        paired = subject_bootstrap_paired_median_delta(
                            subject_effects,
                            subject_effects,
                            replicates=39,
                            master_seed=20260809,
                            phase_tag=2,
                        )
                        discovery_confirmatory = (0.0, paired.ci_low, paired.ci_high)
                    else:
                        discovery_confirmatory = (np.nan, np.nan, np.nan)
                    if geometry == "AIRM":
                        paired = subject_bootstrap_paired_median_delta(
                            subject_effects,
                            subject_effects,
                            replicates=39,
                            master_seed=20260809,
                            phase_tag=2,
                        )
                        airm_minus_le = (0.0, paired.ci_low, paired.ci_high)
                    else:
                        airm_minus_le = (np.nan, np.nan, np.nan)
                rows["subject_bootstrap_summary.csv"].append(
                    {**downstream, "geometry": geometry, "object": object_name, "stage": stage,
                     "observed_median": ordinary_values[0],
                     "ci_low": ordinary_values[1], "ci_high": ordinary_values[2],
                     "replicates": 0 if downstream_unassessed else 39,
                     "discovery_confirmatory_effect_delta_median": discovery_confirmatory[0],
                     "discovery_confirmatory_effect_delta_ci_low": discovery_confirmatory[1],
                     "discovery_confirmatory_effect_delta_ci_high": discovery_confirmatory[2],
                     "airm_minus_le_effect_delta_median": airm_minus_le[0],
                     "airm_minus_le_effect_delta_ci_low": airm_minus_le[1],
                     "airm_minus_le_effect_delta_ci_high": airm_minus_le[2]}
                )
                expected_influence = (
                    np.full(9, np.nan)
                    if downstream_unassessed
                    else leave_one_subject_out_influence(scores)
                )
                for subject_index, subject in enumerate(range(1, 10)):
                    rows["leave_one_subject_out_influence.csv"].append(
                         {**downstream, "geometry": geometry, "object": object_name, "stage": stage,
                         "subject": subject, "influence": expected_influence[subject_index],
                         "subject_score": np.nan if downstream_unassessed else scores[subject_index],
                         "subject_effect": (
                             np.nan if downstream_unassessed else subject_effects[subject_index]
                         ),
                         "discovery_subject_effect": (
                             np.nan if downstream_unassessed else subject_effects[subject_index]
                         ),
                         "confirmatory_subject_effect": (
                             np.nan if downstream_unassessed or phase == "discovery"
                             else subject_effects[subject_index]
                         ),
                         "discovery_confirmatory_effect_delta": (
                             np.nan if downstream_unassessed or phase == "discovery" else 0.0
                         ),
                         "airm_minus_le_subject_effect_delta": (
                             np.nan if downstream_unassessed or geometry == "LE" else 0.0
                         )}
                    )
    return {filename: _ordered_table(filename, values) for filename, values in rows.items()}


def _make_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    passing_chains: set[tuple[str, str]] | None = None,
    failure: str | None = None,
) -> tuple[Path, Path, Path]:
    if passing_chains is None:
        passing_chains = {(geometry, object_name) for geometry in GEOMETRIES for object_name in OBJECTS}
    repo = tmp_path / "repo"
    repo.mkdir()
    protocol_source = ROOT / "docs" / "PROTOCOL_CONDITIONAL_GEOMETRY_V1.md"
    protocol_path = repo / "docs" / protocol_source.name
    protocol_path.parent.mkdir(parents=True)
    shutil.copyfile(protocol_source, protocol_path)
    protocol_sha = _sha(protocol_path)

    config = yaml.safe_load(
        (ROOT / "configs" / "bnci2014_001_conditional_geometry_v1.yaml").read_text(encoding="utf-8")
    )
    config["protocol"]["protocol_path"] = f"docs/{protocol_path.name}"
    config["protocol"]["protocol_sha256"] = protocol_sha
    config["project"]["output_dir"] = "outputs/bnci2014_001_conditional_geometry_v1"
    config["expected_data"]["trials_per_session"] = 18
    for family in ("label_destruction", "semantic_permutation", "oracle_rank"):
        config["nulls"][family]["replicates"] = 39
    config["statistics"]["bootstrap"]["replicates"] = 39
    config_path = repo / "configs" / "bnci2014_001_conditional_geometry_v1.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_sha = _sha(config_path)
    output = repo / config["project"]["output_dir"]
    for directory in ("protocol", "tables", "objects", "nulls", "figures", "report"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "protocol" / "frozen_config.yaml")
    shutil.copyfile(protocol_path, output / "protocol" / protocol_path.name)

    monkeypatch.setattr(
        "src.reporting_conditional_v1.EXPECTED_NULL_REPLICATES",
        {name: 39 for name in REQUIRED_NULLS},
    )
    null_by_phase: dict[str, dict[str, np.ndarray]] = {}
    phase_tables: dict[str, dict[str, pd.DataFrame]] = {}
    for phase in ("discovery", "confirmatory"):
        phase_failed = failure is not None and phase == "confirmatory"
        nulls = {
            stage: np.empty((2, 2, 0 if phase_failed else 39), dtype=np.float64)
            for stage in ("R", "S", "P")
        }
        if not phase_failed:
            for stage in ("R", "S", "P"):
                for geometry_index, geometry in enumerate(GEOMETRIES):
                    for object_index, object_name in enumerate(OBJECTS):
                        if (geometry, object_name) in passing_chains:
                            nulls[stage][geometry_index, object_index] = np.linspace(0.1, 0.3, 39)
                        else:
                            nulls[stage][geometry_index, object_index] = np.linspace(0.2, 0.4, 39)
        null_by_phase[phase] = nulls
        phase_object_values = _object_values(
            0.002 if phase == "confirmatory" else 0.0
        )
        phase_tables[phase] = _phase_tables(
            phase=phase,
            protocol_sha=protocol_sha,
            config_sha=config_sha,
            null_values=nulls,
            passing_chains=passing_chains,
            failure=failure,
            object_values=phase_object_values,
        )
        table_dir = output / "tables" / phase
        table_dir.mkdir(parents=True)
        for filename, frame in phase_tables[phase].items():
            frame.to_csv(table_dir / filename, index=False)
        object_dir = output / "objects" / phase
        object_dir.mkdir(parents=True)
        for filename, arrays in phase_object_values.items():
            np.savez_compressed(object_dir / filename, **arrays)
        null_dir = output / "nulls" / phase
        null_dir.mkdir(parents=True)
        for filename, stage in zip(REQUIRED_NULLS, ("R", "S", "P"), strict=True):
            values = nulls[stage]
            np.savez_compressed(
                null_dir / filename,
                replicate_indices=np.arange(values.shape[-1], dtype=np.int64),
                geometries=np.asarray(GEOMETRIES),
                objects=np.asarray(OBJECTS),
                group_statistics=values,
            )

    for filename in REQUIRED_TABLES:
        if filename in POST_CONFIRMATORY_TABLES:
            continue
        pd.concat(
            [phase_tables["discovery"][filename], phase_tables["confirmatory"][filename]],
            ignore_index=True,
        ).to_csv(output / "tables" / filename, index=False)
    for filename in REQUIRED_OBJECTS:
        shutil.copyfile(output / "objects" / "confirmatory" / filename, output / "objects" / filename)
    for filename in REQUIRED_NULLS:
        shutil.copyfile(output / "nulls" / "confirmatory" / filename, output / "nulls" / filename)

    operands = []
    for filename, stage in GROUP_TABLE_STAGE.items():
        for geometry in GEOMETRIES:
            for object_name in OBJECTS:
                discovery = phase_tables["discovery"][filename]
                confirmatory = phase_tables["confirmatory"][filename]
                left = discovery[(discovery["geometry"] == geometry) & (discovery["object"] == object_name)].iloc[0]
                right = confirmatory[(confirmatory["geometry"] == geometry) & (confirmatory["object"] == object_name)].iloc[0]
                operands.append(
                    {"geometry": geometry, "object": object_name, "stage": stage,
                     "discovery_effect": left["effect"], "confirmatory_effect": right["effect"],
                     "confirmatory_p": right["p_value"]}
                )
    verdicts = compute_frozen_verdicts(
        pd.DataFrame(operands),
        hard_gates_pass=failure is None,
        failure_class=failure,
        failure_details=(f"synthetic {failure}",) if failure else (),
    )
    combined = _prefix(protocol_sha=protocol_sha, config_sha=config_sha, phase="combined",
                       status="UNASSESSED" if failure else "PASS")
    comparison_rows = []
    for row in operands:
        key = (row["geometry"], row["object"], row["stage"])
        table_name = {value: name for name, value in GROUP_TABLE_STAGE.items()}[row["stage"]]
        left = phase_tables["discovery"][table_name]
        right = phase_tables["confirmatory"][table_name]
        left_row = left[(left["geometry"] == key[0]) & (left["object"] == key[1])].iloc[0]
        right_row = right[(right["geometry"] == key[0]) & (right["object"] == key[1])].iloc[0]
        comparison_rows.append(
            {**combined, "geometry": key[0], "object": key[1], "stage": key[2],
             "discovery_observed": left_row["observed"], "discovery_effect": left_row["effect"],
             "confirmatory_observed": right_row["observed"], "confirmatory_effect": right_row["effect"],
             "confirmatory_p": right_row["p_value"]}
        )
    _ordered_table("discovery_confirmatory_comparison.csv", comparison_rows).to_csv(
        output / "tables" / "discovery_confirmatory_comparison.csv", index=False
    )
    robustness_rows = []
    for object_name in OBJECTS:
        for stage in ("R", "S", "P"):
            airm = next(row for row in operands if row["geometry"] == "AIRM" and row["object"] == object_name and row["stage"] == stage)
            le = next(row for row in operands if row["geometry"] == "LE" and row["object"] == object_name and row["stage"] == stage)
            robustness_rows.append(
                {**combined, "object": object_name, "stage": stage,
                 "airm_discovery_effect": airm["discovery_effect"],
                 "airm_confirmatory_effect": airm["confirmatory_effect"],
                 "le_discovery_effect": le["discovery_effect"],
                 "le_confirmatory_effect": le["confirmatory_effect"]}
            )
    _ordered_table("airm_le_robustness.csv", robustness_rows).to_csv(
        output / "tables" / "airm_le_robustness.csv", index=False
    )
    hypothesis_rows = []
    for row in verdicts.stage_status.itertuples(index=False):
        hypothesis_rows.append(
            {**combined, "geometry": row.geometry, "object": row.object, "stage": row.stage,
             "eligible": row.eligible, "criterion_pass": row.criterion_pass,
             "stage_status": row.status,
             "chain_pass": verdicts.chain_pass[(row.geometry, row.object)]}
        )
    _ordered_table("hypothesis_chain_status.csv", hypothesis_rows).to_csv(
        output / "tables" / "hypothesis_chain_status.csv", index=False
    )

    common_json = {
        "protocol_name": config["protocol"]["name"],
        "protocol_version": "1.0",
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
    }
    _write_json(
        output / "manifest.json",
        {
            **common_json, "schema_version": "1.0", "phase": "PROTOCOL_FROZEN",
            "branch": config["protocol"]["branch"], "base_commit": config["protocol"]["base_commit"],
            "confirmatory_designation": "STRICT_CONFIRMATORY",
            "frozen_protocol": (output / "protocol" / protocol_path.name).relative_to(repo).as_posix(),
            "frozen_config": (output / "protocol" / "frozen_config.yaml").relative_to(repo).as_posix(),
        },
    )
    _write_json(output / "git_provenance.json", {**common_json, "head": CODE_COMMIT, "branch": config["protocol"]["branch"]})
    _write_json(output / "environment.json", {**common_json, "python": {"version": "synthetic"}})
    _write_json(
        output / "protocol" / "label_null_dry_run.json",
        {**common_json, "code_commit": CODE_COMMIT, "phase": "discovery", "session": "0train",
         "status": "DRY_RUN_ONLY", "replicates": 1, "workers": 1, "elapsed_seconds": 0.01,
         "seconds_per_replicate_wall": 0.01, "estimated_official_wall_seconds_linear": 19.99,
         "scientific_output_written": False,
         "airm_scalar_crosscheck": {
             "replicate_index": 0,
             "canonical_flat_group_indices": list(range(72)),
             "groups_checked": 72,
             "relative_errors": [1.0e-12] * 72,
             "maximum_relative_error": 1.0e-12,
             "tolerance": 1.0e-10,
             "batched_equivalent_within_tolerance": True,
             "official_scalar_authoritative": True,
             "authoritative_solver": "pyriemann.geometry.mean.mean_riemann",
             "official_all_72_pass": True,
             "official_warning_count": 0,
             "passed": True,
             "maximum_official_karcher_residual": 1.0e-10,
             "maximum_batched_post_karcher_residual": 1.0e-10,
         }},
    )
    decision = {
        **common_json, "code_commit": CODE_COMMIT, "phase": "combined", "session": "0train+1test",
        "status": "UNASSESSED" if failure else "PASS",
        "terminal_decision": verdicts.terminal_decision,
        "le_robustness_label": verdicts.le_robustness_label,
    }
    _write_json(output / "confirmatory_decision.json", decision)

    # Dummy source files let the unlock snapshot prove that reporting code was
    # frozen before confirmatory access without coupling this fixture to Git.
    source_file = repo / "src" / "reporting_conditional_v1.py"
    script_file = repo / "scripts" / "27_conditional_geometry_report.py"
    source_file.parent.mkdir(parents=True)
    script_file.parent.mkdir(parents=True)
    source_file.write_text("# frozen synthetic reporting source\n", encoding="utf-8")
    script_file.write_text("# frozen synthetic reporting script\n", encoding="utf-8")
    discovery_paths = [
        *(output / "tables" / "discovery" / name for name in REQUIRED_TABLES if name not in POST_CONFIRMATORY_TABLES),
        *(output / "objects" / "discovery" / name for name in REQUIRED_OBJECTS),
        *(output / "nulls" / "discovery" / name for name in REQUIRED_NULLS),
    ]
    unlock = {
        "schema_version": "1.0", "status": "CONFIRMATORY_UNLOCKED",
        "created_at_utc": "2026-08-09T00:00:00+00:00", "confirmatory_designation": "STRICT_CONFIRMATORY",
        "prior_non_anatomy_session1_access": True,
        "prior_session1_conditional_object_analysis": False,
        "discovery_session": "0train", "confirmatory_session": "1test",
        "protocol_sha256": protocol_sha, "config_sha256": config_sha,
        "code_commit": CODE_COMMIT, "locked_head": CODE_COMMIT,
        "branch": config["protocol"]["branch"], "working_tree": {"clean": True, "porcelain": ""},
        "code_snapshot": _records(repo, [config_path, protocol_path, source_file, script_file]),
        "discovery_snapshot": _records(repo, discovery_paths),
        "confirmatory_raw_ordered_manifest_sha256": config["confirmatory_inputs"]["ordered_manifest_sha256"],
    }
    unlock["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(unlock)).hexdigest()
    _write_json(output / "confirmatory_unlock.json", unlock)
    return repo, config_path, output


def _convert_to_minimal_failure(
    repo: Path,
    config_path: Path,
    output: Path,
    *,
    failure_class: str,
    phase: str,
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol_path = repo / config["protocol"]["protocol_path"]
    protocol_sha = _sha(protocol_path)
    config_sha = _sha(config_path)
    terminal = {
        "data": "UNASSESSED_DATA_CONTRACT_FAILURE",
        "numerical": "UNASSESSED_NUMERICAL_FAILURE",
        "degenerate": "UNASSESSED_DEGENERATE_GEOMETRY",
    }[failure_class]
    session = "0train" if phase == "discovery" else "1test"
    manifest = {
        "protocol_version": "1.0",
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "code_commit": CODE_COMMIT,
        "phase": phase,
        "session": session,
        "status": "UNASSESSED",
        "schema_version": "conditional-unassessed-v1",
        "failure_class": failure_class,
        "terminal_decision": terminal,
        "reason_code": "SYNTHETIC_HARD_GATE_FAILURE",
        "reason": f"synthetic {failure_class} failure",
        "scientific_nulls_executed": False,
        "downstream_phase_permitted": False,
        "le_robustness_label": "UNASSESSED",
    }
    manifest_path = output / "protocol" / f"{phase}_failure_manifest.json"
    _write_json(manifest_path, manifest)
    decision = {
        **manifest,
        "failure_manifest": manifest_path.relative_to(repo).as_posix(),
        "failure_manifest_sha256": _sha(manifest_path),
        "chains": {
            geometry: {object_name: None for object_name in OBJECTS}
            for geometry in GEOMETRIES
        },
    }
    _write_json(output / "confirmatory_decision.json", decision)

    for filename in REQUIRED_TABLES:
        (output / "tables" / filename).unlink(missing_ok=True)
    for filename in REQUIRED_OBJECTS:
        (output / "objects" / filename).unlink(missing_ok=True)
    for filename in REQUIRED_NULLS:
        (output / "nulls" / filename).unlink(missing_ok=True)
    phase_roots = (
        output / "tables" / phase,
        output / "objects" / phase,
        output / "nulls" / phase,
    )
    for directory in phase_roots:
        if directory.exists():
            shutil.rmtree(directory)
    if phase == "discovery":
        (output / "confirmatory_unlock.json").unlink(missing_ok=True)
        for scope in ("tables", "objects", "nulls"):
            directory = output / scope / "confirmatory"
            if directory.exists():
                shutil.rmtree(directory)


def test_report_literals_are_exactly_frozen() -> None:
    assert REPORT_TITLE == "# BNCI2014_001 Conditional Geometry Anatomy v1"
    assert len(REPORT_HEADINGS) == 18
    assert len(FIGURE_STEMS) == 10
    assert len(set(FIGURE_STEMS)) == 10


def test_all_pass_yields_GO_STRONG_and_consistent_LE() -> None:
    result = compute_frozen_verdicts(_stage_operands(), hard_gates_pass=True)
    assert result.terminal_decision == "GO_STRONG"
    assert result.le_robustness_label == "AIRM+LE CONSISTENT"
    assert all(result.chain_pass.values())
    assert set(result.stage_status["status"]) == {"PASS"}
    assert result.next_question.count("?") == 1


def test_stop_path_preserves_fixed_sequence_descriptive_only() -> None:
    # D fails; G passes.  AIRM therefore takes the exact tangent-only STOP.
    passing = {("AIRM", "G"), ("LE", "G")}
    operands = _stage_operands(passing=passing)
    result = compute_frozen_verdicts(operands, hard_gates_pass=True)
    assert result.terminal_decision == "STOP_TANGENT_ONLY"
    assert result.le_robustness_label == "AIRM+LE CONSISTENT"
    failed_chain = result.stage_status[
        (result.stage_status["geometry"] == "AIRM")
        & (result.stage_status["object"] == "D")
    ].set_index("stage")
    assert failed_chain.loc["R", "status"] == "FAIL"
    assert failed_chain.loc["S", "status"] == "DESCRIPTIVE_ONLY"
    assert failed_chain.loc["P", "status"] == "DESCRIPTIVE_ONLY"


@pytest.mark.parametrize(
    ("failure_class", "expected"),
    [
        ("numerical", "UNASSESSED_NUMERICAL_FAILURE"),
        ("data", "UNASSESSED_DATA_CONTRACT_FAILURE"),
        ("degenerate", "UNASSESSED_DEGENERATE_GEOMETRY"),
    ],
)
def test_global_failure_never_uses_available_case_verdict(
    failure_class: str, expected: str
) -> None:
    result = compute_frozen_verdicts(
        _stage_operands(),
        hard_gates_pass=False,
        failure_class=failure_class,
        failure_details=("synthetic failed row",),
    )
    assert result.terminal_decision == expected
    assert result.le_robustness_label == "UNASSESSED"
    assert not any(result.chain_pass.values())
    assert set(result.stage_status["status"]) == {"UNASSESSED"}
    assert result.failure_details == ("synthetic failed row",)


def test_stage_grid_is_exact_and_finite() -> None:
    duplicate = pd.concat([_stage_operands(), _stage_operands().iloc[[0]]])
    with pytest.raises(ReportingContractError, match="duplicate"):
        compute_frozen_verdicts(duplicate, hard_gates_pass=True)
    missing = _stage_operands().iloc[:-1]
    with pytest.raises(ReportingContractError, match="grid mismatch"):
        compute_frozen_verdicts(missing, hard_gates_pass=True)
    nonfinite = _stage_operands()
    nonfinite.loc[0, "confirmatory_p"] = np.nan
    with pytest.raises(ReportingContractError, match="non-finite"):
        compute_frozen_verdicts(nonfinite, hard_gates_pass=True)


def test_full_producer_shaped_fixture_creates_exact_report_and_30_figure_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    inputs = load_and_validate_reporting_inputs(config_path, repo)
    artifacts = create_reporting_outputs(inputs)
    assert artifacts.verdicts.terminal_decision == "GO_STRONG"
    assert artifacts.verdicts.le_robustness_label == "AIRM+LE CONSISTENT"
    assert len(artifacts.figure_source_paths) == 10
    assert len(artifacts.figure_paths) == 20
    assert artifacts.report_path.is_file()
    assert tuple(
        line[3:] for line in artifacts.report_text.splitlines() if line.startswith("## ")
    ) == REPORT_HEADINGS
    generated = {
        path.name for path in (output / "figures").iterdir() if path.is_file()
    }
    assert generated == {
        *(f"{stem}.csv" for stem in FIGURE_STEMS),
        *(f"{stem}.png" for stem in FIGURE_STEMS),
        *(f"{stem}.pdf" for stem in FIGURE_STEMS),
    }


def test_full_fixture_stop_decision_is_not_positive_spin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passing = {("AIRM", "G"), ("LE", "G")}
    repo, config_path, _ = _make_artifacts(
        tmp_path, monkeypatch, passing_chains=passing
    )
    inputs = load_and_validate_reporting_inputs(config_path, repo)
    operands = pd.DataFrame(
        [
            {
                "geometry": geometry,
                "object": object_name,
                "stage": stage,
                "discovery_effect": inputs.snapshot_tables["discovery"][filename].query(
                    "geometry == @geometry and object == @object_name"
                ).iloc[0]["effect"],
                "confirmatory_effect": inputs.snapshot_tables["confirmatory"][filename].query(
                    "geometry == @geometry and object == @object_name"
                ).iloc[0]["effect"],
                "confirmatory_p": inputs.snapshot_tables["confirmatory"][filename].query(
                    "geometry == @geometry and object == @object_name"
                ).iloc[0]["p_value"],
            }
            for filename, stage in GROUP_TABLE_STAGE.items()
            for geometry in GEOMETRIES
            for object_name in OBJECTS
        ]
    )
    verdicts = compute_frozen_verdicts(operands, hard_gates_pass=True)
    text = render_report(inputs, verdicts)
    assert verdicts.terminal_decision == "STOP_TANGENT_ONLY"
    assert "The frozen terminal rule is STOP_TANGENT_ONLY" in text
    assert "promising" not in text.lower()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("data", "UNASSESSED_DATA_CONTRACT_FAILURE"),
        ("numerical", "UNASSESSED_NUMERICAL_FAILURE"),
        ("degenerate", "UNASSESSED_DEGENERATE_GEOMETRY"),
    ],
)
def test_full_fixture_failure_paths_remain_unassessed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected: str,
) -> None:
    repo, config_path, _ = _make_artifacts(
        tmp_path, monkeypatch, failure=failure
    )
    inputs = load_and_validate_reporting_inputs(config_path, repo)
    assert inputs.failure_class == failure
    operands = _stage_operands()
    operands[["confirmatory_effect", "confirmatory_p"]] = np.nan
    verdicts = compute_frozen_verdicts(
        operands,
        hard_gates_pass=False,
        failure_class=inputs.failure_class,
        failure_details=inputs.failure_details,
    )
    text = render_report(inputs, verdicts)
    assert verdicts.terminal_decision == expected
    assert "No scientific R/S/P conclusion is justified" in text


def test_missing_required_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    (output / "tables" / "oracle_permutation_group_summary.csv").unlink()
    with pytest.raises(ReportingContractError, match="missing required combined tables"):
        load_and_validate_reporting_inputs(config_path, repo)


def test_tampered_locked_discovery_artifact_and_unlock_hash_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    target = output / "tables" / "discovery" / "within_subject_reliability.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ReportingContractError, match="SHA-256 mismatch"):
        load_and_validate_reporting_inputs(config_path, repo)


def test_tampered_null_npz_is_rejected_even_if_csv_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    path = output / "nulls" / "confirmatory" / "label_destruction_group_statistics.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    payload["group_statistics"][0, 0, 0] += 0.2
    np.savez_compressed(path, **payload)
    with pytest.raises(ReportingContractError, match="disagrees with saved null NPZ|root null"):
        load_and_validate_reporting_inputs(config_path, repo)


def test_dry_run_must_record_official_all_72_pyriemann_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    path = output / "protocol" / "label_null_dry_run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["airm_scalar_crosscheck"]["official_all_72_pass"] = False
    _write_json(path, payload)
    with pytest.raises(ReportingContractError, match="authority field"):
        load_and_validate_reporting_inputs(config_path, repo)


def test_confirmatory_templates_must_equal_locked_discovery_f_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    confirmatory_path = output / "tables" / "confirmatory" / "loso_templates.csv"
    root_path = output / "tables" / "loso_templates.csv"
    confirmatory = pd.read_csv(confirmatory_path)
    selected = (
        (confirmatory["geometry"] == "AIRM")
        & (confirmatory["object"] == "D")
        & (confirmatory["target_subject"] == 1)
        & (confirmatory["template_source_split"] == "F")
        & (confirmatory["target_split"] == "A")
    )
    confirmatory.loc[selected, "value"] *= -1.0
    confirmatory.to_csv(confirmatory_path, index=False)
    combined = pd.read_csv(root_path)
    root_selected = (
        (combined["phase"] == "confirmatory")
        & (combined["geometry"] == "AIRM")
        & (combined["object"] == "D")
        & (combined["target_subject"] == 1)
        & (combined["template_source_split"] == "F")
        & (combined["target_split"] == "A")
    )
    combined.loc[root_selected, "value"] *= -1.0
    combined.to_csv(root_path, index=False)
    with pytest.raises(ReportingContractError, match="locked discovery-F templates"):
        load_and_validate_reporting_inputs(config_path, repo)


@pytest.mark.parametrize(
    ("failure_class", "phase", "terminal"),
    [
        ("data", "discovery", "UNASSESSED_DATA_CONTRACT_FAILURE"),
        ("numerical", "confirmatory", "UNASSESSED_NUMERICAL_FAILURE"),
        ("degenerate", "confirmatory", "UNASSESSED_DEGENERATE_GEOMETRY"),
    ],
)
def test_minimal_failure_contract_creates_only_unassessed_figure_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_class: str,
    phase: str,
    terminal: str,
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    _convert_to_minimal_failure(
        repo,
        config_path,
        output,
        failure_class=failure_class,
        phase=phase,
    )
    inputs = load_and_validate_reporting_inputs(config_path, repo)
    assert inputs.failure_class == failure_class
    assert inputs.failure_manifest is not None
    assert inputs.tables == {}
    artifacts = create_reporting_outputs(inputs)
    assert artifacts.verdicts.terminal_decision == terminal
    assert len(artifacts.figure_source_paths) == 10
    assert len(artifacts.figure_paths) == 20
    assert artifacts.report_text.count("\n## ") == 18
    assert "No scientific estimate, null result" in artifacts.report_text
    assert "scientific_nulls_executed=false" in artifacts.report_text
    for path in artifacts.figure_source_paths:
        source = pd.read_csv(path)
        assert len(source) == 1
        assert source.loc[0, "status"] == "UNASSESSED"
        assert not bool(source.loc[0, "scientific_data_plotted"])


def test_tampered_subject_effect_delta_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    snapshot_path = output / "tables" / "confirmatory" / "leave_one_subject_out_influence.csv"
    root_path = output / "tables" / "leave_one_subject_out_influence.csv"
    snapshot = pd.read_csv(snapshot_path)
    selected = (
        (snapshot["geometry"] == "AIRM")
        & (snapshot["object"] == "D")
        & (snapshot["stage"] == "R")
        & (snapshot["subject"] == 1)
    )
    snapshot.loc[selected, "discovery_confirmatory_effect_delta"] += 0.1
    snapshot.to_csv(snapshot_path, index=False)
    combined = pd.read_csv(root_path)
    selected = (
        (combined["phase"] == "confirmatory")
        & (combined["geometry"] == "AIRM")
        & (combined["object"] == "D")
        & (combined["stage"] == "R")
        & (combined["subject"] == 1)
    )
    combined.loc[selected, "discovery_confirmatory_effect_delta"] += 0.1
    combined.to_csv(root_path, index=False)
    with pytest.raises(ReportingContractError, match="effect delta"):
        load_and_validate_reporting_inputs(config_path, repo)


def test_tampered_robustness_operand_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    path = output / "tables" / "airm_le_robustness.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "airm_confirmatory_effect"] += 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(ReportingContractError, match="robustness"):
        load_and_validate_reporting_inputs(config_path, repo)


def test_finite_confirmatory_D_tamper_is_rejected_against_saved_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, config_path, output = _make_artifacts(tmp_path, monkeypatch)
    snapshot_path = output / "objects" / "confirmatory" / "D_matrices.npz"
    root_path = output / "objects" / "D_matrices.npz"
    with np.load(snapshot_path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    payload["matrices"][0, 0, 0, 0, 1] += 0.01
    payload["matrices"][0, 0, 0, 1, 0] += 0.01
    np.savez_compressed(snapshot_path, **payload)
    np.savez_compressed(root_path, **payload)
    with pytest.raises(ReportingContractError, match="mean-to-object"):
        load_and_validate_reporting_inputs(config_path, repo)
