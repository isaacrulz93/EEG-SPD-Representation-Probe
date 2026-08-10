"""Retrospective D/G discrepancy decomposition from immutable v1 objects.

This module never loads EEG, fits a covariance, or recomputes a Frechet mean.
It consumes only the frozen Conditional Geometry Anatomy v1 D/G archives and
their published tables/nulls.  K_exact remains signed; no PSD projection exists.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.conditional_geometry_v1 import D_UPPER_TRIANGLE_INDICES, svec
from src.conditional_nulls_v1 import (
    all_s4_permutations,
    permuted_shape_bank,
    semantic_permutation_indices,
)
from src.conditional_statistics_v1 import (
    confirmatory_oracle_score_sets,
    confirmatory_shared_subject_scores,
    discovery_oracle_score_sets,
    discovery_shared_subject_scores,
    normalize_shape_vectors,
    plus_one_null_summary,
    reliability_subject_scores,
    subject_null_percentiles,
    summarize_oracle_scores,
)


SESSIONS = ("0train", "1test")
GEOMETRIES = ("AIRM", "LE")
OBJECTS = ("D_exact", "D_tan", "K_exact", "G0", "G")
SPLITS = ("A", "B", "F")
CLASSES = ("left_hand", "right_hand", "feet", "tongue")
SUBJECTS = tuple(range(1, 10))
PAIR_LABELS = ("12", "13", "14", "23", "24", "34")
REL_TOL = 1.0e-10
ABS_TOL = 1.0e-12


class DGDecompositionError(RuntimeError):
    """Fail-closed diagnostic error."""


@dataclass(frozen=True)
class DerivedBundle:
    D_exact: np.ndarray
    D_tan: np.ndarray
    K_exact: np.ndarray
    G0: np.ndarray
    G: np.ndarray
    weights: np.ndarray
    H: np.ndarray
    q_cleanup: np.ndarray
    shapes_raw: Mapping[str, np.ndarray]
    shapes_unit: Mapping[str, np.ndarray]
    shape_norms: Mapping[str, np.ndarray]

    def matrix(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self, name), dtype=np.float64)


@dataclass(frozen=True)
class SemanticResults:
    observed_subject: np.ndarray
    null_subject: np.ndarray
    null_group: np.ndarray
    summary: pd.DataFrame
    subject_effects: pd.DataFrame


@dataclass(frozen=True)
class RunResult:
    output_root: Path
    overall_label: str
    session_labels: Mapping[str, str]
    tests: str
    hard_gates: str
    source_unchanged: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_snapshot(root: Path) -> tuple[list[dict[str, Any]], str]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return records, hashlib.sha256(payload).hexdigest()


def relative_error(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    numerator = float(np.linalg.norm(a - b))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    if scale == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / max(scale, np.finfo(np.float64).eps)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise DGDecompositionError("config root must be a mapping")
    return config


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _audit_sources(repo: Path, config: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    contract = config["source_contract"]
    paths: list[tuple[str, str, str]] = []
    for name in ("protocol", "config"):
        entry = contract[name]
        paths.append((f"source_{name}", entry["path"], entry["sha256"]))
    for session in contract["sessions"]:
        for name in ("D", "G"):
            paths.append(
                (
                    f"{session['session']}_{name}",
                    session[f"{name}_path"],
                    session[f"{name}_sha256"],
                )
            )
    for name in ("D", "G"):
        entry = contract["root_replication"]
        paths.append((f"root_{name}", entry[f"{name}_path"], entry[f"{name}_sha256"]))
    for role, relative, expected in paths:
        path = repo / relative
        observed = sha256_file(path) if path.is_file() else "MISSING"
        rows.append(
            {
                "artifact_role": role,
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "bytes": path.stat().st_size if path.is_file() else -1,
                "passed": observed == expected,
            }
        )
    source_root = repo / config["project"]["source_output_dir"]
    records, aggregate = source_tree_snapshot(source_root)
    snap = contract["source_output_snapshot"]
    snapshot_pass = (
        len(records) == int(snap["file_count"])
        and sum(record["bytes"] for record in records) == int(snap["total_bytes"])
        and aggregate == str(snap["aggregate_sha256"])
    )
    rows.append(
        {
            "artifact_role": "complete_source_output_snapshot",
            "path": config["project"]["source_output_dir"],
            "expected_sha256": snap["aggregate_sha256"],
            "observed_sha256": aggregate,
            "bytes": sum(record["bytes"] for record in records),
            "passed": snapshot_pass,
        }
    )
    frame = pd.DataFrame(rows)
    if not bool(frame["passed"].all()):
        failed = frame.loc[~frame["passed"], "artifact_role"].tolist()
        raise DGDecompositionError(f"source artifact integrity failure: {failed}")
    return frame, aggregate


def _load_archive(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {"matrices", "geometries", "subjects", "splits", "classes"}
        if set(archive.files) != expected_keys:
            raise DGDecompositionError(f"unexpected archive schema at {path}")
        if tuple(archive["geometries"].astype(str)) != GEOMETRIES:
            raise DGDecompositionError(f"geometry axis mismatch at {path}")
        if tuple(int(value) for value in archive["subjects"]) != SUBJECTS:
            raise DGDecompositionError(f"subject axis mismatch at {path}")
        if tuple(archive["splits"].astype(str)) != SPLITS:
            raise DGDecompositionError(f"split axis mismatch at {path}")
        if tuple(archive["classes"].astype(str)) != CLASSES:
            raise DGDecompositionError(f"class axis mismatch at {path}")
        matrices = np.asarray(archive["matrices"], dtype=np.float64)
    if matrices.shape != (2, 9, 3, 4, 4) or not np.isfinite(matrices).all():
        raise DGDecompositionError(f"invalid matrices at {path}: {matrices.shape}")
    return matrices


def load_source_matrices(repo: Path, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    d_sessions = []
    g_sessions = []
    for entry in config["source_contract"]["sessions"]:
        d_sessions.append(_load_archive(repo / entry["D_path"]))
        g_sessions.append(_load_archive(repo / entry["G_path"]))
    return np.stack(d_sessions, axis=0), np.stack(g_sessions, axis=0)


def class_weights_from_contract(repo: Path, config: Mapping[str, Any]) -> np.ndarray:
    table = pd.read_csv(
        repo / config["project"]["source_output_dir"] / "tables/dataset_contract.csv"
    )
    rows = table.loc[table["scope"] == "subject_run_class"].copy()
    if len(rows) != 432 or not bool(rows["passed"].all()):
        raise DGDecompositionError("v1 subject_run_class contract is incomplete")
    split_runs = {"A": {0, 1, 2}, "B": {3, 4, 5}, "F": set(range(6))}
    weights = np.empty((2, 9, 3, 4), dtype=np.float64)
    for qi, session in enumerate(SESSIONS):
        for si, subject in enumerate(SUBJECTS):
            for hi, split in enumerate(SPLITS):
                subset = rows[
                    (rows["session"] == session)
                    & (rows["subject"] == subject)
                    & (rows["run"].astype(int).isin(split_runs[split]))
                ]
                counts = []
                for class_label in CLASSES:
                    selected = subset.loc[subset["class_label"] == class_label, "observed_count"]
                    if len(selected) != len(split_runs[split]):
                        raise DGDecompositionError("class count contract is not canonical")
                    counts.append(float(selected.sum()))
                values = np.asarray(counts, dtype=np.float64)
                if np.any(values <= 0.0):
                    raise DGDecompositionError("class count is nonpositive")
                weights[qi, si, hi] = values / values.sum()
    if not np.allclose(weights.sum(axis=-1), 1.0, rtol=0.0, atol=ABS_TOL):
        raise DGDecompositionError("class weights do not sum to one")
    return weights


def distance_from_gram(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = np.asarray(G, dtype=np.float64)
    diagonal = np.diagonal(gram, axis1=-2, axis2=-1)
    q = diagonal[..., :, None] + diagonal[..., None, :] - 2.0 * gram
    scale = np.maximum.reduce(
        [
            np.ones_like(q),
            np.abs(diagonal[..., :, None]) + np.zeros_like(q),
            np.abs(diagonal[..., None, :]) + np.zeros_like(q),
            np.abs(2.0 * gram),
        ]
    )
    if np.any(q < -REL_TOL * scale):
        index = tuple(int(v) for v in np.argwhere(q < -REL_TOL * scale)[0])
        raise DGDecompositionError(f"negative tangent squared distance at {index}")
    cleanup = np.where(q < 0.0, -q, 0.0)
    clean = np.where(q < 0.0, 0.0, q)
    result = np.sqrt(clean)
    diagonal_index = np.arange(4)
    result[..., diagonal_index, diagonal_index] = 0.0
    return result, cleanup


def distance_vector(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    return np.stack([array[..., i, j] for i, j in D_UPPER_TRIANGLE_INDICES], axis=-1)


def gram_vector(matrix: np.ndarray) -> np.ndarray:
    return svec(np.asarray(matrix, dtype=np.float64))


def vectorizer(name: str) -> Callable[[np.ndarray], np.ndarray]:
    return distance_vector if name in ("D_exact", "D_tan") else gram_vector


def _normalize(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1)
    thresholds = 100.0 * np.finfo(np.float64).eps * np.maximum(
        1.0, np.max(np.abs(values), axis=-1)
    )
    if np.any(norms <= thresholds):
        index = tuple(int(v) for v in np.argwhere(norms <= thresholds)[0])
        raise DGDecompositionError(f"DEGENERATE_CLASS_GEOMETRY at {index}")
    return values / norms[..., None], norms


def derive_objects(D_exact: np.ndarray, G: np.ndarray, weights: np.ndarray) -> DerivedBundle:
    if D_exact.shape != (2, 2, 9, 3, 4, 4) or G.shape != D_exact.shape:
        raise DGDecompositionError("source matrix stack has an invalid shape")
    if weights.shape != (2, 9, 3, 4):
        raise DGDecompositionError("weight stack has an invalid shape")
    H = np.eye(4)[None, None, None, :, :] - np.ones((1, 1, 1, 4, 1)) * weights[..., None, :]
    expanded_H = H[:, None]
    G0 = expanded_H @ G @ np.swapaxes(expanded_H, -1, -2)
    K_exact = -0.5 * (expanded_H @ (D_exact**2) @ np.swapaxes(expanded_H, -1, -2))
    D_tan, cleanup = distance_from_gram(G)
    matrices = {
        "D_exact": D_exact.copy(),
        "D_tan": D_tan,
        "K_exact": K_exact,
        "G0": G0,
        "G": G.copy(),
    }
    raw: dict[str, np.ndarray] = {}
    unit: dict[str, np.ndarray] = {}
    norms: dict[str, np.ndarray] = {}
    for name, matrix in matrices.items():
        raw[name] = vectorizer(name)(matrix)
        unit[name], norms[name] = _normalize(raw[name])
    return DerivedBundle(
        D_exact=matrices["D_exact"],
        D_tan=matrices["D_tan"],
        K_exact=matrices["K_exact"],
        G0=matrices["G0"],
        G=matrices["G"],
        weights=weights,
        H=H,
        q_cleanup=cleanup,
        shapes_raw=raw,
        shapes_unit=unit,
        shape_norms=norms,
    )


def identity_gate_table(bundle: DerivedBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    permutations = all_s4_permutations()
    test_permutation = permutations[17]
    matrices = {name: bundle.matrix(name) for name in OBJECTS}
    for qi, session in enumerate(SESSIONS):
        for gi, geometry in enumerate(GEOMETRIES):
            for si, subject in enumerate(SUBJECTS):
                for hi, split in enumerate(SPLITS):
                    G = bundle.G[qi, gi, si, hi]
                    G0 = bundle.G0[qi, gi, si, hi]
                    D = bundle.D_exact[qi, gi, si, hi]
                    Dtan = bundle.D_tan[qi, gi, si, hi]
                    H = bundle.H[qi, si, hi]
                    K = bundle.K_exact[qi, gi, si, hi]
                    direct_G0 = H @ G @ H.T
                    direct_K = -0.5 * H @ (D**2) @ H.T
                    from_G0, _ = distance_from_gram(G0)
                    q_expr = (
                        np.diag(G)[:, None] + np.diag(G)[None, :] - 2.0 * G
                    )
                    q_expr = np.where(q_expr < 0.0, 0.0, q_expr)
                    np.fill_diagonal(q_expr, 0.0)
                    q_error = relative_error(Dtan**2, q_expr)
                    tangent_translation_error = relative_error(Dtan, from_G0)
                    g0_formula_error = relative_error(G0, direct_G0)
                    k_formula_error = relative_error(K, direct_K)
                    le_d_error = relative_error(D**2, Dtan**2) if geometry == "LE" else np.nan
                    le_g_error = relative_error(G, G0) if geometry == "LE" else np.nan
                    le_k_error = relative_error(K, G0) if geometry == "LE" else np.nan
                    curvature = D**2 - Dtan**2
                    curvature_scale = np.maximum.reduce(
                        [np.ones_like(curvature), np.abs(D**2), np.abs(Dtan**2)]
                    )
                    curvature_min_scaled = float(np.min(curvature / curvature_scale))
                    equivariance_errors = []
                    for name in ("D_tan", "G0", "K_exact"):
                        original = matrices[name][qi, gi, si, hi]
                        permuted_parent = (
                            G[np.ix_(test_permutation, test_permutation)]
                            if name in ("D_tan", "G0")
                            else D[np.ix_(test_permutation, test_permutation)]
                        )
                        perm_weights = bundle.weights[qi, si, hi][test_permutation]
                        perm_H = np.eye(4) - np.ones((4, 1)) * perm_weights[None, :]
                        if name == "D_tan":
                            derived, _ = distance_from_gram(permuted_parent)
                        elif name == "G0":
                            derived = perm_H @ permuted_parent @ perm_H.T
                        else:
                            derived = -0.5 * perm_H @ (permuted_parent**2) @ perm_H.T
                        expected = original[np.ix_(test_permutation, test_permutation)]
                        equivariance_errors.append(relative_error(derived, expected))
                    scaling_errors = []
                    scale_factor = 3.25
                    for name in OBJECTS:
                        matrix = matrices[name][qi, gi, si, hi]
                        raw_scaled = vectorizer(name)(scale_factor * matrix)
                        unit_scaled = normalize_shape_vectors(raw_scaled[None, :])[0]
                        unit = bundle.shapes_unit[name][qi, gi, si, hi]
                        scaling_errors.append(relative_error(unit_scaled, unit))
                    passed = (
                        q_error <= REL_TOL
                        and g0_formula_error <= REL_TOL
                        and tangent_translation_error <= REL_TOL
                        and k_formula_error <= REL_TOL
                        and max(equivariance_errors) <= REL_TOL
                        and max(scaling_errors) <= REL_TOL
                        and (geometry != "LE" or max(le_d_error, le_g_error, le_k_error) <= REL_TOL)
                        and (geometry != "AIRM" or curvature_min_scaled >= -REL_TOL)
                    )
                    rows.append(
                        {
                            "session": session,
                            "geometry": geometry,
                            "subject": subject,
                            "split": split,
                            "D_tan_reconstruction_relative_error": q_error,
                            "G0_formula_relative_error": g0_formula_error,
                            "D_tan_translation_relative_error": tangent_translation_error,
                            "K_exact_formula_relative_error": k_formula_error,
                            "LE_D_exact_D_tan_relative_error": le_d_error,
                            "LE_G_G0_relative_error": le_g_error,
                            "LE_K_exact_G0_relative_error": le_k_error,
                            "permutation_equivariance_max_relative_error": max(equivariance_errors),
                            "common_scaling_shape_max_relative_error": max(scaling_errors),
                            "AIRM_curvature_min_scaled": curvature_min_scaled if geometry == "AIRM" else np.nan,
                            "q_cleanup_max": float(np.max(bundle.q_cleanup[qi, gi, si, hi])),
                            "K_exact_psd_clipped": False,
                            "passed": passed,
                        }
                    )
    frame = pd.DataFrame(rows)
    if not bool(frame["passed"].all()):
        raise DGDecompositionError("one or more derived identity hard gates failed")
    return frame


def shapes_table(bundle: DerivedBundle, name: str) -> pd.DataFrame:
    rows = []
    raw = bundle.shapes_raw[name]
    unit = bundle.shapes_unit[name]
    norms = bundle.shape_norms[name]
    for qi, session in enumerate(SESSIONS):
        for gi, geometry in enumerate(GEOMETRIES):
            for si, subject in enumerate(SUBJECTS):
                for hi, split in enumerate(SPLITS):
                    row: dict[str, Any] = {
                        "session": session,
                        "geometry": geometry,
                        "subject": subject,
                        "split": split,
                        "shape_norm": norms[qi, gi, si, hi],
                    }
                    row.update({f"raw_{j}": value for j, value in enumerate(raw[qi, gi, si, hi])})
                    row.update({f"z_{j}": value for j, value in enumerate(unit[qi, gi, si, hi])})
                    rows.append(row)
    return pd.DataFrame(rows)


def descriptive_tables(bundle: DerivedBundle) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchor_rows = []
    curvature_rows = []
    pair_rows = []
    for qi, session in enumerate(SESSIONS):
        for gi, geometry in enumerate(GEOMETRIES):
            for si, subject in enumerate(SUBJECTS):
                for hi, split in enumerate(SPLITS):
                    pi = bundle.weights[qi, si, hi]
                    G = bundle.G[qi, gi, si, hi]
                    energy = float(pi @ G @ pi)
                    trace = float(np.trace(G))
                    threshold = 100.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(G))))
                    fraction = energy / trace if trace > threshold else np.nan
                    anchor_rows.append(
                        {
                            "session": session,
                            "geometry": geometry,
                            "subject": subject,
                            "split": split,
                            **{f"pi_{label}": pi[index] for index, label in enumerate(CLASSES)},
                            "balanced": bool(np.allclose(pi, 0.25, rtol=0.0, atol=ABS_TOL)),
                            "anchor_energy": energy,
                            "trace_G": trace,
                            "anchor_fraction": fraction,
                        }
                    )
                    exact = distance_vector(bundle.D_exact[qi, gi, si, hi])
                    tangent = distance_vector(bundle.D_tan[qi, gi, si, hi])
                    C = exact**2 - tangent**2
                    safe = np.finfo(np.float64).eps * max(1.0, float(np.max(exact**2)))
                    relative = C / np.maximum(exact**2, safe)
                    curvature_rows.append(
                        {
                            "session": session,
                            "geometry": geometry,
                            "subject": subject,
                            "split": split,
                            "mean_C": float(np.mean(C)),
                            "median_C": float(np.median(C)),
                            "max_C": float(np.max(C)),
                            "mean_relative_curvature": float(np.mean(relative)),
                            "max_relative_curvature": float(np.max(relative)),
                        }
                    )
                    for pair, d_exact, d_tan, c_value, relative_value in zip(
                        PAIR_LABELS, exact, tangent, C, relative, strict=True
                    ):
                        pair_rows.append(
                            {
                                "session": session,
                                "geometry": geometry,
                                "subject": subject,
                                "split": split,
                                "class_pair": pair,
                                "D_exact": d_exact,
                                "D_tan": d_tan,
                                "C": c_value,
                                "relative_curvature": relative_value,
                            }
                        )
    return pd.DataFrame(anchor_rows), pd.DataFrame(curvature_rows), pd.DataFrame(pair_rows)


def reliability_table(repo: Path, config: Mapping[str, Any], bundle: DerivedBundle) -> pd.DataFrame:
    source = pd.read_csv(
        repo / config["project"]["source_output_dir"] / "tables/label_destruction_group_summary.csv"
    )
    rows = []
    for qi, session in enumerate(SESSIONS):
        phase = "discovery" if qi == 0 else "confirmatory"
        for gi, geometry in enumerate(GEOMETRIES):
            for name in OBJECTS:
                observed_subject = reliability_subject_scores(bundle.shapes_unit[name][qi, gi])
                observed = float(np.median(observed_subject))
                parent = "D" if name == "D_exact" else "G" if name == "G" else None
                if parent is not None:
                    match = source[
                        (source["phase"] == phase)
                        & (source["geometry"] == geometry)
                        & (source["object"] == parent)
                        & (source["stage"] == "R")
                    ]
                    if len(match) != 1:
                        raise DGDecompositionError("published Stage-R row is missing")
                    record = match.iloc[0]
                    if not np.isclose(observed, record["observed"], rtol=REL_TOL, atol=ABS_TOL):
                        raise DGDecompositionError("observed original Stage-R regression failed")
                    rows.append(
                        {
                            "session": session,
                            "geometry": geometry,
                            "object": name,
                            "observed": observed,
                            "null_median": float(record["null_median"]),
                            "effect": float(record["effect"]),
                            "p_value": float(record["p_value"]),
                            "replicates": int(record["replicates"]),
                            "status": "EXACT_V1_REGRESSION",
                        }
                    )
                else:
                    rows.append(
                        {
                            "session": session,
                            "geometry": geometry,
                            "object": name,
                            "observed": observed,
                            "null_median": np.nan,
                            "effect": np.nan,
                            "p_value": np.nan,
                            "replicates": 0,
                            "status": "NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE",
                        }
                    )
    return pd.DataFrame(rows)


def _selected_candidate_shapes(bank: np.ndarray, plans: np.ndarray, split: int) -> np.ndarray:
    subject_index = np.arange(bank.shape[0], dtype=np.int64)[None, :]
    return bank[:, split][subject_index, plans]


def _batched_loso_templates(selected_shapes: np.ndarray) -> np.ndarray:
    unit = normalize_shape_vectors(selected_shapes)
    sums = np.stack(
        [
            np.sum(np.delete(unit, target, axis=1), axis=1, dtype=np.float64)
            for target in range(unit.shape[1])
        ],
        axis=1,
    )
    return normalize_shape_vectors(sums)


def semantic_analysis(bundle: DerivedBundle, replicates: int = 100_000) -> SemanticResults:
    indices = np.arange(replicates, dtype=np.int64)
    plans = semantic_permutation_indices(indices, np.asarray(SUBJECTS, dtype=np.int64))
    observed = np.empty((2, 2, 5, 9), dtype=np.float64)
    null_subject = np.empty((2, 2, 5, replicates, 9), dtype=np.float64)
    summary_rows = []
    subject_rows = []
    for gi, geometry in enumerate(GEOMETRIES):
        for oi, name in enumerate(OBJECTS):
            matrix_a = bundle.matrix(name)[0, gi]
            matrix_b = bundle.matrix(name)[1, gi]
            shapes_a = bundle.shapes_unit[name][0, gi]
            shapes_b = bundle.shapes_unit[name][1, gi]
            observed[0, gi, oi] = discovery_shared_subject_scores(shapes_a)
            observed[1, gi, oi] = confirmatory_shared_subject_scores(shapes_a, shapes_b)
            bank_a = permuted_shape_bank(matrix_a, vectorizer(name))
            identity_a = bank_a[:, :, 0]
            identity_b = permuted_shape_bank(matrix_b, vectorizer(name))[:, :, 0]
            for start in range(0, replicates, 5_000):
                stop = min(start + 5_000, replicates)
                batch_plans = plans[start:stop]
                source_a = _selected_candidate_shapes(bank_a, batch_plans, 0)
                source_b = _selected_candidate_shapes(bank_a, batch_plans, 1)
                template_a = _batched_loso_templates(source_a)
                template_b = _batched_loso_templates(source_b)
                null_subject[0, gi, oi, start:stop] = 0.5 * (
                    np.einsum("bsf,sf->bs", template_a, identity_a[:, 1], optimize=True)
                    + np.einsum("bsf,sf->bs", template_b, identity_a[:, 0], optimize=True)
                )
                source_f = _selected_candidate_shapes(bank_a, batch_plans, 2)
                template_f = _batched_loso_templates(source_f)
                null_subject[1, gi, oi, start:stop] = 0.5 * (
                    np.einsum("bsf,sf->bs", template_f, identity_b[:, 0], optimize=True)
                    + np.einsum("bsf,sf->bs", template_f, identity_b[:, 1], optimize=True)
                )
    null_group = np.median(null_subject, axis=-1)
    for qi, session in enumerate(SESSIONS):
        for gi, geometry in enumerate(GEOMETRIES):
            for oi, name in enumerate(OBJECTS):
                observed_group = float(np.median(observed[qi, gi, oi]))
                null_summary = plus_one_null_summary(observed_group, null_group[qi, gi, oi])
                percentiles = subject_null_percentiles(
                    observed[qi, gi, oi], null_subject[qi, gi, oi]
                )
                summary_rows.append(
                    {
                        "session": session,
                        "geometry": geometry,
                        "object": name,
                        "observed": observed_group,
                        "null_median": null_summary.null_median,
                        "effect": null_summary.effect,
                        "p_value": null_summary.p_value,
                        "exceedances": null_summary.exceedances,
                        "replicates": null_summary.replicates,
                    }
                )
                subject_null_median = np.median(null_subject[qi, gi, oi], axis=0)
                for si, subject in enumerate(SUBJECTS):
                    subject_rows.append(
                        {
                            "session": session,
                            "geometry": geometry,
                            "object": name,
                            "subject": subject,
                            "observed": observed[qi, gi, oi, si],
                            "null_median": subject_null_median[si],
                            "effect": observed[qi, gi, oi, si] - subject_null_median[si],
                            "null_percentile": percentiles[si],
                            "replicates": replicates,
                        }
                    )
    return SemanticResults(
        observed_subject=observed,
        null_subject=null_subject,
        null_group=null_group,
        summary=pd.DataFrame(summary_rows),
        subject_effects=pd.DataFrame(subject_rows),
    )


def stage_s_regression_gates(
    repo: Path, config: Mapping[str, Any], semantic: SemanticResults
) -> pd.DataFrame:
    output = repo / config["project"]["source_output_dir"]
    published = pd.read_csv(output / "tables/semantic_permutation_null_summary.csv")
    rows = []
    for qi, session in enumerate(SESSIONS):
        phase = "discovery" if qi == 0 else "confirmatory"
        null_path = output / "nulls" / phase / "semantic_permutation_group_statistics.npz"
        with np.load(null_path, allow_pickle=False) as archive:
            source_group = np.asarray(archive["group_statistics"], dtype=np.float64)
            if not np.array_equal(archive["replicate_indices"], np.arange(100_000)):
                raise DGDecompositionError("v1 semantic replicate index mismatch")
        for gi, geometry in enumerate(GEOMETRIES):
            for oi, (name, source_name) in enumerate((("D_exact", "D"), ("G", "G"))):
                object_index = 0 if source_name == "D" else 1
                current_group = semantic.null_group[qi, gi, OBJECTS.index(name)]
                distribution_error = relative_error(current_group, source_group[gi, object_index])
                current = semantic.summary[
                    (semantic.summary["session"] == session)
                    & (semantic.summary["geometry"] == geometry)
                    & (semantic.summary["object"] == name)
                ].iloc[0]
                source = published[
                    (published["phase"] == phase)
                    & (published["geometry"] == geometry)
                    & (published["object"] == source_name)
                ].iloc[0]
                metric_error = max(
                    abs(float(current[key]) - float(source[key]))
                    for key in ("observed", "null_median", "effect", "p_value")
                )
                passed = distribution_error <= REL_TOL and metric_error <= ABS_TOL
                rows.append(
                    {
                        "session": session,
                        "geometry": geometry,
                        "object": name,
                        "v1_null_distribution_relative_error": distribution_error,
                        "v1_summary_max_absolute_error": metric_error,
                        "passed": passed,
                    }
                )
    frame = pd.DataFrame(rows)
    if not bool(frame["passed"].all()):
        raise DGDecompositionError("original D/G Stage-S regression gate failed")
    return frame


def oracle_table(bundle: DerivedBundle) -> tuple[pd.DataFrame, np.ndarray]:
    permutations = all_s4_permutations()
    scores_all = np.empty((2, 2, 5, 9, 24), dtype=np.float64)
    rows = []
    for gi, geometry in enumerate(GEOMETRIES):
        for oi, name in enumerate(OBJECTS):
            discovery_shapes = bundle.shapes_unit[name][0, gi]
            discovery_candidates = permuted_shape_bank(bundle.matrix(name)[0, gi], vectorizer(name))
            confirmatory_candidates = permuted_shape_bank(bundle.matrix(name)[1, gi], vectorizer(name))
            score_a = discovery_oracle_score_sets(discovery_shapes, discovery_candidates)
            score_b = confirmatory_oracle_score_sets(discovery_shapes, confirmatory_candidates)
            scores_all[0, gi, oi] = score_a
            scores_all[1, gi, oi] = score_b
            for qi, (session, scores) in enumerate(zip(SESSIONS, (score_a, score_b), strict=True)):
                summary = summarize_oracle_scores(scores, permutations)
                for si, subject in enumerate(SUBJECTS):
                    for pi, permutation in enumerate(permutations):
                        rows.append(
                            {
                                "session": session,
                                "geometry": geometry,
                                "object": name,
                                "subject": subject,
                                "permutation_index": pi,
                                "permutation": "-".join(str(int(value)) for value in permutation),
                                "is_identity": pi == 0,
                                "score": scores[si, pi],
                                "true_rank": int(summary.identity_ranks[si]) if pi == 0 else np.nan,
                                "normalized_rank": summary.normalized_ranks[si] if pi == 0 else np.nan,
                                "top1": bool(summary.top1_exact[si]) if pi == 0 else np.nan,
                                "margin": summary.margins[si] if pi == 0 else np.nan,
                            }
                        )
    return pd.DataFrame(rows), scores_all


def mechanism_tables(semantic: SemanticResults) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], str]:
    summary = semantic.summary.set_index(["session", "geometry", "object"])
    contrast_rows = []
    support_by_session: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for session in SESSIONS:
        effects = {
            geometry: {
                name: float(summary.loc[(session, geometry, name), "effect"])
                for name in OBJECTS
            }
            for geometry in GEOMETRIES
        }
        per_geometry = {}
        for geometry in GEOMETRIES:
            current = effects[geometry]
            per_geometry[geometry] = {
                "Delta_curvature": current["D_tan"] - current["D_exact"],
                "Delta_anchor": current["G"] - current["G0"],
                "Delta_encoding_exact": current["K_exact"] - current["D_exact"],
                "Delta_encoding_tangent": current["G0"] - current["D_tan"],
            }
            contrast_rows.append(
                {"session": session, "geometry": geometry, **per_geometry[geometry]}
            )
        le_zero_curvature = abs(per_geometry["LE"]["Delta_curvature"]) <= REL_TOL
        le_zero_anchor = abs(per_geometry["LE"]["Delta_anchor"]) <= REL_TOL
        curvature = (
            per_geometry["AIRM"]["Delta_curvature"] > REL_TOL
            and le_zero_curvature
            and np.sign(effects["AIRM"]["D_tan"]) == np.sign(effects["AIRM"]["G0"])
        )
        anchor = per_geometry["AIRM"]["Delta_anchor"] > REL_TOL and le_zero_anchor
        exact_closer = all(
            abs(effects[geometry]["K_exact"] - effects[geometry]["G0"])
            < abs(effects[geometry]["D_exact"] - effects[geometry]["G0"])
            for geometry in GEOMETRIES
        )
        tangent_difference = all(
            abs(per_geometry[geometry]["Delta_encoding_tangent"]) > REL_TOL
            for geometry in GEOMETRIES
        )
        encoding = exact_closer or tangent_difference
        supports = {
            name
            for name, supported in (
                ("curvature", curvature),
                ("anchor", anchor),
                ("encoding", encoding),
            )
            if supported
        }
        support_by_session[session] = supports
        if len(supports) > 1:
            labels[session] = "PROVISIONAL_MIXED"
        elif supports == {"curvature"}:
            labels[session] = "PROVISIONAL_CURVATURE_SUPPORTED"
        elif supports == {"anchor"}:
            labels[session] = "PROVISIONAL_ANCHOR_SUPPORTED"
        elif supports == {"encoding"}:
            labels[session] = "PROVISIONAL_ENCODING_SUPPORTED"
        else:
            labels[session] = "PROVISIONAL_UNRESOLVED"
    replicated = support_by_session["0train"] & support_by_session["1test"]
    if not replicated:
        overall = "PROVISIONAL_UNRESOLVED"
    elif len(replicated) > 1:
        overall = "PROVISIONAL_MIXED"
    elif replicated == {"curvature"}:
        overall = "PROVISIONAL_CURVATURE_SUPPORTED"
    elif replicated == {"anchor"}:
        overall = "PROVISIONAL_ANCHOR_SUPPORTED"
    else:
        overall = "PROVISIONAL_ENCODING_SUPPORTED"
    replication_rows = []
    for mechanism in ("curvature", "anchor", "encoding"):
        replication_rows.append(
            {
                "mechanism": mechanism,
                "session0_supported": mechanism in support_by_session["0train"],
                "session1_supported": mechanism in support_by_session["1test"],
                "direction_replicated": mechanism in replicated,
                "session0_label": labels["0train"],
                "session1_label": labels["1test"],
                "overall_label": overall,
            }
        )
    return pd.DataFrame(contrast_rows), pd.DataFrame(replication_rows), labels, overall


def _save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_figures(
    output: Path,
    semantic: SemanticResults,
    contrasts: pd.DataFrame,
    pairwise: pd.DataFrame,
    anchors: pd.DataFrame,
    gates: pd.DataFrame,
    replication: pd.DataFrame,
) -> None:
    figure_dir = output / "figures"
    colors = {name: plt.cm.tab10(index) for index, name in enumerate(OBJECTS)}

    f1 = semantic.summary[["session", "geometry", "object", "effect"]].copy()
    f1.to_csv(figure_dir / "figure_1_stage_s_effects.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for ax, (session, geometry) in zip(axes.flat, itertools.product(SESSIONS, GEOMETRIES), strict=True):
        data = f1[(f1.session == session) & (f1.geometry == geometry)].set_index("object").loc[list(OBJECTS)]
        ax.bar(OBJECTS, data.effect, color=[colors[name] for name in OBJECTS])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(f"{session} · {geometry}")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("Stage-S effect")
    fig.suptitle("Null-referenced shared-semantic effects")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure_1_stage_s_effects")

    f2 = semantic.summary[semantic.summary.object.isin(["D_exact", "K_exact", "D_tan", "G0"])][
        ["session", "geometry", "object", "effect"]
    ].copy()
    f2["pair"] = f2.object.map({"D_exact": "exact", "K_exact": "exact", "D_tan": "tangent", "G0": "tangent"})
    f2.to_csv(figure_dir / "figure_2_matched_pairs.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, (session, geometry) in zip(axes.flat, itertools.product(SESSIONS, GEOMETRIES), strict=True):
        data = f2[(f2.session == session) & (f2.geometry == geometry)]
        for pair, names in (("exact", ["D_exact", "K_exact"]), ("tangent", ["D_tan", "G0"])):
            values = data.set_index("object").loc[names, "effect"].to_numpy()
            x = [0, 1] if pair == "exact" else [3, 4]
            ax.plot(x, values, marker="o", linewidth=2, label=pair)
        ax.set_xticks([0, 1, 3, 4], ["D_exact", "K_exact", "D_tan", "G0"], rotation=30)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(f"{session} · {geometry}")
    fig.suptitle("Matched-information encoding contrasts")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure_2_matched_pairs")

    f3 = pairwise[pairwise.geometry == "AIRM"].copy()
    f3.to_csv(figure_dir / "figure_3_airm_distance_scatter.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (session, split) in zip(axes.flat, itertools.product(SESSIONS, SPLITS), strict=True):
        data = f3[(f3.session == session) & (f3.split == split)]
        ax.scatter(data.D_exact, data.D_tan, c=data.subject, cmap="viridis", s=22, alpha=.8)
        low = min(data.D_exact.min(), data.D_tan.min())
        high = max(data.D_exact.max(), data.D_tan.max())
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_title(f"{session} · split {split}")
        ax.set_xlabel("D_exact")
        ax.set_ylabel("D_tan")
    fig.suptitle("AIRM exact versus tangent-predicted distances")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure_3_airm_distance_scatter")

    f4 = f3[f3.split == "F"].copy()
    f4.to_csv(figure_dir / "figure_4_airm_curvature_by_pair.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, session in zip(axes, SESSIONS, strict=True):
        pivot = f4[f4.session == session].pivot(index="subject", columns="class_pair", values="C")
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
        ax.set_xticks(range(6), pivot.columns)
        ax.set_yticks(range(9), pivot.index)
        ax.set_title(session)
        ax.set_xlabel("class pair")
        ax.set_ylabel("subject")
        fig.colorbar(image, ax=ax, label="C")
    fig.suptitle("AIRM curvature distortion (F split)")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure_4_airm_curvature_by_pair")

    effects = semantic.summary[semantic.summary.object.isin(["G", "G0"])][["session", "geometry", "object", "effect"]]
    energy = anchors[anchors.split == "F"][["session", "geometry", "subject", "anchor_energy", "anchor_fraction"]]
    f5 = effects.merge(energy.groupby(["session", "geometry"], as_index=False).median(numeric_only=True), on=["session", "geometry"])
    f5.to_csv(figure_dir / "figure_5_anchor_effect.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(SESSIONS) * len(GEOMETRIES))
    labels_x = [f"{s}\n{g}" for s, g in itertools.product(SESSIONS, GEOMETRIES)]
    for offset, name in ((-.18, "G0"), (.18, "G")):
        vals = [float(effects[(effects.session == s) & (effects.geometry == g) & (effects.object == name)].effect.iloc[0]) for s, g in itertools.product(SESSIONS, GEOMETRIES)]
        axes[0].bar(x + offset, vals, width=.36, label=name)
    axes[0].set_xticks(x, labels_x)
    axes[0].set_ylabel("Stage-S effect")
    axes[0].legend()
    anchor_med = energy.groupby(["session", "geometry"], as_index=False).median(numeric_only=True)
    axes[1].bar(x, anchor_med.anchor_energy)
    axes[1].set_xticks(x, labels_x)
    axes[1].set_ylabel("median anchor energy")
    fig.suptitle("Anchor removal and offset energy")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure_5_anchor_effect")

    columns = ["LE_D_exact_D_tan_relative_error", "LE_G_G0_relative_error", "LE_K_exact_G0_relative_error"]
    f6 = gates[gates.geometry == "LE"][["session", "subject", "split", *columns]].copy()
    f6.to_csv(figure_dir / "figure_6_le_algebraic_control.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    values = [f6[column].to_numpy() for column in columns]
    ax.boxplot(values, tick_labels=["D_exact² vs D_tan²", "G vs G0", "K_exact vs G0"], showfliers=True)
    ax.axhline(REL_TOL, color="red", linestyle="--", label="hard tolerance")
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.set_ylabel("relative error")
    ax.legend()
    ax.set_title("LE algebraic controls at numerical zero")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure_6_le_algebraic_control")

    f7 = semantic.subject_effects[["session", "geometry", "object", "subject", "effect"]].copy()
    f7.to_csv(figure_dir / "figure_7_subject_stage_s_effects.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, (session, geometry) in zip(axes.flat, itertools.product(SESSIONS, GEOMETRIES), strict=True):
        data = f7[(f7.session == session) & (f7.geometry == geometry)]
        for name in OBJECTS:
            row = data[data.object == name]
            ax.plot(row.subject, row.effect, marker="o", label=name, color=colors[name])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(f"{session} · {geometry}")
        ax.set_xlabel("subject")
        ax.set_ylabel("subject null-referenced effect")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Subject-level Stage-S effects")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure_7_subject_stage_s_effects")

    f8 = contrasts.copy()
    for column in ("session0_supported", "session1_supported", "direction_replicated"):
        f8[column] = ";".join(
            f"{row.mechanism}:{int(getattr(row, column))}" for row in replication.itertuples()
        )
    f8.to_csv(figure_dir / "figure_8_mechanism_decision.csv", index=False)
    matrix = contrasts.set_index(["session", "geometry"])[
        ["Delta_curvature", "Delta_anchor", "Delta_encoding_exact", "Delta_encoding_tangent"]
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    bound = float(np.max(np.abs(matrix.to_numpy())))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    ax.set_xticks(range(4), ["curvature", "anchor", "encoding exact", "encoding tangent"], rotation=20)
    ax.set_yticks(range(4), [f"{s} · {g}" for s, g in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="effect contrast")
    ax.set_title(f"Mechanism decision: {replication.overall_label.iloc[0]}")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure_8_mechanism_decision")


def _write_report(
    output: Path,
    base_commit: str,
    implementation_commit: str,
    semantic: SemanticResults,
    contrasts: pd.DataFrame,
    labels: Mapping[str, str],
    overall: str,
    curvature: pd.DataFrame,
    anchors: pd.DataFrame,
    reliability: pd.DataFrame,
) -> None:
    lines = [
        "# D/G Discrepancy Decomposition v1",
        "",
        "## Status and scope",
        "",
        "This is a retrospective/post hoc mechanism anatomy, not a new method and not external confirmation. Session `0train` is retrospective mechanism set A; session `1test` is locked internal replication set B. The original `STOP_TANGENT_ONLY` decision and all Conditional Geometry Anatomy v1 artifacts remain untouched.",
        "",
        "D-/G+ must not be reinterpreted as tangent superiority unless matched-object evidence supports that interpretation. This diagnostic introduces no classifier, alignment loss, pseudo-label, neural network, trajectory, HGD, or WINDOW5 analysis. Oracle semantic-name scoring remains descriptive; oracle component recovery is not solved.",
        "",
        f"Base commit: `{base_commit}`. Diagnostic implementation commit: `{implementation_commit}`.",
        "",
        "## Algebraic controls",
        "",
        "All stored-object identity gates passed. LE satisfied `D_exact²=D_tan²` and `K_exact=G0=G` within 1e-10. AIRM curvature residuals satisfied the exponential-metric-increasing inequality within the frozen tolerance. K_exact was retained signed and was never PSD-clipped.",
        "",
        "## Stage R mechanism prerequisite",
        "",
        "Observed A/B reliability was computed for all five objects. Exact v1 D/G label-null summaries were reproduced. The immutable source does not store trial covariances/metadata or per-replicate fitted D/G objects, so the derived-object label nulls cannot be exactly computed from the authorized source. They are explicitly marked `NOT_COMPUTABLE_FROM_IMMUTABLE_SOURCE`; no proxy or approximation was substituted, and Stage R does not vote in the mechanism label.",
        "",
        reliability.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Stage S null-referenced effects",
        "",
        semantic.summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Mechanism contrasts",
        "",
        contrasts.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Provisional decision",
        "",
        f"- Session 0 label: **{labels['0train']}**",
        f"- Session 1 label: **{labels['1test']}**",
        f"- Overall: **{overall}**",
        "",
        "The label is provisional and uses algebraic matching plus replicated contrast direction, not a new post hoc absolute cutoff and not a single p-value. It is not a scientific GO before external confirmation.",
        "",
        "## Descriptive distortion summaries",
        "",
        "AIRM curvature (F split, median across subjects):",
        "",
        curvature[(curvature.geometry == "AIRM") & (curvature.split == "F")].groupby("session")[["median_C", "max_C"]].median().to_markdown(floatfmt=".6g"),
        "",
        "Anchor fraction (F split, median across subjects):",
        "",
        anchors[anchors.split == "F"].groupby(["session", "geometry"])["anchor_fraction"].median().to_frame().to_markdown(floatfmt=".6g"),
        "",
        "## Limits",
        "",
        "The analysis decomposes the stored representation/statistic discrepancy only. It does not establish a biological mechanism, a deployable adaptation method, or unlabeled component recovery. WINDOW5 remains outside this diagnostic.",
    ]
    (output / "report/dg_discrepancy_decomposition_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(output: Path, metadata: Mapping[str, Any]) -> None:
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"):
        records.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {**metadata, "artifacts": records, "artifact_count": len(records)}
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_diagnostic(repo: Path, config_path: Path, *, tests: str = "PASS") -> RunResult:
    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_config(config_path)
    if _git(repo, "rev-parse", config["protocol"]["base_commit"] + "^{commit}") != config["protocol"]["base_commit"]:
        raise DGDecompositionError("base commit is unavailable")
    if _git(repo, "branch", "--show-current") != config["protocol"]["branch"]:
        raise DGDecompositionError("diagnostic branch mismatch")
    output = repo / config["project"]["output_dir"]
    if output.exists():
        raise DGDecompositionError(f"refusing to overwrite existing diagnostic output: {output}")
    source_audit, before_aggregate = _audit_sources(repo, config)
    D, G = load_source_matrices(repo, config)
    weights = class_weights_from_contract(repo, config)
    bundle = derive_objects(D, G, weights)
    gates = identity_gate_table(bundle)
    anchors, curvature, pairwise = descriptive_tables(bundle)
    reliability = reliability_table(repo, config, bundle)
    semantic = semantic_analysis(bundle, int(config["nulls"]["semantic_permutation"]["replicates"]))
    regression = stage_s_regression_gates(repo, config, semantic)
    gates = pd.concat(
        [
            gates,
            regression.assign(
                subject=np.nan,
                split="STAGE_S_REGRESSION",
                D_tan_reconstruction_relative_error=np.nan,
                G0_formula_relative_error=np.nan,
                D_tan_translation_relative_error=np.nan,
                K_exact_formula_relative_error=np.nan,
                LE_D_exact_D_tan_relative_error=np.nan,
                LE_G_G0_relative_error=np.nan,
                LE_K_exact_G0_relative_error=np.nan,
                permutation_equivariance_max_relative_error=np.nan,
                common_scaling_shape_max_relative_error=np.nan,
                AIRM_curvature_min_scaled=np.nan,
                q_cleanup_max=np.nan,
                K_exact_psd_clipped=False,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    oracle, oracle_scores = oracle_table(bundle)
    contrasts, replication, labels, overall = mechanism_tables(semantic)

    for directory in ("protocol", "objects", "tables", "nulls", "figures", "report"):
        (output / directory).mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "protocol/frozen_config.yaml")
    shutil.copy2(repo / config["protocol"]["protocol_path"], output / "protocol/PROTOCOL_DG_DISCREPANCY_DECOMPOSITION_V1.md")

    np.savez_compressed(
        output / "objects/derived_objects.npz",
        sessions=np.asarray(SESSIONS), geometries=np.asarray(GEOMETRIES),
        subjects=np.asarray(SUBJECTS), splits=np.asarray(SPLITS), classes=np.asarray(CLASSES),
        weights=bundle.weights, D_exact=bundle.D_exact, D_tan=bundle.D_tan,
        K_exact=bundle.K_exact, G0=bundle.G0, G=bundle.G,
        q_cleanup=bundle.q_cleanup,
    )
    np.savez_compressed(output / "objects/oracle_permutation_scores.npz", scores=oracle_scores, permutations=all_s4_permutations())
    for qi, session in enumerate(SESSIONS):
        np.savez_compressed(
            output / f"nulls/semantic_permutation_{session}.npz",
            replicate_indices=np.arange(semantic.null_subject.shape[-2], dtype=np.int64),
            geometries=np.asarray(GEOMETRIES), objects=np.asarray(OBJECTS),
            subject_statistics=semantic.null_subject[qi], group_statistics=semantic.null_group[qi],
        )
    table_map = {
        "source_artifact_audit.csv": source_audit,
        "derived_object_identity_gates.csv": gates,
        "D_exact_shapes.csv": shapes_table(bundle, "D_exact"),
        "D_tan_shapes.csv": shapes_table(bundle, "D_tan"),
        "G_shapes.csv": shapes_table(bundle, "G"),
        "G0_shapes.csv": shapes_table(bundle, "G0"),
        "K_exact_shapes.csv": shapes_table(bundle, "K_exact"),
        "anchor_offset_summary.csv": anchors,
        "curvature_distortion_summary.csv": curvature,
        "reliability_summary.csv": reliability,
        "shared_semantic_summary.csv": semantic.summary,
        "subject_semantic_effects.csv": semantic.subject_effects,
        "mechanism_contrasts.csv": contrasts,
        "oracle_descriptive_summary.csv": oracle,
        "session_replication_summary.csv": replication,
    }
    for filename, frame in table_map.items():
        frame.to_csv(output / "tables" / filename, index=False)
    make_figures(output, semantic, contrasts, pairwise, anchors, gates, replication)
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    _write_report(
        output, config["protocol"]["base_commit"], implementation_commit,
        semantic, contrasts, labels, overall, curvature, anchors, reliability,
    )
    _, after_aggregate = _audit_sources(repo, config)
    source_unchanged = before_aggregate == after_aggregate
    if not source_unchanged:
        raise DGDecompositionError("source output changed during diagnostic")
    _manifest(
        output,
        {
            "schema_version": "dg-discrepancy-decomposition-v1",
            "base_commit": config["protocol"]["base_commit"],
            "branch": config["protocol"]["branch"],
            "implementation_commit": implementation_commit,
            "protocol_sha256": sha256_file(repo / config["protocol"]["protocol_path"]),
            "config_sha256": sha256_file(config_path),
            "source_output_aggregate_sha256": after_aggregate,
            "source_outputs_unchanged": source_unchanged,
            "hard_gates": "PASS",
            "session0_label": labels["0train"],
            "session1_label": labels["1test"],
            "overall_provisional_mechanism": overall,
        },
    )
    return RunResult(output, overall, labels, tests, "PASS", source_unchanged)


def final_numerical_summary(output: Path) -> str:
    shared = pd.read_csv(output / "tables/shared_semantic_summary.csv")
    contrasts = pd.read_csv(output / "tables/mechanism_contrasts.csv")
    curvature = pd.read_csv(output / "tables/curvature_distortion_summary.csv")
    anchors = pd.read_csv(output / "tables/anchor_offset_summary.csv")
    lines = []
    for geometry in GEOMETRIES:
        for session in SESSIONS:
            rows = shared[(shared.geometry == geometry) & (shared.session == session)].set_index("object")
            contrast = contrasts[(contrasts.geometry == geometry) & (contrasts.session == session)].iloc[0]
            lines.append(f"{geometry} {session}")
            for name in OBJECTS:
                lines.append(f"E_S({name})={rows.loc[name, 'effect']:.12g}")
            lines.extend(
                [
                    f"Delta_curvature={contrast.Delta_curvature:.12g}",
                    f"Delta_anchor={contrast.Delta_anchor:.12g}",
                    f"Delta_encoding_exact={contrast.Delta_encoding_exact:.12g}",
                    f"Delta_encoding_tangent={contrast.Delta_encoding_tangent:.12g}",
                ]
            )
            c = curvature[(curvature.geometry == geometry) & (curvature.session == session)]
            a = anchors[(anchors.geometry == geometry) & (anchors.session == session)]
            lines.append(f"median_curvature_distortion={c.median_C.median():.12g}")
            lines.append(f"max_curvature_distortion={c.max_C.max():.12g}")
            lines.append(f"median_anchor_fraction={a.anchor_fraction.median():.12g}")
    return "\n".join(lines)

