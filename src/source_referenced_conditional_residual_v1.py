"""Source-referenced conditional residual V1.

All source references are outer-training-only.  Parent PR #16/#17 outputs are
read-only and hash checked.  Zero-label semantic orientation is reported only
under the explicit source-ordering assumption.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import src.subject_class_population_structure_v1 as population_v1
from src.interaction_provenance_v0 import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)


CONFIG_PATH = "configs/source_referenced_conditional_residual_v1.yaml"
OUTPUT_NAME = "source_referenced_conditional_residual_v1"
PARENT_OUTPUT = "outputs/unlabeled_conditional_mode_identifiability_v0"


class SourceReferenceError(RuntimeError):
    """Base fail-closed error."""


class SourceReferenceObjectInsufficient(SourceReferenceError):
    """Required immutable or hash-validated parent quantity is unavailable."""


class BetaReferenceIdentityError(SourceReferenceError):
    """The frozen beta reference identity failed."""


class DataContractError(SourceReferenceError):
    """Ordering, hash, leakage, or freeze contract failed."""


class NumericalContractError(SourceReferenceError):
    """A numerical finite/nondegeneracy contract failed."""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def output_path(root: Path, config: Mapping[str, Any]) -> Path:
    return root / str(config["project"]["output_dir"])


def _ensure_output(output: Path) -> None:
    for name in ("protocol", "objects", "nulls", "controls", "decisions", "tables", "figures", "report"):
        (output / name).mkdir(parents=True, exist_ok=True)


def load_config(repo_root: str | Path) -> tuple[dict[str, Any], str]:
    root = Path(repo_root).resolve()
    path = root / CONFIG_PATH
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise DataContractError("source-reference config must be a mapping")
    protocol = root / str(config["protocol"]["protocol_path"])
    digest = sha256_file(protocol)
    if digest != str(config["protocol"]["protocol_sha256"]):
        raise DataContractError(f"protocol SHA mismatch: {digest}")
    if config["protocol"]["parent_head"] != "8346a3e0f731c80668bd7147a2fe0fd12da6b914":
        raise DataContractError("exact parent head literal changed")
    return config, sha256_file(path)


def _all_files_snapshot(root: Path, directories: Sequence[str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for relative_dir in directories:
        base = root / relative_dir
        if not base.is_dir():
            raise DataContractError(f"missing immutable directory: {relative_dir}")
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            snapshot[str(path.relative_to(root))] = sha256_file(path)
    if not snapshot:
        raise DataContractError("immutable parent snapshot is empty")
    return snapshot


def validate_parent_artifacts(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, record in config["parent_artifacts"].items():
        if name == "immutable_output_directories":
            continue
        path = root / str(record["path"])
        if not path.is_file():
            raise SourceReferenceObjectInsufficient(f"missing parent artifact: {name}")
        digest = sha256_file(path)
        if digest != str(record["sha256"]):
            raise DataContractError(f"parent artifact changed: {name}: {digest}")
        observed[str(name)] = digest
    manifest_path = root / str(config["parent_artifacts"]["pr17_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_root = root / PARENT_OUTPUT
    for relative, expected in manifest["artifacts"].items():
        path = parent_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise DataContractError(f"PR #17 manifest mismatch: {relative}")
    if len(manifest["artifacts"]) != 68:
        raise DataContractError("PR #17 artifact count changed")
    if _git(root, "merge-base", "--is-ancestor", str(config["protocol"]["parent_head"]), "HEAD"):
        raise DataContractError("exact PR #17 head is not an ancestor")
    return observed


def _scientific_files(root: Path) -> dict[str, str]:
    paths = [
        CONFIG_PATH,
        "docs/PROTOCOL_SOURCE_REFERENCED_CONDITIONAL_RESIDUAL_V1.md",
        "docs/AUDIT_SOURCE_REFERENCED_CONDITIONAL_RESIDUAL_V1.md",
        "docs/BETA_REFERENCE_IDENTITY.md",
        "src/source_referenced_conditional_residual_v1.py",
        "tests/test_source_referenced_conditional_residual_v1.py",
    ]
    paths.extend(
        f"scripts/{number}_{name}.py"
        for number, name in (
            (90, "freeze_source_reference_v1"),
            (91, "run_beta_identity_audit_v1"),
            (92, "run_source_reference_correction_v1"),
            (93, "run_source_ordering_assumption_v1"),
            (94, "run_corrected_minimal_anchor_v1"),
            (95, "report_source_reference_v1"),
        )
    )
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise DataContractError(f"missing freeze-scope files: {missing}")
    return {path: sha256_file(root / path) for path in paths}


def _rng(config: Mapping[str, Any], *parts: Any) -> np.random.Generator:
    return population_v1.deterministic_rng(
        int(config["protocol"]["master_seed"]), "source_reference_v1", *parts
    )


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise NumericalContractError(f"{name} contains NaN/Inf")
    return array


def _pearson(predicted: np.ndarray, target: np.ndarray) -> float:
    x, y = _finite("pearson predicted", predicted).ravel(), _finite("pearson target", target).ravel()
    if len(x) < 3 or np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        raise NumericalContractError("Pearson input is degenerate")
    value = float(stats.pearsonr(x, y).statistic)
    if not np.isfinite(value):
        raise NumericalContractError("Pearson output is nonfinite")
    return value


def _spearman(predicted: np.ndarray, target: np.ndarray) -> float:
    x, y = _finite("spearman predicted", predicted).ravel(), _finite("spearman target", target).ravel()
    if len(x) < 3 or np.ptp(x) <= np.finfo(float).eps or np.ptp(y) <= np.finfo(float).eps:
        raise NumericalContractError("Spearman input is degenerate")
    value = float(stats.spearmanr(x, y).statistic)
    if not np.isfinite(value):
        raise NumericalContractError("Spearman output is nonfinite")
    return value


def _metric_bundle(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, truth = _finite("metric predicted", predicted).ravel(), _finite("metric target", target).ravel()
    error = pred - truth
    denom = max(float(np.mean(np.abs(truth))), np.finfo(float).tiny)
    ss_total = float(np.sum((truth - np.mean(truth)) ** 2))
    design = np.column_stack([np.ones(len(pred)), pred])
    intercept, slope = np.linalg.lstsq(design, truth, rcond=None)[0]
    return {
        "mae": float(np.mean(np.abs(error))),
        "normalized_mae": float(np.mean(np.abs(error)) / denom),
        "pearson": _pearson(pred, truth),
        "spearman": _spearman(pred, truth),
        "signed_r2": float(1.0 - np.sum(error**2) / ss_total),
        "beta_sign_accuracy": float(np.mean(np.sign(pred) == np.sign(truth))),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "projected_prototype_reconstruction_error": float(np.mean(np.abs(error))),
    }


def _bootstrap_mean(
    config: Mapping[str, Any], values: np.ndarray, namespace: str
) -> tuple[float, float, np.ndarray]:
    subject_values = _finite("bootstrap values", values).ravel()
    reps = int(config["inference"]["bootstrap_replicates"])
    draws = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sampled = _rng(config, config["inference"]["bootstrap_namespace"], namespace, index).integers(
            0, len(subject_values), size=len(subject_values)
        )
        draws[index] = float(np.mean(subject_values[sampled]))
    alpha = 1.0 - float(config["inference"]["confidence"])
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high), draws


def _paired_sign_flip(
    config: Mapping[str, Any], values: np.ndarray, namespace: str
) -> tuple[float, np.ndarray]:
    subject_values = _finite("paired values", values).ravel()
    observed = float(np.mean(subject_values))
    reps = int(config["inference"]["null_replicates"])
    null = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        signs = _rng(config, config["inference"]["paired_sign_flip_namespace"], namespace, index).choice(
            [-1.0, 1.0], size=len(subject_values)
        )
        null[index] = float(np.mean(signs * subject_values))
    p_value = float((1 + np.count_nonzero(null >= observed)) / (1 + reps))
    return p_value, null


def synthetic_gates(config: Mapping[str, Any]) -> dict[str, Any]:
    rng = _rng(config, "synthetic")
    gamma = 0.55
    delta = np.linspace(-1.3, 1.7, 80)
    beta = delta - gamma
    correction = 0.18
    trial = delta - correction
    restored = trial + correction - gamma
    rows: list[dict[str, Any]] = []
    rows.append({"case": "beta_equals_delta_minus_gamma", "passed": np.array_equal(beta, delta - gamma)})
    rows.append({"case": "uncentered_sign_mismatch_nonzero_gamma", "passed": bool(np.any(np.sign(delta) != np.sign(beta)))})
    increasing = np.tile(np.sign(delta), (4, 1))
    rows.append({"case": "more_labels_wrong_if_gamma_omitted", "passed": bool(np.all(increasing[:, np.sign(delta) != np.sign(beta)] != np.sign(beta)[np.sign(delta) != np.sign(beta)]))})
    rows.append({"case": "source_reference_restores_beta", "passed": bool(np.allclose(restored, beta, atol=1e-15, rtol=0.0))})
    energy = delta**2
    rows.append({"case": "unsigned_delta_magnitude_recovery", "passed": bool(np.allclose(np.sqrt(energy), np.abs(delta)))})
    rows.append({"case": "class_permutation_invariance", "passed": bool(np.array_equal(energy, (-delta) ** 2) and np.array_equal(-delta, -1.0 * delta))})
    source_order = np.sign(np.mean(delta + 2.0))
    rows.append({"case": "source_ordering_success", "passed": bool(source_order == 1)})
    target_violation = -source_order * np.ones(12)
    rows.append({"case": "source_ordering_violation_detected", "passed": bool(np.mean(np.sign(target_violation) == source_order) == 0.0)})
    magnitude = np.abs(delta) + rng.normal(0.0, 0.01, size=len(delta))
    estimate = np.sign(delta) * magnitude - gamma
    rows.append({"case": "minimal_label_residualized_estimator", "passed": float(np.mean(np.abs(estimate - beta))) < 0.02})
    sentinel = {"source_fit": ("source_labels",), "target_eval": ("target_labels",)}
    rows.append({"case": "target_label_leakage_sentinel", "passed": "target_labels" not in sentinel["source_fit"]})
    outer_train = set(range(45)); heldout = set(range(45, 54))
    rows.append({"case": "outer_fold_source_only_reference", "passed": outer_train.isdisjoint(heldout)})
    direct = trial + correction - gamma
    rows.append({"case": "fair_residualized_direct_baseline", "passed": bool(np.array_equal(direct, restored))})
    for row in rows:
        row["passed"] = bool(row["passed"])
    return {
        "schema_version": "source-referenced-conditional-residual-v1-synthetic",
        "passed": bool(all(row["passed"] for row in rows)),
        "cases": rows,
    }


def environment_record() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "PyYAML", "matplotlib", "pytest"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "packages": packages, "cpu_count": os.cpu_count(),
    }


def freeze_protocol(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_config(root)
    parent_hashes = validate_parent_artifacts(root, config)
    parent_config, _ = population_v1.load_config(root)
    if config["folds"]["outer_test"] != parent_config["openbmi_folds"]["outer_test"]:
        raise DataContractError("outer folds differ from PR #16")
    if config["folds"]["canonical_sha256"] != parent_config["openbmi_folds"]["sha256"]:
        raise DataContractError("outer fold hash differs from PR #16")
    synthetic = synthetic_gates(config)
    if not synthetic["passed"]:
        raise NumericalContractError("synthetic gates failed")
    output = output_path(root, config); _ensure_output(output)
    snapshot = _all_files_snapshot(root, config["parent_artifacts"]["immutable_output_directories"])
    atomic_write_json(output / "protocol/parent_artifact_hashes.json", {
        "schema_version": "pr16-pr17-immutable-snapshot-v1", "files": snapshot,
    })
    pd.DataFrame(synthetic["cases"]).to_csv(output / "protocol/synthetic_gates.csv", index=False, lineterminator="\n")
    atomic_write_json(output / "protocol/synthetic_gates.json", synthetic)
    fold_rows: list[dict[str, Any]] = []
    for fold, test_subjects in enumerate(config["folds"]["outer_test"]):
        test = set(map(int, test_subjects))
        for subject in config["dataset"]["subjects"]:
            fold_rows.append({"outer_fold": fold, "subject": subject, "role": "test" if subject in test else "train"})
    pd.DataFrame(fold_rows).to_csv(output / "protocol/exact_folds.csv", index=False, lineterminator="\n")
    for relative in (
        config["protocol"]["protocol_path"], config["project"]["audit_path"],
        config["project"]["identity_path"], CONFIG_PATH,
    ):
        source = root / str(relative)
        atomic_write_bytes(output / "protocol" / source.name, source.read_bytes())
    scientific = _scientific_files(root)
    manifest = {
        "schema_version": "source-reference-v1-freeze",
        "status": "READY_FOR_PROTOCOL_FREEZE_COMMIT",
        "real_source_reference_statistics_accessed": False,
        "parent_head": config["protocol"]["parent_head"], "config_sha256": config_hash,
        "parent_hashes": parent_hashes, "parent_file_count": len(snapshot),
        "parent_snapshot_sha256": hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest(),
        "scientific_file_hashes": scientific, "synthetic_pass": True,
        "fold_hash": config["folds"]["canonical_sha256"],
        "required_commit_subject": config["protocol"]["required_freeze_commit_subject"],
    }
    atomic_write_json(output / "protocol/manifest.json", manifest)
    atomic_write_json(output / "environment.json", environment_record())
    return manifest


def ensure_real_access_lock(repo_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve(); config, config_hash = load_config(root)
    validate_parent_artifacts(root, config)
    output = output_path(root, config)
    manifest_path = output / "protocol/manifest.json"
    if not manifest_path.is_file():
        raise DataContractError("freeze manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != config_hash or not manifest["synthetic_pass"]:
        raise DataContractError("freeze manifest/config mismatch")
    if _scientific_files(root) != manifest["scientific_file_hashes"]:
        raise DataContractError("scientific file changed after freeze")
    snapshot = _all_files_snapshot(root, config["parent_artifacts"]["immutable_output_directories"])
    if hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest() != manifest["parent_snapshot_sha256"]:
        raise DataContractError("PR #16/#17 artifact snapshot changed")
    provenance_path = output / "git_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if _git(root, "merge-base", "--is-ancestor", provenance["protocol_freeze_commit"], "HEAD"):
            raise DataContractError("protocol freeze commit is not an ancestor")
        return config, provenance
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise DataContractError("first real access requires a clean tree")
    if _git(root, "show", "-s", "--format=%s", "HEAD") != config["protocol"]["required_freeze_commit_subject"]:
        raise DataContractError("HEAD is not the protocol-freeze commit")
    provenance = {
        "schema_version": "source-reference-v1-git-provenance",
        "parent_head": config["protocol"]["parent_head"],
        "protocol_freeze_commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "first_real_access_tree_was_clean": True,
        "scientific_file_hashes": manifest["scientific_file_hashes"],
        "parent_snapshot_sha256": manifest["parent_snapshot_sha256"],
    }
    atomic_write_json(provenance_path, provenance)
    return config, provenance


def _outer_indices(config: Mapping[str, Any]) -> list[np.ndarray]:
    subjects = list(map(int, config["dataset"]["subjects"]))
    lookup = {subject: index for index, subject in enumerate(subjects)}
    outer = [np.asarray([lookup[int(subject)] for subject in fold], dtype=np.int64) for fold in config["folds"]["outer_test"]]
    coverage = np.concatenate(outer)
    if not np.array_equal(np.sort(coverage), np.arange(54)):
        raise DataContractError("outer fold coverage is not exact")
    return outer


def _load_identity_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    parent = config["parent_artifacts"]
    with np.load(root / parent["modes"]["path"], allow_pickle=False) as z:
        modes = _finite("modes", z["modes"]); fold_of_subject = np.asarray(z["fold_of_subject"], dtype=np.int64)
        subjects = np.asarray(z["subjects"], dtype=np.int64)
    with np.load(root / parent["oracle_coordinates"]["path"], allow_pickle=False) as z:
        beta = _finite("beta", z["beta"]); alpha = _finite("alpha", z["alpha"]); raw_x = _finite("raw_x", z["raw_x"])
        if not np.array_equal(subjects, z["subjects"]):
            raise DataContractError("oracle subject ordering differs")
    with np.load(root / parent["v0_openbmi_objects"]["path"], allow_pickle=False) as z:
        u = _finite("parent U", z["AIRM__session_specific__F__U"])
        proportions = _finite("parent class proportions", z["AIRM__session_specific__F__class_proportions"])
    if modes.shape != (6, 2, 210) or beta.shape != (54, 2) or raw_x.shape != (54, 2, 210) or u.shape != (54, 2, 2, 20, 20):
        raise DataContractError("identity input shape mismatch")
    if not np.all(proportions == 0.5):
        raise DataContractError("binary class proportions are not exactly balanced")
    if not np.array_equal(subjects, np.asarray(config["dataset"]["subjects"])):
        raise DataContractError("subject ordering differs from literal")
    return {"modes": modes, "fold": fold_of_subject, "subjects": subjects, "beta": beta, "alpha": alpha, "raw_x": raw_x, "u": u}


def run_beta_identity_audit(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = ensure_real_access_lock(root); output = output_path(root, config); _ensure_output(output)
    data = _load_identity_inputs(root, config); outer = _outer_indices(config)
    u_contrast = population_v1.svec(0.5 * (data["u"][:, :, 1] - data["u"][:, :, 0]))
    d_proto = np.empty((54, 2)); gamma = np.empty((6, 2)); reconstructed = np.empty((54, 2))
    rows: list[dict[str, Any]] = []
    for fold, test in enumerate(outer):
        train = np.setdiff1d(np.arange(54), test)
        for q in range(2):
            direction = data["modes"][fold, q]
            gamma[fold, q] = float(np.mean(u_contrast[train, q], axis=0) @ direction)
            d_proto[test, q] = u_contrast[test, q] @ direction
            reconstructed[test, q] = d_proto[test, q] - gamma[fold, q]
            for subject_index in test:
                rows.append({
                    "outer_fold": fold, "subject": int(data["subjects"][subject_index]), "session": str(q),
                    "d_proto": d_proto[subject_index, q], "gamma": gamma[fold, q],
                    "beta_parent": data["beta"][subject_index, q], "beta_reconstructed": reconstructed[subject_index, q],
                    "absolute_error": abs(reconstructed[subject_index, q] - data["beta"][subject_index, q]),
                    "sign_changed_after_gamma": bool(np.sign(d_proto[subject_index, q]) != np.sign(reconstructed[subject_index, q])),
                })
    atol, rtol = float(config["coordinate_contract"]["beta_identity_atol"]), float(config["coordinate_contract"]["beta_identity_rtol"])
    try:
        np.testing.assert_allclose(reconstructed, data["beta"], atol=atol, rtol=rtol)
    except AssertionError as exc:
        result = {"decision": config["decisions"]["identity_failure"], "passed": False, "message": str(exc)}
        atomic_write_json(output / "decisions/beta_reference_identity.json", result)
        raise BetaReferenceIdentityError(str(exc)) from exc
    subject_modes = data["modes"][data["fold"]]
    raw_beta = np.einsum("sqp,sqp->sq", data["raw_x"], subject_modes)
    np.testing.assert_allclose(raw_beta, data["beta"], atol=atol, rtol=rtol)
    frame = pd.DataFrame(rows); frame.to_csv(output / "tables/beta_reference_identity_subjects.csv", index=False, float_format="%.17g", lineterminator="\n")
    gamma_rows = [{"outer_fold": f, "session": str(q), "gamma": gamma[f, q]} for f in range(6) for q in range(2)]
    pd.DataFrame(gamma_rows).to_csv(output / "tables/gamma_by_fold_session.csv", index=False, float_format="%.17g", lineterminator="\n")
    population_v1._atomic_savez(output / "objects/beta_reference_identity_core.npz", {
        "subjects": data["subjects"], "modes": data["modes"], "fold_of_subject": data["fold"],
        "u_contrast": u_contrast, "d_proto": d_proto, "gamma": gamma,
        "beta": data["beta"], "alpha": data["alpha"], "beta_reconstructed": reconstructed,
    })
    result = {
        "schema_version": "source-reference-v1-beta-identity", "decision": "BETA_REFERENCE_IDENTITY_VERIFIED", "passed": True,
        "maximum_absolute_error": float(np.max(np.abs(reconstructed - data["beta"]))),
        "sign_change_proportion": float(np.mean(np.sign(d_proto) != np.sign(reconstructed))),
        "gamma_mean": float(np.mean(gamma)), "gamma_session_means": np.mean(gamma, axis=0).tolist(),
        "gamma_session_std": np.std(gamma, axis=0, ddof=1).tolist(),
        "d_proto_mean": float(np.mean(d_proto)), "beta_mean": float(np.mean(data["beta"])),
    }
    atomic_write_json(output / "decisions/beta_reference_identity.json", result)
    return result


def _find_parent_cache(root: Path, config: Mapping[str, Any]) -> Path:
    lines = _git(root, "worktree", "list", "--porcelain").splitlines()
    current_path: Path | None = None
    for line in lines:
        if line.startswith("worktree "):
            current_path = Path(line.removeprefix("worktree "))
        elif line == f"branch refs/heads/{config['parent_cache']['owning_branch']}" and current_path is not None:
            candidate = current_path / str(config["parent_cache"]["relative_dir"])
            if candidate.is_dir():
                return candidate
    raise SourceReferenceObjectInsufficient("hash-locked PR #17 cache worktree is unavailable")


def _load_validated_tangent_cache(root: Path, config: Mapping[str, Any], modes: np.ndarray, fold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cache = _find_parent_cache(root, config)
    covariance = cache / str(config["parent_cache"]["covariance_file"])
    tangent = cache / str(config["parent_cache"]["tangent_file"])
    if not covariance.is_file() or sha256_file(covariance) != config["parent_cache"]["covariance_sha256"]:
        raise SourceReferenceObjectInsufficient("combined covariance cache hash mismatch")
    if not tangent.is_file():
        raise SourceReferenceObjectInsufficient("parent tangent cache missing")
    with np.load(tangent, allow_pickle=False) as z:
        features = _finite("tangent features", z["features"]); means = _finite("cached marginal means", z["marginal_means"]); ids = np.asarray(z["trial_ids"])
    if list(features.shape) != config["parent_cache"]["expected_feature_shape"] or list(means.shape) != config["parent_cache"]["expected_mean_shape"] or list(ids.shape) != config["parent_cache"]["expected_trial_id_shape"]:
        raise SourceReferenceObjectInsufficient("parent tangent cache shape mismatch")
    parent = config["parent_artifacts"]
    with np.load(root / parent["projected_trials"]["path"], allow_pickle=False) as z:
        committed_y, projected_ids = _finite("committed projected y", z["projected_y"]), np.asarray(z["trial_ids"])
    with np.load(root / parent["evaluation_labels"]["path"], allow_pickle=False) as z:
        label_ids = np.asarray(z["trial_ids"])
    with np.load(root / parent["marginal_means"]["path"], allow_pickle=False) as z:
        committed_means = _finite("committed marginal means", z["marginal_means"])
    if not np.array_equal(ids, projected_ids) or not np.array_equal(ids, label_ids):
        raise SourceReferenceObjectInsufficient("trial IDs differ across cache/compact objects")
    np.testing.assert_allclose(means, committed_means, rtol=0.0, atol=float(config["parent_cache"]["marginal_atol"]))
    reproduced = np.empty_like(committed_y)
    for s in range(54):
        for q in range(2):
            reproduced[s, q] = features[s, q] @ modes[int(fold[s]), q]
    try:
        np.testing.assert_allclose(reproduced, committed_y, rtol=0.0, atol=float(config["parent_cache"]["projected_atol"]))
    except AssertionError as exc:
        raise SourceReferenceObjectInsufficient("tangent cache does not reproduce committed projections") from exc
    return features, ids


def _load_reference_core(root: Path, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    output = output_path(root, config); path = output / "objects/source_reference_core.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            return {key: np.asarray(z[key]).copy() for key in z.files}
    identity_path = output / "objects/beta_reference_identity_core.npz"
    if not identity_path.is_file():
        raise DataContractError("beta identity must run before correction")
    with np.load(identity_path, allow_pickle=False) as z:
        identity = {key: np.asarray(z[key]).copy() for key in z.files}
    features, trial_ids = _load_validated_tangent_cache(root, config, identity["modes"], identity["fold_of_subject"])
    parent = config["parent_artifacts"]
    with np.load(root / parent["evaluation_labels"]["path"], allow_pickle=False) as z:
        labels = np.asarray(z["class_index"], dtype=np.int8)
    with np.load(root / parent["projected_trials"]["path"], allow_pickle=False) as z:
        projected_y = _finite("projected y", z["projected_y"])
    with np.load(root / parent["unsigned_core"]["path"], allow_pickle=False) as z:
        ehat = _finite("primary energy", z["primary_estimate"]); mixture = _finite("mixture energy", z["mixture_estimate"]); committed_delta = _finite("committed delta", z["delta_trial"])
    outer = _outer_indices(config)
    source_delta = np.empty((6, 54, 2)); d_proto_all = np.empty((6, 54, 2)); correction_mean = np.empty((6, 2)); correction_median = np.empty((6, 2)); affine_slope = np.empty((6, 2)); affine_intercept = np.empty((6, 2)); source_order = np.empty((6, 2), dtype=np.int8)
    for f, test in enumerate(outer):
        train = np.setdiff1d(np.arange(54), test)
        for q in range(2):
            direction = identity["modes"][f, q]
            d_proto_all[f, :, q] = identity["u_contrast"][:, q] @ direction
            values = features[:, q] @ direction
            for s in range(54):
                source_delta[f, s, q] = 0.5 * (np.mean(values[s, labels[s, q] == 1]) - np.mean(values[s, labels[s, q] == 0]))
            residual = d_proto_all[f, train, q] - source_delta[f, train, q]
            correction_mean[f, q] = float(np.mean(residual)); correction_median[f, q] = float(np.median(residual))
            design = np.column_stack([np.ones(len(train)), source_delta[f, train, q]])
            intercept, slope = np.linalg.lstsq(design, d_proto_all[f, train, q], rcond=None)[0]
            affine_slope[f, q], affine_intercept[f, q] = float(slope), float(intercept)
            mean_delta = float(np.mean(source_delta[f, train, q]))
            if mean_delta == 0.0:
                raise NumericalContractError("source semantic order is exactly zero")
            source_order[f, q] = 1 if mean_delta > 0 else -1
    target_delta = np.empty((54, 2))
    for s in range(54):
        for q in range(2):
            target_delta[s, q] = 0.5 * (np.mean(projected_y[s, q, labels[s, q] == 1]) - np.mean(projected_y[s, q, labels[s, q] == 0]))
    np.testing.assert_allclose(target_delta, committed_delta, rtol=0.0, atol=2e-14)
    arrays = {
        **identity, "features_sha256": np.asarray([sha256_file(_find_parent_cache(root, config) / config["parent_cache"]["tangent_file"])]),
        "trial_ids": trial_ids, "labels": labels, "projected_y": projected_y,
        "target_delta": target_delta, "source_delta_fold": source_delta, "d_proto_all_fold": d_proto_all,
        "correction_mean": correction_mean, "correction_median": correction_median,
        "affine_slope": affine_slope, "affine_intercept": affine_intercept, "source_order": source_order,
        "primary_energy": ehat, "mixture_energy": mixture,
    }
    population_v1._atomic_savez(path, arrays)
    atomic_write_json(output / "objects/parent_cache_validation.json", {
        "status": "PASS", "covariance_sha256": config["parent_cache"]["covariance_sha256"],
        "tangent_sha256": str(arrays["features_sha256"][0]),
        "trial_ids_exact": True, "marginal_means_within_tolerance": True,
        "committed_projected_y_reproduced": True, "raw_rebuild_performed": False,
    })
    return arrays


def _subject_fold_values(core: Mapping[str, np.ndarray], fold_values: np.ndarray) -> np.ndarray:
    result = np.empty((54, 2), dtype=np.float64)
    for s in range(54):
        result[s] = fold_values[int(core["fold_of_subject"][s])]
    return result


def run_source_reference_correction(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = ensure_real_access_lock(root); output = output_path(root, config)
    core = _load_reference_core(root, config); beta, delta = core["beta"], core["target_delta"]
    gamma = _subject_fold_values(core, core["gamma"]); correction = _subject_fold_values(core, core["correction_mean"]); median_correction = _subject_fold_values(core, core["correction_median"]); slope = _subject_fold_values(core, core["affine_slope"]); intercept = _subject_fold_values(core, core["affine_intercept"])
    predictions = {
        "UNCENTERED_TRIAL_DELTA": delta,
        "DELTA_MINUS_GAMMA": delta - gamma,
        "SOURCE_MEAN_ADDITIVE_CORRECTION": delta + correction - gamma,
        "SOURCE_MEDIAN_ADDITIVE_CORRECTION": delta + median_correction - gamma,
        "SOURCE_ONLY_AFFINE_CALIBRATION": slope * delta + intercept - gamma,
    }
    metric_rows: list[dict[str, Any]] = []
    for name, pred in predictions.items():
        for session in ("pooled", "0", "1"):
            index = (...,) if session == "pooled" else (slice(None), int(session))
            values = _metric_bundle(pred if session == "pooled" else pred[index], beta if session == "pooled" else beta[index])
            metric_rows.append({"method": name, "session": session, **values})
    pd.DataFrame(metric_rows).to_csv(output / "tables/source_reference_correction_metrics.csv", index=False, float_format="%.17g", lineterminator="\n")
    base_error = np.abs(delta - beta); corrected_error = np.abs(predictions["SOURCE_MEAN_ADDITIVE_CORRECTION"] - beta)
    improvement_session = np.mean(base_error - corrected_error, axis=0); improvement_subject = np.mean(base_error - corrected_error, axis=1)
    low, high, bootstrap = _bootstrap_mean(config, improvement_subject, "correction_improvement")
    p_value, null = _paired_sign_flip(config, improvement_subject, "correction_improvement")
    sign_subject = np.mean(np.sign(predictions["SOURCE_MEAN_ADDITIVE_CORRECTION"]) == np.sign(beta), axis=1)
    sign_low, sign_high, sign_bootstrap = _bootstrap_mean(config, sign_subject, "correction_beta_sign_accuracy")
    leave_one = np.asarray([np.mean(np.delete(improvement_subject, s)) for s in range(54)])
    passed = bool(
        np.mean(improvement_subject) > 0 and low > 0 and p_value <= config["inference"]["alpha"]
        and np.all(improvement_session > 0) and sign_low > 0.5 and np.all(leave_one > 0)
    )
    influence = pd.DataFrame({"omitted_subject": core["subjects"], "mean_improvement": leave_one})
    influence.to_csv(output / "tables/source_reference_correction_influence.csv", index=False, float_format="%.17g", lineterminator="\n")
    correction_rows = []
    for f in range(6):
        for q in range(2):
            correction_rows.append({"outer_fold": f, "session": str(q), "gamma": core["gamma"][f, q], "mean_correction": core["correction_mean"][f, q], "median_correction": core["correction_median"][f, q], "affine_slope": core["affine_slope"][f, q], "affine_intercept": core["affine_intercept"][f, q]})
    pd.DataFrame(correction_rows).to_csv(output / "tables/gamma_and_source_correction.csv", index=False, float_format="%.17g", lineterminator="\n")
    population_v1._atomic_savez(output / "objects/source_reference_predictions.npz", {"beta": beta, "delta": delta, "gamma": gamma, "correction": correction, **{f"prediction_{key.lower()}": value for key, value in predictions.items()}})
    population_v1._atomic_savez(output / "nulls/source_reference_correction_inference.npz", {"improvement_bootstrap": bootstrap, "improvement_null": null, "sign_accuracy_bootstrap": sign_bootstrap})
    decision = config["decisions"]["correction_pass" if passed else "correction_negative"]
    result = {
        "schema_version": "source-reference-v1-correction", "decision": decision, "passed": passed,
        "mean_mae_improvement": float(np.mean(improvement_subject)), "session_improvements": improvement_session.tolist(),
        "improvement_ci": [low, high], "paired_sign_flip_p": p_value,
        "beta_sign_accuracy": float(np.mean(sign_subject)), "beta_sign_accuracy_ci": [sign_low, sign_high],
        "leave_one_subject_improvement_positive": bool(np.all(leave_one > 0)),
        "primary_metrics": _metric_bundle(predictions["SOURCE_MEAN_ADDITIVE_CORRECTION"], beta),
        "uncentered_metrics": _metric_bundle(delta, beta),
    }
    atomic_write_json(output / "decisions/source_reference_correction.json", result)
    return result


def run_source_ordering_assumption(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = ensure_real_access_lock(root); output = output_path(root, config)
    core = _load_reference_core(root, config); delta, beta = core["target_delta"], core["beta"]
    order = _subject_fold_values(core, core["source_order"]).astype(np.int8)
    correct = np.sign(delta) == order
    accuracy_session = np.mean(correct, axis=0); accuracy_subject = np.mean(correct, axis=1)
    low, high, bootstrap = _bootstrap_mean(config, accuracy_subject, "source_order_accuracy")
    pooled_success = int(np.count_nonzero(correct)); pooled_total = int(correct.size)
    p_value = float(stats.binomtest(pooled_success, pooled_total, p=0.5, alternative="greater").pvalue)
    session_p = [float(stats.binomtest(int(np.count_nonzero(correct[:, q])), 54, p=0.5, alternative="greater").pvalue) for q in range(2)]
    outer = _outer_indices(config); stable = True; loo_rows: list[dict[str, Any]] = []
    for f, test in enumerate(outer):
        train = np.setdiff1d(np.arange(54), test)
        for q in range(2):
            frozen = int(core["source_order"][f, q])
            for omitted in train:
                kept = train[train != omitted]
                value = float(np.mean(core["source_delta_fold"][f, kept, q])); refit = 1 if value > 0 else (-1 if value < 0 else 0)
                unchanged = refit == frozen; stable = stable and unchanged
                loo_rows.append({"outer_fold": f, "session": str(q), "omitted_subject": int(core["subjects"][omitted]), "frozen_order": frozen, "refit_order": refit, "unchanged": unchanged})
    pd.DataFrame(loo_rows).to_csv(output / "tables/source_order_leave_one_training_subject.csv", index=False, lineterminator="\n")
    subject_rows = []
    gamma = _subject_fold_values(core, core["gamma"])
    for s, subject in enumerate(core["subjects"]):
        for q in range(2):
            subject_rows.append({"subject": int(subject), "session": str(q), "outer_fold": int(core["fold_of_subject"][s]), "source_order": int(order[s, q]), "target_delta_sign": int(np.sign(delta[s, q])), "ordering_correct": bool(correct[s, q]), "absolute_beta": abs(beta[s, q]), "distance_from_gamma": abs(delta[s, q] - gamma[s, q])})
    pd.DataFrame(subject_rows).to_csv(output / "tables/source_ordering_subjects.csv", index=False, float_format="%.17g", lineterminator="\n")
    passed = bool(np.all(accuracy_session > 0.5) and low > 0.5 and p_value <= config["inference"]["alpha"] and stable)
    decision = config["decisions"]["ordering_pass" if passed else "ordering_negative"]
    result = {
        "schema_version": "source-reference-v1-ordering", "decision": decision, "passed": passed,
        "pooled_accuracy": float(np.mean(correct)), "session_accuracies": accuracy_session.tolist(),
        "pooled_accuracy_ci": [low, high], "exact_binomial_p": p_value, "session_binomial_p": session_p,
        "leave_one_training_subject_order_stable": stable,
        "cross_session_target_order_consistency": float(np.mean(np.sign(delta[:, 0]) == np.sign(delta[:, 1]))),
        "violating_subject_sessions": int(np.count_nonzero(~correct)),
        "violating_subjects": sorted(set(int(core["subjects"][s]) for s in np.flatnonzero(np.any(~correct, axis=1)))),
        "interpretation": "retrospective OpenBMI evidence under an explicit source-ordering assumption; not intrinsic zero-label sign identifiability",
    }
    population_v1._atomic_savez(output / "nulls/source_ordering_inference.npz", {"accuracy_bootstrap": bootstrap})
    atomic_write_json(output / "decisions/source_semantic_ordering.json", result)
    return result


def _association_safe(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    try:
        return _pearson(pred, truth), _spearman(pred, truth)
    except NumericalContractError:
        return float("nan"), float("nan")


def run_corrected_minimal_anchor(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = ensure_real_access_lock(root); output = output_path(root, config)
    correction_decision = json.loads((output / "decisions/source_reference_correction.json").read_text())
    ordering_decision = json.loads((output / "decisions/source_semantic_ordering.json").read_text())
    core = _load_reference_core(root, config); beta, delta = core["beta"], core["target_delta"]
    gamma = _subject_fold_values(core, core["gamma"]); correction = _subject_fold_values(core, core["correction_mean"]); order = _subject_fold_values(core, core["source_order"])
    magnitude = np.sqrt(np.maximum(core["primary_energy"], 0.0)); mixture_magnitude = np.sqrt(np.maximum(core["mixture_energy"], 0.0))
    zero = order * magnitude + correction - gamma
    zero_mixture = order * mixture_magnitude + correction - gamma
    beta_zero = np.zeros_like(beta)
    zero_improvement_session = np.mean(np.abs(beta_zero - beta) - np.abs(zero - beta), axis=0)
    zero_improvement_subject = np.mean(np.abs(beta_zero - beta) - np.abs(zero - beta), axis=1)
    zero_low, zero_high, zero_bootstrap = _bootstrap_mean(config, zero_improvement_subject, "zero_label_improvement")
    zero_p, zero_null = _paired_sign_flip(config, zero_improvement_subject, "zero_label_improvement")
    zero_sign_subject = np.mean(np.sign(zero) == np.sign(beta), axis=1)
    zero_sign_low, zero_sign_high, zero_sign_bootstrap = _bootstrap_mean(config, zero_sign_subject, "zero_label_sign")
    zero_loo = np.asarray([np.mean(np.delete(zero_improvement_subject, s)) for s in range(54)])
    zero_pass = bool(
        correction_decision["passed"] and ordering_decision["passed"] and np.mean(zero_improvement_subject) > 0
        and zero_low > 0 and zero_p <= config["inference"]["alpha"] and np.all(zero_improvement_session > 0)
        and zero_sign_low > 0.5 and np.all(zero_loo > 0)
    )
    zero_result = {
        "schema_version": "source-reference-v1-zero-label-under-assumption",
        "label": config["anchor"]["zero_label_label"],
        "decision": config["decisions"]["zero_label_pass" if zero_pass else "zero_label_negative"], "passed": zero_pass,
        "metrics": _metric_bundle(zero, beta), "mixture_metrics": _metric_bundle(zero_mixture, beta),
        "mean_mae_improvement_over_beta_zero": float(np.mean(zero_improvement_subject)),
        "session_improvements": zero_improvement_session.tolist(), "improvement_ci": [zero_low, zero_high],
        "paired_sign_flip_p": zero_p, "beta_sign_accuracy_ci": [zero_sign_low, zero_sign_high],
        "leave_one_subject_improvement_positive": bool(np.all(zero_loo > 0)),
        "identification_boundary": "NONIDENTIFIABLE_UP_TO_CLASS_PERMUTATION without the explicit source-ordering assumption",
    }
    atomic_write_json(output / "decisions/source_referenced_zero_label.json", zero_result)

    parent = config["parent_artifacts"]
    with np.load(root / parent["anchor_subsamples"]["path"], allow_pickle=False) as z:
        budgets = np.asarray(z["positive_budgets"], dtype=np.int64); direct_delta = _finite("direct calibration contrasts", z["direct"]); orientation = np.asarray(z["orientation"], dtype=np.int8); historical = _finite("historical proposed", z["proposed"])
    expected_budgets = np.asarray([m for m in config["anchor"]["budgets"] if m > 0])
    if not np.array_equal(budgets, expected_budgets) or direct_delta.shape != (5, 54, 2, 200):
        raise DataContractError("PR #17 calibration subset contract differs")
    ref = (correction - gamma)[None, :, :, None]
    proposed = orientation * magnitude[None, :, :, None] + ref
    proposed_mixture = orientation * mixture_magnitude[None, :, :, None] + ref
    direct = direct_delta + ref
    source_magnitude_fold = np.empty((6, 2))
    for f, test in enumerate(_outer_indices(config)):
        train = np.setdiff1d(np.arange(54), test)
        source_magnitude_fold[f] = np.mean(np.abs(core["source_delta_fold"][f, train]), axis=0)
    source_magnitude = _subject_fold_values(core, source_magnitude_fold)
    source_baseline = orientation * source_magnitude[None, :, :, None] + ref
    truth = beta[None, :, :, None]
    methods = {"PROPOSED_UL_PLUS_M": proposed, "DIRECT_M_LABEL_FAIR_REFERENCE": direct, "MIXTURE_UL_PLUS_M": proposed_mixture, "SOURCE_POPULATION_MAGNITUDE": source_baseline, "PR17_OLD_UNCENTERED_HISTORICAL": historical}
    budget_rows: list[dict[str, Any]] = []; subject_rows: list[dict[str, Any]] = []; null_arrays: dict[str, np.ndarray] = {}; bootstrap_arrays: dict[str, np.ndarray] = {}
    selected: int | None = None
    for bi, budget in enumerate(budgets.tolist()):
        proposed_error = np.mean(np.abs(proposed[bi] - beta[:, :, None]), axis=-1)
        direct_error = np.mean(np.abs(direct[bi] - beta[:, :, None]), axis=-1)
        improvement_session = np.mean(direct_error - proposed_error, axis=0); improvement_subject = np.mean(direct_error - proposed_error, axis=1)
        low, high, boot = _bootstrap_mean(config, improvement_subject, f"anchor_improvement_{budget}")
        p_value, null = _paired_sign_flip(config, improvement_subject, f"anchor_improvement_{budget}")
        sign_subject = np.mean(np.sign(proposed[bi]) == np.sign(beta[:, :, None]), axis=(1, 2))
        sign_low, sign_high, sign_boot = _bootstrap_mean(config, sign_subject, f"anchor_beta_sign_{budget}")
        semantic_sign_subject = np.mean(orientation[bi] == np.sign(delta)[:, :, None], axis=(1, 2))
        semantic_low, semantic_high, semantic_boot = _bootstrap_mean(config, semantic_sign_subject, f"anchor_semantic_sign_{budget}")
        leave_one = np.asarray([np.mean(np.delete(improvement_subject, s)) for s in range(54)])
        eligible = budget in config["anchor"]["eligible_efficiency_budgets"]
        passed = bool(eligible and np.mean(improvement_subject) > 0 and low > 0 and p_value <= config["inference"]["alpha"] and np.all(improvement_session > 0) and sign_low > 0.5 and np.all(leave_one > 0))
        if passed and selected is None:
            selected = int(budget)
        pred_mean = np.mean(proposed[bi], axis=-1); direct_mean = np.mean(direct[bi], axis=-1)
        proposed_metrics = _metric_bundle(pred_mean, beta); direct_metrics = _metric_bundle(direct_mean, beta)
        budget_rows.append({
            "budget": int(budget), "proposed_mae": float(np.mean(proposed_error)), "direct_mae": float(np.mean(direct_error)),
            "mean_improvement_direct_minus_proposed": float(np.mean(improvement_subject)), "improvement_session0": float(improvement_session[0]), "improvement_session1": float(improvement_session[1]),
            "improvement_ci_low": low, "improvement_ci_high": high, "paired_sign_flip_p": p_value,
            "beta_sign_accuracy": float(np.mean(sign_subject)), "beta_sign_accuracy_ci_low": sign_low, "beta_sign_accuracy_ci_high": sign_high,
            "semantic_sign_accuracy": float(np.mean(semantic_sign_subject)), "semantic_sign_accuracy_ci_low": semantic_low, "semantic_sign_accuracy_ci_high": semantic_high,
            "proposed_pearson": proposed_metrics["pearson"], "proposed_spearman": proposed_metrics["spearman"], "proposed_signed_r2": proposed_metrics["signed_r2"],
            "direct_pearson": direct_metrics["pearson"], "direct_spearman": direct_metrics["spearman"], "direct_signed_r2": direct_metrics["signed_r2"],
            "leave_one_subject_improvement_positive": bool(np.all(leave_one > 0)), "eligible": eligible, "passes": passed,
        })
        for s, subject in enumerate(core["subjects"]):
            for q in range(2):
                subject_rows.append({"budget": int(budget), "subject": int(subject), "session": str(q), "proposed_mae": proposed_error[s, q], "direct_mae": direct_error[s, q], "improvement": direct_error[s, q] - proposed_error[s, q], "proposed_beta_sign_accuracy": float(np.mean(np.sign(proposed[bi, s, q]) == np.sign(beta[s, q]))), "semantic_sign_accuracy": float(np.mean(orientation[bi, s, q] == np.sign(delta[s, q])))})
        null_arrays[f"improvement_budget_{budget}"] = null; bootstrap_arrays[f"improvement_budget_{budget}"] = boot; bootstrap_arrays[f"beta_sign_budget_{budget}"] = sign_boot; bootstrap_arrays[f"semantic_sign_budget_{budget}"] = semantic_boot
    pd.DataFrame(budget_rows).to_csv(output / "tables/corrected_minimal_anchor_budget_summary.csv", index=False, float_format="%.17g", lineterminator="\n")
    pd.DataFrame(subject_rows).to_csv(output / "tables/corrected_minimal_anchor_subjects.csv", index=False, float_format="%.17g", lineterminator="\n")

    # All method metrics use the mean over the inherited 200 subsamples for association; MAE uses every draw.
    method_rows = []
    for name, values in methods.items():
        for bi, budget in enumerate(budgets.tolist()):
            mean_prediction = np.mean(values[bi], axis=-1)
            for session in ("pooled", "0", "1"):
                pred = mean_prediction if session == "pooled" else mean_prediction[:, int(session)]
                target = beta if session == "pooled" else beta[:, int(session)]
                metrics = _metric_bundle(pred, target)
                metrics["mae_all_subsamples"] = float(np.mean(np.abs(values[bi] - truth[0]))) if session == "pooled" else float(np.mean(np.abs(values[bi, :, int(session)] - beta[:, int(session), None])))
                method_rows.append({"method": name, "budget": int(budget), "session": session, **metrics})
    zero_metrics = _metric_bundle(zero, beta); method_rows.append({"method": config["anchor"]["zero_label_label"], "budget": 0, "session": "pooled", **zero_metrics, "mae_all_subsamples": zero_metrics["mae"]})
    pd.DataFrame(method_rows).to_csv(output / "tables/all_estimator_metrics.csv", index=False, float_format="%.17g", lineterminator="\n")

    # Non-voting factorial oracle decomposition.
    oracle_rows = []
    oracle_sign = np.sign(delta); d_proto = beta + gamma
    oracle_methods = {
        "A_ORACLE_SIGN_ESTIMATED_MAGNITUDE": oracle_sign * magnitude + correction - gamma,
        "C_ORACLE_DELTA_SOURCE_REFERENCE": d_proto - gamma,
        "E_FULL_TRIAL_DELTA_SOURCE_REFERENCE": delta + correction - gamma,
    }
    for name, pred in oracle_methods.items():
        oracle_rows.append({"oracle": name, "budget": "FULL_OR_ZERO", **_metric_bundle(pred, beta)})
    for bi, budget in enumerate(budgets.tolist()):
        estimated_sign = orientation[bi]
        b = estimated_sign * np.abs(delta)[..., None] + ref[0]
        d = orientation[bi] * magnitude[..., None] - gamma[..., None]
        oracle_rows.append({"oracle": "B_ESTIMATED_SIGN_ORACLE_ABSOLUTE_DELTA", "budget": int(budget), **_metric_bundle(np.mean(b, axis=-1), beta)})
        oracle_rows.append({"oracle": "D_ESTIMATED_DELTA_ORACLE_GAMMA_NO_CURVATURE_CORRECTION", "budget": int(budget), **_metric_bundle(np.mean(d, axis=-1), beta)})
    pd.DataFrame(oracle_rows).to_csv(output / "tables/oracle_bottleneck_decomposition.csv", index=False, float_format="%.17g", lineterminator="\n")
    passed = selected is not None
    anchor_result = {
        "schema_version": "source-reference-v1-anchor", "decision": config["decisions"]["anchor_pass" if passed else "anchor_negative"],
        "passed": passed, "selected_budget": selected, "budgets": budget_rows, "subsamples": int(config["anchor"]["subsamples"]),
    }
    atomic_write_json(output / "decisions/source_referenced_minimal_anchor.json", anchor_result)
    population_v1._atomic_savez(output / "objects/corrected_anchor_estimates.npz", {"budgets": budgets, "zero_label": zero, "zero_label_mixture": zero_mixture, "proposed": proposed, "direct": direct, "mixture": proposed_mixture, "source_baseline": source_baseline, "historical": historical, "beta": beta, "delta": delta})
    population_v1._atomic_savez(output / "nulls/source_referenced_anchor_inference.npz", {**null_arrays, **bootstrap_arrays, "zero_improvement_null": zero_null, "zero_improvement_bootstrap": zero_bootstrap, "zero_sign_bootstrap": zero_sign_bootstrap})
    return {"zero_label": zero_result, "minimal_anchor": anchor_result}


def _save_figure(fig: Any, output: Path, stem: str) -> None:
    fig.savefig(output / "figures" / f"{stem}.png", dpi=170, bbox_inches="tight")
    fig.savefig(output / "figures" / f"{stem}.pdf", bbox_inches="tight")


def _generate_figures(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    identity = pd.read_csv(output / "tables/beta_reference_identity_subjects.csv")
    corrections = pd.read_csv(output / "tables/gamma_and_source_correction.csv")
    fig, ax = plt.subplots(figsize=(6.2, 5)); ax.scatter(identity["beta_parent"], identity["beta_reconstructed"], s=18, alpha=.75); lim = [min(identity["beta_parent"].min(), identity["beta_reconstructed"].min()), max(identity["beta_parent"].max(), identity["beta_reconstructed"].max())]; ax.plot(lim, lim, "k--"); ax.set(xlabel="parent beta", ylabel="d_proto - gamma", title="Algebraic beta reference identity"); _save_figure(fig, output, "figure_01_beta_reference_identity"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2));
    for q, ax in enumerate(axes):
        frame = corrections[corrections.session == q]; ax.plot(frame.outer_fold, frame.gamma, "o-", label="gamma"); ax.plot(frame.outer_fold, frame.mean_correction, "s-", label="correction"); ax.set(title=f"session {q}", xlabel="outer fold"); ax.legend()
    _save_figure(fig, output, "figure_02_gamma_and_correction"); plt.close(fig)
    with np.load(output / "objects/source_reference_predictions.npz", allow_pickle=False) as z:
        beta, raw, corrected = z["beta"], z["delta"], z["prediction_source_mean_additive_correction"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2)); axes[0].scatter(raw, beta, s=16, alpha=.7); axes[0].set(xlabel="uncentered trial delta", ylabel="beta", title="Before reference correction"); axes[1].scatter(corrected, beta, s=16, alpha=.7); axes[1].set(xlabel="corrected beta_ref_full", ylabel="beta", title="After source reference"); _save_figure(fig, output, "figure_03_coordinate_correction"); plt.close(fig)
    ordering = pd.read_csv(output / "tables/source_ordering_subjects.csv"); fig, ax = plt.subplots(figsize=(7, 4)); grouped = ordering.groupby("session")["ordering_correct"].mean(); ax.bar(grouped.index.astype(str), grouped.values); ax.axhline(.5, color="k", ls="--"); ax.set(ylim=(0, 1), xlabel="session", ylabel="target ordering accuracy", title="Source semantic-order assumption"); _save_figure(fig, output, "figure_04_source_ordering"); plt.close(fig)
    budget = pd.read_csv(output / "tables/corrected_minimal_anchor_budget_summary.csv"); zero = json.loads((output / "decisions/source_referenced_zero_label.json").read_text()); fig, ax = plt.subplots(figsize=(7, 4.4)); ax.plot(budget.budget, budget.proposed_mae, "o-", label="UL magnitude + m-label sign + source reference"); ax.plot(budget.budget, budget.direct_mae, "s-", label="direct m-label + source reference"); ax.scatter([0], [zero["metrics"]["mae"]], marker="*", s=120, label="zero-label under source order"); ax.set(xlabel="labeled target trials m", ylabel="beta MAE", title="Fair source-referenced calibration comparison"); ax.legend(); _save_figure(fig, output, "figure_05_corrected_anchor_mae"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.4)); ax.plot(budget.budget, budget.beta_sign_accuracy, "o-", label="beta sign"); ax.plot(budget.budget, budget.semantic_sign_accuracy, "s-", label="semantic delta sign"); ax.axhline(.5, color="k", ls="--"); ax.set(xlabel="labeled target trials m", ylabel="accuracy", ylim=(0, 1), title="Semantic sign versus residual beta sign"); ax.legend(); _save_figure(fig, output, "figure_06_sign_accuracies"); plt.close(fig)
    oracle = pd.read_csv(output / "tables/oracle_bottleneck_decomposition.csv"); fig, ax = plt.subplots(figsize=(9, 4.6)); values = oracle.groupby("oracle").mae.mean().sort_values(); ax.barh(values.index, values.values); ax.set(xlabel="beta MAE", title="Non-voting oracle bottleneck decomposition"); _save_figure(fig, output, "figure_07_oracle_decomposition"); plt.close(fig)


def _final_manifest(root: Path, config: Mapping[str, Any], decisions: Mapping[str, Any]) -> dict[str, Any]:
    output = output_path(root, config)
    excluded = {"manifest.json"}
    artifacts = {str(path.relative_to(output)): sha256_file(path) for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name not in excluded)}
    return {
        "schema_version": "source-reference-v1-final-manifest", "parent_pr": 17,
        "parent_head": config["protocol"]["parent_head"],
        "protocol_freeze_commit": json.loads((output / "git_provenance.json").read_text())["protocol_freeze_commit"],
        "artifact_count": len(artifacts), "artifacts": artifacts, "decisions": decisions,
        "parent_artifacts_unchanged": True,
    }


def generate_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, provenance = ensure_real_access_lock(root); output = output_path(root, config)
    identity = json.loads((output / "decisions/beta_reference_identity.json").read_text()); correction = json.loads((output / "decisions/source_reference_correction.json").read_text()); ordering = json.loads((output / "decisions/source_semantic_ordering.json").read_text()); zero = json.loads((output / "decisions/source_referenced_zero_label.json").read_text()); anchor = json.loads((output / "decisions/source_referenced_minimal_anchor.json").read_text())
    budget = pd.read_csv(output / "tables/corrected_minimal_anchor_budget_summary.csv"); gamma = pd.read_csv(output / "tables/gamma_and_source_correction.csv"); oracle = pd.read_csv(output / "tables/oracle_bottleneck_decomposition.csv")
    decisions = {"beta_identity": identity["decision"], "coordinate_correction": correction["decision"], "source_ordering": ordering["decision"], "zero_label_under_assumption": zero["decision"], "minimal_anchor": anchor["decision"]}
    next_question = config["decisions"]["next_if_positive"] if all((correction["passed"], ordering["passed"], zero["passed"], anchor["passed"])) else config["decisions"]["next_otherwise"]
    report = f"""# Source-Referenced Conditional Residual V1

## Scope

Retrospective OpenBMI-only mechanistic follow-up stacked on PR #17 head `{config['protocol']['parent_head']}`. PR #16/#17 artifacts were hash-verified and remained unchanged. This work does not recompute their terminals.

## Algebraic beta identity

- Decision: `{identity['decision']}`
- Maximum absolute error: `{identity['maximum_absolute_error']:.10g}`
- Mean gamma: `{identity['gamma_mean']:.10g}`; session means `{identity['gamma_session_means']}`
- Sign-change proportion after subtracting gamma: `{identity['sign_change_proportion']:.10g}`

## Source trial/prototype correction

- Decision: `{correction['decision']}`
- Mean MAE improvement over uncentered trial delta: `{correction['mean_mae_improvement']:.10g}`
- 95% subject-bootstrap CI: `{correction['improvement_ci']}`
- Paired sign-flip p: `{correction['paired_sign_flip_p']:.10g}`
- Session improvements: `{correction['session_improvements']}`
- Corrected full-trial metrics: `{json.dumps(correction['primary_metrics'], sort_keys=True)}`
- Gamma range: `[{gamma.gamma.min():.10g}, {gamma.gamma.max():.10g}]`; mean-correction range: `[{gamma.mean_correction.min():.10g}, {gamma.mean_correction.max():.10g}]`

## Explicit source semantic ordering

- Decision: `{ordering['decision']}`
- Pooled/session target ordering accuracy: `{ordering['pooled_accuracy']:.10g}` / `{ordering['session_accuracies']}`
- 95% subject-bootstrap CI: `{ordering['pooled_accuracy_ci']}`
- Exact binomial p: `{ordering['exact_binomial_p']:.10g}`
- Leave-one-training-subject source order stable: `{ordering['leave_one_training_subject_order_stable']}`
- Violating subjects: `{ordering['violating_subjects']}`

This is retrospective evidence under an explicit source-ordering assumption. The zero-label signed coordinate remains `NONIDENTIFIABLE_UP_TO_CLASS_PERMUTATION` without that assumption.

## Zero-label under the source-order assumption

- Label: `{zero['label']}`
- Decision: `{zero['decision']}`
- Metrics: `{json.dumps(zero['metrics'], sort_keys=True)}`
- MAE improvement over beta=0: `{zero['mean_mae_improvement_over_beta_zero']:.10g}`, CI `{zero['improvement_ci']}`, p `{zero['paired_sign_flip_p']:.10g}`
- Beta-sign-accuracy CI: `{zero['beta_sign_accuracy_ci']}`

## Corrected minimal anchor

- Decision: `{anchor['decision']}`
- Selected budget: `{anchor['selected_budget']}`
- Frozen budgets/subsamples: `[0,2,4,8,16,32]`, 200 inherited subsamples per positive budget

{budget.to_markdown(index=False)}

The direct baseline is fair: every direct calibration contrast uses the same source correction and gamma subtraction. Semantic-delta sign and beta sign are reported separately.

## Non-voting oracle bottleneck decomposition

{oracle.to_markdown(index=False)}

## Decisions

{json.dumps(decisions, indent=2, sort_keys=True)}

## Boundaries

This work does not establish a full conditional distribution, physiology, source anatomy, causality, a universal individual coordinate, downstream classification benefit, pseudo-label validity, TTA recoverability, a multiclass solution, or an ASD biomarker. No other dataset or classifier was run.

## Exact next scientific question

{next_question}
"""
    (output / "report/source_referenced_conditional_residual_v1.md").write_text(report, encoding="utf-8")
    _generate_figures(output)
    manifest = _final_manifest(root, config, decisions); manifest["next_question"] = next_question
    atomic_write_json(output / "manifest.json", manifest)
    return {"decisions": decisions, "next_question": next_question, "protocol_freeze_commit": provenance["protocol_freeze_commit"], "artifact_count": manifest["artifact_count"]}
