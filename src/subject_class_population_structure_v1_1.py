"""Technical-recovery orchestration for Subject-Class Population Structure V1.1.

All scientific computation delegates to the immutable V1 implementation and V1
config.  This module changes only output isolation and optional-control failure
boundaries.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import src.subject_class_population_structure_v1 as v1
from src.interaction_provenance_v0 import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_array,
    sha256_file,
)


CONFIG_PATH = "configs/subject_class_population_structure_v1_1.yaml"
OUTPUT_NAME = "subject_class_population_structure_v1_1"
AMENDMENT_PATH = "docs/TECHNICAL_AMENDMENT_SUBJECT_CLASS_POPULATION_STRUCTURE_V1_1.md"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_amendment_config(repo_root: str | Path) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    root = Path(repo_root).resolve()
    path = root / CONFIG_PATH
    amendment = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(amendment, dict):
        raise v1.DataContractError("V1.1 amendment config must be a mapping")
    amendment_path = root / str(amendment["amendment"]["amendment_path"])
    if sha256_file(amendment_path) != str(amendment["amendment"]["amendment_sha256"]):
        raise v1.DataContractError("V1.1 amendment document hash mismatch")
    v1_path = root / str(amendment["amendment"]["v1_config_path"])
    v1_hash = sha256_file(v1_path)
    if v1_hash != str(amendment["amendment"]["v1_config_sha256"]):
        raise v1.DataContractError("immutable V1 config hash mismatch")
    scientific, loaded_hash = v1.load_config(root)
    if loaded_hash != v1_hash:
        raise v1.DataContractError("V1 loader and V1.1 config disagree on scientific config")
    return amendment, sha256_file(path), scientific, v1_hash


def output_path(root: Path, amendment: Mapping[str, Any]) -> Path:
    return root / str(amendment["amendment"]["output_dir"])


def _ensure_directories(output: Path) -> None:
    for name in ("protocol", "objects", "nulls", "tables", "figures", "controls", "decisions", "report"):
        (output / name).mkdir(parents=True, exist_ok=True)


def validate_immutable_v1_history(root: Path, amendment: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, record in amendment["immutable_v1_artifacts"].items():
        path = root / str(record["path"])
        if not path.is_file():
            raise v1.DataContractError(f"missing immutable V1 artifact: {name}")
        digest = sha256_file(path)
        if digest != str(record["sha256"]):
            raise v1.DataContractError(f"immutable V1 artifact changed: {name}")
        observed[str(name)] = digest
    failed_head = str(amendment["amendment"]["base_failed_result_commit"])
    if _git(root, "merge-base", "--is-ancestor", failed_head, "HEAD") != "":
        raise v1.DataContractError("V1 failed-result commit is not an ancestor")
    return observed


def scientific_equivalence_rows(
    scientific: Mapping[str, Any], amendment: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected = amendment["scientific_contract_sha256"]
    rows: list[dict[str, Any]] = []
    labels = {
        "parent_artifacts": "base parent objects and hashes",
        "datasets": "subject/session/class/channel order and feature source",
        "features": "svec, normalization, and feature rules",
        "openbmi_folds": "outer and inner folds",
        "rank": "rank grids, one-SE selection, and low-rank cap",
        "nulls": "null count, seeds/namespaces, mappings, and selection rerun",
        "bootstrap": "bootstrap count, confidence, and seed namespace",
        "reliability": "measurement-reliability gates",
        "controls": "voting/non-voting declarations",
        "terminal_labels": "terminal labels",
    }
    for key, label in labels.items():
        digest = _mapping_sha(scientific[key])
        if digest != str(expected[key]):
            raise v1.DataContractError(f"V1/V1.1 scientific contract mismatch: {key}")
        rows.append({
            "field": label, "v1": digest, "v1_1": digest,
            "changed": False, "change_class": "SCIENTIFIC",
        })
    terminal_digest = hashlib.sha256(inspect.getsource(v1.terminal_decision).encode("utf-8")).hexdigest()
    if terminal_digest != str(expected["terminal_decision_source"]):
        raise v1.DataContractError("V1 terminal-decision source changed")
    rows.append({
        "field": "terminal decision function", "v1": terminal_digest,
        "v1_1": terminal_digest, "changed": False, "change_class": "SCIENTIFIC",
    })
    rows.append({
        "field": "secondary failure isolation", "v1": "global abort",
        "v1_1": "independent non-voting exception boundaries", "changed": True,
        "change_class": "ORCHESTRATION_ONLY",
    })
    if [row for row in rows if row["changed"]] != [rows[-1]]:
        raise v1.DataContractError("equivalence table permits only orchestration isolation")
    return rows


def _scientific_file_hashes(root: Path) -> dict[str, str]:
    paths = [
        CONFIG_PATH,
        AMENDMENT_PATH,
        "src/subject_class_population_structure_v1_1.py",
        "tests/test_subject_class_population_structure_v1_1.py",
    ]
    paths.extend(f"scripts/{number}_{name}.py" for number, name in (
        (66, "freeze_subject_class_structure_v1_1"),
        (67, "run_reliability_gate_v1_1"),
        (68, "run_openbmi_primary_v1_1"),
        (69, "run_openbmi_nulls_v1_1"),
        (70, "run_openbmi_controls_v1_1"),
        (71, "run_bnci_diagnostic_v1_1"),
        (72, "report_subject_class_structure_v1_1"),
    ))
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise v1.DataContractError(f"missing V1.1 freeze-scope files: {missing}")
    return {path: sha256_file(root / path) for path in paths}


def run_nonvoting_control(name: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"control": name, "status": "CONTROL_COMPLETED", "result": operation()}
    except v1.NumericalContractError as error:
        return {
            "control": name, "status": "CONTROL_UNASSESSED_NUMERICAL_DEGENERACY",
            "exception": type(error).__name__, "message": str(error),
        }
    except v1.DataContractError as error:
        return {
            "control": name, "status": "CONTROL_UNASSESSED_DATA_CONTRACT_FAILURE",
            "exception": type(error).__name__, "message": str(error),
        }
    except Exception as error:  # optional diagnostic isolation is deliberately broad
        return {
            "control": name, "status": "CONTROL_UNASSESSED_EXECUTION_FAILURE",
            "exception": type(error).__name__, "message": str(error),
        }


def run_voting_component(name: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"component": name, "status": "VOTING_COMPLETED", "result": operation()}
    except (v1.PopulationStructureError, np.linalg.LinAlgError, FloatingPointError) as error:
        return {
            "component": name,
            "status": "UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE",
            "exception": type(error).__name__, "message": str(error),
        }


def run_synthetic_recovery_gates(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    amendment, amendment_hash, _, _ = load_amendment_config(root)
    rows: list[dict[str, Any]] = []

    primary_a = run_voting_component("sensor_primary", lambda: {"statistic": 0.5})
    nulls_a = run_voting_component("primary_nulls", lambda: {"pairing_p": 0.01})
    spectrum_a = run_nonvoting_control(
        "ordered_z_eigenspectrum",
        lambda: (_ for _ in ()).throw(v1.NumericalContractError("synthetic projected scale degeneracy")),
    )
    terminal_a = v1.terminal_decision(
        data_contract_pass=True, reliability_pass=True, statistic=0.5,
        forward_median=0.4, reverse_median=0.4, pairing_p=0.01, class_p=0.01,
        random_p=0.01, influence_positive=True, full_space_stable=True,
        selected_ranks=[2] * 6, low_cap=8,
    )
    rows.append({
        "case": "A_primary_valid_spectrum_degenerate",
        "expected": "PRIMARY_AND_NULLS_COMPLETE_SPECTRUM_UNASSESSED_TERMINAL_COMPUTED",
        "observed": f"{primary_a['status']}|{nulls_a['status']}|{spectrum_a['status']}|{terminal_a}",
        "passed": primary_a["status"] == "VOTING_COMPLETED"
        and nulls_a["status"] == "VOTING_COMPLETED"
        and spectrum_a["status"] == "CONTROL_UNASSESSED_NUMERICAL_DEGENERACY"
        and terminal_a == "GO_POPULATION_STRUCTURED_LOW_DIMENSIONAL_INTERACTION",
    })

    primary_b = run_voting_component(
        "sensor_primary",
        lambda: (_ for _ in ()).throw(v1.NumericalContractError("synthetic primary degeneracy")),
    )
    rows.append({
        "case": "B_primary_sensor_degenerate",
        "expected": "UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE",
        "observed": primary_b["status"],
        "passed": primary_b["status"] == "UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE",
    })

    generalized_c = run_nonvoting_control(
        "generalized_eigen_signature",
        lambda: (_ for _ in ()).throw(v1.DataContractError("synthetic generalized-eigen failure")),
    )
    rows.append({
        "case": "C_generalized_eigen_failure",
        "expected": "CONTROL_UNASSESSED_DATA_CONTRACT_FAILURE_PRIMARY_UNAFFECTED",
        "observed": generalized_c["status"],
        "passed": generalized_c["status"] == "CONTROL_UNASSESSED_DATA_CONTRACT_FAILURE"
        and primary_a["status"] == "VOTING_COMPLETED",
    })

    retained_terminal = "STOP_RANDOM_SUBSPACE_EQUIVALENT"
    bnci_d = run_nonvoting_control(
        "bnci_multiclass",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic BNCI failure")),
    )
    rows.append({
        "case": "D_bnci_failure_after_terminal",
        "expected": "OPENBMI_TERMINAL_RETAINED_BNCI_UNASSESSED",
        "observed": f"{retained_terminal}|{bnci_d['status']}",
        "passed": retained_terminal == "STOP_RANDOM_SUBSPACE_EQUIVALENT"
        and bnci_d["status"] == "CONTROL_UNASSESSED_EXECUTION_FAILURE",
    })

    passed = bool(all(row["passed"] for row in rows))
    output = output_path(root, amendment)
    (output / "protocol").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output / "protocol/synthetic_recovery_gates.csv", index=False,
        lineterminator="\n",
    )
    result = {
        "schema_version": "subject-class-population-structure-v1-1-synthetic",
        "passed": passed, "cases": rows, "amendment_config_sha256": amendment_hash,
    }
    atomic_write_json(output / "protocol/synthetic_recovery_gates.json", result)
    if not passed:
        raise v1.NumericalContractError("V1.1 synthetic recovery gates failed")
    return result


def freeze_amendment(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    amendment, amendment_hash, scientific, scientific_hash = load_amendment_config(root)
    historical = validate_immutable_v1_history(root, amendment)
    parent_hashes = v1.validate_parent_hashes(root, scientific)
    openbmi = v1.load_parent_dataset(root, scientific, "openbmi")
    v1.load_parent_dataset(root, scientific, "bnci")
    v1.openbmi_fold_indices(openbmi, scientific)
    v1.validate_helmert(scientific)
    equivalence = scientific_equivalence_rows(scientific, amendment)
    synthetic = run_synthetic_recovery_gates(root)
    output = output_path(root, amendment)
    _ensure_directories(output)
    pd.DataFrame(equivalence).to_csv(
        output / "protocol/v1_v1_1_scientific_contract_equivalence.csv",
        index=False, lineterminator="\n",
    )
    for source_name in (AMENDMENT_PATH, CONFIG_PATH):
        source = root / source_name
        atomic_write_bytes(output / "protocol" / source.name, source.read_bytes())
    files = _scientific_file_hashes(root)
    manifest = {
        "schema_version": "subject-class-population-structure-v1-1-freeze",
        "status": "READY_FOR_TECHNICAL_AMENDMENT_FREEZE_COMMIT",
        "failed_v1_head": str(amendment["amendment"]["base_failed_result_commit"]),
        "branch": str(amendment["amendment"]["branch"]),
        "amendment_config_sha256": amendment_hash,
        "scientific_v1_config_sha256": scientific_hash,
        "parent_hashes": parent_hashes, "immutable_v1_hashes": historical,
        "scientific_file_hashes": files,
        "equivalence_pass": True, "synthetic_pass": bool(synthetic["passed"]),
        "real_data_accessed": False,
        "required_commit_subject": str(amendment["amendment"]["required_freeze_commit_subject"]),
    }
    atomic_write_json(output / "protocol/manifest.json", manifest)
    atomic_write_json(output / "environment.json", v1.environment_record())
    return manifest


def ensure_amendment_real_access_lock(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    amendment, amendment_hash, scientific, scientific_hash = load_amendment_config(root)
    validate_immutable_v1_history(root, amendment)
    v1.validate_parent_hashes(root, scientific)
    scientific_equivalence_rows(scientific, amendment)
    output = output_path(root, amendment)
    manifest_path = output / "protocol/manifest.json"
    if not manifest_path.is_file():
        raise v1.DataContractError("V1.1 freeze manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["amendment_config_sha256"] != amendment_hash:
        raise v1.DataContractError("V1.1 amendment config changed after freeze")
    if manifest["scientific_v1_config_sha256"] != scientific_hash:
        raise v1.DataContractError("V1 scientific config identity changed")
    if not manifest["equivalence_pass"] or not manifest["synthetic_pass"]:
        raise v1.DataContractError("V1.1 freeze gates did not pass")
    current_files = _scientific_file_hashes(root)
    if current_files != manifest["scientific_file_hashes"]:
        raise v1.DataContractError("V1.1 source/config changed after freeze")
    provenance_path = output / "git_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if _git(root, "merge-base", "--is-ancestor", provenance["technical_amendment_commit"], "HEAD") != "":
            raise v1.DataContractError("V1.1 freeze ancestry failure")
        return provenance
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise v1.DataContractError(f"first V1.1 real access requires clean tree: {status.splitlines()[:5]}")
    head = _git(root, "rev-parse", "HEAD")
    subject = _git(root, "show", "-s", "--format=%s", "HEAD")
    if subject != manifest["required_commit_subject"]:
        raise v1.DataContractError("HEAD is not the V1.1 technical-amendment freeze commit")
    provenance = {
        "schema_version": "subject-class-population-structure-v1-1-git-provenance",
        "failed_v1_result_commit": str(amendment["amendment"]["base_failed_result_commit"]),
        "technical_amendment_commit": head,
        "branch": _git(root, "branch", "--show-current"),
        "first_real_access_tree_was_clean": True,
        "scientific_file_hashes": current_files,
    }
    atomic_write_json(provenance_path, provenance)
    return provenance


def _record_voting_failure(
    output: Path, component: str, error: BaseException,
    amendment_hash: str, scientific_hash: str,
) -> None:
    atomic_write_json(output / "decisions/terminal_decision.json", {
        "schema_version": "subject-class-population-structure-v1-1-terminal",
        "terminal_decision": "UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE",
        "failed_voting_component": component,
        "exception": type(error).__name__, "message": str(error),
        "amendment_config_sha256": amendment_hash,
        "scientific_v1_config_sha256": scientific_hash,
    })


def run_reliability_gate(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_amendment_real_access_lock(root)
    amendment, amendment_hash, scientific, scientific_hash = load_amendment_config(root)
    output = output_path(root, amendment)
    _ensure_directories(output)
    try:
        parent_hashes = v1.validate_parent_hashes(root, scientific)
        data = v1.load_parent_dataset(root, scientific, "openbmi")
        observed = v1.evaluate_openbmi_reliability(data, scientific)
        replicates = int(scientific["nulls"]["replicates"])
        mappings = v1.generate_reliability_mappings(data, scientific, replicates)
        null = np.empty((replicates, 2), dtype=np.float64)
        for replicate in range(replicates):
            value = v1.evaluate_openbmi_reliability(data, scientific, mappings=mappings[replicate])
            null[replicate] = [value["session_statistics"][session] for session in data.sessions]
        rows = observed["rows"]
        summaries: list[dict[str, Any]] = []
        influences: list[dict[str, Any]] = []
        passes: list[bool] = []
        for session_index, session in enumerate(data.sessions):
            statistic = float(observed["session_statistics"][session])
            p_value = v1.monte_carlo_p(statistic, null[:, session_index])
            values = rows.loc[rows["session"] == session].sort_values("subject")["delta_average"].to_numpy()
            influence = v1.leave_one_subject_influence(values)
            influence_pass = bool(np.all(influence > 0.0))
            passed = statistic > 0.0 and p_value <= float(scientific["nulls"]["alpha"]) and influence_pass
            passes.append(passed)
            summaries.append({
                "session": session, "observed": statistic,
                "null_median": float(np.median(null[:, session_index])),
                "effect": statistic - float(np.median(null[:, session_index])),
                "p_value": p_value, "leave_one_subject_sign_pass": influence_pass,
                "passed": passed,
            })
            influences.extend({
                "session": session, "omitted_subject": subject,
                "median_without_subject": float(value),
            } for subject, value in zip(data.subjects, influence))
        passed = bool(all(passes))
        rows.to_csv(output / "tables/reliability_subject_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
        pd.DataFrame(summaries).to_csv(output / "tables/reliability_statistics.csv", index=False, lineterminator="\n", float_format="%.17g")
        pd.DataFrame(influences).to_csv(output / "tables/reliability_influence.csv", index=False, lineterminator="\n", float_format="%.17g")
        v1._atomic_savez(output / "nulls/reliability_pairing_null.npz", {
            "statistics": null, "mappings": mappings,
            "subjects": np.asarray(data.subjects), "sessions": np.asarray(data.sessions),
        })
        result = {
            "schema_version": "subject-class-population-structure-v1-1-reliability",
            "passed": passed, "sessions": summaries, "replicates": replicates,
            "terminal_if_failed": None if passed else "UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE",
            "amendment_config_sha256": amendment_hash,
            "scientific_v1_config_sha256": scientific_hash,
            "parent_hashes": parent_hashes,
        }
        atomic_write_json(output / "decisions/reliability_gate.json", result)
        if not passed:
            atomic_write_json(output / "decisions/terminal_decision.json", {
                "schema_version": "subject-class-population-structure-v1-1-terminal",
                "terminal_decision": "UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE",
                "reliability_sessions": summaries,
            })
        return result
    except (v1.PopulationStructureError, np.linalg.LinAlgError, FloatingPointError) as error:
        _record_voting_failure(output, "reliability", error, amendment_hash, scientific_hash)
        raise


def _serialize_primary(output: Path, data: v1.DatasetObjects, scientific: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    fits: Sequence[v1.TwoViewFit] = observed["fits"]
    arrays: dict[str, np.ndarray] = {
        "subjects": np.asarray(data.subjects, dtype=np.int64),
        "selected_ranks": np.asarray(observed["selected_ranks"], dtype=np.int64),
        "subject_delta": observed["subject_rows"].sort_values("subject")["delta_average"].to_numpy(),
        "subject_delta_forward": observed["subject_rows"].sort_values("subject")["delta_forward"].to_numpy(),
        "subject_delta_reverse": observed["subject_rows"].sort_values("subject")["delta_reverse"].to_numpy(),
        "full_subject_delta": observed["full_space_subject_rows"].sort_values("subject")["delta_average"].to_numpy(),
        "mean0": np.stack([fit.mean0 for fit in fits]),
        "mean1": np.stack([fit.mean1 for fit in fits]),
        "left": np.stack([fit.left for fit in fits]),
        "right": np.stack([fit.right for fit in fits]),
        "singular_values": np.stack([fit.singular_values for fit in fits]),
        "scale0": np.stack([fit.scale0 for fit in fits]),
        "scale1": np.stack([fit.scale1 for fit in fits]),
    }
    n = len(data.subjects)
    similarity = np.zeros((n, n), dtype=np.float64)
    mask = np.zeros((n, n), dtype=np.uint8)
    outer, _ = v1.openbmi_fold_indices(data, scientific)
    for fold, block in zip(outer, observed["similarity_blocks"]):
        similarity[np.ix_(fold, fold)] = block
        mask[np.ix_(fold, fold)] = 1
    arrays["heldout_similarity_block_matrix"] = similarity
    arrays["heldout_similarity_mask"] = mask
    v1._atomic_savez(output / "objects/openbmi_observed_core.npz", arrays)


def run_openbmi_primary(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_amendment_real_access_lock(root)
    amendment, amendment_hash, scientific, scientific_hash = load_amendment_config(root)
    output = output_path(root, amendment)
    _ensure_directories(output)
    try:
        reliability_path = output / "decisions/reliability_gate.json"
        if not reliability_path.is_file():
            raise v1.DataContractError("V1.1 reliability gate must precede primary")
        reliability = json.loads(reliability_path.read_text(encoding="utf-8"))
        if not reliability["passed"]:
            raise v1.DataContractError("UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE")
        parent_hashes = v1.validate_parent_hashes(root, scientific)
        data = v1.load_parent_dataset(root, scientific, "openbmi")
        observed = v1.evaluate_openbmi_nested(data, scientific, kind="sensor", keep_fits=True)
        values = observed["subject_rows"].sort_values("subject")["delta_average"].to_numpy()
        low, high, bootstrap = v1.bootstrap_median_ci(values, scientific)
        influence = v1.leave_one_subject_influence(values)
        observed["subject_rows"].to_csv(output / "tables/openbmi_subject_directional_separation.csv", index=False, lineterminator="\n", float_format="%.17g")
        observed["rank_rows"].to_csv(output / "tables/openbmi_rank_by_rank_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
        fold_rows = []
        membership = []
        for row in observed["fold_rows"]:
            fold_rows.append({
                key: value for key, value in row.items()
                if key not in {"audit", "singular_values", "inner_means", "inner_standard_errors", "train_subjects", "test_subjects"}
            } | {
                "train_subjects": "|".join(map(str, row["train_subjects"])),
                "test_subjects": "|".join(map(str, row["test_subjects"])),
                "r_over_p": float(row["selected_rank"] / 210.0),
                "r_over_n_train_minus_1": float(row["selected_rank"] / 44.0),
            })
            membership.extend({"outer_fold": row["outer_fold"], "role": "train", "subject": subject} for subject in row["train_subjects"])
            membership.extend({"outer_fold": row["outer_fold"], "role": "test", "subject": subject} for subject in row["test_subjects"])
        pd.DataFrame(fold_rows).to_csv(output / "tables/openbmi_outer_fold_selected_ranks.csv", index=False, lineterminator="\n", float_format="%.17g")
        pd.DataFrame(membership).to_csv(output / "tables/outer_fold_membership.csv", index=False, lineterminator="\n")
        pd.DataFrame({"subject": data.subjects, "median_without_subject": influence}).to_csv(
            output / "tables/openbmi_leave_one_subject_influence.csv", index=False,
            lineterminator="\n", float_format="%.17g",
        )
        v1._atomic_savez(output / "objects/openbmi_bootstrap.npz", {"statistics": bootstrap})
        _serialize_primary(output, data, scientific, observed)
        result = {
            "schema_version": "subject-class-population-structure-v1-1-openbmi-primary",
            "statistic": float(observed["statistic"]),
            "forward_median": float(observed["forward_median"]),
            "reverse_median": float(observed["reverse_median"]),
            "bootstrap_ci_95": [low, high],
            "selected_ranks": observed["selected_ranks"].tolist(),
            "median_selected_rank": float(np.median(observed["selected_ranks"])),
            "rank_frequency": {str(int(rank)): int(count) for rank, count in zip(*np.unique(observed["selected_ranks"], return_counts=True))},
            "folds_at_or_below_low_cap": int(np.count_nonzero(observed["selected_ranks"] <= 8)),
            "full_space_statistic": float(observed["full_space_statistic"]),
            "influence_sign_pass": bool(np.all(influence > 0.0)),
            "parent_hashes": parent_hashes,
            "amendment_config_sha256": amendment_hash,
            "scientific_v1_config_sha256": scientific_hash,
        }
        atomic_write_json(output / "decisions/openbmi_observed.json", result)
        return result
    except (v1.PopulationStructureError, np.linalg.LinAlgError, FloatingPointError) as error:
        _record_voting_failure(output, "openbmi_sensor_primary", error, amendment_hash, scientific_hash)
        raise


def run_openbmi_nulls(repo_root: str | Path, *, workers: int | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_amendment_real_access_lock(root)
    amendment, amendment_hash, scientific, scientific_hash = load_amendment_config(root)
    output = output_path(root, amendment)
    _ensure_directories(output)
    try:
        reliability = json.loads((output / "decisions/reliability_gate.json").read_text(encoding="utf-8"))
        if not reliability["passed"]:
            raise v1.DataContractError("UNASSESSED_MEASUREMENT_RELIABILITY_FAILURE")
        observed_path = output / "decisions/openbmi_observed.json"
        core_path = output / "objects/openbmi_observed_core.npz"
        if not observed_path.is_file() or not core_path.is_file():
            raise v1.DataContractError("V1.1 primary observed artifacts must precede nulls")
        observed = json.loads(observed_path.read_text(encoding="utf-8"))
        observed_hash = sha256_file(core_path)
        with np.load(core_path, allow_pickle=False) as archive:
            selected_ranks = np.asarray(archive["selected_ranks"], dtype=np.int64)
        parent_hashes = v1.validate_parent_hashes(root, scientific)
        data = v1.load_parent_dataset(root, scientific, "openbmi")
        replicates = int(scientific["nulls"]["replicates"])
        pairing_outer, pairing_inner = v1.generate_openbmi_pairing_mappings(data, scientific, replicates)
        class_mappings = v1.generate_openbmi_class_mappings(data, scientific, replicates)
        v1._atomic_savez(output / "nulls/openbmi_pairing_mappings.npz", {
            "outer_mappings": pairing_outer, "inner_mappings": pairing_inner,
            "subjects": np.asarray(data.subjects),
        })
        v1._atomic_savez(output / "nulls/openbmi_class_mappings.npz", {
            "class_mappings": class_mappings, "subjects": np.asarray(data.subjects),
            "sessions": np.asarray(data.sessions), "classes": np.asarray(data.classes),
        })
        atomic_write_json(output / "nulls/seed_manifest.json", {
            "master_seed": int(scientific["protocol"]["master_seed"]),
            "bit_generator": "PCG64DXSM",
            "derivation": "canonical JSON -> SHA256 -> first 128 bits as little-endian uint32 words",
            "pairing_outer_sha256": sha256_array(pairing_outer),
            "pairing_inner_sha256": sha256_array(pairing_inner),
            "class_mappings_sha256": sha256_array(class_mappings),
            "random_mapping": "keyed by replicate/fold/view; bases not persisted",
            "identical_to_v1_namespaces": True,
        })
        v1._initialize_null_worker(
            data, scientific, pairing_outer, pairing_inner, class_mappings, selected_ranks
        )
        worker_count = int(scientific["resources"]["workers"] if workers is None else workers)
        batch = int(scientific["resources"]["checkpoint_batch"])
        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for kind in ("pairing", "class", "random"):
            results[kind] = v1._execute_null_kind(
                kind, output, config_hash=amendment_hash,
                observed_hash=observed_hash, replicates=replicates,
                workers=worker_count, batch=batch,
            )
        pairing, pairing_ranks = results["pairing"]
        class_null, class_ranks = results["class"]
        random, _ = results["random"]
        pairing_p = v1.monte_carlo_p(float(observed["statistic"]), pairing[:, 0])
        class_p = v1.monte_carlo_p(float(observed["statistic"]), class_null[:, 0])
        random_p = v1.monte_carlo_p(float(observed["statistic"]), random[:, 0])
        full_pairing_p = v1.monte_carlo_p(float(observed["full_space_statistic"]), pairing[:, 3])
        full_space_stable = float(observed["full_space_statistic"]) > 0.0 and full_pairing_p <= 0.05
        decision = v1.terminal_decision(
            data_contract_pass=True, reliability_pass=True,
            statistic=float(observed["statistic"]),
            forward_median=float(observed["forward_median"]),
            reverse_median=float(observed["reverse_median"]),
            pairing_p=pairing_p, class_p=class_p, random_p=random_p,
            influence_positive=bool(observed["influence_sign_pass"]),
            full_space_stable=full_space_stable,
            selected_ranks=observed["selected_ranks"], low_cap=8,
        )
        summaries = [
            {"null": "subject_pairing", "observed": observed["statistic"], "null_median": float(np.median(pairing[:, 0])), "p_value": pairing_p, "replicates": replicates, "reran_rank_selection": True},
            {"null": "class_semantics", "observed": observed["statistic"], "null_median": float(np.median(class_null[:, 0])), "p_value": class_p, "replicates": replicates, "reran_rank_selection": True},
            {"null": "equal_rank_random_subspace", "observed": observed["statistic"], "null_median": float(np.median(random[:, 0])), "p_value": random_p, "replicates": replicates, "reran_rank_selection": False},
            {"null": "full_space_subject_pairing", "observed": observed["full_space_statistic"], "null_median": float(np.median(pairing[:, 3])), "p_value": full_pairing_p, "replicates": replicates, "reran_rank_selection": True},
        ]
        pd.DataFrame(summaries).to_csv(
            output / "tables/openbmi_primary_null_summaries.csv", index=False,
            lineterminator="\n", float_format="%.17g",
        )
        rank_rows: list[dict[str, Any]] = []
        for kind, values in (("pairing", pairing_ranks), ("class", class_ranks)):
            for fold in range(values.shape[1]):
                unique, counts = np.unique(values[:, fold], return_counts=True)
                rank_rows.extend({
                    "null": kind, "outer_fold": fold,
                    "rank": int(rank), "count": int(count),
                } for rank, count in zip(unique, counts))
        pd.DataFrame(rank_rows).to_csv(
            output / "tables/openbmi_null_rank_selection_frequency.csv",
            index=False, lineterminator="\n",
        )
        result = {
            "schema_version": "subject-class-population-structure-v1-1-terminal",
            "terminal_decision": decision,
            "observed": float(observed["statistic"]),
            "pairing_p": pairing_p, "class_p": class_p,
            "random_subspace_p": random_p,
            "full_space_pairing_p": full_pairing_p,
            "full_space_stable": full_space_stable,
            "selected_ranks": observed["selected_ranks"],
            "openbmi_terminal_frozen_before_secondary_controls": True,
            "parent_hashes": parent_hashes,
            "observed_core_sha256": observed_hash,
            "amendment_config_sha256": amendment_hash,
            "scientific_v1_config_sha256": scientific_hash,
        }
        atomic_write_json(output / "decisions/terminal_decision.json", result)
        return result
    except (v1.PopulationStructureError, np.linalg.LinAlgError, FloatingPointError) as error:
        _record_voting_failure(output, "openbmi_primary_nulls", error, amendment_hash, scientific_hash)
        raise


def _nested_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "statistic": float(value["statistic"]),
        "forward_median": float(value["forward_median"]),
        "reverse_median": float(value["reverse_median"]),
        "full_space_statistic": float(value["full_space_statistic"]),
        "selected_ranks": np.asarray(value["selected_ranks"], dtype=int).tolist(),
    }


def _paired_fit_diagnostic(
    x_train: np.ndarray, requested_rank: int, location: Mapping[str, Any]
) -> dict[str, Any]:
    a = np.asarray(x_train[:, 0], dtype=np.float64)
    b = np.asarray(x_train[:, 1], dtype=np.float64)
    centered0 = a - np.mean(a, axis=0)
    centered1 = b - np.mean(b, axis=0)
    u0, s0, vh0 = np.linalg.svd(centered0, full_matrices=False)
    u1, s1, vh1 = np.linalg.svd(centered1, full_matrices=False)
    middle = (s0[:, None] * (u0.T @ u1) * s1[None, :]) / float(len(a) - 1)
    p, singular, qh = np.linalg.svd(middle, full_matrices=False)
    left = vh0.T @ p[:, :requested_rank]
    right = vh1.T @ qh.T[:, :requested_rank]
    scale0 = np.std(centered0 @ left, axis=0, ddof=1)
    scale1 = np.std(centered1 @ right, axis=0, ddof=1)
    threshold = float(np.finfo(float).eps * np.sqrt(a.shape[1]))
    degenerate0 = np.flatnonzero(scale0 <= threshold)
    degenerate1 = np.flatnonzero(scale1 <= threshold)
    cross = centered0.T @ centered1 / float(len(a) - 1)
    return {
        **dict(location), "n_train": int(len(a)), "feature_dimension": int(a.shape[1]),
        "requested_rank": int(requested_rank),
        "centered_session0_numerical_rank": int(np.linalg.matrix_rank(centered0)),
        "centered_session1_numerical_rank": int(np.linalg.matrix_rank(centered1)),
        "cross_covariance_numerical_rank": int(np.linalg.matrix_rank(cross)),
        "cross_covariance_singular_values": np.linalg.svd(cross, compute_uv=False).tolist(),
        "compact_singular_values": singular[:requested_rank].tolist(),
        "projected_score_std_session0": scale0.tolist(),
        "projected_score_std_session1": scale1.tolist(),
        "degenerate_coordinates_session0_zero_based": degenerate0.tolist(),
        "degenerate_coordinates_session1_zero_based": degenerate1.tolist(),
        "machine_epsilon": float(np.finfo(float).eps),
        "epsilon_relative_threshold": threshold,
        "minimum_std_over_epsilon_session0": float(np.min(scale0) / np.finfo(float).eps),
        "minimum_std_over_epsilon_session1": float(np.min(scale1) / np.finfo(float).eps),
        "minimum_std_over_threshold_session0": float(np.min(scale0) / threshold),
        "minimum_std_over_threshold_session1": float(np.min(scale1) / threshold),
        "exact_zero": bool(np.any(scale0 == 0.0) or np.any(scale1 == 0.0)),
        "near_or_exact_degeneracy": bool(len(degenerate0) or len(degenerate1)),
    }


def diagnose_ordered_z_spectrum(
    data: v1.DatasetObjects, scientific: Mapping[str, Any]
) -> dict[str, Any]:
    outer, inner = v1.openbmi_fold_indices(data, scientific)
    all_indices = np.arange(len(data.subjects), dtype=np.int64)
    records: list[dict[str, Any]] = []
    for outer_index, test in enumerate(outer):
        outer_train = np.setdiff1d(all_indices, test, assume_unique=True)
        for inner_index, validation in enumerate(inner[outer_index]):
            train = np.setdiff1d(outer_train, validation, assume_unique=True)
            ranks = v1._rank_grid(
                scientific, "openbmi", len(outer_train) - len(validation),
                v1._feature_dimension(data, "spectrum"),
            )
            x_train, _, _ = v1.fold_features(
                data, "F", train, validation, helmert=None, kind="spectrum"
            )
            record = _paired_fit_diagnostic(x_train, max(ranks), {
                "phase": "inner", "outer_fold": outer_index,
                "inner_fold": inner_index,
                "training_subjects": [int(data.subjects[index]) for index in train],
                "validation_subjects": [int(data.subjects[index]) for index in validation],
            })
            if record["near_or_exact_degeneracy"]:
                records.append(record)
        ranks = v1._rank_grid(
            scientific, "openbmi", len(outer_train),
            v1._feature_dimension(data, "spectrum"),
        )
        x_train, _, _ = v1.fold_features(
            data, "F", outer_train, test, helmert=None, kind="spectrum"
        )
        record = _paired_fit_diagnostic(x_train, max(ranks), {
            "phase": "outer", "outer_fold": outer_index, "inner_fold": None,
            "training_subjects": [int(data.subjects[index]) for index in outer_train],
            "validation_subjects": [int(data.subjects[index]) for index in test],
        })
        if record["near_or_exact_degeneracy"]:
            records.append(record)

    train = np.setdiff1d(all_indices, outer[0], assume_unique=True)
    z_train, _, _ = v1.reconstruct_fold_z(
        data.U["F"], data.proportions["F"], train, outer[0]
    )
    z_sum = z_train[:, :, 0] + z_train[:, :, 1]
    z_denominator = np.maximum(
        np.linalg.norm(z_train[:, :, 0], axis=(-2, -1))
        + np.linalg.norm(z_train[:, :, 1], axis=(-2, -1)),
        np.finfo(float).tiny,
    )
    z_opposition_residual = float(np.max(
        np.linalg.norm(z_sum, axis=(-2, -1)) / z_denominator
    ))
    eigenvalues = np.linalg.eigvalsh(z_train)
    reverse_relation = eigenvalues[:, :, 0] + eigenvalues[:, :, 1, ::-1]
    eigen_denominator = np.maximum(
        np.linalg.norm(eigenvalues[:, :, 0], axis=-1)
        + np.linalg.norm(eigenvalues[:, :, 1], axis=-1),
        np.finfo(float).tiny,
    )
    eigen_opposition_residual = float(np.max(
        np.linalg.norm(reverse_relation, axis=-1) / eigen_denominator
    ))
    raw_signature = 0.5 * (eigenvalues[:, :, 1] - eigenvalues[:, :, 0])
    palindrome_residual = float(np.max(np.abs(raw_signature - raw_signature[..., ::-1])))
    tolerance = 2e-13
    return {
        "schema_version": "ordered-z-spectrum-degeneracy-v1-1",
        "affected_fit_count": len(records), "affected_fits": records,
        "binary_z_opposition_relative_residual": z_opposition_residual,
        "ordered_eigen_opposition_relative_residual": eigen_opposition_residual,
        "binary_spectrum_signature_palindrome_max_abs_residual": palindrome_residual,
        "follows_from_binary_z_algebra_and_ordered_eigen_symmetry": bool(
            z_opposition_residual <= tolerance
            and eigen_opposition_residual <= tolerance
            and palindrome_residual <= tolerance
        ),
        "interpretation": "DIAGNOSTIC_ONLY_NO_REGULARIZATION_OR_WORKAROUND_APPLIED",
    }


def run_openbmi_controls(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_amendment_real_access_lock(root)
    amendment, amendment_hash, scientific, scientific_hash = load_amendment_config(root)
    output = output_path(root, amendment)
    _ensure_directories(output)
    terminal_path = output / "decisions/terminal_decision.json"
    if not terminal_path.is_file():
        raise v1.DataContractError("OpenBMI terminal must be frozen before optional controls")
    terminal_before = sha256_file(terminal_path)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal["terminal_decision"].startswith("UNASSESSED_"):
        raise v1.DataContractError("optional controls cannot run after an unassessed voting terminal")
    observed = json.loads((output / "decisions/openbmi_observed.json").read_text(encoding="utf-8"))
    data = v1.load_parent_dataset(root, scientific, "openbmi")
    selected_ranks = observed["selected_ranks"]
    statuses: list[dict[str, Any]] = []

    def nested_control(name: str, kind: str) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            value = v1.evaluate_openbmi_nested(data, scientific, kind=kind)
            value["subject_rows"].to_csv(
                output / f"tables/openbmi_{name}_subject_scores.csv", index=False,
                lineterminator="\n", float_format="%.17g",
            )
            value["rank_rows"].to_csv(
                output / f"tables/openbmi_{name}_rank_scores.csv", index=False,
                lineterminator="\n", float_format="%.17g",
            )
            return _nested_summary(value)
        return run_nonvoting_control(name, operation)

    statuses.append(nested_control("magnitude_sensitivity", "magnitude"))

    def pca_operation() -> dict[str, Any]:
        frame = v1.same_session_pca_control(data, scientific, selected_ranks)
        frame.to_csv(
            output / "tables/openbmi_same_session_pca_subject_scores.csv",
            index=False, lineterminator="\n", float_format="%.17g",
        )
        return {"statistic": float(np.median(frame["delta_average"])), "selected_ranks": selected_ranks}
    statuses.append(run_nonvoting_control("same_session_pca", pca_operation))

    spectrum = nested_control("ordered_z_eigenspectrum", "spectrum")
    if spectrum["status"] == "CONTROL_UNASSESSED_NUMERICAL_DEGENERACY":
        diagnosis = run_nonvoting_control(
            "ordered_z_eigenspectrum_degeneracy_diagnosis",
            lambda: diagnose_ordered_z_spectrum(data, scientific),
        )
        spectrum["degeneracy_diagnosis"] = diagnosis
        if diagnosis["status"] == "CONTROL_COMPLETED":
            atomic_write_json(
                output / "controls/ordered_z_spectrum_degeneracy.json",
                diagnosis["result"],
            )
    statuses.append(spectrum)
    statuses.append(nested_control("generalized_eigen_signature", "generalized_eigen"))

    def selected_mode_operation() -> dict[str, Any]:
        sensor = v1.evaluate_openbmi_nested(data, scientific, kind="sensor", keep_fits=True)
        split_half, latent = v1.selected_mode_split_half(data, scientific, sensor)
        split_half.to_csv(
            output / "tables/openbmi_selected_mode_split_half.csv", index=False,
            lineterminator="\n", float_format="%.17g",
        )
        latent.to_csv(
            output / "tables/openbmi_latent_mode1_scores.csv", index=False,
            lineterminator="\n", float_format="%.17g",
        )
        return {
            "session0_median": float(np.median(split_half.loc[split_half["session"].astype(str) == "0", "delta_average"])),
            "session1_median": float(np.median(split_half.loc[split_half["session"].astype(str) == "1", "delta_average"])),
        }
    statuses.append(run_nonvoting_control("selected_mode_split_half", selected_mode_operation))
    statuses.append({
        "control": "action_overlap",
        "status": "CONTROL_UNASSESSED_DATA_CONTRACT_FAILURE",
        "exception": "CrossDatasetIdentityContract",
        "message": "PR #4 action gains are BNCI-subject quantities and cannot be associated with OpenBMI subjects; BNCI association is attempted separately.",
    })

    for status in statuses:
        atomic_write_json(output / f"controls/{status['control']}.json", status)
    rows = []
    for status in statuses:
        result = status.get("result", {})
        rows.append({
            "control": status["control"], "status": status["status"],
            "statistic": result.get("statistic", "UNASSESSED"),
            "forward_median": result.get("forward_median", "UNASSESSED"),
            "reverse_median": result.get("reverse_median", "UNASSESSED"),
            "selected_ranks": "|".join(map(str, result.get("selected_ranks", []))),
            "votes_primary": False,
            "message": status.get("message", ""),
        })
    pd.DataFrame(rows).to_csv(
        output / "tables/openbmi_optional_control_status.csv",
        index=False, lineterminator="\n", float_format="%.17g",
    )
    if sha256_file(terminal_path) != terminal_before:
        raise v1.DataContractError("optional control changed the frozen OpenBMI terminal")
    result = {
        "schema_version": "subject-class-population-structure-v1-1-controls",
        "openbmi_terminal": terminal["terminal_decision"],
        "openbmi_terminal_sha256_before_after": terminal_before,
        "statuses": statuses,
        "amendment_config_sha256": amendment_hash,
        "scientific_v1_config_sha256": scientific_hash,
    }
    atomic_write_json(output / "decisions/openbmi_controls.json", result)
    return result


def _bnci_checkpoint(
    path: Path, *, kind: str, amendment_hash: str, terminal_hash: str,
    statistics: np.ndarray, ranks: np.ndarray, completed: np.ndarray,
) -> None:
    metadata = np.asarray(json.dumps({
        "kind": kind, "amendment_config_sha256": amendment_hash,
        "openbmi_terminal_sha256": terminal_hash,
        "replicates": len(completed),
    }, sort_keys=True, separators=(",", ":")))
    v1._atomic_savez(path, {
        "statistics": statistics, "selected_ranks": ranks,
        "completed": completed.astype(np.uint8), "metadata_json": metadata,
    })


def _execute_bnci_null_kind(
    kind: str, output: Path, *, amendment_hash: str, terminal_hash: str,
    replicates: int, workers: int, batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    import multiprocessing as mp
    path = output / f"nulls/bnci_{kind}_null.npz"
    expected_metadata = {
        "kind": kind, "amendment_config_sha256": amendment_hash,
        "openbmi_terminal_sha256": terminal_hash, "replicates": replicates,
    }
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            if metadata != expected_metadata:
                raise v1.DataContractError(f"BNCI {kind} checkpoint identity mismatch")
            statistics = np.asarray(archive["statistics"], dtype=np.float64)
            ranks = np.asarray(archive["selected_ranks"], dtype=np.int8)
            completed = np.asarray(archive["completed"], dtype=np.uint8).astype(bool)
    else:
        statistics = np.zeros((replicates, 4), dtype=np.float64)
        ranks = np.zeros((replicates, 9), dtype=np.int8)
        completed = np.zeros(replicates, dtype=bool)
    if statistics.shape != (replicates, 4) or ranks.shape != (replicates, 9) or completed.shape != (replicates,):
        raise v1.DataContractError(f"BNCI {kind} checkpoint shape mismatch")
    pending = np.flatnonzero(~completed).tolist()
    if not pending:
        return statistics, ranks
    tasks = [(kind, int(index)) for index in pending]
    if workers == 1:
        iterator: Iterable[tuple[int, np.ndarray, np.ndarray]] = map(v1._bnci_null_worker, tasks)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(workers)
        iterator = pool.imap(v1._bnci_null_worker, tasks, chunksize=1)
    try:
        since_save = 0
        for replicate, values, selected in iterator:
            if not np.isfinite(values).all():
                raise v1.NumericalContractError(f"BNCI {kind} null {replicate} is nonfinite")
            statistics[replicate] = values
            ranks[replicate] = selected
            completed[replicate] = True
            since_save += 1
            if since_save >= batch:
                _bnci_checkpoint(
                    path, kind=kind, amendment_hash=amendment_hash,
                    terminal_hash=terminal_hash, statistics=statistics,
                    ranks=ranks, completed=completed,
                )
                since_save = 0
        _bnci_checkpoint(
            path, kind=kind, amendment_hash=amendment_hash,
            terminal_hash=terminal_hash, statistics=statistics,
            ranks=ranks, completed=completed,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if not np.all(completed):
        raise v1.NumericalContractError(f"BNCI {kind} null incomplete")
    return statistics, ranks


def run_bnci_diagnostic(repo_root: str | Path, *, workers: int | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_amendment_real_access_lock(root)
    amendment, amendment_hash, scientific, scientific_hash = load_amendment_config(root)
    output = output_path(root, amendment)
    _ensure_directories(output)
    terminal_path = output / "decisions/terminal_decision.json"
    if not terminal_path.is_file():
        raise v1.DataContractError("OpenBMI terminal must precede BNCI")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal["terminal_decision"].startswith("UNASSESSED_"):
        raise v1.DataContractError("BNCI cannot rescue an unassessed OpenBMI voting pipeline")
    terminal_hash = sha256_file(terminal_path)

    def operation() -> dict[str, Any]:
        data = v1.load_parent_dataset(root, scientific, "bnci")
        observed = v1.evaluate_bnci_nested(data, scientific, keep_fits=True)
        replicates = int(scientific["nulls"]["replicates"])
        pairing = v1.generate_bnci_pairing_mappings(data, scientific, replicates)
        classes = v1.generate_bnci_class_mappings(data, scientific, replicates)
        v1._atomic_savez(output / "nulls/bnci_mappings.npz", {
            "pairing": pairing, "classes": classes,
            "subjects": np.asarray(data.subjects),
        })
        v1._initialize_bnci_worker(data, scientific, pairing, classes)
        worker_count = int(scientific["resources"]["workers"] if workers is None else workers)
        batch = int(scientific["resources"]["checkpoint_batch"])
        pairing_null, pairing_ranks = _execute_bnci_null_kind(
            "pairing", output, amendment_hash=amendment_hash,
            terminal_hash=terminal_hash, replicates=replicates,
            workers=worker_count, batch=batch,
        )
        class_null, class_ranks = _execute_bnci_null_kind(
            "class", output, amendment_hash=amendment_hash,
            terminal_hash=terminal_hash, replicates=replicates,
            workers=worker_count, batch=batch,
        )
        pairing_p = v1.monte_carlo_p(observed["statistic"], pairing_null[:, 0])
        class_p = v1.monte_carlo_p(observed["statistic"], class_null[:, 0])
        influence = v1.leave_one_subject_influence(observed["subject_rows"]["delta_average"])
        class_loadings, pair_loadings = v1.bnci_mode_loading_rows(data, scientific, observed)
        observed["subject_rows"].to_csv(output / "tables/bnci_subject_directional_separation.csv", index=False, lineterminator="\n", float_format="%.17g")
        observed["rank_rows"].to_csv(output / "tables/bnci_rank_by_rank_scores.csv", index=False, lineterminator="\n", float_format="%.17g")
        observed["fold_rows"].to_csv(output / "tables/bnci_selected_ranks.csv", index=False, lineterminator="\n", float_format="%.17g")
        class_loadings.to_csv(output / "tables/bnci_class_mode_loadings.csv", index=False, lineterminator="\n", float_format="%.17g")
        pair_loadings.to_csv(output / "tables/bnci_class_pair_mode_loadings.csv", index=False, lineterminator="\n", float_format="%.17g")
        pd.DataFrame({
            "subject": data.subjects,
            "median_without_subject": influence,
        }).to_csv(output / "tables/bnci_leave_one_subject_influence.csv", index=False, lineterminator="\n", float_format="%.17g")
        rank_frequency_rows = []
        for kind, values in (("pairing", pairing_ranks), ("class", class_ranks)):
            unique, counts = np.unique(values, return_counts=True)
            rank_frequency_rows.extend({"null": kind, "rank": int(rank), "count": int(count)} for rank, count in zip(unique, counts))
        pd.DataFrame(rank_frequency_rows).to_csv(output / "tables/bnci_null_rank_frequency.csv", index=False, lineterminator="\n")

        def action_operation() -> dict[str, Any]:
            action = pd.read_csv(root / str(scientific["parent_artifacts"]["action_stage_b"]["path"]))
            action = action.rename(columns={"target": "subject", "subject_median_gain": "action_stage_b_gain"})
            association = observed["coordinates"].merge(action, on="subject", validate="one_to_one")
            norm_rho, norm_p = stats.spearmanr(association["latent_coordinate_norm"], association["action_stage_b_gain"])
            discrepancy_rho, discrepancy_p = stats.spearmanr(association["latent_session_discrepancy"], association["action_stage_b_gain"])
            association["action_successful"] = association["action_stage_b_gain"] > 0.0
            association.to_csv(output / "tables/bnci_action_score_overlap.csv", index=False, lineterminator="\n", float_format="%.17g")
            association_rows = [
                {"association": "latent_coordinate_norm_vs_stage_b_gain", "spearman_rho": float(norm_rho), "p_value_two_sided_descriptive": float(norm_p)},
                {"association": "latent_session_discrepancy_vs_stage_b_gain", "spearman_rho": float(discrepancy_rho), "p_value_two_sided_descriptive": float(discrepancy_p)},
            ]
            pd.DataFrame(association_rows).to_csv(output / "tables/bnci_action_overlap_associations.csv", index=False, lineterminator="\n", float_format="%.17g")
            return {
                "associations": association_rows,
                "action_successful_coordinate_norm_median": (
                    float(association.loc[association["action_successful"], "latent_coordinate_norm"].median())
                    if bool(association["action_successful"].any()) else "NOT_AVAILABLE"
                ),
                "action_unsuccessful_coordinate_norm_median": (
                    float(association.loc[~association["action_successful"], "latent_coordinate_norm"].median())
                    if bool((~association["action_successful"]).any()) else "NOT_AVAILABLE"
                ),
            }
        action = run_nonvoting_control("action_overlap", action_operation)
        atomic_write_json(output / "controls/bnci_action_overlap.json", action)
        result = {
            "statistic": float(observed["statistic"]),
            "forward_median": float(observed["forward_median"]),
            "reverse_median": float(observed["reverse_median"]),
            "selected_ranks": observed["selected_ranks"].tolist(),
            "full_space_statistic": float(observed["full_space_statistic"]),
            "pairing_p": pairing_p, "class_p": class_p,
            "influence_sign_pass": bool(np.all(influence > 0.0)),
            "action_overlap": action,
            "interpretation": "SECONDARY_MULTI_CLASS_DIAGNOSTIC_ONLY",
        }
        return result

    status = run_nonvoting_control("bnci_multiclass", operation)
    if sha256_file(terminal_path) != terminal_hash:
        raise v1.DataContractError("BNCI changed the frozen OpenBMI terminal")
    result = {
        "schema_version": "subject-class-population-structure-v1-1-bnci",
        "executed": status["status"] == "CONTROL_COMPLETED",
        "status": status["status"],
        "openbmi_terminal_retained": terminal["terminal_decision"],
        "openbmi_terminal_sha256_before_after": terminal_hash,
        "result": status.get("result"),
        "exception": status.get("exception"), "message": status.get("message"),
        "amendment_config_sha256": amendment_hash,
        "scientific_v1_config_sha256": scientific_hash,
    }
    atomic_write_json(output / "decisions/bnci_diagnostic.json", result)
    return result


def terminal_next_question(terminal: str) -> str:
    if terminal.startswith("GO_"):
        return "Can an unseen subject's coordinates in the stable interaction subspace be identified from unlabeled marginal EEG without reliable pseudo-labels?"
    if terminal.startswith("STOP_"):
        return "What additional supervision or physiological anchor is required when stable individual interaction does not admit a transferable population-shared low-rank representation?"
    if terminal.startswith("UNASSESSED_"):
        return "The population-structure hypothesis remains unassessed; the technical/data-contract blocker must be resolved before a scientific next question is selected."
    raise v1.DataContractError(f"unknown terminal prefix: {terminal}")


def bnci_report_sentence(bnci: Mapping[str, Any]) -> str:
    result = bnci.get("result") or {}
    if bnci.get("executed"):
        action_status = (result.get("action_overlap") or {}).get("status", "UNASSESSED_ACTION_OVERLAP")
        return (
            f"Executed: `True`; status `CONTROL_COMPLETED`; T=`{result['statistic']:.6f}`, "
            f"pairing p=`{result['pairing_p']:.6f}`, class p=`{result['class_p']:.6f}`, "
            f"ranks=`{result['selected_ranks']}`, influence sign=`{result['influence_sign_pass']}`, "
            f"action overlap=`{action_status}`."
        )
    return (
        f"Executed: `False`; status `{bnci['status']}`; message `{bnci.get('message', '')}`. "
        f"The OpenBMI terminal `{bnci['openbmi_terminal_retained']}` is retained and BNCI is explicitly unassessed."
    )


def _save_figure(fig: Any, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")


def _write_degeneracy_note(root: Path, output: Path, spectrum_status: Mapping[str, Any]) -> Path:
    diagnosis_path = output / "controls/ordered_z_spectrum_degeneracy.json"
    if diagnosis_path.is_file():
        diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
        affected = diagnosis["affected_fits"]
        first = affected[0] if affected else None
        first_text = "No affected fit was found by the diagnostic scan."
        if first is not None:
            first_text = (
                f"The first affected fit is `{first['phase']}` outer fold `{first['outer_fold']}` "
                f"inner fold `{first['inner_fold']}`, requested rank `{first['requested_rank']}`. "
                f"Centered feature ranks were `{first['centered_session0_numerical_rank']}` and "
                f"`{first['centered_session1_numerical_rank']}`; cross-covariance numerical rank "
                f"was `{first['cross_covariance_numerical_rank']}`. Minimum projected-score "
                f"standard deviations were `{min(first['projected_score_std_session0']):.17g}` and "
                f"`{min(first['projected_score_std_session1']):.17g}` against the frozen threshold "
                f"`{first['epsilon_relative_threshold']:.17g}`. Exact zero: `{first['exact_zero']}`."
            )
        algebra = (
            f"Binary `Z_R=-Z_L` relative residual was `{diagnosis['binary_z_opposition_relative_residual']:.17g}`; "
            f"the reversed ordered-eigenvalue opposition residual was "
            f"`{diagnosis['ordered_eigen_opposition_relative_residual']:.17g}`, and the spectrum-signature "
            f"palindrome residual was `{diagnosis['binary_spectrum_signature_palindrome_max_abs_residual']:.17g}`. "
            f"The frozen diagnostic classifies the degeneracy as following from binary `Z` algebra and "
            f"ordered-eigen symmetry: `{diagnosis['follows_from_binary_z_algebra_and_ordered_eigen_symmetry']}`."
        )
        singular = ""
        if first is not None:
            singular = (
                "\n\nFirst-fit cross-covariance singular values: `"
                + ", ".join(f"{value:.17g}" for value in first["cross_covariance_singular_values"])
                + "`."
            )
    else:
        first_text = "No numerical-degeneracy detail file was produced because the ordered-spectrum control did not fail with the V1 degeneracy."
        algebra = "No binary-algebra conclusion was computed."
        singular = ""
    note = f"""# Ordered-Z Spectrum Degeneracy Note

Status: **{spectrum_status['status']} — NON-VOTING DIAGNOSTIC ONLY**

This note characterizes the V1/V1.1 ordered-`Z` eigenspectrum failure without changing epsilon, adding jitter, clipping singular values, dropping folds or coordinates, reducing the rank grid, or applying any regularization/workaround. It does not vote in, rescue, or overturn the OpenBMI sensor terminal.

## Location and numerical contract

{first_text}{singular}

## Binary algebra diagnostic

{algebra}

All affected-fold ranks, singular values, projected-score standard deviations, epsilon-relative ratios, and subject memberships are stored in `outputs/subject_class_population_structure_v1_1/controls/ordered_z_spectrum_degeneracy.json`.
"""
    path = root / "docs/ORDERED_Z_SPECTRUM_DEGENERACY_NOTE.md"
    atomic_write_bytes(path, note.encode("utf-8"))
    return path


def validate_final_outputs(repo_root: str | Path) -> None:
    root = Path(repo_root).resolve()
    amendment, _, _, _ = load_amendment_config(root)
    output = output_path(root, amendment)
    terminal = json.loads((output / "decisions/terminal_decision.json").read_text(encoding="utf-8"))
    report = (output / "report/subject_class_population_structure_v1_1.md").read_text(encoding="utf-8")
    if terminal["terminal_decision"] not in report:
        raise v1.DataContractError("V1.1 report terminal mismatch")
    observed_path = output / "decisions/openbmi_observed.json"
    if not terminal["terminal_decision"].startswith("UNASSESSED_"):
        observed = json.loads(observed_path.read_text(encoding="utf-8"))
        if f"{observed['statistic']:.6f}" not in report:
            raise v1.DataContractError("V1.1 report primary statistic mismatch")
    if terminal_next_question(terminal["terminal_decision"]) not in report:
        raise v1.DataContractError("V1.1 terminal-specific next question mismatch")
    v1.validate_no_nonfinite_outputs(output)


def run_final_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ensure_amendment_real_access_lock(root)
    amendment, amendment_hash, scientific, scientific_hash = load_amendment_config(root)
    validate_immutable_v1_history(root, amendment)
    output = output_path(root, amendment)
    _ensure_directories(output)
    reliability = json.loads((output / "decisions/reliability_gate.json").read_text(encoding="utf-8"))
    observed = json.loads((output / "decisions/openbmi_observed.json").read_text(encoding="utf-8"))
    terminal = json.loads((output / "decisions/terminal_decision.json").read_text(encoding="utf-8"))
    controls = json.loads((output / "decisions/openbmi_controls.json").read_text(encoding="utf-8"))
    bnci = json.loads((output / "decisions/bnci_diagnostic.json").read_text(encoding="utf-8"))
    decision = terminal["terminal_decision"]
    control_map = {row["control"]: row for row in controls["statuses"]}
    spectrum_status = control_map["ordered_z_eigenspectrum"]
    note_path = _write_degeneracy_note(root, output, spectrum_status)

    parent_hashes = v1.validate_parent_hashes(root, scientific)
    pd.DataFrame([
        {"dataset": "OpenBMI/Lee2019-MI", "role": scientific["datasets"]["openbmi"]["role"], "subjects": 54, "sessions": 2, "classes": 2, "channels": 20, "feature_dimension": 210, "object_sha256": parent_hashes["openbmi_objects"], "object_gate": "PASS"},
        {"dataset": "BNCI2014_001", "role": scientific["datasets"]["bnci"]["role"], "subjects": 9, "sessions": 2, "classes": 4, "channels": 22, "feature_dimension": 759, "object_sha256": parent_hashes["bnci_objects"], "object_gate": "PASS"},
    ]).to_csv(output / "tables/dataset_object_contract.csv", index=False, lineterminator="\n")

    selected = np.asarray(observed["selected_ranks"], dtype=int)
    gate_rows = [
        {"gate": "V1 history and scientific equivalence", "observed": "PASS", "criterion": "PASS", "passed": True},
        {"gate": "measurement reliability", "observed": "PASS" if reliability["passed"] else "FAIL", "criterion": "both sessions positive, p<=0.05, influence", "passed": bool(reliability["passed"])},
        {"gate": "primary statistic", "observed": observed["statistic"], "criterion": ">0", "passed": observed["statistic"] > 0},
        {"gate": "session 0->1 direction", "observed": observed["forward_median"], "criterion": ">0", "passed": observed["forward_median"] > 0},
        {"gate": "session 1->0 direction", "observed": observed["reverse_median"], "criterion": ">0", "passed": observed["reverse_median"] > 0},
        {"gate": "subject-pairing null", "observed": terminal["pairing_p"], "criterion": "p<=0.05", "passed": terminal["pairing_p"] <= 0.05},
        {"gate": "class-semantics null", "observed": terminal["class_p"], "criterion": "p<=0.05", "passed": terminal["class_p"] <= 0.05},
        {"gate": "equal-rank random subspace", "observed": terminal["random_subspace_p"], "criterion": "p<=0.05", "passed": terminal["random_subspace_p"] <= 0.05},
        {"gate": "median selected rank", "observed": float(np.median(selected)), "criterion": "<=8", "passed": float(np.median(selected)) <= 8},
        {"gate": "fold rank frequency", "observed": int(np.count_nonzero(selected <= 8)), "criterion": ">=4 of 6", "passed": int(np.count_nonzero(selected <= 8)) >= 4},
        {"gate": "leave-one-subject sign", "observed": "PASS" if observed["influence_sign_pass"] else "FAIL", "criterion": "all >0", "passed": bool(observed["influence_sign_pass"])},
        {"gate": "full-space stability", "observed": "PASS" if terminal["full_space_stable"] else "FAIL", "criterion": "T>0 and pairing p<=0.05", "passed": bool(terminal["full_space_stable"])},
    ]
    pd.DataFrame(gate_rows).to_csv(output / "tables/terminal_gate_table.csv", index=False, lineterminator="\n", float_format="%.17g")

    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (6.4, 4.2), "font.size": 9, "axes.grid": False})
    figures = output / "figures"

    rank_frame = pd.read_csv(output / "tables/openbmi_rank_by_rank_scores.csv")
    rank_source = rank_frame.groupby("rank", as_index=False).agg(
        median_delta=("delta_average", "median"),
        forward_median=("delta_forward", "median"),
        reverse_median=("delta_reverse", "median"),
    )
    rank_source.to_csv(figures / "figure_1_heldout_score_vs_rank.csv", index=False, lineterminator="\n", float_format="%.17g")
    fig, ax = plt.subplots(); ax.plot(rank_source["rank"], rank_source["median_delta"], marker="o", label="bidirectional"); ax.plot(rank_source["rank"], rank_source["forward_median"], marker=".", label="0→1"); ax.plot(rank_source["rank"], rank_source["reverse_median"], marker=".", label="1→0"); ax.axhline(0, color="black", linewidth=0.8); ax.set(xlabel="rank", ylabel="held-out separation", title="OpenBMI held-out score by frozen rank"); ax.legend(frameon=False); _save_figure(fig, figures, "figure_1_heldout_score_vs_rank"); plt.close(fig)

    rank_distribution = pd.Series(selected).value_counts().sort_index().rename_axis("rank").reset_index(name="outer_fold_count")
    rank_distribution.to_csv(figures / "figure_2_selected_rank_distribution.csv", index=False, lineterminator="\n")
    fig, ax = plt.subplots(); ax.bar(rank_distribution["rank"].astype(str), rank_distribution["outer_fold_count"]); ax.set(xlabel="selected rank", ylabel="outer folds", title="Frozen inner-CV selected ranks"); _save_figure(fig, figures, "figure_2_selected_rank_distribution"); plt.close(fig)

    for number, kind, title in ((3, "pairing", "Subject-pairing destruction"), (4, "class", "Class-semantics destruction"), (5, "random", "Equal-rank random subspace")):
        with np.load(output / f"nulls/openbmi_{kind}_null.npz", allow_pickle=False) as archive:
            values = np.asarray(archive["statistics"])[:, 0]
        pd.DataFrame({"replicate": np.arange(len(values)), "null_statistic": values}).to_csv(figures / f"figure_{number}_{kind}_null.csv", index=False, lineterminator="\n", float_format="%.17g")
        fig, ax = plt.subplots(); ax.hist(values, bins=35, color="#6d86a6", alpha=0.85); ax.axvline(observed["statistic"], color="#b23a48", linewidth=2, label="observed"); ax.set(xlabel="median held-out separation", ylabel="replicates", title=title); ax.legend(frameon=False); _save_figure(fig, figures, f"figure_{number}_{kind}_null"); plt.close(fig)

    with np.load(output / "objects/openbmi_observed_core.npz", allow_pickle=False) as archive:
        similarity = np.asarray(archive["heldout_similarity_block_matrix"]); mask = np.asarray(archive["heldout_similarity_mask"]).astype(bool)
    rr, cc = np.nonzero(mask)
    pd.DataFrame({"session0_subject": rr + 1, "session1_subject": cc + 1, "similarity": similarity[rr, cc]}).to_csv(figures / "figure_6_selected_latent_similarity_matrix.csv", index=False, lineterminator="\n", float_format="%.17g")
    fig, ax = plt.subplots(figsize=(5.4, 4.8)); image = ax.imshow(np.ma.masked_where(~mask, similarity), cmap="coolwarm", aspect="auto"); ax.set(xlabel="session 1 subject", ylabel="session 0 subject", title="Held-out selected-latent similarity (fold blocks)"); fig.colorbar(image, ax=ax, shrink=0.8); _save_figure(fig, figures, "figure_6_selected_latent_similarity_matrix"); plt.close(fig)

    reliability_frame = pd.read_csv(output / "tables/reliability_subject_scores.csv")
    reliability_frame.to_csv(figures / "figure_7_split_half_reliability.csv", index=False, lineterminator="\n", float_format="%.17g")
    fig, ax = plt.subplots()
    for index, session in enumerate(scientific["datasets"]["openbmi"]["sessions"]):
        values = reliability_frame.loc[reliability_frame["session"].astype(str) == str(session), "delta_average"].to_numpy(); ax.scatter(np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)), values, s=12, alpha=0.7); ax.plot(index, np.median(values), marker="_", markersize=18, color="black")
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks([0, 1], ["session 0", "session 1"]); ax.set(ylabel="A/B same-subject separation", title="Outer-fold-safe measurement reliability"); _save_figure(fig, figures, "figure_7_split_half_reliability"); plt.close(fig)

    latent_path = output / "tables/openbmi_latent_mode1_scores.csv"
    if latent_path.is_file():
        latent = pd.read_csv(latent_path); latent.to_csv(figures / "figure_8_openbmi_latent_score_scatter.csv", index=False, lineterminator="\n", float_format="%.17g"); fig, ax = plt.subplots(); ax.scatter(latent["score_session0_mode1"], latent["score_session1_mode1"], s=24, alpha=0.8); ax.axhline(0, color="grey", linewidth=0.6); ax.axvline(0, color="grey", linewidth=0.6); ax.set(xlabel="session 0 standardized mode-1 score", ylabel="session 1 standardized mode-1 score", title="Held-out paired first-mode coordinates"); _save_figure(fig, figures, "figure_8_openbmi_latent_score_scatter"); plt.close(fig)

    if bnci["executed"] and (output / "tables/bnci_class_mode_loadings.csv").is_file():
        loadings = pd.read_csv(output / "tables/bnci_class_mode_loadings.csv"); source = loadings.groupby(["class", "mode"], as_index=False)["loading_energy"].median(); source.to_csv(figures / "figure_9_bnci_class_mode_loading.csv", index=False, lineterminator="\n", float_format="%.17g"); pivot = source.pivot(index="class", columns="mode", values="loading_energy").fillna(0.0); fig, ax = plt.subplots(figsize=(5.6, 3.6)); image = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto"); ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns]); ax.set_yticks(range(len(pivot.index)), pivot.index); ax.set(xlabel="mode", ylabel="class", title="BNCI median class loading energy"); fig.colorbar(image, ax=ax, shrink=0.8); _save_figure(fig, figures, "figure_9_bnci_class_mode_loading"); plt.close(fig)
    else:
        pd.DataFrame([{"status": bnci["status"], "message": bnci.get("message", "")}]).to_csv(figures / "figure_9_bnci_class_mode_loading.csv", index=False, lineterminator="\n"); fig, ax = plt.subplots(); ax.axis("off"); ax.text(0.5, 0.5, f"BNCI: {bnci['status']}", ha="center", va="center"); _save_figure(fig, figures, "figure_9_bnci_class_mode_loading"); plt.close(fig)

    spectrum_result = spectrum_status.get("result", {})
    spectrum_value = spectrum_result.get("statistic")
    comparison = pd.DataFrame([
        {"control": "sensor_primary", "statistic": observed["statistic"], "status": "VOTING_COMPLETED"},
        {"control": "ordered_Z_eigenvalues", "statistic": spectrum_value if spectrum_value is not None else "UNASSESSED", "status": spectrum_status["status"]},
    ])
    comparison.to_csv(figures / "figure_10_sensor_vs_spectrum.csv", index=False, lineterminator="\n", float_format="%.17g")
    fig, ax = plt.subplots(); ax.bar(["sensor primary"], [observed["statistic"]], color=["#315a7d"]); ax.axhline(0, color="black", linewidth=0.8); ax.text(0.5, 0.5, spectrum_status["status"], transform=ax.transAxes, ha="center", va="center", wrap=True); ax.set(ylabel="held-out separation", title="Sensor primary and non-voting spectrum status"); _save_figure(fig, figures, "figure_10_sensor_vs_spectrum"); plt.close(fig)

    next_question = terminal_next_question(decision)
    controls_text = "\n".join(
        f"- `{name}`: `{control_map[name]['status']}`"
        for name in ("magnitude_sensitivity", "same_session_pca", "ordered_z_eigenspectrum", "generalized_eigen_signature", "selected_mode_split_half", "action_overlap")
    )
    bnci_text = bnci_report_sentence(bnci)
    report = f"""# Cross-Session Population Structure of Subject-Class Interaction V1.1

## Outcome

The technical-amendment terminal is **{decision}**. V1 remains immutable at `UNASSESSED_NUMERICAL_OR_DATA_CONTRACT_FAILURE`; V1.1 recomputed the frozen pipeline from parent objects and did not recover any hidden V1 value.

OpenBMI reliability was `{reliability['passed']}`: session 0 `T={reliability['sessions'][0]['observed']:.6f}`, `p={reliability['sessions'][0]['p_value']:.6f}`; session 1 `T={reliability['sessions'][1]['observed']:.6f}`, `p={reliability['sessions'][1]['p_value']:.6f}`.

The held-out primary statistic is `{observed['statistic']:.6f}` (95% subject-bootstrap CI `{observed['bootstrap_ci_95'][0]:.6f}` to `{observed['bootstrap_ci_95'][1]:.6f}`), with direction medians `{observed['forward_median']:.6f}` and `{observed['reverse_median']:.6f}`. Selected ranks are `{observed['selected_ranks']}` (median `{observed['median_selected_rank']:.1f}`, frequency `{observed['rank_frequency']}`). Subject-pairing, class-semantics, and equal-rank random-subspace p-values are `{terminal['pairing_p']:.6f}`, `{terminal['class_p']:.6f}`, and `{terminal['random_subspace_p']:.6f}`. The full-space baseline is `{observed['full_space_statistic']:.6f}` with pairing p `{terminal['full_space_pairing_p']:.6f}`. Leave-one-subject sign stability is `{observed['influence_sign_pass']}`.

## V1/V1.1 equivalence

All scientific rows in `protocol/v1_v1_1_scientific_contract_equivalence.csv` are unchanged. The only `changed=true` row is secondary failure isolation. Parent hashes, folds, ranks, features, normalization, low-rank cap, 1,999-replicate nulls, seeds/namespaces, and terminal function are identical to V1.

## Non-voting controls

{controls_text}

The ordered-spectrum failure is retained without workaround and characterized in `docs/ORDERED_Z_SPECTRUM_DEGENERACY_NOTE.md`. No non-voting status changed the OpenBMI terminal.

## BNCI secondary diagnostic

{bnci_text}

BNCI is explicitly secondary and cannot rescue or overturn OpenBMI. Its basis is not claimed to equal the OpenBMI basis.

## Interpretation boundary

This terminal answers only whether the stable montage-registered mean-level subject×class interaction has a held-out-subject, cross-session, population-shared low-dimensional linear structure under the V1 gates. It does not establish a full conditional distribution, dispersion structure, physiology, source anatomy, causality, unlabeled target identifiability, ASD biomarker, clinical diagnosis, TTA recoverability, globally identifiable `Q_s`, or cross-dataset equality of modes. No classifier, network, adapter, loss, or TTA method is proposed.

## Next question

{next_question}
"""
    report_path = output / "report/subject_class_population_structure_v1_1.md"
    atomic_write_bytes(report_path, report.encode("utf-8"))
    validate_final_outputs(root)
    artifact_hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": "subject-class-population-structure-v1-1-final-manifest",
        "terminal_decision": decision,
        "amendment_config_sha256": amendment_hash,
        "scientific_v1_config_sha256": scientific_hash,
        "parent_hashes": parent_hashes,
        "immutable_v1_hashes": validate_immutable_v1_history(root, amendment),
        "artifact_hashes": artifact_hashes,
        "report_sha256": sha256_file(report_path),
        "ordered_spectrum_note_sha256": sha256_file(note_path),
        "openbmi_terminal_sha256": sha256_file(output / "decisions/terminal_decision.json"),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {
        "terminal_decision": decision,
        "report": str(report_path.relative_to(root)),
        "manifest": str((output / "manifest.json").relative_to(root)),
        "artifact_count": len(artifact_hashes),
    }
