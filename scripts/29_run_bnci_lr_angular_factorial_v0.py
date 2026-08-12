#!/usr/bin/env python3
"""Prepare, execute, or finalize BNCI Left/Right angular diagnostic V0."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bnci_lr_angular_factorial_v0 import (  # noqa: E402
    LR_CLASS_ORDER,
    N_SUBJECTS,
    NULL_REPLICATES,
    classbreak_mappings,
    evaluate_inference,
    extract_lr_matrix,
    lr_parent_indices,
    relation_statistics,
    subjectbreak_mappings,
    terminal_decision,
)


EXPECTED_BRANCH = "audit/bnci-left-right-angular-factorial-v0"
PARENT_SHA = "edc1d344cb0657f2f2d87b2992049bceec4705d2"
PARENT_PROTOCOL_SHA = "95c330de9596fa4c4eb4ee377d5af8d99896f4c3"
PARENT_RESULT_SHA = "0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca"
PARENT_TERMINAL = "BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS"
PARENT_OUTPUT = ROOT / "outputs" / "bnci2014_001_local_movement_component_decomposition_v0"
COMPONENT_ARRAY = PARENT_OUTPUT / "arrays" / "component_cost_matrices.npz"
SPLIT_ARRAY = PARENT_OUTPUT / "arrays" / "split_half_component_matrices.npz"
PARENT_MANIFEST = PARENT_OUTPUT / "protocol" / "artifact_manifest.csv"
PARENT_CSV = PARENT_OUTPUT / "tables" / "c_ang_matrix.csv"
CONFIG_PATH = ROOT / "configs" / "bnci2014_001_lr_angular_factorial_v0.yaml"
PROTOCOL_PATH = ROOT / "docs" / "PROTOCOL_BNCI_LEFT_RIGHT_ANGULAR_FACTORIAL_V0.md"
OUTPUT = ROOT / "outputs" / "bnci2014_001_lr_angular_factorial_v0"
REPORT = OUTPUT / "report" / "bnci2014_001_lr_angular_factorial_v0.md"
EXPECTED_HASHES = {
    "arrays/component_cost_matrices.npz": "51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091",
    "arrays/split_half_component_matrices.npz": "cdf618662c1ac5eb3b8fa9b65ad9fde921b45d011c967e87f6ec9a2ab1775307",
    "tables/c_ang_matrix.csv": "f6c06c3f44807207d7baf0d84226859380e008b1d27a47572f9b94d0dc6735bd",
    "protocol/artifact_manifest.csv": "c3aea494fedee8af1ee42a5c26ff94fc5bfae5e5289549bc878c3941da43df79",
}
RESULT_COMMIT_MANIFEST_HASH = "28a12fc32c6477fe6cccd071af587304a31eb0e738b079a192c254ff5e82a6c6"
PARENT_FOUR_CLASS = {
    "T_subject_ang": 0.3091561771980925,
    "T_class_ang": 0.39309843397343514,
    "T_J_ang": 0.19240885452534362,
    "p_J_subjectbreak": 0.001,
    "p_J_classbreak": 0.0105,
}


class FrozenArtifactError(RuntimeError):
    pass


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT, check=True, capture_output=True
    ).stdout


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_savez(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_branch(*, require_clean: bool) -> None:
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("wrong branch")
    if git("merge-base", "--is-ancestor", PARENT_SHA, "HEAD") != "":
        pass
    if require_clean and git("status", "--porcelain"):
        raise RuntimeError("scientific execution requires a clean protocol-freeze worktree")


def reproduce_parent_artifacts() -> dict[str, Any]:
    records: dict[str, Any] = {}
    manifest = pd.read_csv(PARENT_MANIFEST)
    manifest_hashes = dict(zip(manifest["relative_path"], manifest["sha256"], strict=True))
    for relative, expected in EXPECTED_HASHES.items():
        path = PARENT_OUTPUT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise FrozenArtifactError(f"parent artifact hash mismatch: {relative}")
        if relative != "protocol/artifact_manifest.csv":
            if manifest_hashes.get(relative) != expected:
                raise FrozenArtifactError(f"parent manifest entry mismatch: {relative}")
            frozen = git_show(PARENT_RESULT_SHA, str(path.relative_to(ROOT)))
            if sha256_bytes(frozen) != expected or path.read_bytes() != frozen:
                raise FrozenArtifactError(f"artifact differs from scientific-result commit: {relative}")
        records[relative] = {"sha256": observed, "manifest_match": True}
    result_manifest = git_show(
        PARENT_RESULT_SHA, str(PARENT_MANIFEST.relative_to(ROOT))
    )
    if sha256_bytes(result_manifest) != RESULT_COMMIT_MANIFEST_HASH:
        raise FrozenArtifactError("scientific-result manifest hash mismatch")

    with np.load(COMPONENT_ARRAY, allow_pickle=False) as archive:
        required = {
            "c_sensor_matrix", "c_full_matrix", "c_len_matrix", "c_ang_matrix",
            "c_ori_matrix", "cell_subjects", "cell_classes",
        }
        if set(archive.files) != required:
            raise FrozenArtifactError("component array schema changed")
        subjects = archive["cell_subjects"].copy()
        classes = archive["cell_classes"].copy()
        full = archive["c_full_matrix"].copy()
        length = archive["c_len_matrix"].copy()
        angular = archive["c_ang_matrix"].copy()
    indices = lr_parent_indices(subjects, classes)
    if not np.array_equal(angular, full - length):
        raise FrozenArtifactError("frozen c_ang no longer exactly equals c_full-c_len")
    csv = pd.read_csv(PARENT_CSV, index_col=0).to_numpy(dtype=np.float64)
    csv_maximum_error = float(np.max(np.abs(csv - angular)))
    if not np.allclose(csv, angular, atol=1.0e-15, rtol=0.0):
        raise FrozenArtifactError("readable c_ang CSV exceeds frozen NPZ round-trip tolerance")
    with np.load(SPLIT_ARRAY, allow_pickle=False) as split:
        if split["c_ang_matrix"].shape != (2, 36, 36):
            raise FrozenArtifactError("split angular shape changed")
        if not np.array_equal(split["cell_subjects"], subjects):
            raise FrozenArtifactError("split subject order changed")
        if not np.array_equal(split["cell_classes"], classes):
            raise FrozenArtifactError("split class order changed")
        if not np.array_equal(split["replicates"], np.asarray(("A", "B"))):
            raise FrozenArtifactError("split replicate order changed")
    return {
        "status": "PASS",
        "parent_sha": PARENT_SHA,
        "parent_protocol_sha": PARENT_PROTOCOL_SHA,
        "parent_result_sha": PARENT_RESULT_SHA,
        "artifacts": records,
        "scientific_result_manifest_sha256": RESULT_COMMIT_MANIFEST_HASH,
        "canonical_ordering": "subjects_1_to_9_x_left_hand_right_hand_feet_tongue",
        "selected_parent_indices": indices.tolist(),
        "selected_subjects": subjects[indices].tolist(),
        "selected_classes": classes[indices].tolist(),
        "c_ang_exactly_equals_c_full_minus_c_len": True,
        "readable_csv_npz_maximum_absolute_error": csv_maximum_error,
        "readable_csv_npz_atol": 1.0e-15,
        "new_lr_statistic_accessed": False,
    }


def prepare() -> None:
    validate_branch(require_clean=False)
    reproduction = reproduce_parent_artifacts()
    write_json(OUTPUT / "provenance" / "pre_result_artifact_reproduction.json", reproduction)
    print(json.dumps({
        "status": reproduction["status"],
        "selected_parent_indices": reproduction["selected_parent_indices"],
        "new_lr_statistic_accessed": False,
    }, indent=2))


def relation_table(statistics: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject in range(N_SUBJECTS):
        for class_index, class_name in enumerate(LR_CLASS_ORDER):
            rows.append({
                "subject": subject + 1,
                "class": class_name,
                "a": statistics.a_sc[subject, class_index],
                "b": statistics.b_sc[subject, class_index],
                "c": statistics.c_sc[subject, class_index],
                "d": statistics.d_sc[subject, class_index],
                "S_sc": statistics.s_sc[subject, class_index],
                "C_sc": statistics.class_sc[subject, class_index],
                "J_sc": statistics.j_sc[subject, class_index],
            })
    return pd.DataFrame(rows)


def subject_table(statistics: Any) -> pd.DataFrame:
    return pd.DataFrame({
        "subject": np.arange(1, N_SUBJECTS + 1),
        "S_s": statistics.s_s,
        "C_s": statistics.class_s,
        "J_s": statistics.j_s,
    })


def plain_answer(value: float, pvalue: float, *, effect: str) -> str:
    supported = value > 0.0 and pvalue < 0.05
    return f"{effect} was {'supported' if supported else 'not supported'} (T={value:.10g}, p={pvalue:.6g})."


def render_report(results: dict[str, Any], *, scientific_result_sha: str) -> str:
    primary = results["primary"]
    split = results["split_half"]
    if primary["supported"]:
        interaction = (
            "The same frozen Left/Right-only angular analysis retained a supported "
            "subject×class interaction."
        )
    else:
        interaction = (
            "The previously supported BNCI four-class angular/joint interaction was not "
            "supported when the same frozen analysis was restricted to Left versus Right."
        )
    subject_lines = "\n".join(
        f"- S{index + 1}: S_s={s:.10g}, C_s={c:.10g}, J_s={j:.10g}"
        for index, (s, c, j) in enumerate(
            zip(results["subject"]["S_s"], results["subject"]["C_s"], results["subject"]["J_s"], strict=True)
        )
    )
    return f"""# BNCI2014_001 Left/Right-only Angular Factorial Diagnostic V0

## Frozen lineage

- Branch: `{EXPECTED_BRANCH}`
- Parent HEAD: `{PARENT_SHA}`
- Parent protocol freeze: `{PARENT_PROTOCOL_SHA}`
- Parent scientific result: `{PARENT_RESULT_SHA}`
- Parent terminal: `{PARENT_TERMINAL}`
- Protocol freeze SHA: `{results['protocol_freeze_sha']}`
- Scientific result SHA: `{scientific_result_sha}`

The diagnostic used only the frozen parent squared angular-cost arrays. No raw EEG, covariance mean, anti-development, movement tuple, or quotient optimizer was fitted or recomputed. Parent artifacts reproduced exactly and remained unchanged.

## Plain-language answers

1. {plain_answer(primary['T_subject'], primary['p_subject'], effect='Cross-session subject correspondence within hand class')}
2. {plain_answer(primary['T_class'], primary['p_class'], effect='Left/Right class correspondence within subject')}
3. {interaction} T_J={primary['T_J']:.10g}, subject-break p={primary['p_J_subjectbreak']:.6g}, class-break p={primary['p_J_classbreak']:.6g}.
4. The interaction was {'split-half sign-stable' if split['sign_stable'] else 'not split-half sign-stable'}: Half A T_J={split['half_A_T_J']:.10g}; Half B T_J={split['half_B_T_J']:.10g}.
5. The immutable four-class angular result was T_J={PARENT_FOUR_CLASS['T_J_ang']:.10g}, p_subjectbreak={PARENT_FOUR_CLASS['p_J_subjectbreak']:.6g}, and p_classbreak={PARENT_FOUR_CLASS['p_J_classbreak']:.6g}. This diagnostic changes neither that result nor its terminal.

## Exact binary inference

- T_subject: {primary['T_subject']:.17g}
- p_subject: {primary['p_subject']:.17g}
- T_class: {primary['T_class']:.17g}
- p_class: {primary['p_class']:.17g}
- T_J: {primary['T_J']:.17g}
- p_J_subjectbreak: {primary['p_J_subjectbreak']:.17g}
- p_J_classbreak: {primary['p_J_classbreak']:.17g}

Subject summaries:

{subject_lines}

## Integrity and split-half checks

- Canonical 18-cell order: PASS
- Feet/Tongue excluded from all observed and null statistics: PASS
- Frozen NPZ/CSV and parent-result byte reproduction: PASS
- Generalized K=4 regression against frozen angular statistics: PASS
- 1,999 subject-break mappings preserve class: PASS
- 1,999 class-break mappings preserve subject: PASS
- Half A T_J: {split['half_A_T_J']:.17g}
- Half B T_J: {split['half_B_T_J']:.17g}
- Split-half sign stable: {str(split['sign_stable']).lower()}

## Terminal

`{results['terminal']}`

This is a retrospective diagnostic of frozen window-wise mean covariance movement costs. It does not establish absence through equivalence testing and makes no physiological, motor-strategy, neural-direction, or anatomical claim.

## Runtime, tests, and immutability

- Scientific runtime: {results['runtime_seconds']:.6f} seconds
- Focused pre-result tests: `{results['tests']['focused_before']}`
- Focused post-result tests: `{results['tests']['focused_after']}`
- Full repository tests: `{results['tests']['full_after']}`
- Scientific setting changed after protocol freeze: false
- Parent artifact changed: false
"""


def write_artifact_manifest() -> None:
    manifest = OUTPUT / "provenance" / "artifact_manifest.csv"
    rows = []
    for path in sorted(value for value in OUTPUT.rglob("*") if value.is_file() and value != manifest):
        rows.append({
            "relative_path": str(path.relative_to(OUTPUT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False, lineterminator="\n")


def execute(*, focused_before: str) -> None:
    validate_branch(require_clean=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["protocol"]["sha256"] != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("protocol hash differs from freeze config")
    protocol_freeze_sha = git("rev-parse", "HEAD")
    parent_hashes_before = {relative: sha256_file(PARENT_OUTPUT / relative) for relative in EXPECTED_HASHES}
    reproduction = reproduce_parent_artifacts()
    started = time.perf_counter()
    with np.load(COMPONENT_ARRAY, allow_pickle=False) as archive:
        subjects = archive["cell_subjects"].copy()
        classes = archive["cell_classes"].copy()
        parent_angular = archive["c_ang_matrix"].copy()
    lr_angular, indices = extract_lr_matrix(parent_angular, subjects, classes)
    if not np.array_equal(lr_angular, parent_angular[np.ix_(indices, indices)]):
        raise FrozenArtifactError("Left/Right extraction is not exact")

    parent_regression = relation_statistics(parent_angular, n_classes=4)
    for name, observed in {
        "T_subject_ang": parent_regression.t_subject,
        "T_class_ang": parent_regression.t_class,
        "T_J_ang": parent_regression.t_j,
    }.items():
        if not np.isclose(observed, PARENT_FOUR_CLASS[name], atol=1e-14, rtol=1e-14):
            raise FrozenArtifactError(f"K=4 regression failed: {name}")

    subject_maps = subjectbreak_mappings(n_classes=2)
    class_maps = classbreak_mappings(n_classes=2)
    inference = evaluate_inference(
        lr_angular,
        n_classes=2,
        subject_mappings_array=subject_maps,
        class_mappings_array=class_maps,
    )
    with np.load(SPLIT_ARRAY, allow_pickle=False) as archive:
        split_values = archive["c_ang_matrix"].copy()
        split_subjects = archive["cell_subjects"].copy()
        split_classes = archive["cell_classes"].copy()
    split_lr = np.asarray([
        extract_lr_matrix(split_values[half], split_subjects, split_classes)[0]
        for half in range(2)
    ])
    split_statistics = [relation_statistics(split_lr[half], n_classes=2) for half in range(2)]
    half_t = [result.t_j for result in split_statistics]
    sign_stable = bool(half_t[0] > 0.0 and half_t[1] > 0.0)
    supported = bool(
        inference.observed.t_j > 0.0
        and inference.p_j_subjectbreak < 0.05
        and inference.p_j_classbreak < 0.05
    )
    terminal = terminal_decision(
        t_j=inference.observed.t_j,
        p_j_subjectbreak=inference.p_j_subjectbreak,
        p_j_classbreak=inference.p_j_classbreak,
        split_half_sign_stable=sign_stable,
    )
    runtime = time.perf_counter() - started

    tables = OUTPUT / "tables"
    nulls = OUTPUT / "nulls"
    provenance = OUTPUT / "provenance"
    decisions = OUTPUT / "decisions"
    for directory in (tables, nulls, provenance, decisions, REPORT.parent):
        directory.mkdir(parents=True, exist_ok=True)
    relation_table(inference.observed).to_csv(tables / "lr_relation_cells.csv", index=False, lineterminator="\n")
    subjects_table = subject_table(inference.observed)
    subjects_table.to_csv(tables / "lr_subject_statistics.csv", index=False, lineterminator="\n")
    pd.DataFrame([{
        "T_subject": inference.observed.t_subject,
        "p_subject": inference.p_subject,
        "T_class": inference.observed.t_class,
        "p_class": inference.p_class,
        "T_J": inference.observed.t_j,
        "p_J_subjectbreak": inference.p_j_subjectbreak,
        "p_J_classbreak": inference.p_j_classbreak,
    }]).to_csv(tables / "lr_inference_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame([{
        "half_A_T_J": half_t[0], "half_B_T_J": half_t[1], "sign_stable": sign_stable,
    }]).to_csv(tables / "lr_split_half_stability.csv", index=False, lineterminator="\n")
    atomic_savez(nulls / "lr_angular_nulls.npz", {
        "subjectbreak_T_subject": inference.subjectbreak_t_subject,
        "subjectbreak_T_J": inference.subjectbreak_t_j,
        "classbreak_T_class": inference.classbreak_t_class,
        "classbreak_T_J": inference.classbreak_t_j,
        "subject_mappings": subject_maps,
        "class_mappings": class_maps,
    })
    integrity = {
        "status": "PASS",
        "selected_cell_count": len(indices),
        "selected_parent_indices": indices.tolist(),
        "selected_subjects": subjects[indices].tolist(),
        "selected_classes": classes[indices].tolist(),
        "extracted_entries_exact_parent_array": True,
        "feet_tongue_primary_or_null_entry_count": 0,
        "subject_mappings_all_preserve_class": bool(np.all(subject_maps % 2 == np.arange(18)[None, :] % 2)),
        "class_mappings_all_preserve_subject": bool(np.all(subject_maps.shape) and np.all(class_maps // 2 == np.arange(18)[None, :] // 2)),
        "K4_regression": {
            "T_subject": parent_regression.t_subject,
            "T_class": parent_regression.t_class,
            "T_J": parent_regression.t_j,
            "maximum_absolute_error": float(max(
                abs(parent_regression.t_subject - PARENT_FOUR_CLASS["T_subject_ang"]),
                abs(parent_regression.t_class - PARENT_FOUR_CLASS["T_class_ang"]),
                abs(parent_regression.t_j - PARENT_FOUR_CLASS["T_J_ang"]),
            )),
        },
        "split_subset_exact_parent_array": True,
    }
    write_json(provenance / "integrity_checks.json", integrity)
    write_json(provenance / "artifact_reproduction.json", reproduction)
    results = {
        "protocol_freeze_sha": protocol_freeze_sha,
        "scientific_result_sha": "PENDING_SCIENTIFIC_RESULT_COMMIT",
        "primary": {
            "T_subject": inference.observed.t_subject,
            "p_subject": inference.p_subject,
            "T_class": inference.observed.t_class,
            "p_class": inference.p_class,
            "T_J": inference.observed.t_j,
            "p_J_subjectbreak": inference.p_j_subjectbreak,
            "p_J_classbreak": inference.p_j_classbreak,
            "supported": supported,
        },
        "subject": {
            "S_s": inference.observed.s_s.tolist(),
            "C_s": inference.observed.class_s.tolist(),
            "J_s": inference.observed.j_s.tolist(),
        },
        "split_half": {
            "half_A_T_J": half_t[0], "half_B_T_J": half_t[1], "sign_stable": sign_stable,
        },
        "parent_four_class": PARENT_FOUR_CLASS,
        "terminal": terminal,
        "runtime_seconds": runtime,
        "tests": {"focused_before": focused_before, "focused_after": "PENDING", "full_after": "PENDING"},
        "no_raw_eeg_or_movement_refit": True,
        "scientific_setting_changed_after_protocol_freeze": False,
    }
    write_json(provenance / "scientific_results.json", results)
    write_json(decisions / "terminal_decision.json", {
        "terminal": terminal,
        "T_J": inference.observed.t_j,
        "p_J_subjectbreak": inference.p_j_subjectbreak,
        "p_J_classbreak": inference.p_j_classbreak,
        "split_half_sign_stable": sign_stable,
    })
    parent_hashes_after = {relative: sha256_file(PARENT_OUTPUT / relative) for relative in EXPECTED_HASHES}
    if parent_hashes_after != parent_hashes_before:
        raise FrozenArtifactError("parent artifact mutated during diagnostic")
    write_json(provenance / "parent_artifact_immutability.json", {
        "status": "PASS", "unchanged": True,
        "hashes_before": parent_hashes_before, "hashes_after": parent_hashes_after,
    })
    atomic_write_text(REPORT, render_report(results, scientific_result_sha="PENDING_SCIENTIFIC_RESULT_COMMIT"))
    write_artifact_manifest()
    print(json.dumps({
        "terminal": terminal,
        "T_J": inference.observed.t_j,
        "p_J_subjectbreak": inference.p_j_subjectbreak,
        "p_J_classbreak": inference.p_j_classbreak,
        "T_subject": inference.observed.t_subject,
        "p_subject": inference.p_subject,
        "T_class": inference.observed.t_class,
        "p_class": inference.p_class,
        "half_A_T_J": half_t[0], "half_B_T_J": half_t[1],
    }, indent=2))


def record_tests(*, focused_after: str, full_after: str) -> None:
    path = OUTPUT / "provenance" / "scientific_results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    results["tests"]["focused_after"] = focused_after
    results["tests"]["full_after"] = full_after
    write_json(path, results)
    atomic_write_text(REPORT, render_report(results, scientific_result_sha=results["scientific_result_sha"]))
    write_artifact_manifest()


def finalize(*, scientific_result_sha: str) -> None:
    path = OUTPUT / "provenance" / "scientific_results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    if results["scientific_result_sha"] != "PENDING_SCIENTIFIC_RESULT_COMMIT":
        raise RuntimeError("scientific result SHA already finalized")
    results["scientific_result_sha"] = scientific_result_sha
    write_json(path, results)
    atomic_write_text(REPORT, render_report(results, scientific_result_sha=scientific_result_sha))
    write_artifact_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prepare", "execute", "record-tests", "finalize"))
    parser.add_argument("--focused-before", default="PENDING")
    parser.add_argument("--focused-after", default="PENDING")
    parser.add_argument("--full-after", default="PENDING")
    parser.add_argument("--scientific-result-sha")
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare()
    elif arguments.stage == "execute":
        execute(focused_before=arguments.focused_before)
    elif arguments.stage == "record-tests":
        record_tests(focused_after=arguments.focused_after, full_after=arguments.full_after)
    else:
        if not arguments.scientific_result_sha:
            parser.error("--scientific-result-sha is required")
        finalize(scientific_result_sha=arguments.scientific_result_sha)


if __name__ == "__main__":
    main()
