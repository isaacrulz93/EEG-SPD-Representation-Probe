#!/usr/bin/env python3
"""Prepare, execute, record tests, or finalize frozen angular relation anatomy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bnci_angular_relation_anatomy_v0 import (  # noqa: E402
    CLASS_ORDER,
    CLASS_SHORT,
    N_CLASSES,
    N_SUBJECTS,
    PAIR_INDICES,
    PAIR_NAMES,
    SUBJECT_PAIR_NAMES,
    build_relation_anatomy,
    canonical_cell_classes,
    canonical_cell_subjects,
    leave_one_out_commonality,
    maximum_reconstruction_error,
    off_diagonal_values,
    pairwise_profile_similarity,
    reconstruction_errors,
    relation_statistics,
    six_pair_statistics,
)


EXPECTED_BRANCH = "pilot/bnci-angular-relation-anatomy-v0"
PARENT_BRANCH = "pilot/local-movement-component-decomposition-v0"
PARENT_HEAD = "edc1d344cb0657f2f2d87b2992049bceec4705d2"
PARENT_PROTOCOL_FREEZE = "95c330de9596fa4c4eb4ee377d5af8d99896f4c3"
PARENT_SCIENTIFIC_RESULT = "0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca"
PARENT_TERMINAL = "BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS"
PARENT_OUTPUT = ROOT / "outputs/bnci2014_001_local_movement_component_decomposition_v0"
FROZEN_ARRAY = PARENT_OUTPUT / "arrays/component_cost_matrices.npz"
FROZEN_TABLE = PARENT_OUTPUT / "tables/c_ang_matrix.csv"
FROZEN_MANIFEST = PARENT_OUTPUT / "protocol/artifact_manifest.csv"
EXPECTED_HASHES = {
    FROZEN_ARRAY: "51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091",
    FROZEN_TABLE: "f6c06c3f44807207d7baf0d84226859380e008b1d27a47572f9b94d0dc6735bd",
    FROZEN_MANIFEST: "c3aea494fedee8af1ee42a5c26ff94fc5bfae5e5289549bc878c3941da43df79",
}
PARENT_STATISTICS = {
    "T_subject": 0.3091561771980925,
    "T_class": 0.39309843397343514,
    "T_J": 0.19240885452534362,
}
ATOL = 1.0e-12
RTOL = 1.0e-12
OUTPUT = ROOT / "outputs/bnci2014_001_angular_relation_anatomy_v0"
REPORT = OUTPUT / "report/bnci2014_001_angular_relation_anatomy_v0.md"
PROTOCOL = ROOT / "docs/PROTOCOL_BNCI_ANGULAR_RELATION_ANATOMY_V0.md"
CONFIG = ROOT / "configs/bnci2014_001_angular_relation_anatomy_v0.yaml"
IMPLEMENTATION_FILES = (
    ROOT / "src/bnci_angular_relation_anatomy_v0.py",
    Path(__file__),
    ROOT / "tests/test_bnci_angular_relation_anatomy_v0.py",
    ROOT / "tests/test_protocol_bnci_angular_relation_anatomy_v0.py",
)


class FrozenAngularArtifactError(RuntimeError):
    """The exact frozen angular matrix cannot be recovered or reproduced."""


class ReconstructionError(RuntimeError):
    """The frozen parent statistics or six-pair algebra did not reproduce."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def require_clean_branch() -> None:
    branch = git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"expected branch {EXPECTED_BRANCH}, found {branch}")
    status = git("status", "--porcelain=v1")
    if status:
        raise RuntimeError("execution requires a clean protocol-frozen worktree")


def artifact_hashes() -> dict[str, str]:
    values: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FrozenAngularArtifactError(f"missing frozen artifact: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise FrozenAngularArtifactError(
                f"frozen artifact hash mismatch for {path}: {observed} != {expected}"
            )
        values[str(path.relative_to(ROOT))] = observed
    return values


def load_frozen_matrix() -> tuple[np.ndarray, dict[str, Any]]:
    hashes = artifact_hashes()
    with np.load(FROZEN_ARRAY, allow_pickle=False) as archive:
        required = {
            "c_sensor_matrix",
            "c_full_matrix",
            "c_len_matrix",
            "c_ang_matrix",
            "c_ori_matrix",
            "cell_subjects",
            "cell_classes",
        }
        if set(archive.files) != required:
            raise FrozenAngularArtifactError("unexpected frozen NPZ key set")
        matrix = np.array(archive["c_ang_matrix"], dtype=np.float64, copy=True)
        subjects = np.array(archive["cell_subjects"], copy=True)
        classes = np.array(archive["cell_classes"], copy=True)
        if not np.array_equal(subjects, canonical_cell_subjects()):
            raise FrozenAngularArtifactError("frozen subject ordering mismatch")
        if not np.array_equal(classes, canonical_cell_classes()):
            raise FrozenAngularArtifactError("frozen class ordering mismatch")
        derived = np.asarray(archive["c_full_matrix"] - archive["c_len_matrix"])
        if not np.array_equal(matrix, derived):
            raise FrozenAngularArtifactError("frozen c_ang is not exact c_full-c_len")
    if matrix.shape != (36, 36) or matrix.dtype != np.float64 or not np.isfinite(matrix).all():
        raise FrozenAngularArtifactError("frozen c_ang matrix shape/dtype/finite gate failed")
    readable = pd.read_csv(FROZEN_TABLE, index_col=0).to_numpy(dtype=np.float64)
    csv_max_error = float(np.max(np.abs(readable - matrix)))
    if csv_max_error > 1.0e-15:
        raise FrozenAngularArtifactError("readable c_ang table exceeds frozen text tolerance")
    reproduction = {
        "status": "PASS",
        "hashes": hashes,
        "npz_key": "c_ang_matrix",
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "finite": bool(np.isfinite(matrix).all()),
        "exact_c_full_minus_c_len": True,
        "canonical_subject_order": True,
        "canonical_class_order": True,
        "csv_maximum_absolute_roundtrip_error": csv_max_error,
    }
    return matrix, reproduction


def protocol_hash() -> str:
    digest = hashlib.sha256()
    for path in (PROTOCOL, CONFIG, *IMPLEMENTATION_FILES):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare() -> None:
    matrix, reproduction = load_frozen_matrix()
    observed = relation_statistics(matrix)
    errors = {
        "T_subject": observed.t_subject - PARENT_STATISTICS["T_subject"],
        "T_class": observed.t_class - PARENT_STATISTICS["T_class"],
        "T_J": observed.t_j - PARENT_STATISTICS["T_J"],
    }
    if max(abs(value) for value in errors.values()) > ATOL:
        raise ReconstructionError("parent four-class statistics did not reproduce")
    reproduction["parent_statistics"] = {
        "observed": {
            "T_subject": observed.t_subject,
            "T_class": observed.t_class,
            "T_J": observed.t_j,
        },
        "expected": PARENT_STATISTICS,
        "signed_errors": errors,
        "maximum_absolute_error": max(abs(value) for value in errors.values()),
    }
    reproduction["protocol_source_hash"] = protocol_hash()
    atomic_json(OUTPUT / "provenance/pre_result_reproduction.json", reproduction)
    print(json.dumps({"status": "PASS", "parent_reproduction": reproduction["parent_statistics"]}, indent=2))


def similarity_rows(
    domain: str, labels: list[str], profiles: np.ndarray, adjusted: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    commonality_records: list[dict[str, Any]] = []
    for profile_name, bank in (("raw", profiles), ("baseline_adjusted", adjusted)):
        distances, correlations = pairwise_profile_similarity(bank)
        upper = np.triu_indices(len(labels), k=1)
        for first, second in zip(*upper, strict=True):
            records.append(
                {
                    "domain": domain,
                    "profile": profile_name,
                    "first": labels[first],
                    "second": labels[second],
                    "euclidean_distance": distances[first, second],
                    "centered_correlation": correlations[first, second],
                }
            )
        loo = leave_one_out_commonality(bank)
        for label, value in zip(labels, loo, strict=True):
            commonality_records.append(
                {
                    "domain": domain,
                    "profile": profile_name,
                    "label": label,
                    "leave_one_out_correlation": value,
                }
            )
        for measure, values in (
            ("pairwise_euclidean_distance", distances[upper]),
            ("pairwise_centered_correlation", correlations[upper]),
            ("leave_one_out_correlation", loo),
        ):
            summary.append(
                {
                    "domain": domain,
                    "profile": profile_name,
                    "measure": measure,
                    "count": len(values),
                    "mean": float(np.nanmean(values)),
                    "median": float(np.nanmedian(values)),
                    "minimum": float(np.nanmin(values)),
                    "maximum": float(np.nanmax(values)),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(commonality_records), pd.DataFrame(summary)


def matrix_long_rows(
    matrices: np.ndarray, outer_labels: list[str], inner_labels: list[str], name: str
) -> pd.DataFrame:
    rows = []
    for outer_index, outer in enumerate(outer_labels):
        for first_index, first in enumerate(inner_labels):
            for second_index, second in enumerate(inner_labels):
                rows.append(
                    {
                        "matrix": name,
                        "outer_label": outer,
                        "row": first,
                        "column": second,
                        "value": matrices[outer_index, first_index, second_index],
                    }
                )
    return pd.DataFrame(rows)


def heatmap_grid(
    matrices: np.ndarray,
    titles: list[str],
    labels: list[str],
    output_stem: Path,
    *,
    colorbar_label: str,
    centered: bool,
) -> None:
    count = len(titles)
    columns = 3 if count == 9 else 2
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.8 * columns, 3.4 * rows), squeeze=False)
    minimum = float(np.min(matrices))
    maximum = float(np.max(matrices))
    if centered:
        bound = max(abs(minimum), abs(maximum))
        minimum, maximum, cmap = -bound, bound, "coolwarm"
    else:
        cmap = "viridis"
    image = None
    for index, axis in enumerate(axes.ravel()):
        if index >= count:
            axis.axis("off")
            continue
        image = axis.imshow(matrices[index], cmap=cmap, vmin=minimum, vmax=maximum)
        axis.set_title(titles[index])
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label=colorbar_label)
    figure.suptitle(output_stem.name.replace("_", " "))
    figure.subplots_adjust(left=0.07, right=0.9, bottom=0.08, top=0.92, wspace=0.32, hspace=0.36)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=180)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def subject_pair_j_plot(pair_subject: pd.DataFrame, output_stem: Path) -> None:
    pivot = pair_subject.pivot(index="subject", columns="pair", values="J_s").loc[:, PAIR_NAMES]
    values = pivot.to_numpy()
    bound = float(np.max(np.abs(values)))
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    axis.set_xticks(range(6), PAIR_NAMES)
    axis.set_yticks(range(9), [f"S{s}" for s in range(1, 10)])
    axis.set_xlabel("Binary class pair")
    axis.set_ylabel("Subject")
    axis.set_title("Subject-level angular interaction decomposition")
    figure.colorbar(image, ax=axis, label="J_s,p (squared-cost contrast)")
    figure.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=180)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def fmt(value: float) -> str:
    return f"{value:.10g}"


def render_report(results: dict[str, Any], scientific_sha: str) -> str:
    pair_rows = results["pair_group_statistics"]
    g_summary = results["commonality_summary"]["G"]
    h_summary = results["commonality_summary"]["H"]
    pair_table = "\n".join(
        f"| {row['pair']} | {fmt(row['T_S'])} | {fmt(row['T_C'])} | {fmt(row['T_J'])} |"
        for row in pair_rows
    )
    j_rank = sorted(pair_rows, key=lambda row: row["T_J"], reverse=True)
    j_rank_text = ", ".join(f"{row['pair']}={fmt(row['T_J'])}" for row in j_rank)
    subject_j = results["subject_J_summary"]
    return f"""# BNCI2014_001 Angular Relation Anatomy V0

## Verdict and provenance

`COMPLETED_FROZEN_BNCI_ANGULAR_RELATION_ANATOMY_V0`

- Branch: `{EXPECTED_BRANCH}`
- Exact base: `{PARENT_HEAD}`
- Parent protocol freeze: `{PARENT_PROTOCOL_FREEZE}`
- Parent scientific result: `{PARENT_SCIENTIFIC_RESULT}`
- Protocol freeze SHA: `{results['protocol_freeze_sha']}`
- Scientific result SHA: `{scientific_sha}`
- Frozen matrix SHA-256: `{EXPECTED_HASHES[FROZEN_ARRAY]}`

Only the frozen 36 x 36 cross-session squared `c_ang` matrix was read. No EEG,
covariance mean, anti-development, ordered movement object, or quotient
optimizer was fitted or recomputed. The matrix is directed from session 0 rows
to session 1 columns for formal S/C/J calculations.

## Immutable formal inference

The parent four-class angular result remains unchanged: `T_subject={fmt(results['parent']['T_subject'])}`,
`T_class={fmt(results['parent']['T_class'])}`, and `T_J={fmt(results['parent']['T_J'])}`.
Its previously frozen null p-values and positive subject x class conclusion are
not rerun or redefined here.

## Primary formal retrospective decomposition

| pair | T_S | T_C | T_J |
| --- | ---: | ---: | ---: |
{pair_table}

These are supporting algebraic components, not six competing primary tests;
no pairwise p-values were computed. Ranked descriptively by `T_J`: {j_rank_text}.

The exact mean aggregation gate passed. Across every subject and group statistic,
the four-class value equals the arithmetic mean of the six binary-pair values.
Maximum absolute reconstruction error: `{results['maximum_reconstruction_error']:.17g}`.
Group signed residuals were `S={results['group_reconstruction']['T_S']:.17g}`,
`C={results['group_reconstruction']['T_C']:.17g}`, and
`J={results['group_reconstruction']['T_J']:.17g}`. Thus the frozen four-class
interaction is not a Left/Right-only phenomenon; its exact pairwise anatomy is
reported without selecting a replacement endpoint.

Subject concentration is descriptive: the largest subject-level mean angular
`J_s` is S{subject_j['largest_subject']}={fmt(subject_j['largest_value'])}; the
remaining eight subjects average `{fmt(subject_j['remaining_mean'])}`. This
does not change the subject-as-population-unit inference already frozen in the
parent.

## Descriptive subject-fixed class relation anatomy (G_s)

`G_s` symmetrizes the two directed cross-session costs for each within-subject
class pair. The raw six-entry class-relation profiles have mean pairwise
centered correlation `{fmt(g_summary['raw_pairwise_correlation_mean'])}` and
mean leave-one-subject-out commonality `{fmt(g_summary['raw_loo_mean'])}`.
After subtracting the relevant self-instability baselines, these are
`{fmt(g_summary['adjusted_pairwise_correlation_mean'])}` and
`{fmt(g_summary['adjusted_loo_mean'])}`. Pairwise Euclidean distances and all
individual commonality values are saved in the tables.

The across-subject mean raw profile ranks class costs as
{results['g_raw_rank_text']}. The baseline-adjusted rank is
{results['g_adjusted_rank_text']}. This describes the observed breadth of a
common class hierarchy and the relative Left/Right versus hand/nonhand
organization; it is not a new inferential claim.

## Descriptive class-fixed subject relation anatomy (H_c)

The four 36-entry subject-relation profiles have mean pairwise centered
correlation `{fmt(h_summary['raw_pairwise_correlation_mean'])}` and mean
leave-one-class-out commonality `{fmt(h_summary['raw_loo_mean'])}`. Their
baseline-adjusted counterparts are
`{fmt(h_summary['adjusted_pairwise_correlation_mean'])}` and
`{fmt(h_summary['adjusted_loo_mean'])}`. These values quantify how similarly
the frozen classes arrange subject-to-subject costs. They do not establish a
reusable transformation or generative subject factor.

## Integrated anatomy

The formal scalar contrasts and descriptive matrices are complementary. `S`
summarizes subject correspondence, `C` class correspondence, and `J` their
non-additive correspondence contrast. `G_s` displays which class relations are
close or far within each subject; `H_c` displays which subject relations are
close or far within each class. They anatomize but do not replace S/C/J.

The exact pair decomposition, the observed spread of `J_s,p`, the `G_s`
commonality, and the `H_c` commonality should be read together: shared class
ordering or shared subject ordering can coexist with subject x class-specific
deformation. Coarse hand-versus-nonhand organization is assessed descriptively
from LR versus LF/LT/RF/RT/FT costs and contrasts, not elevated into a new
primary test. Static-versus-movement implications are not identified by this
frozen movement matrix alone.

## Result-to-claim decision table

| Observed pattern | What may be said | What may not be said |
| --- | --- | --- |
| Strong G_s similarity across subjects | Evidence of a common class-relation profile | A universal intrinsic class shape or reusable transform |
| Weak G_s but strong H_c similarity | Subject relations look more reusable across classes than class relations across subjects | A generative subject factor |
| Both strong | Both relation profiles show common descriptive organization | S/C/J are redundant or a separable generative model is proven |
| Both weak | Relation profiles are heterogeneous in both views | Subject/class correspondence is absent without reference to frozen formal tests |
| LR weak but hand-vs-nonhand pairs strong | Coarse hand/nonhand organization may dominate this descriptive resolution | LR represents the complete four-class structure |
| J concentrated in LF/LT/RF/RT rather than LR | The exact four-class J is chiefly carried by those pair components | The best pair becomes a new primary endpoint |
| J concentrated in only a few subjects | The group mean has substantial subject-level concentration | Matrix entries are independent replicates or an effect is population-wide |
| Static versus movement contrast | May be discussed only with already-frozen, directly comparable prior evidence | This movement-only anatomy establishes a static/movement distinction |

## Overclaim risks

1. `G_s` and `H_c` are relation matrices or distance profiles, not strongly
   intrinsic manifold shapes.
2. Baseline-adjusted matrices are descriptive, not inferential primaries.
3. Pair outputs must not redefine the parent study conclusion; no pairwise
   p-values were used here.
4. Common relation geometry is not a reusable transformation.
5. S/C/J alone do not identify a generative subject/class decomposition.
6. No single class pair, especially LR, is the whole four-class story.
7. Upstream artifacts must not be refit or silently modified.
8. Trial pairs and matrix entries are not independent inferential units;
   subjects remain the population unit.

## Validation and immutability

- Frozen input reproduction: PASS
- Parent four-class scalar reproduction: PASS
- Six-pair reconstruction: PASS
- Parent hashes unchanged after execution: {str(results['parent_artifacts_unchanged']).lower()}
- Focused pre-result tests: `{results['tests']['focused_before']}`
- Focused post-result tests: `{results['tests']['focused_after']}`
- Full repository tests: `{results['tests']['full_after']}`
- Scientific settings changed after protocol freeze: false
- Runtime: `{results['runtime_seconds']:.6f}` seconds
"""


def write_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv" and not path.name.endswith(".tmp"):
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUTPUT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = OUTPUT / "provenance/artifact_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(manifest.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "size_bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest)


def execute(*, focused_before: str) -> None:
    require_clean_branch()
    started = time.perf_counter()
    parent_hashes_before = artifact_hashes()
    matrix, reproduction = load_frozen_matrix()
    full = relation_statistics(matrix)
    parent_errors = {
        "T_subject": full.t_subject - PARENT_STATISTICS["T_subject"],
        "T_class": full.t_class - PARENT_STATISTICS["T_class"],
        "T_J": full.t_j - PARENT_STATISTICS["T_J"],
    }
    if max(abs(value) for value in parent_errors.values()) > ATOL:
        raise ReconstructionError("parent statistics failed before new anatomy")

    pairs = six_pair_statistics(matrix)
    reconstruction = reconstruction_errors(full, pairs)
    max_reconstruction = maximum_reconstruction_error(reconstruction)
    if max_reconstruction > ATOL:
        raise ReconstructionError(
            f"six-pair reconstruction error {max_reconstruction} exceeds {ATOL}"
        )
    anatomy = build_relation_anatomy(matrix)

    tables = OUTPUT / "tables"
    figures = OUTPUT / "figures"
    arrays = OUTPUT / "arrays"
    provenance = OUTPUT / "provenance"
    report_dir = OUTPUT / "report"
    for directory in (tables, figures, arrays, provenance, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pair_group_rows = []
    pair_subject_rows = []
    pair_anchor_rows = []
    for pair_name, (first, second) in zip(PAIR_NAMES, PAIR_INDICES, strict=True):
        stats = pairs[pair_name]
        pair_group_rows.append(
            {
                "pair": pair_name,
                "first_class": CLASS_ORDER[first],
                "second_class": CLASS_ORDER[second],
                "T_S": stats.t_subject,
                "T_C": stats.t_class,
                "T_J": stats.t_j,
                "inference_role": "supporting_exact_decomposition_no_pairwise_pvalue",
            }
        )
        for subject in range(N_SUBJECTS):
            pair_subject_rows.append(
                {
                    "subject": subject + 1,
                    "pair": pair_name,
                    "S_s": stats.s_s[subject],
                    "C_s": stats.c_s[subject],
                    "J_s": stats.j_s[subject],
                }
            )
            for local_class, class_index in enumerate((first, second)):
                pair_anchor_rows.append(
                    {
                        "subject": subject + 1,
                        "pair": pair_name,
                        "anchor_class": CLASS_ORDER[class_index],
                        "a": stats.a_sc[subject, local_class],
                        "b": stats.b_sc[subject, local_class],
                        "c": stats.c_sc[subject, local_class],
                        "d": stats.d_sc[subject, local_class],
                        "S_sc": stats.s_sc[subject, local_class],
                        "C_sc": stats.c_sc_effect[subject, local_class],
                        "J_sc": stats.j_sc[subject, local_class],
                    }
                )
    pair_group = pd.DataFrame(pair_group_rows)
    pair_subject = pd.DataFrame(pair_subject_rows)
    atomic_csv(tables / "six_pair_group_statistics.csv", pair_group)
    atomic_csv(tables / "six_pair_subject_statistics.csv", pair_subject)
    atomic_csv(tables / "six_pair_relation_cells.csv", pd.DataFrame(pair_anchor_rows))

    reconstruction_rows = []
    pair_banks = {
        "S": np.stack([pairs[name].s_s for name in PAIR_NAMES], axis=1),
        "C": np.stack([pairs[name].c_s for name in PAIR_NAMES], axis=1),
        "J": np.stack([pairs[name].j_s for name in PAIR_NAMES], axis=1),
    }
    full_banks = {"S": full.s_s, "C": full.c_s, "J": full.j_s}
    for metric in ("S", "C", "J"):
        for subject in range(N_SUBJECTS):
            pair_mean = float(np.mean(pair_banks[metric][subject]))
            full_value = float(full_banks[metric][subject])
            reconstruction_rows.append(
                {
                    "level": "subject",
                    "metric": metric,
                    "subject": subject + 1,
                    "four_class_value": full_value,
                    "six_pair_mean": pair_mean,
                    "signed_error": full_value - pair_mean,
                    "absolute_error": abs(full_value - pair_mean),
                }
            )
        full_group = float(np.mean(full_banks[metric]))
        pair_group_mean = float(np.mean(pair_banks[metric]))
        reconstruction_rows.append(
            {
                "level": "group",
                "metric": metric,
                "subject": "all",
                "four_class_value": full_group,
                "six_pair_mean": pair_group_mean,
                "signed_error": full_group - pair_group_mean,
                "absolute_error": abs(full_group - pair_group_mean),
            }
        )
    atomic_csv(tables / "six_pair_reconstruction_checks.csv", pd.DataFrame(reconstruction_rows))

    g_frame = pd.DataFrame(anatomy.g_profiles, columns=PAIR_NAMES)
    g_frame.insert(0, "subject", np.arange(1, N_SUBJECTS + 1))
    delta_g_frame = pd.DataFrame(anatomy.delta_g_profiles, columns=PAIR_NAMES)
    delta_g_frame.insert(0, "subject", np.arange(1, N_SUBJECTS + 1))
    atomic_csv(tables / "g_subject_class_relation_profiles.csv", g_frame)
    atomic_csv(tables / "delta_g_subject_class_relation_profiles.csv", delta_g_frame)
    h_frame = pd.DataFrame(anatomy.h_profiles, columns=SUBJECT_PAIR_NAMES)
    h_frame.insert(0, "class", CLASS_ORDER)
    delta_h_frame = pd.DataFrame(anatomy.delta_h_profiles, columns=SUBJECT_PAIR_NAMES)
    delta_h_frame.insert(0, "class", CLASS_ORDER)
    atomic_csv(tables / "h_class_subject_relation_profiles.csv", h_frame)
    atomic_csv(tables / "delta_h_class_subject_relation_profiles.csv", delta_h_frame)

    matrix_long = pd.concat(
        [
            matrix_long_rows(anatomy.g, [f"S{s}" for s in range(1, 10)], list(CLASS_SHORT), "G"),
            matrix_long_rows(anatomy.delta_g, [f"S{s}" for s in range(1, 10)], list(CLASS_SHORT), "delta_G"),
            matrix_long_rows(anatomy.h, list(CLASS_SHORT), [f"S{s}" for s in range(1, 10)], "H"),
            matrix_long_rows(anatomy.delta_h, list(CLASS_SHORT), [f"S{s}" for s in range(1, 10)], "delta_H"),
        ],
        ignore_index=True,
    )
    atomic_csv(tables / "relation_matrices_long.csv", matrix_long)

    g_pairwise, g_loo, g_summary = similarity_rows(
        "G_subject_fixed_class", [f"S{s}" for s in range(1, 10)], anatomy.g_profiles, anatomy.delta_g_profiles
    )
    h_pairwise, h_loo, h_summary = similarity_rows(
        "H_class_fixed_subject", list(CLASS_SHORT), anatomy.h_profiles, anatomy.delta_h_profiles
    )
    atomic_csv(tables / "profile_pairwise_similarity.csv", pd.concat([g_pairwise, h_pairwise], ignore_index=True))
    atomic_csv(tables / "leave_one_out_commonality.csv", pd.concat([g_loo, h_loo], ignore_index=True))
    combined_summary = pd.concat([g_summary, h_summary], ignore_index=True)
    atomic_csv(tables / "commonality_summary.csv", combined_summary)

    class_profile_summary = pd.DataFrame(
        {
            "pair": PAIR_NAMES,
            "raw_mean": np.mean(anatomy.g_profiles, axis=0),
            "raw_std": np.std(anatomy.g_profiles, axis=0),
            "baseline_adjusted_mean": np.mean(anatomy.delta_g_profiles, axis=0),
            "baseline_adjusted_std": np.std(anatomy.delta_g_profiles, axis=0),
        }
    )
    atomic_csv(tables / "class_relation_profile_summary.csv", class_profile_summary)

    decision_rows = [
        ("strong_G_similarity", "common class-relation profile", "not an intrinsic universal shape or transform"),
        ("weak_G_strong_H", "subject relations appear more reusable across classes", "not a generative subject factor"),
        ("both_strong", "both relation profiles share descriptive organization", "not proof of separability"),
        ("both_weak", "heterogeneity in both descriptive views", "not absence of frozen formal effects"),
        ("LR_weak_hand_nonhand_strong", "coarse hand/nonhand organization may dominate", "not that LR is the four-class story"),
        ("J_in_LF_LT_RF_RT", "four-class J is chiefly carried by those exact components", "not a new best-pair primary"),
        ("J_few_subjects", "group J is subject-concentrated", "not independent-entry inference"),
        ("static_vs_movement", "only with frozen comparable external evidence", "not identifiable from this matrix alone"),
    ]
    atomic_csv(
        tables / "result_to_claim_decision_table.csv",
        pd.DataFrame(decision_rows, columns=("outcome", "allowed_claim", "forbidden_overclaim")),
    )

    atomic_npz(
        arrays / "relation_anatomy_matrices.npz",
        c_ang_matrix=matrix,
        G=anatomy.g,
        delta_G=anatomy.delta_g,
        H=anatomy.h,
        delta_H=anatomy.delta_h,
        g_profiles=anatomy.g_profiles,
        delta_g_profiles=anatomy.delta_g_profiles,
        h_profiles=anatomy.h_profiles,
        delta_h_profiles=anatomy.delta_h_profiles,
        pair_names=np.asarray(PAIR_NAMES),
        class_names=np.asarray(CLASS_ORDER),
        subject_ids=np.arange(1, N_SUBJECTS + 1),
    )

    heatmap_grid(anatomy.g, [f"Subject {s}" for s in range(1, 10)], list(CLASS_SHORT), figures / "G_subject_class_relation_matrices", colorbar_label="squared c_ang cost", centered=False)
    heatmap_grid(anatomy.delta_g, [f"Subject {s}" for s in range(1, 10)], list(CLASS_SHORT), figures / "delta_G_subject_class_relation_matrices", colorbar_label="baseline-adjusted squared cost", centered=True)
    heatmap_grid(anatomy.h, list(CLASS_ORDER), [f"S{s}" for s in range(1, 10)], figures / "H_class_subject_relation_matrices", colorbar_label="squared c_ang cost", centered=False)
    heatmap_grid(anatomy.delta_h, list(CLASS_ORDER), [f"S{s}" for s in range(1, 10)], figures / "delta_H_class_subject_relation_matrices", colorbar_label="baseline-adjusted squared cost", centered=True)
    subject_pair_j_plot(pair_subject, figures / "subject_pair_J_decomposition")

    def summary_values(frame: pd.DataFrame, domain: str) -> dict[str, float]:
        selected = frame[frame["domain"] == domain]
        lookup = {
            (row.profile, row.measure): row.mean for row in selected.itertuples()
        }
        return {
            "raw_pairwise_correlation_mean": float(lookup[("raw", "pairwise_centered_correlation")]),
            "raw_loo_mean": float(lookup[("raw", "leave_one_out_correlation")]),
            "adjusted_pairwise_correlation_mean": float(lookup[("baseline_adjusted", "pairwise_centered_correlation")]),
            "adjusted_loo_mean": float(lookup[("baseline_adjusted", "leave_one_out_correlation")]),
        }

    raw_rank = class_profile_summary.sort_values("raw_mean")
    adjusted_rank = class_profile_summary.sort_values("baseline_adjusted_mean")
    parent_hashes_after = artifact_hashes()
    unchanged = parent_hashes_before == parent_hashes_after
    if not unchanged:
        raise FrozenAngularArtifactError("parent artifacts changed during execution")
    subject_maximum_index = int(np.argmax(full.j_s))
    results = {
        "verdict": "COMPLETED_FROZEN_BNCI_ANGULAR_RELATION_ANATOMY_V0",
        "protocol_freeze_sha": git("rev-parse", "HEAD"),
        "scientific_result_sha": "PENDING_SCIENTIFIC_RESULT_COMMIT",
        "parent": {
            "T_subject": full.t_subject,
            "T_class": full.t_class,
            "T_J": full.t_j,
            "signed_reproduction_errors": parent_errors,
        },
        "pair_group_statistics": pair_group_rows,
        "maximum_reconstruction_error": max_reconstruction,
        "group_reconstruction": {
            "T_S": float(reconstruction["T_S"]),
            "T_C": float(reconstruction["T_C"]),
            "T_J": float(reconstruction["T_J"]),
        },
        "commonality_summary": {
            "G": summary_values(combined_summary, "G_subject_fixed_class"),
            "H": summary_values(combined_summary, "H_class_fixed_subject"),
        },
        "g_raw_rank_text": ", ".join(
            f"{row.pair}={fmt(row.raw_mean)}" for row in raw_rank.itertuples()
        ),
        "g_adjusted_rank_text": ", ".join(
            f"{row.pair}={fmt(row.baseline_adjusted_mean)}" for row in adjusted_rank.itertuples()
        ),
        "subject_J_summary": {
            "largest_subject": subject_maximum_index + 1,
            "largest_value": float(full.j_s[subject_maximum_index]),
            "remaining_mean": float(np.mean(np.delete(full.j_s, subject_maximum_index))),
        },
        "parent_artifacts_unchanged": unchanged,
        "parent_hashes_before": parent_hashes_before,
        "parent_hashes_after": parent_hashes_after,
        "input_reproduction": reproduction,
        "protocol_source_hash": protocol_hash(),
        "runtime_seconds": time.perf_counter() - started,
        "tests": {"focused_before": focused_before, "focused_after": "PENDING", "full_after": "PENDING"},
        "scientific_settings_changed_after_protocol_freeze": False,
        "optional_permutation_nulls_run": False,
    }
    atomic_json(provenance / "scientific_results.json", results)
    atomic_json(
        provenance / "parent_artifact_immutability.json",
        {"status": "PASS", "unchanged": unchanged, "before": parent_hashes_before, "after": parent_hashes_after},
    )
    atomic_json(
        provenance / "environment.json",
        {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    )
    atomic_text(REPORT, render_report(results, "PENDING_SCIENTIFIC_RESULT_COMMIT"))
    write_manifest()
    print(json.dumps({
        "verdict": results["verdict"],
        "T_J": full.t_j,
        "maximum_reconstruction_error": max_reconstruction,
        "pair_T_J": {row["pair"]: row["T_J"] for row in pair_group_rows},
        "G_commonality": results["commonality_summary"]["G"],
        "H_commonality": results["commonality_summary"]["H"],
    }, indent=2))


def record_tests(*, focused_after: str, full_after: str) -> None:
    path = OUTPUT / "provenance/scientific_results.json"
    results = json.loads(path.read_text())
    results["tests"]["focused_after"] = focused_after
    results["tests"]["full_after"] = full_after
    atomic_json(path, results)
    atomic_text(REPORT, render_report(results, results["scientific_result_sha"]))
    write_manifest()


def finalize(*, scientific_result_sha: str) -> None:
    path = OUTPUT / "provenance/scientific_results.json"
    results = json.loads(path.read_text())
    if results["scientific_result_sha"] != "PENDING_SCIENTIFIC_RESULT_COMMIT":
        raise RuntimeError("scientific result SHA already finalized")
    results["scientific_result_sha"] = scientific_result_sha
    atomic_json(path, results)
    atomic_text(REPORT, render_report(results, scientific_result_sha))
    write_manifest()


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
            parser.error("--scientific-result-sha is required for finalize")
        finalize(scientific_result_sha=arguments.scientific_result_sha)


if __name__ == "__main__":
    main()
