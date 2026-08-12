#!/usr/bin/env python3
"""Run the protocol-frozen BNCI angular dual relation anatomy V0."""

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

from src.bnci_angular_dual_relation_anatomy_v0 import (  # noqa: E402
    CLASS_ORDER,
    CLASS_SHORT,
    N_CLASSES,
    N_SUBJECTS,
    PAIR_INDICES,
    PAIR_NAMES,
    SUBJECT_PAIR_NAMES,
    build_dual_anatomy,
    canonical_classes,
    canonical_subjects,
    coarse_effector_boundary_contrast,
    leave_one_out_commonality,
    leave_one_subject_influence,
    maximum_reconstruction_error,
    pairwise_profile_commonality,
    reconstruction_errors,
    relation_statistics,
    six_pair_statistics,
)


EXPECTED_BRANCH = "audit/bnci-angular-dual-relation-anatomy-v0"
PARENT_BRANCH = "pilot/local-movement-component-decomposition-v0"
PARENT_HEAD = "edc1d344cb0657f2f2d87b2992049bceec4705d2"
PARENT_PROTOCOL_FREEZE = "95c330de9596fa4c4eb4ee377d5af8d99896f4c3"
PARENT_SCIENTIFIC_RESULT = "0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca"
PARENT_STATISTICS = {
    "T_subject": 0.3091561771980925,
    "T_class": 0.39309843397343514,
    "T_J": 0.19240885452534362,
}
PARENT_PVALUES = {"p_J_subjectbreak": 0.001, "p_J_classbreak": 0.0105}
LR_AUDIT = {
    "pr": 12,
    "head": "7e692749e8b14ab1792d7175443ab476676fcb14",
    "T_subject": 0.36482154482234475,
    "p_subject": 0.0005,
    "T_class": 0.15656493093976115,
    "p_class": 0.158,
    "T_J": 0.11912929182411669,
    "p_J_subjectbreak": 0.09,
    "p_J_classbreak": 0.3065,
    "terminal": "BNCI_LR_ANGULAR_INTERACTION_NOT_SUPPORTED",
}
PARENT_OUTPUT = ROOT / "outputs/bnci2014_001_local_movement_component_decomposition_v0"
FROZEN_ARRAY = PARENT_OUTPUT / "arrays/component_cost_matrices.npz"
FROZEN_TABLE = PARENT_OUTPUT / "tables/c_ang_matrix.csv"
FROZEN_MANIFEST = PARENT_OUTPUT / "protocol/artifact_manifest.csv"
EXPECTED_HASHES = {
    FROZEN_ARRAY: "51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091",
    FROZEN_TABLE: "f6c06c3f44807207d7baf0d84226859380e008b1d27a47572f9b94d0dc6735bd",
    FROZEN_MANIFEST: "c3aea494fedee8af1ee42a5c26ff94fc5bfae5e5289549bc878c3941da43df79",
}
OUTPUT = ROOT / "outputs/bnci2014_001_angular_dual_relation_anatomy_v0"
REPORT = OUTPUT / "report/bnci2014_001_angular_dual_relation_anatomy_v0.md"
CONFIG = ROOT / "configs/bnci2014_001_angular_dual_relation_anatomy_v0.yaml"
PROTOCOL = ROOT / "docs/PROTOCOL_BNCI_ANGULAR_DUAL_RELATION_ANATOMY_V0.md"
IMPLEMENTATION = (
    ROOT / "src/bnci_angular_dual_relation_anatomy_v0.py",
    Path(__file__),
    ROOT / "tests/test_bnci_angular_dual_relation_anatomy_v0.py",
    ROOT / "tests/test_protocol_bnci_angular_dual_relation_anatomy_v0.py",
    ROOT / "tests/test_bnci_angular_dual_relation_anatomy_outputs_v0.py",
)
ATOL = 1.0e-12
RTOL = 1.0e-12


class ParentReproductionFailure(RuntimeError):
    pass


class SixPairReconstructionFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def require_clean_execution_state() -> None:
    branch = git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"expected {EXPECTED_BRANCH}, found {branch}")
    if git("status", "--porcelain=v1"):
        raise RuntimeError("scientific execution requires a clean protocol-frozen worktree")


def protocol_source_hash() -> str:
    digest = hashlib.sha256()
    for path in (PROTOCOL, CONFIG, *IMPLEMENTATION):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verified_parent_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise ParentReproductionFailure(f"missing frozen artifact {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ParentReproductionFailure(
                f"hash mismatch {path}: {observed} != {expected}"
            )
        hashes[str(path.relative_to(ROOT))] = observed
    return hashes


def load_frozen_D() -> tuple[np.ndarray, dict[str, Any]]:
    hashes = verified_parent_hashes()
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
            raise ParentReproductionFailure("unexpected frozen NPZ key set")
        matrix = np.array(archive["c_ang_matrix"], dtype=np.float64, copy=True)
        if not np.array_equal(archive["cell_subjects"], canonical_subjects()):
            raise ParentReproductionFailure("canonical parent subjects mismatch")
        if not np.array_equal(archive["cell_classes"], canonical_classes()):
            raise ParentReproductionFailure("canonical parent classes mismatch")
        if not np.array_equal(matrix, archive["c_full_matrix"] - archive["c_len_matrix"]):
            raise ParentReproductionFailure("c_ang is not exact c_full-c_len")
    if matrix.shape != (36, 36) or matrix.dtype != np.float64 or not np.isfinite(matrix).all():
        raise ParentReproductionFailure("frozen D shape/dtype/finite gate failed")
    csv_values = pd.read_csv(FROZEN_TABLE, index_col=0).to_numpy(dtype=np.float64)
    csv_error = float(np.max(np.abs(csv_values - matrix)))
    if csv_error > 1e-15:
        raise ParentReproductionFailure("frozen readable matrix exceeds text tolerance")
    return matrix, {
        "status": "PASS",
        "hashes": hashes,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "canonical_subjects": True,
        "canonical_classes": True,
        "exact_c_full_minus_c_len": True,
        "csv_maximum_absolute_roundtrip_error": csv_error,
    }


def reproduce_parent(matrix: np.ndarray) -> tuple[Any, dict[str, float]]:
    statistics = relation_statistics(matrix, n_classes=4)
    errors = {
        "T_subject": statistics.t_subject - PARENT_STATISTICS["T_subject"],
        "T_class": statistics.t_class - PARENT_STATISTICS["T_class"],
        "T_J": statistics.t_j - PARENT_STATISTICS["T_J"],
    }
    if max(abs(value) for value in errors.values()) > ATOL:
        raise ParentReproductionFailure("frozen K=4 statistics did not reproduce")
    return statistics, errors


def prepare() -> None:
    matrix, reproduction = load_frozen_D()
    statistics, errors = reproduce_parent(matrix)
    reproduction.update(
        {
            "parent_statistics": {
                "observed": {
                    "T_subject": statistics.t_subject,
                    "T_class": statistics.t_class,
                    "T_J": statistics.t_j,
                },
                "expected": PARENT_STATISTICS,
                "signed_errors": errors,
                "maximum_absolute_error": max(abs(value) for value in errors.values()),
            },
            "protocol_source_hash": protocol_source_hash(),
        }
    )
    atomic_json(OUTPUT / "provenance/pre_result_reproduction.json", reproduction)
    print(json.dumps(reproduction["parent_statistics"], indent=2))


def correlation_rows(
    domain: str, labels: list[str], raw: np.ndarray, adjusted: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pairwise_rows = []
    leave_rows = []
    summary_rows = []
    details: dict[str, Any] = {}
    for kind, values in (("raw", raw), ("adjusted", adjusted)):
        correlations, cosines, distances = pairwise_profile_commonality(values)
        upper = np.triu_indices(len(labels), k=1)
        for first, second in zip(*upper, strict=True):
            pairwise_rows.append(
                {
                    "domain": domain,
                    "profile": kind,
                    "first": labels[first],
                    "second": labels[second],
                    "pearson_correlation": correlations[first, second],
                    "centered_cosine_similarity": cosines[first, second],
                    "euclidean_distance": distances[first, second],
                }
            )
        loo_corr, loo_cos = leave_one_out_commonality(values)
        for label, correlation, cosine in zip(labels, loo_corr, loo_cos, strict=True):
            leave_rows.append(
                {
                    "domain": domain,
                    "profile": kind,
                    "label": label,
                    "leave_out_pearson_correlation": correlation,
                    "leave_out_centered_cosine_similarity": cosine,
                }
            )
        for measure, vector in (
            ("pairwise_pearson_correlation", correlations[upper]),
            ("pairwise_centered_cosine_similarity", cosines[upper]),
            ("pairwise_euclidean_distance", distances[upper]),
            ("leave_out_pearson_correlation", loo_corr),
            ("leave_out_centered_cosine_similarity", loo_cos),
        ):
            summary_rows.append(
                {
                    "domain": domain,
                    "profile": kind,
                    "measure": measure,
                    "count": len(vector),
                    "mean": float(np.nanmean(vector)),
                    "median": float(np.nanmedian(vector)),
                    "minimum": float(np.nanmin(vector)),
                    "maximum": float(np.nanmax(vector)),
                }
            )
        details[kind] = {
            "pairwise_pearson_mean": float(np.nanmean(correlations[upper])),
            "pairwise_pearson_median": float(np.nanmedian(correlations[upper])),
            "pairwise_pearson_minimum": float(np.nanmin(correlations[upper])),
            "pairwise_pearson_maximum": float(np.nanmax(correlations[upper])),
            "leave_out_pearson_mean": float(np.nanmean(loo_corr)),
            "leave_out_by_label": {
                label: float(value) for label, value in zip(labels, loo_corr, strict=True)
            },
            "maximum_pearson_centered_cosine_error": float(
                max(
                    np.nanmax(np.abs(correlations[upper] - cosines[upper])),
                    np.nanmax(np.abs(loo_corr - loo_cos)),
                )
            ),
        }
    return (
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(leave_rows),
        pd.DataFrame(summary_rows),
        details,
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def profile_heatmap(
    values: np.ndarray,
    title: str,
    stem: Path,
    *,
    centered: bool,
    colorbar_label: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8.3, 5.4), layout="constrained")
    if centered:
        bound = float(np.max(np.abs(values)))
        image = axis.imshow(values, cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
    else:
        image = axis.imshow(values, cmap="viridis", aspect="auto")
    axis.set_xticks(range(6), PAIR_NAMES)
    axis.set_yticks(range(9), [f"S{s}" for s in range(1, 10)])
    axis.set_xlabel("Class pair")
    axis.set_ylabel("Subject")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    save_figure(figure, stem)


def group_bar_figure(frame: pd.DataFrame, stem: Path) -> None:
    x = np.arange(len(PAIR_NAMES), dtype=np.float64)
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.0, 5.2), layout="constrained")
    for offset, column, label, color in (
        (-width, "T_S", "T_S", "#3A6EA5"),
        (0.0, "T_C", "T_C", "#D9822B"),
        (width, "T_J", "T_J", "#4C956C"),
    ):
        axis.bar(x + offset, frame[column], width, label=label, color=color)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, PAIR_NAMES)
    axis.set_xlabel("Binary class pair")
    axis.set_ylabel("Squared-cost relation contrast")
    axis.set_title("Exact six-pair group decomposition (descriptive magnitudes)")
    axis.legend(frameon=False)
    axis.text(
        0.01,
        0.99,
        "No pairwise significance encoding",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )
    save_figure(figure, stem)


def panel_heatmaps(values: np.ndarray, titles: list[str], stem: Path, *, centered: bool) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 9.3), layout="constrained")
    if centered:
        bound = float(np.max(np.abs(values)))
        minimum, maximum, cmap = -bound, bound, "coolwarm"
        label = "baseline-adjusted squared angular cost"
    else:
        minimum, maximum, cmap = float(np.min(values)), float(np.max(values)), "viridis"
        label = "squared angular cost"
    image = None
    for class_index, axis in enumerate(axes.ravel()):
        image = axis.imshow(values[class_index], cmap=cmap, vmin=minimum, vmax=maximum)
        axis.set_title(titles[class_index])
        axis.set_xticks(range(9), [f"S{s}" for s in range(1, 10)], rotation=45, ha="right")
        axis.set_yticks(range(9), [f"S{s}" for s in range(1, 10)])
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02, label=label)
    save_figure(figure, stem)


def matrix_long(values: np.ndarray, *, adjusted: bool) -> pd.DataFrame:
    rows = []
    for class_index, class_name in enumerate(CLASS_ORDER):
        for first in range(N_SUBJECTS):
            for second in range(N_SUBJECTS):
                rows.append(
                    {
                        "class": class_name,
                        "subject_row": first + 1,
                        "subject_column": second + 1,
                        "value": values[class_index, first, second],
                        "off_diagonal": first != second,
                        "profile": "adjusted" if adjusted else "raw",
                    }
                )
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    return f"{value:.10g}"


def render_report(results: dict[str, Any], scientific_sha: str) -> str:
    pair_rows = results["six_pair_group"]
    pair_table = "\n".join(
        f"| {row['pair']} | {fmt(row['T_S'])} | {fmt(row['T_C'])} | {fmt(row['T_J'])} | {fmt(row['fraction_of_total_J_if_defined'])} |"
        for row in pair_rows
    )
    pair_j = {row["pair"]: row["T_J"] for row in pair_rows}
    subject_patterns = "; ".join(
        f"S{row['subject']}:{row['largest_J_pair']}={fmt(row['largest_J'])}"
        for row in results["subject_pair_patterns"]
    )
    g = results["commonality"]["G"]
    h = results["commonality"]["H"]
    raw_order = ", ".join(
        f"{name}={fmt(value)}" for name, value in results["mean_G_raw_order"]
    )
    adjusted_order = ", ".join(
        f"{name}={fmt(value)}" for name, value in results["mean_G_adjusted_order"]
    )
    h_loo = ", ".join(
        f"{name}={fmt(value)}" for name, value in h["raw"]["leave_out_by_label"].items()
    )
    return f"""# BNCI Angular Six-Pair and Dual Relation Anatomy V0

## Verdict and provenance

`COMPLETED_BNCI_ANGULAR_DUAL_RELATION_ANATOMY_V0`

- Status: retrospective anatomy, not prospective or confirmatory
- Branch: `{EXPECTED_BRANCH}`
- Exact parent: `{PARENT_HEAD}`
- Parent protocol freeze: `{PARENT_PROTOCOL_FREEZE}`
- Parent scientific result: `{PARENT_SCIENTIFIC_RESULT}`
- Protocol freeze SHA: `{results['protocol_freeze_sha']}`
- Scientific result SHA: `{scientific_sha}`
- Frozen D SHA-256: `{EXPECTED_HASHES[FROZEN_ARRAY]}`

Only the frozen directed 36 by 36 squared angular cost matrix was used. No EEG,
covariance mean, anti-development, movement tuple, AIRM mean, or quotient
optimizer was fitted or recomputed.

## Immutable formal prior evidence

The frozen four-class result remains `T_subject={fmt(results['parent']['T_subject'])}`,
`T_class={fmt(results['parent']['T_class'])}`, and
`T_J={fmt(results['parent']['T_J'])}` with subject-break `p=0.001` and
class-break `p=0.0105`. This inference is immutable and is not rerun or
redefined by the descriptive anatomy.

## Formal retrospective six-pair decomposition

| pair | T_S | T_C | T_J | exact fraction of four-class J |
| --- | ---: | ---: | ---: | ---: |
{pair_table}

Fractions are `(T_J,p/6)/T_J,4c` and sum to one; they are additive accounting,
not post-hoc inferential weights. No pairwise p-values were computed.

The hard mean-aggregation gate passed for every subject and group statistic.
Maximum absolute reconstruction error was
`{results['maximum_reconstruction_error']:.17g}`; group signed errors were
`S={results['group_reconstruction']['T_S']:.17g}`,
`C={results['group_reconstruction']['T_C']:.17g}`, and
`J={results['group_reconstruction']['T_J']:.17g}`. The reconstructed four-class
`T_J={fmt(results['parent']['T_J'])}` matches the frozen parent.

## Supporting retrospective Left/Right result

PR #12 remains supporting evidence: `T_subject=0.3648215448, p=0.0005`,
`T_class=0.1565649309, p=0.158`, and `T_J=0.1191292918` with subject-break
`p=0.09` and class-break `p=0.3065`. Its terminal remains
`BNCI_LR_ANGULAR_INTERACTION_NOT_SUPPORTED`. The six-pair anatomy does not turn
LR into a new primary endpoint.

## Q1 — Within-subject class relations

Across-subject mean raw G relations, ordered small to large, are {raw_order}.
The adjusted ordering is {adjusted_order}. The most frequent raw smallest pair
and per-subject orderings are saved in the profile tables. These patterns show
both shared tendencies and subject-specific deviations; G is a relation matrix,
not an intrinsic metric geometry.

## Q2 — Common class-relation backbone

Raw G profiles have mean pairwise Pearson correlation
`{fmt(g['raw']['pairwise_pearson_mean'])}` (median
`{fmt(g['raw']['pairwise_pearson_median'])}`, range
`{fmt(g['raw']['pairwise_pearson_minimum'])}` to
`{fmt(g['raw']['pairwise_pearson_maximum'])}`) and mean leave-one-subject-out
correlation `{fmt(g['raw']['leave_out_pearson_mean'])}`. Adjusted values are
`{fmt(g['adjusted']['pairwise_pearson_mean'])}` and
`{fmt(g['adjusted']['leave_out_pearson_mean'])}`. Centered cosine gives the
same values up to maximum recorded numerical error
`{g['maximum_pearson_centered_cosine_error']:.3g}`. Thus a partial class
backbone is descriptively present, but it is heterogeneous rather than a
universal ordering.

## Q3/Q4 — Subject relations within and across classes

Raw H profiles have mean class-to-class Pearson correlation
`{fmt(h['raw']['pairwise_pearson_mean'])}` and leave-one-class-out mean
`{fmt(h['raw']['leave_out_pearson_mean'])}`; adjusted values are
`{fmt(h['adjusted']['pairwise_pearson_mean'])}` and
`{fmt(h['adjusted']['leave_out_pearson_mean'])}`. Raw class-specific leave-out
correlations are {h_loo}. Subject-pair ordering is therefore partly reused but
class-dependent. This is descriptive relational reuse, not evidence for a
single reusable subject transformation.

## Subject-level interaction anatomy

Largest positive pair contribution by subject: {subject_patterns}.
The largest four-class subject value is S{results['subject_concentration']['largest_subject']}=
`{fmt(results['subject_concentration']['largest_value'])}`; the other eight
average `{fmt(results['subject_concentration']['remaining_mean'])}`. LR is
positive in `{results['LR_subject_summary']['positive_count']}/9` subjects and
has median `J_s,LR={fmt(results['LR_subject_summary']['median'])}`. The 9 by 6
heatmap shows that the group result is substantially concentrated and that
different subjects contribute through different pairs. No subject is removed,
and leave-one-subject values are descriptive only.

## Coarse effector-boundary anatomy

The prespecified descriptive contrast is `K_J={fmt(results['K_J'])}`. The mean
cross hand/nonhand pair `T_J` is `{fmt(results['cross_effector_mean_T_J'])}`
versus `{fmt(results['within_group_mean_T_J'])}` for LR/FT. This does not make
feet and tongue biologically homogeneous and has no new p-value. Pair values
show that a simple coarse hierarchy is not a complete account of the frozen
interaction.

## Pattern assessment

- **Pattern A, common class backbone:** partially present, with substantial
  subject heterogeneity.
- **Pattern B, common subject backbone:** partially present and class-dependent;
  it does not identify a reusable transformation.
- **Pattern C, heterogeneous subject-by-class deformation:** strongly visible
  in the spread of `J_s,p` and subject influence.
- **Pattern D, coarse effector hierarchy:** `K_J` is
  `{results['K_J_direction']}`. Pair heterogeneity determines whether that
  coarse balance is a useful partial summary; it cannot be a complete
  explanation by itself.

Multiple patterns coexist; none is forced into a replacement primary result.
The descriptive G/H deviations are not themselves formal interaction tests.

## Validation

- Parent artifact and K=4 reproduction: PASS
- Six-pair subject/group reconstruction: PASS
- Symmetric A and G/H index/diagonal gates: PASS
- Parent artifacts unchanged: {str(results['parent_artifacts_unchanged']).lower()}
- Focused pre-result tests: `{results['tests']['focused_before']}`
- Focused post-result tests: `{results['tests']['focused_after']}`
- Full repository tests: `{results['tests']['full_after']}`
- Scientific setting changed after protocol freeze: false
- Runtime: `{results['runtime_seconds']:.6f}` seconds

The outputs concern frozen window-wise mean-covariance movement relation costs.
They do not establish physiology, cortical direction, causal motor strategy,
intrinsic manifold shape, or a generative subject/class factor.
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
    path = OUTPUT / "provenance/artifact_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("relative_path", "size_bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def execute(*, focused_before: str) -> None:
    require_clean_execution_state()
    started = time.perf_counter()
    before = verified_parent_hashes()
    D, reproduction = load_frozen_D()
    full, parent_errors = reproduce_parent(D)
    pairs, subset_indices = six_pair_statistics(D)
    errors = reconstruction_errors(full, pairs)
    maximum_error = maximum_reconstruction_error(errors)
    if maximum_error > ATOL or abs(full.t_j - PARENT_STATISTICS["T_J"]) > ATOL:
        raise SixPairReconstructionFailure(
            f"maximum six-pair reconstruction error {maximum_error}"
        )
    dual = build_dual_anatomy(D)
    if not np.array_equal(dual.symmetric_a, dual.symmetric_a.T):
        raise SixPairReconstructionFailure("A is not exactly symmetric")
    diagonal_error = max(
        abs(dual.g[subject, cls, cls] - dual.h[cls, subject, subject])
        for subject in range(N_SUBJECTS)
        for cls in range(N_CLASSES)
    )
    if diagonal_error > ATOL:
        raise SixPairReconstructionFailure("G/H diagonal consistency failed")

    tables = OUTPUT / "tables"
    figures = OUTPUT / "figures"
    arrays = OUTPUT / "arrays"
    provenance = OUTPUT / "provenance"
    for directory in (tables, figures, arrays, provenance, REPORT.parent):
        directory.mkdir(parents=True, exist_ok=True)

    pair_t_j = {name: pairs[name].t_j for name in PAIR_NAMES}
    fraction_denominator = 6.0 * full.t_j
    group_rows = []
    subject_rows = []
    J_bank = np.empty((N_SUBJECTS, len(PAIR_NAMES)), dtype=np.float64)
    S_bank = np.empty_like(J_bank)
    C_bank = np.empty_like(J_bank)
    for pair_index, (name, class_pair) in enumerate(zip(PAIR_NAMES, PAIR_INDICES, strict=True)):
        stats = pairs[name]
        group_rows.append(
            {
                "pair": name,
                "first_class": CLASS_ORDER[class_pair[0]],
                "second_class": CLASS_ORDER[class_pair[1]],
                "T_S": stats.t_subject,
                "T_C": stats.t_class,
                "T_J": stats.t_j,
                "fraction_of_total_J_if_defined": stats.t_j / fraction_denominator
                if full.t_j != 0.0
                else np.nan,
            }
        )
        S_bank[:, pair_index] = stats.s_s
        C_bank[:, pair_index] = stats.class_s
        J_bank[:, pair_index] = stats.j_s
        for subject in range(N_SUBJECTS):
            subject_rows.append(
                {
                    "subject": subject + 1,
                    "pair": name,
                    "S_s": stats.s_s[subject],
                    "C_s": stats.class_s[subject],
                    "J_s": stats.j_s[subject],
                }
            )
    group_frame = pd.DataFrame(group_rows)
    subject_frame = pd.DataFrame(subject_rows)
    atomic_csv(tables / "six_pair_group_decomposition.csv", group_frame)
    atomic_csv(tables / "six_pair_subject_decomposition.csv", subject_frame)

    reconstruction_rows = []
    for metric, full_values, pair_values in (
        ("S", full.s_s, S_bank),
        ("C", full.class_s, C_bank),
        ("J", full.j_s, J_bank),
    ):
        for subject in range(N_SUBJECTS):
            pair_mean = float(np.mean(pair_values[subject]))
            reconstruction_rows.append(
                {
                    "level": "subject",
                    "metric": metric,
                    "subject": subject + 1,
                    "four_class_value": full_values[subject],
                    "six_pair_mean": pair_mean,
                    "signed_error": full_values[subject] - pair_mean,
                    "absolute_error": abs(full_values[subject] - pair_mean),
                }
            )
        pair_mean = float(np.mean(pair_values))
        full_mean = float(np.mean(full_values))
        reconstruction_rows.append(
            {
                "level": "group",
                "metric": metric,
                "subject": "all",
                "four_class_value": full_mean,
                "six_pair_mean": pair_mean,
                "signed_error": full_mean - pair_mean,
                "absolute_error": abs(full_mean - pair_mean),
            }
        )
    atomic_csv(
        tables / "six_pair_reconstruction_checks.csv",
        pd.DataFrame(reconstruction_rows),
    )

    raw_g = pd.DataFrame(dual.g_profiles, columns=PAIR_NAMES)
    raw_g.insert(0, "subject", np.arange(1, N_SUBJECTS + 1))
    adjusted_g = pd.DataFrame(dual.delta_g_profiles, columns=PAIR_NAMES)
    adjusted_g.insert(0, "subject", np.arange(1, N_SUBJECTS + 1))
    atomic_csv(tables / "subject_class_relation_profiles_raw.csv", raw_g)
    atomic_csv(tables / "subject_class_relation_profiles_adjusted.csv", adjusted_g)
    atomic_csv(tables / "class_subject_relation_profiles_raw.csv", matrix_long(dual.h, adjusted=False))
    atomic_csv(tables / "class_subject_relation_profiles_adjusted.csv", matrix_long(dual.delta_h, adjusted=True))

    g_pairwise, g_leave, g_summary, g_details = correlation_rows(
        "G_subject_fixed_class",
        [f"S{s}" for s in range(1, 10)],
        dual.g_profiles,
        dual.delta_g_profiles,
    )
    h_pairwise, h_leave, h_summary, h_details = correlation_rows(
        "H_class_fixed_subject",
        list(CLASS_SHORT),
        dual.h_profiles,
        dual.delta_h_profiles,
    )
    class_commonality = g_leave.pivot(index="label", columns="profile", values=["leave_out_pearson_correlation", "leave_out_centered_cosine_similarity"])
    class_commonality.columns = [f"{profile}_{measure}" for measure, profile in class_commonality.columns]
    class_commonality = class_commonality.reset_index().rename(columns={"label": "subject"})
    subject_commonality = h_leave.pivot(index="label", columns="profile", values=["leave_out_pearson_correlation", "leave_out_centered_cosine_similarity"])
    subject_commonality.columns = [f"{profile}_{measure}" for measure, profile in subject_commonality.columns]
    subject_commonality = subject_commonality.reset_index().rename(columns={"label": "class"})
    atomic_csv(tables / "class_profile_commonality.csv", class_commonality)
    atomic_csv(tables / "subject_profile_commonality.csv", subject_commonality)
    atomic_csv(tables / "class_profile_pairwise_similarity.csv", g_pairwise)
    atomic_csv(tables / "subject_profile_pairwise_similarity.csv", h_pairwise)
    atomic_csv(tables / "commonality_summary.csv", pd.concat([g_summary, h_summary], ignore_index=True))

    leave_values, influence = leave_one_subject_influence(full.j_s)
    influence_frame = pd.DataFrame(
        {
            "excluded_subject": np.arange(1, N_SUBJECTS + 1),
            "full_T_J": full.t_j,
            "leave_one_subject_out_T_J": leave_values,
            "full_minus_leave_one_out": influence,
            "role": "descriptive_only",
        }
    )
    atomic_csv(tables / "leave_one_subject_influence.csv", influence_frame)

    K_J = coarse_effector_boundary_contrast(pair_t_j)
    cross_mean = float(np.mean([pair_t_j[name] for name in ("LF", "LT", "RF", "RT")]))
    within_mean = float(np.mean([pair_t_j[name] for name in ("LR", "FT")]))
    atomic_csv(
        tables / "coarse_effector_boundary_contrast.csv",
        pd.DataFrame(
            [
                {
                    "K_J": K_J,
                    "cross_effector_mean_T_J": cross_mean,
                    "LR_FT_mean_T_J": within_mean,
                    "role": "descriptive_no_pvalue",
                }
            ]
        ),
    )

    order_rows = []
    subject_patterns = []
    for subject in range(N_SUBJECTS):
        raw_order_indices = np.argsort(dual.g_profiles[subject])
        adjusted_order_indices = np.argsort(dual.delta_g_profiles[subject])
        largest_index = int(np.argmax(J_bank[subject]))
        subject_patterns.append(
            {
                "subject": subject + 1,
                "largest_J_pair": PAIR_NAMES[largest_index],
                "largest_J": float(J_bank[subject, largest_index]),
            }
        )
        order_rows.append(
            {
                "subject": subject + 1,
                "raw_pair_order_small_to_large": "<".join(PAIR_NAMES[index] for index in raw_order_indices),
                "adjusted_pair_order_small_to_large": "<".join(PAIR_NAMES[index] for index in adjusted_order_indices),
                "largest_J_pair": PAIR_NAMES[largest_index],
                "largest_J": J_bank[subject, largest_index],
            }
        )
    atomic_csv(tables / "subject_pair_orderings.csv", pd.DataFrame(order_rows))

    atomic_npz(
        arrays / "dual_relation_anatomy.npz",
        D=D,
        A=dual.symmetric_a,
        G=dual.g,
        delta_G=dual.delta_g,
        H=dual.h,
        delta_H=dual.delta_h,
        g_profiles=dual.g_profiles,
        delta_g_profiles=dual.delta_g_profiles,
        h_profiles=dual.h_profiles,
        delta_h_profiles=dual.delta_h_profiles,
        S_subject_pair=S_bank,
        C_subject_pair=C_bank,
        J_subject_pair=J_bank,
        pair_names=np.asarray(PAIR_NAMES),
        class_names=np.asarray(CLASS_ORDER),
        subject_ids=np.arange(1, N_SUBJECTS + 1),
    )

    profile_heatmap(
        dual.g_profiles,
        "Subject-fixed class-relation profiles",
        figures / "figure1_G_profiles_raw",
        centered=False,
        colorbar_label="squared angular cost",
    )
    profile_heatmap(
        dual.delta_g_profiles,
        "Baseline-adjusted subject-fixed class relations",
        figures / "figure2_G_profiles_adjusted",
        centered=True,
        colorbar_label="baseline-adjusted squared angular cost",
    )
    group_bar_figure(group_frame, figures / "figure3_six_pair_group_decomposition")
    profile_heatmap(
        J_bank,
        "Subject by class-pair angular interaction contributions",
        figures / "figure4_J_subject_pair",
        centered=True,
        colorbar_label="J_s,p squared-cost contrast",
    )
    panel_heatmaps(dual.h, list(CLASS_ORDER), figures / "figure5_H_subject_relations", centered=False)
    panel_heatmaps(dual.delta_h, list(CLASS_ORDER), figures / "figure6_H_subject_relations_adjusted", centered=True)

    after = verified_parent_hashes()
    if before != after:
        raise ParentReproductionFailure("parent artifacts changed during execution")
    largest_subject_index = int(np.argmax(full.j_s))
    raw_means = np.mean(dual.g_profiles, axis=0)
    adjusted_means = np.mean(dual.delta_g_profiles, axis=0)
    max_commonality_error = max(
        g_details["raw"]["maximum_pearson_centered_cosine_error"],
        g_details["adjusted"]["maximum_pearson_centered_cosine_error"],
        h_details["raw"]["maximum_pearson_centered_cosine_error"],
        h_details["adjusted"]["maximum_pearson_centered_cosine_error"],
    )
    g_details["maximum_pearson_centered_cosine_error"] = max(
        g_details["raw"]["maximum_pearson_centered_cosine_error"],
        g_details["adjusted"]["maximum_pearson_centered_cosine_error"],
    )
    h_details["maximum_pearson_centered_cosine_error"] = max(
        h_details["raw"]["maximum_pearson_centered_cosine_error"],
        h_details["adjusted"]["maximum_pearson_centered_cosine_error"],
    )
    results = {
        "verdict": "COMPLETED_BNCI_ANGULAR_DUAL_RELATION_ANATOMY_V0",
        "protocol_freeze_sha": git("rev-parse", "HEAD"),
        "scientific_result_sha": "PENDING_SCIENTIFIC_RESULT_COMMIT",
        "parent": {
            "T_subject": full.t_subject,
            "T_class": full.t_class,
            "T_J": full.t_j,
            **PARENT_PVALUES,
            "signed_reproduction_errors": parent_errors,
        },
        "supporting_lr_audit": LR_AUDIT,
        "six_pair_group": group_rows,
        "maximum_reconstruction_error": maximum_error,
        "group_reconstruction": {
            "T_S": float(errors["T_S"]),
            "T_C": float(errors["T_C"]),
            "T_J": float(errors["T_J"]),
        },
        "subset_indices": {name: value.tolist() for name, value in subset_indices.items()},
        "G_H_diagonal_maximum_absolute_error": float(diagonal_error),
        "commonality": {"G": g_details, "H": h_details},
        "maximum_pearson_centered_cosine_error": max_commonality_error,
        "mean_G_raw_order": [
            [PAIR_NAMES[index], float(raw_means[index])]
            for index in np.argsort(raw_means)
        ],
        "mean_G_adjusted_order": [
            [PAIR_NAMES[index], float(adjusted_means[index])]
            for index in np.argsort(adjusted_means)
        ],
        "subject_pair_patterns": subject_patterns,
        "subject_concentration": {
            "largest_subject": largest_subject_index + 1,
            "largest_value": float(full.j_s[largest_subject_index]),
            "remaining_mean": float(np.mean(np.delete(full.j_s, largest_subject_index))),
        },
        "LR_subject_summary": {
            "positive_count": int(np.count_nonzero(J_bank[:, 0] > 0.0)),
            "median": float(np.median(J_bank[:, 0])),
            "mean": float(np.mean(J_bank[:, 0])),
        },
        "K_J": K_J,
        "K_J_direction": "positive" if K_J > 0.0 else "negative" if K_J < 0.0 else "zero",
        "cross_effector_mean_T_J": cross_mean,
        "within_group_mean_T_J": within_mean,
        "parent_artifacts_unchanged": True,
        "parent_hashes_before": before,
        "parent_hashes_after": after,
        "input_reproduction": reproduction,
        "protocol_source_hash": protocol_source_hash(),
        "runtime_seconds": time.perf_counter() - started,
        "tests": {
            "focused_before": focused_before,
            "focused_after": "PENDING",
            "full_after": "PENDING",
        },
        "scientific_setting_changed_after_protocol_freeze": False,
        "pairwise_pvalues_computed": False,
    }
    atomic_json(provenance / "scientific_results.json", results)
    atomic_json(
        provenance / "parent_artifact_immutability.json",
        {"status": "PASS", "unchanged": True, "before": before, "after": after},
    )
    atomic_json(
        provenance / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    )
    atomic_text(REPORT, render_report(results, "PENDING_SCIENTIFIC_RESULT_COMMIT"))
    write_manifest()
    print(
        json.dumps(
            {
                "verdict": results["verdict"],
                "pair_T_J": pair_t_j,
                "maximum_reconstruction_error": maximum_error,
                "K_J": K_J,
                "G_commonality": g_details,
                "H_commonality": h_details,
            },
            indent=2,
        )
    )


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
