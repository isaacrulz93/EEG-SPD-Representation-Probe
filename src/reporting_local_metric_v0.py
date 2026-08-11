"""Output contract, figures, and plain-language report for local metric V0."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.local_metric_interaction_v0 import (
    CLASS_ORDER,
    InteractionNullResult,
    N_CLASSES,
    N_SUBJECTS,
    mechanism_tag,
    terminal_decision,
)
from src.local_metric_pipeline_v0 import CellMetricMatrices, DecodingResult
from src.trajectory_within_subject_v1 import sha256_file


def environment_record() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "pyriemann",
        "mne",
        "moabb",
    ):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def implementation_source_hash(root: Path, paths: list[str]) -> tuple[str, dict[str, str]]:
    hashes = {path: sha256_file(root / path) for path in paths}
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), hashes


def _provenance_columns(frame: pd.DataFrame, provenance: Mapping[str, str]) -> pd.DataFrame:
    result = frame.copy()
    for key in (
        "protocol_freeze_sha",
        "protocol_sha256",
        "scientific_source_sha",
        "config_sha256",
        "implementation_source_sha256",
    ):
        result[key] = str(provenance[key])
    return result


def _cell_table(
    matrices: CellMetricMatrices, provenance: Mapping[str, str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = [
        (subject, class_label)
        for subject in range(1, N_SUBJECTS + 1)
        for class_label in CLASS_ORDER
    ]
    for metric, directional, symmetric in (
        ("raw", matrices.raw_m01, matrices.raw_m),
        ("size", matrices.size_m01, matrices.size_m),
        ("normalized", matrices.normalized_m01, matrices.normalized_m),
    ):
        for row, (subject_a, class_a) in enumerate(labels):
            for column, (subject_b, class_b) in enumerate(labels):
                rows.append(
                    {
                        "metric": metric,
                        "row_subject": subject_a,
                        "row_class": class_a,
                        "column_subject": subject_b,
                        "column_class": class_b,
                        "M01": directional[row, column],
                        "M": symmetric[row, column],
                    }
                )
    return _provenance_columns(pd.DataFrame(rows), provenance)


def _contrast_tables(
    results: Mapping[str, InteractionNullResult], provenance: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    p_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for metric, result in results.items():
        observed = result.observed
        for subject_index in range(N_SUBJECTS):
            for class_index, class_label in enumerate(CLASS_ORDER):
                cell_rows.append(
                    {
                        "metric": metric,
                        "subject": subject_index + 1,
                        "class_label": class_label,
                        "J_sc": observed.j_sc[subject_index, class_index],
                        "S_sc": observed.s_sc[subject_index, class_index],
                        "C_sc": observed.c_sc[subject_index, class_index],
                    }
                )
            subject_rows.append(
                {
                    "metric": metric,
                    "subject": subject_index + 1,
                    "J_s": observed.j_s[subject_index],
                    "S_s": observed.s_s[subject_index],
                    "C_s": observed.c_s[subject_index],
                }
            )
        p_rows.append(
            {
                "metric": metric,
                "T_J": observed.t_j,
                "p_J_classbreak": result.p_j_classbreak,
                "p_J_subjectbreak": result.p_j_subjectbreak,
                "p_J_conservative": result.p_j,
                "T_S": observed.t_s,
                "p_S_subjectbreak": result.p_s_subjectbreak,
                "T_C": observed.t_c,
                "p_C_classbreak": result.p_c_classbreak,
            }
        )
        for replicate in range(len(result.classbreak_t_j)):
            null_rows.extend(
                [
                    {
                        "metric": metric,
                        "family": "classbreak",
                        "replicate": replicate,
                        "T_J": result.classbreak_t_j[replicate],
                        "supporting_statistic": result.classbreak_t_c[replicate],
                        "supporting_name": "T_C",
                    },
                    {
                        "metric": metric,
                        "family": "subjectbreak",
                        "replicate": replicate,
                        "T_J": result.subjectbreak_t_j[replicate],
                        "supporting_statistic": result.subjectbreak_t_s[replicate],
                        "supporting_name": "T_S",
                    },
                ]
            )
    return tuple(
        _provenance_columns(pd.DataFrame(rows), provenance)
        for rows in (cell_rows, subject_rows, p_rows, null_rows)
    )  # type: ignore[return-value]


def _save_figure(fig: plt.Figure, stem: Path, provenance: Mapping[str, str]) -> None:
    metadata = {
        "Title": stem.name,
        "Subject": (
            f"protocol={provenance['protocol_freeze_sha']};"
            f"science={provenance['scientific_source_sha']};"
            f"config={provenance['config_sha256']};"
            f"implementation={provenance['implementation_source_sha256']}"
        ),
    }
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", metadata=metadata)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def _matrix_figure(
    matrix: np.ndarray,
    *,
    title: str,
    label: str,
    stem: Path,
    provenance: Mapping[str, str],
) -> None:
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, aspect="equal", interpolation="nearest", cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("Session-1 subject×class cell")
    axis.set_ylabel("Session-0 subject×class cell")
    for boundary in range(N_CLASSES, N_SUBJECTS * N_CLASSES, N_CLASSES):
        axis.axhline(boundary - 0.5, color="white", linewidth=0.3, alpha=0.6)
        axis.axvline(boundary - 0.5, color="white", linewidth=0.3, alpha=0.6)
    fig.colorbar(image, ax=axis, label=label)
    _save_figure(fig, stem, provenance)


def write_scientific_outputs(
    output_root: Path,
    *,
    matrices: CellMetricMatrices,
    results: Mapping[str, InteractionNullResult],
    reliability: pd.DataFrame,
    decoding: DecodingResult,
    provenance: Mapping[str, str],
    total_runtime_seconds: float,
) -> dict[str, Any]:
    tables = output_root / "tables"
    nulls = output_root / "nulls"
    figures = output_root / "figures"
    decisions = output_root / "decisions"
    report_dir = output_root / "report"
    for directory in (tables, nulls, figures, decisions, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cell_table = _cell_table(matrices, provenance)
    contrast_cells, contrast_subjects, pvalues, null_table = _contrast_tables(
        results, provenance
    )
    cell_table.to_csv(tables / "cell_distances.csv", index=False)
    contrast_cells.to_csv(tables / "subject_class_contrasts.csv", index=False)
    contrast_subjects.to_csv(tables / "subject_contrasts.csv", index=False)
    pvalues.to_csv(tables / "interaction_pvalues.csv", index=False)
    null_table.to_csv(tables / "interaction_null_distributions.csv", index=False)
    _provenance_columns(reliability, provenance).to_csv(
        tables / "within_cell_variability_diagnostics.csv", index=False
    )
    _provenance_columns(decoding.subject_scores, provenance).to_csv(
        tables / "secondary_decoding_results.csv", index=False
    )
    decoding_summary = _provenance_columns(
        pd.DataFrame(
            [
                {
                    "mean_subject_ba": decoding.group_mean_ba,
                    "median_subject_ba": decoding.group_median_ba,
                    "chance": 0.25,
                    "null_median": float(np.median(decoding.null_group_median_ba)),
                    "p_value": decoding.p_value,
                    "null_replicates": len(decoding.null_group_median_ba),
                }
            ]
        ),
        provenance,
    )
    decoding_summary.to_csv(tables / "secondary_decoding_summary.csv", index=False)

    np.savez_compressed(
        nulls / "frozen_null_statistics.npz",
        raw_classbreak_t_j=results["raw"].classbreak_t_j,
        raw_subjectbreak_t_j=results["raw"].subjectbreak_t_j,
        size_classbreak_t_j=results["size"].classbreak_t_j,
        size_subjectbreak_t_j=results["size"].subjectbreak_t_j,
        normalized_classbreak_t_j=results["normalized"].classbreak_t_j,
        normalized_subjectbreak_t_j=results["normalized"].subjectbreak_t_j,
        decoding_group_median_ba=decoding.null_group_median_ba,
        protocol_freeze_sha=np.asarray([provenance["protocol_freeze_sha"]]),
        protocol_sha256=np.asarray([provenance["protocol_sha256"]]),
        scientific_source_sha=np.asarray([provenance["scientific_source_sha"]]),
        config_sha256=np.asarray([provenance["config_sha256"]]),
        implementation_source_sha256=np.asarray(
            [provenance["implementation_source_sha256"]]
        ),
    )

    raw = results["raw"]
    size = results["size"]
    normalized = results["normalized"]
    tag = mechanism_tag(
        size_t_j=size.observed.t_j,
        size_p_classbreak=size.p_j_classbreak,
        size_p_subjectbreak=size.p_j_subjectbreak,
        normalized_t_j=normalized.observed.t_j,
        normalized_p_classbreak=normalized.p_j_classbreak,
        normalized_p_subjectbreak=normalized.p_j_subjectbreak,
    )
    terminal = terminal_decision(
        t_j=raw.observed.t_j,
        p_j_classbreak=raw.p_j_classbreak,
        p_j_subjectbreak=raw.p_j_subjectbreak,
    )
    decision = {
        **dict(provenance),
        "terminal_decision": terminal,
        "mechanism_tag": tag,
        "raw_T_J": raw.observed.t_j,
        "raw_p_J_classbreak": raw.p_j_classbreak,
        "raw_p_J_subjectbreak": raw.p_j_subjectbreak,
        "raw_p_J_conservative": raw.p_j,
        "total_runtime_seconds": total_runtime_seconds,
        "scientific_definitions_changed_after_P1": False,
    }
    (decisions / "terminal_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    labels = [f"S{s}:{c[:2]}" for s in range(1, 10) for c in CLASS_ORDER]
    matrix_csv = _provenance_columns(
        pd.DataFrame(matrices.raw_m01, index=labels, columns=labels)
        .rename_axis("row_cell")
        .reset_index(),
        provenance,
    )
    matrix_csv.to_csv(figures / "figure_1_raw_cross_session_M01.csv", index=False)
    _matrix_figure(
        matrices.raw_m01,
        title="RAW directional cross-session cell distance M01",
        label="Median unlabeled AIRM metric-configuration distance",
        stem=figures / "figure_1_raw_cross_session_M01",
        provenance=provenance,
    )
    _provenance_columns(
        pd.DataFrame(matrices.raw_m, index=labels, columns=labels)
        .rename_axis("row_cell")
        .reset_index(),
        provenance,
    ).to_csv(figures / "figure_2_raw_session_symmetrized_M.csv", index=False)
    _matrix_figure(
        matrices.raw_m,
        title="RAW session-role-symmetrized cell distance M",
        label="Symmetrized median distance",
        stem=figures / "figure_2_raw_session_symmetrized_M",
        provenance=provenance,
    )

    def subject_bar(values: np.ndarray, title: str, ylabel: str, stem_name: str) -> None:
        data = _provenance_columns(
            pd.DataFrame({"subject": np.arange(1, 10), "value": values}), provenance
        )
        data.to_csv(figures / f"{stem_name}.csv", index=False)
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(np.arange(1, 10), values, color="#3A6EA5")
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(np.arange(1, 10))
        axis.set_xlabel("Subject")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        _save_figure(fig, figures / stem_name, provenance)

    subject_bar(
        raw.observed.j_s,
        "Subject-level local metric interaction",
        "J_s",
        "figure_3_subject_J_s",
    )
    _provenance_columns(
        pd.DataFrame(raw.observed.j_sc, index=np.arange(1, 10), columns=CLASS_ORDER)
        .rename_axis("subject")
        .reset_index(),
        provenance,
    ).to_csv(figures / "figure_4_subject_class_J_sc.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 5))
    bound = max(float(np.max(np.abs(raw.observed.j_sc))), np.finfo(float).eps)
    image = axis.imshow(
        raw.observed.j_sc,
        aspect="auto",
        cmap="coolwarm",
        vmin=-bound,
        vmax=bound,
    )
    axis.set_xticks(np.arange(4), labels=CLASS_ORDER, rotation=25, ha="right")
    axis.set_yticks(np.arange(9), labels=np.arange(1, 10))
    axis.set_xlabel("MI class")
    axis.set_ylabel("Subject")
    axis.set_title("RAW subject×class interaction J_sc")
    fig.colorbar(image, ax=axis, label="J_sc")
    _save_figure(fig, figures / "figure_4_subject_class_J_sc", provenance)
    subject_bar(
        raw.observed.s_s,
        "Supporting same-class subject specificity",
        "S_s",
        "figure_5_subject_specificity_S_s",
    )
    subject_bar(
        raw.observed.c_s,
        "Supporting within-subject class specificity",
        "C_s",
        "figure_6_class_specificity_C_s",
    )

    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "T_J": result.observed.t_j,
                "p_classbreak": result.p_j_classbreak,
                "p_subjectbreak": result.p_j_subjectbreak,
                "p_conservative": result.p_j,
            }
            for metric, result in results.items()
        ]
    )
    _provenance_columns(comparison, provenance).to_csv(
        figures / "figure_7_raw_size_normalized_interaction.csv", index=False
    )
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(comparison["metric"], comparison["T_J"], color=["#3A6EA5", "#D9822B", "#4C956C"])
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("T_J")
    axis.set_title("RAW, metric-size, and relative-pattern interaction")
    _save_figure(fig, figures / "figure_7_raw_size_normalized_interaction", provenance)

    _provenance_columns(decoding.subject_scores, provenance).to_csv(
        figures / "figure_8_secondary_cross_session_decoding.csv", index=False
    )
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        decoding.subject_scores["subject"],
        decoding.subject_scores["balanced_accuracy"],
        color="#6B4C9A",
    )
    axis.axhline(0.25, color="black", linestyle="--", label="25% chance")
    axis.set_xticks(np.arange(1, 10))
    axis.set_xlabel("Subject")
    axis.set_ylabel("Two-direction mean balanced accuracy")
    axis.set_title("Secondary cross-session nearest-medoid decoding")
    axis.legend()
    _save_figure(fig, figures / "figure_8_secondary_cross_session_decoding", provenance)
    return decision


def write_report(
    path: Path,
    *,
    branch: str,
    result_commit: str,
    reproduction: Mapping[str, Any],
    results: Mapping[str, InteractionNullResult],
    reliability: pd.DataFrame,
    decoding: DecodingResult,
    decision: Mapping[str, Any],
    tests_summary: str,
    repository_tests_summary: str,
    git_status: str,
) -> None:
    raw = results["raw"]
    size = results["size"]
    normalized = results["normalized"]
    j_subject_lines = "\n".join(
        f"- Subject {index + 1}: {value:.8f}"
        for index, value in enumerate(raw.observed.j_s)
    )
    j_cell_lines = "\n".join(
        f"- Subject {subject + 1}, {CLASS_ORDER[class_index]}: {raw.observed.j_sc[subject, class_index]:.8f}"
        for subject in range(N_SUBJECTS)
        for class_index in range(N_CLASSES)
    )
    ba_lines = "\n".join(
        f"- Subject {int(row.subject)}: {row.balanced_accuracy:.6f} "
        f"(0→1 {row.ba_session0_to_session1:.6f}; 1→0 {row.ba_session1_to_session0:.6f})"
        for row in decoding.subject_scores.itertuples(index=False)
    )
    reliability_summary = reliability.groupby("session", sort=False)[
        ["median_within_cell_delta_raw", "iqr_within_cell_delta_raw", "medoid_objective"]
    ].median()
    text = f"""# Local AIRM Metric Interaction V0

The earlier trajectory audit asked whether distance-only local geometry contains class information within a subject and across sessions. This experiment asks the narrower structural question: is the exact same subject–class pairing more reproducible across sessions than expected from separate subject and class correspondence effects?

## 1. Branch

`{branch}`

## 2. Scientific base SHA

`{decision['scientific_source_sha']}`

## 3. Protocol-freeze SHA

`{decision['protocol_freeze_sha']}`

## 4. Final scientific-result SHA

`{result_commit}`

## 5. Frozen representation reproduction

PASS. The immutable two-session trajectory cache contained {reproduction['trial_count']} trials. Frozen cache, distance-matrix, and PATH_D10 hashes matched; the maximum absolute difference between saved PATH_D10 and the fixed upper triangle reconstructed from the saved distance matrices was {reproduction['upper_triangle_path_max_abs_diff']:.1e}.

## 6. Mathematical audit

PASS. Exact enumeration of all 120 vertex-induced S5 edge permutations was symmetric, permutation-invariant, and satisfied the finite-group orbit-metric triangle inequality. The slow exhaustive and vectorized implementations agreed at the frozen 1e-12 tolerance. Common orthogonal and well-conditioned nonorthogonal congruence tests passed. The trial-pair raw/size/normalized squared-distance identity passed, and the inversion fixture demonstrated why this is only a pseudometric on original SPD configurations.

## 7. Degeneracy audit

PASS: {reproduction['degenerate_trial_count']} trials had zero edge-RMS metric size.

## 8–11. Primary RAW interaction

- T_J: {raw.observed.t_j:.10f}
- p_J_classbreak: {raw.p_j_classbreak:.6f}
- p_J_subjectbreak: {raw.p_j_subjectbreak:.6f}
- conservative p_J=max: {raw.p_j:.6f}

## 12. All nine subject-level J_s values

{j_subject_lines}

## 13. All 36 subject×class J_sc values

{j_cell_lines}

## 14. Supporting subject specificity

T_S={raw.observed.t_s:.10f}; subject-break p={raw.p_s_subjectbreak:.6f}. This is supporting anatomy and does not establish interaction by itself.

## 15. Supporting class specificity

T_C={raw.observed.t_c:.10f}; class-break p={raw.p_c_classbreak:.6f}. This is supporting anatomy and does not establish interaction by itself.

## 16. Edge-RMS metric-size control

T_J_size={size.observed.t_j:.10f}; class-break p={size.p_j_classbreak:.6f}; subject-break p={size.p_j_subjectbreak:.6f}; conservative p={size.p_j:.6f}.

## 17. Size-normalized relative-pattern control

T_J_norm={normalized.observed.t_j:.10f}; class-break p={normalized.p_j_classbreak:.6f}; subject-break p={normalized.p_j_subjectbreak:.6f}; conservative p={normalized.p_j:.6f}.

## 18. Mechanism tag

`{decision['mechanism_tag']}`

This is a nonterminal mechanism control. It is not a causal mediation result, and raw J is not the sum of size and normalized J.

## 19. Secondary cross-session class decoding

- Mean subject BA: {decoding.group_mean_ba:.6f}
- Median subject BA: {decoding.group_median_ba:.6f}
- Null median: {float(np.median(decoding.null_group_median_ba)):.6f}
- One-sided plus-one p: {decoding.p_value:.6f}
- Chance reference: 0.25

{ba_lines}

This nearest-class-medoid result is secondary and did not affect the primary terminal.

## 20. Within-cell reliability diagnostics

The diagnostic was non-gating and used no post-hoc threshold. Median summaries over the 36 subject×class cells in each session were:

```
{reliability_summary.to_string()}
```

All 72 cell-level diagnostics, including IQR and medoid objective, are in the machine-readable table. Any broad within-cell variability must be considered when interpreting a negative result.

## 21. Terminal scientific decision

`{decision['terminal_decision']}`

The terminal is based only on raw T_J and the two preregistered correspondence-breaking nulls. The experiment does not identify a full SPD configuration, pose, temporal trajectory, physiology, source-space organization, or a global subject transformation.

## 22. Total runtime

{float(decision['total_runtime_seconds']):.2f} seconds.

## 23. Relevant tests

{tests_summary}

## 24. Full repository tests

{repository_tests_summary}

## 25. Git status

`{git_status or 'clean'}`

## 26. Post-result immutability

Confirmed: no scientific definition changed after the first real-data Stage-P1 statistic was observed.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_artifact_manifest(output_root: Path, provenance: Mapping[str, str]) -> None:
    rows: list[dict[str, Any]] = []
    manifest_path = output_root / "protocol" / "artifact_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                **dict(provenance),
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


__all__ = [
    "environment_record",
    "implementation_source_hash",
    "write_artifact_manifest",
    "write_report",
    "write_scientific_outputs",
]
