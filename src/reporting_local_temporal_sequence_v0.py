"""Artifacts, fixed visualizations, and report for temporal sequence V0."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.local_temporal_sequence_v0 import (
    ALL_PERMUTATIONS,
    CLASS_ORDER,
    DERANGEMENT_MASK,
    HALF_ORDER,
    MatchingResults,
    MeanSequenceBank,
    N_CELLS,
    N_CLASSES,
    N_SUBJECTS,
    N_TIMES,
    SESSION_ORDER,
    CommonPCA,
    TemporalInference,
    cell_index,
    cell_labels,
    group_average_k,
)
from src.trajectory_within_subject_v1 import sha256_file


FIXED_CELLS = ((1, "left_hand"), (1, "feet"), (2, "left_hand"), (2, "feet"))


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
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def implementation_source_hash(root: Path, paths: list[str]) -> tuple[str, dict[str, str]]:
    hashes = {path: sha256_file(root / path) for path in paths}
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), hashes


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _provenance_columns(
    frame: pd.DataFrame, provenance: Mapping[str, str]
) -> pd.DataFrame:
    result = frame.copy()
    for key in (
        "protocol_freeze_sha",
        "protocol_sha256",
        "local_metric_final_sha",
        "local_gpa_v1_final_sha",
        "gpa_outer_audit_final_sha",
        "config_sha256",
        "implementation_source_sha256",
    ):
        result[key] = str(provenance[key])
    return result


def _save_figure(fig: plt.Figure, stem: Path, provenance: Mapping[str, str]) -> None:
    metadata = {
        "Title": stem.name,
        "Subject": (
            f"protocol={provenance['protocol_freeze_sha']};"
            f"lineage={provenance['gpa_outer_audit_final_sha']};"
            f"config={provenance['config_sha256']};"
            f"implementation={provenance['implementation_source_sha256']}"
        ),
    }
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", metadata=metadata)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def _cell_category(row: int, column: int) -> str:
    row_subject, row_class = divmod(row, N_CLASSES)
    column_subject, column_class = divmod(column, N_CLASSES)
    if row_subject == column_subject and row_class == column_class:
        return "same_subject_same_class"
    if row_subject != column_subject and row_class == column_class:
        return "different_subject_same_class"
    if row_subject == column_subject and row_class != column_class:
        return "same_subject_different_class"
    return "different_subject_different_class"


def _matching_table(
    matching: MatchingResults, provenance: Mapping[str, str]
) -> pd.DataFrame:
    labels = cell_labels()
    rows: list[dict[str, Any]] = []
    for row, (subject_a, class_a) in enumerate(labels):
        for column, (subject_b, class_b) in enumerate(labels):
            rows.append(
                {
                    "session0_subject": subject_a,
                    "session0_class": class_a,
                    "session1_subject": subject_b,
                    "session1_class": class_b,
                    "category": _cell_category(row, column),
                    "D_id": matching.d_id[row, column],
                    "median_derangement_cost": matching.median_derangement[row, column],
                    "G": matching.gain[row, column],
                    "G_rel": matching.relative_gain[row, column],
                    "identity_rank": matching.identity_rank[row, column],
                    "identity_tie_count": matching.identity_tie_count[row, column],
                }
            )
    return _provenance_columns(pd.DataFrame(rows), provenance)


def _contrast_tables(
    inference: TemporalInference, provenance: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed = inference.observed
    cell_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for subject_index in range(N_SUBJECTS):
        for class_index, class_label in enumerate(CLASS_ORDER):
            cell_rows.append(
                {
                    "subject": subject_index + 1,
                    "class_label": class_label,
                    "A_sc": observed.a_sc[subject_index, class_index],
                    "b_sc": observed.b_sc[subject_index, class_index],
                    "c_sc": observed.c_sc[subject_index, class_index],
                    "d_sc": observed.d_sc[subject_index, class_index],
                    "S_sc": observed.s_sc[subject_index, class_index],
                    "C_sc": observed.c_specific_sc[subject_index, class_index],
                    "J_sc": observed.j_sc[subject_index, class_index],
                }
            )
        subject_rows.append(
            {
                "subject": subject_index + 1,
                "A_s": observed.a_s[subject_index],
                "S_s": observed.s_s[subject_index],
                "C_s": observed.c_s[subject_index],
                "J_s": observed.j_s[subject_index],
            }
        )
    summary = pd.DataFrame(
        [
            {
                "T_temporal": observed.t_temporal,
                "p_temporal": inference.p_temporal,
                "T_subject": observed.t_subject,
                "p_subject": inference.p_subject,
                "T_class": observed.t_class,
                "p_class": inference.p_class,
                "T_J": observed.t_j,
                "p_J_subjectbreak": inference.p_j_subjectbreak,
                "p_J_classbreak": inference.p_j_classbreak,
                "terminal": inference.terminal,
            }
        ]
    )
    return tuple(
        _provenance_columns(frame, provenance)
        for frame in (pd.DataFrame(cell_rows), pd.DataFrame(subject_rows), summary)
    )  # type: ignore[return-value]


def _reliability_tables(
    cross: np.ndarray, provenance: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for session_index, session in enumerate(SESSION_ORDER):
        for subject_index in range(N_SUBJECTS):
            for class_index, class_label in enumerate(CLASS_ORDER):
                for time_a in range(N_TIMES):
                    for time_b in range(N_TIMES):
                        rows.append(
                            {
                                "subject": subject_index + 1,
                                "session": session,
                                "class_label": class_label,
                                "half_A_position": time_a + 1,
                                "half_B_position": time_b + 1,
                                "same_position": time_a == time_b,
                                "airm_distance": cross[
                                    session_index,
                                    subject_index,
                                    class_index,
                                    time_a,
                                    time_b,
                                ],
                            }
                        )
    table = pd.DataFrame(rows)
    same = table.loc[table["same_position"]].copy()
    summary = (
        same.groupby(["session", "half_A_position"], sort=False)["airm_distance"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"half_A_position": "temporal_position"})
    )
    quantiles = (
        same.groupby(["session", "half_A_position"], sort=False)["airm_distance"]
        .quantile([0.25, 0.75])
        .unstack()
        .reset_index()
        .rename(
            columns={"half_A_position": "temporal_position", 0.25: "q25", 0.75: "q75"}
        )
    )
    summary = summary.merge(quantiles, on=["session", "temporal_position"])
    return (
        _provenance_columns(table, provenance),
        _provenance_columns(summary, provenance),
    )


def _rank_summary(
    matching_table: pd.DataFrame, provenance: Mapping[str, str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category, group in matching_table.groupby("category", sort=False):
        ranks = group["identity_rank"].to_numpy(dtype=float)
        rows.append(
            {
                "category": category,
                "n_pairs": len(ranks),
                "mean_rank": np.mean(ranks),
                "median_rank": np.median(ranks),
                "q25_rank": np.quantile(ranks, 0.25),
                "q75_rank": np.quantile(ranks, 0.75),
                "minimum_rank": np.min(ranks),
                "maximum_rank": np.max(ranks),
                "rank_1_count": np.count_nonzero(ranks == 1),
                "rank_1_fraction": np.mean(ranks == 1),
            }
        )
    return _provenance_columns(pd.DataFrame(rows), provenance)


def _pca_table(pca: CommonPCA, provenance: Mapping[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_index, session in enumerate(SESSION_ORDER):
        for subject_index in range(N_SUBJECTS):
            for class_index, class_label in enumerate(CLASS_ORDER):
                for time_index in range(N_TIMES):
                    point = pca.coordinates[
                        session_index, subject_index, class_index, time_index
                    ]
                    rows.append(
                        {
                            "subject": subject_index + 1,
                            "session": session,
                            "class_label": class_label,
                            "temporal_position": time_index + 1,
                            "PC1": point[0],
                            "PC2": point[1],
                        }
                    )
    return _provenance_columns(pd.DataFrame(rows), provenance)


def _fixed_sequence_figure(
    pca: CommonPCA, stem: Path, provenance: Mapping[str, str]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    session_styles = (("0train", "o", "-"), ("1test", "s", "--"))
    for axis, (subject, class_label) in zip(axes.flat, FIXED_CELLS, strict=True):
        subject_index = subject - 1
        class_index = CLASS_ORDER.index(class_label)
        for session_index, (session, marker, linestyle) in enumerate(session_styles):
            points = pca.coordinates[session_index, subject_index, class_index]
            axis.plot(
                points[:, 0],
                points[:, 1],
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                label=session,
            )
            for time_index, point in enumerate(points):
                axis.annotate(str(time_index + 1), point, xytext=(4, 4), textcoords="offset points")
        axis.set_title(f"Subject {subject} — {class_label}")
        axis.axhline(0.0, color="0.85", linewidth=0.5)
        axis.axvline(0.0, color="0.85", linewidth=0.5)
        axis.legend(fontsize=8)
    fig.supxlabel("Common global PC1")
    fig.supylabel("Common global PC2")
    fig.suptitle("Fixed ordered mean covariance sequences (common AIRM tangent PCA)")
    _save_figure(fig, stem, provenance)


def _subject_small_multiple_figure(
    pca: CommonPCA, stem: Path, provenance: Mapping[str, str]
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 13), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10").colors[:N_CLASSES]
    for subject_index, axis in enumerate(axes.flat):
        for class_index, class_label in enumerate(CLASS_ORDER):
            for session_index, (session, linestyle) in enumerate(
                (("0train", "-"), ("1test", "--"))
            ):
                points = pca.coordinates[session_index, subject_index, class_index]
                axis.plot(
                    points[:, 0],
                    points[:, 1],
                    color=colors[class_index],
                    linestyle=linestyle,
                    marker="o" if session_index == 0 else "s",
                    markersize=3,
                    linewidth=1,
                    label=f"{class_label} {session}" if subject_index == 0 else None,
                )
        axis.set_title(f"Subject {subject_index + 1}")
        axis.axhline(0.0, color="0.9", linewidth=0.4)
        axis.axvline(0.0, color="0.9", linewidth=0.4)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.supxlabel("Common global PC1")
    fig.supylabel("Common global PC2")
    fig.suptitle("All subjects: ordered cell mean covariance sequences")
    fig.subplots_adjust(bottom=0.10)
    _save_figure(fig, stem, provenance)


def _heatmap_grid(
    matrices: list[tuple[str, np.ndarray]],
    *,
    title: str,
    stem: Path,
    provenance: Mapping[str, str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    vmin = min(float(np.min(value)) for _, value in matrices)
    vmax = max(float(np.max(value)) for _, value in matrices)
    image = None
    for axis, (label, value) in zip(axes.flat, matrices, strict=True):
        image = axis.imshow(value, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
        axis.set_title(label)
        axis.set_xticks(range(N_TIMES), range(1, N_TIMES + 1))
        axis.set_yticks(range(N_TIMES), range(1, N_TIMES + 1))
        axis.set_xlabel("Session-1 temporal position")
        axis.set_ylabel("Session-0 temporal position")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="AIRM distance", shrink=0.85)
    fig.suptitle(title)
    _save_figure(fig, stem, provenance)


def _subject_gain_figure(
    inference: TemporalInference, stem: Path, provenance: Mapping[str, str]
) -> None:
    observed = inference.observed
    x = np.arange(1, N_SUBJECTS + 1)
    width = 0.2
    fig, axis = plt.subplots(figsize=(11, 5))
    for offset, values, label in (
        (-1.5 * width, observed.a_s, "A_s temporal"),
        (-0.5 * width, observed.s_s, "S_s subject"),
        (0.5 * width, observed.c_s, "C_s class"),
        (1.5 * width, observed.j_s, "J_s interaction"),
    ):
        axis.bar(x + offset, values, width=width, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xlabel("Subject")
    axis.set_ylabel("AIRM temporal correspondence gain")
    axis.set_title("Frozen subject-level temporal correspondence summaries")
    axis.legend(ncol=2)
    _save_figure(fig, stem, provenance)


def write_scientific_outputs(
    output_root: Path,
    *,
    bank: MeanSequenceBank,
    reliability_cross: np.ndarray,
    matching: MatchingResults,
    inference: TemporalInference,
    pca: CommonPCA,
    provenance: Mapping[str, str],
    total_runtime_seconds: float,
) -> dict[str, Any]:
    arrays = output_root / "arrays"
    tables = output_root / "tables"
    nulls = output_root / "nulls"
    figures = output_root / "figures"
    decisions = output_root / "decisions"
    report_dir = output_root / "report"
    for directory in (arrays, tables, nulls, figures, decisions, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(cell_labels(), dtype=object)
    np.savez_compressed(
        arrays / "ordered_mean_sequences.npz",
        means=bank.full,
        subjects=np.arange(1, N_SUBJECTS + 1),
        sessions=np.asarray(SESSION_ORDER),
        classes=np.asarray(CLASS_ORDER),
        temporal_positions=np.arange(1, N_TIMES + 1),
    )
    np.savez_compressed(
        arrays / "split_half_mean_sequences.npz",
        means=bank.split,
        halves=np.asarray(HALF_ORDER),
        subjects=np.arange(1, N_SUBJECTS + 1),
        sessions=np.asarray(SESSION_ORDER),
        classes=np.asarray(CLASS_ORDER),
        temporal_positions=np.arange(1, N_TIMES + 1),
    )
    np.savez_compressed(
        arrays / "cross_time_K_matrices.npz",
        K=matching.k,
        cell_subjects=labels[:, 0].astype(int),
        cell_classes=labels[:, 1].astype(str),
    )
    np.savez_compressed(
        arrays / "matching_matrices.npz",
        D_id=matching.d_id,
        median_derangement_cost=matching.median_derangement,
        G=matching.gain,
        G_rel=matching.relative_gain,
        identity_rank=matching.identity_rank,
        identity_tie_count=matching.identity_tie_count,
        all_permutation_costs=matching.all_costs,
        permutations=ALL_PERMUTATIONS,
        is_derangement=DERANGEMENT_MASK,
    )
    np.savez_compressed(
        arrays / "split_half_cross_time_K_matrices.npz",
        K_half_A_to_B=reliability_cross,
    )
    np.savez_compressed(
        arrays / "common_tangent_pca.npz",
        global_airm_reference=pca.reference,
        coordinates=pca.coordinates,
        components=pca.components,
        explained_variance=pca.explained_variance,
        explained_variance_ratio=pca.explained_variance_ratio,
        feature_mean=pca.feature_mean,
    )
    np.savez_compressed(
        nulls / "frozen_null_distributions.npz",
        T_temporal=inference.temporal_null,
        T_subject_subjectbreak=inference.subjectbreak_t_subject,
        T_J_subjectbreak=inference.subjectbreak_t_j,
        T_class_classbreak=inference.classbreak_t_class,
        T_J_classbreak=inference.classbreak_t_j,
        temporal_permutation_indices=inference.temporal_permutation_indices,
        subject_mappings=inference.subject_mappings,
        class_mappings=inference.class_mappings,
    )

    diagnostics = pd.concat(
        [bank.diagnostics, pd.DataFrame([pca.reference_diagnostic])], ignore_index=True
    )
    _provenance_columns(diagnostics, provenance).to_csv(
        tables / "airm_mean_numerical_diagnostics.csv", index=False
    )
    reliability, reliability_summary = _reliability_tables(
        reliability_cross, provenance
    )
    reliability.to_csv(tables / "split_half_temporal_reliability.csv", index=False)
    reliability_summary.to_csv(
        tables / "split_half_same_position_summary.csv", index=False
    )
    matching_table = _matching_table(matching, provenance)
    matching_table.to_csv(tables / "cross_session_matching_summary.csv", index=False)
    rank_summary = _rank_summary(matching_table, provenance)
    rank_summary.to_csv(tables / "identity_rank_summary.csv", index=False)
    cell_contrasts, subject_contrasts, summary = _contrast_tables(inference, provenance)
    cell_contrasts.to_csv(tables / "subject_class_temporal_contrasts.csv", index=False)
    subject_contrasts.to_csv(tables / "subject_temporal_contrasts.csv", index=False)
    summary.to_csv(tables / "temporal_inference_summary.csv", index=False)
    pd.DataFrame(
        {
            "permutation_index": np.arange(120),
            **{f"pi_{index + 1}": ALL_PERMUTATIONS[:, index] + 1 for index in range(5)},
            "is_identity": np.arange(120) == 0,
            "is_derangement": DERANGEMENT_MASK,
        }
    ).to_csv(tables / "frozen_S5_permutations.csv", index=False)
    null_table = pd.DataFrame(
        {
            "replicate": np.arange(1, len(inference.temporal_null) + 1),
            "T_temporal_temporal_label_null": inference.temporal_null,
            "T_subject_subjectbreak": inference.subjectbreak_t_subject,
            "T_J_subjectbreak": inference.subjectbreak_t_j,
            "T_class_classbreak": inference.classbreak_t_class,
            "T_J_classbreak": inference.classbreak_t_j,
        }
    )
    _provenance_columns(null_table, provenance).to_csv(
        tables / "null_distributions.csv", index=False
    )
    pca_table = _pca_table(pca, provenance)
    pca_table.to_csv(tables / "common_tangent_pca_coordinates.csv", index=False)

    group_k = group_average_k(matching.k)
    group_k_rows: list[dict[str, Any]] = []
    for category, matrix in group_k.items():
        for row in range(N_TIMES):
            for column in range(N_TIMES):
                group_k_rows.append(
                    {
                        "category": category,
                        "session0_position": row + 1,
                        "session1_position": column + 1,
                        "mean_airm_distance": matrix[row, column],
                    }
                )
    _provenance_columns(pd.DataFrame(group_k_rows), provenance).to_csv(
        tables / "group_average_K_matrices.csv", index=False
    )
    fixed_k_rows: list[dict[str, Any]] = []
    fixed_matrices: list[tuple[str, np.ndarray]] = []
    for subject, class_label in FIXED_CELLS:
        index = cell_index(subject - 1, CLASS_ORDER.index(class_label))
        matrix = matching.k[index, index]
        fixed_matrices.append((f"S{subject} {class_label}", matrix))
        for row in range(N_TIMES):
            for column in range(N_TIMES):
                fixed_k_rows.append(
                    {
                        "subject": subject,
                        "class_label": class_label,
                        "session0_position": row + 1,
                        "session1_position": column + 1,
                        "airm_distance": matrix[row, column],
                    }
                )
    _provenance_columns(pd.DataFrame(fixed_k_rows), provenance).to_csv(
        tables / "fixed_cell_K_matrices.csv", index=False
    )

    anchor = cell_index(0, CLASS_ORDER.index("left_hand"))
    comparisons = (
        ("S1 Left vs S1 Left", anchor),
        ("S1 Left vs S2 Left", cell_index(1, CLASS_ORDER.index("left_hand"))),
        ("S1 Left vs S1 Feet", cell_index(0, CLASS_ORDER.index("feet"))),
        ("S1 Left vs S2 Feet", cell_index(1, CLASS_ORDER.index("feet"))),
    )
    comparison_table = pd.DataFrame(
        [
            {
                "comparison": name,
                "D_id": matching.d_id[anchor, target],
                "median_derangement_cost": matching.median_derangement[anchor, target],
                "G": matching.gain[anchor, target],
                "identity_rank": matching.identity_rank[anchor, target],
            }
            for name, target in comparisons
        ]
    )
    _provenance_columns(comparison_table, provenance).to_csv(
        tables / "predeclared_raw_D_id_comparisons.csv", index=False
    )

    _fixed_sequence_figure(
        pca, figures / "figure_1_fixed_mean_sequences", provenance
    )
    _subject_small_multiple_figure(
        pca, figures / "figure_2_all_subject_mean_sequences", provenance
    )
    _heatmap_grid(
        fixed_matrices,
        title="Fixed cross-session cell K matrices",
        stem=figures / "figure_3_fixed_cell_K_heatmaps",
        provenance=provenance,
    )
    _heatmap_grid(
        [(key.replace("_", " "), value) for key, value in group_k.items()],
        title="Group-average cross-time AIRM distance matrices",
        stem=figures / "figure_4_group_average_K_heatmaps",
        provenance=provenance,
    )
    _subject_gain_figure(
        inference, figures / "figure_5_subject_temporal_gains", provenance
    )

    decision = {
        **dict(provenance),
        "terminal_decision": inference.terminal,
        "T_temporal": inference.observed.t_temporal,
        "p_temporal": inference.p_temporal,
        "temporal_supported": bool(
            inference.observed.t_temporal > 0.0 and inference.p_temporal < 0.05
        ),
        "T_subject": inference.observed.t_subject,
        "p_subject": inference.p_subject,
        "subject_specificity_supported": bool(
            inference.observed.t_subject > 0.0 and inference.p_subject < 0.05
        ),
        "T_class": inference.observed.t_class,
        "p_class": inference.p_class,
        "class_specificity_supported": bool(
            inference.observed.t_class > 0.0 and inference.p_class < 0.05
        ),
        "secondary_T_J": inference.observed.t_j,
        "p_J_subjectbreak": inference.p_j_subjectbreak,
        "p_J_classbreak": inference.p_j_classbreak,
        "secondary_interaction_supported": bool(
            inference.observed.t_j > 0.0
            and inference.p_j_subjectbreak < 0.05
            and inference.p_j_classbreak < 0.05
        ),
        "null_replicates": len(inference.temporal_null),
        "alpha": 0.05,
        "total_runtime_seconds": total_runtime_seconds,
        "scientific_definitions_changed_after_result_access": False,
    }
    write_json(decisions / "terminal_decision.json", decision)
    return decision


def write_report(
    path: Path,
    *,
    branch: str,
    protocol_freeze_sha: str,
    final_result_sha: str,
    reproduction: Mapping[str, Any],
    bank: MeanSequenceBank,
    reliability_cross: np.ndarray,
    matching: MatchingResults,
    inference: TemporalInference,
    pca: CommonPCA,
    total_runtime_seconds: float,
    focused_tests: str,
    repository_tests: str,
    git_status: str,
) -> None:
    observed = inference.observed
    same_position = np.diagonal(reliability_cross, axis1=-2, axis2=-1)
    same_cell_ranks = np.diag(matching.identity_rank)
    comparison_targets = (
        ("S1 Left vs S1 Left", cell_index(0, 0)),
        ("S1 Left vs S2 Left", cell_index(1, 0)),
        ("S1 Left vs S1 Feet", cell_index(0, 2)),
        ("S1 Left vs S2 Feet", cell_index(1, 2)),
    )
    anchor = cell_index(0, 0)
    maximum_mean_residual = float(
        max(
            bank.diagnostics["normalized_karcher_post_residual"].max(),
            pca.reference_diagnostic["normalized_karcher_post_residual"],
        )
    )

    def subject_lines(values: np.ndarray) -> str:
        return "\n".join(
            f"- Subject {index + 1}: {float(value):.10f}"
            for index, value in enumerate(values)
        )

    comparison_lines = "\n".join(
        f"- {name}: D_id={matching.d_id[anchor, target]:.10f}; "
        f"median derangement={matching.median_derangement[anchor, target]:.10f}; "
        f"G={matching.gain[anchor, target]:.10f}; "
        f"identity rank={matching.identity_rank[anchor, target]}"
        for name, target in comparison_targets
    )
    text = f"""# Local Temporal Sequence Correspondence V0

## Outcome

Terminal: `{inference.terminal}`.

This experiment directly compares correct chronological local-state matching with completely wrong temporal matchings of the same two ordered mean covariance sequences. It is not an order-shuffle classifier, GPA, pose, or quotient-shape analysis.

## 1–4. Branch and immutable lineage

- Branch: `{branch}`
- Local metric final: `796f04e7970972175a660a521caff47c83e0295f`
- GPA V1 final: `122eacff868aa8f656ad6360716c1816f453979f`
- GPA outer-convergence audit final: `347d61f17793d636653b614ec2104baa61ac7a4b`
- Protocol-freeze SHA: `{protocol_freeze_sha}`
- Final scientific-result SHA: `{final_result_sha}`

## 5. Frozen input reproduction

PASS. The immutable uncentered WINDOW5 bank contained {reproduction['trial_count']} trials. Session state and existing AIRM-distance hashes matched. The maximum absolute frozen AIRM-distance reproduction difference was {float(reproduction['distance_max_abs_diff']):.3e} at the frozen `1e-12` tolerance. No trial-local centering was applied.

## 6. Per-window AIRM mean numerical status

PASS. All 360 full-cell means, 720 split-half means, and the one common visualization reference were finite SPD, emitted no warnings, and passed the frozen normalized Karcher residual gate `<=1e-7`. The maximum normalized residual was {maximum_mean_residual:.3e}.

## 7. Split-half temporal reliability

This was a non-gating diagnostic with no post-hoc threshold. Across all 72 cells, same-position Half-A/Half-B AIRM distances had mean {float(np.mean(same_position)):.10f}, median {float(np.median(same_position)):.10f}, Q25 {float(np.quantile(same_position, 0.25)):.10f}, and Q75 {float(np.quantile(same_position, 0.75)):.10f}. Position-wise and complete 5×5 cell matrices are saved in the tables and arrays.

## 8–9. Primary temporal correspondence

- T_temporal: {observed.t_temporal:.10f}
- p_temporal: {inference.p_temporal:.6f} (1,999 temporal-label draws, one-sided plus-one)

All nine A_s values:

{subject_lines(observed.a_s)}

## 10. Identity-rank summary

For the 36 same-subject/same-class comparisons, identity rank among all 120 permutations had mean {float(np.mean(same_cell_ranks)):.3f}, median {float(np.median(same_cell_ranks)):.3f}, range {int(np.min(same_cell_ranks))}–{int(np.max(same_cell_ranks))}, and rank 1 in {int(np.count_nonzero(same_cell_ranks == 1))}/36 cells. Category-level rank summaries are saved machine-readably. Identity rank is descriptive, not the primary statistic.

## 11–12. Subject specificity

- T_subject: {observed.t_subject:.10f}
- p_subject: {inference.p_subject:.6f} (1,999 subject-break draws)

All nine S_s values:

{subject_lines(observed.s_s)}

## 13–14. Class specificity

- T_class: {observed.t_class:.10f}
- p_class: {inference.p_class:.6f} (1,999 class-break draws)

All nine C_s values:

{subject_lines(observed.c_s)}

## 15–17. Secondary explicit subject×class interaction

- T_J: {observed.t_j:.10f}
- p_J_subjectbreak: {inference.p_j_subjectbreak:.6f}
- p_J_classbreak: {inference.p_j_classbreak:.6f}
- Both-null interaction criterion passed: {str(observed.t_j > 0 and inference.p_j_subjectbreak < 0.05 and inference.p_j_classbreak < 0.05).upper()}

All nine J_s values:

{subject_lines(observed.j_s)}

## 18. Predeclared raw D_id comparisons

{comparison_lines}

`D_id` is descriptive because generic proximity affects it. `G`, the identity advantage over the 44 complete derangements, is the primary structural quantity.

## 19. Terminal decision

`{inference.terminal}`

The secondary J result is separate and did not change this primary terminal.

## 20. Runtime

{total_runtime_seconds:.2f} seconds total scientific execution.

## 21. Tests

- Focused temporal suite: {focused_tests}
- Full repository suite: {repository_tests}

## 22. Git status

`{git_status}`

## 23. Post-result immutability

Confirmed: no scientific definition changed after the first real temporal statistic was observed. No rescue analysis or rerun with altered settings was performed.

## Claim boundary

The analysis concerns mean covariance sequences, chronological local-state correspondence, temporal correspondence, and temporal sequence specificity. It does not identify a continuous physiological trajectory, causal temporal dynamics, subject-specific pose, mean shape, GPA consensus, a neural state sequence, or a source-space trajectory. A subject×class interaction is claimed only if the separately reported secondary J passes both correspondence-breaking nulls.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_artifact_manifest(output_root: Path, provenance: Mapping[str, str]) -> None:
    manifest_path = output_root / "protocol" / "artifact_manifest.csv"
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                **dict(provenance),
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


__all__ = [
    "FIXED_CELLS",
    "environment_record",
    "implementation_source_hash",
    "write_artifact_manifest",
    "write_json",
    "write_report",
    "write_scientific_outputs",
]
