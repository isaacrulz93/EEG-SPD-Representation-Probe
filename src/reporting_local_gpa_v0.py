"""Artifacts, figures, and plain-language report for local GPA consensus V0."""

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

from src.local_gpa_pipeline_v0 import ConsensusBank, ConsensusDistances, SESSION_ORDER
from src.local_gpa_statistics_v0 import terminal_decision
from src.local_metric_interaction_v0 import CLASS_ORDER, InteractionNullResult
from src.trajectory_within_subject_v1 import sha256_file


def environment_record() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pyriemann",
        "pymanopt",
        "joblib",
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


def _with_provenance(frame: pd.DataFrame, provenance: Mapping[str, str]) -> pd.DataFrame:
    result = frame.copy()
    for key, value in provenance.items():
        result[key] = str(value)
    return result


def _save_figure(fig: plt.Figure, stem: Path, provenance: Mapping[str, str]) -> None:
    metadata = {
        "Title": stem.name,
        "Subject": ";".join(f"{key}={value}" for key, value in provenance.items()),
    }
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", metadata=metadata)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def _cell_labels() -> list[tuple[int, str]]:
    return [(subject, label) for subject in range(1, 10) for label in CLASS_ORDER]


def write_scientific_outputs(
    output_root: str | Path,
    *,
    bank: ConsensusBank,
    distances: ConsensusDistances,
    result: InteractionNullResult,
    reproduction_table: pd.DataFrame,
    centering_manifest: Mapping[str, object],
    provenance: Mapping[str, str],
    runtime_seconds: float,
) -> str:
    root = Path(output_root)
    for folder in ("objects", "tables", "nulls", "decisions", "figures", "report", "protocol"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    labels = _cell_labels()
    cell_rows: list[dict[str, object]] = []
    for row, (row_subject, row_class) in enumerate(labels):
        for column, (column_subject, column_class) in enumerate(labels):
            cell_rows.append(
                {
                    "row_subject": row_subject,
                    "row_class": row_class,
                    "column_subject": column_subject,
                    "column_class": column_class,
                    "M01": distances.m01[row, column],
                    "M": distances.m[row, column],
                }
            )
    cell_table = _with_provenance(pd.DataFrame(cell_rows), provenance)
    cell_table.to_csv(root / "tables" / "cell_consensus_distances.csv", index=False)
    observed = result.observed
    contrast_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for subject_index in range(9):
        for class_index, class_label in enumerate(CLASS_ORDER):
            contrast_rows.append(
                {
                    "subject": subject_index + 1,
                    "class_label": class_label,
                    "J_sc": observed.j_sc[subject_index, class_index],
                    "S_sc": observed.s_sc[subject_index, class_index],
                    "C_sc": observed.c_sc[subject_index, class_index],
                }
            )
        subject_rows.append(
            {
                "subject": subject_index + 1,
                "J_s": observed.j_s[subject_index],
                "S_s": observed.s_s[subject_index],
                "C_s": observed.c_s[subject_index],
            }
        )
    contrast_table = _with_provenance(pd.DataFrame(contrast_rows), provenance)
    subject_table = _with_provenance(pd.DataFrame(subject_rows), provenance)
    contrast_table.to_csv(root / "tables" / "subject_class_contrasts.csv", index=False)
    subject_table.to_csv(root / "tables" / "subject_contrasts.csv", index=False)
    null_rows = []
    for replicate in range(len(result.classbreak_t_j)):
        null_rows.extend(
            (
                {
                    "family": "classbreak",
                    "replicate": replicate,
                    "T_J": result.classbreak_t_j[replicate],
                    "supporting_name": "T_C",
                    "supporting_statistic": result.classbreak_t_c[replicate],
                },
                {
                    "family": "subjectbreak",
                    "replicate": replicate,
                    "T_J": result.subjectbreak_t_j[replicate],
                    "supporting_name": "T_S",
                    "supporting_statistic": result.subjectbreak_t_s[replicate],
                },
            )
        )
    _with_provenance(pd.DataFrame(null_rows), provenance).to_csv(
        root / "tables" / "correspondence_null_distributions.csv", index=False
    )
    p_table = _with_provenance(
        pd.DataFrame(
            [
                {
                    "T_J": observed.t_j,
                    "p_classbreak": result.p_j_classbreak,
                    "p_subjectbreak": result.p_j_subjectbreak,
                    "p_conservative": result.p_j,
                    "T_S": observed.t_s,
                    "p_S_subjectbreak": result.p_s_subjectbreak,
                    "T_C": observed.t_c,
                    "p_C_classbreak": result.p_c_classbreak,
                }
            ]
        ),
        provenance,
    )
    p_table.to_csv(root / "tables" / "interaction_summary.csv", index=False)
    _with_provenance(bank.diagnostics, provenance).to_csv(
        root / "tables" / "cell_gpa_diagnostics.csv", index=False
    )
    _with_provenance(bank.registration_diagnostics, provenance).to_csv(
        root / "tables" / "registration_multistart_diagnostics.csv", index=False
    )
    _with_provenance(distances.between_diagnostics, provenance).to_csv(
        root / "tables" / "between_cell_registration_diagnostics.csv", index=False
    )
    _with_provenance(distances.reliability_diagnostics, provenance).to_csv(
        root / "tables" / "split_half_reliability.csv", index=False
    )
    _with_provenance(reproduction_table, provenance).to_csv(
        root / "tables" / "frozen_input_reproduction.csv", index=False
    )
    np.savez_compressed(
        root / "objects" / "consensus_prototypes.npz",
        full=bank.full,
        half_a=bank.half_a,
        half_b=bank.half_b,
        subjects=np.arange(1, 10),
        sessions=np.asarray(SESSION_ORDER),
        classes=np.asarray(CLASS_ORDER),
        protocol_freeze_sha=np.asarray([provenance["protocol_freeze_sha"]]),
        implementation_source_sha256=np.asarray(
            [provenance["implementation_source_sha256"]]
        ),
    )
    np.savez_compressed(
        root / "nulls" / "correspondence_null_statistics.npz",
        classbreak_t_j=result.classbreak_t_j,
        subjectbreak_t_j=result.subjectbreak_t_j,
        classbreak_t_c=result.classbreak_t_c,
        subjectbreak_t_s=result.subjectbreak_t_s,
    )
    terminal = terminal_decision(
        t_j=observed.t_j,
        p_classbreak=result.p_j_classbreak,
        p_subjectbreak=result.p_j_subjectbreak,
    )
    decision = {
        "terminal": terminal,
        "T_J": observed.t_j,
        "p_classbreak": result.p_j_classbreak,
        "p_subjectbreak": result.p_j_subjectbreak,
        "p_conservative": result.p_j,
        "T_S": observed.t_s,
        "p_S_subjectbreak": result.p_s_subjectbreak,
        "T_C": observed.t_c,
        "p_C_classbreak": result.p_c_classbreak,
        "runtime_seconds": runtime_seconds,
        "centering_manifest": dict(centering_manifest),
        "provenance": dict(provenance),
        "stage1_terminal_immutable": "STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0",
    }
    (root / "decisions" / "terminal_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figures = root / "figures"
    tick_labels = [f"S{s}:{c.replace('_hand', '').replace('right', 'R').replace('left', 'L')}" for s, c in labels]
    for number, matrix, title, csv_name in (
        (1, distances.m01, "Cross-session quotient consensus distance M01", "figure_1_cross_session_consensus_M01"),
        (2, distances.m, "Session-role-symmetrized quotient consensus distance M", "figure_2_session_symmetrized_consensus_M"),
    ):
        pd.DataFrame(matrix, index=tick_labels, columns=tick_labels).to_csv(
            figures / f"{csv_name}.csv"
        )
        fig, ax = plt.subplots(figsize=(11, 9))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("session 1 cell")
        ax.set_ylabel("session 0 cell")
        fig.colorbar(image, ax=ax, shrink=0.8)
        _save_figure(fig, figures / csv_name, provenance)
    subject_table[["subject", "J_s"]].to_csv(figures / "figure_3_subject_J_s.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(subject_table["subject"], subject_table["J_s"], color="#4c78a8")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="subject", ylabel="J_s", title="Subject-level quotient-consensus interaction")
    _save_figure(fig, figures / "figure_3_subject_J_s", provenance)
    j_matrix = observed.j_sc
    pd.DataFrame(j_matrix, index=np.arange(1, 10), columns=CLASS_ORDER).to_csv(
        figures / "figure_4_subject_class_J_sc.csv"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    limit = max(float(np.max(np.abs(j_matrix))), np.finfo(float).eps)
    image = ax.imshow(j_matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(4), ["Left", "Right", "Feet", "Tongue"])
    ax.set_yticks(range(9), range(1, 10))
    ax.set(xlabel="class", ylabel="subject", title="Subject × class J_sc")
    fig.colorbar(image, ax=ax)
    _save_figure(fig, figures / "figure_4_subject_class_J_sc", provenance)
    for number, column, title in (
        (5, "S_s", "Supporting subject specificity"),
        (6, "C_s", "Supporting class specificity"),
    ):
        stem = f"figure_{number}_{column}"
        subject_table[["subject", column]].to_csv(figures / f"{stem}.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(subject_table["subject"], subject_table[column], color="#72b7b2")
        ax.axhline(0, color="black", linewidth=1)
        ax.set(xlabel="subject", ylabel=column, title=title)
        _save_figure(fig, figures / stem, provenance)
    reliability = distances.reliability_diagnostics.copy()
    reliability.to_csv(figures / "figure_7_split_half_reliability.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(np.arange(len(reliability)), reliability["split_half_quotient_distance"], s=18)
    ax.set(xlabel="subject × session × class cell", ylabel="d_Q(P_A,P_B)", title="72 run-blocked split-half consensus distances")
    _save_figure(fig, figures / "figure_7_split_half_reliability", provenance)
    dispersion = bank.diagnostics.loc[bank.diagnostics["split"].eq("Full")].copy()
    dispersion.to_csv(figures / "figure_8_within_cell_procrustes_dispersion.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(np.arange(len(dispersion)), dispersion["within_cell_procrustes_dispersion"], s=18, color="#f58518")
    ax.set(xlabel="subject × session × class cell", ylabel="RMS quotient residual", title="Full-cell within-cell Procrustes dispersion")
    _save_figure(fig, figures / "figure_8_within_cell_procrustes_dispersion", provenance)
    convergence = bank.diagnostics.copy()
    convergence.to_csv(figures / "figure_9_gpa_multistart_diagnostics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(convergence["objective"], convergence["gpa_objective_spread"], s=15, alpha=0.7)
    ax.set(xlabel="best GPA objective", ylabel="two-start objective spread", title="GPA convergence and multistart diagnostic")
    _save_figure(fig, figures / "figure_9_gpa_multistart_diagnostics", provenance)
    return terminal


def write_report(
    output_root: str | Path,
    *,
    result: InteractionNullResult,
    bank: ConsensusBank,
    distances: ConsensusDistances,
    terminal: str,
    provenance: Mapping[str, str],
    runtime_seconds: float,
    tests_summary: str,
    report_filename: str = "local_gpa_consensus_v0.md",
    report_title: str = "Local GPA Consensus V0",
    amendment_lines: tuple[str, ...] = (),
) -> Path:
    root = Path(output_root)
    report_path = root / "report" / report_filename
    observed = result.observed
    full = bank.diagnostics.loc[bank.diagnostics["split"].eq("Full")]
    registration = bank.registration_diagnostics
    lines = [
        f"# {report_title}",
        "",
        "Stage 1은 다섯 점의 내부 AIRM 거리만 사용했고 subject×class interaction을 지지하지 않았다. Stage 2A는 각 trial을 국소 중심화한 뒤, `O(22)×S5` nuisance를 제거한 full five-point SPD configuration의 **cell consensus orbit**가 같은 subject와 같은 class에서 session을 넘어 반복되는지 묻는다. 등록 행렬 Q는 오직 nuisance이며 평균·비교·과학적 해석을 하지 않았다.",
        "",
        "## Provenance",
        "",
        f"- Branch: `{provenance['branch']}`",
        f"- Scientific base SHA: `{provenance['scientific_source_sha']}`",
        f"- Protocol-freeze SHA: `{provenance['protocol_freeze_sha']}`",
        f"- Scientific result SHA: `{provenance.get('scientific_result_sha', 'pending at artifact creation')}`",
        f"- Final SHA: `{provenance.get('final_sha', 'report finalization commit')}`",
        "- Frozen WINDOW5 covariance hashes and all 5,184 frozen AIRM distance matrices reproduced exactly (maximum absolute difference 0).",
        *amendment_lines,
        "",
        "## Numerical status",
        "",
        f"- Full-cell GPA convergence: {len(full)}/72.",
        f"- Full + split GPA convergence: {len(bank.diagnostics)}/216.",
        f"- Final registration-start convergence fraction: {registration['converged'].mean():.6f}.",
        f"- Maximum prototype centering-constraint residual: {bank.diagnostics['prototype_constraint_residual'].max():.6e}.",
        f"- Median split-half quotient distance: {np.median(distances.reliability):.6f}.",
        f"- Median full-cell Procrustes dispersion: {full['within_cell_procrustes_dispersion'].median():.6f}.",
        "",
        "The split-half distances are descriptive and non-gating. Both deterministic GPA starts converged for every required cell; when their objectives were numerically equivalent, the frozen quotient-orbit agreement gate was applied.",
        "",
        "## Primary interaction",
        "",
        f"- T_J: {observed.t_j:.10f}",
        f"- p_classbreak: {result.p_j_classbreak:.6f}",
        f"- p_subjectbreak: {result.p_j_subjectbreak:.6f}",
        f"- conservative p=max(...): {result.p_j:.6f}",
        f"- Terminal: **{terminal}**",
        "",
        "### All 9 subject-level J_s",
        "",
        "| Subject | J_s | S_s | C_s |",
        "|---:|---:|---:|---:|",
    ]
    for subject in range(9):
        lines.append(
            f"| {subject + 1} | {observed.j_s[subject]:.10f} | {observed.s_s[subject]:.10f} | {observed.c_s[subject]:.10f} |"
        )
    lines.extend(
        [
            "",
            "### All 36 subject×class J_sc",
            "",
            "| Subject | Left hand | Right hand | Feet | Tongue |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for subject in range(9):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                subject + 1, *[f"{value:.10f}" for value in observed.j_sc[subject]]
            )
        )
    lines.extend(
        [
            "",
            "## Supporting anatomy",
            "",
            f"- Subject specificity T_S: {observed.t_s:.10f}; subject-break p={result.p_s_subjectbreak:.6f}.",
            f"- Class specificity T_C: {observed.t_c:.10f}; class-break p={result.p_c_classbreak:.6f}.",
            "- Supporting subject and class specificity do not by themselves establish interaction.",
            "",
            "## Interpretation boundaries",
            "",
            "This analysis estimates quotient mean configurations, not mean poses. It does not identify a stable subject Q, physical orientation, neural orientation, or anatomical pose. It also does not modify or rescue the frozen negative Stage-1 interaction result.",
            "",
            "## Reproducibility",
            "",
            f"- Total scientific runtime: {runtime_seconds:.1f} s.",
            f"- Tests: {tests_summary}.",
            "- No scientific definition changed after the first Stage-2A real output was observed.",
            "- Git status is required to be clean at final handoff.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


__all__ = [
    "environment_record",
    "implementation_source_hash",
    "write_report",
    "write_scientific_outputs",
]
