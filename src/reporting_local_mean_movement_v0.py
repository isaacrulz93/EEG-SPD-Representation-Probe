"""Output, figure, provenance, and report helpers for movement V0."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.local_mean_movement_v0 import CLASS_ORDER, MovementInference, MovementPCA


CLASS_DISPLAY = {
    "left_hand": "Left",
    "right_hand": "Right",
    "feet": "Feet",
    "tongue": "Tongue",
}
SESSION_DISPLAY = {"0train": "Session 0", "1test": "Session 1"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def environment_record() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "pyriemann", "pymanopt", "joblib", "matplotlib"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:  # pragma: no cover - provenance fallback
            packages[name] = f"unavailable:{error}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def implementation_source_hash(root: Path, paths: Sequence[str]) -> tuple[str, dict[str, str]]:
    file_hashes = {path: sha256_file(root / path) for path in paths}
    digest = hashlib.sha256()
    for path in sorted(file_hashes):
        digest.update(path.encode())
        digest.update(file_hashes[path].encode())
    return digest.hexdigest(), file_hashes


def write_artifact_manifest(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    manifest = output_root / "protocol" / "artifact_manifest.csv"
    for path in sorted(value for value in output_root.rglob("*") if value.is_file()):
        if path == manifest:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(output_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)


def cell_labels() -> list[str]:
    return [
        f"S{subject} {CLASS_DISPLAY[class_label]}"
        for subject in range(1, 10)
        for class_label in CLASS_ORDER
    ]


def write_matrix_csv(path: Path, values: np.ndarray) -> None:
    labels = cell_labels()
    frame = pd.DataFrame(np.asarray(values), index=labels, columns=labels)
    frame.index.name = "session0_anchor"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_common_pca(pca: MovementPCA, output_root: Path) -> None:
    fixed = ((0, 0), (0, 2), (1, 0), (1, 2))
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    colors = ("#2563eb", "#dc2626")
    markers = ("o", "s")
    for axis, (subject, class_index) in zip(axes.flat, fixed, strict=True):
        for session in range(2):
            points = pca.coordinates[session, subject, class_index]
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=colors[session],
                marker=markers[session],
                linewidth=1.8,
                label=SESSION_DISPLAY[("0train", "1test")[session]],
            )
            for step in range(3):
                axis.annotate(
                    "",
                    xy=points[step + 1],
                    xytext=points[step],
                    arrowprops={"arrowstyle": "->", "color": colors[session], "lw": 1.4},
                )
            for step, point in enumerate(points, start=1):
                axis.text(point[0], point[1], str(step), fontsize=8)
        axis.set_title(f"Subject {subject + 1} — {CLASS_DISPLAY[CLASS_ORDER[class_index]]}")
        axis.set_xlabel("Global movement PCA 1")
        axis.set_ylabel("Global movement PCA 2")
        axis.axhline(0.0, color="0.85", linewidth=0.7)
        axis.axvline(0.0, color="0.85", linewidth=0.7)
    axes.flat[0].legend(frameon=False)
    figure.suptitle("Ordered anti-developed displacement points (descriptive only)")
    _save_figure(figure, output_root / "figures" / "figure_1_common_movement_pca")


def plot_movement_heatmap(matrix: np.ndarray, output_root: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", aspect="equal")
    ticks = np.arange(0, 36, 4)
    axis.set_xticks(ticks, [f"S{value}" for value in range(1, 10)], rotation=45, ha="right")
    axis.set_yticks(ticks, [f"S{value}" for value in range(1, 10)])
    axis.set_xlabel("Session-1 subject × class cell")
    axis.set_ylabel("Session-0 subject × class cell")
    axis.set_title("36×36 common-O quotient movement discrepancy")
    figure.colorbar(image, ax=axis, label="d_mov")
    _save_figure(figure, output_root / "figures" / "figure_2_d_mov_heatmap")


def plot_subject_contrasts(inference: MovementInference, output_root: Path) -> None:
    subjects = np.arange(1, 10)
    width = 0.24
    observed = inference.observed
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.bar(subjects - width, observed.s_s, width, label="S_s", color="#2563eb")
    axis.bar(subjects, observed.c_s, width, label="C_s", color="#16a34a")
    axis.bar(subjects + width, observed.j_s, width, label="J_s", color="#dc2626")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(subjects)
    axis.set_xlabel("Subject")
    axis.set_ylabel("Distance contrast")
    axis.set_title("Subject-level quotient movement contrasts")
    axis.legend(frameon=False)
    _save_figure(figure, output_root / "figures" / "figure_3_subject_contrasts")


def plot_split_reliability(table: pd.DataFrame, output_root: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for session, color in (("0train", "#2563eb"), ("1test", "#dc2626")):
        values = table.loc[table["session"] == session, "d_mov"].to_numpy()
        axes[0].hist(values, bins=12, alpha=0.55, color=color, label=SESSION_DISPLAY[session])
    axes[0].set_xlabel("Half-A vs Half-B d_mov")
    axes[0].set_ylabel("Cells")
    axes[0].set_title("Split-half movement reliability distances")
    axes[0].legend(frameon=False)
    grouped = table.groupby("subject", sort=True)["d_mov"].mean()
    axes[1].bar(grouped.index, grouped.values, color="#7c3aed")
    axes[1].set_xlabel("Subject")
    axes[1].set_ylabel("Mean split-half d_mov")
    axes[1].set_title("Non-gating subject summaries")
    _save_figure(figure, output_root / "figures" / "figure_4_split_half_reliability")


def plot_speed_profiles(speeds: np.ndarray, output_root: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    colors = ("#2563eb", "#f59e0b", "#16a34a", "#dc2626")
    for session, axis in enumerate(axes):
        for subject in range(9):
            for class_index in range(4):
                axis.plot(
                    np.arange(1, 5),
                    speeds[session, subject, class_index],
                    color=colors[class_index],
                    alpha=0.30,
                    linewidth=0.9,
                )
        axis.set_title(SESSION_DISPLAY[("0train", "1test")[session]])
        axis.set_xlabel("Ordered transition")
        axis.set_xticks(np.arange(1, 5))
    axes[0].set_ylabel("AIRM displacement per second")
    figure.suptitle("All ordered four-step speed profiles")
    _save_figure(figure, output_root / "figures" / "figure_5_speed_profiles")


def write_figures(
    *,
    output_root: Path,
    pca: MovementPCA,
    movement_matrix: np.ndarray,
    movement_inference: MovementInference,
    split_table: pd.DataFrame,
    speeds: np.ndarray,
) -> None:
    plot_common_pca(pca, output_root)
    plot_movement_heatmap(movement_matrix, output_root)
    plot_subject_contrasts(movement_inference, output_root)
    plot_split_reliability(split_table, output_root)
    plot_speed_profiles(speeds, output_root)


def _format_vector(values: np.ndarray) -> str:
    return ", ".join(f"S{index + 1}={float(value):.10f}" for index, value in enumerate(values))


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.loc[:, columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.10f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join((header, divider, *rows))


def _interaction_pass(inference: MovementInference) -> bool:
    return bool(
        inference.observed.t_j > 0.0
        and inference.p_j_subjectbreak < 0.05
        and inference.p_j_classbreak < 0.05
    )


def write_report(
    path: Path,
    *,
    branch: str,
    parent_final_sha: str,
    parent_protocol_sha: str,
    parent_result_sha: str,
    protocol_freeze_sha: str,
    reproduction: Mapping[str, Any],
    geometry_summary: Mapping[str, Any],
    synthetic_gates: Mapping[str, Any],
    optimizer_summary: Mapping[str, Any],
    movement: MovementInference,
    length: MovementInference,
    direct: MovementInference,
    split_table: pd.DataFrame,
    fixed_comparisons: pd.DataFrame,
    terminal: str,
    runtime: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> None:
    movement_interaction = _interaction_pass(movement)
    direct_interaction = _interaction_pass(direct)
    length_interaction = _interaction_pass(length)
    if terminal == "GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT":
        interpretation = (
            "After removing the initial SPD location and one common sequence-level "
            "orthogonal gauge, the ordered displacement pattern of the window-wise mean "
            "covariance trajectory is cross-session reproducible and subject×class-specific."
        )
    elif terminal == "GO_REPRODUCIBLE_ORDERED_MOVEMENT_WITHOUT_INTERACTION":
        interpretation = (
            "The ordered AIRM displacement sequence supports subject and class specificity, "
            "but the separately required distance-level subject×class interaction does not pass."
        )
    else:
        interpretation = (
            "The frozen criteria do not support the full reproducible ordered-movement terminal."
        )
    control_notes = []
    if direct_interaction and not movement_interaction:
        control_notes.append(
            "The reproducible movement information is concentrated in the montage-registered "
            "sequence orientation removed by the common O(d) quotient."
        )
    if length_interaction:
        control_notes.append(
            "The magnitude-only control also passes; ordered displacement magnitude contributes "
            "information, without establishing a unique directional contribution."
        )
    if not control_notes:
        control_notes.append("Neither control changes the primary terminal.")

    lines = [
        "# Local Mean Covariance Movement V0",
        "",
        "## Provenance and frozen scope",
        "",
        f"- Branch: `{branch}`",
        f"- Finalized temporal parent HEAD: `{parent_final_sha}`",
        f"- Temporal protocol freeze: `{parent_protocol_sha}`",
        f"- Temporal scientific result: `{parent_result_sha}`",
        f"- Protocol freeze SHA: `{protocol_freeze_sha}`",
        "- Final result SHA: `FINAL_RESULT_SHA_PENDING`",
        "- Scientific object: window-wise mean covariance movement, not trial-level or continuous velocity.",
        "",
        "The full and split-half mean artifacts were loaded unchanged from the finalized temporal result. "
        f"Full SHA-256 `{reproduction['full_artifact_sha256']}`; split SHA-256 "
        f"`{reproduction['split_artifact_sha256']}`. All {reproduction['matrix_count']} saved mean "
        f"matrices matched the frozen result exactly; maximum absolute difference "
        f"`{float(reproduction['maximum_absolute_difference']):.3e}`.",
        "",
        "## Anti-development mathematical gates",
        "",
        f"All {geometry_summary['sequence_count']} full/split mean sequences and "
        f"{geometry_summary['transition_count']} transitions passed. Maximum norm-identity error: "
        f"`{float(geometry_summary['maximum_norm_absolute_error']):.3e}`; maximum edgewise transport "
        f"relative error: `{float(geometry_summary['maximum_edge_transport_relative_error']):.3e}`; "
        f"maximum Z symmetry relative error: `{float(geometry_summary['maximum_z_symmetry_relative_error']):.3e}`.",
        "",
        f"The synthetic d=22 common-congruence check used one common O across all four steps and had "
        f"maximum relative error `{float(synthetic_gates['common_o_maximum_transition_relative_error']):.3e}`. "
        f"The known-Q quotient distance was `{float(synthetic_gates['known_q_distance']):.3e}`.",
        "",
        "## Quotient optimizer status",
        "",
        f"Status: **{optimizer_summary['status']}**. TrustRegions used "
        f"{optimizer_summary['starts_per_fit']} deterministic starts per fit with equal coverage of both "
        f"determinant sectors. Certified primary fits: {optimizer_summary['primary_fit_count']}; "
        f"certified split-half fits: {optimizer_summary['split_fit_count']}. Maximum selected projected "
        f"gradient norm: `{float(optimizer_summary['maximum_selected_gradient_norm']):.3e}`. "
        f"Synthetic forward/reverse equality: `{synthetic_gates['forward_reverse_exact']}`; gradient "
        f"finite-difference absolute error: `{float(synthetic_gates['gradient_absolute_error']):.3e}`.",
        "",
        "Fitted Q matrices are nuisance quotient variables and are not interpreted scientifically.",
        "",
        "## Primary common-O quotient movement results",
        "",
        f"- `T_subject = {movement.observed.t_subject:.10f}`, `p_subject = {movement.p_subject:.6f}`",
        f"- All S_s: {_format_vector(movement.observed.s_s)}",
        f"- `T_class = {movement.observed.t_class:.10f}`, `p_class = {movement.p_class:.6f}`",
        f"- All C_s: {_format_vector(movement.observed.c_s)}",
        f"- `T_J = {movement.observed.t_j:.10f}`, `p_J_subjectbreak = {movement.p_j_subjectbreak:.6f}`, "
        f"`p_J_classbreak = {movement.p_j_classbreak:.6f}`",
        f"- All J_s: {_format_vector(movement.observed.j_s)}",
        "",
        "All p-values are one-sided plus-one values from exactly 1,999 inherited whole-cell relabelings.",
        "",
        "## Magnitude-only ordered speed control",
        "",
        f"- `T_subject = {length.observed.t_subject:.10f}`, `p_subject = {length.p_subject:.6f}`",
        f"- `T_class = {length.observed.t_class:.10f}`, `p_class = {length.p_class:.6f}`",
        f"- `T_J = {length.observed.t_j:.10f}`, `p_J_subjectbreak = {length.p_j_subjectbreak:.6f}`, "
        f"`p_J_classbreak = {length.p_j_classbreak:.6f}`",
        f"- S_s: {_format_vector(length.observed.s_s)}",
        f"- C_s: {_format_vector(length.observed.c_s)}",
        f"- J_s: {_format_vector(length.observed.j_s)}",
        "",
        "## Direct montage-registered control",
        "",
        f"- `T_subject = {direct.observed.t_subject:.10f}`, `p_subject = {direct.p_subject:.6f}`",
        f"- `T_class = {direct.observed.t_class:.10f}`, `p_class = {direct.p_class:.6f}`",
        f"- `T_J = {direct.observed.t_j:.10f}`, `p_J_subjectbreak = {direct.p_j_subjectbreak:.6f}`, "
        f"`p_J_classbreak = {direct.p_j_classbreak:.6f}`",
        f"- S_s: {_format_vector(direct.observed.s_s)}",
        f"- C_s: {_format_vector(direct.observed.c_s)}",
        f"- J_s: {_format_vector(direct.observed.j_s)}",
        "",
        *control_notes,
        "",
        "## Split-half movement reliability (non-gating)",
        "",
        f"All 72 distances are reported below. Mean `{split_table['d_mov'].mean():.10f}`, median "
        f"`{split_table['d_mov'].median():.10f}`, range "
        f"`[{split_table['d_mov'].min():.10f}, {split_table['d_mov'].max():.10f}]`. No threshold was applied.",
        "",
        _markdown_table(split_table, ("subject", "session", "class", "d_mov")),
        "",
        "## Fixed illustrative comparisons",
        "",
        _markdown_table(fixed_comparisons, ("comparison", "d_mov", "d_len", "d_direct")),
        "",
        "## Terminal and interpretation",
        "",
        f"`{terminal}`",
        "",
        interpretation,
        "",
        "This result concerns an ordered discrete AIRM anti-development of a window-wise mean "
        "covariance trajectory. It does not establish individual-trial neural velocity, continuous-time "
        "dynamics, causal or physiological state transitions, source-space dynamics, physical sensor "
        "orientation, an absolute subject pose, biological privilege of AIRM, or completeness of five windows.",
        "",
        "The prior unordered metric analysis supported subject and class specificity but not its explicit "
        "interaction. The ordered raw mean-sequence analysis supported temporal correspondence, subject "
        "specificity, class specificity, and interaction. This experiment replaces absolute point placement "
        "with ordered adjacent relative movement; it is not a rescue of the unordered result.",
        "",
        "## Runtime, tests, and immutability",
        "",
        f"- Total scientific runtime: `{float(runtime['total_seconds']):.3f}` seconds",
        f"- Quotient matrix runtime: `{float(runtime['primary_quotient_seconds']):.3f}` seconds",
        f"- Split-half quotient runtime: `{float(runtime['split_quotient_seconds']):.3f}` seconds",
        f"- Focused tests: `{tests['focused']}`",
        f"- Full repository tests: `{tests['full_repository']}`",
        "- Git status at scientific execution start: clean; HEAD equaled the protocol-freeze SHA.",
        "- Git status immediately before report-only result finalization: clean at the scientific-result commit.",
        "- Final handoff status is verified after committing this report.",
        "- No scientific setting changed after the first movement result was observed. Finalization is limited "
        "to inserting the result commit SHA, post-result provenance, and refreshed artifact hashes.",
        "",
        "Descriptive PCA uses one global Frobenius-isometric svec basis over all 72×4 full movement points. "
        "No inference was performed in PCA space.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "CLASS_DISPLAY",
    "environment_record",
    "git_value",
    "implementation_source_hash",
    "sha256_file",
    "write_artifact_manifest",
    "write_figures",
    "write_json",
    "write_matrix_csv",
    "write_report",
]
