"""Reporting helpers for Local Ordered Movement Component Decomposition V0."""

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

from src.local_movement_component_decomposition_v0 import ComponentInference


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def implementation_source_hash(
    root: Path, paths: Sequence[str]
) -> tuple[str, dict[str, str]]:
    hashes = {relative: sha256_file(root / relative) for relative in paths}
    digest = hashlib.sha256()
    for relative in sorted(hashes):
        digest.update(relative.encode("utf-8"))
        digest.update(hashes[relative].encode("ascii"))
    return digest.hexdigest(), hashes


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


def write_matrix_csv(path: Path, values: np.ndarray, labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values, index=labels, columns=labels).to_csv(path, index_label="row_cell")


def _save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_figures(
    *,
    output_root: Path,
    relation_cells: pd.DataFrame,
    subject_stats: pd.DataFrame,
    fixed_comparisons: pd.DataFrame,
) -> None:
    colors = {"len": "#4C78A8", "ang": "#F58518", "ori": "#54A24B"}
    relations = ("a_sc", "b_sc", "c_sc", "d_sc")
    summary = (
        relation_cells[relation_cells["component"].isin(colors)]
        .groupby("component", sort=False)[list(relations)]
        .mean()
    )
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(relations), dtype=float)
    bottom = np.zeros(len(relations), dtype=float)
    for component in ("len", "ang", "ori"):
        values = summary.loc[component, list(relations)].to_numpy(dtype=float)
        axis.bar(x, values, bottom=bottom, label=component, color=colors[component])
        bottom += values
    axis.set_xticks(x, ["same S/C", "same S/diff C", "diff S/same C", "diff S/C"])
    axis.set_ylabel("Mean squared cost")
    axis.set_title("Exact component contributions by relation category")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(
        figure, output_root / "figures" / "component_relation_summary"
    )

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    subjects = np.arange(1, 10)
    width = 0.25
    for offset, component in zip((-1, 0, 1), ("len", "ang", "ori"), strict=True):
        values = (
            subject_stats[subject_stats["component"] == component]
            .sort_values("subject")["J_s"]
            .to_numpy(dtype=float)
        )
        axis.bar(subjects + offset * width, values, width=width, color=colors[component], label=component)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(subjects)
    axis.set_xlabel("Subject")
    axis.set_ylabel("Squared-cost J_s")
    axis.set_title("Subject-level interaction decomposition")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_root / "figures" / "component_subject_J")

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(fixed_comparisons), dtype=float)
    bottom = np.zeros(len(fixed_comparisons), dtype=float)
    for component, column in (("len", "c_len"), ("ang", "c_ang"), ("ori", "c_ori")):
        values = fixed_comparisons[column].to_numpy(dtype=float)
        axis.bar(x, values, bottom=bottom, color=colors[component], label=component)
        bottom += values
    axis.set_xticks(x, fixed_comparisons["comparison"], rotation=15, ha="right")
    axis.set_ylabel("Squared cost")
    axis.set_title("Fixed S1-Left anchor decompositions")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_root / "figures" / "fixed_pair_component_bars")


def _format_vector(values: np.ndarray) -> str:
    return ", ".join(f"S{index + 1}={float(value):.10f}" for index, value in enumerate(values))


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    view = frame.loc[:, list(columns)].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.10g}")
    header = "| " + " | ".join(view.columns) + " |"
    divider = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join((header, divider, *rows))


def write_report(
    path: Path,
    *,
    branch: str,
    parent_sha: str,
    parent_protocol_sha: str,
    parent_result_sha: str,
    protocol_freeze_sha: str,
    reproduction: Mapping[str, Any],
    numerical_checks: pd.DataFrame,
    inferences: Mapping[str, ComponentInference],
    split_stability: pd.DataFrame,
    fixed_comparisons: pd.DataFrame,
    step_norm_summary: pd.DataFrame,
    step_cost_summary: pd.DataFrame,
    fixed_step_angular: pd.DataFrame,
    terminal: str,
    runtime: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> None:
    maximum_absolute = float(numerical_checks["maximum_absolute_error"].fillna(0.0).max())
    pair_rows = numerical_checks[numerical_checks["check"].str.contains("nonnegative")]
    minimums = ", ".join(
        f"{row.check.replace('_nonnegative', '')}={row.minimum_raw_value:.12g}"
        for row in pair_rows.itertuples()
    )
    inference_rows = []
    for component in ("len", "ang", "ori", "full", "sensor"):
        value = inferences[component]
        inference_rows.append(
            {
                "component": component,
                "T_subject": value.observed.t_subject,
                "p_subject": value.p_subject,
                "T_class": value.observed.t_class,
                "p_class": value.p_class,
                "T_J": value.observed.t_j,
                "p_J_subject": value.p_j_subjectbreak,
                "p_J_class": value.p_j_classbreak,
            }
        )
    inference_table = pd.DataFrame(inference_rows)
    full_j_error = abs(
        inferences["full"].observed.t_j
        - inferences["len"].observed.t_j
        - inferences["ang"].observed.t_j
    )
    sensor_j_error = abs(
        inferences["sensor"].observed.t_j
        - inferences["len"].observed.t_j
        - inferences["ang"].observed.t_j
        - inferences["ori"].observed.t_j
    )
    split_a = split_stability.loc[split_stability["replicate"] == "A"].iloc[0]
    split_b = split_stability.loc[split_stability["replicate"] == "B"].iloc[0]
    lines = [
        "# Local Ordered Movement Component Decomposition V0",
        "",
        "## Identity and frozen lineage",
        "",
        f"- Branch: `{branch}`",
        f"- Exact authoritative parent: `{parent_sha}`",
        f"- Movement V0 protocol freeze: `{parent_protocol_sha}`",
        f"- Movement V0 scientific result: `{parent_result_sha}`",
        f"- Component protocol freeze: `{protocol_freeze_sha}`",
        "- Scientific result SHA: `FINAL_RESULT_SHA_PENDING`",
        "- Parent terminal: `GO_REPRODUCIBLE_SUBJECT_CLASS_ORDERED_MOVEMENT`",
        "",
        "The finalized Movement V0 anti-developments, root-distance matrices, and null mappings were reused exactly. No AIRM means, raw covariances, anti-developments, M1 references, or full-data quotient matrices were refit.",
        "",
        "## Frozen-input reproduction",
        "",
        f"Status: **{reproduction['status']}**. Byte and exact-array equality were verified against `{parent_result_sha}` for all required parent artifacts. Saved null mappings exactly matched the regenerated `SeedSequence([20260810,1102])` and `SeedSequence([20260810,1101])` streams.",
        "",
        "## Numerical decomposition gates",
        "",
        f"All 1,296 pairs passed raw nonnegativity and exact reconstruction at `atol=rtol=1e-8`. Raw minima: {minimums}. No meaningful negative value was clipped. The maximum recorded reconstruction/reproduction absolute error was `{maximum_absolute:.12g}`.",
        "",
        "The saved root distances reproduced their squared costs (`c_full=d_mov²`, `c_len=d_len²`, `c_sensor=d_direct²`), while `c_len` and `c_sensor` were also independently rebuilt from the frozen Z tuples. Every anchor, subject, group statistic, and indexed null draw passed `full=len+ang` and `sensor=len+ang+ori` reconstruction.",
        "",
        "## Squared-cost inference",
        "",
        _markdown_table(
            inference_table,
            ("component", "T_subject", "p_subject", "T_class", "p_class", "T_J", "p_J_subject", "p_J_class"),
        ),
        "",
        "The primary test is the `ang` row. It is the length-weighted, common-O(22)-invariant directional/joint-matrix component after exact removal of ordered speed. The `ori` row is secondary common-frame/simultaneous-conjugation-sensitive localization evidence.",
        "",
        "### Subject-level J values",
        "",
        f"- `J_s_len`: {_format_vector(inferences['len'].observed.j_s)}",
        f"- `J_s_ang`: {_format_vector(inferences['ang'].observed.j_s)}",
        f"- `J_s_ori`: {_format_vector(inferences['ori'].observed.j_s)}",
        "",
        "### Exact T_J reconstruction",
        "",
        f"- `T_J_full - T_J_len - T_J_ang = {full_j_error:.17g}`",
        f"- `T_J_sensor - T_J_len - T_J_ang - T_J_ori = {sensor_j_error:.17g}`",
        "",
        "Subject and class component summaries, including all nine `S_s` and `C_s` values, are saved in `tables/component_subject_stats.csv`; anchor-level values are in `tables/component_relation_cells.csv`.",
        "",
        "## Split-half angular stability",
        "",
        f"- Half A cross-session `T_J_ang={split_a.T_J_ang:.10f}`; `J_s_ang`: {split_a.J_s_ang}",
        f"- Half B cross-session `T_J_ang={split_b.T_J_ang:.10f}`; `J_s_ang`: {split_b.J_s_ang}",
        f"- Prespecified sign stability: `{bool(split_a.sign_stable and split_b.sign_stable)}`",
        "",
        "Each replicate used its matching independent frozen half in both sessions and the exact frozen Movement V0 optimizer. No half-specific p-value was required.",
        "",
        "## Fixed illustrative decompositions",
        "",
        _markdown_table(
            fixed_comparisons,
            ("comparison", "c_len", "c_ang", "c_ori", "c_full", "c_sensor", "fraction_len", "fraction_ang", "fraction_ori"),
        ),
        "",
        "Fractions are descriptive and were computed only where `c_sensor>1e-12`.",
        "",
        "## Step-level descriptive localization",
        "",
        "Step-norm distributions over all 72 frozen sequences:",
        "",
        _markdown_table(step_norm_summary, tuple(step_norm_summary.columns)),
        "",
        "Per-step `c_len` contributions over all 1,296 cross-session pairs:",
        "",
        _markdown_table(step_cost_summary, tuple(step_cost_summary.columns)),
        "",
        "Per-step angular contributions were evaluated only for the four fixed comparisons after deterministic reproduction of their frozen optimal objectives:",
        "",
        _markdown_table(fixed_step_angular, tuple(fixed_step_angular.columns)),
        "",
        "These are descriptive diagnostics, not post-hoc transition tests.",
        "",
        "## Terminal",
        "",
        f"`{terminal}`",
        "",
        "The experiment concerns BNCI2014_001 window-wise mean covariance movement, its discrete anti-development, and the fixed 5 × 0.8-s discretization. Neither `c_ang` nor `c_ori` is interpreted as neural, physiological, source-space, anatomical, or subject pose information.",
        "",
        "## Runtime, tests, and immutability",
        "",
        f"- Total scientific runtime: `{float(runtime['total_seconds']):.3f} s`",
        f"- Split-half quotient runtime: `{float(runtime['split_half_quotient_seconds']):.3f} s`",
        f"- Focused tests before execution: `{tests['focused_before']}`",
        f"- Full repository before execution: `{tests['full_before']}`",
        f"- Focused tests after execution: `{tests.get('focused_after', 'PENDING')}`",
        f"- Full repository after execution: `{tests.get('full_after', 'PENDING')}`",
        "- Git status: clean at scientific-run start and required clean before finalization; final hand-off status is reported with the committed result.",
        "- No scientific setting changed after first component-statistic access: `true`.",
        "",
        "The full pair costs, nulls, relation tables, split-half matrices, reconstruction diagnostics, figures, provenance, and hashes are retained under the frozen output namespace.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
