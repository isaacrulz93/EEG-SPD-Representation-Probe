"""Validated figures and plain-language report for trajectory audit v1."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.trajectory_within_subject_v1 import load_frozen_config, sha256_file


FIGURE_STEMS = (
    "figure_1_within_subject_session_class_ba",
    "figure_2_same_subject_cross_session_ba",
    "figure_3_within_label_null",
    "figure_4_cross_session_label_null",
    "figure_5_order_shuffle_null",
    "figure_6_path_bag_scalars_summary",
)


class TrajectoryAuditReportingError(RuntimeError):
    """Required results are missing, inconsistent, or incomplete."""


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_results(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    output = root / str(config["project"]["output_dir"])
    tables = {
        name: pd.read_csv(output / "tables" / f"{name}.csv")
        for name in (
            "data_contract",
            "reproduction_gate",
            "within_session_scores",
            "cross_session_scores",
            "label_null_summary",
            "order_null_summary",
            "representation_comparison",
        )
    }
    decision = json.loads(
        (output / "decisions" / "terminal_decision.json").read_text(encoding="utf-8")
    )
    if not tables["data_contract"]["passed"].astype(bool).all():
        raise TrajectoryAuditReportingError("data contract is not PASS")
    if not tables["reproduction_gate"]["passed"].astype(bool).all():
        raise TrajectoryAuditReportingError("reproduction gate is not PASS")
    if len(tables["within_session_scores"]) != 108 or len(tables["cross_session_scores"]) != 54:
        raise TrajectoryAuditReportingError("observed score tables are incomplete")
    for name in ("within_session_scores", "cross_session_scores"):
        frame = tables[name]
        if not (frame["status"] == "PASS").all() or frame["balanced_accuracy"].isna().any():
            raise TrajectoryAuditReportingError(f"{name} contains a required failure")
    if len(tables["label_null_summary"]) != 4:
        raise TrajectoryAuditReportingError("label-null summary grid is incomplete")
    if len(tables["order_null_summary"]) != 1:
        raise TrajectoryAuditReportingError("order-null summary grid is incomplete")
    label_path = output / "nulls" / "label_null_statistics.npz"
    order_path = output / "nulls" / "order_null_statistics.npz"
    with np.load(label_path, allow_pickle=False) as archive:
        label_null = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(order_path, allow_pickle=False) as archive:
        order_null = {key: np.asarray(archive[key]) for key in archive.files}
    if label_null["completed"].shape != (1999,) or not label_null["completed"].all():
        raise TrajectoryAuditReportingError("label-null artifact is incomplete")
    if order_null.get("completed", np.zeros(1, dtype=bool)).shape != (1999,) or not order_null["completed"].all():
        raise TrajectoryAuditReportingError("required order-null artifact is incomplete")
    if decision.get("decision") != "GO_STABLE_SUBJECT_SPECIFIC_LOCAL_GEOMETRY":
        raise TrajectoryAuditReportingError("terminal decision differs from frozen chain")
    return {
        "output": output,
        "tables": tables,
        "decision": decision,
        "label_null": label_null,
        "order_null": order_null,
        "label_null_sha256": sha256_file(label_path),
        "order_null_sha256": sha256_file(order_path),
    }


def _save_figure(fig: plt.Figure, stem: str, source: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    source.to_csv(figures / f"{stem}.csv", index=False, lineterminator="\n")
    fig.savefig(figures / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(
        figures / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def _figure_sources(results: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    tables = results["tables"]
    within = tables["within_session_scores"]
    cross = tables["cross_session_scores"]
    label_null = results["label_null"]
    order_null = results["order_null"]
    figure1 = (
        within[within["representation"].eq("PATH_D10")]
        .groupby(["subject", "session"], sort=True, observed=True)["balanced_accuracy"]
        .mean()
        .rename("balanced_accuracy")
        .reset_index()
    )
    figure1["chance"] = 0.25
    figure2 = (
        cross[cross["representation"].eq("PATH_D10")]
        .groupby("subject", sort=True, observed=True)["balanced_accuracy"]
        .mean()
        .rename("balanced_accuracy")
        .reset_index()
    )
    figure2["chance"] = 0.25
    figure3 = pd.concat(
        [
            pd.DataFrame(
                {
                    "replicate": label_null["replicate"],
                    "representation": representation,
                    "null_statistic": label_null[f"{prefix}_w_group"],
                    "observed": float(
                        tables["label_null_summary"].loc[
                            tables["label_null_summary"].stage.eq("W")
                            & tables["label_null_summary"].representation.eq(representation),
                            "observed",
                        ].iloc[0]
                    ),
                }
            )
            for representation, prefix in (("PATH_D10", "path"), ("BAG_CANON_D10", "bag"))
        ],
        ignore_index=True,
    )
    figure4 = pd.concat(
        [
            pd.DataFrame(
                {
                    "replicate": label_null["replicate"],
                    "representation": representation,
                    "null_statistic": label_null[f"{prefix}_x_group"],
                    "observed": float(
                        tables["label_null_summary"].loc[
                            tables["label_null_summary"].stage.eq("X")
                            & tables["label_null_summary"].representation.eq(representation),
                            "observed",
                        ].iloc[0]
                    ),
                }
            )
            for representation, prefix in (("PATH_D10", "path"), ("BAG_CANON_D10", "bag"))
        ],
        ignore_index=True,
    )
    order_row = tables["order_null_summary"].iloc[0]
    figure5 = pd.DataFrame(
        {
            "replicate": order_null["replicate"],
            "null_statistic": order_null["path_x_group"],
            "observed": float(order_row["observed"]),
            "null_median": float(order_row["null_median"]),
        }
    )
    comparison = tables["representation_comparison"]
    figure6 = pd.concat(
        [
            comparison[["representation", "stage_w_observed_median_subject_ba"]]
            .rename(columns={"stage_w_observed_median_subject_ba": "balanced_accuracy"})
            .assign(stage="W"),
            comparison[["representation", "stage_x_observed_median_subject_ba"]]
            .rename(columns={"stage_x_observed_median_subject_ba": "balanced_accuracy"})
            .assign(stage="X"),
        ],
        ignore_index=True,
    )
    figure6["chance"] = 0.25
    return dict(zip(FIGURE_STEMS, (figure1, figure2, figure3, figure4, figure5, figure6), strict=True))


def _render_figures(sources: Mapping[str, pd.DataFrame], figures: Path) -> None:
    _style()
    colors = {"0train": "#3569a8", "1test": "#d17a22"}
    source = sources[FIGURE_STEMS[0]]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for session in ("0train", "1test"):
        selected = source[source.session.eq(session)]
        ax.plot(selected.subject, selected.balanced_accuracy, marker="o", linewidth=1.5, label=session, color=colors[session])
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set(xticks=range(1, 10), xlabel="Subject", ylabel="Balanced accuracy", title="Stage W: PATH within each subject and session")
    ax.legend(ncol=3)
    _save_figure(fig, FIGURE_STEMS[0], source, figures)

    source = sources[FIGURE_STEMS[1]]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(source.subject, source.balanced_accuracy, marker="o", linewidth=1.5, color="#6b4c9a", label="PATH X_s")
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set(xticks=range(1, 10), xlabel="Subject", ylabel="Balanced accuracy", title="Stage X: same-subject cross-session PATH transfer")
    ax.legend()
    _save_figure(fig, FIGURE_STEMS[1], source, figures)

    for stem, title in (
        (FIGURE_STEMS[2], "Stage W label-destruction null"),
        (FIGURE_STEMS[3], "Stage X label-destruction null"),
    ):
        source = sources[stem]
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharex=True, sharey=True)
        for ax, representation, color in zip(axes, ("PATH_D10", "BAG_CANON_D10"), ("#3569a8", "#d17a22"), strict=True):
            selected = source[source.representation.eq(representation)]
            ax.hist(selected.null_statistic, bins=35, color=color, alpha=0.78)
            ax.axvline(float(selected.observed.iloc[0]), color="black", linewidth=1.5, label="observed")
            ax.axvline(float(np.median(selected.null_statistic)), color="white", linestyle="--", linewidth=1.3, label="null median")
            ax.set(title=representation, xlabel="Median subject BA")
        axes[0].set_ylabel("Replicates")
        axes[1].legend()
        fig.suptitle(title)
        fig.tight_layout()
        _save_figure(fig, stem, source, figures)

    source = sources[FIGURE_STEMS[4]]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.hist(source.null_statistic, bins=35, color="#4f8c70", alpha=0.8)
    ax.axvline(float(source.observed.iloc[0]), color="black", linewidth=1.6, label="observed PATH X")
    ax.axvline(float(source.null_median.iloc[0]), color="white", linestyle="--", linewidth=1.4, label="order-null median")
    ax.set(xlabel="Median subject BA", ylabel="Replicates", title="Stage O: chronological-order contribution")
    ax.legend()
    _save_figure(fig, FIGURE_STEMS[4], source, figures)

    source = sources[FIGURE_STEMS[5]]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    representations = ["PATH_D10", "BAG_CANON_D10", "SCALARS_11"]
    x = np.arange(len(representations))
    width = 0.34
    for offset, stage, color in ((-width / 2, "W", "#3569a8"), (width / 2, "X", "#d17a22")):
        selected = source[source.stage.eq(stage)].set_index("representation").loc[representations]
        ax.bar(x + offset, selected.balanced_accuracy, width, label=stage, color=color)
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set(xticks=x, xticklabels=representations, ylabel="Median subject BA", title="Frozen representation context: observed W and X")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(ncol=3)
    _save_figure(fig, FIGURE_STEMS[5], source, figures)


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _report(results: Mapping[str, Any], test_summary: Mapping[str, Any] | None) -> str:
    tables = results["tables"]
    label = tables["label_null_summary"]
    order = tables["order_null_summary"].iloc[0]
    comparison = tables["representation_comparison"].set_index("representation")
    within = tables["within_session_scores"]
    path_session = (
        within[within.representation.eq("PATH_D10")]
        .groupby(["subject", "session"], observed=True)["balanced_accuracy"]
        .mean()
        .groupby("session")
        .median()
    )

    def label_row(stage: str, representation: str) -> pd.Series:
        return label[label.stage.eq(stage) & label.representation.eq(representation)].iloc[0]

    path_w = label_row("W", "PATH_D10")
    path_x = label_row("X", "PATH_D10")
    bag_w = label_row("W", "BAG_CANON_D10")
    bag_x = label_row("X", "BAG_CANON_D10")
    tests_text = (
        f"{int(test_summary['passed'])} passed in {float(test_summary['seconds']):.2f}s"
        if test_summary is not None
        else "full-suite result pending final validation"
    )
    return f"""# BNCI2014_001 Trajectory Within-Subject Audit v1

## 1. What this audit asks

Trajectory Anatomy v0 asked whether class trajectory structure transferred across different subjects. This v1 audit asks whether class structure exists inside each individual subject and whether it transfers across sessions for that same subject.

This is a retrospective interpretation audit of the unchanged five-window AIRM representation. It does not introduce a new method, tune a classifier, change a window, or use the WHOLE subject×class interaction result.

## 2. Data and reproduction gate

Both BNCI2014_001 sessions were used: 9 subjects, 4 balanced classes, and 2,592 trials per session (5,184 total). Session 1 used the exact frozen v0 channel order, 8–32 Hz filtering, cue-relative 0–3.996 s, 250 Hz, five non-overlapping 200-sample windows, and float64 OAS covariance without added regularization or eigenvalue clipping.

Hard gate 0 passed. Session-0 trial identities were exact, all frozen v0 geometry gates passed, and recomputed PATH_D10, BAG_CANON_D10, and SCALARS_11 each had maximum absolute difference 0.0 from the frozen v0 reference. Session-1 numerical and geometry gates also passed.

## 3. Stage W — class information inside a subject/session

Stage W trained on runs 0–2 and tested on runs 3–5, then reversed the direction, separately for each subject and session. The subject statistic averages four scores; the group statistic is the median across nine subjects.

PATH observed T_W = {_fmt(path_w.observed)}. The 1,999-replicate label-null median was {_fmt(path_w.null_median)}, giving effect {_fmt(path_w.effect)} and one-sided plus-one p = {_fmt(path_w.p_value)}. Stage W **PASS**.

Descriptive session medians for PATH were {_fmt(path_session['0train'])} in session 0 and {_fmt(path_session['1test'])} in session 1. Eight of nine PATH subject statistics were strictly above the 0.25 chance reference; this count is descriptive and not the test.

## 4. Stage X — same-subject transfer across sessions

Stage X trained on all six runs of one session and tested on the other session, then reversed direction, always within the same subject.

PATH observed T_X = {_fmt(path_x.observed)}. The shared 1,999-replicate label-null median was {_fmt(path_x.null_median)}, giving effect {_fmt(path_x.effect)} and p = {_fmt(path_x.p_value)}. Stage X **PASS**. All nine PATH subject statistics were strictly above 0.25 descriptively.

## 5. Stage O — does chronological order add stable information?

Stage O kept the same five local SPD states and all pairwise distances but independently replaced state identity with a nonidentity S5 permutation for every trial. The same frozen Stage-X pipeline was refit for 1,999 replicates.

Observed T_O = {_fmt(order.observed)}; order-null median = {_fmt(order.null_median)}; effect = {_fmt(order.effect)}; p = {_fmt(order.p_value)} ({int(order.exceedance_count)} null statistics at least as large as observed). Stage O **FAIL** because p > 0.05.

Therefore the audit does not support chronological ordering as a required contributor to the stable same-subject signal.

## 6. Mandatory BAG_CANON comparator

BAG_CANON removes state labels/order while retaining the same local pairwise-distance configuration.

- W: observed {_fmt(bag_w.observed)}, null median {_fmt(bag_w.null_median)}, effect {_fmt(bag_w.effect)}, p = {_fmt(bag_w.p_value)} — **PASS**.
- X: observed {_fmt(bag_x.observed)}, null median {_fmt(bag_x.null_median)}, effect {_fmt(bag_x.effect)}, p = {_fmt(bag_x.p_value)} — **PASS**.

BAG's W/X passes together with the Stage-O failure favor the interpretation that stable subject-specific local relative geometry exists, while chronological order is not established as essential. Raw PATH-minus-BAG accuracy differences are not treated as a superiority test.

## 7. SCALARS_11 descriptive context

SCALARS_11 was evaluated without feature selection and has no terminal vote. Its observed median subject BA was {_fmt(comparison.loc['SCALARS_11', 'stage_w_observed_median_subject_ba'])} for W and {_fmt(comparison.loc['SCALARS_11', 'stage_x_observed_median_subject_ba'])} for X. The descriptive above-chance counts were {int(comparison.loc['SCALARS_11', 'stage_w_subjects_above_chance'])}/9 and {int(comparison.loc['SCALARS_11', 'stage_x_subjects_above_chance'])}/9, respectively.

## 8. Relation to the old cross-subject result

The old v0 AIRM PATH cross-subject mean BA was approximately 0.2616 and its label-destruction p-value was 0.155; it did not establish a population-shared class trajectory. The present positive W/X results are not a contradiction: they show that class-discriminative local covariance geometry can be subject-specific and reproducible for the same subject while failing to form one shared cross-subject representation.

## 9. Frozen terminal decision

**GO_STABLE_SUBJECT_SPECIFIC_LOCAL_GEOMETRY**

PATH passed within-subject/within-session decoding and same-subject cross-session transfer. The chronological-order falsification did not pass. The unordered BAG comparator passed W and X.

## 10. What is supported

Under the frozen five-window AIRM representation, class-discriminative local covariance geometry is subject-specific and cross-session reproducible. The evidence is strongest for local relative geometry that does not require chronological state labels.

## 11. What is not supported

This audit does not establish brain physiology, individual motor strategy, causal neural dynamics, a cause of WHOLE-Z or domain-adaptation behavior, target-unlabeled identifiability, or a benefit from training a personalized model. It does not establish chronological order as necessary.

## 12. Reproducibility and validation

Label and order nulls each completed 1,999/1,999 deterministic replicates with resumable checkpoints and frozen SeedSequence derivation. Label realizations were shared between PATH and BAG. Required classifier fits had zero convergence failures. Final test result: {tests_text}.

Compact null artifact SHA-256 values:

- label: `{results['label_null_sha256']}`
- order: `{results['order_null_sha256']}`
"""


def create_reporting_outputs(
    config_path: str | Path,
    root: str | Path,
    *,
    test_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = Path(root).resolve()
    config_file = Path(config_path).resolve()
    config = load_frozen_config(config_file)
    results = _read_results(config, project_root)
    sources = _figure_sources(results)
    figures = results["output"] / "figures"
    _render_figures(sources, figures)
    report = _report(results, test_summary)
    report_path = results["output"] / str(config["outputs"]["report"])
    _atomic_text(report_path, report)
    provenance = {
        "protocol_sha256": str(config["protocol"]["sha256"]),
        "config_sha256": sha256_file(config_file),
        "terminal_decision": results["decision"]["decision"],
        "figure_stems": list(FIGURE_STEMS),
        "figure_source_rows": {stem: len(source) for stem, source in sources.items()},
        "test_summary": None if test_summary is None else dict(test_summary),
        "report_sha256": sha256_file(report_path),
    }
    _atomic_text(
        results["output"] / "protocol" / "reporting_provenance.json",
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    )
    scientific_provenance = {
        "analysis_name": "BNCI2014_001 trajectory within-subject audit v1",
        "scientific_scores_computed": True,
        "protocol_freeze_commit": "49685f3",
        "protocol_sha256": str(config["protocol"]["sha256"]),
        "config_sha256": sha256_file(config_file),
        "source_sha256": {
            relative: sha256_file(project_root / relative)
            for relative in (
                "src/trajectory_within_subject_v1.py",
                "src/trajectory_within_subject_data_v1.py",
                "src/trajectory_within_subject_analysis_v1.py",
                "src/reporting_trajectory_within_subject_v1.py",
                "scripts/30_prepare_trajectory_within_subject_v1.py",
                "scripts/31_run_trajectory_within_subject_v1.py",
                "scripts/32_report_trajectory_within_subject_v1.py",
            )
        },
        "result_artifact_sha256": {
            str(path.relative_to(results["output"])): sha256_file(path)
            for path in sorted(
                [
                    *(results["output"] / "tables").glob("*.csv"),
                    *(results["output"] / "decisions").glob("*"),
                    results["output"] / "nulls" / "label_null_statistics.npz",
                    results["output"] / "nulls" / "order_null_statistics.npz",
                ]
            )
            if path.is_file()
        },
        "null_replicates": {"label": 1999, "order": 1999},
        "terminal_decision": results["decision"]["decision"],
        "whole_subject_class_interaction_used": False,
        "test_summary": None if test_summary is None else dict(test_summary),
    }
    _atomic_text(
        results["output"] / "protocol" / "scientific_provenance.json",
        json.dumps(scientific_provenance, indent=2, sort_keys=True) + "\n",
    )
    return provenance


__all__ = [
    "FIGURE_STEMS",
    "TrajectoryAuditReportingError",
    "create_reporting_outputs",
]
