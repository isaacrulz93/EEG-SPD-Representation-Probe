"""Deterministic tables, figures, decisions, and report for the v0 pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.interaction_pipeline_v0 import _array_key, _load_observed_arrays
from src.interaction_provenance_v0 import atomic_write_json
from src.interaction_statistics_v0 import all_derangements, derangement_statistics, similarity_matrix
from src.subject_class_interaction_v0 import load_frozen_config


PRIMARY = {"geometry": "AIRM", "template": "session_specific", "object": "Z", "signature": "sensor"}


def _select(frame: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    return selected.copy()


def _save_figure(figure: plt.Figure, output: Path, stem: str, source: pd.DataFrame) -> None:
    source.to_csv(output / f"{stem}.csv", index=False, lineterminator="\n", float_format="%.17g")
    figure.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _stage_summaries(output: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(output / "reliability/split_half_null_summary.csv"),
        pd.read_csv(output / "cross_session/derangement_null_summary.csv"),
        pd.read_csv(output / "class_dependence/label_destruction_null_summary.csv"),
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def _passes(row: pd.Series) -> bool:
    return bool(float(row["effect"]) > 0.0 and float(row["p_value"]) <= 0.05)


def _plot_all(output: Path, summaries: pd.DataFrame, arrays: dict[str, np.ndarray]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    interactions = pd.read_csv(output / "tables/interaction_summary.csv")
    primary_objects = _select(interactions, geometry="AIRM", template="session_specific", split="F")
    construction = primary_objects.groupby("class", sort=False)[["R_frobenius_norm", "Z_frobenius_norm"]].mean().reset_index()
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(construction)); width = 0.36
    axis.bar(x - width / 2, construction["R_frobenius_norm"], width, label="R")
    axis.bar(x + width / 2, construction["Z_frobenius_norm"], width, label="Z")
    axis.set_xticks(x, construction["class"], rotation=20); axis.set_ylabel("Mean Frobenius norm"); axis.legend(); axis.set_title("Interaction construction: R versus Z")
    _save_figure(figure, figures, "figure_1_interaction_construction", construction)

    reliability = _select(pd.read_csv(output / "reliability/split_half_subject_scores.csv"), **PRIMARY)
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(reliability["subject"], reliability["session0_half_cosine"], "o-", label="0train")
    axis.plot(reliability["subject"], reliability["session1_half_cosine"], "s-", label="1test")
    axis.axhline(0, color="black", linewidth=0.7); axis.set_xlabel("Subject"); axis.set_ylabel("cos(bA,bB)"); axis.legend(); axis.set_title("Within-session measurement reliability")
    _save_figure(figure, figures, "figure_2_split_half_reliability", reliability)

    full = arrays[_array_key("AIRM", "session_specific", "F", "sensor_Z")]
    similarity = similarity_matrix(full[:, 0], full[:, 1])
    matrix_source = pd.DataFrame(similarity, columns=[f"session1_S{i}" for i in range(1, 10)])
    matrix_source.insert(0, "session0_subject", [f"S{i}" for i in range(1, 10)])
    figure, axis = plt.subplots(figsize=(5.2, 4.5)); image = axis.imshow(similarity, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xlabel("Session 1 subject"); axis.set_ylabel("Session 0 subject"); axis.set_xticks(range(9), range(1, 10)); axis.set_yticks(range(9), range(1, 10)); figure.colorbar(image, ax=axis, label="cosine"); axis.set_title("Cross-session signature similarity")
    _save_figure(figure, figures, "figure_3_cross_session_similarity_matrix", matrix_source)

    unrelated = derangement_statistics(similarity, all_derangements(9))
    same = np.diag(similarity)
    same_unrelated = pd.concat([
        pd.DataFrame({"kind": "same_subject", "index": np.arange(len(same)), "statistic": same}),
        pd.DataFrame({"kind": "unrelated_derangement_median", "index": np.arange(len(unrelated)), "statistic": unrelated}),
    ], ignore_index=True)
    figure, axis = plt.subplots(figsize=(6.2, 3.8)); axis.hist(unrelated, bins=45, alpha=0.75, label="unrelated derangements"); axis.axvline(np.median(same), color="crimson", label="median same subject"); axis.set_xlabel("Median cosine"); axis.set_ylabel("Count"); axis.legend(); axis.set_title("Same subject versus unrelated subjects")
    _save_figure(figure, figures, "figure_4_same_vs_unrelated", same_unrelated)

    null_groups = pd.read_csv(output / "tables/null_group_statistics.csv")
    class_null = _select(null_groups, geometry="AIRM", template="session_specific", object="Z", signature="sensor", stage="C")
    observed_c = float(_select(summaries, **PRIMARY, stage="C")["observed"].iloc[0])
    figure, axis = plt.subplots(figsize=(6.2, 3.8)); axis.hist(class_null["group_statistic"], bins=45, alpha=0.8); axis.axvline(observed_c, color="crimson", label="true labels"); axis.set_xlabel("Same-subject median cosine"); axis.set_ylabel("Count"); axis.legend(); axis.set_title("Class-destruction null")
    _save_figure(figure, figures, "figure_5_class_destruction_null", class_null)

    rz = _select(summaries, geometry="AIRM", template="session_specific", signature="sensor")
    rz = rz[rz["object"].isin(["R", "Z"])].copy()
    figure, axis = plt.subplots(figsize=(6.2, 3.8)); pivot = rz.pivot(index="stage", columns="object", values="effect").reindex(["R", "I", "C"]); pivot.plot.bar(ax=axis); axis.axhline(0, color="black", linewidth=0.7); axis.set_ylabel("Observed − null median"); axis.set_title("R versus Z control")
    _save_figure(figure, figures, "figure_6_R_vs_Z_control", rz)

    gauge = _select(summaries, geometry="AIRM", template="session_specific", object="Z")
    figure, axis = plt.subplots(figsize=(6.2, 3.8)); pivot = gauge.pivot(index="stage", columns="signature", values="effect").reindex(["R", "I", "C"]); pivot.plot.bar(ax=axis); axis.axhline(0, color="black", linewidth=0.7); axis.set_ylabel("Observed − null median"); axis.set_title("Sensor and spectral controls")
    _save_figure(figure, figures, "figure_7_sensor_vs_spectrum", gauge)

    energy = _select(pd.read_csv(output / "tables/descriptive_energy_fractions.csv"), geometry="AIRM", template="session_specific")
    figure, axis = plt.subplots(figsize=(6.2, 3.8)); energy.set_index("session")[["class_independent_Rbar_squared_norm", "Z_interaction_squared_norm"]].plot.bar(ax=axis); axis.set_ylabel("Squared Frobenius energy"); axis.set_title("Descriptive energy anatomy")
    _save_figure(figure, figures, "figure_8_energy_anatomy", energy)

    effects = _select(summaries, **PRIMARY)[["dataset", "stage", "effect", "p_value"]].copy(); effects["replication_status"] = "BNCI retrospective development"; open_rows = pd.DataFrame({"dataset": ["OpenBMI"] * 3, "stage": ["R", "I", "C"], "effect": [np.nan] * 3, "p_value": [np.nan] * 3, "replication_status": ["LOCKED"] * 3}); source = pd.concat([effects, open_rows], ignore_index=True)
    figure, axis = plt.subplots(figsize=(6.2, 3.8)); axis.bar(effects["stage"], effects["effect"]); axis.axhline(0, color="black", linewidth=0.7); axis.set_ylabel("Observed − null median"); axis.set_title("BNCI directions; OpenBMI remains locked")
    _save_figure(figure, figures, "figure_9_bnci_openbmi_effects", source)


def generate_bnci_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(); config, _ = load_frozen_config(root); output = root / config["project"]["output_dir"]
    summaries = _stage_summaries(output); arrays = _load_observed_arrays(output)
    primary = _select(summaries, **PRIMARY).set_index("stage").loc[["R", "I", "C"]]
    directions = {stage: float(primary.loc[stage, "effect"]) > 0.0 for stage in ("R", "I", "C")}
    all_positive = all(directions.values())
    r_control = _select(summaries, geometry="AIRM", template="session_specific", object="R", signature="sensor").set_index("stage").loc[["R", "I", "C"]]
    spectrum = _select(summaries, geometry="AIRM", template="session_specific", object="Z", signature="spectrum").set_index("stage").loc[["R", "I", "C"]]
    le = _select(summaries, geometry="LE", template="session_specific", object="Z", signature="sensor").set_index("stage").loc[["R", "I", "C"]]
    r_stable = all(_passes(r_control.loc[stage]) for stage in ("R", "I", "C"))
    spectrum_support = all(_passes(spectrum.loc[stage]) for stage in ("R", "I", "C"))
    le_support = all(_passes(le.loc[stage]) for stage in ("R", "I", "C"))
    terminal = "OPENBMI_UNLOCK_ELIGIBLE_NOT_TERMINAL" if all_positive else "STOP_BNCI_DIRECTION_FAILURE"
    decision_rows = []
    for stage in ("R", "I", "C"):
        row = primary.loc[stage]
        decision_rows.append({"dataset": "BNCI2014_001", "stage": stage, "observed": row["observed"], "null_median": row["null_median"], "effect": row["effect"], "p_value": row["p_value"], "direction_positive": directions[stage], "role": "RETROSPECTIVE_DEVELOPMENT_ONLY"})
    pd.DataFrame(decision_rows).to_csv(output / "decisions/decision_chain.csv", index=False, lineterminator="\n", float_format="%.17g")
    gauge = pd.concat([
        _select(summaries, geometry="AIRM", template="session_specific", object="Z", signature="sensor").assign(control="primary_sensor"),
        _select(summaries, geometry="AIRM", template="session_specific", object="Z", signature="spectrum").assign(control="spectral_secondary"),
    ], ignore_index=True)
    gauge.to_csv(output / "gauge/sensor_vs_spectrum_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    decision = {"terminal_decision": terminal, "bnci_directions_positive": directions, "openbmi_unlocked": False, "R_control_all_stages_pass": r_stable, "spectrum_control_all_stages_pass": spectrum_support, "LE_robustness_all_stages_pass": le_support, "primary_chain": "AIRM/session-specific/sensor/Z", "result_selected_protocol_modifications": False}
    atomic_write_json(output / "decisions/terminal_decision.json", decision)
    _plot_all(output, summaries, arrays)

    def stat(stage: str) -> str:
        row = primary.loc[stage]
        return f"T_obs={row['observed']:.6g}, null median={row['null_median']:.6g}, E={row['effect']:.6g}, p={row['p_value']:.6g}"
    killed = next((stage for stage in ("R", "I", "C") if not directions[stage]), "none in BNCI direction screen")
    report = f"""# Subject Class Interaction V0

## 1. Scientific question

This premise-falsification experiment asks whether a cross-session-reproducible subject×class interaction in marginally recentered covariance representation remains after marginal subject/session location, the session-specific population class template, and a class-independent subject residual are removed.

## 2. Why this follows prior anatomy

Prior discrepancy decomposition left open whether stable residual structure was class-dependent or merely a global subject residual. This pilot separates those alternatives without a classifier, TTA, neural network, new loss, low-rank factorization, or mixed-effects fit.

## 3. Definitions in plain language

For each subject/session, the marginal covariance mean is moved to identity. Class means in that common tangent coordinate system are called marginally recentered class effects. A target-excluding population class template is then subtracted, followed by the weighted class-independent residual.

## 4. U / R / Z distinction

U is the marginally recentered class effect. R is U minus the session-specific LOSO population class template. Z is R minus its class-weighted mean. Z, not U or R, is the primary subject×class interaction object.

## 5. Dataset roles

BNCI2014_001 is retrospective development only. OpenBMI/Lee2019-MI is the prospective external replication and remains scientifically locked unless all three BNCI primary effect directions are positive.

## 6. Numerical/data gates

All 5,184 expected BNCI trials, 9 subjects, two sessions, four classes, six runs, 22 ordered channels, and frozen file/content hashes matched. Covariance SPD/finite/symmetry checks, AIRM convergence with Karcher residual ≤1e-7, marginal-to-identity checks, U/Z symmetry, class weights, and weighted-Z-zero checks passed.

## 7. BNCI development

The primary chain is AIRM, session-specific LOSO population templates, the montage-registered sensor signature, and Z. BNCI is descriptive/developmental rather than strict confirmation.

## 8. OpenBMI external replication if unlocked

OpenBMI status: **{'eligible for manifest resolution but not yet unlocked' if all_positive else 'not unlocked because the BNCI direction screen failed'}**. No OpenBMI scientific score, similarity matrix, or figure was produced.

## 9. Measurement reliability

Stage R first computes each subject's sessionwise cosine between independently refitted run-halves, averages the two session cosines per subject, and takes the subject median. Its null permutes labels within subject/session 1,999 times, preserves counts, and refits every label-dependent mean and interaction. Primary result: {stat('R')}.

## 10. Same-subject cross-session reproducibility

Stage I takes the median diagonal of the session-0 by session-1 subject similarity matrix. Its unrelated-subject null exhaustively evaluates all 133,496 fixed-point-free mappings. Primary result: {stat('I')}.

## 11. Class-destruction control

Stage C compares the true-label same-subject statistic with 1,999 independent within-subject/session label destructions, each followed by the full refit. Primary result: {stat('C')}.

## 12. R-versus-Z control

The R signature retains class-independent residual structure and has no primary vote. Its three frozen descriptive stage criteria were {'all supported' if r_stable else 'not all supported'}. A stable R cannot rescue a failed Z chain.

## 13. Gauge-sensitive versus spectrum control

The sensor signature is montage-registered and coordinate-dependent. The ascending-eigenvalue spectrum is invariant to orthogonal conjugation and is secondary only. Its three descriptive criteria were {'all supported' if spectrum_support else 'not all supported'}.

## 14. Descriptive energy anatomy

The saved DESCRIPTIVE ENERGY FRACTIONS report squared population-class, class-independent Rbar, Z, split-half discrepancy, and cross-session discrepancy energies. Only the algebraically orthogonal Rbar-versus-Z residual fractions are normalized; they are not identified variance components.

## 15. Frozen terminal decision

**{terminal}**. The decision uses no absolute cosine threshold and no result-selected protocol modification.

## 16. What is justified

The BNCI result justifies only the frozen retrospective direction screen for a cross-session-reproducible subject×class interaction in marginally recentered covariance representation.

## 17. What is NOT justified

It does not establish physiology, personality, a neural trait, source anatomy, a biomarker, unlabeled recoverability, identifiable TTA parameters, low dimensionality, intrinsic Riemannian random effects, full conditional distributions, or a causal brain mechanism.

## 18. Exactly one next structural question if GO

{'Deferred until prospective OpenBMI replication reaches a frozen GO.' if terminal != 'GO_STABLE_SUBJECT_CLASS_INTERACTION' else 'Is the stable subject×class interaction low-dimensional and structured across the population?'}

## 19. What direction is killed if STOP

The first nonpositive BNCI primary effect direction is **{killed}**. {'Accordingly the prospective replication boundary forbids OpenBMI scientific analysis.' if not all_positive else 'No BNCI direction was killed; the next allowed action is metadata-only OpenBMI protocol resolution and manifest freeze.'}
"""
    (output / "report/subject_class_interaction_v0.md").write_text(report, encoding="utf-8")
    return decision
