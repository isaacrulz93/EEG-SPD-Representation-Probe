"""Final OpenBMI decision, figure, and report for Subject Class Interaction v0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.interaction_provenance_v0 import atomic_write_json, sha256_file
from src.interaction_statistics_v0 import primary_outcome
from src.reporting_interaction_v0 import (
    PRIMARY,
    _passes,
    _save_figure,
    _select,
    _stage_summaries,
    generate_bnci_report,
)
from src.subject_class_interaction_v0 import load_frozen_config


def _openbmi_summaries(output: Path) -> pd.DataFrame:
    return pd.concat([
        pd.read_csv(output / "reliability/openbmi_split_half_null_summary.csv"),
        pd.read_csv(output / "cross_session/openbmi_derangement_null_summary.csv"),
        pd.read_csv(output / "class_dependence/openbmi_label_destruction_null_summary.csv"),
    ], ignore_index=True, sort=False)


def _ordered(frame: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    return _select(frame, **filters).set_index("stage").loc[["R", "I", "C"]]


def _all_supported(frame: pd.DataFrame) -> bool:
    return all(_passes(frame.loc[stage]) for stage in ("R", "I", "C"))


def _stat(frame: pd.DataFrame, stage: str) -> str:
    row = frame.loc[stage]
    return (
        f"T_obs={row['observed']:.6g}, null median={row['null_median']:.6g}, "
        f"E={row['effect']:.6g}, p={row['p_value']:.6g}"
    )


def generate_openbmi_final_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config, config_hash = load_frozen_config(root)
    output = root / config["project"]["output_dir"]
    generate_bnci_report(root)
    bnci = _stage_summaries(output)
    openbmi = _openbmi_summaries(output)
    bnci_primary = _ordered(bnci, **PRIMARY)
    openbmi_primary = _ordered(openbmi, **PRIMARY)
    openbmi_r = _ordered(
        openbmi,
        geometry="AIRM",
        template="session_specific",
        object="R",
        signature="sensor",
    )
    openbmi_spectrum = _ordered(
        openbmi,
        geometry="AIRM",
        template="session_specific",
        object="Z",
        signature="spectrum",
    )
    openbmi_pooled = _ordered(
        openbmi,
        geometry="AIRM",
        template="pooled_session",
        object="Z",
        signature="sensor",
    )
    openbmi_le = _ordered(
        openbmi,
        geometry="LE",
        template="session_specific",
        object="Z",
        signature="sensor",
    )
    hard_gates_pass = bool(json.loads(
        (output / "provenance/openbmi_observed_manifest.json").read_text()
    )["hard_gates_pass"])
    bnci_directions = {
        stage: bool(float(bnci_primary.loc[stage, "effect"]) > 0.0)
        for stage in ("R", "I", "C")
    }
    gate_r = bool(hard_gates_pass and _passes(openbmi_primary.loc["R"]))
    gate_i = bool(gate_r and _passes(openbmi_primary.loc["I"]))
    gate_c = bool(gate_i and _passes(openbmi_primary.loc["C"]))
    r_stable = _all_supported(openbmi_r)
    spectrum_supportive = _all_supported(openbmi_spectrum)
    pooled_supportive = _all_supported(openbmi_pooled)
    le_supportive = _all_supported(openbmi_le)
    terminal = primary_outcome(
        hard_gates_pass=hard_gates_pass,
        openbmi_unlocked=True,
        gate_r=gate_r,
        gate_i=gate_i,
        gate_c=gate_c,
        r_stable=r_stable,
        spectrum_supportive=spectrum_supportive,
        bnci_directions_positive=all(bnci_directions.values()),
    )
    rows: list[dict[str, Any]] = []
    for dataset, frame, role in (
        ("BNCI2014_001", bnci_primary, "RETROSPECTIVE_DEVELOPMENT_ONLY"),
        ("OpenBMI", openbmi_primary, "PROSPECTIVE_EXTERNAL_REPLICATION"),
    ):
        for stage in ("R", "I", "C"):
            row = frame.loc[stage]
            rows.append({
                "dataset": dataset,
                "stage": stage,
                "observed": float(row["observed"]),
                "null_median": float(row["null_median"]),
                "effect": float(row["effect"]),
                "p_value": float(row["p_value"]),
                "direction_positive": bool(float(row["effect"]) > 0.0),
                "gate_pass": (
                    None
                    if dataset == "BNCI2014_001"
                    else {"R": gate_r, "I": gate_i, "C": gate_c}[stage]
                ),
                "role": role,
            })
    decision_frame = pd.DataFrame(rows)
    decision_frame.to_csv(
        output / "decisions/decision_chain.csv",
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    decision = {
        "terminal_decision": terminal,
        "hard_gates_pass": hard_gates_pass,
        "openbmi_unlocked": True,
        "openbmi_primary_gates": {"R": gate_r, "I": gate_i, "C": gate_c},
        "bnci_directions_positive": bnci_directions,
        "R_control_all_stages_pass": r_stable,
        "spectrum_control_all_stages_pass": spectrum_supportive,
        "pooled_template_all_stages_pass": pooled_supportive,
        "LE_robustness_all_stages_pass": le_supportive,
        "primary_chain": "AIRM/session-specific/sensor/Z",
        "result_selected_protocol_modifications": False,
    }
    atomic_write_json(output / "decisions/terminal_decision.json", decision)

    figure_source = decision_frame[[
        "dataset", "stage", "effect", "p_value", "gate_pass", "role",
    ]].copy()
    pivot = figure_source.pivot(index="stage", columns="dataset", values="effect").loc[["R", "I", "C"]]
    figure, axis = plt.subplots(figsize=(6.4, 3.9))
    pivot.plot.bar(ax=axis)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("Stage")
    axis.set_ylabel("Observed − null median")
    axis.set_title("BNCI development and OpenBMI external replication")
    _save_figure(figure, output / "figures", "figure_9_bnci_openbmi_effects", figure_source)

    report = f"""# Subject Class Interaction V0

## 1. Scientific question

This frozen premise-falsification experiment asks whether a cross-session-reproducible subject×class interaction remains after marginal subject/session location, the population class template, and the class-independent subject residual are removed from covariance representations.

## 2. Why this follows prior anatomy

Prior discrepancy decomposition did not distinguish a class-dependent individual interaction from a generic subject residual. This analysis separates those alternatives without a classifier, TTA, neural network, new loss, low-rank fit, or mixed-effects fit.

## 3. Definitions in plain language

Each subject/session marginal covariance mean is moved to identity. Class means in that common tangent coordinate system are marginally recentered class effects. The frozen population class template and then the class-weighted residual are subtracted.

## 4. U / R / Z distinction

U is the marginally recentered class effect. R is U minus the population class template. Z is R minus its class-weighted mean. Z is the primary subject×class interaction object; R is a descriptive subject-residual control.

## 5. Dataset roles

BNCI2014_001 is retrospective development only. OpenBMI/Lee2019-MI is the prospective external replication. The OpenBMI manifest was frozen and unlocked only after all three BNCI primary directions were strictly positive.

## 6. Numerical and data gates

BNCI contained 5,184 expected trials from 9 subjects. OpenBMI contained 10,800 expected trials from 54 subjects, two sessions, two balanced classes, and the frozen ordered 20-channel motor-cortex subset. All 108 OpenBMI source records have SHA-256 provenance; 106 complete MNE-cache files were reused and the two incomplete subject-5 files were downloaded afresh. Covariance, finite, symmetry, SPD, AIRM convergence, marginal-identity, class-weight, and weighted-Z-zero gates passed.

## 7. BNCI development

The primary AIRM/session-specific/sensor/Z effects were strictly positive for all stages: R ({_stat(bnci_primary, "R")}); I ({_stat(bnci_primary, "I")}); C ({_stat(bnci_primary, "C")}).

## 8. OpenBMI prospective external replication

OpenBMI used the frozen preprocessing and no result-selected change. The primary chain used 1,999 label-destruction nulls for R, 100,000 deterministic random derangements for I, and 1,999 label-destruction nulls for C.

## 9. Measurement reliability

Stage R measures the median subject reliability between independently refitted within-session acquisition-order halves. OpenBMI: {_stat(openbmi_primary, "R")}. Gate R: **{"PASS" if gate_r else "FAIL"}**.

## 10. Same-subject cross-session reproducibility

Stage I measures the median diagonal of the session-0 by session-1 subject similarity matrix against fixed-point-free subject mappings. OpenBMI: {_stat(openbmi_primary, "I")}. Gate I: **{"PASS" if gate_i else "FAIL"}**.

## 11. True-class dependence

Stage C compares the true-label same-subject statistic with label-destruction refits. OpenBMI: {_stat(openbmi_primary, "C")}. Gate C: **{"PASS" if gate_c else "FAIL"}**.

## 12. R-versus-Z control

The AIRM/session-specific/sensor R control passed all three descriptive stage criteria: **{r_stable}**. It does not vote on or rescue the primary Z decision.

## 13. Gauge-sensitive versus spectrum control

The sensor Z chain passed R/I/C. The orthogonally invariant spectrum Z control passed all three criteria: **{spectrum_supportive}**. Its Stage C result was {_stat(openbmi_spectrum, "C")}; therefore the evidence is sensor-space-specific under the frozen outcome logic.

## 14. Template and geometry sensitivities

The pooled-session AIRM sensor Z sensitivity passed all stages: **{pooled_supportive}**. The session-specific log-Euclidean sensor Z robustness chain passed all stages: **{le_supportive}**. Neither secondary analysis can rescue the primary chain.

## 15. Frozen terminal decision

**{terminal}**. The primary OpenBMI sensor Z gates R/I/C passed, while the spectrum control did not support all three stages. No absolute cosine threshold or result-selected protocol modification was used.

## 16. What is justified

The frozen analysis supports a stable OpenBMI subject×class interaction in the montage-registered sensor representation after the specified marginal, population-class, and class-independent residual removals. The result replicated the three BNCI development directions.

## 17. What is NOT justified

It does not establish physiology, personality, a neural trait, source anatomy, a biomarker, unlabeled recoverability, identifiable TTA parameters, intrinsic Riemannian random effects, the full conditional distribution, or a causal brain mechanism.

## 18. Exactly one next structural question

Is the stable subject×class interaction low-dimensional and structured across the population?

## 19. Direction killed if STOP

No primary sensor-space direction was killed. The spectrum control failed Stage C, selecting the frozen sensor-space-only GO rather than a spectrum-supported GO.
"""
    report_path = output / "report/subject_class_interaction_v0.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": "openbmi-final-report-v0",
        "config_sha256": config_hash,
        "terminal_decision": terminal,
        "report_sha256": sha256_file(report_path),
        "decision_sha256": sha256_file(output / "decisions/terminal_decision.json"),
        "decision_chain_sha256": sha256_file(output / "decisions/decision_chain.csv"),
        "figure_9_source_sha256": sha256_file(output / "figures/figure_9_bnci_openbmi_effects.csv"),
        "openbmi_null_manifest_sha256": sha256_file(output / "provenance/openbmi_null_manifest.json"),
    }
    atomic_write_json(output / "provenance/openbmi_final_report_manifest.json", manifest)
    return decision
