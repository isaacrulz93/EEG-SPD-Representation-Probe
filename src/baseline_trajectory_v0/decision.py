"""Frozen gates, terminal case selection, and report generation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import append_status, atomic_csv, atomic_json, atomic_text


def decide(output: str | Path) -> dict:
    started = time.time(); output = Path(output)
    tests = pd.read_csv(output / "task/paired_tests.csv")
    summaries = pd.read_csv(output / "task/subject_summary.csv")
    task_gates = {"no_ra": {}, "ra": {}}
    for mode, protocol, threshold in (("no_ra", "P2", .05), ("ra", "P3", .02)):
        for candidate in ("F2-S", "F3-G"):
            row = tests[(tests.protocol == protocol) & (tests.candidate == candidate) & (tests.reference == "F0")].iloc[0]
            cross = tests[(tests.protocol == "P1") & (tests.candidate == candidate) & (tests.reference == "F0")].iloc[0]
            passed = bool(row.mean_delta >= threshold and row.wins >= 6 and row.p_holm <= .05
                          and cross.mean_delta >= -.01 and cross.wins >= 5)
            task_gates[mode][candidate] = {"pass": passed, "mean_delta": row.mean_delta,
                                           "wins": int(row.wins), "p_holm": row.p_holm,
                                           "cross_session_delta": cross.mean_delta,
                                           "cross_session_wins": int(cross.wins)}
    order_row = tests[(tests.protocol == "P2") & (tests.reference == "F2-S-SHUFFLE")].iloc[0]
    order = {"pass": bool(order_row.mean_delta >= .01 and order_row.wins >= 6 and order_row.p_raw <= .05),
             "mean_delta": order_row.mean_delta, "wins": int(order_row.wins), "p_raw": order_row.p_raw}
    ident = pd.read_csv(output / "identifiability/subject_summary.csv")
    pivot_nmi = ident.pivot(index="subject", columns="feature", values="nmi")
    pivot_ba = ident.pivot(index="subject", columns="feature", values="mapping_balanced_accuracy")
    ident_candidates = {}
    for candidate in ("F2-S", "F3-G"):
        dnmi = pivot_nmi[candidate] - pivot_nmi.F0; dba = pivot_ba[candidate] - pivot_ba.F0
        ident_candidates[candidate] = {"pass": bool(np.median(dnmi) >= .05 and np.median(dba) >= .05 and (dba > 0).sum() >= 6),
                                       "median_nmi_delta": float(np.median(dnmi)),
                                       "median_mapping_ba_delta": float(np.median(dba)),
                                       "wins": int((dba > 0).sum())}
    ident_pass = any(v["pass"] for v in ident_candidates.values())
    oracle = pd.read_csv(output / "action_bridge/oracle_components.csv")
    oracle_mean = oracle.groupby("representation").semantic_accuracy.mean()
    rank1 = int(((oracle.representation == "F2-S") & (oracle.true_permutation_rank == 1)).sum())
    action_oracle = {"pass": bool(oracle_mean["F2-S"] - oracle_mean["STATIC"] >= .10 and rank1 >= 27),
                     "semantic_accuracy_delta": float(oracle_mean["F2-S"] - oracle_mean["STATIC"]),
                     "f2_rank1_cells": rank1}
    zero = pd.read_csv(output / "action_bridge/zero_label_clusters.csv")
    zero_subject = zero.groupby(["subject", "representation"]).mapping_balanced_accuracy.median().unstack()
    dzero = zero_subject["F2-S"] - zero_subject.STATIC
    action_zero = {"pass": bool(np.median(dzero) >= .05 and (dzero > 0).sum() >= 6),
                   "median_mapping_ba_delta": float(np.median(dzero)), "wins": int((dzero > 0).sum())}
    task_pass = any(v["pass"] for group in task_gates.values() for v in group.values())
    if task_pass and order["pass"]:
        case, terminal = "CASE 2", "GO_ORDERED_BASELINE_RELATIVE_TRAJECTORY"
    elif not task_pass and (ident_pass or action_zero["pass"]):
        case, terminal = "CASE 3", "GO_IDENTIFIABILITY_CANONICALIZATION_ONLY"
    elif (not task_pass and not ident_pass and not action_zero["pass"] and action_oracle["pass"]):
        case, terminal = "CASE 4", "STOP_ZERO_LABEL; ORACLE_TRAJECTORY_STRUCTURE_ONLY"
    elif task_pass or ident_pass or action_zero["pass"]:
        case, terminal = "CASE 1", "GO_BASELINE_RELATIVE_LOCAL_CONFIGURATION"
    else:
        case, terminal = "CASE 0", "STOP_BASELINE_RELATIVE_TRAJECTORY_LINE"
    best_no_ra = max(v["mean_delta"] for v in task_gates["no_ra"].values())
    best_wins = max(v["wins"] for v in task_gates["no_ra"].values())
    cross_ok = any(v["cross_session_delta"] >= -.01 for v in task_gates["no_ra"].values())
    gru_run = bool(case == "CASE 2" or
                   (case == "CASE 1" and best_no_ra >= .03 and best_wins >= 6) or
                   (best_no_ra >= .02 and order["pass"] and cross_ok))
    gates = {"data": {"pass": True}, "task": task_gates, "order": order,
             "ident_cluster": {"pass": ident_pass, "candidates": ident_candidates},
             "action_oracle": action_oracle, "action_zero": action_zero,
             "phase_p_bridge": {"status": "NOT_RUN", "reason": "NOT_RUN_EXACT_TRIAL_IDENTITY_UNAVAILABLE"},
             "conditional_gru": {"run": gru_run}, "final_case": case, "terminal": terminal}
    atomic_json(output / "decisions/gates.json", gates)
    atomic_json(output / "decisions/terminal.json", {"case": case, "terminal": terminal})
    atomic_csv(output / "phase_p_bridge/predictability.csv", pd.DataFrame([{"status": "NOT_RUN_EXACT_TRIAL_IDENTITY_UNAVAILABLE"}]))
    append_status(output, "Phase 0 decision", "COMPLETE", time.time() - started,
                  "decisions/*.json", terminal, "Conditional Phase 1" if gru_run else "Final report")
    return gates


def report(output: str | Path, model_status: str = "NOT_RUN") -> None:
    output = Path(output); gates = json.loads((output / "decisions/gates.json").read_text())
    status = lambda value: "PASS" if value else "FAIL"
    no_ra = any(v["pass"] for v in gates["task"]["no_ra"].values())
    ra = any(v["pass"] for v in gates["task"]["ra"].values())
    lines = [
        f"1. DATA gate: {status(gates['data']['pass'])}",
        f"2. Task no-RA gate: {status(no_ra)}",
        f"3. Task RA gate: {status(ra)}",
        f"4. ORDER gate: {status(gates['order']['pass'])}",
        f"5. IDENT-CLUSTER gate: {status(gates['ident_cluster']['pass'])}",
        f"6. ACTION-ORACLE gate: {status(gates['action_oracle']['pass'])}",
        f"7. ACTION-ZERO gate: {status(gates['action_zero']['pass'])}",
        "8. Phase-P bridge: NOT_RUN",
        f"9. Conditional GRU phase: {model_status}",
        f"10. FINAL CASE: {gates['final_case']} - {gates['terminal']}",
        "", "# Direct answers", "",
        f"- Pre-cue baseline-relative representation beat F0: {status(no_ra or ra)}.",
        f"- Ordered sequence contribution: {status(gates['order']['pass'])}; otherwise interpret as local configuration only.",
        f"- Exact F3 versus partial F2: see task/subject_summary.csv; neither is privileged beyond frozen gates.",
        f"- Target class identifiability improved: {status(gates['ident_cluster']['pass'])}.",
        f"- Zero-label source semantic naming improved: {status(gates['action_zero']['pass'])}.",
        f"- Known-to-zero-label action gap reduced: {status(gates['action_zero']['pass'])}.",
        "- Subject 2: reported explicitly in task and identifiability subject summaries.",
        "- Phase P obstruction: unassessed because exact trial identity was unavailable; no approximate match was attempted.",
        f"- Next step: {gates['terminal']}.", "",
        "# Interpretation boundary", "",
        "Claims are limited to the frozen BNCI2014_001 setting and descriptors. No physiological-source, causal-order, unique-mixing, full-quotient, or full target-conditional recovery claim is made.",
    ]
    atomic_text(output / "REPORT.md", "\n".join(lines) + "\n")
