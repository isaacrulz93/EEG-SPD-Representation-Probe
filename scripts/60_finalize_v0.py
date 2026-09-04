#!/usr/bin/env python3
"""Build compact final artifacts without changing any frozen gate."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.baseline_trajectory_v0.io import atomic_csv, atomic_json, atomic_text

OUT = ROOT / "outputs/bnci2014_001_baseline_trajectory_identifiability_v0"


def main():
    gates = json.loads((OUT / "decisions/gates.json").read_text())
    task = pd.read_csv(OUT / "task/subject_summary.csv")
    ident = pd.read_csv(OUT / "identifiability/subject_summary.csv")
    fold = pd.read_csv(OUT / "task/fold_results.csv")
    confusion = fold[["protocol", "subject", "split", "feature", "confusion_matrix"]].copy()
    (OUT / "task/confusion_matrices").mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "task/confusion_matrices/all_folds.csv", confusion)
    for folder in ("model", "figures"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "model/seed_fold_results.csv", pd.DataFrame([{"status": "NOT_RUN_BY_FROZEN_GATE"}]))
    atomic_csv(OUT / "model/subject_summary.csv", pd.DataFrame([{"status": "NOT_RUN_BY_FROZEN_GATE"}]))
    atomic_csv(OUT / "model/checkpoints_manifest.csv", pd.DataFrame([{"status": "NOT_RUN_BY_FROZEN_GATE"}]))
    means = task[task.protocol == "P2"].groupby("feature").balanced_accuracy.mean()
    names = ["F0", "F1", "F2-S", "F2-V", "F3-G", "F3-D", "F2-S-SHUFFLE"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#22333b", "#5e503f", "#c44536", "#d4a373", "#3a7d44", "#8a817c", "#bcb8b1"]
    ax.bar(names, [means[x] for x in names], color=colors)
    ax.axhline(means["F0"], color="#22333b", linestyle="--", linewidth=1)
    ax.set_ylabel("LOSO no-RA balanced accuracy"); ax.set_title("Frozen Phase 0 task headroom")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(OUT / "figures/task_headroom.png", dpi=180); plt.close(fig)
    p2 = task[(task.protocol == "P2") & (task.subject == 2)].set_index("feature").balanced_accuracy
    p3 = task[(task.protocol == "P3") & (task.subject == 2)].set_index("feature").balanced_accuracy
    i2 = ident[ident.subject == 2].set_index("feature")
    report = [
        "1. DATA gate: PASS", "2. Task no-RA gate: FAIL", "3. Task RA gate: FAIL",
        "4. ORDER gate: PASS", "5. IDENT-CLUSTER gate: FAIL", "6. ACTION-ORACLE gate: FAIL",
        "7. ACTION-ZERO gate: FAIL", "8. Phase-P bridge: NOT_RUN",
        "9. Conditional GRU phase: NOT_RUN", "10. FINAL CASE: CASE 0 - STOP_BASELINE_RELATIVE_TRAJECTORY_LINE",
        "", "# Result", "",
        "Pre-cue F2-S showed non-gating LOSO no-RA headroom over F0 (+0.03318 BA; 8/9 wins; Holm p=0.03125), but missed the predeclared +0.050 effect threshold and failed cross-session consistency (-0.16917; 0/9 wins). It therefore did not beat F0 under the frozen task gate.",
        "", "Chronology was detectable relative to the independent per-trial shuffle (+0.04649 BA; 9/9 wins; exact p=0.001953), but ORDER alone cannot rescue a failed task gate. The terminal result is not an ordered-trajectory go decision.",
        "", "F3-G was worse than F2-S: LOSO no-RA mean BA was 0.26157 versus 0.38966, and RA mean BA was 0.26157 versus 0.38252. The exact invariant did not outperform the partial baseline-relative coordinate.",
        "", "Target cluster identifiability did not improve. F2-S versus F0 had median-subject NMI delta -0.06020 and mapping-BA delta +0.01649 (5/9 wins); F3-G had -0.09685 and -0.07813 (2/9 wins).",
        "", "Zero-label action did not close the correspondence gap. Oracle F2-S semantic accuracy changed by -0.01389 versus STATIC and ranked the true permutation first in 9/36 cells. Zero-label F2-S had median subject mapping-BA delta -0.00434 with 4/9 wins.",
        "", "# Hard subject 2", "",
        f"Subject 2 LOSO no-RA BA changed from F0 {p2['F0']:.5f} to F2-S {p2['F2-S']:.5f}; RA changed from {p3['F0']:.5f} to {p3['F2-S']:.5f}. NMI changed from {i2.loc['F0','nmi']:.5f} to {i2.loc['F2-S','nmi']:.5f}, and source-assignment BA from {i2.loc['F0','mapping_balanced_accuracy']:.5f} to {i2.loc['F2-S','mapping_balanced_accuracy']:.5f}. These small local changes do not alter any gate.",
        "", "# Phase P and next step", "",
        "The Phase-P bridge is NOT_RUN_EXACT_TRIAL_IDENTITY_UNAVAILABLE; no approximate matching was attempted. The prescribed next step is STOP, not a GRU, canonicalizer, transition alignment, or SPDHSW hierarchy.",
        "", "# Interpretation boundary", "",
        "This terminal is limited to the frozen BNCI2014_001 timing, filtering, covariance, descriptor, and zero-label bridge. It makes no physiological-source, causal-order, unique-mixing, full-quotient, or full target-conditional recovery claim.", "",
    ]
    atomic_text(OUT / "REPORT.md", "\n".join(report))
    handoff = """# HANDOFF

- Terminal: `CASE 0 - STOP_BASELINE_RELATIVE_TRAJECTORY_LINE`
- Data timing/covariance gate passed for all 5,184 trials.
- Phase 0 task, identifiability, oracle action, zero-label action, and null controls completed.
- Conditional GRU was not run by the frozen decision tree.
- Phase P was not run because exact T3DA trial identity was unavailable.
- Resume chunks remain local/ignored; aggregate CSV/JSON/MD artifacts are tracked.
"""
    atomic_text(OUT / "HANDOFF.md", handoff)
    failure = {"status": "RECOVERED", "scientific_terminal": gates["terminal"],
               "implementation_failures": ["tangent_mean_type_error", "trusted_chunk_dtype_resume_error", "pymanopt_miniter_api_error", "static_component_axis_error"],
               "outcome_definition_changed": False}
    atomic_json(OUT / "failure.json", failure)
    required = ["STATUS.md", "DATA_TIMING_AUDIT.md", "PRIOR_EVIDENCE_AUDIT.md", "LITERATURE_OVERLAP_AUDIT.md",
                "manifest.json", "environment.json", "git_provenance.json", "protocol/frozen_protocol.md", "protocol/frozen_config.yaml",
                "protocol/hashes.json", "features/feature_manifest.csv", "task/fold_results.csv", "task/subject_summary.csv",
                "task/paired_tests.csv", "identifiability/clustering_seed_results.csv", "identifiability/subject_summary.csv",
                "identifiability/neighbour_overlap.csv", "identifiability/source_assignment.csv", "action_bridge/oracle_components.csv",
                "action_bridge/zero_label_clusters.csv", "action_bridge/permutation_ranks.csv", "action_bridge/optimizer_audit.csv",
                "action_bridge/unrelated_target_null.csv", "phase_p_bridge/predictability.csv", "decisions/gates.json",
                "decisions/terminal.json", "model/seed_fold_results.csv", "REPORT.md", "HANDOFF.md"]
    audit = {name: (OUT / name).exists() for name in required}
    audit["complete"] = all(audit.values()); audit["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(OUT / "output_completeness.json", audit)
    if not audit["complete"]: raise RuntimeError("output completeness failure")

if __name__ == "__main__": main()
