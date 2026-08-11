#!/usr/bin/env python3
"""Run the exact production TrustRegions path on the synthetic V2 gate."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common_action_optimizer_audit_v2 import synthetic_stress_fixtures
from src.common_action_solver_v0 import conjugate, diagnose_multistart
from src.pairwise_common_action_v2 import fit_pairwise_action_v2


def main() -> None:
    started = time.perf_counter()
    rows = []
    for fixture in synthetic_stress_fixtures():
        fit = fit_pairwise_action_v2(
            fixture.fit_targets,
            fixture.fit_templates,
            seed_parts=("v2_synthetic_reproduction", fixture.name),
        )
        diagnostic = diagnose_multistart(fit.fit)
        best_prediction = conjugate(fit.fit.matrix, fixture.heldout_template)
        heldout_error = float(
            np.linalg.norm(best_prediction - fixture.heldout_truth)
            / np.linalg.norm(fixture.heldout_truth)
        )
        rows.append(
            {
                "fixture": fixture.name,
                "family": fixture.family,
                "truth_determinant": fixture.truth_determinant,
                "converged_starts": sum(value.converged for value in fit.fit.starts),
                "both_sectors_certified": (
                    diagnostic.determinant_sectors_with_converged_solution
                    == (-1, 1)
                ),
                "maximum_gradient_norm": max(
                    value.gradient_norm for value in fit.fit.starts
                ),
                "best_heldout_relative_error": heldout_error,
                "iterations": [value.iterations for value in fit.fit.starts],
                "initial_determinants": list(fit.initial_determinants),
                "optimizer_identities": sorted(
                    {value.optimizer for value in fit.fit.starts}
                ),
            }
        )
    exact = [row for row in rows if row["family"] == "generic_exact"]
    noisy = [row for row in rows if row["family"] != "generic_exact"]
    summary = {
        "status": "PASS",
        "scientific_data_accessed": False,
        "dimension": 22,
        "fixtures": len(rows),
        "starts": sum(int(row["converged_starts"]) for row in rows),
        "both_sectors_certified_fixtures": sum(
            bool(row["both_sectors_certified"]) for row in rows
        ),
        "maximum_exact_heldout_relative_error": max(
            float(row["best_heldout_relative_error"]) for row in exact
        ),
        "maximum_noisy_heldout_relative_error": max(
            float(row["best_heldout_relative_error"]) for row in noisy
        ),
        "runtime_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    if summary["fixtures"] != 12 or summary["starts"] != 48:
        raise RuntimeError("TrustRegions V2 did not reproduce 48/48 starts")
    if summary["both_sectors_certified_fixtures"] != 12:
        raise RuntimeError("TrustRegions V2 did not reproduce 12/12 sectors")
    if summary["maximum_exact_heldout_relative_error"] > 1.0e-8:
        raise RuntimeError("exact synthetic held-out recovery failed")
    if summary["maximum_noisy_heldout_relative_error"] > 5.0e-3:
        raise RuntimeError("noisy synthetic held-out recovery failed")
    output = ROOT / "outputs/common_action_optimizer_v2_preflight"
    output.mkdir(parents=True, exist_ok=True)
    (output / "synthetic_reproduction_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
