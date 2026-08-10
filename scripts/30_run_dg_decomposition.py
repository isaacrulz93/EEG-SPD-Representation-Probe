#!/usr/bin/env python3
"""Run the frozen retrospective D/G discrepancy decomposition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dg_decomposition_v1 import final_numerical_summary, run_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/bnci2014_001_dg_decomposition_v1.yaml",
    )
    parser.add_argument("--tests", default="PASS")
    args = parser.parse_args()
    result = run_diagnostic(PROJECT_ROOT, args.config, tests=args.tests)
    manifest = json.loads((result.output_root / "manifest.json").read_text(encoding="utf-8"))
    print("BASE_COMMIT=" + manifest["base_commit"])
    print("BRANCH=" + manifest["branch"])
    print("FINAL_COMMIT=PENDING_OUTPUT_COMMIT")
    print("TESTS=" + result.tests)
    print("SOURCE_OUTPUTS_UNCHANGED=" + ("PASS" if result.source_unchanged else "FAIL"))
    print("HARD_GATES=" + result.hard_gates)
    print("SESSION0_LABEL=" + result.session_labels["0train"])
    print("SESSION1_LABEL=" + result.session_labels["1test"])
    print("OVERALL_PROVISIONAL_MECHANISM=" + result.overall_label)
    print(final_numerical_summary(result.output_root))


if __name__ == "__main__":
    main()
