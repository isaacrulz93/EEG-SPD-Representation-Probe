#!/usr/bin/env python3
"""Render the six frozen figures and final trajectory audit report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting_trajectory_within_subject_v1 import create_reporting_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "bnci2014_001_trajectory_within_subject_v1.yaml",
    )
    parser.add_argument("--tests-passed", type=int)
    parser.add_argument("--test-seconds", type=float)
    args = parser.parse_args()
    test_summary = None
    if args.tests_passed is not None or args.test_seconds is not None:
        if args.tests_passed is None or args.test_seconds is None:
            parser.error("--tests-passed and --test-seconds must be provided together")
        test_summary = {"passed": args.tests_passed, "seconds": args.test_seconds}
    result = create_reporting_outputs(
        args.config, ROOT, test_summary=test_summary
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
