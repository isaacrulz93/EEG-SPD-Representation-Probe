#!/usr/bin/env python3
"""Execute or finalize the frozen trial-movement utility audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trial_movement_utility_v0 import execute_audit, record_test_results


DEFAULT_CONFIG = ROOT / "configs/bnci2014_001_trial_movement_utility_v0.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    execute.add_argument("--pre-result-commit", required=True)
    tests = subparsers.add_parser("record-tests")
    tests.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    tests.add_argument("--focused-before", type=Path, required=True)
    tests.add_argument("--full-tests", type=Path, required=True)
    tests.add_argument("--focused-after", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.command == "execute":
        result = execute_audit(ROOT, arguments.config, arguments.pre_result_commit)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        record_test_results(
            ROOT,
            arguments.config,
            focused_before=arguments.focused_before,
            full_tests=arguments.full_tests,
            focused_after=arguments.focused_after,
        )


if __name__ == "__main__":
    main()
