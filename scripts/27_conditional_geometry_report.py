#!/usr/bin/env python3
"""Validate frozen Conditional v1 artifacts and create the final report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting_conditional_v1 import (  # noqa: E402
    ReportingContractError,
    create_reporting_outputs,
    load_and_validate_reporting_inputs,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        inputs = load_and_validate_reporting_inputs(args.config, args.repo_root)
        artifacts = create_reporting_outputs(inputs)
    except ReportingContractError as error:
        raise SystemExit(f"Conditional v1 reporting prohibited: {error}") from error
    print(
        json.dumps(
            {
                "terminal_decision": artifacts.verdicts.terminal_decision,
                "le_robustness_label": artifacts.verdicts.le_robustness_label,
                "figures": [str(path) for path in artifacts.figure_paths],
                "figure_sources": [str(path) for path in artifacts.figure_source_paths],
                "report": str(artifacts.report_path),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
