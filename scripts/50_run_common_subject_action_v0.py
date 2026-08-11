#!/usr/bin/env python3
"""Run the frozen BNCI common-subject-action falsification audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common_action_pipeline_v0 import run_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent LOCO tasks to execute concurrently.",
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    terminal = run_all(ROOT, workers=arguments.workers)
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
