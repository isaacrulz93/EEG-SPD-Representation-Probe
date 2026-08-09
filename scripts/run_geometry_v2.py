#!/usr/bin/env python3
"""Run the frozen BNCI2014_001 geometry-audit V2 stages in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/bnci2014_001_geometry_v2.yaml"
STAGES = (
    "10_geometry_correctness_v2.py",
    "11_loso_alignment_v2.py",
    "12_v1_leakage_audit.py",
    "13_geometry_report_v2.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Frozen V2 YAML config",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = args.config.expanduser().resolve()
    for stage in STAGES:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / stage),
            "--config",
            str(config),
        ]
        print(f"[geometry-v2] running {stage}", flush=True)
        # check=True is intentional: stage 10's nonzero hard-gate result makes
        # it impossible for any classification or report stage to run.
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
