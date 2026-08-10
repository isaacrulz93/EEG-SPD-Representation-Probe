#!/usr/bin/env python3
"""Prepare and hard-gate the two-session trajectory audit data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trajectory_within_subject_data_v1 import prepare_audit_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "bnci2014_001_trajectory_within_subject_v1.yaml",
    )
    args = parser.parse_args()
    result = prepare_audit_data(
        args.config,
        ROOT,
        progress=lambda message: print(message, flush=True),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "trials": len(result.metadata),
                "reproduction_checks": len(result.reproduction_gate),
                "combined_cache": str(result.combined_cache_path),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
