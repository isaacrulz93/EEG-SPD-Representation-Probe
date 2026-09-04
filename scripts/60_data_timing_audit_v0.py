#!/usr/bin/env python3
"""Run the raw BNCI2014_001 cue/baseline timing and covariance audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_trajectory_v0.data_audit import run_data_timing_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("/Volumes/External_SSD/isaac/EEG-SPD-Representation-Probe/cache/moabb_data"),
    )
    parser.add_argument(
        "--lineage-cache",
        type=Path,
        default=Path("/Volumes/External_SSD/isaac/EEG-SPD-Representation-Probe/cache/bnci2014_001"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_data_timing_audit(
        ROOT,
        args.raw_data_dir,
        args.lineage_cache,
        resume=args.resume,
    )
    print(json.dumps({"data_gate": result["data_gate"], "cache": result["cache_path"]}, indent=2))


if __name__ == "__main__":
    main()
