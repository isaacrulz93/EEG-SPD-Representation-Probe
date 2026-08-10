#!/usr/bin/env python3
"""Run observed and resumable null stages for trajectory audit v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trajectory_within_subject_analysis_v1 import (
    finalize_decision_and_comparison,
    run_label_nulls,
    run_observed,
    run_order_nulls,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "bnci2014_001_trajectory_within_subject_v1.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("observed", "label", "order", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    announce = lambda message: print(message, flush=True)
    if args.stage in {"observed", "all"}:
        run_observed(args.config, ROOT, progress=announce)
    if args.stage in {"label", "all"}:
        run_label_nulls(
            args.config,
            ROOT,
            workers=args.workers,
            batch_size=args.batch_size,
            progress=announce,
        )
    if args.stage in {"order", "all"}:
        run_order_nulls(
            args.config,
            ROOT,
            workers=args.workers,
            batch_size=args.batch_size,
            progress=announce,
        )
    if args.stage in {"finalize", "all"}:
        decision, _ = finalize_decision_and_comparison(args.config, ROOT)
        print(json.dumps({"decision": decision["decision"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
