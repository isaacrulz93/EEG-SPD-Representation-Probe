#!/usr/bin/env python3
"""Run the PASS-gated Trajectory Anatomy v0 scientific discovery stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.discovery_pipeline_trajectory_v0 import (  # noqa: E402
    DiscoveryStructuralError,
    build_discovery_artifacts,
    load_discovery_inputs,
    write_discovery_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen trajectory geometry PASS gate and run exact "
            "tables 11--20 plus the two preregistered null families."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: script parent repository)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "bnci2014_001_trajectory_v0.yaml",
        help="Frozen trajectory-v0 YAML config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    try:
        config, inputs = load_discovery_inputs(config_path, root)

        def progress(message: str) -> None:
            print(f"trajectory discovery: {message}", file=sys.stderr, flush=True)

        artifacts = build_discovery_artifacts(
            inputs,
            config,
            progress=progress,
        )
        result = write_discovery_artifacts(artifacts, config, root)
    except Exception as error:
        category = (
            "STRUCTURAL_FAILURE"
            if isinstance(error, DiscoveryStructuralError)
            else "PIPELINE_FAILURE"
        )
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "failure_category": category,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if artifacts.status != "PASS":
        print(
            "Scientific grid contains recorded FAILED rows; artifacts were "
            "preserved for an UNASSESSED report without available-case inference.",
            file=sys.stderr,
        )
    # Recorded convergence failures are scientific rows, not a structural
    # process failure.  Stage 23 must consume them and emit UNASSESSED.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
