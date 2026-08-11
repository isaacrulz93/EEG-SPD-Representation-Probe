#!/usr/bin/env python3
"""Run the frozen pairwise common-action V2 amendment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

from src.common_action_solver_v0 import ActionSolverError
from src.pairwise_common_action_pipeline_v2 import (
    record_technical_failure,
    run_all,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    started = time.perf_counter()
    try:
        result = run_all(ROOT, workers=arguments.workers)
    except ActionSolverError as error:
        result = record_technical_failure(
            ROOT, error, runtime_seconds=time.perf_counter() - started
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
