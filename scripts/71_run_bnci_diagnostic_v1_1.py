#!/usr/bin/env python3
"""Run the isolated secondary BNCI diagnostic after the OpenBMI terminal."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subject_class_population_structure_v1_1 import run_bnci_diagnostic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    print(json.dumps(run_bnci_diagnostic(ROOT, workers=arguments.workers), indent=2, sort_keys=True))
