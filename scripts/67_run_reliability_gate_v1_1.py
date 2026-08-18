#!/usr/bin/env python3
"""Re-run the unchanged V1 OpenBMI reliability prerequisite in V1.1."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subject_class_population_structure_v1_1 import run_reliability_gate

if __name__ == "__main__":
    print(json.dumps(run_reliability_gate(ROOT), indent=2, sort_keys=True))
