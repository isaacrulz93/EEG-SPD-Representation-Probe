#!/usr/bin/env python3
"""Generate the terminal-aware V1.1 report, figures, note, and manifest."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subject_class_population_structure_v1_1 import run_final_report

if __name__ == "__main__":
    print(json.dumps(run_final_report(ROOT), indent=2, sort_keys=True))
