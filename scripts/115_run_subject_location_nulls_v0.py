#!/usr/bin/env python3
"""Release outcomes for evaluation, then run registered nulls and controls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.subject_location_conditional_configuration_v0 import run_nulls_and_controls


if __name__ == "__main__":
    print(json.dumps(run_nulls_and_controls(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
