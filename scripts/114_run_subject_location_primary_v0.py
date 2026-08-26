#!/usr/bin/env python3
"""Fit source-only models and freeze all held-out predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.subject_location_conditional_configuration_v0 import run_primary_predictions


if __name__ == "__main__":
    print(json.dumps(run_primary_predictions(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
