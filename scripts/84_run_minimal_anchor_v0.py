#!/usr/bin/env python3
"""Run frozen minimal semantic anchoring calibration budgets."""

from __future__ import annotations

import json
from pathlib import Path

from src.unlabeled_conditional_mode_identifiability_v0 import run_minimal_anchor


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_minimal_anchor(root), indent=2, sort_keys=True))
