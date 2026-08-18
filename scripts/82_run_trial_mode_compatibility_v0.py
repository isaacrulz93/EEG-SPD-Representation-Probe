#!/usr/bin/env python3
"""Run the frozen trial/prototype compatibility bridge."""

from __future__ import annotations

import json
from pathlib import Path

from src.unlabeled_conditional_mode_identifiability_v0 import run_trial_mode_compatibility


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_trial_mode_compatibility(root), indent=2, sort_keys=True))
