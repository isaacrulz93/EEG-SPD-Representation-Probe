#!/usr/bin/env python3
"""Run corrected zero/few-label estimators and factorial oracle controls."""
from __future__ import annotations
import json
from pathlib import Path
from src.source_referenced_conditional_residual_v1 import run_corrected_minimal_anchor
if __name__ == "__main__":
    print(json.dumps(run_corrected_minimal_anchor(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
