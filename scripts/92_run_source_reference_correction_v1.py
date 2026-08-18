#!/usr/bin/env python3
"""Run the training-only source reference and curvature correction."""
from __future__ import annotations
import json
from pathlib import Path
from src.source_referenced_conditional_residual_v1 import run_source_reference_correction
if __name__ == "__main__":
    print(json.dumps(run_source_reference_correction(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
