#!/usr/bin/env python3
"""Evaluate the explicit source semantic-ordering assumption."""
from __future__ import annotations
import json
from pathlib import Path
from src.source_referenced_conditional_residual_v1 import run_source_ordering_assumption
if __name__ == "__main__":
    print(json.dumps(run_source_ordering_assumption(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
