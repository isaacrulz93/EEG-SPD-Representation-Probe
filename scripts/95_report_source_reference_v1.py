#!/usr/bin/env python3
"""Create the source-reference V1 report, figures, and final manifest."""
from __future__ import annotations
import json
from pathlib import Path
from src.source_referenced_conditional_residual_v1 import generate_report
if __name__ == "__main__":
    print(json.dumps(generate_report(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
