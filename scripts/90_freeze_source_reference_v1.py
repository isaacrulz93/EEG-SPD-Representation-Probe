#!/usr/bin/env python3
"""Freeze source-referenced conditional residual V1 before real access."""
from __future__ import annotations
import json
from pathlib import Path
from src.source_referenced_conditional_residual_v1 import freeze_protocol
if __name__ == "__main__":
    print(json.dumps(freeze_protocol(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
