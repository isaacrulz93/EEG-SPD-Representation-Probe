#!/usr/bin/env python3
"""Run the exact held-out beta-reference identity gate."""
from __future__ import annotations
import json
from pathlib import Path
from src.source_referenced_conditional_residual_v1 import run_beta_identity_audit
if __name__ == "__main__":
    print(json.dumps(run_beta_identity_audit(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
