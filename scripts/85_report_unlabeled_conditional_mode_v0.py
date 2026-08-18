#!/usr/bin/env python3
"""Render and validate the final V0 report and artifact manifest."""

from __future__ import annotations

import json
from pathlib import Path

from src.unlabeled_conditional_mode_identifiability_v0 import generate_report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(generate_report(root), indent=2, sort_keys=True))
