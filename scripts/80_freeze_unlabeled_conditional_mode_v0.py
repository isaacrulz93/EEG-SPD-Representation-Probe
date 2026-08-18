#!/usr/bin/env python3
"""Freeze Unlabeled Conditional-Mode Identifiability V0 before real access."""

from __future__ import annotations

import json
from pathlib import Path

from src.unlabeled_conditional_mode_identifiability_v0 import freeze_protocol


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(freeze_protocol(root), indent=2, sort_keys=True))
