#!/usr/bin/env python3
"""Freeze V1 protocol and run synthetic-only gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subject_class_population_structure_v1 import freeze_protocol

if __name__ == "__main__":
    print(json.dumps(freeze_protocol(ROOT), indent=2, sort_keys=True))
