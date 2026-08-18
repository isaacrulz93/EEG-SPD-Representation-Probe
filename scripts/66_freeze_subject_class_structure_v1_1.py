#!/usr/bin/env python3
"""Freeze the V1.1 technical amendment and synthetic recovery gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subject_class_population_structure_v1_1 import freeze_amendment

if __name__ == "__main__":
    print(json.dumps(freeze_amendment(ROOT), indent=2, sort_keys=True))
