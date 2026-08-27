#!/usr/bin/env python3
"""Validate the parent and freeze the V0 protocol without target outcomes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.subject_location_conditional_configuration_v0 import freeze_protocol


if __name__ == "__main__":
    print(json.dumps(freeze_protocol(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
