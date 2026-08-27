#!/usr/bin/env python3
"""Lock source/input packets and separate sealed target outcome vaults."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.subject_location_conditional_configuration_v0 import lock_subject_location_objects


if __name__ == "__main__":
    print(json.dumps(lock_subject_location_objects(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
