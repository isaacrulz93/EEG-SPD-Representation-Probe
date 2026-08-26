#!/usr/bin/env python3
"""Write final diagnostics, terminal, report, and validated manifests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.subject_location_conditional_configuration_v0 import generate_final_report


if __name__ == "__main__":
    print(json.dumps(generate_final_report(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
