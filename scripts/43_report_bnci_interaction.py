#!/usr/bin/env python3
"""Generate the frozen BNCI decision screen, report, tables, and figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting_interaction_v0 import generate_bnci_report

if __name__ == "__main__":
    print(json.dumps(generate_bnci_report(ROOT), indent=2, sort_keys=True))
