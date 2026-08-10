#!/usr/bin/env python3
"""Run frozen observed BNCI retrospective-development interaction objects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interaction_pipeline_v0 import run_bnci_observed


if __name__ == "__main__":
    print(json.dumps(run_bnci_observed(ROOT), indent=2, sort_keys=True))
