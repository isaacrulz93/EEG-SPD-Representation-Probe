#!/usr/bin/env python3
"""Materialize the already committed frozen v0 protocol under the output root."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interaction_pipeline_v0 import freeze_output_protocol


if __name__ == "__main__":
    print(json.dumps(freeze_output_protocol(ROOT), indent=2, sort_keys=True))
