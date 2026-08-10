#!/usr/bin/env python3
"""Run/resume frozen BNCI Stage R/C label nulls and exact Stage I nulls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interaction_pipeline_v0 import run_bnci_nulls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_bnci_nulls(ROOT, batch_size=args.batch_size, workers=args.workers), indent=2, sort_keys=True))
