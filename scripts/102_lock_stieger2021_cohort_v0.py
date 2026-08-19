#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from src.stieger2021_multiclass_confirmation_v0 import lock_cohort_and_objects
if __name__ == "__main__":
    print(json.dumps(lock_cohort_and_objects(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
