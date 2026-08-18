#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from src.stieger2021_multiclass_confirmation_v0 import run_component_assignment
if __name__ == "__main__": print(json.dumps(run_component_assignment(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
