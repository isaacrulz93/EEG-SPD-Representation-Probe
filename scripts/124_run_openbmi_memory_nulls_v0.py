#!/usr/bin/env python3
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.returning_user_conditional_memory_v0 import run_openbmi_nulls

if __name__ == "__main__":
    print(json.dumps(run_openbmi_nulls(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
