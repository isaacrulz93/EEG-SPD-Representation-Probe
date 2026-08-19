#!/usr/bin/env python3
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.selective_conditional_memory_feasibility_v0 import run_oracle_ceiling

if __name__ == "__main__":
    print(json.dumps(run_oracle_ceiling(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
