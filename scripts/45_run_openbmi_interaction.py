#!/usr/bin/env python3
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.openbmi_pipeline_v0 import run_openbmi_observed
if __name__=="__main__": print(json.dumps(run_openbmi_observed(ROOT),indent=2,sort_keys=True))
