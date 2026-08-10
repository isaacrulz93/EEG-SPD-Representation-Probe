#!/usr/bin/env python3
"""Generate the final BNCI/OpenBMI decision, report, and Figure 9."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.openbmi_reporting_v0 import generate_openbmi_final_report


if __name__ == "__main__":
    print(json.dumps(generate_openbmi_final_report(ROOT), indent=2, sort_keys=True))
