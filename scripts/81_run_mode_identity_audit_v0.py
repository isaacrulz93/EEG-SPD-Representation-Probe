#!/usr/bin/env python3
"""Rebuild exact trial covariances and audit the frozen rank-1 directions."""

from __future__ import annotations

import json
from pathlib import Path

from src.unlabeled_conditional_mode_identifiability_v0 import mode_identity_audit


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(mode_identity_audit(root), indent=2, sort_keys=True))
