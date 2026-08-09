#!/usr/bin/env python3
"""Build frozen WHOLE and WINDOW5 covariance representations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.covariance import build_and_save_covariances
from src.data import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = build_and_save_covariances(config, PROJECT_ROOT)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
