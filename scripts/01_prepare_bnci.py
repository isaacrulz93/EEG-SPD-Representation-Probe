#!/usr/bin/env python3
"""Prepare the frozen BNCI2014_001 primary-session epochs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import prepare_bnci2014_001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_bnci2014_001(args.config, PROJECT_ROOT)
    metadata = result["dataset_metadata"]
    print(
        json.dumps(
            {
                "prepared_epochs": result["prepared_epochs"],
                "shape": metadata["array_shape"],
                "subjects": metadata["n_subjects"],
                "sessions": metadata["sessions_observed"],
                "classes": metadata["classes_observed"],
                "sampling_frequency_hz": metadata["sampling_frequency_hz"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
