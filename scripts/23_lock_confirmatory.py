#!/usr/bin/env python3
"""Create the clean-HEAD Conditional Geometry v1 confirmatory unlock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conditional_provenance_v1 import create_confirmatory_unlock


DEFAULT_CONFIG = PROJECT_ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional repository-contained unlock path override",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_confirmatory_unlock(
        args.config, args.repo_root, output_path=args.output
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "confirmatory_designation": manifest[
                    "confirmatory_designation"
                ],
                "locked_head": manifest["locked_head"],
                "code_commit": manifest["code_commit"],
                "manifest_sha256": manifest["manifest_sha256"],
                "discovery_file_count": manifest["discovery_snapshot"][
                    "file_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
