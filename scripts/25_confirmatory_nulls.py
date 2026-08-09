#!/usr/bin/env python3
"""Run confirmatory reliability and fixed-discovery-template semantic nulls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conditional_pipeline_v1 import (
    load_phase_geometry_bundle,
    producer_code_commit,
    run_confirmatory_rs_producer,
    validate_label_null_dry_run,
)
from src.conditional_provenance_v1 import (
    validate_confirmatory_unlock,
    validate_frozen_protocol_outputs,
)
from src.data_conditional_v1 import load_conditional_config, load_confirmatory_whole


DEFAULT_CONFIG = PROJECT_ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--unlock", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Shared-memory label-null worker threads (default: min(8, CPU count))",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    root = args.repo_root.expanduser().resolve()
    config, config_path, config_hash, _protocol_hash = load_conditional_config(
        args.config, root
    )
    validate_frozen_protocol_outputs(config_path, root)
    validate_confirmatory_unlock(config_path, root, unlock_path=args.unlock)
    code_commit = producer_code_commit(root)
    discovery = load_phase_geometry_bundle(
        config,
        root,
        phase="discovery",
        session=str(config["dataset"]["discovery_session"]),
        config_sha256=config_hash,
        code_commit=code_commit,
    )
    validate_label_null_dry_run(config, root, discovery)
    confirmatory = load_phase_geometry_bundle(
        config,
        root,
        phase="confirmatory",
        session=str(config["dataset"]["confirmatory_session"]),
        config_sha256=config_hash,
        code_commit=code_commit,
    )
    data = load_confirmatory_whole(config_path, root, unlock_path=args.unlock)
    result = run_confirmatory_rs_producer(
        data,
        discovery,
        confirmatory,
        config,
        root,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
