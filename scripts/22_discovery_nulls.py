#!/usr/bin/env python3
"""Run resumable discovery R/S/P nulls after all geometry gates pass."""

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
    run_discovery_label_dry_run,
    run_discovery_null_producer,
    validate_label_null_dry_run,
)
from src.data_conditional_v1 import load_conditional_config, load_discovery_whole
from src.conditional_provenance_v1 import validate_frozen_protocol_outputs


DEFAULT_CONFIG = PROJECT_ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Shared-memory label-null worker threads (default: min(8, CPU count))",
    )
    parser.add_argument(
        "--dry-run-replicates",
        type=int,
        default=0,
        help="Time only this many label-null refits; 0 runs all frozen official nulls",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.dry_run_replicates < 0:
        raise SystemExit("--workers must be positive and --dry-run-replicates non-negative")
    root = args.repo_root.expanduser().resolve()
    config, config_path, config_hash, _protocol_hash = load_conditional_config(
        args.config, root
    )
    validate_frozen_protocol_outputs(config_path, root)
    data = load_discovery_whole(config_path, root)
    bundle = load_phase_geometry_bundle(
        config,
        root,
        phase="discovery",
        session=data.session,
        config_sha256=config_hash,
        code_commit=producer_code_commit(root),
    )
    if args.dry_run_replicates:
        result = run_discovery_label_dry_run(
            data,
            bundle,
            config,
            root,
            replicates=args.dry_run_replicates,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    validate_label_null_dry_run(config, root, bundle)
    result = run_discovery_null_producer(
        data, bundle, config, root, workers=args.workers
    )
    output = {
        "phase": "discovery",
        "label_replicates": int(result.label_group_statistics.shape[-1]),
        "semantic_replicates": int(result.semantic_group_statistics.shape[-1]),
        "oracle_replicates": int(result.oracle_group_statistics.shape[-1]),
        "group_summary_rows": len(result.group_summary),
        "status": "COMPLETE",
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
