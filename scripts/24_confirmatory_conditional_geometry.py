#!/usr/bin/env python3
"""After a valid unlock, compute confirmatory D/G objects and hard gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conditional_nulls_v1 import PHASE_CONFIRMATORY
from src.conditional_geometry_v1 import ConditionalGeometryError
from src.conditional_pipeline_v1 import (
    ConditionalPipelineError,
    classify_recognized_phase_failure,
    producer_code_commit,
    run_phase_geometry_producer,
    write_unassessed_failure_artifacts,
)
from src.conditional_provenance_v1 import (
    validate_confirmatory_unlock,
    validate_frozen_protocol_outputs,
)
from src.data_conditional_v1 import (
    ConditionalDataError,
    load_conditional_config,
    load_confirmatory_whole,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--unlock", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    config, config_path, config_hash, _protocol_hash = load_conditional_config(
        args.config, root
    )
    validate_frozen_protocol_outputs(config_path, root)
    # The unlock is deliberately validated before the data entry point.  The
    # data loader repeats the same barrier before resolving any raw/cache path.
    validate_confirmatory_unlock(config_path, root, unlock_path=args.unlock)
    try:
        data = load_confirmatory_whole(config_path, root, unlock_path=args.unlock)
        result = run_phase_geometry_producer(
            data,
            config,
            config_sha256=config_hash,
            repo_root=root,
            phase="confirmatory",
            phase_tag=PHASE_CONFIRMATORY,
        )
    except (ConditionalDataError, ConditionalGeometryError, ConditionalPipelineError) as error:
        result = write_unassessed_failure_artifacts(
            config,
            root,
            config_sha256=config_hash,
            code_commit=producer_code_commit(root),
            phase="confirmatory",
            session=str(config["dataset"]["confirmatory_session"]),
            failure_class=classify_recognized_phase_failure(error),
            reason_code=type(error).__name__,
            reason=str(error),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    printable = {key: value for key, value in result.items() if key != "manifest"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0 if result["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
