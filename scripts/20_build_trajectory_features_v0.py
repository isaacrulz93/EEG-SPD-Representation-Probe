#!/usr/bin/env python3
"""Build frozen Trajectory Anatomy v0 features and numerical gates only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_trajectory_v0 import (  # noqa: E402
    load_trajectory_config,
    load_trajectory_window5,
)
from src.feature_pipeline_trajectory_v0 import (  # noqa: E402
    build_trajectory_feature_artifacts,
    write_trajectory_feature_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build 0train-only five-state AIRM/LE features and stop before "
            "scientific evaluation unless every hard gate passes."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: script parent repository)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "bnci2014_001_trajectory_v0.yaml",
        help="Frozen trajectory-v0 YAML config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = load_trajectory_config(config_path)
    data = load_trajectory_window5(config_path, root)

    def progress(completed: int, total: int) -> None:
        if completed == total or completed % 100 == 0:
            print(
                f"trajectory geometry: {completed}/{total} trials",
                file=sys.stderr,
                flush=True,
            )

    artifacts = build_trajectory_feature_artifacts(
        data,
        config,
        progress=progress,
    )
    result = write_trajectory_feature_artifacts(
        artifacts,
        config,
        root,
        config_path=config_path,
        protocol_path=root / str(config["project"]["protocol_path"]),
    )
    print(json.dumps(result["gate_summary"], indent=2, ensure_ascii=False))
    if not artifacts.gate_passed:
        print(
            "Scientific classification blocked: trajectory numerical/data gate failed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
