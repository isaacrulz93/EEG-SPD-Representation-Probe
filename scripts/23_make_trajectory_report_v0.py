#!/usr/bin/env python3
"""Create the frozen Trajectory Anatomy v0 summary, figures, and report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_trajectory_v0 import load_trajectory_config
from src.reporting_trajectory_v0 import (
    FIGURE_STEMS,
    GEOMETRY_GATE_FILENAME,
    NULL_FILENAMES,
    PREREQUISITE_FILENAMES,
    PROTOCOL_VERSION,
    REPORT_TITLE,
    ReportingContractError,
    create_reporting_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "bnci2014_001_trajectory_v0.yaml",
        help="Frozen Trajectory Anatomy v0 YAML config",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ReportingContractError(
            f"{label} escapes the frozen trajectory output namespace: {path}"
        ) from error


def _validate_paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    project = config["project"]
    output_root = _resolve(project["output_dir"])
    expected_root = (PROJECT_ROOT / "outputs" / "bnci2014_001_trajectory_v0").resolve()
    if output_root != expected_root:
        raise ReportingContractError(
            f"output root must be exactly {expected_root}; observed {output_root}"
        )
    if str(project["protocol_version"]) != PROTOCOL_VERSION:
        raise ReportingContractError("protocol version is not frozen Trajectory Anatomy v0")
    configured_stems = tuple(str(value) for value in config["figures"]["stems"])
    if configured_stems != FIGURE_STEMS:
        raise ReportingContractError(
            f"configured figure stems differ from protocol: {configured_stems}"
        )
    formats = tuple(str(value).lower() for value in config["figures"]["format"])
    if formats != ("png", "pdf", "csv"):
        raise ReportingContractError(
            f"figure formats must be exactly png,pdf,csv; observed {formats}"
        )
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    nulls_dir = output_root / "nulls"
    report_path = output_root / "report" / "trajectory_anatomy_v0.md"
    for path, label in (
        (tables_dir, "tables_dir"), (figures_dir, "figures_dir"),
        (nulls_dir, "nulls_dir"), (report_path, "report_path"),
    ):
        _inside(path, output_root, label)
    return tables_dir, figures_dir, nulls_dir, report_path


def _read_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for name in PREREQUISITE_FILENAMES:
        path = tables_dir / name
        if not path.is_file():
            # Missing prerequisites are preserved as a protocol/technical
            # failure so reporting can emit UNASSESSED instead of crashing.
            continue
        try:
            result[name] = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError):
            # An unreadable file is equivalent to a missing required table and
            # will be named by validate_reporting_inputs.
            continue
    return result


def _read_null_artifacts(nulls_dir: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in NULL_FILENAMES:
        path = nulls_dir / name
        if not path.is_file():
            continue
        if path.suffix == ".json":
            try:
                with path.open("r", encoding="utf-8") as handle:
                    result[name] = json.load(handle)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        else:
            try:
                with np.load(path, allow_pickle=False) as archive:
                    result[name] = {key: np.asarray(archive[key]).copy() for key in archive.files}
            except (OSError, ValueError):
                continue
    return result


def _read_geometry_gate(tables_dir: Path) -> object:
    """Read the required stage-20 gate without converting failure to success."""

    path = tables_dir / GEOMETRY_GATE_FILENAME
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_trajectory_config(config_path)
    config_sha256 = _sha256_file(config_path)
    tables_dir, figures_dir, nulls_dir, report_path = _validate_paths(config)
    tables = _read_tables(tables_dir)
    null_artifacts = _read_null_artifacts(nulls_dir)
    geometry_gate = _read_geometry_gate(tables_dir)
    artifacts = create_reporting_outputs(
        tables,
        null_artifacts,
        geometry_gate,
        protocol_version=str(config["project"]["protocol_version"]),
        protocol_sha256=str(config["project"]["protocol_sha256"]),
        config_sha256=config_sha256,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        report_path=report_path,
        strict_counts=True,
    )
    print(
        json.dumps(
            {
                "report_title": REPORT_TITLE,
                "verdict": artifacts.verdict.verdict,
                "failure_status": artifacts.verdict.failure_status or None,
                "numerical_failure_count": len(artifacts.verdict.numerical_failures),
                "technical_failure_count": len(artifacts.verdict.technical_failures),
                "decision_sha256": artifacts.decision_sha256,
                "geometry_gate": str(tables_dir / GEOMETRY_GATE_FILENAME),
                "summary": str(tables_dir / "trajectory_v0_summary.csv"),
                "figure_stems": list(FIGURE_STEMS),
                "report": str(report_path),
                "next_experiment": artifacts.verdict.next_experiment,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
