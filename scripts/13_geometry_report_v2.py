#!/usr/bin/env python3
"""Create the gate-protected frozen V2 summary, five figures, and report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_v2 import load_config
from src.loso_v2 import ClassificationGateError, assert_classification_gate
from src.reporting_v2 import (
    FIGURE_STEMS,
    PREREQUISITE_FILENAMES,
    ReportingContractError,
    create_reporting_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/bnci2014_001_geometry_v2.yaml",
        help="Frozen V2 YAML config",
    )
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ReportingContractError(f"{label} escapes the V2 output root: {path}") from error


def _validate_output_contract(config: dict[str, object]) -> tuple[Path, Path, Path]:
    output_root = _resolve(config["project"]["output_dir"])
    tables_dir = _resolve(config["outputs"]["tables_dir"])
    figures_dir = _resolve(config["outputs"]["figures_dir"])
    report_path = _resolve(config["outputs"]["report_path"])
    for path, label in (
        (tables_dir, "tables_dir"),
        (figures_dir, "figures_dir"),
        (report_path, "report_path"),
    ):
        _inside(path, output_root, label)
    if "bnci2014_001_geometry_v2" not in output_root.parts:
        raise ReportingContractError("reporting output root is not the frozen V2 namespace")
    configured_stems = tuple(str(value) for value in config["outputs"]["required_figure_stems"])
    if configured_stems != FIGURE_STEMS:
        raise ReportingContractError(
            f"configured figure stems differ from protocol: {configured_stems}"
        )
    required_tables = tuple(str(value) for value in config["outputs"]["required_tables"])
    expected_tables = (*PREREQUISITE_FILENAMES, "geometry_v2_summary.csv")
    if required_tables != expected_tables:
        raise ReportingContractError(
            f"configured required tables differ from protocol: {required_tables}"
        )
    expected_report = output_root / "report" / "geometry_audit_v2.md"
    if report_path != expected_report:
        raise ReportingContractError(
            f"report path must be exactly {expected_report}, observed {report_path}"
        )
    return tables_dir, figures_dir, report_path


def _read_prerequisites(tables_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in PREREQUISITE_FILENAMES if not (tables_dir / name).is_file()]
    if missing:
        raise ReportingContractError(f"missing prerequisite tables: {missing}")
    result: dict[str, pd.DataFrame] = {}
    for filename in PREREQUISITE_FILENAMES:
        path = tables_dir / filename
        try:
            result[filename] = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise ReportingContractError(f"could not parse prerequisite table {path}") from error
    return result


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    config_sha256 = _sha256_file(config_path)
    tables_dir, figures_dir, report_path = _validate_output_contract(config)

    # No summary/plot/report write occurs until the existing classification
    # gate and all eight prerequisite tables have been validated.
    try:
        gate = assert_classification_gate(
            tables_dir,
            expected_protocol_version=str(config["protocol"]["version"]),
            expected_protocol_sha256=str(config["protocol"]["protocol_sha256"]),
            expected_config_sha256=config_sha256,
        )
    except ClassificationGateError as error:
        raise SystemExit(f"Reporting prohibited: {error}") from error
    tables = _read_prerequisites(tables_dir)
    artifacts = create_reporting_outputs(
        tables,
        gate,
        config,
        config_sha256=config_sha256,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        report_path=report_path,
    )
    print(
        json.dumps(
            {
                "classification_gate_pass": True,
                "summary": str(tables_dir / "geometry_v2_summary.csv"),
                "summary_rows": len(artifacts.summary),
                "figures": [str(figures_dir / f"{stem}.png") for stem in FIGURE_STEMS],
                "figure_sources": [str(figures_dir / f"{stem}.csv") for stem in FIGURE_STEMS],
                "report": str(report_path),
                "verdicts": {
                    "Q1": artifacts.verdicts.q1,
                    "Q2": artifacts.verdicts.q2,
                    "Q3": artifacts.verdicts.q3,
                },
                "next_experiment": artifacts.verdicts.next_experiment,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
