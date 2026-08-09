#!/usr/bin/env python3
"""Run the preregistered V2 primary LOSO evaluation after the geometry gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_v2 import load_config, load_v2_whole
from src.loso_v2 import (
    assert_classification_gate,
    assert_v2_output_contract,
    run_primary_loso,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tables(result: object, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "loso_logistic_transductive.csv": result.logistic_transductive,
        "loso_logistic_calibration.csv": result.logistic_calibration,
        "loso_mdm_transductive.csv": result.mdm_transductive,
        "loso_mdm_calibration.csv": result.mdm_calibration,
        "domain_shift_diagnostics.csv": result.domain_shift_diagnostics,
        "sample_id_audit.csv": result.sample_id_audit,
    }
    for filename, frame in outputs.items():
        destination = tables_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    config_sha256 = _sha256_file(config_path)
    tables_dir = assert_v2_output_contract(PROJECT_ROOT, config)

    # This is deliberately before V1 data loading and before every numerical
    # transform/classifier operation in the primary runner.
    gate = assert_classification_gate(
        tables_dir,
        expected_protocol_version=str(config["protocol"]["version"]),
        expected_protocol_sha256=str(config["protocol"]["protocol_sha256"]),
        expected_config_sha256=config_sha256,
    )
    data = load_v2_whole(config, PROJECT_ROOT)
    result = run_primary_loso(
        data, config, config_sha256=config_sha256
    )
    _write_tables(result, tables_dir)

    summary = {
        "classification_gate_pass": gate["classification_gate_pass"],
        "classification_failed": result.classification_failed,
        "fatal_error": result.fatal_error,
        "primary_status": result.primary_status,
        "primary_failure_count": result.primary_failure_count,
        "secondary_status": result.secondary_status,
        "secondary_failure_count": result.secondary_failure_count,
        "logistic_T1_rows": len(result.logistic_transductive),
        "logistic_T2_rows": len(result.logistic_calibration),
        "mdm_T1_rows": len(result.mdm_transductive),
        "mdm_T2_rows": len(result.mdm_calibration),
        "domain_shift_rows": len(result.domain_shift_diagnostics),
        "sample_id_audit_rows": len(result.sample_id_audit),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    # Recorded convergence failures are scientific outputs, not structural
    # exceptions.  Returning zero lets the frozen pipeline reach leakage audit
    # and the report, where Q1-Q3 must be marked technical/unassessed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
