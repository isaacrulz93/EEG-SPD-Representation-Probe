#!/usr/bin/env python3
"""Run the gate-protected frozen V1 all-sample/fold-safe leakage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_v2 import load_config, load_v2_whole
from src.leakage_audit_v2 import (
    FROZEN_C,
    FROZEN_FOLDS,
    FROZEN_MAX_ITER,
    FROZEN_SEED,
    FROZEN_SOLVER,
    FROZEN_TOL,
    GeometryGateError,
    require_geometry_gate,
    run_v1_leakage_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Frozen V2 YAML")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_frozen_audit_config(config: Mapping[str, Any]) -> None:
    audit = config["v1_leakage_audit"]
    logistic = config["classifiers"]["primary_logistic"]
    expected = {
        "folds": (int(audit["folds"]), FROZEN_FOLDS),
        "audit_seed": (int(audit["seed"]), FROZEN_SEED),
        "conditions": (list(audit["conditions"]), ["v1_all_sample", "fold_safe"]),
        "C": (float(logistic["C"]), FROZEN_C),
        "solver": (str(logistic["solver"]), FROZEN_SOLVER),
        "max_iter": (int(logistic["max_iter"]), FROZEN_MAX_ITER),
        "tol": (float(logistic["tol"]), FROZEN_TOL),
        "random_state": (int(logistic["random_state"]), FROZEN_SEED),
        "standard_scaler": (bool(logistic["standard_scaler"]), False),
        "normalization": (bool(logistic["normalization"]), False),
        "pca": (bool(logistic["pca"]), False),
        "tuning": (bool(logistic["tuning"]), False),
    }
    mismatches = [
        f"{name}: configured={actual!r}, required={required!r}"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if mismatches:
        raise RuntimeError("frozen V1 audit config mismatch: " + "; ".join(mismatches))


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    assert_frozen_audit_config(config)
    config_sha256 = sha256_file(config_path)
    tables_dir = resolve(config["outputs"]["tables_dir"])
    output_root = resolve(config["project"]["output_dir"])
    try:
        tables_dir.relative_to(output_root)
    except ValueError as error:
        raise RuntimeError("V1 audit table path escapes the V2 output root") from error
    gate_path = tables_dir / "geometry_gate.json"
    correctness_path = tables_dir / "geometry_correctness.csv"
    try:
        gate = require_geometry_gate(
            gate_path,
            correctness_path,
            expected_protocol_version=str(config["protocol"]["version"]),
            expected_protocol_sha256=str(config["protocol"]["protocol_sha256"]),
            expected_config_sha256=config_sha256,
        )
    except GeometryGateError as error:
        raise SystemExit(f"Classification prohibited: {error}") from error

    # Loading WHOLE data and all feature/classifier work occurs only after the
    # exact all-pass geometry gate above.
    data = load_v2_whole(config_path, ROOT)
    result = run_v1_leakage_audit(data.covariances, data.metadata)
    table = result.table.copy()
    table.insert(0, "protocol_version", str(config["protocol"]["version"]))
    table.insert(1, "protocol_sha256", str(config["protocol"]["protocol_sha256"]))
    table.insert(2, "config_sha256", config_sha256)
    table.insert(3, "geometry_gate_json_sha256", gate["gate_json_sha256"])
    table.insert(
        4,
        "geometry_correctness_csv_sha256",
        gate["geometry_correctness_csv_sha256"],
    )
    table.insert(5, "data_source", str(data.provenance["source"]))
    table.insert(
        6,
        "whole_array_content_sha256",
        str(data.provenance["numerical_checks"]["whole_array_content_sha256"]),
    )
    output_path = tables_dir / "v1_leakage_audit.csv"
    atomic_write_csv(table, output_path)

    pooled = table[table["row_type"] == "pooled_oof"]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": len(table),
                "geometry_gate_pass": True,
                "pooled": pooled[
                    [
                        "condition",
                        "balanced_accuracy",
                        "accuracy",
                        "macro_f1",
                        "actual_accuracy_difference_from_benchmark",
                        "classifier_status",
                    ]
                ].to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
