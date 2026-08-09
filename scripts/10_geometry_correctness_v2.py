#!/usr/bin/env python3
"""Run the preregistered V2 geometry hard gate before any classification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_v2 import load_config, load_v2_whole
from src.geometry_audit_v2 import (
    GeometryGateResult,
    run_geometry_gate,
    technical_failure_result,
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


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _freeze_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise RuntimeError(
            f"refusing to overwrite a conflicting frozen artifact: {destination}"
        )
    if not destination.exists():
        shutil.copy2(source, destination)


def _environment() -> dict[str, Any]:
    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "mne": "mne",
        "moabb": "moabb",
        "pyriemann": "pyriemann",
    }
    versions: dict[str, str | None] = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


def _write_result(
    result: GeometryGateResult,
    tables_dir: Path,
    *,
    gate_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    result.correctness.to_csv(tables_dir / "geometry_correctness.csv", index=False)
    result.mean_comparison.to_csv(
        tables_dir / "geometry_mean_comparison.csv", index=False
    )
    gate_document = dict(result.gate)
    if gate_metadata is not None:
        gate_document.update(gate_metadata)
    _write_json(tables_dir / "geometry_gate.json", gate_document)
    return gate_document


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config: dict[str, Any] | None = None
    data_provenance: dict[str, Any] = {}
    setup_error: BaseException | None = None

    try:
        config = load_config(config_path)
    except Exception as error:
        setup_error = error

    if config is None:
        output_root = PROJECT_ROOT / "outputs/bnci2014_001_geometry_v2"
        tables_dir = output_root / "tables"
        protocol_dir = output_root / "protocol"
        protocol_dir.mkdir(parents=True, exist_ok=True)
        result = technical_failure_result(setup_error or RuntimeError("config failure"))
        gate_document = _write_result(result, tables_dir)
        _write_json(protocol_dir / "environment.json", _environment())
        print(json.dumps(gate_document, indent=2, sort_keys=True))
        return 1

    output_root = _resolve(PROJECT_ROOT, config["project"]["output_dir"])
    tables_dir = _resolve(PROJECT_ROOT, config["outputs"]["tables_dir"])
    protocol_dir = output_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    _freeze_copy(config_path, protocol_dir / "frozen_config.yaml")
    protocol_source = _resolve(PROJECT_ROOT, config["protocol"]["protocol_path"])
    _freeze_copy(protocol_source, protocol_dir / "PROTOCOL_GEOMETRY_V2.md")
    _write_json(protocol_dir / "environment.json", _environment())

    try:
        data = load_v2_whole(config, PROJECT_ROOT)
        data_provenance = dict(data.provenance)
        result = run_geometry_gate(data, config)
    except Exception as error:
        setup_error = error
        result = technical_failure_result(error)

    provenance = {
        "protocol_name": config["protocol"]["name"],
        "protocol_version": str(config["protocol"]["version"]),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "protocol_path": str(protocol_source),
        "protocol_sha256": _sha256_file(protocol_source),
        "data": data_provenance,
        "setup_error": None
        if setup_error is None
        else f"{type(setup_error).__name__}: {setup_error}",
    }
    _write_json(protocol_dir / "provenance.json", provenance)
    gate_document = _write_result(
        result,
        tables_dir,
        gate_metadata={
            "protocol_version": str(config["protocol"]["version"]),
            "protocol_sha256": _sha256_file(protocol_source),
            "config_sha256": _sha256_file(config_path),
        },
    )
    print(json.dumps(gate_document, indent=2, sort_keys=True))
    return 0 if result.gate["classification_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
