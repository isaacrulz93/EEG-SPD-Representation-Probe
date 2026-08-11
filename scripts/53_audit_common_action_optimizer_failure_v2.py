#!/usr/bin/env python3
"""Run the preregistered synthetic-only optimizer stress audit."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common_action_optimizer_audit_v2 import (
    run_rows,
    run_stress_suite,
    summarize_stress_suite,
)


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> None:
    output = ROOT / "outputs/common_action_optimizer_failure_audit_v2"
    output.mkdir(parents=True, exist_ok=True)
    runs, fixtures = run_stress_suite()
    rows = run_rows(runs)
    with (output / "synthetic_optimizer_runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "synthetic_fixture_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fixtures[0]))
        writer.writeheader()
        writer.writerows(fixtures)
    summary = summarize_stress_suite(runs, fixtures)
    (output / "synthetic_optimizer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
