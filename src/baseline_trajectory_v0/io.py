"""Atomic, resumable small-artifact I/O for weekend V0."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, target)


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, target)


def append_status(output: str | Path, stage: str, status: str, runtime: float,
                  new_files: str, hard_gate: str, next_action: str,
                  reason: str = "") -> None:
    path = Path(output) / "STATUS.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# STATUS\n\n"
    row = (
        f"\n## {stage}\n\n"
        f"- status: {status}\n- runtime_seconds: {runtime:.3f}\n"
        f"- new_files: {new_files}\n- hard_gate: {hard_gate}\n"
        f"- next_automatic_action: {next_action}\n- reason_if_stopped: {reason or 'none'}\n"
    )
    atomic_text(path, existing + row)
