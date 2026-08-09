#!/usr/bin/env python3
"""Run the frozen primary analysis from preparation through report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.expanduser().resolve()
    for script in [
        "01_prepare_bnci.py",
        "02_build_covariances.py",
        "03_make_embeddings.py",
        "04_compute_diagnostics.py",
        "05_make_report.py",
    ]:
        command = [sys.executable, str(ROOT / "scripts" / script), "--config", str(config)]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

