#!/usr/bin/env python3
"""Freeze protocol/config and write reproducibility manifests before outcomes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline_trajectory_v0.io import atomic_json, atomic_text


OUT = ROOT / "outputs/bnci2014_001_baseline_trajectory_identifiability_v0"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = ROOT / "docs/PROTOCOL_BASELINE_TRAJECTORY_IDENTIFIABILITY_V0.md"
    config = ROOT / "configs/bnci2014_001_baseline_trajectory_v0.yaml"
    frozen = OUT / "protocol"; frozen.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(protocol, frozen / "frozen_protocol.md")
    shutil.copyfile(config, frozen / "frozen_config.yaml")
    shutil.copyfile(ROOT / "docs/PRIOR_EVIDENCE_AUDIT.md", OUT / "PRIOR_EVIDENCE_AUDIT.md")
    shutil.copyfile(ROOT / "docs/LITERATURE_OVERLAP_AUDIT.md", OUT / "LITERATURE_OVERLAP_AUDIT.md")
    atomic_json(frozen / "hashes.json", {
        "frozen_protocol.md": digest(frozen / "frozen_protocol.md"),
        "frozen_config.yaml": digest(frozen / "frozen_config.yaml"),
    })
    packages = ["numpy", "scipy", "pandas", "scikit-learn", "mne", "moabb",
                "pyriemann", "pymanopt", "torch"]
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "machine": platform.machine(), "cpu_count": os.cpu_count(),
                   "packages": {p: importlib.metadata.version(p) for p in packages}}
    atomic_json(OUT / "environment.json", environment)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    atomic_json(OUT / "git_provenance.json", {"branch": branch, "freeze_parent_sha": sha,
                "base_sha": "4a0e7966f676a02838e4300114667ff5e40cd9ae"})
    expected = ["STATUS.md", "DATA_TIMING_AUDIT.md", "PRIOR_EVIDENCE_AUDIT.md",
                "LITERATURE_OVERLAP_AUDIT.md", "manifest.json", "environment.json",
                "git_provenance.json", "REPORT.md", "HANDOFF.md"]
    atomic_json(OUT / "manifest.json", {"protocol": "baseline_trajectory_identifiability_v0",
                "expected_top_level": expected, "raw_eeg_in_git": False,
                "large_cache_in_git": False, "data_gate": "PASS"})
    atomic_text(OUT / "HANDOFF.md", "# HANDOFF\n\nScientific execution pending frozen Phase 0.\n")


if __name__ == "__main__":
    main()
