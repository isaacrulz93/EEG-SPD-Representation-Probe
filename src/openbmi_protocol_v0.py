"""Strict external-replication manifest and scientific-access lock."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from src.interaction_provenance_v0 import git_output, sha256_file


class OpenBMILockError(RuntimeError):
    pass


REQUIRED_MANIFEST_FIELDS = {
    "dataset_identifier", "moabb_api_name", "subject_ids", "session_ids",
    "offline_phase_identifier", "eeg_channels", "epoch_seconds", "bandpass_hz",
    "sampling_frequency_hz", "sample_rate_handling", "expected_counts",
    "data_source", "data_version", "source_citations", "input_hashes",
}


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise OpenBMILockError(f"OpenBMI manifest missing fields: {sorted(missing)}")
    channels = manifest["eeg_channels"]
    if not isinstance(channels, list) or len(channels) != 20 or len(set(channels)) != 20:
        raise OpenBMILockError("OpenBMI manifest requires exactly 20 unique authoritative channels")
    if list(manifest["epoch_seconds"]) != [1.0, 3.5] or list(manifest["bandpass_hz"]) != [8.0, 30.0]:
        raise OpenBMILockError("OpenBMI manifest differs from frozen epoch/bandpass")
    subjects = manifest["subject_ids"]
    if not isinstance(subjects, list) or len(subjects) != 54 or len(set(subjects)) != 54:
        raise OpenBMILockError("OpenBMI manifest requires 54 unique two-session subjects")


def validate_scientific_unlock(repo_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_path = root / str(config["project"]["openbmi_manifest_path"])
    unlock_path = root / str(config["project"]["openbmi_unlock_path"])
    if not manifest_path.is_file() or not unlock_path.is_file():
        raise OpenBMILockError("OpenBMI scientific evaluation is locked until manifest freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    manifest_hash = sha256_file(manifest_path)
    if unlock.get("manifest_sha256") != manifest_hash:
        raise OpenBMILockError("OpenBMI unlock manifest SHA mismatch")
    commit = str(unlock.get("manifest_commit_sha", ""))
    if len(commit) != 40:
        raise OpenBMILockError("OpenBMI unlock lacks a full manifest commit SHA")
    commit_message = git_output(root, "show", "-s", "--format=%s", commit)
    if commit_message != "freeze OpenBMI external replication manifest":
        raise OpenBMILockError("OpenBMI manifest commit message mismatch")
    import hashlib
    committed_bytes = subprocess.run(
        ["git", "show", f"{commit}:{config['project']['openbmi_manifest_path']}"],
        cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(committed_bytes).hexdigest() != manifest_hash:
        raise OpenBMILockError("OpenBMI manifest bytes are not those frozen at unlock commit")
    return unlock
