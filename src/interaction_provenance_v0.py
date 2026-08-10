"""Hashing, canonical serialization, and atomic provenance for v0."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def git_output(repo_root: str | Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=Path(repo_root), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def git_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    tracked = git_output(root, "ls-files", "src", "scripts", "configs", "docs").splitlines()
    digest = hashlib.sha256()
    for relative in sorted(tracked):
        path = root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return {
        "head": git_output(root, "rev-parse", "HEAD"),
        "branch": git_output(root, "branch", "--show-current"),
        "status_porcelain": git_output(root, "status", "--porcelain", "--untracked-files=all").splitlines(),
        "tracked_code_sha256": digest.hexdigest(),
    }


def environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyriemann", "matplotlib", "mne", "moabb", "PyYAML", "pytest"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cpu_count": os.cpu_count(),
    }
