"""Frozen provenance and session-lock utilities for Conditional Geometry v1.

The confirmatory validator deliberately knows nothing about MOABB or raw EEG
loading.  It validates the lock, source tree, and immutable discovery snapshot
before :mod:`src.data_conditional_v1` is allowed to resolve any confirmatory
input path.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


UNLOCK_SCHEMA_VERSION = "1.0"
UNLOCK_STATUS = "CONFIRMATORY_UNLOCKED"

# These final comparison/decision tables require confirmatory data and therefore
# cannot be part of the immutable discovery snapshot.  Every other frozen
# required table, object, and null artifact must already exist before unlock.
_POST_CONFIRMATORY_ONLY_TABLES = {
    "discovery_confirmatory_comparison.csv",
    "airm_le_robustness.csv",
    "hypothesis_chain_status.csv",
}


class ConditionalProvenanceError(RuntimeError):
    """Raised when frozen provenance cannot be established or reproduced."""


class ConfirmatoryLockError(ConditionalProvenanceError):
    """Raised before any confirmatory input may be resolved or opened."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the canonical JSON serialization used by all manifest hashes."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without reading it all into memory."""

    target = Path(path)
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as error:
        raise ConditionalProvenanceError(f"Required file is missing: {target}") from error
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(Path(path), content)


def _freeze_copy(source: Path, destination: Path) -> None:
    source_bytes = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != source_bytes:
            raise ConditionalProvenanceError(
                f"Frozen artifact differs from its source: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(source_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _resolve_inside(root: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ConditionalProvenanceError(
            f"{label} escapes the repository root: {resolved}"
        ) from error
    return resolved


def load_protocol_config(
    config_path: str | Path, repo_root: str | Path
) -> tuple[dict[str, Any], Path, Path, str, str]:
    """Load config and verify the frozen protocol digest.

    This function resolves only the config and protocol paths.  It never
    touches paths in ``confirmatory_inputs.raw_files``.
    """

    root = Path(repo_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise ConditionalProvenanceError(f"Could not load frozen config: {path}") from error
    if not isinstance(config, dict):
        raise ConditionalProvenanceError("Frozen config must be a YAML mapping")
    for section in ("protocol", "project", "dataset", "confirmatory_inputs"):
        if not isinstance(config.get(section), dict):
            raise ConditionalProvenanceError(f"Missing config mapping: {section}")

    protocol_path = _resolve_inside(
        root, config["protocol"].get("protocol_path", ""), label="protocol_path"
    )
    protocol_hash = sha256_file(protocol_path)
    expected_protocol_hash = str(config["protocol"].get("protocol_sha256", ""))
    if protocol_hash != expected_protocol_hash:
        raise ConditionalProvenanceError(
            "Frozen protocol SHA-256 mismatch: "
            f"expected {expected_protocol_hash}, observed {protocol_hash}"
        )
    return config, path, protocol_path, sha256_file(path), protocol_hash


def _run_git(root: Path, *args: str, allow_empty: bool = False) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise ConditionalProvenanceError(
            f"git {' '.join(args)} failed: {process.stderr.strip()}"
        )
    output = process.stdout.strip()
    if not output and not allow_empty:
        raise ConditionalProvenanceError(f"git {' '.join(args)} returned no value")
    return output


def git_head(repo_root: str | Path) -> str:
    return _run_git(Path(repo_root).resolve(), "rev-parse", "HEAD")


def git_branch(repo_root: str | Path) -> str:
    return _run_git(Path(repo_root).resolve(), "branch", "--show-current")


def git_status_porcelain(repo_root: str | Path) -> str:
    return _run_git(
        Path(repo_root).resolve(),
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        allow_empty=True,
    )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ConditionalProvenanceError(f"Path escapes repository: {path}") from error


def _file_records(root: Path, files: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(files, key=lambda item: _relative(root, item)):
        if path.is_symlink():
            raise ConditionalProvenanceError(f"Snapshot symlinks are forbidden: {path}")
        if not path.is_file():
            raise ConditionalProvenanceError(f"Snapshot file is missing: {path}")
        records.append(
            {
                "path": _relative(root, path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return records


def _snapshot_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "algorithm": "sha256_canonical_json_sorted_file_records_v1",
        "file_count": len(records),
        "total_bytes": int(sum(int(record["bytes"]) for record in records)),
        "files": records,
        "aggregate_sha256": payload_sha256(records),
    }


def discovery_snapshot(
    config: Mapping[str, Any], repo_root: str | Path, *, require_tracked: bool
) -> dict[str, Any]:
    """Hash every immutable file in the configured discovery directories."""

    root = Path(repo_root).resolve()
    output_root = _resolve_inside(
        root, config["project"]["output_dir"], label="project.output_dir"
    )
    directories = list(config["project"].get("discovery_snapshot_dirs", []))
    if not directories:
        raise ConditionalProvenanceError("No discovery snapshot directories configured")
    files: list[Path] = []
    seen: set[Path] = set()
    for relative_directory in directories:
        directory = _resolve_inside(
            output_root, relative_directory, label="discovery snapshot directory"
        )
        if not directory.is_dir():
            raise ConditionalProvenanceError(
                f"Discovery snapshot directory is missing: {directory}"
            )
        local_files = sorted(path for path in directory.rglob("*") if path.is_file())
        if not local_files:
            raise ConditionalProvenanceError(
                f"Discovery snapshot directory is empty: {directory}"
            )
        for path in local_files:
            resolved = path.resolve()
            if resolved in seen:
                raise ConditionalProvenanceError(f"Duplicate discovery file: {path}")
            seen.add(resolved)
            files.append(path)

    outputs = config.get("outputs")
    if isinstance(outputs, Mapping):
        required_relative = [
            *(f"objects/discovery/{name}" for name in outputs.get("required_objects", [])),
            *(f"nulls/discovery/{name}" for name in outputs.get("required_nulls", [])),
            *(
                f"tables/discovery/{name}"
                for name in outputs.get("required_tables", [])
                if str(name) not in _POST_CONFIRMATORY_ONLY_TABLES
            ),
        ]
        missing_required = [
            relative
            for relative in required_relative
            if not (output_root / relative).is_file()
        ]
        if missing_required:
            raise ConditionalProvenanceError(
                "Required discovery artifacts are missing before unlock: "
                f"{missing_required}"
            )

    records = _file_records(root, files)
    if require_tracked:
        tracked = set(
            _run_git(root, "ls-files", allow_empty=True).splitlines()
        )
        untracked = [record["path"] for record in records if record["path"] not in tracked]
        if untracked:
            raise ConditionalProvenanceError(
                "Discovery snapshot files must be committed before unlock: "
                f"{untracked}"
            )
    payload = _snapshot_payload(records)
    payload["directories"] = [str(value) for value in directories]
    return payload


def code_snapshot(
    config_path: Path, protocol_path: Path, repo_root: str | Path
) -> dict[str, Any]:
    """Hash the tracked source/script tree plus frozen config and protocol."""

    root = Path(repo_root).resolve()
    tracked = _run_git(root, "ls-files", "-z", "--", "src", "scripts", allow_empty=True)
    tracked_paths = [value for value in tracked.split("\0") if value]
    required = {_relative(root, config_path), _relative(root, protocol_path)}
    requirements = root / "requirements.txt"
    if requirements.is_file():
        required.add(_relative(root, requirements))
    selected = set(tracked_paths) | required
    files = [root / value for value in sorted(selected)]
    records = _file_records(root, files)
    return _snapshot_payload(records)


def _latest_code_commit(
    config_path: Path, protocol_path: Path, repo_root: Path
) -> str:
    paths = [
        "src",
        "scripts",
        _relative(repo_root, config_path),
        _relative(repo_root, protocol_path),
    ]
    if (repo_root / "requirements.txt").is_file():
        paths.append("requirements.txt")
    return _run_git(repo_root, "log", "-1", "--format=%H", "--", *paths)


def _first_definition_commit(
    config_path: Path, protocol_path: Path, repo_root: Path
) -> str:
    commits = _run_git(
        repo_root,
        "log",
        "--reverse",
        "--format=%H",
        "--",
        _relative(repo_root, config_path),
        _relative(repo_root, protocol_path),
    ).splitlines()
    if not commits or len(commits[0]) != 40:
        raise ConditionalProvenanceError(
            "Frozen config/protocol must already be committed before protocol freeze"
        )
    first = commits[0]
    tracked = set(_run_git(repo_root, "ls-files", allow_empty=True).splitlines())
    required = {_relative(repo_root, config_path), _relative(repo_root, protocol_path)}
    if not required.issubset(tracked):
        raise ConditionalProvenanceError(
            "Frozen config/protocol must be tracked before protocol freeze"
        )
    return first


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "os": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "cpu_count": os.cpu_count(),
        "packages": {
            key: _distribution_version(distribution)
            for key, distribution in {
                "numpy": "numpy",
                "scipy": "scipy",
                "pandas": "pandas",
                "scikit_learn": "scikit-learn",
                "pyriemann": "pyriemann",
                "mne": "mne",
                "moabb": "moabb",
                "pyyaml": "PyYAML",
            }.items()
        },
    }


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConditionalProvenanceError(f"Invalid existing JSON: {path}") from error
        stable_keys = ("protocol_sha256", "config_sha256", "protocol_name", "protocol_version")
        for key in stable_keys:
            if key in payload and existing.get(key) != payload.get(key):
                raise ConditionalProvenanceError(
                    f"Existing provenance conflicts at {key}: {path}"
                )
        return
    atomic_write_json(path, payload)


def freeze_conditional_protocol(
    config_path: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    """Copy the exact protocol/config and create initial root provenance."""

    root = Path(repo_root).resolve()
    config, config_file, protocol_file, config_hash, protocol_hash = load_protocol_config(
        config_path, root
    )
    output_root = _resolve_inside(
        root, config["project"]["output_dir"], label="project.output_dir"
    )
    before_status = git_status_porcelain(root)
    head = git_head(root)
    branch = git_branch(root)
    expected_branch = str(config["protocol"]["branch"])
    if branch != expected_branch:
        raise ConditionalProvenanceError(
            f"Protocol freeze branch mismatch: expected {expected_branch}, observed {branch}"
        )
    if before_status:
        raise ConditionalProvenanceError(
            "Protocol freeze requires a clean working tree; observed:\n" + before_status
        )
    definition_commit = _first_definition_commit(config_file, protocol_file, root)

    protocol_dir = output_root / "protocol"
    frozen_protocol = protocol_dir / protocol_file.name
    frozen_config = protocol_dir / "frozen_config.yaml"
    _freeze_copy(protocol_file, frozen_protocol)
    _freeze_copy(config_file, frozen_config)

    common = {
        "protocol_name": str(config["protocol"]["name"]),
        "protocol_version": str(config["protocol"]["version"]),
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
    }
    manifest = {
        **common,
        "schema_version": "1.0",
        "phase": "PROTOCOL_FROZEN",
        "branch": str(config["protocol"]["branch"]),
        "base_commit": str(config["protocol"]["base_commit"]),
        "frozen_definition_commit": definition_commit,
        "confirmatory_designation": str(
            config["protocol"]["confirmatory_designation"]
        ),
        "frozen_protocol": _relative(root, frozen_protocol),
        "frozen_config": _relative(root, frozen_config),
    }
    git_provenance = {
        **common,
        "head": head,
        "branch": branch,
        "frozen_definition_commit": definition_commit,
        "frozen_code_snapshot": code_snapshot(config_file, protocol_file, root),
        "working_tree_clean_before_freeze": before_status == "",
        "working_tree_status_before_freeze": before_status,
    }
    environment = {**common, **collect_environment()}
    _write_once_json(output_root / "manifest.json", manifest)
    _write_once_json(output_root / "git_provenance.json", git_provenance)
    _write_once_json(output_root / "environment.json", environment)
    return {
        **common,
        "output_root": str(output_root),
        "frozen_protocol": str(frozen_protocol),
        "frozen_config": str(frozen_config),
        "head": head,
        "branch": branch,
        "frozen_definition_commit": definition_commit,
    }


def validate_frozen_protocol_outputs(
    config_path: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    """Require byte-identical frozen definitions and root provenance outputs."""

    root = Path(repo_root).resolve()
    config, config_file, protocol_file, config_hash, protocol_hash = load_protocol_config(
        config_path, root
    )
    output_root = _resolve_inside(
        root, config["project"]["output_dir"], label="project.output_dir"
    )
    protocol_dir = output_root / "protocol"
    frozen_protocol = protocol_dir / protocol_file.name
    frozen_config = protocol_dir / "frozen_config.yaml"
    required_files = {
        "manifest": output_root / "manifest.json",
        "git_provenance": output_root / "git_provenance.json",
        "environment": output_root / "environment.json",
        "frozen_protocol": frozen_protocol,
        "frozen_config": frozen_config,
    }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise ConditionalProvenanceError(
            f"Protocol freeze outputs are missing: {sorted(missing)}"
        )
    if frozen_protocol.read_bytes() != protocol_file.read_bytes():
        raise ConditionalProvenanceError("Frozen protocol copy differs from live protocol")
    if frozen_config.read_bytes() != config_file.read_bytes():
        raise ConditionalProvenanceError("Frozen config copy differs from live config")
    try:
        manifest = json.loads(required_files["manifest"].read_text(encoding="utf-8"))
        git_provenance = json.loads(
            required_files["git_provenance"].read_text(encoding="utf-8")
        )
        environment = json.loads(required_files["environment"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConditionalProvenanceError("Invalid root protocol provenance JSON") from error
    definition_commit = _first_definition_commit(config_file, protocol_file, root)
    common = {
        "protocol_name": str(config["protocol"]["name"]),
        "protocol_version": str(config["protocol"]["version"]),
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
    }
    for name, payload in (
        ("manifest", manifest),
        ("git_provenance", git_provenance),
        ("environment", environment),
    ):
        if not isinstance(payload, dict):
            raise ConditionalProvenanceError(f"Root {name} must be a JSON object")
        for key, expected in common.items():
            if payload.get(key) != expected:
                raise ConditionalProvenanceError(f"Root {name} mismatch at {key}")
    expected_manifest = {
        "schema_version": "1.0",
        "phase": "PROTOCOL_FROZEN",
        "branch": str(config["protocol"]["branch"]),
        "base_commit": str(config["protocol"]["base_commit"]),
        "confirmatory_designation": str(config["protocol"]["confirmatory_designation"]),
        "frozen_definition_commit": definition_commit,
        "frozen_protocol": _relative(root, frozen_protocol),
        "frozen_config": _relative(root, frozen_config),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ConditionalProvenanceError(f"Root manifest mismatch at {key}")
    if (
        git_provenance.get("branch") != str(config["protocol"]["branch"])
        or git_provenance.get("frozen_definition_commit") != definition_commit
        or git_provenance.get("working_tree_clean_before_freeze") is not True
        or git_provenance.get("working_tree_status_before_freeze") != ""
    ):
        raise ConditionalProvenanceError("Root git provenance does not describe a clean freeze")
    if git_branch(root) != str(config["protocol"]["branch"]):
        raise ConditionalProvenanceError("Current branch differs from frozen branch")
    observed_code_snapshot = code_snapshot(config_file, protocol_file, root)
    if git_provenance.get("frozen_code_snapshot") != observed_code_snapshot:
        raise ConditionalProvenanceError(
            "Current source/config/protocol bytes differ from frozen code snapshot"
        )
    code_surface = [
        "src",
        "scripts",
        _relative(root, config_file),
        _relative(root, protocol_file),
    ]
    if (root / "requirements.txt").is_file():
        code_surface.append("requirements.txt")
    code_status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *code_surface,
        allow_empty=True,
    )
    if code_status:
        raise ConditionalProvenanceError(
            "Current code surface is dirty relative to the frozen implementation:\n"
            + code_status
        )
    current_environment = collect_environment()
    for section in ("python", "packages"):
        if environment.get(section) != current_environment.get(section):
            raise ConditionalProvenanceError(
                f"Runtime environment differs from frozen {section}"
            )
    return {
        **common,
        "status": "PASS",
        "frozen_definition_commit": definition_commit,
        "freeze_head": str(git_provenance.get("head")),
        "files": {
            name: {"path": _relative(root, path), "sha256": sha256_file(path)}
            for name, path in required_files.items()
        },
    }


def _default_unlock_path(config: Mapping[str, Any], root: Path) -> Path:
    # The unlock file is provenance, not a raw/confirmatory data input.
    return _resolve_inside(
        root,
        config["confirmatory_inputs"]["unlock_filename"],
        label="confirmatory unlock manifest",
    )


def create_confirmatory_unlock(
    config_path: str | Path,
    repo_root: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Lock a committed, clean discovery snapshot and create the unlock file.

    No value under ``confirmatory_inputs.raw_files[*].path`` is resolved,
    stat'ed, hashed, or opened by this function.
    """

    root = Path(repo_root).resolve()
    config, config_file, protocol_file, config_hash, protocol_hash = load_protocol_config(
        config_path, root
    )
    status = git_status_porcelain(root)
    if status:
        raise ConfirmatoryLockError(
            "Confirmatory unlock requires a clean committed working tree; observed:\n"
            + status
        )
    frozen_outputs = validate_frozen_protocol_outputs(config_file, root)
    branch = git_branch(root)
    expected_branch = str(config["protocol"]["branch"])
    if branch != expected_branch:
        raise ConfirmatoryLockError(
            f"Unlock branch mismatch: expected {expected_branch}, observed {branch}"
        )
    locked_head = git_head(root)
    discovery = discovery_snapshot(config, root, require_tracked=True)
    code = code_snapshot(config_file, protocol_file, root)
    code_commit = _latest_code_commit(config_file, protocol_file, root)
    scientific_contract: dict[str, Any] | None = None
    if isinstance(config.get("outputs"), Mapping):
        from src.conditional_pipeline_v1 import validate_discovery_snapshot_contract

        scientific_contract = validate_discovery_snapshot_contract(
            config,
            root,
            config_sha256=config_hash,
            code_commit=code_commit,
        )

    manifest: dict[str, Any] = {
        "schema_version": UNLOCK_SCHEMA_VERSION,
        "status": UNLOCK_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_designation": str(
            config["protocol"]["confirmatory_designation"]
        ),
        "prior_non_anatomy_session1_access": bool(
            config["protocol"]["prior_non_anatomy_session1_access"]
        ),
        "prior_session1_conditional_object_analysis": bool(
            config["protocol"]["prior_session1_conditional_object_analysis"]
        ),
        "discovery_session": str(config["dataset"]["discovery_session"]),
        "confirmatory_session": str(config["dataset"]["confirmatory_session"]),
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "code_commit": code_commit,
        "locked_head": locked_head,
        "branch": branch,
        "working_tree": {"clean": True, "porcelain": ""},
        "code_snapshot": code,
        "discovery_snapshot": discovery,
        "frozen_protocol_outputs": frozen_outputs,
        "confirmatory_raw_ordered_manifest_sha256": str(
            config["confirmatory_inputs"]["ordered_manifest_sha256"]
        ),
    }
    if scientific_contract is not None:
        manifest["discovery_scientific_contract"] = scientific_contract
    manifest["manifest_sha256"] = payload_sha256(manifest)

    if output_path is None:
        destination = _default_unlock_path(config, root)
    else:
        destination = _resolve_inside(root, output_path, label="unlock output")
    if destination.exists():
        raise ConfirmatoryLockError(
            f"Refusing to overwrite an existing unlock manifest: {destination}"
        )
    atomic_write_json(destination, manifest)
    return manifest


def _verify_manifest_self_hash(manifest: Mapping[str, Any]) -> None:
    expected = str(manifest.get("manifest_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    observed = payload_sha256(unsigned)
    if expected != observed:
        raise ConfirmatoryLockError(
            f"Unlock manifest payload hash mismatch: expected {expected}, observed {observed}"
        )


def _status_path(line: str) -> str:
    value = line[3:] if len(line) >= 4 else ""
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value


def _validate_post_unlock_status(
    status: str, config: Mapping[str, Any], root: Path, unlock_path: Path
) -> None:
    if not status:
        return
    output_root = _resolve_inside(
        root, config["project"]["output_dir"], label="project.output_dir"
    )
    allowed_exact = {_relative(root, unlock_path)}
    allowed_prefixes = [
        _relative(root, output_root / str(relative)).rstrip("/") + "/"
        for relative in config["project"].get("confirmatory_snapshot_dirs", [])
    ]
    forbidden: list[str] = []
    for line in status.splitlines():
        path = _status_path(line)
        if path in allowed_exact or any(path.startswith(prefix) for prefix in allowed_prefixes):
            continue
        forbidden.append(line)
    if forbidden:
        raise ConfirmatoryLockError(
            "Post-unlock working tree contains unauthorized changes:\n"
            + "\n".join(forbidden)
        )


def validate_confirmatory_unlock(
    config_path: str | Path,
    repo_root: str | Path,
    *,
    unlock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable discovery lock without touching 1test inputs."""

    root = Path(repo_root).resolve()
    config, config_file, protocol_file, config_hash, protocol_hash = load_protocol_config(
        config_path, root
    )
    try:
        frozen_outputs = validate_frozen_protocol_outputs(config_file, root)
    except ConditionalProvenanceError as error:
        raise ConfirmatoryLockError(
            f"Source/config/protocol freeze validation failed: {error}"
        ) from error
    destination = (
        _default_unlock_path(config, root)
        if unlock_path is None
        else _resolve_inside(root, unlock_path, label="unlock manifest")
    )
    try:
        manifest = json.loads(destination.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfirmatoryLockError(
            f"Confirmatory session is locked; manifest is missing: {destination}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ConfirmatoryLockError(f"Invalid unlock manifest: {destination}") from error
    if not isinstance(manifest, dict):
        raise ConfirmatoryLockError("Unlock manifest must be a JSON object")
    _verify_manifest_self_hash(manifest)

    expected_scalars = {
        "schema_version": UNLOCK_SCHEMA_VERSION,
        "status": UNLOCK_STATUS,
        "confirmatory_designation": config["protocol"]["confirmatory_designation"],
        "prior_non_anatomy_session1_access": bool(
            config["protocol"]["prior_non_anatomy_session1_access"]
        ),
        "prior_session1_conditional_object_analysis": bool(
            config["protocol"]["prior_session1_conditional_object_analysis"]
        ),
        "discovery_session": config["dataset"]["discovery_session"],
        "confirmatory_session": config["dataset"]["confirmatory_session"],
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "branch": config["protocol"]["branch"],
        "confirmatory_raw_ordered_manifest_sha256": config["confirmatory_inputs"][
            "ordered_manifest_sha256"
        ],
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ConfirmatoryLockError(
                f"Unlock field {key!r} mismatch: expected {expected!r}, "
                f"observed {manifest.get(key)!r}"
            )
    if manifest.get("working_tree") != {"clean": True, "porcelain": ""}:
        raise ConfirmatoryLockError("Unlock was not created from a clean working tree")
    if git_head(root) != manifest.get("locked_head"):
        raise ConfirmatoryLockError("HEAD changed after confirmatory unlock")
    if git_branch(root) != manifest.get("branch"):
        raise ConfirmatoryLockError("Branch changed after confirmatory unlock")
    if _latest_code_commit(config_file, protocol_file, root) != manifest.get(
        "code_commit"
    ):
        raise ConfirmatoryLockError("Code commit differs from the locked code commit")

    observed_code = code_snapshot(config_file, protocol_file, root)
    if observed_code != manifest.get("code_snapshot"):
        raise ConfirmatoryLockError("Source/config/protocol snapshot changed after unlock")
    observed_discovery = discovery_snapshot(config, root, require_tracked=True)
    if observed_discovery != manifest.get("discovery_snapshot"):
        raise ConfirmatoryLockError("Locked discovery snapshot changed after unlock")
    if manifest.get("frozen_protocol_outputs") != frozen_outputs:
        raise ConfirmatoryLockError("Frozen protocol provenance changed after unlock")
    if isinstance(config.get("outputs"), Mapping):
        from src.conditional_pipeline_v1 import validate_discovery_snapshot_contract

        scientific_contract = validate_discovery_snapshot_contract(
            config,
            root,
            config_sha256=config_hash,
            code_commit=str(manifest.get("code_commit")),
        )
        if manifest.get("discovery_scientific_contract") != scientific_contract:
            raise ConfirmatoryLockError(
                "Discovery scientific contract changed after unlock"
            )
    _validate_post_unlock_status(git_status_porcelain(root), config, root, destination)
    return manifest


__all__ = [
    "ConditionalProvenanceError",
    "ConfirmatoryLockError",
    "UNLOCK_SCHEMA_VERSION",
    "UNLOCK_STATUS",
    "atomic_write_json",
    "canonical_json_bytes",
    "code_snapshot",
    "collect_environment",
    "create_confirmatory_unlock",
    "discovery_snapshot",
    "freeze_conditional_protocol",
    "git_branch",
    "git_head",
    "git_status_porcelain",
    "load_protocol_config",
    "payload_sha256",
    "sha256_file",
    "validate_frozen_protocol_outputs",
    "validate_confirmatory_unlock",
]
