"""Synthetic clean-HEAD and pre-unlock barrier tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

import src.data_conditional_v1 as conditional_data
from src.conditional_provenance_v1 import (
    ConditionalProvenanceError,
    ConfirmatoryLockError,
    create_confirmatory_unlock,
    freeze_conditional_protocol,
    validate_frozen_protocol_outputs,
    validate_confirmatory_unlock,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _make_lock_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-b", "pilot/conditional-geometry-anatomy-v1")
    _git(root, "config", "user.email", "conditional-test@example.invalid")
    _git(root, "config", "user.name", "Conditional Test")

    protocol = root / "docs/protocol.md"
    protocol.parent.mkdir()
    protocol.write_text("# synthetic frozen protocol\n", encoding="utf-8")
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    output = root / "outputs/study"
    for relative in ("objects/discovery", "tables/discovery", "nulls/discovery"):
        directory = output / relative
        directory.mkdir(parents=True)
        (directory / "snapshot.txt").write_text(relative + "\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src/core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts/run.py").write_text("print('synthetic')\n", encoding="utf-8")

    config = {
        "protocol": {
            "name": "Synthetic Conditional Geometry",
            "version": "1.0",
            "base_commit": "synthetic-base",
            "branch": "pilot/conditional-geometry-anatomy-v1",
            "protocol_path": "docs/protocol.md",
            "protocol_sha256": protocol_hash,
            "confirmatory_designation": "STRICT_CONFIRMATORY",
            "prior_non_anatomy_session1_access": True,
            "prior_session1_conditional_object_analysis": False,
        },
        "project": {
            "output_dir": "outputs/study",
            "discovery_snapshot_dirs": [
                "objects/discovery",
                "tables/discovery",
                "nulls/discovery",
            ],
            "confirmatory_snapshot_dirs": [
                "objects/confirmatory",
                "tables/confirmatory",
                "nulls/confirmatory",
            ],
        },
        "dataset": {
            "discovery_session": "0train",
            "confirmatory_session": "1test",
        },
        "confirmatory_inputs": {
            "ordered_manifest_sha256": "a" * 64,
            "unlock_filename": "outputs/study/confirmatory_unlock.json",
        },
    }
    config_path = root / "configs/conditional.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "synthetic discovery lock input")
    return root, config_path


def _freeze_and_commit(root: Path, config: Path) -> None:
    freeze_conditional_protocol(config, root)
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "freeze synthetic protocol outputs")


def test_clean_committed_snapshot_can_unlock_and_validate(tmp_path: Path) -> None:
    root, config = _make_lock_repo(tmp_path)
    _freeze_and_commit(root, config)
    manifest = create_confirmatory_unlock(config, root)
    assert manifest["status"] == "CONFIRMATORY_UNLOCKED"
    assert manifest["confirmatory_designation"] == "STRICT_CONFIRMATORY"
    assert manifest["discovery_snapshot"]["file_count"] == 3
    validated = validate_confirmatory_unlock(config, root)
    assert validated["manifest_sha256"] == manifest["manifest_sha256"]


def test_unlock_rejects_dirty_working_tree(tmp_path: Path) -> None:
    root, config = _make_lock_repo(tmp_path)
    _freeze_and_commit(root, config)
    (root / "src/core.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ConfirmatoryLockError, match="clean committed"):
        create_confirmatory_unlock(config, root)


def test_validation_rejects_changed_discovery_snapshot(tmp_path: Path) -> None:
    root, config = _make_lock_repo(tmp_path)
    _freeze_and_commit(root, config)
    create_confirmatory_unlock(config, root)
    snapshot = root / "outputs/study/tables/discovery/snapshot.txt"
    snapshot.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ConfirmatoryLockError, match="discovery snapshot"):
        validate_confirmatory_unlock(config, root)


def test_validation_rejects_source_change(tmp_path: Path) -> None:
    root, config = _make_lock_repo(tmp_path)
    _freeze_and_commit(root, config)
    create_confirmatory_unlock(config, root)
    (root / "src/core.py").write_text("VALUE = 99\n", encoding="utf-8")
    with pytest.raises(ConfirmatoryLockError, match="Source/config/protocol"):
        validate_confirmatory_unlock(config, root)


def test_unlock_requires_every_configured_discovery_artifact(tmp_path: Path) -> None:
    root, config_path = _make_lock_repo(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["outputs"] = {
        "required_objects": ["required_object.npz"],
        "required_nulls": ["required_null.npz"],
        "required_tables": [
            "required_table.csv",
            "discovery_confirmatory_comparison.csv",
        ],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _git(root, "add", "configs/conditional.yaml")
    _git(root, "commit", "-m", "declare required discovery artifacts")
    _freeze_and_commit(root, config_path)
    with pytest.raises(ConditionalProvenanceError, match="Required discovery artifacts"):
        create_confirmatory_unlock(config_path, root)


def test_freeze_protocol_is_idempotent(tmp_path: Path) -> None:
    root, config = _make_lock_repo(tmp_path)
    first = freeze_conditional_protocol(config, root)
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "commit frozen protocol outputs")
    second = freeze_conditional_protocol(config, root)
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert (root / "outputs/study/protocol/protocol.md").read_bytes() == (
        root / "docs/protocol.md"
    ).read_bytes()
    assert (root / "outputs/study/protocol/frozen_config.yaml").read_bytes() == config.read_bytes()


def test_unlock_refuses_skipped_protocol_freeze(tmp_path: Path) -> None:
    root, config = _make_lock_repo(tmp_path)
    with pytest.raises(ConditionalProvenanceError, match="freeze outputs are missing"):
        create_confirmatory_unlock(config, root)


def test_protocol_freeze_refuses_dirty_tree(tmp_path: Path) -> None:
    root, config = _make_lock_repo(tmp_path)
    (root / "src/core.py").write_text("VALUE = 9\n", encoding="utf-8")
    with pytest.raises(ConditionalProvenanceError, match="clean working tree"):
        freeze_conditional_protocol(config, root)


def test_frozen_config_rejects_post_freeze_scientific_tamper(tmp_path: Path) -> None:
    root, config_path = _make_lock_repo(tmp_path)
    _freeze_and_commit(root, config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["scientific_test_seed"] = 999
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConditionalProvenanceError, match="Frozen config copy differs"):
        validate_frozen_protocol_outputs(config_path, root)


def test_frozen_environment_rejects_runtime_package_mismatch(tmp_path: Path) -> None:
    root, config_path = _make_lock_repo(tmp_path)
    _freeze_and_commit(root, config_path)
    environment_path = root / "outputs/study/environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["packages"]["numpy"] = "0.0.synthetic-mismatch"
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ConditionalProvenanceError, match="frozen packages"):
        validate_frozen_protocol_outputs(config_path, root)


def test_frozen_code_snapshot_rejects_uncommitted_source_change(tmp_path: Path) -> None:
    root, config_path = _make_lock_repo(tmp_path)
    _freeze_and_commit(root, config_path)
    (root / "src/core.py").write_text("VALUE = 404\n", encoding="utf-8")
    with pytest.raises(ConditionalProvenanceError, match="frozen code snapshot"):
        validate_frozen_protocol_outputs(config_path, root)


def test_confirmatory_entry_fails_before_raw_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def locked(*args: object, **kwargs: object) -> dict[str, object]:
        events.append("unlock")
        raise ConfirmatoryLockError("synthetic locked")

    def forbidden(*args: object, **kwargs: object) -> object:
        events.append("raw")
        raise AssertionError("confirmatory raw resolver was reached")

    monkeypatch.setattr(conditional_data, "validate_confirmatory_unlock", locked)
    monkeypatch.setattr(conditional_data, "_resolve_confirmatory_raw_inputs", forbidden)
    with pytest.raises(ConfirmatoryLockError, match="synthetic locked"):
        conditional_data.prepare_confirmatory_whole(
            ROOT / "configs/bnci2014_001_conditional_geometry_v1.yaml", ROOT
        )
    assert events == ["unlock"]


def test_discovery_public_loader_has_no_session_argument() -> None:
    import inspect

    parameters = inspect.signature(conditional_data.load_discovery_whole).parameters
    assert "session" not in parameters
    assert "confirmatory_session" not in parameters
