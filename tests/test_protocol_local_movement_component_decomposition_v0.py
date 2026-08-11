from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import yaml

from src.reporting_local_movement_component_decomposition_v0 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bnci2014_001_local_movement_component_decomposition_v0.yaml"
PROTOCOL = ROOT / "docs" / "PROTOCOL_LOCAL_MOVEMENT_COMPONENT_DECOMPOSITION_V0.md"
RUNNER = ROOT / "scripts" / "27_run_local_movement_component_decomposition_v0.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("component_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_hash_lineage_branch_and_namespace_are_frozen() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"]["sha256"] == sha256_file(PROTOCOL)
    assert config["protocol"]["branch"] == "pilot/local-movement-component-decomposition-v0"
    assert config["lineage"]["parent_sha"] == "12c19f38266bc76875cffae056e7f9403df299c1"
    assert config["lineage"]["movement_protocol_freeze_sha"] == (
        "e24312147ef3020854ef6f6cd174071d1c6ead02"
    )
    assert config["lineage"]["movement_scientific_result_sha"] == (
        "c3f1d5ff9cf23db2007bbf839cf4b266e2cb8960"
    )
    assert config["project"]["output_dir"] == (
        "outputs/bnci2014_001_local_movement_component_decomposition_v0"
    )


def test_protocol_freezes_squared_costs_nulls_split_rule_and_claims() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for phrase in (
        "`c_ang=c_full-c_len`",
        "`c_ori=c_sensor-c_full`",
        "exactly 1,999 draws",
        "SeedSequence([20260810,1102])",
        "SeedSequence([20260810,1101])",
        "Replicate A compares",
        "split_half_ang_sign_stable",
        "No rescue normalization",
        "neural, physiological, or source-space direction",
    ):
        assert phrase in text


def test_prepare_cannot_compute_or_view_real_component_statistics() -> None:
    source = inspect.getsource(_runner_module().prepare)
    assert "reproduce_frozen_inputs" in source
    assert "build_component_matrices" not in source
    assert "relation_statistics" not in source
    assert "evaluate_all_components" not in source
    assert "_split_half_costs" not in source


def test_parent_artifacts_are_read_only_and_current_hashes_are_exact() -> None:
    runner = _runner_module()
    before = {
        relative: sha256_file(ROOT / relative)
        for relative in runner.FROZEN_ARTIFACT_HASHES
    }
    reproduction = runner.reproduce_frozen_inputs()
    after = {
        relative: sha256_file(ROOT / relative)
        for relative in runner.FROZEN_ARTIFACT_HASHES
    }
    assert reproduction["status"] == "PASS"
    assert reproduction["new_component_statistic_computed"] is False
    assert before == after == runner.FROZEN_ARTIFACT_HASHES


def test_runner_writes_only_new_output_namespace() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'OUTPUT_ROOT = ROOT / "outputs" / "bnci2014_001_local_movement_component_decomposition_v0"' in source
    assert "unlink(" not in source
    assert "rmtree(" not in source
    assert "git reset" not in source
    assert "git stash" not in source


def test_all_required_terminals_are_frozen() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for terminal in (
        "BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS",
        "GO_DIRECTIONAL_JOINT_MOVEMENT_INTERACTION",
        "SENSOR_FRAME_ORIENTATION_ONLY_SUPPORT",
        "SPEED_PROFILE_SUFFICIENT_AT_CURRENT_RESOLUTION",
        "NO_COMPONENT_INTERACTION_AT_SQUARED_COST_RESOLUTION",
        "UNASSESSED_COMPONENT_DECOMPOSITION_UNRELIABLE",
        "UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE",
        "UNASSESSED_COMPONENT_DECOMPOSITION_TECHNICAL_FAILURE",
    ):
        assert terminal in text
