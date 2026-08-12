from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import yaml

import src.openbmi_ordered_movement_v0 as replication


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "openbmi_ordered_movement_replication_v0.yaml"
PROTOCOL = ROOT / "docs" / "PROTOCOL_OPENBMI_ORDERED_MOVEMENT_REPLICATION_V0.md"
RUNNER = ROOT / "scripts" / "28_run_openbmi_ordered_movement_v0.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_hash_branch_lineage_and_namespace_are_frozen() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"]["sha256"] == _sha256(PROTOCOL)
    assert config["protocol"]["branch"] == "replication/openbmi-ordered-movement-v0"
    assert config["lineage"]["bnci_parent_head"] == "edc1d344cb0657f2f2d87b2992049bceec4705d2"
    assert config["lineage"]["openbmi_donor_head"] == "272d775678644aad062df424a70586d4b42de652"
    assert config["project"]["output_dir"] == "outputs/openbmi_ordered_movement_replication_v0"


def test_protocol_freezes_temporal_geometry_components_nulls_and_claims() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for phrase in (
        "five equal consecutive non-overlapping bins",
        "exactly 0.5 seconds",
        "reverse prefix",
        "`c_ang=c_full-c_len`",
        "`c_ori=c_sensor-c_full`",
        "SeedSequence([20260810,1102])",
        "SeedSequence([20260810,1101])",
        "all 44 complete derangements",
        "two-class structural replication",
        "No rescue analysis",
    ):
        assert phrase in text


def test_runner_has_atomic_resumable_contract_for_all_three_quotient_matrices() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for phrase in (
        'fit_quotient_matrix("full"',
        'fit_quotient_matrix(f"split_half_{half_name}"',
        '"completed": completed',
        '"protocol_sha256"',
        '"first_array_sha256"',
        '"second_array_sha256"',
        "atomic_savez(checkpoint",
    ):
        assert phrase in source


def test_data_preparation_cannot_compute_or_view_movement_statistics() -> None:
    source = inspect.getsource(__import__("runpy"))  # import is side-effect free sentinel
    runner_source = RUNNER.read_text(encoding="utf-8")
    body = runner_source.split("def prepare_data()", 1)[1].split("def prepare_provenance()", 1)[0]
    assert source
    assert "fit_mean_sequences" not in body
    assert "fit_quotient_matrix" not in body
    assert "relation_statistics" not in body


def test_original_worktree_is_never_a_write_target() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "/Volumes/External_SSD/isaac/EEG-SPD-Representation-Probe" not in source
    assert "git reset" not in source
    assert "git stash" not in source
    assert "rmtree(" not in source


def test_all_required_terminals_are_frozen() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for value in (
        "REPLICATED_OPENBMI_ANGULAR_JOINT_INTERACTION",
        "NOT_REPLICATED_OPENBMI_ANGULAR_JOINT_INTERACTION",
        "UNASSESSED_OPENBMI_MOVEMENT_NUMERICAL_FAILURE",
        "UNASSESSED_OPENBMI_DATA_CONTRACT_FAILURE",
        "STOP_AND_REPORT_TEMPORAL_CONTRACT_AMBIGUITY",
    ):
        assert value in text


def test_frozen_constants_match_openbmi_binary_contract() -> None:
    assert (replication.N_SUBJECTS, replication.N_SESSIONS, replication.N_CLASSES) == (54, 2, 2)
    assert replication.N_CELLS == 108
    assert replication.CHANNEL_ORDER == (
        "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    )
