from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bnci2014_001_lr_angular_factorial_v0.yaml"
PROTOCOL = ROOT / "docs" / "PROTOCOL_BNCI_LEFT_RIGHT_ANGULAR_FACTORIAL_V0.md"
RUNNER = ROOT / "scripts" / "29_run_bnci_lr_angular_factorial_v0.py"
MODULE = ROOT / "src" / "bnci_lr_angular_factorial_v0.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_hash_lineage_branch_and_output_are_frozen() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"]["sha256"] == sha256(PROTOCOL)
    assert config["protocol"]["branch"] == "audit/bnci-left-right-angular-factorial-v0"
    assert config["lineage"]["parent_head"] == "edc1d344cb0657f2f2d87b2992049bceec4705d2"
    assert config["lineage"]["parent_protocol_freeze"] == "95c330de9596fa4c4eb4ee377d5af8d99896f4c3"
    assert config["lineage"]["parent_scientific_result"] == "0dfa4ab4f94dd35c4d5ec8e74a5b51940083d3ca"
    assert config["project"]["output_dir"] == "outputs/bnci2014_001_lr_angular_factorial_v0"


def test_protocol_freezes_existing_contrasts_binary_nulls_and_terminals() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for phrase in (
        "`S_sc=c_sc-a_sc`",
        "`C_sc=b_sc-a_sc`",
        "`J_sc=b_sc+c_sc-a_sc-d_sc`",
        "alternative balanced factorial S/C definitions are forbidden",
        "SeedSequence([20260810,1102])",
        "SeedSequence([20260810,1101])",
        "Fixed points are allowed",
        "exact K=4 frozen-statistic regression",
        "BNCI_LR_ANGULAR_INTERACTION_SUPPORTED_AND_STABLE",
        "BNCI_LR_ANGULAR_INTERACTION_SUPPORTED_BUT_SPLIT_UNSTABLE",
        "BNCI_LR_ANGULAR_INTERACTION_NOT_SUPPORTED",
        "UNASSESSED_BNCI_LR_ANGULAR_DIAGNOSTIC_FAILURE",
    ):
        assert phrase in text


def test_implementation_has_no_refitting_or_raw_data_imports() -> None:
    source = (MODULE.read_text(encoding="utf-8") + RUNNER.read_text(encoding="utf-8")).lower()
    forbidden = (
        "mne.",
        "moabb",
        "covariances(",
        "airm_mean(",
        "anti_develop_sequence(",
        "movement_distance(",
        "optimize_movement_alignment(",
        "trustregions(",
    )
    for phrase in forbidden:
        assert phrase not in source


def test_prepare_stage_cannot_compute_new_left_right_statistics() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    body = source.split("def prepare()", 1)[1].split("def relation_table", 1)[0]
    assert "reproduce_parent_artifacts" in body
    assert "evaluate_inference" not in body
    assert "relation_statistics" not in body
    assert "extract_lr_matrix" not in body


def test_original_repository_is_not_a_write_target() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "/Volumes/External_SSD/isaac/EEG-SPD-Representation-Probe" not in source
    assert "git reset" not in source
    assert "git stash" not in source
    assert "rmtree(" not in source
