from __future__ import annotations

from pathlib import Path

import yaml

from src.local_gpa_geometry_v0 import CANDIDATE_SETTINGS, GPASettings
from src.trajectory_within_subject_v1 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "PROTOCOL_LOCAL_GPA_CONSENSUS_V0.md"
CONFIG = ROOT / "configs" / "bnci2014_001_local_gpa_consensus_v0.yaml"


def test_protocol_hash_and_runtime_settings_are_frozen_exactly() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"]["sha256"] == sha256_file(PROTOCOL)
    assert GPASettings(**config["gpa"]["settings"]) == CANDIDATE_SETTINGS
    assert config["gpa"]["settings"]["registration_total_starts"] == 4
    assert config["gpa"]["settings"]["gpa_total_starts"] == 2
    assert config["split_half"]["half_A_runs"] == [0, 2, 4]
    assert config["split_half"]["half_B_runs"] == [1, 3, 5]
    assert config["nulls"]["replicates"] == 1999


def test_protocol_forbids_pose_interpretation_and_stage1_rescue() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "never averaged, compared across cells, or interpreted" in text
    assert "not a rescue of Stage 1" in text
    assert "STOP_NO_STABLE_LOCAL_METRIC_INTERACTION_V0" in text
