from __future__ import annotations

from pathlib import Path

import yaml

from src.trajectory_within_subject_v1 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bnci2014_001_local_metric_interaction_v0.yaml"
PROTOCOL = ROOT / "docs" / "PROTOCOL_LOCAL_METRIC_INTERACTION_V0.md"


def test_protocol_content_hash_and_frozen_scientific_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert sha256_file(PROTOCOL) == config["protocol"]["sha256"]
    assert (
        config["protocol"]["scientific_base_sha"]
        == "355fe0b55b1ef692f7b4ddd16d19b7ccc30e72e1"
    )
    assert config["frozen_representation"]["windows"] == 5
    assert config["frozen_representation"]["edge_order"] == [
        "d12",
        "d13",
        "d14",
        "d15",
        "d23",
        "d24",
        "d25",
        "d34",
        "d35",
        "d45",
    ]
    assert config["quotient_metrics"]["enumeration_count"] == 120
    assert config["quotient_metrics"]["arbitrary_S10_forbidden"] is True
    assert config["nulls"]["replicates"] == 1999
    assert config["nulls"]["pass_rule"] == "T_J_gt_0_and_both_pvalues_lt_0.05"
    assert config["interaction"]["scientific_group_unit"] == "target_subject"
    assert config["mechanism_controls"]["changes_raw_terminal"] is False
    assert config["secondary_decoder"]["affects_primary_terminal"] is False
    for relative in config["project"]["implementation_source_files"]:
        assert (ROOT / relative).is_file()


def test_protocol_contains_required_claim_caveats_and_future_boundary() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "only a pseudometric" in text
    assert "No additive or causal decomposition" in text
    assert "subject is the scientific group unit" in text
    assert "strictly below 0.05" in text
    assert "future separately preregistered Procrustes-pose anatomy study" in text
    assert "No classifier tuning" in text
