from __future__ import annotations

from pathlib import Path

import yaml

from src.trajectory_within_subject_v1 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bnci2014_001_local_temporal_sequence_v0.yaml"
PROTOCOL = ROOT / "docs" / "PROTOCOL_LOCAL_TEMPORAL_SEQUENCE_V0.md"


def test_protocol_hash_and_frozen_temporal_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert sha256_file(PROTOCOL) == config["protocol"]["sha256"]
    assert config["protocol"]["branch"] == "pilot/local-temporal-sequence-correspondence-v0"
    assert config["lineage"] == {
        "local_metric_final_sha": "796f04e7970972175a660a521caff47c83e0295f",
        "local_gpa_v1_final_sha": "122eacff868aa8f656ad6360716c1816f453979f",
        "gpa_outer_audit_final_sha": "347d61f17793d636653b614ec2104baa61ac7a4b",
        "branch_parent_sha": "347d61f17793d636653b614ec2104baa61ac7a4b",
    }
    assert config["mean_sequences"]["local_centering"] is False
    assert config["mean_sequences"]["ordered_sequence_count"] == 72
    assert config["split_half"]["half_A_runs"] == [0, 2, 4]
    assert config["split_half"]["half_B_runs"] == [1, 3, 5]
    assert config["matching"]["all_permutations"] == 120
    assert config["matching"]["derangements"] == 44
    assert config["nulls"]["replicates"] == 1999
    assert config["nulls"]["subjectbreak"]["stream_tag"] == 1102
    assert config["nulls"]["classbreak"]["stream_tag"] == 1101
    assert config["visualization"]["pca_basis"] == "one_global_sklearn_full_SVD_basis"
    for path in config["project"]["implementation_source_files"]:
        assert (ROOT / path).is_file()


def test_protocol_contains_required_boundaries_and_terminals() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for phrase in (
        "No trial-local centering",
        "exactly the 44 permutations",
        "1,999 draws",
        "subject×class interaction claim requires",
        "one global AIRM reference mean",
        "GO_REPRODUCIBLE_SUBJECT_CLASS_TEMPORAL_SEQUENCE",
        "STOP_NO_REPRODUCIBLE_TEMPORAL_SEQUENCE_V0",
        "UNASSESSED_NUMERICAL_FAILURE",
        "continuous physiological trajectory",
        "not a rerun of the earlier order-shuffle classifier",
    ):
        assert phrase in text

