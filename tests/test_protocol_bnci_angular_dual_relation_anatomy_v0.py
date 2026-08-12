from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bnci2014_001_angular_dual_relation_anatomy_v0.yaml"
PROTOCOL = ROOT / "docs/PROTOCOL_BNCI_ANGULAR_DUAL_RELATION_ANATOMY_V0.md"
SOURCE = ROOT / "src/bnci_angular_dual_relation_anatomy_v0.py"
PARENT_ARRAY = ROOT / "outputs/bnci2014_001_local_movement_component_decomposition_v0/arrays/component_cost_matrices.npz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_angular_artifact_hash_is_exact() -> None:
    assert sha256(PARENT_ARRAY) == "51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091"


def test_config_freezes_lineage_mean_aggregation_and_six_pairs() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["lineage"]["parent_head"] == "edc1d344cb0657f2f2d87b2992049bceec4705d2"
    assert config["formal_decomposition"]["aggregation"] == "class_mean_within_subject_then_subject_mean"
    assert config["formal_decomposition"]["pairs"] == ["LR", "LF", "LT", "RF", "RT", "FT"]
    assert config["formal_decomposition"]["pairwise_null_pvalues"] == "not_computed"


def test_protocol_is_retrospective_and_forbids_overclaims() -> None:
    text = PROTOCOL.read_text()
    for phrase in (
        "retrospective anatomy analysis",
        "Median aggregation",
        "not intrinsic metric geometry",
        "not proof of a reusable `Q_s`",
        "Trial pairs and matrix",
    ):
        assert phrase in text


def test_analysis_source_has_no_upstream_fitting_imports_or_calls() -> None:
    text = SOURCE.read_text()
    for forbidden in (
        "import mne",
        "from mne",
        "import pyriemann",
        "from pyriemann",
        "movement_distance(",
        "mean_covariance(",
        "optimizer.run(",
    ):
        assert forbidden not in text
