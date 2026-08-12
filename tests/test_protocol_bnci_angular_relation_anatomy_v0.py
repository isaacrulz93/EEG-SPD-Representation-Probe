from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bnci2014_001_angular_relation_anatomy_v0.yaml"
PROTOCOL = ROOT / "docs/PROTOCOL_BNCI_ANGULAR_RELATION_ANATOMY_V0.md"
SOURCE = ROOT / "src/bnci_angular_relation_anatomy_v0.py"


def test_protocol_and_config_freeze_exact_parent() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["lineage"]["parent_head"] == "edc1d344cb0657f2f2d87b2992049bceec4705d2"
    assert config["frozen_input"]["sha256"] == "51af2be73930a8ad77e617dd1b473b0249423c74d030ea2966d489603a250091"
    assert config["formal_decomposition"]["aggregation"] == "mean_classes_within_subject_then_mean_subjects"
    assert config["formal_decomposition"]["pairwise_pvalues"] == "not_computed"


def test_protocol_forbids_upstream_refitting_and_intrinsic_shape_claim() -> None:
    protocol = PROTOCOL.read_text()
    for phrase in (
        "No EEG, covariance mean, anti-development, ordered movement object, or",
        "not asserted to be intrinsic manifold shapes",
        "independent population units",
        "No median",
    ):
        assert phrase in protocol


def test_analysis_module_has_no_upstream_fitting_dependencies() -> None:
    source = SOURCE.read_text()
    forbidden_imports = (
        "import mne",
        "from mne",
        "import pyriemann",
        "from pyriemann",
        "local_mean_movement_v0",
        "movement_distance(",
    )
    for token in forbidden_imports:
        assert token not in source
