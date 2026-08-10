"""Final artifact-contract tests for trajectory within-subject audit v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting_trajectory_within_subject_v1 import FIGURE_STEMS
from src.trajectory_within_subject_v1 import load_frozen_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bnci2014_001_trajectory_within_subject_v1.yaml"
OUTPUT = ROOT / "outputs" / "bnci2014_001_trajectory_within_subject_v1"


def test_required_output_contract_and_plain_language_report() -> None:
    config = load_frozen_config(CONFIG)
    assert tuple(config["outputs"]["figures"]) == FIGURE_STEMS
    for table in config["outputs"]["tables"]:
        assert (OUTPUT / "tables" / table).is_file()
    for artifact in config["outputs"]["nulls"]:
        assert (OUTPUT / "nulls" / artifact).is_file()
    for artifact in config["outputs"]["decisions"]:
        assert (OUTPUT / "decisions" / artifact).is_file()
    for stem in FIGURE_STEMS:
        for suffix in (".png", ".pdf", ".csv"):
            path = OUTPUT / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 0
    report = (OUTPUT / config["outputs"]["report"]).read_text(encoding="utf-8")
    assert report.startswith("# BNCI2014_001 Trajectory Within-Subject Audit v1")
    assert "v0 asked whether class trajectory structure transferred across different subjects" in report
    assert "GO_STABLE_SUBJECT_SPECIFIC_LOCAL_GEOMETRY" in report
    assert "chronological order as necessary" in report
    for forbidden in (
        "causal neural dynamics is established",
        "individual motor strategy was discovered",
        "personalized model improves",
    ):
        assert forbidden not in report


def test_nulls_are_complete_and_frozen_decision_replays() -> None:
    with np.load(OUTPUT / "nulls" / "label_null_statistics.npz", allow_pickle=False) as archive:
        assert archive["completed"].shape == (1999,)
        assert archive["completed"].all()
        assert archive["path_w_group"].shape == (1999,)
        assert archive["path_x_group"].shape == (1999,)
        assert archive["bag_w_group"].shape == (1999,)
        assert archive["bag_x_group"].shape == (1999,)
    with np.load(OUTPUT / "nulls" / "order_null_statistics.npz", allow_pickle=False) as archive:
        assert archive["completed"].shape == (1999,)
        assert archive["completed"].all()
        assert archive["path_x_group"].shape == (1999,)
    label = pd.read_csv(OUTPUT / "tables" / "label_null_summary.csv")
    order = pd.read_csv(OUTPUT / "tables" / "order_null_summary.csv")
    assert len(label) == 4 and label.inferential_pass.all()
    assert len(order) == 1 and not bool(order.iloc[0].passed)
    assert float(order.iloc[0].p_value) == 0.072
    decision = json.loads(
        (OUTPUT / "decisions" / "terminal_decision.json").read_text(encoding="utf-8")
    )
    assert decision["stage_w_pass"] is True
    assert decision["stage_x_pass"] is True
    assert decision["stage_o_pass"] is False
    assert decision["decision"] == "GO_STABLE_SUBJECT_SPECIFIC_LOCAL_GEOMETRY"
    assert decision["whole_subject_class_interaction_used"] is False


def test_reproduction_gate_is_exact_and_observed_grids_have_no_failed_fit() -> None:
    reproduction = pd.read_csv(OUTPUT / "tables" / "reproduction_gate.csv")
    assert reproduction.passed.all()
    for check in (
        "airm_path_d10_machine_precision",
        "airm_bag_canon_d10_machine_precision",
        "airm_scalars_11_machine_precision",
    ):
        row = reproduction[reproduction.check.eq(check)].iloc[0]
        assert float(row.maximum_absolute_difference) == 0.0
        assert float(row.tolerance) == 1e-12
    within = pd.read_csv(OUTPUT / "tables" / "within_session_scores.csv")
    cross = pd.read_csv(OUTPUT / "tables" / "cross_session_scores.csv")
    assert len(within) == 108 and len(cross) == 54
    assert (within.status == "PASS").all() and (cross.status == "PASS").all()
    assert not within.balanced_accuracy.isna().any()
    assert not cross.balanced_accuracy.isna().any()
