import json
from pathlib import Path

import numpy as np

from src.baseline_trajectory_v0.geometry import load_bank


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/bnci2014_001_baseline_trajectory_identifiability_v0"
CACHE = ROOT / "cache/bnci2014_001_baseline_trajectory_identifiability_v0/baseline_trajectory_covariances.npz"


def test_frozen_timing_channel_and_reproduction_gates():
    audit = json.loads((OUT / "data_timing_audit.json").read_text())
    assert audit["data_gate"] == "PASS"
    assert audit["sample_counts"] == {"baseline": 250, "local_windows": [200] * 5, "post": 1000}
    assert audit["baseline_post_overlap"] is False
    assert audit["run_boundary_safe"] is True
    assert audit["trial_ids_unique"] is True
    assert len(audit["channel_names"]) == 22
    assert audit["excluded_channels"] == ["EOG1", "EOG2", "EOG3", "STI"]
    assert audit["old_window5_reproduction"]["status"] == "PASS"


def test_covariance_bank_contract_and_class_balance():
    bank = load_bank(CACHE)
    assert bank.c0.shape == (5184, 22, 22)
    assert bank.local.shape == (5184, 5, 22, 22)
    assert bank.full.shape == (5184, 22, 22)
    counts = bank.metadata.groupby(["subject", "session", "run", "class_label"]).size()
    assert np.array_equal(counts.unique(), [12])
