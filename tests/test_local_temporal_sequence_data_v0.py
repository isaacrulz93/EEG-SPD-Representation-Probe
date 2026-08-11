from __future__ import annotations

from pathlib import Path

import pytest

from src.local_gpa_data_v0 import (
    COMBINED_FEATURE_CACHE,
    SESSION0_CACHE,
    SESSION1_CACHE,
    load_and_reproduce_local_gpa_input,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not all(
        (ROOT / path).exists()
        for path in (SESSION0_CACHE, SESSION1_CACHE, COMBINED_FEATURE_CACHE)
    ),
    reason="large frozen covariance caches are not installed",
)
def test_temporal_input_is_original_uncentered_window5_and_reproduces() -> None:
    data = load_and_reproduce_local_gpa_input(ROOT)
    assert data.states.shape == (5184, 5, 22, 22)
    assert data.reproduction_table["passed"].astype(bool).all()
    row = data.reproduction_table.loc[
        data.reproduction_table["check"].eq(
            "window5_states_reproduce_frozen_AIRM_distance_matrices"
        )
    ].iloc[0]
    assert row["maximum_absolute_difference"] == 0.0
    assert data.provenance["session0_states_sha256"] == (
        "c75044f48552f12ad088306b505b074e930f396fdcb544307fff394717e2ca86"
    )
    assert data.provenance["session1_states_sha256"] == (
        "1afc8cd52d82310a05857d1ffa67859427c4c9aa1302897a140ebda64d0442f8"
    )
