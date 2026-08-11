from __future__ import annotations

from pathlib import Path

from src.local_metric_data_v0 import (
    EXPECTED_TRIALS,
    FROZEN_DISTANCE_SHA256,
    FROZEN_PATH_SHA256,
    load_and_reproduce_local_metric_input,
)
from src.trajectory_within_subject_v1 import sha256_array


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_trajectory_input_reproduces_exactly() -> None:
    data = load_and_reproduce_local_metric_input(ROOT)
    assert data.edges.shape == (EXPECTED_TRIALS, 10)
    assert data.distance_matrices.shape == (EXPECTED_TRIALS, 5, 5)
    assert sha256_array(data.edges) == FROZEN_PATH_SHA256
    assert sha256_array(data.distance_matrices) == FROZEN_DISTANCE_SHA256
    assert data.reproduction_table["passed"].astype(bool).all()
    assert data.provenance["upper_triangle_path_max_abs_diff"] == 0.0
    assert data.degeneracy_table.empty
