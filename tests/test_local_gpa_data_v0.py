from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.local_gpa_data_v0 import (
    COMBINED_FEATURE_CACHE,
    SESSION0_CACHE,
    SESSION1_CACHE,
    load_and_reproduce_local_gpa_input,
)
from src.local_gpa_pipeline_v0 import HALF_RUNS, cell_tasks


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not all((ROOT / path).exists() for path in (SESSION0_CACHE, SESSION1_CACHE, COMBINED_FEATURE_CACHE)),
    reason="large frozen external covariance caches are not installed",
)
def test_frozen_window5_covariance_input_reproduces_exactly() -> None:
    data = load_and_reproduce_local_gpa_input(ROOT)
    assert data.states.shape == (5184, 5, 22, 22)
    assert data.reproduction_table["passed"].astype(bool).all()
    distance_row = data.reproduction_table.loc[
        data.reproduction_table["check"].eq(
            "window5_states_reproduce_frozen_AIRM_distance_matrices"
        )
    ].iloc[0]
    assert distance_row["maximum_absolute_difference"] == 0.0


@pytest.mark.skipif(
    not all((ROOT / path).exists() for path in (SESSION0_CACHE, SESSION1_CACHE, COMBINED_FEATURE_CACHE)),
    reason="large frozen external covariance caches are not installed",
)
def test_run_blocked_halves_are_disjoint_exhaustive_and_balanced() -> None:
    data = load_and_reproduce_local_gpa_input(ROOT)
    tasks = cell_tasks(data.metadata)
    assert len(tasks) == 216
    grouped = {}
    for task in tasks:
        grouped.setdefault((task.subject, task.session, task.class_label), {})[
            task.split
        ] = task
    assert len(grouped) == 72
    for values in grouped.values():
        full = set(values["Full"].indices.tolist())
        half_a = set(values["A"].indices.tolist())
        half_b = set(values["B"].indices.tolist())
        assert half_a.isdisjoint(half_b)
        assert half_a | half_b == full
        assert len(half_a) == len(half_b) == 36
    assert HALF_RUNS["A"] == (0, 2, 4)
    assert HALF_RUNS["B"] == (1, 3, 5)
