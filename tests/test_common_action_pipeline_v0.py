from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common_action_pipeline_v0 import (
    FROZEN_PROTOCOL_SHA,
    deterministic_seed,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_execution_config_matches_corrected_freeze() -> None:
    config = load_config(ROOT)
    assert FROZEN_PROTOCOL_SHA == "9a4dd836fd24c47ffc2d0f80459c478827e93d3d"
    assert config["scientific_model"]["dimension"] == 22
    assert config["scientific_model"]["manifold"] == "O_d"
    assert config["nulls"]["replicates"] == 1999
    assert config["identifiability"]["split_multiplier"] == 1.0
    assert config["identifiability"]["available_case_allowed"] is False


def test_execution_seed_is_stable_and_stream_specific() -> None:
    first = deterministic_seed("target", "A", "F", 0, 0, "0train->0train")
    second = deterministic_seed("target", "A", "F", 0, 0, "0train->0train")
    different = deterministic_seed("target", "B", "F", 0, 0, "0train->1test")
    assert isinstance(first, int)
    assert first == second
    assert first != different
    assert 0 <= first < np.iinfo(np.uint64).max + 1
