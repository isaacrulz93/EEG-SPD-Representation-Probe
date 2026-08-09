"""Shape, ordering, and SPD tests for the frozen covariance construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.covariance import (
    build_covariance_representations,
    covariance_sanity_table,
    equal_window_slices,
    estimate_oas_covariance,
    make_covariance_metadata,
)


def _synthetic_epochs() -> np.ndarray:
    rng = np.random.default_rng(20260809)
    mixing = rng.normal(size=(22, 22))
    return np.stack(
        [mixing @ rng.normal(size=(22, 1000)) for _ in range(3)], axis=0
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_index": [0, 1, 2],
            "subject": [1, 1, 2],
            "session": ["0train"] * 3,
            "run": ["0", "0", "0"],
            "trial_id": [1, 2, 1],
            "class_label": ["left_hand", "feet", "tongue"],
        }
    )


def test_whole_and_window5_shapes_and_order() -> None:
    epochs = _synthetic_epochs()
    whole, windowed = build_covariance_representations(epochs, n_windows=5)
    assert whole.shape == (3, 22, 22)
    assert windowed.shape == (15, 22, 22)
    np.testing.assert_allclose(whole[0], estimate_oas_covariance(epochs[0]))
    np.testing.assert_allclose(
        windowed[0], estimate_oas_covariance(epochs[0, :, :200])
    )
    np.testing.assert_allclose(
        windowed[4], estimate_oas_covariance(epochs[0, :, 800:1000])
    )
    np.testing.assert_allclose(
        windowed[5], estimate_oas_covariance(epochs[1, :, :200])
    )
    assert np.all(np.linalg.eigvalsh(whole) > 0)
    assert np.all(np.linalg.eigvalsh(windowed) > 0)


def test_metadata_repetition_matches_trial_major_window_order() -> None:
    whole_metadata, window_metadata = make_covariance_metadata(_metadata(), 5)
    assert len(whole_metadata) == 3
    assert len(window_metadata) == 15
    assert window_metadata.loc[:4, "trial_id"].tolist() == [1] * 5
    assert window_metadata.loc[:4, "window_index"].tolist() == [1, 2, 3, 4, 5]
    assert window_metadata.loc[5:9, "trial_id"].tolist() == [2] * 5


def test_sanity_table_reports_all_covariances() -> None:
    whole, _ = build_covariance_representations(_synthetic_epochs(), n_windows=5)
    whole_metadata, _ = make_covariance_metadata(_metadata(), 5)
    sanity = covariance_sanity_table(whole, whole_metadata, "WHOLE")
    assert len(sanity) == len(whole)
    assert sanity["is_spd"].all()
    assert not sanity["has_nan"].any()
    assert not sanity["has_inf"].any()
    assert sanity["window_index"].isna().all()
    assert (sanity["symmetry_error"] <= 1e-10).all()


def test_equal_windows_require_exact_division() -> None:
    slices = equal_window_slices(1000, 5)
    assert [(item.start, item.stop) for item in slices] == [
        (0, 200),
        (200, 400),
        (400, 600),
        (600, 800),
        (800, 1000),
    ]
    with pytest.raises(ValueError, match="cannot be divided exactly"):
        equal_window_slices(1001, 5)
