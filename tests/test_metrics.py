"""Synthetic tests for leakage-safe quantitative representation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from src.metrics import (
    deterministic_silhouette,
    deterministic_trial_folds,
    grouped_logistic_probe,
    rms_distance_ratio,
    transition_diagnostics,
)


def _window_metadata(
    *, n_subjects: int = 3, n_classes: int = 2, trials_per_cell: int = 10
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    classes = [f"class_{index}" for index in range(n_classes)]
    for subject in range(1, n_subjects + 1):
        trial_id = 0
        for class_label in classes:
            for _ in range(trials_per_cell):
                trial_id += 1
                for window in range(1, 6):
                    rows.append(
                        {
                            "subject": subject,
                            "session": "0train",
                            "trial_id": trial_id,
                            "class_label": class_label,
                            "window_index": window,
                        }
                    )
    return pd.DataFrame(rows)


def test_trial_folds_are_trial_grouped_stratified_and_row_order_invariant() -> None:
    metadata = _window_metadata(trials_per_cell=10)
    folds = deterministic_trial_folds(metadata, n_splits=5, seed=20260809)
    keyed = metadata.assign(fold=folds)

    assert set(folds) == set(range(5))
    assert (
        keyed.groupby(["subject", "session", "trial_id"])["fold"].nunique() == 1
    ).all()
    unique_trials = keyed.drop_duplicates(["subject", "session", "trial_id"])
    counts = unique_trials.groupby(["subject", "class_label", "fold"]).size()
    assert (counts == 2).all()

    permutation = np.random.default_rng(44).permutation(len(metadata))
    shuffled = metadata.iloc[permutation].reset_index(drop=True)
    shuffled_folds = deterministic_trial_folds(shuffled, n_splits=5, seed=20260809)
    original_map = {
        tuple(key): int(fold)
        for key, fold in keyed[
            ["subject", "session", "trial_id", "fold"]
        ].drop_duplicates().set_index(["subject", "session", "trial_id"])["fold"].items()
    }
    for row, fold in zip(shuffled.itertuples(index=False), shuffled_folds, strict=True):
        assert fold == original_map[(row.subject, row.session, row.trial_id)]


def test_trial_folds_reject_underpopulated_strata() -> None:
    metadata = _window_metadata(trials_per_cell=4)
    with pytest.raises(ValueError, match="too-small strata"):
        deterministic_trial_folds(metadata, n_splits=5, seed=1)


def test_rms_distance_ratio_matches_explicit_unordered_pairs() -> None:
    rng = np.random.default_rng(7)
    coordinates = rng.normal(size=(19, 6))
    labels = np.array(["a"] * 7 + ["b"] * 6 + ["c"] * 6)
    within: list[float] = []
    between: list[float] = []
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            squared = float(np.square(coordinates[left] - coordinates[right]).sum())
            (within if labels[left] == labels[right] else between).append(squared)
    expected = np.sqrt(np.mean(between)) / np.sqrt(np.mean(within))
    assert rms_distance_ratio(coordinates, labels) == pytest.approx(
        expected, rel=1e-13, abs=1e-13
    )


def test_deterministic_silhouette_returns_and_reuses_exact_indices() -> None:
    rng = np.random.default_rng(9)
    coordinates = np.r_[
        rng.normal(loc=-2.0, scale=0.3, size=(50, 4)),
        rng.normal(loc=2.0, scale=0.3, size=(50, 4)),
    ]
    labels = np.repeat(["left", "right"], 50)
    score, indices = deterministic_silhouette(
        coordinates, labels, max_samples=60, seed=123
    )
    repeated, repeated_indices = deterministic_silhouette(
        coordinates, labels, indices=indices
    )
    assert len(indices) == 60
    np.testing.assert_array_equal(indices, repeated_indices)
    assert score == repeated
    assert score == pytest.approx(
        silhouette_score(coordinates[indices], labels[indices]), rel=0, abs=1e-15
    )


def test_window_probe_is_grouped_and_reports_all_oof_diagnostics() -> None:
    metadata = _window_metadata(trials_per_cell=10)
    rng = np.random.default_rng(22)
    n_rows = len(metadata)
    coordinates = rng.normal(scale=0.15, size=(n_rows, 5))
    coordinates[:, 0] += np.where(metadata["class_label"] == "class_0", -3.0, 3.0)
    for subject in range(1, 4):
        coordinates[metadata["subject"].to_numpy() == subject, subject] += 4.0

    result = grouped_logistic_probe(
        coordinates,
        metadata,
        "class_label",
        n_splits=5,
        seed=20260809,
    )
    summary = result["summary"]
    assert summary["standard_scaler"] is False
    assert summary["pca"] is False
    assert summary["pooled_oof_point_accuracy"] > 0.99
    assert summary["pooled_oof_trial_mean_probability_accuracy"] > 0.99
    assert len(result["fold_metrics"]) == 5
    assert len(result["per_window_metrics"]) == 30
    assert len(result["oof_predictions"]) == len(metadata)
    assert len(result["trial_predictions"]) == len(metadata) // 5
    assert not result["oof_predictions"]["fold"].isna().any()
    assert (
        result["oof_predictions"].groupby(
            ["subject", "session", "trial_id"]
        )["fold"].nunique()
        == 1
    ).all()

    # Numeric subject labels exercise a distinct sklearn target dtype path.
    subject_result = grouped_logistic_probe(
        coordinates,
        metadata,
        "subject",
        n_splits=5,
        seed=20260809,
    )
    assert subject_result["summary"]["pooled_oof_point_accuracy"] > 0.99
    assert (
        subject_result["summary"]["pooled_oof_trial_mean_probability_accuracy"]
        > 0.99
    )


def test_transition_diagnostics_validate_invariance_and_summarize() -> None:
    metadata = _window_metadata(n_subjects=2, n_classes=2, trials_per_cell=5)
    raw = np.empty((len(metadata), 3), dtype=np.float64)
    subject_offsets = {1: np.array([10.0, -3.0, 1.0]), 2: np.array([-8.0, 5.0, 2.0])}
    class_steps = {
        "class_0": np.array([1.0, 0.0, 0.0]),
        "class_1": np.array([0.0, 2.0, 0.0]),
    }
    for index, row in enumerate(metadata.itertuples(index=False)):
        pair_scale = 1.0 + 0.1 * (row.window_index - 1)
        # A cumulative state with class-dependent direction and pair magnitude.
        cumulative = sum(
            1.0 + 0.1 * transition for transition in range(row.window_index - 1)
        )
        raw[index] = subject_offsets[row.subject] + cumulative * class_steps[row.class_label]
    centered = raw.copy()
    for subject, offset in subject_offsets.items():
        centered[metadata["subject"].to_numpy() == subject] -= offset

    result = transition_diagnostics(raw, centered, metadata)
    assert result["validation"]["raw_centered_delta_invariant"] is True
    assert result["validation"]["n_trials"] == 20
    assert len(result["per_transition"]) == 80
    assert len(result["class_summary"]) == 10  # two classes x (all + four pairs)
    assert len(result["subject_summary"]) == 10
    assert len(result["class_mean_cosine"]) == 5  # one class pair x five scopes
    assert len(result["eta_squared"]) == 8  # class/subject x four pairs
    class_eta = result["eta_squared"].query("factor == 'class'")["eta_squared"]
    assert (class_eta > 0.99).all()
    assert result["per_transition"]["magnitude_absolute_difference"].max() < 1e-12

    broken = centered.copy()
    broken[0, 0] += 0.1
    with pytest.raises(ValueError, match="not invariant"):
        transition_diagnostics(raw, broken, metadata)


def test_transition_diagnostics_reject_missing_window() -> None:
    metadata = _window_metadata(n_subjects=1, n_classes=2, trials_per_cell=5)
    metadata = metadata.drop(index=0).reset_index(drop=True)
    coordinates = np.zeros((len(metadata), 2))
    with pytest.raises(ValueError, match="each window 1..5 exactly once"):
        transition_diagnostics(coordinates, coordinates, metadata)
