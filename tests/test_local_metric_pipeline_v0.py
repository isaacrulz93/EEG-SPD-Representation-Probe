from __future__ import annotations

import numpy as np

from src.local_metric_interaction_v0 import CLASS_ORDER
from src.local_metric_pipeline_v0 import (
    _balanced_accuracy,
    _permute_within_runs,
    class_medoid_indices,
)


def test_class_medoid_uses_frozen_class_order_and_deterministic_ties() -> None:
    labels = np.repeat(np.asarray(CLASS_ORDER), 72)
    coordinates = np.concatenate(
        [np.linspace(0.0, 1.0, 72) + 10.0 * index for index in range(4)]
    )
    distances = np.abs(coordinates[:, None] - coordinates[None, :])
    medoids = class_medoid_indices(distances, labels)
    # A median-distance objective has a central tie plateau in this fixture;
    # the lowest local/global index on that plateau is the frozen choice.
    assert np.array_equal(medoids, np.asarray([18, 90, 162, 234]))


def test_training_label_null_preserves_per_run_class_counts() -> None:
    labels = np.tile(np.repeat(np.asarray(CLASS_ORDER), 12), 6)
    runs = np.repeat(np.arange(6), 48)
    first = _permute_within_runs(labels, runs, np.random.default_rng(99))
    second = _permute_within_runs(labels, runs, np.random.default_rng(99))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, labels)
    for run in range(6):
        positions = runs == run
        unique, counts = np.unique(first[positions], return_counts=True)
        assert set(unique) == set(CLASS_ORDER)
        assert np.array_equal(counts, np.full(4, 12))


def test_balanced_accuracy_uses_all_four_classes_equally() -> None:
    truth = np.repeat(np.asarray(CLASS_ORDER), [8, 12, 20, 40])
    predicted = truth.copy()
    predicted[:4] = "tongue"
    predicted[8:14] = "feet"
    assert _balanced_accuracy(truth, predicted) == (0.5 + 0.5 + 1.0 + 1.0) / 4.0
