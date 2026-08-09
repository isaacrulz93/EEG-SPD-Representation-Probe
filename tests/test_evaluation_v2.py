"""Unit tests for V2's separated source-fit and target-score boundary."""

from __future__ import annotations

import inspect

import numpy as np

from src.evaluation_v2 import (
    common_log_svec_features,
    evaluate_target_estimator,
    fit_source_logistic,
    prediction_metrics,
)


def _spd(seed: int, n: int = 16, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, channels, channels))
    identity = np.eye(channels)
    return factors @ factors.transpose(0, 2, 1) + 0.5 * identity


def test_common_features_are_unscaled_log_svec() -> None:
    covariances = _spd(2)
    features = common_log_svec_features(covariances)
    assert features.shape == (16, 6)
    assert np.isfinite(features).all()


def test_source_fit_signature_has_no_target_arguments() -> None:
    parameters = set(inspect.signature(fit_source_logistic).parameters)
    assert "target_labels" not in parameters
    assert "target_features" not in parameters


def test_logistic_fit_and_target_evaluation_are_separate() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(80, 5))
    source_labels = np.repeat(["left_hand", "right_hand", "feet", "tongue"], 20)
    source[:, 0] += np.repeat([-2.0, -0.5, 0.5, 2.0], 20)
    target = source.copy()
    model, audit = fit_source_logistic(source, source_labels)
    prediction, metrics = evaluate_target_estimator(model, target, source_labels)
    assert len(prediction) == 80
    assert audit.n_train == 80
    assert not audit.convergence_warning
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert sum(
        metrics[key]
        for key in metrics
        if key.startswith("confusion_") and key != "confusion_matrix_json"
    ) == 80


def test_prediction_metrics_respects_frozen_class_order() -> None:
    truth = np.array(["left_hand", "right_hand", "feet", "tongue"])
    metrics = prediction_metrics(truth, truth)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["recall_feet"] == 1.0
