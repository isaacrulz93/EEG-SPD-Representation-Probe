"""Leakage-bounded decoders and trial-level metrics for geometry audit V2.

This module intentionally knows nothing about subject centering.  Source model
fitting and target scoring are separate functions so target labels only enter
the final evaluation boundary.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from pyriemann.classification import MDM
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.spd_utils import log_svec


DEFAULT_CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")


@dataclass(frozen=True)
class FitAudit:
    """Machine-readable facts about a source-only estimator fit."""

    decoder: str
    config_hash: str
    n_train: int
    n_features: int | None
    convergence_warning: bool
    warning_messages: tuple[str, ...]
    n_iter_max: int | None


def _finite_2d(features: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a non-empty 2-D array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _spd_stack(covariances: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(covariances, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != array.shape[2] or len(array) < 1:
        raise ValueError(f"{name} must be a non-empty N x C x C array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if not np.allclose(array, array.transpose(0, 2, 1), rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} contains a non-symmetric matrix")
    if np.min(np.linalg.eigvalsh(array)) <= 0.0:
        raise ValueError(f"{name} contains a non-SPD matrix")
    return array


def _labels(labels: np.ndarray, n_rows: int, *, name: str) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(f"{name} must have shape ({n_rows},), got {values.shape}")
    if any(value is None for value in values.tolist()):
        raise ValueError(f"{name} contains null values")
    return values.astype(str)


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible configuration with stable key ordering."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-contiguous array content."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def common_log_svec_features(covariances: np.ndarray) -> np.ndarray:
    """Return the fixed unscaled 253-D common-decoder coordinates."""

    matrices = _spd_stack(covariances, name="covariances")
    features = np.asarray(log_svec(matrices), dtype=np.float64)
    expected = matrices.shape[1] * (matrices.shape[1] + 1) // 2
    if features.shape != (len(matrices), expected):
        raise RuntimeError("log-svec returned an unexpected shape")
    return features


def fit_source_logistic(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    *,
    c: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 5000,
    tol: float = 1e-4,
    random_state: int = 20260809,
) -> tuple[LogisticRegression, FitAudit]:
    """Fit the frozen multinomial L2 logistic decoder on source data only."""

    features = _finite_2d(source_features, name="source_features")
    labels = _labels(source_labels, len(features), name="source_labels")
    if len(np.unique(labels)) < 2:
        raise ValueError("source_labels must contain at least two classes")
    if c <= 0.0 or not np.isfinite(c):
        raise ValueError("c must be finite and positive")
    if solver != "lbfgs":
        raise ValueError("geometry audit V2 freezes solver='lbfgs'")
    if max_iter < 1 or tol <= 0.0:
        raise ValueError("max_iter and tol must be positive")

    # scikit-learn >=1.8 encodes L2 as l1_ratio=0 while the explicit
    # penalty='l2' spelling is deprecated.  Multiclass lbfgs is multinomial.
    model = LogisticRegression(
        C=float(c),
        l1_ratio=0.0,
        solver=solver,
        max_iter=int(max_iter),
        tol=float(tol),
        class_weight=None,
        random_state=int(random_state),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, labels)
    convergence = [
        item for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    parameters = {
        "decoder": "multinomial_logistic_regression",
        "penalty": "l2_via_l1_ratio_0",
        "C": float(c),
        "solver": solver,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "class_weight": None,
        "random_state": int(random_state),
        "standard_scaler": False,
    }
    audit = FitAudit(
        decoder="logistic",
        config_hash=stable_json_hash(parameters),
        n_train=len(features),
        n_features=features.shape[1],
        convergence_warning=bool(convergence),
        warning_messages=tuple(str(item.message) for item in caught),
        n_iter_max=int(np.max(model.n_iter_)),
    )
    return model, audit


def fit_source_mdm(
    source_covariances: np.ndarray,
    source_labels: np.ndarray,
    *,
    metric: str,
) -> tuple[MDM, FitAudit]:
    """Fit one frozen metric-native MDM on source matrices only."""

    matrices = _spd_stack(source_covariances, name="source_covariances")
    labels = _labels(source_labels, len(matrices), name="source_labels")
    if metric not in {"riemann", "logeuclid"}:
        raise ValueError("MDM metric must be 'riemann' or 'logeuclid'")
    model = MDM(metric=metric, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(matrices, labels)
    convergence = [
        item for item in caught if "convergence" in str(item.message).lower()
    ]
    parameters = {"decoder": "MDM", "metric": metric, "n_jobs": 1}
    audit = FitAudit(
        decoder=f"mdm_{metric}",
        config_hash=stable_json_hash(parameters),
        n_train=len(matrices),
        n_features=None,
        convergence_warning=bool(convergence),
        warning_messages=tuple(str(item.message) for item in caught),
        n_iter_max=None,
    )
    return model, audit


def prediction_metrics(
    target_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    class_order: Sequence[str] = DEFAULT_CLASS_ORDER,
) -> dict[str, Any]:
    """Evaluate target predictions; this is the only target-label boundary."""

    truth = np.asarray(target_labels).astype(str)
    predicted = np.asarray(predicted_labels).astype(str)
    if truth.ndim != 1 or predicted.shape != truth.shape or len(truth) < 1:
        raise ValueError("target and predicted labels must be equal non-empty vectors")
    classes = tuple(str(value) for value in class_order)
    if len(classes) != len(set(classes)) or len(classes) < 2:
        raise ValueError("class_order must contain unique class labels")
    observed = set(truth.tolist()) | set(predicted.tolist())
    unknown = observed - set(classes)
    if unknown:
        raise ValueError(f"labels outside frozen class order: {sorted(unknown)}")

    recalls = recall_score(
        truth, predicted, labels=list(classes), average=None, zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=list(classes))
    result: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(classes),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix_json": json.dumps(matrix.tolist(), separators=(",", ":")),
        "prediction_sha256": stable_json_hash(
            {
                "truth": truth.tolist(),
                "predicted": predicted.tolist(),
                "class_order": list(classes),
            }
        ),
        "n_evaluation": int(len(truth)),
    }
    for label, recall in zip(classes, recalls, strict=True):
        result[f"recall_{label}"] = float(recall)
    for row, true_label in enumerate(classes):
        for column, predicted_label in enumerate(classes):
            result[f"confusion_{true_label}__{predicted_label}"] = int(
                matrix[row, column]
            )
    for metric in ("balanced_accuracy", "accuracy", "macro_f1"):
        if not 0.0 <= result[metric] <= 1.0:
            raise RuntimeError(f"{metric} fell outside [0,1]")
    return result


def evaluate_target_estimator(
    estimator: Any,
    target_inputs: np.ndarray,
    target_labels: np.ndarray,
    *,
    class_order: Sequence[str] = DEFAULT_CLASS_ORDER,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict target inputs and cross the target-label boundary exactly once."""

    prediction = np.asarray(estimator.predict(target_inputs)).astype(str)
    metrics = prediction_metrics(
        target_labels, prediction, class_order=class_order
    )
    return prediction, metrics
