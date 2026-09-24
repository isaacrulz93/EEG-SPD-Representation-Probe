"""Source-only LOSO probes and falsification statistics for adaptive segments."""
from __future__ import annotations
import warnings
import hashlib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.adaptive_segmentation_v0 import segment_features, exact_sign_flip

CLASSES = ("left_hand", "right_hand", "feet", "tongue")


def trial_features(X, boundaries_by_trial):
    return [segment_features(x, b) for x, b in zip(X, boundaries_by_trial)]


def fit_score(train_features, train_labels, test_features, test_lengths):
    X = np.concatenate(train_features, axis=0)
    y = np.concatenate([np.repeat(label, len(f)) for f, label in zip(train_features, train_labels)])
    scaler = StandardScaler().fit(X)
    # Current scikit-learn removed multi_class; lbfgs uses multinomial loss
    # automatically for four classes. Pin the requested objective explicitly
    # in the implementation audit/config rather than passing the removed arg.
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-4)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(X), y)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    probs = []
    for f, lens in zip(test_features, test_lengths):
        segment_p = model.predict_proba(scaler.transform(f))
        probs.append(np.average(segment_p, axis=0, weights=np.asarray(lens, dtype=float)))
    mean_hash=hashlib.sha256(np.ascontiguousarray(scaler.mean_).tobytes()).hexdigest()
    return np.stack(probs), {"convergence_warnings": convergence, "train_segments": len(y),
                             "scaler_fit_segments": int(len(X)), "scaler_mean_sha256": mean_hash}


def metric_row(y, p, classes=CLASSES):
    pred = np.asarray(classes)[np.argmax(p, axis=1)]
    recalls = recall_score(y, pred, labels=list(classes), average=None, zero_division=0)
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(classes), average="macro", zero_division=0)),
            **{f"recall_{name}": float(value) for name, value in zip(classes, recalls)}}


def paired_table(subject_scores):
    comparisons = [("ADAPT5", "FIXED5", "ADAPT5_gt_FIXED5"),
                   ("ADAPT5", "RANDOM_MATCHED", "ADAPT5_gt_RANDOM_MATCHED"),
                   ("ADAPT5", "RANDOM5", "ADAPT5_gt_RANDOM5")]
    result = []
    for first, second, name in comparisons:
        diffs = [subject_scores[first][s] - subject_scores[second][s]
                 for s in sorted(subject_scores[first])]
        result.append({"comparison": name, **exact_sign_flip(diffs)})
    return result
