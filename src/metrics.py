"""Quantitative diagnostics in the original Log-Euclidean coordinates.

This module deliberately contains no feature scaling, PCA, embedding, model
selection, or learned sequence model.  Every diagnostic consumes the original
``svec(log(C))`` rows produced by the frozen representation pipeline.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score


_REQUIRED_TRIAL_COLUMNS = {"subject", "trial_id", "class_label"}
_WINDOWS = (1, 2, 3, 4, 5)


def _as_finite_matrix(coordinates: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(coordinates, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {matrix.shape}")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have at least one row and one column")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return matrix


def _as_metadata(metadata: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    if len(metadata) != n_rows:
        raise ValueError(
            f"coordinates have {n_rows} rows but metadata has {len(metadata)} rows"
        )
    missing = _REQUIRED_TRIAL_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if metadata.empty:
        raise ValueError("metadata cannot be empty")
    required = list(_REQUIRED_TRIAL_COLUMNS)
    if "session" in metadata:
        required.append("session")
    if metadata[required].isna().any(axis=None):
        raise ValueError("trial identity, class, and subject metadata cannot be null")
    return metadata.reset_index(drop=True).copy()


def _trial_columns(metadata: pd.DataFrame) -> list[str]:
    columns = ["subject"]
    if "session" in metadata:
        columns.append("session")
    columns.append("trial_id")
    return columns


def _stable_value(value: Any) -> dict[str, str]:
    """Represent a scalar without conflating, for example, integer 1 and '1'."""

    if isinstance(value, np.generic):
        value = value.item()
    return {"type": type(value).__name__, "value": str(value)}


def _stable_digest(seed: int, values: Iterable[Any]) -> str:
    payload = {
        "seed": int(seed),
        "values": [_stable_value(value) for value in values],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_trial_consistency(metadata: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    trial_columns = _trial_columns(metadata)
    grouped = metadata.groupby(trial_columns, sort=False, dropna=False)
    class_counts = grouped["class_label"].nunique(dropna=False)
    if not (class_counts == 1).all():
        bad = class_counts[class_counts != 1].index.tolist()[:5]
        raise ValueError(f"a trial has inconsistent class labels: {bad}")

    # ``subject`` is already part of the group index; selecting it again would
    # create a duplicate column when the index is reset.
    trial_table = grouped["class_label"].first().reset_index()
    if trial_table.duplicated(trial_columns).any():
        raise RuntimeError("internal error while constructing the unique-trial table")
    return trial_columns, trial_table


def deterministic_trial_folds(
    metadata: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Assign row-aligned, deterministic trial folds stratified by subject/class.

    Unique trials within every ``subject x class_label`` stratum are ordered by
    a SHA256 digest of the seed and full trial identity, then assigned round
    robin to folds.  Thus assignment is independent of input row order and all
    five rows of a WINDOW5 trial necessarily stay in the same fold.
    """

    if not isinstance(n_splits, (int, np.integer)) or int(n_splits) < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    n_splits = int(n_splits)
    frame = _as_metadata(metadata, len(metadata))
    trial_columns, trials = _validate_trial_consistency(frame)

    stratum_sizes = trials.groupby(
        ["subject", "class_label"], sort=False, dropna=False
    ).size()
    if (stratum_sizes < n_splits).any():
        small = stratum_sizes[stratum_sizes < n_splits].to_dict()
        raise ValueError(
            "each subject x class stratum needs at least n_splits unique trials; "
            f"too-small strata: {small}"
        )

    assignment: dict[tuple[Any, ...], int] = {}
    for _, stratum in trials.groupby(
        ["subject", "class_label"], sort=False, dropna=False
    ):
        ranked: list[tuple[str, tuple[Any, ...]]] = []
        for values in stratum[trial_columns].itertuples(index=False, name=None):
            key = tuple(values)
            digest = _stable_digest(seed, key)
            ranked.append((digest, key))
        # The key is a deterministic collision tie-breaker.  repr supports
        # heterogeneous scalar types without relying on their ordering.
        ranked.sort(key=lambda item: (item[0], repr(item[1])))
        for rank, (_, key) in enumerate(ranked):
            assignment[key] = rank % n_splits

    folds = np.fromiter(
        (
            assignment[tuple(values)]
            for values in frame[trial_columns].itertuples(index=False, name=None)
        ),
        dtype=np.int64,
        count=len(frame),
    )
    if set(folds.tolist()) != set(range(n_splits)):
        raise RuntimeError("fold construction did not populate every requested fold")
    return folds


def _validated_labels(labels: np.ndarray, n_rows: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(f"labels must be one-dimensional with length {n_rows}")
    if pd.isna(values).any():
        raise ValueError("labels cannot contain null values")
    return values


def rms_distance_ratio(coordinates: np.ndarray, labels: np.ndarray) -> float:
    """Return between-label RMS distance divided by within-label RMS distance.

    The sufficient-statistic identity

    ``sum(i<j) ||x_i-x_j||^2 = n*sum_i||x_i||^2 - ||sum_i x_i||^2``

    computes exactly the same statistic as enumerating every unordered pair,
    without allocating a quadratic distance matrix.  No sampling is used.
    """

    matrix = _as_finite_matrix(coordinates, name="coordinates")
    values = _validated_labels(labels, len(matrix))
    unique = pd.unique(values)
    if len(unique) < 2:
        raise ValueError("distance ratio requires at least two label groups")

    n_rows = len(matrix)
    total_pairs = n_rows * (n_rows - 1) // 2
    if total_pairs == 0:
        raise ValueError("distance ratio requires at least two rows")
    total_ss = n_rows * np.einsum("ij,ij->", matrix, matrix) - np.dot(
        matrix.sum(axis=0), matrix.sum(axis=0)
    )

    within_ss = 0.0
    within_pairs = 0
    for label in unique:
        group = matrix[values == label]
        n_group = len(group)
        if n_group < 2:
            continue
        within_pairs += n_group * (n_group - 1) // 2
        within_ss += n_group * np.einsum("ij,ij->", group, group) - np.dot(
            group.sum(axis=0), group.sum(axis=0)
        )

    between_pairs = total_pairs - within_pairs
    if within_pairs == 0:
        raise ValueError("no within-label pairs are available")
    if between_pairs == 0:
        raise ValueError("no between-label pairs are available")

    # Roundoff can make an analytically non-negative residual a few ulps below
    # zero.  Materially negative values indicate an invalid computation.
    scale = max(abs(float(total_ss)), abs(float(within_ss)), 1.0)
    tolerance = 100.0 * np.finfo(np.float64).eps * scale
    between_ss = float(total_ss - within_ss)
    if within_ss < -tolerance or between_ss < -tolerance:
        raise FloatingPointError("pairwise squared-distance sum became negative")
    within_ss = max(float(within_ss), 0.0)
    between_ss = max(between_ss, 0.0)
    within_rms = np.sqrt(within_ss / within_pairs)
    between_rms = np.sqrt(between_ss / between_pairs)
    if within_rms == 0.0:
        if between_rms == 0.0:
            raise ValueError("distance ratio is undefined when all distances are zero")
        return float("inf")
    return float(between_rms / within_rms)


def deterministic_silhouette(
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: int | None = None,
    seed: int = 0,
    indices: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Compute Euclidean silhouette on one auditable deterministic subset.

    Passing ``indices`` reuses an already chosen subset, which is useful when
    comparing class, subject, and window labels on exactly the same points.
    The returned indices are always sorted and can be persisted by the caller.
    """

    matrix = _as_finite_matrix(coordinates, name="coordinates")
    values = _validated_labels(labels, len(matrix))
    if indices is not None:
        selected = np.asarray(indices)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("indices must be a one-dimensional integer array")
        selected = selected.astype(np.int64, copy=False)
        if len(selected) == 0:
            raise ValueError("indices cannot be empty")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("indices cannot contain duplicates")
        if selected.min() < 0 or selected.max() >= len(matrix):
            raise ValueError("indices contain an out-of-range row")
        selected = np.sort(selected)
    else:
        if max_samples is not None and (
            not isinstance(max_samples, (int, np.integer)) or int(max_samples) < 3
        ):
            raise ValueError("max_samples must be an integer of at least 3")
        if max_samples is None or int(max_samples) >= len(matrix):
            selected = np.arange(len(matrix), dtype=np.int64)
        else:
            generator = np.random.default_rng(int(seed))
            selected = np.sort(
                generator.choice(len(matrix), size=int(max_samples), replace=False)
            ).astype(np.int64)

    subset_labels = values[selected]
    n_labels = len(pd.unique(subset_labels))
    if not 2 <= n_labels < len(selected):
        raise ValueError(
            "silhouette needs between 2 and n_samples-1 labels in the selected subset"
        )
    score = silhouette_score(matrix[selected], subset_labels, metric="euclidean")
    return float(score), selected


def _validate_window5(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    if "window_index" not in frame:
        raise ValueError("WINDOW5 metadata must contain window_index")
    if frame["window_index"].isna().any():
        raise ValueError("window_index cannot contain null values")
    numeric = pd.to_numeric(frame["window_index"], errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("window_index values must be integers 1..5")
    frame = frame.copy()
    frame["window_index"] = numeric.astype(np.int64)
    trial_columns, _ = _validate_trial_consistency(frame)
    grouped_windows = frame.groupby(trial_columns, sort=False, dropna=False)[
        "window_index"
    ]
    invalid: list[tuple[Any, ...]] = []
    for trial_key, windows in grouped_windows:
        if tuple(sorted(windows.tolist())) != _WINDOWS:
            key = trial_key if isinstance(trial_key, tuple) else (trial_key,)
            invalid.append(key)
            if len(invalid) == 5:
                break
    if invalid:
        raise ValueError(
            "every WINDOW5 trial must contain each window 1..5 exactly once; "
            f"invalid trials: {invalid}"
        )
    return trial_columns, frame


def _probability_column(label: Any) -> str:
    return f"probability__{str(label)}"


def grouped_logistic_probe(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    target_column: str,
    *,
    n_splits: int = 5,
    seed: int = 0,
    c: float = 1.0,
    max_iter: int = 5000,
) -> dict[str, Any]:
    """Run a fixed grouped multinomial logistic linear information probe.

    Folds are SHA256 trial folds stratified by subject and class.  OOF point
    accuracy is reported for both WHOLE and WINDOW5.  For WINDOW5, probabilities
    are also averaged within each held-out trial before prediction and point
    accuracy is broken down by window.  Features are used exactly as supplied:
    no scaler, PCA, tuning, or t-SNE coordinates enter this function.
    """

    matrix = _as_finite_matrix(coordinates, name="coordinates")
    frame = _as_metadata(metadata, len(matrix))
    if target_column not in {"class_label", "subject"}:
        raise ValueError("target_column must be 'class_label' or 'subject'")
    if not np.isfinite(float(c)) or float(c) <= 0.0:
        raise ValueError("c must be a finite positive number")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")

    is_window5 = "window_index" in frame
    if is_window5:
        trial_columns, frame = _validate_window5(frame)
    else:
        trial_columns, _ = _validate_trial_consistency(frame)
    target = _validated_labels(frame[target_column].to_numpy(), len(frame))
    classes = np.unique(target)
    if len(classes) < 2:
        raise ValueError("the probe target must contain at least two classes")
    probability_columns = [_probability_column(label) for label in classes]
    if len(set(probability_columns)) != len(probability_columns):
        raise ValueError("target labels yield colliding probability column names")

    folds = deterministic_trial_folds(frame, n_splits=n_splits, seed=seed)
    # Preserve the target dtype.  An object-typed prediction array containing
    # integers is classified as an "unknown" target by sklearn metrics.
    oof_prediction = np.empty_like(target)
    oof_probability = np.full((len(matrix), len(classes)), np.nan, dtype=np.float64)
    convergence = np.zeros(int(n_splits), dtype=bool)
    fold_records: list[dict[str, Any]] = []

    for fold in range(int(n_splits)):
        test = folds == fold
        train = ~test
        if not test.any() or not train.any():
            raise RuntimeError(f"fold {fold} has an empty train or test partition")
        if len(np.unique(target[train])) != len(classes):
            raise RuntimeError(f"fold {fold} training data omit a target class")
        model = LogisticRegression(
            penalty="l2",
            C=float(c),
            solver="lbfgs",
            max_iter=int(max_iter),
            random_state=int(seed),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(matrix[train], target[train])
        convergence[fold] = any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        prediction = model.predict(matrix[test])
        probability = model.predict_proba(matrix[test])
        if not np.array_equal(model.classes_, classes):
            # This should be unreachable after the complete-class check, but
            # aligning explicitly prevents silent probability-column leakage.
            order = [int(np.flatnonzero(model.classes_ == label)[0]) for label in classes]
            probability = probability[:, order]
        oof_prediction[test] = prediction
        oof_probability[test] = probability
        fold_records.append(
            {
                "fold": fold,
                "n_train_rows": int(train.sum()),
                "n_test_rows": int(test.sum()),
                "n_train_trials": int(frame.loc[train, trial_columns].drop_duplicates().shape[0]),
                "n_test_trials": int(frame.loc[test, trial_columns].drop_duplicates().shape[0]),
                "point_accuracy": float(accuracy_score(target[test], prediction)),
                "convergence_warning": bool(convergence[fold]),
            }
        )

    if np.isnan(oof_probability).any():
        raise RuntimeError("OOF predictions are incomplete")
    oof = frame.copy()
    oof.insert(0, "fold", folds)
    oof["true_label"] = target
    oof["predicted_label"] = oof_prediction
    oof["correct"] = target == oof_prediction
    for column, probability in zip(probability_columns, oof_probability.T, strict=True):
        oof[column] = probability

    fold_metrics = pd.DataFrame.from_records(fold_records)
    per_window_records: list[dict[str, Any]] = []
    trial_predictions = pd.DataFrame()
    pooled_trial_accuracy: float | None = None
    if is_window5:
        for fold in range(int(n_splits)):
            for window in _WINDOWS:
                mask = (folds == fold) & (frame["window_index"].to_numpy() == window)
                per_window_records.append(
                    {
                        "scope": "fold",
                        "fold": fold,
                        "window_index": window,
                        "n_rows": int(mask.sum()),
                        "accuracy": float(accuracy_score(target[mask], oof_prediction[mask])),
                    }
                )
        for window in _WINDOWS:
            mask = frame["window_index"].to_numpy() == window
            per_window_records.append(
                {
                    "scope": "pooled_oof",
                    "fold": pd.NA,
                    "window_index": window,
                    "n_rows": int(mask.sum()),
                    "accuracy": float(accuracy_score(target[mask], oof_prediction[mask])),
                }
            )

        probability_frame = pd.DataFrame(oof_probability, columns=probability_columns)
        identity_and_target = trial_columns + (
            [] if target_column in trial_columns else [target_column]
        )
        aggregate_source = pd.concat(
            [frame[identity_and_target].reset_index(drop=True), probability_frame],
            axis=1,
        )
        target_counts = aggregate_source.groupby(trial_columns, dropna=False)[
            target_column
        ].nunique(dropna=False)
        if not (target_counts == 1).all():
            raise RuntimeError("probe target varies within a trial")
        probability_means = aggregate_source.groupby(
            trial_columns, sort=False, dropna=False
        )[probability_columns].mean()
        trial_truth = aggregate_source.groupby(
            trial_columns, sort=False, dropna=False
        )[target_column].first()
        trial_predictions = probability_means.reset_index()
        trial_predictions["true_label"] = trial_truth.to_numpy()
        trial_probability = probability_means.to_numpy()
        trial_predictions["predicted_label"] = classes[
            np.argmax(trial_probability, axis=1)
        ]
        trial_predictions["correct"] = (
            trial_predictions["true_label"].to_numpy()
            == trial_predictions["predicted_label"].to_numpy()
        )
        trial_fold_map = (
            oof.groupby(trial_columns, sort=False, dropna=False)["fold"]
            .agg(lambda values: values.iloc[0] if values.nunique() == 1 else -1)
            .to_numpy()
        )
        if (trial_fold_map < 0).any():
            raise RuntimeError("a trial was split across OOF folds")
        trial_predictions.insert(0, "fold", trial_fold_map.astype(np.int64))
        pooled_trial_accuracy = float(trial_predictions["correct"].mean())
        for fold in range(int(n_splits)):
            value = float(
                trial_predictions.loc[
                    trial_predictions["fold"] == fold, "correct"
                ].mean()
            )
            fold_metrics.loc[
                fold_metrics["fold"] == fold, "trial_mean_probability_accuracy"
            ] = value
    else:
        fold_metrics["trial_mean_probability_accuracy"] = np.nan

    per_window_metrics = pd.DataFrame.from_records(
        per_window_records,
        columns=["scope", "fold", "window_index", "n_rows", "accuracy"],
    )
    if not per_window_metrics.empty:
        per_window_metrics["fold"] = per_window_metrics["fold"].astype("Int64")
    summary = {
        "target_column": target_column,
        "representation": "WINDOW5" if is_window5 else "WHOLE",
        "n_rows": int(len(matrix)),
        "n_trials": int(frame[trial_columns].drop_duplicates().shape[0]),
        "n_features": int(matrix.shape[1]),
        "n_target_classes": int(len(classes)),
        "chance_accuracy": float(1.0 / len(classes)),
        "n_splits": int(n_splits),
        "seed": int(seed),
        "model": "LogisticRegression(penalty='l2', solver='lbfgs')",
        "c": float(c),
        "max_iter": int(max_iter),
        "standard_scaler": False,
        "pca": False,
        "pooled_oof_point_accuracy": float(accuracy_score(target, oof_prediction)),
        "pooled_oof_trial_mean_probability_accuracy": pooled_trial_accuracy,
        "any_convergence_warning": bool(convergence.any()),
        "probability_columns": probability_columns,
    }
    return {
        "summary": summary,
        "fold_metrics": fold_metrics,
        "per_window_metrics": per_window_metrics,
        "oof_predictions": oof,
        "trial_predictions": trial_predictions,
    }


def _descriptive_summary(
    transitions: pd.DataFrame, group_column: str
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_value, group in transitions.groupby(group_column, sort=True):
        scopes = [("all", group)] + [
            (pair, pair_group)
            for pair, pair_group in group.groupby("window_pair", sort=True)
        ]
        for scope, values in scopes:
            magnitude = values["transition_magnitude"].to_numpy(dtype=np.float64)
            records.append(
                {
                    group_column: group_value,
                    "window_pair": scope,
                    "n_transitions": int(len(magnitude)),
                    "mean_transition_magnitude": float(magnitude.mean()),
                    "std_transition_magnitude": float(magnitude.std(ddof=1))
                    if len(magnitude) > 1
                    else 0.0,
                    "median_transition_magnitude": float(np.median(magnitude)),
                }
            )
    return pd.DataFrame.from_records(records)


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    grand_mean = float(values.mean())
    total = float(np.square(values - grand_mean).sum())
    if total == 0.0:
        return 0.0
    between = 0.0
    for label in pd.unique(labels):
        group = values[labels == label]
        between += len(group) * float(group.mean() - grand_mean) ** 2
    result = between / total
    return float(np.clip(result, 0.0, 1.0))


def transition_diagnostics(
    raw_coordinates: np.ndarray,
    centered_coordinates: np.ndarray,
    metadata: pd.DataFrame,
    *,
    invariance_atol: float = 1e-10,
    invariance_rtol: float = 1e-10,
) -> dict[str, Any]:
    """Describe WINDOW5 consecutive displacements without learning a sequence.

    The function validates exactly one row for each window 1..5 in every trial
    and rejects centered coordinates unless all four consecutive displacement
    vectors agree with RAW within the declared tolerance.
    """

    raw = _as_finite_matrix(raw_coordinates, name="raw_coordinates")
    centered = _as_finite_matrix(centered_coordinates, name="centered_coordinates")
    if raw.shape != centered.shape:
        raise ValueError(
            f"raw and centered coordinate shapes differ: {raw.shape} vs {centered.shape}"
        )
    if invariance_atol < 0.0 or invariance_rtol < 0.0:
        raise ValueError("invariance tolerances must be non-negative")
    frame = _as_metadata(metadata, len(raw))
    trial_columns, frame = _validate_window5(frame)
    frame["_row_position"] = np.arange(len(frame), dtype=np.int64)

    transition_records: list[dict[str, Any]] = []
    raw_deltas: list[np.ndarray] = []
    centered_deltas: list[np.ndarray] = []
    for trial_key, trial in frame.groupby(trial_columns, sort=True, dropna=False):
        trial = trial.sort_values("window_index")
        row_positions = trial["_row_position"].to_numpy(dtype=np.int64)
        trial_raw = raw[row_positions]
        trial_centered = centered[row_positions]
        raw_delta = np.diff(trial_raw, axis=0)
        centered_delta = np.diff(trial_centered, axis=0)
        raw_deltas.extend(raw_delta)
        centered_deltas.extend(centered_delta)
        key_values = trial_key if isinstance(trial_key, tuple) else (trial_key,)
        identity = dict(zip(trial_columns, key_values, strict=True))
        subject = trial["subject"].iloc[0]
        class_label = trial["class_label"].iloc[0]
        for transition_index in range(4):
            raw_norm = float(np.linalg.norm(raw_delta[transition_index]))
            centered_norm = float(np.linalg.norm(centered_delta[transition_index]))
            transition_records.append(
                {
                    **identity,
                    "class_label": class_label,
                    "window_from": transition_index + 1,
                    "window_to": transition_index + 2,
                    "window_pair": f"{transition_index + 1}->{transition_index + 2}",
                    "transition_magnitude": raw_norm,
                    "raw_transition_magnitude": raw_norm,
                    "centered_transition_magnitude": centered_norm,
                    "magnitude_absolute_difference": abs(raw_norm - centered_norm),
                    "subject": subject,
                }
            )

    raw_delta_array = np.stack(raw_deltas)
    centered_delta_array = np.stack(centered_deltas)
    difference = raw_delta_array - centered_delta_array
    max_absolute_error = float(np.max(np.abs(difference)))
    max_vector_error = float(np.max(np.linalg.norm(difference, axis=1)))
    if not np.allclose(
        raw_delta_array,
        centered_delta_array,
        atol=float(invariance_atol),
        rtol=float(invariance_rtol),
    ):
        raise ValueError(
            "RAW and CENTERED consecutive displacements are not invariant; "
            f"maximum coordinate error={max_absolute_error:.6g}"
        )

    transitions = pd.DataFrame.from_records(transition_records)
    class_summary = _descriptive_summary(transitions, "class_label")
    subject_summary = _descriptive_summary(transitions, "subject")

    cosine_records: list[dict[str, Any]] = []
    delta_frame = transitions[["class_label", "window_pair"]].copy()
    delta_frame["_delta_index"] = np.arange(len(delta_frame), dtype=np.int64)
    class_values = sorted(
        pd.unique(transitions["class_label"]),
        key=lambda value: (type(value).__name__, repr(value)),
    )
    pair_scopes = ["all", "1->2", "2->3", "3->4", "4->5"]
    for scope in pair_scopes:
        scope_mask = (
            np.ones(len(delta_frame), dtype=bool)
            if scope == "all"
            else delta_frame["window_pair"].to_numpy() == scope
        )
        means: dict[Any, np.ndarray] = {}
        counts: dict[Any, int] = {}
        for class_label in class_values:
            mask = scope_mask & (delta_frame["class_label"].to_numpy() == class_label)
            selected_delta = raw_delta_array[mask]
            means[class_label] = selected_delta.mean(axis=0)
            counts[class_label] = int(len(selected_delta))
        for class_a, class_b in combinations(class_values, 2):
            vector_a = means[class_a]
            vector_b = means[class_b]
            norm_a = float(np.linalg.norm(vector_a))
            norm_b = float(np.linalg.norm(vector_b))
            cosine = (
                float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
                if norm_a > 0.0 and norm_b > 0.0
                else float("nan")
            )
            cosine_records.append(
                {
                    "window_pair": scope,
                    "class_a": class_a,
                    "class_b": class_b,
                    "n_transitions_a": counts[class_a],
                    "n_transitions_b": counts[class_b],
                    "mean_vector_norm_a": norm_a,
                    "mean_vector_norm_b": norm_b,
                    "cosine_similarity": cosine,
                }
            )
    class_mean_cosine = pd.DataFrame.from_records(cosine_records)

    eta_records: list[dict[str, Any]] = []
    for pair, pair_frame in transitions.groupby("window_pair", sort=True):
        magnitudes = pair_frame["transition_magnitude"].to_numpy(dtype=np.float64)
        for factor in ("class_label", "subject"):
            factor_values = pair_frame[factor].to_numpy()
            eta_records.append(
                {
                    "window_pair": pair,
                    "factor": "class" if factor == "class_label" else "subject",
                    "n_transitions": int(len(pair_frame)),
                    "n_groups": int(len(pd.unique(factor_values))),
                    "eta_squared": _eta_squared(magnitudes, factor_values),
                }
            )
    eta_squared = pd.DataFrame.from_records(eta_records)

    validation = {
        "n_trials": int(len(frame) // 5),
        "n_transitions": int(len(transitions)),
        "n_features": int(raw.shape[1]),
        "windows_per_trial": 5,
        "raw_centered_delta_invariant": True,
        "max_coordinate_absolute_delta_error": max_absolute_error,
        "max_transition_vector_error": max_vector_error,
        "invariance_atol": float(invariance_atol),
        "invariance_rtol": float(invariance_rtol),
    }
    return {
        "validation": validation,
        "per_transition": transitions,
        "class_summary": class_summary,
        "subject_summary": subject_summary,
        "class_mean_cosine": class_mean_cosine,
        "eta_squared": eta_squared,
    }
