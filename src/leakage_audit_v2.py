"""Frozen V1 all-sample versus fold-safe Log-Euclidean leakage audit.

Geometry correctness is an external hard gate.  This module provides a strict
gate reader plus a label-free subject-mean API and the fixed five-fold audit.
No target/evaluation label enters a center-fit function.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.alignment_v2 import (
    LabelFreeMetadataView,
    assert_t2_disjoint,
    label_free_metadata_view,
    trial_uid_sha256,
)
from src.evaluation_v2 import (
    DEFAULT_CLASS_ORDER,
    common_log_svec_features,
    evaluate_target_estimator,
    fit_source_logistic,
    prediction_metrics,
    stable_json_hash,
)
from src.metrics import deterministic_trial_folds


FROZEN_SEED = 20260809
FROZEN_FOLDS = 5
FROZEN_C = 1.0
FROZEN_SOLVER = "lbfgs"
FROZEN_MAX_ITER = 5000
FROZEN_TOL = 1e-4
V1_BENCHMARK_ACCURACY = 0.6119
CONDITIONS = ("v1_all_sample", "fold_safe")


class GeometryGateError(RuntimeError):
    """Raised before classification when the geometry gate is unavailable/failed."""


@dataclass(frozen=True)
class SubjectLogMeans:
    """Means fitted from a label-free view only."""

    subjects: tuple[int, ...]
    means: np.ndarray
    fit_counts: tuple[int, ...]
    fit_trial_uid_sha256: str
    per_subject_trial_uid_sha256: tuple[str, ...]

    def mean_for(self, subject: int) -> np.ndarray:
        try:
            index = self.subjects.index(int(subject))
        except ValueError as error:
            raise ValueError(f"no fitted mean for subject {subject}") from error
        return self.means[index]


@dataclass(frozen=True)
class LeakageAuditResult:
    """In-memory audit outputs; predictions need not be persisted."""

    table: pd.DataFrame
    fold_assignments: np.ndarray
    oof_predictions: Mapping[str, np.ndarray]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload or isinstance(payload[key], bool):
        raise GeometryGateError(f"geometry gate is missing integer {key!r}")
    try:
        value = int(payload[key])
    except (TypeError, ValueError) as error:
        raise GeometryGateError(f"geometry gate {key!r} is not an integer") from error
    if value < 0 or value != payload[key]:
        raise GeometryGateError(f"geometry gate {key!r} must be non-negative")
    return value


def _strict_csv_bool(series: pd.Series, *, name: str) -> pd.Series:
    """Parse only explicit booleans written by the geometry gate producer."""

    def convert(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value in {"True", "False"}:
            return value == "True"
        raise GeometryGateError(
            f"geometry correctness {name!r} must contain only True/False"
        )

    return series.map(convert)


def require_geometry_gate(
    gate_path: str | Path,
    correctness_csv_path: str | Path,
    *,
    expected_protocol_version: str,
    expected_protocol_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Load the fixed JSON gate and refuse anything other than exact ``true``.

    The correctness CSV is also required and must be parseable/non-empty.  This
    function must be called before feature construction or classifier fitting.
    """

    gate_file = Path(gate_path).expanduser().resolve()
    correctness_file = Path(correctness_csv_path).expanduser().resolve()
    if not gate_file.is_file():
        raise GeometryGateError(f"geometry gate JSON is missing: {gate_file}")
    try:
        payload = json.loads(gate_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeometryGateError(f"geometry gate JSON is invalid: {gate_file}") from error
    if not isinstance(payload, dict):
        raise GeometryGateError("geometry gate JSON must contain an object")
    if payload.get("classification_gate_pass") is not True:
        raise GeometryGateError("classification_gate_pass is not exactly true")

    required = _strict_nonnegative_int(payload, "required_rows")
    passed = _strict_nonnegative_int(payload, "passed_required_rows")
    failed = _strict_nonnegative_int(payload, "failed_required_rows")
    if required < 1 or passed != required or failed != 0:
        raise GeometryGateError(
            "geometry gate required-row counts do not encode an all-pass gate"
        )
    expected = {
        "protocol_version": str(expected_protocol_version),
        "protocol_sha256": str(expected_protocol_sha256),
        "config_sha256": str(expected_config_sha256),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise GeometryGateError(
                f"geometry gate {key} mismatch: expected {value!r}, "
                f"observed {payload.get(key)!r}"
            )

    if not correctness_file.is_file():
        raise GeometryGateError(
            f"geometry correctness CSV is missing: {correctness_file}"
        )
    try:
        correctness = pd.read_csv(correctness_file)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise GeometryGateError(
            f"geometry correctness CSV is invalid: {correctness_file}"
        ) from error
    required_columns = {"required", "passed", "status"}
    missing_columns = required_columns.difference(correctness.columns)
    if correctness.empty or missing_columns:
        raise GeometryGateError(
            "geometry correctness CSV is empty or missing columns: "
            f"{sorted(missing_columns)}"
        )
    csv_required = _strict_csv_bool(correctness["required"], name="required")
    csv_passed = _strict_csv_bool(correctness["passed"], name="passed")
    required_frame = correctness.loc[csv_required]
    required_pass = csv_passed.loc[csv_required] & required_frame["status"].eq("PASS")
    if required_frame.empty or not bool(required_pass.all()):
        raise GeometryGateError(
            "one or more required geometry correctness rows did not PASS"
        )
    observed = (int(csv_required.sum()), int(required_pass.sum()))
    if observed != (required, passed):
        raise GeometryGateError(
            "geometry gate JSON/CSV required-row count mismatch: "
            f"observed={observed}, declared={(required, passed)}"
        )
    result = dict(payload)
    result["gate_json_sha256"] = _sha256_file(gate_file)
    result["geometry_correctness_csv_sha256"] = _sha256_file(correctness_file)
    return result


def _finite_features(raw_log_svec: np.ndarray) -> np.ndarray:
    features = np.asarray(raw_log_svec, dtype=np.float64)
    if features.ndim != 2 or min(features.shape) < 1:
        raise ValueError(
            f"raw_log_svec must be a non-empty 2-D array, got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("raw_log_svec contains NaN or Inf")
    return features


def _validate_view_positions(view: LabelFreeMetadataView, n_rows: int) -> np.ndarray:
    positions = np.asarray(view.row_positions, dtype=np.int64)
    if len(positions) != len(view) or len(positions) < 1:
        raise ValueError("center-fit metadata view cannot be empty")
    if len(np.unique(positions)) != len(positions):
        raise ValueError("center-fit metadata view contains duplicate row positions")
    if positions.min() < 0 or positions.max() >= n_rows:
        raise ValueError("center-fit metadata view refers outside the feature array")
    return positions


def fit_subject_log_means(
    raw_log_svec: np.ndarray,
    fit_metadata: LabelFreeMetadataView,
) -> SubjectLogMeans:
    """Fit one coordinate mean per subject without accepting any label argument."""

    features = _finite_features(raw_log_svec)
    positions = _validate_view_positions(fit_metadata, len(features))
    subjects = np.asarray(fit_metadata.subject, dtype=np.int64)
    uids = np.asarray(fit_metadata.trial_uid, dtype=str)
    unique_subjects = tuple(sorted(int(value) for value in np.unique(subjects)))
    means: list[np.ndarray] = []
    counts: list[int] = []
    subject_hashes: list[str] = []
    for subject in unique_subjects:
        mask = subjects == subject
        if not mask.any():
            raise RuntimeError("internal subject-mean selection is empty")
        means.append(features[positions[mask]].mean(axis=0))
        counts.append(int(mask.sum()))
        subject_hashes.append(trial_uid_sha256(uids[mask]))
    mean_array = np.stack(means).astype(np.float64, copy=False)
    if not np.isfinite(mean_array).all():
        raise FloatingPointError("a fitted subject log-mean is non-finite")
    mean_array.setflags(write=False)
    return SubjectLogMeans(
        subjects=unique_subjects,
        means=mean_array,
        fit_counts=tuple(counts),
        fit_trial_uid_sha256=trial_uid_sha256(fit_metadata.trial_uid),
        per_subject_trial_uid_sha256=tuple(subject_hashes),
    )


def apply_subject_log_means(
    raw_log_svec: np.ndarray,
    apply_metadata: LabelFreeMetadataView,
    fitted_means: SubjectLogMeans,
) -> np.ndarray:
    """Subtract fitted means from rows in the label-free view's canonical order."""

    features = _finite_features(raw_log_svec)
    positions = _validate_view_positions(apply_metadata, len(features))
    centered = np.empty((len(positions), features.shape[1]), dtype=np.float64)
    for row, (position, subject) in enumerate(
        zip(positions, apply_metadata.subject, strict=True)
    ):
        centered[row] = features[position] - fitted_means.mean_for(subject)
    if not np.isfinite(centered).all():
        raise FloatingPointError("centered coordinates contain NaN or Inf")
    return centered


def _validated_metadata(metadata: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame) or len(metadata) != n_rows:
        raise ValueError(f"metadata must contain exactly {n_rows} rows")
    required = {
        "subject",
        "session",
        "run",
        "trial_id",
        "trial_uid",
        "class_label",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    frame = metadata.reset_index(drop=True).copy()
    if frame[list(required)].isna().any(axis=None):
        raise ValueError("audit metadata cannot contain null identity/label values")
    frame["subject"] = pd.to_numeric(frame["subject"], errors="raise").astype(int)
    frame["trial_id"] = pd.to_numeric(frame["trial_id"], errors="raise").astype(int)
    frame["run"] = frame["run"].astype(str)
    frame["session"] = frame["session"].astype(str)
    frame["trial_uid"] = frame["trial_uid"].astype(str)
    frame["class_label"] = frame["class_label"].astype(str)
    if frame["trial_uid"].duplicated().any():
        raise ValueError("trial_uid must be globally unique for the WHOLE audit")
    if set(frame["class_label"].unique()) != set(DEFAULT_CLASS_ORDER):
        raise ValueError("metadata does not contain the frozen four-class vocabulary")
    return frame


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _uid_hash_for_fold_hashes(fold_hashes: Mapping[int, str]) -> str:
    return stable_json_hash(
        {"fold_center_fit_uid_sha256": {str(key): value for key, value in fold_hashes.items()}}
    )


def _metric_columns(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items()}


def _run_condition(
    raw_features: np.ndarray,
    metadata: pd.DataFrame,
    folds: np.ndarray,
    *,
    condition: str,
    class_order: Sequence[str],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown leakage-audit condition: {condition}")
    all_positions = np.arange(len(metadata), dtype=np.int64)
    all_view = label_free_metadata_view(metadata, all_positions)
    all_means = (
        fit_subject_log_means(raw_features, all_view)
        if condition == "v1_all_sample"
        else None
    )
    labels = metadata["class_label"].to_numpy(dtype=str)
    oof_prediction = np.empty(len(metadata), dtype=object)
    predicted_mask = np.zeros(len(metadata), dtype=bool)
    records: list[dict[str, Any]] = []
    fit_hashes: dict[int, str] = {}
    train_hashes: dict[int, str] = {}
    evaluation_hashes: dict[int, str] = {}
    fit_uids_by_fold: dict[int, set[str]] = {}
    evaluation_uids_by_fold: dict[int, set[str]] = {}
    warning_messages: list[str] = []
    any_convergence_warning = False
    n_iter_max = 0
    classifier_hashes: set[str] = set()

    for fold in range(FROZEN_FOLDS):
        evaluation_positions = np.flatnonzero(folds == fold)
        training_positions = np.flatnonzero(folds != fold)
        if len(evaluation_positions) == 0 or len(training_positions) == 0:
            raise RuntimeError(f"fold {fold} has an empty train/evaluation partition")
        training_view = label_free_metadata_view(metadata, training_positions)
        evaluation_view = label_free_metadata_view(metadata, evaluation_positions)
        if condition == "v1_all_sample":
            center_fit_view = all_view
            assert all_means is not None
            means = all_means
            transductive = True
            center_fit_scope = "all_288_per_subject"
        else:
            center_fit_view = training_view
            means = fit_subject_log_means(raw_features, center_fit_view)
            transductive = False
            center_fit_scope = "fold_training_rows_per_subject"

        center_uids = set(center_fit_view.trial_uid)
        training_uids = set(training_view.trial_uid)
        evaluation_uids = set(evaluation_view.trial_uid)
        train_eval_overlap = training_uids & evaluation_uids
        if train_eval_overlap:
            raise AssertionError("classifier training/evaluation trial UIDs overlap")
        center_eval_overlap = center_uids & evaluation_uids
        if condition == "v1_all_sample":
            if center_eval_overlap != evaluation_uids:
                raise AssertionError(
                    "v1_all_sample center fit must contain every evaluation trial"
                )
        else:
            assert_t2_disjoint(center_uids, evaluation_uids)

        training_features = apply_subject_log_means(
            raw_features, training_view, means
        )
        evaluation_features = apply_subject_log_means(
            raw_features, evaluation_view, means
        )
        training_labels = labels[np.asarray(training_view.row_positions, dtype=int)]
        evaluation_labels = labels[
            np.asarray(evaluation_view.row_positions, dtype=int)
        ]
        estimator, fit_audit = fit_source_logistic(
            training_features,
            training_labels,
            c=FROZEN_C,
            solver=FROZEN_SOLVER,
            max_iter=FROZEN_MAX_ITER,
            tol=FROZEN_TOL,
            random_state=FROZEN_SEED,
        )
        prediction, metrics = evaluate_target_estimator(
            estimator,
            evaluation_features,
            evaluation_labels,
            class_order=class_order,
        )
        evaluation_rows = np.asarray(evaluation_view.row_positions, dtype=int)
        if predicted_mask[evaluation_rows].any():
            raise AssertionError("an OOF row received more than one prediction")
        oof_prediction[evaluation_rows] = prediction
        predicted_mask[evaluation_rows] = True

        fit_hashes[fold] = means.fit_trial_uid_sha256
        train_hashes[fold] = trial_uid_sha256(training_view.trial_uid)
        evaluation_hashes[fold] = trial_uid_sha256(evaluation_view.trial_uid)
        fit_uids_by_fold[fold] = center_uids
        evaluation_uids_by_fold[fold] = evaluation_uids
        classifier_hashes.add(fit_audit.config_hash)
        any_convergence_warning |= fit_audit.convergence_warning
        warning_messages.extend(fit_audit.warning_messages)
        n_iter_max = max(n_iter_max, int(fit_audit.n_iter_max or 0))

        record: dict[str, Any] = {
            "condition": condition,
            "row_type": "fold",
            "fold": fold,
            "n_folds": FROZEN_FOLDS,
            "seed": FROZEN_SEED,
            "space": f"whole_raw_log_svec_unscaled_{raw_features.shape[1]}d",
            "standard_scaler": False,
            "center_api_label_free": True,
            "center_fit_scope": center_fit_scope,
            "center_fit_n": len(center_fit_view),
            "center_fit_n_unique": len(center_uids),
            "center_fit_exposures_n": len(center_fit_view),
            "center_fit_min_per_subject": min(means.fit_counts),
            "center_fit_max_per_subject": max(means.fit_counts),
            "center_fit_trial_uid_sha256": means.fit_trial_uid_sha256,
            "center_fit_hash_kind": "single_uid_set",
            "center_fit_fold_hashes_json": _json({str(fold): fit_hashes[fold]}),
            "classifier_train_n": len(training_view),
            "classifier_train_n_unique": len(training_uids),
            "classifier_train_exposures_n": len(training_view),
            "classifier_train_trial_uid_sha256": train_hashes[fold],
            "classifier_train_hash_kind": "single_uid_set",
            "classifier_train_fold_hashes_json": _json(
                {str(fold): train_hashes[fold]}
            ),
            "evaluation_n": len(evaluation_view),
            "evaluation_trial_uid_sha256": evaluation_hashes[fold],
            "center_evaluation_overlap_n": len(center_eval_overlap),
            "train_evaluation_overlap_n": len(train_eval_overlap),
            "train_evaluation_disjoint": len(train_eval_overlap) == 0,
            "transductive_covariate_overlap": transductive,
            "classifier": "multinomial_l2_logistic_regression",
            "classifier_config_sha256": fit_audit.config_hash,
            "classifier_status": (
                "FAILED_CONVERGENCE" if fit_audit.convergence_warning else "PASS"
            ),
            "convergence_warning": fit_audit.convergence_warning,
            "warning_messages_json": _json(list(fit_audit.warning_messages)),
            "n_iter_max": fit_audit.n_iter_max,
            "class_order_json": _json(list(class_order)),
            "original_v1_benchmark_accuracy": V1_BENCHMARK_ACCURACY,
            "actual_accuracy_difference_from_benchmark": np.nan,
            **_metric_columns(metrics),
        }
        records.append(record)

    if not predicted_mask.all():
        missing = np.flatnonzero(~predicted_mask)[:10]
        raise AssertionError(f"OOF predictions are incomplete; missing rows {missing}")
    if len(classifier_hashes) != 1:
        raise AssertionError("frozen classifier configuration changed between folds")

    oof_prediction = oof_prediction.astype(str)
    pooled_metrics = prediction_metrics(
        labels,
        oof_prediction,
        class_order=class_order,
    )
    all_uid_set = set(metadata["trial_uid"].astype(str))
    center_union = set().union(*fit_uids_by_fold.values())
    eval_union = set().union(*evaluation_uids_by_fold.values())
    if eval_union != all_uid_set or center_union != all_uid_set:
        raise AssertionError("aggregate center/evaluation UID union is incomplete")
    overlap_exposures = sum(
        len(fit_uids_by_fold[fold] & evaluation_uids_by_fold[fold])
        for fold in range(FROZEN_FOLDS)
    )
    center_exposures = sum(len(value) for value in fit_uids_by_fold.values())
    train_exposures = sum(int((folds != fold).sum()) for fold in range(FROZEN_FOLDS))
    all_uid_hash = trial_uid_sha256(metadata["trial_uid"].astype(str))
    pooled_record: dict[str, Any] = {
        "condition": condition,
        "row_type": "pooled_oof",
        "fold": pd.NA,
        "n_folds": FROZEN_FOLDS,
        "seed": FROZEN_SEED,
        "space": f"whole_raw_log_svec_unscaled_{raw_features.shape[1]}d",
        "standard_scaler": False,
        "center_api_label_free": True,
        "center_fit_scope": (
            "all_288_per_subject"
            if condition == "v1_all_sample"
            else "fold_training_rows_per_subject"
        ),
        "center_fit_n": len(center_union),
        "center_fit_n_unique": len(center_union),
        "center_fit_exposures_n": center_exposures,
        "center_fit_min_per_subject": min(
            record["center_fit_min_per_subject"] for record in records
        ),
        "center_fit_max_per_subject": max(
            record["center_fit_max_per_subject"] for record in records
        ),
        "center_fit_trial_uid_sha256": (
            all_uid_hash
            if condition == "v1_all_sample"
            else _uid_hash_for_fold_hashes(fit_hashes)
        ),
        "center_fit_hash_kind": (
            "single_uid_set"
            if condition == "v1_all_sample"
            else "sha256_of_fold_uid_hash_mapping"
        ),
        "center_fit_fold_hashes_json": _json(
            {str(key): value for key, value in fit_hashes.items()}
        ),
        "classifier_train_n": len(all_uid_set),
        "classifier_train_n_unique": len(all_uid_set),
        "classifier_train_exposures_n": train_exposures,
        "classifier_train_trial_uid_sha256": _uid_hash_for_fold_hashes(train_hashes),
        "classifier_train_hash_kind": "sha256_of_fold_uid_hash_mapping",
        "classifier_train_fold_hashes_json": _json(
            {str(key): value for key, value in train_hashes.items()}
        ),
        "evaluation_n": len(eval_union),
        "evaluation_trial_uid_sha256": all_uid_hash,
        "center_evaluation_overlap_n": overlap_exposures,
        "train_evaluation_overlap_n": 0,
        "train_evaluation_disjoint": True,
        "transductive_covariate_overlap": condition == "v1_all_sample",
        "classifier": "multinomial_l2_logistic_regression",
        "classifier_config_sha256": next(iter(classifier_hashes)),
        "classifier_status": (
            "FAILED_CONVERGENCE" if any_convergence_warning else "PASS"
        ),
        "convergence_warning": any_convergence_warning,
        "warning_messages_json": _json(sorted(set(warning_messages))),
        "n_iter_max": n_iter_max,
        "class_order_json": _json(list(class_order)),
        "original_v1_benchmark_accuracy": V1_BENCHMARK_ACCURACY,
        "actual_accuracy_difference_from_benchmark": (
            float(pooled_metrics["accuracy"]) - V1_BENCHMARK_ACCURACY
        ),
        **_metric_columns(pooled_metrics),
    }
    records.append(pooled_record)
    return records, oof_prediction


def run_v1_leakage_audit_from_features(
    raw_log_svec: np.ndarray,
    metadata: pd.DataFrame,
) -> LeakageAuditResult:
    """Run both frozen centering conditions from already-computed raw coordinates."""

    features = _finite_features(raw_log_svec)
    frame = _validated_metadata(metadata, len(features))
    folds = deterministic_trial_folds(
        frame,
        n_splits=FROZEN_FOLDS,
        seed=FROZEN_SEED,
    )
    if set(folds.tolist()) != set(range(FROZEN_FOLDS)):
        raise AssertionError("deterministic fold assignment omitted a frozen fold")
    all_records: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        records, oof = _run_condition(
            features,
            frame,
            folds,
            condition=condition,
            class_order=DEFAULT_CLASS_ORDER,
        )
        all_records.extend(records)
        predictions[condition] = oof
    table = pd.DataFrame.from_records(all_records)
    table["fold"] = table["fold"].astype("Int64")
    fold_array = np.asarray(folds, dtype=np.int64)
    fold_array.setflags(write=False)
    for prediction in predictions.values():
        prediction.setflags(write=False)
    return LeakageAuditResult(
        table=table,
        fold_assignments=fold_array,
        oof_predictions=predictions,
    )


def run_v1_leakage_audit(
    whole_covariances: np.ndarray,
    metadata: pd.DataFrame,
) -> LeakageAuditResult:
    """Compute WHOLE raw log-svec coordinates, then run the frozen audit."""

    raw_features = common_log_svec_features(whole_covariances)
    return run_v1_leakage_audit_from_features(raw_features, metadata)
