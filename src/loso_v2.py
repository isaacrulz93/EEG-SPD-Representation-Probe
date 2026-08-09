"""Preregistered LOSO evaluation for the WHOLE-SPD geometry audit V2.

The module keeps three boundaries explicit:

* marginal centers are fitted from covariance matrices only;
* decoders are fitted from source arrays and source labels only; and
* target labels cross the boundary only in ``evaluate_target_estimator``.

No function in this module tunes a classifier or changes the frozen protocol.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.alignment_v2 import (
    CalibrationSplit,
    FROZEN_SUBJECTS,
    LosoPartition,
    assert_t1_overlap,
    assert_t2_disjoint,
    label_free_metadata_view,
    make_calibration_splits,
    make_loso_partition,
    make_sample_id_audit_rows,
    trial_uid_sha256,
)
from src.evaluation_v2 import (
    DEFAULT_CLASS_ORDER,
    FitAudit,
    array_sha256,
    common_log_svec_features,
    evaluate_target_estimator,
    fit_source_logistic,
    fit_source_mdm,
    stable_json_hash,
)
from src.geometry_v2 import (
    AIRM,
    EA,
    GEOMETRIES,
    LE,
    RAW,
    FittedCenter,
    airm_distance,
    airm_mean,
    arithmetic_mean,
    fit_center,
    logeuclidean_distance,
    logeuclidean_mean_custom,
)
from src.metrics import rms_distance_ratio


MDM_SPECS: tuple[tuple[str, str], ...] = (
    (RAW, "riemann"),
    (RAW, "logeuclid"),
    (LE, "logeuclid"),
    (AIRM, "riemann"),
    (EA, "riemann"),
)

_CONFUSION_COLUMNS = tuple(
    f"confusion_{truth}__{prediction}"
    for truth in DEFAULT_CLASS_ORDER
    for prediction in DEFAULT_CLASS_ORDER
)
_RECALL_COLUMNS = tuple(f"recall_{label}" for label in DEFAULT_CLASS_ORDER)

RESULT_COLUMNS = (
    "protocol_version",
    "config_sha256",
    "seed",
    "subject",
    "target_subject",
    "source_subjects",
    "geometry",
    "protocol",
    "split",
    "decoder",
    "metric",
    "native_metric",
    "source_n",
    "source_center_fit_n_per_subject",
    "source_center_fit_n_total",
    "calibration_n",
    "target_center_fit_n",
    "target_center_fit_instances",
    "evaluation_n",
    "source_trial_uid_hash",
    "fit_trial_uid_hash",
    "calibration_trial_uid_hash",
    "evaluation_trial_uid_hash",
    "source_target_overlap_n",
    "center_evaluation_overlap_n",
    "transductive_overlap",
    "source_feature_sha256",
    "source_label_sha256",
    "target_input_sha256",
    "feature_config_sha256",
    "model_config_sha256",
    "fitted_model_sha256",
    "prediction_sha256",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    *_RECALL_COLUMNS,
    "confusion_matrix_json",
    *_CONFUSION_COLUMNS,
    "convergence_warning",
    "warning_messages",
    "status",
    "primary_run_status",
    "primary_failure_count",
    "secondary_run_status",
    "secondary_failure_count",
)

DOMAIN_COLUMNS = (
    "protocol_version",
    "config_sha256",
    "seed",
    "subject",
    "target_subject",
    "geometry",
    "protocol",
    "split",
    "reference_metric",
    "source_n",
    "calibration_n",
    "evaluation_n",
    "source_trial_uid_hash",
    "calibration_trial_uid_hash",
    "evaluation_trial_uid_hash",
    "transductive_overlap",
    "source_transformed_sha256",
    "target_transformed_sha256",
    "source_mean_to_identity",
    "target_mean_to_identity",
    "source_target_mean_distance",
    "source_rms_around_identity",
    "target_rms_around_identity",
    "source_rms_around_own_mean",
    "target_rms_around_own_mean",
    "target_minus_source_dispersion",
    "absolute_dispersion_difference",
    "all_subject_subject_silhouette",
    "all_subject_subject_between_within_rms_ratio",
    "uses_class_labels",
    "status",
)


class ClassificationGateError(RuntimeError):
    """Raised when the geometry correctness gate does not authorize fitting."""


class V2OutputContractError(RuntimeError):
    """Raised before a script could write outside the frozen V2 output root."""


@dataclass(frozen=True)
class TargetTransform:
    """One label-free target transformation and its fitted center."""

    covariances: np.ndarray
    center: FittedCenter


@dataclass(frozen=True)
class LosoV2Result:
    """All logical LOSO tables, including an auditable fatal-warning state."""

    logistic_transductive: pd.DataFrame
    logistic_calibration: pd.DataFrame
    mdm_transductive: pd.DataFrame
    mdm_calibration: pd.DataFrame
    domain_shift_diagnostics: pd.DataFrame
    sample_id_audit: pd.DataFrame
    fatal_error: str | None = None
    primary_failure_count: int = 0
    primary_status: str = "PASS"
    secondary_failure_count: int = 0
    secondary_status: str = "PASS"

    @property
    def classification_failed(self) -> bool:
        return self.fatal_error is not None or self.primary_status == "FAILED"


@dataclass(frozen=True)
class _DomainComponents:
    mean: np.ndarray
    mean_to_identity: float
    rms_around_identity: float
    rms_around_own_mean: float


@dataclass(frozen=True)
class _FixedPredictionEstimator:
    """Expose already-generated split predictions through estimator.predict."""

    predictions: np.ndarray

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if len(inputs) != len(self.predictions):
            raise ValueError("fixed prediction/input row counts differ")
        return self.predictions.copy()


def _strict_csv_bool(series: pd.Series, *, name: str) -> pd.Series:
    def convert(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value in {"True", "False"}:
            return value == "True"
        raise ClassificationGateError(
            f"{name} must contain only explicit True/False values"
        )

    return series.map(convert)


def assert_v2_output_contract(
    project_root: str | Path, config: Mapping[str, Any]
) -> Path:
    """Resolve the exact V2 tables directory and reject every V1 write path."""

    root = Path(project_root).expanduser().resolve()

    def resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    try:
        configured_output = resolve(config["project"]["output_dir"])
        configured_tables = resolve(config["outputs"]["tables_dir"])
    except (KeyError, TypeError) as error:
        raise V2OutputContractError("V2 output path configuration is incomplete") from error
    expected_output = (root / "outputs" / "bnci2014_001_geometry_v2").resolve()
    expected_tables = (expected_output / "tables").resolve()
    v1_output = (root / "outputs" / "bnci2014_001").resolve()
    if configured_output != expected_output or configured_tables != expected_tables:
        raise V2OutputContractError(
            "LOSO outputs must resolve exactly under "
            f"{expected_output}; observed output={configured_output}, "
            f"tables={configured_tables}"
        )
    if configured_tables == v1_output or v1_output in configured_tables.parents:
        raise V2OutputContractError("refusing to write a LOSO artifact into V1 outputs")
    return configured_tables


def assert_classification_gate(
    tables_dir: str | Path,
    *,
    expected_protocol_version: str,
    expected_protocol_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Require an exact PASS geometry gate before any classifier can run."""

    directory = Path(tables_dir).expanduser().resolve()
    gate_path = directory / "geometry_gate.json"
    correctness_path = directory / "geometry_correctness.csv"
    if not gate_path.is_file() or not correctness_path.is_file():
        missing = [
            str(path)
            for path in (gate_path, correctness_path)
            if not path.is_file()
        ]
        raise ClassificationGateError(
            "geometry classification gate artifact is missing: " + ", ".join(missing)
        )
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClassificationGateError(f"could not read {gate_path}") from error
    if not isinstance(gate, dict) or gate.get("classification_gate_pass") is not True:
        raise ClassificationGateError(
            "classification_gate_pass must be the JSON boolean true exactly"
        )
    expected_provenance = {
        "protocol_version": str(expected_protocol_version),
        "protocol_sha256": str(expected_protocol_sha256),
        "config_sha256": str(expected_config_sha256),
    }
    for key, expected in expected_provenance.items():
        if gate.get(key) != expected:
            raise ClassificationGateError(
                f"geometry gate provenance mismatch for {key}: "
                f"{gate.get(key)!r} != {expected!r}"
            )
    try:
        correctness = pd.read_csv(correctness_path)
    except (OSError, pd.errors.ParserError) as error:
        raise ClassificationGateError(f"could not read {correctness_path}") from error
    required_columns = {"required", "passed", "status"}
    missing_columns = required_columns - set(correctness.columns)
    if correctness.empty or missing_columns:
        raise ClassificationGateError(
            "geometry_correctness.csv is empty or missing columns: "
            f"{sorted(missing_columns)}"
        )
    required = _strict_csv_bool(correctness["required"], name="required")
    passed = _strict_csv_bool(correctness["passed"], name="passed")
    required_rows = correctness.loc[required]
    required_pass = passed.loc[required] & required_rows["status"].eq("PASS")
    if required_rows.empty or not bool(required_pass.all()):
        raise ClassificationGateError(
            "one or more required geometry correctness rows did not PASS"
        )
    expected_counts = {
        "n_required_rows": int(required.sum()),
        "n_required_passed": int(required_pass.sum()),
        "n_required_failed": 0,
    }
    for key, expected in expected_counts.items():
        if key in gate and gate[key] != expected:
            raise ClassificationGateError(
                f"geometry gate/count mismatch for {key}: {gate[key]!r} != {expected}"
            )
    return gate


def _positions(values: Sequence[int], n_rows: int, *, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    result = result.astype(np.int64, copy=False)
    if len(result) == 0 or len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and unique")
    if result.min() < 0 or result.max() >= n_rows:
        raise ValueError(f"{name} contains an out-of-range row")
    return result


def fit_target_transform(
    covariances: np.ndarray,
    *,
    fit_positions: Sequence[int],
    evaluation_positions: Sequence[int],
    geometry: str,
    tol: float = 1e-9,
    maxiter: int = 100,
) -> TargetTransform:
    """Fit a target marginal transform from covariates and transform eval rows.

    The deliberately narrow signature has no metadata, label, or target-label
    argument.  UID overlap/disjointness is asserted by ``alignment_v2`` before
    this numerical function is called.
    """

    matrices = np.asarray(covariances, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("covariances must have shape (trials, channels, channels)")
    fitted = _positions(fit_positions, len(matrices), name="fit_positions")
    evaluated = _positions(
        evaluation_positions, len(matrices), name="evaluation_positions"
    )
    center = fit_center(
        matrices[fitted], geometry, tol=float(tol), maxiter=int(maxiter)
    )
    transformed = center.transform(matrices[evaluated])
    return TargetTransform(covariances=transformed, center=center)


def _fitted_model_sha256(model: Any) -> Any:
    if model is None:
        return pd.NA
    payload: dict[str, Any] = {"type": type(model).__name__}
    arrays: dict[str, str] = {}
    for name in ("coef_", "intercept_", "covmeans_"):
        if hasattr(model, name):
            arrays[name] = array_sha256(np.asarray(getattr(model, name)))
    payload["arrays"] = arrays
    if hasattr(model, "classes_"):
        payload["classes"] = [str(value) for value in model.classes_]
    return stable_json_hash(payload)


def _empty_result() -> dict[str, list[dict[str, Any]]]:
    return {"log_t1": [], "log_t2": [], "mdm_t1": [], "mdm_t2": []}


def _frame(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


def _finalize(
    rows: Mapping[str, list[dict[str, Any]]],
    domain_rows: list[dict[str, Any]],
    audit_rows: list[pd.DataFrame],
    *,
    fatal_error: str | None,
    primary_failure_count: int,
    secondary_failure_count: int,
    secondary_status: str,
) -> LosoV2Result:
    if secondary_status not in {"PASS", "FAILED", "NOT_RUN"}:
        raise ValueError(f"invalid secondary status: {secondary_status!r}")
    primary_status = (
        "FAILED"
        if fatal_error is not None or int(primary_failure_count) > 0
        else "PASS"
    )
    result_frames = {
        key: _frame(rows[key], RESULT_COLUMNS)
        for key in ("log_t1", "log_t2", "mdm_t1", "mdm_t2")
    }
    for frame in result_frames.values():
        frame["primary_run_status"] = primary_status
        frame["primary_failure_count"] = int(primary_failure_count)
        frame["secondary_run_status"] = secondary_status
        frame["secondary_failure_count"] = int(secondary_failure_count)
    return LosoV2Result(
        logistic_transductive=result_frames["log_t1"],
        logistic_calibration=result_frames["log_t2"],
        mdm_transductive=result_frames["mdm_t1"],
        mdm_calibration=result_frames["mdm_t2"],
        domain_shift_diagnostics=_frame(domain_rows, DOMAIN_COLUMNS),
        sample_id_audit=(
            pd.concat(audit_rows, ignore_index=True)
            if audit_rows
            else pd.DataFrame()
        ),
        fatal_error=fatal_error,
        primary_failure_count=int(primary_failure_count),
        primary_status=primary_status,
        secondary_failure_count=int(secondary_failure_count),
        secondary_status=secondary_status,
    )


def _configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {"protocol", "dataset", "geometry", "evaluation", "classifiers"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"LOSO config is missing sections: {sorted(missing)}")
    result = dict(config)
    configured_classes = tuple(str(x) for x in result["dataset"]["classes"])
    metric_classes = tuple(
        str(x) for x in result["classifiers"]["metrics"]["class_order"]
    )
    if configured_classes != DEFAULT_CLASS_ORDER or metric_classes != DEFAULT_CLASS_ORDER:
        raise ValueError(f"class order must be exactly {DEFAULT_CLASS_ORDER}")
    return result


def _subject_positions(
    metadata: pd.DataFrame, subjects: Sequence[int]
) -> tuple[dict[int, LosoPartition], dict[int, np.ndarray]]:
    partitions: dict[int, LosoPartition] = {}
    positions: dict[int, np.ndarray] = {}
    for subject in subjects:
        partition = make_loso_partition(metadata, int(subject))
        partitions[int(subject)] = partition
        positions[int(subject)] = np.asarray(
            partition.target_row_positions, dtype=np.int64
        )
    return partitions, positions


def _full_subject_transforms(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    positions: Mapping[int, np.ndarray],
    geometries: Sequence[str],
    *,
    tol: float,
    maxiter: int,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], FittedCenter]]:
    transformed: dict[tuple[str, int], np.ndarray] = {}
    centers: dict[tuple[str, int], FittedCenter] = {}
    for geometry in geometries:
        for subject in sorted(positions):
            selected = positions[subject]
            # Executable proof that the center-fit metadata view carries no labels.
            label_free_metadata_view(metadata, selected)
            center = fit_center(
                covariances[selected], geometry, tol=tol, maxiter=maxiter
            )
            transformed[(geometry, subject)] = center.transform(
                covariances[selected]
            )
            centers[(geometry, subject)] = center
    return transformed, centers


def _source_covariance_pool(
    partition: LosoPartition,
    geometry: str,
    subject_positions: Mapping[int, np.ndarray],
    full_transforms: Mapping[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.concatenate(
        [subject_positions[subject] for subject in partition.source_subjects]
    )
    expected = np.asarray(partition.source_row_positions, dtype=np.int64)
    if not np.array_equal(positions, expected):
        raise AssertionError("source pool order changed from canonical LOSO identity")
    matrices = np.concatenate(
        [full_transforms[(geometry, subject)] for subject in partition.source_subjects],
        axis=0,
    )
    return matrices, positions


def _source_pool(
    partition: LosoPartition,
    geometry: str,
    subject_positions: Mapping[int, np.ndarray],
    full_transforms: Mapping[tuple[str, int], np.ndarray],
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices, positions = _source_covariance_pool(
        partition, geometry, subject_positions, full_transforms
    )
    labels = metadata.iloc[positions]["class_label"].astype(str).to_numpy()
    return matrices, labels, positions


def _target_transforms(
    covariances: np.ndarray,
    partition: LosoPartition,
    splits: Sequence[CalibrationSplit],
    geometry: str,
    *,
    tol: float,
    maxiter: int,
    t1_state: TargetTransform | None = None,
) -> dict[str, TargetTransform]:
    assert_t1_overlap(partition.target_trial_uids, partition.target_trial_uids)
    result = {
        "ALL": (
            t1_state
            if t1_state is not None
            else fit_target_transform(
                covariances,
                fit_positions=partition.target_row_positions,
                evaluation_positions=partition.target_row_positions,
                geometry=geometry,
                tol=tol,
                maxiter=maxiter,
            )
        )
    }
    if result["ALL"].center.geometry != geometry:
        raise AssertionError("precomputed T1 target geometry does not match")
    if len(result["ALL"].covariances) != partition.n_target_trials:
        raise AssertionError("precomputed T1 target row count does not match")
    for split in splits:
        assert_t2_disjoint(
            split.calibration_trial_uids, split.evaluation_trial_uids
        )
        result[split.name] = fit_target_transform(
            covariances,
            fit_positions=split.calibration_row_positions,
            evaluation_positions=split.evaluation_row_positions,
            geometry=geometry,
            tol=tol,
            maxiter=maxiter,
        )
    return result


def _identity_fields(
    partition: LosoPartition,
    geometry: str,
    protocol: str,
    split: CalibrationSplit | None,
    *,
    aggregate: bool = False,
) -> dict[str, Any]:
    centered = geometry != RAW
    if protocol == "T1":
        evaluation_hash = partition.target_trial_uid_sha256
        fit_hash = partition.target_trial_uid_sha256 if centered else "not_applicable"
        calibration_hash = fit_hash
        calibration_n = 288 if centered else 0
        fit_n = 288 if centered else 0
        fit_instances = 1 if centered else 0
        overlap: bool | str = True if centered else "not_applicable"
        overlap_n = 288 if centered else 0
        split_name = "ALL"
        evaluation_n = 288
    elif aggregate:
        evaluation_hash = partition.target_trial_uid_sha256
        calibration_hash = partition.target_trial_uid_sha256
        fit_hash = partition.target_trial_uid_sha256 if centered else "not_applicable"
        calibration_n = 288
        fit_n = 288 if centered else 0
        fit_instances = 2 if centered else 0
        overlap = False if centered else "not_applicable"
        overlap_n = 0
        split_name = "AGGREGATE"
        evaluation_n = 288
    else:
        if split is None:
            raise ValueError("T2 identity fields require a calibration split")
        evaluation_hash = split.evaluation_trial_uid_sha256
        calibration_hash = split.calibration_trial_uid_sha256
        fit_hash = calibration_hash if centered else "not_applicable"
        calibration_n = split.n_calibration_trials
        fit_n = split.n_calibration_trials if centered else 0
        fit_instances = 1 if centered else 0
        overlap = False if centered else "not_applicable"
        overlap_n = 0
        split_name = split.name
        evaluation_n = split.n_evaluation_trials
    return {
        "protocol": protocol,
        "split": split_name,
        "calibration_n": calibration_n,
        "target_center_fit_n": fit_n,
        "target_center_fit_instances": fit_instances,
        "evaluation_n": evaluation_n,
        "fit_trial_uid_hash": fit_hash,
        "calibration_trial_uid_hash": calibration_hash,
        "evaluation_trial_uid_hash": evaluation_hash,
        "center_evaluation_overlap_n": overlap_n,
        "transductive_overlap": overlap,
    }


def _base_result_row(
    config: Mapping[str, Any],
    config_hash: str,
    partition: LosoPartition,
    geometry: str,
    decoder: str,
    native_metric: str,
    identity: Mapping[str, Any],
    source_feature_hash: str,
    source_label_hash: str,
    target_input_hash: str,
    feature_config_hash: str,
    model: Any,
    audit: FitAudit,
) -> dict[str, Any]:
    source_center_n = 0 if geometry == RAW else 288
    row: dict[str, Any] = {
        "protocol_version": str(config["protocol"]["version"]),
        "config_sha256": config_hash,
        "seed": int(config["protocol"]["seed"]),
        "subject": partition.target_subject,
        "target_subject": partition.target_subject,
        "source_subjects": json.dumps(list(partition.source_subjects), separators=(",", ":")),
        "geometry": geometry,
        "decoder": decoder,
        "metric": native_metric,
        "native_metric": native_metric,
        "source_n": partition.n_source_trials,
        "source_center_fit_n_per_subject": source_center_n,
        "source_center_fit_n_total": source_center_n * len(partition.source_subjects),
        "source_trial_uid_hash": partition.source_trial_uid_sha256,
        "source_target_overlap_n": 0,
        "source_feature_sha256": source_feature_hash,
        "source_label_sha256": source_label_hash,
        "target_input_sha256": target_input_hash,
        "feature_config_sha256": feature_config_hash,
        "model_config_sha256": audit.config_hash,
        "fitted_model_sha256": _fitted_model_sha256(model),
        "convergence_warning": bool(audit.convergence_warning),
        "warning_messages": json.dumps(
            list(audit.warning_messages), ensure_ascii=False, separators=(",", ":")
        ),
        "status": "FAILED" if audit.convergence_warning else "PASS",
    }
    row.update(identity)
    return row


def _metric_row(base: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(base)
    row.update(metrics)
    if int(row["evaluation_n"]) != int(metrics["n_evaluation"]):
        raise AssertionError("identity and prediction evaluation counts differ")
    row.pop("n_evaluation", None)
    return row


def _failed_row(base: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(base)
    for name in (
        "prediction_sha256",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        *_RECALL_COLUMNS,
        "confusion_matrix_json",
        *_CONFUSION_COLUMNS,
    ):
        row[name] = pd.NA
    row["status"] = "FAILED"
    return row


def _mdm_failure_audit(
    metric: str,
    source_n: int,
    warning_messages: Sequence[str],
    *,
    convergence_warning: bool,
) -> FitAudit:
    parameters = {"decoder": "MDM", "metric": metric, "n_jobs": 1}
    return FitAudit(
        decoder=f"mdm_{metric}",
        config_hash=stable_json_hash(parameters),
        n_train=int(source_n),
        n_features=None,
        convergence_warning=bool(convergence_warning),
        warning_messages=tuple(str(value) for value in warning_messages),
        n_iter_max=None,
    )


def _failed_logistic_condition_rows(
    config: Mapping[str, Any],
    config_hash: str,
    partition: LosoPartition,
    geometry: str,
    splits: Sequence[CalibrationSplit],
    states: Mapping[str, TargetTransform],
    source_feature_hash: str,
    source_label_hash: str,
    feature_config_hash: str,
    model: Any,
    audit: FitAudit,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Materialize all four rows without scoring a nonconverged model."""

    t1_inputs = common_log_svec_features(states["ALL"].covariances)
    t1_identity = _identity_fields(partition, geometry, "T1", None)
    t1_base = _base_result_row(
        config,
        config_hash,
        partition,
        geometry,
        "logistic",
        "euclidean_log_svec",
        t1_identity,
        source_feature_hash,
        source_label_hash,
        array_sha256(t1_inputs),
        feature_config_hash,
        model,
        audit,
    )
    t2_rows: list[dict[str, Any]] = []
    split_inputs: list[np.ndarray] = []
    for split in splits:
        inputs = common_log_svec_features(states[split.name].covariances)
        split_inputs.append(inputs)
        identity = _identity_fields(partition, geometry, "T2", split)
        base = _base_result_row(
            config,
            config_hash,
            partition,
            geometry,
            "logistic",
            "euclidean_log_svec",
            identity,
            source_feature_hash,
            source_label_hash,
            array_sha256(inputs),
            feature_config_hash,
            model,
            audit,
        )
        t2_rows.append(_failed_row(base))
    aggregate_identity = _identity_fields(
        partition, geometry, "T2", None, aggregate=True
    )
    aggregate_base = _base_result_row(
        config,
        config_hash,
        partition,
        geometry,
        "logistic",
        "euclidean_log_svec",
        aggregate_identity,
        source_feature_hash,
        source_label_hash,
        array_sha256(np.concatenate(split_inputs)),
        feature_config_hash,
        model,
        audit,
    )
    t2_rows.append(_failed_row(aggregate_base))
    return _failed_row(t1_base), t2_rows


def _failed_mdm_condition_rows(
    config: Mapping[str, Any],
    config_hash: str,
    partition: LosoPartition,
    geometry: str,
    metric: str,
    splits: Sequence[CalibrationSplit],
    states: Mapping[str, TargetTransform],
    source_feature_hash: str,
    source_label_hash: str,
    feature_config_hash: str,
    model: Any,
    audit: FitAudit,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Materialize all four required rows for one failed MDM condition."""

    t1_identity = _identity_fields(partition, geometry, "T1", None)
    t1_base = _base_result_row(
        config,
        config_hash,
        partition,
        geometry,
        "MDM",
        metric,
        t1_identity,
        source_feature_hash,
        source_label_hash,
        array_sha256(states["ALL"].covariances),
        feature_config_hash,
        model,
        audit,
    )
    t2_rows: list[dict[str, Any]] = []
    split_inputs: list[np.ndarray] = []
    for split in splits:
        inputs = states[split.name].covariances
        split_inputs.append(inputs)
        identity = _identity_fields(partition, geometry, "T2", split)
        base = _base_result_row(
            config,
            config_hash,
            partition,
            geometry,
            "MDM",
            metric,
            identity,
            source_feature_hash,
            source_label_hash,
            array_sha256(inputs),
            feature_config_hash,
            model,
            audit,
        )
        t2_rows.append(_failed_row(base))
    aggregate_identity = _identity_fields(
        partition, geometry, "T2", None, aggregate=True
    )
    aggregate_base = _base_result_row(
        config,
        config_hash,
        partition,
        geometry,
        "MDM",
        metric,
        aggregate_identity,
        source_feature_hash,
        source_label_hash,
        array_sha256(np.concatenate(split_inputs)),
        feature_config_hash,
        model,
        audit,
    )
    t2_rows.append(_failed_row(aggregate_base))
    return _failed_row(t1_base), t2_rows


def _evaluate(
    model: Any,
    inputs: np.ndarray,
    metadata: pd.DataFrame,
    positions: Sequence[int],
    class_order: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = np.asarray(positions, dtype=np.int64)
    # This local read is intentionally immediately adjacent to the only API
    # that accepts target labels.
    target_labels = metadata.iloc[selected]["class_label"].astype(str).to_numpy()
    prediction, metrics = evaluate_target_estimator(
        model, inputs, target_labels, class_order=class_order
    )
    return prediction, metrics


def _evaluate_concatenated_predictions(
    predictions: Sequence[np.ndarray],
    metadata: pd.DataFrame,
    position_groups: Sequence[Sequence[int]],
    class_order: Sequence[str],
) -> dict[str, Any]:
    """Evaluate fixed A/B predictions through the sole target-label boundary."""

    combined_prediction = np.concatenate(
        [np.asarray(values).astype(str) for values in predictions]
    )
    combined_positions = np.concatenate(
        [np.asarray(values, dtype=np.int64) for values in position_groups]
    )
    # As in _evaluate, target labels are read locally and passed immediately to
    # evaluate_target_estimator.  They are never returned to the LOSO runner.
    target_labels = (
        metadata.iloc[combined_positions]["class_label"].astype(str).to_numpy()
    )
    estimator = _FixedPredictionEstimator(combined_prediction)
    observed, metrics = evaluate_target_estimator(
        estimator,
        np.arange(len(combined_prediction), dtype=np.int64)[:, np.newaxis],
        target_labels,
        class_order=class_order,
    )
    if not np.array_equal(observed, combined_prediction):
        raise AssertionError("fixed aggregate prediction estimator changed predictions")
    return metrics


def _assert_source_reuse(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    for column in (
        "source_feature_sha256",
        "source_label_sha256",
        "feature_config_sha256",
        "model_config_sha256",
        "fitted_model_sha256",
    ):
        if len({str(row[column]) for row in rows}) != 1:
            raise AssertionError(f"T1/T2 source/model invariant failed for {column}")


def _assert_complete_rows(
    rows: Mapping[str, list[dict[str, Any]]],
    domain_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[pd.DataFrame],
    *,
    targets: Sequence[int],
    geometries: Sequence[str],
    mdm_specs: Sequence[tuple[str, str]],
    include_mdm: bool,
    include_domain: bool,
) -> None:
    """Assert exact preregistered logical-row coverage, including FAILED rows."""

    expected_log_t1 = {
        (int(target), geometry, "T1", "ALL", "logistic", "euclidean_log_svec")
        for target in targets
        for geometry in geometries
    }
    expected_log_t2 = {
        (int(target), geometry, "T2", split, "logistic", "euclidean_log_svec")
        for target in targets
        for geometry in geometries
        for split in ("A", "B", "AGGREGATE")
    }
    expected_mdm_t1 = (
        {
            (int(target), geometry, "T1", "ALL", "MDM", metric)
            for target in targets
            for geometry, metric in mdm_specs
        }
        if include_mdm
        else set()
    )
    expected_mdm_t2 = (
        {
            (int(target), geometry, "T2", split, "MDM", metric)
            for target in targets
            for geometry, metric in mdm_specs
            for split in ("A", "B", "AGGREGATE")
        }
        if include_mdm
        else set()
    )

    def observed(values: Sequence[Mapping[str, Any]]) -> set[tuple[Any, ...]]:
        keys = [
            (
                int(row["target_subject"]),
                str(row["geometry"]),
                str(row["protocol"]),
                str(row["split"]),
                str(row["decoder"]),
                str(row["native_metric"]),
            )
            for row in values
        ]
        if len(keys) != len(set(keys)):
            raise AssertionError("duplicate LOSO logical result row detected")
        if any(row.get("status") not in {"PASS", "FAILED"} for row in values):
            raise AssertionError("LOSO result row has an invalid status")
        return set(keys)

    comparisons = (
        ("logistic T1", observed(rows["log_t1"]), expected_log_t1),
        ("logistic T2", observed(rows["log_t2"]), expected_log_t2),
        ("MDM T1", observed(rows["mdm_t1"]), expected_mdm_t1),
        ("MDM T2", observed(rows["mdm_t2"]), expected_mdm_t2),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            raise AssertionError(
                f"{label} logical rows are incomplete: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    expected_domain = (
        len(targets)
        * sum(len(_domain_metric(geometry)) for geometry in geometries)
        * 3
        if include_domain
        else 0
    )
    if len(domain_rows) != expected_domain:
        raise AssertionError(
            f"domain diagnostic rows are incomplete: {len(domain_rows)} != "
            f"{expected_domain}"
        )
    expected_audit = len(targets) * 33
    observed_audit = sum(len(frame) for frame in audit_rows)
    if observed_audit != expected_audit:
        raise AssertionError(
            f"sample audit rows are incomplete: {observed_audit} != {expected_audit}"
        )


def _domain_metric(geometry: str) -> tuple[str, ...]:
    if geometry == RAW:
        return ("logeuclid", "riemann")
    if geometry == LE:
        return ("logeuclid",)
    if geometry == AIRM:
        return ("riemann",)
    if geometry == EA:
        return ("arithmetic_frobenius",)
    raise ValueError(f"unknown geometry {geometry!r}")


def _domain_components(
    covariances: np.ndarray,
    metric: str,
    *,
    tol: float,
    maxiter: int,
) -> _DomainComponents:
    matrices = np.asarray(covariances, dtype=np.float64)
    identity = np.eye(matrices.shape[-1], dtype=np.float64)
    if metric == "logeuclid":
        mean = logeuclidean_mean_custom(matrices)
        distance = logeuclidean_distance
    elif metric == "riemann":
        mean = airm_mean(matrices, tol=tol, maxiter=maxiter).matrix
        distance = airm_distance
    elif metric == "arithmetic_frobenius":
        mean = arithmetic_mean(matrices)

        def distance(first: np.ndarray, second: np.ndarray) -> np.ndarray | float:
            result = np.linalg.norm(np.asarray(first) - np.asarray(second), axis=(-2, -1))
            return float(result) if np.ndim(result) == 0 else result

    else:
        raise ValueError(f"unknown domain diagnostic metric {metric!r}")
    distances_identity = np.asarray(distance(matrices, identity), dtype=np.float64)
    distances_mean = np.asarray(distance(matrices, mean), dtype=np.float64)
    return _DomainComponents(
        mean=mean,
        mean_to_identity=float(distance(mean, identity)),
        rms_around_identity=float(np.sqrt(np.mean(distances_identity**2))),
        rms_around_own_mean=float(np.sqrt(np.mean(distances_mean**2))),
    )


def _mean_distance(first: np.ndarray, second: np.ndarray, metric: str) -> float:
    if metric == "logeuclid":
        return float(logeuclidean_distance(first, second))
    if metric == "riemann":
        return float(airm_distance(first, second))
    return float(np.linalg.norm(first - second, ord="fro"))


def _domain_row(
    config: Mapping[str, Any],
    config_hash: str,
    partition: LosoPartition,
    geometry: str,
    identity: Mapping[str, Any],
    metric: str,
    source: np.ndarray,
    target: np.ndarray,
    source_components: _DomainComponents,
    subject_structure: tuple[float, float] | None,
    *,
    tol: float,
    maxiter: int,
) -> dict[str, Any]:
    target_components = _domain_components(
        target, metric, tol=tol, maxiter=maxiter
    )
    signed = (
        target_components.rms_around_own_mean
        - source_components.rms_around_own_mean
    )
    return {
        "protocol_version": str(config["protocol"]["version"]),
        "config_sha256": config_hash,
        "seed": int(config["protocol"]["seed"]),
        "subject": partition.target_subject,
        "target_subject": partition.target_subject,
        "geometry": geometry,
        "protocol": identity["protocol"],
        "split": identity["split"],
        "reference_metric": metric,
        "source_n": len(source),
        "calibration_n": identity["calibration_n"],
        "evaluation_n": len(target),
        "source_trial_uid_hash": partition.source_trial_uid_sha256,
        "calibration_trial_uid_hash": identity["calibration_trial_uid_hash"],
        "evaluation_trial_uid_hash": identity["evaluation_trial_uid_hash"],
        "transductive_overlap": identity["transductive_overlap"],
        "source_transformed_sha256": array_sha256(source),
        "target_transformed_sha256": array_sha256(target),
        "source_mean_to_identity": source_components.mean_to_identity,
        "target_mean_to_identity": target_components.mean_to_identity,
        "source_target_mean_distance": _mean_distance(
            source_components.mean, target_components.mean, metric
        ),
        "source_rms_around_identity": source_components.rms_around_identity,
        "target_rms_around_identity": target_components.rms_around_identity,
        "source_rms_around_own_mean": source_components.rms_around_own_mean,
        "target_rms_around_own_mean": target_components.rms_around_own_mean,
        "target_minus_source_dispersion": signed,
        "absolute_dispersion_difference": abs(signed),
        "all_subject_subject_silhouette": (
            subject_structure[0] if subject_structure is not None else np.nan
        ),
        "all_subject_subject_between_within_rms_ratio": (
            subject_structure[1] if subject_structure is not None else np.nan
        ),
        "uses_class_labels": False,
        "status": "PASS",
    }


def _subject_structure(
    geometries: Sequence[str],
    subjects: Sequence[int],
    full_transforms: Mapping[tuple[str, int], np.ndarray],
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    labels = np.concatenate(
        [np.full(len(full_transforms[(geometries[0], s)]), s) for s in subjects]
    )
    for geometry in geometries:
        matrices = np.concatenate(
            [full_transforms[(geometry, subject)] for subject in subjects], axis=0
        )
        features = common_log_svec_features(matrices)
        result[geometry] = (
            float(silhouette_score(features, labels, metric="euclidean")),
            float(rms_distance_ratio(features, labels)),
        )
    return result


def _logistic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["classifiers"]["primary_logistic"]
    return {
        "c": float(frozen["C"]),
        "solver": str(frozen["solver"]),
        "max_iter": int(frozen["max_iter"]),
        "tol": float(frozen["tol"]),
        "random_state": int(frozen["random_state"]),
    }


def _target_cache_key(target: int, geometry: str) -> tuple[int, str]:
    return int(target), str(geometry)


def run_primary_loso(
    data: Any,
    config: Mapping[str, Any],
    *,
    target_subjects: Sequence[int] | None = None,
    geometries: Sequence[str] | None = None,
    include_mdm: bool = True,
    compute_domain_diagnostics: bool = True,
    config_sha256: str | None = None,
) -> LosoV2Result:
    """Run the frozen T1/T2 LOSO logistic probe and metric-native MDM checks.

    ``target_subjects`` and ``geometries`` exist solely to make deterministic
    synthetic tests inexpensive.  The command-line primary run supplies neither
    and therefore executes every preregistered target and geometry.
    """

    cfg = _configuration(config)
    covariances = np.asarray(data.covariances, dtype=np.float64)
    metadata = data.metadata.reset_index(drop=True).copy()
    if len(covariances) != len(metadata):
        raise ValueError("covariance and metadata row counts differ")
    if covariances.ndim != 3 or covariances.shape[1] != covariances.shape[2]:
        raise ValueError("WHOLE covariances must be a square 3-D stack")
    if "class_label" not in metadata:
        raise ValueError("evaluation metadata is missing class_label")
    identity_metadata = metadata.drop(columns=["class_label"])
    subjects = tuple(int(x) for x in cfg["dataset"]["subjects"])
    targets = tuple(subjects if target_subjects is None else map(int, target_subjects))
    if not targets or len(set(targets)) != len(targets) or set(targets) - set(subjects):
        raise ValueError("target_subjects must be a non-empty unique configured subset")
    chosen_geometries = tuple(
        GEOMETRIES if geometries is None else (str(x) for x in geometries)
    )
    if (
        not chosen_geometries
        or len(set(chosen_geometries)) != len(chosen_geometries)
        or set(chosen_geometries) - set(GEOMETRIES)
    ):
        raise ValueError(f"geometries must be a unique subset of {GEOMETRIES}")

    airm = cfg["geometry"]["airm_mean"]
    tol, maxiter = float(airm["tol"]), int(airm["maxiter"])
    if config_sha256 is None:
        config_hash = stable_json_hash(cfg)
    else:
        config_hash = str(config_sha256)
        if len(config_hash) != 64 or any(
            character not in "0123456789abcdef" for character in config_hash
        ):
            raise ValueError("config_sha256 must be a lowercase 64-character SHA-256")
    class_order = tuple(cfg["classifiers"]["metrics"]["class_order"])
    rows = _empty_result()
    domain_rows: list[dict[str, Any]] = []
    audit_rows: list[pd.DataFrame] = []

    partitions, subject_positions = _subject_positions(identity_metadata, subjects)
    splits_by_target = {
        target: make_calibration_splits(identity_metadata, target) for target in targets
    }
    for target in targets:
        partition = partitions[target]
        target_audits = [
            make_sample_id_audit_rows(
                identity_metadata, partition, protocol="T1"
            )
        ]
        for split in splits_by_target[target]:
            target_audits.append(
                make_sample_id_audit_rows(
                    identity_metadata,
                    partition,
                    protocol="T2",
                    calibration_split=split,
                )
            )
        for audit in target_audits:
            audit.insert(0, "seed", int(cfg["protocol"]["seed"]))
            audit.insert(0, "config_sha256", config_hash)
            audit.insert(0, "protocol_version", str(cfg["protocol"]["version"]))
            audit_rows.append(audit)

    full_transforms, full_centers = _full_subject_transforms(
        covariances,
        identity_metadata,
        subject_positions,
        chosen_geometries,
        tol=tol,
        maxiter=maxiter,
    )
    subject_structure = (
        _subject_structure(chosen_geometries, subjects, full_transforms)
        if compute_domain_diagnostics
        else {}
    )
    target_transforms: dict[tuple[int, str], dict[str, TargetTransform]] = {}

    def target_states(target: int, geometry: str) -> dict[str, TargetTransform]:
        key = _target_cache_key(target, geometry)
        if key not in target_transforms:
            target_transforms[key] = _target_transforms(
                covariances,
                partitions[target],
                splits_by_target[target],
                geometry,
                tol=tol,
                maxiter=maxiter,
                t1_state=TargetTransform(
                    covariances=full_transforms[(geometry, target)],
                    center=full_centers[(geometry, target)],
                ),
            )
        return target_transforms[key]

    feature_config_hash = stable_json_hash(
        {
            "feature": "svec(log(SPD))",
            "off_diagonal_scale": "sqrt2",
            "standard_scaler": False,
            "normalization": False,
            "pca": False,
        }
    )
    logistic_parameters = _logistic_config(cfg)

    # Primary logistic is completed first.  A convergence warning invalidates
    # the shared source model for T1 and both T2 halves, but independent frozen
    # conditions continue so the technical-failure grid remains complete.
    primary_failure_count = 0
    for target in targets:
        partition = partitions[target]
        splits = splits_by_target[target]
        for geometry in chosen_geometries:
            source_covs, source_labels, _ = _source_pool(
                partition,
                geometry,
                subject_positions,
                full_transforms,
                metadata,
            )
            source_features = common_log_svec_features(source_covs)
            source_feature_hash = array_sha256(source_features)
            source_label_hash = stable_json_hash({"labels": source_labels.tolist()})
            model, fit_audit = fit_source_logistic(
                source_features, source_labels, **logistic_parameters
            )
            states = target_states(target, geometry)
            t1_inputs = common_log_svec_features(states["ALL"].covariances)
            t1_identity = _identity_fields(partition, geometry, "T1", None)
            base = _base_result_row(
                cfg,
                config_hash,
                partition,
                geometry,
                "logistic",
                "euclidean_log_svec",
                t1_identity,
                source_feature_hash,
                source_label_hash,
                array_sha256(t1_inputs),
                feature_config_hash,
                model,
                fit_audit,
            )
            if fit_audit.convergence_warning:
                failed_t1, failed_t2 = _failed_logistic_condition_rows(
                    cfg,
                    config_hash,
                    partition,
                    geometry,
                    splits,
                    states,
                    source_feature_hash,
                    source_label_hash,
                    feature_config_hash,
                    model,
                    fit_audit,
                )
                rows["log_t1"].append(failed_t1)
                rows["log_t2"].extend(failed_t2)
                primary_failure_count += 1
                continue
            _, t1_metrics = _evaluate(
                model,
                t1_inputs,
                metadata,
                partition.target_row_positions,
                class_order,
            )
            condition_rows = [_metric_row(base, t1_metrics)]
            rows["log_t1"].append(condition_rows[0])

            split_predictions: list[np.ndarray] = []
            split_inputs: list[np.ndarray] = []
            split_bas: list[float] = []
            for split in splits:
                inputs = common_log_svec_features(states[split.name].covariances)
                identity = _identity_fields(partition, geometry, "T2", split)
                split_base = _base_result_row(
                    cfg,
                    config_hash,
                    partition,
                    geometry,
                    "logistic",
                    "euclidean_log_svec",
                    identity,
                    source_feature_hash,
                    source_label_hash,
                    array_sha256(inputs),
                    feature_config_hash,
                    model,
                    fit_audit,
                )
                prediction, metrics = _evaluate(
                    model,
                    inputs,
                    metadata,
                    split.evaluation_row_positions,
                    class_order,
                )
                result_row = _metric_row(split_base, metrics)
                rows["log_t2"].append(result_row)
                condition_rows.append(result_row)
                split_predictions.append(prediction)
                split_inputs.append(inputs)
                split_bas.append(float(metrics["balanced_accuracy"]))
            aggregate_metrics = _evaluate_concatenated_predictions(
                split_predictions,
                metadata,
                [split.evaluation_row_positions for split in splits],
                class_order,
            )
            tolerance = float(
                cfg["evaluation"]["T2"]["equivalent_flat_mean_tolerance"]
            )
            if abs(
                float(aggregate_metrics["balanced_accuracy"])
                - float(np.mean(split_bas))
            ) > tolerance:
                raise AssertionError(
                    "T2 aggregate BA differs from the arithmetic mean of split BA"
                )
            aggregate_identity = _identity_fields(
                partition, geometry, "T2", None, aggregate=True
            )
            aggregate_base = _base_result_row(
                cfg,
                config_hash,
                partition,
                geometry,
                "logistic",
                "euclidean_log_svec",
                aggregate_identity,
                source_feature_hash,
                source_label_hash,
                array_sha256(np.concatenate(split_inputs)),
                feature_config_hash,
                model,
                fit_audit,
            )
            aggregate_row = _metric_row(aggregate_base, aggregate_metrics)
            rows["log_t2"].append(aggregate_row)
            condition_rows.append(aggregate_row)
            _assert_source_reuse(condition_rows)

    if compute_domain_diagnostics:
        for target in targets:
            partition = partitions[target]
            splits = splits_by_target[target]
            for geometry in chosen_geometries:
                source_covs, _ = _source_covariance_pool(
                    partition,
                    geometry,
                    subject_positions,
                    full_transforms,
                )
                states = target_states(target, geometry)
                for metric in _domain_metric(geometry):
                    source_components = _domain_components(
                        source_covs, metric, tol=tol, maxiter=maxiter
                    )
                    t1_identity = _identity_fields(partition, geometry, "T1", None)
                    domain_rows.append(
                        _domain_row(
                            cfg,
                            config_hash,
                            partition,
                            geometry,
                            t1_identity,
                            metric,
                            source_covs,
                            states["ALL"].covariances,
                            source_components,
                            subject_structure[geometry],
                            tol=tol,
                            maxiter=maxiter,
                        )
                    )
                    for split in splits:
                        target_covs = states[split.name].covariances
                        identity = _identity_fields(
                            partition, geometry, "T2", split
                        )
                        domain_rows.append(
                            _domain_row(
                                cfg,
                                config_hash,
                                partition,
                                geometry,
                                identity,
                                metric,
                                source_covs,
                                target_covs,
                                source_components,
                                None,
                                tol=tol,
                                maxiter=maxiter,
                            )
                        )

    selected_specs = [
        item for item in MDM_SPECS if item[0] in set(chosen_geometries)
    ]
    secondary_failure_count = 0
    if include_mdm:
        mdm_feature_config = stable_json_hash(
            {"feature": "SPD matrix", "scaling": False, "tuning": False}
        )
        for target in targets:
            partition = partitions[target]
            splits = splits_by_target[target]
            for geometry, metric in selected_specs:
                source_covs, source_labels, _ = _source_pool(
                    partition,
                    geometry,
                    subject_positions,
                    full_transforms,
                    metadata,
                )
                source_feature_hash = array_sha256(source_covs)
                source_label_hash = stable_json_hash({"labels": source_labels.tolist()})
                states = target_states(target, geometry)
                model: Any = None
                fit_audit: FitAudit
                try:
                    model, fit_audit = fit_source_mdm(
                        source_covs, source_labels, metric=metric
                    )
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    fit_audit = _mdm_failure_audit(
                        metric,
                        len(source_covs),
                        (message,),
                        convergence_warning=False,
                    )
                    failed_t1, failed_t2 = _failed_mdm_condition_rows(
                        cfg,
                        config_hash,
                        partition,
                        geometry,
                        metric,
                        splits,
                        states,
                        source_feature_hash,
                        source_label_hash,
                        mdm_feature_config,
                        model,
                        fit_audit,
                    )
                    rows["mdm_t1"].append(failed_t1)
                    rows["mdm_t2"].extend(failed_t2)
                    secondary_failure_count += 1
                    continue
                if fit_audit.convergence_warning:
                    failed_t1, failed_t2 = _failed_mdm_condition_rows(
                        cfg,
                        config_hash,
                        partition,
                        geometry,
                        metric,
                        splits,
                        states,
                        source_feature_hash,
                        source_label_hash,
                        mdm_feature_config,
                        model,
                        fit_audit,
                    )
                    rows["mdm_t1"].append(failed_t1)
                    rows["mdm_t2"].extend(failed_t2)
                    secondary_failure_count += 1
                    continue

                t1_identity = _identity_fields(partition, geometry, "T1", None)
                base = _base_result_row(
                    cfg,
                    config_hash,
                    partition,
                    geometry,
                    "MDM",
                    metric,
                    t1_identity,
                    source_feature_hash,
                    source_label_hash,
                    array_sha256(states["ALL"].covariances),
                    mdm_feature_config,
                    model,
                    fit_audit,
                )
                try:
                    _, t1_metrics = _evaluate(
                        model,
                        states["ALL"].covariances,
                        metadata,
                        partition.target_row_positions,
                        class_order,
                    )
                    condition_t1 = _metric_row(base, t1_metrics)
                    condition_t2: list[dict[str, Any]] = []
                    condition_rows = [condition_t1]

                    split_predictions: list[np.ndarray] = []
                    split_inputs: list[np.ndarray] = []
                    split_bas: list[float] = []
                    for split in splits:
                        inputs = states[split.name].covariances
                        identity = _identity_fields(
                            partition, geometry, "T2", split
                        )
                        split_base = _base_result_row(
                            cfg,
                            config_hash,
                            partition,
                            geometry,
                            "MDM",
                            metric,
                            identity,
                            source_feature_hash,
                            source_label_hash,
                            array_sha256(inputs),
                            mdm_feature_config,
                            model,
                            fit_audit,
                        )
                        prediction, metrics = _evaluate(
                            model,
                            inputs,
                            metadata,
                            split.evaluation_row_positions,
                            class_order,
                        )
                        result_row = _metric_row(split_base, metrics)
                        condition_t2.append(result_row)
                        condition_rows.append(result_row)
                        split_predictions.append(prediction)
                        split_inputs.append(inputs)
                        split_bas.append(float(metrics["balanced_accuracy"]))
                    aggregate_metrics = _evaluate_concatenated_predictions(
                        split_predictions,
                        metadata,
                        [split.evaluation_row_positions for split in splits],
                        class_order,
                    )
                except Exception as error:
                    messages = (
                        *fit_audit.warning_messages,
                        f"{type(error).__name__}: {error}",
                    )
                    failed_audit = _mdm_failure_audit(
                        metric,
                        len(source_covs),
                        messages,
                        convergence_warning=False,
                    )
                    failed_t1, failed_t2 = _failed_mdm_condition_rows(
                        cfg,
                        config_hash,
                        partition,
                        geometry,
                        metric,
                        splits,
                        states,
                        source_feature_hash,
                        source_label_hash,
                        mdm_feature_config,
                        model,
                        failed_audit,
                    )
                    rows["mdm_t1"].append(failed_t1)
                    rows["mdm_t2"].extend(failed_t2)
                    secondary_failure_count += 1
                    continue

                tolerance = float(
                    cfg["evaluation"]["T2"]["equivalent_flat_mean_tolerance"]
                )
                if abs(
                    float(aggregate_metrics["balanced_accuracy"])
                    - float(np.mean(split_bas))
                ) > tolerance:
                    raise AssertionError(
                        "MDM T2 aggregate BA differs from split BA arithmetic mean"
                    )
                aggregate_identity = _identity_fields(
                    partition, geometry, "T2", None, aggregate=True
                )
                aggregate_base = _base_result_row(
                    cfg,
                    config_hash,
                    partition,
                    geometry,
                    "MDM",
                    metric,
                    aggregate_identity,
                    source_feature_hash,
                    source_label_hash,
                    array_sha256(np.concatenate(split_inputs)),
                    mdm_feature_config,
                    model,
                    fit_audit,
                )
                aggregate_row = _metric_row(aggregate_base, aggregate_metrics)
                condition_rows.append(aggregate_row)
                _assert_source_reuse(condition_rows)
                rows["mdm_t1"].append(condition_t1)
                rows["mdm_t2"].extend([*condition_t2, aggregate_row])

    _assert_complete_rows(
        rows,
        domain_rows,
        audit_rows,
        targets=targets,
        geometries=chosen_geometries,
        mdm_specs=selected_specs,
        include_mdm=include_mdm,
        include_domain=compute_domain_diagnostics,
    )
    secondary_status = (
        "FAILED"
        if secondary_failure_count
        else ("PASS" if include_mdm else "NOT_RUN")
    )
    return _finalize(
        rows,
        domain_rows,
        audit_rows,
        fatal_error=None,
        primary_failure_count=primary_failure_count,
        secondary_failure_count=secondary_failure_count,
        secondary_status=secondary_status,
    )


def target_transform_signatures_are_label_free() -> bool:
    """Machine-checkable guard used by tests and downstream audits."""

    forbidden = {"y", "label", "labels", "class_label", "target_labels"}
    for callable_object in (fit_center, fit_target_transform, FittedCenter.transform):
        names = set(inspect.signature(callable_object).parameters)
        if names & forbidden:
            return False
    return True
