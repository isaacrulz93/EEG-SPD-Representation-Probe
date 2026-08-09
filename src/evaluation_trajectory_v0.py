"""Frozen evaluation and null machinery for Trajectory Anatomy v0.

Geometry construction is intentionally outside this module.  Every public
scientific evaluator consumes already-computed trial-aligned arrays, allowing
the primary AIRM and secondary LE implementations to share exactly the same
LOSO, RNG, hashing, warning, and failure semantics.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


MASTER_SEED = 20260809
ORDER_STREAM_TAG = 0x4F52444552
LABEL_STREAM_TAG = 0x4C4142454C
NULL_REPLICATES = 199
DEFAULT_CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
COMMON_COLUMNS = (
    "protocol_version",
    "protocol_sha256",
    "config_sha256",
    "seed",
    "session",
    "generated_at_utc",
    "status",
)

CLASS_LOSO_COLUMNS = COMMON_COLUMNS + (
    "geometry",
    "representation",
    "target_subject",
    "source_subjects",
    "train_n",
    "test_n",
    "train_uid_sha256",
    "test_uid_sha256",
    "scaler_fit_uid_sha256",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "recall_left_hand",
    "recall_right_hand",
    "recall_feet",
    "recall_tongue",
    "confusion_matrix_json",
    "prediction_sha256",
    "classifier_config_sha256",
    "convergence_warning",
    "warning_messages",
)

SUBJECT_PROBE_COLUMNS = COMMON_COLUMNS + (
    "geometry",
    "representation",
    "split",
    "train_runs",
    "evaluation_runs",
    "train_n",
    "test_n",
    "train_uid_sha256",
    "test_uid_sha256",
    "chance_level",
    "balanced_accuracy",
    "accuracy",
    "direction_average_ba",
    "direction_average_accuracy",
    "prediction_sha256",
    "classifier_config_sha256",
    "convergence_warning",
    "warning_messages",
)

FACTOR_COLUMNS = COMMON_COLUMNS + (
    "geometry",
    "scalar",
    "n_subjects",
    "n_classes",
    "n_per_cell",
    "grand_mean",
    "ss_subject",
    "ss_class",
    "ss_interaction",
    "ss_residual",
    "ss_total",
    "eta2_subject",
    "eta2_class",
    "eta2_interaction",
    "eta2_residual",
    "ss_reconstruction_relative_error",
    "degenerate",
    "uses_p_value",
)

NULL_SUBJECT_COLUMNS = COMMON_COLUMNS + (
    "geometry",
    "representation",
    "replicate",
    "replicate_seed",
    "target_subject",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "observed_ba",
    "subject_null_median_ba",
    "subject_effect",
    "train_uid_sha256",
    "test_uid_sha256",
    "classifier_status",
    "convergence_warning",
    "warning_messages",
)

ORDER_GROUP_COLUMNS = COMMON_COLUMNS + (
    "geometry",
    "representation",
    "observed_median_subject_ba",
    "null_replicates",
    "null_median",
    "null_mean",
    "null_sd_ddof1",
    "null_min",
    "null_max",
    "effect",
    "p_value",
    "exceedance_count",
    "median_subject_path_minus_bag",
    "hypothesis_operand_pass",
)

LABEL_GROUP_COLUMNS = tuple(
    column for column in ORDER_GROUP_COLUMNS if column != "median_subject_path_minus_bag"
)

MDM_COLUMNS = COMMON_COLUMNS + (
    "representation",
    "target_subject",
    "train_n",
    "test_n",
    "train_uid_sha256",
    "test_uid_sha256",
    "metric",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "recall_left_hand",
    "recall_right_hand",
    "recall_feet",
    "recall_tongue",
    "confusion_matrix_json",
    "prediction_sha256",
    "convergence_warning",
    "warning_messages",
)


class TrajectoryEvaluationError(RuntimeError):
    """Raised on a structural, leakage, schema, or null-plan violation."""


@dataclass(frozen=True)
class LogisticFitAudit:
    status: str
    config_sha256: str
    train_n: int
    n_features: int
    train_feature_sha256: str
    train_label_sha256: str
    scaler_mean_sha256: str
    scaler_scale_sha256: str
    fitted_estimator_sha256: str | None
    convergence_warning: bool
    warning_messages: tuple[str, ...]
    n_iter_max: int | None


@dataclass(frozen=True)
class ScaledLogisticFit:
    scaler: StandardScaler
    model: LogisticRegression
    audit: LogisticFitAudit

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.audit.status != "PASS":
            raise TrajectoryEvaluationError("a FAILED logistic fit cannot be scored")
        return np.asarray(self.model.predict(self.scaler.transform(_features(features))))


@dataclass(frozen=True)
class NullSeedPlan:
    family: str
    master_seed: int
    stream_tag: int
    seeds: np.ndarray
    audit_sha256: str

    def to_json_dict(self, *, protocol_version: str = "0.0") -> dict[str, Any]:
        return {
            "protocol_version": str(protocol_version),
            "master_seed": int(self.master_seed),
            "stream_tag_hex": f"0x{self.stream_tag:X}",
            "seedsequence_entropy": [int(self.master_seed), int(self.stream_tag)],
            "child_count": int(len(self.seeds)),
            "seed_dtype": "uint64",
            "seed_extraction": "int(child.generate_state(1, dtype=np.uint64)[0])",
            "replicates": [
                {"replicate": index + 1, "seed": int(seed)}
                for index, seed in enumerate(self.seeds)
            ],
        }


@dataclass(frozen=True)
class OrderPermutationPlan:
    seed_plan: NullSeedPlan
    sample_indices: np.ndarray
    permutation_indices: np.ndarray
    trial_uid_sha256: str
    audit_sha256: str

    @property
    def replicates(self) -> int:
        return len(self.seed_plan.seeds)


@dataclass(frozen=True)
class LabelPermutationPlan:
    seed_plan: NullSeedPlan
    sample_indices: np.ndarray
    source_row_indices: np.ndarray
    trial_uid_sha256: str
    audit_sha256: str

    @property
    def replicates(self) -> int:
        return len(self.seed_plan.seeds)


@dataclass(frozen=True)
class NullSummary:
    table: pd.DataFrame
    replicate_statistics: np.ndarray
    complete: bool


@dataclass(frozen=True)
class FrozenVerdict:
    verdict: str
    failure_status: str
    h_path_class: bool | None
    h_bag_class: bool | None
    h_order: bool | None
    median_subject_path_minus_bag: float | None
    operands: Mapping[str, Any]


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_audit_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def trial_uid_sha256(values: Sequence[str]) -> str:
    strings = [str(value) for value in values]
    if not strings or len(strings) != len(set(strings)) or any(not value for value in strings):
        raise TrajectoryEvaluationError("trial UID hash input must be non-empty and unique")
    return stable_json_sha256({"trial_uids": sorted(strings)})


def _read_only(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array)
    value.setflags(write=False)
    return value


def _features(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise TrajectoryEvaluationError(
            f"features must be a finite non-empty 2-D float array, got {value.shape}"
        )
    return value


def _labels(labels: Sequence[Any] | np.ndarray, n_rows: int) -> np.ndarray:
    value = np.asarray(labels)
    if value.ndim != 1 or len(value) != n_rows or pd.isna(value).any():
        raise TrajectoryEvaluationError(f"labels must be non-null with shape ({n_rows},)")
    return value.astype(str)


def _metadata(metadata: pd.DataFrame, *, require_class: bool = True) -> pd.DataFrame:
    required = {"sample_index", "subject", "session", "run", "trial_uid"}
    if require_class:
        required.add("class_label")
    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        raise TrajectoryEvaluationError("metadata must be a non-empty DataFrame")
    missing = required - set(metadata.columns)
    if missing:
        raise TrajectoryEvaluationError(f"metadata is missing columns: {sorted(missing)}")
    frame = metadata.copy(deep=True).reset_index(drop=True)
    if frame[list(required)].isna().any(axis=None):
        raise TrajectoryEvaluationError("metadata identity/label fields contain nulls")
    for column in ("sample_index", "subject", "run"):
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise TrajectoryEvaluationError(f"metadata {column} must contain integers")
        frame[column] = numeric.astype(np.int64)
    frame["session"] = frame["session"].astype(str)
    frame["trial_uid"] = frame["trial_uid"].astype(str)
    if require_class:
        frame["class_label"] = frame["class_label"].astype(str)
    if tuple(sorted(frame["session"].unique())) != ("0train",):
        raise TrajectoryEvaluationError("evaluation accepts session '0train' only")
    if frame["sample_index"].duplicated().any() or frame["trial_uid"].duplicated().any():
        raise TrajectoryEvaluationError("sample_index and trial_uid must be globally unique")
    return frame


def _provenance(provenance: Mapping[str, Any] | None, *, status: str) -> dict[str, Any]:
    source = dict(provenance or {})
    return {
        "protocol_version": str(source.get("protocol_version", "0.0")),
        "protocol_sha256": str(source.get("protocol_sha256", "")),
        "config_sha256": str(source.get("config_sha256", "")),
        "seed": int(source.get("seed", MASTER_SEED)),
        "session": str(source.get("session", "0train")),
        "generated_at_utc": str(source.get("generated_at_utc", "")),
        "status": status,
    }


def _validate_trial_counts(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    require_class: bool,
) -> None:
    dataset = config["dataset"]
    subjects = tuple(int(value) for value in dataset["subjects"])
    runs = tuple(int(value) for value in dataset["runs"])
    if len(frame) != int(dataset["expected_trials"]):
        raise TrajectoryEvaluationError("trial count differs from the frozen dataset contract")
    if tuple(sorted(frame["subject"].unique())) != subjects:
        raise TrajectoryEvaluationError("subject set differs from the frozen dataset contract")
    if tuple(sorted(frame["run"].unique())) != runs:
        raise TrajectoryEvaluationError("run set differs from the frozen dataset contract")
    subject_counts = frame.groupby("subject", observed=True).size()
    run_counts = frame.groupby(["subject", "run"], observed=True).size()
    if not (subject_counts == int(dataset["expected_trials_per_subject"])).all():
        raise TrajectoryEvaluationError("per-subject trial counts are not frozen")
    if not (run_counts == int(dataset["expected_trials_per_subject_run"])).all():
        raise TrajectoryEvaluationError("per-subject×run counts are not frozen")
    if require_class:
        classes = tuple(str(value) for value in dataset["classes"])
        if set(frame["class_label"].unique()) != set(classes):
            raise TrajectoryEvaluationError("class vocabulary differs from the frozen contract")
        class_counts = frame.groupby(["subject", "class_label"], observed=True).size()
        run_class_counts = frame.groupby(
            ["subject", "run", "class_label"], observed=True
        ).size()
        if not (
            class_counts == int(dataset["expected_trials_per_subject_class"])
        ).all() or not (
            run_class_counts
            == int(dataset["expected_trials_per_subject_run_class"])
        ).all():
            raise TrajectoryEvaluationError("class-balanced trial counts are not frozen")


def _probe_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("class_probe", config)


def classifier_config_sha256(config: Mapping[str, Any]) -> str:
    probe = _probe_config(config)
    if str(probe.get("scaler")) != "StandardScaler" or str(
        probe.get("scaler_fit_scope")
    ) != "source_only":
        raise TrajectoryEvaluationError("classifier scaler scope is not frozen")
    if str(probe.get("classifier")) != "multinomial_logistic_regression":
        raise TrajectoryEvaluationError("classifier family is not frozen")
    if bool(probe.get("tuning", False)):
        raise TrajectoryEvaluationError("classifier tuning is forbidden")
    frozen = {
        "scaler": "StandardScaler",
        "scaler_fit_scope": "source_only",
        "with_mean": True,
        "with_std": True,
        "classifier": "multinomial_logistic_regression",
        "C": float(probe.get("c", probe.get("C", 1.0))),
        "solver": str(probe.get("solver", "lbfgs")),
        "max_iter": int(probe.get("max_iter", 5000)),
        "tol": float(probe.get("tol", 1e-4)),
        "random_state": int(probe.get("random_state", MASTER_SEED)),
        "class_weight": probe.get("class_weight"),
        "l2_encoding_sklearn_1_9": "l1_ratio=0.0",
        "pca": False,
        "tuning": False,
    }
    if frozen["solver"] != "lbfgs" or frozen["C"] != 1.0:
        raise TrajectoryEvaluationError("classifier config differs from frozen C=1/lbfgs")
    if frozen["max_iter"] != 5000 or frozen["tol"] != 1e-4:
        raise TrajectoryEvaluationError("classifier iteration/tolerance config is not frozen")
    if frozen["random_state"] != MASTER_SEED or frozen["class_weight"] is not None:
        raise TrajectoryEvaluationError("classifier seed/class_weight config is not frozen")
    return stable_json_sha256(frozen)


def _fitted_estimator_hash(scaler: StandardScaler, model: LogisticRegression) -> str:
    payload = {
        "scaler_mean": array_audit_sha256(scaler.mean_),
        "scaler_scale": array_audit_sha256(scaler.scale_),
        "coef": array_audit_sha256(model.coef_),
        "intercept": array_audit_sha256(model.intercept_),
        "classes": array_audit_sha256(np.asarray(model.classes_).astype("U")),
        "n_iter": array_audit_sha256(np.asarray(model.n_iter_, dtype=np.int64)),
    }
    return stable_json_sha256(payload)


def fit_source_scaled_logistic(
    source_features: np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    config: Mapping[str, Any],
) -> ScaledLogisticFit:
    """Fit the frozen scaler and LR from source arrays only."""

    x = _features(source_features)
    y = _labels(source_labels, len(x))
    if len(np.unique(y)) < 2:
        raise TrajectoryEvaluationError("source labels must contain at least two classes")
    probe = _probe_config(config)
    config_hash = classifier_config_sha256(config)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        C=float(probe.get("c", probe.get("C", 1.0))),
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-4,
        class_weight=None,
        random_state=MASTER_SEED,
        warm_start=False,
    )
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(scaled, y)
    convergence = tuple(
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    )
    all_warnings = tuple(str(item.message) for item in caught)
    status = "FAILED" if convergence else "PASS"
    audit = LogisticFitAudit(
        status=status,
        config_sha256=config_hash,
        train_n=len(x),
        n_features=x.shape[1],
        train_feature_sha256=array_audit_sha256(x),
        train_label_sha256=array_audit_sha256(y.astype("U")),
        scaler_mean_sha256=array_audit_sha256(scaler.mean_),
        scaler_scale_sha256=array_audit_sha256(scaler.scale_),
        fitted_estimator_sha256=(
            None if convergence else _fitted_estimator_hash(scaler, model)
        ),
        convergence_warning=bool(convergence),
        warning_messages=all_warnings,
        n_iter_max=int(np.max(model.n_iter_)),
    )
    return ScaledLogisticFit(scaler, model, audit)


def evaluate_target_predictions(
    target_labels: Sequence[Any] | np.ndarray,
    predictions: Sequence[Any] | np.ndarray,
    *,
    class_order: Sequence[str] = DEFAULT_CLASS_ORDER,
) -> dict[str, Any]:
    """The sole target-class-label scoring boundary."""

    truth = np.asarray(target_labels).astype(str)
    predicted = np.asarray(predictions).astype(str)
    classes = tuple(str(value) for value in class_order)
    if truth.ndim != 1 or predicted.shape != truth.shape or len(truth) == 0:
        raise TrajectoryEvaluationError("truth and predictions must be equal non-empty vectors")
    if set(truth) - set(classes) or set(predicted) - set(classes):
        raise TrajectoryEvaluationError("target/predicted labels leave the frozen class order")
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
        "prediction_sha256": stable_json_sha256(
            {
                "truth": truth.tolist(),
                "predicted": predicted.tolist(),
                "class_order": list(classes),
            }
        ),
    }
    for label, value in zip(classes, recalls, strict=True):
        result[f"recall_{label}"] = float(value)
    return result


def _failed_class_metrics(class_order: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "balanced_accuracy": np.nan,
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "confusion_matrix_json": "",
        "prediction_sha256": "",
    }
    for label in class_order:
        result[f"recall_{label}"] = np.nan
    return result


def run_class_loso(
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    geometry: str,
    representation: str,
    labels: Sequence[Any] | np.ndarray | None = None,
    target_subjects: Sequence[int] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run the fixed nine-fold class LOSO with no target preprocessing fit."""

    x = _features(features)
    frame = _metadata(metadata)
    _validate_trial_counts(frame, config, require_class=True)
    if len(x) != len(frame):
        raise TrajectoryEvaluationError("feature and metadata row counts differ")
    y = _labels(frame["class_label"] if labels is None else labels, len(frame))
    class_order = tuple(str(value) for value in config["dataset"]["classes"])
    if set(y) != set(class_order):
        raise TrajectoryEvaluationError("class labels differ from the frozen class vocabulary")
    subjects = tuple(int(value) for value in config["dataset"]["subjects"])
    targets = subjects if target_subjects is None else tuple(int(value) for value in target_subjects)
    if not targets or set(targets) - set(subjects):
        raise TrajectoryEvaluationError("target_subjects leave the frozen subject set")
    rows: list[dict[str, Any]] = []
    for target in targets:
        test = frame["subject"].to_numpy() == target
        train = ~test
        if not train.any() or not test.any():
            raise TrajectoryEvaluationError(f"empty LOSO split for subject {target}")
        train_uids = frame.loc[train, "trial_uid"].tolist()
        test_uids = frame.loc[test, "trial_uid"].tolist()
        if set(train_uids) & set(test_uids):
            raise TrajectoryEvaluationError("LOSO train/test UID overlap")
        fitted = fit_source_scaled_logistic(x[train], y[train], config)
        if fitted.audit.status == "PASS":
            prediction = fitted.predict(x[test])
            metrics = evaluate_target_predictions(
                y[test], prediction, class_order=class_order
            )
        else:
            metrics = _failed_class_metrics(class_order)
        row = {
            **_provenance(provenance, status=fitted.audit.status),
            "geometry": str(geometry),
            "representation": str(representation),
            "target_subject": target,
            "source_subjects": json.dumps(
                [value for value in subjects if value != target], separators=(",", ":")
            ),
            "train_n": int(train.sum()),
            "test_n": int(test.sum()),
            "train_uid_sha256": trial_uid_sha256(train_uids),
            "test_uid_sha256": trial_uid_sha256(test_uids),
            "scaler_fit_uid_sha256": trial_uid_sha256(train_uids),
            **metrics,
            "classifier_config_sha256": fitted.audit.config_sha256,
            "convergence_warning": fitted.audit.convergence_warning,
            "warning_messages": json.dumps(
                list(fitted.audit.warning_messages), ensure_ascii=False, separators=(",", ":")
            ),
        }
        rows.append(row)
    result = pd.DataFrame.from_records(rows)
    return result.reindex(columns=CLASS_LOSO_COLUMNS)


def make_null_seed_plan(
    family: str,
    *,
    replicates: int = NULL_REPLICATES,
    master_seed: int = MASTER_SEED,
    stream_tag: int | None = None,
) -> NullSeedPlan:
    """Construct the exact frozen SeedSequence child list."""

    family_normalized = str(family).lower()
    expected_tag = {
        "order": ORDER_STREAM_TAG,
        "order_shuffle": ORDER_STREAM_TAG,
        "label": LABEL_STREAM_TAG,
        "label_destruction": LABEL_STREAM_TAG,
    }.get(family_normalized)
    if expected_tag is None:
        raise TrajectoryEvaluationError("null family must be order or label")
    tag = expected_tag if stream_tag is None else int(stream_tag)
    if tag != expected_tag or int(master_seed) != MASTER_SEED:
        raise TrajectoryEvaluationError("null seed/tag differs from the frozen protocol")
    if not isinstance(replicates, (int, np.integer)) or int(replicates) < 1:
        raise TrajectoryEvaluationError("replicates must be a positive integer")
    children = np.random.SeedSequence([MASTER_SEED, tag]).spawn(int(replicates))
    seeds = np.asarray(
        [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children],
        dtype=np.uint64,
    )
    audit = stable_json_sha256(
        {
            "family": family_normalized,
            "master_seed": MASTER_SEED,
            "stream_tag": tag,
            "seeds": [int(value) for value in seeds],
        }
    )
    return NullSeedPlan(family_normalized, MASTER_SEED, tag, _read_only(seeds), audit)


def replay_null_seed_plan(plan: NullSeedPlan) -> bool:
    replayed = make_null_seed_plan(
        plan.family,
        replicates=len(plan.seeds),
        master_seed=plan.master_seed,
        stream_tag=plan.stream_tag,
    )
    return bool(
        np.array_equal(replayed.seeds, plan.seeds)
        and replayed.audit_sha256 == plan.audit_sha256
    )


def lexicographic_state_permutations() -> np.ndarray:
    permutations = np.asarray(
        list(itertools.permutations(range(5))), dtype=np.int8
    )
    if permutations.shape != (120, 5) or not np.array_equal(
        permutations[0], np.arange(5)
    ):
        raise RuntimeError("internal S5 permutation construction failed")
    return _read_only(permutations)


def _plan_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = _metadata(metadata)
    order = np.argsort(frame["sample_index"].to_numpy(), kind="stable")
    if not np.array_equal(
        frame["sample_index"].to_numpy()[order], np.arange(len(frame))
    ):
        raise TrajectoryEvaluationError("null plans require sample_index exactly 0..N-1")
    return frame


def make_order_permutation_plan(
    metadata: pd.DataFrame,
    *,
    seed_plan: NullSeedPlan | None = None,
    replicates: int = NULL_REPLICATES,
) -> OrderPermutationPlan:
    """Draw one nonidentity S5 index per replicate/trial in sample order."""

    frame = _plan_metadata(metadata)
    seeds = seed_plan or make_null_seed_plan("order", replicates=replicates)
    if seeds.stream_tag != ORDER_STREAM_TAG:
        raise TrajectoryEvaluationError("order plan received a non-order seed stream")
    row_order = np.argsort(frame["sample_index"].to_numpy(), kind="stable")
    choices = np.empty((len(seeds.seeds), len(frame)), dtype=np.uint8)
    for replicate_index, stored_seed in enumerate(seeds.seeds):
        generator = np.random.default_rng(int(stored_seed))
        drawn = generator.integers(1, 120, size=len(frame))
        choices[replicate_index, row_order] = drawn.astype(np.uint8)
    if np.any((choices < 1) | (choices > 119)):
        raise RuntimeError("order null generated identity or out-of-range permutation")
    sample_indices = frame["sample_index"].to_numpy(dtype=np.int64)
    uid_hash = trial_uid_sha256(frame["trial_uid"].tolist())
    audit = stable_json_sha256(
        {
            "seed_plan": seeds.audit_sha256,
            "sample_indices": array_audit_sha256(sample_indices),
            "ordered_trial_uids": array_audit_sha256(
                frame["trial_uid"].to_numpy(dtype="U")
            ),
            "permutation_indices": array_audit_sha256(choices),
            "trial_uid_sha256": uid_hash,
        }
    )
    return OrderPermutationPlan(
        seeds,
        _read_only(sample_indices),
        _read_only(choices),
        uid_hash,
        audit,
    )


def replay_order_permutation_plan(
    metadata: pd.DataFrame, plan: OrderPermutationPlan
) -> bool:
    replayed = make_order_permutation_plan(metadata, seed_plan=plan.seed_plan)
    return bool(
        np.array_equal(replayed.sample_indices, plan.sample_indices)
        and np.array_equal(replayed.permutation_indices, plan.permutation_indices)
        and replayed.audit_sha256 == plan.audit_sha256
    )


_PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))


def _edge_reindex_table() -> np.ndarray:
    edge = np.full((5, 5), -1, dtype=np.int8)
    for index, (first, second) in enumerate(_PAIR_ORDER):
        edge[first, second] = edge[second, first] = index
    permutations = lexicographic_state_permutations()
    result = np.asarray(
        [
            [edge[permutation[first], permutation[second]] for first, second in _PAIR_ORDER]
            for permutation in permutations
        ],
        dtype=np.int8,
    )
    return _read_only(result)


EDGE_REINDEX_TABLE = _edge_reindex_table()


def apply_order_permutation(
    distances_or_path: np.ndarray,
    plan: OrderPermutationPlan,
    replicate: int,
) -> np.ndarray:
    """Rebuild PATH_D10 from a planned state relabeling without SPD recomputation."""

    replicate_index = int(replicate) - 1
    if not 0 <= replicate_index < plan.replicates:
        raise TrajectoryEvaluationError("replicate index is outside the order plan")
    value = np.asarray(distances_or_path, dtype=np.float64)
    if value.ndim == 3 and value.shape[1:] == (5, 5):
        path = np.stack([value[:, first, second] for first, second in _PAIR_ORDER], axis=1)
    elif value.ndim == 2 and value.shape[1] == 10:
        path = value
    else:
        raise TrajectoryEvaluationError(
            "order permutation input must have shape (N,5,5) or (N,10)"
        )
    if len(path) != len(plan.sample_indices) or not np.isfinite(path).all():
        raise TrajectoryEvaluationError("order permutation input is misaligned or non-finite")
    maps = EDGE_REINDEX_TABLE[plan.permutation_indices[replicate_index]]
    result = np.take_along_axis(path, maps, axis=1)
    return _read_only(result)


def make_label_permutation_plan(
    metadata: pd.DataFrame,
    *,
    seed_plan: NullSeedPlan | None = None,
    replicates: int = NULL_REPLICATES,
) -> LabelPermutationPlan:
    """Build exact within-subject×run label-source row mappings."""

    frame = _plan_metadata(metadata)
    seeds = seed_plan or make_null_seed_plan("label", replicates=replicates)
    if seeds.stream_tag != LABEL_STREAM_TAG:
        raise TrajectoryEvaluationError("label plan received a non-label seed stream")
    groups: list[np.ndarray] = []
    for _, group in frame.groupby(["subject", "run"], sort=True, observed=True):
        groups.append(
            group.sort_values("sample_index", kind="stable").index.to_numpy(dtype=np.int64)
        )
    mappings = np.empty((len(seeds.seeds), len(frame)), dtype=np.int64)
    for replicate_index, stored_seed in enumerate(seeds.seeds):
        generator = np.random.default_rng(int(stored_seed))
        mapping = np.arange(len(frame), dtype=np.int64)
        for rows in groups:
            mapping[rows] = rows[generator.permutation(len(rows))]
        mappings[replicate_index] = mapping
    sample_indices = frame["sample_index"].to_numpy(dtype=np.int64)
    uid_hash = trial_uid_sha256(frame["trial_uid"].tolist())
    audit = stable_json_sha256(
        {
            "seed_plan": seeds.audit_sha256,
            "sample_indices": array_audit_sha256(sample_indices),
            "ordered_trial_uids": array_audit_sha256(
                frame["trial_uid"].to_numpy(dtype="U")
            ),
            "source_row_indices": array_audit_sha256(mappings),
            "trial_uid_sha256": uid_hash,
        }
    )
    return LabelPermutationPlan(
        seeds,
        _read_only(sample_indices),
        _read_only(mappings),
        uid_hash,
        audit,
    )


def apply_label_permutation(
    labels: Sequence[Any] | np.ndarray,
    plan: LabelPermutationPlan,
    replicate: int,
) -> np.ndarray:
    replicate_index = int(replicate) - 1
    if not 0 <= replicate_index < plan.replicates:
        raise TrajectoryEvaluationError("replicate index is outside the label plan")
    values = _labels(labels, len(plan.sample_indices))
    return _read_only(values[plan.source_row_indices[replicate_index]])


def replay_label_permutation_plan(
    metadata: pd.DataFrame, plan: LabelPermutationPlan
) -> bool:
    replayed = make_label_permutation_plan(metadata, seed_plan=plan.seed_plan)
    return bool(
        np.array_equal(replayed.sample_indices, plan.sample_indices)
        and np.array_equal(replayed.source_row_indices, plan.source_row_indices)
        and replayed.audit_sha256 == plan.audit_sha256
    )


def run_subject_runhalf_probe(
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    geometry: str,
    representation: str,
    provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Predict subject across the two frozen, disjoint 0train run halves."""

    allowed_representations = tuple(
        str(value) for value in config["subject_probe"]["representations"]
    )
    if str(geometry) != str(config["subject_probe"]["geometry"]) or str(
        representation
    ) not in allowed_representations:
        raise TrajectoryEvaluationError("subject probe geometry/representation is not frozen")
    x = _features(features)
    frame = _metadata(metadata, require_class=False)
    _validate_trial_counts(frame, config, require_class=False)
    if len(x) != len(frame):
        raise TrajectoryEvaluationError("feature and metadata row counts differ")
    subjects = tuple(int(value) for value in config["dataset"]["subjects"])
    if tuple(sorted(frame["subject"].unique())) != subjects:
        raise TrajectoryEvaluationError("subject-probe metadata has the wrong subjects")
    directions = config["subject_probe"]["directions"]
    expected_names = ("A_TO_B", "B_TO_A")
    if tuple(directions) != expected_names:
        raise TrajectoryEvaluationError("subject-probe directions/order are not frozen")
    rows: list[dict[str, Any]] = []
    labels = frame["subject"].astype(str).to_numpy()
    for split in expected_names:
        specification = directions[split]
        train_runs = tuple(int(value) for value in specification["train_runs"])
        evaluation_runs = tuple(int(value) for value in specification["evaluation_runs"])
        if set(train_runs) & set(evaluation_runs):
            raise TrajectoryEvaluationError("subject-probe run halves overlap")
        train = frame["run"].isin(train_runs).to_numpy()
        test = frame["run"].isin(evaluation_runs).to_numpy()
        if np.any(train & test) or np.any(~(train | test)):
            raise TrajectoryEvaluationError("subject-probe halves do not partition trials")
        train_uids = frame.loc[train, "trial_uid"].tolist()
        test_uids = frame.loc[test, "trial_uid"].tolist()
        if set(train_uids) & set(test_uids):
            raise TrajectoryEvaluationError("subject-probe train/test UIDs overlap")
        per_subject_train = frame.loc[train].groupby("subject", observed=True).size()
        per_subject_test = frame.loc[test].groupby("subject", observed=True).size()
        if not (per_subject_train == 144).all() or not (per_subject_test == 144).all():
            # Synthetic tests can declare a smaller balanced contract explicitly.
            expected_half = int(config.get("subject_probe", {}).get("trials_per_subject_half", 144))
            if not (per_subject_train == expected_half).all() or not (
                per_subject_test == expected_half
            ).all():
                raise TrajectoryEvaluationError("subject-probe half counts are not balanced")
        fitted = fit_source_scaled_logistic(x[train], labels[train], config)
        if fitted.audit.status == "PASS":
            prediction = fitted.predict(x[test]).astype(str)
            truth = labels[test]
            ba = float(balanced_accuracy_score(truth, prediction))
            accuracy = float(accuracy_score(truth, prediction))
            prediction_hash = stable_json_sha256(
                {"truth": truth.tolist(), "predicted": prediction.tolist()}
            )
        else:
            ba = accuracy = np.nan
            prediction_hash = ""
        rows.append(
            {
                **_provenance(provenance, status=fitted.audit.status),
                "geometry": str(geometry),
                "representation": str(representation),
                "split": split,
                "train_runs": json.dumps(list(train_runs), separators=(",", ":")),
                "evaluation_runs": json.dumps(
                    list(evaluation_runs), separators=(",", ":")
                ),
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "train_uid_sha256": trial_uid_sha256(train_uids),
                "test_uid_sha256": trial_uid_sha256(test_uids),
                "chance_level": float(config["subject_probe"]["chance"]),
                "balanced_accuracy": ba,
                "accuracy": accuracy,
                "direction_average_ba": np.nan,
                "direction_average_accuracy": np.nan,
                "prediction_sha256": prediction_hash,
                "classifier_config_sha256": fitted.audit.config_sha256,
                "convergence_warning": fitted.audit.convergence_warning,
                "warning_messages": json.dumps(
                    list(fitted.audit.warning_messages),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    result = pd.DataFrame.from_records(rows)
    if (result["status"] == "PASS").all():
        result["direction_average_ba"] = float(result["balanced_accuracy"].mean())
        result["direction_average_accuracy"] = float(result["accuracy"].mean())
    return result.reindex(columns=SUBJECT_PROBE_COLUMNS)


def balanced_factor_decomposition(
    scalar_values: pd.DataFrame | np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    geometry: str,
    scalar_names: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute the exact balanced subject/class/interaction/residual SS grid."""

    frame = _metadata(metadata)
    _validate_trial_counts(frame, config, require_class=True)
    names = tuple(
        str(value)
        for value in (
            scalar_names
            if scalar_names is not None
            else config["representations"]["scalar_columns"]
        )
    )
    frozen_names = tuple(
        str(value) for value in config["representations"]["scalar_columns"]
    )
    if names != frozen_names or str(geometry) not in tuple(
        str(value) for value in config["factor_decomposition"]["geometries"]
    ):
        raise TrajectoryEvaluationError("factor scalar order/geometry is not frozen")
    if isinstance(scalar_values, pd.DataFrame):
        missing = set(names) - set(scalar_values.columns)
        if missing:
            raise TrajectoryEvaluationError(f"scalar table is missing: {sorted(missing)}")
        values = scalar_values.loc[:, list(names)].to_numpy(dtype=np.float64)
    else:
        values = np.asarray(scalar_values, dtype=np.float64)
    if values.shape != (len(frame), len(names)) or not np.isfinite(values).all():
        raise TrajectoryEvaluationError("scalar values have the wrong shape or are non-finite")
    factor = config["factor_decomposition"]
    expected_subjects = int(factor["n_subjects"])
    expected_classes = int(factor["n_classes"])
    expected_per_cell = int(factor["n_per_cell"])
    subject_values = tuple(int(value) for value in config["dataset"]["subjects"])
    class_values = tuple(str(value) for value in config["dataset"]["classes"])
    if len(subject_values) != expected_subjects or len(class_values) != expected_classes:
        raise TrajectoryEvaluationError("factor config dimensions are inconsistent")
    cell_counts = frame.groupby(["subject", "class_label"], observed=True).size()
    if len(cell_counts) != expected_subjects * expected_classes or not (
        cell_counts == expected_per_cell
    ).all():
        raise TrajectoryEvaluationError("factor decomposition design is not exactly balanced")
    threshold = float(config["geometry"]["hard_gate"]["ss_closure_relative_error_max"])
    subject_codes = frame["subject"].to_numpy()
    class_codes = frame["class_label"].to_numpy()
    rows: list[dict[str, Any]] = []
    for scalar_index, scalar_name in enumerate(names):
        y = values[:, scalar_index]
        grand = float(y.mean())
        subject_means = {
            subject: float(y[subject_codes == subject].mean())
            for subject in subject_values
        }
        class_means = {
            label: float(y[class_codes == label].mean()) for label in class_values
        }
        cell_means = {
            (subject, label): float(
                y[(subject_codes == subject) & (class_codes == label)].mean()
            )
            for subject in subject_values
            for label in class_values
        }
        ss_subject = expected_classes * expected_per_cell * sum(
            (subject_means[subject] - grand) ** 2 for subject in subject_values
        )
        ss_class = expected_subjects * expected_per_cell * sum(
            (class_means[label] - grand) ** 2 for label in class_values
        )
        ss_interaction = expected_per_cell * sum(
            (
                cell_means[(subject, label)]
                - subject_means[subject]
                - class_means[label]
                + grand
            )
            ** 2
            for subject in subject_values
            for label in class_values
        )
        ss_residual = 0.0
        for subject in subject_values:
            for label in class_values:
                selected = y[(subject_codes == subject) & (class_codes == label)]
                ss_residual += float(
                    np.square(selected - cell_means[(subject, label)]).sum()
                )
        ss_total = float(np.square(y - grand).sum())
        components = np.asarray(
            [ss_subject, ss_class, ss_interaction, ss_residual], dtype=np.float64
        )
        if np.any(components < -100 * np.finfo(np.float64).eps * max(ss_total, 1.0)):
            raise TrajectoryEvaluationError("factor SS component became materially negative")
        components = np.maximum(components, 0.0)
        degenerate = bool(ss_total == 0.0)
        if degenerate:
            eta = np.full(4, np.nan)
            closure = np.nan
            status = "FAILED"
        else:
            eta = components / ss_total
            closure = float(abs(ss_total - float(components.sum())) / ss_total)
            status = "PASS" if closure <= threshold else "FAILED"
        rows.append(
            {
                **_provenance(provenance, status=status),
                "geometry": str(geometry),
                "scalar": scalar_name,
                "n_subjects": expected_subjects,
                "n_classes": expected_classes,
                "n_per_cell": expected_per_cell,
                "grand_mean": grand,
                "ss_subject": float(components[0]),
                "ss_class": float(components[1]),
                "ss_interaction": float(components[2]),
                "ss_residual": float(components[3]),
                "ss_total": ss_total,
                "eta2_subject": float(eta[0]),
                "eta2_class": float(eta[1]),
                "eta2_interaction": float(eta[2]),
                "eta2_residual": float(eta[3]),
                "ss_reconstruction_relative_error": closure,
                "degenerate": degenerate,
                "uses_p_value": False,
            }
        )
    return pd.DataFrame.from_records(rows).reindex(columns=FACTOR_COLUMNS)


def _spd_stack(covariances: np.ndarray) -> np.ndarray:
    value = np.asarray(covariances, dtype=np.float64)
    if value.ndim != 3 or value.shape[1] != value.shape[2] or len(value) < 1:
        raise TrajectoryEvaluationError("MDM input must have shape (N,C,C)")
    if not np.isfinite(value).all() or not np.allclose(
        value, value.transpose(0, 2, 1), rtol=0.0, atol=1e-12
    ):
        raise TrajectoryEvaluationError("MDM input is non-finite or non-symmetric")
    if np.min(np.linalg.eigvalsh(value)) <= 0.0:
        raise TrajectoryEvaluationError("MDM input contains a non-SPD matrix")
    return value


def run_mdm_loso(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    representation: str,
    metric: str = "riemann",
    target_subjects: Sequence[int] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run the fixed metric-native MDM contextual LOSO control."""

    if str(representation) not in {"LOCAL_BARYCENTER", "WHOLE-1000"}:
        raise TrajectoryEvaluationError("MDM representation is not a frozen contextual control")
    matrices = _spd_stack(covariances)
    frame = _metadata(metadata)
    _validate_trial_counts(frame, config, require_class=True)
    if len(matrices) != len(frame):
        raise TrajectoryEvaluationError("MDM covariance and metadata rows differ")
    if metric != "riemann":
        raise TrajectoryEvaluationError("Trajectory v0 contextual MDM metric is riemann")
    class_order = tuple(str(value) for value in config["dataset"]["classes"])
    labels = _labels(frame["class_label"], len(frame))
    subjects = tuple(int(value) for value in config["dataset"]["subjects"])
    targets = subjects if target_subjects is None else tuple(int(value) for value in target_subjects)
    rows: list[dict[str, Any]] = []
    for target in targets:
        test = frame["subject"].to_numpy() == target
        train = ~test
        train_uids = frame.loc[train, "trial_uid"].tolist()
        test_uids = frame.loc[test, "trial_uid"].tolist()
        if set(train_uids) & set(test_uids):
            raise TrajectoryEvaluationError("MDM LOSO train/test UID overlap")
        model = MDM(metric="riemann", n_jobs=1)
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(matrices[train], labels[train])
        convergence = tuple(
            str(item.message)
            for item in caught
            if "convergence" in str(item.message).lower()
        )
        status = "FAILED" if convergence else "PASS"
        if status == "PASS":
            prediction = np.asarray(model.predict(matrices[test])).astype(str)
            metrics = evaluate_target_predictions(
                labels[test], prediction, class_order=class_order
            )
        else:
            metrics = _failed_class_metrics(class_order)
        rows.append(
            {
                **_provenance(provenance, status=status),
                "representation": str(representation),
                "target_subject": target,
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "train_uid_sha256": trial_uid_sha256(train_uids),
                "test_uid_sha256": trial_uid_sha256(test_uids),
                "metric": "riemann",
                **metrics,
                "convergence_warning": bool(convergence),
                "warning_messages": json.dumps(
                    [str(item.message) for item in caught],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame.from_records(rows).reindex(columns=MDM_COLUMNS)


def _observed_ba_by_subject(
    observed_loso: pd.DataFrame,
    *,
    geometry: str,
    representation: str,
    expected_subjects: Sequence[int],
) -> dict[int, float]:
    required = {"geometry", "representation", "target_subject", "balanced_accuracy", "status"}
    if not isinstance(observed_loso, pd.DataFrame) or required - set(observed_loso.columns):
        raise TrajectoryEvaluationError("observed LOSO table lacks required columns")
    selected = observed_loso[
        (observed_loso["geometry"].astype(str) == str(geometry))
        & (observed_loso["representation"].astype(str) == str(representation))
    ]
    expected = tuple(int(value) for value in expected_subjects)
    if len(selected) != len(expected) or tuple(
        sorted(pd.to_numeric(selected["target_subject"]).astype(int))
    ) != expected:
        raise TrajectoryEvaluationError("observed LOSO condition is incomplete")
    if not (selected["status"] == "PASS").all() or selected["balanced_accuracy"].isna().any():
        raise TrajectoryEvaluationError("observed LOSO condition contains a FAILED row")
    return {
        int(row.target_subject): float(row.balanced_accuracy)
        for row in selected.itertuples(index=False)
    }


def _assert_null_plan_contract(
    config: Mapping[str, Any],
    *,
    family: str,
    seed_plan: NullSeedPlan,
) -> int:
    section_name = "order_shuffle" if family == "order" else "label_destruction"
    expected_tag = ORDER_STREAM_TAG if family == "order" else LABEL_STREAM_TAG
    section = config["nulls"][section_name]
    expected_replicates = int(section["replicates"])
    entropy = [int(value) for value in section["seedsequence_entropy"]]
    if int(config["project"]["seed"]) != MASTER_SEED:
        raise TrajectoryEvaluationError("project master seed is not frozen")
    if entropy != [MASTER_SEED, expected_tag]:
        raise TrajectoryEvaluationError("null SeedSequence entropy is not frozen")
    if int(str(section["stream_tag_hex"]), 16) != expected_tag:
        raise TrajectoryEvaluationError("null stream tag is not frozen")
    if (
        seed_plan.master_seed != MASTER_SEED
        or seed_plan.stream_tag != expected_tag
        or len(seed_plan.seeds) != expected_replicates
        or not replay_null_seed_plan(seed_plan)
    ):
        raise TrajectoryEvaluationError("null seed plan fails exact replay/config contract")
    return expected_replicates


def _null_subject_rows(
    loso: pd.DataFrame,
    *,
    geometry: str,
    representation: str,
    replicate: int,
    replicate_seed: int,
    observed: Mapping[int, float],
    provenance: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in loso.itertuples(index=False):
        subject = int(result.target_subject)
        status = str(result.status)
        rows.append(
            {
                **_provenance(provenance, status=status),
                "geometry": str(geometry),
                "representation": str(representation),
                "replicate": int(replicate),
                "replicate_seed": int(replicate_seed),
                "target_subject": subject,
                "balanced_accuracy": result.balanced_accuracy,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "observed_ba": float(observed[subject]),
                "subject_null_median_ba": np.nan,
                "subject_effect": np.nan,
                "train_uid_sha256": str(result.train_uid_sha256),
                "test_uid_sha256": str(result.test_uid_sha256),
                "classifier_status": status,
                "convergence_warning": bool(result.convergence_warning),
                "warning_messages": str(result.warning_messages),
            }
        )
    return rows


def _finalize_null_subject_table(
    rows: list[dict[str, Any]],
    *,
    expected_subjects: Sequence[int],
    expected_replicates: int,
) -> pd.DataFrame:
    table = pd.DataFrame.from_records(rows).reindex(columns=NULL_SUBJECT_COLUMNS)
    expected = {
        (replicate, int(subject))
        for replicate in range(1, expected_replicates + 1)
        for subject in expected_subjects
    }
    observed_pairs = set(
        zip(
            pd.to_numeric(table["replicate"]).astype(int),
            pd.to_numeric(table["target_subject"]).astype(int),
            strict=True,
        )
    )
    complete = (
        observed_pairs == expected
        and len(table) == len(expected)
        and (table["status"] == "PASS").all()
        and table["balanced_accuracy"].notna().all()
    )
    if complete:
        medians = table.groupby("target_subject", observed=True)[
            "balanced_accuracy"
        ].median()
        table["subject_null_median_ba"] = table["target_subject"].map(medians)
        table["subject_effect"] = (
            table["observed_ba"] - table["subject_null_median_ba"]
        )
    return table


def run_order_shuffle_null(
    distances_or_path: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    plan: OrderPermutationPlan,
    observed_loso: pd.DataFrame,
    *,
    geometry: str,
    representation: str = "PATH_D10",
    provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate every frozen order-shuffle replicate on one PATH condition."""

    if str(representation) != "PATH_D10" or str(geometry) not in {"AIRM", "LE"}:
        raise TrajectoryEvaluationError("order-null grid is restricted to AIRM/LE PATH_D10")
    frame = _plan_metadata(metadata)
    if trial_uid_sha256(frame["trial_uid"].tolist()) != plan.trial_uid_sha256:
        raise TrajectoryEvaluationError("order plan UID contract differs from metadata")
    expected_replicates = _assert_null_plan_contract(
        config, family="order", seed_plan=plan.seed_plan
    )
    if not replay_order_permutation_plan(frame, plan):
        raise TrajectoryEvaluationError("order permutation plan fails exact replay")
    subjects = tuple(int(value) for value in config["dataset"]["subjects"])
    observed = _observed_ba_by_subject(
        observed_loso,
        geometry=geometry,
        representation=representation,
        expected_subjects=subjects,
    )
    rows: list[dict[str, Any]] = []
    for replicate, seed in enumerate(plan.seed_plan.seeds, start=1):
        shuffled = apply_order_permutation(distances_or_path, plan, replicate)
        loso = run_class_loso(
            shuffled,
            frame,
            config,
            geometry=geometry,
            representation=representation,
            target_subjects=subjects,
            provenance=provenance,
        )
        rows.extend(
            _null_subject_rows(
                loso,
                geometry=geometry,
                representation=representation,
                replicate=replicate,
                replicate_seed=int(seed),
                observed=observed,
                provenance=provenance,
            )
        )
    return _finalize_null_subject_table(
        rows, expected_subjects=subjects, expected_replicates=expected_replicates
    )


def run_label_destruction_null(
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    plan: LabelPermutationPlan,
    observed_loso: pd.DataFrame,
    *,
    geometry: str,
    representation: str,
    provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate one representation using the shared within-cell label plan."""

    if str(geometry) != "AIRM" or str(representation) not in {
        "PATH_D10",
        "BAG_CANON_D10",
    }:
        raise TrajectoryEvaluationError(
            "label-null grid is restricted to AIRM PATH_D10/BAG_CANON_D10"
        )
    x = _features(features)
    frame = _plan_metadata(metadata)
    if len(x) != len(frame):
        raise TrajectoryEvaluationError("label-null features and metadata differ")
    if trial_uid_sha256(frame["trial_uid"].tolist()) != plan.trial_uid_sha256:
        raise TrajectoryEvaluationError("label plan UID contract differs from metadata")
    expected_replicates = _assert_null_plan_contract(
        config, family="label", seed_plan=plan.seed_plan
    )
    if not replay_label_permutation_plan(frame, plan):
        raise TrajectoryEvaluationError("label permutation plan fails exact replay")
    subjects = tuple(int(value) for value in config["dataset"]["subjects"])
    observed = _observed_ba_by_subject(
        observed_loso,
        geometry=geometry,
        representation=representation,
        expected_subjects=subjects,
    )
    labels = frame["class_label"].to_numpy()
    # The plan is generated once and replayed here; callers reuse this same
    # object for PATH and BAG_CANON so RNG can never diverge by representation.
    rows: list[dict[str, Any]] = []
    for replicate, seed in enumerate(plan.seed_plan.seeds, start=1):
        permuted_labels = apply_label_permutation(labels, plan, replicate)
        # Preserve the exact class multiset within every subject×run cell.
        for _, group in frame.groupby(["subject", "run"], sort=True, observed=True):
            positions = group.index.to_numpy(dtype=np.int64)
            if sorted(permuted_labels[positions].tolist()) != sorted(labels[positions].tolist()):
                raise TrajectoryEvaluationError("label null changed a subject×run class multiset")
        loso = run_class_loso(
            x,
            frame,
            config,
            geometry=geometry,
            representation=representation,
            labels=permuted_labels,
            target_subjects=subjects,
            provenance=provenance,
        )
        rows.extend(
            _null_subject_rows(
                loso,
                geometry=geometry,
                representation=representation,
                replicate=replicate,
                replicate_seed=int(seed),
                observed=observed,
                provenance=provenance,
            )
        )
    return _finalize_null_subject_table(
        rows, expected_subjects=subjects, expected_replicates=expected_replicates
    )


def paired_median_path_minus_bag(
    path_loso: pd.DataFrame,
    bag_loso: pd.DataFrame,
    *,
    geometry: str = "AIRM",
) -> float:
    """Return the frozen median of nine paired subject PATH-minus-BAG BAs."""

    def selected(table: pd.DataFrame, representation: str) -> pd.DataFrame:
        value = table[
            (table["geometry"].astype(str) == str(geometry))
            & (table["representation"].astype(str) == representation)
        ][["target_subject", "balanced_accuracy", "status"]].copy()
        if len(value) == 0 and table["representation"].nunique() == 1:
            value = table[["target_subject", "balanced_accuracy", "status"]].copy()
        if len(value) != 9 or not (value["status"] == "PASS").all():
            raise TrajectoryEvaluationError("paired PATH/BAG observed grid is incomplete")
        return value.rename(columns={"balanced_accuracy": representation})

    path = selected(path_loso, "PATH_D10")
    bag = selected(bag_loso, "BAG_CANON_D10")
    merged = path.merge(bag, on="target_subject", validate="one_to_one", suffixes=("_p", "_b"))
    if len(merged) != 9:
        raise TrajectoryEvaluationError("PATH/BAG subject identities do not pair exactly")
    return float(np.median(merged["PATH_D10"] - merged["BAG_CANON_D10"]))


def summarize_null_distribution(
    subject_metrics: pd.DataFrame,
    observed_loso: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    family: str,
    geometry: str,
    representation: str,
    median_subject_path_minus_bag: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> NullSummary:
    """Build the frozen median-subject Monte Carlo operand without available cases."""

    family_name = str(family).lower()
    if family_name not in {"order", "order_shuffle", "label", "label_destruction"}:
        raise TrajectoryEvaluationError("null family must be order or label")
    is_order = family_name.startswith("order")
    expected_replicates = int(
        config["nulls"]["order_shuffle" if is_order else "label_destruction"][
            "replicates"
        ]
    )
    subjects = tuple(int(value) for value in config["dataset"]["subjects"])
    observed = _observed_ba_by_subject(
        observed_loso,
        geometry=geometry,
        representation=representation,
        expected_subjects=subjects,
    )
    selected = subject_metrics[
        (subject_metrics["geometry"].astype(str) == str(geometry))
        & (subject_metrics["representation"].astype(str) == str(representation))
    ].copy()
    expected_pairs = {
        (replicate, subject)
        for replicate in range(1, expected_replicates + 1)
        for subject in subjects
    }
    actual_pairs = set(
        zip(
            pd.to_numeric(selected.get("replicate", pd.Series(dtype=int))).astype(int),
            pd.to_numeric(selected.get("target_subject", pd.Series(dtype=int))).astype(int),
            strict=True,
        )
    )
    complete = bool(
        len(selected) == len(expected_pairs)
        and actual_pairs == expected_pairs
        and (selected["status"] == "PASS").all()
        and selected["balanced_accuracy"].notna().all()
    )
    observed_statistic = float(np.median(list(observed.values())))
    if complete:
        grouped = selected.groupby("replicate", sort=True, observed=True)[
            "balanced_accuracy"
        ].median()
        if not np.array_equal(grouped.index.to_numpy(), np.arange(1, expected_replicates + 1)):
            raise TrajectoryEvaluationError("null replicate identities are incomplete")
        statistics = grouped.to_numpy(dtype=np.float64)
        null_median = float(np.median(statistics))
        effect = observed_statistic - null_median
        exceedance = int(np.count_nonzero(statistics >= observed_statistic))
        p_value = float((1 + exceedance) / (expected_replicates + 1))
        alpha = float(config["nulls"]["monte_carlo"]["alpha"])
        operand_pass = bool(effect > 0.0 and p_value <= alpha)
        if is_order:
            if median_subject_path_minus_bag is None or not np.isfinite(
                median_subject_path_minus_bag
            ):
                raise TrajectoryEvaluationError(
                    "order summary requires median subject PATH-minus-BAG"
                )
            operand_pass = bool(operand_pass and median_subject_path_minus_bag > 0.0)
        status = "PASS"
        statistics_out = _read_only(statistics)
        row = {
            **_provenance(provenance, status=status),
            "geometry": str(geometry),
            "representation": str(representation),
            "observed_median_subject_ba": observed_statistic,
            "null_replicates": expected_replicates,
            "null_median": null_median,
            "null_mean": float(statistics.mean()),
            "null_sd_ddof1": float(statistics.std(ddof=1)),
            "null_min": float(statistics.min()),
            "null_max": float(statistics.max()),
            "effect": effect,
            "p_value": p_value,
            "exceedance_count": exceedance,
            "hypothesis_operand_pass": operand_pass,
        }
        if is_order:
            row["median_subject_path_minus_bag"] = float(
                median_subject_path_minus_bag
            )
    else:
        statistics_out = _read_only(np.asarray([], dtype=np.float64))
        row = {
            **_provenance(provenance, status="FAILED"),
            "geometry": str(geometry),
            "representation": str(representation),
            "observed_median_subject_ba": observed_statistic,
            "null_replicates": expected_replicates,
            "null_median": np.nan,
            "null_mean": np.nan,
            "null_sd_ddof1": np.nan,
            "null_min": np.nan,
            "null_max": np.nan,
            "effect": np.nan,
            "p_value": np.nan,
            "exceedance_count": np.nan,
            "hypothesis_operand_pass": False,
        }
        if is_order:
            row["median_subject_path_minus_bag"] = median_subject_path_minus_bag
    columns = ORDER_GROUP_COLUMNS if is_order else LABEL_GROUP_COLUMNS
    table = pd.DataFrame.from_records([row]).reindex(columns=columns)
    return NullSummary(table, statistics_out, complete)


def evaluate_frozen_verdict(
    *,
    numerical_gate_pass: bool,
    technical_grid_pass: bool,
    label_path_effect: float | None = None,
    label_path_p: float | None = None,
    label_bag_effect: float | None = None,
    label_bag_p: float | None = None,
    order_path_effect: float | None = None,
    order_path_p: float | None = None,
    median_subject_path_minus_bag: float | None = None,
    alpha: float = 0.05,
) -> FrozenVerdict:
    """Evaluate the frozen hypotheses and terminal decision in protocol order."""

    if float(alpha) != 0.05:
        raise TrajectoryEvaluationError("frozen verdict alpha must be exactly 0.05")
    operands = {
        "label_path_effect": label_path_effect,
        "label_path_p": label_path_p,
        "label_bag_effect": label_bag_effect,
        "label_bag_p": label_bag_p,
        "order_path_effect": order_path_effect,
        "order_path_p": order_path_p,
        "median_subject_path_minus_bag": median_subject_path_minus_bag,
        "alpha": float(alpha),
    }
    if not numerical_gate_pass:
        return FrozenVerdict(
            "UNASSESSED",
            "UNASSESSED — NUMERICAL/DATA FAILURE",
            None,
            None,
            None,
            median_subject_path_minus_bag,
            operands,
        )
    if not technical_grid_pass:
        return FrozenVerdict(
            "UNASSESSED",
            "UNASSESSED—PROTOCOL/TECHNICAL FAILURE",
            None,
            None,
            None,
            median_subject_path_minus_bag,
            operands,
        )
    required = (
        label_path_effect,
        label_path_p,
        label_bag_effect,
        label_bag_p,
        order_path_effect,
        order_path_p,
        median_subject_path_minus_bag,
    )
    if any(value is None or not np.isfinite(float(value)) for value in required):
        return FrozenVerdict(
            "UNASSESSED",
            "UNASSESSED—PROTOCOL/TECHNICAL FAILURE",
            None,
            None,
            None,
            median_subject_path_minus_bag,
            operands,
        )
    h_path = bool(float(label_path_effect) > 0.0 and float(label_path_p) <= alpha)
    h_bag = bool(float(label_bag_effect) > 0.0 and float(label_bag_p) <= alpha)
    h_order = bool(
        float(order_path_effect) > 0.0
        and float(order_path_p) <= alpha
        and float(median_subject_path_minus_bag) > 0.0
    )
    if h_path and h_order:
        verdict = "GO_TRAJECTORY_ORDER"
    elif h_bag and not h_order:
        verdict = "GO_UNORDERED_DISTRIBUTION"
    elif not h_path and not h_bag:
        verdict = "STOP_LOCAL_TRAJECTORY_V0"
    else:
        verdict = "MIXED_TRAJECTORY_SIGNAL"
    return FrozenVerdict(
        verdict,
        "",
        h_path,
        h_bag,
        h_order,
        float(median_subject_path_minus_bag),
        operands,
    )


__all__ = [
    "MASTER_SEED",
    "ORDER_STREAM_TAG",
    "LABEL_STREAM_TAG",
    "NULL_REPLICATES",
    "DEFAULT_CLASS_ORDER",
    "COMMON_COLUMNS",
    "CLASS_LOSO_COLUMNS",
    "SUBJECT_PROBE_COLUMNS",
    "FACTOR_COLUMNS",
    "NULL_SUBJECT_COLUMNS",
    "ORDER_GROUP_COLUMNS",
    "LABEL_GROUP_COLUMNS",
    "MDM_COLUMNS",
    "TrajectoryEvaluationError",
    "LogisticFitAudit",
    "ScaledLogisticFit",
    "NullSeedPlan",
    "OrderPermutationPlan",
    "LabelPermutationPlan",
    "NullSummary",
    "FrozenVerdict",
    "stable_json_sha256",
    "array_audit_sha256",
    "trial_uid_sha256",
    "classifier_config_sha256",
    "fit_source_scaled_logistic",
    "evaluate_target_predictions",
    "run_class_loso",
    "make_null_seed_plan",
    "replay_null_seed_plan",
    "lexicographic_state_permutations",
    "make_order_permutation_plan",
    "replay_order_permutation_plan",
    "apply_order_permutation",
    "make_label_permutation_plan",
    "apply_label_permutation",
    "replay_label_permutation_plan",
    "run_subject_runhalf_probe",
    "balanced_factor_decomposition",
    "run_mdm_loso",
    "run_order_shuffle_null",
    "run_label_destruction_null",
    "paired_median_path_minus_bag",
    "summarize_null_distribution",
    "evaluate_frozen_verdict",
]
