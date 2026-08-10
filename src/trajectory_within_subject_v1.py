"""Frozen evaluation primitives for the trajectory within-subject audit v1.

This module contains no data download, result-selected branch, or output writer.
It reuses the exact v0 classifier family and S5 edge action while changing only
the preregistered evaluation boundary: within one subject/session (Stage W) and
between the same subject's two sessions (Stage X).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
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

from src.evaluation_trajectory_v0 import EDGE_REINDEX_TABLE
from src.trajectory_geometry_v0 import (
    ALL_PERMUTATIONS_5,
    PATH_D10_NAMES,
    SCALAR_11_NAMES,
    bag_canon_d10,
    path_d10,
    permute_distance_matrix,
)


MASTER_SEED = 20260810
CLASSIFIER_RANDOM_STATE = 20260809
LABEL_STREAM_TAG = 0x4C4142454C5758
ORDER_STREAM_TAG = 0x4F52444552
NULL_REPLICATES = 1999
CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
SESSION_ORDER = ("0train", "1test")
RUN_ORDER = (0, 1, 2, 3, 4, 5)
HALF_A = (0, 1, 2)
HALF_B = (3, 4, 5)
REPRESENTATIONS = ("PATH_D10", "BAG_CANON_D10", "SCALARS_11")

IDENTITY_COLUMNS = (
    "global_sample_index",
    "sample_index",
    "subject",
    "session",
    "run",
    "trial_id",
    "trial_uid",
    "class_label",
)

SCORE_COLUMNS = (
    "stage",
    "representation",
    "subject",
    "session",
    "direction",
    "train_session",
    "test_session",
    "train_runs",
    "test_runs",
    "train_n",
    "test_n",
    "train_subjects",
    "test_subjects",
    "train_uid_sha256",
    "test_uid_sha256",
    "scaler_fit_uid_sha256",
    "scaler_mean_sha256",
    "scaler_scale_sha256",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "recall_left_hand",
    "recall_right_hand",
    "recall_feet",
    "recall_tongue",
    "confusion_matrix_json",
    "prediction_sha256",
    "status",
    "convergence_warning",
    "warning_messages",
)


class TrajectoryWithinSubjectError(RuntimeError):
    """A frozen contract, leakage, numerical, or evaluation violation."""


class IncompleteRequiredGrid(TrajectoryWithinSubjectError):
    """Raised instead of silently using available required fits."""


@dataclass(frozen=True)
class FrozenFit:
    scaler: StandardScaler
    model: LogisticRegression
    status: str
    convergence_warning: bool
    warning_messages: tuple[str, ...]
    scaler_mean_sha256: str
    scaler_scale_sha256: str


@dataclass(frozen=True)
class MonteCarloResult:
    observed: float
    null_median: float
    effect: float
    p_value: float
    exceedance_count: int
    replicates: int
    passed: bool


@dataclass(frozen=True)
class TerminalDecision:
    decision: str
    stage_w_pass: bool | None
    stage_x_pass: bool | None
    stage_o_pass: bool | None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def uid_sha256(values: Sequence[str]) -> str:
    items = [str(value) for value in values]
    if not items or len(items) != len(set(items)):
        raise TrajectoryWithinSubjectError("trial UIDs must be nonempty and unique")
    return stable_json_sha256({"trial_uids": sorted(items)})


def load_frozen_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TrajectoryWithinSubjectError(f"cannot read config: {config_path}") from error
    validate_frozen_config(config)
    protocol_path = config_path.parents[1] / str(config["protocol"]["path"])
    if sha256_file(protocol_path) != str(config["protocol"]["sha256"]):
        raise TrajectoryWithinSubjectError("protocol SHA-256 differs from frozen config")
    return config


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    required = {
        "protocol",
        "project",
        "dataset",
        "preprocessing",
        "window5",
        "representations",
        "hard_gates",
        "classifier",
        "stage_w",
        "stage_x",
        "nulls",
        "decisions",
        "outputs",
    }
    if required - set(config):
        raise TrajectoryWithinSubjectError(
            f"config is missing sections: {sorted(required - set(config))}"
        )
    if str(config["protocol"].get("version")) != "1.0":
        raise TrajectoryWithinSubjectError("protocol version must be 1.0")
    if str(config["protocol"].get("reference_commit")) != (
        "fcb55ccdccdd4613290b8e8d93be91ea256edd45"
    ):
        raise TrajectoryWithinSubjectError("reference commit changed")
    dataset = config["dataset"]
    if tuple(int(value) for value in dataset["subjects"]) != tuple(range(1, 10)):
        raise TrajectoryWithinSubjectError("subjects must be exactly 1..9")
    if tuple(str(value) for value in dataset["sessions"]) != SESSION_ORDER:
        raise TrajectoryWithinSubjectError("sessions/order changed")
    if tuple(int(value) for value in dataset["runs"]) != RUN_ORDER:
        raise TrajectoryWithinSubjectError("runs/order changed")
    if tuple(str(value) for value in dataset["classes"]) != CLASS_ORDER:
        raise TrajectoryWithinSubjectError("class order changed")
    if int(dataset["expected_trials_total"]) != 5184:
        raise TrajectoryWithinSubjectError("combined trial count changed")
    prep = config["preprocessing"]
    exact_prep = {
        "bandpass_hz": [8.0, 32.0],
        "epoch_tmin_seconds": 0.0,
        "epoch_tmax_seconds": 3.996,
        "sampling_frequency_hz": 250.0,
        "samples_per_trial": 1000,
        "resample_hz": None,
        "baseline": None,
        "epoch_cache_dtype": "float32",
        "covariance_estimator": "oas",
        "covariance_dtype": "float64",
        "extra_diagonal_loading": "none",
        "eigenvalue_clipping": False,
    }
    for key, expected in exact_prep.items():
        if prep.get(key) != expected:
            raise TrajectoryWithinSubjectError(f"preprocessing field changed: {key}")
    window = config["window5"]
    if (
        int(window["n_windows"]) != 5
        or int(window["samples_per_window"]) != 200
        or int(window["overlap_samples"]) != 0
    ):
        raise TrajectoryWithinSubjectError("five-window construction changed")
    reps = config["representations"]
    if tuple(reps["path_d10_columns"]) != PATH_D10_NAMES:
        raise TrajectoryWithinSubjectError("PATH_D10 order changed")
    if tuple(reps["scalar_columns"]) != SCALAR_11_NAMES:
        raise TrajectoryWithinSubjectError("SCALARS_11 order changed")
    classifier = config["classifier"]
    exact_classifier = {
        "scaler": "StandardScaler",
        "scaler_fit_scope": "train_only",
        "c": 1.0,
        "solver": "lbfgs",
        "max_iter": 5000,
        "tol": 1e-4,
        "class_weight": None,
        "random_state": CLASSIFIER_RANDOM_STATE,
        "tuning": False,
    }
    for key, expected in exact_classifier.items():
        if classifier.get(key) != expected:
            raise TrajectoryWithinSubjectError(f"classifier field changed: {key}")
    nulls = config["nulls"]
    if int(nulls["master_seed"]) != MASTER_SEED:
        raise TrajectoryWithinSubjectError("null master seed changed")
    if int(nulls["label_destruction"]["replicates"]) != NULL_REPLICATES:
        raise TrajectoryWithinSubjectError("label-null replicate count changed")
    if int(nulls["order_shuffle"]["replicates"]) != NULL_REPLICATES:
        raise TrajectoryWithinSubjectError("order-null replicate count changed")
    if int(str(nulls["label_destruction"]["stream_tag_hex"]), 16) != LABEL_STREAM_TAG:
        raise TrajectoryWithinSubjectError("label-null stream changed")
    if int(str(nulls["order_shuffle"]["stream_tag_hex"]), 16) != ORDER_STREAM_TAG:
        raise TrajectoryWithinSubjectError("order-null stream changed")


def validate_metadata(metadata: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        raise TrajectoryWithinSubjectError("metadata must be a nonempty DataFrame")
    missing = set(IDENTITY_COLUMNS) - set(metadata.columns)
    if missing:
        raise TrajectoryWithinSubjectError(f"metadata missing columns: {sorted(missing)}")
    frame = metadata.loc[:, list(IDENTITY_COLUMNS)].copy().reset_index(drop=True)
    if frame.isna().any().any():
        raise TrajectoryWithinSubjectError("metadata contains nulls")
    for column in ("global_sample_index", "sample_index", "subject", "run", "trial_id"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise TrajectoryWithinSubjectError(f"metadata {column} must be integer")
        frame[column] = values.astype(np.int64)
    for column in ("session", "trial_uid", "class_label"):
        frame[column] = frame[column].astype(str)
    expected_n = int(config["dataset"]["expected_trials_total"])
    if len(frame) != expected_n:
        raise TrajectoryWithinSubjectError(f"expected {expected_n} trials, got {len(frame)}")
    if not np.array_equal(frame["global_sample_index"], np.arange(expected_n)):
        raise TrajectoryWithinSubjectError("global_sample_index must be exact 0..N-1")
    if frame["global_sample_index"].duplicated().any() or frame["trial_uid"].duplicated().any():
        raise TrajectoryWithinSubjectError("global indices and UIDs must be unique")
    if tuple(sorted(frame["subject"].unique())) != tuple(range(1, 10)):
        raise TrajectoryWithinSubjectError("subject set changed")
    if set(frame["session"].unique()) != set(SESSION_ORDER):
        raise TrajectoryWithinSubjectError("session set changed")
    if set(frame["run"].unique()) != set(RUN_ORDER):
        raise TrajectoryWithinSubjectError("run set changed")
    if set(frame["class_label"].unique()) != set(CLASS_ORDER):
        raise TrajectoryWithinSubjectError("class set changed")
    dataset = config["dataset"]
    specifications = (
        (["session"], int(dataset["expected_trials_per_session"])),
        (["subject", "session"], int(dataset["expected_trials_per_subject_session"])),
        (
            ["subject", "session", "class_label"],
            int(dataset["expected_trials_per_subject_session_class"]),
        ),
        (
            ["subject", "session", "run"],
            int(dataset["expected_trials_per_subject_session_run"]),
        ),
        (
            ["subject", "session", "run", "class_label"],
            int(dataset["expected_trials_per_subject_session_run_class"]),
        ),
    )
    for columns, expected in specifications:
        counts = frame.groupby(columns, sort=True, observed=True).size()
        if counts.empty or not (counts == expected).all():
            raise TrajectoryWithinSubjectError(f"balanced count failure: {columns}")
    for session in SESSION_ORDER:
        selected = frame[frame["session"].eq(session)]
        if not np.array_equal(selected["sample_index"], np.arange(len(selected))):
            raise TrajectoryWithinSubjectError(f"{session} sample_index is not canonical")
    return frame


def make_seed_vector(family: str, *, replicates: int = NULL_REPLICATES) -> np.ndarray:
    normalized = str(family).lower()
    if normalized in {"label", "label_destruction"}:
        tag = LABEL_STREAM_TAG
    elif normalized in {"order", "order_shuffle"}:
        tag = ORDER_STREAM_TAG
    else:
        raise TrajectoryWithinSubjectError("family must be label or order")
    if int(replicates) < 1:
        raise TrajectoryWithinSubjectError("replicates must be positive")
    children = np.random.SeedSequence([MASTER_SEED, tag]).spawn(int(replicates))
    result = np.asarray(
        [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children],
        dtype=np.uint64,
    )
    result.setflags(write=False)
    return result


def _canonical_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    session_rank = {session: index for index, session in enumerate(SESSION_ORDER)}
    ordered = frame.assign(_session_rank=frame["session"].map(session_rank)).sort_values(
        ["subject", "_session_rank", "run", "global_sample_index"], kind="stable"
    )
    groups: list[np.ndarray] = []
    for _, group in ordered.groupby(
        ["subject", "_session_rank", "run"], sort=True, observed=True
    ):
        groups.append(group.index.to_numpy(dtype=np.int64))
    return groups


def permute_labels(
    labels: Sequence[str] | np.ndarray,
    metadata: pd.DataFrame,
    stored_seed: int,
) -> np.ndarray:
    frame = metadata.reset_index(drop=True)
    values = np.asarray(labels).astype(str)
    if values.shape != (len(frame),):
        raise TrajectoryWithinSubjectError("labels and metadata are misaligned")
    result = values.copy()
    generator = np.random.default_rng(int(stored_seed))
    for positions in _canonical_groups(frame):
        result[positions] = values[positions[generator.permutation(len(positions))]]
        if sorted(result[positions].tolist()) != sorted(values[positions].tolist()):
            raise RuntimeError("label permutation changed a group multiset")
    result.setflags(write=False)
    return result


def order_permutation_indices(n_trials: int, stored_seed: int) -> np.ndarray:
    if int(n_trials) < 1:
        raise TrajectoryWithinSubjectError("n_trials must be positive")
    generator = np.random.default_rng(int(stored_seed))
    result = generator.integers(1, 120, size=int(n_trials), dtype=np.uint8)
    if np.any((result < 1) | (result > 119)):
        raise RuntimeError("order shuffle emitted identity/out-of-range S5 index")
    result.setflags(write=False)
    return result


def apply_order_shuffle(path_features: np.ndarray, permutation_indices: np.ndarray) -> np.ndarray:
    path = np.asarray(path_features, dtype=np.float64)
    indices = np.asarray(permutation_indices)
    if path.ndim != 2 or path.shape[1] != 10 or indices.shape != (len(path),):
        raise TrajectoryWithinSubjectError("PATH/order permutation shape mismatch")
    if not np.isfinite(path).all() or np.any((indices < 1) | (indices > 119)):
        raise TrajectoryWithinSubjectError("PATH/order permutation values invalid")
    edge_maps = EDGE_REINDEX_TABLE[indices]
    result = np.take_along_axis(path, edge_maps, axis=1)
    result.setflags(write=False)
    return result


def assert_bag_invariance(distance_matrix: np.ndarray, permutation_index: int) -> None:
    index = int(permutation_index)
    if not 1 <= index <= 119:
        raise TrajectoryWithinSubjectError("order permutation must be nonidentity")
    reference = bag_canon_d10(distance_matrix).vector
    shuffled = bag_canon_d10(
        permute_distance_matrix(distance_matrix, ALL_PERMUTATIONS_5[index])
    ).vector
    if not np.allclose(reference, shuffled, rtol=0.0, atol=1e-12):
        raise TrajectoryWithinSubjectError("BAG_CANON changed under S5 permutation")


def _fit(x_train: np.ndarray, y_train: np.ndarray, config: Mapping[str, Any]) -> FrozenFit:
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train).astype(str)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise TrajectoryWithinSubjectError("invalid training arrays")
    if set(y) != set(CLASS_ORDER):
        raise TrajectoryWithinSubjectError("training fold lacks the fixed class vocabulary")
    validate_frozen_config(config)
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    scaled = scaler.fit_transform(x)
    classifier = config["classifier"]
    model = LogisticRegression(
        C=float(classifier["c"]),
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-4,
        class_weight=None,
        random_state=CLASSIFIER_RANDOM_STATE,
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
    messages = tuple(str(item.message) for item in caught)
    return FrozenFit(
        scaler=scaler,
        model=model,
        status="FAILED" if convergence else "PASS",
        convergence_warning=bool(convergence),
        warning_messages=messages,
        scaler_mean_sha256=sha256_array(np.asarray(scaler.mean_)),
        scaler_scale_sha256=sha256_array(np.asarray(scaler.scale_)),
    )


def _metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth).astype(str)
    predicted = np.asarray(predicted).astype(str)
    recalls = recall_score(
        truth, predicted, labels=list(CLASS_ORDER), average=None, zero_division=0
    )
    result: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(CLASS_ORDER),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix_json": json.dumps(
            confusion_matrix(truth, predicted, labels=list(CLASS_ORDER)).tolist(),
            separators=(",", ":"),
        ),
        "prediction_sha256": stable_json_sha256(
            {"truth": truth.tolist(), "predicted": predicted.tolist()}
        ),
    }
    for label, value in zip(CLASS_ORDER, recalls, strict=True):
        result[f"recall_{label}"] = float(value)
    return result


def _failed_metrics() -> dict[str, Any]:
    result: dict[str, Any] = {
        "balanced_accuracy": np.nan,
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "confusion_matrix_json": "",
        "prediction_sha256": "",
    }
    for label in CLASS_ORDER:
        result[f"recall_{label}"] = np.nan
    return result


def _score_split(
    features: np.ndarray,
    labels: np.ndarray,
    frame: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    config: Mapping[str, Any],
    *,
    stage: str,
    representation: str,
    subject: int,
    session: str,
    direction: str,
) -> dict[str, Any]:
    if np.any(train & test) or not train.any() or not test.any():
        raise TrajectoryWithinSubjectError("empty or overlapping split")
    train_uids = frame.loc[train, "trial_uid"].tolist()
    test_uids = frame.loc[test, "trial_uid"].tolist()
    if set(train_uids) & set(test_uids):
        raise TrajectoryWithinSubjectError("train/test UID overlap")
    train_subjects = sorted(frame.loc[train, "subject"].unique().astype(int).tolist())
    test_subjects = sorted(frame.loc[test, "subject"].unique().astype(int).tolist())
    if train_subjects != [int(subject)] or test_subjects != [int(subject)]:
        raise TrajectoryWithinSubjectError("other-subject leakage")
    for mask, name in ((train, "train"), (test, "test")):
        counts = pd.Series(labels[mask]).value_counts()
        if set(counts.index) != set(CLASS_ORDER) or counts.nunique() != 1:
            raise TrajectoryWithinSubjectError(f"{name} class counts are not balanced")
    fitted = _fit(features[train], labels[train], config)
    if fitted.status == "PASS":
        predicted = fitted.model.predict(fitted.scaler.transform(features[test]))
        metric_values = _metrics(labels[test], predicted)
    else:
        metric_values = _failed_metrics()
    train_sessions = sorted(frame.loc[train, "session"].unique().tolist())
    test_sessions = sorted(frame.loc[test, "session"].unique().tolist())
    train_runs = sorted(frame.loc[train, "run"].unique().astype(int).tolist())
    test_runs = sorted(frame.loc[test, "run"].unique().astype(int).tolist())
    return {
        "stage": stage,
        "representation": representation,
        "subject": int(subject),
        "session": session,
        "direction": direction,
        "train_session": json.dumps(train_sessions, separators=(",", ":")),
        "test_session": json.dumps(test_sessions, separators=(",", ":")),
        "train_runs": json.dumps(train_runs, separators=(",", ":")),
        "test_runs": json.dumps(test_runs, separators=(",", ":")),
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
        "train_subjects": json.dumps(train_subjects, separators=(",", ":")),
        "test_subjects": json.dumps(test_subjects, separators=(",", ":")),
        "train_uid_sha256": uid_sha256(train_uids),
        "test_uid_sha256": uid_sha256(test_uids),
        "scaler_fit_uid_sha256": uid_sha256(train_uids),
        "scaler_mean_sha256": fitted.scaler_mean_sha256,
        "scaler_scale_sha256": fitted.scaler_scale_sha256,
        **metric_values,
        "status": fitted.status,
        "convergence_warning": fitted.convergence_warning,
        "warning_messages": json.dumps(
            list(fitted.warning_messages), ensure_ascii=False, separators=(",", ":")
        ),
    }


def run_stage_w(
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    representation: str,
    labels: Sequence[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Run both frozen run-half directions within every subject/session."""

    frame = validate_metadata(metadata, config)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(frame) or not np.isfinite(x).all():
        raise TrajectoryWithinSubjectError("feature array is invalid or misaligned")
    if representation not in REPRESENTATIONS:
        raise TrajectoryWithinSubjectError("representation is outside the frozen set")
    y = frame["class_label"].to_numpy() if labels is None else np.asarray(labels).astype(str)
    if y.shape != (len(frame),):
        raise TrajectoryWithinSubjectError("label vector is misaligned")
    rows: list[dict[str, Any]] = []
    directions = (("A_TO_B", HALF_A, HALF_B), ("B_TO_A", HALF_B, HALF_A))
    for subject in range(1, 10):
        for session in SESSION_ORDER:
            scope = frame["subject"].eq(subject) & frame["session"].eq(session)
            for name, train_runs, test_runs in directions:
                train = (scope & frame["run"].isin(train_runs)).to_numpy()
                test = (scope & frame["run"].isin(test_runs)).to_numpy()
                if int(train.sum()) != 144 or int(test.sum()) != 144:
                    raise TrajectoryWithinSubjectError("Stage W half count changed")
                rows.append(
                    _score_split(
                        x, y, frame, train, test, config,
                        stage="W", representation=representation, subject=subject,
                        session=session, direction=name,
                    )
                )
    return pd.DataFrame.from_records(rows).reindex(columns=SCORE_COLUMNS)


def run_stage_x(
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    representation: str,
    labels: Sequence[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Run both same-subject cross-session transfer directions."""

    frame = validate_metadata(metadata, config)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(frame) or not np.isfinite(x).all():
        raise TrajectoryWithinSubjectError("feature array is invalid or misaligned")
    if representation not in REPRESENTATIONS:
        raise TrajectoryWithinSubjectError("representation is outside the frozen set")
    y = frame["class_label"].to_numpy() if labels is None else np.asarray(labels).astype(str)
    if y.shape != (len(frame),):
        raise TrajectoryWithinSubjectError("label vector is misaligned")
    rows: list[dict[str, Any]] = []
    directions = (
        ("0train_TO_1test", "0train", "1test"),
        ("1test_TO_0train", "1test", "0train"),
    )
    for subject in range(1, 10):
        subject_mask = frame["subject"].eq(subject)
        for name, train_session, test_session in directions:
            train = (subject_mask & frame["session"].eq(train_session)).to_numpy()
            test = (subject_mask & frame["session"].eq(test_session)).to_numpy()
            if int(train.sum()) != 288 or int(test.sum()) != 288:
                raise TrajectoryWithinSubjectError("Stage X session count changed")
            rows.append(
                _score_split(
                    x, y, frame, train, test, config,
                    stage="X", representation=representation, subject=subject,
                    session="cross_session", direction=name,
                )
            )
    return pd.DataFrame.from_records(rows).reindex(columns=SCORE_COLUMNS)


def stage_subject_statistics(scores: pd.DataFrame, stage: str) -> tuple[np.ndarray, float]:
    expected_rows = 36 if stage == "W" else 18 if stage == "X" else None
    if expected_rows is None:
        raise TrajectoryWithinSubjectError("stage must be W or X")
    if len(scores) != expected_rows or set(scores["subject"].astype(int)) != set(range(1, 10)):
        raise IncompleteRequiredGrid(f"Stage {stage} required grid is incomplete")
    if not (scores["status"].astype(str) == "PASS").all():
        raise IncompleteRequiredGrid(f"Stage {stage} contains a FAILED required fit")
    if scores["balanced_accuracy"].isna().any():
        raise IncompleteRequiredGrid(f"Stage {stage} contains a missing required metric")
    expected_per_subject = 4 if stage == "W" else 2
    counts = scores.groupby("subject", observed=True).size()
    if len(counts) != 9 or not (counts == expected_per_subject).all():
        raise IncompleteRequiredGrid(f"Stage {stage} subject grid is incomplete")
    subject = (
        scores.groupby("subject", sort=True, observed=True)["balanced_accuracy"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    if subject.shape != (9,) or not np.isfinite(subject).all():
        raise IncompleteRequiredGrid(f"Stage {stage} subject statistic is incomplete")
    return subject, float(np.median(subject))


def monte_carlo_result(observed: float, null_statistics: np.ndarray) -> MonteCarloResult:
    values = np.asarray(null_statistics, dtype=np.float64)
    if values.shape != (NULL_REPLICATES,) or not np.isfinite(values).all():
        raise IncompleteRequiredGrid("null distribution is not the exact 1,999-vector")
    null_median = float(np.median(values))
    effect = float(observed) - null_median
    exceedance = int(np.count_nonzero(values >= float(observed)))
    p_value = float((1 + exceedance) / (NULL_REPLICATES + 1))
    return MonteCarloResult(
        observed=float(observed),
        null_median=null_median,
        effect=effect,
        p_value=p_value,
        exceedance_count=exceedance,
        replicates=NULL_REPLICATES,
        passed=bool(effect > 0.0 and p_value <= 0.05),
    )


def compare_reproduction(
    observed_metadata: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    observed_arrays: Mapping[str, np.ndarray],
    reference_arrays: Mapping[str, np.ndarray],
    *,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Return fail-closed v0 reproduction rows for identities and three arrays."""

    identity = ["sample_index", "subject", "session", "run", "trial_id", "trial_uid", "class_label"]
    rows: list[dict[str, Any]] = []
    identity_pass = bool(
        list(observed_metadata.columns.intersection(identity)) == identity
        and list(reference_metadata.columns.intersection(identity)) == identity
        and observed_metadata.loc[:, identity].reset_index(drop=True).equals(
            reference_metadata.loc[:, identity].reset_index(drop=True)
        )
    )
    rows.append(
        {
            "check": "session0_trial_metadata_exact",
            "maximum_absolute_difference": 0.0 if identity_pass else np.inf,
            "tolerance": 0.0,
            "passed": identity_pass,
        }
    )
    for name in ("airm_path_d10", "airm_bag_canon_d10", "airm_scalars_11"):
        observed = np.asarray(observed_arrays[name], dtype=np.float64)
        reference = np.asarray(reference_arrays[name], dtype=np.float64)
        shape_pass = observed.shape == reference.shape
        maximum = (
            float(np.max(np.abs(observed - reference)))
            if shape_pass and observed.size and np.isfinite(observed).all() and np.isfinite(reference).all()
            else np.inf
        )
        rows.append(
            {
                "check": f"{name}_machine_precision",
                "maximum_absolute_difference": maximum,
                "tolerance": float(atol),
                "passed": bool(shape_pass and maximum <= float(atol)),
            }
        )
    return pd.DataFrame.from_records(rows)


def terminal_decision(
    *,
    reproduction_gate_pass: bool,
    technical_grid_pass: bool,
    stage_w_pass: bool | None,
    stage_x_pass: bool | None,
    stage_o_pass: bool | None,
) -> TerminalDecision:
    if not reproduction_gate_pass:
        return TerminalDecision(
            "UNASSESSED_TRAJECTORY_REPRODUCTION_FAILURE", None, None, None
        )
    if not technical_grid_pass:
        return TerminalDecision("UNASSESSED_TECHNICAL_FAILURE", None, None, None)
    if stage_w_pass is not True:
        return TerminalDecision(
            "STOP_WITHIN_SUBJECT_TRAJECTORY_CLASS_POOR", False, None, None
        )
    if stage_x_pass is not True:
        return TerminalDecision(
            "STOP_SESSION_SPECIFIC_TRAJECTORY_ONLY", True, False, None
        )
    if stage_o_pass is not True:
        return TerminalDecision(
            "GO_STABLE_SUBJECT_SPECIFIC_LOCAL_GEOMETRY", True, True, False
        )
    return TerminalDecision(
        "GO_STABLE_SUBJECT_SPECIFIC_ORDERED_TRAJECTORY_COMPONENT", True, True, True
    )


__all__ = [
    "MASTER_SEED",
    "CLASSIFIER_RANDOM_STATE",
    "LABEL_STREAM_TAG",
    "ORDER_STREAM_TAG",
    "NULL_REPLICATES",
    "CLASS_ORDER",
    "SESSION_ORDER",
    "RUN_ORDER",
    "HALF_A",
    "HALF_B",
    "REPRESENTATIONS",
    "IDENTITY_COLUMNS",
    "SCORE_COLUMNS",
    "TrajectoryWithinSubjectError",
    "IncompleteRequiredGrid",
    "FrozenFit",
    "MonteCarloResult",
    "TerminalDecision",
    "sha256_file",
    "sha256_array",
    "stable_json_sha256",
    "uid_sha256",
    "load_frozen_config",
    "validate_frozen_config",
    "validate_metadata",
    "make_seed_vector",
    "permute_labels",
    "order_permutation_indices",
    "apply_order_shuffle",
    "assert_bag_invariance",
    "run_stage_w",
    "run_stage_x",
    "stage_subject_statistics",
    "monte_carlo_result",
    "compare_reproduction",
    "terminal_decision",
]
