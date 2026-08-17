"""Frozen trial-level ordered-SPD movement incremental-utility audit.

Feature constructors and centering functions deliberately accept covariates and
identity positions only.  Labels cross the boundary only in source decoder
fitting and final target metric evaluation.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.alignment_v2 import (
    assert_t1_overlap,
    assert_t2_disjoint,
    make_calibration_splits,
    make_loso_partition,
    trial_uid_sha256,
)
from src.data import collect_environment_metadata, sha256_file
from src.evaluation_v2 import array_sha256, prediction_metrics, stable_json_hash
from src.geometry_v2 import AIRM, FittedCenter, fit_center
from src.local_mean_movement_v0 import (
    CONGRUENCE_RELATIVE_TOLERANCE,
    anti_develop_sequence,
    conjugate_movement,
)
from src.spd_utils import log_svec, svec


BASE_COMMIT = "969f80fc29b31993df69980c23f52240ce591ff1"
BRANCH = "pilot/trial-movement-incremental-utility-v0"
CLASS_ORDER = ("left_hand", "right_hand", "feet", "tongue")
CONDITIONS = (
    "STATIC",
    "MOV_LEN",
    "MOV_GRAM",
    "MOV_SENSOR",
    "STATIC_PLUS_LEN",
    "STATIC_PLUS_GRAM",
    "STATIC_PLUS_SENSOR",
)
DIMENSIONS = {
    "STATIC": 253,
    "MOV_LEN": 4,
    "MOV_GRAM": 10,
    "MOV_SENSOR": 1012,
    "STATIC_PLUS_LEN": 257,
    "STATIC_PLUS_GRAM": 263,
    "STATIC_PLUS_SENSOR": 1265,
}
PRIMARY_COMPARISONS = {
    "DELTA_LEN": ("STATIC_PLUS_LEN", "STATIC"),
    "DELTA_GRAM": ("STATIC_PLUS_GRAM", "STATIC"),
    "DELTA_SENSOR": ("STATIC_PLUS_SENSOR", "STATIC"),
}
SECONDARY_COMPARISONS = {
    "GRAM_OVER_LEN": ("STATIC_PLUS_GRAM", "STATIC_PLUS_LEN"),
    "SENSOR_OVER_GRAM": ("STATIC_PLUS_SENSOR", "STATIC_PLUS_GRAM"),
}


class TrialMovementAuditError(RuntimeError):
    """An auditable frozen-contract failure."""


@dataclass(frozen=True)
class FrozenInputs:
    whole: np.ndarray
    window5: np.ndarray
    channel_names: np.ndarray
    whole_metadata: pd.DataFrame
    window_metadata: pd.DataFrame
    data_contract: pd.DataFrame


@dataclass(frozen=True)
class MovementFeatures:
    mov_len: np.ndarray
    mov_gram: np.ndarray
    mov_sensor: np.ndarray
    metadata: pd.DataFrame
    geometry_gates: pd.DataFrame
    feature_contract: pd.DataFrame


@dataclass(frozen=True)
class DecoderAudit:
    target_subject: int
    condition: str
    n_source: int
    n_features: int
    scaler_fit_n: int
    model_fit_n: int
    convergence_warning: bool
    warning_messages: tuple[str, ...]
    n_iter_max: int
    status: str
    source_uid_sha256: str


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    if tuple(config["features"]["condition_order"]) != CONDITIONS:
        raise ValueError("feature condition order left the frozen contract")
    if tuple(config["dataset"]["classes"]) != CLASS_ORDER:
        raise ValueError("class order left the frozen contract")
    if {key: int(config["features"][key]) for key in CONDITIONS} != DIMENSIONS:
        raise ValueError("feature dimensions left the frozen contract")
    return config


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _relative_error(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    return float(
        np.linalg.norm(a - b)
        / max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    )


def _normalize_metadata(frame: pd.DataFrame, *, windowed: bool) -> pd.DataFrame:
    required = {
        "covariance_index",
        "sample_index",
        "subject",
        "session",
        "run",
        "trial_id",
        "run_trial_id",
        "trial_uid",
        "class_label",
    }
    if windowed:
        required.add("window_index")
    missing = required - set(frame.columns)
    if missing:
        raise TrialMovementAuditError(f"metadata missing columns: {sorted(missing)}")
    result = frame.copy().reset_index(drop=True)
    for column in (
        "covariance_index",
        "sample_index",
        "subject",
        "run",
        "trial_id",
        "run_trial_id",
    ):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(np.int64)
    if windowed:
        result["window_index"] = pd.to_numeric(
            result["window_index"], errors="raise"
        ).astype(np.int64)
    for column in ("session", "trial_uid", "class_label"):
        result[column] = result[column].astype(str)
    return result


def load_frozen_inputs(project_root: str | Path, config: Mapping[str, Any]) -> FrozenInputs:
    """Load and hard-gate exact frozen V1 artifacts without modifying them."""

    root = Path(project_root).resolve()
    paths = {
        key: root / str(config["inputs"][key]["path"])
        for key in ("v1_config", "covariances", "whole_metadata", "window5_metadata")
    }
    for key, path in paths.items():
        if not path.is_file():
            raise TrialMovementAuditError(
                f"UNASSESSED_MISSING_FROZEN_INPUT: {key} absent at {path}"
            )
        actual = sha256_file(path)
        expected = str(config["inputs"][key]["sha256"])
        if actual != expected:
            raise TrialMovementAuditError(
                f"UNASSESSED_FROZEN_INPUT_HASH_MISMATCH: {key} {actual} != {expected}"
            )

    with np.load(paths["covariances"], allow_pickle=False) as archive:
        required_keys = {"whole", "window5", "channel_names"}
        if set(archive.files) != required_keys:
            raise TrialMovementAuditError("covariance NPZ keys changed")
        whole = np.asarray(archive["whole"], dtype=np.float64)
        window5 = np.asarray(archive["window5"], dtype=np.float64)
        channels = np.asarray(archive["channel_names"]).astype(str)
    whole_meta = _normalize_metadata(
        pd.read_csv(paths["whole_metadata"], dtype={"session": str, "run": str}),
        windowed=False,
    )
    window_meta = _normalize_metadata(
        pd.read_csv(paths["window5_metadata"], dtype={"session": str, "run": str}),
        windowed=True,
    )

    expected = config["dataset"]
    checks: list[dict[str, Any]] = []

    def gate(name: str, observed: Any, wanted: Any, passed: bool) -> None:
        checks.append(
            {
                "gate": name,
                "observed": str(observed),
                "expected": str(wanted),
                "passed": bool(passed),
            }
        )
        if not passed:
            raise TrialMovementAuditError(
                f"UNASSESSED_DATA_CONTRACT_FAILURE: {name}: {observed} != {wanted}"
            )

    gate("whole_shape", tuple(whole.shape), (2592, 22, 22), whole.shape == (2592, 22, 22))
    gate(
        "window5_shape",
        tuple(window5.shape),
        (12960, 22, 22),
        window5.shape == (12960, 22, 22),
    )
    gate("channel_count", len(channels), 22, len(channels) == 22)
    gate("whole_rows", len(whole_meta), 2592, len(whole_meta) == 2592)
    gate("window_rows", len(window_meta), 12960, len(window_meta) == 12960)
    gate(
        "whole_covariance_index",
        "sequential",
        "sequential",
        np.array_equal(whole_meta["covariance_index"], np.arange(2592)),
    )
    gate(
        "window_covariance_index",
        "sequential",
        "sequential",
        np.array_equal(window_meta["covariance_index"], np.arange(12960)),
    )
    subjects = tuple(sorted(whole_meta["subject"].unique().tolist()))
    gate("subjects", subjects, tuple(expected["subjects"]), subjects == tuple(expected["subjects"]))
    sessions = tuple(sorted(whole_meta["session"].unique().tolist()))
    gate("session", sessions, ("0train",), sessions == ("0train",))
    classes = tuple(value for value in CLASS_ORDER if value in set(whole_meta["class_label"]))
    gate("classes", classes, CLASS_ORDER, classes == CLASS_ORDER)
    gate(
        "whole_finite",
        bool(np.isfinite(whole).all()),
        True,
        bool(np.isfinite(whole).all()),
    )
    gate(
        "window_finite",
        bool(np.isfinite(window5).all()),
        True,
        bool(np.isfinite(window5).all()),
    )
    gate(
        "whole_symmetric",
        float(np.max(np.abs(whole - whole.transpose(0, 2, 1)))),
        "<=1e-12 relative",
        bool(
            np.max(
                np.linalg.norm(whole - whole.transpose(0, 2, 1), axis=(1, 2))
                / np.maximum(np.linalg.norm(whole, axis=(1, 2)), np.finfo(float).tiny)
            )
            <= 1e-12
        ),
    )
    gate(
        "window_symmetric",
        float(np.max(np.abs(window5 - window5.transpose(0, 2, 1)))),
        "<=1e-12 relative",
        bool(
            np.max(
                np.linalg.norm(window5 - window5.transpose(0, 2, 1), axis=(1, 2))
                / np.maximum(np.linalg.norm(window5, axis=(1, 2)), np.finfo(float).tiny)
            )
            <= 1e-12
        ),
    )
    gate("whole_spd", float(np.min(np.linalg.eigvalsh(whole))), ">0", bool(np.min(np.linalg.eigvalsh(whole)) > 0))
    gate("window_spd", float(np.min(np.linalg.eigvalsh(window5))), ">0", bool(np.min(np.linalg.eigvalsh(window5)) > 0))

    subject_counts = whole_meta.groupby("subject", observed=True).size()
    gate(
        "trials_per_subject",
        subject_counts.to_dict(),
        "288 each",
        bool((subject_counts == 288).all()),
    )
    cell_counts = whole_meta.groupby(["subject", "class_label"], observed=True).size()
    gate(
        "trials_per_subject_class",
        sorted(cell_counts.unique().tolist()),
        [72],
        bool((cell_counts == 72).all()),
    )
    grouped = window_meta.groupby("trial_uid", sort=False, observed=True)
    window_counts = grouped.size()
    gate("five_windows_per_trial", sorted(window_counts.unique()), [5], bool((window_counts == 5).all()))
    orders_ok = all(
        tuple(frame["window_index"].tolist()) == (1, 2, 3, 4, 5)
        for _, frame in grouped
    )
    gate("window_order", orders_ok, True, orders_ok)
    identity = ["subject", "session", "run", "trial_id", "run_trial_id", "trial_uid", "class_label"]
    constant_ok = all(
        all(frame[column].nunique(dropna=False) == 1 for column in identity)
        for _, frame in grouped
    )
    gate("window_identity_constant", constant_ok, True, constant_ok)
    first = window_meta.groupby("trial_uid", sort=False, observed=True).first().reset_index(drop=False)
    first = first.set_index("trial_uid").loc[whole_meta["trial_uid"]].reset_index()
    aligned = all(
        np.array_equal(first[column].astype(str), whole_meta[column].astype(str))
        for column in identity
    )
    gate("whole_window_identity_alignment", aligned, True, aligned)
    return FrozenInputs(whole, window5, channels, whole_meta, window_meta, pd.DataFrame(checks))


def movement_feature_tuple(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return label-free length, Gram matrix/svec, and sensor tuple."""

    movement = np.asarray(z, dtype=np.float64)
    if movement.shape != (4, 22, 22):
        raise ValueError("movement must have shape (4,22,22)")
    if not np.isfinite(movement).all() or not np.allclose(
        movement, movement.transpose(0, 2, 1), rtol=1e-10, atol=1e-12
    ):
        raise ValueError("movement must be finite and symmetric")
    lengths = np.linalg.norm(movement, axis=(1, 2))
    flat = movement.reshape(4, -1)
    gram = 0.5 * (flat @ flat.T + (flat @ flat.T).T)
    gram_feature = svec(gram)
    sensor = svec(movement).reshape(-1)
    return lengths, gram, gram_feature, sensor


def _fixed_orthogonal(seed: int, dimension: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    q *= np.sign(np.diag(r))[None, :]
    return q


def build_trial_movement_features(
    frozen: FrozenInputs, config: Mapping[str, Any]
) -> MovementFeatures:
    """Compute each trial's movement independently; no label argument exists."""

    n_trials = len(frozen.whole_metadata)
    mov_len = np.empty((n_trials, 4), dtype=np.float64)
    mov_gram = np.empty((n_trials, 10), dtype=np.float64)
    mov_sensor = np.empty((n_trials, 1012), dtype=np.float64)
    maximum_norm_error = 0.0
    maximum_symmetry_error = 0.0
    maximum_transport_error = 0.0
    minimum_gram_eigenvalue = np.inf
    maximum_diagonal_error = 0.0
    maximum_o_error = 0.0
    maximum_gl_error = 0.0
    delta_t = float(config["geometry"]["delta_t_seconds"])
    q = _fixed_orthogonal(int(config["protocol"]["seed"]) + 101, 22)
    q2 = _fixed_orthogonal(int(config["protocol"]["seed"]) + 102, 22)
    congruence = q2 @ np.diag(np.linspace(0.75, 1.25, 22)) @ q2.T
    window_index = frozen.window_metadata.set_index(["trial_uid", "window_index"])[
        "covariance_index"
    ]
    for row, uid in enumerate(frozen.whole_metadata["trial_uid"].astype(str)):
        indices = np.asarray([window_index.loc[(uid, step)] for step in range(1, 6)], dtype=np.int64)
        sequence = frozen.window5[indices]
        anti = anti_develop_sequence(sequence, delta_t=delta_t)
        lengths, gram, gram_vec, sensor = movement_feature_tuple(anti.z)
        mov_len[row] = lengths
        mov_gram[row] = gram_vec
        mov_sensor[row] = sensor
        maximum_norm_error = max(
            maximum_norm_error,
            float(anti.diagnostics["maximum_norm_absolute_error"].max()),
        )
        maximum_symmetry_error = max(
            maximum_symmetry_error,
            float(anti.diagnostics["z_symmetry_relative_error"].max()),
        )
        maximum_transport_error = max(
            maximum_transport_error,
            float(anti.diagnostics["maximum_edge_transport_relative_error"].max()),
        )
        minimum_gram_eigenvalue = min(minimum_gram_eigenvalue, float(np.min(np.linalg.eigvalsh(gram))))
        maximum_diagonal_error = max(maximum_diagonal_error, float(np.max(np.abs(np.diag(gram) - lengths**2))))
        _, o_gram, _, _ = movement_feature_tuple(conjugate_movement(anti.z, q))
        maximum_o_error = max(maximum_o_error, _relative_error(gram, o_gram))
        transformed_sequence = np.einsum(
            "ij,kjl,ml->kim", congruence, sequence, congruence, optimize=True
        )
        transformed = anti_develop_sequence(transformed_sequence, delta_t=delta_t)
        _, gl_gram, _, _ = movement_feature_tuple(transformed.z)
        maximum_gl_error = max(maximum_gl_error, _relative_error(gram, gl_gram))

    tolerances = config["geometry"]
    gate_specs = [
        ("trial_sequences", n_trials, 2592, n_trials == 2592),
        ("transitions", n_trials * 4, 10368, n_trials * 4 == 10368),
        ("movement_finite", bool(np.isfinite(mov_sensor).all()), True, bool(np.isfinite(mov_sensor).all())),
        ("movement_norm_preservation", maximum_norm_error, float(tolerances["movement_tolerance"]), maximum_norm_error <= float(tolerances["movement_tolerance"])),
        ("movement_symmetry", maximum_symmetry_error, float(tolerances["symmetry_relative_tolerance"]), maximum_symmetry_error <= float(tolerances["symmetry_relative_tolerance"])),
        ("transport_preservation", maximum_transport_error, float(tolerances["movement_tolerance"]), maximum_transport_error <= float(tolerances["movement_tolerance"])),
        ("gram_psd", minimum_gram_eigenvalue, -float(tolerances["gram_psd_eigenvalue_tolerance"]), minimum_gram_eigenvalue >= -float(tolerances["gram_psd_eigenvalue_tolerance"])),
        ("gram_diagonal", maximum_diagonal_error, float(tolerances["gram_diagonal_absolute_tolerance"]), maximum_diagonal_error <= float(tolerances["gram_diagonal_absolute_tolerance"])),
        ("gram_common_o_invariance", maximum_o_error, float(tolerances["invariance_relative_tolerance"]), maximum_o_error <= float(tolerances["invariance_relative_tolerance"])),
        ("gram_gl_congruence_invariance", maximum_gl_error, float(tolerances["invariance_relative_tolerance"]), maximum_gl_error <= float(tolerances["invariance_relative_tolerance"])),
    ]
    gates = pd.DataFrame(
        [
            {"scope": "all_trials", "gate": name, "observed": observed, "threshold_or_expected": wanted, "passed": passed}
            for name, observed, wanted, passed in gate_specs
        ]
    )
    if not bool(gates["passed"].all()):
        raise TrialMovementAuditError("UNASSESSED_MOVEMENT_GEOMETRY_FAILURE")
    feature_rows = []
    for name, values in (
        ("MOV_LEN", mov_len),
        ("MOV_GRAM", mov_gram),
        ("MOV_SENSOR", mov_sensor),
    ):
        feature_rows.append(
            {
                "feature": name,
                "rows": len(values),
                "dimensions": values.shape[1],
                "expected_dimensions": DIMENSIONS[name],
                "finite": bool(np.isfinite(values).all()),
                "label_input": False,
                "passed": bool(values.shape == (2592, DIMENSIONS[name]) and np.isfinite(values).all()),
            }
        )
    contract = pd.DataFrame(feature_rows)
    if not bool(contract["passed"].all()):
        raise TrialMovementAuditError("UNASSESSED_FEATURE_CONTRACT_FAILURE")
    metadata = frozen.whole_metadata[
        ["trial_uid", "subject", "class_label", "run", "session", "trial_id"]
    ].copy()
    return MovementFeatures(mov_len, mov_gram, mov_sensor, metadata, gates, contract)


def condition_features(
    condition: str,
    static: np.ndarray,
    mov_len: np.ndarray,
    mov_gram: np.ndarray,
    mov_sensor: np.ndarray,
) -> np.ndarray:
    """Construct one frozen condition from already-computed covariate features."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    components = {
        "STATIC": np.asarray(static, dtype=np.float64),
        "MOV_LEN": np.asarray(mov_len, dtype=np.float64),
        "MOV_GRAM": np.asarray(mov_gram, dtype=np.float64),
        "MOV_SENSOR": np.asarray(mov_sensor, dtype=np.float64),
    }
    mapping = {
        "STATIC": ("STATIC",),
        "MOV_LEN": ("MOV_LEN",),
        "MOV_GRAM": ("MOV_GRAM",),
        "MOV_SENSOR": ("MOV_SENSOR",),
        "STATIC_PLUS_LEN": ("STATIC", "MOV_LEN"),
        "STATIC_PLUS_GRAM": ("STATIC", "MOV_GRAM"),
        "STATIC_PLUS_SENSOR": ("STATIC", "MOV_SENSOR"),
    }
    arrays = [components[name] for name in mapping[condition]]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("condition components have different row counts")
    result = arrays[0].copy() if len(arrays) == 1 else np.concatenate(arrays, axis=1)
    if result.ndim != 2 or result.shape[1] != DIMENSIONS[condition] or not np.isfinite(result).all():
        raise TrialMovementAuditError(f"feature contract failed for {condition}")
    return result


def fit_fixed_source_decoder(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    *,
    target_subject: int,
    condition: str,
    source_trial_uids: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[StandardScaler, LogisticRegression, DecoderAudit]:
    """Fit scaler and logistic only on supplied source rows."""

    features = np.asarray(source_features, dtype=np.float64)
    labels = np.asarray(source_labels).astype(str)
    if features.ndim != 2 or len(features) != len(labels) or not np.isfinite(features).all():
        raise ValueError("invalid source training arrays")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    frozen = config["decoder"]
    model = LogisticRegression(
        penalty=str(frozen["penalty"]),
        C=float(frozen["C"]),
        solver=str(frozen["solver"]),
        max_iter=int(frozen["max_iter"]),
        tol=float(frozen["tol"]),
        random_state=int(frozen["random_state"]),
        class_weight=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(scaled, labels)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    audit = DecoderAudit(
        target_subject=int(target_subject),
        condition=str(condition),
        n_source=len(features),
        n_features=features.shape[1],
        scaler_fit_n=int(scaler.n_samples_seen_),
        model_fit_n=len(labels),
        convergence_warning=bool(convergence),
        warning_messages=tuple(f"{item.category.__name__}: {item.message}" for item in caught),
        n_iter_max=int(np.max(model.n_iter_)),
        status="INVALID_CONVERGENCE" if convergence else "PASS",
        source_uid_sha256=trial_uid_sha256(source_trial_uids),
    )
    return scaler, model, audit


def exact_signflip_test(deltas: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("exact sign-flip requires nine finite subject deltas")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=9)))
    statistics = (signs * values).mean(axis=1)
    observed = float(values.mean())
    p_value = float(np.count_nonzero(statistics >= observed - 1e-15) / len(statistics))
    return {
        "n_subjects": 9,
        "n_patterns": 512,
        "observed_mean": observed,
        "p_raw_one_sided": p_value,
        "null_min": float(statistics.min()),
        "null_max": float(statistics.max()),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("Holm family cannot be empty")
    items = sorted(((str(key), float(value)) for key, value in p_values.items()), key=lambda item: (item[1], item[0]))
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(items):
        if not 0.0 <= value <= 1.0:
            raise ValueError("p-values must lie in [0,1]")
        running = max(running, (m - rank) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def _compositions(total: int, length: int, prefix: tuple[int, ...] = ()) -> Iterable[tuple[int, ...]]:
    if length == 1:
        yield prefix + (total,)
        return
    for value in range(total + 1):
        yield from _compositions(total - value, length - 1, prefix + (value,))


def exact_subject_bootstrap_interval(deltas: Sequence[float]) -> tuple[float, float]:
    """Exact percentile interval over all 9^9 ordered bootstrap samples."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("exact subject bootstrap requires nine finite values")
    factorial = math.factorial
    records: list[tuple[float, int]] = []
    for counts_tuple in _compositions(9, 9):
        counts = np.asarray(counts_tuple, dtype=np.int64)
        weight = factorial(9)
        for count in counts:
            weight //= factorial(int(count))
        records.append((float(np.dot(counts, values) / 9.0), int(weight)))
    records.sort(key=lambda item: item[0])
    total_weight = 9**9
    if sum(weight for _, weight in records) != total_weight or len(records) != 24310:
        raise RuntimeError("exact bootstrap enumeration contract failed")

    def quantile(probability: float) -> float:
        rank = int(math.ceil(probability * total_weight))
        cumulative = 0
        for value, weight in records:
            cumulative += weight
            if cumulative >= rank:
                return value
        raise RuntimeError("bootstrap quantile was not reached")

    return quantile(0.025), quantile(0.975)


def delta_summary(deltas: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("delta summary requires nine finite values")
    loo = np.asarray([np.delete(values, index).mean() for index in range(9)])
    low, high = exact_subject_bootstrap_interval(values)
    denominator = float(np.sum(np.abs(values)))
    return {
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "positive_subjects": int(np.count_nonzero(values > 0.0)),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
        "loo_mean_min": float(loo.min()),
        "loo_mean_max": float(loo.max()),
        "every_loo_mean_positive": bool(np.all(loo > 0.0)),
        "top_contributor_share": 0.0 if denominator == 0.0 else float(np.max(np.abs(values)) / denominator),
        "loo_means": loo,
    }


def terminal_decision(primary: pd.DataFrame, holm: Mapping[str, float]) -> tuple[str, dict[str, dict[str, bool]]]:
    states: dict[str, dict[str, bool]] = {}
    for comparison in PRIMARY_COMPARISONS:
        row = primary.loc[primary["comparison"] == comparison].iloc[0]
        supported = bool(
            float(row["mean_delta"]) > 0.0
            and float(row["median_delta"]) > 0.0
            and float(holm[comparison]) <= 0.05
        )
        broad = bool(
            supported
            and int(row["positive_subjects"]) >= 6
            and bool(row["every_loo_mean_positive"])
        )
        states[comparison] = {"supported": supported, "broad": broad}
    if states["DELTA_GRAM"]["supported"]:
        decision = (
            "GO_INVARIANT_MOVEMENT_INCREMENTAL_UTILITY"
            if states["DELTA_GRAM"]["broad"]
            else "GO_HETEROGENEOUS_INVARIANT_MOVEMENT_UTILITY"
        )
    elif states["DELTA_SENSOR"]["supported"]:
        decision = "GO_SENSOR_ONLY_MOVEMENT_INCREMENTAL_UTILITY"
    elif states["DELTA_LEN"]["supported"]:
        decision = "GO_SPEED_ONLY_INCREMENTAL_UTILITY"
    else:
        decision = "STOP_NO_TRIAL_MOVEMENT_INCREMENTAL_UTILITY"
    return decision, states


def feature_functions_are_label_free() -> bool:
    forbidden = {"label", "labels", "y", "class_label", "target_label"}
    functions = (movement_feature_tuple, build_trial_movement_features, condition_features)
    return all(not (set(inspect.signature(function).parameters) & forbidden) for function in functions)


def _manifest(root: Path, output_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for directory in (root / "configs", root / "docs", root / "outputs"):
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if output_root == path or output_root in path.parents:
                continue
            result[str(path.relative_to(root))] = sha256_file(path)
    return result


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _score_rows(
    *,
    protocol: str,
    split: str,
    target: int,
    condition: str,
    uids: np.ndarray,
    metadata: pd.DataFrame,
    scaler: StandardScaler,
    model: LogisticRegression,
    features: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    transformed = scaler.transform(features)
    predicted = model.predict(transformed).astype(str)
    probabilities = model.predict_proba(transformed)
    class_columns = {str(value): index for index, value in enumerate(model.classes_)}
    truth = metadata["class_label"].astype(str).to_numpy()
    metrics = prediction_metrics(truth, predicted, class_order=CLASS_ORDER)
    rows = metadata[["trial_uid", "subject", "session", "run", "trial_id", "class_label"]].copy()
    if not np.array_equal(rows["trial_uid"].astype(str).to_numpy(), uids.astype(str)):
        raise TrialMovementAuditError("evaluation UID/features alignment failure")
    rows.insert(0, "condition", condition)
    rows.insert(0, "target_subject", int(target))
    rows.insert(0, "split", split)
    rows.insert(0, "protocol", protocol)
    rows = rows.rename(columns={"class_label": "true_class_label"})
    rows["predicted_class"] = predicted
    for label in CLASS_ORDER:
        rows[f"probability_{label}"] = probabilities[:, class_columns[label]]
    rows["source_scaler_fit_n"] = int(scaler.n_samples_seen_)
    rows["source_model_fit_n"] = int(len(model._sklearn_version) if False else 2304)
    return rows, metrics, predicted


def _metric_record(protocol: str, split: str, target: int, condition: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "protocol": protocol,
        "split": split,
        "target_subject": int(target),
        "condition": condition,
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "evaluation_n": int(metrics["n_evaluation"]),
        "confusion_matrix_json": str(metrics["confusion_matrix_json"]),
    }
    for label in CLASS_ORDER:
        result[f"recall_{label}"] = float(metrics[f"recall_{label}"])
    return result


def _precompute_static(
    frozen: FrozenInputs, config: Mapping[str, Any]
) -> tuple[np.ndarray, dict[tuple[int, str], tuple[np.ndarray, np.ndarray]], pd.DataFrame, pd.DataFrame]:
    metadata = frozen.whole_metadata
    identity = metadata.drop(columns=["class_label"])
    full_static = np.empty((len(metadata), 253), dtype=np.float64)
    target_states: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    gates: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    tol = float(config["geometry"]["airm_mean_tol"])
    maxiter = int(config["geometry"]["airm_mean_maxiter"])
    for subject in config["dataset"]["subjects"]:
        partition = make_loso_partition(identity, int(subject))
        positions = np.asarray(partition.target_row_positions, dtype=np.int64)
        center = fit_center(frozen.whole[positions], AIRM, tol=tol, maxiter=maxiter)
        full_static[positions] = log_svec(center.transform(frozen.whole[positions]))
        gates.append(_center_gate_row(center, int(subject), "FULL", len(positions)))
        assert_t1_overlap(partition.target_trial_uids, partition.target_trial_uids)
        splits = make_calibration_splits(identity, int(subject))
        for split in splits:
            calibration = np.asarray(split.calibration_row_positions, dtype=np.int64)
            evaluation = np.asarray(split.evaluation_row_positions, dtype=np.int64)
            assert_t2_disjoint(split.calibration_trial_uids, split.evaluation_trial_uids)
            target_center = fit_center(frozen.whole[calibration], AIRM, tol=tol, maxiter=maxiter)
            target_states[(int(subject), split.name)] = (
                evaluation,
                log_svec(target_center.transform(frozen.whole[evaluation])),
            )
            gates.append(_center_gate_row(target_center, int(subject), split.name, len(calibration)))
            source = np.asarray(partition.source_row_positions, dtype=np.int64)
            source_subjects = set(metadata.iloc[source]["subject"].astype(int))
            target_subjects = set(metadata.iloc[evaluation]["subject"].astype(int))
            window_uids = set(frozen.window_metadata.iloc[
                frozen.window_metadata["trial_uid"].isin(split.evaluation_trial_uids).to_numpy()
            ]["trial_uid"].astype(str))
            split_rows.append(
                {
                    "target_subject": int(subject),
                    "split": split.name,
                    "source_subjects": json.dumps(sorted(source_subjects)),
                    "calibration_runs": json.dumps(list(split.calibration_runs)),
                    "evaluation_runs": json.dumps(list(split.evaluation_runs)),
                    "source_n": len(source),
                    "calibration_n": len(calibration),
                    "evaluation_n": len(evaluation),
                    "source_target_disjoint": bool(int(subject) not in source_subjects and target_subjects == {int(subject)}),
                    "calibration_evaluation_uid_disjoint": not bool(set(split.calibration_trial_uids) & set(split.evaluation_trial_uids)),
                    "evaluation_window_trial_uids_exact": window_uids == set(split.evaluation_trial_uids),
                    "target_labels_used_for_center": False,
                    "passed": True,
                }
            )
    gate_frame = pd.DataFrame(gates)
    if not bool(gate_frame["passed"].all()):
        raise TrialMovementAuditError("UNASSESSED_STATIC_GEOMETRY_FAILURE")
    return full_static, target_states, gate_frame, pd.DataFrame(split_rows)


def _center_gate_row(center: FittedCenter, subject: int, split: str, fit_n: int) -> dict[str, Any]:
    residual = float(center.normalized_karcher_post_residual or 0.0)
    warnings_present = bool(center.solver_warning_messages)
    passed = bool(
        center.geometry == AIRM
        and center.fit_sample_count == fit_n
        and residual <= 1e-7
        and not warnings_present
    )
    return {
        "scope": "static_center",
        "gate": "airm_center",
        "target_subject": subject,
        "split": split,
        "fit_n": fit_n,
        "normalized_karcher_residual": residual,
        "warning_messages": json.dumps(list(center.solver_warning_messages)),
        "passed": passed,
    }


def _condition_input(
    condition: str,
    positions: np.ndarray,
    static: np.ndarray,
    movement: MovementFeatures,
) -> np.ndarray:
    return condition_features(
        condition,
        static,
        movement.mov_len[positions],
        movement.mov_gram[positions],
        movement.mov_sensor[positions],
    )


def _aggregate_metrics(prediction_rows: pd.DataFrame) -> dict[str, Any]:
    return prediction_metrics(
        prediction_rows["true_class_label"].astype(str).to_numpy(),
        prediction_rows["predicted_class"].astype(str).to_numpy(),
        class_order=CLASS_ORDER,
    )


def _group_condition_metrics(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, frame in subject_metrics.groupby("condition", sort=False):
        values = frame["balanced_accuracy"].to_numpy(dtype=float)
        loo = [float(np.delete(values, index).mean()) for index in range(len(values))]
        rows.append(
            {
                "condition": condition,
                "n_subjects": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "above_chance_subjects": int(np.count_nonzero(values > 0.25)),
                "leave_one_subject_out_mean_min": float(min(loo)),
                "leave_one_subject_out_mean_max": float(max(loo)),
                "leave_one_subject_out_means_json": json.dumps(loo),
                "largest_subject_contribution_share": float(np.max(np.abs(values)) / np.sum(np.abs(values))),
            }
        )
    return pd.DataFrame(rows)


def _paired_tables(subject_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot = subject_metrics.pivot(index="target_subject", columns="condition", values="balanced_accuracy")
    primary_delta_rows: list[dict[str, Any]] = []
    primary_summary_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for comparison, (augmented, baseline) in PRIMARY_COMPARISONS.items():
        deltas = (pivot[augmented] - pivot[baseline]).to_numpy(dtype=float)
        signflip = exact_signflip_test(deltas)
        summary = delta_summary(deltas)
        raw_p[comparison] = float(signflip["p_raw_one_sided"])
        for subject, delta in zip(pivot.index, deltas, strict=True):
            primary_delta_rows.append(
                {
                    "comparison": comparison,
                    "target_subject": int(subject),
                    "augmented_condition": augmented,
                    "baseline_condition": baseline,
                    "augmented_ba": float(pivot.loc[subject, augmented]),
                    "baseline_ba": float(pivot.loc[subject, baseline]),
                    "delta": float(delta),
                }
            )
        primary_summary_rows.append(
            {
                "comparison": comparison,
                **{key: value for key, value in summary.items() if key != "loo_means"},
                **signflip,
            }
        )
        for omitted, value in zip(pivot.index, summary["loo_means"], strict=True):
            influence_rows.append(
                {
                    "comparison": comparison,
                    "omitted_subject": int(omitted),
                    "leave_one_subject_out_mean_delta": float(value),
                    "all_subject_mean_delta": float(summary["mean_delta"]),
                    "top_contributor_share": float(summary["top_contributor_share"]),
                }
            )
    adjusted = holm_adjust(raw_p)
    primary_summary = pd.DataFrame(primary_summary_rows)
    holm_rows = []
    order = sorted(raw_p, key=lambda name: (raw_p[name], name))
    for rank, comparison in enumerate(order, start=1):
        holm_rows.append(
            {
                "comparison": comparison,
                "holm_rank": rank,
                "p_raw": raw_p[comparison],
                "p_holm": adjusted[comparison],
                "reject_0_05": adjusted[comparison] <= 0.05,
            }
        )
    secondary_rows: list[dict[str, Any]] = []
    for comparison, (augmented, baseline) in SECONDARY_COMPARISONS.items():
        deltas = (pivot[augmented] - pivot[baseline]).to_numpy(dtype=float)
        signflip = exact_signflip_test(deltas)
        summary = delta_summary(deltas)
        for subject, delta in zip(pivot.index, deltas, strict=True):
            secondary_rows.append(
                {
                    "row_type": "subject_delta",
                    "comparison": comparison,
                    "target_subject": int(subject),
                    "augmented_condition": augmented,
                    "baseline_condition": baseline,
                    "delta": float(delta),
                    "mean_delta": pd.NA,
                    "median_delta": pd.NA,
                    "positive_subjects": pd.NA,
                    "p_raw_one_sided": pd.NA,
                }
            )
        secondary_rows.append(
            {
                "row_type": "summary",
                "comparison": comparison,
                "target_subject": pd.NA,
                "augmented_condition": augmented,
                "baseline_condition": baseline,
                "delta": pd.NA,
                "mean_delta": summary["mean_delta"],
                "median_delta": summary["median_delta"],
                "positive_subjects": summary["positive_subjects"],
                "p_raw_one_sided": signflip["p_raw_one_sided"],
            }
        )
    return (
        pd.DataFrame(primary_delta_rows),
        primary_summary,
        pd.DataFrame(holm_rows),
        pd.DataFrame(secondary_rows),
        pd.DataFrame(influence_rows),
    )


def _recall_and_confusion(subject_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recalls = []
    confusions = []
    for row in subject_metrics.itertuples(index=False):
        matrix = np.asarray(json.loads(row.confusion_matrix_json), dtype=int)
        for index, label in enumerate(CLASS_ORDER):
            recalls.append(
                {
                    "protocol": "T2",
                    "target_subject": int(row.target_subject),
                    "condition": row.condition,
                    "class_label": label,
                    "recall": float(getattr(row, f"recall_{label}")),
                }
            )
            for column, predicted in enumerate(CLASS_ORDER):
                confusions.append(
                    {
                        "protocol": "T2",
                        "target_subject": int(row.target_subject),
                        "condition": row.condition,
                        "true_class": label,
                        "predicted_class": predicted,
                        "count": int(matrix[index, column]),
                    }
                )
    return pd.DataFrame(recalls), pd.DataFrame(confusions)


def _baseline_reproduction(root: Path, subject_metrics: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    prior = pd.read_csv(root / str(config["inputs"]["geometry_v2_results"]))
    prior = prior[
        (prior["geometry"] == "AIRM")
        & (prior["protocol"] == "T2")
        & (prior["split"] == "AGGREGATE")
        & (prior["decoder"] == "logistic")
    ][["target_subject", "balanced_accuracy"]].rename(columns={"balanced_accuracy": "geometry_v2_airm_ba"})
    current = subject_metrics[subject_metrics["condition"] == "STATIC"][
        ["target_subject", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "audit_static_ba"})
    result = current.merge(prior, on="target_subject", validate="one_to_one")
    result["audit_minus_v2"] = result["audit_static_ba"] - result["geometry_v2_airm_ba"]
    result["split_and_centering_direction_reproduced"] = True
    result["exact_decoder_reproduction"] = False
    result["difference_reason"] = "audit adds source-only StandardScaler and freezes max_iter=20000,tol=1e-6; V2 used no scaler"
    return result


def _save_figures(output: Path, subject_metrics: pd.DataFrame, primary_deltas: pd.DataFrame, recalls: pd.DataFrame, influence: pd.DataFrame) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str, source: pd.DataFrame) -> None:
        source.to_csv(figure_dir / f"{name}.csv", index=False)
        plt.tight_layout()
        plt.savefig(figure_dir / f"{name}.png", dpi=180)
        plt.savefig(figure_dir / f"{name}.pdf")
        plt.close()

    emphasized = ["STATIC", "STATIC_PLUS_LEN", "STATIC_PLUS_GRAM", "STATIC_PLUS_SENSOR"]
    source = subject_metrics.copy()
    plt.figure(figsize=(10, 5.5))
    for condition in CONDITIONS:
        frame = source[source["condition"] == condition].sort_values("target_subject")
        emphasis = condition in emphasized
        plt.plot(frame["target_subject"], frame["balanced_accuracy"], marker="o", linewidth=2.0 if emphasis else 0.8, alpha=1.0 if emphasis else 0.35, label=condition)
    plt.axhline(0.25, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Target subject")
    plt.ylabel("T2 balanced accuracy")
    plt.xticks(range(1, 10))
    plt.legend(ncol=2, fontsize=8)
    save("subject_balanced_accuracy_by_condition", source)

    plt.figure(figsize=(8, 5.5))
    for index, comparison in enumerate(PRIMARY_COMPARISONS):
        frame = primary_deltas[primary_deltas["comparison"] == comparison]
        x = np.full(len(frame), index) + np.linspace(-0.12, 0.12, len(frame))
        plt.scatter(x, frame["delta"], label=comparison)
        plt.plot([index - 0.18, index + 0.18], [frame["delta"].mean()] * 2, color="black", linewidth=2)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(range(3), list(PRIMARY_COMPARISONS))
    plt.ylabel("Subject BA delta vs STATIC")
    save("paired_delta_vs_static", primary_deltas)

    recall_source = recalls[recalls["condition"].isin(["STATIC", "STATIC_PLUS_GRAM", "STATIC_PLUS_SENSOR"])]
    plt.figure(figsize=(9, 5.5))
    for condition in ["STATIC", "STATIC_PLUS_GRAM", "STATIC_PLUS_SENSOR"]:
        frame = recall_source[recall_source["condition"] == condition]
        means = frame.groupby("class_label")["recall"].mean().reindex(CLASS_ORDER)
        plt.plot(CLASS_ORDER, means, marker="o", label=condition)
    plt.ylabel("Mean T2 per-class recall")
    plt.legend()
    save("per_class_recall", recall_source)

    movement_source = subject_metrics[subject_metrics["condition"].isin(["MOV_LEN", "MOV_GRAM", "MOV_SENSOR"])]
    plt.figure(figsize=(8, 5.5))
    for condition in ["MOV_LEN", "MOV_GRAM", "MOV_SENSOR"]:
        frame = movement_source[movement_source["condition"] == condition].sort_values("target_subject")
        plt.plot(frame["target_subject"], frame["balanced_accuracy"], marker="o", label=condition)
    plt.axhline(0.25, color="black", linestyle="--", linewidth=1)
    plt.xticks(range(1, 10))
    plt.xlabel("Target subject")
    plt.ylabel("T2 balanced accuracy")
    plt.legend()
    save("movement_only_performance", movement_source)

    plt.figure(figsize=(9, 5.5))
    for comparison in PRIMARY_COMPARISONS:
        frame = influence[influence["comparison"] == comparison].sort_values("omitted_subject")
        plt.plot(frame["omitted_subject"], frame["leave_one_subject_out_mean_delta"], marker="o", label=comparison)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(range(1, 10))
    plt.xlabel("Omitted subject")
    plt.ylabel("Leave-one-subject-out mean delta")
    plt.legend()
    save("influence_summary", influence)


def _required_output_paths() -> tuple[str, ...]:
    return (
        "protocol/PROTOCOL_TRIAL_MOVEMENT_INCREMENTAL_UTILITY_V0.md",
        "protocol/frozen_config.yaml",
        "protocol/pre_result_commit.json",
        "protocol/input_hashes.csv",
        "protocol/environment.json",
        "protocol/test_results.json",
        "protocol/parent_immutability.json",
        "features/trial_movement_features.npz",
        "features/trial_feature_metadata.csv",
        "features/feature_hashes.csv",
        "tables/data_contract.csv",
        "tables/geometry_gates.csv",
        "tables/feature_contract.csv",
        "tables/target_split_audit.csv",
        "tables/convergence_audit.csv",
        "tables/subject_condition_metrics.csv",
        "tables/group_condition_metrics.csv",
        "tables/primary_paired_deltas.csv",
        "tables/primary_signflip_results.csv",
        "tables/holm_results.csv",
        "tables/secondary_paired_deltas.csv",
        "tables/per_class_recall.csv",
        "tables/confusion_matrices.csv",
        "tables/influence_summary.csv",
        "tables/t1_sensitivity_metrics.csv",
        "predictions/t2_trial_predictions.csv",
        "predictions/t1_trial_predictions.csv",
        "report/trial_movement_incremental_utility_v0.md",
        "decisions/terminal_decision.json",
    )


def assert_output_contract(output: Path) -> None:
    missing = [path for path in _required_output_paths() if not (output / path).is_file()]
    for figure in (
        "subject_balanced_accuracy_by_condition",
        "paired_delta_vs_static",
        "per_class_recall",
        "movement_only_performance",
        "influence_summary",
    ):
        for suffix in (".png", ".pdf", ".csv"):
            if not (output / "figures" / f"{figure}{suffix}").is_file():
                missing.append(f"figures/{figure}{suffix}")
    if missing:
        raise TrialMovementAuditError(f"UNASSESSED_OUTPUT_CONTRACT_FAILURE: {missing}")


def execute_audit(project_root: str | Path, config_path: str | Path, pre_result_commit: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    if _git(root, "branch", "--show-current") != BRANCH:
        raise TrialMovementAuditError("execution is not on the frozen branch")
    if _git(root, "rev-parse", "HEAD") != pre_result_commit:
        raise TrialMovementAuditError("HEAD does not equal supplied protocol-freeze commit")
    if _git(root, "status", "--porcelain"):
        raise TrialMovementAuditError("scientific execution requires a clean worktree")
    output = root / str(config["project"]["output_dir"])
    if output.exists():
        raise TrialMovementAuditError(f"refusing to overwrite existing output root: {output}")
    prior_manifest = _manifest(root, output)
    for subdirectory in ("protocol", "features", "predictions", "tables", "figures", "report", "decisions"):
        (output / subdirectory).mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / "docs/PROTOCOL_TRIAL_MOVEMENT_INCREMENTAL_UTILITY_V0.md", output / "protocol/PROTOCOL_TRIAL_MOVEMENT_INCREMENTAL_UTILITY_V0.md")
    shutil.copy2(config_file, output / "protocol/frozen_config.yaml")
    _json_dump(
        {"base_commit": BASE_COMMIT, "branch": BRANCH, "protocol_freeze_commit": pre_result_commit},
        output / "protocol/pre_result_commit.json",
    )
    _json_dump({"status": "PENDING_POST_EXECUTION_TESTS"}, output / "protocol/test_results.json")

    input_rows = []
    input_paths = [config_file]
    for key in ("v1_config", "covariances", "whole_metadata", "window5_metadata"):
        input_paths.append(root / str(config["inputs"][key]["path"]))
    input_paths.extend(root / str(path) for path in config["inputs"]["relevant_sources"])
    input_paths.extend(
        [root / str(config["inputs"]["geometry_v2_config"]), root / str(config["inputs"]["geometry_v2_protocol"])]
    )
    for path in input_paths:
        input_rows.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    pd.DataFrame(input_rows).to_csv(output / "protocol/input_hashes.csv", index=False)
    environment = collect_environment_metadata()
    environment["git"] = {"head": pre_result_commit, "branch": BRANCH}
    environment["thread_environment"] = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    _json_dump(environment, output / "protocol/environment.json")

    frozen = load_frozen_inputs(root, config)
    frozen.data_contract.to_csv(output / "tables/data_contract.csv", index=False)
    movement = build_trial_movement_features(frozen, config)
    movement.metadata.to_csv(output / "features/trial_feature_metadata.csv", index=False)
    np.savez_compressed(
        output / "features/trial_movement_features.npz",
        mov_len=movement.mov_len,
        mov_gram=movement.mov_gram,
        mov_sensor=movement.mov_sensor,
        trial_uid=movement.metadata["trial_uid"].astype(str).to_numpy(dtype=str),
        subject=movement.metadata["subject"].to_numpy(dtype=np.int64),
        class_label=movement.metadata["class_label"].astype(str).to_numpy(dtype=str),
        run=movement.metadata["run"].to_numpy(dtype=np.int64),
        session=movement.metadata["session"].astype(str).to_numpy(dtype=str),
    )
    feature_hashes = pd.DataFrame(
        [
            {"array": name, "sha256": array_sha256(values), "shape": json.dumps(list(values.shape)), "dtype": str(values.dtype)}
            for name, values in (("mov_len", movement.mov_len), ("mov_gram", movement.mov_gram), ("mov_sensor", movement.mov_sensor))
        ]
    )
    feature_hashes.to_csv(output / "features/feature_hashes.csv", index=False)
    movement.feature_contract.to_csv(output / "tables/feature_contract.csv", index=False)

    full_static, target_states, static_gates, split_audit = _precompute_static(frozen, config)
    geometry_gates = pd.concat([movement.geometry_gates, static_gates], ignore_index=True, sort=False)
    geometry_gates.to_csv(output / "tables/geometry_gates.csv", index=False)
    split_audit.to_csv(output / "tables/target_split_audit.csv", index=False)

    metadata = frozen.whole_metadata
    identity = metadata.drop(columns=["class_label"])
    decoder_audits: list[dict[str, Any]] = []
    t2_predictions: list[pd.DataFrame] = []
    t1_predictions: list[pd.DataFrame] = []
    subject_metric_rows: list[dict[str, Any]] = []
    t1_metric_rows: list[dict[str, Any]] = []
    split_metric_rows: list[dict[str, Any]] = []
    for target in config["dataset"]["subjects"]:
        partition = make_loso_partition(identity, int(target))
        source = np.asarray(partition.source_row_positions, dtype=np.int64)
        target_all = np.asarray(partition.target_row_positions, dtype=np.int64)
        source_labels = metadata.iloc[source]["class_label"].astype(str).to_numpy()
        source_uids = metadata.iloc[source]["trial_uid"].astype(str).tolist()
        splits = make_calibration_splits(identity, int(target))
        for condition in CONDITIONS:
            source_features = _condition_input(condition, source, full_static[source], movement)
            scaler, model, audit = fit_fixed_source_decoder(
                source_features,
                source_labels,
                target_subject=int(target),
                condition=condition,
                source_trial_uids=source_uids,
                config=config,
            )
            decoder_audits.append(asdict(audit))
            if audit.convergence_warning:
                continue
            t1_features = _condition_input(condition, target_all, full_static[target_all], movement)
            t1_rows, t1_metrics, _ = _score_rows(
                protocol="T1",
                split="ALL",
                target=int(target),
                condition=condition,
                uids=metadata.iloc[target_all]["trial_uid"].astype(str).to_numpy(),
                metadata=metadata.iloc[target_all].reset_index(drop=True),
                scaler=scaler,
                model=model,
                features=t1_features,
            )
            t1_predictions.append(t1_rows)
            t1_metric_rows.append(_metric_record("T1", "ALL", int(target), condition, t1_metrics))
            condition_t2_rows = []
            split_bas = []
            for split in splits:
                evaluation, target_static = target_states[(int(target), split.name)]
                target_features = _condition_input(condition, evaluation, target_static, movement)
                prediction_rows, metrics, _ = _score_rows(
                    protocol="T2",
                    split=split.name,
                    target=int(target),
                    condition=condition,
                    uids=metadata.iloc[evaluation]["trial_uid"].astype(str).to_numpy(),
                    metadata=metadata.iloc[evaluation].reset_index(drop=True),
                    scaler=scaler,
                    model=model,
                    features=target_features,
                )
                condition_t2_rows.append(prediction_rows)
                t2_predictions.append(prediction_rows)
                split_metric_rows.append(_metric_record("T2", split.name, int(target), condition, metrics))
                split_bas.append(float(metrics["balanced_accuracy"]))
            combined_predictions = pd.concat(condition_t2_rows, ignore_index=True)
            combined_metrics = _aggregate_metrics(combined_predictions)
            if abs(float(combined_metrics["balanced_accuracy"]) - float(np.mean(split_bas))) > 1e-12:
                raise TrialMovementAuditError("T2 direction aggregation contract failed")
            subject_metric_rows.append(_metric_record("T2", "SUBJECT_AVERAGE", int(target), condition, combined_metrics))

    convergence = pd.DataFrame(decoder_audits)
    convergence["warning_messages"] = convergence["warning_messages"].map(lambda value: json.dumps(list(value)))
    convergence.to_csv(output / "tables/convergence_audit.csv", index=False)
    t2_prediction_table = pd.concat(t2_predictions, ignore_index=True) if t2_predictions else pd.DataFrame()
    t1_prediction_table = pd.concat(t1_predictions, ignore_index=True) if t1_predictions else pd.DataFrame()
    t2_prediction_table.to_csv(output / "predictions/t2_trial_predictions.csv", index=False)
    t1_prediction_table.to_csv(output / "predictions/t1_trial_predictions.csv", index=False)
    subject_metrics = pd.DataFrame(subject_metric_rows)
    subject_metrics.to_csv(output / "tables/subject_condition_metrics.csv", index=False)
    group_metrics = _group_condition_metrics(subject_metrics)
    group_metrics.to_csv(output / "tables/group_condition_metrics.csv", index=False)
    t1_subject = pd.DataFrame(t1_metric_rows)
    t1_group = _group_condition_metrics(t1_subject.rename(columns={"target_subject": "target_subject"}))
    t1_group.insert(0, "row_type", "group")
    t1_subject.insert(0, "row_type", "subject")
    pd.concat([t1_subject, t1_group], ignore_index=True, sort=False).to_csv(output / "tables/t1_sensitivity_metrics.csv", index=False)

    primary_deltas, signflip, holm, secondary, influence = _paired_tables(subject_metrics)
    primary_deltas.to_csv(output / "tables/primary_paired_deltas.csv", index=False)
    signflip.to_csv(output / "tables/primary_signflip_results.csv", index=False)
    holm.to_csv(output / "tables/holm_results.csv", index=False)
    secondary.to_csv(output / "tables/secondary_paired_deltas.csv", index=False)
    influence.to_csv(output / "tables/influence_summary.csv", index=False)
    recalls, confusions = _recall_and_confusion(subject_metrics)
    recalls.to_csv(output / "tables/per_class_recall.csv", index=False)
    confusions.to_csv(output / "tables/confusion_matrices.csv", index=False)
    pd.DataFrame(split_metric_rows).to_csv(output / "tables/t2_direction_metrics.csv", index=False)
    reproduction = _baseline_reproduction(root, subject_metrics, config)
    reproduction.to_csv(output / "tables/baseline_reproduction.csv", index=False)

    invalid_primary = convergence[
        convergence["condition"].isin(["STATIC", "STATIC_PLUS_LEN", "STATIC_PLUS_GRAM", "STATIC_PLUS_SENSOR"])
        & convergence["convergence_warning"]
    ]
    static_group = group_metrics[group_metrics["condition"] == "STATIC"]
    if len(subject_metrics) != 9 * 7 or not invalid_primary.empty:
        decision = "UNASSESSED_CONVERGENCE_FAILURE"
        states: dict[str, Any] = {}
    elif static_group.empty or float(static_group.iloc[0]["mean"]) < float(config["statistics"]["static_sanity_mean_ba_min"]):
        decision = "UNASSESSED_STATIC_SANITY_FAILURE"
        states = {}
    else:
        holm_map = dict(zip(holm["comparison"], holm["p_holm"], strict=True))
        decision, states = terminal_decision(signflip, holm_map)
    decision_payload = {
        "terminal_decision": decision,
        "primary_protocol": "T2 complementary held-out-run target centering; directions averaged within subject",
        "states": states,
        "static_sanity_threshold": float(config["statistics"]["static_sanity_mean_ba_min"]),
        "static_mean_ba": None if static_group.empty else float(static_group.iloc[0]["mean"]),
        "claim_boundaries": "No causal, neural-mechanism, complete-quotient, adaptation-method, or guaranteed-SPDNet claim.",
    }
    _json_dump(decision_payload, output / "decisions/terminal_decision.json")
    _save_figures(output, subject_metrics, primary_deltas, recalls, influence)

    group_lookup = group_metrics.set_index("condition")["mean"].to_dict()
    report_lines = [
        "# Trial Movement Incremental Utility Audit V0",
        "",
        f"Terminal decision: **{decision}**.",
        "",
        "Primary protocol was V2's complementary held-out-run T2 centering. All movement features were computed independently per trial before classifier fitting. Target labels entered only metric evaluation.",
        "",
        "## T2 condition means",
        "",
    ]
    for condition in CONDITIONS:
        report_lines.append(f"- {condition}: {group_lookup.get(condition, float('nan')):.6f}")
    report_lines.extend(["", "## Primary subject-level deltas", ""])
    holm_lookup = holm.set_index("comparison")
    for row in signflip.itertuples(index=False):
        report_lines.append(
            f"- {row.comparison}: mean {row.mean_delta:.6f}, median {row.median_delta:.6f}, positive subjects {int(row.positive_subjects)}/9, raw p {row.p_raw_one_sided:.6f}, Holm p {holm_lookup.loc[row.comparison, 'p_holm']:.6f}."
        )
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The result is limited to frozen trial-level features, a fixed linear decoder, BNCI2014_001 session 0train, and the frozen LOSO protocol. MOV_GRAM is not a complete quotient geometry. A negative result does not prove that every nonlinear network fails; a positive result does not establish mechanism or causality.",
            "",
            "## Baseline reproduction note",
            "",
            "V2 AIRM T2 split/centering direction was reproduced and exact subject differences were saved. Exact decoder reproduction is not claimed because this audit prespecified source-only StandardScaler and tighter logistic stopping parameters, whereas V2 did not scale.",
        ]
    )
    (output / "report/trial_movement_incremental_utility_v0.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    current_manifest = _manifest(root, output)
    changed = sorted(
        set(prior_manifest) ^ set(current_manifest)
        | {key for key in set(prior_manifest) & set(current_manifest) if prior_manifest[key] != current_manifest[key]}
    )
    cache_paths = [root / str(config["inputs"][key]["path"]) for key in ("covariances", "whole_metadata", "window5_metadata")]
    immutability = {
        "prior_file_count": len(prior_manifest),
        "prior_manifest_sha256": stable_json_hash(prior_manifest),
        "post_manifest_sha256": stable_json_hash(current_manifest),
        "changed_prior_paths": changed,
        "cache_input_hashes": {str(path.relative_to(root)): sha256_file(path) for path in cache_paths},
        "passed": not changed,
    }
    _json_dump(immutability, output / "protocol/parent_immutability.json")
    if changed:
        raise TrialMovementAuditError(f"UNASSESSED_IMMUTABILITY_FAILURE: {changed}")
    assert_output_contract(output)
    return {
        "terminal_decision": decision,
        "output_dir": str(output),
        "group_metrics": group_lookup,
        "primary": signflip.to_dict(orient="records"),
        "holm": holm.to_dict(orient="records"),
    }


def record_test_results(
    project_root: str | Path,
    config_path: str | Path,
    *,
    focused_before: Path,
    full_tests: Path,
    focused_after: Path,
) -> None:
    root = Path(project_root).resolve()
    config = load_config(config_path)
    output = root / str(config["project"]["output_dir"])
    records = {}
    for name, path in (
        ("focused_before", focused_before),
        ("full_repository", full_tests),
        ("focused_after", focused_after),
    ):
        text = Path(path).read_text(encoding="utf-8")
        records[name] = {
            "log_path": str(Path(path).resolve()),
            "log_sha256": sha256_file(path),
            "passed": "TEST_EXIT_CODE=0" in text,
            "tail": text[-4000:],
        }
    records["all_required_passed"] = all(value["passed"] for value in records.values())
    _json_dump(records, output / "protocol/test_results.json")


__all__ = [
    "BRANCH",
    "CLASS_ORDER",
    "CONDITIONS",
    "DIMENSIONS",
    "FrozenInputs",
    "MovementFeatures",
    "TrialMovementAuditError",
    "assert_output_contract",
    "build_trial_movement_features",
    "condition_features",
    "delta_summary",
    "exact_signflip_test",
    "exact_subject_bootstrap_interval",
    "execute_audit",
    "feature_functions_are_label_free",
    "fit_fixed_source_decoder",
    "holm_adjust",
    "load_config",
    "load_frozen_inputs",
    "movement_feature_tuple",
    "record_test_results",
    "terminal_decision",
]
