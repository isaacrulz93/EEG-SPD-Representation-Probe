"""Exact squared-cost decomposition of frozen Local Mean Movement V0 artifacts.

The module never fits covariance means or anti-developments.  Full-data
common-O costs are supplied by the frozen Movement V0 distance matrix.  Only
the two prespecified split-half cross-session replicates require new quotient
fits, using the unchanged Movement V0 optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.local_mean_movement_v0 import plus_one_pvalue
from src.local_temporal_sequence_v0 import (
    CLASS_ORDER,
    DEFAULT_MASTER_SEED,
    DEFAULT_NULL_REPLICATES,
    N_CELLS,
    N_CLASSES,
    N_SUBJECTS,
    classbreak_mappings,
    subjectbreak_mappings,
)


N_STEPS = 4
N_CHANNELS = 22
N_SESSIONS = 2
N_HALVES = 2
COMPONENT_ORDER = ("len", "ang", "ori", "full", "sensor")
ADDITIVE_COMPONENT_ORDER = ("len", "ang", "ori")
ABSOLUTE_TOLERANCE = 1.0e-8
RELATIVE_TOLERANCE = 1.0e-8
SAFE_FRACTION_DENOMINATOR = 1.0e-12
ALPHA = 0.05


class ComponentDecompositionNumericalError(RuntimeError):
    """A nonnegativity, reproduction, reconstruction, or quotient gate failed."""


@dataclass(frozen=True)
class ComponentMatrices:
    sensor: np.ndarray
    full: np.ndarray
    length: np.ndarray
    angular: np.ndarray
    orientation: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "len": self.length,
            "ang": self.angular,
            "ori": self.orientation,
            "full": self.full,
            "sensor": self.sensor,
        }


@dataclass(frozen=True)
class RelationStatistics:
    a_sc: np.ndarray
    b_sc: np.ndarray
    c_sc: np.ndarray
    d_sc: np.ndarray
    s_sc: np.ndarray
    c_specific_sc: np.ndarray
    j_sc: np.ndarray
    s_s: np.ndarray
    c_s: np.ndarray
    j_s: np.ndarray
    t_subject: float
    t_class: float
    t_j: float


@dataclass(frozen=True)
class ComponentInference:
    observed: RelationStatistics
    subjectbreak_t_subject: np.ndarray
    subjectbreak_t_j: np.ndarray
    classbreak_t_class: np.ndarray
    classbreak_t_j: np.ndarray
    p_subject: float
    p_class: float
    p_j_subjectbreak: float
    p_j_classbreak: float


def cell_index(subject_index: int, class_index: int) -> int:
    if not 0 <= subject_index < N_SUBJECTS:
        raise ValueError("subject index must be in 0..8")
    if not 0 <= class_index < N_CLASSES:
        raise ValueError("class index must be in 0..3")
    return subject_index * N_CLASSES + class_index


def canonical_cell_subjects() -> np.ndarray:
    return np.repeat(np.arange(1, N_SUBJECTS + 1, dtype=np.int64), N_CLASSES)


def canonical_cell_classes() -> np.ndarray:
    return np.tile(np.asarray(CLASS_ORDER), N_SUBJECTS)


def _matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (N_CELLS, N_CELLS):
        raise ValueError(f"{name} must have shape (36,36)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def sensor_and_length_squared_costs(
    session0: np.ndarray, session1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independently compute sensor and speed costs from frozen Z tuples."""

    first = np.asarray(session0, dtype=np.float64)
    second = np.asarray(session1, dtype=np.float64)
    expected = (N_CELLS, N_STEPS, N_CHANNELS, N_CHANNELS)
    if first.shape != expected or second.shape != expected:
        raise ValueError(f"movement banks must each have shape {expected}")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("movement banks must be finite")
    speed0 = np.linalg.norm(first, axis=(2, 3))
    speed1 = np.linalg.norm(second, axis=(2, 3))
    residual = first[:, None, ...] - second[None, :, ...]
    sensor = np.mean(np.sum(residual * residual, axis=(3, 4)), axis=2)
    length_step = 0.25 * (speed0[:, None, :] - speed1[None, :, :]) ** 2
    length = np.sum(length_step, axis=2)
    return sensor, length, speed0, speed1


def build_component_matrices(
    d_mov: np.ndarray,
    d_len: np.ndarray,
    d_direct: np.ndarray,
) -> ComponentMatrices:
    """Square frozen root distances and form the exact three-layer components."""

    full = np.square(_matrix(d_mov, name="d_mov"))
    length = np.square(_matrix(d_len, name="d_len"))
    sensor = np.square(_matrix(d_direct, name="d_direct"))
    angular = full - length
    orientation = sensor - full
    return ComponentMatrices(
        sensor=sensor,
        full=full,
        length=length,
        angular=angular,
        orientation=orientation,
    )


def _maximum_error(left: np.ndarray | float, right: np.ndarray | float) -> tuple[float, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    absolute = np.abs(a - b)
    relative = absolute / np.maximum(np.maximum(np.abs(a), np.abs(b)), np.finfo(float).tiny)
    return float(np.max(absolute)), float(np.max(relative))


def _assert_close(
    left: np.ndarray | float,
    right: np.ndarray | float,
    *,
    name: str,
    atol: float = ABSOLUTE_TOLERANCE,
    rtol: float = RELATIVE_TOLERANCE,
) -> tuple[float, float]:
    absolute, relative = _maximum_error(left, right)
    if not np.allclose(left, right, atol=atol, rtol=rtol):
        raise ComponentDecompositionNumericalError(
            "UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE: "
            f"{name} failed (max_abs={absolute:.17g}, max_rel={relative:.17g})"
        )
    return absolute, relative


def _negative_diagnostics(
    values: np.ndarray,
    scale_left: np.ndarray,
    scale_right: np.ndarray,
    *,
    name: str,
    atol: float = ABSOLUTE_TOLERANCE,
    rtol: float = RELATIVE_TOLERANCE,
) -> dict[str, float | int | str]:
    array = np.asarray(values, dtype=np.float64)
    bound = atol + rtol * np.maximum(np.abs(scale_left), np.abs(scale_right))
    meaningful = array < -bound
    record: dict[str, float | int | str] = {
        "check": f"{name}_nonnegative",
        "minimum_raw_value": float(np.min(array)),
        "negative_raw_count": int(np.count_nonzero(array < 0.0)),
        "meaningful_negative_count": int(np.count_nonzero(meaningful)),
        "maximum_allowed_negative_tolerance": float(np.max(bound)),
    }
    if np.any(meaningful):
        index = tuple(int(value) for value in np.argwhere(meaningful)[0])
        raise ComponentDecompositionNumericalError(
            "UNASSESSED_COMPONENT_DECOMPOSITION_NUMERICAL_FAILURE: "
            f"meaningful negative {name} at {index}: {array[index]:.17g}"
        )
    return record


def pairwise_numerical_gates(
    matrices: ComponentMatrices,
    *,
    independently_computed_sensor: np.ndarray,
    independently_computed_length: np.ndarray,
    d_mov: np.ndarray,
    d_len: np.ndarray,
    d_direct: np.ndarray,
    atol: float = ABSOLUTE_TOLERANCE,
    rtol: float = RELATIVE_TOLERANCE,
) -> list[dict[str, float | int | str]]:
    """Certify all 1,296 raw squared-cost decompositions without clipping."""

    records: list[dict[str, float | int | str]] = []
    records.append(
        _negative_diagnostics(
            matrices.length,
            matrices.length,
            np.zeros_like(matrices.length),
            name="c_len",
            atol=atol,
            rtol=rtol,
        )
    )
    records.append(
        _negative_diagnostics(
            matrices.angular,
            matrices.full,
            matrices.length,
            name="c_ang",
            atol=atol,
            rtol=rtol,
        )
    )
    records.append(
        _negative_diagnostics(
            matrices.orientation,
            matrices.sensor,
            matrices.full,
            name="c_ori",
            atol=atol,
            rtol=rtol,
        )
    )
    checks = {
        "pair_full_equals_len_plus_ang": (matrices.full, matrices.length + matrices.angular),
        "pair_sensor_equals_full_plus_ori": (
            matrices.sensor,
            matrices.full + matrices.orientation,
        ),
        "pair_three_way_reconstruction": (
            matrices.sensor,
            matrices.length + matrices.angular + matrices.orientation,
        ),
        "independent_c_len_reproduction": (matrices.length, independently_computed_length),
        "independent_c_sensor_reproduction": (
            matrices.sensor,
            independently_computed_sensor,
        ),
        "root_d_mov_squared_reproduction": (matrices.full, np.square(d_mov)),
        "root_d_len_squared_reproduction": (matrices.length, np.square(d_len)),
        "root_d_direct_squared_reproduction": (matrices.sensor, np.square(d_direct)),
    }
    for name, (left, right) in checks.items():
        absolute, relative = _assert_close(
            left, right, name=name, atol=atol, rtol=rtol
        )
        records.append(
            {
                "check": name,
                "maximum_absolute_error": absolute,
                "maximum_relative_error": relative,
                "meaningful_negative_count": 0,
            }
        )
    return records


def relation_statistics(matrix: np.ndarray) -> RelationStatistics:
    """Compute frozen relation-cell means and S/C/J squared-cost contrasts."""

    values = _matrix(matrix, name="component cost")
    a = np.empty((N_SUBJECTS, N_CLASSES), dtype=np.float64)
    b = np.empty_like(a)
    c = np.empty_like(a)
    d = np.empty_like(a)
    for subject in range(N_SUBJECTS):
        for class_index in range(N_CLASSES):
            anchor = cell_index(subject, class_index)
            a[subject, class_index] = values[anchor, anchor]
            b[subject, class_index] = np.mean(
                [
                    values[anchor, cell_index(subject, other_class)]
                    for other_class in range(N_CLASSES)
                    if other_class != class_index
                ]
            )
            c[subject, class_index] = np.mean(
                [
                    values[anchor, cell_index(other_subject, class_index)]
                    for other_subject in range(N_SUBJECTS)
                    if other_subject != subject
                ]
            )
            d[subject, class_index] = np.mean(
                [
                    values[anchor, cell_index(other_subject, other_class)]
                    for other_subject in range(N_SUBJECTS)
                    if other_subject != subject
                    for other_class in range(N_CLASSES)
                    if other_class != class_index
                ]
            )
    s_sc = c - a
    c_specific_sc = b - a
    j_sc = b + c - a - d
    s_s = np.mean(s_sc, axis=1)
    c_s = np.mean(c_specific_sc, axis=1)
    j_s = np.mean(j_sc, axis=1)
    return RelationStatistics(
        a_sc=a,
        b_sc=b,
        c_sc=c,
        d_sc=d,
        s_sc=s_sc,
        c_specific_sc=c_specific_sc,
        j_sc=j_sc,
        s_s=s_s,
        c_s=c_s,
        j_s=j_s,
        t_subject=float(np.mean(s_s)),
        t_class=float(np.mean(c_s)),
        t_j=float(np.mean(j_s)),
    )


def statistic_reconstruction_gates(
    statistics: Mapping[str, RelationStatistics],
    *,
    atol: float = ABSOLUTE_TOLERANCE,
    rtol: float = RELATIVE_TOLERANCE,
) -> list[dict[str, float | str]]:
    """Require full and three-layer reconstruction at every S/C/J level."""

    if set(statistics) != set(COMPONENT_ORDER):
        raise ValueError(f"statistics must contain exactly {COMPONENT_ORDER}")
    fields = ("s_sc", "c_specific_sc", "j_sc", "s_s", "c_s", "j_s", "t_subject", "t_class", "t_j")
    records: list[dict[str, float | str]] = []
    for field in fields:
        sensor = getattr(statistics["sensor"], field)
        full = getattr(statistics["full"], field)
        length = getattr(statistics["len"], field)
        angular = getattr(statistics["ang"], field)
        orientation = getattr(statistics["ori"], field)
        for name, left, right in (
            (f"{field}_full_equals_len_plus_ang", full, length + angular),
            (
                f"{field}_sensor_equals_len_plus_ang_plus_ori",
                sensor,
                length + angular + orientation,
            ),
        ):
            absolute, relative = _assert_close(
                left, right, name=name, atol=atol, rtol=rtol
            )
            records.append(
                {
                    "check": name,
                    "maximum_absolute_error": absolute,
                    "maximum_relative_error": relative,
                }
            )
    return records


def evaluate_all_components(
    matrices: Mapping[str, np.ndarray],
    *,
    subject_mappings: np.ndarray | None = None,
    class_mappings: np.ndarray | None = None,
    replicates: int = DEFAULT_NULL_REPLICATES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> dict[str, ComponentInference]:
    """Evaluate all components with the same indexed frozen mapping draws."""

    if set(matrices) != set(COMPONENT_ORDER):
        raise ValueError(f"matrices must contain exactly {COMPONENT_ORDER}")
    checked = {key: _matrix(value, name=key) for key, value in matrices.items()}
    generated_subject = subjectbreak_mappings(replicates=replicates, master_seed=master_seed)
    generated_class = classbreak_mappings(replicates=replicates, master_seed=master_seed)
    subject_maps = generated_subject if subject_mappings is None else np.asarray(subject_mappings)
    class_maps = generated_class if class_mappings is None else np.asarray(class_mappings)
    if subject_maps.shape != (replicates, N_CELLS) or not np.array_equal(
        subject_maps, generated_subject
    ):
        raise ValueError("saved subject-break mappings differ from the frozen stream")
    if class_maps.shape != (replicates, N_CELLS) or not np.array_equal(
        class_maps, generated_class
    ):
        raise ValueError("saved class-break mappings differ from the frozen stream")
    observed = {key: relation_statistics(value) for key, value in checked.items()}
    nulls = {
        key: {
            "subject": np.empty(replicates, dtype=np.float64),
            "subject_j": np.empty(replicates, dtype=np.float64),
            "class": np.empty(replicates, dtype=np.float64),
            "class_j": np.empty(replicates, dtype=np.float64),
        }
        for key in COMPONENT_ORDER
    }
    for replicate in range(replicates):
        subject_columns = subject_maps[replicate]
        class_columns = class_maps[replicate]
        for key in COMPONENT_ORDER:
            subject_result = relation_statistics(checked[key][:, subject_columns])
            class_result = relation_statistics(checked[key][:, class_columns])
            nulls[key]["subject"][replicate] = subject_result.t_subject
            nulls[key]["subject_j"][replicate] = subject_result.t_j
            nulls[key]["class"][replicate] = class_result.t_class
            nulls[key]["class_j"][replicate] = class_result.t_j
    return {
        key: ComponentInference(
            observed=observed[key],
            subjectbreak_t_subject=nulls[key]["subject"],
            subjectbreak_t_j=nulls[key]["subject_j"],
            classbreak_t_class=nulls[key]["class"],
            classbreak_t_j=nulls[key]["class_j"],
            p_subject=plus_one_pvalue(observed[key].t_subject, nulls[key]["subject"]),
            p_class=plus_one_pvalue(observed[key].t_class, nulls[key]["class"]),
            p_j_subjectbreak=plus_one_pvalue(
                observed[key].t_j, nulls[key]["subject_j"]
            ),
            p_j_classbreak=plus_one_pvalue(observed[key].t_j, nulls[key]["class_j"]),
        )
        for key in COMPONENT_ORDER
    }


def null_reconstruction_gates(
    inferences: Mapping[str, ComponentInference],
    *,
    atol: float = ABSOLUTE_TOLERANCE,
    rtol: float = RELATIVE_TOLERANCE,
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    fields = (
        "subjectbreak_t_subject",
        "subjectbreak_t_j",
        "classbreak_t_class",
        "classbreak_t_j",
    )
    for field in fields:
        full = getattr(inferences["full"], field)
        sensor = getattr(inferences["sensor"], field)
        length = getattr(inferences["len"], field)
        angular = getattr(inferences["ang"], field)
        orientation = getattr(inferences["ori"], field)
        for name, left, right in (
            (f"null_{field}_full_equals_len_plus_ang", full, length + angular),
            (
                f"null_{field}_sensor_equals_len_plus_ang_plus_ori",
                sensor,
                length + angular + orientation,
            ),
        ):
            absolute, relative = _assert_close(
                left, right, name=name, atol=atol, rtol=rtol
            )
            records.append(
                {
                    "check": name,
                    "maximum_absolute_error": absolute,
                    "maximum_relative_error": relative,
                }
            )
    return records


def support(t_j: float, p_subjectbreak: float, p_classbreak: float) -> bool:
    return bool(
        t_j > 0.0 and p_subjectbreak < ALPHA and p_classbreak < ALPHA
    )


def terminal_decision(
    *,
    t_j_ang: float,
    p_j_ang_subjectbreak: float,
    p_j_ang_classbreak: float,
    t_j_ori: float,
    p_j_ori_subjectbreak: float,
    p_j_ori_classbreak: float,
    t_j_len: float,
    p_j_len_subjectbreak: float,
    p_j_len_classbreak: float,
    split_half_ang_sign_stable: bool,
) -> str:
    ang_support = support(t_j_ang, p_j_ang_subjectbreak, p_j_ang_classbreak)
    ori_support = support(t_j_ori, p_j_ori_subjectbreak, p_j_ori_classbreak)
    len_support = support(t_j_len, p_j_len_subjectbreak, p_j_len_classbreak)
    if ang_support and not split_half_ang_sign_stable:
        return "UNASSESSED_COMPONENT_DECOMPOSITION_UNRELIABLE"
    if ang_support and ori_support:
        return "BOTH_INTRINSIC_RELATIVE_AND_SENSOR_FRAME_COMPONENTS"
    if ang_support:
        return "GO_DIRECTIONAL_JOINT_MOVEMENT_INTERACTION"
    if ori_support:
        return "SENSOR_FRAME_ORIENTATION_ONLY_SUPPORT"
    if len_support:
        return "SPEED_PROFILE_SUFFICIENT_AT_CURRENT_RESOLUTION"
    return "NO_COMPONENT_INTERACTION_AT_SQUARED_COST_RESOLUTION"


def split_replicate_banks(split_z: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return exactly Half-A cross-session and Half-B cross-session banks."""

    values = np.asarray(split_z, dtype=np.float64)
    expected = (N_HALVES, N_SESSIONS, N_SUBJECTS, N_CLASSES, N_STEPS, N_CHANNELS, N_CHANNELS)
    if values.shape != expected:
        raise ValueError(f"split movement bank must have shape {expected}")
    return tuple(
        (
            values[half, 0].reshape(N_CELLS, N_STEPS, N_CHANNELS, N_CHANNELS),
            values[half, 1].reshape(N_CELLS, N_STEPS, N_CHANNELS, N_CHANNELS),
        )
        for half in range(N_HALVES)
    )


def descriptive_fractions(matrices: ComponentMatrices) -> dict[str, np.ndarray]:
    valid = matrices.sensor > SAFE_FRACTION_DENOMINATOR
    result: dict[str, np.ndarray] = {"valid": valid}
    for name, values in (
        ("fraction_len", matrices.length),
        ("fraction_ang", matrices.angular),
        ("fraction_ori", matrices.orientation),
    ):
        fraction = np.full_like(matrices.sensor, np.nan)
        fraction[valid] = values[valid] / matrices.sensor[valid]
        result[name] = fraction
    return result
