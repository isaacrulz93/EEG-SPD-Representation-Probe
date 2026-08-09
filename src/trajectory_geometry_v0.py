"""Intrinsic five-state SPD trajectory geometry for Trajectory Anatomy v0.

This module is deliberately independent of data loading, labels, classifiers,
and output paths.  It implements the frozen AIRM-primary and Log-Euclidean
secondary geometry from ``PROTOCOL_TRAJECTORY_ANATOMY_V0.md`` using the V2
float64 symmetric-EVD primitives.  Eigenvalues are never clipped.

The central implementation invariant is that PATH and BAG descriptors consume
the *same* already-computed 5 x 5 distance matrix.  Order-shuffle nulls should
therefore reindex that matrix with :func:`permute_distance_matrix`, rather than
recomputing distances after shuffling covariance matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Literal, Sequence

import numpy as np

from src.geometry_v2 import (
    AIRM_MAXITER,
    AIRM_TOLERANCE,
    airm_distance,
    airm_mean,
    logeuclidean_mean_custom,
    spd_diagnostics,
    spd_invsqrt,
    spd_log,
    symmetric_exp,
    symmetrize,
)


__all__ = [
    "AIRM_METRIC",
    "LE_METRIC",
    "METRICS",
    "N_STATES",
    "PATH_D10_NAMES",
    "SCALAR_11_NAMES",
    "ALL_PERMUTATIONS_5",
    "CanonicalBag",
    "GeometryMeanResult",
    "TurningAngleResult",
    "TrajectoryGeometryResult",
    "SPDStackChecks",
    "DistanceMatrixChecks",
    "GeodesicEndpointChecks",
    "IntrinsicChecks",
    "TrajectoryHardChecks",
    "PermutationInvarianceChecks",
    "CenteringIsometryChecks",
    "FactorialDecomposition",
    "symmetric_sqrt",
    "airm_whitened_log",
    "airm_log_map",
    "airm_inner_product",
    "airm_norm",
    "airm_geodesic",
    "le_geodesic",
    "distance_matrix",
    "path_d10",
    "bag_canon_d10",
    "bag_sorted_d10",
    "permute_distance_matrix",
    "five_state_airm_barycenter",
    "five_state_le_barycenter",
    "five_state_barycenter",
    "turning_angles",
    "geodesic_deviations",
    "compute_five_state_geometry",
    "spd_stack_hard_checks",
    "distance_matrix_hard_checks",
    "geodesic_endpoint_hard_checks",
    "intrinsic_hard_checks",
    "trajectory_hard_checks",
    "check_bag_permutation_invariance",
    "compare_airm_centering_isometry",
    "balanced_factorial_decomposition",
]


AIRM_METRIC = "AIRM"
LE_METRIC = "LE"
METRICS = (AIRM_METRIC, LE_METRIC)
Metric = Literal["AIRM", "LE"]

N_STATES = 5
PATH_D10_NAMES = (
    "d12",
    "d13",
    "d14",
    "d15",
    "d23",
    "d24",
    "d25",
    "d34",
    "d35",
    "d45",
)
SCALAR_11_NAMES = (
    "total_path_length",
    "endpoint_distance",
    "efficiency",
    "excess",
    "mean_turn",
    "max_turn",
    "mean_geodesic_deviation",
    "max_geodesic_deviation",
    "frechet_variance",
    "frechet_radius_mean",
    "diameter",
)
ALL_PERMUTATIONS_5: tuple[tuple[int, ...], ...] = tuple(
    permutations(range(N_STATES))
)
UPPER_TRIANGLE = np.triu_indices(N_STATES, k=1)

SYMMETRY_TOLERANCE = 1e-12
CONDITION_NUMBER_MAX = 1e12
DISTANCE_SYMMETRY_TOLERANCE = 1e-10
DISTANCE_DIAGONAL_TOLERANCE = 1e-12
DISTANCE_NONNEGATIVE_TOLERANCE = 1e-12
TRIANGLE_ABSOLUTE_TOLERANCE = 1e-10
TRIANGLE_RELATIVE_TOLERANCE = 1e-10
PATH_ZERO_TOLERANCE = 1e-12
PATH_INEQUALITY_ABSOLUTE_TOLERANCE = 1e-10
PATH_INEQUALITY_RELATIVE_TOLERANCE = 1e-10
BOUND_TOLERANCE = 1e-12
COSINE_DOMAIN_TOLERANCE = 1e-10
KARCHER_RESIDUAL_MAX = 1e-7
GEODESIC_ENDPOINT_TOLERANCE = 1e-10
ISOMETRY_TOLERANCE = 1e-10
BAG_INVARIANCE_TOLERANCE = 1e-12
SS_CLOSURE_TOLERANCE = 1e-10


def _metric_name(metric: str) -> Metric:
    normalized = str(metric).strip().upper().replace("-", "_")
    aliases = {
        "AIRM": AIRM_METRIC,
        "AI": AIRM_METRIC,
        "RIEMANN": AIRM_METRIC,
        "LE": LE_METRIC,
        "LOGEUCLID": LE_METRIC,
        "LOG_EUCLIDEAN": LE_METRIC,
    }
    try:
        return aliases[normalized]  # type: ignore[return-value]
    except KeyError as error:
        raise ValueError(f"metric must identify AIRM or LE, got {metric!r}") from error


def _as_square(matrix: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 1:
        raise ValueError(f"{name} must have shape (channels, channels), got {value.shape}")
    return value


def _as_five_states(states: np.ndarray, *, name: str = "states") -> np.ndarray:
    value = np.asarray(states, dtype=np.float64)
    if (
        value.ndim != 3
        or value.shape[0] != N_STATES
        or value.shape[1] != value.shape[2]
        or value.shape[1] < 1
    ):
        raise ValueError(
            f"{name} must have shape (5, channels, channels), got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    relative_symmetry = np.linalg.norm(
        value - value.transpose(0, 2, 1), axis=(1, 2)
    ) / np.maximum(
        np.linalg.norm(value, axis=(1, 2)), np.finfo(np.float64).tiny
    )
    if float(np.max(relative_symmetry)) > SYMMETRY_TOLERANCE:
        raise ValueError(
            f"{name} is not symmetric within {SYMMETRY_TOLERANCE:g}"
        )
    symmetric = symmetrize(value)
    if float(np.min(np.linalg.eigvalsh(symmetric))) <= 0.0:
        raise ValueError(f"{name} contains a non-SPD matrix")
    return symmetric


def _as_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (N_STATES, N_STATES):
        raise ValueError(f"distance matrix must have shape (5, 5), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("distance matrix contains NaN or Inf")
    return value


def _relative_frobenius(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(first) - np.asarray(second), ord="fro")
        / max(np.linalg.norm(np.asarray(second), ord="fro"), np.finfo(float).tiny)
    )


def _max_absolute_relative(
    observed: np.ndarray | float,
    reference: np.ndarray | float,
) -> tuple[float, float]:
    observed_array = np.asarray(observed, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    difference = np.abs(observed_array - reference_array)
    absolute = float(np.max(difference))
    relative = float(
        np.max(
            difference
            / np.maximum(np.abs(reference_array), np.finfo(np.float64).tiny)
        )
    )
    return absolute, relative


def symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric SPD square root without eigenvalue clipping."""

    value = _as_square(matrix, name="matrix")
    return symmetric_exp(0.5 * spd_log(value))


def airm_whitened_log(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Return ``log(base^-1/2 target base^-1/2)``.

    This is the whitened tangent representation.  Its Frobenius inner product
    is the AIRM tangent inner product at ``base``.
    """

    target_value = _as_square(target, name="target")
    base_value = _as_square(base, name="base")
    if target_value.shape != base_value.shape:
        raise ValueError("target and base channel dimensions differ")
    whitening = spd_invsqrt(base_value)
    relative = symmetrize(whitening @ target_value @ whitening)
    return spd_log(relative)


def airm_log_map(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Return the full affine-invariant logarithmic map at ``base``."""

    base_value = _as_square(base, name="base")
    tangent_whitened = airm_whitened_log(target, base_value)
    root = symmetric_sqrt(base_value)
    return symmetrize(root @ tangent_whitened @ root)


def airm_inner_product(
    first_tangent: np.ndarray,
    second_tangent: np.ndarray,
    base: np.ndarray,
) -> float:
    """AIRM tangent inner product ``Tr(C^-1 U C^-1 V)``.

    The implementation uses inverse-root whitening and never forms an explicit
    matrix inverse.
    """

    base_value = _as_square(base, name="base")
    first = _as_square(first_tangent, name="first_tangent")
    second = _as_square(second_tangent, name="second_tangent")
    if first.shape != base_value.shape or second.shape != base_value.shape:
        raise ValueError("tangent and base channel dimensions differ")
    whitening = spd_invsqrt(base_value)
    first_white = whitening @ first @ whitening
    second_white = whitening @ second @ whitening
    return float(np.einsum("ij,ij->", first_white, second_white))


def airm_norm(tangent: np.ndarray, base: np.ndarray) -> float:
    """Return the AIRM tangent norm at ``base``."""

    squared = airm_inner_product(tangent, tangent, base)
    if squared < -BOUND_TOLERANCE:
        raise FloatingPointError("AIRM tangent squared norm is materially negative")
    return float(np.sqrt(max(0.0, squared)))


def airm_geodesic(first: np.ndarray, second: np.ndarray, t: float) -> np.ndarray:
    """Return the fixed affine-invariant geodesic point at parameter ``t``."""

    first_value = _as_square(first, name="first")
    second_value = _as_square(second, name="second")
    if first_value.shape != second_value.shape:
        raise ValueError("geodesic endpoint channel dimensions differ")
    if not np.isfinite(float(t)):
        raise ValueError("geodesic parameter must be finite")
    root = symmetric_sqrt(first_value)
    whitening = spd_invsqrt(first_value)
    relative = symmetrize(whitening @ second_value @ whitening)
    powered = symmetric_exp(float(t) * spd_log(relative))
    result = symmetrize(root @ powered @ root)
    # Validate the result without applying any repair.
    if float(np.min(np.linalg.eigvalsh(result))) <= 0.0:
        raise FloatingPointError("AIRM geodesic produced a non-SPD matrix")
    return result


def le_geodesic(first: np.ndarray, second: np.ndarray, t: float) -> np.ndarray:
    """Return the fixed Log-Euclidean geodesic point at parameter ``t``."""

    first_value = _as_square(first, name="first")
    second_value = _as_square(second, name="second")
    if first_value.shape != second_value.shape:
        raise ValueError("geodesic endpoint channel dimensions differ")
    if not np.isfinite(float(t)):
        raise ValueError("geodesic parameter must be finite")
    coordinate = (1.0 - float(t)) * spd_log(first_value) + float(t) * spd_log(
        second_value
    )
    return symmetric_exp(coordinate)


def distance_matrix(states: np.ndarray, metric: str) -> np.ndarray:
    """Build one exact-symmetric 5 x 5 intrinsic distance matrix."""

    values = _as_five_states(states)
    geometry = _metric_name(metric)
    result = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    if geometry == LE_METRIC:
        logs = spd_log(values)
        for left, right in combinations(range(N_STATES), 2):
            distance = float(np.linalg.norm(logs[left] - logs[right], ord="fro"))
            result[left, right] = result[right, left] = distance
    else:
        for left, right in combinations(range(N_STATES), 2):
            distance = float(airm_distance(values[left], values[right]))
            result[left, right] = result[right, left] = distance
    return result


def path_d10(matrix: np.ndarray) -> np.ndarray:
    """Return D10 in frozen chronological upper-triangle order."""

    value = _as_distance_matrix(matrix)
    return np.asarray(value[UPPER_TRIANGLE], dtype=np.float64).copy()


@dataclass(frozen=True)
class CanonicalBag:
    """Permutation-quotiented D10 plus its audit canonical labeling."""

    vector: np.ndarray
    permutation: tuple[int, ...]

    @property
    def permutation_one_based(self) -> str:
        return "-".join(str(index + 1) for index in self.permutation)


def bag_canon_d10(matrix: np.ndarray) -> CanonicalBag:
    """Return the lexicographically minimum D10 over all 120 S5 actions."""

    value = _as_distance_matrix(matrix)
    candidates: list[np.ndarray] = []
    for permutation in ALL_PERMUTATIONS_5:
        reindexed = value[np.ix_(permutation, permutation)]
        candidates.append(np.asarray(reindexed[UPPER_TRIANGLE], dtype=np.float64))
    winner = min(
        range(len(candidates)),
        key=lambda index: tuple(candidates[index].tolist()),
    )
    return CanonicalBag(
        vector=candidates[winner].copy(),
        permutation=ALL_PERMUTATIONS_5[winner],
    )


def bag_sorted_d10(matrix: np.ndarray) -> np.ndarray:
    """Return the sorted multiset of the same ten pairwise distances."""

    return np.sort(path_d10(matrix), kind="stable")


def permute_distance_matrix(
    matrix: np.ndarray,
    permutation: Sequence[int],
) -> np.ndarray:
    """Reindex a distance matrix without recomputing any metric distances."""

    value = _as_distance_matrix(matrix)
    indices = tuple(int(index) for index in permutation)
    if len(indices) != N_STATES or tuple(sorted(indices)) != tuple(range(N_STATES)):
        raise ValueError("permutation must contain each zero-based index 0 through 4")
    return value[np.ix_(indices, indices)].copy()


@dataclass(frozen=True)
class GeometryMeanResult:
    metric: Metric
    matrix: np.ndarray
    normalized_karcher_residual: float | None
    normalized_log_residual: float | None
    warning_messages: tuple[str, ...]
    solver_tol: float | None
    solver_maxiter: int | None


def five_state_airm_barycenter(states: np.ndarray) -> GeometryMeanResult:
    """Fit the frozen five-state AIRM mean and expose its audited residual."""

    values = _as_five_states(states)
    result = airm_mean(values, tol=AIRM_TOLERANCE, maxiter=AIRM_MAXITER)
    return GeometryMeanResult(
        metric=AIRM_METRIC,
        matrix=result.matrix,
        normalized_karcher_residual=result.normalized_post_residual,
        normalized_log_residual=None,
        warning_messages=result.warning_messages,
        solver_tol=result.tol,
        solver_maxiter=result.maxiter,
    )


def five_state_le_barycenter(states: np.ndarray) -> GeometryMeanResult:
    """Return the metric-consistent five-state Log-Euclidean mean."""

    values = _as_five_states(states)
    matrix = logeuclidean_mean_custom(values)
    residual = float(
        np.linalg.norm((spd_log(values) - spd_log(matrix)).mean(axis=0), ord="fro")
        / np.sqrt(values.shape[-1])
    )
    return GeometryMeanResult(
        metric=LE_METRIC,
        matrix=matrix,
        normalized_karcher_residual=None,
        normalized_log_residual=residual,
        warning_messages=(),
        solver_tol=None,
        solver_maxiter=None,
    )


def five_state_barycenter(states: np.ndarray, metric: str) -> GeometryMeanResult:
    geometry = _metric_name(metric)
    if geometry == AIRM_METRIC:
        return five_state_airm_barycenter(states)
    return five_state_le_barycenter(states)


@dataclass(frozen=True)
class TurningAngleResult:
    angles: np.ndarray
    raw_cosines: np.ndarray
    tangent_norms: np.ndarray
    degenerate_mask: np.ndarray


def turning_angles(
    states: np.ndarray,
    metric: str,
    *,
    zero_tolerance: float = PATH_ZERO_TOLERANCE,
) -> TurningAngleResult:
    """Compute frozen forward-facing intrinsic angles at states 2, 3, 4.

    Undefined angles caused by a zero tangent norm are returned as NaN and
    explicitly marked in ``degenerate_mask``.  No imputation is performed.
    """

    values = _as_five_states(states)
    geometry = _metric_name(metric)
    if not np.isfinite(float(zero_tolerance)) or float(zero_tolerance) < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    angles = np.full(3, np.nan, dtype=np.float64)
    cosines = np.full(3, np.nan, dtype=np.float64)
    norms = np.full((3, 2), np.nan, dtype=np.float64)
    degenerate = np.zeros(3, dtype=bool)
    logs = spd_log(values) if geometry == LE_METRIC else None
    for output_index, current_index in enumerate((1, 2, 3)):
        if geometry == AIRM_METRIC:
            incoming = -airm_whitened_log(
                values[current_index - 1], values[current_index]
            )
            outgoing = airm_whitened_log(
                values[current_index + 1], values[current_index]
            )
        else:
            assert logs is not None
            incoming = logs[current_index] - logs[current_index - 1]
            outgoing = logs[current_index + 1] - logs[current_index]
        incoming_norm = float(np.linalg.norm(incoming, ord="fro"))
        outgoing_norm = float(np.linalg.norm(outgoing, ord="fro"))
        norms[output_index] = (incoming_norm, outgoing_norm)
        if (
            incoming_norm <= float(zero_tolerance)
            or outgoing_norm <= float(zero_tolerance)
        ):
            degenerate[output_index] = True
            continue
        cosine = float(
            np.einsum("ij,ij->", incoming, outgoing)
            / (incoming_norm * outgoing_norm)
        )
        cosines[output_index] = cosine
        angles[output_index] = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return TurningAngleResult(
        angles=angles,
        raw_cosines=cosines,
        tangent_norms=norms,
        degenerate_mask=degenerate,
    )


def geodesic_deviations(states: np.ndarray, metric: str) -> np.ndarray:
    """Distances to fixed endpoint-geodesic times 1/4, 2/4, and 3/4."""

    values = _as_five_states(states)
    geometry = _metric_name(metric)
    result = np.empty(3, dtype=np.float64)
    if geometry == LE_METRIC:
        logs = spd_log(values)
        for output_index, current_index in enumerate((1, 2, 3)):
            t = current_index / 4.0
            geodesic_log = (1.0 - t) * logs[0] + t * logs[4]
            result[output_index] = np.linalg.norm(
                logs[current_index] - geodesic_log, ord="fro"
            )
    else:
        for output_index, current_index in enumerate((1, 2, 3)):
            geodesic = airm_geodesic(values[0], values[4], current_index / 4.0)
            result[output_index] = float(
                airm_distance(values[current_index], geodesic)
            )
    return result


@dataclass(frozen=True)
class TrajectoryGeometryResult:
    metric: Metric
    distance_matrix: np.ndarray
    path: np.ndarray
    bag_canon: CanonicalBag
    bag_sorted: np.ndarray
    steps: np.ndarray
    total_path_length: float
    endpoint_distance: float
    efficiency: float
    excess: float
    angles: np.ndarray
    angle_raw_cosines: np.ndarray
    angle_tangent_norms: np.ndarray
    angle_degenerate_mask: np.ndarray
    deviations: np.ndarray
    frechet_variance: float
    frechet_radius_mean: float
    diameter: float
    mean_result: GeometryMeanResult
    path_degenerate: bool

    @property
    def path_d10(self) -> np.ndarray:
        """Alias with the frozen representation name."""

        return self.path

    @property
    def bag_canon_d10(self) -> np.ndarray:
        """Canonical BAG vector without dropping its audit permutation."""

        return self.bag_canon.vector

    @property
    def bag_sorted_d10(self) -> np.ndarray:
        return self.bag_sorted

    @property
    def mean_matrix(self) -> np.ndarray:
        return self.mean_result.matrix

    @property
    def normalized_karcher_residual(self) -> float | None:
        return self.mean_result.normalized_karcher_residual

    @property
    def mean_turn(self) -> float:
        return float(np.mean(self.angles))

    @property
    def max_turn(self) -> float:
        return float(np.max(self.angles))

    @property
    def mean_geodesic_deviation(self) -> float:
        return float(np.mean(self.deviations))

    @property
    def max_geodesic_deviation(self) -> float:
        return float(np.max(self.deviations))

    @property
    def scalar_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.total_path_length,
                self.endpoint_distance,
                self.efficiency,
                self.excess,
                self.mean_turn,
                self.max_turn,
                self.mean_geodesic_deviation,
                self.max_geodesic_deviation,
                self.frechet_variance,
                self.frechet_radius_mean,
                self.diameter,
            ],
            dtype=np.float64,
        )

    @property
    def scalar_dict(self) -> dict[str, float]:
        return dict(zip(SCALAR_11_NAMES, self.scalar_vector.tolist(), strict=True))


def compute_five_state_geometry(
    states: np.ndarray,
    metric: str,
) -> TrajectoryGeometryResult:
    """Compute all frozen descriptors and intrinsic quantities for one trial."""

    values = _as_five_states(states)
    geometry = _metric_name(metric)
    matrix = distance_matrix(values, geometry)
    path = path_d10(matrix)
    canonical = bag_canon_d10(matrix)
    sorted_bag = bag_sorted_d10(matrix)
    steps = np.asarray([matrix[index, index + 1] for index in range(4)])
    total = float(np.sum(steps))
    endpoint = float(matrix[0, 4])
    path_degenerate = bool(total <= PATH_ZERO_TOLERANCE)
    efficiency = float(endpoint / total) if not path_degenerate else float("nan")
    excess = float(total - endpoint)
    turns = turning_angles(values, geometry)
    deviations = geodesic_deviations(values, geometry)
    mean_result = five_state_barycenter(values, geometry)
    if geometry == AIRM_METRIC:
        radii = np.asarray(
            [airm_distance(mean_result.matrix, state) for state in values],
            dtype=np.float64,
        )
    else:
        mean_log = spd_log(mean_result.matrix)
        radii = np.linalg.norm(spd_log(values) - mean_log, axis=(1, 2))
    return TrajectoryGeometryResult(
        metric=geometry,
        distance_matrix=matrix,
        path=path,
        bag_canon=canonical,
        bag_sorted=sorted_bag,
        steps=steps,
        total_path_length=total,
        endpoint_distance=endpoint,
        efficiency=efficiency,
        excess=excess,
        angles=turns.angles,
        angle_raw_cosines=turns.raw_cosines,
        angle_tangent_norms=turns.tangent_norms,
        angle_degenerate_mask=turns.degenerate_mask,
        deviations=deviations,
        frechet_variance=float(np.mean(radii**2)),
        frechet_radius_mean=float(np.mean(radii)),
        diameter=float(np.max(path)),
        mean_result=mean_result,
        path_degenerate=path_degenerate,
    )


@dataclass(frozen=True)
class SPDStackChecks:
    matrix_count: int
    nonfinite_count: int
    maximum_relative_symmetry_error: float
    minimum_eigenvalue: float
    maximum_condition_number: float
    passed: bool


def spd_stack_hard_checks(matrices: np.ndarray) -> SPDStackChecks:
    """Evaluate frozen finite/symmetry/PD/conditioning gates without repair."""

    value = np.asarray(matrices, dtype=np.float64)
    if value.ndim == 2:
        value = value[np.newaxis]
    if value.ndim != 3 or value.shape[1] != value.shape[2] or len(value) < 1:
        raise ValueError("matrices must have shape (n_matrices, channels, channels)")
    diagnostics = spd_diagnostics(value)
    nonfinite = int((~diagnostics["finite"]).sum())
    symmetry = float(np.max(diagnostics["symmetry_error"]))
    finite_eigenvalues = diagnostics["min_eigenvalue"][
        np.isfinite(diagnostics["min_eigenvalue"])
    ]
    minimum = (
        float(np.min(finite_eigenvalues)) if len(finite_eigenvalues) else float("nan")
    )
    condition = float(np.max(diagnostics["condition_number"]))
    passed = bool(
        nonfinite == 0
        and np.isfinite(symmetry)
        and symmetry <= SYMMETRY_TOLERANCE
        and np.isfinite(minimum)
        and minimum > 0.0
        and np.isfinite(condition)
        and condition <= CONDITION_NUMBER_MAX
    )
    return SPDStackChecks(
        matrix_count=len(value),
        nonfinite_count=nonfinite,
        maximum_relative_symmetry_error=symmetry,
        minimum_eigenvalue=minimum,
        maximum_condition_number=condition,
        passed=passed,
    )


@dataclass(frozen=True)
class DistanceMatrixChecks:
    finite: bool
    maximum_absolute_symmetry_error: float
    maximum_absolute_diagonal_error: float
    minimum_distance: float
    triangle_inequality_count: int
    maximum_triangle_excess: float
    maximum_triangle_excess_over_tolerance: float
    passed: bool


def distance_matrix_hard_checks(matrix: np.ndarray) -> DistanceMatrixChecks:
    """Evaluate every frozen D-matrix check, including all 30 triangles."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (N_STATES, N_STATES):
        raise ValueError(f"distance matrix must have shape (5, 5), got {value.shape}")
    finite = bool(np.isfinite(value).all())
    if finite:
        symmetry = float(np.max(np.abs(value - value.T)))
        diagonal = float(np.max(np.abs(np.diag(value))))
        minimum = float(np.min(value))
        excesses: list[float] = []
        margins: list[float] = []
        for first, second, third in combinations(range(N_STATES), 3):
            for left, middle, right in (
                (first, third, second),
                (first, second, third),
                (second, first, third),
            ):
                direct = float(value[left, right])
                two_edge = float(value[left, middle] + value[middle, right])
                excess = direct - two_edge
                tolerance = (
                    TRIANGLE_ABSOLUTE_TOLERANCE
                    + TRIANGLE_RELATIVE_TOLERANCE * abs(two_edge)
                )
                excesses.append(excess)
                margins.append(excess - tolerance)
        maximum_excess = float(max(excesses))
        maximum_margin = float(max(margins))
    else:
        symmetry = diagonal = minimum = maximum_excess = maximum_margin = float("inf")
    passed = bool(
        finite
        and symmetry <= DISTANCE_SYMMETRY_TOLERANCE
        and diagonal <= DISTANCE_DIAGONAL_TOLERANCE
        and minimum >= -DISTANCE_NONNEGATIVE_TOLERANCE
        and maximum_margin <= 0.0
    )
    return DistanceMatrixChecks(
        finite=finite,
        maximum_absolute_symmetry_error=symmetry,
        maximum_absolute_diagonal_error=diagonal,
        minimum_distance=minimum,
        triangle_inequality_count=30,
        maximum_triangle_excess=maximum_excess,
        maximum_triangle_excess_over_tolerance=maximum_margin,
        passed=passed,
    )


@dataclass(frozen=True)
class GeodesicEndpointChecks:
    metric: Metric
    t0_relative_error: float
    t1_relative_error: float
    passed: bool


def geodesic_endpoint_hard_checks(
    first: np.ndarray,
    second: np.ndarray,
    metric: str,
) -> GeodesicEndpointChecks:
    geometry = _metric_name(metric)
    first_value = _as_square(first, name="first")
    second_value = _as_square(second, name="second")
    function = airm_geodesic if geometry == AIRM_METRIC else le_geodesic
    at_zero = function(first_value, second_value, 0.0)
    at_one = function(first_value, second_value, 1.0)
    zero_error = _relative_frobenius(at_zero, first_value)
    one_error = _relative_frobenius(at_one, second_value)
    return GeodesicEndpointChecks(
        metric=geometry,
        t0_relative_error=zero_error,
        t1_relative_error=one_error,
        passed=bool(
            np.isfinite(zero_error)
            and np.isfinite(one_error)
            and zero_error <= GEODESIC_ENDPOINT_TOLERANCE
            and one_error <= GEODESIC_ENDPOINT_TOLERANCE
        ),
    )


@dataclass(frozen=True)
class IntrinsicChecks:
    finite_features: bool
    path_degenerate: bool
    path_inequality_excess_over_tolerance: float
    efficiency_lower_violation: float
    efficiency_upper_violation: float
    degenerate_angle_count: int
    maximum_cosine_domain_excess: float
    minimum_angle: float
    maximum_angle: float
    minimum_deviation: float
    passed: bool


def intrinsic_hard_checks(result: TrajectoryGeometryResult) -> IntrinsicChecks:
    """Evaluate the frozen length, efficiency, angle, and deviation gates."""

    arrays = (
        result.distance_matrix,
        result.path,
        result.bag_canon.vector,
        result.bag_sorted,
        result.scalar_vector,
        result.steps,
        result.angles,
        result.deviations,
    )
    finite = bool(all(np.isfinite(value).all() for value in arrays))
    tolerance = PATH_INEQUALITY_ABSOLUTE_TOLERANCE + (
        PATH_INEQUALITY_RELATIVE_TOLERANCE * abs(result.total_path_length)
    )
    path_margin = float(
        result.endpoint_distance - result.total_path_length - tolerance
    )
    lower_efficiency = float(-result.efficiency) if np.isfinite(result.efficiency) else float("inf")
    upper_efficiency = (
        float(result.efficiency - 1.0)
        if np.isfinite(result.efficiency)
        else float("inf")
    )
    degenerate_count = int(np.sum(result.angle_degenerate_mask))
    finite_cosines = result.angle_raw_cosines[np.isfinite(result.angle_raw_cosines)]
    cosine_excess = (
        float(np.max(np.maximum(np.abs(finite_cosines) - 1.0, 0.0)))
        if len(finite_cosines)
        else float("inf")
    )
    minimum_angle = (
        float(np.nanmin(result.angles))
        if np.isfinite(result.angles).any()
        else float("nan")
    )
    maximum_angle = (
        float(np.nanmax(result.angles))
        if np.isfinite(result.angles).any()
        else float("nan")
    )
    minimum_deviation = (
        float(np.min(result.deviations))
        if np.isfinite(result.deviations).all()
        else float("nan")
    )
    passed = bool(
        finite
        and not result.path_degenerate
        and path_margin <= 0.0
        and result.efficiency >= -BOUND_TOLERANCE
        and result.efficiency <= 1.0 + BOUND_TOLERANCE
        and degenerate_count == 0
        and cosine_excess <= COSINE_DOMAIN_TOLERANCE
        and minimum_angle >= -BOUND_TOLERANCE
        and maximum_angle <= np.pi + BOUND_TOLERANCE
        and minimum_deviation >= -BOUND_TOLERANCE
    )
    return IntrinsicChecks(
        finite_features=finite,
        path_degenerate=result.path_degenerate,
        path_inequality_excess_over_tolerance=path_margin,
        efficiency_lower_violation=lower_efficiency,
        efficiency_upper_violation=upper_efficiency,
        degenerate_angle_count=degenerate_count,
        maximum_cosine_domain_excess=cosine_excess,
        minimum_angle=minimum_angle,
        maximum_angle=maximum_angle,
        minimum_deviation=minimum_deviation,
        passed=passed,
    )


@dataclass(frozen=True)
class TrajectoryHardChecks:
    metric: Metric
    states: SPDStackChecks
    distance: DistanceMatrixChecks
    mean: SPDStackChecks
    endpoints: GeodesicEndpointChecks
    intrinsic: IntrinsicChecks
    mean_warning_count: int
    normalized_karcher_residual: float | None
    mean_solver_passed: bool
    passed: bool


def trajectory_hard_checks(
    states: np.ndarray,
    result: TrajectoryGeometryResult | None = None,
    *,
    metric: str = AIRM_METRIC,
) -> TrajectoryHardChecks:
    """Run every geometry-only hard gate for one five-state trajectory."""

    values = _as_five_states(states)
    computed = result or compute_five_state_geometry(values, metric)
    if computed.metric not in METRICS:
        raise ValueError("result has an unsupported metric")
    state_checks = spd_stack_hard_checks(values)
    distance_checks = distance_matrix_hard_checks(computed.distance_matrix)
    mean_checks = spd_stack_hard_checks(computed.mean_result.matrix)
    endpoint_checks = geodesic_endpoint_hard_checks(
        values[0], values[4], computed.metric
    )
    intrinsic_checks = intrinsic_hard_checks(computed)
    warning_count = len(computed.mean_result.warning_messages)
    if computed.metric == AIRM_METRIC:
        residual = computed.mean_result.normalized_karcher_residual
        mean_solver_passed = bool(
            warning_count == 0
            and residual is not None
            and np.isfinite(residual)
            and residual <= KARCHER_RESIDUAL_MAX
            and computed.mean_result.solver_tol == AIRM_TOLERANCE
            and computed.mean_result.solver_maxiter == AIRM_MAXITER
        )
    else:
        residual = None
        mean_solver_passed = bool(
            warning_count == 0
            and computed.mean_result.normalized_log_residual is not None
            and np.isfinite(computed.mean_result.normalized_log_residual)
        )
    passed = bool(
        state_checks.passed
        and distance_checks.passed
        and mean_checks.passed
        and endpoint_checks.passed
        and intrinsic_checks.passed
        and mean_solver_passed
    )
    return TrajectoryHardChecks(
        metric=computed.metric,
        states=state_checks,
        distance=distance_checks,
        mean=mean_checks,
        endpoints=endpoint_checks,
        intrinsic=intrinsic_checks,
        mean_warning_count=warning_count,
        normalized_karcher_residual=residual,
        mean_solver_passed=mean_solver_passed,
        passed=passed,
    )


@dataclass(frozen=True)
class PermutationInvarianceChecks:
    permutation_count: int
    maximum_absolute_error: float
    exact_equal: bool
    passed: bool


def check_bag_permutation_invariance(
    matrix: np.ndarray,
) -> PermutationInvarianceChecks:
    """Rebuild canonical BAG after every one of all 120 S5 permutations."""

    value = _as_distance_matrix(matrix)
    reference = bag_canon_d10(value).vector
    maximum = 0.0
    exact = True
    for permutation in ALL_PERMUTATIONS_5:
        observed = bag_canon_d10(
            permute_distance_matrix(value, permutation)
        ).vector
        maximum = max(maximum, float(np.max(np.abs(observed - reference))))
        exact = exact and bool(np.array_equal(observed, reference))
    return PermutationInvarianceChecks(
        permutation_count=len(ALL_PERMUTATIONS_5),
        maximum_absolute_error=maximum,
        exact_equal=exact,
        passed=bool(maximum <= BAG_INVARIANCE_TOLERANCE),
    )


@dataclass(frozen=True)
class CenteringIsometryChecks:
    distance_maximum_absolute_error: float
    distance_maximum_relative_error: float
    path_maximum_absolute_error: float
    path_maximum_relative_error: float
    bag_maximum_absolute_error: float
    bag_maximum_relative_error: float
    path_length_absolute_error: float
    path_length_relative_error: float
    passed: bool


def compare_airm_centering_isometry(
    states: np.ndarray,
    subject_whole_airm_mean: np.ndarray,
) -> CenteringIsometryChecks:
    """Compare raw versus WHOLE-mean-congruence local AIRM descriptors."""

    values = _as_five_states(states)
    mean = _as_square(subject_whole_airm_mean, name="subject_whole_airm_mean")
    if mean.shape != values.shape[1:]:
        raise ValueError("subject mean and local-state channels differ")
    whitening = spd_invsqrt(mean)
    centered = symmetrize(whitening @ values @ whitening)
    raw_distance = distance_matrix(values, AIRM_METRIC)
    centered_distance = distance_matrix(centered, AIRM_METRIC)
    raw_path = path_d10(raw_distance)
    centered_path = path_d10(centered_distance)
    raw_bag = bag_canon_d10(raw_distance).vector
    centered_bag = bag_canon_d10(centered_distance).vector
    raw_length = float(sum(raw_distance[index, index + 1] for index in range(4)))
    centered_length = float(
        sum(centered_distance[index, index + 1] for index in range(4))
    )
    distance_absolute, distance_relative = _max_absolute_relative(
        centered_distance, raw_distance
    )
    path_absolute, path_relative = _max_absolute_relative(centered_path, raw_path)
    bag_absolute, bag_relative = _max_absolute_relative(centered_bag, raw_bag)
    length_absolute, length_relative = _max_absolute_relative(
        centered_length, raw_length
    )
    passed = bool(
        distance_absolute <= ISOMETRY_TOLERANCE
        and distance_relative <= ISOMETRY_TOLERANCE
        and path_absolute <= ISOMETRY_TOLERANCE
        and path_relative <= ISOMETRY_TOLERANCE
        and bag_absolute <= ISOMETRY_TOLERANCE
        and bag_relative <= ISOMETRY_TOLERANCE
        and length_absolute <= ISOMETRY_TOLERANCE
        and length_relative <= ISOMETRY_TOLERANCE
    )
    return CenteringIsometryChecks(
        distance_maximum_absolute_error=distance_absolute,
        distance_maximum_relative_error=distance_relative,
        path_maximum_absolute_error=path_absolute,
        path_maximum_relative_error=path_relative,
        bag_maximum_absolute_error=bag_absolute,
        bag_maximum_relative_error=bag_relative,
        path_length_absolute_error=length_absolute,
        path_length_relative_error=length_relative,
        passed=passed,
    )


@dataclass(frozen=True)
class FactorialDecomposition:
    n_subjects: int
    n_classes: int
    cell_count: int
    grand_mean: float
    ss_subject: float
    ss_class: float
    ss_subject_class: float
    ss_residual: float
    ss_total: float
    eta2_subject: float
    eta2_class: float
    eta2_subject_class: float
    eta2_residual: float
    closure_relative_error: float
    degenerate: bool
    passed: bool


def balanced_factorial_decomposition(
    values: np.ndarray,
    subjects: np.ndarray,
    classes: np.ndarray,
    *,
    expected_subjects: int = 9,
    expected_classes: int = 4,
    expected_cell_count: int = 72,
) -> FactorialDecomposition:
    """Compute frozen orthogonal SS and ordinary eta-squared components."""

    observations = np.asarray(values, dtype=np.float64)
    subject_values = np.asarray(subjects)
    class_values = np.asarray(classes)
    if observations.ndim != 1 or len(observations) < 1:
        raise ValueError("values must be a non-empty one-dimensional array")
    if subject_values.ndim != 1 or class_values.ndim != 1:
        raise ValueError("subjects and classes must be one-dimensional")
    if not (len(observations) == len(subject_values) == len(class_values)):
        raise ValueError("values, subjects, and classes must have equal length")
    if not np.isfinite(observations).all():
        raise ValueError("values contain NaN or Inf")
    unique_subjects = np.unique(subject_values)
    unique_classes = np.unique(class_values)
    if len(unique_subjects) != int(expected_subjects):
        raise ValueError(
            f"expected {expected_subjects} subjects, observed {len(unique_subjects)}"
        )
    if len(unique_classes) != int(expected_classes):
        raise ValueError(
            f"expected {expected_classes} classes, observed {len(unique_classes)}"
        )
    cell_means = np.empty((len(unique_subjects), len(unique_classes)), dtype=float)
    for subject_index, subject in enumerate(unique_subjects):
        for class_index, class_label in enumerate(unique_classes):
            mask = (subject_values == subject) & (class_values == class_label)
            count = int(mask.sum())
            if count != int(expected_cell_count):
                raise ValueError(
                    "unbalanced subject x class cell: "
                    f"subject={subject!r}, class={class_label!r}, "
                    f"expected={expected_cell_count}, observed={count}"
                )
            cell_means[subject_index, class_index] = float(np.mean(observations[mask]))
    grand = float(np.mean(observations))
    subject_means = cell_means.mean(axis=1)
    class_means = cell_means.mean(axis=0)
    n_subjects = len(unique_subjects)
    n_classes = len(unique_classes)
    repetitions = int(expected_cell_count)
    ss_subject = float(
        n_classes * repetitions * np.sum((subject_means - grand) ** 2)
    )
    ss_class = float(
        n_subjects * repetitions * np.sum((class_means - grand) ** 2)
    )
    interaction = (
        cell_means - subject_means[:, None] - class_means[None, :] + grand
    )
    ss_interaction = float(repetitions * np.sum(interaction**2))
    ss_residual = 0.0
    for subject_index, subject in enumerate(unique_subjects):
        for class_index, class_label in enumerate(unique_classes):
            mask = (subject_values == subject) & (class_values == class_label)
            ss_residual += float(
                np.sum((observations[mask] - cell_means[subject_index, class_index]) ** 2)
            )
    ss_total = float(np.sum((observations - grand) ** 2))
    degenerate = bool(ss_total == 0.0)
    if degenerate:
        eta_subject = eta_class = eta_interaction = eta_residual = float("nan")
        closure = float("nan")
        passed = False
    else:
        eta_subject = ss_subject / ss_total
        eta_class = ss_class / ss_total
        eta_interaction = ss_interaction / ss_total
        eta_residual = ss_residual / ss_total
        closure = abs(
            ss_total
            - (ss_subject + ss_class + ss_interaction + ss_residual)
        ) / ss_total
        passed = bool(np.isfinite(closure) and closure <= SS_CLOSURE_TOLERANCE)
    return FactorialDecomposition(
        n_subjects=n_subjects,
        n_classes=n_classes,
        cell_count=repetitions,
        grand_mean=grand,
        ss_subject=ss_subject,
        ss_class=ss_class,
        ss_subject_class=ss_interaction,
        ss_residual=ss_residual,
        ss_total=ss_total,
        eta2_subject=eta_subject,
        eta2_class=eta_class,
        eta2_subject_class=eta_interaction,
        eta2_residual=eta_residual,
        closure_relative_error=closure,
        degenerate=degenerate,
        passed=passed,
    )
