"""Scientific discovery orchestration for frozen Trajectory Anatomy v0.

This stage consumes only the PASS-gated feature archive produced by stage 20.
It cannot download EEG, open session 1test, rebuild geometry, or write outside
the exact trajectory-v0 tables/nulls namespaces.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from src.data_trajectory_v0 import load_trajectory_config
from src.evaluation_trajectory_v0 import (
    CLASS_LOSO_COLUMNS,
    FACTOR_COLUMNS,
    LABEL_GROUP_COLUMNS,
    MDM_COLUMNS,
    NULL_SUBJECT_COLUMNS,
    ORDER_GROUP_COLUMNS,
    SUBJECT_PROBE_COLUMNS,
    EDGE_REINDEX_TABLE,
    LabelPermutationPlan,
    NullSummary,
    OrderPermutationPlan,
    array_audit_sha256,
    balanced_factor_decomposition,
    make_label_permutation_plan,
    make_order_permutation_plan,
    run_class_loso,
    run_label_destruction_null,
    run_mdm_loso,
    run_order_shuffle_null,
    run_subject_runhalf_probe,
    stable_json_sha256,
    summarize_null_distribution,
    trial_uid_sha256,
)
from src.trajectory_geometry_v0 import (
    ALL_PERMUTATIONS_5,
    PATH_D10_NAMES,
    SCALAR_11_NAMES,
    bag_canon_d10,
)


DISCOVERY_TABLE_NAMES = (
    "class_loso_metrics",
    "subject_runhalf_probe",
    "scalar_factor_decomposition",
    "order_shuffle_subject_metrics",
    "order_shuffle_group_metrics",
    "label_null_subject_metrics",
    "label_null_group_metrics",
    "local_barycenter_mdm",
    "whole_context_mdm",
    "airm_le_robustness",
)
NULL_ARTIFACT_NAMES = (
    "order_shuffle_seeds.json",
    "label_permutation_seeds.json",
    "order_shuffle_group_stats.npz",
    "label_null_group_stats.npz",
)
COMMON_COLUMNS = (
    "protocol_version",
    "protocol_sha256",
    "config_sha256",
    "seed",
    "session",
    "generated_at_utc",
    "status",
)
ROBUSTNESS_COLUMNS = COMMON_COLUMNS + (
    "analysis",
    "representation",
    "subject",
    "scalar",
    "airm_value",
    "le_value",
    "paired_delta",
    "delta_category",
    "airm_status",
    "le_status",
    "agreement_category",
    "interpretation",
)
WHOLE_MDM_COLUMNS = MDM_COLUMNS + (
    "covariance_samples_per_estimate",
    "estimator_regime_confounded",
    "interpretation_limit",
)

FEATURE_NPZ_KEYS = (
    "airm_distance_matrices",
    "le_distance_matrices",
    "airm_path_d10",
    "le_path_d10",
    "airm_bag_canon_d10",
    "le_bag_canon_d10",
    "airm_bag_sorted_d10",
    "le_bag_sorted_d10",
    "airm_scalars_11",
    "le_scalars_11",
    "airm_canonical_permutation",
    "le_canonical_permutation",
    "local_airm_barycenters",
    "whole_covariances",
    "sample_index",
    "subject",
    "run",
    "trial_id",
    "trial_uid",
    "class_label",
    "airm_trial_gate_pass",
    "le_trial_gate_pass",
    "path_d10_names",
    "scalar_11_names",
    "protocol_version",
    "protocol_sha256",
    "config_sha256",
    "session",
    "seed",
    "generated_at_utc",
    "geometry_gate_passed",
)


class DiscoveryStructuralError(RuntimeError):
    """Raised before writing when a gate/cache/schema/leakage contract fails."""


@dataclass(frozen=True)
class DiscoveryInputs:
    metadata: pd.DataFrame
    arrays: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any]
    gate: Mapping[str, Any]
    feature_npz_path: Path
    feature_npz_sha256: str


@dataclass(frozen=True)
class DiscoveryArtifacts:
    tables: Mapping[str, pd.DataFrame]
    order_plan: OrderPermutationPlan
    label_plan: LabelPermutationPlan
    order_group_statistics: Mapping[str, np.ndarray]
    label_group_statistics: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any]
    technical_failure_count: int
    status: str
    bag_plan_audit_sha256: str


def _fail(message: str) -> None:
    raise DiscoveryStructuralError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as error:
        raise DiscoveryStructuralError(f"required discovery input is missing: {path}") from error
    return digest.hexdigest()


def _canonical_config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _config_identity(
    config_or_path: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], str]:
    if isinstance(config_or_path, (str, Path)):
        path = Path(config_or_path).expanduser().resolve()
        return load_trajectory_config(path), _file_sha256(path)
    config = copy.deepcopy(dict(config_or_path))
    return config, _canonical_config_sha256(config)


def _scalar_text(array: np.ndarray, *, name: str) -> str:
    value = np.asarray(array)
    if value.shape != (1,):
        _fail(f"feature cache provenance {name} must have shape (1,)")
    return str(value[0])


def _scalar_int(array: np.ndarray, *, name: str) -> int:
    value = np.asarray(array)
    if value.shape != (1,) or not np.issubdtype(value.dtype, np.integer):
        _fail(f"feature cache provenance {name} must be an integer shape-(1,) array")
    return int(value[0])


def _read_only(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _validate_gate(
    gate: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
) -> None:
    exact = {
        "protocol_version": str(config["project"]["protocol_version"]),
        "protocol_sha256": str(config["project"]["protocol_sha256"]),
        "config_sha256": config_sha256,
        "seed": int(config["project"]["seed"]),
        "session": str(config["dataset"]["allowed_session"]),
        "status": "PASS",
    }
    for key, expected in exact.items():
        if gate.get(key) != expected:
            _fail(f"trajectory geometry gate {key} mismatch: {gate.get(key)!r} != {expected!r}")
    for key in ("gate_passed", "scientific_classification_allowed", "feature_npz_written"):
        if gate.get(key) is not True:
            _fail(f"trajectory geometry gate requires exact boolean true for {key}")
    failure_counts = gate.get("required_failure_counts")
    expected_failure_keys = {
        "dataset_contract",
        "covariance_sanity",
        "trajectory_geometry_correctness",
    }
    if (
        not isinstance(failure_counts, Mapping)
        or set(failure_counts) != expected_failure_keys
        or any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) != 0
            for value in failure_counts.values()
        )
    ):
        _fail("trajectory geometry gate contains a required failure")
    if not isinstance(gate.get("feature_npz_sha256"), str) or len(
        gate["feature_npz_sha256"]
    ) != 64:
        _fail("trajectory geometry gate has no valid feature_npz_sha256")
    if not isinstance(gate.get("feature_npz_path"), str) or not gate[
        "feature_npz_path"
    ]:
        _fail("trajectory geometry gate has no feature_npz_path")


def _validate_metadata(metadata: pd.DataFrame, config: Mapping[str, Any]) -> None:
    required = {
        "sample_index",
        "subject",
        "session",
        "run",
        "trial_id",
        "trial_uid",
        "class_label",
    }
    if required - set(metadata) or metadata.empty or metadata[list(required)].isna().any(axis=None):
        _fail("feature-cache trial metadata is incomplete or null")
    dataset = config["dataset"]
    n_trials = int(dataset["expected_trials"])
    if len(metadata) != n_trials:
        _fail(f"feature-cache trial count mismatch: {len(metadata)} != {n_trials}")
    if tuple(sorted(metadata["session"].astype(str).unique())) != (
        str(dataset["allowed_session"]),
    ):
        _fail("feature cache crossed the forbidden-session barrier")
    if not np.array_equal(metadata["sample_index"].to_numpy(), np.arange(n_trials)):
        _fail("feature-cache sample_index is not exactly 0..N-1")
    if metadata["sample_index"].duplicated().any() or metadata["trial_uid"].duplicated().any():
        _fail("feature-cache sample_index/trial_uid is not globally unique")
    subjects = tuple(int(value) for value in dataset["subjects"])
    runs = tuple(int(value) for value in dataset["runs"])
    classes = tuple(str(value) for value in dataset["classes"])
    if tuple(sorted(metadata["subject"].unique())) != subjects:
        _fail("feature-cache subject set mismatch")
    if tuple(sorted(metadata["run"].unique())) != runs:
        _fail("feature-cache run set mismatch")
    if set(metadata["class_label"].astype(str).unique()) != set(classes):
        _fail("feature-cache class vocabulary mismatch")
    checks = (
        (["subject"], int(dataset["expected_trials_per_subject"])),
        (["subject", "class_label"], int(dataset["expected_trials_per_subject_class"])),
        (["subject", "run"], int(dataset["expected_trials_per_subject_run"])),
        (
            ["subject", "run", "class_label"],
            int(dataset["expected_trials_per_subject_run_class"]),
        ),
    )
    for columns, expected in checks:
        counts = metadata.groupby(columns, sort=True, observed=True).size()
        if len(counts) == 0 or not (counts == expected).all():
            _fail(f"feature-cache balanced count failure for {columns}")
    expected_uid_hash = dataset.get("expected_trial_uid_set_sha256")
    if expected_uid_hash is not None:
        observed = trial_uid_sha256(metadata["trial_uid"].astype(str).tolist())
        if observed != str(expected_uid_hash):
            _fail("feature-cache trial UID-set hash mismatch")


def _validate_feature_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    n_trials = len(metadata)
    n_channels = len(config["dataset"]["eeg_channels"])
    for name in ("sample_index", "subject", "run", "trial_id"):
        value = arrays[name]
        if value.shape != (n_trials,) or value.dtype != np.dtype("int64"):
            _fail(f"feature-cache {name} must have exact int64 shape ({n_trials},)")
    for name in ("trial_uid", "class_label"):
        value = arrays[name]
        if value.shape != (n_trials,) or value.dtype.kind != "U":
            _fail(f"feature-cache {name} must have exact Unicode shape ({n_trials},)")
    shapes = {
        "airm_distance_matrices": (n_trials, 5, 5),
        "le_distance_matrices": (n_trials, 5, 5),
        "airm_path_d10": (n_trials, 10),
        "le_path_d10": (n_trials, 10),
        "airm_bag_canon_d10": (n_trials, 10),
        "le_bag_canon_d10": (n_trials, 10),
        "airm_bag_sorted_d10": (n_trials, 10),
        "le_bag_sorted_d10": (n_trials, 10),
        "airm_scalars_11": (n_trials, 11),
        "le_scalars_11": (n_trials, 11),
        "airm_canonical_permutation": (n_trials, 5),
        "le_canonical_permutation": (n_trials, 5),
        "local_airm_barycenters": (n_trials, n_channels, n_channels),
        "whole_covariances": (n_trials, n_channels, n_channels),
    }
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape:
            _fail(f"feature-cache {name} shape mismatch: {value.shape} != {shape}")
        if name.endswith("canonical_permutation"):
            if not np.issubdtype(value.dtype, np.integer):
                _fail(f"feature-cache {name} is not integer")
        elif value.dtype != np.dtype("float64") or not np.isfinite(value).all():
            _fail(f"feature-cache {name} must be finite float64")
    for name in ("airm_trial_gate_pass", "le_trial_gate_pass"):
        value = arrays[name]
        if value.shape != (n_trials,) or value.dtype != np.dtype(bool) or not value.all():
            _fail(f"feature-cache {name} did not PASS for every trial")
    for prefix in ("airm", "le"):
        distances = arrays[f"{prefix}_distance_matrices"]
        if not np.allclose(
            distances, distances.transpose(0, 2, 1), rtol=0.0, atol=1e-12
        ):
            _fail(f"feature-cache {prefix} distance matrices are not symmetric")
        diagonals = np.diagonal(distances, axis1=1, axis2=2)
        if np.max(np.abs(diagonals)) > 1e-12 or np.min(distances) < -1e-12:
            _fail(f"feature-cache {prefix} distances violate diagonal/nonnegative gates")
        permutations = arrays[f"{prefix}_canonical_permutation"]
        if not all(
            np.array_equal(np.sort(row), np.arange(5, dtype=row.dtype))
            for row in permutations
        ):
            _fail(f"feature-cache {prefix} canonical permutations are invalid")
    if tuple(arrays["path_d10_names"].astype(str)) != tuple(PATH_D10_NAMES):
        _fail("feature-cache PATH_D10 names/order mismatch")
    if tuple(arrays["scalar_11_names"].astype(str)) != tuple(SCALAR_11_NAMES):
        _fail("feature-cache SCALARS_11 names/order mismatch")
    for name in ("local_airm_barycenters", "whole_covariances"):
        value = arrays[name]
        if not np.allclose(value, value.transpose(0, 2, 1), rtol=0.0, atol=1e-12):
            _fail(f"feature-cache {name} contains a non-symmetric matrix")
        eigenvalues = np.linalg.eigvalsh(value)
        if np.any(eigenvalues[:, 0] <= 0.0):
            _fail(f"feature-cache {name} contains a non-SPD matrix")


def load_discovery_inputs(
    config_or_path: Mapping[str, Any] | str | Path,
    root: str | Path,
) -> tuple[dict[str, Any], DiscoveryInputs]:
    """Load only the exact PASS-gated local feature archive."""

    config, config_sha256 = _config_identity(config_or_path)
    project_root = Path(root).expanduser().resolve()
    if str(config["project"]["output_dir"]) != "outputs/bnci2014_001_trajectory_v0":
        _fail("trajectory discovery output_dir left the frozen namespace")
    if str(config["project"]["local_cache_dir"]) != "cache/bnci2014_001_trajectory_v0":
        _fail("trajectory discovery cache_dir left the frozen namespace")
    gate_path = (
        project_root
        / str(config["project"]["output_dir"])
        / "tables"
        / "trajectory_geometry_gate.json"
    ).resolve()
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise DiscoveryStructuralError(f"cannot load trajectory geometry gate: {gate_path}") from error
    if not isinstance(gate, Mapping):
        _fail("trajectory geometry gate must be a JSON object")
    _validate_gate(gate, config, config_sha256)
    feature_path = (
        project_root
        / str(config["project"]["local_cache_dir"])
        / "trajectory_features_v0.npz"
    ).resolve()
    declared_path = _resolve(project_root, str(gate["feature_npz_path"]))
    if declared_path != feature_path:
        _fail(f"gate feature path left exact cache location: {declared_path}")
    observed_file_hash = _file_sha256(feature_path)
    if observed_file_hash != str(gate["feature_npz_sha256"]):
        _fail("feature NPZ SHA-256 differs from the PASS gate")

    try:
        with np.load(feature_path, allow_pickle=False) as archive:
            if tuple(archive.files) != FEATURE_NPZ_KEYS:
                _fail(
                    "feature NPZ key/order mismatch: "
                    f"expected {FEATURE_NPZ_KEYS}, observed {tuple(archive.files)}"
                )
            # Session/provenance are inspected before any scientific numeric array.
            cached_session = _scalar_text(archive["session"], name="session")
            if cached_session != "0train" or cached_session != str(
                config["dataset"]["allowed_session"]
            ):
                _fail("feature NPZ crossed the forbidden-session barrier")
            cached_protocol_version = _scalar_text(
                archive["protocol_version"], name="protocol_version"
            )
            cached_protocol_hash = _scalar_text(
                archive["protocol_sha256"], name="protocol_sha256"
            )
            cached_config_hash = _scalar_text(
                archive["config_sha256"], name="config_sha256"
            )
            cached_seed = _scalar_int(archive["seed"], name="seed")
            gate_array = np.asarray(archive["geometry_gate_passed"])
            if (
                gate_array.shape != (1,)
                or gate_array.dtype != np.dtype(bool)
                or not bool(gate_array[0])
            ):
                _fail("feature NPZ geometry_gate_passed is not exact true")
            expected_provenance = (
                str(config["project"]["protocol_version"]),
                str(config["project"]["protocol_sha256"]),
                config_sha256,
                int(config["project"]["seed"]),
            )
            observed_provenance = (
                cached_protocol_version,
                cached_protocol_hash,
                cached_config_hash,
                cached_seed,
            )
            if observed_provenance != expected_provenance:
                _fail("feature NPZ protocol/config/seed provenance mismatch")
            arrays = {name: _read_only(np.asarray(archive[name])) for name in archive.files}
    except (OSError, ValueError) as error:
        raise DiscoveryStructuralError(f"cannot safely load feature NPZ: {feature_path}") from error

    metadata = pd.DataFrame(
        {
            "sample_index": arrays["sample_index"].astype(np.int64),
            "subject": arrays["subject"].astype(np.int64),
            "session": np.repeat(cached_session, len(arrays["sample_index"])),
            "run": arrays["run"].astype(np.int64),
            "trial_id": arrays["trial_id"].astype(np.int64),
            "trial_uid": arrays["trial_uid"].astype(str),
            "class_label": arrays["class_label"].astype(str),
        }
    )
    _validate_metadata(metadata, config)
    _validate_feature_arrays(arrays, metadata, config)
    provenance = {
        "protocol_version": cached_protocol_version,
        "protocol_sha256": cached_protocol_hash,
        "config_sha256": cached_config_hash,
        "seed": cached_seed,
        "session": cached_session,
        "feature_npz_sha256": observed_file_hash,
        "geometry_gate_sha256": _file_sha256(gate_path),
    }
    return config, DiscoveryInputs(
        metadata=metadata,
        arrays=arrays,
        provenance=provenance,
        gate=dict(gate),
        feature_npz_path=feature_path,
        feature_npz_sha256=observed_file_hash,
    )


def _assert_order_and_bag_contract(
    arrays: Mapping[str, np.ndarray],
    order_plan: OrderPermutationPlan,
    *,
    tolerance: float,
) -> str:
    """Exhaustively prove nonidentity draws and BAG orbit invariance."""

    choices = order_plan.permutation_indices
    if choices.size == 0 or np.any((choices < 1) | (choices > 119)):
        _fail("order plan contains identity or an out-of-range S5 index")
    if tuple(ALL_PERMUTATIONS_5[0]) != (0, 1, 2, 3, 4) or len(
        ALL_PERMUTATIONS_5
    ) != 120:
        _fail("geometry S5 table is not the frozen lexicographic 120-permutation set")

    pairs = (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    )
    edge = np.full((5, 5), -1, dtype=np.int8)
    for index, (first, second) in enumerate(pairs):
        edge[first, second] = edge[second, first] = index
    expected_maps = np.asarray(
        [
            [edge[permutation[first], permutation[second]] for first, second in pairs]
            for permutation in ALL_PERMUTATIONS_5
        ],
        dtype=np.int8,
    )
    if not np.array_equal(expected_maps, EDGE_REINDEX_TABLE):
        _fail("evaluation and geometry S5 edge actions disagree")
    if len({tuple(row) for row in expected_maps.tolist()}) != 120:
        _fail("S5 edge action is not faithful/complete")
    permutation_set = set(ALL_PERMUTATIONS_5)
    for first in ALL_PERMUTATIONS_5:
        for second in ALL_PERMUTATIONS_5:
            composed = tuple(first[second[index]] for index in range(5))
            if composed not in permutation_set:
                _fail("S5 action is not closed under composition")

    bag_hashes: dict[str, str] = {}
    for prefix in ("airm", "le"):
        distances = arrays[f"{prefix}_distance_matrices"]
        cached = arrays[f"{prefix}_bag_canon_d10"]
        rebuilt = np.empty_like(cached)
        rebuilt_permutations = np.empty_like(
            arrays[f"{prefix}_canonical_permutation"]
        )
        for index in range(len(distances)):
            canonical = bag_canon_d10(distances[index])
            rebuilt[index] = canonical.vector
            rebuilt_permutations[index] = canonical.permutation
        difference = np.abs(rebuilt - cached)
        if not np.isfinite(difference).all() or float(difference.max()) > tolerance:
            _fail(
                f"{prefix.upper()} cached BAG canonical descriptor is not invariant; "
                f"maximum error={float(difference.max()):.6g}"
            )
        if not np.array_equal(
            rebuilt_permutations, arrays[f"{prefix}_canonical_permutation"]
        ):
            _fail(f"{prefix.upper()} cached BAG canonical permutation changed")
        rebuilt_path = np.stack(
            [distances[:, first, second] for first, second in pairs], axis=1
        )
        if not np.allclose(
            rebuilt_path,
            arrays[f"{prefix}_path_d10"],
            rtol=0.0,
            atol=tolerance,
        ):
            _fail(f"{prefix.upper()} cached PATH_D10 is not distance-matrix aligned")
        if not np.allclose(
            np.sort(rebuilt_path, axis=1, kind="stable"),
            arrays[f"{prefix}_bag_sorted_d10"],
            rtol=0.0,
            atol=tolerance,
        ):
            _fail(f"{prefix.upper()} cached BAG_SORTED_D10 is not edge-multiset aligned")
        bag_hashes[prefix] = array_audit_sha256(rebuilt)
    return stable_json_sha256(
        {
            "order_plan": order_plan.audit_sha256,
            "nonidentity_min": int(choices.min()),
            "nonidentity_max": int(choices.max()),
            "draw_count": int(choices.size),
            "s5_edge_action": array_audit_sha256(expected_maps),
            "bag_rebuilt_hashes": bag_hashes,
            "tolerance": float(tolerance),
        }
    )


def _condition_rows(
    arrays: Mapping[str, np.ndarray],
) -> tuple[tuple[str, str, np.ndarray], ...]:
    return (
        ("AIRM", "PATH_D10", arrays["airm_path_d10"]),
        ("AIRM", "BAG_CANON_D10", arrays["airm_bag_canon_d10"]),
        ("AIRM", "BAG_SORTED_D10", arrays["airm_bag_sorted_d10"]),
        ("AIRM", "SCALARS_11", arrays["airm_scalars_11"]),
        ("LE", "PATH_D10", arrays["le_path_d10"]),
        ("LE", "BAG_CANON_D10", arrays["le_bag_canon_d10"]),
        ("LE", "SCALARS_11", arrays["le_scalars_11"]),
    )


def _status_is_pass(value: Any) -> bool:
    return str(value) == "PASS"


def _observed_condition_pass(
    table: pd.DataFrame, geometry: str, representation: str, subjects: Sequence[int]
) -> bool:
    selected = table[
        (table["geometry"] == geometry) & (table["representation"] == representation)
    ]
    return bool(
        len(selected) == len(subjects)
        and set(pd.to_numeric(selected["target_subject"]).astype(int)) == set(subjects)
        and selected["status"].map(_status_is_pass).all()
        and selected["balanced_accuracy"].notna().all()
    )


def _paired_path_minus_bag(
    observed: pd.DataFrame, subjects: Sequence[int], *, geometry: str
) -> float:
    path = observed[
        (observed["geometry"] == str(geometry))
        & (observed["representation"] == "PATH_D10")
    ][["target_subject", "balanced_accuracy", "status"]].rename(
        columns={"balanced_accuracy": "path"}
    )
    bag = observed[
        (observed["geometry"] == str(geometry))
        & (observed["representation"] == "BAG_CANON_D10")
    ][["target_subject", "balanced_accuracy", "status"]].rename(
        columns={"balanced_accuracy": "bag"}
    )
    merged = path.merge(bag, on="target_subject", validate="one_to_one", suffixes=("_p", "_b"))
    if (
        len(merged) != len(subjects)
        or not merged["status_p"].map(_status_is_pass).all()
        or not merged["status_b"].map(_status_is_pass).all()
        or merged[["path", "bag"]].isna().any(axis=None)
    ):
        return float("nan")
    return float(np.median(merged["path"] - merged["bag"]))


def _uid_hashes_for_target(
    metadata: pd.DataFrame, target: int
) -> tuple[str, str, int, int]:
    test = metadata["subject"].to_numpy() == int(target)
    train = ~test
    return (
        trial_uid_sha256(metadata.loc[train, "trial_uid"].astype(str).tolist()),
        trial_uid_sha256(metadata.loc[test, "trial_uid"].astype(str).tolist()),
        int(train.sum()),
        int(test.sum()),
    )


def _failed_null_subject_grid(
    metadata: pd.DataFrame,
    plan: OrderPermutationPlan | LabelPermutationPlan,
    observed: pd.DataFrame,
    *,
    geometry: str,
    representation: str,
    provenance: Mapping[str, Any],
    message: str,
) -> pd.DataFrame:
    subjects = tuple(sorted(int(value) for value in metadata["subject"].unique()))
    observed_rows = observed[
        (observed["geometry"] == geometry)
        & (observed["representation"] == representation)
    ].set_index("target_subject")
    rows: list[dict[str, Any]] = []
    for replicate, seed in enumerate(plan.seed_plan.seeds, start=1):
        for target in subjects:
            train_hash, test_hash, _, _ = _uid_hashes_for_target(metadata, target)
            observed_ba = (
                float(observed_rows.loc[target, "balanced_accuracy"])
                if target in observed_rows.index
                and pd.notna(observed_rows.loc[target, "balanced_accuracy"])
                else np.nan
            )
            rows.append(
                {
                    **provenance,
                    "status": "FAILED",
                    "geometry": geometry,
                    "representation": representation,
                    "replicate": replicate,
                    "replicate_seed": int(seed),
                    "target_subject": target,
                    "balanced_accuracy": np.nan,
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "observed_ba": observed_ba,
                    "subject_null_median_ba": np.nan,
                    "subject_effect": np.nan,
                    "train_uid_sha256": train_hash,
                    "test_uid_sha256": test_hash,
                    "classifier_status": "FAILED",
                    "convergence_warning": False,
                    "warning_messages": json.dumps([message], separators=(",", ":")),
                }
            )
    return pd.DataFrame.from_records(rows).reindex(columns=NULL_SUBJECT_COLUMNS)


def _failed_group_row(
    config: Mapping[str, Any],
    *,
    family: str,
    geometry: str,
    representation: str,
    provenance: Mapping[str, Any],
    observed_median: float,
    path_minus_bag: float | None = None,
) -> NullSummary:
    replicates = int(config["nulls"]["order_shuffle" if family == "order" else "label_destruction"]["replicates"])
    row: dict[str, Any] = {
        **provenance,
        "status": "FAILED",
        "geometry": geometry,
        "representation": representation,
        "observed_median_subject_ba": observed_median,
        "null_replicates": replicates,
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
    columns = LABEL_GROUP_COLUMNS
    if family == "order":
        row["median_subject_path_minus_bag"] = path_minus_bag
        columns = ORDER_GROUP_COLUMNS
    return NullSummary(
        pd.DataFrame.from_records([row]).reindex(columns=columns),
        np.asarray([], dtype=np.float64),
        False,
    )


def _run_null_condition(
    *,
    family: str,
    geometry: str,
    representation: str,
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    plan: OrderPermutationPlan | LabelPermutationPlan,
    observed: pd.DataFrame,
    provenance: Mapping[str, Any],
    path_minus_bag: float,
) -> tuple[pd.DataFrame, NullSummary]:
    subjects = tuple(int(value) for value in config["dataset"]["subjects"])
    condition_pass = _observed_condition_pass(
        observed, geometry, representation, subjects
    )
    selected_observed = observed[
        (observed["geometry"] == geometry)
        & (observed["representation"] == representation)
        & observed["balanced_accuracy"].notna()
    ]
    observed_median = (
        float(np.median(selected_observed["balanced_accuracy"]))
        if condition_pass and len(selected_observed) == len(subjects)
        else float("nan")
    )
    if not condition_pass:
        rows = _failed_null_subject_grid(
            metadata,
            plan,
            observed,
            geometry=geometry,
            representation=representation,
            provenance=provenance,
            message="observed LOSO condition FAILED; frozen null not executed",
        )
        return rows, _failed_group_row(
            config,
            family=family,
            geometry=geometry,
            representation=representation,
            provenance=provenance,
            observed_median=observed_median,
            path_minus_bag=path_minus_bag,
        )
    if family == "order":
        if not isinstance(plan, OrderPermutationPlan):
            _fail("order condition received a label plan")
        rows = run_order_shuffle_null(
            features,
            metadata,
            config,
            plan,
            observed,
            geometry=geometry,
            representation=representation,
            provenance=provenance,
        )
        if not np.isfinite(path_minus_bag):
            # PATH null rows remain exact and are retained, but the frozen
            # order operand also requires the same-geometry observed BAG.
            # Do not fabricate that cross-representation operand or expose a
            # null-only available-case group statistic.
            return rows, _failed_group_row(
                config,
                family="order",
                geometry=geometry,
                representation=representation,
                provenance=provenance,
                observed_median=observed_median,
                path_minus_bag=float("nan"),
            )
        summary = summarize_null_distribution(
            rows,
            observed,
            config,
            family="order",
            geometry=geometry,
            representation=representation,
            median_subject_path_minus_bag=path_minus_bag,
            provenance=provenance,
        )
    else:
        if not isinstance(plan, LabelPermutationPlan):
            _fail("label condition received an order plan")
        rows = run_label_destruction_null(
            features,
            metadata,
            config,
            plan,
            observed,
            geometry=geometry,
            representation=representation,
            provenance=provenance,
        )
        summary = summarize_null_distribution(
            rows,
            observed,
            config,
            family="label",
            geometry=geometry,
            representation=representation,
            provenance=provenance,
        )
    return rows, summary


def _validate_discovery_config(config: Mapping[str, Any]) -> None:
    """Reject any drift in the stage-21 scientific grid before fitting."""

    if str(config["dataset"]["allowed_session"]) != "0train":
        _fail("discovery is restricted to session 0train")
    if str(config["dataset"].get("forbidden_session")) != "1test":
        _fail("forbidden-session declaration differs from 1test")
    observed = tuple(
        (str(item[0]), str(item[1]))
        for item in config["class_probe"]["observed_conditions"]
    )
    expected_observed = (
        ("AIRM", "PATH_D10"),
        ("AIRM", "BAG_CANON_D10"),
        ("AIRM", "BAG_SORTED_D10"),
        ("AIRM", "SCALARS_11"),
        ("LE", "PATH_D10"),
        ("LE", "BAG_CANON_D10"),
        ("LE", "SCALARS_11"),
    )
    if observed != expected_observed:
        _fail("observed seven-condition LOSO grid differs from the frozen order")
    order_conditions = tuple(
        tuple(str(value) for value in item)
        for item in config["nulls"]["order_shuffle"]["conditions"]
    )
    label_conditions = tuple(
        tuple(str(value) for value in item)
        for item in config["nulls"]["label_destruction"]["shared_conditions"]
    )
    if order_conditions != (("AIRM", "PATH_D10"), ("LE", "PATH_D10")):
        _fail("order-null condition grid differs from the frozen protocol")
    if label_conditions != (
        ("AIRM", "PATH_D10"),
        ("AIRM", "BAG_CANON_D10"),
    ):
        _fail("label-null shared condition grid differs from the frozen protocol")
    if tuple(config["nulls"]["order_shuffle"]["nonidentity_permutation_indices"]) != (
        1,
        119,
    ):
        _fail("order-null permutation range is not exactly nonidentity indices 1..119")
    if str(config["nulls"]["label_destruction"]["grouping"]) != "subject_x_run":
        _fail("label-null grouping is not subject_x_run")
    if tuple(str(value) for value in config["subject_probe"]["representations"]) != (
        "PATH_D10",
        "BAG_CANON_D10",
        "SCALARS_11",
    ) or str(config["subject_probe"]["geometry"]) != "AIRM":
        _fail("run-half subject-probe grid differs from the frozen protocol")
    if tuple(str(value) for value in config["factor_decomposition"]["geometries"]) != (
        "AIRM",
        "LE",
    ):
        _fail("factor-decomposition geometry grid differs from the frozen protocol")
    if tuple(str(value) for value in config["representations"]["scalar_columns"]) != tuple(
        SCALAR_11_NAMES
    ):
        _fail("factor-decomposition scalar order differs from SCALAR_11")
    if (
        str(config["controls"]["local_barycenter"]["metric"]) != "riemann"
        or str(config["controls"]["whole_1000"]["metric"]) != "riemann"
    ):
        _fail("contextual MDM metric differs from riemann")


def _null_statistics_or_nan(summary: NullSummary, replicates: int) -> np.ndarray:
    """Preserve the full replicate axis; never substitute available cases."""

    values = np.asarray(summary.replicate_statistics, dtype=np.float64)
    if summary.complete:
        if values.shape != (replicates,) or not np.isfinite(values).all():
            _fail("a complete null summary has an invalid replicate-statistic vector")
        return _read_only(values)
    if values.size != 0:
        _fail("an incomplete null summary exposed available-case statistics")
    return _read_only(np.full(replicates, np.nan, dtype=np.float64))


def _delta_category(delta: float, *, tolerance: float = 1e-12) -> str:
    if not np.isfinite(delta):
        return "failed"
    if delta > tolerance:
        return "improved"
    if delta < -tolerance:
        return "worsened"
    return "tied"


def _agreement(
    airm_value: float,
    le_value: float,
    airm_status: str,
    le_status: str,
    *,
    threshold: float | None,
    lower_is_positive: bool = False,
) -> str:
    if airm_status != "PASS" or le_status != "PASS":
        return "FAILED"
    if not np.isfinite(airm_value) or not np.isfinite(le_value):
        return "FAILED"
    if threshold is None:
        return "AGREE"
    if lower_is_positive:
        first = airm_value <= threshold
        second = le_value <= threshold
    else:
        first = airm_value > threshold
        second = le_value > threshold
    return "AGREE" if first == second else "DISAGREE"


def _robustness_record(
    provenance: Mapping[str, Any],
    *,
    analysis: str,
    representation: str,
    subject: int | str,
    scalar: str,
    airm_value: Any,
    le_value: Any,
    airm_status: Any,
    le_status: Any,
    interpretation: str,
    threshold: float | None = None,
    lower_is_positive: bool = False,
) -> dict[str, Any]:
    airm = float(airm_value) if pd.notna(airm_value) else float("nan")
    le = float(le_value) if pd.notna(le_value) else float("nan")
    a_status = str(airm_status)
    l_status = str(le_status)
    delta = le - airm if np.isfinite(airm) and np.isfinite(le) else float("nan")
    status = "PASS" if a_status == l_status == "PASS" and np.isfinite(delta) else "FAILED"
    return {
        **provenance,
        "status": status,
        "analysis": analysis,
        "representation": representation,
        "subject": subject,
        "scalar": scalar,
        "airm_value": airm,
        "le_value": le,
        "paired_delta": delta,
        "delta_category": _delta_category(delta),
        "airm_status": a_status,
        "le_status": l_status,
        "agreement_category": _agreement(
            airm,
            le,
            a_status,
            l_status,
            threshold=threshold,
            lower_is_positive=lower_is_positive,
        ),
        "interpretation": interpretation,
    }


def _build_robustness_table(
    observed: pd.DataFrame,
    order_subject: pd.DataFrame,
    order_group: pd.DataFrame,
    factors: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    chance = 1.0 / len(config["dataset"]["classes"])
    subjects = tuple(int(value) for value in config["dataset"]["subjects"])

    for representation in ("PATH_D10", "BAG_CANON_D10", "SCALARS_11"):
        airm = observed[
            (observed["geometry"] == "AIRM")
            & (observed["representation"] == representation)
        ].set_index("target_subject")
        le = observed[
            (observed["geometry"] == "LE")
            & (observed["representation"] == representation)
        ].set_index("target_subject")
        if set(airm.index.astype(int)) != set(subjects) or set(le.index.astype(int)) != set(subjects):
            _fail(f"robustness observed pairing failed for {representation}")
        for subject in subjects:
            rows.append(
                _robustness_record(
                    provenance,
                    analysis="observed_class_loso_subject_ba",
                    representation=representation,
                    subject=subject,
                    scalar="balanced_accuracy",
                    airm_value=airm.loc[subject, "balanced_accuracy"],
                    le_value=le.loc[subject, "balanced_accuracy"],
                    airm_status=airm.loc[subject, "status"],
                    le_status=le.loc[subject, "status"],
                    threshold=chance,
                    interpretation=(
                        "LE is a paired secondary robustness repeat; delta is LE minus AIRM."
                    ),
                )
            )
        for scalar_name, reducer in (
            ("mean_subject_ba", np.mean),
            ("median_subject_ba", np.median),
        ):
            a_pass = bool((airm["status"] == "PASS").all())
            l_pass = bool((le["status"] == "PASS").all())
            a_value = (
                float(reducer(airm["balanced_accuracy"].to_numpy(dtype=float)))
                if a_pass
                else np.nan
            )
            l_value = (
                float(reducer(le["balanced_accuracy"].to_numpy(dtype=float)))
                if l_pass
                else np.nan
            )
            rows.append(
                _robustness_record(
                    provenance,
                    analysis="observed_class_loso_group",
                    representation=representation,
                    subject="GROUP",
                    scalar=scalar_name,
                    airm_value=a_value,
                    le_value=l_value,
                    airm_status="PASS" if a_pass else "FAILED",
                    le_status="PASS" if l_pass else "FAILED",
                    threshold=chance,
                    interpretation="Nine-subject aggregate; no geometry selection or voting.",
                )
            )

    order_airm = order_subject[
        (order_subject["geometry"] == "AIRM")
        & (order_subject["representation"] == "PATH_D10")
    ]
    order_le = order_subject[
        (order_subject["geometry"] == "LE")
        & (order_subject["representation"] == "PATH_D10")
    ]
    for subject in subjects:
        first = order_airm[order_airm["target_subject"] == subject]
        second = order_le[order_le["target_subject"] == subject]
        a_pass = bool(
            len(first) > 0
            and (first["status"] == "PASS").all()
            and first["subject_effect"].notna().all()
        )
        l_pass = bool(
            len(second) > 0
            and (second["status"] == "PASS").all()
            and second["subject_effect"].notna().all()
        )
        a_value = float(first["subject_effect"].iloc[0]) if a_pass else np.nan
        l_value = float(second["subject_effect"].iloc[0]) if l_pass else np.nan
        rows.append(
            _robustness_record(
                provenance,
                analysis="order_null_subject_effect",
                representation="PATH_D10",
                subject=subject,
                scalar="observed_minus_null_median_ba",
                airm_value=a_value,
                le_value=l_value,
                airm_status="PASS" if a_pass else "FAILED",
                le_status="PASS" if l_pass else "FAILED",
                threshold=0.0,
                interpretation="Positive values indicate chronological order above shuffled order.",
            )
        )

    a_group = order_group[
        (order_group["geometry"] == "AIRM")
        & (order_group["representation"] == "PATH_D10")
    ]
    l_group = order_group[
        (order_group["geometry"] == "LE")
        & (order_group["representation"] == "PATH_D10")
    ]
    if len(a_group) != 1 or len(l_group) != 1:
        _fail("order-null AIRM/LE group robustness pairing is incomplete")
    for scalar_name, threshold, lower in (
        ("observed_median_subject_ba", chance, False),
        ("null_median", chance, False),
        ("effect", 0.0, False),
        ("p_value", float(config["nulls"]["monte_carlo"]["alpha"]), True),
    ):
        rows.append(
            _robustness_record(
                provenance,
                analysis="order_null_group",
                representation="PATH_D10",
                subject="GROUP",
                scalar=scalar_name,
                airm_value=a_group.iloc[0][scalar_name],
                le_value=l_group.iloc[0][scalar_name],
                airm_status=a_group.iloc[0]["status"],
                le_status=l_group.iloc[0]["status"],
                threshold=threshold,
                lower_is_positive=lower,
                interpretation="Paired frozen order-null group operand; AIRM alone votes.",
            )
        )

    for scalar in SCALAR_11_NAMES:
        airm = factors[(factors["geometry"] == "AIRM") & (factors["scalar"] == scalar)]
        le = factors[(factors["geometry"] == "LE") & (factors["scalar"] == scalar)]
        if len(airm) != 1 or len(le) != 1:
            _fail(f"factor robustness pairing is incomplete for {scalar}")
        for component in (
            "eta2_subject",
            "eta2_class",
            "eta2_interaction",
            "eta2_residual",
        ):
            rows.append(
                _robustness_record(
                    provenance,
                    analysis="scalar_factor_component",
                    representation="SCALARS_11",
                    subject="ALL",
                    scalar=f"{scalar}:{component}",
                    airm_value=airm.iloc[0][component],
                    le_value=le.iloc[0][component],
                    airm_status=airm.iloc[0]["status"],
                    le_status=le.iloc[0]["status"],
                    threshold=None,
                    interpretation="Paired eta-squared component; magnitude comparison is descriptive.",
                )
            )
    return pd.DataFrame.from_records(rows).reindex(columns=ROBUSTNESS_COLUMNS)


def _assert_discovery_table_contracts(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any]
) -> None:
    expected_columns = {
        "class_loso_metrics": CLASS_LOSO_COLUMNS,
        "subject_runhalf_probe": SUBJECT_PROBE_COLUMNS,
        "scalar_factor_decomposition": FACTOR_COLUMNS,
        "order_shuffle_subject_metrics": NULL_SUBJECT_COLUMNS,
        "order_shuffle_group_metrics": ORDER_GROUP_COLUMNS,
        "label_null_subject_metrics": NULL_SUBJECT_COLUMNS,
        "label_null_group_metrics": LABEL_GROUP_COLUMNS,
        "local_barycenter_mdm": MDM_COLUMNS,
        "whole_context_mdm": WHOLE_MDM_COLUMNS,
        "airm_le_robustness": ROBUSTNESS_COLUMNS,
    }
    if tuple(tables) != DISCOVERY_TABLE_NAMES:
        _fail("discovery table names/order differ from required tables 11..20")
    for name, columns in expected_columns.items():
        frame = tables[name]
        if tuple(frame.columns) != tuple(columns):
            _fail(f"{name} schema differs from its exact frozen columns")
        if frame.empty:
            _fail(f"{name} is unexpectedly empty")
        if set(frame["session"].astype(str)) != {"0train"}:
            _fail(f"{name} crossed the forbidden-session barrier")
        if not set(frame["status"].astype(str)) <= {"PASS", "FAILED"}:
            _fail(f"{name} contains an invalid status token")

    subjects = set(int(value) for value in config["dataset"]["subjects"])
    replicates = int(config["nulls"]["order_shuffle"]["replicates"])
    observed_expected = {
        (str(g), str(r), subject)
        for g, r in config["class_probe"]["observed_conditions"]
        for subject in subjects
    }
    observed = tables["class_loso_metrics"]
    observed_keys = set(
        zip(
            observed["geometry"].astype(str),
            observed["representation"].astype(str),
            pd.to_numeric(observed["target_subject"]).astype(int),
            strict=True,
        )
    )
    if observed_keys != observed_expected or len(observed) != len(observed_expected):
        _fail("observed LOSO exact condition×subject grid is incomplete")

    probe = tables["subject_runhalf_probe"]
    probe_expected = {
        ("AIRM", str(representation), split)
        for representation in config["subject_probe"]["representations"]
        for split in ("A_TO_B", "B_TO_A")
    }
    if set(zip(probe["geometry"], probe["representation"], probe["split"], strict=True)) != probe_expected or len(probe) != len(probe_expected):
        _fail("run-half subject-probe exact grid is incomplete")

    factors = tables["scalar_factor_decomposition"]
    factor_expected = {
        (geometry, scalar)
        for geometry in ("AIRM", "LE")
        for scalar in SCALAR_11_NAMES
    }
    if set(zip(factors["geometry"], factors["scalar"], strict=True)) != factor_expected or len(factors) != len(factor_expected):
        _fail("factor-decomposition exact AIRM/LE×11 grid is incomplete")

    for family, conditions in (
        ("order_shuffle", (("AIRM", "PATH_D10"), ("LE", "PATH_D10"))),
        ("label_null", (("AIRM", "PATH_D10"), ("AIRM", "BAG_CANON_D10"))),
    ):
        subject_table = tables[f"{family}_subject_metrics"]
        expected = {
            (geometry, representation, replicate, subject)
            for geometry, representation in conditions
            for replicate in range(1, replicates + 1)
            for subject in subjects
        }
        actual = set(
            zip(
                subject_table["geometry"].astype(str),
                subject_table["representation"].astype(str),
                pd.to_numeric(subject_table["replicate"]).astype(int),
                pd.to_numeric(subject_table["target_subject"]).astype(int),
                strict=True,
            )
        )
        if actual != expected or len(subject_table) != len(expected):
            _fail(f"{family} exact replicate×subject grid is incomplete")
        group_table = tables[f"{family}_group_metrics"]
        group_keys = set(zip(group_table["geometry"], group_table["representation"], strict=True))
        if group_keys != set(conditions) or len(group_table) != len(conditions):
            _fail(f"{family} group grid is incomplete")

    for name, representation in (
        ("local_barycenter_mdm", "LOCAL_BARYCENTER"),
        ("whole_context_mdm", "WHOLE-1000"),
    ):
        frame = tables[name]
        if (
            set(pd.to_numeric(frame["target_subject"]).astype(int)) != subjects
            or len(frame) != len(subjects)
            or set(frame["representation"].astype(str)) != {representation}
        ):
            _fail(f"{name} exact subject grid/representation is incomplete")


def build_discovery_artifacts(
    inputs: DiscoveryInputs,
    config: Mapping[str, Any],
    *,
    generated_at_utc: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> DiscoveryArtifacts:
    """Run the frozen scientific discovery grid without writing any files."""

    _validate_discovery_config(config)
    metadata = inputs.metadata.copy(deep=True).reset_index(drop=True)
    _validate_metadata(metadata, config)
    _validate_feature_arrays(inputs.arrays, metadata, config)
    timestamp = generated_at_utc or _utc_now()
    provenance = {
        **dict(inputs.provenance),
        "generated_at_utc": timestamp,
    }

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    announce("observed class LOSO: 7 frozen conditions")
    observed_parts = [
        run_class_loso(
            features,
            metadata,
            config,
            geometry=geometry,
            representation=representation,
            provenance=provenance,
        )
        for geometry, representation, features in _condition_rows(inputs.arrays)
    ]
    observed = pd.concat(observed_parts, ignore_index=True).reindex(columns=CLASS_LOSO_COLUMNS)

    announce("AIRM run-half subject probe: 3 representations x 2 directions")
    subject_feature_map = {
        "PATH_D10": inputs.arrays["airm_path_d10"],
        "BAG_CANON_D10": inputs.arrays["airm_bag_canon_d10"],
        "SCALARS_11": inputs.arrays["airm_scalars_11"],
    }
    subject_probe = pd.concat(
        [
            run_subject_runhalf_probe(
                subject_feature_map[representation],
                metadata,
                config,
                geometry="AIRM",
                representation=representation,
                provenance=provenance,
            )
            for representation in ("PATH_D10", "BAG_CANON_D10", "SCALARS_11")
        ],
        ignore_index=True,
    ).reindex(columns=SUBJECT_PROBE_COLUMNS)

    announce("balanced scalar factor decomposition: AIRM + LE")
    factors = pd.concat(
        [
            balanced_factor_decomposition(
                inputs.arrays[f"{geometry.lower()}_scalars_11"],
                metadata,
                config,
                geometry=geometry,
                scalar_names=SCALAR_11_NAMES,
                provenance=provenance,
            )
            for geometry in ("AIRM", "LE")
        ],
        ignore_index=True,
    ).reindex(columns=FACTOR_COLUMNS)

    announce("contextual MDM controls: LOCAL_BARYCENTER + WHOLE-1000")
    local_mdm = run_mdm_loso(
        inputs.arrays["local_airm_barycenters"],
        metadata,
        config,
        representation="LOCAL_BARYCENTER",
        metric="riemann",
        provenance=provenance,
    ).reindex(columns=MDM_COLUMNS)
    whole_mdm = run_mdm_loso(
        inputs.arrays["whole_covariances"],
        metadata,
        config,
        representation="WHOLE-1000",
        metric="riemann",
        provenance=provenance,
    )
    whole_mdm["covariance_samples_per_estimate"] = int(
        config["preprocessing"]["samples_per_trial"]
    )
    whole_mdm["estimator_regime_confounded"] = True
    whole_mdm["interpretation_limit"] = (
        "WHOLE-1000 uses 1000 samples versus 200 per local covariance; "
        "this contextual comparison is estimator-regime confounded."
    )
    whole_mdm = whole_mdm.reindex(columns=WHOLE_MDM_COLUMNS)

    order_replicates = int(config["nulls"]["order_shuffle"]["replicates"])
    label_replicates = int(config["nulls"]["label_destruction"]["replicates"])
    if order_replicates != label_replicates:
        _fail("order and label null replicate counts differ")
    announce(f"constructing/replaying frozen null plans: B={order_replicates}")
    order_plan = make_order_permutation_plan(metadata, replicates=order_replicates)
    label_plan = make_label_permutation_plan(metadata, replicates=label_replicates)
    bag_audit = _assert_order_and_bag_contract(
        inputs.arrays,
        order_plan,
        tolerance=float(
            config["geometry"]["hard_gate"]["bag_invariance_absolute_error_max"]
        ),
    )
    subjects = tuple(int(value) for value in config["dataset"]["subjects"])
    path_minus_bag = {
        geometry: _paired_path_minus_bag(
            observed,
            subjects,
            geometry=geometry,
        )
        for geometry in ("AIRM", "LE")
    }

    order_subject_parts: list[pd.DataFrame] = []
    order_summaries: dict[str, NullSummary] = {}
    for geometry, values in (
        ("AIRM", inputs.arrays["airm_distance_matrices"]),
        ("LE", inputs.arrays["le_distance_matrices"]),
    ):
        announce(f"order-shuffle null: {geometry}/PATH_D10")
        rows, summary = _run_null_condition(
            family="order",
            geometry=geometry,
            representation="PATH_D10",
            features=values,
            metadata=metadata,
            config=config,
            plan=order_plan,
            observed=observed,
            provenance=provenance,
            path_minus_bag=path_minus_bag[geometry],
        )
        order_subject_parts.append(rows)
        order_summaries[geometry.lower()] = summary
    order_subject = pd.concat(order_subject_parts, ignore_index=True).reindex(
        columns=NULL_SUBJECT_COLUMNS
    )
    order_group = pd.concat(
        [order_summaries[geometry].table for geometry in ("airm", "le")],
        ignore_index=True,
    ).reindex(columns=ORDER_GROUP_COLUMNS)

    label_subject_parts: list[pd.DataFrame] = []
    label_summaries: dict[str, NullSummary] = {}
    for representation, values in (
        ("PATH_D10", inputs.arrays["airm_path_d10"]),
        ("BAG_CANON_D10", inputs.arrays["airm_bag_canon_d10"]),
    ):
        announce(f"shared label-destruction null: AIRM/{representation}")
        rows, summary = _run_null_condition(
            family="label",
            geometry="AIRM",
            representation=representation,
            features=values,
            metadata=metadata,
            config=config,
            plan=label_plan,
            observed=observed,
            provenance=provenance,
            path_minus_bag=path_minus_bag["AIRM"],
        )
        label_subject_parts.append(rows)
        label_summaries[representation.lower()] = summary
    label_subject = pd.concat(label_subject_parts, ignore_index=True).reindex(
        columns=NULL_SUBJECT_COLUMNS
    )
    label_group = pd.concat(
        [
            label_summaries[representation].table
            for representation in ("path_d10", "bag_canon_d10")
        ],
        ignore_index=True,
    ).reindex(columns=LABEL_GROUP_COLUMNS)

    robustness = _build_robustness_table(
        observed,
        order_subject,
        order_group,
        factors,
        config,
        provenance,
    )
    tables = {
        "class_loso_metrics": observed,
        "subject_runhalf_probe": subject_probe,
        "scalar_factor_decomposition": factors,
        "order_shuffle_subject_metrics": order_subject,
        "order_shuffle_group_metrics": order_group,
        "label_null_subject_metrics": label_subject,
        "label_null_group_metrics": label_group,
        "local_barycenter_mdm": local_mdm,
        "whole_context_mdm": whole_mdm,
        "airm_le_robustness": robustness,
    }
    _assert_discovery_table_contracts(tables, config)

    order_statistics = {
        "replicate": np.arange(1, order_replicates + 1, dtype=np.int64),
        "replicate_seed": np.asarray(order_plan.seed_plan.seeds, dtype=np.uint64),
        "airm__path_d10__median_subject_ba": _null_statistics_or_nan(
            order_summaries["airm"], order_replicates
        ),
        "le__path_d10__median_subject_ba": _null_statistics_or_nan(
            order_summaries["le"], order_replicates
        ),
    }
    label_statistics = {
        "replicate": np.arange(1, label_replicates + 1, dtype=np.int64),
        "replicate_seed": np.asarray(label_plan.seed_plan.seeds, dtype=np.uint64),
        "airm__path_d10__median_subject_ba": _null_statistics_or_nan(
            label_summaries["path_d10"], label_replicates
        ),
        "airm__bag_canon_d10__median_subject_ba": _null_statistics_or_nan(
            label_summaries["bag_canon_d10"], label_replicates
        ),
    }
    for value in (*order_statistics.values(), *label_statistics.values()):
        np.asarray(value).setflags(write=False)
    failure_count = sum(
        int((frame["status"].astype(str) != "PASS").sum()) for frame in tables.values()
    )
    return DiscoveryArtifacts(
        tables=tables,
        order_plan=order_plan,
        label_plan=label_plan,
        order_group_statistics=order_statistics,
        label_group_statistics=label_statistics,
        provenance=provenance,
        technical_failure_count=failure_count,
        status="PASS" if failure_count == 0 else "FAILED",
        bag_plan_audit_sha256=bag_audit,
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_null_artifacts(
    artifacts: DiscoveryArtifacts, config: Mapping[str, Any]
) -> None:
    order_replicates = int(config["nulls"]["order_shuffle"]["replicates"])
    label_replicates = int(config["nulls"]["label_destruction"]["replicates"])
    specifications = (
        (
            "order",
            artifacts.order_group_statistics,
            artifacts.order_plan.seed_plan.seeds,
            order_replicates,
            (
                "replicate",
                "replicate_seed",
                "airm__path_d10__median_subject_ba",
                "le__path_d10__median_subject_ba",
            ),
            artifacts.tables["order_shuffle_group_metrics"],
        ),
        (
            "label",
            artifacts.label_group_statistics,
            artifacts.label_plan.seed_plan.seeds,
            label_replicates,
            (
                "replicate",
                "replicate_seed",
                "airm__path_d10__median_subject_ba",
                "airm__bag_canon_d10__median_subject_ba",
            ),
            artifacts.tables["label_null_group_metrics"],
        ),
    )
    for family, payload, seeds, replicates, keys, group in specifications:
        if tuple(payload) != keys:
            _fail(f"{family} null NPZ keys/order differ from the frozen contract")
        replicate = np.asarray(payload["replicate"])
        replicate_seed = np.asarray(payload["replicate_seed"])
        if replicate.dtype != np.dtype("int64") or not np.array_equal(
            replicate, np.arange(1, replicates + 1, dtype=np.int64)
        ):
            _fail(f"{family} null replicate identity vector is invalid")
        if (
            replicate_seed.dtype != np.dtype("uint64")
            or replicate_seed.shape != (replicates,)
            or not np.array_equal(replicate_seed, np.asarray(seeds, dtype=np.uint64))
        ):
            _fail(f"{family} null seed vector differs from its replay plan")
        for key in keys[2:]:
            values = np.asarray(payload[key])
            if values.dtype != np.dtype("float64") or values.shape != (replicates,):
                _fail(f"{family} null statistic {key} has an invalid dtype/shape")
            geometry, representation, _ = key.split("__", maxsplit=2)
            selected = group[
                group["geometry"].astype(str).str.lower().eq(geometry)
                & group["representation"].astype(str).str.lower().eq(representation)
            ]
            if len(selected) != 1:
                _fail(f"{family} null statistic {key} has no unique group row")
            if str(selected.iloc[0]["status"]) == "PASS":
                if not np.isfinite(values).all():
                    _fail(f"PASS {family} null statistic {key} is non-finite")
            elif not np.isnan(values).all():
                _fail(
                    f"FAILED {family} null statistic {key} used available cases"
                )

    if artifacts.order_plan.permutation_indices.shape != (
        order_replicates,
        int(config["dataset"]["expected_trials"]),
    ) or np.any(
        (artifacts.order_plan.permutation_indices < 1)
        | (artifacts.order_plan.permutation_indices > 119)
    ):
        _fail("order plan shape/nonidentity contract changed before writing")
    if artifacts.label_plan.source_row_indices.shape != (
        label_replicates,
        int(config["dataset"]["expected_trials"]),
    ):
        _fail("label plan shape changed before writing")
    observed_failures = sum(
        int((frame["status"].astype(str) != "PASS").sum())
        for frame in artifacts.tables.values()
    )
    if observed_failures != int(artifacts.technical_failure_count):
        _fail("technical_failure_count does not equal recorded FAILED table rows")
    expected_status = "PASS" if observed_failures == 0 else "FAILED"
    if artifacts.status != expected_status:
        _fail("discovery artifact status disagrees with recorded FAILED rows")


def write_discovery_artifacts(
    artifacts: DiscoveryArtifacts,
    config: Mapping[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    """Atomically write only exact trajectory-v0 tables 11--20 and null artifacts."""

    _validate_discovery_config(config)
    _assert_discovery_table_contracts(artifacts.tables, config)
    _validate_null_artifacts(artifacts, config)
    project_root = Path(root).expanduser().resolve()
    if str(config["project"]["output_dir"]) != "outputs/bnci2014_001_trajectory_v0":
        _fail("discovery writer output_dir left the frozen namespace")
    output_root = (project_root / str(config["project"]["output_dir"])).resolve()
    tables_dir = output_root / "tables"
    nulls_dir = output_root / "nulls"
    written: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name in DISCOVERY_TABLE_NAMES:
        path = tables_dir / f"{name}.csv"
        _atomic_text(path, artifacts.tables[name].to_csv(index=False, lineterminator="\n"))
        written[f"table_{name}"] = str(path)
        hashes[f"table_{name}"] = _file_sha256(path)

    protocol_version = str(config["project"]["protocol_version"])
    order_json = artifacts.order_plan.seed_plan.to_json_dict(
        protocol_version=protocol_version
    )
    label_json = artifacts.label_plan.seed_plan.to_json_dict(
        protocol_version=protocol_version
    )
    json_payloads = {
        "order_shuffle_seeds.json": order_json,
        "label_permutation_seeds.json": label_json,
    }
    npz_payloads = {
        "order_shuffle_group_stats.npz": artifacts.order_group_statistics,
        "label_null_group_stats.npz": artifacts.label_group_statistics,
    }
    for name, payload in json_payloads.items():
        path = nulls_dir / name
        _atomic_text(
            path,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        written[name] = str(path)
        hashes[name] = _file_sha256(path)
    for name, payload in npz_payloads.items():
        path = nulls_dir / name
        _atomic_npz(path, payload)
        written[name] = str(path)
        hashes[name] = _file_sha256(path)

    return {
        "status": artifacts.status,
        "technical_failure_count": int(artifacts.technical_failure_count),
        "protocol_version": protocol_version,
        "protocol_sha256": str(config["project"]["protocol_sha256"]),
        "config_sha256": str(artifacts.provenance["config_sha256"]),
        "session": str(artifacts.provenance["session"]),
        "feature_npz_sha256": str(artifacts.provenance["feature_npz_sha256"]),
        "order_plan_audit_sha256": artifacts.order_plan.audit_sha256,
        "label_plan_audit_sha256": artifacts.label_plan.audit_sha256,
        "bag_plan_audit_sha256": artifacts.bag_plan_audit_sha256,
        "written": written,
        "sha256": hashes,
    }


__all__ = [
    "DISCOVERY_TABLE_NAMES",
    "NULL_ARTIFACT_NAMES",
    "ROBUSTNESS_COLUMNS",
    "WHOLE_MDM_COLUMNS",
    "DiscoveryArtifacts",
    "DiscoveryInputs",
    "DiscoveryStructuralError",
    "build_discovery_artifacts",
    "load_discovery_inputs",
    "write_discovery_artifacts",
]
