"""Observed, resumable-null, and frozen-decision orchestration for audit v1."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.trajectory_within_subject_data_v1 import load_prepared_audit_data
from src.trajectory_within_subject_v1 import (
    CLASS_ORDER,
    NULL_REPLICATES,
    IncompleteRequiredGrid,
    _fit,
    apply_order_shuffle,
    load_frozen_config,
    make_seed_vector,
    monte_carlo_result,
    order_permutation_indices,
    permute_labels,
    run_stage_w,
    run_stage_x,
    sha256_array,
    sha256_file,
    stable_json_sha256,
    stage_subject_statistics,
    terminal_decision,
)


REPRESENTATION_ARRAYS = {
    "PATH_D10": "airm_path_d10",
    "BAG_CANON_D10": "airm_bag_canon_d10",
    "SCALARS_11": "airm_scalars_11",
}

LABEL_CHECKPOINT_KEYS = (
    "replicate",
    "replicate_seed",
    "completed",
    "path_w_subject",
    "path_w_group",
    "path_x_subject",
    "path_x_group",
    "bag_w_subject",
    "bag_w_group",
    "bag_x_subject",
    "bag_x_group",
    "identity_sha256",
)

ORDER_CHECKPOINT_KEYS = (
    "replicate",
    "replicate_seed",
    "completed",
    "path_x_subject",
    "path_x_group",
    "identity_sha256",
)

_WORKER: dict[str, Any] = {}


class TrajectoryAuditAnalysisError(RuntimeError):
    """A required observed/null grid or checkpoint contract failed."""


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".npz", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _seed_manifest(family: str, seeds: np.ndarray) -> dict[str, Any]:
    tag = 0x4C4142454C5758 if family == "label" else 0x4F52444552
    return {
        "protocol_version": "1.0",
        "master_seed": 20260810,
        "family": family,
        "stream_tag_hex": f"0x{tag:X}",
        "seedsequence_entropy": [20260810, tag],
        "child_count": len(seeds),
        "seed_dtype": "uint64",
        "seed_extraction": "int(child.generate_state(1, dtype=np.uint64)[0])",
        "replicates": [
            {"replicate": index + 1, "seed": int(seed)}
            for index, seed in enumerate(seeds)
        ],
    }


def _split_indices(metadata: pd.DataFrame) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], list[tuple[int, np.ndarray, np.ndarray]]]:
    w: list[tuple[int, np.ndarray, np.ndarray]] = []
    x: list[tuple[int, np.ndarray, np.ndarray]] = []
    for subject in range(1, 10):
        subject_mask = metadata["subject"].eq(subject)
        for session in ("0train", "1test"):
            scope = subject_mask & metadata["session"].eq(session)
            a = (scope & metadata["run"].isin([0, 1, 2])).to_numpy()
            b = (scope & metadata["run"].isin([3, 4, 5])).to_numpy()
            if int(a.sum()) != 144 or int(b.sum()) != 144 or np.any(a & b):
                raise TrajectoryAuditAnalysisError("Stage W split contract failed")
            w.extend(((subject, np.flatnonzero(a), np.flatnonzero(b)), (subject, np.flatnonzero(b), np.flatnonzero(a))))
        session0 = (subject_mask & metadata["session"].eq("0train")).to_numpy()
        session1 = (subject_mask & metadata["session"].eq("1test")).to_numpy()
        if int(session0.sum()) != 288 or int(session1.sum()) != 288 or np.any(session0 & session1):
            raise TrajectoryAuditAnalysisError("Stage X split contract failed")
        x.extend(((subject, np.flatnonzero(session0), np.flatnonzero(session1)), (subject, np.flatnonzero(session1), np.flatnonzero(session0))))
    return w, x


def _fit_balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    config: Mapping[str, Any],
) -> float:
    fitted = _fit(features[train], labels[train], config)
    if fitted.status != "PASS":
        raise IncompleteRequiredGrid("required null fit emitted a convergence warning")
    prediction = fitted.model.predict(fitted.scaler.transform(features[test])).astype(str)
    truth = labels[test].astype(str)
    recalls = []
    for label in CLASS_ORDER:
        mask = truth == label
        if int(mask.sum()) == 0:
            raise IncompleteRequiredGrid("required null test fold lacks a class")
        recalls.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(recalls))


def _fast_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[int, np.ndarray, np.ndarray]],
    fits_per_subject: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    scores = np.full((9, fits_per_subject), np.nan, dtype=np.float64)
    positions = np.zeros(9, dtype=np.int64)
    for subject, train, test in splits:
        subject_position = int(subject) - 1
        slot = int(positions[subject_position])
        scores[subject_position, slot] = _fit_balanced_accuracy(
            features, labels, train, test, config
        )
        positions[subject_position] += 1
    if not np.array_equal(positions, np.full(9, fits_per_subject)) or not np.isfinite(scores).all():
        raise IncompleteRequiredGrid("required null subject grid is incomplete")
    subject = scores.mean(axis=1)
    return subject, float(np.median(subject))


def _worker_init(config_path: str, root: str) -> None:
    global _WORKER
    config = load_frozen_config(config_path)
    prepared = load_prepared_audit_data(config_path, root)
    w_splits, x_splits = _split_indices(prepared.metadata)
    _WORKER = {
        "config": config,
        "metadata": prepared.metadata,
        "labels": prepared.metadata["class_label"].to_numpy(dtype=str),
        "path": prepared.arrays["airm_path_d10"],
        "bag": prepared.arrays["airm_bag_canon_d10"],
        "w_splits": w_splits,
        "x_splits": x_splits,
    }


def _label_batch(task: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    replicate_indices, seeds = task
    n = len(replicate_indices)
    result = {
        "replicate_index": np.asarray(replicate_indices, dtype=np.int64),
        "path_w_subject": np.full((n, 9), np.nan),
        "path_w_group": np.full(n, np.nan),
        "path_x_subject": np.full((n, 9), np.nan),
        "path_x_group": np.full(n, np.nan),
        "bag_w_subject": np.full((n, 9), np.nan),
        "bag_w_group": np.full(n, np.nan),
        "bag_x_subject": np.full((n, 9), np.nan),
        "bag_x_group": np.full(n, np.nan),
    }
    for local, seed in enumerate(seeds):
        labels = permute_labels(_WORKER["labels"], _WORKER["metadata"], int(seed))
        for prefix, features in (("path", _WORKER["path"]), ("bag", _WORKER["bag"])):
            subject_w, group_w = _fast_statistics(
                features, labels, _WORKER["w_splits"], 4, _WORKER["config"]
            )
            subject_x, group_x = _fast_statistics(
                features, labels, _WORKER["x_splits"], 2, _WORKER["config"]
            )
            result[f"{prefix}_w_subject"][local] = subject_w
            result[f"{prefix}_w_group"][local] = group_w
            result[f"{prefix}_x_subject"][local] = subject_x
            result[f"{prefix}_x_group"][local] = group_x
    return result


def _order_batch(task: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    replicate_indices, seeds = task
    n = len(replicate_indices)
    result = {
        "replicate_index": np.asarray(replicate_indices, dtype=np.int64),
        "path_x_subject": np.full((n, 9), np.nan),
        "path_x_group": np.full(n, np.nan),
    }
    for local, seed in enumerate(seeds):
        indices = order_permutation_indices(len(_WORKER["path"]), int(seed))
        shuffled = apply_order_shuffle(_WORKER["path"], indices)
        subject, group = _fast_statistics(
            shuffled,
            _WORKER["labels"],
            _WORKER["x_splits"],
            2,
            _WORKER["config"],
        )
        result["path_x_subject"][local] = subject
        result["path_x_group"][local] = group
    return result


def _checkpoint_identity(config_path: Path, root: Path, family: str, prepared_hash: str) -> dict[str, Any]:
    source_paths = [
        root / "src" / "trajectory_within_subject_v1.py",
        root / "src" / "trajectory_within_subject_analysis_v1.py",
    ]
    return {
        "family": family,
        "protocol_sha256": load_frozen_config(config_path)["protocol"]["sha256"],
        "config_sha256": sha256_file(config_path),
        "combined_cache_sha256": prepared_hash,
        "source_sha256": {path.name: sha256_file(path) for path in source_paths},
        "replicates": NULL_REPLICATES,
    }


def _new_label_checkpoint(seeds: np.ndarray, identity_hash: str) -> dict[str, np.ndarray]:
    return {
        "replicate": np.arange(1, NULL_REPLICATES + 1, dtype=np.int64),
        "replicate_seed": np.asarray(seeds, dtype=np.uint64),
        "completed": np.zeros(NULL_REPLICATES, dtype=bool),
        "path_w_subject": np.full((NULL_REPLICATES, 9), np.nan),
        "path_w_group": np.full(NULL_REPLICATES, np.nan),
        "path_x_subject": np.full((NULL_REPLICATES, 9), np.nan),
        "path_x_group": np.full(NULL_REPLICATES, np.nan),
        "bag_w_subject": np.full((NULL_REPLICATES, 9), np.nan),
        "bag_w_group": np.full(NULL_REPLICATES, np.nan),
        "bag_x_subject": np.full((NULL_REPLICATES, 9), np.nan),
        "bag_x_group": np.full(NULL_REPLICATES, np.nan),
        "identity_sha256": np.asarray([identity_hash], dtype=str),
    }


def _new_order_checkpoint(seeds: np.ndarray, identity_hash: str) -> dict[str, np.ndarray]:
    return {
        "replicate": np.arange(1, NULL_REPLICATES + 1, dtype=np.int64),
        "replicate_seed": np.asarray(seeds, dtype=np.uint64),
        "completed": np.zeros(NULL_REPLICATES, dtype=bool),
        "path_x_subject": np.full((NULL_REPLICATES, 9), np.nan),
        "path_x_group": np.full(NULL_REPLICATES, np.nan),
        "identity_sha256": np.asarray([identity_hash], dtype=str),
    }


def _load_checkpoint(
    path: Path,
    keys: Sequence[str],
    factory: Callable[[], dict[str, np.ndarray]],
    identity_hash: str,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        return factory()
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != tuple(keys):
            raise TrajectoryAuditAnalysisError(f"checkpoint keys changed: {path}")
        checkpoint = {key: np.asarray(archive[key]) for key in keys}
    if str(checkpoint["identity_sha256"][0]) != identity_hash:
        raise TrajectoryAuditAnalysisError(f"checkpoint identity changed: {path}")
    if checkpoint["completed"].shape != (NULL_REPLICATES,):
        raise TrajectoryAuditAnalysisError("checkpoint completion shape changed")
    return checkpoint


def _tasks(pending: np.ndarray, seeds: np.ndarray, batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (batch.astype(np.int64), seeds[batch].astype(np.uint64))
        for batch in np.array_split(pending, np.arange(batch_size, len(pending), batch_size))
        if len(batch)
    ]


def run_observed(
    config_path: str | Path,
    root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(root).resolve()
    config_file = Path(config_path).resolve()
    config = load_frozen_config(config_file)
    prepared = load_prepared_audit_data(config_file, project_root)
    announce = progress or (lambda _message: None)
    within_parts: list[pd.DataFrame] = []
    cross_parts: list[pd.DataFrame] = []
    for representation, key in REPRESENTATION_ARRAYS.items():
        announce(f"observed Stage W: {representation}")
        within = run_stage_w(
            prepared.arrays[key], prepared.metadata, config, representation=representation
        )
        subject_w, statistic_w = stage_subject_statistics(within, "W")
        within["subject_statistic"] = within["subject"].astype(int).map(
            {subject: float(subject_w[subject - 1]) for subject in range(1, 10)}
        )
        within["group_statistic"] = statistic_w
        within["chance"] = 0.25
        within_parts.append(within)
        announce(f"observed Stage X: {representation}")
        cross = run_stage_x(
            prepared.arrays[key], prepared.metadata, config, representation=representation
        )
        subject_x, statistic_x = stage_subject_statistics(cross, "X")
        cross["subject_statistic"] = cross["subject"].astype(int).map(
            {subject: float(subject_x[subject - 1]) for subject in range(1, 10)}
        )
        cross["group_statistic"] = statistic_x
        cross["chance"] = 0.25
        cross_parts.append(cross)
    within_table = pd.concat(within_parts, ignore_index=True)
    cross_table = pd.concat(cross_parts, ignore_index=True)
    output_root = project_root / str(config["project"]["output_dir"])
    _atomic_text(
        output_root / "tables" / "within_session_scores.csv",
        within_table.to_csv(index=False, lineterminator="\n"),
    )
    _atomic_text(
        output_root / "tables" / "cross_session_scores.csv",
        cross_table.to_csv(index=False, lineterminator="\n"),
    )
    return within_table, cross_table


def load_observed(config: Mapping[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root = root / str(config["project"]["output_dir"])
    within = pd.read_csv(output_root / "tables" / "within_session_scores.csv")
    cross = pd.read_csv(output_root / "tables" / "cross_session_scores.csv")
    for table, stage, expected in ((within, "W", 108), (cross, "X", 54)):
        if len(table) != expected or set(table["representation"]) != set(REPRESENTATION_ARRAYS):
            raise TrajectoryAuditAnalysisError(f"observed Stage {stage} table is incomplete")
        if not (table["status"] == "PASS").all() or table["balanced_accuracy"].isna().any():
            raise IncompleteRequiredGrid(f"observed Stage {stage} contains a failed fit")
    return within, cross


def _observed_statistic(table: pd.DataFrame, representation: str, stage: str) -> tuple[np.ndarray, float]:
    selected = table[table["representation"].eq(representation)].copy()
    return stage_subject_statistics(selected, stage)


def run_label_nulls(
    config_path: str | Path,
    root: str | Path,
    *,
    workers: int = 4,
    batch_size: int = 8,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    project_root = Path(root).resolve()
    config_file = Path(config_path).resolve()
    config = load_frozen_config(config_file)
    prepared = load_prepared_audit_data(config_file, project_root)
    within, cross = load_observed(config, project_root)
    announce = progress or (lambda _message: None)
    seeds = make_seed_vector("label")
    identity = _checkpoint_identity(
        config_file, project_root, "label", sha256_file(prepared.combined_cache_path)
    )
    identity_hash = stable_json_sha256(identity)
    checkpoint_path = project_root / str(config["project"]["cache_dir"]) / "label_null_checkpoint.npz"
    checkpoint = _load_checkpoint(
        checkpoint_path,
        LABEL_CHECKPOINT_KEYS,
        lambda: _new_label_checkpoint(seeds, identity_hash),
        identity_hash,
    )
    if not np.array_equal(checkpoint["replicate_seed"], seeds):
        raise TrajectoryAuditAnalysisError("label checkpoint seed vector changed")
    pending = np.flatnonzero(~checkpoint["completed"])
    announce(f"label null resume {int(checkpoint['completed'].sum())}/{NULL_REPLICATES}")
    task_list = _tasks(pending, seeds, int(batch_size))
    if task_list:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=int(workers),
            initializer=_worker_init,
            initargs=(str(config_file), str(project_root)),
        ) as pool:
            for result in pool.imap_unordered(_label_batch, task_list, chunksize=1):
                positions = result.pop("replicate_index")
                for key, values in result.items():
                    checkpoint[key][positions] = values
                checkpoint["completed"][positions] = True
                _atomic_npz(checkpoint_path, checkpoint)
                announce(f"label null {int(checkpoint['completed'].sum())}/{NULL_REPLICATES}")
    if not checkpoint["completed"].all():
        raise IncompleteRequiredGrid("label null checkpoint is incomplete")
    for key in LABEL_CHECKPOINT_KEYS[3:-1]:
        if not np.isfinite(checkpoint[key]).all():
            raise IncompleteRequiredGrid(f"label null statistic is incomplete: {key}")

    rows: list[dict[str, Any]] = []
    observed_map: dict[tuple[str, str], float] = {}
    for representation, prefix in (("PATH_D10", "path"), ("BAG_CANON_D10", "bag")):
        for stage, table in (("W", within), ("X", cross)):
            _, observed = _observed_statistic(table, representation, stage)
            observed_map[(representation, stage)] = observed
            result = monte_carlo_result(observed, checkpoint[f"{prefix}_{stage.lower()}_group"])
            rows.append(
                {
                    "stage": stage,
                    "representation": representation,
                    **asdict(result),
                    "shared_label_realization": True,
                    "role": "primary" if representation == "PATH_D10" else "mandatory_unordered_comparator",
                }
            )
    summary = pd.DataFrame.from_records(rows)
    path_w = bool(summary.loc[(summary.stage.eq("W")) & summary.representation.eq("PATH_D10"), "passed"].iloc[0])
    path_x_raw = bool(summary.loc[(summary.stage.eq("X")) & summary.representation.eq("PATH_D10"), "passed"].iloc[0])
    bag_w = bool(summary.loc[(summary.stage.eq("W")) & summary.representation.eq("BAG_CANON_D10"), "passed"].iloc[0])
    bag_x_raw = bool(summary.loc[(summary.stage.eq("X")) & summary.representation.eq("BAG_CANON_D10"), "passed"].iloc[0])
    summary["prerequisite_pass"] = summary.apply(
        lambda row: True if row.stage == "W" else (path_w if row.representation == "PATH_D10" else bag_w),
        axis=1,
    )
    summary["inferential_pass"] = summary["passed"] & summary["prerequisite_pass"]
    output_root = project_root / str(config["project"]["output_dir"])
    _atomic_text(
        output_root / "tables" / "label_null_summary.csv",
        summary.to_csv(index=False, lineterminator="\n"),
    )
    nulls_dir = output_root / "nulls"
    _atomic_text(
        nulls_dir / "label_null_seeds.json",
        json.dumps(_seed_manifest("label", seeds), indent=2, sort_keys=True) + "\n",
    )
    final_payload = {key: value for key, value in checkpoint.items() if key != "identity_sha256"}
    final_payload["identity_sha256"] = np.asarray([identity_hash], dtype=str)
    _atomic_npz(nulls_dir / "label_null_statistics.npz", final_payload)
    announce(
        f"label-null decisions PATH W={path_w} X_raw={path_x_raw}; BAG W={bag_w} X_raw={bag_x_raw}"
    )
    return summary, checkpoint


def run_order_nulls(
    config_path: str | Path,
    root: str | Path,
    *,
    workers: int = 4,
    batch_size: int = 8,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    project_root = Path(root).resolve()
    config_file = Path(config_path).resolve()
    config = load_frozen_config(config_file)
    prepared = load_prepared_audit_data(config_file, project_root)
    within, cross = load_observed(config, project_root)
    label_summary = pd.read_csv(
        project_root / str(config["project"]["output_dir"]) / "tables" / "label_null_summary.csv"
    )
    path_w = bool(label_summary.loc[(label_summary.stage.eq("W")) & label_summary.representation.eq("PATH_D10"), "inferential_pass"].iloc[0])
    path_x = bool(label_summary.loc[(label_summary.stage.eq("X")) & label_summary.representation.eq("PATH_D10"), "inferential_pass"].iloc[0])
    _, observed_x = _observed_statistic(cross, "PATH_D10", "X")
    announce = progress or (lambda _message: None)
    seeds = make_seed_vector("order")
    output_root = project_root / str(config["project"]["output_dir"])
    nulls_dir = output_root / "nulls"
    _atomic_text(
        nulls_dir / "order_null_seeds.json",
        json.dumps(_seed_manifest("order", seeds), indent=2, sort_keys=True) + "\n",
    )
    if not (path_w and path_x):
        summary = pd.DataFrame.from_records(
            [
                {
                    "stage": "O",
                    "representation": "PATH_D10",
                    "observed": observed_x,
                    "null_median": np.nan,
                    "effect": np.nan,
                    "p_value": np.nan,
                    "exceedance_count": np.nan,
                    "replicates": 0,
                    "passed": False,
                    "status": "NOT_RUN_PREREQUISITE",
                    "prerequisite_w_pass": path_w,
                    "prerequisite_x_pass": path_x,
                }
            ]
        )
        _atomic_text(
            output_root / "tables" / "order_null_summary.csv",
            summary.to_csv(index=False, lineterminator="\n"),
        )
        _atomic_npz(
            nulls_dir / "order_null_statistics.npz",
            {
                "replicate": np.asarray([], dtype=np.int64),
                "replicate_seed": np.asarray([], dtype=np.uint64),
                "path_x_subject": np.empty((0, 9), dtype=np.float64),
                "path_x_group": np.asarray([], dtype=np.float64),
                "status": np.asarray(["NOT_RUN_PREREQUISITE"], dtype=str),
            },
        )
        announce("order null not run: PATH W/X prerequisite did not pass")
        return summary, None

    identity = _checkpoint_identity(
        config_file, project_root, "order", sha256_file(prepared.combined_cache_path)
    )
    identity_hash = stable_json_sha256(identity)
    checkpoint_path = project_root / str(config["project"]["cache_dir"]) / "order_null_checkpoint.npz"
    checkpoint = _load_checkpoint(
        checkpoint_path,
        ORDER_CHECKPOINT_KEYS,
        lambda: _new_order_checkpoint(seeds, identity_hash),
        identity_hash,
    )
    if not np.array_equal(checkpoint["replicate_seed"], seeds):
        raise TrajectoryAuditAnalysisError("order checkpoint seed vector changed")
    pending = np.flatnonzero(~checkpoint["completed"])
    announce(f"order null resume {int(checkpoint['completed'].sum())}/{NULL_REPLICATES}")
    task_list = _tasks(pending, seeds, int(batch_size))
    if task_list:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=int(workers),
            initializer=_worker_init,
            initargs=(str(config_file), str(project_root)),
        ) as pool:
            for result in pool.imap_unordered(_order_batch, task_list, chunksize=1):
                positions = result["replicate_index"]
                checkpoint["path_x_subject"][positions] = result["path_x_subject"]
                checkpoint["path_x_group"][positions] = result["path_x_group"]
                checkpoint["completed"][positions] = True
                _atomic_npz(checkpoint_path, checkpoint)
                announce(f"order null {int(checkpoint['completed'].sum())}/{NULL_REPLICATES}")
    if not checkpoint["completed"].all() or not np.isfinite(checkpoint["path_x_group"]).all():
        raise IncompleteRequiredGrid("order null checkpoint is incomplete")
    result = monte_carlo_result(observed_x, checkpoint["path_x_group"])
    summary = pd.DataFrame.from_records(
        [
            {
                "stage": "O",
                "representation": "PATH_D10",
                **asdict(result),
                "status": "PASS",
                "prerequisite_w_pass": path_w,
                "prerequisite_x_pass": path_x,
            }
        ]
    )
    _atomic_text(
        output_root / "tables" / "order_null_summary.csv",
        summary.to_csv(index=False, lineterminator="\n"),
    )
    final_payload = {key: value for key, value in checkpoint.items() if key != "identity_sha256"}
    final_payload["identity_sha256"] = np.asarray([identity_hash], dtype=str)
    _atomic_npz(nulls_dir / "order_null_statistics.npz", final_payload)
    announce(f"order-null decision pass={result.passed}")
    return summary, checkpoint


def finalize_decision_and_comparison(
    config_path: str | Path, root: str | Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    project_root = Path(root).resolve()
    config_file = Path(config_path).resolve()
    config = load_frozen_config(config_file)
    prepared = load_prepared_audit_data(config_file, project_root)
    within, cross = load_observed(config, project_root)
    output_root = project_root / str(config["project"]["output_dir"])
    label = pd.read_csv(output_root / "tables" / "label_null_summary.csv")
    order = pd.read_csv(output_root / "tables" / "order_null_summary.csv")
    path_w = bool(label.loc[(label.stage.eq("W")) & label.representation.eq("PATH_D10"), "inferential_pass"].iloc[0])
    path_x = bool(label.loc[(label.stage.eq("X")) & label.representation.eq("PATH_D10"), "inferential_pass"].iloc[0])
    order_status = str(order.iloc[0]["status"])
    order_pass = None if order_status == "NOT_RUN_PREREQUISITE" else bool(order.iloc[0]["passed"])
    decision = terminal_decision(
        reproduction_gate_pass=bool(prepared.reproduction_gate["passed"].all()),
        technical_grid_pass=True,
        stage_w_pass=path_w,
        stage_x_pass=path_x,
        stage_o_pass=order_pass,
    )
    comparison_rows: list[dict[str, Any]] = []
    subject_context: dict[str, Any] = {}
    for representation in REPRESENTATION_ARRAYS:
        subject_w, statistic_w = _observed_statistic(within, representation, "W")
        subject_x, statistic_x = _observed_statistic(cross, representation, "X")
        subject_context[representation] = {
            "stage_w_subject_values": subject_w.tolist(),
            "stage_x_subject_values": subject_x.tolist(),
            "stage_w_above_chance_count": int(np.count_nonzero(subject_w > 0.25)),
            "stage_x_above_chance_count": int(np.count_nonzero(subject_x > 0.25)),
        }
        comparison_rows.append(
            {
                "representation": representation,
                "role": (
                    "primary" if representation == "PATH_D10" else
                    "mandatory_unordered_comparator" if representation == "BAG_CANON_D10" else
                    "descriptive_only"
                ),
                "stage_w_observed_median_subject_ba": statistic_w,
                "stage_x_observed_median_subject_ba": statistic_x,
                "stage_w_subjects_above_chance": int(np.count_nonzero(subject_w > 0.25)),
                "stage_x_subjects_above_chance": int(np.count_nonzero(subject_x > 0.25)),
                "terminal_vote": representation == "PATH_D10",
            }
        )
    comparison = pd.DataFrame.from_records(comparison_rows)
    _atomic_text(
        output_root / "tables" / "representation_comparison.csv",
        comparison.to_csv(index=False, lineterminator="\n"),
    )
    chain = pd.DataFrame.from_records(
        [
            {"step": "reproduction_gate", "status": "PASS", "vote": True},
            {"step": "technical_grid", "status": "PASS", "vote": True},
            {"step": "PATH_STAGE_W", "status": "PASS" if path_w else "FAIL", "vote": path_w},
            {"step": "PATH_STAGE_X", "status": "PASS" if path_x else "FAIL", "vote": path_x},
            {"step": "PATH_STAGE_O", "status": order_status if order_status == "NOT_RUN_PREREQUISITE" else ("PASS" if order_pass else "FAIL"), "vote": order_pass},
            {"step": "terminal", "status": decision.decision, "vote": ""},
        ]
    )
    decisions_dir = output_root / "decisions"
    _atomic_text(decisions_dir / "decision_chain.csv", chain.to_csv(index=False, lineterminator="\n"))
    payload = {
        "protocol_version": "1.0",
        "protocol_sha256": str(config["protocol"]["sha256"]),
        "config_sha256": sha256_file(config_file),
        "decision": decision.decision,
        "stage_w_pass": decision.stage_w_pass,
        "stage_x_pass": decision.stage_x_pass,
        "stage_o_pass": decision.stage_o_pass,
        "order_status": order_status,
        "label_null_summary": label.to_dict(orient="records"),
        "order_null_summary": order.where(pd.notna(order), None).to_dict(orient="records"),
        "subject_context": subject_context,
        "claim_restrictions_applied": True,
        "whole_subject_class_interaction_used": False,
    }
    _atomic_text(decisions_dir / "terminal_decision.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload, comparison


__all__ = [
    "REPRESENTATION_ARRAYS",
    "TrajectoryAuditAnalysisError",
    "run_observed",
    "load_observed",
    "run_label_nulls",
    "run_order_nulls",
    "finalize_decision_and_comparison",
]
