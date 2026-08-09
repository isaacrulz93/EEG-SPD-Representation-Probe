#!/usr/bin/env python3
"""Compute fixed quantitative diagnostics in original Log-Euclidean space."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (
    deterministic_silhouette,
    deterministic_trial_folds,
    grouped_logistic_probe,
    rms_distance_ratio,
    transition_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def normalize_metadata(frame: pd.DataFrame, expected_rows: int, windowed: bool) -> pd.DataFrame:
    rename = {}
    if "class" in frame and "class_label" not in frame:
        rename["class"] = "class_label"
    if "label" in frame and "class_label" not in frame:
        rename["label"] = "class_label"
    frame = frame.rename(columns=rename).reset_index(drop=True)
    required = {"subject", "session", "trial_id", "trial_uid", "class_label"}
    if windowed:
        required.add("window_index")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} metadata rows, observed {len(frame)}")
    frame["subject"] = frame.subject.astype(int)
    frame["trial_id"] = frame.trial_id.astype(int)
    if windowed:
        frame["window_index"] = frame.window_index.astype(int)
    return frame


def balanced_window_subset(
    metadata: pd.DataFrame, max_points: int, seed: int
) -> np.ndarray:
    """Select whole trials, balanced over subject×class, for global silhouettes."""
    if max_points >= len(metadata):
        return np.arange(len(metadata), dtype=np.int64)
    if max_points < 5:
        raise ValueError("WINDOW5 silhouette cap must contain at least one trial")
    max_trials = max_points // 5
    trials = metadata[
        ["trial_uid", "subject", "class_label"]
    ].drop_duplicates("trial_uid")
    if len(trials) * 5 != len(metadata):
        raise ValueError("WINDOW5 metadata does not contain exactly five rows per trial")
    strata = list(trials.groupby(["subject", "class_label"], sort=True))
    base, remainder = divmod(max_trials, len(strata))
    selected: list[str] = []
    for stratum_index, (_, frame) in enumerate(strata):
        take = base + int(stratum_index < remainder)
        if take > len(frame):
            raise ValueError("Silhouette cap requests more trials than a stratum contains")
        ranked = frame.assign(
            _digest=[
                hashlib.sha256(f"{seed}|silhouette|{uid}".encode()).hexdigest()
                for uid in frame.trial_uid
            ]
        ).sort_values(["_digest", "trial_uid"])
        selected.extend(ranked.trial_uid.iloc[:take].astype(str))
    mask = metadata.trial_uid.astype(str).isin(selected).to_numpy()
    indices = np.flatnonzero(mask)
    if len(indices) != max_trials * 5:
        raise AssertionError("Trial-balanced silhouette subset has an unexpected size")
    return indices


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    grand = float(values.mean())
    total = float(np.square(values - grand).sum())
    if total == 0.0:
        return 0.0
    between = sum(
        int((labels == label).sum())
        * float(values[labels == label].mean() - grand) ** 2
        for label in pd.unique(labels)
    )
    return float(np.clip(between / total, 0.0, 1.0))


def add_separability(
    rows: list[dict[str, Any]],
    representation: str,
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    silhouette_indices: np.ndarray,
    windowed: bool,
) -> None:
    targets = [
        ("class_silhouette", metadata.class_label.to_numpy()),
        ("subject_silhouette", metadata.subject.to_numpy()),
    ]
    if windowed:
        targets.append(("window_silhouette", metadata.window_index.to_numpy()))
    for metric, labels in targets:
        score, used = deterministic_silhouette(
            coordinates, labels, indices=silhouette_indices
        )
        rows.append(
            {
                "representation": representation,
                "window_index": pd.NA,
                "metric": metric,
                "value": score,
                "n_samples": len(used),
                "n_groups": len(pd.unique(labels[used])),
                "space": "log_svec_253d_unscaled",
                "subset_rule": "all" if len(used) == len(coordinates) else "balanced_whole_trials",
            }
        )
    for metric, labels in [
        ("class_separation_ratio", metadata.class_label.to_numpy()),
        ("subject_separation_ratio", metadata.subject.to_numpy()),
    ]:
        rows.append(
            {
                "representation": representation,
                "window_index": pd.NA,
                "metric": metric,
                "value": rms_distance_ratio(coordinates, labels),
                "n_samples": len(coordinates),
                "n_groups": len(pd.unique(labels)),
                "space": "log_svec_253d_unscaled",
                "subset_rule": "exact_all_rows_pairwise_identity",
            }
        )
    if windowed:
        for window in range(1, 6):
            mask = metadata.window_index.to_numpy() == window
            labels = metadata.loc[mask, "class_label"].to_numpy()
            score, used = deterministic_silhouette(coordinates[mask], labels)
            rows.extend(
                [
                    {
                        "representation": representation,
                        "window_index": window,
                        "metric": "class_silhouette",
                        "value": score,
                        "n_samples": len(used),
                        "n_groups": len(pd.unique(labels)),
                        "space": "log_svec_253d_unscaled",
                        "subset_rule": "all_trials_at_window",
                    },
                    {
                        "representation": representation,
                        "window_index": window,
                        "metric": "class_separation_ratio",
                        "value": rms_distance_ratio(coordinates[mask], labels),
                        "n_samples": int(mask.sum()),
                        "n_groups": len(pd.unique(labels)),
                        "space": "log_svec_253d_unscaled",
                        "subset_rule": "exact_all_trials_at_window",
                    },
                ]
            )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    cache_dir = resolve(ROOT, config["project"]["cache_dir"])
    output_dir = resolve(ROOT, config["project"]["output_dir"])
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    coordinate_path = cache_dir / "log_coordinates.npz"
    if not coordinate_path.exists():
        raise FileNotFoundError(
            f"Missing {coordinate_path}; run scripts/03_make_embeddings.py first"
        )
    with np.load(coordinate_path) as arrays:
        coordinates = {
            "whole_raw": np.asarray(arrays["whole_raw"], dtype=np.float64),
            "whole_centered": np.asarray(arrays["whole_centered"], dtype=np.float64),
            "window5_raw": np.asarray(arrays["window5_raw"], dtype=np.float64),
            "window5_centered": np.asarray(arrays["window5_centered"], dtype=np.float64),
        }
    if any(matrix.shape[1] != 253 for matrix in coordinates.values()):
        raise ValueError("Every diagnostic coordinate matrix must have 253 columns")
    whole_metadata = normalize_metadata(
        pd.read_csv(cache_dir / "whole_metadata.csv"), len(coordinates["whole_raw"]), False
    )
    window_metadata = normalize_metadata(
        pd.read_csv(cache_dir / "window5_metadata.csv"), len(coordinates["window5_raw"]), True
    )

    seed = int(config["project"]["seed"])
    max_silhouette = int(config["diagnostics"]["silhouette_max_samples"])
    whole_indices = np.arange(len(whole_metadata), dtype=np.int64)
    window_indices = balanced_window_subset(window_metadata, max_silhouette, seed)
    pd.DataFrame(
        {
            "row_index": window_indices,
            "trial_uid": window_metadata.iloc[window_indices].trial_uid.to_numpy(),
            "window_index": window_metadata.iloc[window_indices].window_index.to_numpy(),
        }
    ).to_csv(tables_dir / "window5_silhouette_subset.csv", index=False)

    separability_rows: list[dict[str, Any]] = []
    for representation, matrix in coordinates.items():
        is_window = representation.startswith("window5")
        add_separability(
            separability_rows,
            representation,
            matrix,
            window_metadata if is_window else whole_metadata,
            window_indices if is_window else whole_indices,
            is_window,
        )
    separability = pd.DataFrame.from_records(separability_rows)
    separability["window_index"] = separability["window_index"].astype("Int64")
    separability.to_csv(tables_dir / "separability_metrics.csv", index=False)

    probe_config = config["diagnostics"]["linear_probe"]
    n_splits = int(probe_config["folds"])
    whole_folds = deterministic_trial_folds(whole_metadata, n_splits, seed)
    fold_table = whole_metadata[
        ["subject", "session", "trial_id", "trial_uid", "class_label"]
    ].copy()
    fold_table.insert(0, "fold", whole_folds)
    fold_table.to_csv(tables_dir / "trial_folds.csv", index=False)
    fold_map = dict(zip(fold_table.trial_uid.astype(str), whole_folds, strict=True))
    expected_window_folds = window_metadata.trial_uid.astype(str).map(fold_map).to_numpy()
    observed_window_folds = deterministic_trial_folds(window_metadata, n_splits, seed)
    if not np.array_equal(expected_window_folds, observed_window_folds):
        raise AssertionError("WHOLE and WINDOW5 trial fold assignments differ")

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[pd.DataFrame] = []
    window_probe_rows: list[pd.DataFrame] = []
    probe_audit: dict[str, Any] = {}
    for representation, matrix in coordinates.items():
        metadata = window_metadata if representation.startswith("window5") else whole_metadata
        for target_column, target_name, chance in [
            ("class_label", "class", 0.25),
            ("subject", "subject", 1.0 / 9.0),
        ]:
            print(f"Linear information probe: {representation} -> {target_name}", flush=True)
            result = grouped_logistic_probe(
                matrix,
                metadata,
                target_column,
                n_splits=n_splits,
                seed=seed,
                c=float(probe_config["c"]),
                max_iter=int(probe_config["max_iter"]),
            )
            detail = result["fold_metrics"].copy()
            detail.insert(0, "target", target_name)
            detail.insert(0, "representation", representation)
            fold_rows.append(detail)
            if not result["per_window_metrics"].empty:
                per_window = result["per_window_metrics"].copy()
                per_window.insert(0, "target", target_name)
                per_window.insert(0, "representation", representation)
                window_probe_rows.append(per_window)
            is_window = representation.startswith("window5")
            primary_column = (
                "trial_mean_probability_accuracy" if is_window else "point_accuracy"
            )
            fold_accuracies = detail[primary_column].to_numpy(dtype=float)
            summary = result["summary"]
            primary_accuracy = (
                summary["pooled_oof_trial_mean_probability_accuracy"]
                if is_window
                else summary["pooled_oof_point_accuracy"]
            )
            summary_rows.append(
                {
                    "representation": representation,
                    "target": target_name,
                    "accuracy_mean": float(primary_accuracy),
                    "accuracy_std": float(fold_accuracies.std(ddof=1)),
                    "chance_level": chance,
                    "n_folds": n_splits,
                    "n_trials": int(summary["n_trials"]),
                    "n_points": int(summary["n_rows"]),
                    "aggregation": "trial_mean_probability" if is_window else "single_covariance",
                    "pooled_oof_point_accuracy": float(summary["pooled_oof_point_accuracy"]),
                    "any_convergence_warning": bool(summary["any_convergence_warning"]),
                    "model": summary["model"],
                    "c": float(probe_config["c"]),
                    "max_iter": int(probe_config["max_iter"]),
                    "standard_scaler": False,
                    "space": "log_svec_253d_unscaled",
                }
            )
            probe_audit[f"{representation}_{target_name}"] = summary
    pd.DataFrame.from_records(summary_rows).to_csv(
        tables_dir / "linear_probe_metrics.csv", index=False
    )
    pd.concat(fold_rows, ignore_index=True).to_csv(
        tables_dir / "linear_probe_fold_metrics.csv", index=False
    )
    if window_probe_rows:
        pd.concat(window_probe_rows, ignore_index=True).to_csv(
            tables_dir / "linear_probe_window_metrics.csv", index=False
        )

    transition = transition_diagnostics(
        coordinates["window5_raw"],
        coordinates["window5_centered"],
        window_metadata,
    )
    per_transition = transition["per_transition"].copy()
    per_transition["l2_norm"] = per_transition["transition_magnitude"]
    per_transition.to_csv(tables_dir / "transition_magnitudes.csv", index=False)
    transition["class_summary"].to_csv(
        tables_dir / "transition_class_summary.csv", index=False
    )
    transition["subject_summary"].to_csv(
        tables_dir / "transition_subject_summary.csv", index=False
    )
    transition["class_mean_cosine"].to_csv(
        tables_dir / "transition_cosine_similarity.csv", index=False
    )
    transition["eta_squared"].to_csv(
        tables_dir / "transition_eta_squared.csv", index=False
    )
    pair_summary = (
        per_transition.groupby("window_pair", sort=True)["transition_magnitude"]
        .agg([("mean_transition_magnitude", "mean"), ("std_transition_magnitude", "std"), ("n_transitions", "size")])
        .reset_index()
    )
    pair_summary.to_csv(tables_dir / "transition_pair_summary.csv", index=False)

    trial_columns = ["subject", "session", "trial_id", "class_label"]
    trial_transition = (
        per_transition.groupby(trial_columns, sort=False)["transition_magnitude"]
        .mean()
        .reset_index(name="mean_transition_magnitude")
    )
    class_means = trial_transition.groupby("class_label")[
        "mean_transition_magnitude"
    ].mean()
    overall_mean = float(trial_transition.mean_transition_magnitude.mean())
    class_range_fraction = float(
        (class_means.max() - class_means.min())
        / max(abs(overall_mean), np.finfo(float).eps)
    )
    overall_class_eta = eta_squared(
        trial_transition.mean_transition_magnitude.to_numpy(),
        trial_transition.class_label.to_numpy(),
    )
    overall_subject_eta = eta_squared(
        trial_transition.mean_transition_magnitude.to_numpy(),
        trial_transition.subject.to_numpy(),
    )
    effects = pd.DataFrame.from_records(
        [
            {"metric": "class_eta_squared", "value": overall_class_eta},
            {"metric": "subject_eta_squared", "value": overall_subject_eta},
            {"metric": "class_mean_range_fraction", "value": class_range_fraction},
            {"metric": "overall_mean", "value": overall_mean},
            {
                "metric": "centered_invariance_max_abs",
                "value": transition["validation"]["max_coordinate_absolute_delta_error"],
            },
        ]
    )
    effects.to_csv(tables_dir / "transition_effects.csv", index=False)

    diagnostics_summary = {
        "coordinate_space": "253-dimensional unscaled svec(log(C))",
        "seed": seed,
        "silhouette": {
            "whole_points": int(len(whole_indices)),
            "window5_points": int(len(window_indices)),
            "window5_trials": int(len(window_indices) // 5),
            "window5_subset_rule": "subject-class balanced whole trials",
        },
        "linear_probes": probe_audit,
        "transitions": {
            **transition["validation"],
            "class_eta_squared_trial_mean_magnitude": overall_class_eta,
            "subject_eta_squared_trial_mean_magnitude": overall_subject_eta,
            "class_mean_range_fraction": class_range_fraction,
            "overall_mean_transition_magnitude": overall_mean,
        },
    }
    (tables_dir / "diagnostics_summary.json").write_text(
        json.dumps(diagnostics_summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved original-space diagnostics under {tables_dir}")


if __name__ == "__main__":
    main()

