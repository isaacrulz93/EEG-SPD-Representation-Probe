#!/usr/bin/env python3
"""Create Log-Euclidean coordinates, subject centering, and fixed embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.centering import centered_mean_max_abs, subject_center
from src.embedding import pca_tsne
from src.plotting import (
    example_trajectories,
    scatter_embedding,
    window_class_panels,
)
from src.spd_utils import log_svec, svec_dimension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def log_svec_in_batches(covariances: np.ndarray, batch_size: int = 512) -> np.ndarray:
    output = np.empty(
        (len(covariances), svec_dimension(covariances.shape[-1])), dtype=np.float64
    )
    for start in range(0, len(covariances), batch_size):
        stop = min(start + batch_size, len(covariances))
        output[start:stop] = log_svec(covariances[start:stop])
        print(f"log-svec {stop}/{len(covariances)}", flush=True)
    return output


def normalized_metadata(frame: pd.DataFrame, expected_rows: int, windowed: bool) -> pd.DataFrame:
    rename = {}
    if "class" in frame.columns and "class_label" not in frame.columns:
        rename["class"] = "class_label"
    if "label" in frame.columns and "class_label" not in frame.columns:
        rename["label"] = "class_label"
    frame = frame.rename(columns=rename).copy()
    required = {"subject", "session", "trial_id", "class_label", "trial_uid"}
    if windowed:
        required.add("window_index")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"Metadata rows {len(frame)} != covariance rows {expected_rows}")
    frame["subject"] = frame.subject.astype(int)
    frame["trial_id"] = frame.trial_id.astype(int)
    if windowed:
        frame["window_index"] = frame.window_index.astype(int)
    return frame


def add_embedding(metadata: pd.DataFrame, coordinates: np.ndarray) -> pd.DataFrame:
    frame = metadata.copy()
    frame.insert(0, "tsne2", coordinates[:, 1])
    frame.insert(0, "tsne1", coordinates[:, 0])
    return frame


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    cache_dir = resolve(ROOT, config["project"]["cache_dir"])
    output_dir = resolve(ROOT, config["project"]["output_dir"])
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    covariance_path = cache_dir / "covariances.npz"
    if not covariance_path.exists():
        raise FileNotFoundError(
            f"Missing {covariance_path}; run scripts/02_build_covariances.py first"
        )
    with np.load(covariance_path) as arrays:
        whole_covariances = np.asarray(arrays["whole"], dtype=np.float64)
        window_covariances = np.asarray(arrays["window5"], dtype=np.float64)

    whole_metadata = normalized_metadata(
        pd.read_csv(cache_dir / "whole_metadata.csv"), len(whole_covariances), False
    )
    window_metadata = normalized_metadata(
        pd.read_csv(cache_dir / "window5_metadata.csv"), len(window_covariances), True
    )

    expected_dimension = svec_dimension(len(config["dataset"]["eeg_channels"]))
    if expected_dimension != 253:
        raise ValueError(f"Frozen coordinate dimension is not 253: {expected_dimension}")

    whole_raw = log_svec_in_batches(whole_covariances)
    window_raw = log_svec_in_batches(window_covariances)
    whole_centered, whole_means = subject_center(
        whole_raw, whole_metadata.subject.to_numpy()
    )
    window_centered, window_means = subject_center(
        window_raw, window_metadata.subject.to_numpy()
    )
    whole_residual = centered_mean_max_abs(
        whole_centered, whole_metadata.subject.to_numpy()
    )
    window_residual = centered_mean_max_abs(
        window_centered, window_metadata.subject.to_numpy()
    )
    if whole_residual > 1e-10 or window_residual > 1e-10:
        raise AssertionError(
            f"Subject centering residual too large: {whole_residual}, {window_residual}"
        )

    np.savez_compressed(
        cache_dir / "log_coordinates.npz",
        whole_raw=whole_raw,
        whole_centered=whole_centered,
        window5_raw=window_raw,
        window5_centered=window_centered,
    )
    np.savez_compressed(
        cache_dir / "subject_means.npz",
        subjects=np.array(sorted(whole_means)),
        whole=np.stack([whole_means[key] for key in sorted(whole_means)]),
        window5=np.stack([window_means[key] for key in sorted(window_means)]),
    )

    seed = int(config["project"]["seed"])
    embedding_config = config["embedding"]
    specs = [
        ("whole_raw", whole_raw, whole_metadata),
        ("window5_raw", window_raw, window_metadata),
        ("whole_centered", whole_centered, whole_metadata),
        ("window5_centered", window_centered, window_metadata),
    ]
    embedded_frames: dict[str, pd.DataFrame] = {}
    embedding_metadata: dict[str, object] = {
        "coordinate_dimension": expected_dimension,
        "standard_scaler_applied": False,
        "subject_centering_max_abs_mean": {
            "whole": whole_residual,
            "window5": window_residual,
        },
        "representations": {},
    }
    for name, coordinates, metadata in specs:
        print(f"Fitting fixed PCA+t-SNE: {name}", flush=True)
        embedded, details = pca_tsne(coordinates, embedding_config, seed)
        frame = add_embedding(metadata, embedded)
        frame.to_csv(tables_dir / f"{name}_tsne_coordinates.csv", index=False)
        embedded_frames[name] = frame
        embedding_metadata["representations"][name] = details

    (tables_dir / "embedding_metadata.json").write_text(
        json.dumps(embedding_metadata, indent=2, sort_keys=True) + "\n"
    )

    scatter_embedding(
        embedded_frames["whole_raw"],
        "class_label",
        "WHOLE RAW — class",
        figures_dir / "figure_1a_whole_raw_class.png",
    )
    scatter_embedding(
        embedded_frames["whole_raw"],
        "subject",
        "WHOLE RAW — subject",
        figures_dir / "figure_1b_whole_raw_subject.png",
    )
    scatter_embedding(
        embedded_frames["window5_raw"],
        "class_label",
        "WINDOW5 RAW — class",
        figures_dir / "figure_2a_window5_raw_class.png",
    )
    scatter_embedding(
        embedded_frames["window5_raw"],
        "subject",
        "WINDOW5 RAW — subject",
        figures_dir / "figure_2b_window5_raw_subject.png",
    )
    scatter_embedding(
        embedded_frames["window5_raw"],
        "window_index",
        "WINDOW5 RAW — window index",
        figures_dir / "figure_2c_window5_raw_window.png",
    )
    window_class_panels(
        embedded_frames["window5_raw"],
        "WINDOW5 RAW — class by temporal window (one global embedding)",
        figures_dir / "figure_2d_window5_raw_class_panels.png",
    )
    scatter_embedding(
        embedded_frames["whole_centered"],
        "class_label",
        "WHOLE CENTERED — class",
        figures_dir / "figure_3a_whole_centered_class.png",
    )
    scatter_embedding(
        embedded_frames["whole_centered"],
        "subject",
        "WHOLE CENTERED — subject",
        figures_dir / "figure_3b_whole_centered_subject.png",
    )
    scatter_embedding(
        embedded_frames["window5_centered"],
        "class_label",
        "WINDOW5 CENTERED — class",
        figures_dir / "figure_4a_window5_centered_class.png",
    )
    scatter_embedding(
        embedded_frames["window5_centered"],
        "subject",
        "WINDOW5 CENTERED — subject",
        figures_dir / "figure_4b_window5_centered_subject.png",
    )
    scatter_embedding(
        embedded_frames["window5_centered"],
        "window_index",
        "WINDOW5 CENTERED — window index",
        figures_dir / "figure_4c_window5_centered_window.png",
    )
    window_class_panels(
        embedded_frames["window5_centered"],
        "WINDOW5 CENTERED — class by temporal window (one global embedding)",
        figures_dir / "figure_4d_window5_centered_class_panels.png",
    )

    selected_subjects = [int(x) for x in config["diagnostics"]["trajectory_subjects"]]
    raw_selected = example_trajectories(
        embedded_frames["window5_raw"],
        selected_subjects,
        "WINDOW5 RAW — deterministic example trajectories",
        figures_dir / "figure_5a_window5_raw_example_trajectories.png",
    )
    centered_selected = example_trajectories(
        embedded_frames["window5_centered"],
        selected_subjects,
        "WINDOW5 CENTERED — deterministic example trajectories",
        figures_dir / "figure_5b_window5_centered_example_trajectories.png",
    )
    if set(raw_selected.trial_uid) != set(centered_selected.trial_uid):
        raise AssertionError("RAW and CENTERED trajectory subsets differ")
    raw_selected[
        ["subject", "class_label", "session", "trial_id", "trial_uid"]
    ].drop_duplicates().sort_values(["subject", "class_label"]).to_csv(
        tables_dir / "example_trajectory_selection.csv", index=False
    )

    print(f"Saved coordinates, four embeddings, and figures under {output_dir}")


if __name__ == "__main__":
    main()

