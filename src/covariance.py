"""WHOLE and fixed five-window OAS covariance construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances

from src.data import project_path, write_json


def estimate_oas_covariance(
    eeg: np.ndarray,
    *,
    symmetrize: bool = True,
) -> np.ndarray:
    """Estimate one channel covariance from a ``channels x time`` epoch."""

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"Expected channels x time data, got shape {eeg.shape}")
    if eeg.shape[0] < 1 or eeg.shape[1] < 2:
        raise ValueError("Covariance estimation needs channels and at least two samples")
    if not np.isfinite(eeg).all():
        raise ValueError("EEG segment contains NaN or Inf")
    # pyRiemann 0.12's named ``oas`` implementation delegates to
    # sklearn.covariance.oas.  Keeping this wrapper makes the estimator used
    # here identical to the batched implementation below.
    covariance = Covariances(estimator="oas").transform(eeg[np.newaxis])[0]
    if symmetrize:
        covariance = 0.5 * (covariance + covariance.T)
    return np.asarray(covariance, dtype=np.float64)


def equal_window_slices(
    n_samples: int,
    n_windows: int,
    *,
    remainder_policy: str = "require_exact_division",
) -> list[slice]:
    """Return deterministic, ordered, non-overlapping equal window slices."""

    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if remainder_policy != "require_exact_division":
        raise ValueError(f"Unsupported remainder policy: {remainder_policy}")
    quotient, remainder = divmod(int(n_samples), int(n_windows))
    if remainder:
        raise ValueError(
            f"{n_samples} samples cannot be divided exactly into {n_windows} windows"
        )
    if quotient < 2:
        raise ValueError("Each window needs at least two time samples")
    return [slice(index * quotient, (index + 1) * quotient) for index in range(n_windows)]


def build_covariance_representations(
    epochs: np.ndarray,
    *,
    n_windows: int = 5,
    estimator: str = "oas",
    symmetrize: bool = True,
    remainder_policy: str = "require_exact_division",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Build trial-major WHOLE and flattened trial-major WINDOW covariances."""

    epochs = np.asarray(epochs)
    if epochs.ndim != 3:
        raise ValueError(f"Expected trials x channels x time, got {epochs.shape}")
    if estimator.lower() != "oas":
        raise ValueError("The frozen pipeline only supports the OAS estimator")
    n_trials, n_channels, n_samples = epochs.shape
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    slices = equal_window_slices(
        n_samples, n_windows, remainder_policy=remainder_policy
    )
    whole = np.empty((n_trials, n_channels, n_channels), dtype=np.float64)
    windowed = np.empty(
        (n_trials * n_windows, n_channels, n_channels), dtype=np.float64
    )
    window_samples = slices[0].stop - slices[0].start
    covariance_transformer = Covariances(estimator="oas")
    for start in range(0, n_trials, batch_size):
        stop = min(start + batch_size, n_trials)
        trial_batch = np.asarray(epochs[start:stop], dtype=np.float64)
        whole_batch = covariance_transformer.transform(trial_batch)
        # Exact division means this is a pure reshape into temporal order.
        # Transposing before flattening yields trial-major rows w1..w5.
        window_batch = (
            trial_batch.reshape(stop - start, n_channels, n_windows, window_samples)
            .transpose(0, 2, 1, 3)
            .reshape((stop - start) * n_windows, n_channels, window_samples)
        )
        window_covariance_batch = covariance_transformer.transform(window_batch)
        if symmetrize:
            whole_batch = 0.5 * (whole_batch + whole_batch.transpose(0, 2, 1))
            window_covariance_batch = 0.5 * (
                window_covariance_batch + window_covariance_batch.transpose(0, 2, 1)
            )
        whole[start:stop] = whole_batch
        windowed[start * n_windows : stop * n_windows] = window_covariance_batch
    return whole, windowed


def make_covariance_metadata(
    trial_metadata: pd.DataFrame,
    n_windows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create metadata aligned exactly with the covariance array rows."""

    required = {"subject", "session", "trial_id", "class_label"}
    missing = required - set(trial_metadata.columns)
    if missing:
        raise ValueError(f"Trial metadata is missing columns: {sorted(missing)}")
    whole = trial_metadata.reset_index(drop=True).copy()
    whole.insert(0, "covariance_index", np.arange(len(whole), dtype=np.int64))
    windowed = whole.loc[whole.index.repeat(n_windows)].reset_index(drop=True)
    windowed["window_index"] = np.tile(
        np.arange(1, n_windows + 1, dtype=np.int64), len(whole)
    )
    windowed["covariance_index"] = np.arange(len(windowed), dtype=np.int64)
    return whole, windowed


def covariance_sanity_table(
    covariances: np.ndarray,
    metadata: pd.DataFrame,
    representation: str,
    *,
    symmetry_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Compute row-level numerical checks without dropping invalid matrices."""

    covariances = np.asarray(covariances)
    if covariances.ndim != 3 or covariances.shape[1] != covariances.shape[2]:
        raise ValueError(f"Expected N x C x C covariances, got {covariances.shape}")
    if len(covariances) != len(metadata):
        raise ValueError("Covariance and metadata row counts differ")

    n_matrices = len(covariances)
    has_nan = np.isnan(covariances).any(axis=(1, 2))
    has_inf = np.isinf(covariances).any(axis=(1, 2))
    finite = ~(has_nan | has_inf)
    numerator = np.linalg.norm(covariances - covariances.transpose(0, 2, 1), axis=(1, 2))
    denominator = np.maximum(
        np.linalg.norm(covariances, axis=(1, 2)), np.finfo(np.float64).tiny
    )
    symmetry_error = numerator / denominator
    min_eig = np.full(n_matrices, np.nan, dtype=np.float64)
    max_eig = np.full(n_matrices, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices):
        eigenvalues = np.linalg.eigvalsh(covariances[finite_indices])
        min_eig[finite_indices] = eigenvalues[:, 0]
        max_eig[finite_indices] = eigenvalues[:, -1]
    condition_number = np.full(n_matrices, np.inf, dtype=np.float64)
    positive = finite & (min_eig > 0.0)
    condition_number[positive] = max_eig[positive] / min_eig[positive]
    is_spd = finite & (symmetry_error <= symmetry_tolerance) & positive

    result = metadata.reset_index(drop=True).copy()
    result.insert(0, "representation", str(representation).upper())
    if "window_index" not in result:
        result["window_index"] = pd.array([pd.NA] * len(result), dtype="Int64")
    result["symmetry_error"] = symmetry_error
    result["min_eig"] = min_eig
    result["max_eig"] = max_eig
    result["condition_number"] = condition_number
    result["is_spd"] = is_spd
    result["has_nan"] = has_nan
    result["has_inf"] = has_inf
    preferred = [
        "representation",
        "subject",
        "session",
        "trial_id",
        "class_label",
        "window_index",
        "symmetry_error",
        "min_eig",
        "max_eig",
        "condition_number",
        "is_spd",
        "has_nan",
        "has_inf",
    ]
    remainder = [column for column in result.columns if column not in preferred]
    return result[preferred + remainder]


def _sanity_summary(table: pd.DataFrame) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for representation, frame in table.groupby("representation", sort=False):
        grouped[str(representation)] = {
            "count": int(len(frame)),
            "spd_count": int(frame["is_spd"].sum()),
            "non_spd_count": int((~frame["is_spd"]).sum()),
            "nan_count": int(frame["has_nan"].sum()),
            "inf_count": int(frame["has_inf"].sum()),
            "min_eigenvalue": float(frame["min_eig"].min()),
            "max_eigenvalue": float(frame["max_eig"].max()),
            "max_condition_number": float(frame["condition_number"].max()),
            "max_symmetry_error": float(frame["symmetry_error"].max()),
        }
    return grouped


def build_and_save_covariances(
    config: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Load prepared epochs, build both representations, and save all checks."""

    root = Path(project_root).expanduser().resolve()
    cache_dir = project_path(root, config["project"]["cache_dir"])
    output_dir = project_path(root, config["project"]["output_dir"])
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = cache_dir / "prepared_epochs.npz"
    metadata_path = cache_dir / "prepared_metadata.csv"
    if not prepared_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Prepared data are missing; run scripts/01_prepare_bnci.py first"
        )

    with np.load(prepared_path, allow_pickle=False) as prepared:
        epochs = prepared["X"]
        labels = prepared["y"].astype(str)
        channel_names = prepared["channel_names"].astype(str)
    metadata = pd.read_csv(metadata_path, dtype={"session": str, "run": str})
    if len(labels) != len(metadata) or not np.array_equal(
        labels, metadata["class_label"].astype(str).to_numpy()
    ):
        raise RuntimeError("Cached labels and metadata are not aligned")
    expected_channels = np.asarray(config["dataset"]["eeg_channels"], dtype=str)
    if not np.array_equal(channel_names, expected_channels):
        raise RuntimeError("Cached channel order does not match the frozen config")

    representation = config["representation"]
    n_windows = int(representation["n_windows"])
    expected_window_samples = int(representation["expected_window_samples"])
    if epochs.shape[-1] // n_windows != expected_window_samples:
        raise RuntimeError("Cached epochs do not yield the configured window length")
    whole, windowed = build_covariance_representations(
        epochs,
        n_windows=n_windows,
        estimator=str(representation["covariance_estimator"]),
        symmetrize=bool(representation["symmetrize_covariance"]),
        remainder_policy=str(representation["remainder_policy"]),
    )
    whole_metadata, window_metadata = make_covariance_metadata(metadata, n_windows)
    if len(whole) != len(whole_metadata) or len(windowed) != len(window_metadata):
        raise RuntimeError("Constructed covariance arrays and metadata are misaligned")

    covariance_path = cache_dir / "covariances.npz"
    np.savez(
        covariance_path,
        whole=whole,
        window5=windowed,
        channel_names=channel_names,
    )
    whole_metadata_path = cache_dir / "whole_metadata.csv"
    window_metadata_path = cache_dir / "window5_metadata.csv"
    whole_metadata.to_csv(whole_metadata_path, index=False)
    window_metadata.to_csv(window_metadata_path, index=False)

    sanity = pd.concat(
        [
            covariance_sanity_table(whole, whole_metadata, "WHOLE"),
            covariance_sanity_table(windowed, window_metadata, "WINDOW5"),
        ],
        ignore_index=True,
    )
    sanity_path = table_dir / "covariance_sanity.csv"
    sanity.to_csv(sanity_path, index=False)
    summary = {
        "estimator": str(representation["covariance_estimator"]),
        "estimator_implementation": (
            "pyriemann.estimation.Covariances(estimator='oas'); "
            "pyRiemann delegates to sklearn.covariance.oas"
        ),
        "oas_assume_centered": False,
        "extra_regularization": str(
            representation["covariance_extra_regularization"]
        ),
        "symmetrized": bool(representation["symmetrize_covariance"]),
        "whole_shape": [int(value) for value in whole.shape],
        "window5_shape": [int(value) for value in windowed.shape],
        "n_windows": n_windows,
        "window_samples": expected_window_samples,
        "remainder_policy": str(representation["remainder_policy"]),
        "sanity": _sanity_summary(sanity),
    }
    summary_path = table_dir / "covariance_summary.json"
    write_json(summary, summary_path)
    return {
        "covariances": str(covariance_path),
        "whole_metadata": str(whole_metadata_path),
        "window5_metadata": str(window_metadata_path),
        "sanity_csv": str(sanity_path),
        "summary_json": str(summary_path),
        "summary": summary,
    }
