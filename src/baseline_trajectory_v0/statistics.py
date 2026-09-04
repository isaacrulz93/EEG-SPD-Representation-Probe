"""Frozen subject-level statistics and decision helpers."""

from __future__ import annotations

import itertools

import numpy as np


def exact_sign_flip_test(deltas: np.ndarray) -> float:
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("exact test requires nine finite subject deltas")
    observed = float(values.mean())
    statistics = [
        float(np.mean(values * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=9)
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def subject_bootstrap_ci(
    deltas: np.ndarray, *, replicates: int = 20000, seed: int = 20260904
) -> tuple[float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_summary(deltas: np.ndarray, *, tie_tolerance: float = 1e-12) -> dict[str, float | int]:
    values = np.asarray(deltas, dtype=np.float64)
    low, high = subject_bootstrap_ci(values)
    return {
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "wins": int(np.sum(values > tie_tolerance)),
        "losses": int(np.sum(values < -tie_tolerance)),
        "ties": int(np.sum(np.abs(values) <= tie_tolerance)),
        "p_raw": exact_sign_flip_test(values),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }


__all__ = ["exact_sign_flip_test", "holm_adjust", "paired_summary", "subject_bootstrap_ci"]
