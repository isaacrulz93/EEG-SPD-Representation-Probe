"""Core numerical and API hard gates for the frozen geometry V2 protocol."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.geometry_v2 import (
    AIRM,
    EA,
    GEOMETRIES,
    LE,
    RAW,
    fit_center,
    logeuclidean_mean_custom,
    logeuclidean_mean_official,
    select_deterministic_pairs,
    smat,
    spd_diagnostics,
    spd_invsqrt,
    spd_log,
    symmetric_exp,
)
from src.spd_utils import svec


def _synthetic_spd(n_matrices: int = 18, n_channels: int = 5) -> np.ndarray:
    rng = np.random.default_rng(20260809)
    matrices = []
    for _ in range(n_matrices):
        factor = rng.normal(size=(n_channels, n_channels))
        matrices.append(factor @ factor.T + 0.75 * np.eye(n_channels))
    return np.asarray(matrices, dtype=np.float64)


def test_1_custom_and_official_logeuclidean_means_agree() -> None:
    covariances = _synthetic_spd()
    custom = logeuclidean_mean_custom(covariances)
    official = logeuclidean_mean_official(covariances)
    relative = np.linalg.norm(custom - official, ord="fro") / np.linalg.norm(
        official, ord="fro"
    )
    assert relative <= 1e-10


def test_stable_evd_functions_round_trip_and_never_clip() -> None:
    covariances = _synthetic_spd(n_matrices=4)
    logged = spd_log(covariances)
    reconstructed = symmetric_exp(logged)
    relative = np.linalg.norm(reconstructed - covariances, axis=(1, 2)) / np.linalg.norm(
        covariances, axis=(1, 2)
    )
    assert relative.max() <= 1e-10

    inverse_root = spd_invsqrt(covariances)
    whitened = inverse_root @ covariances @ inverse_root
    np.testing.assert_allclose(
        whitened, np.broadcast_to(np.eye(5), whitened.shape), rtol=1e-10, atol=1e-10
    )

    semidefinite = np.eye(3)
    semidefinite[-1, -1] = 0.0
    with pytest.raises(ValueError, match="strictly positive definite"):
        spd_log(semidefinite)
    with pytest.raises(ValueError, match="strictly positive definite"):
        spd_invsqrt(semidefinite)


def test_smat_exactly_inverts_the_v1_svec_convention() -> None:
    rng = np.random.default_rng(71)
    matrices = rng.normal(size=(7, 5, 5))
    matrices = 0.5 * (matrices + matrices.transpose(0, 2, 1))
    vectors = svec(matrices)
    np.testing.assert_allclose(smat(vectors), matrices, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(smat(svec(matrices[0])), matrices[0], atol=1e-15)
    with pytest.raises(ValueError, match="not a symmetric-vector dimension"):
        smat(np.zeros(8))


def test_spd_diagnostics_report_evd_and_visible_failures() -> None:
    valid = _synthetic_spd(n_matrices=2, n_channels=3)
    invalid = valid[0].copy()
    invalid[0, 0] = -1.0
    matrices = np.concatenate([valid, invalid[np.newaxis]], axis=0)
    diagnostics = spd_diagnostics(matrices)
    assert diagnostics["is_spd"].tolist() == [True, True, False]
    assert diagnostics["finite"].all()
    assert diagnostics["evd_reconstruction_error"][:2].max() <= 5e-12
    assert np.isfinite(diagnostics["condition_number"][:2]).all()
    assert np.isinf(diagnostics["condition_number"][-1])


def test_9_fit_api_is_label_free_and_pair_selection_is_row_invariant() -> None:
    signature = inspect.signature(fit_center)
    assert "y" not in signature.parameters
    assert "labels" not in signature.parameters
    assert set(GEOMETRIES) == {RAW, LE, AIRM, EA}
    covariances = _synthetic_spd(n_matrices=10, n_channels=3)
    with pytest.raises(TypeError):
        fit_center(covariances, LE, y=np.zeros(len(covariances)))  # type: ignore[call-arg]

    uids = np.asarray([f"S01_T{index:03d}" for index in range(12)])
    first = select_deterministic_pairs(uids, seed=20260809, subject=1, n_pairs=10)
    permutation = np.random.default_rng(3).permutation(len(uids))
    second = select_deterministic_pairs(
        uids[permutation], seed=20260809, subject=1, n_pairs=10
    )
    assert first.uid_pairs == second.uid_pairs
    assert first.digests == second.digests
    for row_indices, uid_pair in zip(second.indices, second.uid_pairs, strict=True):
        assert tuple(uids[permutation][row_indices]) == uid_pair

    raw = fit_center(covariances, RAW)
    assert raw.fit_sample_count == 0
    np.testing.assert_array_equal(raw.transform(covariances), covariances)
