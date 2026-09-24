import itertools
import numpy as np
from src.adaptive_segmentation_v0 import (adaptive_boundaries, exact_sign_flip,
    matched_length_permutations, oas_cov, probe_sequence, svec, spd_function,
    valid_random_boundaries)
from src.evaluation_adaptive_segmentation_v0 import fit_score


def test_piecewise_probe_dp_matches_bruteforce_small_grid():
    # Strong transition at sample 250 on a 25-sample grid.
    centers = np.arange(12.0, 500, 25.0)
    vectors = np.zeros((len(centers), 1))
    vectors[centers >= 250, 0] = 8.0
    b = adaptive_boundaries(vectors, centers, 500, k=2, grid=25, min_length=125)
    assert 200 <= b[1] <= 300
    def objective(cut):
        out = 0.0
        for lo, hi in ((0, cut), (cut, 500)):
            z = vectors[(centers >= lo) & (centers < hi), 0]
            out += np.sum((z-z.mean())**2)
        return out
    brute = min(objective(c) for c in range(125, 376, 25))
    assert np.isclose(objective(b[1]), brute)


def test_spd_svec_shape_and_random_validity():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 1000))
    cov, vectors, centers = probe_sequence(x)
    assert cov.shape == (36, 4, 4)
    assert vectors.shape == (36, 10)
    assert centers.shape == (36,)
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.isfinite(svec(spd_function(oas_cov(x), "log"))).all()
    sets = valid_random_boundaries(seed=3, reps=20)
    for b in sets:
        assert b[0] == 0 and b[-1] == 1000 and np.diff(b).min() >= 125


def test_matched_lengths_and_exact_sign_flip():
    b = np.array([0, 175, 375, 575, 800, 1000])
    matched = matched_length_permutations(b, seed=7, max_reps=20)
    assert 1 <= len(matched) <= 20
    for m in matched:
        assert sorted(np.diff(m)) == sorted(np.diff(b))
        assert not np.array_equal(m, b)
    result = exact_sign_flip(np.ones(9))
    assert result["n_sign_configurations"] == 512 and result["p_one_sided"] == 1/512


def test_scaler_fit_uses_source_features_only():
    rng=np.random.default_rng(9)
    train=[rng.normal(loc=float(k),size=(3,8)) for k in range(4)]
    labels=np.arange(4)
    test=[rng.normal(size=(2,8)) for _ in range(3)]
    _,a=fit_score(train,labels,test,[[1,1],[1,1],[1,1]])
    altered=[v+10000 for v in test]
    _,b=fit_score(train,labels,altered,[[1,1],[1,1],[1,1]])
    assert a["scaler_mean_sha256"]==b["scaler_mean_sha256"]
    assert a["scaler_fit_segments"]==12
