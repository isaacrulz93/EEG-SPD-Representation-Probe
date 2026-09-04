import inspect
import numpy as np

from src.baseline_trajectory_v0.geometry import (
    apply_state_permutations, baseline_relative_logs, congruence,
    fixed_nonidentity_permutations, gram_features, spd_invsqrt, spd_log,
    spd_sqrt, symmetric_exp,
)
from src.baseline_trajectory_v0.identifiability import assign_clusters_to_source
from src.spd_utils import svec


def spd(rng, n=22):
    a = rng.normal(size=(n, n))
    return a @ a.T + np.eye(n)


def test_svec_frobenius_isometry_and_spectral_reconstruction():
    rng = np.random.default_rng(3)
    a, b = spd(rng), spd(rng)
    assert np.allclose(np.dot(svec(a), svec(b)), np.sum(a * b))
    assert np.allclose(symmetric_exp(spd_log(a)), a, rtol=1e-10, atol=1e-10)
    assert np.allclose(spd_sqrt(a) @ spd_sqrt(a), a, rtol=1e-10, atol=1e-10)
    assert np.allclose(spd_invsqrt(a) @ a @ spd_invsqrt(a), np.eye(22), atol=1e-10)


def test_baseline_relative_identity_and_common_congruence_gram_invariance():
    rng = np.random.default_rng(4)
    c0 = np.stack([spd(rng)])
    local = np.repeat(c0[:, None], 5, axis=1)
    assert np.allclose(baseline_relative_logs(c0, local), 0, atol=1e-10)
    local = np.stack([[spd(rng) for _ in range(5)]])
    original = baseline_relative_logs(c0, local)
    g = rng.normal(size=(22, 22)) + 2 * np.eye(22)
    transformed = baseline_relative_logs(congruence(g, c0), congruence(g, local))
    assert np.allclose(gram_features(original), gram_features(transformed), atol=1e-8, rtol=1e-8)


def test_shuffle_multiset_reverse_involution_and_label_free_assignment_api():
    states = np.arange(2 * 5 * 2 * 2).reshape(2, 5, 2, 2)
    permutations = fixed_nonidentity_permutations(2, 7)
    shuffled = apply_state_permutations(states, permutations)
    assert np.array_equal(np.sort(shuffled.reshape(2, 5, -1), axis=1), np.sort(states.reshape(2, 5, -1), axis=1))
    assert np.array_equal(states[:, ::-1][:, ::-1], states)
    assert "y_target" not in inspect.signature(assign_clusters_to_source).parameters
