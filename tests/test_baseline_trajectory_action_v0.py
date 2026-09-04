import inspect
import numpy as np

from src.baseline_trajectory_v0.action import (
    PERMUTATIONS, _mean_components, action_error, deterministic_starts,
    select_semantic_permutation,
)


def test_all_permutations_and_determinant_coverage():
    assert len(PERMUTATIONS) == 24
    rng = np.random.default_rng(9)
    a = np.stack([np.eye(3) * (i + 1) for i in range(6)])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    b = q.T @ a @ q
    starts = deterministic_starts(a, b)
    assert len(starts) == 4
    assert sum(np.linalg.det(x) < 0 for x in starts) == 2
    assert sum(np.linalg.det(x) > 0 for x in starts) == 2
    assert min(action_error(a, b, x) for x in starts) < 1e-20


def test_zero_label_selector_api_has_no_target_labels():
    assert "y_target" not in inspect.signature(select_semantic_permutation).parameters


def test_tangent_component_mean_is_arithmetic_not_spd_mean():
    values = np.stack([np.eye(2), -np.eye(2), 2 * np.eye(2), -2 * np.eye(2)])
    labels = np.arange(4)
    means = _mean_components(values, labels, spd=False)
    assert np.array_equal(means, values)


def test_known_semantic_permutation_recovery_and_wrong_rank():
    rng = np.random.default_rng(55)
    source = []
    for label in range(4):
        trajectory = []
        for time in range(5):
            x = rng.normal(size=(4, 4))
            trajectory.append((x + x.T) / 2 + (3 * label + time) * np.eye(4))
        source.append(trajectory)
    source = np.asarray(source)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    hidden_order = np.array([2, 0, 3, 1])
    target = (q.T @ source @ q)[hidden_order]
    selected, ranked, _ = select_semantic_permutation(source, target, workers=4)
    assert selected == tuple(np.argsort(hidden_order))
    assert len(ranked) == 24
    assert ranked[1]["score"] > ranked[0]["score"] + 1e-6
