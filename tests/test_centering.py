import numpy as np

from src.centering import centered_mean_max_abs, subject_center


def test_each_subject_centered_mean_is_zero():
    rng = np.random.default_rng(20260809)
    subjects = np.repeat(np.arange(1, 10), 20)
    offsets = rng.normal(size=(9, 253)) * 4.0
    coordinates = rng.normal(size=(len(subjects), 253)) + offsets[subjects - 1]
    centered, means = subject_center(coordinates, subjects)

    assert len(means) == 9
    assert centered_mean_max_abs(centered, subjects) < 1e-12


def test_window_centering_uses_one_mean_per_subject_not_per_window():
    subjects = np.repeat([1, 2], 10)
    windows = np.tile(np.arange(1, 6), 4)
    coordinates = np.column_stack([subjects * 10.0 + windows, windows])
    centered, _ = subject_center(coordinates, subjects)

    assert centered_mean_max_abs(centered, subjects) < 1e-12
    window_means = [centered[windows == window].mean(axis=0) for window in range(1, 6)]
    assert not np.allclose(window_means, 0.0)

