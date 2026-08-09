import numpy as np

from src.spd_utils import matrix_log_spd, svec, svec_dimension


def test_svec_dimension_for_22_channels():
    assert svec_dimension(22) == 253
    assert svec(np.eye(22)).shape == (253,)


def test_svec_preserves_frobenius_inner_product():
    rng = np.random.default_rng(20260809)
    for _ in range(20):
        a = rng.normal(size=(22, 22))
        b = rng.normal(size=(22, 22))
        a = 0.5 * (a + a.T)
        b = 0.5 * (b + b.T)
        assert np.allclose(np.dot(svec(a), svec(b)), np.trace(a @ b), atol=1e-10)


def test_matrix_log_is_symmetric_and_matches_diagonal_case():
    diagonal = np.linspace(0.2, 3.0, 22)
    logged = matrix_log_spd(np.diag(diagonal))
    assert np.allclose(logged, logged.T, atol=1e-14)
    assert np.allclose(np.diag(logged), np.log(diagonal), atol=1e-12)


def test_matrix_log_rejects_non_spd():
    matrix = np.eye(3)
    matrix[0, 0] = 0.0
    with np.testing.assert_raises(ValueError):
        matrix_log_spd(matrix)

