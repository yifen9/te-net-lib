import numpy as np

from te_net_lib.te import lasso_te_matrix, ols_te_matrix


def _simulate_var_deterministic(A: np.ndarray, T: int, x0: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    X = np.zeros((T, n), dtype=np.float64)
    X[0] = x0.astype(np.float64)
    for t in range(1, T):
        X[t] = X[t - 1] @ A.T
    return X


def test_lasso_te_shapes():
    T = 90
    N = 7
    g = np.random.default_rng(10)
    R = g.normal(size=(T, N)).astype(np.float64)
    out = lasso_te_matrix(R, 1, 0.1, 200, 1e-8, True, True, True)
    assert out.beta.shape == (N, N)
    assert out.intercept is not None
    assert out.intercept.shape == (N,)
    assert out.n_iter.shape == (N,)
    assert np.all(np.diag(out.beta) == 0.0)


def test_lasso_te_alpha_large_zero_beta():
    T = 100
    N = 6
    g = np.random.default_rng(11)
    R = g.normal(size=(T, N)).astype(np.float64)
    out = lasso_te_matrix(R, 1, 1e6, 200, 1e-10, True, True, True)
    assert np.max(np.abs(out.beta)) == 0.0


def test_lasso_te_small_alpha_close_to_ols_on_deterministic_var():
    T = 220
    A = np.array(
        [
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, -0.3, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    x0 = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float64)
    R = _simulate_var_deterministic(A, T, x0)
    ols = ols_te_matrix(R, 1, False, False).beta
    las = lasso_te_matrix(R, 1, 0.0, 2000, 1e-10, False, False, False).beta
    assert np.max(np.abs(las - ols)) < 1e-8


def test_lasso_te_reproducible():
    T = 130
    N = 9
    g1 = np.random.default_rng(12)
    g2 = np.random.default_rng(12)
    R1 = g1.normal(size=(T, N)).astype(np.float64)
    R2 = g2.normal(size=(T, N)).astype(np.float64)
    out1 = lasso_te_matrix(R1, 1, 0.05, 500, 1e-8, True, True, True)
    out2 = lasso_te_matrix(R2, 1, 0.05, 500, 1e-8, True, True, True)
    assert np.allclose(out1.beta, out2.beta)
    assert np.allclose(out1.intercept, out2.intercept)
    assert np.array_equal(out1.n_iter, out2.n_iter)
