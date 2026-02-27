import numpy as np
import pytest

from te_net_lib.te import lasso_te_path


def test_lasso_te_path_shapes_no_intercept():
    g = np.random.default_rng(1)
    T = 120
    N = 8
    R = g.normal(size=(T, N)).astype(np.float64)

    alphas = [0.2, 0.1, 0.0]
    out = lasso_te_path(R, 1, alphas, 500, 1e-8, False, True, True)

    assert out.alphas.shape == (3,)
    assert out.betas.shape == (3, N, N)
    assert out.intercepts is None
    assert out.n_iters.shape == (3, N)
    assert np.all(np.diag(out.betas[0]) == 0.0)


def test_lasso_te_path_shapes_with_intercept():
    g = np.random.default_rng(2)
    T = 140
    N = 6
    R = g.normal(size=(T, N)).astype(np.float64)

    alphas = [0.3, 0.05]
    out = lasso_te_path(R, 1, alphas, 400, 1e-8, True, True, False)

    assert out.intercepts is not None
    assert out.intercepts.shape == (2, N)


def test_lasso_te_path_alpha_monotone_sparsity_basic():
    g = np.random.default_rng(3)
    T = 150
    N = 7
    R = g.normal(size=(T, N)).astype(np.float64)

    alphas = [0.5, 0.1, 0.0]
    out = lasso_te_path(R, 1, alphas, 800, 1e-8, False, True, True)

    nz = np.array(
        [int(np.count_nonzero(out.betas[i])) for i in range(out.betas.shape[0])],
        dtype=np.int64,
    )
    assert nz[0] <= nz[1] <= nz[2]


def test_lasso_te_path_rejects_empty_alphas():
    g = np.random.default_rng(4)
    R = g.normal(size=(50, 5)).astype(np.float64)
    with pytest.raises(ValueError):
        lasso_te_path(R, 1, [], 200, 1e-8, False, False, False)
