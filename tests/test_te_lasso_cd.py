import numpy as np

from te_net_lib.te.lasso_cd import lasso_cd


def test_lasso_cd_shapes():
    g = np.random.default_rng(1)
    X = g.normal(size=(80, 10)).astype(np.float64)
    y = g.normal(size=(80,)).astype(np.float64)
    out = lasso_cd(X, y, 0.1, 200, 1e-8)
    assert out.coef.shape == (10,)
    assert out.n_iter >= 1


def test_lasso_cd_alpha_large_zeroes():
    g = np.random.default_rng(2)
    X = g.normal(size=(100, 8)).astype(np.float64)
    y = g.normal(size=(100,)).astype(np.float64)
    out = lasso_cd(X, y, 1e6, 200, 1e-10)
    assert np.max(np.abs(out.coef)) == 0.0


def test_lasso_cd_reproducible():
    g1 = np.random.default_rng(3)
    g2 = np.random.default_rng(3)
    X1 = g1.normal(size=(120, 12)).astype(np.float64)
    y1 = g1.normal(size=(120,)).astype(np.float64)
    X2 = g2.normal(size=(120, 12)).astype(np.float64)
    y2 = g2.normal(size=(120,)).astype(np.float64)
    out1 = lasso_cd(X1, y1, 0.05, 300, 1e-8)
    out2 = lasso_cd(X2, y2, 0.05, 300, 1e-8)
    assert np.allclose(out1.coef, out2.coef)
