import numpy as np

from te_net_lib.dgp import simulate_garch_factor


def test_garch_factor_reproducible(N, T, burnin, g1, g2):
    x1 = simulate_garch_factor(g1, N, T, 2, 0.05, 0.05, 0.9, 0.3, burnin)
    x2 = simulate_garch_factor(g2, N, T, 2, 0.05, 0.05, 0.9, 0.3, burnin)
    assert np.allclose(x1.returns, x2.returns)
    assert x1.true_adj is None
    assert x2.true_adj is None


def test_garch_factor_shapes(N, T, burnin):
    g = np.random.default_rng(19)
    out = simulate_garch_factor(g, N, T, 3, 0.05, 0.05, 0.9, 0.2, burnin)
    assert out.returns.shape == (T, N)
    assert out.extras["factors"].shape == (T, 3)
    assert out.extras["loadings"].shape == (N, 3)
