import numpy as np

from te_net_lib.dgp import simulate_gaussian_var


def test_gaussian_var_reproducible(N, T, burnin, g1, g2):
    adj = np.zeros((N, N), dtype=np.int8)
    adj[0, 1] = 1
    adj[2, 3] = 1
    x1 = simulate_gaussian_var(g1, N, T, adj, 0.3, 1.0, burnin).returns
    x2 = simulate_gaussian_var(g2, N, T, adj, 0.3, 1.0, burnin).returns
    assert np.allclose(x1, x2)


def test_gaussian_var_shapes(N, T, burnin):
    adj = np.eye(N, dtype=np.int8)
    g = np.random.default_rng(7)
    out = simulate_gaussian_var(g, N, T, adj, 0.2, 1.0, burnin)
    assert out.returns.shape == (T, N)
    assert out.true_adj is not None
    assert out.true_adj.shape == (N, N)
