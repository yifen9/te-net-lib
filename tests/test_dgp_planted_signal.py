import numpy as np

from te_net_lib.dgp import simulate_planted_signal_var


def test_planted_signal_reproducible(N, T, burnin, g1, g2):
    x1 = simulate_planted_signal_var(g1, N, T, 0.2, 0.4, 1.0, burnin)
    x2 = simulate_planted_signal_var(g2, N, T, 0.2, 0.4, 1.0, burnin)
    assert np.allclose(x1.returns, x2.returns)
    assert np.array_equal(x1.true_adj, x2.true_adj)


def test_planted_signal_adj_constraints(N, T, burnin):
    g = np.random.default_rng(11)
    out = simulate_planted_signal_var(g, N, T, 0.3, 0.4, 1.0, burnin)
    assert out.true_adj is not None
    assert np.all(np.diag(out.true_adj) == 0)
