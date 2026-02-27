import numpy as np

from te_net_lib.eval import (
    cross_sectional_tstat,
    ols_1d,
    power_curve_from_null,
    rejection_rate,
)


def test_ols_1d_recovers_slope():
    g = np.random.default_rng(1)
    n = 400
    x = g.normal(size=n).astype(np.float64)
    y = (2.0 + 3.0 * x + g.normal(scale=0.5, size=n)).astype(np.float64)
    out = ols_1d(y, x, True)
    assert abs(out.beta1 - 3.0) < 0.1
    assert out.dof == n - 2


def test_cross_sectional_tstat_zero_signal():
    g = np.random.default_rng(2)
    T = 200
    N = 20
    R = g.normal(size=(T, N)).astype(np.float64)
    sig = np.zeros((N,), dtype=np.float64)
    t = cross_sectional_tstat(R, sig, True)
    assert abs(t) < 1e-6


def test_rejection_rate_two_sided():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float64)
    r = rejection_rate(x, 2.0, True)
    assert abs(r - 0.4) < 1e-12


def test_power_curve_from_null_monotone():
    g = np.random.default_rng(3)
    null = g.normal(size=2000).astype(np.float64)
    alt = (g.normal(size=2000) + 1.0).astype(np.float64)
    alphas = [0.1, 0.05, 0.01]
    p = power_curve_from_null(null, alt, alphas, False)
    assert p.shape == (3,)
    assert p[0] >= p[1] >= p[2]
