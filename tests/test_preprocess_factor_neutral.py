import numpy as np

from te_net_lib.preprocess import factor_neutralize_svd


def test_factor_neutral_shapes_and_reproducible(N, T):
    g = np.random.default_rng(23)
    R = g.normal(size=(T, N)).astype(np.float64)
    out1 = factor_neutralize_svd(R, 3, True)
    out2 = factor_neutralize_svd(R, 3, True)
    assert np.allclose(out1.returns_neutral, out2.returns_neutral)
    assert out1.returns_neutral.shape == (T, N)
    assert out1.factors.shape == (T, 3)
    assert out1.components.shape == (3, N)
    assert out1.singular_values.shape == (3,)
    assert out1.explained_variance_ratio.shape == (3,)
    assert out1.mean_.shape == (N,)


def test_factor_neutral_orthogonality(N, T):
    g = np.random.default_rng(29)
    R = g.normal(size=(T, N)).astype(np.float64)
    out = factor_neutralize_svd(R, 2, True)
    M = out.factors.T @ out.returns_neutral
    assert float(np.abs(M).max()) < 1e-10


def test_factor_neutral_k0(N, T):
    g = np.random.default_rng(31)
    R = g.normal(size=(T, N)).astype(np.float64)
    out = factor_neutralize_svd(R, 0, True)
    assert out.factors.shape == (T, 0)
    assert out.components.shape == (0, N)
    assert out.singular_values.shape == (0,)
    assert out.explained_variance_ratio.shape == (0,)
