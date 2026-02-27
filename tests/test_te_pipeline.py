import numpy as np

from te_net_lib.preprocess import factor_neutralize_svd
from te_net_lib.te import (
    lasso_raw_and_neutral,
    lasso_te_matrix,
    ols_raw_and_neutral,
    ols_te_matrix,
)


def test_ols_raw_and_neutral_shapes():
    g = np.random.default_rng(1)
    T = 120
    N = 9
    R = g.normal(size=(T, N)).astype(np.float64)
    out = ols_raw_and_neutral(R, 1, True, True, 2, True)
    assert out.beta_raw.shape == (N, N)
    assert out.beta_neutral.shape == (N, N)
    assert out.intercept_raw is not None
    assert out.intercept_neutral is not None
    assert out.intercept_raw.shape == (N,)
    assert out.intercept_neutral.shape == (N,)
    assert out.preprocess.returns_neutral.shape == (T, N)
    assert np.all(np.diag(out.beta_raw) == 0.0)
    assert np.all(np.diag(out.beta_neutral) == 0.0)


def test_ols_pipeline_matches_manual_calls():
    g = np.random.default_rng(2)
    T = 140
    N = 7
    R = g.normal(size=(T, N)).astype(np.float64)

    out = ols_raw_and_neutral(R, 1, False, False, 3, True)

    pre = factor_neutralize_svd(R, 3, True)
    raw = ols_te_matrix(R, 1, False, False).beta
    neu = ols_te_matrix(pre.returns_neutral, 1, False, False).beta

    assert np.allclose(out.beta_raw, raw)
    assert np.allclose(out.beta_neutral, neu)


def test_lasso_pipeline_matches_manual_calls():
    g = np.random.default_rng(3)
    T = 160
    N = 8
    R = g.normal(size=(T, N)).astype(np.float64)

    alpha = 0.05
    max_iter = 500
    tol = 1e-8
    add_intercept = True
    standardize = True
    exclude_self = True
    n_components = 2
    center = True

    out = lasso_raw_and_neutral(
        R,
        1,
        alpha,
        max_iter,
        tol,
        add_intercept,
        standardize,
        exclude_self,
        n_components,
        center,
    )

    pre = factor_neutralize_svd(R, n_components, center)
    raw = lasso_te_matrix(
        R, 1, alpha, max_iter, tol, add_intercept, standardize, exclude_self
    ).beta
    neu = lasso_te_matrix(
        pre.returns_neutral,
        1,
        alpha,
        max_iter,
        tol,
        add_intercept,
        standardize,
        exclude_self,
    ).beta

    assert np.allclose(out.beta_raw, raw)
    assert np.allclose(out.beta_neutral, neu)


def test_pipeline_k0_neutral_is_centered_or_raw():
    g = np.random.default_rng(4)
    T = 110
    N = 6
    R = g.normal(size=(T, N)).astype(np.float64)
    out_c = ols_raw_and_neutral(R, 1, False, False, 0, True)
    out_nc = ols_raw_and_neutral(R, 1, False, False, 0, False)

    pre_c = factor_neutralize_svd(R, 0, True).returns_neutral
    pre_nc = factor_neutralize_svd(R, 0, False).returns_neutral

    assert np.allclose(out_c.preprocess.returns_neutral, pre_c)
    assert np.allclose(out_nc.preprocess.returns_neutral, pre_nc)
