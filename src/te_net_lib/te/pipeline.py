from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from te_net_lib.preprocess import FactorNeutralOut, factor_neutralize_svd
from te_net_lib.te.linear_ols import ols_te_matrix
from te_net_lib.te.lasso_te import lasso_te_matrix


@dataclass(frozen=True, slots=True)
class TeRawNeutralOut:
    beta_raw: np.ndarray
    beta_neutral: np.ndarray
    intercept_raw: np.ndarray | None
    intercept_neutral: np.ndarray | None
    preprocess: FactorNeutralOut


def ols_raw_and_neutral(
    returns: np.ndarray,
    lag: int,
    add_intercept: bool,
    exclude_self: bool,
    n_components: int,
    center: bool,
) -> TeRawNeutralOut:
    pre = factor_neutralize_svd(returns, n_components, center)
    raw = ols_te_matrix(returns, lag, add_intercept, exclude_self)
    neu = ols_te_matrix(pre.returns_neutral, lag, add_intercept, exclude_self)
    return TeRawNeutralOut(
        beta_raw=raw.beta,
        beta_neutral=neu.beta,
        intercept_raw=raw.intercept,
        intercept_neutral=neu.intercept,
        preprocess=pre,
    )


def lasso_raw_and_neutral(
    returns: np.ndarray,
    lag: int,
    alpha: float,
    max_iter: int,
    tol: float,
    add_intercept: bool,
    standardize: bool,
    exclude_self: bool,
    n_components: int,
    center: bool,
) -> TeRawNeutralOut:
    pre = factor_neutralize_svd(returns, n_components, center)
    raw = lasso_te_matrix(
        returns, lag, alpha, max_iter, tol, add_intercept, standardize, exclude_self
    )
    neu = lasso_te_matrix(
        pre.returns_neutral,
        lag,
        alpha,
        max_iter,
        tol,
        add_intercept,
        standardize,
        exclude_self,
    )
    return TeRawNeutralOut(
        beta_raw=raw.beta,
        beta_neutral=neu.beta,
        intercept_raw=raw.intercept,
        intercept_neutral=neu.intercept,
        preprocess=pre,
    )
