from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from te_net_lib.preprocess import FactorNeutralOut, factor_neutralize_svd
from te_net_lib.te.linear_ols import ols_te_matrix
from te_net_lib.te.lasso_te import lasso_te_matrix


@dataclass(frozen=True, slots=True)
class TeRawNeutralOut:
    """
    Combined output for "raw" and "factor-neutral" TE estimation.

    Attributes
    ----------

    beta_raw:
        TE coefficient matrix estimated from the raw return panel, shape (N, N).

    beta_neutral:
        TE coefficient matrix estimated from the factor-neutralized return panel,
        shape (N, N).

    intercept_raw:
        Optional intercept vector for the raw regression, shape (N,) or None.

    intercept_neutral:
        Optional intercept vector for the neutral regression, shape (N,) or None.

    preprocess:
        The preprocessing output, including neutral returns and estimated factors.

    Notes
    -----

    This is an algorithmic convenience wrapper only. It does not perform any I/O or
    run management. It exists to reduce call-site drift in experiment code.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.te.pipeline import ols_raw_and_neutral
    >>> g = np.random.default_rng(0)
    >>> R = g.normal(size=(120, 6)).astype(np.float64)
    >>> out = ols_raw_and_neutral(R, 1, True, True, 2, True)
    >>> out.beta_raw.shape
    (6, 6)
    >>> out.preprocess.factors_hat.shape
    (120, 2)
    """

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
    """
    Compute OLS TE on both raw returns and factor-neutralized returns.

    Parameters
    ----------

    returns:
        Return panel with shape (T, N).

    lag:
        Positive lag used by the TE estimator.

    add_intercept:
        If True, include an intercept in each per-target regression.

    exclude_self:
        If True, exclude self-lag regressors per target and enforce diagonal zeros.

    n_components:
        Number of SVD components removed by factor-neutral preprocessing.

    center:
        If True, center the return panel before SVD.

    Returns
    -------

    TeRawNeutralOut

        Dataclass containing raw/neutral beta matrices and preprocessing output.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.te.pipeline import ols_raw_and_neutral
    >>> g = np.random.default_rng(1)
    >>> R = g.normal(size=(100, 5)).astype(np.float64)
    >>> out = ols_raw_and_neutral(R, 1, False, False, 3, True)
    >>> out.beta_neutral.shape
    (5, 5)
    """
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
    """
    Compute Lasso TE on both raw returns and factor-neutralized returns.

    Parameters
    ----------

    returns:
        Return panel with shape (T, N).

    lag:
        Positive lag used by the TE estimator.

    alpha:
        L1 penalty strength for the per-target Lasso regressions.

    max_iter:
        Maximum coordinate-descent passes per target.

    tol:
        Convergence tolerance on maximum absolute coefficient update.

    add_intercept:
        If True, include an intercept by centering per regression.

    standardize:
        If True, standardize predictors within each regression.

    exclude_self:
        If True, exclude self-lag regressors and enforce diagonal zeros.

    n_components:
        Number of SVD components removed by factor-neutral preprocessing.

    center:
        If True, center the return panel before SVD.

    Returns
    -------

    TeRawNeutralOut

        Dataclass containing raw/neutral beta matrices and preprocessing output.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.te.pipeline import lasso_raw_and_neutral
    >>> g = np.random.default_rng(2)
    >>> R = g.normal(size=(160, 7)).astype(np.float64)
    >>> out = lasso_raw_and_neutral(R, 1, 0.1, 400, 1e-8, True, True, True, 2, True)
    >>> out.beta_raw.shape
    (7, 7)
    """
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
