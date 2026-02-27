from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from te_net_lib.te.lasso_cd import lasso_cd


@dataclass(frozen=True, slots=True)
class LassoTeOut:
    """
    Output of Lasso-based TE estimation (per-target sparse regression).

    Attributes
    ----------

    beta:
        Coefficient matrix with shape (N, N). Entry beta[j, i] corresponds to i -> j.
        When `exclude_self=True`, the diagonal is forced to zero.

    intercept:
        Optional intercept vector with shape (N,). Present if `add_intercept=True`.

    n_iter:
        Number of coordinate-descent passes used for each target regression, shape (N,).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.te.lasso_te import lasso_te_matrix
    >>> g = np.random.default_rng(0)
    >>> R = g.normal(size=(200, 6)).astype(np.float64)
    >>> out = lasso_te_matrix(R, 1, 0.05, 500, 1e-8, True, True, True)
    >>> out.beta.shape
    (6, 6)
    >>> out.n_iter.shape
    (6,)
    """

    beta: np.ndarray
    intercept: np.ndarray | None
    n_iter: np.ndarray


def lasso_te_matrix(
    returns: np.ndarray,
    lag: int,
    alpha: float,
    max_iter: int,
    tol: float,
    add_intercept: bool,
    standardize: bool,
    exclude_self: bool,
    init_beta: np.ndarray | None = None,
) -> LassoTeOut:
    """
    Estimate a sparse TE coefficient matrix using Lasso regressions.

    For each target j, the regression is:

        y = returns[lag:, j]
        X = returns[:-lag, :]

    with optional intercept removal and optional column standardization. The fitted
    coefficients are assembled into a matrix beta such that beta[j, i] corresponds
    to i -> j.

    Parameters
    ----------

    returns:
        Return panel with shape (T, N).

    lag:
        Positive lag used to define predictors X and targets Y.

    alpha:
        L1 penalty strength passed to the per-target Lasso solver.

    max_iter:
        Maximum coordinate-descent passes per target regression.

    tol:
        Convergence tolerance on maximum absolute coefficient update.

    add_intercept:
        If True, center y and X per target and return an intercept term.

    standardize:
        If True, scale each predictor column to have RMS 1 within the regression
        after optional centering, then rescale coefficients back.

    exclude_self:
        If True, exclude the target's own lagged predictor from the regression and
        set beta[j, j] = 0.

    init_beta:
        Optional initial beta matrix with shape (N, N). If provided, each target regression
        is warm-started from init_beta[j, :], respecting exclude_self and standardization.
        This is intended for computing alpha paths efficiently.

    Returns
    -------

    LassoTeOut

        Dataclass containing beta, optional intercept, and per-target iteration counts.

    Raises
    ------

    ValueError

        If input shapes or parameter constraints are violated.

    Notes
    -----

    Direction convention:
        beta[j, i] represents i -> j.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.te.lasso_te import lasso_te_matrix
    >>> g = np.random.default_rng(1)
    >>> R = g.normal(size=(150, 5)).astype(np.float64)
    >>> out = lasso_te_matrix(R, 1, 0.1, 300, 1e-8, False, True, True)
    >>> out.beta.shape
    (5, 5)
    """
    if returns.ndim != 2:
        raise ValueError("returns must be 2D")
    if lag <= 0:
        raise ValueError("lag must be positive")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    T, N = returns.shape

    if init_beta is not None:
        if init_beta.shape != (N, N):
            raise ValueError("init_beta shape must be (N, N)")

    if T <= lag:
        raise ValueError("T must be larger than lag")

    X = returns[:-lag, :].astype(np.float64, copy=False)
    Y = returns[lag:, :].astype(np.float64, copy=False)

    beta = np.zeros((N, N), dtype=np.float64)
    intercept = np.zeros((N,), dtype=np.float64) if add_intercept else None
    n_iter = np.zeros((N,), dtype=np.int64)

    for j in range(N):
        y = Y[:, j].astype(np.float64, copy=False)
        Xj = X

        if exclude_self:
            mask = np.ones((N,), dtype=bool)
            mask[j] = False
            Xj = X[:, mask]

        y_mean = 0.0
        x_mean = None
        if add_intercept:
            y_mean = float(y.mean())
            y = y - y_mean
            x_mean = Xj.mean(axis=0)
            Xj = Xj - x_mean

        scale = None
        if standardize:
            scale = np.sqrt((Xj * Xj).mean(axis=0))
            if np.any(scale == 0.0):
                raise ValueError("standardize=True but some predictors have zero scale")
            Xj = Xj / scale

        init_coef = None
        if init_beta is not None:
            init_row = init_beta[j].astype(np.float64, copy=False)
            init_coef = init_row[mask].copy() if exclude_self else init_row.copy()

        out = lasso_cd(Xj, y, alpha, max_iter, tol, init_coef=init_coef)
        b = out.coef.astype(np.float64, copy=False)

        if standardize:
            b = b / scale

        if exclude_self:
            beta[j, mask] = b
            beta[j, j] = 0.0
        else:
            beta[j, :] = b
            beta[j, j] = 0.0 if exclude_self else beta[j, j]

        if add_intercept:
            b0 = y_mean - float(x_mean.dot(b)) if x_mean is not None else y_mean
            intercept[j] = b0

        n_iter[j] = int(out.n_iter)

    return LassoTeOut(beta=beta, intercept=intercept, n_iter=n_iter)
