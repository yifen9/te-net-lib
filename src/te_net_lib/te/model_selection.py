from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from te_net_lib.te.lasso_te import lasso_te_matrix


@dataclass(frozen=True, slots=True)
class LassoPathOut:
    """
    Output of a Lasso TE coefficient path over a grid of penalty strengths.

    Attributes
    ----------

    alphas:
        Array of alpha values used, shape (K,).

    betas:
        Stacked beta matrices, shape (K, N, N), where betas[k][j, i] is i -> j.

    intercepts:
        Optional stacked intercepts, shape (K, N), present if `add_intercept=True`.

    n_iters:
        Iteration counts per alpha and per target regression, shape (K, N).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.te.model_selection import lasso_te_path
    >>> g = np.random.default_rng(0)
    >>> R = g.normal(size=(120, 5)).astype(np.float64)
    >>> out = lasso_te_path(R, 1, [0.2, 0.1, 0.0], 400, 1e-8, False, True, True)
    >>> out.betas.shape
    (3, 5, 5)
    """

    alphas: np.ndarray
    betas: np.ndarray
    intercepts: np.ndarray | None
    n_iters: np.ndarray


def lasso_te_path(
    returns: np.ndarray,
    lag: int,
    alphas: Sequence[float],
    max_iter: int,
    tol: float,
    add_intercept: bool,
    standardize: bool,
    exclude_self: bool,
) -> LassoPathOut:
    """
    Compute a Lasso TE coefficient path over a user-provided alpha grid.

    This function evaluates `lasso_te_matrix` for each alpha in `alphas` and stacks
    results into a single object. No model-selection logic is performed here; selection
    rules belong in experiment code.

    Parameters
    ----------

    returns:
        Return panel with shape (T, N).

    lag:
        Positive lag used by the TE estimator.

    alphas:
        Sequence of non-negative penalty strengths.

    max_iter:
        Maximum coordinate-descent passes per target regression.

    tol:
        Convergence tolerance on maximum absolute coefficient update.

    add_intercept:
        If True, include intercepts in each regression.

    standardize:
        If True, standardize predictors within each regression.

    exclude_self:
        If True, exclude self-lag regressors and enforce diagonal zeros.

    Returns
    -------

    LassoPathOut

        Dataclass containing alpha grid, beta path, optional intercept path, and
        iteration counts.

    Raises
    ------

    ValueError

        If alpha grid is empty, not 1D, or contains negative values.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.te.model_selection import lasso_te_path
    >>> g = np.random.default_rng(1)
    >>> R = g.normal(size=(100, 4)).astype(np.float64)
    >>> out = lasso_te_path(R, 1, [0.3, 0.05], 300, 1e-8, True, True, False)
    >>> out.intercepts.shape
    (2, 4)
    """
    a = np.array(list(alphas), dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("alphas must be 1D")
    if a.size == 0:
        raise ValueError("alphas must be non-empty")
    if np.any(a < 0.0):
        raise ValueError("alphas must be non-negative")

    T, N = returns.shape
    betas = np.zeros((a.size, N, N), dtype=np.float64)
    intercepts = np.zeros((a.size, N), dtype=np.float64) if add_intercept else None
    n_iters = np.zeros((a.size, N), dtype=np.int64)

    for i, alpha in enumerate(a.tolist()):
        out = lasso_te_matrix(
            returns,
            lag,
            float(alpha),
            max_iter,
            tol,
            add_intercept,
            standardize,
            exclude_self,
        )
        betas[i] = out.beta
        n_iters[i] = out.n_iter
        if add_intercept:
            intercepts[i] = out.intercept

    return LassoPathOut(alphas=a, betas=betas, intercepts=intercepts, n_iters=n_iters)
