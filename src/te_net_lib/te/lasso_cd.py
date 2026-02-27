from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def soft_threshold(x: float, t: float) -> float:
    """
    Apply the scalar soft-thresholding operator.

    Parameters
    ----------

    x:
        Input scalar.

    t:
        Non-negative threshold.

    Returns
    -------

    float

        sign(x) * max(|x| - t, 0).

    Examples
    --------
    >>> from te_net_lib.te.lasso_cd import soft_threshold
    >>> soft_threshold(3.0, 1.0)
    2.0
    >>> soft_threshold(-3.0, 1.0)
    -2.0
    >>> soft_threshold(0.5, 1.0)
    0.0
    """
    if x > t:
        return x - t
    if x < -t:
        return x + t
    return 0.0


@dataclass(frozen=True, slots=True)
class LassoCdOut:
    """
    Output of coordinate-descent Lasso optimization for a single response.

    Attributes
    ----------

    coef:
        Estimated coefficient vector with shape (p,).

    n_iter:
        Number of coordinate-descent passes performed.

    max_delta:
        Maximum absolute coefficient update in the final iteration.

    """

    coef: np.ndarray
    n_iter: int
    max_delta: float


def lasso_cd(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
) -> LassoCdOut:
    """
    Solve a Lasso regression problem using coordinate descent.

    The objective is:
        (1 / (2n)) * ||y - X b||_2^2 + alpha * ||b||_1

    If `alpha == 0`, this function returns the ordinary least squares solution via
    `np.linalg.lstsq`.

    Parameters
    ----------

    X:
        Design matrix with shape (n, p).

    y:
        Response vector with shape (n,).

    alpha:
        L1 penalty strength (non-negative).

    max_iter:
        Maximum number of coordinate-descent passes.

    tol:
        Convergence tolerance on the maximum absolute coefficient update.

    Returns
    -------

    LassoCdOut

        Dataclass containing coefficients and convergence diagnostics.

    Raises
    ------

    ValueError

        If shapes are invalid, `alpha` is negative, `max_iter` is not positive,
        `tol` is negative, or a column of X has zero variance when alpha > 0.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.te.lasso_cd import lasso_cd
    >>> g = np.random.default_rng(0)
    >>> X = g.normal(size=(100, 5)).astype(np.float64)
    >>> y = g.normal(size=(100,)).astype(np.float64)
    >>> out = lasso_cd(X, y, 0.1, 200, 1e-8)
    >>> out.coef.shape
    (5,)
    >>> out.n_iter >= 1
    True
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("X and y size mismatch")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    Xf = X.astype(np.float64, copy=False)
    yf = y.astype(np.float64, copy=False)

    if alpha == 0.0:
        coef, residuals, rank, s = np.linalg.lstsq(Xf, yf, rcond=None)
        return LassoCdOut(
            coef=coef.astype(np.float64, copy=False), n_iter=1, max_delta=0.0
        )

    z = (Xf * Xf).sum(axis=0) / float(n)
    if np.any(z <= 0.0):
        raise ValueError("columns of X must have nonzero variance")

    b = np.zeros((p,), dtype=np.float64)
    r = yf.copy()

    it = 0
    max_delta = 0.0

    for it in range(1, max_iter + 1):
        max_delta = 0.0
        for j in range(p):
            xj = Xf[:, j]
            bj_old = b[j]
            rho = (xj @ (r + bj_old * xj)) / float(n)
            bj_new = soft_threshold(float(rho), float(alpha)) / float(z[j])
            delta = bj_new - bj_old
            if delta != 0.0:
                r -= delta * xj
                b[j] = bj_new
                ad = abs(delta)
                if ad > max_delta:
                    max_delta = ad
        if max_delta <= tol:
            break

    return LassoCdOut(coef=b, n_iter=int(it), max_delta=float(max_delta))
