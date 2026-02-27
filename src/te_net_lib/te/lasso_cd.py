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

    max_update:
        Maximum absolute coefficient update at termination.

    """

    coef: np.ndarray
    n_iter: int
    max_update: float


def lasso_cd(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    init_coef: np.ndarray | None = None,
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

    init_coef:
        Optional initial coefficient vector with shape (p,). If provided, coordinate descent
        starts from this vector (warm start). Ignored when alpha == 0.

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

    if init_coef is not None:
        if init_coef.ndim != 1:
            raise ValueError("init_coef must be 1D")
        if init_coef.shape[0] != p:
            raise ValueError("init_coef must have shape (p,)")

    Xf = X.astype(np.float64, copy=False)
    yf = y.astype(np.float64, copy=False)

    if p == 0:
        return LassoCdOut(
            coef=np.zeros((0,), dtype=np.float64), n_iter=0, max_update=0.0
        )

    if alpha == 0.0:
        coef = np.linalg.lstsq(Xf, yf, rcond=None)[0].astype(np.float64, copy=False)
        return LassoCdOut(coef=coef, n_iter=1, max_update=0.0)

    z = (Xf * Xf).sum(axis=0) / float(n)
    if np.any(z == 0.0):
        raise ValueError("X must have nonzero variance columns when alpha > 0")

    b = (
        init_coef.astype(np.float64, copy=True)
        if init_coef is not None
        else np.zeros(p, dtype=np.float64)
    )
    r = yf - Xf @ b

    max_update = 0.0
    n_iter = 0

    for it in range(max_iter):
        max_update = 0.0
        for j in range(p):
            bj_old = float(b[j])
            if bj_old != 0.0:
                r = r + Xf[:, j] * bj_old
            rho = float(Xf[:, j].dot(r) / float(n))
            bj_new = soft_threshold(rho, alpha) / float(z[j])
            b[j] = bj_new
            if bj_new != 0.0:
                r = r - Xf[:, j] * bj_new
            upd = abs(bj_new - bj_old)
            if upd > max_update:
                max_update = upd

        n_iter = it + 1
        if max_update <= tol:
            break

    return LassoCdOut(coef=b, n_iter=n_iter, max_update=float(max_update))
