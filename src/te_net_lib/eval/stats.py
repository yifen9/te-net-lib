from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Ols1dOut:
    beta0: float
    beta1: float
    se_beta1: float
    t_beta1: float
    dof: int
    r2: float


def ols_1d(
    y: np.ndarray,
    x: np.ndarray,
    add_intercept: bool,
) -> Ols1dOut:
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("y and x must be 1D")
    if y.shape[0] != x.shape[0]:
        raise ValueError("y and x must have same length")
    n = int(y.shape[0])
    if n <= 2:
        raise ValueError("need more observations")

    yy = y.astype(np.float64, copy=False)
    xx = x.astype(np.float64, copy=False)

    if add_intercept:
        X = np.column_stack([np.ones((n,), dtype=np.float64), xx])
    else:
        X = xx.reshape(n, 1)

    coef, residuals, rank, s = np.linalg.lstsq(X, yy, rcond=None)
    yhat = X @ coef
    e = yy - yhat

    p = int(X.shape[1])
    dof = n - p
    if dof <= 0:
        raise ValueError("degrees of freedom must be positive")

    sse = float(e @ e)
    s2 = sse / float(dof)

    XtX = X.T @ X
    XtX_pinv = np.linalg.pinv(XtX)
    cov = s2 * XtX_pinv

    if add_intercept:
        b0 = float(coef[0])
        b1 = float(coef[1])
        se_b1 = float(np.sqrt(max(cov[1, 1], 0.0)))
    else:
        b0 = 0.0
        b1 = float(coef[0])
        se_b1 = float(np.sqrt(max(cov[0, 0], 0.0)))

    t_b1 = b1 / se_b1 if se_b1 > 0.0 else 0.0

    ym = float(yy.mean())
    tss = float(((yy - ym) @ (yy - ym)))
    r2 = 1.0 - (sse / tss) if tss > 0.0 else 0.0

    return Ols1dOut(
        beta0=b0, beta1=b1, se_beta1=se_b1, t_beta1=t_b1, dof=int(dof), r2=float(r2)
    )


def cross_sectional_tstat(
    returns: np.ndarray,
    signal: np.ndarray,
    add_intercept: bool,
) -> float:
    if returns.ndim != 2:
        raise ValueError("returns must be 2D")
    T, N = returns.shape
    if signal.ndim != 1 or signal.shape[0] != N:
        raise ValueError("signal must be 1D with length N")
    mu = returns.astype(np.float64, copy=False).mean(axis=0)
    out = ols_1d(mu, signal.astype(np.float64, copy=False), add_intercept)
    return float(out.t_beta1)


def rejection_rate(
    stats: np.ndarray,
    threshold: float,
    two_sided: bool,
) -> float:
    if stats.ndim != 1:
        raise ValueError("stats must be 1D")
    s = stats.astype(np.float64, copy=False)
    if two_sided:
        return float((np.abs(s) > threshold).mean())
    return float((s > threshold).mean())


def power_curve_from_null(
    null_stats: np.ndarray,
    alt_stats: np.ndarray,
    alphas: Sequence[float],
    two_sided: bool,
) -> np.ndarray:
    if null_stats.ndim != 1 or alt_stats.ndim != 1:
        raise ValueError("stats must be 1D")
    a = np.array(list(alphas), dtype=np.float64)
    if np.any(a <= 0.0) or np.any(a >= 1.0):
        raise ValueError("alphas must be in (0, 1)")
    null_s = null_stats.astype(np.float64, copy=False)
    alt_s = alt_stats.astype(np.float64, copy=False)

    out = np.zeros((a.shape[0],), dtype=np.float64)
    for i, alpha in enumerate(a.tolist()):
        if two_sided:
            crit = float(np.quantile(np.abs(null_s), 1.0 - alpha))
            out[i] = rejection_rate(alt_s, crit, True)
        else:
            crit = float(np.quantile(null_s, 1.0 - alpha))
            out[i] = rejection_rate(alt_s, crit, False)
    return out
