from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Ols1dOut:
    """
    Output of a univariate OLS regression y ~ x (optionally with intercept).

    Attributes
    ----------

    beta0:
        Intercept term (0.0 if add_intercept=False).

    beta1:
        Slope coefficient for x.

    se_beta1:
        Standard error estimate for beta1.

    t_beta1:
        t-statistic for beta1 (beta1 / se_beta1, or 0 if se_beta1 == 0).

    dof:
        Degrees of freedom (n - p).

    r2:
        Coefficient of determination.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.stats import ols_1d
    >>> x = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y = 2.0 + 3.0 * x
    >>> out = ols_1d(y, x, True)
    >>> round(out.beta1, 6)
    3.0
    """

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
    """
    Fit a 1D OLS regression and return slope t-statistic and diagnostics.

    This function uses least squares for coefficient estimation and a pseudo-inverse
    covariance estimate for robustness under rank deficiency (e.g., constant x).

    Parameters
    ----------

    y:
        Response vector with shape (n,).

    x:
        Predictor vector with shape (n,).

    add_intercept:
        If True, fit y = beta0 + beta1 x. If False, fit y = beta1 x.

    Returns
    -------

    Ols1dOut

        Dataclass containing coefficients, standard error, t-statistic, dof, and r2.

    Raises
    ------

    ValueError

        If inputs are not 1D, have different lengths, or n is too small.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.stats import ols_1d
    >>> g = np.random.default_rng(0)
    >>> x = g.normal(size=200)
    >>> y = 2.0 + 3.0 * x + g.normal(scale=0.5, size=200)
    >>> out = ols_1d(y, x, True)
    >>> abs(out.beta1 - 3.0) < 0.2
    True
    """
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
    """
    Compute a cross-sectional t-statistic of mean returns on a signal.

    This computes the time-series mean return per asset:
        mu_i = mean_t returns[t, i]

    and runs a 1D regression mu ~ signal to return the slope t-statistic.

    Parameters
    ----------

    returns:
        Return panel with shape (T, N).

    signal:
        Cross-sectional signal with shape (N,).

    add_intercept:
        If True, include an intercept in the cross-sectional regression.

    Returns
    -------

    float

        t-statistic for the signal slope coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.stats import cross_sectional_tstat
    >>> g = np.random.default_rng(0)
    >>> R = g.normal(size=(200, 10))
    >>> sig = np.zeros((10,))
    >>> abs(cross_sectional_tstat(R, sig, True)) < 1e-6
    True
    """
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
    """
    Compute empirical rejection rate for a 1D array of test statistics.

    Parameters
    ----------

    stats:
        Array of statistics with shape (M,).

    threshold:
        Critical value.

    two_sided:
        If True, reject when |stat| > threshold. If False, reject when stat > threshold.

    Returns
    -------

    float

        Fraction of rejected samples.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.stats import rejection_rate
    >>> x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    >>> rejection_rate(x, 2.0, True)
    0.4
    """
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
    """
    Estimate a power curve by deriving critical values from a null distribution.

    For each alpha in `alphas`, compute a critical value from `null_stats`:

    - two_sided=True: crit = quantile(|null|, 1 - alpha)
    - two_sided=False: crit = quantile(null, 1 - alpha)

    Then compute rejection rate of `alt_stats` using that critical value.

    Parameters
    ----------

    null_stats:
        Null statistics array with shape (M,).

    alt_stats:
        Alternative statistics array with shape (K,).

    alphas:
        Sequence of significance levels, each in (0, 1).

    two_sided:
        Whether to compute a two-sided critical value.

    Returns
    -------

    numpy.ndarray

        Power estimates with shape (len(alphas),).

    Raises
    ------

    ValueError

        If arrays are not 1D or alphas are not in (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.stats import power_curve_from_null
    >>> g = np.random.default_rng(0)
    >>> null = g.normal(size=2000)
    >>> alt = g.normal(size=2000) + 1.0
    >>> p = power_curve_from_null(null, alt, [0.1, 0.05, 0.01], False)
    >>> p.shape
    (3,)
    """
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
