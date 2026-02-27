from __future__ import annotations

import numpy as np

from te_net_lib.dgp.base import DgpSample


def simulate_garch_factor(
    rng: np.random.Generator,
    N: int,
    T: int,
    k: int,
    omega: float,
    alpha: float,
    beta: float,
    loading_scale: float,
    burnin: int,
) -> DgpSample:
    """
    Simulate a factor model with idiosyncratic GARCH(1,1) volatility per asset.

    The simulated return is:
        r_t = f_t B^T + eps_t

    where:

    - f_t is a k-dimensional standard normal factor
    - B is an (N, k) loading matrix
    - eps_t has asset-specific conditional variance following GARCH(1,1):
        sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

    Parameters
    ----------

    rng:
        NumPy Generator used to draw loadings, factor series, and shocks.

    N:
        Number of assets/nodes.

    T:
        Number of returned time steps after burn-in.

    k:
        Number of factors (must be positive).

    omega:
        GARCH constant term (must be positive).

    alpha:
        GARCH ARCH parameter (must satisfy alpha >= 0).

    beta:
        GARCH GARCH parameter (must satisfy beta >= 0 and alpha + beta < 1).

    loading_scale:
        Standard deviation used to sample loadings B.

    burnin:
        Number of initial steps discarded.

    Returns
    -------

    DgpSample

    A sample with:

    - returns: array with shape (T, N)
    - true_adj: None (this DGP does not define a ground-truth directed graph)
    - extras: dict containing "factors", "loadings", "sigma2", "eps", and "params"

    Raises
    ------

    ValueError

        If parameter constraints are violated.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.dgp.garch_factor import simulate_garch_factor
    >>> g = np.random.default_rng(0)
    >>> out = simulate_garch_factor(g, 6, 100, 2, 0.05, 0.05, 0.9, 0.3, 10)
    >>> out.returns.shape
    (100, 6)
    >>> out.extras["factors"].shape
    (100, 2)
    >>> out.true_adj is None
    True
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if burnin < 0:
        raise ValueError("burnin must be non-negative")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if alpha < 0.0 or beta < 0.0 or (alpha + beta) >= 1.0:
        raise ValueError("alpha and beta must satisfy alpha>=0, beta>=0, alpha+beta<1")

    total = T + burnin

    B = rng.normal(loc=0.0, scale=loading_scale, size=(N, k)).astype(np.float64)
    F = rng.normal(loc=0.0, scale=1.0, size=(total, k)).astype(np.float64)

    sigma2 = np.zeros((total, N), dtype=np.float64)
    eps = np.zeros((total, N), dtype=np.float64)

    s2_0 = omega / (1.0 - alpha - beta)
    sigma2[0, :] = s2_0

    z = rng.normal(loc=0.0, scale=1.0, size=(total, N)).astype(np.float64)
    eps[0, :] = np.sqrt(sigma2[0, :]) * z[0, :]

    for t in range(1, total):
        sigma2[t, :] = omega + alpha * (eps[t - 1, :] ** 2) + beta * sigma2[t - 1, :]
        eps[t, :] = np.sqrt(sigma2[t, :]) * z[t, :]

    R = (F @ B.T) + eps
    returns = R[burnin:].copy()

    extras = {
        "factors": F[burnin:].copy(),
        "loadings": B.copy(),
        "sigma2": sigma2[burnin:].copy(),
        "eps": eps[burnin:].copy(),
        "params": {
            "k": k,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "loading_scale": loading_scale,
        },
    }
    return DgpSample(returns=returns, true_adj=None, extras=extras)
