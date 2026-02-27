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
