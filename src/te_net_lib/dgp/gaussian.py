from __future__ import annotations

import numpy as np

from te_net_lib.dgp.base import DgpSample


def simulate_gaussian_var(
    rng: np.random.Generator,
    N: int,
    T: int,
    adj: np.ndarray,
    coef_scale: float,
    noise_scale: float,
    burnin: int,
) -> DgpSample:
    if adj.shape != (N, N):
        raise ValueError("adj shape must be (N, N)")
    if burnin < 0:
        raise ValueError("burnin must be non-negative")

    adj_bin = (adj != 0).astype(np.float64)
    np.fill_diagonal(adj_bin, 0.0)

    out_deg = adj_bin.sum(axis=1)
    denom = np.maximum(out_deg, 1.0)
    A = coef_scale * (adj_bin / denom[:, None])

    sign = rng.choice(np.array([-1.0, 1.0]), size=(N, N))
    A = A * sign
    np.fill_diagonal(A, 0.0)

    total = T + burnin
    x = np.zeros((total, N), dtype=np.float64)
    eps = rng.normal(loc=0.0, scale=noise_scale, size=(total, N)).astype(np.float64)

    for t in range(1, total):
        x[t] = x[t - 1] @ A.T + eps[t]

    returns = x[burnin:].copy()
    extras = {"A": A}
    return DgpSample(returns=returns, true_adj=adj_bin.astype(np.int8), extras=extras)
