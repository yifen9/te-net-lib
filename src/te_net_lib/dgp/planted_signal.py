from __future__ import annotations

import numpy as np

from te_net_lib.dgp.base import DgpSample


def simulate_planted_signal_var(
    rng: np.random.Generator,
    N: int,
    T: int,
    edge_prob: float,
    coef_strength: float,
    noise_scale: float,
    burnin: int,
) -> DgpSample:
    if not (0.0 <= edge_prob <= 1.0):
        raise ValueError("edge_prob must be in [0, 1]")
    if burnin < 0:
        raise ValueError("burnin must be non-negative")

    adj = (rng.random(size=(N, N)) < edge_prob).astype(np.int8)
    np.fill_diagonal(adj, 0)

    out_deg = adj.sum(axis=1).astype(np.float64)
    denom = np.maximum(out_deg, 1.0)

    A = (coef_strength * (adj.astype(np.float64) / denom[:, None])).astype(np.float64)
    sign = rng.choice(np.array([-1.0, 1.0]), size=(N, N))
    A = A * sign
    np.fill_diagonal(A, 0.0)

    total = T + burnin
    x = np.zeros((total, N), dtype=np.float64)
    eps = rng.normal(loc=0.0, scale=noise_scale, size=(total, N)).astype(np.float64)

    for t in range(1, total):
        x[t] = x[t - 1] @ A.T + eps[t]

    returns = x[burnin:].copy()
    extras = {"A": A, "edge_prob": edge_prob}
    return DgpSample(returns=returns, true_adj=adj, extras=extras)
