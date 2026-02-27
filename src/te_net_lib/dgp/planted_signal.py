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
    """
    Simulate a VAR(1) process with a randomly planted directed graph (ground truth).

    This is intended for power/edge-recovery experiments where a "true" adjacency
    is available. The process is:
        x_t = A x_{t-1} + eps_t

    The adjacency is sampled i.i.d. Bernoulli(edge_prob) off-diagonal. Coefficients
    are constructed by:

    - normalizing each column by out-degree (with protection for zero out-degree)
    - scaling by `coef_strength`
    - applying random +/- signs

    Self-edges are always removed.

    Parameters
    ----------

    rng:
        NumPy Generator used to sample the adjacency, signs, and innovations.

    N:
        Number of nodes/assets.

    T:
        Number of returned time steps after burn-in.

    edge_prob:
        Probability of an off-diagonal edge (must be in [0, 1]).

    coef_strength:
        Global scaling for realized VAR coefficients.

    noise_scale:
        Standard deviation of Gaussian innovations.

    burnin:
        Number of initial steps discarded.

    Returns
    -------

    DgpSample

    A sample with:

    - returns: array with shape (T, N)
    - true_adj: planted adjacency with shape (N, N) and entries in {0, 1}
    - extras: dict containing "A" (the realized coefficient matrix) and "edge_prob"

    Raises
    ------

    ValueError

        If `edge_prob` is outside [0, 1] or `burnin` is negative.

    Notes
    -----

    Direction convention:
        true_adj[j, i] == 1 indicates i -> j (i influences j).

    This matches the TE coefficient layout used throughout the library.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.dgp.planted_signal import simulate_planted_signal_var
    >>> g = np.random.default_rng(0)
    >>> out = simulate_planted_signal_var(g, 5, 40, 0.2, 0.4, 1.0, 10)
    >>> out.returns.shape
    (40, 5)
    >>> out.true_adj.shape
    (5, 5)
    >>> int(np.diag(out.true_adj).sum())
    0
    """
    if not (0.0 <= edge_prob <= 1.0):
        raise ValueError("edge_prob must be in [0, 1]")
    if burnin < 0:
        raise ValueError("burnin must be non-negative")

    adj = (rng.random(size=(N, N)) < edge_prob).astype(np.int8)
    np.fill_diagonal(adj, 0)

    out_deg = adj.sum(axis=0).astype(np.float64)
    denom = np.maximum(out_deg, 1.0)

    A = (coef_strength * (adj.astype(np.float64) / denom[None, :])).astype(np.float64)
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
