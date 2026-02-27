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
    """
    Simulate a VAR(1) process with a user-provided directed adjacency pattern.

    The process is:
        x_t = A x_{t-1} + eps_t

    where eps_t are i.i.d. Gaussian innovations with scalar standard deviation
    `noise_scale`. The coefficient matrix A is constructed from `adj` by:

    - binarizing edges (nonzero entries indicate edges)
    - zeroing self-edges
    - normalizing each row by its out-degree (with protection for zero out-degree)
    - scaling by `coef_scale`
    - applying random +/- signs to edge weights

    Parameters
    ----------

    rng:
        NumPy Generator used for randomness (signs and innovations).

    N:
        Number of nodes/assets.

    T:
        Number of returned time steps after burn-in.

    adj:
        Adjacency-like matrix with shape (N, N). Nonzero entries indicate edges.
        Self-edges are ignored. Output ground truth is the binarized version.

    coef_scale:
        Global scaling for VAR coefficients derived from adjacency normalization.

    noise_scale:
        Standard deviation of Gaussian innovations.

    burnin:
        Number of initial steps discarded to reduce dependence on initialization.

    Returns
    -------

    DgpSample

    A sample with:

    - returns: array with shape (T, N)
    - true_adj: binarized adjacency with shape (N, N)
    - extras: dict containing "A" (the realized coefficient matrix)

    Raises
    ------

    ValueError

        If `adj` does not have shape (N, N) or `burnin` is negative.

    Notes
    -----

    The direction convention used by the library is:
        A[j, i] corresponds to i -> j (i influences j).

    This matches the regression layout used by TE estimators in `te_net_lib.te`,
    where beta[j, i] represents i -> j.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.dgp.gaussian import simulate_gaussian_var
    >>> g = np.random.default_rng(0)
    >>> N, T = 4, 50
    >>> adj = np.zeros((N, N), dtype=np.int8)
    >>> adj[0, 1] = 1
    >>> adj[2, 3] = 1
    >>> out = simulate_gaussian_var(g, N, T, adj, 0.3, 1.0, 10)
    >>> out.returns.shape
    (50, 4)
    >>> out.true_adj.shape
    (4, 4)
    """
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
