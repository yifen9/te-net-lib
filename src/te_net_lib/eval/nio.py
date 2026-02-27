from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SignalOut:
    """
    Signals derived from a directed adjacency matrix.

    Conventions
    -----------
    This library uses the standard adjacency convention:

        adj[i, j] corresponds to a directed edge i -> j.

    Therefore:
        out_strength[i] = sum_j adj[i, j]   (row sum)
        in_strength[i]  = sum_j adj[j, i]   (column sum)
        nio[i]          = out_strength[i] - in_strength[i]

    If normalize=True, nio is divided by (N-1) for cross-N comparability.

    Attributes
    ----------

    nio:
        Net information outflow (NIO), shape (N,).

    out_strength:
        Outgoing edge strength, shape (N,).

    in_strength:
        Incoming edge strength, shape (N,).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.nio import compute_nio
    >>> N = 5
    >>> A = np.zeros((N, N), dtype=np.int8)
    >>> A[0, 1] = 1
    >>> A[0, 2] = 1
    >>> A[3, 0] = 1
    >>> out = compute_nio(A, exclude_self=True, normalize=True)
    >>> out.out_strength.tolist()
    [2.0, 0.0, 0.0, 1.0, 0.0]
    >>> out.in_strength.tolist()
    [1.0, 1.0, 1.0, 0.0, 0.0]
    """

    nio: np.ndarray
    out_strength: np.ndarray
    in_strength: np.ndarray


def compute_nio(adj: np.ndarray, exclude_self: bool, normalize: bool) -> SignalOut:
    """
    Compute NIO (net information outflow) from a directed adjacency matrix.

    Conventions
    -----------

    adj[i, j] corresponds to i -> j.


    Definitions
    -----------

    out_i = sum_j adj[i, j]
    in_i  = sum_j adj[j, i]
    nio_i = out_i - in_i

    Parameters
    ----------

    adj:
        Adjacency matrix with shape (N, N). Nonzero values are treated as edge weights.

    exclude_self:
        If True, ignore diagonal entries.

    normalize:
        If True, divide nio by (N-1).

    Returns
    -------

    SignalOut

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.nio import compute_nio
    >>> A = np.zeros((3, 3), dtype=np.int8)
    >>> A[0, 1] = 1
    >>> A[1, 2] = 2
    >>> out = compute_nio(A, exclude_self=True, normalize=False)
    >>> out.out_strength.tolist()
    [1.0, 2.0, 0.0]
    >>> out.in_strength.tolist()
    [0.0, 1.0, 2.0]
    >>> out.nio.tolist()
    [1.0, 1.0, -2.0]
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square 2D")

    a = adj.astype(np.float64, copy=False)

    if exclude_self:
        b = a.copy()
        np.fill_diagonal(b, 0.0)
        out_s = b.sum(axis=1)
        in_s = b.sum(axis=0)
    else:
        out_s = a.sum(axis=1)
        in_s = a.sum(axis=0)

    nio = out_s - in_s
    if normalize:
        N = adj.shape[0]
        denom = float(max(N - 1, 1))
        nio = nio / denom

    return SignalOut(
        nio=nio.astype(np.float64, copy=False),
        out_strength=out_s.astype(np.float64, copy=False),
        in_strength=in_s.astype(np.float64, copy=False),
    )


def hub_recovery_from_signal(
    true_signal: np.ndarray, estimated_signal: np.ndarray, topk: int
) -> float:
    """
    Fractional overlap between top-k indices of true vs estimated signals.

    Parameters
    ----------

    true_signal:
        True signal vector, shape (N,).

    estimated_signal:
        Estimated signal vector, shape (N,).

    topk:
        Number of top entries to compare. If topk <= 0, returns 0.0.

    Returns
    -------
    float

        |TopK(true) ∩ TopK(est)| / k, with k = min(topk, N).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.nio import hub_recovery_from_signal
    >>> true_sig = np.array([0.0, 1.0, 5.0, 2.0], dtype=np.float64)
    >>> pred_sig = np.array([0.0, 2.0, 4.0, 1.0], dtype=np.float64)
    >>> hub_recovery_from_signal(true_sig, pred_sig, 2)
    1.0
    >>> hub_recovery_from_signal(true_sig, pred_sig, 0)
    0.0
    """
    if true_signal.ndim != 1 or estimated_signal.ndim != 1:
        raise ValueError("signals must be 1D")
    if true_signal.shape != estimated_signal.shape:
        raise ValueError("signals must have same shape")

    if topk <= 0:
        return 0.0

    k = min(int(topk), int(true_signal.size))
    if k <= 0:
        return 0.0

    true_idx = np.argpartition(true_signal, -k)[-k:]
    est_idx = np.argpartition(estimated_signal, -k)[-k:]
    return float(np.intersect1d(true_idx, est_idx).size) / float(k)
