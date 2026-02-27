from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SignalOut:
    """
    Output signals derived from a directed adjacency matrix.

    Attributes
    ----------

    nio:
        Net information outflow (NIO) signal, shape (N,).

    out_strength:
        Outgoing edge strength (row sum), shape (N,).

    in_strength:
        Incoming edge strength (column sum), shape (N,).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.nio import compute_nio
    >>> A = np.zeros((4, 4), dtype=np.int8)
    >>> A[0, 1] = 1
    >>> out = compute_nio(A, True, True)
    >>> out.nio.shape
    (4,)
    """

    nio: np.ndarray
    out_strength: np.ndarray
    in_strength: np.ndarray


def compute_nio(
    adj: np.ndarray,
    exclude_self: bool,
    normalize: bool,
) -> SignalOut:
    """
    Compute net information outflow (NIO) from a directed adjacency matrix.

    For an adjacency matrix A (binary or weighted), define:
        out_i = sum_j A[i, j]
        in_i  = sum_j A[j, i]
        nio_i = out_i - in_i

    If `normalize=True`, NIO is divided by (N-1) to scale into a comparable range.

    Parameters
    ----------

    adj:
        Adjacency matrix with shape (N, N). Nonzero entries are treated as edge weights.
        Convention is adj[j, i] corresponds to i -> j.

    exclude_self:
        If True, ignore diagonal entries.

    normalize:
        If True, scale NIO by (N-1).

    Returns
    -------

    SignalOut

        Dataclass containing nio, out_strength, and in_strength.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.nio import compute_nio
    >>> N = 5
    >>> A = np.zeros((N, N), dtype=np.int8)
    >>> A[0, 1] = 1
    >>> A[0, 2] = 1
    >>> A[3, 0] = 1
    >>> out = compute_nio(A, True, True)
    >>> out.out_strength.tolist()
    [2.0, 0.0, 0.0, 1.0, 0.0]
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square 2D")
    N = adj.shape[0]
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
        denom = float(max(N - 1, 1))
        nio = nio / denom

    return SignalOut(
        nio=nio.astype(np.float64, copy=False), out_strength=out_s, in_strength=in_s
    )


def hub_recovery_from_signal(
    true_signal: np.ndarray,
    pred_signal: np.ndarray,
    topk: int,
) -> float:
    """
    Measure hub recovery rate by overlap of top-k nodes under two signals.

    Parameters
    ----------

    true_signal:
        Ground-truth 1D signal of length N.

    pred_signal:
        Predicted 1D signal of length N.

    topk:
        Size of the hub set (non-negative). If 0, returns 0.

    Returns
    -------

    float

        Overlap fraction |TopK(true) ∩ TopK(pred)| / k.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.nio import hub_recovery_from_signal
    >>> t = np.array([0.0, 1.0, 5.0, 2.0, 3.0])
    >>> p = np.array([0.0, 2.0, 4.0, 1.0, 3.0])
    >>> hub_recovery_from_signal(t, p, 2)
    1.0
    """
    if true_signal.ndim != 1 or pred_signal.ndim != 1:
        raise ValueError("signals must be 1D")
    if true_signal.shape[0] != pred_signal.shape[0]:
        raise ValueError("signals must have same length")
    if topk < 0:
        raise ValueError("topk must be non-negative")
    N = true_signal.shape[0]
    if topk == 0 or N == 0:
        return 0.0
    k = min(topk, N)
    true_idx = np.argpartition(true_signal, -k)[-k:]
    pred_idx = np.argpartition(pred_signal, -k)[-k:]
    true_set = set(int(i) for i in true_idx.tolist())
    pred_set = set(int(i) for i in pred_idx.tolist())
    return float(len(true_set & pred_set)) / float(k)
