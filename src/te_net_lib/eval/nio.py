from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SignalOut:
    """
    Signals derived from a directed adjacency matrix.

    Conventions
    -----------

    This library uses the TE coefficient layout convention:

        adj[j, i] corresponds to a directed edge i -> j.

    Therefore:

        out_strength[i] = sum_j adj[j, i]   (column sum)
        in_strength[i]  = sum_j adj[i, j]   (row sum)
        nio[i]          = out_strength[i] - in_strength[i]

    If normalize=True, nio is divided by (N-1) for cross-N comparability.

    Attributes
    ----------

    nio:
        Net information outflow (NIO), shape (N,).

    out_strength:
        Out-strength vector, shape (N,).

    in_strength:
        In-strength vector, shape (N,).
    """

    nio: np.ndarray
    out_strength: np.ndarray
    in_strength: np.ndarray


def compute_nio(adj: np.ndarray, exclude_self: bool, normalize: bool) -> SignalOut:
    """
    Compute NIO (net information outflow) and strength vectors from adjacency.

    Parameters
    ----------

    adj:
        Adjacency matrix with shape (N, N). Nonzero entries are treated as edges/weights.

    exclude_self:
        If True, diagonal entries are zeroed before computing strengths.

    normalize:
        If True, divide nio by (N-1) for cross-N comparability.

    Returns
    -------

    SignalOut

        Dataclass containing nio, out_strength, and in_strength.

    Raises
    ------

    ValueError

        If adj is not a square 2D array.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.nio import compute_nio
    >>> adj = np.array([[0, 1], [0, 0]], dtype=np.int8)
    >>> out = compute_nio(adj, True, False)
    >>> out.nio.shape
    (2,)
    """
    if adj.ndim != 2:
        raise ValueError("adj must be 2D")
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square")
    N = adj.shape[0]

    adj0 = adj.astype(np.float64, copy=True)
    if exclude_self:
        np.fill_diagonal(adj0, 0.0)

    out_strength = adj0.sum(axis=0).astype(np.float64)
    in_strength = adj0.sum(axis=1).astype(np.float64)
    nio = (out_strength - in_strength).astype(np.float64)

    if normalize:
        denom = float(max(N - 1, 1))
        nio = nio / denom

    return SignalOut(nio=nio, out_strength=out_strength, in_strength=in_strength)


def hub_recovery_from_signal(
    signal: np.ndarray, true_hubs: np.ndarray, k: int
) -> float:
    """
    Compute hub recovery rate by selecting top-k signal entries.

    Parameters
    ----------

    signal:
        Signal vector, shape (N,).

    true_hubs:
        Binary indicator vector for true hubs, shape (N,).

    k:
        Number of hubs selected from signal.

    Returns
    -------

    float

        Fraction of recovered hubs among the true hubs.

    Raises
    ------

    ValueError

        If shapes mismatch or k is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.eval.nio import hub_recovery_from_signal
    >>> signal = np.array([0.0, 2.0, 1.0])
    >>> true_hubs = np.array([0, 1, 1], dtype=np.int8)
    >>> hub_recovery_from_signal(signal, true_hubs, 2)
    1.0
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if true_hubs.ndim != 1:
        raise ValueError("true_hubs must be 1D")
    if signal.shape[0] != true_hubs.shape[0]:
        raise ValueError("signal and true_hubs must have same length")
    N = signal.shape[0]
    if k < 0 or k > N:
        raise ValueError("k must satisfy 0 <= k <= N")

    if k == 0:
        return 0.0

    idx = np.argpartition(-signal.astype(np.float64, copy=False), k - 1)[:k]
    selected = np.zeros(N, dtype=np.int8)
    selected[idx] = 1

    true_pos = int((selected * true_hubs.astype(np.int8, copy=False)).sum())
    denom = int(true_hubs.astype(np.int8, copy=False).sum())
    return float(true_pos) / float(denom) if denom > 0 else 0.0
