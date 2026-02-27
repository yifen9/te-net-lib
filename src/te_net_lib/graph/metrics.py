from __future__ import annotations

import numpy as np


def graph_density(adj: np.ndarray, exclude_self: bool) -> float:
    """
    Compute directed graph density for a binary adjacency matrix.

    Parameters
    ----------

    adj:
        Adjacency matrix with shape (N, N). Nonzero entries are treated as edges.

    exclude_self:
        If True, exclude diagonal positions from both numerator and denominator.

    Returns
    -------

    float

        Density in [0, 1]. For exclude_self=True, denominator is N*(N-1); otherwise N*N.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import graph_density
    >>> A = np.zeros((4, 4), dtype=np.int8)
    >>> A[0, 1] = 1
    >>> A[1, 2] = 1
    >>> abs(graph_density(A, True) - (2.0 / (4 * 3))) < 1e-12
    True
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square 2D")
    N = adj.shape[0]
    a = (adj != 0).astype(np.int8, copy=False)
    if exclude_self:
        m = N * (N - 1)
        if m == 0:
            return 0.0
        count = int(a.sum() - int(np.diag(a).sum()))
        return float(count) / float(m)
    m = N * N
    if m == 0:
        return 0.0
    return float(int(a.sum())) / float(m)


def in_out_degree(adj: np.ndarray, exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute in-degree and out-degree vectors for a directed adjacency matrix.

    Parameters
    ----------

    adj:
        Adjacency matrix with shape (N, N). Nonzero entries are treated as edges.
        Entry adj[j, i] corresponds to i -> j.

    exclude_self:
        If True, ignore diagonal entries.

    Returns
    -------

    (numpy.ndarray, numpy.ndarray)

        (in_degree, out_degree), each with shape (N,).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import in_out_degree
    >>> A = np.zeros((5, 5), dtype=np.int8)
    >>> A[0, 1] = 1
    >>> A[0, 2] = 1
    >>> A[3, 2] = 1
    >>> indeg, outdeg = in_out_degree(A, True)
    >>> int(outdeg[0]), int(indeg[2])
    (2, 2)
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square 2D")
    a = (adj != 0).astype(np.int64, copy=False)
    if exclude_self:
        b = a.copy()
        np.fill_diagonal(b, 0)
        return b.sum(axis=0), b.sum(axis=1)
    return a.sum(axis=0), a.sum(axis=1)


def hub_indices(
    adj: np.ndarray, topk: int, direction: str, exclude_self: bool
) -> np.ndarray:
    """
    Return indices of top-k hubs by in-degree or out-degree.

    Parameters
    ----------

    adj:
        Adjacency matrix with shape (N, N).

    topk:
        Number of indices to return. If 0, returns an empty array.

    direction:
        "in" to rank by in-degree, "out" to rank by out-degree.

    exclude_self:
        If True, ignore diagonal entries.

    Returns
    -------

    numpy.ndarray

        Array of node indices with shape (k,), sorted by descending degree.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import hub_indices
    >>> A = np.zeros((6, 6), dtype=np.int8)
    >>> A[0, 1] = 1
    >>> A[0, 2] = 1
    >>> A[0, 3] = 1
    >>> top = hub_indices(A, 2, "out", True)
    >>> int(top[0]) == 0
    True
    """
    if topk < 0:
        raise ValueError("topk must be non-negative")
    if direction not in ("in", "out"):
        raise ValueError("direction must be in or out")
    indeg, outdeg = in_out_degree(adj, exclude_self)
    deg = indeg if direction == "in" else outdeg
    N = deg.shape[0]
    if topk == 0 or N == 0:
        return np.zeros((0,), dtype=np.int64)
    k = min(topk, N)
    idx = np.argpartition(deg, -k)[-k:]
    idx = idx[np.argsort(deg[idx])[::-1]]
    return idx.astype(np.int64, copy=False)


def confusion_counts(
    pred_adj: np.ndarray, true_adj: np.ndarray, exclude_self: bool
) -> tuple[int, int, int, int]:
    """
    Compute confusion-matrix counts between a predicted and true adjacency matrix.

    Parameters
    ----------

    pred_adj:
        Predicted adjacency with shape (N, N). Nonzero entries are treated as edges.

    true_adj:
        Ground-truth adjacency with shape (N, N). Nonzero entries are treated as edges.

    exclude_self:
        If True, ignore diagonal entries in both matrices.

    Returns
    -------

    (int, int, int, int)

        (tp, fp, fn, tn) counts.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import confusion_counts
    >>> T = np.zeros((4, 4), dtype=np.int8)
    >>> P = np.zeros((4, 4), dtype=np.int8)
    >>> T[0, 1] = 1
    >>> T[1, 2] = 1
    >>> P[0, 1] = 1
    >>> P[2, 3] = 1
    >>> confusion_counts(P, T, True)[:3]
    (1, 1, 1)
    """
    if pred_adj.shape != true_adj.shape:
        raise ValueError("pred_adj and true_adj must have same shape")
    if pred_adj.ndim != 2 or pred_adj.shape[0] != pred_adj.shape[1]:
        raise ValueError("adj must be square 2D")
    N = pred_adj.shape[0]
    p = pred_adj != 0
    t = true_adj != 0
    if exclude_self:
        diag = np.eye(N, dtype=bool)
        p = p & (~diag)
        t = t & (~diag)
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())
    return tp, fp, fn, tn


def precision_recall_f1(
    pred_adj: np.ndarray, true_adj: np.ndarray, exclude_self: bool
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1-score for adjacency prediction.

    Parameters
    ----------

    pred_adj:
        Predicted adjacency with shape (N, N).

    true_adj:
        Ground-truth adjacency with shape (N, N).

    exclude_self:
        If True, ignore diagonal entries.

    Returns
    -------

    (float, float, float)

        (precision, recall, f1).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import precision_recall_f1
    >>> T = np.zeros((4, 4), dtype=np.int8)
    >>> P = np.zeros((4, 4), dtype=np.int8)
    >>> T[0, 1] = 1
    >>> T[1, 2] = 1
    >>> P[0, 1] = 1
    >>> P[2, 3] = 1
    >>> tuple(round(x, 6) for x in precision_recall_f1(P, T, True))
    (0.5, 0.5, 0.5)
    """
    tp, fp, fn, tn = confusion_counts(pred_adj, true_adj, exclude_self)
    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1
