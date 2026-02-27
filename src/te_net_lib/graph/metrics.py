from __future__ import annotations

import numpy as np


def graph_density(adj: np.ndarray, exclude_self: bool) -> float:
    """
    Directed graph density for a binary adjacency matrix.

    Conventions
    -----------

    adj[i, j] corresponds to i -> j.

    Parameters
    ----------

    adj:
        Adjacency matrix, shape (N, N). Nonzero entries are treated as edges.

    exclude_self:
        If True, exclude diagonal entries from both numerator and denominator.

    Returns
    -------

    float

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import graph_density
    >>> N = 4
    >>> A = np.zeros((N, N), dtype=np.int8)
    >>> A[0, 1] = 1
    >>> A[1, 2] = 1
    >>> graph_density(A, True)
    0.16666666666666666
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square 2D")
    N = adj.shape[0]
    a = adj != 0

    if exclude_self:
        m = N * (N - 1)
        if m <= 0:
            return 0.0
        count = int(a.sum() - np.diag(a).sum())
        return float(count) / float(m)

    m = N * N
    if m <= 0:
        return 0.0
    return float(int(a.sum())) / float(m)


def in_out_degree(adj: np.ndarray, exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    In-degree and out-degree vectors for a directed adjacency matrix.

    Conventions
    -----------

    adj[i, j] corresponds to i -> j.

    Therefore:
        out_degree[i] = sum_j adj[i, j]  (row sum)
        in_degree[i]  = sum_j adj[j, i]  (column sum)

    Parameters
    ----------

    adj:
        Adjacency matrix, shape (N, N). Nonzero entries are treated as edges.

    exclude_self:
        If True, ignore diagonal entries.

    Returns
    -------

    (numpy.ndarray, numpy.ndarray)

        (in_degree, out_degree), each shape (N,).

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
        out_deg = b.sum(axis=1)
        in_deg = b.sum(axis=0)
        return in_deg, out_deg

    out_deg = a.sum(axis=1)
    in_deg = a.sum(axis=0)
    return in_deg, out_deg


def hub_indices(
    adj: np.ndarray, topk: int, direction: str, exclude_self: bool
) -> np.ndarray:
    """
    Indices of top-k hubs by in-degree or out-degree, returned in deterministic order.

    Conventions
    -----------

    adj[i, j] corresponds to i -> j.

    Ordering
    --------

    The returned indices are sorted by hub strength in descending order. Ties are
    broken by smaller index first, so results are deterministic across runs and
    platforms.

    Parameters
    ----------

    adj:
        Adjacency matrix, shape (N, N). Nonzero entries are treated as edges.

    topk:
        Number of indices to return. Must be positive.

    direction:
        "in" or "out".

    exclude_self:
        If True, ignore diagonal entries.

    Returns
    -------

    numpy.ndarray

        Indices of selected hubs, shape (k,).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import hub_indices
    >>> N = 6
    >>> A = np.zeros((N, N), dtype=np.int8)
    >>> A[0, 1] = 1
    >>> A[0, 2] = 1
    >>> A[0, 3] = 1
    >>> A[4, 2] = 1
    >>> hub_indices(A, 2, "out", True).tolist()
    [0, 4]
    """
    if topk <= 0:
        raise ValueError("topk must be positive")

    indeg, outdeg = in_out_degree(adj, exclude_self)
    if direction == "in":
        s = indeg
    elif direction == "out":
        s = outdeg
    else:
        raise ValueError("direction must be 'in' or 'out'")

    k = min(int(topk), int(s.size))
    if k <= 0:
        return np.array([], dtype=np.int64)

    cand = np.argpartition(s, -k)[-k:]
    order = np.lexsort((cand, -s[cand]))
    return cand[order]


def confusion_counts(
    pred_adj: np.ndarray, true_adj: np.ndarray, exclude_self: bool
) -> tuple[int, int, int, int]:
    """
    Confusion counts for binary adjacency matrices.

    Parameters
    ----------

    pred_adj:
        Predicted adjacency matrix, shape (N, N).

    true_adj:
        Ground-truth adjacency matrix, shape (N, N).

    exclude_self:
        If True, exclude diagonal positions from evaluation.

    Returns
    -------

    (int, int, int, int)

        (tp, fp, fn, tn)

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import confusion_counts
    >>> N = 3
    >>> T = np.zeros((N, N), dtype=np.int8)
    >>> P = np.zeros((N, N), dtype=np.int8)
    >>> T[0, 1] = 1
    >>> P[0, 1] = 1
    >>> P[1, 2] = 1
    >>> confusion_counts(P, T, True)[:3]
    (1, 1, 0)
    """
    if pred_adj.shape != true_adj.shape:
        raise ValueError("shapes must match")
    if pred_adj.ndim != 2 or pred_adj.shape[0] != pred_adj.shape[1]:
        raise ValueError("adj must be square 2D")

    p = pred_adj != 0
    t = true_adj != 0

    if exclude_self:
        mask = ~np.eye(pred_adj.shape[0], dtype=bool)
        p = p[mask]
        t = t[mask]

    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())
    return tp, fp, fn, tn


def precision_recall_f1(
    pred_adj: np.ndarray, true_adj: np.ndarray, exclude_self: bool
) -> tuple[float, float, float]:
    """
    Precision, recall, and F1 for binary adjacency matrices.

    Parameters
    ----------

    pred_adj:
        Predicted adjacency matrix, shape (N, N).

    true_adj:
        Ground-truth adjacency matrix, shape (N, N).

    exclude_self:
        If True, exclude diagonal positions from evaluation.

    Returns
    -------

    (float, float, float)

        (precision, recall, f1)

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import precision_recall_f1
    >>> N = 4
    >>> T = np.zeros((N, N), dtype=np.int8)
    >>> P = np.zeros((N, N), dtype=np.int8)
    >>> T[0, 1] = 1
    >>> T[1, 2] = 1
    >>> P[0, 1] = 1
    >>> P[2, 3] = 1
    >>> precision_recall_f1(P, T, True)
    (0.5, 0.5, 0.5)
    """
    tp, fp, fn, _tn = confusion_counts(pred_adj, true_adj, exclude_self)
    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1
