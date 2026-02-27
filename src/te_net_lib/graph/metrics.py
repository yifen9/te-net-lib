from __future__ import annotations

import numpy as np


def graph_density(adj: np.ndarray, exclude_self: bool) -> float:
    """
    Directed graph density for a binary adjacency matrix.

    Conventions
    -----------

    adj[j, i] corresponds to i -> j.

    Parameters
    ----------

    adj:
        Adjacency matrix, shape (N, N). Nonzero entries are treated as edges.

    exclude_self:
        If True, exclude diagonal entries from both numerator and denominator.

    Returns
    -------

    float

        Graph density in [0, 1].

    Raises
    ------

    ValueError

        If adj is not square.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import graph_density
    >>> A = np.array([[0, 1], [0, 0]], dtype=np.int8)
    >>> graph_density(A, True)
    0.5
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square 2D")
    N = adj.shape[0]
    if N <= 1:
        return 0.0

    a = (adj != 0).astype(np.int8)
    if exclude_self:
        np.fill_diagonal(a, 0)
        denom = float(N * (N - 1))
    else:
        denom = float(N * N)

    return float(a.sum()) / denom


def in_out_degree(adj: np.ndarray, exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    In-degree and out-degree vectors for a directed adjacency matrix.

    Conventions
    -----------

    adj[j, i] corresponds to i -> j.

    Therefore:

        out_degree[i] = sum_j adj[j, i]  (column sum)
        in_degree[i]  = sum_j adj[i, j]  (row sum)

    Parameters
    ----------

    adj:
        Adjacency matrix, shape (N, N). Nonzero entries are treated as edges.

    exclude_self:
        If True, diagonal entries are excluded.

    Returns
    -------

    tuple[np.ndarray, np.ndarray]

        (in_degree, out_degree), each shape (N,).

    Raises
    ------

    ValueError

        If adj is not square.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import in_out_degree
    >>> A = np.array([[0, 1], [0, 0]], dtype=np.int8)
    >>> indeg, outdeg = in_out_degree(A, True)
    >>> indeg.tolist()
    [1.0, 0.0]
    >>> outdeg.tolist()
    [0.0, 1.0]
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square 2D")
    N = adj.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    a = (adj != 0).astype(np.int8)
    if exclude_self:
        b = a.copy()
        np.fill_diagonal(b, 0)
        out_deg = b.sum(axis=0)
        in_deg = b.sum(axis=1)
        return in_deg.astype(np.float64), out_deg.astype(np.float64)

    out_deg = a.sum(axis=0)
    in_deg = a.sum(axis=1)
    return in_deg.astype(np.float64), out_deg.astype(np.float64)


def hub_indices(adj: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    """
    Select hub indices by out-degree, using a deterministic tie-break.

    Parameters
    ----------

    adj:
        Adjacency matrix, shape (N, N). Nonzero entries are treated as edges.

    k:
        Number of hubs to return, must satisfy 0 <= k <= N.

    exclude_self:
        If True, diagonal entries are excluded.

    Returns
    -------

    np.ndarray

        Array of selected hub indices, shape (k,).

    Raises
    ------

    ValueError

        If adj is not square or k is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import hub_indices
    >>> A = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
    >>> hub_indices(A, 1, True).tolist()
    [1]
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square 2D")
    N = adj.shape[0]
    if k < 0 or k > N:
        raise ValueError("k must satisfy 0 <= k <= N")
    if k == 0:
        return np.zeros((0,), dtype=np.int64)

    _, outdeg = in_out_degree(adj, exclude_self)
    s = outdeg.astype(np.float64, copy=False)

    idx = np.argpartition(-s, k - 1)[:k]
    cand = np.array(idx, dtype=np.int64, copy=False)
    order = np.lexsort((cand, -s[cand]))
    return cand[order]


def confusion_counts(
    true_adj: np.ndarray, pred_adj: np.ndarray, exclude_self: bool
) -> tuple[int, int, int, int]:
    """
    Confusion-matrix counts for binary edge prediction.

    Parameters
    ----------

    true_adj:
        Ground truth adjacency, shape (N, N). Nonzero entries treated as edges.

    pred_adj:
        Predicted adjacency, shape (N, N). Nonzero entries treated as edges.

    exclude_self:
        If True, exclude diagonal entries.

    Returns
    -------

    tuple[int, int, int, int]

        (tp, fp, fn, tn)

    Raises
    ------

    ValueError

        If input shapes mismatch or are not square.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import confusion_counts
    >>> t = np.array([[0, 1], [0, 0]], dtype=np.int8)
    >>> p = np.array([[0, 1], [1, 0]], dtype=np.int8)
    >>> confusion_counts(t, p, True)
    (1, 1, 0, 0)
    """
    if true_adj.ndim != 2 or pred_adj.ndim != 2:
        raise ValueError("inputs must be 2D")
    if true_adj.shape != pred_adj.shape:
        raise ValueError("true_adj and pred_adj shapes must match")
    if true_adj.shape[0] != true_adj.shape[1]:
        raise ValueError("inputs must be square")

    t = true_adj != 0
    p = pred_adj != 0

    if exclude_self:
        mask = ~np.eye(true_adj.shape[0], dtype=bool)
        t = t[mask]
        p = p[mask]

    tp = int(np.logical_and(t, p).sum())
    fp = int(np.logical_and(~t, p).sum())
    fn = int(np.logical_and(t, ~p).sum())
    tn = int(np.logical_and(~t, ~p).sum())
    return tp, fp, fn, tn


def precision_recall_f1(
    true_adj: np.ndarray, pred_adj: np.ndarray, exclude_self: bool
) -> tuple[float, float, float]:
    """
    Precision, recall, and F1 for binary edge prediction.

    Parameters
    ----------

    true_adj:
        Ground truth adjacency, shape (N, N).

    pred_adj:
        Predicted adjacency, shape (N, N).

    exclude_self:
        If True, exclude diagonal entries.

    Returns
    -------

    tuple[float, float, float]

        (precision, recall, f1)

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.graph.metrics import precision_recall_f1
    >>> t = np.array([[0, 1], [0, 0]], dtype=np.int8)
    >>> p = np.array([[0, 1], [1, 0]], dtype=np.int8)
    >>> precision_recall_f1(t, p, True)[0] > 0
    True
    """
    tp, fp, fn, _ = confusion_counts(true_adj, pred_adj, exclude_self)
    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1
