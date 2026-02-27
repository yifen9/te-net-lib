from __future__ import annotations

import numpy as np


def graph_density(adj: np.ndarray, exclude_self: bool) -> float:
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
    tp, fp, fn, tn = confusion_counts(pred_adj, true_adj, exclude_self)
    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1
