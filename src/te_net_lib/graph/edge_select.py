from __future__ import annotations

import numpy as np


def select_fixed_density(
    scores: np.ndarray,
    density: float,
    exclude_self: bool,
    mode: str,
) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("scores must be 2D")
    if scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be square")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0, 1]")
    if mode not in ("abs", "pos", "neg"):
        raise ValueError("mode must be one of: abs, pos, neg")

    N = scores.shape[0]
    s = scores.astype(np.float64, copy=False)

    mask = np.ones((N, N), dtype=bool)
    if exclude_self:
        np.fill_diagonal(mask, False)

    vals = s[mask]
    if mode == "abs":
        key = np.abs(vals)
    elif mode == "pos":
        key = vals.copy()
        key[key < 0.0] = -np.inf
    else:
        key = (-vals).copy()
        key[key < 0.0] = -np.inf

    m = int(mask.sum())
    k = int(np.floor(density * m + 1e-12))

    adj = np.zeros((N, N), dtype=np.int8)
    if k <= 0 or m == 0:
        return adj

    finite_mask = np.isfinite(key)
    if finite_mask.sum() == 0:
        return adj

    key2 = key[finite_mask]
    idx2 = np.flatnonzero(mask)[finite_mask]

    if k >= key2.size:
        adj_flat = adj.reshape(-1)
        adj_flat[idx2] = 1
        return adj

    thresh = np.partition(key2, -k)[-k]
    pick = key2 >= thresh
    if pick.sum() > k:
        over = np.flatnonzero(pick)
        extra = pick.sum() - k
        take = over[np.argsort(key2[over])[:extra]]
        pick[take] = False

    adj_flat = adj.reshape(-1)
    adj_flat[idx2[pick]] = 1
    return adj
