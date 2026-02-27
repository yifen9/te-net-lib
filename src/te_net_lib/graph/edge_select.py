from __future__ import annotations

import numpy as np


def select_fixed_density(
    scores: np.ndarray,
    density: float,
    exclude_self: bool,
    mode: str,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """
    Select a directed adjacency matrix by keeping a fixed fraction of highest-score edges.

    This is a deterministic thresholding rule over a score matrix (typically a TE
    coefficient matrix). The output adjacency is binary with entries in {0, 1}.

    Parameters
    ----------

    scores:
        Score matrix with shape (N, N). Larger values indicate stronger directed edges
        i -> j at location [j, i] if you pass in TE coefficients with the library
        convention beta[j, i] = i -> j.

    density:
        Fraction of eligible edges to keep, in [0, 1]. The number of selected edges is
        floor(density * M), where M is the number of eligible positions (N*(N-1) if
        `exclude_self=True`, else N*N).

    exclude_self:
        If True, disallow self edges by excluding the diagonal positions.

    mode:

    How to rank edges:

    - "abs": rank by absolute value |score|
    - "pos": rank by positive scores only (negative treated as -inf)
    - "neg": rank by negative scores only, using -score as key (positive treated as -inf)

    Returns
    -------

    numpy.ndarray

        Binary adjacency matrix with shape (N, N) and dtype int8.

        If return_info=True, returns (adj, info) where info includes:
        - density_target, density_real
        - k_target, k_selected
        - mode, exclude_self

    Raises
    ------

    ValueError

        If shapes are invalid, density is out of range, or mode is invalid.

    Notes
    -----

    Ties at the selection threshold are broken deterministically by dropping the
    smallest keys among the tied set until exactly k edges remain.
    """
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
        if return_info:
            info = {
                "density_target": float(density),
                "density_real": 0.0,
                "k_target": int(k),
                "k_selected": 0,
                "mode": mode,
                "exclude_self": bool(exclude_self),
            }
            return adj, info
        return adj

    finite_mask = np.isfinite(key)
    if finite_mask.sum() == 0:
        if return_info:
            info = {
                "density_target": float(density),
                "density_real": 0.0,
                "k_target": int(k),
                "k_selected": 0,
                "mode": mode,
                "exclude_self": bool(exclude_self),
            }
            return adj, info
        return adj

    key2 = key[finite_mask]
    idx2 = np.flatnonzero(mask)[finite_mask]

    if k >= key2.size:
        adj_flat = adj.reshape(-1)
        adj_flat[idx2] = 1
        k_sel = int(key2.size)
        dens_real = float(k_sel) / float(m) if m > 0 else 0.0
        if return_info:
            info = {
                "density_target": float(density),
                "density_real": float(dens_real),
                "k_target": int(k),
                "k_selected": int(k_sel),
                "mode": mode,
                "exclude_self": bool(exclude_self),
            }
            return adj, info
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
    k_sel = int(pick.sum())
    dens_real = float(k_sel) / float(m) if m > 0 else 0.0
    if return_info:
        info = {
            "density_target": float(density),
            "density_real": float(dens_real),
            "k_target": int(k),
            "k_selected": int(k_sel),
            "mode": mode,
            "exclude_self": bool(exclude_self),
        }
        return adj, info
    return adj
