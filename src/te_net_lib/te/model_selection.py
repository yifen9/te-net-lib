from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from te_net_lib.te.lasso_te import lasso_te_matrix


@dataclass(frozen=True, slots=True)
class LassoPathOut:
    alphas: np.ndarray
    betas: np.ndarray
    intercepts: np.ndarray | None
    n_iters: np.ndarray


def lasso_te_path(
    returns: np.ndarray,
    lag: int,
    alphas: Sequence[float],
    max_iter: int,
    tol: float,
    add_intercept: bool,
    standardize: bool,
    exclude_self: bool,
) -> LassoPathOut:
    a = np.array(list(alphas), dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("alphas must be 1D")
    if a.size == 0:
        raise ValueError("alphas must be non-empty")
    if np.any(a < 0.0):
        raise ValueError("alphas must be non-negative")

    T, N = returns.shape
    betas = np.zeros((a.size, N, N), dtype=np.float64)
    intercepts = np.zeros((a.size, N), dtype=np.float64) if add_intercept else None
    n_iters = np.zeros((a.size, N), dtype=np.int64)

    for i, alpha in enumerate(a.tolist()):
        out = lasso_te_matrix(
            returns,
            lag,
            float(alpha),
            max_iter,
            tol,
            add_intercept,
            standardize,
            exclude_self,
        )
        betas[i] = out.beta
        n_iters[i] = out.n_iter
        if add_intercept:
            intercepts[i] = out.intercept

    return LassoPathOut(alphas=a, betas=betas, intercepts=intercepts, n_iters=n_iters)
