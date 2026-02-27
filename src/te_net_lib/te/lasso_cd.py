from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def soft_threshold(x: float, t: float) -> float:
    if x > t:
        return x - t
    if x < -t:
        return x + t
    return 0.0


@dataclass(frozen=True, slots=True)
class LassoCdOut:
    coef: np.ndarray
    n_iter: int
    max_delta: float


def lasso_cd(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
) -> LassoCdOut:
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("X and y size mismatch")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    Xf = X.astype(np.float64, copy=False)
    yf = y.astype(np.float64, copy=False)

    if alpha == 0.0:
        coef, residuals, rank, s = np.linalg.lstsq(Xf, yf, rcond=None)
        return LassoCdOut(
            coef=coef.astype(np.float64, copy=False), n_iter=1, max_delta=0.0
        )

    z = (Xf * Xf).sum(axis=0) / float(n)
    if np.any(z <= 0.0):
        raise ValueError("columns of X must have nonzero variance")

    b = np.zeros((p,), dtype=np.float64)
    r = yf.copy()

    it = 0
    max_delta = 0.0

    for it in range(1, max_iter + 1):
        max_delta = 0.0
        for j in range(p):
            xj = Xf[:, j]
            bj_old = b[j]
            rho = (xj @ (r + bj_old * xj)) / float(n)
            bj_new = soft_threshold(float(rho), float(alpha)) / float(z[j])
            delta = bj_new - bj_old
            if delta != 0.0:
                r -= delta * xj
                b[j] = bj_new
                ad = abs(delta)
                if ad > max_delta:
                    max_delta = ad
        if max_delta <= tol:
            break

    return LassoCdOut(coef=b, n_iter=int(it), max_delta=float(max_delta))
