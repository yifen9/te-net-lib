from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FactorNeutralOut:
    returns_neutral: np.ndarray
    factors_hat: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    mean_: np.ndarray


def factor_neutralize_svd(
    returns: np.ndarray,
    n_components: int,
    center: bool,
) -> FactorNeutralOut:
    if returns.ndim != 2:
        raise ValueError("returns must be 2D")
    T, N = returns.shape
    if n_components < 0:
        raise ValueError("n_components must be non-negative")
    if n_components > min(T, N):
        raise ValueError("n_components must be <= min(T, N)")

    X = returns.astype(np.float64, copy=False)
    mean_ = np.zeros((1, N), dtype=np.float64)
    if center:
        mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - mean_
    else:
        Xc = X

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = n_components

    if k == 0:
        factors_hat = np.zeros((T, 0), dtype=np.float64)
        components = np.zeros((0, N), dtype=np.float64)
        explained_variance_ratio = np.zeros((0,), dtype=np.float64)
        returns_neutral = Xc.copy()
        return FactorNeutralOut(
            returns_neutral=returns_neutral,
            factors_hat=factors_hat,
            components=components,
            explained_variance_ratio=explained_variance_ratio,
            mean_=mean_.reshape(-1),
        )

    components = Vt[:k, :].astype(np.float64, copy=True)
    factors_hat = (U[:, :k] * S[:k]).astype(np.float64, copy=True)

    recon = factors_hat @ components
    resid = Xc - recon

    denom = float(max(T - 1, 1))
    total_var = (S**2) / denom
    explained = total_var[:k]
    explained_ratio = explained / total_var.sum()

    return FactorNeutralOut(
        returns_neutral=resid.astype(np.float64, copy=False),
        factors_hat=factors_hat,
        components=components,
        explained_variance_ratio=explained_ratio.astype(np.float64, copy=False),
        mean_=mean_.reshape(-1),
    )
