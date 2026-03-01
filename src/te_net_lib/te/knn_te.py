from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sklearn.neighbors import KNeighborsRegressor


@dataclass(frozen=True, slots=True)
class KnnTeOut:
    """
    Output of K-Nearest Neighbors based nonparametric TE estimation.

    Attributes
    ----------
    beta:
        Score matrix with shape (N, N). Entry beta[j, i] corresponds to i -> j.
        Computed as max(0, 0.5 * log(MSE_base / MSE_full)).
    """

    beta: np.ndarray


def knn_te_matrix(
    returns: np.ndarray,
    lag: int,
    k_neighbors: int,
    exclude_self: bool,
) -> KnnTeOut:
    """
    Estimate a nonparametric TE score matrix using Bivariate KNN Regression.

    For each pair (i, j), we compare the residual variance of two KNN models:
      1. Base model: predict y_j from its own lag y_{j, t-lag}
      2. Full model: predict y_j from (y_{j, t-lag}, x_{i, t-lag})

    The surrogate TE score is 0.5 * log(MSE_base / MSE_full).

    Parameters
    ----------
    returns:
        Return panel with shape (T, N).
    lag:
        Positive lag used to define predictors.
    k_neighbors:
        Number of neighbors for the KNN regressor.
    exclude_self:
        If True, beta[j, j] is set to 0.0.

    Returns
    -------
    KnnTeOut
        Dataclass containing the estimated TE score matrix.
    """
    if returns.ndim != 2:
        raise ValueError("returns must be 2D")
    if lag <= 0:
        raise ValueError("lag must be positive")
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")

    T, N = returns.shape
    if T <= lag:
        raise ValueError("T must be larger than lag")

    X = returns[:-lag, :].astype(np.float64, copy=False)
    Y = returns[lag:, :].astype(np.float64, copy=False)

    beta = np.zeros((N, N), dtype=np.float64)
    eps = 1e-12

    for j in range(N):
        y = Y[:, j]

        X_red = X[:, j].reshape(-1, 1)
        n_samples = X_red.shape[0]
        actual_k = min(k_neighbors, n_samples)

        knn_red = KNeighborsRegressor(n_neighbors=actual_k)
        knn_red.fit(X_red, y)
        y_pred_red = knn_red.predict(X_red)

        mse_red = np.mean((y - y_pred_red) ** 2)
        mse_red = max(mse_red, eps)

        for i in range(N):
            if exclude_self and i == j:
                beta[j, i] = 0.0
                continue

            X_full = X[:, [j, i]]
            knn_full = KNeighborsRegressor(n_neighbors=actual_k)
            knn_full.fit(X_full, y)
            y_pred_full = knn_full.predict(X_full)

            mse_full = np.mean((y - y_pred_full) ** 2)
            mse_full = max(mse_full, eps)

            te_score = 0.5 * np.log(mse_red / mse_full)

            beta[j, i] = max(0.0, float(te_score))

    return KnnTeOut(beta=beta)
