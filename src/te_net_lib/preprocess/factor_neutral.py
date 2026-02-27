from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FactorNeutralOut:
    """
    Output of factor-neutral preprocessing based on a low-rank SVD decomposition.

    Attributes
    ----------

    returns_neutral:
        Residualized return panel with shape (T, N). This is the input returns after
        removing the rank-k reconstruction implied by the estimated factors.

    factors_hat:
        Estimated factor time series with shape (T, k). For k=0, this has shape (T, 0).

    components:
        Estimated loading directions with shape (k, N). For k=0, this has shape (0, N).
        These are the top-k right singular vectors (rows of V^T) of the centered return
        matrix when `center=True`, otherwise of the raw return matrix.

    explained_variance_ratio:
        Fraction of total variance explained by each of the top-k components, shape (k,).
        For k=0, this has shape (0,).

    mean_:
        Column mean used for centering, shape (N,). If `center=False`, this is all zeros.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.preprocess.factor_neutral import FactorNeutralOut
    >>> x = np.zeros((10, 3), dtype=np.float64)
    >>> out = FactorNeutralOut(x, np.zeros((10, 0)), np.zeros((0, 3)), np.zeros((0,)), np.zeros((3,)))
    >>> out.returns_neutral.shape
    (10, 3)
    """

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
    """
    Compute a factor-neutralized return panel using truncated SVD.

    Given an input return panel R with shape (T, N), this function optionally centers
    it across time and computes the compact SVD:
        R_c = U S V^T

    The top-k reconstruction is:
        R_k = (U_k S_k) V_k^T

    and the factor-neutral residual is:
        R_neutral = R_c - R_k

    The returned `factors_hat` corresponds to U_k S_k with shape (T, k), and
    `components` corresponds to V_k^T with shape (k, N). This matches the standard
    PCA/SVD low-rank approximation viewpoint.

    Parameters
    ----------

    returns:
        Return panel with shape (T, N).

    n_components:
        Number of components k to remove. Must satisfy 0 <= k <= min(T, N).

    center:
        If True, subtract the time-wise mean of each column before SVD. If False,
        SVD is computed on the raw input.

    Returns
    -------

    FactorNeutralOut

        A dataclass containing the residualized returns, estimated factors/loadings,
        explained variance ratios, and the centering mean.

    Raises
    ------

    ValueError

        If `returns` is not 2D, `n_components` is negative, or exceeds min(T, N).

    Notes
    -----

    Shape convention across this library is `returns` as (T, N). The neutralized output
    preserves the same shape.

    The orthogonality property used in tests is:
        factors_hat^T @ returns_neutral ≈ 0

    which holds for the truncated SVD residual in exact arithmetic.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.preprocess.factor_neutral import factor_neutralize_svd
    >>> g = np.random.default_rng(0)
    >>> R = g.normal(size=(50, 6)).astype(np.float64)
    >>> out = factor_neutralize_svd(R, 2, True)
    >>> out.returns_neutral.shape
    (50, 6)
    >>> out.factors_hat.shape
    (50, 2)
    >>> out.components.shape
    (2, 6)
    >>> float(np.max(np.abs(out.factors_hat.T @ out.returns_neutral))) < 1e-8
    True
    """
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
