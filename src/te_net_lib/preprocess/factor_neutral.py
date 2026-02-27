from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FactorNeutralOut:
    """
    Output of factor neutralization by truncated SVD.

    Attributes
    ----------

    returns_neutral:
        Residual return panel after removing top-k components, shape (T, N).

    factors:
        Estimated factor time series, shape (T, k). Defined as U_k * S_k.

    components:
        Estimated component loadings, shape (k, N). Defined as V_k^T.

    singular_values:
        Top-k singular values, shape (k,).

    explained_variance_ratio:
        Explained variance ratio for top-k components, shape (k,).

    mean_:
        Mean vector used for centering (zeros if center=False), shape (N,).

    """

    returns_neutral: np.ndarray
    factors: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    mean_: np.ndarray


def factor_neutralize_svd(
    returns: np.ndarray, n_components: int, center: bool
) -> FactorNeutralOut:
    """
    Remove a low-rank factor structure from returns using truncated SVD.

    If center=True, the column mean is removed prior to SVD and added back after
    reconstruction removal, so that the returned residual panel has the same mean
    as the input.

    Parameters
    ----------

    returns:
        Return panel with shape (T, N).

    n_components:
        Number of principal components to remove, must satisfy 0 <= n_components <= min(T, N).

    center:
        If True, subtract the column mean before SVD.

    Returns
    -------

    FactorNeutralOut

        Dataclass containing neutralized returns and SVD diagnostics.

    Raises
    ------

    ValueError

        If input is not 2D or n_components is outside valid range.

    Notes
    -----

    Orthogonality check:
        factors^T @ returns_neutral is approximately zero (within numerical tolerance).

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.preprocess.factor_neutral import factor_neutralize_svd
    >>> g = np.random.default_rng(0)
    >>> R = g.normal(size=(50, 6)).astype(np.float64)
    >>> out = factor_neutralize_svd(R, 2, True)
    >>> out.returns_neutral.shape
    (50, 6)
    >>> out.factors.shape
    (50, 2)
    >>> out.components.shape
    (2, 6)
    """
    if returns.ndim != 2:
        raise ValueError("returns must be 2D")
    T, N = returns.shape
    if n_components < 0:
        raise ValueError("n_components must be non-negative")
    if n_components > min(T, N):
        raise ValueError("n_components must be <= min(T, N)")

    R = returns.astype(np.float64, copy=False)

    if center:
        mean_ = R.mean(axis=0, keepdims=True)
        Rc = R - mean_
    else:
        mean_ = np.zeros((1, N), dtype=np.float64)
        Rc = R

    if n_components == 0:
        k = 0
        resid = Rc.copy()
        return FactorNeutralOut(
            returns_neutral=(resid + mean_).astype(np.float64, copy=False),
            factors=np.zeros((T, k), dtype=np.float64),
            components=np.zeros((k, N), dtype=np.float64),
            singular_values=np.zeros((k,), dtype=np.float64),
            explained_variance_ratio=np.zeros((k,), dtype=np.float64),
            mean_=mean_.reshape(-1),
        )

    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    k = int(n_components)

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    recon = (U_k * S_k) @ Vt_k
    resid = Rc - recon

    denom_var = max(T - 1, 1)
    total_var = (S * S) / float(denom_var)
    explained = (S_k * S_k) / float(denom_var)
    den = float(total_var.sum())
    explained_ratio = explained / den if den > 0.0 else np.zeros_like(explained)

    return FactorNeutralOut(
        returns_neutral=(resid + mean_).astype(np.float64, copy=False),
        factors=(U_k * S_k).astype(np.float64, copy=False),
        components=Vt_k.astype(np.float64, copy=False),
        singular_values=S_k.astype(np.float64, copy=False),
        explained_variance_ratio=explained_ratio.astype(np.float64, copy=False),
        mean_=mean_.reshape(-1),
    )
