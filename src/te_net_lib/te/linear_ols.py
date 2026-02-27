from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LinearOlsTeOut:
    beta: np.ndarray
    intercept: np.ndarray | None
    resid_var: np.ndarray
    dof: int


def ols_te_matrix(
    returns: np.ndarray,
    lag: int,
    add_intercept: bool,
    exclude_self: bool,
) -> LinearOlsTeOut:
    if returns.ndim != 2:
        raise ValueError("returns must be 2D")
    if lag <= 0:
        raise ValueError("lag must be positive")

    T, N = returns.shape
    if T <= lag:
        raise ValueError("T must be greater than lag")

    Y = returns[lag:, :].astype(np.float64, copy=False)
    Xfull = returns[:-lag, :].astype(np.float64, copy=False)

    beta = np.zeros((N, N), dtype=np.float64)
    intercept = np.zeros((N,), dtype=np.float64) if add_intercept else None
    resid_var = np.zeros((N,), dtype=np.float64)

    n_obs = Y.shape[0]

    for j in range(N):
        if exclude_self:
            cols = np.arange(N) != j
            Xj = Xfull[:, cols]
        else:
            cols = slice(None)
            Xj = Xfull

        if add_intercept:
            Xdesign = np.concatenate(
                [np.ones((n_obs, 1), dtype=np.float64), Xj], axis=1
            )
        else:
            Xdesign = Xj

        y = Y[:, j]

        coef, residuals, rank, s = np.linalg.lstsq(Xdesign, y, rcond=None)

        if add_intercept:
            b0 = float(coef[0])
            bj = coef[1:]
            intercept[j] = b0
        else:
            bj = coef

        if exclude_self:
            beta[j, cols] = bj
            beta[j, j] = 0.0
        else:
            beta[j, :] = bj

        yhat = Xdesign @ coef
        e = y - yhat
        p = int(Xdesign.shape[1])
        dof = n_obs - p
        if dof <= 0:
            raise ValueError("degrees of freedom must be positive")
        resid_var[j] = float((e @ e) / dof)

    dof_out = int(
        (n_obs - (N + 1))
        if add_intercept and not exclude_self
        else (
            n_obs - N
            if not add_intercept and not exclude_self
            else n_obs - (N if add_intercept else (N - 1))
        )
    )
    return LinearOlsTeOut(
        beta=beta,
        intercept=intercept,
        resid_var=resid_var,
        dof=int(min(dof_out, n_obs - 1)),
    )
