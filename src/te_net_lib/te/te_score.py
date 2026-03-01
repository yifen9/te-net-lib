from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from te_net_lib.te.lasso_cd import lasso_cd


@dataclass(frozen=True, slots=True)
class TeScoreOut:
    score: np.ndarray
    coef: np.ndarray | None
    resid_var_full: np.ndarray
    resid_var_reduced: np.ndarray | None
    meta: dict[str, Any]


def _require_returns(returns: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if returns.ndim != 2:
        raise ValueError("returns must be 2D")
    if lag <= 0:
        raise ValueError("lag must be positive")
    T, _ = returns.shape
    if T <= lag:
        raise ValueError("T must be larger than lag")
    X = returns[:-lag, :].astype(np.float64, copy=False)
    Y = returns[lag:, :].astype(np.float64, copy=False)
    return X, Y


def _ols_fit(
    X: np.ndarray,
    y: np.ndarray,
    add_intercept: bool,
) -> tuple[np.ndarray, float, float]:
    n = int(X.shape[0])
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("y must be 1D with length n")
    if add_intercept:
        Xd = np.concatenate([np.ones((n, 1), dtype=np.float64), X], axis=1)
    else:
        Xd = X
    coef, _, _, _ = np.linalg.lstsq(Xd, y.astype(np.float64, copy=False), rcond=None)
    yhat = Xd @ coef
    e = y - yhat
    p = int(Xd.shape[1])
    dof = n - p
    if dof <= 0:
        raise ValueError("degrees of freedom must be positive")
    s2 = float((e @ e) / float(dof))
    b = coef[1:] if add_intercept else coef
    b0 = float(coef[0]) if add_intercept else 0.0
    return b.astype(np.float64, copy=False), float(b0), float(s2)


def _lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
    add_intercept: bool,
    standardize: bool,
    init_coef: np.ndarray | None,
) -> tuple[np.ndarray, float, float, int]:
    n, p = X.shape
    yy = y.astype(np.float64, copy=False)
    Xj = X.astype(np.float64, copy=False)

    y_mean = 0.0
    x_mean = None
    if add_intercept:
        y_mean = float(yy.mean())
        yy = yy - y_mean
        x_mean = Xj.mean(axis=0)
        Xj = Xj - x_mean

    scale = None
    if standardize:
        scale = np.sqrt((Xj * Xj).mean(axis=0))
        if np.any(scale == 0.0):
            raise ValueError("standardize=True but some predictors have zero scale")
        Xj = Xj / scale

    out = lasso_cd(Xj, yy, float(alpha), int(max_iter), float(tol), init_coef=init_coef)
    b = out.coef.astype(np.float64, copy=False)
    if standardize:
        b = b / scale

    if add_intercept:
        b0 = y_mean - float(x_mean.dot(b)) if x_mean is not None else y_mean
    else:
        b0 = 0.0

    yhat = (X.astype(np.float64, copy=False) @ b) + float(b0)
    e = y.astype(np.float64, copy=False) - yhat
    dof = int(n - (p + (1 if add_intercept else 0)))
    dof = max(dof, 1)
    s2 = float((e @ e) / float(dof))
    return b, float(b0), float(s2), int(out.n_iter)


def te_score_matrix_ols(
    returns: np.ndarray,
    *,
    lag: int,
    add_intercept: bool,
    exclude_self: bool,
    score_mode: str,
    eps: float = 1e-12,
) -> TeScoreOut:
    if score_mode not in ("coef", "te_leave_one_out"):
        raise ValueError("score_mode must be one of: coef, te_leave_one_out")

    X, Y = _require_returns(returns, int(lag))
    n_obs, N = X.shape

    coef_mat = np.zeros((N, N), dtype=np.float64)
    score = np.zeros((N, N), dtype=np.float64)
    s2_full = np.zeros((N,), dtype=np.float64)
    s2_red = np.full((N, N), np.nan, dtype=np.float64)

    for j in range(N):
        y = Y[:, j].astype(np.float64, copy=False)
        cols_full = np.arange(N) != j if exclude_self else np.ones((N,), dtype=bool)
        Xfull = X[:, cols_full]

        b_full, b0_full, var_full = _ols_fit(Xfull, y, bool(add_intercept))
        s2_full[j] = float(var_full)

        if exclude_self:
            coef_mat[j, cols_full] = b_full
            coef_mat[j, j] = 0.0
        else:
            coef_mat[j, :] = b_full
            if exclude_self:
                coef_mat[j, j] = 0.0

        if score_mode == "coef":
            score[j, :] = coef_mat[j, :]
            if exclude_self:
                score[j, j] = 0.0
            continue

        base = max(float(var_full), float(eps))

        for i in range(N):
            if exclude_self and i == j:
                score[j, i] = 0.0
                s2_red[j, i] = float("nan")
                continue
            cols_red = cols_full.copy()
            if cols_red[i]:
                cols_red[i] = False
            Xred = X[:, cols_red]
            if Xred.shape[1] == 0:
                var_red = float(np.var(y, ddof=1)) if n_obs >= 2 else float(np.var(y))
            else:
                _, _, var_red = _ols_fit(Xred, y, bool(add_intercept))
            s2_red[j, i] = float(var_red)
            var_red2 = max(float(var_red), float(eps))
            score[j, i] = 0.5 * float(np.log(var_red2 / base))

    if exclude_self:
        np.fill_diagonal(score, 0.0)
        np.fill_diagonal(coef_mat, 0.0)

    meta = {
        "estimator": "ols",
        "lag": int(lag),
        "add_intercept": bool(add_intercept),
        "exclude_self": bool(exclude_self),
        "score_mode": score_mode,
        "n_obs": int(n_obs),
    }
    return TeScoreOut(
        score=score,
        coef=coef_mat,
        resid_var_full=s2_full,
        resid_var_reduced=(s2_red if score_mode == "te_leave_one_out" else None),
        meta=meta,
    )


def te_score_matrix_lasso(
    returns: np.ndarray,
    *,
    lag: int,
    alpha: float,
    max_iter: int,
    tol: float,
    add_intercept: bool,
    standardize: bool,
    exclude_self: bool,
    score_mode: str,
    eps: float = 1e-12,
) -> TeScoreOut:
    if score_mode not in ("coef", "te_leave_one_out"):
        raise ValueError("score_mode must be one of: coef, te_leave_one_out")
    if float(alpha) < 0.0:
        raise ValueError("alpha must be non-negative")

    X, Y = _require_returns(returns, int(lag))
    n_obs, N = X.shape

    coef_mat = np.zeros((N, N), dtype=np.float64)
    score = np.zeros((N, N), dtype=np.float64)
    s2_full = np.zeros((N,), dtype=np.float64)
    s2_red = np.full((N, N), np.nan, dtype=np.float64)
    n_iter = np.zeros((N,), dtype=np.int64)

    for j in range(N):
        y = Y[:, j].astype(np.float64, copy=False)
        cols_full = np.arange(N) != j if exclude_self else np.ones((N,), dtype=bool)
        Xfull = X[:, cols_full]

        b_full, b0_full, var_full, it_full = _lasso_fit(
            Xfull,
            y,
            alpha=float(alpha),
            max_iter=int(max_iter),
            tol=float(tol),
            add_intercept=bool(add_intercept),
            standardize=bool(standardize),
            init_coef=None,
        )
        n_iter[j] = int(it_full)
        s2_full[j] = float(var_full)

        if exclude_self:
            coef_mat[j, cols_full] = b_full
            coef_mat[j, j] = 0.0
        else:
            coef_mat[j, :] = b_full
            if exclude_self:
                coef_mat[j, j] = 0.0

        if score_mode == "coef":
            score[j, :] = coef_mat[j, :]
            if exclude_self:
                score[j, j] = 0.0
            continue

        base = max(float(var_full), float(eps))

        for i in range(N):
            if exclude_self and i == j:
                score[j, i] = 0.0
                s2_red[j, i] = float("nan")
                continue
            cols_red = cols_full.copy()
            if cols_red[i]:
                cols_red[i] = False
            Xred = X[:, cols_red]
            if Xred.shape[1] == 0:
                var_red = float(np.var(y, ddof=1)) if n_obs >= 2 else float(np.var(y))
            else:
                _, _, var_red, _ = _lasso_fit(
                    Xred,
                    y,
                    alpha=float(alpha),
                    max_iter=int(max_iter),
                    tol=float(tol),
                    add_intercept=bool(add_intercept),
                    standardize=bool(standardize),
                    init_coef=None,
                )
            s2_red[j, i] = float(var_red)
            var_red2 = max(float(var_red), float(eps))
            score[j, i] = 0.5 * float(np.log(var_red2 / base))

    if exclude_self:
        np.fill_diagonal(score, 0.0)
        np.fill_diagonal(coef_mat, 0.0)

    meta = {
        "estimator": "lasso",
        "lag": int(lag),
        "alpha": float(alpha),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "add_intercept": bool(add_intercept),
        "standardize": bool(standardize),
        "exclude_self": bool(exclude_self),
        "score_mode": score_mode,
        "n_obs": int(n_obs),
        "n_iter_q": {
            "min": float(np.min(n_iter)) if n_iter.size else 0.0,
            "p25": float(np.quantile(n_iter, 0.25)) if n_iter.size else 0.0,
            "median": float(np.quantile(n_iter, 0.50)) if n_iter.size else 0.0,
            "p75": float(np.quantile(n_iter, 0.75)) if n_iter.size else 0.0,
            "max": float(np.max(n_iter)) if n_iter.size else 0.0,
        },
    }
    return TeScoreOut(
        score=score,
        coef=coef_mat,
        resid_var_full=s2_full,
        resid_var_reduced=(s2_red if score_mode == "te_leave_one_out" else None),
        meta=meta,
    )
