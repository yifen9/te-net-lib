from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SignalOut:
    nio: np.ndarray
    out_strength: np.ndarray
    in_strength: np.ndarray


def compute_nio(
    adj: np.ndarray,
    exclude_self: bool,
    normalize: bool,
) -> SignalOut:
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square 2D")
    N = adj.shape[0]
    a = adj.astype(np.float64, copy=False)

    if exclude_self:
        b = a.copy()
        np.fill_diagonal(b, 0.0)
        out_s = b.sum(axis=1)
        in_s = b.sum(axis=0)
    else:
        out_s = a.sum(axis=1)
        in_s = a.sum(axis=0)

    nio = out_s - in_s
    if normalize:
        denom = float(max(N - 1, 1))
        nio = nio / denom

    return SignalOut(
        nio=nio.astype(np.float64, copy=False), out_strength=out_s, in_strength=in_s
    )


def hub_recovery_from_signal(
    true_signal: np.ndarray,
    pred_signal: np.ndarray,
    topk: int,
) -> float:
    if true_signal.ndim != 1 or pred_signal.ndim != 1:
        raise ValueError("signals must be 1D")
    if true_signal.shape[0] != pred_signal.shape[0]:
        raise ValueError("signals must have same length")
    if topk < 0:
        raise ValueError("topk must be non-negative")
    N = true_signal.shape[0]
    if topk == 0 or N == 0:
        return 0.0
    k = min(topk, N)
    true_idx = np.argpartition(true_signal, -k)[-k:]
    pred_idx = np.argpartition(pred_signal, -k)[-k:]
    true_set = set(int(i) for i in true_idx.tolist())
    pred_set = set(int(i) for i in pred_idx.tolist())
    return float(len(true_set & pred_set)) / float(k)
