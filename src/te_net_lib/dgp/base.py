from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DgpSample:
    """
    Container for simulated return panels and optional ground-truth structure.

    Attributes
    ----------

    returns:
        Simulated return panel with shape (T, N), where T is the number of time
        steps and N is the number of assets/nodes.

    true_adj:
        Optional ground-truth adjacency matrix with shape (N, N) and entries in {0, 1}.
        If present, convention is that true_adj[j, i] == 1 indicates a directed edge
        i -> j (i influences j).

    extras:
        Additional diagnostic objects produced by the DGP (e.g., VAR coefficient matrix,
        factor series, loadings, conditional variances). This is intended for analysis
        and validation, not for stable public API guarantees.

    Examples
    --------
    >>> import numpy as np
    >>> from te_net_lib.dgp.base import DgpSample
    >>> s = DgpSample(returns=np.zeros((10, 3)), true_adj=None, extras={})
    >>> s.returns.shape
    (10, 3)
    """

    returns: np.ndarray
    true_adj: np.ndarray | None
    extras: dict[str, Any]
