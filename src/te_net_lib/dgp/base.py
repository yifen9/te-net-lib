from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DgpSample:
    returns: np.ndarray
    true_adj: np.ndarray | None
    extras: dict[str, Any]
