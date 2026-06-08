#!/usr/bin/env python3
"""Time-decay sample reweighter for practice stage2.

Weights decay exponentially with sample age, normalized to mean 1 and clipped
by a floor to avoid vanishing gradients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.data.dataset.weight import Reweighter


class TimeDecayReweighter(Reweighter):
    def __init__(self, half_life: int = 252, floor: float = 0.2):
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        if not (0 < floor <= 1):
            raise ValueError("floor must be in (0, 1]")
        self.half_life = int(half_life)
        self.floor = float(floor)

    def reweight(self, data: object) -> object:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.Series(dtype=float)
        if not isinstance(data.index, pd.MultiIndex) or "datetime" not in data.index.names:
            return pd.Series(1.0, index=data.index)

        dt = pd.to_datetime(data.index.get_level_values("datetime"), errors="coerce")
        if dt.isna().all():
            return pd.Series(1.0, index=data.index)

        latest = dt.max()
        age_days = np.asarray((latest - dt).days, dtype=float)
        weights = np.exp(-np.log(2.0) * age_days / float(self.half_life))
        mean_w = np.nanmean(weights)
        if not np.isfinite(mean_w) or mean_w <= 0:
            weights = np.ones(len(data), dtype=float)
        else:
            weights = weights / mean_w
        weights = np.clip(weights, self.floor, None)
        return pd.Series(weights, index=data.index)
