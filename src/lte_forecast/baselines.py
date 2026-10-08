"""Naive forecasters, all *aligned to the target slot* and using only information available at the origin."""
from __future__ import annotations

import numpy as np

from .data import STEPS_PER_DAY, STEPS_PER_HOUR, STEPS_PER_WEEK
from .features import seasonal_mean, shift

BASELINES = ["persistence", "seasonal_naive_1d", "seasonal_naive_7d", "seasonal_mean_7d"]


def baseline_panels(x: np.ndarray, h: int) -> dict[str, np.ndarray]:
    """Forecast arrays indexed by origin t, for target slot t+4h.

    seasonal_naive_1d: value at the same time yesterday, i.e. slot t+4h-96   (NOT t-96, which is off by the horizon)
    seasonal_naive_7d: same time last week, slot t+4h-672
    seasonal_mean_7d : mean of the same time-of-day over the previous 7 days
    """
    hs = h * STEPS_PER_HOUR
    return {
        "persistence": x.copy(),
        "seasonal_naive_1d": shift(x, STEPS_PER_DAY - hs),
        "seasonal_naive_7d": shift(x, STEPS_PER_WEEK - hs),
        "seasonal_mean_7d": seasonal_mean(x, hs, 7),
    }
