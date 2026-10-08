"""One-sided split-conformal calibration of upper quantile forecasts (CQR-style).

Given a quantile model q_{1-alpha}(x) and a held-out calibration block, the conformity score is
s = y - q_{1-alpha}(x). The calibrated bound is  U(x) = q_{1-alpha}(x) + Q_{ceil((n+1)(1-alpha))/n}(s),
which covers y with probability >= 1-alpha under exchangeability. Time series are not exactly
exchangeable, therefore realised coverage is always re-measured on the untouched test block.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_offset(y_cal: np.ndarray, q_cal: np.ndarray, alpha: float) -> float:
    s = y_cal - q_cal
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])
