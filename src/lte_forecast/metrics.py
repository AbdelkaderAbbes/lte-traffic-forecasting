"""Forecast metrics implemented in numpy (no scikit-learn dependency)."""
from __future__ import annotations

import numpy as np


def mae(y, p) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def bias(y, p) -> float:
    return float(np.mean(p - y))


def wape(y, p) -> float:
    d = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / d) if d > 0 else float("nan")


def pinball(y, q, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def coverage(y, upper) -> float:
    return float(np.mean(y <= upper))


def point_metrics(y, p) -> dict[str, float]:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "wape": wape(y, p), "bias": bias(y, p)}


def paired_block_bootstrap(abs_err_a, abs_err_b, groups, n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """CI for mean(abs_err_a) - mean(abs_err_b), resampling whole groups (e.g. cell-days).

    Resampling blocks rather than rows keeps the strong within-cell autocorrelation intact, so the
    interval is not over-confident.
    """
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    d = abs_err_a - abs_err_b
    s = np.bincount(inv, weights=d, minlength=G)
    n = np.bincount(inv, minlength=G).astype(float)
    est = float(d.mean())
    idx = rng.integers(0, G, size=(n_boot, G))
    boots = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": est, "lo": float(lo), "hi": float(hi)}
