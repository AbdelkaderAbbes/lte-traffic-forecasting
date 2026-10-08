"""Leakage-safe feature and target construction on the dense (cell x time) panel.

Conventions
-----------
* ``t`` is the forecast origin (a 15-min slot). A horizon of ``h`` hours predicts slot ``t + 4h``.
* Every input feature at origin ``t`` uses information from slots ``<= t`` only.
* Seasonal features are aligned to the *target* slot: yesterday's value at ``t+4h-96`` is known at ``t``
  for every ``h <= 24`` hours.
* Cell-level statistics are computed on the training window only (``static_features``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelCfg
from .data import STEPS_PER_DAY, STEPS_PER_HOUR, STEPS_PER_WEEK, Panel


def shift(a: np.ndarray, k: int) -> np.ndarray:
    """out[:, t] = a[:, t-k].  k>0 looks into the past, k<0 into the future. Out-of-range -> NaN."""
    out = np.full_like(a, np.nan)
    T = a.shape[1]
    if k == 0:
        return a.copy()
    if abs(k) >= T:
        return out
    if k > 0:
        out[:, k:] = a[:, : T - k]
    else:
        out[:, : T + k] = a[:, -k:]
    return out


def rolling(a: np.ndarray, window: int, fn: str, min_periods: int | None = None) -> np.ndarray:
    """Trailing window ending at t (inclusive). Uses only past + current values."""
    df = pd.DataFrame(a.T)
    r = df.rolling(window, min_periods=min_periods or max(1, window // 2))
    return getattr(r, fn)().to_numpy().T.astype(np.float32)


def seasonal_mean(a: np.ndarray, hs: int, days: int = 7) -> np.ndarray:
    """Mean of the values at the target slot's time-of-day over the previous ``days`` days (aligned to t+hs)."""
    stack = np.stack([shift(a, STEPS_PER_DAY * k - hs) for k in range(1, days + 1)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(stack, axis=0).astype(np.float32)


class FeatureBuilder:
    def __init__(self, panel: Panel, mcfg: ModelCfg):
        self.p = panel
        self.m = mcfg
        self.C, self.T = panel.shape
        self._holidays = {pd.Timestamp(d).date() for d in mcfg.holidays}
        self.cal = self._calendar()
        self.dyn = self._dynamic()
        self._extras: dict[int, dict[str, np.ndarray]] = {}

    # ---- calendar (vectors over T)
    def _calendar(self) -> dict[str, np.ndarray]:
        ix = self.p.index
        slot = (ix.hour * 4 + ix.minute // 15).to_numpy()
        dow = ix.dayofweek.to_numpy()
        hol = np.array([d in self._holidays for d in ix.date], dtype=np.float32)
        return {
            "slot_sin": np.sin(2 * np.pi * slot / 96).astype(np.float32),
            "slot_cos": np.cos(2 * np.pi * slot / 96).astype(np.float32),
            "dow": dow.astype(np.float32),
            "weekend": (dow >= 5).astype(np.float32),
            "holiday": hol,
        }

    # ---- features that do not depend on the horizon
    def _dynamic(self) -> dict[str, np.ndarray]:
        P = self.p
        rb, vol, users = P["rb"], P["vol_dl"], P["users_dl"]
        lvol = np.log1p(vol)
        d: dict[str, np.ndarray] = {
            "rb_t": rb, "lvol_t": lvol, "users_t": users, "rrc_t": P["rrc"],
            "cqi_t": P["cqi_mean"], "mimo_t": P["mimo"], "energy_ru_t": P["energy_ru"],
            "ul_dl_ratio_t": (P["vol_ul"] / (vol + 1e-3)).astype(np.float32),
        }
        for k in (1, 2, 3, 4, 8, 12, 24, 48, 96):
            d[f"rb_lag_{k}"] = shift(rb, k)
        for k in (1, 4, 96):
            d[f"lvol_lag_{k}"] = shift(lvol, k)
        for w, name in ((4, "1h"), (16, "4h"), (96, "24h")):
            d[f"rb_mean_{name}"] = rolling(rb, w, "mean")
            d[f"lvol_mean_{name}"] = rolling(lvol, w, "mean")
        d["rb_std_4h"] = rolling(rb, 16, "std")
        d["rb_max_4h"] = rolling(rb, 16, "max")
        d["rb_min_4h"] = rolling(rb, 16, "min")
        d["rb_zero_share_4h"] = rolling((rb == 0).astype(np.float32) + np.where(np.isnan(rb), np.nan, 0), 16, "mean")
        d["obs_share_24h"] = rolling(P.observed.astype(np.float32), 96, "mean")
        d["rb_trend_1h"] = (rb - shift(rb, 4)).astype(np.float32)
        return d

    # ---- features aligned to the target slot (depend on h)
    def horizon_extras(self, h: int) -> dict[str, np.ndarray]:
        if h in self._extras:
            return self._extras[h]
        hs = h * STEPS_PER_HOUR
        rb, lvol = self.p["rb"], np.log1p(self.p["vol_dl"])
        e = {
            "rb_same_slot_1d": shift(rb, STEPS_PER_DAY - hs),
            "rb_same_slot_7d": shift(rb, STEPS_PER_WEEK - hs),
            "rb_seasmean_7d": seasonal_mean(rb, hs, 7),
            "lvol_same_slot_1d": shift(lvol, STEPS_PER_DAY - hs),
            "lvol_same_slot_7d": shift(lvol, STEPS_PER_WEEK - hs),
            "lvol_seasmean_7d": seasonal_mean(lvol, hs, 7),
        }
        self._extras[h] = e
        return e

    # ---- training-window statistics (no future information)
    def static_features(self, upto: int) -> dict[str, np.ndarray]:
        """Per-cell statistics computed from slots ``< upto`` only."""
        rb = self.p["rb"][:, :upto]
        lvol = np.log1p(self.p["vol_dl"][:, :upto])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return {
                "cell_rb_mean": np.nanmean(rb, axis=1).astype(np.float32),
                "cell_rb_p90": np.nanpercentile(rb, 90, axis=1).astype(np.float32),
                "cell_idle_share": np.nanmean(np.where(np.isnan(rb), np.nan, (rb < 10).astype(np.float32)), axis=1).astype(np.float32),
                "cell_lvol_mean": np.nanmean(lvol, axis=1).astype(np.float32),
            }

    # ---- targets
    def target(self, name: str, h: int) -> np.ndarray:
        """Value of ``name`` at slot t+4h, indexed by origin t. ``vol`` is returned on the log1p scale."""
        base = self.p["rb"] if name == "rb" else np.log1p(self.p["vol_dl"])
        return shift(base, -h * STEPS_PER_HOUR)

    # ---- row selection and matrix assembly
    def rows(self, name: str, h: int, tt_lo: int, tt_hi: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Origins (cell, t) whose TARGET slot t+4h lies in [tt_lo, tt_hi), with an observed origin and a valid target."""
        hs = h * STEPS_PER_HOUR
        y = self.target(name, h)
        valid = self.p.observed & ~np.isnan(y)
        t_lo, t_hi = max(tt_lo - hs, 0), min(tt_hi - hs, self.T)
        mask = np.zeros_like(valid)
        if t_hi > t_lo:
            mask[:, t_lo:t_hi] = True
        if stride > 1:
            mask[:, np.arange(self.T) % stride != 0] = False
        c, t = np.nonzero(valid & mask)
        return c, t

    def matrix(self, c: np.ndarray, t: np.ndarray, h: int, static: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
        hs = h * STEPS_PER_HOUR
        tt = np.minimum(t + hs, self.T - 1)
        cols: list[np.ndarray] = []
        names: list[str] = []
        for k, a in self.dyn.items():
            cols.append(a[c, t]); names.append(k)
        for k, v in self.cal.items():
            cols.append(v[t]); names.append(f"{k}_t")
        for k in ("slot_sin", "slot_cos", "dow", "weekend", "holiday"):
            cols.append(self.cal[k][tt]); names.append(f"{k}_target")
        for k, a in self.horizon_extras(h).items():
            cols.append(a[c, t]); names.append(k)
        for k, v in static.items():
            cols.append(v[c]); names.append(k)
        cols.append(np.full(len(c), h, dtype=np.float32)); names.append("horizon_h")
        return np.column_stack(cols).astype(np.float32), names
