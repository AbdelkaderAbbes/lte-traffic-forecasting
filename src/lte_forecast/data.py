"""Loading, cleaning and gridding of the Zenodo LTE PM-counter data.

Design rules
------------
* Every (cell, 15-min slot) lives on a complete grid. Slots with no measurement stay NaN (gap).
* An *idle* slot (zero DL volume and zero active DL users but RB utilisation reported as NaN) is
  **not** a gap: the counter is simply not computed when nothing is scheduled. It is set to 0 %.
  Dropping these rows (as ``dropna`` would) removes exactly the low-load periods that matter most
  for energy saving and biases every metric.
* Nothing is ever ``fillna(0)``-ed blindly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import DataCfg

STEP_MIN = 15
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96
STEPS_PER_WEEK = 672

RENAME = {
    "Base station": "site",
    "Sector": "sector",
    "Timestamp": "ts",
    "Radio unit energy consumption": "energy_ru",
    "Baseband energy consumption": "energy_bb",
    "4G max active users DL": "max_users_dl",
    "4G max active users UL": "max_users_ul",
    "4G data volume DL": "vol_dl",
    "4G data volume UL": "vol_ul",
    "4G max RRC users": "max_rrc",
    "4G RB utilization": "rb",
    "4G CQI rank 1": "cqi1",
    "4G CQI rank 2": "cqi2",
    "4G CQI rank 3": "cqi3",
    "4G CQI rank 4": "cqi4",
    "4G RRC users": "rrc",
    "4G active users UL": "users_ul",
    "4G active users DL": "users_dl",
    "4G MIMO rank DL": "mimo",
}
SUM_COLS = ["vol_dl", "vol_ul", "users_dl", "users_ul", "rrc", "max_users_dl", "max_users_ul", "max_rrc"]
MEAN_COLS = ["rb", "cqi1", "cqi2", "cqi3", "cqi4", "mimo", "energy_ru", "energy_bb"]
KEY_COUNTERS = ["rb", "vol_dl", "users_dl"]
CQI_COLS = ["cqi1", "cqi2", "cqi3", "cqi4"]


@dataclass
class Panel:
    """Dense (cells x time) arrays on a complete 15-minute grid."""

    cells: np.ndarray                 # (C,) cell ids "Site 61_S1"
    index: pd.DatetimeIndex           # (T,)
    arrays: dict[str, np.ndarray]     # name -> (C, T) float32, NaN = no measurement
    observed: np.ndarray              # (C, T) bool: any key counter present

    @property
    def shape(self) -> tuple[int, int]:
        return self.observed.shape

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]


# --------------------------------------------------------------------------------------- loading
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RENAME if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    df = df[list(RENAME)].rename(columns=RENAME)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def audit_raw(df: pd.DataFrame) -> dict[str, Any]:
    """Facts about the raw table that justify (or refute) the cleaning rules."""
    a: dict[str, Any] = {}
    a["n_rows"] = int(len(df))
    a["n_sites"] = int(df["site"].nunique())
    a["n_cells"] = int(df.groupby(["site", "sector"]).ngroups)
    a["ts_min"], a["ts_max"] = str(df["ts"].min()), str(df["ts"].max())
    off_grid = (df["ts"].dt.minute % STEP_MIN != 0) | (df["ts"].dt.second != 0)
    a["n_off_grid_timestamps"] = int(off_grid.sum())
    a["n_exact_duplicate_rows"] = int(df.duplicated().sum())
    a["n_duplicate_keys_after_exact_dedup"] = int(df.drop_duplicates().duplicated(["site", "sector", "ts"]).sum())
    a["missing_share_by_column"] = {c: round(float(df[c].isna().mean()), 4) for c in RENAME.values() if c not in ("site", "sector", "ts")}
    rb_nan = df["rb"].isna()
    idle = (df["vol_dl"] == 0) & (df["users_dl"] == 0)
    a["rb_nan_share"] = round(float(rb_nan.mean()), 4)
    a["rb_nan_share_that_is_zero_traffic"] = round(float((rb_nan & idle).sum() / max(rb_nan.sum(), 1)), 4)
    a["rb_zero_traffic_share_that_is_nan"] = round(float((rb_nan & idle).sum() / max(idle.sum(), 1)), 4)
    vol_zero, vol_seen = df["vol_dl"] == 0, df["vol_dl"].notna()
    a["rb_nan_share_with_zero_volume"] = round(float((rb_nan & vol_zero).sum() / max(rb_nan.sum(), 1)), 4)
    a["rb_nan_share_with_observed_volume"] = round(float((rb_nan & vol_seen).sum() / max(rb_nan.sum(), 1)), 4)
    a["rb_nan_zero_volume_rows_with_users_nan"] = round(float((rb_nan & vol_zero & df["users_dl"].isna()).sum() / max((rb_nan & vol_zero).sum(), 1)), 4)
    a["rb_max"] = float(df["rb"].max())
    a["rb_share_above_85"] = round(float((df["rb"] > 85).mean()), 6)
    a["negative_values"] = {c: int((df[c] < 0).sum()) for c in SUM_COLS + MEAN_COLS if (df[c] < 0).any()}
    return a


# --------------------------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame, cfg: DataCfg) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dedup, validate and aggregate to one row per (cell, 15-min slot)."""
    info: dict[str, Any] = {}
    n0 = len(df)
    df = df.drop_duplicates().copy()
    info["rows_removed_exact_duplicates"] = int(n0 - len(df))

    df["cell"] = df["site"].astype(str) + "_S" + df["sector"].astype(str)
    df["ts"] = df["ts"].dt.floor(f"{STEP_MIN}min")
    num = SUM_COLS + MEAN_COLS
    df[num] = df[num].astype("float32")
    for c in num:                                    # a counter cannot be negative
        df.loc[df[c] < 0, c] = np.nan
    info["rb_clipped_above_max"] = int((df["rb"] > cfg.max_rb).sum())
    df["rb"] = df["rb"].clip(upper=cfg.max_rb)

    keys = ["cell", "ts"]
    g = df.groupby(keys, sort=True)
    agg = pd.concat([g[SUM_COLS].sum(min_count=1), g[MEAN_COLS].mean()], axis=1)
    info["rows_merged_same_key"] = int(len(df) - len(agg))   # multi-carrier / repeated rows -> sum volumes, mean ratios

    if cfg.idle_fill:
        # Idle slot: DL volume was REPORTED as exactly 0 while RB utilisation (and usually the user counters)
        # are NaN - the scheduler-based counters are simply not computed when nothing is scheduled.
        # Users are accepted as 0 or NaN: on the real file they are NaN in these rows.
        users_zero_or_nan = agg["users_dl"].isna() | (agg["users_dl"] == 0)
        idle = agg["rb"].isna() & (agg["vol_dl"] == 0) & users_zero_or_nan
        info["rb_idle_filled_slots"] = int(idle.sum())
        agg.loc[idle, "rb"] = 0.0
        agg.loc[idle & agg["users_dl"].isna(), "users_dl"] = 0.0
    for c in CQI_COLS:                               # CQI index 0 = "out of range / no report", not a measurement
        agg.loc[agg[c] == 0, c] = np.nan
    return agg.reset_index(), info


def build_panel(clean_df: pd.DataFrame, cfg: DataCfg) -> tuple[Panel, dict[str, Any]]:
    t0, t1 = clean_df["ts"].min(), clean_df["ts"].max()
    index = pd.date_range(t0, t1, freq=f"{STEP_MIN}min")
    cells = np.sort(clean_df["cell"].unique())
    c_idx = pd.Index(cells).get_indexer(clean_df["cell"])
    t_idx = ((clean_df["ts"] - t0) // pd.Timedelta(minutes=STEP_MIN)).to_numpy().astype(int)

    C, T = len(cells), len(index)
    arrays: dict[str, np.ndarray] = {}
    for col in SUM_COLS + MEAN_COLS:
        a = np.full((C, T), np.nan, dtype=np.float32)
        a[c_idx, t_idx] = clean_df[col].to_numpy(dtype=np.float32)
        arrays[col] = a
    cq = np.stack([arrays[c] for c in CQI_COLS])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices are expected (idle slots)
        arrays["cqi_mean"] = np.nanmean(cq, axis=0).astype(np.float32)

    observed = np.zeros((C, T), dtype=bool)
    for k in KEY_COUNTERS:
        observed |= ~np.isnan(arrays[k])

    info: dict[str, Any] = {"grid_cells": int(C), "grid_slots": int(T), "grid_days": round(T / STEPS_PER_DAY, 1)}
    cov = observed.mean(axis=1)
    keep = cov >= cfg.min_cell_coverage
    info["cells_dropped_low_coverage"] = [str(c) for c in cells[~keep]]
    info["cell_coverage_median"] = round(float(np.median(cov)), 4)
    panel = Panel(cells[keep], index, {k: v[keep] for k, v in arrays.items()}, observed[keep])
    info["cells_kept"] = int(keep.sum())
    info["observed_share_kept"] = round(float(panel.observed.mean()), 4)
    info["idle_share_of_observed"] = round(float(((panel["rb"] == 0) & panel.observed).sum() / panel.observed.sum()), 4)
    info["gap_slots_share"] = round(float(1 - panel.observed.mean()), 4)
    info["longest_gap_slots"] = int(_longest_gap(panel.observed))
    return panel, info


def _longest_gap(observed: np.ndarray) -> int:
    best = 0
    for row in observed:
        miss = ~row
        if not miss.any():
            continue
        edges = np.diff(np.concatenate([[0], miss.astype(int), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        best = max(best, int((ends - starts).max()))
    return best


def load_panel(cfg: DataCfg) -> tuple[Panel, dict[str, Any]]:
    raw = load_raw(cfg.path)
    audit = {"raw": audit_raw(raw)}
    cleaned, cinfo = clean(raw, cfg)
    panel, pinfo = build_panel(cleaned, cfg)
    audit["cleaning"], audit["panel"] = cinfo, pinfo
    return panel, audit