"""Markdown summary written next to the raw CSV results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "horizon":
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("-" if np.isnan(v) else format(v, fmt.get(c, ".3f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(out: Path, cfg: Config, audit: dict, metrics: pd.DataFrame, calib: pd.DataFrame,
                  boot: pd.DataFrame, sleep: pd.DataFrame, synthetic: bool) -> None:
    L: list[str] = []
    if synthetic:
        L += ["> **SYNTHETIC DATA - pipeline smoke test. These numbers say nothing about real networks. Do not publish.**", ""]
    pa, ra = audit.get("panel", {}), audit.get("raw", {})
    L += ["## Data audit",
          f"- Raw rows: {ra.get('n_rows')}, cells: {pa.get('cells_kept')} kept of {pa.get('grid_cells')}, span: {pa.get('grid_days')} days "
          f"({ra.get('ts_min')} to {ra.get('ts_max')})",
          f"- Exact duplicate rows removed: {ra.get('n_exact_duplicate_rows')}; same-key rows merged: {audit.get('cleaning', {}).get('rows_merged_same_key')}",
          f"- RB utilisation NaN share: {ra.get('rb_nan_share')}; of which rows with DL volume reported as 0 (idle, set to 0 %): {ra.get('rb_nan_share_with_zero_volume')}",
          f"- Idle slots (RB = 0) in the grid: {pa.get('idle_share_of_observed')}; true gaps: {pa.get('gap_slots_share')}",
          f"- Max RB utilisation in the data: {ra.get('rb_max')} % (share above 85 %: {ra.get('rb_share_above_85')})", ""]

    rb = metrics[metrics.target == "rb"]
    agg = rb.groupby(["horizon", "model"])["mae"].agg(["mean", "std"]).reset_index()
    pv = agg.pivot(index="horizon", columns="model", values="mean").reset_index()
    cols = ["horizon", "xgb", "persistence", "seasonal_naive_1d", "seasonal_naive_7d", "seasonal_mean_7d"]
    L += ["## RB utilisation: MAE (%-points), mean over test folds", md_table(pv[cols]), ""]
    b = boot.copy()
    b["skill_pct"] = 100 * b["skill"]
    L += ["## Model vs strongest baseline (pooled test rows; 95 % block-bootstrap CI of MAE difference, model - baseline)",
          md_table(b[["horizon", "best_baseline", "mae_model", "mae_baseline", "diff", "diff_lo", "diff_hi", "skill_pct"]]), ""]
    w = rb[rb.model == "xgb"].groupby("horizon")[["rmse", "wape", "bias"]].mean().reset_index()
    w["wape"] *= 100
    L += ["## XGBoost RB: other point metrics", md_table(w), ""]

    vol = metrics[metrics.target == "vol"]
    if len(vol):
        agg = vol.groupby(["horizon", "model"])["wape"].mean().reset_index().pivot(index="horizon", columns="model", values="wape").reset_index()
        for c in agg.columns[1:]:
            agg[c] *= 100
        L += ["## DL data volume: WAPE (%), mean over test folds", md_table(agg[[c for c in cols if c in agg.columns]]), ""]

    cal = calib[calib.target == "rb"].groupby("nominal")[["coverage_raw", "coverage_conformal"]].mean().reset_index()
    L += ["## Upper-bound coverage on test blocks (RB, mean over horizons and folds)", md_table(cal), ""]

    for h in sorted(sleep.horizon.unique()):
        s = sleep[(sleep.horizon == h) & (sleep.fold == "all") & ~sleep.policy.str.startswith("point_thr_")]
        keep = ["policy", "sleep_share", "false_sleep_rate", "severe_rate", "displaced_share", "energy_saved_pct", "capture", "precision", "recall"]
        L += [f"## Sleep-mode policies, {h} h ahead (theta = {cfg.sleep.theta} % RB, residual power {cfg.sleep.sleep_power_fraction:.0%})",
              md_table(s[keep]), ""]
    (out / "summary.md").write_text("\n".join(L))