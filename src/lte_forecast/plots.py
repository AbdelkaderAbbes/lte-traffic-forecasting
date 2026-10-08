"""Figures. Every figure carries a visible watermark when produced from synthetic data."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
COL = {"xgb": "#1f6feb", "persistence": "#8b949e", "seasonal_naive_1d": "#d29922", "seasonal_naive_7d": "#bf8700",
       "seasonal_mean_7d": "#2da44e", "oracle": "#000000"}


def _finish(fig, path: Path, watermark: str | None):
    if watermark:
        fig.text(0.5, 0.5, watermark, fontsize=34, color="red", alpha=0.18, ha="center", va="center", rotation=25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def eda_profiles(panel, out: Path, wm=None):
    ix = panel.index
    rb = pd.DataFrame(panel["rb"].T, index=ix)
    obs = pd.DataFrame(panel.observed.T, index=ix)
    slot = ix.hour + ix.minute / 60
    wk = np.where(ix.dayofweek >= 5, "weekend", "weekday")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for lab in ("weekday", "weekend"):
        m = wk == lab
        prof = rb[m].mean(axis=1).groupby(slot[m]).mean()
        ax[0].plot(prof.index, prof.values, label=lab)
    ax[0].set(title="Mean RB utilisation by time of day", xlabel="hour", ylabel="RB utilisation (%)"); ax[0].legend()
    vals = panel["rb"][panel.observed]
    ax[1].hist(vals, bins=80, color=COL["xgb"], log=True)
    ax[1].set(title="RB utilisation distribution (log count)", xlabel="RB utilisation (%)")
    idle = ((rb < 10) & obs).mean(axis=1).groupby(slot).mean()
    ax[2].plot(idle.index, 100 * idle.values, color="tab:green")
    ax[2].set(title="Share of cell-slots below 10 % RB (sleep candidates)", xlabel="hour", ylabel="%")
    _finish(fig, out / "eda_profiles.png", wm)


def eda_energy(panel, out: Path, wm=None):
    rb, e = panel["rb"][panel.observed], panel["energy_ru"][panel.observed]
    ok = ~np.isnan(rb) & ~np.isnan(e)
    rb, e = rb[ok], e[ok]
    bins = np.linspace(0, max(rb.max(), 1), 21)
    ix = np.digitize(rb, bins)
    xs = [rb[ix == i].mean() for i in range(1, len(bins)) if (ix == i).sum() > 50]
    ys = [e[ix == i].mean() for i in range(1, len(bins)) if (ix == i).sum() > 50]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set(title="Radio-unit energy vs RB utilisation (binned mean)", xlabel="RB utilisation (%)", ylabel="radio unit energy (dataset units)")
    _finish(fig, out / "eda_energy_vs_load.png", wm)


def horizon_curves(metrics: pd.DataFrame, target: str, out: Path, wm=None):
    d = metrics[metrics.target == target].groupby(["model", "horizon"], as_index=False)[["mae", "wape"]].mean()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for model, g in d.groupby("model"):
        c = COL["xgb"] if model.startswith("xgb") else COL.get(model)
        ax[0].plot(g.horizon, g.mae, marker="o", label=model, color=c, lw=2.2 if model.startswith("xgb") else 1.4)
        ax[1].plot(g.horizon, 100 * g.wape, marker="o", label=model, color=c, lw=2.2 if model.startswith("xgb") else 1.4)
    unit = "%" if target == "rb" else "log1p(volume)"
    ax[0].set(title=f"{target.upper()} - MAE vs horizon", xlabel="horizon (h)", ylabel=f"MAE ({unit})"); ax[0].legend(fontsize=8)
    ax[1].set(title=f"{target.upper()} - WAPE vs horizon", xlabel="horizon (h)", ylabel="WAPE (%)")
    _finish(fig, out / f"horizon_curves_{target}.png", wm)


def calibration_plot(calib: pd.DataFrame, out: Path, wm=None):
    d = calib[calib.target == "rb"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="ideal")
    for lab, col, mk in (("coverage_raw", "tab:red", "s"), ("coverage_conformal", COL["xgb"], "o")):
        g = d.groupby("nominal", as_index=False)[lab].mean()
        ax.plot(g.nominal, g[lab], marker=mk, color=col, label=lab.replace("coverage_", "").replace("conformal", "conformal-calibrated"))
    ax.set(title="Upper-bound coverage on test blocks (RB)", xlabel="nominal coverage", ylabel="empirical coverage"); ax.legend()
    _finish(fig, out / "calibration_rb.png", wm)


def pareto(sleep: pd.DataFrame, h: int, out: Path, wm=None):
    d = sleep[sleep.horizon == h]
    fig, ax = plt.subplots(figsize=(7, 5))
    pt = d[d.policy.str.startswith("point_thr_")].sort_values("false_sleep_rate")
    ax.plot(100 * pt.false_sleep_rate, 100 * pt.capture, marker=".", color=COL["xgb"], alpha=0.6, label="XGB median, threshold sweep")
    cf = d[d.policy.str.startswith("xgb_conformal")]
    ax.scatter(100 * cf.false_sleep_rate, 100 * cf.capture, marker="*", s=170, color="tab:red", zorder=5, label="XGB conformal upper bound")
    for _, r in cf.iterrows():
        ax.annotate("alpha=" + r.policy.split("_a")[-1], (100 * r.false_sleep_rate, 100 * r.capture), textcoords="offset points", xytext=(6, -10), fontsize=8)
    for name in ("persistence", "seasonal_naive_1d", "seasonal_mean_7d", "xgb_median"):
        r = d[d.policy == name]
        if len(r):
            ax.scatter(100 * r.false_sleep_rate, 100 * r.capture, s=60, label=name, color=COL.get(name, "tab:purple"), zorder=4)
    ax.set(title=f"Sleep-mode trade-off, {h} h ahead", xlabel="false-sleep rate  P(RB >= theta | slept)  (%)", ylabel="share of oracle energy saving captured (%)")
    ax.legend(fontsize=8)
    _finish(fig, out / f"sleep_pareto_{h}h.png", wm)


def example_forecast(ex: pd.DataFrame, cell: str, h: int, out: Path, wm=None):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(ex.ts, ex.y, color="black", lw=1.4, label="actual")
    ax.plot(ex.ts, ex.q50, color=COL["xgb"], lw=1.4, label="XGB median")
    ax.fill_between(ex.ts, ex.q50, ex.u90, color=COL["xgb"], alpha=0.2, label="calibrated 90 % upper bound")
    ax.plot(ex.ts, ex.persistence, color=COL["persistence"], lw=0.9, alpha=0.8, label="persistence")
    ax.axhline(10, color="tab:green", ls=":", label="sleep threshold")
    ax.set(title=f"{h} h-ahead RB utilisation forecast, {cell} (one test week)", ylabel="RB utilisation (%)"); ax.legend(ncol=5, fontsize=8)
    _finish(fig, out / f"example_forecast_{h}h.png", wm)


def importance_plot(imp: pd.DataFrame, out: Path, wm=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, (h, g) in zip(ax, imp.groupby("horizon")):
        g = g.sort_values("gain").tail(12)
        a.barh(g.feature, g.gain, color=COL["xgb"]); a.set_title(f"RB model - feature gain, {h} h ahead")
    _finish(fig, out / "feature_importance_rb.png", wm)
