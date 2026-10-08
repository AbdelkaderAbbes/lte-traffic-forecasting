"""End-to-end experiment: rolling-origin training, evaluation, conformal calibration, sleep-mode analysis."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .baselines import BASELINES, baseline_panels
from .config import Config
from .conformal import conformal_offset
from .data import STEPS_PER_DAY, STEPS_PER_HOUR, Panel
from .features import FeatureBuilder
from .metrics import coverage, paired_block_bootstrap, pinball, point_metrics
from .models import fit_quantiles, predict_quantiles
from .report import write_summary
from .sleep import evaluate_policy
from .splits import make_folds


def _level_index(quantiles: list[float], level: float) -> int:
    for i, q in enumerate(quantiles):
        if abs(q - level) < 1e-9:
            return i
    raise ValueError(f"Quantile level {level} must be listed in the model quantiles {quantiles}")


def run_experiment(cfg: Config, panel: Panel, audit: dict, watermark: str | None = None, verbose: bool = True) -> dict[str, pd.DataFrame]:
    t_start = time.time()
    out = Path(cfg.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    m, sl = cfg.model, cfg.sleep
    for a in sl.alphas:
        _level_index(m.rb_quantiles, 1 - a)
    _level_index(m.rb_quantiles, 0.5)
    _level_index(m.vol_quantiles, 0.5)

    fb = FeatureBuilder(panel, m)
    folds = make_folds(panel.shape[1], cfg.split)
    C, T = panel.shape
    energy = panel["energy_ru"].copy()
    med = np.nanmedian(energy, axis=1)
    med = np.where(np.isnan(med), np.nanmedian(energy), med)
    energy = np.where(np.isnan(energy), med[:, None], energy)
    log(f"panel: {C} cells x {T} slots ({T / STEPS_PER_DAY:.0f} days); folds: " +
        ", ".join(f"[{f.calib_start // STEPS_PER_DAY}d|{f.test_start // STEPS_PER_DAY}d-{f.test_end // STEPS_PER_DAY}d]" for f in folds))

    metric_rows, calib_rows, imp_rows = [], [], []
    store: dict[int, list[dict]] = {h: [] for h in m.horizons_h}

    for fold in folds:
        static = fb.static_features(fold.train_end - max(m.horizons_h) * STEPS_PER_HOUR)
        for target in ("rb", "vol"):
            quant = m.rb_quantiles if target == "rb" else m.vol_quantiles
            xfull = panel["rb"] if target == "rb" else panel["vol_dl"]
            for h in m.horizons_h:
                hs = h * STEPS_PER_HOUR
                t0 = time.time()
                c, t = fb.rows(target, h, 0, fold.train_end, cfg.split.train_stride)
                X, names = fb.matrix(c, t, h, static)
                booster = fit_quantiles(X, fb.target(target, h)[c, t], names, quant, m)

                def predict(lo, hi):
                    cc, tt = fb.rows(target, h, lo, hi)
                    Xp, _ = fb.matrix(cc, tt, h, static)
                    Q = predict_quantiles(booster, Xp, names)
                    if target == "vol":
                        Q = np.clip(np.expm1(Q), 0, None)
                    return cc, tt, Q, xfull[cc, tt + hs]

                cc_cal, tt_cal, Q_cal, y_cal = predict(fold.calib_start, fold.test_start)
                cc, tt, Q, y = predict(fold.test_start, fold.test_end)
                bp = baseline_panels(xfull, h)
                B_cal = {k: v[cc_cal, tt_cal] for k, v in bp.items()}
                B = {k: v[cc, tt] for k, v in bp.items()}
                common = np.all([~np.isnan(v) for v in B.values()], axis=0)
                common_cal = np.all([~np.isnan(v) for v in B_cal.values()], axis=0)
                i50 = _level_index(quant, 0.5)

                preds = {"xgb": Q[:, i50], **B}
                for name, p in preds.items():
                    metric_rows.append({"fold": fold.k, "target": target, "horizon": h, "model": name,
                                        "n": int(common.sum()), **point_metrics(y[common], p[common])})

                bounds = {}
                for j, level in enumerate(quant):
                    if level <= 0.5:
                        continue
                    off = conformal_offset(y_cal[common_cal], Q_cal[common_cal, j], 1 - level)
                    U = Q[:, j] + off
                    if target == "rb":
                        U = np.clip(U, 0, 100)
                    bounds[round(1 - level, 6)] = U
                    calib_rows.append({"fold": fold.k, "target": target, "horizon": h, "nominal": level, "offset": off,
                                       "coverage_raw": coverage(y[common], Q[common, j]),
                                       "coverage_conformal": coverage(y[common], U[common]),
                                       "pinball_raw": pinball(y[common], Q[common, j], level),
                                       "mean_margin_over_median": float(np.mean(U[common] - Q[common, i50]))})
                if target == "rb":
                    store[h].append({
                        "fold": np.full(common.sum(), fold.k), "c": cc[common], "tt": (tt + hs)[common], "y": y[common],
                        "q50": Q[common, i50], "bounds": {a: u[common] for a, u in bounds.items()},
                        "base": {k: v[common] for k, v in B.items()},
                        "vol": np.nan_to_num(panel["vol_dl"][cc, tt + hs][common]),
                        "energy": energy[cc, tt + hs][common],
                    })
                    if fold.k == folds[-1].k and h in (min(m.horizons_h), max(m.horizons_h)):
                        gain = booster.get_score(importance_type="gain")
                        imp_rows += [{"horizon": h, "feature": k, "gain": v} for k, v in gain.items()]
                log(f"  fold {fold.k} {target:3s} h={h}  rows train={len(c):>7d} test={int(common.sum()):>6d}  {time.time() - t0:5.1f}s")

    # ----------------------------------------------------------------------------- aggregate
    metrics = pd.DataFrame(metric_rows)
    calib = pd.DataFrame(calib_rows)
    imp = pd.DataFrame(imp_rows)

    boot_rows = []
    for h, parts in store.items():
        y = np.concatenate([p["y"] for p in parts]); q50 = np.concatenate([p["q50"] for p in parts])
        grp = np.concatenate([p["c"] * 100000 + p["tt"] // STEPS_PER_DAY for p in parts])
        base = {k: np.concatenate([p["base"][k] for p in parts]) for k in BASELINES}
        best = min(BASELINES, key=lambda k: np.mean(np.abs(y - base[k])))
        r = paired_block_bootstrap(np.abs(y - q50), np.abs(y - base[best]), grp, cfg.bootstrap.n_boot, cfg.bootstrap.seed)
        mb = float(np.mean(np.abs(y - base[best])))
        boot_rows.append({"horizon": h, "best_baseline": best, "mae_model": float(np.mean(np.abs(y - q50))), "mae_baseline": mb,
                          "diff": r["diff"], "diff_lo": r["lo"], "diff_hi": r["hi"], "skill": 1 - float(np.mean(np.abs(y - q50))) / mb})
    boot = pd.DataFrame(boot_rows)

    sleep_rows = []
    for h in sl.horizons_h:
        if h not in store:
            continue
        parts = store[h]
        for fold_sel in ["all"] + [f.k for f in folds]:
            sel = [p for p in parts if fold_sel == "all" or p["fold"][0] == fold_sel]
            if not sel:
                continue
            cat = lambda key: np.concatenate([p[key] for p in sel])
            y, q50, vol, en = cat("y"), cat("q50"), cat("vol"), cat("energy")
            base = {k: np.concatenate([p["base"][k] for p in sel]) for k in BASELINES}
            pol: dict[str, np.ndarray] = {"oracle": y < sl.theta, "persistence": base["persistence"] < sl.theta,
                                          "seasonal_naive_1d": base["seasonal_naive_1d"] < sl.theta,
                                          "seasonal_mean_7d": base["seasonal_mean_7d"] < sl.theta,
                                          "xgb_median": q50 < sl.theta}
            for a in sl.alphas:
                U = np.concatenate([p["bounds"][round(a, 6)] for p in sel])
                pol[f"xgb_conformal_a{a}"] = U < sl.theta
            for thr in sl.point_thresholds:
                pol[f"point_thr_{thr:g}"] = q50 < thr
            for name, d in pol.items():
                sleep_rows.append({"horizon": h, "fold": fold_sel, "policy": name,
                                   **evaluate_policy(d, y, vol, en, sl.theta, sl.severe_theta, sl.sleep_power_fraction)})
    sleep = pd.DataFrame(sleep_rows)

    # ----------------------------------------------------------------------------- write
    metrics.to_csv(out / "metrics_by_fold.csv", index=False)
    calib.to_csv(out / "calibration.csv", index=False)
    boot.to_csv(out / "rb_vs_best_baseline.csv", index=False)
    sleep.to_csv(out / "sleep_policy.csv", index=False)
    imp.to_csv(out / "feature_importance_rb.csv", index=False)
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    fig = out / "figures"
    plots.eda_profiles(panel, fig, watermark)
    plots.eda_energy(panel, fig, watermark)
    plots.horizon_curves(metrics, "rb", fig, watermark)
    plots.horizon_curves(metrics, "vol", fig, watermark)
    plots.calibration_plot(calib, fig, watermark)
    for h in sl.horizons_h:
        if (sleep.horizon == h).any():
            plots.pareto(sleep[sleep.fold == "all"], h, fig, watermark)
    if len(imp):
        plots.importance_plot(imp, fig, watermark)
    h_ex = min(sl.horizons_h)
    if h_ex in store and 0.1 in store[h_ex][-1]["bounds"]:
        last = store[h_ex][-1]
        cell = int(np.bincount(last["c"], weights=last["y"]).argmax())
        s = last["c"] == cell
        order = np.argsort(last["tt"][s])
        ex = pd.DataFrame({"ts": panel.index[last["tt"][s][order]], "y": last["y"][s][order], "q50": last["q50"][s][order],
                           "u90": last["bounds"][0.1][s][order], "persistence": last["base"]["persistence"][s][order]})
        plots.example_forecast(ex, str(panel.cells[cell]), h_ex, fig, watermark)

    write_summary(out, cfg, audit, metrics, calib, boot, sleep, synthetic=bool(watermark))
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")
    return {"metrics": metrics, "calibration": calib, "bootstrap": boot, "sleep": sleep}
