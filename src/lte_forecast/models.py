"""Multi-quantile gradient boosting with the native XGBoost API (no scikit-learn)."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from .config import ModelCfg


def fit_quantiles(X: np.ndarray, y: np.ndarray, names: list[str], quantiles: list[float], m: ModelCfg) -> xgb.Booster:
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": np.asarray(quantiles, dtype=float),
        "tree_method": "hist",
        "eta": m.learning_rate,
        "max_depth": m.max_depth,
        "min_child_weight": m.min_child_weight,
        "subsample": m.subsample,
        "colsample_bytree": m.colsample_bytree,
        "max_bin": m.max_bin,
        "seed": m.seed,
        "nthread": m.n_jobs,
        "verbosity": 0,
    }
    dtr = xgb.DMatrix(X, label=y, feature_names=names)
    return xgb.train(params, dtr, num_boost_round=m.n_rounds)


def predict_quantiles(booster: xgb.Booster, X: np.ndarray, names: list[str]) -> np.ndarray:
    pred = booster.predict(xgb.DMatrix(X, feature_names=names))
    pred = pred.reshape(len(X), -1)
    return np.sort(pred, axis=1)      # enforce non-crossing quantiles
