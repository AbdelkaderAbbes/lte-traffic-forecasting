"""Typed configuration, loaded from a TOML file (stdlib ``tomllib``, no extra dependency)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class DataCfg:
    path: str = "data/Dataset_02_LTE_1800.csv"
    idle_fill: bool = True          # RB utilisation NaN + zero traffic + zero users  ->  0 % (idle), not a gap
    min_cell_coverage: float = 0.5  # drop cells observed in < 50 % of the global timeline
    max_rb: float = 100.0


@dataclass
class SplitCfg:
    n_folds: int = 3                # rolling-origin folds (expanding training window)
    test_days: int = 7
    calib_days: int = 7             # held-out block used ONLY for conformal calibration
    train_stride: int = 2           # use every k-th 15-min slot for training (neighbouring slots are ~duplicates)


@dataclass
class ModelCfg:
    horizons_h: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    rb_quantiles: list[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 0.95, 0.99])
    vol_quantiles: list[float] = field(default_factory=lambda: [0.5, 0.9])
    n_rounds: int = 250
    learning_rate: float = 0.08
    max_depth: int = 6
    min_child_weight: float = 20.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    max_bin: int = 128
    n_jobs: int = 0                 # 0 = all cores
    seed: int = 42
    holidays: list[str] = field(default_factory=lambda: [
        "2025-01-01", "2025-01-06", "2025-04-18", "2025-04-21", "2025-05-01", "2025-05-08",
    ])


@dataclass
class SleepCfg:
    theta: float = 10.0             # sleep if (forecast) RB utilisation < theta %
    severe_theta: float = 30.0      # a "severe" false sleep: actual load >= this
    alphas: list[float] = field(default_factory=lambda: [0.2, 0.1, 0.05, 0.01])
    sleep_power_fraction: float = 0.3   # residual power of a sleeping radio unit (assumption, see docs)
    point_thresholds: list[float] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10])
    horizons_h: list[int] = field(default_factory=lambda: [1, 3, 6])


@dataclass
class BootstrapCfg:
    n_boot: int = 500
    seed: int = 7


@dataclass
class Config:
    out_dir: str = "results"
    data: DataCfg = field(default_factory=DataCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    sleep: SleepCfg = field(default_factory=SleepCfg)
    bootstrap: BootstrapCfg = field(default_factory=BootstrapCfg)

    @classmethod
    def from_toml(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        return _build(cls, raw)

    def fast(self) -> "Config":
        """Quick smoke-test profile (1 fold, 3 horizons, few trees)."""
        self.split.n_folds = 1
        self.split.train_stride = 4
        self.model.horizons_h = [1, 3, 6]
        self.model.n_rounds = 60
        self.model.rb_quantiles = [0.5, 0.9, 0.95]
        self.model.vol_quantiles = [0.5, 0.9]
        self.sleep.alphas = [0.1, 0.05]
        self.sleep.horizons_h = [1, 3]
        self.bootstrap.n_boot = 100
        return self


def _build(cls, raw: dict):
    names = {f.name: f for f in fields(cls)}
    kwargs = {}
    for key, value in raw.items():
        if key not in names:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        f = names[key]
        default = f.default_factory() if f.default_factory is not None else None  # type: ignore[misc]
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)
