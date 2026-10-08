#!/usr/bin/env python
"""Run the full experiment.

    python scripts/run_pipeline.py --data data/Dataset_02_LTE_1800.csv
    python scripts/run_pipeline.py --synthetic --fast          # smoke test, no download needed
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import tomllib
from dataclasses import is_dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lte_forecast.config import Config  # noqa: E402
from lte_forecast.data import load_panel  # noqa: E402
from lte_forecast.pipeline import run_experiment  # noqa: E402
from lte_forecast.synthetic import make_synthetic  # noqa: E402


def load_config(path: str | None) -> Config:
    """Read the TOML file and apply it on top of the defaults (self-contained, no dependency on Config.from_toml)."""
    cfg = Config()
    if not path or not Path(path).exists():
        return cfg
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    for key, value in raw.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown config key '{key}'")
        current = getattr(cfg, key)
        if is_dataclass(current) and isinstance(value, dict):
            for k, v in value.items():
                if not hasattr(current, k):
                    raise KeyError(f"Unknown config key '{key}.{k}'")
                setattr(current, k, v)
        else:
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--data", help="path to Dataset_02_LTE_1800.csv (overrides config)")
    ap.add_argument("--out", help="output directory (overrides config)")
    ap.add_argument("--synthetic", action="store_true", help="use generated data (smoke test only)")
    ap.add_argument("--fast", action="store_true", help="1 fold, 3 horizons, few trees")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fast:
        cfg.fast()
    if args.data:
        cfg.data.path = args.data
    if args.out:
        cfg.out_dir = args.out

    watermark = None
    if args.synthetic:
        watermark = "SYNTHETIC DATA"
        tmp = Path(tempfile.mkdtemp()) / "synthetic.csv"
        make_synthetic(n_sites=12, days=70).to_csv(tmp, index=False)
        cfg.data.path = str(tmp)
        if not args.out:
            cfg.out_dir = "results_synthetic"
    elif not Path(cfg.data.path).exists():
        sys.exit(f"Data file not found: {cfg.data.path}\nRun `python scripts/download_data.py` first (see data/README.md).")

    panel, audit = load_panel(cfg.data)
    run_experiment(cfg, panel, audit, watermark=watermark)


if __name__ == "__main__":
    main()