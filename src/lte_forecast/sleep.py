"""Decision layer: forecast-driven carrier sleep mode.

A policy decides, at origin ``t``, whether the cell may sleep during slot ``t + h``.
Ground truth for scoring is the realised RB utilisation in that slot.

Metrics (all on the test blocks)
  false_sleep_rate : P(actual RB >= theta | policy slept)      -> users would have been affected
  severe_rate      : P(actual RB >= severe_theta | slept)
  displaced_share  : share of total DL volume that fell into slept slots (must be absorbed by other layers)
  energy_saved     : radio-unit energy avoided by CORRECT sleeps, as % of total radio-unit energy (scenario: see sleep_power_fraction)
  capture          : energy_saved / energy_saved of the oracle that sleeps exactly when actual RB < theta
"""
from __future__ import annotations

import numpy as np


def evaluate_policy(decide: np.ndarray, y: np.ndarray, vol: np.ndarray, energy: np.ndarray, theta: float,
                    severe_theta: float, sleep_fraction: float) -> dict[str, float]:
    decide = decide.astype(bool)
    n = int(decide.sum())
    total_e, total_v = energy.sum(), vol.sum()
    oracle = decide_oracle = y < theta
    # Energy is credited only for correct sleeps. A false sleep is assumed to be noticed and the cell woken
    # (no saving credited); its cost is reported separately via false_sleep_rate / displaced_share.
    saved = energy[decide & decide_oracle].sum() * (1 - sleep_fraction)
    oracle_saved = energy[oracle].sum() * (1 - sleep_fraction)
    tp = int((decide & decide_oracle).sum())
    return {
        "sleep_share": n / len(y),
        "false_sleep_rate": float((y[decide] >= theta).mean()) if n else float("nan"),
        "severe_rate": float((y[decide] >= severe_theta).mean()) if n else float("nan"),
        "displaced_share": float(vol[decide].sum() / total_v) if total_v > 0 else float("nan"),
        "energy_saved_pct": float(100 * saved / total_e),
        "capture": float(saved / oracle_saved) if oracle_saved > 0 else float("nan"),
        "precision": tp / n if n else float("nan"),
        "recall": tp / max(int(oracle.sum()), 1),
    }
