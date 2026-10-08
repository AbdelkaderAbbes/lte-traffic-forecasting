"""Rolling-origin splits defined on the TARGET slot (not the forecast origin).

A row belongs to the block that contains its target slot ``t + 4h``. Training rows therefore have
targets strictly before the calibration block, and calibration rows have targets strictly before the
test block - an embargo of exactly one horizon falls out of the definition with no extra gap logic.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import SplitCfg
from .data import STEPS_PER_DAY


@dataclass(frozen=True)
class Fold:
    k: int
    calib_start: int   # target-slot indices
    test_start: int
    test_end: int

    @property
    def train_end(self) -> int:       # exclusive
        return self.calib_start


def make_folds(n_slots: int, cfg: SplitCfg, min_train_days: int = 14) -> list[Fold]:
    td, cd = cfg.test_days * STEPS_PER_DAY, cfg.calib_days * STEPS_PER_DAY
    folds: list[Fold] = []
    for k in range(cfg.n_folds):
        test_end = n_slots - (cfg.n_folds - 1 - k) * td
        test_start = test_end - td
        calib_start = test_start - cd
        folds.append(Fold(k, calib_start, test_start, test_end))
    need = min_train_days * STEPS_PER_DAY
    if folds[0].calib_start < need:
        raise ValueError(
            f"Not enough history: first fold would train on {folds[0].calib_start / STEPS_PER_DAY:.1f} days "
            f"(< {min_train_days}). Reduce n_folds / test_days / calib_days."
        )
    return folds
