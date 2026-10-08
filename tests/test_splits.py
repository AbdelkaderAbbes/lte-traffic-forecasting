import numpy as np
import pytest

from lte_forecast.config import ModelCfg, SplitCfg
from lte_forecast.data import STEPS_PER_DAY
from lte_forecast.features import FeatureBuilder
from lte_forecast.splits import make_folds


def test_blocks_are_ordered_and_cover_the_end():
    folds = make_folds(80 * STEPS_PER_DAY, SplitCfg(n_folds=3, test_days=7, calib_days=7))
    assert folds[-1].test_end == 80 * STEPS_PER_DAY
    for f in folds:
        assert f.calib_start < f.test_start < f.test_end
    assert folds[0].test_end == folds[1].test_start


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        make_folds(30 * STEPS_PER_DAY, SplitCfg(n_folds=3))


def test_train_calib_test_targets_do_not_overlap(small_panel):
    panel, _ = small_panel
    fb = FeatureBuilder(panel, ModelCfg())
    f = make_folds(panel.shape[1], SplitCfg(n_folds=1, test_days=4, calib_days=4))[0]
    for h in (1, 6):
        hs = 4 * h
        _, t_tr = fb.rows("rb", h, 0, f.calib_start)
        _, t_ca = fb.rows("rb", h, f.calib_start, f.test_start)
        _, t_te = fb.rows("rb", h, f.test_start, f.test_end)
        assert (t_tr + hs).max() < f.calib_start <= (t_ca + hs).min()
        assert (t_ca + hs).max() < f.test_start <= (t_te + hs).min()
        assert np.all(t_te + hs < panel.shape[1])
