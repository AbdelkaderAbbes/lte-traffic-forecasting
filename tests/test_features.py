import copy

import numpy as np

from lte_forecast.baselines import baseline_panels
from lte_forecast.config import ModelCfg
from lte_forecast.data import STEPS_PER_DAY
from lte_forecast.features import FeatureBuilder, shift


def test_shift_semantics():
    a = np.arange(8, dtype=np.float32)[None]
    assert np.array_equal(shift(a, 2)[0, 2:], a[0, :-2]) and np.isnan(shift(a, 2)[0, :2]).all()
    assert np.array_equal(shift(a, -3)[0, :-3], a[0, 3:]) and np.isnan(shift(a, -3)[0, -3:]).all()


def test_target_is_value_h_hours_ahead(small_panel):
    panel, _ = small_panel
    fb = FeatureBuilder(panel, ModelCfg())
    y = fb.target("rb", 2)
    t = 500
    c = 0
    assert y[c, t] == panel["rb"][c, t + 8] or (np.isnan(y[c, t]) and np.isnan(panel["rb"][c, t + 8]))


def test_seasonal_naive_is_aligned_to_target_slot(small_panel):
    panel, _ = small_panel
    rb = panel["rb"]
    for h in (1, 6):
        b = baseline_panels(rb, h)["seasonal_naive_1d"]
        t = 400
        expected = rb[:, t + 4 * h - STEPS_PER_DAY]
        assert np.array_equal(b[:, t], expected, equal_nan=True), "baseline must be yesterday's value AT THE TARGET TIME"


def test_no_future_information_in_features(small_panel):
    """Corrupt everything after t0: features at origins <= t0 must not change."""
    panel, _ = small_panel
    t0 = 600
    base = FeatureBuilder(panel, ModelCfg())
    p2 = copy.deepcopy(panel)
    rng = np.random.default_rng(0)
    for k, a in p2.arrays.items():
        a[:, t0 + 1:] = rng.random(a[:, t0 + 1:].shape).astype(np.float32) * 1000
    p2.observed[:, t0 + 1:] = True
    mod = FeatureBuilder(p2, ModelCfg())
    for k in base.dyn:
        assert np.array_equal(base.dyn[k][:, : t0 + 1], mod.dyn[k][:, : t0 + 1], equal_nan=True), f"leak in {k}"
    for h in (1, 6):
        for k, v in base.horizon_extras(h).items():
            assert np.array_equal(v[:, : t0 + 1], mod.horizon_extras(h)[k][:, : t0 + 1], equal_nan=True), f"leak in {k} (h={h})"


def test_static_features_use_training_window_only(small_panel):
    panel, _ = small_panel
    upto = 1000
    fb = FeatureBuilder(panel, ModelCfg())
    ref = fb.static_features(upto)
    p2 = copy.deepcopy(panel)
    p2.arrays["rb"][:, upto:] = 99.0
    p2.arrays["vol_dl"][:, upto:] = 1e6
    alt = FeatureBuilder(p2, ModelCfg()).static_features(upto)
    for k in ref:
        assert np.array_equal(ref[k], alt[k], equal_nan=True)
