import numpy as np

from lte_forecast.conformal import conformal_offset
from lte_forecast.metrics import mae, paired_block_bootstrap, pinball, wape
from lte_forecast.sleep import evaluate_policy


def test_basic_metrics():
    y, p = np.array([1.0, 2, 3]), np.array([1.0, 3, 5])
    assert mae(y, p) == 1.0 and abs(wape(y, p) - 0.5) < 1e-9
    assert abs(pinball(np.array([1.0]), np.array([0.0]), 0.9) - 0.9) < 1e-9


def test_conformal_upper_bound_reaches_nominal_coverage():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60000)
    y = x + rng.normal(size=60000)
    q_bad = x - 1.0                        # deliberately miscalibrated upper quantile
    cal, test = slice(0, 20000), slice(20000, None)
    off = conformal_offset(y[cal], q_bad[cal], 0.1)
    cov = np.mean(y[test] <= q_bad[test] + off)
    assert 0.885 < cov < 0.915
    assert np.mean(y[test] <= q_bad[test]) < 0.7


def test_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(200), 20)
    a = np.abs(rng.normal(1.0, 0.2, len(g)))
    b = np.abs(rng.normal(1.5, 0.2, len(g)))
    r = paired_block_bootstrap(a, b, g, 300, 0)
    assert r["hi"] < 0 < 1 and r["diff"] < 0


def test_oracle_policy_is_perfect_and_false_sleep_gets_no_credit():
    rng = np.random.default_rng(2)
    y = rng.uniform(0, 30, 5000)
    vol, en = rng.uniform(0, 10, 5000), np.full(5000, 20.0)
    o = evaluate_policy(y < 10, y, vol, en, 10, 30, 0.3)
    assert o["false_sleep_rate"] == 0 and abs(o["capture"] - 1) < 1e-9
    always = evaluate_policy(np.ones(5000, bool), y, vol, en, 10, 30, 0.3)
    assert always["capture"] <= 1 and always["false_sleep_rate"] > 0.5
