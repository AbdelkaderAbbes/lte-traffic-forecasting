from lte_forecast.config import Config
from lte_forecast.pipeline import run_experiment


def test_end_to_end_tiny(small_panel, tmp_path):
    panel, audit = small_panel
    cfg = Config()
    cfg.out_dir = str(tmp_path)
    cfg.split.n_folds, cfg.split.test_days, cfg.split.calib_days, cfg.split.train_stride = 1, 3, 3, 4
    cfg.model.horizons_h, cfg.model.n_rounds = [1, 3], 15
    cfg.model.rb_quantiles, cfg.model.vol_quantiles = [0.5, 0.9], [0.5, 0.9]
    cfg.sleep.alphas, cfg.sleep.horizons_h = [0.1], [1]
    cfg.bootstrap.n_boot = 20
    res = run_experiment(cfg, panel, audit, watermark="SYNTHETIC DATA", verbose=False)
    for f in ("summary.md", "metrics_by_fold.csv", "sleep_policy.csv", "data_audit.json", "figures/sleep_pareto_1h.png"):
        assert (tmp_path / f).exists(), f
    cal = res["calibration"]
    assert (cal.coverage_conformal.sub(cal.nominal).abs() < 0.08).all()
