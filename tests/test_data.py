import numpy as np

from lte_forecast.config import DataCfg
from lte_forecast.data import clean, load_raw


def test_idle_slots_are_zero_not_dropped_and_gaps_stay_nan(small_panel):
    panel, audit = small_panel
    rb, vol, users = panel["rb"], panel["vol_dl"], panel["users_dl"]
    idle = (vol == 0) & (users == 0)
    assert idle.any()
    assert not np.any(np.isnan(rb[idle])), "zero-traffic slots must carry a utilisation value, not NaN"
    assert np.any(rb[idle] == 0), "idle slots reported as NaN must become 0 %"
    gaps = ~panel.observed
    assert gaps.any(), "synthetic outages should survive as gaps"
    assert np.all(np.isnan(rb[gaps])), "gaps must NOT be zero-filled"


def test_duplicates_collapse_to_one_row_per_cell_slot(small_csv):
    df = load_raw(str(small_csv))
    out, info = clean(df, DataCfg())
    assert not out.duplicated(["cell", "ts"]).any()
    assert info["rows_removed_exact_duplicates"] > 0


def test_cqi_zero_is_not_a_measurement(small_panel):
    panel, _ = small_panel
    assert not np.any(panel["cqi1"] == 0)


def test_audit_reports_idle_semantics(small_panel):
    _, audit = small_panel
    assert audit["raw"]["rb_nan_share_that_is_zero_traffic"] > 0.99
