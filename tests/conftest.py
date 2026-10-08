import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lte_forecast.config import DataCfg  # noqa: E402
from lte_forecast.data import load_panel  # noqa: E402
from lte_forecast.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def small_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "syn.csv"
    make_synthetic(n_sites=3, days=30, seed=3).to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def small_panel(small_csv):
    panel, audit = load_panel(DataCfg(path=str(small_csv)))
    return panel, audit
