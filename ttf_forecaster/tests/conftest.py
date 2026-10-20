"""Shared fixtures. All data is SYNTHETIC (ttf_forecaster.data.synthetic)."""

from __future__ import annotations

import pandas as pd
import pytest

from ttf_forecaster.data import synthetic
from ttf_forecaster.settings import load_config


@pytest.fixture(autouse=True)
def _clear_config_cache() -> None:
    load_config.cache_clear()


@pytest.fixture(scope="session")
def syn_cfg() -> synthetic.SyntheticConfig:
    """Short synthetic world (null: prices independent of fundamentals)."""
    return synthetic.SyntheticConfig(start="2022-01-03", end="2023-12-29")


@pytest.fixture(scope="session")
def syn_storage(syn_cfg: synthetic.SyntheticConfig) -> pd.DataFrame:
    return synthetic.storage_observations(syn_cfg)


@pytest.fixture(scope="session")
def syn_curve(syn_cfg: synthetic.SyntheticConfig) -> pd.DataFrame:
    return synthetic.ttf_curve(syn_cfg)


@pytest.fixture(scope="session")
def syn_weather(syn_cfg: synthetic.SyntheticConfig) -> pd.DataFrame:
    return synthetic.weather_forecast_runs(syn_cfg)
