"""Fixtures for the fundamentals tests. Every frame is SYNTHETIC."""

from __future__ import annotations

import pandas as pd
import pytest

from ttf_forecaster.data.base import DataBundle
from ttf_forecaster.data.fundamentals import (
    SyntheticFundamentalsConfig,
    load_fundamentals_registry,
    synthetic_fundamentals,
)


@pytest.fixture(scope="session")
def registry() -> dict:
    return load_fundamentals_registry()


@pytest.fixture(scope="session")
def world_cfg() -> SyntheticFundamentalsConfig:
    return SyntheticFundamentalsConfig(start="2017-10-01", end="2024-03-31", seed=5)


@pytest.fixture(scope="session")
def world(registry: dict, world_cfg: SyntheticFundamentalsConfig) -> pd.DataFrame:
    return synthetic_fundamentals(registry, world_cfg)


@pytest.fixture()
def bundle(world: pd.DataFrame) -> DataBundle:
    b = DataBundle.empty()
    b.observations = world.copy()
    return b


def local_origins(*days: str, hour: int = 19) -> pd.DatetimeIndex:
    """Origins at ``hour`` local time on the given dates (tz-aware)."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(days))) + pd.Timedelta(hours=hour)
    return idx.tz_localize("Europe/Amsterdam")
