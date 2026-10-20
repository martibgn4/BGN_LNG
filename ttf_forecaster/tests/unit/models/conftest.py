"""Synthetic helpers for model tests. All data is SYNTHETIC; nothing here is evidence of skill."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data import synthetic


def origin_index(n: int, start: str = "2019-01-02") -> pd.DatetimeIndex:
    """Business-day origins at 19:00 Amsterdam, stored as UTC (one row per trading day)."""
    days = pd.bdate_range(start, periods=n)
    return (days + pd.Timedelta(hours=19)).tz_localize("Europe/Amsterdam").tz_convert("UTC")


def linear_world(n: int = 600, k: int = 6, beta: float = 0.5, noise: float = 1.0,
                 seed: int = 0, nan_frac: float = 0.02) -> tuple[pd.DataFrame, pd.Series]:
    """X ~ N(0,1) with family-prefixed names; y = beta * x0 + noise. A few NaNs in X."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n, k))
    y = beta * arr[:, 0] + rng.normal(scale=noise, size=n)
    arr[rng.random((n, k)) < nan_frac] = np.nan
    arr[:, 0] = np.where(np.isnan(arr[:, 0]), 0.0, arr[:, 0])
    cols = [f"{'storage' if j < k // 2 else 'price'}__f{j}" for j in range(k)]
    idx = origin_index(n)
    return pd.DataFrame(arr, index=idx, columns=cols), pd.Series(y, index=idx, name="y")


def null_world_returns(horizon: int, seed: int = 7) -> pd.Series:
    """h-day forward log return of the synthetic TTF factor in the NULL world, by origin.

    Every listed contract carries the same random-walk factor plus a constant per-contract
    premium, so within-contract daily log returns equal the factor's daily return. The h-day
    target at origin t is the sum of the next h daily returns (a fixed-contract return).
    """
    cfg = synthetic.SyntheticConfig(start="2021-01-04", end="2024-12-31", seed=seed, world="null")
    curve = synthetic.ttf_curve(cfg).sort_values(["contract_id", "trade_date"])
    curve["r"] = curve.groupby("contract_id")["value"].transform(lambda s: np.log(s).diff())
    daily = curve.dropna(subset=["r"]).groupby("trade_date")["r"].mean()
    fwd = daily.rolling(horizon).sum().shift(-horizon).dropna()
    fwd.index = (fwd.index.tz_convert("Europe/Amsterdam") + pd.Timedelta(hours=19)).tz_convert("UTC")
    return fwd.rename("y")


@pytest.fixture
def lin_xy() -> tuple[pd.DataFrame, pd.Series]:
    return linear_world()
