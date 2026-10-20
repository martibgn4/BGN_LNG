"""SYNTHETIC fixtures for evaluation tests: fixed-contract return targets built from the synthetic
curve, a PIT storage-surprise feature, and a split config. Nothing here is evidence about TTF."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import MARKET_TZ
from ttf_forecaster.evaluation.splits import SplitConfig


def settle_utc(local_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """19:00 Europe/Amsterdam settlement publication time of each local date, in UTC."""
    return (local_dates + pd.Timedelta(hours=19)).tz_localize(MARKET_TZ).tz_convert("UTC")


def build_return_targets(curve: pd.DataFrame, horizon: int, roll_buffer: int = 5) -> pd.DataFrame:
    """TARGET_SCHEMA rows of ``r(t,h) = ln F_c(t+h) - ln F_c(t)`` on the held contract: the nearest
    monthly contract whose last trading day is at least ``h + roll_buffer`` trading days after t.
    Test helper only (the real builder belongs to targets/)."""
    c = curve.loc[curve["field"] == "settle"].copy()
    c["d"] = c["trade_date"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None).dt.normalize()
    F = c.pivot(index="d", columns="contract_id", values="value").sort_index()  # noqa: N806
    days = F.index
    starts = c.groupby("contract_id")["delivery_start"].first().sort_values()
    last_idx = {k: days.get_loc(F[k].last_valid_index()) for k in starts.index}
    rows: list[dict[str, Any]] = []
    pub = settle_utc(days)
    for i in range(len(days) - horizon):
        held = next((k for k in starts.index if last_idx[k] >= i + horizon + roll_buffer), None)
        if held is None:  # end of the synthetic sample: no contract trades long enough
            break
        p0, p1 = F.iat[i, F.columns.get_loc(held)], F.iat[i + horizon, F.columns.get_loc(held)]
        rows.append({
            "origin_time": pub[i],
            "target_id": f"m1_logret_{horizon}",
            "horizon": horizon,
            "value": float(np.log(p1) - np.log(p0)),
            "label_known_time": pub[i + horizon],
            "contract_id": held,
            "construction": "listed",
            "quality_flag": "synthetic",
        })
    out = pd.DataFrame(rows)
    out["horizon"] = out["horizon"].astype("int64")
    return out


def storage_feature(cfg: synthetic.SyntheticConfig, storage: pd.DataFrame,
                    origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Storage surprise known at each origin (published by the origin's local date)."""
    s = synthetic._storage_surprise_by_day(storage, cfg)  # first releases, by publication date
    local = origins.tz_convert(MARKET_TZ).tz_localize(None).normalize()
    vals = s.reindex(s.index.union(local)).ffill().reindex(local).to_numpy()
    return pd.DataFrame({"storage__surprise": vals}, index=origins)


def make_world(world: str, horizon: int = 1, beta: float = 0.004) -> dict[str, Any]:
    cfg = synthetic.SyntheticConfig(world=world, planted_beta=beta)
    storage = synthetic.storage_observations(cfg)
    curve = synthetic.ttf_curve(cfg, storage)
    targets = build_return_targets(curve, horizon)
    X = storage_feature(cfg, storage, pd.DatetimeIndex(targets["origin_time"]))  # noqa: N806
    return {"cfg": cfg, "curve": curve, "targets": targets, "X": X}


@pytest.fixture(scope="session")
def planted_world() -> dict[str, Any]:
    return make_world("planted")


@pytest.fixture(scope="session")
def null_world() -> dict[str, Any]:
    return make_world("null")


@pytest.fixture()
def split_cfg() -> SplitConfig:
    return SplitConfig(min_train_origins=250, holdout_start="2024-07-01", retrain_frequency="block")
