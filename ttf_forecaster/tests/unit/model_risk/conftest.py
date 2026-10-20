"""SYNTHETIC toy worlds for the model-risk harness. Nothing here is evidence about TTF.

``toy_world`` builds a feature matrix and a TARGET_SCHEMA frame directly (no feature builders), so
the harness tests are fast:

    daily return  r(d) = beta * s(d-1) + e(d),   s, e ~ white
    target        y(t) = r(t+1) + ... + r(t+h)   (label known at origin t+h)

Columns: ``storage__signal`` = s(t) (legitimate, known at t), ``price__ret_1d`` = r(t) (past
return), ``weather__noise`` / ``lng__noise`` (pure noise), ``calendar__dow``. With
``leak=True`` the column ``flows__leak`` = r(t+1) is added: a deliberate look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.evaluation.splits import SplitConfig


def session_origins(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    days = pd.bdate_range(start, periods=n)
    return (days + pd.Timedelta(hours=19)).tz_localize("Europe/Amsterdam").tz_convert("UTC")


def toy_world(n: int = 700, h: int = 1, beta: float = 0.5, seed: int = 0,
              leak: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    m = n + h + 2
    s = rng.normal(size=m)
    e = rng.normal(size=m)
    r = np.r_[0.0, beta * s[:-1]] + e
    idx = session_origins(m)
    y = pd.Series(r).rolling(h).sum().shift(-h).to_numpy()
    X = pd.DataFrame({
        "price__ret_1d": r,
        "storage__signal": s,
        "weather__noise": rng.normal(size=m),
        "lng__noise": rng.normal(size=m),
        "calendar__dow": idx.tz_convert("Europe/Amsterdam").dayofweek.astype(float),
    }, index=idx)
    if leak:
        X["flows__leak"] = np.r_[r[1:], np.nan]
    X.index.name = "origin_time"
    lkt = pd.Series(idx).shift(-h)
    t = pd.DataFrame({
        "origin_time": idx, "target_id": "toy_logret", "horizon": np.int64(h), "value": y,
        "label_known_time": lkt, "contract_id": "TTF-M-TOY", "construction": "single_contract",
        "quality_flag": "synthetic",
    }).iloc[:n].reset_index(drop=True)
    t["label_known_time"] = pd.to_datetime(t["label_known_time"], utc=True)
    t["contract_id"] = t["contract_id"].astype(object)
    return X.iloc[:n], t


def toy_split() -> SplitConfig:
    return SplitConfig(test_block_months=3, min_train_origins=150, min_test_origins=20,
                       holdout_start="2035-01-01")


@pytest.fixture(scope="module")
def planted() -> tuple[pd.DataFrame, pd.DataFrame]:
    return toy_world(beta=0.5, seed=1)


@pytest.fixture(scope="module")
def null() -> tuple[pd.DataFrame, pd.DataFrame]:
    return toy_world(beta=0.0, seed=2)
