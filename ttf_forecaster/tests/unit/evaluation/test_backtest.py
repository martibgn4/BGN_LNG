"""Walk-forward end-to-end on planted vs null synthetic worlds; trading-simulation conventions.

Uses sklearn LinearRegression as a stand-in model (no dependency on models/). SYNTHETIC only:
these numbers test the harness and are not evidence about TTF.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tests.unit.evaluation.conftest import make_world, settle_utc
from ttf_forecaster.data.base import PREDICTION_SCHEMA, PointInTimeViolation, validate_frame
from ttf_forecaster.evaluation.backtest import (
    TradingConventions,
    ZeroForecast,
    simulate_trading,
    walk_forward,
)
from ttf_forecaster.evaluation.splits import (
    Fold,
    HoldoutLockedError,
    RefitSegment,
    holdout_fold,
    make_splits,
)
from ttf_forecaster.evaluation.statistical_tests import evaluate_success_criteria
from ttf_forecaster.settings import ConfigError

PLANTED_BETA = 0.012  # stronger than the generator default so the harness check is not marginal


def _run(world: dict[str, Any], split_cfg) -> Any:
    folds = make_splits(world["targets"], cfg=split_cfg, valid_origins=world["X"].index)
    return walk_forward(LinearRegression, world["X"], world["targets"], folds, model_id="lin")


@pytest.mark.parametrize("h", [1, 5])
def test_planted_world_found(split_cfg, h: int) -> None:
    res = _run(make_world("planted", h, beta=PLANTED_BETA), split_cfg)
    crit = evaluate_success_criteria(res.paired("lin", "zero"), h, clean_pit=True)
    assert crit["criterion_2"]["oos_r2_all"] > 0.05
    assert crit["criterion_1"]["dm_p_one_sided"] < 0.05
    assert crit["criterion_1"]["passed"]


@pytest.mark.parametrize("h", [1, 5])
def test_null_world_not_found(split_cfg, h: int) -> None:
    res = _run(make_world("null", h), split_cfg)
    crit = evaluate_success_criteria(res.paired("lin", "zero"), h, clean_pit=True)
    assert crit["criterion_2"]["oos_r2_all"] < 0.005
    assert crit["criterion_1"]["dm_p_one_sided"] > 0.05
    assert not crit["criterion_1"]["passed"]


def test_prediction_provenance(planted_world, split_cfg) -> None:
    res = _run(planted_world, split_cfg)
    p = res.predictions
    validate_frame(p, PREDICTION_SCHEMA)
    assert set(p["model_id"]) == {"lin", "zero"}
    assert p["is_synthetic"].all()
    assert (p["train_end"] <= p["origin_time"]).all()
    assert p["feature_hash"].nunique() == 1
    assert (p.loc[p.model_id == "zero", "prediction"] == 0).all()
    assert (res.fit_log["last_train_origin"] < res.fit_log["refit_time"]).all()
    # model and benchmark scored on identical origins
    a = p.loc[p.model_id == "lin", "origin_time"].to_numpy()
    b = p.loc[p.model_id == "zero", "origin_time"].to_numpy()
    assert (a == b).all()


def test_leaky_fold_rejected(planted_world) -> None:
    t = planted_world["targets"]
    o = pd.DatetimeIndex(t["origin_time"])
    seg = RefitSegment(o[300], o[:300], o[300:330])  # h=1 label of o[299] known at o[300]: ok
    walk_forward(LinearRegression, planted_world["X"], t,
                 [Fold("ok", "outer", o[300:330], (seg,), 0, 0, 1)])
    t5 = make_world("planted", 5)["targets"]
    o5 = pd.DatetimeIndex(t5["origin_time"])
    bad = RefitSegment(o5[300], o5[:300], o5[300:330])  # last 4 labels still open: leak
    with pytest.raises(PointInTimeViolation):
        walk_forward(LinearRegression, make_world("planted", 5)["X"], t5,
                     [Fold("bad", "outer", o5[300:330], (bad,), 0, 0, 5)])


def test_holdout_fold_needs_unlock(planted_world, split_cfg) -> None:
    t = planted_world["targets"]
    lkt = pd.Series(t["label_known_time"].to_numpy(), index=pd.DatetimeIndex(t["origin_time"]))
    with pytest.warns(UserWarning):
        hf = holdout_fold(pd.DatetimeIndex(t["origin_time"]), lkt, 1, split_cfg,
                          unlock_holdout=True)
    with pytest.raises(HoldoutLockedError):
        walk_forward(LinearRegression, planted_world["X"], t, [hf])
    res = walk_forward(LinearRegression, planted_world["X"], t, [hf], unlock_holdout=True)
    assert (res.predictions["fold_id"] == "holdout").all()


# ---------------------------------------------------------------------------------------------
# Trading simulation on a hand-built market
# ---------------------------------------------------------------------------------------------


def _market(n: int = 60, roll_at: int | None = None) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    days = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(4)
    path = 30 + np.cumsum(rng.normal(0, 0.5, n))
    prices = pd.concat([
        pd.DataFrame({"trade_date": days.tz_localize("Europe/Amsterdam").tz_convert("UTC"),
                      "contract_id": c, "field": "settle", "value": path + off})
        for c, off in (("TTF.M.2024-03", 0.0), ("TTF.M.2024-04", 1.0))
    ], ignore_index=True)
    origins = settle_utc(days)
    contract = np.where(np.arange(n) < (roll_at or n), "TTF.M.2024-03", "TTF.M.2024-04")
    targets = pd.DataFrame({
        "origin_time": origins, "target_id": "m1_logret_1", "horizon": np.int64(1),
        "value": 0.0, "label_known_time": origins, "contract_id": contract,
        "construction": "listed", "quality_flag": "synthetic",
    })
    fc = pd.Series(rng.normal(0, 0.01, n), index=origins)
    return fc, targets, prices


def test_execution_delay_shifts_pnl() -> None:
    fc, targets, prices = _market()
    base = TradingConventions(cost_multiplier=0.0)
    d1 = simulate_trading(fc, targets, prices, base.with_(execution_delay=1)).daily
    d2 = simulate_trading(fc, targets, prices, base.with_(execution_delay=2)).daily
    p1 = d1.set_index("trade_date")["position_held"]
    p2 = d2.set_index("trade_date")["position_held"]
    shifted = p1.shift(1, freq="B")
    common = shifted.index.intersection(p2.index)[:-1]
    assert np.allclose(shifted.loc[common], p2.loc[common])  # same positions, one day later
    g1 = d1.set_index("trade_date")["gross_pnl"].reindex(common)
    g2 = d2.set_index("trade_date")["gross_pnl"].reindex(common)
    assert not np.allclose(g1, g2)  # ...earning different price changes
    # first fill is the settlement AFTER the origin, never the same one
    assert d1["trade_date"].iloc[0] == pd.Timestamp("2024-01-03")
    with pytest.raises(ConfigError):
        TradingConventions(execution_delay=0)


def test_costs_reduce_net_pnl_monotonically() -> None:
    fc, targets, prices = _market()
    nets = [simulate_trading(fc, targets, prices,
                             TradingConventions(cost_multiplier=k)).summary["net_pnl"]
            for k in (0.0, 1.0, 2.0, 4.0)]
    gross = simulate_trading(fc, targets, prices, TradingConventions()).summary["gross_pnl"]
    assert nets[0] == pytest.approx(gross)
    assert all(a > b for a, b in zip(nets, nets[1:], strict=False))


def test_roll_trades_both_legs_and_no_cross_roll_pnl() -> None:
    fc, targets, prices = _market(roll_at=30)
    fc[:] = 0.02  # constant long, full size
    d = simulate_trading(fc, targets, prices, TradingConventions(sizing_method="sign")).daily
    roll = d.loc[d["is_roll"]]
    assert len(roll) == 1 and roll["traded"].iloc[0] == pytest.approx(2.0)
    # On the roll day the held (old) contract's own change is used: the +1.0 inter-contract
    # offset never enters P&L.
    old = prices.loc[prices.contract_id == "TTF.M.2024-03"].set_index("trade_date")["value"]
    old.index = old.index.tz_convert("Europe/Amsterdam").tz_localize(None)
    r = roll.index[0]
    assert d.loc[r, "contract_held"] == "TTF.M.2024-03"
    assert d.loc[r, "price_change"] == pytest.approx(old.diff().loc[d.loc[r, "trade_date"]])
    assert d.loc[r + 1, "contract_held"] == "TTF.M.2024-04"


def test_exercise_is_stated_and_hedging_not_faked() -> None:
    fc, targets, prices = _market()
    res = simulate_trading(fc, targets, prices, TradingConventions.from_config())
    assert res.statement.startswith("FORECAST EVALUATION")
    with pytest.raises(NotImplementedError):
        simulate_trading(fc, targets, prices, TradingConventions(exercise="hedging_improvement"))


def test_zero_forecast_is_flat() -> None:
    fc, targets, prices = _market()
    res = simulate_trading(fc * 0, targets, prices, TradingConventions())
    assert res.summary["gross_pnl"] == 0 and res.summary["total_cost"] == 0
    assert ZeroForecast().fit(None, None).predict(np.zeros((3, 1))).tolist() == [0, 0, 0]
