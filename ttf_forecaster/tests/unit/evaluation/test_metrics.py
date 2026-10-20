"""Metric definitions on hand-checkable inputs. SYNTHETIC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.evaluation import metrics as m


def test_point_metrics_basic() -> None:
    y = np.array([1.0, -1.0, 2.0, -2.0])
    out = m.point_metrics(y, y * 0.5, np.zeros(4))
    assert out["mae"] == pytest.approx(0.75)
    assert out["oos_r2"] == pytest.approx(0.75)  # 1 - 0.25
    assert out["bias"] == pytest.approx(0.0)
    assert np.isnan(out["mase"])  # no training scale given -> NaN, never test-scaled
    assert m.point_metrics(y, y, np.zeros(4), mase_scale_=1.0)["mase"] == 0.0
    assert m.oos_r2(y, np.zeros(4), np.zeros(4)) == 0.0


def test_nan_rows_dropped_jointly() -> None:
    y = np.array([1.0, np.nan, 2.0])
    out = m.point_metrics(y, [1.0, 5.0, np.nan], [0.0, 0.0, 0.0])
    assert out["n"] == 1 and out["oos_r2"] == 1.0


def test_mase_scale() -> None:
    assert m.mase_scale([1.0, -3.0]) == 2.0
    assert m.mase_scale([1.0, 3.0, 6.0], naive="lag") == 2.5


def test_classification() -> None:
    y = np.array([1, 1, 0, 0])
    out = m.classification_metrics(y, [0.9, 0.2, 0.1, 0.6])
    assert out["balanced_accuracy"] == pytest.approx(0.5)
    assert out["brier"] == pytest.approx((0.01 + 0.64 + 0.01 + 0.36) / 4)
    assert m.classification_metrics(y, [0.9, 0.8, 0.1, 0.2])["mcc"] == pytest.approx(1.0)
    cal = m.calibration_curve(y, [0.9, 0.8, 0.1, 0.2], n_bins=2)
    assert list(cal["n"]) == [2, 2] and list(cal["observed_freq"]) == [0.0, 1.0]


def test_probabilistic() -> None:
    y = np.array([0.0, 1.0])
    assert m.pinball_loss(y, [0.5, 0.5], 0.5) == pytest.approx(0.25)
    iv = m.interval_metrics(y, [-1.0, 1.5], [1.0, 2.0])
    assert iv["coverage"] == 0.5 and iv["width"] == 1.25
    q = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    crps = m.crps_from_quantiles(y, q, [0.1, 0.5, 0.9])
    assert crps > 0
    with pytest.raises(ValueError):
        m.interval_metrics(y, [1.0, 1.0], [0.0, 0.0])


def test_trading_metrics() -> None:
    pnl = np.array([1.0, -2.0, 3.0, -1.0, 0.0])
    daily = pd.DataFrame({"gross_pnl": pnl, "net_pnl": pnl, "cost": 0.0, "traded": 0.0,
                          "position_held": [1, 1, 1, 1, 0]})
    out = m.trading_metrics(daily)
    assert out["max_drawdown"] == pytest.approx(2.0)
    assert out["hit_rate"] == 0.5
    assert out["avg_win"] == 2.0 and out["avg_loss"] == -1.5
    assert m.pnl_concentration(pnl) == pytest.approx(3.0)  # best day / total
    assert np.isnan(m.pnl_concentration(-np.abs(pnl)))


def test_grouping() -> None:
    t = pd.date_range("2022-06-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    paired = pd.DataFrame({"origin_time": t, "fold_id": "f", "actual": y, "model": 0.5 * y,
                           "bench": 0.0, "horizon": 1})
    regimes = pd.Series(np.where(np.arange(400) < 200, "tight", "normal"), index=t)
    g = m.add_group_columns(paired, regimes)
    by = m.point_metrics_by(g, ["year", "regime"])
    assert set(by["year"]) == {2022, 2023}
    assert np.allclose(by["oos_r2"], 0.75)
