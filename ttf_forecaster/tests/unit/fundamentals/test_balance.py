"""Daily balance: identity, NaN propagation, units, residual monitoring. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.contracts.units import GasQuality, UnitError
from ttf_forecaster.data.base import SYNTHETIC_SOURCE, QualityFlag
from ttf_forecaster.data.fundamentals import SeriesSpec, observation_rows
from ttf_forecaster.fundamentals.balance import (
    build_balance,
    compute_balance,
    contributions,
    load_balance_config,
    monitor_residual,
    to_gwhd,
)

DAYS = pd.date_range("2024-01-01", periods=10, freq="D")


def _spec(sid: str, unit: str) -> SeriesSpec:
    return SeriesSpec.from_config(sid, {"vendor": "public", "ticker": None, "unit": unit,
                                        "status": "requires_discovery", "pit_grade": "B"}, "test")


def _rows(sid: str, values, unit: str = "GWh/d") -> pd.DataFrame:
    pub = pd.DatetimeIndex(DAYS).tz_localize("UTC") + pd.Timedelta(days=1)
    return observation_rows(_spec(sid, unit), DAYS, np.asarray(values, float), source=SYNTHETIC_SOURCE,
                            quality_flag=QualityFlag.SYNTHETIC.value, entitlement="synthetic",
                            publication_time=pub)


def _cfg() -> dict:
    return {
        "perimeter": {"name": "toy"},
        "gas_quality": {"gcv_kwh_per_m3": 10.55, "reference": "GCV", "source": "test"},
        "components": {
            "production": {"sign": 1, "series": ["p"]},
            "lng_sendout": {"sign": 1, "series": ["lng"]},
            "demand": {"sign": -1, "series": ["d"]},
        },
        "balancing": {"treatment": "absorbed_in_residual"},
        "observed_storage_change": {"method": "injection_minus_withdrawal",
                                    "injection_series": "inj", "withdrawal_series": "wd"},
        "residual_monitoring": {"window_days": 5, "min_periods": 3, "abs_band_gwhd": 100.0,
                                "mean_band_gwhd": 50.0, "z_band": 3.0},
    }


def _toy(lng_values=None, d_unit="GWh/d", d_values=None) -> pd.DataFrame:
    n = len(DAYS)
    lng = np.full(n, 300.0) if lng_values is None else lng_values
    d = np.full(n, 1000.0) if d_values is None else d_values
    return pd.concat([
        _rows("p", np.full(n, 500.0)), _rows("lng", lng), _rows("d", d, d_unit),
        _rows("inj", np.zeros(n)), _rows("wd", np.full(n, 190.0)),
    ], ignore_index=True)


class TestIdentity:
    def test_residual_exact(self) -> None:
        res = compute_balance(_toy(), _cfg(), DAYS[0], DAYS[-1])
        t = res.table
        np.testing.assert_allclose(t["implied_storage_change_gwhd"], -200.0)
        np.testing.assert_allclose(t["observed_storage_change_gwhd"], -190.0)
        np.testing.assert_allclose(t["residual_gwhd"], 10.0)
        assert (t["residual_reason"] == "").all()
        assert any("absorbed in the residual" in n for n in res.notes)

    def test_missing_component_is_nan_with_reason(self) -> None:
        lng = np.full(len(DAYS), 300.0)
        lng[3] = np.nan
        t = compute_balance(_toy(lng_values=lng), _cfg(), DAYS[0], DAYS[-1]).table
        assert np.isnan(t["lng_sendout"].iloc[3]) and np.isnan(t["residual_gwhd"].iloc[3])
        assert "lng_sendout:lng" in t["residual_reason"].iloc[3]
        assert t["production"].iloc[3] == 500.0  # other components unaffected, nothing zero-filled

    def test_absent_series_never_zero(self) -> None:
        obs = _toy()
        obs = obs.loc[obs["series_id"] != "lng"]
        t = compute_balance(obs, _cfg(), DAYS[0], DAYS[-1]).table
        assert t["lng_sendout"].isna().all() and t["residual_gwhd"].isna().all()

    def test_volume_converted_with_gcv(self) -> None:
        t = compute_balance(_toy(d_unit="mcm/d", d_values=np.full(len(DAYS), 100.0)), _cfg(),
                            DAYS[0], DAYS[-1]).table
        np.testing.assert_allclose(t["demand"], -1055.0)  # 100 mcm/d x 10.55 kWh/m3

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(UnitError):
            to_gwhd(pd.Series([1.0]), "therm/d", GasQuality(10.55, "GCV"))

    def test_point_in_time(self) -> None:
        # as_of before the last day's publication -> last day unknown -> NaN, not extrapolated
        as_of = pd.Timestamp(DAYS[-1]).tz_localize("UTC") + pd.Timedelta(hours=12)
        t = build_balance(_toy(), as_of, DAYS[0], DAYS[-1], _cfg()).table
        assert np.isnan(t["residual_gwhd"].iloc[-1]) and np.isfinite(t["residual_gwhd"].iloc[-2])


class TestMonitoring:
    def test_spike_flagged_and_missing_reported(self) -> None:
        d = np.full(len(DAYS), 1000.0)
        d[7] = 1500.0
        lng = np.full(len(DAYS), 300.0)
        lng[2] = np.nan
        t = compute_balance(_toy(lng_values=lng, d_values=d), _cfg(), DAYS[0], DAYS[-1]).table
        m = monitor_residual(t, _cfg()["residual_monitoring"])
        assert bool(m["flag_abs"].iloc[7]) and m["flag_any"].sum() >= 1
        assert bool(m["residual_missing"].iloc[2]) and not bool(m["flag_any"].iloc[2])

    def test_contributions_nan_for_incomplete_period(self) -> None:
        lng = np.full(len(DAYS), 300.0)
        lng[0] = np.nan
        res = compute_balance(_toy(lng_values=lng), _cfg(), DAYS[0], DAYS[-1])
        c = contributions(res, freq="MS")
        assert np.isnan(c["lng_sendout"].iloc[0]) and c["production"].iloc[0] == 500.0
        assert c["n_complete_days"].iloc[0] == len(DAYS) - 1


class TestSyntheticWorld:
    def test_residual_small_and_unbiased(self, world: pd.DataFrame) -> None:
        cfg = load_balance_config()
        res = build_balance(world, pd.Timestamp("2024-03-31", tz="UTC"),
                            pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), cfg)
        r = res.table["residual_gwhd"].dropna()
        assert len(r) > 350
        assert abs(r.mean()) < 30.0 and r.std() < 200.0  # balancing noise sd 80 + revisions
