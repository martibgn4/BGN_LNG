"""Price / curve / cross-commodity families on SYNTHETIC data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.contracts.units import MMBTU_PER_MWH
from ttf_forecaster.data.base import MARKET_TZ, DataBundle, NonPositivePriceError
from ttf_forecaster.features import prices
from ttf_forecaster.pipelines.build_features import family_params

from .helpers import bundle, config, settle_origins


@pytest.fixture(scope="module")
def b() -> DataBundle:
    return bundle()


def _params(family: str) -> dict[str, Any]:
    return family_params(config(), family)


def _local(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=MARKET_TZ).tz_convert("UTC")


def test_spliced_returns_invariant_to_per_contract_scaling(b: DataBundle) -> None:
    """Within-contract returns do not change if each contract's level is scaled; a naive
    continuous series differenced across rolls would."""
    p = _params("price")
    base = prices.front_return_series(b.curves, b.contracts, p)
    curves = b.curves.copy()
    ids = curves["contract_id"].unique()
    scale = dict(zip(ids, np.random.default_rng(0).uniform(0.5, 2.0, len(ids)), strict=True))
    curves["value"] = curves["value"] * curves["contract_id"].map(scale)
    scaled = prices.front_return_series(curves, b.contracts, p)
    np.testing.assert_allclose(base["log_return"], scaled["log_return"], equal_nan=True)
    # and the level series itself does jump (so the test is not vacuous)
    assert not np.allclose(base["front_settle"], scaled["front_settle"])


def test_spliced_return_equals_held_contract_return(b: DataBundle) -> None:
    p = _params("price")
    s = prices.front_return_series(b.curves, b.contracts, p).set_index("date")
    ttf = b.curves.loc[b.curves["hub"] == "TTF"].copy()
    ttf["date"] = ttf["trade_date"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    px = ttf.pivot(index="date", columns="delivery_start", values="value")
    for t in s.index[5::97]:
        prev = s.index[s.index.get_loc(t) - 1]
        held = s.at[prev, "front_month"].tz_localize(MARKET_TZ).tz_convert("UTC")
        expect = np.log(px.at[t, held]) - np.log(px.at[prev, held])
        assert s.at[t, "log_return"] == pytest.approx(expect, abs=1e-12)


def test_front_roll_rule(b: DataBundle) -> None:
    """April-2022 last trade = Wed 30 Mar (2 bdays before Fri 1 Apr). On Tue 22 Mar there are 6
    bdays to it (> R=5) -> front Apr; on Wed 23 Mar only 5 -> front May."""
    s = prices.front_return_series(b.curves, b.contracts, _params("price")).set_index("date")
    assert s.at[pd.Timestamp("2022-03-22"), "front_month"] == pd.Timestamp("2022-04-01")
    assert s.at[pd.Timestamp("2022-03-23"), "front_month"] == pd.Timestamp("2022-05-01")


def test_price_features_respect_settlement_publication(b: DataBundle) -> None:
    p = _params("price")
    o = pd.DatetimeIndex([_local("2022-06-15 18:59"), _local("2022-06-15 19:00")])
    f = prices.build_price_features(b, o, p)
    s = prices.front_return_series(b.curves, b.contracts, p).set_index("date")
    assert f["price__ret_1d"].iloc[0] == s.at[pd.Timestamp("2022-06-14"), "log_return"]
    assert f["price__ret_1d"].iloc[1] == s.at[pd.Timestamp("2022-06-15"), "log_return"]
    assert f.attrs["pit_grade"]["price__ret_1d"] == "A"
    assert f["price__volume_rel_20d"].isna().all()  # field not in the bundle -> NaN, not 0


def test_rolling_definitions(b: DataBundle) -> None:
    p = _params("price")
    s = prices.front_return_series(b.curves, b.contracts, p)
    f = prices.price_features_from_series(s, p)
    r = s["log_return"]
    i = 200
    assert f["ret_5d"].iloc[i] == pytest.approx(r.iloc[i - 4: i + 1].sum())
    assert f["rvol_20d"].iloc[i] == pytest.approx(r.iloc[i - 19: i + 1].std() * np.sqrt(252))


def test_revised_settlement_is_delayed_not_leaked(b: DataBundle) -> None:
    p = _params("price")
    curves = b.curves.copy()
    day = pd.Timestamp("2022-06-15", tz=MARKET_TZ).tz_convert("UTC")
    rev = curves.loc[(curves["hub"] == "TTF") & (curves["trade_date"] == day)].copy()
    rev["revision"] = 1
    rev["publication_time"] = rev["publication_time"] + pd.Timedelta(days=2)
    bb = DataBundle(b.observations, b.forecasts, pd.concat([curves, rev]), b.contracts)
    o = pd.DatetimeIndex([_local("2022-06-16 19:00"), _local("2022-06-17 19:00")])
    f = prices.build_price_features(bb, o, p)
    s = prices.front_return_series(b.curves, b.contracts, p).set_index("date")
    # on the 16th neither the 15th nor the 16th is usable yet (cummax availability) -> the 14th
    assert f["price__ret_1d"].iloc[0] == s.at[pd.Timestamp("2022-06-14"), "log_return"]
    assert f["price__ret_1d"].iloc[1] == s.at[pd.Timestamp("2022-06-17"), "log_return"]


def test_non_positive_price_raises(b: DataBundle) -> None:
    curves = b.curves.copy()
    curves.loc[curves.index[100], "value"] = -1.0
    with pytest.raises(NonPositivePriceError):
        prices.front_return_series(curves, b.contracts, _params("price"))


def test_return_series_provider_hook(b: DataBundle) -> None:
    p = {**_params("price"), "return_series_provider": "tests.unit.features.leaky_builders:"
         "provider_constant_returns"}
    o = settle_origins("2022-06-01", "2022-06-10")
    f = prices.build_price_features(b, o, p)
    assert (f["price__ret_1d"] == 0.001).all()


def test_curve_features_and_strip_weights(b: DataBundle) -> None:
    p = _params("curve")
    # origin 17 Jan 2022: next Summer = Apr-Sep 2022, following Winter = Oct 2022 - Mar 2023
    o = pd.DatetimeIndex([_local("2022-01-17 19:00")])
    f = prices.build_curve_features(b, o, p)
    ttf = b.curves.loc[(b.curves["hub"] == "TTF")
                       & (b.curves["trade_date"] == pd.Timestamp("2022-01-17", tz=MARKET_TZ))]
    px = ttf.set_index(ttf["delivery_start"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None))["value"]
    m = lambda s: px[pd.Timestamp(s)]  # noqa: E731

    def strip(months: pd.DatetimeIndex) -> float:
        h = prices.month_hours(months)
        return float(np.average([m(x) for x in months], weights=h))

    # sum_minus_win = next Summer - following Winter; the mirror win_minus_sum was dropped (D-032 F5)
    assert "curve__win_minus_sum" not in f.columns
    summer = pd.date_range("2022-04-01", periods=6, freq="MS")
    winter = pd.date_range("2022-10-01", periods=6, freq="MS")
    assert prices.month_hours(winter)[0] == 745 and prices.month_hours(winter)[5] == 743
    assert f["curve__sum_minus_win"].iloc[0] == pytest.approx(strip(summer) - strip(winter),
                                                             rel=1e-9, abs=1e-9)


def test_cross_commodity_uses_next_session_rule(b: DataBundle) -> None:
    p = _params("cross_commodity")
    o = pd.DatetimeIndex([_local("2022-06-15 19:00")])
    f = prices.build_cross_commodity_features(b, o, p)
    s = prices.front_return_series(b.curves, b.contracts, p).set_index("date")
    d_prev, d_same = pd.Timestamp("2022-06-14"), pd.Timestamp("2022-06-15")
    obs = b.observations

    def by_day(sid: str) -> pd.Series:
        x = obs.loc[obs["series_id"] == sid]
        return x.set_index(x["event_time"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None))["value"]

    fx, gbp = by_day("fx.eurusd.spot_usd_per_eur"), by_day("fx.eurgbp.spot_gbp_per_eur")
    ttf_prev = s.at[d_prev, "front_settle"]
    ttf_usd = ttf_prev * fx[d_prev] / MMBTU_PER_MWH
    jkm_prev = ttf_prev * 0.35 + 1.0  # helpers.cross_curve transform, same delivery month
    assert f["cross_commodity__jkm_minus_ttf_usd_mmbtu"].iloc[0] == pytest.approx(
        jkm_prev - ttf_usd)
    hh_prev = ttf_prev * 0.1 + 2.0
    assert f["cross_commodity__ttf_minus_hh_cost_usd_mmbtu"].iloc[0] == pytest.approx(
        ttf_usd - (1.15 * hh_prev + 3.0))
    # NBP publishes same day but its GBP FX next day -> also session d-1
    nbp_eur = ttf_prev * 7.0 / 100.0 * MMBTU_PER_MWH * 10.0 / gbp[d_prev]
    assert f["cross_commodity__nbp_minus_ttf_eur_mwh"].iloc[0] == pytest.approx(
        nbp_eur - ttf_prev)
    # THE is EUR and published same day -> session d itself
    assert f["cross_commodity__the_minus_ttf_eur_mwh"].iloc[0] == pytest.approx(1.5)
    assert s.at[d_same, "front_settle"] != ttf_prev
    assert f["cross_commodity__peg_minus_ttf_eur_mwh"].isna().all()


def test_price_to_eur_mwh_units() -> None:
    assert prices.price_to_eur_mwh(10.0, "EUR/MWh") == 10.0
    usd = prices.price_to_eur_mwh(10.0, "USD/MMBtu", 1.25)
    assert usd == pytest.approx(10.0 * MMBTU_PER_MWH / 1.25)
    with pytest.raises(Exception, match="unsupported"):
        prices.price_to_eur_mwh(1.0, "USD/bbl")


def test_m1_m2_dev_uses_only_prior_gas_years() -> None:
    dates = pd.bdate_range("2021-10-01", "2023-09-29")
    front = pd.DatetimeIndex((dates + pd.offsets.MonthBegin(1)).normalize())
    rng = np.random.default_rng(1)
    v = pd.Series(rng.normal(0, 1, len(dates)), index=dates)
    dev = prices.prior_gas_year_deviation(v, front)
    gy = np.where(dates.month >= 10, dates.year, dates.year - 1)
    assert dev[gy == 2021].isna().all()  # no prior gas year yet
    t = pd.Timestamp("2023-02-15")
    same = (front.month == front[dates.get_loc(t)].month) & (gy == 2021)
    assert dev[t] == pytest.approx(v[t] - v[same].mean())
    # altering the current and later gas years never changes the normal used
    v2 = v.copy()
    v2[gy == 2022] += 100.0
    dev2 = prices.prior_gas_year_deviation(v2, front)
    np.testing.assert_allclose((v2 - dev2)[gy == 2022], (v - dev)[gy == 2022])


def test_curve_has_m1_m2_dev(b: DataBundle) -> None:
    o = settle_origins("2022-11-01", "2023-03-31")
    f = prices.build_curve_features(b, o, _params("curve"))
    assert "curve__m1_m2_dev" in f.columns and f["curve__m1_m2_dev"].notna().any()
