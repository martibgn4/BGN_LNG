"""Target builder tests (T1-T9). SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.contracts.futures import FuturesContract, hours_weights
from ttf_forecaster.data.base import (
    SYNTHETIC_SOURCE,
    TARGET_SCHEMA,
    NonPositivePriceError,
    validate_frame,
)
from ttf_forecaster.targets.build import build_targets
from ttf_forecaster.targets.common import CurveContext, publication_time
from ttf_forecaster.targets.futures_targets import (
    build_delivery_excess_futures,
    build_m1_logret,
    build_realised_vol,
    build_regime_change,
    build_spread_change,
    build_strip_logret,
    build_tail,
)
from ttf_forecaster.targets.spot_targets import build_da_level, build_da_month_avg
from ttf_forecaster.targets.strips import strip_price


@pytest.fixture(scope="module")
def ctx(syn_curve: pd.DataFrame) -> CurveContext:
    return CurveContext.from_curve(syn_curve)


def _close(day: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} 19:00", tz="Europe/Amsterdam").tz_convert("UTC")


def _da_frame(days: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    """SYNTHETIC day-ahead frame: event_time = gas day 06:00 local; published 19:00 the day before."""
    n = len(days)
    ev = (days + pd.Timedelta(hours=6)).tz_localize("Europe/Amsterdam").tz_convert("UTC")
    pub = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=19)).tz_localize(
        "Europe/Amsterdam").tz_convert("UTC")
    return pd.DataFrame({
        "series_id": "price.ttf.da_eur_mwh", "event_time": ev, "value": values.astype(float),
        "geography": "NL", "publication_time": pub, "ingestion_time": pub,
        "revision": np.zeros(n, dtype="int64"), "unit": "EUR/MWh", "source": SYNTHETIC_SOURCE,
        "quality_flag": "synthetic", "entitlement": "synthetic", "pit_grade": "A",
        "publication_time_estimated": np.zeros(n, dtype=bool), "source_timezone": "Europe/Amsterdam",
    })


class TestFuturesTargets:
    def test_logret_equals_single_contract_difference(self, ctx: CurveContext) -> None:
        df = build_m1_logret(ctx, [5])
        validate_frame(df, TARGET_SCHEMA)
        for row in df.sample(40, random_state=1).itertuples():
            t = pd.Timestamp(row.origin_time).tz_convert("Europe/Amsterdam").tz_localize(
                None).normalize()
            exit_day = ctx.calendar.add_sessions(t, 5)
            p = ctx.panel[row.contract_id]
            assert row.value == pytest.approx(np.log(p[exit_day]) - np.log(p[t]))

    def test_label_known_time_is_exit_close(self, ctx: CurveContext) -> None:
        df = build_m1_logret(ctx, [20], origins=[pd.Timestamp("2022-03-01")])
        row = df.iloc[0]
        assert row.origin_time == _close("2022-03-01")
        assert row.label_known_time == _close("2022-03-29")   # 20 weekdays later
        assert row.quality_flag == "synthetic"
        # winter / summer DST: 19:00 CET = 18:00 UTC, 19:00 CEST = 17:00 UTC
        assert _close("2022-03-29").hour == 17 and _close("2022-03-01").hour == 18

    def test_rows_without_observable_exit_are_dropped(self, ctx: CurveContext) -> None:
        df = build_m1_logret(ctx, [20])
        last_close = ctx.pub_time(ctx.lookup.last_day)
        assert (df["label_known_time"] <= last_close).all()

    def test_missing_settlement_gives_nan(self, syn_curve: pd.DataFrame) -> None:
        t, exit_day = pd.Timestamp("2022-05-02"), pd.Timestamp("2022-05-09")
        c0 = CurveContext.from_curve(syn_curve)
        held = build_m1_logret(c0, [5], origins=[t]).iloc[0].contract_id
        trade_local = syn_curve["trade_date"].dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)
        drop = (syn_curve["contract_id"] == held) & (trade_local == exit_day)
        c1 = CurveContext.from_curve(syn_curve.loc[~drop])
        row = build_m1_logret(c1, [5], origins=[t]).iloc[0]
        assert np.isnan(row.value) and row.quality_flag == "missing_input"
        rv = build_realised_vol(c1, [5], origins=[t]).iloc[0]
        assert np.isnan(rv.value)

    def test_non_positive_price_raises(self, syn_curve: pd.DataFrame) -> None:
        bad = syn_curve.copy()
        bad.loc[bad["contract_id"] == "TTF.M.2022-06", "value"] = -1.0
        c = CurveContext.from_curve(bad)
        with pytest.raises(NonPositivePriceError):
            build_m1_logret(c, [1], origins=[pd.Timestamp("2022-05-02")])

    def test_realised_vol(self, ctx: CurveContext) -> None:
        t = pd.Timestamp("2022-09-01")
        row = build_realised_vol(ctx, [5], origins=[t]).iloc[0]
        days = [ctx.calendar.add_sessions(t, j) for j in range(6)]
        r = np.diff(np.log(ctx.panel.loc[days, row.contract_id].to_numpy()))
        assert row.value == pytest.approx(np.sqrt(252 / 5 * np.sum(r**2)))

    def test_delivery_excess_known_at_expiry(self, ctx: CurveContext) -> None:
        df = build_delivery_excess_futures(ctx, [2], origins=[pd.Timestamp("2022-03-15")])
        fut = df.loc[df.target_id == "delivery_excess_fut"].iloc[0]
        assert fut.contract_id == "TTF.M.2022-05"
        ltd = ctx.chain.last_trade[ctx.chain.index_of["TTF.M.2022-05"]]
        assert fut.label_known_time == ctx.pub_time(ltd)
        p = ctx.panel["TTF.M.2022-05"]
        assert fut.value == pytest.approx(p[ltd] - p[pd.Timestamp("2022-03-15")])

    def test_strip_uses_hours_weights(self, ctx: CurveContext) -> None:
        q = FuturesContract.quarter("TTF", 2023, 1)
        day = pd.Timestamp("2022-06-01")
        months = q.constituent_months()
        expected = np.dot(hours_weights(months), [ctx.panel.at[day, m.contract_id] for m in months])
        assert strip_price(ctx.lookup, q, day) == pytest.approx(expected)
        df = build_strip_logret(ctx, [20], "Q", origins=[day])
        assert df.iloc[0].construction == "strip"

    def test_strip_falls_back_to_listed_then_nan(self, syn_curve: pd.DataFrame) -> None:
        t, e = pd.Timestamp("2022-06-01"), pd.Timestamp("2022-06-29")
        trade_local = syn_curve["trade_date"].dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)
        drop = (syn_curve["contract_id"] == "TTF.M.2022-08") & (trade_local == t)
        holed = syn_curve.loc[~drop]
        row = build_strip_logret(CurveContext.from_curve(holed), [20], "Q", origins=[t]).iloc[0]
        assert row.contract_id == "TTF.Q.2022-Q3"
        assert np.isnan(row.value) and row.quality_flag == "missing_input"
        # a listed Q3-22 settlement on both dates is used instead, flagged construction=listed
        listed = holed.loc[(holed["contract_id"] == "TTF.M.2022-07")
                           & trade_local.loc[~drop].isin([t, e])].copy()
        listed["contract_id"] = "TTF.Q.2022-Q3"
        listed["tenor_type"] = "Q"
        listed["value"] = [40.0, 44.0]
        c = CurveContext.from_curve(pd.concat([holed, listed], ignore_index=True))
        row = build_strip_logret(c, [20], "Q", origins=[t]).iloc[0]
        assert row.construction == "listed"
        assert row.value == pytest.approx(np.log(44.0 / 40.0))

    def test_spread_change(self, ctx: CurveContext) -> None:
        t = pd.Timestamp("2022-09-01")
        row = build_spread_change(ctx, [5], "m1m2", origins=[t]).iloc[0]
        c1, c2 = row.contract_id.split("|")
        e = ctx.calendar.add_sessions(t, 5)
        p = ctx.panel
        expected = (p.at[e, c1] - p.at[e, c2]) - (p.at[t, c1] - p.at[t, c2])
        assert row.value == pytest.approx(expected)

    def test_tail_uses_sigma_known_at_t(self, ctx: CurveContext) -> None:
        df = build_tail(ctx, [1], k=1.5)
        assert set(df["value"].dropna().unique()) <= {0.0, 1.0}
        early = df.iloc[:5]
        assert early["value"].isna().all()   # fewer than min_periods returns before t

    def test_regime_change(self, ctx: CurveContext) -> None:
        days = ctx.default_origins()
        regimes = pd.Series(np.where(days < pd.Timestamp("2023-01-01"), "normal", "tight"),
                            index=days)
        df = build_regime_change(ctx, regimes, [20])
        t_before = df.loc[df.origin_time == _close("2022-12-15")].iloc[0]
        assert t_before.value == 1.0
        assert df.loc[df.origin_time == _close("2022-06-01")].iloc[0].value == 0.0

    def test_build_all_targets(self, ctx: CurveContext) -> None:
        df = build_targets(ctx, origins=ctx.default_origins()[:60])
        validate_frame(df, TARGET_SCHEMA)
        assert {"m1_logret", "rv", "tail", "q_logret", "spread_m1m2_chg"} <= set(df.target_id)
        assert df.attrs["skipped"] == ["T1", "T4", "T5b", "T9c"]
        assert (df["label_known_time"] > df["origin_time"]).all()


class TestSpotTargets:
    def test_da_level_label_is_next_assessment(self, ctx: CurveContext) -> None:
        days = pd.date_range("2022-01-01", "2023-12-31", freq="D")
        da = _da_frame(days, np.arange(len(days), dtype=float) + 10.0)
        t = pd.Timestamp("2022-03-03")   # Thursday
        row = build_da_level(ctx, da, origins=[t]).iloc[0]
        # assessed Fri 2022-03-04 for the next working gas day Mon 2022-03-07
        gas_day = pd.Timestamp("2022-03-07")
        assert row.value == pytest.approx(10.0 + days.get_loc(gas_day))
        assert row.label_known_time == _close("2022-03-06")   # published 19:00 the day before
        assert row.label_known_time > row.origin_time

    def test_da_month_avg_and_missing_day(self, ctx: CurveContext) -> None:
        days = pd.date_range("2022-01-01", "2023-12-31", freq="D")
        vals = np.linspace(20, 40, len(days))
        da = _da_frame(days, vals)
        t = pd.Timestamp("2022-03-15")
        row = build_da_month_avg(ctx, da, [1], origins=[t]).iloc[0]
        april = (days >= "2022-04-01") & (days <= "2022-04-30")
        assert row.value == pytest.approx(vals[april].mean())
        expected_known = pd.Timestamp("2022-05-01 06:00", tz="Europe/Amsterdam").tz_convert("UTC")
        assert row.label_known_time == expected_known
        holed = da.drop(index=int(np.flatnonzero(days == pd.Timestamp("2022-04-10"))[0]))
        row2 = build_da_month_avg(ctx, holed, [1], origins=[t]).iloc[0]
        assert np.isnan(row2.value) and row2.quality_flag == "missing_input"

    def test_build_all_with_da_and_regimes(self, ctx: CurveContext) -> None:
        days = pd.date_range("2022-01-01", "2023-12-31", freq="D")
        da = _da_frame(days, np.full(len(days), 30.0))
        origins = ctx.default_origins()[100:130]
        regimes = pd.Series("normal", index=ctx.default_origins())
        df = build_targets(ctx, origins=origins, da_observations=da, regimes=regimes)
        assert {"da_level", "da_month_avg", "delivery_excess_phys", "regime_change"} <= set(
            df.target_id)
        assert df.attrs["skipped"] == []


def test_publication_time_helper() -> None:
    from ttf_forecaster.data.base import PublicationLag

    lag = PublicationLag(days=0, time="19:00")
    assert publication_time(pd.Timestamp("2026-10-26"), lag) == pd.Timestamp(
        "2026-10-26 18:00", tz="UTC")
