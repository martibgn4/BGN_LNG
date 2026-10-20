"""Tests for the frozen contract: schemas, as-of resolver, units, synthetic generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.contracts import units
from ttf_forecaster.data.base import (
    CURVE_SCHEMA,
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    PitGrade,
    PointInTimeViolation,
    SchemaError,
    empty_frame,
    validate_frame,
    worst_grade,
)
from ttf_forecaster.features.point_in_time import (
    latest_value_per_series,
    resolve_as_of,
    resolve_forecasts_as_of,
)


def _obs(rows: list[tuple[str, str, float, str, int]], grade: str = "A") -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["series_id", "event_time", "value", "publication_time",
                                     "revision"])
    n = len(df)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["publication_time"] = pd.to_datetime(df["publication_time"], utc=True)
    df["ingestion_time"] = df["publication_time"]
    df["value"] = df["value"].astype(float)
    df["revision"] = df["revision"].astype("int64")
    for col, val in {"geography": "EU", "unit": "TWh", "source": "SYNTHETIC",
                     "quality_flag": "synthetic", "entitlement": "synthetic",
                     "source_timezone": "UTC"}.items():
        df[col] = [val] * n
    df["pit_grade"] = [grade] * n
    df["publication_time_estimated"] = np.zeros(n, dtype=bool)
    return df


class TestSchema:
    def test_empty_frame_validates(self) -> None:
        for schema in (OBSERVATION_SCHEMA, FORECAST_SCHEMA, CURVE_SCHEMA):
            validate_frame(empty_frame(schema), schema)

    def test_naive_timestamp_rejected(self) -> None:
        df = _obs([("s", "2024-01-01", 1.0, "2024-01-02", 0)])
        df["event_time"] = df["event_time"].dt.tz_localize(None)
        with pytest.raises(SchemaError, match="tz-aware"):
            validate_frame(df, OBSERVATION_SCHEMA)

    def test_non_utc_rejected(self) -> None:
        df = _obs([("s", "2024-01-01", 1.0, "2024-01-02", 0)])
        df["event_time"] = df["event_time"].dt.tz_convert("Europe/Amsterdam")
        with pytest.raises(SchemaError, match="expected UTC"):
            validate_frame(df, OBSERVATION_SCHEMA)

    def test_missing_column_listed(self) -> None:
        df = _obs([("s", "2024-01-01", 1.0, "2024-01-02", 0)]).drop(columns=["pit_grade"])
        with pytest.raises(SchemaError, match="pit_grade"):
            validate_frame(df, OBSERVATION_SCHEMA)

    def test_worst_grade(self) -> None:
        assert worst_grade(["A", "B"]) == PitGrade.B
        assert worst_grade(["A", "C", "B"]) == PitGrade.C


class TestResolveAsOf:
    def test_revision_not_visible_before_publication(self) -> None:
        df = _obs([
            ("s", "2024-01-01", 100.0, "2024-01-02 08:00", 0),
            ("s", "2024-01-01", 105.0, "2024-01-05 08:00", 1),
        ])
        early = resolve_as_of(df, pd.Timestamp("2024-01-03", tz="UTC"))
        late = resolve_as_of(df, pd.Timestamp("2024-01-06", tz="UTC"))
        assert early["value"].tolist() == [100.0]
        assert late["value"].tolist() == [105.0]

    def test_unpublished_row_invisible(self) -> None:
        df = _obs([("s", "2024-01-01", 1.0, "2024-01-03", 0)])
        assert resolve_as_of(df, pd.Timestamp("2024-01-02", tz="UTC")).empty

    def test_publication_exactly_at_as_of_is_visible(self) -> None:
        df = _obs([("s", "2024-01-01", 1.0, "2024-01-03 07:00", 0)])
        assert len(resolve_as_of(df, pd.Timestamp("2024-01-03 07:00", tz="UTC"))) == 1

    def test_grade_c_excluded_by_default(self) -> None:
        df = _obs([("s", "2024-01-01", 1.0, "2024-01-02", 0)], grade="C")
        assert resolve_as_of(df, pd.Timestamp("2024-02-01", tz="UTC")).empty
        assert len(resolve_as_of(df, pd.Timestamp("2024-02-01", tz="UTC"), allowed_grades=None)) == 1

    def test_naive_as_of_rejected(self) -> None:
        df = _obs([("s", "2024-01-01", 1.0, "2024-01-02", 0)])
        with pytest.raises(PointInTimeViolation):
            resolve_as_of(df, pd.Timestamp("2024-02-01"))

    def test_staleness_masks_old_values(self) -> None:
        df = _obs([("s", "2024-01-01", 1.0, "2024-01-02", 0)])
        res = resolve_as_of(df, pd.Timestamp("2024-02-01", tz="UTC"))
        val = latest_value_per_series(res, pd.Timestamp("2024-02-01", tz="UTC"),
                                      max_staleness=pd.Timedelta(days=5))
        assert np.isnan(val["s"])

    def test_leakage_canary(self, syn_storage: pd.DataFrame) -> None:
        """Perturbing anything published after as_of must not change the resolved view."""
        as_of = pd.Timestamp("2023-03-15 12:00", tz="UTC")
        before = resolve_as_of(syn_storage, as_of)
        perturbed = syn_storage.copy()
        future = perturbed["publication_time"] > as_of
        perturbed.loc[future, "value"] = perturbed.loc[future, "value"] * 1000 + 1
        after = resolve_as_of(perturbed, as_of)
        pd.testing.assert_frame_equal(before, after)


class TestForecastResolver:
    def test_latest_visible_run_wins(self, syn_weather: pd.DataFrame) -> None:
        as_of = pd.Timestamp("2023-01-10 12:00", tz="UTC")
        res = resolve_forecasts_as_of(syn_weather, as_of)
        assert (res["publication_time"] <= as_of).all()
        # For a valid day 3 days ahead, the run used must be the most recent visible one.
        target = pd.Timestamp("2023-01-13", tz="UTC")
        row = res.loc[res["valid_time"] == target]
        assert row["issue_time"].iloc[0] == pd.Timestamp("2023-01-10", tz="UTC")

    def test_min_lead_filters(self, syn_weather: pd.DataFrame) -> None:
        as_of = pd.Timestamp("2023-01-10 12:00", tz="UTC")
        res = resolve_forecasts_as_of(syn_weather, as_of, min_lead_hours=72)
        assert (res["lead_hours"] >= 72).all()


class TestUnits:
    def test_mmbtu_per_mwh(self) -> None:
        assert units.MMBTU_PER_MWH == pytest.approx(3.412141633, rel=1e-9)

    def test_round_trip_price(self) -> None:
        usd = units.eur_mwh_to_usd_mmbtu(35.0, 1.08)
        assert units.usd_mmbtu_to_eur_mwh(usd, 1.08) == pytest.approx(35.0)

    def test_rate_conversion(self) -> None:
        assert units.convert_energy_rate(1.0, "TWh/d", "GWh/d") == pytest.approx(1000.0)

    def test_unknown_unit(self) -> None:
        with pytest.raises(units.UnitError):
            units.convert_energy(1.0, "bcm", "TWh")

    def test_gas_quality_round_trip(self) -> None:
        q = units.GasQuality(gcv_kwh_per_m3=11.0, reference="25/0 C")
        assert q.gwh_to_mcm(q.mcm_to_gwh(3.0)) == pytest.approx(3.0)


class TestSynthetic:
    def test_frames_validate_and_are_labelled(self, syn_storage, syn_curve, syn_weather) -> None:
        validate_frame(syn_storage, OBSERVATION_SCHEMA)
        validate_frame(syn_curve, CURVE_SCHEMA)
        validate_frame(syn_weather, FORECAST_SCHEMA)
        for df in (syn_storage, syn_curve, syn_weather):
            assert (df["source"] == "SYNTHETIC").all()

    def test_contract_stops_before_delivery(self, syn_curve: pd.DataFrame) -> None:
        last = syn_curve.groupby("contract_id")["trade_date"].max()
        start = syn_curve.groupby("contract_id")["delivery_start"].first()
        assert (last < start).all()


class TestPublicationLag:
    def test_calendar_lag_local_time(self) -> None:
        from ttf_forecaster.data.base import PublicationLag, apply_publication_lag

        ev = pd.Series(pd.to_datetime(["2024-01-15 00:00"]).tz_localize("Europe/Amsterdam")
                       .tz_convert("UTC"))
        pub = apply_publication_lag(ev, PublicationLag(days=2, time="07:00"))
        assert pub.iloc[0] == pd.Timestamp("2024-01-17 07:00", tz="Europe/Amsterdam")

    def test_business_day_lag_skips_weekend(self) -> None:
        from ttf_forecaster.data.base import PublicationLag, apply_publication_lag

        friday = pd.Series(pd.to_datetime(["2024-01-19"]).tz_localize("Europe/Amsterdam")
                           .tz_convert("UTC"))
        pub = apply_publication_lag(friday, PublicationLag(days=1, time="19:00",
                                                           business_days=True))
        assert pub.iloc[0] == pd.Timestamp("2024-01-22 19:00", tz="Europe/Amsterdam")

    def test_dst_summer_offset(self) -> None:
        from ttf_forecaster.data.base import PublicationLag, apply_publication_lag

        ev = pd.Series(pd.to_datetime(["2024-07-01"]).tz_localize("Europe/Amsterdam")
                       .tz_convert("UTC"))
        pub = apply_publication_lag(ev, PublicationLag(days=0, time="19:00"))
        assert pub.iloc[0] == pd.Timestamp("2024-07-01 17:00", tz="UTC")


class TestCurveResolver:
    def test_settlement_invisible_before_publication(self, syn_curve: pd.DataFrame) -> None:
        from ttf_forecaster.features.point_in_time import resolve_curves_as_of

        before = pd.Timestamp("2023-03-15 18:59", tz="Europe/Amsterdam")
        after = pd.Timestamp("2023-03-15 19:00", tz="Europe/Amsterdam")
        day = pd.Timestamp("2023-03-15", tz="Europe/Amsterdam").tz_convert("UTC")
        assert not (resolve_curves_as_of(syn_curve, before)["trade_date"] == day).any()
        assert (resolve_curves_as_of(syn_curve, after)["trade_date"] == day).any()
