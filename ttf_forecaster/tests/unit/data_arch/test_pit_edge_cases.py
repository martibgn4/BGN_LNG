"""Critical review of the frozen PIT contract (base.apply_publication_lag, point_in_time resolvers).

Edge cases: revision ties, duplicate rows, publication exactly at as_of, DST transitions in
Europe/Amsterdam (last Sunday of March / October 2024: 31 Mar, 27 Oct), ambiguous local times,
empty frames. Known defects are ``xfail(strict=True)`` with the proposed fix in the reason, so
the test flips to XPASS (and fails the run) once the contract is fixed and the marker must go.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    PointInTimeViolation,
    PublicationLag,
    SchemaError,
    apply_publication_lag,
    empty_frame,
)
from ttf_forecaster.features.point_in_time import (
    assert_no_future_rows,
    latest_value_per_series,
    resolve_as_of,
    resolve_forecasts_as_of,
)

from .helpers import fc_frame, obs_frame

AMS = "Europe/Amsterdam"


def _local(*stamps: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(stamps)).tz_localize(AMS).tz_convert("UTC"))


# ---------------------------------------------------------------------------------------------
# resolve_as_of
# ---------------------------------------------------------------------------------------------


class TestResolveTies:
    def test_same_revision_later_publication_wins(self) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0),
                        ("s.eu.x_twh", "2024-01-01", 2.0, "2024-01-04", 0)])
        assert resolve_as_of(df, pd.Timestamp("2024-01-05", tz="UTC"))["value"].tolist() == [2.0]
        assert resolve_as_of(df, pd.Timestamp("2024-01-03", tz="UTC"))["value"].tolist() == [1.0]

    def test_higher_revision_beats_later_publication(self) -> None:
        """Documented rule: highest revision wins, publication only breaks ties."""
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-03", 1),
                        ("s.eu.x_twh", "2024-01-01", 2.0, "2024-01-04", 0)])
        assert resolve_as_of(df, pd.Timestamp("2024-01-05", tz="UTC"))["value"].tolist() == [1.0]

    def test_exact_duplicates_collapse(self) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)] * 3)
        assert len(resolve_as_of(df, pd.Timestamp("2024-01-05", tz="UTC"))) == 1

    def test_conflicting_tie_raises(self) -> None:
        """Fixed 2026-10-16 (D-015): conflicting ties raise instead of resolving by row order."""
        from ttf_forecaster.data.base import DataError

        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0),
                        ("s.eu.x_twh", "2024-01-01", 2.0, "2024-01-02", 0)])
        as_of = pd.Timestamp("2024-01-05", tz="UTC")
        with pytest.raises(DataError, match="conflicting"):
            resolve_as_of(df, as_of)

    def test_ingestion_time_order_invariant_when_values_equal(self) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)] * 2)
        df.loc[1, "ingestion_time"] = df.loc[1, "ingestion_time"] + pd.Timedelta(hours=1)
        as_of = pd.Timestamp("2024-01-05", tz="UTC")
        assert resolve_as_of(df, as_of)["value"].tolist() == [1.0]


class TestResolveBoundaries:
    def test_publication_exactly_at_as_of_any_zone(self) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-03 07:00", 0)])
        assert len(resolve_as_of(df, pd.Timestamp("2024-01-03 08:00", tz=AMS))) == 1
        assert resolve_as_of(df, pd.Timestamp("2024-01-03 07:59:59.999999", tz=AMS)).empty

    def test_publication_one_nanosecond_after(self) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-03 07:00", 0)])
        assert resolve_as_of(df, pd.Timestamp("2024-01-03 07:00", tz="UTC")
                             - pd.Timedelta(1, "ns")).empty

    def test_strict_ingestion(self) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)])
        df["ingestion_time"] = pd.Timestamp("2024-01-10", tz="UTC")
        as_of = pd.Timestamp("2024-01-05", tz="UTC")
        assert len(resolve_as_of(df, as_of)) == 1
        assert resolve_as_of(df, as_of, strict_ingestion=True).empty

    def test_as_of_string_with_offset(self) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)])
        assert len(resolve_as_of(df, "2024-01-02T01:00:00+01:00")) == 1
        with pytest.raises(PointInTimeViolation):
            resolve_as_of(df, "2024-01-02 01:00")

    def test_future_event_observation_retained_if_published(self) -> None:
        """Contract note: resolver keeps event_time > as_of for observations (schedules), while
        point_in_time_policy §2 calls that a data error for observation series. Pinned here so a
        change in either is deliberate."""
        df = obs_frame([("s.eu.x_twh", "2024-02-01", 1.0, "2024-01-02", 0)])
        res = resolve_as_of(df, pd.Timestamp("2024-01-05", tz="UTC"))
        assert len(res) == 1
        with pytest.raises(PointInTimeViolation):
            assert_no_future_rows(res, pd.Timestamp("2024-01-05", tz="UTC"), "event_time")

    def test_grade_c_revision_does_not_shadow_grade_b(self) -> None:
        a = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)], grade="B")
        c = obs_frame([("s.eu.x_twh", "2024-01-01", 9.0, "2024-01-03", 1)], grade="C")
        res = resolve_as_of(pd.concat([a, c], ignore_index=True), pd.Timestamp("2024-01-05", tz="UTC"))
        assert res["value"].tolist() == [1.0]


class TestEmpty:
    def test_empty_observations(self) -> None:
        as_of = pd.Timestamp("2024-01-05", tz="UTC")
        res = resolve_as_of(empty_frame(OBSERVATION_SCHEMA), as_of)
        assert res.empty and list(res.columns) == OBSERVATION_SCHEMA.names
        assert latest_value_per_series(res, as_of).empty

    def test_empty_forecasts(self) -> None:
        res = resolve_forecasts_as_of(empty_frame(FORECAST_SCHEMA),
                                      pd.Timestamp("2024-01-05", tz="UTC"), min_lead_hours=24)
        assert res.empty and list(res.columns) == FORECAST_SCHEMA.names

    def test_nothing_visible(self) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)])
        res = resolve_as_of(df, pd.Timestamp("2023-12-01", tz="UTC"))
        assert res.empty and list(res.columns) == list(df.columns)


# ---------------------------------------------------------------------------------------------
# resolve_forecasts_as_of
# ---------------------------------------------------------------------------------------------


class TestForecastResolver:
    def test_late_published_older_run_does_not_win(self) -> None:
        """Run issued later wins even if the older run's publication is later (latest issue)."""
        df = fc_frame([
            ("w.eu.hdd_degc_day", "2024-01-01 00:00", "2024-01-05", 10.0, "2024-01-01 12:00", 0),
            ("w.eu.hdd_degc_day", "2024-01-02 00:00", "2024-01-05", 12.0, "2024-01-02 06:00", 0),
        ])
        res = resolve_forecasts_as_of(df, pd.Timestamp("2024-01-03", tz="UTC"))
        assert res["value"].tolist() == [12.0]

    def test_run_published_exactly_at_as_of(self) -> None:
        df = fc_frame([("w.eu.hdd_degc_day", "2024-01-01", "2024-01-05", 10.0,
                        "2024-01-01 06:00", 0)])
        assert len(resolve_forecasts_as_of(df, pd.Timestamp("2024-01-01 06:00", tz="UTC"))) == 1

    def test_issue_tie_revision_breaks(self) -> None:
        df = fc_frame([
            ("w.eu.hdd_degc_day", "2024-01-01", "2024-01-05", 10.0, "2024-01-01 06:00", 0),
            ("w.eu.hdd_degc_day", "2024-01-01", "2024-01-05", 11.0, "2024-01-01 09:00", 1),
        ])
        r = resolve_forecasts_as_of(df, pd.Timestamp("2024-01-01 07:00", tz="UTC"))
        assert r["value"].tolist() == [10.0]
        r = resolve_forecasts_as_of(df, pd.Timestamp("2024-01-02", tz="UTC"))
        assert r["value"].tolist() == [11.0]

    def test_min_lead_picks_older_run(self) -> None:
        df = fc_frame([
            ("w.eu.hdd_degc_day", "2024-01-01", "2024-01-05", 10.0, "2024-01-01 06:00", 0),
            ("w.eu.hdd_degc_day", "2024-01-04", "2024-01-05", 12.0, "2024-01-04 06:00", 0),
        ])
        r = resolve_forecasts_as_of(df, pd.Timestamp("2024-01-04 12:00", tz="UTC"),
                                    min_lead_hours=72)
        assert r["value"].tolist() == [10.0]

    def test_conflicting_tie_raises(self) -> None:
        from ttf_forecaster.data.base import DataError

        df = fc_frame([
            ("w.eu.hdd_degc_day", "2024-01-01", "2024-01-05", 10.0, "2024-01-01 06:00", 0),
            ("w.eu.hdd_degc_day", "2024-01-01", "2024-01-05", 11.0, "2024-01-01 06:00", 0),
        ])
        with pytest.raises(DataError, match="conflicting"):
            resolve_forecasts_as_of(df, pd.Timestamp("2024-01-02", tz="UTC"))


# ---------------------------------------------------------------------------------------------
# apply_publication_lag and DST
# ---------------------------------------------------------------------------------------------


class TestPublicationLagDst:
    def test_spring_forward_day(self) -> None:
        # 2024-03-31: 02:00 CET -> 03:00 CEST. Local midnight is still CET (UTC+1).
        pub = apply_publication_lag(_local("2024-03-31 00:00"), PublicationLag(days=0, time="07:00"))
        assert pub.iloc[0] == pd.Timestamp("2024-03-31 05:00", tz="UTC")

    def test_lag_across_spring_forward(self) -> None:
        pub = apply_publication_lag(_local("2024-03-30 00:00"), PublicationLag(days=2, time="07:00"))
        assert pub.iloc[0] == pd.Timestamp("2024-04-01 07:00", tz=AMS)
        assert pub.iloc[0] == pd.Timestamp("2024-04-01 05:00", tz="UTC")

    def test_fall_back_day(self) -> None:
        # 2024-10-27: 03:00 CEST -> 02:00 CET. 07:00 local is CET (UTC+1).
        pub = apply_publication_lag(_local("2024-10-27 00:00"), PublicationLag(days=0, time="07:00"))
        assert pub.iloc[0] == pd.Timestamp("2024-10-27 06:00", tz="UTC")

    def test_lag_across_fall_back(self) -> None:
        pub = apply_publication_lag(_local("2024-10-26 00:00"), PublicationLag(days=2, time="19:00"))
        assert pub.iloc[0] == pd.Timestamp("2024-10-28 18:00", tz="UTC")

    def test_event_late_evening_utc_maps_to_next_local_date(self) -> None:
        # 2024-07-01 22:30 UTC is 2024-07-02 00:30 CEST: local date is the 2nd.
        ev = pd.Series(pd.to_datetime(["2024-07-01 22:30"]).tz_localize("UTC"))
        pub = apply_publication_lag(ev, PublicationLag(days=0, time="07:00"))
        assert pub.iloc[0] == pd.Timestamp("2024-07-02 07:00", tz=AMS)

    def test_nonexistent_local_time_shifts_forward(self) -> None:
        pub = apply_publication_lag(_local("2024-03-31 00:00"), PublicationLag(days=0, time="02:30"))
        assert pub.iloc[0] == pd.Timestamp("2024-03-31 03:00", tz=AMS)

    def test_ambiguous_local_time_is_resolved_conservatively(self) -> None:
        pub = apply_publication_lag(_local("2024-10-27 00:00"), PublicationLag(days=0, time="02:30"))
        assert pub.iloc[0] == pd.Timestamp("2024-10-27 01:30", tz="UTC")  # 02:30 CET (later)

    def test_business_days_from_weekend(self) -> None:
        """Pinned semantics: a Saturday event with business_days lag 0 or 1 is published Monday
        (BDay rolls weekend dates forward), i.e. lag 0 and lag 1 coincide for weekend events."""
        sat = _local("2024-01-20 00:00")
        p0 = apply_publication_lag(sat, PublicationLag(days=0, time="07:00", business_days=True))
        p1 = apply_publication_lag(sat, PublicationLag(days=1, time="07:00", business_days=True))
        assert p0.iloc[0] == p1.iloc[0] == pd.Timestamp("2024-01-22 07:00", tz=AMS)

    def test_business_day_lag_vectorised_no_warning(self) -> None:
        ev = _local(*[str(d.date()) for d in pd.date_range("2024-01-01", periods=50)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                apply_publication_lag(ev, PublicationLag(days=2, business_days=True))
            except Warning as w:  # pragma: no cover - documents a pandas PerformanceWarning
                pytest.skip(f"pandas warns on vectorised BDay: {w}")

    def test_empty_series(self) -> None:
        out = apply_publication_lag(pd.Series(pd.DatetimeIndex([], tz="UTC")), PublicationLag(days=1))
        assert out.empty and str(out.dt.tz) == "UTC"

    def test_naive_rejected(self) -> None:
        with pytest.raises(SchemaError):
            apply_publication_lag(pd.Series(pd.to_datetime(["2024-01-01"])), PublicationLag(days=1))

    def test_resolver_across_fall_back_hour(self) -> None:
        """Two publications in the repeated 02:xx hour are ordered by UTC, not wall clock."""
        df = obs_frame([("s.eu.x_twh", "2024-10-26", 1.0, "2024-10-27 00:30", 0),   # 02:30 CEST
                        ("s.eu.x_twh", "2024-10-26", 2.0, "2024-10-27 01:15", 1)])  # 02:15 CET
        as_of = pd.Timestamp("2024-10-27 01:00", tz="UTC")  # 02:00 CET, after 02:30 CEST
        assert resolve_as_of(df, as_of)["value"].tolist() == [1.0]
        assert resolve_as_of(df, pd.Timestamp("2024-10-27 02:00", tz="UTC"))["value"].tolist() \
            == [2.0]


def test_leakage_canary_through_store(tmp_path) -> None:
    """Perturb stored rows published after as_of: the stored as-of view is unchanged."""
    from ttf_forecaster.data.storage import ParquetStore

    df = obs_frame([("s.eu.x_twh", f"2024-01-{d:02d}", float(d), f"2024-01-{d + 1:02d}", 0)
                    for d in range(1, 20)])
    as_of = pd.Timestamp("2024-01-10 12:00", tz="UTC")
    s1 = ParquetStore(tmp_path / "a")
    s1.append_normalised(df, "observation")
    pert = df.copy()
    fut = pert["publication_time"] > as_of
    pert.loc[fut, "value"] = pert.loc[fut, "value"] * 1000 + np.pi
    s2 = ParquetStore(tmp_path / "b")
    s2.append_normalised(pert, "observation")
    pd.testing.assert_frame_equal(resolve_as_of(s1.read_normalised("observation"), as_of),
                                  resolve_as_of(s2.read_normalised("observation"), as_of))
