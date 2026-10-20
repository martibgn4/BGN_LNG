"""Reviewed events -> OBSERVATION rows -> text__ features, with PIT grades (SYNTHETIC only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import OBSERVATION_SCHEMA, DataBundle, PitGrade, validate_frame
from ttf_forecaster.text.review import ReviewLedger
from ttf_forecaster.text.to_features import (
    TextPitMode,
    TextProvenance,
    build_text_features,
    events_to_observations,
    gas_day_start,
    slug,
    text_pit_grade,
)

from .conftest import t

LIVE = TextProvenance(TextPitMode.LIVE, assembled_with_hindsight=False)
HIST = TextProvenance(TextPitMode.HISTORICAL, assembled_with_hindsight=False)
HINDSIGHT = TextProvenance(TextPitMode.HISTORICAL, assembled_with_hindsight=True)
COUNT = "text.nwe.outage_count"
MCMD = "text.synthetic_field_a.offline_mcmd"
PARAMS = {"series": [COUNT, MCMD], "offset_days": 1, "coverage_start": "2026-01-01T00:00Z"}


def _ledger(make_event) -> ReviewLedger:
    led = ReviewLedger()
    ev = led.add_extraction(make_event(0))
    led.approve(ev.event_id, "alice", t(10, 10))
    led.edit(ev.event_id, "bob", t(10, 12),
             {"estimated_volume_impact_mcmd": 12.5, "impact_quote": "reduced by 12.5 mcm/d"})
    led.add_extraction(make_event(1, region="SYNTHETIC-SOUTH"))  # stays pending: never visible
    return led


def _features(obs: pd.DataFrame, origins: list[pd.Timestamp], **params) -> pd.DataFrame:
    bundle = DataBundle.empty()
    bundle.observations = obs
    return build_text_features(bundle, pd.DatetimeIndex(origins), {**PARAMS, **params})


def test_grade_logic():
    assert text_pit_grade("live", False) == PitGrade.A
    assert text_pit_grade("historical", False) == PitGrade.B
    assert text_pit_grade("live", True) == PitGrade.C
    assert text_pit_grade("historical", True) == PitGrade.C


def test_slug_and_gas_day():
    assert slug(" North-West Europe ") == "north_west_europe"
    with pytest.raises(ValueError):
        slug("--")
    # 04:00 UTC = 05:00 CET is still the previous gas day (starts 06:00 local)
    assert gas_day_start(pd.Timestamp("2026-01-10 04:00Z")) == pd.Timestamp("2026-01-09 05:00Z")
    assert gas_day_start(pd.Timestamp("2026-07-10 04:30Z")) == pd.Timestamp("2026-07-10 04:00Z")


def test_live_observations_are_bitemporal(make_event):
    obs = events_to_observations(_ledger(make_event).all_versions(), LIVE)
    validate_frame(obs, OBSERVATION_SCHEMA)
    assert set(obs["series_id"]) == {COUNT, MCMD}
    assert (obs["pit_grade"] == "A").all() and (obs["source"] == "SYNTHETIC").all()
    assert not obs["publication_time_estimated"].any()
    mc = obs.loc[obs["series_id"] == MCMD].sort_values(["event_time", "revision"])
    first_day = mc.loc[mc["event_time"] == mc["event_time"].min()]
    # approval at 10:00 -> number unknown (NaN), edit at 12:00 -> 12.5 as revision 1
    assert np.isnan(first_day.iloc[0]["value"]) and first_day.iloc[0]["revision"] == 0
    assert first_day.iloc[0]["publication_time"] == pd.Timestamp(t(10, 10))
    assert first_day.iloc[1]["value"] == 12.5 and first_day.iloc[1]["revision"] == 1
    assert first_day.iloc[1]["publication_time"] == pd.Timestamp(t(10, 12))
    # count did not change with the edit -> no new revision
    assert (obs.loc[obs["series_id"] == COUNT, "revision"] == 0).all()
    assert len(obs.loc[obs["series_id"] == COUNT]) == 3  # gas days 10, 11, 12 Jan


def test_rejection_withdraws_event(make_event):
    led = _ledger(make_event)
    eid = led.passing()[0].event_id
    led.reject(eid, "carol", t(11, 0))
    obs = events_to_observations(led.all_versions(), LIVE)
    latest = obs.sort_values("revision").groupby(["series_id", "event_time"]).tail(1)
    assert (latest["value"] == 0.0).all()
    assert (latest["publication_time"] == pd.Timestamp(t(11, 0))).all()


def test_features_respect_review_time(make_event):
    obs = events_to_observations(_ledger(make_event).all_versions(), LIVE)
    origins = [pd.Timestamp(t(10, 9)), pd.Timestamp(t(10, 11)), pd.Timestamp(t(10, 13))]
    f = _features(obs, origins)
    c, m = "text__nwe_outage_count_d1", "text__synthetic_field_a_offline_mcmd_d1"
    assert list(f.columns) == [c, m]
    assert f[c].tolist() == [0.0, 1.0, 1.0]  # before approval: covered, no known event
    assert f[m].iloc[0] == 0.0 and np.isnan(f[m].iloc[1]) and f[m].iloc[2] == 12.5
    assert f.attrs["pit_grade"] == {c: "A", m: "A"}
    assert f.attrs["family"] == "text" and f.attrs["clean_backtest_eligible"] is False


def test_no_coverage_means_nan_not_zero(make_event):
    obs = events_to_observations(_ledger(make_event).all_versions(), LIVE)
    f = _features(obs, [pd.Timestamp(t(10, 9))], coverage_start=None)
    assert f.isna().all().all()


def test_historical_mode_uses_lag_and_grade_b(make_event):
    obs = events_to_observations(_ledger(make_event).all_versions(), HIST)
    assert (obs["pit_grade"] == "B").all() and obs["publication_time_estimated"].all()
    assert (obs["publication_time"] == pd.Timestamp(t(11, 8))).all()  # source pub + 1 day
    # only the final reviewed version is used: volume known from the start
    assert (obs.loc[obs["series_id"] == MCMD, "value"] == 12.5).all()
    f = _features(obs, [pd.Timestamp(t(10, 13)), pd.Timestamp(t(11, 9))])
    assert f["text__nwe_outage_count_d1"].tolist() == [0.0, 1.0]


def test_hindsight_set_is_grade_c_and_excluded(make_event):
    obs = events_to_observations(_ledger(make_event).all_versions(), HINDSIGHT)
    assert (obs["pit_grade"] == "C").all()
    f = _features(obs, [pd.Timestamp(t(11, 9))], coverage_start=None)
    assert f.isna().all().all()
    assert set(f.attrs["pit_grade"].values()) == {"C"}
    g = _features(obs, [pd.Timestamp(t(11, 9))], allowed_grades=None)  # labelled experiment only
    assert g["text__nwe_outage_count_d1"].iloc[0] == 1.0


def test_leakage_canary(make_event):
    obs = events_to_observations(_ledger(make_event).all_versions(), LIVE)
    origin = pd.Timestamp(t(10, 11))
    base = _features(obs, [origin])
    perturbed = obs.copy()
    future = perturbed["publication_time"] > origin
    assert future.any()
    perturbed.loc[future, "value"] = 999.0
    pd.testing.assert_frame_equal(base, _features(perturbed, [origin]))


def test_open_ended_event_is_flagged_estimated(make_event):
    led = ReviewLedger()
    ev = led.add_extraction(make_event(expected_end=None, timing_quote=None,
                                       source_entitlement="public", source_publisher="P"))
    led.approve(ev.event_id, "alice", t(10, 10))
    obs = events_to_observations(led.all_versions(), TextProvenance(TextPitMode.LIVE, False,
                                                                    open_ended_days=3))
    # 07:00 UTC + 3 days reaches into a 4th gas day (gas day 13 Jan starts 05:00 UTC)
    cnt = obs.loc[obs["series_id"] == COUNT]
    assert len(cnt) == 4 and (cnt["quality_flag"] == "estimated").all()
    assert (cnt["source"] == "text_events").all() and (cnt["entitlement"] == "public").all()


def test_non_text_series_rejected(make_event):
    obs = events_to_observations(_ledger(make_event).all_versions(), LIVE)
    with pytest.raises(ValueError, match="text"):
        _features(obs, [pd.Timestamp(t(10, 11))], series=["storage.eu.inventory_twh"])


def test_empty_ledger_gives_empty_valid_frame():
    obs = events_to_observations([], LIVE)
    validate_frame(obs, OBSERVATION_SCHEMA)
    assert obs.empty
