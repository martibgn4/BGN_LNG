"""Derived series with signed terms, a constant and forecast components (D-044).

Everything here is SYNTHETIC: the ids are the real French outage id and a stand-in, the values test the rules only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data import derived as dv
from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    CURVE_SCHEMA,
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    DataBundle,
    DataError,
    empty_frame,
    validate_frame,
)
from ttf_forecaster.settings import ConfigError, load_config

UNAV = "power.fr.outage_nuclear_unavail_mw"
OTHER = "power.fr.outage_stand_in_unavail_mw"
AVAIL_OBS = "power.fr.nuclear_avail_gw"
AVAIL_FC = "power.fr.nuclear_avail_fc_gw"
CONST = 63000.0
PROV = "test fixture, not a fleet figure"
CFG = {
    "derived": {
        AVAIL_OBS: {"terms": [{"series": UNAV, "coef": -1}], "constant": CONST, "constant_status": "assumed",
                    "constant_provenance": PROV, "scale": 0.001},
        AVAIL_FC: {"components_kind": "forecast", "terms": [{"series": UNAV, "coef": -1}], "constant": CONST,
                   "constant_status": "assumed", "constant_provenance": PROV, "scale": 0.001},
    },
    "series": {AVAIL_OBS: {"unit": "GW", "schema": "observation"}, AVAIL_FC: {"unit": "GW", "schema": "forecast"}},
}
SPECS = dv.load_derived_specs(CFG)


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def obs_row(sid: str, event: str, value: float, pub: str, revision: int = 0) -> dict[str, object]:
    return {"series_id": sid, "event_time": utc(event), "value": float(value), "geography": "FR",
            "publication_time": utc(pub), "ingestion_time": utc(pub) + pd.Timedelta(minutes=5),
            "revision": revision, "unit": "MW", "source": "bloomberg_bsrch", "quality_flag": "ok",
            "entitlement": "licensed_bloomberg", "pit_grade": "B", "publication_time_estimated": True,
            "source_timezone": "UTC"}


def fc_row(sid: str, issue: str, valid: str, value: float, pub: str, grade: str = "A") -> dict[str, object]:
    return {"series_id": sid, "issue_time": utc(issue), "valid_time": utc(valid), "lead_hours": 24.0,
            "model": "bsrch_snapshot", "member": "daily_plan", "value": float(value), "geography": "FR",
            "publication_time": utc(pub), "ingestion_time": utc(pub) + pd.Timedelta(minutes=5), "revision": 0,
            "unit": "MW", "source": "bloomberg_bsrch", "quality_flag": "ok", "entitlement": "licensed_bloomberg",
            "pit_grade": grade, "publication_time_estimated": False, "source_timezone": "UTC"}


def obs(rows: list[dict[str, object]]) -> pd.DataFrame:
    return validate_frame(pd.DataFrame(rows), OBSERVATION_SCHEMA) if rows else empty_frame(OBSERVATION_SCHEMA)


def fc(rows: list[dict[str, object]]) -> pd.DataFrame:
    return validate_frame(pd.DataFrame(rows), FORECAST_SCHEMA) if rows else empty_frame(FORECAST_SCHEMA)


# ---------------------------------------------------------------- subtraction and the constant (observations)


def test_observation_is_constant_minus_the_unavailable_mw() -> None:
    o = obs([obs_row(UNAV, "2026-10-19", 1000.0, "2026-10-20 12:00")])
    d = dv.derive_observations(o, {AVAIL_OBS: SPECS[AVAIL_OBS]})
    assert len(d) == 1
    assert d["value"].iloc[0] == pytest.approx((CONST - 1000.0) * 0.001)  # 62.0 GW


def test_constant_is_applied_once_per_complete_state() -> None:
    o = obs([obs_row(UNAV, "2026-10-19", 0.0, "2026-10-20 12:00"),
             obs_row(UNAV, "2026-10-20", 2000.0, "2026-10-21 12:00")])
    d = dv.derive_observations(o, {AVAIL_OBS: SPECS[AVAIL_OBS]}).sort_values("event_time")
    assert list(d["value"]) == pytest.approx([63.0, 61.0])


def test_a_missing_component_gives_nan_not_a_partial_value() -> None:
    two = {"derived": {"power.fr.test_gw": {"terms": [{"series": UNAV, "coef": -1},
                                                     {"series": OTHER, "coef": 1}],
                                            "constant": CONST, "constant_status": "assumed",
                                            "constant_provenance": PROV, "scale": 0.001}},
           "series": {"power.fr.test_gw": {"unit": "GW", "schema": "observation"}}}
    spec = dv.load_derived_specs(two)["power.fr.test_gw"]
    o = obs([obs_row(UNAV, "2026-10-19", 1000.0, "2026-10-20 12:00"),
             obs_row(OTHER, "2026-10-19", 5.0, "2026-10-20 12:00"),
             obs_row(UNAV, "2026-10-20", 1000.0, "2026-10-21 12:00")])  # OTHER missing for 2026-10-20
    d = dv.derive_observations(o, {spec.series_id: spec}).sort_values("event_time")
    assert d["value"].iloc[0] == pytest.approx((CONST - 1000.0 + 5.0) * 0.001)
    assert np.isnan(d["value"].iloc[1])
    assert d["quality_flag"].iloc[1] == "missing_input"


# ---------------------------------------------------------------- forecast components


def test_forecast_row_joins_components_on_the_same_issue_and_valid_day() -> None:
    f = fc([fc_row(UNAV, "2026-10-13", "2026-10-20", 1000.0, "2026-10-19 16:00"),
            fc_row(OTHER, "2026-10-13", "2026-10-20", 5.0, "2026-10-19 16:00")])
    out = dv.derive_forecasts(f, {AVAIL_FC: _spec_with_two()})
    assert len(out) == 1
    assert out["value"].iloc[0] == pytest.approx((CONST - 1000.0 + 5.0) * 0.001)
    assert out["issue_time"].iloc[0] == utc("2026-10-13") and out["valid_time"].iloc[0] == utc("2026-10-20")
    assert out["lead_hours"].iloc[0] == pytest.approx(7 * 24.0)


def _spec_with_two() -> dv.DerivedSpec:
    cfg = {"derived": {AVAIL_FC: {"components_kind": "forecast",
                                  "terms": [{"series": UNAV, "coef": -1}, {"series": OTHER, "coef": 1}],
                                  "constant": CONST, "constant_status": "assumed",
                                  "constant_provenance": PROV, "scale": 0.001}},
           "series": {AVAIL_FC: {"unit": "GW", "schema": "forecast"}}}
    return dv.load_derived_specs(cfg)[AVAIL_FC]


def test_a_missing_forecast_component_gives_no_row_for_that_issue() -> None:
    f = fc([fc_row(UNAV, "2026-10-13", "2026-10-20", 1000.0, "2026-10-19 16:00"),
            fc_row(OTHER, "2026-10-13", "2026-10-21", 5.0, "2026-10-19 16:00")])  # different valid day
    assert len(dv.derive_forecasts(f, {AVAIL_FC: _spec_with_two()})) == 0


def test_forecast_components_from_different_issues_do_not_join() -> None:
    f = fc([fc_row(UNAV, "2026-10-13", "2026-10-20", 1000.0, "2026-10-19 16:00"),
            fc_row(OTHER, "2026-10-14", "2026-10-20", 5.0, "2026-10-19 16:00")])
    assert len(dv.derive_forecasts(f, {AVAIL_FC: _spec_with_two()})) == 0


def test_forecast_row_is_published_at_the_latest_component_and_carries_the_worst_grade() -> None:
    f = fc([fc_row(UNAV, "2026-10-13", "2026-10-20", 1000.0, "2026-10-19 16:00", grade="A"),
            fc_row(OTHER, "2026-10-13", "2026-10-20", 5.0, "2026-10-19 18:30", grade="B")])
    out = dv.derive_forecasts(f, {AVAIL_FC: _spec_with_two()})
    assert out["publication_time"].iloc[0] == utc("2026-10-19 18:30")
    assert out["pit_grade"].iloc[0] == "B"


# ---------------------------------------------------------------- leakage canary


def test_leakage_canary_derived_row_is_not_visible_before_its_last_input() -> None:
    """A later re-publication of one component must move the derived publication forward. A reader at an
    earlier time must still see the old derived value, never the new one."""
    spec = {AVAIL_FC: _spec_with_two()}
    first = fc([fc_row(UNAV, "2026-10-13", "2026-10-20", 1000.0, "2026-10-19 16:00"),
                fc_row(OTHER, "2026-10-13", "2026-10-20", 5.0, "2026-10-19 16:00")])
    later = fc([fc_row(UNAV, "2026-10-13", "2026-10-20", 3000.0, "2026-10-20 09:00")])  # a revised component
    d_first = dv.derive_forecasts(first, spec)
    d_both = dv.derive_forecasts(pd.concat([first, later], ignore_index=True), spec)
    assert d_first["publication_time"].iloc[0] == utc("2026-10-19 16:00")
    # the revision is not in the first derived row, and it is published only at its own time
    assert d_first["value"].iloc[0] == pytest.approx((CONST - 1000.0 + 5.0) * 0.001)
    assert d_both["publication_time"].max() == utc("2026-10-20 09:00")


def test_derived_publication_is_never_earlier_than_any_component() -> None:
    f = fc([fc_row(UNAV, "2026-10-13", "2026-10-20", 1000.0, "2026-10-19 16:00"),
            fc_row(OTHER, "2026-10-13", "2026-10-20", 5.0, "2026-10-19 17:00")])
    out = dv.derive_forecasts(f, {AVAIL_FC: _spec_with_two()})
    assert (out["publication_time"] >= f["publication_time"].max()).all()


# ---------------------------------------------------------------- the bundle path


def test_with_derived_series_adds_the_forecast_and_the_observation_rows() -> None:
    o = obs([obs_row(UNAV, "2026-10-12", 900.0, "2026-10-13 12:00")])
    f = fc([fc_row(UNAV, "2026-10-13", "2026-10-20", 1000.0, "2026-10-19 16:00")])
    b = DataBundle(o, f, empty_frame(CURVE_SCHEMA), empty_frame(CONTRACT_METADATA_SCHEMA))
    out = dv.with_derived_series(b, SPECS)
    obs_out = out.observations.loc[out.observations.series_id == AVAIL_OBS, "value"]
    assert list(obs_out) == pytest.approx([(CONST - 900.0) * 0.001])
    fc_out = out.forecasts.loc[out.forecasts.series_id == AVAIL_FC]
    assert len(fc_out) == 1 and fc_out["value"].iloc[0] == pytest.approx((CONST - 1000.0) * 0.001)


def test_a_forecast_component_cannot_drive_an_observation_spec() -> None:
    """The observation spec reads observations only: a forecast row is not an observation."""
    f = fc([fc_row(UNAV, "2026-10-13", "2026-10-20", 1000.0, "2026-10-19 16:00")])
    assert len(dv.derive_observations(obs([]), {AVAIL_OBS: SPECS[AVAIL_OBS]})) == 0
    assert len(dv.derive_forecasts(f, {AVAIL_OBS: SPECS[AVAIL_OBS]})) == 0


# ---------------------------------------------------------------- config rules


def test_sum_and_terms_together_are_refused() -> None:
    bad = {"derived": {AVAIL_OBS: {"sum": [UNAV], "terms": [{"series": UNAV, "coef": -1}], "scale": 0.001}},
           "series": {AVAIL_OBS: {"unit": "GW", "schema": "observation"}}}
    with pytest.raises(ConfigError, match="not both"):
        dv.load_derived_specs(bad)


def test_a_non_zero_constant_needs_status_and_provenance() -> None:
    bad = {"derived": {AVAIL_OBS: {"terms": [{"series": UNAV, "coef": -1}], "constant": CONST, "scale": 0.001}},
           "series": {AVAIL_OBS: {"unit": "GW", "schema": "observation"}}}
    with pytest.raises(ConfigError, match="constant_status"):
        dv.load_derived_specs(bad)


def test_forecast_components_need_a_forecast_series() -> None:
    bad = {"derived": {AVAIL_OBS: {"components_kind": "forecast", "terms": [{"series": UNAV, "coef": -1}],
                                   "scale": 0.001}},
           "series": {AVAIL_OBS: {"unit": "GW", "schema": "observation"}}}
    with pytest.raises(ConfigError, match="forecast series"):
        dv.load_derived_specs(bad)


def test_zero_coefficient_is_refused() -> None:
    bad = {"derived": {AVAIL_OBS: {"terms": [{"series": UNAV, "coef": 0}], "scale": 0.001}},
           "series": {AVAIL_OBS: {"unit": "GW", "schema": "observation"}}}
    with pytest.raises(ConfigError, match="non-zero"):
        dv.load_derived_specs(bad)


def test_the_committed_power_config_defines_both_nuclear_series_from_the_outage_id() -> None:
    specs = dv.load_derived_specs(load_config("sources/power"))
    obs_spec, fc_spec = specs[AVAIL_OBS], specs[AVAIL_FC]
    assert obs_spec.components == (UNAV,) and obs_spec.coefficient(UNAV) == -1.0
    assert fc_spec.components_kind == "forecast" and fc_spec.coefficient(UNAV) == -1.0
    assert obs_spec.constant == CONST and fc_spec.constant == CONST


def _unused() -> None:  # keeps the import list honest for linters
    _ = DataError
