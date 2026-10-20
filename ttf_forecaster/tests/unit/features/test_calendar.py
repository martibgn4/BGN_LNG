"""Calendar family and expiry schedule. SYNTHETIC / deterministic inputs only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import MARKET_TZ, DataBundle
from ttf_forecaster.features.calendar import (
    build_calendar_features,
    front_months,
    holiday_array,
    monthly_last_trade_dates,
)
from ttf_forecaster.pipelines.build_features import family_params

from .helpers import config


def _o(*days: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(f"{d} 19:00", tz=MARKET_TZ) for d in days]).tz_convert(
        "UTC")


def test_basic_calendar_columns() -> None:
    p = family_params(config(), "calendar")
    f = build_calendar_features(DataBundle.empty(), _o("2022-09-30", "2022-10-03", "2023-03-31"), p)
    assert f["calendar__gas_year"].tolist() == [2021.0, 2022.0, 2022.0]
    assert f["calendar__gas_year_day"].tolist() == [364.0, 2.0, 181.0]
    assert f["calendar__is_heating_season"].tolist() == [0.0, 1.0, 1.0]
    assert f["calendar__is_injection_season"].tolist() == [1.0, 0.0, 0.0]
    assert f["calendar__weekday"].tolist() == [4.0, 0.0, 4.0]
    assert f["calendar__next_bday_gap_days"].tolist() == [3.0, 1.0, 3.0]
    assert set(f.attrs["pit_grade"].values()) == {"A"}


def test_days_to_expiry_rule_and_holidays() -> None:
    p = family_params(config(), "calendar")
    # Nov-2022 last trade = Fri 28 Oct (2 bdays before Tue 1 Nov). On Mon 24 Oct: 4 bdays after.
    f = build_calendar_features(DataBundle.empty(), _o("2022-10-24"), p)
    assert f["calendar__bdays_to_prompt_expiry"].iloc[0] == 4.0
    assert f["calendar__is_expiry_week"].iloc[0] == 1.0
    # front (R=5) has already rolled to Dec-22: LTD Tue 29 Nov
    assert f["calendar__bdays_to_front_expiry"].iloc[0] == float(
        np.busday_count(np.datetime64("2022-10-25"), np.datetime64("2022-11-30")))
    # a holiday on Wed 26 Oct pushes expiry one bday earlier counted, and is the next holiday
    p2 = {**p, "holidays": ["2022-10-26"]}
    f2 = build_calendar_features(DataBundle.empty(), _o("2022-10-24"), p2)
    assert f2["calendar__bdays_to_next_holiday"].iloc[0] == 2.0
    assert f2["calendar__bdays_to_prompt_expiry"].iloc[0] == 3.0


def test_vendor_metadata_overrides_rule() -> None:
    months = pd.date_range("2022-10-01", "2022-12-01", freq="MS")
    meta = pd.DataFrame({
        "contract_id": ["TTF.M.2022-11"], "hub": ["TTF"], "tenor_type": ["M"],
        "delivery_start": [pd.Timestamp("2022-11-01", tz=MARKET_TZ).tz_convert("UTC")],
        "delivery_end": [pd.Timestamp("2022-12-01", tz=MARKET_TZ).tz_convert("UTC")],
        "last_trade_date": [pd.Timestamp("2022-10-27", tz=MARKET_TZ).tz_convert("UTC")],
        "last_trade_date_source": ["vendor"], "currency": ["EUR"], "unit": ["EUR/MWh"],
    })
    ltd = monthly_last_trade_dates(months, hub="TTF", contracts=meta, bdays_before=2,
                                   holidays=holiday_array([]))
    assert ltd[pd.Timestamp("2022-11-01")] == pd.Timestamp("2022-10-27")
    assert ltd.attrs["source"][pd.Timestamp("2022-11-01")] == "vendor"
    assert ltd[pd.Timestamp("2022-12-01")] == pd.Timestamp("2022-11-29")


def test_front_month_needs_long_enough_schedule() -> None:
    months = pd.date_range("2022-01-01", "2022-02-01", freq="MS")
    ltd = monthly_last_trade_dates(months, hub="TTF", contracts=None, bdays_before=2,
                                   holidays=holiday_array([]))
    with pytest.raises(ValueError, match="far enough"):
        front_months(pd.DatetimeIndex(["2022-03-15"]), ltd, roll_offset_bdays=5,
                     holidays=holiday_array([]))


def test_missing_season_config_is_an_error() -> None:
    p = family_params(config(), "calendar")
    p.pop("maintenance_season_months")
    with pytest.raises(KeyError, match="maintenance_season_months"):
        build_calendar_features(DataBundle.empty(), _o("2022-10-24"), p)
