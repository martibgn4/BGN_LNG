"""Crop-calendar gating for agricultural weather.

Degree days do not transfer to ags. The mechanism is crop stress inside a
phenological window, and the window is most of the model: July rain over Iowa
is decisive in the third week of July and close to irrelevant in April. An
ungated ag weather feature averages the informative days with a much larger
number of uninformative ones, and the standard result is a signal diluted into
noise that nonetheless fits in sample.

Hemispheres are kept separate because they are genuinely different crops in
different years - Brazilian pod-fill is January, and a northern-hemisphere
calendar applied to Mato Grosso is not merely offset, it is wrong.

Windows are day-of-year and therefore wrap the year end for southern crops.
`_in_window` handles the wrap rather than assuming start <= end.
"""

from __future__ import annotations

import logging

import pandas as pd

from wxreturns.config import get

log = logging.getLogger(__name__)

_MONTH_DAY = "%m-%d"


def _to_month_day(value: str) -> tuple[int, int]:
    month, day = value.split("-")
    return int(month), int(day)


def _in_window(dates: pd.Series, start: str, end: str) -> pd.Series:
    """Month-day membership, wrapping across the year boundary when needed."""
    stamps = pd.to_datetime(dates)
    key = stamps.dt.month * 100 + stamps.dt.day
    start_month, start_day = _to_month_day(start)
    end_month, end_day = _to_month_day(end)
    lo, hi = start_month * 100 + start_day, end_month * 100 + end_day
    if lo <= hi:
        return (key >= lo) & (key <= hi)
    # Wraps December into January - every southern-hemisphere critical window.
    return (key >= lo) | (key <= hi)


def calendar_for(crop: str, hemisphere: str) -> list[dict]:
    calendars = get("features", "crop_calendar")
    if crop not in calendars:
        raise KeyError(
            f"no crop calendar for {crop!r}; features.yaml defines "
            f"{sorted(calendars)}")
    if hemisphere not in calendars[crop]:
        raise KeyError(
            f"crop {crop!r} has no {hemisphere!r} calendar. Southern windows "
            "must be declared explicitly rather than derived by a six-month "
            "shift, which is not how planting dates work."
        )
    return calendars[crop][hemisphere]


def hemisphere_for_region(region: str) -> str:
    """South if every point in the region is below the equator."""
    points = get("regions", "regions", region, "points")
    return "southern" if all(p["lat"] < 0.0 for p in points) else "northern"


def label_phases(frame: pd.DataFrame, crop: str, region: str,
                 date_column: str = "target_date") -> pd.DataFrame:
    """Attach the phenological phase and whether it is a critical window."""
    hemisphere = hemisphere_for_region(region)
    phases = calendar_for(crop, hemisphere)

    out = frame.copy()
    out["phase"] = None
    out["critical"] = False
    out["hemisphere"] = hemisphere

    for phase in phases:
        mask = _in_window(out[date_column], phase["start"], phase["end"])
        out.loc[mask, "phase"] = phase["name"]
        out.loc[mask, "critical"] = bool(phase["critical"])

    return out


def heat_stress_days(frame: pd.DataFrame, crop: str,
                     max_column: str = "value") -> pd.Series:
    """Days whose maximum temperature exceeds the crop's stress threshold.

    A count rather than a mean: yield loss above the threshold is not linear in
    temperature, and averaging a 38C day with a 20C day produces a number that
    describes neither.
    """
    threshold = get("features", "agriculture_features",
                    "heat_stress_threshold_c", crop)
    return (frame[max_column] > float(threshold)).astype(int)


def gate_to_critical(frame: pd.DataFrame, crop: str, region: str,
                     date_column: str = "target_date") -> pd.DataFrame:
    """Keep only rows inside a critical window.

    Rows outside are dropped rather than zeroed. Zero would enter a regression
    as a real observation of "no stress", which is a claim about a crop that is
    not in the ground.
    """
    labelled = label_phases(frame, crop, region, date_column)
    gated = labelled[labelled["critical"]]
    if gated.empty:
        raise ValueError(
            f"no rows for {crop!r} in {region!r} fall inside a critical window. "
            f"For a {hemisphere_for_region(region)}-hemisphere region the "
            "critical windows are "
            f"{[p['name'] + ' ' + p['start'] + '..' + p['end'] for p in calendar_for(crop, hemisphere_for_region(region)) if p['critical']]}. "
            "Outside those, ag weather carries close to zero information and "
            "the model deliberately declines to trade it."
        )
    return gated


def crop_year_window(commodity: str, contract_month: int, contract_year: int
                     ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The growing season a given contract references.

    Unlike gas, an ag contract does not reference a month of weather - it
    references a HARVEST. A December corn contract is exposed to the northern
    growing season of that same year; a March contract to the season already
    harvested, so its weather sensitivity is to South America instead. This
    returns the northern-hemisphere season bounds and callers switch region on
    the basis of it.
    """
    contracts = get("targets", "contracts")
    if commodity not in contracts:
        raise KeyError(f"unknown commodity {commodity!r}")
    crop = contracts[commodity]["crop"]
    phases = calendar_for(crop, "northern")

    first, last = phases[0], phases[-1]
    start_month, start_day = _to_month_day(first["start"])
    end_month, end_day = _to_month_day(last["end"])

    # A contract expiring before the season ends belongs to the PREVIOUS crop
    # year: March corn is old-crop, and its weather story finished last autumn.
    year = contract_year if contract_month > end_month else contract_year - 1
    start = pd.Timestamp(year=year, month=start_month, day=start_day)
    end = pd.Timestamp(year=year, month=end_month, day=end_day)
    if end < start:
        end = end + pd.DateOffset(years=1)
    return start, end
