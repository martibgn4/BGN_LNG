"""Climatological normals, and the anomalies measured against them.

Weather is only news relative to normal, so an anomaly is the only form in
which a LEVEL is allowed into the model at all (SPEC 0). Getting the normal
wrong therefore biases every level feature.

Two choices worth stating.

**Smoothing.** A raw day-of-year mean over thirty years is itself an estimate
with sampling noise, and its wiggles are not climate - they are the accident of
which Februaries fell in the window. Left unsmoothed, that noise enters every
anomaly as fake signal with a fixed annual pattern, which a seasonal model will
happily fit. A centred moving average removes it.

**Leap day.** 29 February gets a thirtieth of the sample of any other day. It
is mapped onto 28 February rather than being given its own thin, noisy normal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wxreturns.config import get

log = logging.getLogger(__name__)

_FEB_29 = 60          # day-of-year of 29 Feb in a leap year
_DAYS_IN_YEAR = 366


def _day_of_year(dates: pd.Series) -> pd.Series:
    """Day of year with 29 Feb folded onto 28 Feb, per config."""
    stamps = pd.to_datetime(dates)
    doy = stamps.dt.dayofyear.copy()
    if get("features", "normals", "leap_day_policy") != "map_to_feb28":
        raise ValueError("only the map_to_feb28 leap policy is implemented")
    leap = stamps.dt.is_leap_year
    # In a leap year every day after 29 Feb sits one ahead of the common-year
    # calendar; undo that so a day-of-year means the same DATE in every year,
    # and fold 29 Feb itself onto 28 Feb.
    #
    # Both rules must read the ORIGINAL day-of-year. Applying them in sequence
    # to a mutated series makes the second rule catch 1 March - which the first
    # has just shifted onto 60 - and push it to 59, so 1 March would mean a
    # different day in leap and common years and every March normal would be
    # joined against the wrong climatology.
    shifted = np.where(leap & (doy > _FEB_29), doy - 1,
                       np.where(leap & (doy == _FEB_29), _FEB_29 - 1, doy))
    return pd.Series(shifted, index=doy.index)


def _smooth_circular(values: pd.Series, window: int) -> pd.Series:
    """Centred moving average that wraps at the year boundary.

    Wrapping matters: without it the normal for 1 January is averaged over
    fewer days than the normal for 1 July, so the anomaly would be noisier at
    exactly the point in the calendar where gas trades hardest.
    """
    padded = pd.concat([values, values, values], ignore_index=True)
    smoothed = padded.rolling(window, center=True, min_periods=1).mean()
    return smoothed.iloc[len(values):(len(values) + len(values))].reset_index(drop=True)


def build_normals(history: pd.DataFrame, columns: tuple[str, ...] = ("hdd", "cdd")
                  ) -> pd.DataFrame:
    """Day-of-year normals from a regional history of realised weather.

    ``history`` is the output of `degree_days.region_degree_days` run over ERA5
    reanalysis, so every row is an ACTUAL, not a forecast.
    """
    window_years = get("features", "normals", "window_years")
    end_year = get("features", "normals", "window_end_year")
    smoothing = get("features", "normals", "smoothing_window_days")
    min_years = get("features", "normals", "min_years_required")

    frame = history.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    start_year = end_year - window_years + 1
    year = frame["target_date"].dt.year
    frame = frame[(year >= start_year) & (year <= end_year)]

    if frame.empty:
        raise ValueError(
            f"no reanalysis rows in {start_year}..{end_year}; normals cannot be "
            "built. Fetch the ERA5 archive for that window first.")

    observed_years = frame["target_date"].dt.year.nunique()
    if observed_years < min_years:
        raise ValueError(
            f"normals need at least {min_years} years and the history has "
            f"{observed_years}. A short-window normal carries the weather of "
            "those particular years into every anomaly, which is exactly the "
            "bias the anomaly exists to remove."
        )

    frame["doy"] = _day_of_year(frame["target_date"])
    grouped = (frame.groupby(["region", "doy"], as_index=False)[list(columns)]
               .mean())

    pieces: list[pd.DataFrame] = []
    for region, block in grouped.groupby("region", sort=False):
        block = block.sort_values("doy").reset_index(drop=True)
        smooth = block.copy()
        for column in columns:
            smooth[f"{column}_normal"] = _smooth_circular(block[column], smoothing)
        pieces.append(smooth.drop(columns=list(columns)))

    out = pd.concat(pieces, ignore_index=True)
    out["window_start_year"] = start_year
    out["window_end_year"] = end_year
    return out


def attach_anomalies(frame: pd.DataFrame, normals: pd.DataFrame,
                     columns: tuple[str, ...] = ("hdd", "cdd")) -> pd.DataFrame:
    """Join normals on (region, day-of-year) and difference.

    Anomaly is observed minus normal, so a positive HDD anomaly is colder than
    normal. Stated because the sign convention is a recurring source of
    confusion and the opposite reading inverts every trade.
    """
    out = frame.copy()
    out["target_date"] = pd.to_datetime(out["target_date"])
    out["doy"] = _day_of_year(out["target_date"])

    keep = ["region", "doy"] + [f"{c}_normal" for c in columns]
    merged = out.merge(normals[keep], on=["region", "doy"], how="left")

    unmatched = merged[merged[f"{columns[0]}_normal"].isna()]
    if not unmatched.empty:
        missing = sorted(unmatched["region"].unique())
        raise ValueError(
            f"no normal found for region(s) {missing} on "
            f"{len(unmatched)} row(s). Build normals for every region in the "
            "panel before attaching anomalies; an unmatched row would silently "
            "become a null anomaly."
        )

    for column in columns:
        merged[f"{column}_anomaly"] = merged[column] - merged[f"{column}_normal"]
    return merged


def fill_window_with_normals(covered: pd.DataFrame, window: pd.DatetimeIndex,
                             normals: pd.DataFrame, region: str,
                             column: str = "hdd") -> tuple[float, float]:
    """Sum a column over a delivery window, using normals past the forecast.

    Returns ``(total, coverage_fraction)``. A 15-day forecast does not reach
    the end of next month, so the far part of the window has to come from
    somewhere; climatology is the honest filler because it is what the market
    would also assume absent information.

    The coverage fraction is returned rather than logged because it must travel
    with the row: a revision computed across two different coverages is not a
    revision (SPEC 3), and `revisions.py` uses this number to refuse one.
    """
    if len(window) == 0:
        raise ValueError("empty delivery window")

    have = covered.set_index("target_date")[column]
    have = have[~have.index.duplicated(keep="last")]
    present = window.intersection(have.index)

    doy = _day_of_year(pd.Series(window, index=window))
    region_normals = normals[normals["region"] == region].set_index("doy")
    normal_column = f"{column}_normal"
    if normal_column not in region_normals.columns:
        raise KeyError(f"normals carry no {normal_column!r} for {region!r}")
    filler = doy.map(region_normals[normal_column])

    if filler.isna().any():
        raise ValueError(
            f"normals for {region!r} do not cover every day of the window; "
            "cannot fill the uncovered tail")

    combined = filler.copy()
    combined.loc[present] = have.loc[present]
    coverage = float(len(present)) / float(len(window))
    return float(combined.sum()), coverage
