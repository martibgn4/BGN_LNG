"""Calendar features and the monthly-expiry / front-contract schedule they share with prices.

Everything here is a deterministic function of the origin's **local date** (Europe/Amsterdam),
a holiday calendar and contract expiry dates passed in (``params`` / ``bundle.contracts``), so the
PIT grade of every column is ``A``. Season definitions are config (``configs/features.yaml``).

Expiry dates: vendor contract metadata (``bundle.contracts.last_trade_date``) when present for the
month, else the configured rule "``expiry_bdays_before_delivery`` business days before the first
calendar day of delivery" (``target_definitions.md`` §1, **[CONFIRM]**) on the configured holiday
calendar. Futures expiries are fixed by rule in advance, so using them is not hindsight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import MARKET_TZ, DataBundle, PitGrade
from ttf_forecaster.features.panel import as_origin_index

FAMILY = "calendar"


# ---------------------------------------------------------------------------------------------
# Holiday calendar and expiry schedule
# ---------------------------------------------------------------------------------------------


def holiday_array(holidays: Sequence[str | pd.Timestamp] | None) -> np.ndarray:
    """Configured holidays (local dates) as ``datetime64[D]`` for ``numpy.busday_*``."""
    if not holidays:
        return np.array([], dtype="datetime64[D]")
    return np.array(sorted(pd.Timestamp(h).date() for h in holidays), dtype="datetime64[D]")


def local_dates(ts: pd.DatetimeIndex | pd.Series, tz: str = MARKET_TZ) -> pd.DatetimeIndex:
    """tz-aware timestamps -> naive local midnight (the local calendar date)."""
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        raise ValueError("local_dates needs tz-aware timestamps")
    return idx.tz_convert(tz).tz_localize(None).normalize()


def monthly_last_trade_dates(
    delivery_months: pd.DatetimeIndex,
    *,
    hub: str,
    contracts: pd.DataFrame | None,
    bdays_before: int,
    holidays: np.ndarray,
    tz: str = MARKET_TZ,
) -> pd.Series:
    """Last trading (local) date per monthly delivery month.

    Args:
        delivery_months: naive month-start dates.
        hub: hub code to look up in ``contracts``.
        contracts: CONTRACT_METADATA_SCHEMA rows (may be empty / None).
        bdays_before: rule fallback, business days before the first delivery day.
        holidays: ``holiday_array`` output.

    Returns:
        Series indexed by month start (naive) -> naive local last trade date; attribute
        ``attrs["source"]`` maps month -> ``"vendor"`` or ``"rule"``.
    """
    months = pd.DatetimeIndex(delivery_months).normalize()
    first = months.values.astype("datetime64[D]")
    rule = np.busday_offset(first, -int(bdays_before), roll="forward", holidays=holidays)
    out = pd.Series(pd.DatetimeIndex(rule), index=months, name="last_trade_date")
    source = pd.Series("rule", index=months)
    if contracts is not None and len(contracts):
        meta = contracts.loc[(contracts["hub"] == hub) & (contracts["tenor_type"] == "M")]
        if len(meta):
            m_start = local_dates(meta["delivery_start"], tz)
            ltd = local_dates(meta["last_trade_date"], tz)
            vendor = pd.Series(ltd, index=m_start).groupby(level=0).last()
            hit = vendor.index.intersection(months)
            out.loc[hit] = vendor.loc[hit].to_numpy()
            source.loc[hit] = "vendor"
    out.attrs["source"] = source.to_dict()
    return out


def month_range(start: pd.Timestamp, end: pd.Timestamp, pad_months: int = 0) -> pd.DatetimeIndex:
    """Month starts covering [start, end + pad_months]."""
    s = pd.Timestamp(start).to_period("M").to_timestamp()
    e = (pd.Timestamp(end) + pd.DateOffset(months=pad_months)).to_period("M").to_timestamp()
    return pd.date_range(s, e, freq="MS")


def front_months(
    dates: pd.DatetimeIndex,
    ltd: pd.Series,
    *,
    roll_offset_bdays: int,
    holidays: np.ndarray,
) -> pd.DatetimeIndex:
    """Front monthly contract per local date: nearest month whose last trading day is **more
    than** ``roll_offset_bdays`` business days after the date (``target_definitions.md`` §1).

    ``roll_offset_bdays = 0`` gives the *prompt* (nearest not yet expired, ``ltd > date``) and
    ``-1`` the nearest with ``ltd >= date`` (still trading today).
    """
    d = pd.DatetimeIndex(dates).values.astype("datetime64[D]")
    ltd_sorted = ltd.sort_index()
    ltd_vals = ltd_sorted.values.astype("datetime64[D]")
    if not np.all(np.diff(ltd_vals.astype("int64")) > 0):
        raise ValueError("last trade dates are not strictly increasing with delivery month")
    if roll_offset_bdays >= 0:
        # ltd must be >= the (R+1)-th business day after d
        threshold = np.busday_offset(d, roll_offset_bdays + 1, roll="backward", holidays=holidays)
        pos = np.searchsorted(ltd_vals, threshold, side="left")
    else:
        pos = np.searchsorted(ltd_vals, d, side="left")
    if (pos >= len(ltd_vals)).any():
        raise ValueError("expiry schedule does not extend far enough for the dates requested")
    return pd.DatetimeIndex(ltd_sorted.index[pos])


# ---------------------------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------------------------


def _months_flag(months: pd.Index, wanted: Sequence[int]) -> np.ndarray:
    return np.isin(np.asarray(months), np.asarray(list(wanted))).astype(float)


def build_calendar_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: Mapping[str, object]
) -> pd.DataFrame:
    """Calendar family. Params (``configs/features.yaml``):

    ``hub``, ``roll_offset_bdays``, ``expiry_bdays_before_delivery``, ``holidays`` (list of local
    dates), ``gas_year_start_month``, ``heating_season_months``, ``injection_season_months``,
    ``withdrawal_season_months``, ``maintenance_season_months``, ``holiday_horizon_bdays``,
    ``expiry_week_bdays``.

    Columns (all grade A; ``bdays`` = business days on the configured calendar):
        weekday (0=Mon), month, doy_sin, doy_cos, gas_year, gas_year_day, is_heating_season,
        is_injection_season, is_withdrawal_season, is_maintenance_season, next_bday_gap_days,
        bdays_to_next_holiday (NaN if none within the horizon), bdays_to_prompt_expiry,
        bdays_to_front_expiry, is_expiry_week.
    """
    idx = as_origin_index(origins)
    p = dict(params)
    hol = holiday_array(p.get("holidays"))  # type: ignore[arg-type]
    dates = local_dates(idx)
    out = pd.DataFrame(index=idx)
    if len(idx) == 0:
        out.attrs["pit_grade"] = {}
        return out
    d64 = dates.values.astype("datetime64[D]")
    gy_start = int(p.get("gas_year_start_month", 10))  # type: ignore[call-overload]

    out[f"{FAMILY}__weekday"] = dates.weekday.astype(float)
    out[f"{FAMILY}__month"] = dates.month.astype(float)
    angle = 2 * np.pi * (dates.dayofyear.to_numpy() - 1) / 365.25
    out[f"{FAMILY}__doy_sin"] = np.sin(angle)
    out[f"{FAMILY}__doy_cos"] = np.cos(angle)
    gas_year = np.where(dates.month >= gy_start, dates.year, dates.year - 1)
    out[f"{FAMILY}__gas_year"] = gas_year.astype(float)
    gy_first = pd.to_datetime({"year": gas_year, "month": gy_start, "day": 1})
    out[f"{FAMILY}__gas_year_day"] = (dates - pd.DatetimeIndex(gy_first)).days.astype(float)
    for name, key in [
        ("is_heating_season", "heating_season_months"),
        ("is_injection_season", "injection_season_months"),
        ("is_withdrawal_season", "withdrawal_season_months"),
        ("is_maintenance_season", "maintenance_season_months"),
    ]:
        if key not in p:
            raise KeyError(f"calendar params missing {key!r} (configs/features.yaml)")
        out[f"{FAMILY}__{name}"] = _months_flag(dates.month, p[key])  # type: ignore[arg-type]

    nxt = np.busday_offset(d64, 1, roll="backward", holidays=hol)
    out[f"{FAMILY}__next_bday_gap_days"] = (nxt - d64).astype("int64").astype(float)
    horizon = int(p.get("holiday_horizon_bdays", 20))  # type: ignore[call-overload]
    if len(hol):
        pos = np.searchsorted(hol, d64, side="right")
        has_next = pos < len(hol)
        nxt_h = hol[np.minimum(pos, len(hol) - 1)]
        gap = np.busday_count(d64, nxt_h, holidays=hol).astype(float)
        gap[~has_next | (gap > horizon)] = np.nan
    else:
        gap = np.full(len(d64), np.nan)
    out[f"{FAMILY}__bdays_to_next_holiday"] = gap

    hub = str(p.get("hub", "TTF"))
    months = month_range(dates.min(), dates.max(), pad_months=4)
    ltd = monthly_last_trade_dates(
        months, hub=hub, contracts=bundle.contracts,
        bdays_before=int(p["expiry_bdays_before_delivery"]),  # type: ignore[call-overload]
        holidays=hol,
    )
    ltd_d = ltd.values.astype("datetime64[D]")
    for name, r in [("prompt", -1), ("front", int(p["roll_offset_bdays"]))]:  # type: ignore[call-overload]
        fm = front_months(dates, ltd, roll_offset_bdays=r, holidays=hol)
        fm_ltd = ltd_d[np.searchsorted(ltd.index.values, fm.values)]
        # business days strictly after the date up to and including the last trade date
        n_bd = np.busday_count(d64 + 1, fm_ltd + 1, holidays=hol).astype(float)
        out[f"{FAMILY}__bdays_to_{name}_expiry"] = n_bd
    week = int(p.get("expiry_week_bdays", 5))  # type: ignore[call-overload]
    out[f"{FAMILY}__is_expiry_week"] = (out[f"{FAMILY}__bdays_to_prompt_expiry"] <= week).astype(
        float
    )
    out.index.name = "origin_time"
    out.attrs["pit_grade"] = {c: PitGrade.A.value for c in out.columns}
    return out
