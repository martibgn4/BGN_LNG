"""Day-ahead-based targets T1, T4, T5b (``docs/target_definitions.md`` §2).

Input: a day-ahead observation frame (OBSERVATION_SCHEMA) for one series, with
``event_time`` = start of the delivery gas day (06:00 Europe/Amsterdam, or local midnight - only the
local date is used) and ``publication_time`` = when the assessment was published. The TTF DA
identifier is unknown (``price.ttf.da_eur_mwh``, ``requires_user_input``), so in tests the frame is
synthetic.

T1 timing: the DA assessed on the origin's own session is already known at its close, so the T1
label is the DA assessed on session ``t + assessment_session_offset`` (default 1) for the next
working gas day after that session (``configs/targets.yaml``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.rolls import safe_log
from ttf_forecaster.data.base import MARKET_TZ, PublicationLag
from ttf_forecaster.targets.common import (
    CurveContext,
    RowCollector,
    normalise_origins,
    publication_time,
)


def da_by_gas_day(da_obs: pd.DataFrame, series_id: str, tz: str = MARKET_TZ) -> pd.DataFrame:
    """Latest vintage per gas day: index naive local gas day; columns ``value``, ``publication_time``.

    Labels are ex-post, so the latest revision is the label (a DA revision is not expected).
    """
    df = da_obs.loc[da_obs["series_id"] == series_id]
    if df.empty:
        raise ValueError(f"No rows for {series_id!r} in the day-ahead frame")
    df = df.sort_values(["revision", "publication_time"])
    day = df["event_time"].dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    out = (df.assign(gas_day=day).drop_duplicates("gas_day", keep="last")
           .set_index("gas_day")[["value", "publication_time"]].sort_index())
    return out


def build_da_level(
    ctx: CurveContext, da_obs: pd.DataFrame, *, series_id: str = "price.ttf.da_eur_mwh",
    assessment_session_offset: int = 1, publication_fallback: PublicationLag | None = None,
    origins: Iterable[pd.Timestamp] | None = None, target_id: str = "da_level",
) -> pd.DataFrame:
    """T1: DA price for the next working gas day after session ``t + offset`` (horizon 1)."""
    if assessment_session_offset < 1:
        raise ValueError("assessment_session_offset must be >= 1: offset 0 is known at origin")
    da = da_by_gas_day(da_obs, series_id, ctx.tz)
    fallback = publication_fallback or ctx.settle_lag
    last_gas_day = da.index.max()
    rows = RowCollector(ctx.is_synthetic)
    for t in normalise_origins(origins, ctx):
        s1 = ctx.calendar.add_sessions(t, assessment_session_offset)
        gas_day = ctx.calendar.next_session(s1)
        if gas_day > last_gas_day:
            continue
        if gas_day in da.index:
            val = float(da.at[gas_day, "value"])
            known = pd.Timestamp(da.at[gas_day, "publication_time"])
        else:
            val, known = np.nan, publication_time(s1, fallback, ctx.tz)
        rows.add(origin_time=ctx.pub_time(t), target_id=target_id, horizon=1, value=val,
                 label_known_time=known, contract_id=None, construction="da_assessment")
    return rows.frame()


def _month_average(da: pd.DataFrame, month_start: pd.Timestamp, ctx: CurveContext,
                   every_calendar_day: bool, gas_day_start_hour: int
                   ) -> tuple[float, pd.Timestamp]:
    month_end = month_start + pd.offsets.MonthBegin(1)
    if every_calendar_day:
        days = pd.date_range(month_start, month_end - pd.Timedelta(days=1), freq="D")
    else:
        days = ctx.calendar.sessions_in(month_start, month_end - pd.Timedelta(days=1))
    end_of_last_gas_day = (month_end + pd.Timedelta(hours=gas_day_start_hour)).tz_localize(
        ctx.tz).tz_convert("UTC")
    present = da.reindex(days)
    known = end_of_last_gas_day
    pubs = present["publication_time"].dropna()
    if len(pubs):
        known = max(known, pd.Timestamp(pubs.max()))
    vals = present["value"].to_numpy(dtype=float)
    if len(vals) == 0 or not np.all(np.isfinite(vals)):
        return float("nan"), known
    return float(vals.mean()), known


def build_da_month_avg(
    ctx: CurveContext, da_obs: pd.DataFrame, ks: Sequence[int], *,
    series_id: str = "price.ttf.da_eur_mwh", require_every_calendar_gas_day: bool = True,
    gas_day_start_hour: int = 6, origins: Iterable[pd.Timestamp] | None = None,
    target_id: str = "da_month_avg",
) -> pd.DataFrame:
    """T4: arithmetic mean of DA prices over the gas days of calendar month M+k (horizon = k).

    Any missing gas day makes the label NaN (no averaging over a partial month).
    """
    da = da_by_gas_day(da_obs, series_id, ctx.tz)
    last_gas_day = da.index.max()
    rows = RowCollector(ctx.is_synthetic)
    for k in ks:
        for t in normalise_origins(origins, ctx):
            m0 = t.replace(day=1) + pd.DateOffset(months=k)
            if m0 + pd.offsets.MonthEnd(0) > last_gas_day:
                continue
            val, known = _month_average(da, m0, ctx, require_every_calendar_gas_day,
                                        gas_day_start_hour)
            rows.add(origin_time=ctx.pub_time(t), target_id=target_id, horizon=k, value=val,
                     label_known_time=known, contract_id=None, construction="da_month_average")
    return rows.frame()


def build_delivery_excess_physical(
    ctx: CurveContext, da_obs: pd.DataFrame, ks: Sequence[int], *,
    series_id: str = "price.ttf.da_eur_mwh", require_every_calendar_gas_day: bool = True,
    gas_day_start_hour: int = 6, origins: Iterable[pd.Timestamp] | None = None,
    target_id: str = "delivery_excess_phys", log_target_id: str = "delivery_excess_phys_log",
) -> pd.DataFrame:
    """T5b: T4 (DA average of month M+k) minus the settlement of month M+k at t; plus log ratio.

    Origins on or after the M+k contract's last trading day are not emitted.
    """
    da = da_by_gas_day(da_obs, series_id, ctx.tz)
    last_gas_day = da.index.max()
    rows = RowCollector(ctx.is_synthetic)
    for k in ks:
        for t in normalise_origins(origins, ctx):
            m0 = t.replace(day=1) + pd.DateOffset(months=k)
            if m0 + pd.offsets.MonthEnd(0) > last_gas_day:
                continue
            cid = f"{ctx.hub}.M.{m0.year:04d}-{m0.month:02d}"
            avg, known = _month_average(da, m0, ctx, require_every_calendar_gas_day,
                                        gas_day_start_hour)
            i = ctx.chain.index_of.get(cid)
            if i is not None and t >= ctx.chain.last_trade[i]:
                continue  # M+k contract expired: no futures price at t to compare against
            f0, cons = ctx.lookup.get(cid, t), "da_month_average_vs_settle"
            diff = avg - f0
            logr = float(safe_log(avg) - safe_log(f0)) if np.isfinite(diff) else np.nan
            for tid, v in ((target_id, diff), (log_target_id, logr)):
                rows.add(origin_time=ctx.pub_time(t), target_id=tid, horizon=k, value=v,
                         label_known_time=known, contract_id=cid, construction=cons)
    return rows.frame()
