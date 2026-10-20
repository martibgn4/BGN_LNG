"""Futures-based targets T2, T3, T5a, T6, T7, T8, T9 (``docs/target_definitions.md`` §2).

Every return is on **one fixed contract** (or one fixed strip) between origin ``t`` and exit
``t + h`` sessions. Spreads are differences in EUR/MWh, never logs. Each builder returns
TARGET_SCHEMA rows (see :mod:`ttf_forecaster.targets.common` for timing and missing-data rules).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.futures import FuturesContract, parse_contract_id
from ttf_forecaster.contracts.rolls import safe_log
from ttf_forecaster.targets.common import (
    CurveContext,
    RowCollector,
    normalise_origins,
    publication_time,
)
from ttf_forecaster.targets.strips import held_strip, leg_prices

_MONTH_ID = "{hub}.M.{y:04d}-{m:02d}"


@dataclass(frozen=True)
class _Held:
    t: pd.Timestamp
    exit: pd.Timestamp
    cid: str | None
    idx: int | None
    p0: float
    p1: float


def _iter_held(ctx: CurveContext, origins: pd.DatetimeIndex, h: int) -> Iterator[_Held]:
    """Held monthly contract and its two settlements for each origin whose exit is observable."""
    if h < 1:
        raise ValueError("horizon must be >= 1 session")
    last = ctx.lookup.last_day
    for t in origins:
        exit_day = ctx.calendar.add_sessions(t, h)
        if exit_day > last:
            continue
        i = ctx.chain.held_index(ctx.calendar.position(t), h, ctx.rule)
        if i is None:
            yield _Held(t, exit_day, None, None, np.nan, np.nan)
            continue
        cid = ctx.chain.contract_ids[i]
        yield _Held(t, exit_day, cid, i, ctx.lookup.get(cid, t), ctx.lookup.get(cid, exit_day))


def _logret(p0: float, p1: float) -> float:
    return float(safe_log(p1) - safe_log(p0))


def build_m1_settle(ctx: CurveContext, horizons: Sequence[int], *,
                    origins: Iterable[pd.Timestamp] | None = None,
                    target_id: str = "m1_settle") -> pd.DataFrame:
    """T2: settlement of the held front contract at ``t + h`` (EUR/MWh)."""
    rows = RowCollector(ctx.is_synthetic)
    days = normalise_origins(origins, ctx)
    for h in horizons:
        for x in _iter_held(ctx, days, h):
            rows.add(origin_time=ctx.pub_time(x.t), target_id=target_id, horizon=h, value=x.p1,
                     label_known_time=ctx.pub_time(x.exit), contract_id=x.cid,
                     construction="single_contract")
    return rows.frame()


def build_m1_logret(ctx: CurveContext, horizons: Sequence[int], *,
                    origins: Iterable[pd.Timestamp] | None = None,
                    target_id: str = "m1_logret") -> pd.DataFrame:
    """T3: ``r(t, h) = ln F_c(t+h) - ln F_c(t)`` on the held contract ``c``."""
    rows = RowCollector(ctx.is_synthetic)
    days = normalise_origins(origins, ctx)
    for h in horizons:
        for x in _iter_held(ctx, days, h):
            rows.add(origin_time=ctx.pub_time(x.t), target_id=target_id, horizon=h,
                     value=_logret(x.p0, x.p1), label_known_time=ctx.pub_time(x.exit),
                     contract_id=x.cid, construction="single_contract")
    return rows.frame()


def build_direction(ctx: CurveContext, horizons: Sequence[int], *,
                    origins: Iterable[pd.Timestamp] | None = None,
                    target_id: str = "dir") -> pd.DataFrame:
    """T9a: ``1[r(t, h) > 0]`` (1.0 / 0.0; NaN when r is missing)."""
    rows = RowCollector(ctx.is_synthetic)
    days = normalise_origins(origins, ctx)
    for h in horizons:
        for x in _iter_held(ctx, days, h):
            r = _logret(x.p0, x.p1)
            rows.add(origin_time=ctx.pub_time(x.t), target_id=target_id, horizon=h,
                     value=float(r > 0) if np.isfinite(r) else np.nan,
                     label_known_time=ctx.pub_time(x.exit), contract_id=x.cid,
                     construction="single_contract")
    return rows.frame()


def build_tail(ctx: CurveContext, horizons: Sequence[int], *, k: float = 1.5,
               mode: str = "ewma", ewma_span: int = 60, ewma_min_periods: int = 20,
               fixed_threshold: float = 0.10, origins: Iterable[pd.Timestamp] | None = None,
               target_id: str = "tail") -> pd.DataFrame:
    """T9b: ``1[|r(t,h)| > k * sigma_t * sqrt(h)]`` with sigma_t the EWMA daily vol of the spliced
    front-month return series **up to and including t**; ``mode="fixed"`` uses
    ``|r| > fixed_threshold`` instead (sensitivity)."""
    if mode not in ("ewma", "fixed"):
        raise ValueError(f"tail mode must be 'ewma' or 'fixed', got {mode!r}")
    sigma = ctx.front_return_sigma(ewma_span, ewma_min_periods) if mode == "ewma" else None
    rows = RowCollector(ctx.is_synthetic)
    days = normalise_origins(origins, ctx)
    for h in horizons:
        for x in _iter_held(ctx, days, h):
            r = _logret(x.p0, x.p1)
            if mode == "ewma":
                assert sigma is not None
                s = float(sigma.get(x.t, np.nan))
                thr = k * s * np.sqrt(h)
            else:
                thr = fixed_threshold
            val = float(abs(r) > thr) if np.isfinite(r) and np.isfinite(thr) else np.nan
            rows.add(origin_time=ctx.pub_time(x.t), target_id=target_id, horizon=h, value=val,
                     label_known_time=ctx.pub_time(x.exit), contract_id=x.cid,
                     construction=f"single_contract_{mode}")
    return rows.frame()


def build_realised_vol(ctx: CurveContext, horizons: Sequence[int], *,
                       annualisation: int = 252, origins: Iterable[pd.Timestamp] | None = None,
                       target_id: str = "rv") -> pd.DataFrame:
    """T8: ``sqrt(annualisation / h * sum r_d^2)`` over daily log returns of the held contract on
    sessions (t, t+h]. Any missing daily settlement makes the label NaN."""
    rows = RowCollector(ctx.is_synthetic)
    days = normalise_origins(origins, ctx)
    cal = ctx.calendar
    for h in horizons:
        for x in _iter_held(ctx, days, h):
            val = np.nan
            if x.cid is not None:
                p0 = cal.position(x.t)
                path = [ctx.lookup.get(x.cid, cal.session_at(p0 + j)) for j in range(h + 1)]
                logs = np.asarray(safe_log(np.array(path)))
                if np.all(np.isfinite(logs)):
                    val = float(np.sqrt(annualisation / h * np.sum(np.diff(logs) ** 2)))
            rows.add(origin_time=ctx.pub_time(x.t), target_id=target_id, horizon=h, value=val,
                     label_known_time=ctx.pub_time(x.exit), contract_id=x.cid,
                     construction="single_contract")
    return rows.frame()


def build_delivery_excess_futures(
    ctx: CurveContext, ks: Sequence[int], *, origins: Iterable[pd.Timestamp] | None = None,
    target_id: str = "delivery_excess_fut", log_target_id: str = "delivery_excess_fut_log",
) -> pd.DataFrame:
    """T5a: final settlement of month M+k minus its settlement at t (EUR/MWh), plus the log ratio.

    M is the calendar month of t. The label is known at the contract's last trading day. Origins on
    or after that last trading day (label already known / contract expired) are not emitted.
    """
    rows = RowCollector(ctx.is_synthetic)
    days = normalise_origins(origins, ctx)
    last = ctx.lookup.last_day
    for k in ks:
        for t in days:
            target = t.replace(day=1) + pd.DateOffset(months=k)
            cid = _MONTH_ID.format(hub=ctx.hub, y=target.year, m=target.month)
            i = ctx.chain.index_of.get(cid)
            if i is None:
                continue
            ltd = ctx.chain.last_trade[i]
            if ltd > last:
                continue
            known = ctx.pub_time(ltd)
            if t >= ltd:
                continue  # contract no longer trades after t: no forecastable label
            f0, fin = ctx.lookup.get(cid, t), ctx.lookup.get(cid, ltd)
            rows.add(origin_time=ctx.pub_time(t), target_id=target_id, horizon=k, value=fin - f0,
                     label_known_time=known, contract_id=cid, construction="single_contract")
            rows.add(origin_time=ctx.pub_time(t), target_id=log_target_id, horizon=k,
                     value=_logret(f0, fin), label_known_time=known, contract_id=cid,
                     construction="single_contract")
    return rows.frame()


def build_strip_logret(ctx: CurveContext, horizons: Sequence[int], tenor: str, *,
                       origins: Iterable[pd.Timestamp] | None = None,
                       target_id: str | None = None) -> pd.DataFrame:
    """T6: log return of the held prompt quarter (``tenor="Q"``) or next season (``"S"``),
    hours-weighted strip of monthly settlements, else listed contract (flagged)."""
    tid = target_id or {"Q": "q_logret", "S": "season_logret"}[tenor]
    rows = RowCollector(ctx.is_synthetic)
    days = normalise_origins(origins, ctx)
    last = ctx.lookup.last_day
    for h in horizons:
        for t in days:
            exit_day = ctx.calendar.add_sessions(t, h)
            if exit_day > last:
                continue
            c = held_strip(ctx, t, exit_day, tenor)
            if c is None:
                val, cid, cons = np.nan, None, "strip"
            else:
                p0, p1, cons = leg_prices(ctx, c, t, exit_day)
                val, cid = _logret(p0, p1), c.contract_id
            rows.add(origin_time=ctx.pub_time(t), target_id=tid, horizon=h, value=val,
                     label_known_time=ctx.pub_time(exit_day), contract_id=cid, construction=cons)
    return rows.frame()


def _spread_legs(ctx: CurveContext, which: str, t: pd.Timestamp, exit_day: pd.Timestamp, h: int
                 ) -> tuple[FuturesContract, FuturesContract] | None:
    if which == "m1m2":
        i = ctx.chain.held_index(ctx.calendar.position(t), h, ctx.rule)
        if i is None or i + 1 >= len(ctx.chain.contract_ids):
            return None
        return (parse_contract_id(ctx.chain.contract_ids[i]),
                parse_contract_id(ctx.chain.contract_ids[i + 1]))
    if which == "q1q2":
        q = held_strip(ctx, t, exit_day, "Q")
        return None if q is None else (q, q.next())
    if which == "sum_win":
        s = held_strip(ctx, t, exit_day, "S", season="SUM")
        return None if s is None else (s, s.next())
    if which == "win_sum":
        w = held_strip(ctx, t, exit_day, "S", season="WIN")
        return None if w is None else (w, w.next())
    raise ValueError(f"unknown spread {which!r}; expected m1m2, q1q2, sum_win, win_sum")


def build_spread_change(ctx: CurveContext, horizons: Sequence[int], which: str, *,
                        origins: Iterable[pd.Timestamp] | None = None,
                        target_id: str | None = None) -> pd.DataFrame:
    """T7a-d: change over h of (near leg - far leg) in EUR/MWh, both legs fixed at t.

    ``which``: ``m1m2`` (held front month vs next month), ``q1q2`` (prompt vs next quarter
    strips), ``sum_win`` (next Summer vs following Winter), ``win_sum`` (next Winter vs following
    Summer). ``contract_id`` is ``"<near>|<far>"``.
    """
    tid = target_id or f"spread_{which}_chg"
    rows = RowCollector(ctx.is_synthetic)
    days = normalise_origins(origins, ctx)
    last = ctx.lookup.last_day
    for h in horizons:
        for t in days:
            exit_day = ctx.calendar.add_sessions(t, h)
            if exit_day > last:
                continue
            legs = _spread_legs(ctx, which, t, exit_day, h)
            if legs is None:
                val, cid, cons = np.nan, None, "spread"
            else:
                a0, a1, ca = leg_prices(ctx, legs[0], t, exit_day)
                b0, b1, cb = leg_prices(ctx, legs[1], t, exit_day)
                val = (a1 - b1) - (a0 - b0)
                cid = f"{legs[0].contract_id}|{legs[1].contract_id}"
                cons = ca if ca == cb else f"{ca}|{cb}"
            rows.add(origin_time=ctx.pub_time(t), target_id=tid, horizon=h, value=val,
                     label_known_time=ctx.pub_time(exit_day), contract_id=cid, construction=cons)
    return rows.frame()


def build_regime_change(ctx: CurveContext, regimes: pd.Series, horizons: Sequence[int], *,
                        known_times: pd.Series | None = None,
                        origins: Iterable[pd.Timestamp] | None = None,
                        target_id: str = "regime_change") -> pd.DataFrame:
    """T9c: ``1[regime(t+h) != regime(t)]``.

    Args:
        regimes: regime label per trading date (naive local dates), each computed by the
            regime model with information available at that date (``models/regimes.py``).
        known_times: optional UTC time each regime label became known; the label's known time is
            ``max(settlement publication of t+h, known_times[t+h])``. Supply it when regimes use
            data published after the settlement (e.g. storage at D+2).
    """
    reg = regimes.copy()
    reg.index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in reg.index])
    kt = None
    if known_times is not None:
        kt = known_times.copy()
        kt.index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in kt.index])
    rows = RowCollector(ctx.is_synthetic)
    days = normalise_origins(origins, ctx)
    last = min(ctx.lookup.last_day, reg.index.max())
    for h in horizons:
        for t in days:
            exit_day = ctx.calendar.add_sessions(t, h)
            if exit_day > last:
                continue
            a, b = reg.get(t), reg.get(exit_day)
            val = np.nan if (a is None or b is None or pd.isna(a) or pd.isna(b)) else float(a != b)
            known = ctx.pub_time(exit_day)
            if kt is not None and exit_day in kt.index and pd.notna(kt[exit_day]):
                known = max(known, pd.Timestamp(kt[exit_day]).tz_convert("UTC"))
            rows.add(origin_time=ctx.pub_time(t), target_id=target_id, horizon=h, value=val,
                     label_known_time=known, contract_id=None, construction="regime_series")
    return rows.frame()


__all__ = [
    "build_delivery_excess_futures",
    "build_direction",
    "build_m1_logret",
    "build_m1_settle",
    "build_realised_vol",
    "build_regime_change",
    "build_spread_change",
    "build_strip_logret",
    "build_tail",
    "publication_time",
]
