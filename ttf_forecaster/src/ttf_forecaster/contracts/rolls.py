"""Front / held contract selection and continuous series (``docs/target_definitions.md`` §1).

* **Front contract on t** (roll rule R): the nearest monthly contract whose last trading day is
  *more than* ``R`` sessions after ``t``.
* **Held contract for (t, h)**: must still trade on the exit session ``t + h``
  (last trade >= exit + ``R'`` sessions). By default it is also at or beyond the front contract
  under R (``held_respects_roll_rule``), so a 1-day target never sits in expiry week; set it false
  for the literal doc rule.
* **Continuous level** (features only): unadjusted, switching contract on the roll.
* **Continuous return** (features only): daily within-contract log returns; the return on session
  ``t_i`` is measured on the front contract *as of* ``t_{i-1}`` (the position held over the day), so
  both prices are always from one contract. A difference across a roll is never a return.

Settlement panels are wide frames: index = naive local trading dates, columns = contract ids,
values = settlement prices. Missing settlements stay NaN; nothing is interpolated or skipped over.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.calendars import DateLike, TradingCalendar, to_local_day
from ttf_forecaster.contracts.futures import parse_contract_id
from ttf_forecaster.data.base import (
    MARKET_TZ,
    DataError,
    NonPositivePriceError,
    PitGrade,
    worst_grade,
)
from ttf_forecaster.settings import get, load_config


class RollError(DataError):
    """No listed contract satisfies the roll / holding rule."""


@dataclass(frozen=True)
class RollRule:
    """Roll parameters (``configs/targets.yaml`` ``roll``).

    Attributes:
        roll_days: R - front contract must have last trade more than R sessions after t.
        exit_buffer_days: R' - held contract's last trade must be >= exit + R' sessions.
        held_respects_roll_rule: held contract is never nearer than the front contract under R.
    """

    roll_days: int = 5
    exit_buffer_days: int = 0
    held_respects_roll_rule: bool = True

    def __post_init__(self) -> None:
        if self.roll_days < 0 or self.exit_buffer_days < 0:
            raise ValueError("roll_days and exit_buffer_days must be >= 0")

    @classmethod
    def from_config(cls, config_dir: str | None = None, roll_days: int | None = None) -> RollRule:
        cfg = get(load_config("targets", config_dir), "roll")
        return cls(
            roll_days=int(cfg["R"] if roll_days is None else roll_days),
            exit_buffer_days=int(cfg.get("R_prime", 0)),
            held_respects_roll_rule=bool(cfg.get("held_respects_roll_rule", True)),
        )


# ---------------------------------------------------------------------------------------------
# Panels and lookups
# ---------------------------------------------------------------------------------------------


def settlement_panel(
    curve: pd.DataFrame,
    *,
    hub: str = "TTF",
    field: str = "settle",
    tenor_types: tuple[str, ...] | None = None,
    tz: str = MARKET_TZ,
) -> pd.DataFrame:
    """Wide settlement panel from a CURVE_SCHEMA frame (latest revision per contract and date)."""
    mask = (curve["hub"] == hub) & (curve["field"] == field)
    if tenor_types is not None:
        mask &= curve["tenor_type"].isin(tenor_types)
    df = curve.loc[mask, ["trade_date", "contract_id", "value", "revision", "publication_time"]]
    if df.empty:
        raise DataError(f"No {hub} {field} rows in curve frame")
    df = df.sort_values(["revision", "publication_time"]).drop_duplicates(
        ["trade_date", "contract_id"], keep="last"
    )
    local = df["trade_date"].dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    panel = df.assign(day=local).pivot(index="day", columns="contract_id", values="value")
    panel.index.name = "trade_date"
    return panel.sort_index().astype(float)


class SettlementLookup:
    """Fast ``(contract_id, day) -> price`` lookups on a wide panel; NaN when absent."""

    def __init__(self, panel: pd.DataFrame) -> None:
        self.panel = panel
        self._row = {d: i for i, d in enumerate(pd.DatetimeIndex(panel.index))}
        self._cols = {c: panel[c].to_numpy(dtype=float) for c in panel.columns}

    def get(self, cid: str, day: pd.Timestamp) -> float:
        i = self._row.get(day)
        col = self._cols.get(cid)
        if i is None or col is None:
            return float("nan")
        return float(col[i])

    @property
    def last_day(self) -> pd.Timestamp:
        return pd.Timestamp(self.panel.index.max())

    @property
    def first_day(self) -> pd.Timestamp:
        return pd.Timestamp(self.panel.index.min())


def safe_log(x: float | np.ndarray) -> float | np.ndarray:
    """Natural log; NaN passes through, a finite value <= 0 raises ``NonPositivePriceError``."""
    arr = np.asarray(x, dtype=float)
    bad = np.isfinite(arr) & (arr <= 0)
    if bad.any():
        raise NonPositivePriceError(
            f"log of non-positive price(s) {arr[bad][:5].tolist()}; use EUR/MWh differences"
        )
    with np.errstate(invalid="ignore"):
        out = np.log(arr)
    return float(out) if np.ndim(out) == 0 else out


# ---------------------------------------------------------------------------------------------
# Monthly chain: contracts sorted by delivery, with last-trade session positions
# ---------------------------------------------------------------------------------------------


class MonthlyChain:
    """Monthly contracts ordered by delivery with their last-trade positions on ``calendar``."""

    def __init__(self, last_trade: Mapping[str, DateLike] | pd.Series, calendar: TradingCalendar
                 ) -> None:
        items = dict(last_trade.items()) if isinstance(last_trade, pd.Series) else dict(last_trade)
        parsed = []
        for cid, ltd in items.items():
            c = parse_contract_id(str(cid))
            if c.tenor_type != "M":
                continue
            parsed.append((c.delivery_start, str(cid), to_local_day(ltd)))
        if not parsed:
            raise RollError("No monthly contracts in last-trade map")
        parsed.sort()
        self.calendar = calendar
        self.contract_ids: list[str] = [p[1] for p in parsed]
        self.last_trade: list[pd.Timestamp] = [p[2] for p in parsed]
        self.ltd_pos = np.array([calendar.position(d) for d in self.last_trade], dtype=int)
        if np.any(np.diff(self.ltd_pos) < 0):
            raise RollError("Last trading days are not monotone in delivery month")
        self.index_of = {cid: i for i, cid in enumerate(self.contract_ids)}

    def front_index(self, t_pos: int, roll_days: int) -> int | None:
        """First contract with ltd_pos - t_pos > roll_days, or None if beyond the chain."""
        i = int(np.searchsorted(self.ltd_pos, t_pos + roll_days, side="right"))
        return i if i < len(self.ltd_pos) else None

    def held_index(self, t_pos: int, h: int, rule: RollRule) -> int | None:
        """First contract trading on exit (ltd_pos >= t_pos + h + R'), not before the front."""
        i = int(np.searchsorted(self.ltd_pos, t_pos + h + rule.exit_buffer_days, side="left"))
        if rule.held_respects_roll_rule:
            f = self.front_index(t_pos, rule.roll_days)
            if f is None:
                return None
            i = max(i, f)
        return i if i < len(self.ltd_pos) else None


def front_contract(t: DateLike, chain: MonthlyChain, roll_days: int = 5) -> str:
    """Front monthly contract on ``t`` under roll rule R. Raises :class:`RollError` if none."""
    i = chain.front_index(chain.calendar.position(t), roll_days)
    if i is None:
        raise RollError(f"No listed contract is front on {to_local_day(t).date()} (R={roll_days})")
    return chain.contract_ids[i]


def held_contract(t: DateLike, h: int, chain: MonthlyChain, rule: RollRule = RollRule()) -> str:
    """Monthly contract held from ``t`` for ``h`` sessions; it still trades on the exit session."""
    if h < 1:
        raise ValueError("horizon h must be >= 1 session")
    i = chain.held_index(chain.calendar.position(t), h, rule)
    if i is None:
        raise RollError(f"No listed contract can be held from {to_local_day(t).date()} for h={h}")
    return chain.contract_ids[i]


def front_contract_series(days: pd.DatetimeIndex, chain: MonthlyChain, roll_days: int = 5
                          ) -> pd.Series:
    """Front contract id for every day (None where the chain is exhausted)."""
    pos = chain.calendar.positions(pd.DatetimeIndex(days))
    idx = np.searchsorted(chain.ltd_pos, pos + roll_days, side="right")
    ids = [chain.contract_ids[i] if i < len(chain.contract_ids) else None for i in idx]
    return pd.Series(ids, index=pd.DatetimeIndex(days), name="contract_id", dtype=object)


# ---------------------------------------------------------------------------------------------
# Continuous series (features only)
# ---------------------------------------------------------------------------------------------


def _sessions_of(panel: pd.DataFrame, calendar: TradingCalendar) -> pd.DatetimeIndex:
    return calendar.sessions_in(panel.index.min(), panel.index.max())


def continuous_level(panel: pd.DataFrame, chain: MonthlyChain, roll_days: int = 5
                     ) -> pd.DataFrame:
    """Unadjusted front-contract level per session: columns ``value``, ``contract_id``.

    Levels jump at rolls; use only for levels/spreads, never difference it into a return.
    """
    days = _sessions_of(panel, chain.calendar)
    front = front_contract_series(days, chain, roll_days)
    look = SettlementLookup(panel)
    values = [look.get(c, d) if c is not None else np.nan for d, c in front.items()]
    return pd.DataFrame({"value": np.asarray(values, dtype=float), "contract_id": front.values},
                        index=days)


def continuous_returns(panel: pd.DataFrame, chain: MonthlyChain, roll_days: int = 5
                       ) -> pd.DataFrame:
    """Spliced within-contract daily log returns: columns ``value``, ``contract_id``.

    Return on session ``t_i`` = ln F_c(t_i) - ln F_c(t_{i-1}) with c = front contract on
    ``t_{i-1}``. NaN if either settlement is missing. Equivalent to a ratio back-adjusted series.
    """
    days = _sessions_of(panel, chain.calendar)
    front = front_contract_series(days, chain, roll_days)
    look = SettlementLookup(panel)
    out = np.full(len(days), np.nan)
    held: list[Any] = [None] * len(days)
    for i in range(1, len(days)):
        c = front.iloc[i - 1]
        if c is None:
            continue
        held[i] = c
        p0, p1 = look.get(c, days[i - 1]), look.get(c, days[i])
        out[i] = safe_log(p1) - safe_log(p0)
    return pd.DataFrame({"value": out, "contract_id": held}, index=days)


def contract_log_returns(panel: pd.DataFrame, cid: str, calendar: TradingCalendar
                         ) -> pd.Series:
    """Daily log returns of one contract between consecutive calendar sessions (NaN on gaps)."""
    if cid not in panel.columns:
        raise DataError(f"{cid} not in settlement panel")
    days = _sessions_of(panel, calendar)
    prices = panel[cid].reindex(days).to_numpy(dtype=float)
    logs = safe_log(prices)
    r = np.full(len(days), np.nan)
    r[1:] = np.asarray(logs)[1:] - np.asarray(logs)[:-1]
    return pd.Series(r, index=days, name=cid)


def held_log_return(lookup: SettlementLookup, t: DateLike, h: int, chain: MonthlyChain,
                    rule: RollRule = RollRule()) -> tuple[float, str, pd.Timestamp]:
    """``r(t, h) = ln F_c(t+h) - ln F_c(t)`` on the held contract ``c``; returns (r, c, exit)."""
    day = to_local_day(t)
    c = held_contract(day, h, chain, rule)
    exit_day = chain.calendar.add_sessions(day, h)
    r = safe_log(lookup.get(c, exit_day)) - safe_log(lookup.get(c, day))
    return float(r), c, exit_day


# ---------------------------------------------------------------------------------------------
# Feature-side provider (SQ-06 / D-025): the canonical splice, consumed by features/prices.py
# ---------------------------------------------------------------------------------------------

#: Output columns required by ``features.prices.front_return_series`` (kept literal to avoid a
#: features import here; ``features.prices.SERIES_COLUMNS`` is the same tuple).
PROVIDER_SERIES_COLUMNS = ("date", "event_time", "available_time", "front_month", "front_settle",
                           "log_return")


def canonical_front_series(curves: pd.DataFrame, contracts: pd.DataFrame | None,
                           params: Mapping[str, Any]) -> pd.DataFrame:
    """``return_series_provider`` for the price family: this module's roll, not a second splice.

    Uses :class:`targets.common.CurveContext` (the same chain, roll rule ``R`` from
    ``targets.yaml`` and empirical session calendar from ``instruments.yaml`` as the targets), so
    features and labels cannot disagree about which contract is front on a session (SQ-06).
    Returns one row per OBSERVED trade date with ``PROVIDER_SERIES_COLUMNS``; ``front_settle`` is
    the unadjusted front level, ``log_return`` the within-contract return, ``front_month`` the
    delivery start of the front contract on that date.

    Params read: ``hub``, ``allowed_grades``, ``config_dir`` (optional; None -> ``configs/``).
    ``holidays`` and ``roll_offset_bdays`` in the price family are NOT used by this provider: the
    calendar and roll rule come from their configs, so there is one source for each.
    """
    from ttf_forecaster.targets.common import CurveContext  # lazy: targets imports this module

    hub = str(params.get("hub", "TTF"))
    config_dir = params.get("config_dir")
    allowed = params.get("allowed_grades")
    cur = curves
    if "hub" in cur.columns:
        cur = cur.loc[cur["hub"] == hub]
    if allowed is not None and "pit_grade" in cur.columns:
        cur = cur.loc[cur["pit_grade"].isin([str(g) for g in allowed])]
    if cur.empty:
        raise DataError(f"no {hub} settlements in bundle.curves for the roll splice")
    units = cur["unit"].dropna().unique().tolist() if "unit" in cur.columns else ["EUR/MWh"]
    if any(u != "EUR/MWh" for u in units):
        raise DataError(f"roll splice expects EUR/MWh settlements, found {units}")

    from ttf_forecaster.features.prices import prepare_curve  # lazy: features imports contracts

    # Availability and PIT grade come from the same final-vintage path as the reference splice,
    # so a revised settlement is not usable before its revision is published (test_prices).
    prep = prepare_curve(cur, hub, allowed_grades=None if allowed is None else
                         frozenset(str(g) for g in allowed))
    if prep.empty:
        raise DataError(f"no {hub} monthly settlements for the roll splice")
    observed = pd.DatetimeIndex(sorted(prep["date"].unique()))
    avail = prep.groupby("date")["available_time"].max()
    grades = [str(g) for g in prep["pit_grade"].dropna().unique()]

    ctx = CurveContext.from_curve(cur, metadata=None, hub=hub, config_dir=config_dir)
    days = pd.DatetimeIndex(ctx.calendar.sessions_in(ctx.panel.index.min(), ctx.panel.index.max()))
    lvl = continuous_level(ctx.panel, ctx.chain, ctx.rule.roll_days)
    ret = continuous_returns(ctx.panel, ctx.chain, ctx.rule.roll_days)
    frame = pd.DataFrame({"front": lvl["contract_id"].to_numpy(), "level": lvl["value"].to_numpy(),
                          "log_return": ret["value"].to_numpy()}, index=days)
    frame = frame.loc[frame.index.isin(observed)]  # observed trade dates only, as the reference
    delivery = [None if c is None else parse_contract_id(str(c)).delivery_start
                for c in frame["front"]]  # front on t (log_return uses the contract held into t)
    out = pd.DataFrame({
        "date": frame.index,
        "event_time": frame.index.tz_localize(MARKET_TZ).tz_convert("UTC"),
        "available_time": pd.to_datetime(avail.reindex(frame.index).to_numpy(), utc=True),
        "front_month": pd.to_datetime(delivery),
        "front_settle": frame["level"].to_numpy(),
        "log_return": frame["log_return"].to_numpy(),
    })
    out.attrs["pit_grade"] = worst_grade(grades).value if grades else PitGrade.A.value
    return out[list(PROVIDER_SERIES_COLUMNS)]
