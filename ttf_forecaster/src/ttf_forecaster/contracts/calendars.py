"""Trading calendar for futures settlement dates and business-day arithmetic.

``docs/target_definitions.md`` §0: historically, the sessions are the **empirical** set of dates on
which a TTF monthly contract settled (point-in-time safe: a date is a session because a settlement
was published on it). Before the first and after the last observed date, sessions are weekdays
minus a **configured** holiday list (``configs/instruments.yaml`` ``calendars``), whose status is
``requires_confirmation`` until checked against the ICE Endex calendar.

All dates handled here are naive, normalised **local** dates (Europe/Amsterdam trading dates).
tz-aware inputs are converted to the market zone first; nothing is assumed about naive datetimes
other than that they already are local dates.

Caveat of the empirical rule: a date missing from the vendor history for *every* contract is
indistinguishable from a holiday. A gap for only some contracts is still a missing settlement.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import MARKET_TZ, DataError
from ttf_forecaster.settings import get, load_config


class CalendarError(DataError):
    """A date is outside the calendar's range or the calendar is malformed."""


DateLike = pd.Timestamp | str | np.datetime64 | Any


def to_local_day(value: DateLike, tz: str = MARKET_TZ) -> pd.Timestamp:
    """Normalise ``value`` to a naive local date (midnight) in ``tz``.

    tz-aware timestamps are converted to ``tz`` before the time is dropped, so 2026-03-30 23:30 UTC
    becomes 2026-03-31 in Amsterdam. Naive values are taken to be local already.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(tz).tz_localize(None)
    return ts.normalize()


def to_local_days(values: Iterable[DateLike], tz: str = MARKET_TZ) -> pd.DatetimeIndex:
    """Vectorised :func:`to_local_day`; returns a sorted unique naive DatetimeIndex."""
    if not isinstance(values, (pd.Index, pd.Series, np.ndarray)):
        values = list(values)
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert(tz).tz_localize(None)
    return pd.DatetimeIndex(idx.normalize().unique()).sort_values()


class TradingCalendar:
    """Sessions (trading dates) with business-day arithmetic.

    Args:
        observed_sessions: dates with an observed settlement (empirical part). May be empty.
        future_holidays: configured non-trading weekdays used outside the observed range.
        weekend: pandas ``dayofweek`` values that are never sessions under the rule part.
        rule_start: first date of the rule-based part before the observed range.
        horizon_end: last date the calendar extends to (expiry projection).
        observed_until: last date covered by the empirical part (defaults to the last observed
            session). Dates after it are rule-based.
        status: confirmation status of the configured holiday list (reported, not enforced).
    """

    def __init__(
        self,
        observed_sessions: Iterable[DateLike] = (),
        future_holidays: Iterable[DateLike] = (),
        *,
        weekend: Sequence[int] = (5, 6),
        rule_start: DateLike = "2000-01-03",
        horizon_end: DateLike = "2045-12-29",
        observed_until: DateLike | None = None,
        status: str = "requires_confirmation",
    ) -> None:
        observed = to_local_days(observed_sessions)
        holidays = to_local_days(future_holidays)
        start = to_local_day(rule_start)
        end = to_local_day(horizon_end)
        if len(observed):
            start = min(start, observed[0])
            end = max(end, observed[-1])
        if end <= start:
            raise CalendarError(f"horizon_end {end.date()} must be after rule_start {start.date()}")
        all_days = pd.date_range(start, end, freq="D")
        rule_days = all_days[~all_days.dayofweek.isin(list(weekend)) & ~all_days.isin(holidays)]
        if len(observed):
            obs_start = observed[0]
            obs_end = to_local_day(observed_until) if observed_until is not None else observed[-1]
            if obs_end < observed[-1]:
                raise CalendarError("observed_until is before the last observed session")
            outside = rule_days[(rule_days < obs_start) | (rule_days > obs_end)]
            sessions = outside.union(observed)
        else:
            obs_start = obs_end = None
            sessions = rule_days
        self._sessions: pd.DatetimeIndex = pd.DatetimeIndex(sessions).sort_values()
        self._values = self._sessions.values  # datetime64[ns], sorted
        self.observed_start: pd.Timestamp | None = obs_start
        self.observed_end: pd.Timestamp | None = obs_end
        self.future_holidays = holidays
        self.weekend = tuple(weekend)
        self.status = status

    # ---------------------------------------------------------------- construction helpers

    @classmethod
    def from_config(
        cls,
        observed_sessions: Iterable[DateLike] = (),
        *,
        hub: str = "TTF",
        config_dir: str | None = None,
        observed_until: DateLike | None = None,
    ) -> TradingCalendar:
        """Calendar for ``hub`` with parameters from ``configs/instruments.yaml``."""
        cfg = get(load_config("instruments", config_dir), "calendars", hub)
        return cls(
            observed_sessions,
            cfg.get("future_holidays") or (),
            weekend=tuple(cfg.get("weekend", (5, 6))),
            rule_start=cfg.get("rule_start", "2000-01-03"),
            horizon_end=cfg.get("horizon_end", "2045-12-29"),
            observed_until=observed_until,
            status=str(cfg.get("status", "requires_confirmation")),
        )

    @classmethod
    def from_curve(
        cls,
        curve: pd.DataFrame,
        *,
        hub: str = "TTF",
        tenor_type: str = "M",
        field: str = "settle",
        config_dir: str | None = None,
    ) -> TradingCalendar:
        """Empirical calendar: every local trade date with a non-null ``field`` on a ``tenor_type``
        contract of ``hub`` in a CURVE_SCHEMA frame, plus the configured rule outside that range."""
        mask = (curve["hub"] == hub) & (curve["tenor_type"] == tenor_type) & (
            curve["field"] == field
        ) & curve["value"].notna()
        days = to_local_days(curve.loc[mask, "trade_date"])
        if not len(days):
            raise CalendarError(f"No {hub} {tenor_type} {field} rows to build an empirical calendar")
        return cls.from_config(days, hub=hub, config_dir=config_dir)

    def as_known_at(self, as_of: DateLike) -> TradingCalendar:
        """The calendar as it could be known on ``as_of``: empirical sessions up to ``as_of``,
        configured rule afterwards. Use when the calendar itself feeds a feature."""
        cutoff = to_local_day(as_of)
        if self.observed_start is None or self.observed_end is None or cutoff < self.observed_start:
            observed = pd.DatetimeIndex([])
            until = None
        else:
            until = min(cutoff, self.observed_end)
            observed = self._sessions[
                (self._sessions >= self.observed_start) & (self._sessions <= until)
            ]
        return TradingCalendar(
            observed,
            self.future_holidays,
            weekend=self.weekend,
            rule_start=self._sessions[0],
            horizon_end=self._sessions[-1],
            observed_until=until,
            status=self.status,
        )

    # ---------------------------------------------------------------- queries

    @property
    def sessions(self) -> pd.DatetimeIndex:
        """All sessions, sorted."""
        return self._sessions

    def sessions_in(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        """Sessions in the closed interval [start, end]."""
        s, e = to_local_day(start), to_local_day(end)
        return self._sessions[(self._sessions >= s) & (self._sessions <= e)]

    def is_session(self, day: DateLike) -> bool:
        d = to_local_day(day)
        i = int(np.searchsorted(self._values, d.to_datetime64(), side="left"))
        return i < len(self._values) and self._values[i] == d.to_datetime64()

    def _check(self, i: int, day: pd.Timestamp) -> int:
        if i < 0 or i >= len(self._values):
            raise CalendarError(
                f"{day.date()} is outside the calendar range "
                f"{self._sessions[0].date()}..{self._sessions[-1].date()}"
            )
        return i

    def position(self, day: DateLike) -> int:
        """Index of the last session on or before ``day``."""
        d = to_local_day(day)
        i = int(np.searchsorted(self._values, d.to_datetime64(), side="right")) - 1
        return self._check(i, d)

    def positions(self, days: pd.DatetimeIndex) -> np.ndarray:
        """Vectorised :meth:`position` for naive local dates."""
        idx = np.searchsorted(self._values, days.values, side="right") - 1
        if len(idx) and (idx.min() < 0):
            raise CalendarError("dates before the calendar start")
        return idx

    def session_at(self, i: int) -> pd.Timestamp:
        """The session with index ``i``."""
        if i < 0 or i >= len(self._values):
            raise CalendarError(f"session index {i} outside 0..{len(self._values) - 1}")
        return self._sessions[i]

    def add_sessions(self, day: DateLike, n: int) -> pd.Timestamp:
        """Move ``n`` sessions from ``day`` (negative = backwards).

        For a non-session ``day`` the count starts from the adjacent session in the direction of
        travel: Saturday + 1 = Monday, Saturday - 1 = Friday (no holidays). ``n = 0`` on a
        non-session returns the previous session.
        """
        d = to_local_day(day)
        if n >= 0:
            base = int(np.searchsorted(self._values, d.to_datetime64(), side="right")) - 1
        else:
            base = int(np.searchsorted(self._values, d.to_datetime64(), side="left"))
        return self._sessions[self._check(base + n, d)]

    def sessions_between(self, start: DateLike, end: DateLike) -> int:
        """Number of sessions in (start, end]; negative if ``end`` precedes ``start``."""
        return self.position(end) - self.position(start)

    def next_session(self, day: DateLike) -> pd.Timestamp:
        """First session strictly after ``day``."""
        d = to_local_day(day)
        i = int(np.searchsorted(self._values, d.to_datetime64(), side="right"))
        return self._sessions[self._check(i, d)]

    def previous_session(self, day: DateLike) -> pd.Timestamp:
        """Last session strictly before ``day``."""
        d = to_local_day(day)
        i = int(np.searchsorted(self._values, d.to_datetime64(), side="left")) - 1
        return self._sessions[self._check(i, d)]

    def __repr__(self) -> str:
        obs = (
            f"{self.observed_start.date()}..{self.observed_end.date()}"
            if self.observed_start is not None and self.observed_end is not None
            else "none"
        )
        return (
            f"TradingCalendar(sessions={len(self._sessions)}, observed={obs}, "
            f"holidays={len(self.future_holidays)}, status={self.status})"
        )
