"""Bloomberg: futures settles and the roll calendar.

Needs xbbg against a logged-in Terminal on the same machine, so the import is
deferred - constructing a fetcher is free, calling ``fetch`` is what needs the
Terminal. A missing xbbg raises rather than degrading to a stub.

TWO THINGS THIS MODULE EXISTS TO GET RIGHT.

First, an unverified root is refused. `targets.yaml` carries a `verified` flag
per contract and only NG is set, because only NG has been pulled from a
Terminal in this repo (``Utils/pull_daily_data_bbg.py``). Pulling ``CA1
Comdty`` and hoping it is MATIF wheat is how a model ends up fitted to the
wrong asset, so it raises with the exact check to run instead.

Second, the roll calendar is OBSERVED, not assumed. ``blp.fut_ticker`` answers
"which contract was front on this date", which is the only honest way to build
history: exchange calendars move, holidays shift expiries, and a rule that is
right 95% of the time puts a fake return in the series twelve times a year.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from wxreturns.config import get
from wxreturns.data.base import Fetcher, FetchResult, SourceUnavailable

log = logging.getLogger(__name__)


def _blp():
    """Import xbbg on demand, with an actionable message when it is absent."""
    try:
        from xbbg import blp
    except ImportError as exc:
        raise SourceUnavailable(
            "xbbg is not importable, so no Bloomberg series can be fetched. "
            f"Install the pinned version ({get('sources', 'bloomberg', 'pinned_version')}) "
            "and run from a machine with a logged-in Terminal. No substitute "
            "prices are produced."
        ) from exc
    return blp


def contract_config(commodity: str) -> dict:
    contracts = get("targets", "contracts")
    if commodity not in contracts:
        raise KeyError(
            f"unknown commodity {commodity!r}; targets.yaml defines "
            f"{sorted(contracts)}")
    return contracts[commodity]


def require_verified(commodity: str) -> dict:
    """Refuse a root nobody has confirmed against a Terminal."""
    cfg = contract_config(commodity)
    if not cfg.get("verified", False):
        root, key = cfg["bloomberg_root"], cfg["yellow_key"]
        raise SourceUnavailable(
            f"{commodity} root {root!r} is marked verified: false in "
            f"targets.yaml, so it will not be pulled. Confirm it first:\n\n"
            f"    blp.bdp(['{root}1 {key}'], ['name', 'fut_gen_month'])\n\n"
            f"then set verified: true. Pulling an unconfirmed root risks "
            f"fitting the model to a different contract entirely."
        )
    return cfg


def generic_ticker(commodity: str, generic: int) -> str:
    cfg = contract_config(commodity)
    return f"{cfg['bloomberg_root']}{generic} {cfg['yellow_key']}"


class SettleFetcher(Fetcher):
    """Daily settles for the generic slots named in targets.yaml.

    Generics are pulled because they are what has continuous history. They are
    NOT a return series - the roll gap has to be removed first, which is what
    ``market/contracts.py`` uses the roll calendar for.
    """

    source = "bloomberg_settles"

    def fetch(self, commodity: str = "", start: date | None = None,
              end: date | None = None, generics: list[int] | None = None
              ) -> FetchResult:
        if not commodity:
            raise ValueError("commodity is required")
        require_verified(commodity)
        blp = _blp()

        if generics is None:
            slots = get("targets", "slots")
            generics = sorted(slot["generic"] for slot in slots.values())
        tickers = [generic_ticker(commodity, g) for g in generics]
        fields = [get("sources", "bloomberg", "settle_field"),
                  get("sources", "bloomberg", "volume_field")]

        frame = blp.bdh(tickers, fields, start, end)
        if frame.empty:
            raise SourceUnavailable(
                f"Bloomberg returned no rows for {tickers} over {start}..{end}")

        tidy = (frame.stack(level=0, future_stack=True)
                .rename_axis(index=["obs_date", "ticker"])
                .reset_index())
        lookup = {generic_ticker(commodity, g): g for g in generics}
        tidy["generic"] = tidy["ticker"].map(lookup)
        tidy["commodity"] = commodity
        # xbbg lowercases field names on the way back; normalise so downstream
        # code never has to know that.
        tidy = tidy.rename(columns={c: c.lower() for c in tidy.columns})
        return self._result(tidy, f"{self.source}:{commodity}")


class RollCalendarFetcher(Fetcher):
    """Which contract was actually front (or front+1) on each date.

    ``blp.fut_ticker(generic, dt, freq)`` is the primitive. It is one call per
    date, so this is slow over years of history - it is meant to be run once
    per commodity and cached, and to CROSS-CHECK a constructed calendar rather
    than to be queried in a loop at fit time.
    """

    source = "bloomberg_roll_calendar"

    def fetch(self, commodity: str = "", dates: list[date] | None = None,
              generic: int = 1) -> FetchResult:
        if not commodity:
            raise ValueError("commodity is required")
        if not dates:
            raise ValueError("an explicit list of dates is required")
        require_verified(commodity)
        blp = _blp()

        ticker = generic_ticker(commodity, generic)
        freq = get("sources", "bloomberg", "roll_resolution", "frequency")

        records: list[dict] = []
        unresolved: list[date] = []
        for when in dates:
            resolved = blp.fut_ticker(ticker, dt=when, freq=freq)
            if not resolved:
                unresolved.append(when)
                continue
            records.append({"obs_date": pd.Timestamp(when),
                            "commodity": commodity,
                            "generic": generic,
                            "contract": resolved})

        if unresolved:
            log.warning("%s: fut_ticker did not resolve %d of %d dates (first: %s)",
                        commodity, len(unresolved), len(dates), unresolved[0])
        if not records:
            raise SourceUnavailable(
                f"fut_ticker resolved no contract for {ticker} on any of the "
                f"{len(dates)} dates requested")
        return self._result(pd.DataFrame.from_records(records),
                            f"{self.source}:{commodity}")
