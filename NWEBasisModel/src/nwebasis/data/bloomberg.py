"""Bloomberg: TTF and JKM curves, EU storage, ALSI terminal sendout.

Needs xbbg against a logged-in Terminal on the same machine. That is available
on the desk box and not in every environment this package runs in, so the
import is deferred: constructing a fetcher is free, calling ``fetch`` is what
requires the Terminal. A missing xbbg raises rather than degrading to a stub.

Ticker provenance is in ``config/sources.yaml``. The storage family (CGIE*) was
verified against real values in GasImbalanceModel on 2026-08-25 and is reused
unchanged. The JKM root is NOT verified for history depth - see the note on
that config entry before trusting a long backfill.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from nwebasis.config import get
from nwebasis.data.base import Fetcher, FetchResult, SourceUnavailable

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


class StorageFetcher(Fetcher):
    """EU aggregate and NWE country gas storage, from the CGIE* family.

    Inventory and capacity come back in TWh, injection and withdrawal in GWh/d,
    per the Bloomberg security descriptions themselves. The unit is carried on
    every row rather than assumed downstream.
    """

    source = "bloomberg_storage"

    def fetch(self, start: date | None = None, end: date | None = None) -> FetchResult:
        blp = _blp()
        agg = get("sources", "bloomberg", "storage", "aggregate")
        field = get("sources", "bloomberg", "settle_field")
        frame = blp.bdh(list(agg.values()), field, start, end)
        if frame.empty:
            raise SourceUnavailable("Bloomberg returned no storage rows for the window")
        tidy = (frame.stack(level=0, future_stack=True)
                .rename_axis(index=["obs_date", "ticker"])
                .reset_index())
        inverse = {ticker: name for name, ticker in agg.items()}
        tidy["series_id"] = tidy["ticker"].map(inverse)
        return self._result(tidy, self.source)


class CurveFetcher(Fetcher):
    """A monthly forward curve - TTF or JKM - as settlements by delivery month.

    Ticker construction follows ``Data/lng_report/bloomberg.py``: root plus the
    Bloomberg month code plus a two-digit year. That logic already runs in
    production for this desk, so it is reused in shape rather than reinvented.
    """

    source = "bloomberg_curve"

    def fetch(self, curve: str = "ttf", start: date | None = None,
              end: date | None = None, tickers: list[str] | None = None) -> FetchResult:
        blp = _blp()
        cfg = get("sources", "bloomberg", "curves", curve)
        if tickers is None:
            raise SourceUnavailable(
                f"no explicit ticker list given for {curve!r}. Build the strip with "
                f"root {cfg.get('monthly_root_usd') or cfg.get('monthly_root')!r} using "
                "Data.lng_report.bloomberg.BloombergExtractor, or pass tickers=[...]. "
                "This fetcher will not guess a strip."
            )
        field = get("sources", "bloomberg", "settle_field")
        frame = blp.bdh(tickers, field, start, end)
        if frame.empty:
            raise SourceUnavailable(f"Bloomberg returned no rows for the {curve} strip")
        tidy = (frame.stack(level=0, future_stack=True)
                .rename_axis(index=["obs_date", "ticker"])
                .reset_index())
        tidy["curve"] = curve
        return self._result(tidy, f"{self.source}:{curve}")
