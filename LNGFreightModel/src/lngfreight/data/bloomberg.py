"""Bloomberg: the price complex, LNG on water, routing and US exports.

Needs xbbg against a logged-in Terminal. Constructing a fetcher is free;
calling ``fetch`` is what needs the Terminal, so the import is deferred and a
missing xbbg raises rather than degrading to a stub.

Every ticker used here was executed from this machine on 2026-09-17 and its
description checked - see config/sources.yaml, where the two unentitled
families are also recorded so they are not re-derived later.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from lngfreight.config import get
from lngfreight.data.base import Fetcher, FetchResult, SourceUnavailable

log = logging.getLogger(__name__)


def _blp():
    try:
        from xbbg import blp
    except ImportError as exc:
        raise SourceUnavailable(
            "xbbg is not importable, so no Bloomberg series can be fetched. "
            f"Run from {get('sources', 'bloomberg', 'environment')}, which has "
            "it against a logged-in Terminal. No substitute data is produced."
        ) from exc
    return blp


def _tidy(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Wide xbbg frame -> long (obs_date, series_id, value).

    xbbg returns a two-level column index of (ticker, field) and lowercases
    the field name. Only the ticker level is needed here; the field is fixed
    by the caller.
    """
    flat = frame.copy()
    if isinstance(flat.columns, pd.MultiIndex):
        flat.columns = [c[0] for c in flat.columns]
    inverse = {ticker: name for name, ticker in mapping.items()}
    missing = [t for t in mapping.values() if t not in flat.columns]
    if missing:
        # An entitlement wall looks exactly like this: the ticker resolves but
        # never appears in the payload. Name it rather than dropping it.
        log.warning("no data returned for %s", ", ".join(missing))
    flat = flat.rename(columns=inverse)
    keep = [c for c in flat.columns if c in inverse.values()]
    if not keep:
        raise SourceUnavailable(
            f"Bloomberg returned none of the requested series: {sorted(mapping)}"
        )
    long = (flat[keep]
            .rename_axis(index="obs_date")
            .reset_index()
            .melt(id_vars="obs_date", var_name="series_id", value_name="value")
            .dropna(subset=["value"]))
    long["obs_date"] = pd.to_datetime(long["obs_date"])
    return long.sort_values(["series_id", "obs_date"]).reset_index(drop=True)


class BloombergFetcher(Fetcher):
    """Pulls one configured block of sources.yaml as a long frame.

    ``block`` is a path under ``bloomberg``: ("prices",) or
    ("on_water", "series"). Each block carries its own publication lag, which
    is attached to every row here rather than being re-looked-up downstream -
    the lag travels with the data so it cannot be forgotten.
    """

    source = "bloomberg"

    def __init__(self, block: tuple[str, ...], lag_path: tuple[str, ...]) -> None:
        self.block = block
        self.lag_path = lag_path

    def _mapping(self) -> dict[str, str]:
        node = get("sources", "bloomberg", *self.block)
        return {name: spec["ticker"] for name, spec in node.items()}

    def _lag(self) -> int:
        node = get("sources", "bloomberg", *self.lag_path)
        if isinstance(node, dict):
            # a per-series block: every series shares the block-level lag
            return int(node["publication_lag_days"])
        return int(node)

    def fetch(self, start: date | None = None, end: date | None = None) -> FetchResult:
        blp = _blp()
        mapping = self._mapping()
        field = get("sources", "bloomberg", "settle_field")
        frame = blp.bdh(list(mapping.values()), field, start, end)
        if frame.empty:
            raise SourceUnavailable(
                f"Bloomberg returned no rows for {'.'.join(self.block)} "
                f"over {start}..{end}"
            )
        long = _tidy(frame, mapping)
        long["publication_lag_days"] = self._lag()
        long["block"] = ".".join(self.block)
        return self._result(long, f"{self.source}:{'.'.join(self.block)}")


def price_fetcher() -> BloombergFetcher:
    """HH, TTF (USD and EUR), JKM. Per-ticker lag, all zero: they settle daily."""
    return BloombergFetcher(("prices",), ("prices", "henry_hub"))


def freight_futures_fetcher() -> BloombergFetcher:
    """Baltic BLNG1/2/3 futures. Cross-check only - history starts 2024-02."""
    return BloombergFetcher(("freight_futures",), ("freight_futures", "blng1_aus_jpn"))


def on_water_fetcher() -> BloombergFetcher:
    """LNG afloat by age and origin. Daily from 2017, one-day publication lag."""
    return BloombergFetcher(("on_water", "series"), ("on_water",))


def routing_fetcher() -> BloombergFetcher:
    """Laden transits per chokepoint. Two-day lag - the slowest feed here."""
    return BloombergFetcher(("routing", "series"), ("routing",))


def supply_fetcher() -> BloombergFetcher:
    """US daily LNG liftings."""
    return BloombergFetcher(("supply", "series"), ("supply",))


def flows_fetcher() -> BloombergFetcher:
    """Daily LNG tons by origin-destination leg, plus global export/import totals.

    The tonne-mile family is built on this: a ton to North Asia occupies a
    ship roughly three times as long as a ton to NWE, so the mix of these
    legs is what sets the ship-days the market is consuming.
    """
    return BloombergFetcher(("flows", "series"), ("flows",))
