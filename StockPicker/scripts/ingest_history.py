"""Pull and cache the full price + fundamental history for the universe.

Incremental and interruptible: every security is cached to its own parquet as
soon as it arrives, so a crash or a Ctrl-C costs only the current batch.

    python -m StockPicker.scripts.ingest_history            # everything
    python -m StockPicker.scripts.ingest_history --limit 50 # pilot run
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from ..bbg.session import BloombergSession
from ..bbg.universe import all_securities
from ..config import (
    BENCHMARK_TICKER,
    FUNDA_DIR,
    HISTORY_START,
    INDEX_TICKER,
    PRICES_DIR,
    PRICE_FIELDS,
    SLOW_FIELDS,
)
from ..data.store import fetch_series

MEMBERSHIP_PATH = PRICES_DIR.parent / "universe" / "membership.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="pilot: only N securities")
    ap.add_argument("--prices-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    log = logging.getLogger("ingest")

    if not MEMBERSHIP_PATH.exists():
        log.error("No membership panel. Run: python -m StockPicker.scripts.build_universe")
        return 1

    membership = pd.read_parquet(MEMBERSHIP_PATH)
    membership["date"] = pd.to_datetime(membership["date"])
    securities = all_securities(membership)

    if args.limit:
        # Prefer currently-live names for a pilot so results are inspectable.
        latest = set(membership.loc[membership["date"] == membership["date"].max(), "security"])
        live = [s for s in securities if s in latest]
        securities = live[: args.limit]

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    log.info("Ingesting %s securities | %s -> %s", len(securities), HISTORY_START, end.date())

    with BloombergSession(timeout_ms=60_000) as bbg:
        # Benchmark and index first -- every residual/beta feature needs them.
        log.info("--- benchmark series ---")
        fetch_series(
            bbg, [BENCHMARK_TICKER, INDEX_TICKER], PRICE_FIELDS,
            HISTORY_START, end, PRICES_DIR, kind="price", force=args.force,
        )

        log.info("--- daily prices ---")
        fetch_series(
            bbg, securities, PRICE_FIELDS, HISTORY_START, end,
            PRICES_DIR, kind="price", batch_size=25, force=args.force,
        )

        if not args.prices_only:
            log.info("--- weekly fundamentals / estimates / sentiment ---")
            fetch_series(
                bbg, securities, SLOW_FIELDS, HISTORY_START, end,
                FUNDA_DIR, kind="funda", periodicity="WEEKLY",
                batch_size=15, force=args.force, scale_market_cap=False,
            )

        log.info("Bloomberg usage: %s", bbg.stats.summary())
        if bbg.stats.errors:
            log.info("First 10 of %s errors:", len(bbg.stats.errors))
            for e in bbg.stats.errors[:10]:
                log.info("   %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
