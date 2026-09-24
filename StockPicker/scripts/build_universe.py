"""Build the point-in-time S&P 500 membership panel and static metadata.

Run once, then re-run monthly to extend. Safe to interrupt -- snapshots are
cached incrementally, so a re-run only pulls what is missing.

    python -m StockPicker.scripts.build_universe
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from ..bbg.session import BloombergSession
from ..bbg.universe import all_securities, build_membership, fetch_metadata
from ..config import HISTORY_START, SECTOR_COL


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("build_universe")

    with BloombergSession() as bbg:
        membership = build_membership(bbg, start=HISTORY_START)
        securities = all_securities(membership)
        log.info("Universe spans %s unique securities.", len(securities))

        meta = fetch_metadata(bbg, securities)
        log.info("Metadata rows: %s", len(meta))
        log.info("Bloomberg usage: %s", bbg.stats.summary())

    # ---- diagnostics: prove the survivorship-bias fix is actually working ----
    counts = membership.groupby("date").size()
    log.info("Snapshots: %s (%s -> %s)", len(counts), counts.index.min().date(), counts.index.max().date())
    log.info("Members per snapshot: min=%s median=%s max=%s", counts.min(), int(counts.median()), counts.max())

    current = set(membership.loc[membership["date"] == membership["date"].max(), "security"])
    ever = set(membership["security"])
    log.info(
        "Names in index today: %s | names ever in index: %s | churn (dropped out): %s",
        len(current),
        len(ever),
        len(ever - current),
    )

    if SECTOR_COL in meta.columns:
        missing = meta[SECTOR_COL].isna().sum()
        log.info("Sector coverage: %s/%s populated (%s missing)", len(meta) - missing, len(meta), missing)
        log.info("Sector breakdown:\n%s", meta[SECTOR_COL].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
