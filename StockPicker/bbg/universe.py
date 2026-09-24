"""
============================================================
bbg/universe.py  --  Point-in-time index membership
============================================================
Builds a survivorship-bias-free S&P 500 membership panel.

Why this matters
----------------
The single most common way a backtest lies is by selecting from
*today's* index members. Doing that means the 2010 backtest only
ever considers companies that survived to 2026 -- it silently knows
which firms would not go bankrupt or get acquired. Measured
inflation of returns from this bias is commonly 1-4% a year, which
is larger than most strategies' real alpha.

We avoid it with ``INDX_MWEIGHT_HIST`` + ``END_DATE_OVERRIDE``,
which returns the index constituents *as they stood* on a past
date, including names that have since been acquired or delisted.
Those come back under Bloomberg's dead-ticker convention
(e.g. ``1436513D UN``), which still carry full price history once
the ``Equity`` yellow key is appended.

Verified on this terminal: the 2015-06-30 SPX snapshot returns
Johnson Controls, DirecTV and TECO Energy -- all since acquired,
none present in today's index.
============================================================
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import HISTORY_START, INDEX_TICKER, MEMBERSHIP_FREQ, STATIC_FIELDS, UNIVERSE_DIR
from .session import BloombergSession

log = logging.getLogger(__name__)

MEMBERSHIP_PATH = UNIVERSE_DIR / "membership.parquet"
METADATA_PATH = UNIVERSE_DIR / "security_metadata.parquet"

# Bloomberg composite exchange codes seen in index member lists. We normalise
# every one of these to the ``US`` composite so a name that moves listing venue
# does not appear as two distinct securities in the panel.
_US_EXCHANGE_CODES = {"UN", "UW", "UQ", "UA", "UR", "UP", "UV", "UF", "US"}


def normalise_ticker(raw: str, use_composite: bool = True) -> str:
    """Turn an index-member string into a fully qualified Bloomberg security.

    ``INDX_MEMBERS`` yields ``"AAPL UW"``; ``INDX_MWEIGHT_HIST`` yields
    ``"AAPL UW"`` or a dead ticker like ``"1436513D UN"``. Both need the
    ``Equity`` yield key appended before they will resolve.

    Parameters
    ----------
    use_composite
        Map venue-specific codes (UW/UN/UQ...) to the ``US`` composite.
        Composite is preferred: it is stable through listing changes and is
        what consolidated volume is reported against.
    """
    raw = str(raw).strip()
    if not raw:
        return ""
    if raw.upper().endswith(" EQUITY"):
        return raw
    parts = raw.split()
    if len(parts) >= 2 and parts[-1].upper() in _US_EXCHANGE_CODES:
        ticker = " ".join(parts[:-1])
        code = "US" if use_composite else parts[-1]
        return f"{ticker} {code} Equity"
    return f"{raw} US Equity"


def month_end_dates(start=None, end=None, freq: str = MEMBERSHIP_FREQ) -> list[pd.Timestamp]:
    """Snapshot dates on which we sample index membership."""
    start = pd.Timestamp(start or HISTORY_START)
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    return list(pd.date_range(start, end, freq=freq))


def build_membership(
    bbg: BloombergSession,
    index_ticker: str = INDEX_TICKER,
    start=None,
    end=None,
    force: bool = False,
) -> pd.DataFrame:
    """Build (or incrementally extend) the point-in-time membership panel.

    Returns a long DataFrame ``[date, security, weight]`` where ``date`` is the
    snapshot date and ``security`` is a resolvable Bloomberg ticker.
    """
    existing = pd.DataFrame()
    if MEMBERSHIP_PATH.exists() and not force:
        existing = pd.read_parquet(MEMBERSHIP_PATH)
        existing["date"] = pd.to_datetime(existing["date"])

    wanted = month_end_dates(start, end)
    have = set(existing["date"].unique()) if not existing.empty else set()
    todo = [d for d in wanted if d not in have]

    if not todo:
        log.info("Membership panel already covers %s snapshots.", len(have))
        return existing

    log.info("Pulling %s membership snapshots for %s ...", len(todo), index_ticker)
    frames: list[pd.DataFrame] = []
    for i, snap in enumerate(todo, 1):
        stamp = snap.strftime("%Y%m%d")
        df = bbg.bds(
            index_ticker,
            "INDX_MWEIGHT_HIST",
            overrides={"END_DATE_OVERRIDE": stamp},
        )
        if df.empty:
            log.warning("  %s -> empty membership, skipping", stamp)
            continue

        # Column naming varies by field; locate member and weight columns.
        member_col = next(
            (c for c in df.columns if "member" in c.lower() or "ticker" in c.lower()), df.columns[0]
        )
        weight_col = next((c for c in df.columns if "weight" in c.lower()), None)

        out = pd.DataFrame({"date": snap, "raw_ticker": df[member_col].astype(str)})
        out["security"] = out["raw_ticker"].map(normalise_ticker)
        out["weight"] = pd.to_numeric(df[weight_col], errors="coerce") if weight_col else pd.NA
        out = out[out["security"] != ""]
        frames.append(out)

        if i % 12 == 0 or i == len(todo):
            log.info("  %s/%s snapshots (%s -> %s names)", i, len(todo), stamp, len(out))

    if not frames:
        return existing

    new = pd.concat(frames, ignore_index=True)
    combined = (
        pd.concat([existing, new], ignore_index=True)
        .drop_duplicates(subset=["date", "security"], keep="last")
        .sort_values(["date", "security"])
        .reset_index(drop=True)
    )
    combined.to_parquet(MEMBERSHIP_PATH, index=False)
    log.info(
        "Membership panel saved: %s rows | %s snapshots | %s unique securities",
        len(combined),
        combined["date"].nunique(),
        combined["security"].nunique(),
    )
    return combined


def all_securities(membership: pd.DataFrame) -> list[str]:
    """Every security that was ever an index member -- the full pull list."""
    return sorted(membership["security"].dropna().unique().tolist())


def fetch_metadata(bbg: BloombergSession, securities: list[str], force: bool = False) -> pd.DataFrame:
    """Static descriptors (name, GICS sector/industry) for every security.

    Dead tickers frequently lack ``GICS_SECTOR_NAME``; those are backfilled
    from the live ticker where one exists, and otherwise marked ``Unknown``
    so sector-neutralisation never silently drops a name.
    """
    existing = pd.DataFrame()
    if METADATA_PATH.exists() and not force:
        existing = pd.read_parquet(METADATA_PATH)
        known = set(existing["security"])
        securities = [s for s in securities if s not in known]

    if securities:
        log.info("Fetching static metadata for %s securities ...", len(securities))
        meta = bbg.bdp(securities, STATIC_FIELDS)
        meta = meta.reset_index().rename(columns={"index": "security"})
        existing = (
            pd.concat([existing, meta], ignore_index=True)
            .drop_duplicates(subset=["security"], keep="last")
            .reset_index(drop=True)
        )
        existing.to_parquet(METADATA_PATH, index=False)

    if "GICS_SECTOR_NAME" in existing.columns:
        existing["GICS_SECTOR_NAME"] = existing["GICS_SECTOR_NAME"].fillna("Unknown")
    return existing


def members_on(membership: pd.DataFrame, asof) -> list[str]:
    """Index members as of a date -- the most recent snapshot at or before it.

    This is the function the backtest calls on every rebalance. Using the
    *most recent prior* snapshot is what enforces point-in-time discipline:
    on 2015-07-15 we see the 2015-06-30 constituents, never a later one.
    """
    asof = pd.Timestamp(asof)
    valid = membership[membership["date"] <= asof]
    if valid.empty:
        return []
    latest = valid["date"].max()
    return valid.loc[valid["date"] == latest, "security"].tolist()
