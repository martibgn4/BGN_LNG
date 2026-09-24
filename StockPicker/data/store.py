"""
============================================================
data/store.py  --  Cached, incremental market-data store
============================================================
Bloomberg Desktop API is a metered, rate-limited resource and the
terminal must be logged in for any of it to work. Re-pulling 20
years of daily data for 1,200 securities on every research
iteration would be both slow and wasteful, so every pull lands in
parquet and subsequent calls only fetch the gap.

Layout
------
    cache/prices/<SAFE_TICKER>.parquet     daily OHLCV + mcap
    cache/fundamentals/<SAFE_TICKER>.parquet   weekly slow fields
    cache/_manifest.parquet                coverage bookkeeping

One file per security (rather than one giant frame) means a failed
or interrupted pull never corrupts existing data, and delisted
names are simply never refreshed again.
============================================================
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import pandas as pd

from ..bbg.session import BloombergSession
from ..config import FUNDA_DIR, MKT_CAP_BDH_SCALE, PRICES_DIR

log = logging.getLogger(__name__)

MANIFEST_PATH = PRICES_DIR.parent / "_manifest.parquet"

# Bloomberg tickers contain characters illegal in Windows filenames (BRK/B).
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(security: str) -> str:
    return _SAFE.sub("_", security)


def _path(security: str, directory: Path) -> Path:
    return directory / f"{safe_name(security)}.parquet"


# ----------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------
def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_parquet(MANIFEST_PATH)
    return pd.DataFrame(columns=["security", "kind", "first_date", "last_date", "rows", "updated"])


def _update_manifest(records: list[dict]) -> None:
    if not records:
        return
    manifest = load_manifest()
    new = pd.DataFrame(records)
    manifest = (
        pd.concat([manifest, new], ignore_index=True)
        .drop_duplicates(subset=["security", "kind"], keep="last")
        .reset_index(drop=True)
    )
    manifest.to_parquet(MANIFEST_PATH, index=False)


# ----------------------------------------------------------------------
# read
# ----------------------------------------------------------------------
def read_prices(securities: list[str] | None = None) -> pd.DataFrame:
    """Load cached daily price panel for the given securities (all if None)."""
    return _read_dir(PRICES_DIR, securities)


def read_fundamentals(securities: list[str] | None = None) -> pd.DataFrame:
    return _read_dir(FUNDA_DIR, securities)


def _read_dir(directory: Path, securities: list[str] | None) -> pd.DataFrame:
    if securities is None:
        files = sorted(directory.glob("*.parquet"))
    else:
        files = [_path(s, directory) for s in securities]
        files = [f for f in files if f.exists()]
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["security", "date"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# write / fetch
# ----------------------------------------------------------------------
def fetch_series(
    bbg: BloombergSession,
    securities: list[str],
    fields: list[str],
    start: dt.date | str,
    end: dt.date | str,
    directory: Path,
    kind: str,
    periodicity: str = "DAILY",
    batch_size: int = 25,
    force: bool = False,
    scale_market_cap: bool = True,
    overrides: dict | None = None,
) -> dict:
    """Fetch and cache a historical panel, pulling only what is missing.

    A security is skipped when the cache already covers ``end``. Delisted
    names naturally stop updating: their last date never advances, so after
    one attempt they are only retried if ``force`` is set.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    cov = {
        (r.security, r.kind): (pd.Timestamp(r.first_date), pd.Timestamp(r.last_date))
        for r in manifest.itertuples()
    }

    todo: list[tuple[str, pd.Timestamp]] = []
    for sec in securities:
        if force or (sec, kind) not in cov:
            todo.append((sec, start_ts))
            continue
        first, last = cov[(sec, kind)]
        # Refresh only if the cache genuinely lags. 5 calendar days of slack
        # absorbs weekends/holidays without triggering pointless pulls.
        if last < end_ts - pd.Timedelta(days=5):
            todo.append((sec, last + pd.Timedelta(days=1)))

    stats = {"requested": len(securities), "fetched": 0, "skipped": len(securities) - len(todo), "empty": 0}
    if not todo:
        log.info("[%s] cache already current for all %s securities.", kind, len(securities))
        return stats

    log.info("[%s] fetching %s securities (%s already current) ...", kind, len(todo), stats["skipped"])

    # Group by identical start date so batches share one request.
    by_start: dict[pd.Timestamp, list[str]] = {}
    for sec, s in todo:
        by_start.setdefault(s, []).append(sec)

    records: list[dict] = []
    done = 0
    for s_date, secs in by_start.items():
        for i in range(0, len(secs), batch_size):
            batch = secs[i : i + batch_size]
            df = bbg.bdh(batch, fields, s_date, end_ts, periodicity=periodicity,
                         overrides=overrides)
            done += len(batch)

            for sec in batch:
                sub = df[df["security"] == sec].copy() if not df.empty else pd.DataFrame()
                if sub.empty:
                    stats["empty"] += 1
                    # Record the attempt so we do not retry a genuinely dead
                    # ticker on every run.
                    records.append(
                        {
                            "security": sec, "kind": kind,
                            "first_date": pd.NaT, "last_date": end_ts,
                            "rows": 0, "updated": pd.Timestamp.now(),
                        }
                    )
                    continue

                if scale_market_cap and "CUR_MKT_CAP" in sub.columns:
                    sub["CUR_MKT_CAP"] = sub["CUR_MKT_CAP"] * MKT_CAP_BDH_SCALE

                path = _path(sec, directory)
                if path.exists() and not force:
                    old = pd.read_parquet(path)
                    old["date"] = pd.to_datetime(old["date"])
                    sub = pd.concat([old, sub], ignore_index=True)
                sub = (
                    sub.drop_duplicates(subset=["date"], keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                sub.to_parquet(path, index=False)
                stats["fetched"] += 1
                records.append(
                    {
                        "security": sec, "kind": kind,
                        "first_date": sub["date"].min(), "last_date": sub["date"].max(),
                        "rows": len(sub), "updated": pd.Timestamp.now(),
                    }
                )

            if done % 200 < batch_size:
                log.info("  [%s] %s/%s securities | %s", kind, done, len(todo), bbg.stats.summary())

    _update_manifest(records)
    log.info("[%s] done: %s fetched, %s empty, %s skipped.", kind, stats["fetched"], stats["empty"], stats["skipped"])
    return stats
