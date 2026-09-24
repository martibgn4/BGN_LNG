"""Vintaged parquet cache.

Every pull is written with the timestamp it was taken at. Two reasons, both
learned in the sibling models: a refreshed series that silently changed its
own history is invisible without a vintage, and a backtest that cannot say
which snapshot it ran on is not reproducible.

The cache is a convenience, never a source of truth: ``load`` returns None on
a miss and the caller re-fetches. It never fabricates a frame.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from lngfreight.config import cache_dir

log = logging.getLogger(__name__)

VINTAGE_COL = "_vintage"
SECONDS_PER_HOUR = 60 * 60


def _path(key: str) -> Path:
    return cache_dir() / f"{key}.parquet"


def save(key: str, frame: pd.DataFrame) -> Path:
    if frame.empty:
        raise ValueError(f"refusing to cache an empty frame for {key!r}")
    out = frame.copy()
    out[VINTAGE_COL] = datetime.now(UTC).isoformat()
    path = _path(key)
    out.to_parquet(path, index=True)
    log.info("cached %s rows to %s", len(out), path.name)
    return path


def load(key: str, max_age_hours: float | None = None) -> pd.DataFrame | None:
    path = _path(key)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if VINTAGE_COL not in frame.columns:
        log.warning("%s has no vintage column; ignoring the cache", path.name)
        return None
    vintage = pd.to_datetime(frame[VINTAGE_COL].iloc[0])
    if max_age_hours is not None:
        age = (datetime.now(UTC) - vintage.to_pydatetime()).total_seconds() / SECONDS_PER_HOUR
        if age > max_age_hours:
            log.info("%s is %.1fh old, past the %.1fh limit", path.name, age, max_age_hours)
            return None
    return frame.drop(columns=[VINTAGE_COL])


def vintage_of(key: str) -> datetime | None:
    path = _path(key)
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=[VINTAGE_COL])
    return pd.to_datetime(frame[VINTAGE_COL].iloc[0]).to_pydatetime()
