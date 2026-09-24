"""Vintaged local cache.

The vintaging rule is inherited from GasImbalanceModel and matters just as much
here. Spark restates: a release published on day t can be revised, and terminal
slot counts for a delivery month are updated continuously as slots are sold. A
backtest that reads today's view of a 2024 slot count is reading the future.

So every write is stamped with the moment WE fetched it, and `read_as_of`
returns only rows whose vintage precedes a chosen instant. Reading the raw
frame instead gives restated data and a flatteringly good backtest.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nwebasis.config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data_cache"

VINTAGE_COLUMN = "vintage"
_STAMP_FORMAT = "%Y%m%dT%H%M%S"


def _dataset_dir(dataset: str) -> Path:
    path = CACHE_DIR / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def write(dataset: str, frame: pd.DataFrame, vintage: datetime | None = None) -> Path:
    """Append a fetch to the cache at a new vintage. Never overwrites."""
    if frame.empty:
        raise ValueError(
            f"refusing to cache an empty frame for {dataset!r}; an empty payload "
            "is a source failure and must surface, not be stored as fact"
        )
    stamp = vintage or datetime.now(UTC)
    out = frame.copy()
    out[VINTAGE_COLUMN] = stamp
    path = _dataset_dir(dataset) / f"{dataset}__{stamp.strftime(_STAMP_FORMAT)}.parquet"
    out.to_parquet(path, index=False)
    return path


def read(dataset: str) -> pd.DataFrame:
    """Every vintage of a dataset, concatenated. For inspection, not backtests."""
    files = sorted(_dataset_dir(dataset).glob(f"{dataset}__*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no cached data for {dataset!r}. Run `nwebasis fetch` first."
        )
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def read_latest(dataset: str) -> pd.DataFrame:
    """Most recent vintage only. The right read for producing today's forecast."""
    frame = read(dataset)
    newest = frame[VINTAGE_COLUMN].max()
    return frame[frame[VINTAGE_COLUMN] == newest].drop(columns=[VINTAGE_COLUMN])


def read_as_of(dataset: str, as_of: datetime) -> pd.DataFrame:
    """What we had actually fetched by `as_of`. The only correct backtest read."""
    frame = read(dataset)
    visible = frame[frame[VINTAGE_COLUMN] <= as_of]
    if visible.empty:
        raise FileNotFoundError(
            f"{dataset!r} has no vintage at or before {as_of.isoformat()}; "
            "the cache does not go back far enough to backtest this date"
        )
    newest = visible[VINTAGE_COLUMN].max()
    return visible[visible[VINTAGE_COLUMN] == newest].drop(columns=[VINTAGE_COLUMN])
