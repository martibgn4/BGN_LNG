"""Vintaged local cache, with the extra time axis this project needs.

The sibling projects carry two timestamps. This one carries three, and the
distinction is the difference between a real backtest and a fictional one:

  ``vintage``      when WE fetched the row. Protects against the upstream
                   archive being restated under a finished backtest - Open-Meteo
                   does reprocess when a model is upgraded.
  ``issue_date``   when the FORECAST was produced. Protects against using a
                   forecast before it existed.
  ``target_date``  the day being forecast.

`read_as_of` filters on vintage alone and is the right read for a
non-forecast dataset. For anything carrying `issue_date` - which is every
weather panel here - use `read_forecast_as_of`, which filters on both. Using
the wrong one gives a backtest that trades on forecasts issued days into its
own future, and it will look excellent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wxreturns.config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data_cache"

VINTAGE_COLUMN = "vintage"
ISSUE_COLUMN = "issue_date"
_STAMP_FORMAT = "%Y%m%dT%H%M%S"


def _dataset_dir(dataset: str) -> Path:
    path = CACHE_DIR / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def _aligned_cutoff(column: pd.Series, as_of: datetime) -> pd.Timestamp:
    """A cutoff timestamp comparable with ``column``.

    The two time axes here genuinely differ in awareness and it is not an
    accident worth normalising away. ``vintage`` is stamped with
    ``datetime.now(UTC)`` and is tz-aware, because when WE fetched something is
    a real instant. ``issue_date`` is a model run date and is naive, because a
    00z run is a calendar label rather than a moment in anyone's timezone.

    Comparing a naive cutoff against the aware column raises rather than
    silently mis-comparing, which is the right behaviour and is why this exists
    instead of a blanket tz-strip.
    """
    stamp = pd.Timestamp(as_of)
    aware = pd.api.types.is_datetime64tz_dtype(column)
    if aware and stamp.tz is None:
        return stamp.tz_localize("UTC")
    if not aware and stamp.tz is not None:
        return stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def write(dataset: str, frame: pd.DataFrame, vintage: datetime | None = None,
          tag: str | None = None) -> Path:
    """Append a fetch to the cache at a new vintage. Never overwrites.

    ``tag`` distinguishes several writes within one second - backfilling a
    decade of forecasts issues many requests in quick succession, and without a
    tag they would collide on the timestamp and silently overwrite each other.
    """
    if frame.empty:
        raise ValueError(
            f"refusing to cache an empty frame for {dataset!r}; an empty payload "
            "is a source failure and must surface, not be stored as fact"
        )
    stamp = vintage or datetime.now(UTC)
    out = frame.copy()
    out[VINTAGE_COLUMN] = stamp
    suffix = f"__{tag}" if tag else ""
    name = f"{dataset}__{stamp.strftime(_STAMP_FORMAT)}{suffix}.parquet"
    path = _dataset_dir(dataset) / name
    out.to_parquet(path, index=False)
    return path


def datasets() -> list[str]:
    """Every dataset with at least one cached vintage."""
    if not CACHE_DIR.exists():
        return []
    return sorted(p.name for p in CACHE_DIR.iterdir() if p.is_dir())


def read(dataset: str) -> pd.DataFrame:
    """Every vintage of a dataset, concatenated. For inspection, not backtests."""
    files = sorted(_dataset_dir(dataset).glob(f"{dataset}__*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no cached data for {dataset!r}. Run `wxreturns fetch-weather` "
            "or `wxreturns fetch-prices` first."
        )
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def read_latest(dataset: str) -> pd.DataFrame:
    """Most recent vintage only. The right read for producing today's forecast."""
    frame = read(dataset)
    newest = frame[VINTAGE_COLUMN].max()
    return frame[frame[VINTAGE_COLUMN] == newest].drop(columns=[VINTAGE_COLUMN])


def read_as_of(dataset: str, as_of: datetime) -> pd.DataFrame:
    """What we had fetched by ``as_of``. Correct for non-forecast datasets."""
    frame = read(dataset)
    cutoff = _aligned_cutoff(frame[VINTAGE_COLUMN], as_of)
    visible = frame[frame[VINTAGE_COLUMN] <= cutoff]
    if visible.empty:
        raise FileNotFoundError(
            f"{dataset!r} has no vintage at or before {as_of.isoformat()}; "
            "the cache does not go back far enough to backtest this date"
        )
    newest = visible[VINTAGE_COLUMN].max()
    return visible[visible[VINTAGE_COLUMN] == newest].drop(columns=[VINTAGE_COLUMN])


def read_forecast_as_of(dataset: str, as_of: datetime) -> pd.DataFrame:
    """Every forecast ISSUED at or before ``as_of``, deduplicated to one row.

    Unlike `read_as_of` this keeps the whole issue-date history rather than a
    single vintage, because a revision needs consecutive issues side by side.
    Where the same (issue_date, target_date, region, variable) has been fetched
    more than once - a backfill overlapping a daily pull - the latest vintage
    at or before ``as_of`` wins.
    """
    frame = read(dataset)
    if ISSUE_COLUMN not in frame.columns:
        raise KeyError(
            f"{dataset!r} carries no {ISSUE_COLUMN!r} column, so it is not a "
            "forecast panel. Use read_as_of for this dataset."
        )
    issued = pd.to_datetime(frame[ISSUE_COLUMN])
    visible = frame[
        (frame[VINTAGE_COLUMN] <= _aligned_cutoff(frame[VINTAGE_COLUMN], as_of))
        & (issued <= _aligned_cutoff(issued, as_of))]
    if visible.empty:
        raise FileNotFoundError(
            f"{dataset!r} has nothing issued and fetched by {as_of.isoformat()}"
        )
    keys = [c for c in ("issue_date", "target_date", "region", "point",
                        "variable", "model", "member") if c in visible.columns]
    return (visible.sort_values(VINTAGE_COLUMN)
            .drop_duplicates(subset=keys, keep="last")
            .drop(columns=[VINTAGE_COLUMN])
            .reset_index(drop=True))
