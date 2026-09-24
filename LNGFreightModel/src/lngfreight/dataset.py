"""Pull every source into one long frame, and cache it with a vintage.

Long format throughout: (obs_date, series_id, value, publication_lag_days).
The lag travels on the row so that the feature builder cannot apply the wrong
one, and so an added source cannot inherit a neighbour's lag by accident.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from lngfreight import store
from lngfreight.config import get
from lngfreight.data import bloomberg, spark
from lngfreight.data.base import SourceUnavailable

log = logging.getLogger(__name__)

RAW_KEY = "raw_panel"
REQUIRED_COLUMNS = ["obs_date", "series_id", "value", "publication_lag_days"]

def _history_start() -> date:
    """How far back sources are pulled. See target.yaml `pipeline`."""
    return date.fromisoformat(get("target", "pipeline", "history_start"))


def _cache_max_age_hours() -> float:
    return float(get("target", "pipeline", "cache_max_age_hours"))


def _sources():
    """Every fetcher, with a label used for logging and partial-failure reports."""
    return [
        ("bbg_prices", bloomberg.price_fetcher()),
        ("bbg_on_water", bloomberg.on_water_fetcher()),
        ("bbg_routing", bloomberg.routing_fetcher()),
        ("bbg_supply", bloomberg.supply_fetcher()),
        ("bbg_flows", bloomberg.flows_fetcher()),
        ("bbg_freight_futures", bloomberg.freight_futures_fetcher()),
        ("spark_atlantic", spark.FreightSpotFetcher("atlantic_spot")),
        ("spark_pacific", spark.FreightSpotFetcher("pacific_spot")),
        ("spark_ffa_atlantic", spark.FfaCurveFetcher("atlantic_monthly")),
    ]


def build(start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Fetch everything. Raises if any source fails - no partial panels.

    A partial panel is worse than no panel: the model would fit on whatever
    happened to load that morning and its results would not be comparable to
    yesterday's. The one acceptable partial is a source that is *configured*
    as optional, and none currently is.
    """
    start = start or _history_start()
    end = end or date.today()
    frames, failures = [], []

    for label, fetcher in _sources():
        try:
            result = fetcher.fetch(start=start, end=end)
        except SourceUnavailable as exc:
            failures.append(f"{label}: {exc}")
            continue
        frame = result.frame
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            failures.append(f"{label}: fetcher returned no {missing}")
            continue
        log.info("%-20s %6d rows  %s -> %s", label, len(frame),
                 frame["obs_date"].min().date(), frame["obs_date"].max().date())
        frames.append(frame[REQUIRED_COLUMNS])

    if failures:
        raise SourceUnavailable(
            "the panel is incomplete, so nothing was built:\n  "
            + "\n  ".join(failures)
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(subset=["obs_date", "series_id"], keep="last")
    return panel.sort_values(["series_id", "obs_date"]).reset_index(drop=True)


def load_or_build(refresh: bool = False,
                  max_age_hours: float | None = None) -> pd.DataFrame:
    if max_age_hours is None:
        max_age_hours = _cache_max_age_hours()
    if not refresh:
        cached = store.load(RAW_KEY, max_age_hours=max_age_hours)
        if cached is not None:
            log.info("using cached panel, vintage %s", store.vintage_of(RAW_KEY))
            return cached
    panel = build()
    store.save(RAW_KEY, panel)
    return panel


def to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide on a business-day grid, WITHOUT forward filling.

    Forward filling is a feature-building decision, not a loading one, and it
    is made in features.py where the resulting staleness can be recorded. A
    NaN here means "no print that day", which is the truth.
    """
    wide = panel.pivot(index="obs_date", columns="series_id", values="value")
    freq = get("target", "calendar", "frequency")
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq=freq)
    return wide.reindex(grid).rename_axis(index="obs_date")


def lags(panel: pd.DataFrame) -> dict[str, int]:
    """series_id -> publication lag in days, as carried on the rows."""
    per = panel.groupby("series_id")["publication_lag_days"].nunique()
    inconsistent = per[per > 1]
    if len(inconsistent):
        raise ValueError(
            f"these series carry more than one publication lag: "
            f"{list(inconsistent.index)}. A lag must be single-valued or the "
            "feature builder cannot apply it."
        )
    return (panel.groupby("series_id")["publication_lag_days"]
            .first().astype(int).to_dict())
