"""Snapshot pull of BSRCH power-plant outages into the store (D-043).

One call takes one as-of snapshot for each location: the forward plan becomes FORECAST rows and the realised days
OBSERVATION rows (see data/outages.py). Idempotent per (location, as-of, request date): a second call on the same day
finds the raw batch already stored and writes nothing.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from ttf_forecaster.data import outages as out
from ttf_forecaster.data.storage import Store
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.pipelines.ingest import already_stored, ingest

log = get_logger(__name__)


def _batch_name(location: str, as_of: date, request_time: pd.Timestamp, kind: str) -> str:
    geo_key = out.LOCATIONS[location][0]
    return f"bsrch_outage_{geo_key}_asof{as_of:%Y%m%d}_req{request_time:%Y%m%d}_{kind}"


def pull_snapshot(as_of: date, store: Store, adapter: out.LiveOutageAdapter, *,
                  request_time: pd.Timestamp | None = None,
                  locations: Sequence[str] = tuple(out.LOCATIONS)) -> dict[str, str]:
    """Store one snapshot per location. Returns ``{location: 'written fc=.. obs=..' | 'skipped: ...'}``."""
    out.registered_outage_ids(adapter.config)  # every series is registered and known before any call
    request_time = request_time if request_time is not None else out.request_time_now()
    report: dict[str, str] = {}
    for loc in locations:
        names = {k: _batch_name(loc, as_of, request_time, k) for k in ("fc", "obs")}
        if all(already_stored(store, out.SOURCE, n) for n in names.values()):
            report[loc] = "skipped: already stored for this request date"
            log.info("outage snapshot %s as-of %s already stored", loc, as_of)
            continue
        raw = adapter.snapshot(loc, as_of)
        fc, obs = out.split_snapshot(raw, loc, as_of, request_time)
        n_fc = n_obs = 0
        if len(fc):
            n_fc = ingest(lambda f=fc: f, store, kind="forecast", source=out.SOURCE, raw_name=names["fc"],
                          keep_adapter_ingestion_time=True, fail_on_quality=False).n_written
        if len(obs):
            n_obs = ingest(lambda f=obs: f, store, kind="observation", source=out.SOURCE, raw_name=names["obs"],
                           keep_adapter_ingestion_time=True, fail_on_quality=False).n_written
        report[loc] = f"written fc={n_fc} obs={n_obs}"
    return report
