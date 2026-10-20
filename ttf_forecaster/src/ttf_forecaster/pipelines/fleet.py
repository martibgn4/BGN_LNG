"""Daily LNG fleet snapshot: BSRCH COMDTY:SHIP -> vessel snapshots -> arrival forecasts and destination observations (user request, 2026-10-20).

One run takes today's snapshot (request time = snapshot time = publication time) and:
* appends the snapshot's vessel rows to the vintage table ``data/live/side/lng_vessel_snapshots.parquet`` (VESSEL_SNAPSHOT_SCHEMA)
  and the raw side table ``lng_fleet_raw.parquet`` (every vessel, with the staleness and ETA-unknown flags);
* runs ``build_arrival_forecasts`` and ``destination_observations`` at the snapshot time and stores the rows
  (grade A: live snapshots; no row exists before the first snapshot, so nothing is back-filled).

Idempotent per UTC day: when a snapshot for today is already in the vintage table, nothing is requested or written.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ttf_forecaster.data import fleet as fl
from ttf_forecaster.data.spark import VESSEL_SNAPSHOT_SCHEMA, build_arrival_forecasts, destination_observations
from ttf_forecaster.data.storage import Store
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.pipelines.ingest import ingest
from ttf_forecaster.settings import load_config

log = get_logger(__name__)
Runner = Callable[[str, dict], pd.DataFrame]  # (domain, options) -> the raw BSRCH result
VESSEL_FILE = "lng_vessel_snapshots.parquet"
RAW_FILE = "lng_fleet_raw.parquet"


def _side(store: Store, name: str) -> Path:
    return Path(store.root) / "side" / name  # type: ignore[attr-defined]


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns or [])


def already_taken(store: Store, day: pd.Timestamp) -> bool:
    """True when a snapshot with this UTC day's date is in the vintage table."""
    path = _side(store, VESSEL_FILE)
    if not path.exists():
        return False
    snaps = pd.read_parquet(path, columns=["snapshot_time"])
    return bool((pd.to_datetime(snaps["snapshot_time"], utc=True).dt.date == day.date()).any())


def run_snapshot(store: Store, runner: Runner, *, request_time: pd.Timestamp | None = None,
                 cfg: dict | None = None) -> dict[str, Any]:
    """Take today's snapshot and store it with the forecasts it supports. Returns a summary."""
    cfg = cfg if cfg is not None else load_config("sources/lng_fleet")["fleet_snapshot"]
    snap = pd.Timestamp(request_time if request_time is not None else pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    if already_taken(store, snap):
        return {"skipped": "snapshot for this UTC day already stored"}
    raw = runner(cfg["domain"], dict(cfg["options"]))
    result = fl.normalise(raw, snap, cfg)

    vessel_path = _side(store, VESSEL_FILE)
    vessel_path.parent.mkdir(parents=True, exist_ok=True)
    prev_v = _read(vessel_path)
    all_v = pd.concat([prev_v, result.vessels], ignore_index=True) if len(prev_v) else result.vessels
    all_v.to_parquet(vessel_path, index=False)
    raw_path = _side(store, RAW_FILE)
    prev_r = _read(raw_path)
    (pd.concat([prev_r, result.side], ignore_index=True) if len(prev_r) else result.side).to_parquet(raw_path, index=False)

    issues = [snap]
    arrivals = build_arrival_forecasts(
        all_v, issues, terminal_regions={}, regions=list(cfg["regions"]),
        horizons_days=tuple(int(h) for h in cfg["horizons_days"]), volume_unit=str(cfg["volume_unit"]),
        max_snapshot_age=pd.Timedelta(days=int(cfg["max_last_seen_age_days"])))
    dest = destination_observations(all_v, issues, terminal_regions={}, focus_region="EU",
                                    max_snapshot_age=pd.Timedelta(days=int(cfg["max_last_seen_age_days"])))
    written = {}
    if len(arrivals):
        written["arrivals"] = int(ingest(lambda f=arrivals: f, store, kind="forecast", source=fl.SOURCE,
                                         raw_name=f"fleet_arrivals_{snap:%Y%m%d}", keep_adapter_ingestion_time=True,
                                         fail_on_quality=False).n_written)
    if len(dest):
        written["destination"] = int(ingest(lambda f=dest: f, store, kind="observation", source=fl.SOURCE,
                                            raw_name=f"fleet_destination_{snap:%Y%m%d}",
                                            keep_adapter_ingestion_time=True, fail_on_quality=False).n_written)
    summary = {"snapshot_time": str(snap), "vessels": len(result.side),
               "fresh": len(result.vessels), "laden_fresh": int(result.vessels["laden"].sum()),
               "eta_unknown": int(result.side["eta_unknown"].sum()), "written": written}
    log.info("fleet snapshot %s: %s", snap, summary)
    return summary
