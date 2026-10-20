"""LNG fleet snapshot from Bloomberg BSRCH COMDTY:SHIP, mapped to VESSEL_SNAPSHOT_SCHEMA (user request, 2026-10-20).

The table is CURRENT STATE: each call is one snapshot, taken at the request time. Its value comes from the daily snapshots
from now on; there is no history. Each row becomes one vessel snapshot:

* snapshot_time = publication_time = the request time (the conservative availability stamp);
* vessel_id = IMO; laden = Cargo status == 'laden';
* declared_destination = Destination (normalized), or null for 'FOR ORDERS', 'Unknown' and blank;
* declared_destination_region = the region of its destination country (config ``country_regions``), or null when unmapped;
* declared_eta = ETA, UNLESS the ETA is earlier than the snapshot: then the ETA is unknown (null), never 'arrived';
* cargo_volume = Cargo quantity for laden vessels, unit ``t_LNG`` (an ASSUMPTION, see configs/sources/lng_fleet.yaml);
* origin = null: the output has no departure-port field, so nothing is guessed.

Staleness: a vessel whose Last Seen is older than ``max_last_seen_age_days`` before the snapshot is kept in the side table
but left out of the snapshot rows that feed the forecasts.

Grade A: this is a live snapshot, vintaged from its first snapshot date onward. Before that date no row exists (never back-filled).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import Entitlement, PitGrade, QualityFlag, validate_frame
from ttf_forecaster.data.spark import VESSEL_SNAPSHOT_SCHEMA

SOURCE = "bloomberg_bsrch_comdty_ship"
EXPECTED_COLUMNS = (
    "IMO", "Vessel Name", "Last Seen", "Navigation Status", "Draft %", "Cargo status", "Cargo product",
    "Cargo quantity", "Destination (normalized)", "Destination Country/Region (normalized)", "ETA", "Depart time",
    "Arrival port", "Arrival time", "Vessel Type 4", "Vessel Class", "Beneficial Owner",
)
NON_DESTINATIONS = {"", "unknown", "for orders"}
SIDE_COLUMNS = ["snapshot_time", "imo", "vessel_name", "last_seen", "navigation_status", "cargo_status", "cargo_product",
                "cargo_quantity", "destination", "destination_country_region", "declared_destination_region", "eta",
                "stale", "eta_unknown"]


@dataclass(frozen=True)
class FleetSnapshot:
    """One snapshot: the side-table rows (every vessel) and the VESSEL_SNAPSHOT_SCHEMA rows (fresh vessels only)."""

    side: pd.DataFrame
    vessels: pd.DataFrame


def _utc(series: pd.Series) -> pd.Series:
    """Mixed datetime and string timestamps -> UTC; unparseable -> NaT."""
    return pd.to_datetime(series.map(lambda v: v if not isinstance(v, str) else (v if v.strip() else None)),
                          utc=True, errors="coerce")


def _destination(text: Any) -> str | None:
    s = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text).strip()
    return None if s.lower() in NON_DESTINATIONS else s


def _country(text: Any) -> str:
    return str(text or "").split(",")[0].strip()


def normalise(raw: pd.DataFrame, snapshot_time: pd.Timestamp, cfg: Mapping[str, Any]) -> FleetSnapshot:
    """Map the BSRCH result to the side table and to VESSEL_SNAPSHOT_SCHEMA rows. Refuses a result missing a column."""
    missing = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"fleet result lacks columns {missing}")
    if raw.empty:
        raise ValueError("fleet result is empty")
    if pd.Timestamp(snapshot_time).tzinfo is None:
        raise ValueError("snapshot_time must be tz-aware UTC")
    snap = pd.Timestamp(snapshot_time).tz_convert("UTC")
    country_regions: Mapping[str, str] = cfg.get("country_regions", {}) or {}
    max_age = pd.Timedelta(days=int(cfg.get("max_last_seen_age_days", 3)))

    last_seen = _utc(raw["Last Seen"])
    eta = _utc(raw["ETA"])
    laden = raw["Cargo status"].astype(str).str.strip().str.lower().eq("laden")
    qty = pd.to_numeric(raw["Cargo quantity"], errors="coerce")
    destination = raw["Destination (normalized)"].map(_destination)
    country = raw["Destination Country/Region (normalized)"].map(_country)
    region = country.map(lambda c: country_regions.get(c))
    stale = (last_seen.isna()) | (snap - last_seen > max_age)
    eta_unknown = eta.notna() & (eta < snap)

    side = pd.DataFrame({
        "snapshot_time": snap,
        "imo": pd.to_numeric(raw["IMO"], errors="coerce").astype("Int64"),
        "vessel_name": raw["Vessel Name"].astype(str),
        "last_seen": last_seen,
        "navigation_status": raw["Navigation Status"].astype(str),
        "cargo_status": raw["Cargo status"].astype(str),
        "cargo_product": raw["Cargo product"].astype(str),
        "cargo_quantity": qty,
        "destination": destination,
        "destination_country_region": raw["Destination Country/Region (normalized)"].astype(str),
        "declared_destination_region": region,
        "eta": eta,
        "stale": stale.to_numpy(),
        "eta_unknown": eta_unknown.to_numpy(),
    }, columns=SIDE_COLUMNS)
    if side["imo"].isna().any():
        raise ValueError("fleet result has rows without an IMO")

    fresh = ~stale.to_numpy()
    n = int(fresh.sum())
    vessels = pd.DataFrame({
        "snapshot_time": [snap] * n,
        "vessel_id": side.loc[fresh, "imo"].astype(str).to_numpy(),
        "voyage_id": [None] * n,
        "laden": laden.to_numpy()[fresh],
        "origin": [None] * n,
        "load_time": [pd.NaT] * n,
        "declared_destination": destination.to_numpy()[fresh],
        "declared_destination_region": region.to_numpy()[fresh],
        "declared_eta": eta.mask(eta_unknown).to_numpy()[fresh],
        "cargo_volume": qty.where(laden).to_numpy()[fresh].astype(float),
        "cargo_volume_unit": [str(cfg["volume_unit"])] * n,
        "publication_time": [snap] * n,
        "ingestion_time": [snap] * n,
        "revision": [0] * n,
        "unit": [str(cfg["volume_unit"])] * n,
        "source": [SOURCE] * n,
        "quality_flag": [QualityFlag.OK.value] * n,
        "entitlement": [Entitlement.LICENSED_BLOOMBERG.value] * n,
        "pit_grade": [PitGrade.A.value] * n,
        "publication_time_estimated": [False] * n,
        "source_timezone": ["UTC"] * n,
    })
    for col in ("snapshot_time", "publication_time", "ingestion_time", "load_time", "declared_eta"):
        vessels[col] = pd.to_datetime(vessels[col], utc=True)
    validate_frame(vessels, VESSEL_SNAPSHOT_SCHEMA)
    return FleetSnapshot(side=side, vessels=vessels)
