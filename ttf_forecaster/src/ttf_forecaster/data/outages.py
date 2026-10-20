"""Power-plant unavailability from Bloomberg BSRCH ``ASSET_OUTAGE:POWER_PLANT`` (D-043, verified 2026-10-19).

What the domain returns (measured 2026-10-19, see docs/bloomberg_discovery.md section 8):

* one row per applicable day, columns = unavailable MW per fuel (``STACK=FALSE``); FR, DE and NORDPOOL return
  data, other locations returned 0 rows;
* the forward plan runs to at least 2028-12-31 for one as-of date;
* ``AS_OF_DATE`` is honoured back to 2026-10-13 (2026-10-12 and earlier returned nothing). Values differ between
  as-of dates for the same future day, so the snapshots are real vintages from 2026-10-13 onward, not before;
* the realised window is rolling one year before the as-of date, and one window longer than about three years
  returns nothing, so a snapshot is two requests (realised, then forward plan);

Store layout (per snapshot, as_of ``A`` taken at request time ``R``):

* applicable day ``D >= A`` -> FORECAST_SCHEMA row: issue_time = the as-of date (the vintage identity),
  publication_time = ``R`` (the request time, the conservative stamp: Bloomberg gives no publication time),
  valid_time = D, grade A;
* applicable day ``D < A`` -> OBSERVATION_SCHEMA row: event_time = D, publication_time = max(D+1 at 12:00 UTC, R),
  grade B; a later snapshot that revises D is stored as a new revision (numbered by its as-of date).

Series ids: ``power.<geo>.outage_<fuel>_unavail_mw``, one per (location, fuel) in ``OUTAGE_SERIES``; the same id
carries forecast and observation rows. The registry is ``configs/sources/bloomberg_outages.yaml``; a test keeps the
two in step. Entries have no ticker, so the ticker adapter refuses them; they are read only by
``LiveOutageAdapter``, under the same authorisation as every live request.

No gas-asset outage domain was found: candidates tried (ASSET_OUTAGE:GAS, PIPELINE, GAS_FIELD, LNG, LNG_TERMINAL,
GAS_STORAGE) returned 0 rows, and an unknown domain also returns 0 rows with no error, so domains cannot be
discovered from the API. None is invented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    PitGrade,
    validate_frame,
)
from ttf_forecaster.data.bloomberg import BloombergConfig, check_live_environment

DOMAIN = "ASSET_OUTAGE:POWER_PLANT"
SOURCE = "bloomberg_bsrch"
ENTITLEMENT = "licensed_bloomberg"
MODEL = "bsrch_snapshot"
MEMBER = "daily_plan"
REALISED_PUBLICATION_DAYS = 1  # applicable day D is published D+1 ...
REALISED_PUBLICATION_HOUR_UTC = 12  # ... at 12:00 UTC
HORIZON_END = date(2028, 12, 31)  # the forward plan requested in every snapshot
ROLLING_WINDOW_DAYS = 365  # START_DATE = as-of minus this
EARLIEST_AS_OF = date(2026, 10, 13)  # earliest AS_OF_DATE measured to return data (2026-10-12 returned none)
UNIT = "MW"

# location -> (geography key used in series ids, geography, fuel columns as Bloomberg names them)
LOCATIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "FR": ("fr", "FR", ("Biomass", "Coal", "Hydro", "Natural Gas", "Nuclear", "Oil", "Other", "Storage", "Wind")),
    "DE": ("de", "DE", ("Biomass", "Coal", "Hydro", "Lignite", "Natural Gas", "Oil", "Other", "Other Gas",
                        "Solar", "Storage", "Waste", "Wind")),
    "NORDPOOL": ("nordpool", "NORDPOOL", ("Biomass", "Coal", "Hydro", "Natural Gas", "Nuclear", "Oil", "Other",
                                          "Solar", "Storage", "Waste", "Wind")),
}


def series_id(location: str, fuel: str) -> str:
    """``power.<geo>.outage_<fuel>_unavail_mw``, e.g. ``power.fr.outage_nuclear_unavail_mw``."""
    geo_key, _, _ = LOCATIONS[location]
    return f"power.{geo_key}.outage_{fuel.lower().replace(' ', '_')}_unavail_mw"


OUTAGE_SERIES: dict[str, tuple[str, str]] = {
    series_id(loc, fuel): (loc, fuel) for loc, (_, _, fuels) in LOCATIONS.items() for fuel in fuels
}


def bsrch_options(location: str, as_of: date, *, realised: bool) -> dict[str, str]:
    """BSRCH overrides for one part of a snapshot. Two requests per snapshot, because one window from the rolling
    year to the plan horizon is longer than BSRCH returns (2026-10-19 probe: 2025-10-13..2028-12-31 gave 0 rows,
    2026-01-01..2028-12-31 gave 1,096).

    ``realised``: the rolling year before the as-of date, up to the day before it.
    forward: the as-of date through the plan horizon.
    """
    if location not in LOCATIONS:
        raise ValueError(f"no outage domain location {location!r}")
    if realised:
        start, end = as_of - timedelta(days=ROLLING_WINDOW_DAYS), as_of - timedelta(days=1)
    else:
        start, end = as_of, HORIZON_END
    return {"LOCATION": location, "START_DATE": f"{start:%Y-%m-%d}", "END_DATE": f"{end:%Y-%m-%d}",
            "AS_OF_DATE": f"{as_of:%Y-%m-%d}", "STACK": "FALSE"}


def split_snapshot(df: pd.DataFrame, location: str, as_of: date, request_time: pd.Timestamp
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(FORECAST_SCHEMA rows, OBSERVATION_SCHEMA rows) from one BSRCH result, split at the as-of date.

    Realised rows: ``publication_time`` = max(D+1 12:00 UTC, ``request_time``); ``revision`` = the as-of date's
    offset from ``EARLIEST_AS_OF``. A later snapshot that revises a day adds a revision instead of conflicting.

    Vintage: ``issue_time`` = the as-of date at 00:00 UTC (it identifies the snapshot, so two as-of dates never
    share a key). ``publication_time`` = ``request_time``: the conservative availability, since Bloomberg gives no
    publication time and the snapshot is not known to exist before it was requested.

    Raises ``ValueError`` when the result lacks the location's fuel columns or is empty.
    """
    _, geo, fuels = LOCATIONS[location]
    if as_of < EARLIEST_AS_OF:
        raise ValueError(f"as-of {as_of} is before {EARLIEST_AS_OF}: BSRCH returns nothing earlier")
    # Realised values of different snapshots can differ (2026-10-19 probe: hydro). Each snapshot is its own revision,
    # numbered by its as-of date, so a re-delivery never conflicts with a stored value.
    revision = (as_of - EARLIEST_AS_OF).days
    if "Applicable Date" not in df.columns:
        raise ValueError(f"{location}: result without 'Applicable Date': {list(df.columns)}")
    missing = [f for f in fuels if f not in df.columns]
    if missing:
        raise ValueError(f"{location}: result lacks fuel columns {missing}")
    if df.empty:
        raise ValueError(f"{location}: empty result for as-of {as_of}")
    days = pd.to_datetime(df["Applicable Date"]).dt.date
    fc_rows, obs_rows = [], []
    request_time = pd.Timestamp(request_time).tz_convert("UTC")
    issue = pd.Timestamp(as_of, tz="UTC")
    for fuel in fuels:
        sid = series_id(location, fuel)
        values = pd.to_numeric(df[fuel], errors="coerce").to_numpy(dtype=float)
        future = np.array([d >= as_of for d in days])
        event = [pd.Timestamp(datetime.combine(d, time(0), tzinfo=timezone.utc)) for d in days]
        if future.any():
            idx = np.flatnonzero(future)
            n = len(idx)
            fc_rows.append(pd.DataFrame({
                "series_id": [sid] * n,
                "issue_time": [issue] * n,
                "valid_time": [event[i] for i in idx],
                "lead_hours": [(event[i] - pd.Timestamp(as_of, tz="UTC")).total_seconds() / 3600.0 for i in idx],
                "model": [MODEL] * n,
                "member": [MEMBER] * n,
                "value": values[idx],
                "geography": [geo] * n,
                "publication_time": [request_time] * n,
                "ingestion_time": [request_time] * n,
                "revision": np.zeros(n, dtype="int64"),
                "unit": [UNIT] * n,
                "source": [SOURCE] * n,
                "quality_flag": ["ok"] * n,
                "entitlement": [ENTITLEMENT] * n,
                "pit_grade": [PitGrade.A.value] * n,
                "publication_time_estimated": [False] * n,
                "source_timezone": ["UTC"] * n,
            }))
        idx = np.flatnonzero(~future)
        if len(idx):
            n = len(idx)
            # D+1 12:00 UTC, but never before the request: a snapshot cannot be known before it was taken.
            pub = [max(event[i] + pd.Timedelta(days=REALISED_PUBLICATION_DAYS, hours=REALISED_PUBLICATION_HOUR_UTC),
                       request_time) for i in idx]
            obs_rows.append(pd.DataFrame({
                "series_id": [sid] * n,
                "event_time": [event[i] for i in idx],
                "value": values[idx],
                "geography": [geo] * n,
                "publication_time": pub,
                "ingestion_time": [request_time] * n,
                "revision": np.full(n, revision, dtype="int64"),
                "unit": [UNIT] * n,
                "source": [SOURCE] * n,
                "quality_flag": ["ok"] * n,
                "entitlement": [ENTITLEMENT] * n,
                "pit_grade": [PitGrade.B.value] * n,
                "publication_time_estimated": [True] * n,
                "source_timezone": ["UTC"] * n,
            }))
    fc = pd.concat(fc_rows, ignore_index=True) if fc_rows else pd.DataFrame(columns=FORECAST_SCHEMA.names)
    obs = pd.concat(obs_rows, ignore_index=True) if obs_rows else pd.DataFrame(columns=OBSERVATION_SCHEMA.names)
    fc = validate_frame(fc, FORECAST_SCHEMA) if len(fc) else fc
    obs = validate_frame(obs, OBSERVATION_SCHEMA) if len(obs) else obs
    return fc, obs


@dataclass
class LiveOutageAdapter:
    """BSRCH through xbbg. Refuses unless live calls are authorised (``check_live_environment``). ``client``
    (default ``xbbg.blp``) is injectable for tests."""

    config: BloombergConfig
    client: Any | None = None

    def __post_init__(self) -> None:
        check_live_environment(self.config)

    def _blp(self) -> Any:
        if self.client is None:
            from xbbg import blp  # noqa: PLC0415 - only on an authorised live request

            self.client = blp
        return self.client

    def snapshot(self, location: str, as_of: date) -> pd.DataFrame:
        """The raw BSRCH result for one location and as-of date: the realised days, then the forward plan."""
        blp = self._blp()
        parts = [blp.bsrch(DOMAIN, bsrch_options(location, as_of, realised=r)) for r in (True, False)]
        return pd.concat([p for p in parts if len(p)], ignore_index=True)


def registered_outage_ids(config: BloombergConfig) -> list[str]:
    """Series in the outage registry that the config declares ``known``. Checked before any live call."""
    ids = []
    for sid, spec in config.series.items():
        if spec.source_file == "bloomberg_outages.yaml" and spec.status == "known":
            ids.append(sid)
    missing = sorted(set(OUTAGE_SERIES) - set(ids))
    if missing:
        raise ValueError(f"bloomberg_outages.yaml lacks known entries for {missing}")
    return sorted(ids)


def request_time_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))
