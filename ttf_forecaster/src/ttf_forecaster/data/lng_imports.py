"""LNG cargo imports per country from BSRCH TPD:DEX saved searches (user requests, 2026-10-20).

A country's saved search(es) return one row per cargo visit. This module turns the cargo rows into:
* a side table, one row per cargo keyed by (country, IMO, arrival): vessel, berth, port, departure, tonnes, mcm. Several keys per
  country are merged and de-duplicated on that key;
* OBSERVATION rows per country and arrival day: cargo count, tonnes and mcm. Days from the first to the last arrival of the
  calendar with no cargo are 0 (a real zero). Days before a country's ``history_truncated_before`` are NOT stored: a capped search
  (2,000 rows) does not contain them, so they are unknown, never zero. Grade B, published D+1 12:00 UTC.
* regional aggregates (config ``regions``): the sum of the member countries; a day is stored only when every member has it.

Quantities: the export is negative for an import, so the absolute value is used. The unit is TONNES (``Qtymtonnes``), not
million tonnes; the series unit is therefore ``t``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import OBSERVATION_SCHEMA, PitGrade, validate_frame

SOURCE = "bloomberg_bsrch_tpd_dex"
ENTITLEMENT = "licensed_bloomberg"
PUBLICATION_HOUR_UTC = 12  # known at D+1 12:00 UTC
CARGO_COLUMNS = ["country", "imo", "arrival", "vessel_name", "port_name", "berth_name", "departure", "tonnes",
                 "mcm", "bikey_source"]
FIELDS = (("cargoes", "count", "imports_cargoes_count"), ("tonnes", "t", "imports_tonnes"),
          ("mcm", "MCM", "imports_mcm"))


def cargo_rows(raw: pd.DataFrame, country: str, bikey: str) -> pd.DataFrame:
    """Import cargo rows (one per IMO and arrival date) from one saved search's result."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=CARGO_COLUMNS)
    df = raw[raw["Visit_Type"].astype(str).str.lower() == "import"].copy()
    arrival = pd.to_datetime(df["Arrival"]).dt.normalize()
    out = pd.DataFrame({
        "country": country,
        "imo": pd.to_numeric(df["Imo"], errors="coerce").astype("Int64"),
        "arrival": arrival,
        "vessel_name": df["Vessel_Name"].astype(str),
        "port_name": df["Port_Name"].astype(str),
        "berth_name": df["Berth_Name"].astype(str),
        "departure": pd.to_datetime(df["Departure"], errors="coerce"),
        "tonnes": pd.to_numeric(df["Qtymtonnes"], errors="coerce").abs(),
        "mcm": pd.to_numeric(df["Qtymcm"], errors="coerce").abs(),
        "bikey_source": bikey,
    })
    if out["imo"].isna().any() or out["arrival"].isna().any():
        raise ValueError(f"{country}: cargo rows without IMO or arrival date")
    if out.duplicated(["imo", "arrival"]).any():
        raise ValueError(f"{country}: duplicate IMO and arrival rows in one result")
    return out[CARGO_COLUMNS].reset_index(drop=True)


def merge_cargoes(*tables: pd.DataFrame) -> pd.DataFrame:
    """Merge cargo tables (several keys, or an earlier side table) de-duplicated by (country, IMO, arrival)."""
    frames = [t for t in tables if len(t)]
    if not frames:
        return pd.DataFrame(columns=CARGO_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(["country", "imo", "arrival"], keep="last").reset_index(drop=True)


def calendar_of(cargoes: pd.DataFrame) -> pd.DatetimeIndex:
    """Every day from the first to the last arrival of ALL countries (one shared calendar)."""
    if cargoes.empty:
        return pd.DatetimeIndex([])
    return pd.date_range(cargoes["arrival"].min(), cargoes["arrival"].max(), freq="D")


def country_daily(cargoes: pd.DataFrame, calendar: pd.DatetimeIndex,
                  truncated_before: str | None = None) -> pd.DataFrame:
    """Daily cargo count, tonnes and mcm on the calendar. A day with no arrival is 0; a day before ``truncated_before``
    is NaN (the capped search does not contain it)."""
    if len(cargoes):
        g = cargoes.groupby("arrival")
        daily = pd.DataFrame({"cargoes": g.size().astype(float), "tonnes": g["tonnes"].sum(),
                              "mcm": g["mcm"].sum()}).reindex(calendar, fill_value=0.0)
    else:
        daily = pd.DataFrame({"cargoes": 0.0, "tonnes": 0.0, "mcm": 0.0}, index=calendar)
    if truncated_before is not None:
        cut = pd.Timestamp(truncated_before)
        daily.loc[daily.index < cut, :] = np.nan
    return daily


def regional_daily(daily_by_country: Mapping[str, pd.DataFrame], members: Sequence[str]) -> pd.DataFrame:
    """Sum of the member countries' daily tables; a day is NaN when any member is NaN (no partial regional totals)."""
    parts = [daily_by_country[c] for c in members if c in daily_by_country]
    if len(parts) != len(members):
        missing = [c for c in members if c not in daily_by_country]
        raise ValueError(f"regional total needs every member; missing {missing}")
    total = parts[0].copy()
    for p in parts[1:]:
        total = total.add(p, fill_value=None)  # NaN propagates: a missing member makes the day unknown
    return total


def _rows(daily: pd.DataFrame, prefix: str, geography: str, ingestion_time: pd.Timestamp) -> pd.DataFrame:
    """OBSERVATION rows from a daily table. NaN days are not stored (unknown is absent, never zero)."""
    frames = []
    for field, unit, name in FIELDS:
        col = daily[field].dropna()
        n = len(col)
        if n == 0:
            continue
        event = [pd.Timestamp(d, tz="UTC") for d in col.index]
        pub = [e + pd.Timedelta(days=1, hours=PUBLICATION_HOUR_UTC) for e in event]
        frames.append(pd.DataFrame({
            "series_id": [f"{prefix}.{name}"] * n,
            "event_time": event,
            "value": col.to_numpy(dtype=float),
            "geography": [geography] * n,
            "publication_time": pub,
            "ingestion_time": [ingestion_time] * n,
            "revision": np.zeros(n, dtype="int64"),
            "unit": [unit] * n,
            "source": [SOURCE] * n,
            "quality_flag": ["ok"] * n,
            "entitlement": [ENTITLEMENT] * n,
            "pit_grade": [PitGrade.B.value] * n,
            "publication_time_estimated": [True] * n,
            "source_timezone": ["UTC"] * n,
        }))
    if not frames:
        return pd.DataFrame(columns=OBSERVATION_SCHEMA.names)
    return validate_frame(pd.concat(frames, ignore_index=True), OBSERVATION_SCHEMA)


def country_observations(daily: pd.DataFrame, country: str, geography: str,
                         ingestion_time: pd.Timestamp) -> pd.DataFrame:
    return _rows(daily, f"lng.{country.lower()}", geography, ingestion_time)


def region_observations(daily: pd.DataFrame, region: str, ingestion_time: pd.Timestamp) -> pd.DataFrame:
    return _rows(daily, f"lng.region.{region.lower()}", region, ingestion_time)


def country_entries(cfg: Mapping) -> dict[str, dict]:
    """Countries with status known. Each has at least one key; keys are merged per country."""
    out: dict[str, dict] = {}
    for cc, entry in (cfg.get("countries") or {}).items():
        if str(entry.get("status")) != "known":
            continue
        keys = [str(k).strip() for k in (entry.get("bikeys") or []) if str(k).strip()]
        if not keys:
            raise ValueError(f"{cc}: status known but no bikeys")
        out[str(cc)] = {**dict(entry), "bikeys": keys}
    return out
