"""Bloomberg weather through BQL ``weather()``, read via the user's Excel (``bql_excel``).

Why: the Python API login is not entitled to BQL, but Excel is. Bloomberg's weather is only reachable through
BQL (no instrument tickers), so this is the only route for European HDD/CDD and temperature forecasts (D-042).

Lead mapping (proved 2026-10-19 with single-offset requests, see tests/unit/market/test_weather_bql.py):
  * a query for ``as_of_date = D`` and ``offset = range(1, 16)`` returns 15 rows per location, in valid-date order;
  * row ``i`` (0-based) is the valid day ``D + i``; lead = ``i`` days;
  * a single-offset query for offset ``k`` returns the value of row ``k - 1`` of the range query and the valid
    date ``D + k - 1`` (checked for k = 1, 2, 5, 9);
  * a date field and a value field in one ``get`` do NOT align (values come back empty), so dates are requested
    on their own (``dates_query``) as an audit, never beside the values.
Forecast horizon: 15 valid days, of which lead 0 (= D) is the analysis day and HDD is null there.

Point in time (ASSUMPTIONS, logged in D-042): the archive is vintaged by ``as_of_date``. The 0z run is issued at
``D 00:00 UTC``. It is treated as available from ``D 10:00 UTC`` (conservative for ECMWF 0z dissemination).
Sensitivity to a later publication time is measured by moving ``PUBLICATION_HOUR_UTC``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    Entitlement,
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    DataError,
    PitGrade,
    validate_frame,
)

SOURCE = "bloomberg_bql_excel"
ENTITLEMENT = Entitlement.LICENSED_BLOOMBERG.value
PUBLICATION_HOUR_UTC = 10  # ASSUMPTION (D-042): forecast run available from D 10:00 UTC
OBS_PUBLICATION_DAYS = 1  # observed day D available D+1 12:00 UTC (ASSUMPTION, grade B)
OBS_PUBLICATION_HOUR_UTC = 12
LEADS = 15  # valid days returned per as_of query: lead 0..14

# BQL location string -> (internal key, geography). Countries use POPULATION weights.
LOCATIONS: dict[str, tuple[str, str]] = {
    "DE Country": ("de", "DE"), "NL Country": ("nl", "NL"), "FR Country": ("fr", "FR"),
    "GB Country": ("gb", "GB"), "IT Country": ("it", "IT"), "BE Country": ("be", "BE"),
    "AT Country": ("at", "AT"), "PL Country": ("pl", "PL"),
    "NORTHWEST EUROPE REGION": ("nwe", "NWE"), "CENTRAL EUROPE REGION": ("cen", "CEN"),
    "EASTERN EUROPE REGION": ("eeu", "EEU"), "SOUTHEAST EUROPE REGION": ("seu", "SEU"),
}

# BQL field -> (measure name in series ids, unit). Units are the BQL convention, ASSUMED; check on the Terminal.
FIELDS: dict[str, tuple[str, str]] = {
    "HDD_18C": ("hdd18", "degC_day"),
    "CDD_18C": ("cdd18", "degC_day"),
    "TEMPERATURE": ("t2m_mean", "degC"),
    "WIND_SPEED": ("wind_speed", "m_s"),
    "SOLAR_RADIATION": ("solar_rad", "W_m2"),
}

# One BQL query per (as_of, source) carries several model types, each in its own let (verified 2026-10-19:
# the columns match the single-model queries). (model_type, member label) per source.
# ENSEMBLE_PERCENTILE_5/25 and CONTROL return "#N/A Invalid Parameter" and are not used.
FORECAST_BUNDLES: dict[str, tuple[tuple[str, str], ...]] = {
    "ECMWF": (("ENSEMBLE_MEAN", "mean"), ("DETERMINISTIC", "det"), ("ENSEMBLE_MIN", "min"),
              ("ENSEMBLE_MAX", "max"), ("ENSEMBLE_MEDIAN", "median")),
    "GFS": (("ENSEMBLE_MEAN", "mean"), ("DETERMINISTIC", "det")),
}
NORMAL_SOURCES: tuple[str, ...] = ("NORMALS_10Y", "NORMALS_30Y")


# Field availability (verified 2026-10-19): SOLAR_RADIATION is returned for forecasts only. Normals and
# actuals return HDD_18C, CDD_18C, TEMPERATURE and WIND_SPEED; a request with SOLAR_RADIATION fails for both.
FORECAST_FIELDS: tuple[str, ...] = tuple(FIELDS)
OBSERVED_FIELDS: tuple[str, ...] = ("HDD_18C", "CDD_18C", "TEMPERATURE", "WIND_SPEED")

ACTUAL_LABEL_MAX_SHIFT_DAYS = 1  # a location's observed-day labels start at the request or the next day (D-045)


def series_id(location_key: str, field: str, *, observed: bool, normal: str | None = None) -> str:
    """``weather.<geo>.<measure>_<unit>`` for forecasts; ``..._obs_...`` for actuals; ``..._norm10y_...``."""
    measure, unit = FIELDS[field]
    if normal:
        tag = f"_{normal.lower()}"
    else:
        tag = "_obs" if observed else ""
    return f"weather.{location_key}.{measure}{tag}_{unit.lower()}"


def _bql_fields(fields: Sequence[str] = FORECAST_FIELDS) -> str:
    return ", ".join(f"#w().{f}" for f in fields)


def _locations() -> str:
    return "[" + ", ".join(f"'{loc}'" for loc in LOCATIONS) + "]"


def _iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def forecast_query(as_of: date, est_source: str, models: Sequence[tuple[str, str]] | None = None) -> str:
    """One query per (as_of, source): every model type (own let, alias ``#m<j>``), all locations and fields,
    leads 0..14 (ECMWF) or 0..15 (GFS). Columns are ``#m<j>().<FIELD>``."""
    models = models if models is not None else FORECAST_BUNDLES[est_source]
    lets = "; ".join(
        f"#m{j}=weather(ae='E', per='D', as_of_date='{_iso(as_of)}', est_source='{est_source}', "
        f"model_type=['{mt}'], weight_type='POPULATION', run='0z', offset=range(1,16))"
        for j, (mt, _member) in enumerate(models))
    cols = ", ".join(f"#m{j}().{f}" for j in range(len(models)) for f in FORECAST_FIELDS)
    return f"let({lets}; ) get({cols}) for({_locations()})"


def normals_query(as_of: date, est_source: str) -> str:
    """Normals are climatology: their values do not depend on as_of. One as_of gives ~15 calendar days."""
    return (
        f"let(#w=weather(ae='E', per='D', as_of_date='{_iso(as_of)}', est_source='{est_source}', "
        f"weight_type='POPULATION', offset=range(1,16)); ) get({_bql_fields(OBSERVED_FIELDS)}) for({_locations()})"
    )


def actual_query(start: date, end: date) -> str:
    """Observed daily values for the requested period (literal dates). BQL labels each value with the NEXT day:
    see ``actual_event_dates``."""
    return (
        f"let(#w=weather(ae='A', per='D', period_reference=range('{_iso(start)}','{_iso(end)}'), "
        f"weight_type='POPULATION'); ) get({_bql_fields(OBSERVED_FIELDS)}) for({_locations()})"
    )


def dates_query(as_of: date, est_source: str = "ECMWF", model_type: str = "ENSEMBLE_MEAN") -> str:
    """Audit query: the valid dates of an as_of query, on their own (never beside values)."""
    return (
        f"let(#w=weather(ae='E', per='D', as_of_date='{_iso(as_of)}', est_source='{est_source}', "
        f"model_type=['{model_type}'], weight_type='POPULATION', run='0z', offset=range(1,16)); ) "
        f"get(#w().period_reference_date) for('DE Country')"
    )


def _split_by_id(df: pd.DataFrame, expected: int | None) -> dict[str, pd.DataFrame]:
    """Group the flat result by ID (a location), in order. Every location must return the same number of
    rows (``expected`` if given; otherwise inferred and required to be 15 or 16 leads)."""
    if "ID" not in df.columns:
        raise ValueError(f"result without an ID column: {list(df.columns)}")
    out: dict[str, pd.DataFrame] = {}
    for bql_id, grp in df.groupby("ID", sort=False):
        out[str(bql_id)] = grp.reset_index(drop=True)
    counts = {len(g) for g in out.values()}
    if len(counts) != 1:
        raise ValueError(f"locations return different row counts: {sorted(counts)}")
    n = counts.pop()
    if expected is not None and n != expected:
        raise ValueError(f"{n} rows per location, expected {expected}")
    if expected is None and n not in (LEADS, LEADS + 1):
        raise ValueError(f"{n} rows per location, expected {LEADS} or {LEADS + 1}")
    unknown = set(out) - set(LOCATIONS)
    if unknown:
        raise ValueError(f"unknown location ids in result: {sorted(unknown)}")
    return out


def _value_column(grp: pd.DataFrame, field: str) -> np.ndarray:
    col = f"#w().{field}"
    if col not in grp.columns:
        raise ValueError(f"missing column {col}: {list(grp.columns)}")
    return pd.to_numeric(grp[col], errors="coerce").to_numpy(dtype=float)


def forecast_frame(df: pd.DataFrame, as_of: date, est_source: str, ingestion_time: pd.Timestamp,
                   models: Sequence[tuple[str, str]] | None = None) -> pd.DataFrame:
    """FORECAST_SCHEMA rows from one (as_of, source) bundle result, every model and location.

    Lead: row i of a location's block is the valid day ``as_of + i`` (proved in the module docstring).
    ``models`` = ``[(model_type, member)]`` in the order of the query's ``#m<j>`` aliases.
    """
    models = models if models is not None else FORECAST_BUNDLES[est_source]
    issue = pd.Timestamp(as_of, tz="UTC")
    publication = issue + pd.Timedelta(hours=PUBLICATION_HOUR_UTC)
    rows = []
    for bql_id, grp in _split_by_id(df, None).items():
        key, geo = LOCATIONS[bql_id]
        n = len(grp)
        leads = np.arange(n)
        for j, (model_type, member) in enumerate(models):
            for field in FORECAST_FIELDS:
                col = f"#m{j}().{field}"
                if col not in grp.columns:
                    raise ValueError(f"missing column {col}: {list(grp.columns)}")
                rows.append(pd.DataFrame({
                    "series_id": [series_id(key, field, observed=False)] * n,
                    "issue_time": [issue] * n,
                    "valid_time": [issue + pd.Timedelta(days=int(i)) for i in leads],
                    "lead_hours": leads * 24.0,
                    "model": [f"{est_source}_{model_type}"] * n,
                    "member": [member] * n,
                    "value": pd.to_numeric(grp[col], errors="coerce").to_numpy(dtype=float),
                    "geography": [geo] * n,
                    "publication_time": [publication] * n,
                    "ingestion_time": [ingestion_time] * n,
                    "revision": np.zeros(n, dtype="int64"),
                    "unit": [FIELDS[field][1]] * n,
                    "source": [SOURCE] * n,
                    "quality_flag": ["ok"] * n,
                    "entitlement": [ENTITLEMENT] * n,
                    "pit_grade": [PitGrade.A.value] * n,
                    "publication_time_estimated": [True] * n,
                    "source_timezone": ["UTC"] * n,
                }))
    out = pd.concat(rows, ignore_index=True)
    return validate_frame(out, FORECAST_SCHEMA)


def actual_dates_query(start: date, end: date) -> str:
    """Dates-only query for the same actual period (never beside values: see the module docstring)."""
    return (
        f"let(#w=weather(ae='A', per='D', period_reference=range('{_iso(start)}','{_iso(end)}'), "
        f"weight_type='POPULATION'); ) get(#w().period_reference_date) for({_locations()})"
    )


def actual_event_dates(dates: pd.DataFrame, start: date, end: date) -> dict[str, list[date]]:
    """Observed day of each row of an actual block, per location, from ``actual_dates_query``.

    The label IS the observed day, but the offset from the requested start depends on the location (measured
    2026-10-19, D-045): DE Country is labelled one day AFTER the requested start (the value of the request for
    2025-09-01 is labelled 09-02, and it lines up with WSI on 09-02); GB Country is labelled ON the requested start
    (and lines up with Heathrow on that day). So each location's first label must be ``start`` or ``start`` + 1, its
    labels must be strictly increasing, and they must end by ``end`` + 1. A gap is allowed: BQL can lack a day for
    some locations (2026-08-05 is missing for most of them, measured). ``missing_days`` reports it.
    The alignment check (pipelines/bql_weather.py) verifies every location against its WSI reference.
    """
    if "DATES" not in dates.columns or "ID" not in dates.columns or dates.empty:
        raise DataError(f"actual dates result without ID/DATES rows: {list(dates.columns)}")
    allowed = {start + timedelta(days=k) for k in range(ACTUAL_LABEL_MAX_SHIFT_DAYS + 1)}
    out: dict[str, list[date]] = {}
    for loc, grp in dates.groupby("ID", sort=False):
        labels = [pd.Timestamp(d).date() for d in grp["DATES"]]
        if labels[0] not in allowed:
            raise DataError(f"{loc}: actual labels start {labels[0]}, expected one of {sorted(allowed)}")
        if any(b <= a for a, b in zip(labels, labels[1:])):
            raise DataError(f"{loc}: actual labels are not strictly increasing")
        if labels[-1] > end + timedelta(days=ACTUAL_LABEL_MAX_SHIFT_DAYS):
            raise DataError(f"{loc}: actual labels run to {labels[-1]}, past the requested period end {end}")
        out[str(loc)] = labels
    return out


def missing_days(labels: Sequence[date]) -> list[date]:
    """Days inside the span of ``labels`` that BQL did not return (a gap is kept, not filled)."""
    have = set(labels)
    span = [labels[0] + timedelta(days=i) for i in range((labels[-1] - labels[0]).days + 1)]
    return [d for d in span if d not in have]


def observation_frame(df: pd.DataFrame, labels: Mapping[str, Sequence[date]],
                      ingestion_time: pd.Timestamp) -> pd.DataFrame:
    """OBSERVATION_SCHEMA rows from an actual_query result. Row i of a location = the observed day ``labels[loc][i]``.

    ``labels`` come from ``actual_event_dates`` (the dates-only query, per location). The value and label rows of a
    location come from the same result, so their counts must match; a mismatch is refused, never trimmed. (Before
    2026-10-19 the rows were stamped with the requested start + i: one day early.)
    """
    groups = {str(k): g.reset_index(drop=True) for k, g in df.groupby("ID", sort=False)}
    if not groups:
        raise ValueError("empty actual result")
    unknown = set(groups) - set(LOCATIONS)
    if unknown:
        raise ValueError(f"unknown location ids in result: {sorted(unknown)}")
    if set(groups) != set(labels):
        raise ValueError(f"values for {sorted(groups)} but labels for {sorted(labels)}")
    rows = []
    for bql_id, grp in groups.items():
        key, geo = LOCATIONS[bql_id]
        lab = labels[bql_id]
        if len(grp) != len(lab):
            raise ValueError(f"{bql_id}: {len(grp)} value rows but {len(lab)} observed-day labels")
        for field in OBSERVED_FIELDS:
            vals = _value_column(grp, field)
            events = [pd.Timestamp(d, tz="UTC") for d in lab]
            pub = [e + pd.Timedelta(days=OBS_PUBLICATION_DAYS, hours=OBS_PUBLICATION_HOUR_UTC) for e in events]
            n = len(events)
            rows.append(pd.DataFrame({
                "series_id": [series_id(key, field, observed=True)] * n,
                "event_time": events,
                "value": vals,
                "geography": [geo] * n,
                "publication_time": pub,
                "ingestion_time": [ingestion_time] * n,
                "revision": np.zeros(n, dtype="int64"),
                "unit": [FIELDS[field][1]] * n,
                "source": [SOURCE] * n,
                "quality_flag": ["ok"] * n,
                "entitlement": [ENTITLEMENT] * n,
                "pit_grade": [PitGrade.B.value] * n,
                "publication_time_estimated": [True] * n,
                "source_timezone": ["UTC"] * n,
            }))
    out = pd.concat(rows, ignore_index=True)
    return validate_frame(out, OBSERVATION_SCHEMA)


def normal_frame(df: pd.DataFrame, valid_dates: Sequence[date], source: str,
                 ingestion_time: pd.Timestamp) -> pd.DataFrame:
    """OBSERVATION_SCHEMA rows for normals. Row i of each location = ``valid_dates[i]`` (the dates-only audit)."""
    n = len(valid_dates)
    rows = []
    for bql_id, grp in _split_by_id(df, n).items():
        key, geo = LOCATIONS[bql_id]
        for field in OBSERVED_FIELDS:
            events = [pd.Timestamp(d, tz="UTC") for d in valid_dates]
            rows.append(pd.DataFrame({
                "series_id": [series_id(key, field, observed=False, normal=source)] * n,
                "event_time": events,
                "value": _value_column(grp, field),
                "geography": [geo] * n,
                "publication_time": [pd.Timestamp(d, tz="UTC") for d in valid_dates],
                "ingestion_time": [ingestion_time] * n,
                "revision": np.zeros(n, dtype="int64"),
                "unit": [FIELDS[field][1]] * n,
                "source": [SOURCE] * n,
                "quality_flag": ["ok"] * n,
                "entitlement": [ENTITLEMENT] * n,
                # C: a 10Y or 30Y normal computed today uses years after the event. Excluded from clean backtests.
                "pit_grade": [PitGrade.C.value] * n,
                "publication_time_estimated": [False] * n,
                "source_timezone": ["UTC"] * n,
            }))
    return validate_frame(pd.concat(rows, ignore_index=True), OBSERVATION_SCHEMA)


def iter_as_of(first: date, last: date) -> Iterable[date]:
    d = first
    while d <= last:
        yield d
        d += timedelta(days=1)


def batches(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
