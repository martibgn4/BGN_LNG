"""Canonical data contract: schemas, enums, adapter protocols and error types.

This module is the frozen contract every other module builds against (``docs/agent_plan.md``).
Changes go through the orchestrator and are recorded in ``docs/modelling_log.md``.

Design rules (``docs/point_in_time_policy.md``):

* Every stored fact is a bitemporal row: ``event_time`` (when it applies) and ``publication_time``
  (when it became knowable) are separate columns, both tz-aware UTC.
* Vendor identifiers never appear in these schemas; ``series_id`` is an internal name mapped to
  vendor identifiers in ``configs/sources/*.yaml``.
* Adapters return frames in these schemas and raise rather than return partial or invented data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------------------------

MARKET_TZ = "Europe/Amsterdam"
SYNTHETIC_SOURCE = "SYNTHETIC"


class QualityFlag(StrEnum):
    OK = "ok"
    ESTIMATED = "estimated"
    MISSING_INPUT = "missing_input"
    OUTLIER_SUSPECT = "outlier_suspect"
    SYNTHETIC = "synthetic"


class Entitlement(StrEnum):
    PUBLIC = "public"
    LICENSED_BLOOMBERG = "licensed_bloomberg"
    LICENSED_SPARK = "licensed_spark"
    LICENSED_OTHER = "licensed_other"
    SYNTHETIC = "synthetic"


class PitGrade(StrEnum):
    """A = true vintages; B = lag-approximated; C = contains hindsight (excluded from clean runs)."""

    A = "A"
    B = "B"
    C = "C"


class IdentifierStatus(StrEnum):
    """Status of a vendor identifier in configuration. Only ``KNOWN`` may be requested live."""

    KNOWN = "known"
    REQUIRES_DISCOVERY = "requires_discovery"
    REQUIRES_USER_INPUT = "requires_user_input"


class Backend(StrEnum):
    MOCK = "mock"
    LOCAL_FILE = "local_file"
    LIVE = "live"


PIT_GRADE_ORDER = {PitGrade.A: 0, PitGrade.B: 1, PitGrade.C: 2}


def worst_grade(grades: Sequence[str | PitGrade]) -> PitGrade:
    """Return the worst PIT grade (A < B < C). A feature inherits the worst of its inputs."""
    if not grades:
        raise ValueError("worst_grade() needs at least one grade")
    return max((PitGrade(g) for g in grades), key=PIT_GRADE_ORDER.__getitem__)


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------


class DataError(RuntimeError):
    """Base class for data-layer failures."""


class SchemaError(DataError):
    """A frame does not satisfy its declared schema. Message lists every violation."""


class PointInTimeViolation(DataError):
    """Data dated after the as-of time reached a code path that must not see it."""


class BackendUnavailable(DataError):
    """A live backend cannot run: package missing, credentials absent, or not authorised to call."""


class IdentifierUnknown(DataError):
    """A series has no confirmed vendor identifier (status != known)."""


class NonPositivePriceError(DataError):
    """A log transform was requested on a price <= 0."""


# ---------------------------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------------------------

_KINDS = ("str", "float", "int", "bool", "ts_utc")


@dataclass(frozen=True)
class Column:
    """One column of a schema. ``kind`` is one of ``str, float, int, bool, ts_utc``."""

    name: str
    kind: str
    nullable: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"Column {self.name}: unknown kind {self.kind!r}")


@dataclass(frozen=True)
class Schema:
    """An ordered column contract. Extra columns are allowed unless ``strict``."""

    name: str
    columns: tuple[Column, ...]
    strict: bool = False

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]


_PROVENANCE: tuple[Column, ...] = (
    Column("publication_time", "ts_utc"),
    Column("ingestion_time", "ts_utc"),
    Column("revision", "int"),
    Column("unit", "str"),
    Column("source", "str"),
    Column("quality_flag", "str"),
    Column("entitlement", "str"),
    Column("pit_grade", "str"),
    Column("publication_time_estimated", "bool"),
    Column("source_timezone", "str"),
)

PROVENANCE_COLUMNS: tuple[Column, ...] = _PROVENANCE
"""Public alias: the provenance columns every canonical schema carries."""

OBSERVATION_SCHEMA = Schema(
    "observation",
    (
        Column("series_id", "str"),
        Column("event_time", "ts_utc"),
        Column("value", "float", nullable=True),
        Column("geography", "str"),
        *_PROVENANCE,
    ),
)
"""Time series of facts: settlements, storage, flows, observed weather, assessments."""

FORECAST_SCHEMA = Schema(
    "forecast",
    (
        Column("series_id", "str"),
        Column("issue_time", "ts_utc"),
        Column("valid_time", "ts_utc"),
        Column("lead_hours", "float"),
        Column("model", "str"),
        Column("member", "str"),
        Column("value", "float", nullable=True),
        Column("geography", "str"),
        *_PROVENANCE,
    ),
)
"""Forecast vintages: weather runs, LNG arrival forecasts, storage projections.

``issue_time`` is the forecast's nominal creation time; ``publication_time`` is when it became
available to us/the market (>= issue_time). ``member`` is an ensemble member id or a statistic
label (``"mean"``, ``"spread"``, ``"p10"``...).
"""

CURVE_SCHEMA = Schema(
    "curve",
    (
        Column("hub", "str"),
        Column("contract_id", "str"),
        Column("tenor_type", "str"),
        Column("delivery_start", "ts_utc"),
        Column("delivery_end", "ts_utc"),
        Column("trade_date", "ts_utc"),
        Column("field", "str"),
        Column("value", "float", nullable=True),
        Column("currency", "str"),
        *_PROVENANCE,
    ),
)
"""Forward-curve snapshots. ``trade_date`` is local midnight of the trading date in UTC;
``delivery_end`` is exclusive. ``field`` in {settle, bid, ask, volume, open_interest}."""

CONTRACT_METADATA_SCHEMA = Schema(
    "contract_metadata",
    (
        Column("contract_id", "str"),
        Column("hub", "str"),
        Column("tenor_type", "str"),
        Column("delivery_start", "ts_utc"),
        Column("delivery_end", "ts_utc"),
        Column("last_trade_date", "ts_utc"),
        Column("last_trade_date_source", "str"),  # "rule" or "vendor"
        Column("currency", "str"),
        Column("unit", "str"),
    ),
)

TARGET_SCHEMA = Schema(
    "target",
    (
        Column("origin_time", "ts_utc"),
        Column("target_id", "str"),
        Column("horizon", "int"),
        Column("value", "float", nullable=True),
        Column("label_known_time", "ts_utc"),
        Column("contract_id", "str", nullable=True),
        Column("construction", "str"),
        Column("quality_flag", "str"),
    ),
)
"""One row per (origin, target, horizon). ``label_known_time`` drives purging."""

PREDICTION_SCHEMA = Schema(
    "prediction",
    (
        Column("origin_time", "ts_utc"),
        Column("target_id", "str"),
        Column("horizon", "int"),
        Column("model_id", "str"),
        Column("prediction", "float", nullable=True),
        Column("train_end", "ts_utc"),
        Column("label_known_time", "ts_utc"),
        Column("fold_id", "str"),
        Column("feature_hash", "str"),
        Column("is_synthetic", "bool"),
    ),
)
"""Optional extra columns ``q<level>`` (e.g. ``q0.1``) for quantile forecasts and ``prob`` for
classification probabilities."""


TENOR_TYPES = ("DA", "WE", "BOM", "M", "Q", "S", "Y")


def contract_id(hub: str, tenor_type: str, label: str) -> str:
    """Canonical internal contract id, e.g. ``TTF.M.2026-11``, ``TTF.Q.2027-Q1``,
    ``TTF.S.2027-SUM``, ``TTF.S.2026-WIN`` (Winter starting Oct 2026), ``TTF.Y.2027``."""
    if tenor_type not in TENOR_TYPES:
        raise ValueError(f"tenor_type {tenor_type!r} not in {TENOR_TYPES}")
    return f"{hub}.{tenor_type}.{label}"


def validate_frame(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Check ``df`` against ``schema``; return it unchanged or raise ``SchemaError`` listing
    every problem (missing columns, wrong dtypes, naive timestamps, unexpected nulls).

    Timestamps must be tz-aware **UTC**. Naive or non-UTC timestamps are an error rather than being
    converted silently, because a silent conversion is exactly how timezone leakage happens.
    """
    problems: list[str] = []
    missing = [c.name for c in schema.columns if c.name not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    if schema.strict:
        extra = [c for c in df.columns if c not in schema.names]
        if extra:
            problems.append(f"unexpected columns {extra}")
    for col in schema.columns:
        if col.name not in df.columns:
            continue
        s = df[col.name]
        if not col.nullable and s.isna().any():
            problems.append(f"{col.name}: {int(s.isna().sum())} nulls in non-nullable column")
        if col.kind == "ts_utc":
            if not isinstance(s.dtype, pd.DatetimeTZDtype):
                problems.append(f"{col.name}: expected tz-aware UTC datetime, got {s.dtype}")
            elif str(s.dtype.tz) != "UTC":
                problems.append(f"{col.name}: expected UTC, got {s.dtype.tz}")
        elif col.kind == "float" and not pd.api.types.is_float_dtype(s):
            problems.append(f"{col.name}: expected float, got {s.dtype}")
        elif col.kind == "int" and not pd.api.types.is_integer_dtype(s):
            problems.append(f"{col.name}: expected integer, got {s.dtype}")
        elif col.kind == "bool" and not pd.api.types.is_bool_dtype(s):
            problems.append(f"{col.name}: expected bool, got {s.dtype}")
        elif col.kind == "str" and not (
            pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
        ):
            problems.append(f"{col.name}: expected string, got {s.dtype}")
    if "pit_grade" in df.columns and len(df):
        bad = set(df["pit_grade"].dropna().unique()) - {g.value for g in PitGrade}
        if bad:
            problems.append(f"pit_grade: invalid values {sorted(bad)}")
    # publication_time < event_time is legitimate for observations (e.g. a published schedule), so
    # only the forecast ordering issue_time <= publication_time is enforced here.
    if {"publication_time", "issue_time"} <= set(df.columns) and len(df):
        early = (df["publication_time"] < df["issue_time"]).sum()
        if early:
            problems.append(f"{int(early)} rows with publication_time < issue_time")
    if problems:
        raise SchemaError(f"{schema.name} schema violations: " + "; ".join(problems))
    return df


def empty_frame(schema: Schema) -> pd.DataFrame:
    """An empty frame with the schema's columns and dtypes (useful as a typed 'no data')."""
    dtypes = {
        "str": "object",
        "float": "float64",
        "int": "int64",
        "bool": "bool",
        "ts_utc": "datetime64[ns, UTC]",
    }
    return pd.DataFrame({c.name: pd.Series(dtype=dtypes[c.kind]) for c in schema.columns})


# ---------------------------------------------------------------------------------------------
# Publication-lag rules (grade-B series): publication_time derived from event_time
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicationLag:
    """``publication_time = local(event_date + days) at time`` in ``tz``.

    ``business_days=True`` counts weekdays only. Configured per series as
    ``publication_lag: {days: 2, time: "07:00", tz: "Europe/Amsterdam", business_days: false}``.
    """

    days: int
    time: str = "00:00"
    tz: str = MARKET_TZ
    business_days: bool = False

    @classmethod
    def from_config(cls, cfg: dict[str, object]) -> PublicationLag:
        return cls(
            days=int(cfg["days"]),  # type: ignore[call-overload]
            time=str(cfg.get("time", "00:00")),
            tz=str(cfg.get("tz", MARKET_TZ)),
            business_days=bool(cfg.get("business_days", False)),
        )


def apply_publication_lag(event_time: pd.Series, lag: PublicationLag) -> pd.Series:
    """Publication times (UTC) for tz-aware UTC ``event_time`` under ``lag``.

    The event's **local** date in ``lag.tz`` is shifted by ``lag.days`` (calendar or business days)
    and set to ``lag.time`` local, then converted to UTC. DST is handled by the zone database.
    """
    if not isinstance(event_time.dtype, pd.DatetimeTZDtype):
        raise SchemaError("apply_publication_lag needs tz-aware event_time")
    local_date = event_time.dt.tz_convert(lag.tz).dt.tz_localize(None).dt.normalize()
    offset = pd.offsets.BDay(lag.days) if lag.business_days else pd.Timedelta(days=lag.days)
    hh, mm = (int(x) for x in lag.time.split(":"))
    shifted = (local_date + offset) + pd.Timedelta(hours=hh, minutes=mm)
    # Ambiguous wall times (repeated hour at the autumn DST change) resolve to the LATER, standard-time
    # instant: the conservative availability. Non-existent times (spring) shift forward.
    later = np.zeros(len(shifted), dtype=bool)
    return shifted.dt.tz_localize(lag.tz, ambiguous=later, nonexistent="shift_forward").dt.tz_convert(
        "UTC"
    )


# ---------------------------------------------------------------------------------------------
# Data bundle and feature-builder contract
# ---------------------------------------------------------------------------------------------


@dataclass
class DataBundle:
    """All vintages available to a feature build. Builders must apply the as-of rule themselves.

    Attributes:
        observations: OBSERVATION_SCHEMA rows, all series, all revisions.
        forecasts: FORECAST_SCHEMA rows, all runs.
        curves: CURVE_SCHEMA rows, all hubs/contracts.
        contracts: CONTRACT_METADATA_SCHEMA rows.
    """

    observations: pd.DataFrame
    forecasts: pd.DataFrame
    curves: pd.DataFrame
    contracts: pd.DataFrame

    @classmethod
    def empty(cls) -> DataBundle:
        return cls(
            empty_frame(OBSERVATION_SCHEMA),
            empty_frame(FORECAST_SCHEMA),
            empty_frame(CURVE_SCHEMA),
            empty_frame(CONTRACT_METADATA_SCHEMA),
        )


class FeatureBuilder(Protocol):
    """Signature every feature family exposes as ``build_<family>_features``.

    Returns a frame indexed by ``origin_time`` (the as-of timestamps passed in, tz-aware UTC) with
    columns named ``<family>__<feature>``. ``frame.attrs["pit_grade"]`` maps each column to the
    worst PIT grade of its inputs. A value must be computable from data with
    ``publication_time <= origin_time``; unavailable values are NaN, never 0.
    """

    def __call__(
        self, bundle: DataBundle, origins: pd.DatetimeIndex, params: dict[str, object]
    ) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------------------------
# Adapter protocols. Internal interface names, not vendor API names.
# ---------------------------------------------------------------------------------------------


@runtime_checkable
class MarketDataAdapter(Protocol):
    """Market prices and curves (Bloomberg or replacement). Returns canonical frames."""

    backend: Backend

    def get_historical_series(
        self, series_ids: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        """OBSERVATION_SCHEMA rows for internal ``series_ids`` over [start, end] (event dates)."""
        ...

    def get_reference_data(
        self, instrument_ids: Sequence[str], fields: Sequence[str]
    ) -> pd.DataFrame:
        """Long frame: ``instrument_id, field, value`` (static/reference attributes)."""
        ...

    def get_curve_snapshot(
        self, hub: str, start: date, end: date, fields: Sequence[str] = ("settle",)
    ) -> pd.DataFrame:
        """CURVE_SCHEMA rows for every listed contract of ``hub`` on trading dates in [start, end]."""
        ...

    def get_contract_metadata(self, contract_ids: Sequence[str]) -> pd.DataFrame:
        """CONTRACT_METADATA_SCHEMA rows."""
        ...


@runtime_checkable
class LngDataAdapter(Protocol):
    """LNG shipping and assessments (Spark or replacement).

    Backends may add ``get_realised_arrivals`` and ``get_destination_statistics`` (see
    ``data/spark.py``); they are deliberately not part of the protocol because the live backend has no
    known vendor dataset for them yet (status requires_user_input).
    """

    backend: Backend

    def get_lng_vessels(self, start: date, end: date) -> pd.DataFrame:
        """Vessel snapshot rows (schema owned by data/spark.py; must carry publication_time)."""
        ...

    def get_lng_arrivals(self, start: date, end: date) -> pd.DataFrame:
        """FORECAST_SCHEMA rows: arrival forecasts by terminal/region with their issue times."""
        ...

    def get_freight_assessments(
        self, series_ids: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        """OBSERVATION_SCHEMA rows."""
        ...

    def get_gas_assessments(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """OBSERVATION_SCHEMA rows."""
        ...

    def get_route_economics(
        self, series_ids: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        """OBSERVATION_SCHEMA rows (netbacks, route costs, arbitrage)."""
        ...


@runtime_checkable
class FundamentalsAdapter(Protocol):
    """European gas fundamentals: storage, flows, send-out, demand."""

    backend: Backend

    def get_series(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """OBSERVATION_SCHEMA rows."""
        ...


@runtime_checkable
class WeatherAdapter(Protocol):
    """Weather forecasts (vintaged) and observations."""

    backend: Backend

    def get_forecast_runs(
        self, series_ids: Sequence[str], issue_start: date, issue_end: date
    ) -> pd.DataFrame:
        """FORECAST_SCHEMA rows for runs issued in [issue_start, issue_end]."""
        ...

    def get_observed(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """OBSERVATION_SCHEMA rows (actuals; never a forecast substitute)."""
        ...
