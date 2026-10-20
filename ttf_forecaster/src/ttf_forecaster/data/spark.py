"""Spark Commodities LNG adapter, vessel-snapshot schema and point-in-time arrival forecasts.

Three backends implement :class:`~ttf_forecaster.data.base.LngDataAdapter`:

* :class:`MockSparkAdapter` — deterministic SYNTHETIC world (vessel voyages with declared
  destinations/ETAs that drift and divert, release-dated assessments). For tests only.
* :class:`LocalFileSparkAdapter` — user-placed extracts (Parquet/CSV) already in canonical schemas.
* :class:`LiveSparkAdapter` — a skeleton that validates configuration, authorisation and
  credentials, builds requests for identifiers whose status is ``known`` and parses their payloads.
  It refuses to run unless ``live_calls_authorised`` is true. Vessel and arrival requests always
  raise :class:`IdentifierUnknown`: no Spark vessel / AIS / arrival dataset is known in this repo.

Point-in-time design (``docs/point_in_time_policy.md``):

* Vessel snapshots (:data:`VESSEL_SNAPSHOT_SCHEMA`) record a vessel's state **as declared at**
  ``snapshot_time`` and carry the ``publication_time`` at which that snapshot became available.
* :func:`build_arrival_forecasts` uses, for each issue time ``t``, only snapshots with
  ``publication_time <= t`` and the destinations / ETAs declared in them. It emits
  FORECAST_SCHEMA rows ``lng.arrivals.<geo>.cargoes_<h>d`` and ``lng.arrivals.<geo>.volume_<h>d_<unit>``.
* Realised arrivals are stored separately as observations (:func:`realised_arrival_observations`)
  so forecast errors can be computed once the realisation is published — never used as inputs.
* The PIT grade of every derived row is the worst grade of the snapshots it used: grade C unless
  the snapshot history is genuinely vintaged.

Units are explicit: arrival counts are cargoes; volumes keep the snapshot's unit (no silent
m3 -> t -> MMBtu conversion: LNG density and GCV are cargo-specific).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from base64 import b64encode
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    MARKET_TZ,
    OBSERVATION_SCHEMA,
    SYNTHETIC_SOURCE,
    Backend,
    BackendUnavailable,
    Column,
    DataError,
    Entitlement,
    IdentifierStatus,
    IdentifierUnknown,
    PitGrade,
    PublicationLag,
    QualityFlag,
    Schema,
    SchemaError,
    apply_publication_lag,
    empty_frame,
    validate_frame,
    worst_grade,
)
from ttf_forecaster.settings import (
    PROJECT_ROOT,
    ConfigError,
    SecretMissingError,
    get_secret,
    load_config,
)

log = logging.getLogger(__name__)

SPARK_CONFIG_NAME = "sources/spark"
SOURCE_LABEL = "spark"
ARRIVAL_HORIZONS_DAYS: tuple[int, ...] = (7, 14, 30, 45)
ARRIVAL_MODEL = "declared_eta"
LONDON_TZ = "Europe/London"


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------


class EmptyPayloadError(DataError):
    """A vendor call returned HTTP success with no data (e.g. netbacks without ``via-point``)."""


class ViaPointMissing(DataError):
    """A netback request was attempted for a FOB port without a configured ``via_point``."""


# ---------------------------------------------------------------------------------------------
# Schemas owned by this module
# ---------------------------------------------------------------------------------------------

_PROVENANCE_NAMES = (
    "publication_time", "ingestion_time", "revision", "unit", "source", "quality_flag",
    "entitlement", "pit_grade", "publication_time_estimated", "source_timezone",
)
_PROV_COLUMNS: tuple[Column, ...] = tuple(
    c for c in OBSERVATION_SCHEMA.columns if c.name in _PROVENANCE_NAMES
)

VESSEL_SNAPSHOT_SCHEMA = Schema(
    "lng_vessel_snapshot",
    (
        Column("snapshot_time", "ts_utc"),          # when the vessel state below was true/declared
        Column("vessel_id", "str"),                 # vendor-neutral id (e.g. IMO as string)
        Column("voyage_id", "str", nullable=True),  # optional; distinguishes laden legs
        Column("laden", "bool"),                    # True laden, False ballast
        Column("origin", "str", nullable=True),     # load country/terminal code
        Column("load_time", "ts_utc", nullable=True),
        Column("declared_destination", "str", nullable=True),         # terminal code as declared
        Column("declared_destination_region", "str", nullable=True),  # vendor region label
        Column("declared_eta", "ts_utc", nullable=True),
        Column("cargo_volume", "float", nullable=True),
        Column("cargo_volume_unit", "str"),         # "m3_LNG", "t_LNG" or "MMBtu"
        *_PROV_COLUMNS,                             # publication_time = snapshot availability
    ),
)
"""One row per (vessel, snapshot). ``publication_time`` (>= ``snapshot_time``) is when the snapshot
could first have been read. ``unit`` (provenance) duplicates ``cargo_volume_unit`` for
compatibility with the store. Destinations/ETAs are AS DECLARED at ``snapshot_time``."""

REALISED_ARRIVAL_SCHEMA = Schema(
    "lng_realised_arrival",
    (
        Column("vessel_id", "str"),
        Column("arrival_time", "ts_utc"),
        Column("terminal", "str"),
        Column("cargo_volume", "float", nullable=True),
        Column("cargo_volume_unit", "str"),
        *_PROV_COLUMNS,                             # publication_time = when the discharge was known
    ),
)
"""Event-level realised discharges. Converted to daily OBSERVATION rows by
:func:`realised_arrival_observations`; never an input to a forecast."""


def _validate_snapshot_extras(df: pd.DataFrame) -> None:
    problems: list[str] = []
    if len(df) and (df["publication_time"] < df["snapshot_time"]).any():
        problems.append("publication_time < snapshot_time")
    if problems:
        raise SchemaError("lng_vessel_snapshot: " + "; ".join(problems))


def validate_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a vessel-snapshot frame (schema + publication after snapshot)."""
    validate_frame(df, VESSEL_SNAPSHOT_SCHEMA)
    _validate_snapshot_extras(df)
    return df


# ---------------------------------------------------------------------------------------------
# Series ids
# ---------------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def arrival_series_id(geo: str, horizon_days: int, measure: str = "cargoes",
                      volume_unit: str | None = None) -> str:
    """``lng.arrivals.<geo>.cargoes_<h>d`` or ``lng.arrivals.<geo>.volume_<h>d_<unit>``."""
    if measure == "cargoes":
        return f"lng.arrivals.{_slug(geo)}.cargoes_{int(horizon_days)}d"
    if measure == "volume":
        if not volume_unit:
            raise ValueError("volume series need a volume_unit")
        return f"lng.arrivals.{_slug(geo)}.volume_{int(horizon_days)}d_{_slug(volume_unit)}"
    raise ValueError(f"unknown arrival measure {measure!r}")


def realised_series_id(geo: str, measure: str = "cargoes", volume_unit: str | None = None) -> str:
    """Daily realised arrivals: ``lng.arrivals_realised.<geo>.cargoes_count`` / ``.volume_<unit>``."""
    if measure == "cargoes":
        return f"lng.arrivals_realised.{_slug(geo)}.cargoes_count"
    if not volume_unit:
        raise ValueError("volume series need a volume_unit")
    return f"lng.arrivals_realised.{_slug(geo)}.volume_{_slug(volume_unit)}"


def diversion_series_id(region: str, direction: str, lookback_days: int) -> str:
    """``lng.diversions.<region>.<into|away>_count_<L>d``."""
    if direction not in ("into", "away"):
        raise ValueError("direction must be 'into' or 'away'")
    return f"lng.diversions.{_slug(region)}.{direction}_count_{int(lookback_days)}d"


UNDECLARED_SHARE_SERIES = "lng.destination.global.undeclared_share"


# ---------------------------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------------------------


def load_spark_config(config_dir: str | None = None) -> dict[str, Any]:
    """``configs/sources/spark.yaml`` (merged with a git-ignored local override if present)."""
    return load_config(SPARK_CONFIG_NAME, config_dir)


def _template_regex(template: str) -> re.Pattern[str]:
    """Turn ``a.{port}.{field}_mi{n}_x`` into a regex with named groups. Numeric placeholders
    ``k``/``n`` match digits; others match ``[a-z0-9_]+?`` without dots."""
    out, pos = "", 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out += re.escape(template[pos:m.start()])
        name = m.group(1)
        out += rf"(?P<{name}>\d+)" if name in ("k", "n") else rf"(?P<{name}>[a-z0-9_]+?)"
        pos = m.end()
    out += re.escape(template[pos:])
    return re.compile(f"^{out}$")


@dataclass(frozen=True)
class SeriesSpec:
    """A resolved series: its config entry and any template parameters (``k``, ``port``...)."""

    series_id: str
    entry: Mapping[str, Any]
    params: Mapping[str, str]

    @property
    def status(self) -> IdentifierStatus:
        return IdentifierStatus(str(self.entry.get("status", "requires_user_input")))

    @property
    def kind(self) -> str:
        return str(self.entry.get("kind", "price_release"))

    @property
    def grade(self) -> PitGrade:
        return PitGrade(str(self.entry.get("pit_grade", "C")))

    @property
    def lag(self) -> PublicationLag:
        cfg = self.entry.get("publication_lag")
        if not isinstance(cfg, Mapping):
            raise ConfigError(f"{self.series_id}: publication_lag missing in spark config")
        return PublicationLag.from_config(dict(cfg))


def resolve_series(config: Mapping[str, Any], series_id: str) -> SeriesSpec:
    """Find ``series_id`` among explicit ``series`` or ``series_templates``.

    Raises:
        IdentifierUnknown: the id matches nothing in the Spark config.
    """
    explicit = config.get("series") or {}
    if series_id in explicit:
        return SeriesSpec(series_id, explicit[series_id], {})
    for template, entry in (config.get("series_templates") or {}).items():
        m = _template_regex(template).match(series_id)
        if m:
            params = m.groupdict()
            if entry.get("kind") == "netback":
                fields = (config.get("netbacks") or {}).get("fields") or {}
                ports = (config.get("netbacks") or {}).get("fob_ports") or {}
                if params["field"] not in fields or params["port"] not in ports:
                    continue
            if entry.get("kind") == "terminal_slots":
                buckets = (config.get("terminal_slots") or {}).get("buckets") or {}
                if params["bucket"] not in buckets:
                    continue
            return SeriesSpec(series_id, entry, params)
    raise IdentifierUnknown(
        f"Series {series_id!r} is not defined in configs/sources/spark.yaml (series or "
        "series_templates). Add it with a status and provenance; never guess a dataset id."
    )


def _release_event_time(release_dates: Iterable[Any]) -> pd.DatetimeIndex:
    """Release date -> local (London) midnight in UTC. Spark releases are London-dated."""
    d = pd.to_datetime(pd.Series(list(release_dates))).dt.tz_localize(None).dt.normalize()
    return pd.DatetimeIndex(d).tz_localize(LONDON_TZ).tz_convert("UTC")


def _observation_frame(
    series_id: str,
    event_time: pd.DatetimeIndex,
    values: Sequence[float] | np.ndarray,
    *,
    unit: str,
    geography: str,
    publication_time: pd.DatetimeIndex,
    ingestion_time: pd.DatetimeIndex,
    source: str,
    quality: QualityFlag,
    entitlement: Entitlement,
    grade: PitGrade,
    estimated: bool,
    source_tz: str,
) -> pd.DataFrame:
    n = len(event_time)
    return pd.DataFrame(
        {
            "series_id": [series_id] * n,
            "event_time": pd.DatetimeIndex(event_time),
            "value": np.asarray(values, dtype="float64"),
            "geography": [geography] * n,
            "publication_time": pd.DatetimeIndex(publication_time),
            "ingestion_time": pd.DatetimeIndex(ingestion_time),
            "revision": np.zeros(n, dtype="int64"),
            "unit": [unit] * n,
            "source": [source] * n,
            "quality_flag": [quality.value] * n,
            "entitlement": [entitlement.value] * n,
            "pit_grade": [grade.value] * n,
            "publication_time_estimated": np.full(n, estimated, dtype=bool),
            "source_timezone": [source_tz] * n,
        }
    )


def _date_mask(ts: pd.Series, start: date, end: date, tz: str = LONDON_TZ) -> pd.Series:
    local = ts.dt.tz_convert(tz).dt.date
    return (local >= start) & (local <= end)


# ---------------------------------------------------------------------------------------------
# PIT arrival forecasts from vessel snapshots
# ---------------------------------------------------------------------------------------------


def _regions_for(dest: Any, dest_region: Any, terminal_regions: Mapping[str, Sequence[str]]
                 ) -> list[str]:
    if isinstance(dest, str) and dest in terminal_regions:
        return [str(r) for r in terminal_regions[dest]]
    if isinstance(dest_region, str) and dest_region:
        return [dest_region]
    return []


def _latest_state(snapshots: pd.DataFrame, as_of: pd.Timestamp, max_age: pd.Timedelta,
                  snapshot_cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Latest visible snapshot per vessel at ``as_of`` (publication_time <= as_of), dropping
    vessels whose latest snapshot is older than ``max_age`` (they may have discharged unseen)."""
    vis = snapshots.loc[snapshots["publication_time"] <= as_of]
    if snapshot_cutoff is not None:
        vis = vis.loc[vis["snapshot_time"] <= snapshot_cutoff]
    if vis.empty:
        return vis
    vis = vis.sort_values(["vessel_id", "snapshot_time", "publication_time", "revision"])
    last = vis.drop_duplicates("vessel_id", keep="last")
    ref = snapshot_cutoff if snapshot_cutoff is not None else as_of
    return last.loc[(ref - last["snapshot_time"]) <= max_age]


def _forecast_rows(
    series_id: str, issue: pd.Timestamp, valid: pd.Timestamp, horizon_days: int, value: float,
    geography: str, unit: str, ingestion: pd.Timestamp, grade: PitGrade, source: str,
    quality: str, entitlement: str,
) -> dict[str, Any]:
    return {
        "series_id": series_id, "issue_time": issue, "valid_time": valid,
        "lead_hours": float(horizon_days * 24), "model": ARRIVAL_MODEL, "member": "point",
        "value": float(value), "geography": geography, "publication_time": issue,
        "ingestion_time": max(ingestion, issue), "revision": 0, "unit": unit, "source": source,
        "quality_flag": quality, "entitlement": entitlement, "pit_grade": grade.value,
        "publication_time_estimated": False, "source_timezone": "UTC",
    }


def _common_label(series: pd.Series, default: str) -> str:
    vals = series.dropna().unique()
    return str(vals[0]) if len(vals) == 1 else default


def build_arrival_forecasts(
    snapshots: pd.DataFrame,
    issue_times: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    terminal_regions: Mapping[str, Sequence[str]],
    regions: Sequence[str],
    horizons_days: Sequence[int] = ARRIVAL_HORIZONS_DAYS,
    volume_unit: str = "m3_LNG",
    max_snapshot_age: pd.Timedelta = pd.Timedelta(days=3),
    include_terminals: bool = False,
) -> pd.DataFrame:
    """PIT expected arrivals by region (and optionally terminal) and horizon.

    For each issue time ``t`` only snapshots with ``publication_time <= t`` are used, and each
    vessel contributes its **latest declared** destination and ETA as of ``t``. A laden vessel is
    counted in horizon ``h`` when its declared ETA falls in the window of UTC calendar days
    ``[floor(t) + 1d, floor(t) + 1d + h)``; ``valid_time`` is the (exclusive) window end. The same
    day-window is used for realised arrivals, so errors are comparable.

    Args:
        snapshots: VESSEL_SNAPSHOT_SCHEMA frame (all snapshot vintages).
        issue_times: tz-aware UTC timestamps at which forecasts are issued (e.g. TTF origins).
        terminal_regions: declared-destination terminal code -> region labels (e.g. NWE, EU).
        regions: region labels to emit (zero-count rows are emitted for them).
        horizons_days: window lengths.
        volume_unit: unit all snapshot volumes must carry; mixed units raise (no silent density).
        max_snapshot_age: a vessel whose latest visible snapshot is older is ignored.
        include_terminals: also emit one series per declared terminal seen.

    Returns:
        FORECAST_SCHEMA rows. ``pit_grade`` = worst grade among visible snapshots used at ``t``.
        No rows for an issue time with no fresh visible snapshot (unknown, not zero).
    """
    validate_snapshots(snapshots)
    units = set(snapshots["cargo_volume_unit"].dropna().unique())
    if units - {volume_unit}:
        raise SchemaError(
            f"snapshot volumes in {sorted(units)} but volume_unit={volume_unit!r}; convert "
            "explicitly with a stated density/GCV before building arrival forecasts"
        )
    issues = pd.DatetimeIndex(issue_times)
    if issues.tz is None or str(issues.tz) != "UTC":
        raise SchemaError("issue_times must be tz-aware UTC")
    rows: list[dict[str, Any]] = []
    source = _common_label(snapshots["source"], SOURCE_LABEL)
    quality = _common_label(snapshots["quality_flag"], QualityFlag.OK.value)
    entitlement = _common_label(snapshots["entitlement"], Entitlement.LICENSED_OTHER.value)
    for t in issues:
        state = _latest_state(snapshots, t, max_snapshot_age)
        if state.empty:
            continue
        grade = worst_grade(state["pit_grade"].tolist())
        ingestion = state["ingestion_time"].max()
        laden = state.loc[state["laden"] & state["declared_eta"].notna()]
        day0 = t.floor("D") + pd.Timedelta(days=1)
        geo_lists = [
            _regions_for(d, r, terminal_regions)
            for d, r in zip(laden["declared_destination"], laden["declared_destination_region"],
                            strict=True)
        ]
        for h in horizons_days:
            end = day0 + pd.Timedelta(days=int(h))
            in_win = ((laden["declared_eta"] >= day0) & (laden["declared_eta"] < end)).to_numpy()
            counts: dict[str, float] = {r: 0.0 for r in regions}
            vols: dict[str, float] = {r: 0.0 for r in regions}
            term_counts: dict[str, float] = {}
            term_vols: dict[str, float] = {}
            for flag, geos, dest, vol in zip(in_win, geo_lists, laden["declared_destination"],
                                             laden["cargo_volume"], strict=True):
                if not flag:
                    continue
                v = 0.0 if pd.isna(vol) else float(vol)
                for g in geos:
                    if g in counts:
                        counts[g] += 1.0
                        vols[g] += v
                if include_terminals and isinstance(dest, str):
                    term_counts[dest] = term_counts.get(dest, 0.0) + 1.0
                    term_vols[dest] = term_vols.get(dest, 0.0) + v
            if include_terminals:
                for term in terminal_regions:
                    term_counts.setdefault(term, 0.0)
                    term_vols.setdefault(term, 0.0)
            for geo, c in [*counts.items(), *term_counts.items()]:
                vol_total = vols.get(geo, term_vols.get(geo, 0.0))
                rows.append(_forecast_rows(arrival_series_id(geo, h), t, end, h, c, geo,
                                           "cargoes", ingestion, grade, source, quality,
                                           entitlement))
                rows.append(_forecast_rows(arrival_series_id(geo, h, "volume", volume_unit), t,
                                           end, h, vol_total, geo, volume_unit, ingestion, grade,
                                           source, quality, entitlement))
    if not rows:
        return empty_frame(FORECAST_SCHEMA)
    out = pd.DataFrame(rows)
    for col in ("issue_time", "valid_time", "publication_time", "ingestion_time"):
        out[col] = pd.to_datetime(out[col], utc=True)
    out["revision"] = out["revision"].astype("int64")
    out["publication_time_estimated"] = out["publication_time_estimated"].astype(bool)
    return validate_frame(out, FORECAST_SCHEMA)


def destination_observations(
    snapshots: pd.DataFrame,
    issue_times: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    terminal_regions: Mapping[str, Sequence[str]],
    focus_region: str = "EU",
    lookback_days: int = 14,
    max_snapshot_age: pd.Timedelta = pd.Timedelta(days=3),
) -> pd.DataFrame:
    """Diversion counts and destination uncertainty as of each issue time (OBSERVATION rows).

    * ``lng.diversions.<focus>.away_count_<L>d``: laden vessels whose declared destination was in
      ``focus_region`` ``L`` days ago and is now declared outside it (unknown excluded).
    * ``...into_count_<L>d``: the reverse.
    * ``lng.destination.global.undeclared_share``: share of fresh laden vessels with no mappable
      declared destination.

    Both "then" and "now" states come from snapshots published by ``t``. ``event_time`` =
    ``publication_time`` = ``t``. Grade = worst grade of snapshots used (C unless vintaged).
    """
    validate_snapshots(snapshots)
    issues = pd.DatetimeIndex(issue_times)
    lb = pd.Timedelta(days=int(lookback_days))
    records: list[tuple[str, pd.Timestamp, float, str, pd.Timestamp, str]] = []
    for t in issues:
        now = _latest_state(snapshots, t, max_snapshot_age)
        now = now.loc[now["laden"]] if not now.empty else now
        if now.empty:
            continue
        then = _latest_state(snapshots, t, max_snapshot_age, snapshot_cutoff=t - lb)
        grades = now["pit_grade"].tolist() + (then["pit_grade"].tolist() if len(then) else [])
        grade = worst_grade(grades).value
        ingestion = max(now["ingestion_time"].max(), t)

        def _in_focus(row: pd.Series) -> bool | None:
            geos = _regions_for(row["declared_destination"], row["declared_destination_region"],
                                terminal_regions)
            return None if not geos else focus_region in geos

        now_focus = {v: _in_focus(r) for v, r in now.set_index("vessel_id").iterrows()}
        undeclared = sum(1 for f in now_focus.values() if f is None) / len(now_focus)
        into = away = 0
        if len(then):
            then_l = then.loc[then["laden"]].set_index("vessel_id")
            for v, r in then_l.iterrows():
                if v not in now_focus:
                    continue
                if "voyage_id" in r and isinstance(r["voyage_id"], str):
                    cur_voy = now.loc[now["vessel_id"] == v, "voyage_id"].iloc[0]
                    if isinstance(cur_voy, str) and cur_voy != r["voyage_id"]:
                        continue
                before, after = _in_focus(r), now_focus[v]
                if before is None or after is None:
                    continue
                into += int((not before) and after)
                away += int(before and (not after))
        records += [
            (diversion_series_id(focus_region, "into", lookback_days), t, float(into), "cargoes",
             ingestion, grade),
            (diversion_series_id(focus_region, "away", lookback_days), t, float(away), "cargoes",
             ingestion, grade),
            (UNDECLARED_SHARE_SERIES, t, float(undeclared), "fraction", ingestion, grade),
        ]
    if not records:
        return empty_frame(OBSERVATION_SCHEMA)
    df = pd.DataFrame(records, columns=["series_id", "event_time", "value", "unit",
                                        "ingestion_time", "pit_grade"])
    n = len(df)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["publication_time"] = df["event_time"]
    df["ingestion_time"] = pd.to_datetime(df["ingestion_time"], utc=True)
    df["geography"] = [focus_region if s != UNDECLARED_SHARE_SERIES else "GLOBAL"
                       for s in df["series_id"]]
    df["revision"] = np.zeros(n, dtype="int64")
    df["source"] = [_common_label(snapshots["source"], SOURCE_LABEL)] * n
    df["quality_flag"] = [_common_label(snapshots["quality_flag"], QualityFlag.OK.value)] * n
    df["entitlement"] = [_common_label(snapshots["entitlement"],
                                       Entitlement.LICENSED_OTHER.value)] * n
    df["publication_time_estimated"] = np.zeros(n, dtype=bool)
    df["source_timezone"] = ["UTC"] * n
    return validate_frame(df[OBSERVATION_SCHEMA.names], OBSERVATION_SCHEMA)


def realised_arrival_observations(
    events: pd.DataFrame,
    *,
    terminal_regions: Mapping[str, Sequence[str]],
    regions: Sequence[str],
    start: date,
    end: date,
    volume_unit: str = "m3_LNG",
    publication_lag: PublicationLag | None = None,
) -> pd.DataFrame:
    """Daily realised arrivals per region as OBSERVATION rows (zero-filled over [start, end]).

    ``event_time`` is the UTC calendar day of discharge. ``publication_time`` is the latest
    ``publication_time`` of the events that day, or — when ``publication_lag`` is given — the lag
    rule applied to the day (then ``publication_time_estimated``). A zero day gets the same lag
    rule (or day end + the median observed delay if no rule). The grade of each row is the worst
    grade of its events (zero days: worst grade in the frame).

    These rows are the **realisation**; features may use them only to measure past forecast error
    once published, never as a forecast input (PIT policy §5.1).
    """
    validate_frame(events, REALISED_ARRIVAL_SCHEMA)
    units = set(events["cargo_volume_unit"].dropna().unique())
    if units - {volume_unit}:
        raise SchemaError(f"realised volumes in {sorted(units)} but volume_unit={volume_unit!r}")
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", tz="UTC")
    ev = events.copy()
    ev["day"] = ev["arrival_time"].dt.floor("D")
    ev = ev.loc[(ev["day"] >= days[0]) & (ev["day"] <= days[-1])]
    worst_all = worst_grade(events["pit_grade"].tolist()).value if len(events) else "C"
    delay = ((ev["publication_time"] - ev["day"]).median() if len(ev)
             else pd.Timedelta(days=2))
    frames: list[pd.DataFrame] = []
    for geo in regions:
        mask = [geo in _regions_for(t, None, terminal_regions) for t in ev["terminal"]]
        sub = ev.loc[mask] if len(ev) else ev
        grouped = sub.groupby("day")
        cnt = grouped.size().reindex(days, fill_value=0).astype(float)
        vol = grouped["cargo_volume"].sum().reindex(days, fill_value=0.0).astype(float)
        pub_obs = grouped["publication_time"].max().reindex(days)
        grade = grouped["pit_grade"].agg(lambda s: worst_grade(s.tolist()).value)
        grade = grade.reindex(days).fillna(worst_all)
        if publication_lag is not None:
            rule = apply_publication_lag(pd.Series(days, index=days), publication_lag)
            # never earlier than the events' own publication times
            keep_rule = pub_obs.isna() | (rule >= pub_obs)
            pub = pd.DatetimeIndex(rule.where(keep_rule, pub_obs))
            estimated = True
        else:
            fallback = pd.Series(days + pd.Timedelta(days=1) + delay, index=days)
            pub = pd.DatetimeIndex(pub_obs.fillna(fallback))
            estimated = False
        ingest = pub
        for series_id, values, unit in (
            (realised_series_id(geo), cnt.to_numpy(), "cargoes"),
            (realised_series_id(geo, "volume", volume_unit), vol.to_numpy(), volume_unit),
        ):
            f = _observation_frame(
                series_id, days, values, unit=unit, geography=geo, publication_time=pub,
                ingestion_time=ingest, source=_common_label(events["source"], SOURCE_LABEL),
                quality=QualityFlag(_common_label(events["quality_flag"], QualityFlag.OK.value)),
                entitlement=Entitlement(_common_label(events["entitlement"],
                                                      Entitlement.LICENSED_OTHER.value)),
                grade=PitGrade.A, estimated=estimated, source_tz="UTC",
            )
            f["pit_grade"] = grade.to_numpy()
            frames.append(f)
    out = pd.concat(frames, ignore_index=True) if frames else empty_frame(OBSERVATION_SCHEMA)
    return validate_frame(out, OBSERVATION_SCHEMA)


# ---------------------------------------------------------------------------------------------
# Payload parsers (pure functions; exercised in tests with SYNTHETIC payloads)
# ---------------------------------------------------------------------------------------------


def _dig(node: Any, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _month_offset(release: pd.Timestamp, delivery_start: pd.Timestamp) -> int:
    return (delivery_start.year - release.year) * 12 + (delivery_start.month - release.month)


def parse_price_releases(releases: Sequence[Mapping[str, Any]], spec: SeriesSpec) -> pd.DataFrame:
    """Parse ``/v1.0/contracts/{id}/price-releases/`` ``data`` into ``release_date, k, value``.

    Spot contracts (``curve: false``) must have exactly one data point per release; curve contracts
    are keyed by month offset ``k`` (delivery month minus release month).

    Raises:
        EmptyPayloadError: no releases.
        DataError: unexpected payload shape.
    """
    if not releases:
        raise EmptyPayloadError(f"{spec.series_id}: price-release payload is empty")
    field = str(spec.entry["field"])
    if not field.startswith("derivedPrices."):
        raise ConfigError(f"{spec.series_id}: field must be a derivedPrices.* path, got {field}")
    rows: list[tuple[pd.Timestamp, int, float]] = []
    curve = bool(spec.entry.get("curve", False))
    for rel in releases:
        ts = pd.Timestamp(rel["releaseDate"])
        rd = (ts.tz_localize(None) if ts.tzinfo else ts).normalize()
        points = [p for block in rel.get("data", []) for p in block.get("dataPoints", [])]
        if not curve and len(points) != 1:
            raise DataError(
                f"{spec.series_id}: spot release {rd.date()} has {len(points)} data points; "
                "expected exactly 1 (multi-point spot payloads were never inspected)"
            )
        for p in points:
            value = _dig(p, field)
            if value is None:
                continue
            k = 0
            if curve:
                k = _month_offset(rd, pd.Timestamp(p["deliveryPeriod"]["startAt"]))
            rows.append((rd, k, float(value)))
    return pd.DataFrame(rows, columns=["release_date", "k", "value"])


def parse_netbacks_payload(
    payload: Mapping[str, Any], *, port_key: str, release_date: str,
    fields: Mapping[str, str], month_index_key: str = "monthIdx",
) -> pd.DataFrame:
    """Parse one ``/v1.0/netbacks/`` response into ``release_date, port, field, month_index, value``.

    Raises:
        EmptyPayloadError: ``data`` or ``data.netbacks`` empty — the silent failure Spark returns
            when ``via-point`` is omitted. Never converted into an empty frame.
    """
    data = payload.get("data") if isinstance(payload, Mapping) else None
    points = data.get("netbacks") if isinstance(data, Mapping) else None
    if not points:
        raise EmptyPayloadError(
            f"netbacks {port_key} {release_date}: empty payload (HTTP 200 with no data). Spark "
            "returns this when via-point is missing; refusing to pass an empty frame downstream."
        )
    rows = []
    for p in points:
        idx = p.get(month_index_key)
        if idx is None:
            raise DataError(f"netbacks {port_key} {release_date}: point without {month_index_key}")
        for name, path in fields.items():
            v = _dig(p, path)
            rows.append((release_date, port_key, name, int(idx),
                         np.nan if v is None else float(v)))
    return pd.DataFrame(rows, columns=["release_date", "port", "field", "month_index", "value"])


def parse_terminal_slots_csv(text: str, *, terminal: str, buckets: Mapping[str, str],
                             release_col: str = "ReleaseDate") -> pd.DataFrame:
    """Parse a terminal-slots CSV into ``release_date, terminal, bucket, value`` (unsold slots)."""
    df = pd.read_csv(StringIO(text))
    if df.empty:
        raise EmptyPayloadError(f"terminal slots {terminal}: empty CSV")
    missing = [c for c in [release_col, *buckets.values()] if c not in df.columns]
    if missing:
        raise DataError(f"terminal slots {terminal}: CSV lacks columns {missing}")
    long = df.melt(id_vars=[release_col], value_vars=list(buckets.values()),
                   var_name="column", value_name="value")
    inv = {v: k for k, v in buckets.items()}
    return pd.DataFrame({
        "release_date": pd.to_datetime(long[release_col]),
        "terminal": terminal,
        "bucket": long["column"].map(inv),
        "value": pd.to_numeric(long["value"], errors="coerce").astype(float),
    })


# ---------------------------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------------------------


class _SparkBase:
    """Shared config handling. Subclasses set ``backend``."""

    backend: Backend

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: Mapping[str, Any] = config if config is not None else load_spark_config()

    def spec(self, series_id: str) -> SeriesSpec:
        return resolve_series(self.config, series_id)

    def _kind_check(self, series_ids: Sequence[str], kinds: set[str], method: str
                    ) -> list[SeriesSpec]:
        specs = [self.spec(s) for s in series_ids]
        bad = [s.series_id for s in specs if s.kind not in kinds]
        if bad:
            raise IdentifierUnknown(f"{method}: series {bad} are not of kind {sorted(kinds)}")
        return specs

    def terminal_regions(self) -> dict[str, list[str]]:
        """Terminal code -> regions from config (empty until the user supplies codes)."""
        regions = (self.config.get("regions") or {}).get("terminal_regions") or {}
        return {str(k): [str(x) for x in v] for k, v in regions.items()}


# --- mock -------------------------------------------------------------------------------------

MOCK_TERMINAL_REGIONS: dict[str, list[str]] = {
    "SYN_NWE_A": ["NWE", "EU"],
    "SYN_NWE_B": ["NWE", "EU"],
    "SYN_MED_A": ["MED", "EU"],
    "SYN_ASIA_A": ["ASIA"],
    "SYN_ASIA_B": ["ASIA"],
}
_MOCK_ORIGINS = ("US", "QA", "NG", "AU", "RU")


def synthetic_vessel_world(
    start: str = "2022-01-01", end: str = "2023-12-31", *, seed: int = 11,
    voyages_per_day: float = 3.0, snapshot_grade: PitGrade = PitGrade.A,
    publication_delay: pd.Timedelta = pd.Timedelta(hours=1),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SYNTHETIC laden voyages: daily snapshots with drifting ETAs and some diversions.

    Returns ``(snapshots, realised_events)`` in VESSEL_SNAPSHOT_SCHEMA / REALISED_ARRIVAL_SCHEMA.
    Each voyage loads on day ``L``, sails 10–35 days, and has a true terminal. Its declared
    destination is the true one with probability 0.6, another terminal with 0.25 (later diverted to
    the true one) or undeclared with 0.15 (declared later). Declared ETA = true arrival + noise
    shrinking with time to arrival. Realised events are published 2 days after discharge.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D", tz="UTC")
    terminals = list(MOCK_TERMINAL_REGIONS)
    snaps: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    vid = 0
    for d in days:
        for _ in range(rng.poisson(voyages_per_day)):
            vid += 1
            vessel = f"SYN{vid:06d}"
            dur = int(rng.integers(10, 36))
            arrival = d + pd.Timedelta(days=dur, hours=int(rng.integers(0, 24)))
            true_term = terminals[int(rng.integers(0, len(terminals)))]
            vol = float(rng.uniform(140_000, 175_000))
            u = rng.random()
            switch_day = int(rng.integers(1, max(2, dur - 2)))
            if u < 0.6:
                initial: str | None = true_term
            elif u < 0.85:
                others = [t for t in terminals if t != true_term]
                initial = others[int(rng.integers(0, len(others)))]
            else:
                initial = None
            for k in range(dur):
                snap_t = d + pd.Timedelta(days=k)
                if snap_t > days[-1]:
                    break
                dest = initial if k < switch_day else true_term
                remaining = (arrival - snap_t) / pd.Timedelta(days=1)
                eta = arrival + pd.Timedelta(hours=float(rng.normal(0, 4 + 2.0 * remaining)))
                snaps.append({
                    "snapshot_time": snap_t, "vessel_id": vessel, "voyage_id": f"{vessel}-1",
                    "laden": True, "origin": _MOCK_ORIGINS[int(rng.integers(0, 5))],
                    "load_time": d, "declared_destination": dest,
                    "declared_destination_region": None, "declared_eta": eta,
                    "cargo_volume": vol, "cargo_volume_unit": "m3_LNG",
                })
            if arrival <= days[-1] + pd.Timedelta(days=1):
                events.append({"vessel_id": vessel, "arrival_time": arrival, "terminal": true_term,
                               "cargo_volume": vol, "cargo_volume_unit": "m3_LNG"})
    snap_df = pd.DataFrame(snaps)
    n = len(snap_df)
    snap_df["publication_time"] = snap_df["snapshot_time"] + publication_delay
    snap_df["ingestion_time"] = snap_df["publication_time"] + pd.Timedelta(minutes=5)
    for col, val in {"unit": "m3_LNG", "source": SYNTHETIC_SOURCE,
                     "quality_flag": QualityFlag.SYNTHETIC.value,
                     "entitlement": Entitlement.SYNTHETIC.value,
                     "pit_grade": snapshot_grade.value, "source_timezone": "UTC"}.items():
        snap_df[col] = [val] * n
    snap_df["revision"] = np.zeros(n, dtype="int64")
    snap_df["publication_time_estimated"] = np.zeros(n, dtype=bool)
    for col in ("declared_destination", "declared_destination_region", "voyage_id", "origin"):
        snap_df[col] = snap_df[col].astype(object)
    snap_df["declared_eta"] = pd.to_datetime(snap_df["declared_eta"], utc=True)
    snap_df = snap_df[VESSEL_SNAPSHOT_SCHEMA.names]

    ev_df = pd.DataFrame(events)
    m = len(ev_df)
    ev_df["publication_time"] = ev_df["arrival_time"].dt.floor("D") + pd.Timedelta(days=2, hours=7)
    ev_df["ingestion_time"] = ev_df["publication_time"] + pd.Timedelta(minutes=5)
    for col, val in {"unit": "m3_LNG", "source": SYNTHETIC_SOURCE,
                     "quality_flag": QualityFlag.SYNTHETIC.value,
                     "entitlement": Entitlement.SYNTHETIC.value,
                     "pit_grade": PitGrade.B.value, "source_timezone": "UTC"}.items():
        ev_df[col] = [val] * m
    ev_df["revision"] = np.zeros(m, dtype="int64")
    ev_df["publication_time_estimated"] = np.ones(m, dtype=bool)
    ev_df = ev_df[REALISED_ARRIVAL_SCHEMA.names]
    return validate_snapshots(snap_df), validate_frame(ev_df, REALISED_ARRIVAL_SCHEMA)


class MockSparkAdapter(_SparkBase):
    """SYNTHETIC Spark backend. Serves every series the config defines (any status) and a
    synthetic vessel world. Values are random walks / simulated voyages — never evidence."""

    backend = Backend.MOCK

    def __init__(self, config: Mapping[str, Any] | None = None, *, seed: int = 11,
                 world_start: str = "2022-01-01", world_end: str = "2023-12-31",
                 snapshot_grade: PitGrade = PitGrade.A,
                 issue_time_local: str = "19:00") -> None:
        super().__init__(config)
        self.seed = seed
        self.world = (world_start, world_end)
        self.snapshot_grade = snapshot_grade
        self.issue_time_local = issue_time_local
        self._world: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def _vessel_world(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._world is None:
            self._world = synthetic_vessel_world(*self.world, seed=self.seed,
                                                 snapshot_grade=self.snapshot_grade)
        return self._world

    def terminal_regions(self) -> dict[str, list[str]]:
        return dict(MOCK_TERMINAL_REGIONS)

    def issue_times(self, start: date, end: date) -> pd.DatetimeIndex:
        """Daily issue times at ``issue_time_local`` Europe/Amsterdam (TTF settlement convention)."""
        hh, mm = (int(x) for x in self.issue_time_local.split(":"))
        local = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
        return (local + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(MARKET_TZ).tz_convert("UTC")

    def get_lng_vessels(self, start: date, end: date) -> pd.DataFrame:
        snaps, _ = self._vessel_world()
        return snaps.loc[_date_mask(snaps["snapshot_time"], start, end, "UTC")].reset_index(
            drop=True)

    def get_lng_arrivals(self, start: date, end: date) -> pd.DataFrame:
        snaps, _ = self._vessel_world()
        return build_arrival_forecasts(
            snaps, self.issue_times(start, end), terminal_regions=MOCK_TERMINAL_REGIONS,
            regions=("NWE", "EU"), include_terminals=False,
        )

    def get_realised_arrivals(self, start: date, end: date) -> pd.DataFrame:
        _, events = self._vessel_world()
        return realised_arrival_observations(events, terminal_regions=MOCK_TERMINAL_REGIONS,
                                             regions=("NWE", "EU"), start=start, end=end)

    def get_destination_statistics(self, start: date, end: date, lookback_days: int = 14
                                   ) -> pd.DataFrame:
        snaps, _ = self._vessel_world()
        return destination_observations(snaps, self.issue_times(start, end),
                                        terminal_regions=MOCK_TERMINAL_REGIONS,
                                        lookback_days=lookback_days)

    def _synthetic(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        seed = self.seed + sum(ord(c) for c in spec.series_id)
        rng = np.random.default_rng(seed)
        local = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
        if len(local) == 0:
            return empty_frame(OBSERVATION_SCHEMA)
        unit = str(spec.entry.get("unit", ""))
        if spec.kind == "terminal_slots":
            values = np.clip(np.round(20 + np.cumsum(rng.normal(0, 1, len(local)))), 0, None)
        elif unit.startswith("USD/d"):
            values = 60_000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(local))))
        else:
            values = np.cumsum(rng.normal(0, 0.15, len(local))) + 1.0
        event = pd.DatetimeIndex(local).tz_localize(LONDON_TZ).tz_convert("UTC")
        pub = pd.DatetimeIndex(apply_publication_lag(pd.Series(event), spec.lag))
        geography = spec.params.get("terminal") or spec.params.get("port") or str(
            spec.entry.get("geography", ""))
        return _observation_frame(
            spec.series_id, event, values, unit=unit, geography=geography,
            publication_time=pub, ingestion_time=pub + pd.Timedelta(minutes=5),
            source=SYNTHETIC_SOURCE, quality=QualityFlag.SYNTHETIC,
            entitlement=Entitlement.SYNTHETIC, grade=spec.grade, estimated=True,
            source_tz=LONDON_TZ,
        )

    def _many(self, specs: Sequence[SeriesSpec], start: date, end: date) -> pd.DataFrame:
        frames = [self._synthetic(s, start, end) for s in specs]
        out = pd.concat(frames, ignore_index=True) if frames else empty_frame(OBSERVATION_SCHEMA)
        return validate_frame(out, OBSERVATION_SCHEMA)

    def get_freight_assessments(self, series_ids: Sequence[str], start: date, end: date
                                ) -> pd.DataFrame:
        return self._many(self._kind_check(series_ids, {"price_release"}, "freight"), start, end)

    def get_gas_assessments(self, series_ids: Sequence[str], start: date, end: date
                            ) -> pd.DataFrame:
        return self._many(self._kind_check(series_ids, {"price_release"}, "gas"), start, end)

    def get_route_economics(self, series_ids: Sequence[str], start: date, end: date
                            ) -> pd.DataFrame:
        specs = self._kind_check(series_ids, {"netback", "terminal_slots"}, "route_economics")
        return self._many(specs, start, end)


# --- local file -------------------------------------------------------------------------------

_TS_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def _read_extract(base: Path, stem: str, schema: Schema) -> pd.DataFrame:
    """Read ``<base>/<stem>.parquet`` or ``.csv`` and validate against ``schema``. CSV timestamps
    must carry an explicit offset — naive strings are rejected, not assumed UTC."""
    pq, csv = base / f"{stem}.parquet", base / f"{stem}.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
        for col in (c.name for c in schema.columns if c.kind == "ts_utc"):
            if col not in df.columns:
                continue
            s = df[col].dropna().astype(str)
            if len(s) and not s.str.contains(_TS_OFFSET).all():
                raise SchemaError(f"{csv}: column {col} has timestamps without a UTC offset")
            df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
        for col in (c.name for c in schema.columns if c.kind == "bool"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
        for col in (c.name for c in schema.columns if c.kind == "float"):
            if col in df.columns:
                df[col] = df[col].astype(float)
    else:
        raise DataError(f"No extract {stem}.parquet/.csv under {base}")
    return validate_frame(df, schema)


class LocalFileSparkAdapter(_SparkBase):
    """Reads user-placed extracts in canonical schemas from ``<extracts_dir>/<subdir>/``.

    Files (Parquet preferred): ``vessels`` (VESSEL_SNAPSHOT_SCHEMA), ``arrivals``
    (FORECAST_SCHEMA), ``realised_arrivals`` and ``observations`` (OBSERVATION_SCHEMA). A requested
    series absent from ``observations`` raises rather than returning a partial frame.
    """

    backend = Backend.LOCAL_FILE

    def __init__(self, config: Mapping[str, Any] | None = None, *,
                 extracts_dir: str | Path | None = None) -> None:
        super().__init__(config)
        if extracts_dir is None:
            sources = load_config("data_sources")
            extracts_dir = PROJECT_ROOT / str(sources["local_file"]["extracts_dir"])
        lf = self.config.get("local_file") or {}
        self.base = Path(extracts_dir) / str(lf.get("subdir", "spark"))
        self.files: Mapping[str, str] = lf.get("files") or {
            "vessels": "vessels", "arrivals": "arrivals",
            "realised_arrivals": "realised_arrivals", "observations": "observations"}

    def get_lng_vessels(self, start: date, end: date) -> pd.DataFrame:
        df = validate_snapshots(_read_extract(self.base, self.files["vessels"],
                                              VESSEL_SNAPSHOT_SCHEMA))
        return df.loc[_date_mask(df["snapshot_time"], start, end, "UTC")].reset_index(drop=True)

    def get_lng_arrivals(self, start: date, end: date) -> pd.DataFrame:
        df = _read_extract(self.base, self.files["arrivals"], FORECAST_SCHEMA)
        return df.loc[_date_mask(df["issue_time"], start, end, "UTC")].reset_index(drop=True)

    def get_realised_arrivals(self, start: date, end: date) -> pd.DataFrame:
        df = _read_extract(self.base, self.files["realised_arrivals"], OBSERVATION_SCHEMA)
        return df.loc[_date_mask(df["event_time"], start, end, "UTC")].reset_index(drop=True)

    def _observations(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        for s in series_ids:
            self.spec(s)  # unknown ids raise IdentifierUnknown
        df = _read_extract(self.base, self.files["observations"], OBSERVATION_SCHEMA)
        out = df.loc[df["series_id"].isin(series_ids)
                     & _date_mask(df["event_time"], start, end)].reset_index(drop=True)
        missing = sorted(set(series_ids) - set(out["series_id"]))
        if missing:
            raise DataError(f"Local Spark extract has no rows for {missing} in {start}..{end}")
        return out

    def get_freight_assessments(self, series_ids: Sequence[str], start: date, end: date
                                ) -> pd.DataFrame:
        return self._observations(series_ids, start, end)

    def get_gas_assessments(self, series_ids: Sequence[str], start: date, end: date
                            ) -> pd.DataFrame:
        return self._observations(series_ids, start, end)

    def get_route_economics(self, series_ids: Sequence[str], start: date, end: date
                            ) -> pd.DataFrame:
        return self._observations(series_ids, start, end)


# --- live skeleton ----------------------------------------------------------------------------


class HttpTransport(Protocol):
    """Minimal HTTP interface so the live adapter can be exercised offline with a fake."""

    def request(self, method: str, url: str, headers: Mapping[str, str],
                body: bytes | None, timeout: float) -> tuple[int, bytes]: ...


class UrllibTransport:
    """Standard-library transport (no ``requests`` dependency). Never used in tests."""

    def request(self, method: str, url: str, headers: Mapping[str, str],
                body: bytes | None, timeout: float) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return int(resp.status), resp.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()


class LiveSparkAdapter(_SparkBase):
    """Live Spark skeleton. Requests only ``known`` identifiers, only when authorised.

    Raises at construction:
        BackendUnavailable: ``live_calls_authorised`` is false.
    """

    backend = Backend.LIVE

    def __init__(self, config: Mapping[str, Any] | None = None, *,
                 live_calls_authorised: bool | None = None,
                 transport: HttpTransport | None = None,
                 secret_getter: Callable[..., str | None] = get_secret) -> None:
        super().__init__(config)
        if live_calls_authorised is None:
            live_calls_authorised = bool(load_config("data_sources").get("live_calls_authorised"))
        if not live_calls_authorised:
            raise BackendUnavailable(
                "Live Spark calls are not authorised. Set live_calls_authorised: true in the "
                "git-ignored configs/local_data_sources.yaml, and only on explicit user instruction."
            )
        api = self.config.get("api") or {}
        if IdentifierStatus(str(api.get("status", "requires_user_input"))) is not \
                IdentifierStatus.KNOWN:
            raise IdentifierUnknown("Spark api block status is not 'known'")
        self._api = api
        self._secret = secret_getter
        self._transport: HttpTransport = transport or UrllibTransport()
        self._token: str | None = None
        override = secret_getter(str(api.get("base_url_env", "SPARK_BASE_URL")), required=False)
        self.base_url = str(override or api.get("base_url") or "")
        if not self.base_url:
            raise ConfigError("Spark base URL missing (api.base_url or SPARK_BASE_URL)")

    # -- auth / http --
    def _credentials(self) -> tuple[str, str]:
        creds = self._api.get("credentials") or {}
        try:
            cid = self._secret(str(creds["client_id_env"]))
            csec = self._secret(str(creds["client_secret_env"]))
        except (SecretMissingError, KeyError) as exc:
            raise BackendUnavailable(f"Spark credentials unavailable: {exc}") from exc
        if not cid or not csec:
            raise BackendUnavailable("Spark credentials are empty")
        return cid, csec

    def _access_token(self) -> str:
        if self._token:
            return self._token
        cid, csec = self._credentials()
        headers = {"Authorization": b64encode(f"{cid}:{csec}".encode()).decode(),
                   "Accept": "application/json", "Content-Type": "application/json"}
        body = json.dumps({"grantType": self._api.get("grant_type", "clientCredentials"),
                           "scopes": self._api.get("scopes", "read:access,read:prices")}).encode()
        status, raw = self._transport.request(
            "POST", urljoin(self.base_url, str(self._api.get("token_uri", "/oauth/token/"))),
            headers, body, float(self._api.get("timeout_seconds", 60)))
        if status not in (200, 201):
            raise BackendUnavailable(f"Spark auth failed: HTTP {status}")
        token = json.loads(raw).get("accessToken")
        if not token:
            raise BackendUnavailable("Spark auth response carried no accessToken")
        self._token = str(token)
        return self._token

    def _get(self, uri: str, accept: str = "application/json") -> Any:
        headers = {"Authorization": f"Bearer {self._access_token()}", "accept": accept}
        status, raw = self._transport.request("GET", urljoin(self.base_url, uri), headers, None,
                                              float(self._api.get("timeout_seconds", 60)))
        if status != 200:
            raise BackendUnavailable(f"Spark GET {uri} failed: HTTP {status}")
        text = raw.decode("utf-8")
        return text if accept != "application/json" else json.loads(text)

    def _require_known(self, specs: Sequence[SeriesSpec]) -> None:
        unknown = [s.series_id for s in specs if s.status is not IdentifierStatus.KNOWN]
        if unknown:
            raise IdentifierUnknown(f"Series {unknown} are not status 'known'; refusing live call")

    # -- vessels / arrivals: no known dataset --
    def _vessel_block(self, block: str) -> None:
        cfg = self.config.get(block) or {}
        status = IdentifierStatus(str(cfg.get("status", "requires_user_input")))
        if status is not IdentifierStatus.KNOWN or not cfg.get("dataset"):
            raise IdentifierUnknown(
                f"No Spark {block} dataset is known in this repo (status {status.value}). The user "
                "must name a vintaged vessel/arrival source; realised destinations are grade C."
            )
        raise NotImplementedError(f"{block} payload parser not written: payload shape unverified")

    def get_lng_vessels(self, start: date, end: date) -> pd.DataFrame:
        self._vessel_block("vessels")
        raise AssertionError("unreachable")

    def get_lng_arrivals(self, start: date, end: date) -> pd.DataFrame:
        self._vessel_block("arrivals")
        raise AssertionError("unreachable")

    # -- price releases --
    def _price_release_obs(self, specs: Sequence[SeriesSpec], start: date, end: date
                           ) -> pd.DataFrame:
        self._require_known(specs)
        frames: list[pd.DataFrame] = []
        by_contract: dict[str, list[SeriesSpec]] = {}
        for s in specs:
            by_contract.setdefault(str(s.entry["ticker"]), []).append(s)
        uri_tpl = str(self._api["price_releases_uri"])
        now = pd.Timestamp.now(tz="UTC")
        for contract, group in by_contract.items():
            query = f"?limit={int(self._api.get('max_releases', 4000))}"
            vessel = group[0].entry.get("vessel_type")
            if vessel:
                query += f"&vessel-type={vessel}"
            payload = self._get(uri_tpl.format(contract=contract) + query)
            for s in group:
                parsed = parse_price_releases(payload.get("data") or [], s)
                k = int(s.params.get("k", 0))
                sub = parsed.loc[(parsed["k"] == k)
                                 & (parsed["release_date"].dt.date >= start)
                                 & (parsed["release_date"].dt.date <= end)]
                if sub.empty:
                    raise DataError(f"{s.series_id}: no releases in {start}..{end}")
                event = _release_event_time(sub["release_date"])
                pub = pd.DatetimeIndex(apply_publication_lag(pd.Series(event), s.lag))
                frames.append(_observation_frame(
                    s.series_id, event, sub["value"].to_numpy(), unit=str(s.entry["unit"]),
                    geography=str(s.entry.get("geography", "")), publication_time=pub,
                    ingestion_time=pd.DatetimeIndex([now] * len(event)), source=SOURCE_LABEL,
                    quality=QualityFlag.OK, entitlement=Entitlement.LICENSED_SPARK,
                    grade=s.grade, estimated=True, source_tz=LONDON_TZ))
        return validate_frame(pd.concat(frames, ignore_index=True), OBSERVATION_SCHEMA)

    def get_freight_assessments(self, series_ids: Sequence[str], start: date, end: date
                                ) -> pd.DataFrame:
        return self._price_release_obs(
            self._kind_check(series_ids, {"price_release"}, "freight"), start, end)

    def get_gas_assessments(self, series_ids: Sequence[str], start: date, end: date
                            ) -> pd.DataFrame:
        return self._price_release_obs(
            self._kind_check(series_ids, {"price_release"}, "gas"), start, end)

    # -- netbacks / terminal slots --
    def netback_uri(self, port_key: str, release_date: str) -> str:
        """Build a netbacks request URI. Refuses a port without ``via_point`` (silent-empty trap)."""
        nb = self.config.get("netbacks") or {}
        port = (nb.get("fob_ports") or {}).get(port_key)
        if port is None:
            raise IdentifierUnknown(f"FOB port {port_key!r} not in spark.yaml netbacks.fob_ports")
        if IdentifierStatus(str(port.get("status", "requires_user_input"))) is not \
                IdentifierStatus.KNOWN:
            raise IdentifierUnknown(f"FOB port {port_key!r} status is not 'known'")
        via = port.get("via_point")
        if not via:
            raise ViaPointMissing(
                f"FOB port {port_key!r} has no via_point; Spark returns an empty payload without it"
            )
        return f"{nb['endpoint']}?fob-port={port['uuid']}&via-point={via}&release-date={release_date}"

    def _netback_obs(self, specs: Sequence[SeriesSpec], start: date, end: date) -> pd.DataFrame:
        nb = self.config.get("netbacks") or {}
        ref = self._get(str(nb["reference_endpoint"]))
        releases = _dig(ref, "data.staticData.sparkReleases")
        if not releases:
            raise EmptyPayloadError("netbacks reference data carried no release dates")
        dates = [r for r in releases if start <= pd.Timestamp(r).date() <= end]
        if not dates:
            raise DataError(f"No netback releases in {start}..{end}")
        fields = nb["fields"]
        now = pd.Timestamp.now(tz="UTC")
        frames: list[pd.DataFrame] = []
        for port in sorted({s.params["port"] for s in specs}):
            parsed = pd.concat([
                parse_netbacks_payload(self._get(self.netback_uri(port, d)), port_key=port,
                                       release_date=d, fields=fields,
                                       month_index_key=str(nb.get("month_index_key", "monthIdx")))
                for d in dates], ignore_index=True)
            for s in (x for x in specs if x.params["port"] == port):
                sub = parsed.loc[(parsed["field"] == s.params["field"])
                                 & (parsed["month_index"] == int(s.params["n"]))]
                if sub.empty:
                    raise DataError(f"{s.series_id}: netback payloads had no such point")
                event = _release_event_time(sub["release_date"])
                pub = pd.DatetimeIndex(apply_publication_lag(pd.Series(event), s.lag))
                frames.append(_observation_frame(
                    s.series_id, event, sub["value"].to_numpy(), unit="USD/MMBtu",
                    geography=port, publication_time=pub,
                    ingestion_time=pd.DatetimeIndex([now] * len(event)), source=SOURCE_LABEL,
                    quality=QualityFlag.OK, entitlement=Entitlement.LICENSED_SPARK,
                    grade=s.grade, estimated=True, source_tz=LONDON_TZ))
        return pd.concat(frames, ignore_index=True)

    def _slot_obs(self, specs: Sequence[SeriesSpec], start: date, end: date) -> pd.DataFrame:
        ts = self.config.get("terminal_slots") or {}
        accept = str(ts.get("accept_header", "text/csv"))
        listing = pd.read_csv(StringIO(self._get(str(ts["list_endpoint"]), accept=accept)))
        if "TerminalCode" not in listing.columns or "TerminalUUID" not in listing.columns:
            raise DataError("terminal list CSV lacks TerminalCode/TerminalUUID")
        codes = {_slug(c): (c, u) for c, u in zip(listing["TerminalCode"],
                                                   listing["TerminalUUID"], strict=True)}
        now = pd.Timestamp.now(tz="UTC")
        frames: list[pd.DataFrame] = []
        for term in sorted({s.params["terminal"] for s in specs}):
            if term not in codes:
                raise IdentifierUnknown(f"terminal {term!r} not in Spark terminal list")
            code, uuid = codes[term]
            text = self._get(str(ts["detail_endpoint"]).format(uuid=uuid), accept=accept)
            parsed = parse_terminal_slots_csv(text, terminal=str(code), buckets=ts["buckets"],
                                              release_col=str(ts.get("release_date_column",
                                                                     "ReleaseDate")))
            for s in (x for x in specs if x.params["terminal"] == term):
                sub = parsed.loc[(parsed["bucket"] == s.params["bucket"])
                                 & (parsed["release_date"].dt.date >= start)
                                 & (parsed["release_date"].dt.date <= end)]
                event = _release_event_time(sub["release_date"])
                pub = pd.DatetimeIndex(apply_publication_lag(pd.Series(event), s.lag))
                frames.append(_observation_frame(
                    s.series_id, event, sub["value"].to_numpy(), unit="slots", geography=str(code),
                    publication_time=pub, ingestion_time=pd.DatetimeIndex([now] * len(event)),
                    source=SOURCE_LABEL, quality=QualityFlag.OK,
                    entitlement=Entitlement.LICENSED_SPARK, grade=s.grade, estimated=True,
                    source_tz=LONDON_TZ))
        return pd.concat(frames, ignore_index=True)

    def get_route_economics(self, series_ids: Sequence[str], start: date, end: date
                            ) -> pd.DataFrame:
        specs = self._kind_check(series_ids, {"netback", "terminal_slots"}, "route_economics")
        self._require_known(specs)
        frames = []
        nb = [s for s in specs if s.kind == "netback"]
        sl = [s for s in specs if s.kind == "terminal_slots"]
        if nb:
            frames.append(self._netback_obs(nb, start, end))
        if sl:
            frames.append(self._slot_obs(sl, start, end))
        return validate_frame(pd.concat(frames, ignore_index=True), OBSERVATION_SCHEMA)


# ---------------------------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------------------------


def make_spark_adapter(backend: str | Backend | None = None, *,
                       config: Mapping[str, Any] | None = None, **kwargs: Any) -> _SparkBase:
    """Build the Spark adapter for ``backend`` (default: ``data_sources.yaml`` default_backend).

    ``kwargs`` go to the backend constructor (e.g. ``extracts_dir``, ``live_calls_authorised``,
    ``transport``, ``seed``).
    """
    if backend is None:
        backend = str(load_config("data_sources").get("default_backend", "mock"))
    b = Backend(backend)
    if b is Backend.MOCK:
        return MockSparkAdapter(config, **kwargs)
    if b is Backend.LOCAL_FILE:
        return LocalFileSparkAdapter(config, **kwargs)
    return LiveSparkAdapter(config, **kwargs)
