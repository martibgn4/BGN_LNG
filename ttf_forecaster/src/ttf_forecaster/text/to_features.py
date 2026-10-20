"""Reviewed events -> bitemporal OBSERVATION rows -> ``text__*`` features.

Two steps, so text features go through the same as-of resolver as every other family:

1. :func:`events_to_observations` turns the review ledger's versions into OBSERVATION_SCHEMA rows
   on a gas-day grid. Every change in what was knowable (an approval, an edit, a rejection)
   becomes a new ``revision`` with its own ``publication_time``.
2. :func:`build_text_features` (a ``FeatureBuilder``) reads those rows with ``resolve_as_of``.

Series emitted (``<slug>`` = lowercase, non-alphanumerics -> ``_``):

* ``text.<region>.outage_count`` - number of reviewed outage/maintenance events active on the
  gas day.
* ``text.<asset>.offline_mcmd`` - sum of *supported* offline volumes on the gas day; NaN when any
  active event on that asset has no supported number (unknown is not zero).

Point-in-time (``docs/llm_event_schema.md`` §5):

* **live**: ``publication_time = max(source publication, extracted_at, review_time)`` of the
  version that made the event visible; grade **A** (the review happened at that time).
* **historical**: reviews were done after the fact, so their timestamps are useless as availability
  times. ``publication_time = source publication + historical_review_lag`` (configured, flagged
  ``publication_time_estimated``); only the final reviewed version is used; grade **B** at best.
* any event set **assembled with hindsight** (documents searched for after the fact, selected
  because they mattered, archive not complete as-of): grade **C**, excluded from clean runs by
  ``resolve_as_of``'s default.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import pandas as pd

from ttf_forecaster.data.base import (
    MARKET_TZ,
    OBSERVATION_SCHEMA,
    SYNTHETIC_SOURCE,
    DataBundle,
    Entitlement,
    PitGrade,
    QualityFlag,
    SchemaError,
    empty_frame,
    validate_frame,
    worst_grade,
)
from ttf_forecaster.features.point_in_time import resolve_as_of, to_utc_timestamp
from ttf_forecaster.text.schema import PASSING_STATUSES, EventRecord, EventType, ReviewStatus

FAMILY = "text"
SERIES_PREFIX = "text."
GAS_DAY_START_HOUR = 6  # gas day D runs 06:00 local D -> 06:00 local D+1 (Europe/Amsterdam)
DEFAULT_COUNT_TYPES: tuple[str, ...] = (EventType.OUTAGE.value, EventType.MAINTENANCE.value)


class TextPitMode(StrEnum):
    LIVE = "live"
    HISTORICAL = "historical"


@dataclass(frozen=True)
class TextProvenance:
    """How the event set was produced; decides publication times and PIT grade.

    Attributes:
        mode: ``live`` (extracted and reviewed in real time) or ``historical`` (back-filled).
        assembled_with_hindsight: True if the *set* of documents was chosen after the fact or the
            archive was not complete as of each origin. Forces grade C.
        historical_review_lag: assumed delay from source publication to a timely review in
            historical mode (configured in ``configs/text.yaml``; an assumption, not a measurement).
        open_ended_days: horizon over which an event with a start but no stated end is assumed
            active; rows it touches are flagged ``estimated``.
    """

    mode: TextPitMode
    assembled_with_hindsight: bool
    historical_review_lag: pd.Timedelta = pd.Timedelta(days=1)
    open_ended_days: int = 14

    def __post_init__(self) -> None:
        if self.historical_review_lag < pd.Timedelta(0):
            raise ValueError("historical_review_lag must be >= 0")
        if self.open_ended_days < 1:
            raise ValueError("open_ended_days must be >= 1")


def text_pit_grade(mode: TextPitMode | str, assembled_with_hindsight: bool) -> PitGrade:
    """A: live extraction+review; B: historical back-fill from a complete archive; C: hindsight."""
    if assembled_with_hindsight:
        return PitGrade.C
    return PitGrade.A if TextPitMode(mode) == TextPitMode.LIVE else PitGrade.B


def slug(name: str) -> str:
    """Series-id-safe slug: lowercase, runs of non-alphanumerics -> ``_``."""
    s = re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")
    if not s:
        raise ValueError(f"cannot build a series id from {name!r}")
    return s


def _utc(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        raise SchemaError(f"timezone-naive timestamp {dt!r} in event record")
    return ts.tz_convert("UTC")


def knowledge_timeline(
    versions: Iterable[EventRecord], prov: TextProvenance
) -> list[tuple[pd.Timestamp, str, EventRecord | None]]:
    """When each event's visible state changed: ``(time, event_id, state or None)`` sorted by time.

    Pending versions are invisible. A rejected latest version withdraws the event (state None).
    """
    by_id: dict[str, list[EventRecord]] = {}
    for v in versions:
        by_id.setdefault(v.event_id, []).append(v)
    out: list[tuple[pd.Timestamp, str, EventRecord | None]] = []
    for eid, vs in by_id.items():
        vs = sorted(vs, key=lambda r: r.version)
        if prov.mode == TextPitMode.HISTORICAL:
            final = vs[-1]
            if final.review_status in PASSING_STATUSES:
                out.append((_utc(final.publication_time) + prov.historical_review_lag, eid, final))
            continue
        for v in vs:
            if v.review_status == ReviewStatus.PENDING:
                continue
            if v.review_time is None or v.extracted_at is None:
                raise SchemaError(f"event {eid} v{v.version}: reviewed version lacks times")
            kt = max(_utc(v.publication_time), _utc(v.extracted_at), _utc(v.review_time))
            out.append((kt, eid, v if v.review_status in PASSING_STATUSES else None))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def gas_day_start(ts: pd.Timestamp, tz: str = MARKET_TZ) -> pd.Timestamp:
    """UTC start of the gas day containing ``ts``."""
    local = to_utc_timestamp(ts).tz_convert(tz)
    day = local.normalize()
    if local.hour < GAS_DAY_START_HOUR:
        day -= pd.Timedelta(days=1)
    return _gas_day_at(day.date(), tz)


def _gas_day_at(d: Any, tz: str = MARKET_TZ) -> pd.Timestamp:
    return pd.Timestamp(d).replace(hour=GAS_DAY_START_HOUR).tz_localize(tz).tz_convert("UTC")


def _active_days(ev: EventRecord, prov: TextProvenance) -> tuple[list[pd.Timestamp], bool]:
    """Gas-day starts (UTC) on which the event is active, and whether the window was assumed."""
    start_dt = ev.expected_start or ev.event_time
    if start_dt is None:
        return [], False
    start = _utc(start_dt)
    assumed = ev.expected_end is None
    if ev.expected_end is None:
        end = start + pd.Timedelta(days=prov.open_ended_days)
    else:
        end = _utc(ev.expected_end)
    first = gas_day_start(start)
    days = []
    d = first
    while d < end or (d == first and end == start):
        days.append(d)
        d = _gas_day_at((d.tz_convert(MARKET_TZ) + pd.Timedelta(days=1)).date())
    return days, assumed


def _aggregate(
    states: Mapping[str, EventRecord], prov: TextProvenance, count_types: frozenset[str]
) -> dict[tuple[str, pd.Timestamp], dict[str, Any]]:
    agg: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    for eid, ev in sorted(states.items()):
        if ev.event_type.value not in count_types:
            continue
        days, assumed = _active_days(ev, prov)
        region = slug(ev.region) if ev.region else "unknown"
        keys: list[tuple[str, str, str]] = [(f"text.{region}.outage_count", region, "count")]
        if ev.asset:
            keys.append((f"text.{slug(ev.asset)}.offline_mcmd", region, "mcm/d"))
        for d in days:
            for sid, geo, unit in keys:
                cell = agg.setdefault(
                    (sid, d),
                    {"value": 0.0, "geography": geo, "unit": unit, "assumed": False, "events": [],
                     "entitlements": set()},
                )
                if unit == "count":
                    cell["value"] += 1.0
                else:
                    vol = ev.estimated_volume_impact_mcmd
                    cell["value"] = float("nan") if vol is None else cell["value"] + float(vol)
                cell["assumed"] |= assumed
                cell["events"].append(f"{eid}:v{ev.version}")
                cell["entitlements"].add(ev.source_entitlement)
    return agg


def _same(a: float, b: float) -> bool:
    return (pd.isna(a) and pd.isna(b)) or a == b


def events_to_observations(
    versions: Iterable[EventRecord],
    prov: TextProvenance,
    *,
    count_types: Iterable[str] = DEFAULT_COUNT_TYPES,
    source_label: str = "text_events",
) -> pd.DataFrame:
    """Bitemporal OBSERVATION_SCHEMA rows from all versions in a review ledger.

    A row is emitted for a ``(series_id, gas day)`` whenever its value changes as events become
    visible, are edited or are rejected; each emission is a new ``revision``. Extra column
    ``lineage`` lists the contributing ``event_id:vN``.
    """
    grade = text_pit_grade(prov.mode, prov.assembled_with_hindsight).value
    types = frozenset(count_types)
    timeline = knowledge_timeline(versions, prov)
    states: dict[str, EventRecord] = {}
    last: dict[tuple[str, pd.Timestamp], tuple[float, int, str]] = {}
    rows: list[dict[str, Any]] = []
    i = 0
    while i < len(timeline):
        kt = timeline[i][0]
        while i < len(timeline) and timeline[i][0] == kt:
            _, eid, st = timeline[i]
            if st is None:
                states.pop(eid, None)
            else:
                states[eid] = st
            i += 1
        agg = _aggregate(states, prov, types)
        keys = set(agg) | set(last)
        for key in sorted(keys):
            cell = agg.get(key)
            prev = last.get(key)
            if cell is None:  # event withdrawn/edited away: the known value is now zero
                assert prev is not None
                cell = {"value": 0.0, "geography": prev[2], "unit": "count"
                        if key[0].endswith("outage_count") else "mcm/d", "assumed": False,
                        "events": [], "entitlements": set()}
            if prev is not None and _same(prev[0], cell["value"]):
                continue
            ents = cell["entitlements"]
            synthetic = bool(ents) and ents == {Entitlement.SYNTHETIC.value}
            if synthetic:
                quality = QualityFlag.SYNTHETIC.value
            elif cell["assumed"] or prov.mode == TextPitMode.HISTORICAL:
                quality = QualityFlag.ESTIMATED.value
            else:
                quality = QualityFlag.OK.value
            ent = (
                next(iter(ents)) if len(ents) == 1
                else Entitlement.LICENSED_OTHER.value if ents - {Entitlement.PUBLIC.value,
                                                                 Entitlement.SYNTHETIC.value}
                else Entitlement.PUBLIC.value
            )
            rev = 0 if prev is None else prev[1] + 1
            rows.append({
                "series_id": key[0],
                "event_time": key[1],
                "value": float(cell["value"]),
                "geography": cell["geography"],
                "publication_time": kt,
                "ingestion_time": kt,
                "revision": rev,
                "unit": cell["unit"],
                "source": SYNTHETIC_SOURCE if synthetic else source_label,
                "quality_flag": quality,
                "entitlement": ent,
                "pit_grade": grade,
                "publication_time_estimated": prov.mode == TextPitMode.HISTORICAL,
                "source_timezone": "UTC",
                "lineage": ",".join(cell["events"]),
            })
            last[key] = (float(cell["value"]), rev, cell["geography"])
    if not rows:
        out = empty_frame(OBSERVATION_SCHEMA)
        out["lineage"] = pd.Series(dtype="object")
        return out
    df = pd.DataFrame(rows)
    for c in ("event_time", "publication_time", "ingestion_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["revision"] = df["revision"].astype("int64")
    df["publication_time_estimated"] = df["publication_time_estimated"].astype(bool)
    return validate_frame(df, OBSERVATION_SCHEMA)


def _column_name(series_id: str, offset_days: int) -> str:
    return f"{FAMILY}__{series_id[len(SERIES_PREFIX):].replace('.', '_')}_d{offset_days}"


def build_text_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: dict[str, object]
) -> pd.DataFrame:
    """FeatureBuilder for the ``text`` family.

    Params:
        series: list of ``text.*`` series ids (fixed column universe; recommended). Default: every
            ``text.*`` series in the bundle (note: that universe is chosen with hindsight).
        offset_days: gas-day offset from the origin's gas day (default 1 = next gas day).
        coverage_start: tz-aware ISO timestamp from which the reviewed event feed is complete. For
            origins at/after it a missing row means "no known event" -> 0; before it -> NaN.
            Absent -> always NaN when missing.
        allowed_grades: PIT grades admitted (default A, B; grade C only in labelled experiments).

    Returns:
        Frame indexed by ``origins`` with ``text__*`` columns; ``attrs["pit_grade"]`` per column,
        ``attrs["family"] = "text"``, ``attrs["clean_backtest_eligible"]`` from params
        (default False).
    """
    obs = bundle.observations
    text = obs.loc[obs["series_id"].astype(str).str.startswith(SERIES_PREFIX)]
    validate_frame(text, OBSERVATION_SCHEMA)
    offset = int(params.get("offset_days", 1))  # type: ignore[call-overload]
    series = list(
        params.get("series") or sorted(text["series_id"].unique())  # type: ignore[call-overload]
    )
    bad = [s for s in series if not str(s).startswith(SERIES_PREFIX)]
    if bad:
        raise ValueError(f"text features only read text.* series, got {bad}")
    cov_raw = params.get("coverage_start")
    coverage = to_utc_timestamp(str(cov_raw)) if cov_raw else None
    ag = params.get("allowed_grades", (PitGrade.A.value, PitGrade.B.value))
    allowed = None if ag is None else frozenset(str(g) for g in ag)  # type: ignore[union-attr]

    if origins.tz is None:
        raise SchemaError("origins must be tz-aware")
    cols = {s: _column_name(str(s), offset) for s in series}
    out = pd.DataFrame(
        index=origins.tz_convert("UTC"), columns=list(cols.values()), dtype="float64"
    )
    out.index.name = "origin_time"
    for origin in out.index:
        resolved = resolve_as_of(text, origin, allowed_grades=allowed, validate=False)
        day = _gas_day_at(
            (gas_day_start(origin).tz_convert(MARKET_TZ) + pd.Timedelta(days=offset)).date()
        )
        hit = resolved.loc[resolved["event_time"] == day].set_index("series_id")["value"]
        fill = 0.0 if coverage is not None and origin >= coverage else float("nan")
        for s, c in cols.items():
            out.at[origin, c] = float(hit[s]) if s in hit.index else fill

    grades: dict[str, str] = {}
    all_grades = text["pit_grade"].dropna().unique().tolist()
    for s, c in cols.items():
        g = text.loc[text["series_id"] == s, "pit_grade"].dropna().unique().tolist() or all_grades
        grades[c] = worst_grade(g).value if g else PitGrade.C.value
    out.attrs["pit_grade"] = grades
    out.attrs["family"] = FAMILY
    out.attrs["clean_backtest_eligible"] = bool(params.get("clean_backtest_eligible", False))
    return out
