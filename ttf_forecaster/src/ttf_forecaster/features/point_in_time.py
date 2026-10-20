"""The as-of resolver: the only sanctioned read path from stored data into features.

``docs/point_in_time_policy.md`` §2:

    visible(as_of)  = rows with publication_time <= as_of  (and ingestion_time <= as_of if strict)
    resolved(as_of) = per key, the visible row with the highest revision (latest publication on ties)

For observations the key is ``(series_id, event_time)``. For forecasts it is
``(series_id, valid_time, model, member)`` and, among visible runs, the latest ``issue_time`` wins
unless a specific lead is requested.
"""

from __future__ import annotations

import pandas as pd

from ttf_forecaster.data.base import (
    CURVE_SCHEMA,
    FORECAST_SCHEMA,
    DataError,
    OBSERVATION_SCHEMA,
    PitGrade,
    PointInTimeViolation,
    validate_frame,
)

_OBS_KEY = ["series_id", "event_time"]
_FC_KEY = ["series_id", "valid_time", "model", "member"]


def to_utc_timestamp(as_of: pd.Timestamp | str) -> pd.Timestamp:
    """Coerce ``as_of`` to a tz-aware UTC Timestamp. Naive input is rejected, not assumed."""
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        raise PointInTimeViolation(
            f"as_of {as_of!r} is timezone-naive; pass an explicit zone (e.g. Europe/Amsterdam)"
        )
    return ts.tz_convert("UTC")


def _visible(df: pd.DataFrame, as_of: pd.Timestamp, strict_ingestion: bool) -> pd.DataFrame:
    mask = df["publication_time"] <= as_of
    if strict_ingestion:
        mask &= df["ingestion_time"] <= as_of
    return df.loc[mask]


def _exclude_grades(df: pd.DataFrame, allowed: frozenset[str] | None) -> pd.DataFrame:
    if allowed is None:
        return df
    return df.loc[df["pit_grade"].isin(allowed)]


def _raise_on_conflicting_ties(df: pd.DataFrame, key: list[str]) -> None:
    """Rows identical in ``key`` (incl. revision and publication_time) but different in value are a
    data error: no sort order can choose between them without making the as-of view depend on
    file or concat order."""
    dup = df.duplicated(key, keep=False)
    if not dup.any():
        return
    sub = df.loc[dup]
    n_values = sub.groupby(key, dropna=False)["value"].nunique(dropna=False)
    bad = n_values[n_values > 1]
    if len(bad):
        raise DataError(
            f"{len(bad)} key(s) have conflicting values at identical revision and publication_time, "
            f"e.g. {bad.index[0]}; fix the source data rather than letting row order decide"
        )


raise_on_conflicting_ties = _raise_on_conflicting_ties  # public name (used by features/panel.py)


def resolve_as_of(
    observations: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    strict_ingestion: bool = False,
    allowed_grades: frozenset[str] | None = frozenset({PitGrade.A.value, PitGrade.B.value}),
    validate: bool = True,
) -> pd.DataFrame:
    """Return the latest visible vintage of every ``(series_id, event_time)`` as of ``as_of``.

    Args:
        observations: OBSERVATION_SCHEMA frame (all vintages).
        as_of: tz-aware timestamp of the forecast origin.
        strict_ingestion: also require ``ingestion_time <= as_of`` (live operation).
        allowed_grades: PIT grades admitted; default excludes grade C. ``None`` admits all
            (only for explicitly labelled contaminated experiments).
        validate: run schema validation first.

    Returns:
        Frame with one row per key, sorted by ``series_id, event_time``. Rows with
        ``event_time > as_of`` are retained only if published by ``as_of`` (e.g. schedules).
    """
    if validate:
        validate_frame(observations, OBSERVATION_SCHEMA)
    ts = to_utc_timestamp(as_of)
    vis = _exclude_grades(_visible(observations, ts, strict_ingestion), allowed_grades)
    if vis.empty:
        return vis.copy()
    _raise_on_conflicting_ties(vis, [*_OBS_KEY, "revision", "publication_time"])
    ordered = vis.sort_values(
        [*_OBS_KEY, "revision", "publication_time", "ingestion_time"], kind="mergesort"
    )
    out = ordered.drop_duplicates(_OBS_KEY, keep="last")
    return out.sort_values(_OBS_KEY).reset_index(drop=True)


def resolve_forecasts_as_of(
    forecasts: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    min_lead_hours: float | None = None,
    strict_ingestion: bool = False,
    allowed_grades: frozenset[str] | None = frozenset({PitGrade.A.value, PitGrade.B.value}),
    validate: bool = True,
) -> pd.DataFrame:
    """Latest forecast run visible at ``as_of`` for each ``(series_id, valid_time, model, member)``.

    ``min_lead_hours`` restricts to runs whose lead is at least that many hours, which reproduces
    "what the forecast said N hours ahead" for revision features.
    """
    if validate:
        validate_frame(forecasts, FORECAST_SCHEMA)
    ts = to_utc_timestamp(as_of)
    vis = _exclude_grades(_visible(forecasts, ts, strict_ingestion), allowed_grades)
    if min_lead_hours is not None:
        vis = vis.loc[vis["lead_hours"] >= min_lead_hours]
    if vis.empty:
        return vis.copy()
    _raise_on_conflicting_ties(vis, [*_FC_KEY, "issue_time", "revision", "publication_time"])
    ordered = vis.sort_values(
        [*_FC_KEY, "issue_time", "revision", "publication_time", "ingestion_time"], kind="mergesort"
    )
    out = ordered.drop_duplicates(_FC_KEY, keep="last")
    return out.sort_values(_FC_KEY).reset_index(drop=True)


_CURVE_KEY = ["hub", "contract_id", "trade_date", "field"]


def resolve_curves_as_of(
    curves: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    strict_ingestion: bool = False,
    allowed_grades: frozenset[str] | None = frozenset({PitGrade.A.value, PitGrade.B.value}),
    validate: bool = True,
) -> pd.DataFrame:
    """Latest visible vintage of every ``(hub, contract_id, trade_date, field)`` at ``as_of``.

    Curve rows follow the same as-of rule as observations: a settlement for trading date *t* is
    invisible before its ``publication_time`` (19:00 Europe/Amsterdam on *t* by convention).
    """
    if validate:
        validate_frame(curves, CURVE_SCHEMA)
    ts = to_utc_timestamp(as_of)
    vis = _exclude_grades(_visible(curves, ts, strict_ingestion), allowed_grades)
    if vis.empty:
        return vis.copy()
    _raise_on_conflicting_ties(vis, [*_CURVE_KEY, "revision", "publication_time"])
    ordered = vis.sort_values(
        [*_CURVE_KEY, "revision", "publication_time", "ingestion_time"], kind="mergesort"
    )
    out = ordered.drop_duplicates(_CURVE_KEY, keep="last")
    return out.sort_values(_CURVE_KEY).reset_index(drop=True)


def latest_value_per_series(
    resolved: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    max_staleness: pd.Timedelta | None = None,
) -> pd.Series:
    """From an as-of-resolved frame, the most recent ``event_time <= as_of`` value per series.

    Values older than ``max_staleness`` (measured as_of − event_time) become NaN instead of being
    carried forward indefinitely.
    """
    ts = to_utc_timestamp(as_of)
    past = resolved.loc[resolved["event_time"] <= ts].sort_values("event_time")
    last = past.groupby("series_id").tail(1).set_index("series_id")
    values = last["value"].astype(float)
    if max_staleness is not None:
        stale = (ts - last["event_time"]) > max_staleness
        values = values.mask(stale)
    return values


def assert_no_future_rows(frame: pd.DataFrame, as_of: pd.Timestamp | str, column: str) -> None:
    """Raise ``PointInTimeViolation`` if ``frame[column]`` contains anything after ``as_of``."""
    ts = to_utc_timestamp(as_of)
    n = int((frame[column] > ts).sum())
    if n:
        raise PointInTimeViolation(f"{n} rows with {column} > as_of={ts.isoformat()}")
