"""Vectorised point-in-time panels: the performance path for every feature family.

``asof_panel`` returns, for many origins at once, exactly what this loop returns::

    for o in origins:
        resolved = resolve_as_of(observations, o, strict_ingestion=..., allowed_grades=...)
        row = latest_value_per_series(resolved, o, max_staleness=...)

i.e. per ``(origin, series)``: among keys ``(series_id, event_time)`` with at least one visible row
and ``event_time <= origin``, take the latest ``event_time``; its value is the visible row with the
highest ``revision`` (later ``publication_time``, then later input row, on ties). Equivalence is
asserted in ``tests/unit/features/test_panel.py`` on synthetic data with revisions and late
publications.

Algorithm (two ``merge_asof`` passes, O(N log N) instead of O(origins x N)):

1. A key becomes a *candidate* at ``avail = max(first visibility time, event_time)``. Sorted by
   ``avail``, the running max of ``event_time`` per series is the key the loop would pick.
2. Per key, the *best visible vintage* only changes when a row of higher priority becomes visible.
   Keeping those change points gives a step function, read with a second as-of join.

``align_to_origins`` is the companion for features computed in trading-date space (prices, curves):
each row carries an ``available_time`` (cumulative max of its inputs' publication times) and is
joined backward onto the origins.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    OBSERVATION_SCHEMA,
    DataError,
    PitGrade,
    PointInTimeViolation,
    SchemaError,
    validate_frame,
    worst_grade,
)
from ttf_forecaster.features.point_in_time import _raise_on_conflicting_ties

DEFAULT_ALLOWED_GRADES: frozenset[str] = frozenset({PitGrade.A.value, PitGrade.B.value})
ORIGIN_INDEX_NAME = "origin_time"


def as_origin_index(origins: Iterable[pd.Timestamp] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Validate and normalise forecast origins to a unique tz-aware UTC ``DatetimeIndex``.

    Raises:
        PointInTimeViolation: origins are timezone-naive.
        ValueError: origins contain NaT or duplicates.
    """
    idx = pd.DatetimeIndex(origins)
    if len(idx) and idx.tz is None:
        raise PointInTimeViolation(
            "origins are timezone-naive; pass tz-aware timestamps (e.g. 19:00 Europe/Amsterdam)"
        )
    if idx.tz is None:  # empty and naive: make it UTC so downstream dtypes agree
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("UTC").as_unit("ns")
    if idx.hasnans:
        raise ValueError("origins contain NaT")
    if not idx.is_unique:
        dup = idx[idx.duplicated()][:3].tolist()
        raise ValueError(f"origins must be unique; duplicates e.g. {dup}")
    return idx.rename(ORIGIN_INDEX_NAME)


def _ns(s: pd.Series) -> np.ndarray:
    """tz-aware datetime Series -> int64 nanoseconds since epoch (UTC)."""
    if not isinstance(s.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{s.name}: expected tz-aware datetime, got {s.dtype}")
    return s.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()


def asof_panel(
    observations: pd.DataFrame,
    origins: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    series_ids: Sequence[str] | None = None,
    max_staleness: pd.Timedelta | None = None,
    *,
    strict_ingestion: bool = False,
    allowed_grades: frozenset[str] | None = DEFAULT_ALLOWED_GRADES,
    validate: bool = True,
    return_event_time: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Origins x series panel of the latest value visible at each origin.

    Args:
        observations: OBSERVATION_SCHEMA rows, all vintages.
        origins: tz-aware forecast origins (any order; output keeps the given order).
        series_ids: columns to return (absent series become all-NaN columns). ``None`` = all.
        max_staleness: NaN where ``origin - event_time > max_staleness`` (no indefinite ffill).
        strict_ingestion: also require ``ingestion_time <= origin`` (live operation).
        allowed_grades: PIT grades admitted (default A, B). ``None`` admits all.
        validate: run OBSERVATION_SCHEMA validation first.
        return_event_time: also return the panel of the ``event_time`` each value refers to.

    Raises:
        DataError: conflicting duplicate vintages (same key, revision and publication_time,
            different value) visible at the latest origin -- as ``resolve_as_of``.

    Returns:
        ``values`` (float, index ``origin_time``, columns = series ids) and, if requested, the
        matching ``event_time`` panel. ``values.attrs["pit_grade"]`` maps each series to the worst
        grade among its admitted rows (``"A"`` for series with no rows).
    """
    if validate:
        validate_frame(observations, OBSERVATION_SCHEMA)
    idx = as_origin_index(origins)
    if series_ids is None:
        sids = sorted(observations["series_id"].unique().tolist())
    else:
        sids = list(series_ids)
        if len(set(sids)) != len(sids):
            raise ValueError("series_ids contains duplicates")

    obs = observations.loc[observations["series_id"].isin(sids)]
    if allowed_grades is not None:
        obs = obs.loc[obs["pit_grade"].isin(allowed_grades)]
    obs = obs.reset_index(drop=True)

    cols = pd.Index(sids, name="series_id")
    values = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    events = pd.DataFrame(pd.NaT, index=idx, columns=cols, dtype="datetime64[ns, UTC]")
    values.attrs["pit_grade"] = series_pit_grades(obs, sids)
    if obs.empty or len(idx) == 0:
        return (values, events) if return_event_time else values

    n = len(obs)
    sid = obs["series_id"].to_numpy()
    et = _ns(obs["event_time"])
    pub = _ns(obs["publication_time"])
    vis = np.maximum(pub, _ns(obs["ingestion_time"])) if strict_ingestion else pub
    rev = obs["revision"].to_numpy(dtype="int64")
    ing = _ns(obs["ingestion_time"])
    # Same data-error rule as resolve_as_of (modelling_log D-015): rows equal in key, revision and
    # publication_time but different in value raise DataError once both are visible at some origin.
    _raise_on_conflicting_ties(
        obs.loc[vis <= idx.asi8.max()], ["series_id", "event_time", "revision", "publication_time"]
    )
    # Priority of a row within its key: (revision, publication_time, ingestion_time, input
    # position), as in resolve_as_of's stable sort + keep="last".
    order = np.lexsort((np.arange(n), ing, pub, rev))
    rank = np.empty(n, dtype="int64")
    rank[order] = np.arange(n)

    # -- pass 1: which event_time is "latest visible" at each origin -------------------------
    keys = pd.DataFrame({"series_id": sid, "event_time": et, "vis": vis})
    first = keys.groupby(["series_id", "event_time"], sort=False)["vis"].min().reset_index()
    first["avail"] = np.maximum(first["vis"].to_numpy(), first["event_time"].to_numpy())
    first = first.sort_values(["series_id", "avail", "event_time"], kind="mergesort")
    first["best_et"] = first.groupby("series_id", sort=False)["event_time"].cummax()
    avail_tbl = (
        first.drop_duplicates(["series_id", "avail"], keep="last")[["series_id", "avail", "best_et"]]
        .sort_values("avail", kind="mergesort")
        .reset_index(drop=True)
    )
    o_sorted = np.sort(idx.asi8)
    left = pd.DataFrame(
        {
            "origin": np.repeat(o_sorted, len(sids)),
            "series_id": np.tile(np.asarray(sids, dtype=object), len(o_sorted)),
        }
    )
    m1 = pd.merge_asof(
        left, avail_tbl, left_on="origin", right_on="avail", by="series_id", direction="backward"
    )
    m1 = m1.dropna(subset=["best_et"])
    if m1.empty:
        return (values, events) if return_event_time else values
    m1["best_et"] = m1["best_et"].astype("int64")

    # -- pass 2: best visible vintage of that key at the origin -------------------------------
    st = pd.DataFrame(
        {"series_id": sid, "event_time": et, "vis": vis, "rank": rank,
         "value": obs["value"].astype(float).to_numpy()}
    ).sort_values(["series_id", "event_time", "vis", "rank"], kind="mergesort")
    run = st.groupby(["series_id", "event_time"], sort=False)["rank"].cummax()
    st = st.loc[st["rank"].to_numpy() == run.to_numpy()]
    st = st.drop_duplicates(["series_id", "event_time", "vis"], keep="last")
    st = st.sort_values("vis", kind="mergesort").reset_index(drop=True)
    m2 = pd.merge_asof(
        m1.sort_values("origin", kind="mergesort"),
        st[["series_id", "event_time", "vis", "value"]],
        left_on="origin",
        right_on="vis",
        left_by=["series_id", "best_et"],
        right_by=["series_id", "event_time"],
        direction="backward",
    )
    if m2["vis"].isna().any():  # cannot happen if pass 1 and 2 agree; guard the invariant
        raise RuntimeError("asof_panel internal error: candidate key without a visible vintage")
    if max_staleness is not None:
        stale = (m2["origin"] - m2["best_et"]) > pd.Timedelta(max_staleness).value
        m2.loc[stale, "value"] = np.nan

    wide = m2.pivot(index="origin", columns="series_id", values="value")
    wide.index = pd.to_datetime(wide.index, utc=True)
    values.loc[:, :] = wide.reindex(index=idx, columns=cols).to_numpy(dtype=float)
    if return_event_time:
        ev = m2.pivot(index="origin", columns="series_id", values="best_et")
        ev.index = pd.to_datetime(ev.index, utc=True)
        ev = ev.reindex(index=idx, columns=cols)
        events = ev.apply(lambda c: pd.to_datetime(c, utc=True))
        events.index = idx
        return values, events
    return values


def series_pit_grades(observations: pd.DataFrame, series_ids: Sequence[str]) -> dict[str, str]:
    """Worst ``pit_grade`` per series among the given rows (``"A"`` if a series has no rows)."""
    out: dict[str, str] = {}
    grouped = observations.groupby("series_id")["pit_grade"].unique() if len(observations) else {}
    for s in series_ids:
        grades = list(grouped[s]) if s in grouped else []
        out[s] = worst_grade(grades).value if grades else PitGrade.A.value
    return out


def final_vintage(
    frame: pd.DataFrame,
    key: Sequence[str],
    *,
    allowed_grades: frozenset[str] | None = DEFAULT_ALLOWED_GRADES,
) -> pd.DataFrame:
    """Keep the highest-revision row per ``key`` and stamp ``available_time`` = its publication.

    This is the **conservative** path for (almost) unrevised market data (settlements): a value is
    used only from the time its final stored vintage was published, so a revision delays the value
    instead of leaking it. Earlier vintages are ignored, never back-filled.
    """
    df = frame
    if allowed_grades is not None:
        df = df.loc[df["pit_grade"].isin(allowed_grades)]
    df = df.sort_values([*key, "revision", "publication_time"], kind="mergesort")
    df = df.drop_duplicates(list(key), keep="last").copy()
    df["available_time"] = df["publication_time"]
    return df.reset_index(drop=True)


def align_to_origins(
    frame: pd.DataFrame,
    origins: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    value_columns: Sequence[str],
    *,
    event_col: str = "event_time",
    available_col: str = "available_time",
    max_staleness: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Join rows computed in event (trading-date) space onto origins, as of availability.

    Rows are ordered by ``event_col``; availability is made monotone with a cumulative max, so a
    row is never used before *any* earlier row it may depend on (rolling windows) was available.
    At each origin the latest row with ``available_time <= origin`` is taken; values older than
    ``max_staleness`` (origin - event_time) become NaN.

    Args:
        frame: one row per event date with ``event_col``, ``available_col`` (tz-aware UTC) and
            ``value_columns``.
        origins: forecast origins.
        value_columns: columns to carry.

    Returns:
        Frame indexed by ``origin_time`` with ``value_columns`` (float).
    """
    idx = as_origin_index(origins)
    out = pd.DataFrame(np.nan, index=idx, columns=list(value_columns), dtype=float)
    if frame.empty or len(idx) == 0:
        return out
    if frame[event_col].duplicated().any():
        raise ValueError(f"align_to_origins: duplicate {event_col} values")
    df = frame.sort_values(event_col, kind="mergesort").reset_index(drop=True)
    right = pd.DataFrame(
        {
            "_et": _ns(df[event_col]),
            "_avail": np.maximum.accumulate(_ns(df[available_col])),
        }
    )
    for c in value_columns:
        right[c] = df[c].astype(float).to_numpy()
    # equal availability -> keep the latest event (cummax makes ties only in that direction)
    right = right.drop_duplicates("_avail", keep="last")
    left = pd.DataFrame({"_origin": np.sort(idx.asi8)})
    m = pd.merge_asof(left, right, left_on="_origin", right_on="_avail", direction="backward")
    if max_staleness is not None:
        stale = (m["_origin"] - m["_et"]) > pd.Timedelta(max_staleness).value
        m.loc[stale.fillna(False).to_numpy(dtype=bool), list(value_columns)] = np.nan
    m.index = pd.to_datetime(m["_origin"], utc=True)
    out.loc[:, :] = m[list(value_columns)].reindex(idx).to_numpy(dtype=float)
    return out
