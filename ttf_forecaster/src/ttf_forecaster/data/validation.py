"""Data-quality checks beyond schema validation. Pure functions: inputs are never modified.

Schema validation (columns, dtypes, UTC, nulls) is ``data.base.validate_frame``. This module checks
the *content* of canonical frames:

* duplicate dedupe keys, separating harmless exact duplicates from conflicting re-deliveries;
* revision monotonicity (revision n must be published strictly after revision n-1);
* unit consistency per series (and against an expected unit);
* staleness of each series against its expected frequency at a given ``as_of``;
* robust-z outlier flags computed on a **trailing** window (flag, never drop);
* coverage gaps against a supplied calendar of expected dates;
* series-id naming convention (``<domain>.<geography>.<measure>_<unit>``, advisory only);
* a per-series summary report combining the above.

Outlier statistics use only observations *before* each row (trailing median/MAD), so a flag can be
reproduced at any origin and does not encode hindsight. Thresholds live in ``QualityParams`` and
are overridable from configuration (``QualityParams.from_config``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    MARKET_TZ,
    DataError,
    QualityFlag,
)
from ttf_forecaster.data.storage import get_kind, is_synthetic_frame

MAD_TO_SIGMA = 1.4826  # consistency constant: MAD * 1.4826 estimates sigma for normal data

HARD_CHECKS: tuple[str, ...] = (
    "conflicting_duplicates",
    "unit_inconsistency",
    "revision_non_monotone",
)

SERIES_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z0-9_]+\.[a-z0-9_]+$")

# Columns grouped when checking unit consistency (curve units are per hub and field).
_UNIT_GROUP: dict[str, list[str]] = {
    "observation": ["series_id"],
    "forecast": ["series_id"],
    "curve": ["hub", "field"],
    "contract_metadata": ["hub"],
}
# Columns that identify one ordered series for outlier/staleness/coverage purposes.
_SERIES_GROUP: dict[str, list[str]] = {
    "observation": ["series_id"],
    "forecast": ["series_id", "model", "member"],
    "curve": ["contract_id", "field"],
}


class DataQualityError(DataError):
    """One or more hard data-quality checks failed. The message lists each failure."""


@dataclass(frozen=True)
class QualityParams:
    """Thresholds for the quality checks.

    Attributes:
        outlier_window: trailing number of observations used for the median/MAD.
        outlier_min_periods: minimum trailing observations before a z-score is computed.
        outlier_z_threshold: |robust z| above which a row is flagged ``outlier_suspect``.
        outlier_transform: ``"diff"`` (first differences, robust to trends/seasonality) or
            ``"level"``.
        staleness_periods: a series is stale if ``as_of`` is later than the last event plus this
            many periods of its expected frequency (plus any per-series grace).
        hard_checks: names of checks whose failure blocks ingestion (subset of ``HARD_CHECKS``).
    """

    outlier_window: int = 60
    outlier_min_periods: int = 20
    outlier_z_threshold: float = 6.0
    outlier_transform: str = "diff"
    staleness_periods: int = 3
    hard_checks: tuple[str, ...] = HARD_CHECKS

    def __post_init__(self) -> None:
        if self.outlier_transform not in ("diff", "level"):
            raise ValueError(f"outlier_transform must be 'diff' or 'level', got "
                             f"{self.outlier_transform!r}")
        if self.outlier_min_periods > self.outlier_window:
            raise ValueError("outlier_min_periods cannot exceed outlier_window")
        unknown = set(self.hard_checks) - set(HARD_CHECKS)
        if unknown:
            raise ValueError(f"Unknown hard checks {sorted(unknown)}; valid: {HARD_CHECKS}")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> QualityParams:
        """Build from a mapping (e.g. a ``data_quality`` YAML block); unknown keys raise."""
        if not cfg:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(f"Unknown QualityParams keys {sorted(unknown)}; allowed {sorted(allowed)}")
        kwargs = dict(cfg)
        if "hard_checks" in kwargs:
            kwargs["hard_checks"] = tuple(kwargs["hard_checks"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------------------------


def find_duplicate_keys(
    df: pd.DataFrame, key: Sequence[str], value_columns: Sequence[str] = ("value",)
) -> pd.DataFrame:
    """Rows sharing ``key``, one output row per duplicated key.

    Returns:
        Columns ``*key, n_rows, n_distinct, conflicting``. ``conflicting`` is True when the rows
        differ in any of ``value_columns`` (NaN equals NaN); exact duplicates are harmless.
    """
    key = list(key)
    cols = [*key, "n_rows", "n_distinct", "conflicting"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    dup = df.loc[df.duplicated(subset=key, keep=False), key + list(value_columns)]
    if dup.empty:
        return pd.DataFrame(columns=cols)
    filled = dup.copy()
    for c in value_columns:
        filled[c] = filled[c].astype(object).where(filled[c].notna(), "<NA>")
    distinct = filled.drop_duplicates()
    out = (
        dup.groupby(key, dropna=False).size().rename("n_rows").to_frame()
        .join(distinct.groupby(key, dropna=False).size().rename("n_distinct"))
        .reset_index()
    )
    out["conflicting"] = out["n_distinct"] > 1
    return out[cols]


def check_revision_monotonicity(df: pd.DataFrame, fact_key: Sequence[str]) -> pd.DataFrame:
    """Rows whose ``revision`` n is not published strictly after revision n-1 of the same fact.

    Args:
        df: vintaged frame with ``revision`` and ``publication_time``.
        fact_key: columns identifying one fact across vintages (dedupe key without revision and
            publication_time), e.g. ``["series_id", "event_time"]``.

    Returns:
        Offending rows with ``*fact_key, revision, publication_time, prev_revision,
        prev_publication_time``. Revision numbering gaps (0 then 2) are **not** violations: we may
        not have captured every vintage.
    """
    fact_key = list(fact_key)
    cols = [*fact_key, "revision", "publication_time", "prev_revision", "prev_publication_time"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    per_rev = (
        df.sort_values([*fact_key, "revision", "publication_time"])
        .drop_duplicates([*fact_key, "revision"], keep="first")  # earliest pub of each revision
        .loc[:, [*fact_key, "revision", "publication_time"]]
        .reset_index(drop=True)
    )
    g = per_rev.groupby(fact_key, sort=False, dropna=False)
    per_rev["prev_revision"] = g["revision"].shift(1)
    per_rev["prev_publication_time"] = g["publication_time"].shift(1)
    bad = per_rev["prev_revision"].notna() & (
        per_rev["publication_time"] <= per_rev["prev_publication_time"]
    )
    out = per_rev.loc[bad, cols].reset_index(drop=True)
    out["prev_revision"] = out["prev_revision"].astype("int64")
    return out


def check_unit_consistency(
    df: pd.DataFrame,
    group_columns: Sequence[str] = ("series_id",),
    expected_units: Mapping[str, str] | None = None,
    unit_column: str = "unit",
) -> pd.DataFrame:
    """One row per group: the units seen, whether there is exactly one, and whether it matches.

    ``expected_units`` is keyed by the first group column (e.g. series_id); groups not in it get
    ``expected_unit = None`` and ``matches_expected = None``.
    """
    group_columns = list(group_columns)
    cols = [*group_columns, "units", "n_units", "consistent", "expected_unit", "matches_expected"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    units = (
        df.groupby(group_columns, dropna=False)[unit_column]
        .agg(lambda s: tuple(sorted(set(s.dropna().astype(str)))))
        .rename("units")
        .reset_index()
    )
    units["n_units"] = units["units"].map(len)
    units["consistent"] = units["n_units"] == 1
    exp = expected_units or {}
    first = group_columns[0]
    units["expected_unit"] = units[first].map(lambda k: exp.get(k))
    units["matches_expected"] = pd.Series([
        None if e is None else (u == (e,)) for u, e in zip(units["units"], units["expected_unit"],
                                                          strict=True)
    ], index=units.index, dtype=object)
    return units[cols]


def _offset(freq: str | pd.DateOffset) -> pd.DateOffset:
    try:
        return pd.tseries.frequencies.to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"Invalid expected frequency {freq!r}") from exc


def check_staleness(
    df: pd.DataFrame,
    as_of: pd.Timestamp | str,
    expected_frequency: str | pd.DateOffset | Mapping[str, str | pd.DateOffset],
    *,
    series_column: str = "series_id",
    event_column: str = "event_time",
    grace: Mapping[str, pd.Timedelta] | pd.Timedelta | None = None,
    params: QualityParams | None = None,
) -> pd.DataFrame:
    """Per series: last event and publication visible at ``as_of`` and whether it is stale.

    Only rows with ``publication_time <= as_of`` are considered (a value not yet published cannot
    make a series fresh). A series is stale when
    ``as_of > last_event + staleness_periods * frequency + grace``. ``grace`` should include the
    series' publication lag (e.g. 2 days for lag-approximated storage).

    Returns:
        Columns ``series, last_event_time, last_publication_time, age, deadline, is_stale``.
        Series with an expected frequency but no visible rows are stale with NaT times.
    """
    p = params or QualityParams()
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        raise ValueError("as_of must be tz-aware")
    ts = ts.tz_convert("UTC")
    vis = df.loc[df["publication_time"] <= ts] if len(df) else df
    last = (
        vis.groupby(series_column)
        .agg(last_event_time=(event_column, "max"), last_publication_time=("publication_time", "max"))
        if len(vis) else pd.DataFrame(columns=["last_event_time", "last_publication_time"])
    )
    if isinstance(expected_frequency, Mapping):
        freq_map = dict(expected_frequency)
        series = sorted(set(freq_map) | set(df[series_column].unique() if len(df) else []))
    else:
        series = sorted(df[series_column].unique()) if len(df) else []
        freq_map = {s: expected_frequency for s in series}
    rows = []
    for s in series:
        freq = freq_map.get(s)
        last_ev = last["last_event_time"].get(s, pd.NaT) if len(last) else pd.NaT
        last_pub = last["last_publication_time"].get(s, pd.NaT) if len(last) else pd.NaT
        g = grace.get(s, pd.Timedelta(0)) if isinstance(grace, Mapping) else (grace or pd.Timedelta(0))
        if freq is None:
            deadline, stale = pd.NaT, None
        elif pd.isna(last_ev):
            deadline, stale = pd.NaT, True
        else:
            deadline = last_ev + p.staleness_periods * _offset(freq) + g
            stale = bool(ts > deadline)
        rows.append({
            "series": s,
            "last_event_time": last_ev,
            "last_publication_time": last_pub,
            "age": (ts - last_ev) if not pd.isna(last_ev) else pd.NaT,
            "deadline": deadline,
            "is_stale": stale,
        })
    return pd.DataFrame(rows, columns=["series", "last_event_time", "last_publication_time", "age",
                                       "deadline", "is_stale"])


def _trailing_mad(window: np.ndarray) -> float:
    w = window[~np.isnan(window)]
    if not len(w):
        return float("nan")
    med = np.median(w)
    return float(np.median(np.abs(w - med)))


def flag_outliers_robust_z(
    df: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("series_id",),
    order_column: str = "event_time",
    value_column: str = "value",
    params: QualityParams | None = None,
    set_quality_flag: bool = False,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``robust_z`` and ``outlier_suspect`` columns. Never drops rows.

    For each group ordered by ``order_column``, ``x`` is the value (or its first difference) and
    ``robust_z = (x - median) / (1.4826 * MAD)`` where median and MAD use only the previous
    ``outlier_window`` values of ``x`` (the current row is excluded). Fewer than
    ``outlier_min_periods`` prior values gives ``robust_z = NaN`` and no flag. A zero MAD with
    ``x != median`` gives ``±inf`` (flagged).

    The frame must have one row per ``(group, order)``: pass a single-vintage view (first release
    or as-of resolved), not all revisions.

    Args:
        set_quality_flag: also set ``quality_flag = outlier_suspect`` on flagged rows whose flag is
            ``ok`` (other flags, e.g. ``synthetic``, are left untouched).
    """
    p = params or QualityParams()
    group_columns = list(group_columns)
    out = df.copy()
    if out.empty:
        out["robust_z"] = pd.Series(dtype="float64")
        out["outlier_suspect"] = pd.Series(dtype="bool")
        return out
    if out.duplicated(subset=[*group_columns, order_column]).any():
        raise ValueError(
            f"flag_outliers_robust_z needs unique {[*group_columns, order_column]}; resolve "
            "vintages first (resolve_as_of or a first-release view)"
        )
    out = out.sort_values([*group_columns, order_column])
    z = pd.Series(np.nan, index=out.index)
    for _, g in out.groupby(group_columns, sort=False, dropna=False):
        x = g[value_column].astype(float)
        if p.outlier_transform == "diff":
            x = x.diff()
        prior = x.shift(1)
        roll = prior.rolling(p.outlier_window, min_periods=p.outlier_min_periods)
        med = roll.median()
        mad = roll.apply(_trailing_mad, raw=True)
        dev = x - med
        scale = MAD_TO_SIGMA * mad
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = np.where(scale > 0, dev / scale,
                          np.where(dev.abs() > 0, np.sign(dev) * np.inf, 0.0))
        zz = pd.Series(zz, index=g.index).where(med.notna() & mad.notna() & x.notna())
        z.loc[g.index] = zz
    out["robust_z"] = z.astype(float)
    out["outlier_suspect"] = (out["robust_z"].abs() > p.outlier_z_threshold).fillna(False)
    out["outlier_suspect"] = out["outlier_suspect"].astype(bool)
    if set_quality_flag and "quality_flag" in out.columns:
        hit = out["outlier_suspect"] & (out["quality_flag"] == QualityFlag.OK.value)
        out.loc[hit, "quality_flag"] = QualityFlag.OUTLIER_SUSPECT.value
    return out.loc[df.index]


def _to_local_dates(values: pd.Series | pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert(tz).tz_localize(None)
    return idx.normalize()


def _calendar_local(calendar: pd.DatetimeIndex | Sequence[pd.Timestamp], tz: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(_to_local_dates(pd.DatetimeIndex(calendar), tz))))


def coverage_summary(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex | Sequence[pd.Timestamp],
    *,
    series_column: str = "series_id",
    event_column: str = "event_time",
    tz: str = MARKET_TZ,
    within_series_span: bool = True,
    series_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calendar coverage per series, comparing **local dates** in ``tz``.

    Args:
        calendar: expected dates. Naive values are local dates in ``tz``; tz-aware values are
            converted to ``tz`` first.
        within_series_span: only expect calendar dates between a series' first and last event
            (avoids flagging pre-history). With ``False`` the whole calendar is expected.
        series_ids: series to check (default: those present). A listed series with no rows is
            missing every calendar date.

    Returns:
        ``(summary, gaps)``: ``summary`` has ``series, n_expected, n_missing, coverage_ratio``;
        ``gaps`` has ``series, missing_date`` (naive local dates), one row per gap. Rows on dates
        outside the calendar (e.g. weekends for a business-day calendar) are ignored.
    """
    cal = _calendar_local(calendar, tz)
    ids = list(series_ids) if series_ids is not None else (
        sorted(df[series_column].unique()) if len(df) else [])
    gaps: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for s in ids:
        sub = df.loc[df[series_column] == s, event_column] if len(df) else pd.Series(dtype=object)
        have = set(_to_local_dates(sub, tz)) if len(sub) else set()
        expected = cal
        if within_series_span and have:
            expected = cal[(cal >= min(have)) & (cal <= max(have))]
        missing = [d for d in expected if d not in have]
        gaps.extend({"series": s, "missing_date": d} for d in missing)
        n_exp = len(expected)
        summary.append({"series": s, "n_expected": n_exp, "n_missing": len(missing),
                        "coverage_ratio": (1.0 - len(missing) / n_exp) if n_exp else np.nan})
    return (pd.DataFrame(summary, columns=["series", "n_expected", "n_missing", "coverage_ratio"]),
            pd.DataFrame(gaps, columns=["series", "missing_date"]))


def check_coverage(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex | Sequence[pd.Timestamp],
    **kwargs: Any,
) -> pd.DataFrame:
    """Gaps only (``series, missing_date``); see ``coverage_summary`` for arguments."""
    return coverage_summary(df, calendar, **kwargs)[1]


def unit_token(unit: str) -> str:
    """Lower-case token of a unit for series-id suffixes: ``EUR/MWh`` -> ``eur_mwh``,
    ``GWh/d`` -> ``gwh_d``, ``%`` -> ``pct``."""
    t = unit.strip().lower().replace("%", "pct").replace("/", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", t)


def unit_suffix_matches(series_id: str, unit: str) -> bool:
    """Whether the last segment of ``series_id`` ends with a token of ``unit``.

    Underscores are ignored and ``/`` may be written ``_``, ``per`` or omitted, so ``GWh/d`` matches
    ``..._gwh_d`` and ``..._gwhd``, and ``USD/EUR`` matches ``..._usd_per_eur``.
    """
    tail = series_id.rsplit(".", 1)[-1].replace("_", "")
    base = unit_token(unit).replace("_", "")
    per = unit_token(unit.replace("/", " per ")).replace("_", "")
    return tail.endswith(base) or tail.endswith(per)


def check_series_id_convention(
    series_units: Mapping[str, str] | Sequence[str],
) -> pd.DataFrame:
    """Advisory check of ``<domain>.<geography>.<measure>_<unit>``.

    Args:
        series_units: series ids, or a mapping ``series_id -> unit`` to also check that the last
            segment ends with ``unit_token(unit)``.

    Returns:
        Columns ``series_id, pattern_ok, unit_suffix_ok`` (``None`` when no unit was given).
    """
    items = series_units.items() if isinstance(series_units, Mapping) else \
        ((s, None) for s in series_units)
    rows = []
    for sid, unit in items:
        pattern_ok = bool(SERIES_ID_PATTERN.match(sid))
        suffix_ok = None if unit is None else unit_suffix_matches(sid, unit)
        rows.append({"series_id": sid, "pattern_ok": pattern_ok, "unit_suffix_ok": suffix_ok})
    return pd.DataFrame(rows, columns=["series_id", "pattern_ok", "unit_suffix_ok"])


# ---------------------------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------------------------


@dataclass
class QualityResult:
    """Output of ``assess_quality``.

    Attributes:
        report: one row per series (see ``assess_quality``).
        hard_failures: human-readable descriptions of failed hard checks.
        warnings: advisory findings (outliers, staleness, coverage, naming, exact duplicates).
        details: the individual check frames, keyed by check name.
    """

    report: pd.DataFrame
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def raise_if_failed(self) -> None:
        if self.hard_failures:
            raise DataQualityError("Hard data-quality checks failed: " + "; ".join(self.hard_failures))


def _single_vintage_view(df: pd.DataFrame, fact_key: list[str]) -> pd.DataFrame:
    """Latest stored revision per fact — a diagnostic view for outliers only (not for features)."""
    return (df.sort_values([*fact_key, "revision", "publication_time"])
            .drop_duplicates(fact_key, keep="last"))


def assess_quality(
    df: pd.DataFrame,
    kind: str,
    *,
    as_of: pd.Timestamp | str | None = None,
    expected_frequency: str | Mapping[str, str] | None = None,
    staleness_grace: Mapping[str, pd.Timedelta] | pd.Timedelta | None = None,
    calendar: pd.DatetimeIndex | None = None,
    expected_units: Mapping[str, str] | None = None,
    params: QualityParams | None = None,
) -> QualityResult:
    """Run every applicable check on a canonical frame of ``kind`` and summarise per series.

    The report has one row per series (``series_column`` of the kind) with: ``n_rows``,
    ``n_facts``, ``max_revision``, ``first_event``, ``last_event``, ``last_publication``,
    ``units``, ``unit_consistent``, ``unit_matches_expected``, ``n_duplicate_keys``,
    ``n_conflicting_keys``, ``n_revision_violations``, ``n_outliers``, ``is_stale``,
    ``n_missing_dates``, ``coverage_ratio``, ``pit_grades``, ``is_synthetic``.

    Staleness needs ``as_of`` and ``expected_frequency``; coverage needs ``calendar``; otherwise
    those columns are null. Hard failures are listed per ``params.hard_checks``.
    """
    p = params or QualityParams()
    k = get_kind(kind)
    res = QualityResult(report=pd.DataFrame())
    series_col = k.series_column
    if df.empty:
        res.warnings.append(f"{kind}: empty frame")
        res.report = pd.DataFrame(columns=[series_col, "n_rows"])
        return res

    # duplicates on the dedupe key
    dups = find_duplicate_keys(df, k.dedupe_key, k.value_columns)
    res.details["duplicates"] = dups
    n_conf = int(dups["conflicting"].sum()) if len(dups) else 0
    if n_conf and "conflicting_duplicates" in p.hard_checks:
        res.hard_failures.append(f"{n_conf} dedupe keys with conflicting values")
    if len(dups) - n_conf:
        res.warnings.append(f"{len(dups) - n_conf} exact duplicate keys (will be deduplicated)")

    # revisions
    fact_key = [c for c in k.dedupe_key if c not in ("revision", "publication_time")]
    if k.vintaged:
        revs = check_revision_monotonicity(df, fact_key)
        res.details["revisions"] = revs
        if len(revs) and "revision_non_monotone" in p.hard_checks:
            res.hard_failures.append(
                f"{len(revs)} revisions not published strictly after their predecessor")
    else:
        revs = pd.DataFrame(columns=fact_key)

    # units
    ugroup = _UNIT_GROUP[kind]
    units = check_unit_consistency(df, ugroup, expected_units)
    res.details["units"] = units
    bad_units = units.loc[~units["consistent"]]
    if len(bad_units) and "unit_inconsistency" in p.hard_checks:
        res.hard_failures.append(
            f"mixed units within {ugroup}: "
            + ", ".join(f"{r[ugroup[0]]}={list(r['units'])}" for _, r in bad_units.iterrows()))
    mism = units.loc[units["matches_expected"] == False]  # noqa: E712 (object column with None)
    if len(mism) and "unit_inconsistency" in p.hard_checks:
        res.hard_failures.append(
            "unit differs from expected: "
            + ", ".join(f"{r[ugroup[0]]}: {list(r['units'])} != {r['expected_unit']}"
                        for _, r in mism.iterrows()))

    # outliers (diagnostic, trailing window, on the latest stored vintage per fact)
    n_out = pd.Series(dtype="int64")
    if kind in _SERIES_GROUP:
        sgroup = _SERIES_GROUP[kind]
        view = _single_vintage_view(df, fact_key)
        if kind == "forecast":
            # one value per (series, model, member, valid_time): latest issue
            view = (view.sort_values([*sgroup, k.event_column, "issue_time"])
                    .drop_duplicates([*sgroup, k.event_column], keep="last"))
        flagged = flag_outliers_robust_z(view, group_columns=sgroup, order_column=k.event_column,
                                         params=p)
        res.details["outliers"] = flagged.loc[flagged["outlier_suspect"]]
        n_out = flagged.groupby(series_col)["outlier_suspect"].sum().astype("int64")
        if n_out.sum():
            res.warnings.append(f"{int(n_out.sum())} outlier-suspect rows (flagged, not dropped)")

    # staleness
    stale = None
    if as_of is not None and expected_frequency is not None and k.event_column is not None:
        stale = check_staleness(df, as_of, expected_frequency, series_column=series_col,
                                event_column=k.event_column, grace=staleness_grace, params=p)
        res.details["staleness"] = stale
        n_stale = int((stale["is_stale"] == True).sum())  # noqa: E712
        if n_stale:
            res.warnings.append(f"{n_stale} stale series at {as_of}")

    # coverage
    gaps = None
    cov = None
    if calendar is not None and k.event_column is not None:
        cov, gaps = coverage_summary(df, calendar, series_column=series_col,
                                     event_column=k.event_column)
        res.details["coverage"] = gaps
        res.details["coverage_summary"] = cov
        if len(gaps):
            res.warnings.append(f"{len(gaps)} missing calendar dates across series")

    # naming (observation/forecast series ids only)
    if kind in ("observation", "forecast"):
        first_unit = df.groupby("series_id")["unit"].first().to_dict()
        naming = check_series_id_convention(first_unit)
        res.details["naming"] = naming
        off = naming.loc[~naming["pattern_ok"] | (naming["unit_suffix_ok"] == False)]  # noqa: E712
        if len(off):
            res.warnings.append("series ids off naming convention: "
                                + ", ".join(off["series_id"].tolist()))

    # report
    g = df.groupby(series_col)
    rep = pd.DataFrame({"n_rows": g.size()})
    rep["n_facts"] = df.drop_duplicates(fact_key).groupby(series_col).size()
    if k.vintaged:
        rep["max_revision"] = g["revision"].max()
        rep["last_publication"] = g["publication_time"].max()
    if k.event_column is not None:
        rep["first_event"] = g[k.event_column].min()
        rep["last_event"] = g[k.event_column].max()
    if "unit" in df.columns:
        rep["units"] = g["unit"].agg(lambda s: tuple(sorted(set(s.astype(str)))))
        rep["unit_consistent"] = rep["units"].map(len) == 1
        if expected_units:
            rep["unit_matches_expected"] = [
                None if s not in expected_units else u == (expected_units[s],)
                for s, u in rep["units"].items()]
    rep["n_duplicate_keys"] = (dups.groupby(series_col).size() if len(dups) and series_col in dups
                               else pd.Series(dtype="int64"))
    rep["n_conflicting_keys"] = (dups.loc[dups["conflicting"]].groupby(series_col).size()
                                 if len(dups) and series_col in dups else pd.Series(dtype="int64"))
    rep["n_revision_violations"] = (revs.groupby(series_col).size()
                                    if len(revs) and series_col in revs
                                    else pd.Series(dtype="int64"))
    rep["n_outliers"] = n_out
    for c in ("n_duplicate_keys", "n_conflicting_keys", "n_revision_violations", "n_outliers"):
        rep[c] = rep[c].fillna(0).astype("int64")
    if stale is not None:
        rep["is_stale"] = stale.set_index("series")["is_stale"]
    if cov is not None:
        c = cov.set_index("series")
        rep["n_missing_dates"] = c["n_missing"].astype("int64")
        rep["coverage_ratio"] = c["coverage_ratio"]
    if "pit_grade" in df.columns:
        rep["pit_grades"] = g["pit_grade"].agg(lambda s: "".join(sorted(set(s))))
    rep["is_synthetic"] = [is_synthetic_frame(sub) for _, sub in g]
    res.report = rep.reset_index().rename(columns={"index": series_col})
    return res
