"""Bulk pull of Bloomberg weather through BQL in Excel, stored with the ingest pipeline (D-042).

Forecasts: one query per (as_of, source, model), all locations and fields, leads 0..14. Actuals: one query per
calendar year. Normals: one query per 16-day block of a reference year (a day-of-year table).

The runner is injectable so the orchestration can be tested without Excel. Production uses
``data.bql_excel.run_queries`` (user-authorised route, 2026-10-19). Nothing here runs a model.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ttf_forecaster.data import weather_bql as wb
from ttf_forecaster.data.base import DataError
from ttf_forecaster.data.storage import Store
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.pipelines.ingest import already_stored, ingest

log = get_logger(__name__)

Runner = Callable[..., Mapping[str, object]]
SOURCE = wb.SOURCE
AS_OF_PER_BATCH = 10  # 10 as_of x 2 sources = 20 queries per Excel book (limit 20)
NORMAL_REFERENCE_YEAR = 2026
TODAY = date(2026, 10, 19)  # the pull's last observed day
ACTUAL_TRAILING_TOLERANCE = 5  # days a location may lag the period end (latest observations)
NORMAL_BLOCK_DAYS = 16


@dataclass
class PullSummary:
    """What one pull did: rows stored per (kind, model) and every failure, for the report."""

    rows: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    audits: list[str] = field(default_factory=list)
    batches: int = 0
    seconds: float = 0.0

    def add(self, key: str, n: int) -> None:
        self.rows[key] = self.rows.get(key, 0) + n


def _ingest(frame: pd.DataFrame, store: Store, kind: str, raw_name: str, ingestion: pd.Timestamp) -> int:
    """Store one chunk. Idempotent: a chunk whose raw name is already stored is not written again."""
    if already_stored(store, SOURCE, raw_name):
        log.info("skipped %s: already stored", raw_name)
        return 0
    s = ingest(lambda f=frame: f, store, kind=kind, source=SOURCE, raw_name=raw_name,
               keep_adapter_ingestion_time=True, fail_on_quality=False)
    return int(s.n_written)


def audit_dates(runner: Runner, as_of: date) -> list[str]:
    """Dates-only audit for one as_of. Valid dates must be as_of + 0..14 (lead = row index)."""
    res = runner({"audit": wb.dates_query(as_of)}, timeout_s=240)["audit"]
    if not isinstance(res, pd.DataFrame):
        raise DataError(f"date audit failed for {as_of}: {res}")
    got = [pd.Timestamp(d).date() for d in res["DATES"]]
    want = [as_of + timedelta(days=i) for i in range(wb.LEADS)]
    if got != want:
        raise DataError(f"lead mapping audit failed for {as_of}: got {got[:3]}..., want {want[:3]}...")
    return [f"{as_of} dates ok ({got[0]}..{got[-1]})"]


def pull_forecasts(first: date, last: date, store: Store, runner: Runner, *,
                   batch_size: int = AS_OF_PER_BATCH, chunk_days: int = 60,
                   summary: PullSummary | None = None) -> PullSummary:
    """Forecast vintages for every as_of in [first, last]: per as_of, one query per source (ECMWF, GFS), each
    carrying every model type, all locations and fields. ``batch_size`` as_of days per Excel book (<= 20 queries)."""
    summary = summary or PullSummary()
    t0 = datetime.now(timezone.utc)
    days = list(wb.iter_as_of(first, last))
    buffer: list[pd.DataFrame] = []
    chunk_start = days[0] if days else first
    for i, batch in enumerate(wb.batches(days, batch_size)):
        queries: dict[str, str] = {}
        index: dict[str, tuple[date, str]] = {}
        for d in batch:
            for source in wb.FORECAST_BUNDLES:
                name = f"{d:%Y%m%d}_{source}"
                queries[name] = wb.forecast_query(d, source)
                index[name] = (d, source)
        ingestion = pd.Timestamp(datetime.now(timezone.utc))
        res = runner(queries, timeout_s=300)
        summary.batches += 1
        for name, value in res.items():
            d, source = index[name]
            if not isinstance(value, pd.DataFrame):
                summary.failures.append(f"{name}: {str(value)[:200]}")
                continue
            try:
                frame = wb.forecast_frame(value, d, source, ingestion)
            except ValueError as exc:
                summary.failures.append(f"{name}: shape {exc}")
                continue
            buffer.append(frame)
            for model, grp in frame.groupby("model"):
                summary.add(f"forecast {model}", len(grp))
        if i == 0:
            summary.audits.extend(audit_dates(runner, batch[0]))
        if len(buffer) >= len(wb.FORECAST_BUNDLES) * chunk_days:
            _ingest(pd.concat(buffer, ignore_index=True), store, "forecast",
                    f"bql_weather_fc_{chunk_start:%Y%m%d}_{batch[-1]:%Y%m%d}", ingestion)
            buffer = []
            chunk_start = batch[-1] + timedelta(days=1)
            log.info("forecast chunk written up to %s", batch[-1])
    if buffer:
        _ingest(pd.concat(buffer, ignore_index=True), store, "forecast",
                f"bql_weather_fc_{chunk_start:%Y%m%d}_{days[-1]:%Y%m%d}", pd.Timestamp(datetime.now(timezone.utc)))
    summary.seconds += (datetime.now(timezone.utc) - t0).total_seconds()
    return summary


# (BQL actual series, WSI reference series). The WSI rows are stored at Amsterdam midnight (23:00 or 22:00 UTC the
# day before), so their Amsterdam date is the one that counts. Each pair must correlate best at lag 0.
ALIGNMENT_PAIRS = (
    ("weather.de.t2m_mean_obs_degc", "weather.de.regional_t2m_obs_degc"),
    ("weather.gb.t2m_mean_obs_degc", "weather.gb.heathrow_t2m_obs_degc"),
    ("weather.nl.t2m_mean_obs_degc", "weather.nl.schiphol_t2m_obs_degc"),
)
ALIGNMENT_LAGS = range(-2, 3)
ACTUAL_YEARS_PER_BATCH = 8  # two queries per year (dates and values): 16 queries per Excel book
REFERENCE_SERIES = tuple(sorted({ref for _, ref in ALIGNMENT_PAIRS}))


def _daily(frame: pd.DataFrame, tz: str | None) -> pd.Series:
    """One value per calendar day: the UTC date for BQL (tz None), the local date for WSI."""
    ts = pd.to_datetime(frame["event_time"], utc=True)
    day = ts.dt.tz_localize(None).dt.normalize() if tz is None else ts.dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    return pd.Series(frame["value"].to_numpy(), index=day.to_numpy()).groupby(level=0).first()


def check_actuals_alignment(actuals: pd.DataFrame, reference: pd.DataFrame,
                            pairs: Sequence[tuple[str, str]] = ALIGNMENT_PAIRS) -> dict[str, dict[int, float]]:
    """Data check (D-045): each BQL actual series must correlate best with its WSI reference on the SAME day.

    Returns ``{bql_series: {lag: correlation}}``. ``lag k`` pairs the BQL value of day D with the WSI value of day D + k.
    Raises ``DataError`` if a series or its reference is missing, if there is too little overlap, or if the best lag
    is not 0 for any pair.
    """
    result: dict[str, dict[int, float]] = {}
    for bql_sid, ref_sid in pairs:
        bql = actuals.loc[actuals["series_id"] == bql_sid]
        ref = reference.loc[reference["series_id"] == ref_sid]
        if bql.empty or ref.empty:
            raise DataError(f"alignment check needs {bql_sid} and {ref_sid}; got {len(bql)} and {len(ref)} rows")
        b = _daily(bql, None)
        r = _daily(ref, "Europe/Amsterdam")
        corr: dict[int, float] = {}
        for lag in ALIGNMENT_LAGS:
            shifted = r.copy()
            shifted.index = shifted.index - pd.Timedelta(days=lag)  # the WSI value of D + lag, indexed at D
            joined = pd.concat([b, shifted], axis=1, join="inner").dropna()
            if len(joined) < 30:
                raise DataError(f"{bql_sid}: only {len(joined)} overlapping days with {ref_sid} at lag {lag}")
            corr[lag] = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        best = max(corr, key=corr.get)
        if best != 0:
            raise DataError(f"{bql_sid} aligns best at lag {best:+d} with {ref_sid} (correlations {corr}); "
                            "the observed-day labels are wrong (D-045)")
        result[bql_sid] = corr
    return result


def pull_actuals(years: Sequence[int], store: Store, runner: Runner, *,
                 summary: PullSummary | None = None) -> PullSummary:
    """Observed daily values for each calendar year, stamped with the observed day.

    BQL labels the value of a requested day with the next day (D-045), so the request for year Y covers the
    period (Y-01-01 minus 1 day) .. (Y-12-31 minus 1 day), and the labels (from the dates-only query) give Y-01-01..Y-12-31.
    Each year's block is checked against the WSI regional series before it is stored.
    """
    summary = summary or PullSummary()
    years = list(years)
    for batch in wb.batches(years, ACTUAL_YEARS_PER_BATCH):
        queries: dict[str, str] = {}
        periods: dict[int, tuple[date, date]] = {}
        for y in batch:
            req_start = date(y, 1, 1) - timedelta(days=1)
            req_end = date(y, 12, 31) - timedelta(days=1)
            periods[y] = (req_start, req_end)
            queries[f"actd_{y}"] = wb.actual_dates_query(req_start, req_end)
            queries[f"act_{y}"] = wb.actual_query(req_start, req_end)
        res = runner(queries, timeout_s=300)
        summary.batches += 1
        ingestion = pd.Timestamp(datetime.now(timezone.utc))
        for y in batch:
            req_start, req_end = periods[y]
            value = res[f"act_{y}"]
            dates = res[f"actd_{y}"]
            if not isinstance(value, pd.DataFrame) or not isinstance(dates, pd.DataFrame):
                summary.failures.append(f"act_{y}: {str(value)[:120]} / dates {str(dates)[:120]}")
                continue
            try:
                labels = wb.actual_event_dates(dates, req_start, req_end)
            except DataError as exc:
                summary.failures.append(f"act_{y} dates: {exc}")
                continue
            last = min(date(y, 12, 31), TODAY)
            expected = (last - date(y, 1, 1)).days + 1
            try:
                frame = wb.observation_frame(value, labels, ingestion)
            except ValueError as exc:
                summary.failures.append(f"act_{y}: {exc}")
                continue
            # per location: the span of its observed days must cover the year up to the last published day
            spans = {loc: (lab[-1] - lab[0]).days + 1 for loc, lab in labels.items()}
            short = {loc: n for loc, n in spans.items() if n < expected - ACTUAL_TRAILING_TOLERANCE or n > expected}
            if short:
                summary.failures.append(f"act_{y}: spans {short}, want {expected}")
                continue
            for loc, lab in labels.items():
                gaps = wb.missing_days(lab)
                if gaps:
                    summary.audits.append(f"act_{y} {loc}: BQL returned no value for {[g.isoformat() for g in gaps]}")
            if any(n < expected for n in spans.values()):
                summary.audits.append(f"act_{y}: trailing days not yet published (spans {sorted(set(spans.values()))})")
            reference = store.read_normalised("observation", series_ids=list(REFERENCE_SERIES))
            align = check_actuals_alignment(frame, reference)
            summary.audits.append(f"act_{y}: alignment best lag 0 for {sorted(align)}")
            n = _ingest(frame, store, "observation", f"bql_weather_obs_{y}", ingestion)
            summary.add("actual", n)
    return summary


def pull_normals(store: Store, runner: Runner, *, summary: PullSummary | None = None) -> PullSummary:
    """Day-of-year normals (NORMALS_10Y and NORMALS_30Y). Stored with grade C: they use hindsight for backtests.

    The table lives in the reference year: each 16-day block is requested with its own as_of, and its rows are
    the dates of that block. Calendar days beyond the reference year are not stored.
    """
    summary = summary or PullSummary()
    starts = [date(NORMAL_REFERENCE_YEAR, 1, 1) + timedelta(days=NORMAL_BLOCK_DAYS * k) for k in range(24)]
    ingestion = pd.Timestamp(datetime.now(timezone.utc))
    for source in wb.NORMAL_SOURCES:
        for batch in wb.batches(starts, 20):
            queries = {f"nrm_{s:%Y%m%d}_{source}": wb.normals_query(s, source) for s in batch}
            res = runner(queries, timeout_s=240)
            for s in batch:
                value = res[f"nrm_{s:%Y%m%d}_{source}"]
                if not isinstance(value, pd.DataFrame):
                    summary.failures.append(f"nrm {s} {source}: {str(value)[:200]}")
                    continue
                frame = wb.normal_frame(value, [s + timedelta(days=i) for i in range(NORMAL_BLOCK_DAYS)],
                                        source, ingestion)
                frame = frame[frame["event_time"].dt.year == NORMAL_REFERENCE_YEAR]
                n = _ingest(frame, store, "observation", f"bql_weather_norm_{s:%Y%m%d}_{source}", ingestion)
                summary.add(f"normal {source}", n)
    summary.batches += 1
    return summary
