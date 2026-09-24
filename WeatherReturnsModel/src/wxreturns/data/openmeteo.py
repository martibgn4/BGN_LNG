"""Open-Meteo client: the four endpoints this project runs on.

WHY THIS SOURCE. It is the only free provider that answers "what did the
forecast say last January", which is the question the whole project rests on
(SPEC 0). Everything else here follows from that.

HOW HISTORY IS RECONSTRUCTED. Appending ``_previous_dayN`` to an HOURLY
variable returns what the model run from N days earlier said for that same
target hour, and it works against the multi-year archive rather than only the
live endpoint. So for a past target day we can recover the forecast at leads
1..7 and difference them into revisions.

The limits were probed against the live API on 2026-09-07 and are enforced
here rather than trusted:

  * N runs 1..7. Beyond that the API returns ALL-NULL rather than an error,
    which is the dangerous failure mode - a silent column of nulls becomes a
    silent column of zeros in a careless pipeline, and a zero revision reads as
    "the forecast did not change". `_previous_run_columns` refuses N > 7.
  * The previous-run archive starts around mid-2021.
  * ``_previous_dayN`` is hourly-only; the daily form is rejected outright, so
    daily aggregates are built here from the hourly series.

Licence: CC-BY-4.0, free tier non-commercial. Desk use needs the paid tier.
Set OPEN_METEO_API_KEY to route through it.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from wxreturns.config import get
from wxreturns.data.base import (
    FetchResult,
    SourceUnavailable,
    WeatherFetcher,
)

log = logging.getLogger(__name__)

DETERMINISTIC_MEMBER = -1
REANALYSIS_MODEL = "era5"


def _http_get_json(url: str, params: dict[str, Any]) -> Any:
    """GET with retry and backoff. Raises rather than returning a partial body.

    Open-Meteo signals errors two ways - an HTTP 4xx with a JSON ``reason``,
    and a 200 carrying nulls. Only the first is visible here; the null case is
    caught downstream where the expected shape is known.
    """
    timeout = get("sources", "open_meteo", "request_timeout_seconds")
    retries = get("sources", "open_meteo", "max_retries")
    backoff = get("sources", "open_meteo", "backoff_seconds")

    query = urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None}, safe=",")
    full = f"{url}?{query}"

    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:400]
            # A 400 is a malformed request and will fail identically on retry.
            if exc.code < 500 and exc.code != 429:
                raise SourceUnavailable(
                    f"Open-Meteo rejected the request ({exc.code}): {body}"
                ) from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
        time.sleep(backoff * (attempt + 1))

    raise SourceUnavailable(
        f"Open-Meteo unreachable after {retries} attempts: {last}. No substitute "
        "weather is produced."
    ) from last


def _as_list(payload: Any) -> list[dict]:
    """Open-Meteo returns a bare object for one point and a list for many."""
    return payload if isinstance(payload, list) else [payload]


def _chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def region_points(region: str) -> list[dict]:
    """The named points of a region, with weights normalised to sum to one."""
    points = get("regions", "regions", region, "points")
    total = float(sum(p["weight"] for p in points))
    if total <= 0.0:
        raise ValueError(f"region {region!r} has non-positive total weight")
    return [{**p, "weight": float(p["weight"]) / total} for p in points]


class OpenMeteoFetcher(WeatherFetcher):
    """Base for the four endpoints. Handles batching, pacing and reshaping."""

    source = "open_meteo"
    credential_env_var = "OPEN_METEO_API_KEY"
    credential_required = False          # free tier needs no key

    def _base_params(self, points: list[dict]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "latitude": ",".join(str(p["lat"]) for p in points),
            "longitude": ",".join(str(p["lon"]) for p in points),
            "timezone": get("sources", "open_meteo", "timezone"),
        }
        key = self.credential()
        if key:
            params["apikey"] = key
        return params

    def _paced_batches(self, points: list[dict]) -> list[list[dict]]:
        size = get("sources", "open_meteo", "max_points_per_request")
        return _chunk(points, size)

    @staticmethod
    def _pace() -> None:
        time.sleep(get("sources", "open_meteo", "min_seconds_between_requests"))

    @staticmethod
    def _daily_block(payload: dict, source: str) -> dict:
        block = payload.get("daily")
        if not block or "time" not in block:
            raise SourceUnavailable(f"{source} returned no daily block")
        return block

    @staticmethod
    def _hourly_block(payload: dict, source: str) -> dict:
        block = payload.get("hourly")
        if not block or "time" not in block:
            raise SourceUnavailable(f"{source} returned no hourly block")
        return block


class ArchiveFetcher(OpenMeteoFetcher):
    """ERA5 reanalysis: the actuals, and the basis for climatological normals.

    Reanalysis has no issue date - it is not a forecast - so ``issue_date`` is
    NaT and ``lead`` is undefined for these rows. Downstream code keys on that
    rather than on a sentinel date.
    """

    source = "open_meteo_archive"
    supports_historical_issue = True

    def fetch(self, region: str = "", start: date | None = None,
              end: date | None = None, variables: list[str] | None = None
              ) -> FetchResult:
        if not region:
            raise ValueError("region is required")
        if start is None or end is None:
            raise ValueError("archive fetch needs an explicit start and end")

        kind = get("regions", "regions", region, "kind")
        variables = variables or get("sources", "open_meteo",
                                     "daily_variables", kind)
        points = region_points(region)
        url = get("sources", "open_meteo", "endpoints", "archive")

        rows: list[pd.DataFrame] = []
        for batch in self._paced_batches(points):
            params = self._base_params(batch) | {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": ",".join(variables),
            }
            payloads = _as_list(_http_get_json(url, params))
            for point, payload in zip(batch, payloads, strict=True):
                block = self._daily_block(payload, self.source)
                frame = pd.DataFrame(block)
                melted = frame.melt(id_vars="time", var_name="variable",
                                    value_name="value")
                melted["target_date"] = pd.to_datetime(melted["time"])
                melted["issue_date"] = pd.NaT
                melted["region"] = region
                melted["point"] = point["name"]
                melted["model"] = REANALYSIS_MODEL
                melted["member"] = DETERMINISTIC_MEMBER
                rows.append(melted.drop(columns=["time"]))
            self._pace()

        panel = self._validate_panel(pd.concat(rows, ignore_index=True), self.source)
        return self._result(panel.dropna(subset=["value"]), self.source)


class HistoricalForecastFetcher(OpenMeteoFetcher):
    """Archived forecasts as issued, at leads 1..7. The heart of the project.

    For each target day this returns the analysis plus what the run from N days
    earlier said, for the requested N. Differencing consecutive N gives the
    revision panel of SPEC 0.

    Hourly is requested and aggregated to daily here, because the
    ``_previous_dayN`` suffix is rejected on daily variables. Aggregation is
    mean/max/min over the UTC day, matching how the daily endpoint defines its
    own aggregates - with the caveat that a "day" for degree-day purposes is a
    local-time construct, and UTC is used throughout so that DST cannot shift a
    day silently.
    """

    source = "open_meteo_historical_forecast"
    supports_historical_issue = True

    def _previous_run_columns(self, leads: list[int]) -> list[int]:
        limit = get("sources", "open_meteo", "previous_runs", "max_lead_days")
        bad = [n for n in leads if n < 1 or n > limit]
        if bad:
            raise SourceUnavailable(
                f"previous-run leads {bad} are outside the supported range "
                f"1..{limit}. Beyond {limit} the API returns a column of NULLS "
                "rather than an error, and a null revision silently becomes a "
                "zero revision - which asserts the forecast did not change. "
                "Far-lead revisions can only be accumulated live; see "
                "features.yaml lead_buckets historical: false."
            )
        return sorted(leads)

    def fetch(self, region: str = "", start: date | None = None,
              end: date | None = None, leads: list[int] | None = None,
              base_variable: str = "temperature_2m") -> FetchResult:
        if not region:
            raise ValueError("region is required")
        if start is None or end is None:
            raise ValueError("historical forecast fetch needs a start and end")

        limit = get("sources", "open_meteo", "previous_runs", "max_lead_days")
        leads = self._previous_run_columns(leads or list(range(1, limit + 1)))
        floor = pd.Timestamp(get("sources", "open_meteo", "previous_runs",
                                 "archive_verified_from")).date()
        if start < floor:
            raise SourceUnavailable(
                f"previous-run data is not available before {floor}; asked for "
                f"{start}. Earlier dates return nulls, not an error."
            )

        suffix = get("sources", "open_meteo", "previous_runs", "suffix_template")
        columns = [base_variable] + [
            base_variable + suffix.format(n=n) for n in leads]
        points = region_points(region)
        url = get("sources", "open_meteo", "endpoints", "historical_forecast")

        rows: list[pd.DataFrame] = []
        for batch in self._paced_batches(points):
            params = self._base_params(batch) | {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": ",".join(columns),
            }
            payloads = _as_list(_http_get_json(url, params))
            for point, payload in zip(batch, payloads, strict=True):
                block = self._hourly_block(payload, self.source)
                rows.append(self._reshape(block, region, point, base_variable,
                                          leads, suffix))
            self._pace()

        panel = self._validate_panel(pd.concat(rows, ignore_index=True), self.source)
        return self._result(panel, self.source)

    def _reshape(self, block: dict, region: str, point: dict,
                 base_variable: str, leads: list[int], suffix: str
                 ) -> pd.DataFrame:
        """Hourly columns to a daily panel keyed by (issue_date, target_date).

        The analysis column becomes lead 0. Each ``_previous_dayN`` column
        becomes issue_date = target_date - N days, which is what makes the
        revision a difference along issue_date at fixed target_date.
        """
        frame = pd.DataFrame(block)
        frame["timestamp"] = pd.to_datetime(frame["time"])
        frame["target_date"] = frame["timestamp"].dt.normalize()

        out: list[pd.DataFrame] = []
        for column in frame.columns:
            if not column.startswith(base_variable):
                continue
            tail = column[len(base_variable):]
            if tail == "":
                lead = 0
            else:
                expected = [suffix.format(n=n) for n in leads]
                if tail not in expected:
                    continue
                lead = leads[expected.index(tail)]

            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() == 0:
                # All-null is how the API refuses an unavailable lead. Loud,
                # because silently dropping it would leave a bucket of the
                # design matrix quietly empty.
                raise SourceUnavailable(
                    f"{self.source}: lead {lead} came back entirely null for "
                    f"{point['name']}. The previous-run archive does not cover "
                    "this window; shorten the range or drop the lead."
                )
            daily = (pd.DataFrame({"target_date": frame["target_date"],
                                   "value": values})
                     .groupby("target_date", as_index=False)
                     .agg(mean=("value", "mean"),
                          max=("value", "max"),
                          min=("value", "min")))
            for stat in ("mean", "max", "min"):
                piece = daily[["target_date", stat]].rename(columns={stat: "value"})
                piece["variable"] = f"{base_variable}_{stat}"
                piece["issue_date"] = piece["target_date"] - timedelta(days=lead)
                piece["region"] = region
                piece["point"] = point["name"]
                piece["model"] = "archive_best_match"
                piece["member"] = DETERMINISTIC_MEMBER
                out.append(piece)
        return pd.concat(out, ignore_index=True)


class ForecastFetcher(OpenMeteoFetcher):
    """Today's live deterministic run. The production signal, not backtestable."""

    source = "open_meteo_forecast"
    supports_historical_issue = False

    def fetch(self, region: str = "", model: str = "ecmwf_ifs025",
              forecast_days: int | None = None,
              variables: list[str] | None = None) -> FetchResult:
        if not region:
            raise ValueError("region is required")
        kind = get("regions", "regions", region, "kind")
        variables = variables or get("sources", "open_meteo",
                                     "daily_variables", kind)
        max_lead = get("sources", "open_meteo", "models", model, "max_lead_days")
        days = min(forecast_days or max_lead, max_lead)

        points = region_points(region)
        url = get("sources", "open_meteo", "endpoints", "forecast")
        issue = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

        rows: list[pd.DataFrame] = []
        for batch in self._paced_batches(points):
            params = self._base_params(batch) | {
                "daily": ",".join(variables),
                "forecast_days": days,
                "models": model,
            }
            payloads = _as_list(_http_get_json(url, params))
            for point, payload in zip(batch, payloads, strict=True):
                block = self._daily_block(payload, self.source)
                melted = (pd.DataFrame(block)
                          .melt(id_vars="time", var_name="variable",
                                value_name="value"))
                melted["target_date"] = pd.to_datetime(melted["time"])
                melted["issue_date"] = issue
                melted["region"] = region
                melted["point"] = point["name"]
                melted["model"] = model
                melted["member"] = DETERMINISTIC_MEMBER
                rows.append(melted.drop(columns=["time"]))
            self._pace()

        panel = self._validate_panel(pd.concat(rows, ignore_index=True), self.source)
        return self._result(panel.dropna(subset=["value"]), self.source)


class EnsembleFetcher(OpenMeteoFetcher):
    """Per-member forecasts. Spread is the volatility signal, not a direction.

    ``gfs05`` is the default because it is the ONLY member set that actually
    reaches a 30-day lead, which is the horizon this project was asked for.
    ECMWF-ENS is the better model and has more members, but stops around day
    14; inside two weeks it is the right choice and config says so.

    THE NULL PADDING. Every ensemble model here returns whatever time axis is
    asked for and pads beyond its real horizon with nulls rather than
    erroring. Asking gfs025 for 35 days yields a 35-day axis carrying 10 days
    of data. Dropping the nulls silently - the obvious implementation - returns
    a panel that looks complete and is a third the length it claims. So usable
    depth is measured and checked against config before anything is returned.
    """

    source = "open_meteo_ensemble"
    supports_historical_issue = False

    def _check_depth(self, frame: pd.DataFrame, requested: int, model: str,
                     point: str) -> None:
        """Refuse a response padded well past the model's real horizon."""
        usable = int(frame["target_date"].nunique())
        floor = get("sources", "open_meteo", "ensemble_min_usable_fraction")
        if usable < requested * floor:
            raise SourceUnavailable(
                f"{model} returned {usable} usable days of the {requested} "
                f"requested at {point}. Beyond its real horizon the API pads "
                "the axis with nulls instead of erroring, so this is a "
                "truncated panel, not a short forecast. Check "
                "ensemble_models.max_lead_days in sources.yaml - gfs05 is the "
                "only member set that reaches 30 days."
            )

    def fetch(self, region: str = "", model: str = "gfs05",
              forecast_days: int | None = None,
              variable: str = "temperature_2m_mean") -> FetchResult:
        if not region:
            raise ValueError("region is required")
        max_lead = get("sources", "open_meteo", "ensemble_models", model,
                       "max_lead_days")
        if forecast_days is not None and forecast_days > max_lead:
            # Clamping silently would hand back 10 days to a caller who asked
            # for 30 and thought they got it. The horizon is the whole question
            # here, so a model that cannot reach it is an error, not a detail.
            raise SourceUnavailable(
                f"{model} reaches {max_lead} usable days and {forecast_days} "
                f"were requested. It will return a {forecast_days}-day axis "
                "padded with nulls rather than refusing, which is why this is "
                "checked here. Use gfs05 for horizons beyond two weeks - it is "
                "the only member set that reaches 30 days."
            )
        days = min(forecast_days or max_lead, max_lead)

        points = region_points(region)
        url = get("sources", "open_meteo", "endpoints", "ensemble")
        issue = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

        rows: list[pd.DataFrame] = []
        for batch in self._paced_batches(points):
            params = self._base_params(batch) | {
                "daily": variable,
                "forecast_days": days,
                "models": model,
            }
            payloads = _as_list(_http_get_json(url, params))
            for point, payload in zip(batch, payloads, strict=True):
                block = self._daily_block(payload, self.source)
                frame = pd.DataFrame(block)
                target = pd.to_datetime(frame["time"])
                point_rows: list[pd.DataFrame] = []
                for column in frame.columns:
                    if column == "time":
                        continue
                    member = (DETERMINISTIC_MEMBER if column == variable
                              else int(column.rsplit("member", 1)[-1]))
                    point_rows.append(pd.DataFrame({
                        "issue_date": issue,
                        "target_date": target,
                        "region": region,
                        "point": point["name"],
                        "variable": variable,
                        "value": pd.to_numeric(frame[column], errors="coerce"),
                        "model": model,
                        "member": member,
                    }))
                usable = pd.concat(point_rows, ignore_index=True).dropna(
                    subset=["value"])
                self._check_depth(usable, days, model, point["name"])
                rows.append(usable)
            self._pace()

        panel = self._validate_panel(pd.concat(rows, ignore_index=True), self.source)
        return self._result(panel, self.source)


def ensemble_spread(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-member standard deviation per (issue, target, region, point).

    The control/deterministic row is excluded: including it would pull the
    spread towards the mean of a set it is not a member of.
    """
    members = panel[panel["member"] != DETERMINISTIC_MEMBER]
    if members.empty:
        raise SourceUnavailable(
            "no ensemble members in the panel, so no spread can be computed")
    keys = ["issue_date", "target_date", "region", "point", "variable"]
    spread = (members.groupby(keys, as_index=False)["value"]
              .agg(spread=lambda s: float(np.std(s, ddof=1)),
                   member_count="count"))
    return spread
