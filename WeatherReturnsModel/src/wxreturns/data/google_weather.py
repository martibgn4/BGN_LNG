"""Google Maps Platform Weather API - live forecasts only.

Wired up because the project brief named it, and it is a reasonable
cross-check on today's run: two independent providers disagreeing about
tomorrow is worth seeing before trading on either.

IT CANNOT BE THE PRIMARY SOURCE. It serves the current forecast and keeps no
archive of what past forecasts said, so the revision series that the model
actually runs on cannot be reconstructed from it for any date before collection
began. That is a property of the API, not a configuration problem.

The one thing this module must never do is answer a request for a past issue
date with the current run. That would be lookahead of the most damaging kind -
invisible in the data, and it would make a backtest look superb. `fetch`
raises `HistoricalIssueUnsupported` instead, and there is deliberately no
option to override it.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Any

import pandas as pd

from wxreturns.config import get
from wxreturns.data.base import (
    FetchResult,
    HistoricalIssueUnsupported,
    SourceUnavailable,
    WeatherFetcher,
)
from wxreturns.data.openmeteo import DETERMINISTIC_MEMBER, region_points

log = logging.getLogger(__name__)

# Google returns Celsius when asked; the panel is Celsius throughout so that
# degree-day conversion happens in exactly one place (weather/degree_days.py).
_UNITS = "METRIC"


class GoogleWeatherFetcher(WeatherFetcher):
    source = "google_weather"
    credential_env_var = "GOOGLE_WEATHER_API_KEY"
    credential_required = True
    supports_historical_issue = False

    def fetch(self, region: str = "", forecast_days: int | None = None,
              issue_date: date | None = None) -> FetchResult:
        if not region:
            raise ValueError("region is required")

        if issue_date is not None:
            today = pd.Timestamp.now(tz="UTC").date()
            if issue_date < today:
                raise HistoricalIssueUnsupported(
                    f"{self.source} has no forecast archive, so it cannot say "
                    f"what was forecast on {issue_date}. Returning the current "
                    "run under a past issue date would be undetectable "
                    "lookahead. Use OpenMeteo HistoricalForecastFetcher for "
                    "history; this provider is live-only by construction."
                )

        key = self.credential()
        cap = get("sources", "google_weather", "max_forecast_days")
        days = min(forecast_days or cap, cap)
        url = get("sources", "google_weather", "endpoints", "daily_forecast")
        timeout = get("sources", "google_weather", "request_timeout_seconds")
        issue = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

        rows: list[pd.DataFrame] = []
        for point in region_points(region):
            params = {
                "key": key,
                "location.latitude": point["lat"],
                "location.longitude": point["lon"],
                "days": days,
                "unitsSystem": _UNITS,
            }
            payload = self._get(url, params, timeout)
            rows.append(self._reshape(payload, region, point, issue))

        panel = self._validate_panel(pd.concat(rows, ignore_index=True), self.source)
        return self._result(panel.dropna(subset=["value"]), self.source)

    def _get(self, url: str, params: dict[str, Any], timeout: int) -> dict:
        full = f"{url}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(full, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:400]
            raise SourceUnavailable(
                f"{self.source} rejected the request ({exc.code}): {body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SourceUnavailable(f"{self.source} unreachable: {exc}") from exc

    def _reshape(self, payload: dict, region: str, point: dict,
                 issue: pd.Timestamp) -> pd.DataFrame:
        """Google's nested daily forecast into the canonical flat panel."""
        days = payload.get("forecastDays")
        if not days:
            raise SourceUnavailable(
                f"{self.source} returned no forecastDays for {point['name']}")

        records: list[dict] = []
        for day in days:
            stamp = day.get("displayDate") or {}
            try:
                target = pd.Timestamp(year=stamp["year"], month=stamp["month"],
                                      day=stamp["day"])
            except (KeyError, TypeError) as exc:
                raise SourceUnavailable(
                    f"{self.source} returned a day without a usable date"
                ) from exc
            readings = {
                "temperature_2m_max": (day.get("maxTemperature") or {}).get("degrees"),
                "temperature_2m_min": (day.get("minTemperature") or {}).get("degrees"),
            }
            high, low = readings["temperature_2m_max"], readings["temperature_2m_min"]
            if high is not None and low is not None:
                # Google publishes no daily mean. Midpoint of max and min is
                # the classic degree-day convention anyway - NOAA HDDs are
                # built from exactly this - so it is the right reconstruction
                # rather than a workaround, though it differs slightly from
                # Open-Meteo's true hourly mean. Do not mix the two providers
                # inside one feature.
                readings["temperature_2m_mean"] = (high + low) / (1.0 + 1.0)
            for variable, value in readings.items():
                records.append({
                    "issue_date": issue,
                    "target_date": target,
                    "region": region,
                    "point": point["name"],
                    "variable": variable,
                    "value": value,
                    "model": "google_weather",
                    "member": DETERMINISTIC_MEMBER,
                })
        return pd.DataFrame.from_records(records)
