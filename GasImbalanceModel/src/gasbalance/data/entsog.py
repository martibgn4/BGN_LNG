"""ENTSOG Transparency Platform. SPEC 6.

Public and keyless - verified 200 OK with no token. Two jobs:

  `fetch_point_directions`  the point registry, used to resolve the perimeter
  `fetch`                   daily physical flows at those points

ENTSOG publishes flows in kWh/d (confirmed on the wire, `unit` field), so every
value goes through the conversion layer before it leaves this module.

Rows with `isNA` set or a blank value are NOT filled, interpolated or zeroed.
They are dropped and counted, and the count is reported. SPEC 3 and SPEC 11.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from gasbalance.data.base import Fetcher, FetchResult, SourceUnavailable
from gasbalance.units import Units

log = logging.getLogger(__name__)

API_ROOT = "https://transparency.entsog.eu/api/v1"
PHYSICAL_FLOW_INDICATOR = "Physical Flow"
PAGE_LIMIT = 10000


class EntsogFetcher(Fetcher):
    source = "entsog"
    requires_credentials = False

    def __init__(self, units: Units | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units or Units()

    # -- raw HTTP -----------------------------------------------------------

    def _get(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{API_ROOT}/{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=180)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise SourceUnavailable(f"ENTSOG {endpoint} request failed: {exc}") from exc
        except ValueError as exc:
            raise SourceUnavailable(f"ENTSOG {endpoint} returned non-JSON: {exc}") from exc

        keys = [k for k in payload if k != "meta"]
        if not keys:
            raise SourceUnavailable(
                f"ENTSOG {endpoint} response had no data key; the API shape may have "
                f"changed. Keys seen: {list(payload)}"
            )
        return payload[keys[0]]

    def _get_paged(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self._get(endpoint, {**params, "limit": PAGE_LIMIT, "offset": offset})
            rows.extend(page)
            if len(page) < PAGE_LIMIT:
                return rows
            offset += PAGE_LIMIT

    # -- point registry -----------------------------------------------------

    def fetch_point_directions(self) -> pd.DataFrame:
        """The full operator/point/direction registry, one row per direction."""
        rows = self._get_paged("operatorpointdirections", {})
        if not rows:
            raise SourceUnavailable("ENTSOG returned an empty point registry")
        frame = pd.DataFrame(rows)
        self.cache_raw(frame.astype(str), "operatorpointdirections", self.now())
        log.info("fetched %d ENTSOG point-directions", len(frame))
        return frame

    # -- physical flows -----------------------------------------------------

    def fetch(self, start: date, end: date, **kwargs: object) -> FetchResult:
        """Daily physical flows, converted to mcm/d.

        `point_keys` restricts the pull to the perimeter; without it ENTSOG
        returns every point in Europe for the window, which is very large.
        """
        point_keys = kwargs.get("point_keys")
        vintage = self.now()

        params: dict[str, Any] = {
            "indicator": PHYSICAL_FLOW_INDICATOR,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "periodType": "day",
        }
        if point_keys:
            params["pointKey"] = ",".join(sorted(set(point_keys)))  # type: ignore[arg-type]

        rows = self._get_paged("operationaldata", params)
        raw = pd.DataFrame(rows)
        if raw.empty:
            raise SourceUnavailable(
                f"ENTSOG returned no physical-flow rows for {start}..{end}"
            )
        self.cache_raw(raw.astype(str), f"flows_{start}_{end}", vintage)

        return self._normalise(raw, vintage)

    def _normalise(self, raw: pd.DataFrame, vintage: datetime) -> FetchResult:
        rows_raw = len(raw)
        drop_reasons: dict[str, int] = {}

        frame = raw.copy()
        frame["value_num"] = pd.to_numeric(frame["value"], errors="coerce")

        na_flag = frame.get("isNA")
        if na_flag is not None:
            is_na = pd.to_numeric(na_flag, errors="coerce").fillna(0).astype(int) == 1
            drop_reasons["isNA flagged by ENTSOG"] = int(is_na.sum())
            frame = frame[~is_na]

        blank = frame["value_num"].isna()
        drop_reasons["blank or non-numeric value"] = int(blank.sum())
        frame = frame[~blank]

        # SPEC 5: never assume the unit. ENTSOG says kWh/d; anything else is a
        # change we must see rather than silently mis-scale.
        declared = self.units.series_spec(self.source, "physical_flow").raw_unit
        wrong_unit = frame["unit"] != declared
        if wrong_unit.any():
            seen = sorted(frame.loc[wrong_unit, "unit"].unique())
            raise SourceUnavailable(
                f"ENTSOG physical flows arrived in unexpected units {seen}; "
                f"conversions.yaml declares {declared!r}. Update the config "
                "deliberately rather than letting this convert wrongly."
            )

        gcv = self.units.canonical_gcv
        scale = self.units.energy_to_gwh(1.0, declared.split("/")[0]) / gcv

        out = pd.DataFrame(
            {
                "obs_date": pd.to_datetime(
                    frame["periodFrom"], utc=True, format="mixed"
                ).dt.date,
                "country": frame["operatorKey"].str.slice(0, 2),
                "point_key": frame["pointKey"],
                "series_id": "physical_flow_" + frame["directionKey"],
                "value": frame["value_num"] * scale,
                "unit": "mcm/d",
                "source": self.source,
                "origin": None,
                "vintage": vintage,
                "source_updated_at": pd.to_datetime(
                    frame["lastUpdateDateTime"], utc=True, format="mixed", errors="coerce"
                ).dt.tz_localize(None),
                "is_estimated": False,
                "run_id": "",
                "operator_key": frame["operatorKey"],
                "direction_key": frame["directionKey"],
            }
        )

        dropped = rows_raw - len(out)
        if dropped:
            log.warning(
                "ENTSOG: dropped %d of %d rows (not filled, not interpolated): %s",
                dropped,
                rows_raw,
                drop_reasons,
            )
        return FetchResult(
            frame=out,
            source=self.source,
            vintage=vintage,
            rows_raw=rows_raw,
            rows_dropped=dropped,
            drop_reasons=drop_reasons,
        )
