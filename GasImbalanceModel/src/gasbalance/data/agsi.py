"""GIE AGSI+ storage. SPEC 6.

Requires a free GIE API key, passed as the `x-key` header. Verified on the
wire: without it the endpoint answers HTTP 200 with a JSON error body
`{"error":"access denied","message":"Invalid or missing API key"}`, so a
naive client would treat a refusal as success. This module checks the payload,
not the status code.

Set the key in the environment:  GIE_API_KEY
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from gasbalance.data.base import (
    CredentialsMissing,
    Fetcher,
    FetchResult,
    SourceUnavailable,
)
from gasbalance.units import Units

log = logging.getLogger(__name__)

API_ROOT = "https://agsi.gie.eu/api"
PAGE_SIZE = 300

# AGSI field -> internal series_id. Every field listed here must also be
# declared in conversions.yaml, or the conversion layer raises.
FIELD_MAP = {
    "gasInStorage": "gas_in_storage",
    "injection": "injection",
    "withdrawal": "withdrawal",
    "workingGasVolume": "working_gas_volume",
    "injectionCapacity": "injection_capacity",
    "withdrawalCapacity": "withdrawal_capacity",
}

STOCK_SERIES = {"gas_in_storage", "working_gas_volume"}


class AgsiFetcher(Fetcher):
    source = "agsi"
    requires_credentials = True
    credential_env_var = "GIE_API_KEY"

    def __init__(self, units: Units | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units or Units()

    def _get_page(self, country: str, start: date, end: date, page: int) -> dict[str, Any]:
        headers = {"x-key": self.credential()}
        params = {
            "country": country,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "size": PAGE_SIZE,
            "page": page,
        }
        try:
            resp = requests.get(API_ROOT, params=params, headers=headers, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise SourceUnavailable(f"AGSI request failed: {exc}") from exc
        except ValueError as exc:
            raise SourceUnavailable(f"AGSI returned non-JSON: {exc}") from exc

        # AGSI answers 200 even when it is refusing. Check the body.
        if payload.get("error"):
            raise CredentialsMissing(
                f"AGSI refused the request: {payload.get('message', payload['error'])}. "
                "Set a valid key in $GIE_API_KEY."
            )
        if "data" not in payload:
            raise SourceUnavailable(f"AGSI response had no 'data' key: {list(payload)}")
        return payload

    def fetch(self, start: date, end: date, **kwargs: object) -> FetchResult:
        countries = kwargs.get("countries")
        if not countries:
            raise ValueError("AGSI fetch requires an explicit `countries` list")

        vintage = self.now()
        raw_frames: list[pd.DataFrame] = []
        for country in countries:  # type: ignore[union-attr]
            page = 1
            while True:
                payload = self._get_page(country, start, end, page)
                rows = payload["data"]
                if rows:
                    chunk = pd.DataFrame(rows)
                    chunk["country"] = country
                    raw_frames.append(chunk)
                last_page = int(payload.get("last_page") or 0)
                if page >= last_page or not rows:
                    break
                page += 1

        if not raw_frames:
            raise SourceUnavailable(f"AGSI returned no rows for {start}..{end}")

        raw = pd.concat(raw_frames, ignore_index=True)
        self.cache_raw(raw.astype(str), f"storage_{start}_{end}", vintage)
        return self._normalise(raw, vintage)

    def _normalise(self, raw: pd.DataFrame, vintage: datetime) -> FetchResult:
        rows_raw = len(raw)
        drop_reasons: dict[str, int] = {}
        records: list[pd.DataFrame] = []

        obs_date = pd.to_datetime(raw["gasDayStart"], errors="coerce").dt.date
        bad_date = pd.isna(obs_date)
        if bad_date.any():
            drop_reasons["unparseable gasDayStart"] = int(bad_date.sum())

        for api_field, series_id in FIELD_MAP.items():
            if api_field not in raw.columns:
                log.warning("AGSI did not return field %s; skipping", api_field)
                continue
            value = pd.to_numeric(raw[api_field], errors="coerce")
            keep = value.notna() & ~bad_date
            missing = int((~keep).sum())
            if missing:
                drop_reasons[f"{series_id}: missing or non-numeric"] = missing

            spec = self.units.series_spec(self.source, series_id)
            unit_head = spec.raw_unit.split("/")[0]
            if series_id in STOCK_SERIES:
                scale = self.units.energy_to_gwh(1.0, unit_head)
                converted = value[keep] * scale / self.units.energy_to_gwh(1.0, "TWh")
                out_unit = "TWh"
            else:
                scale = self.units.energy_to_gwh(1.0, unit_head) / self.units.canonical_gcv
                converted = value[keep] * scale
                out_unit = "mcm/d"

            records.append(
                pd.DataFrame(
                    {
                        "obs_date": obs_date[keep],
                        "country": raw.loc[keep, "country"],
                        "point_key": None,
                        "operator_key": None,
                        "direction_key": None,
                        "series_id": series_id,
                        "value": converted,
                        "unit": out_unit,
                        "source": self.source,
                        "origin": None,
                        "vintage": vintage,
                        "source_updated_at": pd.NaT,
                        "is_estimated": False,
                        "run_id": "",
                    }
                )
            )

        out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        if drop_reasons:
            log.warning("AGSI: dropped values, not filled: %s", drop_reasons)
        return FetchResult(
            frame=out,
            source=self.source,
            vintage=vintage,
            rows_raw=rows_raw,
            rows_dropped=sum(drop_reasons.values()),
            drop_reasons=drop_reasons,
        )
