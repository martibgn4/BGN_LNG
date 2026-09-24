"""Kpler cargo-level LNG. SPEC 6 - paid source, no free substitute.

The trader has a Kpler subscription, so this is a real interface rather than a
permanent stub. It is not used in Phase 4: SPEC 3 makes call-on-LNG the
residual of the balance, and Kpler is the independent series that residual is
validated against. Wiring Kpler into the residual itself would destroy the
check.

Two implementations:

  `KplerApiFetcher`  the live API. Needs KPLER_API_KEY. Normalisation is left
                     unimplemented until a real payload has been seen, because
                     column layouts differ by product and guessing them would
                     put unverified numbers into the balance.

  `KplerCsvFetcher`  loads a Kpler CSV export from disk. This is the clearly
                     marked loader SPEC 6 asks for. It reads a real export the
                     trader produced; it does not generate anything.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from gasbalance.data.base import (
    CredentialsMissing,
    Fetcher,
    FetchResult,
    SourceUnavailable,
)
from gasbalance.units import Units

log = logging.getLogger(__name__)

REQUIRED_CSV_COLUMNS = {"date", "country", "volume", "unit"}


class KplerApiFetcher(Fetcher):
    source = "kpler"
    requires_credentials = True
    credential_env_var = "KPLER_API_KEY"

    def fetch(self, start: date, end: date, **kwargs: object) -> FetchResult:
        self.credential()  # raises CredentialsMissing if absent
        raise NotImplementedError(
            "Kpler API normalisation is unimplemented until a real payload has "
            "been captured. Export a CSV from Kpler and use KplerCsvFetcher, or "
            "share one response body and this will be wired to it. No synthetic "
            "cargo data is produced here (SPEC 6, SPEC 11)."
        )


class KplerCsvFetcher(Fetcher):
    """Loads a Kpler CSV export. Clearly marked, reads real data only."""

    source = "kpler"
    requires_credentials = False

    def __init__(self, csv_path: Path | str | None = None, units: Units | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.csv_path = Path(csv_path) if csv_path else None
        self.units = units or Units()

    def is_available(self) -> bool:
        return self.csv_path is not None and self.csv_path.exists()

    def fetch(self, start: date, end: date, **kwargs: object) -> FetchResult:
        if self.csv_path is None:
            raise CredentialsMissing(
                "KplerCsvFetcher needs a path to a real Kpler export. It will not "
                "invent cargo data."
            )
        if not self.csv_path.exists():
            raise SourceUnavailable(f"Kpler export not found: {self.csv_path}")

        vintage = self.now()
        raw = pd.read_csv(self.csv_path)
        missing = REQUIRED_CSV_COLUMNS - set(raw.columns)
        if missing:
            raise SourceUnavailable(
                f"Kpler export {self.csv_path.name} is missing columns {sorted(missing)}; "
                f"found {sorted(raw.columns)}. Map them explicitly rather than guessing."
            )
        self.cache_raw(raw.astype(str), f"cargoes_{start}_{end}", vintage)

        obs_date = pd.to_datetime(raw["date"], errors="coerce").dt.date
        value = pd.to_numeric(raw["volume"], errors="coerce")
        keep = value.notna() & pd.notna(obs_date)
        dropped = int((~keep).sum())

        units_seen = set(raw.loc[keep, "unit"].unique())
        unknown = {u for u in units_seen if u not in ("mcm", "bcm", "m3", "km3")}
        if unknown:
            raise SourceUnavailable(
                f"Kpler export uses volume units {sorted(unknown)} that are not in "
                "conversions.yaml. Declare them before ingesting."
            )
        converted = [
            self.units.volume_to_mcm(v, u)
            for v, u in zip(value[keep], raw.loc[keep, "unit"], strict=True)
        ]

        out = pd.DataFrame(
            {
                "obs_date": obs_date[keep],
                "country": raw.loc[keep, "country"],
                "point_key": None,
                "operator_key": None,
                "direction_key": None,
                "series_id": "lng_cargo_arrival",
                "value": converted,
                "unit": "mcm/d",
                "source": self.source,
                "origin": raw["origin"] if "origin" in raw.columns else None,
                "vintage": vintage,
                "source_updated_at": pd.NaT,
                "is_estimated": False,
                "run_id": "",
            }
        )
        return FetchResult(
            frame=out,
            source=self.source,
            vintage=vintage,
            rows_raw=len(raw),
            rows_dropped=dropped,
            drop_reasons={"missing date or volume": dropped} if dropped else {},
        )
