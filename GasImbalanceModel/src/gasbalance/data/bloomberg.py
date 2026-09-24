"""Bloomberg via xbbg. SPEC 6.

This is the storage source for the model. The trader has a Terminal licence, so
GIE AGSI+ is taken from Bloomberg's `CGIE*` redistribution rather than from the
GIE API, and no GIE key is needed.

Identifiers live in config/bloomberg_series.yaml and were discovered by querying
Bloomberg's own //blp/instruments search, then verified against real data. None
was guessed. Units come from the Bloomberg security description, which states
them explicitly ("Germany Gas Inventory (Daily,TWh,GIE)"), and are cross-checked
at fetch time against conversions.yaml - a disagreement raises rather than
converting on an assumption.

Environment: `xbbg` plus `blpapi` (from Bloomberg's own index, not PyPI) and a
logged-in Terminal on this machine. Pin xbbg to the 0.12.x line: 1.x returns
polars frames and silently breaks every `.empty` / pandas idiom in this repo.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

from gasbalance.config import ConfigError, load
from gasbalance.data.base import CredentialsMissing, Fetcher, FetchResult, SourceUnavailable
from gasbalance.units import Units

log = logging.getLogger(__name__)

SERIES_CONFIG = "bloomberg_series.yaml"
PRICE_FIELD = "px_last"  # xbbg lowercases field names on the way back
STOCK_SERIES = {"gas_in_storage", "working_gas_volume"}


class BloombergUnavailable(CredentialsMissing):
    """xbbg/blpapi not importable, or no Terminal session on this machine."""


class BloombergFetcher(Fetcher):
    source = "bloomberg"
    requires_credentials = True
    credential_env_var = None  # a live Terminal session, not an env var

    def __init__(self, units: Units | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units or Units()
        self._blp: Any | None = None

    # -- availability -------------------------------------------------------

    def _import_blp(self) -> Any:
        if self._blp is not None:
            return self._blp
        try:
            from xbbg import blp  # noqa: PLC0415
        except ImportError as exc:
            raise BloombergUnavailable(
                "xbbg/blpapi not importable. Install blpapi from Bloomberg's index "
                "(uv pip install --index-url="
                "https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi) "
                "plus 'xbbg==0.12.2', and ensure a logged-in Terminal is running."
            ) from exc
        self._blp = blp
        return blp

    def is_available(self) -> bool:
        try:
            self._import_blp()
        except BloombergUnavailable:
            return False
        return True

    def credential(self) -> str:
        self._import_blp()
        return "terminal-session"

    # -- config -------------------------------------------------------------

    def config(self) -> dict[str, Any]:
        try:
            return load(SERIES_CONFIG)
        except ConfigError as exc:
            raise BloombergUnavailable(
                f"{SERIES_CONFIG} is missing; Bloomberg identifiers must be declared "
                "there, never inferred at runtime."
            ) from exc

    def storage_tickers(self, countries: list[str]) -> dict[str, tuple[str, str]]:
        """Map ticker -> (perimeter country, series_id) for the requested ring."""
        cfg = self.config()
        code_map: dict[str, str] = cfg["country_code_map"]
        no_storage: dict[str, str] = cfg.get("countries_without_storage") or {}

        out: dict[str, tuple[str, str]] = {}
        for country in countries:
            if country in no_storage:
                log.info(
                    "%s has no storage (%s); expected-zero, not a data gap",
                    country,
                    no_storage[country],
                )
                continue
            if country not in code_map:
                raise SourceUnavailable(
                    f"country {country!r} has no Bloomberg code in {SERIES_CONFIG} and "
                    f"is not listed as storage-free. Declare it explicitly."
                )
            for suffix, spec in cfg["storage_series"].items():
                ticker = f"CGIE{code_map[country]}{suffix} Index"
                out[ticker] = (country, spec["series_id"])
        return out

    # -- unit verification --------------------------------------------------

    def verify_units(self, tickers: list[str]) -> dict[str, str]:
        """Check each ticker's Bloomberg description against conversions.yaml.

        Bloomberg states the unit inside the security description. Reading it
        back turns the declared unit into a checked fact instead of a belief.
        """
        blp = self._import_blp()
        cfg = self.config()
        expected_by_suffix = {
            suffix: spec["expect_unit"] for suffix, spec in cfg["storage_series"].items()
        }
        try:
            desc = blp.bdp(tickers=tickers, flds=["NAME"])
        except Exception as exc:  # noqa: BLE001 - xbbg raises a wide variety
            raise SourceUnavailable(f"Bloomberg description lookup failed: {exc}") from exc

        mismatches: dict[str, str] = {}
        for ticker in tickers:
            suffix = ticker.split()[0][-2:]
            want = expected_by_suffix.get(suffix)
            if want is None or ticker not in desc.index:
                continue
            name = str(desc.loc[ticker].iloc[0])
            if want.replace("/", "/") not in name:
                mismatches[ticker] = f"expected {want!r} in description, got {name!r}"
        return mismatches

    # -- fetch --------------------------------------------------------------

    def fetch(self, start: date, end: date, **kwargs: object) -> FetchResult:
        countries = kwargs.get("countries")
        if not countries:
            raise ValueError("Bloomberg fetch requires an explicit `countries` list")

        blp = self._import_blp()
        vintage = self.now()
        mapping = self.storage_tickers(list(countries))  # type: ignore[arg-type]
        tickers = sorted(mapping)

        try:
            hist = blp.bdh(
                tickers=tickers,
                flds=[PRICE_FIELD],
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            )
        except Exception as exc:  # noqa: BLE001
            raise SourceUnavailable(f"Bloomberg bdh failed: {exc}") from exc

        if hist is None or hist.empty:
            raise SourceUnavailable(
                f"Bloomberg returned no storage data for {start}..{end}. "
                "This is a refusal, not an empty balance - nothing is substituted."
            )
        self.cache_raw(hist.reset_index().astype(str), f"storage_{start}_{end}", vintage)
        return self._normalise(hist, mapping, vintage)

    def _normalise(
        self,
        hist: pd.DataFrame,
        mapping: dict[str, tuple[str, str]],
        vintage: datetime,
    ) -> FetchResult:
        returned = {col[0] for col in hist.columns}
        absent = sorted(set(mapping) - returned)
        drop_reasons: dict[str, int] = {}
        if absent:
            drop_reasons["ticker returned nothing"] = len(absent)
            log.warning("Bloomberg returned no series for: %s", absent)

        records: list[pd.DataFrame] = []
        rows_raw = 0
        for ticker, (country, series_id) in mapping.items():
            if ticker not in returned:
                continue
            series = hist[(ticker, PRICE_FIELD)].dropna()
            rows_raw += len(hist[(ticker, PRICE_FIELD)])
            missing = len(hist) - len(series)
            if missing:
                drop_reasons[f"{country}.{series_id}: blank days"] = (
                    drop_reasons.get(f"{country}.{series_id}: blank days", 0) + missing
                )

            spec = self.units.series_spec(self.source, series_id)
            unit_head = spec.raw_unit.split("/")[0]
            if series_id in STOCK_SERIES:
                gwh = series * self.units.energy_to_gwh(1.0, unit_head)
                values = gwh / self.units.energy_to_gwh(1.0, "TWh")
                out_unit = "TWh"
            else:
                gwh = series * self.units.energy_to_gwh(1.0, unit_head)
                values = gwh / self.units.canonical_gcv
                out_unit = "mcm/d"

            records.append(
                pd.DataFrame(
                    {
                        "obs_date": pd.to_datetime(series.index).date,
                        "country": country,
                        "point_key": None,
                        "operator_key": None,
                        "direction_key": None,
                        "series_id": series_id,
                        "value": values.to_numpy(),
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
            log.warning("Bloomberg: dropped values, not filled: %s", drop_reasons)
        return FetchResult(
            frame=out,
            source=self.source,
            vintage=vintage,
            rows_raw=rows_raw,
            rows_dropped=sum(drop_reasons.values()),
            drop_reasons=drop_reasons,
        )

    # -- independent aggregate ---------------------------------------------

    def fetch_europe_aggregate(self, start: date, end: date) -> pd.DataFrame:
        """GIE's own Europe total, as a cross-check on the sum of countries.

        Deliberately not part of the balance. SPEC 8 wants the reconciliation
        gap visible; comparing our country sum to a published total is how the
        gap gets measured rather than absorbed.
        """
        blp = self._import_blp()
        cfg = self.config()
        tickers = sorted(cfg["aggregate"].values())
        hist = blp.bdh(
            tickers=tickers,
            flds=[PRICE_FIELD],
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        if hist is None or hist.empty:
            raise SourceUnavailable("Bloomberg returned no GIE Europe aggregate")
        frame = pd.DataFrame(
            {name: hist[(ticker, PRICE_FIELD)] for name, ticker in cfg["aggregate"].items()}
        )
        frame.index = pd.to_datetime(frame.index).date
        return frame
