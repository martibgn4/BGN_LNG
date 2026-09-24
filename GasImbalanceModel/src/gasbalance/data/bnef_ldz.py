"""BNEF Europe LDZ Gas Demand Monitor. SPEC 2a dependent variable.

Loads the Excel export the trader pulls from the Bloomberg Terminal. LDZ here
means residential + commercial gas demand, which is exactly the SPEC 2a bucket,
published daily in mcm/d - already the internal flow unit.

VALIDATION. This loader is trusted because the file was reconciled against raw
TSO data pulled independently from Bloomberg: for France, file vs
(NaTran FRGCSDPD + Terega TFGDPITD) over Oct-24..Mar-25 gives mean difference
+0.11 mcm/d and MAE 0.66 mcm/d, 0.7% of level, correlation 0.9971. Teréga alone
accounts for the 6.8 mcm/d that NaTran on its own does not cover.

THREE COLUMNS, THREE DIFFERENT THINGS. Only one of them is data:

  actual     realised demand. The dependent variable. is_estimated = False.
  expected   BNEF's OWN MODEL output, a regression trained to 2022-01-01.
             is_estimated = True. Never fit against this - fitting a model to
             another model teaches it to reproduce BNEF, not reality. It is
             loaded solely as a competitor benchmark.
  benchmark  2016-2020 average demand. A seasonal normal, not an observation.
             is_estimated = True.

GERMANY HAS TWO SERIES and the difference matters:

  "Germany implied flow"        realised, from Trading Hub Europe, published
                                with a lag (file ends 2026-06-01). USE THIS for
                                fitting and backtesting.
  "Germany flow nominations"    grid-operator forecast nominations, timely
                                (file ends 2026-07-26). BNEF's own note says it
                                is "not reflective of actual demand". Measured
                                against implied: +1.38 mcm/d bias, 5.0% MAE,
                                worst day 28.6 mcm/d. Load it, label it, and use
                                it only as the timely proxy in live running.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gasbalance.config import load
from gasbalance.data.base import Fetcher, FetchResult, SourceUnavailable
from gasbalance.units import Units

log = logging.getLogger(__name__)

SHEET_DAILY = "Europe LDZ gas demand (daily)"
HEADER_SKIPROWS = 8
COLUMNS = [
    "region",
    "date",
    "actual",
    "benchmark",
    "expected",
    "vs_benchmark_pct",
    "vs_expected_pct",
]

#: value column -> (series_id, is_estimated). See the module docstring.
VALUE_COLUMNS = {
    "actual": ("ldz_demand", False),
    "expected": ("ldz_demand_bnef_expected", True),
    "benchmark": ("ldz_demand_normal_2016_2020", True),
}


class BnefLdzFetcher(Fetcher):
    """Reads a real BNEF export from disk. Generates nothing."""

    source = "bnef_ldz"
    requires_credentials = False

    def __init__(
        self,
        workbook: Path | str | None = None,
        units: Units | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units or Units()
        cfg = self.mapping_config()
        self.workbook = Path(workbook) if workbook else Path(cfg["workbook_path"])

    # -- config -------------------------------------------------------------

    @staticmethod
    def mapping_config() -> dict[str, Any]:
        return load("demand_sources.yaml")["bnef_ldz"]

    def region_map(self) -> dict[str, str]:
        return self.mapping_config()["region_to_country"]

    def is_available(self) -> bool:
        return self.workbook.exists()

    # -- provenance ---------------------------------------------------------

    def dataset_date(self) -> datetime | None:
        """The 'Date' on the COVER sheet: when BNEF cut the dataset."""
        try:
            cover = pd.read_excel(self.workbook, sheet_name="COVER", header=None)
        except Exception:  # noqa: BLE001 - cover sheet is optional metadata
            return None
        for _, row in cover.iterrows():
            values = [v for v in row.tolist() if isinstance(v, str)]
            if any(v.strip() == "Date" for v in values):
                for v in values:
                    parsed = pd.to_datetime(v, errors="coerce")
                    if pd.notna(parsed):
                        return parsed.to_pydatetime()
        return None

    # -- fetch --------------------------------------------------------------

    def fetch(self, start: date, end: date, **kwargs: object) -> FetchResult:
        if not self.workbook.exists():
            raise SourceUnavailable(
                f"BNEF LDZ workbook not found at {self.workbook}. This loader reads a "
                "real Terminal export; it does not generate demand data."
            )
        vintage = self.now()
        raw = pd.read_excel(self.workbook, sheet_name=SHEET_DAILY, skiprows=HEADER_SKIPROWS)
        if len(raw.columns) != len(COLUMNS):
            raise SourceUnavailable(
                f"BNEF workbook has {len(raw.columns)} columns, expected {len(COLUMNS)}: "
                f"{list(raw.columns)}. The export layout changed - remap it deliberately."
            )
        raw.columns = COLUMNS
        self.cache_raw(raw.astype(str), f"ldz_{start}_{end}", vintage)
        return self._normalise(raw, start, end, vintage)

    def _normalise(
        self, raw: pd.DataFrame, start: date, end: date, vintage: datetime
    ) -> FetchResult:
        rows_raw = len(raw)
        drop_reasons: dict[str, int] = {}
        mapping = self.region_map()

        frame = raw.copy()
        frame["obs_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date

        unknown = sorted(set(frame["region"].dropna()) - set(mapping))
        if unknown:
            raise SourceUnavailable(
                f"BNEF workbook contains regions not mapped in demand_sources.yaml: "
                f"{unknown}. Map each to a country or to null explicitly - an "
                "unmapped region would silently drop out of the balance."
            )

        frame["country"] = frame["region"].map(mapping)
        outside = frame["country"].isna()
        drop_reasons["region outside the perimeter"] = int(outside.sum())
        frame = frame[~outside]

        in_window = (frame["obs_date"] >= start) & (frame["obs_date"] <= end)
        drop_reasons["outside requested date range"] = int((~in_window).sum())
        frame = frame[in_window]

        records: list[pd.DataFrame] = []
        source_updated = self.dataset_date()
        for column, (series_id, is_estimated) in VALUE_COLUMNS.items():
            value = pd.to_numeric(frame[column], errors="coerce")
            keep = value.notna() & frame["obs_date"].notna()
            missing = int((~keep).sum())
            if missing:
                drop_reasons[f"{series_id}: missing or non-numeric"] = missing
            if is_estimated:
                log.warning(
                    "%s is a modelled series, not an observation - loaded with "
                    "is_estimated=True and must never be used as a fitting target",
                    series_id,
                )
            records.append(
                pd.DataFrame(
                    {
                        "obs_date": frame.loc[keep, "obs_date"],
                        "country": frame.loc[keep, "country"],
                        "point_key": None,
                        "operator_key": None,
                        "direction_key": None,
                        "series_id": series_id,
                        "value": value[keep],
                        "unit": "mcm/d",
                        "source": self.source,
                        "origin": None,
                        "vintage": vintage,
                        "source_updated_at": source_updated,
                        "is_estimated": is_estimated,
                        "run_id": "",
                    }
                )
            )

        out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        log.info(
            "BNEF LDZ: %d rows for %d countries, %s..%s",
            len(out),
            out["country"].nunique() if not out.empty else 0,
            start,
            end,
        )
        return FetchResult(
            frame=out,
            source=self.source,
            vintage=vintage,
            rows_raw=rows_raw,
            rows_dropped=sum(drop_reasons.values()),
            drop_reasons={k: v for k, v in drop_reasons.items() if v},
        )
