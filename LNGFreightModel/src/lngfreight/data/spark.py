"""Spark: the freight assessments that are the target, and the FFA curve.

Spark30S (Atlantic, USG->Continent) and Spark25S (Pacific) are assessed in
USD per charter day for a 174k cbm 2-stroke vessel. They are the desk's
working freight numbers and the settlement reference for the spark30ffa
curve, so a forecast of them is directly comparable to a hedgeable price.

Reuses ``Utils.utils_spark``, which already holds this desk's OAuth flow and
payload parsing and runs in production. Re-implementing it here would create
a second thing to keep correct.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from lngfreight.config import get
from lngfreight.data.base import (CredentialsMissing, Fetcher, FetchResult,
                                  SourceUnavailable)

log = logging.getLogger(__name__)

# utils_spark lives at the repo root, one level above this project.
_REPO_ROOT = Path(__file__).resolve().parents[4]

def _max_releases() -> int:
    """Release-count ceiling for a pull. See sources.yaml for why it exists."""
    return int(get("sources", "spark", "max_releases"))


def _utils():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        from Utils import utils_spark
    except ImportError as exc:
        raise SourceUnavailable(
            f"Utils.utils_spark is not importable from {_REPO_ROOT}. "
            "It carries this desk's Spark OAuth flow; no substitute is used."
        ) from exc
    return utils_spark


def _token():
    utils = _utils()
    path = get("sources", "spark", "credentials_path")
    if not Path(path).exists():
        raise CredentialsMissing(
            f"Spark credentials not found at {path}. Set the path in "
            "config/sources.yaml. No substitute prices are produced."
        )
    client_id, client_secret = utils.retrieve_credentials(file_path=path)
    return utils.get_access_token(client_id, client_secret)


class FreightSpotFetcher(Fetcher):
    """Daily spot assessment for one route, as (obs_date, series_id, value).

    The returned frame is exactly what Spark released, on the dates it
    released. No reindexing and no forward fill happen here: the gaps are
    real information about the assessment cadence and the feature builder
    needs to see them to mark staleness.
    """

    source = "spark_freight_spot"

    def __init__(self, route: str) -> None:
        self.route = route          # "atlantic_spot" | "pacific_spot"

    def fetch(self, start: date | None = None, end: date | None = None) -> FetchResult:
        utils = _utils()
        spec = get("sources", "spark", "freight", self.route)
        frame = utils.fetch_freight_prices(
            _token(), spec["id"], _max_releases(), my_vessel=spec["vessel"],
        )
        if frame.empty:
            raise SourceUnavailable(f"Spark returned no releases for {spec['id']}")
        out = (frame[["Release Date", "USDperday"]]
               .rename(columns={"Release Date": "obs_date", "USDperday": "value"})
               .dropna(subset=["value"]))
        out["obs_date"] = pd.to_datetime(out["obs_date"])
        out = out.drop_duplicates(subset="obs_date", keep="last")
        out["series_id"] = self.route.replace("_spot", "") + "_spot"
        out["publication_lag_days"] = int(spec["publication_lag_days"])
        if start is not None:
            out = out[out["obs_date"] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out["obs_date"] <= pd.Timestamp(end)]
        return self._result(out.sort_values("obs_date").reset_index(drop=True),
                            f"{self.source}:{spec['id']}")


class FfaCurveFetcher(Fetcher):
    """Monthly FFA curve by release date and tenor label (M+0, M+1, ...).

    Kept in tenor space rather than delivery-month space on purpose. At a
    1-15 business day horizon the question is "what does the market think
    the next month costs", which is a constant-maturity question; pinning to
    a delivery month would make the feature jump every month roll.
    """

    source = "spark_ffa"

    def __init__(self, curve: str = "atlantic_monthly") -> None:
        self.curve = curve

    def fetch(self, start: date | None = None, end: date | None = None) -> FetchResult:
        utils = _utils()
        spec = get("sources", "spark", "ffa", self.curve)
        frame = utils.fetch_ffa_prices(_token(), spec["id"], _max_releases())
        if frame.empty:
            raise SourceUnavailable(f"Spark returned no FFA releases for {spec['id']}")
        keep = ["Release Date", "Period Name", "Spark"]
        missing = [c for c in keep if c not in frame.columns]
        if missing:
            raise SourceUnavailable(
                f"Spark FFA payload is missing {missing}; its shape changed and "
                "the parser must be updated rather than guessed around."
            )
        out = (frame[keep]
               .rename(columns={"Release Date": "obs_date",
                                "Period Name": "tenor", "Spark": "value"})
               .dropna(subset=["value"]))
        out["obs_date"] = pd.to_datetime(out["obs_date"])
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["value"])
        wanted = set(get("features", "families", "curve", "ffa_tenors"))
        out = out[out["tenor"].isin(wanted)]
        if out.empty:
            raise SourceUnavailable(
                f"none of the configured tenors {sorted(wanted)} are in the "
                f"{spec['id']} payload"
            )
        out["series_id"] = "ffa_" + out["tenor"].str.replace("+", "", regex=False).str.lower()
        out["publication_lag_days"] = int(spec["publication_lag_days"])
        out = out.drop_duplicates(subset=["obs_date", "series_id"], keep="last")
        if start is not None:
            out = out[out["obs_date"] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out["obs_date"] <= pd.Timestamp(end)]
        return self._result(
            out[["obs_date", "series_id", "value", "publication_lag_days"]]
            .sort_values(["series_id", "obs_date"]).reset_index(drop=True),
            f"{self.source}:{spec['id']}")
