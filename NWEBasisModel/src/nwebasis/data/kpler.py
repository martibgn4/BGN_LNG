"""Kpler: LNG on water. The largest missing feature in this model.

Why it matters more than the other gaps. Every other driver here tells you what
the market is worth; on-water volume tells you how much cargo has to find a
home and by when. In a long market the NWE basis is set by the marginal cargo
that cannot be placed, and that cargo is visible on the water two to three
weeks before it prices. Without it, the model reacts to length instead of
anticipating it.

No synthetic data. ``KplerApiFetcher`` raises without a key, and its
normalisation is deliberately unimplemented until a real payload has been seen,
because column layouts differ by product and guessing them would put unverified
numbers into a model that informs positions. ``KplerCsvFetcher`` reads a real
export from disk and is the supported path today.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nwebasis.config import get
from nwebasis.data.base import Fetcher, FetchResult, SourceUnavailable

log = logging.getLogger(__name__)

REQUIRED_CSV_COLUMNS = frozenset({"date", "series", "value", "unit"})


class KplerApiFetcher(Fetcher):
    source = "kpler_api"
    credential_env_var = "KPLER_API_KEY"

    def fetch(self, **kwargs: object) -> FetchResult:
        self.credential()  # raises CredentialsMissing when absent
        raise NotImplementedError(
            "Kpler API normalisation is unimplemented until a real payload has "
            "been captured. Export the series listed under kpler.required_series "
            "in config/sources.yaml and load them with KplerCsvFetcher, or share "
            "one response body and this will be wired to it."
        )


class KplerCsvFetcher(Fetcher):
    """Loads a Kpler export. Clearly marked, reads real data only.

    Expected long format, one row per (date, series):

        date,series,value,unit
        2026-09-01,lng_on_water_atlantic,412.5,mcm

    The accepted series names are exactly the keys under
    ``kpler.required_series`` in config/sources.yaml. An unrecognised series
    name is an error, not a passthrough: a typo that silently creates a new
    feature column is how a model ends up fitted to nothing.
    """

    source = "kpler_csv"

    def fetch(self, path: str | Path | None = None) -> FetchResult:
        if path is None:
            raise SourceUnavailable("KplerCsvFetcher needs an export path")
        path = Path(path)
        if not path.is_file():
            raise SourceUnavailable(f"Kpler export not found at {path}")
        frame = pd.read_csv(path)
        missing = REQUIRED_CSV_COLUMNS - set(frame.columns)
        if missing:
            raise SourceUnavailable(
                f"{path} is missing required columns {sorted(missing)}; "
                f"expected {sorted(REQUIRED_CSV_COLUMNS)}"
            )
        known = set(get("sources", "kpler", "required_series"))
        unknown = sorted(set(frame["series"]) - known)
        if unknown:
            raise SourceUnavailable(
                f"{path} carries unrecognised series {unknown}. Add them to "
                "kpler.required_series in config/sources.yaml first, so the "
                "feature they feed is declared rather than inferred."
            )
        frame["date"] = pd.to_datetime(frame["date"])
        return self._result(frame, self.source)
