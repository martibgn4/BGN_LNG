"""Fetcher contract.

Three rules, the first two carried over from the sibling projects because they
were learned the hard way there, the third specific to forecast data:

  1. A fetcher with no credentials raises `CredentialsMissing`. It does not
     return an empty frame, a zero, or a plausible number.
  2. A source that answers with nothing raises `SourceUnavailable`.
  3. A provider that cannot serve a PAST issue date raises rather than serving
     the current run. Silently returning today's forecast stamped with last
     January's issue date is undetectable lookahead, and it would make a
     backtest look extraordinary.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

log = logging.getLogger(__name__)

# The canonical weather panel. Every weather fetcher returns exactly these
# columns so that revisions, normals and features never have to care which
# provider or model a row came from.
WEATHER_COLUMNS = (
    "issue_date",    # when the forecast was produced (NaT for reanalysis)
    "target_date",   # the day being forecast
    "region",        # key into regions.yaml
    "point",         # named point within the region
    "variable",      # Open-Meteo variable name
    "value",
    "model",         # e.g. ecmwf_ifs025, or "era5" for reanalysis
    "member",        # ensemble member index, or -1 for deterministic
)


class CredentialsMissing(RuntimeError):
    """No credential for a source that needs one."""


class SourceUnavailable(RuntimeError):
    """The source answered, but not with usable data."""


class HistoricalIssueUnsupported(SourceUnavailable):
    """The provider serves only the current run and a past issue was asked for.

    Its own exception type because it is not a transient failure and must never
    be retried or swallowed: falling back to the current run here is the single
    most damaging bug available in this project.
    """


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    source: str
    vintage: datetime
    rows: int


class Fetcher(ABC):
    source: str = "unset"
    credential_env_var: str | None = None
    credential_required: bool = True

    def credential(self) -> str | None:
        """The API key, or None where the source has an unauthenticated tier."""
        if self.credential_env_var is None:
            raise NotImplementedError(f"{self.source} declares no credential variable")
        value = os.environ.get(self.credential_env_var)
        if not value:
            if not self.credential_required:
                return None
            raise CredentialsMissing(
                f"{self.source} needs {self.credential_env_var}. Set it and re-run. "
                "No substitute data is produced."
            )
        return value

    @abstractmethod
    def fetch(self, **kwargs: object) -> FetchResult:
        ...

    @staticmethod
    def _result(frame: pd.DataFrame, source: str) -> FetchResult:
        if frame.empty:
            raise SourceUnavailable(
                f"{source} returned no rows. Treating this as a source failure "
                "rather than an empty dataset."
            )
        return FetchResult(frame=frame, source=source, vintage=datetime.now(UTC),
                           rows=len(frame))


class WeatherFetcher(Fetcher):
    """A weather source. Returns the canonical panel of WEATHER_COLUMNS."""

    supports_historical_issue: bool = False

    @staticmethod
    def _validate_panel(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        missing = set(WEATHER_COLUMNS) - set(frame.columns)
        if missing:
            raise SourceUnavailable(
                f"{source} produced a panel missing {sorted(missing)}. Every "
                "weather fetcher must return the canonical columns so that "
                "downstream code never branches on provider."
            )
        return frame.loc[:, list(WEATHER_COLUMNS)]
