"""Fetcher contract.

Two rules, both carried over from GasImbalanceModel because they were learned
the hard way there:

  1. A fetcher with no credentials raises `CredentialsMissing`. It does not
     return an empty frame, a zero, or a plausible number.
  2. A source that answers with nothing raises `SourceUnavailable`. The Spark
     netbacks endpoint in particular returns HTTP 200 with an empty data list
     when a required query parameter is absent, which is exactly the failure
     mode that ends up in a model as a silent hole.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

log = logging.getLogger(__name__)


class CredentialsMissing(RuntimeError):
    """No credential for a source that needs one."""


class SourceUnavailable(RuntimeError):
    """The source answered, but not with usable data."""


@dataclass(frozen=True)
class FetchResult:
    frame: pd.DataFrame
    source: str
    vintage: datetime
    rows: int


class Fetcher(ABC):
    source: str = "unset"
    credential_env_var: str | None = None

    def credential(self) -> str:
        if self.credential_env_var is None:
            raise NotImplementedError(f"{self.source} declares no credential variable")
        value = os.environ.get(self.credential_env_var)
        if not value:
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
