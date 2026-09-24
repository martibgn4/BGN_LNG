"""Fetcher contract.

Two rules, carried across from the sibling models because both were learned
the hard way:

  1. A fetcher with no credentials raises ``CredentialsMissing``. It does not
     return an empty frame, a zero, or a plausible number.
  2. A source that answers with nothing raises ``SourceUnavailable``. Several
     endpoints in this stack answer HTTP 200 with an empty payload when a
     parameter is missing, which is precisely the failure that ends up in a
     model as a silent hole rather than an error.
"""

from __future__ import annotations

import logging
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
        return FetchResult(frame=frame, source=source,
                           vintage=datetime.now(UTC), rows=len(frame))
